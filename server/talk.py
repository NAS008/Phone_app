# Talking-head engine: TTS → WhisperX alignment → audio playback → LivePortrait animation.
#
# Usage:
#   engine = TalkEngine(lp_root=Path("..."), api_key="...")
#   engine.load()
#   engine.speak(enhanced_bgr_image, "Hello world")
#   while True:
#       frame = engine.get_frame()   # BGR ndarray, or None before source ready
#   engine.stop()
#
# lip_sync="whisper"  → quality: phoneme-level timeline via WhisperX (default)
# lip_sync="volume"   → speed:   RMS volume + spectral oh-score only (no Whisper)

import io, queue, threading, time, wave, contextlib, warnings
from math import gcd
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd
import torch
from google import genai
from google.genai import types

try:
    import whisperx
    _HAS_WHISPERX = True
except ImportError:
    _HAS_WHISPERX = False
    print("[Warning] whisperx not installed — pip install whisperx  (falling back to volume lip sync)")

try:
    import nltk
    nltk.download("cmudict", quiet=True)
    _CMU = nltk.corpus.cmudict.dict()
    _HAS_CMU = True
except Exception:
    _HAS_CMU = False
    _CMU = {}

_PHONEME_LIP: dict[str, float] = {
    # Open vowels — widened ~1.5×
    "AA": 0.34, "AH": 0.31, "AO": 0.33, "AW": 0.29, "AY": 0.28, "AE": 0.29,
    # Mid vowels
    "EH": 0.23, "ER": 0.17, "EY": 0.20,
    # Close vowels
    "IH": 0.15, "IY": 0.12, "UH": 0.15, "UW": 0.12,
    # Round back vowels
    "OW": 0.22, "OY": 0.23,
    # Bilabials — lip contact
    "B":  0.00, "M":  0.00, "P":  0.00,
    # Labiodentals
    "F":  0.05, "V":  0.06,
    # Interdentals
    "TH": 0.08, "DH": 0.09,
    # Alveolars
    "T":  0.07, "D":  0.08, "N":  0.07, "L":  0.09,
    "S":  0.07, "Z":  0.07,
    # Postalveolars / affricates
    "SH": 0.10, "ZH": 0.10, "CH": 0.08, "JH": 0.08,
    # Retroflex approximant
    "R":  0.11,
    # Velars
    "K":  0.07, "G":  0.08, "NG": 0.07,
    # Glottals / semivowels
    "HH": 0.10, "W":  0.13, "Y":  0.11,
}

_SAMPLE_RATE       = 24000
_BLOCK_SIZE        = 512
_VOLUME_GAIN       = 18.0
_VOLUME_SMOOTHING  = 0.15
_SILENCE_THRESHOLD = 0.0025
_LIP_MAX_TARGET    = 0.34   # matches widest open-vowel in _PHONEME_LIP
_OH_LIP_BOOST      = 0.15
_BLINK_MIN         = 10.0
_BLINK_MAX         = 22.0
_BLINK_DUR         = 0.12

@contextlib.contextmanager
def _quiet():
    buf = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            yield

_BLINK_VOWEL_THRESHOLD = 0.78
_BLINK_VOWEL_COOLDOWN  = 2.0


# ── Controllers ────────────────────────────────────────────────────────────────

class _BlinkController:
    def __init__(self):
        self._next              = time.time() + np.random.uniform(_BLINK_MIN, _BLINK_MAX)
        self._start             = None
        self._blink_queued      = False
        self._last_vowel_blink  = 0.0

    def update(self, mouth_open: float = 0.0) -> float:
        now = time.time()

        if (mouth_open >= _BLINK_VOWEL_THRESHOLD
                and self._start is None
                and not self._blink_queued
                and now - self._last_vowel_blink >= _BLINK_VOWEL_COOLDOWN):
            self._blink_queued     = True
            self._last_vowel_blink = now

        if self._start is None and not self._blink_queued and now >= self._next:
            self._blink_queued = True
            self._next = now + np.random.uniform(_BLINK_MIN, _BLINK_MAX)

        if self._blink_queued and self._start is None:
            self._start        = now
            self._blink_queued = False

        if self._start is not None:
            t = (now - self._start) / _BLINK_DUR
            if t < 1.0:
                return float(t / 0.4) if t < 0.4 else float(1.0 - (t - 0.4) / 0.6)
            self._start = None

        return 0.0

class _HeadMotionController:
    def __init__(self):
        self._t         = np.random.uniform(0, 10)
        self._yaw_phase = np.random.uniform(0, 2 * np.pi)

    def update(self, dt: float) -> float:
        self._t += dt
        return float(0.006 * np.sin(2 * np.pi * 0.055 * self._t + self._yaw_phase))

class _BrowController:
    def __init__(self):
        self._raise_t   = 1.0
        self._raise_amp = 0.0
        self._prev_vol  = 0.0
        self._idle_t    = np.random.uniform(0, 10)

    def update(self, dt: float, volume: float, oh: float) -> float:
        self._idle_t += dt
        d_vol = volume - self._prev_vol
        if d_vol > 0.020 and volume > _SILENCE_THRESHOLD:
            self._raise_t   = 0.0
            self._raise_amp = 0.24 + oh * 0.18
        self._prev_vol = volume

        raise_val = 0.0
        if self._raise_t < 0.65:
            raise_val = self._raise_amp * np.sin(np.pi * self._raise_t / 0.65)
            self._raise_t += dt

        idle = 0.018 * np.sin(2 * np.pi * 0.22 * self._idle_t)
        return float(raise_val + idle)


# ── WhisperManager ─────────────────────────────────────────────────────────────

class WhisperManager:
    """
    Quality lip sync via WhisperX phoneme alignment.

    Two entry points:
      align(pcm_f32, text, sample_rate) → lip timeline
          Fast path: known text (e.g. TTS output) — skips ASR, runs only
          forced word-level alignment.  Use with TTS-generated audio.

      transcribe(pcm_16k, total_dur) → (text, lip timeline)
          Full path: unknown audio — runs ASR then alignment.
          Use with pre-recorded speech.

    Models are lazy-loaded on first call and reused across calls.
    Returns None (or empty string + None) gracefully when whisperx is absent.
    """

    def __init__(self, fps: int = 25):
        self._fps   = fps
        self._model = None
        self._align = None
        self._meta  = None

    @property
    def available(self) -> bool:
        return _HAS_WHISPERX

    def load(self) -> None:
        """Load WhisperX ASR + alignment models (idempotent)."""
        if self._align is not None or not _HAS_WHISPERX:
            return
        device       = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        print("[WhisperX] Loading models (first run only)…")
        self._model = whisperx.load_model(
            "base", device, compute_type=compute_type, language="en"
        )
        self._align, self._meta = whisperx.load_align_model(
            language_code="en", device=device
        )
        print(f"[WhisperX] Ready on {device} ✓")

    def align(self, pcm_f32: np.ndarray, text: str,
              sample_rate: int = _SAMPLE_RATE) -> "np.ndarray | None":
        """
        Fast path: forced word alignment when the transcript is already known.
        pcm_f32 may be at any sample rate; resampled internally to 16 kHz.
        Returns a float32 lip-ratio timeline at self._fps, or None on failure.
        """
        if not _HAS_WHISPERX:
            return None
        try:
            self.load()
            device    = "cuda" if torch.cuda.is_available() else "cpu"
            audio_16  = self._resample(pcm_f32, sample_rate, 16000)
            total_dur = len(pcm_f32) / sample_rate

            print("[WhisperX] Aligning…")
            t0       = time.time()
            segments = [{"start": 0.0, "end": total_dur, "text": text}]
            aligned  = whisperx.align(
                segments, self._align, self._meta, audio_16, device,
                return_char_alignments=False,
            )
            words = [
                w for seg in aligned.get("segments", [])
                for w in seg.get("words", [])
                if "start" in w and "end" in w
            ]
            print(f"[WhisperX] {len(words)} words aligned in {time.time() - t0:.1f}s")
            if not words:
                return None
            timeline = self._build_lip_timeline(words, total_dur)
            print(f"[WhisperX] {len(timeline)} frames  peak={timeline.max():.3f}")
            return timeline
        except Exception as exc:
            print(f"[WhisperX] Alignment failed ({exc})")
            return None

    def transcribe(self, pcm_16k: np.ndarray,
                   total_dur: float) -> "tuple[str, np.ndarray | None]":
        """
        Full path: ASR transcription + alignment for unknown audio.
        pcm_16k must already be at 16 kHz float32.
        Returns (transcript_text, lip_timeline) — either may be empty/None on failure.
        """
        if not _HAS_WHISPERX:
            return "", None
        try:
            self.load()
            device = "cuda" if torch.cuda.is_available() else "cpu"

            print("[WhisperX] Transcribing…")
            t0     = time.time()
            result = self._model.transcribe(pcm_16k, batch_size=8)
            if not result.get("segments"):
                print("[WhisperX] No speech detected")
                return "", None

            text = " ".join(s["text"].strip() for s in result["segments"])
            print(f"[WhisperX] '{text}'")

            aligned = whisperx.align(
                result["segments"], self._align, self._meta, pcm_16k, device,
                return_char_alignments=False,
            )
            words = [
                w for seg in aligned.get("segments", [])
                for w in seg.get("words", [])
                if "start" in w and "end" in w
            ]
            print(f"[WhisperX] {len(words)} words in {time.time() - t0:.1f}s")
            if not words:
                return text, None
            timeline = self._build_lip_timeline(words, total_dur)
            print(f"[WhisperX] {len(timeline)} frames  peak={timeline.max():.3f}")
            return text, timeline
        except Exception as exc:
            print(f"[WhisperX] Transcription failed ({exc})")
            return "", None

    @staticmethod
    def _resample(pcm: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        if from_sr == to_sr:
            return pcm
        try:
            from scipy.signal import resample_poly
            g = gcd(to_sr, from_sr)
            return resample_poly(pcm, to_sr // g, from_sr // g).astype(np.float32)
        except ImportError:
            n_out = int(len(pcm) * to_sr / from_sr)
            return np.interp(
                np.linspace(0, len(pcm) - 1, n_out), np.arange(len(pcm)), pcm,
            ).astype(np.float32)

    @staticmethod
    def _word_to_phones(word: str) -> list[str]:
        w = word.lower().strip(".,!?;:'\"—")
        if _HAS_CMU and w in _CMU:
            return [p.rstrip("012") for p in _CMU[w][0]]
        return ["AH"] * max(1, len(w) // 3)

    def _build_lip_timeline(self, words: list[dict], total_duration: float) -> np.ndarray:
        n_frames = int(total_duration * self._fps) + 4
        timeline = np.zeros(n_frames, dtype=np.float32)

        for w in words:
            t0 = w.get("start", 0.0)
            t1 = w.get("end",   0.0)
            if t1 <= t0:
                continue
            phones = self._word_to_phones(w.get("word", ""))
            ph_dur = (t1 - t0) / len(phones)
            for i, ph in enumerate(phones):
                f0 = int((t0 + i * ph_dur) * self._fps)
                f1 = int((t0 + (i + 1) * ph_dur) * self._fps) + 1
                ratio = _PHONEME_LIP.get(ph, 0.08)
                timeline[f0 : min(f1, n_frames)] = ratio

        sigma = self._fps * 0.06
        try:
            from scipy.ndimage import gaussian_filter1d
            timeline = gaussian_filter1d(timeline, sigma=sigma).astype(np.float32)
        except ImportError:
            ks = max(3, int(sigma * 6) | 1)
            x  = np.arange(ks) - ks // 2
            k  = np.exp(-x ** 2 / (2 * sigma ** 2)).astype(np.float32)
            k /= k.sum()
            timeline = np.convolve(timeline, k, mode="same").astype(np.float32)

        return timeline


# ── LivePortraitManager ────────────────────────────────────────────────────────

class LivePortraitManager:
    """
    Manages InsightFace face detection + LivePortrait animation pipeline.

    Typical usage:
        lp = LivePortraitManager(lp_root=Path("..."))
        lp.load()
        source = lp.prepare_source(bgr_image)
        frame  = lp.animate_frame(source, mouth_open, blink, oh, head_yaw, brow_raise)
    """

    def __init__(self, lp_root: Path):
        self._lp_root  = Path(lp_root)
        self._pipeline = None
        self._face_app = None

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def load(self) -> None:
        """Load InsightFace and LivePortrait models. Must be called before any other method."""
        import sys
        sys.path.insert(0, str(self._lp_root))
        sys.path.insert(0, str(self._lp_root / "src"))
        self._face_app = self._load_face_detector()
        self._pipeline = self._load_liveportrait()

    def prepare_source(self, src_bgr: np.ndarray, face_index: int = 0) -> dict:
        """
        Detect face, crop to 256×256, extract LivePortrait features.

        face_index: 0 = largest face, 1 = second-largest, etc.
        Returns a source dict consumed by animate_frame().
        """
        if not self.loaded:
            raise RuntimeError("Call load() first.")

        faces = sorted(
            self._face_app.get(src_bgr),
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
        if not faces:
            raise RuntimeError("No face detected in source image.")

        if face_index >= len(faces):
            print(f"[LivePortrait] face_index={face_index} unavailable ({len(faces)} detected), using 0")
            face_index = 0

        if face_index == 0:
            work_img = src_bgr
        else:
            x1, y1, x2, y2 = faces[face_index].bbox.astype(int)
            fw, fh = x2 - x1, y2 - y1
            pad  = max(fw, fh)
            H, W = src_bgr.shape[:2]
            work_img = src_bgr[
                max(0, y1 - pad) : min(H, y2 + pad),
                max(0, x1 - pad) : min(W, x2 + pad),
            ].copy()

        pipeline  = self._pipeline
        wrapper   = pipeline.live_portrait_wrapper
        cropper   = pipeline.cropper
        crop_info = cropper.crop_source_image(work_img, cropper.crop_cfg)
        if crop_info is None:
            raise RuntimeError("LivePortrait could not find a face in the source region.")

        source_lmk      = crop_info["lmk_crop"]
        crop_rgb        = cv2.cvtColor(crop_info["img_crop_256x256"], cv2.COLOR_BGR2RGB)
        source_prepared = wrapper.prepare_source(crop_rgb)

        with torch.no_grad():
            f_s      = wrapper.extract_feature_3d(source_prepared)
            x_s_info = wrapper.get_kp_info(source_prepared)
            x_s      = wrapper.transform_keypoint(x_s_info)

        eye_ratios, lip_ratios = wrapper.calc_ratio([source_lmk])
        lip_rest = float(lip_ratios[0][0][0])
        eye_rest = float(eye_ratios[0].mean())

        M_c2o = crop_info.get("M_c2o")
        if M_c2o is not None:
            M_c2o = np.array(M_c2o, dtype=np.float32)
            if M_c2o.shape == (3, 3):
                M_c2o = M_c2o[:2, :]
            elif M_c2o.shape != (2, 3):
                M_c2o = None

        print(f"[Source] Ready (face[{face_index}])  lip_ratio={lip_rest:.3f}  eye_ratio={eye_rest:.3f}")
        return {
            "f_s": f_s, "x_s": x_s,
            "crop_info": crop_info, "M_c2o": M_c2o, "src_bgr": work_img,
            "source_lmk": source_lmk,
            "source_lip_ratio": lip_rest,
            "source_eye_ratio": eye_rest,
        }

    def animate_frame(self, source: dict, mouth_open: float, blink: float,
                      oh: float, head_yaw: float = 0.0,
                      brow_raise: float = 0.0) -> np.ndarray:
        """
        Produce one animated BGR frame.

        mouth_open : 0–1   (lip parting)
        blink      : 0–1   (1 = fully closed, from _BlinkController)
        oh         : 0–1   (open-vowel score; slightly widens mouth)
        head_yaw   : radians, in-plane micro-rotation
        brow_raise : 0–~0.25, added to eye_target
        """
        pipeline = self._pipeline
        wrapper  = pipeline.live_portrait_wrapper
        f_s, x_s = source["f_s"], source["x_s"]
        lmk      = source["source_lmk"]
        lip_rest = source["source_lip_ratio"]
        eye_rest = source["source_eye_ratio"]

        with torch.no_grad():
            lip_drive    = min(mouth_open + oh * _OH_LIP_BOOST, 1.0)
            lip_target   = lip_rest + lip_drive * (_LIP_MAX_TARGET - lip_rest)
            combined_lip = wrapper.calc_combined_lip_ratio([lip_target], lmk)
            lip_delta    = wrapper.retarget_lip(x_s, combined_lip)

            eye_target   = eye_rest * (1.0 - blink) + brow_raise * 0.30
            combined_eye = wrapper.calc_combined_eye_ratio([[eye_target]], lmk)
            eye_delta    = wrapper.retarget_eye(x_s, combined_eye)

            x_d = x_s + lip_delta + eye_delta

            if abs(head_yaw) > 1e-6:
                cx    = x_s[0, :, 0].mean()
                cy    = x_s[0, :, 1].mean()
                cos_y = float(np.cos(head_yaw))
                sin_y = float(np.sin(head_yaw))
                x_d   = x_d.clone()
                dx    = x_d[0, :, 0] - cx
                dy    = x_d[0, :, 1] - cy
                x_d[0, :, 0] = cx + dx * cos_y - dy * sin_y
                x_d[0, :, 1] = cy + dx * sin_y + dy * cos_y

            out      = wrapper.warp_decode(f_s, x_s, x_d)
            out_imgs = wrapper.parse_output(out["out"])

        out_bgr = cv2.cvtColor(out_imgs[0], cv2.COLOR_RGB2BGR)
        return self._paste_back(out_bgr, source)

    def _paste_back(self, animated_crop: np.ndarray, source: dict) -> np.ndarray:
        M_c2o    = source.get("M_c2o")
        original = source["src_bgr"]
        h, w     = original.shape[:2]
        if M_c2o is None:
            return cv2.resize(animated_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        try:
            if animated_crop.dtype != np.uint8:
                animated_crop = (animated_crop * 255).clip(0, 255).astype(np.uint8)
            warped   = cv2.warpAffine(animated_crop, M_c2o, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
            mask_src = np.ones(animated_crop.shape[:2], dtype=np.float32)
            mask     = cv2.warpAffine(mask_src, M_c2o, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mask     = cv2.GaussianBlur(mask, (31, 31), 15)[:, :, np.newaxis]
            result   = warped.astype(np.float32) * mask + original.astype(np.float32) * (1.0 - mask)
            return result.clip(0, 255).astype(np.uint8)
        except Exception as exc:
            print(f"[paste_back] {exc}")
            return cv2.resize(animated_crop, (w, h), interpolation=cv2.INTER_LINEAR)

    def _load_face_detector(self):
        try:
            import onnxruntime as ort
            ort.set_default_logger_severity(3)
        except Exception:
            pass
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l",
                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        with _quiet():
            app.prepare(ctx_id=0, det_size=(512, 512))
        print("[InsightFace] Loaded ✓")
        return app

    def _load_liveportrait(self):
        from src.config.inference_config import InferenceConfig
        from src.config.crop_config import CropConfig
        from src.live_portrait_pipeline import LivePortraitPipeline
        with _quiet():
            pipeline = LivePortraitPipeline(
                inference_cfg=InferenceConfig(),
                crop_cfg=CropConfig(),
            )
        print("[LivePortrait] Loaded ✓")
        return pipeline


# ── TalkEngine ─────────────────────────────────────────────────────────────────

class TalkEngine:
    """
    Self-contained talking-head engine built on LivePortraitManager + WhisperManager.

    Call load() once to initialise models, then speak(image, text) to start
    an utterance.  Call get_frame() on every display tick to receive the
    current animated BGR frame.

    lip_sync="whisper"  Quality: phoneme-timeline via WhisperX (default).
    lip_sync="volume"   Speed:   RMS volume + spectral oh-score only.
    """

    def __init__(self, lp_root: Path, api_key: str,
                 voice: str = "Orus", fps: int = 25,
                 lip_sync: str = "whisper"):
        self._api_key  = api_key
        self._voice    = voice
        self._fps      = fps
        self._lip_sync = lip_sync

        # ── sub-managers ──────────────────────────────────────────────────────
        self._lp      = LivePortraitManager(lp_root)
        self._whisper = WhisperManager(fps=fps)

        # ── public state (read by display loop / HUD) ─────────────────────────
        self.is_speaking    = False
        self.display_text   = ""
        self.mouth_open     = 0.0
        self.using_timeline = False
        self.oh_score       = 0.0

        # ── internal audio state ──────────────────────────────────────────────
        self._running        = False
        self._volume         = 0.0
        self._lip_timeline   = None
        self._samples_played = 0
        self._audio_queue    = queue.Queue(maxsize=400)

        # ── animation state ───────────────────────────────────────────────────
        self._source     = None
        self._last_frame = None
        self._last_t     = time.time()

        # ── controllers ───────────────────────────────────────────────────────
        self._blink     = _BlinkController()
        self._head_ctrl = _HeadMotionController()
        self._brow_ctrl = _BrowController()

    # ── Public API ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load LivePortrait + InsightFace models and start the audio thread."""
        self._lp.load()
        self._running = True
        threading.Thread(target=self._audio_thread, daemon=True).start()

    def speak(self, image_bgr: np.ndarray, text: str, face_index: int = 0) -> None:
        """
        Prepare the source image (blocking, fast) then start TTS + audio in a
        background thread.  Returns immediately so the display loop can begin.

        face_index: 0 = largest detected face, 1 = second-largest, etc.
        """
        if not self._lp.loaded:
            raise RuntimeError("Call load() before speak().")
        self._source = self._lp.prepare_source(image_bgr, face_index)
        threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()

    def get_frame(self) -> "np.ndarray | None":
        """
        Advance animation controllers and return the current BGR frame.
        Returns None until speak() has been called at least once.
        """
        if self._source is None:
            return None

        now = time.time()
        dt  = now - self._last_t
        self._last_t = now

        vol = self._volume
        oh  = self.oh_score if self.is_speaking else 0.0

        # ── mouth open ────────────────────────────────────────────────────────
        if self.is_speaking and self._lip_timeline is not None:
            idx = int(self._samples_played / _SAMPLE_RATE * self._fps)
            idx = max(0, min(idx, len(self._lip_timeline) - 1))
            raw = float(self._lip_timeline[idx])
            lip_rest        = self._source["source_lip_ratio"]
            self.mouth_open = float(np.clip(
                (raw - lip_rest) / max(_LIP_MAX_TARGET - lip_rest, 1e-4),
                0.0, 1.0,
            ))
            self.using_timeline = True
        elif self.is_speaking:
            self.mouth_open     = self._volume_to_mouth(vol)
            self.using_timeline = False
        else:
            self.mouth_open     = 0.0
            self.using_timeline = False

        blink_val  = self._blink.update(self.mouth_open)
        head_yaw   = self._head_ctrl.update(dt)
        brow_raise = self._brow_ctrl.update(dt, vol if self.is_speaking else 0.0, oh)

        try:
            self._last_frame = self._lp.animate_frame(
                self._source, self.mouth_open, blink_val, oh, head_yaw, brow_raise
            )
        except Exception as exc:
            print(f"[TalkEngine] animate: {exc}")

        return self._last_frame

    def prepare(self, image_bgr: np.ndarray, face_index: int = 0) -> None:
        """Prepare source image without starting speech (shows resting face immediately)."""
        if not self._lp.loaded:
            raise RuntimeError("Call load() before prepare().")
        self._source = self._lp.prepare_source(image_bgr, face_index)

    def speak_audio(self, image_bgr: np.ndarray, pcm_f32: np.ndarray,
                    sample_rate: int = 16000) -> None:
        """
        Lip-sync to pre-recorded audio.  Transcribes with WhisperX then plays
        back. Returns immediately; processing happens in a background thread.
        """
        if not self._lp.loaded:
            raise RuntimeError("Call load() before speak_audio().")
        self._source = self._lp.prepare_source(image_bgr)
        threading.Thread(
            target=self._play_audio_thread,
            args=(pcm_f32, sample_rate),
            daemon=True,
        ).start()

    def stop(self) -> None:
        """Signal audio thread to exit."""
        self._running = False

    # ── Audio playback thread ──────────────────────────────────────────────────

    def _audio_thread(self) -> None:
        ema    = 0.0
        oh_ema = 0.0

        out_stream = sd.OutputStream(samplerate=_SAMPLE_RATE, channels=1, dtype="float32")
        out_stream.start()

        while self._running:
            try:
                block = self._audio_queue.get(timeout=0.05)
            except queue.Empty:
                if not self.is_speaking:
                    self._volume  *= 0.85
                    self.oh_score *= 0.85
                continue

            out_stream.write(block)
            self._samples_played += len(block)

            rms    = float(np.sqrt(np.mean(block ** 2)))
            ema    = _VOLUME_SMOOTHING * ema + (1.0 - _VOLUME_SMOOTHING) * rms
            self._volume = ema

            oh_raw       = self._detect_oh(block)
            oh_ema       = 0.25 * oh_ema + 0.75 * oh_raw
            self.oh_score = float(oh_ema)

        out_stream.stop()
        out_stream.close()

    # ── TTS + alignment thread ─────────────────────────────────────────────────

    def _speak_thread(self, text: str) -> None:
        client = genai.Client(api_key=self._api_key)
        self.display_text = text
        print(f"[Text] {text}\n")

        print("[Gemini] Generating speech...")
        tts_resp = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self._voice
                        )
                    )
                ),
            ),
        )

        inline    = tts_resp.candidates[0].content.parts[0].inline_data
        audio_raw = inline.data

        if audio_raw[:4] == b"RIFF":
            with wave.open(io.BytesIO(audio_raw)) as wf:
                raw_pcm = wf.readframes(wf.getnframes())
        else:
            raw_pcm = audio_raw

        n       = len(raw_pcm) // 2
        pcm_f32 = np.frombuffer(raw_pcm[: n * 2], dtype="<i2").astype(np.float32) / 32768.0

        if self._lip_sync == "whisper":
            self._lip_timeline = self._whisper.align(pcm_f32, text, _SAMPLE_RATE)
        else:
            self._lip_timeline = None   # volume-only mode

        self._samples_played = 0

        print("[Audio] Queuing for playback...")
        self.is_speaking = True
        for i in range(0, len(pcm_f32), _BLOCK_SIZE):
            if not self._running:
                break
            chunk = pcm_f32[i : i + _BLOCK_SIZE]
            while self._running:
                try:
                    self._audio_queue.put(chunk, timeout=0.1)
                    break
                except queue.Full:
                    time.sleep(0.01)

        drain_s = max(0.3, self._audio_queue.qsize() * _BLOCK_SIZE / _SAMPLE_RATE)
        time.sleep(drain_s + 0.5)
        self.is_speaking   = False
        self._lip_timeline = None
        print("[Audio] Done.")

    # ── Recorded-audio playback thread ────────────────────────────────────────

    def _play_audio_thread(self, pcm_f32: np.ndarray, sample_rate: int) -> None:
        audio_16 = (pcm_f32 if sample_rate == 16000
                    else WhisperManager._resample(pcm_f32, sample_rate, 16000))
        pcm_play = (pcm_f32 if sample_rate == _SAMPLE_RATE
                    else WhisperManager._resample(pcm_f32, sample_rate, _SAMPLE_RATE))

        total_dur = len(pcm_play) / _SAMPLE_RATE

        if self._lip_sync == "whisper":
            text, self._lip_timeline = self._whisper.transcribe(audio_16, total_dur)
            if text:
                self.display_text = text
        else:
            self._lip_timeline = None

        self._samples_played = 0

        self.is_speaking = True
        for i in range(0, len(pcm_play), _BLOCK_SIZE):
            if not self._running:
                break
            chunk = pcm_play[i : i + _BLOCK_SIZE]
            while self._running:
                try:
                    self._audio_queue.put(chunk, timeout=0.1)
                    break
                except queue.Full:
                    time.sleep(0.01)

        drain_s = max(0.3, self._audio_queue.qsize() * _BLOCK_SIZE / _SAMPLE_RATE)
        time.sleep(drain_s + 0.5)
        self.is_speaking   = False
        self._lip_timeline = None

    # ── Audio analysis ─────────────────────────────────────────────────────────

    @staticmethod
    def _spectral_centroid(block: np.ndarray) -> float:
        fft   = np.abs(np.fft.rfft(block * np.hanning(len(block))))
        freqs = np.fft.rfftfreq(len(block), 1.0 / _SAMPLE_RATE)
        return float(np.dot(freqs, fft) / (fft.sum() + 1e-9))

    @staticmethod
    def _band_energy(block: np.ndarray, f_low: float, f_high: float) -> float:
        fft   = np.abs(np.fft.rfft(block * np.hanning(len(block)))) ** 2
        freqs = np.fft.rfftfreq(len(block), 1.0 / _SAMPLE_RATE)
        mask  = (freqs >= f_low) & (freqs <= f_high)
        return float(np.sqrt(fft[mask].mean() + 1e-12))

    def _detect_oh(self, block: np.ndarray) -> float:
        if float(np.sqrt(np.mean(block ** 2))) < _SILENCE_THRESHOLD * 0.5:
            return 0.0
        centroid = self._spectral_centroid(block)
        low_e    = self._band_energy(block, 300, 900)
        high_e   = self._band_energy(block, 1500, 4000)
        oh_c = float(np.clip(1.0 - abs(centroid - 1000) / 700, 0.0, 1.0))
        oh_r = float(np.clip(low_e / (high_e + 1e-9) - 0.5, 0.0, 1.5) / 1.5)
        return float(np.clip(oh_c * 0.5 + oh_r * 0.5, 0.0, 1.0))

    @staticmethod
    def _volume_to_mouth(volume: float) -> float:
        if volume < _SILENCE_THRESHOLD:
            return 0.0
        return float(min(1.0, (volume - _SILENCE_THRESHOLD) * _VOLUME_GAIN))
