import os
import time
import math
import threading
import cv2
import numpy as np
import sounddevice as sd
import mediapipe as mp

class LowPass:
    def __init__(self, x0=0.0):
        self.y = x0
        self.ready = False

    def apply(self, x, alpha):
        if not self.ready:
            self.y = x
            self.ready = True
        else:
            self.y = alpha * x + (1.0 - alpha) * self.y
        return self.y

class OneEuro:
    def __init__(self, min_cutoff=1.2, beta=0.02, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.xf = LowPass()
        self.dxf = LowPass()
        self.t_prev = None
        self.x_prev = None

    def _smoothing_alpha(self, dt, cutoff):
        r = 2.0 * math.pi * cutoff * dt
        return r / (r + 1.0)

    def apply(self, x, t):
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            return self.xf.apply(x, 1.0)

        dt = max(1e-6, t - self.t_prev)
        dx = (x - self.x_prev) / dt
        ad = self._smoothing_alpha(dt, self.d_cutoff)
        dx_hat = self.dxf.apply(dx, ad)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_alpha(dt, cutoff)
        x_hat = self.xf.apply(x, a)

        self.t_prev = t
        self.x_prev = x
        return x_hat

def _compute_vel(pos_new, pos_old, dt, z_gain=0.5):
    vx = (pos_new[0] - pos_old[0]) / dt
    vy = (pos_new[1] - pos_old[1]) / dt
    vz = z_gain * math.sqrt(vx * vx + vy * vy)
    return np.array([vx, vy, vz], dtype=np.float32)

def _lowpass_vel(prev_vel, raw_vel, alpha=0.2):
    return alpha * raw_vel + (1.0 - alpha) * prev_vel

def _clamp_vel(vel, max_xy=3.0, max_z=1.0):
    xy = vel[:2]
    speed = np.linalg.norm(xy)
    if speed > max_xy:
        xy = xy * (max_xy / speed)
        vel = vel.copy()
        vel[0], vel[1] = xy[0], xy[1]
    vel[2] = min(vel[2], max_z)
    return vel

class Mouse:
    def __init__(self, dt, width, height):
        self.dt = dt
        self.W = width
        self.H = height
        self.on = False
        self.pos = np.array([0.5, 0.5, 0.0])
        self.vel = np.array([0.0, 0.0, 0.0])

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            old_pos = self.pos.copy()
            self.pos[0] = x / self.W
            self.pos[1] = 1.0 - (y / self.H)
            self.pos[2] = 0.0
            raw_vel = _compute_vel(self.pos, old_pos, self.dt, z_gain=0.5)
            raw_vel = _clamp_vel(raw_vel, max_xy=2.5, max_z=0.5)
            self.vel = _lowpass_vel(self.vel, raw_vel, alpha=0.2)

        elif event == cv2.EVENT_LBUTTONDOWN:
            self.on = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.on = False
            # Decay velocity when button released, not on every misc event
            self.vel *= 0.9

    def update(self, now: float) -> bool:
        """Uniform interface. Returns True if active."""
        return self.on

class Camera:
    WRIST = 16

    LANDMARK_NAMES = {
        0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
        4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
        7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
        11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow", 14: "right_elbow",
        15: "left_wrist", 16: "right_wrist", 17: "left_pinky", 18: "right_pinky",
        19: "left_index", 20: "right_index", 21: "left_thumb", 22: "right_thumb",
        23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
        27: "left_ankle", 28: "right_ankle", 29: "left_heel", 30: "right_heel",
        31: "left_foot_index", 32: "right_foot_index",
    }

    CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (24, 26), (26, 28),
    ]

    KEY_LANDMARKS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    def __init__(self, model_path, dt, width=640, height=480,
                 camera_candidates=None,
                 num_poses=1,
                 mirror=True,
                 show_labels=True,
                 visibility_threshold=0.4):
        self.model_path = model_path
        self.dt = dt
        self.W = width
        self.H = height
        self.mirror = mirror
        self.show_labels = show_labels
        self.visibility_threshold = visibility_threshold
        self.num_poses = num_poses
        self.camera_candidates = camera_candidates or [
            (1, cv2.CAP_DSHOW), (0, cv2.CAP_DSHOW),
            (1, cv2.CAP_MSMF),  (0, cv2.CAP_MSMF),
            (1, cv2.CAP_ANY),   (0, cv2.CAP_ANY),
            (2, cv2.CAP_DSHOW), (2, cv2.CAP_ANY),
        ]

        # Shared interface
        self.on = False
        self.pos = np.array([0.5, 0.5, 0.0])
        self.vel = np.array([0.0, 0.0, 0.0])

        # One-Euro filters per axis
        self.fx = OneEuro(min_cutoff=1.5, beta=0.03, d_cutoff=1.0)
        self.fy = OneEuro(min_cutoff=1.5, beta=0.03, d_cutoff=1.0)
        self.fz = OneEuro(min_cutoff=1.0, beta=0.02, d_cutoff=1.0)

        self.mp = mp
        self.cap = None
        self.landmarker = None
        self.latest_result = None
        self.latest_timestamp = -1
        self.start_time = None
        self.last_sent_ts = -1

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f'Model not found: {self.model_path}')

    def _result_callback(self, result, output_image, timestamp_ms):
        self.latest_result = result
        self.latest_timestamp = timestamp_ms

    def _create_landmarker(self):
        BaseOptions = self.mp.tasks.BaseOptions
        PoseLandmarker = self.mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = self.mp.tasks.vision.PoseLandmarkerOptions
        RunningMode = self.mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.LIVE_STREAM,
            num_poses=self.num_poses,
            result_callback=self._result_callback,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def _open_camera(self):
        for index, backend in self.camera_candidates:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                self.cap = cap
                print(f'Opened camera index={index}, backend={backend}, shape={frame.shape}')
                self.on = True
                return
            cap.release()
        raise RuntimeError('Could not open any camera.')

    def _valid_landmark(self, lm, margin=0.15):
        if hasattr(lm, 'visibility') and lm.visibility < self.visibility_threshold:
            return False
        return (-margin <= lm.x <= 1 + margin) and (-margin <= lm.y <= 1 + margin)

    def _get_timestamp_ms(self):
        ts = int((time.perf_counter() - self.start_time) * 1000)
        if ts <= self.last_sent_ts:
            ts = self.last_sent_ts + 1
        self.last_sent_ts = ts
        return ts

    def _failure_canvas(self):
        canvas = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        cv2.putText(canvas, 'No camera frame', (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return canvas

    def start(self):
        self._create_landmarker()
        self._open_camera()
        self.start_time = time.perf_counter()

    def read(self):
        if self.cap is None or self.landmarker is None:
            raise RuntimeError('Camera.start() must be called before read().')
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, self._failure_canvas(), None, None
        if self.mirror:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        self.landmarker.detect_async(mp_image, self._get_timestamp_ms())
        canvas = self.draw(frame, self.latest_result)
        return True, canvas, frame, self.latest_result

    def update(self, now: float) -> bool:
        ok, canvas, frame, result = self.read()
        if not ok:
            self.on = False
            return False

        result = result or self.latest_result
        if result is None or not result.pose_landmarks:
            self.on = False
            return False

        lm = result.pose_landmarks[0][self.WRIST]
        if not self._valid_landmark(lm):
            self.on = False
            return False

        raw_x = float(np.clip(lm.x, 0.0, 1.0))
        raw_y = float(np.clip(1.0 - lm.y, 0.0, 1.0))  # flip Y to match screen convention
        raw_z = float(np.clip(lm.z, -1.0, 1.0))

        old_pos = self.pos.copy()
        self.pos[0] = self.fx.apply(raw_x, now)
        self.pos[1] = self.fy.apply(raw_y, now)
        self.pos[2] = self.fz.apply(raw_z * 0.5 + 0.5, now)  # map [-1,1] → [0,1]
        self.vel = _compute_vel(self.pos, old_pos, self.dt)
        self.on = True
        return True

    def draw(self, frame, result=None):
        h, w = frame.shape[:2]
        canvas = np.zeros_like(frame)
        result = self.latest_result if result is None else result

        if result is None or not result.pose_landmarks:
            cv2.putText(canvas, 'Camera OK - no pose detected yet', (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
            return canvas

        landmarks = result.pose_landmarks[0]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        gray, white, dim, red = (192, 192, 192), (255, 255, 255), (170, 170, 170), (0, 0, 255)

        for i, j in self.CONNECTIONS:
            if self._valid_landmark(landmarks[i]) and self._valid_landmark(landmarks[j]):
                cv2.line(canvas, pts[i], pts[j], gray, 2, cv2.LINE_AA)

        if all(self._valid_landmark(landmarks[k]) for k in (0, 11, 12)):
            shoulder_mid = ((pts[11][0] + pts[12][0]) // 2, (pts[11][1] + pts[12][1]) // 2)
            cv2.line(canvas, pts[0], shoulder_mid, gray, 2, cv2.LINE_AA)

        for idx in self.KEY_LANDMARKS:
            lm = landmarks[idx]
            if not self._valid_landmark(lm):
                continue
            x, y = np.clip(pts[idx][0], 0, w - 1), np.clip(pts[idx][1], 0, h - 1)
            cv2.circle(canvas, (x, y), 4, gray, -1, cv2.LINE_AA)
            if self.show_labels:
                tx, ty = min(x + 10, w - 220), max(y - 10, 20)
                cv2.putText(canvas, self.LANDMARK_NAMES[idx], (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, white, 1, cv2.LINE_AA)
                cv2.putText(canvas, f'({lm.x:.2f},{lm.y:.2f},{lm.z:.2f})', (tx, ty + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, dim, 1, cv2.LINE_AA)

        for i, j in [(0, 16), (16, 28), (28, 27), (27, 15), (15, 0)]:
            if self._valid_landmark(landmarks[i]) and self._valid_landmark(landmarks[j]):
                cv2.line(canvas, pts[i], pts[j], red, 3, cv2.LINE_AA)

        cv2.putText(canvas, 'Pose detected', (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        return canvas

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

class Mic:
    FREQ_MIN = 60.0
    FREQ_MAX = 6000.0

    def __init__(
        self,
        sample_rate=16000,
        block_size=1024,
        channels=1,
        gain=5.0,
        decay=0.72,
        num_bands=16,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.channels = channels
        self.gain = gain
        self.decay = decay
        self.num_bands = num_bands

        self.on = False
        self.stream = None
        self._lock = threading.Lock()

        self.bands = np.zeros(self.num_bands, dtype=np.float32)
        self._peak = np.ones(self.num_bands, dtype=np.float32) * 1e-3
        self._prev_spec = np.zeros(self.block_size // 2 + 1, dtype=np.float32)
        self.flux = 0.0

        self._window = np.hanning(self.block_size).astype(np.float32)
        self._fft_freqs = np.fft.rfftfreq(self.block_size, d=1.0 / self.sample_rate).astype(np.float32)
        self._band_edges = np.geomspace(self.FREQ_MIN, self.FREQ_MAX, self.num_bands + 1).astype(np.float32)

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[Mic] audio status: {status}")

        mono = indata[:, 0].astype(np.float32)
        if mono.shape[0] != self.block_size:
            return

        x = mono * self._window
        spec = np.abs(np.fft.rfft(x)).astype(np.float32)

        delta = spec - self._prev_spec
        flux = float(np.mean(np.maximum(delta, 0.0)))
        self._prev_spec = spec

        flux = np.log1p(10.0 * flux)

        band_vals = np.zeros(self.num_bands, dtype=np.float32)
        for bi in range(self.num_bands):
            f0 = self._band_edges[bi]
            f1 = self._band_edges[bi + 1]
            mask = (self._fft_freqs >= f0) & (self._fft_freqs < f1)
            if np.any(mask):
                band_vals[bi] = float(np.mean(spec[mask]))

        band_vals = np.log1p(self.gain * band_vals)

        with self._lock:
            self._peak = np.maximum(self._peak * 0.995, band_vals)
            norm = band_vals / (self._peak + 1e-6)
            norm = np.clip(norm, 0.0, 1.0)

            self.bands = self.decay * self.bands + (1.0 - self.decay) * norm
            self.flux = 0.82 * self.flux + 0.18 * min(flux, 1.0)

    def start(self):
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            blocksize=self.block_size,
            callback=self._callback,
        )
        self.stream.start()
        self.on = True
        print(f"[Mic] started — {self.sample_rate} Hz, block {self.block_size}, bands {self.num_bands}")

    def update(self):
        if not self.on:
            return None, 0.0
        with self._lock:
            return self.bands.copy(), float(self.flux)

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.on = False