import io
import os
import cv2
import glob
import random
import numpy as np
from PIL import Image


class File:
    def __init__(self, image_w: int, image_h: int, folder: str = ""):
        self.image_w = image_w
        self.image_h = image_h
        self.paths = self._init_image_list(folder) if folder else []

    # ── Folder helpers ────────────────────────────────────────────────────────

    def _init_image_list(self, folder, extensions=("*.png", "*.jpg", "*.jpeg")):
        paths = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(folder, ext)))
        if not paths:
            print("No images found in folder!")
        paths.sort()
        return paths

    def load_image(self, id: int | None = None) -> np.ndarray:
        """Return a cv2 BGR image from the folder, resized to (image_w, image_h)."""
        path = self.paths[id] if id is not None else random.choice(self.paths)
        img = cv2.imread(path)
        return self.resize_to_fit(img)

    # ── Resize helpers ────────────────────────────────────────────────────────

    def _cover_crop(self, img: np.ndarray, w: int, h: int) -> np.ndarray:
        """Scale-to-cover then center-crop to exactly w×h (no distortion, no bars)."""
        ih, iw = img.shape[:2]
        scale = max(w / iw, h / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
        resized = cv2.resize(img, (nw, nh), interpolation=interp)
        x0, y0 = (nw - w) // 2, (nh - h) // 2
        canvas = np.zeros((h, w, img.shape[2]), dtype=img.dtype)
        canvas[:h, :w] = resized[y0:y0 + h, x0:x0 + w]
        return canvas

    def resize_to_fit(self, img: np.ndarray) -> np.ndarray:
        """Center-crop-resize to (image_w, image_h)."""
        return self._cover_crop(img, self.image_w, self.image_h)

    def resize_to_fit_window(self, img: np.ndarray, window_w: int, window_h: int) -> np.ndarray:
        """Center-crop-resize to an arbitrary window size."""
        return self._cover_crop(img, window_w, window_h)

    # ── Format converters ─────────────────────────────────────────────────────
    # All image-returning methods accept fit=False.
    # Pass fit=True to center-crop-resize the result to (image_w, image_h).

    def bytes_to_cv2(self, data: bytes, fit: bool = False) -> np.ndarray:
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return self.resize_to_fit(img) if fit else img

    def bytes_to_pil(self, data: bytes, fit: bool = False) -> Image.Image:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if fit:
            w, h = img.size
            scale = max(self.image_w / w, self.image_h / h)
            nw, nh = int(w * scale), int(h * scale)
            img = img.resize((nw, nh), Image.LANCZOS)
            x0, y0 = (nw - self.image_w) // 2, (nh - self.image_h) // 2
            return img.crop((x0, y0, x0 + self.image_w, y0 + self.image_h))
        return img

    def cv2_to_bytes(self, img: np.ndarray, ext: str = ".jpg", quality: int = 95) -> bytes:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
        ok, buf = cv2.imencode(ext, img, params)
        if not ok:
            raise ValueError(f"cv2.imencode failed for ext={ext!r}")
        return buf.tobytes()

    def cv2_to_pil(self, img: np.ndarray, fit: bool = False) -> Image.Image:
        if fit:
            img = self.resize_to_fit(img)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def pil_to_bytes(self, img: Image.Image, fmt: str = "JPEG", quality: int = 95) -> bytes:
        buf = io.BytesIO()
        kwargs = {"quality": quality} if fmt.upper() == "JPEG" else {}
        img.convert("RGB").save(buf, format=fmt, **kwargs)
        return buf.getvalue()

    def pil_to_cv2(self, img: Image.Image, fit: bool = False) -> np.ndarray:
        result = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        return self.resize_to_fit(result) if fit else result

    # ── Frame accumulator ─────────────────────────────────────────────────────

    def accumulate(self, frame: np.ndarray, weight: float = 0.85) -> np.ndarray:
        """Blend frame into a running float buffer; returns the blended uint8 result.

        weight controls how much history is retained (0 = no history, 1 = freeze).
        Call reset_accumulator() to clear the buffer between sequences.
        """
        current = frame.astype(np.float32)
        if not hasattr(self, "_accumulator") or self._accumulator is None or self._accumulator.shape != current.shape:
            self._accumulator = current.copy()
        self._accumulator = self._accumulator * weight + current * (1.0 - weight)
        return self._accumulator.astype(np.uint8)

    def reset_accumulator(self):
        self._accumulator = None
