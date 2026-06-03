import cv2
import numpy as np
import threading
from aiortc import VideoStreamTrack
from av import VideoFrame

class FrameBus:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None

    def publish(self, frame):
        if frame is None:
            return
        frame = np.ascontiguousarray(frame)
        with self._lock:
            self._frame = frame.copy()

    def latest(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

class CvVideoTrack(VideoStreamTrack):
    def __init__(self, bus, fallback_size=(720, 1280)):
        super().__init__()
        self.bus = bus
        self.last = np.zeros((fallback_size[0], fallback_size[1], 3), dtype=np.uint8)

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        img = self.bus.latest()
        if img is not None:
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = np.ascontiguousarray(img)
            self.last = img

        frame = VideoFrame.from_ndarray(self.last, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame
