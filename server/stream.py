# Stream video to browser from cv2 frames
# Check IP address of the PC generating frames and update in 'host'
# Open in browser using http://{self.host}:8080/viewer.html
#
# USAGE
# frame_bus = FrameBus()
# streaming = StreamingServer(
#     frame_bus=frame_bus,
#     config=config,
#     viewer_html=VIEWER_HTML,
#     host="192.168.68.61",
#     port=8080,
# )
# runner = await streaming.start()
#
# while True:
#     frame_bus.publish(frame)

import asyncio
import threading
from pathlib import Path
import cv2
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
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

class StreamingServer:
    def __init__(self, frame_bus, W, H, viewer_html, host="0.0.0.0", port=8080):
        self.frame_bus = frame_bus
        self.W = W
        self.H = H
        self.viewer_html = Path(viewer_html)
        self.host = host
        self.port = port
        self.pcs = set()

    async def viewer(self, request):
        return web.FileResponse(self.viewer_html)

    async def offer(self, request):
        params = await request.json()

        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print("Connection state:", pc.connectionState)
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await pc.close()
                self.pcs.discard(pc)

        track = CvVideoTrack(
            self.frame_bus,
            fallback_size=(self.H, self.W),
        )
        pc.addTrack(track)

        offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        await pc.setRemoteDescription(offer_desc)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        })

    async def on_shutdown(self, app):
        coros = [pc.close() for pc in self.pcs]
        if coros:
            await asyncio.gather(*coros)
        self.pcs.clear()

    async def start(self):
        app = web.Application()
        app.router.add_get("/", self.viewer)
        app.router.add_get("/viewer.html", self.viewer)
        app.router.add_post("/offer", self.offer)
        app.on_shutdown.append(self.on_shutdown)

        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        print(f"Open viewer at http://{self.host}:{self.port}/")
        return runner