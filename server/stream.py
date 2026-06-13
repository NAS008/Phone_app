# Stream video to browser from cv2 frames via WebRTC
# frame_bus = FrameBus()
# streaming = StreamingServer(frame_bus=frame_bus, W=1920, H=1080, viewer_html=VIEWER_HTML, host="0.0.0.0", port=8080)
# runner = await streaming.start()
# while True:
#     frame_bus.publish(frame)

import asyncio
import threading
from pathlib import Path
import cv2
import numpy as np
from aiohttp import web
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCRtpSender,
    RTCSessionDescription,
    VideoStreamTrack,
)
import aiortc.codecs.vpx as _vpx
from av import VideoFrame

# aiortc 1.14 hard-caps VP8 at 1.5 Mbps; raise it so target_bitrate is honoured.
_vpx.MAX_BITRATE = 80_000_000

STUN_SERVERS = ["stun:stun.l.google.com:19302"]


async def _apply_bitrate(sender, target_bitrate, retries=40, delay=0.05):
    """Set VP8 encoder target_bitrate once the sender's encoder is running."""
    for _ in range(retries):
        enc = sender._RTCRtpSender__encoder  # created lazily after ICE/DTLS
        if enc is not None:
            enc.target_bitrate = target_bitrate
            return
        await asyncio.sleep(delay)

def build_ice_servers(turn_url="", turn_username="", turn_password=""):
    servers = [RTCIceServer(urls=STUN_SERVERS)]
    if turn_url:
        servers.append(
            RTCIceServer(
                urls=[f"{turn_url}?transport=udp", f"{turn_url}?transport=tcp"],
                username=turn_username,
                credential=turn_password,
            )
        )
    return servers

def ice_servers_from_dicts(entries):
    # Browser-style [{"urls": [...], "username"?, "credential"?}] → aiortc
    return [
        RTCIceServer(
            urls=entry.get("urls"),
            username=entry.get("username"),
            credential=entry.get("credential"),
        )
        for entry in entries or []
    ]

class FrameBus:
    def __init__(self, max_side=1920):
        self._lock = threading.Lock()
        self._frame = None
        self._max_side = max_side

    def publish(self, frame):
        if frame is None:
            return
        # Downscale once here so every viewer track encodes a stream-sized frame
        # instead of the full window resolution.
        h, w = frame.shape[:2]
        side = max(h, w)
        if self._max_side > 0 and side > self._max_side:
            scale = self._max_side / side
            # Video encoders require even dimensions (yuv420p)
            new_w = max(2, int(w * scale) // 2 * 2)
            new_h = max(2, int(h * scale) // 2 * 2)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
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
        h, w = fallback_size
        side = max(h, w)
        if bus._max_side > 0 and side > bus._max_side:
            scale = bus._max_side / side
            h = max(2, int(h * scale) // 2 * 2)
            w = max(2, int(w * scale) // 2 * 2)
        self.last = np.zeros((h, w, 3), dtype=np.uint8)

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
    def __init__(self, frame_bus, W, H, viewer_html, host="0.0.0.0", port=8080, max_viewers=8, ice_servers=None, turn_provider=None, target_bitrate=80_000_000):
        self.frame_bus = frame_bus
        self.W = W
        self.H = H
        self.viewer_html = Path(viewer_html)
        self.host = host
        self.port = port
        self.max_viewers = max_viewers
        self.ice_servers = ice_servers or build_ice_servers()
        # Optional callable returning browser-style ice server dicts (e.g.
        # CloudflareTurn.get_ice_servers). Called per connection because the
        # credentials are short-lived; may block, so it runs in a thread.
        self.turn_provider = turn_provider
        # maxBitrate ceiling for the VP9 encoder (aiortc default ~900 kbps causes heavy
        # macroblocking). 80 Mbps = 0.34 bpp at 2560×3840@24fps — excellent quality.
        # WebRTC congestion control will reduce the actual rate on limited networks.
        self.target_bitrate = target_bitrate
        self.pcs = set()

    async def viewer(self, request):
        return web.FileResponse(self.viewer_html)

    async def answer_offer(self, sdp, sdp_type):
        # Shared by the local LAN viewer route and bus-relayed offers from the webapp
        if len(self.pcs) >= self.max_viewers:
            raise RuntimeError(f"viewer limit reached ({self.max_viewers})")

        ice_servers = list(self.ice_servers)
        if self.turn_provider:
            extra = await asyncio.to_thread(self.turn_provider)
            ice_servers += ice_servers_from_dicts(extra)

        pc = RTCPeerConnection(
            RTCConfiguration(iceServers=ice_servers)
        )
        self.pcs.add(pc)

        track = CvVideoTrack(
            self.frame_bus,
            fallback_size=(self.H, self.W),
        )
        sender = pc.addTrack(track)
        # Prefer VP9 — better detail/bitrate than VP8 (larger transforms, superior prediction).
        # Falls back to VP8 automatically if the browser doesn't offer VP9.
        caps = RTCRtpSender.getCapabilities("video")
        if caps:
            vp9  = [c for c in caps.codecs if c.mimeType == "video/VP9"]
            rest = [c for c in caps.codecs if c.mimeType != "video/VP9"]
            pc.getTransceivers()[-1].setCodecPreferences(vp9 + rest)
        target_bitrate = self.target_bitrate

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print("Connection state:", pc.connectionState)
            if pc.connectionState == "connected":
                asyncio.create_task(_apply_bitrate(sender, target_bitrate))
            elif pc.connectionState in ("failed", "closed", "disconnected"):
                await pc.close()
                self.pcs.discard(pc)

        offer_desc = RTCSessionDescription(sdp=sdp, type=sdp_type)
        await pc.setRemoteDescription(offer_desc)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }

    async def offer(self, request):
        params = await request.json()
        try:
            answer = await self.answer_offer(params["sdp"], params["type"])
        except RuntimeError as exc:
            return web.json_response({"error": str(exc)}, status=503)
        return web.json_response(answer)

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