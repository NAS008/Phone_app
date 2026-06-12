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
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from av import VideoFrame

STUN_SERVERS = ["stun:stun.l.google.com:19302"]

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
    def __init__(self, max_side=1280):
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

_TV_HTML = """\
<!doctype html><html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stream</title>
<style>
  html,body{margin:0;height:100%;background:#000;overflow:hidden;cursor:none}
  #v,#m{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
  #s{position:fixed;top:10px;left:10px;color:#0f0;font:13px monospace;
     background:rgba(0,0,0,.5);padding:4px 8px;pointer-events:none}
</style></head><body>
<div id="s">connecting...</div>
<video id="v" autoplay playsinline muted style="display:none"></video>
<img  id="m" style="display:none">
<script>
const s=document.getElementById('s'),v=document.getElementById('v'),m=document.getElementById('m');
const WEBRTC_TIMEOUT=6000;

function goFullscreen(){
  const el=document.documentElement;
  const fn=el.requestFullscreen||el.webkitRequestFullscreen||el.mozRequestFullScreen||el.msRequestFullscreen;
  if(fn)fn.call(el).catch(()=>{});
}
// try on any remote/keyboard key or tap — TV remotes fire keydown
document.addEventListener('keydown',goFullscreen,{once:true});
document.addEventListener('click',goFullscreen,{once:true});
// also try when video actually starts playing (works on some TV browsers without gesture)
v.addEventListener('playing',goFullscreen,{once:true});

function mjpeg(){
  m.src='/mjpeg';m.style.display='';v.style.display='none';s.textContent='mjpeg';
}
async function webrtc(){
  const pc=new RTCPeerConnection();
  pc.addTransceiver('video',{direction:'recvonly'});
  pc.ontrack=e=>{v.srcObject=e.streams[0];v.style.display='';m.style.display='none';s.textContent='webrtc';};
  pc.onconnectionstatechange=()=>{
    if(['failed','disconnected'].includes(pc.connectionState))mjpeg();
  };
  const timer=setTimeout(()=>{pc.close();mjpeg();},WEBRTC_TIMEOUT);
  const offer=await pc.createOffer();
  await pc.setLocalDescription(offer);
  const r=await fetch('/offer',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({sdp:pc.localDescription.sdp,type:pc.localDescription.type})});
  if(!r.ok){clearTimeout(timer);return mjpeg();}
  const ans=await r.json();
  await pc.setRemoteDescription(ans);
  clearTimeout(timer);
}
if(window.RTCPeerConnection)webrtc().catch(mjpeg);else mjpeg();
</script></body></html>
"""

class StreamingServer:
    def __init__(self, frame_bus, W, H, viewer_html, host="0.0.0.0", port=8080, max_viewers=8, ice_servers=None, turn_provider=None):
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
        self.pcs = set()
        self._mjpeg_queues: set[asyncio.Queue] = set()

    async def viewer(self, request):
        return web.FileResponse(self.viewer_html)

    async def tv(self, request):
        return web.Response(text=_TV_HTML, content_type="text/html")

    async def mjpeg(self, request):
        q: asyncio.Queue = asyncio.Queue(maxsize=4)
        self._mjpeg_queues.add(q)
        resp = web.StreamResponse(headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            "Cache-Control": "no-cache",
        })
        await resp.prepare(request)
        try:
            while True:
                jpg: bytes = await q.get()
                await resp.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            self._mjpeg_queues.discard(q)
        return resp

    def publish_mjpeg(self, frame_bgr: np.ndarray, quality: int = 70) -> None:
        if not self._mjpeg_queues:
            return
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return
        jpg = buf.tobytes()
        for q in list(self._mjpeg_queues):
            if not q.full():
                q.put_nowait(jpg)

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
        app.router.add_get("/tv", self.tv)
        app.router.add_get("/mjpeg", self.mjpeg)
        app.router.add_post("/offer", self.offer)
        app.on_shutdown.append(self.on_shutdown)

        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        print(f"Open viewer at http://{self.host}:{self.port}/")
        return runner