# # # # # # #
# The       #
# First     #
# NonCarbon #
# Artist    #
# # # # # # #

# Phone App uses the phone sensors to bridge interaction with user and NonCarbon Artist
# Has all the user command messages and settings
# Input: Receives text (or text from audio) and/ or image bytes, receives commands in Admin mode
# Output: Post messages in the bus with instructions
# ----------------------------------------------------------------

import asyncio
import base64
import io
import json
import threading
import time
import uuid
import msgpack
import redis
import urllib.parse
import sys as _sys, os as _os
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from PIL import Image
from google import genai
from google.genai import types
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'shared'))
from config import Config
from bus import Bus
from stream import CloudflareTurn

def _now_ms():
    return int(time.time() * 1000)

def _coerce_bytes(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return base64.b64decode(value)
    raise TypeError(f"Unsupported binary type: {type(value).__name__}")


def _crop_image(data: bytes, iw: int, ih: int) -> bytes:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    scale = max(iw / w, ih / h)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    x0, y0 = (nw - iw) // 2, (nh - ih) // 2
    img = img.crop((x0, y0, x0 + iw, y0 + ih))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()

def _normalize_http_part(part):
    kind = part.get("kind")

    if kind == "text":
        return {"kind": "text", "text": part.get("text")}

    if kind == "image":
        raw = _coerce_bytes(part.get("data"))
        processed = _crop_image(raw, Config.IMAGE_W, Config.IMAGE_H)
        return {
            "kind": "image",
            "mime_type": "image/jpeg",
            "purpose": part.get("purpose", "input"),
            "data": processed,
        }

    raise ValueError(f"Unsupported part kind: {kind}")

def _normalize_http_payload_to_bus_message(payload):
    parts = payload.get("parts")
    if parts is None:
        parts = []
        if payload.get("text") is not None:
            parts.append({"kind": "text", "text": payload.get("text")})

        if payload.get("image_bytes") is not None:
            raw = _coerce_bytes(payload.get("image_bytes"))
            parts.append({
                "kind": "image",
                "mime_type": "image/jpeg",
                "purpose": payload.get("image_purpose", "input"),
                "data": _crop_image(raw, Config.IMAGE_W, Config.IMAGE_H),
            })
    else:
        parts = [_normalize_http_part(part) for part in parts]

    return {
        "v": payload.get("v", 1),
        "id": payload.get("id"),
        "session_id": payload.get("session_id"),
        "nickname": payload.get("nickname"),
        "role": payload.get("role", "user"),
        "ts": payload.get("ts", _now_ms()),
        "turn_id": payload.get("turn_id"),
        "parts": parts,
        "final": bool(payload.get("final", True)),
    }

def _extract_text_from_parts(parts):
    chunks = []
    for part in parts or []:
        if part.get("kind") == "text" and part.get("text"):
            chunks.append(part["text"])
    return "\n".join(chunks).strip() if chunks else None

class AudioProcessor:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
        last_exc = None
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[
                        "Transcribe this audio to text. Return only the transcript.",
                        types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                    ],
                )
                text = (response.text or "").strip()
                if not text:
                    raise ValueError("Empty transcript returned")
                return text
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(1.5)
        raise last_exc

class MessageStore:
    def __init__(self, max_messages=200):
        self.messages = deque(maxlen=max_messages)
        self.lock = threading.Lock()

    def add(self, message):
        with self.lock:
            self.messages.appendleft(message)

    def get_messages(self, session_id=None, limit=20, after_ms=0):
        with self.lock:
            items = list(self.messages)
            if session_id:
                items = [m for m in items if m.get("session_id") == session_id]
            if after_ms:
                items = [m for m in items if m.get("received_at_ms", 0) > after_ms]
            return items[:limit]

class GifStore:
    def __init__(self):
        self._data = None
        self._lock = threading.Lock()

    def set(self, data: bytes):
        with self._lock:
            self._data = data

    def get(self):
        with self._lock:
            return self._data

class SettingsStore:
    """Tracks the last known value of each settings key, merged from all SETTINGS bus messages."""
    _SKIP = {"session_id", "nickname"}

    def __init__(self):
        self._state = {}
        self._lock = threading.Lock()

    def update(self, settings: dict):
        with self._lock:
            for k, v in settings.items():
                if k not in self._SKIP:
                    self._state[k] = v

    def get(self) -> dict:
        with self._lock:
            return dict(self._state)

class WebRtcSignalStore:
    """Holds pending WebRTC offers until the PC publishes the matching answer on the bus."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pending = {}  # offer_id -> {"event": Event, "answer": dict | None}

    def create(self, offer_id):
        with self._lock:
            self._pending[offer_id] = {"event": threading.Event(), "answer": None}

    def resolve(self, offer_id, answer):
        with self._lock:
            entry = self._pending.get(offer_id)
        if entry is None:
            return
        entry["answer"] = answer
        entry["event"].set()

    def wait(self, offer_id, timeout):
        with self._lock:
            entry = self._pending.get(offer_id)
        if entry is None:
            return None
        entry["event"].wait(timeout)
        with self._lock:
            self._pending.pop(offer_id, None)
        return entry["answer"]

class MessageBusRequestHandler(BaseHTTPRequestHandler):
    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def parse_body(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            return {}

        body = self.rfile.read(content_length)
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()

        if content_type == "application/msgpack":
            return msgpack.unpackb(body, raw=False)

        if content_type == "application/json":
            return json.loads(body.decode("utf-8"))

        raise ValueError(f"Unsupported Content-Type: {content_type}")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/ai_messages":
            params = urllib.parse.parse_qs(parsed.query)
            session_id = params.get("sessionId", [None])[0]
            limit = int(params.get("limit", [20])[0])
            after_ms = int(params.get("after", [0])[0])

            messages = self.server.message_store.get_messages(
                session_id=session_id,
                limit=limit,
                after_ms=after_ms,
            )
            self.send_json({"success": True, "messages": messages})
            return

        if parsed.path == "/api/server_time":
            self.send_json({"ts": _now_ms()})
            return

        if parsed.path == "/api/video":
            gif_data = self.server.gif_store.get()
            if gif_data is None:
                self.send_json({"success": False, "error": "No video available"}, status=404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/gif")
            self.send_header("Content-Length", str(len(gif_data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(gif_data)
            return

        if parsed.path == "/api/styles":
            self.send_json({"names": list(Config.STYLE.keys())})
            return

        if parsed.path == "/api/settings":
            self.send_json({"success": True, "settings": self.server.settings_store.get()})
            return

        if parsed.path == "/api/config":
            self.send_json({"adminNickname": Config.ADMIN_NICKNAME})
            return

        if parsed.path == "/api/turn_credentials":
            # Short-lived Cloudflare TURN credentials for the webapp's
            # RTCPeerConnection; empty list when TURN is not configured.
            servers = self.server.cloudflare_turn.get_ice_servers()
            self.send_json({"success": True, "iceServers": servers or []})
            return

        if parsed.path == "/health":
            self.send_json({"success": True, "status": "ok"})
            return

        if parsed.path == "/api/debug":
            gif = self.server.gif_store.get()
            all_msgs = self.server.message_store.get_messages(limit=200)
            self.send_json({
                "gif_store_kb": len(gif) // 1024 if gif else 0,
                "message_count": len(all_msgs),
                "last_messages": [
                    {k: v for k, v in m.items() if k not in ("image_base64", "parts")}
                    for m in all_msgs[:5]
                ],
            })
            return

        self.send_json({"success": False, "error": "Not found"}, status=404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)

        # Handle raw-body endpoints before parse_body() which only accepts msgpack/json
        if parsed.path == "/api/gif_upload":
            try:
                self.handle_gif_upload(parsed)
            except Exception as exc:
                self.send_json({"success": False, "error": str(exc)}, status=500)
            return

        if parsed.path == "/api/image_upload":
            try:
                self.handle_image_upload(parsed)
            except Exception as exc:
                self.send_json({"success": False, "error": str(exc)}, status=500)
            return

        try:
            payload = self.parse_body()

            if parsed.path == "/api/publish/user_joined":
                self.handle_user_joined(payload)
                return

            if parsed.path == "/api/publish/user_message":
                self.handle_user_message(payload)
                return

            if parsed.path == "/api/publish/user_like":
                self.handle_user_like(payload)
                return

            if parsed.path == "/api/publish/user_gesture":
                self.handle_user_gesture(payload)
                return

            if parsed.path == "/api/publish/settings":
                self.handle_settings(payload)
                return

            if parsed.path == "/api/publish/audio":
                self.handle_transcribe_audio(payload)
                return

            if parsed.path == "/api/publish/user_video":
                self.handle_user_video(payload)
                return

            if parsed.path == "/api/publish/webrtc_offer":
                self.handle_webrtc_offer(payload)
                return

            self.send_json({"success": False, "error": "Not found"}, status=404)

        except Exception as exc:
            self.send_json({"success": False, "error": str(exc)}, status=500)

    def handle_user_joined(self, payload):
        self.server.bus._publish(Bus.USER_JOINED, {
            "session_id": str(payload.get("session_id") or ""),
            "nickname": payload.get("nickname"),
        })
        self.send_json({"success": True, "result": "user_joined published"})

    def handle_user_message(self, payload):
        bus_message = _normalize_http_payload_to_bus_message(payload)
        self.server.bus._publish(Bus.USER_MESSAGE, bus_message)
        self.send_json({"success": True, "result": "user_message published"})

    def handle_user_like(self, payload):
        self.server.bus._publish(Bus.USER_LIKE, {
            "session_id": str(payload.get("session_id") or ""),
            "nickname": payload.get("nickname"),
        })
        self.send_json({"success": True, "result": "user_like published"})

    def handle_user_video(self, payload):
        self.server.bus._publish(Bus.USER_VIDEO, {
            "session_id": str(payload.get("session_id") or ""),
            "nickname": payload.get("nickname"),
        })
        self.send_json({"success": True, "result": "user_video published"})

    def handle_webrtc_offer(self, payload):
        sdp = payload.get("sdp")
        sdp_type = payload.get("type")
        if not sdp or not sdp_type:
            self.send_json({"success": False, "error": "Missing sdp/type"}, status=400)
            return

        offer_id = uuid.uuid4().hex
        self.server.webrtc_signal.create(offer_id)
        self.server.bus._publish(Bus.WEBRTC_OFFER, {
            "offer_id": offer_id,
            "session_id": str(payload.get("session_id") or ""),
            "nickname": payload.get("nickname"),
            "sdp": sdp,
            "type": sdp_type,
        })

        # Each request runs in its own thread (ThreadingHTTPServer), so block
        # here until the PC answers over the bus or the wait times out.
        answer = self.server.webrtc_signal.wait(offer_id, timeout=20.0)
        if answer is None:
            self.send_json({"success": False, "error": "Stream is not available"}, status=504)
            return

        self.send_json({"success": True, "sdp": answer.get("sdp"), "type": answer.get("type")})

    def handle_gif_upload(self, parsed):
        params = urllib.parse.parse_qs(parsed.query)
        session_id = (params.get("session_id", [None])[0]) or ""
        nickname   = (params.get("nickname",   ["NonCarbon Artist"])[0])
        text       = (params.get("text",       ["Video clip"])[0])

        content_length = int(self.headers.get("Content-Length", 0))
        gif_bytes = self.rfile.read(content_length) if content_length > 0 else b""
        if not gif_bytes:
            self.send_json({"success": False, "error": "No GIF data received"}, status=400)
            return

        self.server.gif_store.set(gif_bytes)
        kb = len(gif_bytes) // 1024
        print(f"✓ Phone: GIF uploaded via HTTP ({kb} KB) → /api/video")

        message = {
            "v": 1,
            "id": None,
            "session_id": session_id,
            "nickname": nickname,
            "role": "assistant",
            "turn_id": None,
            "final": True,
            "received_at_ms": _now_ms(),
            "text": text,
            "parts": [],
            "video_url": f"/api/video?t={_now_ms()}",
        }
        self.server.message_store.add(message)
        self.send_json({"success": True, "kb": kb})

    def handle_image_upload(self, parsed):
        params = urllib.parse.parse_qs(parsed.query)
        session_id = (params.get("session_id", [None])[0]) or ""
        nickname   = (params.get("nickname",   ["NonCarbon Artist"])[0])
        text       = (params.get("text",       [""])[0])

        content_length = int(self.headers.get("Content-Length", 0))
        image_bytes = self.rfile.read(content_length) if content_length > 0 else b""
        if not image_bytes:
            self.send_json({"success": False, "error": "No image data received"}, status=400)
            return

        mime = (self.headers.get("Content-Type") or "image/jpeg").split(";")[0].strip()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        kb = len(image_bytes) // 1024
        print(f"✓ Phone: image uploaded via HTTP ({kb} KB)")

        message = {
            "v": 1,
            "id": None,
            "session_id": session_id,
            "nickname": nickname,
            "role": "assistant",
            "turn_id": None,
            "final": True,
            "received_at_ms": _now_ms(),
            "text": text,
            "parts": [],
            "image_base64": image_base64,
            "image_mime_type": mime,
            "image_purpose": "output",
        }
        self.server.message_store.add(message)
        self.send_json({"success": True, "kb": kb})

    def handle_user_gesture(self, payload):
        self.server.bus._publish(Bus.USER_GESTURE, {
            "session_id": str(payload.get("session_id") or ""),
            "nickname": payload.get("nickname"),
            "x": float(payload.get("x", 0.0)),
            "y": float(payload.get("y", 0.0)),
            "z": float(payload.get("z", 0.0)),
        })
        self.send_json({"success": True, "result": "user_gesture published"})

    def handle_settings(self, payload):
        bus_payload = dict(payload)
        if bus_payload.get("session_id") is not None:
            bus_payload["session_id"] = str(bus_payload["session_id"])
        self.server.bus._publish(Bus.SETTINGS, bus_payload)
        self.server.settings_store.update(payload)
        self.send_json({"success": True, "result": "settings published"})

    def handle_transcribe_audio(self, payload):
        if self.server.audio_processor is None:
            self.send_json({"success": False, "error": "Audio processor not available"}, status=503)
            return

        audio_bytes = _coerce_bytes(payload.get("audio_bytes"))
        if not audio_bytes:
            self.send_json({"success": False, "error": "No audio data provided"}, status=400)
            return

        try:
            transcript = self.server.audio_processor.transcribe(audio_bytes)
            self.send_json({"success": True, "status": "success", "transcript": transcript})
        except Exception:
            self.send_json({
                "success": False,
                "status": "error",
                "transcription_error": "Transcription failed. Please try again.",
                "error": "Transcription failed. Please try again.",
            }, status=500)

    def log_message(self, format, *args):
        return

class AIMessageListener(threading.Thread):
    def __init__(self, config, store, gif_store, settings_store, webrtc_signal):
        super().__init__(daemon=True)
        self.config = config
        self.store = store
        self.gif_store = gif_store
        self.settings_store = settings_store
        self.webrtc_signal = webrtc_signal
        self.stopped = threading.Event()

    def run(self):
        print("✓ Phone listener: thread started, subscribing to AI_MESSAGE_TO_PHONE + SETTINGS + WEBRTC_ANSWER")

        while not self.stopped.is_set():
            client = None
            pubsub = None
            try:
                client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    password=self.config.redis_password,
                    ssl=self.config.redis_ssl,
                    decode_responses=False,
                )
                pubsub = client.pubsub(ignore_subscribe_messages=True)
                pubsub.subscribe(Bus.AI_MESSAGE_TO_PHONE, Bus.SETTINGS, Bus.WEBRTC_ANSWER)
                print("✓ Phone listener: subscribed, waiting for messages...")

                while not self.stopped.is_set():
                    message = pubsub.get_message(timeout=1.0)
                    if not message:
                        continue
                    if message.get("type") != "message":
                        continue

                    channel = message.get("channel")
                    if isinstance(channel, bytes):
                        channel = channel.decode("utf-8")

                    payload = msgpack.unpackb(message["data"], raw=False)
                    if not isinstance(payload, dict):
                        print(f"✗ Phone listener: unexpected payload type {type(payload)}")
                        continue

                    if channel == Bus.SETTINGS:
                        self.settings_store.update(payload)
                        continue

                    if channel == Bus.WEBRTC_ANSWER:
                        self.webrtc_signal.resolve(payload.get("offer_id"), payload)
                        continue

                    parts = payload.get("parts", [])
                    has_text = any(p.get("kind") == "text" and p.get("text") for p in parts)
                    has_image = any(p.get("kind") == "image" and p.get("data") is not None for p in parts)

                    print(
                        f"✓ Phone listener: message received "
                        f"text={has_text} image={has_image} "
                        f"from {payload.get('nickname')}"
                    )

                    serialized = self.serialize_payload(payload)
                    self.store.add(serialized)
                    print(f"✓ Phone listener: stored (store size: {len(self.store.messages)})")

            except Exception as exc:
                print(f"✗ Phone listener: error — {exc}. Reconnecting in 3s...")
                if self.stopped.is_set():
                    break
                time.sleep(3)
            finally:
                try:
                    if pubsub is not None:
                        pubsub.close()
                except Exception:
                    pass
                try:
                    if client is not None:
                        client.close()
                except Exception:
                    pass

        print("✓ Phone listener: thread stopped")

    def serialize_payload(self, payload):
        parts = payload.get("parts", [])

        result = {
            "v": payload.get("v", 1),
            "id": payload.get("id"),
            "session_id": payload.get("session_id"),
            "nickname": payload.get("nickname"),
            "role": payload.get("role"),
            "turn_id": payload.get("turn_id"),
            "final": bool(payload.get("final", True)),
            "received_at_ms": _now_ms(),
            "text": _extract_text_from_parts(parts),
            "parts": [],
        }

        for part in parts:
            kind = part.get("kind")

            if kind == "text":
                result["parts"].append({"kind": "text", "text": part.get("text")})

            elif kind == "image" and part.get("data") is not None:
                mime = part.get("mime_type", "image/jpeg")
                data = _coerce_bytes(part.get("data"))
                if mime == "image/gif":
                    # Serve GIF via /api/video to avoid embedding MBs in JSON poll
                    self.gif_store.set(data)
                    result["video_url"] = "/api/video"
                    print(f"✓ Phone listener: GIF stored ({len(data) // 1024} KB) → /api/video")
                else:
                    result["parts"].append({
                        "kind": "image",
                        "mime_type": mime,
                        "purpose": part.get("purpose", "output"),
                        "image_base64": base64.b64encode(data).decode("utf-8"),
                    })

        for part in result["parts"]:
            if part["kind"] == "image" and "image_base64" in part:
                result["image_base64"] = part["image_base64"]
                result["image_mime_type"] = part.get("mime_type", "image/jpeg")
                result["image_purpose"] = part.get("purpose", "output")
                break

        return result

    def stop(self):
        self.stopped.set()

def run_backend(host="0.0.0.0", port=None):
    port = port or int(__import__('os').environ.get('PORT', 8888))
    config = Config()

    bus = Bus(
        config.redis_host,
        config.redis_port,
        config.redis_password,
        config.redis_ssl,
    )
    asyncio.run(bus.connect())

    message_store = MessageStore()
    gif_store = GifStore()
    settings_store = SettingsStore()
    webrtc_signal = WebRtcSignalStore()
    listener = AIMessageListener(config, message_store, gif_store, settings_store, webrtc_signal)
    listener.start()

    try:
        audio_processor = AudioProcessor(api_key=config.GEMINI_API_KEY, model=config.GEMINI_STT_MODEL)
        print("✓ Audio processor: Gemini transcription ready")
    except Exception as exc:
        audio_processor = None
        print(f"✗ Audio processor: failed to initialize — {exc}")

    cloudflare_turn = CloudflareTurn(
        config.CF_TURN_KEY_ID, config.CF_TURN_API_TOKEN, config.CF_TURN_TTL
    )
    if cloudflare_turn.enabled:
        print("✓ Backend: Cloudflare TURN relay enabled")
    else:
        print("⚠ Backend: Cloudflare TURN not configured — viewers on mobile data may fail to connect")

    server = ThreadingHTTPServer((host, port), MessageBusRequestHandler)
    server.bus = bus
    server.cloudflare_turn = cloudflare_turn
    server.message_store = message_store
    server.gif_store = gif_store
    server.settings_store = settings_store
    server.webrtc_signal = webrtc_signal
    server.audio_processor = audio_processor

    print(f"✓ Backend bus server running on http://{host}:{port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        asyncio.run(bus.close())
        server.server_close()

if __name__ == "__main__":
    run_backend()
