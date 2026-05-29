# # # # # # #
# The       #
# First     #
# NonCarbon #
# Artist    #
# # # # # # #

# Phone App uses the phone sensors to bridge interaction with user and NonCarbon Artist
# Has all the UI and design window
# Input: Receives text or audio and/ or image bytes, receives commands in Admin mode
# Output: Post messages in the bus with instructions
# ----------------------------------------------------------------
# ✗ Complete tests and publish on the web
# ✗ Test gesture with phone and latency

import asyncio
import base64
import json
import threading
import time
import msgpack
import redis
import urllib.parse
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'shared'))
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from google import genai
from google.genai import types
from config import Config
from bus import Bus

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

def _normalize_http_part(part):
    kind = part.get("kind")

    if kind == "text":
        return {"kind": "text", "text": part.get("text")}

    if kind == "image":
        return {
            "kind": "image",
            "mime_type": part.get("mime_type", "image/jpeg"),
            "purpose": part.get("purpose", "input"),
            "data": _coerce_bytes(part.get("data")),
        }

    raise ValueError(f"Unsupported part kind: {kind}")

def _normalize_http_payload_to_bus_message(payload):
    parts = payload.get("parts")
    if parts is None:
        parts = []
        if payload.get("text") is not None:
            parts.append({"kind": "text", "text": payload.get("text")})
        if payload.get("image_bytes") is not None:
            parts.append({
                "kind": "image",
                "mime_type": payload.get("image_mime_type", "image/jpeg"),
                "purpose": payload.get("image_purpose", "input"),
                "data": _coerce_bytes(payload.get("image_bytes")),
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

        if parsed.path == "/health":
            self.send_json({"success": True, "status": "ok"})
            return

        self.send_json({"success": False, "error": "Not found"}, status=404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)

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

            self.send_json({"success": False, "error": "Not found"}, status=404)

        except Exception as exc:
            self.send_json({"success": False, "error": str(exc)}, status=500)

    def handle_user_joined(self, payload):
        asyncio.run(self.server.bus.publish_user_joined(
            session_id=payload.get("session_id"),
            nickname=payload.get("nickname"),
        ))
        self.send_json({"success": True, "result": "user_joined published"})

    def handle_user_message(self, payload):
        bus_message = _normalize_http_payload_to_bus_message(payload)
        self.server.bus._publish(Bus.USER_MESSAGE, bus_message)
        self.send_json({"success": True, "result": "user_message published"})

    def handle_user_like(self, payload):
        asyncio.run(self.server.bus.publish_user_like(
            session_id=payload.get("session_id"),
            nickname=payload.get("nickname"),
        ))
        self.send_json({"success": True, "result": "user_like published"})

    def handle_user_gesture(self, payload):
        asyncio.run(self.server.bus.publish_user_gesture(
            session_id=payload.get("session_id"),
            nickname=payload.get("nickname"),
            x=float(payload.get("x", 0.0)),
            y=float(payload.get("y", 0.0)),
            z=float(payload.get("z", 0.0)),
        ))
        self.send_json({"success": True, "result": "user_gesture published"})

    def handle_settings(self, payload):
        settings = {
            k: v for k, v in payload.items()
            if k not in {"session_id", "nickname"}
        }
        asyncio.run(self.server.bus.publish_settings(
            session_id=payload.get("session_id"),
            nickname=payload.get("nickname"),
            **settings,
        ))
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
        except Exception as exc:
            self.send_json({
                "success": False,
                "status": "error",
                "transcription_error": str(exc),
                "error": str(exc),
            }, status=500)

    def log_message(self, format, *args):
        return

class AIMessageListener(threading.Thread):
    def __init__(self, config, store):
        super().__init__(daemon=True)
        self.config = config
        self.store = store
        self.stopped = threading.Event()

    def run(self):
        print("✓ Phone listener: thread started, subscribing to AI_MESSAGE_TO_PHONE")

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
                pubsub.subscribe(Bus.AI_MESSAGE_TO_PHONE)
                print("✓ Phone listener: subscribed, waiting for messages...")

                while not self.stopped.is_set():
                    message = pubsub.get_message(timeout=1.0)
                    if not message:
                        continue
                    if message.get("type") != "message":
                        continue

                    payload = msgpack.unpackb(message["data"], raw=False)
                    if not isinstance(payload, dict):
                        print(f"✗ Phone listener: unexpected payload type {type(payload)}")
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
                data = _coerce_bytes(part.get("data"))
                result["parts"].append({
                    "kind": "image",
                    "mime_type": part.get("mime_type", "image/jpeg"),
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

def run_backend(host="0.0.0.0", port=8888):
    config = Config()

    bus = Bus(
        config.redis_host,
        config.redis_port,
        config.redis_password,
        config.redis_ssl,
    )
    asyncio.run(bus.connect())

    message_store = MessageStore()
    listener = AIMessageListener(config, message_store)
    listener.start()

    try:
        audio_processor = AudioProcessor(api_key=config.GEMINI_API_KEY, model=config.GEMINI_STT_MODEL)
        print("✓ Audio processor: Gemini transcription ready")
    except Exception as exc:
        audio_processor = None
        print(f"✗ Audio processor: failed to initialize — {exc}")

    server = ThreadingHTTPServer((host, port), MessageBusRequestHandler)
    server.bus = bus
    server.message_store = message_store
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

# import asyncio
# import base64
# import json
# import threading
# import time
# import msgpack
# import redis
# import urllib.parse
# from collections import deque
# from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
# from config import Config
# from bus import Bus

# class MessageStore:
#     def __init__(self, max_messages=200):
#         self.messages = deque(maxlen=max_messages)
#         self.lock = threading.Lock()

#     def add(self, message):
#         with self.lock:
#             self.messages.appendleft(message)

#     def get_messages(self, session_id=None, limit=20, after_ms=0):
#         with self.lock:
#             items = list(self.messages)
#             if session_id:
#                 items = [m for m in items if m.get("session_id") == session_id]
#             if after_ms:
#                 items = [m for m in items if m.get("received_at_ms", 0) > after_ms]
#             return items[:limit]

# class MessageBusRequestHandler(BaseHTTPRequestHandler):
#     def send_json(self, data, status=200):
#         body = json.dumps(data).encode("utf-8")
#         self.send_response(status)
#         self.send_header("Content-Type", "application/json")
#         self.send_header("Content-Length", str(len(body)))
#         self.send_header("Access-Control-Allow-Origin", "*")
#         self.send_header("Access-Control-Allow-Headers", "Content-Type")
#         self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
#         self.end_headers()
#         self.wfile.write(body)

#     def do_OPTIONS(self):
#         self.send_response(204)
#         self.send_header("Access-Control-Allow-Origin", "*")
#         self.send_header("Access-Control-Allow-Headers", "Content-Type")
#         self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
#         self.end_headers()

#     def parse_body(self):
#         content_length = int(self.headers.get("Content-Length", 0))
#         if content_length <= 0:
#             return {}

#         body = self.rfile.read(content_length)
#         content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()

#         if content_type == "application/msgpack":
#             return msgpack.unpackb(body, raw=False)

#         if content_type == "application/json":
#             return json.loads(body.decode("utf-8"))

#         raise ValueError(f"Unsupported Content-Type: {content_type}")

#     def do_GET(self):
#         parsed = urllib.parse.urlparse(self.path)

#         if parsed.path == "/api/ai_messages":
#             params = urllib.parse.parse_qs(parsed.query)
#             session_id = params.get("sessionId", [None])[0]
#             limit = int(params.get("limit", [20])[0])
#             after_ms = int(params.get("after", [0])[0])
#             all_messages = self.server.message_store.get_messages(
#                 session_id=None,
#                 limit=200,
#                 after_ms=0,
#             )
#             messages = self.server.message_store.get_messages(
#                 session_id=session_id,
#                 limit=limit,
#                 after_ms=after_ms,
#             )
#             # stored_ids = {m.get("session_id") for m in all_messages}
#             # if messages:
#             #     has_images = sum(1 for m in messages if "image_base64" in m)
#             #     print(f"✓ Phone poll: returning {len(messages)} msg(s), {has_images} with image (after={after_ms}, session={session_id})")
#             # else:
#             #     print(f"✗ Phone poll: 0 messages (store has {len(all_messages)} total, sessions={stored_ids}, requested session={session_id}, after={after_ms})")
#             self.send_json({"success": True, "messages": messages})
#             return

#         if parsed.path == "/api/server_time":
#             self.send_json({"ts": int(time.time() * 1000)})
#             return

#         if parsed.path == "/health":
#             self.send_json({"success": True, "status": "ok"})
#             return

#         self.send_json({"success": False, "error": "Not found"}, status=404)

#     def do_POST(self):
#         parsed = urllib.parse.urlparse(self.path)

#         try:
#             payload = self.parse_body()

#             if parsed.path == "/api/publish/user_joined":
#                 self.handle_user_joined(payload)
#                 return

#             if parsed.path == "/api/publish/user_message":
#                 self.handle_user_message(payload)
#                 return

#             if parsed.path == "/api/publish/user_like":
#                 self.handle_user_like(payload)
#                 return

#             if parsed.path == "/api/publish/user_gesture":
#                 self.handle_user_gesture(payload)
#                 return

#             if parsed.path == "/api/publish/settings":
#                 self.handle_settings(payload)
#                 return

#             self.send_json({"success": False, "error": "Not found"}, status=404)

#         except Exception as exc:
#             self.send_json({"success": False, "error": str(exc)}, status=500)

#     def handle_user_joined(self, payload):
#         asyncio.run(self.server.bus.publish_user_joined(
#             session_id=payload.get("session_id"),
#             nickname=payload.get("nickname"),
#         ))
#         self.send_json({"success": True, "result": "user_joined published"})

#     def handle_user_message(self, payload):
#         asyncio.run(self.server.bus.publish_user_message(
#             session_id=payload.get("session_id"),
#             nickname=payload.get("nickname"),
#             text=payload.get("text"),
#             audio_bytes=payload.get("audio_bytes"),
#             image_bytes=payload.get("image_bytes"),
#         ))
#         self.send_json({"success": True, "result": "user_message published"})

#     def handle_user_like(self, payload):
#         asyncio.run(self.server.bus.publish_user_like(
#             session_id=payload.get("session_id"),
#             nickname=payload.get("nickname"),
#         ))
#         self.send_json({"success": True, "result": "user_like published"})

#     def handle_user_gesture(self, payload):
#         asyncio.run(self.server.bus.publish_user_gesture(
#             session_id=payload.get("session_id"),
#             nickname=payload.get("nickname"),
#             x=float(payload.get("x", 0.0)),
#             y=float(payload.get("y", 0.0)),
#             z=float(payload.get("z", 0.0)),
#         ))
#         self.send_json({"success": True, "result": "user_gesture published"})

#     def handle_settings(self, payload):
#         settings = {
#             k: v for k, v in payload.items()
#             if k not in {"session_id", "nickname"}
#         }
#         asyncio.run(self.server.bus.publish_settings(
#             session_id=payload.get("session_id"),
#             nickname=payload.get("nickname"),
#             **settings,
#         ))
#         self.send_json({"success": True, "result": "settings published"})

#     def log_message(self, format, *args):
#         return

# class AIMessageListener(threading.Thread):
#     def __init__(self, config, store):
#         super().__init__(daemon=True)
#         self.config = config
#         self.store = store
#         self.stopped = threading.Event()

#     def run(self):
#         print("✓ Phone listener: thread started, subscribing to AI_MESSAGE_TO_PHONE")

#         while not self.stopped.is_set():
#             client = None
#             pubsub = None
#             try:
#                 client = redis.Redis(
#                     host=self.config.redis_host,
#                     port=self.config.redis_port,
#                     password=self.config.redis_password,
#                     ssl=self.config.redis_ssl,
#                     decode_responses=False,
#                 )
#                 pubsub = client.pubsub(ignore_subscribe_messages=True)
#                 pubsub.subscribe(Bus.AI_MESSAGE_TO_PHONE)
#                 print("✓ Phone listener: subscribed, waiting for messages...")

#                 while not self.stopped.is_set():
#                     message = pubsub.get_message(timeout=1.0)
#                     if not message:
#                         continue
#                     if message.get("type") != "message":
#                         continue

#                     payload = msgpack.unpackb(message["data"], raw=False)
#                     if not isinstance(payload, dict):
#                         print(f"✗ Phone listener: unexpected payload type {type(payload)}")
#                         continue

#                     has_image = payload.get("image_bytes") is not None
#                     has_text = payload.get("text") is not None
#                     print(f"✓ Phone listener: message received text={has_text} image={has_image} from {payload.get('nickname')}")

#                     serialized = self.serialize_payload(payload)
#                     self.store.add(serialized)
#                     print(f"✓ Phone listener: stored (store size: {len(self.store.messages)})")

#             except Exception as exc:
#                 print(f"✗ Phone listener: error — {exc}. Reconnecting in 3s...")
#                 if self.stopped.is_set():
#                     break
#                 time.sleep(3)
#             finally:
#                 try:
#                     if pubsub is not None:
#                         pubsub.close()
#                 except Exception:
#                     pass
#                 try:
#                     if client is not None:
#                         client.close()
#                 except Exception:
#                     pass

#         print("✓ Phone listener: thread stopped")

#     def serialize_payload(self, payload):
#         result = {
#             "text": payload.get("text"),
#             "session_id": payload.get("session_id"),
#             "nickname": payload.get("nickname"),
#             "received_at_ms": int(time.time() * 1000),
#         }

#         if payload.get("audio_bytes") is not None:
#             result["audio_base64"] = base64.b64encode(payload["audio_bytes"]).decode("utf-8")

#         if payload.get("image_bytes") is not None:
#             result["image_base64"] = base64.b64encode(payload["image_bytes"]).decode("utf-8")

#         return result

#     def stop(self):
#         self.stopped.set()

# def run_backend(host="0.0.0.0", port=8888):
#     config = Config()

#     bus = Bus(
#         config.redis_host,
#         config.redis_port,
#         config.redis_password,
#         config.redis_ssl,
#     )
#     asyncio.run(bus.connect())

#     message_store = MessageStore()
#     listener = AIMessageListener(config, message_store)
#     listener.start()

#     server = ThreadingHTTPServer((host, port), MessageBusRequestHandler)
#     server.bus = bus
#     server.message_store = message_store

#     print(f"✓ Backend bus server running on http://{host}:{port}")

#     try:
#         server.serve_forever()
#     except KeyboardInterrupt:
#         pass
#     finally:
#         listener.stop()
#         asyncio.run(bus.close())
#         server.server_close()

# if __name__ == "__main__":
#     run_backend()
