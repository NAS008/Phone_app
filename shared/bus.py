import inspect
import time
import uuid
import msgpack
import redis

class Bus:
    SESSION = "session"
    USER_JOINED = "user_joined"
    USER_MESSAGE = "user_message"
    USER_LIKE = "user_like"
    USER_GESTURE = "user_gesture"
    USER_VIDEO = "user_video"
    AI_MESSAGE_TO_PC = "ai_message_to_pc"
    AI_MESSAGE_TO_PHONE = "ai_message_to_phone"
    SETTINGS = "settings"

    CHANNELS = (
        SESSION,
        USER_JOINED,
        USER_MESSAGE,
        USER_LIKE,
        USER_GESTURE,
        USER_VIDEO,
        AI_MESSAGE_TO_PC,
        AI_MESSAGE_TO_PHONE,
        SETTINGS,
    )

    def __init__(self, host, port, password, ssl):
        self.host = host
        self.port = port
        self.password = password
        self.ssl = ssl
        self.client = None
        self.pubsub = None
        self._handlers = {}

    def on(self, channel, handler):
        if channel not in self.CHANNELS:
            raise ValueError(f"Unknown channel: {channel}")
        self._handlers[channel] = handler

    async def connect(self):
        self.client = redis.Redis(
            host=self.host,
            port=self.port,
            password=self.password,
            ssl=self.ssl,
            decode_responses=False,
        )
        self.client.ping()
        self.pubsub = self.client.pubsub(ignore_subscribe_messages=True)
        self.pubsub.subscribe(*self.CHANNELS)
        print("✓ BUS: connected")

    def _publish(self, channel, payload):
        if self.client is None:
            raise RuntimeError("Bus is not connected")
        packed = msgpack.packb(payload, use_bin_type=True)
        self.client.publish(channel, packed)

    @staticmethod
    def _now_ms():
        return int(time.time() * 1000)

    @staticmethod
    def _msg_id():
        return uuid.uuid4().hex

    @staticmethod
    def _text_part(text):
        return {"kind": "text", "text": text}

    @staticmethod
    def _image_part(image_bytes, mime_type="image/jpeg", purpose="input"):
        return {
            "kind": "image",
            "mime_type": mime_type,
            "purpose": purpose,
            "data": image_bytes,
        }

    def _message_payload(self, session_id, nickname, text=None, image_bytes=None,
                         image_mime_type="image/jpeg", image_purpose="input",
                         message_id=None, turn_id=None, final=True):
        parts = []
        if text is not None:
            parts.append(self._text_part(text))
        if image_bytes is not None:
            parts.append(self._image_part(image_bytes, mime_type=image_mime_type, purpose=image_purpose))
        return {
            "v": 1,
            "id": message_id or self._msg_id(),
            "session_id": str(session_id),
            "nickname": nickname,
            "role": "user" if nickname != "NonCarbon Artist" else "assistant",
            "ts": self._now_ms(),
            "turn_id": turn_id,
            "parts": parts,
            "final": bool(final),
        }

    async def publish_session(self, session_id):
        self._publish(self.SESSION, {"session_id": str(session_id)})

    async def publish_user_joined(self, session_id, nickname):
        self._publish(self.USER_JOINED, {
            "session_id": str(session_id),
            "nickname": nickname,
        })

    async def publish_user_message(self, session_id="0", nickname="Nuno", text=None,
                                   image_bytes=None, image_mime_type="image/jpeg",
                                   image_purpose="input", turn_id=None, final=True):
        self._publish(self.USER_MESSAGE, self._message_payload(
            session_id=session_id, nickname=nickname, text=text,
            image_bytes=image_bytes, image_mime_type=image_mime_type,
            image_purpose=image_purpose, turn_id=turn_id, final=final,
        ))

    async def publish_user_like(self, session_id="0", nickname="Nuno"):
        self._publish(self.USER_LIKE, {
            "session_id": str(session_id),
            "nickname": nickname,
        })

    async def publish_user_video(self, session_id="0", nickname="Nuno"):
        self._publish(self.USER_VIDEO, {
            "session_id": str(session_id),
            "nickname": nickname,
        })

    async def publish_user_gesture(self, session_id="0", nickname="Nuno", x=0.0, y=0.0, z=0.0):
        self._publish(self.USER_GESTURE, {
            "session_id": str(session_id),
            "nickname": nickname,
            "x": float(x),
            "y": float(y),
            "z": float(z),
        })

    async def publish_ai_message_to_pc(self, session_id="0", nickname="NonCarbon Artist",
                                       text=None, image_bytes=None, image_mime_type="image/jpeg",
                                       image_purpose="output", turn_id=None, final=True):
        self._publish(self.AI_MESSAGE_TO_PC, self._message_payload(
            session_id=session_id, nickname=nickname, text=text,
            image_bytes=image_bytes, image_mime_type=image_mime_type,
            image_purpose=image_purpose, turn_id=turn_id, final=final,
        ))

    async def publish_ai_message_to_phone(self, session_id="0", nickname="NonCarbon Artist",
                                          text=None, image_bytes=None, image_mime_type="image/jpeg",
                                          image_purpose="output", turn_id=None, final=True):
        self._publish(self.AI_MESSAGE_TO_PHONE, self._message_payload(
            session_id=session_id, nickname=nickname, text=text,
            image_bytes=image_bytes, image_mime_type=image_mime_type,
            image_purpose=image_purpose, turn_id=turn_id, final=final,
        ))

    async def publish_settings(self, session_id=None, nickname=None, metadata=None, **settings):
        payload = settings.copy()
        if session_id is not None:
            payload["session_id"] = str(session_id)
        if nickname is not None:
            payload["nickname"] = nickname
        if metadata:
            payload.update(metadata)
        self._publish(self.SETTINGS, payload)

    async def _call_handler(self, handler, *args):
        result = handler(*args)
        if inspect.isawaitable(result):
            return await result
        return result

    async def poll(self, timeout=0.01):
        if self.pubsub is None:
            raise RuntimeError("Bus is not connected")

        msg = self.pubsub.get_message(ignore_subscribe_messages=True, timeout=timeout)
        if not msg:
            return False

        channel = msg["channel"]
        if isinstance(channel, bytes):
            channel = channel.decode("utf-8")

        handler = self._handlers.get(channel)
        if handler is None:
            return True

        try:
            payload = msgpack.unpackb(msg["data"], raw=False)
            if not isinstance(payload, dict):
                print(f"✗ Bus: invalid payload for {channel}: {type(payload).__name__}")
                return True

            if channel == self.SESSION:
                return await self._call_handler(handler, payload.get("session_id"))

            elif channel == self.USER_JOINED:
                return await self._call_handler(
                    handler,
                    payload.get("session_id"),
                    payload.get("nickname"),
                )

            elif channel in (self.USER_MESSAGE, self.AI_MESSAGE_TO_PC, self.AI_MESSAGE_TO_PHONE):
                return await self._call_handler(
                    handler,
                    payload.get("session_id"),
                    payload.get("nickname"),
                    payload.get("parts", []),
                    payload,
                )

            elif channel in (self.USER_LIKE, self.USER_VIDEO):
                return await self._call_handler(
                    handler,
                    payload.get("session_id"),
                    payload.get("nickname"),
                )

            elif channel == self.USER_GESTURE:
                return await self._call_handler(
                    handler,
                    payload.get("session_id"),
                    payload.get("nickname"),
                    payload.get("x", 0.0),
                    payload.get("y", 0.0),
                    payload.get("z", 0.0),
                )

            elif channel == self.SETTINGS:
                return await self._call_handler(handler, payload)

        except Exception as e:
            print(f"✗ Bus: failed to process {channel}: {e}")

        return True

    async def close(self):
        if self.pubsub is not None:
            try:
                self.pubsub.close()
            finally:
                self.pubsub = None
        if self.client is not None:
            try:
                self.client.close()
            finally:
                self.client = None
