"""
archivist_server.py — Local WebSocket server bridging the Archivist Gemini Live session
between the phone browser (audio in) and the PC browser (audio + transcripts out).

Architecture:
  Phone browser  →  Railway backend  →  Redis bus  →  [this server]  →  Gemini Live
  PC browser  ←  ws://localhost:8890  ←  [this server]  ←  Gemini Live

Run on the PC alongside app_pc.py. Requires the same env vars as live.py:
  GOOGLE_SERVICE_ACCOUNT_JSON, VERTEX_PROJECT (optional)
"""

import asyncio
import base64
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared'))
from config import Config
from bus import Bus

from google import genai
from google.genai import types
from google.oauth2 import service_account
import websockets

WS_HOST = os.environ.get("ARCHIVIST_WS_HOST", "localhost")
WS_PORT = int(os.environ.get("ARCHIVIST_WS_PORT", 8890))
SEND_RATE    = Config.LIVE_SEND_RATE    # 16 000 Hz  phone → Gemini
RECEIVE_RATE = Config.LIVE_RECEIVE_RATE # 24 000 Hz  Gemini → PC browser


def _build_gemini_config() -> dict:
    return {
        "response_modalities": ["AUDIO"],
        "temperature": 0.7,
        "speech_config": {
            "voice_config": {
                "prebuilt_voice_config": {"voice_name": "Aoede"}
            },
            "language_code": "pt-PT",
        },
        "system_instruction": {
            "parts": [{"text": (
                "Tu és a Vera, uma arquivista virtual atenciosa e natural. "
                "Falas APENAS Português de Portugal (PT-PT). "
                "Sê concisa, calorosa e conversacional. "
                "Quando te referes a ti própria, usa o género feminino. "
                "REGRA CRÍTICA: se o áudio for inaudível ou incompreensível, "
                "pede para repetir de forma curta (ex.: 'Desculpa, não percebi.'). "
                "Nunca alucines."
            )}],
            "role": "system",
        },
        "input_audio_transcription":  {},
        "output_audio_transcription": {},
        "realtime_input_config": {
            "automatic_activity_detection": {"disabled": True},
        },
    }


class ArchivistServer:
    def __init__(self):
        self._pc_clients: set = set()
        self._ptt_locked_by: str | None = None
        self._audio_queue:   asyncio.Queue = asyncio.Queue(maxsize=80)
        self._ptt_queue:     asyncio.Queue = asyncio.Queue(maxsize=20)

        sa_json     = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
        sa_info     = json.loads(sa_json)
        credentials = service_account.Credentials.from_service_account_info(
            sa_info, scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        self._vertex_project = os.environ.get("VERTEX_PROJECT") or sa_info["project_id"]
        self._gemini = genai.Client(
            vertexai=True,
            project=self._vertex_project,
            location=Config.LIVE_VERTEX_LOCATION,
            credentials=credentials,
        )

        self._bus = Bus(
            Config.redis_host, Config.redis_port,
            Config.redis_password, Config.redis_ssl,
        )

    # ── WebSocket helpers ──────────────────────────────────────────────────────

    async def _broadcast(self, msg: dict):
        if not self._pc_clients:
            return
        data = json.dumps(msg, ensure_ascii=False)
        stale = set()
        for ws in list(self._pc_clients):
            try:
                await ws.send(data)
            except (websockets.ConnectionClosed, Exception):
                stale.add(ws)
        self._pc_clients -= stale

    async def _handle_pc_ws(self, websocket):
        self._pc_clients.add(websocket)
        print(f"✓ Archivist: PC browser connected  ({len(self._pc_clients)} clients)")
        await self._broadcast({"type": "status", "state": "connected"})
        try:
            await websocket.wait_closed()
        finally:
            self._pc_clients.discard(websocket)
            print(f"✓ Archivist: PC browser disconnected ({len(self._pc_clients)} clients)")

    # ── Redis bus poller ───────────────────────────────────────────────────────

    async def _bus_poller(self):
        await self._bus.connect()

        def on_ptt(payload):
            action   = payload.get("action")
            nickname = payload.get("nickname", "phone")

            if action == "start":
                if self._ptt_locked_by is None:
                    self._ptt_locked_by = nickname
                    try:
                        self._ptt_queue.put_nowait(("start", nickname))
                    except asyncio.QueueFull:
                        pass
                    print(f"🎙️  PTT start — {nickname}")
                else:
                    print(f"⚠️  PTT blocked for {nickname} (held by {self._ptt_locked_by})")
            elif action == "stop":
                if self._ptt_locked_by == nickname:
                    self._ptt_locked_by = None
                    try:
                        self._ptt_queue.put_nowait(("stop", nickname))
                    except asyncio.QueueFull:
                        pass
                    print(f"🎙️  PTT stop  — {nickname}")

        def on_audio(payload):
            if self._ptt_locked_by is None:
                return
            raw = payload.get("audio_bytes")
            if raw is None:
                return
            if isinstance(raw, (bytes, bytearray, memoryview)):
                audio_bytes = bytes(raw)
            else:
                try:
                    audio_bytes = base64.b64decode(raw)
                except Exception:
                    return
            try:
                self._audio_queue.put_nowait(audio_bytes)
            except asyncio.QueueFull:
                pass

        self._bus.on(Bus.ARCHIVIST_PTT,   on_ptt)
        self._bus.on(Bus.ARCHIVIST_AUDIO,  on_audio)

        print("✓ Archivist: bus polling started")
        while True:
            await self._bus.poll(timeout=0.01)

    # ── Gemini send loop ───────────────────────────────────────────────────────

    async def _send_to_gemini(self, session):
        talking = False
        try:
            while True:
                # Drain PTT events first
                while not self._ptt_queue.empty():
                    try:
                        action, nick = self._ptt_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    if action == "start" and not talking:
                        await session.send_realtime_input(activity_start=types.ActivityStart())
                        talking = True
                        await self._broadcast({"type": "status", "state": "user_speaking", "nickname": nick})

                    elif action == "stop" and talking:
                        await session.send_realtime_input(activity_end=types.ActivityEnd())
                        talking = False
                        await self._broadcast({"type": "status", "state": "processing"})

                # Forward audio chunk when PTT is active
                try:
                    chunk = self._audio_queue.get_nowait()
                    if talking:
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=chunk,
                                mime_type=f"audio/pcm;rate={SEND_RATE}",
                            )
                        )
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            print(f"❌ send_to_gemini: {exc}")
            traceback.print_exc()

    # ── Gemini receive loop ────────────────────────────────────────────────────

    async def _receive_from_gemini(self, session):
        user_buf   = ""
        gemini_buf = ""

        try:
            while True:
                async for response in session.receive():
                    sc = getattr(response, "server_content", None)

                    # Gemini audio → PC browser
                    if sc and getattr(sc, "model_turn", None):
                        for part in sc.model_turn.parts:
                            inline = getattr(part, "inline_data", None)
                            if inline and inline.data:
                                audio_b64 = base64.b64encode(bytes(inline.data)).decode()
                                await self._broadcast({
                                    "type": "audio",
                                    "data": audio_b64,
                                    "rate": RECEIVE_RATE,
                                })

                    # Barge-in
                    if sc and getattr(sc, "interrupted", False):
                        await self._broadcast({"type": "interrupted"})

                    # User transcript (from phone audio)
                    if sc and getattr(sc, "input_transcription", None):
                        txt = getattr(sc.input_transcription, "text", None) or ""
                        if txt:
                            user_buf += txt
                            await self._broadcast({
                                "type": "user_transcript",
                                "text": user_buf,
                                "delta": txt,
                            })
                            # Publish to bus so phone can poll it
                            ts = int(time.time() * 1000)
                            await asyncio.to_thread(
                                self._bus._publish,
                                Bus.ARCHIVIST_USER_TRANSCRIPT,
                                {"text": txt, "full": user_buf, "ts": ts},
                            )

                    # Gemini transcript
                    if sc and getattr(sc, "output_transcription", None):
                        txt = getattr(sc.output_transcription, "text", None) or ""
                        if txt:
                            gemini_buf += txt
                            await self._broadcast({
                                "type": "gemini_transcript",
                                "text": gemini_buf,
                                "delta": txt,
                            })

                    # Turn complete
                    if sc and getattr(sc, "turn_complete", False):
                        if gemini_buf:
                            await self._broadcast({
                                "type": "gemini_transcript",
                                "text": gemini_buf,
                                "final": True,
                            })
                        user_buf   = ""
                        gemini_buf = ""
                        await self._broadcast({"type": "status", "state": "idle"})

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            try:
                import websockets.exceptions as _wse
                from google.genai import errors as _ge
                normal = (
                    isinstance(exc, _wse.ConnectionClosedOK)
                    or (isinstance(exc, _ge.APIError) and getattr(exc, "code", None) == 1000)
                )
                if not normal:
                    raise
                print("\nℹ️  Gemini connection closed normally.")
            except (ImportError, AttributeError):
                print(f"\n❌ receive_from_gemini: {exc}")
                traceback.print_exc()

    # ── Entry point ────────────────────────────────────────────────────────────

    async def run(self):
        print("=" * 60)
        print("  ARCHIVIST SERVER — Gemini Live via WebSocket")
        print("=" * 60)
        print(f"  Project : {self._vertex_project}")
        print(f"  Model   : {Config.LIVE_MODEL}")
        print(f"  WS      : ws://{WS_HOST}:{WS_PORT}")
        print("=" * 60)

        config = _build_gemini_config()

        async with self._gemini.aio.live.connect(
            model=Config.LIVE_MODEL, config=config
        ) as session:
            print("✅ Gemini Live connected")

            # Kick off initial greeting
            await session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part(text="(Sessão iniciada. Apresenta-te brevemente.)")],
                ),
                turn_complete=True,
            )

            ws_server = await websockets.serve(self._handle_pc_ws, WS_HOST, WS_PORT)
            print(f"✅ WebSocket server on ws://{WS_HOST}:{WS_PORT}")
            print("   Open tfnca.com/archivist in the PC browser to connect.\n")

            tasks = [
                asyncio.create_task(self._bus_poller(),          name="bus_poller"),
                asyncio.create_task(self._send_to_gemini(session), name="send"),
                asyncio.create_task(self._receive_from_gemini(session), name="receive"),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                ws_server.close()
                await ws_server.wait_closed()
                await self._bus.close()


async def main():
    server = ArchivistServer()
    await server.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⛔ Archivist server stopped.")
    except Exception as exc:
        print(f"\n❌ Fatal: {exc}")
        traceback.print_exc()
