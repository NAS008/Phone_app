"""
live.py — Conversa por voz com o Gemini Live API usando o microfone e as colunas do PC.
A deteção de fim de turno (VAD) e o barge-in (interromper o agente ao falar por cima)
são tratados pelo lado do servidor do Gemini — basta transmitir o áudio do microfone
continuamente e reproduzir o áudio que chega de volta.
"""

import asyncio
import json
import os
import traceback
import sys as _sys
import keyboard
import numpy as np
import pyaudio
from scipy.signal import butter, lfilter, lfilter_zi
from google import genai
from google.genai import types
from google.oauth2 import service_account

_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared'))
from config import Config

FORMAT = pyaudio.paInt16   # formato de amostra (constante do pyaudio)

class BandpassFilter:
    def __init__(self, rate, low=Config.LIVE_BANDPASS_LOW_HZ,
                 high=Config.LIVE_BANDPASS_HIGH_HZ, order=Config.LIVE_BANDPASS_ORDER):
        nyq = rate / 2.0
        high = min(high, nyq - 1)
        self.b, self.a = butter(order, [low / nyq, high / nyq], btype="band")
        self._zi = None   # estado do filtro, inicializado no primeiro chunk

    def process(self, data: bytes) -> bytes:
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return data
        if self._zi is None:
            # Arranca o estado no valor da primeira amostra para evitar transiente inicial.
            self._zi = lfilter_zi(self.b, self.a) * samples[0]
        filtered, self._zi = lfilter(self.b, self.a, samples, zi=self._zi)
        return np.clip(filtered, -32768, 32767).astype(np.int16).tobytes()

def default_system_prompt(name: str) -> str:
    """Prompt de sistema predefinido, parametrizado pelo nome do agente."""
    return (
        f"Tu és a {name}, uma assistente virtual simpática e natural. "
        "Falas APENAS Português de Portugal (PT-PT). "
        "Sê concisa, calorosa e conversacional, como numa conversa real ao telefone. "
        "Quando te referes a ti própria, usa o género feminino. "
        "Começa a interação por te apresentares brevemente e perguntares em que podes ajudar. "
        "REGRA CRÍTICA: se o áudio do utilizador for inaudível, ruído ou incompreensível, "
        "não inventes uma resposta — pede para repetir de forma curta e natural "
        "(ex.: 'Desculpa, não percebi. Podes repetir?'). "
        "Evita silêncios longos e nunca alucines."
    )

def build_config(system_prompt: str) -> dict:
    """Constrói a config do Gemini Live com o prompt de sistema dado."""
    return {
        "response_modalities": ["AUDIO"],
        "temperature": 0.6,

        "speech_config": {
            "voice_config": {
                "prebuilt_voice_config": {"voice_name": "Aoede"}
            },
            "language_code": "pt-PT",
        },

        "system_instruction": {
            "parts": [{"text": system_prompt}],
            "role": "system",
        },

        # Transcrições para vermos no terminal o que cada lado disse.
        "input_audio_transcription":  {},
        "output_audio_transcription": {},

        # VAD do servidor DESLIGADO — usamos push-to-talk (ver PTT_KEY)
        "realtime_input_config": {
            "automatic_activity_detection": {"disabled": True},
        },
    }

class Live:
    def __init__(self, name: str = "Vera", system_prompt: str | None = None,
                 use_filter: bool = True,
                 bandpass_low: int = Config.LIVE_BANDPASS_LOW_HZ,
                 bandpass_high: int = Config.LIVE_BANDPASS_HIGH_HZ,
                 bandpass_order: int = Config.LIVE_BANDPASS_ORDER):
        self.name           = name
        self.system_prompt  = system_prompt if system_prompt is not None else default_system_prompt(name)
        self.config         = build_config(self.system_prompt)
        self.use_filter     = use_filter
        self.bandpass_low   = bandpass_low
        self.bandpass_high  = bandpass_high
        self.bandpass_order = bandpass_order
        self.mic_queue      = asyncio.Queue(maxsize=20)
        self.speaker_queue  = asyncio.Queue()
        self.stop_event     = asyncio.Event()
        self.worker_tasks   = []

        # Autenticação Google + cliente genai (Vertex AI) para o Gemini Live.
        sa_json      = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
        sa_info      = json.loads(sa_json)
        credentials  = service_account.Credentials.from_service_account_info(
            sa_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        self.vertex_project = os.environ.get("VERTEX_PROJECT") or sa_info["project_id"]
        self.client = genai.Client(
            vertexai=True,
            project=self.vertex_project,
            location=Config.LIVE_VERTEX_LOCATION,
            credentials=credentials,
        )

    async def capture_microphone(self):
        pya    = pyaudio.PyAudio()
        stream = None
        try:
            device_info = (
                pya.get_device_info_by_index(Config.LIVE_MIC_DEVICE_INDEX)
                if Config.LIVE_MIC_DEVICE_INDEX is not None
                else pya.get_default_input_device_info()
            )
            print(f"🎤 Microfone: [{int(device_info['index'])}] {device_info['name']}")

            stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT, channels=Config.LIVE_CHANNELS, rate=Config.LIVE_SEND_RATE,
                input=True, input_device_index=int(device_info["index"]),
                frames_per_buffer=Config.LIVE_CHUNK,
            )

            bandpass = BandpassFilter(
                Config.LIVE_SEND_RATE, low=self.bandpass_low,
                high=self.bandpass_high, order=self.bandpass_order,
            ) if self.use_filter else None

            while not self.stop_event.is_set():
                data = await asyncio.to_thread(stream.read, Config.LIVE_CHUNK, exception_on_overflow=False)
                if bandpass is not None:
                    data = bandpass.process(data)   # mantém só a banda de voz antes de enviar
                try:
                    self.mic_queue.put_nowait(data)
                except asyncio.QueueFull:
                    pass   # se o consumidor estiver atrasado, descarta — áudio é tempo-real
        except Exception as exc:
            print(f"❌ capture_microphone: {exc}")
            traceback.print_exc()
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            pya.terminate()

    async def play_audio(self):
        pya    = pyaudio.PyAudio()
        stream = None
        try:
            device_info = (
                pya.get_device_info_by_index(Config.LIVE_SPEAKER_DEVICE_INDEX)
                if Config.LIVE_SPEAKER_DEVICE_INDEX is not None
                else pya.get_default_output_device_info()
            )
            print(f"🔈 Saída:    [{int(device_info['index'])}] {device_info['name']}")

            stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT, channels=Config.LIVE_CHANNELS, rate=Config.LIVE_RECEIVE_RATE,
                output=True, output_device_index=int(device_info["index"]),
                frames_per_buffer=Config.LIVE_CHUNK,
            )

            while not self.stop_event.is_set():
                try:
                    audio_data = await asyncio.wait_for(self.speaker_queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                if audio_data is None:        # sentinela de "limpar fila" (barge-in)
                    continue
                await asyncio.to_thread(stream.write, audio_data)
        except Exception as exc:
            print(f"❌ play_audio: {exc}")
            traceback.print_exc()
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            pya.terminate()

    def _drain_speaker_queue(self):
        """Esvazia a fila das colunas sem bloquear (calar)."""
        while not self.speaker_queue.empty():
            try:
                self.speaker_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def send_to_gemini(self, session):
        talking = False
        try:
            while not self.stop_event.is_set():
                pressed = keyboard.is_pressed(Config.LIVE_PTT_KEY)

                # ── Início do turno: tecla acabou de ser premida ──────────────
                if pressed and not talking:
                    self._drain_speaker_queue()   # barge-in: cala já localmente
                    await session.send_realtime_input(activity_start=types.ActivityStart())
                    talking = True
                    print("\n🎙️  [a falar — solta a tecla quando acabares]")

                # ── Fim do turno: tecla acabou de ser solta ───────────────────
                elif not pressed and talking:
                    await session.send_realtime_input(activity_end=types.ActivityEnd())
                    talking = False
                    print("⏳ [a processar…]")

                # Recolhe um chunk do microfone (timeout curto para reagir à tecla).
                try:
                    chunk = await asyncio.wait_for(self.mic_queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue

                # Só envia áudio enquanto a tecla está premida; caso contrário, descarta.
                if talking:
                    await session.send_realtime_input(
                        audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={Config.LIVE_SEND_RATE}")
                    )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            print(f"❌ send_to_gemini: {exc}")
            traceback.print_exc()

    async def receive_from_gemini(self, session):
        agent_line = ""
        user_line  = ""
        current    = None   # "user" | "agent" — quem está a "escrever" a linha ao vivo
        live_len   = 0      # comprimento da última linha ao vivo, para apagar o que sobra

        def render(prefix: str, text: str):
            nonlocal live_len
            line = f"{prefix}{text}"
            pad  = max(0, live_len - len(line))      # apaga restos de uma linha anterior maior
            print(f"\r{line}{' ' * pad}", end="", flush=True)
            live_len = len(line)

        def finish_line():
            nonlocal live_len
            if live_len:
                print()                              # fixa a linha atual e passa à seguinte
                live_len = 0

        try:
            while not self.stop_event.is_set():
                async for response in session.receive():
                    sc = getattr(response, "server_content", None)

                    # ── Áudio do agente ───────────────────────────────────────
                    if sc and getattr(sc, "model_turn", None):
                        for part in sc.model_turn.parts:
                            inline = getattr(part, "inline_data", None)
                            if inline and inline.data:
                                await self.speaker_queue.put(bytes(inline.data))

                    # ── Barge-in: o utilizador interrompeu o agente ───────────
                    if sc and getattr(sc, "interrupted", False):
                        # Esvazia o áudio por reproduzir para o agente parar já de falar.
                        self._drain_speaker_queue()
                        finish_line()
                        current = None
                        print("[interrompido]")

                    # ── Transcrição do utilizador (chega em deltas) ───────────
                    if sc and getattr(sc, "input_transcription", None):
                        txt = getattr(sc.input_transcription, "text", None) or ""
                        if txt:
                            if current != "user":
                                finish_line()
                                current, user_line = "user", ""
                            user_line += txt
                            render("🗣  Tu:     ", user_line)

                    # ── Transcrição do agente (chega em deltas) ───────────────
                    if sc and getattr(sc, "output_transcription", None):
                        txt = getattr(sc.output_transcription, "text", None) or ""
                        if txt:
                            if current != "agent":
                                finish_line()
                                current, agent_line = "agent", ""
                            agent_line += txt
                            render(f"🤖 {self.name}:   ", agent_line)

                    # ── Fim do turno do agente ────────────────────────────────
                    if sc and getattr(sc, "turn_complete", False):
                        finish_line()
                        current, agent_line, user_line = None, "", ""
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            # O SDK envolve um fecho normal do WebSocket (código 1000) como APIError.
            import websockets.exceptions as _wse
            from google.genai import errors as _ge
            normal_close = (
                isinstance(exc, _wse.ConnectionClosedOK)
                or (isinstance(exc, _ge.APIError) and getattr(exc, "code", None) == 1000)
            )
            if normal_close:
                print("\nℹ️  Conexão encerrada pelo servidor.")
            else:
                print(f"\n❌ receive_from_gemini: {exc}")
                traceback.print_exc()
        finally:
            self.stop_event.set()

    async def wait_for_exit(self):
        while not self.stop_event.is_set():
            try:
                cmd = await asyncio.to_thread(input, "")
            except (EOFError, KeyboardInterrupt):
                self.stop_event.set()
                return
            except Exception:
                continue
            if cmd.strip().lower() in {"q", "quit", "exit", "sair"}:
                print("\n[Q] premido — a terminar…")
                self.stop_event.set()
                return

    async def run(self):
        print("=" * 60)
        print("  GEMINI LIVE — Conversa Local (microfone + colunas)")
        print("=" * 60)
        print(f"  Projeto : {self.vertex_project}")
        print(f"  Modelo  : {Config.LIVE_MODEL}")
        print(f"  Voz     : Aoede  |  Língua: pt-PT")
        print("=" * 60)
        print("  A conectar ao Gemini Live API…\n")

        try:
            async with self.client.aio.live.connect(model=Config.LIVE_MODEL, config=self.config) as session:
                print("✅ Conectado!")
                print(f"   🎙️  Mantém [{Config.LIVE_PTT_KEY.upper()}] premido para falar; solta para a {self.name} responder.")
                print("   (carrega na tecla enquanto ela fala para a interromper)")
                print("   (escreve 'q' + Enter ou Ctrl+C para terminar)\n")

                self.worker_tasks = [
                    asyncio.create_task(self.capture_microphone(),       name="Microfone"),
                    asyncio.create_task(self.play_audio(),               name="Saída"),
                    asyncio.create_task(self.send_to_gemini(session),    name="Enviar"),
                    asyncio.create_task(self.receive_from_gemini(session), name="Receber"),
                ]

                # Com o VAD desligado, o agente não fala sem ser provocado. Damos-lhe um
                # turno de utilizador inicial (só texto, sem microfone) para ele se
                # apresentar primeiro, como manda o system_prompt.
                await session.send_client_content(
                    turns=types.Content(
                        role="user",
                        parts=[types.Part(text="(A chamada começou. Apresenta-te brevemente e pergunta em que podes ajudar.)")],
                    ),
                    turn_complete=True,
                )

                await asyncio.create_task(self.wait_for_exit(), name="Sair")

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            print(f"❌ Erro na sessão: {exc}")
            traceback.print_exc()
        finally:
            self.stop_event.set()
            for task in self.worker_tasks:
                task.cancel()
            if self.worker_tasks:
                await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        print("\n" + "=" * 60)
        print("✅ Conversa terminada.")
        print("=" * 60)

async def main():
    agent = Live(name="Sara", use_filter=False)
    await agent.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⛔ Terminado pelo utilizador.")
    except Exception as exc:
        print(f"\n❌ Erro fatal: {exc}")
        traceback.print_exc()
    finally:
        print("🔌 Desligado!")
