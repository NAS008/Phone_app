import asyncio
import random
import uuid
import cv2
import numpy as np
from google.genai import types as gtypes


class Director:
    """
    Auto-play controller.  When enabled it drives two independent loops:
      - Prompt loop  (every DIRECTOR_PROMPT_INTERVAL s): asks Gemini for a
        creative prompt, then injects a fake USER_MESSAGE on the bus so the
        normal app_server image-generation flow runs untouched.
      - Scene loop   (every DIRECTOR_SCENE_INTERVAL s): randomly either
        changes the ray shape or sweeps the pointer along a smooth spline.

    The Director identifies its own bus messages via nickname == NICKNAME so
    app_server can allow them through without resetting auto-play.
    """

    NICKNAME = "Director"
    _SYSTEM_PROMPT = (
        "You are an art director for a live generative art installation. "
        "Based on the conversation history and current visual theme, invent a fresh, "
        "evocative one-sentence image prompt. Be poetic and specific. "
        "Return only the prompt text — no preamble, no quotes."
    )

    def __init__(self, bus, gemini, config):
        self.bus    = bus
        self.gemini = gemini
        self.config = config
        self._enabled   = False
        self._tasks     = []
        self._get_state = None   # injected by app_server

    # ── Public API ─────────────────────────────────────────────────────────────

    def bind(self, get_state):
        """
        Inject a zero-arg callable that returns:
          (session_id, ai_mode, current_style, last_image_bgr, history)
        Called each time the Director needs current state.
        """
        self._get_state = get_state

    @property
    def enabled(self):
        return self._enabled

    def enable(self):
        if self._enabled:
            return
        self._enabled = True
        self._tasks = [
            asyncio.ensure_future(self._prompt_loop()),
            asyncio.ensure_future(self._scene_loop()),
        ]
        print("✓ Director: auto-play started")

    def disable(self):
        if not self._enabled:
            return
        self._enabled = False
        for t in self._tasks:
            t.cancel()
        self._tasks = []
        print("✓ Director: auto-play stopped")

    # ── Prompt loop ────────────────────────────────────────────────────────────

    async def _prompt_loop(self):
        await asyncio.sleep(self.config.DIRECTOR_PROMPT_INTERVAL)
        while self._enabled:
            try:
                await self._generate_and_send_prompt()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"✗ Director: prompt loop error: {e}")
            await asyncio.sleep(self.config.DIRECTOR_PROMPT_INTERVAL)

    async def _generate_and_send_prompt(self):
        session_id, ai_mode, current_style, last_image_bgr, history = self._get_state()
        if not session_id:
            return

        # Build history context (text only — no large image blobs from prior turns)
        contents = []
        for turn in (history or []):
            role = turn.get("role", "user")
            text_parts = [
                gtypes.Part(text=p["text"])
                for p in turn.get("parts", [])
                if p.get("kind") == "text" and p.get("text")
            ]
            if text_parts:
                contents.append(gtypes.Content(role=role, parts=text_parts))

        # Attach the current artwork as context for the next prompt
        style_name = current_style.get("name", "")
        current_parts = []
        if last_image_bgr is not None and last_image_bgr.any():
            ok, enc = cv2.imencode(".jpg", last_image_bgr,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                current_parts.append(
                    gtypes.Part.from_bytes(data=enc.tobytes(), mime_type="image/jpeg")
                )
        current_parts.append(gtypes.Part(
            text=f"Current style: {style_name}. Generate a fresh creative prompt."
        ))
        contents.append(gtypes.Content(role="user", parts=current_parts))

        prompt_text = ""
        try:
            response = self.gemini.client.models.generate_content(
                model=self.gemini.TEXT_MODEL,
                contents=contents,
                config=gtypes.GenerateContentConfig(
                    system_instruction=self._SYSTEM_PROMPT,
                ),
            )
            if hasattr(response, "text") and response.text:
                prompt_text = response.text.strip()
        except Exception as e:
            print(f"✗ Director: Gemini prompt generation failed: {e}")

        if not prompt_text:
            prompt_text = f"Abstract {style_name.lower()} composition with flowing forms"

        # Encode latest image so app_server can use it as reference
        image_bytes = None
        if last_image_bgr is not None and last_image_bgr.any():
            ok, enc = cv2.imencode(".jpg", last_image_bgr,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                image_bytes = enc.tobytes()

        print(f"✓ Director: prompt → '{prompt_text[:70]}…'")
        await self.bus.publish_user_message(
            session_id=session_id,
            nickname=self.NICKNAME,
            text=prompt_text,
            image_bytes=image_bytes,
            image_purpose="input",
            turn_id=uuid.uuid4().hex,
        )

    # ── Scene loop ─────────────────────────────────────────────────────────────

    async def _scene_loop(self):
        await asyncio.sleep(self.config.DIRECTOR_SCENE_INTERVAL)
        while self._enabled:
            try:
                roll = random.random()
                if roll < 0.33:
                    await self._change_shape()
                elif roll < 0.66:
                    await self._sweep_pointer()
                else:
                    await self._change_style()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"✗ Director: scene loop error: {e}")
            await asyncio.sleep(self.config.DIRECTOR_SCENE_INTERVAL)

    async def _change_shape(self):
        new_shape = random.randint(0, 4)
        print(f"✓ Director: changing ray shape → {new_shape}")
        await self.bus.publish_settings(shape=new_shape)

    async def _change_style(self):
        styles = list(self.config.STYLE.keys())
        idx = random.randrange(len(styles))
        print(f"✓ Director: changing style → {styles[idx]}")
        await self.bus.publish_settings(style_index=idx)

    async def _sweep_pointer(self):
        """Catmull-Rom spline through random points, published as USER_GESTURE."""
        session_id, _, _, _, _ = self._get_state()
        if not session_id:
            return

        n        = self.config.DIRECTOR_SPLINE_POINTS
        substeps = self.config.DIRECTOR_SPLINE_SUBSTEPS
        ctrl     = [(random.random(), random.random()) for _ in range(n)]

        path = []
        for i in range(len(ctrl) - 1):
            p0 = ctrl[max(0, i - 1)]
            p1 = ctrl[i]
            p2 = ctrl[i + 1]
            p3 = ctrl[min(len(ctrl) - 1, i + 2)]
            for j in range(substeps):
                t  = j / substeps
                t2 = t * t
                t3 = t2 * t
                x = 0.5 * (
                    2 * p1[0]
                    + (-p0[0] + p2[0]) * t
                    + (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2
                    + (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3
                )
                y = 0.5 * (
                    2 * p1[1]
                    + (-p0[1] + p2[1]) * t
                    + (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2
                    + (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3
                )
                path.append((float(np.clip(x, 0.0, 1.0)),
                             float(np.clip(y, 0.0, 1.0))))

        frame_dt = 1.0 / self.config.FPS
        print(f"✓ Director: sweeping pointer — {len(path)} points")
        for x, y in path:
            if not self._enabled:
                break
            await self.bus.publish_user_gesture(
                session_id=session_id,
                nickname=self.NICKNAME,
                x=x, y=y, z=0.0,
            )
            await asyncio.sleep(frame_dt)
