# # # # # # #
# The       #
# First     #
# NonCarbon #
# Artist    #
# # # # # # #

# Server App uses best GPU RTX5090 to run StableDiffusion and AnimateDiff open models. Also runs Gemini
# Has no UI, has no window
# Input: Receives text and/ or image bytes
# Output: Generates images
# ----------------------------------------------------------------
# ✓ Add Gemini ai mode for image generation from text and/ or image and context
# ✓ Add SD ai mode for image to image journey
# ✓ Add AD ai mode for image motion journey
# ✓ Add auto-gen mode (ai_mode=3): idle timer triggers story-driven generation loop
# ✓ Add AnimateDiff auto-gen mode (ai_mode=4): idle timer triggers AnimateDiff loop when in mode 2

import asyncio
import concurrent.futures
import random
import cv2
import numpy as np
from PIL import Image
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'shared'))
from config import Config
from bus import Bus
from gemini import Gemini, GeminiImage, GeminiBlockedError
from sd35 import StableDiffusion, AnimateDiff

def extract_first_text(parts):
    for part in parts or []:
        if part.get("kind") == "text" and part.get("text"):
            return part["text"]
    return None

def extract_first_image(parts):
    for part in parts or []:
        if part.get("kind") == "image" and part.get("data"):
            return part
    return None

def jpeg_to_bgr(image_bytes):
    return cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

def bgr_to_jpeg(image_bgr, quality=85):
    ok, enc = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Failed to encode image")
    return enc.tobytes()

def bgr_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def pil_to_bgr(pil_img):
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def resize_image_bytes_jpeg(image_bytes, size, quality=85):
    img_bgr = jpeg_to_bgr(image_bytes)
    if img_bgr is None:
        raise ValueError("Failed to decode image bytes")
    img_bgr = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    return bgr_to_jpeg(img_bgr, quality)

async def main():
    config = Config()
    loop = asyncio.get_running_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    bus = Bus(config.redis_host, config.redis_port, config.redis_password, config.redis_ssl)
    await bus.connect()

    gemini = Gemini(
        GEMINI_API_KEY=config.GEMINI_API_KEY,
        TEXT_MODEL=config.GEMINI_TEXT_MODEL,
        IMAGE_MODEL=config.GEMINI_IMAGE_MODEL,
        STYLE = list(config.STYLE.values())[16],
    )

    current_session_id = ""
    joined_users = set()
    session_histories = {}
    pending_parts = {}
    ai_mode = 0
    last_user_parts = []
    last_generated_image_bgr = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
    _sd = None
    _ad = None
    ad_style = config.AD_STYLE[4]

    # Auto-gen state
    _last_user_message_time = None  # set on first USER_MESSAGE
    _auto_gen_task = None
    _auto_gen_cancel = [False]
    _pre_auto_gen_mode = 0  # ai_mode saved before entering auto-gen (mode 3)
    _pc_queue_size = 0

    def get_sd():
        nonlocal _sd
        if _sd is None:
            print("✓ Server: loading StableDiffusion...")
            _sd = StableDiffusion(
                SD_MODEL=config.SD_MODEL,
                IW=config.IMAGE_SIZE,
                IH=config.IMAGE_SIZE,
                INFERENCE_STEPS=config.SD_INFERENCE_STEPS,
                GUIDANCE_SCALE=config.SD_GUIDANCE_SCALE,
                SEED=config.SD_SEED,
            )
            print("✓ Server: StableDiffusion ready")
        return _sd

    def get_ad():
        nonlocal _ad
        if _ad is None:
            print("✓ Server: loading AnimateDiff...")
            _ad = AnimateDiff(
                CONTROLNET_ID=config.AD_CONTROLNET_ID,
                MOTION_ADAPTER=config.AD_MOTION_ADAPTER,
                SD_BASE=config.AD_SD_BASE,
                MOTION_LORAS=config.MOTION_LORAS,
                IW=config.IMAGE_SIZE,
                IH=config.IMAGE_SIZE,
                NUM_FRAMES=config.AD_NUM_FRAMES,
                INFERENCE_STEPS=config.AD_INFERENCE_STEPS,
                GUIDANCE_SCALE=config.AD_GUIDANCE_SCALE,
                CONTROLNET_SCALE=config.CONTROLNET_SCALE,
                SEED=config.AD_SEED,
            )
        return _ad

    # ── Auto-generation loop (ai_mode == 3) ──────────────────────────────────
    # Driven by idle timer. Generates story-based prompts, creates Gemini images,
    # runs SD journeys between them, and publishes each frame to PC + phone.
    # Every 5 cycles the style is rotated through Config.STYLE.

    async def run_auto_gen_loop(cancel):
        nonlocal last_generated_image_bgr
        session_id = current_session_id or config.ADMIN_SESSION_ID
        style_values = list(config.STYLE.values())
        cycle = 0
        prompts = []

        try:
            while not cancel[0]:
                # Rotate style every 5 cycles
                if cycle % 5 == 0:
                    gemini.STYLE = random.choice(style_values)
                    print(f"✓ Server: auto-gen cycle {cycle} — style rotated")

                # Refresh story prompts when the batch is exhausted (or on first run)
                if not prompts or (cycle > 0 and cycle % len(prompts) == 0):
                    parts_snapshot = list(last_user_parts)
                    try:
                        prompts = await loop.run_in_executor(
                            executor, lambda: gemini.generate_story_prompts(parts_snapshot)
                        )
                    except Exception as e:
                        print(f"✗ Server: auto-gen story prompts failed: {e}")
                        await asyncio.sleep(5)
                        continue
                    if not prompts:
                        print("✗ Server: auto-gen produced no prompts")
                        await asyncio.sleep(5)
                        continue

                prompt = prompts[cycle % len(prompts)]
                print(f"✓ Server: auto-gen [{cycle + 1}] — {prompt}")

                # 1. Generate Gemini image from prompt + current style
                try:
                    image_bytes = await loop.run_in_executor(
                        executor,
                        lambda p=prompt: gemini.generate_image(p + " " + gemini.STYLE),
                    )
                except GeminiBlockedError as e:
                    print(f"✗ Server: auto-gen blocked ({e.reason}): {e.user_message}")
                    cycle += 1
                    continue
                except Exception as e:
                    print(f"✗ Server: auto-gen image failed: {e}")
                    cycle += 1
                    continue

                if not image_bytes or cancel[0]:
                    cycle += 1
                    continue

                new_bgr = jpeg_to_bgr(image_bytes)
                if new_bgr is None:
                    cycle += 1
                    continue
                new_bgr = cv2.resize(new_bgr, (config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=cv2.INTER_AREA)

                # 2. SD journey from last image to new image, streaming frames to PC
                prev_bgr = last_generated_image_bgr
                q = asyncio.Queue()

                def sd_worker(pb=prev_bgr, nb=new_bgr, p=prompt, c=cancel):
                    try:
                        for frame in get_sd().generate_between_images(pb, nb, p):
                            if c[0]:
                                break
                            loop.call_soon_threadsafe(q.put_nowait, frame)
                    except Exception as exc:
                        print(f"✗ Server: auto-gen SD failed: {exc}")
                    finally:
                        loop.call_soon_threadsafe(q.put_nowait, None)

                executor.submit(sd_worker)

                while not cancel[0]:
                    frame = await q.get()
                    if frame is None:
                        break
                    while _pc_queue_size > config.AUTO_GEN_MAX_QUEUE_FRAMES and not cancel[0]:
                        await asyncio.sleep(0.5)
                    if cancel[0]:
                        break
                    await bus.publish_ai_message_to_pc(
                        session_id=session_id, nickname="NonCarbon Artist",
                        image_bytes=bgr_to_jpeg(frame), image_mime_type="image/jpeg",
                        image_purpose="output",
                    )
                    await asyncio.sleep(0)

                # 3. Publish final image to phone and update last frame
                if not cancel[0]:
                    last_generated_image_bgr = new_bgr
                    await _pub_phone(
                        session_id,
                        text=f"[{cycle + 1}] {prompt}",
                        image_bytes=bgr_to_jpeg(new_bgr),
                        image_mime_type="image/jpeg",
                        image_purpose="output",
                    )

                cycle += 1

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"✗ Server: auto-gen loop crashed: {e}")

    # ── AnimateDiff auto-generation loop (ai_mode == 4) ─────────────────────────
    # Triggered by idle timer when pre-mode was 2 (AnimateDiff).
    # Gemini generates short subjects (< 75 tokens, no style bloat),
    # AnimateDiff animates them from the last rendered frame.

    async def run_auto_gen_animatediff_loop(cancel):
        nonlocal last_generated_image_bgr, ad_style
        session_id = current_session_id or config.ADMIN_SESSION_ID
        cycle = 0
        subjects = []

        try:
            while not cancel[0]:
                if not subjects or (cycle > 0 and cycle % len(subjects) == 0):
                    parts_snapshot = list(last_user_parts)
                    try:
                        subjects = await loop.run_in_executor(
                            executor, lambda: gemini.generate_animatediff_subjects(parts_snapshot)
                        )
                    except Exception as e:
                        print(f"✗ Server: auto-gen AD subjects failed: {e}")
                        await asyncio.sleep(5)
                        continue
                    if not subjects:
                        print("✗ Server: auto-gen AD produced no subjects")
                        await asyncio.sleep(5)
                        continue

                subject = subjects[cycle % len(subjects)]
                idx = random.randrange(len(config.MOTION_LORAS))
                lora_name, weight, hint, repo = config.MOTION_LORAS[idx]
                loras = [(repo, lora_name, weight)] if repo else []
                full_prompt = f"{subject}, {hint}, {ad_style}" if hint else subject
                entry = {
                    "prompt": full_prompt,
                    "negative": "close-up, indoor, blurry, watermark, text",
                    "loras": loras,
                }
                anchor_pil = bgr_to_pil(last_generated_image_bgr)
                print(f"✓ Server: auto-gen AD [{cycle + 1}] — {full_prompt}")

                try:
                    frames = await loop.run_in_executor(
                        executor, lambda e=entry, a=anchor_pil: get_ad().generate(e, a)
                    )
                except Exception as e:
                    print(f"✗ Server: auto-gen AD failed: {e}")
                    cycle += 1
                    continue

                if cancel[0] or not frames:
                    cycle += 1
                    continue

                for pil_frame in frames:
                    if cancel[0]:
                        break
                    while _pc_queue_size > config.AUTO_GEN_MAX_QUEUE_FRAMES and not cancel[0]:
                        await asyncio.sleep(0.5)
                    if cancel[0]:
                        break
                    await bus.publish_ai_message_to_pc(
                        session_id=session_id, nickname="NonCarbon Artist",
                        image_bytes=bgr_to_jpeg(pil_to_bgr(pil_frame)), image_mime_type="image/jpeg",
                        image_purpose="output",
                    )
                    await asyncio.sleep(0)

                last_frame_bgr = pil_to_bgr(frames[-1])
                last_generated_image_bgr = last_frame_bgr

                if not cancel[0]:
                    await _pub_phone(
                        session_id,
                        text=f"[{cycle + 1}] {subject}",
                        image_bytes=bgr_to_jpeg(last_frame_bgr),
                        image_mime_type="image/jpeg",
                        image_purpose="output",
                    )

                cycle += 1

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"✗ Server: auto-gen AD loop crashed: {e}")

    def start_auto_gen():
        nonlocal _auto_gen_task, _auto_gen_cancel
        _auto_gen_cancel[0] = True
        if _auto_gen_task and not _auto_gen_task.done():
            _auto_gen_task.cancel()
        _auto_gen_cancel = [False]
        if ai_mode == 4:
            _auto_gen_task = asyncio.create_task(run_auto_gen_animatediff_loop(_auto_gen_cancel))
        else:
            _auto_gen_task = asyncio.create_task(run_auto_gen_loop(_auto_gen_cancel))

    def stop_auto_gen():
        nonlocal _auto_gen_cancel
        _auto_gen_cancel[0] = True
        if _auto_gen_task and not _auto_gen_task.done():
            _auto_gen_task.cancel()

    async def _pub_phone(session_id, **kwargs):
        """Publish to both the current QR session and the admin session so both see the message."""
        await bus.publish_ai_message_to_phone(session_id=session_id, nickname="NonCarbon Artist", **kwargs)
        if session_id != config.ADMIN_SESSION_ID:
            await bus.publish_ai_message_to_phone(session_id=config.ADMIN_SESSION_ID, nickname="NonCarbon Artist", **kwargs)

    async def idle_watcher():
        nonlocal ai_mode, _pre_auto_gen_mode, _last_user_message_time
        while True:
            await asyncio.sleep(5)
            if ai_mode in (3, 4):
                continue
            if not last_user_parts or _last_user_message_time is None:
                continue
            elapsed = loop.time() - _last_user_message_time
            if elapsed >= config.AUTO_GEN_IDLE_SECONDS:
                _pre_auto_gen_mode = ai_mode
                session_id = current_session_id or config.ADMIN_SESSION_ID
                if _pre_auto_gen_mode == 2:
                    ai_mode = 4
                    print(f"✓ Server: idle {elapsed:.0f}s → entering auto-gen AnimateDiff (ai_mode=4)")
                    notice = "Auto-generation (AnimateDiff) loop started — watch the art flow"
                else:
                    ai_mode = 3
                    print(f"✓ Server: idle {elapsed:.0f}s → entering auto-gen (ai_mode=3)")
                    notice = "Auto-generation loop started — watch the art flow"
                await _pub_phone(session_id, text=notice)
                start_auto_gen()

    # ── Event handlers ────────────────────────────────────────────────────────

    async def _publish_error(session_id, text, turn_id):
        await bus.publish_ai_message_to_pc(
            session_id=session_id, nickname="NonCarbon Artist",
            text=text, turn_id=turn_id,
        )
        await bus.publish_ai_message_to_phone(
            session_id=session_id, nickname="NonCarbon Artist",
            text=text, turn_id=turn_id,
        )

    async def on_session(session_id):
        nonlocal current_session_id, joined_users, last_generated_image_bgr
        current_session_id = str(session_id or "")
        joined_users.clear()
        session_histories.clear()
        pending_parts.clear()
        last_generated_image_bgr = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
        print(f"✓ Server: active session set to {current_session_id}")

    async def on_user_joined(session_id, nickname):
        nonlocal current_session_id, joined_users
        if str(session_id) != str(current_session_id) and str(session_id) != config.ADMIN_SESSION_ID:
            return
        if not nickname:
            return
        joined_users.add(nickname)
        print(f"✓ Server: user joined '{nickname}'")

    async def on_user_message(session_id, nickname, parts, payload):
        nonlocal current_session_id, joined_users, ai_mode, last_generated_image_bgr, last_user_parts, _last_user_message_time, ad_style

        if str(session_id) != str(current_session_id) and str(session_id) != config.ADMIN_SESSION_ID:
            print(f"✗ Server: ignored message from wrong session {session_id}")
            return

        if nickname not in joined_users:
            if str(session_id) == config.ADMIN_SESSION_ID:
                joined_users.add(nickname)  # auto-rejoin admin if session was reset
            else:
                print(f"✗ Server: ignored message from unjoined user '{nickname}'")
                return

        effective_session_id = session_id if str(session_id) == config.ADMIN_SESSION_ID else (current_session_id or session_id)

        if not parts:
            print("✗ Server: empty parts")
            return

        session_key = str(effective_session_id)
        history = session_histories.setdefault(session_key, [])
        saved = pending_parts.pop(session_key, [])
        effective_parts = saved + parts if saved else parts
        turn_id = payload.get("turn_id")

        last_user_parts = list(effective_parts)
        _last_user_message_time = loop.time()

        # Exit auto-gen on any user message, restore previous mode
        if ai_mode in (3, 4):
            stop_auto_gen()
            ai_mode = _pre_auto_gen_mode
            print(f"✓ Server: user message received → exiting auto-gen, restoring ai_mode={ai_mode}")

        # ── Mode 2: AnimateDiff ──────────────────────────────────────────────
        if ai_mode == 2:
            user_text_parts = [p for p in effective_parts if p.get("kind") == "text" and p.get("text")]
            if user_text_parts:
                history.append({"role": "user", "parts": user_text_parts})
            if len(history) > config.CONTEXT_SIZE:
                del history[: len(history) - config.CONTEXT_SIZE]

            image_part = extract_first_image(effective_parts)
            if image_part:
                anchor_bgr = jpeg_to_bgr(image_part["data"])
                if anchor_bgr is not None:
                    anchor_bgr = cv2.resize(anchor_bgr, (config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
                else:
                    anchor_bgr = last_generated_image_bgr
            else:
                anchor_bgr = last_generated_image_bgr

            prompt_text = extract_first_text(effective_parts) or "an artistic scene"
            idx = random.randrange(len(config.MOTION_LORAS))
            lora_name, weight, hint, repo = config.MOTION_LORAS[idx]
            loras = [(repo, lora_name, weight)] if repo else []
            full_prompt = f"{prompt_text}, {hint}, {ad_style}" if hint else prompt_text
            entry = {
                "prompt": full_prompt,
                "negative": "close-up, indoor, blurry, watermark, text",
                "loras": loras,
            }
            anchor_pil = bgr_to_pil(anchor_bgr)
            print(f"✓ Server: AnimateDiff — prompt='{full_prompt}', lora={lora_name}")

            try:
                frames = await loop.run_in_executor(executor, lambda: get_ad().generate(entry, anchor_pil))
            except Exception as e:
                print(f"✗ Server: AnimateDiff failed: {e}")
                await _publish_error(effective_session_id, "Animation failed. Please try again.", turn_id)
                return

            for pil_frame in frames:
                frame_bytes = bgr_to_jpeg(pil_to_bgr(pil_frame))
                await bus.publish_ai_message_to_pc(
                    session_id=effective_session_id, nickname="NonCarbon Artist",
                    image_bytes=frame_bytes, image_mime_type="image/jpeg",
                    image_purpose="output", turn_id=turn_id,
                )

            last_frame_bgr = pil_to_bgr(frames[-1])
            last_generated_image_bgr = last_frame_bgr
            await bus.publish_ai_message_to_phone(
                session_id=effective_session_id, nickname="NonCarbon Artist",
                text="Moving!",
                image_bytes=bgr_to_jpeg(last_frame_bgr), image_mime_type="image/jpeg",
                image_purpose="output", turn_id=turn_id,
            )
            return

        # ── Modes 0 and 1: Gemini first ──────────────────────────────────────
        try:
            result = gemini.handle(effective_parts, history=history)
        except GeminiBlockedError as e:
            print(f"✗ Server: Gemini blocked ({e.reason}): {e.user_message}")
            await _publish_error(effective_session_id, e.user_message, turn_id)
            return
        except Exception as e:
            print(f"✗ Server: Gemini handle failed: {e}")
            await _publish_error(effective_session_id, "I wasn't able to create an image right now. Please try again.", turn_id)
            return

        action = result.get("action")
        out_parts = result.get("parts", [])

        user_text_parts = [p for p in effective_parts if p.get("kind") == "text" and p.get("text")]
        if user_text_parts:
            history.append({"role": "user", "parts": user_text_parts})
        if len(history) > config.CONTEXT_SIZE:
            del history[: len(history) - config.CONTEXT_SIZE]

        if action == "ASK_FOLLOWUP":
            image_parts = [p for p in effective_parts if p.get("kind") == "image" and p.get("data")]
            if image_parts:
                pending_parts[session_key] = image_parts
            followup_text = extract_first_text(out_parts)
            if not followup_text:
                print("✗ Server: missing follow-up text")
                return
            history.append({"role": "model", "parts": [{"kind": "text", "text": followup_text}]})
            print(f"✓ Server: follow-up -> {followup_text}")
            await bus.publish_ai_message_to_pc(
                session_id=effective_session_id, nickname="NonCarbon Artist",
                text=followup_text, turn_id=turn_id,
            )
            await bus.publish_ai_message_to_phone(
                session_id=effective_session_id, nickname="NonCarbon Artist",
                text=followup_text, turn_id=turn_id,
            )
            return

        if action == "GENERATE_IMAGE":
            image_part = extract_first_image(out_parts)
            if not image_part:
                print("✗ Server: Gemini returned GENERATE_IMAGE without image")
                fallback = "I wasn't able to create an image right now. Try describing your idea differently."
                await _publish_error(effective_session_id, fallback, turn_id)
                return

            new_bgr = jpeg_to_bgr(image_part["data"])
            if new_bgr is None:
                await _publish_error(effective_session_id, "Failed to decode generated image.", turn_id)
                return
            new_bgr = cv2.resize(new_bgr, (config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            prompt_used = result.get("prompt", "")
            kb = len(image_part["data"]) // 1024
            print(f"✓ Server: Gemini image {kb} KB — {prompt_used}")
            history.append({"role": "model", "parts": [{"kind": "text", "text": f"[Image generated: {prompt_used}]"}]})

            if ai_mode == 1 and last_generated_image_bgr is not None:
                # SD journey: stream each interpolation frame to PC as it's generated.
                prev_bgr = last_generated_image_bgr
                q = asyncio.Queue()

                def sd_worker():
                    try:
                        for frame in get_sd().generate_between_images(prev_bgr, new_bgr, prompt_used):
                            loop.call_soon_threadsafe(q.put_nowait, frame)
                    except Exception as exc:
                        print(f"✗ Server: SD generate_between_images failed: {exc}")
                    finally:
                        loop.call_soon_threadsafe(q.put_nowait, None)

                executor.submit(sd_worker)
                frame_count = 0
                while True:
                    frame = await q.get()
                    if frame is None:
                        break
                    await bus.publish_ai_message_to_pc(
                        session_id=effective_session_id, nickname="NonCarbon Artist",
                        image_bytes=bgr_to_jpeg(frame), image_mime_type="image/jpeg",
                        image_purpose="output", turn_id=turn_id,
                    )
                    frame_count += 1
                print(f"✓ Server: SD journey published {frame_count} frames to PC")
            else:
                # Mode 0, or mode 1 without a prior image: send Gemini result directly.
                await bus.publish_ai_message_to_pc(
                    session_id=effective_session_id, nickname="NonCarbon Artist",
                    image_bytes=bgr_to_jpeg(new_bgr), image_mime_type="image/jpeg",
                    image_purpose="output", turn_id=turn_id,
                )

            last_generated_image_bgr = new_bgr

            await bus.publish_ai_message_to_phone(
                session_id=effective_session_id, nickname="NonCarbon Artist",
                text="Check this!",
                image_bytes=bgr_to_jpeg(new_bgr), image_mime_type="image/jpeg",
                image_purpose="output", turn_id=turn_id,
            )
            return

        print(f"✗ Server: unknown Gemini action {action}")

    async def on_settings(params):
        nonlocal current_session_id, ai_mode, _pc_queue_size
        if "session_id" in params and str(params["session_id"]) != str(current_session_id) and str(params["session_id"]) != config.ADMIN_SESSION_ID:
            return
        if "pc_queue_size" in params:
            _pc_queue_size = int(params["pc_queue_size"])
        if "mode" in params:
            ai_mode = int(params["mode"])
            print(f"✓ Server: ai_mode set to {ai_mode}")
        if "style_index" in params:
            idx = int(params["style_index"])
            style_values = list(config.STYLE.values())
            if 0 <= idx < len(style_values):
                gemini.STYLE = style_values[idx]
                print(f"✓ Server: style set to index {idx} ({list(config.STYLE.keys())[idx]})")

    bus.on(Bus.SESSION, on_session)
    bus.on(Bus.USER_JOINED, on_user_joined)
    bus.on(Bus.USER_MESSAGE, on_user_message)
    bus.on(Bus.SETTINGS, on_settings)

    asyncio.create_task(idle_watcher())

    while True:
        await bus.poll()


if __name__ == '__main__':
    asyncio.run(main())
