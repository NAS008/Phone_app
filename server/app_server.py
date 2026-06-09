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
from gemini import Gemini, GeminiBlockedError
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

    current_style = list(config.STYLE.values())[0]

    gemini = Gemini(
        GEMINI_API_KEY=config.GEMINI_API_KEY,
        TEXT_MODEL=config.GEMINI_TEXT_MODEL,
        IMAGE_MODEL=config.GEMINI_IMAGE_MODEL,
        STYLE=current_style["long"],
    )

    current_session_id = ""
    joined_users = set()
    session_histories = {}
    pending_parts = {}
    ai_mode = 0
    last_generated_image_bgr = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
    _sd = None
    _ad = None

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
        joined_users.add("Director")   # Director (running on PC) is always a valid sender
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
        nonlocal current_session_id, joined_users, ai_mode, last_generated_image_bgr, current_style

        if str(session_id) != str(current_session_id) and str(session_id) != config.ADMIN_SESSION_ID:
            print(f"✗ Server: ignored message from wrong session {session_id}")
            return

        if nickname not in joined_users:
            if str(session_id) == config.ADMIN_SESSION_ID or str(session_id) == str(current_session_id):
                joined_users.add(nickname)
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
            style_suffix = current_style["short"]
            full_prompt = f"{prompt_text}, {hint}, {style_suffix}" if hint else f"{prompt_text}, {style_suffix}"
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
                await asyncio.sleep(1.0 / config.FPS)

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
            result = await loop.run_in_executor(executor, lambda: gemini.handle(effective_parts, history))
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

            # Capture old image and update immediately so any concurrent Director cycle
            # sees the new image as its starting point rather than replaying the same transition.
            prev_bgr = last_generated_image_bgr
            last_generated_image_bgr = new_bgr

            if ai_mode == 1:
                # SD journey: stream each interpolation frame to PC as it's generated.
                # Use the short style for SD — the long style exceeds the CLIP 77-token limit.
                sd_prompt = current_style.get("short") or " ".join(prompt_used.split()[:40])
                q = asyncio.Queue()

                def sd_worker():
                    try:
                        for frame in get_sd().generate_between_images(prev_bgr, new_bgr, sd_prompt):
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
                # Mode 0: send Gemini result directly.
                await bus.publish_ai_message_to_pc(
                    session_id=effective_session_id, nickname="NonCarbon Artist",
                    image_bytes=bgr_to_jpeg(new_bgr), image_mime_type="image/jpeg",
                    image_purpose="output", turn_id=turn_id,
                )

            await bus.publish_ai_message_to_phone(
                session_id=effective_session_id, nickname="NonCarbon Artist",
                text="Check this!",
                image_bytes=bgr_to_jpeg(new_bgr), image_mime_type="image/jpeg",
                image_purpose="output", turn_id=turn_id,
            )
            return

        print(f"✗ Server: unknown Gemini action {action}")

    async def on_settings(params):
        nonlocal current_session_id, ai_mode, current_style
        if "session_id" in params and str(params["session_id"]) != str(current_session_id) and str(params["session_id"]) != config.ADMIN_SESSION_ID:
            return
        if "mode" in params:
            ai_mode = int(params["mode"])
            print(f"✓ Server: ai_mode set to {ai_mode}")
        if "style_index" in params:
            idx = int(params["style_index"])
            style_values = list(config.STYLE.values())
            if 0 <= idx < len(style_values):
                current_style = style_values[idx]
                gemini.STYLE = current_style["long"]
                print(f"✓ Server: style set to '{current_style['name']}'")
    bus.on(Bus.SESSION, on_session)
    bus.on(Bus.USER_JOINED, on_user_joined)
    bus.on(Bus.USER_MESSAGE, on_user_message)
    bus.on(Bus.SETTINGS, on_settings)

    while True:
        await bus.poll()

if __name__ == '__main__':
    asyncio.run(main())
