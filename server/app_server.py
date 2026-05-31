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
# ✗ Add SD ai mode for image to image journey
# ✗ Add AD ai mode for image motion journey

import asyncio
import cv2
import numpy as np
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

def resize_image_bytes_jpeg(image_bytes, size, quality=85):
    img_bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode image bytes")
    img_bgr = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("Failed to encode resized image")
    return enc.tobytes()

async def main():
    config = Config()

    bus = Bus(config.redis_host, config.redis_port, config.redis_password, config.redis_ssl)
    await bus.connect()

    gemini = Gemini(
        GEMINI_API_KEY=config.GEMINI_API_KEY,
        TEXT_MODEL=config.GEMINI_TEXT_MODEL,
        IMAGE_MODEL=config.GEMINI_IMAGE_MODEL,
    )

    current_session_id = ""
    joined_users = set()
    session_histories = {}   # session_id -> list of {"role", "parts"} turns
    pending_parts = {}        # session_id -> image parts held across a follow-up turn

    async def on_session(session_id):
        nonlocal current_session_id, joined_users
        current_session_id = str(session_id or "")
        joined_users.clear()
        session_histories.clear()
        pending_parts.clear()
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
        nonlocal current_session_id, joined_users

        if str(session_id) != str(current_session_id) and str(session_id) != config.ADMIN_SESSION_ID:
            print(f"✗ Server: ignored message from wrong session {session_id}")
            return

        if nickname not in joined_users:
            print(f"✗ Server: ignored message from unjoined user '{nickname}'")
            return

        effective_session_id = session_id if str(session_id) == config.ADMIN_SESSION_ID else (current_session_id or session_id)

        if not parts:
            print("✗ Server: empty parts")
            return

        session_key = str(effective_session_id)
        history = session_histories.setdefault(session_key, [])

        # Merge any image parts saved from a prior ASK_FOLLOWUP so the AI still
        # has the image in context when the user replies with text only.
        saved = pending_parts.pop(session_key, [])
        effective_parts = saved + parts if saved else parts

        try:
            result = gemini.handle(effective_parts, history=history)
        except GeminiBlockedError as e:
            print(f"✗ Server: Gemini blocked ({e.reason}): {e.user_message}")
            await bus.publish_ai_message_to_pc(
                session_id=effective_session_id,
                nickname="NonCarbon Artist",
                text=e.user_message,
                turn_id=payload.get("turn_id"),
            )
            await bus.publish_ai_message_to_phone(
                session_id=effective_session_id,
                nickname="NonCarbon Artist",
                text=e.user_message,
                turn_id=payload.get("turn_id"),
            )
            return
        except Exception as e:
            print(f"✗ Server: Gemini handle failed: {e}")
            fallback = "I wasn't able to create an image right now. Please try again."
            await bus.publish_ai_message_to_pc(
                session_id=effective_session_id,
                nickname="NonCarbon Artist",
                text=fallback,
                turn_id=payload.get("turn_id"),
            )
            await bus.publish_ai_message_to_phone(
                session_id=effective_session_id,
                nickname="NonCarbon Artist",
                text=fallback,
                turn_id=payload.get("turn_id"),
            )
            return

        action = result.get("action")
        out_parts = result.get("parts", [])

        # Record the user's text in history (skip image bytes — too large to replay).
        user_text_parts = [p for p in effective_parts if p.get("kind") == "text" and p.get("text")]
        if user_text_parts:
            history.append({"role": "user", "parts": user_text_parts})

        # Trim to keep only the most recent turns.
        if len(history) > config.CONTEXT_SIZE:
            del history[: len(history) - config.CONTEXT_SIZE]

        if action == "ASK_FOLLOWUP":
            # Preserve image parts so the next reply still has the image in context.
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
                session_id=effective_session_id,
                nickname="NonCarbon Artist",
                text=followup_text,
                turn_id=payload.get("turn_id"),
            )
            await bus.publish_ai_message_to_phone(
                session_id=effective_session_id,
                nickname="NonCarbon Artist",
                text=followup_text,
                turn_id=payload.get("turn_id"),
            )
            return

        if action == "GENERATE_IMAGE":
            image_part = extract_first_image(out_parts)
            if not image_part:
                print("✗ Server: Gemini returned GENERATE_IMAGE without image")
                fallback = "I wasn't able to create an image right now. Try describing your idea differently."
                await bus.publish_ai_message_to_pc(
                    session_id=effective_session_id,
                    nickname="NonCarbon Artist",
                    text=fallback,
                    turn_id=payload.get("turn_id"),
                )
                await bus.publish_ai_message_to_phone(
                    session_id=effective_session_id,
                    nickname="NonCarbon Artist",
                    text=fallback,
                    turn_id=payload.get("turn_id"),
                )
                return

            image_bytes_out = resize_image_bytes_jpeg(
                image_bytes=image_part["data"],
                size=config.IMAGE_SIZE,
                quality=85,
            )
            prompt_used = result.get("prompt", "")
            kb = len(image_bytes_out) // 1024
            print(f"✓ Server: image generated {kb} KB — {prompt_used}")

            history.append({"role": "model", "parts": [{"kind": "text", "text": f"[Image generated: {prompt_used}]"}]})

            await bus.publish_ai_message_to_pc(
                session_id=effective_session_id,
                nickname="NonCarbon Artist",
                image_bytes=image_bytes_out,
                image_mime_type=image_part.get("mime_type", "image/jpeg"),
                image_purpose=image_part.get("purpose", "output"),
                turn_id=payload.get("turn_id"),
            )
            await bus.publish_ai_message_to_phone(
                session_id=effective_session_id,
                nickname="NonCarbon Artist",
                text="Check this!",
                image_bytes=image_bytes_out,
                image_mime_type=image_part.get("mime_type", "image/jpeg"),
                image_purpose=image_part.get("purpose", "output"),
                turn_id=payload.get("turn_id"),
            )
            return

        print(f"✗ Server: unknown Gemini action {action}")

    async def on_settings(params):
        nonlocal current_session_id
        if "session_id" in params and str(params["session_id"]) != str(current_session_id) and str(params["session_id"]) != config.ADMIN_SESSION_ID:
            return
        print("✓ Server: settings received")

    bus.on(Bus.SESSION, on_session)
    bus.on(Bus.USER_JOINED, on_user_joined)
    bus.on(Bus.USER_MESSAGE, on_user_message)
    bus.on(Bus.SETTINGS, on_settings)

    while True:
        await bus.poll()


if __name__ == '__main__':
    asyncio.run(main())
