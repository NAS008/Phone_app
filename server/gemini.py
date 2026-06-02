import os
import cv2
import glob
import random
import time
import numpy as np
from google import genai
from google.genai import types
from typing import Optional, List, Dict, Any, Tuple

# Finish reasons that mean Gemini actively blocked the request — retrying won't help.
_BLOCK_REASONS = frozenset({
    "SAFETY", "RECITATION", "BLOCKLIST",
    "PROHIBITED_CONTENT", "SPII", "IMAGE_OTHER", "IMAGE_SAFETY",
})

_BLOCK_MESSAGES: Dict[str, str] = {
    "SAFETY": "This description was flagged for safety reasons. Try rephrasing your idea or describing a different scene.",
    "IMAGE_OTHER": "The AI couldn't generate this image. Try making your description more specific, or choose a different subject or style.",
    "PROHIBITED_CONTENT": "This content isn't allowed. Try a different subject or approach.",
    "RECITATION": "The request matched restricted content. Try rephrasing your description.",
    "BLOCKLIST": "Your description contains blocked terms. Try using different words.",
    "SPII": "The description may contain sensitive personal information. Try making it more general.",
    "IMAGE_SAFETY": "The image was flagged for safety. Try a different scene or subject.",
}

class GeminiBlockedError(Exception):
    """Raised when Gemini refuses to generate an image for a policy reason."""
    def __init__(self, reason: str, user_message: str):
        super().__init__(user_message)
        self.reason = reason
        self.user_message = user_message

class Gemini:
    IMAGE_INPUT_MIME = "image/jpeg"

    def __init__(
        self,
        GEMINI_API_KEY: str,
        TEXT_MODEL: str = "gemini-2.5-flash",
        IMAGE_MODEL: str = "gemini-2.5-flash-image",
        STYLE = "None",
    ):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.TEXT_MODEL = TEXT_MODEL
        self.IMAGE_MODEL = IMAGE_MODEL
        self.STYLE = STYLE["long"] if isinstance(STYLE, dict) else STYLE
        print("✓ Gemini: unified text/image model ready")

    def _extract_text(self, response) -> str:
        text = getattr(response, "text", None)
        if text:
            return text.strip()

        chunks = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                if getattr(part, "text", None):
                    chunks.append(part.text.strip())
        return "\n".join([x for x in chunks if x]).strip()

    def _extract_first_image_bytes(self, response) -> Tuple[Optional[bytes], Optional[str]]:
        """Returns (image_bytes, block_reason). block_reason is set when Gemini actively refused."""
        for candidate in getattr(response, "candidates", []) or []:
            finish = getattr(candidate, "finish_reason", None)
            finish_name = getattr(finish, "name", str(finish)) if finish is not None else "UNKNOWN"
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    return inline.data, None
            if finish_name not in ("STOP", "0", "None", "UNKNOWN"):
                print(f"✗ Gemini: candidate finish_reason={finish_name} — no image produced")
                if finish_name in _BLOCK_REASONS:
                    return None, finish_name
        return None, None

    def _bus_parts_to_sdk_parts(self, parts: List[Dict[str, Any]]) -> List[Any]:
        sdk_parts: List[Any] = []
        for part in parts or []:
            kind = part.get("kind")
            if kind == "text":
                text = (part.get("text") or "").strip()
                if text:
                    sdk_parts.append(types.Part(text=text))
            elif kind == "image":
                data = part.get("data")
                if data:
                    sdk_parts.append(types.Part.from_bytes(
                        data=data,
                        mime_type=part.get("mime_type", self.IMAGE_INPUT_MIME),
                    ))
        return sdk_parts

    def _history_to_sdk_contents(self, history: List[Dict[str, Any]]) -> List[Any]:
        contents = []
        for turn in history or []:
            role = turn.get("role", "user")
            sdk_parts = self._bus_parts_to_sdk_parts(turn.get("parts", []))
            if sdk_parts:
                contents.append(types.Content(role=role, parts=sdk_parts))
        return contents

    def _build_decision_prompt(self) -> str:
        return (
            "You are helping create an image.\n"
            "You may receive text, audio, image references, or a combination.\n"
            "Decide between exactly two actions:\n"
            "1. ASK_FOLLOWUP: if the request is too vague or missing key visual details needed to generate a strong image.\n"
            "2. GENERATE_IMAGE: if there is enough information to generate the image now.\n\n"
            "Return strict JSON only with this schema:\n"
            "{"
            "\"action\":\"ASK_FOLLOWUP\"|\"GENERATE_IMAGE\","
            "\"question\":\"string\","
            "\"prompt\":\"string\""
            "}\n\n"
            "Rules:\n"
            "- If action is ASK_FOLLOWUP, fill 'question' and leave 'prompt' as empty string.\n"
            "- If action is GENERATE_IMAGE, fill 'prompt' and leave 'question' as empty string.\n"
            "- Keep the question short and useful.\n"
            "- Make the prompt ready for image generation.\n"
            "- Use any attached image as reference if relevant.\n"
        )

    def decide(self, parts: List[Dict[str, Any]], history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, str]:
        contents = self._history_to_sdk_contents(history)

        current_parts = self._bus_parts_to_sdk_parts(parts)
        if current_parts:
            contents.append(types.Content(role="user", parts=current_parts))

        response = self.client.models.generate_content(
            model=self.TEXT_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=self._build_decision_prompt(),
                response_mime_type="application/json",
            ),
        )

        text = self._extract_text(response)
        if not text:
            raise ValueError("empty decision response")

        import json
        data = json.loads(text)

        action = data.get("action", "").strip()
        question = (data.get("question") or "").strip()
        prompt = (data.get("prompt") or "").strip()

        if action not in ("ASK_FOLLOWUP", "GENERATE_IMAGE"):
            raise ValueError(f"invalid action: {action}")

        return {
            "action": action,
            "question": question,
            "prompt": prompt,
        }

    def generate_image(self, prompt: str, parts: Optional[List[Dict[str, Any]]] = None,
                       history: Optional[List[Dict[str, Any]]] = None) -> Optional[bytes]:
        # Text-only history gives the image model conversation context without
        # re-sending large image blobs from prior turns.
        contents: List[Any] = []
        for turn in (history or []):
            role = turn.get("role", "user")
            text_parts = []
            for p in turn.get("parts", []):
                if p.get("kind") == "text":
                    text = (p.get("text") or "").strip()
                    if text:
                        text_parts.append(types.Part(text=text))
            if text_parts:
                contents.append(types.Content(role=role, parts=text_parts))

        # Current turn: generation prompt + reference image from the user's parts
        current_parts: List[Any] = [types.Part(text=prompt)]
        for part in (parts or []):
            if part.get("kind") == "image":
                data = part.get("data")
                if data:
                    current_parts.append(types.Part.from_bytes(
                        data=data,
                        mime_type=part.get("mime_type", self.IMAGE_INPUT_MIME),
                    ))
        contents.append(types.Content(role="user", parts=current_parts))

        response = self.client.models.generate_content(
            model=self.IMAGE_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"]
            ),
        )
        image_bytes, block_reason = self._extract_first_image_bytes(response)
        if block_reason:
            msg = _BLOCK_MESSAGES.get(
                block_reason,
                "The image couldn't be generated. Try rephrasing your description.",
            )
            raise GeminiBlockedError(block_reason, msg)
        return image_bytes

    def handle(self, parts: List[Dict[str, Any]], history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        decision = self.decide(parts, history=history)

        if decision["action"] == "ASK_FOLLOWUP":
            return {
                "action": "ASK_FOLLOWUP",
                "parts": [
                    {
                        "kind": "text",
                        "text": decision["question"],
                    }
                ],
            }

        image_bytes = None
        for attempt in range(2):
            try:
                image_bytes = self.generate_image(
                    prompt=decision["prompt"] + " " + self.STYLE,
                    parts=parts,
                    history=history,
                )
            except GeminiBlockedError:
                raise  # policy block — retrying won't help
            if image_bytes:
                break
            if attempt == 0:
                print("✗ Gemini: no image on attempt 1, retrying in 1s…")
                time.sleep(1)

        if not image_bytes:
            raise ValueError("image generation returned no image")

        return {
            "action": "GENERATE_IMAGE",
            "parts": [
                {
                    "kind": "image",
                    "mime_type": "image/jpeg",
                    "purpose": "output",
                    "data": image_bytes,
                }
            ],
            "prompt": decision["prompt"],
        }

    # def describe_image(self, image_path: str, instruction: str) -> str:
    #     """Image + text prompt → text response (no image output)."""
    #     try:
    #         data, mime = self._resolve_to_bytes(image_path)
    #         parts      = [{"mime_type": mime, "data": data}, instruction]
    #         response   = self.image_client.generate_content(parts)
    #         for candidate in response.candidates:
    #             for part in getattr(getattr(candidate, "content", None), "parts", []):
    #                 if getattr(part, "text", None):
    #                     return part.text.strip()
    #     except Exception as e:
    #         print(f"✗ [gemini] describe_image failed: {e}")
    #     return ""

    # def generate_text(self, prompt: str) -> str:
    #     try:
    #         response = self.text_client.models.generate_content(
    #             model=self.config.text_model,
    #             contents=prompt,
    #         )
    #         return response.text or ""
    #     except Exception as e:
    #         print(f"✗ [gemini] Text generation failed: {e}")
    #         return ""

    # def generate_prompts_for_topic(self, topic: str) -> List[str]:
    #     """Generate SD image prompts for a given topic."""
    #     instruction = (
    #         f"Generate exactly 10 short image prompts for Stable Diffusion 3.5 on the topic: {topic}.\n"
    #         "Rules:\n"
    #         "- Each prompt must be under 15 words\n"
    #         "- Each must feature ONE strong vertical or centered subject\n"
    #         "- No camera/lens terms, no style adjectives beyond lighting\n"
    #         "Return ONLY a plain numbered list 1-10, one prompt per line, no extra text."
    #     )
    #     raw = self.generate_text(instruction)
    #     if not raw.strip():
    #         print("✗ [gemini] Empty response for prompt generation")
    #         return []

    #     prompts = []
    #     for line in raw.strip().splitlines():
    #         line = line.strip()
    #         if line and line[0].isdigit():
    #             line = line.lstrip("0123456789").lstrip(". )-").strip()
    #         if line:
    #             prompts.append(line)

    #     prompts = prompts[:10]
    #     print(f"✓ [gemini] Generated {len(prompts)} prompts for '{topic}'")
    #     for i, p in enumerate(prompts, 1):
    #         print(f"   {i:02d}. {p}")
    #     return prompts
    
class GeminiImage:
    def __init__(self, IMAGE_SIZE, ARTIST_FOLDER, GEMINI_API_KEY, MODEL="gemini-2.5-flash-image"):
        self.IW = IMAGE_SIZE
        self.IH = IMAGE_SIZE
        self.paths = self._init_image_list(ARTIST_FOLDER)

        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.MODEL = MODEL
        print("✓ Gemini: image model ready")

    def _init_image_list(self, folder, extensions=("*.png", "*.jpg", "*.jpeg")):
        paths = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(folder, ext)))
        if not paths:
            print("No images found in folder!")
        paths.sort()
        return paths
   
    def load_image_bytes(self, url=None, quality=85):
        if url is None:
            image = cv2.imread(random.choice(self.paths))
        else:
            image = cv2.imread(url)
        image = cv2.resize(image, (self.IW, self.IH), interpolation=cv2.INTER_AREA)

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        ok, buffer = cv2.imencode(".jpg", image, encode_params)
        if not ok:
            raise ValueError("✗ AI: JPEG encoding failed")
        return buffer.tobytes()
       
    def generate_image(self, prompt, image_bytes=None, image_mime="image/jpeg"):
        contents = []

        if image_bytes is not None:
            instruction = (
                "Use the attached image as reference for subject, palette, "
                f"or composition. Generate a new image: {prompt}"
            )
            contents.append(instruction)
            contents.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=image_mime,
                )
            )
        else:
            contents.append(f"Generate an image only. No text. {prompt}")

        try:
            response = self.client.models.generate_content(
                model=self.MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"]
                ),
            )

            for candidate in getattr(response, "candidates", []):
                content = getattr(candidate, "content", None)
                if not content:
                    continue

                for part in getattr(content, "parts", []):
                    inline = getattr(part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        return inline.data

        except Exception as e:
            print(f"✗ Gemini: image generation failed: {e}")

        return None

    # def edit_image(
    #     self,
    #     prompt: str,
    #     image_paths: List[str | Image.Image],
    #     instruction_style: str = "edit",  # "edit" | "style" | "composite"
    # ) -> Optional[str]:
    #     """
    #     Text + image(s) → image via Gemini.

    #     instruction_style:
    #         "edit"      — modify the image based on the text prompt
    #                     e.g. prompt="change the sky to a stormy sunset"
    #         "style"     — apply the visual style of the image to the prompt subject
    #                     e.g. prompt="a cat", images=[style_ref.png]
    #         "composite" — combine subject from prompt with elements from the images
    #                     e.g. prompt="a knight standing in this landscape"
    #     """
    #     if not image_paths:
    #         print("✗ [gemini] edit_image requires at least one image")
    #         return None

    #     # Build instruction based on style
    #     if instruction_style == "edit":
    #         instruction = (
    #             f"Edit the provided image(s) as follows: {prompt}. "
    #             "Keep everything else unchanged. Output a single edited image."
    #         )
    #     elif instruction_style == "style":
    #         instruction = (
    #             f"Using the visual style, color palette, and texture of the provided image(s), "
    #             f"generate a NEW image of: {prompt}."
    #         )
    #     elif instruction_style == "composite":
    #         instruction = (
    #             f"Combine the visual content of the provided image(s) with the following: {prompt}. "
    #             "Produce a seamless, photorealistic composite image."
    #         )
    #     else:
    #         instruction = prompt

    #     # Resolve all input images to bytes
    #     parts = []
    #     for img_input in image_paths:
    #         data, mime = self._resolve_to_bytes(img_input, self.config.INPUT_FOLDER)
    #         parts.append({"mime_type": mime, "data": data})
    #     parts.append(instruction)

    #     try:
    #         response = self.image_client.generate_content(parts)
    #         for candidate in response.candidates:
    #             fr = getattr(getattr(candidate, "finish_reason", None), "name",
    #                         str(getattr(candidate, "finish_reason", "")))
    #             if fr in ("SAFETY", "3"):
    #                 print("✗ Safety block — skipping candidate")
    #                 continue
    #             for part in getattr(getattr(candidate, "content", None), "parts", []):
    #                 if getattr(part, "text", None):
    #                     print(f"Gemini note: {part.text[:120]}")
    #                 inline = getattr(part, "inline_data", None)
    #                 if inline and getattr(inline, "data", None):
    #                     return self._save_image(inline)
    #     except Exception as e:
    #         print(f"✗ Gemini edit_image error: {e}")
    #     return None
