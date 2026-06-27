import json
import re
import math
import cv2
import time
import hashlib
import numpy as np
import torch
import spandrel
from PIL import Image
from diffusers import StableDiffusion3Pipeline, AnimateDiffSparseControlNetPipeline, DPMSolverMultistepScheduler
from diffusers.models import MotionAdapter, SparseControlNetModel
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

        self.brand_context: Optional[BrandContext] = None

    def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
        last_exc = None
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.TEXT_MODEL,
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
            text_snippets = []
            for part in getattr(content, "parts", []) or []:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    return inline.data, None
                if getattr(part, "text", None):
                    text_snippets.append(part.text.strip()[:80])
            if finish_name not in ("STOP", "0", "None", "UNKNOWN"):
                print(f"✗ Gemini: candidate finish_reason={finish_name} — no image produced")
                if finish_name in _BLOCK_REASONS:
                    return None, finish_name
            elif text_snippets:
                print(f"✗ Gemini: text-only response (finish={finish_name}): {' | '.join(text_snippets)}")
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

        # Current turn: reference images first so Gemini weights them as the
        # primary spatial/visual constraint before reading the text prompt.
        current_parts: List[Any] = []
        for part in (parts or []):
            if part.get("kind") == "image":
                data = part.get("data")
                if data:
                    current_parts.append(types.Part.from_bytes(
                        data=data,
                        mime_type=part.get("mime_type", self.IMAGE_INPUT_MIME),
                    ))
        current_parts.append(types.Part(text=prompt))
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

    def handle(self, parts: List[Dict[str, Any]], history: Optional[List[Dict[str, Any]]] = None, force_generate: bool = False) -> Dict[str, Any]:
        if force_generate:
            text_parts = [p for p in parts if p.get("kind") == "text" and p.get("text")]
            prompt = " ".join(p["text"] for p in text_parts).strip() or "generate an image"
        else:
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

            prompt = decision["prompt"]

        # force_generate is Director auto-gen — each prompt is independent, so
        # passing accumulated chat history confuses the image model into replying
        # with text ("[Image generated: …]") rather than producing an image.
        gen_history = None if force_generate else history

        image_bytes = None
        for attempt in range(3):
            try:
                image_bytes = self.generate_image(
                    prompt=prompt + " " + self.STYLE,
                    parts=parts,
                    history=gen_history,
                )
            except GeminiBlockedError:
                raise  # policy block — retrying won't help
            if image_bytes:
                break
            wait = 2 ** attempt  # 1 s, 2 s
            print(f"✗ Gemini: no image on attempt {attempt + 1}, retrying in {wait}s…")
            time.sleep(wait)

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
            "prompt": prompt,
        }
    
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

    def generate_themes(self, topic: str) -> List[str]:
        """Ask Gemini to generate 10 short image-generation prompts about a topic."""
        prompt = (
            f"Generate exactly 10 short, vivid image-generation prompts inspired by this theme: '{topic}'.\n"
            "Each prompt must be 4-10 words, evocative, and suitable for a large-scale art installation.\n"
            "Return only a JSON array of 10 strings, nothing else.\n"
            "Example format: [\"colossal shell on the beach\", \"jellyfish beneath a hot air balloon\"]"
        )
        response = self.client.models.generate_content(
            model=self.TEXT_MODEL,
            contents=[prompt],
        )
        text = self._extract_text(response)
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if not match:
            raise ValueError(f"Gemini returned no JSON array for themes: {text[:200]}")
        themes = json.loads(match.group())
        if not isinstance(themes, list) or not themes:
            raise ValueError(f"Gemini returned invalid theme list: {themes}")
        return [str(t) for t in themes[:10]]

    def generate_subjects(self, theme: str, count: int = 5) -> Dict[str, List[str]]:
        """From a theme, invent paired subjects for the box installation.

        Returns {"main": [...], "support": [...]} with `count` ultradetailed
        descriptions each. MAIN subjects sit in the bottom half and are pulled
        upward by the SUPPORT subjects in the top half, so the pairing plays with
        weight vs. lightness (e.g. theme 'coral reef' -> main: living corals,
        support: a bloated balloon of plastic-bottle trash dragging them up)."""
        prompt = (
            "You invent subjects for a surreal fashion-editorial photo set inside a box-like chamber.\n"
            f"THEME: '{theme}'.\n\n"
            "Two roles:\n"
            "- MAIN SUBJECT: sits in the bottom half, floating just above the floor, pulled upward by thin "
            "ropes. It is heavy, tactile, physically real and rooted in the theme.\n"
            "- SUPPORT SUBJECT: fills the top half, crushed against the ceiling as if trying to escape "
            "upward. It adds drama and tension by pulling the main subject up — play with weight vs lightness, "
            "or an ironic counterpart to the theme (e.g. theme 'coral reef' -> support: a bloated balloon of "
            "plastic-bottle trash; theme 'harvest' -> support: an impossibly heavy iron anvil dragged upward).\n\n"
            f"Generate exactly {count} MAIN subjects and {count} SUPPORT subjects, all inspired by the theme.\n"
            "Each description must be ONE long, ultradetailed sentence (40-70 words) specifying textures, "
            "patterns, materials, colors, surface wear and accessories — the way a macro studio photograph "
            "would reveal them. MAIN descriptions: name the subject, then describe its physical surfaces. "
            "SUPPORT descriptions: describe an inflated or suspended mass deforming against the box walls with "
            "ropes or chains descending from its base.\n\n"
            "Return strict JSON only, no markdown, with this schema:\n"
            "{\"main\": [\"...\"], \"support\": [\"...\"]}"
        )
        response = self.client.models.generate_content(
            model=self.TEXT_MODEL,
            contents=[prompt],
        )
        text = self._extract_text(response)
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            raise ValueError(f"Gemini returned no JSON object for subjects: {text[:200]}")
        data = json.loads(match.group())
        main = [str(s) for s in data.get("main", []) if str(s).strip()]
        support = [str(s) for s in data.get("support", []) if str(s).strip()]
        if not main or not support:
            raise ValueError(f"Gemini returned incomplete subjects: main={len(main)} support={len(support)}")
        return {"main": main[:count], "support": support[:count]}

    def generate_motion(self, main_subject: str, support_subject: str) -> str:
        """Author a short image-to-video motion prompt (for Wan) describing how
        to ANIMATE a generated still in place. The scene is a MAIN subject at the
        bottom, suspended by thin ropes from an inflated SUPPORT subject above.

        The clip should show the MAIN subject's own natural locomotion (a
        four-legged animal kicking/paddling its legs, a fish or whale undulating
        its fins and tail, a bird beating its wings, a flower swaying) while the
        SUPPORT subject drifts slowly upward and the taut ropes lift the MAIN
        subject gently up after it. The camera stays static."""
        prompt = (
            "You write a short motion description for an image-to-video model (Wan).\n"
            "The still shows a MAIN subject at the bottom of the frame, suspended by thin "
            "ropes from an inflated SUPPORT subject pressed against the top.\n"
            f"MAIN SUBJECT: {main_subject}\n"
            f"SUPPORT SUBJECT: {support_subject}\n\n"
            "Describe ONLY the motion in a 5-second clip, in 2-3 plain sentences:\n"
            "- The MAIN subject performs its own natural locomotion in place — pick what fits "
            "the creature (legs walking/paddling, fins and tail undulating, wings beating, "
            "petals and stems swaying).\n"
            "- The SUPPORT subject slowly rises and floats upward; the thin ropes pull taut and "
            "lift the MAIN subject gently upward with it.\n"
            "- The camera is static; motion is smooth, continuous and subtle.\n\n"
            "Output only the motion description — no preamble, no list, no quotes."
        )
        response = self.client.models.generate_content(
            model=self.TEXT_MODEL,
            contents=[prompt],
        )
        text = self._extract_text(response)
        if not text:
            raise ValueError("Gemini returned no motion prompt")
        return text.strip()

class StableDiffusion:
    def __init__(self, SD_MODEL, IW, IH, INFERENCE_STEPS=12, GUIDANCE_SCALE=3.5, SEED=80367253):

        self.IW = IW
        self.IH = IH
        self.INFERENCE_STEPS = INFERENCE_STEPS
        self.GUIDANCE_SCALE = GUIDANCE_SCALE
        self.SEED = SEED
        self.NEGATIVE = "frame, wooden frame, canvas frame, picture frame, painting frame, text, watermark"
        self.DEVICE = "cuda"

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            SD_MODEL,
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True,
        ).to(self.DEVICE)
        self.pipe.transformer.to(memory_format=torch.channels_last)
        self.pipe.vae.to(memory_format=torch.channels_last)

    def encode_prompt(self, prompt):
        return self.pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, prompt_3=prompt,
            negative_prompt=self.NEGATIVE,
            negative_prompt_2=self.NEGATIVE,
            negative_prompt_3=self.NEGATIVE,
            do_classifier_free_guidance=True,
            device=self.DEVICE,
        )

    def prepare_reference(self, image_bgr):
        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).resize(
            (self.IW, self.IH), Image.LANCZOS
        )
        pixel_tensor = self.pipe.image_processor.preprocess(pil).to(
            device=self.DEVICE, dtype=torch.float16
        )
        with torch.no_grad():
            latents = self.pipe.vae.encode(pixel_tensor).latent_dist.sample()
            latents = latents * self.pipe.vae.config.scaling_factor
        return latents

    def decode_latents(self, latents):
        with torch.no_grad():
            latents = latents / self.pipe.vae.config.scaling_factor
            image_tensor = self.pipe.vae.decode(latents, return_dict=False)[0]
        image_np = self.pipe.image_processor.postprocess(image_tensor, output_type="np")[0]
        image_rgb = (image_np * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    def slerp(self, v0, v1, t, dot_threshold=0.9995):
        dtype = v0.dtype
        v0 = v0.float()
        v1 = v1.float()
        v0_norm = torch.nn.functional.normalize(v0, dim=-1)
        v1_norm = torch.nn.functional.normalize(v1, dim=-1)
        dot = (v0_norm * v1_norm).sum(dim=-1, keepdim=True).clamp(-1, 1)
        is_close = dot.abs() > dot_threshold
        theta = torch.acos(dot.clamp(-1 + 1e-6, 1 - 1e-6))
        sin_theta = torch.sin(theta).clamp(min=1e-6)
        s0 = torch.sin((1.0 - t) * theta) / sin_theta
        s1 = torch.sin(t * theta) / sin_theta
        slerp_val = s0 * v0 + s1 * v1
        lerp_val = (1.0 - t) * v0 + t * v1
        return torch.where(is_close, lerp_val, slerp_val).to(dtype)

    def denoise_from_sigma(self, base_latents, noise, sigma_level, embeds, pooled):
        self.pipe.scheduler.set_timesteps(self.INFERENCE_STEPS, device=self.DEVICE)
        all_timesteps = self.pipe.scheduler.timesteps.clone()
        all_sigmas = self.pipe.scheduler.sigmas.clone()

        start_idx = int(torch.argmin(torch.abs(all_sigmas[:-1] - sigma_level)).item())
        timesteps = all_timesteps[start_idx:]
        t_start = all_timesteps[start_idx]
        t_tensor = t_start.reshape(1).to(device=self.DEVICE, dtype=torch.float32)

        self.pipe.scheduler.sigmas = all_sigmas[start_idx:]
        self.pipe.scheduler.timesteps = timesteps
        self.pipe.scheduler._step_index = None

        latents = self.pipe.scheduler.scale_noise(base_latents, t_tensor, noise)

        for t in timesteps:
            if isinstance(t, torch.Tensor):
                timestep = t.unsqueeze(0).to(self.DEVICE) if t.dim() == 0 else t.to(self.DEVICE)
            else:
                timestep = torch.tensor([t], dtype=torch.long, device=self.DEVICE)

            latent_model_input = torch.cat([latents] * 2, dim=0)
            timestep_input = torch.cat([timestep] * 2, dim=0)

            with torch.no_grad():
                noise_pred = self.pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep_input,
                    encoder_hidden_states=embeds,
                    pooled_projections=pooled,
                    return_dict=False,
                )[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
            latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        return latents
   
    def generate_from_text(self, prompt: str) -> np.ndarray:
        """Text-to-image with SD3.5. Returns BGR numpy array."""
        generator = torch.Generator(device=self.DEVICE).manual_seed(self.SEED)
        result = self.pipe(
            prompt=prompt,
            negative_prompt=self.NEGATIVE,
            num_inference_steps=self.INFERENCE_STEPS,
            guidance_scale=self.GUIDANCE_SCALE,
            generator=generator,
            width=self.IW,
            height=self.IH,
        )
        pil_img = result.images[0]
        rgb = np.array(pil_img.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _noise_for(self, image_bgr, like):
        # Seed derived from the image content: the same endpoint always maps to
        # the same noise, so consecutive journeys (A->B, B->C) chain continuously
        # while every transition gets its own midpoint character.
        seed = int.from_bytes(hashlib.sha1(image_bgr.tobytes()).digest()[:4], "big")
        gen = torch.Generator(device=self.DEVICE).manual_seed(seed)
        return torch.randn(like.shape, generator=gen, device=like.device, dtype=like.dtype)

    # prompt_a/prompt_b are accepted for call-site compatibility; only `prompt`
    # steers the journey.
    def generate_between_images(self, image_a_bgr, image_b_bgr, prompt="", prompt_a=None, prompt_b=None,
                                interpolation_range=(0.0, 1.0)):
        latents_a = self.prepare_reference(image_a_bgr)
        latents_b = self.prepare_reference(image_b_bgr)

        prompt_embeds, negative_embeds, pooled_embeds, negative_pooled = self.encode_prompt(prompt)
        embeds = torch.cat([negative_embeds, prompt_embeds], dim=0)
        pooled = torch.cat([negative_pooled, pooled_embeds], dim=0)

        noise_a = self._noise_for(image_a_bgr, latents_a)
        noise_b = self._noise_for(image_b_bgr, latents_b)

        sigma_min = 0.60
        sigma_max = 1.0

        t_lo, t_hi = interpolation_range
        for i in range(1, self.INFERENCE_STEPS + 1):
            t = i / (self.INFERENCE_STEPS + 1)
            alpha_unit = 0.5 - 0.5 * np.cos(np.pi * t)   # cosine ease within the step range
            alpha = t_lo + (t_hi - t_lo) * alpha_unit      # mapped into interpolation_range
            sigma = sigma_min + (sigma_max - sigma_min) * (1.0 - (2.0 * alpha_unit - 1.0) ** 2)
            base_latents = self.slerp(latents_a, latents_b, alpha)
            noise = self.slerp(noise_a, noise_b, alpha)
            latents = self.denoise_from_sigma(base_latents, noise, sigma, embeds, pooled)
            frame = self.decode_latents(latents)

            yield frame

        if t_hi >= 1.0:
            yield image_b_bgr

    def generate_journey(self, image_a_bgr, image_b_bgr, prompt="",
                         interpolation_range=(0.0, 1.0),
                         sigma_min=0.15, sigma_max=0.35):
        # Low sigma keeps each frame anchored to the slerp'd reference (img2img at ~15-35%
        # strength) instead of regenerating from near-pure noise like generate_between_images.
        # This prevents the journey from detouring through unrelated content.
        latents_a = self.prepare_reference(image_a_bgr)
        latents_b = self.prepare_reference(image_b_bgr)

        prompt_embeds, negative_embeds, pooled_embeds, negative_pooled = self.encode_prompt(prompt)
        embeds = torch.cat([negative_embeds, prompt_embeds], dim=0)
        pooled = torch.cat([negative_pooled, pooled_embeds], dim=0)

        noise_a = self._noise_for(image_a_bgr, latents_a)
        noise_b = self._noise_for(image_b_bgr, latents_b)

        t_lo, t_hi = interpolation_range
        for i in range(1, self.INFERENCE_STEPS + 1):
            t = i / (self.INFERENCE_STEPS + 1)
            alpha_unit = 0.5 - 0.5 * np.cos(np.pi * t)
            alpha = t_lo + (t_hi - t_lo) * alpha_unit
            sigma = sigma_min + (sigma_max - sigma_min) * (1.0 - (2.0 * alpha_unit - 1.0) ** 2)
            base_latents = self.slerp(latents_a, latents_b, alpha)
            noise = self.slerp(noise_a, noise_b, alpha)
            latents = self.denoise_from_sigma(base_latents, noise, sigma, embeds, pooled)
            yield self.decode_latents(latents)

        if t_hi >= 1.0:
            yield image_b_bgr

class AnimateDiff:
    def __init__(self, CONTROLNET_ID, MOTION_ADAPTER, SD_BASE, MOTION_LORAS, IW, IH, NUM_FRAMES=16, INFERENCE_STEPS=10, GUIDANCE_SCALE=7.5, CONTROLNET_SCALE=0.5, SEED=80367253):
        self.MOTION_LORAS = MOTION_LORAS
        self.IW = IW
        self.IH = IH
        self.NUM_FRAMES = NUM_FRAMES
        self.INFERENCE_STEPS = INFERENCE_STEPS
        self.GUIDANCE_SCALE = GUIDANCE_SCALE
        self.CONTROLNET_SCALE = CONTROLNET_SCALE
        self.DEVICE = "cuda"

        print("Loading SparseCtrl RGB ControlNet...")
        controlnet = SparseControlNetModel.from_pretrained(
            CONTROLNET_ID, torch_dtype=torch.float16)
        print("Loading motion adapter...")
        adapter = MotionAdapter.from_pretrained(
            MOTION_ADAPTER, torch_dtype=torch.float16)
        print("Loading SD1.5 base + building pipeline...")
        self.pipe = AnimateDiffSparseControlNetPipeline.from_pretrained(
            SD_BASE, motion_adapter=adapter,
            controlnet=controlnet, torch_dtype=torch.float16,
        ).to(self.DEVICE)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            beta_schedule="scaled_linear",
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        self.pipe.vae.enable_slicing()
        self.pipe.set_progress_bar_config(disable=True)
        self.generator = torch.Generator(device=self.DEVICE).manual_seed(SEED)
        self._active_loras = []
        print("✓ AnimateDiff ready")

    def _load_loras(self, loras):
        if self._active_loras:
            self.pipe.unload_lora_weights()
            self._active_loras = []

        if not loras:
            return

        for repo_id, adapter_name, _ in loras:
            self.pipe.load_lora_weights(
                repo_id,
                weight_name="diffusion_pytorch_model.safetensors",
                adapter_name=adapter_name,
            )

        names   = [n for _, n, _ in loras]
        weights = [w for _, _, w in loras]
        self.pipe.set_adapters(names, adapter_weights=weights)
        self._active_loras = loras
        print(f"  [LoRA] active: { {n: w for _, n, w in loras} }", flush=True)

    def _pil_to_bgr(self, img):
        return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

    def _bgr_to_pil(self, bgr):
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def generate_simple(self, subject, motion_lora_id, anchor, style="minimalist", negative="close-up, indoor, blurry, watermark, text"):
        lora_name, weight, hint, repo = self.MOTION_LORAS[motion_lora_id]

        loras = []
        if repo is not None:
            loras = [(repo, lora_name, weight)]

        prompt = f"{subject}, {hint}, {style}"
        prompt = ", ".join(part for part in [subject, hint, style] if part)

        self._load_loras(loras)

        anchor_pil = self._bgr_to_pil(anchor)

        return self.pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_frames=self.NUM_FRAMES,
            guidance_scale=self.GUIDANCE_SCALE,
            num_inference_steps=self.INFERENCE_STEPS,
            generator=self.generator,
            width=self.IW, height=self.IH,
            conditioning_frames=[anchor_pil],
            controlnet_frame_indices=[0],
            controlnet_conditioning_scale=self.CONTROLNET_SCALE,
        ).frames[0]
    
    def generate(self, entry, anchor):
        self._load_loras(entry["loras"])
        return self.pipe(
            prompt=entry["prompt"],
            negative_prompt=entry["negative"],
            num_frames=self.NUM_FRAMES,
            guidance_scale=self.GUIDANCE_SCALE,
            num_inference_steps=self.INFERENCE_STEPS,
            generator=self.generator,
            width=self.IW, height=self.IH,
            conditioning_frames=[anchor],
            controlnet_frame_indices=[0],
            controlnet_conditioning_scale=self.CONTROLNET_SCALE,
        ).frames[0]

class SuperResolution:
    def __init__(self, folder):
        self.model = (
            spandrel.ModelLoader()
            .load_from_file(f"{folder}/RealESRGAN_x4plus.pth")
            .eval()
            .cuda()
        )
        print(f"✓ [ai] Super resolution model ready | scale: {self.model.scale}x")

    def upscale(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).cuda()
        with torch.no_grad():
            result = self.model(tensor).squeeze(0).clamp(0, 1)
        out = result.permute(1, 2, 0).mul(255.0).byte().cpu().numpy()
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

class FramePack:
    def __init__(self, transformer_path, base_path, image_w, image_h,
                 inference_steps=10, guidance_scale=6.0, true_cfg_scale=1.0,
                 latent_window=9, seed=42,
                 siglip_path="google/siglip-so400m-patch14-384",
                 cpu_offload=True):
        from diffusers import (
            HunyuanVideoFramepackPipeline,
            HunyuanVideoFramepackTransformer3DModel,
        )
        from transformers import SiglipVisionModel, SiglipImageProcessor

        self.IW = image_w
        self.IH = image_h
        self.INFERENCE_STEPS = inference_steps
        self.GUIDANCE_SCALE = guidance_scale
        self.TRUE_CFG_SCALE = true_cfg_scale
        self.LATENT_WINDOW = latent_window
        self.DEVICE = "cuda"

        print("Loading FramePack transformer...")
        transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
            transformer_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

        # SigLIP is not part of the base HunyuanVideo model; load separately.
        # Downloads ~1.1 GB on first run, then uses the HuggingFace cache.
        print(f"Loading SigLIP image encoder ({siglip_path})...")
        image_encoder   = SiglipVisionModel.from_pretrained(siglip_path, torch_dtype=torch.bfloat16)
        feature_extractor = SiglipImageProcessor.from_pretrained(siglip_path)

        print("Loading FramePack pipeline (HunyuanVideo base)...")
        self.pipe = HunyuanVideoFramepackPipeline.from_pretrained(
            base_path,
            transformer=transformer,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        if cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        self.pipe.set_progress_bar_config(disable=True)
        self._seed = seed
        print("✓ FramePack ready")

    def _bgr_to_pil(self, bgr):
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize(
            (self.IW, self.IH), Image.LANCZOS
        )

    def _pil_to_bgr(self, pil):
        return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    def generate(self, image_bgr, prompt="", num_frames=9, last_frame_bgr=None):
        """Predict num_frames of continuation from image_bgr.
        Returns a list of BGR numpy arrays (the generated frames)."""
        pil       = self._bgr_to_pil(image_bgr)
        last_pil  = self._bgr_to_pil(last_frame_bgr) if last_frame_bgr is not None else None
        generator = torch.Generator("cpu").manual_seed(self._seed)

        with torch.inference_mode():
            result = self.pipe(
                image=pil,
                last_image=last_pil,
                prompt=prompt,
                height=self.IH,
                width=self.IW,
                num_frames=num_frames,
                latent_window_size=self.LATENT_WINDOW,
                num_inference_steps=self.INFERENCE_STEPS,
                guidance_scale=self.GUIDANCE_SCALE,
                true_cfg_scale=self.TRUE_CFG_SCALE,
                generator=generator,
            )
        return [self._pil_to_bgr(f) for f in result.frames[0]]

class Wan:
    """Wan2.2 TI2V-5B image-to-video / first-last-frame-to-video.

    Animates a single still (I2V) driven by a text motion prompt, or morphs
    between two stills (FLF2V) when a last frame is supplied. BGR numpy in/out
    to match the rest of ai.py (StableDiffusion, FramePack, FILM, OpticalFlow).

    The model needs height/width on a fixed grid (vae spatial factor × patch
    size); arbitrary input sizes are snapped down to the nearest valid multiple,
    so callers can hand in whatever resolution their pipeline uses."""

    NEGATIVE = ("blurry, low quality, distorted, deformed, extra limbs, "
                "duplicate face, text, watermark, jpeg artifacts, flickering")

    def __init__(self, MODEL_ID="Wan-AI/Wan2.2-TI2V-5B-Diffusers", SIZE=512,
                 INFERENCE_STEPS=25, GUIDANCE_SCALE=5.5, NUM_FRAMES=17,
                 FPS=24, SEED=42, negative=None, OFFLOAD=True, QUANTIZE=None):
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        from huggingface_hub import snapshot_download

        self.SIZE = SIZE
        self.INFERENCE_STEPS = INFERENCE_STEPS
        self.GUIDANCE_SCALE = GUIDANCE_SCALE
        self.NUM_FRAMES = NUM_FRAMES
        self.FPS = FPS
        self.SEED = SEED
        if negative is not None:
            self.NEGATIVE = negative
        self.DEVICE = "cuda"
        self.quantized = bool(QUANTIZE)

        print(f"Loading {MODEL_ID}...")
        local_dir = snapshot_download(MODEL_ID)
        vae = AutoencoderKLWan.from_pretrained(
            local_dir, subfolder="vae", torch_dtype=torch.float32
        )

        # int8-quantize the 14B transformer (the only component large enough to
        # matter: ~28 GB bf16 → ~16 GB int8). VAE/text/image encoders stay full
        # precision — they're small and quality-critical. bitsandbytes places the
        # quantized weights on GPU itself, so the pipeline must NOT be .to("cuda")'d
        # later; the cpu-offload path below handles every component correctly.
        transformer = None
        if self.quantized:
            from diffusers import BitsAndBytesConfig, WanTransformer3DModel
            qcfg = BitsAndBytesConfig(load_in_8bit=True)
            transformer = WanTransformer3DModel.from_pretrained(
                local_dir, subfolder="transformer",
                quantization_config=qcfg, torch_dtype=torch.bfloat16,
            )
            print("✓ Wan: transformer quantized to int8 (bitsandbytes)")

        # FLF2V-14B needs CLIPVisionModel (not CLIPVisionModelWithProjection).
        # Loading it explicitly prevents the pipeline from auto-loading the wrong
        # variant, which silently breaks last_image conditioning.
        extra = {"transformer": transformer} if transformer is not None else {}
        try:
            from transformers import CLIPVisionModel
            image_encoder = CLIPVisionModel.from_pretrained(
                local_dir, subfolder="image_encoder", torch_dtype=torch.float32
            )
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                local_dir, vae=vae, image_encoder=image_encoder,
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **extra,
            )
            print("✓ Wan: CLIPVisionModel image encoder loaded")
        except Exception as _e:
            print(f"• Wan: image_encoder subfolder not found ({_e}), using pipeline default")
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                local_dir, vae=vae, torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True, **extra,
            )
        # Keep Wan's ~10 GB of weights OUT of VRAM while idle. It only fires
        # occasionally (on 'a'), whereas SD + upscale + interpolate run
        # continuously and need the VRAM. On Windows the driver silently spills
        # overflow into shared system memory instead of raising OutOfMemoryError,
        # so a "successful" .to("cuda") can still thrash the whole pipeline to a
        # crawl — hence offload is the default, with full residency opt-in.
        if self.quantized:
            # bnb already pinned the int8 transformer to GPU; .to("cuda") on the
            # whole pipe would raise. cpu-offload moves the other components and
            # keeps the transformer resident across the denoising loop (it only
            # offloads between distinct components), so denoising runs full-speed.
            self.pipe.enable_model_cpu_offload()
            print("✓ Wan: model CPU offload (int8 transformer GPU-resident)")
        elif OFFLOAD:
            self.pipe.enable_model_cpu_offload()
            print("✓ Wan: model CPU offload (weights stream to GPU on demand)")
        else:
            try:
                self.pipe.to(self.DEVICE)
                print("✓ Wan: full GPU residency")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.pipe.enable_model_cpu_offload()
                print("✓ Wan: model CPU offload (OOM fallback)")

        try:
            self.pipe.transformer.set_attention_backend("sage")
            print("✓ Wan: attention backend SageAttention")
        except Exception:
            print("• Wan: attention backend default SDPA")

        # TaylorSeer: replaces TeaCache in diffusers ≥0.33 — predicts transformer
        # activations via Taylor series, skipping full compute on cached steps (~1.5–2×).
        try:
            from diffusers import TaylorSeerCacheConfig
            self.pipe.transformer.enable_cache(TaylorSeerCacheConfig())
            print("✓ Wan: TaylorSeer cache enabled")
        except Exception:
            print("• Wan: activation cache not available")

        self.pipe.vae.enable_slicing()   # decode temporal frames one at a time
        self.pipe.vae.enable_tiling()    # decode spatial dims in tiles — essential at ≥704px
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # torch.compile fuses CUDA kernels — 1.5–2× faster after first-run compilation.
        # First call takes ~5 min to compile; subsequent calls use the cache.
        # Skip it on the int8 transformer: bnb's Linear8bitLt graph-breaks under
        # compile and the int8 matmul kernels are already fused.
        if self.quantized:
            print("• Wan: torch.compile skipped (int8 transformer)")
        else:
            try:
                self.pipe.transformer = torch.compile(
                    self.pipe.transformer, mode="default", fullgraph=False
                )
                print("✓ Wan: torch.compile enabled (first run compiles ~2 min)")
            except Exception as _e:
                print(f"• Wan: torch.compile skipped ({_e})")

        # Frames must be sized on this grid; snap any request down to it.
        self._mod = self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]
        self.size = SIZE // self._mod * self._mod
        if self.size != SIZE:
            print(f"⚠ Wan: WAN_SIZE={SIZE} snapped to {self.size} (grid={self._mod}px)")
        print(f"✓ Wan ready | grid {self._mod}px, default {self.size}×{self.size}")

    def _snap(self, value):
        return max(self._mod, int(value) // self._mod * self._mod)

    def _bgr_to_pil(self, bgr, w, h):
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize(
            (w, h), Image.LANCZOS
        )

    def _pil_to_bgr(self, pil):
        return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    def generate(self, image_bgr, prompt="", last_image_bgr=None,
                 num_frames=None, width=None, height=None, seed=None):
        """Animate image_bgr with a text motion prompt. Pass last_image_bgr to
        morph toward a second still (FLF2V). Returns a list of BGR frames.
        Width/height default to the configured SIZE (not the input image size),
        so large Gemini images are automatically downscaled to the safe range."""
        num_frames = self.NUM_FRAMES if num_frames is None else num_frames
        seed = self.SEED if seed is None else seed
        w = self._snap(width  if width  is not None else self.size)
        h = self._snap(height if height is not None else self.size)

        mode = "FLF2V" if last_image_bgr is not None else "I2V"
        print(f"• Wan [{mode}]: {w}×{h} | {num_frames}f | {self.INFERENCE_STEPS} steps | seed={seed}")

        first = self._bgr_to_pil(image_bgr, w, h)
        kwargs = dict(
            image=first, prompt=prompt, negative_prompt=self.NEGATIVE,
            height=h, width=w, num_frames=num_frames,
            num_inference_steps=self.INFERENCE_STEPS,
            guidance_scale=self.GUIDANCE_SCALE, output_type="pil",
            generator=torch.Generator(self.DEVICE).manual_seed(seed),
        )
        if last_image_bgr is not None:
            kwargs["last_image"] = self._bgr_to_pil(last_image_bgr, w, h)

        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                result = self.pipe(**kwargs)
            frames = result.frames[0]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"✗ Wan: OOM at {w}×{h} × {num_frames}f — try reducing WAN_SIZE or WAN_NUM_FRAMES")
            return []
        except Exception as e:
            print(f"✗ Wan: pipeline error: {e}")
            return []

        dt = time.perf_counter() - t0
        print(f"✓ Wan: {len(frames)} frames in {dt:.1f}s ({len(frames)/dt:.1f} fps) @ {w}×{h}")
        return [self._pil_to_bgr(f) for f in frames]

class WanI2V:
    """Wan2.2 TI2V-5B image-to-video: a single still + a text prompt drive the
    motion. Kept SEPARATE from `Wan` (FLF2V-14B) so the morph/first-last path is
    untouched — the two models are architecturally different:

      * FLF2V-14B (`Wan`)  — CLIP image encoder, needs a first AND last frame
        (514 image tokens); feeding one frame reshapes wrong and crashes.
      * TI2V-5B   (`WanI2V`) — NO image encoder (transformer.config.image_dim is
        null). The pipeline VAE-encodes the input frame and concatenates the
        latent (expand_timesteps), so single-image prompt-driven I2V is native.

    At ~5B (~10 GB bf16) the whole model fits on 32 GB with room to spare, so the
    default is full GPU residency at bf16 — faster than int8 here (int8 adds a
    per-matmul fp16 cast and would only help if VRAM were tight). QUANTIZE="8bit"
    stays available for smaller cards. BGR numpy in/out, like the rest of ai.py."""

    NEGATIVE = ("blurry, low quality, distorted, deformed, extra limbs, "
                "duplicate face, text, watermark, jpeg artifacts, flickering")

    def __init__(self, MODEL_ID="Wan-AI/Wan2.2-TI2V-5B-Diffusers", SIZE=480,
                 INFERENCE_STEPS=30, GUIDANCE_SCALE=5.0, NUM_FRAMES=33,
                 FPS=24, SEED=42, negative=None, OFFLOAD=False, QUANTIZE=None):
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        from huggingface_hub import snapshot_download

        self.SIZE = SIZE
        self.INFERENCE_STEPS = INFERENCE_STEPS
        self.GUIDANCE_SCALE = GUIDANCE_SCALE
        self.NUM_FRAMES = NUM_FRAMES
        self.FPS = FPS
        self.SEED = SEED
        if negative is not None:
            self.NEGATIVE = negative
        self.DEVICE = "cuda"
        self.quantized = bool(QUANTIZE)

        print(f"Loading {MODEL_ID} (I2V)...")
        local_dir = snapshot_download(MODEL_ID)
        vae = AutoencoderKLWan.from_pretrained(
            local_dir, subfolder="vae", torch_dtype=torch.float32
        )

        # No image_encoder for TI2V-5B — the pipeline conditions via VAE-latent
        # concat, guarded internally by transformer.config.image_dim (null here),
        # so we deliberately don't load CLIP. Optional int8 only if requested.
        transformer = None
        if self.quantized:
            from diffusers import BitsAndBytesConfig, WanTransformer3DModel
            qcfg = BitsAndBytesConfig(load_in_8bit=True)
            transformer = WanTransformer3DModel.from_pretrained(
                local_dir, subfolder="transformer",
                quantization_config=qcfg, torch_dtype=torch.bfloat16,
            )
            print("✓ WanI2V: transformer quantized to int8 (bitsandbytes)")
        extra = {"transformer": transformer} if transformer is not None else {}

        self.pipe = WanImageToVideoPipeline.from_pretrained(
            local_dir, vae=vae, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, **extra,
        )
        print("✓ WanI2V: pipeline loaded (latent-concat I2V, no CLIP encoder)")

        if self.quantized:
            self.pipe.enable_model_cpu_offload()
            print("✓ WanI2V: model CPU offload (int8 transformer GPU-resident)")
        elif OFFLOAD:
            self.pipe.enable_model_cpu_offload()
            print("✓ WanI2V: model CPU offload (weights stream to GPU on demand)")
        else:
            try:
                self.pipe.to(self.DEVICE)
                print("✓ WanI2V: full GPU residency (bf16)")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.pipe.enable_model_cpu_offload()
                print("✓ WanI2V: model CPU offload (OOM fallback)")

        try:
            self.pipe.transformer.set_attention_backend("sage")
            print("✓ WanI2V: attention backend SageAttention")
        except Exception:
            print("• WanI2V: attention backend default SDPA")

        try:
            from diffusers import TaylorSeerCacheConfig
            self.pipe.transformer.enable_cache(TaylorSeerCacheConfig())
            print("✓ WanI2V: TaylorSeer cache enabled")
        except Exception:
            print("• WanI2V: activation cache not available")

        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        if self.quantized:
            print("• WanI2V: torch.compile skipped (int8 transformer)")
        else:
            try:
                self.pipe.transformer = torch.compile(
                    self.pipe.transformer, mode="default", fullgraph=False
                )
                print("✓ WanI2V: torch.compile enabled (first run compiles ~2 min)")
            except Exception as _e:
                print(f"• WanI2V: torch.compile skipped ({_e})")

        self._mod = self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]
        self.size = SIZE // self._mod * self._mod
        if self.size != SIZE:
            print(f"⚠ WanI2V: SIZE={SIZE} snapped to {self.size} (grid={self._mod}px)")
        print(f"✓ WanI2V ready | grid {self._mod}px, default {self.size}×{self.size}")

    def _snap(self, value):
        return max(self._mod, int(value) // self._mod * self._mod)

    def _bgr_to_pil(self, bgr, w, h):
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize(
            (w, h), Image.LANCZOS
        )

    def _pil_to_bgr(self, pil):
        return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    def generate(self, image_bgr, prompt="", num_frames=None,
                 width=None, height=None, seed=None):
        """Animate image_bgr with a text motion prompt (single-image I2V — no
        last frame). Returns a list of BGR frames. Width/height default to the
        configured SIZE so large Gemini images are downscaled to the safe range."""
        num_frames = self.NUM_FRAMES if num_frames is None else num_frames
        seed = self.SEED if seed is None else seed
        w = self._snap(width  if width  is not None else self.size)
        h = self._snap(height if height is not None else self.size)

        print(f"• WanI2V: {w}×{h} | {num_frames}f | {self.INFERENCE_STEPS} steps | seed={seed}")
        first = self._bgr_to_pil(image_bgr, w, h)
        kwargs = dict(
            image=first, prompt=prompt, negative_prompt=self.NEGATIVE,
            height=h, width=w, num_frames=num_frames,
            num_inference_steps=self.INFERENCE_STEPS,
            guidance_scale=self.GUIDANCE_SCALE, output_type="pil",
            generator=torch.Generator(self.DEVICE).manual_seed(seed),
        )

        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                result = self.pipe(**kwargs)
            frames = result.frames[0]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"✗ WanI2V: OOM at {w}×{h} × {num_frames}f — reduce SIZE or NUM_FRAMES")
            return []
        except Exception as e:
            print(f"✗ WanI2V: pipeline error: {e}")
            return []

        dt = time.perf_counter() - t0
        print(f"✓ WanI2V: {len(frames)} frames in {dt:.1f}s ({len(frames)/dt:.1f} fps) @ {w}×{h}")
        return [self._pil_to_bgr(f) for f in frames]

class FILM:
    """Google FILM (Frame Interpolation for Large Motion) — drop-in replacement
    for OpticalFlow/RIFE exposing the same .interpolate(a, b, steps) contract.

    Unlike RIFE, which warps pixels along a single estimated flow field (and so
    smears or ghosts under rotation, scaling and large translation), FILM predicts
    a shared multi-scale feature pyramid and *synthesizes* the in-between frame,
    including disoccluded regions. This makes affine-style motion between two SD
    keyframes (turning, zooming, sliding) look far cleaner. Note: like all frame
    interpolators it only fills BETWEEN two fixed endpoints — it cannot invent new
    content (e.g. grow hair); that remains the SD journey's job.

    Uses the TorchScript export from dajes/frame-interpolation-pytorch:
        https://github.com/dajes/frame-interpolation-pytorch/releases
    Drop `film_net_fp16.pt` (or `film_net_fp32.pt`) into MODELS_FOLDER. The scripted
    model's forward is model(img0, img1, dt): images are [B,3,H,W] in [0,1], dt is
    [B,1]; it returns the frame at time dt (trained on the interior, best near 0.5).
    """
    ALIGN = 64  # FILM's feature pyramid needs H,W divisible by 2^(levels)

    def __init__(self, folder, half=True, model_name=None):
        self.DEVICE = "cuda"
        self.half = bool(half) and torch.cuda.is_available()
        self.dtype = torch.float16 if self.half else torch.float32
        name = model_name or ("film_net_fp16.pt" if self.half else "film_net_fp32.pt")
        self.model = torch.jit.load(f"{folder}/{name}", map_location=self.DEVICE)
        self.model.eval().to(self.DEVICE)
        print(f"✓ [ai] FILM frame interpolation ready | {name}")

    def _to_tensor(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
        return t.unsqueeze(0).to(self.DEVICE, dtype=self.dtype)

    def _from_tensor(self, t, h, w):
        arr = t[0, :, :h, :w].float().clamp(0, 1).mul(255.0).byte().cpu().numpy()
        return cv2.cvtColor(arr.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)

    def interpolate(self, frame1_bgr, frame2_bgr, steps):
        img0 = self._to_tensor(frame1_bgr)
        img1 = self._to_tensor(frame2_bgr)

        _, _, h, w = img0.shape
        ph = ((h - 1) // self.ALIGN + 1) * self.ALIGN
        pw = ((w - 1) // self.ALIGN + 1) * self.ALIGN
        padding = (0, pw - w, 0, ph - h)
        img0 = torch.nn.functional.pad(img0, padding)
        img1 = torch.nn.functional.pad(img1, padding)

        # FILM is trained for the MIDPOINT (t=0.5). Arbitrary-t calls are unreliable:
        # the interior frames barely move and never span the full start→end motion, so
        # most of the journey goes "missing" and the keyframe snaps in at the end. The
        # correct usage is RECURSIVE MIDPOINT BISECTION — every call asks only for
        # t=0.5 (FILM's strong regime), which yields 2^depth-1 frames uniformly spaced
        # in time that cover the whole journey. We then arc-length resample to `steps`
        # constant-velocity frames, with the true endpoint as the last (the caller no
        # longer appends it).
        def _mid(x0, x1):
            dt = x0.new_full((x0.shape[0], 1), 0.5)
            out = self.model(x0, x1, dt)
            if isinstance(out, dict):  # some exports return {'image': tensor, ...}
                out = out.get("image", next(iter(out.values())))
            return out

        def _bisect(x0, x1, depth):
            if depth == 0:
                return []
            m = _mid(x0, x1)
            return _bisect(x0, m, depth - 1) + [m] + _bisect(m, x1, depth - 1)

        depth = max(1, math.ceil(math.log2(steps + 1)))
        with torch.no_grad():
            ladder = [img0] + _bisect(img0, img1, depth) + [img1]

            # FILM eases in/out, so equal TIME steps are not equal MOTION steps and the
            # transition visibly slows at each keyframe. Resample to constant velocity
            # by walking equal ARC-LENGTH increments (arc length = cumulative mean-abs
            # pixel change). We snap to the nearest ladder frame (no cross-fade) so
            # FILM's sharpness is preserved.
            cum = [0.0]
            for i in range(len(ladder) - 1):
                cum.append(cum[-1] + (ladder[i + 1] - ladder[i]).abs().mean().item())
            total = cum[-1] or 1.0

            out = []
            for k in range(1, steps + 1):
                target = total * k / steps          # k=steps -> total -> exact endpoint
                j = min(range(len(cum)), key=lambda i: abs(cum[i] - target))
                out.append(self._from_tensor(ladder[j], h, w))
        return out

class AMT:
    """AMT (All-Pairs Multi-Field Transforms for Efficient Frame Interpolation,
    CVPR 2023) — drop-in interpolator exposing the same .interpolate(a, b, steps)
    contract as FILM and OpticalFlow(RIFE).

    AMT builds an all-pairs correlation volume and jointly refines bidirectional
    multi-field flows + occlusion, giving sharper results than RIFE under large
    motion while staying far lighter than FILM. Like every VFI model it only fills
    BETWEEN two fixed endpoints — it cannot synthesize new content. It supports
    arbitrary in-between time t, so we use the same interior-t scheme as FILM.

    Needs the official repo (MCG-NKU/AMT) checked out so its `networks` package and
    `cfgs/` are importable (set AMT_REPO), plus the matching checkpoint
    (amt-s.pth / amt-l.pth / amt-g.pth) in MODELS_FOLDER (set AMT_MODEL). The network
    hyperparameters are read straight from the repo's cfg YAML so they always match
    the checkpoint.
    """
    DIVISOR = 16  # AMT's correlation pyramid needs H,W divisible by 16

    def __init__(self, folder, repo, model_name="amt-s", scale_factor=1.0, half=False):
        import sys, importlib
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from omegaconf import OmegaConf

        self.DEVICE = "cuda"
        # AMT's correlation runs fp32 in the reference demo; keep fp16 opt-in.
        self.half = bool(half) and torch.cuda.is_available()
        self.dtype = torch.float16 if self.half else torch.float32
        self.scale_factor = scale_factor

        cfg = OmegaConf.load(f"{repo}/cfgs/{model_name.upper()}.yaml")
        net = cfg.network
        module_path, cls_name = net.name.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_path), cls_name)
        params = OmegaConf.to_container(net.params, resolve=True) if "params" in net else {}
        self.model = cls(**params)

        ckpt = torch.load(f"{folder}/{model_name}.pth", map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt.get("state_dict", ckpt))
        self.model = self.model.to(self.DEVICE).eval()
        if self.half:
            self.model = self.model.half()
        print(f"✓ [ai] AMT frame interpolation ready | {model_name}")

    def _to_tensor(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
        return t.unsqueeze(0).to(self.DEVICE, dtype=self.dtype)

    def _from_tensor(self, t, h, w):
        arr = t[0, :, :h, :w].float().clamp(0, 1).mul(255.0).byte().cpu().numpy()
        return cv2.cvtColor(arr.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)

    def interpolate(self, frame1_bgr, frame2_bgr, steps):
        img0 = self._to_tensor(frame1_bgr)
        img1 = self._to_tensor(frame2_bgr)

        _, _, h, w = img0.shape
        ph = ((h - 1) // self.DIVISOR + 1) * self.DIVISOR
        pw = ((w - 1) // self.DIVISOR + 1) * self.DIVISOR
        padding = (0, pw - w, 0, ph - h)
        img0 = torch.nn.functional.pad(img0, padding)
        img1 = torch.nn.functional.pad(img1, padding)

        flow = []
        with torch.no_grad():
            for i in range(1, steps + 1):
                # Times i/steps so the LAST frame is t=1.0 (the endpoint), matching
                # RIFE/FILM. The caller no longer appends the raw SD keyframe, so the
                # transition stays uniformly spaced with no sharp keyframe "snap".
                t = i / steps
                embt = torch.tensor(t, dtype=self.dtype, device=self.DEVICE).view(1, 1, 1, 1)
                pred = self.model(img0, img1, embt, scale_factor=self.scale_factor, eval=True)["imgt_pred"]
                flow.append(self._from_tensor(pred, h, w))
        return flow

class OpticalFlow:
    def __init__(self):
        import sys
        sys.path.insert(0, r"..\..\models\Practical-RIFE")
        sys.path.insert(0, r"..\..\models\Practical-RIFE\train_log")
        from RIFE_HDv3 import Model

        self.DEVICE = "cuda"
        self.model = Model()
        self.model.load_model(r"..\..\models\Practical-RIFE\train_log", -1)
        self.model.eval()
        self.model.device()

    def interpolate(self, frame1_bgr, frame2_bgr, steps):
        def to_tensor(bgr):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return torch.from_numpy(
                rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            ).unsqueeze(0).to(self.DEVICE)

        img0 = to_tensor(frame1_bgr)
        img1 = to_tensor(frame2_bgr)

        _, _, h, w = img0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)

        img0 = torch.nn.functional.pad(img0, padding)
        img1 = torch.nn.functional.pad(img1, padding)

        # Collect all frames on GPU first, then do one batched PCIe transfer
        tensors = []
        with torch.no_grad():
            for i in range(1, steps + 1):
                mid = self.model.inference(img0, img1, i / steps)
                tensors.append(mid[:, :, :h, :w])

        batch = torch.cat(tensors, 0).cpu().numpy()  # single H2D→D2H transfer
        flow = []
        for i in range(steps):
            frame_np = (batch[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            flow.append(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
        return flow

class LTXVideo:
    """LTX-Video (Lightricks/LTX-Video) image-to-video — ~2B params, much faster
    than Wan. Exposes generate(image_bgr, prompt, ...) matching the Wan interface
    so it can be swapped in wherever Wan is used. BGR numpy in/out.

    Constraints:
      - Width/height must be multiples of 32
      - num_frames must satisfy (N-1) % 8 == 0 (e.g. 9, 17, 25, 33)
    LTX-Video I2V animates FROM the first frame only; last_image_bgr is accepted
    for interface compatibility but not used."""

    NEGATIVE = ("low quality, worst quality, deformed, distorted, disfigured, "
                "motion smear, motion artifacts, fused fingers, bad anatomy, "
                "ugly, watermark, text")

    def __init__(self, MODEL_ID="Lightricks/LTX-Video", WIDTH=768, HEIGHT=512,
                 INFERENCE_STEPS=25, GUIDANCE_SCALE=3.0, NUM_FRAMES=25,
                 FPS=24, SEED=42, OFFLOAD=False, negative=None):
        from diffusers import LTXImageToVideoPipeline

        self.INFERENCE_STEPS = INFERENCE_STEPS
        self.GUIDANCE_SCALE  = GUIDANCE_SCALE
        self.FPS  = FPS
        self.SEED = SEED
        if negative is not None:
            self.NEGATIVE = negative
        self.DEVICE = "cuda"

        print("Loading LTX-Video...")
        self.pipe = LTXImageToVideoPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16
        )
        if OFFLOAD:
            self.pipe.enable_model_cpu_offload()
            print("✓ LTX-Video: model CPU offload")
        else:
            try:
                self.pipe.to(self.DEVICE)
                print("✓ LTX-Video: full GPU residency")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.pipe.enable_model_cpu_offload()
                print("✓ LTX-Video: model CPU offload (OOM fallback)")
        self.pipe.vae.enable_slicing()

        self.width      = max(32, int(WIDTH)  // 32 * 32)
        self.height     = max(32, int(HEIGHT) // 32 * 32)
        nf = int(NUM_FRAMES)
        self.num_frames = nf if (nf - 1) % 8 == 0 else (nf // 8) * 8 + 1
        print(f"✓ LTX-Video ready | {self.width}×{self.height} {self.num_frames} frames")

    def _snap_spatial(self, v):
        return max(32, int(v) // 32 * 32)

    def _snap_temporal(self, n):
        n = int(n)
        return n if (n - 1) % 8 == 0 else (n // 8) * 8 + 1

    def _bgr_to_pil(self, bgr, w, h):
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize(
            (w, h), Image.LANCZOS
        )

    def _pil_to_bgr(self, pil):
        return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    def generate(self, image_bgr, prompt="", last_image_bgr=None,
                 num_frames=None, width=None, height=None, seed=None):
        """Animate image_bgr with a text prompt. last_image_bgr is ignored —
        LTX-Video I2V conditions on the first frame only. Returns BGR frames."""
        w  = self._snap_spatial(width  or self.width)
        h  = self._snap_spatial(height or self.height)
        nf = self._snap_temporal(num_frames or self.num_frames)
        seed = self.SEED if seed is None else seed
        if last_image_bgr is not None:
            print("• LTX-Video: last_image_bgr ignored (no FLF2V support)")

        first = self._bgr_to_pil(image_bgr, w, h)

        t0 = time.perf_counter()
        with torch.inference_mode():
            result = self.pipe(
                image=first,
                prompt=prompt,
                negative_prompt=self.NEGATIVE,
                width=w, height=h,
                num_frames=nf,
                num_inference_steps=self.INFERENCE_STEPS,
                guidance_scale=self.GUIDANCE_SCALE,
                output_type="pil",
                generator=torch.Generator(self.DEVICE).manual_seed(seed),
            )
        frames = result.frames[0]
        dt = time.perf_counter() - t0
        print(f"✓ LTX-Video: {len(frames)} frames in {dt:.1f}s ({len(frames)/dt:.1f} fps) @ {w}×{h} seed={seed}")
        return [self._pil_to_bgr(f) for f in frames]
