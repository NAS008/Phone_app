import os
import cv2
import glob
import random
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

class Folder:
    def __init__(self, image_size, input_folder):
        self.image_size = image_size
        self.paths = self._init_image_list(input_folder)

    def _init_image_list(self, folder, extensions=("*.png", "*.jpg", "*.jpeg")):
        paths = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(folder, ext)))
        if not paths:
            print("No images found in folder!")
        paths.sort()
        return paths

    def _adjust_image(self, path, IW, IH):
        img = cv2.imread(path)
        h, w = img.shape[:2]

        # Step 1: scale to fill IW x IH (cover, no black bars)
        scale = max(IW / w, IH / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

        # Step 2: crop center to exactly IW x IH
        x_off = (new_w - IW) // 2
        y_off = (new_h - IH) // 2
        cropped = resized[y_off:y_off + IH, x_off:x_off + IW]

        # Step 3: paste into clean canvas (guarantees exact output size)
        canvas = np.zeros((IH, IW, img.shape[2]), dtype=img.dtype)
        canvas[0:IH, 0:IW] = cropped
        return canvas
     
    def load_image(self, id=None):
        if id is None:
            image_path = random.choice(self.paths)
        else:
            image_path = self.paths[id]
        image = self._adjust_image(image_path, self.image_size, self.image_size)
        return image
    
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
    def generate_between_images(self, image_a_bgr, image_b_bgr, prompt="", prompt_a=None, prompt_b=None):
        latents_a = self.prepare_reference(image_a_bgr)
        latents_b = self.prepare_reference(image_b_bgr)

        prompt_embeds, negative_embeds, pooled_embeds, negative_pooled = self.encode_prompt(prompt)
        embeds = torch.cat([negative_embeds, prompt_embeds], dim=0)
        pooled = torch.cat([negative_pooled, pooled_embeds], dim=0)

        noise_a = self._noise_for(image_a_bgr, latents_a)
        noise_b = self._noise_for(image_b_bgr, latents_b)

        sigma_min = 0.60
        sigma_max = 1.0

        for i in range(1, self.INFERENCE_STEPS + 1):
            t = i / (self.INFERENCE_STEPS + 1)
            alpha = 0.5 - 0.5 * np.cos(np.pi * t)
            sigma = sigma_min + (sigma_max - sigma_min) * (1.0 - (2.0 * alpha - 1.0) ** 2)
            base_latents = self.slerp(latents_a, latents_b, alpha)
            noise = self.slerp(noise_a, noise_b, alpha)
            latents = self.denoise_from_sigma(base_latents, noise, sigma, embeds, pooled)
            frame = self.decode_latents(latents)

            yield frame

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

        flow = []
        with torch.no_grad():
            for i in range(1, steps + 1):
                t = i / steps
                mid = self.model.inference(img0, img1, t)
                mid = mid[:, :, :h, :w]
                mid_np = (mid[0].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                flow.append(cv2.cvtColor(mid_np, cv2.COLOR_RGB2BGR))
        return flow
