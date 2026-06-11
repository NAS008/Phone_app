import cv2
import hashlib
import numpy as np
import torch
import os
import glob
import random
import spandrel
from PIL import Image
from diffusers import StableDiffusion3Pipeline, AnimateDiffSparseControlNetPipeline, DPMSolverMultistepScheduler
from diffusers.models import MotionAdapter, SparseControlNetModel

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

    def generate_between_images(self, image_a_bgr, image_b_bgr, prompt="", prompt_a=None, prompt_b=None):
        latents_a = self.prepare_reference(image_a_bgr)
        latents_b = self.prepare_reference(image_b_bgr)

        prompt_a = prompt_a or prompt
        prompt_b = prompt_b or prompt
        embeds_a, negative_embeds, pooled_a, negative_pooled = self.encode_prompt(prompt_a)
        if prompt_b == prompt_a:
            embeds_b, pooled_b = embeds_a, pooled_a
        else:
            embeds_b, _, pooled_b, _ = self.encode_prompt(prompt_b)

        noise_a = self._noise_for(image_a_bgr, latents_a)
        noise_b = self._noise_for(image_b_bgr, latents_b)

        sigma_min = 0.60
        sigma_max = 0.90  # below 1.0 so the slerped endpoints still bias the midpoint

        for i in range(1, self.INFERENCE_STEPS + 1):
            t = i / (self.INFERENCE_STEPS + 1)
            alpha = float(0.5 - 0.5 * np.cos(np.pi * t))
            sigma = sigma_min + (sigma_max - sigma_min) * (1.0 - (2.0 * alpha - 1.0) ** 2)
            base_latents = self.slerp(latents_a, latents_b, alpha)
            noise = self.slerp(noise_a, noise_b, alpha)
            # lerp (not slerp) for text embeddings: per-token norms aren't spherical
            prompt_embeds = torch.lerp(embeds_a, embeds_b, alpha)
            pooled_embeds = torch.lerp(pooled_a, pooled_b, alpha)
            embeds = torch.cat([negative_embeds, prompt_embeds], dim=0)
            pooled = torch.cat([negative_pooled, pooled_embeds], dim=0)
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
