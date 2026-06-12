from dataclasses import dataclass
from typing import List
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

@dataclass
class OverlayTextSpec:
    text: str
    x: float
    y: float
    width: float
    align: str
    background_alpha: float
    padding: float

@dataclass
class OverlayLogoSpec:
    asset_path: str
    x: float
    y: float
    width: float
    align: str

@dataclass
class CompositionSpec:
    text: List[OverlayTextSpec]
    logos: List[OverlayLogoSpec]

class Brand:
    _BRANDED_THEMES = [
            "labrador dog",
            "labrador puppy",
            "labrador dog fetching",
            "labrador dog riding a car",
            "labrador dog house",
            "labrador dog dressed as a doctor",
        ]
    _BRAND_STYLE = "inspired by Soviet Constructivist posters, warm red, soft steel grey, bone white, gentle diagonals, friendly labrador mascot as central figure"
    _BRAND_LOGO = "../../brand/fidelidade/logo.png"
    _BRAND_SELLING_LINE = "Não se fala de dinheiro à mesa, mas devia"
    _BRAND_FONT = "C:/Windows/Fonts/arial.ttf"

    def __init__(self):
        # Make sure brand.py is saved as UTF-8 in your editor.
        self.logo_path = self._BRAND_LOGO
        self.selling_line = self._BRAND_SELLING_LINE
        self.font = self._BRAND_FONT
        self.dark_color_transparent = True  # red logo on white bg
        self.line_spacing = 1.2

        # Text layout config
        text_cfg = {
            "x": 0.03,   # 3% margin from right
            "y": 0.95,
            "width": 0.7,
            "align": "right",
            "background_alpha": 0.5,
            "padding": 0.01,
        }

        logo_cfg = {
            "x": 0.03,   # 3% margin from right
            "y": 0.99,
            "width": 0.18,
            "align": "right",
        }

        self.comp = CompositionSpec(
            text=[
                OverlayTextSpec(
                    text=self.selling_line,
                    x=float(text_cfg["x"]),
                    y=float(text_cfg["y"]),
                    width=float(text_cfg["width"]),
                    align=text_cfg["align"],
                    background_alpha=float(text_cfg["background_alpha"]),
                    padding=float(text_cfg["padding"]),
                )
            ],
            logos=[
                OverlayLogoSpec(
                    asset_path=self.logo_path,
                    x=float(logo_cfg["x"]),
                    y=float(logo_cfg["y"]),
                    width=float(logo_cfg["width"]),
                    align=logo_cfg["align"],
                )
            ],
        )

    def _load_and_normalize_logo_rgba(
        self,
        logo_path: str,
        target_w: int,
        dark_color_transparent: bool = True,
    ) -> np.ndarray:
        img = cv2.imread(logo_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load logo at {logo_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        img_f = img_rgb.astype(np.float32) / 255.0
        r = img_f[..., 0]
        g = img_f[..., 1]
        b = img_f[..., 2]

        white_mask = (r > 0.9) & (g > 0.9) & (b > 0.9)
        red_mask = (r > 0.5) & (g < 0.4) & (b < 0.4)

        if dark_color_transparent:
            logo_mask = red_mask
        else:
            logo_mask = white_mask

        alpha = np.zeros_like(r, dtype=np.float32)
        alpha[logo_mask] = 1.0

        out_rgb = np.zeros_like(img_f)
        out_rgb[..., 0] = 1.0
        out_rgb[..., 1] = 1.0
        out_rgb[..., 2] = 1.0

        rgba = np.dstack([
            out_rgb[..., 0] * 255.0,
            out_rgb[..., 1] * 255.0,
            out_rgb[..., 2] * 255.0,
            alpha * 255.0,
        ]).astype(np.uint8)

        scale = target_w / float(w)
        target_h = max(1, int(h * scale))
        rgba_resized = cv2.resize(rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return rgba_resized

    def render_mask(self, window_w: int, window_h: int) -> np.ndarray:
        # Fully transparent RGBA canvas
        canvas_pil = Image.new("RGBA", (window_w, window_h), (0, 0, 0, 0))

        # Text
        for txt in self.comp.text:
            draw = ImageDraw.Draw(canvas_pil)

            right_margin = int(txt.x * window_w)
            x_right = window_w - right_margin
            y_center = int(txt.y * window_h)
            max_width = int(txt.width * window_w)

            font_path = self.font
            font_size = int(window_h * 0.035)
            font = ImageFont.truetype(font_path, font_size)

            lines = txt.text.split("\n")

            def _measure(fnt):
                metrics = []
                mw = 0
                for ln in lines:
                    bb = draw.textbbox((0, 0), ln if ln else " ", font=fnt)
                    lw = bb[2] - bb[0]
                    lh = bb[3] - bb[1]
                    metrics.append((ln, lw, lh, bb[1], bb[3]))
                    mw = max(mw, lw)
                return metrics, mw

            line_metrics, max_line_w = _measure(font)

            while max_line_w > max_width and font_size > 8:
                font_size = int(font_size * 0.9)
                font = ImageFont.truetype(font_path, font_size)
                line_metrics, max_line_w = _measure(font)

            # Compute where each line's anchor y sits relative to block_y=0
            line_y_offsets = []
            cy = 0
            for _, _, lh, _, _ in line_metrics:
                line_y_offsets.append(cy)
                cy += int(lh * self.line_spacing)

            # Visual bounds: first glyph top to last glyph bottom
            visual_top = line_y_offsets[0] + line_metrics[0][3]   # + bbox[1]
            visual_bottom = line_y_offsets[-1] + line_metrics[-1][4]  # + bbox[3]
            visual_h = visual_bottom - visual_top

            # Position block_y so visible glyphs are centered on y_center
            block_y = int(y_center - visual_top - visual_h / 2)
            block_x = x_right - max_line_w

            pad = int(txt.padding * window_h)
            bg_alpha = int(txt.background_alpha * 255)
            draw.rectangle(
                (0, block_y + visual_top - pad, window_w, window_h),#block_y + visual_bottom + pad),
                fill=(0, 0, 0, bg_alpha),
            )

            text_fill = (255, 255, 255, 255)
            for (line, lw, lh, _, _), y_off in zip(line_metrics, line_y_offsets):
                line_x = block_x + (max_line_w - lw)
                cur_y = block_y + y_off
                draw.text((line_x, cur_y), line, font=font, fill=text_fill)

        # Logo
        for logo in self.comp.logos:
            target_w = int(logo.width * window_w)
            if target_w <= 0:
                continue

            try:
                logo_rgba = self._load_and_normalize_logo_rgba(
                    logo.asset_path,
                    target_w=target_w,
                    dark_color_transparent=self.dark_color_transparent,
                )
            except Exception:
                continue

            target_h = logo_rgba.shape[0]
            right_margin = int(logo.x * window_w)
            x2 = window_w - right_margin
            x1 = x2 - target_w
            logo_pad = int(0.01 * window_h)
            cy = int(logo.y * window_h) - logo_pad
            y1 = cy - target_h // 2

            logo_pil = Image.fromarray(logo_rgba, "RGBA")
            # Crop to visible region before pasting
            src_x0 = max(0, -x1); src_y0 = max(0, -y1)
            dst_x = max(0, x1);   dst_y = max(0, y1)
            src_x1 = src_x0 + min(window_w, x2) - dst_x
            src_y1 = src_y0 + min(window_h, y1 + target_h) - dst_y
            if src_x1 > src_x0 and src_y1 > src_y0:
                cropped = logo_pil.crop((src_x0, src_y0, src_x1, src_y1))
                canvas_pil.paste(cropped, (dst_x, dst_y), cropped)

        return np.array(canvas_pil)  # RGBA (H, W, 4)

    @staticmethod
    def composite_mask_over_frame(
        frame_bgr: np.ndarray,
        mask_rgba: np.ndarray,
        strength: float = 1.0,
    ) -> np.ndarray:
        assert frame_bgr.shape[:2] == mask_rgba.shape[:2]

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        mask_rgb = mask_rgba[:, :, :3].astype(np.float32)
        mask_a = mask_rgba[:, :, 3:4].astype(np.float32) / 255.0 * strength

        out_rgb = mask_a * mask_rgb + (1.0 - mask_a) * frame_rgb
        return cv2.cvtColor(np.clip(out_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    def resize_to_fit_window(self, img, window_w, window_h):
        target_w = window_w
        target_h = window_h

        img_h, img_w = img.shape[:2]

        # Scale to cover the whole target area
        scale = max(float(target_w) / float(img_w), float(target_h) / float(img_h))
        new_w = max(1, int(round(img_w * scale)))
        new_h = max(1, int(round(img_h * scale)))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Center-crop to exact window size
        x0 = max(0, (new_w - target_w) // 2)
        y0 = max(0, (new_h - target_h) // 2)

        frame = resized[y0:y0 + target_h, x0:x0 + target_w]

        return frame
