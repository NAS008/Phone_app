# # # # # # #
# The       #
# First     #
# NonCarbon #
# Artist    #
# # # # # # #

# PC App uses a GPU PC to run OpticalFlow, Simulator and RayTracing on site
# Has no UI, just a borderless display window
# Input: Receives images from the bus to update Simulator gradients and xyz_goal
# Interpolates with OpticalFlow to get smooth color transitions between images while updates particles rgb
# Does particles xyz simulation (heightmap, fluid, ui, constraints, collision, boundaries)
# Output: Sends the current rendered frame when a like is received
# ----------------------------------------------------------------
# ✓ Update sim gradients and xyz_goal
# ✓ OpticalFlow interpolation, update rgb
# ✓ Simulation update
# ✗ Add FLIP fluid
# ✗ Add collisions
# ✗ Add world boundaries
# ✗ Add slime simulation
# ✗ Fix gradient wind
# ✗ Fix noise on ellipsoid
# ✗ Fix detail on prisms with multiple intersect triangles
# ✗ Fix insert triangle to reduce the number of hit cells

import cv2
import io
import urllib.request
import urllib.parse
import numpy as np
import math
import time
import asyncio
import ctypes
import time
from collections import deque
from PIL import Image
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'shared'))
from config import Config
from session import Session
from bus import Bus
from ray import RayTracer
from sd35 import OpticalFlow
from sim import Simulator
from ui import Camera
ctypes.windll.user32.SetProcessDPIAware()

def get_first_part(parts, kind):
    for part in parts or []:
        if part.get("kind") == kind:
            return part
    return None

def get_image_bytes_from_parts(parts):
    part = get_first_part(parts, "image")
    if not part:
        return None
    return part.get("data")

def get_image_bytes(image_bgr, image_size=0, quality=85):
    if image_size > 0:
        h, w = image_bgr.shape[:2]

        if h <= w:
            new_h = image_size
            new_w = int(round(w * (image_size / h)))
        else:
            new_w = image_size
            new_h = int(round(h * (image_size / w)))
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    else:
        ok, buffer = cv2.imencode('.jpg', image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        
    if not ok:
        raise ValueError('✗ PC: JPEG encoding failed')
    return buffer.tobytes()

def overlay(frame, overlay_img, proportion=16, alignment="center"):
    out = frame.copy()

    H, W = out.shape[:2]
    h, w = overlay_img.shape[:2]
    ar = float(w) / h
    h = H // proportion
    if h < 100:
        h = 100
    w = int(h * ar)

    overlay_img = cv2.resize(overlay_img, (w, h), interpolation=cv2.INTER_NEAREST)

    if alignment == "center":
        x0 = (W - w) // 2
        y0 = (H - h) // 2
    elif alignment == "bottom center":
        x0 = (W - w) // 2
        y0 = H - h - h // 2
    elif alignment == "bottom left":
        x0 = w // 4
        y0 = H - h - h // 2
    elif alignment == "bottom right":
        x0 = W - w - w // 2
        y0 = H - h - h // 2
    elif alignment == "top center":
        x0 = (W - w) // 2
        y0 = h // 2
    elif alignment == "top left":
        x0 = w // 2
        y0 = h // 2
    elif alignment == "top right":
        x0 = W - w - w // 2
        y0 = h // 2
    else:
        raise ValueError(f"Unknown alignment: {alignment}")

    region = out[y0:y0+h, x0:x0+w].astype(np.float32)
    overlay_bgr = overlay_img.astype(np.float32)

    # Compute luminance in [0, 255]
    b = overlay_bgr[:, :, 0]
    g = overlay_bgr[:, :, 1]
    r = overlay_bgr[:, :, 2]
    luminance = 0.114 * b + 0.587 * g + 0.299 * r  # BGR order

    # Normalize to [0, 1] and make it (h, w, 1)
    alpha = (luminance / 255.0)[..., None].astype(np.float32)
    blended = overlay_bgr * alpha + region * (1.0 - alpha)
    out[y0:y0+h, x0:x0+w] = blended.astype(out.dtype)
    return out

def resize_to_fit_window(img, window_w, window_h):
    target_w = window_w
    target_h = window_h

    img_h, img_w = img.shape[:2]

    scale = min(float(target_w) / float(img_w), float(target_h) / float(img_h))
    new_w = max(1, int(round(img_w * scale)))
    new_h = max(1, int(round(img_h * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    frame = np.zeros((target_h, target_w, 3), dtype=img.dtype)

    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2

    frame[y0:y0 + new_h, x0:x0 + new_w] = resized
    return frame

class LowPass:
    def __init__(self, x0=0.0):
        self.y = x0
        self.ready = False

    def apply(self, x, alpha):
        if not self.ready:
            self.y = x
            self.ready = True
        else:
            self.y = alpha * x + (1.0 - alpha) * self.y
        return self.y

def smoothing_alpha(dt, cutoff):
    r = 2.0 * math.pi * cutoff * dt
    return r / (r + 1.0)

class OneEuro:
    def __init__(self, min_cutoff=1.2, beta=0.02, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.xf = LowPass()
        self.dxf = LowPass()
        self.t_prev = None
        self.x_prev = None

    def apply(self, x, t):
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            return self.xf.apply(x, 1.0)

        dt = max(1e-6, t - self.t_prev)
        dx = (x - self.x_prev) / dt
        ad = smoothing_alpha(dt, self.d_cutoff)
        dx_hat = self.dxf.apply(dx, ad)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_alpha(dt, cutoff)
        x_hat = self.xf.apply(x, a)

        self.t_prev = t
        self.x_prev = x
        return x_hat
        
async def main():
    config = Config()

    # Optical flow setup
    of = OpticalFlow()

    # Pointer setup
    pc_cam_on = False
    if pc_cam_on:
        cam = Camera(config.POSE_MODEL)
        cam.start()
    pointer = [0.5, 0.5, 0.0]
    pointer_goal = [0.5, 0.5, 0.0]

    # Simulator setup
    sim = Simulator(
        IMAGE_SIZE=config.IMAGE_SIZE,
        PIXELS_PER_CELL=config.PIXELS_PER_CELL,
        G=config.G, L=1,
        smooth=15,
    )
    sim_constraints_on = True
    sim_go_back_on = True
    sim_gradient_on = False
    sim_depth_factor = 1.0
    #img_a = cv2.imread(r"..\..\brand\logo_square.png")
    img_a = cv2.imread(r"..\..\input\19.png")
    if img_a is None:
        img_a = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
    else:
        img_a = cv2.resize(img_a, (config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    sim.new_image(img_a)
    logo = cv2.imread(r"..\..\brand\logo_white.png")

    # Ray tracer setup
    ray = RayTracer(
        W=config.WINDOW_W,
        H=config.WINDOW_H,
        G=config.G,
        camera=config.camera,
        target=config.target,
        light=config.light,
        fov=config.fov,
        samples=config.samples,
        background=config.background,
        ambient=config.ambient,
        shadow=config.shadow,
    )
    ray_shape = 4
    frame = img_a.copy()
    frames = deque()
    thumb_w = max(config.WINDOW_W // 8, 256)
    thumb_h = int(thumb_w * config.WINDOW_H / config.WINDOW_W)
    last_frames = deque(maxlen=config.FPS * config.VIDEO_SECONDS)
    gif_last_frame = None
    GIF_DIFF_THRESHOLD = 4.0  # mean abs pixel diff (0-255) required to add a frame

    # Window setup
    cv2.namedWindow(config.APP_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(config.APP_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(config.APP_NAME, config.WINDOW_W, config.WINDOW_H)

    # Bus setup
    bus = Bus(config.redis_host, config.redis_port, config.redis_password, config.redis_ssl)
    await bus.connect()

    async def on_ai_message(session_id, nickname, parts, payload):
        nonlocal img_a

        if session_id != session.session_id and session_id != config.ADMIN_SESSION_ID:
            return

        image_bytes = get_image_bytes_from_parts(parts)
        if not image_bytes:
            return

        kb = int(len(image_bytes) / 1024)
        print(f"✓ PC: message received with image {kb} KB from {nickname}")

        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        img_b = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img_b is None:
            print("✗ PC: failed to decode received image")
            return

        img_b = cv2.resize(img_b, (config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=cv2.INTER_AREA)

        start_img = frames[-1] if frames else img_a
        interpolated = of.interpolate(start_img, img_b, config.OF_FRAMES)

        added = 0
        if interpolated is not None:
            for interp in interpolated:
                if interp is not None:
                    frames.append(interp)
                    added += 1

        frames.append(img_b)
        added += 1
        img_a = img_b

        print(f"✓ PC: appended {added} frames, queue size is now {len(frames)} at {config.FPS} FPS")
        await bus.publish_settings(pc_queue_size=len(frames))

    async def on_user_like(session_id, nickname):
        nonlocal frame

        if session_id != session.session_id and session_id != config.ADMIN_SESSION_ID:
            return

        out = overlay(frame, logo, proportion=20, alignment="bottom center")
        image_bytes = get_image_bytes(out)
        await bus.publish_ai_message_to_phone(
            session_id=session_id,
            nickname="NonCarbon Artist",
            text=f"Glad you liked it, {nickname}",
            image_bytes=image_bytes,
        )
        kb = int(len(image_bytes) / 1024)
        print(f"✓ PC: got a like from {nickname} and sent the current frame {kb} KB")

    async def on_user_gesture(session_id, nickname, x, y, z):
        nonlocal pointer_goal

        if session_id != session.session_id and session_id != config.ADMIN_SESSION_ID:
            return

        pointer_goal[0] = max(0.0, min(1.0, x))
        pointer_goal[1] = max(0.0, min(1.0, y))
        pointer_goal[2] = z

    async def on_settings(params):
        nonlocal ray_shape, sim_constraints_on, sim_go_back_on, sim_gradient_on, sim_depth_factor

        if 'constraints_on' in params:
            sim_constraints_on = bool(params['constraints_on'])
            print(f"✓ PC: sim constraints set to {sim_constraints_on}")

        if 'go_back_on' in params:
            sim_go_back_on = bool(params['go_back_on'])
            print(f"✓ PC: sim go back set to {sim_go_back_on}")

        if 'gradient_on' in params:
            sim_gradient_on = bool(params['gradient_on'])
            print(f"✓ PC: sim gradient set to {sim_gradient_on}")

        if 'shape' in params:
            ray_shape = int(params['shape'])
            print(f"✓ PC: ray shape set to {ray_shape}")

        if 'zoom' in params:
            ray.fov = float(params['zoom'])
            print(f"✓ PC: ray zoom set to {ray.fov:.1f}")

        if 'depth_factor' in params:
            sim_depth_factor = float(params['depth_factor'])
            print(f"✓ PC: sim depth factor set to {sim_depth_factor:.2f}")

    async def on_user_video(session_id, nickname):
        if session_id != session.session_id and session_id != config.ADMIN_SESSION_ID:
            return

        snapshot = list(last_frames)
        if not snapshot:
            print("✗ PC: video requested but frame buffer is empty")
            return

        pil_frames = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            for f in snapshot
        ]
        buf = io.BytesIO()
        duration_ms = max(1, int(1000 / (config.FPS * 2)))
        pil_frames[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=pil_frames[1:],
            loop=0,
            duration=duration_ms,
        )
        gif_bytes = buf.getvalue()
        kb = len(gif_bytes) // 1024
        print(f"✓ PC: video gif — {len(snapshot)} frames, {kb} KB")

        # Upload directly to phone backend via HTTP — bypasses Redis pub/sub size limits
        qs = urllib.parse.urlencode({
            "session_id": session_id,
            "nickname": "NonCarbon Artist",
            "text": f"Last {config.VIDEO_SECONDS}s of the artwork",
        })
        upload_url = f"{config.PHONE_BACKEND_URL}/api/gif_upload?{qs}"
        loop = asyncio.get_event_loop()
        def _upload():
            req = urllib.request.Request(
                upload_url,
                data=gif_bytes,
                method="POST",
                headers={"Content-Type": "image/gif", "Content-Length": str(len(gif_bytes))},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        try:
            result = await loop.run_in_executor(None, _upload)
            print(f"✓ PC: GIF uploaded to phone backend ({kb} KB) — response: {result}")
        except Exception as exc:
            print(f"✗ PC: GIF upload failed — {exc}")

    bus.on(Bus.AI_MESSAGE_TO_PC, on_ai_message)
    bus.on(Bus.USER_LIKE, on_user_like)
    bus.on(Bus.USER_GESTURE, on_user_gesture)
    bus.on(Bus.USER_VIDEO, on_user_video)
    bus.on(Bus.SETTINGS, on_settings)

    # Session setup
    session = Session(config.URL)
    session.create_session()
    await bus.publish_session(session_id=session.session_id)
    qr_img = session.generate_qr_code()

    render_frame_dur = 1.0 / config.FPS
    release_frame_dur = 1.0 / config.FPS
    next_tick = time.perf_counter()
    next_release = next_tick

    fx = OneEuro(min_cutoff=1.5, beta=0.03, d_cutoff=1.0)
    fy = OneEuro(min_cutoff=1.5, beta=0.03, d_cutoff=1.0)
    fz = OneEuro(min_cutoff=1.0, beta=0.02, d_cutoff=1.0)
    while True:
        await bus.poll()

        now = time.perf_counter()
        if now < next_tick:
            await asyncio.sleep(next_tick - now)
        now = time.perf_counter()
        next_tick += render_frame_dur

        if now >= next_release:
            if frames:
                img_a = frames.popleft()
                if ray_shape != 5:
                    sim.new_image(img_a, depth_factor=sim_depth_factor)
            if sim_gradient_on and ray_shape != 5:
                sim.inject_gradient(depth_factor=sim_depth_factor)
            next_release += release_frame_dur
            if now > next_release + release_frame_dur:
                next_release = now + release_frame_dur

        if ray_shape == 5:
            frame = resize_to_fit_window(img_a, config.WINDOW_W, config.WINDOW_H)
        else:
            if pc_cam_on:
                ok, canvas, frame, result = cam.read()
                if not ok:
                    continue

                hand = cam.get_hand_raw(result)
                if hand is not None:
                    raw_x = float(np.clip(hand["x"], 0.0, 1.0))
                    raw_y = float(np.clip(1.0 - hand["y"], 0.0, 1.0))

                    pointer_goal[0] = fx.apply(raw_x, now)
                    pointer_goal[1] = fy.apply(raw_y, now)

            pointer_prior_x = pointer[0]
            pointer_prior_y = pointer[1]
            follow = 0.18
            pointer[0] += follow * (pointer_goal[0] - pointer[0])
            pointer[1] += follow * (pointer_goal[1] - pointer[1])
            dt = render_frame_dur
            vx = (pointer[0] - pointer_prior_x) / max(dt, 1e-6)
            vy = (pointer[1] - pointer_prior_y) / max(dt, 1e-6)
            # clamp spikes
            vmax = 1.5
            vx = float(np.clip(vx, -vmax, vmax))
            vy = float(np.clip(vy, -vmax, vmax))
            # optional extra smoothing
            vel_smooth = 0.25
            if not hasattr(main, "_vx"):
                main._vx = 0.0
                main._vy = 0.0
            main._vx += vel_smooth * (vx - main._vx)
            main._vy += vel_smooth * (vy - main._vy)
            vz = 0.15 * np.hypot(main._vx, main._vy)
            sim.inject_mouse(pointer, [main._vx, main._vy, vz])

            sim.update(constraints_on=sim_constraints_on, go_back_on=sim_go_back_on)

            if ray_shape == 0:
                frame = ray.quad(sim.xyz, sim.rgb, sim.rot, 1.0 * sim.r)
            elif ray_shape == 1:
                frame = ray.prism(sim.xyz, sim.rgb, sim.rot, 0.8 * sim.r, 0.8 * sim.r, 8.0 * sim.r)
            elif ray_shape == 2:
                frame = ray.ellipsoid(sim.xyz, sim.rgb, sim.rot, 1.8 * sim.r, 1.8 * sim.r, 0.6 * sim.r)
            elif ray_shape == 3:
                frame = ray.cylinder(sim.xyz, sim.rgb, sim.next_y, 0.4 * sim.r)
            else:
                frame = ray.sphere(sim.xyz, sim.rgb, 1.2 * sim.r)

        thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        if gif_last_frame is None or np.mean(np.abs(thumb.astype(np.float32) - gif_last_frame.astype(np.float32))) > GIF_DIFF_THRESHOLD:
            last_frames.append(thumb)
            gif_last_frame = thumb

        out = overlay(frame, qr_img, proportion=20, alignment="bottom center")
        cv2.imshow(config.APP_NAME, out)

        key = cv2.waitKeyEx(1)
        if key in (ord('q'), 27):
            break

    cv2.destroyAllWindows()
    await bus.close()

if __name__ == '__main__':
    asyncio.run(main())
