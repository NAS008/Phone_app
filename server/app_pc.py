# # # # # # #
# The       #
# First     #
# NonCarbon #
# Artist    #
# # # # # # #

# PC App uses a GPU PC to run OpticalFlow, Simulator and RayTracing on site
# Detects UI gestures from mouse and camera and sounds from mic to interact with Simulator
# Input: Receives images from the bus to inject gradients in Simulator and xyz_goal. Receives gestures or audio to inject in Simulator
# Interpolates with OpticalFlow to get smooth color transitions between images while updates particles rgb
# Does particles xyz simulation (heightmap, fluid, ui, constraints, collision, boundaries)
# Output: Sends the current rendered frame when a like is received
# ----------------------------------------------------------------
# ✓ Update sim gradients and xyz_goal
# ✓ OpticalFlow interpolation, update rgb
# ✓ Simulation update
# ✓ Add camera interaction
# ✓ Add mouse interaction
# ✓ Add audio frequencies interaction
# ✓ Add FLIP fluid
# ✗ Add collisions
# ✓ Add world boundaries
# ✗ Add world round boundaries
# ✗ Add slime simulation
# ✓ Fix gradient wind
# ✓ Fix noise on ellipsoid
# ✓ Fix detail on prisms with multiple intersect triangles
# ✓ Fix insert triangle to reduce the number of hit cells

import cv2
import io
import urllib.request
import urllib.parse
import numpy as np
import time
import asyncio
import ctypes
ctypes.windll.user32.SetProcessDPIAware()
import time
import concurrent.futures
from collections import deque
from PIL import Image
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'shared'))
from config import Config
from bus import Bus
from turn import CloudflareTurn
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'server'))
from session import Session
from ray import RayTracer
from ai import OpticalFlow, SuperResolution
from sim import Simulator
from ui import Camera, Mic, Mouse
from director import Director
from stream import FrameBus, StreamingServer, build_ice_servers
from brand import Brand

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
 
async def main():
    config = Config()

    of = OpticalFlow()
    super = SuperResolution(config.MODELS_FOLDER)

    # Simulator setup
    sim = Simulator(
        IMAGE_SIZE=config.IMAGE_SIZE,
        PIXELS_PER_CELL=config.PIXELS_PER_CELL,
        G=config.G, L=config.LAYERS,
        smooth=3,
        dt = 1.0 / config.FPS_SIM,
    )
    sim_go_back_on = True
    sim_constraints_mode = 0
    sim_gradient_mode = 0
    sim_world_mode = 0
    img_a = cv2.imread(r"..\..\input\19.png")
    img_a = cv2.resize(img_a, (config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    img_a_hires = super.upscale(img_a)
    sim.new_image(img_a)
    logo = cv2.imread(r"..\..\brand\logo_white.png")
    brand = Brand()
    brand_on = False
    _brand_mask = None

    # Raytracer setup
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
    ray_shape = 0
    fov_target = config.fov
    frame = img_a.copy()
    frames = deque()
    frames_hires = deque()
    pending_images = asyncio.Queue(maxsize=16)
    processing_task = None
    processing_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    thumb_w = max(config.WINDOW_W // 8, 256)
    thumb_h = int(thumb_w * config.WINDOW_H / config.WINDOW_W)
    last_frames = deque(maxlen=config.FPS * config.VIDEO_SECONDS)
    gif_last_frame = None
    stream_last_frame = None
    overlay_on = True # To show or hide QR code
    joined_users = {"Director"}

    # Window setup
    cv2.namedWindow(config.APP_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(config.APP_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(config.APP_NAME, config.WINDOW_W, config.WINDOW_H)

    # UI setup
    cam = None
    # cam = Camera(config.UI_POSE_MODEL, dt=1.0 / config.FPS, width=640, height=480)
    # cam.start()

    mic = None
    # mic = Mic(
    #     sample_rate=16000,
    #     block_size=1024,
    #     channels=1,
    #     gain=5.0,
    #     decay=0.72,
    #     num_bands=16,
    # )
    # mic.start()

    mouse = Mouse(dt=1.0 / config.FPS_SIM, width=config.WINDOW_W, height=config.WINDOW_H)
    cv2.setMouseCallback(config.APP_NAME, mouse.callback)

    frame_bus = None
    streaming = None
    runner = None
    if config.stream_on:
        frame_bus = FrameBus()
        cloudflare_turn = CloudflareTurn(
            config.CF_TURN_KEY_ID, config.CF_TURN_API_TOKEN, config.CF_TURN_TTL
        )
        if cloudflare_turn.enabled:
            print("✓ PC: Cloudflare TURN relay enabled")
        elif not config.TURN_URL:
            print("⚠ PC: no TURN relay configured — viewers on mobile data may fail to connect")
        streaming = StreamingServer(
            frame_bus=frame_bus,
            W=config.WINDOW_W,
            H=config.WINDOW_H,
            viewer_html=config.VIEWER_HTML,
            host=config.HOST_IP,
            port=8080,
            ice_servers=build_ice_servers(
                config.TURN_URL, config.TURN_USERNAME, config.TURN_PASSWORD
            ),
            turn_provider=cloudflare_turn.get_ice_servers if cloudflare_turn.enabled else None,
        )
        # The LAN viewer is a debug convenience — webapp viewers connect via
        # bus-relayed signaling, which works even if this bind fails.
        try:
            runner = await streaming.start()
        except OSError as exc:
            print(f"⚠ PC: local viewer server failed to bind {config.HOST_IP}:8080 — {exc}")

    # Bus setup
    bus = Bus(config.redis_host, config.redis_port, config.redis_password, config.redis_ssl)
    await bus.connect()

    async def on_ai_message(session_id, nickname, parts, payload):
        if session_id != session.session_id and session_id != config.ADMIN_SESSION_ID:
            return

        image_bytes = get_image_bytes_from_parts(parts)
        if not image_bytes:
            return

        kb = int(len(image_bytes) / 1024)

        if pending_images.full():
            try:
                pending_images.get_nowait()
                pending_images.task_done()
                print("⚠ PC: processing backlog full, dropped oldest pending image")
            except asyncio.QueueEmpty:
                pass

        await pending_images.put((nickname, image_bytes))
        print(f"✓ PC: queued image {kb} KB from {nickname}, pending queue size is now {pending_images.qsize()}")

    def process_image_payload(ray_shape_value, image_bytes, start_img, start_img_hires):
        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        img_b = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img_b is None:
            return {"ok": False, "error": "✗ PC: failed to decode received image"}

        img_b = cv2.resize(img_b, (config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        if ray_shape_value == 5:
            img_b = cv2.resize(img_b, (config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            img_b_hires = super.upscale(img_b)
            interpolated = of.interpolate(start_img_hires, img_b_hires, config.OF_FRAMES)
            generated_frames = [interp for interp in (interpolated or []) if interp is not None]
            generated_frames.append(img_b_hires)
            return {
                "ok": True,
                "mode": "hires",
                "target": img_b_hires,
                "frames": generated_frames,
            }

        interpolated = of.interpolate(start_img, img_b, config.OF_FRAMES)
        generated_frames = [interp for interp in (interpolated or []) if interp is not None]
        generated_frames.append(img_b)
        return {
            "ok": True,
            "mode": "normal",
            "target": img_b,
            "frames": generated_frames,
        }

    async def kick_processing():
        nonlocal img_a, img_a_hires, processing_task, ray_shape

        if processing_task is not None and not processing_task.done():
            return
        if pending_images.empty():
            return

        nickname, image_bytes = await pending_images.get()
        start_img = frames[-1] if frames else img_a
        start_img_hires = frames_hires[-1] if frames_hires else img_a_hires
        loop = asyncio.get_running_loop()

        async def runner():
            nonlocal img_a, img_a_hires
            try:
                result = await loop.run_in_executor(
                    processing_executor,
                    process_image_payload,
                    ray_shape,
                    image_bytes,
                    start_img,
                    start_img_hires,
                )
                if not result.get("ok"):
                    print(result.get("error", "✗ PC: image processing failed"))
                    return

                if result["mode"] == "hires":
                    for interp in result["frames"]:
                        frames_hires.append(interp)
                    img_a_hires = result["target"]
                    print(
                        f"✓ PC: processed hires image from {nickname}, appended {len(result['frames'])} frames, "
                        f"hires queue size is now {len(frames_hires)}, pending queue {pending_images.qsize()}"
                    )
                else:
                    for interp in result["frames"]:
                        frames.append(interp)
                    img_a = result["target"]
                    print(
                        f"✓ PC: processed image from {nickname}, appended {len(result['frames'])} frames, "
                        f"queue size is now {len(frames)}, pending queue {pending_images.qsize()}"
                    )
            finally:
                pending_images.task_done()

        processing_task = asyncio.create_task(runner())
       
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

    async def on_settings(params):
        nonlocal ray_shape, sim_go_back_on, sim_constraints_mode, sim_gradient_mode, sim_world_mode, overlay_on, fov_target
        nonlocal img_a, img_a_hires, brand_on, _brand_mask

        if 'shape' in params:
            new_shape = int(params['shape'])
            if new_shape != ray_shape:
                if new_shape == 5:
                    start_img = frames[-1] if frames else img_a
                    img_a_hires = super.upscale(start_img)
                    # Leftover hires frames from a previous flat-mode run would
                    # play first and the next journey would chain off them.
                    frames_hires.clear()
                else:
                    start_img = frames_hires[-1] if frames_hires else img_a_hires
                    img_a = cv2.resize(start_img, (config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
                    frames.clear()
                # In-flight images from the old mode would chain the new run
                # off stale content — drop them.
                while not pending_images.empty():
                    try:
                        pending_images.get_nowait()
                        pending_images.task_done()
                    except asyncio.QueueEmpty:
                        break
            ray_shape = new_shape
            print(f"✓ PC: ray shape set to {ray_shape}")

        if 'overlay_on' in params:
            overlay_on = bool(params['overlay_on'])
            print(f"✓ PC: overlay set to {overlay_on}")

        if 'brand_on' in params:
            brand_on = bool(params['brand_on'])
            _brand_mask = None  # invalidate cached mask on toggle
            director.set_brand_on(brand_on, Brand._BRANDED_THEMES)
            print(f"✓ PC: brand overlay set to {brand_on}")

        if 'go_back_on' in params:
            sim_go_back_on = bool(params['go_back_on'])
            print(f"✓ PC: sim go back set to {sim_go_back_on}")

        if 'constraints_mode' in params:
            sim_constraints_mode = int(params['constraints_mode'])
            print(f"✓ PC: sim constraints set to {sim_constraints_mode}")

        if 'gradient_mode' in params:
            sim_gradient_mode = int(params['gradient_mode'])
            print(f"✓ PC: sim gradient set to {sim_gradient_mode}")

        if 'world_mode' in params:
            sim_world_mode = int(params['world_mode'])
            print(f"✓ PC: sim world set to {sim_world_mode}")

        if 'zoom' in params:
            fov_target = float(params['zoom'])
            if not director.enabled:
                ray.fov = fov_target
            print(f"✓ PC: ray zoom target set to {fov_target:.1f}")

        if 'director_mode' in params:
            new_mode = params['director_mode']
            if new_mode == 'auto_play' and director.mode != 'auto_play':
                director.sync_from_state(ray_shape, sim_go_back_on, sim_constraints_mode, sim_gradient_mode, sim_world_mode, ray.fov)
                director.enable_auto_play()
                await bus.publish_settings(director_mode='auto_play')
            elif new_mode == 'auto_gen' and director.mode != 'auto_gen':
                director.enable_auto_gen()
                await bus.publish_settings(director_mode='auto_gen')
            elif new_mode == 'user' and director.enabled:
                # Director.disable() publishes the full state snapshot itself
                director.disable()

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

    async def on_user_joined(session_id, nickname):
        nonlocal overlay_on
        if str(session_id) != str(session.session_id) and str(session_id) != config.ADMIN_SESSION_ID:
            return
        if not nickname:
            return
        joined_users.add(nickname)
        print(f"✓ PC: user joined '{nickname}' (total {len(joined_users) - 1} human users)")
        if len(joined_users) > config.MAX_USERS:
            overlay_on = False
            await bus.publish_settings(overlay_on=False)
            print(f"✓ PC: overlay hidden — max users ({config.MAX_USERS}) reached")

    async def on_webrtc_offer(payload):
        session_id = str(payload.get("session_id") or "")
        print(f"✓ PC: webrtc_offer received session_id={session_id!r} expected={str(session.session_id)!r}")
        if session_id != str(session.session_id) and session_id != config.ADMIN_SESSION_ID:
            print(f"✗ PC: webrtc_offer dropped — session_id mismatch")
            return
        offer_id = payload.get("offer_id")
        sdp = payload.get("sdp")
        sdp_type = payload.get("type")
        if not offer_id or not sdp or not sdp_type:
            return

        nickname = payload.get("nickname") or "viewer"

        async def answer_task():
            try:
                answer = await streaming.answer_offer(sdp, sdp_type)
                await bus.publish_webrtc_answer(
                    offer_id=offer_id,
                    sdp=answer["sdp"],
                    sdp_type=answer["type"],
                )
                print(f"✓ PC: webrtc answer sent to {nickname} ({len(streaming.pcs)} viewers)")
            except Exception as exc:
                print(f"✗ PC: webrtc offer from {nickname} failed — {exc}")

        # ICE gathering takes a moment; don't block the render loop on it
        asyncio.create_task(answer_task())

    bus.on(Bus.AI_MESSAGE_TO_PC, on_ai_message)
    bus.on(Bus.USER_JOINED, on_user_joined)
    bus.on(Bus.USER_LIKE, on_user_like)
    bus.on(Bus.USER_VIDEO, on_user_video)
    bus.on(Bus.SETTINGS, on_settings)
    if streaming is not None:
        bus.on(Bus.WEBRTC_OFFER, on_webrtc_offer)

    # Session setup
    session = Session(config.URL)
    session.create_session()
    await bus.publish_session(session_id=session.session_id)
    qr_img = session.generate_qr_code()

    # Director — routes virtual mouse moves through the same Mouse callback as real input
    director = Director(
        bus, config,
        mouse_move_fn=lambda px, py: mouse.callback(cv2.EVENT_MOUSEMOVE, px, py, 0, None),
        session_getter=lambda: session.session_id,
    )
    director.start()
    director.enable_auto_play()

    # Time cadence
    ray_period = 1.0 / config.FPS
    sim_period = 1.0 / config.FPS_SIM
    now = time.perf_counter()
    next_ray_tick = now
    next_sim_tick = now
    while True:
        now = time.perf_counter()

        if director.enabled:
            director.tick(now)
            # Settings changes (shape, go_back, constraints, gradient, zoom) arrive via
            # on_settings() after Director publishes them to the bus.  Only per-frame
            # mouse state and the fov lerp are applied directly here.
            ray.fov += (fov_target - ray.fov) * 0.1

        sim_step_count = 0
        while now >= next_sim_tick and sim_step_count < config.MAX_SIM_STEPS_PER_LOOP:
            # UI
            if director.enabled:
                if director.ms_on:
                    pos = mouse.pos.copy()
                    pos[2] = sim.h + sim.r
                    sim.inject_mouse(pos, mouse.vel)
            else:
                if cam is not None:
                    if cam.update(now):
                        sim.inject_mouse(cam.pos, cam.vel)
                if mouse is not None:
                    if mouse.update(now):
                        sim.inject_mouse(mouse.pos, mouse.vel)
                if mic is not None:
                    bands, flux = mic.update()
                    if bands is not None:
                        sim.inject_audio(bands, flux)
            # Simulate
            sim.update_flip(
                go_back_on=sim_go_back_on,
                constraints_mode=sim_constraints_mode,
                gradient_mode=sim_gradient_mode,
                world_mode = sim_world_mode, world_center=config.world_center, world_radius=config.world_radius
            )
            sim_step_count += 1
            next_sim_tick += sim_period
        # If we fell too far behind, drop backlog instead of spiraling
        if now > next_sim_tick + sim_period * config.MAX_SIM_STEPS_PER_LOOP:
            next_sim_tick = now + sim_period
        # Raytrace only on visual FPS cadence
        if now < next_ray_tick:
            await asyncio.sleep(min(next_ray_tick - now, sim_period))
            continue
        # ── at ray FPS: poll bus and start any pending image processing ──
        await bus.poll(timeout=0)
        await kick_processing()
        next_ray_tick += ray_period
        if now > next_ray_tick + ray_period:
            next_ray_tick = now + ray_period

        if ray_shape == 5:
            if frames_hires:
                img_a_hires = frames_hires.popleft()
        else:
            if frames:
                img_a = frames.popleft()
                sim.new_image(img_a)

        # Raytrace
        if ray_shape == 0:
            frame = ray.triangle(sim.xyz, sim.rgb, sim.next_x, sim.next_y)
        elif ray_shape == 1:
            frame = ray.prism(sim.xyz, sim.rgb, sim.rot, 1.0 * sim.r, 1.0 * sim.r, 5.0 * sim.r)
        elif ray_shape == 2:
            frame = ray.ellipsoid(sim.xyz, sim.rgb, sim.rot, 3.0 * sim.r, 3.0 * sim.r, 0.5 * sim.r)  
            #frame = ray.pixel(sim.xyz, sim.rgb, 1.0 * sim.r)
        elif ray_shape == 3:
            frame = ray.cylinder(sim.xyz, sim.rgb, sim.next_y, 0.5 * sim.r)
        elif ray_shape == 4:
            frame = ray.sphere(sim.xyz, sim.rgb, 4.0 * sim.r)
        else:
            frame = resize_to_fit_window(img_a_hires, config.WINDOW_W, config.WINDOW_H)

        if brand_on:
            if _brand_mask is None:
                _brand_mask = brand.render_mask(config.WINDOW_W, config.WINDOW_H)
            frame = Brand.composite_mask_over_frame(frame, _brand_mask)

        thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        gif_changed = (
            gif_last_frame is None or
            np.mean(np.abs(thumb.astype(np.float32) - gif_last_frame.astype(np.float32))) > config.GIF_DIFF_THRESHOLD
        )
        if gif_changed:
            last_frames.append(thumb)
            gif_last_frame = thumb
        if frame_bus is not None and gif_changed:
            frame_bus.publish(frame)

        if overlay_on:
            out = overlay(frame, qr_img, proportion=20, alignment="bottom center")
        else:
            out = frame
        cv2.imshow(config.APP_NAME, out)

        key = cv2.waitKeyEx(1)
        if key in (ord('q'), 27):
            break
        elif key == ord('d'):
            current_mode = director.mode  # None | "auto_play" | "auto_gen"
            if current_mode is None:
                director.sync_from_state(ray_shape, sim_go_back_on, sim_constraints_mode, sim_gradient_mode, sim_world_mode, ray.fov)
                director.enable_auto_play()
                await bus.publish_settings(director_mode='auto_play')
            elif current_mode == "auto_play":
                director.enable_auto_gen()
                await bus.publish_settings(director_mode='auto_gen')
            else:  # auto_gen → user
                # Director.disable() publishes the full state snapshot itself
                director.disable()

    cv2.destroyAllWindows()
    if streaming is not None:
        await streaming.on_shutdown(None)
    if runner is not None:
        await runner.cleanup()
    if cam is not None:
        cam.close()
    if mic is not None:
        mic.stop()
    if processing_task is not None:
        await asyncio.gather(processing_task, return_exceptions=True)
    processing_executor.shutdown(wait=False, cancel_futures=True)
    await bus.close()

if __name__ == '__main__':
    asyncio.run(main())
