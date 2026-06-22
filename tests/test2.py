# Test video streaming to a connected TV
# Check the IPv4 address to update in 'host' parameter

import asyncio
import ctypes
import sys as _sys
import os as _os
import time
from pathlib import Path

import cv2
import numpy as np

ctypes.windll.user32.SetProcessDPIAware()

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'shared'))
from config import Config

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'server'))
from ray import RayTracer
from sim import Simulator
from ui import Mouse
from file import File
from stream import FrameBus, StreamingServer

BASE_DIR = Path(__file__).resolve().parent
VIEWER_HTML = BASE_DIR / "viewer.html"

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

    sim = Simulator(
        IMAGE_W=config.IMAGE_W,
        IMAGE_H=config.IMAGE_H,
        PIXELS_PER_CELL=config.PIXELS_PER_CELL,
        G=config.G,
        L=config.LAYERS,
        smooth=3,
        dt = 1.0 / config.FPS_SIM,
    )
    folder = File(config.IMAGE_W, config.IMAGE_H, config.INPUT_FOLDER)
    img = folder.load_image()
    sim.new_image(img)
    sim_gradient_on = False
    sim_gradient_mode = 0
    sim_goback_on = False
    sim_constraints_on = True
    sim_flip_on = True

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

    ms = Mouse(1.0 / config.FPS_SIM, config.WINDOW_W, config.WINDOW_H)

    cv2.namedWindow(config.APP_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(config.APP_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(config.APP_NAME, config.WINDOW_W, config.WINDOW_H)
    cv2.setMouseCallback(config.APP_NAME, ms.callback)

    frame_bus = FrameBus()
    streaming = StreamingServer(
        frame_bus=frame_bus,
        W=config.WINDOW_W,
        H=config.WINDOW_H,
        viewer_html=VIEWER_HTML,
        host="192.168.68.61",
        port=8080,
    )
    runner = await streaming.start()

    ray_period = 1.0 / config.FPS
    sim_period = 1.0 / config.FPS_SIM

    now = time.perf_counter()
    next_ray_tick = now
    next_sim_tick = now

    sim_steps = 0
    try:
        while True:
            # Run sim at a bounded multiple of ray FPS
            now = time.perf_counter()
            sim_step_count = 0
            while now >= next_sim_tick and sim_step_count < config.MAX_SIM_STEPS_PER_LOOP:
                if sim_gradient_on:
                    sim.inject_gradient(mode=sim_gradient_mode)
                if ms.on:
                    sim.inject_mouse(ms.pos, ms.vel)
                sim.update(
                    constraints_on=sim_constraints_on,
                    constraints_mode=1,
                    go_back_on=sim_goback_on,
                    flip_on=sim_flip_on)
                sim_steps += 1
                sim_step_count += 1
                next_sim_tick += sim_period
            # If we fell too far behind, drop backlog instead of spiraling
            if now > next_sim_tick + sim_period * config.MAX_SIM_STEPS_PER_LOOP:
                next_sim_tick = now + sim_period

            # Raytrace only on visual FPS cadence
            if now < next_ray_tick:
                await asyncio.sleep(min(next_ray_tick - now, sim_period))
                continue

            next_ray_tick += ray_period
            if now > next_ray_tick + ray_period:
                next_ray_tick = now + ray_period

            # Raytrace
            if ray_shape == 0:
                frame = ray.quad(sim.xyz, sim.rgb, sim.rot, 1.0 * sim.r)
            elif ray_shape == 1:
                frame = ray.prism(sim.xyz, sim.rgb, sim.rot, 1.0 * sim.r, 1.0 * sim.r, 8.0 * sim.r)
            elif ray_shape == 2:
                frame = ray.ellipsoid(sim.xyz, sim.rgb, sim.rot, 4.0 * sim.r, 4.0 * sim.r, 1.0 * sim.r)
            elif ray_shape == 3:
                frame = ray.cylinder(sim.xyz, sim.rgb, sim.next_y, 0.4 * sim.r)
            elif ray_shape == 4:
                frame = ray.sphere(sim.xyz, sim.rgb, 2.0 * sim.r)
            else:
                frame = resize_to_fit_window(img, config.WINDOW_W, config.WINDOW_H)  

            frame_bus.publish(frame)
            cv2.imshow(config.APP_NAME, frame)

            key = cv2.waitKeyEx(1)
            if key in (ord('q'), 27):
                break
            elif key == ord('b'):
                sim_goback_on = not sim_goback_on
            elif key == ord('c'):
                sim_constraints_on = not sim_constraints_on
            elif key == ord('g'):
                sim_gradient_on = not sim_gradient_on
            elif key == ord('n'):
                img = folder.load_image()
                sim.new_image(img, depth_factor=0.0)
            elif key == ord('r'):
                ray_shape += 1
                if ray_shape >= 6:
                    ray_shape = 0
            elif key == ord('z'):
                ray.fov -= 0.1
                if ray.fov < 0.05:
                    ray.fov = 1.1

            await asyncio.sleep(0)
    finally:
        cv2.destroyAllWindows()
        await runner.cleanup()

if __name__ == '__main__':
    asyncio.run(main())
