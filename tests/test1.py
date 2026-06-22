# Simply test sim and ray code

import cv2
import numpy as np
import time
import asyncio
import ctypes
ctypes.windll.user32.SetProcessDPIAware()

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'shared'))

from config import Config
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'server'))
from ray import RayTracer
from sim import Simulator
from ui import Mouse
from file import File

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

    # Simulator setup
    sim = Simulator(
        IMAGE_W=config.IMAGE_W,
        IMAGE_H=config.IMAGE_H,
        PIXELS_PER_CELL=config.PIXELS_PER_CELL,
        G=config.G, L=config.LAYERS,
        smooth=15,
        dt = 1.0 / config.FPS,
    )
    folder = File(config.IMAGE_W, config.IMAGE_H, config.INPUT_FOLDER)
    img = folder.load_image()
    sim.new_image(img, depth_factor=0.0)
    sim_constraints_mode = 0
    sim_gradient_mode = 0
    sim_world_mode = 0
    sim_goback_on = False

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
    ray_shape = 0

    ms = Mouse(1.0 / config.FPS, config.WINDOW_W, config.WINDOW_H)

    # Window setup
    cv2.namedWindow(config.APP_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(config.APP_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(config.APP_NAME, config.WINDOW_W, config.WINDOW_H)
    cv2.setMouseCallback(config.APP_NAME, ms.callback)

    frame_dur = 1.0 / config.FPS
    next_tick = time.perf_counter()
    while True:

        # Simulate
        if ms.on:
            sim.inject_mouse(ms.pos, ms.vel)
        
        sim.update_flip(
            go_back_on=sim_goback_on,
            constraints_mode=sim_constraints_mode,
            gradient_mode=sim_gradient_mode,
            world_mode = sim_world_mode, world_center=config.world_center, world_radius=config.world_radius
        )

        # Keep FPS cadence on raytracing to save CPU, since it's the bottleneck
        now = time.perf_counter()
        if now < next_tick:
            await asyncio.sleep(next_tick - now)
            continue
        next_tick += frame_dur
        if now > next_tick + frame_dur:
            next_tick = now + frame_dur

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

        cv2.imshow(config.APP_NAME, frame)

        key = cv2.waitKeyEx(1)
        if key in (ord('q'), 27):
            break
        elif key == ord('b'):
            sim_goback_on = not sim_goback_on
        elif key == ord('c'):
            sim_constraints_mode += 1
            if sim_constraints_mode >= 3:
                sim_constraints_mode = 0
        elif key == ord('g'):
            sim_gradient_mode += 1
            if sim_gradient_mode >= 3:
                sim_gradient_mode = 0
        elif key == ord('w'):
            sim_world_mode += 1
            if sim_world_mode >= 3:
                sim_world_mode = 0
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

    cv2.destroyAllWindows()

if __name__ == '__main__':
    asyncio.run(main())
