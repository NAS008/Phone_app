# Ray at low res 512 x 512 and then super res

import cv2
import numpy as np
import asyncio
import ctypes
import time
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'shared'))
from config import Config
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'server'))
from ray import RayTracer
from sim import Simulator
from ui import Mouse
from sd35 import Folder, SuperResolution
ctypes.windll.user32.SetProcessDPIAware()

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
        G=config.G, L=4,
        smooth=15,
    )
    folder = Folder(config.IMAGE_W, config.IMAGE_H, config.INPUT_FOLDER)
    img = folder.load_image()
    sim.new_image(img)

    # Ray tracer setup
    factor = 1
    ray = RayTracer(
        W=config.WINDOW_W // factor,
        H=config.WINDOW_H // factor,
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

    super = SuperResolution(config.MODELS_FOLDER)

    ms = Mouse(1.0 / config.FPS, config.WINDOW_W, config.WINDOW_H)

    # Window setup
    cv2.namedWindow(config.APP_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(config.APP_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(config.APP_NAME, config.WINDOW_W, config.WINDOW_H)
    cv2.setMouseCallback(config.APP_NAME, ms.callback)

    frame_dur = 1.0 / config.FPS
    next_tick = time.perf_counter()
    while True:

        sim.inject_gradient()

        if ms.on:
            sim.inject_mouse(ms.pos, ms.vel)
        sim.update(constraints_on=True, go_back_on=True)

        now = time.perf_counter()
        if now < next_tick:
            await asyncio.sleep(next_tick - now)
            continue

        next_tick += frame_dur
        if now > next_tick + frame_dur:
            next_tick = now + frame_dur

        if ray_shape == 0:
            frame = ray.quad(sim.xyz, sim.rgb, sim.rot, 1.0 * sim.r)
        elif ray_shape == 1:
            frame = ray.prism(sim.xyz, sim.rgb, sim.rot, 1.0 * sim.r, 0.25 * sim.r, 2.75 * sim.r)
        elif ray_shape == 2:
            frame = ray.ellipsoid(sim.xyz, sim.rgb, sim.rot, 1.0 * sim.r, 1.0 * sim.r, 0.3 * sim.r)
        elif ray_shape == 3:
            frame = ray.cylinder(sim.xyz, sim.rgb, sim.next_y, 0.3 * sim.r)
        elif ray_shape == 4:
            frame = ray.sphere(sim.xyz, sim.rgb, 1.0 * sim.r)
        else:
            frame = resize_to_fit_window(img, config.WINDOW_W // factor, config.WINDOW_H // factor)  

        #out = super.super_image(frame)
        out = resize_to_fit_window(frame, config.WINDOW_W, config.WINDOW_H)
        cv2.imshow(config.APP_NAME, out)

        key = cv2.waitKeyEx(1)
        if key in (ord('q'), 27):
            break
        elif key == ord('n'):
            img = folder.load_image()
            sim.new_image(img)
        elif key == ord('r'):
            ray_shape += 1
            if ray_shape >= 6:
                ray_shape = 0
        elif key == ord('z'):
            ray.fov -= 0.1
            if ray.fov < 0.1:
                ray.fov = 2.0

    cv2.destroyAllWindows()

if __name__ == '__main__':
    asyncio.run(main())
