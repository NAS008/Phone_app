# Simply text sim and ray code

import cv2
import numpy as np
import asyncio
import ctypes
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'shared'))
from config import Config
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'server'))
from ray import RayTracer
from sim import Simulator
from ui import Mouse
from sd35 import Folder
ctypes.windll.user32.SetProcessDPIAware()
    
async def main():
    config = Config()

    # Simulator setup
    sim = Simulator(
        IMAGE_SIZE=config.IMAGE_SIZE,
        PIXELS_PER_CELL=config.PIXELS_PER_CELL,
        G=config.G, L=3,
        smooth=15,
    )
    folder = Folder(config.IMAGE_SIZE, config.INPUT_FOLDER)
    img = folder.load_image()
    sim.new_image(img)

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

    ms = Mouse(config.WINDOW_W, config.WINDOW_H)

    # Window setup
    cv2.namedWindow(config.APP_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(config.APP_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(config.APP_NAME, config.WINDOW_W, config.WINDOW_H)
    cv2.setMouseCallback(config.APP_NAME, ms.mouse_callback)

    while True:

        if ray_shape == 5:
            target_w = config.WINDOW_W
            target_h = config.WINDOW_H

            img_h, img_w = img.shape[:2]

            scale = min(float(target_w) / float(img_w), float(target_h) / float(img_h))
            new_w = max(1, int(round(img_w * scale)))
            new_h = max(1, int(round(img_h * scale)))

            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            frame = np.zeros((target_h, target_w, 3), dtype=img.dtype)

            x0 = (target_w - new_w) // 2
            y0 = (target_h - new_h) // 2

            frame[y0:y0 + new_h, x0:x0 + new_w] = resized
        else:
            sim.inject_gradient()

            if ms.on:
                sim.inject_mouse(
                    np.array([ms.mouse_x, ms.mouse_y, 0.0]),
                    20.0 * np.array([ms.mouse_vx, ms.mouse_vy, ms.mouse_vz])
                )
            sim.update(constraints_on=False, go_back_on=True)

            if ray_shape == 0:
                frame = ray.quad(sim.xyz, sim.rgb, sim.rot, 1.0 * sim.r)
            elif ray_shape == 1:
                frame = ray.prism(sim.xyz, sim.rgb, sim.rot, 2.0 * sim.r, 0.5 * sim.r, 4.5 * sim.r)
            elif ray_shape == 2:
                frame = ray.ellipsoid(sim.xyz, sim.rgb, sim.rot, 1.4 * sim.r, 1.4 * sim.r, 0.6 * sim.r)
            elif ray_shape == 3:
                frame = ray.cylinder(sim.xyz, sim.rgb, sim.next_y, 0.5 * sim.r)
            else:
                frame = ray.sphere(sim.xyz, sim.rgb, 1.4 * sim.r)

        cv2.imshow(config.APP_NAME, frame)

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
