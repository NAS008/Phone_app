# Test painter: GA brushstroke painting displayed via raytrace_sphere

import cv2
import time
import asyncio
import ctypes
ctypes.windll.user32.SetProcessDPIAware()

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'shared'))
from config import Config

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'server'))
from ai import Folder
from painter import Painter
from ray import RayTracer

async def main():
    config = Config()
    # folder = Folder(config.IMAGE_W, config.IMAGE_H, config.INPUT_FOLDER)
    # image  = folder.load_image()
    image = cv2.imread(r"..\..\output\noncarbon-artwork-1780049753086.png")
    BRUSH_PATH = r"..\..\brand\brush03.png"

    MAX_DIM = 386   # painter working resolution (GA runs at this size)

    painter = Painter(
        canvas_w        = config.WINDOW_W,
        canvas_h        = config.WINDOW_H,
        population      = 1024,
        n_strokes       = 500,
        gens_per_stroke = 60,
        elite_n         = 20,
        mutation_rate   = 0.05,
        n_colors        = 256,
        max_brush_hw    = 32,
        stroke_alpha    = 0.95,
        seed_frac       = 0.6,
        explore_r       = 0.20,
        max_dim         = MAX_DIM,
        impasto_dz_frac = 0.3,
        device          = "cuda",
        seed            = 42,
    )
    painter.init_image(image)
    painter.init_brush(BRUSH_PATH)
    painter.init_canvas()

    # Use a shallow G.z=4 so the depth formula gives bright values for
    # floor-level particles (z≈painter.z_active). With G.z=32 (config default)
    # floor particles read as depth≈0.34; with G.z=4 they read as depth≈0.58
    # and rise toward 1.0 as impasto accumulates.
    G_painter = [config.G[0], config.G[1], 4]

    ray = RayTracer(
        W           = config.WINDOW_W,
        H           = config.WINDOW_H,
        G           = G_painter,
        camera      = config.camera,
        target      = config.target,
        light       = config.light,
        fov         = config.fov,
        samples     = config.samples,
        background  = config.background,
        ambient     = config.ambient,
        shadow      = config.shadow,
        n_particles = painter.particles,   # pre-allocate grid particle_ids
    )

    cv2.namedWindow(config.APP_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(config.APP_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(config.APP_NAME, config.WINDOW_W, config.WINDOW_H)

    t_last = time.perf_counter()

    while True:
        # Paint one stroke (blocks briefly — GA takes ~100ms at pop=1024, gens=60)
        if not painter.done:
            stroke = painter.update()
            if stroke:
                dt = time.perf_counter() - t_last
                t_last = time.perf_counter()
                print(
                    f"stroke {stroke['step']:4d}/{painter.n_strokes}"
                    f"  active {stroke['active_particles']:6,}"
                    f"  mse {stroke['mse']:.5f}"
                    f"  {dt*1000:.0f}ms"
                )

        # Raytrace current particle state and display
        frame = ray.sphere(painter.xyz, painter.rgb, 3.0 * painter.r)
        cv2.imshow(config.APP_NAME, frame)

        key = cv2.waitKeyEx(1)
        if key in (ord('q'), 27):
            break

        await asyncio.sleep(0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    asyncio.run(main())
