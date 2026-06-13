# Test gradient wind

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

async def main():
    config = Config()
    folder = Folder(config.IMAGE_SIZE, config.INPUT_FOLDER)
    image = folder.load_image()
    BRUSH_PATH  = r"..\..\brand\brush03.png"      # greyscale, black = paint

    painter = Painter(
        canvas_w        = config.WINDOW_W,
        canvas_h        = config.WINDOW_H,
        population      = 1024,
        n_strokes       = 1000,
        gens_per_stroke = 60,
        elite_n         = 20,
        mutation_rate   = 0.05,
        n_colors        = 256,
        max_brush_hw    = 32,
        stroke_alpha    = 0.95,
        seed_frac       = 0.6,
        explore_r       = 0.20,
        max_dim         = 256,
        device          = "cuda",
        seed            = 42,
    )
    painter.init_image(image)
    painter.init_brush(BRUSH_PATH)
    painter.init_canvas()

    # Window setup
    cv2.namedWindow(config.APP_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(config.APP_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(config.APP_NAME, config.WINDOW_W, config.WINDOW_H)

    period = 1.0 / config.FPS_SIM
    next_paint_tick = time.perf_counter()
    while True:
        now = time.perf_counter()

        if now >= next_paint_tick:
            if not painter.done:
                painter.update()
            next_paint_tick += period

            if now - next_paint_tick > period:
                next_paint_tick = now + period

        frame = painter.get_canvas_bgr()
        cv2.imshow(config.APP_NAME, frame)

        key = cv2.waitKeyEx(1)
        if key in (ord('q'), 27):
            break

        await asyncio.sleep(0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    asyncio.run(main())