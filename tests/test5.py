# Test branded overlay

import cv2
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'shared'))
from config import Config
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'server'))
from file import File
from brand import Brand

config = Config()
brand = Brand()
folder = File(image_w=config.IMAGE_W, image_h=config.IMAGE_H, input_folder=config.INPUT_FOLDER)
img = folder.load_image()
img = brand.resize_to_fit_window(img, config.WINDOW_W, config.WINDOW_H)

mask = brand.render_mask(
    config.WINDOW_W,
    config.WINDOW_H,
)

overlayed = brand.composite_mask_over_frame(img, mask, strength=1.0)
cv2.imwrite(r"../../output/overlayed.png", overlayed) 
s = "Não se fala de dinheiro à mesa, mas devia."
print(repr(s))