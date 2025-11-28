from pygetwindow import getWindowsWithTitle
from mss import mss
import numpy as np
from PIL import Image

def get_clip_studio_paint_window():
    windows = getWindowsWithTitle('Clip Studio Paint')
    if windows:
        window = windows[0]
        return {'top': window.top, 'left': window.left, 'width': window.width, 'height': window.height}
    return None

def capture_screen(region):
    with mss() as sct:
        screenshot = sct.grab(region)
        img = Image.frombytes('RGB', (screenshot.width, screenshot.height), screenshot.rgb)
        return np.array(img)
