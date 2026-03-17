import cv2
import numpy as np


def read_bgr(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def bgr_to_rgb_float(img):
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def rgb_float_to_gray(img):
    if img is None:
        return None

    img = np.asarray(img).astype(np.float32)

    if img.ndim == 2:
        return img

    return 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]


def load_rgb_and_gray(path):
    bgr = read_bgr(path)

    if bgr is None:
        return None, None

    rgb = bgr_to_rgb_float(bgr)
    gray = rgb_float_to_gray(rgb)

    return rgb, gray
