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

    gray = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
    return np.clip(gray, 0.0, 1.0).astype(np.float32)


def load_rgb_and_gray(path):
    bgr = read_bgr(path)

    if bgr is None:
        return None, None

    rgb = bgr_to_rgb_float(bgr)
    gray = rgb_float_to_gray(rgb)

    return rgb, gray


def rgb_to_uint8(image):
    img = np.asarray(image)
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    if img.max() <= 1.0:
        img = img * 255.0
    img = np.clip(img, 0, 255).round().astype(np.uint8)
    return img


def to_uint8_vis(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)
