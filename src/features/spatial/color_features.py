import cv2
import numpy as np
from src.utils.stats_utils import safe_mean, safe_std, safe_var, safe_entropy, safe_uniformity

class ColorFeatureExtractor:
    def _color_stats(self, image):
        features = {}

        features.update(self._rgb_stats(image))
        features.update(self._hsv_stats(image))
        features.update(self._lab_stats(image))
        features.update(self._ycrcb_stats(image))
        features.update(self._colorfulness(image))
        features.update(self._color_entropy(image))
        features.update(self._chroma_noise(image))
        features.update(self._lab_chroma_variance(image))

        return features

    def _rgb_stats(self, image):
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]

        return {
            "rgb_mean_r": safe_mean(r),
            "rgb_mean_g": safe_mean(g),
            "rgb_mean_b": safe_mean(b),
            "rgb_std_r": safe_std(r),
            "rgb_std_g": safe_std(g),
            "rgb_std_b": safe_std(b)
        }

    def _hsv_stats(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        return {
            "hsv_mean_h": safe_mean(h),
            "hsv_mean_s": safe_mean(s),
            "hsv_mean_v": safe_mean(v),

            "hsv_std_h": safe_std(h),
            "hsv_std_s": safe_std(s),
            "hsv_std_v": safe_std(v),
        }

    def _lab_stats(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        l = lab[:, :, 0]
        a = lab[:, :, 1]
        b = lab[:, :, 2]

        return {
            "lab_mean_l": safe_mean(l),
            "lab_mean_a": safe_mean(a),
            "lab_mean_b": safe_mean(b),

            "lab_std_l": safe_std(l),
            "lab_std_a": safe_std(a),
            "lab_std_b": safe_std(b),
        }

    def _ycrcb_stats(self, image):
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

        y = ycrcb[:, :, 0]
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]

        return {
            "ycrcb_mean_y": safe_mean(y),
            "ycrcb_mean_cr": safe_mean(cr),
            "ycrcb_mean_cb": safe_mean(cb),

            "ycrcb_std_y": safe_std(y),
            "ycrcb_std_cr": safe_std(cr),
            "ycrcb_std_cb": safe_std(cb),
        }

    # reference: https://pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
    def _colorfulness(self, image):
        r = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        b = image[:, :, 2].astype(np.float32)

        rg = r - g
        yb = 0.5 * (r + g) - b

        std_rg = np.std(rg)
        std_yb = np.std(yb)

        mean_rg = np.mean(rg)
        mean_yb = np.mean(yb)

        std_root = np.sqrt(std_rg ** 2 + std_yb ** 2)
        mean_root = np.sqrt(mean_rg ** 2 + mean_yb ** 2)

        return {
            "colorfulness": float(std_root + 0.3 * mean_root)
        }

    def _color_entropy(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]

        hist = cv2.calcHist([saturation], [0], None, [256], [0, 256])
        hist = hist / np.sum(hist)

        entropy = -np.sum(hist * np.log2(hist + 1e-8))

        return {
            "color_entropy": float(entropy)
        }

    def _chroma_noise(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        a = lab[:, :, 1].astype(np.float32)
        b = lab[:, :, 2].astype(np.float32)

        a_blur = cv2.GaussianBlur(a, (5, 5), 0)
        b_blur = cv2.GaussianBlur(b, (5, 5), 0)

        a_noise = a - a_blur
        b_noise = b - b_blur

        chroma_noise = np.sqrt(
            safe_var(a_noise) + safe_var(b_noise)
        )

        return {
            "chroma_noise": chroma_noise
        }

    def _lab_chroma_variance(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        a = lab[:, :, 1]
        b = lab[:, :, 2]

        chroma = np.sqrt(a.astype(np.float32) ** 2 + b.astype(np.float32) ** 2)

        return {
            "lab_chroma_var": safe_var(chroma),
            "lab_chroma_mean": safe_mean(chroma)
        }
