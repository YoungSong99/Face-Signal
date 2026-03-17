import cv2
import numpy as np
from src.utils.stats_utils import safe_mean, safe_std, safe_var


class DCTFeatureExtractor:
    def _dct_stats(self, gray_img: np.ndarray) -> dict[str, float]:
        gray_f = gray_img.astype(np.float32) / 255.0
        dct = cv2.dct(gray_f)
        dct_abs = np.abs(dct)

        h, w = dct_abs.shape

        y, x = np.ogrid[:h, :w]
        yn = y / max(h - 1, 1)
        xn = x / max(w - 1, 1)
        dist = np.sqrt(yn ** 2 + xn ** 2)

        dc = float(dct_abs[0, 0])

        ac = dct_abs.copy()
        ac[0, 0] = 0.0

        total_energy = float(np.sum(dct_abs ** 2)) + 1e-8
        ac_energy = float(np.sum(ac ** 2))

        low_mask = dist <= 0.2
        mid_mask = (dist > 0.2) & (dist <= 0.6)
        high_mask = dist > 0.6

        low_energy = float(np.sum((dct_abs[low_mask]) ** 2))
        mid_energy = float(np.sum((dct_abs[mid_mask]) ** 2))
        high_energy = float(np.sum((dct_abs[high_mask]) ** 2))

        ac_vals = ac.ravel()
        ac_vals = ac_vals[ac_vals > 0]

        return {
            "dct_dc": dc,
            "dct_ac_mean": safe_mean(ac_vals),
            "dct_ac_std": safe_std(ac_vals),
            "dct_ac_var": safe_var(ac_vals),
            "dct_dc_ratio": (dc ** 2) / total_energy,
            "dct_ac_ratio": ac_energy / total_energy,
            "dct_low_ratio": low_energy / total_energy,
            "dct_mid_ratio": mid_energy / total_energy,
            "dct_high_ratio": high_energy / total_energy,
        }


