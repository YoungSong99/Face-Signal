import numpy as np
import pywt

from src.utils.stats_utils import safe_mean, safe_std, safe_var, safe_entropy


class DWTFeatureExtractor:
    def _dwt_stats(self, gray_img: np.ndarray) -> dict[str, float]:
        gray_f = gray_img.astype(np.float32) / 255.0
        gray_f = self._pad_to_even(gray_f)

        ll, (lh, hl, hh) = pywt.dwt2(gray_f, "haar")

        ll_energy = float(np.sum(ll ** 2))
        lh_energy = float(np.sum(lh ** 2))
        hl_energy = float(np.sum(hl ** 2))
        hh_energy = float(np.sum(hh ** 2))

        high_energy = lh_energy + hl_energy + hh_energy
        total_energy = ll_energy + high_energy + 1e-8

        subband = np.array([ll_energy, lh_energy, hl_energy, hh_energy], dtype=np.float64)
        subband /= (subband.sum() + 1e-12)

        return {
            "wavelet_ll_energy": ll_energy,
            "wavelet_lh_energy": lh_energy,
            "wavelet_hl_energy": hl_energy,
            "wavelet_hh_energy": hh_energy,
            "wavelet_ll_ratio": ll_energy / total_energy,
            "wavelet_high_ratio": high_energy / total_energy,
            "wavelet_subband_entropy": safe_entropy(subband),

            "wavelet_lh_mean": safe_mean(np.abs(lh)),
            "wavelet_hl_mean": safe_mean(np.abs(hl)),
            "wavelet_hh_mean": safe_mean(np.abs(hh)),

            "wavelet_lh_std": safe_std(lh),
            "wavelet_hl_std": safe_std(hl),
            "wavelet_hh_std": safe_std(hh),

            "wavelet_lh_var": safe_var(lh),
            "wavelet_hl_var": safe_var(hl),
            "wavelet_hh_var": safe_var(hh),
        }

    def _pad_to_even(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        pad_h = h % 2
        pad_w = w % 2

        if pad_h == 0 and pad_w == 0:
            return img
        
        padded_img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")
        
        return padded_img
