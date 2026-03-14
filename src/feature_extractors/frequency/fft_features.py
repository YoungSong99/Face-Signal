import cv2
import numpy as np

from src.utils import safe_mean, safe_std, safe_var, safe_entropy


class FFTFeatureExtractor:
    def _fft_stats(self, gray_img: np.ndarray) -> dict[str, float]:
        gray_f = gray_img.astype(np.float32)

        fft = np.fft.fft2(gray_f)
        fft_shift = np.fft.fftshift(fft)

        magnitude = np.abs(fft_shift)
        power = magnitude ** 2
        log_mag = np.log1p(magnitude)

        h, w = gray_img.shape
        cy, cx = h // 2, w // 2

        y, x = np.ogrid[:h, :w]
        r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

        r_norm = r / (r.max() + 1e-8)

        low_mask = r_norm <= 0.15
        mid_mask = (r_norm > 0.15) & (r_norm <= 0.5)
        high_mask = r_norm > 0.5

        total_energy = float(np.sum(power)) + 1e-8
        low_energy = float(np.sum(power[low_mask]))
        mid_energy = float(np.sum(power[mid_mask]))
        high_energy = float(np.sum(power[high_mask]))

        spectral_centroid = float(np.sum(r_norm * power) / total_energy)

        radial_profile = self._radial_profile(power, num_bins=32)
        radial_profile = radial_profile.astype(np.float64)
        radial_profile /= (radial_profile.sum() + 1e-12)

        power_prob = power.ravel().astype(np.float64)
        power_prob /= (power_prob.sum() + 1e-12)

        peak_thresh = log_mag.mean() + 2.5 * log_mag.std()
        peak_mask = log_mag > peak_thresh
        outer_peak_mask = peak_mask & (r_norm > 0.2)

        return {
            "fft_mean": safe_mean(log_mag),
            "fft_std": safe_std(log_mag),
            "fft_var": safe_var(log_mag),
            "fft_spectral_entropy": safe_entropy(power_prob),
            "fft_low_ratio": low_energy / total_energy,
            "fft_mid_ratio": mid_energy / total_energy,
            "fft_high_ratio": high_energy / total_energy,
            "fft_radial_entropy": safe_entropy(radial_profile),
            "fft_spectral_centroid": spectral_centroid,
            "fft_peak_count": int(np.sum(peak_mask)),
            "fft_outer_peak_ratio": float(np.mean(outer_peak_mask)),
        }

    def _radial_profile(self, power: np.ndarray, num_bins: int = 32) -> np.ndarray:
        h, w = power.shape
        cy, cx = h // 2, w // 2

        y, x = np.ogrid[:h, :w]
        r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        r_norm = r / (r.max() + 1e-8)

        bins = np.linspace(0.0, 1.0, num_bins + 1)
        radial = np.zeros(num_bins, dtype=np.float64)

        for i in range(num_bins):
            mask = (r_norm >= bins[i]) & (r_norm < bins[i + 1])
            if np.any(mask):
                radial[i] = power[mask].mean()

        return radial
