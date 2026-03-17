import numpy as np
from src.features.frequency.fft_features import fft

class CheckerboardFeatureExtractor:

    def _checkerboard_stats(self, gray: np.ndarray) -> dict:
        
        features = {}

        power, cy, cx = self._checkerboard_fft_power(gray)
        features.update(self._checkerboard_peak_stats(power, cy, cx))
        
        return features


    def _checkerboard_fft_power(self, gray_img):
        
        _, power, _ = fft(gray_img)
        
        h, w = gray_img.shape
        cy, cx = h // 2, w // 2
        
        return power, cy, cx


    def _checkerboard_peak_stats(self, power, cy, cx, periods=(2, 4, 8, 16)):
        
        h, w = power.shape
        total_energy = float(power.mean() + 1e-10)
        checkerboard_dict = {}

        for period in periods:
            freq_y = int((1.0 / period) * h)
            freq_x = int((1.0 / period) * w)

            if freq_y < 1 or freq_x < 1:
                continue

            r = max(1, min(period // 2, freq_y - 1, freq_x - 1))
            peaks = []

            for dy in (-freq_y, freq_y):
                for dx in (-freq_x, freq_x):
                    py = np.clip(cy + dy, r, h - r - 1)
                    px = np.clip(cx + dx, r, w - r - 1)
                    patch = power[py - r:py + r + 1, px - r:px + r + 1]
                    peaks.append(float(np.mean(patch)))

            if peaks and not any(np.isnan(p) for p in peaks):
                peak_mean = float(np.mean(peaks))
                checkerboard_dict[f"cb_period{period}_peak_mean"] = peak_mean
                checkerboard_dict[f"cb_period{period}_peak_ratio"] = peak_mean / total_energy

        return checkerboard_dict