import cv2
import numpy as np
from src.utils.stats_utils import safe_mean, safe_std, safe_entropy


def fft(gray_f):
    dft = np.fft.fft2(gray_f)
    dft_shift = np.fft.fftshift(dft)

    magnitude = np.abs(dft_shift)
    power = magnitude ** 2
    log_magnitude = np.log(magnitude + 1e-8)

    return magnitude, power, log_magnitude


class FFTFeatureExtractor:
    def fft_features(self, gray_img):
        
        features = {}
        
        magnitude, power, log_magnitude = fft(gray_img)
        polar, radial_profile, angular_profile = self._polar_spectrum(log_magnitude)
        log_freq, log_power, slope, intercept = self._frequency_slope(radial_profile)

        features.update(self._radial_distribution(radial_profile, n_bands=10))
        features.update(self._angular_distribution(angular_profile, n_bands=10))
        features.update(self._spectrum_stats(log_magnitude, slope))

        return features
    
    
    def _polar_spectrum(self, log_magnitude):
        
        h, w = log_magnitude.shape
        center = (w // 2, h // 2)
        max_radius = min(center)
        
        polar = cv2.linearPolar(log_magnitude, center, max_radius, cv2.WARP_FILL_OUTLIERS)
        radial_profile = np.mean(polar, axis=0)
        angular_profile = np.mean(polar, axis=1)
        
        return polar, radial_profile, angular_profile
    
    
    def _frequency_slope(self, radial_profile):
        
        freq = np.arange(1, len(radial_profile) + 1)
        log_freq = np.log(freq)
        power = radial_profile + 1e-8
        log_power = np.log(power + 1e-8)
        slope, intercept = np.polyfit(log_freq, log_power, 1)
        
        return log_freq, log_power, float(slope), float(intercept)
        
    
    def _radial_distribution(self, radial_profile, n_bands=10):
        n = len(radial_profile)
        band_size = n // n_bands        
        radial_dict  = {}
        
        for i in range(n_bands):
            start = i * band_size
            end = start + band_size if i < n_bands - 1 else n
            band = radial_profile[start:end]
            radial_dict[f"radial_band_{i}"] = float(np.mean(band))
        
        return radial_dict
    
    
    def _angular_distribution(self, angular_profile, n_bands=10):
        n = len(angular_profile)
        angles = np.linspace(0, 360, n, endpoint=False)
        
        angular_dict  = {}
        
        for i in range(n_bands):
            angle_start = i * (360 / n_bands)
            angle_end   = (i + 1) * (360 / n_bands)
            mask = (angles >= angle_start) & (angles < angle_end)
            
            if mask.sum() > 0:
                angular_dict[f"angular_band_{i}"] = float(np.mean(angular_profile[mask]))
            else:
                angular_dict[f"angular_band_{i}"] = 0.0
        
        return angular_dict
    
    
    def _spectrum_stats(self, log_magnitude, slope):
        logmag_flat = log_magnitude.ravel()

        return {
            "fft_log_magnitude_mean": safe_mean(logmag_flat),
            "fft_log_magnitude_std": safe_std(logmag_flat),
            "fft_log_magnitude_entropy": safe_entropy(self._normalize_hist(logmag_flat, bins=64)),
            "fft_frequency_slope": slope
        }
        
        
    def _normalize_hist(self, values, bins=64):
        hist, _ = np.histogram(values, bins=bins, density=True)
        hist = hist.astype(np.float64)
        hist /= (hist.sum() + 1e-12)
        return hist