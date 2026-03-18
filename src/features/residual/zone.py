import cv2
import numpy as np

from src.features.frequency.dwt_features import DWTFeatureExtractor
from src.features.frequency.fft_features import FFTFeatureExtractor
from src.features.frequency.dct_features import DCTFeatureExtractor
from src.utils.stats_utils import safe_kurtosis, safe_mean, safe_skewness, safe_std, safe_var


class ResidualZoneFeatures:
    def __init__(
        self,
        zone_map: np.ndarray,
        use_global_stats: bool = True,
        use_global_frequency: bool = True,
        use_zone_stats: bool = True,
        fft_extractor: FFTFeatureExtractor | None = None,
        dct_extractor: DCTFeatureExtractor | None = None,
        dwt_extractor: DWTFeatureExtractor | None = None,
    ):
        self.zone_map = zone_map.astype(np.uint8)

        self.use_global_stats = use_global_stats
        self.use_global_frequency = use_global_frequency
        self.use_zone_stats = use_zone_stats

        self.fft_extractor = fft_extractor or FFTFeatureExtractor()
        self.dct_extractor = dct_extractor or DCTFeatureExtractor()
        self.dwt_extractor = dwt_extractor or DWTFeatureExtractor()

    def extract(self, gray: np.ndarray) -> dict[str, float]:
        
        features = {}

        if self.use_global_stats:
            features.update(self._global_stats(gray))

        if self.use_global_frequency:
            features.update(self._global_frequency(gray))

        if self.use_zone_stats:
            features.update(self._zone_stats(gray))
            
        return features

   
    def _global_stats(self, gray):
        vals = gray.ravel()

        return {
            "global_mse": float(np.mean(vals ** 2)),
            "global_mae": float(np.mean(np.abs(vals))),
            "global_mean": safe_mean(vals),
            "global_std": safe_std(vals),
            "global_var": safe_var(vals),
            "global_kurtosis": safe_kurtosis(vals),
            "global_skewness": safe_skewness(vals),
            "global_p95": float(np.percentile(vals, 95)),
        }


    def _global_frequency(self, gray):
        features = {}

        fft_features = self.fft_extractor.fft_features(gray)
        
        for key, value in fft_features.items():
            features[f"global_{key}"] = value

        dct_features = self.dct_extractor._dct_stats(gray)
        
        for key, value in dct_features.items():
            features[f"global_{key}"] = value

        dwt_features = self.dwt_extractor.dwt_features(gray)
        
        for key, value in dwt_features.items():
            features[f"global_{key}"] = value

        return features


    def _resize_zone_map(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        return cv2.resize(self.zone_map,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )


    def _zone_stats(self, gray):
        resized_zone_map = self._resize_zone_map(gray)

        zone_means = {}

        for zone_id in sorted(np.unique(resized_zone_map).tolist()):
            zone_mask = (resized_zone_map == zone_id)

            if not np.any(zone_mask):
                continue

            values = gray[zone_mask]
            if values.size == 0:
                continue

            zone_means[f"zone{zone_id}_mean"] = safe_mean(values)

        mean_values = list(zone_means.values())

        features = dict(zone_means)

        if mean_values:
            features.update({
                "zone_mean_var": safe_var(mean_values),
                "zone_mean_max": float(np.max(mean_values)),
                "zone_mean_min": float(np.min(mean_values)),
            })

        return features