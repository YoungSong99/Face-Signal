import numpy as np
from src.utils.stats_utils import safe_stats, hist_entropy

# https://github.com/chuangchuangtan/NPR-DeepfakeDetection
class NPRFeatureExtractor:

    def _npr_stats(self, gray) -> dict:
        
        features = {}

        features.update(self._npr_direction_stats(gray, j=self.npr_j))
        features.update(self._npr_window_stats(gray, l=self.npr_l, j=self.npr_j))

        return features


    def _npr_direction_stats(self, gray: np.ndarray, j: int = 1) -> dict:
        directions = {
            "h": gray[:, j:] - gray[:, :-j],
            "v": gray[j:, :] - gray[:-j, :],
            "d1": gray[j:, j:] - gray[:-j, :-j],
            "d2": gray[j:, :-j] - gray[:-j, j:],
        }

        npr_dict = {}

        for direction_name, diff in directions.items():
            stats = safe_stats(diff)
            for stat_name, value in stats.items():
                npr_dict[f"npr_{direction_name}_{stat_name}"] = value

        return npr_dict


    def _npr_window_stats(self, gray: np.ndarray, l: int = 2, j: int = 1) -> dict:
        npr_window_dict = {}

        if gray.shape[0] <= 2 * l or gray.shape[1] <= 2 * l:
            return npr_window_dict

        center = gray[l:-l, l:-l]
        neighbor = gray[l - j:gray.shape[0] - l - j, l:gray.shape[1] - l]

        if center.shape != neighbor.shape:
            return npr_window_dict

        npr_map = center - neighbor
        stats = safe_stats(npr_map)

        for stat_name, value in stats.items():
            npr_window_dict[f"npr_window_{stat_name}"] = value

        npr_window_dict["npr_window_entropy"] = hist_entropy(np.abs(npr_map), bins=32)
        npr_window_dict["npr_high_diff_ratio"] = float((np.abs(npr_map) > 0.1).mean())

        return npr_window_dict