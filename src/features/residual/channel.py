import cv2
import numpy as np

from src.utils.stats_utils import safe_std, safe_corrcoef


class ResidualChannelFeatures:
    def __init__(self, color_space="rgb"):
        self.color_space = color_space.lower()

        if self.color_space not in {"rgb", "ycrcb"}:
            raise ValueError(
                f"Unsupported color_space={color_space}. "
                f"Use 'rgb' or 'ycrcb'."
            )

        if self.color_space == "rgb":
            self.channel_names = ("r", "g", "b")
            self.prefix = "rgb"
        else:
            self.channel_names = ("y", "cr", "cb")
            self.prefix = "ycrcb"


    def extract(self, residual):
        if residual.ndim != 3 or residual.shape[2] != 3:
            raise ValueError(f"Residual must be H x W x 3, got {residual.shape}")

        residual = residual.astype(np.float32, copy=False)

        c1 = residual[:, :, 0]
        c2 = residual[:, :, 1]
        c3 = residual[:, :, 2]

        n1, n2, n3 = self.channel_names

        features = {}

        features.update(self._channel_energy(c1, c2, c3, n1, n2, n3))
        features.update(self._channel_corr(c1, c2, c3, n1, n2, n3))

        return features


    def _channel_energy(self, c1, c2, c3, n1, n2, n3):
        e1 = self._energy(c1)
        e2 = self._energy(c2)
        e3 = self._energy(c3)

        return {
            f"{self.prefix}_energy_{n1}": e1,
            f"{self.prefix}_energy_{n2}": e2,
            f"{self.prefix}_energy_{n3}": e3,

            f"{self.prefix}_energy_{n1}_minus_{n2}": e1 - e2,
            f"{self.prefix}_energy_{n1}_minus_{n3}": e1 - e3,
            f"{self.prefix}_energy_{n2}_minus_{n3}": e2 - e3,

            f"{self.prefix}_energy_ratio_{n2}_{n1}": e2 / (e1 + 1e-8),
            f"{self.prefix}_energy_ratio_{n3}_{n1}": e3 / (e1 + 1e-8),
            f"{self.prefix}_energy_ratio_{n3}_{n2}": e3 / (e2 + 1e-8),
        }

    def _channel_corr(self, c1, c2, c3, n1, n2, n3):
        return {
            f"{self.prefix}_corr_{n1}_{n2}": safe_corrcoef(c1, c2),
            f"{self.prefix}_corr_{n1}_{n3}": safe_corrcoef(c1, c3),
            f"{self.prefix}_corr_{n2}_{n3}": safe_corrcoef(c2, c3),
        }

    def _energy(self, x):
        x = x.astype(np.float64, copy=False)
        return float(np.mean(x ** 2))