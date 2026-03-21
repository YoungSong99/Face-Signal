import numpy as np

from .channel import ResidualChannelFeatures
from .zone import ResidualZoneFeatures
from src.utils.image_utils import rgb_float_to_gray


class ResidualFeatureExtractor:
    def __init__(
        self,
        zone_map: np.ndarray,
        use_zone_stats: bool = True,
        use_global_stats: bool = True,
        use_global_frequency: bool = True,
        use_rgb_channel_features: bool = True,
        use_ycrcb_channel_features: bool = True,
        gray_source: str = "rgb",
    ):
        self.zone_extractor = ResidualZoneFeatures(
            zone_map=zone_map,
            use_zone_stats=use_zone_stats,
            use_global_stats=use_global_stats,
            use_global_frequency=use_global_frequency,
        )

        self.rgb_channel_extractor = ResidualChannelFeatures(color_space="rgb")
        self.ycrcb_channel_extractor = ResidualChannelFeatures(color_space="ycrcb")

        self.use_rgb_channel_features = use_rgb_channel_features
        self.use_ycrcb_channel_features = use_ycrcb_channel_features
        self.gray_source = gray_source.lower()

    def extract(self, rgb_residual=None, ycrcb_residual=None):
        if rgb_residual is None and ycrcb_residual is None:
            raise ValueError("At least one residual input must be provided.")

        if rgb_residual is not None:
            self._validate_residual(rgb_residual, "rgb_residual")
            rgb_residual = rgb_residual.astype(np.float32)

        if ycrcb_residual is not None:
            self._validate_residual(ycrcb_residual, "ycrcb_residual")
            ycrcb_residual = ycrcb_residual.astype(np.float32)

        gray = self._make_gray(rgb_residual, ycrcb_residual)

        features = {}
        features.update(self.zone_extractor.extract(gray))

        if self.use_rgb_channel_features and rgb_residual is not None:
            features.update(self.rgb_channel_extractor.extract(rgb_residual))

        if self.use_ycrcb_channel_features and ycrcb_residual is not None:
            features.update(self.ycrcb_channel_extractor.extract(ycrcb_residual))

        return features

    def _make_gray(self, rgb_residual=None, ycrcb_residual=None):
        if self.gray_source == "rgb":
            if rgb_residual is None:
                raise ValueError("rgb_residual is required when gray_source='rgb'")
            return rgb_float_to_gray(rgb_residual / 255.0)

        if self.gray_source == "ycrcb":
            if ycrcb_residual is None:
                raise ValueError("ycrcb_residual is required when gray_source='ycrcb'")
            return ycrcb_residual[:, :, 0].astype(np.float32)

        if rgb_residual is not None and ycrcb_residual is not None:
            gray_rgb = rgb_float_to_gray(rgb_residual / 255.0)
            gray_ycc = ycrcb_residual[:, :, 0].astype(np.float32) / 255.0
            return ((gray_rgb + gray_ycc) / 2.0).astype(np.float32)

        if ycrcb_residual is not None:
            return ycrcb_residual[:, :, 0].astype(np.float32)

        if rgb_residual is not None:
            return rgb_float_to_gray(rgb_residual / 255.0)

        raise ValueError("No valid residual input.")

    def _validate_residual(self, residual: np.ndarray, name: str):
        if residual is None:
            raise ValueError(f"{name} is None.")
        if residual.ndim != 3 or residual.shape[2] != 3:
            raise ValueError(f"{name} must be H x W x 3, got {residual.shape}")