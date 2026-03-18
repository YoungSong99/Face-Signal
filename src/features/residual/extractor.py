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
        gray_source: str = "rgb", # "rgb" or "ycrcb"
    ):

        self.zone_extractor = ResidualZoneFeatures(
            zone_map=zone_map,
            use_zone_stats=use_zone_stats,
            use_global_stats=use_global_stats,
            use_global_frequency=use_global_frequency
        )
        

        self.rgb_channel_extractor = ResidualChannelFeatures(
            color_space="rgb",
        )

        self.ycrcb_channel_extractor = ResidualChannelFeatures(
            color_space="ycrcb",
        )

        self.use_rgb_channel_features = use_rgb_channel_features
        self.use_ycrcb_channel_features = use_ycrcb_channel_features
        self.gray_source = gray_source.lower()


    def extract(self, rgb_residual, ycrcb_residual):
    
        self._validate_residual(rgb_residual, "rgb_residual")
        self._validate_residual(ycrcb_residual, "ycrcb_residual")

        rgb_residual = rgb_residual.astype(np.float32)
        ycrcb_residual = ycrcb_residual.astype(np.float32)

        features = {}
        
        gray = self._make_gray(rgb_residual, ycrcb_residual)
        features.update(self.zone_extractor.extract(gray))


        if self.use_rgb_channel_features:
            features.update(self.rgb_channel_extractor.extract(rgb_residual))

        if self.use_ycrcb_channel_features:
            features.update(self.ycrcb_channel_extractor.extract(ycrcb_residual))

        return features
    

    def _make_gray(self, rgb_residual: np.ndarray, ycrcb_residual: np.ndarray):
        if self.gray_source == "rgb":
            return rgb_float_to_gray(rgb_residual / 255.0)

        if self.gray_source == "ycrcb":
            return ycrcb_residual[:, :, 0].astype(np.float32)

        gray_rgb = rgb_float_to_gray(rgb_residual / 255.0)
        gray_ycc = rgb_float_to_gray(ycrcb_residual / 255.0)

        return ((gray_rgb + gray_ycc) / 2.0).astype(np.float32)



    def _validate_residual(self, residual: np.ndarray, name: str):
        if residual is None:
            raise ValueError(f"{name} is None.")
        if residual.ndim != 3 or residual.shape[2] != 3:
            raise ValueError(f"{name} must be H x W x 3, got {residual.shape}")