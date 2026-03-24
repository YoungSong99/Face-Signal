import cv2
import numpy as np

from .color_features import ColorFeatureExtractor
from .texture_features import TextureFeatureExtractor
from .edge_features import EdgeFeatureExtractor


class SpatialFeatureExtractor(
    ColorFeatureExtractor,
    TextureFeatureExtractor,
    EdgeFeatureExtractor,
):
    def __init__(
        self,
        use_color: bool = True,
        use_texture: bool = True,
        use_edge: bool = True,
    ):
        self.use_color = use_color
        self.use_texture = use_texture
        self.use_edge = use_edge

    def extract(self, image_rgb=None, gray=None):

        features = {}

        if self.use_color:
            if image_rgb is None:
                raise ValueError("image_rgb is required when use_color=True")
            features.update(self._color_stats(image_rgb))

        if self.use_texture:
            if gray is None:
                raise ValueError("gray is required when use_texture=True")
            features.update(self._lbp_stats(gray))
            features.update(self._glcm_stats(gray))
            features.update(self._gabor_stats(gray))

        if self.use_edge:
            if gray is None:
                raise ValueError("gray is required when use_edge=True")
            features.update(self._edge_stats(gray))

        return features