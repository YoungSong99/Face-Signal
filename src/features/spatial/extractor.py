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

    def extract(self, image: np.ndarray) -> dict[str, float]:

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be an RGB image with shape (H, W, 3).")

        features = {}

        if self.use_color:
            features.update(self._color_stats(image))

        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if self.use_texture:
            features.update(self._lbp_stats(gray_img))
            features.update(self._glcm_stats(gray_img))
            features.update(self._gabor_stats(gray_img))

        if self.use_edge:
            features.update(self._edge_stats(gray_img))

        return features