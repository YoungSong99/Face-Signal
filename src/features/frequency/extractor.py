import cv2
import numpy as np

from .fft_features import FFTFeatureExtractor
from .dct_features import DCTFeatureExtractor
from .dwt_features import DWTFeatureExtractor


class FrequencyFeatureExtractor(
    FFTFeatureExtractor,
    DCTFeatureExtractor,
    DWTFeatureExtractor,
):
    def __init__(
        self,
        use_fft: bool = True,
        use_dct: bool = True,
        use_dwt: bool = True,

    ):
        self.use_fft = use_fft
        self.use_dct = use_dct
        self.use_dwt = use_dwt

    def extract(self, gray):

        features = {}

        if self.use_fft:
            features.update(self.fft_features(gray))

        if self.use_dct:
            features.update(self._dct_stats(gray))

        if self.use_dwt:
            features.update(self._dwt_stats(gray))

        return features
