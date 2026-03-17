from .npr import NPRFeatureExtractor
from .checkerboard import CheckerboardFeatureExtractor
from .srm import SRMFeatureExtractor


class ArtifactFeatureExtractor(
    NPRFeatureExtractor,
    CheckerboardFeatureExtractor,
    SRMFeatureExtractor,
):
    def __init__(
        self,
        use_npr=True,
        use_checkerboard=True,
        use_srm=True,
        npr_l=2,
        npr_j=1,
        srm_truncation=3.0,
        srm_quant=1,
    ):
        self.use_npr = use_npr
        self.use_checkerboard = use_checkerboard
        self.use_srm = use_srm
        self.npr_l = npr_l
        self.npr_j = npr_j
        self.srm_truncation = srm_truncation
        self.srm_quant = srm_quant

        self.srm_kernels = self._build_srm_kernels()

    def extract(self, gray_img):
        features = {}

        if self.use_npr:
            features.update(self._npr_stats(gray_img))

        if self.use_checkerboard:
            features.update(self._checkerboard_stats(gray_img))

        if self.use_srm:
            features.update(self._srm_stats(gray_img))

        return features