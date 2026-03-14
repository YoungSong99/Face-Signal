import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
from skimage.filters import gabor
from src.utils import safe_mean, safe_std, safe_var, safe_entropy, safe_uniformity


class TextureFeatureExtractor:
    def extract(self, image):
        features = {}

        features.update(self._lbp_stats(image))
        features.update(self._glcm_stats(image))
        features.update(self._gabor_stats(image))

        return features

    # https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html
    def _lbp_stats(self, gray_img):

        test_pairs = [(8, 1), (16, 2), (24, 3), (32, 4)]
        hist_list = {}

        for P, R in test_pairs:
            lbp = local_binary_pattern(gray_img, P=P, R=R, method='uniform')
            n_bins = int(lbp.max() + 1)

            hist, _ = np.histogram(lbp.ravel(),
                                   bins=n_bins,
                                   range=(0, n_bins),
                                   density=True)

            hist = hist.astype(np.float64)
            normalized_hist = hist / (hist.sum() + 1e-12)
            hist_list[f"lbp_hist_{P}_{R}"] = normalized_hist
            hist_list[f"lbp_entropy_{P}_{R}"] = safe_entropy(normalized_hist)
            hist_list[f"lbp_uniformity_{P}_{R}"] = safe_uniformity(normalized_hist)

        return hist_list

    # https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_glcm.html
    def _glcm_stats(self, gray_img):

        levels = 16
        distances = (1, 2, 4)
        angles = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
        props = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")

        gray_q = np.floor(gray_img / (256 / levels)).astype(np.uint8)
        gray_q = np.clip(gray_q, 0, levels - 1)

        glcm = graycomatrix(gray_q,
                            distances=list(distances),
                            angles=list(angles),
                            levels=levels,
                            symmetric=True,
                            normed=True,
                            )

        glcm_dict = {}

        for prop in props:
            values = graycoprops(glcm, prop)

            for d_idx, dist in enumerate(distances):
                row = values[d_idx]
                glcm_dict[f"glcm_{prop}_d{dist}_mean"] = safe_mean(row)
                glcm_dict[f"glcm_{prop}_d{dist}_std"] = safe_std(row)

        return glcm_dict


     # https://scikit-image.org/docs/0.24.x/auto_examples/features_detection/plot_gabor.html
    def _gabor_stats(self, gray_img):
            thetas = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
            frequencies = (0.1, 0.2, 0.3, 0.4)

            gabor_dict = {}

            for freq in frequencies:
                for theta in thetas:
                    real, imag = gabor(gray_img, frequency=freq, theta=theta)
                    magnitude = np.sqrt(real ** 2 + imag ** 2)

                    theta_deg = int(np.rad2deg(theta))
                    prefix = f"gabor_f{str(freq).replace('.', 'p')}_t{theta_deg}"

                    gabor_dict[f"{prefix}_mean"] = safe_mean(magnitude)
                    gabor_dict[f"{prefix}_std"] = safe_std(magnitude)
                    gabor_dict[f"{prefix}_var"] = safe_var(magnitude)
                    gabor_dict[f"{prefix}_energy"] = float(np.mean(magnitude ** 2))

            return gabor_dict