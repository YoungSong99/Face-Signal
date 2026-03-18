import cv2
import numpy as np
from src.utils.stats_utils import safe_stats, hist_entropy


class SRMFeatureExtractor:

    def _srm_stats(self, gray):
        gray_255 = np.clip((gray * 255).round(), 0, 255).astype(np.uint8)

        features = {}
        residuals = []

        for idx, kernel in enumerate(self.srm_kernels):
            residual = self._srm_residual_map(
                gray_255,
                kernel,
                truncation=self.srm_truncation,
                quant=self.srm_quant,
            )
            residuals.append(residual.ravel())

            features.update(self._srm_single_filter_stats(residual, idx))

        features.update(self._srm_cross_filter_stats(residuals))

        return features


    def _build_srm_kernels(self):
        kernels_raw = [
            (
                np.array(
                    [[0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, -1, 1, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]],
                    dtype=np.float32,
                ),
                2.0,
            ),
            (
                np.array(
                    [[0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, -1, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]],
                    dtype=np.float32,
                ),
                2.0,
            ),
            (
                np.array(
                    [[0, 0, 0, 0, 0],
                     [0, -1, 2, -1, 0],
                     [0, 2, -4, 2, 0],
                     [0, -1, 2, -1, 0],
                     [0, 0, 0, 0, 0]],
                    dtype=np.float32,
                ),
                4.0,
            ),
            (
                np.array(
                    [[-1, 2, -2, 2, -1],
                     [2, -6, 8, -6, 2],
                     [-2, 8, -12, 8, -2],
                     [2, -6, 8, -6, 2],
                     [-1, 2, -2, 2, -1]],
                    dtype=np.float32,
                ),
                12.0,
            ),
            (
                np.array(
                    [[0, 0, 0, 0, 0],
                     [0, 1, -2, 1, 0],
                     [0, -2, 4, -2, 0],
                     [0, 1, -2, 1, 0],
                     [0, 0, 0, 0, 0]],
                    dtype=np.float32,
                ),
                4.0,
            ),
        ]
        

        return [kernel / norm for kernel, norm in kernels_raw]


    def _srm_residual_map(self, gray_img, kernel, truncation = 3.0, quant = 1):
        
        residual = cv2.filter2D(gray_img, cv2.CV_32F, kernel)
        residual = np.round(residual / quant)
        residual = np.clip(residual, -truncation, truncation)

        return residual
    

    def _srm_single_filter_stats(self, residual, filter_idx):
        srm_dict = {}
        stats = safe_stats(residual)

        for stat_name, value in stats.items():
            srm_dict[f"srm_f{filter_idx}_{stat_name}"] = value

        n_bins = int(2 * self.srm_truncation / self.srm_quant) + 1
        srm_dict[f"srm_f{filter_idx}_entropy"] = hist_entropy(
            residual,
            bins=n_bins,
            value_range=(-self.srm_truncation, self.srm_truncation),
        )

        srm_dict[f"srm_f{filter_idx}_asymmetry"] = float(
            np.mean(((residual - residual.mean()) / (residual.std() + 1e-8)) ** 3)
        )

        srm_dict[f"srm_f{filter_idx}_zero_ratio"] = float(
            (np.abs(residual) < 0.5).mean()
        )

        return srm_dict


    def _srm_cross_filter_stats(self, residuals) -> dict:
        srm_cross_dict = {}

        for i in range(len(residuals)):
            for j in range(i + 1, len(residuals)):
                ri = residuals[i]
                rj = residuals[j]

                if np.std(ri) < 1e-8 or np.std(rj) < 1e-8:
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(ri, rj)[0, 1])

                srm_cross_dict[f"srm_cross_f{i}f{j}_corr"] = corr

        return srm_cross_dict