import cv2
import numpy as np
from src.utils.stats_utils import safe_mean, safe_std, safe_var, safe_entropy, safe_uniformity

class EdgeFeatureExtractor:

    def _edge_stats(self, gray_img):

        sobel_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)

        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        grad_angle = np.arctan2(sobel_y, sobel_x)

        lap = cv2.Laplacian(gray_img, cv2.CV_32F, ksize=3)
        
        gray_u8 = np.clip((gray_img * 255).round(), 0, 255).astype(np.uint8)
        
        high = np.percentile(gray_u8, 90)
        low = 0.5 * high
        
        canny = cv2.Canny(gray_u8, low, high)
        
        
        
        edge_pixels = canny > 0

        feature_dict = {}

        feature_dict["sobel_grad_mean"] = safe_mean(grad_mag)
        feature_dict["sobel_grad_std"] = safe_std(grad_mag)
        feature_dict["sobel_grad_energy"] = float(np.mean(grad_mag ** 2)) if grad_mag.size else np.nan
        feature_dict["laplacian_var"] = safe_var(lap)
        feature_dict["canny_edge_density"] = float(np.mean(edge_pixels)) if edge_pixels.size else np.nan

        if np.any(edge_pixels):
            edge_angles = grad_angle[edge_pixels]
            angle_hist, _ = np.histogram(edge_angles, bins=8, range=(-np.pi, np.pi))
            angle_hist = angle_hist.astype(np.float64)
            angle_hist /= (angle_hist.sum() + 1e-12)
            feature_dict["canny_edge_orientation_entropy"] = safe_entropy(angle_hist)
        else:
            feature_dict["canny_edge_orientation_entropy"] = np.nan

        return feature_dict