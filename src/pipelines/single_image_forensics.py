from pathlib import Path
import cv2
import numpy as np

from src.features.spatial.extractor import SpatialFeatureExtractor
from src.features.frequency.extractor import FrequencyFeatureExtractor
from src.features.artifact.extractor import ArtifactFeatureExtractor
from src.features.residual.extractor import ResidualFeatureExtractor
from src.reconstruction.ycrcb_residual_builder import YCrCbResidualBuilder
from src.preprocessing.face_extractor import FaceExtractor
from src.utils.image_utils import to_uint8_vis


class SingleImageForensicsPipeline:
    def __init__(
        self,
        zone_map: np.ndarray,
        gray_source: str = "ycrcb",
        residual_builder_kwargs: dict | None = None,
        use_original_spatial: bool = True,
        use_original_frequency: bool = True,
        use_original_artifact: bool = True,
        use_residual_features: bool = True,
        use_face_only: bool = True,
        use_skin_only: bool = False,
        retinaface_confidence_threshold: float = 0.5,
    ):
        residual_builder_kwargs = residual_builder_kwargs or {}

        self.builder = YCrCbResidualBuilder(**residual_builder_kwargs)
        self.spatial_extractor = SpatialFeatureExtractor()
        self.frequency_extractor = FrequencyFeatureExtractor()
        self.artifact_extractor = ArtifactFeatureExtractor()

        self.residual_extractor = ResidualFeatureExtractor(
            zone_map=zone_map,
            use_zone_stats=True,
            use_rgb_channel_features=False,
            use_ycrcb_channel_features=True,
            gray_source=gray_source,
        )

        self.use_original_spatial = use_original_spatial
        self.use_original_frequency = use_original_frequency
        self.use_original_artifact = use_original_artifact
        self.use_residual_features = use_residual_features

        self.use_face_only = use_face_only
        self.use_skin_only = use_skin_only

        self.face_helper = FaceExtractor(
            retinaface_confidence_threshold=retinaface_confidence_threshold
        )

    def _prepare_analysis_region(self, img_bgr):
        return self.face_helper.prepare_region(
            img_bgr,
            use_face_only=self.use_face_only,
            use_skin_only=self.use_skin_only,
        )
        

    def _save_images(self, save_dir, image_stem, method, recon_rgb, residual_ycrcb):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        recon_path = save_dir / f"{image_stem}_{method}_recon.png"
        residual_y_path = save_dir / f"{image_stem}_{method}_residual_y.png"

        cv2.imwrite(str(recon_path), cv2.cvtColor(recon_rgb, cv2.COLOR_RGB2BGR))

        residual_y = residual_ycrcb[:, :, 0]
        residual_y_vis = to_uint8_vis(residual_y)
        cv2.imwrite(str(residual_y_path), residual_y_vis)


    def _save_face_and_skin(self, save_dir, image_stem, face_info):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        face_img = face_info.get("face_crop")
        skin_img = face_info.get("skin_region")

        if face_img is not None:
            cv2.imwrite(
                str(save_dir / f"{image_stem}_face.png"),
                face_img,
            )

        if skin_img is not None:
            cv2.imwrite(
                str(save_dir / f"{image_stem}_skin.png"),
                skin_img,
            )


    def analyze_one_image(self, image_path, save_dir=None):
        image_path = Path(image_path)
                    
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Failed to read image: {image_path}")

        analysis_bgr, face_info = self._prepare_analysis_region(img_bgr)

        if save_dir is not None:
            self._save_face_and_skin(
                save_dir=save_dir,
                image_stem=image_path.stem,
                face_info=face_info,
            )
        
        face_crop = face_info.get("face_crop")
        skin_region = face_info.get("skin_region")
        
        if self.use_skin_only and skin_region is not None:
            original_bgr = skin_region
        elif self.use_face_only and face_crop is not None:
            original_bgr = face_crop
        else:
            original_bgr = img_bgr

        if face_crop is not None:
            residual_input_bgr = face_crop
        else:
            residual_input_bgr = original_bgr
        

        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        original_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        original_rgb_f = original_rgb.astype(np.float32) / 255.0

        result = {
            "image_path": str(image_path),
            "used_face_only": self.use_face_only,
            "used_skin_only": self.use_skin_only,
            "face_info": face_info,
            "original_features": {},
            "reconstruction_features": {},
            "residual_features": {},
            "reconstructions_rgb": {},
            "reconstructions_ycrcb": {},
            "residuals_ycrcb": {},
        }

        orig_feats = {}

        if self.use_original_spatial:
            orig_feats.update(self.spatial_extractor.extract(original_rgb_f, original_gray))

        if self.use_original_frequency:
            orig_feats.update(self.frequency_extractor.extract(original_gray))

        if self.use_original_artifact:
            orig_feats.update(self.artifact_extractor.extract(original_gray))

        result["original_features"] = orig_feats

        residual_input_rgb = cv2.cvtColor(residual_input_bgr, cv2.COLOR_BGR2RGB)
        recon_dict = self.builder.extract_with_recons(residual_input_rgb)

        for method, pack in recon_dict.items():
            recon_rgb = pack["reconstruction_rgb"]
            recon_ycrcb = pack["reconstruction_ycrcb"]
            residual_ycrcb = pack["residual_ycrcb"]

            result["reconstructions_rgb"][method] = recon_rgb
            result["reconstructions_ycrcb"][method] = recon_ycrcb
            result["residuals_ycrcb"][method] = residual_ycrcb

            recon_feats = {}
            recon_gray = cv2.cvtColor(recon_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            recon_rgb_f = recon_rgb.astype(np.float32) / 255.0

            if self.use_original_spatial:
                recon_feats.update(self.spatial_extractor.extract(recon_rgb_f, recon_gray))

            if self.use_original_frequency:
                recon_feats.update(self.frequency_extractor.extract(recon_gray))

            if self.use_original_artifact:
                recon_feats.update(self.artifact_extractor.extract(recon_gray))

            for extra_key in ("lpips", "delta_re"):
                if extra_key in pack:
                    recon_feats[extra_key] = pack[extra_key]

            result["reconstruction_features"][method] = recon_feats

            if self.use_residual_features:
                residual_feats = self.residual_extractor.extract(
                    rgb_residual=None,
                    ycrcb_residual=residual_ycrcb,
                )
                result["residual_features"][method] = residual_feats

            if save_dir is not None:
                self._save_images(
                    save_dir=save_dir,
                    image_stem=image_path.stem,
                    method=method,
                    recon_rgb=recon_rgb,
                    residual_ycrcb=residual_ycrcb,
                )

        return result


def flatten_forensics_result(result: dict) -> dict:
    row = {
        "image_path": result["image_path"],
        "used_face_only": result.get("used_face_only"),
        "used_skin_only": result.get("used_skin_only"),
    }

    face_info = result.get("face_info", {})
    row["face_detected"] = face_info.get("face_detected")
    row["skin_available"] = face_info.get("skin_available")
    row["used_region"] = face_info.get("used_region")

    bbox = face_info.get("face_bbox")
    if bbox is not None and len(bbox) == 4:
        row["face_x1"], row["face_y1"], row["face_x2"], row["face_y2"] = bbox
    else:
        row["face_x1"] = None
        row["face_y1"] = None
        row["face_x2"] = None
        row["face_y2"] = None

    for k, v in result.get("original_features", {}).items():
        row[f"orig_{k}"] = v

    for method, feat_dict in result.get("reconstruction_features", {}).items():
        for k, v in feat_dict.items():
            row[f"{method}_recon_{k}"] = v

    for method, feat_dict in result.get("residual_features", {}).items():
        for k, v in feat_dict.items():
            row[f"{method}_res_{k}"] = v

    return row