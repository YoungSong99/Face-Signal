import cv2
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.preprocessing.face_extractor import FaceExtractor
from .utils import map_output_path, normalize_image_path


class ControlledFaceDatasetBuilder:
    def __init__(
        self,
        data_dir,
        face_dir,
        skin_dir,
        retinaface_confidence_threshold=0.5,
        save_ext=".png",
    ):
        self.data_dir = Path(data_dir).resolve()
        self.face_dir = Path(face_dir).resolve()
        self.skin_dir = Path(skin_dir).resolve()
        self.save_ext = save_ext

        self.face_extractor = FaceExtractor(
            retinaface_confidence_threshold=retinaface_confidence_threshold
        )

    def save_face_crop(self, face_crop, image_path):
        out_path = map_output_path(
            image_path=image_path,
            src_root=self.data_dir,
            dst_root=self.face_dir,
            suffix=self.save_ext,
        )

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(out_path, face_crop)

        return out_path if ok else None

    def save_skin_region(self, skin_region, image_path):
        out_path = map_output_path(
            image_path=image_path,
            src_root=self.data_dir,
            dst_root=self.skin_dir,
            suffix=self.save_ext,
        )

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(out_path, skin_region)

        return out_path if ok else None

    def analyze_image(self, image_path):
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return

        bbox = self.face_extractor.detect_face_bbox(img_bgr)
        if bbox is None:
            return

        face_crop = self.face_extractor.crop_face_square(img_bgr, bbox)
        if face_crop is None or face_crop.size == 0:
            return

        skin_mask = self.face_extractor.parse_skin_mask(face_crop)
        if skin_mask is None:
            return

        skin_region = self.face_extractor.extract_skin_region(face_crop, skin_mask)
        if skin_region is None:
            return

        self.save_face_crop(face_crop, image_path)
        self.save_skin_region(skin_region, image_path)

    def run_from_csv(self, input_csv, image_path_column):
        df = pd.read_csv(input_csv)
        image_paths = df[image_path_column].dropna().tolist()

        for raw_path in tqdm(image_paths, desc="Processing images"):
            image_path = normalize_image_path(raw_path, self.data_dir)
            try:
                self.analyze_image(str(image_path))
            except Exception as e:
                print(f"pipeline_crash: {image_path} | {e}")