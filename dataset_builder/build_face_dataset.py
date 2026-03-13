from __future__ import annotations

import cv2
import pandas as pd
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import normalize_image_path, map_output_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "mediapipe" / "blaze_face_short_range.tflite"


@dataclass(frozen=True)
class FaceCropConfig:
    input_csv: Path
    data_dir: Path
    output_dir: Path
    image_path_col: str = "image_path"
    save_ext: str = ".png"
    min_detection_confidence: float = 0.5
    checkpoint_every: int = 10000


class FaceCropper:
    def __init__(self, config: FaceCropConfig):
        self.config = config
        self.data_dir = Path(config.data_dir).resolve()
        self.output_dir = Path(config.output_dir).resolve()
        self.input_csv = Path(config.input_csv).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH),delegate=python.BaseOptions.Delegate.GPU)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=config.min_detection_confidence
        )
        self.detector = vision.FaceDetector.create_from_options(options)

    def crop_face_square(self, img_path: Path) -> tuple[Path, int, int] | tuple[None, None, None]:
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            return None, None, None

        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = self.detector.detect(mp_image)

        if not detection_result.detections:
            return None, None, None

        def get_area(det):
            return det.bounding_box.width * det.bounding_box.height

        largest_face = max(detection_result.detections, key=get_area)
        bbox = largest_face.bounding_box

        bw, bh = bbox.width, bbox.height
        cx = bbox.origin_x + (bw / 2)
        cy = bbox.origin_y + (bh / 2)

        side = max(bw, bh)
        half = side / 2.0

        x1, y1 = int(round(cx - half)), int(round(cy - half))
        x2, y2 = int(round(cx + half)), int(round(cy + half))

        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > w:
            x1 -= (x2 - w)
            x2 = w
        if y2 > h:
            y1 -= (y2 - h)
            y2 = h

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None, None, None

        face_crop = image_bgr[y1:y2, x1:x2]
        
        crop_h, crop_w = face_crop.shape[:2]
        
        saved_path = self.save_face_crop(face_crop, img_path)
        return saved_path, crop_w, crop_h        
    

    def save_face_crop(self, face_crop, image_path: Path) -> Path | None:
        out_path = map_output_path(
            image_path=image_path,
            src_root=self.data_dir,
            dst_root=self.output_dir,
            suffix=self.config.save_ext,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)

        ok = cv2.imwrite(str(out_path), face_crop)
        return out_path if ok else None

    def run_from_csv(self, save_csv_path: Path | None = None) -> pd.DataFrame:
        df = pd.read_csv(self.input_csv)
        results = []
        image_paths = df[self.config.image_path_col].dropna().tolist()

        def flush_partial() -> None:
            if save_csv_path is None:
                return
            save_csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(results).to_csv(save_csv_path, index=False)

        try:
            for i, raw_path in enumerate(tqdm(image_paths, desc="Cropping faces"), start=1):
                image_path = normalize_image_path(raw_path, self.data_dir)
                row = {
                    "raw_path": str(raw_path),
                    "cropped_face_path": None,
                    "face_width": None,
                    "face_height": None,
                    "status": "failed",
                }

                try:
                    saved_path, w, h = self.crop_face_square(image_path)

                    if saved_path:
                        row["cropped_face_path"] = str(saved_path.relative_to(PROJECT_ROOT))
                        row["face_width"] = w
                        row["face_height"] = h
                        row["status"] = "ok"
                    else:
                        row["status"] = "no_face_detected"

                except Exception as e:
                    row["status"] = f"error: {str(e)}"

                results.append(row)

                if save_csv_path is not None and i % self.config.checkpoint_every == 0:
                    flush_partial()

        except KeyboardInterrupt:
            print("\nInterrupted. Saving partial results...")

        finally:
            flush_partial()

        return pd.DataFrame(results)


if __name__ == "__main__":
    config = FaceCropConfig(
        input_csv=Path(PROJECT_ROOT / "data/metadata/dataset_face_ge_0p5.csv").resolve(),
        data_dir=Path(PROJECT_ROOT / "data/AI_Face_imagesV2").resolve(),
        output_dir=Path(PROJECT_ROOT / "data/cropped_faces").resolve(),
        image_path_col="image_path",
        checkpoint_every=10,
    )

    out_csv = Path(PROJECT_ROOT / "data/metadata/face_crop_results.csv").resolve()

    cropper = FaceCropper(config)
    out_df = cropper.run_from_csv(save_csv_path=out_csv)
    print(out_df["status"].value_counts())