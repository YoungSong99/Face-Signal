from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

ort.preload_dlls(directory="")

from uniface.analyzer import FaceAnalyzer
from uniface.constants import ParsingWeights
from uniface.detection import RetinaFace
from uniface.parsing import BiSeNet

from src.preprocessing.utils import get_largest_face


class FaceExtractor:
    def __init__(
        self,
        retinaface_confidence_threshold: float = 0.5,
        skin_label: int = 1,
    ):
        self.skin_label = skin_label

        self.analyzer = FaceAnalyzer(
            detector=RetinaFace(
                confidence_threshold=retinaface_confidence_threshold
            ),
        )
        self.parser = BiSeNet(model_name=ParsingWeights.RESNET34)

    def detect_face_bbox(self, img_bgr: np.ndarray) -> list[int] | None:
        if img_bgr is None or img_bgr.size == 0:
            return None

        try:
            faces = self.analyzer.analyze(img_bgr)
        except Exception as e:
            print(f"detect_face_bbox failed: {e}")
            return None

        if not faces:
            return None

        face = get_largest_face(faces)
        return list(map(int, face.bbox))

    def crop_face_square(
        self,
        img_bgr: np.ndarray,
        bbox: list[int] | tuple[int, int, int, int] | None,
    ) -> np.ndarray | None:
        if bbox is None:
            return None

        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        bw = x2 - x1
        bh = y2 - y1
        side = max(bw, bh)
        half = side // 2

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        left = max(cx - half, 0)
        right = min(cx + half, w)
        top = max(cy - half, 0)
        bottom = min(cy + half, h)

        crop = img_bgr[top:bottom, left:right]
        if crop.size == 0:
            return None

        return crop

    def parse_skin_mask(self, face_bgr: np.ndarray) -> np.ndarray | None:
        if face_bgr is None or face_bgr.size == 0:
            return None

        try:
            mask = self.parser.parse(face_bgr)
        except Exception as e:
            print(f"parse_skin_mask failed: {e}")
            return None

        mask = np.asarray(mask)
        skin_mask = (mask == self.skin_label).astype(np.uint8) * 255

        if skin_mask.sum() == 0:
            return None

        return skin_mask

    def extract_skin_region(
        self,
        face_bgr: np.ndarray,
        skin_mask: np.ndarray | None = None,
    ) -> np.ndarray | None:
        if face_bgr is None or face_bgr.size == 0:
            return None

        if skin_mask is None:
            skin_mask = self.parse_skin_mask(face_bgr)

        if skin_mask is None:
            return None

        skin_region = cv2.bitwise_and(face_bgr, face_bgr, mask=skin_mask)
        return skin_region

    def prepare_region(
        self,
        img_bgr: np.ndarray,
        use_face_only: bool = True,
        use_skin_only: bool = False,
    ) -> tuple[np.ndarray, dict]:


        info = {
            "face_detected": False,
            "face_bbox": None,
            "skin_available": False,
            "used_region": "full",
            "face_crop": None,
            "skin_region": None,
        }

        if not use_face_only and not use_skin_only:
            return img_bgr, info

        bbox = self.detect_face_bbox(img_bgr)
        if bbox is None:
            return img_bgr, info

        info["face_detected"] = True
        info["face_bbox"] = bbox

        face_crop = self.crop_face_square(img_bgr, bbox)
        if face_crop is None or face_crop.size == 0:
            return img_bgr, info

        info["face_crop"] = face_crop
        
        if use_skin_only:
            skin_mask = self.parse_skin_mask(face_crop)
            if skin_mask is None:
                info["used_region"] = "face"
                return face_crop, info

            skin_region = self.extract_skin_region(face_crop, skin_mask)
            if skin_region is None:
                info["used_region"] = "face"
                return face_crop, info

            info["skin_available"] = True
            info["skin_region"] = skin_region
            info["used_region"] = "skin"
            return skin_region, info

        info["used_region"] = "face"
        return face_crop, info