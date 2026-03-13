import cv2
import numpy as np
import pandas as pd
from skimage.util import crop
from tqdm import tqdm
from pathlib import Path
import onnxruntime as ort

ort.preload_dlls(directory="")
ort.print_debug_info()

from uniface.analyzer import FaceAnalyzer
from uniface.constants import ParsingWeights
from uniface.detection import RetinaFace
from uniface.parsing import BiSeNet

from .utils import get_largest_face, map_output_path, normalize_image_path


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

        self.analyzer = FaceAnalyzer(
            detector=RetinaFace(confidence_threshold=retinaface_confidence_threshold),
        )

        self.parser = BiSeNet(model_name=ParsingWeights.RESNET34)



    def analyze_face(self, img_bgr):

        try:
            faces = self.analyzer.analyze(img_bgr)
        except Exception as e:
            print(f"analyze face failed: {e}")
            return None

        if faces:
            face = get_largest_face(faces)
            return list(face.bbox)
        else:
            return None



    def crop_face_square(self, img_bgr, bbox):

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
    


    def save_face_crop(self, face_crop, image_path):

        out_path = map_output_path(
            image_path=image_path,
            src_root=self.data_dir,
            dst_root=self.face_dir,
            suffix=self.save_ext,
        )

        ok = cv2.imwrite(out_path, face_crop)

        return out_path if ok else None
    


    def parse_skin(self, face_crop):
        
        if face_crop is None or face_crop.size == 0:
            return None

        try:
            mask = self.parser.parse(face_crop)
        except Exception as e:
            print(f"parsing_failed: {e}")
            return None

        mask = np.asarray(mask)
        skin_mask = (mask == 1).astype(np.uint8) * 255

        if skin_mask.sum() == 0:
            return None

        return skin_mask



    def save_skin_region(self, face_crop, skin_mask, image_path):

        skin_region = cv2.bitwise_and(face_crop, face_crop, mask=skin_mask)

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
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return

        face_bbox = self.analyze_face(img_bgr)
        if face_bbox is None:
            return

        face_crop = self.crop_face_square(img_bgr, face_bbox)
        if face_crop is None or face_crop.size == 0:
            return

        skin_mask = self.parse_skin(face_crop)
        if skin_mask is None:
            return

        self.save_face_crop(face_crop, image_path)
        self.save_skin_region(face_crop, skin_mask, image_path)



    def run_from_csv(self, input_csv, image_path_column):

        df = pd.read_csv(input_csv)
        image_paths = df[image_path_column].dropna().tolist()

        for raw_path in tqdm(image_paths, desc="Processing images"):
            image_path = normalize_image_path(raw_path, self.data_dir)
            try:
                self.analyze_image(str(image_path))
            except Exception as e:
                print(f"pipeline_crash: {image_path} | {e}")