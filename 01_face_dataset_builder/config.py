from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "AI_Face_imagesV2"
FACE_DIR = PROJECT_ROOT / "AI_Face_imagesV2_face"
SKIN_DIR = PROJECT_ROOT / "AI_Face_imagesV2_skin"
IMAGE_PATH = "image_path"
INPUT_CSV = PROJECT_ROOT / "outputs/full_dataset.csv"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "face_dataset_build_result.csv"

DEFAULT_TARGET = -1
RETINAFACE_CONFIDENCE_THRESHOLD = 0.5
