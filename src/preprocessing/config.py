from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data" /"AI_Face_imagesV2"
FACE_DIR = PROJECT_ROOT / "data" / "AI_Face_imagesV2_face"
SKIN_DIR = PROJECT_ROOT / "data" / "AI_Face_imagesV2_skin"
IMAGE_PATH = "image_path"
INPUT_CSV = PROJECT_ROOT / "data" / "metadata" /"dataset_face_ge_0p5.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "metadata" / "face_dataset_build_result.csv"

DEFAULT_TARGET = -1
RETINAFACE_CONFIDENCE_THRESHOLD = 0.5


# print(f"Project root: {PROJECT_ROOT}")
# print(f"Data directory: {DATA_DIR}")
# print(f"Face directory: {FACE_DIR}")
# print(f"Skin directory: {SKIN_DIR}")
# print(f"Input CSV: {INPUT_CSV}")
# print(f"Output CSV: {OUTPUT_CSV}")