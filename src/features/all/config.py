from pathlib import Path


PROJECT_ROOT = Path(r"C:\Users\Young\projects_win\Face-Signal").resolve()

INPUT_CSV = PROJECT_ROOT / "data" / "metadata" / "dataset_face.csv"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "face_features_all.csv"
IMAGE_PATH_COLUMN = "image_path"