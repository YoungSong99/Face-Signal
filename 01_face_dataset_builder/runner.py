from .config import (
    DATA_DIR,
    FACE_DIR,
    SKIN_DIR,
    IMAGE_PATH,
    INPUT_CSV,
    DEFAULT_TARGET,
    RETINAFACE_CONFIDENCE_THRESHOLD,
)
from .pipeline import ControlledFaceDatasetBuilder


def main():
    builder = ControlledFaceDatasetBuilder(
        data_dir=DATA_DIR,
        face_dir=FACE_DIR,
        skin_dir=SKIN_DIR,
        retinaface_confidence_threshold=RETINAFACE_CONFIDENCE_THRESHOLD,
    )

    builder.run_from_csv(
        input_csv=str(INPUT_CSV),
        image_path_column=IMAGE_PATH,
    )

    print("Done")


if __name__ == "__main__":
    main()