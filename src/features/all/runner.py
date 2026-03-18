from .config import INPUT_CSV, OUTPUT_CSV, IMAGE_PATH_COLUMN
from .pipeline import AllFeatureExtractionPipeline


def main():
    pipeline = AllFeatureExtractionPipeline(
        input_csv=str(INPUT_CSV),
        output_csv=str(OUTPUT_CSV),
        image_path_column=IMAGE_PATH_COLUMN,
    )
    pipeline.run()
    print("Done")


if __name__ == "__main__":
    main()