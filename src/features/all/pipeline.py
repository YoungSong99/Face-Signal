import cv2
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.utils.image_utils import load_rgb_and_gray
from src.features.spatial.extractor import SpatialFeatureExtractor
from src.features.frequency.extractor import FrequencyFeatureExtractor
from src.features.artifact.extractor import ArtifactFeatureExtractor


class AllFeatureExtractionPipeline:
    def __init__(
        self,
        input_csv,
        output_csv,
        image_path_column="image_path",
    ):
        self.input_csv = Path(input_csv)
        self.output_csv = Path(output_csv)
        self.image_path_column = image_path_column

        self.spatial_extractor = SpatialFeatureExtractor()
        self.frequency_extractor = FrequencyFeatureExtractor()
        self.artifact_extractor = ArtifactFeatureExtractor()


    def extract_single_image(self, image_path):

        image_rgb, gray = load_rgb_and_gray(str(image_path))

        features = {}
        features.update(self.spatial_extractor.extract(image_rgb, gray))
        features.update(self.frequency_extractor.extract(gray))
        features.update(self.artifact_extractor.extract(gray))

        return features


    def run(self):

        df = pd.read_csv(self.input_csv)

        all_features = []

        image_paths = df[self.image_path_column].tolist()

        for path in tqdm(image_paths):
            features = self.extract_single_image(path)
            all_features.append(features)

        feature_df = pd.DataFrame(all_features)
        result_df = pd.concat([df, feature_df], axis=1)

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(self.output_csv, index=False)