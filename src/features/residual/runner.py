import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.features.residual.extractor import RGBYCrCbFusionResidualFeatureExtractor, ResidualFeatureExtractor

CSV_PATH = r"C:\Users\Young\projects_win\Face-Signal\data\metadata\sample_rgb.csv"
OUTPUT_PATH = r"C:\Users\Young\projects_win\Face-Signal\outputs\rgb_residual_features.csv"

METHODS = ["blur", "jpeg", "vae", "sr", "rigid"]

ZONE_MAP_4x4 = np.array([
    [ 1,  2,  3,  4],
    [ 5,  6,  7,  8],
    [ 9, 10, 11, 12],
    [13, 14, 15, 16],
], dtype=np.uint8)

extractor = ResidualFeatureExtractor(
    zone_map=ZONE_MAP_4x4,
    use_zone_stats=True,
    use_zone_frequency=True,
    use_rgb_channel_features=True,
    use_ycrcb_channel_features=False,
    min_freq_patch_size=8,
)

def load_residual(path):
    try:
        p = Path(path)
        if not p.exists():
            return None
        return np.load(p).astype(np.float32)
    except Exception:
        return None

df = pd.read_csv(CSV_PATH)

rows = []

for row in tqdm(df.itertuples(index=False), total=len(df)):
    out = {
        "image_path": row.image_path,
        "size_bucket": row.size_bucket,
        "Target": row.Target,
        "Type": row.Type,
        "Model": row.Model,
        "Skin_Tone": getattr(row, "Skin_Tone", None),
        "Skin_Tone_Group": getattr(row, "Skin_Tone_Group", None),
        "Predicted_Gender": getattr(row, "Predicted_Gender", None),
        "Predicted_Age": getattr(row, "Predicted_Age", None),
        "Intersection": getattr(row, "Intersection", None),
    }

    for method in METHODS:
        path = getattr(row, f"{method}_path", None)
        if path is None or (isinstance(path, float) and np.isnan(path)):
            continue

        residual = load_residual(path)

        print("\n====================")
        print("method:", method)
        print("path:", path)

        if residual is None:
            print("❌ residual is None")
            out[f"{method}_load_failed"] = 1
            continue
        try:
            feats = extractor.extract(
                rgb_residual=residual,
                ycrcb_residual=residual,
            )

            print("✅ success, num_feats:", len(feats))

            for k, v in feats.items():
                out[f"{method}_{k}"] = v

        except Exception as e:
            print("❌ ERROR:", repr(e))
            out[f"{method}_feature_failed"] = 1
            out[f"{method}_feature_error"] = str(e)

    rows.append(out)

feature_df = pd.DataFrame(rows)
feature_df.to_csv(OUTPUT_PATH, index=False)

print("saved:", OUTPUT_PATH)
print(feature_df.shape)