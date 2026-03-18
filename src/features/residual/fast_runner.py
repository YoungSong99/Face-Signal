import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.features.residual.extractor import ResidualFeatureExtractor

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
    use_rgb_channel_features=True,
    use_ycrcb_channel_features=False,
)

def load_residual(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return np.load(p, mmap_mode="r")
    except Exception:
        return None

df = pd.read_csv(CSV_PATH)

# resume
if Path(OUTPUT_PATH).exists():
    done_df = pd.read_csv(OUTPUT_PATH, usecols=["image_path"])
    done_set = set(done_df["image_path"].tolist())
    df = df[~df["image_path"].isin(done_set)]
    print(f"resume: {len(done_set)} skipped")

rows = []
SAVE_EVERY = 1000

for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df))):
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
        if residual is None:
            out[f"{method}_load_failed"] = 1
            continue

        try:
            feats = extractor.extract(
                rgb_residual=residual,
                ycrcb_residual=residual,  # 중요
            )
            for k, v in feats.items():
                out[f"{method}_{k}"] = v
        except Exception as e:
            out[f"{method}_feature_failed"] = 1
            out[f"{method}_feature_error"] = str(e)

    rows.append(out)

    if (i + 1) % SAVE_EVERY == 0:
        chunk_df = pd.DataFrame(rows)
        if Path(OUTPUT_PATH).exists():
            chunk_df.to_csv(OUTPUT_PATH, mode="a", header=False, index=False)
        else:
            chunk_df.to_csv(OUTPUT_PATH, index=False)
        rows.clear()

if rows:
    chunk_df = pd.DataFrame(rows)
    if Path(OUTPUT_PATH).exists():
        chunk_df.to_csv(OUTPUT_PATH, mode="a", header=False, index=False)
    else:
        chunk_df.to_csv(OUTPUT_PATH, index=False)

print("saved:", OUTPUT_PATH)