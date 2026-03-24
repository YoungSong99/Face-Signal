import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.features.residual.extractor import ResidualFeatureExtractor

CSV_PATH = r"C:\Users\Young\projects_win\Face-Signal\data\sample.csv"
OUTPUT_PATH = r"C:\Users\Young\projects_win\Face-Signal\data\residual_features.csv"

METHODS = ["blur", "jpeg", "vae", "sr"]

ZONE_MAP_4x4 = np.array([
    [ 1,  2,  3,  4],
    [ 5,  6,  7,  8],
    [ 9, 10, 11, 12],
    [13, 14, 15, 16],
], dtype=np.uint8)

ZONE_MAP_8x8 = np.arange(1, 64 + 1, dtype=np.uint16).reshape(8, 8)
ZONE_MAP_16x16 = np.arange(1, 256 + 1, dtype=np.uint16).reshape(16, 16)

extractor = ResidualFeatureExtractor(
    zone_map=ZONE_MAP_4x4,
    use_zone_stats=False,
    use_rgb_channel_features=True,
    use_ycrcb_channel_features=False,
    gray_source="ycrcb",
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

rows = []
SAVE_EVERY = 1000

if Path(OUTPUT_PATH).exists():
    Path(OUTPUT_PATH).unlink()

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
                ycrcb_residual=residual,
            )
            for k, v in feats.items():
                out[f"{method}_{k}"] = v
        except Exception as e:
            out[f"{method}_feature_failed"] = 1
            out[f"{method}_feature_error"] = str(e)

    rows.append(out)

    if (i + 1) % SAVE_EVERY == 0:
        chunk_df = pd.DataFrame(rows)
        write_header = not Path(OUTPUT_PATH).exists()
        chunk_df.to_csv(
            OUTPUT_PATH,
            mode="a",
            header=write_header,
            index=False
        )
        rows.clear()

if rows:
    chunk_df = pd.DataFrame(rows)
    write_header = not Path(OUTPUT_PATH).exists()
    chunk_df.to_csv(
        OUTPUT_PATH,
        mode="a",
        header=write_header,
        index=False
    )

print("saved:", OUTPUT_PATH)