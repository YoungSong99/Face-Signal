import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.features.residual.extractor import ResidualFeatureExtractor

CSV_PATH = r"C:\Users\Young\projects_win\Face-Signal\data\sample.csv"
OUTPUT_PATH = r"C:\Users\Young\projects_win\Face-Signal\data\residual_features.csv"

METHODS = ["blur", "jpeg", "vae", "sr"]
SAVE_EVERY = 1000
NUM_WORKERS = max(1, os.cpu_count() - 1)

ZONE_MAP_4x4 = np.array([
    [ 1,  2,  3,  4],
    [ 5,  6,  7,  8],
    [ 9, 10, 11, 12],
    [13, 14, 15, 16],
], dtype=np.uint8)

# 각 프로세스에서 lazy init
_EXTRACTOR = None


def get_extractor():
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = ResidualFeatureExtractor(
            zone_map=ZONE_MAP_4x4,
            use_zone_stats=False,
            use_rgb_channel_features=True,
            use_ycrcb_channel_features=False,
            gray_source="rgb",
        )
    return _EXTRACTOR


def load_residual(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        # mmap보다 그냥 load가 더 나은 경우가 많음
        return np.load(p)
    except Exception:
        return None


def process_row(row_dict):
    extractor = get_extractor()

    out = {
        "image_path": row_dict["image_path"],
        "size_bucket": row_dict["size_bucket"],
        "Target": row_dict["Target"],
        "Type": row_dict["Type"],
        "Model": row_dict["Model"],
        "Skin_Tone": row_dict.get("Skin_Tone"),
        "Skin_Tone_Group": row_dict.get("Skin_Tone_Group"),
        "Predicted_Gender": row_dict.get("Predicted_Gender"),
        "Predicted_Age": row_dict.get("Predicted_Age"),
        "Intersection": row_dict.get("Intersection"),
    }

    for method in METHODS:
        path = row_dict.get(f"{method}_path", None)

        if path is None or (isinstance(path, float) and np.isnan(path)) or str(path).strip() == "":
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

    return out


def main():
    df = pd.read_csv(CSV_PATH)

    if Path(OUTPUT_PATH).exists():
        Path(OUTPUT_PATH).unlink()

    rows_buffer = []
    row_dicts = df.to_dict("records")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_row, row) for row in row_dicts]

        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
            rows_buffer.append(future.result())

            if len(rows_buffer) >= SAVE_EVERY:
                chunk_df = pd.DataFrame(rows_buffer)
                write_header = not Path(OUTPUT_PATH).exists()
                chunk_df.to_csv(
                    OUTPUT_PATH,
                    mode="a",
                    header=write_header,
                    index=False,
                )
                rows_buffer.clear()

    if rows_buffer:
        chunk_df = pd.DataFrame(rows_buffer)
        write_header = not Path(OUTPUT_PATH).exists()
        chunk_df.to_csv(
            OUTPUT_PATH,
            mode="a",
            header=write_header,
            index=False,
        )

    print("saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()