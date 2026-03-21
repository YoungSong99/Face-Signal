import os
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ.pop("PYTORCH_ALLOC_CONF", None)

from src.pipelines.single_image_forensics import (
    SingleImageForensicsPipeline,
    flatten_forensics_result,
)
import numpy as np
import torch

ZONE_MAP_8x8 = np.arange(1, 64 + 1, dtype=np.uint16).reshape(8, 8)
device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline = SingleImageForensicsPipeline(
    zone_map=ZONE_MAP_8x8,
    use_face_only=True,
    use_skin_only=True,
    residual_builder_kwargs={
        "use_jpeg": True,
        "use_blur": True,
        "use_vae": True,
        "use_sr": True,
        "device": device,
    },
)

result = pipeline.analyze_one_image(
    image_path=r"C:\Users\Young\projects_win\Face-Signal\jenny.jpg",
    save_dir=r"C:\Users\Young\projects_win\Face-Signal\one_image_debug",
)

flat_row = flatten_forensics_result(result)

print("face info:", result["face_info"])
print("original feature count:", len(result["original_features"]))
print("reconstruction methods:", list(result["reconstruction_features"].keys()))
print("residual methods:", list(result["residual_features"].keys()))
print("flat row feature count:", len(flat_row))