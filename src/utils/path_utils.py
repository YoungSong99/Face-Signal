from pathlib import Path
import numpy as np


def resolve_image_path(image_path_raw, image_base_dir=None):
    
    path = Path(image_path_raw.strip().replace("\\", "/"))

    if not path.is_absolute() and image_base_dir is not None:
        path = Path(image_base_dir) / path

    path = path.resolve()

    return path


def to_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)