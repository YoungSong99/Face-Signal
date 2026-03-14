from pathlib import Path


def resolve_image_path(image_path_raw, image_base_dir=None):
    
    path = Path(image_path_raw.strip().replace("\\", "/"))

    if not path.is_absolute() and image_base_dir is not None:
        path = Path(image_base_dir) / path

    path = path.resolve()

    return path