from pathlib import Path


def normalize_image_path(image_path, data_dir: Path) -> Path:
    image_path = str(image_path).strip().lstrip("/\\")
    return (data_dir / image_path).resolve()

def map_output_path(image_path, src_root, dst_root, suffix=".png") -> Path:
    image_path = Path(image_path).resolve()
    src_root = Path(src_root).resolve()
    dst_root = Path(dst_root).resolve()

    rel = image_path.relative_to(src_root)
    out_path = (dst_root / rel).with_suffix(suffix)

    return out_path