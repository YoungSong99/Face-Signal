from pathlib import Path


def get_largest_face(faces):
    if not faces:
        return None
    
    largest_area = 0
    largest_face = None

    for face in faces:
        width = face.bbox[2] - face.bbox[0]
        height = face.bbox[3] - face.bbox[1]
        area = width * height

        if area > largest_area:
            largest_area = area
            largest_face = face
            
    return largest_face


def normalize_image_path(image_path, data_dir: Path) -> Path:
    image_path = Path(str(image_path))

    if image_path.is_absolute():
        return image_path.resolve()

    return (data_dir / image_path).resolve()


def map_output_path(image_path: str, src_root: str | Path, dst_root: str | Path, suffix: str = ".png") -> str:
    image_path = Path(image_path).resolve()
    src_root = Path(src_root).resolve()
    dst_root = Path(dst_root).resolve()

    rel = image_path.relative_to(src_root)
    out_path = dst_root / rel
    out_path = out_path.with_suffix(suffix)
    
    return str(out_path)