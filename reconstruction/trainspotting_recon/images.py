from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXIF_IFD = 34665
FOCAL_LENGTH_35MM = 41989


def list_images(root: Path) -> list[str]:
    """Image names relative to `root`, in sorted order."""
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def stage_images(image_dir: Path, staged_dir: Path) -> dict[str, str]:
    """Link every photo under a short index name; returns staged name -> source name relative to `image_dir`."""
    staged_dir.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for i, source in enumerate(list_images(image_dir)):
        name = f"{i:06d}{Path(source).suffix.lower()}"
        link = staged_dir / name
        if not link.is_symlink():
            link.symlink_to(image_dir / source)
        mapping[name] = source
    return mapping


def stem(name: str) -> str:
    return Path(name).stem


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def focal_length_35mm(path: Path) -> float | None:
    with Image.open(path) as image:
        exif = image.getexif()
        values = dict(exif.items())
        try:
            values.update(dict(exif.get_ifd(EXIF_IFD).items()))
        except Exception:
            pass
    try:
        focal = float(values.get(FOCAL_LENGTH_35MM))
    except (TypeError, ValueError):
        return None
    return focal if np.isfinite(focal) and focal > 0 else None


def initial_focal(path: Path, width: int, height: int, default_factor: float) -> tuple[float, bool]:
    """Focal length in pixels from the EXIF 35 mm equivalent, else a default prior."""
    focal_35 = focal_length_35mm(path)
    if focal_35 is not None:
        return focal_35 / 36.0 * width, True
    return default_factor * max(width, height), False
