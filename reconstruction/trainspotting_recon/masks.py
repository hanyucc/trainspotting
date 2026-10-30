"""Object mask selection and the per-frame object checks."""

import json
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image
from tqdm import tqdm

from .config import Config
from .gating import load_masks
from .images import stem
from .mvs import confident_depth
from .scene import Scene


def _distortion(camera: pycolmap.Camera) -> dict:
    p = list(camera.params)
    model = camera.model_name
    if model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = p
        fx = fy = f
        k2 = 0.0
    elif model == "RADIAL":
        f, cx, cy, k1, k2 = p
        fx = fy = f
    elif model == "PINHOLE":
        fx, fy, cx, cy = p
        k1 = k2 = 0.0
    elif model == "SIMPLE_PINHOLE":
        f, cx, cy = p
        fx = fy = f
        k1 = k2 = 0.0
    else:
        raise ValueError(f"unsupported camera model {model}")
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "k1": k1, "k2": k2, "w": camera.width, "h": camera.height}


def warp_to_undistorted(mask: np.ndarray, dist: dict, K: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resample a mask of the original photo onto the undistorted image grid."""
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    r2 = x * x + y * y
    radial = 1.0 + dist["k1"] * r2 + dist["k2"] * r2 * r2
    xs = np.round(dist["fx"] * x * radial + dist["cx"]).astype(int)
    ys = np.round(dist["fy"] * y * radial + dist["cy"]).astype(int)
    inside = (xs >= 0) & (xs < dist["w"]) & (ys >= 0) & (ys < dist["h"])
    out = np.zeros((height, width), bool)
    out[inside] = mask[ys[inside], xs[inside]]
    return out


def _containment(a: np.ndarray, b: np.ndarray) -> float:
    smaller = min(int(a.sum()), int(b.sum()))
    return int(np.count_nonzero(a & b)) / smaller if smaller else 0.0


def select_mask(confident: np.ndarray, masks: list[np.ndarray], cfg: Config) -> tuple[int, list[dict]]:
    """The mask covering the largest share of confident depth; among nested near-ties, the tighter mask."""
    n_conf = int(confident.sum())
    candidates = []
    for m in masks:
        inter = int(np.count_nonzero(confident & m))
        candidates.append({"recall": inter / n_conf if n_conf else 0.0,
                           "precision": inter / int(m.sum()) if m.any() else 0.0})
    order = sorted(range(len(masks)), key=lambda i: (candidates[i]["recall"], candidates[i]["precision"]), reverse=True)
    selected = order[0]
    for j in order[1:]:
        if candidates[selected]["recall"] - candidates[j]["recall"] > cfg.recall_tie:
            break
        if (_containment(masks[selected], masks[j]) > cfg.nested_containment
                and candidates[j]["precision"] > candidates[selected]["precision"]):
            selected = j
    return selected, candidates


def is_main_subject(masks: np.ndarray, selected: int, cfg: Config) -> bool:
    """Whether the selected mask's object is at least `main_subject_ratio` times larger than every other object.

    Masks smaller than `min_mask_pixels` are ignored; masks nested at least `nested_containment` inside one
    another count as one object whose extent is its largest mask.
    """
    flat = [m.reshape(-1) for m in masks]
    areas = [int(f.sum()) for f in flat]
    keep = [i for i, a in enumerate(areas) if a >= cfg.min_mask_pixels]
    if selected not in keep:
        return False
    parent = {i: i for i in keep}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a in range(len(keep)):
        for b in range(a + 1, len(keep)):
            i, j = keep[a], keep[b]
            if np.count_nonzero(flat[i] & flat[j]) / min(areas[i], areas[j]) >= cfg.nested_containment:
                parent[root(i)] = root(j)
    extent = {}
    for i in keep:
        extent[root(i)] = max(extent.get(root(i), 0), areas[i])
    own = extent[root(selected)]
    others = [e for r, e in extent.items() if r != root(selected)]
    return not others or own >= cfg.main_subject_ratio * max(others)


def run_masks(scene: Scene, model_dir: Path, names: list[str], gating_dir: Path, mvs_dir: Path, cfg: Config,
              out_dir: Path) -> dict[str, dict]:
    """Choose the object mask of every view and record the per-frame object checks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    original = pycolmap.Reconstruction(str(model_dir))
    cameras = {image.name: original.cameras[image.camera_id] for image in original.images.values()}
    report = {}
    for name in tqdm(names, desc="mask selection"):
        view = scene.views[scene.index[name]]
        _, confident = confident_depth(mvs_dir, name, cfg)
        source_masks = load_masks(gating_dir, name)
        if len(source_masks) == 0:
            report[name] = {"status": "no_masks"}
            continue
        dist = _distortion(cameras[name])
        warped = [warp_to_undistorted(m, dist, view.K, view.width, view.height) for m in source_masks]
        selected, candidates = select_mask(confident, warped, cfg)
        n_conf = int(confident.sum())
        status = "low_mvs_support" if n_conf < cfg.min_mask_mvs_pixels else "ok"
        Image.fromarray(warped[selected].astype(np.uint8) * 255).save(out_dir / f"{stem(name)}.png")
        report[name] = {
            "status": status,
            "selected": selected,
            "confident_pixels": n_conf,
            "mask_recall": candidates[selected]["recall"],
            "mask_area_fraction": float(warped[selected].mean()),
            "main_subject": is_main_subject(source_masks, selected, cfg),
            "candidates": candidates,
        }
    (out_dir / "masks.json").write_text(json.dumps(report, indent=1))
    return report
