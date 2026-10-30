"""Verification of MVS depth against monocular MoGe-2 geometry."""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import label
from tqdm import tqdm

from .config import Config
from .images import load_rgb, stem
from .mvs import confident_depth
from .scene import Scene

MOGE_MODEL = "Ruicheng/moge-2-vitl-normal"


def _remove_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    labels, count = label(mask, structure=np.ones((3, 3), np.uint8))
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = np.flatnonzero(sizes >= min_pixels)
    return np.isin(labels, keep[keep != 0])


def _global_consistency(mvs: np.ndarray, mono: np.ndarray, cfg: Config):
    """MVS pixels whose log depth ratio to MoGe-2 lies within a robust band around the median ratio."""
    overlap = np.isfinite(mono) & (mono > 0) & np.isfinite(mvs) & (mvs > 0)
    if not overlap.any():
        return np.zeros_like(overlap), {"overlap_pixels": 0, "mad_log_ratio": np.inf, "global_inlier_fraction": 0.0}
    ratio = np.full(mvs.shape, np.nan)
    ratio[overlap] = np.log(mono[overlap]) - np.log(mvs[overlap])
    median = float(np.median(ratio[overlap]))
    mad = float(np.median(np.abs(ratio[overlap] - median)))
    tolerance = max(cfg.log_ratio_sigma * 1.4826 * mad, cfg.log_ratio_floor)
    inliers = overlap & (np.abs(ratio - median) <= tolerance)
    return inliers, {"overlap_pixels": int(overlap.sum()), "mad_log_ratio": mad,
                     "global_inlier_fraction": float(inliers.sum() / overlap.sum())}


def _gate(stats: dict, verified_pixels: int, cfg: Config) -> str | None:
    if stats["overlap_pixels"] < cfg.min_overlap_pixels:
        return "too_few_overlap_pixels"
    if stats["mad_log_ratio"] > cfg.max_mad_log_ratio:
        return "inconsistent_scale"
    if stats["global_inlier_fraction"] < cfg.min_global_inlier_fraction:
        return "too_few_global_inliers"
    if verified_pixels < cfg.min_confident_pixels:
        return "too_few_verified_pixels"
    return None


def _verify_view(model, scene: Scene, image_dir: Path, mvs_dir: Path, name: str, cfg: Config, device: str):
    import utils3d
    from .vendor.vigeo import multi_scale_filter_depth

    depth, confident = confident_depth(mvs_dir, name, cfg)
    prior = np.where(confident, depth, 0.0).astype(np.float32)
    if int(confident.sum()) < cfg.min_confident_pixels:
        return None, {"status": "too_few_confident_pixels"}

    view = scene.views[scene.index[name]]
    h, w = prior.shape
    K = view.K.astype(np.float64)
    K_norm = K.copy()
    K_norm[0] /= w
    K_norm[1] /= h
    focal = 1.0 / np.sqrt(1.0 / K_norm[0, 0] ** 2 + 1.0 / K_norm[1, 1] ** 2)
    fov_x = float(np.degrees(2.0 * np.arctan(w / (2.0 * K[0, 0]))))

    image = torch.from_numpy(load_rgb(image_dir / name)).permute(2, 0, 1).float().div(255.0)[None].to(device)
    prior_t = torch.from_numpy(prior)[None].to(device)
    with torch.inference_mode():
        out = model.infer(image, fov_x=torch.tensor([fov_x], device=device), apply_mask=False, force_projection=True)
        valid_mono = out["mask"].bool()
        mono = out["depth"].float().masked_fill(~valid_mono, 0.0)
        points_mvs = utils3d.pt.depth_map_to_point_map(prior_t, intrinsics=torch.from_numpy(K_norm).float()[None].to(device))
        spherical = multi_scale_filter_depth(out["points"].float(), points_mvs, valid_mono & (prior_t > 1e-4),
                                             focal=torch.tensor([focal], dtype=torch.float32, device=device),
                                             trunc=cfg.spherical_truncation)
    mono = mono.squeeze().cpu().numpy()
    mono = np.where(np.isfinite(mono) & (mono > 0), mono, np.nan)
    spherical = spherical.squeeze().cpu().numpy().astype(bool)

    inliers, stats = _global_consistency(prior, mono, cfg)
    verified = _remove_small_components(inliers & spherical, cfg.min_component_pixels)
    reason = _gate(stats, int(verified.sum()), cfg)
    stats.update(verified_pixels=int(verified.sum()), status=reason or "ok")
    if reason is not None:
        return None, stats
    return np.where(verified, prior, 0.0).astype(np.float32), stats


def run_verification(scene: Scene, image_dir: Path, names: list[str], mvs_dir: Path, cfg: Config,
                     out_dir: Path, device: str) -> list[str]:
    """Verified depth for every view with MVS depth; views failing the frame checks get none."""
    from moge.model.v2 import MoGeModel

    out_dir.mkdir(parents=True, exist_ok=True)
    model = MoGeModel.from_pretrained(MOGE_MODEL).to(device).eval()
    report, done = {}, []
    for name in tqdm(names, desc="depth verification"):
        verified, stats = _verify_view(model, scene, image_dir, mvs_dir, name, cfg, device)
        report[name] = stats
        if verified is not None:
            np.savez_compressed(out_dir / f"{stem(name)}.npz", depth=verified)
            done.append(name)
    del model
    torch.cuda.empty_cache()
    (out_dir / "verification.json").write_text(json.dumps(report, indent=1))
    return done


def load_verified(out_dir: Path, name: str) -> np.ndarray:
    with np.load(out_dir / f"{stem(name)}.npz") as data:
        return data["depth"]
