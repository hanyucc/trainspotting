"""Multi-view stereo with CasDiffMVS."""

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import Config
from .images import load_rgb, stem
from .scene import Scene

CHECKPOINT_ID = "1O-Tkc9NiA2C0cqsrw0CZghUH7TUp1-pN"
CHECKPOINT_NAME = "casdiffmvs_blendmvg.ckpt"
NUM_DEPTHS = 384
SEED = 123
MODEL_ARGS = Namespace(
    numdepth_initial=48,
    numdepth=NUM_DEPTHS,
    scale=[0.0, 0.125, 0.025],
    sampling_timesteps=[0, 1, 1],
    ddim_eta=[0.0, 1.0, 1.0],
    timesteps=[1000, 1000, 1000],
    stage_iters=[1, 3, 3],
    cost_dim_stage=[4, 4, 4],
    CostNum=[0, 4, 4],
    hidden_dim=[0, 32, 20],
    context_dim=[32, 32, 16],
    unet_dim=[0, 16, 8],
    min_radius=0.125,
    max_radius=8.0,
)


def _load_model(weights_dir: Path, device: str):
    from .vendor.casdiffmvs import CasDiffMVS

    checkpoint = weights_dir / CHECKPOINT_NAME
    if not checkpoint.is_file():
        import zipfile

        import gdown

        weights_dir.mkdir(parents=True, exist_ok=True)
        archive = weights_dir / "casdiffmvs_blendmvg.zip"
        gdown.download(id=CHECKPOINT_ID, output=str(archive), quiet=False)
        with zipfile.ZipFile(archive) as z:
            z.extract(CHECKPOINT_NAME, weights_dir)
        archive.unlink()
    model = CasDiffMVS(MODEL_ARGS, test=True)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def plan_views(scene: Scene, names: list[str], cfg: Config) -> dict[str, dict]:
    """Per reference view: up to `source_views` source views ranked by shared points, and the depth range."""
    shared, _, _ = scene.covisibility()
    valid = [scene.index[n] for n in names]
    plans = {}
    for i in valid:
        view = scene.views[i]
        lo, hi = scene.depth_range(view, cfg.min_valid_depth)
        span = hi - lo
        ranked = sorted((k for k in valid if k != i and shared[i, k] > 0), key=lambda k: -shared[i, k])
        if not ranked:
            continue
        plans[view.name] = {
            "sources": [scene.views[k].name for k in ranked[:cfg.source_views]],
            "depth_range": [max(cfg.min_valid_depth, lo - cfg.depth_range_expand * span),
                            hi + cfg.depth_range_expand * span],
        }
    return plans


def _pad(image: np.ndarray, base: int = 32) -> np.ndarray:
    h, w = image.shape[:2]
    out = np.zeros((int(np.ceil(h / base) * base), int(np.ceil(w / base) * base), 3), np.float32)
    out[:h, :w] = image
    return out


def _inputs(scene: Scene, image_dir: Path, ref: str, plan: dict, device: str):
    images, projections = [], []
    for name in [ref] + plan["sources"]:
        view = scene.views[scene.index[name]]
        rgb = load_rgb(image_dir / name).astype(np.float32) / 255.0
        images.append(torch.from_numpy(_pad(rgb).transpose(2, 0, 1))[None].to(device))
        P = np.zeros((2, 4, 4), np.float32)
        P[0] = view.world_to_camera
        P[1, :3, :3] = view.K
        projections.append(P)
    projections = np.stack(projections)
    stages = {}
    for stage, factor in (("stage1", 0.125), ("stage2", 0.25), ("stage3", 0.5), ("stage4", 1.0)):
        P = projections.copy()
        P[:, 1, :2, :] *= factor
        stages[stage] = torch.from_numpy(P)[None].to(device)
    near, far = plan["depth_range"]
    depth_values = torch.from_numpy(np.linspace(1.0 / far, 1.0 / near, NUM_DEPTHS, dtype=np.float32))[None].to(device)
    return images, stages, depth_values


def run_mvs(scene: Scene, image_dir: Path, names: list[str], cfg: Config, weights_dir: Path,
            out_dir: Path, device: str) -> list[str]:
    """Depth and per-stage confidence for every selected view with at least one source view."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plans = plan_views(scene, names, cfg)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    model = _load_model(weights_dir, device)
    done = []
    for ref, plan in tqdm(plans.items(), desc="multi-view stereo"):
        view = scene.views[scene.index[ref]]
        images, stages, depth_values = _inputs(scene, image_dir, ref, plan, device)
        with torch.no_grad():
            outputs = model(images, stages, depth_values)
        h, w = view.height, view.width
        depth = outputs["depth"][-1].reshape(outputs["depth"][-1].shape[-2:])[:h, :w]
        confidences = [c.reshape(c.shape[-2:])[:h, :w] for c in outputs["photometric_confidence"]]
        np.savez_compressed(out_dir / f"{stem(ref)}.npz",
                            depth=depth.float().cpu().numpy(),
                            **{f"conf{i}": c.float().cpu().numpy() for i, c in enumerate(confidences)})
        done.append(ref)
    del model
    torch.cuda.empty_cache()
    (out_dir / "mvs.json").write_text(json.dumps(plans, indent=1))
    return done


def confident_depth(mvs_dir: Path, name: str, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """MVS depth and the mask of pixels above the confidence threshold of every stage."""
    with np.load(mvs_dir / f"{stem(name)}.npz") as data:
        depth = data["depth"]
        confident = np.isfinite(depth) & (depth > 0)
        for i, threshold in enumerate(cfg.confidence_thresholds):
            conf = data[f"conf{i}"]
            confident &= np.isfinite(conf) & (conf > threshold)
    return depth, confident
