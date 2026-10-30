"""Pair retrieval with MASt3R and dense matching with RoMa v2."""

import json
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .config import Config

MAST3R_URL = "https://download.europe.naverlabs.com/ComputerVision/MASt3R/"
MAST3R_NAME = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
GEOMETRY_SIZE = 1000.0


@dataclass
class PairMatches:
    name0: str
    name1: str
    xy0: np.ndarray
    xy1: np.ndarray
    scores: np.ndarray
    epipolar_errors: np.ndarray
    matches: np.ndarray | None = None
    stats: dict = field(default_factory=dict)


def _download(url: str, path: Path) -> Path:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(path))
    return path


def _retrieval_inputs(image_root: Path, names: list[str], cache_dir: Path) -> list[str]:
    """Image paths readable by the MASt3R loader, which accepts JPEG and PNG only."""
    paths = []
    for name in names:
        path = image_root / name
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            converted = cache_dir / f"{path.stem}.png"
            if not converted.is_file():
                cache_dir.mkdir(parents=True, exist_ok=True)
                with Image.open(path) as image:
                    image.convert("RGB").save(converted)
            path = converted
        paths.append(str(path))
    return paths


def retrieval_pairs(image_root: Path, names: list[str], cfg: Config, weights_dir: Path, cache_dir: Path,
                    device: str) -> list[tuple[str, str]]:
    """Pair every image with its most similar images under the MASt3R-SfM retrieval model."""
    if len(names) < 2:
        return []
    from mast3r.model import AsymmetricMASt3R
    from mast3r.retrieval.processor import Retriever

    backbone_path = _download(MAST3R_URL + f"{MAST3R_NAME}.pth", weights_dir / f"{MAST3R_NAME}.pth")
    retrieval_path = _download(MAST3R_URL + f"{MAST3R_NAME}_retrieval_trainingfree.pth",
                               weights_dir / f"{MAST3R_NAME}_retrieval_trainingfree.pth")
    _download(MAST3R_URL + f"{MAST3R_NAME}_retrieval_codebook.pkl",
              weights_dir / f"{MAST3R_NAME}_retrieval_codebook.pkl")

    backbone = AsymmetricMASt3R.from_pretrained(str(backbone_path)).to(device)
    with torch.serialization.safe_globals([Namespace]):
        retriever = Retriever(str(retrieval_path), backbone=backbone, device=device)
    with torch.no_grad():
        similarity = np.asarray(retriever(_retrieval_inputs(image_root, names, cache_dir)), dtype=np.float32)
    del retriever, backbone
    torch.cuda.empty_cache()

    k = min(cfg.retrieval_neighbours, len(names) - 1)
    pairs = set()
    for i in range(len(names)):
        neighbours = [j for j in np.argsort(similarity[i])[::-1] if j != i][:k]
        for j in neighbours:
            a, b = sorted((i, int(j)))
            pairs.add((names[a], names[b]))
    return sorted(pairs)


def _to_pixel(points: np.ndarray, width: int, height: int) -> np.ndarray:
    return (points + 1.0) * 0.5 * np.array([width, height], dtype=np.float32)


def _sample_grid(grid: np.ndarray, points: np.ndarray) -> np.ndarray:
    grid = grid[:, :, None] if grid.ndim == 2 else grid
    h, w = grid.shape[:2]
    map_x = (((points[:, 0] + 1.0) * w - 1.0) * 0.5).reshape(-1, 1).astype(np.float32)
    map_y = (((points[:, 1] + 1.0) * h - 1.0) * 0.5).reshape(-1, 1).astype(np.float32)
    sampled = cv2.remap(grid, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))
    return sampled.reshape(-1, grid.shape[2])


def _cycle_consistent(p0, p1, warp_ab, warp_ba, overlap_ab, overlap_ba, fraction):
    """Forward-backward consistency in normalized coordinates and the two-way certainty."""
    back1 = _sample_grid(warp_ab, p0)
    back0 = _sample_grid(warp_ba, p1)
    c_ab = _sample_grid(overlap_ab, p0).ravel()
    c_ba = _sample_grid(overlap_ba, p1).ravel()
    limit = 2.0 * fraction
    valid = (np.isfinite(back0).all(1) & np.isfinite(back1).all(1) & np.isfinite(c_ab) & np.isfinite(c_ba)
             & (np.abs(back0 - p0) <= limit).all(1) & (np.abs(back1 - p1) <= limit).all(1))
    return valid, np.minimum(c_ab, c_ba)


def _epipolar_errors(F: np.ndarray, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    h0 = np.c_[x0, np.ones(len(x0))].astype(np.float64)
    h1 = np.c_[x1, np.ones(len(x1))].astype(np.float64)
    l1 = h0 @ F.T
    l0 = h1 @ F
    r = np.sum(h1 * l1, axis=1)
    d = 1.0 / np.maximum(l0[:, 0] ** 2 + l0[:, 1] ** 2, 1e-12) + 1.0 / np.maximum(l1[:, 0] ** 2 + l1[:, 1] ** 2, 1e-12)
    return np.sqrt(r * r * d).astype(np.float32)


def _epipolar_inliers(xy0, xy1, size0, size1, cfg: Config):
    """MAGSAC++ fundamental matrix on coordinates rescaled to a fixed 1000 x 1000 frame."""
    g0 = xy0 * (GEOMETRY_SIZE / np.asarray(size0, np.float32))
    g1 = xy1 * (GEOMETRY_SIZE / np.asarray(size1, np.float32))
    none = np.zeros(len(g0), bool), np.full(len(g0), np.inf, np.float32)
    if len(g0) < 8:
        return none
    cv2.setRNGSeed(0)
    try:
        F, mask = cv2.findFundamentalMat(g0, g1, cv2.USAC_MAGSAC, cfg.epipolar_fraction * GEOMETRY_SIZE,
                                         cfg.magsac_confidence, cfg.magsac_max_iters)
    except cv2.error:
        return none
    if F is None or mask is None or F.shape != (3, 3):
        return none
    return mask.ravel().astype(bool), _epipolar_errors(F, g0, g1)


def _match_pair(model, image_root: Path, name0: str, name1: str, size0, size1, cfg: Config) -> PairMatches:
    preds = model.match(str(image_root / name0), str(image_root / name1))
    sampled, certainty, _, _ = model.sample(preds, cfg.samples_per_pair)
    p0 = sampled[:, :2].float().cpu().numpy()
    p1 = sampled[:, 2:].float().cpu().numpy()
    grid = lambda key: preds[key][0].float().cpu().numpy()
    cycle_ok, overlap = _cycle_consistent(
        p0, p1, grid("warp_AB"), grid("warp_BA"),
        preds["overlap_AB"][0, :, :, 0].float().cpu().numpy(),
        preds["overlap_BA"][0, :, :, 0].float().cpu().numpy(),
        cfg.cycle_fraction)
    scores = np.minimum(certainty.float().cpu().numpy(), overlap)
    keep = cycle_ok & (scores >= cfg.certainty_threshold)
    xy0 = _to_pixel(p0[keep], *size0)
    xy1 = _to_pixel(p1[keep], *size1)
    inliers, errors = _epipolar_inliers(xy0, xy1, size0, size1, cfg)
    return PairMatches(name0, name1, xy0[inliers], xy1[inliers], scores[keep][inliers], errors[inliers],
                       stats={"sampled": len(p0), "certain_and_cycle_consistent": int(keep.sum()),
                              "epipolar_inliers": int(inliers.sum())})


def _merge_keypoints(points: np.ndarray, scores: np.ndarray, size, radius: float):
    """Greedy, score-ordered merge of nearby endpoints into shared keypoints (radius in normalized units)."""
    if len(points) == 0:
        return points.copy(), np.zeros(0, np.uint32)
    norm = points / np.asarray(size, np.float32)
    order = np.lexsort((np.arange(len(points)), norm[:, 1], norm[:, 0], -scores))
    cells: dict[tuple[int, int], list[int]] = {}
    reps: list[np.ndarray] = []
    assign = np.empty(len(points), np.uint32)
    r2 = radius * radius
    for idx in order:
        p = norm[idx]
        cell = (int(np.floor(p[0] / radius)), int(np.floor(p[1] / radius)))
        best, best_d = -1, r2
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for rep in cells.get((cell[0] + dx, cell[1] + dy), []):
                    d = float(np.sum((reps[rep] - p) ** 2))
                    if d <= r2 and (best < 0 or d < best_d or (d == best_d and rep < best)):
                        best, best_d = rep, d
        if best < 0:
            best = len(reps)
            reps.append(p.copy())
            cells.setdefault(cell, []).append(best)
        assign[idx] = best
    return np.asarray(reps, np.float32) * np.asarray(size, np.float32), assign


def _one_to_one(matches: np.ndarray, scores: np.ndarray, errors: np.ndarray) -> np.ndarray:
    order = np.lexsort((matches[:, 1], matches[:, 0], -scores, errors))
    used0, used1, keep = set(), set(), []
    for idx in order:
        a, b = int(matches[idx, 0]), int(matches[idx, 1])
        if a in used0 or b in used1:
            continue
        used0.add(a)
        used1.add(b)
        keep.append(idx)
    return matches[np.asarray(keep, np.int64)] if keep else np.zeros((0, 2), np.uint32)


def build_tracks(pairs: list[PairMatches], sizes: dict[str, tuple[int, int]], radius: float) -> dict[str, np.ndarray]:
    """Shared keypoints per image; sets `matches` (keypoint index pairs) on every pair."""
    endpoints: dict[str, list[tuple[int, int]]] = {name: [] for name in sizes}
    for k, pair in enumerate(pairs):
        endpoints[pair.name0].append((k, 0))
        endpoints[pair.name1].append((k, 1))
    keypoints, assignment = {}, {}
    for name, refs in endpoints.items():
        if not refs:
            keypoints[name] = np.zeros((0, 2), np.float32)
            continue
        points = np.concatenate([pairs[k].xy0 if side == 0 else pairs[k].xy1 for k, side in refs])
        scores = np.concatenate([pairs[k].scores for k, _ in refs])
        keypoints[name], assign = _merge_keypoints(points, scores, sizes[name], radius)
        offset = 0
        for k, side in refs:
            n = len(pairs[k].scores)
            assignment[(k, side)] = assign[offset:offset + n]
            offset += n
    for k, pair in enumerate(pairs):
        raw = np.column_stack([assignment[(k, 0)], assignment[(k, 1)]]).astype(np.uint32)
        pair.matches = _one_to_one(raw, pair.scores, pair.epipolar_errors)
    return keypoints


def run_matching(image_root: Path, names: list[str], sizes: dict, cfg: Config, weights_dir: Path,
                 out_dir: Path, device: str) -> tuple[dict[str, np.ndarray], list[PairMatches]]:
    """Retrieve pairs, match them densely, and link the matches into shared keypoints."""
    from romav2 import RoMaV2

    out_dir.mkdir(parents=True, exist_ok=True)
    pair_names = retrieval_pairs(image_root, names, cfg, weights_dir, out_dir / "retrieval_images", device)
    torch.set_float32_matmul_precision("highest")
    model = RoMaV2()
    model.apply_setting(cfg.roma_setting)
    pairs = []
    for name0, name1 in tqdm(pair_names, desc="dense matching"):
        with torch.inference_mode():
            pairs.append(_match_pair(model, image_root, name0, name1, sizes[name0], sizes[name1], cfg))
    del model
    torch.cuda.empty_cache()

    keypoints = build_tracks(pairs, sizes, cfg.keypoint_merge_fraction)
    np.savez_compressed(out_dir / "keypoints.npz", **{f"k{i}": keypoints[n] for i, n in enumerate(names)})
    np.savez_compressed(out_dir / "matches.npz", **{f"m{i}": p.matches for i, p in enumerate(pairs)})
    (out_dir / "matching.json").write_text(json.dumps({
        "images": names,
        "pairs": [{"name0": p.name0, "name1": p.name1, "matches": int(len(p.matches)), **p.stats} for p in pairs],
    }, indent=1))
    return keypoints, pairs


def load_matching(out_dir: Path) -> tuple[dict[str, np.ndarray], list[tuple[str, str, np.ndarray]]]:
    info = json.loads((out_dir / "matching.json").read_text())
    with np.load(out_dir / "keypoints.npz") as k, np.load(out_dir / "matches.npz") as m:
        keypoints = {n: k[f"k{i}"] for i, n in enumerate(info["images"])}
        pairs = [(p["name0"], p["name1"], m[f"m{i}"]) for i, p in enumerate(info["pairs"])]
    return keypoints, pairs
