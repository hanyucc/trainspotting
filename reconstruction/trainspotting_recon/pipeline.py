"""End-to-end reconstruction of one object from a set of photos and a text prompt."""

import json
from pathlib import Path

from .config import Config
from .export import export_frames
from .gating import run_gating
from .images import image_size, stage_images
from .masks import run_masks
from .matching import load_matching, run_matching
from .mvs import run_mvs
from .scene import Scene
from .sfm import run_sfm
from .verification import run_verification
from .view_selection import select_views


def _read_json(path: Path):
    return json.loads(path.read_text())


def run(image_dir: Path, prompt: str, out_dir: Path, weights_dir: Path, device: str = "cuda",
        cfg: Config = Config()) -> dict:
    """Run every stage; a stage whose outputs exist in `out_dir` is not run again."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = stage_images(image_dir.resolve(), out_dir / "images")
    (out_dir / "images.json").write_text(json.dumps(sources, indent=1))
    image_dir = out_dir / "images"
    names = list(sources)
    sizes = {name: image_size(image_dir / name) for name in names}
    summary = {"prompt": prompt, "images": len(names)}

    def finish(status: str) -> dict:
        summary["status"] = status
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        return summary

    gating_dir = out_dir / "gating"
    if (gating_dir / "gating.json").is_file():
        kept = [n for n, r in _read_json(gating_dir / "gating.json").items() if r["kept"]]
    else:
        kept = run_gating(image_dir, names, prompt, gating_dir, cfg, device)
    summary["object_gating"] = len(kept)
    if len(kept) < 2:
        return finish("too_few_images_after_gating")

    matching_dir = out_dir / "matching"
    if not (matching_dir / "matching.json").is_file():
        run_matching(image_dir, kept, sizes, cfg, weights_dir, matching_dir, device)
    keypoints, pairs = load_matching(matching_dir)

    sfm_dir = out_dir / "sfm"
    model_dir, dense_dir = sfm_dir / "merge" / "largest", sfm_dir / "dense"
    if not (dense_dir / "sparse" / "images.bin").is_file():
        try:
            model_dir, dense_dir = run_sfm(image_dir, kept, sizes, keypoints, pairs, cfg, sfm_dir)
        except RuntimeError as error:
            summary["error"] = str(error)
            return finish("structure_from_motion_failed")
    scene = Scene(dense_dir / "sparse")
    summary["registered"] = len(scene.views)

    selection_dir = out_dir / "view_selection"
    if (selection_dir / "valid_images.txt").is_file():
        selected = (selection_dir / "valid_images.txt").read_text().split("\n")
        selected = [n for n in selected if n]
    else:
        selected = select_views(scene, cfg, selection_dir)
        (selection_dir / "valid_images.txt").write_text("".join(f"{n}\n" for n in selected))
    summary["view_selection"] = len(selected)
    if len(selected) < 2:
        return finish("too_few_views_after_selection")

    mvs_dir = out_dir / "mvs"
    if (mvs_dir / "mvs.json").is_file():
        with_depth = [n for n in _read_json(mvs_dir / "mvs.json")]
    else:
        with_depth = run_mvs(scene, dense_dir / "images", selected, cfg, weights_dir, mvs_dir, device)
    summary["mvs_depth"] = len(with_depth)

    verified_dir = out_dir / "verified_depth"
    if (verified_dir / "verification.json").is_file():
        verified = [n for n, r in _read_json(verified_dir / "verification.json").items() if r["status"] == "ok"]
    else:
        verified = run_verification(scene, dense_dir / "images", with_depth, mvs_dir, cfg, verified_dir, device)
    summary["verified_depth"] = len(verified)

    mask_dir = out_dir / "masks"
    if (mask_dir / "masks.json").is_file():
        mask_report = _read_json(mask_dir / "masks.json")
    else:
        mask_report = run_masks(scene, model_dir, verified, gating_dir, mvs_dir, cfg, mask_dir)

    frames = export_frames(scene, dense_dir, verified, sources, verified_dir, mask_dir, mask_report, cfg,
                           out_dir / "final")
    summary["final_frames"] = len(frames)
    return finish("ok" if frames else "too_few_final_frames")
