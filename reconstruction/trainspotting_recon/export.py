"""Final frames and their reconstruction."""

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Config
from .images import stem
from .scene import Scene
from .verification import load_verified


def frame_checks(view, verified: np.ndarray, mask: np.ndarray, mask_info: dict, cfg: Config) -> str | None:
    """None if the frame passes every check, otherwise the first failed check."""
    if mask_info.get("status") != "ok":
        return mask_info.get("status", "no_mask")
    if not mask.any():
        return "empty_mask"
    if np.count_nonzero((verified > 0) & mask) / np.count_nonzero(mask) < cfg.min_verified_fraction:
        return "low_verified_depth"
    if not mask_info["main_subject"]:
        return "not_main_subject"
    if not cfg.min_focal_ratio <= view.K[0, 0] / view.width <= cfg.max_focal_ratio:
        return "implausible_focal_length"
    return None


def _write_text_model(scene: Scene, names: list[str], out_dir: Path) -> None:
    """COLMAP text model restricted to `names`; points keep only observations from these images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = scene.reconstruction
    keep = {scene.views[scene.index[n]].image_id for n in names}
    tracks = {}
    for pid, point in rec.points3D.items():
        elements = [(e.image_id, e.point2D_idx) for e in point.track.elements if e.image_id in keep]
        if len({i for i, _ in elements}) >= 2:
            tracks[pid] = elements

    with open(out_dir / "cameras.txt", "w") as f:
        for camera_id in sorted({rec.images[i].camera_id for i in keep}):
            camera = rec.cameras[camera_id]
            fx, fy, cx, cy = camera.params
            f.write(f"{camera.camera_id} PINHOLE {camera.width} {camera.height} {fx!r} {fy!r} {cx!r} {cy!r}\n")
    with open(out_dir / "images.txt", "w") as f:
        for image_id in sorted(keep):
            image = rec.images[image_id]
            pose = image.cam_from_world()
            qx, qy, qz, qw = pose.rotation.quat
            tx, ty, tz = pose.translation
            f.write(f"{image_id} {qw!r} {qx!r} {qy!r} {qz!r} {tx!r} {ty!r} {tz!r} {image.camera_id} {image.name}\n")
            f.write(" ".join(f"{p.xy[0]!r} {p.xy[1]!r} {p.point3D_id if p.point3D_id in tracks else -1}"
                             for p in image.points2D) + "\n")
    with open(out_dir / "points3D.txt", "w") as f:
        for pid, elements in tracks.items():
            point = rec.points3D[pid]
            x, y, z = point.xyz
            r, g, b = (int(c) for c in point.color)
            track = " ".join(f"{i} {j}" for i, j in elements)
            f.write(f"{pid} {x!r} {y!r} {z!r} {r} {g} {b} {point.error!r} {track}\n")


def export_frames(scene: Scene, dense_dir: Path, names: list[str], sources: dict[str, str], verified_dir: Path,
                  mask_dir: Path, mask_report: dict, cfg: Config, out_dir: Path) -> list[dict]:
    """Write the frames that pass every check: image, verified depth, mask, camera and pose."""
    shutil.rmtree(out_dir, ignore_errors=True)
    for sub in ("images", "depths", "masks"):
        (out_dir / sub).mkdir(parents=True)
    frames, dropped = [], {}
    for name in names:
        view = scene.views[scene.index[name]]
        info = mask_report.get(name, {"status": "no_masks"})
        mask_path = mask_dir / f"{stem(name)}.png"
        verified = load_verified(verified_dir, name)
        mask = np.asarray(Image.open(mask_path)) > 0 if mask_path.is_file() else np.zeros_like(verified, bool)
        reason = frame_checks(view, verified, mask, info, cfg)
        if reason is not None:
            dropped[sources[name]] = reason
            continue
        shutil.copyfile(dense_dir / "images" / name, out_dir / "images" / name)
        np.save(out_dir / "depths" / f"{stem(name)}.npy", verified)
        shutil.copyfile(mask_path, out_dir / "masks" / f"{stem(name)}.png")
        frames.append({
            "name": name,
            "source": sources[name],
            "image": f"images/{name}",
            "depth": f"depths/{stem(name)}.npy",
            "mask": f"masks/{stem(name)}.png",
            "width": view.width,
            "height": view.height,
            "intrinsics": view.K.tolist(),
            "camera_to_world": np.linalg.inv(view.world_to_camera).tolist(),
            "verified_depth_fraction": float(np.count_nonzero((verified > 0) & mask) / np.count_nonzero(mask)),
            "mask_recall": info["mask_recall"],
        })
    if len(frames) < cfg.min_frames:
        shutil.rmtree(out_dir)
        return []
    _write_text_model(scene, [f["name"] for f in frames], out_dir / "sparse")
    (out_dir / "frames.json").write_text(json.dumps({
        "camera_convention": "OpenCV: +x right, +y down, +z forward; camera_to_world maps camera to world",
        "depth": "float32, same size as the image, 0 where no verified depth, in the units of the reconstruction",
        "mask": "uint8 PNG, 255 on the object",
        "frames": frames,
        "dropped": dropped,
    }, indent=1))
    return frames
