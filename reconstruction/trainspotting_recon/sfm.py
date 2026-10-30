"""Structure from motion with COLMAP on the shared keypoints."""

import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pycolmap

from .config import Config
from .images import initial_focal

SIMPLE_RADIAL = 2
PAIR_ID_FACTOR = 2147483647

SCHEMA = """
CREATE TABLE rigs (rig_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, ref_sensor_id INTEGER NOT NULL,
    ref_sensor_type INTEGER NOT NULL);
CREATE TABLE rig_sensors (rig_id INTEGER NOT NULL, sensor_id INTEGER NOT NULL, sensor_type INTEGER NOT NULL,
    sensor_from_rig BLOB, FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE);
CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, model INTEGER NOT NULL,
    width INTEGER NOT NULL, height INTEGER NOT NULL, params BLOB, prior_focal_length INTEGER NOT NULL);
CREATE TABLE frames (frame_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, rig_id INTEGER NOT NULL,
    FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE);
CREATE TABLE frame_data (frame_id INTEGER NOT NULL, data_id INTEGER NOT NULL, sensor_id INTEGER NOT NULL,
    sensor_type INTEGER NOT NULL, FOREIGN KEY(frame_id) REFERENCES frames(frame_id) ON DELETE CASCADE);
CREATE TABLE images (image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL, CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < 2147483647),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));
CREATE TABLE pose_priors (pose_prior_id INTEGER PRIMARY KEY NOT NULL, corr_data_id INTEGER NOT NULL,
    corr_sensor_id INTEGER NOT NULL, corr_sensor_type INTEGER NOT NULL, position BLOB, position_covariance BLOB,
    gravity BLOB, coordinate_system INTEGER NOT NULL);
CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, cols INTEGER NOT NULL,
    data BLOB, FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
CREATE TABLE descriptors (image_id INTEGER PRIMARY KEY NOT NULL, type INTEGER NOT NULL, rows INTEGER NOT NULL,
    cols INTEGER NOT NULL, data BLOB, FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
CREATE TABLE matches (pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB);
CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL,
    cols INTEGER NOT NULL, data BLOB, config INTEGER NOT NULL, F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB);
CREATE UNIQUE INDEX rig_ref_sensor_assignment ON rigs(ref_sensor_id, ref_sensor_type);
CREATE UNIQUE INDEX rig_sensor_assignment ON rig_sensors(sensor_id, sensor_type);
CREATE UNIQUE INDEX frame_sensor_assignment ON frame_data(data_id, sensor_type);
CREATE UNIQUE INDEX index_name ON images(name);
CREATE UNIQUE INDEX pose_prior_data_assignment ON pose_priors(corr_data_id, corr_sensor_id, corr_sensor_type);
"""


COLMAP = Path(sys.executable).with_name("colmap")


def _colmap(*args: str) -> subprocess.CompletedProcess:
    binary = str(COLMAP) if COLMAP.is_file() else "colmap"
    return subprocess.run([binary, *map(str, args)], capture_output=True, text=True)


def write_database(db_path: Path, image_root: Path, names: list[str], sizes: dict, keypoints: dict,
                   pairs: list[tuple[str, str, np.ndarray]], cfg: Config) -> Path:
    """One SIMPLE_RADIAL camera per image, the shared keypoints, and the raw matches."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    ids = {}
    for i, name in enumerate(names, start=1):
        w, h = sizes[name]
        focal, has_prior = initial_focal(image_root / name, w, h, cfg.default_focal_factor)
        params = np.array([focal, w / 2, h / 2, 0.0], np.float64)
        con.execute("INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)", (i, SIMPLE_RADIAL, w, h, params.tobytes(), int(has_prior)))
        con.execute("INSERT INTO rigs VALUES (?, ?, ?)", (i, i, 0))
        con.execute("INSERT INTO images VALUES (?, ?, ?)", (i, name, i))
        con.execute("INSERT INTO frames VALUES (?, ?)", (i, i))
        con.execute("INSERT INTO frame_data VALUES (?, ?, ?, ?)", (i, i, i, 0))
        kp = np.asarray(keypoints[name], np.float32).reshape(-1, 2)
        con.execute("INSERT INTO keypoints VALUES (?, ?, ?, ?)", (i, len(kp), 2, kp.tobytes()))
        ids[name] = i
    lines = []
    for name0, name1, matches in pairs:
        m = np.asarray(matches, np.uint32).reshape(-1, 2)
        if len(m) == 0:
            continue
        i0, i1 = ids[name0], ids[name1]
        if i0 > i1:
            i0, i1, m = i1, i0, m[:, ::-1].copy()
        con.execute("INSERT INTO matches VALUES (?, ?, ?, ?)", (i0 * PAIR_ID_FACTOR + i1, len(m), 2, m.tobytes()))
        lines.append(f"{name0} {name1}")
    con.commit()
    con.close()
    pairs_path = db_path.with_name("pairs.txt")
    pairs_path.write_text("\n".join(lines) + "\n")
    return pairs_path


def verify_and_map(db_path: Path, pairs_path: Path, image_root: Path, mapper_dir: Path, cfg: Config) -> list[Path]:
    """Two-view geometric verification followed by COLMAP incremental mapping."""
    pycolmap.verify_matches(db_path, pairs_path, options={
        "ransac": {"max_num_trials": cfg.verify_max_trials, "min_inlier_ratio": cfg.verify_min_inlier_ratio}})
    mapper_dir.mkdir(parents=True, exist_ok=True)
    result = _colmap("mapper", "--database_path", db_path, "--image_path", image_root, "--output_path", mapper_dir)
    if result.returncode != 0:
        raise RuntimeError(f"COLMAP mapper failed:\n{result.stderr[-2000:]}")
    return sorted((p for p in mapper_dir.iterdir() if (p / "images.bin").is_file()), key=lambda p: int(p.name))


def _image_names(model: Path) -> set[str]:
    return {image.name for image in pycolmap.Reconstruction(model).images.values()}


def merge_components(components: list[Path], work_dir: Path, cfg: Config) -> Path:
    """Repeatedly merge the two models sharing the most images; return the largest model."""
    work_dir.mkdir(parents=True, exist_ok=True)
    groups = [{"path": c, "names": _image_names(c), "merged": False} for c in components]
    failed: set[tuple[str, str]] = set()
    history, step = [], 0
    while True:
        best = None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                key = tuple(sorted((str(groups[a]["path"]), str(groups[b]["path"]))))
                shared = len(groups[a]["names"] & groups[b]["names"])
                if key not in failed and shared >= cfg.min_shared_images and (best is None or shared > best[0]):
                    best = (shared, a, b, key)
        if best is None:
            break
        shared, a, b, key = best
        out = work_dir / f"merge_{step:04d}"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        result = _colmap("model_merger", "--input_path1", groups[a]["path"], "--input_path2", groups[b]["path"],
                         "--output_path", out)
        log = result.stdout + result.stderr
        if result.returncode != 0 or "=> Merge failed" in log or not (out / "images.bin").is_file():
            failed.add(key)
            shutil.rmtree(out, ignore_errors=True)
            history.append({"status": "failed", "shared_images": shared})
            continue
        merged = {"path": out, "names": _image_names(out), "merged": True}
        groups = [g for k, g in enumerate(groups) if k not in (a, b)] + [merged]
        history.append({"status": "merged", "shared_images": shared, "images": len(merged["names"])})
        step += 1

    largest = max(groups, key=lambda g: len(g["names"]))
    final = work_dir / "largest"
    shutil.rmtree(final, ignore_errors=True)
    final.mkdir(parents=True)
    if largest["merged"]:
        result = _colmap("bundle_adjuster", "--input_path", largest["path"], "--output_path", final,
                         "--BundleAdjustmentCeres.max_num_images_direct_dense_cpu_solver", 1000,
                         "--BundleAdjustmentCeres.max_num_images_direct_sparse_cpu_solver", 2000)
        if result.returncode != 0:
            raise RuntimeError(f"COLMAP bundle_adjuster failed:\n{result.stderr[-2000:]}")
    else:
        for f in largest["path"].iterdir():
            if f.is_file():
                shutil.copy2(f, final / f.name)
    (work_dir / "merge.json").write_text(json.dumps(
        {"components": len(components), "history": history, "largest_images": len(largest["names"])}, indent=1))
    return final


def undistort(model: Path, image_root: Path, dense_dir: Path, cfg: Config) -> Path:
    """Undistort the model and its images to PINHOLE cameras, capped at `max_image_size`."""
    shutil.rmtree(dense_dir, ignore_errors=True)
    dense_dir.mkdir(parents=True)
    result = _colmap("image_undistorter", "--image_path", image_root, "--input_path", model,
                     "--output_path", dense_dir, "--output_type", "COLMAP", "--max_image_size", cfg.max_image_size)
    if result.returncode != 0:
        raise RuntimeError(f"COLMAP image_undistorter failed:\n{result.stderr[-2000:]}")
    return dense_dir


def run_sfm(image_root: Path, names: list[str], sizes: dict, keypoints: dict, pairs, cfg: Config,
            out_dir: Path) -> tuple[Path, Path]:
    """Database, verification, mapping, component merging and undistortion.

    Returns the largest model (original cameras) and the undistorted workspace.
    """
    pairs_path = write_database(out_dir / "database.db", image_root, names, sizes, keypoints, pairs, cfg)
    components = verify_and_map(out_dir / "database.db", pairs_path, image_root, out_dir / "mapper", cfg)
    if not components:
        raise RuntimeError("COLMAP did not reconstruct any model")
    model = merge_components(components, out_dir / "merge", cfg)
    dense = undistort(model, image_root, out_dir / "dense", cfg)
    return model, dense
