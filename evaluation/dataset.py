"""The Trainspotting held-out evaluation samples.

Reads the released per-object folders (``<root>/<instance_id>/metadata.json`` with ``images/``,
``depths/`` and ``masks/``) and the sample list ``eval_samples.tsv`` (``instance_id`` and the
comma-separated ``image_ids`` of each 4- or 16-view sample, in evaluation order: the first image
is the reference frame).

Each sample is returned preprocessed exactly as in the paper's protocol: depth outside the object
mask set to 0, every view cropped around its principal point and resized to 512x512, and the
ground truth expressed in the first camera's frame with unit mean distance of the valid points.
"""

import csv
import json
import os

import cv2
import numpy as np
import torch

from preprocess import preprocess_frame

IMAGE_SIZE = 512


def read_samples(samples_tsv):
    with open(samples_tsv, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [{"instance_id": r["instance_id"], "image_ids": r["image_ids"].split(",")} for r in rows]


def camera_from_world(camera_to_world):
    c2w = np.asarray(camera_to_world, dtype=np.float64)
    if c2w.shape == (3, 4):
        c2w = np.vstack([c2w, [0, 0, 0, 1]])
    return np.linalg.inv(c2w).astype(np.float32)[:3]


def intrinsic_matrix(intrinsics_px):
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = intrinsics_px["fx"]
    K[1, 1] = intrinsics_px["fy"]
    K[0, 2] = intrinsics_px["cx"]
    K[1, 2] = intrinsics_px["cy"]
    return K


def read_rgb(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def normalize_to_first_camera(extrinsics, world_points, cam_points, depths, point_masks):
    """Express cameras and points in the first camera's frame and scale to unit mean distance.

    All inputs are numpy arrays over the S views: extrinsics (S, 3, 4) camera-from-world,
    world_points / cam_points (S, H, W, 3), depths (S, H, W), point_masks (S, H, W).
    """
    S = extrinsics.shape[0]
    ext_h = np.concatenate([extrinsics, np.tile(np.array([[[0, 0, 0, 1]]], dtype=np.float32), (S, 1, 1))], axis=1)
    first_inv = np.linalg.inv(ext_h[0].astype(np.float64))
    new_ext = (ext_h.astype(np.float64) @ first_inv).astype(np.float32)

    R0 = extrinsics[0, :3, :3]
    t0 = extrinsics[0, :3, 3]
    new_world = world_points @ R0.T + t0

    dist = np.linalg.norm(new_world, axis=-1)
    valid = point_masks.astype(np.float32)
    avg_scale = float(np.clip((dist * valid).sum() / (valid.sum() + 1e-3), 1e-6, 1e6))

    new_world = new_world / avg_scale
    new_ext[:, :3, 3] /= avg_scale
    return new_ext[:, :3], new_world.astype(np.float32), cam_points / avg_scale, depths / avg_scale, avg_scale


class HeldOutSamples:
    def __init__(self, data_root, samples_tsv, image_size=IMAGE_SIZE):
        self.data_root = data_root
        self.samples = read_samples(samples_tsv)
        self.image_size = image_size
        self._metadata = {}

    def __len__(self):
        return len(self.samples)

    def metadata(self, instance_id):
        if instance_id not in self._metadata:
            path = os.path.join(self.data_root, instance_id, "metadata.json")
            with open(path) as handle:
                meta = json.load(handle)
            self._metadata[instance_id] = {entry["id"]: entry for entry in meta["images"]}
        return self._metadata[instance_id]

    def load_frame(self, instance_id, image_id):
        entry = self.metadata(instance_id)[image_id]
        folder = os.path.join(self.data_root, instance_id)
        image = read_rgb(os.path.join(folder, entry["image"]))
        depth = np.load(os.path.join(folder, entry["depth"])).astype(np.float32)
        depth[~np.isfinite(depth)] = 0
        depth[depth < 0] = 0
        mask = cv2.imread(os.path.join(folder, entry["mask"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(os.path.join(folder, entry["mask"]))
        if image.shape[:2] != depth.shape or image.shape[:2] != mask.shape:
            raise ValueError(f"size mismatch for {instance_id}/{image_id}")
        depth[~(mask > 127)] = 0  # supervise and score the object only
        return preprocess_frame(
            image, depth, camera_from_world(entry["camera_to_world_opencv"]), intrinsic_matrix(entry["intrinsics_px"]),
            target_shape=(self.image_size, self.image_size),
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        frames = [self.load_frame(sample["instance_id"], image_id) for image_id in sample["image_ids"]]
        stack = lambda key: np.stack([f[key] for f in frames])
        extrinsics, world_points, cam_points, depths, scale = normalize_to_first_camera(
            stack("extrinsic"), stack("world_points"), stack("cam_points"), stack("depth"), stack("point_mask")
        )
        images = torch.from_numpy(stack("image")).permute(0, 3, 1, 2).float().div(255)
        return {
            "instance_id": sample["instance_id"],
            "image_ids": sample["image_ids"],
            "num_views": len(frames),
            "images": images,
            "depths": torch.from_numpy(depths),
            "extrinsics": torch.from_numpy(extrinsics),
            "intrinsics": torch.from_numpy(stack("intrinsic")),
            "world_points": torch.from_numpy(world_points),
            "point_masks": torch.from_numpy(stack("point_mask")),
            "gt_scale": scale,
        }
