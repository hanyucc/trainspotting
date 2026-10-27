# Adapted from the VGGT training code (vggt/training/data/dataset_util.py and
# base_dataset.py), Copyright (c) Meta Platforms, Inc. and affiliates, released under
# the VGGT license (see LICENSE_VGGT in this folder). The functions are reproduced so that
# the evaluation applies exactly the preprocessing the Trainspotting protocol was frozen
# with: a crop around the principal point, a resize to a 4-pixel safety margin above the
# target size, and a final crop to the target size.

import math

import cv2
import numpy as np
from PIL import Image

LANCZOS = Image.Resampling.LANCZOS
BICUBIC = Image.Resampling.BICUBIC


def crop_by_principal_point(image, depth_map, intrinsic, target_shape, strict=False):
    """Crop image and depth around the principal point and shift the intrinsics.

    target_shape is (height, width). With strict=True the output is zero-padded to exactly
    target_shape when the crop comes out smaller.
    """
    original_size = np.array(image.shape)
    intrinsic = np.copy(intrinsic)
    if original_size[0] < target_shape[0] or original_size[1] < target_shape[1]:
        raise AssertionError(f"image {original_size[:2]} smaller than target {target_shape}")

    cx = intrinsic[1, 2]  # row coordinate of the principal point
    cy = intrinsic[0, 2]  # column coordinate
    if strict:
        half_x = min(target_shape[0] / 2, cx)
        half_y = min(target_shape[1] / 2, cy)
    else:
        half_x = min(target_shape[0] / 2, cx, original_size[0] - cx)
        half_y = min(target_shape[1] / 2, cy, original_size[1] - cy)

    start_x = math.floor(cx) - math.floor(half_x)
    start_y = math.floor(cy) - math.floor(half_y)
    assert start_x >= 0 and start_y >= 0
    if strict:
        end_x = start_x + target_shape[0]
        end_y = start_y + target_shape[1]
    else:
        end_x = start_x + 2 * math.floor(half_x)
        end_y = start_y + 2 * math.floor(half_y)

    image = image[start_x:end_x, start_y:end_y, :]
    depth_map = depth_map[start_x:end_x, start_y:end_y]
    intrinsic[1, 2] -= start_x
    intrinsic[0, 2] -= start_y

    if strict and (np.array(image.shape[:2]) != np.asarray(target_shape)).any():
        pad_h = int(target_shape[0] - image.shape[0])
        pad_w = int(target_shape[1] - image.shape[1])
        if pad_h < 0 or pad_w < 0:
            raise ValueError("cropped image larger than the target shape")
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
        depth_map = np.pad(depth_map, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return image, depth_map, intrinsic


def resize_with_margin(image, depth_map, intrinsic, target_shape, safe_bound=4):
    """Resize so that both sides are at least target + safe_bound, updating the intrinsics.

    LANCZOS when downscaling, BICUBIC when upscaling; nearest neighbour for the depth map.
    """
    original_size = np.array(image.shape[:2])
    target_shape = np.asarray(target_shape)
    max_resize_scale = np.max((target_shape + safe_bound) / original_size)
    intrinsic = np.copy(intrinsic)

    pil_image = Image.fromarray(image)
    output_resolution = np.floor(np.array(pil_image.size) * max_resize_scale).astype(int)
    pil_image = pil_image.resize(tuple(output_resolution), resample=LANCZOS if max_resize_scale < 1 else BICUBIC)
    image = np.array(pil_image)
    depth_map = cv2.resize(depth_map, output_resolution, fx=max_resize_scale, fy=max_resize_scale, interpolation=cv2.INTER_NEAREST)

    actual_resize_scale = np.max(np.array(image.shape[:2]) / original_size)
    # Pixel centres: shift by half a pixel before scaling the principal point.
    intrinsic[0, 2] += 0.5
    intrinsic[1, 2] += 0.5
    intrinsic[:2, :] *= actual_resize_scale
    intrinsic[0, 2] -= 0.5
    intrinsic[1, 2] -= 0.5
    assert image.shape[:2] == depth_map.shape[:2]
    return image, depth_map, intrinsic


def depth_to_camera_points(depth_map, intrinsic):
    """(H, W) depth + 3x3 intrinsics -> (H, W, 3) points in the camera frame."""
    H, W = depth_map.shape
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    return np.stack(((u - cu) * depth_map / fu, (v - cv) * depth_map / fv, depth_map), axis=-1).astype(np.float32)


def depth_to_world_points(depth_map, extrinsic, intrinsic, eps=1e-8):
    """Depth + camera-from-world (3, 4) + intrinsics -> (world points, camera points, valid mask)."""
    point_mask = depth_map > eps
    cam_points = depth_to_camera_points(depth_map, intrinsic)
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    R_c2w = R.T
    t_c2w = -R.T @ t
    world_points = cam_points @ R_c2w.T + t_c2w
    return world_points.astype(np.float32), cam_points, point_mask


def preprocess_frame(image, depth_map, extrinsic, intrinsic, target_shape=(512, 512)):
    """The evaluation preprocessing of one frame: crop around the principal point, resize with a
    4-pixel margin, crop to target_shape; then unproject the depth.

    image (H, W, 3) uint8 RGB; depth_map (H, W) float32 with 0 where unknown; extrinsic (3, 4)
    camera-from-world; intrinsic (3, 3). Returns a dict of numpy arrays.
    """
    image = np.copy(image)
    depth_map = np.copy(depth_map)
    target_shape = np.asarray(target_shape)
    original_size = np.array(image.shape[:2])
    image, depth_map, intrinsic = crop_by_principal_point(image, depth_map, intrinsic, original_size, strict=False)
    image, depth_map, intrinsic = resize_with_margin(image, depth_map, intrinsic, target_shape)
    image, depth_map, intrinsic = crop_by_principal_point(image, depth_map, intrinsic, target_shape, strict=True)
    world_points, cam_points, point_mask = depth_to_world_points(depth_map, extrinsic, intrinsic)
    return {
        "image": image,
        "depth": depth_map.astype(np.float32),
        "extrinsic": np.asarray(extrinsic, dtype=np.float32),
        "intrinsic": intrinsic.astype(np.float32),
        "world_points": world_points,
        "cam_points": cam_points,
        "point_mask": point_mask,
    }
