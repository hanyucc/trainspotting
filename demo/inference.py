"""Run Spotter on a list of image files. Shared by the Gradio demo and the smoke test."""

import numpy as np
import torch

from spotter.utils.load_fn import load_and_preprocess_images


def run_inference(model, image_paths, image_resolution=512):
    """Returns numpy arrays for the S input views:
    pose_enc (S, 9), extrinsics (S, 3, 4), intrinsics (S, 3, 3), depth (S, H, W, 1), depth_conf (S, H, W),
    object_mask (S, H, W), images (S, 3, H, W), world_points (S, H, W, 3) in the first camera's frame,
    valid (S, H, W): False on the padding added when the views have different aspect ratios.
    """
    device = next(model.parameters()).device
    images, valid = load_images_with_padding_mask(image_paths, image_resolution)
    with torch.inference_mode():
        predictions = model(images.to(device))
    out = {key: value[0].detach().float().cpu().numpy() for key, value in predictions.items() if torch.is_tensor(value)}
    out["valid"] = valid.numpy()
    out["world_points"] = unproject_depth(out["depth"][..., 0], out["extrinsics"], out["intrinsics"])
    return out


def load_images_with_padding_mask(image_paths, image_resolution=512):
    """Preprocess each view at its own aspect ratio, pad to a common size (as the model's loader does) and
    return the padding mask alongside: images (S, 3, H, W), valid (S, H, W) bool."""
    views = [load_and_preprocess_images([path], image_resolution=image_resolution)[0] for path in image_paths]
    height = max(v.shape[1] for v in views)
    width = max(v.shape[2] for v in views)
    images = torch.ones(len(views), 3, height, width)
    valid = torch.zeros(len(views), height, width, dtype=torch.bool)
    for i, view in enumerate(views):
        top = (height - view.shape[1]) // 2
        left = (width - view.shape[2]) // 2
        images[i, :, top : top + view.shape[1], left : left + view.shape[2]] = view
        valid[i, top : top + view.shape[1], left : left + view.shape[2]] = True
    return images, valid


def unproject_depth(depth, extrinsics, intrinsics):
    """depth (S, H, W), camera-from-world extrinsics (S, 3, 4), intrinsics (S, 3, 3) -> points (S, H, W, 3)."""
    num_frames, height, width = depth.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    fx = intrinsics[:, 0, 0][:, None, None]
    fy = intrinsics[:, 1, 1][:, None, None]
    cx = intrinsics[:, 0, 2][:, None, None]
    cy = intrinsics[:, 1, 2][:, None, None]
    camera_points = np.stack([(x[None] - cx) / fx * depth, (y[None] - cy) / fy * depth, depth], axis=-1)
    rotation = extrinsics[:, :3, :3]
    translation = extrinsics[:, :3, 3]
    return np.einsum("sij,shwj->shwi", np.transpose(rotation, (0, 2, 1)), camera_points - translation[:, None, None, :])
