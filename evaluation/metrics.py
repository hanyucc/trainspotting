"""Per-sample metrics of the Trainspotting protocol.

All inputs are for one sample: S views at the evaluation resolution, ground truth in the first
camera's frame with unit mean point distance (see dataset.py), predictions in the model's own
first-camera frame.

Depth: one median scale per sample (ground truth over prediction, over the valid pixels of all
views), then AbsRel and the inlier ratios delta < 1.03 / 1.05 / 1.10 pooled over the sample's
object pixels.

Camera pose: pairwise relative poses over all view pairs; RRA / RTA at 5, 15 and 30 degrees and
the AUC of max(rotation, translation-direction error) up to each threshold, with integer-degree
bins, following the VGGT / Pi3 evaluation. Plus the mean absolute rotation error of each view in
the first-camera frame.

Point map: the predicted depth unprojected with the predicted cameras, at the pixels that have
ground truth, aligned to the ground-truth points by a similarity transform (Umeyama) refined by
point-to-point ICP; accuracy and completeness in percent of the object's extent (the diagonal of
the 5-95 percentile bounding box of the ground-truth points). At most 20,000 pixels per sample
are used, a fixed-seed random subset.
"""

import numpy as np
import torch

from alignment import icp_point_to_point, nearest_distances, transform_points, umeyama

POSE_THRESHOLDS_DEG = (5, 15, 30)
DEPTH_DELTA_THRESHOLDS = ((1.03, "1_03"), (1.05, "1_05"), (1.10, "1_10"))
MIN_VALID_PIXELS = 100
MIN_VALID_FRAME_PIXELS = 100
MAX_POINTS = 20000
ICP_EXTENT_FRACTION = 0.10
POINT_SEED = 0


def valid_depth_mask(pred_depth, gt_depth, point_masks):
    return point_masks.bool() & torch.isfinite(gt_depth) & torch.isfinite(pred_depth) & (gt_depth > 0)


def depth_scale(pred_depth, gt_depth, point_masks):
    """Median ground-truth / predicted depth ratio over the sample's valid pixels, or None."""
    mask = valid_depth_mask(pred_depth, gt_depth, point_masks) & (pred_depth > 0)
    if int(mask.sum()) < MIN_VALID_PIXELS:
        return None
    ratio = gt_depth[mask] / pred_depth[mask]
    ratio = ratio[torch.isfinite(ratio)]
    return float(ratio.median()) if ratio.numel() else None


def _delta_ratio(pred, gt):
    pred = pred.clamp_min(1e-8)
    return torch.maximum(pred / gt, gt / pred)


def depth_metrics(pred_depth, gt_depth, point_masks, scale):
    mask = valid_depth_mask(pred_depth, gt_depth, point_masks)
    if mask.sum() == 0 or scale is None:
        return {}
    gt_valid = gt_depth[mask].clamp_min(1e-8)
    aligned = pred_depth[mask] * scale
    out = {"depth_scale": scale, "depth_abs_rel": float(((aligned - gt_depth[mask]).abs() / gt_valid).mean())}
    ratio = _delta_ratio(aligned, gt_valid)
    for threshold, tag in DEPTH_DELTA_THRESHOLDS:
        out[f"depth_delta{tag}"] = float((ratio < threshold).float().mean())
    return out


def _to_se3(ext):
    out = torch.eye(4, dtype=ext.dtype).repeat(ext.shape[0], 1, 1)
    out[:, :3, :4] = ext[..., :3, :4]
    return out


def _se3_inverse(ext):
    r = ext[..., :3, :3]
    t = ext[..., :3, 3:]
    inv = torch.eye(4, dtype=ext.dtype).repeat(ext.shape[0], 1, 1)
    inv[:, :3, :3] = r.transpose(-1, -2)
    inv[:, :3, 3:] = -r.transpose(-1, -2) @ t
    return inv


def relative_pose_errors(pred_extrinsics, gt_extrinsics):
    """Pairwise rotation and translation-direction errors in degrees over all pairs i < j."""
    n = pred_extrinsics.shape[0]
    i, j = torch.combinations(torch.arange(n), 2).unbind(-1)
    pred, gt = _to_se3(pred_extrinsics.float()), _to_se3(gt_extrinsics.float())
    rel_pred = pred[j] @ _se3_inverse(pred[i, :3, :4])
    rel_gt = gt[j] @ _se3_inverse(gt[i, :3, :4])

    r = rel_pred[:, :3, :3] @ rel_gt[:, :3, :3].transpose(-1, -2)
    cos = ((r.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    rot_deg = torch.rad2deg(torch.acos(cos))

    eps = 1e-15
    t_pred = rel_pred[:, :3, 3]
    t_gt = rel_gt[:, :3, 3]
    t_pred = t_pred / (t_pred.norm(dim=-1, keepdim=True) + eps)
    t_gt = t_gt / (t_gt.norm(dim=-1, keepdim=True) + eps)
    loss_t = (1.0 - (t_pred * t_gt).sum(-1) ** 2).clamp_min(eps)
    trans_deg = torch.rad2deg(torch.acos(torch.sqrt(1.0 - loss_t)))
    trans_deg = torch.minimum(trans_deg, (180.0 - trans_deg).abs())
    return rot_deg, trans_deg


def pose_auc(rot_deg, trans_deg, max_threshold):
    max_err = torch.maximum(rot_deg, trans_deg)
    hist = torch.histc(max_err, bins=max_threshold, min=0, max=max_threshold)
    return float(torch.cumsum(hist / max_err.numel(), dim=0).mean() * 100.0)


def pose_metrics(pred_extrinsics, pred_intrinsics, gt_extrinsics, gt_intrinsics, point_masks):
    pred_extrinsics = pred_extrinsics.float().cpu()
    gt_extrinsics = gt_extrinsics.float().cpu()
    valid_frames = point_masks.sum(dim=(-1, -2)) > MIN_VALID_FRAME_PIXELS
    if valid_frames.sum() == 0:
        return {}
    out = {}
    pred_r = pred_extrinsics[..., :3, :3][valid_frames]
    gt_r = gt_extrinsics[..., :3, :3][valid_frames]
    rel_r = pred_r @ gt_r.transpose(-1, -2)
    cos_angle = ((rel_r.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    out["pose_rot_deg"] = float(torch.rad2deg(torch.acos(cos_angle)).mean())

    pred_valid = pred_extrinsics[..., :3, :4][valid_frames]
    gt_valid = gt_extrinsics[..., :3, :4][valid_frames]
    if pred_valid.shape[0] >= 2:
        rel_rot, rel_trans = relative_pose_errors(pred_valid, gt_valid)
        out["pose_rra_mean_deg"] = float(rel_rot.mean())
        out["pose_rta_mean_deg"] = float(rel_trans.mean())
        for tau in POSE_THRESHOLDS_DEG:
            out[f"pose_rra_at{tau}"] = float((rel_rot < tau).float().mean() * 100)
            out[f"pose_rta_at{tau}"] = float((rel_trans < tau).float().mean() * 100)
            out[f"pose_auc_at{tau}"] = pose_auc(rel_rot, rel_trans, tau)

    if pred_intrinsics is not None:
        pred_intrinsics = pred_intrinsics.float().cpu()
        gt_intrinsics = gt_intrinsics.float().cpu()
        pred_fx = pred_intrinsics[..., 0, 0][valid_frames]
        pred_fy = pred_intrinsics[..., 1, 1][valid_frames]
        gt_fx = gt_intrinsics[..., 0, 0][valid_frames].clamp_min(1e-8)
        gt_fy = gt_intrinsics[..., 1, 1][valid_frames].clamp_min(1e-8)
        out["focal_rel_abs"] = float((0.5 * ((pred_fx - gt_fx).abs() / gt_fx + (pred_fy - gt_fy).abs() / gt_fy)).mean())
    return out


def unproject(depth, extrinsics, intrinsics):
    """depth (S, H, W), camera-from-world extrinsics (S, 3, 4), intrinsics (S, 3, 3) -> world points (S, H, W, 3)."""
    S, H, W = depth.shape
    v, u = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    pix = torch.stack([u, v, torch.ones_like(u)], dim=-1)
    rays = torch.einsum("sij,hwj->shwi", torch.linalg.inv(intrinsics.float()), pix)
    cam_pts = rays * depth.float()[..., None]
    R = extrinsics[..., :3, :3].float()
    t = extrinsics[..., :3, 3].float()
    return torch.einsum("sji,shwj->shwi", R, cam_pts - t[:, None, None, :])


def object_extent(gt_points):
    lo, hi = np.percentile(gt_points, [5, 95], axis=0)
    return float(np.linalg.norm(hi - lo))


def point_map_metrics(pred_points, gt_points, seed=POINT_SEED, max_points=MAX_POINTS):
    """pred_points, gt_points: (N, 3) at the same pixels. {} if the sample cannot be scored."""
    pred = np.asarray(pred_points, dtype=np.float64).reshape(-1, 3)
    gt = np.asarray(gt_points, dtype=np.float64).reshape(-1, 3)
    ok = np.isfinite(pred).all(axis=1) & np.isfinite(gt).all(axis=1)
    pred, gt = pred[ok], gt[ok]
    if len(gt) < MIN_VALID_PIXELS:
        return {}
    if len(gt) > max_points:
        keep = np.random.default_rng(seed).choice(len(gt), max_points, replace=False)
        pred, gt = pred[keep], gt[keep]
    extent = object_extent(gt)
    if not np.isfinite(extent) or extent <= 0:
        return {}
    c, R, t = umeyama(pred.T, gt.T)
    pred = (c * (R @ pred.T) + t).T
    refinement = icp_point_to_point(pred, gt, ICP_EXTENT_FRACTION * extent)
    rec = transform_points(pred, refinement)
    acc = nearest_distances(rec, gt)
    comp = nearest_distances(gt, rec)
    pct = 100.0 / extent
    return {
        "points_acc": float(acc.mean() * pct), "points_acc_med": float(np.median(acc) * pct),
        "points_comp": float(comp.mean() * pct), "points_comp_med": float(np.median(comp) * pct),
        "points_extent": extent,
    }


def point_metrics(pred_depth, pred_extrinsics, pred_intrinsics, gt_depth, gt_world_points, point_masks):
    mask = valid_depth_mask(pred_depth, gt_depth, point_masks) & (pred_depth > 0)
    if int(mask.sum()) == 0:
        return {}
    pred_points = unproject(pred_depth.cpu(), pred_extrinsics.cpu(), pred_intrinsics.cpu())
    return point_map_metrics(pred_points[mask.cpu()].numpy(), gt_world_points.float().cpu()[mask.cpu()].numpy())


def sample_metrics(prediction, sample):
    """All metrics of one sample. prediction: extrinsics (S, 3, 4), intrinsics (S, 3, 3), depth (S, H, W),
    float tensors in the model's first-camera frame (any device). sample: an item of HeldOutSamples."""
    pred_depth = prediction["depth"].float().cpu()
    pred_ext = prediction["extrinsics"].float().cpu()
    pred_intr = prediction["intrinsics"].float().cpu() if prediction.get("intrinsics") is not None else None
    gt_depth = sample["depths"].float()
    point_masks = sample["point_masks"].bool()

    out = {}
    scale = depth_scale(pred_depth, gt_depth, point_masks)
    out.update(depth_metrics(pred_depth, gt_depth, point_masks, scale))
    out.update(pose_metrics(pred_ext, pred_intr, sample["extrinsics"], sample["intrinsics"], point_masks))
    if pred_intr is not None:
        out.update(point_metrics(pred_depth, pred_ext, pred_intr, gt_depth, sample["world_points"], point_masks))
    return out
