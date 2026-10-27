"""Point-cloud alignment and distances for the point-map metrics.

The similarity alignment is Umeyama's closed form on the pixel correspondences, followed by a
point-to-point ICP refinement that reproduces Open3D 0.18's ``registration_icp`` with
``TransformationEstimationPointToPoint`` and its default convergence criteria, in numpy / scipy
(so the evaluation has no Open3D dependency). Accuracy is the mean nearest-neighbour distance
from the aligned prediction to the ground truth, completeness the reverse.
"""

import os

import numpy as np
from scipy.spatial import cKDTree

try:
    WORKERS = len(os.sched_getaffinity(0))
except AttributeError:
    WORKERS = -1

ICP_MAX_ITERATION = 30
ICP_RELATIVE_FITNESS = 1e-6
ICP_RELATIVE_RMSE = 1e-6


def umeyama(source, target):
    """Similarity c, R, t with c * R @ x_source + t ~ x_target. source, target: (3, N)."""
    mu_s = source.mean(axis=1, keepdims=True)
    mu_t = target.mean(axis=1, keepdims=True)
    var_s = np.square(source - mu_s).sum(axis=0).mean()
    cov = ((target - mu_t) @ (source - mu_s).T) / source.shape[1]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_s
    R = U @ S @ Vt
    t = mu_t - c * R @ mu_s
    return c, R, t


def _as_points(points):
    return np.ascontiguousarray(np.asarray(points, dtype=np.float64).reshape(-1, 3))


def _transform(points, T):
    return points @ T[:3, :3].T + T[:3, 3]


def _correspondences(target_tree, source_points, max_distance):
    dist, idx = target_tree.query(source_points, k=1, distance_upper_bound=max_distance, workers=WORKERS)
    found = np.isfinite(dist)
    n = int(found.sum())
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), 0.0, 0.0
    fitness = n / float(len(source_points))
    rmse = float(np.sqrt(np.sum(dist[found] ** 2) / n))
    return np.nonzero(found)[0], idx[found], fitness, rmse


def _rigid_fit(src, dst):
    """Rotation + translation (no scale) minimising |R src + t - dst|."""
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sigma = (dst - mu_d).T @ (src - mu_s) / float(len(src))
    U, _, Vt = np.linalg.svd(sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = mu_d - R @ mu_s
    return T


def icp_point_to_point(source, target, max_correspondence_distance, max_iteration=ICP_MAX_ITERATION):
    """4x4 rigid transform refining source onto target (Open3D point-to-point ICP semantics)."""
    source = _as_points(source)
    target = _as_points(target)
    transformation = np.eye(4)
    tree = cKDTree(target)
    pcd = source
    src_idx, tgt_idx, fitness, rmse = _correspondences(tree, pcd, max_correspondence_distance)
    for _ in range(max_iteration):
        update = np.eye(4) if len(src_idx) == 0 else _rigid_fit(pcd[src_idx], target[tgt_idx])
        transformation = update @ transformation
        pcd = _transform(pcd, update)
        prev_fitness, prev_rmse = fitness, rmse
        src_idx, tgt_idx, fitness, rmse = _correspondences(tree, pcd, max_correspondence_distance)
        if abs(prev_fitness - fitness) < ICP_RELATIVE_FITNESS and abs(prev_rmse - rmse) < ICP_RELATIVE_RMSE:
            break
    return transformation


def transform_points(points, transformation):
    return _transform(_as_points(points), np.asarray(transformation, dtype=np.float64))


def nearest_distances(query_points, reference_points):
    distances, _ = cKDTree(reference_points).query(query_points, workers=WORKERS)
    return distances
