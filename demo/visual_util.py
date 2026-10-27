# Adapted from the VGGT-Omega demo (visual_util.py), Copyright (c) Meta Platforms, Inc. and affiliates,
# under the license in the repository root.

import numpy as np
import trimesh
from matplotlib import colormaps
from scipy.spatial.transform import Rotation


def predictions_to_glb(
    predictions: dict,
    mask_threshold: float = 0.5,
    show_cameras: bool = True,
    max_points: int = 1_000_000,
    view: int | None = None,
    filter_depth_edges: bool = True,
    depth_edge_rtol: float = 0.03,
) -> trimesh.Scene:
    """Point cloud and camera frusta of a Spotter prediction as a GLB scene.

    mask_threshold keeps the pixels whose object-mask probability is at least the threshold
    (0 keeps everything). view restricts the points to one input photo (0-based); the cameras of
    all photos are still drawn. Pixels on depth discontinuities are always dropped.
    """
    points = predictions["world_points"]
    conf = predictions["depth_conf"].copy()
    if filter_depth_edges:
        conf[depth_edge(predictions["depth"][..., 0], rtol=depth_edge_rtol)] = 0.0
    images = np.transpose(predictions["images"], (0, 2, 3, 1))
    extrinsics = predictions["extrinsics"]

    vertices = points.reshape(-1, 3)
    colors = (images.reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)
    conf = conf.reshape(-1)

    keep = np.isfinite(vertices).all(axis=1) & np.isfinite(conf) & (conf > 1e-5)
    if "valid" in predictions:
        keep &= predictions["valid"].reshape(-1)
    if mask_threshold > 0 and "object_mask" in predictions:
        keep &= predictions["object_mask"].reshape(-1) >= float(mask_threshold)
    if view is not None:
        num_views, height, width = points.shape[:3]
        keep &= np.repeat(np.arange(num_views) == int(view), height * width)

    vertices, colors = _limit_points(vertices[keep], colors[keep], max_points)
    if vertices.size == 0:
        vertices = np.zeros((1, 3), dtype=np.float32)
        colors = np.full((1, 3), 255, dtype=np.uint8)
        scene_scale = 1.0
    else:
        lower, upper = np.percentile(vertices, [5, 95], axis=0)
        scene_scale = float(np.linalg.norm(upper - lower)) or 1.0

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=vertices, colors=colors))

    world_to_camera = np.tile(np.eye(4), (len(extrinsics), 1, 1))
    world_to_camera[:, :3, :4] = extrinsics
    if show_cameras:
        colormap = colormaps.get_cmap("gist_rainbow")
        for i, w2c in enumerate(world_to_camera):
            rgba = colormap(i / max(len(world_to_camera), 1))
            _add_camera(scene, np.linalg.inv(w2c), tuple(int(255 * x) for x in rgba[:3]), scene_scale)
    scene.apply_transform(np.linalg.inv(world_to_camera[0]) @ _opengl_conversion())
    return scene


def _limit_points(vertices, colors, max_points):
    if max_points <= 0 or len(vertices) <= max_points:
        return vertices, colors
    indices = np.linspace(0, len(vertices) - 1, max_points).astype(np.int64)
    return vertices[indices], colors[indices]


def depth_edge(depth, rtol=0.03, kernel_size=3):
    depth = np.asarray(depth)
    shape = depth.shape
    depth = depth.reshape(-1, *shape[-2:])
    pad = kernel_size // 2
    padded = np.pad(depth, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    depth_max = np.full_like(depth, -np.inf)
    depth_min = np.full_like(depth, np.inf)
    for y in range(kernel_size):
        for x in range(kernel_size):
            window = padded[:, y : y + depth.shape[-2], x : x + depth.shape[-1]]
            depth_max = np.maximum(depth_max, window)
            depth_min = np.minimum(depth_min, window)
    return ((depth_max - depth_min) / np.maximum(np.abs(depth), 1e-6) > rtol).reshape(shape)


def _opengl_conversion():
    matrix = np.identity(4)
    matrix[1, 1] = -1
    matrix[2, 2] = -1
    return matrix


def _transform_points(transformation, points):
    return points @ transformation[:3, :3].T + transformation[:3, 3]


def _add_camera(scene, camera_to_world, face_color, scene_scale):
    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1
    rot_45 = np.eye(4)
    rot_45[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45[2, 3] = -cam_height
    transform = camera_to_world @ _opengl_conversion() @ rot_45
    cone = trimesh.creation.cone(cam_width, cam_height, sections=4)
    slight = np.eye(4)
    slight[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()
    vertices = np.concatenate([cone.vertices, 0.95 * cone.vertices, _transform_points(slight, cone.vertices)])
    mesh = trimesh.Trimesh(vertices=_transform_points(transform, vertices), faces=_camera_faces(cone))
    mesh.visual.face_colors[:, :3] = face_color
    scene.add_geometry(mesh)


def _camera_faces(cone):
    faces = []
    n = len(cone.vertices)
    for face in cone.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        o1, o2, o3 = face + n
        p1, p2, p3 = face + 2 * n
        faces.extend([(v1, v2, o2), (v1, o1, v3), (o3, v2, v3), (v1, v2, p2), (v1, p1, v3), (p3, v2, v3)])
    faces += [(c, b, a) for a, b, c in faces]
    return np.array(faces)
