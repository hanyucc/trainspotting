"""Read access to an undistorted COLMAP model."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap


@dataclass
class View:
    name: str
    image_id: int
    width: int
    height: int
    K: np.ndarray
    world_to_camera: np.ndarray
    point_ids: np.ndarray


class Scene:
    """Registered views, their pinhole cameras and the 3D points of a COLMAP model."""

    def __init__(self, model_dir: Path):
        self.reconstruction = pycolmap.Reconstruction(str(model_dir))
        rec = self.reconstruction
        self.views: list[View] = []
        for image_id in sorted(rec.reg_image_ids()):
            image = rec.images[image_id]
            camera = rec.cameras[image.camera_id]
            w2c = np.eye(4)
            w2c[:3, :4] = image.cam_from_world().matrix()
            point_ids = np.array([p.point3D_id for p in image.points2D if p.has_point3D()], np.int64)
            self.views.append(View(image.name, image_id, camera.width, camera.height,
                                   camera.calibration_matrix(), w2c, point_ids))
        self.index = {v.name: i for i, v in enumerate(self.views)}
        self.points = {pid: p.xyz for pid, p in rec.points3D.items()}

    def tracks(self):
        """Per 3D point, the indices of the views observing it (each view once)."""
        view_of = {v.image_id: i for i, v in enumerate(self.views)}
        for point in self.reconstruction.points3D.values():
            ids = {view_of[e.image_id] for e in point.track.elements if e.image_id in view_of}
            yield np.fromiter(sorted(ids), np.int64)

    def covisibility(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Shared-point counts between views, and per view the points on tracks of length >= 3 and in total."""
        n = len(self.views)
        shared = np.zeros((n, n), np.int64)
        confirmed = np.zeros(n, np.int64)
        total = np.zeros(n, np.int64)
        for ids in self.tracks():
            total[ids] += 1
            if len(ids) >= 3:
                confirmed[ids] += 1
            shared[np.ix_(ids, ids)] += 1
        np.fill_diagonal(shared, 0)
        return shared, confirmed, total

    def point_depths(self, view: View) -> np.ndarray:
        xyz = [self.points[p] for p in view.point_ids if p in self.points]
        if not xyz:
            return np.zeros(0)
        xyz = np.stack(xyz)
        return (xyz @ view.world_to_camera[:3, :3].T + view.world_to_camera[:3, 3])[:, 2]

    def depth_range(self, view: View, min_depth: float) -> tuple[float, float] | None:
        """1st to 99th percentile of the view's sparse depths, or None when degenerate."""
        z = self.point_depths(view)
        z = z[np.isfinite(z) & (z > min_depth)]
        if len(z) == 0:
            return None
        lo, hi = np.percentile(z, [1, 99])
        return (float(lo), float(hi)) if hi > lo > min_depth else None
