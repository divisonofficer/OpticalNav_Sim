"""Measured camera calibrations: OpenCV's YAML (cv2.FileStorage) of a multi-camera rig.

A rig file holds, for each camera ``X``: ``k_X`` (3x3 camera matrix), ``d_X`` (distortion, OpenCV's order) and
``t_X_transform_in_<ref>`` (4x4: the pose of X's optical frame in the reference camera's, so a point p in X is
``T p`` in the reference: in the rig of docs/hardware_platform_plan.md, ``12_left`` is 10 cm to the left of ``12_rear``
and the transform's x is -0.0998). The image size is not in the file: it is the size the camera recorded at.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .optics import Intrinsics

_MATRIX = re.compile(r"^(\w+):\s*!!opencv-matrix\s*\n\s*rows:\s*(\d+)\s*\n\s*cols:\s*(\d+)\s*\n\s*dt:\s*(\w+)\s*\n\s*data:\s*\[(.*?)\]",
                     re.S | re.M)


def load_opencv_yaml(path) -> dict:
    """Every ``!!opencv-matrix`` of an OpenCV YAML file, by name (no OpenCV needed)."""
    out = {}
    for name, rows, cols, _, data in _MATRIX.findall(Path(path).read_text()):
        values = np.array([float(v) for v in data.replace("\n", " ").split(",") if v.strip()], dtype=np.float64)
        if values.size != int(rows) * int(cols):
            raise ValueError(f"{path}: {name} has {values.size} values for {rows} x {cols}")
        out[name] = values.reshape(int(rows), int(cols))
    if not out:
        raise ValueError(f"{path}: no !!opencv-matrix entries")
    return out


@dataclass(frozen=True)
class CalibratedCamera:
    name: str
    k: np.ndarray            # 3x3 at the calibration's image size
    dist: tuple
    pose: np.ndarray         # 4x4: this camera's optical frame in the reference camera's optical frame

    def intrinsics(self, width: int, height: int) -> Intrinsics:
        """The calibration as intrinsics of a ``width`` x ``height`` image: the size it was measured at."""
        return Intrinsics(int(width), int(height), float(self.k[0, 0]), float(self.k[1, 1]), float(self.k[0, 2]),
                          float(self.k[1, 2]), tuple(float(v) for v in self.dist))


@dataclass(frozen=True)
class RigCalibration:
    reference: str
    cameras: dict            # name -> CalibratedCamera

    @classmethod
    def load(cls, path) -> "RigCalibration":
        m = load_opencv_yaml(path)
        refs = {k.rsplit("_transform_in_", 1)[1] for k in m if "_transform_in_" in k}
        if len(refs) != 1:
            raise ValueError(f"{path}: expected one reference camera in the t_X_transform_in_<ref> entries, found {sorted(refs)}")
        ref = refs.pop()
        cams = {}
        for key in m:
            if key.startswith("k_") and f"t_{key[2:]}_transform_in_{ref}" in m and f"d_{key[2:]}" in m:
                name = key[2:]
                cams[name] = CalibratedCamera(name, m[key], tuple(m[f"d_{name}"].ravel()), m[f"t_{name}_transform_in_{ref}"])
        if ref not in cams:
            raise ValueError(f"{path}: the reference camera {ref!r} has no k_ / d_ entries")
        return cls(ref, cams)

    def relative(self, name: str, to: str | None = None) -> np.ndarray:
        """Pose of camera ``name``'s optical frame in camera ``to``'s (default: the reference's)."""
        if name not in self.cameras:
            raise KeyError(f"no camera {name!r} in the calibration (have {sorted(self.cameras)})")
        pose = self.cameras[name].pose
        return pose if to in (None, self.reference) else np.linalg.inv(self.cameras[to].pose) @ pose
