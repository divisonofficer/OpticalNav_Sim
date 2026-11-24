"""Rigid transforms and the frames of the hardware layer.

* World: the R2R frame of ``frames.py`` (right-handed, z up, metres): ``(x, y, height)``. A yaw of 0 looks along +x and
  grows counter-clockwise seen from above (ROS), so MatterSim's ``heading = pi / 2 - yaw``.
* Body (every actor and link): ROS REP-103, x forward, y left, z up. A positive pitch looks down.
* A camera's frame is a body frame too (x along the optical axis). Its optical frame (z forward, x right, y down:
  OpenCV, ROS ``*_optical_frame``) and the renderer's camera matrix follow from it here, in one place.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# world (x, y, z up) -> Mitsuba / dataset world (X, Y up, Z): (x, z, -y)
WORLD_TO_MITSUBA = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
# camera body frame -> optical frame: columns are the optical x (right), y (down), z (forward) axes in the body frame
BODY_TO_OPTICAL = np.array([[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
# camera body frame -> the renderer's camera axes (right, up, back), as columns in the body frame
_BODY_TO_RENDER = np.array([[0.0, 0.0, -1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """3x3 rotation of ROS roll, pitch, yaw (radians; fixed axes x, y, z: R = Rz(yaw) Ry(pitch) Rx(roll))."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


@dataclass(frozen=True)
class Transform:
    """A pose: ``xyz`` (m) and ``rpy_deg`` (roll, pitch, yaw) of a child frame in its parent's frame."""
    xyz: tuple = (0.0, 0.0, 0.0)
    rpy_deg: tuple = (0.0, 0.0, 0.0)

    def matrix(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = rpy_matrix(*(math.radians(float(a)) for a in self.rpy_deg))
        m[:3, 3] = [float(v) for v in self.xyz]
        return m

    @classmethod
    def from_matrix(cls, m) -> "Transform":
        m = np.asarray(m, dtype=np.float64)
        sp = -float(m[2, 0])
        if abs(sp) < 1.0 - 1e-9:
            roll, pitch, yaw = math.atan2(m[2, 1], m[2, 2]), math.asin(sp), math.atan2(m[1, 0], m[0, 0])
        else:  # looking straight up or down: roll and yaw turn about the same axis, all of it goes to yaw
            roll, pitch, yaw = 0.0, math.copysign(math.pi / 2.0, sp), math.atan2(-m[0, 1], m[1, 1])
        return cls(tuple(float(v) for v in m[:3, 3]), tuple(math.degrees(a) for a in (roll, pitch, yaw)))

    def __matmul__(self, other: "Transform") -> "Transform":
        """``self @ other``: ``other`` expressed in the frame ``self`` places."""
        return Transform.from_matrix(self.matrix() @ other.matrix())

    def inverse(self) -> "Transform":
        return Transform.from_matrix(np.linalg.inv(self.matrix()))


def pose2d(x: float, y: float, z: float = 0.0, yaw_deg: float = 0.0) -> Transform:
    """A pose on the floor: position and yaw (0 along +x, counter-clockwise)."""
    return Transform((float(x), float(y), float(z)), (0.0, 0.0, float(yaw_deg)))


def yaw_from_heading(heading_rad: float) -> float:
    """ROS yaw of a MatterSim heading (0 along +y, positive to the right)."""
    return math.pi / 2.0 - float(heading_rad)


def heading_from_yaw(yaw_rad: float) -> float:
    return (math.pi / 2.0 - float(yaw_rad)) % (2.0 * math.pi)


def to_mitsuba(world_from_body: np.ndarray) -> np.ndarray:
    """A body's pose (4x4, world frame) as the renderer's object-to-world matrix. The object's own axes stay ROS
    (x forward, z up): a link mesh is modelled in its link frame, as in a URDF."""
    return WORLD_TO_MITSUBA @ np.asarray(world_from_body, dtype=np.float64)


def camera_to_world(world_from_camera: np.ndarray) -> np.ndarray:
    """A camera's pose (4x4, world frame, body axes) as the render server's ``camera_to_world`` (dataset convention:
    columns right, up, -forward, origin, in the Y-up world)."""
    return WORLD_TO_MITSUBA @ np.asarray(world_from_camera, dtype=np.float64) @ _BODY_TO_RENDER


def optical_pose(world_from_camera: np.ndarray) -> np.ndarray:
    """World pose of a camera's optical frame (z forward, x right, y down): what a calibration's extrinsics refer to."""
    return np.asarray(world_from_camera, dtype=np.float64) @ BODY_TO_OPTICAL
