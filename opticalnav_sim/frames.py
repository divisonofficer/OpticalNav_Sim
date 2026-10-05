"""Coordinate frames and camera poses.

Two frames meet here.

* Dataset frame (OpticalNav exports: index.jsonl, viewpoint/support graphs,
  episodes). Floor positions are ``[x, y]``. Mitsuba renders in a Y-up world
  where dataset ``x`` is Mitsuba X and dataset ``y`` is Mitsuba Z. A yaw of
  ``yaw`` looks along ``(sin yaw, cos yaw)`` in ``(x, y)``. Increasing yaw
  turns the camera LEFT, which is why the dataset's ``turn_right`` action
  decreases yaw. ``camera_to_world`` matrices are row-major 4x4 with columns
  ``(right, up, -forward, origin)``.

* R2R frame (connectivity JSON, ``Simulator`` state). This frame is
  right-handed and z-up like Matterport3D: ``(x, -y_dataset, height)``.
  MatterSim's heading convention then holds as published. Heading 0 looks
  along +y, positive heading turns right, and ``heading = pi - yaw``.

``camera_to_world`` reproduces the production rig transform exactly. That
includes its mount-offset convention, which puts the optical centre 0.1 m from
the support pose in a heading-dependent direction. tests/test_sim.py checks it
against stored dataset manifests.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

TWO_PI = 2.0 * math.pi


def wrap(angle: float) -> float:
    angle = math.fmod(angle, TWO_PI)
    return angle + TWO_PI if angle < 0.0 else angle


def heading_from_yaw(yaw_rad: float) -> float:
    return wrap(math.pi - yaw_rad)


def yaw_from_heading(heading_rad: float) -> float:
    return wrap(math.pi - heading_rad)


def r2r_xy(dataset_x: float, dataset_y: float) -> tuple[float, float]:
    return float(dataset_x), -float(dataset_y)


def dataset_xy(r2r_x: float, r2r_y: float) -> tuple[float, float]:
    return float(r2r_x), -float(r2r_y)


def _normalize(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _viewpoint_matrix(x: float, y: float, yaw_deg: float, eye_height_m: float) -> np.ndarray:
    """Port of robomituba_bridge.camera_pose.resolve_viewpoint_pose (Mitsuba matrix)."""
    yaw = math.radians(float(yaw_deg))
    fwd_h = _normalize((math.sin(yaw), 0.0, math.cos(yaw)))
    origin = (float(x), float(eye_height_m), float(y))
    target = (origin[0] + fwd_h[0], eye_height_m * 0.9, origin[2] + fwd_h[2])
    forward = _normalize(tuple(t - o for t, o in zip(target, origin)))
    right = _normalize(_cross(forward, (0.0, 1.0, 0.0)))
    up = _normalize(_cross(right, forward))
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, up, [-f for f in forward], origin
    return m


def rig_camera_to_world(x: float, y: float, yaw_rad: float, mount: Mapping | None,
                        fallback_height_m: float = 1.0) -> np.ndarray:
    """Port of navigation_dataset.sensor_sweep._sensor_pose_from_xy_yaw.

    ``x, y`` are the dataset floor position of the robot base, ``mount`` the
    camera's ``robot_mount`` (``xyz_m`` = [lateral, height, forward]).
    """
    mount = mount or {}
    xyz = mount.get("xyz_m") or [0.0, fallback_height_m, 0.0]
    rpy = mount.get("rpy_deg") or [0.0, 0.0, 0.0]
    mx, my, mz = (float(v) for v in xyz)
    if mx == 0.0 and my == 0.0 and mz == 0.0:
        my = fallback_height_m
    yaw = float(yaw_rad) + math.radians(float(rpy[2]))
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    world_x = float(x) + c * mx - s * mz
    world_z = float(y) + s * mx + c * mz
    return _viewpoint_matrix(world_x, world_z, math.degrees(yaw), my)


def nominal_pitch(mount: Mapping | None, fallback_height_m: float = 1.0) -> float:
    """Pitch of every dataset view: the rig looks at 0.9 x eye height, 1 m ahead."""
    eye = float(((mount or {}).get("xyz_m") or [0.0, fallback_height_m, 0.0])[1]) or fallback_height_m
    return math.atan2(-0.1 * eye, 1.0)


def pitch_camera(c2w: np.ndarray, elevation_rad: float) -> np.ndarray:
    """Rotate a camera about its own right axis; positive looks up."""
    if elevation_rad == 0.0:
        return c2w
    c, s = math.cos(elevation_rad), math.sin(elevation_rad)
    up, back = c2w[:3, 1].copy(), c2w[:3, 2].copy()
    out = c2w.copy()
    out[:3, 1] = c * up + s * back
    out[:3, 2] = c * back - s * up
    return out


def camera_at_viewpoint(x_r2r: float, y_r2r: float, heading_rad: float, elevation_rad: float,
                        mount: Mapping | None) -> np.ndarray:
    """Dataset camera for a graph viewpoint; elevation 0 reproduces dataset views."""
    x, y = dataset_xy(x_r2r, y_r2r)
    return pitch_camera(rig_camera_to_world(x, y, yaw_from_heading(heading_rad), mount), elevation_rad)


def camera_at_position(position_r2r: Sequence[float], heading_rad: float, elevation_rad: float,
                       mount: Mapping | None) -> np.ndarray:
    """Free camera: optical centre exactly at ``position`` (R2R frame, z = height)."""
    x, y = dataset_xy(position_r2r[0], position_r2r[1])
    yaw = yaw_from_heading(heading_rad)
    pitch = nominal_pitch(mount) + elevation_rad
    forward = (math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw))
    right = _normalize(_cross(forward, (0.0, 1.0, 0.0)))
    up = _normalize(_cross(right, forward))
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2] = right, up, [-f for f in forward]
    m[:3, 3] = (x, float(position_r2r[2]), y)
    return m


def legacy_flat_to_matrix(values: Sequence[float]) -> np.ndarray:
    """Manifest / index.jsonl ``camera_to_world`` (column-major flat) -> 4x4."""
    return np.asarray(values, dtype=np.float64).reshape(4, 4).T


def lookat(c2w: np.ndarray) -> tuple[list, list, list]:
    """Mitsuba look_at arguments for a dataset-convention camera matrix."""
    origin = c2w[:3, 3]
    forward = -c2w[:3, 2] / np.linalg.norm(c2w[:3, 2])
    return origin.tolist(), (origin + forward).tolist(), (c2w[:3, 1] / np.linalg.norm(c2w[:3, 1])).tolist()
