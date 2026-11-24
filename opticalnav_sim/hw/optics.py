"""Lens geometry: the OpenCV camera model (pinhole + Brown-Conrady distortion), and how a distorted camera is rendered.

The render server renders a pinhole. A distorted camera is rendered as a pinhole wide enough to hold every ray the
real sensor sees, then resampled: ``render_plan`` gives the pinhole to ask for and, for every sensor pixel, where its
ray lands in that render; ``remap`` resamples. Stokes components resample like radiance (they add linearly).

Normalized image coordinates are x right, y down, at unit distance along the optical axis. Pixel centres are at
integer coordinates (OpenCV): pixel (0, 0) is the top left pixel's centre.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """A camera matrix and distortion at one image size. ``dist``: k1, k2, p1, p2, k3 (and k4, k5, k6 when given)."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple = ()

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def scaled(self, width: int, height: int) -> "Intrinsics":
        """The same camera at another image size (binned or decimated readout of the whole sensor)."""
        sx, sy = width / self.width, height / self.height
        return Intrinsics(int(width), int(height), self.fx * sx, self.fy * sy, (self.cx + 0.5) * sx - 0.5,
                          (self.cy + 0.5) * sy - 0.5, self.dist)

    def hfov_deg(self) -> float:
        """Field of view across the image through the principal point, distortion included."""
        left, right = self.rays(np.array([[-0.5, self.cy], [self.width - 0.5, self.cy]]))[:, 0]
        return math.degrees(math.atan(right) - math.atan(left))

    def rays(self, pixels: np.ndarray) -> np.ndarray:
        """Normalized undistorted coordinates (N x 2) of pixel coordinates (N x 2): the ray through each."""
        p = np.asarray(pixels, dtype=np.float64)
        return undistort(np.stack([(p[:, 0] - self.cx) / self.fx, (p[:, 1] - self.cy) / self.fy], 1), self.dist)

    def project(self, points: np.ndarray) -> np.ndarray:
        """Pixel coordinates (N x 2) of points in the optical frame (N x 3, z forward)."""
        p = np.asarray(points, dtype=np.float64)
        d = distort(p[:, :2] / p[:, 2:3], self.dist)
        return np.stack([d[:, 0] * self.fx + self.cx, d[:, 1] * self.fy + self.cy], 1)


def distort(xy: np.ndarray, dist) -> np.ndarray:
    """Brown-Conrady, as cv2.projectPoints applies it to normalized coordinates."""
    xy = np.asarray(xy, dtype=np.float64)
    if not len(dist):
        return xy.copy()
    k1, k2, p1, p2, k3, k4, k5, k6 = (list(dist) + [0.0] * 8)[:8]
    x, y = xy[:, 0], xy[:, 1]
    r2 = x * x + y * y
    radial = (1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1.0 + r2 * (k4 + r2 * (k5 + r2 * k6)))
    return np.stack([x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
                     y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y], 1)


def undistort(xy_d: np.ndarray, dist, iterations: int = 30) -> np.ndarray:
    """Inverse of ``distort`` by fixed-point iteration (converges for the distortion of real lenses over their image)."""
    xy_d = np.asarray(xy_d, dtype=np.float64)
    if not len(dist):
        return xy_d.copy()
    xy = xy_d.copy()
    for _ in range(iterations):
        xy = xy + (xy_d - distort(xy, dist))
    return xy


@dataclass(frozen=True)
class RenderPlan:
    """The pinhole render that holds a camera's rays (``width`` x ``height`` at ``hfov_deg``), and each sensor pixel's
    place in it (``map_x``, ``map_y``: float32, sensor height x width, in render pixel coordinates)."""
    width: int
    height: int
    hfov_deg: float
    map_x: np.ndarray
    map_y: np.ndarray


def render_plan(k: Intrinsics, oversample: float = 1.0, margin_px: float = 2.0) -> RenderPlan:
    """``oversample`` > 1 renders the pinhole denser than the sensor (less blur from the resampling, more pixels)."""
    u, v = np.meshgrid(np.arange(k.width, dtype=np.float64), np.arange(k.height, dtype=np.float64))
    rays = k.rays(np.stack([u.ravel(), v.ravel()], 1))
    f = max(k.fx, k.fy) * float(oversample)  # render pixels per unit of normalized coordinate at the centre
    half_w = float(np.max(np.abs(rays[:, 0]))) * f + margin_px
    half_h = float(np.max(np.abs(rays[:, 1]))) * f + margin_px
    width, height = 2 * int(math.ceil(half_w)), 2 * int(math.ceil(half_h))
    map_x = (rays[:, 0] * f + width / 2.0 - 0.5).reshape(k.height, k.width).astype(np.float32)
    map_y = (rays[:, 1] * f + height / 2.0 - 0.5).reshape(k.height, k.width).astype(np.float32)
    return RenderPlan(width, height, math.degrees(2.0 * math.atan(width / (2.0 * f))), map_x, map_y)


def remap(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Bilinear resampling of ``image`` (H x W or H x W x C) at (map_x, map_y); the edge is repeated outside."""
    img = np.asarray(image)
    if img.dtype == np.float32 and (img.ndim == 2 or img.shape[2] <= 4):
        try:  # the same resampling, several times faster (it interpolates in 1/32 pixel steps: 5e-4 of a step's contrast)
            import cv2
            return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        except ImportError:
            pass
    h, w = img.shape[:2]
    x, y = np.clip(map_x, 0.0, w - 1.0), np.clip(map_y, 0.0, h - 1.0)
    x0, y0 = np.minimum(np.floor(x).astype(np.int64), w - 2), np.minimum(np.floor(y).astype(np.int64), h - 2)
    fx, fy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
    if img.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    top = img[y0, x0] * (1.0 - fx) + img[y0, x0 + 1] * fx
    bottom = img[y0 + 1, x0] * (1.0 - fx) + img[y0 + 1, x0 + 1] * fx
    return (top * (1.0 - fy) + bottom * fy).astype(img.dtype if img.dtype.kind == "f" else np.float32)
