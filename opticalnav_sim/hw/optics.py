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
    """A camera matrix and distortion at one image size. ``dist``: OpenCV's coefficients (see ``distort``)."""
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
    """OpenCV's distortion of normalized coordinates, as cv2.projectPoints applies it: radial (k1, k2, k3 over
    k4, k5, k6: the rational model), tangential (p1, p2) and thin prism (s1..s4), from 4, 5, 8 or 12 coefficients in
    OpenCV's order (k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4). The tilted sensor terms are not modelled."""
    xy = np.asarray(xy, dtype=np.float64)
    if not len(dist):
        return xy.copy()
    if len(dist) > 12 and any(float(v) != 0.0 for v in dist[12:]):
        raise ValueError("the tilted sensor model (distortion coefficients 13 and 14) is not supported")
    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 = (list(dist) + [0.0] * 12)[:12]
    x, y = xy[:, 0], xy[:, 1]
    r2 = x * x + y * y
    radial = (1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1.0 + r2 * (k4 + r2 * (k5 + r2 * k6)))
    return np.stack([x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) + r2 * (s1 + s2 * r2),
                     y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y + r2 * (s3 + s4 * r2)], 1)


def undistort(xy_d: np.ndarray, dist, iterations: int = 40) -> np.ndarray:
    """Inverse of ``distort``: Newton's method from the distorted point, a step halved while it does not come closer.
    (OpenCV's fixed-point iteration diverges at the image edge of some rational-model calibrations.) A point the
    model does not reach (it folds back outside the calibrated field) comes back as the nearest the iteration found."""
    xy_d = np.asarray(xy_d, dtype=np.float64)
    if not len(dist):
        return xy_d.copy()
    xy = xy_d.copy()
    err = distort(xy, dist) - xy_d
    h = 1e-6
    for _ in range(iterations):
        size = np.hypot(err[:, 0], err[:, 1])
        todo = size > 1e-13
        if not todo.any():
            break
        p, e = xy[todo], err[todo]
        jx = (distort(p + (h, 0.0), dist) - distort(p - (h, 0.0), dist)) / (2.0 * h)   # d(distorted) / dx
        jy = (distort(p + (0.0, h), dist) - distort(p - (0.0, h), dist)) / (2.0 * h)
        det = jx[:, 0] * jy[:, 1] - jy[:, 0] * jx[:, 1]
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        step = np.stack([(jy[:, 1] * e[:, 0] - jy[:, 0] * e[:, 1]) / det, (jx[:, 0] * e[:, 1] - jx[:, 1] * e[:, 0]) / det], 1)
        best, best_err, best_size, scale = p, e, size[todo], np.ones(len(p))
        for _ in range(8):
            trial = p - step * scale[:, None]
            trial_err = distort(trial, dist) - xy_d[todo]
            trial_size = np.hypot(trial_err[:, 0], trial_err[:, 1])
            better = trial_size < best_size
            best, best_err, best_size = (np.where(better[:, None], trial, best), np.where(better[:, None], trial_err, best_err),
                                         np.where(better, trial_size, best_size))
            if better.all():
                break
            scale = np.where(better, scale, scale * 0.5)
        xy[todo], err[todo] = best, best_err
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
