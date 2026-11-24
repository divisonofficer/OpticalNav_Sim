"""Cameras as hardware: an image sensor and a lens. The numbers are the makers' (see docs/hardware_platform_plan.md for
the sources and for what is still an estimate); a calibration replaces the lens model with measured intrinsics.

A camera renders at ``render_scale`` of its sensor's resolution by default (0.25: a 5 MP sensor as 0.3 MP). Its
intrinsics, and the mosaic the sensor model lays over the image, are those of the reduced image.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .actor import Spec
from .optics import Intrinsics
from .robots import Visual
from .transform import BODY_TO_OPTICAL, Transform


@dataclass(frozen=True)
class SensorSpec:
    name: str
    width: int
    height: int
    pixel_um: float
    bit_depth: int
    cfa: str = "RGGB"            # "mono", a Bayer pattern, or "polar_RGGB" (2x2 polarizers under each colour of a quad Bayer)
    shutter: str = "global"
    polarizer_deg: tuple = ()    # polar sensors: the angles of a 2x2 polarizer block, row by row
    extinction_ratio: float = 0.0
    note: str = ""

    def size_mm(self) -> tuple[float, float]:
        return self.width * self.pixel_um / 1000.0, self.height * self.pixel_um / 1000.0


@dataclass(frozen=True)
class LensSpec:
    name: str
    focal_mm: float
    f_number: float
    image_circle_mm: float
    dist: tuple = ()             # k1, k2, p1, p2, k3 in normalized coordinates
    barrel_mm: tuple = (39.0, 50.0)  # diameter, length (drawn; the optical centre is at its front)
    note: str = ""


def tv_distortion_k1(tv_percent: float, focal_mm: float, format_mm: tuple) -> float:
    """k1 of a one-term radial model that bends the long edge of ``format_mm`` (width, height) by a datasheet's TV
    distortion (the bow of that edge as a share of the image height, in percent; negative: barrel). A stand-in until
    the lens is calibrated: a datasheet gives one number, a real lens needs k1..k3."""
    x0 = format_mm[0] / 2.0 / focal_mm
    return 2.0 * tv_percent / 100.0 / (x0 * x0)


def FujinonHF6XA5M(f_number: float = 4.0) -> LensSpec:
    """Fujinon HF6XA-5M: 6 mm, F1.9 to F16, 2/3" (Sony IMX250, 11.1 mm), TV distortion -2.85 %.

    The datasheet's angle of view on 2/3" (74.7 x 58.1 degrees over 8.8 x 6.6 mm) fixes a one-term radial model: an
    effective focal length of 6.14 mm and k1 = -0.104. Its TV distortion alone gives k1 = -0.106 (tv_distortion_k1):
    the two agree. A calibration of the lens in hand replaces both (CameraSpec.calibration)."""
    return LensSpec("Fujinon HF6XA-5M", 6.14, float(f_number), 11.1, (-0.104, 0.0, 0.0, 0.0, 0.0),
                    note="effective focal length and k1 fitted to the datasheet's angle of view: calibrate")


IMX490 = SensorSpec("Sony IMX490", 2880, 1860, 3.0, 24, "RGGB", "rolling",
                    note="24-bit linear HDR from two photodiodes x two gains; shutter type and CFA phase unverified")
IMX265 = SensorSpec("Sony IMX265", 2048, 1536, 3.45, 12, "RGGB", "global", note="CFA phase unverified")
IMX250MYR = SensorSpec("Sony IMX250MYR", 2448, 2048, 3.45, 12, "polar_RGGB", "global", (90.0, 45.0, 135.0, 0.0), 300.0,
                       note="polarizer layout and extinction ratio from Sony's flyer, unverified here")


@dataclass
class CameraSpec(Spec):
    kind = "camera"
    sensor: SensorSpec = IMX265
    lens: LensSpec = field(default_factory=FujinonHF6XA5M)
    model: str = "camera"
    render_scale: float = 0.25
    body_mm: tuple = (29.0, 29.0, 45.0)      # width, height, length of the housing (drawn behind the lens)
    calibration: Intrinsics | None = None    # measured intrinsics at full resolution: replaces the lens model

    def __post_init__(self):
        w, h = self.sensor.size_mm()
        if math.hypot(w, h) > self.lens.image_circle_mm + 1e-6:
            raise ValueError(f"{self.lens.name}'s image circle ({self.lens.image_circle_mm} mm) does not cover "
                             f"{self.sensor.name} ({math.hypot(w, h):.2f} mm diagonal)")
        if not 0.0 < self.render_scale <= 1.0:
            raise ValueError("render_scale is a share of the sensor's resolution: 0 < render_scale <= 1")

    def full_intrinsics(self) -> Intrinsics:
        if self.calibration is not None:
            if (self.calibration.width, self.calibration.height) != (self.sensor.width, self.sensor.height):
                raise ValueError("a calibration is given at the sensor's full resolution")
            return self.calibration
        f = self.lens.focal_mm * 1000.0 / self.sensor.pixel_um
        return Intrinsics(self.sensor.width, self.sensor.height, f, f, (self.sensor.width - 1) / 2.0,
                          (self.sensor.height - 1) / 2.0, tuple(self.lens.dist))

    def image_size(self, scale: float | None = None) -> tuple[int, int]:
        """Rendered image size: the sensor's at ``scale``, even in both directions (a mosaic's 2x2 blocks stay whole)."""
        s = self.render_scale if scale is None else float(scale)
        return max(2, 2 * round(self.sensor.width * s / 2)), max(2, 2 * round(self.sensor.height * s / 2))

    def intrinsics(self, scale: float | None = None) -> Intrinsics:
        return self.full_intrinsics().scaled(*self.image_size(scale))

    def link_frames(self, state) -> dict:
        """``camera``: x along the optical axis, origin at the optical centre. ``optical``: OpenCV's frame there."""
        return {"camera": np.eye(4), "optical": BODY_TO_OPTICAL.copy()}

    def visuals(self) -> dict:
        """The lens barrel and the housing, behind the optical centre (the camera does not see itself)."""
        d, length = (v / 1000.0 for v in self.lens.barrel_mm)
        w, h, body = (v / 1000.0 for v in self.body_mm)
        black, grey = ("plastic", (0.02, 0.02, 0.02), 0.3), ("metal", (0.55, 0.55, 0.58), 0.35)
        return {"camera": [
            Visual("cylinder", (d / 2.0, length), Transform((-0.002 - length / 2.0, 0.0, 0.0), (0.0, 0.0, 90.0)), black),
            Visual("box", (body, w, h), Transform((-0.002 - length - body / 2.0, 0.0, 0.0)), grey)]}


def LucidTDR054S(lens: LensSpec | None = None, **kw) -> CameraSpec:
    """LUCID Triton HDR TDR054S (Sony IMX490, 5.4 MP, 24-bit HDR)."""
    return CameraSpec(sensor=IMX490, lens=lens or FujinonHF6XA5M(), model="LUCID TDR054S", **kw)


def LucidTRI032S(lens: LensSpec | None = None, **kw) -> CameraSpec:
    """LUCID Triton TRI032S (Sony IMX265, 3.2 MP, 12-bit)."""
    return CameraSpec(sensor=IMX265, lens=lens or FujinonHF6XA5M(), model="LUCID TRI032S", **kw)


def FlirBFS51S5PC(lens: LensSpec | None = None, **kw) -> CameraSpec:
    """Teledyne FLIR Blackfly S BFS-U3-51S5PC-C (Sony IMX250MYR, 5.0 MP, colour polarization)."""
    kw.setdefault("body_mm", (29.0, 29.0, 30.0))
    return CameraSpec(sensor=IMX250MYR, lens=lens or FujinonHF6XA5M(), model="FLIR BFS-U3-51S5PC-C", **kw)
