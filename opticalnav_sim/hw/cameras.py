"""Cameras as hardware: an image sensor and a lens. The numbers are the makers' (see docs/hardware_platform_plan.md for
the sources and for what is still an estimate); a calibration replaces the lens model with measured intrinsics.

A camera renders at ``render_scale`` of its sensor's resolution by default (0.25: a 5 MP sensor as 0.3 MP). Its
intrinsics, and the mosaic the sensor model lays over the image, are those of the reduced image.

Another camera is a ``SensorSpec`` and a ``LensSpec`` (or a calibration) in a ``CameraSpec``: nothing else knows the models.
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
    """An image sensor: its geometry, its mosaic, and its electrical model in EMVA 1288 terms (hw.sensor turns
    radiance into its raw values).

    ``full_well_e`` / ``read_noise_e``: saturation capacity and temporal dark noise of a pixel. ``readouts``: an HDR
    sensor's readouts of one exposure, most sensitive first, each (relative sensitivity, full well, read noise); the
    camera combines them into one linear value of ``bit_depth`` bits. ``sensitivity``: the pixel's response relative
    to a plain 3.45 um pixel behind the same lens (a polarizer halves it, at best)."""
    name: str
    width: int
    height: int
    pixel_um: float
    bit_depth: int
    cfa: str = "RGGB"            # "mono", a Bayer pattern, or "polar_RGGB" (2x2 polarizers under each colour of a quad Bayer)
    shutter: str = "global"
    polarizer_deg: tuple = ()    # polar sensors: the angles of a 2x2 polarizer block, row by row
    extinction_ratio: float = 0.0
    full_well_e: float = 10000.0
    read_noise_e: float = 2.5
    dark_current_e_s: float = 0.0
    black_level: float = 0.0     # the raw value of no light, as a share of the raw range
    sensitivity: float = 1.0
    readouts: tuple = ()
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
    falloff: float = 4.0         # light towards the corners: cos^falloff of the angle off the axis (4: a thin lens)
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


# Sony IMX490 (LUCID's EMVA 1288 brief for the TRI054S: temporal dark noise 0.54 e-, dynamic range 123.6 dB, maximum SNR
# 50.8 dB, combined saturation capacity 120 000 e-). Each pixel has a large and a small photodiode, each read at a high
# and a low conversion gain: four 12-bit readouts of one exposure. How sensitivity and capacity split over the four is
# not published: the split below reproduces the brief's three numbers (noise floor of the first readout, 123.6 dB to
# the last one's saturation, 50.8 dB at 120 000 e-) and is otherwise an assumption.
IMX490 = SensorSpec("Sony IMX490", 2880, 1860, 3.0, 24, "RGGB", "rolling", full_well_e=120000.0, read_noise_e=0.54,
                    black_level=0.0, sensitivity=0.76,
                    readouts=((1.0, 3000.0, 0.54), (1.0, 20000.0, 4.0), (0.147, 12000.0, 4.0), (0.147, 120000.0, 30.0)),
                    note="24-bit linear HDR from four readouts; the readouts' split and the shutter type are assumptions")
# Sony IMX265 (LUCID's EMVA 1288 data for the TRI032S-C: saturation capacity 10 565 e-, temporal dark noise 2.11 e-).
# The rig's recordings go down to 7 of 4095 in a short exposure: the black level is at most that, taken as 0.
IMX265 = SensorSpec("Sony IMX265", 2048, 1536, 3.45, 12, "RGGB", "global", full_well_e=10565.0, read_noise_e=2.11)
# Sony IMX250MYR (FLIR's EMVA page for the BFS-U3-51S5PC: 0.17 e- per 16-bit count, 11.88 bits of dynamic range: about
# 11 100 e- and 2.9 e-). Polarizer angles row by row and the extinction ratio are Sony's published layout and typical
# value, not checked against a camera.
IMX250MYR = SensorSpec("Sony IMX250MYR", 2448, 2048, 3.45, 12, "polar_RGGB", "global", (90.0, 45.0, 135.0, 0.0), 300.0,
                       full_well_e=11100.0, read_noise_e=2.9, sensitivity=0.45,
                       note="polarizer layout, extinction ratio and sensitivity are typical values, not measured")


@dataclass
class CameraState:
    exposure: object


@dataclass
class CameraSpec(Spec):
    kind = "camera"
    sensor: SensorSpec = IMX265
    lens: LensSpec = field(default_factory=FujinonHF6XA5M)
    model: str = "camera"
    render_scale: float = 0.25
    body_mm: tuple = (29.0, 29.0, 45.0)      # width, height, length of the housing (drawn behind the lens)
    calibration: Intrinsics | None = None    # measured intrinsics, at the size they were measured: replaces the lens model

    def __post_init__(self):
        w, h = self.sensor.size_mm()
        if math.hypot(w, h) > self.lens.image_circle_mm + 1e-6:
            raise ValueError(f"{self.lens.name}'s image circle ({self.lens.image_circle_mm} mm) does not cover "
                             f"{self.sensor.name} ({math.hypot(w, h):.2f} mm diagonal)")
        if not 0.0 < self.render_scale <= 1.0:
            raise ValueError("render_scale is a share of the sensor's resolution: 0 < render_scale <= 1")

    def full_intrinsics(self) -> Intrinsics:
        """The camera's own intrinsics: its calibration at the size it was measured (a real camera is often read out
        binned: the rig's TDR054S at 1440 x 928, its TRI032S at 1024 x 768), else the lens model at full resolution."""
        if self.calibration is not None:
            return self.calibration
        f = self.lens.focal_mm * 1000.0 / self.sensor.pixel_um
        return Intrinsics(self.sensor.width, self.sensor.height, f, f, (self.sensor.width - 1) / 2.0,
                          (self.sensor.height - 1) / 2.0, tuple(self.lens.dist))

    def image_size(self, scale: float | None = None) -> tuple[int, int]:
        """Rendered image size: ``scale`` of the sensor's width, with the shape of the camera's own image, even in both
        directions (a mosaic's 2x2 blocks stay whole)."""
        s = self.render_scale if scale is None else float(scale)
        full = self.full_intrinsics()
        width = max(2, 2 * round(self.sensor.width * s / 2))
        return width, max(2, 2 * round(width * full.height / full.width / 2))

    def intrinsics(self, scale: float | None = None) -> Intrinsics:
        return self.full_intrinsics().scaled(*self.image_size(scale))

    def make_state(self):
        """A camera's settings that change while it runs: ``state.exposure`` (hw.sensor.ExposureControl: auto exposure
        on; set ``auto = False`` and ``exposure_s`` / ``gain_db`` for a fixed exposure)."""
        from .sensor import ExposureControl
        return CameraState(ExposureControl())

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
