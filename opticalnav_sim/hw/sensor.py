"""From the light on a pixel to the raw value the camera delivers: polarizer, colour mosaic, photoelectrons, noise,
saturation, the analogue-to-digital converter (EMVA 1288's linear camera model).

    raw = sensor.expose(frame, camera.spec, exposure_s=0.005)     # frame: a SimWorld render (Stokes images)
    raw.data            # H x W unsigned integers, one colour (and, on a polarization sensor, one angle) a pixel
    image = isp.process(raw)

The simulator's radiance is not in physical units, so one scale is free: ``RESPONSIVITY``, the electrons a second a
plain 3.45 um pixel behind an f/4 lens collects from radiance 1. Auto exposure (``ExposureControl``, on by default)
sets the exposure time as a camera would; exposures of two cameras of one rig are then in the right ratio (pixel size, f-number, sensitivity).
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np

from .optics import Intrinsics

RESPONSIVITY = 1.0e6   # e- / s for radiance 1.0: a 3.45 um pixel of sensitivity 1 at f/4 (see the module's note)
BAYER = {"RGGB": ((0, 1), (1, 2)), "BGGR": ((2, 1), (1, 0)), "GRBG": ((1, 0), (2, 1)), "GBRG": ((1, 2), (0, 1))}


@dataclass
class Raw:
    """A raw frame. ``data``: H x W, the sensor's counts (``black`` at no light, ``white`` at saturation).
    ``colour``: H x W, the channel (0 R, 1 G, 2 B) of each pixel; ``angle``: H x W polarizer angles (degrees, from the
    image's x axis towards its up), or None."""
    data: np.ndarray
    cfa: str
    bit_depth: int
    black: float
    white: float
    exposure_s: float
    gain_db: float
    intrinsics: Intrinsics | None = None
    pose: np.ndarray | None = None
    polarizer_deg: tuple = ()
    time: float = 0.0
    meta: dict = field(default_factory=dict)

    def colour(self) -> np.ndarray:
        return mosaic(self.cfa, *self.data.shape)[0]

    def angle(self) -> np.ndarray | None:
        return mosaic(self.cfa, *self.data.shape, self.polarizer_deg)[1]

    def linear(self) -> np.ndarray:
        """Counts as a share of the range above black: 0 no light, 1 saturation."""
        return (self.data.astype(np.float64) - self.black) / (self.white - self.black)


def mosaic(cfa: str, height: int, width: int, polarizer_deg: tuple = ()) -> tuple[np.ndarray, np.ndarray | None]:
    """(colour index, polarizer angle or None) of every pixel. ``polar_<bayer>``: 2x2 blocks of one colour (a quad
    Bayer), the four polarizer angles inside each block."""
    r, c = np.indices((height, width))
    if cfa == "mono":
        return np.ones((height, width), np.int64), None
    if cfa.startswith("polar_"):
        table = np.array(BAYER[cfa[6:]])
        angles = np.array(polarizer_deg, dtype=np.float64).reshape(2, 2)
        return table[(r // 2) % 2, (c // 2) % 2], angles[r % 2, c % 2]
    return np.array(BAYER[cfa])[r % 2, c % 2], None


def sensor_frame_rotation(k: Intrinsics, world_from_optical: np.ndarray) -> np.ndarray:
    """Per pixel, the angle (radians) from the renderer's Stokes x axis to the sensor's, about the ray.

    The render server gives Stokes vectors in the basis ``world_gravity_y_v1``: looking along a ray d, x is d x up
    (horizontal, to the right) and y is x x d (up as far as the ray allows). A sensor's polarizers are fixed to its
    rows: x is the image's x axis as that ray sees it. The two agree for a level camera on its axis and differ by the
    roll of the camera and, off the axis, by a little that grows towards the corners."""
    u, v = np.meshgrid(np.arange(k.width, dtype=np.float64), np.arange(k.height, dtype=np.float64))
    rays = k.rays(np.stack([u.ravel(), v.ravel()], 1))
    d = np.c_[rays, np.ones(len(rays))]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    rot = np.asarray(world_from_optical, dtype=np.float64)[:3, :3]
    d = d @ rot.T                                         # world, z up
    up = np.array([0.0, 0.0, 1.0])
    bx = np.cross(d, up)
    bx /= np.maximum(np.linalg.norm(bx, axis=1, keepdims=True), 1e-12)
    by = np.cross(bx, d)
    right = rot[:, 0]                                     # the image's x axis (optical x)
    h = right - d * (d @ right)[:, None]
    return np.arctan2((h * by).sum(1), (h * bx).sum(1)).reshape(k.height, k.width)


def to_sensor_frame(s1: np.ndarray, s2: np.ndarray, angle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """S1, S2 in a basis turned by ``angle`` (radians, H x W) from the one they are given in."""
    c, s = np.cos(2.0 * angle), np.sin(2.0 * angle)
    if s1.ndim == 3:
        c, s = c[..., None], s[..., None]
    return s1 * c + s2 * s, -s1 * s + s2 * c


@functools.lru_cache(maxsize=64)
def vignetting(k: Intrinsics, power: float = 4.0) -> np.ndarray:
    """A lens's fall-off towards the corners: cos^power of the ray's angle off the axis (H x W, 1 on the axis).
    4 is the natural vignetting of a thin lens; a real wide-angle lens falls off less (LensSpec.falloff). Kept per
    camera (every pixel's ray is an inverse of the distortion: a third of a second for a 0.3 MP image); do not write
    into the result."""
    u, v = np.meshgrid(np.arange(k.width, dtype=np.float64), np.arange(k.height, dtype=np.float64))
    rays = k.rays(np.stack([u.ravel(), v.ravel()], 1))
    return ((1.0 + (rays ** 2).sum(1)) ** (-float(power) / 2.0)).reshape(k.height, k.width)


def responsivity(spec) -> float:
    """Electrons a second from radiance 1.0 on a pixel of this camera (pixel area, f-number, the sensor's sensitivity)."""
    s = spec.sensor
    return RESPONSIVITY * s.sensitivity * (s.pixel_um / 3.45) ** 2 * (4.0 / spec.lens.f_number) ** 2


def irradiance(frame, spec, vignette: bool = True) -> np.ndarray:
    """The light each pixel's well sees, H x W, in the simulator's radiance units: the colour of its filter, through
    its polarizer when it has one, dimmed by the lens towards the corners."""
    sensor, k = spec.sensor, frame.intrinsics
    s0 = np.asarray(frame.s0, dtype=np.float64)
    h, w = s0.shape[:2]
    colour, angle = mosaic(sensor.cfa, h, w, sensor.polarizer_deg)
    rows, cols = np.indices((h, w))
    if sensor.cfa == "mono":
        light = s0 @ np.array([0.2126, 0.7152, 0.0722])
        pick = lambda a: np.asarray(a, dtype=np.float64) @ np.array([0.2126, 0.7152, 0.0722])  # noqa: E731
    else:
        pick = lambda a: np.asarray(a, dtype=np.float64)[rows, cols, colour]  # noqa: E731
        light = pick(s0)
    if angle is not None:
        if frame.s1 is None or frame.s2 is None:
            raise ValueError("a polarization sensor needs the frame's S1 and S2")
        s1, s2 = to_sensor_frame(pick(frame.s1), pick(frame.s2), sensor_frame_rotation(k, frame.pose))
        t = np.radians(angle)
        er = sensor.extinction_ratio or math.inf
        d = (er - 1.0) / (er + 1.0) if math.isfinite(er) else 1.0   # diattenuation of the wire grid
        light = 0.5 * (light + d * (s1 * np.cos(2.0 * t) + s2 * np.sin(2.0 * t)))
    if vignette:
        light = light * vignetting(k, spec.lens.falloff)
    return np.maximum(light, 0.0)


@dataclass
class ExposureControl:
    """A camera's exposure time and gain, and its auto exposure (a machine vision camera's ExposureAuto and GainAuto
    set to Continuous: on by default here too).

    Each frame is metered (the mean of the image as a share of the sensor's range) and the next frame's exposure moves
    towards the one that would put that mean at ``target``, by ``1 - damping`` of the way in stops: a camera driven
    from a bright room into a dark one takes a few frames to follow. The time is used first, up to its limit, then
    gain. ``auto = False`` holds ``exposure_s`` and ``gain_db`` as set."""
    auto: bool = True
    exposure_s: float | None = None      # None: not started (the first frame starts settled, as a running camera is)
    gain_db: float = 0.0
    target: float = 0.27                 # LUCID's default target brightness, 70 of 255
    damping: float = 0.5
    exposure_limits: tuple = (2.0e-5, 0.03)
    gain_limits: tuple = (0.0, 24.0)

    def level(self, frame, spec, exposure_s: float, gain_db: float, light: np.ndarray | None = None) -> float:
        """What the meter reads: the mean of the image at this setting (clipped pixels count as full). ``light``: the
        frame's ``irradiance``, when the caller has it."""
        well = spec.sensor.readouts[0][1] * 8.0 if spec.sensor.readouts else spec.sensor.full_well_e
        light = irradiance(frame, spec) if light is None else light
        signal = light * (responsivity(spec) * exposure_s * 10.0 ** (gain_db / 20.0) / well)
        return float(np.minimum(signal, 1.0).mean())

    def _towards(self, frame, spec, exposure_s: float, gain_db: float, share: float, light) -> tuple[float, float]:
        level = max(self.level(frame, spec, exposure_s, gain_db, light), 1e-9)
        total = exposure_s * 10.0 ** (gain_db / 20.0) * (self.target / level) ** share   # time x gain: the sensitivity
        low_gain = 10.0 ** (self.gain_limits[0] / 20.0)
        t = float(np.clip(total / low_gain, *self.exposure_limits))
        return t, float(np.clip(20.0 * math.log10(max(total / t, 1e-12)), *self.gain_limits))

    def settle(self, frame, spec, light: np.ndarray | None = None) -> tuple[float, float]:
        """The setting the auto exposure ends at on this frame (as after many frames of the same view)."""
        light = irradiance(frame, spec) if light is None else light
        t, g = self.exposure_s or 0.005, self.gain_db
        for _ in range(30):  # clipped highlights make it more than one step
            t, g = self._towards(frame, spec, t, g, 1.0, light)
        return t, g

    def next(self, frame, spec, light: np.ndarray | None = None) -> tuple[float, float]:
        """(exposure, gain) to take ``frame`` with; the setting then moves on for the frame after it."""
        if not self.auto:
            if self.exposure_s is None:
                self.exposure_s = 0.005
            return self.exposure_s, self.gain_db
        light = irradiance(frame, spec) if light is None else light
        if self.exposure_s is None:
            self.exposure_s, self.gain_db = self.settle(frame, spec, light)
        used = (self.exposure_s, self.gain_db)
        self.exposure_s, self.gain_db = self._towards(frame, spec, *used, 1.0 - self.damping, light)
        return used


def auto_exposure(frame, spec, **settings) -> float:
    """The exposure time a camera's auto exposure settles at on ``frame`` (ExposureControl's ``settings``)."""
    return ExposureControl(**settings).settle(frame, spec)[0]


def expose(frame, spec, exposure_s: float | None = None, gain_db: float = 0.0, noise: bool = True, vignette: bool = True,
           seed: int | None = 0, pixels_per_sample: float = 1.0, light: np.ndarray | None = None) -> Raw:
    """The raw frame a camera delivers of ``frame`` (a SimWorld render at the camera's image size).

    ``exposure_s``: None is a settled auto exposure (ExposureControl). ``gain_db``: analogue gain. ``noise`` off gives the mean
    (still clipped and quantized). ``pixels_per_sample``: how many sensor pixels are averaged into one rendered pixel
    (binning: 4 for 2x2); the noise of the mean shrinks accordingly. The mosaic is laid over the rendered image: at a
    reduced render size it is a sensor of that size, not a crop of the real one."""
    sensor = spec.sensor
    if light is None:  # ``light``: the frame's irradiance, when the caller has computed it (auto exposure needs it too)
        light = irradiance(frame, spec, vignette)
    if exposure_s is None:  # a settled auto exposure, gain included
        exposure_s, auto_gain = ExposureControl().settle(frame, spec, light if vignette else None)
        gain_db = gain_db + auto_gain
    rng = np.random.default_rng(seed)
    electrons = light * (responsivity(spec) * float(exposure_s))
    n = float(pixels_per_sample)

    def read(mean_e: np.ndarray, full_well: float, read_noise: float) -> np.ndarray:
        """Electrons read from a well: shot noise, the well's capacity, the readout's noise."""
        dark = sensor.dark_current_e_s * float(exposure_s)
        if not noise:
            return np.minimum(mean_e + dark, full_well)
        got = rng.poisson((mean_e + dark) * n).astype(np.float64) / n
        got = np.minimum(got, full_well)
        return got + rng.normal(0.0, read_noise / math.sqrt(n), mean_e.shape)

    full = float(2 ** sensor.bit_depth - 1)
    black = round(sensor.black_level * full)
    gain = 10.0 ** (float(gain_db) / 20.0)
    if sensor.readouts:
        # each readout is a 12-bit conversion of its own well; the camera takes the most sensitive one that is not
        # near its top and scales it to the common linear axis (electrons of the first readout)
        value = np.zeros_like(electrons)
        taken = np.zeros(electrons.shape, bool)
        for i, (share, well, read_noise) in enumerate(sensor.readouts):
            counts = np.clip(np.rint(read(electrons * share, well, read_noise) / well * 4095.0), 0, 4095)
            last = i == len(sensor.readouts) - 1
            use = ~taken & ((counts < 0.9 * 4095.0) | last)
            value[use] = counts[use] / 4095.0 * well / share
            taken |= use
        top = sensor.readouts[-1][1] / sensor.readouts[-1][0]
        data = black + value / top * (full - black) * gain
    else:
        data = black + read(electrons, sensor.full_well_e, sensor.read_noise_e) / sensor.full_well_e * (full - black) * gain
    data = np.clip(np.rint(data), 0, full).astype(np.uint16 if sensor.bit_depth <= 16 else np.uint32)
    return Raw(data, sensor.cfa, sensor.bit_depth, float(black), full, float(exposure_s), float(gain_db), frame.intrinsics,
               frame.pose, tuple(sensor.polarizer_deg), frame.time,
               {"camera": spec.model, "sensor": sensor.name, "noise": bool(noise), "vignette": bool(vignette)})
