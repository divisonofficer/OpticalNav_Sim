"""From the light on a pixel to the raw value the camera delivers: polarizer, colour mosaic, photoelectrons, noise,
saturation, the analogue-to-digital converter (EMVA 1288's linear camera model).

    raw = sensor.expose(frame, camera.spec, exposure_s=0.005)     # frame: a SimWorld render (Stokes images)
    raw.data            # H x W unsigned integers, one colour (and, on a polarization sensor, one angle) a pixel
    image = isp.process(raw)

The simulator's radiance is not in physical units, so one scale is free: ``RESPONSIVITY``, the electrons a second a
plain 3.45 um pixel behind an f/4 lens collects from radiance 1. ``auto_exposure`` picks the exposure time as a camera
would; exposures of two cameras of one rig are then in the right ratio (pixel size, f-number, sensitivity).
"""
from __future__ import annotations

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


def vignetting(k: Intrinsics, power: float = 4.0) -> np.ndarray:
    """A lens's fall-off towards the corners: cos^power of the ray's angle off the axis (H x W, 1 on the axis).
    4 is the natural vignetting of a thin lens; a real wide-angle lens falls off less (LensSpec.falloff)."""
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


def auto_exposure(frame, spec, target: float = 0.6, percentile: float = 99.0, limits: tuple = (2e-5, 0.1)) -> float:
    """The exposure time that puts the ``percentile`` brightest pixel at ``target`` of saturation (of the most
    sensitive readout's range times 8, on an HDR sensor: its highlights go to the later readouts)."""
    sensor = spec.sensor
    level = float(np.percentile(irradiance(frame, spec), percentile)) * responsivity(spec)
    well = sensor.readouts[0][1] * 8.0 if sensor.readouts else sensor.full_well_e
    return float(np.clip(target * well / max(level, 1e-12), *limits))


def expose(frame, spec, exposure_s: float | None = None, gain_db: float = 0.0, noise: bool = True, vignette: bool = True,
           seed: int | None = 0, pixels_per_sample: float = 1.0) -> Raw:
    """The raw frame a camera delivers of ``frame`` (a SimWorld render at the camera's image size).

    ``exposure_s``: None picks it (``auto_exposure``). ``gain_db``: analogue gain. ``noise`` off gives the mean
    (still clipped and quantized). ``pixels_per_sample``: how many sensor pixels are averaged into one rendered pixel
    (binning: 4 for 2x2); the noise of the mean shrinks accordingly. The mosaic is laid over the rendered image: at a
    reduced render size it is a sensor of that size, not a crop of the real one."""
    sensor = spec.sensor
    if exposure_s is None:
        exposure_s = auto_exposure(frame, spec)
    rng = np.random.default_rng(seed)
    electrons = irradiance(frame, spec, vignette) * responsivity(spec) * float(exposure_s)
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
