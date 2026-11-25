"""Sensors: what turns the scene into a measurement. Every sensor answers the same two questions, so the world, the
viewer and recordings need not know its kind:

    request = spec.request(world, actor, **quality)      # what to render: {"views": [...], "time", "pose", ...}
    outs = world.client.render(request["views"], "float32")
    reading = spec.read(world, actor, request, outs)     # Reading: its data, a picture of it, a line about it

``world.sense(actor)`` is the three in a row. Cameras (hw.cameras.CameraSpec) are sensors whose picture needs light
(``lit``: many samples, the denoiser); ``DepthCamera`` and ``Lidar`` measure geometry: one exact ray a pixel, their
noise added after. A new sensor subclasses ``Sensor`` and is named in configurations with hw.config.register_sensor.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field

import numpy as np

from .actor import Actor, Spec
from .transform import Transform, camera_to_world, optical_pose


@dataclass
class Reading:
    """One measurement. ``data``: arrays by name (a camera: ``raw`` (hw.sensor.Raw), ``frame``; a depth camera:
    ``depth``; a LiDAR: ``ranges``, ``points``, ...). ``picture``: H x W x 3 uint8, to look at. ``info``: what a
    display says about it, in order."""
    sensor: str
    modality: str
    time: float
    pose: np.ndarray
    data: dict
    picture: np.ndarray
    info: dict = field(default_factory=dict)


@dataclass
class Sensor(Spec):
    """``rate_hz``: measurements a second at most (0: as often as asked). ``lit``: the measurement depends on light (it
    is rendered with many samples and denoised), else on geometry alone. ``polarized``: it needs polarized light."""
    kind = "sensor"
    modality = "sensor"
    lit = True
    rate_hz: float = 0.0

    @property
    def polarized(self) -> bool:
        return False

    def request(self, world, actor: Actor, spp: int | None = None, **options) -> dict:
        raise NotImplementedError

    def read(self, world, actor: Actor, request: dict, outs: list) -> Reading:
        raise NotImplementedError

    def snapshot(self, state):
        """The state a measurement changes (a camera's auto exposure), to take one again as if it had not been."""
        return None

    def restore(self, state, snap) -> None:
        pass


def _ramp(x: np.ndarray) -> np.ndarray:
    """0..1 to colour, near to far (dark red, orange, yellow, green, blue); uint8."""
    stops = np.array([[0.45, 0.02, 0.03], [0.95, 0.45, 0.10], [0.98, 0.90, 0.25], [0.15, 0.80, 0.55], [0.15, 0.35, 0.85]])
    t = np.clip(x, 0.0, 1.0) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2)
    f = (t - i)[..., None]
    return np.clip(np.rint((stops[i] * (1 - f) + stops[i + 1] * f) * 255), 0, 255).astype(np.uint8)


def _rng(actor, request) -> np.random.Generator:
    """Noise of one measurement: the same for the same sensor at the same time."""
    return np.random.default_rng(zlib.crc32(f"{actor.name}/{request['time']:.4f}".encode()))


def _depth_view(world, pose: np.ndarray, width: int, height: int, hfov_deg: float) -> dict:
    """A depth view from a body frame (x along the axis): no light, no samples."""
    view = {"scan": world.scene, "variant": world.variant, "mode": "polar", "seed": 0, "spp": 1, "renderer": "depth", "depth": True,
            "preview": False, "camera_to_world": camera_to_world(pose), "width": int(width), "height": int(height), "hfov_deg": float(hfov_deg)}
    bodies = world.bodies()
    if bodies:
        view["bodies"] = bodies
    return view


@dataclass
class DepthCamera(Sensor):
    """A depth camera: distance along its axis per pixel. ``baseline_m`` and ``disparity_noise_px`` give a stereo
    pair's error, growing with the distance squared (sigma = z^2 * noise / (focal length in pixels * baseline)); 0
    leaves the true depth. Out of ``range_m`` (and where nothing is): 0. Links: ``depth`` (x along the axis),
    ``optical``. Glass and mirrors are surfaces to it (a real one often sees through or into them)."""
    modality = "depth"
    lit = False
    width: int = 640
    height: int = 400
    hfov_deg: float = 80.0
    range_m: tuple = (0.3, 12.0)
    baseline_m: float = 0.0
    disparity_noise_px: float = 0.0
    render_scale: float = 0.5
    body_mm: tuple = (97.0, 30.0, 23.0)     # width, height, depth of the housing
    drawn: bool = True
    model: str = "depth camera"

    def link_frames(self, state) -> dict:
        from .transform import BODY_TO_OPTICAL
        return {"depth": np.eye(4), "optical": BODY_TO_OPTICAL.copy()}

    def image_size(self) -> tuple[int, int]:
        w = max(16, 2 * round(self.width * self.render_scale / 2))
        return w, max(16, 2 * round(w * self.height / self.width / 2))

    def focal_px(self, width: int) -> float:
        return width / 2.0 / math.tan(math.radians(self.hfov_deg) / 2.0)

    def visuals(self) -> dict:
        from .robots import Visual
        if not self.drawn:
            return {}
        w, h, d = (v / 1000.0 for v in self.body_mm)
        dark = ("plastic", (0.03, 0.03, 0.035), 0.35)
        eyes = [Visual("cylinder", (0.006, 0.004), Transform((-0.003, y, 0.0), (0.0, 0.0, 90.0)), ("plastic", (0.01, 0.01, 0.015), 0.1))
                for y in (-0.375 * w, 0.0, 0.375 * w)]
        return {"depth": [Visual("box", (d, w, h), Transform((-d / 2.0 - 0.006, 0.0, 0.0)), dark), *eyes]}

    def request(self, world, actor, spp=None, **options) -> dict:
        pose = actor.link_matrix("depth")
        w, h = self.image_size()
        return {"views": [_depth_view(world, pose, w, h, self.hfov_deg)], "time": world.time, "pose": optical_pose(pose)}

    def read(self, world, actor, request, outs) -> Reading:
        z = np.array(outs[0]["depth"], dtype=np.float64)
        w = z.shape[1]
        if self.baseline_m > 0.0 and self.disparity_noise_px > 0.0:
            # the noise of the sensor's own pixels: a rendered pixel stands for one of them
            sigma = z ** 2 * self.disparity_noise_px / (self.focal_px(self.width) * self.baseline_m)
            z = z + _rng(actor, request).normal(0.0, 1.0, z.shape) * sigma
        ok = (z >= self.range_m[0]) & (z <= self.range_m[1]) & (np.array(outs[0]["depth"]) > 0.0)
        z = np.where(ok, z, 0.0).astype(np.float32)
        lo, hi = math.log(max(self.range_m[0], 0.05)), math.log(self.range_m[1])
        picture = _ramp((np.log(np.maximum(z, 1e-3)) - lo) / (hi - lo))
        picture[~ok] = 0
        f = self.focal_px(w)
        return Reading(actor.name, self.modality, request["time"], request["pose"],
                       {"depth": z, "focal_px": f, "valid": ok}, picture,
                       {"model": self.model, "size": f"{w}x{z.shape[0]}", "range": f"{self.range_m[0]:g}-{self.range_m[1]:g} m",
                        "valid": f"{100 * float(ok.mean()):.0f}%"})


@dataclass
class Lidar(Sensor):
    """A spinning LiDAR: ``samples`` directions a turn at each of ``elevations_deg`` (one, 0: a 2D scanner). ``noise_m``
    and ``noise_share`` (of the distance) are its range error's standard deviation. Links: ``lidar`` (x forward,
    z up: the scan plane). ``data``: ``ranges`` (elevations x samples, 0: no return), ``azimuth_deg`` (counter-clockwise
    from x), ``points`` (N x 3, the lidar's frame). A whole turn is taken at one moment: a moving robot's scan is not
    sheared."""
    modality = "lidar"
    lit = False
    samples: int = 360
    elevations_deg: tuple = (0.0,)
    range_m: tuple = (0.15, 12.0)
    noise_m: float = 0.0
    noise_share: float = 0.0
    resolution_px: int = 512                 # of each of the four quadrants it is rendered as
    body_mm: tuple = (70.0, 41.0)            # diameter, height
    drawn: bool = True
    model: str = "LiDAR"

    def link_frames(self, state) -> dict:
        return {"lidar": np.eye(4)}

    def visuals(self) -> dict:
        from .robots import Visual
        if not self.drawn:
            return {}
        d, h = (v / 1000.0 for v in self.body_mm)
        # under the scan plane, so that it does not stand in its own light
        return {"lidar": [Visual("cylinder", (d / 2.0, h), Transform((0.0, 0.0, -h / 2.0 - 0.006), (90.0, 0.0, 0.0)), ("plastic", (0.03, 0.03, 0.035), 0.3))]}

    def _rows(self) -> tuple[int, float]:
        """(image rows, tan of the largest elevation the image must hold at a quadrant's edge)."""
        top = max(abs(float(e)) for e in self.elevations_deg)
        t = math.tan(math.radians(min(top, 80.0)))               # at the quadrant's axis; its edge needs no more height
        rows = 2 * max(8, math.ceil(t * math.sqrt(2.0) * self.resolution_px / 2.0) + 2)
        return min(rows, 2 * self.resolution_px), t

    def request(self, world, actor, spp=None, **options) -> dict:
        pose = actor.link_matrix("lidar")
        rows, _ = self._rows()
        views = []
        for q in range(4):
            turn = Transform((0.0, 0.0, 0.0), (0.0, 0.0, 90.0 * q)).matrix()
            views.append(_depth_view(world, pose @ turn, self.resolution_px, rows, 90.0))
        return {"views": views, "time": world.time, "pose": pose}

    def read(self, world, actor, request, outs) -> Reading:
        az = np.arange(self.samples) * (360.0 / self.samples)
        n_px = self.resolution_px
        ranges = np.zeros((len(self.elevations_deg), self.samples))
        for i, e in enumerate(self.elevations_deg):
            te = math.tan(math.radians(float(e)))
            for q in range(4):
                rel = (az - 90.0 * q + 180.0) % 360.0 - 180.0
                sel = np.nonzero((rel >= -45.0) & (rel < 45.0))[0]
                z = np.asarray(outs[q]["depth"], dtype=np.float64)
                r = np.radians(rel[sel])
                # a direction's place in the quadrant's image: left is +y, up is +z
                u = (1.0 - np.tan(r)) * 0.5 * n_px - 0.5
                v = z.shape[0] / 2.0 - te / np.cos(r) * 0.5 * n_px - 0.5
                col = np.clip(np.rint(u).astype(int), 0, n_px - 1)
                row = np.clip(np.rint(v).astype(int), 0, z.shape[0] - 1)
                d = z[row, col]
                # depth along the quadrant's axis to distance along the ray the pixel stands for
                ray = np.sqrt(1.0 + ((2 * (col + 0.5) / n_px - 1.0)) ** 2 + ((z.shape[0] - 2 * (row + 0.5)) / n_px) ** 2)
                ranges[i, sel] = d * ray
        hit = ranges > 0.0
        if self.noise_m > 0.0 or self.noise_share > 0.0:
            rng = _rng(actor, request)
            ranges = ranges + rng.normal(0.0, 1.0, ranges.shape) * np.hypot(self.noise_m, self.noise_share * ranges)
        ok = hit & (ranges >= self.range_m[0]) & (ranges <= self.range_m[1])
        ranges = np.where(ok, ranges, 0.0)
        a, el = np.radians(az)[None], np.radians(np.asarray(self.elevations_deg, dtype=np.float64))[:, None]
        xyz = np.stack([ranges * np.cos(el) * np.cos(a), ranges * np.cos(el) * np.sin(a), ranges * np.sin(el) * np.ones_like(a)], -1)
        points = xyz[ok]
        return Reading(actor.name, self.modality, request["time"], request["pose"],
                       {"ranges": ranges.astype(np.float32), "azimuth_deg": az, "points": points.astype(np.float32)}, self._picture(points),
                       {"model": self.model, "samples": f"{self.samples} x {len(self.elevations_deg)}",
                        "range": f"{self.range_m[0]:g}-{self.range_m[1]:g} m", "returns": f"{100 * float(ok.mean()):.0f}%"})

    def _picture(self, points: np.ndarray, side: int = 480) -> np.ndarray:
        """The scan from above, forward up, the farthest return near the edge; a ring every metre."""
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (side, side), (8, 11, 13))
        d = ImageDraw.Draw(img)
        far = max(1.0, float(np.hypot(points[:, 0], points[:, 1]).max()) if len(points) else 1.0)
        k = (side / 2 - 12) / far
        for r in range(1, int(far) + 1):
            d.ellipse((side / 2 - r * k, side / 2 - r * k, side / 2 + r * k, side / 2 + r * k), outline=(32, 41, 46))
        for x, y, _ in points:
            px, py = side / 2 - y * k, side / 2 - x * k
            c = tuple(int(v) for v in _ramp(np.array([math.hypot(x, y) / far]))[0])
            d.rectangle((px - 1.5, py - 1.5, px + 1.5, py + 1.5), fill=c)
        d.polygon([(side / 2, side / 2 - 9), (side / 2 - 6, side / 2 + 6), (side / 2 + 6, side / 2 + 6)], fill=(63, 201, 212))
        return np.asarray(img)
