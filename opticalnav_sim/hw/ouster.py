"""An Ouster digital LiDAR (OS0, OS1, OS2) with its measurement noise, from what the beams meet in the scene.

    part = {"name": "lidar", "type": "ouster", "params": {"model": "OS1-64"}}     # a robot's or a rig's part

The render server traces each beam's footprint (renderer "lidar", server.Resident.lidar_image): the path length to the
first surface that sends light back, through glass and off mirrors; that surface's monostatic near-infrared reflectance
(pi times its BSDF with the light going back the way it came, the cosine of incidence included: a white Lambertian wall
head-on gives 1, at 60 degrees 0.5; a glossy surface its highlight's value); and whether the first surface was glass or
a mirror. ``measure`` turns that into what the sensor reports:

1. **Footprint and returns.** A beam is seven rays across its divergence (the centre, and six at half the FWHM, half the
   weight: a Gaussian beam). Rays whose ranges lie within ``pulse_m`` of each other make one return, at their
   signal-weighted mean range: a beam across an edge whose sides are that close gives a mixed point between them.
   Farther apart they are separate returns; the strongest is the first, the next the second (dual-return mode).
2. **Detection.** The datasheet's ranges at >90 % detection for 10 % and 80 % Lambertian targets set the range at which
   a surface of any reflectance is still seen 90 % of the time, R90(rho) = R10 (rho / 0.1)^g (g from the two: 0.31 for
   the OS1, not the 0.5 of an inverse-square law); below that the probability falls as the Rev 6 datasheets' 50 % ranges
   say (a normal curve in log range). Indoors a Lambertian surface is always seen; what drops out are returns with
   almost nothing coming back: steep glossy metal, black glass, a mirror's or a window's surface off its normal.
3. **Glass and mirrors.** Through glass the beam goes on (0.92 a surface) to what is behind it; a mirror sends it on, so
   the sensor sees phantom points behind the mirror. Near normal incidence (``specular_cone_deg``) the glass or the
   mirror itself returns the beam too: one more candidate return.
4. **Range noise.** Gaussian; its standard deviation is the Rev 6 datasheet's precision for a 10 % Lambertian target at
   that range, scaled by sqrt(0.1 / rho) (shot noise: the signal goes with the reflectance), within the Rev 7 datasheet's
   span (OS1: 0.5 to 3 cm, up to the Rev 6 table's 5 cm far out). Each beam also has a constant offset (``bias_m``: its
   calibration error; the datasheet's accuracy is +-2.5 cm). Rounded to the millimetre the sensor reports.
5. **Channels.** RANGE (m, 0: no return or under the minimum range), SIGNAL (photons, Poisson), REFLECTIVITY (Ouster's
   8-bit code: 1-100 the Lambertian reflectance in percent, range-corrected but not for the incidence; above, a log2
   scale up to 255 for 864 times a white target), RANGE2 / REFLECTIVITY2 (the second return, kept only when it is at
   least ``second_ratio`` of the first: real OS0 data has second returns on 0.3 % of beams). NEAR_IR (ambient light)
   is not simulated: 0.

Each beam leaves at its own azimuth offset from the column's (the staggering of the real sensors' metadata) and from its
origin on the optics' circle (``origin_mm``). The data's B x C arrays are as the sensor gives them, staggered
(``destagger`` lines a row up with the azimuths, as ouster-sdk's does); the picture is destaggered. A whole turn is
taken at one moment.
Sources: Ouster Rev 7 and Rev 6 datasheets (data.ouster.io/downloads/datasheets), the sensor docs (static.ouster.dev),
ouster-sdk's test metadata; what they do not give is marked "estimate".
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

import numpy as np

from .sensing import Lidar, Reading, _ramp, _rng
from .transform import Transform

# One Ouster model (Rev 7 unless said): vertical field of view (deg, top, bottom), beams, columns, range (m, the minimum
# and what we keep), the 10 % and 80 % Lambertian targets' ranges at >90 % detection (1024 x 10 Hz, 100 klx), the log
# range from 90 % to 50 % detection (Rev 6: 10 % target), precision for a 10 % Lambertian target per range band (Rev 6,
# 1 sigma, m: up to that range), the Rev 7 precision span, beam divergence (deg, FWHM), the beams' azimuth offsets
# (deg, repeating; ouster-sdk metadata), the beam origin's offset from the axis (mm), the housing (mm).
MODELS = {
    "OS0-128": dict(vfov_deg=(45.4, -45.4), beams=128, columns=1024, range_m=(0.5, 100.0), range10_m=35.0, range80_m=75.0,
                    fall=math.log(20.0 / 15.0), precision=((1.0, 0.02), (10.0, 0.01), (15.0, 0.015), (1e9, 0.05)),
                    span=(0.008, 0.04), divergence_deg=0.35, stagger_deg=(10.85, 3.67, -3.40, -10.32), origin_mm=27.116,
                    body_mm=(85.0, 58.0)),
    "OS1-64": dict(vfov_deg=(21.2, -21.2), beams=64, columns=1024, range_m=(0.5, 200.0), range10_m=90.0, range80_m=170.0,
                   fall=math.log(55.0 / 45.0), precision=((1.0, 0.007), (20.0, 0.01), (50.0, 0.02), (1e9, 0.05)),
                   span=(0.005, 0.03), divergence_deg=0.18, stagger_deg=(4.2, 1.4, -1.4, -4.2), origin_mm=16.721,
                   body_mm=(87.0, 74.0)),
    "OS1-128": dict(vfov_deg=(21.2, -21.2), beams=128, columns=1024, range_m=(0.5, 200.0), range10_m=90.0, range80_m=170.0,
                    fall=math.log(55.0 / 45.0), precision=((1.0, 0.007), (20.0, 0.01), (50.0, 0.02), (1e9, 0.05)),
                    span=(0.005, 0.03), divergence_deg=0.18, stagger_deg=(4.2, 1.4, -1.4, -4.2), origin_mm=16.721,
                    body_mm=(87.0, 74.0)),
    "OS2-128": dict(vfov_deg=(11.25, -11.25), beams=128, columns=1024, range_m=(0.8, 400.0), range10_m=200.0, range80_m=350.0,
                    fall=math.log(100.0 / 80.0), precision=((30.0, 0.025), (60.0, 0.04), (1e9, 0.08)),
                    span=(0.02, 0.10), divergence_deg=0.09, stagger_deg=(2.07, 0.69, -0.69, -2.07), origin_mm=13.762,
                    body_mm=(124.0, 88.0)),
}
MODE_RANGE = {512: 1.19, 1024: 1.0, 2048: 0.84}     # range multiplier of the column count at 10 Hz (sensor docs)
Z90 = 1.2815515655446004      # the standard normal's 90 % point
RETRO_TOP = 864.0             # REFLECTIVITY 255: this many times a white Lambertian target


def _phi(x: np.ndarray) -> np.ndarray:
    from scipy.special import ndtr
    return ndtr(x)


def reflectivity_code(rho: np.ndarray) -> np.ndarray:
    """Ouster's 8-bit REFLECTIVITY of a reflectance (1 = white Lambertian): 1-100 linear in percent, then log2 up to 255
    at RETRO_TOP (a retroreflector); 0 is kept for no return."""
    pct = np.maximum(rho, 0.0) * 100.0
    above = 100.0 + 155.0 * np.log2(np.maximum(pct, 100.0) / 100.0) / math.log2(RETRO_TOP)
    return np.clip(np.where(pct <= 100.0, np.maximum(np.rint(pct), 1.0), np.rint(above)), 1.0, 255.0)


@dataclass
class OusterLidar(Lidar):
    """An Ouster OS-series LiDAR (MODELS) and its noise (module docstring). ``noise``: off gives the beams' true first
    surfaces (the footprint's centre, every return detected, no range noise), to compare with. ``altitudes_deg``: the
    beams' elevations from a real sensor's metadata (beam_altitude_angles), top first; empty: evenly over the field of
    view. Links: ``lidar`` (x forward, z up)."""
    model: str = "OS1-64"
    columns: int = 0                  # 0: the model's 1024 (512, 1024 or 2048)
    noise: bool = True
    altitudes_deg: tuple = ()
    pulse_m: float = 0.30             # returns closer than this merge into one (estimate)
    second_ratio: float = 0.3         # a second return is kept when at least this share of the first (estimate)
    specular_cone_deg: float = 2.0    # glass or a mirror returns the beam within this of normal incidence (estimate)
    specular_gain: float = 30.0       # its reflectance then, times its Fresnel share (0.04 glass, 0.9 mirror) (estimate)
    bias_m: float = 0.008             # each beam's constant range offset, 1 sigma (estimate: 3 sigma = the +-2.5 cm accuracy)
    photons_10: float = 60.0          # SIGNAL of the 10 % target at its 10 % range (estimate)
    oversample: int = 2               # rendered pixels per column, each way
    rate_hz: float = 10.0
    display: str = "all"              # the picture (VIEWS): "all" (the 3D view over the range and reflectivity images),
                                      # "cloud" (the 3D view), "range" or "intensity" (one image); a viewer's buttons switch it

    def __post_init__(self):
        if self.model not in MODELS:
            raise ValueError(f"Ouster model {self.model!r}: one of {sorted(MODELS)}")
        m = self.spec = MODELS[self.model]
        top, bottom = m["vfov_deg"]
        alt = tuple(float(a) for a in self.altitudes_deg) or tuple(float(e) for e in np.linspace(top, bottom, m["beams"]))
        self.elevations_deg = alt
        self.samples = int(self.columns or m["columns"])
        if self.samples not in MODE_RANGE:
            raise ValueError("an Ouster has 512, 1024 or 2048 columns")
        self.range_m = tuple(m["range_m"])
        self.body_mm = tuple(m["body_mm"])
        self.resolution_px = int(self.samples // 4 * max(1, int(self.oversample)))
        self.model_label = f"Ouster {self.model}"
        k = MODE_RANGE[self.samples]
        self.range10 = m["range10_m"] * k
        self.gamma = math.log(m["range80_m"] / m["range10_m"]) / math.log(8.0)
        st = np.asarray(m["stagger_deg"], dtype=np.float64)
        self.stagger = st[np.arange(len(alt)) % len(st)]

    def visuals(self) -> dict:
        """The housing as far as the beams leave it free: a puck under the lowest beam's cone (the beams start inside
        the real housing's window, which a ray from the axis would hit); the window and the cap are not drawn."""
        from .robots import Visual
        if not self.drawn:
            return {}
        d, h = (v / 1000.0 for v in self.body_mm)
        below = math.tan(math.radians(max(1.0, -min(self.elevations_deg)))) * d / 2.0 + 0.004   # the lowest beam at the rim
        tall = max(0.01, h - below - h / 3.0)
        return {"lidar": [Visual("cylinder", (d / 2.0, tall), Transform((0.0, 0.0, -below - tall / 2.0), (90.0, 0.0, 0.0)),
                                 ("metal", (0.55, 0.56, 0.58), 0.35))]}

    def request(self, world, actor, spp=None, **options) -> dict:
        req = super().request(world, actor, spp, **options)
        for v in req["views"]:
            v["renderer"] = "lidar"
            v.pop("depth", None)
        return req

    # --- the beams ---
    def _pixel(self, az_rel: np.ndarray, te: np.ndarray, rows: int) -> tuple[np.ndarray, np.ndarray]:
        """Fractional pixel (u, v) of directions at ``az_rel`` (rad, within a quadrant) and tan elevation ``te``."""
        n = self.resolution_px
        u = (1.0 - np.tan(az_rel)) * 0.5 * n - 0.5
        v = rows / 2.0 - te / np.cos(az_rel) * 0.5 * n - 0.5
        return u, v

    def beam_azimuths(self) -> np.ndarray:
        """B x C: each beam's azimuth (deg, counter-clockwise from x) at each column: the column's and its offset."""
        col = np.arange(self.samples) * (360.0 / self.samples)
        return col[None, :] + self.stagger[:, None]

    def footprint(self) -> tuple[list, np.ndarray]:
        """The rays a beam is traced as: offsets (rad, azimuth, elevation) and weights (summing to 1): the centre, and
        six at half the FWHM with half the centre's weight (a Gaussian beam)."""
        half = math.radians(self.spec["divergence_deg"]) / 2.0
        offs = [(0.0, 0.0)] + [(half * math.cos(math.pi * i / 3), half * math.sin(math.pi * i / 3)) for i in range(6)]
        w = np.array([1.0] + [0.5] * 6)
        return offs, w / w.sum()

    def _gather(self, outs: list) -> tuple[np.ndarray, np.ndarray]:
        """Each beam's footprint rays: (B x C x K x 6 channels, the columns' azimuths in deg)."""
        az = self.beam_azimuths()
        el = np.radians(np.asarray(self.elevations_deg))[:, None] * np.ones_like(az)
        offs, _ = self.footprint()
        out = np.zeros(az.shape + (len(offs), 6), np.float32)
        rows = outs[0]["lidar"].shape[0]
        for j, (da, de) in enumerate(offs):
            a = az + math.degrees(da)
            q = np.floor(((a + 45.0) % 360.0) / 90.0).astype(int)              # the quadrant each ray falls in
            rel = np.radians((a - 90.0 * q + 180.0) % 360.0 - 180.0)
            u, v = self._pixel(rel, np.tan(el + de), rows)
            col = np.clip(np.rint(u).astype(int), 0, self.resolution_px - 1)
            row = np.clip(np.rint(v).astype(int), 0, rows - 1)
            for k in range(4):
                sel = q == k
                img = np.asarray(outs[k]["lidar"], np.float32)
                out[sel, j] = img[row[sel], col[sel]]
        return out, np.arange(self.samples) * (360.0 / self.samples)

    def precision(self, r: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """1-sigma range noise: the Rev 6 table's (10 % Lambertian) at range ``r``, scaled by sqrt(0.1 / rho), within
        the Rev 7 span (and the table's largest)."""
        bands = self.spec["precision"]
        edges = np.array([b[0] for b in bands])
        sig = np.array([b[1] for b in bands])
        base = sig[np.minimum(np.searchsorted(edges, r, side="left"), len(sig) - 1)]
        lo, hi = self.spec["span"]
        return np.clip(base * np.sqrt(0.1 / np.maximum(rho, 1e-6)), lo, max(hi, float(sig.max())))

    def detection(self, r: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """The probability of detecting a return of reflectance ``rho`` at range ``r`` (module docstring, 2)."""
        r90 = self.range10 * np.power(np.maximum(rho, 1e-9) / 0.1, self.gamma)
        return _phi(Z90 + np.log(r90 / np.maximum(r, 1e-6)) * Z90 / self.spec["fall"])

    def measure(self, rays: np.ndarray, rng: np.random.Generator | None, unit: str = "") -> dict:
        """What the sensor reports from the footprint rays (B x C x K x 6, Resident.LIDAR_CHANNELS); ``rng`` None: the
        ideal sensor (the footprint's centre, no noise, every return). ``unit`` names the sensor: its beams' offsets."""
        lo, hi = self.range_m
        rng_m, ret, cos, first_m, first_kind, first_cos = (rays[..., i].astype(np.float64) for i in range(6))
        if rng is None:
            r = rng_m[..., 0]
            ok = (r >= lo) & (r <= hi)
            rho = ret[..., 0]
            return {"range": np.where(ok, r, 0.0), "reflectivity": np.where(ok, reflectivity_code(rho), 0.0),
                    "signal": np.where(ok, rho / 0.1 * (self.range10 / np.maximum(r, lo)) ** 2 * self.photons_10, 0.0),
                    "range2": np.zeros_like(r), "reflectivity2": np.zeros_like(r), "mixed": np.zeros_like(r, bool),
                    "specular": np.zeros_like(r, bool), "phantom": ok & (first_kind[..., 0] == 1)}
        _, w = self.footprint()
        # 1. candidate returns: the footprint's rays by their weight, and glass or a mirror met first near its normal
        cand_r = np.concatenate([rng_m, first_m[..., :1]], axis=-1)                         # B x C x (K + 1)
        fresnel = np.where(first_kind[..., 0] == 1, 0.9, 0.04)
        ang = np.degrees(np.arccos(np.clip(first_cos[..., 0], 0.0, 1.0)))
        spec_rho = np.where(first_kind[..., 0] > 0, fresnel * self.specular_gain * np.exp(-(ang / self.specular_cone_deg) ** 2), 0.0)
        cand_rho = np.concatenate([ret * w, spec_rho[..., None]], axis=-1)
        live = (cand_r > 0.0) & (cand_rho > 0.0)
        FAR = 1e9                                                    # no candidate (sorted last)
        order = np.argsort(np.where(live, cand_r, FAR), axis=-1)
        rs = np.take_along_axis(np.where(live, cand_r, FAR), order, -1)
        ps = np.take_along_axis(np.where(live, cand_rho, 0.0), order, -1)
        real = rs < FAR
        # merge candidates within a pulse length: a new return wherever the gap exceeds pulse_m
        gap = np.diff(rs, axis=-1) > self.pulse_m
        group = np.concatenate([np.zeros(rs.shape[:-1] + (1,), int), np.cumsum(gap, axis=-1)], axis=-1)
        onehot = (group[..., None] == np.arange(rs.shape[-1])) & real[..., None]   # B x C x candidate x return
        # a return's signal goes with rho / r^2; its range is the signal-weighted mean of its candidates'
        sig = ps / np.maximum(np.where(real, rs, 1.0), lo) ** 2
        g_sig = (onehot * sig[..., None]).sum(-2)
        g_rng = (onehot * (sig * np.where(real, rs, 0.0))[..., None]).sum(-2) / np.maximum(g_sig, 1e-30)
        g_rho = g_sig * np.maximum(g_rng, lo) ** 2                   # the return's reflectance, as the sensor infers it
        spread = (np.where(onehot, rs[..., None], -FAR).max(-2) - np.where(onehot, rs[..., None], FAR).min(-2)) > 0.10
        best = np.argsort(-g_sig, axis=-1)[..., :2]
        take = lambda a, i: np.take_along_axis(a, best[..., i:i + 1], -1)[..., 0]
        s1, s2, r1, r2, p1, p2 = take(g_sig, 0), take(g_sig, 1), take(g_rng, 0), take(g_rng, 1), take(g_rho, 0), take(g_rho, 1)
        mixed = take(spread, 0)
        # whether the first return is the glass's or the mirror's own surface (candidate K), or what a mirror shows
        K = rays.shape[2]
        spec_group = np.take_along_axis(group, np.argmax(order == K, axis=-1)[..., None], -1)[..., 0]
        specular = (cand_rho[..., K] > 0) & (cand_r[..., K] > 0) & (spec_group == best[..., 0])
        # 2. detection
        d1 = (s1 > 0) & (rng.random(s1.shape) < self.detection(r1, p1))
        d2 = (s2 > 0) & (s2 >= self.second_ratio * s1) & (rng.random(s2.shape) < self.detection(r2, p2))
        # 4. range noise, each beam's offset, the millimetre
        beam = np.random.default_rng(zlib.crc32(f"{self.model}/{unit}".encode()))
        bias = beam.normal(0.0, self.bias_m, (s1.shape[0], 1))
        r1n = np.round((r1 + bias + rng.normal(0.0, 1.0, r1.shape) * self.precision(r1, p1)) * 1000.0) / 1000.0
        r2n = np.round((r2 + bias + rng.normal(0.0, 1.0, r2.shape) * self.precision(r2, p2)) * 1000.0) / 1000.0
        ok1 = d1 & (r1 >= lo) & (r1n <= hi)
        ok2 = d2 & (r2 >= lo) & (r2n <= hi) & ok1
        # 5. channels: photons (Poisson) and the reflectance the sensor makes of them
        expect = p1 / 0.1 * (self.range10 / np.maximum(r1, lo)) ** 2 * self.photons_10
        photons = rng.poisson(np.minimum(expect, 1e12)).astype(np.float64)
        rho1 = photons / np.maximum(expect, 1e-9) * p1                # the reflectance the sensor makes of the count
        return {"range": np.where(ok1, r1n, 0.0), "reflectivity": np.where(ok1, reflectivity_code(rho1), 0.0),
                "signal": np.where(ok1, np.minimum(photons, 65535.0), 0.0), "range2": np.where(ok2, r2n, 0.0),
                "reflectivity2": np.where(ok2, reflectivity_code(p2), 0.0), "mixed": ok1 & mixed,
                "specular": ok1 & specular, "phantom": ok1 & ~specular & (first_kind[..., 0] == 1)}

    def points(self, ranges: np.ndarray) -> np.ndarray:
        """B x C ranges to points (B x C x 3, the lidar frame): each beam at its own azimuth, from its origin on the
        optics' circle (the sensor docs' equations)."""
        az = np.radians(self.beam_azimuths())
        col = np.radians(np.arange(self.samples) * (360.0 / self.samples))[None, :]
        el = np.radians(np.asarray(self.elevations_deg))[:, None]
        o = self.spec["origin_mm"] / 1000.0
        rr = np.where(ranges > 0, ranges - o, 0.0)
        x = rr * np.cos(el) * np.cos(az) + o * np.cos(col)
        y = rr * np.cos(el) * np.sin(az) + o * np.sin(col)
        z = rr * np.sin(el) * np.ones_like(az)
        return np.stack([x, y, z], -1)

    def read(self, world, actor, request, outs) -> Reading:
        rays, az = self._gather(outs)
        m = self.measure(rays, _rng(actor, request) if self.noise else None, actor.name)
        pts = self.points(m["range"])
        ok = m["range"] > 0
        pts2 = self.points(m["range2"])[m["range2"] > 0]
        data = {"ranges": m["range"].astype(np.float32), "azimuth_deg": az, "elevation_deg": np.asarray(self.elevations_deg),
                "reflectivity": m["reflectivity"].astype(np.float32), "signal": m["signal"].astype(np.float32),
                "ranges2": m["range2"].astype(np.float32), "reflectivity2": m["reflectivity2"].astype(np.float32),
                "near_ir": np.zeros_like(m["range"], np.float32), "points": pts[ok].astype(np.float32),
                "points2": pts2.astype(np.float32), "mixed": m["mixed"], "specular": m["specular"], "phantom": m["phantom"]}
        return Reading(actor.name, self.modality, request["time"], request["pose"], data, self.picture(m),
                       {"model": self.model_label, "beams": f"{len(self.elevations_deg)} x {self.samples}",
                        "range": f"{self.range_m[0]:g}-{self.range_m[1]:g} m", "returns": f"{100 * float(ok.mean()):.0f}%",
                        "second": f"{100 * float((m['range2'] > 0).mean()):.1f}%", "noise": "on" if self.noise else "off"})

    def destagger(self, image: np.ndarray) -> np.ndarray:
        """A B x C image as the sensor gives it (a column: one firing, each beam at its own azimuth offset) to one where
        each column is one azimuth (ouster-sdk's destagger): each row shifted by its offset in columns."""
        px = np.rint(self.stagger / (360.0 / self.samples)).astype(int)
        return np.stack([np.roll(row, int(s)) for row, s in zip(image, px)])

    # what a viewer can show of a scan (``display``): value, label
    VIEWS = (("cloud", "3D 점"), ("range", "깊이"), ("intensity", "강도"), ("all", "전체"))

    def picture(self, m: dict, width: int = 1024) -> np.ndarray:
        """What a viewer shows of a scan (``display``): the point cloud in 3D (``cloud_picture``), Ouster Studio's range
        or reflectivity image (``images``), or all three (the 3D view over the two images)."""
        if self.display == "cloud":
            return self.cloud_picture(m, width, round(width * 9 / 16))
        if self.display in ("range", "intensity"):
            return self.images(m, width, stretch=6, which=self.display)
        strips = self.images(m, width, stretch=2)
        cloud = self.cloud_picture(m, width, round(width * 9 / 16))
        return np.concatenate([cloud, np.full((4, width, 3), 20, np.uint8), strips], axis=0)

    def cloud_picture(self, m: dict, width: int = 1024, height: int = 576, back: float = 4.0, up: float = 6.5,
                      ceiling: float = 0.8, far: float = 12.0) -> np.ndarray:
        """The returns in 3D, as Ouster Studio's point view: seen from ``back`` m behind and ``up`` m over the sensor,
        looking ahead and down; each point coloured by its range (as the range image: near red to far blue), the second
        returns violet (colours from 0.5 to ``far`` m); what is more than ``ceiling`` m over the sensor left out (the
        ceiling would hide the room)."""
        pts = self.points(m["range"])
        ok = m["range"] > 0
        p, r = pts[ok], m["range"][ok]
        p2 = self.points(m["range2"])[m["range2"] > 0]
        keep, keep2 = p[:, 2] < ceiling, p2[:, 2] < ceiling
        p, r, p2 = p[keep], r[keep], p2[keep2]
        lo, hi = math.log(0.5), math.log(far)
        col = _ramp((np.log(np.maximum(r, 1e-3)) - lo) / (hi - lo))
        allp = np.concatenate([p, p2]) if len(p2) else p
        allc = np.concatenate([col, np.tile(np.array([[200, 90, 255]], np.uint8), (len(p2), 1))]) if len(p2) else col
        # the camera: behind and over the sensor, looking at a point ahead on the floor's side
        eye = np.array([-back, 0.0, up])
        look = np.array([2.5, 0.0, -1.5]) - eye
        f = look / np.linalg.norm(look)
        left = np.cross([0.0, 0.0, 1.0], f)
        left /= np.linalg.norm(left)
        upv = np.cross(f, left)
        rel = allp - eye
        x, y, z = rel @ f, rel @ left, rel @ upv
        img = np.zeros((height, width, 3), np.uint8)
        img[:] = (10, 13, 16)
        front = x > 0.3
        foc = width / 2.0 / math.tan(math.radians(70.0) / 2.0)
        u = np.rint(width / 2.0 - y[front] / x[front] * foc).astype(int)
        v = np.rint(height / 2.0 - z[front] / x[front] * foc).astype(int)
        d, c = x[front], allc[front]
        order = np.argsort(-d)                       # far first: near points drawn over them
        u, v, d, c = u[order], v[order], d[order], c[order]
        for du in (0, 1):
            for dv in (0, 1):
                uu, vv = u + du, v + dv
                inside = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
                img[vv[inside], uu[inside]] = c[inside]
        # the sensor: a cyan mark where it stands, and its heading
        s = eye * 0.0 - eye
        sx, sy, sz = s @ f, s @ left, s @ upv
        if sx > 0.3:
            cu, cv = int(width / 2 - sy / sx * foc), int(height / 2 - sz / sx * foc)
            from PIL import Image, ImageDraw
            im = Image.fromarray(img)
            dr = ImageDraw.Draw(im)
            dr.ellipse((cu - 6, cv - 6, cu + 6, cv + 6), outline=(63, 201, 212), width=2)
            dr.text((8, 6), f"{self.model_label}  {int(ok.sum())} points  (3D, from behind)", fill=(200, 200, 200))
            img = np.asarray(im)
        return img

    def images(self, m: dict, width: int = 1024, stretch: int = 4, which: str = "both") -> np.ndarray:
        """Ouster Studio's two images, one over the other: range (near red to far blue, black: no return) and
        reflectivity (grey), each beam a row (``stretch`` pixels tall), the columns destaggered, forward in the middle."""
        r, refl = self.destagger(m["range"]), self.destagger(m["reflectivity"])
        shift = self.samples // 2                                  # azimuth 0 (forward) to the middle, left on the left
        r = np.roll(r[:, ::-1], shift, axis=1)
        refl = np.roll(refl[:, ::-1], shift, axis=1)
        lo, hi = math.log(max(self.range_m[0], 0.3)), math.log(min(self.range_m[1], 30.0))
        a = _ramp((np.log(np.maximum(r, 1e-3)) - lo) / (hi - lo))
        a[r <= 0] = 0
        g = np.clip(np.sqrt(refl / 100.0) * 255.0, 0, 255).astype(np.uint8)
        b = np.repeat(g[..., None], 3, axis=2)
        img = a if which == "range" else b if which == "intensity" else \
            np.concatenate([a, np.zeros((2, a.shape[1], 3), np.uint8), b], axis=0)
        from PIL import Image
        rows = max(1, round(img.shape[0] * width / img.shape[1] * stretch))   # beams stretched: they are far apart
        return np.asarray(Image.fromarray(img).resize((width, rows), Image.NEAREST))
