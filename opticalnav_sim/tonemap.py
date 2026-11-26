"""Display exposure for LDR output: robomituba's scene-global extended Reinhard.

The dataset's polar previews re-expose every frame from its own percentile, so a
window or lamp entering the view darkens everything else and a walkthrough
flickers. ``Tone`` fixes one (exposure, white) pair and applies it to every frame:

* ``scene``   - from the scene's dataset reference frames (the same frames for
                every run, so a scene always looks the same);
* ``episode`` - from the first frame rendered with this ``Tone``;
* ``fixed``   - given exposure and white;
* ``auto``    - per frame (the legacy behaviour, for comparison).

``reinhard`` and ``global_params`` follow mitsuba_converter.multimodal
(_tonemap_reinhard, compute_global_tone_params), so exported JPEGs match the
robomituba export pipeline's scene-global tonemap.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .stokes import LUMA, _srgb

EXPOSURE_PERCENTILE = 0.90  # this luminance maps to ~1.0 before roll-off
WHITE_PERCENTILE = 0.999    # this luminance sits at the roll-off knee


def reinhard(radiance: np.ndarray, exposure: float, white: float) -> np.ndarray:
    """Extended Reinhard per channel, then sRGB: uint8 HxWx3."""
    x = np.clip(np.nan_to_num(np.asarray(radiance, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
                0.0, 1e4) * float(exposure)
    w = max(float(white), 1e-6)
    y = x * (1.0 + x / (w * w)) / (1.0 + x)
    return np.clip(np.round(_srgb(y) * 255.0), 0, 255).astype(np.uint8)


def filmic(radiance: np.ndarray, exposure: float, white: float) -> np.ndarray:
    """A filmic S-curve (Narkowicz's fit of the ACES reference rendering transform) per channel, then sRGB:
    uint8 HxWx3. Against ``reinhard`` it keeps more contrast in the mid tones (shadows deeper, wood and cloth
    richer) and rolls the highlights off later and harder, so a window stays a window instead of a white
    patch. ``white`` is the exposed luminance that lands near paper white, as in reinhard."""
    x = np.clip(np.nan_to_num(np.asarray(radiance, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
                0.0, 1e4) * float(exposure)
    x = x * (1.5 / max(float(white), 1e-6))  # ``white`` lands at 1.5 on the curve, 0.93 of paper white
    y = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    return np.clip(np.round(_srgb(np.clip(y, 0.0, 1.0)) * 255.0), 0, 255).astype(np.uint8)


CURVES = {"reinhard": reinhard, "filmic": filmic}


def curve(name: str):
    """The display operator called ``name`` ('reinhard', the export pipeline's; or 'filmic')."""
    if name not in CURVES:
        raise ValueError(f"tone curve must be one of {sorted(CURVES)}")
    return CURVES[name]


LUT_SIZE = 4096
_LUTS: dict = {}


def fast(name: str, radiance: np.ndarray, exposure: float, white: float) -> np.ndarray:
    """``curve(name)(radiance, exposure, white)`` from a table: the curve is per channel, so for one (exposure, white)
    it is a function of a channel's radiance alone, sampled once at LUT_SIZE points evenly spaced in sqrt(radiance)
    (fine in the shadows, where sRGB's steps are) up to where it reaches white. Within one 8-bit step of the exact
    curve, at a quarter of its time (a 512 x 384 frame: ~1.1 ms against ~4, the viewer's cost of every frame shown).
    NaN and negative radiance are black, as in the exact curve; +inf is white (the exact curve makes it black)."""
    exact = curve(name)
    key = (name, float(exposure), float(white))
    entry = _LUTS.get(key)
    if entry is None:
        top = 1.0 / max(float(exposure), 1e-12)   # the radiance at which the output is white: found by doubling
        while top < 1e4 and int(exact(np.array([top], np.float32), exposure, white)[0]) < 255:
            top *= 2.0
        top = min(top, 1e4)
        samples = (np.linspace(0.0, 1.0, LUT_SIZE) ** 2 * top).astype(np.float32)
        entry = (exact(samples, exposure, white), np.float32(top), np.float32((LUT_SIZE - 1) / np.sqrt(top)))
        if len(_LUTS) > 64:   # exposures the page tried: keep the table small
            _LUTS.clear()
        _LUTS[key] = entry
    table, top, scale = entry
    a = np.fmax(np.asarray(radiance, dtype=np.float32), np.float32(0.0))   # NaN -> 0 too
    np.fmin(a, top, out=a)
    np.sqrt(a, out=a)
    a *= scale
    a += np.float32(0.5)
    return table[a.astype(np.int32)]


def global_params(radiance_frames, stride: int = 4) -> dict[str, float]:
    """(exposure, white) from the positive luminance of one or more linear RGB frames."""
    pool = np.concatenate([np.asarray(f, dtype=np.float32)[::stride, ::stride].reshape(-1, 3) @ LUMA
                           for f in radiance_frames])
    pool = pool[np.isfinite(pool) & (pool > 0)]
    if not pool.size:
        return {"exposure": 1.0, "white": 1.0}
    exposure = 1.0 / max(float(np.quantile(pool, EXPOSURE_PERCENTILE)), 1e-6)
    white = max(float(np.quantile(pool, WHITE_PERCENTILE)) * exposure, 1.0)
    return {"exposure": exposure, "white": white}


def scene_params(scene_dir: Path, variant: str = "base") -> dict[str, float]:
    """Scene-global params from the pack's stored dataset frames (1024 spp) of one variant,
    falling back to any variant's frames."""
    meta = json.loads((Path(scene_dir) / "scene.json").read_text())
    refs = [r for r in meta.get("reference_views", []) if r["variant"] == variant] or meta.get("reference_views", [])
    frames = [np.load(Path(scene_dir) / r["path"] / "stokes_data.npz")["s0"] for r in refs]
    if not frames:
        raise ValueError(f"{scene_dir} has no reference frames to set a scene exposure from")
    return global_params(frames)


class Tone:
    def __init__(self, mode: str = "scene", exposure: float | None = None, white: float | None = None,
                 scene_dir: Path | None = None, variant: str = "base"):
        if mode not in ("scene", "episode", "fixed", "auto"):
            raise ValueError(f"unknown exposure mode {mode!r}")
        self.mode = mode
        self.params: dict[str, float] | None = None
        if mode == "fixed":
            if exposure is None:
                raise ValueError("fixed exposure needs an exposure value")
            self.params = {"exposure": float(exposure), "white": float(white if white is not None else 1e4)}
        elif mode == "scene":
            self.params = scene_params(scene_dir, variant)

    @classmethod
    def parse(cls, spec: str, scene_dir: Path | None = None, variant: str = "base") -> "Tone":
        """'scene' | 'episode' | 'auto' | 'fixed:<exposure>[,<white>]'."""
        mode, _, rest = spec.partition(":")
        if mode == "fixed":
            values = [float(v) for v in rest.split(",") if v]
            return cls("fixed", *values[:2])
        return cls(mode, scene_dir=scene_dir, variant=variant)

    def __call__(self, radiance: np.ndarray) -> np.ndarray:
        if self.mode == "auto":
            p = global_params([radiance])
        else:
            if self.params is None:  # episode: the first frame decides
                self.params = global_params([radiance])
            p = self.params
        return reinhard(radiance, p["exposure"], p["white"])

    def describe(self) -> dict:
        return {"operator": "reinhard_ext", "scope": self.mode, "exposure_percentile": EXPOSURE_PERCENTILE,
                "white_percentile": WHITE_PERCENTILE, **({} if self.params is None else self.params)}
