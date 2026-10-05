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
