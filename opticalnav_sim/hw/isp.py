"""A software image signal processor: a raw frame (hw.sensor.Raw, or a real camera's) to the image a camera delivers.

    image = isp.process(raw)                                  # uint8 RGB, defaults
    image = isp.process(raw, isp.Isp(white_balance="gray_world", tone="reinhard"))
    pol = isp.polarization(raw)                               # a polarization sensor: s0, s1, s2, dolp, aolp

Stages, each of which can be changed or left out: black level, lens shading, demosaic, white balance, colour matrix,
exposure, tone curve, 8 bits. They work on any raw with a Bayer mosaic, so a recorded frame goes through the same code
as a simulated one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import sensor as sensor_model
from .sensor import BAYER, Raw


@dataclass
class Isp:
    demosaic: str = "malvar"            # "malvar" (Malvar-He-Cutler, 5x5) or "bilinear"
    white_balance: object = (1.0, 1.0, 1.0)   # gains for R, G, B, or "gray_world"
    ccm: np.ndarray = field(default_factory=lambda: np.eye(3))   # camera RGB -> output RGB (linear)
    shading: float = 0.0                # divide the lens's fall-off out: its power (4: cos^4), 0 leaves it in
    exposure: object = "auto"           # a factor on the linear image, or "auto" (srgb: the 99th percentile to
    #                                     ``white_point``; reinhard: the log-average luminance to middle grey)
    white_point: float = 0.9
    clip_highlights: bool = True        # a pixel at saturation stays neutral after white balance (no magenta skies)
    tone: str = "srgb"                  # "srgb" (the sRGB curve, clipped), "reinhard" (highlights rolled off), "linear"


def _filter(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Correlation with a small kernel, the edge mirrored (so a mosaic's phase carries over the border)."""
    r = kernel.shape[0] // 2
    pad = np.pad(img, r, mode="reflect")
    out = np.zeros_like(img, dtype=np.float64)
    h, w = img.shape
    for dy in range(kernel.shape[0]):
        for dx in range(kernel.shape[1]):
            if kernel[dy, dx]:
                out += kernel[dy, dx] * pad[dy:dy + h, dx:dx + w]
    return out


# Malvar, He and Cutler, "High-quality linear interpolation for demosaicing of Bayer-patterned color images" (2004)
_G_AT_RB = np.array([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]]) / 8.0
_RB_AT_G_ROW = np.array([[0, 0, 0.5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1], [0, -1, 0, -1, 0], [0, 0, 0.5, 0, 0]]) / 8.0
_RB_AT_G_COL = _RB_AT_G_ROW.T
_RB_AT_BR = np.array([[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5], [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]]) / 8.0
_BILINEAR_G = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 4.0
_BILINEAR_RB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4.0


def demosaic(values: np.ndarray, cfa: str = "RGGB", method: str = "malvar") -> np.ndarray:
    """A Bayer mosaic (H x W, linear) to H x W x 3."""
    if cfa == "mono":
        return np.repeat(np.asarray(values, dtype=np.float64)[..., None], 3, axis=2)
    v = np.asarray(values, dtype=np.float64)
    h, w = v.shape
    r, c = np.indices((h, w))
    colour = np.array(BAYER[cfa])[r % 2, c % 2]
    red, green, blue = colour == 0, colour == 1, colour == 2
    out = np.zeros((h, w, 3))
    if method == "bilinear":
        out[..., 0] = _filter(np.where(red, v, 0.0), _BILINEAR_RB)
        out[..., 1] = _filter(np.where(green, v, 0.0), _BILINEAR_G)
        out[..., 2] = _filter(np.where(blue, v, 0.0), _BILINEAR_RB)
        return out
    if method != "malvar":
        raise ValueError(f"demosaic method {method!r}: malvar or bilinear")
    # a green pixel's row holds red or blue: which, by the colour of its row neighbours
    red_row = np.zeros((h, w), bool)
    red_row[np.any(red, axis=1)] = True
    g_at_rb, rb_row, rb_col, rb_at_br = (_filter(v, k) for k in (_G_AT_RB, _RB_AT_G_ROW, _RB_AT_G_COL, _RB_AT_BR))
    out[..., 1] = np.where(green, v, g_at_rb)
    out[..., 0] = np.where(red, v, np.where(blue, rb_at_br, np.where(red_row, rb_row, rb_col)))
    out[..., 2] = np.where(blue, v, np.where(red, rb_at_br, np.where(red_row, rb_col, rb_row)))
    return out


def srgb(linear: np.ndarray) -> np.ndarray:
    a = np.clip(linear, 0.0, 1.0)
    return np.where(a <= 0.0031308, 12.92 * a, 1.055 * np.power(a, 1.0 / 2.4) - 0.055)


def linear_rgb(raw: Raw, config: Isp | None = None) -> np.ndarray:
    """The raw as linear RGB (H x W x 3; 1.0 is the sensor's saturation before white balance): black level, shading,
    demosaic, white balance, colour matrix. A polarization sensor gives its unpolarized image (half size)."""
    cfg = config or Isp()
    v = raw.linear()
    if cfg.shading:
        if raw.intrinsics is None:
            raise ValueError("lens shading correction needs the raw's intrinsics")
        v = v / sensor_model.vignetting(raw.intrinsics, float(cfg.shading))
    if raw.cfa.startswith("polar_"):
        rgb = polarization(raw, cfg, _linear=v)["s0"]
    else:
        rgb = demosaic(v, raw.cfa, cfg.demosaic)
    if isinstance(cfg.white_balance, str):
        if cfg.white_balance != "gray_world":
            raise ValueError("white_balance: gains for R, G, B, or 'gray_world'")
        flat = rgb.reshape(-1, 3)
        seen = flat[flat.max(1) < 0.95]  # a saturated pixel says nothing about the light's colour
        means = (seen if len(seen) > 100 else flat).mean(0)
        gains = means[1] / np.maximum(means, 1e-12)
    else:
        gains = np.asarray(cfg.white_balance, dtype=np.float64)
    rgb = rgb * gains
    if cfg.clip_highlights:  # saturation is 1.0 in every channel before the gains: the smallest gain after them
        rgb = np.minimum(rgb, float(np.min(gains)))
    return rgb @ np.asarray(cfg.ccm, dtype=np.float64).T


def process(raw: Raw, config: Isp | None = None) -> np.ndarray:
    """The 8-bit RGB image of a raw frame."""
    cfg = config or Isp()
    rgb = linear_rgb(raw, cfg)
    if cfg.tone not in ("srgb", "reinhard", "linear"):
        raise ValueError(f"tone {cfg.tone!r}: srgb, reinhard or linear")
    if cfg.tone == "reinhard":
        # Reinhard's global operator: the scene's log-average luminance to middle grey ("auto"), then highlights
        # rolled off so that the brightest part of the image, not the sensor's range, is white
        lum = np.maximum(rgb @ np.array([0.2126, 0.7152, 0.0722]), 1e-9)
        scale = 0.18 / float(np.exp(np.mean(np.log(lum)))) if cfg.exposure == "auto" else float(cfg.exposure)
        rgb = rgb * scale
        white = max(1.0, float(np.percentile(lum * scale, 99.9)))
        rgb = rgb * (1.0 + rgb / (white * white)) / (1.0 + rgb)
    elif cfg.exposure == "auto":
        rgb = rgb * (cfg.white_point / max(float(np.percentile(rgb, 99.0)), 1e-12))
    else:
        rgb = rgb * float(cfg.exposure)
    out = np.clip(rgb, 0.0, 1.0) if cfg.tone == "linear" else srgb(rgb)
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


def polarization(raw: Raw, config: Isp | None = None, _linear: np.ndarray | None = None) -> dict:
    """A polarization sensor's raw to Stokes images at half its size: every polarizer angle's pixels form a Bayer
    mosaic of their own (the colours come in 2x2 blocks), demosaiced separately. Returns ``i`` (angle -> H/2 x W/2 x
    3), ``s0``, ``s1``, ``s2`` (linear, in the sensor's frame: x along the image's rows, angles towards its up),
    ``dolp`` and ``aolp_deg`` (0..180)."""
    if not raw.cfa.startswith("polar_"):
        raise ValueError(f"not a polarization sensor's raw ({raw.cfa})")
    cfg = config or Isp()
    v = raw.linear() if _linear is None else _linear
    angles = np.array(raw.polarizer_deg, dtype=np.float64).reshape(2, 2)
    images = {float(angles[i, j]): demosaic(v[i::2, j::2], raw.cfa[6:], cfg.demosaic) for i in (0, 1) for j in (0, 1)}
    # least squares for (S0, S1, S2) from I(t) = (S0 + S1 cos 2t + S2 sin 2t) / 2: any four angles, not only 0/45/90/135
    t = np.radians(sorted(images))
    design = 0.5 * np.stack([np.ones_like(t), np.cos(2.0 * t), np.sin(2.0 * t)], 1)
    stack = np.stack([images[a] for a in sorted(images)], -1)
    s = stack @ np.linalg.pinv(design).T
    s0, s1, s2 = s[..., 0], s[..., 1], s[..., 2]
    return {"i": images, "s0": s0, "s1": s1, "s2": s2,
            "dolp": np.sqrt(s1 * s1 + s2 * s2) / np.maximum(s0, 1e-12),
            "aolp_deg": np.degrees(0.5 * np.arctan2(s2, s1)) % 180.0}
