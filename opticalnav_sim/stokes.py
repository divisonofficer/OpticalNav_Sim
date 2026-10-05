"""Stokes post-processing, matching the OpticalNav dataset writer.

Ported from mitsuba_converter.multimodal (save_polarization_products,
_rgb_preview_array, _fill_invalid_preview_pixels, _despeckle_dark_preview_pixels)
so a simulator frame and a dataset frame go through the same arithmetic:

* ``s0..s3``: linear RGB Stokes components, float32 HxWx3, in the dataset's
  ``world_gravity_y_v1`` basis (reference axis = world up).
* ``rgb``: the ``polar_rgb_preview`` image (per-frame 99.2th-percentile
  exposure, sRGB, despeckle, 0.45 px blur), uint8 HxWx3 RGB.
"""
from __future__ import annotations

import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def split_channels(image: np.ndarray) -> tuple[np.ndarray, ...]:
    """Mitsuba Stokes film (15 or 16 channels) -> rgb, s0, s1, s2, s3."""
    off = {15: 3, 16: 4}[image.shape[2]]
    rgb = np.asarray(image[:, :, 0:3], dtype=np.float32)
    return (rgb,) + tuple(np.asarray(image[:, :, off + 3 * k:off + 3 * k + 3], dtype=np.float32) for k in range(4))


def _srgb(a: np.ndarray) -> np.ndarray:
    a = np.clip(a, 0.0, None)
    return np.where(a <= 0.0031308, 12.92 * a, 1.055 * np.power(a, 1.0 / 2.4) - 0.055)


def _fill_invalid(img: np.ndarray, valid: np.ndarray, fallback: np.ndarray, iterations: int) -> np.ndarray:
    out = img.astype(np.float32).copy()
    if valid.all():
        return np.clip(out, 0.0, 1.0)
    h, w = valid.shape
    out[~valid] = fallback[~valid]
    known = valid.copy()
    for _ in range(iterations):
        if known.all():
            break
        pad = np.pad(out, ((1, 1), (1, 1), (0, 0)), mode="edge")
        kpad = np.pad(known, 1, mode="constant", constant_values=False)
        acc = np.zeros_like(out)
        cnt = np.zeros((h, w), np.float32)
        for dy in range(3):
            for dx in range(3):
                if dx == 1 and dy == 1:
                    continue
                k = kpad[dy:dy + h, dx:dx + w]
                acc += pad[dy:dy + h, dx:dx + w] * k[:, :, None]
                cnt += k
        fill = ~known & (cnt > 0)
        if not fill.any():
            break
        out[fill] = acc[fill] / cnt[fill, None]
        known[fill] = True
    return np.clip(out, 0.0, 1.0)


def _despeckle(img: np.ndarray, iterations: int = 3) -> np.ndarray:
    out = np.clip(img.astype(np.float32), 0.0, 1.0)
    h, w = out.shape[:2]
    for _ in range(iterations):
        lum = out @ LUMA
        pad = np.pad(out, ((1, 1), (1, 1), (0, 0)), mode="edge")
        lpad = np.pad(lum, 1, mode="edge")
        acc = np.zeros_like(out)
        lacc = np.zeros((h, w), np.float32)
        for dy in range(3):
            for dx in range(3):
                if dx == 1 and dy == 1:
                    continue
                acc += pad[dy:dy + h, dx:dx + w]
                lacc += lpad[dy:dy + h, dx:dx + w]
        mean_lum = lacc / 8.0
        speckle = (lum < 0.08) & (mean_lum > lum + 0.10) & (mean_lum > 0.12)
        if not speckle.any():
            break
        out[speckle] = (acc / 8.0)[speckle]
    return np.clip(out, 0.0, 1.0)


def rgb_preview(rgb: np.ndarray, s0: np.ndarray, s1: np.ndarray, s2: np.ndarray,
                percentile: float = 0.992, blur: float = 0.45) -> np.ndarray:
    """Defaults are the polar preview recipe; production's plain RGB camera uses 0.995 and no blur."""
    from PIL import Image, ImageFilter

    s0_l = s0 @ LUMA
    finite = np.isfinite(s0_l) & np.isfinite(s1 @ LUMA) & np.isfinite(s2 @ LUMA)
    pos = s0_l[finite & (s0_l > 0)]
    ctx_scale = max(float(np.quantile(pos, 0.995)) if pos.size else 1.0, 1e-6)
    ctx = np.clip(np.sqrt(np.clip(np.where(finite, s0_l, 0.0), 0.0, None) / ctx_scale), 0.0, 1.0)
    context = np.repeat((0.12 + 0.88 * ctx)[:, :, None], 3, axis=2).astype(np.float32)
    context[~finite] = 0.18

    safe = np.where(np.isfinite(rgb), rgb, 0.0)
    positive = safe[safe > 0]
    scale = max(float(np.quantile(positive, percentile)) if positive.size else 1.0, 1e-6)
    preview = np.clip(_srgb(safe / scale), 0.0, 1.0).astype(np.float32)
    preview = _fill_invalid(preview, np.all(np.isfinite(rgb), axis=2), context, iterations=8)
    preview = _despeckle(preview, iterations=3)
    u8 = np.clip(np.round(preview * 255.0), 0, 255).astype(np.uint8)
    if blur <= 0.0:
        return u8
    return np.asarray(Image.fromarray(u8, mode="RGB").filter(ImageFilter.GaussianBlur(radius=blur)))


def quick_preview(radiance: np.ndarray, percentile: float = 0.992) -> np.ndarray:
    """Plain display tonemap of linear RGB (exposure from a percentile, then sRGB), uint8 RGB.
    No despeckle, hole filling or blur, so it costs a few ms instead of the preview recipe's ~40 ms."""
    safe = np.nan_to_num(np.asarray(radiance, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    sample = safe[::4, ::4]
    positive = sample[sample > 0]
    scale = max(float(np.quantile(positive, percentile)) if positive.size else 1.0, 1e-6)
    return np.clip(np.round(_srgb(safe / scale) * 255.0), 0, 255).astype(np.uint8)


def dolp_rgb(dolp: np.ndarray, scale: float = 0.2) -> np.ndarray:
    """DoLP as uint8 RGB from black (0) to full red (``scale`` and above)."""
    t = np.clip(np.nan_to_num(np.asarray(dolp, dtype=np.float32)) / scale, 0.0, 1.0)
    out = np.zeros(t.shape + (3,), np.uint8)
    out[..., 0] = np.round(255.0 * t)
    return out


def aolp_rgb(aolp_deg: np.ndarray, dolp: np.ndarray, scale: float = 0.2) -> np.ndarray:
    """AoLP as hue around the whole colour circle (AoLP repeats every 180 degrees, so 0 and 180 meet in
    red, 90 is cyan), brightness DoLP / ``scale``. Unpolarised light is black; no image underneath."""
    h = np.mod(np.nan_to_num(np.asarray(aolp_deg, dtype=np.float32)), 180.0) / 180.0
    v = np.clip(np.nan_to_num(np.asarray(dolp, dtype=np.float32)) / scale, 0.0, 1.0)
    k = (np.array([5.0, 3.0, 1.0], np.float32) + h[..., None] * 6.0) % 6.0  # HSV -> RGB at full saturation
    rgb = v[..., None] * (1.0 - np.clip(np.minimum(k, 4.0 - k), 0.0, 1.0))
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def derived(s0: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> dict[str, np.ndarray]:
    """Luminance DoLP, AoLP (degrees, [0, 180)) and normalised S1/S0, S2/S0, as the dataset defines them."""
    l0, l1, l2 = s0 @ LUMA, s1 @ LUMA, s2 @ LUMA
    safe0 = np.maximum(l0, 1e-8)
    return {
        "dolp": np.clip(np.sqrt(np.maximum(0.0, l1 * l1 + l2 * l2)) / safe0, 0.0, 1.0),
        "aolp_deg": np.degrees(np.mod(0.5 * np.arctan2(l2, l1), np.pi)),
        "s1_over_s0": np.clip(np.where(np.isfinite(l1), l1, 0.0) / safe0, -1.0, 1.0),
        "s2_over_s0": np.clip(np.where(np.isfinite(l2), l2, 0.0) / safe0, -1.0, 1.0),
    }
