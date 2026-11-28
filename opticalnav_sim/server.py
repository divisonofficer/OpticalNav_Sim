"""OpticalNav render server: resident Mitsuba scenes, polar or RGB renders on request.

    python -m opticalnav_sim.server --pack <pack_dir> [--port 18770] [--preload SCAN[:VARIANT[:MODE]]]

Mitsuba must be importable (PYTHONPATH of a build with an ``*_rgb_polarized``
variant; an ``*_rgb`` variant enables the faster RGB mode). Endpoints (JSON unless noted):

    GET  /v1/info                       server, renderers, variants, resident scenes
    GET  /v1/scans                      scan ids
    GET  /v1/scans/<scan>/meta          rig, intrinsics, variants, conventions
    GET  /v1/scans/<scan>/connectivity  Matterport3D connectivity JSON
    POST /v1/render                     {"views": [...]} -> NPZ (application/octet-stream)
    POST /v1/denoise                    NPZ (s0, s1, s2 or radiance) -> the same, OptiX-denoised

All Mitsuba calls run on the main thread; HTTP requests queue render jobs for it.

A view is ``{"scan", "variant", "camera_to_world" (4x4, dataset convention),
"width", "height", "hfov_deg", "spp", "seed", "mode" ("polar" | "rgb"),
"denoise" (bool)}``. The NPZ holds ``rgb_<i>`` (uint8 HxWx3 preview) plus
``s0_<i>..s3_<i>`` (polar mode) or ``radiance_<i>`` (rgb mode), float16 or
float32 HxWx3. The ``X-Render-Timing`` header gives render / denoise / post seconds.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hmac
import io
import json
import math
import os
import queue
import re
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter, OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from . import bodies as body_models, frames, progress, solid_filters, stokes

MAX_PIXELS = 2048 * 2048


class Activity:
    """What the render loop is doing now, readable from HTTP threads while it works (GET /v1/status)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.what, self.since = "", None

    @contextlib.contextmanager
    def doing(self, what: str):
        with self.lock:
            prev = (self.what, self.since)
            self.what, self.since = what, time.monotonic()
        try:
            yield
        finally:
            with self.lock:
                self.what, self.since = prev

    def rename(self, what: str) -> None:
        """The same wait, now at another step (its elapsed time starts again)."""
        with self.lock:
            self.what, self.since = what, time.monotonic()

    def snapshot(self) -> dict:
        with self.lock:
            out = {"what": self.what, "elapsed_s": round(time.monotonic() - self.since, 1) if self.since else 0.0}
        # how far a scene load or a freeze recording has got (see progress.py)
        found = progress.FREEZE.snapshot() or progress.LOAD.snapshot()
        if found:
            out["progress"] = found
        return out


ACTIVITY = Activity()
MAX_SPP = 8192
MAX_VIEWS = 64
MAX_BODY = 1 << 20
MAX_DENOISE_BODY = 256 << 20  # a few 2048x2048 float32 Stokes images
# POST /v1/render with "format": "raw" answers with this: 8 bytes little-endian header length, a JSON header
# [[name, numpy dtype str, shape, offset, nbytes], ...], then the arrays' bytes at those offsets
# path_nocaustics' specular_pdf_threshold (sr^-1) for the flash pass: a bounce whose BSDF pdf exceeds it counts as
# near-specular, and light reaching the flash through one after a rough bounce is left out (no caustics). 20 (its
# default, GGX alpha below ~0.06) is the dataset's; lower values also leave out glossy surfaces (alpha up to ~0.2
# at 2), whose diffuse -> glossy -> small flash paths were the bright outliers left with the flash on.
FLASH_PDF = 20.0
RAW_ARRAYS_TYPE = "application/x-opticalnav-arrays"
# polar: cuda_rgbn_polarized first (R, G, B and near-infrared lanes in one path, each with its own Mueller
# matrix: both bands in one render); --variant cuda_rgb_polarized keeps the dual-sensor render instead
MODE_VARIANTS = {"polar": ("cuda_rgbn_polarized", "cuda_rgb_polarized", "cuda_ad_rgb_polarized"),
                 "rgb": ("cuda_rgbn", "cuda_rgb", "cuda_ad_rgb")}
# rgb: cuda_rgbn first (R, G, B and near-infrared lanes, no Mueller matrices: the stokes integrator writes S0 per lane and
# zero S1-S3, the polarized RGBN layout), else cuda_rgb, which renders RGB only


def mode_bands(variant: str, nir_plugin: bool) -> list:
    """The bands a mode's variant renders: an RGBN variant both lanes in one path, a polarized RGB one the near-infrared
    band through its second sensor; plain cuda_rgb RGB only (its band switch failed in the frozen recording)."""
    return ["rgb", "nir"] if nir_plugin and ("_rgbn" in variant or "polarized" in variant) else ["rgb"]
# Without dr.freeze, high spp is rendered in passes of at most this many samples (wavefront memory);
# a polarized path carries Mueller matrices, so its passes are smaller.
# What a frozen render takes as input. "scene": the whole scene, so Dr.Jit walks every texture and mesh
# on each replay (~23 ms on Device 1). "sensor": only the camera, with the static scene captured when the
# function is recorded; a replay then walks the sensor alone. Flash scenes always use "scene" (the flash
# moves with the camera).
FREEZE_INPUT = os.environ.get("OPTICALNAV_SIM_FREEZE_INPUT", "scene")
SPP_CHUNK = {"polar": max(1, int(os.environ.get("OPTICALNAV_SIM_SPP_CHUNK", "64"))),
             "rgb": max(1, int(os.environ.get("OPTICALNAV_SIM_SPP_CHUNK_RGB", "256")))}


# what a recording of each pass is, for the progress the viewer shows ("step k of n: ...")
PASS_PHASE = {"active": "lights + flash", "passive": "lights only", "own": "render", "flash": "flash only", "split": "lights + flash"}


def despeckle(img: np.ndarray, ratio: float = 4.0) -> np.ndarray:
    """A pixel ``ratio`` times brighter than the brightest of its eight neighbours (a firefly: one path that found a
    small bright light through a glossy bounce) scaled down to that neighbour. A single pass has them on every wall;
    the OptiX denoiser keeps them as detail. Edges and highlights are left alone: a neighbour is about as bright."""
    lum = img.mean(axis=-1) if img.ndim == 3 else img
    pad = np.pad(lum, 1, mode="edge")
    h, w = lum.shape
    near = np.zeros_like(lum)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if dy == 1 and dx == 1:
                continue
            np.maximum(near, pad[dy:dy + h, dx:dx + w], out=near)
    hot = lum > ratio * near + 1e-6
    if not hot.any():
        return img
    out = img.copy()
    scale = np.where(hot, near / np.maximum(lum, 1e-12), 1.0)
    return out * (scale[..., None] if img.ndim == 3 else scale)


def trim_host_memory() -> None:
    """Hand freed host memory back to the system. Compiling a render's kernels leaves gigabytes in the C library's
    arenas (a 75-mesh outdoor scene: 16.4 GB resident after its first frame, 3.3 GB after a trim), and the next scene's
    compilation came on top: the kernel killed the server at 20.7 GB on a 24 GB machine."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):   # not glibc
        pass


def rgb_scene_text(text: str) -> str:
    """Production Stokes scene -> plain radiance scene: its nested integrator replaces the stokes wrapper."""
    root = ET.fromstring(text)
    integrator = root.find("./integrator")
    if integrator is not None and integrator.get("type") == "stokes":
        nested = integrator.find("./integrator")
        index = list(root).index(integrator)
        root.remove(integrator)
        root.insert(index, nested)
    return ET.tostring(root, encoding="unicode")


ASSIST_PREFIX = "camera_assist"  # the flash rectangle (and the old protocol's polarizer) in front of the camera
# The room softboxes robomituba's importer adds for even fill (light_groups: "fill"): path_nocaustics's ``hidden``
# lets camera rays through them, so they light the room without white squares under every ceiling (Blender's
# camera visibility off, as the dataset meant them).
FILL_PREFIX = "light_softbox_"
# The lights of a scene converted from its .blend (tools/blend_to_pack.py) whose Blender Ray Visibility > Camera is off (all of
# PH19's and PH33's): the same camera-transparent lights, under their own prefix, so they stay "fixtures", not "fill".
HIDDEN_LIGHT_PREFIX = "light_hidden_"
# a Blender light that is off for the camera and also for glossy and transmission rays (path_nocaustics's
# ``hidden_specular``): only diffuse bounces see it, a window pane or a mirror does not show it
HIDDEN_SPEC_PREFIX = "light_hidden_spec_"
ASSIST_SHAPE = re.compile(r'\s*<shape type="rectangle" id="camera_assist_[a-z]+">.*?</shape>', re.S)
SENSOR_BLOCK = re.compile(r"\s*<sensor\b.*?</sensor>", re.S)
COMMENT = re.compile(r"\s*<!--.*?-->", re.S)
# The film's reconstruction filter. The scenes' hdrfilm names none: Mitsuba's default Gaussian spreads each sample over
# its neighbours, and noise came out correlated pixel to pixel (0.37 at one pixel's distance). The OptiX denoiser takes
# that for texture once the noise is small: 16 spp denoised well, 64 spp hardly, 256 spp not at all. A box filter keeps
# each sample in its pixel. OPTICALNAV_SIM_RFILTER=gaussian renders as the dataset's exports did.
RFILTER = os.environ.get("OPTICALNAV_SIM_RFILTER", "box")
FILM_OPEN = re.compile(r'(<film type="hdrfilm"[^>]*>)(.*?)(</film>)', re.S)


def film_filter(text: str) -> str:
    """The scene text with RFILTER as every hdrfilm's reconstruction filter (a film naming its own keeps it)."""
    if RFILTER == "gaussian":
        return text
    return FILM_OPEN.sub(lambda m: m.group(0) if "<rfilter" in m.group(2)
                         else f'{m.group(1)}{m.group(2)}<rfilter type="{RFILTER}" />{m.group(3)}', text)
TOP_ELEMENT = re.compile(r'\n  <(shape|bsdf|texture)\b([^>]*?)(?:/>|>.*?\n  </\1>)', re.S)


def extra_shapes(host: str, variant: str) -> tuple[str, ...] | None:
    """Ids of the shapes a host scene file adds to a variant's, when that is all that differs (perturbed is
    base plus mirror and glass cubes; the sensor, comments and camera-assist shapes aside). None when
    the files differ in anything else."""
    def norm(text: str) -> str:
        return SENSOR_BLOCK.sub("", ASSIST_SHAPE.sub("", COMMENT.sub("", text)))

    h, v = norm(host), norm(variant)
    have = {m.group(0) for m in TOP_ELEMENT.finditer(v)}
    added = [m for m in TOP_ELEMENT.finditer(h) if m.group(0) not in have]
    for m in added:
        h = h.replace(m.group(0), "", 1)
    if h.split() != v.split():
        return None
    ids = []
    for m in added:
        if m.group(1) == "shape":
            found = re.search(r'id="([^"]+)"', m.group(2))
            if found is None:  # an unnamed shape cannot be looked up to hide
                return None
            ids.append(found.group(1))
    return tuple(ids)


def upgrade_flash_text(text: str, polarizer_angle_deg: float) -> tuple[str, bool]:
    """An old one-pass active scene (rgb_stokes_12_active_polar_compact_flash_v2) with the flash of the current
    protocol (active_polar_flash_nocaustics_v1): the flash rectangle's ``area`` emitter becomes ``polarized_area``
    (linearly polarized at ``polarizer_angle_deg``, radiance = S0), and the ``polarizer`` rectangle goes. That
    rectangle sat 1 cm behind the light, on the side it never emits, so the old flash was unpolarized (DoLP of
    its reflection in a mirror ~0, PC_OPTICALNAV_RENDER). Rendered as two passes (a flash pass without caustics),
    the light's fireflies drop as in the current protocol (4236 -> 29 on its worst view). Returns (text, changed)."""
    light = re.search(rf'\n  <shape type="[a-z_]+" id="{ASSIST_PREFIX}_light">.*?\n  </shape>', text, re.S)
    if light is None or '<emitter type="area">' not in light.group(0):
        return text, False
    shape = light.group(0).replace('<emitter type="area">', f'<emitter type="polarized_area">\n      '
                                   f'<float name="theta" value="{float(polarizer_angle_deg):.4f}" />', 1)
    text = text[:light.start()] + shape + text[light.end():]
    text = re.sub(rf'\n  <shape type="[a-z_]+" id="{ASSIST_PREFIX}_polarizer">.*?\n  </shape>', "", text, flags=re.S)
    return text, True


# the dataset's camera flash (active_polar_flash_nocaustics_v1): what a scene packed without an active variant gets
SYNTHETIC_FLASH = {"distance_m": 0.2, "size_world": [0.12, 0.08], "radiance": 2500.0, "emitter": "polarized_area",
                   "polarizer_angle_deg": 0.0, "synthetic": True}
SYNTHETIC_POOL_SHARE = (1.0, 0, False)  # a synthetic perturbed variant's mirrors: robomituba's density 1, no glass


OWN_PROTOCOLS = ("staged_without_render_v1", "blend_direct_v1")  # the simulator's own scenes (Renderer.own_scene)


def synthesize_variants(meta: dict, has_pool: bool, has_nocaustics: bool) -> list[str]:
    """Give a scene packed with its base variant alone the variants the viewer switches among, on the same scene
    file (so on the same resident scene): "perturbed" shows its perturbation pool's share at density 1
    (needs tools/build_perturbation_pool.py), "active_polar" adds the dataset's camera flash (SYNTHETIC_FLASH,
    two passes: needs path_nocaustics) on top of that. Marked ``"synthetic"``; returns the names added."""
    variants = meta.get("variants") or {}
    base = variants.get("base")
    if base is None:
        return []
    added = []
    if "perturbed" not in variants and has_pool:
        variants["perturbed"] = {"xml": base["xml"], "synthetic": "perturbation pool",
                                 "pool_share": list(SYNTHETIC_POOL_SHARE)}
        added.append("perturbed")
    if not any(v.get("flash") for v in variants.values()) and has_nocaustics and "active_polar" not in variants:
        lit = variants.get("perturbed", base)  # the active variant is the perturbed one plus the flash
        variants["active_polar"] = {"xml": lit["xml"], "synthetic": "camera flash", "flash": dict(SYNTHETIC_FLASH),
                                    "protocol": "active_polar_flash_nocaustics_v1", "passes": 2,
                                    "pool_share": lit.get("pool_share")}
        added.append("active_polar")
    if added:
        meta["variants_synthetic"] = added
    return added


def flash_scene_text(text: str, flash: dict) -> str:
    """``text`` with the camera flash rectangle (as an active_polar_flash.xml carries it, placed by every frame)
    before ``</scene>``; the passive pass's emitter sampling barely picks it, as in lit_scene_text."""
    half = [float(v) * 0.5 for v in flash["size_world"]]
    r = float(flash["radiance"])
    shape = (f'\n  <shape type="rectangle" id="{ASSIST_PREFIX}_light">\n    <transform name="to_world">\n      <matrix value="'
             f'{half[0]} 0 0 0 0 {half[1]} 0 0 0 0 -1 {float(flash["distance_m"])} 0 0 0 1" />\n    </transform>\n'
             f'    <bsdf type="null" />\n    <emitter type="{flash.get("emitter") or "polarized_area"}">'
             f'<float name="sampling_weight" value="0.001" />\n      <rgb name="radiance" value="{r},{r},{r}" />\n'
             f'      <float name="theta" value="{float(flash.get("polarizer_angle_deg") or 0.0):.4f}" />\n    </emitter>\n  </shape>')
    end = text.rindex("</scene>")
    return text[:end] + shape + "\n" + text[end:]


def lit_scene_text(xml: Path, flash_xml: Path) -> str:
    """Passive scene plus the camera-assist flash of its flash-only scene, so one resident scene renders
    both passes of a two-pass active variant. The flash is sampled directly by the flash pass, so the
    passive pass's emitter selection barely picks it (sampling_weight)."""
    flash = "".join(m.group(0) for m in ASSIST_SHAPE.finditer(flash_xml.read_text()))
    if not flash:
        raise RuntimeError(f"{flash_xml}: no {ASSIST_PREFIX}_* shape")
    flash = re.sub(r'(<emitter type="[a-z_]+">)', r'\1<float name="sampling_weight" value="0.001" />', flash)
    text = xml.read_text()
    end = text.rindex("</scene>")
    return text[:end] + flash + "\n" + text[end:]


NIR_BAND_ID = "nir_band"
NIR_SLOTS = ("diffuse_reflectance", "specular_reflectance")  # base colours (pplastic, roughconductor)
# robomituba's NIR_PASSIVE_SCALE_POLICY (tools/infinigen/adapt_stage2_for_hssd_renderer.py): near-infrared
# radiance of a light class relative to its visible radiance (a light's colour collapses to its luminance)
NIR_DEFAULTS = {"led": 0.0, "incandescent": 0.45, "daylight": 0.75}
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def nir_scene_text(text: str) -> str:
    """Wrap every base-colour slot in an ``rgb2nir`` texture switched by one shared ``nir_band`` value, so
    the loaded scene renders the visible or the pseudo near-infrared band (robomituba's pseudo_nir_albedo,
    computed per lookup: no second set of textures)."""
    slots = "|".join(NIR_SLOTS)
    # the shared switch, and the shared range the pseudo near-infrared reflectance is mapped onto (Look.nir_range)
    band = f'<ref name="band" id="{NIR_BAND_ID}" /><ref name="low" id="nir_low" /><ref name="high" id="nir_high" />'
    text = re.sub(rf'<texture name="({slots})" type="([a-z_]+)">(.*?)</texture>',
                  lambda m: (f'<texture name="{m.group(1)}" type="rgb2nir"><texture name="nested" type="{m.group(2)}">'
                             f'{m.group(3)}</texture>{band}</texture>'), text, flags=re.S)
    text = re.sub(rf'<rgb name="({slots})" value="([^"]*)"\s*/>',
                  lambda m: (f'<texture name="{m.group(1)}" type="rgb2nir"><rgb name="nested" value="{m.group(2)}" />'
                             f'{band}</texture>'), text)
    opening = re.search(r"<scene\b[^>]*>", text)
    declare = (f'\n  <texture type="uniform" id="{NIR_BAND_ID}"><float name="value" value="{uniform_tag(NIR_BAND_ID)}" />'
               f'</texture>')
    return text[:opening.end()] + declare + text[opening.end():]


LIGHTING = {"none": (), "fixtures": ("fixture",), "all": ("fixture", "fill")}  # light groups kept per mode
LIGHT_GROUPS = ("fixture", "fill", "lamp")  # lamp: lights added to lamps the scene left dark (lamp_lights_text)


BAND_UNIFORMS = ("light_rgb_fixture", "light_rgb_fill", "light_nir_fixture_led", "light_nir_fixture_incandescent",
                 "light_nir_fill_led", "light_nir_fill_incandescent", "flash_rgb", "flash_nir", "nir_band",
                 "light_rgb_lamp", "light_nir_lamp_led", "light_nir_lamp_incandescent", "nir_low", "nir_high")


def uniform_tag(name: str) -> float:
    """A distinct value to declare a shared uniform with. The loader merges objects with equal properties
    (merge_equivalent): uniforms declared alike (0, 1) became one object that every switch wrote at once.
    Each is found after loading by its tag (Resident) and then given its real value."""
    return round(0.731019 + 0.001317 * BAND_UNIFORMS.index(name), 6)  # odd values: no scene uniform holds one


def band_lights_text(text: str, classes: dict[str, str], groups: dict[str, str]) -> str:
    """Every area emitter's radiance behind a ``bandscale`` texture, scaled per band by shared uniforms: an
    indoor light by ``light_rgb_<group>`` and ``light_nir_<group>_<class>`` (group: fixture or fill, class:
    led or incandescent), the camera flash by ``flash_rgb`` and ``flash_nir``. One render then lights each
    band right (a batch sensor's second slot, or ``nir_band``), and a level or lighting mode is a few
    values written, never a light's own radiance."""
    band = f'<ref name="band" id="{NIR_BAND_ID}" />'

    def shape(m):
        sid, body = m.group(2), m.group(3)
        flash = sid.startswith(ASSIST_PREFIX)
        group = groups.get(sid, "fixture")
        rgb_id, nir_id = (("flash_rgb", "flash_nir") if flash else
                          (f"light_rgb_{group}", f"light_nir_{group}_{classes.get(sid, 'led')}"))
        body = re.sub(r'<rgb name="radiance" value="([^"]*)"\s*/>',
                      lambda r: (f'<texture name="radiance" type="bandscale"><rgb name="nested" value="{r.group(1)}" />'
                                 f'<ref name="rgb" id="{rgb_id}" /><ref name="nir" id="{nir_id}" />{band}</texture>'), body)
        return f'\n  <shape type="{m.group(1)}" id="{sid}">{body}\n  </shape>'

    text = re.sub(r'\n  <shape type="([a-z_]+)" id="([^"]+)">(.*?)\n  </shape>',
                  lambda m: shape(m) if '<emitter type="' in m.group(3) else m.group(0), text, flags=re.S)

    def delta(m):   # a point or spot light (point_light_xml): its intensity, as a shape's radiance
        sid, body = m.group(2), m.group(3)
        group = groups.get(sid, "fixture")
        body = re.sub(r'<rgb name="intensity" value="([^"]*)"\s*/>',
                      lambda r: (f'<texture name="intensity" type="bandscale"><rgb name="nested" value="{r.group(1)}" />'
                                 f'<ref name="rgb" id="light_rgb_{group}" /><ref name="nir" id="light_nir_{group}_'
                                 f'{classes.get(sid, "led")}" />{band}</texture>'), body)
        return f'\n  <emitter type="{m.group(1)}" id="{sid}">{body}\n  </emitter>'

    text = DELTA_LIGHT.sub(delta, text)
    opening = re.search(r"<scene\b[^>]*>", text)
    declare = "".join(f'\n  <texture type="uniform" id="{u}"><float name="value" value="{uniform_tag(u)}" />'
                      f'</texture>' for u in BAND_UNIFORMS if u != NIR_BAND_ID)
    return text[:opening.end()] + declare + text[opening.end():]


_UNIT_RECT: list[Path] = []
_UNIT_DISK: list[Path] = []


_TAG = re.compile(r'<!--.*?-->|<(/?)([A-Za-z_][\w.:-]*)((?:[^>"]|"[^"]*")*?)(/?)>', re.S)
_REF_ID = re.compile(r'<ref\b[^>]*\bid="([^"]+)"')
_ATTR_ID = re.compile(r'\bid="([^"]+)"')


def strip_unreferenced(text: str, keep=()) -> tuple[str, int]:
    """Drop top-level <bsdf> and <texture> elements nothing refers to (no <ref id=...> outside the dropped
    ones), until none is left: Mitsuba loads every top-level object, and Dr.Jit traces every registered
    BSDF into a render's material call, dead or not. A Device 1 apartment's base.xml carries 562 shared
    BSDFs that no shape uses (its shapes have inline ones): ~3/4 of a recording's tracing and codegen, and
    1724 textures loaded for nothing. ``keep``: ids the server looks up by itself. Returns (text, count)."""
    spans, depth, open_ = [], 0, None
    for m in _TAG.finditer(text):
        if m.group(0).startswith("<!--"):
            continue
        closing, name, attrs, self_closing = m.groups()
        if closing:
            depth -= 1
            if depth == 1 and open_ is not None and open_[1] == name:
                spans.append((open_[0], m.end(), open_[2]))
                open_ = None
            continue
        if depth == 1 and name in ("bsdf", "texture"):
            found = _ATTR_ID.search(attrs)
            if self_closing:
                spans.append((m.start(), m.end(), found.group(1) if found else None))
            else:
                open_ = (m.start(), name, found.group(1) if found else None)
        if not self_closing:
            depth += 1

    def without(removed: set) -> str:
        out, last = [], 0
        for i in sorted(removed, key=lambda i: spans[i][0]):
            out.append(text[last:spans[i][0]])
            last = spans[i][1]
        return "".join(out) + text[last:]

    keep, removed = set(keep), set()
    while True:  # an element only a dropped one referred to goes too
        refs = set(_REF_ID.findall(without(removed)))
        dead = {i for i, (_, _, sid) in enumerate(spans)
                if i not in removed and sid is not None and sid not in refs and sid not in keep}
        if not dead:
            return without(removed), len(removed)
        removed |= dead


def unit_rectangle_ply() -> Path:
    """Mitsuba's ``rectangle`` as a two-triangle PLY mesh (the same vertices, uvs and emitting side)."""
    if not _UNIT_RECT:
        import tempfile

        path = Path(tempfile.gettempdir()) / "opticalnav_unit_rectangle.ply"
        path.write_text("ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\n"
                        "property float z\nproperty float u\nproperty float v\nelement face 2\n"
                        "property list uchar int vertex_indices\nend_header\n"
                        "-1 -1 0 0 0\n1 -1 0 1 0\n1 1 0 1 1\n-1 1 0 0 1\n3 0 1 2\n3 2 3 0\n")
        _UNIT_RECT.append(path)
    return _UNIT_RECT[0]


def unit_disk_ply(segments: int = 32) -> Path:
    """A unit disk in z = 0 facing +z, as a PLY fan (a pendant's glowing opening; moved by its vertices like
    the rectangle lights, so lighting modes never re-record)."""
    if not _UNIT_DISK:
        import tempfile

        path = Path(tempfile.gettempdir()) / f"opticalnav_unit_disk_{segments}.ply"
        rim = [(math.cos(2 * math.pi * i / segments), math.sin(2 * math.pi * i / segments)) for i in range(segments)]
        verts = ["0 0 0 0.5 0.5"] + [f"{x:.6f} {y:.6f} 0 {0.5 + 0.5 * x:.6f} {0.5 + 0.5 * y:.6f}" for x, y in rim]
        faces = [f"3 0 {i + 1} {(i + 1) % segments + 1}" for i in range(segments)]  # counter-clockwise from +z
        path.write_text("ply\nformat ascii 1.0\n" f"element vertex {len(verts)}\n"
                        "property float x\nproperty float y\nproperty float z\nproperty float u\nproperty float v\n"
                        f"element face {segments}\nproperty list uchar int vertex_indices\nend_header\n"
                        + "\n".join(verts + faces) + "\n")
        _UNIT_DISK.append(path)
    return _UNIT_DISK[0]


def lights_as_meshes(text: str, ids) -> str:
    """Rectangle lights as PLY meshes. A lighting mode lifts a removed light out of the room; for a
    rectangle that is a write of its to_world, and the first such write after a recording changed the
    frozen render's inputs (64 opaque variables per rectangle) and recorded again. A mesh moves by its
    vertices, which a replay takes as they are."""
    ply = unit_rectangle_ply()
    for sid in ids:
        text = text.replace(f'<shape type="rectangle" id="{sid}">',
                            f'<shape type="ply" id="{sid}">\n    <string name="filename" value="{ply}" />')
    return text


_LAMP = re.compile(r"^(\w*(?:Lamp|CeilingLight|Sconce|Chandelier|Pendant)\w*)Factory")
PLACEHOLDER = "PointLampFactory"  # in the id of every light box robomituba's importer makes
_SPAWN = re.compile(r"_(\d+)_\.spawn_asset_(\d+)$")
FIXTURE = "fixture"  # the kind of a lamp found by position alone (pair_lamp_lights), whatever its name
# a part named as the one that glows (a kit's "..._ir_led_panel_2"): of a ceiling fixture, only it glows
_GLOWING_PART = re.compile(r"led|panel|diffuser|bulb|glow|emissi|lens", re.I)
ORPHAN_DOWNLIGHT_R = 0.1  # m: the round downlight a light by no fixture becomes (instead of a 0.6 m square)


def notes_from_shapes(shapes: dict) -> list[dict]:
    """For a scene file written without opticalnav-obj notes (the oldest packs): what pair_lamp_lights needs, from
    the shapes themselves. A lamp by its id (``DeskLampFactory_<seed>_.spawn_asset_<n>``), a light box by its id,
    radiance and to_world scale (a rectangle's and a cube's half sides). Pairs by seed and spawn id only."""
    out = []
    for sid, m in shapes.items():
        if "__part_" in sid:
            continue
        if PLACEHOLDER in sid and "<emitter" in m.group(3):
            rgb = re.search(r'name="radiance" value="([^"]+)"', m.group(3))
            scale = {a: float(v) for a, v in re.findall(r'<scale\b[^>]*?\b([xyz])="([-\d.e]+)"', m.group(3))}
            sx, sy = 2 * scale.get("x", 1.0), 2 * scale.get("y", 1.0)
            size = [sx, 0.0, sy] if m.group(1) == "rectangle" else [sx, sy, 2 * scale.get("z", 1.0)]
            out.append({"id": sid, "is_emitter": True, "label": f"light:{sid}",
                        "emitter_radiance": [float(v) for v in rgb.group(1).replace(",", " ").split()] if rgb else None,
                        "geometry": {"size_m": size}})
        elif _LAMP.match(sid) and _SPAWN.search(sid):
            out.append({"id": sid, "label": sid})
    return out


def pair_lamp_lights(notes: list[dict]) -> tuple[list[dict], dict, dict, list[dict]]:
    """The importer's light boxes (``PointLampFactory`` ids, a 0.6 m square or flat cube each) and the lamps that
    glow with them: (lamps, lamp id -> kind, lamp id -> its light, the lights left with no fixture).

    0. By the importer's own tag: a light whose note has ``metadata.fixture_id`` (apps/import_infinigen_scene.py,
       from the fixture mesh whose bounding box holds the light) takes that object. A light tagged ``null`` is
       one the importer found no fixture mesh for (kit_svc, sun, fill, windows): an orphan, whatever lies near.
       Several lights tagged with one object (the importer merges a track rail or a row of hanging bars into one
       mesh, so every spot on it names it) stay lights at their own places, as orphans: the mesh does not glow.
       A tag naming no object of the scene falls through to the rules below; a scene without tags only has them.
    1. By name: a lamp whose label names a lamp (_LAMP) takes the light of its factory seed and spawn id, else the
       nearest light with no seed of its own within 0.75 of its width (hand-built kits name lights by role:
       ``PointLampFactory_pendant_.spawn_asset_3``). A named lamp with no light stays a lamp (a dark one).
    2. By position, whatever the names: a light still alone takes the nearest asset (an id, a ``...Factory``
       label) whose footprint holds it (10 cm to spare) and whose height is within 0.6 m of it: a track spot, a
       ceiling fixture of a kit, any fixture a new kit brings. Kind FIXTURE: what glows follows from its parts.
    3. Whatever is still alone (a light the importer placed by no mesh) is an orphan."""
    lights = [o for o in notes if o.get("is_emitter") and PLACEHOLDER in str(o.get("id", ""))]
    keyed = {_SPAWN.search(o["id"]).groups(): o for o in lights if _SPAWN.search(o["id"])}
    loose = [o for o in lights if not _SPAWN.search(o["id"]) and (o.get("geometry") or {}).get("center")]
    lamps = [o for o in notes if not o.get("is_emitter") and o.get("id") and _LAMP.match(str(o.get("label", "")))]
    kinds = {o["id"]: _LAMP.match(o["label"]).group(1) for o in lamps}
    paired: dict[str, dict] = {}
    objects = {o["id"]: o for o in notes if not o.get("is_emitter") and o.get("id")}
    no_fixture = set()  # lights that stay where they are: tagged with no fixture, or sharing one with others
    shared = Counter((L.get("metadata") or {}).get("fixture_id") for L in lights)
    for light in lights:
        md = light.get("metadata") or {}
        if "fixture_id" not in md:
            continue
        fid = md["fixture_id"]
        if not fid or shared[fid] > 1:
            no_fixture.add(light["id"])
        elif fid in objects and fid not in paired:
            paired[fid] = light
            if fid not in kinds:
                lamps.append(objects[fid])
                kinds[fid] = _LAMP.match(str(objects[fid].get("label", ""))).group(1) \
                    if _LAMP.match(str(objects[fid].get("label", ""))) else FIXTURE
    tagged = {L["id"] for L in paired.values()}
    loose = [o for o in loose if o["id"] not in tagged]
    for o in lamps:
        if o["id"] in paired:
            continue
        spawn = _SPAWN.search(o["id"])
        light = keyed.get(spawn.groups()) if spawn else None
        if light is not None and light["id"] in tagged:
            light = None
        g = o.get("geometry") or {}
        if light is None and loose and g.get("center"):
            near = min(loose, key=lambda L: math.dist(L["geometry"]["center"], g["center"]))
            reach = 0.75 * max(float(v) for v in (g.get("size_m") or [0.0, 0.0, 0.0])[::2])
            if math.dist(near["geometry"]["center"], g["center"]) <= reach:
                light = near
                loose.remove(near)
        if light is not None:
            paired[o["id"]] = light
    taken = {L["id"] for L in paired.values()}
    named = {o["id"] for o in lamps}
    assets = [o for o in notes if not o.get("is_emitter") and o.get("id") and o["id"] not in named
              and "Factory" in str(o.get("label", "")) and (o.get("geometry") or {}).get("center")]

    def height(o: dict) -> tuple[float, float]:  # (base, top) as the note has them
        g = o.get("geometry") or {}
        base = float(g.get("base_height_m") or 0.0)
        return base, base + float((g.get("size_m") or [0.0, 0.0, 0.0])[1])

    orphans = []
    for light in lights:
        if light["id"] in taken:
            continue
        if light["id"] in no_fixture:
            orphans.append(light)
            continue
        g = light.get("geometry") or {}
        if not g.get("center"):
            orphans.append(light)
            continue
        (x, z), (lo, hi) = g["center"], height(light)
        y = (lo + hi) / 2
        fits = []
        for o in assets:
            c, s = o["geometry"]["center"], [float(v) for v in o["geometry"].get("size_m") or [0.0, 0.0, 0.0]]
            b, t = height(o)
            if abs(x - c[0]) <= s[0] / 2 + 0.1 and abs(z - c[1]) <= s[2] / 2 + 0.1 and b - 0.6 <= y <= t + 0.6:
                fits.append((math.dist((x, z), c), o))
        if not fits:
            orphans.append(light)
            continue
        o = min(fits, key=lambda f: f[0])[1]
        assets.remove(o)
        lamps.append(o)
        kinds[o["id"]] = FIXTURE
        paired[o["id"]] = light
    return lamps, kinds, paired, orphans


def cone_recess(radius: float, height: float, cone_deg: float) -> tuple[float, float]:
    """(depth, gain) to make a glowing disk of ``radius`` under a housing ``height`` tall shine in a cone of
    ``cone_deg`` (a Blender spot's spot_size). The disk is set back inside the housing by the depth at which the
    housing's mouth, the same radius, is seen under the cone's half angle (radius / tan(half)), at most 85% of the
    housing's height; its walls then cut what leaves to a cone, with no emitter that has a direction (every
    emitter here is a Lambertian area light, which the light levels, groups and bands all act on). The walls take
    flux that Blender's spot keeps (a spot is a point light cut to its cone), so the radiance is raised by
    1 / F, F the share of a Lambertian disk's flux that reaches a coaxial disk of the same radius across the
    depth ( (X - sqrt(X^2 - 4)) / 2 with X = 2 + (depth / radius)^2 ), at most 3: the walls (metal) reflect some."""
    half = math.radians(min(max(cone_deg, 5.0), 170.0) / 2.0)
    depth = min(0.85 * max(height, 0.0), radius / math.tan(half))
    x = 2.0 + (depth / max(radius, 1e-6)) ** 2
    share = 0.5 * (x - math.sqrt(max(x * x - 4.0, 0.0)))
    return depth, min(3.0, 1.0 / max(share, 1e-6))


IMPORTER_WATTS = 0.025  # the importer's box radiance, colour-free, is 1 + IMPORTER_WATTS x the Blender light's watts


def lamp_lights_text(text: str, mesh_info=None, lit: bool = True, mesh_bounds=None,
                     blender_power: bool = False, watt_gain: float = 1.0) -> tuple[str, list[str]]:
    """Lamps glow from their own parts instead of from boxes. Infinigen spawns a light (``PointLampFactory``) with
    only some of its lamps; robomituba's importer turns those into a flat emitter cube (0.6 x 0.6 x 0.08 m) at a
    desk or floor lamp's top and a 0.6 m square under a ceiling light, and the other lamps stay dark. A lamp is
    lit when a light shares its factory seed and spawn id (``DeskLampFactory_<seed>_.spawn_asset_<n>`` and
    ``light_.._PointLampFactory_<seed>_.spawn_asset_<n>``).

    * A lit lamp (``lit``): its light's power (radiance x emitting area) moves to the lamp's glowing part and the
      box goes: a desk or floor lamp's shade, every non-metal part of a ceiling light or chandelier. Labelled
      ``light:fixture:<lamp>``, so it stays in the scene's own lights (group "fixture", lighting modes) with the
      box's class. The boxes stuck out of the shades as glowing slabs and turned every lamp into a square.
    * A dark lamp: its shade gets an area emitter at the power of the scene's lamp lights of the same kind (desk,
      floor, ...; else their mean), group ``lamp`` (dark unless a view asks for it).

    A shade is, of the lamp's parts that are not metal, the one whose bottom is highest (the shade sits on the
    stand). A light hidden inside the shade, as first tried, reached the room only through paths that bounced
    into the shade and hit it by chance: fireflies over every wall (464 vs 49 bright outliers in a 256 spp view).
    A glowing part is sampled directly from everywhere. ``mesh_info(shape_type, filename)`` gives a part's
    (surface area, lowest y in its own frame): both sides of a shell emit. Without it, only a dark lamp with one
    non-metal part is lit (a cylinder from the lamp's recorded size standing in for its area) and lit lamps keep
    their boxes. Returns (text, the ids of the parts that now emit).

    Which lamp a light belongs to: pair_lamp_lights, by name and then by position. A fixture found by position
    alone (a kit's track spot or ceiling fixture, whatever it is called) glows as its parts say: all metal, from
    a disk across its opening (as a pendant's dome); flat (half as tall as wide), from its named diffuser (``led``,
    ``panel``, ...) or else every part that is not metal (as a ceiling light); else from its shade, or whole if
    it is one part. A light by no fixture at all
    becomes a round downlight (ORPHAN_DOWNLIGHT_R) where its square hung, at the square's power: no light of
    the scene stays a square (tools/light_audit.py lists what a scene's lights pair with).

    A pendant with a metal dome glows from a disk across the dome's opening, facing down, given
    ``mesh_bounds(shape_type, shape_body)`` (a part's world bounding box). Its bulb, the first try, held the
    light box's power in ~0.02 m2: a source that bright and small left fireflies on every glossy surface of
    the room. The disk spreads it over ~6x the area and is sampled directly from below. Without a dome
    (or the callback) the pendant glows from its lowest non-metal part, the bulb (its highest is the cord).

    ``blender_power``: a light with ``metadata.light.energy`` (Blender watts; not a SUN), or in a scene imported before the
    light export one whose watts the importer's radiance gives back (1 + 0.025 W, checked on 17 lights of 3 scenes
    and on the 3 kinds of stillroom's lights against its Cycles dump), glows at Blender's own
    power: a radiant flux of energy / pi (a point light's intensity is energy / (4 pi^2) W/sr, an area light's radiance
    energy / (pi^2 area)); a spot keeps the point light's intensity inside its cone. The importer's box radiance came
    to 1/7.5 of that for floorG's 60 W area lights and 1/3.5 to 1/10 for its point lights (its spots match)."""
    raw: dict[str, str] = {}  # note id -> the note as written, to drop a replaced box's note
    notes = []
    for note in re.findall(r"<!--opticalnav-obj:(\{.*?\})-->", text):
        try:
            obj = json.loads(note)
        except ValueError:
            continue
        notes.append(obj)
        if obj.get("id"):
            raw[obj["id"]] = note
    shapes = {m.group(2): m for m in re.finditer(r'\n  <shape type="([a-z_]+)" id="([^"]+)">(.*?)\n  </shape>', text, re.S)}
    lamps, kinds, paired, orphans = pair_lamp_lights(notes or notes_from_shapes(shapes))

    def blender_light(light: dict | None) -> tuple[str, float] | None:
        """(type, Blender watts) of a light when ``blender_power``: from metadata.light, else read back from the
        importer's box radiance (see IMPORTER_WATTS) for a scene imported before the light export; a SUN has no power
        here."""
        if not blender_power or not light:
            return None
        info = (light.get("metadata") or {}).get("light") or {}
        if info.get("energy"):
            return (info.get("type"), float(info["energy"])) if info.get("type") != "SUN" else None
        rad = light.get("emitter_radiance")
        if not rad or not str(light.get("id", "")).startswith("light_") or "softbox" in str(light["id"]):
            return None
        watts = (max(float(v) for v in rad) * float(light.get("emitter_intensity") or 1.0) - 1.0) / IMPORTER_WATTS
        if watts <= 0:
            return None
        kind = "SPOT" if "_track_" in light["id"] else "AREA" if "_hung_" in light["id"] else "POINT"
        return kind, watts

    def from_blender(light: dict | None) -> bool:
        return blender_light(light) is not None

    def light_power(light: dict) -> float | None:
        """Emitted power / pi of an importer's light box: radiance x its emitting area (``blender_power``: the
        light's own Blender energy / pi^2, the same measure)."""
        known = blender_light(light)
        if known is not None:
            # a spot keeps a point light's intensity inside its cone (energy / 4 pi^2 W/sr), which is what a
            # Lambertian disk's radiance x area gives on its axis
            return known[1] * watt_gain / ((4.0 if known[0] == "SPOT" else 1.0) * math.pi ** 2)
        if not light.get("emitter_radiance"):
            return None
        sx, sy, sz = (float(v) for v in (light.get("geometry") or {}).get("size_m") or [0.6, 0.08, 0.6])
        rad = float(np.mean(light["emitter_radiance"])) * float(light.get("emitter_intensity") or 1.0)
        kind = shapes[light["id"]].group(1) if light["id"] in shapes else "cube"
        return rad * (sx * sz if kind in ("rectangle", "disk") else 2 * (sx * sz + sx * sy + sy * sz))

    def tint(light: dict | None) -> tuple[float, float, float]:
        """The colour of a light that glows at its Blender power (``blender_power``): its metadata.light.color,
        else the importer's radiance over its largest channel (Blender's colour: the largest channel 1); white
        otherwise (a lamp the scene has no light for, or the importer's own power)."""
        if blender_light(light) is None:
            return (1.0, 1.0, 1.0)
        info = (light.get("metadata") or {}).get("light") or {}
        col = np.asarray(info.get("color") or light.get("emitter_radiance") or [1.0] * 3, float)
        col = col / max(float(col.max()), 1e-9) if not info.get("color") else col
        return tuple(float(c) for c in col[:3])

    def disk(sid: str, r: float, center, rad: float, note: dict, aim=None, col=(1.0, 1.0, 1.0)) -> None:
        """A disk of radius r facing the floor (or ``aim``, a unit vector), centred at ``center`` (x, y, z), glowing
        at ``rad``."""
        tilt = ""
        if aim is not None:  # turn the floor-facing normal (0, -1, 0) onto aim
            axis = np.cross([0.0, -1.0, 0.0], aim)
            sin = float(np.linalg.norm(axis))
            if sin > 1e-3:
                axis /= sin
                angle = math.degrees(math.atan2(sin, float(np.dot([0.0, -1.0, 0.0], aim))))
                tilt = (f'\n      <rotate x="{axis[0]:.6f}" y="{axis[1]:.6f}" z="{axis[2]:.6f}" angle="{angle:.4f}" />')
        xml.append(f'\n  <shape type="ply" id="{sid}">\n    <string name="filename" value="{unit_disk_ply()}" />'
                   f'\n    <transform name="to_world">\n      <scale value="{r:.6f}" />'
                   f'\n      <rotate x="1" angle="90" />{tilt}'  # +z (the disk's front) to -y: facing the floor
                   f'\n      <translate x="{center[0]:.6f}" y="{center[1]:.6f}" z="{center[2]:.6f}" />'
                   f'\n    </transform>\n    <bsdf type="diffuse" />'
                   f'\n    <emitter type="area">\n      <rgb name="radiance" value="{rad * col[0]:.4f} {rad * col[1]:.4f} {rad * col[2]:.4f}" />'
                   f'\n    </emitter>\n  </shape>')
        xml.append(f'\n  <!--opticalnav-obj:{json.dumps(note)}-->')
        added.append(sid)

    power: dict[str, list[float]] = {}
    for o in lamps:  # the scene's own lamp lights, per kind
        light = paired.get(o["id"])
        if light is not None and light_power(light) is not None:
            power.setdefault(kinds[o["id"]], []).append(light_power(light))
    every = [p for ps in power.values() for p in ps]
    default = float(np.mean(every)) if every else 2.0 * 0.912
    edits, added, xml, gone = [], [], [], []
    for o in lamps:
        kind = kinds[o["id"]]
        light = paired.get(o["id"])
        if light is not None and not (lit and mesh_info is not None and light["id"] in shapes):
            continue
        parts = [m for sid, m in shapes.items() if sid == o["id"] or sid.startswith(o["id"] + "__part_")]
        shades = [m for m in parts if "conductor" not in m.group(3) and "<emitter" not in m.group(3)]
        g = o.get("geometry") or {}
        sx, sy, sz = (float(v) for v in g.get("size_m") or [0.2, 0.5, 0.2])
        if kind == FIXTURE:  # found by position: what glows follows from its parts and its shape
            kind = "Pendant" if not shades else "Ceiling" if sy < 0.5 * max(sx, sz) else FIXTURE
        ceiling = "Ceiling" in kind or "Chandelier" in kind
        info = {}
        for m in shades:
            found = re.search(r'<string name="filename" value="([^"]+)"', m.group(3))
            if mesh_info is not None and found:
                with contextlib.suppress(Exception):
                    info[m.group(2)] = tuple(float(v) for v in mesh_info(m.group(1), found.group(1)))
        height = sy
        if light is not None and ceiling:  # a ceiling light's fixture glows whole, or its named diffuser
            shades = [m for m in shades if m.group(2) in info]
            shades = [m for m in shades if _GLOWING_PART.search(m.group(2)[len(o["id"]):])] or shades
        elif "Pendant" in kind:  # the dome's opening glows (see above), else the bulb
            dome = None
            for m in parts:
                if "conductor" in m.group(3) and mesh_bounds is not None:
                    with contextlib.suppress(Exception):
                        lo, hi = (np.asarray(v, float) for v in mesh_bounds(m.group(1), m.group(3)))
                        dome = (lo, hi) if dome is None else (np.minimum(dome[0], lo), np.maximum(dome[1], hi))
            if dome is not None:
                lo, hi = dome
                r = 0.47 * float(min(hi[0] - lo[0], hi[2] - lo[2]))
                own = light_power(light) if light is not None else None
                rad = (own if own is not None else float(np.mean(power.get(kind) or [default]))) / (math.pi * r * r)
                # a spot (metadata.light.spot_size_deg, from the importer): the disk sits back in the housing
                cone = (((light or {}).get("metadata") or {}).get("light") or {}).get("spot_size_deg")
                sink = 0.0
                aim = (((light or {}).get("metadata") or {}).get("light") or {}).get("direction")
                if aim is not None and len(aim) == 3:  # plan frame (x, y, z up) -> the scene's (x, up, y)
                    aim = np.array([aim[0], aim[2], aim[1]], float)
                    aim = aim / np.linalg.norm(aim) if np.linalg.norm(aim) > 1e-6 and aim[1] < -0.2 else None
                else:
                    aim = None
                if cone and 5.0 <= float(cone) < 170.0:
                    sink, gain = cone_recess(r, float(hi[1] - lo[1]), float(cone))
                    rad *= 1.0 if from_blender(light) else gain  # Blender's spot cone loses the flux the walls take
                sid = f"{o['id']}__glow"
                col = tint(light)
                disk(sid, r, ((lo[0] + hi[0]) / 2, lo[1] + sink, (lo[2] + hi[2]) / 2), rad,
                     {"id": sid, "type": "landmark", "is_emitter": True, "emitter_radiance": [round(rad * c, 4) for c in col],
                      "emitter_intensity": 1.0, "emitter_shape": "lamp_opening",
                      "label": f"light:{'fixture' if light is not None else 'lamp'}:{o['label']}",
                      "metadata": {"lamp": o["id"], "glowing_area_m2": round(math.pi * r * r, 4),
                                   "replaces": (light or {}).get("id"), "added_by": "opticalnav_sim lamp_lights_text"},
                      "geometry": o.get("geometry")}, aim, col)
                if light is not None:
                    gone.append(light["id"])
                continue
            shades = sorted((m for m in shades if m.group(2) in info), key=lambda m: info[m.group(2)][1])[:1]
        elif len(shades) > 1 or light is not None:  # the shade: the highest bottom, a quarter of the lamp up at least
            whole = [m for m in shades if m.group(2) in info] if kinds[o["id"]] == FIXTURE and len(shades) == 1 else []
            shades = sorted((m for m in shades if m.group(2) in info and info[m.group(2)][1] > 0.25 * height),
                            key=lambda m: info[m.group(2)][1])[-1:] or whole  # a fixture of one part: all of it
        if not shades or (light is None and len(shades) != 1):  # no telling which part glows: as it was
            continue
        area = sum(info.get(m.group(2), (0.0, 0.0))[0] for m in shades)
        if not area:  # a cylinder of the lamp's width, a third of its height, both sides
            area = 2 * math.pi * 0.5 * min(sx, sz) * sy / 3 * 2
        own = light_power(light) if light is not None else None
        rad = (own if own is not None else float(np.mean(power.get(kind) or [default]))) / max(area, 1e-4)
        col = tint(light)
        for shade in shades:
            sid = shade.group(2)
            edits.append((shade.end(3), f'\n    <emitter type="area">\n      <rgb name="radiance" value="{rad * col[0]:.4f} {rad * col[1]:.4f} {rad * col[2]:.4f}" />'
                                        f'\n    </emitter>'))
            note = {"id": sid, "type": "landmark", "is_emitter": True, "emitter_radiance": [round(rad * c, 4) for c in col],
                    "emitter_intensity": 1.0, "emitter_shape": "lamp_part" if ceiling else "lamp_shade",
                    "label": f"light:{'fixture' if light is not None else 'lamp'}:{o['label']}",
                    "metadata": {"lamp": o["id"], "glowing_area_m2": round(area, 4), "replaces": (light or {}).get("id"),
                                 "added_by": "opticalnav_sim lamp_lights_text"},
                    "geometry": o.get("geometry")}
            xml.append(f'\n  <!--opticalnav-obj:{json.dumps(note)}-->')
            added.append(sid)
        if light is not None:
            gone.append(light["id"])
    for light in orphans if lit else ():  # a light by no fixture: a round downlight where its square hung
        m = shapes.get(light["id"])
        own = light_power(light)
        if m is None or own is None:
            continue
        at = re.search(r'<translate x="([-\d.e]+)" y="([-\d.e]+)" z="([-\d.e]+)"', m.group(3))
        g = light.get("geometry") or {}
        size = [float(v) for v in g.get("size_m") or [0.6, 0.08, 0.6]]
        if at:
            center = [float(v) for v in at.groups()]
            center[1] -= size[1] / 2 if m.group(1) == "cube" else 0.0  # a box's bottom face
        elif g.get("center"):
            center = [float(g["center"][0]), float(g.get("base_height_m") or 0.0), float(g["center"][1])]
        else:
            continue
        r = ORPHAN_DOWNLIGHT_R
        rad = own / (math.pi * r * r)
        sid = f"{light['id']}__downlight"
        col = tint(light)
        disk(sid, r, center, rad,
             {"id": sid, "type": "landmark", "is_emitter": True, "emitter_radiance": [round(rad * c, 4) for c in col],
              "emitter_intensity": 1.0, "emitter_shape": "downlight",
              "label": f"light:fixture:CeilingDownlight:{light.get('label', light['id'])}",
              "metadata": {"glowing_area_m2": round(math.pi * r * r, 4), "replaces": light["id"],
                           "added_by": "opticalnav_sim lamp_lights_text (a light by no fixture)"},
              "geometry": g}, None, col)
        gone.append(light["id"])
    for at, emitter in sorted(edits, reverse=True):  # from the end, so earlier offsets hold
        text = text[:at] + emitter + text[at:]
    for lid in gone:  # the replaced boxes, and their notes (light_groups / light_classes read the notes)
        text = re.sub(rf'\n  <shape type="[a-z_]+" id="{re.escape(lid)}">.*?\n  </shape>', "", text, count=1, flags=re.S)
        if lid in raw:
            text = text.replace(f"<!--opticalnav-obj:{raw[lid]}-->", "", 1)
    if xml:
        end = text.rindex("</scene>")
        text = text[:end] + "".join(xml) + "\n" + text[end:]
    if blender_power:
        text = boxes_at_blender_power(text, notes, watt_gain)
    return text, added


POINT_EMITTERS = os.environ.get("OPTICALNAV_POINT_EMITTERS", "1") != "0"  # 0: spots as the earlier disk in a tube
SPOT_RADIUS_M = 0.25    # a spot this soft or harder is a point spot (its radius only softens shadows; PH33: 0.1 m)
POINT_SPHERE_M = 0.01   # a point light with a larger radius glows from a sphere of it (Cycles 4's sphere light)
DELTA_LIGHT = re.compile(r'\n  <emitter type="(spot|point)" id="([^"]+)">(.*?)\n  </emitter>', re.S)


def _light_frame(info: dict) -> np.ndarray | None:
    """A Blender light's axes in the scene's frame (columns: its x, y and the way it shines, Blender's -z), from
    metadata.light.basis (tools/blend_to_pack.py, the scene's frame) or its direction alone (plan frame: x, y, up)."""
    if info.get("basis"):
        b = np.asarray(info["basis"], float).reshape(3, 3)
        x, y, z = (b[:, k] / max(float(np.linalg.norm(b[:, k])), 1e-9) for k in range(3))
        return np.stack([x, y, -z], 1)
    d = info.get("direction")
    if not d or len(d) != 3:
        return None
    s = np.array([d[0], d[2], d[1]], float)
    s /= max(float(np.linalg.norm(s)), 1e-9)
    x = np.array([1.0, 0.0, 0.0]) - s[0] * s
    if np.linalg.norm(x) < 1e-3:
        x = np.array([0.0, 0.0, 1.0]) - s[2] * s
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(s, x), s], 1)


def area_rotation_xml(info: dict) -> str | None:
    """The rotation that turns a rectangle or disk (it shines along its +z) to shine the light's way, as a to_world
    ``<matrix>``; None when it shines straight down with no basis known (the converter's ``rotate x 90`` holds)."""
    f = _light_frame(info)
    if f is None or (not info.get("basis") and f[1, 2] < -0.999):
        return None
    x, s = f[:, 0], f[:, 2]
    y = np.cross(s, x)   # right-handed with x and the normal s
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2] = x, y, s
    return '<matrix value="' + " ".join(f"{v:.6f}" for v in m.ravel()) + '" />'


def spot_beam_deg(cone_deg: float, blend: float) -> float:
    """Mitsuba's spot beam_width for Blender's spot_size and spot_blend: Cycles 4 eases in over the inner
    ``blend`` of the cone's (1 - cos) range (a smoothstep); Mitsuba ramps linearly in angle from beam_width to
    cutoff_angle. The ramp starts where Cycles' smoothstep reaches full, so the beam's edge spans the same angles."""
    half = math.radians(cone_deg / 2.0)
    c = math.cos(half) + (1.0 - math.cos(half)) * min(max(blend, 0.0), 1.0)
    return max(min(math.degrees(math.acos(min(c, 1.0))), cone_deg / 2.0 - 1e-3), 0.0)


def point_light_xml(sid: str, info: dict, intensity: float, colour) -> str:
    """A Blender point or spot light as Mitsuba's own: ``point`` or ``spot`` (``intensity`` W/sr, times colour),
    a top-level emitter whose id is the light's; a point light with a radius over POINT_SPHERE_M glows from a
    sphere of that radius (radiance intensity / (pi r^2): a sphere's intensity is its radiance x pi r^2)."""
    p = info["position"]
    c = (float(p[0]), float(p[2]), float(p[1]))   # plan (x, y, up) -> the scene's (x, up, y)
    r = float(info.get("radius_m") or 0.0)
    rgb_ = " ".join(f"{intensity * float(v):.5f}" for v in colour)
    if info.get("type") == "POINT" and r > POINT_SPHERE_M:
        rad = " ".join(f"{intensity * float(v) / (math.pi * r * r):.5f}" for v in colour)
        return (f'\n  <shape type="sphere" id="{sid}">\n    <point name="center" x="{c[0]:.6f}" y="{c[1]:.6f}" z="{c[2]:.6f}" />'
                f'\n    <float name="radius" value="{r:.6f}" />\n    <bsdf type="diffuse" />'
                f'\n    <emitter type="area">\n      <rgb name="radiance" value="{rad}" />\n    </emitter>\n  </shape>')
    if info.get("type") == "POINT":
        return (f'\n  <emitter type="point" id="{sid}">\n    <point name="position" x="{c[0]:.6f}" y="{c[1]:.6f}" z="{c[2]:.6f}" />'
                f'\n    <rgb name="intensity" value="{rgb_}" />\n  </emitter>')
    f = _light_frame(info)
    t = (c[0] + f[0, 2], c[1] + f[1, 2], c[2] + f[2, 2])
    up = f[:, 1] if abs(float(f[1, 2])) > 0.99 else np.array([0.0, 1.0, 0.0])
    cone = float(info["spot_size_deg"])
    return (f'\n  <emitter type="spot" id="{sid}">\n    <transform name="to_world">'
            f'\n      <lookat origin="{c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}" target="{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}"'
            f' up="{up[0]:.6f}, {up[1]:.6f}, {up[2]:.6f}" />\n    </transform>'
            f'\n    <rgb name="intensity" value="{rgb_}" />'
            f'\n    <float name="cutoff_angle" value="{min(cone / 2.0, 90.0):.4f}" />'
            f'\n    <float name="beam_width" value="{spot_beam_deg(min(cone, 180.0), float(info.get("spot_blend") or 0.0)):.4f}" />'
            f'\n  </emitter>')


def boxes_at_blender_power(text: str, notes: list[dict], watt_gain: float = 1.0) -> str:
    """The importer's light boxes that no lamp took over (``light_*`` shapes with an area emitter: a corridor's ceiling
    panels ``light_0_al``, say) at Blender's power as well: radiance = energy / (pi^2 x emitting area), times the
    light's colour (a spot: energy / 4 pi^2). The energy is the light's metadata.light.energy, else read back from the
    box's radiance (1 + IMPORTER_WATTS x watts, as for the lamps). Fill softboxes are left alone.

    ``watt_gain``: measured against Cycles (a lone light over a Lambertian plane, tools/blend_to_pack.py's check), Blender's
    area light radiates a flux of its watts (radiance W / (pi A)) and its point and spot lights W / (4 pi) W/sr: pi times what
    the formulas above give. The scenes converted straight from their .blend (``blend_direct_v1``) get pi; the importer's
    scenes keep 1, their fill and sky having been tuned against it."""
    meta = {n["id"]: ((n.get("metadata") or {}).get("light") or {}) for n in notes if n.get("id")}
    energy = {k: (v.get("energy"), v.get("type")) for k, v in meta.items()}
    housings: list[str] = []  # the tubes that cut a spot's emitter to its cone, added after the shapes
    has_sun = '<emitter type="directional"' in text

    def shape(m):
        sid, body = m.group(2), m.group(3)
        if not sid.startswith("light_") or sid.startswith(FILL_PREFIX) or "<emitter" not in body:
            return m.group(0)
        rad = re.search(r'(<rgb name="radiance" value=")([^"]+)(")', body)
        sc = re.search(r'<scale x="([\d.]+)" y="([\d.]+)"(?: z="([\d.]+)")? />', body)
        if rad is None or sc is None:
            return m.group(0)
        values = [float(v) for v in rad.group(2).split()]
        watts, kind = energy.get(sid, (None, None))
        if not watts:
            watts = (max(values) - 1.0) / IMPORTER_WATTS
        if kind == "SUN" and has_sun:  # the converter's box at a sun (tools/blend_to_pack.py before it skipped them): the
            return ""                  # scene's directional emitter is the sun, the box was a glowing square besides
        if watts <= 0 or kind == "SUN":
            return m.group(0)
        sx, sy = 2 * float(sc.group(1)), 2 * float(sc.group(2))
        kind_of_shape, shape_text = m.group(1), m.group(0)
        info = meta.get(sid, {})
        size = info.get("area_size_m")
        cone = info.get("spot_size_deg")
        pos = info.get("position")
        colour = np.asarray(info.get("color") or values, float)[:3]
        colour = colour if info.get("color") else colour / max(float(colour.max()), 1e-9)
        if POINT_EMITTERS and pos and len(pos) == 3 and (
                (info.get("type") == "SPOT" and cone and info.get("direction") and float(info.get("radius_m") or 0.0) <= SPOT_RADIUS_M)
                or info.get("type") == "POINT"):
            # Blender's own light at its intensity, energy / 4 pi W/sr (watt_gain pi on energy / 4 pi^2): a spot as
            # Mitsuba's ``spot``, a point as ``point`` or, with a radius, a glowing sphere (point_light_xml)
            return point_light_xml(sid, info, float(watts) * watt_gain / (4.0 * math.pi ** 2), colour)
        if info.get("type") == "SPOT" and cone and 5.0 <= float(cone) < 170.0 and pos and len(pos) == 3 and info.get("direction"):
            # a spot: a small disk at the light's place facing its direction, inside an absorbing tube as long as the
            # cone needs (the tube's mouth cuts the disk's Lambertian emission to the spot's cone, as the recess of a
            # fixture does), at the spot's intensity (energy / 4 pi^2 W/sr) on its axis
            r = min(max(float(info.get("radius_m") or 0.05), 0.02), 0.15)
            half = math.radians(float(cone) / 2.0)
            length = min(2.0 * r / math.tan(half), 0.6)
            d = np.array([info["direction"][0], info["direction"][2], info["direction"][1]], float)
            d /= max(float(np.linalg.norm(d)), 1e-9)
            tilt = ""
            axis = np.cross([0.0, -1.0, 0.0], d)
            sin = float(np.linalg.norm(axis))
            if sin > 1e-3:
                angle = math.degrees(math.atan2(sin, float(np.dot([0.0, -1.0, 0.0], d))))
                axis /= sin
                tilt = f'\n      <rotate x="{axis[0]:.6f}" y="{axis[1]:.6f}" z="{axis[2]:.6f}" angle="{angle:.4f}" />'
            c = [float(pos[0]), float(pos[2]), float(pos[1])]
            e = [c[i] + length * d[i] for i in range(3)]
            level = float(watts) * watt_gain / (4.0 * math.pi ** 2 * math.pi * r * r)
            new = " ".join(f"{level * x:.4f}" for x in colour)
            housings.append(
                f'\n  <shape type="cylinder" id="{sid}__housing">\n    <point name="p0" x="{c[0]:.6f}" y="{c[1]:.6f}" z="{c[2]:.6f}" />'
                f'\n    <point name="p1" x="{e[0]:.6f}" y="{e[1]:.6f}" z="{e[2]:.6f}" />\n    <float name="radius" value="{r:.6f}" />'
                '\n    <bsdf type="diffuse">\n      <rgb name="reflectance" value="0.02 0.02 0.02" />\n    </bsdf>\n  </shape>')
            return (f'\n  <shape type="disk" id="{sid}">\n    <transform name="to_world">\n      <scale value="{r:.6f}" />'
                    f'\n      <rotate x="1" angle="90" />{tilt}\n      <translate x="{c[0]:.6f}" y="{c[1]:.6f}" z="{c[2]:.6f}" />'
                    f'\n    </transform>\n    <bsdf type="diffuse" />\n    <emitter type="area">\n      <rgb name="radiance" value="{new}" />'
                    '\n    </emitter>\n  </shape>')
        if info.get("type") == "AREA" and size and info.get("area_shape") in ("DISK", "SQUARE", "RECTANGLE", "ELLIPSE"):
            # the light's own shape and size instead of the importer's 0.6 m square: a Blender area light's size is the
            # side of a square, the diameter of a disk or an ellipse's axes
            if info["area_shape"] in ("DISK", "ELLIPSE"):
                rx, ry = float(size[0]) / 2, float(size[-1]) / 2
                kind_of_shape = "disk"
                area = math.pi * rx * ry
            else:
                rx, ry = float(size[0]) / 2, float(size[-1]) / 2
                kind_of_shape = "rectangle"
                area = 4 * rx * ry
            shape_text = shape_text.replace(f'<shape type="{m.group(1)}"', f'<shape type="{kind_of_shape}"', 1)
            shape_text = shape_text.replace(sc.group(0), f'<scale x="{rx:.6f}" y="{ry:.6f}" />', 1)
            turn = area_rotation_xml(info)  # facing the light's own way (a wall washer, a tilted panel), not the floor
            if turn:
                shape_text = shape_text.replace('<rotate x="1" angle="90" />', turn, 1)
            pos = info.get("position")
            if pos and len(pos) == 3:  # plan frame (x, y, up) -> the scene's height
                shape_text = re.sub(r'(<translate x="[-\d.]+" y=")[-\d.]+(")', lambda t: f"{t.group(1)}{float(pos[2]):.6f}{t.group(2)}", shape_text, count=1)
        elif m.group(1) == "cube":
            sz = 2 * float(sc.group(3) or sc.group(2))
            area = 2 * (sx * sy + sx * sz + sy * sz)
        else:
            area = sx * sy
        level = watts * watt_gain / ((4.0 if kind == "SPOT" or "_track_" in sid else 1.0) * math.pi ** 2 * max(area, 1e-6))
        new = " ".join(f"{level * c:.4f}" for c in colour)
        return shape_text.replace(rad.group(0), rad.group(1) + new + rad.group(3), 1)

    out = re.sub(r'(?:\n  <shape type="([a-z_]+)" id="([^"]+)">)(.*?)(?:\n  </shape>)', lambda m: shape(m), text, flags=re.S)
    if housings:
        end = out.rindex("</scene>")
        out = out[:end] + "".join(housings) + "\n" + out[end:]
    return out


def scale_fill_text(text: str, gain: float) -> str:
    """The room softboxes' (``light_softbox_*``) emitter radiance times ``gain``, in a scene file's text."""
    def shape(m):
        if not m.group(2).startswith(FILL_PREFIX):
            return m.group(0)
        body = re.sub(r'(<rgb name="radiance" value=")([^"]+)(")',
                      lambda r: r.group(1) + " ".join(f"{float(v) * gain:.4f}" for v in r.group(2).split()) + r.group(3),
                      m.group(3))
        return m.group(1) + body + m.group(4)

    return re.sub(r'(\n  <shape type="[a-z_]+" id="([^"]+)">)(.*?)(\n  </shape>)', shape, text, flags=re.S)


ALPHA_CACHE = Path(os.environ.get("OPTICALNAV_ALPHA_CACHE", "~/.cache/opticalnav_sim/alpha_power")).expanduser()
ROUGH_BSDF = re.compile(r'(<bsdf type="(?:pplastic|roughconductor|roughdielectric|roughplastic)">)(.*?)(</bsdf>)', re.S)


def powered_alpha_texture(path: Path, power: float) -> Path:
    """A copy of a roughness texture with every value raised to ``power`` (16-bit grey PNG in ALPHA_CACHE,
    named by the source's file name: textures are content-addressed), written once."""
    out = ALPHA_CACHE / f"{path.stem}.p{power:g}.png"
    if not out.is_file():
        from PIL import Image

        with Image.open(path) as im:
            data = np.asarray(im)
        if data.ndim == 3:  # a grey map stored as RGB(A): the colour channels' mean
            data = data[..., :3].mean(axis=2)
        scale = 65535.0 if data.dtype == np.uint16 else 255.0
        grey = np.clip(np.rint((data.astype(np.float64) / scale) ** power * 65535.0), 0, 65535).astype(np.uint16)
        ALPHA_CACHE.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{os.getpid()}_{out.name}")
        Image.fromarray(grey).save(tmp)  # mode I;16
        os.replace(tmp, out)
    return out


def power_alpha_text(text: str, base_dir: Path, power: float = 2.0) -> str:
    """Blender's Principled roughness r is a microfacet alpha of r**2; the importer wrote r as the alpha, so
    every surface of its scenes came out twice as rough as in Blender (a floor of roughness 0.25 is alpha 0.25
    here, 0.0625 there: its window reflections smeared into a haze). The alpha of every rough plastic,
    conductor and dielectric of ``text``, a number or a roughness texture, raised to ``power``."""
    def block(m):
        body = re.sub(r'(<float name="alpha" value=")([-\d.eE]+)(")',
                      lambda a: f"{a.group(1)}{float(a.group(2)) ** power:.6f}{a.group(3)}", m.group(2))
        body = re.sub(r'(<texture name="alpha" type="bitmap">\s*<string name="filename" value=")([^"]+)(")',
                      lambda a: a.group(1) + str(powered_alpha_texture(base_dir / a.group(2), power)) + a.group(3), body)
        return m.group(1) + body + m.group(3)

    return ROUGH_BSDF.sub(block, text)


def scene_stats(mi, scene, text: str) -> dict:
    """What a loaded scene holds, for the viewer's scene tab: shapes, meshes, triangles, distinct BSDFs, image
    textures, emitters, and the scene text's size."""
    import drjit as dr

    shapes = scene.shapes()
    meshes = [sh for sh in shapes if sh.is_mesh()]
    try:
        bsdfs = len(np.unique(np.array(dr.reinterpret_array(mi.UInt32, scene.shapes_dr().bsdf()))))
    except Exception:  # noqa: BLE001 - a scalar variant has no pointer arrays
        bsdfs = None
    return {"shapes": len(shapes), "meshes": len(meshes), "triangles": int(sum(m.face_count() for m in meshes)),
            "vertices": int(sum(m.vertex_count() for m in meshes)), "bsdfs": bsdfs,
            "textures": len(set(re.findall(r'<texture\b[^>]*\btype="bitmap"[^>]*>\s*<string name="filename" value="([^"]+)"',
                                           text))),
            "emitters": len(scene.emitters()), "text_mb": round(len(text) / 1e6, 2)}


def gpu_memory() -> list[dict]:
    """Every GPU's memory in use (MB; all processes: WSL's nvidia-smi lists no per-process use)."""
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        index, name, used, total = (v.strip() for v in line.split(","))
        rows.append({"index": int(index), "name": name, "used_mb": int(used), "total_mb": int(total)})
    return rows


def light_of_key(key: str) -> str:
    """The light's id in a radiance (``<id>.emitter.radiance...``) or point light's (``<id>.intensity...``) parameter key."""
    return key.split(".emitter.radiance")[0].split(".intensity.")[0]


def emitter_radiance(text: str) -> dict[str, float]:
    """Shape id -> mean radiance of its area emitter, as the scene file gives it."""
    out = {}
    for m in re.finditer(r'<shape type="[a-z_]+" id="([^"]+)">(.*?)</shape>', text, re.S):
        e = re.search(r'<emitter type="[a-z_]+">(.*?)</emitter>', m.group(2), re.S)
        r = re.search(r'name="radiance" value="([^"]+)"', e.group(1)) if e else None
        if r:
            with contextlib.suppress(ValueError):
                out[m.group(1)] = float(np.mean([float(v) for v in re.split(r"[ ,]+", r.group(1).strip())]))
    return out


# Large light meshes cut into pieces (split_emitter_meshes) and lights weighted by their distance from the camera
# (Resident.aim_emitter_weights): PH33's light is 58 % three LED meshes spanning the building (27 x 26 m); one emitter
# each, a light sample went to a strip in another corridor most of the time. Measured on PH33 (16 spp, relative MSE
# against 4096 spp): 1.37 -> 0.81, the same mean (unbiased: only the choice of light changes).
SPLIT_EMITTERS = os.environ.get("OPTICALNAV_SPLIT_EMITTERS", "1") != "0"
VIEW_WEIGHTS = os.environ.get("OPTICALNAV_VIEW_EMITTER_WEIGHTS", "1") != "0"
SPLIT_MIN_M = 2.0      # an emitting mesh whose box diagonal is longer than this is cut ...
SPLIT_CELL_M = 1.0     # ... into pieces of a cell this big (its triangles first cut to edges this long)
REWEIGHT_MOVE_M = 1.0  # the camera this far from where the weights were made: made again
SPLIT_CACHE = Path(os.environ.get("OPTICALNAV_SIM_CACHE", Path.home() / ".cache" / "opticalnav_sim")) / "emitter_pieces"
_EMITTING_PLY = re.compile(r'(\n[ \t]*<!--opticalnav-obj:(\{[^\n]*\})-->)?\n([ \t]*)<shape type="ply" id="([^"]+)">(.*?)\n\3</shape>', re.S)


def _subdivide(v: np.ndarray, f: np.ndarray, edge: float) -> tuple[np.ndarray, np.ndarray]:
    """Triangles halved along their longest edge until no edge is longer than ``edge``."""
    v = [tuple(x) for x in v]
    todo, done = [tuple(x) for x in f], []
    while todo:
        a, b, c = todo.pop()
        P = (np.array(v[a]), np.array(v[b]), np.array(v[c]))
        lens = (np.linalg.norm(P[1] - P[0]), np.linalg.norm(P[2] - P[1]), np.linalg.norm(P[0] - P[2]))
        k = int(np.argmax(lens))
        if lens[k] <= edge or len(done) + len(todo) > 200_000:
            done.append((a, b, c))
            continue
        i, j, o = ((a, b, c), (b, c, a), (c, a, b))[k]
        v.append(tuple((np.array(v[i]) + np.array(v[j])) / 2.0))
        m = len(v) - 1
        todo += [(i, m, o), (m, j, o)]
    return np.array(v, np.float64), np.array(done, np.int64)


def split_emitter_meshes(mi, text: str, base_dir: Path) -> tuple[str, int]:
    """The scene text with every emitting PLY mesh longer than SPLIT_MIN_M (an LED strip along every corridor, a glow
    under every wall) written as pieces of SPLIT_CELL_M, each an emitter of its own (``<id>__e<k>``, its note's id
    changed, its geometry in world space): an emitter's own samples pick its triangles by area, wherever they are, so
    one mesh across a building sent most light samples away; as pieces, the emitter weights (aim_emitter_weights) pick
    the near ones. Only a constant radiance (a texture's uvs are not kept). The pieces are cached by the mesh file and
    its transform. Returns (text, pieces written)."""
    import hashlib
    count = 0

    def piece(m) -> str:
        nonlocal count
        note_raw, indent, sid, inner = m.group(2), m.group(3), m.group(4), m.group(5)
        if "<emitter" not in inner or "<texture" in inner.split("<emitter", 1)[1] or "<shape" in inner:
            return m.group(0)
        file = re.search(r'<string name="filename" value="([^"]+)"', inner)
        if file is None:
            return m.group(0)
        to_world = re.search(r'\s*<transform name="to_world"\s*>.*?</transform>', inner, re.S)
        path = Path(file.group(1)) if file.group(1).startswith("/") else base_dir / file.group(1)
        try:
            key = hashlib.sha1(f"{path.stat().st_size}|{path.stat().st_mtime_ns}|{to_world.group(0) if to_world else ''}|"
                               f"{SPLIT_MIN_M}|{SPLIT_CELL_M}".encode()).hexdigest()[:16]
        except OSError:
            return m.group(0)
        folder = SPLIT_CACHE / f"{path.stem}_{key}"
        listing = folder / "pieces.json"
        if listing.is_file():
            files = json.loads(listing.read_text())
        else:
            geo = f'<shape version="3.0.0" type="ply"><string name="filename" value="{path}" />' \
                  f'{to_world.group(0) if to_world else ""}</shape>'
            mesh = mi.load_string(geo)
            ps = mi.traverse(mesh)
            v = np.array(ps["vertex_positions"], np.float64).reshape(-1, 3)
            f = np.array(ps["faces"], np.int64).reshape(-1, 3)
            if not len(f) or np.linalg.norm(v.max(0) - v.min(0)) < SPLIT_MIN_M:
                files = []
            else:
                v, f = _subdivide(v, f, SPLIT_CELL_M)
                cell = np.floor(v[f].mean(1) / SPLIT_CELL_M).astype(np.int64)
                _, group = np.unique(cell, axis=0, return_inverse=True)
                folder.mkdir(parents=True, exist_ok=True)
                files = []
                for g in range(int(group.max()) + 1):
                    tri = f[group.reshape(-1) == g]
                    used, inv = np.unique(tri, return_inverse=True)
                    pm = mi.Mesh(f"{sid}_{g}", len(used), len(tri))
                    q = mi.traverse(pm)
                    q["vertex_positions"] = type(q["vertex_positions"])(v[used].astype(np.float32).ravel())
                    q["faces"] = type(q["faces"])(inv.reshape(-1).astype(np.uint32))
                    q.update()
                    out = folder / f"{g}.ply"
                    pm.write_ply(str(out))
                    files.append(str(out))
            listing.parent.mkdir(parents=True, exist_ok=True)
            listing.write_text(json.dumps(files))
        if len(files) < 2:
            return m.group(0)
        rest = re.sub(r'\s*<string name="filename" value="[^"]+"\s*/>', "", inner, count=1)
        if to_world:
            rest = rest.replace(to_world.group(0), "", 1)
        rest = re.sub(r'\s*<boolean name="face_normals"[^>]*/>', "", rest)
        out = ""
        for k, fp in enumerate(files):
            pid = f"{sid}__e{k}"
            note = ""
            if note_raw:
                with contextlib.suppress(ValueError):
                    n = json.loads(note_raw)
                    n["id"] = pid
                    note = f"\n{indent}<!--opticalnav-obj:{json.dumps(n, separators=(',', ':'))}-->"
            out += (f'{note}\n{indent}<shape type="ply" id="{pid}">\n{indent}    <string name="filename" value="{fp}" />'
                    f'\n{indent}    <boolean name="face_normals" value="true" />{rest}\n{indent}</shape>')
        count += len(files)
        return out

    return _EMITTING_PLY.sub(piece, text), count


LIGHT_GRID = os.environ.get("OPTICALNAV_LIGHT_GRID", "1") != "0"
GRID_CELL_M = 1.0        # the light grid's cells (larger in a large scene: at most GRID_MAX_CELLS)
GRID_MAX_CELLS = 16384    # x the emitters x 12 bytes (PH33: 289 emitters, 3136 cells, 11 MB)
GRID_MAX_ENTRIES = 8_000_000   # cells x emitters (96 MB of tables)
GRID_PAD_M = 2.0
GRID_WIDE_M = 25.0
GRID_POINT_R = 0.05   # a point or spot light's size in the grid's weights
GRID_MIX = 0.1           # each cell's pmf mixed with this share of the scene's own (no light starves behind a wall)
GRID_SKY_M = 3.0         # the environment weighed as a light this far away


def build_light_grid(mi, scene) -> dict | None:
    """path_nocaustics' spatial light selection (its grid_* parameters; Mitsuba build b7 and later): the scene's box
    in cells, each cell a pmf over the scene's emitters, their sampling weight (the power weights:
    emitter_power_weights; the flash's 0.001) over the squared distance from the cell's centre (at least the light's
    own size and half a cell), mixed with GRID_MIX of the plain weights, and its alias table. PH33: 16 spp light
    samples land on the near pieces of the LED strips instead of the building's (prototype: relative MSE 1.37 ->
    0.59 with the strips cut, the same mean). Returns the arrays by parameter name, or None (no or one emitter)."""
    import drjit as dr
    emitters = list(scene.emitters())
    n = len(emitters)
    if n < 2:
        return None
    shape_of = {id(s.emitter()): s for s in scene.shapes() if s.emitter() is not None}
    bb = scene.bbox()
    lo, hi = np.array(bb.min, np.float64).ravel(), np.array(bb.max, np.float64).ravel()
    base, centre, radius = np.zeros(n), np.zeros((n, 3)), np.zeros(n)
    spots: dict[int, tuple] = {}   # emitter index -> (axis, cutoff angle)
    for i, e in enumerate(emitters):
        base[i] = max(float(e.sampling_weight()), 0.0)
        s = shape_of.get(id(e))
        place = delta_light_place(mi, e) if s is None else None
        if place is not None:   # a point or spot light: where it is; a spot lights only its cone (below)
            centre[i], radius[i] = place[0], GRID_POINT_R
            if place[1] is not None:
                spots[i] = (place[1], math.radians(float(np.array(mi.traverse(e)["cutoff_angle"]).ravel()[0])))
        elif s is None or "__light" in s.id():   # the environment, and a body's lamp (it moves): everywhere
            centre[i], radius[i] = (lo + hi) / 2, -1.0
        else:
            b = s.bbox()
            a, z = np.array(b.min, np.float64).ravel(), np.array(b.max, np.float64).ravel()
            centre[i], radius[i] = (a + z) / 2, 0.5 * float(np.linalg.norm(z - a))
    # the grid spans the lights (and GRID_PAD_M around them), not the scene: a ground plane or the robot's far props
    # made PH33's cells 7.5 m; a point outside takes the nearest edge cell
    radius[radius > GRID_WIDE_M] = -1.0   # a light wider than this (a sky dome): everywhere, as the environment
    lit = radius >= 0
    if lit.any():
        lo = np.maximum(lo, (centre[lit] - radius[lit, None]).min(0) - GRID_PAD_M)
        hi = np.minimum(hi, (centre[lit] + radius[lit, None]).max(0) + GRID_PAD_M)
    ext = np.maximum(hi - lo, 1e-3)
    cell = GRID_CELL_M
    max_cells = max(64, min(GRID_MAX_CELLS, GRID_MAX_ENTRIES // n))
    while np.prod(np.maximum(1, np.ceil(ext / cell))) > max_cells:
        cell *= 1.25
    dims = np.maximum(1, np.ceil(ext / cell)).astype(np.int64)
    axes = [lo[a] + (np.arange(dims[a]) + 0.5) * cell for a in range(3)]
    zz, yy, xx = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    cells = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], -1)        # x fastest, as grid_cell
    d2 = ((cells[:, None, :] - centre[None]) ** 2).sum(-1)
    floor = np.where(radius < 0, GRID_SKY_M ** 2, radius ** 2 + (cell / 2) ** 2)
    w = base[None] / np.where(radius[None] < 0, floor[None], np.maximum(d2, floor[None]))
    for i, (axis, cut) in spots.items():   # a cell outside a spot's cone (widened by the cell's half diagonal): 2 %
        v = cells - centre[i]
        dist = np.maximum(np.linalg.norm(v, axis=1), 1e-6)
        ang = np.arccos(np.clip(v @ axis / dist, -1.0, 1.0))
        reach = cut + np.arcsin(np.minimum(1.0, 0.87 * cell / dist))
        w[:, i] *= np.where(ang <= reach, 1.0, 0.02)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-30)
    w = (1.0 - GRID_MIX) * w + GRID_MIX * (base / max(base.sum(), 1e-30))[None]
    pdf = w.astype(np.float64)
    # alias tables (Vose), every cell at once: n rounds, each settling one small entry per cell
    q = pdf * n
    prob, alias = np.ones_like(q), np.tile(np.arange(n), (len(q), 1))
    done = np.zeros_like(q, bool)
    rows = np.arange(len(q))
    for _ in range(n - 1):
        qs = np.where(done, np.inf, q)
        s_ = qs.argmin(1)
        ql = np.where(done, -np.inf, q)
        ql[rows, s_] = -np.inf
        l_ = ql.argmax(1)
        small = qs[rows, s_] < 1.0
        r = rows[small]
        prob[r, s_[small]] = q[r, s_[small]]
        alias[r, s_[small]] = l_[small]
        q[r, l_[small]] -= 1.0 - q[r, s_[small]]
        done[r, s_[small]] = True
        if not small.any():
            break
    ids = np.array(dr.reinterpret_array(mi.UInt32, scene.emitters_dr()), np.int64)
    lut = np.zeros(int(ids.max()) + 1, np.uint32)
    lut[ids] = np.arange(n, dtype=np.uint32)
    meta = [lo[0], lo[1], lo[2], cell, dims[0], dims[1], dims[2], n]
    return {"grid_meta": mi.Float(np.array(meta, np.float32)), "grid_pdf": mi.Float(pdf.astype(np.float32).ravel()),
            "grid_prob": mi.Float(np.clip(prob, 0, 1).astype(np.float32).ravel()),
            "grid_alias": mi.UInt32(alias.astype(np.uint32).ravel()), "grid_lut": mi.UInt32(lut),
            "cells": int(len(cells)), "cell_m": float(cell), "emitters": n, "everywhere": int((radius < 0).sum()),
            "span_m": [round(float(v), 1) for v in hi - lo]}


EMITTER_PLACES: dict = {}   # id(scene) -> {weight key: (centre, radius)}
EMITTER_POWER: dict = {}    # id(scene) -> {weight key: power}


def view_emitter_weights(power: dict, places: dict, eye: np.ndarray) -> dict:
    """Weights for an eye: each emitter's power over the square of its distance from the eye (at least its own radius
    and 1 m), averaging 1 (the environment keeps its share) with no light under 5 % of the mean."""
    raw = {}
    for k, p in power.items():
        c, r = places[k]
        raw[k] = p / max(float(np.linalg.norm(c - eye)), r, 1.0) ** 2
    mean = float(np.mean(list(raw.values())))
    return {k: max(w, 0.05 * mean) / mean for k, w in raw.items()}


def delta_light_place(mi, emitter):
    """(position, shining axis, solid angle) of a point or spot light (point_light_xml), else None."""
    kind = str(emitter).split("[", 1)[0].strip()
    if kind not in ("PointLight", "SpotLight"):
        return None
    pe = mi.traverse(emitter)
    if kind == "PointLight":
        pos = np.array(pe["position"], np.float64).ravel()[:3] if "position" in pe else \
            np.array(pe["to_world"].matrix, np.float64).reshape(4, 4)[:3, 3]
        return pos, None, 4.0 * math.pi
    m = np.array(pe["to_world"].matrix, np.float64).reshape(4, 4)
    cut = math.radians(float(np.array(pe["cutoff_angle"]).ravel()[0]))
    beam = math.radians(float(np.array(pe["beam_width"]).ravel()[0]))
    # the ramp from beam to cutoff counted at half: about the falloff's mean there
    half = 0.5 * (cut + beam)
    return m[:3, 3], m[:3, 2] / max(float(np.linalg.norm(m[:3, 2])), 1e-12), 2.0 * math.pi * (1.0 - math.cos(half))


def emitter_power_weights(mi, scene, radiance: dict[str, float]) -> dict[str, float]:
    """Pick the area emitters for next-event estimation in proportion to their power (radiance x area) instead of
    uniformly (Mitsuba's default, every ``sampling_weight`` 1): with 20-30 lights through an apartment, most
    light samples went to dim or distant ones and the rare one that met a bright lamp nearby was a speck. Unbiased
    (the pdf follows the weights). Weights average 1, so the environment (weight 1) keeps its share; a weight is
    at least 5% of the mean (no light starves); an emitter set otherwise (the flash: 0.001) keeps its own. Set
    once before any freeze recording. Returns parameter key -> weight."""
    import drjit as dr

    params = mi.traverse(scene)
    power = {}
    for shape in scene.shapes():
        if shape.emitter() is None or shape.id() not in radiance:
            continue
        key = f"{shape.id()}.emitter.sampling_weight"
        if key in params and float(params[key]) == 1.0:
            power[key] = radiance[shape.id()] * float(dr.ravel(shape.surface_area())[0])
            b = shape.bbox()
            lo, hi = np.array(b.min, np.float64).ravel(), np.array(b.max, np.float64).ravel()
            EMITTER_PLACES.setdefault(id(scene), {})[key] = ((lo + hi) / 2.0, 0.5 * float(np.linalg.norm(hi - lo)))
    for e in scene.emitters():   # point and spot lights (point_light_xml): power / pi = intensity x solid angle / pi
        place = delta_light_place(mi, e)
        key = f"{e.id()}.sampling_weight"
        if place is None or key not in params or float(params[key]) != 1.0:
            continue
        pe = mi.traverse(e)
        ik = next((k for k in ("intensity.nested.value", "intensity.value") if k in pe), None)
        level = float(np.mean(np.array(pe[ik]).ravel()[:3])) if ik else 1.0
        power[key] = level * place[2] / math.pi
        EMITTER_PLACES.setdefault(id(scene), {})[key] = (place[0], 0.0)
    if len(power) < 2:
        return {}
    mean = float(np.mean(list(power.values())))
    weights = {k: max(p, 0.05 * mean) / mean for k, p in power.items()}
    EMITTER_POWER[id(scene)] = power
    for k, w in weights.items():
        params[k] = w
    params.update()
    return weights


def light_groups(text: str) -> dict[str, str]:
    """Shape id -> 'fixture' (Infinigen's own lights, converted to emitter cubes or ceiling panels at their
    lamp and ceiling-light meshes) or 'fill' (room softboxes robomituba's importer synthesizes to light rooms
    evenly: apps/import_infinigen_scene.py, label light:softbox:...)."""
    out = {}
    for note in re.findall(r"<!--opticalnav-obj:(\{.*?\})-->", text):
        try:
            obj = json.loads(note)
        except ValueError:
            continue
        if obj.get("is_emitter"):
            meta = obj.get("metadata") or {}
            if str(obj.get("label", "")).startswith("light:lamp:"):  # added to a dark lamp (lamp_lights_text)
                out[obj["id"]] = "lamp"
                continue
            fill = str(obj.get("label", "")).startswith("light:softbox") or meta.get("softbox_proxy") or meta.get("synthesized")
            out[obj["id"]] = "fill" if fill else "fixture"
    return out


def light_classes(text: str) -> dict[str, str]:
    """Shape id -> light class ('led' or 'incandescent') from the scene file's opticalnav-obj notes: a
    PointLampFactory lamp without a panel shape is a bulb (incandescent); ceiling panels and softboxes are
    LED. Lights without a note count as LED."""
    out = {}
    for note in re.findall(r"<!--opticalnav-obj:(\{.*?\})-->", text):
        try:
            obj = json.loads(note)
        except ValueError:
            continue
        if obj.get("is_emitter"):
            label = str(obj.get("label", ""))
            bulb = label.startswith("light:PointLampFactory") and not obj.get("emitter_shape")
            if label.startswith(("light:lamp:", "light:fixture:")):  # a lamp's glowing part: a bulb, unless a ceiling lamp
                bulb = "Ceiling" not in label and "Chandelier" not in label
            out[obj["id"]] = "incandescent" if bulb else "led"
    return out


SCENE_SKY = (0.55, 0.57, 0.60)  # the constant sky of every scene file (environment: constant)
ENV_RES = (1024, 512)  # every environment map is resampled to this, so swapping maps never re-records
DUAL_MARGIN = 3  # pixels a side each slot of a dual (visible | near-infrared) render has to spare at the seam
_BLANK_ENV: list[Path] = []


def blank_env_file(mi) -> Path:
    """An ENV_RES image file to create envmap emitters from (their pixels are written after loading)."""
    if not _BLANK_ENV:
        import tempfile

        path = Path(tempfile.gettempdir()) / f"opticalnav_env_{ENV_RES[0]}x{ENV_RES[1]}.exr"
        mi.Bitmap(np.ones((ENV_RES[1], ENV_RES[0], 3), np.float32)).write(str(path))
        _BLANK_ENV.append(path)
    return _BLANK_ENV[0]
ENV_ID = "environment"
ENV_EMITTER = re.compile(r'<emitter type="(constant|envmap)"\s*>(.*?)</emitter>', re.S)


@dataclass(frozen=True)
class Look:
    """What a frame shows besides the camera and the variant's passes. All of it is scene data, changed in
    GPU memory between frames, never part of a freeze recording."""
    hidden: frozenset = frozenset()  # shape ids left out (base: perturbed's mirror and glass cubes)
    lights: float = 1.0              # multiplier on every indoor emitter (the camera flash aside)
    env: str = "original"            # environment map name, or the scene's own sky
    env_scale: float = 1.0
    env_rotate_deg: float = 0.0
    band: int = 0                    # 0 visible, 1 pseudo near-infrared, 2 both in one render (dual sensor)
    nir: tuple = tuple(sorted(NIR_DEFAULTS.items()))  # near-infrared radiance per light class (and daylight)
    flash_rgb: bool = True           # the camera flash emits in the visible band
    flash_nir: bool = False          # ... and in the near-infrared band
    groups: tuple = ("fixture", "fill")  # light groups on (lighting mode); the others are dark and removed
    flash_scale: float = 1.0         # the flash's side x this, its radiance / this^2: the same power (fewer fireflies)
    flash_pdf: float = 20.0          # the flash pass's near-specular cut (path_nocaustics specular_pdf_threshold)
    max_depth: int = 0               # path length cap of every pass (Mitsuba's max_depth); 0: the scene's own
    rr_depth: int = 0                # the depth Russian roulette starts at (Mitsuba's rr_depth); 0: Mitsuba's 5
    caustics: bool = True            # the lights' passes keep caustics (light reaching an emitter past a near-specular bounce)
    split: bool = False              # the active pass as one path_nocaustics "split" path, not a passive and a flash path
    nir_range: tuple = (0.5, 1.0)    # pseudo near-infrared reflectance mapped onto this (rgb2nir low, high): 0.5..1 as is
    flash_offset: tuple = (0.0, 0.0, 0.0)  # the flash moved this far (m): right, up, forward of the level view
    unpolarized: bool = False        # a polarized scene rendered without polarization (path_nocaustics unpolarized)

    def light_level(self, light_class: str) -> float:
        """Indoor emitter radiance factor for a light class in this band (scenes without bandscale lights)."""
        return self.lights * (1.0 if self.band != 1 else dict(self.nir).get(light_class, 0.0))

    def sky_level(self) -> float:
        """The sky's scale: near-infrared only renders scale it by daylight's share; in a dual render the
        integrator scales the near-infrared slot's sky instead (path_nocaustics sky_nir)."""
        return self.env_scale * (1.0 if self.band != 1 else dict(self.nir).get("daylight", 0.0))

    def uniforms(self) -> dict[str, float]:
        """Values of the shared per-band light levels (band_lights_text)."""
        nir, out = dict(self.nir), {}
        for g in LIGHT_GROUPS:
            on = self.lights if g in self.groups else 0.0
            out[f"light_rgb_{g}"] = on
            for c in ("led", "incandescent"):
                out[f"light_nir_{g}_{c}"] = on * nir.get(c, 0.0)
        power = 1.0 / (self.flash_scale * self.flash_scale)  # a larger flash at the same power
        out.update(flash_rgb=float(self.flash_rgb) * power, flash_nir=float(self.flash_nir) * power,
                   nir_band=float(self.band == 1), nir_low=float(self.nir_range[0]), nir_high=float(self.nir_range[1]))
        return out


def env_data(mi, path: Path, scale: float = 1.0, sun_deg: float = 0.0) -> np.ndarray:
    """An HDR environment image as linear RGB float32 at ENV_RES (height x width x 3), its sun spread over
    ``sun_deg`` degrees when given (see soften_sun)."""
    bitmap = mi.Bitmap(str(path)).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=False)
    data = np.array(bitmap.resample([ENV_RES[0], ENV_RES[1]]), dtype=np.float32) * scale
    return np.ascontiguousarray(soften_sun(data, sun_deg) if sun_deg > 0 else data)


def soften_sun(data: np.ndarray, sigma_deg: float, threshold: float = 50.0) -> np.ndarray:
    """Spread each tiny, very bright spot of an equirectangular environment (a sun) over a Gaussian of
    ``sigma_deg`` degrees, keeping its power and colour.

    Indoors, sunlight reaches a wall through window glass only along delta (glass) paths, which emitter
    sampling cannot trace: a path finds the sun only by chance, and a 0.5 degree sun carrying most of a
    map's light turns those rare hits into fireflies. A sun spread over a few degrees is hit hundreds of
    times more often at a matching fraction of the radiance, so the same light arrives with far less
    noise; sun shadows soften. Pixels brighter than ``threshold`` x the median luminance give up their
    excess; skies without such spots (overcast) are unchanged."""
    h, w, _ = data.shape
    lum = data @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    limit = threshold * max(float(np.median(lum)), 1e-6)
    hot = lum > limit
    if not hot.any():
        return data
    theta = (np.arange(h) + 0.5) / h * np.pi
    phi = (np.arange(w) + 0.5) / w * 2.0 * np.pi
    st, ct, sp, cp = np.sin(theta)[:, None], np.cos(theta)[:, None], np.sin(phi)[None, :], np.cos(phi)[None, :]
    dirs = np.stack([st * sp, np.broadcast_to(ct, (h, w)), -st * cp], axis=-1)  # any consistent sphere map
    area = np.broadcast_to(st, (h, w))  # solid angle per pixel, up to a constant
    out = data.copy()
    excess = np.where(hot[..., None], data * (1.0 - limit / np.maximum(lum, 1e-12))[..., None], 0.0)
    out -= excess
    # group the hot pixels into spots (greedy, by angle), then lay each spot's power down as a Gaussian
    ys, xs = np.nonzero(hot)
    order = np.argsort(-(lum[ys, xs] * area[ys, xs]))
    spots: list[list[int]] = []
    centres: list[np.ndarray] = []
    for i in order:
        d = dirs[ys[i], xs[i]]
        j = next((j for j, c in enumerate(centres) if float(c @ d) > np.cos(np.radians(10.0))), None)
        if j is None:
            spots.append([i]); centres.append(d)
        else:
            spots[j].append(i)
    sigma = np.radians(sigma_deg)
    for members in spots:
        my, mx = ys[members], xs[members]
        power = (excess[my, mx] * area[my, mx][:, None]).sum(axis=0)  # rgb
        weight = (lum[my, mx] * area[my, mx])[:, None]
        centre = (dirs[my, mx] * weight).sum(axis=0)
        centre /= np.linalg.norm(centre)
        angle = np.arccos(np.clip(dirs @ centre, -1.0, 1.0))
        kernel = np.exp(-0.5 * (angle / sigma) ** 2)
        out += kernel[..., None] * (power / (kernel * area).sum())[None, None, :]
    return out.astype(np.float32)


def controllable_env(mi, text: str, base: Path, blank: Path) -> tuple[str, np.ndarray | None]:
    """Replace the scene's environment emitter (a constant sky, or an envmap) with an envmap at ENV_RES
    under id ENV_ID, and return the image that reproduces the original. A constant becomes a uniform image
    (the same radiance from every direction); a scaled envmap file becomes its resampled pixels times the
    scale. Scenes with neither keep their text and get no environment control (None)."""
    found = ENV_EMITTER.search(text)
    if found is None:
        return text, None
    body = found.group(2)
    if found.group(1) == "constant":
        rgb = re.search(r'name="radiance" value="([^"]+)"', body)
        value = [float(v) for v in re.split(r"[ ,]+", rgb.group(1).strip())] if rgb else [1.0]
        original = np.empty((ENV_RES[1], ENV_RES[0], 3), np.float32)
        original[...] = np.array(value * 3 if len(value) == 1 else value, np.float32)
    else:
        file = re.search(r'name="filename" value="([^"]+)"', body).group(1)
        scale = re.search(r'name="scale" value="([^"]+)"', body)
        if re.search(r'name="to_world"', body):
            return text, None  # a rotated original is not reproduced; leave the scene as it is
        original = env_data(mi, base / file, float(scale.group(1)) if scale else 1.0)
    emitter = f'<emitter type="envmap" id="{ENV_ID}"><string name="filename" value="{blank}" /></emitter>'
    return text[:found.start()] + emitter + text[found.end():], original


def shape_vertices(mi, text: str, shape_id: str) -> np.ndarray:
    """World-space vertex positions (N x 3) of one top-level shape of a scene file, loaded on its own: the
    same plugin code the scene load runs, so the values match the loaded scene's bit for bit."""
    found = re.search(rf'\n  <shape type="([a-z_]+)" id="{re.escape(shape_id)}">(.*?)\n  </shape>', text, re.S)
    if found is None:
        raise RuntimeError(f"shape {shape_id!r} not found in the scene file")
    # geometry only: materials, emitters and references to other objects stay out of a shape loaded alone.
    # BSDFs nest (twosided > normalmap > pplastic): remove the innermost ones until none is left
    body = re.sub(r'<ref\b[^>]*/>|<bsdf\b[^>]*/>|<emitter\b.*?</emitter>', "", found.group(2), flags=re.S)
    while True:
        stripped = re.sub(r'<bsdf\b(?:(?!<bsdf\b).)*?</bsdf>', "", body, flags=re.S)
        if stripped == body:
            break
        body = stripped
    shape = mi.load_string(f'<shape version="3.0.0" type="{found.group(1)}">{body}<bsdf type="diffuse"/></shape>')
    return np.array(mi.traverse(shape)["vertex_positions"], dtype=np.float32).reshape(-1, 3)


OBJECT_GROUP_PREFIX, OBJECT_INSTANCE_PREFIX = "objgrp_", "objinst_"
# Pushed objects as instances (object_instances_text) in a scene loaded with bodies; 0: their meshes' vertices rewritten
OBJECT_INSTANCES = os.environ.get("OPTICALNAV_OBJECT_INSTANCES", "1") != "0"


def object_instances_text(text: str, objects: dict) -> tuple[str, dict]:
    """``text`` with each object's dyn_ meshes (``objects``: physics object id -> shape ids) moved into a shapegroup of its
    own and drawn through an instance, at rest where they were. Returns (text, object id -> instance id). Moving an
    object then writes its instance's transform alone: the top-level structure is rebuilt, no mesh is (rewriting a
    mesh's vertices cost ~0.4 ms a mesh, a GPU sync each for its bounding box, 21 ms for CS02's 14 objects, 63 meshes).
    A shape not found stays as it is."""
    groups, out, n = "", {}, 0
    for oid, sids in sorted(objects.items()):
        children = ""
        for sid in sids:
            m = re.search(rf'\n[ \t]*(?:<!--opticalnav-obj:[^\n]*-->\n[ \t]*)?<shape type="[a-z]+" id="{re.escape(sid)}">', text)
            if m is None:
                continue
            end = text.find("</shape>", m.end())
            if end < 0 or "<shape" in text[m.end():end]:   # a nested shape: not a plain mesh, left alone
                continue
            end += len("</shape>")
            element = text[m.start():end]
            element = re.sub(r'^\s*<!--opticalnav-obj:[^\n]*-->', "", element)   # the note stays out of the group
            children += element
            text = text[:m.start()] + text[end:]
        if not children:
            continue
        gid, iid = f"{OBJECT_GROUP_PREFIX}{n}", f"{OBJECT_INSTANCE_PREFIX}{n}"
        groups += (f'\n  <shape type="shapegroup" id="{gid}">{children}\n  </shape>'
                   f'\n  <shape type="instance" id="{iid}">\n    <ref id="{gid}" />\n  </shape>')
        out[oid] = iid
        n += 1
    if groups:
        end = text.rindex("</scene>")
        text = text[:end] + groups + "\n" + text[end:]
    return text, out


def instance_refs(text: str) -> dict:
    """Each instance shape's id -> the id of the shapegroup it draws (its ``<ref id>``)."""
    out = {}
    for m in re.finditer(r'<shape\b[^>]*\btype="instance"[^>]*>', text):
        sid = re.search(r'\bid="([^"]*)"', m.group(0))
        end = text.find("</shape>", m.end())
        ref = re.search(r'<ref\b[^>]*\bid="([^"]*)"', text[m.end():end if end >= 0 else None])
        if sid and ref:
            out[sid.group(1)] = ref.group(1)
    return out


def locate_blocks(meshes: dict, blocks: dict) -> dict:
    """Where each vertex block sits among the scene's meshes: {name: (mesh key, first vertex, count)}.
    The loader merges meshes that share a BSDF into one mesh by appending their vertices, so a shape's
    vertices are one contiguous run in some mesh (its own, if nothing merged with it)."""
    out = {}
    for name, block in blocks.items():
        for key, verts in meshes.items():
            starts = np.flatnonzero(np.all(verts == block[0], axis=1))
            hit = next((int(s) for s in starts if s + len(block) <= len(verts)
                        and np.array_equal(verts[s:s + len(block)], block)), None)
            if hit is not None:
                out[name] = (key, hit, len(block))
                break
        else:
            raise RuntimeError(f"shape {name!r}: its vertices were not found in the loaded scene")
    return out


_RASTER_MTIME: list[float] = []


def _raster_module():
    """raster.py, imported when first asked for (torch, nvdiffrast); with OPTICALNAV_RASTER_RELOAD=1 reloaded
    whenever it or raster_gi.py changed, and each resident then builds its preview again (for working on it live)."""
    from . import raster, raster_gi
    if os.environ.get("OPTICALNAV_RASTER_RELOAD"):
        import importlib
        mtime = max(Path(m.__file__).stat().st_mtime for m in (raster, raster_gi))
        if _RASTER_MTIME and _RASTER_MTIME[0] != mtime:
            importlib.reload(raster_gi)
            raster = importlib.reload(raster)
        _RASTER_MTIME[:] = [mtime]
    return raster


def integrator_spec(xml: Path) -> dict:
    """max_depth and Stokes reference axis of a scene file's integrator."""
    text = xml.read_text()
    start = text.find("<integrator")
    head = text[start:text.find("<sensor", start)] if start >= 0 else ""
    depth = re.search(r'name="max_depth" value="(-?\d+)"', head)
    axis = re.search(r'name="reference_axis" x="([-\d.e]+)" y="([-\d.e]+)" z="([-\d.e]+)"', head)
    return {"max_depth": int(depth.group(1)) if depth else 6,
            "reference_axis": tuple(float(a) for a in axis.groups()) if axis else None}


class Resident:
    """One loaded Mitsuba scene, driven only through mi.traverse() between renders.

    A scene with camera-assist shapes (an active variant's flash) renders several passes: "own" (the
    scene's integrator), "passive" (the assist shapes ignored, as if absent: the lights-only image) and
    "flash" (the flash as the only light, for two-pass active variants). The passes need
    path_nocaustics's assist modes; switching the flash on or off then reuses the loaded scene.

    Shapes a variant lacks (base has none of perturbed's mirror and glass cubes; ``hideable``) are hidden by
    collapsing their vertices to one point, which leaves only degenerate triangles that no ray or shadow
    meets, and restored from a copy: a variant switch edits memory and loads nothing.

    The rest of a frame's Look is scene data too: the indoor emitters' radiance (scaled from their loaded
    values) and the environment (an envmap at ENV_RES standing in for the scene's own sky, whose pixels,
    scale and rotation change in place). Each is written once as an opaque value before the first
    recording, so later changes replay the same recording."""

    hidden_specular = False  # the plugin knows ``hidden_specular`` (set by the server at start)

    def __init__(self, mi, xml: Path, mode: str, flash: dict | None = None, freeze: bool = False,
                 flash_xml: Path | None = None, hideable=(), envs=None, freeze_spp: int = 0, nir: bool = False,
                 pool: dict | None = None, upgrade_flash: bool = False, hidden: str = "", fill_gain: float = 1.0,
                 blender_power: bool = False, alpha_power: float = 1.0, watt_gain: float = 1.0, bodies: str = "",
                 objects: dict | None = None):
        self.mi, self.mode = mi, mode
        # an RGBN variant: every render holds both bands (R, G, B lanes and the near-infrared lane)
        self.rgbn = "_rgbn" in mi.variant()
        started = time.monotonic()
        text = lit_scene_text(xml, flash_xml) if flash_xml is not None else xml.read_text()
        if bodies:  # robots and their equipment (bodies.scene_xml): instances a view places (set_bodies)
            end = text.rindex("</scene>")
            text = text[:end] + bodies + text[end:]
        # the objects a robot can push (physics.json, kept apart by tools/merge_meshes.py), each a shapegroup of its dyn_
        # meshes and an instance: pushed, only the instance's transform is written (set_objects)
        text, self.object_groups = object_instances_text(text, objects or {})
        self.hidden = hidden  # shape id prefix camera rays pass through (path_nocaustics "hidden"; FILL_PREFIX)
        self.upgraded_flash = False  # an old one-pass flash rendered as the current protocol (upgrade_flash_text)
        if flash is not None and flash.get("synthetic") and flash_xml is None:
            # a scene packed without an active variant: the dataset's flash added, rendered as the current protocol
            text, self.upgraded_flash = flash_scene_text(text, flash), True
        elif upgrade_flash and flash is not None and flash_xml is None:
            text, self.upgraded_flash = upgrade_flash_text(text, float(flash.get("polarizer_angle_deg") or 0.0))
        if mode == "rgb" and "_rgbn" not in mi.variant():   # an RGBN variant keeps the stokes wrapper: its AOVs hold the NIR lane
            text = rgb_scene_text(text)
        base_dir = xml.parent  # dark lamps' shades glow (group "lamp", dark by default)

        def mesh_info(kind: str, filename: str) -> tuple[float, float]:
            mesh = mi.load_dict({"type": kind, "filename": str(base_dir / filename)})
            return float(dr.ravel(mesh.surface_area())[0]), float(mesh.bbox().min.y)

        def mesh_bounds(kind: str, body: str):
            """A part's world bounding box: the shape loaded alone with its transform (as shape_vertices does)."""
            # the material and light out (a BSDF nests: twosided around roughconductor, so first <bsdf to last </bsdf>)
            body = re.sub(r'<ref\b[^>]*/>|<bsdf\b.*</bsdf>|<bsdf\b[^>]*/>|<emitter\b.*?</emitter>', "", body, flags=re.S)
            body = re.sub(r'(<string name="filename" value=")(?!/)', rf'\g<1>{base_dir}/', body)
            box = mi.load_string(f'<shape version="3.0.0" type="{kind}">{body}<bsdf type="diffuse"/></shape>').bbox()
            return [float(v) for v in dr.ravel(box.min)], [float(v) for v in dr.ravel(box.max)]

        import drjit as dr
        # OPTICALNAV_BOX_LIGHTS=1: lit lamps keep the importer's boxes (the earlier look, for comparisons)
        text, self.lamp_lights = lamp_lights_text(text, mesh_info, lit=not os.environ.get("OPTICALNAV_BOX_LIGHTS"),
                                                  mesh_bounds=mesh_bounds, blender_power=blender_power, watt_gain=watt_gain)
        if alpha_power != 1.0:  # the simulator's own scenes: the importer's roughness as Blender's alpha (r**2)
            text = power_alpha_text(text, base_dir, alpha_power)
        if fill_gain != 1.0:  # the simulator's own scenes: the importer's room softboxes at a share of their radiance
            text = scale_fill_text(text, fill_gain)
        self.emitter_pieces = 0
        if SPLIT_EMITTERS:   # long light meshes as pieces (split_emitter_meshes): their light samples go to the near ones
            text, self.emitter_pieces = split_emitter_meshes(mi, text, base_dir)
        radiance = emitter_radiance(text)  # before the band transforms wrap them
        self.classes = light_classes(text)  # shape id -> 'led' | 'incandescent'
        self.groups = light_groups(text)  # shape id -> 'fixture' | 'fill' | 'lamp' (lighting modes remove whole groups)
        text = lights_as_meshes(text, self.groups)
        hideable = set(hideable) | {s for s in self.groups if re.search(rf'<shape type="[a-z_]+" id="{re.escape(s)}">', text)}
        # the scene's own perturbation (the dataset's mirrors and glass walls), hidden when a pool is used
        self.dataset_perturbation = {s for s in hideable if s.startswith(("mirror_auto_", "glass_auto_"))}
        # a pool of candidate mirrors and glass walls (tools/build_perturbation_pool.py), loaded hidden-able:
        # a view shows a density's share of it (see Renderer._pool_hidden)
        self.pool = pool
        if pool:
            end = text.rindex("</scene>")
            shapes = "".join(m["xml"] for m in pool["mirrors"]) + "".join(g["xml"] for g in pool["glass"])
            text = text[:end] + pool["bsdf_xml"] + shapes + "\n" + text[end:]
            hideable |= {m["id"] for m in pool["mirrors"]} | {g["id"] for g in pool["glass"]}
        if nir:  # base colours behind one visible / near-infrared switch (rgb2nir), lights per band (bandscale)
            text = band_lights_text(nir_scene_text(text), self.classes, self.groups)
        text, self.env_original = controllable_env(mi, text, xml.parent, blank_env_file(mi))
        if not os.environ.get("OPTICALNAV_KEEP_UNREFERENCED"):  # after every transform: they add referenced ids
            size = len(text)
            text, dropped = strip_unreferenced(text, keep={NIR_BAND_ID, *BAND_UNIFORMS})
            print(f"[resident] {dropped} unreferenced top-level bsdf/texture elements left out "
                  f"({(size - len(text)) / 1e6:.1f} MB of scene text)", flush=True)
        mi.Thread.thread().file_resolver().prepend(str(xml.parent))  # assets are relative to the XML
        self.name = f"{xml.parent.name}/{xml.name}" + (f"+{flash_xml.name}" if flash_xml is not None else "")
        self.score = progress.scene_score(text)  # sizes the recording-time estimate before any history
        text = film_filter(text)
        self.instance_groups = instance_refs(text)   # instance id -> its shapegroup's id: the raster preview draws their meshes
        self.geometry_epoch = 0   # counts moves of bodies and pushed objects (set_bodies, set_objects): the raster preview follows
        with progress.watch_load(mi, text, self.name):
            self.scene = mi.load_string(text)
        # OPTICALNAV_UNIFORM_EMITTERS=1: Mitsuba's default, every emitter picked as often (for comparisons)
        self.emitter_weights = {} if os.environ.get("OPTICALNAV_UNIFORM_EMITTERS") else \
            emitter_power_weights(mi, self.scene, radiance)
        self.weights_eye = None   # where the emitter weights were last aimed from (aim_emitter_weights)
        self.light_grid = build_light_grid(mi, self.scene) if LIGHT_GRID and self.emitter_weights else None
        if self.light_grid:
            g = self.light_grid
            print(f"[resident] light grid: {g['cells']} cells of {g['cell_m']:.2f} m over {g['emitters']} emitters "
                  f"({g['everywhere']} everywhere), span {g['span_m']} m", flush=True)
        self.load_s = time.monotonic() - started
        self.stats = scene_stats(mi, self.scene, text)  # the viewer's scene tab
        self.envs = envs  # name -> environment image at ENV_RES (Renderer.env_image)
        self.depth = {"own": integrator_spec(xml), "passive": integrator_spec(xml)}
        if flash_xml is not None:
            self.depth["flash"] = integrator_spec(flash_xml)
        elif self.upgraded_flash:  # the flash pass of an upgraded one-pass scene: its own integrator's depth
            self.depth["flash"] = integrator_spec(xml)
        self.integrators: dict[str, object] = {}
        keys = list(mi.traverse(self.scene).keys())
        self.k_to_world = next(k for k in keys if k.endswith("sensor.to_world"))
        self.k_fov = next(k for k in keys if k.endswith("sensor.x_fov"))
        self.k_size = next((k for k in keys if k.endswith("sensor.film.size")), None)
        self.flash = flash
        self.k_flash = next((k for k in keys if k.startswith("camera_assist_light") and k.endswith("to_world")), None)
        self.k_polarizer = next((k for k in keys if k.startswith("camera_assist_polarizer") and k.endswith("to_world")), None)
        # transforms a frame may write (flash, lights kept as rectangles, environment): see _evaluate_transforms
        self.k_transforms = [k for k in keys if k.endswith(".to_world") and k != self.k_to_world]
        if flash is not None and self.k_flash is None:
            raise RuntimeError(f"{xml}: flash scene has no camera_assist_light transform")
        # Camera-only updates traverse just the sensor: traversing and updating the whole scene (thousands
        # of textures and meshes) cost ~46 ms per frame on Device 1. A frame with the flash on also moves
        # the flash, written through the assist shapes alone (see _write_assist); a frame that changes the
        # Look takes the whole-scene path; with the flash off the assist shapes are parked once.
        self.sensor = self.scene.sensors()[0]
        self.assist_shapes = {s.id(): s for s in self.scene.shapes() if s.id().startswith(ASSIST_PREFIX)}
        self.body_shapes = {s.id(): s for s in self.scene.shapes() if s.id().startswith(body_models.BODY_PREFIX)}
        # physics object id -> its instance (object_instances_text), its transform written once now (evaluated: the
        # frozen renders' input layout is the same before and after the first push)
        by_id = {s.id(): s for s in self.scene.shapes() if s.id().startswith(OBJECT_INSTANCE_PREFIX)}
        self.object_instances = {oid: by_id[iid] for oid, iid in self.object_groups.items() if iid in by_id}
        if self.object_instances:
            import drjit as dr
            for inst in self.object_instances.values():
                ps = mi.traverse(inst)
                ps["to_world"] = mi.ScalarTransform4f()
                ps.update()
                dr.make_opaque(ps["to_world"])
            self.scene.parameters_changed([])
        self.body_pose: dict[str, bytes] = {}  # instance id -> the matrix last written
        sensor_keys = set(mi.traverse(self.sensor).keys())
        self.ks_size = "film.size" if "film.size" in sensor_keys else None
        self.size = self.fov = None
        self.pose = None  # (camera, size, fov, follow, look) last written to the scene
        self.parked = self.k_flash is None  # assist shapes out of the way (tiny), or none
        self.look = Look()  # what the scene shows now
        self.blocks: dict = {}  # shape id -> (mesh key, first vertex, count) in the loaded (merged) meshes
        self.transforms: dict[str, np.ndarray] = {}  # shape id -> loaded to_world, for shapes hidden by shrinking
        self.originals: dict[str, np.ndarray] = {}  # vertex positions (N x 3) of the meshes holding them
        if hideable:
            t = time.monotonic()
            source = text  # the scene as loaded (with any pool shapes)
            params = mi.traverse(self.scene)
            meshes = {k: np.array(params[k], dtype=np.float32).reshape(-1, 3)
                      for k in keys if k.endswith(".vertex_positions")}
            # shapes kept as a transform (rectangles: ceiling panels, softboxes) shrink in place instead
            self.transforms = {s: np.array(params[f"{s}.to_world"].matrix, dtype=np.float64).reshape(4, 4)
                               for s in sorted(hideable) if f"{s}.to_world" in params}
            self.blocks = locate_blocks(meshes, {s: shape_vertices(mi, source, s) for s in sorted(hideable)
                                                 if s not in self.transforms})
            self.originals = {key: meshes[key] for key, _, _ in self.blocks.values()}
            del meshes, params
            print(f"[resident] {len(self.blocks)} hideable shapes in {len(self.originals)} meshes "
                  f"({sum(len(v) for v in self.originals.values())} vertices kept, {time.monotonic() - t:.1f} s)", flush=True)
        # Indoor emitters: every emitter radiance but the camera flash's. Written back once (opaque) here.
        import drjit as dr

        params = mi.traverse(self.scene)
        # (type, loaded values) copied to the host: a reference to the parameter would alias the emitter's own
        # buffer, which the first write (lights off: 0) overwrites
        # plain, or behind bandscale; a point or spot light (point_light_xml) has an intensity instead
        self.delta_lights = {m.group(2) for m in DELTA_LIGHT.finditer(text)}
        emitted = (".emitter.radiance.value", ".emitter.radiance.nested.value")
        self.radiance = {k: (type(params[k]), np.array(params[k], dtype=np.float64).ravel()) for k in keys
                         if (k.endswith(emitted) or (k.split(".intensity.")[0] in self.delta_lights and
                                                     k.endswith((".intensity.value", ".intensity.nested.value"))))
                         and not k.startswith(ASSIST_PREFIX)}
        self.light_class = {k: self.classes.get(light_of_key(k), "led") for k in self.radiance}
        # per-band light levels as shared uniforms (band_lights_text): written once per change, read per ray
        tags = {uniform_tag(u): u for u in BAND_UNIFORMS}
        self.band_keys = {}
        for k in keys:
            if k.endswith(".value") and not k.endswith(emitted):
                v = np.array(params[k], dtype=np.float64).ravel()
                if v.size == 1 and round(float(v[0]), 6) in tags:
                    self.band_keys.setdefault(tags[round(float(v[0]), 6)], k)
        for k in self.radiance:
            params[k] = self._radiance(k, 1.0)
        for u, k in self.band_keys.items():
            params[k] = dr.opaque(mi.Float, Look().uniforms()[u])
        # the camera flash's radiance, switched per band (an emitter without a colour texture: rgb or uniform)
        self.k_flash_radiance = next((k for k in keys if k.startswith("camera_assist_light")
                                      and k.endswith(emitted)), None)
        if self.k_flash_radiance is not None:
            self.radiance[self.k_flash_radiance] = (type(params[self.k_flash_radiance]),
                                                    np.array(params[self.k_flash_radiance], dtype=np.float64).ravel())
            params[self.k_flash_radiance] = self._radiance(self.k_flash_radiance, 1.0)
        self.k_band = self.band_keys.pop(NIR_BAND_ID, None)  # nir_band: the near-infrared-only switch
        if self.k_band is not None:
            params[self.k_band] = dr.opaque(mi.Float, 0.0)
        if self.env_original is not None:
            params[f"{ENV_ID}.data"] = mi.TensorXf(self.env_original)
            params[f"{ENV_ID}.scale"] = dr.opaque(mi.Float, 1.0)
            params[f"{ENV_ID}.to_world"] = mi.ScalarTransform4f()
        params.update()
        self._evaluate_transforms(params)
        del params
        counts = {c: list(self.light_class.values()).count(c) for c in sorted(set(self.light_class.values()))}
        lamp_parts = sum(1 for sid in self.lamp_lights if self.groups.get(sid) == "lamp")
        print(f"[resident] {len(self.light_class)} indoor emitters {counts} ({len(self.lamp_lights) - lamp_parts} lit "
              f"lamps' parts in place of boxes, {lamp_parts} dark lamps' shades in group 'lamp'), "
              f"{len(self.emitter_weights)} emitters weighted by power, environment "
              f"{'controllable' if self.env_original is not None else 'fixed'}, "
              f"near-infrared {'on' if self.k_band else 'off'}", flush=True)
        # dr.freeze records the render once per (spp, film size, seed) and replays its kernels, like
        # the production renderer on Device 1; without it every frame re-traces the scene.
        self.freeze = freeze
        # a request whose spp is a multiple of this replays the recording at this spp that many times (new
        # seeds, averaged), so a new spp records nothing; 0 records every spp as asked
        self.freeze_spp = int(freeze_spp)
        self.sensor_input = FREEZE_INPUT == "sensor" and self.k_flash is None
        self.frozen: dict[tuple, object] = {}
        self.batch_sensors: dict[tuple, object] = {}
        self.dual_sensors: dict[tuple, object] = {}  # (width, height, hfov) -> batch sensor, visible | near-infrared
        self.dual_pose: dict[tuple, bytes] = {}
        self.frozen_args: dict[tuple, tuple] = {}  # recording key -> its inputs before the seed
        self.sky_nir = NIR_DEFAULTS["daylight"]
        self.flash_pdf = FLASH_PDF  # the flash pass's near-specular cut, set per launch from the view's Look
        self.max_depth = 0  # path length cap of every pass, set per launch (Look.max_depth); 0: the scene's own
        self.rr_depth = 0  # Russian roulette's first depth, set per launch (Look.rr_depth); 0: Mitsuba's 5
        self.caustics = True  # the passive and own passes keep caustics, set per launch (Look.caustics)
        self.split = False  # the active pass on one path (Look.split), set per launch
        self.unpolarized = False  # Look.unpolarized: the passes' integrators without Mueller matrices, set per camera

    def integrator(self, name: str, sky: float | None = None):
        """path_nocaustics in an assist mode (or, for "own", as plain path: caustics kept, every emitter),
        with ``sky`` (default: daylight's near-infrared share) on the sky seen by a dual render's
        near-infrared slot. Without bandscale lights, "own" is the scene's own integrator (None)."""
        if name == "own" and not self.band_keys and not self.max_depth and not self.rr_depth and self.caustics \
                and not self.hidden and not self.unpolarized:
            return None
        sky = self.sky_nir if sky is None else float(sky)
        # the flash's near-specular cut (Look.flash_pdf); split cuts the flash's paths with it too
        pdf = float(self.flash_pdf) if name in ("flash", "split") else None
        key = (name, sky, pdf, self.max_depth, self.rr_depth, self.caustics, self.unpolarized)
        if key not in self.integrators:
            spec = dict(self.depth["passive" if name == "split" else name])
            if name == "split":  # one path long enough for both passes
                spec["max_depth"] = max(spec["max_depth"], self.depth["flash"]["max_depth"])
            if self.max_depth:  # a viewer's diagnostic: 1 emitters seen directly, 2 direct light, 3+ indirect bounces
                spec["max_depth"] = int(self.max_depth)
            if name == "own":
                inner = {"type": "path_nocaustics", "max_depth": spec["max_depth"], "caustics": self.caustics}
            elif name == "split":  # passive's emitters (caustics as asked) and the flash's own sample (never caustics)
                inner = {"type": "path_nocaustics", "max_depth": spec["max_depth"], "assist": ASSIST_PREFIX,
                         "assist_mode": "split", "caustics": self.caustics}
            else:
                inner = {"type": "path_nocaustics", "max_depth": spec["max_depth"], "assist": ASSIST_PREFIX,
                         "assist_mode": "exclude" if name == "passive" else "only",
                         "caustics": name == "passive" and self.caustics}
            inner["sky_nir"] = sky
            if self.hidden:
                inner["hidden"] = self.hidden
                if self.hidden == HIDDEN_LIGHT_PREFIX and Resident.hidden_specular:
                    inner["hidden_specular"] = HIDDEN_SPEC_PREFIX
            if self.rr_depth:
                inner["rr_depth"] = int(self.rr_depth)
            if pdf is not None:
                inner["specular_pdf_threshold"] = pdf
            if self.unpolarized:   # the light's intensity alone: the unpolarized variant's speed on this scene
                inner["unpolarized"] = True
            if self.mode != "rgb" or self.rgbn:   # (unpolarized RGBN: the stokes wrapper writes the lanes, NIR among them)
                inner = {"type": "stokes", "integrator": inner}
                if spec["reference_axis"] is not None:
                    inner["reference_axis"] = self.mi.ScalarVector3f(*spec["reference_axis"])
            self.integrators[key] = self.mi.load_dict(inner)
            self._give_light_grid(self.integrators[key])
        return self.integrators[key]

    def _give_light_grid(self, integrator) -> bool:
        """The scene's light grid (build_light_grid) into a path_nocaustics (inside its stokes wrapper or not); a build
        without grid parameters (before b7) keeps the scene's own light sampling. True when given."""
        grid = getattr(self, "light_grid", None)
        if not grid:
            return False
        p = self.mi.traverse(integrator)
        prefix = "integrator." if "integrator.grid_pdf" in p.keys() else ""
        if prefix + "grid_pdf" not in p.keys():
            return False
        for k in ("grid_meta", "grid_pdf", "grid_prob", "grid_alias", "grid_lut"):
            p[prefix + k] = grid[k]
        p.update()
        if not getattr(self, "grid_used", False):
            print("[resident] light grid in the integrator: emitters picked per place", flush=True)
        self.grid_used = True
        return True

    def _key(self, spp: int, width: int, height: int, pass_name: str, dual, sky) -> tuple:
        """A recording's key: what is baked into it (spp, film size, the integrators and their settings)."""
        tail = ("dual", sky) if dual is not None else ("rgbn", sky) if self.rgbn else ()
        if pass_name in ("active", "flash") and self.flash_pdf != FLASH_PDF:
            tail += ("flash_pdf", float(self.flash_pdf))
        if self.max_depth:
            tail += ("max_depth", int(self.max_depth))
        if self.rr_depth:
            tail += ("rr_depth", int(self.rr_depth))
        if not self.caustics and pass_name != "flash":
            tail += ("no_caustics",)
        if pass_name == "active" and self.split:
            tail += ("split",)
        if self.unpolarized:
            tail += ("unpolarized",)
        return (int(spp), int(width), int(height), pass_name) + tail

    def pass_integrators(self, pass_name: str, sky: float | None = None) -> tuple:
        """The integrators a pass renders and adds up: "active" is the passive and the flash pass in one
        call (one frozen replay a frame instead of two: Dr.Jit walks the scene's inputs once per replay)."""
        names = (("split",) if self.split else ("passive", "flash")) if pass_name == "active" else (pass_name,)
        return tuple(self.integrator(n, sky) for n in names)

    def render_passes(self, scene, sensor, integrators: tuple, spp: int, seed):
        """The sum of one render per integrator (same sensor, spp and seed). A two-pass frame is linear in
        its passes (active = passive + flash, a Stokes sum), so the sum is the frame."""
        mi = self.mi
        img = None
        for it in integrators:
            one = mi.render(scene, sensor=sensor if sensor is not None else 0, integrator=it, spp=spp, seed=seed)
            img = one if img is None else img + one
        return img

    def dual_sensor(self, c2w: np.ndarray, width: int, height: int, hfov_deg: float, spp: int = 1):
        """A batch sensor of two copies of the camera, the second's rays at time 1 (slot_time): the visible
        band left, the near-infrared band right (rgb2nir, bandscale and sky_nir read the ray's time), in one
        render. Each slot has DUAL_MARGIN extra pixels a side (field of view widened to match), cropped off
        afterwards, so the reconstruction filter does not blend the two bands at the seam."""
        import drjit as dr

        mi = self.mi
        key = (int(width), int(height), round(float(hfov_deg), 6))
        if key not in self.dual_sensors:
            wm = int(width) + 2 * DUAL_MARGIN
            fov = math.degrees(2.0 * math.atan(math.tan(math.radians(float(hfov_deg)) / 2.0) * wm / int(width)))
            slot = {"type": "perspective", "fov": fov, "fov_axis": "x"}
            self.dual_sensors[key] = mi.load_dict({
                "type": "batch", "slot_time": True, "slot0": dict(slot), "slot1": dict(slot),
                "film": {"type": "hdrfilm", "width": 2 * wm, "height": int(height), "rfilter": {"type": RFILTER}},
                "sampler": {"type": "independent", "sample_count": 1}})
            # Every frozen render sees this sensor (registry), and its first render filled two things that
            # start empty: the slot of each ray last sampled (m_last_index) and the film's storage. That
            # changed every recording's inputs and each recorded again. Filled here as a render fills them:
            # one sample_ray at a render's wavefront size, the film prepared for the integrator's channels.
            sensor = self.dual_sensors[key]
            zero = dr.zeros(mi.Float, 2 * wm * int(height) * int(spp))
            dr.eval(sensor.sample_ray(zero, zero, mi.Point2f(zero, zero), mi.Point2f(zero, zero)))
            sensor.film().prepare(self.integrator("own").aov_names())
        sensor = self.dual_sensors[key]
        pose = np.asarray(c2w, dtype=np.float64).tobytes()
        if self.dual_pose.get(key) != pose:
            origin, target, up = frames.lookat(c2w)
            camera = mi.ScalarTransform4f.look_at(origin=origin, target=target, up=up)
            p = mi.traverse(sensor)
            first = key not in self.dual_pose
            for k in [k for k in p.keys() if k.endswith("to_world")]:
                p[k] = camera
            if first:
                p.update()
            for k in [k for k in p.keys() if k.endswith("to_world")]:  # one input layout (_evaluate_transforms);
                dr.make_opaque(p[k])  # later moves skip parameters_changed: it waits for the GPU (_camera_only)
            self.dual_pose[key] = pose
        return sensor

    def render(self, c2w: np.ndarray, width: int, height: int, hfov_deg: float, spp: int, seed: int,
               pass_name: str = "own", follow: bool | None = None, look: Look = Look()) -> np.ndarray:
        img, self.breakdown = self.finish(self.launch(c2w, width, height, hfov_deg, spp, seed, pass_name, follow,
                                                      look))
        return img

    def aim_emitter_weights(self, eye: np.ndarray) -> bool:
        """The emitters' sampling weights for a camera at ``eye`` (view_emitter_weights), made again when it has moved
        REWEIGHT_MOVE_M since: the lights near the view get the light samples. True when written."""
        power, places = EMITTER_POWER.get(id(self.scene)), EMITTER_PLACES.get(id(self.scene))
        if not VIEW_WEIGHTS or not self.emitter_weights or not power or not places or getattr(self, "grid_used", False):
            return False   # (the light grid picks per point: the camera's weights would only be what its base reads)
        if self.weights_eye is not None and float(np.linalg.norm(eye - self.weights_eye)) < REWEIGHT_MOVE_M:
            return False
        p = self.mi.traverse(self.scene)
        for k, w in view_emitter_weights(power, places, eye).items():
            p[k] = w
        p.update()
        self.weights_eye = np.array(eye, np.float64)
        return True

    def _set_camera(self, c2w: np.ndarray, width: int, height: int, hfov_deg: float, follow: bool,
                    look: Look = Look()) -> dict:
        """Write the camera, the assist shapes and the Look into the scene, unless already there."""
        mi = self.mi
        self.unpolarized = bool(look.unpolarized) and self.mode == "polar"   # integrators and recording keys (see integrator)
        pose = (np.asarray(c2w, dtype=np.float64).tobytes(), int(width), int(height), float(hfov_deg), follow, look)
        if pose == self.pose:
            return {"traverse_s": 0.0, "update_s": 0.0}
        t0 = time.perf_counter()
        self.aim_emitter_weights(np.asarray(c2w, np.float64)[:3, 3])   # the lights near this camera picked more often
        origin, target, up = frames.lookat(c2w)
        camera = mi.ScalarTransform4f.look_at(origin=origin, target=target, up=up)
        # The camera is always written through the sensor alone. Written through the whole scene's parameters
        # instead (a frame that also moves the flash or changes the Look), it left the frozen render's inputs
        # 64 opaque variables apart from a camera-only frame (43627 vs 43691 on Device 1), so a viewer that
        # walks and switches modes recorded again and again.
        ps = mi.traverse(self.sensor)  # fresh map: a frozen replay swaps the variables an old map holds
        t1 = time.perf_counter()
        ps["to_world"] = camera
        lens = self.fov != float(hfov_deg) or (self.ks_size is not None and self.size != (int(width), int(height)))
        if self.fov != float(hfov_deg):  # only when changed: a Python float rewritten every frame is a new input
            ps["x_fov"] = float(hfov_deg)
        if self.ks_size is not None and self.size != (int(width), int(height)):
            ps[self.ks_size] = mi.ScalarVector2u(int(width), int(height))
        if lens or self.pose is None:
            ps.update()
        else:
            # A camera move only: the value written, evaluated, and no parameters_changed. The perspective sensor's
            # parameters_changed reassigns its to_world field, whose scalar copy is read back from the GPU
            # (field::operator=, dr::slice): it waited for every pass still on the GPU, so a frame's camera write
            # never overlapped the previous frame's rendering (GPU 61 ms + CPU 58 ms per 16 spp pass, in series).
            # Rays are made from the value alone (sample_ray), so the image is the same (5e-4 on a test scene).
            self._camera_only(ps)
        t2 = time.perf_counter()
        if not (not follow and self.parked and look == self.look):
            # a Look change writes through the whole scene's parameters; a frame that only moves the flash
            # writes the assist shapes alone (the whole-scene traverse was ~46 ms of every moving frame)
            p = mi.traverse(self.scene) if look != self.look else None
            if self.k_flash is not None:
                # following: the flash rides 0.2 m behind the camera, facing its view. Parked: with bandscale lights it is
                # dark (its uniforms) and lifted 3 m at full size, as removed lights are; without them it
                # shrinks to nothing, as before. Either write is evaluated after the update (see
                # _evaluate_transforms), so moving it never changes the frozen render's inputs.
                lift = not follow and bool(self.band_keys)
                # PC_OPTICALNAV_RENDER's ablation (worst view, 1024 spp): the dataset's 0.12 x 0.08 m flash left 4414
                # bright outliers; the same power from 2x and 3x the side, 408 and 19 (look.flash_scale)
                k = float(look.flash_scale) if follow else 1.0
                size = [s * k for s in self.flash["size_world"]] if follow or lift else (1e-6, 1e-6)
                d = float(self.flash["distance_m"])
                up = np.eye(4)
                up[1, 3] = 3.0 if lift else 0.0
                writes = {self.k_flash: up @ flash_matrix(c2w, size, d)}
                if self.k_polarizer is not None:
                    writes[self.k_polarizer] = up @ flash_matrix(
                        c2w, size, d + 0.01, float(self.flash.get("polarizer_angle_deg") or 0.0))
                if follow and any(look.flash_offset):  # off the camera's axis (the viewer's rig editor), still facing its view
                    for k in writes:
                        writes[k][:3, 3] += level_offset(c2w, look.flash_offset)
                self._write_assist(p, writes)
                self.parked = not follow
            if p is not None:
                self._write_look(p, look)
                p.update()
                self._evaluate_transforms(p)
        self.fov, self.size, self.pose = float(hfov_deg), (int(width), int(height)), pose
        return {"traverse_s": t1 - t0, "update_s": t2 - t1, "scene_update_s": time.perf_counter() - t2}

    def depth_image(self, c2w: np.ndarray, width: int, height: int, hfov_deg: float):
        """Queue the depth (m along the view axis; 0 where a ray leaves the scene) through each pixel's centre of
        the camera ``c2w``, against the scene as the last launch left it (hidden shapes, mirrors). One ray a pixel,
        no antialiasing. The pixel directions are cached per size; the camera goes in as opaque values, so every
        frame replays the same kernel. Returns a Dr.Jit array scheduled on the GPU (read it with np.array)."""
        import drjit as dr
        mi = self.mi
        key = (int(width), int(height), round(float(hfov_deg), 6))
        if getattr(self, "_depth_key", None) != key:
            t = math.tan(math.radians(float(hfov_deg)) / 2.0)
            u = (2.0 * (np.arange(width) + 0.5) / width - 1.0) * t
            v = (1.0 - 2.0 * (np.arange(height) + 0.5) / height) * t * height / width
            uu, vv = np.meshgrid(u, v)
            inv = 1.0 / np.sqrt(uu * uu + vv * vv + 1.0)  # a unit ray's distance per metre of depth
            self._depth_dirs = [mi.Float(np.ascontiguousarray(a.ravel(), dtype=np.float32))
                                for a in (uu * inv, vv * inv, -inv)]
            self._depth_inv = mi.Float(np.ascontiguousarray(inv.ravel(), dtype=np.float32))
            self._depth_key = key
        m = np.asarray(c2w, dtype=np.float64)
        r = [[dr.opaque(mi.Float, float(m[i, j])) for j in range(3)] for i in range(3)]
        cx, cy, cz = self._depth_dirs
        d = mi.Vector3f(*(r[i][0] * cx + r[i][1] * cy + r[i][2] * cz for i in range(3)))
        o = mi.Point3f(*(dr.opaque(mi.Float, float(m[i, 3])) for i in range(3)))
        pi = self.scene.ray_intersect_preliminary(mi.Ray3f(o, d), coherent=True)
        t = pi.t
        valid = pi.is_valid()
        if self.hidden:   # the room softboxes the camera sees through: a depth sensor too
            for _ in range(2):
                unseen = valid & self._unseen(pi.shape)
                if not getattr(self, "_unseen_shapes", None):
                    break
                pi2 = self.scene.ray_intersect_preliminary(mi.Ray3f(o + d * (t + 1e-4), d), unseen)
                valid = dr.select(unseen, pi2.is_valid(), valid)
                t = dr.select(unseen, t + 1e-4 + pi2.t, t)
                pi.shape = dr.select(unseen, pi2.shape, pi.shape)
        z = dr.select(valid, t * self._depth_inv, 0.0)
        dr.schedule(z)
        return z

    LIDAR_BOUNCES = 3        # a LiDAR ray follows mirrors and glass this many surfaces deep (lidar_image)
    LIDAR_GLASS_T = 0.92     # what one glass surface lets through (Fresnel at normal incidence, n = 1.5)
    LIDAR_MIRROR_R = 0.90    # what a smooth conductor (a mirror) gives back of a LiDAR's light
    LIDAR_CHANNELS = ("range_m", "return", "cos", "first_m", "first_kind", "first_cos")
    LIDAR_LAMP = 0.5         # what a lamp's glowing part (black to the path tracer: it only emits) returns (estimate)

    LIDAR_MIRROR_ALPHA = 0.05   # a conductor this smooth (GGX alpha, no diffuse part) is a mirror to a LiDAR (raster.py's too)

    def _mirror_like(self, bsdf) -> bool:
        """Whether a BSDF is a conductor alone, smoother than LIDAR_MIRROR_ALPHA (a mirror made of a rough conductor of
        tiny roughness: the path tracer draws it as a mirror, but its flags say glossy)."""
        keys = list(self.mi.traverse(bsdf).keys())
        if any(k.split(".")[-1] in ("diffuse_reflectance", "reflectance") or "diffuse_reflectance." in k
               or ".reflectance." in k or k.startswith("reflectance.") for k in keys):
            return False
        if not any(k.split(".")[-1] in ("k", "specular_reflectance") or ".k." in k or "specular_reflectance." in k for k in keys):
            return False
        params = self.mi.traverse(bsdf)
        alphas = [k for k in keys if k.split(".")[-1] in ("alpha", "alpha_u", "alpha_v")]
        return all(np.size(params[k]) <= 4 and float(np.max(np.array(params[k]))) < self.LIDAR_MIRROR_ALPHA for k in alphas)

    def _shape_table(self, name: str, pick):
        """A Bool table by shape registry id: ``pick(shape)`` for each of the scene's shapes (built once per name)."""
        import drjit as dr
        mi = self.mi
        tables = self.__dict__.setdefault("_shape_tables", {})
        if name not in tables:
            ids = []
            for s in self.scene.shapes():
                try:
                    yes = pick(s)
                except Exception:  # noqa: BLE001 - a BSDF the test cannot read: not picked
                    yes = False
                if yes:
                    ids.append(int(dr.reinterpret_array(mi.UInt32, mi.ShapePtr(s))[0]))
            table = np.zeros(max(ids, default=0) + 2, bool)
            table[ids] = True
            tables[name] = (mi.Bool(table), len(ids))
        return tables[name]

    def _in_table(self, name: str, pick, shape_ptr):
        import drjit as dr
        mi = self.mi
        table, _ = self._shape_table(name, pick)
        idx = dr.reinterpret_array(mi.UInt32, shape_ptr)
        n = dr.width(table)
        return dr.gather(mi.Bool, table, dr.minimum(idx, n - 1)) & (idx < n)

    def _unseen_build(self) -> None:
        import drjit as dr
        mi = self.mi
        if getattr(self, "_unseen_table", None) is None:
            ids = [int(dr.reinterpret_array(mi.UInt32, mi.ShapePtr(s))[0]) for s in self.scene.shapes()
                   if self.hidden and s.id().startswith(self.hidden)]
            table = np.zeros(max(ids, default=0) + 2, bool)
            table[ids] = True
            self._unseen_shapes = ids
            self._unseen_table = mi.Bool(table)

    def _unseen(self, shape_ptr):
        """Whether each ray's shape is one the camera sees through (the room softboxes: ``hidden``), as a Dr.Jit Bool:
        a table by the shapes' registry ids, one gather."""
        import drjit as dr
        mi = self.mi
        self._unseen_build()
        idx = dr.reinterpret_array(mi.UInt32, shape_ptr)
        n = dr.width(self._unseen_table)
        return dr.gather(mi.Bool, self._unseen_table, dr.minimum(idx, n - 1)) & (idx < n)

    def lidar_image(self, c2w: np.ndarray, width: int, height: int, hfov_deg: float) -> list:
        """What a LiDAR beam through each pixel's centre meets (hw.sensing.OusterLidar's noise model reads it), as
        Dr.Jit arrays (LIDAR_CHANNELS), queued on the GPU:

        * ``range_m``: the path length to the first surface that sends light back (0: none), along the ray: through
          glass and off mirrors (delta BSDFs), up to LIDAR_BOUNCES surfaces, so a mirror gives phantom points behind it;
        * ``return``: that surface's monostatic reflectance in the near-infrared lane (the RGBN variant's fourth, as
          rgb2nir gives it; a LiDAR's 865 nm): pi times the BSDF evaluated with the light going back the way it came,
          its cosine included, so a white Lambertian surface head-on gives 1 and at 60 degrees 0.5; glossy surfaces
          give their highlight's value toward the sensor; times what the glass and mirrors on the way let through;
        * ``cos``: the cosine of the incidence there;
        * ``first_m``, ``first_kind``, ``first_cos``: the first surface when it is glass (2) or a mirror (1), else 0:
          a real LiDAR sometimes sees the glass itself near normal incidence.

        Glass is passed straight (no refraction offset). The scene as the last launch left it, as depth_image."""
        import drjit as dr
        mi = self.mi
        self.depth_image(c2w, width, height, hfov_deg)   # the pixel directions, cached per size (and a depth, unused)
        m = np.asarray(c2w, dtype=np.float64)
        r = [dr.opaque(mi.Float, float(m[i, j])) for i in range(3) for j in range(3)]
        o = [dr.opaque(mi.Float, float(m[i, 3])) for i in range(3)]
        cx, cy, cz = self._depth_dirs
        # the shape tables the kernel reads, made before it is traced
        self._unseen_build()
        mirror, _ = self._shape_table("mirror", lambda sh: sh.bsdf() is not None and self._mirror_like(sh.bsdf()))
        tables = [self._unseen_table, mirror]
        if not self.freeze:
            out = self._lidar_trace(self.scene, cx, cy, cz, o, r, tables)
        else:
            # frozen as the renders are: tracing the BSDF calls of every material (a robot's bodies bring dozens) took a
            # minute a scan; a replay takes milliseconds
            frozen = self.__dict__.setdefault("_lidar_frozen", {})
            key = (int(width), int(height))
            if key not in frozen:
                frozen[key] = dr.freeze(lambda scene, cx, cy, cz, o, r, t: self._lidar_trace(scene, cx, cy, cz, o, r, t),
                                        backend=dr.JitBackend.CUDA, auto_opaque=False, warn_after=4)
            out = frozen[key](self.scene, cx, cy, cz, o, r, tables)
        out = list(out)
        dr.schedule(*out)
        return out

    def _lidar_trace(self, scene, cx, cy, cz, o, r, tables) -> list:
        """lidar_image's kernel: ``o`` the origin (3 opaque Floats), ``r`` the rotation (9, row by row), ``tables`` the
        shapes it passes through and the mirrors (Bool tables by registry id: inputs, for a frozen kernel)."""
        import drjit as dr
        mi = self.mi

        def lookup(table, shape_ptr):
            idx = dr.reinterpret_array(mi.UInt32, shape_ptr)
            n = dr.width(table)
            return dr.gather(mi.Bool, table, dr.minimum(idx, n - 1)) & (idx < n)
        d = mi.Vector3f(*(r[3 * i] * cx + r[3 * i + 1] * cy + r[3 * i + 2] * cz for i in range(3)))
        o = mi.Point3f(*o)
        ray = mi.Ray3f(o, d)
        n = dr.width(cx)
        F = mi.Float
        first_m, first_kind, first_cos = dr.zeros(F, n), dr.zeros(F, n), dr.zeros(F, n)
        total, through = dr.zeros(F, n), dr.full(F, 1.0, n)
        end_o, end_d, end_total = mi.Point3f(o), mi.Vector3f(d), dr.zeros(F, n)
        found = dr.zeros(mi.Bool, n)
        active = dr.full(mi.Bool, True, n)
        first = dr.full(mi.Bool, True, n)          # no glass or mirror met yet
        BF = mi.BSDFFlags
        # follow each ray through what it passes (glass, the room softboxes) and off mirrors, by the BSDFs' flags alone;
        # the BSDF itself is evaluated once, at the surface each ray ends on (tracing an evaluation of every material
        # of the scene is what costs)
        for k in range(self.LIDAR_BOUNCES + 2):
            si = scene.ray_intersect(ray, active)
            hit = active & si.is_valid()
            unseen = hit & lookup(tables[0], si.shape)     # what the camera sees through, the LiDAR does too
            valid = hit & ~unseen
            fl = si.bsdf(ray).flags()
            smooth_r = mi.has_flag(fl, BF.DeltaReflection)
            smooth_t = mi.has_flag(fl, BF.DeltaTransmission)
            rough = mi.has_flag(fl, BF.Diffuse) | mi.has_flag(fl, BF.Glossy)
            mirror = lookup(tables[1], si.shape)
            smooth_r |= mirror
            delta = ((smooth_r | smooth_t) & ~rough) | mirror
            stop = valid & ~delta
            end_o = dr.select(stop, ray.o, end_o)
            end_d = dr.select(stop, ray.d, end_d)
            end_total = dr.select(stop, total, end_total)
            found |= stop
            c = dr.abs(mi.Frame3f.cos_theta(si.wi))
            meet = valid & delta & first
            first_m = dr.select(meet, total + si.t, first_m)
            first_kind = dr.select(meet, dr.select(smooth_t, 2.0, 1.0), first_kind)
            first_cos = dr.select(meet, c, first_cos)
            first &= ~(valid & delta)
            go = valid & delta
            reflected = ray.d - 2.0 * dr.dot(ray.d, si.n) * si.n
            through = dr.select(go, through * dr.select(smooth_t, self.LIDAR_GLASS_T, self.LIDAR_MIRROR_R), through)
            total = dr.select(go | unseen, total + si.t, total)
            ray = si.spawn_ray(dr.select(smooth_t | unseen, ray.d, reflected))
            active = go | unseen
        # the surface each ray ends on, again, and its BSDF toward the sensor
        ray = mi.Ray3f(end_o, end_d)
        si = scene.ray_intersect(ray, found)
        found &= si.is_valid()
        cos_i = mi.Frame3f.cos_theta(si.wi)
        # the back of a one-sided BSDF is black to the path tracer, but a real surface returns a laser either way: seen
        # from behind, it is read as from the front
        si.wi = dr.select(cos_i < 0.0, mi.Vector3f(si.wi.x, si.wi.y, -si.wi.z), si.wi)
        f = mi.unpolarized_spectrum(si.bsdf(ray).eval(mi.BSDFContext(), si, si.wi, found))
        rho = dr.pi * dr.maximum(f[3], 0.0)
        lamp = found & (rho <= 0.0) & (si.emitter(scene) != None)  # noqa: E711 - a Dr.Jit pointer array
        c = dr.abs(cos_i)
        rho = dr.select(lamp, self.LIDAR_LAMP * c, rho)
        rng = dr.select(found, end_total + si.t, 0.0)
        ret = dr.select(found, through * rho, 0.0)
        cos = dr.select(found, c, 0.0)
        return [rng, ret, cos, first_m, first_kind, first_cos]

    def raster_views(self, cams: list, width: int, height: int, hfov_deg: float, follow: bool, look: Look,
                     aa: str = "edge", solid: bool = False, solid_filter: str | None = None) -> tuple:
        """The rasterized preview (raster.py) of each camera, with the scene as ``look`` has it (written into the
        scene as a launch would, so the path tracer and the preview always show the same state). Returns
        (each view's Stokes images [S0..S3] in R, G, B and near-infrared lanes, depths, timing parts)."""
        raster = _raster_module()
        follow = bool(follow and self.k_flash is not None)
        parts = {"scene_update_s": 0.0, "raster_build_s": 0.0, "raster_s": 0.0}
        views, depths = [], []
        for c2w in cams:
            t0 = time.perf_counter()
            # the flash stays parked: the preview lights it itself, and moving its rectangle every frame rebuilt its
            # acceleration structure (a path frame puts it back at the camera when it writes its own pose)
            self._set_camera(c2w, width, height, hfov_deg, False, look)
            t1 = time.perf_counter()
            if getattr(self, "_raster", None) is None or type(self._raster) is not raster.RasterScene:
                self._raster = None
                self._raster = raster.RasterScene(self)
                self._raster.scene_dir, self._raster.scene_xml = getattr(self, "scene_dir", None), getattr(self, "scene_xml", "base.xml")
                parts["raster_build_s"] += self._raster.build_s
            self._raster.sync(look.hidden)
            self._raster.follow_moves()   # bodies and pushed objects where this view placed them
            t2 = time.perf_counter()
            lanes, z = self._raster.render_solid(c2w, width, height, hfov_deg, look, aa, solid_filter) if solid else \
                self._raster.render(c2w, width, height, hfov_deg, look, follow, aa)
            parts["scene_update_s"] += t1 - t0
            parts["raster_s"] += time.perf_counter() - t2
            views.append(lanes)
            depths.append(z)
        parts["wait_s"] = parts["raster_s"]
        return views, depths, parts

    def _write_assist(self, p, writes: dict) -> None:
        """Write the assist shapes' transforms: into ``p`` (the whole scene's parameters, updated by the
        caller), or, without it, through each shape's own parameters and then the scene, whose update
        rebuilds only the small acceleration structure they are in (the Mitsuba build's selective GAS
        rebuild; before it, every move rebuilt every mesh of the scene, ~40 ms on Device 1)."""
        import drjit as dr

        mi = self.mi
        if p is not None:
            for k, m in writes.items():
                p[k] = mi.ScalarTransform4f(m.tolist())
            return
        for k, m in writes.items():
            ps = mi.traverse(self.assist_shapes[k[:-len(".to_world")]])
            ps["to_world"] = mi.ScalarTransform4f(m.tolist())
            ps.update()
            dr.make_opaque(ps["to_world"])  # evaluated, as _evaluate_transforms would
        self.scene.parameters_changed([])

    def set_bodies(self, poses: dict) -> float:
        """Place the bodies' links: instance id -> object-to-world 4x4. A changed instance is written through its own
        parameters and evaluated (as the flash is, see _write_assist), then the scene rebuilds its top-level
        structure alone: the links' meshes are in shapegroups, built once. Returns the seconds it took."""
        import drjit as dr

        t0, mi, moved = time.perf_counter(), self.mi, False
        for sid, m in poses.items():
            if sid not in self.body_shapes:
                raise ValueError(f"no body link {sid!r} in this scene")
            m = np.ascontiguousarray(m, dtype=np.float64).reshape(4, 4)
            if self.body_pose.get(sid) == m.tobytes():
                continue
            ps = mi.traverse(self.body_shapes[sid])
            ps["to_world"] = mi.ScalarTransform4f(m.tolist())
            ps.update()
            dr.make_opaque(ps["to_world"])
            self.body_pose[sid], moved = m.tobytes(), True
        if moved:
            self.scene.parameters_changed([])
            self.geometry_epoch += 1
        return time.perf_counter() - t0

    # simulator frame (x = X, y = -Z, z = Y of the scene files) <- Mitsuba's (Y up): a physics pose change into the scene's
    SIM_FROM_MI = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    def set_objects(self, poses: dict, shapes_of: dict) -> float:
        """Move pushed objects: physics object id -> the change of its pose since the scene file (simulator frame, 4x4).
        Each of its dyn_ meshes (a mesh of its own: tools/merge_meshes.py BODY_ATTRIBUTE) gets its loaded vertices and
        normals moved by it, written through the mesh's own parameters; then the scene rebuilds just those meshes'
        acceleration structures (the selective GAS rebuild the flash uses, _write_assist). An object not named any
        more goes back where the file put it. Returns the seconds it took."""
        import drjit as dr

        t0, mi = time.perf_counter(), self.mi
        if not hasattr(self, "object_pose"):
            self.object_pose: dict[str, bytes] = {}     # object id -> the change last written
            self.object_rest: dict[str, tuple] = {}     # dyn_ shape id -> (its shape, positions N x 3, normals or None)
            self.dyn_shapes = {s.id(): s for s in self.scene.shapes() if s.id().startswith("dyn_")}
        wanted = dict(poses)
        for oid in [o for o in self.object_pose if o not in wanted]:
            wanted[oid] = np.eye(4)
        changed = False
        for oid, m in wanted.items():
            m = np.ascontiguousarray(m, dtype=np.float64).reshape(4, 4)
            if self.object_pose.get(oid, np.eye(4).tobytes()) == m.tobytes():
                continue
            mm = np.linalg.inv(self.SIM_FROM_MI) @ m @ self.SIM_FROM_MI   # the same change in the scene's frame
            inst = getattr(self, "object_instances", {}).get(oid)
            if inst is not None:   # an instance (a scene loaded with bodies): its transform alone, no mesh rebuilt
                ps = mi.traverse(inst)
                ps["to_world"] = mi.ScalarTransform4f(mm.tolist())
                ps.update()
                dr.make_opaque(ps["to_world"])
                changed = True
                if np.allclose(m, np.eye(4)):
                    self.object_pose.pop(oid, None)
                else:
                    self.object_pose[oid] = m.tobytes()
                continue
            rot, move = mm[:3, :3].astype(np.float32), mm[:3, 3].astype(np.float32)
            for sid in shapes_of.get(oid, ()):
                shape = self.dyn_shapes.get(sid)
                if shape is None:
                    continue   # merged into another mesh by the loader (a scene merged before BODY_ATTRIBUTE): stays put
                if sid not in self.object_rest:
                    ps = mi.traverse(shape)
                    pos = np.array(ps["vertex_positions"], dtype=np.float32).reshape(-1, 3)
                    nrm = np.array(ps["vertex_normals"], dtype=np.float32).reshape(-1, 3) \
                        if "vertex_normals" in ps.keys() and dr.width(ps["vertex_normals"]) else None
                    self.object_rest[sid] = (shape, pos, nrm)
                shape, pos, nrm = self.object_rest[sid]
                ps = mi.traverse(shape)
                ps["vertex_positions"] = mi.Float((pos @ rot.T + move).ravel())
                if nrm is not None:
                    ps["vertex_normals"] = mi.Float((nrm @ rot.T).ravel())
                ps.update()
                changed = True
            if np.allclose(m, np.eye(4)):
                self.object_pose.pop(oid, None)
            else:
                self.object_pose[oid] = m.tobytes()
        if changed:
            self.scene.parameters_changed([])
            self.geometry_epoch += 1
        self.objects_s = time.perf_counter() - t0   # in the next launch's time split (objects_s)
        return self.objects_s

    def intensity_lanes(self, img):
        """An RGBN Stokes film (a Dr.Jit tensor, 19 or 20 channels) cut down on the GPU to R, G, B and the near-infrared
        lane of S0 before it is read back: an unpolarized frame needs nothing else (15 MB -> 3 MB at 512 x 384)."""
        import drjit as dr
        if not hasattr(img, "shape") or len(img.shape) != 3 or img.shape[2] not in (19, 20):
            return img
        off = img.shape[2] - 16
        return dr.concat([img[:, :, 0:3], img[:, :, off + 3:off + 4]], axis=2)

    def park_bodies(self) -> None:
        """Every body link (and lamp) back where a load puts it, far below the scene: out of sight and out of the light,
        while the scene stays loaded for the next view that places them."""
        far = np.eye(4)
        far[1, 3] = -1000.0
        lamp = np.diag([0.001, 0.001, 0.001, 1.0])
        lamp[1, 3] = -1000.0
        self.set_bodies({sid: (lamp if "__light" in sid else far) for sid in self.body_shapes})

    def _camera_only(self, ps) -> None:
        """A sensor's new to_world, evaluated in place, without the sensor's parameters_changed (see _set_camera)."""
        import drjit as dr

        dr.make_opaque(ps["to_world"])

    def _evaluate_transforms(self, p) -> None:
        """Evaluate every shape and environment transform in place. A transform written from the host (a
        ScalarTransform4f) stays a literal until a render evaluates it, so a frame that moved the flash and
        one that did not gave the frozen render two input layouts (the flash rectangle's matrix and inverse:
        64 opaque variables, 43627 vs 43691 on Device 1), and each recording recorded once more on the
        first frame of the other kind. Evaluated right after the write, it always has one layout."""
        import drjit as dr

        for k in self.k_transforms:
            dr.make_opaque(p[k])

    def _radiance(self, key: str, level: float):
        """An indoor emitter's loaded radiance times ``level``, as a new opaque value: an evaluated variable
        the frozen replay reads, never a literal (0 * radiance would fold to one and record again)."""
        import drjit as dr

        # One variable per channel, made with dr.opaque: built from Python floats and then made opaque,
        # equal values (grey lights, lights of one colour, every channel at level 0) share one literal, so
        # the count of the frozen render's opaque inputs followed the values (42405 vs 42469 on Device 1)
        # and the first light switch recorded again.
        mi, (kind, values) = self.mi, self.radiance[key]
        channels = [dr.opaque(mi.Float, float(v) * float(level)) for v in values]
        return kind(*channels) if len(channels) > 1 else kind(channels[0])

    def _emitter_level(self, key: str, look: Look) -> float:
        """An indoor emitter's radiance factor: 0 when removed; else 1 with bandscale lights (the shared
        uniforms carry the levels), or its level for this band without them."""
        if light_of_key(key) in look.hidden and light_of_key(key) in self.delta_lights:
            return 0.0   # a point or spot light: no shape to lift away, so dark
        if self.band_keys:  # levels and lighting modes are in the shared uniforms: the radiance stays as loaded
            return 1.0
        if light_of_key(key) in look.hidden:
            return 0.0
        return look.light_level(self.light_class[key])

    @staticmethod
    def _flash_level(look: Look) -> float:
        return float(look.flash_nir if look.band == 1 else look.flash_rgb) / (look.flash_scale * look.flash_scale)

    def _write_look(self, p, look: Look) -> None:
        """Hidden shapes, indoor light level and environment: only what differs from the scene's Look."""
        import drjit as dr

        mi, now = self.mi, self.look
        for shape in (look.hidden ^ now.hidden) & set(self.transforms):
            # lifted 3 m, through the ceiling, at full size: shrinking it changed the frozen render's inputs
            # (Dr.Jit: opaque variables 42431 -> 43327) and every lighting switch recorded again (~90 s)
            m = self.transforms[shape].copy()
            if shape in look.hidden:
                m[1, 3] += 3.0
            p[f"{shape}.to_world"] = mi.ScalarTransform4f(m.tolist())
        for key in {self.blocks[s][0] for s in look.hidden ^ now.hidden if s in self.blocks}:
            verts = self.originals[key].copy()
            for shape in look.hidden & set(self.blocks):
                k, start, n = self.blocks[shape]
                if k != key:
                    continue
                if shape in self.groups:  # a light: lifted 3 m at full size (a collapsed emitter's area-sampling
                    verts[start:start + n] += np.array([0.0, 3.0, 0.0], np.float32)  # data changes layout)
                else:  # every vertex onto the first: degenerate triangles, never hit
                    verts[start:start + n] = verts[start]
            p[key] = mi.Float(verts.ravel())
        for k in self.light_class:
            level, before = self._emitter_level(k, look), self._emitter_level(k, now)
            if level != before:  # a removed light (lighting mode) is hidden and dark
                p[k] = self._radiance(k, level)
        if self.band_keys:  # per-band levels: shared uniforms the lights (bandscale) read per ray
            vals, was = look.uniforms(), now.uniforms()
            for u, k in self.band_keys.items():  # (nir_band is k_band, below)
                if vals[u] != was[u]:
                    p[k] = dr.opaque(mi.Float, vals[u])
        elif self.k_flash_radiance is not None and self._flash_level(look) != self._flash_level(now):
            p[self.k_flash_radiance] = self._radiance(self.k_flash_radiance, self._flash_level(look))
        if (look.band == 1) != (now.band == 1):
            if self.k_band is None:
                raise ValueError("near-infrared needs the rgb2nir plugin (see mitsuba/rgb2nir.cpp)")
            p[self.k_band] = dr.opaque(mi.Float, 1.0 if look.band == 1 else 0.0)
        if self.env_original is not None:
            if look.env != now.env:
                p[f"{ENV_ID}.data"] = mi.TensorXf(self.env_original if look.env == "original" else self.envs(look.env))
            if look.sky_level() != now.sky_level():
                p[f"{ENV_ID}.scale"] = dr.opaque(mi.Float, float(look.sky_level()))
            if look.env_rotate_deg != now.env_rotate_deg:
                p[f"{ENV_ID}.to_world"] = mi.ScalarTransform4f().rotate([0, 1, 0], float(look.env_rotate_deg))
        elif (look.env, look.env_scale, look.env_rotate_deg) != ("original", 1.0, 0.0):
            raise ValueError("this scene's environment cannot be changed (no constant sky or plain envmap)")
        self.look = look

    def launch(self, c2w: np.ndarray, width: int, height: int, hfov_deg: float, spp: int, seed: int,
               pass_name: str = "own", follow: bool | None = None, look: Look = Look()) -> tuple:
        """Set the camera and queue the render on the GPU. Returns a handle for ``finish``, which waits for
        the GPU and reads the image back, so the caller can queue the next view in between (CUDA runs
        the queued kernels in order). ``follow``: keep the flash at the camera (default: when the scene
        has one); ``look``: hidden shapes, lights and environment. Every pass of one frame passes the same values."""
        mi = self.mi
        follow = (self.k_flash is not None) if follow is None else bool(follow and self.k_flash is not None)
        breakdown = self._set_camera(c2w, width, height, hfov_deg, follow, look)
        self.flash_pdf = float(look.flash_pdf)  # the flash pass's near-specular cut: integrators and recording keys
        self.max_depth = int(look.max_depth)  # likewise
        self.rr_depth = int(look.rr_depth)  # likewise
        self.caustics = bool(look.caustics)  # likewise
        self.split = bool(look.split)  # likewise
        # both bands at once: the near-infrared lane of an RGBN variant, else the dual sensor; the sky in the
        # near-infrared at daylight's share either way
        fs = self.freeze_spp
        per_call = fs if self.freeze and fs and int(spp) > fs and int(spp) % fs == 0 else int(spp)  # see below
        both = look.band == 2 or self.rgbn
        dual = self.dual_sensor(c2w, width, height, hfov_deg, per_call) if both and not self.rgbn else None
        sky = dict(look.nir).get("daylight", self.sky_nir) if both else None
        if not self.freeze:
            # passes of at most SPP_CHUNK samples, averaged by sample count
            its = self.pass_integrators(pass_name, sky)
            total, done, acc, i = int(spp), 0, None, 0
            while done < total:
                n = min(SPP_CHUNK[self.mode], total - done)
                img = np.array(self.render_passes(self.scene, dual if dual is not None else self.sensor, its, n,
                                                  (int(seed) + 104729 * i) & 0xFFFFFFFF), dtype=np.float64) * n
                acc = img if acc is None else acc + img
                done, i = done + n, i + 1
            return (acc / total).astype(np.float32), breakdown
        import drjit as dr

        spp, repeats = int(spp), 1
        if self.freeze_spp and spp > self.freeze_spp and spp % self.freeze_spp == 0:
            spp, repeats = self.freeze_spp, spp // self.freeze_spp
        key = self._key(spp, width, height, pass_name, dual, sky)
        t2 = time.perf_counter()
        if key not in self.frozen:  # spp and film size are baked into each recording
            # A frozen render's inputs include every live sensor, shape and emitter (Dr.Jit's registry), not
            # just its arguments: the dual sensor made on the first two-band frame changed every recording
            # made before it ("number of elements in the input changed from 181902 to 181905") and each
            # recorded again. Made with the first recording at this size, it is there for all of them.
            if self.band_keys and dual is None and not self.rgbn:
                self.dual_sensor(c2w, width, height, hfov_deg, spp)
            # A two-pass scene records the light on ("active": passive + flash in one recording) and off
            # ("passive") together, so switching the light later replays at once instead of stopping to
            # record. The flash's place and the hidden shapes are scene data, not part of a recording.
            names = ("active", "passive") if "flash" in self.depth and pass_name != "own" else ()
            names += (pass_name,) if pass_name not in names else ()  # a pass asked for alone ("flash")
            # one wait to the viewer: each pass's recording, then settling (progress.FREEZE.job: "step k of n")
            progress.FREEZE.job_begin(len(names) + 1)
            try:
                for name in names:
                    progress.FREEZE.job_step(PASS_PHASE.get(name, name))
                    self._prepare_inputs(dual if dual is not None else self.sensor, self.pass_integrators(name, sky)[0])
                    self._record(spp, int(width), int(height), name, dual, sky)
                progress.FREEZE.job_step("settle")
                self._absorb([self._key(spp, width, height, n, dual, sky) for n in names])
            finally:
                progress.FREEZE.job_end()
            t2 = time.perf_counter()
        # a replay whose inputs changed layout records again: report it like a recording (see progress.py)
        progress.FREEZE.begin(f"{self.name}|{self.mode}|{pass_name}|{spp}|{width}x{height}", calls=1, remember=False,
                              score=self.score)
        try:
            args = self.frozen_args[key]
            img = self.frozen[key](*args, dr.opaque(mi.UInt32, (int(seed) * repeats) & 0xFFFFFFFF))  # replays (the seed wraps in 32 bits: a seed of 4e9 x 64 repeats overflowed), launching the kernels
            for i in range(1, repeats):  # same distribution as one render at spp x repeats
                img = img + self.frozen[key](*args, dr.opaque(mi.UInt32, (int(seed) * repeats + i) & 0xFFFFFFFF))
        finally:
            progress.FREEZE.end(False)
        if repeats > 1:
            img = img / float(repeats)
        breakdown["replay_s"] = time.perf_counter() - t2
        breakdown["objects_s"] = self.__dict__.pop("objects_s", 0.0)   # pushed objects moved for this frame (set_objects)
        return img, breakdown

    def _place_shapes(self, p, hidden: frozenset) -> None:
        """Write every hideable shape as ``hidden`` has it (see _write_look): lights lifted 3 m at full
        size, other shapes collapsed to a point, the rest as loaded."""
        mi = self.mi
        for shape, m0 in self.transforms.items():
            m = m0.copy()
            if shape in hidden:
                m[1, 3] += 3.0
            p[f"{shape}.to_world"] = mi.ScalarTransform4f(m.tolist())
        for key in {k for k, _, _ in self.blocks.values()}:
            verts = self.originals[key].copy()
            for shape, (k, start, n) in self.blocks.items():
                if k == key and shape in hidden:
                    if shape in self.groups:
                        verts[start:start + n] += np.array([0.0, 3.0, 0.0], np.float32)
                    else:
                        verts[start:start + n] = verts[start]
            p[key] = mi.Float(verts.ravel())

    def _absorb(self, keys: list) -> None:
        """After a recording, the first time a shape moves (an acceleration-structure rebuild) and the first
        change of some values made every frozen function record once more (Dr.Jit: "The number of opaque
        variables in the input changed from 43831 to 44727", 64 per rectangle moved for the first time;
        ~85 s each on Device 1, then never again for that shape). Make those changes now (every light lifted
        and put back, lights and sky off, another sky map, the flash moved) and replay every recording after
        each, so the cost lands in this recording instead of on a later switch."""
        import drjit as dr

        mi = self.mi
        if not self.radiance and self.env_original is None:
            return
        source = self.sensor if self.sensor_input else self.scene
        before = sum(getattr(f, "n_recordings", 0) for f in self.frozen.values())
        lights = {s for s in self.groups if s in self.transforms or s in self.blocks}
        keys = sorted(self.frozen, key=str)  # every recording, older ones too: the scene's layout is shared
        saved: dict[str, np.ndarray] = {}
        with ACTIVITY.doing("recording (settling: one value change)"):
            other = next(iter(sorted(getattr(self.envs, "__self__", None).envmaps)), None) \
                if self.envs is not None and getattr(self.envs, "__self__", None) is not None else None
            for factor in (0.0, 1.0):
                p = mi.traverse(self.scene)
                for k in self.light_class:
                    p[k] = self._radiance(k, self._emitter_level(k, self.look) * factor)
                if self.env_original is not None:
                    p[f"{ENV_ID}.scale"] = dr.opaque(mi.Float, float(self.look.sky_level()) * factor)
                    if other is not None:  # another map's pixels, then the current sky's again
                        current = self.env_original if self.look.env == "original" else self.envs(self.look.env)
                        p[f"{ENV_ID}.data"] = mi.TensorXf(self.envs(other) if factor == 0.0 else current)
                shift = np.eye(4)
                shift[1, 3] = 1e-3 * (1.0 - factor)
                # a moved transform: the flash (an acceleration-structure rebuild)
                for k in [k for k in (self.k_flash,) if k is not None]:
                    if factor == 0.0:
                        saved[k] = np.array(p[k].matrix, dtype=np.float64).reshape(4, 4)
                    p[k] = mi.ScalarTransform4f((shift @ saved[k]).tolist())  # shifted, then exactly back
                if lights:  # every light lifted (or put down) once, then as the scene has it
                    flipped = lights - self.look.hidden if factor == 0.0 else frozenset()
                    self._place_shapes(p, (self.look.hidden - lights) | flipped | (
                        (self.look.hidden & lights) if factor == 1.0 else frozenset()))
                p.update()
                self._evaluate_transforms(p)
                for key in keys:
                    # A replay that records again bakes the film's size of the moment into the recording: a recording
                    # of another size, replayed here with the film at the size just recorded, came back at that size
                    # from then on (16 x 16 images for 792 x 514 views, after a 16 x 16 view was recorded on the same
                    # scene). Those are left alone: used next at their own size, they settle there. (Setting the film
                    # to each recording's size first changed the inputs again: recorded over and over, then CUDA 700.)
                    if key[0] != "batch" and "dual" not in key and self.size is not None \
                            and (int(key[1]), int(key[2])) != self.size:
                        continue
                    progress.FREEZE.begin(f"{self.name}|{self.mode}|{key[3]}|{key[0]}|{key[1]}x{key[2]}", calls=1,
                                          remember=False, score=self.score)
                    try:
                        dr.eval(self.frozen[key](*self.frozen_args[key], dr.opaque(mi.UInt32, 0)))
                    finally:
                        progress.FREEZE.end(False)
        after = sum(getattr(f, "n_recordings", 0) for f in self.frozen.values())
        print(f"[freeze] settle after a value change: {after - before} more recording(s)", flush=True)

    def _record(self, spp: int, width: int, height: int, pass_name: str, dual=None, sky=None) -> None:
        """Record the frozen render of one pass. The seed is an input rather than a constant, so one
        recording serves every seed (progressive viewers render the same camera again and again with new
        seeds); every input changes between frames, so all are recorded opaque rather than as literals."""
        import drjit as dr

        key = self._key(spp, width, height, pass_name, dual, sky)
        if key in self.frozen:
            return
        mi, its = self.mi, self.pass_integrators(pass_name, sky)
        progress.FREEZE.begin(f"{self.name}|{self.mode}|{pass_name}|{spp}|{width}x{height}", calls=2, score=self.score)
        ok = False
        with ACTIVITY.doing(f"recording {spp} spp {width}x{height}" + ("" if pass_name == "own" else f" {pass_name}")):
            # the integrators are inputs too: their light grid (light_grid) is arrays a recording must not capture
            if dual is not None:  # scene and dual sensor are inputs: slot poses change between frames
                frozen = dr.freeze(lambda scene, sn, it_, sd, n=spp: self.render_passes(scene, sn, it_, n, sd),
                                   backend=dr.JitBackend.CUDA, auto_opaque=False, warn_after=2)
                self.frozen_args[key] = (self.scene, dual, its)
                self._settle(frozen, self.scene, dual, its)
            elif self.sensor_input:
                scene = self.scene
                frozen = dr.freeze(lambda sn, it_, sd, n=spp: self.render_passes(scene, sn, it_, n, sd),
                                   backend=dr.JitBackend.CUDA, auto_opaque=False, warn_after=2)
                self.frozen_args[key] = (self.sensor, its)
                self._settle(frozen, self.sensor, its)
            else:
                frozen = dr.freeze(lambda scene, it_, sd, n=spp: self.render_passes(scene, None, it_, n, sd),
                                   backend=dr.JitBackend.CUDA, auto_opaque=False, warn_after=2)
                self.frozen_args[key] = (self.scene, its)
                self._settle(frozen, self.scene, its)
            self.frozen[key] = frozen  # listed by /v1/status only once both recordings are done
            ok = True
        progress.FREEZE.end(ok)

    def finish(self, handle: tuple) -> tuple[np.ndarray, dict]:
        img, breakdown = handle
        if isinstance(img, np.ndarray):
            return img, breakdown
        import drjit as dr

        t0 = time.perf_counter()
        if hasattr(img, "result"):  # a copy queued behind this view's kernels (readback.AsyncReadback)
            out = img.result()  # waits for this view alone, not for views queued after it
            return out, {**breakdown, "wait_s": time.perf_counter() - t0, "readback_s": 0.0}
        dr.sync_thread()  # wait for this view's kernels (and any queued before them)
        t1 = time.perf_counter()
        out = np.array(img)
        return out, {**breakdown, "wait_s": t1 - t0, "readback_s": time.perf_counter() - t1}

    def _prepare_inputs(self, sensor, integrator) -> None:
        """Put the scene and ``sensor`` in the state a render leaves them in, before recording with them.
        A frozen render's first call evaluates its literal inputs (every shape, emitter and sensor of the
        scene: Dr.Jit's registry) and its render prepares the film's storage, so its second call saw other
        inputs and recorded the whole render again (79 s, then 71 s on Device 1). A frozen function that
        only takes the scene and the sensor evaluates the same inputs in a moment."""
        import drjit as dr

        mi = self.mi
        integrator = integrator if integrator is not None else self.scene.integrator()
        sensor.film().prepare(integrator.aov_names())
        if getattr(self, "_touch", None) is None:
            self._touch = dr.freeze(lambda scene, sn: dr.zeros(mi.Float, 1), backend=dr.JitBackend.CUDA,
                                    auto_opaque=False)
        dr.eval(self._touch(self.scene, sensor))

    def _settle(self, frozen, *args) -> None:
        """Record, then replay once. The second call used to record the render again (79 s, then 71 s on
        Device 1: the first call had evaluated its inputs, see _prepare_inputs); it now replays, and still
        catches anything else a first render leaves different, here rather than on a viewer's next frame."""
        import drjit as dr

        for _ in range(2):
            progress.FREEZE.next_call()
            dr.eval(frozen(*args, dr.opaque(self.mi.UInt32, 0)))
            trim_host_memory()

    def render_many(self, cams: list, width: int, height: int, hfov_deg: float, spp: int, seed: int,
                    pass_name: str = "own", follow: bool | None = None, look: Look = Look()) -> list:
        """K views in one render call through a resident ``batch`` sensor.

        Tracing the scene (materials, lights, geometry) dominates a call and does not depend on the
        cameras, so one call for K views pays it once. The camera-aligned flash of an active scene
        follows a single camera, so those views still render one at a time.
        """
        follow = (self.k_flash is not None) if follow is None else bool(follow and self.k_flash is not None)
        fovs = [float(f) for f in (hfov_deg if isinstance(hfov_deg, (list, tuple)) else [hfov_deg] * len(cams))]
        # one at a time: a single view; the flash, which follows one camera; two bands through the dual sensor (an
        # RGBN variant holds both in one render, so it batches)
        if len(cams) == 1 or follow or (look.band == 2 and not self.rgbn):
            return [self.render(c, width, height, f, spp, seed, pass_name, follow, look) for c, f in zip(cams, fovs)]
        hfov_deg = fovs[0]
        if not self.parked or look != self.look:  # the batch sensor sees the scene as set here
            self._set_camera(cams[0], width, height, hfov_deg, False, look)
        mi, k = self.mi, len(cams)
        integrator = self.integrator(pass_name)
        looks = []
        for c in cams:
            origin, target, up = frames.lookat(c)
            looks.append(mi.ScalarTransform4f.look_at(origin=origin, target=target, up=up))
        key = (k, int(width), int(height), tuple(round(f, 6) for f in fovs))
        if key not in self.batch_sensors:
            spec = {"type": "batch", "film": {"type": "hdrfilm", "width": k * int(width), "height": int(height),
                                               "rfilter": {"type": RFILTER}},
                    "sampler": {"type": "independent", "sample_count": 1}}
            for i, look in enumerate(looks):   # each slot its own lens (field of view)
                spec[f"slot{i}"] = {"type": "perspective", "fov": fovs[i], "fov_axis": "x", "to_world": look}
            self.batch_sensors[key] = mi.load_dict(spec)
        sensor = self.batch_sensors[key]
        p = mi.traverse(sensor)
        for i, look in enumerate(looks):
            p[f"slot_{i}.to_world"] = look
        p.update()
        if not self.freeze:
            # the per-pass memory budget is SPP_CHUNK samples for each of the K views together
            per_pass = max(1, SPP_CHUNK[self.mode] // k)
            total, done, acc, i = int(spp), 0, None, 0
            while done < total:
                n = min(per_pass, total - done)
                img = np.array(mi.render(self.scene, sensor=sensor, integrator=integrator, spp=n,
                                         seed=(int(seed) + 104729 * i) & 0xFFFFFFFF), dtype=np.float64) * n
                acc = img if acc is None else acc + img
                done, i = done + n, i + 1
            img = (acc / total).astype(np.float32)
        else:
            import drjit as dr

            fkey = ("batch",) + key + (int(spp), pass_name)
            if fkey not in self.frozen:
                # the sensor and seed are arguments, not closure values: a frozen replay only sees updated
                # slot poses on objects it receives as inputs (production keeps its batch sensor inside the scene)
                self.frozen[fkey] = dr.freeze(
                    lambda scene, sn, it, sd, n=int(spp):
                    mi.render(scene, sensor=sn, integrator=it, spp=n, seed=sd),
                    backend=dr.JitBackend.CUDA, auto_opaque=False, warn_after=2)
                self.frozen_args[fkey] = (self.scene, sensor, integrator)   # what _absorb replays it with, as the others
                self._prepare_inputs(sensor, integrator)
                self._settle(self.frozen[fkey], self.scene, sensor, integrator)
            img = self.frozen[fkey](self.scene, sensor, integrator, dr.opaque(mi.UInt32, int(seed)))
            img = np.array(self.intensity_lanes(img) if self.unpolarized and self.rgbn else img)
        w = int(width)
        return [img[:, i * w:(i + 1) * w] for i in range(k)]


def level_offset(c2w: np.ndarray, offset) -> np.ndarray:
    """``offset`` (right, up, forward; metres) in world space, in the level frame of the view: the camera's right and
    forward with the pitch taken out, and world up. A pitched camera moves the light as a level one would."""
    right = np.array(c2w[:3, 0], dtype=np.float64)
    right[1] = 0.0
    right /= np.linalg.norm(right)
    forward = np.cross((0.0, 1.0, 0.0), right)  # columns (right, up, -forward): up x right = forward
    o = [float(v) for v in offset] + [0.0] * (3 - len(offset))
    return right * o[0] + np.array((0.0, o[1], 0.0)) + forward * o[2]


def flash_matrix(c2w: np.ndarray, size_world, distance_m: float, roll_deg: float = 0.0) -> np.ndarray:
    """Camera-aligned rectangle (port of multimodal._camera_aligned_rectangle_matrix)."""
    right = c2w[:3, 0] / np.linalg.norm(c2w[:3, 0])
    up = c2w[:3, 1] / np.linalg.norm(c2w[:3, 1])
    forward = -c2w[:3, 2] / np.linalg.norm(c2w[:3, 2])
    if abs(roll_deg) > 1e-6:
        a = np.deg2rad(roll_deg)
        right, up = right * np.cos(a) + up * np.sin(a), -right * np.sin(a) + up * np.cos(a)
        right, up = right / np.linalg.norm(right), up / np.linalg.norm(up)
    m = np.eye(4)
    m[:3, 0] = right * (float(size_world[0]) * 0.5)
    m[:3, 1] = up * (float(size_world[1]) * 0.5)
    m[:3, 2] = forward
    m[:3, 3] = c2w[:3, 3] - forward * float(distance_m)
    return m


class MainThreadExecutor:
    """Runs submitted jobs on the thread that calls run_forever(). Mitsuba keeps its variant and
    file resolver per thread, and loading an RGB scene from a worker thread segfaulted on the OptiX 7
    build, so all Mitsuba work stays on the main thread while HTTP is served from a background one."""

    def __init__(self):
        self.jobs: "queue.Queue[tuple]" = queue.Queue()
        self.main = threading.get_ident()

    def submit(self, fn, *args) -> Future:
        future = Future()
        if threading.get_ident() == self.main:  # start-up work (preload) before the loop runs
            future.set_result(fn(*args))
        else:
            self.jobs.put((fn, args, future, False))
        return future

    def submit_pipelined(self, start, *args) -> Future:
        """``start(*args)`` queues GPU work and returns ``finish()``, which completes the job. The loop starts
        the next queued job before finishing this one, so the CPU work of one view (camera update, kernel
        launch, readback, post-processing) overlaps the GPU work of its neighbour."""
        future = Future()
        self.jobs.put((start, args, future, True))
        return future

    @staticmethod
    def _complete(pending) -> None:
        finish, future = pending
        try:
            future.set_result(finish())
        except BaseException as exc:  # noqa: BLE001 - delivered to the waiting request
            future.set_exception(exc)

    def run_forever(self) -> None:
        pending = None  # (finish, future) of the job whose GPU work is in flight
        while True:
            try:  # with work in flight, take the next job only if it is already waiting
                fn, args, future, pipelined = self.jobs.get(timeout=0.002) if pending else self.jobs.get()
            except queue.Empty:
                self._complete(pending)
                pending = None
                continue
            if fn is None:
                if pending:
                    self._complete(pending)
                return
            if not pipelined:
                if pending:
                    self._complete(pending)
                    pending = None
                try:
                    future.set_result(fn(*args))
                except BaseException as exc:  # noqa: BLE001
                    future.set_exception(exc)
                continue
            try:
                finish = fn(*args)
            except BaseException as exc:  # noqa: BLE001
                future.set_exception(exc)
                continue
            if pending:
                self._complete(pending)
            pending = (finish, future)


class Renderer:
    """Mitsuba work runs on the main thread (see MainThreadExecutor); one process serialises its renders."""

    def __init__(self, pack, variant: str | None, max_resident: int, freeze: str = "auto",
                 rgb_variant: str | None = None, modes: tuple[str, ...] = ("polar",), kernel_history: bool = False,
                 envmap_dir: Path | None = None, sun_deg: float = 8.0, freeze_spp: int = 16,
                 default_sky: str | None = None, default_sky_rotate: float = 0.0, fill_gain: float = 1.0,
                 default_depth: int = 0, lamp_power: str = "importer", alpha_power: float = 1.0,
                 cycles_gpu: str = "share", cycles_device: str = "OPTIX", ser: bool = True):
        import drjit as dr
        import mitsuba as mi

        # Shader execution reordering (see ser_lock). Before anything is recorded: it is compiled into the kernels.
        self.ser = bool(ser)
        dr.set_flag(dr.JitFlag.ShaderExecutionReordering, self.ser)

        available = mi.variants()
        wanted = {"polar": variant, "rgb": rgb_variant}
        self.variant_of = {}
        for mode in modes:
            chosen = wanted[mode] or next((v for v in MODE_VARIANTS[mode] if v in available), None)
            if chosen not in available:
                raise SystemExit(f"no Mitsuba variant for {mode} mode (available: {available})")
            self.variant_of[mode] = chosen
        # one or more packs (--pack repeated): a scene comes from the first pack that has it
        self.packs = [Path(p) for p in (pack if isinstance(pack, (list, tuple)) else [pack])]
        self.mi, self.dr, self.pack = mi, dr, self.packs[0]
        # frames read back behind their own kernels, so the next frame's CPU half overlaps this one's GPU
        # (readback.py); OPTICALNAV_ASYNC_READBACK=0 reads back as before (a whole-stream sync)
        self.readback = None
        if os.environ.get("OPTICALNAV_ASYNC_READBACK", "1") != "0":
            from . import readback
            self.readback = readback.AsyncReadback.create()
        print(f"[serve] async readback: {'on' if self.readback is not None else 'off'}", flush=True)
        self.variant = next(iter(self.variant_of.values()))
        mi.set_variant(self.variant)
        # Per-kernel codegen / compile / GPU times split a frame into tracing vs ray tracing, but reading the
        # history synchronises the device, which stops one view's GPU work overlapping the next one's CPU
        # work. Off by default; --kernel-history turns it on for measurements (benchmark_modes, frame_overhead).
        self.kernel_history = bool(kernel_history)
        dr.set_flag(dr.JitFlag.KernelHistory, self.kernel_history)
        # ponytail: one render loop per server process; run one server per GPU to scale out.
        self.executor = MainThreadExecutor()
        self.max_resident = max(1, max_resident)
        self.meta, self.pack_of = {}, {}  # scan -> scene.json, scan -> its pack
        for p in self.packs:
            for scene_json in sorted((p / "scenes").glob("*/scene.json")):
                scan = scene_json.parent.name
                if scan in self.pack_of:
                    print(f"[pack] {scan} is in {self.pack_of[scan]} and {p}: serving the first", flush=True)
                    continue
                self.meta[scan], self.pack_of[scan] = json.loads(scene_json.read_text()), p
        # resident scenes by (scan, xml, flash_xml, mode); a variant renders on one of them (see plan)
        self.cache: OrderedDict[tuple, Resident] = OrderedDict()
        self.body_models: dict[str, dict] = {}  # name -> bodies.build_model (POST /v1/bodies)
        self.plans: dict[tuple[str, str, str], tuple] = {}
        self.merge_checked: set[str] = set()   # scans merge_if_needed has looked at in this process
        self.merge_lock = threading.Lock()
        self.denoisers: dict[tuple[str, int, int], object] = {}
        # path_nocaustics with camera-assist modes (an older build of the plugin has none: active variants
        # that need it are hidden, and one-pass active variants keep a resident scene of their own)
        self.progress_log = progress.install()  # Dr.Jit's log feeds freeze-recording progress
        self.has_rgb2nir = self._has_plugin({"type": "rgb2nir", "nested": {"type": "rgb", "value": [0.5, 0.5, 0.5]}})
        self.has_nocaustics = self._has_plugin({"type": "path_nocaustics", "assist": ASSIST_PREFIX,
                                                "assist_mode": "exclude"})
        # path_nocaustics's "split" mode: an active frame on one path (an older plugin has none: two paths)
        self.has_split = self.has_nocaustics and self._has_plugin({"type": "path_nocaustics", "assist": ASSIST_PREFIX,
                                                                    "assist_mode": "split"})
        # path_nocaustics's "hidden": the room softboxes stay out of the camera's view (an older plugin: they show)
        self.has_hidden = self.has_nocaustics and self._has_plugin({"type": "path_nocaustics", "hidden": FILL_PREFIX})
        # an older plugin lacks ``hidden_specular``: such lights then show in reflections and through glass
        Resident.hidden_specular = self.has_hidden and self._has_plugin(
            {"type": "path_nocaustics", "hidden": HIDDEN_LIGHT_PREFIX, "hidden_specular": HIDDEN_SPEC_PREFIX})
        # path_nocaustics's "unpolarized" (2026-11-26): the rgb mode renders the polarized RGBN scene without Mueller
        # matrices (S0 per lane, 5.5 ms against 67 ms a 512x384 16 spp CS02 frame, the cuda_rgbn variant's own 5.4 ms):
        # one scene in memory for both modes, one load, recordings of one scene
        self.shared_rgb = False
        if "polar" in self.variant_of and "_rgbn_polarized" in self.variant_of["polar"] and self.has_nocaustics:
            prev = mi.variant()
            mi.set_variant(self.variant_of["polar"])
            try:
                probe = mi.load_dict({"type": "path_nocaustics", "unpolarized": True})
                self.shared_rgb = "unpolarized = true" in str(probe)
            except Exception:  # noqa: BLE001 - an older plugin: the rgb mode keeps its own scene
                self.shared_rgb = False
            mi.set_variant(prev)
            if self.shared_rgb and "rgb" in self.variant_of:
                self.variant_of["rgb"] = self.variant_of["polar"]
        self.freeze = hasattr(dr, "freeze") if freeze == "auto" else freeze == "on"
        self.preloading: list[list[str]] = []  # [scan, variant, mode] of --preload loads not done yet (main)
        # environment maps a view may ask for by name (file stem): <pack>/envmaps unless --envmaps says otherwise
        folder = Path(envmap_dir) if envmap_dir else next((p / "envmaps" for p in self.packs if (p / "envmaps").is_dir()),
                                                          self.pack / "envmaps")
        self.envmaps = {f.stem: f for f in sorted(folder.glob("*")) if f.suffix.lower() in (".exr", ".hdr")} \
            if folder.is_dir() else {}
        self.env_cache: dict[str, np.ndarray] = {}
        self.sky_gain: dict[str, float] = {}
        self.sun_deg = float(sun_deg)
        self.default_sky, self.default_sky_rotate = default_sky, float(default_sky_rotate)  # see sky_for
        self.fill_gain, self.default_depth = float(fill_gain), int(default_depth)  # the simulator's own scenes, see own_scene
        self.alpha_power = float(alpha_power)
        self.lamp_power = lamp_power  # "blender": the simulator's own scenes light at their Blender watts
        self.freeze_spp = int(freeze_spp)
        # the Cycles backend (cycles_backend.py): one Blender process, for the last scene asked for. "share": it and the
        # Mitsuba scenes stay on the GPU together, each freed for the other when the GPU runs out of memory;
        # "exclusive": asking for one frees the other at once
        self.cycles_gpu, self.cycles_device = cycles_gpu, cycles_device
        self.cycles = None  # (scan, CyclesWorker)
        for scan in self.meta:
            synthesize_variants(self.meta[scan], (self.scene_dir(scan) / "perturbation_pool.json").is_file(),
                                self.has_nocaustics)
            self.register_sky(scan)

    blend_pack: Path | None = None
    blend_files: dict = {}  # id -> .blend offered in the viewer's scene list (--blend-dir / --blend-tsv), converted when chosen

    def register_body(self, name: str, descriptor: dict) -> dict:
        """A body model views may place (bodies.py): its links' meshes are written once, by the descriptor's hash."""
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", str(name)):
            raise ValueError("a body model's name is 1..64 letters, digits, '_', '.', '-'")
        links = dict(descriptor.get("links") or {})
        if not links or len(links) > 256 or sum(len(v) for v in links.values()) > 2048:
            raise ValueError("a body model has 1..256 links and at most 2048 primitives")
        model = body_models.build_model(str(name), {"links": links})
        self.body_models[str(name)] = model
        return {"model": model["name"], "key": model["key"], "links": sorted(model["links"])}

    def register_scene(self, pack: Path, scan: str) -> None:
        """Serve a scene that appeared after the start (one just converted from its .blend)."""
        if pack not in self.packs:
            self.packs.append(pack)
        self.meta[scan] = json.loads((pack / "scenes" / scan / "scene.json").read_text())
        self.pack_of[scan] = pack
        self.plans = {k: v for k, v in self.plans.items() if k[0] != scan}
        synthesize_variants(self.meta[scan], (pack / "scenes" / scan / "perturbation_pool.json").is_file(), self.has_nocaustics)
        self.register_sky(scan)

    def blend_list(self) -> list[dict]:
        """Every .blend the viewer may pick: id, file, and whether its scene is already served."""
        out = {sid: {"id": sid, "path": str(path), "ready": sid in self.meta} for sid, path in self.blend_files.items()}
        for scan, meta in self.meta.items():  # served ones, whatever their pack, with their source file
            if meta.get("blend", {}).get("linked") or meta.get("converter"):
                out.setdefault(scan, {"id": scan, "path": (meta.get("converter") or {}).get("source") or meta["blend"]["file"], "ready": True})
        return sorted(out.values(), key=lambda b: b["id"])

    def open_blend(self, sid_or_path: str) -> str:
        """Convert (once) and serve a .blend by listed id or file path; returns the scene id."""
        from . import blend_scenes
        if sid_or_path in self.blend_files:
            sid, blend = sid_or_path, Path(self.blend_files[sid_or_path])
        else:
            sid, blend = blend_scenes.parse_spec(sid_or_path)
        with ACTIVITY.doing(f"converting {sid} in Blender"):
            blend_scenes.ensure(self.blend_pack, sid, blend, moved=getattr(self, "blend_moves", {}))
        self.register_scene(self.blend_pack, sid)  # also when it was converted again: its scene.json is read anew
        return sid

    def register_sky(self, scan: str) -> None:
        """The world a scene converted from its .blend carries (scene.json "sky": the Blender world rendered as an equirectangular
        EXR, strength included) as an environment map named ``<scan>:sky``."""
        sky = self.meta[scan].get("sky")
        if sky and (self.scene_dir(scan) / sky["file"]).is_file():
            self.envmaps[f"{scan}:sky"] = self.scene_dir(scan) / sky["file"]

    def scene_dir(self, scan: str) -> Path:
        return self.pack_of.get(scan, self.pack) / "scenes" / scan

    def blend_of(self, scan: str) -> tuple[Path, tuple] | None:
        """(the scene's .blend, its origin offset) when scene.json carries one (tools/attach_blend.py), else None."""
        blend = self.meta[scan].get("blend")
        if not blend:
            return None
        path = self.scene_dir(scan) / blend["file"]
        if not path.is_file():
            return None
        if blend.get("origin"):   # a working copy (blend_paths): Cycles needs where the original is, for its relative paths
            return path, tuple(blend["origin_offset"]), {"origin": blend["origin"], "moves": blend.get("moves")}
        return path, tuple(blend["origin_offset"])

    def cycles_worker(self, scan: str):
        """The Blender process of a scene's .blend, started when needed (the one of another scene is closed first)."""
        from . import cycles_backend
        if self.cycles is not None and self.cycles[0] == scan and self.cycles[1].alive():
            return self.cycles[1]
        self.free_cycles()
        blend = self.blend_of(scan)
        if blend is None:
            raise ValueError(f"{scan} has no .blend (tools/attach_blend.py)")
        if self.cycles_gpu == "exclusive":
            self.free_mitsuba()
        def stage(name: str) -> None:   # the .blend is open; Cycles prepares its GPU kernels for this scene's features
            ACTIVITY.rename(f"preparing Cycles kernels for {scan}")
        with ACTIVITY.doing(f"loading {scan} in Blender"):
            try:
                worker = cycles_backend.CyclesWorker(blend[0], blend[1], self.cycles_device, on_stage=stage,
                                                     paths=blend[2] if len(blend) > 2 else None)
            except cycles_backend.CyclesError as exc:
                if not cycles_backend.is_oom(exc):
                    raise
                self.free_mitsuba()  # the GPU is full: hand the Mitsuba scenes' memory back and try again
                ACTIVITY.rename(f"loading {scan} in Blender")
                worker = cycles_backend.CyclesWorker(blend[0], blend[1], self.cycles_device, on_stage=stage,
                                                     paths=blend[2] if len(blend) > 2 else None)
        self.cycles = (scan, worker)
        print(f"[cycles] {scan}: {blend[0].name} ready in {worker.load_s:.0f}s (GPU kernels {worker.info.get('warm_s')}s) {worker.info}", flush=True)
        return worker

    def free_cycles(self) -> bool:
        """Stop the Blender process (its GPU memory goes back); True when there was one."""
        if self.cycles is None:
            return False
        scan, worker = self.cycles
        self.cycles = None
        worker.close()
        print(f"[cycles] {scan} freed", flush=True)
        return True

    def free_mitsuba(self) -> bool:
        """Drop every resident Mitsuba scene and Dr.Jit's cached GPU buffers (main thread; from elsewhere it queues)."""
        if threading.current_thread() is not threading.main_thread():
            return self.executor.submit(self.free_mitsuba).result()
        had = bool(self.cache)
        self.cache.clear()
        gc.collect()
        self.dr.flush_malloc_cache()
        if had:
            print("[mitsuba] resident scenes freed", flush=True)
        return had

    def refresh_physics(self, scan: str) -> dict:
        """Decisions set by hand changed (the viewer's 고정 / 밂 / 움직임: physics_overrides.json): find the movable objects
        and merge again now (merge_if_needed), and drop the loaded scene when that changed its files (the next view
        loads it again, the objects to push kept apart as the new decisions say)."""
        if scan not in self.meta:
            raise ValueError(f"unknown scan {scan!r}")
        with self.merge_lock:
            self.merge_checked.discard(scan)
        changed = bool(self.merge_if_needed(scan))
        if changed:
            self.drop_scene(scan)
        return {"changed": changed}

    def drop_scene(self, scan: str) -> bool:
        """The resident scenes of ``scan`` dropped (main thread; from elsewhere it queues)."""
        if threading.current_thread() is not threading.main_thread():
            return self.executor.submit(self.drop_scene, scan).result()
        keys = [k for k in self.cache if k[0] == scan]
        for k in keys:
            del self.cache[k]
        self.last_solid = None
        gc.collect()
        self.dr.flush_malloc_cache()
        if keys:
            print(f"[mitsuba] {scan} dropped: its files changed", flush=True)
        return bool(keys)

    def merge_if_needed(self, scan: str) -> bool:
        """The first time a scene is opened in this process: its static meshes of one material merged into one
        (tools/merge_meshes.py --if-needed, a subprocess: its Mitsuba runs scalar_rgb, which must not switch this
        process's variant), then its scene.json read again. Thousands of shapes in a scene file left the loaded meshes
        with registry ids in the thousands, and with a robot (instances) in the scene every frozen replay rebuilt call
        tables that long (REF_CS02: 4964 shapes -> 215, a replay 16.5 -> 8.6 ms, the load 0.8 -> 0.1 s). A scene already
        merged costs a text scan (~2 s with the interpreter's start); a failure serves the scene as it is.
        OPTICALNAV_SIM_MERGE=0 turns it off."""
        import subprocess
        import sys

        with self.merge_lock:
            if scan in self.merge_checked or os.environ.get("OPTICALNAV_SIM_MERGE", "1") == "0":
                return False
            self.merge_checked.add(scan)
            tools = Path(__file__).resolve().parents[1] / "tools"
            if not (tools / "merge_meshes.py").is_file() or scan not in self.pack_of:
                return False
            started = time.monotonic()
            changed = False
            # first which objects a robot could push (tools/physics_tag.py: physics.json), then the merge, which keeps
            # those apart as one body each (physics.keep: the reachable ones by default)
            for tool, what in (("physics_tag.py", "finding movable objects in"), ("merge_meshes.py", "merging meshes of")):
                if not (tools / tool).is_file():
                    continue
                with ACTIVITY.doing(f"{what} {scan}"):
                    try:
                        done = subprocess.run([sys.executable, str(tools / tool), "--pack", str(self.pack_of[scan]),
                                               "--scene", scan, "--if-needed"], capture_output=True, text=True, timeout=3600)
                    except subprocess.TimeoutExpired:
                        print(f"[merge] {scan}: {tool} gave no answer in an hour, served as it is", flush=True)
                        return changed
                said = (done.stdout or "").strip()
                if done.returncode:
                    print(f"[merge] {scan}: {tool} failed (exit {done.returncode}), served as it is: "
                          f"{(done.stderr or said)[-800:]}", flush=True)
                    return changed
                if "nothing to do" not in said:
                    changed = True
                    print(f"[merge] {said[-400:]}", flush=True)
            if changed:
                self.reload_meta(scan)
                print(f"[merge] {scan} ready in {time.monotonic() - started:.0f} s", flush=True)
            return changed

    def object_shapes(self, scan: str) -> dict:
        """Physics object id -> the dyn_ shapes that draw it (physics.json's merge record, tools/merge_meshes.py), for the
        objects kept apart as bodies; read again when the file changes."""
        path = self.scene_dir(scan) / "physics.json"
        try:
            stamp = path.stat().st_mtime_ns
        except OSError:
            return {}
        cache = self.__dict__.setdefault("_object_shapes", {})
        if cache.get(scan, (None,))[0] != stamp:
            merge = (json.loads(path.read_text()).get("merge") or {}).get("objects") or {}
            cache[scan] = (stamp, {oid: [s for s in v.get("shapes", []) if s.startswith("dyn_")]
                                   for oid, v in merge.items() if v.get("kept")})
        return cache[scan][1]

    def reload_meta(self, scan: str) -> None:
        """Read a scene's scene.json again (after compact_on_start rewrote it); its plans go with the old one."""
        self.meta[scan] = json.loads((self.scene_dir(scan) / "scene.json").read_text())
        synthesize_variants(self.meta[scan], (self.scene_dir(scan) / "perturbation_pool.json").is_file(),
                            self.has_nocaustics)
        self.plans = {k: v for k, v in self.plans.items() if k[0] != scan}
    def pool(self, scan: str) -> dict | None:
        """The scene's perturbation pool (scenes/<scan>/perturbation_pool.json), if built."""
        path = self.scene_dir(scan) / "perturbation_pool.json"
        return json.loads(path.read_text()) if path.is_file() else None

    def own_scene(self, scan: str, variant: str = "base") -> bool:
        """A scene staged without a render (``staged_without_render_v1``): the simulator's own, with no dataset
        frames to match, so its look defaults (sky, fill, depth) may differ from the dataset's."""
        variants = self.meta[scan]["variants"]
        entry = variants.get(variant, {})
        if entry.get("synthetic"):  # "perturbed", "active_polar": the viewer's flash and mirrors on the base scene file
            entry = variants.get("base", {})
        return entry.get("protocol") in OWN_PROTOCOLS

    def direct_scene(self, scan: str) -> bool:
        """A scene converted straight from its .blend (tools/blend_to_pack.py): its alpha is already Blender's roughness
        squared, so ``--alpha-power`` is not applied again."""
        return self.meta[scan]["variants"].get("base", {}).get("protocol") == "blend_direct_v1"

    def sky_for(self, scan: str, variant: str) -> tuple[str, float, float] | None:
        """(map, gain, rotation) of the default sky for a view that asks for the scene's own environment, or None.

        A scene staged without a render (the simulator's own scenes, ``staged_without_render_v1``) has only a
        flat constant sky (radiance 0.55 0.57 0.60): no sun, no gradient, no horizon, so light through a window
        has no direction and no shadow. They get ``--default-sky`` instead, an HDR sky with its sun (spread over
        --sun-deg), scaled so its mean radiance equals the constant's (the interior lights were set against it,
        so the exterior stays as bright overall). The dataset's scenes keep their own sky: their frames are
        compared with the stored ones."""
        sky = self.meta[scan].get("sky")
        if sky and f"{scan}:sky" in self.envmaps and self.direct_scene(scan):
            return f"{scan}:sky", float(sky.get("gain", 1.0)), float(sky.get("rotation_deg", 0.0))  # its own world, as Blender has it
        if not self.default_sky or self.default_sky not in self.envmaps:
            return None
        if not self.own_scene(scan, variant):
            return None
        if self.default_sky not in self.sky_gain:
            data = self.env_image(self.default_sky)
            rows = np.cos(np.linspace(np.pi / 2, -np.pi / 2, data.shape[0], dtype=np.float32))[:, None]  # solid angle
            mean = float(((data @ stokes.LUMA) * rows).sum() / (rows.sum() * data.shape[1]))
            self.sky_gain[self.default_sky] = float(np.mean(SCENE_SKY)) / max(mean, 1e-6)
        return self.default_sky, self.sky_gain[self.default_sky], self.default_sky_rotate

    def env_image(self, name: str) -> np.ndarray:
        """An environment map of the --envmaps directory at ENV_RES (loaded once)."""
        if name not in self.env_cache:
            self.env_cache[name] = env_data(self.mi, self.envmaps[name], sun_deg=self.sun_deg)
        return self.env_cache[name]

    def _has_plugin(self, spec: dict) -> bool:
        try:
            self.mi.load_dict(spec)
            return True
        except Exception:  # noqa: BLE001 - absent plugin raises a variant-specific error type
            return False

    def variants(self, scan: str) -> list[str]:
        return [name for name, v in self.meta[scan]["variants"].items()
                if self.has_nocaustics or not v.get("flash_xml")]

    def scene_mode(self, mode: str) -> str:
        """The mode whose resident scene renders ``mode``: the rgb mode on the polarized scene when shared_rgb."""
        return "polar" if mode == "rgb" and self.shared_rgb else mode

    def _on_main(self, mode: str, fn, *args):
        def job():
            if self.mi.variant() != self.variant_of[mode]:
                self.mi.set_variant(self.variant_of[mode])
            return fn(*args)
        return self.executor.submit(job).result()

    def preload(self, scan: str, variant: str, mode: str = "polar") -> float:
        mode = self.scene_mode(mode)
        return self._on_main(mode, lambda: self._resident(self.plan(scan, variant, mode)[0], variant).load_s)

    def plan(self, scan: str, variant: str, mode: str) -> tuple[tuple, tuple[str, ...], bool, frozenset]:
        """(resident scene key, passes, flash follows the camera, hidden shape ids) for one variant.

        One resident scene per scan hosts every variant it can: the active variant's scene (the one with
        a flash; needs path_nocaustics) or else perturbed's. A variant whose scene file is the host's minus
        some shapes renders on the host with those shapes hidden: base is perturbed without its mirror and
        glass cubes, perturbed is the active scene without its flash. On an active host the "passive" pass
        ignores the flash, and a two-pass active variant adds a "flash" pass. Switching among them edits
        memory and loads nothing; each pass records its freeze once per spp and size."""
        key = (scan, variant, mode)
        if key in self.plans:
            return self.plans[key]
        self.merge_if_needed(scan)   # before anything reads the scene's files (the variants' shapes, its scene.json)
        variants = self.meta[scan]["variants"]
        spec = variants[variant]
        if spec.get("flash_xml") and not self.has_nocaustics:
            raise ValueError(f"variant {variant!r} needs the path_nocaustics integrator, absent from this Mitsuba build")
        lit_name = next((name for name, v in variants.items() if v.get("flash")), None)
        usable = lit_name is not None and (self.has_nocaustics or not variants[lit_name].get("flash_xml"))
        host_name = lit_name if usable else "perturbed" if "perturbed" in variants else None
        hidden = self._hidden(scan, variant, host_name) if host_name is not None else None
        if hidden is not None:
            host = variants[host_name]
            rkey = (scan, host["xml"], host.get("flash_xml"), mode)
            # an old one-pass active scene is rendered as the current protocol too (upgrade_flash_text): two passes
            two_pass = bool(host.get("flash_xml")) or (self.has_nocaustics and bool(host.get("flash")))
            if two_pass:  # the passive pass leaves the camera-assist shapes out by itself; never hide them
                hidden = tuple(s for s in hidden if not s.startswith(ASSIST_PREFIX))
            # Two-pass: "passive" ignores the flash (it sits at the camera when the light is on) and "flash"
            # adds it. One pass: the scene's own integrator for all variants, the flash and its polarizer
            # shrunk to nothing while the light is off, so on and off share one freeze recording.
            if variant == host_name and host.get("flash"):
                plan = (rkey, ("passive", "flash") if two_pass else ("own",), True, frozenset())
            else:
                plan = (rkey, ("passive",) if two_pass else ("own",), False, frozenset(hidden))
        else:
            plan = ((scan, spec["xml"], None, mode), ("own",), bool(spec.get("flash")), frozenset())
        self.plans[key] = plan
        return plan

    def _hidden(self, scan: str, variant: str, host_name: str) -> tuple[str, ...] | None:
        """Shape ids to hide on the host's scene to get this variant's, or None when it is not the host's
        scene minus some shapes (see plan)."""
        variants = self.meta[scan]["variants"]
        spec, host = variants[variant], variants[host_name]
        if variant == host_name:
            return ()
        if spec.get("flash"):
            return None
        if spec["xml"] == host["xml"]:  # two-pass: the active variant's passive scene file
            return ()
        base = self.scene_dir(scan)
        return extra_shapes((base / host["xml"]).read_text(), (base / spec["xml"]).read_text())

    def served(self) -> list[tuple[tuple, Resident, tuple]]:
        """((scan, variant, mode), resident scene, passes) for every variant a resident scene renders."""
        cache, plans = dict(self.cache), list(self.plans.items())
        return [(key, cache[plan[0]], plan[1]) for key, plan in plans if plan[0] in cache]

    @staticmethod
    def recorded(resident: Resident, passes: tuple) -> list:
        """[spp, width, height] of the single-view freeze recordings that cover every pass."""
        keys = [f for f in list(resident.frozen) if f[0] != "batch"]
        run = ("active",) if tuple(passes) == ("passive", "flash") else tuple(passes)
        return sorted(list(s) for s in {f[:3] for f in keys} if all(any(f[:4] == s + (p,) for f in keys) for p in run))

    def _resident(self, rkey: tuple, variant: str) -> Resident:
        if rkey in self.cache:
            self.cache.move_to_end(rkey)
            return self.cache[rkey]
        scan, xml, flash_xml, mode, *with_bodies = rkey
        if self.cycles is not None and self.cycles_gpu == "exclusive":
            self.free_cycles()
        while len(self.cache) >= self.max_resident:
            self.cache.popitem(last=False)
            gc.collect()
            # Dr.Jit keeps freed GPU buffers in its allocation cache; hand them back before the next scene
            # loads, or the old scene's memory and the new one's add up (28 of 32.6 GB on Device 1 while a
            # variant switch loaded, with another process holding 8.8 GB) and freeze recordings fail
            self.dr.flush_malloc_cache()
            trim_host_memory()
        base = self.scene_dir(scan)  # XML asset paths are relative to this directory
        # the flash of the active variant this scene file carries (or, two-pass, takes from flash_xml)
        flash = next((v["flash"] for v in self.meta[scan]["variants"].values()
                      if v.get("flash") and v["xml"] == xml and v.get("flash_xml") == flash_xml), None)
        hideable = set()  # shapes some variant on this scene leaves out (plans also feed /v1/status)
        for other in self.variants(scan):
            with contextlib.suppress(Exception):
                plan = self.plan(scan, other, mode)
                if plan[0] == rkey:
                    hideable |= plan[3]
        with ACTIVITY.doing(f"loading {scan} {variant}"):
            own = self.own_scene(scan, variant)
            loaded = self._load_resident(
                self.mi, base / xml, mode, flash=flash, freeze=self.freeze,
                flash_xml=base / flash_xml if flash_xml else None, hideable=hideable,
                envs=self.env_image, freeze_spp=self.freeze_spp, nir=self.has_rgb2nir,
                pool=self.pool(scan), upgrade_flash=self.has_nocaustics,
                hidden=((HIDDEN_LIGHT_PREFIX if self.direct_scene(scan) else FILL_PREFIX)
                        if getattr(self, "has_hidden", False) else ""),
                fill_gain=self.fill_gain if own else 1.0,
                blender_power=self.lamp_power == "blender" and own,
                alpha_power=self.alpha_power if own and not self.direct_scene(scan) else 1.0,
                watt_gain=math.pi if self.direct_scene(scan) else 1.0,
                bodies=body_models.scene_xml(self.body_models, [(b, name) for b, name, _ in with_bodies[0]])
                if with_bodies else "",
                objects=self.object_shapes(scan) if with_bodies and OBJECT_INSTANCES else None)   # pushed: instances
        loaded.scene_dir, loaded.scene_xml = base, xml   # the Solid preview's object filters read the scene's parts
        self.cache[rkey] = loaded
        return loaded

    def _load_resident(self, *args, **kwargs) -> Resident:
        """Resident(...), but a load that runs the GPU out of memory while the Blender process holds some gives that back
        and tries once more."""
        try:
            return Resident(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - Dr.Jit raises RuntimeError("... out of memory ...")
            from . import cycles_backend
            if self.cycles is None or not cycles_backend.is_oom(exc):
                raise
            self.free_cycles()
            gc.collect()
            self.dr.flush_malloc_cache()
            return Resident(*args, **kwargs)

    def _denoise(self, mode: str, img: np.ndarray) -> np.ndarray:
        """OptiX AI denoiser on one non-negative HxWx3 radiance image (runs on the mode's worker), lone fireflies
        brought down to their neighbours first (despeckle): the denoiser keeps an isolated bright pixel as a feature."""
        img = despeckle(np.clip(np.nan_to_num(img), 0.0, None))
        h, w = img.shape[:2]
        key = (mode, w, h)
        if key not in self.denoisers:
            self.denoisers[key] = self.mi.OptixDenoiser(input_size=self.mi.ScalarVector2u(w, h))
        noisy = self.mi.TensorXf(np.ascontiguousarray(np.clip(np.nan_to_num(img), 0.0, None), dtype=np.float32))
        return np.array(self.denoisers[key](noisy))[:, :, :3]

    def _denoise_stokes(self, mode: str, s0, s1, s2) -> tuple:
        """An intensity denoiser cannot take signed S1/S2: denoise what linear polarizers at 0/90/45/135
        degrees would see, then recombine (S3 stays as given)."""
        i0, i90 = self._denoise(mode, (s0 + s1) / 2), self._denoise(mode, (s0 - s1) / 2)
        i45, i135 = self._denoise(mode, (s0 + s2) / 2), self._denoise(mode, (s0 - s2) / 2)
        return (i0 + i90 + i45 + i135) / 2, i0 - i90, i45 - i135

    def denoise_images(self, arrays: dict) -> dict:
        """OptiX-denoise finished images (``s0``, ``s1``, ``s2`` or ``radiance``, HxWx3, and/or the
        near-infrared ``nir_s0``, ``nir_s1``, ``nir_s2``, HxW): a progressive viewer accumulates raw passes
        and denoises the running mean, rather than averaging denoised passes, which keeps only what every
        pass's denoiser agreed on and smears detail. Both bands in one call: one trip to the main thread."""
        mode = "polar" if "s0" in arrays or "nir_s0" in arrays else "rgb"
        mode = self.scene_mode(mode) if hasattr(self, "scene_mode") else mode
        mode = mode if mode in self.variant_of else next(iter(self.variant_of))

        def job():
            out = {}
            if "s0" in arrays:
                s0, s1, s2 = (np.asarray(arrays[k], dtype=np.float32) for k in ("s0", "s1", "s2"))
                out.update(zip(("s0", "s1", "s2"), self._denoise_stokes(mode, s0, s1, s2)))
            if "nir_s0" in arrays:  # grey: denoised as three equal channels
                grey = [np.repeat(np.asarray(arrays[f"nir_s{k}"], dtype=np.float32)[..., None], 3, axis=2)
                        for k in range(3)]
                out.update({f"nir_s{k}": v[..., 0] for k, v in enumerate(self._denoise_stokes(mode, *grey))})
            if "radiance" in arrays:
                out["radiance"] = self._denoise(mode, np.asarray(arrays["radiance"], dtype=np.float32))
            if "nir" in arrays:  # an unpolarized near-infrared intensity (HxW): one run, as three equal channels
                grey = np.repeat(np.asarray(arrays["nir"], dtype=np.float32)[..., None], 3, axis=2)
                out["nir"] = self._denoise(mode, grey)[..., 0]
            return out
        return self._on_main(mode, job)

    @staticmethod
    def pool_selection(pool: dict, mirrors: tuple) -> tuple[list[dict], list[dict]]:
        """(mirrors, glass walls) of a pool a view's ``"mirrors"`` shows: a density's share of the mirrors
        (density in robomituba's units, the pool holding ``density_max``'s worth), and the first glass walls by
        rank, only those that leave the R2R graph whole unless ``glass_blocking``."""
        density, glass, blocking = mirrors
        n = min(len(pool["mirrors"]), int(round(len(pool["mirrors"]) * density / max(pool["density_max"], 1e-9))))
        walls = [g for g in pool["glass"] if blocking or not g["splits_r2r"]][:glass]
        return pool["mirrors"][:n], walls

    def _pool_hidden(self, resident: Resident, mirrors: tuple | None) -> frozenset:
        """Pool shapes to hide: all of them for the variant's own perturbation, else the ones the view's share
        leaves out together with the dataset's own mirrors and glass walls."""
        pool = resident.pool
        if not pool:
            if mirrors is not None:
                raise ValueError("this scene has no perturbation pool (tools/build_perturbation_pool.py)")
            return frozenset()
        every = {m["id"] for m in pool["mirrors"]} | {g["id"] for g in pool["glass"]}
        if mirrors is None:
            return frozenset(every)
        shown_m, shown_g = self.pool_selection(pool, mirrors)
        return frozenset(every - {m["id"] for m in shown_m} - {g["id"] for g in shown_g}) | resident.dataset_perturbation

    def _check(self, view: dict) -> tuple:
        """Validate one view; returns (group key, camera_to_world)."""
        scan, variant = view["scan"], view["variant"]
        mode, denoise = view.get("mode", "polar"), bool(view.get("denoise", False))
        if scan not in self.meta:
            raise ValueError(f"unknown scan {scan!r}")
        if variant not in self.meta[scan]["variants"]:
            raise ValueError(f"unknown variant {variant!r} for {scan}")
        if mode not in self.variant_of:
            raise ValueError(f"mode {mode!r} unavailable on this server (have {sorted(self.variant_of)})")
        c2w = np.asarray(view["camera_to_world"], dtype=np.float64).reshape(4, 4)
        width, height = int(view["width"]), int(view["height"])
        spp, seed, hfov = int(view["spp"]), int(view.get("seed", 0)), float(view["hfov_deg"])
        if not (0 < width * height <= MAX_PIXELS and 0 < spp <= MAX_SPP and 1.0 <= hfov <= 170.0):
            raise ValueError("width/height/spp/hfov_deg out of range")
        if not np.all(np.isfinite(c2w)):
            raise ValueError("camera_to_world must be finite")
        lights = float(view.get("lights", 1.0))
        env = view.get("env") or {}
        env = {"map": env} if isinstance(env, str) else dict(env)
        env_map, env_scale, env_rot = str(env.get("map", "original")), float(env.get("scale", 1.0)), float(env.get("rotate_deg", 0.0))
        if not (0.0 <= lights <= 1e3 and 0.0 <= env_scale <= 1e3 and np.isfinite(env_rot)):
            raise ValueError("lights / env scale out of range")
        if env_map != "original" and env_map not in self.envmaps:
            raise ValueError(f"unknown environment map {env_map!r} (have {sorted(self.envmaps)})")
        if env_map == "original" and (sky := self.sky_for(scan, variant)) is not None:  # the simulator's scenes: a real sky
            env_map, env_scale, env_rot = sky[0], env_scale * sky[1], env_rot + sky[2]
        bands, flash_band = str(view.get("bands", "rgb")), str(view.get("flash_band", "rgb"))
        if bands not in ("rgb", "nir", "both") or flash_band not in ("rgb", "nir", "both"):
            raise ValueError("bands and flash_band are 'rgb', 'nir' or 'both'")
        if bands != "rgb" and not self.has_rgb2nir:
            raise ValueError("near-infrared needs the rgb2nir plugin in this Mitsuba build (mitsuba/rgb2nir.cpp)")
        nir = dict(NIR_DEFAULTS)
        nir.update({k: float(v) for k, v in dict(view.get("nir") or {}).items() if k in NIR_DEFAULTS})
        lighting = str(view.get("lighting", "all"))
        if lighting not in LIGHTING:
            raise ValueError(f"lighting must be one of {sorted(LIGHTING)}")
        lamps = bool(view.get("lamps", False))  # also light the lamps the scene left dark (lamp_lights_text)
        flash_scale = float(view.get("flash_scale", 1.0))  # a larger flash at the same power (Look.flash_scale)
        if not 0.25 <= flash_scale <= 8.0:
            raise ValueError("flash_scale must be 0.25..8")
        flash_offset = tuple(float(v) for v in (view.get("flash_offset") or (0.0, 0.0, 0.0)))  # right, up, forward (m)
        if len(flash_offset) not in (2, 3) or not all(abs(v) <= 2.0 for v in flash_offset):
            raise ValueError("flash_offset is [right, up, forward] in metres, each within 2")
        flash_offset = (flash_offset + (0.0,))[:3]
        max_depth = int(view.get("max_depth") or 0)  # every pass's path length cap; 0 or absent: the scene's own
        if not max_depth and self.default_depth and self.own_scene(scan, variant):
            max_depth = self.default_depth
        rr_depth = int(view.get("rr_depth") or 0)  # Russian roulette from this depth; 0 or absent: Mitsuba's 5
        if not 0 <= rr_depth <= 64:
            raise ValueError("rr_depth must be 0 (Mitsuba's 5) .. 64")
        caustics = bool(view.get("caustics", True))  # False: the lights' passes drop caustics as the flash pass does
        # the active pass on one path (path_nocaustics "split") when the plugin can; "active_split": false, two paths.
        # Checked against two paths at 1024 spp (20263004, sim vanilla_r2): means within 0.1% (RGB and NIR), the
        # same outliers, GPU per 16 spp frame 143 -> 101 and 111 -> 77 ms
        split = bool(view.get("active_split", True)) and bool(getattr(self, "has_split", False))
        # pseudo near-infrared reflectance, 0.5..1 by robomituba's formula, mapped onto [low, high] (rgb2nir)
        nir_range = tuple(float(v) for v in (view.get("nir_range") or (0.5, 1.0)))
        if len(nir_range) != 2 or not 0.0 <= nir_range[0] <= nir_range[1] <= 1.0:
            raise ValueError("nir_range is [low, high] with 0 <= low <= high <= 1")
        if not 0 <= max_depth <= 64:
            raise ValueError("max_depth must be 0 (the scene's own) .. 64")
        flash_pdf = float(view.get("flash_caustic_pdf", FLASH_PDF))  # the flash pass's near-specular cut (FLASH_PDF)
        if not 0.1 <= flash_pdf <= 1e4:
            raise ValueError("flash_caustic_pdf must be 0.1..10000")
        mirrors = view.get("mirrors")  # None: the variant's own; else a share of the scene's perturbation pool
        if mirrors is not None:
            mirrors = (float(dict(mirrors).get("density", 1.0)), int(dict(mirrors).get("glass", 0)),
                       bool(dict(mirrors).get("glass_blocking", False)))
            if not (0.0 <= mirrors[0] <= 100.0 and 0 <= mirrors[1] <= 64):
                raise ValueError("mirrors.density must be 0..100 and mirrors.glass 0..64")
        renderer = str(view.get("renderer", "path"))  # "raster": the rasterized preview (raster.py); "cycles": its .blend
        # "depth": no image, the depth alone (one ray a pixel, nothing sampled or recorded: hw.sensing's depth sensors)
        # "solid": a structural preview of the raster scene (raster.RasterScene.render_solid: material colours, a studio light)
        # "lidar": what a LiDAR beam meets through each pixel (Resident.lidar_image: path length through glass and off
        # mirrors, near-infrared return, incidence), for hw.sensing's LiDAR noise models
        if renderer not in ("path", "raster", "solid", "cycles", "depth", "lidar"):
            raise ValueError("renderer must be 'path', 'raster', 'cycles' or 'depth'")
        # the Solid preview's filter (solid_filters.NAMES): one property in colour, the rest grey
        solid_filter = solid_filters.check(view.get("solid_filter")) if renderer == "solid" else None
        if renderer in ("raster", "solid"):   # drawn from the RGBN polar variant whatever the mode: its Stokes images, not "radiance"
            mode = "polar"         # (an "rgb" raster request came back empty: the rgb reply keeps "radiance" alone)
        if renderer == "cycles" and self.blend_of(scan) is None:
            raise ValueError(f"{scan} has no .blend for the Cycles backend (tools/attach_blend.py)")
        if renderer == "cycles" and bands == "nir":
            raise ValueError("the Cycles backend renders RGB only")
        raster_aa = str(view.get("raster_aa", "edge"))  # the preview's antialiasing (raster.AA_MODES; edge: nearly free)
        if raster_aa not in ("off", "edge", "ssaa2"):
            raise ValueError("raster_aa must be 'off', 'edge' or 'ssaa2'")
        # bodies: [{"id", "model" (POST /v1/bodies), "links": {link: object-to-world 4x4}}]. Which bodies a view
        # holds picks its resident scene (they are loaded with it); their poses are scene data, written per frame.
        placed = []
        for b in view.get("bodies") or ():
            model = self.body_models.get(str(b.get("model")))
            if model is None:
                raise ValueError(f"unknown body model {b.get('model')!r} (register it: POST /v1/bodies)")
            if set(b.get("links") or {}) != set(model["links"]):
                raise ValueError(f"body {b.get('id')!r} needs a pose for each link of {model['name']}: {sorted(model['links'])}")
            poses = [(body_models.instance_id(b["id"], link), np.ascontiguousarray(m, dtype=np.float64).reshape(4, 4).tobytes())
                     for link, m in b["links"].items()]
            for link, m in b["links"].items():   # the lights on a link move with it (rectangles of their own)
                poses += [(body_models.light_id(b["id"], link, i), np.ascontiguousarray(at).tobytes())
                          for i, at in body_models.light_poses(model, link, m)]
            poses = tuple(sorted(poses))
            placed.append((str(b["id"]), f"{model['name']}", model["key"], poses))
        placed = tuple(sorted(placed))
        if len({p[0] for p in placed}) != len(placed):
            raise ValueError("body ids repeat in one view")
        if placed and renderer not in ("path", "depth", "lidar", "raster", "solid"):
            raise ValueError("bodies are drawn by the path tracer and the raster previews (not by Cycles)")
        # objects: {physics object id: 4x4}: a movable object pushed by the physics step (hw/physics.py), as the change of
        # its pose since the scene file (simulator frame: x = X, y = -Z, z = Y); its dyn_ meshes are moved by it
        moved = []
        for oid, m in sorted(dict(view.get("objects") or {}).items()):
            m = np.ascontiguousarray(m, dtype=np.float64).reshape(4, 4)
            if not np.all(np.isfinite(m)):
                raise ValueError(f"object {oid!r}: a pose that is not finite")
            moved.append((str(oid), m.tobytes()))
        moved = tuple(moved)
        if moved and renderer not in ("path", "depth", "lidar", "raster", "solid"):
            raise ValueError("moved objects are drawn by the path tracer and the raster previews (not by Cycles)")
        look = (lights, env_map, env_scale, env_rot % 360.0, bands, flash_band, tuple(sorted(nir.items())), lighting,
                mirrors, lamps, flash_scale, flash_pdf, flash_offset, max_depth, caustics, split, nir_range, rr_depth,
                renderer, raster_aa, placed, moved, solid_filter)
        return (scan, variant, mode, width, height, hfov, spp, seed, denoise, look), c2w

    def _start(self, key: tuple, cams: list, submitted: float, depth: bool = False, hfovs: list | None = None):
        """Main-thread half one: queue the views on the GPU. Returns the second half (see submit_pipelined).
        ``hfovs``: each camera's field of view (several cameras in one batch call); default the key's."""
        started = time.perf_counter()
        scan, variant, mode, width, height, hfov, spp, seed, denoise, (lights, env, env_scale, env_rot, bands,
                                                                        flash_band, nir, lighting, mirrors, lamps,
                                                                        flash_scale, flash_pdf, flash_offset, max_depth,
                                                                        caustics, split, nir_range, rr_depth,
                                                                        renderer, raster_aa, placed, moved,
                                                                        solid_filter) = key
        if renderer == "cycles":  # a Blender process, not a Mitsuba scene: nothing of the path tracer is loaded for it
            return self._cycles_start(key, cams, started - submitted)
        if self.mi.variant() != self.variant_of[mode]:
            self.mi.set_variant(self.variant_of[mode])
        unpolarized = mode == "rgb" and self.shared_rgb   # the polarized scene, its integrators without Mueller matrices
        with ACTIVITY.doing("rendering"):
            rkey, passes, follow, hidden = self.plan(scan, variant, self.scene_mode(mode))
            parked = False
            if placed:  # a scene with these bodies in it is another resident scene (their meshes load with it)
                rkey = rkey + (tuple((b, name, mkey) for b, name, mkey, _ in placed),)
            else:
                # no bodies asked for (a robot taken away, the raster preview): the scene already loaded with bodies
                # serves, its bodies parked below the scene, instead of loading the place again without them
                with_bodies = next((k for k in reversed(self.cache) if k[:4] == rkey and len(k) > 4), None)
                if with_bodies is not None:
                    rkey, parked = with_bodies, True
            resident = self._resident(rkey, variant)
            if placed:
                resident.set_bodies({sid: np.frombuffer(m, dtype=np.float64).reshape(4, 4)
                                     for _, _, _, poses in placed for sid, m in poses})
            elif parked:
                resident.park_bodies()
            resident.set_objects({oid: np.frombuffer(m, dtype=np.float64).reshape(4, 4) for oid, m in moved},
                                 self.object_shapes(scan))
            # lighting mode (and the lamps added to dark lamps, when asked for): the light groups it leaves out
            # are removed from the scene (hidden), not just dimmed
            groups = LIGHTING[lighting] + (("lamp",) if lamps else ())
            # a lamp's own glowing part (lamp_lights_text) is the lamp: it goes dark (its group's uniforms), not away
            parts = set(resident.lamp_lights)
            hidden = frozenset(hidden) | {s for s, g in resident.groups.items() if g not in groups and g != "lamp"
                                          and s not in parts and (s in resident.blocks or s in resident.transforms)}
            if mirrors is None and self.meta[scan]["variants"][variant].get("pool_share"):
                mirrors = tuple(self.meta[scan]["variants"][variant]["pool_share"])  # a synthetic variant's mirrors
            hidden = hidden | self._pool_hidden(resident, mirrors)
            # one look per band: visible (0) and pseudo near-infrared (1), the same recordings for both
            # the flash emits only for the active variant (elsewhere it is parked, dark)
            flash = {"flash_rgb": follow and flash_band in ("rgb", "both"),
                     "flash_nir": follow and flash_band in ("nir", "both")}
            if resident.band_keys and (bands == "both" or resident.rgbn):
                # one render, both bands: the RGBN variant's lanes (whichever bands are asked: the same render
                # serves all three), else the dual sensor
                looks = {2: Look(hidden, lights, env, env_scale, env_rot, band=2, nir=nir, **flash, groups=groups, flash_scale=flash_scale,
                               flash_pdf=flash_pdf, flash_offset=flash_offset, max_depth=max_depth, caustics=caustics,
                               split=split, nir_range=nir_range, rr_depth=rr_depth, unpolarized=unpolarized)}
            else:  # one band per render
                looks = {b: Look(hidden, lights, env, env_scale, env_rot, band=b, nir=nir, **flash, groups=groups, flash_scale=flash_scale,
                               flash_pdf=flash_pdf, flash_offset=flash_offset, max_depth=max_depth, caustics=caustics,
                               split=split, nir_range=nir_range, rr_depth=rr_depth, unpolarized=unpolarized)
                         for b in {"rgb": (0,), "nir": (1,), "both": (0, 1)}[bands]}
            if renderer in ("raster", "solid"):  # the rasterized preview (raster.py): drawn here, one image a view
                return self._raster_start(key, resident, cams, next(iter(looks.values())), follow, started - submitted,
                                          bands, depth, raster_aa, solid=renderer == "solid", solid_filter=solid_filter)
            if renderer == "depth":
                return self._depth_start(resident, cams, width, height, hfov, next(iter(looks.values())), started - submitted)
            if renderer == "lidar":
                return self._lidar_start(resident, cams, width, height, hfov, next(iter(looks.values())), started - submitted)
            if self.kernel_history:
                self.dr.kernel_history()  # discard kernels launched before this call
            t0 = time.perf_counter()
            if len(cams) == 1:  # the pipelined path
                # a two-pass frame (active = passive + flash, a linear Stokes sum) is one "active" render call
                run = ("active",) if passes == ("passive", "flash") else passes
                handles = {b: [resident.launch(cams[0], width, height, hfov, spp, seed, p, follow, look)
                               for p in run] for b, look in looks.items()}
                if len(run) > 1:  # added on the GPU, read back once
                    handles = {b: [self._sum_handles(hs)] for b, hs in handles.items()}
                if unpolarized and resident.rgbn:  # S0 is all an unpolarized frame reads: 4 of the film's 19 channels
                    handles = {b: [(resident.intensity_lanes(img), bd) for img, bd in hs] for b, hs in handles.items()}
                if self.readback is not None:  # each image copied out behind its own kernels (readback.py)
                    handles = {b: [(img if isinstance(img, np.ndarray) else self.readback.start(img), bd)
                                   for img, bd in hs] for b, hs in handles.items()}
            else:  # batch sensor: completes here
                batch = {b: [resident.render_many(cams, width, height, hfovs or hfov, spp, seed, p, follow, look)
                             for p in passes] for b, look in looks.items()}
            # each view's depth (Resident.depth_image), queued behind its passes: the scene as they rendered it
            depths = [resident.depth_image(c, width, height, (hfovs or [hfov] * len(cams))[n])
                      for n, c in enumerate(cams)] if depth else None
            if depths is not None:
                self.dr.eval(*depths)
                if self.readback is not None and len(cams) == 1:
                    depths = [self.readback.start(z, (int(height), int(width))) for z in depths]
            launch_s = time.perf_counter() - t0

        def finish():
            with ACTIVITY.doing("rendering"):
                t1 = time.perf_counter()
                parts: dict[str, float] = {"queue_s": started - submitted, "launch_s": launch_s}
                if len(cams) == 1:
                    done = {b: [resident.finish(h) for h in hs] for b, hs in handles.items()}
                    per_band = {b: [[img] for img, _ in d] for b, d in done.items()}
                    breakdowns = [bd for d in done.values() for _, bd in d]
                else:
                    per_band, breakdowns = batch, [getattr(resident, "breakdown", None) or {}]
                for b in breakdowns:
                    for k, v in b.items():
                        parts[k] = parts.get(k, 0.0) + v
                depth_maps = None if depths is None else [
                    z.result() if hasattr(z, "result") else np.array(z, dtype=np.float32).reshape(int(height), int(width))
                    for z in depths]
                render_s = launch_s + time.perf_counter() - t1

            def assemble():
                bands_out = per_band
                if resident.rgbn:  # R, G, B and near-infrared lanes: the RGB layout, per band asked for
                    bands_out = {}
                    for b, views in per_band.items():
                        for want in ((0, 1) if b == 2 else (b,)):
                            if want == 0 and bands == "nir" or want == 1 and bands == "rgb":
                                continue
                            split = stokes.rgbn_intensity if mode == "rgb" else stokes.rgbn_band   # (rgb: S0 is all it reads)
                            bands_out[want] = [[split(v, want) for v in vs] for vs in views]
                elif 2 in per_band:  # dual: visible slot left, near-infrared right, margins cropped
                    wm, m = int(width) + 2 * DUAL_MARGIN, DUAL_MARGIN
                    bands_out = dict(per_band)
                    dual = bands_out.pop(2)
                    bands_out[0] = [[v[:, m:m + int(width)] for v in views] for views in dual]
                    bands_out[1] = [[v[:, wm + m:wm + m + int(width)] for v in views] for views in dual]
                outs, *rest = self._assemble(key, cams, bands_out, parts, render_s)
                for out, z in zip(outs, depth_maps or []):
                    out["depth"] = z
                return (outs, *rest)

            # The render thread only reads back (above). Splitting the film into bands and Stokes images, adding
            # the passes up and converting them (5-6 ms a pass at 512x384) run on the request's thread
            # (render_views), so the render thread starts the next pass meanwhile. The denoiser (OptiX) and the
            # kernel history stay on the render thread.
            if denoise or self.kernel_history:
                return assemble()
            return assemble
        return finish

    def _cycles_start(self, key: tuple, cams: list, queue_s: float):
        """_start for the Cycles backend: the Blender process is brought up here (main thread: in "exclusive" it frees the
        Mitsuba scenes first); the frames are rendered by the second half, on the request's thread, so the Mitsuba
        renders go on meanwhile. RGB only: S0 is the radiance, S1 to S3 are zero (Cycles is not polarized)."""
        scan, mode, width, height, hfov, spp, seed, denoise = key[0], key[2], key[3], key[4], key[5], key[6], key[7], key[8]
        worker = self.cycles_worker(scan)

        def assemble():
            from . import cycles_backend
            t0 = time.perf_counter()
            outs, gpu_s = [], 0.0
            for i, c2w in enumerate(cams):
                try:
                    img, rep = worker.render(c2w, width, height, hfov, spp, seed + i, denoise)
                except cycles_backend.CyclesError as exc:
                    if not cycles_backend.is_oom(exc):
                        raise
                    self.free_mitsuba()  # the GPU is full: the Mitsuba scenes' memory goes back, once more
                    img, rep = worker.render(c2w, width, height, hfov, spp, seed + i, denoise)
                gpu_s += rep.get("render_s", 0.0)
                if mode == "rgb":
                    outs.append({"radiance": img})
                else:
                    zero = np.zeros_like(img)
                    outs.append({"rgb_src": img, "s0": img, "s1": zero, "s2": zero, "s3": zero})
            render_s = time.perf_counter() - t0
            kernels = {"kernels": 0, "codegen_s": 0.0, "compile_s": 0.0, "gpu_s": gpu_s, "queue_s": queue_s,
                       "launch_s": 0.0, "split_s": 0.0}
            return outs, kernels, render_s, 0.0
        return lambda: assemble

    def _depth_start(self, resident, cams: list, width: int, height: int, hfov: float, look, queue_s: float):
        """_start for a view that asks for depth alone: the look written into the scene as a launch would write it, one
        ray a pixel (Resident.depth_image)."""
        t0 = time.perf_counter()
        with ACTIVITY.doing("rendering"):
            resident._set_camera(cams[0], width, height, hfov, False, look)
            depths = [resident.depth_image(c, width, height, hfov) for c in cams]
            self.dr.eval(*depths)
            maps = [np.array(z, dtype=np.float32).reshape(int(height), int(width)) for z in depths]
        render_s = time.perf_counter() - t0
        kernels = {"kernels": 0, "codegen_s": 0.0, "compile_s": 0.0, "gpu_s": 0.0, "queue_s": queue_s, "launch_s": 0.0, "split_s": 0.0}
        return lambda: lambda: ([{"depth": z} for z in maps], kernels, render_s, 0.0)

    def _lidar_start(self, resident, cams: list, width: int, height: int, hfov: float, look, queue_s: float):
        """_start for a LiDAR view (Resident.lidar_image): H x W x 6 float32 (LIDAR_CHANNELS) per camera."""
        t0 = time.perf_counter()
        with ACTIVITY.doing("rendering"):
            resident._set_camera(cams[0], width, height, hfov, False, look)
            chans = [resident.lidar_image(c, width, height, hfov) for c in cams]
            self.dr.eval(*[a for ch in chans for a in ch])
            maps = [np.stack([np.array(a, dtype=np.float32).reshape(int(height), int(width)) for a in ch], axis=-1)
                    for ch in chans]
        render_s = time.perf_counter() - t0
        kernels = {"kernels": 0, "codegen_s": 0.0, "compile_s": 0.0, "gpu_s": 0.0, "queue_s": queue_s, "launch_s": 0.0, "split_s": 0.0}
        return lambda: lambda: ([{"lidar": z} for z in maps], kernels, render_s, 0.0)

    def _raster_start(self, key: tuple, resident, cams: list, look, follow: bool, queue_s: float, bands: str,
                      depth: bool, aa: str = "edge", solid: bool = False, solid_filter: str | None = None):
        """_start for the rasterized preview: the views drawn and read back here, handed on in the path tracer's
        bands (RGB Stokes images, near-infrared ones) without its film split."""
        if not resident.rgbn:
            raise ValueError("the raster preview needs the RGBN polar variant (cuda_rgbn_polarized)")
        t0 = time.perf_counter()
        width, height, hfov = key[3], key[4], key[5]
        with ACTIVITY.doing("rasterizing"):
            views, depths, parts = resident.raster_views(cams, width, height, hfov, follow, look, aa, solid, solid_filter)
        if solid:
            self.last_solid = resident._raster   # what GET /v1/pick reads
        render_s = time.perf_counter() - t0
        parts.update(queue_s=queue_s, launch_s=0.0)

        def finish():
            def assemble():
                t1 = time.perf_counter()
                outs = []
                for lanes, z in zip(views, depths):
                    zero = np.zeros_like(lanes[0])
                    # S0..S3: R, G, B, near-infrared; the Solid preview has S0 alone (no polarization to send)
                    stokes_ = [v for v in lanes[:1]] if solid else [zero if v is None else v for v in lanes]
                    out = {}
                    if bands != "nir":
                        out.update({"rgb_src": stokes_[0][..., :3], **{f"s{k}": v[..., :3] for k, v in enumerate(stokes_)}})
                    if bands != "rgb":
                        out.update({f"nir_s{k}": v[..., 3] for k, v in enumerate(stokes_)})
                    if depth:
                        out["depth"] = z
                    outs.append(out)
                kernels = {"kernels": 0, "codegen_s": 0.0, "compile_s": 0.0, "gpu_s": 0.0, **parts,
                           "split_s": time.perf_counter() - t1}
                return outs, kernels, render_s, 0.0
            return assemble
        return finish

    @staticmethod
    def _sum_handles(handles: list) -> tuple:
        """One handle for several passes of a view: their images added (lazily, on the GPU when frozen) and
        their timings summed. The film's colour channels are S0, so the sum's are the summed S0 too."""
        img, breakdown = handles[0][0], dict(handles[0][1])
        for other, parts in handles[1:]:
            img = img + other
            for k, v in parts.items():
                breakdown[k] = breakdown.get(k, 0.0) + v
        return img, breakdown

    def _assemble(self, key: tuple, cams: list, per_band: dict, parts: dict, render_s: float) -> tuple:
        scan, variant, mode, width, height, hfov, spp, seed, denoise = key[:9]
        t1 = time.perf_counter()
        kernels = {"kernels": 0, "codegen_s": 0.0, "compile_s": 0.0, "gpu_s": 0.0}
        if self.kernel_history:
            history = self.dr.kernel_history()
            kernels = {  # milliseconds in Dr.Jit's history -> seconds
                "kernels": len(history),
                "codegen_s": sum(k.get("codegen_time", 0.0) for k in history) / 1000.0,
                "compile_s": sum(k.get("backend_time", 0.0) for k in history) / 1000.0,
                "gpu_s": sum(k.get("execution_time", 0.0) for k in history) / 1000.0,
            }
        outs = []
        for i in range(len(cams)):
            out = {}
            if 1 in per_band:  # near-infrared: the luminance of the band's (grey-material) render
                passes = [views[i] for views in per_band[1]]
                if mode == "rgb":
                    rad = passes[0][:, :, :3] if len(passes) == 1 else sum(p[:, :, :3] for p in passes)
                    out["nir"] = (self._denoise(mode, rad) if denoise else rad) @ LUMA
                else:
                    split = [stokes.split_channels(p)[1:] for p in passes]
                    n0, n1, n2, n3 = (sum(s[k] for s in split) for k in range(4))
                    if denoise:
                        n0, n1, n2 = self._denoise_stokes(mode, n0, n1, n2)
                    out.update({f"nir_s{k}": v @ LUMA for k, v in enumerate((n0, n1, n2, n3))})
            if 0 not in per_band:
                outs.append(out)
                continue
            passes = [views[i] for views in per_band[0]]
            if mode == "rgb":
                # one pass: its R, G, B as they are (the reply's dtype conversion makes the one copy); several: summed
                out["radiance"] = passes[0][:, :, :3] if len(passes) == 1 else sum(p[:, :, :3] for p in passes)
                if denoise:
                    out["radiance"] = self._denoise(mode, out["radiance"])
            else:
                rgb, s0, s1, s2, s3 = stokes.split_channels(passes[0])
                for extra in passes[1:]:  # active = passive (path) + flash (path_nocaustics), linear Stokes sum
                    _, f0, f1, f2, f3 = stokes.split_channels(extra)
                    s0, s1, s2, s3 = s0 + f0, s1 + f1, s2 + f2, s3 + f3
                    rgb = s0
                if denoise:
                    s0, s1, s2 = self._denoise_stokes(mode, s0, s1, s2)
                    rgb = s0
                out.update({"rgb_src": rgb, "s0": s0, "s1": s1, "s2": s2, "s3": s3})
            outs.append(out)
        kernels.update(parts)
        kernels["split_s"] = time.perf_counter() - t1
        return outs, kernels, render_s, time.perf_counter() - t1

    @staticmethod
    def _finish(mode: str, out: dict, dtype, preview: bool = True, s3: bool = True) -> dict[str, np.ndarray]:
        """``preview=False`` skips the dataset preview recipe (about 40 ms at 512x384); the client then
        tonemaps the radiance itself (MatterSim.setPreviewEnabled). ``s3=False`` leaves S3 (circular
        polarization, which a viewer does not show) out of the reply: a quarter less to encode and send."""
        stokes_keys = ("s0", "s1", "s2", "s3") if s3 else ("s0", "s1", "s2")
        result = {k: v.astype(dtype) for k, v in out.items()
                  if k == "nir" or (k.startswith("nir_s") and (s3 or k != "nir_s3"))}
        if "depth" in out:  # metres, float32 whatever the Stokes dtype (float16 steps 8 mm at 10 m)
            result["depth"] = out["depth"]
        if "lidar" in out:  # H x W x 6 (Resident.LIDAR_CHANNELS), float32
            result["lidar"] = out["lidar"]
        if mode == "rgb":
            if "radiance" in out:
                rad = out["radiance"]
                result["radiance"] = rad.astype(dtype)
                if preview:
                    result["rgb"] = stokes.rgb_preview(rad, rad, np.zeros_like(rad), np.zeros_like(rad),
                                                       percentile=0.995, blur=0.0)
            return result
        if "s0" in out:
            result.update({k: out[k].astype(dtype) for k in stokes_keys if k in out})   # (S0 alone: the Solid preview)
            if preview:
                result["rgb"] = stokes.rgb_preview(out["rgb_src"], out["s0"], out["s1"], out["s2"])
        return result

    def render_views(self, views: list[dict], dtype) -> tuple[list[dict[str, np.ndarray]], dict[str, float]]:
        """Consecutive views with the same scan, variant, mode, size, spp, seed and denoiser share one
        render call (see Resident.render_many). Timing sums over calls."""
        checked = [self._check(v) for v in views]
        results, total = [], {}

        def batchable(key):   # the key without the field of view: path-traced views of different lenses share a call
            return key[:5] + (None,) + key[6:] if key[9][18] == "path" else key
        i = 0
        while i < len(checked):
            key = checked[i][0]
            j = i + 1
            while j < len(checked) and batchable(checked[j][0]) == batchable(key):
                j += 1
            hfovs = [k[5] for k, _ in checked[i:j]]
            done = self.executor.submit_pipelined(self._start, key, [c for _, c in checked[i:j]],
                                                  time.perf_counter(),
                                                  any(bool(v.get("depth")) for v in views[i:j]), hfovs).result()
            outs, kernels, render_s, denoise_s = done() if callable(done) else done  # (see _start: assemble)
            t0 = time.perf_counter()
            results.extend(self._finish(key[2], out, dtype, bool(views[i + n].get("preview", True)),
                                        bool(views[i + n].get("s3", True)))
                           for n, out in enumerate(outs))
            timing = {"render_s": render_s, "denoise_s": denoise_s if key[8] else 0.0,
                      "post_s": time.perf_counter() - t0, **kernels, "calls": 1,
                      # what is left of the render call once kernels are accounted for: tracing the scene
                      # into JIT IR, plus readback. dr.freeze removes the tracing, codegen and compile parts.
                      "trace_s": (max(0.0, render_s - kernels["codegen_s"] - kernels["compile_s"] - kernels["gpu_s"])
                                  if self.kernel_history else 0.0)}
            for k, v in timing.items():
                total[k] = total.get(k, 0.0) + v
            i = j
        total["views"] = len(views)
        return results, total

    def render(self, view: dict, dtype) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        results, timing = self.render_views([view], dtype)
        return results[0], timing


def make_handler(renderer: Renderer, token: str | None = None):

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _authorized(self) -> bool:
            if not token:
                return True
            sent = self.headers.get("Authorization", "")
            if hmac.compare_digest(sent.encode(), f"Bearer {token}".encode()):
                return True
            self._json(401, {"error": "missing or wrong bearer token"})
            return False

        def log_message(self, fmt, *args):  # quieter than the default per-request stderr line
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json", headers: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _send_parts(self, code: int, parts: list, ctype: str, headers: dict | None = None):
            """A reply written piece by piece (each array's own buffer), never joined into one bytes object."""
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(sum(len(p) for p in parts)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            for part in parts:
                self.wfile.write(part)

        def _json(self, code: int, payload) -> None:
            self._send(code, json.dumps(payload).encode())

        def do_GET(self):
            if not self._authorized():
                return
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            try:
                if parts == ["v1", "status"]:  # cheap; answers while a render, load or recording runs
                    served = renderer.served()
                    from . import blend_scenes
                    return self._json(200, {
                        "activity": ACTIVITY.snapshot(), "freeze": renderer.freeze, "freeze_spp": renderer.freeze_spp,
                        "converting": {k: dict(x) for k, x in list(blend_scenes.PROGRESS.items())},  # .blend conversions: fraction, stage
                        # every variant a resident scene renders (an active variant and its flash-off twin share one)
                        "resident": [list(k) for k, _, _ in served],
                        "preloading": list(getattr(renderer, "preloading", [])),  # --preload scenes still loading
                        # freeze recordings per variant: [spp, width, height] recorded for all its passes
                        "recorded": {"|".join(k): renderer.recorded(r, passes) for k, r, passes in served},
                        # freeze recordings made so far by the resident scenes (one per new spp, size and pass);
                        # a change that should replay leaves it unchanged
                        "recordings": sum(getattr(f, "n_recordings", 0) for r in list(renderer.cache.values())
                                          for f in list(r.frozen.values()))})
                if parts == ["v1", "pick"]:  # ?x=&y=: what the last Solid frame shows at that pixel (raster.RasterScene.pick)
                    q = dict(urllib.parse.parse_qsl(self.path.partition("?")[2]))
                    scene = getattr(renderer, "last_solid", None)
                    if scene is None:
                        return self._json(404, {"error": "no Solid frame yet"})
                    return self._json(200, scene.pick(int(q["x"]), int(q["y"])))
                if parts == ["v1", "info"]:
                    return self._json(200, {
                        "server": "opticalnav_sim", "mitsuba_variant": renderer.variant,
                        "mitsuba_version": getattr(getattr(renderer, "mi", None), "__version__", None),
                        "modes": renderer.variant_of, "path_nocaustics": renderer.has_nocaustics,
                        "mode_bands": {m: mode_bands(v, bool(renderer.has_rgb2nir)) for m, v in renderer.variant_of.items()},
                        "shared_rgb": getattr(renderer, "shared_rgb", False),   # the rgb mode renders the polarized scene unpolarized
                        "envmaps": sorted(renderer.envmaps), "nir": renderer.has_rgb2nir,
                        "active_split": getattr(renderer, "has_split", False),
                        "nir_defaults": NIR_DEFAULTS,
                        "drjit_freeze": renderer.freeze, "spp_chunk": SPP_CHUNK,
                        # scenes the Cycles backend can render (their .blend travels in the pack), what Blender holds
                        "cycles": {"scans": sorted(s for s in renderer.meta if getattr(renderer, "blend_of", lambda _: None)(s)),
                                   "gpu": getattr(renderer, "cycles_gpu", None), "device": getattr(renderer, "cycles_device", None),
                                   "loaded": (getattr(renderer, "cycles", None) or [None])[0]},
                        "scans": {s: renderer.variants(s) for s in renderer.meta},
                        "resident": [list(k) + [round(r.load_s, 1)] for k, r, _ in renderer.served()],
                        "preloading": list(getattr(renderer, "preloading", [])),
                    })
                if parts == ["v1", "scans"]:
                    return self._json(200, sorted(renderer.meta))
                if parts == ["v1", "blends"]:
                    return self._json(200, renderer.blend_list() if hasattr(renderer, "blend_list") else [])
                if len(parts) == 4 and parts[:2] == ["v1", "scans"] and parts[2] in renderer.meta:
                    scan = parts[2]
                    if parts[3] == "meta":
                        meta = dict(renderer.meta[scan])
                        meta["variants_available"] = renderer.variants(scan)
                        if hasattr(renderer, "scene_dir"):  # where a viewer on this machine finds tone frames, room map
                            meta["scene_dir"] = str(renderer.scene_dir(scan).resolve())
                        pool = renderer.pool(scan)
                        if pool:  # what a view's "mirrors" can show, and the R2R edges each glass wall blocks
                            meta["perturbation_pool"] = {
                                "density_max": pool["density_max"], "mirrors": len(pool["mirrors"]),
                                "wall_source": pool["wall_source"],
                                "glass": [{k: g[k] for k in ("id", "rank", "splits_r2r", "blocked_edges")}
                                          for g in pool["glass"]]}
                        return self._json(200, meta)
                    if parts[3] == "stats":  # the resident scene that renders this scan, if loaded
                        served = [(k, r, passes) for k, r, passes in renderer.served() if k[0] == scan]
                        if not served:
                            return self._json(200, {"scan": scan, "resident": False, "gpu": gpu_memory()})
                        r = served[0][1]
                        groups: dict[str, int] = {}
                        for g in r.groups.values():
                            groups[g] = groups.get(g, 0) + 1
                        weights = list(r.emitter_weights.values())
                        return self._json(200, {
                            "scan": scan, "resident": True, "name": r.name, "load_s": round(r.load_s, 1), **r.stats,
                            "light_groups": groups, "lamps_lit_from_parts": len(r.lamp_lights),
                            "emitter_weights": [round(min(weights), 3), round(max(weights), 3)] if weights else None,
                            "pool": {"mirrors": len(r.pool["mirrors"]), "glass": len(r.pool["glass"])} if r.pool else None,
                            "variants": sorted(k[1] for k, _, _ in served),
                            "recordings": sum(getattr(f, "n_recordings", 0) for f in list(r.frozen.values())),
                            "mitsuba_variant": renderer.variant, "gpu": gpu_memory()})
                    if parts[3] == "connectivity":
                        pack = getattr(renderer, "pack_of", {}).get(scan, renderer.pack)
                        path = pack / "connectivity" / f"{scan}_connectivity.json"
                        if not path.is_file():  # packed without a navigation graph (scene.json "spawn")
                            return self._json(404, {"error": f"{scan} has no navigation graph"})
                        return self._send(200, path.read_bytes())
                self._json(404, {"error": f"no route {self.path}"})
            except Exception as exc:  # noqa: BLE001 - report, keep serving
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def _denoise_post(self):
            """NPZ in (s0, s1, s2 or radiance, float16/32 HxWx3; nir_s0..2 HxW), the same arrays denoised out
            (float16 when every input is, else float32)."""
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_DENOISE_BODY:
                    raise ValueError(f"request body must be 1..{MAX_DENOISE_BODY} bytes")
                npz = np.load(io.BytesIO(self.rfile.read(length)))
                started = time.perf_counter()
                arrays = {k: npz[k] for k in npz.files}
                out = renderer.denoise_images(arrays)
                dtype = np.float16 if all(v.dtype == np.float16 for v in arrays.values()) else np.float32
                buf = io.BytesIO()
                np.savez(buf, **{k: np.asarray(v, dtype=dtype) for k, v in out.items()})
                self._send(200, buf.getvalue(), "application/octet-stream",
                           {"X-Render-Seconds": f"{time.perf_counter() - started:.4f}"})
            except (KeyError, ValueError, TypeError, OSError) as exc:
                self._json(400, {"error": f"{type(exc).__name__}: {exc}"})
            except Exception as exc:  # noqa: BLE001 - report, keep serving
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self):
            if not self._authorized():
                return
            route = self.path.split("?")[0].rstrip("/")
            if route == "/v1/denoise":
                return self._denoise_post()
            if route == "/v1/blend":  # {"id": listed id | "path": file}: convert once, serve, answer the scene id
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
                    scan = renderer.open_blend(str(body.get("id") or body["path"]))
                    return self._json(200, {"scan": scan, "variants": renderer.variants(scan)})
                except (KeyError, ValueError) as exc:
                    return self._json(400, {"error": f"{type(exc).__name__}: {exc}"})
                except SystemExit as exc:
                    return self._json(500, {"error": str(exc)})
                except Exception as exc:  # noqa: BLE001
                    return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            if route == "/v1/bodies":  # {"name", "links": {...}} (bodies.py): a model views may then place
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
                    return self._json(200, renderer.register_body(body["name"], body))
                except (KeyError, ValueError, TypeError) as exc:
                    return self._json(400, {"error": f"{type(exc).__name__}: {exc}"})
            if route == "/v1/gpu/free":  # {"backend": "cycles" | "mitsuba" | "all"}: hand that renderer's GPU memory back
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
                    which = body.get("backend", "all")
                    if which not in ("cycles", "mitsuba", "all"):
                        raise ValueError("backend must be 'cycles', 'mitsuba' or 'all'")
                    freed = {"cycles": renderer.free_cycles() if which in ("cycles", "all") else False,
                             "mitsuba": renderer.free_mitsuba() if which in ("mitsuba", "all") else False}
                    return self._json(200, {"freed": freed, "gpu": gpu_memory()})
                except ValueError as exc:
                    return self._json(400, {"error": str(exc)})
                except Exception as exc:  # noqa: BLE001
                    return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            if route == "/v1/physics/refresh":   # {"scan"}: decisions set by hand changed (Renderer.refresh_physics)
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(length) or b"{}")
                    return self._json(200, renderer.refresh_physics(str(body["scan"])))
                except Exception as exc:  # noqa: BLE001
                    return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            if route != "/v1/render":
                return self._json(404, {"error": f"no route {self.path}"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise ValueError(f"request body must be 1..{MAX_BODY} bytes")
                req = json.loads(self.rfile.read(length))
                views = req["views"]
                if not 0 < len(views) <= MAX_VIEWS:
                    raise ValueError(f"1..{MAX_VIEWS} views per request")
                dtype = {"float16": np.float16, "float32": np.float32}[req.get("stokes_dtype", "float16")]
                started = time.monotonic()
                results, total = renderer.render_views(views, dtype)
                arrays = {f"{k}_{i}": v for i, result in enumerate(results) for k, v in result.items()}
                t_enc = time.perf_counter()
                if req.get("format") == "raw":  # header + raw bytes: no zip to build here or to unpack there
                    arrays = {k: np.ascontiguousarray(v) for k, v in arrays.items()}
                    header, offset = [], 0
                    for k, v in arrays.items():
                        header.append([k, v.dtype.str, list(v.shape), offset, v.nbytes])
                        offset += v.nbytes
                    head = json.dumps(header).encode()
                    total["encode_s"] = time.perf_counter() - t_enc
                    self._send_parts(200, [len(head).to_bytes(8, "little"), head] + [memoryview(v).cast("B")
                                                                                     for v in arrays.values()],
                                     RAW_ARRAYS_TYPE,
                                     {"X-Render-Seconds": f"{time.monotonic() - started:.4f}",
                                      "X-Render-Timing": json.dumps({k: round(v, 4) for k, v in total.items()})})
                    return
                buf = io.BytesIO()
                np.savez(buf, **arrays)
                total["encode_s"] = time.perf_counter() - t_enc
                self._send(200, buf.getvalue(), "application/octet-stream",
                           {"X-Render-Seconds": f"{time.monotonic() - started:.4f}",
                            "X-Render-Timing": json.dumps({k: round(v, 4) for k, v in total.items()})})
            except (KeyError, ValueError, TypeError) as exc:
                self._json(400, {"error": f"{type(exc).__name__}: {exc}"})
            except (BrokenPipeError, ConnectionResetError):  # the client left mid-reply (a viewer restarted)
                return
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def compact_on_start(renderer: Renderer, skip=()) -> None:
    """Compact every served scene that still needs it (tools/compact_assets.py), one subprocess per scene: its
    Mitsuba runs scalar_rgb, which must not switch this process's variant. Listed in /v1/status as the activity
    ``compacting <scan>`` meanwhile; renders wait in the queue."""
    import importlib.util
    import subprocess
    import sys

    tool = Path(__file__).resolve().parents[1] / "tools" / "compact_assets.py"
    if not tool.is_file():
        print("[compact] tools/compact_assets.py not found: scenes are served as they are", flush=True)
        return
    spec = importlib.util.spec_from_file_location("compact_assets", tool)
    compact = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compact)  # needs_compaction is a text scan; nothing of Mitsuba is imported
    todo = [s for s in sorted(renderer.meta) if s not in skip and not renderer.direct_scene(s)  # a converted .blend is served as it is
            and compact.needs_compaction(renderer.scene_dir(s))]
    print(f"[compact] {len(todo)} of {len(renderer.meta)} scenes need compacting: {todo}", flush=True)
    for scan in todo:
        started = time.monotonic()
        with ACTIVITY.doing(f"compacting {scan}"):
            done = subprocess.run([sys.executable, str(tool), "--pack", str(renderer.pack_of[scan]), "--scene", scan,
                                   "--if-needed"], check=False)
        if done.returncode == 0:
            renderer.reload_meta(scan)
            print(f"[compact] {scan} in {time.monotonic() - started:.0f} s", flush=True)
        else:
            print(f"[compact] {scan} failed (exit {done.returncode}): served as it is", flush=True)


def ser_lock(gpu: str, folder: str | None = None):
    """The right to run polarized kernels with shader execution reordering (SER) on a GPU: an open, locked file, or
    None when another server has it.

    With SER on (Dr.Jit's default on RTX 40 / 50), polarized OptiX kernels die of CUDA_ERROR_ILLEGAL_ADDRESS (700) when
    another process keeps the GPU busy, whatever that process runs, and not when they have it to themselves. Measured on
    driver 580.97 under WSL2 (2026-11-25), 1024 spp frames: next to another busy process 21 crashes in 159 frames; the
    same with SER off, none in 95; one server alone with SER on, none in 1005. Stock Mitsuba does it too (tools/ser_crash_repro.py), so it is not this server's doing.
    Another job on the GPU that is not one of ours cannot be seen here: --ser off is for that."""
    import fcntl
    import tempfile

    path = Path(folder or tempfile.gettempdir()) / f"opticalnav_ser_gpu{re.sub(r'[^0-9A-Za-z]+', '_', str(gpu))}.lock"
    f = open(path, "a")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", action="append", default=[],
                    help="scene pack directory (tools/build_pack.py output); repeat to serve several (a scene name "
                         "in two packs comes from the first)")
    ap.add_argument("--blend", action="append", default=[], metavar="[ID=]FILE",
                    help="serve a scene straight from its .blend (repeatable): converted once by tools/blend_to_pack.py into the blend "
                         "pack (seconds to minutes the first time, then cached until the file changes), the id blend_<file name>; "
                         "the Cycles renderer renders the same file")
    ap.add_argument("--blend-dir", action="append", default=[], metavar="DIR",
                    help="offer every *.blend directly in DIR in the viewer's scene list (converted when chosen, not at start)")
    ap.add_argument("--blend-library", action="append", default=[], metavar="DIR",
                    help="a scene library (<series>/<id>/<file>.blend with MANIFEST*.json, as /bean/infinigen_invers3r): its scenes are "
                         "listed too; a file is opened only as a copy (opticalnav_sim/blend_paths.py). Also $OPTICALNAV_BLEND_LIBRARY")
    ap.add_argument("--blend-tsv", action="append", default=[], metavar="FILE",
                    help="offer the .blend files of a promoted_ref_scenes tsv (columns id, passed_blend) the same way")
    ap.add_argument("--blend-pack", type=Path, default=None,
                    help="where converted .blend scenes are kept (default packs/blend-cache); scenes already in it are served too")
    ap.add_argument("--no-compact", action="store_true",
                    help="serve scenes as they are: by default a scene not yet compacted (tools/compact_assets.py: "
                         "OBJ meshes, textures not named by content) is compacted once at start, before it loads")
    ap.add_argument("--compact-skip", action="append", default=[], metavar="SCAN",
                    help="leave this scene out of the start-up compaction (another process is writing it)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18770)
    ap.add_argument("--variant", default=None,
                    help="polar-mode Mitsuba variant (default: cuda_rgb_polarized, else cuda_ad_rgb_polarized)")
    ap.add_argument("--rgb-variant", default=None, help="rgb-mode variant (default: cuda_rgb, else cuda_ad_rgb)")
    ap.add_argument("--modes", default="polar",
                    help="render modes this process serves: polar, rgb or polar,rgb. Two Mitsuba variants in one "
                         "process crashed on the OptiX 7 build, so run one server per mode there")
    ap.add_argument("--max-resident", type=int, default=1,
                    help="scenes kept loaded (1: the one shown; a frozen replay walks every loaded scene's shapes, so a "
                         "second one made every CS02 frame 74 ms of CPU instead of 6)")
    ap.add_argument("--preload", action="append", default=[], help="SCAN[:VARIANT[:MODE]] to load at start (repeatable)")
    ap.add_argument("--ser", choices=["on", "off"], default="on",
                    help="shader execution reordering. On: the fast way, for a GPU no other CUDA job is busy on. Off: for a "
                         "GPU shared with other work (a second server, training): with it on, polarized kernels die of "
                         "CUDA error 700 about once a minute there; off they do not, and take 2.8 times as long")
    ap.add_argument("--freeze", choices=["auto", "on", "off"], default="auto",
                    help="replay renders with dr.freeze (auto: when this Dr.Jit has it)")
    ap.add_argument("--freeze-spp", type=int, default=16,
                    help="record frozen renders at this spp only: a request for a multiple of it replays that "
                         "recording several times (new seeds, averaged) instead of recording again; 0 records each spp")
    ap.add_argument("--sun-deg", type=float, default=8.0,
                    help="spread an environment map's sun over a Gaussian of this many degrees, keeping its power: "
                         "sunlight through window glass otherwise renders as fireflies (0: as the file has it)")
    ap.add_argument("--default-sky", default="urban_street_01", metavar="MAP",
                    help="an --envmaps map (file stem) used as the sky of the simulator's own scenes (staged without "
                         "a render) when a view asks for the scene's own environment; scaled to the constant sky's "
                         "mean radiance. Default urban_street_01: a street behind the windows (kloofendal_48d_partly_cloudy_puresky "
                         "was a white sheet). The dataset's scenes keep theirs")
    ap.add_argument("--default-sky-rotate", type=float, default=0.0, help="turn that sky about the vertical (degrees)")
    ap.add_argument("--lamp-power", choices=("importer", "blender"), default="blender",
                    help="'blender': the lights of the simulator's own scenes that carry metadata.light.energy glow at "
                         "Blender's power (flux energy / pi) instead of the importer's box radiance, which is 3 to 10 "
                         "times weaker. The dataset's scenes keep theirs")
    ap.add_argument("--alpha-power", type=float, default=2.0,
                    help="raise the microfacet alpha of the simulator's own scenes (the importer wrote Blender's "
                         "roughness r as alpha) to this power; 2 is Blender's r**2 (a glossy floor reflects the "
                         "windows instead of smearing them). Roughness textures are copied once to "
                         "~/.cache/opticalnav_sim/alpha_power. The dataset's scenes keep theirs")
    ap.add_argument("--fill-gain", type=float, default=2.0,
                    help="scale the radiance of the room softboxes (light_softbox_*, the importer's fill) of the "
                         "simulator's own scenes (default 2: against Cycles at AgX exposure 1, floorG, lphigh and stillroom come "
                         "within 1.15-1.85x of its brightness; at 1.0 within 1.4-2.3x, at 0 the fill's share "
                         "of the light and the noise it keeps down are gone). 1: the importer's own. The dataset's "
                         "scenes keep theirs")
    ap.add_argument("--default-depth", type=int, default=10,
                    help="path length of the simulator's own scenes when a view asks for the scene's own (0: its "
                         "own, 6): a pane of glass is two bounces now, a window and a room three more")
    ap.add_argument("--cycles-gpu", choices=("share", "exclusive"), default="share",
                    help="the Cycles backend (a view's renderer 'cycles', the scene's .blend in Blender) and the Mitsuba scenes "
                         "on one GPU. 'share': both stay loaded, and one that runs out of GPU memory takes it from the other "
                         "(Blender is stopped, or the resident scenes dropped) and retries. 'exclusive': using one frees the "
                         "other at once. POST /v1/gpu/free frees either by hand")
    ap.add_argument("--cycles-device", choices=("OPTIX", "CUDA"), default="OPTIX", help="Cycles' compute device")
    ap.add_argument("--envmaps", type=Path, default=None,
                    help="folder of .exr/.hdr environment maps a view may name (default: <pack>/envmaps)")
    ap.add_argument("--kernel-history", action="store_true",
                    help="record per-kernel codegen/compile/GPU times (X-Render-Timing gpu_s etc.). Reading them "
                         "synchronises the GPU every view, so views no longer overlap; use it for measurements")
    ap.add_argument("--token", default=os.environ.get("OPTICALNAV_SIM_TOKEN"),
                    help="require 'Authorization: Bearer <token>' (default $OPTICALNAV_SIM_TOKEN)")
    args = ap.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.token:
        raise SystemExit("refusing to listen beyond localhost without --token (HTTP has no other access control)")
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    if not modes or any(m not in MODE_VARIANTS for m in modes):
        raise SystemExit(f"--modes must name polar and/or rgb, got {args.modes!r}")
    held = None   # kept open (and so locked) as long as the process lives
    if "polar" in modes and args.ser == "on":
        held = ser_lock(os.environ.get("CUDA_VISIBLE_DEVICES") or "0")
        if held is None:
            raise SystemExit("another render server with shader execution reordering on runs on this GPU: two of them kill each other "
                             "(CUDA error 700, about once a minute). Use that one, or start both with --ser off (2.8 times slower).")
    from . import blend_scenes
    packs = [Path(p).resolve() for p in args.pack]
    args.blend_dir = list(args.blend_dir) + [d for d in os.environ.get("OPTICALNAV_BLEND_DIRS", "").split(":") if d]
    args.blend_tsv = list(args.blend_tsv) + [f for f in os.environ.get("OPTICALNAV_BLEND_TSV", "").split(":") if f]
    blend_pack = Path(args.blend_pack or blend_scenes.DEFAULT_PACK).resolve()
    if args.blend:
        blend_scenes.prepare(args.blend, blend_pack)
    if (blend_pack / "scenes").is_dir() and blend_pack not in packs:
        packs.append(blend_pack)  # scenes converted earlier stay available
    if not packs and not (args.blend_dir or args.blend_tsv):
        raise SystemExit("nothing to serve: --blend FILE, --blend-dir DIR, --blend-tsv FILE or --pack DIR")
    if not packs:  # only a list of .blend files: the blend pack, empty until one is chosen
        (blend_pack / "scenes").mkdir(parents=True, exist_ok=True)
        packs.append(blend_pack)
    renderer = Renderer(packs, args.variant, args.max_resident, args.freeze, args.rgb_variant, modes,
                        kernel_history=args.kernel_history, envmap_dir=args.envmaps, sun_deg=args.sun_deg, default_sky=args.default_sky, default_sky_rotate=args.default_sky_rotate,
                        fill_gain=args.fill_gain, default_depth=args.default_depth, lamp_power=args.lamp_power, alpha_power=args.alpha_power,
                        freeze_spp=args.freeze_spp, cycles_gpu=args.cycles_gpu, cycles_device=args.cycles_device, ser=args.ser == "on")
    renderer.blend_pack = blend_pack
    libraries = list(args.blend_library) + [p for p in os.environ.get("OPTICALNAV_BLEND_LIBRARY", "").split(os.pathsep) if p]
    renderer.blend_files = blend_scenes.listed(args.blend_dir, args.blend_tsv, libraries)
    renderer.blend_moves = {}
    for root in libraries:
        from . import blend_paths
        renderer.blend_moves.update(blend_paths.moves(root))
    # listen first: while --preload loads, /v1/status reports the load's progress and renders wait in the queue
    server = ThreadingHTTPServer((args.host, args.port), make_handler(renderer, args.token))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[serve] http://{args.host}:{args.port}  modes={renderer.variant_of}  shared_rgb={renderer.shared_rgb}  scans={len(renderer.meta)}  "
          f"path_nocaustics={renderer.has_nocaustics}  freeze={renderer.freeze}  "
          f"kernel_history={renderer.kernel_history}  ser={renderer.ser}", flush=True)
    if not args.no_compact:
        compact_on_start(renderer, set(args.compact_skip))
    for item in args.preload:
        parts = item.split(":")
        renderer.preloading.append([parts[0], (parts[1:2] or ["base"])[0], (parts[2:3] or [modes[0]])[0]])
    while renderer.preloading:
        scan, variant, mode = renderer.preloading[0]
        print(f"[preload] {scan}:{variant}:{mode} in {renderer.preload(scan, variant, mode):.1f}s", flush=True)
        renderer.preloading.pop(0)
    try:
        renderer.executor.run_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
