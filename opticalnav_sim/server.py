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
import xml.etree.ElementTree as ET
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from . import frames, progress, stokes

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
            self.what, self.since = what, time.time()
        try:
            yield
        finally:
            with self.lock:
                self.what, self.since = prev

    def snapshot(self) -> dict:
        with self.lock:
            out = {"what": self.what, "elapsed_s": round(time.time() - self.since, 1) if self.since else 0.0}
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
                 "rgb": ("cuda_rgb", "cuda_ad_rgb")}
# Without dr.freeze, high spp is rendered in passes of at most this many samples (wavefront memory);
# a polarized path carries Mueller matrices, so its passes are smaller.
# What a frozen render takes as input. "scene": the whole scene, so Dr.Jit walks every texture and mesh
# on each replay (~23 ms on Device 1). "sensor": only the camera, with the static scene captured when the
# function is recorded; a replay then walks the sensor alone. Flash scenes always use "scene" (the flash
# moves with the camera).
FREEZE_INPUT = os.environ.get("OPTICALNAV_SIM_FREEZE_INPUT", "scene")
SPP_CHUNK = {"polar": max(1, int(os.environ.get("OPTICALNAV_SIM_SPP_CHUNK", "64"))),
             "rgb": max(1, int(os.environ.get("OPTICALNAV_SIM_SPP_CHUNK_RGB", "256")))}


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
ASSIST_SHAPE = re.compile(r'\s*<shape type="rectangle" id="camera_assist_[a-z]+">.*?</shape>', re.S)
SENSOR_BLOCK = re.compile(r"\s*<sensor\b.*?</sensor>", re.S)
COMMENT = re.compile(r"\s*<!--.*?-->", re.S)
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
    for o in lamps:
        spawn = _SPAWN.search(o["id"])
        light = keyed.get(spawn.groups()) if spawn else None
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


def lamp_lights_text(text: str, mesh_info=None, lit: bool = True, mesh_bounds=None) -> tuple[str, list[str]]:
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
    (or the callback) the pendant glows from its lowest non-metal part, the bulb (its highest is the cord)."""
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

    def light_power(light: dict) -> float | None:
        """Emitted power / pi of an importer's light box: radiance x its emitting area."""
        if not light.get("emitter_radiance"):
            return None
        sx, sy, sz = (float(v) for v in (light.get("geometry") or {}).get("size_m") or [0.6, 0.08, 0.6])
        rad = float(np.mean(light["emitter_radiance"])) * float(light.get("emitter_intensity") or 1.0)
        kind = shapes[light["id"]].group(1) if light["id"] in shapes else "cube"
        return rad * (sx * sz if kind in ("rectangle", "disk") else 2 * (sx * sz + sx * sy + sy * sz))

    def disk(sid: str, r: float, center, rad: float, note: dict) -> None:
        """A disk of radius r facing the floor, centred at ``center`` (x, y, z), glowing at ``rad``."""
        xml.append(f'\n  <shape type="ply" id="{sid}">\n    <string name="filename" value="{unit_disk_ply()}" />'
                   f'\n    <transform name="to_world">\n      <scale value="{r:.6f}" />'
                   f'\n      <rotate x="1" angle="90" />'  # +z (the disk's front) to -y: facing the floor
                   f'\n      <translate x="{center[0]:.6f}" y="{center[1]:.6f}" z="{center[2]:.6f}" />'
                   f'\n    </transform>\n    <bsdf type="diffuse" />'
                   f'\n    <emitter type="area">\n      <rgb name="radiance" value="{rad:.4f} {rad:.4f} {rad:.4f}" />'
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
                sid = f"{o['id']}__glow"
                disk(sid, r, ((lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2), rad,
                     {"id": sid, "type": "landmark", "is_emitter": True, "emitter_radiance": [round(rad, 4)] * 3,
                      "emitter_intensity": 1.0, "emitter_shape": "lamp_opening",
                      "label": f"light:{'fixture' if light is not None else 'lamp'}:{o['label']}",
                      "metadata": {"lamp": o["id"], "glowing_area_m2": round(math.pi * r * r, 4),
                                   "replaces": (light or {}).get("id"), "added_by": "opticalnav_sim lamp_lights_text"},
                      "geometry": o.get("geometry")})
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
        for shade in shades:
            sid = shade.group(2)
            edits.append((shade.end(3), f'\n    <emitter type="area">\n      <rgb name="radiance" value="{rad:.4f} {rad:.4f} {rad:.4f}" />'
                                        f'\n    </emitter>'))
            note = {"id": sid, "type": "landmark", "is_emitter": True, "emitter_radiance": [round(rad, 4)] * 3,
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
        disk(sid, r, center, rad,
             {"id": sid, "type": "landmark", "is_emitter": True, "emitter_radiance": [round(rad, 4)] * 3,
              "emitter_intensity": 1.0, "emitter_shape": "downlight",
              "label": f"light:fixture:CeilingDownlight:{light.get('label', light['id'])}",
              "metadata": {"glowing_area_m2": round(math.pi * r * r, 4), "replaces": light["id"],
                           "added_by": "opticalnav_sim lamp_lights_text (a light by no fixture)"},
              "geometry": g})
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
    return text, added


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
    if len(power) < 2:
        return {}
    mean = float(np.mean(list(power.values())))
    weights = {k: max(p, 0.05 * mean) / mean for k, p in power.items()}
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

    def __init__(self, mi, xml: Path, mode: str, flash: dict | None = None, freeze: bool = False,
                 flash_xml: Path | None = None, hideable=(), envs=None, freeze_spp: int = 0, nir: bool = False,
                 pool: dict | None = None, upgrade_flash: bool = False, hidden: str = ""):
        self.mi, self.mode = mi, mode
        # an RGBN variant: every render holds both bands (R, G, B lanes and the near-infrared lane)
        self.rgbn = "_rgbn" in mi.variant()
        started = time.time()
        text = lit_scene_text(xml, flash_xml) if flash_xml is not None else xml.read_text()
        self.hidden = hidden  # shape id prefix camera rays pass through (path_nocaustics "hidden"; FILL_PREFIX)
        self.upgraded_flash = False  # an old one-pass flash rendered as the current protocol (upgrade_flash_text)
        if flash is not None and flash.get("synthetic") and flash_xml is None:
            # a scene packed without an active variant: the dataset's flash added, rendered as the current protocol
            text, self.upgraded_flash = flash_scene_text(text, flash), True
        elif upgrade_flash and flash is not None and flash_xml is None:
            text, self.upgraded_flash = upgrade_flash_text(text, float(flash.get("polarizer_angle_deg") or 0.0))
        if mode == "rgb":
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
                                                  mesh_bounds=mesh_bounds)
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
        with progress.watch_load(mi, text, self.name):
            self.scene = mi.load_string(text)
        # OPTICALNAV_UNIFORM_EMITTERS=1: Mitsuba's default, every emitter picked as often (for comparisons)
        self.emitter_weights = {} if os.environ.get("OPTICALNAV_UNIFORM_EMITTERS") else \
            emitter_power_weights(mi, self.scene, radiance)
        self.load_s = time.time() - started
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
            t = time.time()
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
                  f"({sum(len(v) for v in self.originals.values())} vertices kept, {time.time() - t:.1f} s)", flush=True)
        # Indoor emitters: every emitter radiance but the camera flash's. Written back once (opaque) here.
        import drjit as dr

        params = mi.traverse(self.scene)
        # (type, loaded values) copied to the host: a reference to the parameter would alias the emitter's own
        # buffer, which the first write (lights off: 0) overwrites
        emitted = (".emitter.radiance.value", ".emitter.radiance.nested.value")  # plain, or behind bandscale
        self.radiance = {k: (type(params[k]), np.array(params[k], dtype=np.float64).ravel()) for k in keys
                         if k.endswith(emitted) and not k.startswith(ASSIST_PREFIX)}
        self.light_class = {k: self.classes.get(k.split(".emitter.radiance")[0], "led") for k in self.radiance}
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

    def integrator(self, name: str, sky: float | None = None):
        """path_nocaustics in an assist mode (or, for "own", as plain path: caustics kept, every emitter),
        with ``sky`` (default: daylight's near-infrared share) on the sky seen by a dual render's
        near-infrared slot. Without bandscale lights, "own" is the scene's own integrator (None)."""
        if name == "own" and not self.band_keys and not self.max_depth and not self.rr_depth and self.caustics \
                and not self.hidden:
            return None
        sky = self.sky_nir if sky is None else float(sky)
        # the flash's near-specular cut (Look.flash_pdf); split cuts the flash's paths with it too
        pdf = float(self.flash_pdf) if name in ("flash", "split") else None
        key = (name, sky, pdf, self.max_depth, self.rr_depth, self.caustics)
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
            if self.rr_depth:
                inner["rr_depth"] = int(self.rr_depth)
            if pdf is not None:
                inner["specular_pdf_threshold"] = pdf
            if self.mode != "rgb":
                inner = {"type": "stokes", "integrator": inner}
                if spec["reference_axis"] is not None:
                    inner["reference_axis"] = self.mi.ScalarVector3f(*spec["reference_axis"])
            self.integrators[key] = self.mi.load_dict(inner)
        return self.integrators[key]

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
                "film": {"type": "hdrfilm", "width": 2 * wm, "height": int(height)},
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

    def _set_camera(self, c2w: np.ndarray, width: int, height: int, hfov_deg: float, follow: bool,
                    look: Look = Look()) -> dict:
        """Write the camera, the assist shapes and the Look into the scene, unless already there."""
        mi = self.mi
        pose = (np.asarray(c2w, dtype=np.float64).tobytes(), int(width), int(height), float(hfov_deg), follow, look)
        if pose == self.pose:
            return {"traverse_s": 0.0, "update_s": 0.0}
        t0 = time.perf_counter()
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
        z = dr.select(pi.is_valid(), pi.t * self._depth_inv, 0.0)
        dr.schedule(z)
        return z

    def raster_views(self, cams: list, width: int, height: int, hfov_deg: float, follow: bool, look: Look,
                     aa: str = "off") -> tuple:
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
                parts["raster_build_s"] += self._raster.build_s
            self._raster.sync(look.hidden)
            t2 = time.perf_counter()
            lanes, z = self._raster.render(c2w, width, height, hfov_deg, look, follow, aa)
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
        if self.band_keys:  # levels and lighting modes are in the shared uniforms: the radiance stays as loaded
            return 1.0
        if key.split(".emitter.radiance")[0] in look.hidden:
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
                                                  int(seed) + 104729 * i), dtype=np.float64) * n
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
            for name in names:
                self._prepare_inputs(dual if dual is not None else self.sensor, self.pass_integrators(name, sky)[0])
                self._record(spp, int(width), int(height), name, dual, sky)
            self._absorb([self._key(spp, width, height, n, dual, sky) for n in names])
            t2 = time.perf_counter()
        # a replay whose inputs changed layout records again: report it like a recording (see progress.py)
        progress.FREEZE.begin(f"{self.name}|{self.mode}|{pass_name}|{spp}|{width}x{height}", calls=1, remember=False,
                              score=self.score)
        try:
            args = self.frozen_args[key]
            img = self.frozen[key](*args, dr.opaque(mi.UInt32, int(seed) * repeats))  # replays, launching the kernels
            for i in range(1, repeats):  # same distribution as one render at spp x repeats
                img = img + self.frozen[key](*args, dr.opaque(mi.UInt32, int(seed) * repeats + i))
        finally:
            progress.FREEZE.end(False)
        if repeats > 1:
            img = img / float(repeats)
        breakdown["replay_s"] = time.perf_counter() - t2
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
            if dual is not None:  # scene and dual sensor are inputs: slot poses change between frames
                frozen = dr.freeze(lambda scene, sn, sd, n=spp, its=its: self.render_passes(scene, sn, its, n, sd),
                                   backend=dr.JitBackend.CUDA, auto_opaque=False, warn_after=2)
                self.frozen_args[key] = (self.scene, dual)
                self._settle(frozen, self.scene, dual)
            elif self.sensor_input:
                scene = self.scene
                frozen = dr.freeze(lambda sn, sd, n=spp, its=its: self.render_passes(scene, sn, its, n, sd),
                                   backend=dr.JitBackend.CUDA, auto_opaque=False, warn_after=2)
                self.frozen_args[key] = (self.sensor,)
                self._settle(frozen, self.sensor)
            else:
                frozen = dr.freeze(lambda scene, sd, n=spp, its=its: self.render_passes(scene, None, its, n, sd),
                                   backend=dr.JitBackend.CUDA, auto_opaque=False, warn_after=2)
                self.frozen_args[key] = (self.scene,)
                self._settle(frozen, self.scene)
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


    def render_many(self, cams: list, width: int, height: int, hfov_deg: float, spp: int, seed: int,
                    pass_name: str = "own", follow: bool | None = None, look: Look = Look()) -> list:
        """K views in one render call through a resident ``batch`` sensor.

        Tracing the scene (materials, lights, geometry) dominates a call and does not depend on the
        cameras, so one call for K views pays it once. The camera-aligned flash of an active scene
        follows a single camera, so those views still render one at a time.
        """
        follow = (self.k_flash is not None) if follow is None else bool(follow and self.k_flash is not None)
        if len(cams) == 1 or follow or look.band == 2:
            return [self.render(c, width, height, hfov_deg, spp, seed, pass_name, follow, look) for c in cams]
        if not self.parked or look != self.look:  # the batch sensor sees the scene as set here
            self._set_camera(cams[0], width, height, hfov_deg, False, look)
        mi, k = self.mi, len(cams)
        integrator = self.integrator(pass_name)
        looks = []
        for c in cams:
            origin, target, up = frames.lookat(c)
            looks.append(mi.ScalarTransform4f.look_at(origin=origin, target=target, up=up))
        key = (k, int(width), int(height), round(float(hfov_deg), 6))
        if key not in self.batch_sensors:
            spec = {"type": "batch", "film": {"type": "hdrfilm", "width": k * int(width), "height": int(height)},
                    "sampler": {"type": "independent", "sample_count": 1}}
            for i, look in enumerate(looks):
                spec[f"slot{i}"] = {"type": "perspective", "fov": float(hfov_deg), "fov_axis": "x", "to_world": look}
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
                                         seed=int(seed) + 104729 * i), dtype=np.float64) * n
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
                    lambda scene, sn, sd, n=int(spp), it=integrator:
                    mi.render(scene, sensor=sn, integrator=it, spp=n, seed=sd),
                    backend=dr.JitBackend.CUDA, auto_opaque=False, warn_after=2)
                self._prepare_inputs(sensor, integrator)
                self._settle(self.frozen[fkey], self.scene, sensor)
            img = np.array(self.frozen[fkey](self.scene, sensor, dr.opaque(mi.UInt32, int(seed))))
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
                 envmap_dir: Path | None = None, sun_deg: float = 8.0, freeze_spp: int = 16):
        import drjit as dr
        import mitsuba as mi

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
        self.plans: dict[tuple[str, str, str], tuple] = {}
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
        self.freeze = hasattr(dr, "freeze") if freeze == "auto" else freeze == "on"
        self.preloading: list[list[str]] = []  # [scan, variant, mode] of --preload loads not done yet (main)
        # environment maps a view may ask for by name (file stem): <pack>/envmaps unless --envmaps says otherwise
        folder = Path(envmap_dir) if envmap_dir else next((p / "envmaps" for p in self.packs if (p / "envmaps").is_dir()),
                                                          self.pack / "envmaps")
        self.envmaps = {f.stem: f for f in sorted(folder.glob("*")) if f.suffix.lower() in (".exr", ".hdr")} \
            if folder.is_dir() else {}
        self.env_cache: dict[str, np.ndarray] = {}
        self.sun_deg = float(sun_deg)
        self.freeze_spp = int(freeze_spp)
        for scan in self.meta:
            synthesize_variants(self.meta[scan], (self.scene_dir(scan) / "perturbation_pool.json").is_file(),
                                self.has_nocaustics)

    def scene_dir(self, scan: str) -> Path:
        return self.pack_of.get(scan, self.pack) / "scenes" / scan

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

    def _on_main(self, mode: str, fn, *args):
        def job():
            if self.mi.variant() != self.variant_of[mode]:
                self.mi.set_variant(self.variant_of[mode])
            return fn(*args)
        return self.executor.submit(job).result()

    def preload(self, scan: str, variant: str, mode: str = "polar") -> float:
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
        scan, xml, flash_xml, mode = rkey
        while len(self.cache) >= self.max_resident:
            self.cache.popitem(last=False)
            gc.collect()
            # Dr.Jit keeps freed GPU buffers in its allocation cache; hand them back before the next scene
            # loads, or the old scene's memory and the new one's add up (28 of 32.6 GB on Device 1 while a
            # variant switch loaded, with another process holding 8.8 GB) and freeze recordings fail
            self.dr.flush_malloc_cache()
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
            loaded = Resident(self.mi, base / xml, mode, flash=flash, freeze=self.freeze,
                              flash_xml=base / flash_xml if flash_xml else None, hideable=hideable,
                              envs=self.env_image, freeze_spp=self.freeze_spp, nir=self.has_rgb2nir,
                              pool=self.pool(scan), upgrade_flash=self.has_nocaustics,
                              hidden=FILL_PREFIX if getattr(self, "has_hidden", False) else "")
        self.cache[rkey] = loaded
        return loaded

    def _denoise(self, mode: str, img: np.ndarray) -> np.ndarray:
        """OptiX AI denoiser on one non-negative HxWx3 radiance image (runs on the mode's worker)."""
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
        renderer = str(view.get("renderer", "path"))  # "raster": the rasterized preview (raster.py)
        if renderer not in ("path", "raster"):
            raise ValueError("renderer must be 'path' or 'raster'")
        raster_aa = str(view.get("raster_aa", "off"))  # the preview's antialiasing (raster.AA_MODES)
        if raster_aa not in ("off", "edge", "ssaa2"):
            raise ValueError("raster_aa must be 'off', 'edge' or 'ssaa2'")
        look = (lights, env_map, env_scale, env_rot % 360.0, bands, flash_band, tuple(sorted(nir.items())), lighting,
                mirrors, lamps, flash_scale, flash_pdf, flash_offset, max_depth, caustics, split, nir_range, rr_depth,
                renderer, raster_aa)
        return (scan, variant, mode, width, height, hfov, spp, seed, denoise, look), c2w

    def _start(self, key: tuple, cams: list, submitted: float, depth: bool = False):
        """Main-thread half one: queue the views on the GPU. Returns the second half (see submit_pipelined)."""
        started = time.perf_counter()
        scan, variant, mode, width, height, hfov, spp, seed, denoise, (lights, env, env_scale, env_rot, bands,
                                                                        flash_band, nir, lighting, mirrors, lamps,
                                                                        flash_scale, flash_pdf, flash_offset, max_depth,
                                                                        caustics, split, nir_range, rr_depth,
                                                                        renderer, raster_aa) = key
        if self.mi.variant() != self.variant_of[mode]:
            self.mi.set_variant(self.variant_of[mode])
        with ACTIVITY.doing("rendering"):
            rkey, passes, follow, hidden = self.plan(scan, variant, mode)
            resident = self._resident(rkey, variant)
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
                               split=split, nir_range=nir_range, rr_depth=rr_depth)}
            else:  # one band per render
                looks = {b: Look(hidden, lights, env, env_scale, env_rot, band=b, nir=nir, **flash, groups=groups, flash_scale=flash_scale,
                               flash_pdf=flash_pdf, flash_offset=flash_offset, max_depth=max_depth, caustics=caustics,
                               split=split, nir_range=nir_range, rr_depth=rr_depth)
                         for b in {"rgb": (0,), "nir": (1,), "both": (0, 1)}[bands]}
            if renderer == "raster":  # the rasterized preview (raster.py): drawn here, one image a view
                return self._raster_start(key, resident, cams, next(iter(looks.values())), follow, started - submitted,
                                          bands, depth, raster_aa)
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
                if self.readback is not None:  # each image copied out behind its own kernels (readback.py)
                    handles = {b: [(img if isinstance(img, np.ndarray) else self.readback.start(img), bd)
                                   for img, bd in hs] for b, hs in handles.items()}
            else:  # batch sensor: completes here
                batch = {b: [resident.render_many(cams, width, height, hfov, spp, seed, p, follow, look)
                             for p in passes] for b, look in looks.items()}
            # each view's depth (Resident.depth_image), queued behind its passes: the scene as they rendered it
            depths = [resident.depth_image(c, width, height, hfov) for c in cams] if depth else None
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
                            bands_out[want] = [[stokes.rgbn_band(v, want) for v in vs] for vs in views]
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

    def _raster_start(self, key: tuple, resident, cams: list, look, follow: bool, queue_s: float, bands: str,
                      depth: bool, aa: str = "off"):
        """_start for the rasterized preview: the views drawn and read back here, handed on in the path tracer's
        bands (RGB Stokes images, near-infrared ones) without its film split."""
        if not resident.rgbn:
            raise ValueError("the raster preview needs the RGBN polar variant (cuda_rgbn_polarized)")
        t0 = time.perf_counter()
        width, height, hfov = key[3], key[4], key[5]
        with ACTIVITY.doing("rasterizing"):
            views, depths, parts = resident.raster_views(cams, width, height, hfov, follow, look, aa)
        render_s = time.perf_counter() - t0
        parts.update(queue_s=queue_s, launch_s=0.0)

        def finish():
            def assemble():
                t1 = time.perf_counter()
                outs = []
                for lanes, z in zip(views, depths):
                    zero = np.zeros_like(lanes[0])
                    stokes_ = [zero if v is None else v for v in lanes]  # S0..S3: R, G, B, near-infrared
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
                    rad = sum(p[:, :, :3] for p in passes)
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
                out["radiance"] = sum(p[:, :, :3] for p in passes)
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
        if mode == "rgb":
            if "radiance" in out:
                rad = out["radiance"]
                result["radiance"] = rad.astype(dtype)
                if preview:
                    result["rgb"] = stokes.rgb_preview(rad, rad, np.zeros_like(rad), np.zeros_like(rad),
                                                       percentile=0.995, blur=0.0)
            return result
        if "s0" in out:
            result.update({k: out[k].astype(dtype) for k in stokes_keys})
            if preview:
                result["rgb"] = stokes.rgb_preview(out["rgb_src"], out["s0"], out["s1"], out["s2"])
        return result

    def render_views(self, views: list[dict], dtype) -> tuple[list[dict[str, np.ndarray]], dict[str, float]]:
        """Consecutive views with the same scan, variant, mode, size, spp, seed and denoiser share one
        render call (see Resident.render_many). Timing sums over calls."""
        checked = [self._check(v) for v in views]
        results, total = [], {}
        i = 0
        while i < len(checked):
            key = checked[i][0]
            j = i + 1
            while j < len(checked) and checked[j][0] == key:
                j += 1
            done = self.executor.submit_pipelined(self._start, key, [c for _, c in checked[i:j]],
                                                  time.perf_counter(),
                                                  any(bool(v.get("depth")) for v in views[i:j])).result()
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
                    return self._json(200, {
                        "activity": ACTIVITY.snapshot(), "freeze": renderer.freeze, "freeze_spp": renderer.freeze_spp,
                        # every variant a resident scene renders (an active variant and its flash-off twin share one)
                        "resident": [list(k) for k, _, _ in served],
                        "preloading": list(getattr(renderer, "preloading", [])),  # --preload scenes still loading
                        # freeze recordings per variant: [spp, width, height] recorded for all its passes
                        "recorded": {"|".join(k): renderer.recorded(r, passes) for k, r, passes in served},
                        # freeze recordings made so far by the resident scenes (one per new spp, size and pass);
                        # a change that should replay leaves it unchanged
                        "recordings": sum(getattr(f, "n_recordings", 0) for r in list(renderer.cache.values())
                                          for f in list(r.frozen.values()))})
                if parts == ["v1", "info"]:
                    return self._json(200, {
                        "server": "opticalnav_sim", "mitsuba_variant": renderer.variant,
                        "mitsuba_version": getattr(getattr(renderer, "mi", None), "__version__", None),
                        "modes": renderer.variant_of, "path_nocaustics": renderer.has_nocaustics,
                        "envmaps": sorted(renderer.envmaps), "nir": renderer.has_rgb2nir,
                        "active_split": getattr(renderer, "has_split", False),
                        "nir_defaults": NIR_DEFAULTS,
                        "drjit_freeze": renderer.freeze, "spp_chunk": SPP_CHUNK,
                        "scans": {s: renderer.variants(s) for s in renderer.meta},
                        "resident": [list(k) + [round(r.load_s, 1)] for k, r, _ in renderer.served()],
                        "preloading": list(getattr(renderer, "preloading", [])),
                    })
                if parts == ["v1", "scans"]:
                    return self._json(200, sorted(renderer.meta))
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
                started = time.time()
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
                                     {"X-Render-Seconds": f"{time.time() - started:.4f}",
                                      "X-Render-Timing": json.dumps({k: round(v, 4) for k, v in total.items()})})
                    return
                buf = io.BytesIO()
                np.savez(buf, **arrays)
                total["encode_s"] = time.perf_counter() - t_enc
                self._send(200, buf.getvalue(), "application/octet-stream",
                           {"X-Render-Seconds": f"{time.time() - started:.4f}",
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
    todo = [s for s in sorted(renderer.meta) if s not in skip and compact.needs_compaction(renderer.scene_dir(s))]
    print(f"[compact] {len(todo)} of {len(renderer.meta)} scenes need compacting: {todo}", flush=True)
    for scan in todo:
        started = time.time()
        with ACTIVITY.doing(f"compacting {scan}"):
            done = subprocess.run([sys.executable, str(tool), "--pack", str(renderer.pack_of[scan]), "--scene", scan,
                                   "--if-needed"], check=False)
        if done.returncode == 0:
            renderer.reload_meta(scan)
            print(f"[compact] {scan} in {time.time() - started:.0f} s", flush=True)
        else:
            print(f"[compact] {scan} failed (exit {done.returncode}): served as it is", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, action="append",
                    help="scene pack directory (tools/build_pack.py output); repeat to serve several (a scene name "
                         "in two packs comes from the first)")
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
    ap.add_argument("--max-resident", type=int, default=2, help="scene/variant/mode triples kept loaded")
    ap.add_argument("--preload", action="append", default=[], help="SCAN[:VARIANT[:MODE]] to load at start (repeatable)")
    ap.add_argument("--freeze", choices=["auto", "on", "off"], default="auto",
                    help="replay renders with dr.freeze (auto: when this Dr.Jit has it)")
    ap.add_argument("--freeze-spp", type=int, default=16,
                    help="record frozen renders at this spp only: a request for a multiple of it replays that "
                         "recording several times (new seeds, averaged) instead of recording again; 0 records each spp")
    ap.add_argument("--sun-deg", type=float, default=8.0,
                    help="spread an environment map's sun over a Gaussian of this many degrees, keeping its power: "
                         "sunlight through window glass otherwise renders as fireflies (0: as the file has it)")
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
    renderer = Renderer([Path(p).resolve() for p in args.pack], args.variant, args.max_resident, args.freeze, args.rgb_variant, modes,
                        kernel_history=args.kernel_history, envmap_dir=args.envmaps, sun_deg=args.sun_deg,
                        freeze_spp=args.freeze_spp)
    # listen first: while --preload loads, /v1/status reports the load's progress and renders wait in the queue
    server = ThreadingHTTPServer((args.host, args.port), make_handler(renderer, args.token))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[serve] http://{args.host}:{args.port}  modes={renderer.variant_of}  scans={len(renderer.meta)}  "
          f"path_nocaustics={renderer.has_nocaustics}  freeze={renderer.freeze}  "
          f"kernel_history={renderer.kernel_history}", flush=True)
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
