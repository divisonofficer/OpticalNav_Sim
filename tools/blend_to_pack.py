#!/usr/bin/env python3
"""Convert a scene's .blend straight into a pack scene for the Mitsuba render server.

    python3 tools/blend_to_pack.py --pack packs/opticalnav-sim-v1 --scene sim_v2a_REF_PH19 [--scene-id sim_v3_REF_PH19]
                                   [--blend FILE] [--origin-offset OX OY OZ] [--max-texture 4096]

robomituba's importer (Blender bake per object -> authoring map -> stage -> build_pack -> compact) takes hours on a large
scene (its own log comment: 976 objects in 13.6 h) because it bakes every object's albedo, coverage, normal, roughness and
metallic maps through Cycles. This converter reads the .blend's node graphs instead, in one Blender process:

* meshes: each object's evaluated, triangulated mesh, split by material slot, into the Mitsuba world
  (``Mitsuba = (bx + ox, bz + oz, oy - by)``, see opticalnav_sim/cycles_backend.py), welded, as binary PLY with normals and UVs;
* materials: a Principled BSDF becomes ``pplastic`` (dielectric base; alpha = roughness squared, written here, so the
  server's ``--alpha-power`` does not apply to these scenes), ``roughconductor`` (metallic: eta and k from the base colour,
  Gulbrandsen), ``blendbsdf`` between the two for a metallic map or a middle value, ``dielectric`` / ``roughdielectric``
  (transmission), under ``normalmap`` (the author's own normal image), ``mask`` (alpha) and ``twosided``. Colour, roughness and
  metallic that are a constant or an image (through a Mapping node and a channel split) are written as constants or
  bitmaps with the image's own bytes; what is not (a composite with noise: dirt, scuffs) is counted in the report and
  falls back to its image;
* Emission shaders and Principled emission become area emitters on their shapes; Blender lights become the same boxes
  with ``metadata.light`` the importer wrote, so the server's lamp and Blender-power logic reads them as before.

The scene goes to ``<pack>/scenes/<scene-id>/``: ``base.xml``, ``assets/``, ``scene.json`` (the template scene's, with the
new ids and ``variants.base.protocol = "blend_direct_v1"``) and the ``.blend`` itself (``blend/scene.blend``, packed). Run from
the repo; Blender starts like the render server starts it (OPTICALNAV_BLENDER, the WSL driver dirs).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INSIDE = os.environ.get("OPTICALNAV_B2M_INSIDE") == "1"  # set for the Blender process this script starts
IMPORTS = Path("/jarvis/project/robomituba/out/infinigen_imports")
ALPHA_MIN = 0.01  # a roughness under 0.1 (a puddle's 0.02: alpha 0.0004) is a near-delta lobe the path tracer cannot sample: black patches
IMPORTER_WATTS = 0.025  # the importer's light boxes glow at (1 + 0.025 W) x colour (server.IMPORTER_WATTS)


# ---------------------------------------------------------------- inside Blender
def run_inside(cfg: dict) -> None:
    import bpy
    import numpy as np

    t_start = time.perf_counter()
    out = Path(cfg["out"])
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    report = {"objects": 0, "shapes": 0, "triangles": 0, "materials": {}, "warnings": [], "lights": 0}
    bpy.ops.wm.open_mainfile(filepath=cfg["blend"])
    scene = bpy.context.scene
    if cfg["origin_offset"] is None:
        # no importer offset (a scene the old pipeline never imported): the plan frame starts at the scene's bounding box, so x, y and height
        # are non-negative like the other scenes' (plan x = bx + ox, plan y = oy - by, height = bz + oz)
        lo_all, hi_all = np.full(3, np.inf), np.full(3, -np.inf)
        for o in scene.objects:
            if o.type == "MESH" and not o.hide_render:
                corners = np.array([np.array(o.matrix_world @ __import__("mathutils").Vector(c)) for c in o.bound_box])
                lo_all, hi_all = np.minimum(lo_all, corners.min(0)), np.maximum(hi_all, corners.max(0))
        cfg["origin_offset"] = [round(float(-lo_all[0]), 3), round(float(hi_all[1]), 3), round(float(-lo_all[2]), 3)]
    ox, oy, oz = cfg["origin_offset"]
    report["origin_offset"] = cfg["origin_offset"]
    L = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])  # Blender -> Mitsuba, rotation part
    T = np.array([ox, oz, oy])
    report["cameras"] = []
    for o in scene.objects:  # the authors' viewpoints: spawn poses (R2R frame: x, -y_plan; heading = 180 - yaw, yaw looks along (sin, cos) of plan (x, y))
        if o.type == "CAMERA":
            wm = np.array(o.matrix_world)
            fwd = -wm[:3, 2]
            yaw = math.degrees(math.atan2(fwd[0], -fwd[1]))
            report["cameras"].append({"name": o.name, "position": [round(float(wm[0, 3] + ox), 3), round(float(wm[1, 3] - oy), 3), round(float(wm[2, 3] + oz), 3)],
                                      "heading_deg": round((180.0 - yaw) % 360.0, 1)})
    dg = bpy.context.evaluated_depsgraph_get()

    def warn(msg: str) -> None:
        if msg not in report["warnings"]:
            report["warnings"].append(msg)

    def san(name: str) -> str:
        return re.sub(r"[^0-9A-Za-z._:-]+", "_", str(name)).strip("_") or "x"

    def f(v, n=6) -> str:
        return f"{float(v):.{n}f}"

    def rgb(v) -> str:
        return " ".join(f"{max(float(c), 0.0):.5f}" for c in list(v)[:3])

    # ------------------------------------------------ textures
    saved: dict[str, str] = {}

    def image_file(img) -> str | None:
        """The image's own bytes (packed data or the file on disk) as assets/tex_<hash>.<ext>; None if there are none."""
        key = img.name
        if key in saved:
            return saved[key]
        data = None
        if img.packed_file is not None:
            data = bytes(img.packed_file.data)
            ext = Path(img.filepath or "x.png").suffix.lower() or ".png"
        else:
            p = Path(bpy.path.abspath(img.filepath))
            if p.is_file():
                data, ext = p.read_bytes(), p.suffix.lower()
        if data is None:
            warn(f"image {img.name}: no data")
            saved[key] = None
            return None
        if ext not in (".png", ".jpg", ".jpeg", ".exr", ".bmp", ".tga"):
            ext = ".png"
        name = f"tex_{hashlib.sha1(data).hexdigest()[:20]}{ext}"
        if not (assets / name).exists():
            (assets / name).write_bytes(data)  # the driver shrinks what is over --max-texture
        saved[key] = f"assets/{name}"
        return saved[key]

    # ------------------------------------------------ node graph evaluation
    # A socket is evaluated to a constant or to an image-sized array (an Image Texture seen through Mapping, colour ops, Mix,
    # Math ...), so a composite the importer had to bake through Cycles is computed here from the image's own pixels. A noise
    # texture counts as its mean (0.5): the dirt's average tone stays, its grain does not.
    class V:
        __slots__ = ("c", "a", "uv", "coord", "src", "proj")

        def __init__(self, c=None, a=None, uv=None, coord="UV", src=None, proj=None):
            # src: the Image of an untouched colour read; proj: for Object / Generated coordinates, how the image is projected
            self.c, self.a, self.uv, self.coord, self.src, self.proj = c, a, uv, coord, src, proj

    IDENT = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
    px_cache: dict[str, "np.ndarray"] = {}

    def skip_reroute(sock):
        while sock.is_linked and sock.links[0].from_node.type == "REROUTE":
            sock = sock.links[0].from_node.inputs[0]
        return sock

    def srgb_to_linear(x):
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    def linear_to_srgb(x):
        x = np.clip(x, 0, 1)
        return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)

    def image_array(img) -> "np.ndarray":
        """(h, w, 4) float32, linear for colour images (sRGB decoded here: pixels returns the stored values), raw otherwise."""
        key = img.name
        if key not in px_cache:
            w, h = img.size
            a = np.empty(w * h * 4, dtype=np.float32)
            img.pixels.foreach_get(a)
            a = a.reshape(h, w, 4)[::-1].copy()
            if img.colorspace_settings.name.startswith("sRGB"):
                a[..., :3] = srgb_to_linear(a[..., :3])
            px_cache[key] = a
        return px_cache[key]

    def coords_of(node) -> tuple[list, str, dict | None] | None:
        """(2D uv matrix, coordinate source UV | OBJECT | GENERATED, projection) an Image Texture's Vector input amounts to; None if
        it is something a mesh cannot carry (camera, window, normal ...). For Object / Generated coordinates the Mapping node is
        a 3D transform applied to the coordinate before the image's projection (FLAT: x, y; BOX: the axis pair of the face's
        dominant normal axis), so it is returned as ``proj`` and the uv matrix is the identity."""
        sock = skip_reroute(node.inputs["Vector"])
        if not sock.is_linked:
            return IDENT, "UV", None
        src = sock.links[0].from_node
        mapping = None
        if src.type == "MAPPING":
            if any(src.inputs[n].is_linked for n in ("Location", "Rotation", "Scale")) or src.vector_type not in ("POINT", "TEXTURE"):
                return None
            mapping = src
            sock = skip_reroute(src.inputs["Vector"])
            if not sock.is_linked:
                src, name = None, "UV"
            else:
                src, name = sock.links[0].from_node, sock.links[0].from_socket.name
        else:
            name = sock.links[0].from_socket.name
        if src is not None and src.type == "UVMAP":  # the UV Map node: the mesh's uv (a named map: the active one is used)
            src, name = None, "UV"
        if src is not None and src.type == "NEW_GEOMETRY" and name == "Position":  # the Geometry node's Position: world space
            coord = "WORLD"
        elif src is not None and not (src.type == "TEX_COORD" and name in ("UV", "Object", "Generated")):
            return None
        else:
            coord = name.upper() if src is not None else "UV"
        if coord == "UV":
            if mapping is None:
                return IDENT, "UV", None
            loc, rot, sc = (np.array(mapping.inputs[n].default_value, dtype=float) for n in ("Location", "Rotation", "Scale"))
            c, s = math.cos(rot[2]), math.sin(rot[2])
            return [sc[0] * c, -sc[1] * s, loc[0], sc[0] * s, sc[1] * c, loc[1], 0, 0, 1.0], "UV", None
        M, loc = np.eye(3), np.zeros(3)
        if mapping is not None:
            lo, ro, so = (np.array(mapping.inputs[n].default_value, dtype=float) for n in ("Location", "Rotation", "Scale"))
            cx, cy, cz = np.cos(ro)
            sx, sy, sz = np.sin(ro)
            R = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ \
                np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            if mapping.vector_type == "POINT":  # vec' = R (S vec) + loc
                M, loc = R @ np.diag(so), lo
            else:  # TEXTURE: the inverse
                M = np.diag(1 / so) @ R.T
                loc = -M @ lo
        return IDENT, coord, {"coord": coord, "M": M.ravel().tolist(), "loc": loc.tolist(), "projection": node.projection}

    def const(v) -> V:
        a = np.array(v if hasattr(v, "__len__") else [v] * 3, dtype=np.float32)[:3]
        return V(c=a)

    def lerp(fac, a, b):
        return a + (b - a) * fac

    def blend(kind: str, a, b):
        if kind == "MULTIPLY":
            return a * b
        if kind == "ADD":
            return a + b
        if kind == "SUBTRACT":
            return a - b
        if kind == "SCREEN":
            return 1 - (1 - a) * (1 - b)
        if kind == "DARKEN":
            return np.minimum(a, b)
        if kind == "LIGHTEN":
            return np.maximum(a, b)
        if kind == "DIVIDE":
            return a / np.maximum(b, 1e-6)
        if kind == "OVERLAY":
            return np.where(a < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
        if kind == "DIFFERENCE":
            return np.abs(a - b)
        return b  # MIX

    def fit(a: "np.ndarray", shape: tuple) -> "np.ndarray":
        """``a`` (h, w, c) resampled (nearest) to ``shape`` (H, W): two images of different sizes meeting in one node."""
        h, w = a.shape[:2]
        return a[(np.arange(shape[0]) * h // shape[0])][:, (np.arange(shape[1]) * w // shape[1])]

    def combine(f, *vs: "V") -> V:
        """f applied to the constants / arrays of vs (the biggest array's mapping is kept; smaller ones are resampled to it)."""
        arrs = [v for v in vs if v.a is not None]
        arr = max(arrs, key=lambda v: v.a.shape[0] * v.a.shape[1]) if arrs else None
        vals = [(fit(v.a, arr.a.shape[:2]) if v.a is not None else v.c) for v in vs]
        out = f(*vals)
        if arr is None:
            return V(c=np.asarray(out, dtype=np.float32))
        return V(a=np.asarray(out, dtype=np.float32), uv=arr.uv, coord=arr.coord, proj=arr.proj)

    def value_of(sock, depth: int = 0) -> V | None:
        sock = skip_reroute(sock)
        if not sock.is_linked:
            dv = getattr(sock, "default_value", 0.0)
            return const(dv if hasattr(dv, "__len__") else float(dv))
        if depth > 24:
            return None
        link = sock.links[0]
        n, out = link.from_node, link.from_socket.name
        t = n.type

        def inp(name, i=None):
            return value_of(n.inputs[name if i is None else i], depth + 1)
        if t == "TEX_IMAGE":
            if n.image is None:
                return None
            co = coords_of(n)
            if co is None:
                warn("image texture on camera/window/normal coordinates")
                return None
            a = image_array(n.image)
            a = a[..., 3:4].repeat(3, axis=2) if out == "Alpha" else a[..., :3]
            return V(a=a, uv=co[0], coord=co[1], src=n.image if out == "Color" else None, proj=co[2])
        if t in ("TEX_NOISE", "TEX_VORONOI", "TEX_MUSGRAVE", "TEX_WHITE_NOISE", "TEX_WAVE", "TEX_MAGIC", "TEX_GRADIENT", "TEX_CHECKER", "SEPXYZ",
                 "TEX_COORD", "NEW_GEOMETRY", "OBJECT_INFO", "AMBIENT_OCCLUSION", "LAYER_WEIGHT", "FRESNEL", "BEVEL"):
            return const(0.5)  # a procedural pattern: its mean
        if t == "ATTRIBUTE":  # a vertex colour (baked occlusion): neutral, it varies per vertex
            warn("attribute node read as 1.0 (a vertex colour)")
            return const(1.0)
        if t in ("COMBINE_COLOR", "COMBINE_RGB"):
            parts = [value_of(i, depth + 1) for i in n.inputs[:3]]
            if None in parts:
                return None
            return combine(lambda r, g, b: np.stack([np.asarray(r)[..., 0] if np.ndim(r) == 3 else np.asarray(r).ravel()[0] * np.ones(1)[0],
                                                     np.asarray(g)[..., 0] if np.ndim(g) == 3 else np.asarray(g).ravel()[0] * np.ones(1)[0],
                                                     np.asarray(b)[..., 0] if np.ndim(b) == 3 else np.asarray(b).ravel()[0] * np.ones(1)[0]], axis=-1),
                           *parts)
        if t == "RGB":
            return const(list(n.outputs[0].default_value))
        if t == "VALUE":
            return const(float(n.outputs[0].default_value))
        if t in ("SEPARATE_COLOR", "SEPARATE_RGB"):
            v = inp(0)
            if v is None:
                return None
            ch = {"R": 0, "G": 1, "B": 2, "Red": 0, "Green": 1, "Blue": 2}[out]
            return combine(lambda x: np.repeat(np.asarray(x)[..., ch:ch + 1], 3, axis=-1) if np.ndim(x) == 3 else np.full(3, np.asarray(x)[ch]), v)
        if t == "HUE_SAT":
            c, h, s, vv, fac = inp("Color"), inp("Hue"), inp("Saturation"), inp("Value"), inp("Fac")
            if None in (c, h, s, vv, fac):
                return None
            hue, sat, val, fc = float(h.c[0]), float(s.c[0]), float(vv.c[0]), float(fac.c[0])

            def hsv(x):
                x = np.asarray(x, dtype=np.float32)
                r, g, b = x[..., 0], x[..., 1], x[..., 2]
                mx, mn = x.max(-1), x.min(-1)
                d = mx - mn
                hh = np.where(d == 0, 0, np.where(mx == r, ((g - b) / np.maximum(d, 1e-9)) % 6, np.where(mx == g, (b - r) / np.maximum(d, 1e-9) + 2, (r - g) / np.maximum(d, 1e-9) + 4))) / 6
                ss = np.where(mx == 0, 0, d / np.maximum(mx, 1e-9))
                hh = (hh + hue - 0.5) % 1.0
                ss = np.clip(ss * sat, 0, 1)
                vv_ = mx * val
                k = lambda nn: (nn + hh * 6) % 6
                fn = lambda nn: vv_ - vv_ * ss * np.clip(np.minimum(k(nn), 4 - k(nn)), 0, 1)
                res = np.stack([fn(5), fn(3), fn(1)], axis=-1)
                return lerp(fc, x, res)
            return combine(hsv, c)
        if t in ("MIX_RGB", "MIX"):
            if t == "MIX" and n.data_type == "VECTOR":
                return None
            # the Mix node has a Factor / A / B for each data type under the same names: the enabled ones are this type's
            live = [i for i in n.inputs if i.enabled]
            fac, a_, b_ = (value_of(live[0], depth + 1), value_of(live[1], depth + 1), value_of(live[2], depth + 1))
            if None in (fac, a_, b_):
                return None
            kind = n.blend_type
            def mix(f_, x, y):
                f_ = np.asarray(f_)[..., :1] if np.ndim(f_) == 3 else float(f_.ravel()[0])
                return lerp(f_, x, blend(kind, x, y))
            return combine(mix, fac, a_, b_)
        if t == "MAP_RANGE":
            v, a0, a1, b0, b1 = (inp(k) for k in ("Value", "From Min", "From Max", "To Min", "To Max"))
            if None in (v, a0, a1, b0, b1) or n.data_type != "FLOAT":
                return None
            f0, f1, t0, t1 = (float(x.c[0]) for x in (a0, a1, b0, b1))
            clamp = n.clamp
            def mr(x):
                y = (x - f0) / (f1 - f0 if f1 != f0 else 1e-9) * (t1 - t0) + t0
                return np.clip(y, min(t0, t1), max(t0, t1)) if clamp else y
            return combine(mr, v)
        if t == "MATH":
            xs = [value_of(i, depth + 1) for i in n.inputs[:3] if i.enabled]
            if None in xs:
                return None
            op = n.operation
            tab = {"ADD": lambda a, b=0, c=0: a + b, "SUBTRACT": lambda a, b=0, c=0: a - b, "MULTIPLY": lambda a, b=1, c=0: a * b,
                   "DIVIDE": lambda a, b=1, c=0: a / np.where(np.asarray(b) == 0, 1e-9, b), "POWER": lambda a, b=1, c=0: np.power(np.maximum(a, 0), b),
                   "MAXIMUM": lambda a, b=0, c=0: np.maximum(a, b), "MINIMUM": lambda a, b=0, c=0: np.minimum(a, b),
                   "MULTIPLY_ADD": lambda a, b=1, c=0: a * b + c, "SQRT": lambda a, b=0, c=0: np.sqrt(np.maximum(a, 0)),
                   "ABSOLUTE": lambda a, b=0, c=0: np.abs(a), "LESS_THAN": lambda a, b=0, c=0: (a < b) * 1.0,
                   "GREATER_THAN": lambda a, b=0, c=0: (a > b) * 1.0}
            if op not in tab:
                warn(f"math operation {op}")
                return None
            r = combine(tab[op], *xs)
            if n.use_clamp:
                r = combine(lambda x: np.clip(x, 0, 1), r)
            return r
        if t == "BRIGHTCONTRAST":
            c, br, co_ = inp("Color"), inp("Bright"), inp("Contrast")
            if None in (c, br, co_):
                return None
            a_, b_ = 1 + float(co_.c[0]), float(br.c[0]) - float(co_.c[0]) * 0.5
            return combine(lambda x: np.maximum(a_ * x + b_, 0), c)
        if t == "RGBTOBW":
            c = inp("Color")
            if c is None:
                return None
            return combine(lambda x: np.tensordot(np.asarray(x, dtype=np.float32), np.array([0.2126, 0.7152, 0.0722], np.float32), axes=([-1], [0]))[..., None]
                           * np.ones(3, np.float32), c)
        if t == "INVERT":
            fac, c = inp("Fac"), inp("Color")
            if None in (fac, c):
                return None
            f_ = float(fac.c[0])
            return combine(lambda x: lerp(f_, x, 1 - x), c)
        if t == "CLAMP":
            v = inp("Value")
            lo, hi = (float(value_of(n.inputs[k], depth + 1).c[0]) for k in ("Min", "Max"))
            return None if v is None else combine(lambda x: np.clip(x, lo, hi), v)
        if t == "VALTORGB":
            v = inp("Fac")
            if v is None:
                return None
            els = [(e.position, np.array(e.color[:3], dtype=np.float32)) for e in n.color_ramp.elements]
            pos = np.array([e[0] for e in els], dtype=np.float32)
            cols = np.stack([e[1] for e in els])
            def ramp(x):
                x = np.asarray(x)[..., 0] if np.ndim(x) == 3 else np.asarray(x)[0]
                return np.stack([np.interp(x, pos, cols[:, k]) for k in range(3)], axis=-1)
            return combine(ramp, v)
        warn(f"node {t}")
        return None

    def emission_xml(color_sock, strength: float, ind: str) -> tuple[str, list] | None:
        """(the emitter's radiance element, its mean radiance) of an Emission colour socket. An image (a painting, a screen, a sign)
        is flattened to its mean: a textured area emitter makes Dr.Jit's frozen recording fail in the polarized CUDA variant
        (20 of PH33's emitters; replaced by constants it renders)."""
        v = value_of(color_sock)
        if v is None:
            return None
        if v.a is None:
            rad = [float(c) * strength for c in np.asarray(v.c).ravel()[:3]]
        else:
            rad = [float(x) * strength for x in v.a.reshape(-1, 3).mean(0)]
            report["textured_emission_flattened"] = report.get("textured_emission_flattened", 0) + 1
        return f'{ind}<rgb name="radiance" value="{rgb(rad)}" />\n', rad

    def write_tex(name: str, v: V, kind: str, ind: str, stat: dict, squared: bool = False) -> str:
        """A bitmap texture of an array value. An untouched image keeps its own bytes; else colours go out as 8-bit sRGB, normal
        maps as 8-bit raw, scalars as 16-bit (saved as .npy: the driver makes the PNG)."""
        if stat.setdefault("coord", v.coord) != v.coord:
            warn(f"material mixes {stat['coord']} and {v.coord} texture coordinates")
        if v.proj is not None:
            stat.setdefault("proj", v.proj)
        if v.src is not None and kind in ("colour", "normal"):
            path = image_file(v.src)
            if path:
                return tex_xml(name, path, v.uv or IDENT, kind != "colour", ind)
        if kind == "colour":
            arr, tag, ext = (linear_to_srgb(np.clip(v.a, 0, None)) * 255 + 0.5).astype(np.uint8), "c", "jpg"  # smooth composites: JPEG q95
        elif kind == "normal":
            arr, tag, ext = (np.clip(v.a, 0, 1) * 255 + 0.5).astype(np.uint8), "n", "png"
        else:  # roughness / metallic / alpha: smooth, 1024 px is enough (the driver shrinks them)
            g = np.clip(v.a[..., 0] if v.a.ndim == 3 else v.a, 0, 1)
            arr, tag, ext = ((np.maximum(g ** 2, ALPHA_MIN) if squared else g) * 65535 + 0.5).astype(np.uint16), "s2" if squared else "s", "png"
        stem = f"tex_{hashlib.sha1(arr.tobytes() + tag.encode()).hexdigest()[:20]}_{tag[0]}"
        if not (assets / f"{stem}.{ext}").exists() and not (assets / f"{stem}.npy").exists():
            np.save(assets / f"{stem}.npy", arr)
        return tex_xml(name, f"assets/{stem}.{ext}", v.uv or IDENT, kind != "colour", ind)

    # ------------------------------------------------ materials -> Mitsuba bsdf xml
    def tex_xml(name: str, path: str, uv: list, raw: bool, ind: str) -> str:
        """A bitmap texture; ``uv`` is Blender's 3x3 coordinate matrix (v up). Mitsuba's PLY loader does not flip v and its images
        run top down, so the matrix written is flipped: v' = 1 - v (checked on PH19's "16" graffiti: IoU 0.95 against Cycles,
        0.35 without)."""
        m = np.array(uv, dtype=float).reshape(3, 3)
        m = np.array([[1.0, 0, 0], [0, -1.0, 1.0], [0, 0, 1.0]]) @ m
        s = f'{ind}<texture name="{name}" type="bitmap">\n{ind}  <string name="filename" value="{path}" />\n'
        if raw:
            s += f'{ind}  <boolean name="raw" value="true" />\n'
        s += f'{ind}  <transform name="to_uv"><matrix value="{" ".join(f(v) for v in m.ravel())}" /></transform>\n'
        return s + f"{ind}</texture>\n"

    def spec(sock, name: str, ind: str, kind: str, stat: dict, squared: bool = False):
        """A pplastic / conductor parameter of a socket: (xml, constant or None). ``kind``: colour | scalar."""
        v = value_of(sock)
        if v is None:
            stat["unsupported"] = stat.get("unsupported", 0) + 1
            dv = getattr(skip_reroute(sock), "default_value", 0.5)
            v = const(dv if hasattr(dv, "__len__") else float(dv))
        if v.a is None:
            c = [float(x) for x in np.asarray(v.c).ravel()]
            if kind == "colour":
                c = (c * 3)[:3] if len(c) == 1 else c[:3]
                return f'{ind}<rgb name="{name}" value="{rgb(c)}" />\n', c
            return f'{ind}<float name="{name}" value="{f(max(c[0] ** 2, ALPHA_MIN) if squared else c[0])}" />\n', c[0]
        stat["textured"] = stat.get("textured", 0) + 1
        return write_tex(name, v, kind, ind, stat, squared), None

    def gulbrandsen(r) -> tuple[list, list]:
        r = [min(max(float(c), 0.0), 0.995) for c in r]
        return [(1 - c) / (1 + c) for c in r], [2 * math.sqrt(c) / math.sqrt(1 - c) for c in r]

    def scalar_const(sock, default=0.0):
        v = value_of(sock)
        return (float(np.asarray(v.c).ravel()[0]), True) if v is not None and v.a is None else (default, False)

    alpha_of: dict[str, V] = {}  # material name -> its alpha texture (cutouts are cut from it)
    emission_img: dict[str, tuple[V, float]] = {}  # material name -> (its emission image, strength): the glowing part is cut from it
    cutting = [False]  # while True, an alpha texture does not make a mask: the shape's geometry is cut instead

    def principled(node, stat: dict) -> tuple[str, list | None]:
        ind = "      "
        base = node.inputs["Base Color"]
        rough = node.inputs["Roughness"]
        metal, metal_const = scalar_const(node.inputs["Metallic"])
        trans, _ = scalar_const(node.inputs["Transmission Weight"])
        ior = float(node.inputs["IOR"].default_value) if not node.inputs["IOR"].is_linked else 1.5
        emission = None
        es = node.inputs["Emission Strength"]
        if not es.is_linked and float(es.default_value) > 0:
            emission = emission_xml(node.inputs["Emission Color"], float(es.default_value), "      ")
            if emission is not None and max(emission[1]) <= 0:
                emission = None
        if trans >= 0.5:
            rv, _ = scalar_const(rough, 0.1)
            stat["class"] = "dielectric"
            tint = value_of(base)
            tint_xml = ""
            if tint is not None and tint.a is None and min(tint.c[:3]) < 0.95:
                tint_xml = f'{ind}  <rgb name="specular_transmittance" value="{rgb(tint.c)}" />\n'
            if rv >= 0.15:
                core = f'{ind}<bsdf type="roughdielectric">\n{ind}  <float name="int_ior" value="{f(ior, 4)}" />\n' \
                       f'{ind}  <float name="alpha" value="{f(rv ** 2)}" />\n{tint_xml}{ind}</bsdf>\n'
            else:
                core = f'{ind}<bsdf type="dielectric">\n{ind}  <float name="int_ior" value="{f(ior, 4)}" />\n{tint_xml}{ind}</bsdf>\n'
            return core, emission
        diffuse, bcol = spec(base, "diffuse_reflectance", ind + "    ", "colour", stat)
        alpha_xml, _ = spec(rough, "alpha", ind + "    ", "scalar", stat, squared=True)
        plastic = f'{ind}<bsdf type="pplastic">\n{ind}  <float name="int_ior" value="{f(ior, 4)}" />\n{diffuse}{alpha_xml}{ind}</bsdf>\n'
        two = lambda body: f'{ind}<bsdf type="twosided">\n{body}{ind}</bsdf>\n'  # mask / normalmap wrap it: twosided cannot nest a mask
        if metal_const and metal <= 0.05:
            stat["class"] = "pplastic"
            core = plastic
        else:
            if bcol is not None:
                eta, k = gulbrandsen(bcol)
                spec_xml = f'{ind}    <rgb name="eta" value="{rgb(eta)}" />\n{ind}    <rgb name="k" value="{rgb(k)}" />\n'
            else:  # a textured metal: aluminium's optics, the texture as its reflectance
                spec_xml = f'{ind}    <string name="material" value="Al" />\n' + \
                    spec(base, "specular_reflectance", ind + "    ", "colour", stat)[0]
            conductor = f'{ind}<bsdf type="roughconductor">\n{spec_xml}{alpha_xml}{ind}</bsdf>\n'
            if metal_const and metal >= 0.95:
                stat["class"] = "roughconductor"
                core = conductor
            else:
                stat["class"] = "blend(pplastic, roughconductor)"
                w_xml, _ = spec(node.inputs["Metallic"], "weight", ind + "  ", "scalar", stat)
                core = f'{ind}<bsdf type="blendbsdf">\n{w_xml}{plastic}{conductor}{ind}</bsdf>\n'
        core = two(core)
        av = value_of(node.inputs["Alpha"])
        if av is not None and av.a is None and float(av.c[0]) < 0.99:
            core = f'{ind}<bsdf type="mask">\n{ind}  <float name="opacity" value="{f(av.c[0])}" />\n{core}{ind}</bsdf>\n'
        elif av is not None and av.a is not None:
            alpha_of[stat["name"]] = av
            stat["alpha_texture"] = True
            if not cutting[0]:
                core = f'{ind}<bsdf type="mask">\n{write_tex("opacity", av, "scalar", ind + "  ", stat)}{core}{ind}</bsdf>\n'
        nsock = skip_reroute(node.inputs["Normal"])
        if nsock.is_linked and nsock.links[0].from_node.type == "NORMAL_MAP":
            nm = nsock.links[0].from_node
            tv = value_of(nm.inputs["Color"])
            if tv is not None and tv.a is not None:
                core = f'{ind}<bsdf type="normalmap">\n{write_tex("normalmap", tv, "normal", ind + "  ", stat)}{core}{ind}</bsdf>\n'
                stat["normalmap"] = True
        elif nsock.is_linked:
            stat["bump_ignored"] = True
        return core, emission

    cache: dict[tuple, tuple[str, list | None]] = {}

    def material(m, cutout: bool = False) -> tuple[str, list | None]:
        """(bsdf xml, emission radiance or None) of a material (cached); ``cutout``: no mask for an alpha texture."""
        if m is None:
            return default_bsdf(), None
        if (m.name, cutout) in cache:
            return cache[(m.name, cutout)]
        cutting[0] = cutout
        stat: dict = {"name": m.name}
        out_node = next((n for n in m.node_tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None) if m.use_nodes else None
        shader = skip_reroute(out_node.inputs["Surface"]).links[0].from_node if out_node and skip_reroute(out_node.inputs["Surface"]).is_linked else None
        res = (default_bsdf(), None)
        emission = None
        if shader is not None and shader.type == "MIX_SHADER":  # Transparent + Principled (glass that shadow rays pass): the Principled
            branches = [skip_reroute(i).links[0].from_node for i in shader.inputs[1:] if skip_reroute(i).is_linked]
            shader = next((b for b in branches if b.type != "BSDF_TRANSPARENT"), shader)
        if shader is not None and shader.type == "BSDF_PRINCIPLED":
            core, emission = principled(shader, stat)
            res = (core, emission)
        elif shader is not None and shader.type == "EMISSION":
            st, _ = scalar_const(shader.inputs["Strength"], 1.0)
            img = value_of(shader.inputs["Color"])
            if img is not None and img.a is not None and st <= float(cfg["print_max_strength"]):
                # a picture the author showed by Emission (PH33's banners at strength 1, its wayfinding signs at 3) stands in for a print: in the
                # room's light it is a diffuse, slightly glossy surface, so it polarizes like one (an emitter's light is unpolarized: DoLP 0).
                # Image emissions stronger than --print-max-strength (the room numerals, 14) stay emitters, flattened to their mean
                alb, _ = spec(shader.inputs["Color"], "diffuse_reflectance", "        ", "colour", stat)
                res = ('    <bsdf type="twosided">\n      <bsdf type="pplastic">\n        <float name="int_ior" value="1.5000" />\n'
                       f'{alb}        <float name="alpha" value="0.3000" />\n      </bsdf>\n    </bsdf>\n', None)
                stat["class"] = "print (image emission)"
                report["emission_prints"] = report.get("emission_prints", 0) + 1
            else:
                if img is not None and img.a is not None:
                    emission_img[m.name] = (img, st)
                emission = emission_xml(shader.inputs["Color"], st, "      ")
                stat["class"] = "emission"
                res = ('    <bsdf type="diffuse">\n      <rgb name="reflectance" value="0.02 0.02 0.02" />\n    </bsdf>\n', emission)
        else:
            stat["class"] = f"fallback:{shader.type if shader is not None else 'none'}"
        cutting[0] = False
        if emission is not None:
            stat["emission"] = [round(float(x), 4) for x in emission[1]]
        if not cutout:
            report["materials"][m.name] = {k: v for k, v in stat.items() if k != "name"}
        cache[(m.name, cutout)] = res
        return res

    def default_bsdf() -> str:
        return ('    <bsdf type="twosided">\n      <bsdf type="pplastic">\n        <float name="int_ior" value="1.5000" />\n'
                '        <rgb name="diffuse_reflectance" value="0.6 0.6 0.6" />\n        <float name="alpha" value="0.3600" />\n'
                "      </bsdf>\n    </bsdf>\n")

    # ------------------------------------------------ alpha cutouts as geometry
    def cut_quad(u: "np.ndarray", av: V, cells: int = 256):
        """A planar quad (4 welded vertices x, y, z, nx, ny, nz, u, v) with an alpha texture, as opaque rectangles: the cells of a
        grid over its uv range where the alpha is above 0.5, merged along rows. Mitsuba's ``mask`` hides a shape from the camera
        only: a shadow ray still stops at it, so the surface behind the transparent part of a decal went dark and
        sparkled. None if the quad is not planar / affine in uv."""
        P, N, UV = u[:, :3].astype(np.float64), u[:, 3:6].astype(np.float64), u[:, 6:8].astype(np.float64)
        X = np.c_[UV, np.ones(4)]
        coef, *_ = np.linalg.lstsq(X, P, rcond=None)  # position = [u, v, 1] @ coef
        if np.abs(X @ coef - P).max() > 1e-4 * max(np.abs(P).max(), 1.0):
            return None
        (u0, v0), (u1, v1) = UV.min(0), UV.max(0)
        if u1 - u0 < 1e-9 or v1 - v0 < 1e-9:
            return None
        lu = np.linalg.norm(coef[0]) * (u1 - u0)
        lv = np.linalg.norm(coef[1]) * (v1 - v0)
        nu = cells
        nv_ = int(min(512, max(16, round(cells * lv / max(lu, 1e-6)))))
        cu = u0 + (np.arange(nu) + 0.5) / nu * (u1 - u0)
        cv = v0 + (np.arange(nv_) + 0.5) / nv_ * (v1 - v0)
        gu, gv = np.meshgrid(cu, cv)
        M = np.array(av.uv or IDENT, dtype=float).reshape(3, 3)
        tc = np.stack([gu, gv, np.ones_like(gu)], axis=-1) @ M.T
        tu, tv = tc[..., 0] % 1.0, tc[..., 1] % 1.0
        h, w = av.a.shape[:2]
        alpha = av.a[np.clip(((1 - tv) * h).astype(int), 0, h - 1), np.clip((tu * w).astype(int), 0, w - 1), 0]
        covered = alpha > 0.5
        quads = []
        for j in range(nv_):
            row, i = covered[j], 0
            while i < nu:
                if row[i]:
                    k = i
                    while k + 1 < nu and row[k + 1]:
                        k += 1
                    quads.append((u0 + i / nu * (u1 - u0), u0 + (k + 1) / nu * (u1 - u0),
                                  v0 + j / nv_ * (v1 - v0), v0 + (j + 1) / nv_ * (v1 - v0)))
                    i = k + 1
                else:
                    i += 1
        if not quads:
            return None
        q = np.array(quads)
        corners = np.stack([np.stack([q[:, 0], q[:, 2]], -1), np.stack([q[:, 1], q[:, 2]], -1),
                            np.stack([q[:, 1], q[:, 3]], -1), np.stack([q[:, 0], q[:, 3]], -1)], axis=1).reshape(-1, 2)
        pos = np.c_[corners, np.ones(len(corners))] @ coef
        n0 = N.mean(0)
        n0 /= np.linalg.norm(n0)
        idx = np.arange(len(corners)).reshape(-1, 4)
        tris = np.concatenate([idx[:, [0, 1, 2]], idx[:, [0, 2, 3]]], axis=0)
        e1, e2 = pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]]
        if (np.cross(e1, e2) @ n0).mean() < 0:  # keep the quad's own facing
            tris = tris[:, ::-1]
        out = np.c_[pos, np.tile(n0, (len(corners), 1)), corners].astype(np.float32)
        return out, tris.astype(np.uint32), len(quads)

    # ------------------------------------------------ meshes
    def weld(arr: "np.ndarray"):
        void = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
        _, first, inverse = np.unique(void, return_index=True, return_inverse=True)
        return first, inverse.ravel()

    def write_ply(P, N, UV, tris) -> str:
        nv, nf = len(P), len(tris)
        head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {nv}\nproperty float x\nproperty float y\nproperty float z\n"
                f"property float nx\nproperty float ny\nproperty float nz\nproperty float u\nproperty float v\n"
                f"element face {nf}\nproperty list uchar uint vertex_indices\nend_header\n").encode()
        vert = np.concatenate([P, N, UV], axis=1).astype("<f4").tobytes()
        face = np.empty(nf, dtype=[("n", "u1"), ("i", "<u4", 3)])
        face["n"], face["i"] = 3, tris
        data = head + vert + face.tobytes()
        name = f"mesh_{hashlib.sha1(data).hexdigest()[:20]}.ply"
        if not (assets / name).exists():
            (assets / name).write_bytes(data)
        return f"assets/{name}"

    shapes_xml: list[str] = []
    used_ids: set[str] = set()

    def unique_id(base: str) -> str:
        sid, k = san(base), 1
        while sid in used_ids:
            sid, k = f"{san(base)}__{k}", k + 1
        used_ids.add(sid)
        return sid

    def note(obj_id: str, label: str, material_name: str, center: list, size: list, base_h: float, emission) -> str:
        n = {"id": obj_id, "type": "landmark", "label": label, "placement": "point", "material": san(material_name or "default"),
             "source_ref": None, "is_emitter": emission is not None, "emitter_radiance": emission, "emitter_intensity": 1.0,
             "emitter_shape": None, "emitter_polarized": False, "emitter_polarizer_angle_deg": 0.0, "emitter_pattern": None,
             "geometry": {"type": "point", "center": center, "start": None, "end": None, "height_m": None, "thickness_m": None,
                          "size_m": size, "base_height_m": base_h, "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0, "scale": None},
             "nav": {"blocks": True, "hazard_type": None, "include_hazard_mask": False, "goal_candidate": False,
                     "instruction_candidate": False}}
        return json.dumps(n, separators=(",", ":"))

    view_layer = bpy.context.view_layer
    # a big mesh drawn three or more times (linked duplicates, a tree instanced on the vertices of a ground: PH37's two tree meshes of
    # 29,000 polygons, 986 times, are 29 M of its 30 M triangles) is written once in object space as a Mitsuba shapegroup, each use an
    # ``instance`` with its matrix; mirrored copies, emitters, alpha cutouts and world-space textures stay separate shapes. A small one
    # (a 300-polygon leaf card: 7,800 instances made Dr.Jit's frozen recording take over 450 s, its tracing grows with the shape
    # count) is expanded, and Mitsuba merges meshes of one material into a few
    def mesh_key(ev_, o_):
        return (o_.data.name, tuple(s.material.name if s.material else None for s in o_.material_slots))
    use_counts: dict = collections.Counter()
    for inst_ in dg.object_instances:
        if inst_.object.type == "MESH" and (inst_.is_instance or (not inst_.object.original.hide_render and inst_.object.original.name in view_layer.objects)):
            use_counts[mesh_key(inst_.object, inst_.object.original)] += 1
    groups: dict = {}  # (mesh key, material slot) -> shapegroup id
    A4 = np.eye(4)
    A4[:3, :3], A4[:3, 3] = L, T
    # every drawn instance, not just scene.objects: trees, vines and grass are leaf / branch objects instanced on the vertices or faces of a
    # parent (PH37: 1,976 instances; scene.objects alone left them out), each with its own matrix
    for inst in dg.object_instances:
        ev = inst.object
        o = ev.original
        if ev.type != "MESH":
            continue
        if not inst.is_instance and (o.hide_render or o.name not in view_layer.objects):
            continue
        me = ev.to_mesh()
        try:
            me.calc_loop_triangles()
            nt = len(me.loop_triangles)
            if nt == 0:
                continue
            tl = np.empty(nt * 3, np.int32)
            me.loop_triangles.foreach_get("loops", tl)
            tmat = np.empty(nt, np.int32)
            me.loop_triangles.foreach_get("material_index", tmat)
            co = np.empty(len(me.vertices) * 3, np.float32)
            me.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3).astype(np.float64)
            lv = np.empty(len(me.loops), np.int32)
            me.loops.foreach_get("vertex_index", lv)
            nrm = np.empty(len(me.loops) * 3, np.float32)
            me.corner_normals.foreach_get("vector", nrm)
            nrm = nrm.reshape(-1, 3).astype(np.float64)
            uvl = me.uv_layers.active
            if uvl is not None:
                uv = np.empty(len(me.loops) * 2, np.float32)
                uvl.data.foreach_get("uv", uv)
                uv = uv.reshape(-1, 2)
            else:
                uv = np.zeros((len(me.loops), 2), np.float32)
            W = np.array(inst.matrix_world, dtype=np.float64)
            W3 = W[:3, :3]
            flip = np.linalg.det(W3) < 0
            nmat = np.linalg.inv(W3).T
            slots = [s.material for s in o.material_slots] or [None]
            report["objects"] += 1
            parts = sorted(set(tmat.tolist()))
            key = mesh_key(ev, o)
            # one shapegroup holds every material part of the mesh, one ``instance`` per use: all parts must qualify
            grp = (use_counts[key] >= 3 and len(me.polygons) >= int(cfg["instance_min_polys"]) and not o.modifiers and not flip)
            for k in parts if grp else ():
                mat_ = slots[min(k, len(slots) - 1)]
                if mat_ is None:
                    grp = False
                    break
                _, em_ = material(mat_)
                ms_ = report["materials"].get(mat_.name, {})
                if em_ is not None or ms_.get("coord") == "WORLD" or ms_.get("alpha_texture"):
                    grp = False
                    break
            obj_parts: list = []
            for k in parts:
                sel = np.nonzero(tmat == k)[0]
                idx = tl.reshape(-1, 3)[sel]
                if flip:
                    idx = idx[:, ::-1]
                flat = idx.ravel()
                P = (co[lv[flat]] @ W3.T + W[:3, 3]) @ L.T + T
                N = nrm[flat] @ nmat.T @ L.T
                N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
                mat = slots[min(k, len(slots) - 1)]
                bsdf, emission = material(mat)
                mstat = report["materials"].get(mat.name, {}) if mat else {}
                coord = mstat.get("coord", "UV")
                if coord in ("OBJECT", "GENERATED", "WORLD") and mstat.get("proj"):
                    # the image is looked up by the object's own coordinates: the Mapping applied here, then FLAT (x, y) or BOX (the axis
                    # pair and signs Cycles uses for the triangle's dominant normal axis, below)
                    pr = mstat["proj"]
                    base = co[lv[idx]]  # (triangles, 3 corners, xyz)
                    if coord == "WORLD":  # Blender's world space (the Geometry node's Position), before the Mitsuba frame
                        base = base @ W3.T + W[:3, 3]
                    if coord == "GENERATED":  # the object's bounding box, 0..1
                        lo_, hi_ = co.min(0), co.max(0)
                        base = (base - lo_) / np.maximum(hi_ - lo_, 1e-9)
                    X = base @ np.array(pr["M"]).reshape(3, 3).T + np.array(pr["loc"])
                    if pr["projection"] == "BOX":
                        # measured on a cube in Cycles (6 faces, a u = R / v = G gradient image): seen from outside u grows to the right,
                        #   +x: u = +y   -x: u = -y   +y: u = -x   -y: u = +x   +z: u = -y   -z: u = +y;   v = z on the x and y faces, x on z
                        fn = nrm[idx].mean(1)
                        axis = np.abs(fn).argmax(1)
                        s = np.sign(np.take_along_axis(fn, axis[:, None], 1))[:, 0]
                        s[s == 0] = 1.0
                        ax_ = axis[:, None]
                        uu = np.where(ax_ == 0, s[:, None] * X[..., 1], np.where(ax_ == 1, -s[:, None] * X[..., 0], -s[:, None] * X[..., 1]))
                        vv = np.where(ax_ == 2, X[..., 0], X[..., 2])
                        uvs = np.stack([uu, vv], axis=-1).reshape(-1, 2)
                    else:
                        uvs = X[..., :2].reshape(-1, 2)
                else:
                    uvs = uv[flat].astype(np.float64)
                if grp:
                    loc = np.concatenate([co[lv[flat]], nrm[flat] / np.maximum(np.linalg.norm(nrm[flat], axis=1, keepdims=True), 1e-12),
                                          uvs], axis=1).astype(np.float32)
                    f_, inv_ = weld(loc)
                    ul = loc[f_]
                    obj_parts.append((write_ply(ul[:, :3], ul[:, 3:6], ul[:, 6:8], inv_.reshape(-1, 3)), bsdf))
                    continue
                data = np.concatenate([P, N, uvs], axis=1).astype(np.float32)
                first, inverse = weld(data)
                tris = inverse.reshape(-1, 3)
                u = data[first]
                if len(u) == 4 and mat is not None and report["materials"].get(mat.name, {}).get("alpha_texture") and mat.name in alpha_of:
                    cut = cut_quad(u, alpha_of[mat.name])
                    if cut is not None:
                        u, tris, n_quads = cut
                        bsdf, emission = material(mat, cutout=True)
                        report["cutouts"] = report.get("cutouts", 0) + 1
                        report["cutout_quads"] = report.get("cutout_quads", 0) + n_quads
                digits = None
                if emission and mat is not None and mat.name in emission_img and len(u) == 4:
                    # a lit sign (the room numerals, strength 14): the plate stays dark, the bright texels glow as opaque rectangles cut
                    # from the image (a textured emitter breaks Dr.Jit's frozen recording), on both sides as Cycles' Emission is
                    pic, strength = emission_img[mat.name]
                    lum = pic.a.mean(-1)
                    on = lum > 0.5 * lum.max()
                    cut = cut_quad(u, V(a=np.repeat(on[..., None].astype(np.float32), 3, axis=-1), uv=pic.uv))
                    if cut is not None:
                        digits = (cut[0], cut[1], [float(x) * strength for x in pic.a[on].mean(0)])
                        bsdf, emission = '    <bsdf type="diffuse">\n      <rgb name="reflectance" value="0.02 0.02 0.02" />\n    </bsdf>\n', None
                        report["glowing_cutouts"] = report.get("glowing_cutouts", 0) + 1
                path = write_ply(u[:, :3], u[:, 3:6], u[:, 6:8], tris)
                sid = unique_id(o.name if len(parts) == 1 else f"{o.name}__part_{k}_")
                lo, hi = u[:, :3].min(0), u[:, :3].max(0)
                note_xml = note(sid, o.name, mat.name if mat else "default", [float((lo[0] + hi[0]) / 2), float((lo[2] + hi[2]) / 2)],
                                [float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2])], float(lo[1]),
                                emission[1] if emission else None)
                emit = f'    <emitter type="area">\n{emission[0]}    </emitter>\n' if emission else ""
                shapes_xml.append(f'\n  <!--opticalnav-obj:{note_xml}-->\n  <shape type="ply" id="{sid}">\n'
                                  f'    <string name="filename" value="{path}" />\n{bsdf}{emit}  </shape>')
                if digits is not None:
                    gu, gt, grad = digits
                    nn = gu[:, 3:6].mean(0)
                    nn /= max(np.linalg.norm(nn), 1e-9)
                    gem = f'    <emitter type="area">\n      <rgb name="radiance" value="{rgb(grad)}" />\n    </emitter>\n'
                    dark = '    <bsdf type="diffuse">\n      <rgb name="reflectance" value="0.02 0.02 0.02" />\n    </bsdf>\n'
                    for side, sgn in (("a", 1.0), ("b", -1.0)):
                        sid3 = unique_id(f"{sid}__glow_{side}")
                        pos3 = gu[:, :3] + sgn * 0.0005 * nn
                        pth = write_ply(pos3, sgn * gu[:, 3:6], gu[:, 6:8], np.ascontiguousarray(gt if sgn > 0 else gt[:, ::-1]))
                        note3 = note(sid3, o.name, mat.name, [float((lo[0] + hi[0]) / 2), float((lo[2] + hi[2]) / 2)],
                                     [float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2])], float(lo[1]), grad)
                        shapes_xml.append(f'\n  <!--opticalnav-obj:{note3}-->\n  <shape type="ply" id="{sid3}">\n'
                                          f'    <string name="filename" value="{pth}" />\n{dark}{gem}  </shape>')
                elif emission and float((hi - lo).min()) < 0.03:
                    # Cycles' Emission is two-sided (checked: a plane reads 5.0 from both sides), Mitsuba's area emitter is not: a thin
                    # emitter (a painting, a sign, a screen) gets a copy that faces the other way
                    sid2 = unique_id(f"{sid}__back")
                    # displaced 0.5 mm along its own facing: coplanar with the original, a ray from that side hit the original first
                    # (which emits the other way) and the painting stayed black
                    nback = -u[:, 3:6].mean(0)
                    nback /= max(np.linalg.norm(nback), 1e-9)
                    path2 = write_ply(u[:, :3] + 0.0005 * nback, -u[:, 3:6], u[:, 6:8], np.ascontiguousarray(tris[:, ::-1]))
                    note2 = note(sid2, o.name, mat.name if mat else "default", [float((lo[0] + hi[0]) / 2), float((lo[2] + hi[2]) / 2)],
                                 [float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2])], float(lo[1]), emission[1])
                    shapes_xml.append(f'\n  <!--opticalnav-obj:{note2}-->\n  <shape type="ply" id="{sid2}">\n'
                                      f'    <string name="filename" value="{path2}" />\n{bsdf}{emit}  </shape>')
                    report["two_sided_emitters"] = report.get("two_sided_emitters", 0) + 1
                report["shapes"] += 1
                report["triangles"] += len(tris)
            if grp and obj_parts:
                if key not in groups:
                    gid = unique_id(f"group_{o.data.name}")
                    inner = "".join(f'    <shape type="ply">\n      <string name="filename" value="{gp}" />\n{gb}    </shape>\n' for gp, gb in obj_parts)
                    shapes_xml.append(f'\n  <shape type="shapegroup" id="{gid}">\n{inner}  </shape>')
                    groups[key] = gid
                    report["shapegroups"] = report.get("shapegroups", 0) + 1
                mw = A4 @ W
                sid = unique_id(f"{o.name}__inst")
                shapes_xml.append(f'\n  <shape type="instance" id="{sid}">\n    <ref id="{groups[key]}" />\n'
                                  f'    <transform name="to_world"><matrix value="{" ".join(f(v) for v in mw.ravel())}" /></transform>\n  </shape>')
                report["instances"] = report.get("instances", 0) + 1
                report["shapes"] += 1
        finally:
            ev.to_mesh_clear()

    # ------------------------------------------------ lights (the importer's boxes: the server reads them as before)
    lights_xml: list[str] = []
    for i, o in enumerate(o for o in scene.objects if o.type == "LIGHT" and not o.hide_render):
        d = o.data
        W = np.array(o.matrix_world, dtype=np.float64)
        pos = L @ W[:3, 3] + T  # Mitsuba
        direction = L @ (W[:3, :3] @ np.array([0.0, 0.0, -1.0]))
        direction /= np.linalg.norm(direction)
        col = [float(c) for c in d.color]
        info = {"type": d.type, "energy": float(d.energy), "color": col,
                "position": [float(pos[0]), float(pos[2]), float(pos[1])],  # plan x, y, height
                "direction": [float(direction[0]), float(direction[2]), float(direction[1])],
                "spot_size_deg": math.degrees(d.spot_size) if d.type == "SPOT" else None,
                "spot_blend": float(d.spot_blend) if d.type == "SPOT" else None,
                "radius_m": float(getattr(d, "shadow_soft_size", 0.0)) if d.type in ("SPOT", "POINT") else None,
                "area_shape": d.shape if d.type == "AREA" else None,
                "area_size_m": [float(d.size), float(d.size_y if d.shape in ("RECTANGLE", "ELLIPSE") else d.size)] if d.type == "AREA" else None,
                "spread_deg": math.degrees(d.spread) if d.type == "AREA" else None,
                "sun_angle_deg": math.degrees(d.angle) if d.type == "SUN" else None,
                "use_shadow": bool(getattr(d, "use_shadow", True))}
        radiance = [round(c * (1.0 + IMPORTER_WATTS * float(d.energy)), 5) for c in col]
        # a Blender light is not seen by the camera when its Ray Visibility > Camera is off (every light of PH19 and PH33): the
        # server's ``hidden`` prefix lets camera rays through such a box, which lights the room without a white square on the ceiling
        sid = unique_id(f"light_hidden_{i}_{o.name}" if not o.visible_camera else f"light_{i}_{o.name}")
        n = {"id": sid, "type": "landmark", "label": f"light:{o.name}", "placement": "point", "material": "infinigen_default",
             "source_ref": None, "is_emitter": True, "emitter_radiance": radiance, "emitter_intensity": 1.0,
             "emitter_shape": "ceiling_panel", "emitter_polarized": False, "emitter_polarizer_angle_deg": 0.0, "emitter_pattern": None,
             "geometry": {"type": "point", "center": [float(pos[0]), float(pos[2])], "start": None, "end": None, "height_m": None,
                          "thickness_m": None, "size_m": [0.6, 0.08, 0.6], "base_height_m": float(pos[1]), "yaw_deg": 0.0,
                          "pitch_deg": 0.0, "roll_deg": 0.0, "scale": None},
             "nav": {"blocks": False, "hazard_type": None, "include_hazard_mask": False, "goal_candidate": False,
                     "instruction_candidate": False}, "metadata": {"light": info}}
        lights_xml.append(f'\n  <!--opticalnav-obj:{json.dumps(n, separators=(",", ":"))}-->\n  <shape type="rectangle" id="{sid}">\n'
                          f'    <transform name="to_world">\n      <scale x="0.300000" y="0.300000" />\n      <rotate x="1" angle="90" />\n'
                          f'      <translate x="{f(pos[0])}" y="{f(pos[1] + 0.01)}" z="{f(pos[2])}" />\n    </transform>\n'
                          f'    <emitter type="area">\n      <rgb name="radiance" value="{rgb(radiance)}" />\n    </emitter>\n  </shape>')
        report["lights"] += 1

    import xml.etree.ElementTree as ET
    # the template's integrator, sensor and constant sky as they are; its notes, materials and shapes are not carried over
    keep = []
    for child in ET.parse(cfg["template_xml"]).getroot():
        if child.tag in ("integrator", "sensor", "emitter"):
            child.tail = None
            keep.append("  " + ET.tostring(child, encoding="unicode").strip())
    # suns: a directional emitter (Cycles' sun strength is its irradiance on a surface facing it: measured 3.0 for 3.0)
    for o in scene.objects:
        if o.type == "LIGHT" and o.data.type == "SUN" and not o.hide_render:
            travel = L @ (np.array(o.matrix_world)[:3, :3] @ np.array([0.0, 0.0, -1.0]))
            travel /= np.linalg.norm(travel)
            col = [float(c) * float(o.data.energy) for c in o.data.color]
            keep.append(f'  <emitter type="directional">\n    <vector name="direction" x="{f(travel[0])}" y="{f(travel[1])}" z="{f(travel[2])}" />\n'
                        f'    <rgb name="irradiance" value="{rgb(col)}" />\n  </emitter>')
            report["suns"] = report.get("suns", 0) + 1
    xml = '<scene version="3.0.0">\n' + "\n".join(keep) + "\n" + "".join(shapes_xml) + "".join(lights_xml) + "\n</scene>\n"
    (out / "base.xml").write_text(xml)
    # the world (HDRI, sky texture, a gradient with noise ...) as one equirectangular EXR, rendered by Cycles with everything else hidden:
    # the camera looks along -x with +z up, which is Blender's own world lookup (u = atan2(y, -x) / 2 pi + 0.5)
    if cfg.get("sky", True) and scene.world is not None:
        (out / "sky").mkdir(exist_ok=True)
        cam = bpy.data.objects.new("OPTICALNAV_WORLD_CAM", bpy.data.cameras.new("OPTICALNAV_WORLD_CAM"))
        scene.collection.objects.link(cam)
        cam.data.type = "PANO"
        cam.data.panorama_type = "EQUIRECTANGULAR"
        from mathutils import Matrix
        cam.matrix_world = Matrix([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        scene.camera = cam
        for o in scene.objects:
            if o is not cam:
                o.hide_render = True
        r = scene.render
        r.engine, r.resolution_x, r.resolution_y, r.resolution_percentage = "CYCLES", 1024, 512, 100
        scene.cycles.device, scene.cycles.samples, scene.cycles.use_denoising = "CPU", 24, False
        scene.cycles.max_bounces = 0
        r.film_transparent, r.use_compositing, r.use_sequencer = False, False, False
        r.image_settings.file_format, r.image_settings.color_depth, r.image_settings.exr_codec = "OPEN_EXR", "32", "NONE"
        scene.view_settings.view_transform = "Standard"
        r.filepath, r.use_file_extension = str(out / "sky" / "world.exr"), False
        bpy.ops.render.render(write_still=True)
        report["sky"] = "sky/world.exr"
    report["seconds"] = round(time.perf_counter() - t_start, 1)
    (out / "convert_report.json").write_text(json.dumps(report, indent=1))
    print(f"[blend_to_pack] {report['objects']} objects -> {report['shapes']} shapes, {report['triangles']} triangles, "
          f"{len(report['materials'])} materials, {report['lights']} lights in {report['seconds']} s", flush=True)


# ---------------------------------------------------------------- driver
def postprocess_assets(assets: Path, cap: int) -> None:
    """16-bit PNGs for the roughness / metallic maps Blender saved as .npy; textures over ``cap`` px shrunk."""
    import numpy as np
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    for npy in assets.glob("*.npy"):
        arr = np.load(npy)
        im = Image.fromarray(arr)
        if arr.dtype == np.uint16 and max(im.size) > 1024:  # scalar maps (_s): smooth
            s = 1024 / max(im.size)
            small = Image.fromarray(arr.astype(np.float32), "F").resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR)
            im = Image.fromarray(np.clip(np.asarray(small) + 0.5, 0, 65535).astype(np.uint16))
        if npy.stem.endswith("_c"):
            im.save(npy.with_suffix(".jpg"), quality=95, subsampling=0)
        else:
            im.save(npy.with_suffix(".png"))
        npy.unlink()
    for p in list(assets.glob("tex_*.png")) + list(assets.glob("tex_*.jp*g")):
        with Image.open(p) as im:
            if max(im.size) <= cap:
                continue
            s = cap / max(im.size)
            small = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
        small.save(p)


def main() -> int:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--scene", help="a pack scene to take the .blend, origin offset, camera rig and spawns from (without it: --blend and "
                                    "--scene-id, the offset from the scene's bounding box, spawns from the .blend's cameras)")
    ap.add_argument("--template", default="sim_v2a_REF_PH19", help="the pack scene whose camera rig, Stokes basis and integrator a scene "
                                                                  "without --scene takes")
    ap.add_argument("--scene-id", help="the new scene (default: sim_v3_ + the scene's id without sim_v2a_)")
    ap.add_argument("--blend", type=Path)
    ap.add_argument("--origin-offset", type=float, nargs=3)
    ap.add_argument("--max-texture", type=int, default=4096)
    ap.add_argument("--instance-min-polys", type=int, default=2000,
                    help="a mesh drawn 3+ times becomes a shapegroup with instances only from this many polygons (smaller ones are "
                         "expanded: fewer shapes for Dr.Jit to trace)")
    ap.add_argument("--link-blend", action="store_true",
                    help="development: scene.json points at the source .blend instead of a packed copy (seconds, not minutes, for a big one)")
    ap.add_argument("--sky-rotation", type=float, default=270.0,
                    help="degrees the world map is turned about the vertical when the server looks it up (Blender's and Mitsuba's "
                         "equirectangular conventions differ by a quarter turn)")
    ap.add_argument("--print-max-strength", type=float, default=5.0,
                    help="an Emission shader coloured by an image, up to this strength, becomes a lit print (image albedo, polarizing) "
                         "instead of an emitter (default 5: banners 1, wayfinding signs 3; the room numerals, 14, glow)")
    args = ap.parse_args()
    import attach_blend
    from opticalnav_sim import cycles_backend as cb

    pack = args.pack.resolve()
    own_template = bool(args.scene)  # the scene's own pack folder: its blend, offset, spawns (else the generic template's rig only)
    template = pack / "scenes" / (args.scene or args.template)
    meta = json.loads((template / "scene.json").read_text())
    if own_template:
        blend = args.blend or (template / meta["blend"]["file"] if meta.get("blend") else attach_blend.find_source(args.scene, attach_blend.TSV))
        offset = args.origin_offset or (meta.get("blend") or {}).get("origin_offset")
    else:
        blend, offset = args.blend, args.origin_offset  # offset None: from the bounding box
    if blend is None or (own_template and offset is None) or (not own_template and not args.scene_id):
        raise SystemExit("no .blend / origin offset (--scene S, or --blend FILE --scene-id ID)")
    scene_id = args.scene_id or "sim_v3_" + args.scene.removeprefix("sim_v2a_")
    out = pack / "scenes" / scene_id
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"blend": str(blend), "out": str(out), "origin_offset": list(offset) if offset else None, "max_texture": args.max_texture,
           "print_max_strength": args.print_max_strength, "instance_min_polys": args.instance_min_polys,
           "template_xml": str(template / "base.xml")}
    env = dict(os.environ, OPTICALNAV_B2M_INSIDE="1", OPTICALNAV_B2M_CFG=json.dumps(cfg),
               LD_LIBRARY_PATH=f"{cb.DRIVER_DIRS}:{cb.BLENDER.parent / 'lib'}:{cb.lib_dir()}")
    t0 = time.time()
    for stale in ("base.xml", "convert_report.json"):
        (out / stale).unlink(missing_ok=True)  # a failed run must not look like the last one's
    for old_asset in (out / "assets").glob("*"):
        old_asset.unlink()  # content-addressed names: what this run does not write is not wanted
    (out / "sky" / "world.exr").unlink(missing_ok=True)
    r = subprocess.run([str(cb.BLENDER), "-b", "--python-exit-code", "1", "--python", str(Path(__file__).resolve())], env=env,
                       capture_output=True, text=True)
    print("\n".join(l for l in r.stdout.splitlines() if l.startswith("[blend_to_pack]")))
    if r.returncode or not (out / "base.xml").is_file():
        print((r.stdout + r.stderr)[-2500:])
        return 1
    postprocess_assets(out / "assets", args.max_texture)
    # scene.json: the template's, with this scene's ids
    report = json.loads((out / "convert_report.json").read_text())
    offset = report["origin_offset"]
    meta["scan"] = scene_id
    meta["variants"] = {"base": {**meta["variants"]["base"], "xml": "base.xml", "protocol": "blend_direct_v1"}}
    meta["converter"] = {"tool": "blend_to_pack", "source": str(blend), "seconds": round(time.time() - t0, 1), "report": report}
    for k in ("flat_colors", "flat_normalmaps", "gray_textures", "asset_compaction", "emissive_parts", "staged", "spawn_before", "blend", "sky"):
        meta.pop(k, None)
    if report.get("sky"):
        meta["sky"] = {"file": report["sky"], "gain": 1.0, "rotation_deg": args.sky_rotation}
    if not own_template:  # the authors' camera positions are the spawns (up to 12), nothing of the template's scene
        cams = report["cameras"][:12] or [{"position": [0.0, 0.0, 1.5], "heading_deg": 0.0, "name": "origin"}]
        meta["spawn"] = [{"position": c["position"], "heading_deg": c["heading_deg"], "clearance_m": 0.0, "note": c["name"]} for c in cams]
        meta["graph"], meta["episodes"], meta["reference_views"] = None, {}, []
    elif meta.get("spawn"):
        pass  # the template's own spawns
    (out / "scene.json").write_text(json.dumps(meta, indent=1))
    # the .blend travels with the scene (images packed): attach_blend, or the template's packed copy when it is the same file
    if args.link_blend:
        m2 = json.loads((out / "scene.json").read_text())
        m2["blend"] = {"file": str(Path(blend).resolve()), "origin_offset": list(offset), "linked": True}
        (out / "scene.json").write_text(json.dumps(m2, indent=1))
    elif own_template and (template / "blend" / "scene.blend").is_file() and template.name != scene_id:
        (out / "blend").mkdir(exist_ok=True)
        (out / "blend" / "scene.blend").write_bytes((template / "blend" / "scene.blend").read_bytes())
        m2 = json.loads((out / "scene.json").read_text())
        m2["blend"] = {"file": "blend/scene.blend", "origin_offset": list(offset)}
        (out / "scene.json").write_text(json.dumps(m2, indent=1))
    else:
        print(attach_blend.attach(pack, scene_id, blend, attach_blend.TSV, attach_blend.IMPORTS, False, offset=offset))
    conn = pack / "connectivity"
    if own_template and (conn / f"{args.scene}_connectivity.json").is_file():
        (conn / f"{scene_id}_connectivity.json").write_text((conn / f"{args.scene}_connectivity.json").read_text().replace(args.scene, scene_id))
    print(f"{scene_id}: done in {time.time() - t0:.0f} s -> {out}")
    return 0


if __name__ == "__main__":
    if INSIDE:
        run_inside(json.loads(os.environ["OPTICALNAV_B2M_CFG"]))
    else:
        raise SystemExit(main())
