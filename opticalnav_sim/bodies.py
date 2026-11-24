"""Rigid bodies a view moves: robots and the equipment on them (hw.Actor), drawn in a resident scene.

A model is a set of links, each a few primitives in the link's own frame (ROS axes, as in a URDF). In the scene a link
of a model is one ``shapegroup`` (its acceleration structure is built once) and every body of that model draws it
through an ``instance``. Moving a body writes its instances' ``to_world``: only the scene's top-level structure is
rebuilt, no mesh is, and a frozen render replays (measured: 15 links moved in 3 ms a frame on a 437k triangle scene,
no recording, the image equal to a scene loaded with the links there).

    descriptor = {"links": {"chassis": [{"shape": "box", "size": [x, y, z], "origin": 4x4, "material": [kind, rgb, roughness]},
                                        {"shape": "cylinder", "size": [radius, length], ...},   # axis: the link's y
                                        {"shape": "profile", "size": [side, length], ...},      # T-slot extrusion along z
                                        {"shape": "mesh", "mesh": "/abs/part.ply", ...}], ...}}

No Mitsuba here: the scene text and the PLY files only.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

BODY_PREFIX = "body_"
GROUP_PREFIX = "bodygrp_"
CACHE = Path.home() / ".cache" / "opticalnav_sim" / "bodies"
MATERIALS = ("plastic", "metal", "rubber", "aluminium")


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_")


def instance_id(body: str, link: str) -> str:
    """The scene id of a body's link: what a view's pose is written to."""
    return f"{BODY_PREFIX}{safe(body)}__{safe(link)}"


def model_key(descriptor: dict) -> str:
    return hashlib.sha1(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()[:12]


def box(size) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = (float(s) / 2.0 for s in size)
    v = np.array([[sx * x, sy * y, sz * z] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float32)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4],
                  [1, 5, 7], [1, 7, 3]], np.int32)
    return v, f


def cylinder(radius: float, length: float, segments: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """A closed cylinder about the y axis (a wheel on its axle), centred."""
    a = np.arange(segments) * (2.0 * math.pi / segments)
    ring = np.stack([np.cos(a) * radius, np.zeros(segments), np.sin(a) * radius], 1)
    h = np.array([0.0, float(length) / 2.0, 0.0])
    # side: its own rings; caps: rings of their own (a shared vertex would round the rim when normals are smoothed)
    v = np.concatenate([ring - h, ring + h, ring - h, ring + h, [-h], [h]]).astype(np.float32)
    n, faces = segments, []
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, n + j, j], [i, n + i, n + j], [4 * n, 2 * n + i, 2 * n + j], [4 * n + 1, 3 * n + j, 3 * n + i]]
    return v, np.array(faces, np.int32)


def profile(side: float, length: float, slot: float = 0.008, lip: float = 0.002, web: float = 0.003) -> tuple[np.ndarray, np.ndarray]:
    """A T-slot aluminium extrusion of square section ``side``, ``length`` long along z, centred: a core, a web from it
    to each corner, and at each corner two lips that leave a ``slot`` wide groove in the middle of every face, wider
    behind the lips (the T). Built of boxes that overlap inside: from outside, the section's outline and the grooves
    running along the faces."""
    h = side / 2.0
    core = max(side / 3.0, slot + 2.0 * web)
    span = h - slot / 2.0                         # a lip runs from the groove's edge to the corner
    parts = [(np.eye(3), (0.0, 0.0), (core, core))]
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            a = math.atan2(sy, sx)                # the web: from the core to the corner
            rot = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
            reach = math.hypot(h - lip, h - lip)
            parts.append((rot, (sx * (h - lip) / 2.0, sy * (h - lip) / 2.0), (reach, web)))
            parts.append((np.eye(3), (sx * (h - span / 2.0), sy * (h - lip / 2.0)), (span, lip)))   # along the y faces
            parts.append((np.eye(3), (sx * (h - lip / 2.0), sy * (h - span / 2.0)), (lip, span)))   # along the x faces
            parts.append((np.eye(3), (sx * (h - 1.5 * lip), sy * (h - 1.5 * lip)), (3.0 * lip, 3.0 * lip)))  # the corner
    vs, fs, n = [], [], 0
    for rot, (cx, cy), (dx, dy) in parts:
        v, f = box((dx, dy, length))
        vs.append(v @ rot.T.astype(np.float32) + np.array([cx, cy, 0.0], np.float32))
        fs.append(f + n)
        n += len(v)
    return np.concatenate(vs), np.concatenate(fs)


def write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(vertices)}\nproperty float x\nproperty float y\n"
            f"property float z\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
    rows = np.zeros(len(faces), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    rows["n"], rows["i"] = 3, faces
    path.write_bytes(head.encode() + np.asarray(vertices, "<f4").tobytes() + rows.tobytes())


def bsdf_xml(material, indent: str = "      ") -> str:
    """The converter's classes (tools/blend_to_pack.py): a dielectric coat over a diffuse base (pplastic), or a
    conductor with the colour as its reflectance. Roughness is Blender's (alpha = roughness squared)."""
    kind, rgb, rough = material[0], [min(max(float(c), 0.0), 1.0) for c in material[1]], float(material[2])
    if kind not in MATERIALS:
        raise ValueError(f"material kind {kind!r}: one of {MATERIALS}")
    alpha = max(rough * rough, 1e-3)
    if kind == "aluminium":
        # anodized aluminium (extrusions, machined parts): the metal (Mitsuba's measured "Al") under a clear, slightly
        # scattering oxide layer: satin and light, not a mirror. As the converter's part-metal materials: a blend.
        return (f'{indent}<bsdf type="blendbsdf">\n{indent}  <float name="weight" value="0.55" />\n'
                f'{indent}  <bsdf type="pplastic">\n{indent}    <float name="int_ior" value="1.5000" />\n'
                f'{indent}    <rgb name="diffuse_reflectance" value="{rgb[0]:.5f} {rgb[1]:.5f} {rgb[2]:.5f}" />\n'
                f'{indent}    <float name="alpha" value="{alpha:.5f}" />\n{indent}  </bsdf>\n'
                f'{indent}  <bsdf type="roughconductor">\n{indent}    <string name="material" value="Al" />\n'
                f'{indent}    <float name="alpha" value="{alpha:.5f}" />\n{indent}  </bsdf>\n{indent}</bsdf>\n')
    if kind == "metal":
        r = [min(c, 0.995) for c in rgb]
        eta, k = [(1 - c) / (1 + c) for c in r], [2 * math.sqrt(c) / math.sqrt(1 - c) for c in r]
        return (f'{indent}<bsdf type="roughconductor">\n{indent}  <rgb name="eta" value="{eta[0]:.5f} {eta[1]:.5f} {eta[2]:.5f}" />\n'
                f'{indent}  <rgb name="k" value="{k[0]:.5f} {k[1]:.5f} {k[2]:.5f}" />\n'
                f'{indent}  <float name="alpha" value="{alpha:.5f}" />\n{indent}</bsdf>\n')
    return (f'{indent}<bsdf type="pplastic">\n{indent}  <float name="int_ior" value="1.5000" />\n'
            f'{indent}  <rgb name="diffuse_reflectance" value="{rgb[0]:.5f} {rgb[1]:.5f} {rgb[2]:.5f}" />\n'
            f'{indent}  <float name="alpha" value="{alpha:.5f}" />\n{indent}</bsdf>\n')


def build_model(name: str, descriptor: dict, cache: Path | None = None) -> dict:
    """Write a model's primitives as PLY files (their origins applied) and return ``{"name", "key", "links": {link:
    shapegroup XML}}``. Files are named by the descriptor's hash: a changed model is another model."""
    key = model_key(descriptor)
    folder = (cache or CACHE) / f"{safe(name)}-{key}"
    folder.mkdir(parents=True, exist_ok=True)
    links = {}
    for link, visuals in descriptor["links"].items():
        shapes = ""
        for i, vis in enumerate(visuals):
            origin = np.asarray(vis.get("origin") or np.eye(4), dtype=np.float64).reshape(4, 4)
            mat = bsdf_xml(vis.get("material") or ("plastic", (0.5, 0.5, 0.5), 0.4))
            if vis["shape"] == "mesh":  # the file as it is, placed by its origin
                src = Path(vis["mesh"])
                if src.suffix.lower() not in (".ply", ".obj") or not src.is_file():
                    raise ValueError(f"{name}.{link}: mesh {src} is not a .ply or .obj file on the render server")
                matrix = " ".join(f"{x:.8g}" for x in origin.ravel())
                shapes += (f'    <shape type="{src.suffix[1:].lower()}">\n      <string name="filename" value="{src}" />\n'
                           f'      <transform name="to_world"><matrix value="{matrix}" /></transform>\n{mat}    </shape>\n')
                continue
            if vis["shape"] == "box":
                v, f = box(vis["size"])
            elif vis["shape"] == "cylinder":
                v, f = cylinder(*vis["size"])
            elif vis["shape"] == "profile":  # [side, length]: a T-slot extrusion along the origin's z
                v, f = profile(*vis["size"])
            else:
                raise ValueError(f"{name}.{link}: shape {vis['shape']!r} (box, cylinder, profile or mesh)")
            v = (np.c_[v, np.ones(len(v))] @ origin.T)[:, :3]
            file = folder / f"{safe(link)}_{i}.ply"
            if not file.is_file():
                write_ply(file, v, f)
            shapes += f'    <shape type="ply">\n      <string name="filename" value="{file}" />\n{mat}    </shape>\n'
        links[link] = f'  <shape type="shapegroup" id="{GROUP_PREFIX}{safe(name)}_{key}__{safe(link)}">\n{shapes}  </shape>\n'
    return {"name": name, "key": key, "links": links}


def scene_xml(models: dict, bodies) -> str:
    """Scene text for ``bodies`` ((body id, model name) pairs): each model's shapegroups once, an instance per body and
    link. Instances start far below the scene; the first view places them."""
    out, done = "", set()
    for body, model_name in bodies:
        model = models[model_name]
        for link, group in model["links"].items():
            gid = f"{GROUP_PREFIX}{safe(model_name)}_{model['key']}__{safe(link)}"
            if gid not in done:
                out += group
                done.add(gid)
            out += (f'  <shape type="instance" id="{instance_id(body, link)}">\n    <ref id="{gid}" />\n'
                    f'    <transform name="to_world"><translate x="0" y="-1000" z="0" /></transform>\n  </shape>\n')
    return out
