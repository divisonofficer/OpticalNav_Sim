"""A scene's ground, from its own files: where a robot can stand, what blocks it, and how much grip each surface gives.

    gm = groundmap.build(scene_dir, floor_z=0.18)          # or tools/build_ground_map.py
    ground = gm.ground()                                    # hw.ground.Ground with heights, materials and friction

No renderer: the scene's meshes (the PLY files its XML names) are sampled on a grid seen from above.

* ``heights``: the highest surface in each cell between half a metre under the floor and the robot's top. A cell
  whose height is more than a step above its neighbours' floor blocks (hw.ground.Ground.blocked).
* ``material``: the material of the highest surface a wheel could stand on (facing up) in each cell.
* friction: each material's coefficient, from what the scene says about it. Its name gives the kind of surface
  (asphalt, tile, grass ...) and its state when the author named one (wet, polished, icy); its PBR parameters give the
  rest: a smooth dielectric (low roughness) is slippery as a polished or wet floor is, a metal plate grips less than
  stone. Optical roughness is not a friction measurement: these are informed defaults, listed with their reasons in
  ``GroundMap.friction`` and overridden per material or per region (hw.ground.Ground).

Frames: the scene files are Mitsuba's (X, Y up, Z); the map is the simulator's world (x = X, y = -Z, z = Y).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# The kind of surface a name says, with the dry coefficient of a rubber tyre on it (typical textbook values).
SURFACES = (("asphalt", 0.80), ("tarmac", 0.80), ("concrete", 0.75), ("cement", 0.75), ("pavers", 0.70), ("paving", 0.70),
            ("pavement", 0.70), ("pave", 0.70), ("brick", 0.70), ("cobble", 0.65), ("granite", 0.55), ("terrazzo", 0.50),
            ("marble", 0.45), ("stone", 0.60), ("tile", 0.50), ("epoxy", 0.60), ("vinyl", 0.55), ("linoleum", 0.55),
            ("laminate", 0.45), ("parquet", 0.50), ("wood", 0.50), ("carpet", 0.90), ("rug", 0.85), ("rubber", 0.95),
            ("gravel", 0.55), ("grass", 0.45), ("lawn", 0.45), ("mulch", 0.50), ("soil", 0.50), ("earth", 0.50),
            ("dirt", 0.50), ("mud", 0.30), ("sand", 0.40), ("steel", 0.45), ("metal", 0.45), ("grating", 0.55),
            ("glass", 0.40), ("paint", 0.55), ("floor", 0.60), ("ground", 0.60), ("road", 0.75), ("apron", 0.75))
# The state a name says: a factor on the dry coefficient, or (ice, snow) the coefficient itself.
STATES = (("ice", None, 0.10), ("icy", None, 0.10), ("snow", None, 0.20), ("oil", 0.30, None), ("puddle", 0.45, None),
          ("wet", 0.55, None), ("damp", 0.75, None), ("polish", 0.75, None), ("gloss", 0.80, None), ("wax", 0.70, None))
DEFAULT_MU = 0.60


def _words(name: str) -> str:
    return re.sub(r"[^a-z]+", " ", str(name).lower())


def material_friction(name: str, roughness: float | None = None, metallic: float | None = None) -> tuple[float, str]:
    """(friction coefficient, how it was reached) of a scene material.

    The name's surface kind sets the dry value (else the PBR class: a metal 0.45, anything else 0.60). A state in the
    name scales it. Without a named state the roughness speaks: under 0.35 the surface is taken as smoother than its
    kind's usual finish, down to 0.6 of the dry value at a mirror finish (a polished or wet-look floor)."""
    words = _words(name)
    kind = next(((k, mu) for k, mu in SURFACES if re.search(rf"\b{k}", words)), None)  # "corrugated" is not a rug
    if kind is not None:
        mu, why = kind[1], f"{kind[0]} {kind[1]:.2f}"
    elif metallic is not None and metallic >= 0.5:
        mu, why = 0.45, "metallic 0.45"
    else:
        mu, why = DEFAULT_MU, f"default {DEFAULT_MU:.2f}"
    state = next(((s, f, absolute) for s, f, absolute in STATES if re.search(rf"\b{s}", words)), None)
    if state is not None:
        if state[2] is not None:
            return state[2], f"{state[0]} {state[2]:.2f}"
        return round(mu * state[1], 3), f"{why} x {state[0]} {state[1]:.2f}"
    if roughness is not None and roughness < 0.35:
        factor = 0.6 + 0.4 * max(0.0, float(roughness)) / 0.35
        return round(mu * factor, 3), f"{why} x smooth (roughness {roughness:.2f}) {factor:.2f}"
    return mu, why


_SHAPE = re.compile(r'<!--opticalnav-obj:(\{.*?\})-->\s*<shape type="(\w+)" id="([^"]*)">(.*?)</shape>', re.S)


def scene_surfaces(scene_dir, xml: str = "base.xml") -> list[dict]:
    """The scene's mesh shapes with what its file says of their surface: ``{"id", "material", "file", "roughness",
    "metallic", "emitter", "transmissive"}``. Roughness is Blender's (the square root of the BSDF's alpha; a
    texture's mean); metallic is the conductor's share of the BSDF."""
    scene_dir = Path(scene_dir)
    text = (scene_dir / xml).read_text()
    out = []
    for meta, kind, sid, body in _SHAPE.findall(text):
        if kind not in ("ply", "obj"):
            continue
        file = re.search(r'<string name="filename" value="([^"]+)"', body)
        if file is None:
            continue
        try:
            material = json.loads(meta).get("material") or sid
        except json.JSONDecodeError:
            material = sid
        rough = None
        alpha = re.search(r'<float name="alpha" value="([-\d.eE+]+)"', body)
        if alpha is not None:
            rough = math.sqrt(max(float(alpha.group(1)), 0.0))
        else:
            tex = re.search(r'<texture[^>]*name="alpha"[^>]*>.*?<string name="filename" value="([^"]+)"', body, re.S)
            if tex is not None:
                rough = _texture_mean_sqrt(scene_dir / tex.group(1))
        weight = re.search(r'<bsdf type="blendbsdf">\s*<float name="weight" value="([-\d.eE+]+)"', body)
        metallic = float(weight.group(1)) if weight is not None and "roughconductor" in body else \
            1.0 if ('type="roughconductor"' in body or 'type="conductor"' in body) else 0.0
        out.append({"id": sid, "material": str(material), "file": file.group(1), "roughness": rough, "metallic": metallic,
                    "emitter": "<emitter" in body, "transmissive": 'dielectric"' in body and "pplastic" not in body})
    return out


def _texture_mean_sqrt(path: Path) -> float | None:
    try:
        from PIL import Image
        a = np.asarray(Image.open(path))
    except Exception:  # noqa: BLE001 - a map that does not open says nothing
        return None
    top = 65535.0 if a.dtype == np.uint16 or a.max() > 255 else 255.0 if a.dtype.kind in "ui" else 1.0
    return float(np.sqrt(np.clip(a.astype(np.float64) / top, 0.0, 1.0)).mean())


def read_ply(path) -> tuple[np.ndarray, np.ndarray]:
    """(vertices N x 3, triangles M x 3) of a binary little-endian PLY (the converter's and Mitsuba's layout: float
    properties per vertex, faces as a list of three or four indices)."""
    data = Path(path).read_bytes()
    end = data.index(b"end_header") + len(b"end_header\n")
    head = data[:end].decode("ascii", "replace").splitlines()
    if not any("binary_little_endian" in line for line in head):
        raise ValueError(f"{path}: not a binary little-endian PLY")
    types = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8", "uchar": "u1", "uint8": "u1", "char": "i1",
             "int8": "i1", "short": "<i2", "ushort": "<u2", "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4"}
    elements, current = [], None
    for line in head:
        parts = line.split()
        if parts[:1] == ["element"]:
            current = {"name": parts[1], "count": int(parts[2]), "props": []}
            elements.append(current)
        elif parts[:1] == ["property"] and current is not None:
            current["props"].append(parts[1:])
    offset, verts, faces = end, None, np.zeros((0, 3), np.int64)
    for el in elements:
        if all(p[0] != "list" for p in el["props"]):
            dtype = np.dtype([(p[1], types[p[0]]) for p in el["props"]])
            block = np.frombuffer(data, dtype=dtype, count=el["count"], offset=offset)
            offset += dtype.itemsize * el["count"]
            if el["name"] == "vertex":
                verts = np.stack([block["x"], block["y"], block["z"]], 1).astype(np.float64)
        elif el["name"] == "face" and el["count"]:
            count_t, index_t = types[el["props"][0][1]], types[el["props"][0][2]]
            first = int(np.frombuffer(data, dtype=count_t, count=1, offset=offset)[0])
            row = np.dtype([("n", count_t), ("i", index_t, (first,))])
            block = np.frombuffer(data, dtype=row, count=el["count"], offset=offset)
            if not np.all(block["n"] == first):
                raise ValueError(f"{path}: faces of mixed size")
            offset += row.itemsize * el["count"]
            idx = block["i"].astype(np.int64)
            faces = idx if first == 3 else np.concatenate([idx[:, [0, 1, 2]], idx[:, [0, 2, 3]]])
    if verts is None:
        raise ValueError(f"{path}: no vertices")
    return verts, faces


@dataclass
class GroundMap:
    heights: np.ndarray            # rows along y, columns along x (m; NaN: nothing seen)
    material: np.ndarray           # index into ``materials`` of the surface to stand on; -1: none
    materials: list                # names
    friction: dict                 # name -> {"mu", "why", "roughness", "metallic"}
    origin: tuple                  # world (x, y) of cell [0, 0]
    cell: float
    floor_z: float
    top: float
    meta: dict = field(default_factory=dict)

    def mu(self) -> np.ndarray:
        """Friction coefficient per cell (NaN where there is no surface to stand on)."""
        table = np.array([self.friction[m]["mu"] for m in self.materials] + [np.nan])
        return table[self.material]

    def ground(self, step: float = 0.03):
        """The hw.ground.Ground of this map."""
        from .hw.ground import Ground
        g = Ground(mu=DEFAULT_MU, z=self.floor_z, step=step)
        g.set_height_map(self.heights, self.origin, self.cell)
        g.set_material_map(self.material, self.materials, {m: v["mu"] for m, v in self.friction.items()})
        return g

    def save(self, path) -> None:
        np.savez_compressed(path, heights=self.heights.astype(np.float32), material=self.material.astype(np.int16),
                            info=json.dumps({"materials": self.materials, "friction": self.friction, "origin": self.origin,
                                             "cell": self.cell, "floor_z": self.floor_z, "top": self.top, "meta": self.meta}))

    @classmethod
    def load(cls, path) -> "GroundMap":
        z = np.load(path, allow_pickle=False)
        info = json.loads(str(z["info"]))
        return cls(z["heights"].astype(np.float64), z["material"].astype(np.int64), info["materials"], info["friction"],
                   tuple(info["origin"]), info["cell"], info["floor_z"], info["top"], info.get("meta", {}))


def _vertex_extent(scene_dir: Path, surfaces: list, lo: float, hi: float, cache: dict):
    """(x0, y0, x1, y1) of the vertices within the height band, in the world: the default bounds."""
    x0 = y0 = math.inf
    x1 = y1 = -math.inf
    for s in surfaces:
        mesh = _mesh(scene_dir / s["file"], cache)
        if mesh is None:
            continue
        v = mesh[0]
        v = v[(v[:, 1] >= lo) & (v[:, 1] <= hi)]
        if len(v):
            x0, x1 = min(x0, float(v[:, 0].min())), max(x1, float(v[:, 0].max()))
            y0, y1 = min(y0, float(-v[:, 2].max())), max(y1, float(-v[:, 2].min()))
    if not math.isfinite(x0):
        raise ValueError(f"{scene_dir}: nothing between {lo:.2f} and {hi:.2f} m")
    return x0, y0, x1, y1


def _mesh(file: Path, cache: dict):
    if file not in cache:
        try:
            cache[file] = read_ply(file)
        except (OSError, ValueError):
            cache[file] = None
    return cache[file]


POINTS_AT_ONCE = 2_000_000   # samples held at a time: a scene of any size is binned in pieces of this many


def build(scene_dir, floor_z: float, top: float = 1.4, cell: float = 0.05, bounds=None, xml: str = "base.xml",
          below: float = 0.5, progress=None) -> GroundMap:
    """Sample the scene's meshes into a ground map around the floor at ``floor_z``.

    ``top``: the robot's height with what it carries: anything between the floor and it is in the way. ``bounds``:
    (x0, y0, x1, y1) in the world; default: the extent of what stands within the height band. A surface counts as
    one to stand on when it faces up (within 45 degrees) and lies within ``below`` under and above the floor.
    Memory is the grids' (25 bytes a cell) plus ``POINTS_AT_ONCE`` samples, whatever the scene's size."""
    scene_dir = Path(scene_dir)
    surfaces = [s for s in scene_surfaces(scene_dir, xml) if not s["emitter"]]
    names = sorted({s["material"] for s in surfaces})
    index = {n: i for i, n in enumerate(names)}
    lo, hi = float(floor_z) - below, float(floor_z) + float(top)
    spacing = cell * 0.5
    cache: dict = {}
    if bounds is None:
        bounds = _vertex_extent(scene_dir, surfaces, lo, hi, cache)
    x0, y0, x1, y1 = (float(v) for v in bounds)
    cols, rows = int(math.ceil((x1 - x0) / cell)) + 1, int(math.ceil((y1 - y0) / cell)) + 1
    if cols * rows > 40_000_000:
        raise ValueError(f"a {cols} x {rows} map: give bounds or a larger cell")
    heights = np.full(rows * cols, -np.inf)
    stand_z = np.full(rows * cols, -np.inf)
    material = np.full(rows * cols, -1, np.int16)
    samples = 0

    def bin_points(pts: np.ndarray, stand: np.ndarray, mat: int) -> int:
        """Samples (Mitsuba axes) into the grids: the highest of all, the highest to stand on and its material."""
        inside = (pts[:, 1] >= lo) & (pts[:, 1] <= hi)
        pts, stand = pts[inside], stand[inside]
        j = np.rint((pts[:, 0] - x0) / cell).astype(np.int64)
        i = np.rint((-pts[:, 2] - y0) / cell).astype(np.int64)
        ok = (j >= 0) & (j < cols) & (i >= 0) & (i < rows)
        flat, z = (i * cols + j)[ok], pts[ok, 1]
        if not len(flat):
            return 0
        np.maximum.at(heights, flat, z)
        walk = stand[ok] & (z <= floor_z + below)
        if walk.any():
            wf, wz = flat[walk], z[walk]
            np.maximum.at(stand_z, wf, wz)
            best = wz >= stand_z[wf] - 1e-9   # a later, higher surface takes the cell over
            material[wf[best]] = mat
        return int(len(flat))

    for n, s in enumerate(surfaces):
        if progress is not None and n % 200 == 0:
            progress(n / max(len(surfaces), 1))
        mesh = _mesh(scene_dir / s["file"], cache)
        if mesh is None or not len(mesh[1]):
            continue
        v, f = mesh
        tri = v[f]                                            # M x 3 x 3, Mitsuba axes
        ys, xs, zs = tri[..., 1], tri[..., 0], -tri[..., 2]
        keep = (ys.max(1) >= lo) & (ys.min(1) <= hi) & (xs.max(1) >= x0) & (xs.min(1) <= x1) & (zs.max(1) >= y0) & (zs.min(1) <= y1)
        if not keep.any():
            continue
        tri = tri[keep]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = 0.5 * np.linalg.norm(normal, axis=1)
        up = np.abs(normal[:, 1]) / np.maximum(2.0 * area, 1e-18) > math.cos(math.radians(45.0))   # either winding
        # a lattice over each triangle fine enough that no cell under it is missed (random points leave holes, and an
        # unseen cell blocks): n steps along each edge, n from the longest edge
        edge = np.maximum.reduce([np.linalg.norm(tri[:, a] - tri[:, b], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))])
        steps = np.clip(np.ceil(edge / spacing).astype(np.int64), 1, 4000)
        mat = index[s["material"]]
        for n_steps in np.unique(steps):
            group, group_up = tri[steps == n_steps], up[steps == n_steps]
            a, b = np.meshgrid(np.arange(n_steps + 1), np.arange(n_steps + 1), indexing="ij")
            keep_ab = (a + b) <= n_steps
            wa, wb = a[keep_ab] / n_steps, b[keep_ab] / n_steps
            bary = np.stack([1.0 - wa - wb, wa, wb], 1)                           # K x 3
            # pieces of at most POINTS_AT_ONCE: triangles at a time, and for one huge triangle its lattice at a time
            per = max(1, POINTS_AT_ONCE // len(bary))
            for m0 in range(0, len(group), per):
                part, part_up = group[m0:m0 + per], group_up[m0:m0 + per]
                for k0 in range(0, len(bary), POINTS_AT_ONCE):
                    w = bary[k0:k0 + POINTS_AT_ONCE]
                    pts = np.einsum("kc,mcd->mkd", w, part).reshape(-1, 3)
                    samples += bin_points(pts, np.repeat(part_up, len(w)), mat)
        if len(cache) > 64:       # the meshes are read once each (instances aside): do not keep them all
            cache.clear()
    if not samples:
        raise ValueError(f"{scene_dir}: nothing between {lo:.2f} and {hi:.2f} m within the bounds")
    heights[np.isinf(heights)] = np.nan
    friction = {}
    for name in names:
        group = [s for s in surfaces if s["material"] == name]
        rough = [s["roughness"] for s in group if s["roughness"] is not None]
        r = float(np.mean(rough)) if rough else None
        m = float(np.mean([s["metallic"] for s in group]))
        mu, why = material_friction(name, r, m)
        friction[name] = {"mu": mu, "why": why, "roughness": None if r is None else round(r, 3), "metallic": round(m, 3)}
    return GroundMap(heights.reshape(rows, cols), material.reshape(rows, cols).astype(np.int64), names, friction, (x0, y0), float(cell),
                     float(floor_z), float(top), {"scene": scene_dir.name, "xml": xml, "samples": int(samples), "shapes": len(surfaces)})
