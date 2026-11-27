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
import xml.etree.ElementTree as ET
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


READER = 3   # how scene files are read; a saved map of another reading is built again (hw.sim.SimWorld.load_ground)


def pushed_parts(scene_dir) -> set:
    """The shapes of the objects the physics step moves (physics.json's objects kept apart by tools/merge_meshes.py): not
    in the ground map, which is what does not move. Counted as obstacles, CS02's cart stopped the robot ahead of contact."""
    path = Path(scene_dir) / "physics.json"
    if not path.is_file():
        return set()
    data = json.loads(path.read_text())
    kept = {k for k, v in ((data.get("merge") or {}).get("objects") or {}).items() if v.get("kept")}
    return {str(p) for o in data.get("objects") or [] if o["id"] in kept for p in o.get("parts") or []}


def pushed_key(scene_dir) -> str:
    """What a saved ground map left out for the physics step (pushed_parts), so it is built again when that changes."""
    import hashlib
    return hashlib.sha1("\n".join(sorted(pushed_parts(scene_dir))).encode()).hexdigest()[:12]

# Mitsuba's own shapes as meshes: (vertices, triangles)
_RECTANGLE = (np.array([[-1.0, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]), np.array([[0, 1, 2], [0, 2, 3]]))
_CUBE = (np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]),
         np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4],
                   [1, 5, 7], [1, 7, 3]]))
_PRIMITIVES = {"rectangle": _RECTANGLE, "cube": _CUBE}


def _to_world(shape) -> np.ndarray | None:
    """A shape's ``to_world`` as a 4x4 (Mitsuba applies the listed steps in order); None: none."""
    tf = next((t for t in shape.findall("transform") if t.get("name") == "to_world"), None)
    if tf is None:
        return None
    m = np.eye(4)
    for step in tf:
        if not isinstance(step.tag, str):
            continue
        a = np.eye(4)
        v = [float(x) for x in re.split(r"[\s,]+", (step.get("value") or "").strip()) if x]
        if step.tag == "translate":
            a[:3, 3] = v if len(v) == 3 else [float(step.get(k, 0.0)) for k in "xyz"]
        elif step.tag == "scale":
            a[[0, 1, 2], [0, 1, 2]] = v if len(v) == 3 else v * 3 if len(v) == 1 else [float(step.get(k, 1.0)) for k in "xyz"]
        elif step.tag == "rotate":
            k = np.array(v if len(v) == 3 else [float(step.get(c, 0.0)) for c in "xyz"])
            k = k / np.linalg.norm(k)
            t = math.radians(float(step.get("angle")))
            cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            a[:3, :3] = np.eye(3) + math.sin(t) * cross + (1.0 - math.cos(t)) * (cross @ cross)
        elif step.tag == "matrix":
            a = np.array(v).reshape((4, 4) if len(v) == 16 else (3, 3))
            if a.shape == (3, 3):
                a = np.block([[a, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])
        else:
            raise ValueError(f"shape {shape.get('id')!r}: a to_world step <{step.tag}> is not read here")
        m = a @ m
    return m


def scene_surfaces(scene_dir, xml: str = "base.xml") -> list[dict]:
    """The scene's mesh shapes with what its file says of their surface: ``{"id", "material", "file", "kind", "matrix",
    "roughness", "metallic", "emitter", "transmissive"}``, instances of a shape group written out. ``matrix`` is the
    shape's 4x4 in the world (None: as the mesh lies), ``file`` the mesh of a "ply" or "obj" (None for a "rectangle" or
    "cube"). Roughness is Blender's (the square root of the BSDF's alpha); metallic is the conductor's share of the BSDF.
    Where a texture gives one, ``roughness_map`` and ``metallic_map`` name it (``build`` reads the maps of what a robot
    can stand on: their mean).

    A scene merged by tools/merge_meshes.py is read as it was before (merged_meshes.original_text: every object in its own
    file, in its place): where coplanar surfaces meet, the later one in the file names the cell, and read merged, 26 cells of
    CS02's floor at the walls' feet went from the epoxy floor to the wall paint."""
    from .merged_meshes import original_text

    scene_dir = Path(scene_dir)
    root = ET.fromstring(original_text(scene_dir, xml), parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    shared = {b.get("id"): b for b in root.iter("bsdf") if b.get("id")}

    def read(shape, meta: str | None, matrix) -> dict | None:
        kind, sid = shape.get("type"), shape.get("id") or ""
        if kind not in ("ply", "obj", "rectangle", "cube"):
            return None
        file = shape.find("string[@name='filename']")
        if kind in ("ply", "obj") and file is None:
            return None
        material = sid
        if meta is not None:
            try:
                material = json.loads(meta).get("material") or sid
            except json.JSONDecodeError:
                pass
        bsdfs = [b for b in shape.findall("bsdf")] + [shared[r.get("id")] for r in shape.findall("ref") if r.get("id") in shared]
        parts = [b for top in bsdfs for b in top.iter("bsdf")]
        types = {b.get("type") for b in parts}
        rough = rough_map = metal_map = None
        alpha = next((a for b in parts for a in b if a.get("name") == "alpha"), None)
        if alpha is not None and alpha.tag == "float":
            rough = math.sqrt(max(float(alpha.get("value")), 0.0))
        elif alpha is not None and alpha.tag == "texture" and alpha.find("string[@name='filename']") is not None:
            rough_map = alpha.find("string[@name='filename']").get("value")
        weight = next((w for b in parts if b.get("type") == "blendbsdf" for w in b if w.get("name") == "weight"), None)
        if weight is not None and weight.tag != "float":
            image = weight.find("string[@name='filename']")
            metal_map, weight = image.get("value") if image is not None else None, None
        conductor = bool(types & {"roughconductor", "conductor"})
        metallic = float(weight.get("value")) if weight is not None and conductor else \
            0.0 if "blendbsdf" in types else 1.0 if conductor else 0.0
        return {"id": sid, "material": str(material), "file": None if kind in _PRIMITIVES else file.get("value"), "kind": kind,
                "matrix": matrix, "roughness": rough, "metallic": metallic, "roughness_map": rough_map,
                "metallic_map": metal_map if conductor else None, "emitter": shape.find("emitter") is not None,
                "transmissive": any(t and t.endswith("dielectric") for t in types)}

    def walk(parent):
        """(shape, the metadata comment before it) of an element's own shapes."""
        meta = None
        for el in parent:
            if el.tag is ET.Comment:
                text = (el.text or "").strip()
                meta = text[len("opticalnav-obj:"):] if text.startswith("opticalnav-obj:") else meta
            elif el.tag == "shape":
                yield el, meta
                meta = None
            else:
                meta = None

    out, groups = [], {}
    for shape, meta in walk(root):
        if shape.get("type") == "shapegroup":
            groups[shape.get("id")] = [r for r in (read(c, m or meta, _to_world(c)) for c, m in walk(shape)) if r is not None]
    for shape, meta in walk(root):
        if shape.get("type") == "instance":
            ref = shape.find("ref")
            inner = shape.find("shape")
            at = _to_world(shape)
            members = groups.get(ref.get("id"), []) if ref is not None else \
                [r for r in (read(c, m or meta, _to_world(c)) for c, m in walk(inner)) if r is not None] if inner is not None else []
            for k in members:
                m = at if k["matrix"] is None else k["matrix"] if at is None else at @ k["matrix"]
                out.append({**k, "matrix": m})
        elif shape.get("type") != "shapegroup":
            r = read(shape, meta, _to_world(shape))
            if r is not None:
                out.append(r)
    return out


def _texture_mean(path: Path, power: float = 1.0) -> float | None:
    try:
        from PIL import Image
        a = np.asarray(Image.open(path))
    except Exception:  # noqa: BLE001 - a map that does not open says nothing
        return None
    top = 65535.0 if a.dtype == np.uint16 or a.max() > 255 else 255.0 if a.dtype.kind in "ui" else 1.0
    return float((np.clip(a.astype(np.float64) / top, 0.0, 1.0) ** power).mean())


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
        mesh = _surface_mesh(scene_dir, s, cache)
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


_EXTENTS: dict = {}   # mesh file -> the 8 corners of its vertices' box: the second map of a scene reads only what is near


def _mesh(file: Path, cache: dict):
    if file not in cache:
        try:
            cache[file] = read_ply(file)
        except (OSError, ValueError):
            cache[file] = None
        v = cache[file][0] if cache[file] is not None else np.zeros((0, 3))
        _EXTENTS[file] = np.array([[x, y, z] for x in (v[:, 0].min(), v[:, 0].max()) for y in (v[:, 1].min(), v[:, 1].max())
                                   for z in (v[:, 2].min(), v[:, 2].max())]) if len(v) else None
    return cache[file]


def _surface_mesh(scene_dir: Path, s: dict, cache: dict, lo: float = -math.inf, hi: float = math.inf, bounds=None):
    """(vertices in the world, Mitsuba axes; triangles) of a surface; None: not readable, empty, or wholly outside the
    height band and the ``bounds``."""
    m = s["matrix"]

    def placed(v):
        return v if m is None else v @ m[:3, :3].T + m[:3, 3]
    if s["kind"] in _PRIMITIVES:
        v, f = _PRIMITIVES[s["kind"]]
        return placed(v), f
    file = scene_dir / s["file"]
    if file in _EXTENTS:
        box = _EXTENTS[file]
        if box is None:
            return None
        c = placed(box)
        if c[:, 1].max() < lo or c[:, 1].min() > hi:
            return None
        if bounds is not None and (c[:, 0].max() < bounds[0] or c[:, 0].min() > bounds[2]
                                   or -c[:, 2].min() < bounds[1] or -c[:, 2].max() > bounds[3]):
            return None
    mesh = _mesh(file, cache)
    if mesh is None or not len(mesh[1]):
        return None
    return placed(mesh[0]), mesh[1]


POINTS_AT_ONCE = 2_000_000   # samples held at a time: a scene of any size is binned in pieces of this many


def build(scene_dir, floor_z: float, top: float = 1.4, cell: float = 0.05, bounds=None, xml: str = "base.xml",
          below: float = 0.5, progress=None) -> GroundMap:
    """Sample the scene's meshes into a ground map around the floor at ``floor_z``.

    ``top``: the robot's height with what it carries: anything between the floor and it is in the way. ``bounds``:
    (x0, y0, x1, y1) in the world; default: the extent of what stands within the height band. A surface counts as
    one to stand on when it faces up (within 45 degrees) and lies within ``below`` under and above the floor.
    Memory is the grids' (25 bytes a cell) plus ``POINTS_AT_ONCE`` samples, whatever the scene's size."""
    scene_dir = Path(scene_dir)
    pushed = pushed_parts(scene_dir)   # moved by the physics step: not obstacles of the map
    surfaces = [s for s in scene_surfaces(scene_dir, xml) if not s["emitter"] and s["id"] not in pushed]
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
        mesh = _surface_mesh(scene_dir, s, cache, lo, hi, (x0, y0, x1, y1))
        if mesh is None:
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
    stood = {int(k) for k in np.unique(material)}
    groups: dict = {n: [] for n in names}
    for s in surfaces:
        groups[s["material"]].append(s)

    def value(s: dict, what: str, power: float):
        file = s[what + "_map"]
        if file is None or index[s["material"]] not in stood:
            return s[what]
        if (file, power) not in maps:
            maps[file, power] = _texture_mean(scene_dir / file, power)
        return s[what] if maps[file, power] is None else maps[file, power]
    maps: dict = {}
    for name in names:
        rough = [v for v in (value(s, "roughness", 0.5) for s in groups[name]) if v is not None]
        r = float(np.mean(rough)) if rough else None
        m = float(np.mean([value(s, "metallic", 1.0) for s in groups[name]]))
        mu, why = material_friction(name, r, m)
        friction[name] = {"mu": mu, "why": why, "roughness": None if r is None else round(r, 3), "metallic": round(m, 3)}
    return GroundMap(heights.reshape(rows, cols), material.reshape(rows, cols).astype(np.int64), names, friction, (x0, y0), float(cell),
                     float(floor_z), float(top), {"scene": scene_dir.name, "xml": xml, "samples": int(samples), "shapes": len(surfaces), "reader": READER,
                      "pushed": pushed_key(scene_dir)})


def floor_under(gm: GroundMap, x: float, y: float, near: float | None = None) -> float | None:
    """The floor at (x, y) by a map seen from the camera's height (``floor_probe``): the low end of what lies within
    0.4 m, and where that is furniture (a table under the camera), a level lower by up to 1.3 m that fills a good part
    of the 3 m around. A drop of a storey beside a balcony is not it. ``near``: only surfaces within 0.6 m of it."""
    i, j = np.indices(gm.heights.shape)
    d = np.hypot(gm.origin[0] + j * gm.cell - x, gm.origin[1] + i * gm.cell - y)
    stand = (gm.material >= 0) & ~np.isnan(gm.heights)
    if near is not None:
        stand &= np.abs(gm.heights - near) <= 0.6
    z = gm.heights[stand & (d <= 0.4)]
    around = gm.heights[stand & (d <= 1.5)]
    if not len(around):
        return None
    start = float(np.percentile(z, 20)) if len(z) else float(np.median(around))
    bins = np.round(around / 0.05).astype(int)
    for level in np.unique(bins):              # lowest first
        at = around[np.abs(bins - level) <= 1]
        if start - 1.3 <= level * 0.05 < start - 0.25 and len(at) >= 0.15 * len(around):
            return float(np.median(at))
    return start


def floor_probe(scene_dir, eye, reach=(0.0, 0.0)) -> GroundMap | None:
    """What lies under a camera at ``eye`` (x, y, height), up to ``reach`` (dx, dy) from it, seen from its height."""
    x, y = sorted((eye[0], eye[0] + reach[0])), sorted((eye[1], eye[1] + reach[1]))
    try:
        return build(scene_dir, eye[2] - 0.05, top=0.0, cell=0.1, below=40.0,
                     bounds=(x[0] - 1.5, y[0] - 1.5, x[1] + 1.5, y[1] + 1.5))
    except ValueError:
        return None
