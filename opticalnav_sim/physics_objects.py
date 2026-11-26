"""Which objects of a scene a robot could push, decided from the scene's own files (no clicking): ``physics.json``.

    from opticalnav_sim import physics_objects
    result = physics_objects.analyse(scene_dir)     # the physics.json content (tools/physics_tag.py writes it)

A pack scene is a list of shapes, one per Blender object and material slot (tools/blend_to_pack.py), each after an
``opticalnav-obj`` note: its label (the Blender object's name), material, and from new conversions ``object`` (the root of
its parent chain) and ``collection``. A physics engine needs bodies, not shapes, so this module

1. reads every shape's world mesh: a merged scene (tools/merge_meshes.py) object by object, its parts from the merged PLY's
   vertex and face ranges in merged_meshes.json, so the answer does not depend on whether the scene is merged;
2. groups parts into objects (``group``): the note's ``object`` when it names a parent shared by several labels; else
   parts of one asset (``<scene>_rNNN``, an Infinigen ``...Factory(..).spawn_asset(..)``) or of one label stem
   (``bench leg.003`` -> ``bench leg``) whose boxes touch; then, between those pieces: one held up by another that it
   overhangs (a seat on its stem, a top on its legs), is fixed to (side contact: a front panel on its case) or lies
   inside; one held up by nothing, with everything it touches (a shelf board between a trolley's uprights); a small
   part (a caster, a key, a bolt) or a decal with the one piece it belongs to (the most label words in common, then the
   most contact). Pieces that stand on their own (two bins side by side) or sit on top of another with nothing of it
   above them (a cup on a table, a case on a shelf board) stay apart, and so do two assets unless fixed to each other.
   Structure (walls, floors, ...) never joins anything; a part longer than 2.5 m only what it holds up or stands on;
3. measures each object (``features``): world box, footprint, mass (closed-mesh volume, or a shell of the material's wall
   thickness when that is lighter: a cardboard box is not solid), centre of mass, what it stands on (vertical rays from
   its lowest points onto the parts below), whether it touches a wall or the ceiling, the floor under it;
4. classifies it (``classify``) by RULES and keyword tables (each object gets a reason that names what decided);
5. for a movable object: a convex hull of at most ``hull_vertices`` vertices around its centre of mass, the friction of
   its main material (groundmap.material_friction), so a physics engine can make a free body of it.

Frames: the scene files are Mitsuba's (X, Y up, Z); physics.json is the simulator's world (x = X, y = -Z, z = Y), as
groundmap's maps. The analysis is pure numpy/scipy on the PLY files (no Mitsuba).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .groundmap import _PRIMITIVES, _to_world, material_friction, read_ply
from .merged_meshes import is_merged, original_text, read_manifest

VERSION = 1
PHYSICS = "physics.json"
NOTE_HEAD = "opticalnav-obj:"
# shapes that are not scene objects: lights and their boxes, the camera flash, robot bodies
SKIP_PREFIXES = ("light_", "camera_assist", "body_")

RULES = {
    "touch_m": 0.005,           # parts of one asset or label stem whose boxes come this close are one piece
    "contact_m": 0.015,         # a vertex this close to another object's surface touches it (walls, joined pieces)
    "support_below_m": 0.03,    # a surface this far under an object's lowest points carries it ...
    "support_above_m": 0.02,    # ... or this far above them (meshes sink into each other by a few mm)
    "thin_share": 0.05,         # the share given to a narrow support found by rays up (a column under a table top)
    "sunk_m": 0.10,             # floors and ground an object is sunk into by up to this much still carry it
    "bottom_band_m": 0.02,      # an object's lowest points: within this of its bottom
    "large_side_m": 2.5,        # an object longer or taller than this is not pushed
    "large_footprint_m2": 2.0,
    "structure_side_m": 2.5,    # a single part this long joins only by support, and counts as a wall for what touches it
    "ground_footprint_m2": 4.0,  # a part this wide and flat, with nothing of the kind under it, is ground
    "ground_height_m": 0.3,
    "max_mass_kg": 50.0,        # heavier than a small robot pushes (a TurtleBot 4 is 3.9 kg, a Ranger Mini 64.5 kg)
    "free_height_m": 1.2,       # a floor object without a keyword: lower, smaller and lighter than this
    "free_footprint_m2": 0.6,
    "free_mass_kg": 30.0,
    "item_mass_kg": 10.0,       # an object on top of another: lighter and smaller than this
    "item_side_m": 0.6,
    "piece_m": 0.15,            # a part word on something smaller than this: a piece of whatever it touches most
    "flat_m": 0.003,           # thinner than this: a decal, a wear mark, a sheet (no body to push)
    "small_m": 0.05,            # nothing longer than this: a detail (a screw, a connector) left fixed ...
    "light_kg": 0.05,           # ... nor lighter than this (a glowing trace before a display, a case's loose handle)
    "reach_m": 0.4,           # a movable object whose bottom is this close to the floor: the robot's body can meet it
    "hull_vertices": 64,
}

# ---------------------------------------------------------------- words
# The class a label's words say. The first keyword of a label decides (``service cart shelf`` is a cart, ``round stool
# upholstered seat`` a stool), except a MODIFIER, which says where a thing is as often as what it is: followed by another
# keyword, that one decides (``bench tool driver`` is a tool, ``wired bench unit 0 case`` an instrument case, ``shelf
# manuals binder`` a binder); followed by part words only, the modifier does (``west bench top``, ``bench drawer``);
# followed by other words, nothing does (``bench measurement cable``, ``stacked bench controller shell``).
STRUCTURE = set("""wall walls floor floors flooring ceiling ceilings roof door doors doorway window windows frame frames mullion
mullions transom jamb lintel header sill casing column columns pillar pillars beam beams girder rail rails railing railings
handrail balustrade conduit conduits pipe pipes duct ducts ductwork stair stairs staircase step steps builtin counter countertop
skirting baseboard molding moulding cornice glazing pane facade cladding curb kerb road street sidewalk walkway pavement paving
pave plaza slab foundation terrain ground grass lawn soil mulch landing ramp gutter drain grate grating vent vents louver louvre
sprinkler pilaster partition fence hoarding awning canopy shutter curtain blind blinds concrete bedding tile tiles track dome
sky skylight wires pole poles mirror elevator escalator platform deck court street storefront shopfront fascia shopfascia
cove coffer truss scaffold""".split())
FURNITURE = set("""shelf shelves shelving rack racks cabinet cabinets cupboard wardrobe locker lockers bookcase bookshelf
showcase vitrine kiosk booth vending vend machine fridge refrigerator freezer oven stove sink toilet urinal bathtub shower
radiator boiler bench benches workbench bed sofa couch piano drawer drawers dresser island gate turnstile barrier bollard tree
trees hedge bush bushes shrub rock rocks stone stones boulder planted planter car cars bus truck van vehicle sign signs signage
billboard banner poster board whiteboard noticeboard ad ads fixture chandelier cassette gantry rig carrier housing harness
extrusion stage""".split())
MOVABLE = set("""chair chairs stool stools cart carts trolley trolleys dolly box boxes carton cartons crate crates case cases
suitcase luggage bin bins bucket buckets basket baskets cone cones bag bags trashbag backpack ball balls bottle bottles can
cans jar jars pot pots vase vases cup cups mug mugs plate plates bowl bowls toolbox tool tools spool driver screwdriver wrench
hammer pliers monitor keyboard mouse laptop phone book books binder binders magazine folder pillow cushion toy toys pallet
aframe
barrel drum tote container canister jerrycan extinguisher bike bicycle scooter fan printer speaker sanitizer kettle
microwave pcb instrument oscilloscope multimeter meter stack foodbox cupstack caliper""".split())
MODIFIERS = set("""bench wall floor table desk counter shelf window door office kitchen cafe coffee storage garden bar back
service lab""".split())
PART_WORDS = set("""top body frame panel side base front back rear edge corner cap cover surface apron under lower upper left
right inner outer leg legs foot feet shelf drawer door handle knob rail stretcher brace support""".split())
SHORT_WORDS = {"tv", "pc", "ac"}
IGNORED = {"factory", "spawn", "asset", "copy", "part", "inst", "item", "mesh", "object", "obj"}
CABLE_TRAY = re.compile(r"cable\W*tray")

# Density (kg/m^3) and the wall thickness (m) of a hollow object of the material: a body weighs the lighter of its
# closed volume and its surface as such a shell (a cardboard box modelled as a closed cube is not 0.06 m^3 of
# cardboard). Typical handbook densities; corrugated board as 4 mm at 200 kg/m^3 (about 0.8 kg/m^2).
MATERIALS = (  # (name, words, density, wall)
    ("steel", "steel iron chrome chromed metal metallic stainless galvanised galvanized brass bronze copper zinc tin conductor",
     7850.0, 0.0015),
    ("aluminium", "aluminium aluminum alu alum", 2700.0, 0.002),
    ("wood", "wood wooden oak birch pine walnut maple plywood mdf timber beech teak bamboo cork", 600.0, 0.018),
    ("rubber", "rubber silicone tire tyre tires caster", 1200.0, 0.006),
    ("fabric", "fabric cloth upholstery upholstered leather textile felt cushion foam velvet linen cotton", 200.0, 0.04),
    ("glass", "glass", 2500.0, 0.005),
    ("cardboard", "cardboard carton kraft corrugated", 200.0, 0.004),
    ("paper", "paper binder manual book books magazine", 800.0, 0.002),
    ("ceramic", "ceramic porcelain stoneware clay terracotta", 2300.0, 0.006),
    ("stone", "stone concrete marble granite brick cement", 2400.0, 0.04),
    ("plant", "plant leaf leaves foliage fern bush soil", 500.0, 0.005),
    ("plastic", "plastic abs pvc acrylic nylon polycarbonate vinyl resin poly polymer epoxy", 1050.0, 0.003),
)
DEFAULT_MATERIAL = "plastic"   # unknown: plastic-ish (most loose things in these scenes are)


def words(text: str) -> list[str]:
    """Lower-case words of a label or material: camel case split (``CardboardBoxFactory`` -> cardboard box factory),
    digits and punctuation dropped, the scene's codes (``cs02``, ``r008``, ``ph46``) with them."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text))
    out = []
    for w in re.findall(r"[a-z]+", text.lower()):
        if (len(w) > 2 or w in SHORT_WORDS) and w not in IGNORED:
            out.append(w)
    return out


def _known(word: str, table: set) -> bool:
    return word in table or (word.endswith("s") and word[:-1] in table) or (word.endswith("es") and word[:-2] in table)


def label_class(label: str) -> tuple[str | None, str | None]:
    """(class, keyword) a label's words give: "structure", "furniture" (fixed furniture and fixtures), "movable" or
    (None, None) for no keyword."""
    if CABLE_TRAY.search(str(label).lower()):
        return "structure", "cable tray"
    ws = words(label)
    hits = []
    for k, w in enumerate(ws):
        for cls, table in (("movable", MOVABLE), ("furniture", FURNITURE), ("structure", STRUCTURE)):
            if _known(w, table):
                hits.append((k, cls, w))
                break
    if not hits:
        return None, None
    k, cls, w = hits[0]
    if w in MODIFIERS:
        if len(hits) > 1:
            return hits[1][1:]
        if not all(_known(x, PART_WORDS) or _known(x, PIECES) for x in ws[k + 1:]):
            return None, None
    return cls, w


def material_of(*names: str) -> str:
    """The MATERIALS entry the names' words say (the shape's material first, then its label); else DEFAULT_MATERIAL."""
    for name in names:
        ws = set(words(name or ""))
        for entry in MATERIALS:
            if ws & set(entry[1].split()):
                return entry[0]
    return DEFAULT_MATERIAL


_DENSITY = {m[0]: (m[2], m[3]) for m in MATERIALS}


# ---------------------------------------------------------------- reading the scene
def label_stem(label: str) -> str:
    """A label without Blender's duplicate numbers: ``bench leg.003`` -> ``bench leg``, ``mullion_10`` -> ``mullion``."""
    s = re.sub(r"\.\d+$", "", str(label))
    while True:
        t = re.sub(r"[\s_./]+\d+$", "", s)
        if t == s or not t:
            return s
        s = t


_ASSET = re.compile(r"^(.+?_r\d{3})(?=[_\s]|$)")
_INFINIGEN = re.compile(r"^(.*?Factory)[(_](.*?)[)_]\.spawn_asset[(_]([^)_]*)[)_]?(\.\d+)?")


def asset_key(label: str) -> str | None:
    """The asset instance a label belongs to: ``CS02_r008_round_stool...`` -> ``CS02_r008``; an Infinigen asset
    (``CafeChairFactory(9).spawn_asset(9)``) is its own label; None for a label of the scene author's own parts."""
    m = _ASSET.match(str(label))
    if m:
        return m.group(1)
    # Infinigen, as written in a note (``ChairFactory(10).spawn_asset(10).001``) or as a shape id where a part has no
    # note (``ChairFactory_10_.spawn_asset_10_.001``): one key for both, a Blender copy (.001) an asset of its own
    m = _INFINIGEN.match(str(label))
    if m:
        return "|".join(re.sub(r"\W", "_", g or "") for g in m.groups())   # ids write "/" as "_"
    return None


def _ply(path: Path, cache: dict):
    if path not in cache:
        try:
            cache[path] = read_ply(path)
        except (OSError, ValueError):
            cache[path] = _obj(path) if path.suffix.lower() == ".obj" else None
    return cache[path]


def _obj(path: Path):
    try:
        v, f = [], []
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("v "):
                v.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(x.split("/")[0]) for x in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(v) + i for i in idx]
                f += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
        return np.array(v, np.float64).reshape(-1, 3), np.array(f, np.int64).reshape(-1, 3)
    except (OSError, ValueError, IndexError):
        return None


def _merged_slices(scene_dir: Path, text: str) -> dict:
    """part id -> (merged file, vertex range, face range) for the merged shapes the current scene file holds."""
    manifest = read_manifest(scene_dir) if is_merged(text) else None
    out = {}
    for g in (manifest or {}).get("merged", []):
        if g["xml"] in text:
            for p in g["parts"]:
                out[p["id"]] = (g["file"], p["vertex_range"], p["face_range"])
    return out


def read_parts(scene_dir, xml: str = "base.xml") -> list[dict]:
    """Every object shape of the scene with its world mesh (Mitsuba axes): ``{"id", "label", "material", "note",
    "kind", "verts", "faces", "lo", "hi", "order"}``. Lights, the flash and robot bodies are left out; an ``instance``
    is one part (its shape group's meshes at its matrix), labelled by its id without ``__inst``. A merged scene is read
    object by object (merged_meshes.original_text), the geometry of merged parts from their slice of the merged mesh."""
    scene_dir = Path(scene_dir)
    current = (scene_dir / xml).read_text()
    text = original_text(scene_dir, xml)
    slices = _merged_slices(scene_dir, current)
    root = ET.fromstring(text, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    cache: dict = {}

    def walk(parent):
        meta = None
        for el in parent:
            if el.tag is ET.Comment:
                t = (el.text or "").strip()
                meta = t[len(NOTE_HEAD):] if t.startswith(NOTE_HEAD) else meta
            elif el.tag == "shape":
                yield el, meta
                meta = None
            else:
                meta = None

    def mesh_of(shape, matrix=None):
        kind = shape.get("type")
        if kind in _PRIMITIVES:
            v, f = _PRIMITIVES[kind]
            v = v.astype(np.float64)
        elif kind in ("ply", "obj"):
            file = shape.find("string[@name='filename']")
            if file is None:
                return None
            got = _ply(scene_dir / file.get("value"), cache)
            if got is None or not len(got[1]):
                return None
            v, f = got
        else:
            return None
        m = _to_world(shape)
        if matrix is not None:
            m = matrix if m is None else matrix @ m
        if m is not None:
            v = v @ m[:3, :3].T + m[:3, 3]
        return v, f

    def note_of(meta):
        try:
            note = json.loads(meta) if meta else None
        except json.JSONDecodeError:
            return None
        return note if isinstance(note, dict) else None

    groups, labels, thinned = {}, {}, {}
    for shape, meta in walk(root):
        if shape.get("type") == "shapegroup":
            groups[shape.get("id")] = [c for c, _ in walk(shape)]
        elif (note_of(meta) or {}).get("label"):
            labels[shape.get("id") or ""] = str(note_of(meta)["label"])
    parts = []
    for order, (shape, meta) in enumerate(walk(root)):
        sid, kind = shape.get("id") or "", shape.get("type")
        if kind == "shapegroup" or sid.startswith(SKIP_PREFIXES):
            continue
        note = note_of(meta)
        # a part without a note (the older packs write one for an object's first material only: "X__part_001_...")
        # takes the label of the shape it is a part of
        base = re.sub(r"(__part_.*|__inst.*)$", "", sid) or sid
        label = str((note or {}).get("label") or labels.get(base) or base)
        if label.startswith("light:"):
            continue
        if kind == "instance":
            # a shape group drawn many times (PH43: 300 shrubs of 287,000 triangles, 76 M triangles in all): its meshes
            # thinned once to INSTANCE_TRIANGLES for contact and support, its box from all its vertices
            ref = shape.find("ref")
            members = groups.get(ref.get("id"), []) if ref is not None else []
            at = _to_world(shape)
            at = np.eye(4) if at is None else at
            meshes, full_lo, full_hi = [], [], []
            for c in members:
                if id(c) not in thinned:
                    m = mesh_of(c)
                    thinned[id(c)] = None if m is None else (m[0],) + _thin(m[0], m[1], INSTANCE_TRIANGLES)
                if thinned[id(c)] is None:
                    continue
                full, tv, tf = thinned[id(c)]
                world_full = full @ at[:3, :3].T + at[:3, 3]
                full_lo.append(world_full.min(0))
                full_hi.append(world_full.max(0))
                meshes.append((tv @ at[:3, :3].T + at[:3, 3], tf))
            if not meshes:
                continue
            off = np.cumsum([0] + [len(m[0]) for m in meshes])
            v = np.concatenate([m[0] for m in meshes])
            f = np.concatenate([m[1] + o for m, o in zip(meshes, off)])
            material = ""
        elif sid in slices:
            file, (v0, v1), (f0, f1) = slices[sid]
            got = _ply(scene_dir / file, cache)
            if got is None:
                continue
            v, f = got[0][v0:v1], got[1][f0:f1] - v0
            material = str((note or {}).get("material") or "")
        else:
            got = mesh_of(shape)
            if got is None:
                continue
            v, f = got
            material = str((note or {}).get("material") or "")
        if not len(v) or not len(f):
            continue
        v = np.asarray(v, np.float32)   # single precision: half the memory (see _mass)
        parts.append({"id": sid, "label": label, "material": material, "note": note, "kind": kind, "verts": v,
                      "faces": np.asarray(f, np.int32), "lo": v.min(0).astype(np.float64),
                      "hi": v.max(0).astype(np.float64), "order": order,
                      "emitter": shape.find("emitter") is not None})
        if kind == "instance":
            parts[-1]["lo"], parts[-1]["hi"] = np.min(full_lo, 0), np.max(full_hi, 0)
    return parts


INSTANCE_TRIANGLES = 20_000


def _thin(v: np.ndarray, f: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """At most ``limit`` of the triangles, evenly through the list, with only the vertices they use."""
    if len(f) <= limit:
        return v, f
    f = f[np.linspace(0, len(f) - 1, limit).astype(np.int64)]
    used, inverse = np.unique(f, return_inverse=True)
    return v[used], inverse.reshape(-1, 3)


# ---------------------------------------------------------------- geometry helpers
def _tris(part) -> np.ndarray:
    """A part's triangles (M x 3 x 3), with their boxes cached (``tlo``, ``thi``) for _tris_near."""
    if "tri" not in part:
        tri = part["verts"][part["faces"]]
        lo = tri.min(1)
        order = np.argsort(lo[:, 0], kind="stable")      # by their lowest x: _tris_near cuts a window, not a scan
        part["tri"], part["tlo"], part["thi"] = tri[order], lo[order], tri.max(1)[order]
        part["tspan"] = float((part["thi"][:, 0] - part["tlo"][:, 0]).max()) if len(tri) else 0.0
    return part["tri"]


def _tris_near(part_or_tri, lo, hi, pad: float) -> np.ndarray:
    """The triangles (of a part, or an array) whose boxes come within ``pad`` of the box lo..hi. A part's are sorted by
    their lowest x, so only those starting within [lo - pad - widest triangle, hi + pad] along x are tested (a ground of
    200,000 triangles asked about the corner under a bin, 700 times)."""
    lo, hi = np.asarray(lo, float) - pad, np.asarray(hi, float) + pad
    if isinstance(part_or_tri, dict):
        tri, tlo, thi = _tris(part_or_tri), part_or_tri["tlo"], part_or_tri["thi"]
        a = int(np.searchsorted(tlo[:, 0], lo[0] - part_or_tri["tspan"], side="left"))
        b = int(np.searchsorted(tlo[:, 0], hi[0], side="right"))
        tri, tlo, thi = tri[a:b], tlo[a:b], thi[a:b]
    else:
        tri = part_or_tri
        tlo, thi = tri.min(1), tri.max(1)
    keep = np.all((thi >= lo) & (tlo <= hi), 1)
    return tri[keep]


def _sample(points: np.ndarray, n: int) -> np.ndarray:
    if len(points) <= n:
        return points
    return points[np.linspace(0, len(points) - 1, n).astype(np.int64)]


def _keys(points: np.ndarray, quantum: float) -> np.ndarray:
    """One int64 per point, equal for points equal after rounding to ``quantum`` (21 bits an axis, from the points'
    minimum; the quantum grows for a cloud wider than 2^21 quanta)."""
    lo = points.min(0)
    q = max(quantum, float((points.max(0) - lo).max()) / (2 ** 21 - 2))
    g = np.round((points - lo) / q).astype(np.int64)
    return (g[:, 0] << 42) | (g[:, 1] << 21) | g[:, 2]


def _unique_points(v: np.ndarray, quantum: float = 1e-4, limit: int = 60_000) -> np.ndarray:
    _, first = np.unique(_keys(v, quantum), return_index=True)
    return _sample(v[np.sort(first)], limit)


def point_triangle_distance(p: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Distance of points to triangles, broadcast: p (..., 3) against tri (..., 3, 3) (Ericson's closest point on a
    triangle). Points K x 1 x 3 against triangles 1 x M x 3 x 3 give K x M."""
    a, b, c = tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]
    ab, ac, ap, bp, cp = b - a, c - a, p - a, p - b, p - c
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    with np.errstate(divide="ignore", invalid="ignore"):
        den = va + vb + vc
        v, w = vb / den, vc / den
        q = a + ab * v[..., None] + ac * w[..., None]
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        q = np.where(((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[..., None], b + (c - b) * t[..., None], q)
        t = d2 / (d2 - d6)
        q = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None], a + ac * t[..., None], q)
        q = np.where(((d6 >= 0) & (d5 <= d6))[..., None], c, q)
        t = d1 / (d1 - d3)
        q = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None], a + ab * t[..., None], q)
        q = np.where(((d3 >= 0) & (d4 <= d3))[..., None], b, q)
        q = np.where(((d1 <= 0) & (d2 <= 0))[..., None], a, q)
        dist = np.linalg.norm(p - q, axis=-1)
    corner = np.minimum(np.minimum(np.linalg.norm(ap, axis=-1), np.linalg.norm(bp, axis=-1)), np.linalg.norm(cp, axis=-1))
    return np.where(np.isfinite(dist), np.minimum(dist, corner), corner)   # a degenerate triangle: its corners


def near(points: np.ndarray, tri: np.ndarray, tol: float, chunk: int = 1_000_000) -> np.ndarray:
    """Which points lie within ``tol`` of the triangles (bool per point). Only the (point, triangle) pairs whose boxes
    meet are measured."""
    out = np.zeros(len(points), bool)
    if not len(points) or not len(tri):
        return out
    tlo, thi = tri.min(1) - tol, tri.max(1) + tol
    step = max(1, chunk // len(tri))
    for i0 in range(0, len(points), step):
        p = points[i0:i0 + step]
        pi, ti = np.nonzero(np.all((p[:, None] >= tlo[None]) & (p[:, None] <= thi[None]), -1))
        for k0 in range(0, len(pi), chunk):
            a, t = pi[k0:k0 + chunk], ti[k0:k0 + chunk]
            hit = point_triangle_distance(p[a], tri[t]) <= tol
            out[i0 + a[hit]] = True
    return out


def heights_at(xz: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Height (Y) of each triangle over each point (x, z): K x M, NaN where the point is outside the triangle seen from
    above (or the triangle is vertical)."""
    a, b, c = tri[None, :, 0], tri[None, :, 1], tri[None, :, 2]
    px, pz = xz[:, None, 0], xz[:, None, 1]
    v0x, v0z, v1x, v1z = b[..., 0] - a[..., 0], b[..., 2] - a[..., 2], c[..., 0] - a[..., 0], c[..., 2] - a[..., 2]
    den = v0x * v1z - v1x * v0z
    with np.errstate(divide="ignore", invalid="ignore"):
        u = ((px - a[..., 0]) * v1z - v1x * (pz - a[..., 2])) / den
        w = (v0x * (pz - a[..., 2]) - (px - a[..., 0]) * v0z) / den
        h = a[..., 1] + u * (b[..., 1] - a[..., 1]) + w * (c[..., 1] - a[..., 1])
        ok = (np.abs(den) > 1e-12) & (u >= -1e-6) & (w >= -1e-6) & (u + w <= 1 + 1e-6)
    return np.where(ok, h, np.nan)


def _hits_in_band(points: np.ndarray, tri: np.ndarray, below: float, above: float, chunk: int = 2_000_000) -> np.ndarray:
    """Which points have one of the triangles under (or over) them within [y - below, y + above] (bool per point)."""
    out = np.zeros(len(points), bool)
    if not len(points) or not len(tri):
        return out
    step = max(1, chunk // len(tri))
    for i0 in range(0, len(points), step):
        p = points[i0:i0 + step]
        with np.errstate(invalid="ignore"):
            d = heights_at(p[:, [0, 2]], tri) - p[:, 1:2]
            out[i0:i0 + step] = np.any((d >= -below) & (d <= above), axis=1)
    return out


def mesh_volume(tri: np.ndarray) -> float:
    """Signed volume of a closed triangle soup (divergence theorem)."""
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def closed_fraction(tri: np.ndarray, quantum: float = 1e-4) -> float:
    """Share of the edges met by an even number of triangles (positions rounded to ``quantum``: the converter splits
    vertices at normal and UV seams, the surface stays closed). 1.0: watertight."""
    if not len(tri):
        return 0.0
    _, ids = np.unique(_keys(tri.reshape(-1, 3), quantum), return_inverse=True)
    ids = ids.reshape(-1, 3).astype(np.int64)
    e = np.sort(np.concatenate([ids[:, [0, 1]], ids[:, [1, 2]], ids[:, [2, 0]]]), axis=1)
    e = e[e[:, 0] != e[:, 1]]
    if not len(e):
        return 0.0
    _, counts = np.unique(e[:, 0] * (int(ids.max()) + 1) + e[:, 1], return_counts=True)
    return float((counts % 2 == 0).sum() / len(counts))


# ---------------------------------------------------------------- grouping
# words of surfaces one stands on (an object on them is on the floor), and of pieces that belong to something else (a
# bolt, a caster, a key of a keyboard: joined to what they touch, never an object of their own)
GROUND = set("""floor floors flooring ground terrain grass lawn soil mulch road street sidewalk walkway pavement paving pave
plaza slab landing ramp deck platform court tile tiles bedding stair stairs step steps carpet rug""".split())
PIECES = set("""bolt bolts screw screws nut nuts washer washers foot feet socket sockets knob knobs key keys handle handles
port ports button buttons switch led leds slot slots hinge hinges bracket brackets leg legs caster casters wheel wheels spoke
spokes stem grip grips trace traces grid display screen bezel lid strip seal gasket upright uprights apron piping trim
encoder collar knob glide glides latch latches rib ribs wear scuff decal decals sticker stickers overlay smudge scratch
scratches grime""".split())


class _Union:
    def __init__(self, n: int):
        self.up = list(range(n))

    def find(self, i: int) -> int:
        while self.up[i] != i:
            self.up[i] = self.up[self.up[i]]
            i = self.up[i]
        return i

    def join(self, i: int, j: int) -> None:
        a, b = self.find(i), self.find(j)
        if a != b:
            self.up[max(a, b)] = min(a, b)


def _touching_boxes(lo: np.ndarray, hi: np.ndarray, tol: float) -> list[tuple[int, int]]:
    """Index pairs of boxes that overlap or come within ``tol`` (a sweep along x)."""
    order = np.argsort(lo[:, 0], kind="stable")
    pairs = []
    for k, i in enumerate(order):
        for j in order[k + 1:]:
            if lo[j, 0] > hi[i, 0] + tol:
                break
            if np.all(lo[j] <= hi[i] + tol) and np.all(lo[i] <= hi[j] + tol):
                pairs.append((int(i), int(j)))
    return pairs


def _part_flags(parts: list, rules: dict) -> None:
    """Per part: ``class``/``keyword`` (label_class), ``ground`` (a surface to stand on: a ground word, or wide and flat),
    ``structure`` (a structure word or ground: never part of an object), ``large`` (longer than structure_side_m: joins
    only what it holds up or stands on, counts as a wall) and ``piece`` (a part word and no other keyword)."""
    for p in parts:
        cls, kw = label_class(p["label"])
        size = p["hi"] - p["lo"]
        ws = set(words(p["label"]))
        p["class"], p["keyword"] = cls, kw
        flat_wide = size[0] * size[2] > rules["ground_footprint_m2"] and size[1] < rules["ground_height_m"]
        p["ground"] = bool((cls == "structure" and (ws & GROUND)) or flat_wide)
        p["structure"] = bool(cls == "structure" or p["ground"])
        p["large"] = bool(size.max() > rules["structure_side_m"])
        # a piece: small, and its head word says it is part of something ("monitor key", "cart caster"), or flat (a
        # decal: dial ticks printed on an instrument's front, which also graze the bench under it); a leg, an upright or
        # a bezel is a member that joins everything it touches, not a piece that joins the one thing it touches most
        last = words(p["label"])[-1:]
        p["piece"] = bool((last and last[0] in PIECES and size.max() < rules["piece_m"]) or size.min() < rules["flat_m"])
        p["flat_wide"] = bool(flat_wide and not (cls == "structure" and (ws & GROUND)))
    # a wide flat part with ground under it is not ground (a skylight, a big table top): only the lowest is
    named = [p for p in parts if p["ground"] and not p["flat_wide"]]
    wide = [p for p in parts if p["flat_wide"]]
    for p in wide:
        if any(q is not p and q["hi"][1] < p["lo"][1] - 0.5 and np.all(q["lo"][[0, 2]] < p["hi"][[0, 2]])
               and np.all(q["hi"][[0, 2]] > p["lo"][[0, 2]]) for q in named + wide):
            p["ground"] = False
            p["structure"] = p["class"] == "structure"


def _seed_keys(parts: list) -> list:
    """The seed each part starts in: its note's ``object`` when that parent holds several label stems and no structure
    (a root over a whole room says nothing; a root that is the object itself neither), else its asset, else its label
    stem. Structure: its own label stem."""
    stems_of, structural = defaultdict(set), defaultdict(bool)
    for p in parts:
        root = (p["note"] or {}).get("object")
        if root:
            stems_of[root].add(label_stem(p["label"]))
            structural[root] |= p["structure"]
    keys = []
    for p in parts:
        root = (p["note"] or {}).get("object")
        if p["structure"]:
            keys.append(("structure", label_stem(p["label"])))
        elif root and len(stems_of[root]) > 1 and not structural[root]:
            keys.append(("object", root))
        elif asset_key(p["label"]):
            keys.append(("asset", asset_key(p["label"])))
        else:
            keys.append(("stem", label_stem(p["label"])))
    return keys


def _components(parts: list, keys: list, tol: float) -> list[list[int]]:
    """Parts of one seed whose boxes touch, as lists of part indices."""
    by_key = defaultdict(list)
    for i, k in enumerate(keys):
        by_key[k].append(i)
    comps = []
    for idx in by_key.values():
        uf = _Union(len(idx))
        lo = np.array([parts[i]["lo"] for i in idx])
        hi = np.array([parts[i]["hi"] for i in idx])
        for a, b in _touching_boxes(lo, hi, tol):
            uf.join(a, b)
        groups = defaultdict(list)
        for k, i in enumerate(idx):
            groups[uf.find(k)].append(i)
        comps += list(groups.values())
    return sorted(comps, key=lambda c: parts[c[0]]["order"])


class Body:
    """Parts as one body: their box, lowest points, and (on demand) triangles and vertices. Triangles near a box are
    found part by part (``near``): a tree or a bus is millions of them, and most contact questions are about a corner."""

    def __init__(self, parts: list, idx: list[int], rules: dict):
        self.idx = list(idx)
        self.parts = [parts[i] for i in idx]
        self.lo = np.min([p["lo"] for p in self.parts], 0)
        self.hi = np.max([p["hi"] for p in self.parts], 0)
        self._tri = self._verts = None
        # the lowest points: on the faces too, not only corners (a bench top overhangs its apron: its four bottom
        # corners are over nothing)
        band = self.lo[1] + rules["bottom_band_m"]
        low = [_tris(p)[p["tlo"][:, 1] <= band] for p in self.parts if p["lo"][1] <= band]
        low = surface_points(np.concatenate(low), 200)
        self.bottom_pts = _sample(low[low[:, 1] <= band], 200)

    @property
    def tri(self) -> np.ndarray:
        if self._tri is None:
            self._tri = np.concatenate([_tris(p) for p in self.parts])
        return self._tri

    @property
    def verts(self) -> np.ndarray:
        if self._verts is None:
            self._verts = np.concatenate([p["verts"] for p in self.parts])
        return self._verts

    def near(self, lo, hi, pad: float) -> np.ndarray:
        """Its triangles whose boxes come within ``pad`` of the box lo..hi."""
        lo, hi = np.asarray(lo, float) - pad, np.asarray(hi, float) + pad
        out = [_tris_near(p, lo, hi, 0.0) for p in self.parts if np.all(p["hi"] >= lo) and np.all(p["lo"] <= hi)]
        return np.concatenate(out) if out else np.zeros((0, 3, 3))


def supporters(body: Body, parts: list, part_lo: np.ndarray, part_hi: np.ndarray, own: set, rules: dict) -> dict:
    """Parts under a body's lowest points: {part index: share of the points it carries}. A part carries a point when its
    surface lies right under it (``support_below_m``) or a little above (``support_above_m``; ``sunk_m`` for ground).
    A narrow support under a wide body (a table's column under its top: 0.5% of the top's underside, missed by 200 rays
    down) is found the other way when nothing else carries it, by rays up from the support's own top: it carries a token
    share (``thin_share``). Not for structure parts."""
    return _supporters(body, parts, part_lo, part_hi, own, rules, rules["support_above_m"])


def _supporters(body: Body, parts: list, part_lo: np.ndarray, part_hi: np.ndarray, own: set, rules: dict,
                above: float) -> dict:
    below, sunk = rules["support_below_m"], rules["sunk_m"]
    y = body.lo[1]
    cand = np.nonzero((part_hi[:, 1] >= y - below) & (part_lo[:, 1] <= y + sunk)
                      & (part_lo[:, 0] <= body.hi[0]) & (part_hi[:, 0] >= body.lo[0])
                      & (part_lo[:, 2] <= body.hi[2]) & (part_hi[:, 2] >= body.lo[2]))[0]
    out, near_tris = {}, {}
    for i in cand:
        if int(i) in own:
            continue
        up = sunk if parts[i]["ground"] else above
        tri = _tris_near(parts[i], [body.lo[0], y - below, body.lo[2]], [body.hi[0], y + up, body.hi[2]], 0.0)
        if not len(tri):
            continue
        hit = _hits_in_band(body.bottom_pts, tri, below, up)
        if hit.any():
            out[int(i)] = float(hit.mean())
        elif not parts[i]["structure"]:
            near_tris[int(i)] = (tri, up)
    if out or (body.hi - body.lo).max() > rules["structure_side_m"]:
        return out
    # Nothing under its lowest points: from below then, points on the parts' faces in the band under the body's lowest
    # faces. Not for structure (a conduit running up a wall into a junction box does not carry the box), and only for a
    # body nothing else carries (every body asked so: PH27 took 51 s instead of 10).
    for i, (tri, up) in near_tris.items():
        pts = surface_points(tri, 200)
        pts = pts[(pts[:, 1] >= y - below) & (pts[:, 1] <= y + up) & np.all(pts[:, [0, 2]] >= body.lo[[0, 2]], 1)
                  & np.all(pts[:, [0, 2]] <= body.hi[[0, 2]], 1)]
        if len(pts):
            low = body.near([body.lo[0], y, body.lo[2]], [body.hi[0], y + below + up, body.hi[2]], 0.0)
            if len(low):
                with np.errstate(invalid="ignore"):
                    d = heights_at(pts[:, [0, 2]], low) - pts[:, 1:2]
                if np.any((d >= -up) & (d <= below)):
                    out[i] = rules["thin_share"]
    return out


def _open_above(item: Body, base: Body) -> bool:
    """Whether nothing of ``base`` lies over ``item`` (a cup on a table; not a binder on a bookcase's shelf board)."""
    lo, hi = item.lo, item.hi
    c, inset = (lo + hi) / 2, 0.25 * (hi - lo)
    xz = np.array([[c[0], c[2]], [lo[0] + inset[0], lo[2] + inset[2]], [hi[0] - inset[0], lo[2] + inset[2]],
                   [lo[0] + inset[0], hi[2] - inset[2]], [hi[0] - inset[0], hi[2] - inset[2]]])
    tri = base.near([lo[0], lo[1] + 0.05, lo[2]], [hi[0], np.inf, hi[2]], 0.0)
    if not len(tri):
        return True
    return not np.any(heights_at(xz, tri) > lo[1] + 0.05)


def _footprint_inside(item: Body, base: Body, slack: float = 0.02) -> bool:
    return bool(np.all(item.lo[[0, 2]] >= base.lo[[0, 2]] - slack) and np.all(item.hi[[0, 2]] <= base.hi[[0, 2]] + slack))


def _overhangs(item: Body, base: Body) -> bool:
    """Whether ``item`` is held up by something narrower than itself that it reaches past (a seat on its stem, a top on
    its legs, a shelf on its casters), not an item over a table's edge (a cable hanging off a bench)."""
    area = (item.hi - item.lo)[[0, 2]].prod(), (base.hi - base.lo)[[0, 2]].prod()
    return bool(not _footprint_inside(item, base) and area[1] < area[0])


def _contained(item: Body, base: Body, slack: float = 0.01) -> bool:
    """Whether ``item`` lies wholly inside ``base``'s box (loose parts in a bin, tools in a case: they go with it), its top
    under the base's top (a phone lying on a table is not in it)."""
    return bool(np.all(item.lo >= base.lo - slack) and np.all(item.hi <= base.hi + slack) and item.hi[1] <= base.hi[1])


def surface_points(tri: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """The triangles' corners and ``n`` points spread over them by area (a post through a shelf board touches it along
    its sides, far from any vertex of either)."""
    if not len(tri):
        return np.zeros((0, 3))
    corners = _sample(tri.reshape(-1, 3), n)
    a = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if a.sum() <= 0:
        return corners
    rng = np.random.default_rng(seed)
    k = rng.choice(len(tri), n, p=a / a.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[k]
    return np.concatenate([corners, t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])])


def contact(a: Body, b: Body, tol: float, above: float | None = None, n: int = 400) -> float:
    """Share of a's surface samples (within b's box) that lie within ``tol`` of b's surface, and of b's within a's: the
    larger. ``above``: only a's samples this far above a's bottom count (contact on a's sides, not under it)."""
    best = 0.0
    for x, y, first in ((a, b, True), (b, a, False)):
        if above is not None and not first:
            continue
        tri = x.near(y.lo, y.hi, tol)
        if above is not None:
            tri = tri[tri[:, :, 1].max(1) > x.lo[1] + above]
        if not len(tri):
            continue
        pts = surface_points(tri, n)
        pts = pts[np.all((pts >= y.lo - tol) & (pts <= y.hi + tol), 1)]
        if above is not None:
            pts = pts[pts[:, 1] > x.lo[1] + above]
        if not len(pts):
            continue
        near_tri = y.near(pts.min(0), pts.max(0), tol)
        if len(near_tri):
            best = max(best, float(near(pts, near_tri, tol).mean()))
    return best


def _attached(item: Body, base: Body, tol: float, band: float = 0.04, share: float = 0.05) -> bool:
    """Whether ``item`` touches ``base`` (within ``tol``) above its own lowest ``band`` (half its height when lower, at
    least twice ``tol``) over at least ``share`` of its surface there: fixed to its side (a front panel on its case, a
    knob on a front panel), not resting on it, nor lying beside it on the same bench (a foot grazing a screwdriver)."""
    return contact(item, base, tol, above=max(min(band, 0.5 * float(item.hi[1] - item.lo[1])), 2 * tol)) >= share


def group(parts: list, rules: dict = RULES) -> list[list[int]]:
    """Objects as lists of part indices (file order), see the module doc. Sets the parts' flags (_part_flags)."""
    _part_flags(parts, rules)
    keys = _seed_keys(parts)
    comps = _components(parts, keys, rules["touch_m"])
    part_lo = np.array([p["lo"] for p in parts])
    part_hi = np.array([p["hi"] for p in parts])
    comp_of = {i: c for c, idx in enumerate(comps) for i in idx}
    bodies = [Body(parts, idx, rules) for idx in comps]
    seed = [keys[idx[0]] for idx in comps]
    structure = [all(parts[i]["structure"] for i in idx) for idx in comps]
    large = [any(parts[i]["large"] for i in idx) for idx in comps]
    piece = [all(parts[i]["piece"] for i in idx) for idx in comps]
    names = [{w for i in idx for w in words(parts[i]["label"])} for idx in comps]
    kinds = [_word_class(parts, idx)[0] for idx in comps]
    sup = [supporters(b, parts, part_lo, part_hi, set(idx), rules) for b, idx in zip(bodies, comps)]

    def may_join(a: int, b: int, attached: bool = False) -> bool:
        if structure[a] or structure[b]:
            return False
        if "object" in (seed[a][0], seed[b][0]):      # the author's hierarchy says what an object is
            return False
        if kinds[a] and kinds[b] and kinds[a] != kinds[b] and not (piece[a] or piece[b]):
            return False   # their words disagree: a bin and the stones (or the shrub) around it are two things
        # two assets stay two (a case on a cart), unless one is fixed to the other (an asset of knobs, of wear decals)
        return attached or seed[a] == seed[b] or not (seed[a][0] == "asset" and seed[b][0] == "asset")

    uf = _Union(len(comps))
    tol = rules["contact_m"]
    lo = np.array([b.lo for b in bodies])
    hi = np.array([b.hi for b in bodies])
    for c in range(len(comps)):
        if structure[c]:
            continue
        if piece[c]:
            # a part (a bolt, a caster, a key, a decal) belongs to one body: of those it touches, the one whose label
            # shares most words with its own (a screwdriver's handle goes with its shaft, not the bench it lies on),
            # then the one it touches most. Joined to everything it touched, a row of punched slots on a rack's posts
            # took the trolley parked beside it.
            best, best_o = (0, 0.0), None
            for o in np.nonzero(np.all((lo <= bodies[c].hi + tol) & (hi >= bodies[c].lo - tol), 1))[0]:
                o = int(o)
                if o != c and seed[o] != seed[c] and may_join(c, o, attached=True):   # not its own kind: key to key
                    touch = contact(bodies[c], bodies[o], tol)
                    if touch > 0.0:
                        score = (len(names[c] & names[o]), touch)
                        if score > best:
                            best, best_o = score, o
            if best_o is not None:
                uf.join(c, best_o)
            continue
        share = defaultdict(float)     # the share of its lowest points each body (or the structure) carries
        for i, s in sup[c].items():
            key = -1 if parts[i]["structure"] else comp_of[i]
            share[key] = max(share[key], s)
        for u in [k for k in share if k >= 0]:
            # held up by another: one body with it when fixed to it (touches it above its own bottom: a knob on a front
            # panel), when it overhangs it (a seat on its stem, a top on its legs) or, not an asset of its own, lies
            # inside it (a binder on a shelf board); an item that only sits on its top (a cup on a table, a case on a
            # cart) stays apart. A body under a small corner of something that stands on a third (a keyboard on the
            # bench with one edge on a circuit board lying there) does not hold it up.
            if not may_join(c, u, attached=True):
                continue
            if _attached(bodies[c], bodies[u], rules["touch_m"]) or _contained(bodies[c], bodies[u]):
                uf.join(c, u)
            elif share[u] < 0.5 and max([s for k, s in share.items() if k != u] or [0.0]) >= 0.5:
                continue
            elif may_join(c, u) and (_overhangs(bodies[c], bodies[u])
                                     or (seed[c][0] != "asset" and _footprint_inside(bodies[c], bodies[u])
                                         and not _open_above(bodies[c], bodies[u]))):
                uf.join(c, u)
        if not sup[c] and not large[c]:
            # held by nothing from below: it hangs on what it touches (a shelf board between a trolley's uprights, a
            # stool's stem in its hub and under its seat), or lies inside it (parts stored in a carton)
            for o in np.nonzero(np.all((lo <= bodies[c].hi + tol) & (hi >= bodies[c].lo - tol), 1))[0]:
                o = int(o)
                if o != c and not large[o] and may_join(c, o, attached=True) and uf.find(c) != uf.find(o) \
                        and (_contained(bodies[c], bodies[o]) or contact(bodies[c], bodies[o], tol) > 0.0):
                    uf.join(c, o)
    objects = defaultdict(list)
    for c, idx in enumerate(comps):
        objects[uf.find(c)] += idx
    out = [sorted(v, key=lambda i: parts[i]["order"]) for v in objects.values()]
    return sorted(out, key=lambda v: parts[v[0]]["order"])


# ---------------------------------------------------------------- features
def _mass(parts: list, idx: list[int], body: Body) -> dict:
    """Volume, mass and centre of mass (Mitsuba axes). The volume is the closed mesh's (divergence theorem) when the
    object's surface is closed (98% of its edges paired) and not larger than its convex hull, else half its hull's.
    Each part weighs its share (by area) of that volume, or its area as a shell of its material's wall thickness when
    that is lighter."""
    from scipy.spatial import ConvexHull, QhullError

    # in double precision about the object's corner: the meshes are single (half the memory: PH43 is 9 M triangles) and
    # the divergence sum of a bin 500 m from the origin cancels terms of 1e8
    origin = body.lo.astype(np.float64)
    tri = body.tri.astype(np.float64) - origin
    closed = closed_fraction(tri)
    try:
        hull = ConvexHull(_unique_points(body.verts.astype(np.float64) - origin))
        hull_v = float(hull.volume)
    except (QhullError, ValueError):
        hull, hull_v = None, 0.0
    signed = mesh_volume(tri)
    if closed >= 0.98 and 0.0 < abs(signed) <= hull_v * 1.05 + 1e-9:
        volume, method = abs(signed), "closed mesh"
    else:
        volume, method = 0.5 * hull_v, "half the convex hull"
    areas = {i: _area(_tris(parts[i])) for i in idx}
    total = sum(areas.values()) or 1.0
    mass, by_material = 0.0, Counter()
    for i in idx:
        name = material_of(parts[i]["material"], parts[i]["label"])
        rho, wall = _DENSITY[name]
        m = rho * min(volume * areas[i] / total, areas[i] * wall)
        mass += m
        by_material[name] += m
    cross = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))
    if method == "closed mesh" and abs(cross.sum()) > 1e-12:
        com = (cross[:, None] * tri.sum(1)).sum(0) / (4.0 * cross.sum())   # tetrahedra from the origin
    else:
        a = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        com = (a[:, None] * tri.mean(1)).sum(0) / max(a.sum(), 1e-12)
    com = com + origin
    return {"volume_m3": volume, "volume_method": method, "closed": closed, "mass_kg": mass, "com": com,
            "materials": by_material, "area_m2": total, "hull_m3": hull_v}


def _area(tri: np.ndarray) -> float:
    return float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())


def footprint(verts: np.ndarray) -> float:
    """Area (m^2) of the convex hull of the points seen from above (a turned box is not its axis-aligned box)."""
    from scipy.spatial import ConvexHull, QhullError

    flat = verts * [1.0, 0.0, 1.0]
    xz = _unique_points(flat)[:, [0, 2]]
    try:
        return float(ConvexHull(xz).volume)
    except (QhullError, ValueError):
        span = verts.max(0) - verts.min(0)
        return float(span[0] * span[2])


_BOXES: dict = {}


def _boxes(parts: list, idx: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """The boxes of a list of parts as two arrays (cached per list: the ground and wall lists are asked once per object)."""
    key = (id(parts), id(idx), len(idx))
    if key not in _BOXES:
        if len(_BOXES) > 8:
            _BOXES.clear()
        lo = np.array([parts[i]["lo"] for i in idx]).reshape(-1, 3)
        _BOXES[key] = (lo, np.array([parts[i]["hi"] for i in idx]).reshape(-1, 3))
    return _BOXES[key]


def floor_under(body: Body, parts: list, ground: list[int], rules: dict) -> float | None:
    """The highest ground surface under the object's footprint (centre and four inset points), at most ``sunk_m`` above
    its bottom: the floor it stands on, or the floor under the table it is on."""
    lo, hi = body.lo, body.hi
    c, inset = (lo + hi) / 2, 0.25 * (hi - lo)
    xz = np.array([[c[0], c[2]], [lo[0] + inset[0], lo[2] + inset[2]], [hi[0] - inset[0], lo[2] + inset[2]],
                   [lo[0] + inset[0], hi[2] - inset[2]], [hi[0] - inset[0], hi[2] - inset[2]]])
    best = None
    g_lo, g_hi = _boxes(parts, ground)
    hit = (g_lo[:, 1] <= lo[1] + rules["sunk_m"]) & (g_hi[:, 0] >= lo[0]) & (g_lo[:, 0] <= hi[0]) \
        & (g_hi[:, 2] >= lo[2]) & (g_lo[:, 2] <= hi[2])
    for i in np.asarray(ground, np.int64)[hit]:
        p = parts[i]
        tri = _tris_near(p, [lo[0], -np.inf, lo[2]], [hi[0], lo[1] + rules["sunk_m"], hi[2]], 0.0)
        if not len(tri):
            continue
        h = heights_at(xz, tri)
        h = h[np.isfinite(h) & (h <= lo[1] + rules["sunk_m"])]
        if len(h) and (best is None or h.max() > best):
            best = float(h.max())
    return best


def wall_contact(body: Body, parts: list, walls: list[int], own: set, rules: dict) -> dict:
    """Which structure (or large parts) the object touches: {"wall": [labels], "ceiling": [labels]}. A wall is a face
    within 45 degrees of vertical; a ceiling a horizontal face of structure (only) at the object's top."""
    tol = rules["contact_m"]
    out = {"wall": set(), "ceiling": set()}
    w_lo, w_hi = _boxes(parts, walls)
    hit = np.all((w_lo <= body.hi + tol) & (w_hi >= body.lo - tol), 1)
    for i in np.asarray(walls, np.int64)[hit]:
        if int(i) in own:
            continue
        p = parts[i]
        tri = _tris_near(p, body.lo, body.hi, tol)
        if not len(tri):
            continue
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        ny = np.abs(n[:, 1]) / np.maximum(np.linalg.norm(n, axis=1), 1e-18)
        v = body.verts[np.all((body.verts >= p["lo"] - tol) & (body.verts <= p["hi"] + tol), 1)]
        if not len(v):
            continue
        v = _sample(v, 1500)
        side = tri[ny < 0.7]
        if len(side) and near(v, side, tol).any():
            out["wall"].add(p["label"])
        # a ceiling: a horizontal face of structure at or over the object's top (not the table top a flat thing lies on,
        # not a large shelf board or awning over it: a case in a rack and an A-frame sign under a canopy are free)
        if not p["structure"]:
            continue
        top = tri[(ny >= 0.7) & (tri[:, :, 1].min(1) >= body.hi[1] - tol)
                  & (tri[:, :, 1].min(1) > body.lo[1] + 0.5 * (body.hi[1] - body.lo[1]))]
        if len(top) and near(v[v[:, 1] >= body.hi[1] - tol], top, tol).any():
            out["ceiling"].add(p["label"])
    return {k: sorted(v) for k, v in out.items()}


def _sim(v) -> list:
    """Mitsuba (X, Y up, Z) -> the simulator's world (x = X, y = -Z, z = Y), rounded to 0.1 mm."""
    v = np.asarray(v, np.float64)
    return np.round(np.stack([v[..., 0], -v[..., 2], v[..., 1]], -1), 4).tolist()


def convex_hull(verts: np.ndarray, centre: np.ndarray, limit: int) -> dict:
    """Convex hull of the points (Mitsuba axes) with at most ``limit`` vertices (farthest-point picks among the hull's
    vertices, hulled again: it lies inside the full hull), as {"vertices" (sim frame, relative to ``centre``),
    "faces" (outward, counter-clockwise)}. A flat object (no volume) is joggled (QJ)."""
    from scipy.spatial import ConvexHull, QhullError

    pts = _unique_points(verts)
    try:
        hull = ConvexHull(pts)
    except (QhullError, ValueError):
        hull = ConvexHull(pts + np.random.default_rng(0).normal(0, 1e-4, pts.shape), qhull_options="QJ")
    hv = pts[hull.vertices]
    if len(hv) > limit:
        pick = [int(np.argmax(np.linalg.norm(hv - hv.mean(0), axis=1)))]
        d = np.linalg.norm(hv - hv[pick[0]], axis=1)
        while len(pick) < limit:
            k = int(np.argmax(d))
            pick.append(k)
            d = np.minimum(d, np.linalg.norm(hv - hv[k], axis=1))
        hv = hv[pick]
        try:
            hull = ConvexHull(hv)
        except (QhullError, ValueError):
            hull = ConvexHull(hv, qhull_options="QJ")
        hv = hv[hull.vertices] if len(hull.vertices) < len(hv) else hv
        hull = ConvexHull(hv, qhull_options="QJ") if len(hv) < 4 else ConvexHull(hv)
    else:
        remap = {int(v): k for k, v in enumerate(hull.vertices)}
        faces = np.array([[remap[int(a)] for a in s] for s in hull.simplices])
        return _hull_out(hv, faces, centre)
    return _hull_out(hv, hull.simplices, centre)


def _hull_out(hv: np.ndarray, faces: np.ndarray, centre: np.ndarray) -> dict:
    mid = hv.mean(0)
    f = np.array(faces, np.int64)
    n = np.cross(hv[f[:, 1]] - hv[f[:, 0]], hv[f[:, 2]] - hv[f[:, 0]])
    flip = (n * (hv[f].mean(1) - mid)).sum(1) < 0
    f[flip] = f[flip][:, ::-1]
    used = np.unique(f)
    remap = -np.ones(len(hv), np.int64)
    remap[used] = np.arange(len(used))
    rel = np.array(_sim(hv[used])) - np.array(_sim(centre))
    return {"vertices": np.round(rel, 4).tolist(), "faces": remap[f].tolist()}


# ---------------------------------------------------------------- classification
def classify(f: dict, rules: dict = RULES) -> tuple[bool, float, str]:
    """(movable, confidence, reason) of an object's features (``size_m`` in Mitsuba axes: x, height, z). The rules, in
    order, the first that applies deciding: fixed for a structure word, too large, flat (a decal), tiny or weightless,
    holding up structure, touching the ceiling, too heavy, nothing under it (mounted on a wall, or hanging), a
    fixed-furniture word; movable for a movable word on a support; else movable for a free-standing floor object (low,
    small, light, off the walls) and a small light item on top of something, fixed for anything else."""
    cls, kw = f["class"], f["keyword"]
    sx, sy, sz = f["size_m"]
    side, fp, mass = max(sx, sy, sz), f["footprint_m2"], f["mass_kg"]
    on, under = f["support"]["kind"], f["support"]["on"]
    walls, ceiling = f["touches"]["wall"], f["touches"]["ceiling"]
    what = f"'{kw}'" if kw else "no keyword"
    if cls == "structure":
        return False, 0.95, f"structure: {what} in '{f['label']}'"
    if side > rules["large_side_m"] or fp > rules["large_footprint_m2"]:
        return False, 0.9, f"large: {sx:.2f} x {sz:.2f} m, {sy:.2f} m tall, footprint {fp:.2f} m2"
    if min(sx, sy, sz) < rules["flat_m"]:
        return False, 0.8, f"flat ({min(sx, sy, sz) * 1000:.1f} mm): a decal, a mark or a sheet, nothing to push"
    if side < rules["small_m"] or mass < rules["light_kg"]:
        return False, 0.7, f"tiny ({side * 100:.1f} cm, {mass * 1000:.0f} g): a detail, not a body worth simulating"
    if f["holds_up"]:
        return False, 0.85, f"holds up {', '.join(f['holds_up'][:3])}"
    if ceiling:
        return False, 0.9, f"touches the ceiling ({', '.join(ceiling[:2])})"
    if mass > rules["max_mass_kg"]:
        return False, 0.8, f"heavy: ~{mass:.0f} kg (> {rules['max_mass_kg']:.0f})"
    if on == "none":
        if walls:
            return False, 0.85, f"nothing under it, touches {', '.join(walls[:2])}: mounted"
        return False, 0.6, "nothing under it: hangs or floats"
    if cls == "furniture":
        return False, 0.8, f"fixed furniture: {what}"
    where = "the floor" if on == "floor" else ", ".join(under[:2]) or on
    if cls == "movable":
        conf = (0.9 if on == "floor" else 0.8) - (0.15 if walls else 0.0)
        return True, round(conf, 2), f"{what} on {where}, ~{mass:.1f} kg" + (f", touches {walls[0]}" if walls else "")
    if on == "floor":
        bad = [w for w, ok in ((f"{sy:.2f} m tall", sy < rules["free_height_m"]),
                               (f"footprint {fp:.2f} m2", fp < rules["free_footprint_m2"]),
                               (f"~{mass:.0f} kg", mass < rules["free_mass_kg"]),
                               (f"touches {', '.join(walls[:2])}", not walls)) if not ok]
        if not bad:
            return True, 0.6, f"free-standing on the floor, no keyword: {sy:.2f} m tall, footprint {fp:.2f} m2, ~{mass:.1f} kg"
        return False, 0.6, f"floor object, no keyword, not free: {'; '.join(bad)}"
    if mass < rules["item_mass_kg"] and side < rules["item_side_m"]:
        return True, 0.55, f"small item on {where}, no keyword: {side:.2f} m, ~{mass:.1f} kg"
    return False, 0.6, f"on {where}, no keyword, not a small item: {side:.2f} m, ~{mass:.0f} kg"


# ---------------------------------------------------------------- the scene
def source_sha1(scene_dir, xml: str = "base.xml") -> str:
    """What the analysis reads: the scene file as converted (before any merge; its mesh files are named by content)."""
    return hashlib.sha1(original_text(Path(scene_dir), xml).encode()).hexdigest()


def rules_sha1(rules: dict = RULES) -> str:
    return hashlib.sha1(json.dumps([VERSION, rules, sorted(STRUCTURE), sorted(FURNITURE), sorted(MOVABLE), sorted(MODIFIERS),
                                    sorted(GROUND), sorted(PIECES), MATERIALS], sort_keys=True).encode()).hexdigest()[:16]


def _summary(labels: list[str]) -> list[str]:
    return [f"{s} x{n}" if n > 1 else s for s, n in Counter(label_stem(x) for x in labels).most_common(6)]


def _word_class(parts: list, idx: list[int]) -> tuple[str | None, str | None]:
    """The class and keyword of an object: its parts' label classes, each weighted by its area (a cart's shelves,
    uprights and casters all say cart; a stool's seat, stem and spokes all say stool)."""
    votes, keyword = Counter(), {}
    for i in idx:
        cls, kw = parts[i]["class"], parts[i]["keyword"]
        a = _area(_tris(parts[i]))
        votes[cls] += a          # parts without a word vote too: a table whose phone says "phone" is not a phone
        keyword.setdefault(cls, Counter())[kw] += a
    cls = votes.most_common(1)[0][0]
    return (cls, keyword[cls].most_common(1)[0][0]) if cls else (None, None)


def _main_label(parts: list, idx: list[int]) -> str:
    """The label stem of the object's biggest parts (by box volume)."""
    main = Counter()
    for i in idx:
        main[label_stem(parts[i]["label"])] += float(np.prod(np.maximum(parts[i]["hi"] - parts[i]["lo"], 1e-3)))
    return main.most_common(1)[0][0]


def analyse(scene_dir, xml: str = "base.xml", rules: dict | None = None) -> dict:
    """The physics.json content of a scene (see the module doc)."""
    rules = dict(RULES, **(rules or {}))
    started = time.monotonic()
    scene_dir = Path(scene_dir)
    parts = read_parts(scene_dir, xml)
    groups = group(parts, rules)
    t_group = time.monotonic() - started
    part_lo = np.array([p["lo"] for p in parts]).reshape(-1, 3)
    part_hi = np.array([p["hi"] for p in parts]).reshape(-1, 3)
    obj_of = {i: k for k, idx in enumerate(groups) for i in idx}
    ground = [i for i, p in enumerate(parts) if p["ground"]]
    walls = [i for i, p in enumerate(parts) if (p["structure"] or p["large"]) and not p["ground"]]
    ids = [parts[idx[0]]["id"] for idx in groups]
    labels = [_main_label(parts, idx) for idx in groups]
    # what each structure part stands on: an object under a counter top or a wall holds it up
    held = defaultdict(set)
    for i, p in enumerate(parts):
        if p["structure"] and not p["ground"]:
            for j in supporters(Body(parts, [i], rules), parts, part_lo, part_hi, {i}, rules):
                if not parts[j]["structure"]:
                    held[obj_of[j]].add(label_stem(p["label"]))
    objects = []
    for k, idx in enumerate(groups):
        body, own = Body(parts, idx, rules), set(idx)
        sup = supporters(body, parts, part_lo, part_hi, own, rules)
        carried = defaultdict(float)
        for i, share in sup.items():
            key = "floor" if parts[i]["ground"] else "structure" if parts[i]["structure"] else obj_of[i]
            carried[key] = max(carried[key], share)
        on_objects = sorted((x for x in carried if x not in ("floor", "structure")), key=lambda x: -carried[x])
        # on the floor when some of it stands on ground (a trolley with two casters on a mat is still on the floor)
        if carried.get("floor", 0.0) >= 0.05:
            kind = "floor"
        elif on_objects:
            kind = "object"
        elif carried.get("structure"):
            kind = "structure"
        else:
            kind = "floor" if carried.get("floor") else "none"
        if kind == "object":
            under = [labels[x] for x in on_objects]
        elif kind == "structure":
            under = sorted({label_stem(parts[i]["label"]) for i in sup if parts[i]["structure"] and not parts[i]["ground"]})
        else:
            under = []
        floor_y = floor_under(body, parts, ground, rules)
        if kind == "floor" and (floor_y is None or floor_y < body.lo[1] - rules["support_below_m"]):
            floor_y = float(body.lo[1])
        cls, kw = _word_class(parts, idx)
        size = body.hi - body.lo
        # structure and large objects are fixed whatever they weigh or touch: their volume (hulls of whole facades) and
        # contacts are not worked out (PH46's 8.5 M triangles are mostly buildings, buses and trees)
        weigh = cls != "structure" and size.max() <= rules["large_side_m"]
        m = _mass(parts, idx, body) if weigh else None
        fp = footprint(body.verts) if weigh else float(size[0] * size[2])
        touches = wall_contact(body, parts, walls, own, rules) if weigh else {"wall": [], "ceiling": []}
        f = {"label": labels[k], "class": cls, "keyword": kw, "size_m": [float(size[0]), float(size[1]), float(size[2])],
             "footprint_m2": fp, "mass_kg": m["mass_kg"] if m else None, "support": {"kind": kind, "on": under},
             "touches": touches, "holds_up": sorted(held.get(k, ()))}
        movable, conf, reason = classify(f, rules)
        reach = bool(movable and floor_y is not None and body.lo[1] - floor_y <= rules["reach_m"])
        out = {"id": ids[k], "parts": [parts[i]["id"] for i in idx], "label": labels[k],
               "labels": _summary([parts[i]["label"] for i in idx]), "movable": movable, "robot_reachable": reach,
               "confidence": conf, "reason": reason, "class": cls, "keyword": kw,
               "mass_kg": round(m["mass_kg"], 3) if m else None, "volume_m3": round(m["volume_m3"], 6) if m else None,
               "volume_method": m["volume_method"] if m else None,
               "material": max(m["materials"], key=m["materials"].get) if m and m["materials"] else None,
               "com": _sim(m["com"]) if m else None,
               "bbox": [_sim(body.lo * [1, 1, 0] + body.hi * [0, 0, 1]), _sim(body.hi * [1, 1, 0] + body.lo * [0, 0, 1])],
               "size_m": [round(float(size[0]), 4), round(float(size[2]), 4), round(float(size[1]), 4)],
               "footprint_m2": round(fp, 4),
               "support": {"kind": kind, "on": under[:4], "bottom_z": round(float(body.lo[1]), 4),
                           "floor_z": None if floor_y is None else round(floor_y, 4)},
               "touches": f["touches"], "holds_up": f["holds_up"]}
        if movable:
            # the friction of the material most of its surface is (by the scene's material name, as groundmap reads it)
            area = Counter()
            for i in idx:
                area[parts[i]["material"] or out["material"] or ""] += _area(_tris(parts[i]))
            name = area.most_common(1)[0][0]
            mu, why = material_friction(name)
            out["pose"] = {"position": _sim(m["com"]), "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}
            out["hull"] = convex_hull(body.verts, m["com"], rules["hull_vertices"])
            out["friction"] = {"mu": mu, "why": why, "material": name}
        objects.append(out)
    movable = [o for o in objects if o["movable"]]
    return {"version": VERSION, "tool": "tools/physics_tag.py", "scene": scene_dir.name, "xml": xml,
            "source_sha1": source_sha1(scene_dir, xml), "rules_sha1": rules_sha1(rules),
            "frame": "simulator world: x = X, y = -Z, z = Y of the scene files (Mitsuba, Y up); metres, kg; "
                     "hull vertices relative to pose.position (the centre of mass)",
            "rules": rules,
            "counts": {"parts": len(parts), "objects": len(objects), "movable": len(movable),
                       "robot_reachable": sum(o["robot_reachable"] for o in movable), "fixed": len(objects) - len(movable),
                       "seconds": round(time.monotonic() - started, 1), "grouping_seconds": round(t_group, 1)},
            "movable": [o["id"] for o in movable],
            "movable_parts": [p for o in movable for p in o["parts"]],
            "objects": objects}
