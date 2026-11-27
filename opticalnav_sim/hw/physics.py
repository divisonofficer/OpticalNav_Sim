"""Objects a robot drives into get pushed: a scene's movable objects (physics.json) as free bodies in MuJoCo, the robot
as a kinematic pusher.

    from opticalnav_sim.hw.physics import PushWorld
    push = PushWorld.from_scene(scene_dir, spec.footprint(), robot_height=0.4)     # objects="reachable" | "movable"
    moved = push.step(robot.pose, world.dt)     # {object id: 4x4}: the objects that moved since they were last returned
    push.contacts()                             # the object ids touching the robot now (a "pushed" indicator)
    push.poses()                                # every object's 4x4

A pose is a 4x4 in the simulator's world (x, y, z up; physics.json's frame): an object's CURRENT frame relative to its
INITIAL one (physics.json's pose), ``delta = current · initial⁻¹``, so a renderer moves the object's original vertices
by it. The first step() returns every body the build's settling moved (most by the few mm their hull floated).

The robot's own motion stays the drive model's (hw.drive). Here it is a prism of its footprint, ``robot_height`` tall,
set every substep where the drive model's pose says (interpolated over the step), colliding with the free bodies only:
it pushes them and is not pushed back, nor stopped by the static world. It pushes with at most ``push_force_n``: a body
jammed against a rack stops it (``lag``, ``stalled``: the drive model goes on, the collider waits) instead of being
squeezed out of the world (CS02's cart pushed into its rack went 66 cm up and turned 79 degrees without the limit).

What the world holds (``from_scene``; ``info`` has the counts and what was decided):

* free bodies: the chosen movable objects ("reachable": those at the robot's height, the default; "movable": all), and
  what lies on one of them, by physics.json's support and their boxes (the carton on CS02's cart goes with it). Each
  has its mass, an inertia from its hull, its friction (on casters or wheels it rolls: ``rolling_mu``), and collides
  as physics.json's convex hull, or as its parts' convex pieces when their volume fills less than ``compound_fill``
  of the hull (a cart's hull is a block from its casters to its top shelf: the carton would stand 3.7 cm higher on it
  than it sits between the cart's uprights);
* static colliders: a plane at the floor; each fixed part of the scene within ``margin_m`` of a free body's box (from
  the floor to ``above_m`` over the body's top), clipped to that region, as convex pieces (``convex_pieces``: the
  part's hull where that hull stays within ``convex_tol_m`` of the part's surface, else the part cut in two along its
  longest side, again). A piece that fills its box is a box geom (box-box contacts are cheap and steady; 71 of 134 on
  CS02), else a mesh. Left out: parts under ``min_side_m`` (a pin, a screw) unless they carry a free body, and parts
  inside a free body's box (a fixed carry handle on a free case would hold it down). A fixed part reaching into a
  free body's box is cut away there (a rack post through an open carton's flaps would throw the carton out). The
  movable objects not chosen are static, as their hulls;
* the robot.

Settling (``_settle``): bodies are first moved out of what they penetrate, then simulated ``settle_s`` so they rest on
what carries them; one that does more than come down was balanced in a way its hull does not stand, and is put back
where the scene has it. Then every body is asleep (MuJoCo's island sleeping) until something touches it: a scene at
rest costs the collision of the robot alone.

Measured on blend_REF_CS02 (tools/push_check.py; one core): built in 2.7 s (1.6 s of it reading the scene's meshes) with
14 free bodies (13 reachable and the carton on the cart) and 134 static pieces; 79 movable: 6.1 s, 218 pieces. At rest
a 50 ms step costs 0.3 ms (0.4 ms with 79), all asleep; awake (sleeping off) 2.8 ms, drifting at most 0.22 mm and 0.05
degrees in 5 s (14 ms with all 79 awake). Pushing: 0.3 to 1.1 ms a step.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SETTINGS = {
    "timestep_s": 0.002,       # 1 mm of robot travel per substep at 0.5 m/s
    "margin_m": 1.0,           # fixed parts this far around a free body's box are colliders: where a push can take it
    "above_m": 0.25,           # ... from the floor up to this over its top
    "min_side_m": 0.04,        # smaller fixed parts are left out (unless they carry a free body)
    "convex_tol_m": 0.01,      # a part's hull this close to its surface stands for it; else it is cut
    "split_depth": 4,          # at most 2^4 pieces a part
    "min_piece_m": 0.08,       # a piece this small is not cut again
    "box_fill": 0.97,          # a hull filling this share of its box is the box
    "max_static": 800,         # the nearest pieces are kept beyond this
    "floor_slack_m": 0.01,     # a ground part whose top is this close to the floor plane is the plane
    "rolling_mu": 0.1,         # an object on casters or wheels: their rolling and swivel resistance, as a friction
    "robot_mu": 0.5,
    "robot_bottom_m": 0.01,    # the robot's prism starts this far above its pose (it never touches the floor plane)
    "moved_m": 5e-4,           # step() reports an object once it moved this far (or turned moved_deg) since last
    "moved_deg": 0.1,
    "teleport_m": 0.5,         # a robot pose this far from the last one is a jump (a respawn), not interpolated
    "compound_fill": 0.6,      # a free body whose parts fill less of its hull than this collides as its parts
    "max_pieces": 16,
    "carve_m": 0.01,           # a fixed part reaching this far into a free body's box is cut away there
    "settle_s": 1.0,           # simulated at build: the bodies come to rest on what carries them ...
    "settle_shift_m": 0.01,    # ... and one that goes this far sideways, or turns settle_deg, is put back and held
    "settle_deg": 5.0,
    "sleep_tolerance": 5e-3,   # MuJoCo's sleep_tolerance (default 1e-4: CS02's bench cases rock at ~1e-3 for good)
    "push_force_n": 400.0,     # the most the robot pushes with: a Ranger Mini's grip, 64.5 kg x 9.81 x ~0.65
    "lag_m": 0.3,              # a collider this far behind the robot's pose jumps to it
}

ROLLING = ("caster", "castor", "wheel", "wheels", "casters", "castors")

# collision bits: static geoms meet free bodies, free bodies meet everything, the robot meets free bodies only
STATIC, FREE, ROBOT = 1, 2, 4


@dataclass
class FreeObject:
    id: str
    label: str
    mass: float
    mu: float
    vertices: np.ndarray            # hull, relative to the initial frame (sim world axes)
    faces: np.ndarray
    initial: np.ndarray             # 4x4: the object's frame at physics.json's pose
    rolls: bool = False
    pieces: list = field(default_factory=list)   # its parts' convex pieces (relative): collide instead of the hull


@dataclass
class StaticPiece:
    name: str                       # the part (or object) it comes from
    vertices: np.ndarray            # convex hull points, sim world
    mu: float = 0.6
    box: tuple | None = None        # (centre, half sizes, 3x3 rotation) when the piece is a box


@dataclass
class _Built:
    objects: list
    statics: list
    floor_z: float | None
    floor_mu: float = 0.6
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------- geometry
def _to_sim(v: np.ndarray) -> np.ndarray:
    """Mitsuba (X, Y up, Z) -> the simulator's world (x = X, y = -Z, z = Y)."""
    v = np.asarray(v, np.float64)
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], -1)


def _clip(tri: np.ndarray, axis: int, value: float, keep: int) -> np.ndarray:
    """Triangles (M x 3 x 3) clipped to the half space ``keep * (x[axis] - value) >= 0`` (Sutherland-Hodgman, a
    triangle at a time: one corner inside gives a triangle, two give a quad as two)."""
    if not len(tri):
        return tri
    s = keep * (tri[..., axis] - value)
    inside = s >= 0.0
    n = inside.sum(1)
    out = [tri[n == 3]]
    for count in (1, 2):
        sel = n == count
        if not sel.any():
            continue
        t, ss, ins = tri[sel], s[sel], inside[sel]
        # rotate the corners: the lone corner first (inside for one, outside for two), the order (orientation) kept
        first = np.argmax(ins if count == 1 else ~ins, axis=1)
        order = (first[:, None] + np.arange(3)[None]) % 3
        t = np.take_along_axis(t, order[..., None], 1)
        ss = np.take_along_axis(ss, order, 1)
        a, b, c = t[:, 0], t[:, 1], t[:, 2]
        sa, sb, sc = ss[:, 0:1], ss[:, 1:2], ss[:, 2:3]
        with np.errstate(divide="ignore", invalid="ignore"):
            if count == 1:      # a inside: a, (a..b), (a..c)
                p = a + (b - a) * np.nan_to_num(sa / (sa - sb))
                q = a + (c - a) * np.nan_to_num(sa / (sa - sc))
                out.append(np.stack([a, p, q], 1))
            else:               # a outside: (b..a), b, c, (c..a)
                p = b + (a - b) * np.nan_to_num(sb / (sb - sa))
                q = c + (a - c) * np.nan_to_num(sc / (sc - sa))
                out += [np.stack([p, b, c], 1), np.stack([p, c, q], 1)]
    return np.concatenate(out)


def clip_box(tri: np.ndarray, lo, hi) -> np.ndarray:
    for axis in range(3):
        tri = _clip(tri, axis, float(lo[axis]), 1)
        tri = _clip(tri, axis, float(hi[axis]), -1)
    return tri


def _subtract(cell, box) -> list:
    """The box ``cell`` (lo, hi) without ``box``: at most six boxes (below and above it, then its four sides)."""
    lo, hi = np.array(cell[0], float), np.array(cell[1], float)
    blo, bhi = np.maximum(box[0], lo), np.minimum(box[1], hi)
    if np.any(blo >= bhi):
        return [(lo, hi)]
    out = []
    for axis in (2, 0, 1):
        if blo[axis] > lo[axis]:
            h = hi.copy()
            h[axis] = blo[axis]
            out.append((lo.copy(), h))
        if bhi[axis] < hi[axis]:
            low = lo.copy()
            low[axis] = bhi[axis]
            out.append((low, hi.copy()))
        lo[axis], hi[axis] = blo[axis], bhi[axis]
    return out


def _hull(points: np.ndarray, thick: float = 0.004):
    """(hull points, scipy hull) of a point set; a flat set (no volume: a sheet, a clipped face) is given ``thick``
    behind it first, below it when it faces up (its top stays where it is)."""
    from scipy.spatial import ConvexHull, QhullError

    pts = np.unique(np.round(points, 5), axis=0)
    if len(pts) < 4:
        return None, None
    mid = pts.mean(0)
    _, sv, vt = np.linalg.svd(pts - mid, full_matrices=False)
    if sv[-1] / math.sqrt(len(pts)) < 1e-4 or np.ptp((pts - mid) @ vt[-1]) < 1e-3:
        n = vt[-1] * (1.0 if vt[-1][2] >= 0 else -1.0)
        if abs(n[2]) > 0.7:
            pts = np.concatenate([pts, pts - n * thick])
        else:
            pts = np.concatenate([pts - n * thick / 2, pts + n * thick / 2])
    try:
        h = ConvexHull(pts)
    except (QhullError, ValueError):
        try:
            h = ConvexHull(pts, qhull_options="QJ")
        except (QhullError, ValueError):
            return None, None
    return pts[h.vertices], h


def _as_box(points: np.ndarray, volume: float, fill: float):
    """(centre, half sizes, rotation) of the box a hull is, when it fills ``fill`` of it: the world axes' box, or one
    turned about z (by the footprint's principal axes)."""
    best = None
    xy = points[:, :2] - points[:, :2].mean(0)
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    yaw = math.atan2(vt[0, 1], vt[0, 0])
    for a in (0.0, yaw):
        c, s = math.cos(a), math.sin(a)
        r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        local = points @ r
        lo, hi = local.min(0), local.max(0)
        v = float(np.prod(np.maximum(hi - lo, 1e-9)))
        if best is None or v < best[0]:
            best = (v, r @ ((lo + hi) / 2), (hi - lo) / 2, r)
    if best is not None and volume >= fill * best[0]:
        return best[1], np.maximum(best[2], 1e-4), best[3]
    return None


def convex_pieces(tri: np.ndarray, lo, hi, settings: dict = SETTINGS, depth: int = 0, own=None) -> list:
    """A fixed part's triangles (sim world) within the box lo..hi as convex point sets: its hull where that hull stays
    within ``convex_tol_m`` of the part's surface (a box, a board, a post), else the box cut in two along its longest
    side and each half again. ``own``: the part's own box (a hull face on a cut is not held against the surface)."""
    from ..physics_objects import near, surface_points

    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    own = (lo, hi) if own is None else own
    tri = clip_box(tri, lo, hi)
    if not len(tri):
        return []
    pts = tri.reshape(-1, 3)
    plo, phi = pts.min(0), pts.max(0)
    hv, h = _hull(pts)
    if hv is None:
        return []
    tol = settings["convex_tol_m"]
    size = phi - plo
    if depth >= settings["split_depth"] or size.max() < settings["min_piece_m"]:
        return [(hv, h)]
    samples = surface_points(h.points[h.simplices], 200)
    cut = np.zeros(len(samples), bool)
    for axis in range(3):   # the faces where the box cut the part: no surface there to hold the hull to
        if lo[axis] > own[0][axis] + tol:
            cut |= np.abs(samples[:, axis] - lo[axis]) < tol
        if hi[axis] < own[1][axis] - tol:
            cut |= np.abs(samples[:, axis] - hi[axis]) < tol
    samples = samples[~cut]
    if not len(samples) or near(samples, tri, tol).mean() >= 0.95:
        return [(hv, h)]
    axis = int(np.argmax(size))
    mid = 0.5 * (plo[axis] + phi[axis])
    a_hi, b_lo = np.minimum(hi, phi), np.maximum(lo, plo)
    a_hi[axis], b_lo[axis] = mid, mid
    return (convex_pieces(tri, np.maximum(lo, plo), a_hi, settings, depth + 1, own)
            + convex_pieces(tri, b_lo, np.minimum(hi, phi), settings, depth + 1, own))


def hull_inertia(vertices: np.ndarray, faces: np.ndarray, mass: float) -> np.ndarray | None:
    """3x3 inertia about the origin of a solid of uniform density bounded by the triangles (outward), of ``mass``:
    tetrahedra from the origin, their second moments summed. None for a hull with no volume."""
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    total = float(vol.sum())
    if total <= 1e-9:
        return None
    s = a + b + c
    cov = (np.einsum("n,ni,nj->ij", vol, a, a) + np.einsum("n,ni,nj->ij", vol, b, b)
           + np.einsum("n,ni,nj->ij", vol, c, c) + np.einsum("n,ni,nj->ij", vol, s, s)) / 20.0
    inertia = np.trace(cov) * np.eye(3) - cov
    return inertia * (mass / total)


def _matrix(pose) -> np.ndarray:
    """A pose as a 4x4: a 4x4, something with ``matrix()`` (hw.Transform: ``robot.pose``) or (x, y, z, yaw_deg)."""
    if hasattr(pose, "matrix"):
        return np.asarray(pose.matrix(), np.float64)
    p = np.asarray(pose, np.float64)
    if p.shape == (4, 4):
        return p
    x, y, z, yaw = (list(p.ravel()) + [0.0] * 4)[:4]
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return np.array([[c, -s, 0.0, x], [s, c, 0.0, y], [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]])


def _quat(r: np.ndarray) -> np.ndarray:
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(r, np.float64).ravel())
    return q


def _rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _fmt(a) -> str:
    return " ".join(f"{float(v):.6g}" for v in np.asarray(a).ravel())


# ---------------------------------------------------------------- the scene
def _box(o: dict) -> tuple[np.ndarray, np.ndarray]:
    b = np.asarray(o["bbox"], np.float64)
    return b.min(0), b.max(0)


def _rides_on(o: dict, base: dict) -> bool:
    """Whether ``o`` lies on ``base``: physics.json says it stands on an object of that label (labels repeat: binders,
    cases), and it is there: over its footprint, its bottom between the base's bottom and just over its top."""
    if base["label"] not in ((o.get("support") or {}).get("on") or ()):
        return False
    (alo, ahi), (blo, bhi) = _box(o), _box(base)
    over = np.all(alo[:2] < bhi[:2]) and np.all(ahi[:2] > blo[:2])
    return bool(over and blo[2] <= alo[2] <= bhi[2] + 0.05)


def choose(data: dict, objects: str = "reachable") -> list[dict]:
    """The physics.json objects to make free bodies of: "reachable" (at the robot's height) or "movable" (all), and
    whatever movable object lies on one of them (a carton on a cart), over and over."""
    if objects not in ("reachable", "movable"):
        raise ValueError(f'objects is "reachable" or "movable", not {objects!r}')
    movable = [o for o in data["objects"] if o.get("movable") and o.get("hull")]
    chosen = [o for o in movable if objects == "movable" or o.get("robot_reachable")]
    while True:
        ids = {o["id"] for o in chosen}
        more = [o for o in movable if o["id"] not in ids and any(_rides_on(o, c) for c in chosen)]
        if not more:
            return [o for o in movable if o["id"] in ids]
        chosen += more


def free_object(o: dict, settings: dict = SETTINGS, parts: dict | None = None) -> FreeObject:
    """A physics.json object as a free body. ``parts`` (part id -> read_parts' part): its parts' own convex pieces
    when its one hull is a poor fit (compound_pieces)."""
    initial = np.eye(4)
    initial[:3, :3] = _rot(np.asarray(o["pose"].get("quat_wxyz", [1.0, 0.0, 0.0, 0.0]), np.float64))
    initial[:3, 3] = o["pose"]["position"]
    words = {w for lab in o.get("labels") or () for w in str(lab).lower().replace("_", " ").split()}
    rolls = bool(words & set(ROLLING))
    mu = float((o.get("friction") or {}).get("mu", 0.6))
    out = FreeObject(o["id"], o.get("label", o["id"]), max(float(o.get("mass_kg") or 0.0), 0.05),
                     min(mu, settings["rolling_mu"]) if rolls else mu,
                     np.asarray(o["hull"]["vertices"], np.float64), np.asarray(o["hull"]["faces"], np.int64), initial, rolls)
    if parts:
        out.pieces = compound_pieces(o, out, parts, settings)
    return out


def compound_pieces(o: dict, free: FreeObject, parts: dict, settings: dict = SETTINGS) -> list:
    """The convex pieces of an object's own parts (relative to its frame), when its one hull is a poor fit: their
    volume under ``compound_fill`` of the hull's. A cart's hull is a solid block from its casters to its top shelf (a
    carton on that shelf would stand on the block, 3.7 cm higher than it sits between the cart's uprights), a stool's a
    cone from its glides to its seat; an instrument case's hull is the case. At most ``max_pieces``, the largest, and
    always the lowest ones (casters, glides, feet: what it stands on)."""
    from scipy.spatial import ConvexHull, QhullError

    try:
        hull_v = float(ConvexHull(free.vertices).volume)
    except (QhullError, ValueError):
        return []
    lo, hi = _box(o)
    cut = dict(settings, split_depth=2)
    pieces = []
    for pid in o.get("parts") or ():
        p = parts.get(pid)
        if p is None:
            continue
        tri = _to_sim(p["verts"][p["faces"]].astype(np.float64))
        plo, phi = tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0)
        low = plo[2] <= lo[2] + 0.01
        if (phi - plo).max() < settings["min_side_m"] and not low:
            continue
        for hv, h in convex_pieces(tri, plo - 1e-3, phi + 1e-3, cut):
            pieces.append((float(h.volume), low, hv))
    if not pieces or sum(v for v, _, _ in pieces) >= settings["compound_fill"] * hull_v:
        return []
    low = [x for x in pieces if x[1]]
    rest = sorted((x for x in pieces if not x[1]), key=lambda x: -x[0])
    keep = low[:settings["max_pieces"]] + rest[:max(0, settings["max_pieces"] - len(low))]
    r, p = free.initial[:3, :3], free.initial[:3, 3]
    return [(hv - p) @ r for _, _, hv in keep]


def _boxes_of(objects: list) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([np.min(np.asarray(o["bbox"]), 0) for o in objects]).reshape(-1, 3)
    hi = np.array([np.max(np.asarray(o["bbox"]), 0) for o in objects]).reshape(-1, 3)
    return lo, hi


def build(scene_dir, objects: str = "reachable", settings: dict | None = None) -> _Built:
    """What a PushWorld of a scene holds (see the module doc): the free objects, the static pieces, the floor."""
    import time

    from .. import physics_objects as po
    from ..groundmap import material_friction

    settings = dict(SETTINGS, **(settings or {}))
    started = time.monotonic()
    scene_dir = Path(scene_dir)
    data = json.loads((scene_dir / po.PHYSICS).read_text())
    chosen = choose(data, objects)
    if not chosen:
        return _Built([], [], None, info={"free": 0, "static": 0})
    lo, hi = _boxes_of(chosen)
    floors = [o["support"].get("floor_z") for o in chosen if (o.get("support") or {}).get("floor_z") is not None]
    floor_z = float(min(floors)) if floors else float(lo[:, 2].min())
    # where a free body may go: around its box, from the floor to a little over its top
    margin = settings["margin_m"]
    reg_lo = lo - [margin, margin, 0.0]
    reg_hi = hi + [margin, margin, settings["above_m"]]
    reg_lo[:, 2] = floor_z - 0.05

    parts = po.read_parts(scene_dir, data.get("xml", "base.xml"))
    po._part_flags(parts, dict(po.RULES, **(data.get("rules") or {})))
    t_read = time.monotonic() - started
    by_id = {p["id"]: p for p in parts}
    free = [free_object(o, settings, by_id) for o in chosen]
    movable_parts = set(data.get("movable_parts") or [])
    inset_lo, inset_hi = lo + settings["carve_m"], hi - settings["carve_m"]
    # what carries a free body (kept whatever its size): the fixed objects it stands on, by label, and structure parts,
    # by label stem
    carriers = {lab for o in chosen for lab in ((o.get("support") or {}).get("on") or ())}
    carrier_parts = {p for o in data["objects"] if not o.get("movable") and o.get("label") in carriers for p in o["parts"]}
    statics, skipped, carved = [], {"small": 0, "detail": 0, "floor": 0, "far": 0}, set()
    for p in parts:
        if p["id"] in movable_parts:
            continue
        a, b = _to_sim(p["lo"]), _to_sim(p["hi"])
        plo, phi = np.minimum(a, b), np.maximum(a, b)
        meets = np.all((reg_hi >= plo) & (reg_lo <= phi), 1)
        if not meets.any():
            skipped["far"] += 1
            continue
        if p.get("ground") and phi[2] <= floor_z + settings["floor_slack_m"]:
            skipped["floor"] += 1        # the plane
            continue
        carrier = p["id"] in carrier_parts or po.label_stem(p["label"]) in carriers
        if (phi - plo).max() < settings["min_side_m"] and not carrier:
            skipped["small"] += 1
            continue
        mid = (plo + phi) / 2
        inside = np.all((mid >= lo - 0.02) & (mid <= hi + 0.02), 1) & np.all(phi - plo <= hi - lo + 0.02, 1)
        if inside.any() and not carrier:
            skipped["detail"] += 1       # a fixed detail of a free body (its carry handle): it would hold it
            continue
        # clipped to the regions it meets (their box), and to the floor
        clo = np.maximum(plo, reg_lo[meets].min(0))
        chi = np.minimum(phi, reg_hi[meets].max(0))
        tri = _to_sim(p["verts"][p["faces"]].astype(np.float64))
        mu = material_friction(p.get("material") or p["label"])[0]
        # a fixed part through a free body's box (a rack post through an open carton's flaps): cut away there, or the
        # body starts inside it and is thrown out
        cells = [(clo, chi)]
        for k in np.nonzero(np.all((inset_hi >= plo) & (inset_lo <= phi), 1))[0]:
            if len(clip_box(tri, inset_lo[k], inset_hi[k])):
                cells = [c for cell in cells for c in _subtract(cell, (inset_lo[k], inset_hi[k]))]
                carved.add(p["label"])
        for cell in cells:
            for hv, h in convex_pieces(tri, cell[0], cell[1], settings, own=(plo, phi)):
                statics.append(StaticPiece(p["label"], hv, mu, _as_box(hv, float(h.volume), settings["box_fill"])))
    # the movable objects that stay where they are: their hulls
    others = [o for o in data["objects"] if o.get("movable") and o.get("hull") and o["id"] not in {c["id"] for c in chosen}]
    if others:
        olo, ohi = _boxes_of(others)
        for o, a, b in zip(others, olo, ohi):
            if np.all((reg_hi >= a) & (reg_lo <= b), 1).any():
                r = _rot(np.asarray(o["pose"].get("quat_wxyz", [1.0, 0.0, 0.0, 0.0]), np.float64))
                hv, h = _hull(np.asarray(o["hull"]["vertices"]) @ r.T + o["pose"]["position"])
                if hv is not None:
                    statics.append(StaticPiece(o["label"], hv, float(o["friction"]["mu"]),
                                               _as_box(hv, float(h.volume), settings["box_fill"])))
    if len(statics) > settings["max_static"]:   # the nearest to a free body
        def gap(s):
            slo, shi = s.vertices.min(0), s.vertices.max(0)
            return float(np.min(np.linalg.norm(np.maximum(0.0, np.maximum(lo - shi, slo - hi)), axis=1)))
        skipped["over_max"] = len(statics) - settings["max_static"]
        statics = sorted(statics, key=gap)[:settings["max_static"]]
    floor_mu = 0.6
    ground = [p for p in parts if p.get("ground") and p.get("material")]
    if ground:   # the floor's own material under the free bodies
        g = min(ground, key=lambda p: abs(float(_to_sim(p["hi"])[2]) - floor_z))
        floor_mu = material_friction(g["material"])[0]
    info = {"free": len(free), "free_compound": {f.id: len(f.pieces) for f in free if f.pieces},
            "rolling": [f.id for f in free if f.rolls], "static": len(statics),
            "static_boxes": sum(s.box is not None for s in statics), "static_meshes": sum(s.box is None for s in statics),
            "skipped": skipped, "carved": sorted(carved), "floor_z": floor_z,
            "floor_mu": floor_mu, "read_s": round(t_read, 2), "build_s": round(time.monotonic() - started, 2)}
    return _Built(free, statics, floor_z, floor_mu, info)


# ---------------------------------------------------------------- the world
def _has_sleep() -> bool:
    """Whether this MuJoCo sleeps islands at rest (3.9 does; the world works without, every body simulated)."""
    import mujoco
    return hasattr(mujoco.mjtEnableBit, "mjENBL_SLEEP") and hasattr(mujoco, "mjtSleepPolicy")


def _yaw(r: np.ndarray) -> float:
    return math.atan2(float(r[1, 0]), float(r[0, 0]))


class PushWorld:
    """Free bodies MuJoCo moves, static colliders, and the robot as a kinematic pusher (see the module doc)."""

    def __init__(self, objects: list, statics: list, footprint_xy, robot_height: float, floor_z: float | None = 0.0,
                 floor_mu: float = 0.6, settings: dict | None = None, info: dict | None = None):
        import mujoco

        self.settings = s = dict(SETTINGS, **(settings or {}))
        self.objects = list(objects)
        self.ids = [o.id for o in self.objects]
        self.info = dict(info or {})
        self.sleeps = _has_sleep()
        foot = np.asarray(footprint_xy, np.float64).reshape(-1, 2)
        z0, z1 = s["robot_bottom_m"], s["robot_bottom_m"] + float(robot_height)
        prism = np.concatenate([np.c_[foot, np.full(len(foot), z0)], np.c_[foot, np.full(len(foot), z1)]])
        self.xml = self._xml(prism, statics, floor_z, floor_mu)
        self.model = m = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(m)
        self._body = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"o{k}") for k in range(len(self.objects))],
                              np.int64)
        of_body = {int(b): k for k, b in enumerate(self._body)}
        self._geom_object = {g: of_body[int(m.geom_bodyid[g])] for g in range(m.ngeom) if int(m.geom_bodyid[g]) in of_body}
        self._robot_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "robot")
        self._robot_q = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot_x")])
        self._robot_v = int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot_x")])
        self._qadr = np.array([m.jnt_qposadr[m.body_jntadr[b]] for b in self._body], np.int64)
        self._initial = np.array([o.initial for o in self.objects]).reshape(-1, 4, 4)
        self._initial_inv = np.linalg.inv(self._initial) if len(self.objects) else self._initial
        self._robot = None          # the robot's last pose (x, y, z, yaw), None before the first step
        self._phys = None           # where its collider is (behind ``_robot`` while it cannot push on)
        self.lag = 0.0              # metres the collider is behind the pose step() was given (a push it could not make)
        self.stalled = 0.0          # share of the last step's substeps it could not push on
        self._ghost = False         # after a jump: the robot passes through bodies until it overlaps none
        self._carry = 0.0           # time asked for but not yet stepped (dt not a multiple of the timestep)
        self.info["depenetrated"] = self._depenetrate()
        self.info["held"] = self._settle()
        # the first step() returns every body the settling moved, so the renderer agrees with the physics from then on
        self._reported = self._initial.copy()
        now = self._current()
        self.info["settled_mm"] = {i: round(float(np.linalg.norm(now[k, :3, 3] - self._initial[k, :3, 3])) * 1000.0, 1)
                                   for k, i in enumerate(self.ids)}
        self.info["geoms"] = int(m.ngeom)

    @classmethod
    def from_scene(cls, scene_dir, footprint_xy, robot_height: float, objects: str = "reachable",
                   settings: dict | None = None) -> "PushWorld":
        """A scene's PushWorld from its physics.json (tools/physics_tag.py) and its fixed parts (see the module doc)."""
        built = build(scene_dir, objects, settings)
        return cls(built.objects, built.statics, footprint_xy, robot_height, built.floor_z, built.floor_mu, settings,
                   built.info)

    # --- the model ------------------------------------------------------------------------------------------------
    def _xml(self, prism, statics, floor_z, floor_mu) -> str:
        s = self.settings
        assets = [f'<mesh name="robot" vertex="{_fmt(prism)}"/>']
        world = []
        if floor_z is not None:
            world.append(f'<geom name="floor" type="plane" size="0 0 1" pos="0 0 {floor_z:.6g}" '
                         f'friction="{floor_mu:.3g} 0.005 0.0001" contype="{STATIC}" conaffinity="0"/>')
        for k, st in enumerate(statics):
            common = f'friction="{st.mu:.3g} 0.005 0.0001" contype="{STATIC}" conaffinity="0" group="1"'
            if st.box is not None:
                c, half, r = st.box
                world.append(f'<geom name="s{k}" type="box" pos="{_fmt(c)}" quat="{_fmt(_quat(r))}" size="{_fmt(half)}" '
                             f'{common}/>')
            else:
                assets.append(f'<mesh name="s{k}" vertex="{_fmt(st.vertices)}" maxhullvert="64"/>')
                world.append(f'<geom name="s{k}" type="mesh" mesh="s{k}" {common}/>')
        for k, o in enumerate(self.objects):
            # its own friction against whatever it meets (priority: MuJoCo would take the larger of the two)
            common = (f'friction="{o.mu:.3g} 0.005 0.0001" priority="1" contype="{FREE}" '
                      f'conaffinity="{STATIC | FREE | ROBOT}" group="2"')
            if o.pieces:
                geoms = []
                for j, piece in enumerate(o.pieces):
                    assets.append(f'<mesh name="o{k}_{j}" vertex="{_fmt(piece)}" maxhullvert="64"/>')
                    geoms.append(f'<geom name="o{k}_{j}" type="mesh" mesh="o{k}_{j}" {common}/>')
            else:
                assets.append(f'<mesh name="o{k}" vertex="{_fmt(o.vertices)}" face="{_fmt(o.faces)}" maxhullvert="64"/>')
                geoms = [f'<geom name="o{k}" type="mesh" mesh="o{k}" {common}/>']
            inertia = hull_inertia(o.vertices, o.faces, o.mass)
            if inertia is None:   # no volume: the box of its hull
                ext = np.ptp(o.vertices, 0) if len(o.vertices) else np.full(3, 0.05)
                inertia = np.diag(o.mass / 12.0 * (ext[[1, 0, 0]] ** 2 + ext[[2, 2, 1]] ** 2))
            inertia = inertia + np.eye(3) * 1e-7
            full = [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]
            world.append(f'<body name="o{k}" pos="{_fmt(o.initial[:3, 3])}" quat="{_fmt(_quat(o.initial[:3, :3]))}">'
                         f'<freejoint/><inertial pos="0 0 0" mass="{o.mass:.6g}" fullinertia="{_fmt(full)}"/>'
                         f'{"".join(geoms)}</body>')
        # The robot: a body on x, y, z slides and a yaw hinge whose pose and velocity step() sets every substep, heavy
        # (1 t) and held against gravity, so a substep's contacts barely move it before it is set again. Not a mocap
        # body: MuJoCo 3.9 does not wake a sleeping body a mocap body runs into (measured: it passes through it).
        never = ' sleep="never"' if self.sleeps else ""
        joints = "".join(f'<joint name="robot_{n}" type="{t}" axis="{a}"/>' for n, t, a in
                         (("x", "slide", "1 0 0"), ("y", "slide", "0 1 0"), ("z", "slide", "0 0 1"), ("yaw", "hinge", "0 0 1")))
        world.append(f'<body name="robot" pos="0 0 -100" gravcomp="1"{never}>{joints}<geom name="robot" type="mesh" '
                     f'mesh="robot" mass="1000" friction="{s["robot_mu"]:.3g} 0.005 0.0001" priority="2" '
                     f'contype="{ROBOT}" conaffinity="0" group="3"/></body>')
        sleep = '<flag sleep="enable"/>' if self.sleeps else ""
        tol = f' sleep_tolerance="{s["sleep_tolerance"]:.6g}"' if self.sleeps else ""
        return (f'<mujoco model="push"><compiler balanceinertia="true"/>'
                f'<option timestep="{s["timestep_s"]:.6g}" integrator="implicitfast"{tol}>{sleep}</option>'
                f'<asset>{"".join(assets)}</asset><worldbody>{"".join(world)}</worldbody></mujoco>')

    def _depenetrate(self, iterations: int = 30, slack: float = 5e-4) -> dict:
        """Move the free bodies out of what they penetrate at their start: along the contact normal of the deepest
        contact each, the upper body of two free ones, until none is deeper than ``slack``. {object id: metres moved}."""
        import mujoco

        m, d = self.model, self.data
        moved = np.zeros((len(self.objects), 3))
        for _ in range(iterations):
            mujoco.mj_fwdPosition(m, d)
            push = {}
            for c in d.contact[:d.ncon]:
                depth = -float(c.dist) - slack
                if depth <= 0.0:
                    continue
                n = np.array(c.frame[:3])          # from geom1 to geom2
                k1, k2 = self._geom_object.get(int(c.geom1)), self._geom_object.get(int(c.geom2))
                if k1 is not None and k2 is not None:   # two free bodies: the upper one goes
                    k, n = (k2, n) if d.xipos[self._body[k2]][2] >= d.xipos[self._body[k1]][2] else (k1, -n)
                elif k2 is not None:
                    k = k2
                elif k1 is not None:
                    k, n = k1, -n
                else:
                    continue
                if k not in push or depth > push[k][0]:
                    push[k] = (depth + slack, n)
            if not push:
                break
            for k, (depth, n) in push.items():
                d.qpos[self._qadr[k]:self._qadr[k] + 3] += n * depth
                moved[k] += n * depth
        d.qvel[:] = 0.0
        return {self.ids[k]: round(float(np.linalg.norm(v)), 4) for k, v in enumerate(moved) if np.linalg.norm(v) > 0}

    def _settle(self, rounds: int = 3) -> list:
        """Let the bodies come to rest on what carries them (``settle_s``): a hull a few mm over its shelf
        (physics.json's support band) or a cart whose casters float 15 mm over the floor (CS02) comes down now, not when
        the robot first touches it. A body that does more than come down (moves ``settle_shift_m`` sideways or turns
        ``settle_deg``) was balanced in the scene the way its hull does not stand (a cable coil on its edge on a flat
        reel support rolls off it): it is put back and held asleep, as is everything at the end, until touched.
        Returns the ids held."""
        import mujoco

        m, d, s = self.model, self.data, self.settings
        steps = int(round(s["settle_s"] / m.opt.timestep))
        if not len(self.objects) or steps <= 0:
            return []
        base = d.qpos.copy()
        held: set = set()
        for _ in range(rounds):
            m.qpos0[:] = base
            mujoco.mj_resetData(m, d)
            hold = [(int(self._qadr[k]), int(m.jnt_dofadr[m.body_jntadr[self._body[k]]])) for k in held]
            for _ in range(steps):
                mujoco.mj_step(m, d)
                for q, v in hold:   # where the scene has it (a sleeping body could not be: its island is awake)
                    d.qpos[q:q + 7], d.qvel[v:v + 6] = base[q:q + 7], 0.0
            if not self.sleeps:
                break
            more = set()
            for k, a in enumerate(self._qadr):
                shift = float(np.linalg.norm(d.qpos[a:a + 2] - base[a:a + 2]))
                q0, q1 = base[a + 3:a + 7], d.qpos[a + 3:a + 7]
                turn = math.degrees(2.0 * math.acos(min(1.0, abs(float(q0 @ q1)))))
                if k not in held and (shift > s["settle_shift_m"] or turn > s["settle_deg"]):
                    more.add(k)
            if not more:
                break
            held |= more
        final = d.qpos.copy()
        for k in held:
            final[self._qadr[k]:self._qadr[k] + 7] = base[self._qadr[k]:self._qadr[k] + 7]
        m.qpos0[:] = final
        if self.sleeps:
            for b in self._body:
                m.tree_sleep_policy[int(m.body_treeid[b])] = int(mujoco.mjtSleepPolicy.mjSLEEP_INIT)
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        return sorted(self.ids[k] for k in held)

    # --- stepping -------------------------------------------------------------------------------------------------
    def _current(self) -> np.ndarray:
        d = self.data
        out = np.tile(np.eye(4), (len(self.objects), 1, 1))
        out[:, :3, :3] = d.xmat[self._body].reshape(-1, 3, 3)
        out[:, :3, 3] = d.xpos[self._body]
        return out

    def _set_robot(self, pose, vel) -> None:
        q, v = self._robot_q, self._robot_v
        self.data.qpos[q:q + 4] = (pose[0], pose[1], pose[2] + 100.0, pose[3])   # its body hangs at z = -100
        self.data.qvel[v:v + 4] = vel

    def _overlaps(self) -> bool:
        """Whether the robot's prism overlaps a free body (positions as of the last kinematics)."""
        import mujoco
        m, d = self.model, self.data
        return any(mujoco.mj_geomDistance(m, d, self._robot_geom, g, 0.01, None) < 0.0 for g in self._geom_object)

    def step(self, robot_pose, dt: float) -> dict:
        """Put the robot at ``robot_pose`` (a 4x4, hw.Transform or (x, y, z, yaw_deg); simulator world; only its yaw
        turns it) and advance ``dt`` seconds in substeps of ``timestep_s``, the robot moved there from its last pose
        at an even speed. Returns the objects that moved more than ``moved_m`` / ``moved_deg`` since they were last
        returned: {id: 4x4 current · initial⁻¹}.

        The robot pushes with at most ``push_force_n``: beyond it (a cart jammed against a rack) its collider stops
        while the drive model's pose goes on (``lag``; it catches up once the push gives), and ``lag_m`` behind it
        jumps there. A jump (more than ``teleport_m`` between two poses, the first pose too) is taken at once, and the
        robot then passes through bodies until it overlaps none: put down on a case, it would throw it."""
        import mujoco

        m, d, s = self.model, self.data, self.settings
        ts = m.opt.timestep
        r = _matrix(robot_pose)
        target = np.array([r[0, 3], r[1, 3], r[2, 3], _yaw(r)])
        start = self._robot
        if start is None or np.linalg.norm(target[:3] - start[:3]) > s["teleport_m"]:
            start, self._phys, self._ghost = target, None, True
        target[3] = start[3] + (target[3] - start[3] + math.pi) % (2.0 * math.pi) - math.pi   # the short way round
        if self._phys is not None and np.linalg.norm(target[:3] - self._phys[:3]) > s["lag_m"]:
            self._phys, self._ghost = None, True
        phys = target.copy() if self._phys is None else self._phys.copy()
        phys[3] = target[3] + (phys[3] - target[3] + math.pi) % (2.0 * math.pi) - math.pi
        self._carry += float(dt)
        n = int(self._carry / ts + 1e-9)
        self._carry -= n * ts
        if self._ghost:
            self._set_robot(phys, 0.0)
            mujoco.mj_kinematics(m, d)
            self._ghost = self._overlaps()
            m.geom_contype[self._robot_geom] = 0 if self._ghost else ROBOT
        # catching up after a stall: at most twice the commanded speed (1 m/s at least), not a kick
        reach = max(1.0, 2.0 * float(np.linalg.norm(target[:3] - start[:3])) / max(n * ts, ts)) * ts
        stalled = 0
        for i in range(1, n + 1):
            want = start + (target - start) * (i / n)
            nxt = phys
            if float(np.linalg.norm(d.qfrc_constraint[self._robot_v:self._robot_v + 2])) <= s["push_force_n"]:
                gap = float(np.linalg.norm(want[:3] - phys[:3]))
                nxt = want if gap <= reach else phys + (want - phys) * (reach / gap)
            else:
                stalled += 1
            self._set_robot(nxt, (nxt - phys) / ts)
            mujoco.mj_step(m, d)
            phys = nxt
        self._set_robot(phys, 0.0)
        self._robot, self._phys = target, phys
        self.lag = float(np.linalg.norm(target[:3] - phys[:3]))
        self.stalled = stalled / n if n else 0.0
        return self._moved()

    def _moved(self) -> dict:
        now = self._current()
        if not len(now):
            return {}
        shift = np.linalg.norm(now[:, :3, 3] - self._reported[:, :3, 3], axis=1)
        rel = np.einsum("nji,njk->nik", self._reported[:, :3, :3], now[:, :3, :3])
        angle = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)))
        out = {}
        for k in np.nonzero((shift > self.settings["moved_m"]) | (angle > self.settings["moved_deg"]))[0]:
            self._reported[k] = now[k]
            out[self.ids[k]] = now[k] @ self._initial_inv[k]
        return out

    def poses(self) -> dict:
        """Every object's pose: {id: 4x4 current · initial⁻¹}."""
        now = self._current()
        return {i: now[k] @ self._initial_inv[k] for k, i in enumerate(self.ids)}

    def world_poses(self) -> dict:
        """Every object's frame in the world now: {id: 4x4} (its initial one is physics.json's pose)."""
        return dict(zip(self.ids, self._current()))

    def contacts(self) -> list:
        """The objects touching the robot now (as of the last substep)."""
        d, out = self.data, set()
        for c in d.contact[:d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if self._robot_geom in (g1, g2):
                k = self._geom_object.get(g2 if g1 == self._robot_geom else g1)
                if k is not None:
                    out.add(self.ids[k])
        return sorted(out)

    def awake(self) -> int:
        """How many free bodies are simulated now (the others sleep until something touches them)."""
        if not self.sleeps or not len(self.objects):
            return len(self.objects)
        return int(np.count_nonzero(np.asarray(self.data.body_awake)[self._body] > 0))
