"""A URDF as the simulator draws it: links (their visuals) and the joints between them.

    model = urdf.load("ranger_mini_v3.urdf")
    model.frames({"fl_steering_joint": 0.3})      # link name -> 4x4 in the root link's frame
    model.visuals["base_link"]                    # [robots.Visual, ...]: boxes, cylinders, meshes

Visual geometry only (collision and inertia are a robot configuration's own numbers: hw/config.py). Meshes: ``.ply``
and ``.obj`` are drawn as they are; ``.stl`` is converted to a PLY beside the model's cache; anything else (``.dae``)
goes through trimesh when it can read it (Collada needs pycollada), else the file is named in the error with what
to do. A mesh's ``scale`` and a visual's ``origin`` are applied when the model is drawn.
"""
from __future__ import annotations

import hashlib
import math
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .robots import Visual
from .transform import Transform, rpy_matrix

CACHE = Path.home() / ".cache" / "opticalnav_sim" / "urdf"


@dataclass
class Joint:
    name: str
    kind: str                 # fixed, revolute, continuous, prismatic
    parent: str
    child: str
    origin: np.ndarray        # 4x4: the child link's frame in the parent's, at position 0
    axis: tuple = (1.0, 0.0, 0.0)
    limit: tuple | None = None   # (lower, upper)

    def motion(self, q: float) -> np.ndarray:
        m = np.eye(4)
        if self.kind in ("revolute", "continuous"):
            m[:3, :3] = _axis_angle(self.axis, q)
        elif self.kind == "prismatic":
            m[:3, 3] = np.asarray(self.axis) * q
        return m


@dataclass
class UrdfModel:
    name: str
    root: str
    visuals: dict = field(default_factory=dict)    # link -> [Visual] (mesh visuals carry their scale in ``size``)
    joints: dict = field(default_factory=dict)     # name -> Joint
    path: Path | None = None

    def frames(self, q: dict | None = None) -> dict:
        """Link name -> 4x4 in the root link's frame, with the joints at ``q`` (name -> position; others at 0)."""
        q = q or {}
        out = {self.root: np.eye(4)}
        pending = list(self.joints.values())
        while pending:
            rest = [j for j in pending if j.parent not in out]
            for j in pending:
                if j.parent in out:
                    out[j.child] = out[j.parent] @ j.origin @ j.motion(float(q.get(j.name, 0.0)))
            if len(rest) == len(pending):
                raise ValueError(f"{self.name}: joints hang from links that are not in the tree: {[j.name for j in rest]}")
            pending = rest
        return out

    def joint(self, name: str) -> Joint:
        if name not in self.joints:
            raise KeyError(f"{self.name} has no joint {name!r} (has: {', '.join(sorted(self.joints))})")
        return self.joints[name]


def _axis_angle(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    x, y, z = a / n
    c, s = math.cos(angle), math.sin(angle)
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + s * k + (1.0 - c) * (k @ k)


def _floats(text: str | None, n: int, default) -> tuple:
    if not text:
        return tuple(default)
    v = tuple(float(x) for x in text.split())
    if len(v) != n:
        raise ValueError(f"expected {n} numbers, got {text!r}")
    return v


def _origin(el) -> np.ndarray:
    o = el.find("origin") if el is not None else None
    xyz = _floats(o.get("xyz") if o is not None else None, 3, (0.0, 0.0, 0.0))
    r, p, y = _floats(o.get("rpy") if o is not None else None, 3, (0.0, 0.0, 0.0))
    m = np.eye(4)
    m[:3, :3] = rpy_matrix(r, p, y)
    m[:3, 3] = xyz
    return m


def read_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and triangles of an STL file (binary or text)."""
    data = Path(path).read_bytes()
    count = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else -1
    if len(data) == 84 + 50 * count:
        rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=count, offset=84)
        v = rec["v"].reshape(-1, 3)
    else:
        v = np.array([[float(x) for x in line.split()[1:4]] for line in data.decode("ascii", "replace").splitlines()
                      if line.strip().startswith("vertex")], dtype=np.float32)
    if len(v) == 0 or len(v) % 3:
        raise ValueError(f"{path}: not an STL mesh")
    return np.asarray(v, dtype=np.float32), np.arange(len(v), dtype=np.int32).reshape(-1, 3)


def drawable_mesh(path: Path, cache: Path | None = None) -> Path:
    """A mesh file the render server loads: the file itself (.ply, .obj) or a PLY made from it once."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"mesh {path} is not there")
    if path.suffix.lower() in (".ply", ".obj"):
        return path.resolve()
    from ..bodies import write_ply
    folder = cache or CACHE
    folder.mkdir(parents=True, exist_ok=True)
    stamp = hashlib.sha1(f"{path.resolve()}|{path.stat().st_size}|{path.stat().st_mtime_ns}".encode()).hexdigest()[:12]
    out = folder / f"{path.stem}-{stamp}.ply"
    if out.is_file():
        return out
    if path.suffix.lower() == ".stl":
        v, f = read_stl(path)
    else:
        try:
            import trimesh
            mesh = trimesh.load(str(path), force="mesh")
            v, f = np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32)
        except Exception as exc:  # noqa: BLE001 - whatever the reader said, with what to do about it
            raise ValueError(f"cannot read the mesh {path.name} ({type(exc).__name__}: {exc}). Export it as .obj, .ply or .stl "
                             "(Blender: File > Export), or install a reader trimesh can use (pycollada for .dae)") from exc
    write_ply(out, v, f)
    return out


def load(path, cache: Path | None = None, grey=(0.6, 0.6, 0.62)) -> UrdfModel:
    """Read a URDF: every link's visuals (boxes, cylinders, meshes) and the joint tree."""
    path = Path(path)
    root = ET.parse(path).getroot()
    if root.tag != "robot":
        raise ValueError(f"{path}: not a URDF (its root element is <{root.tag}>)")
    colours = {m.get("name"): _floats(m.find("color").get("rgba"), 4, (0.6, 0.6, 0.6, 1.0))[:3]
               for m in root.findall("material") if m.find("color") is not None}
    model = UrdfModel(root.get("name") or path.stem, "", path=path)
    for link in root.findall("link"):
        out = []
        for vis in link.findall("visual"):
            geo = vis.find("geometry")
            if geo is None or len(geo) == 0:
                continue
            shape, at = geo[0], _origin(vis)
            mat = vis.find("material")
            rgb = grey
            if mat is not None:
                rgb = _floats(mat.find("color").get("rgba"), 4, (*grey, 1.0))[:3] if mat.find("color") is not None \
                    else colours.get(mat.get("name"), grey)
            material = ("plastic", tuple(float(c) for c in rgb), 0.45)
            if shape.tag == "box":
                out.append(Visual("box", _floats(shape.get("size"), 3, (0.1, 0.1, 0.1)), Transform.from_matrix(at), material))
            elif shape.tag == "cylinder":   # a URDF cylinder stands along z; the simulator's lies along y
                turn = np.eye(4)
                turn[:3, :3] = rpy_matrix(math.pi / 2.0, 0.0, 0.0)
                out.append(Visual("cylinder", (float(shape.get("radius")), float(shape.get("length"))),
                                  Transform.from_matrix(at @ turn), material))
            elif shape.tag == "sphere":     # drawn as the box that holds it: the simulator has no sphere primitive
                d = 2.0 * float(shape.get("radius"))
                out.append(Visual("box", (d, d, d), Transform.from_matrix(at), material))
            elif shape.tag == "mesh":
                name = shape.get("filename", "")
                file = Path(name.split("://", 1)[-1]) if "://" in name else Path(name)
                if "://" in name:   # package://pkg/meshes/x.stl: the path under the package, looked for beside the URDF
                    file = next((c for c in (path.parent / file, path.parent / Path(*file.parts[1:]), path.parent.parent / file)
                                 if c.is_file()), path.parent / file)
                elif not file.is_absolute():
                    file = path.parent / file
                scale = _floats(shape.get("scale"), 3, (1.0, 1.0, 1.0))
                out.append(Visual("mesh", (), Transform.from_matrix(at), material, str(drawable_mesh(file, cache)), scale))
            else:
                raise ValueError(f"{path.name}: link {link.get('name')} has a <{shape.tag}> visual (box, cylinder, sphere or mesh)")
        model.visuals[link.get("name")] = out
    for j in root.findall("joint"):
        kind = j.get("type", "fixed")
        if kind not in ("fixed", "revolute", "continuous", "prismatic"):
            kind = "fixed"   # floating and planar joints: held where the file puts them
        axis = j.find("axis")
        lim = j.find("limit")
        model.joints[j.get("name")] = Joint(
            j.get("name"), kind, j.find("parent").get("link"), j.find("child").get("link"), _origin(j),
            _floats(axis.get("xyz") if axis is not None else None, 3, (1.0, 0.0, 0.0)),
            (float(lim.get("lower", 0.0)), float(lim.get("upper", 0.0))) if lim is not None and kind != "continuous" else None)
    children = {j.child for j in model.joints.values()}
    roots = [name for name in model.visuals if name not in children]
    if len(roots) != 1:
        raise ValueError(f"{path.name}: expected one root link, found {roots}")
    model.root = roots[0]
    model.frames()   # a broken tree fails here, not at the first frame
    return model
