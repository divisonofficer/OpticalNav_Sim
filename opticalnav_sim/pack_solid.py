"""A Solid preview straight from a pack scene's files: to look round a scene while the render server prepares it.

    solid = PackSolid(scene_dir)                     # reads base.xml and its meshes (no Mitsuba)
    s0, depth = solid.render(c2w, 640, 360, 70.0)    # H x W x 4 (R, G, B, near-infrared lane), metres along the view

The path tracer has nothing to show until its scene is loaded and its frozen render recorded (a converted .blend's first
start: the conversion, merging its meshes, finding its movable objects, loading, recording: minutes). The raster preview's
Solid view (raster.RasterScene.render_solid) draws the loaded Mitsuba scene, so it waits as long. This one reads the
scene file as the converter wrote it: every ``ply`` / ``obj`` / ``rectangle`` / ``cube`` shape at its ``to_world``,
``instance`` of a ``shapegroup`` too; each in its material's colour, as Blender's Solid viewport shows a scene: a
texture's mean colour, else the colour value (the first of the BSDF's diffuse, base or specular colour; a blend's first
leaf); glass and camera helpers left out; lamps' glowing parts light. Flat shaded by the face normals under the raster
Solid view's studio light (raster.SOLID_*), on nvdiffrast. Display-referred like it: the viewer shows it at its fixed
tone. No filters, nothing to pick: those need the loaded scene (raster.RasterScene).

Thread-safe (a lock): the render server draws it on a request's own thread, beside its render thread.
"""
from __future__ import annotations

import math
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SKIP_PREFIXES = ("camera_assist", "light_hidden_", "light_softbox_")
COLOUR_NAMES = ("diffuse_reflectance", "base_color", "reflectance", "specular_reflectance")
DEFAULT = (0.6, 0.6, 0.6)
GLASS = ("dielectric", "thindielectric", "roughdielectric")


def _floats(text: str) -> list:
    return [float(v) for v in text.replace(",", " ").split()]


def to_world(el) -> np.ndarray:
    """A shape's or instance's to_world (4x4): matrix, translate, scale, rotate in the order written."""
    m = np.eye(4)
    t = el.find("transform[@name='to_world']")
    if t is None:
        return m
    for op in t:
        if op.tag == "matrix":
            m = np.array(_floats(op.get("value")), dtype=np.float64).reshape(4, 4) @ m
        elif op.tag == "translate":
            v = _floats(op.get("value")) if op.get("value") else [float(op.get(k, 0)) for k in "xyz"]
            s = np.eye(4)
            s[:3, 3] = v
            m = s @ m
        elif op.tag == "scale":
            v = _floats(op.get("value")) if op.get("value") else [float(op.get(k, 1)) for k in "xyz"]
            v = v * 3 if len(v) == 1 else v
            m = np.diag(list(v) + [1.0]) @ m
        elif op.tag == "rotate":
            axis = np.array([float(op.get(k, 0)) for k in "xyz"])
            a = math.radians(float(op.get("angle", 0)))
            axis = axis / max(np.linalg.norm(axis), 1e-12)
            x, y, z = axis
            k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            r = np.eye(4)
            r[:3, :3] = np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * (k @ k)
            m = r @ m
    return m


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


class _Materials:
    """A BSDF element's Solid colour (linear RGB), texture means read once."""

    def __init__(self, scene_dir: Path, refs: dict):
        self.dir, self.refs, self.tex = scene_dir, refs, {}

    def texture_mean(self, path: str, srgb: bool = True) -> np.ndarray | None:
        if path not in self.tex:
            try:
                from PIL import Image
                im = Image.open(self.dir / path).convert("RGB")
                im.draft("RGB", (64, 64))
                a = np.asarray(im.resize((16, 16)), dtype=np.float64) / 255.0
                a = _srgb_to_linear(a) if srgb else a
                self.tex[path] = a.reshape(-1, 3).mean(0)
            except Exception:  # noqa: BLE001 - an unreadable texture: the default colour
                self.tex[path] = None
        return self.tex[path]

    def colour(self, bsdf) -> tuple | None:
        """None: not drawn (glass)."""
        if bsdf is None:
            return DEFAULT
        if bsdf.tag == "ref":
            return self.colour(self.refs.get(bsdf.get("id")))
        kind = bsdf.get("type", "")
        if kind in GLASS:
            return None
        if kind == "null":
            return None
        for name in COLOUR_NAMES:
            el = bsdf.find(f"*[@name='{name}']")
            if el is None:
                continue
            if el.tag in ("rgb", "spectrum", "float"):
                v = _floats(el.get("value", "0.6"))
                return tuple((v * 3)[:3]) if len(v) == 1 else tuple(v[:3])
            if el.tag == "texture":
                f = el.find(".//string[@name='filename']")
                raw = el.find(".//boolean[@name='raw']")
                if f is not None:
                    m = self.texture_mean(f.get("value"), srgb=not (raw is not None and raw.get("value") == "true"))
                    if m is not None:
                        return tuple(m)
                v = el.find("rgb")
                if v is not None:
                    return tuple(_floats(v.get("value"))[:3])
        for child in bsdf:   # wrappers (twosided, normalmap, mask, blendbsdf: its first leaf)
            if child.tag in ("bsdf", "ref"):
                return self.colour(child)
        return DEFAULT


def _read_mesh(path: Path):
    if path.suffix.lower() == ".ply":
        from .groundmap import read_ply
        return read_ply(path)
    v, f = [], []
    for line in path.read_text(errors="replace").splitlines():   # a plain OBJ: v and f lines
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(p.split("/")[0]) for p in line.split()[1:]]
            idx = [i - 1 if i > 0 else len(v) + i for i in idx]
            for k in range(1, len(idx) - 1):
                f.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(v, np.float32).reshape(-1, 3), np.asarray(f, np.int64).reshape(-1, 3)


_RECT = (np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], np.float32), np.array([[0, 1, 2], [0, 2, 3]]))
_CUBE = (np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float32),
         np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4],
                   [1, 5, 7], [1, 7, 3]]))


class PackSolid:
    """A pack scene's geometry and Solid colours on the GPU (module docstring)."""

    def __init__(self, scene_dir, xml: str = "base.xml"):
        import torch
        t0 = time.perf_counter()
        self.dir = Path(scene_dir)
        self.path = self.dir / xml
        self.stamp = self.path.stat().st_mtime_ns
        root = ET.parse(self.path).getroot()
        refs = {e.get("id"): e for e in root.iter("bsdf") if e.get("id")}
        mats = _Materials(self.dir, refs)
        groups = {e.get("id"): e for e in root.iter("shape") if e.get("type") == "shapegroup"}
        meshes: dict = {}
        verts, faces, colours, lit, owner = [], [], [], [], []
        self.shapes: list[dict] = []   # each drawn shape: id, kind, file, material, colour, light, its instance
        count = 0

        def mesh_of(shape):
            kind = shape.get("type")
            if kind in ("ply", "obj"):
                f = shape.find("string[@name='filename']")
                if f is None:
                    return None
                key = f.get("value")
                if key not in meshes:
                    try:
                        meshes[key] = _read_mesh(self.dir / key)
                    except Exception:  # noqa: BLE001 - a mesh that does not read: left out
                        meshes[key] = None
                return meshes[key]
            if kind == "rectangle":
                return _RECT
            if kind == "cube":
                return _CUBE
            return None

        def add(shape, outer: np.ndarray, instance: str | None = None):
            nonlocal count
            sid = shape.get("id") or ""
            if sid.startswith(SKIP_PREFIXES):
                return
            mesh = mesh_of(shape)
            if mesh is None or not len(mesh[1]):
                return
            bsdf = shape.find("bsdf")
            if bsdf is None:
                bsdf = shape.find("ref")
            emitter = shape.find("emitter") is not None
            colour = (1.0, 0.95, 0.85) if emitter else mats.colour(bsdf)
            if colour is None:
                return
            m = outer @ to_world(shape)
            v = np.asarray(mesh[0], np.float64) @ m[:3, :3].T + m[:3, 3]
            verts.append(v.astype(np.float32))
            faces.append(np.asarray(mesh[1], np.int64) + count)
            count += len(v)
            colours.append(np.tile(np.asarray(colour, np.float32)[:3], (len(mesh[1]), 1)))
            lit.append(np.full(len(mesh[1]), emitter))
            owner.append(np.full(len(mesh[1]), len(self.shapes), np.int32))
            f = shape.find("string[@name='filename']")
            self.shapes.append({"id": sid, "kind": shape.get("type"), "file": f.get("value") if f is not None else None,
                                "material": _material_name(bsdf, refs), "colour": [round(float(c), 3) for c in colour[:3]],
                                "emitter": emitter, "instance": instance})

        for el in root:
            if el.tag != "shape":
                continue
            kind = el.get("type")
            if kind == "shapegroup":
                continue
            if kind == "instance":
                ref = el.find("ref")
                group = groups.get(ref.get("id")) if ref is not None else None
                if group is not None:
                    at = to_world(el)
                    for child in group.findall("shape"):
                        add(child, at, el.get("id"))
                continue
            add(el, np.eye(4))
        dev = torch.device("cuda")
        self.torch, self.dev = torch, dev
        v = np.concatenate(verts) if verts else np.zeros((3, 3), np.float32)
        f = np.concatenate(faces) if faces else np.array([[0, 1, 2]])
        c = np.concatenate(colours) if colours else np.ones((1, 3), np.float32)
        a, b, cc = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        n = np.cross(b - a, cc - a)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        self.pos = torch.as_tensor(np.c_[v, np.ones(len(v), np.float32)], device=dev)
        self.tri = torch.as_tensor(f.astype(np.int32), device=dev)
        self.normal = torch.as_tensor(n.astype(np.float32), device=dev)
        self.colour = torch.as_tensor(np.clip(c, 0.0, 1.0), device=dev)
        self.lit = torch.as_tensor(np.concatenate(lit) if lit else np.zeros(1, bool), device=dev)
        self.triangles = int(len(f))
        # on the host, for picking: the vertices, the triangles, each triangle's shape (self.shapes) and its normal
        self.v, self.f, self.n = v, f, n.astype(np.float32)
        self.tri_shape = np.concatenate(owner) if owner else np.zeros(1, np.int32)
        self._ids_key, self._ids = None, None
        self.lock = threading.Lock()
        self._ctx = None
        self.build_s = time.perf_counter() - t0

    CHUNK = (1 << 24) - 1   # nvdiffrast's triangle ids are float32: exact up to 2^24 (PH45: 47.6 M triangles)

    def _rasterize(self, c2w, w: int, h: int, hfov_deg: float, drawn):
        """(clip positions, the camera matrix, nvdiffrast's rast (1 x H x W x 4) or None when drawn in chunks, each
        pixel's index into ``drawn`` (H*W, -1: none), its NDC depth (H*W)). Over CHUNK triangles the scene is drawn in
        chunks, the nearest hit of each pixel kept (no edge antialiasing then)."""
        import nvdiffrast.torch as drt
        torch = self.torch
        if self._ctx is None:
            self._ctx = drt.RasterizeCudaContext(device=self.dev)
        m = torch.as_tensor(np.asarray(c2w, dtype=np.float64), device=self.dev, dtype=torch.float32)
        t = math.tan(math.radians(float(hfov_deg)) / 2.0)
        near, far = 0.05, 400.0
        proj = torch.tensor([[1 / t, 0, 0, 0], [0, -(w / h) / t, 0, 0],
                             [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)], [0, 0, -1, 0]],
                            device=self.dev, dtype=torch.float32)
        clip = (self.pos @ (proj @ torch.linalg.inv(m)).T)[None].contiguous()
        if len(drawn) <= self.CHUNK:
            rast, _ = drt.rasterize(self._ctx, clip, drawn, (h, w))
            return clip, m, rast, rast[0, :, :, 3].reshape(-1).round().long() - 1, rast[0, :, :, 2].reshape(-1)
        best_z = torch.full((h * w,), math.inf, device=self.dev)
        best_t = torch.full((h * w,), -1, dtype=torch.long, device=self.dev)
        for a in range(0, len(drawn), self.CHUNK):
            part = drawn[a:a + self.CHUNK].contiguous()
            r, _ = drt.rasterize(self._ctx, clip, part, (h, w))
            tid = r[0, :, :, 3].reshape(-1).round().long() - 1
            z = r[0, :, :, 2].reshape(-1)
            win = (tid >= 0) & (z < best_z)
            best_z = torch.where(win, z, best_z)
            best_t = torch.where(win, tid + a, best_t)
        return clip, m, None, best_t, torch.where(best_t >= 0, best_z, torch.zeros_like(best_z))

    def render(self, c2w: np.ndarray, width: int, height: int, hfov_deg: float, subset=None, tint=None) -> tuple[np.ndarray, np.ndarray]:
        """S0 (H x W x 4: R, G, B and its luminance as the near-infrared lane) and depth (H x W, m along the view; 0: none)
        from a camera (``c2w``: the render server's camera_to_world, Mitsuba axes). ``subset``: these triangles alone
        (indices); ``tint``: these triangles in orange over their colour (tools/physics_review.py's highlight)."""
        from . import raster   # (also its nvdiffrast plugin path)
        import nvdiffrast.torch as drt
        torch = self.torch
        with self.lock:
            w, h = int(width), int(height)
            t = math.tan(math.radians(float(hfov_deg)) / 2.0)
            near, far = 0.05, 400.0
            drawn = self.tri
            if subset is not None:
                sub = torch.as_tensor(np.asarray(subset, np.int64), device=self.dev)
                drawn = self.tri[sub].contiguous()
            clip, m, rast, tid, z = self._rasterize(c2w, w, h, hfov_deg, drawn)
            hit = tid >= 0
            tri = tid.clamp_min(0)
            if subset is not None:
                tri = sub[tri] if len(sub) else tri
            n = self.normal[tri]
            right, up, back = m[:3, 0], m[:3, 1], m[:3, 2]
            eye = m[:3, 3]
            # face the camera: a flat-shaded face's normal toward the eye
            pa = self.pos[self.tri[tri, 0].long(), :3]
            n = torch.where(((eye - pa) * n).sum(1, keepdim=True) < 0, -n, n)
            key = back * 0.75 + up * 0.55 - right * 0.35
            fill = back * 0.6 - up * 0.15 + right * 0.75
            key, fill = key / key.norm(), fill / fill.norm()
            sky = 0.35 + 0.65 * (0.5 + 0.5 * n[:, 1])
            light = raster.SOLID_AMBIENT + raster.SOLID_SKY * sky + raster.SOLID_KEY * (n @ key).clamp_min(0.0) \
                + raster.SOLID_FILL * (n @ fill).clamp_min(0.0)
            col = self.colour[tri] * light[:, None]
            col = torch.where(self.lit[tri][:, None], self.colour[tri], col)
            if tint is not None:
                mark = torch.zeros(len(self.normal), dtype=torch.bool, device=self.dev)
                mark[torch.as_tensor(np.asarray(tint, np.int64), device=self.dev)] = True
                orange = torch.tensor([1.0, 0.42, 0.05], device=self.dev)
                col = torch.where(mark[tri][:, None], 0.35 * col + 0.65 * orange * (0.4 + 0.6 * light[:, None]), col)
            # the backdrop: grey, lighter above the horizon (each pixel's ray, as raster's)
            u = (2.0 * (torch.arange(w, device=self.dev) + 0.5) / w - 1.0) * t
            vv = (1.0 - 2.0 * (torch.arange(h, device=self.dev) + 0.5) / h) * t * h / w
            gy, gx = torch.meshgrid(vv, u, indexing="ij")
            d = torch.stack([gx, gy, -torch.ones_like(gx)], -1).view(-1, 3) @ m[:3, :3].T
            dy = d[:, 1] / d.norm(dim=1)
            back_c = raster.SOLID_BACK[0] + (raster.SOLID_BACK[1] - raster.SOLID_BACK[0]) * (0.5 + 0.5 * dy)
            out = torch.where(hit[:, None], col, back_c[:, None].expand(-1, 3))
            lum = out @ torch.tensor([0.2126, 0.7152, 0.0722], device=self.dev)
            film = torch.cat([out, lum[:, None]], 1).view(1, h, w, 4).contiguous()
            film = drt.antialias(film, rast, clip, drawn)[0] if rast is not None else film[0]
            # depth along the view axis: clip w of the hit point
            depth = torch.where(hit, 2 * far * near / ((far + near) - z * (far - near)), torch.zeros_like(z)).view(h, w)
            return film.cpu().numpy(), depth.cpu().numpy()


    def ids(self, c2w: np.ndarray, width: int, height: int, hfov_deg: float) -> tuple[np.ndarray, np.ndarray]:
        """Each pixel's triangle (H x W int32, -1: none) and depth along the view (m; 0: none) for a camera: what a
        click on a picture of that camera is on. The last camera's are kept (a mouse moving over one picture)."""
        from . import raster   # noqa: F401 - its nvdiffrast plugin path (raster.py: torch's load leaves it off sys.path)
        key = (np.asarray(c2w, np.float64).tobytes(), int(width), int(height), round(float(hfov_deg), 6))
        with self.lock:
            if self._ids_key == key:
                return self._ids
            torch = self.torch
            w, h = int(width), int(height)
            near, far = 0.05, 400.0
            _, _, _, tid, z = self._rasterize(c2w, w, h, hfov_deg, self.tri)
            tid, z = tid.view(h, w), z.view(h, w)
            depth = torch.where(tid >= 0, 2 * far * near / ((far + near) - z * (far - near)), torch.zeros_like(z))
            self._ids_key, self._ids = key, (tid.int().cpu().numpy(), depth.cpu().numpy())
            return self._ids

    def triangle(self, t: int) -> dict:
        """A triangle's shape (self.shapes), its centroid and normal (scene frame: Y up)."""
        a, b, c = self.v[self.f[t]]
        return {"shape": int(self.tri_shape[t]), "centroid": (a + b + c) / 3.0, "normal": self.n[t]}

    def shape_box(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """A shape's bounding box (scene frame): its triangles' corners."""
        tri = self.f[self.tri_shape == i]
        p = self.v[np.unique(tri)]
        return p.min(0), p.max(0)


def _material_name(bsdf, refs: dict) -> dict:
    """A shape's material as the scene file names it: its id and the BSDF's types, outermost first (twosided ->
    pplastic)."""
    if bsdf is None:
        return {"id": None, "types": []}
    mid = bsdf.get("id")
    if bsdf.tag == "ref":
        mid, bsdf = bsdf.get("id"), refs.get(bsdf.get("id"))
    types, el = [], bsdf
    while el is not None and len(types) < 4:
        types.append(el.get("type"))
        el = next((c for c in el if c.tag == "bsdf"), None)
    return {"id": mid, "types": types}


_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


def for_scene(scene_dir, xml: str = "base.xml") -> PackSolid:
    """The scene's PackSolid, read again when its file changed (a merge rewrites it); one at a time is kept."""
    path = Path(scene_dir) / xml
    with _CACHE_LOCK:
        got = _CACHE.get(str(path))
        if got is not None and got.stamp == path.stat().st_mtime_ns:
            return got
        _CACHE.clear()
        s = PackSolid(scene_dir, xml)
        _CACHE[str(path)] = s
        return s
