"""A configured robot's picture for the viewer's robot list: its parts as they are drawn in the scene (boxes, wheels,
extrusions, the converted meshes), seen from the front left and above, flat shaded, without the render server.

    png = hw.preview.robot_png("ranger_mini", rig="hdr_12bit_mast")      # bytes; kept in ~/.cache/opticalnav_sim/robot_previews

A depth buffer filled by splatting each triangle as points about a pixel apart, at twice the size, scaled down. A converted Ranger is
about 200,000 triangles: a few seconds the first time, then the cached file.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".cache" / "opticalnav_sim" / "robot_previews"
BACKGROUND = (22, 27, 33)
# a material kind's look when the visual names none of its own (bodies.bsdf_xml kinds)
KIND_RGB = {"metal": (0.62, 0.64, 0.68), "aluminium": (0.75, 0.77, 0.80), "rubber": (0.10, 0.10, 0.11), "plastic": (0.55, 0.55, 0.57)}


def _primitive(vis) -> tuple[np.ndarray, np.ndarray] | None:
    from .. import bodies
    if vis.shape == "box":
        return bodies.box(vis.size)
    if vis.shape == "cylinder":
        return bodies.cylinder(*vis.size, segments=24)
    if vis.shape == "profile":   # the extrusion's outline is enough at this size
        side, length = vis.size
        return bodies.box((side, side, length))
    if vis.shape == "light":     # the lamp's face (the housing is a box of its own)
        w, h = vis.size
        return (np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]], np.float32),
                np.array([[0, 1, 2], [0, 2, 3]], np.int32))
    if vis.shape == "mesh" and vis.mesh and Path(vis.mesh).suffix.lower() == ".ply":
        from ..groundmap import read_ply
        v, f = read_ply(vis.mesh)
        return (v * np.asarray(vis.scale, dtype=np.float64)).astype(np.float32), f.astype(np.int32)
    return None


def _colour(vis) -> tuple:
    kind, rgb, _ = vis.material if len(vis.material) == 3 else (vis.material[0], None, None)
    if kind == "emitter":
        return (1.0, 0.92, 0.75)
    if rgb is None:
        return KIND_RGB.get(kind, (0.5, 0.5, 0.5))
    rgb = tuple(float(c) for c in rgb)
    # very dark colours (black plastic, tyres) are lifted a little: a picture on a dark card, not a measurement
    return tuple(max(c, 0.07) for c in rgb)


def triangles(robot) -> tuple[np.ndarray, np.ndarray]:
    """(T x 3 x 3 world vertices, T x 3 colours) of every drawn part of a spawned robot."""
    tris, cols = [], []
    for actor in [robot.base, *robot.base.descendants()]:
        frames = actor.link_matrices()
        for link, visuals in actor.spec.visuals().items():
            if link not in frames:
                continue
            for vis in visuals:
                mesh = _primitive(vis)
                if mesh is None:
                    continue
                v, f = mesh
                m = frames[link] @ vis.origin.matrix()
                w = (np.c_[v, np.ones(len(v))] @ m.T)[:, :3]
                tris.append(w[f])
                cols.append(np.tile(np.asarray(_colour(vis), dtype=np.float64), (len(f), 1)))
    if not tris:
        return np.zeros((0, 3, 3)), np.zeros((0, 3))
    return np.concatenate(tris), np.concatenate(cols)


def render(robot, size=(320, 240), yaw_deg: float = 35.0, pitch_deg: float = 24.0) -> bytes:
    """A PNG of the robot from the front left (the robot looks along its x, its left is y), above by ``pitch_deg``."""
    from PIL import Image
    tri, col = triangles(robot)
    w, h = size[0] * 2, size[1] * 2
    img = Image.new("RGB", (w, h), BACKGROUND)
    if len(tri):
        # camera: turned about z by yaw (from the robot's front), tilted down by pitch; view axis -> depth
        a, p = math.radians(yaw_deg), math.radians(pitch_deg)
        rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        # screen axes: right = (sin a, cos a, 0) seen from the front-left ..., up tilted by the pitch
        forward = rz @ np.array([-1.0, 0.0, 0.0])                       # looking back at the robot's front
        forward = forward * math.cos(p) + np.array([0, 0, -math.sin(p)])
        right = np.cross(forward, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        flat = tri.reshape(-1, 3)
        sx, sy, depth = flat @ right, flat @ up, flat @ forward
        lo, hi = np.array([sx.min(), sy.min()]), np.array([sx.max(), sy.max()])
        scale = 0.86 * min(w / max(hi[0] - lo[0], 1e-6), h / max(hi[1] - lo[1], 1e-6))
        cx, cy = (lo + hi) / 2.0
        px = (w / 2 + (sx - cx) * scale).reshape(-1, 3)
        py = (h / 2 - (sy - cy) * scale).reshape(-1, 3)
        dz = depth.reshape(-1, 3).mean(1)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        norm = np.linalg.norm(n, axis=1)
        ok = norm > 1e-12
        n[ok] /= norm[ok, None]
        light = np.array([0.35, 0.55, 0.75])
        light /= np.linalg.norm(light)
        facing = n @ -forward
        n = np.where(facing[:, None] < 0, -n, n)             # either winding: the side towards the camera
        shade = 0.38 + 0.62 * np.clip(n @ light, 0.0, 1.0)
        rgb = np.clip(col * shade[:, None] * 255.0, 0, 255).astype(np.uint8)
        # a depth buffer by splatting: each triangle covered by a lattice of points about a pixel apart, and in each pixel
        # the nearest point's colour (drawing far triangles first mixed a shell with the parts inside it)
        edge = np.maximum.reduce([np.hypot(px[:, a] - px[:, b], py[:, a] - py[:, b]) for a, b in ((0, 1), (1, 2), (2, 0))])
        steps = np.clip(np.ceil(edge / 0.7).astype(np.int64), 1, 400)
        pix_all, z_all, c_all = [], [], []
        for k in np.unique(steps[ok]):
            sel = np.where(ok & (steps == k))[0]
            u, v = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
            keep = (u + v) <= k
            wa, wb = u[keep] / k, v[keep] / k
            bary = np.stack([1.0 - wa - wb, wa, wb], 1)                          # K x 3
            for m0 in range(0, len(sel), max(1, 2_000_000 // len(bary))):
                part = sel[m0:m0 + max(1, 2_000_000 // len(bary))]
                x = (bary @ px[part].T).T.ravel()
                y = (bary @ py[part].T).T.ravel()
                z = (bary @ depth.reshape(-1, 3)[part].T).T.ravel()
                xi, yi = np.rint(x).astype(np.int64), np.rint(y).astype(np.int64)
                inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
                pix_all.append((yi * w + xi)[inside])
                z_all.append(z[inside])
                c_all.append(np.repeat(part, len(bary))[inside])
        pix, z, tri_of = np.concatenate(pix_all), np.concatenate(z_all), np.concatenate(c_all)
        order = np.lexsort((z, pix))                    # by pixel, nearest first
        first = np.ones(len(order), bool)
        first[1:] = pix[order][1:] != pix[order][:-1]
        win = order[first]
        frame = np.asarray(img).reshape(-1, 3).copy()
        frame[pix[win]] = rgb[tri_of[win]]
        img = Image.fromarray(frame.reshape(h, w, 3))
    img = img.resize(size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def robot_png(source, rig=None, size=(320, 240)) -> bytes:
    """The picture of a configuration (and rig), made once per content of its files and kept in CACHE."""
    from . import config as robot_config
    from .actor import World
    cfg = robot_config.load(source, rig)
    key = hashlib.sha1(json.dumps([cfg.data, list(size)], sort_keys=True, default=str).encode()).hexdigest()[:12]
    path = CACHE / f"{cfg.name}-{cfg.rig or 'none'}-{key}.png"
    if path.is_file():
        return path.read_bytes()
    png = render(robot_config.spawn(World(), source, rig=rig), size)
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return png
