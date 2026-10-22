#!/usr/bin/env python3
"""Decimate the heavy meshes of an already built (and compacted) pack scene, keeping their textures.

    source /home/jinnyeong/robomituba-build/mitsuba3/setpath.sh
    python3.10 tools/decimate_meshes.py --pack packs/opticalnav-v0.2 [--scene SCENE ...] [--ratio R] [--min-faces N] [--dry-run]

Needs Mitsuba (mesh I/O), pymeshlab and xatlas (``pip install pymeshlab xatlas``); run
tools/compact_assets.py first.

robomituba decimates at import (tools/infinigen/mesh_decimation.py, in Blender, before the texture bake).
This is the same reduction after the fact, for scenes already built without it:

* Policy: robomituba's ``ir_semantic_lod_v1`` ladder on triangle count (meshes under 50k triangles kept;
  at most 30%, 10% from 250k, 3% from 1M, 1% from 5M; never under 2000 triangles). ``--ratio`` replaces the
  ladder by one fraction; ``--min-faces`` moves the 50k line (a market's 277 product meshes of 20-50k hold
  16 of its 24 M triangles). Shapes behind glass, mirror or metal BSDFs keep at least 60% (Fresnel and
  reflections follow the surface normal), and emitting shapes are left alone.
* A mesh whose material samples textures through a continuous UV atlas: texture-aware quadric collapse
  (pymeshlab) on the mesh welded by position. UV seams stay seams, so the existing textures still fit.
* A mesh whose atlas gives every triangle its own island (Infinigen bakes some objects that way; at
  ~1.6 texels per triangle the colour lives in the dense geometry): the welded geometry is collapsed, xatlas
  lays out a new atlas, and each texture is baked again at its own resolution. A texel takes the colour at
  the closest point of the original surface. Normal maps are dropped (their tangent frames belonged to
  the old atlas; the detail they carried is mostly geometry at this density).
* A mesh whose material samples no texture keeps only its geometry; its UVs meant nothing.
* Normals: the originals are mostly per-face (flat) normals that read as smooth only because the mesh is
  dense. The reduced mesh gets area-weighted normals smoothed across edges under 60 degrees, so creases
  stay sharp and curved surfaces stay smooth.
* Each result is measured against its source: the distance between the surfaces (both ways, as a fraction
  of the bounding-box diagonal) and, for a kept atlas, how far a point's UV moved (95th percentile, in
  texels of a 512 px texture). A mesh over the limits is retried at twice the fraction and kept at full
  resolution if it still fails.

New meshes and textures are written as ``assets/mesh_<sha1>.ply`` / ``assets/tex_<sha1>.png`` and the scene
XMLs rewritten, as compact_assets does. The full-resolution PLYs no file names any more are unlinked
(``--keep-unused`` keeps them; build_pack and compact_assets rebuild them from robomituba's OBJs).
scene.json records every decision under ``mesh_decimation``. The geometry then differs slightly from the
dataset's reference images.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compact_assets import FILENAME_RE, json_refs, scene_lock, sha1, xml_refs  # noqa: E402

PLY_SHAPE_RE = re.compile(r'(\n  <shape type="ply" id="([^"]+)">)(.*?)(\n  </shape>)', re.S)
BITMAP_RE = re.compile(r'<texture name="([a-z_]+)" type="bitmap">\s*<string name="filename" value="([^"]+)"')
MIN_FACES, FLOOR_FACES, OPTICAL_RATIO = 50_000, 2_000, 0.60
OPTICAL_BSDFS = ("dielectric", "roughdielectric", "thindielectric", "conductor", "roughconductor")
ROLES = ("plain", "optical", "emitter")  # least to most protected
PER_FACE_UV = 2.0  # distinct UVs per welded vertex above which the atlas is per triangle
CREASE_DEG = 60.0
MAX_SURFACE_ERROR = 0.01  # of the bounding-box diagonal
MAX_UV_TEXELS = 2.0  # 95th percentile, 512 px texture
TEXELS = 512


def ladder(faces: int) -> float:
    """robomituba's ir_semantic_lod_v1 (tools/infinigen/mesh_decimation.py: IRSemanticLodPolicy)."""
    ratio = 0.01 if faces >= 5_000_000 else 0.03 if faces >= 1_000_000 else 0.10 if faces >= 250_000 else 0.30
    return min(0.30, max(ratio, FLOOR_FACES / max(faces, 1)))


def shape_meshes(text: str) -> list[tuple[str, str, str]]:
    """(shape id, mesh ref, body) of every PLY shape of a scene file whose mesh is a pack asset."""
    out = []
    for _, sid, body, _ in PLY_SHAPE_RE.findall(text):
        found = FILENAME_RE.search(body)
        if found is not None:
            out.append((sid, found.group(2), body))
    return out


def role(body: str) -> str:
    return ("emitter" if "<emitter" in body else
            "optical" if any(f'<bsdf type="{b}"' in body for b in OPTICAL_BSDFS) else "plain")


def shape_roles(text: str) -> dict[str, str]:
    """Mesh file -> the most protected role among the shapes of a scene file using it."""
    out: dict[str, str] = {}
    for _, ref, body in shape_meshes(text):
        out[ref] = max(out.get(ref, "plain"), role(body), key=ROLES.index)
    return out


def material(body: str) -> tuple | None:
    """What of a shape's material reads the mesh's UVs: the bitmaps it samples (normal maps apart) and whether
    it has a normal map. None when it comes by reference, where the shape cannot tell."""
    if "<ref " in body:
        return None
    maps = tuple(sorted({f for name, f in BITMAP_RE.findall(body) if name != "normalmap"}))
    return maps, any(name == "normalmap" for name, _ in BITMAP_RE.findall(body))


_MI = []


def _mitsuba():
    if not _MI:
        import mitsuba as mi

        mi.set_variant("scalar_rgb")
        mi.set_log_level(mi.LogLevel.Warn)
        _MI.append(mi)
    return _MI[0]


def read_ply(path: Path):
    import numpy as np

    mi = _mitsuba()
    p = mi.traverse(mi.load_dict({"type": "ply", "filename": str(path)}))
    get = lambda k, n, t: np.array(p[k], dtype=t).reshape(-1, n) if k in p else None  # noqa: E731
    return (get("vertex_positions", 3, np.float64), get("vertex_normals", 3, np.float64),
            get("vertex_texcoords", 2, np.float64), get("faces", 3, np.int32))


def write_ply(path: Path, pos, nor, uv, faces) -> None:
    import numpy as np

    mi = _mitsuba()
    mesh = mi.Mesh("decimated", len(pos), len(faces), has_vertex_normals=True, has_vertex_texcoords=uv is not None)
    p = mi.traverse(mesh)
    p["vertex_positions"] = np.ascontiguousarray(pos, np.float32).ravel()
    if uv is not None:
        p["vertex_texcoords"] = np.ascontiguousarray(uv, np.float32).ravel()
    p["faces"] = np.ascontiguousarray(faces, np.uint32).ravel()
    p.update()  # recomputes smooth normals from the positions; ours go in after
    p["vertex_normals"] = np.ascontiguousarray(nor, np.float32).ravel()
    p.update()
    mesh.write_ply(str(path))


def crease_normals(V, G):
    """Per-corner normals (3F x 3): area-weighted face normals of the faces around a vertex that meet the
    corner's face at under CREASE_DEG."""
    import numpy as np

    fn = np.cross(V[G[:, 1]] - V[G[:, 0]], V[G[:, 2]] - V[G[:, 0]])  # length = 2 x area
    unit = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-30)
    corner_v, corner_f = G.ravel(), np.repeat(np.arange(len(G)), 3)
    order = np.argsort(corner_v, kind="stable")
    cv, cf = corner_v[order], corner_f[order]
    starts = np.flatnonzero(np.r_[True, cv[1:] != cv[:-1]])
    counts = np.diff(np.r_[starts, len(cv)])
    # all (corner, face around the same vertex) pairs, grouped by vertex
    group = np.repeat(np.arange(len(starts)), counts)
    rep = counts[group]
    a = np.repeat(np.arange(len(cv)), rep)
    offs = np.arange(len(a)) - np.repeat(np.cumsum(rep) - rep, rep)
    b = starts[group[a]] + offs
    keep = np.einsum("ij,ij->i", unit[cf[a]], unit[cf[b]]) >= np.cos(np.radians(CREASE_DEG))
    acc = np.zeros((len(cv), 3))
    np.add.at(acc, a[keep], fn[cf[b[keep]]])
    acc /= np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-30)
    out = np.empty_like(acc)
    out[order] = acc
    return out


def surface_error(P, F, V, G, diag: float) -> float:
    """Hausdorff distance between the two surfaces (both ways) over the bounding-box diagonal."""
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=P, face_matrix=F))
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=V, face_matrix=G))
    samples = max(20_000, len(G))
    h1 = ms.get_hausdorff_distance(sampledmesh=1, targetmesh=0, samplenum=samples)["max"]
    h2 = ms.get_hausdorff_distance(sampledmesh=0, targetmesh=1, samplenum=samples)["max"]
    return max(h1, h2) / diag


def uv_shift(P, UV, V, G, W) -> float:
    """95th-percentile distance, in texels, from each reduced corner's UV to the nearest source UV around it."""
    import numpy as np
    from scipy.spatial import cKDTree

    _, idx = cKDTree(P).query(V[G].reshape(-1, 3), k=8)
    shift = np.linalg.norm(UV[idx] - W.reshape(-1, 1, 2), axis=2).min(1) * TEXELS
    return float(np.percentile(shift, 95))


def sample(img, uv):
    """Bilinear lookup with repeat wrap, in Mitsuba's bitmap convention (texel centres at (i + 0.5) / size,
    row 0 at v = 0)."""
    import numpy as np

    h, w = img.shape[:2]
    x, y = uv[:, 0] * w - 0.5, uv[:, 1] * h - 0.5
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    at = lambda yy, xx: img[yy % h, xx % w]  # noqa: E731
    return (at(y0, x0) * (1 - fx) * (1 - fy) + at(y0, x0 + 1) * fx * (1 - fy)
            + at(y0 + 1, x0) * (1 - fx) * fy + at(y0 + 1, x0 + 1) * fx * fy)


def closest_on_triangles(X, A, B, C):
    """Barycentrics of the point of each triangle (A, B, C) closest to X, and the squared distance."""
    import numpy as np

    dot = lambda u, v: np.einsum("ij,ij->i", u, v)  # noqa: E731
    ab, ac = B - A, C - A
    d1, d2 = dot(ab, X - A), dot(ac, X - A)
    d3, d4 = dot(ab, X - B), dot(ac, X - B)
    d5, d6 = dot(ab, X - C), dot(ac, X - C)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    bary = np.stack([va / den, vb / den, vc / den], 1)
    outside = (va < 0) | (vb < 0) | (vc < 0)
    if outside.any():  # the closest point is on an edge: best of the three clamped edge projections
        edges = []
        for i, j, P0, P1 in ((0, 1, A, B), (0, 2, A, C), (1, 2, B, C)):
            e = P1 - P0
            t = np.clip(dot(X - P0, e) / np.maximum(dot(e, e), 1e-30), 0.0, 1.0)
            b = np.zeros_like(bary)
            b[:, i], b[:, j] = 1 - t, t
            edges.append(b)
        edges = np.stack(edges, 1)
        pts = np.einsum("nek,nkd->ned", edges, np.stack([A, B, C], 1))
        best = ((pts - X[:, None]) ** 2).sum(2).argmin(1)
        bary[outside] = edges[np.arange(len(X)), best][outside]
    pt = np.einsum("nk,nkd->nd", bary, np.stack([A, B, C], 1))
    return bary, ((X - pt) ** 2).sum(1)


def rebake(P, UV, F, V, G, sources: dict[str, Path]):
    """A new atlas for the reduced mesh (V, G) and each source texture baked into it from the original
    surface. Returns (atlas vertex index into V, faces, uvs, {source: HxWxC uint8}, worst texel distance)."""
    import numpy as np
    import xatlas
    from PIL import Image
    from scipy import ndimage
    from scipy.spatial import cKDTree

    images = {ref: Image.open(path) for ref, path in sources.items()}
    size = max((max(im.size) for im in images.values()), default=TEXELS)
    atlas = xatlas.Atlas()
    atlas.add_mesh(V.astype(np.float32), G.astype(np.uint32))
    pack = xatlas.PackOptions()
    pack.resolution, pack.padding, pack.bilinear = size, 2, True
    atlas.generate(pack_options=pack)
    vmap, faces, uvs = atlas[0]
    if not np.array_equal(vmap[faces], G):  # per-corner data of G is reused for the atlas mesh
        raise RuntimeError("xatlas reordered triangles or corners")
    pos = V[vmap]
    # every texel centre inside a triangle of the new atlas, and its point on the reduced surface
    corners = uvs[faces] * size - 0.5
    lo = np.floor(corners.min(1)).astype(np.int64)
    span = np.ceil(corners.max(1)).astype(np.int64) - lo + 1
    n = span[:, 0] * span[:, 1]
    tri = np.repeat(np.arange(len(faces)), n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    px, py = lo[tri, 0] + k % span[tri, 0], lo[tri, 1] + k // span[tri, 0]
    a = corners[tri, 0]
    e1, e2, q = corners[tri, 1] - a, corners[tri, 2] - a, np.stack([px, py], 1) - a
    den = e1[:, 0] * e2[:, 1] - e2[:, 0] * e1[:, 1]
    den = np.where(np.abs(den) < 1e-20, 1e-20, den)
    bv = (q[:, 0] * e2[:, 1] - e2[:, 0] * q[:, 1]) / den
    bw = (e1[:, 0] * q[:, 1] - q[:, 0] * e1[:, 1]) / den
    inside = (bv >= -1e-4) & (bw >= -1e-4) & (bv + bw <= 1 + 1e-4) & (px >= 0) & (py >= 0) & (px < size) & (py < size)
    tri, px, py, bv, bw = tri[inside], px[inside], py[inside], bv[inside], bw[inside]
    f = faces[tri]
    X = pos[f[:, 0]] * (1 - bv - bw)[:, None] + pos[f[:, 1]] * bv[:, None] + pos[f[:, 2]] * bw[:, None]
    # its closest point on the original surface (among the triangles with the nearest centroids) and that UV
    _, cand = cKDTree(P[F].mean(1)).query(X, k=6)
    best, src_uv = np.full(len(X), np.inf), np.zeros((len(X), 2))
    for j in range(cand.shape[1]):
        fi = F[cand[:, j]]
        bary, d = closest_on_triangles(X, P[fi[:, 0]], P[fi[:, 1]], P[fi[:, 2]])
        better = d < best
        best[better] = d[better]
        src_uv[better] = np.einsum("nk,nkd->nd", bary, UV[fi])[better]
    filled = np.zeros((size, size), bool)
    filled[py, px] = True
    # texels outside every triangle take their nearest filled neighbour, so bilinear lookups never bleed
    nearest = ndimage.distance_transform_edt(~filled, return_distances=False, return_indices=True)
    baked = {}
    for ref, im in images.items():
        mode = im.mode if im.mode in ("L", "RGB", "RGBA") else "RGB"
        img = np.asarray(im.convert(mode), np.float32)
        img = img[..., None] if img.ndim == 2 else img  # sampled at its own size, baked at the largest
        out = np.zeros((size, size, img.shape[2]), np.float32)
        out[py, px] = sample(img, src_uv)
        out = out[nearest[0], nearest[1]]
        out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        baked[ref] = (out[..., 0] if mode == "L" else out, mode)
    return vmap, faces, uvs, baked, float(np.sqrt(best.max())) if len(best) else 0.0


def collapse(P, F, UV, target: int):
    """Quadric collapse of the mesh welded by position; with UV, texture-aware (UV seams kept)."""
    import pymeshlab

    ms = pymeshlab.MeshSet()
    if UV is not None:
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=P, face_matrix=F, v_tex_coords_matrix=UV))
        ms.compute_texcoord_transfer_vertex_to_wedge()
    else:
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=P, face_matrix=F))
    ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(1e-5))
    kw = dict(targetfacenum=max(4, target), qualitythr=0.3, preserveboundary=True, boundaryweight=1.0,
              optimalplacement=True, preservenormal=True, planarquadric=False)
    if UV is not None:
        ms.meshing_decimation_quadric_edge_collapse_with_texture(extratcoordw=1.0, **kw)
    else:
        ms.meshing_decimation_quadric_edge_collapse(preservetopology=False, **kw)
    m = ms.current_mesh()
    W = m.wedge_tex_coord_matrix().reshape(-1, 3, 2) if UV is not None else None
    return m.vertex_matrix(), m.face_matrix(), W


def store(scene_dir: Path, ref: str, write) -> str:
    """Write through ``write(tmp)`` and file the result under its content hash, like compact_assets."""
    suffix = Path(ref).suffix
    tmp = scene_dir / "assets" / f".{Path(ref).stem}.{os.getpid()}.dec{suffix}"
    try:
        write(tmp)
        new = f"assets/{'mesh' if suffix == '.ply' else 'tex'}_{sha1(tmp)[:20]}{suffix}"
        if (scene_dir / new).exists():
            tmp.unlink()
        else:
            os.replace(tmp, scene_dir / new)
        return new
    finally:
        tmp.unlink(missing_ok=True)


def decimate_mesh(job: tuple) -> dict:
    """One PLY -> a reduced PLY (and, for a per-triangle atlas, re-baked textures), or a reason to keep it.
    ``job[3]``: the material's (bitmaps sampled apart from normal maps, has a normal map), or None when the
    shapes using the mesh disagree."""
    import numpy as np

    scene_dir, ref, ratio, mat = Path(job[0]), job[1], job[2], job[3]
    maps, normal_map = mat if mat is not None else ((), False)
    P, N, UV, F = read_ply(scene_dir / ref)
    row = {"ref": ref, "faces_before": len(F), "vertices_before": len(P), "bytes_before": (scene_dir / ref).stat().st_size}
    diag = float(np.linalg.norm(P.max(0) - P.min(0)))
    welded, inverse = np.unique(P, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    textured = UV is not None and (mat is None or bool(maps) or normal_map)  # something reads the UVs
    per_face = textured and len(np.unique(np.c_[P, UV], axis=0)) / len(welded) > PER_FACE_UV
    if per_face and mat is None:  # a re-bake rewrites the material, which must be one
        return {**row, "status": "kept", "reason": "shapes_disagree_on_material"}
    for attempt in range(2):
        r = min(1.0, ratio * 2 ** attempt)
        if textured and not per_face:
            V, G, W = collapse(P, F, UV, int(len(F) * r))
            shift = uv_shift(P, UV, V, G, W)
        else:
            V, G, W = collapse(welded, inverse[F], None, int(len(F) * r))
            shift = 0.0
        surface = surface_error(P, F, V, G, diag)
        if surface <= MAX_SURFACE_ERROR and shift <= MAX_UV_TEXELS:
            break
    else:
        return {**row, "status": "kept", "reason": f"error over limits (surface {surface:.4f}, uv {shift:.2f} px)"}
    textures, bake_distance = {}, None
    cn = crease_normals(V, G)
    if per_face:
        vmap, faces, uvs, baked, bake_distance = rebake(P, UV, F, V, G, {m: scene_dir / m for m in maps})
        from PIL import Image

        for src, (pixels, mode) in baked.items():
            textures[src] = store(scene_dir, src, lambda tmp, px=pixels, md=mode: Image.fromarray(px, md).save(tmp, format="PNG"))
        corner_vertex, corner_uv = faces.reshape(-1), uvs[faces].reshape(-1, 2)  # cn is per corner of G = faces
        key = np.c_[corner_vertex[:, None], np.round(cn, 4)]
        corner_pos = V[vmap][corner_vertex]
    else:
        corner_pos = V[G].reshape(-1, 3)
        corner_uv = None if W is None else W.reshape(-1, 2)
        key = np.c_[G.reshape(-1, 1), np.round(cn, 4)] if W is None else np.c_[G.reshape(-1, 1), W.reshape(-1, 2), np.round(cn, 4)]
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    new = store(scene_dir, ref, lambda tmp: write_ply(tmp, corner_pos[first], cn[first],
                                                      None if corner_uv is None or not textured else corner_uv[first],
                                                      inv.reshape(-1, 3)))
    return {**row, "status": "reduced", "kind": "rebaked" if per_face else "atlas_kept" if textured else "untextured",
            "new": new, "textures": textures, "ratio": round(r, 4), "faces_after": len(G), "vertices_after": len(first),
            "surface_error": round(surface, 5), "uv_shift_px": round(shift, 3),
            "bake_distance_m": None if bake_distance is None else round(bake_distance, 5),
            "bytes_after": (scene_dir / new).stat().st_size + sum((scene_dir / t).stat().st_size for t in textures.values())}


def plan(scene_dir: Path, fixed: float | None, min_faces: int = MIN_FACES) -> list[tuple[str, float, str, tuple | None]]:
    """(mesh ref, target fraction, role, material (see decimate_mesh)) for each mesh worth reducing."""
    roles: dict[str, str] = {}
    materials: dict[str, set] = {}
    for f in sorted(scene_dir.glob("*.xml")):
        for _, ref, body in shape_meshes(f.read_text()):
            roles[ref] = max(roles.get(ref, "plain"), role(body), key=ROLES.index)
            materials.setdefault(ref, set()).add(material(body))
    out = []
    for ref, kind in sorted(roles.items()):
        if kind == "emitter" or not ref.endswith(".ply"):
            continue
        with (scene_dir / ref).open("rb") as fh:  # face count from the PLY header
            found = re.search(r"element face (\d+)", fh.read(1024).decode("latin-1"))
        faces = int(found.group(1)) if found else 0
        if faces < min_faces:
            continue
        ratio = fixed if fixed is not None else ladder(faces)
        if kind == "optical":
            ratio = max(ratio, OPTICAL_RATIO)
        mats = materials[ref]
        if ratio < 1.0:
            out.append((ref, ratio, kind, next(iter(mats)) if len(mats) == 1 else None))
    return out


def rewrite_shape(body: str, mesh: str, row: dict) -> str:
    """A shape's body naming its reduced mesh, and for a re-baked one its new textures without the normal map."""
    body = FILENAME_RE.sub(lambda m: f"{m.group(1)}{mesh if m.group(2) == row['ref'] else row['textures'].get(m.group(2), m.group(2))}{m.group(3)}", body)
    if row.get("kind") != "rebaked" or 'type="normalmap"' not in body:
        return body
    start, end = body.index("<bsdf"), body.rindex("</bsdf>") + len("</bsdf>")
    bsdf = ET.fromstring(body[start:end])
    for parent in [bsdf, *bsdf.iter("bsdf")]:
        for i, child in enumerate(list(parent)):
            if child.tag == "bsdf" and child.get("type") == "normalmap":
                parent.remove(child)
                parent.insert(i, child.find("bsdf"))
    if bsdf.get("type") == "normalmap":
        bsdf = bsdf.find("bsdf")
    return body[:start] + ET.tostring(bsdf, encoding="unicode") + body[end:]


def decimate_scene(scene_dir: Path, jobs: int, fixed: float | None, dry_run: bool, keep_unused: bool,
                   min_faces: int = MIN_FACES) -> dict:
    with scene_lock(scene_dir):  # one writer per scene (compact_assets.scene_lock)
        return _decimate_scene(scene_dir, jobs, fixed, dry_run, keep_unused, min_faces)


def _decimate_scene(scene_dir: Path, jobs: int, fixed: float | None, dry_run: bool, keep_unused: bool,
                    min_faces: int = MIN_FACES) -> dict:
    started = time.time()
    _mitsuba()
    todo = plan(scene_dir, fixed, min_faces)
    rows = []
    with ProcessPoolExecutor(max(1, jobs), mp_context=multiprocessing.get_context("spawn")) as pool:
        for row, (_, _, kind, _) in zip(pool.map(decimate_mesh, [(str(scene_dir), r, q, m) for r, q, _, m in todo]), todo):
            rows.append({**row, "role": kind})
            print(f"  {row['ref']}: {row['faces_before']} -> {row.get('faces_after', row['faces_before'])} "
                  f"{row.get('kind', row['status'])} {row.get('reason', '')}".rstrip(), flush=True)
    done = {r["ref"]: r for r in rows if r["status"] == "reduced"}
    if dry_run:
        made = {r["new"] for r in done.values()} | {t for r in done.values() for t in r["textures"].values()}
        for new in made - xml_refs(scene_dir):
            (scene_dir / new).unlink(missing_ok=True)
    else:
        for f in sorted(scene_dir.glob("*.xml")):
            text = f.read_text()

            def shape(m):
                ref = FILENAME_RE.search(m.group(3))
                row = done.get(ref.group(2)) if ref else None
                return m.group(0) if row is None else m.group(1) + rewrite_shape(m.group(3), row["new"], row) + m.group(4)

            new = PLY_SHAPE_RE.sub(shape, text)
            if new != text:
                tmp = f.with_suffix(".xml.tmp")
                tmp.write_text(new)
                os.replace(tmp, f)
        if not keep_unused:
            kept = xml_refs(scene_dir) | json_refs(scene_dir)
            for old in {r["ref"] for r in done.values()} | {s for r in done.values() for s in r["textures"]}:
                if old not in kept:
                    (scene_dir / old).unlink(missing_ok=True)
    reduced = list(done.values())
    summary = {
        "at": datetime.now(timezone.utc).isoformat(), "policy": (f"ratio={fixed}" if fixed else "ir_semantic_lod_v1") + (f", min_faces={min_faces}" if min_faces != MIN_FACES else ""),
        "dry_run": dry_run, "meshes_reduced": len(reduced),
        "by_kind": {k: sum(1 for r in reduced if r["kind"] == k) for k in ("atlas_kept", "rebaked", "untextured")},
        "faces": [sum(r["faces_before"] for r in reduced), sum(r["faces_after"] for r in reduced)],
        "bytes": [sum(r["bytes_before"] for r in reduced), sum(r["bytes_after"] for r in reduced)],
        "kept": {r["ref"]: r["reason"] for r in rows if r["status"] == "kept"},
        "meshes": rows,
    }
    meta_path = scene_dir / "scene.json"
    if not dry_run and reduced and meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        meta["mesh_decimation"] = summary
        tmp = meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=1))
        os.replace(tmp, meta_path)
    print(f"{scene_dir.name}: {len(reduced)} of {len(todo)} meshes reduced {summary['by_kind']}, triangles "
          f"{summary['faces'][0]} -> {summary['faces'][1]}, {summary['bytes'][0] / 1e9:.2f} -> "
          f"{summary['bytes'][1] / 1e9:.2f} GB, {len(summary['kept'])} kept "
          f"({', '.join(sorted(set(summary['kept'].values()))) or '-'}){' [dry run]' if dry_run else ''} "
          f"in {time.time() - started:.0f} s", flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--scene", action="append", default=[], help="scene id (repeatable; default: every scene)")
    ap.add_argument("--ratio", type=float, default=None, help="one retained triangle fraction instead of the ladder")
    ap.add_argument("--min-faces", type=int, default=MIN_FACES,
                    help=f"meshes under this many triangles are kept (default {MIN_FACES}; a scene of many mid-size "
                         "product meshes, like a market, holds most of its triangles below it)")
    ap.add_argument("--jobs", type=int, default=4, help="meshes reduced at once")
    ap.add_argument("--dry-run", action="store_true", help="measure and report, change nothing")
    ap.add_argument("--keep-unused", action="store_true", help="keep the full-resolution files no XML names any more")
    args = ap.parse_args()
    import pymeshlab  # noqa: F401  (fail before touching anything)
    import xatlas  # noqa: F401

    _mitsuba()
    scenes = args.scene or sorted(p.parent.name for p in (args.pack / "scenes").glob("*/scene.json"))
    for scene in scenes:
        decimate_scene((args.pack / "scenes" / scene).resolve(), args.jobs, args.ratio, args.dry_run, args.keep_unused,
                       args.min_faces)
    return 0


if __name__ == "__main__":
    sys.exit(main())
