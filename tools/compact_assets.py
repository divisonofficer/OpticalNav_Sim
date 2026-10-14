#!/usr/bin/env python3
"""Shrink what a pack scene loads: identical textures become one file, OBJ meshes become binary PLY.

    source /home/jinnyeong/robomituba-build/mitsuba3/setpath.sh
    python3.10 tools/compact_assets.py --pack packs/opticalnav-v0.2 [--scene SCENE ...] [--jobs 6] [--keep-unused]

Run it after tools/build_pack.py; a second run changes nothing. ``--if-needed`` skips scenes already
compacted, so it is safe to run on every start (a pack shared before compaction is converted once);
needs_compaction / compact_pack do the same from Python.

* Textures. build_pack names an asset by its source path, so one image under several paths
  becomes several files. Here each texture takes its content hash as name (``assets/tex_<sha1>.png``,
  a hard link to an existing copy). Then equal materials name equal files, and the loader's
  merge_equivalent folds them into one object: a texture is read once, a BSDF traced once by every
  freeze recording, and the meshes sharing it merge (infinigen_apartment_natural_v1_20268504: 1227 live
  PNGs hold 399 images; 577 inline BSDFs become 172).
* Meshes. Parsing ASCII OBJ is most of a scene load (3.1 GB, 12 M triangles in that apartment; its
  largest mesh takes 3.2 s as OBJ, 0.07 s as PLY). Each OBJ is loaded by Mitsuba's own OBJ plugin and
  written with Mesh.write_ply, so the PLY holds what the scene loaded before (UVs already flipped,
  vertices already split per normal and UV). Every PLY is loaded back and compared: positions, faces and
  UVs bit for bit, normals within 1e-6 (the PLY loader renormalises: 1 ulp). A mesh that fails stays OBJ.

Pack assets are hard links into the robomituba project, so no file is ever written in place: new
files go next to the old ones, the scene XMLs are rewritten to name them, and assets no XML or JSON of
the scene names any more are unlinked (``--keep-unused`` keeps them). scene.json records the run,
which also makes tools/export_pack.py pack the scene again.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import multiprocessing
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

FILENAME_RE = re.compile(r'(<string name="filename" value=")(assets/[^"]+)(")')
# build_pack writes a mesh's filename as the shape's first child
OBJ_SHAPE_RE = re.compile(r'<shape type="obj"(\s[^>]*)?>(\s*<string name="filename" value=")(assets/[^"]+\.obj)(")')
MESH_SUFFIXES = (".obj", ".ply", ".serialized")
TEXTURE_NAME = re.compile(r"assets/tex_[0-9a-f]{20}\.[a-z0-9]+$")  # a texture already named by its content
OBJ_ONLY = ("flip_tex_coords",)  # OBJ-plugin properties a converted shape must not carry
NORMAL_TOLERANCE = 1e-6


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def texture_names(scene_dir: Path, refs) -> dict[str, str]:
    """Texture ref -> content-addressed ref (hard-linked into place); refs already so named map to themselves."""
    out = {}
    for ref in sorted(refs):
        path = scene_dir / ref
        new = f"assets/tex_{sha1(path)[:20]}{path.suffix.lower()}"
        if not (scene_dir / new).exists():
            os.link(path, scene_dir / new)
        out[ref] = new
    return out


def rewrite(text: str, textures: dict[str, str], meshes: dict[str, str]) -> str:
    """A scene file naming the compacted assets: textures renamed, converted OBJ shapes turned into PLY shapes."""
    def shape(m):
        new = meshes.get(m.group(3))
        return m.group(0) if new is None else f'<shape type="ply"{m.group(1) or ""}>{m.group(2)}{new}{m.group(4)}'

    text = OBJ_SHAPE_RE.sub(shape, text)
    text = FILENAME_RE.sub(lambda m: f"{m.group(1)}{textures.get(m.group(2), m.group(2))}{m.group(3)}", text)
    left = [ref for ref in meshes if f'value="{ref}"' in text]
    if left:
        raise ValueError(f"OBJ refs outside a <shape type=\"obj\"> filename: {left[:3]}")
    return text


_MI = []


def _mitsuba():
    if not _MI:
        import mitsuba as mi

        mi.set_variant("scalar_rgb")
        mi.set_log_level(mi.LogLevel.Warn)
        _MI.append(mi)
    return _MI[0]


def _arrays(mi, mesh) -> dict:
    import numpy as np

    params = mi.traverse(mesh)
    return {k: np.array(params[k]) for k in ("vertex_positions", "vertex_normals", "vertex_texcoords", "faces")
            if k in params}


def convert_mesh(job: tuple[str, str]) -> dict:
    """One OBJ -> assets/mesh_<sha1>.ply, checked against the OBJ as Mitsuba loads it."""
    import numpy as np

    scene_dir, ref = Path(job[0]), job[1]
    mi = _mitsuba()
    obj = mi.load_dict({"type": "obj", "filename": str(scene_dir / ref)})
    tmp = scene_dir / "assets" / f".{Path(ref).stem}.{os.getpid()}.ply.tmp"
    try:
        obj.write_ply(str(tmp))
        ply = mi.load_dict({"type": "ply", "filename": str(tmp)})
        a, b = _arrays(mi, obj), _arrays(mi, ply)
        bad = [k for k in a if k not in b or a[k].shape != b[k].shape]
        bad += [k for k in a if k not in bad and not (np.allclose(a[k], b[k], rtol=0, atol=NORMAL_TOLERANCE)
                                                      if k == "vertex_normals" else np.array_equal(a[k], b[k]))]
        bad += [k for k in b if k not in a]
        if bad:
            return {"ref": ref, "error": f"PLY differs from OBJ in {bad}"}
        new = f"assets/mesh_{sha1(tmp)[:20]}.ply"
        if (scene_dir / new).exists():  # an identical mesh converted already
            tmp.unlink()
        else:
            os.replace(tmp, scene_dir / new)
        return {"ref": ref, "new": new, "triangles": int(obj.face_count()),
                "obj_bytes": (scene_dir / ref).stat().st_size, "ply_bytes": (scene_dir / new).stat().st_size}
    finally:
        tmp.unlink(missing_ok=True)


def xml_refs(scene_dir: Path) -> set[str]:
    return {m.group(2) for f in scene_dir.glob("*.xml") for m in FILENAME_RE.finditer(f.read_text())}


def json_refs(scene_dir: Path) -> set[str]:
    """assets/ paths a scene's JSON files mention (kept, to be safe, though no tool reads assets through them)."""
    return {r for f in scene_dir.rglob("*.json") for r in re.findall(r'assets/[^"\\\s]+', f.read_text())}


def needs_compaction(scene_dir: Path) -> bool:
    """Whether a scene's XMLs still name an OBJ mesh or a texture not yet named by its content. A text scan
    (no Mitsuba, milliseconds): what a server or fetch_pack runs before deciding to compact."""
    for f in scene_dir.glob("*.xml"):
        text = f.read_text()
        if OBJ_SHAPE_RE.search(text):
            return True
        if any(not ref.lower().endswith(MESH_SUFFIXES) and not TEXTURE_NAME.match(ref)
               for ref in (m.group(2) for m in FILENAME_RE.finditer(text))):
            return True
    return False


LOCK_NAME = ".compact.lock"


class SceneBusy(RuntimeError):
    """Another process holds the scene's lock (see scene_lock)."""


@contextlib.contextmanager
def scene_lock(scene_dir: Path):
    """Exclusive lock (flock on <scene>/.compact.lock) for a tool rewriting a scene: compact_assets,
    decimate_meshes, and the render server's compaction on start. A second writer gets
    SceneBusy at once instead of racing the first one over the same XMLs and assets."""
    handle = os.open(scene_dir / LOCK_NAME, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SceneBusy(f"{scene_dir.name}: another process is rewriting this scene") from None
        os.ftruncate(handle, 0)
        os.write(handle, f"{os.getpid()}\n".encode())
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        os.close(handle)


def compact_pack(pack: Path, scenes=(), jobs: int = 6, keep_unused: bool = False, if_needed: bool = True) -> dict:
    """Compact the scenes of a pack (default: all), by default only those that need it. Scene id -> summary,
    None for a scene skipped as already compacted, or "busy" for one another process is rewriting."""
    pack = Path(pack)
    scenes = list(scenes) or sorted(p.parent.name for p in (pack / "scenes").glob("*/scene.json"))
    out = {}
    for scene in scenes:
        scene_dir = (pack / "scenes" / scene).resolve()
        if if_needed and not needs_compaction(scene_dir):
            out[scene] = None
            continue
        _mitsuba()
        try:
            out[scene] = compact_scene(scene_dir, jobs, keep_unused)
        except SceneBusy as exc:
            print(f"skipped: {exc}", flush=True)
            out[scene] = "busy"
    return out


def compact_scene(scene_dir: Path, jobs: int, keep_unused: bool) -> dict:
    with scene_lock(scene_dir):
        return _compact_scene(scene_dir, jobs, keep_unused)


def _compact_scene(scene_dir: Path, jobs: int, keep_unused: bool) -> dict:
    started = time.time()
    xmls = sorted(scene_dir.glob("*.xml"))
    texts = {f: f.read_text() for f in xmls}
    for f, text in texts.items():
        for m in re.finditer(r'<shape type="obj"[^>]*>(.*?)</shape>', text, re.S):
            if any(f'name="{p}"' in m.group(1) for p in OBJ_ONLY):
                raise ValueError(f"{f.name}: an OBJ shape sets an OBJ-only property ({OBJ_ONLY})")
    refs = {m.group(2) for text in texts.values() for m in FILENAME_RE.finditer(text)}
    objs = {m.group(3) for text in texts.values() for m in OBJ_SHAPE_RE.finditer(text)}
    textures = texture_names(scene_dir, {r for r in refs if not r.lower().endswith(MESH_SUFFIXES)})

    meshes, failed, stats = {}, [], []
    # spawn: a process forked from one where Mitsuba's thread pool runs can deadlock
    with ProcessPoolExecutor(max(1, jobs), mp_context=multiprocessing.get_context("spawn")) as pool:
        for i, row in enumerate(pool.map(convert_mesh, [(str(scene_dir), r) for r in sorted(objs)]), 1):
            if "error" in row:
                failed.append(row)
                print(f"  ! {row['ref']}: {row['error']} (kept as OBJ)", flush=True)
            else:
                meshes[row["ref"]] = row["new"]
                stats.append(row)
            if i % 50 == 0 or i == len(objs):
                print(f"  meshes {i}/{len(objs)}", flush=True)

    for f, text in texts.items():
        new = rewrite(text, textures, meshes)
        if new != text:
            tmp = f.with_suffix(".xml.tmp")
            tmp.write_text(new)
            os.replace(tmp, f)

    named = xml_refs(scene_dir)
    missing = sorted(r for r in named if not (scene_dir / r).exists())
    if missing:
        raise FileNotFoundError(f"{scene_dir.name}: rewritten scene names missing assets {missing[:3]}")
    kept = named | json_refs(scene_dir)
    unused = sorted(p for p in (scene_dir / "assets").iterdir() if p.is_file() and f"assets/{p.name}" not in kept)
    unused_bytes = sum(p.stat().st_size for p in unused)  # mostly hard links: the disk freed is less
    if not keep_unused:
        for p in unused:
            p.unlink()

    summary = {
        "at": datetime.now(timezone.utc).isoformat(),
        "textures": [len(textures), len(set(textures.values()))],  # refs -> distinct files
        "meshes_converted": len(meshes), "meshes_failed": [r["ref"] for r in failed],
        "mesh_files": len(set(meshes.values())),
        "triangles": sum(r["triangles"] for r in stats),
        "obj_bytes": sum(r["obj_bytes"] for r in stats),
        "ply_bytes": sum(r["ply_bytes"] for r in {r["new"]: r for r in stats}.values()),
        "unused_assets": len(unused), "unused_removed": not keep_unused, "unused_bytes": unused_bytes,
    }
    meta_path = scene_dir / "scene.json"
    changed = bool(meshes) or any(k != v for k, v in textures.items())
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else None
    if meta is not None and (changed or (unused and not keep_unused and "asset_compaction" in meta)):
        if changed:
            meta["asset_compaction"] = summary
        else:  # a later run that only unlinks what the first one kept: its counts stay
            meta["asset_compaction"].update({k: summary[k] for k in ("unused_assets", "unused_removed", "unused_bytes")})
        tmp = meta_path.with_suffix(".json.tmp")  # replaced, never written through a hard link
        tmp.write_text(json.dumps(meta, indent=1))
        os.replace(tmp, meta_path)
    print(f"{scene_dir.name}: textures {summary['textures'][0]} refs -> {summary['textures'][1]} files, "
          f"meshes {len(meshes)} OBJ -> {summary['mesh_files']} PLY ({summary['obj_bytes'] / 1e9:.2f} -> "
          f"{summary['ply_bytes'] / 1e9:.2f} GB), "
          f"{len(failed)} kept OBJ, {len(unused)} unused assets {'kept' if keep_unused else 'unlinked'} "
          f"({unused_bytes / 1e9:.2f} GB) in {time.time() - started:.0f} s", flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--scene", action="append", default=[], help="scene id (repeatable; default: every scene)")
    ap.add_argument("--jobs", type=int, default=6, help="meshes converted at once (each holds one mesh in memory)")
    ap.add_argument("--keep-unused", action="store_true", help="keep assets the rewritten scene no longer names")
    ap.add_argument("--if-needed", action="store_true", help="skip scenes already compacted (a text scan)")
    args = ap.parse_args()
    _mitsuba()  # fail before touching anything when Mitsuba is not importable
    done = compact_pack(args.pack, args.scene, args.jobs, args.keep_unused, args.if_needed)
    skipped = [s for s, r in done.items() if r is None]
    if skipped:
        print(f"already compacted: {len(skipped)} scene(s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
