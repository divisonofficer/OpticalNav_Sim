#!/usr/bin/env python3
"""Cut the resolution of textures that sit on small objects, in an already built (and compacted) pack scene.

    python3 tools/shrink_textures.py --pack packs/opticalnav-sim-v1 [--scene SCENE ...] [--px-per-m 512]
        [--min-px 128] [--dry-run]

A texture's size follows the largest object that uses it: ``px-per-m`` texels per metre of that object's longest
side, rounded up to a power of two, at least ``min-px`` and never more than the texture has. At the default 512
px/m a 0.2 m bottle gets 128 px, a 0.5 m crate 256 px and anything of a metre or more keeps 512; walls, floors
and ceilings (2048 px, several metres) are untouched. Textures no shape of the scene uses stay as they are.

Why: the GPU holds every texture as float32. sim_v2a_floorM2 (a market, 1772 objects) has 4275 textures of 512 px,
15 GB, on a 32 GB card that also holds the BVH, the recordings and the raster preview: frames took 10 to 112 s
and the server died with CUDA out of memory. Most of those textures cover a product a few centimetres wide.

New textures are written as ``assets/tex_<sha1>.png`` and the scene XMLs rewritten, as compact_assets does; the
old ones no file names any more are unlinked (build_pack and compact_assets rebuild them). scene.json records
the run under ``texture_shrink``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compact_assets import FILENAME_RE, json_refs, scene_lock, xml_refs  # noqa: E402
from decimate_meshes import shape_meshes  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "char": "i1",
              "short": "<i2", "ushort": "<u2", "int": "<i4", "uint": "<u4"}


def extent(path: Path) -> float | None:
    """Longest side of a binary little-endian PLY's bounding box (vertex positions), or None if unreadable."""
    with path.open("rb") as f:
        head = b""
        while b"end_header\n" not in head:
            block = f.read(4096)
            if not block:
                return None
            head += block
        header, body = head.split(b"end_header\n", 1)
        lines = header.decode("latin-1").splitlines()
        if "format binary_little_endian" not in header.decode("latin-1"):
            return None
        count, fields, in_vertex = 0, [], False
        for line in lines:
            parts = line.split()
            if parts[:2] == ["element", "vertex"]:
                count, in_vertex = int(parts[2]), True
            elif parts[:1] == ["element"]:
                in_vertex = False
            elif in_vertex and parts[:1] == ["property"]:
                if parts[1] == "list" or parts[1] not in _PLY_TYPES:
                    return None
                fields.append((parts[2], _PLY_TYPES[parts[1]]))
        if not count or not {"x", "y", "z"} <= {n for n, _ in fields}:
            return None
        dtype = np.dtype(fields)
        f.seek(len(header) + len(b"end_header\n"))
        v = np.fromfile(f, dtype, count)
    pos = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    return float((pos.max(0) - pos.min(0)).max())


def target_px(current: int, size_m: float, px_per_m: float, min_px: int) -> int:
    want = max(min_px, 2 ** math.ceil(math.log2(max(size_m * px_per_m, 1.0))))
    return min(current, want)


def shrink(job: tuple[str, str, int]) -> tuple[str, str | None]:
    """(old ref, new ref) of one texture resized so its longest side is ``px``; None if nothing to do."""
    scene_dir, ref, px = Path(job[0]), job[1], job[2]
    with Image.open(scene_dir / ref) as im:
        w, h = im.size
        if max(w, h) <= px:
            return ref, None
        scale = px / max(w, h)
        out = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    tmp = scene_dir / f"assets/.shrink_{os.getpid()}_{hashlib.sha1(ref.encode()).hexdigest()[:8]}.png"
    out.save(tmp, optimize=False)
    digest = hashlib.sha1(tmp.read_bytes()).hexdigest()[:20]
    new = f"assets/tex_{digest}.png"
    if (scene_dir / new).exists():
        tmp.unlink()
    else:
        os.replace(tmp, scene_dir / new)
    return ref, new


def shrink_scene(scene_dir: Path, px_per_m: float = 512.0, min_px: int = 128, dry_run: bool = False,
                 jobs: int = 6) -> dict:
    with scene_lock(scene_dir):  # one writer per scene (compact_assets.scene_lock)
        return _shrink_scene(scene_dir, px_per_m, min_px, dry_run, jobs)


def _shrink_scene(scene_dir: Path, px_per_m: float, min_px: int, dry_run: bool, jobs: int) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    started = time.time()
    uses: dict[str, float] = defaultdict(float)  # texture ref -> its largest user's longest side
    sizes: dict[str, float | None] = {}
    for f in sorted(scene_dir.glob("*.xml")):
        for _, mesh, body in shape_meshes(f.read_text()):
            if mesh not in sizes:
                sizes[mesh] = extent(scene_dir / mesh)
            for m in FILENAME_RE.finditer(body):
                ref = m.group(2)
                if ref != mesh and ref.endswith((".png", ".jpg", ".jpeg")):
                    uses[ref] = max(uses[ref], sizes[mesh] if sizes[mesh] is not None else math.inf)
    todo, before, after = [], 0, 0
    for ref, size_m in sorted(uses.items()):
        with Image.open(scene_dir / ref) as im:
            w, h = im.size
            ch = max(len(im.getbands()), 3)
        px = target_px(max(w, h), size_m, px_per_m, min_px)
        before += w * h * ch * 4
        after += round(w * px / max(w, h)) * round(h * px / max(w, h)) * ch * 4 if px < max(w, h) else w * h * ch * 4
        if px < max(w, h):
            todo.append((str(scene_dir), ref, px))
    with ThreadPoolExecutor(max(1, jobs)) as pool:  # PIL releases the GIL while it resamples and encodes
        done = dict(pool.map(shrink, todo)) if not dry_run else {}
    renames = {old: new for old, new in done.items() if new}
    if renames:
        for f in sorted(scene_dir.glob("*.xml")):
            text = f.read_text()
            new = FILENAME_RE.sub(lambda m: f"{m.group(1)}{renames.get(m.group(2), m.group(2))}{m.group(3)}", text)
            if new != text:
                tmp = f.with_suffix(".xml.tmp")
                tmp.write_text(new)
                os.replace(tmp, f)
        kept = xml_refs(scene_dir) | json_refs(scene_dir)
        for old in renames:
            if old not in kept:
                (scene_dir / old).unlink(missing_ok=True)
    summary = {"at": datetime.now(timezone.utc).isoformat(), "px_per_m": px_per_m, "min_px": min_px,
               "dry_run": dry_run, "textures": len(uses), "shrunk": len(todo),
               "gpu_bytes_float32": [before, after]}
    meta_path = scene_dir / "scene.json"
    if not dry_run and renames and meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        meta["texture_shrink"] = summary
        tmp = meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=1))
        os.replace(tmp, meta_path)
    print(f"{scene_dir.name}: {len(todo)} of {len(uses)} textures shrunk, as float32 {before / 1e9:.1f} -> "
          f"{after / 1e9:.1f} GB{' [dry run]' if dry_run else ''} in {time.time() - started:.0f} s", flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--scene", action="append", default=[], help="scene id (repeatable; default: every scene)")
    ap.add_argument("--px-per-m", type=float, default=512.0, help="texels per metre of the largest object a texture is on")
    ap.add_argument("--min-px", type=int, default=128, help="smallest side a texture is cut to")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true", help="measure and report, change nothing")
    args = ap.parse_args()
    scenes = args.scene or sorted(p.parent.name for p in (args.pack / "scenes").glob("*/scene.json"))
    for scene in scenes:
        shrink_scene((args.pack / "scenes" / scene).resolve(), args.px_per_m, args.min_px, args.dry_run, args.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
