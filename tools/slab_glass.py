#!/usr/bin/env python3
"""Give single-sheet glass a thickness, in pack scenes already built, so it refracts and comes out straight.

    python3 tools/slab_glass.py --pack packs/opticalnav-sim-v1 [--scene SCENE ...] [--thickness 0.006] [--dry-run]

Infinigen's glass partitions, window panes and big glass panes are one zero-thickness sheet each (a mesh with a
flat bounding box: 5.67 x 2.5 x 0.0 m; the 6 mm slabs the importer's newer generators build are left alone), and robomituba's importer gives them ``dielectric`` with ior 1.5. A
dielectric refracts a ray at its surface and the ray goes on bent: with no back face to leave through, it travels
on as if inside a half-space of glass. A ray crossing such a pane at 40 degrees leaves it 23 degrees off
(measured, glass partition of floorO2), so everything seen through the glass is shifted and warped, more the more
obliquely it is seen.

``thindielectric`` would give the straight ray but has no polarized variant ("This BSDF does not have support for
polarized variants": the reflection loses its polarization), so a sheet becomes a closed slab instead: the sheet's
triangles twice, ``--thickness`` metres apart along its normal, the back copy turned inside out. A ray enters by one
face and leaves by the other, parallel to where it came (displaced by about a millimetre), and both faces use the
polarized ``dielectric``.

Only shapes whose mesh is flat (thinnest side under 3 mm) and wider than 10 cm, with a ``dielectric`` (or the
``thindielectric`` an earlier version of this tool wrote) material, are changed. The new meshes are written as
``assets/mesh_<sha1>.ply``, the scene XMLs rewritten, the old meshes no file names any more unlinked, and
scene.json records the shapes under ``glass_slabs``. Rebuild the scene to undo it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compact_assets import json_refs, scene_lock, xml_refs  # noqa: E402
from decimate_meshes import read_ply, write_ply  # noqa: E402
from open_doors import SHAPE, bounds  # noqa: E402

GLASS = re.compile(r'<bsdf type="(?:thin)?dielectric">(\s*<float name="int_ior" value="([^"]+)" />\s*)</bsdf>')
FILENAME = re.compile(r'(<string name="filename" value=")([^"]+\.ply)(")')


def flat(scene_dir: Path, ref: str, max_thickness: float = 0.003) -> bool:
    box = bounds(scene_dir / ref)
    if box is None:
        return False
    ext = sorted(float(v) for v in box[1] - box[0])
    return ext[0] < max_thickness and ext[1] > 0.1


def slab(scene_dir: Path, ref: str, thickness: float) -> str:
    """The mesh ``ref`` as a closed-ish slab (front and back faces), written as assets/mesh_<sha1>.ply."""
    pos, nor, uv, faces = read_ply(scene_dir / ref)
    a, b, c = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    cross = np.cross(b - a, c - a)
    n = cross.sum(0)
    n = n / np.linalg.norm(n)  # a flat sheet: one normal, by its winding
    count = len(pos)
    front, back = pos + n * thickness / 2, pos - n * thickness / 2
    new_pos = np.vstack([front, back])
    new_nor = np.vstack([np.tile(n, (count, 1)), np.tile(-n, (count, 1))])
    new_uv = np.vstack([uv, uv]) if uv is not None else None
    new_faces = np.vstack([faces, faces[:, ::-1] + count])  # the back copy turned inside out
    tmp = scene_dir / "assets" / f".slab_{os.getpid()}.ply"
    write_ply(tmp, new_pos, new_nor, new_uv, new_faces)
    new = f"assets/mesh_{hashlib.sha1(tmp.read_bytes()).hexdigest()[:20]}.ply"
    if (scene_dir / new).exists():
        tmp.unlink()
    else:
        os.replace(tmp, scene_dir / new)
    return new


def slab_scene(scene_dir: Path, thickness: float = 0.006, dry_run: bool = False) -> dict:
    with scene_lock(scene_dir):
        files = sorted(scene_dir.glob("*.xml"))
        text = files[0].read_text()
        todo, made, glass = {}, {}, 0
        for _, sid, body, _ in SHAPE.findall(text):
            if not GLASS.search(body):
                continue
            glass += 1
            ref = FILENAME.search(body)
            if ref and ref.group(2) not in todo:
                todo[ref.group(2)] = flat(scene_dir, ref.group(2))
        sheets = {r for r, ok in todo.items() if ok}
        summary = {"glass_shapes": glass, "sheet_meshes": len(sheets), "thickness_m": thickness, "dry_run": dry_run}
        if dry_run or not sheets:
            return summary
        for ref in sorted(sheets):
            made[ref] = slab(scene_dir, ref, thickness)
        count = 0
        for f in files:
            def shape(m):
                nonlocal count
                ref = FILENAME.search(m.group(3))
                if not GLASS.search(m.group(3)) or ref is None or ref.group(2) not in made:
                    return m.group(0)
                count += 1
                body = FILENAME.sub(lambda r: r.group(1) + made[r.group(2)] + r.group(3), m.group(3), count=1)
                return m.group(1) + GLASS.sub(r'<bsdf type="dielectric">\1</bsdf>', body) + m.group(4)

            new = SHAPE.sub(shape, f.read_text())
            tmp = f.with_suffix(".xml.tmp")
            tmp.write_text(new)
            os.replace(tmp, f)
        kept = xml_refs(scene_dir) | json_refs(scene_dir)
        for old in made:
            if old not in kept:
                (scene_dir / old).unlink(missing_ok=True)
        summary["slab_shapes"] = count
        meta_path = scene_dir / "scene.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            meta.pop("thin_glass", None)
            meta["glass_slabs"] = summary
            tmp = meta_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(meta, indent=1))
            os.replace(tmp, meta_path)
        return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--thickness", type=float, default=0.006)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    scenes = args.scene or sorted(p.parent.name for p in (args.pack / "scenes").glob("*/base.xml"))
    total = 0
    for name in scenes:
        s = slab_scene((args.pack / "scenes" / name).resolve(), args.thickness, args.dry_run)
        total += s["sheet_meshes"]
        if s["sheet_meshes"]:
            print(f"{name}: {s['sheet_meshes']} sheet meshes of {s['glass_shapes']} glass shapes"
                  f"{' [dry run]' if args.dry_run else ' -> slabs of %.0f mm' % (args.thickness * 1000)}", flush=True)
    print(f"{total} sheet meshes in {len(scenes)} scenes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
