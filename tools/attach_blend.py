#!/usr/bin/env python3
"""Put a scene's .blend into its pack folder, for the Cycles render backend and for distribution.

    python3 tools/attach_blend.py --pack packs/opticalnav-sim-v1 [--scene SCENE ...] [--blend FILE] [--tsv FILE] [--dry-run]

The simulator's own scenes (``sim_v2a_REF_PH19`` ...) come from a promoted ``.blend`` (robomituba's
promoted_ref_scenes tsv lists them by id: ``REF_PH19``). The render server's ``"renderer": "cycles"`` renders that file
in Blender (opticalnav_sim/cycles_backend.py), so it travels with the pack: the file is opened in Blender, every
external image (textures, the sky HDR: ``//../../_ph192021/tex/facade.png``) is packed into it, and it is saved as
``scenes/<scene>/blend/scene.blend``. scene.json gets ``"blend": {"file", "origin_offset", "source", "images_packed",
"images_missing"}``; the origin offset is the importer's (the authoring map's ``origin_offset``), which places the
Blender scene in the Mitsuba one (see cycles_backend.py). An image that cannot be found is listed, and renders as
Blender's pink.

Run from the repo; Blender is started like the render server does it (OPTICALNAV_BLENDER, the WSL driver dirs).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

IMPORTS = Path("/jarvis/project/robomituba/out/infinigen_imports")
TSV = Path("/bean/ir_dataset_work/codex_review/promoted_ref_scenes_20261102.tsv")
INSIDE = os.environ.get("OPTICALNAV_ATTACH_INSIDE") == "1"  # set for the Blender process this script starts


def pack_inside(src: str, dst: str, report: str) -> None:
    """Inside Blender: pack the images, save a compressed copy, write what was packed and what is missing."""
    import bpy
    bpy.ops.wm.open_mainfile(filepath=src)
    missing = []
    for im in list(bpy.data.images):  # one by one: pack_all fails as a whole on a file that is gone
        if im.source != "FILE" or im.packed_file:
            continue
        if not Path(bpy.path.abspath(im.filepath)).is_file():
            missing.append(im.filepath)
            continue
        try:
            im.pack()
        except RuntimeError:
            missing.append(im.filepath)
    packed = sum(1 for im in bpy.data.images if im.packed_file)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=dst, compress=True, copy=True)
    Path(report).write_text(json.dumps({"images_packed": packed, "images_missing": sorted(set(missing))}))


def find_source(scene: str, tsv: Path) -> Path | None:
    ident = scene.removeprefix("sim_v2a_")
    for line in tsv.read_text().splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) >= 5 and cols[1] == ident and Path(cols[4]).is_file():
            return Path(cols[4])
    return None


def attach(pack: Path, scene: str, blend: Path | None, tsv: Path, imports: Path, dry_run: bool, offset: list | None = None) -> str:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from compact_assets import scene_lock
    from opticalnav_sim import cycles_backend as cb

    scene_dir = pack / "scenes" / scene
    src = blend or find_source(scene, tsv)
    if src is None:
        return f"{scene}: no .blend in {tsv.name} (--blend FILE)"
    amap = imports / f"{scene}__authoring_map.json"
    if offset is None:
        offset = json.loads(amap.read_text())["metadata"].get("origin_offset") if amap.is_file() else None
    if offset is None:
        return f"{scene}: no origin_offset (authoring map {amap})"
    if dry_run:
        return f"{scene}: would pack {src} ({src.stat().st_size / 1e6:.1f} MB), origin offset {offset}"
    dst, report = scene_dir / "blend" / "scene.blend", scene_dir / "blend" / ".report.json"
    env = dict(os.environ, OPTICALNAV_ATTACH_INSIDE="1",
               LD_LIBRARY_PATH=f"{cb.DRIVER_DIRS}:{cb.BLENDER.parent / 'lib'}:{cb.lib_dir()}")
    with scene_lock(scene_dir):
        r = subprocess.run([str(cb.BLENDER), "-b", "--python", str(Path(__file__).resolve()), "--", str(src), str(dst), str(report)],
                           env=env, capture_output=True, text=True)
        if r.returncode or not report.is_file():
            return f"{scene}: Blender failed ({r.returncode}): {' | '.join((r.stdout + r.stderr).strip().splitlines()[-3:])}"
        info = json.loads(report.read_text())
        report.unlink()
        meta_path = scene_dir / "scene.json"
        meta = json.loads(meta_path.read_text())
        meta["blend"] = {"file": "blend/scene.blend", "origin_offset": offset, "source": str(src), **info}
        tmp = meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=1))
        os.replace(tmp, meta_path)
    miss = info["images_missing"]
    return (f"{scene}: {dst.relative_to(pack)} {dst.stat().st_size / 1e6:.1f} MB, {info['images_packed']} images packed"
            + (f", MISSING {len(miss)}: {miss[:3]}" if miss else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--blend", type=Path, help="the .blend (one --scene only; default: the promoted tsv's)")
    ap.add_argument("--tsv", type=Path, default=TSV)
    ap.add_argument("--imports", type=Path, default=IMPORTS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    scenes = args.scene or sorted(p.parent.name for p in (args.pack / "scenes").glob("*/base.xml"))
    if args.blend and len(scenes) != 1:
        raise SystemExit("--blend takes exactly one --scene")
    for scene in scenes:
        print(attach(args.pack.resolve(), scene, args.blend, args.tsv, args.imports, args.dry_run), flush=True)
    return 0


if __name__ == "__main__":
    if INSIDE:
        pack_inside(*sys.argv[sys.argv.index("--") + 1:][:3])
    else:
        raise SystemExit(main())
