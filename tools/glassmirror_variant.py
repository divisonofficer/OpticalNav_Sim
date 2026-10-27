#!/usr/bin/env python3
"""Add the ``glassmirror`` variant (glass partitions, wall mirrors, floor mirrors) to the pack scenes of a render kit.

    python3 tools/glassmirror_variant.py --pack packs/opticalnav-sim-v1 --kit <jn1027_render_kit dir> [--scene SCENE ...]

The jn1027 render kit ships, per scene, a ``glassmirror.xml``: the scene's base XML of the kit's own date plus shapes
named ``gm_glass_partition_*``, ``gm_wall_mirror_*`` and ``gm_floor_mirror_*`` (with their ``opticalnav-obj`` notes) and
the BSDFs ``glassmirror_*`` they refer to. The scene layout of this pack is the same (every shape of the new base.xml is
in the kit's file; the added shapes use meshes and BSDFs that exist here), but its base.xml has been repaired since (glass
slabs, doors, grey and metal textures, light metadata), so the kit's file cannot replace it. This tool copies only
the added shapes, their notes and BSDFs into a new ``glassmirror.xml`` made from this pack's current ``base.xml``, and
registers the variant in scene.json (the entry the kit's own scene.json carries). A kit shape that names a file this
pack no longer has (floorO2's ``GlassPartitionFrost_*``, removed by open_doors.py) is left out. Rebuild or re-run it
after a scene is repacked.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compact_assets import scene_lock  # noqa: E402

SHAPE = re.compile(r'(?:  <!--opticalnav-obj:\{[^\n]*?\}-->\n)?  <shape type="[a-z_]+" id="([^"]+)">.*?\n  </shape>\n', re.S)
BSDF = re.compile(r'  <bsdf type="[a-z_]+" id="(glassmirror_[^"]+)">.*?\n  </bsdf>\n', re.S)


def variant_scene(scene_dir: Path, kit: Path) -> dict:
    name = scene_dir.name
    patch = kit / "patch" / name / "glassmirror.xml"
    if not patch.is_file():
        return {"added": 0, "skipped": "no patch in the kit"}
    with scene_lock(scene_dir):
        base = (scene_dir / "base.xml").read_text()
        have = set(re.findall(r'<shape type="[a-z_]+" id="([^"]+)"', base))
        text = patch.read_text()
        bsdfs = [m.group(0) for m in BSDF.finditer(text) if f'id="{m.group(1)}"' not in base]
        shapes, dropped = [], []
        for m in SHAPE.finditer(text):
            sid = m.group(1)
            if sid in have:
                continue
            if any(not (scene_dir / f).is_file() for f in re.findall(r'value="(assets/[^"]+)"', m.group(0))):
                dropped.append(sid)
                continue
            shapes.append(m.group(0))
        end = base.rindex("</scene>")
        (scene_dir / "glassmirror.xml.tmp").write_text(base[:end] + "".join(bsdfs) + "".join(shapes) + base[end:])
        os.replace(scene_dir / "glassmirror.xml.tmp", scene_dir / "glassmirror.xml")
        meta_path = scene_dir / "scene.json"
        meta = json.loads(meta_path.read_text())
        kit_meta = json.loads((kit / "patch" / name / "scene.json").read_text())
        entry = dict(kit_meta["variants"]["glassmirror"])
        entry["note"] = "base.xml of this pack + the kit's gm_* shapes (tools/glassmirror_variant.py)"
        meta["variants"]["glassmirror"] = entry
        meta["glassmirror_variant"] = {"shapes_added": len(shapes), "bsdfs_added": len(bsdfs), "dropped": dropped}
        tmp = meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=1))
        os.replace(tmp, meta_path)
    return {"added": len(shapes), "bsdfs": len(bsdfs), "dropped": dropped}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--kit", required=True, type=Path, help="the unpacked jn1027_render_kit folder")
    ap.add_argument("--scene", action="append", default=[])
    args = ap.parse_args()
    scenes = args.scene or sorted(json.loads((args.kit / "manifest.json").read_text())["scenes"])
    for name in scenes:
        r = variant_scene((args.pack / "scenes" / name).resolve(), args.kit)
        print(f"{name}: {r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
