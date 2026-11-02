#!/usr/bin/env python3
"""Make the emissive materials of pack scenes glow, from a per-scene table of emission colour and strength.

    python3 tools/emissive_parts.py --pack packs/opticalnav-sim-v1 [--scene SCENE ...] [--imports DIR] [--dry-run]

A Blender Emission shader (an LED strip, a sconce, a downlight's lens) is not carried by the importer: the authoring
map holds ``emission_strength`` 0 for these materials and the stage writes them as plain plastic, so a corridor lit in
Blender by glowing strips and bulbs loses that light (sim_v2a_REF_PH33: its blue and white LED strips, 12 downlight
bulbs and a sconce; the lights that remain are 8 times dimmer than the Cycles render). The emission of such a material is
a radiance, colour x strength. ``tools/emission_overrides/<scene>.json`` gives it per material (read from the scene's
Blender build script); every shape whose authoring-map object has one of those materials gets an area emitter at that
radiance and an ``opticalnav-obj`` note (group fixture, like the lamps), so the viewer's light levels keep working.
The proper fix is in the importer (export the Emission shader's colour and strength); this is the stopgap. scene.json
records the run under ``emissive_parts``. Rebuild the scene to undo it.
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

IMPORTS = Path("/jarvis/project/robomituba/out/infinigen_imports")
TABLES = Path(__file__).resolve().parent / "emission_overrides"
SHAPE = re.compile(r'(\n  <shape type="[a-z_]+" id="([^"]+)">)(.*?)(\n  </shape>)', re.S)


def object_of(shape_id: str) -> str:
    return re.split(r"__part_\d+_|__\d+$", shape_id)[0]


def glow_scene(scene_dir: Path, imports: Path = IMPORTS, dry_run: bool = False) -> dict:
    table_path = TABLES / f"{scene_dir.name}.json"
    amap = imports / f"{scene_dir.name}__authoring_map.json"
    if not table_path.is_file() or not amap.is_file():
        return {"glowing": 0}
    materials = json.loads(table_path.read_text())["materials"]
    material_of = {o["id"]: o.get("material") for o in json.loads(amap.read_text()).get("objects", [])}
    with scene_lock(scene_dir):
        counts, hit = [], []
        for f in sorted(scene_dir.glob("*.xml")):
            text = f.read_text()
            n = 0

            def edit(m):
                nonlocal n
                obj = object_of(m.group(2))
                spec = materials.get(material_of.get(obj) or "")
                if spec is None or "<emitter" in m.group(3):
                    return m.group(0)
                rad = [c * float(spec["strength"]) for c in spec["color"]]
                n += 1
                hit.append(m.group(2))
                note = {"id": m.group(2), "type": "landmark", "is_emitter": True, "emitter_radiance": [round(v, 4) for v in rad],
                        "emitter_intensity": 1.0, "emitter_shape": "emissive_part", "label": f"light:fixture:{material_of[obj]}",
                        "metadata": {"added_by": "tools/emissive_parts.py"}}
                emitter = ('\n    <emitter type="area">\n      <rgb name="radiance" value="' + " ".join(f"{v:.4f}" for v in rad)
                           + '" />\n    </emitter>')
                return f"\n  <!--opticalnav-obj:{json.dumps(note)}-->" + m.group(1) + m.group(3) + emitter + m.group(4)

            new = SHAPE.sub(edit, text)
            counts.append(n)
            if n and not dry_run:
                tmp = f.with_suffix(".xml.tmp")
                tmp.write_text(new)
                os.replace(tmp, f)
        summary = {"glowing": counts[0] if counts else 0, "dry_run": dry_run}
        meta_path = scene_dir / "scene.json"
        if any(counts) and not dry_run and meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            meta["emissive_parts"] = {**summary, "parts": sorted(set(hit))}
            tmp = meta_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(meta, indent=1))
            os.replace(tmp, meta_path)
        return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--imports", type=Path, default=IMPORTS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    scenes = args.scene or sorted(p.stem for p in TABLES.glob("*.json"))
    for name in scenes:
        s = glow_scene((args.pack / "scenes" / name).resolve(), args.imports, args.dry_run)
        if s["glowing"]:
            print(f"{name}: {s['glowing']} parts glow{' [dry run]' if args.dry_run else ''}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
