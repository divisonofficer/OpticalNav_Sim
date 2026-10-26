#!/usr/bin/env python3
"""Make the see-through parts of pack scenes see-through again (sheer curtains).

    python3 tools/sheer_parts.py --pack packs/opticalnav-sim-v1 [--scene SCENE ...] [--dry-run]

The importer's materials carry an alpha (``params.pbr.alpha`` in the authoring map): ``ir_cur_sheer``, the net
between a window's curtains, is 0.12. The export drops it: the part comes out as opaque ``pplastic``, a white
panel across the window where Blender shows the street (sim_v2a_cafeS_lphigh).

For every material with an alpha below 0.99, the objects whose GLB has a primitive of that material (the GLB's
primitives are the object's parts: part 0 is the shape named for the object, part k the shape ``__part_<k>_`` or
``__<k>``) have that part's BSDF wrapped in ``mask`` with ``opacity`` set to the alpha: a ``mask`` is that
fraction of the surface and see-through the rest. The scene XMLs are rewritten, scene.json records the run under
``sheer_parts``. Rebuild the scene to undo it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compact_assets import scene_lock  # noqa: E402

ROBOMITUBA = Path("/jarvis/project/robomituba")
IMPORTS = ROBOMITUBA / "out/infinigen_imports"
SHAPE = re.compile(r'(\n  <shape type="[a-z_]+" id="([^"]+)">)(.*?)(\n  </shape>)', re.S)
MIN_ALPHA = 0.99


def primitive_materials(glb: Path) -> list[str]:
    """The material name of each primitive of a GLB's first mesh, in order."""
    data = glb.read_bytes()
    length = struct.unpack("<I", data[12:16])[0]
    doc = json.loads(data[20:20 + length])
    names = [m.get("name") for m in doc.get("materials", [])]
    out = []
    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            out.append(names[prim["material"]] if "material" in prim else None)
        break
    return out


def part_of(shape_id: str) -> tuple[str, int]:
    """(object id, primitive index) of a pack shape id."""
    m = re.search(r"__part_(\d+)_", shape_id)
    if m:
        return shape_id[:m.start()], int(m.group(1))
    m = re.search(r"__(\d+)$", shape_id)
    if m:
        return shape_id[:m.start()], int(m.group(1))
    return shape_id, 0


def sheer_targets(authoring_map: dict) -> dict[tuple[str, int], float]:
    """(object id, primitive index) -> alpha of the primitive's material, for alphas below MIN_ALPHA."""
    alpha = {x["material_id"]: float(x["params"]["pbr"]["alpha"]) for x in authoring_map.get("materials", [])
             if float(((x.get("params") or {}).get("pbr") or {}).get("alpha", 1.0)) < MIN_ALPHA}
    out = {}
    for obj in authoring_map.get("objects", []):
        ref = obj.get("source_ref") or (obj.get("metadata") or {}).get("glb_ref")
        if not ref or not alpha:
            continue
        path = ROBOMITUBA / ref
        if not path.is_file() or path.suffix != ".glb":
            continue
        for k, name in enumerate(primitive_materials(path)):
            if name in alpha:
                out[(obj["id"], k)] = alpha[name]
    return out


def sheer_scene(scene_dir: Path, dry_run: bool = False, imports: Path = IMPORTS) -> dict:
    amap = imports / f"{scene_dir.name}__authoring_map.json"
    if not amap.is_file():
        return {"sheer_parts": 0, "dry_run": dry_run}
    targets = sheer_targets(json.loads(amap.read_text()))
    if not targets:
        return {"sheer_parts": 0, "dry_run": dry_run}
    with scene_lock(scene_dir):
        counts, hit = [], []
        for f in sorted(scene_dir.glob("*.xml")):
            text = f.read_text()
            n = 0

            def edit(m):
                nonlocal n
                alpha = targets.get(part_of(m.group(2)))
                body = m.group(3)
                if alpha is None or 'type="mask"' in body:
                    return m.group(0)
                a, b = body.find("<bsdf"), body.rfind("</bsdf>")
                if a < 0 or b < 0:
                    return m.group(0)
                n += 1
                hit.append(m.group(2))
                wrapped = (f'<bsdf type="mask">\n      <float name="opacity" value="{alpha:.4f}" />\n      '
                           + body[a:b + 7] + "\n    </bsdf>")
                return m.group(1) + body[:a] + wrapped + body[b + 7:] + m.group(4)

            new = SHAPE.sub(edit, text)
            counts.append(n)
            if n and not dry_run:
                tmp = f.with_suffix(".xml.tmp")
                tmp.write_text(new)
                os.replace(tmp, f)
        summary = {"sheer_parts": counts[0] if counts else 0, "dry_run": dry_run}
        meta_path = scene_dir / "scene.json"
        if any(counts) and not dry_run and meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            meta["sheer_parts"] = {**summary, "parts": sorted(set(hit))[:200]}
            tmp = meta_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(meta, indent=1))
            os.replace(tmp, meta_path)
        return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--imports", type=Path, default=IMPORTS, help="robomituba's infinigen_imports directory")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    scenes = args.scene or sorted(p.parent.name for p in (args.pack / "scenes").glob("*/base.xml"))
    for name in scenes:
        s = sheer_scene((args.pack / "scenes" / name).resolve(), args.dry_run, args.imports)
        if s["sheer_parts"]:
            print(f"{name}: {s['sheer_parts']} see-through parts{' [dry run]' if args.dry_run else ' masked'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
