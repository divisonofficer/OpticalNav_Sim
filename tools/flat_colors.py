#!/usr/bin/env python3
"""Give the flat white parts of pack scenes their material's own colour.

    python3 tools/flat_colors.py --pack packs/opticalnav-sim-v1 [--scene SCENE ...] [--imports DIR] [--dry-run]

robomituba's stage writes a part whose material is waiting for a texture bake as a ``pplastic`` with
``diffuse_reflectance`` ``1 1 1`` (white): the bake multiplies it. A scene whose materials are plain colours (the
authored ``sim_v2a_REF_PH19`` corridor: walls, doors, floor, skirting, window frames ...) is never baked, so every one of
its parts came out white although its authoring map holds the colour (``render_binding.base_color_factor``, e.g. 0.55
0.52 0.47 for the vestibule wall, 0.03 0.03 0.05 for the doors).

For a shape whose id is an object of the authoring map (a single-material object; the parts of a multi-material object
cannot be told apart here) and whose ``pplastic`` has the white placeholder, the object's material colour replaces
it. Textured shapes, and materials whose colour is white or the importer's generic 0.6 grey fallback of an unbaked
procedural material, are left alone. scene.json records the run under ``flat_colors``. Rebuild the scene to undo it.
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
SHAPE = re.compile(r'(\n  <shape type="[a-z_]+" id="([^"]+)">)(.*?)(\n  </shape>)', re.S)
WHITE = re.compile(r'(<bsdf type="pplastic">\s*<rgb name="diffuse_reflectance" value=")1(?:\.0+)? 1(?:\.0+)? 1(?:\.0+)?(")')


def colours(authoring_map: dict) -> dict[str, list[float]]:
    """Object id -> the colour of its material, for the materials that carry a real one."""
    by_material = {}
    for m in authoring_map.get("materials", []):
        rb = m.get("render_binding") or {}
        factor = rb.get("base_color_factor")
        pbr = (m.get("params") or {}).get("pbr") or {}
        generic = factor is not None and all(abs(v - 0.6) < 1e-6 for v in factor) and pbr.get("needs_bake")
        if factor and not generic and not all(v >= 0.999 for v in factor) and rb.get("bsdf_strategy") == "pplastic":
            by_material[m["material_id"]] = [float(v) for v in factor[:3]]
    return {o["id"]: by_material[o["material"]] for o in authoring_map.get("objects", []) if o.get("material") in by_material}


def colour_scene(scene_dir: Path, imports: Path = IMPORTS, dry_run: bool = False) -> dict:
    amap = imports / f"{scene_dir.name}__authoring_map.json"
    if not amap.is_file():
        return {"coloured": 0, "skipped": "no authoring map"}
    by_object = colours(json.loads(amap.read_text()))
    with scene_lock(scene_dir):
        counts, hit = [], []
        for f in sorted(scene_dir.glob("*.xml")):
            text = f.read_text()
            n = 0

            def edit(m):
                nonlocal n
                col = by_object.get(m.group(2))
                if col is None or not WHITE.search(m.group(3)):
                    return m.group(0)
                n += 1
                hit.append(m.group(2))
                body = WHITE.sub(lambda w: w.group(1) + " ".join(f"{c:.4f}" for c in col) + w.group(2), m.group(3), count=1)
                return m.group(1) + body + m.group(4)

            new = SHAPE.sub(edit, text)
            counts.append(n)
            if n and not dry_run:
                tmp = f.with_suffix(".xml.tmp")
                tmp.write_text(new)
                os.replace(tmp, f)
        summary = {"coloured": counts[0] if counts else 0, "dry_run": dry_run}
        meta_path = scene_dir / "scene.json"
        if any(counts) and not dry_run and meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            meta["flat_colors"] = {**summary, "parts": sorted(set(hit))}
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
    scenes = args.scene or sorted(p.parent.name for p in (args.pack / "scenes").glob("*/base.xml"))
    for name in scenes:
        s = colour_scene((args.pack / "scenes" / name).resolve(), args.imports, args.dry_run)
        if s["coloured"]:
            print(f"{name}: {s['coloured']} white parts take their material colour{' [dry run]' if args.dry_run else ''}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
