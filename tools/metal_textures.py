#!/usr/bin/env python3
"""Give the metal parts of pack scenes the object's own textures.

    python3 tools/metal_textures.py --pack packs/opticalnav-sim-v1 [--scene SCENE ...] [--dry-run]

robomituba bakes one albedo and one roughness texture per object (an atlas: every part of the mesh has UVs
into it) and writes them on the object's plastic parts (``pplastic``: ``diffuse_reflectance`` and ``alpha``).
The metal parts of the same object (``roughconductor`` with a named material and a flat alpha) get no texture:
a counter's steel appliances came out as bright rough aluminium, white and speckled under the sky, where Blender
shows grey brushed steel (sim_v2a_cafeS_lphigh's island counter).

A ``roughconductor`` part with no texture, whose object has plastic parts with textures, is rewritten as a
conductor that reflects its atlas colour: ``material`` ``none`` (a perfect reflector) with the atlas as
``specular_reflectance`` and the roughness texture as ``alpha``. Objects without textured plastic parts, and
mirrors (they are not parts of an object), are left as they are. A part is an object's when its id, cut at the
first ``__part_<n>_`` or ``__<n>``, is the plastic parts' id; scene.json records the run under ``metal_textures``.
Rebuild the scene to undo it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compact_assets import scene_lock  # noqa: E402

SHAPE = re.compile(r'(\n  <shape type="[a-z_]+" id="([^"]+)">)(.*?)(\n  </shape>)', re.S)
OBJECT = re.compile(r"__(?:part_\d+_|\d+$)")
TEXTURE = re.compile(r'<texture name="(diffuse_reflectance|alpha)" type="bitmap">\s*<string name="filename" '
                     r'value="([^"]+\.png)" />')
METAL = re.compile(r'<bsdf type="roughconductor">\s*<string name="material" value="[A-Za-z]+" />\s*'
                   r'<float name="alpha" value="[\d.]+" />\s*</bsdf>')


def object_of(shape_id: str) -> str:
    return OBJECT.split(shape_id)[0]


def textured(text: str) -> dict[str, tuple[str, str]]:
    """Object id -> (albedo, roughness) texture files, from the plastic parts that carry both."""
    found: dict[str, Counter] = defaultdict(Counter)
    for _, sid, body, _ in SHAPE.findall(text):
        if 'type="pplastic"' not in body:
            continue
        tex = dict(TEXTURE.findall(body))
        if "diffuse_reflectance" in tex and "alpha" in tex:
            found[object_of(sid)][(tex["diffuse_reflectance"], tex["alpha"])] += 1
    return {obj: c.most_common(1)[0][0] for obj, c in found.items()}


def metal_scene(scene_dir: Path, dry_run: bool = False) -> dict:
    with scene_lock(scene_dir):
        files = sorted(scene_dir.glob("*.xml"))
        counts: list[int] = []
        parts: list[str] = []
        for f in files:
            changed = 0
            text = f.read_text()
            by_object = textured(text)

            def edit(m):
                nonlocal changed
                sid, body = m.group(2), m.group(3)
                tex = by_object.get(object_of(sid)) if METAL.search(body) else None
                if tex is None:
                    return m.group(0)
                changed += 1
                parts.append(sid)
                new = ('<bsdf type="roughconductor">\n        <string name="material" value="none" />\n'
                       '        <texture name="specular_reflectance" type="bitmap">\n'
                       f'          <string name="filename" value="{tex[0]}" />\n        </texture>\n'
                       '        <texture name="alpha" type="bitmap">\n'
                       f'          <string name="filename" value="{tex[1]}" />\n'
                       '          <boolean name="raw" value="true" />\n        </texture>\n      </bsdf>')
                return m.group(1) + METAL.sub(lambda _: new, body, count=1) + m.group(4)

            new_text = SHAPE.sub(edit, text)
            counts.append(changed)
            if new_text != text and not dry_run:
                tmp = f.with_suffix(".xml.tmp")
                tmp.write_text(new_text)
                os.replace(tmp, f)
        summary = {"metal_parts": counts[0] if counts else 0, "dry_run": dry_run}
        meta_path = scene_dir / "scene.json"
        if any(counts) and not dry_run and meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            meta["metal_textures"] = {**summary, "parts": sorted(set(parts))[:200]}
            tmp = meta_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(meta, indent=1))
            os.replace(tmp, meta_path)
        return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    scenes = args.scene or sorted(p.parent.name for p in (args.pack / "scenes").glob("*/base.xml"))
    for name in scenes:
        s = metal_scene((args.pack / "scenes" / name).resolve(), args.dry_run)
        if s["metal_parts"]:
            print(f"{name}: {s['metal_parts']} metal parts take their object's textures"
                  f"{' [dry run]' if args.dry_run else ''}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
