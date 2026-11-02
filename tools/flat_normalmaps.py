#!/usr/bin/env python3
"""Drop the normal maps of pack scenes that carry no detail, only a constant tilt.

    python3 tools/flat_normalmaps.py --pack packs/opticalnav-sim-v1 [--scene SCENE ...] [--dry-run] [--max-std 0.02] [--min-tilt 3]

A tangent-space normal map holds (0.5, 0.5, 1.0) where the surface is flat. robomituba bakes a ``normal`` channel for
every textured part; for a material with no relief (sim_v2a_REF_PH19's corridor walls and floor tiles) the baked map is
a constant colour that is not (0.5, 0.5, 1.0) (0.57 0.43 0.99 for the wall: every normal tilted 12 degrees the same
way). Applied over a large flat surface that tilts the shading normal of each triangle against its own tangent frame:
shading jumps along the diagonals of the mesh's big triangles (pale and dark triangular sheets on the floor, a slanted
bright line on a wall that moves with the camera). Removing the ``normalmap`` wrapper of such a part gives a clean,
glossy floor and wall.

A normal map whose three channels each have a standard deviation under ``--max-std`` (default 0.02: no detail) and a mean
tilting the normal by more than ``--min-tilt`` degrees (default 3) is dropped from every shape that uses it; one that is
neutral (a harmless constant) or has detail is left alone. The XMLs are rewritten, the PNGs no XML names any more are
unlinked, and scene.json records the run under ``flat_normalmaps``. Rebuild the scene to undo it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compact_assets import json_refs, scene_lock, xml_refs  # noqa: E402

SHAPE = re.compile(r'(\n  <shape type="[a-z_]+" id="([^"]+)">)(.*?)(\n  </shape>)', re.S)
NORMALMAP = re.compile(r'<bsdf type="normalmap">\s*<texture name="normalmap" type="bitmap">\s*<string name="filename" '
                       r'value="([^"]+)" />.*?</texture>\s*(<bsdf type="[a-z]+">.*?</bsdf>)\s*</bsdf>', re.S)


def verdict(path: Path, max_std: float, min_tilt: float) -> tuple[bool, float, float]:
    """(drop it, tilt in degrees of the mean normal, largest channel std)."""
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB"), np.float32) / 255.0
    flat = a.reshape(-1, 3)
    std = float(flat.std(axis=0).max())
    n = flat.mean(axis=0) * 2 - 1
    tilt = float(np.degrees(np.arccos(np.clip(n[2] / max(np.linalg.norm(n), 1e-9), -1, 1))))
    return std <= max_std and tilt >= min_tilt, tilt, std


def flat_scene(scene_dir: Path, dry_run: bool = False, max_std: float = 0.02, min_tilt: float = 3.0) -> dict:
    with scene_lock(scene_dir):
        drop: dict[str, tuple[float, float]] = {}
        for f in sorted(scene_dir.glob("*.xml")):
            for ref in set(re.findall(r'<texture name="normalmap" type="bitmap">\s*<string name="filename" value="([^"]+)"', f.read_text())):
                if ref not in drop and (scene_dir / ref).is_file():
                    bad, tilt, std = verdict(scene_dir / ref, max_std, min_tilt)
                    if bad:
                        drop[ref] = (round(tilt, 1), round(std, 4))
        parts = 0
        if drop and not dry_run:
            for f in sorted(scene_dir.glob("*.xml")):
                text = f.read_text()

                def edit(m):
                    nonlocal parts
                    new = NORMALMAP.sub(lambda w: w.group(2) if w.group(1) in drop else w.group(0), m.group(3))
                    parts += new != m.group(3)
                    return m.group(1) + new + m.group(4)

                new_text = SHAPE.sub(edit, text)
                if new_text != text:
                    tmp = f.with_suffix(".xml.tmp")
                    tmp.write_text(new_text)
                    os.replace(tmp, f)
            kept = xml_refs(scene_dir) | json_refs(scene_dir)
            for ref in drop:
                if ref not in kept:
                    (scene_dir / ref).unlink(missing_ok=True)
            meta_path = scene_dir / "scene.json"
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text())
                meta["flat_normalmaps"] = {"maps": len(drop), "details": {k: v for k, v in drop.items()}}
                tmp = meta_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(meta, indent=1))
                os.replace(tmp, meta_path)
        return {"maps": len(drop), "details": drop, "dry_run": dry_run}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-std", type=float, default=0.02)
    ap.add_argument("--min-tilt", type=float, default=3.0)
    args = ap.parse_args()
    scenes = args.scene or sorted(p.parent.name for p in (args.pack / "scenes").glob("*/base.xml"))
    total = 0
    for name in scenes:
        s = flat_scene((args.pack / "scenes" / name).resolve(), args.dry_run, args.max_std, args.min_tilt)
        if s["maps"]:
            total += s["maps"]
            print(f"{name}: {s['maps']} constant normal maps (tilt degrees, std): {list(s['details'].values())[:4]}"
                  f"{' [dry run]' if args.dry_run else ' dropped'}", flush=True)
    print(f"{total} constant normal maps over {len(scenes)} scenes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
