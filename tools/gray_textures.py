#!/usr/bin/env python3
"""Store the grey textures of built pack scenes as one channel instead of three.

    python3 tools/gray_textures.py --pack packs/opticalnav-sim-v1 [--scene SCENE ...] [--dry-run] [--keep-unused]

Roughness, metallic, ambient-occlusion and many albedo maps are grey: R = G = B at every pixel. The importer
saves them as RGB PNGs, and the GPU holds every texture as float32, so each costs three channels for one. A texture
whose three channels agree within one 8-bit level everywhere (an RGB PNG, no alpha) and that no normal map reads
is rewritten as an 8-bit single-channel PNG: Mitsuba reads it as a monochrome bitmap, which every plugin that takes
a texture (``diffuse_reflectance``, ``alpha``, ``weight``, ...) evaluates as the same value in all channels. The
picture is unchanged (at most one 8-bit level, the tolerance), and the texture takes a third of the memory:
floorO2 5.1 GB of textures on the GPU, 437 of 1,023 grey, 1.4 GB saved; the market 1.8 GB of 5.1.

New files are ``assets/tex_<sha1>.png``; the scene XMLs are rewritten, the old files no XML or JSON names any more
are unlinked (the pack's assets are hard links into robomituba's output: nothing is written in place), and
scene.json records the run under ``gray_textures``. Rebuild the scene to undo it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compact_assets import FILENAME_RE, json_refs, scene_lock, xml_refs  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
TOLERANCE = 1  # 8-bit levels the channels may differ by


def probe(job: tuple[Path, str]) -> tuple[str, str | None, int]:
    """(ref, new ref or None, bytes of float32 saved): the grey version of one texture, written, or None."""
    scene_dir, ref = job
    with Image.open(scene_dir / ref) as im:
        if im.mode != "RGB":  # RGBA, palette, 16-bit, already grey: left as they are
            return ref, None, 0
        a = np.asarray(im)
    r, g, b = (a[..., k].astype(np.int16) for k in range(3))
    if np.abs(r - g).max() > TOLERANCE or np.abs(g - b).max() > TOLERANCE:
        return ref, None, 0
    grey = np.rint(a.astype(np.float32).mean(axis=2)).astype(np.uint8)
    tmp = scene_dir / "assets" / f".gray_{os.getpid()}_{hashlib.sha1(ref.encode()).hexdigest()[:10]}.png"
    Image.fromarray(grey, mode="L").save(tmp)
    new = f"assets/tex_{hashlib.sha1(tmp.read_bytes()).hexdigest()[:20]}.png"
    if (scene_dir / new).exists():
        tmp.unlink()
    else:
        os.replace(tmp, scene_dir / new)
    return ref, new, int(a.shape[0] * a.shape[1] * 2 * 4)


def gray_scene(scene_dir: Path, dry_run: bool = False, keep_unused: bool = False, jobs: int = 8) -> dict:
    with scene_lock(scene_dir):
        files = sorted(scene_dir.glob("*.xml"))
        roles: dict[str, set[str]] = defaultdict(set)
        for f in files:
            for m in re.finditer(r'<texture name="([a-z_]+)" type="bitmap">\s*(?:<[^>]*>\s*)*?<string name="filename" '
                                 r'value="([^"]+\.png)"', f.read_text()):
                roles[m.group(2)].add(m.group(1))
        todo = sorted(r for r, names in roles.items() if "normalmap" not in names)
        if dry_run:  # the same test, nothing written
            def look(ref):
                with Image.open(scene_dir / ref) as im:
                    if im.mode != "RGB":
                        return 0
                    a = np.asarray(im).astype(np.int16)
                return int(a.shape[0] * a.shape[1] * 8) if (np.abs(a[..., 0] - a[..., 1]).max() <= TOLERANCE
                                                           and np.abs(a[..., 1] - a[..., 2]).max() <= TOLERANCE) else 0
            with ThreadPoolExecutor(jobs) as pool:
                saved = list(pool.map(look, todo))
            return {"textures": len(roles), "grey": sum(1 for s in saved if s), "gpu_bytes_saved": sum(saved),
                    "dry_run": True}
        with ThreadPoolExecutor(jobs) as pool:
            done = list(pool.map(probe, [(scene_dir, r) for r in todo]))
        renames = {old: new for old, new, _ in done if new}
        saved = sum(s for _, new, s in done if new)
        if renames:
            for f in files:
                text = f.read_text()
                new = FILENAME_RE.sub(lambda m: f"{m.group(1)}{renames.get(m.group(2), m.group(2))}{m.group(3)}", text)
                if new != text:
                    tmp = f.with_suffix(".xml.tmp")
                    tmp.write_text(new)
                    os.replace(tmp, f)
            if not keep_unused:
                kept = xml_refs(scene_dir) | json_refs(scene_dir)
                for old in renames:
                    if old not in kept:
                        (scene_dir / old).unlink(missing_ok=True)
        summary = {"textures": len(roles), "grey": len(renames), "gpu_bytes_saved": saved, "dry_run": False}
        meta_path = scene_dir / "scene.json"
        if renames and meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            meta["gray_textures"] = summary
            tmp = meta_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(meta, indent=1))
            os.replace(tmp, meta_path)
        return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-unused", action="store_true", help="keep the RGB files (to compare, or to undo)")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    scenes = args.scene or sorted(p.parent.name for p in (args.pack / "scenes").glob("*/base.xml"))
    total = 0
    for name in scenes:
        s = gray_scene((args.pack / "scenes" / name).resolve(), args.dry_run, args.keep_unused, args.jobs)
        total += s["gpu_bytes_saved"]
        print(f"{name}: {s['grey']} of {s['textures']} textures are grey, {s['gpu_bytes_saved'] / 1e9:.2f} GB less on the GPU"
              f"{' [dry run]' if args.dry_run else ''}", flush=True)
    print(f"{total / 1e9:.1f} GB saved over {len(scenes)} scenes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
