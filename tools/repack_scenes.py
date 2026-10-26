#!/usr/bin/env python3
"""Rebuild pack scenes from robomituba's staged output and run the pack-side repairs on them, one scene after another.

    python3 tools/repack_scenes.py SCENE [SCENE ...] [--glass-slabs] [--open-doors] [--keep-spawns]

For each scene, with the render server's GPU left alone (``CUDA_VISIBLE_DEVICES=``; the CPU Mitsuba variant is not
needed either):

1. the scene's folder moves to ``packs/old_<scene>_<time>`` (a re-import replaces it; delete the old ones when sure),
2. ``build_pack.py`` rebuilds it from ``out/opticalnav/opticalnav-sim-v1`` (the staged scene: robomituba's
   ``stage_sim_scene.py`` is NOT run here, so a texture size chosen at staging, e.g. ``--texture-max 4096``, stays),
3. ``compact_assets.py`` and ``build_perturbation_pool.py``,
4. the repairs: ``metal_textures.py``, ``sheer_parts.py``, ``gray_textures.py`` (and, only with the flags,
   ``slab_glass.py`` and ``open_doors.py``: a scene staged from a glass-fixed blend already has its glass slabs and
   door openings), then ``light_audit.py``,
5. with ``--keep-spawns`` (default on) the old scene.json's start poses are copied into the new one: the geometry did
   not move, and the new ``build_pack`` picks grid corners. ``check_spawns.py`` with a running server is the better pick.

Prints one line per step; a failing step stops that scene and the batch goes on.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROJECT = Path("/jarvis/project/robomituba/out/opticalnav/opticalnav-sim-v1")
PACK = REPO / "packs/opticalnav-sim-v1"
MITSUBA = Path("/home/jinnyeong/robomituba-build/mitsuba3-3.9.1")


def run(label: str, cmd: list[str], env: dict) -> bool:
    t = time.time()
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    tail = " | ".join(l[:200] for l in (r.stdout + r.stderr).strip().splitlines()[-2:])
    print(f"    {label:14s} [{time.time() - t:4.0f}s exit {r.returncode}] {tail}", flush=True)
    return r.returncode == 0


def repack(scene: str, glass_slabs: bool, open_doors: bool, keep_spawns: bool, env: dict) -> bool:
    print(f"== {scene} {time.strftime('%H:%M:%S')}", flush=True)
    d = PACK / "scenes" / scene
    old_spawns = None
    if d.exists():
        meta = d / "scene.json"
        if keep_spawns and meta.is_file():
            old_spawns = json.loads(meta.read_text()).get("spawn")
        d.rename(PACK.parent / f"old_{scene}_{time.strftime('%m%d%H%M')}")
    py = "python3.10"
    nice = ["nice", "-n", "19"]
    pack = ["--pack", "packs/opticalnav-sim-v1", "--scene", scene]
    steps = [
        ("build_pack", nice + [py, "tools/build_pack.py", "--project", str(PROJECT), "--out", "packs/opticalnav-sim-v1",
                               "--scene", scene]),
        ("compact", nice + [py, "tools/compact_assets.py", *pack, "--jobs", "3"]),
        ("pool", nice + [py, "tools/build_perturbation_pool.py", *pack]),
    ]
    if glass_slabs:
        steps.append(("slab_glass", [py, "tools/slab_glass.py", *pack]))
    if open_doors:
        steps.append(("open_doors", [py, "tools/open_doors.py", *pack]))
    steps += [("metal_textures", [py, "tools/metal_textures.py", *pack]),
              ("sheer_parts", [py, "tools/sheer_parts.py", *pack]),
              ("gray_textures", [py, "tools/gray_textures.py", *pack]),
              ("light_audit", [py, "tools/light_audit.py", *pack])]
    for label, cmd in steps:
        if not run(label, cmd, env):
            print(f"  FAILED at {label}", flush=True)
            return False
    meta = d / "scene.json"
    if old_spawns and meta.is_file():
        data = json.loads(meta.read_text())
        data.setdefault("spawn_before", data.get("spawn"))
        data["spawn"] = old_spawns
        tmp = meta.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=1))
        os.replace(tmp, meta)
        print(f"    spawns kept   {len(old_spawns)} start poses of the old scene", flush=True)
    print(f"  READY {scene} {time.strftime('%H:%M:%S')}", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--glass-slabs", action="store_true", help="also run slab_glass.py")
    ap.add_argument("--open-doors", action="store_true", help="also run open_doors.py")
    ap.add_argument("--no-keep-spawns", action="store_true")
    args = ap.parse_args()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=f"{MITSUBA}/python:{REPO}",
               LD_LIBRARY_PATH=f"/usr/lib/wsl/lib:{MITSUBA}")
    failed = [s for s in args.scenes if not repack(s, args.glass_slabs, args.open_doors, not args.no_keep_spawns, env)]
    print(f"== done, {len(args.scenes) - len(failed)} of {len(args.scenes)} scenes" + (f"; failed: {failed}" if failed else ""),
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
