#!/usr/bin/env python3
"""A scene's ground map (heights, obstacles, surface materials and their friction) from its own files.

    python3 tools/build_ground_map.py --scene packs/blend-cache/scenes/blend_REF_PH11 [--floor Z] [--top 1.4] [--cell 0.05]

Writes ``<scene>/map/ground.npz`` (opticalnav_sim.groundmap.GroundMap) and ``<scene>/map/ground_friction.png`` (the
friction coefficient seen from above: red slippery, green grippy, black blocked, grey unseen), and prints each
material's coefficient with the reason. ``--floor``: the floor's height in the scene's world (default: 1.5 m under the
first start pose's camera, then the height found there). No renderer, no GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opticalnav_sim import groundmap  # noqa: E402


def picture(gm: groundmap.GroundMap, step: float = 0.03) -> np.ndarray:
    """RGB image, north up: friction from red (0.1) to green (0.9) where a robot can stand; black: something higher
    than a step above the surface to stand on; grey: nothing seen."""
    mu = gm.mu()
    t = np.clip((mu - 0.1) / 0.8, 0.0, 1.0)
    img = np.stack([1.0 - t, t, 0.15 + 0.0 * t], -1)
    img[np.isnan(mu)] = 0.5
    ground = gm.ground(step)
    stand = np.where(gm.material >= 0, gm.heights, np.nan)
    floor = np.nanmedian(stand)
    # a rough picture of what blocks: cells well above the floor around them (the Ground's own test is local)
    from_floor = gm.heights - floor
    img[(from_floor > 0.5) & ~np.isnan(gm.heights)] = 0.0
    del ground
    return (img[::-1] * 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--floor", type=float, default=None)
    ap.add_argument("--top", type=float, default=1.4, help="the robot's height with what it carries (m)")
    ap.add_argument("--cell", type=float, default=0.05)
    ap.add_argument("--bounds", type=float, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"))
    args = ap.parse_args()
    floor = args.floor
    if floor is None:
        spawn = (json.loads((args.scene / "scene.json").read_text()).get("spawn") or [{"position": [0.0, 0.0, 1.5]}])[0]["position"]
        guess = groundmap.build(args.scene, spawn[2] - 1.5, top=0.5, cell=0.25, below=1.0,
                                bounds=(spawn[0] - 2, spawn[1] - 2, spawn[0] + 2, spawn[1] + 2))
        stand = np.where(guess.material >= 0, guess.heights, np.nan)
        floor = float(np.nanmedian(stand))
        print(f"floor at {floor:.3f} m (under the first start pose, {spawn[2]:.2f} m)")
    bounds = args.bounds
    if bounds is None:  # the area the scene's own map shows, else 25 m around its start poses (not its far backdrop)
        meta = json.loads((args.scene / "scene.json").read_text())
        if (meta.get("map") or {}).get("bounds"):
            bounds = tuple(meta["map"]["bounds"])
        elif meta.get("spawn"):
            p = np.array([s["position"][:2] for s in meta["spawn"]])
            bounds = (p[:, 0].min() - 25, p[:, 1].min() - 25, p[:, 0].max() + 25, p[:, 1].max() + 25)
    t0 = time.time()
    gm = groundmap.build(args.scene, floor, args.top, args.cell, bounds)
    (args.scene / "map").mkdir(exist_ok=True)
    gm.save(args.scene / "map" / "ground.npz")
    from PIL import Image
    Image.fromarray(picture(gm)).save(args.scene / "map" / "ground_friction.png")
    stand = gm.material >= 0
    idx, counts = np.unique(gm.material[stand], return_counts=True)
    print(f"{gm.heights.shape[1]} x {gm.heights.shape[0]} cells of {gm.cell} m, {gm.meta['samples']} samples of {gm.meta['shapes']} shapes, "
          f"{time.time() - t0:.1f} s -> {args.scene / 'map' / 'ground.npz'}")
    for k, c in sorted(zip(idx, counts), key=lambda kc: -kc[1])[:25]:
        f = gm.friction[gm.materials[k]]
        print(f"  {gm.materials[k][:44]:44s} {c * gm.cell ** 2:8.1f} m2  mu {f['mu']:.3f}  {f['why']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
