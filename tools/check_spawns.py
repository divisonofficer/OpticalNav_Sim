#!/usr/bin/env python3
"""Check (and fix) the start poses of graph-less pack scenes against what the renderer sees.

    python3 tools/check_spawns.py --pack packs/opticalnav-sim-v1 --server http://127.0.0.1:18770 \
        --project <robomituba>/out/opticalnav/opticalnav-sim-v1 [--scene SCENE ...] [--apply]

tools/build_pack.py picks each start pose from the authoring map's footprints, which cover the whole floor
("clearance_m" is 0.0 in every scene), so it takes a grid corner: next to a wall, or in the cases of
sim_v2a_floorC (spawn 3) and sim_v2a_floorA2 (spawn 2), inside one, where every view is black.

Here each candidate point of a traversable region (build_pack.spawn_candidates) is rendered by the server's
raster preview in four headings with a depth pass, and scored by the least open of the four views: the median
depth to the nearest surface along the view axis, capped at 5 m, 0 if the view is black or the depth is mostly
missing. The best point of each region becomes its spawn. ``--apply`` writes them to scene.json (``spawn``,
each with ``open_m``, the score) and records the old ones under ``spawn_before``; without it only the table is
printed. A pose that renders black through the path tracer too is rejected, so run the server on the pack.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from opticalnav_sim import MatterSim, frames  # noqa: E402
import build_pack  # noqa: E402

HEADINGS = (0.0, 90.0, 180.0, 270.0)
OPEN_CAP_M, MIN_LUMA = 5.0, 0.004


def score_views(results: list[dict]) -> float:
    """Least open of the views of one pose, see the module docstring; 0 for a black or depthless view."""
    worst = OPEN_CAP_M
    for r in results:
        depth = np.asarray(r.get("depth", []), np.float32)
        ok = np.isfinite(depth) & (depth > 0)
        if depth.size == 0 or ok.mean() < 0.5:
            return 0.0
        worst = min(worst, float(np.median(depth[ok])))
    return worst


def evaluate(sim, scene: str, mount: dict, pose: dict, renderer: str) -> tuple[float, float]:
    """(openness score, least mean luminance of the four views) of a pose."""
    x, y, z = pose["position"]
    views = []
    for h in HEADINGS:
        c2w = frames.camera_at_position((x, y, z), math.radians(h), 0.0, mount)
        v = sim._view(scene, "base", c2w, 128, 96, 90.0)
        v.update(renderer=renderer, depth=True, spp=4 if renderer == "path" else v["spp"])
        views.append(v)
    out = sim._client.render(views, "float32")
    luma = min(float(np.asarray(r["s0"], np.float32).mean()) for r in out)
    return score_views(out), luma


def check_scene(sim, pack: Path, project: Path, scene: str, apply: bool) -> None:
    meta_path = pack / "scenes" / scene / "scene.json"
    meta = json.loads(meta_path.read_text())
    mount = meta["camera"]["mount"]
    eye = float(meta["spawn"][0]["position"][2])
    sim.setRenderer("raster")
    chosen, report = [], []
    for k, candidates in enumerate(build_pack.spawn_candidates(project / "scenes" / scene, eye)):
        scored = sorted(((evaluate(sim, scene, mount, p, "raster")[0], i) for i, p in enumerate(candidates)), reverse=True)
        sim.setRenderer("path")  # the pose must render in the path tracer too
        pick = None
        for s, i in scored[:6]:
            _, luma = evaluate(sim, scene, mount, candidates[i], "path")
            if s > 0 and luma > MIN_LUMA:
                pick = (s, candidates[i])
                break
        sim.setRenderer("raster")
        old = meta["spawn"][k] if k < len(meta["spawn"]) else None
        old_s = evaluate(sim, scene, mount, old, "raster")[0] if old else None
        report.append(f"  region {k}: old {old['position'][:2] if old else '-'} open {old_s if old_s is None else round(old_s, 2)} m"
                      f" -> new {pick[1]['position'][:2] if pick else '-'} open {round(pick[0], 2) if pick else '-'} m")
        if pick:
            chosen.append({**pick[1], "open_m": round(pick[0], 2)})
    print(f"{scene}\n" + "\n".join(report), flush=True)
    if apply and chosen:
        meta["spawn_before"] = meta.get("spawn_before") or meta["spawn"]
        meta["spawn"] = chosen
        tmp = meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=1))
        os.replace(tmp, meta_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--server", required=True)
    ap.add_argument("--project", required=True, type=Path, help="robomituba project holding scenes/<scene>/authoring_map.json")
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    sim = MatterSim.Simulator()
    sim.setDatasetPath(args.server)
    sim.setNavGraphPath(str(args.pack / "connectivity"))
    sim.setStokesDtype("float32")
    sim.initialize()
    scenes = args.scene or sorted(p.parent.name for p in (args.pack / "scenes").glob("*/scene.json"))
    for scene in scenes:
        check_scene(sim, args.pack.resolve(), args.project.resolve(), scene, args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
