#!/usr/bin/env python3
"""Separate fixed per-frame cost from GPU work on a resident scene.

    python tools/frame_overhead.py --pack packs/opticalnav-v0.2 --scene SCENE --server URL [--mode polar]

The scene is already resident on the server, so nothing here reloads it. Renders
(1) the same view several times, (2) changing views, and (3) a 64x48 frame at
1 spp. If (3) costs about as much as a full 512x384 frame, the time is per-call
overhead (kernel tracing), not pixels or samples on the GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opticalnav_sim import MatterSim, frames  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--mode", default="polar")
    ap.add_argument("--spp", type=int, default=128)
    args = ap.parse_args()
    meta = json.loads((args.pack / "scenes" / args.scene / "scene.json").read_text())
    refs = [r for r in meta["reference_views"] if r["variant"] == "base"]
    sim = MatterSim.Simulator()
    sim.setDatasetPath(args.server)
    sim.setNavGraphPath(str(args.pack / "connectivity"))
    sim.setRenderMode(args.mode)
    sim.initialize()

    def step(ref, width=512, height=384, spp=args.spp):
        sim.setCameraResolution(width, height)
        sim.setRenderSpp(spp)
        heading = frames.heading_from_yaw(math.radians(int(ref["heading_id"].split("_")[1])))
        t = time.perf_counter()
        sim.newEpisode([args.scene], [ref["node_id"]], [heading], [0.0])
        return round(time.perf_counter() - t, 3), sim._client.last_timing

    rows = []
    for label, ref, kw in [("same view, 1st", refs[0], {}), ("same view, 2nd", refs[0], {}), ("same view, 3rd", refs[0], {}),
                           ("new view", refs[1 % len(refs)], {}), ("new view", refs[2 % len(refs)], {}),
                           ("64x48 @1 spp", refs[0], {"width": 64, "height": 48, "spp": 1}),
                           ("64x48 @1 spp", refs[1 % len(refs)], {"width": 64, "height": 48, "spp": 1})]:
        seconds, timing = step(ref, **kw)
        rows.append({"case": label, "view": f"{ref['node_id']}:{ref['heading_id']}", "frame_s": seconds, **timing})
        print(json.dumps(rows[-1]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
