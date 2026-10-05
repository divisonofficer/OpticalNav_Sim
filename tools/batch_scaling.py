#!/usr/bin/env python3
"""How one render call's cost splits between the scene and the views.

    python tools/batch_scaling.py --pack packs/opticalnav-v0.2 --scene SCENE --server URL \
        --mode rgb --spp 64 --k 1,2,4,8,12

Sends K camera views (the 12 discretised headings at one viewpoint) in a single
request, which the server renders as one call. If tracing stays flat as K
grows, it is the scene's cost (materials, lights, geometry) and is shared by
the views; per-view cost is then trace/K + GPU per view. Each K runs twice:
the first call compiles the kernel for the new film width.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opticalnav_sim import frames  # noqa: E402
from opticalnav_sim.client import RenderClient  # noqa: E402
from opticalnav_sim.navgraph import NavGraph  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--variant", default="base")
    ap.add_argument("--mode", default="rgb")
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--k", default="1,2,4,8,12")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    meta = json.loads((args.pack / "scenes" / args.scene / "scene.json").read_text())
    node = next(r["node_id"] for r in meta["reference_views"] if r["variant"] == args.variant)
    graph = NavGraph.load(args.pack / "connectivity", args.scene)
    loc = graph.viewpoint(graph.index(node))
    cam = meta["camera"]
    client = RenderClient(args.server)
    rows = []
    for k in (int(v) for v in args.k.split(",")):
        views = [{"scan": args.scene, "variant": args.variant, "mode": args.mode, "spp": args.spp, "seed": 0,
                  "width": cam["width"], "height": cam["height"], "hfov_deg": cam["hfov_deg"],
                  "camera_to_world": frames.camera_at_viewpoint(loc.x, loc.y, i * math.pi / 6, 0.0, cam["mount"])}
                 for i in range(k)]
        for rep in range(args.repeat):
            t = time.perf_counter()
            try:
                client.render(views)
            except RuntimeError as exc:  # e.g. out of GPU memory at large K: record it and move on
                rows.append({"mode": args.mode, "spp": args.spp, "k": k, "error": str(exc)[:200]})
                print(json.dumps(rows[-1]), flush=True)
                break
            wall = time.perf_counter() - t
            tm = client.last_timing
            row = {"mode": args.mode, "spp": args.spp, "k": k, "rep": rep, "wall_s": round(wall, 2),
                   "per_view_s": round(wall / k, 2), "calls": tm.get("calls"), "kernels": tm.get("kernels"),
                   **{key: round(tm.get(key, 0.0), 3) for key in ("render_s", "trace_s", "codegen_s", "compile_s", "gpu_s", "post_s")}}
            rows.append(row)
            print(json.dumps(row), flush=True)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
