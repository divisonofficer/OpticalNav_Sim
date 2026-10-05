#!/usr/bin/env python3
"""Generate OpticalNav episodes (v4 schema) on a pack scene's navigation support graph.

    python tools/generate_episodes.py --pack packs/opticalnav-v0.2 --scene SCENE \
        --count 50 --split train --seed 0 --min-m 3 --max-m 30 --out runs/gen
    python tools/replay_episode.py --pack packs/opticalnav-v0.2 --server URL --episodes runs/gen \
        --scene SCENE --variants base,perturbed --spp 64 --out runs/gen      # render them into the same bundle

Episodes go to <out>/episodes/<split>/<episode_id>.json. Every step is a transition
of the support graph (turn_left / turn_right 15 degrees, move_forward 0.25 m), and
the route is the fewest-action one between a random start and goal state (see
opticalnav_sim.generate for how that differs from robomituba's route policy).
Rendering is a separate step: replay_episode.py renders any episode source, these
included, and writes the export layout next to them.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opticalnav_sim.generate import SupportGraph, generate  # noqa: E402
from opticalnav_sim.sources import open_source  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--scene", required=True, action="append", help="repeatable")
    ap.add_argument("--graph-source", type=Path, default=None,
                    help="where to read the support graph if the pack lacks it (project or export bundle)")
    ap.add_argument("--count", type=int, default=10, help="episodes per scene")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-m", type=float, default=3.0, help="shortest lane distance from start to goal")
    ap.add_argument("--max-m", type=float, default=30.0)
    ap.add_argument("--goal-heading", choices=["random", "arrive"], default="random")
    ap.add_argument("--id-prefix", default="sim")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    for scene in args.scene:
        data = None
        for where in (args.pack, args.graph_source):
            data = open_source(where).graph(scene) if where else None
            if data:
                break
        if not data:
            raise SystemExit(f"{scene}: no navigation_support_graph.json in the pack (rebuild it) or --graph-source")
        graph = SupportGraph(json.loads(data))
        eps = generate(graph, args.count, args.split, args.seed, args.min_m, args.max_m, args.goal_heading, args.id_prefix)
        folder = args.out / "episodes" / args.split
        folder.mkdir(parents=True, exist_ok=True)
        for e in eps:
            (folder / f"{e['episode_id']}.json").write_text(json.dumps(e, ensure_ascii=False, indent=1))
        (args.out / "graph").mkdir(parents=True, exist_ok=True)
        (args.out / "graph" / f"{scene}__navigation_support_graph.json").write_bytes(data)
        steps = [len(e["actions"]) for e in eps]
        dist = [e["metadata"]["path_distance_m"] for e in eps]
        acts = Counter(a for e in eps for a in e["actions"])
        print(f"[generate] {scene}: {len(eps)}/{args.count} episodes  steps {min(steps, default=0)}-{max(steps, default=0)} "
              f"(mean {sum(steps) / max(len(steps), 1):.0f})  path {min(dist, default=0):.1f}-{max(dist, default=0):.1f} m  "
              f"actions {dict(acts)} -> {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
