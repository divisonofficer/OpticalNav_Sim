#!/usr/bin/env python3
"""R2R-style evaluation loop against the OpticalNav simulator.

    python examples/run_agent.py --pack packs/opticalnav-v0.2 --split val_unseen \
        --server http://127.0.0.1:18770 --agent shortest --limit 5 --out results.json
    python -m opticalnav_sim.eval --pack packs/opticalnav-v0.2 --split val_unseen --results results.json

The loop is the one R2R agents run (Matterport3DSimulator tasks/R2R/env.py):
discretised viewing angles, one action per step chosen from
``state.navigableLocations`` or a heading/elevation change. Replace
``choose_action`` with a policy that reads ``state.rgb`` / ``state.stokes``.
Pass ``--no-render`` to check graph logic without the render server.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opticalnav_sim import MatterSim  # noqa: E402
from opticalnav_sim.navgraph import NavGraph  # noqa: E402


def choose_action(agent: str, state, goal_path: list[str], rng: random.Random) -> tuple[int, int, int] | None:
    """(index, heading, elevation) in MatterSim's discretised units; None = stop."""
    if agent == "random":
        if rng.random() < 0.05:
            return None
        if len(state.navigableLocations) > 1 and rng.random() < 0.6:
            return rng.randrange(1, len(state.navigableLocations)), 0, 0
        return 0, rng.choice((-1, 1)), 0
    # shortest-path teacher, as in R2R's ShortestAgent / teacher action
    here = state.location.viewpointId
    if here == goal_path[-1]:
        return None
    nxt = goal_path[goal_path.index(here) + 1] if here in goal_path else None
    for i, loc in enumerate(state.navigableLocations[1:], start=1):
        if loc.viewpointId == nxt:
            return i, 0, 0
    if state.elevation != 0.0:  # level the camera before turning, as the R2R teacher does
        return 0, 0, -1 if state.elevation > 0 else 1
    return 0, 1, 0  # target outside the field of view: turn right one 30 degree step


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--split", default="val_unseen")
    ap.add_argument("--server", default=None, help="render server URL (default $OPTICALNAV_SIM_URL)")
    ap.add_argument("--agent", choices=["shortest", "random"], default="shortest")
    ap.add_argument("--variant", default="base", choices=["base", "perturbed", "active_polar"])
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=1000, help="lattice episodes average ~230 steps")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results.json"))
    args = ap.parse_args()

    episodes = json.loads((args.pack / "tasks" / "R2R" / "data" / f"R2R_{args.split}.json").read_text())[: args.limit]
    sim = MatterSim.Simulator()
    if args.server:
        sim.setDatasetPath(args.server)
    sim.setNavGraphPath(str(args.pack / "connectivity"))
    sim.setDiscretizedViewingAngles(True)
    sim.setRenderingEnabled(not args.no_render)
    sim.setVariant(args.variant)
    sim.setRenderSpp(args.spp)
    sim.initialize()
    rng = random.Random(0)
    results = []
    started = time.time()
    for item in episodes:
        graph = NavGraph.load(args.pack / "connectivity", item["scan"])
        goal_path = graph.shortest_path(item["path"][0], item["path"][-1])
        sim.newEpisode([item["scan"]], [item["path"][0]], [item["heading"]], [0.0])
        state = sim.getState()[0]
        trajectory = [[state.location.viewpointId, state.heading, state.elevation]]
        for _ in range(args.max_steps):
            action = choose_action(args.agent, state, goal_path, rng)
            if action is None:
                break
            sim.makeAction([action[0]], [action[1]], [action[2]])
            state = sim.getState()[0]
            trajectory.append([state.location.viewpointId, state.heading, state.elevation])
            if args.agent == "shortest" and state.location.viewpointId not in goal_path:
                goal_path = graph.shortest_path(state.location.viewpointId, item["path"][-1])
        for k in range(len(item["instructions"])):
            results.append({"instr_id": f"{item['path_id']}_{k}", "trajectory": trajectory})
        print(f"path {item['path_id']} {item['scan']}: {len(trajectory) - 1} steps, "
              f"ends at {trajectory[-1][0]} (goal {item['path'][-1]})", flush=True)
    args.out.write_text(json.dumps(results))
    print(f"{len(results)} trajectories in {time.time() - started:.1f}s -> {args.out}")
    print(sim.timingInfo())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
