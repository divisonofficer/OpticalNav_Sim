"""R2R-style evaluation (port of Matterport3DSimulator tasks/R2R/eval.py).

    python -m opticalnav_sim.eval --pack <pack_dir> --split val_unseen --results results.json

``results.json`` uses the R2R submission format::

    [{"instr_id": "<path_id>_<k>", "trajectory": [[viewpoint_id, heading, elevation], ...]}, ...]

Metrics match R2R: navigation error and oracle error are shortest-path
distances on the connectivity graph, success means nav_error < error_margin,
SPL = success * shortest / max(shortest, trajectory_length).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .navgraph import NavGraph


class Evaluation:
    def __init__(self, pack: Path, splits: list[str], error_margin: float = 3.0):
        self.error_margin = error_margin
        self.gt, self.instr_ids = {}, set()
        for split in splits:
            for item in json.loads((pack / "tasks" / "R2R" / "data" / f"R2R_{split}.json").read_text()):
                self.gt[item["path_id"]] = item
                self.instr_ids |= {f"{item['path_id']}_{k}" for k in range(len(item["instructions"]))}
        self.graphs = {scan: NavGraph.load(pack / "connectivity", scan) for scan in {g["scan"] for g in self.gt.values()}}
        self._dist: dict[tuple[str, str], dict[str, float]] = {}

    def distance(self, scan: str, a: str, b: str) -> float:
        if (scan, b) not in self._dist:  # undirected graph: distances to b == distances from b
            self._dist[(scan, b)] = self.graphs[scan].shortest_paths(b)[0]
        return self._dist[(scan, b)].get(a, math.inf)

    def score_item(self, instr_id: str, path: list) -> dict:
        gt = self.gt[int(instr_id.split("_")[0])]
        scan, start, goal = gt["scan"], gt["path"][0], gt["path"][-1]
        if path[0][0] != start:
            raise ValueError(f"{instr_id}: trajectories must include the start position")
        graph = self.graphs[scan]
        length, prev = 0.0, path[0][0]
        for vp, *_ in path[1:]:
            if vp != prev:
                if graph.index(vp) not in graph.adjacent[graph.index(prev)]:
                    raise ValueError(f"{instr_id}: moves from {prev} to {vp} without a graph edge")
                length += self.distance(scan, prev, vp)
            prev = vp
        nav_error = self.distance(scan, path[-1][0], goal)
        oracle_error = min(self.distance(scan, vp, goal) for vp, *_ in path)
        shortest = self.distance(scan, start, goal)
        success = nav_error < self.error_margin
        return {"instr_id": instr_id, "nav_error": nav_error, "oracle_error": oracle_error,
                "trajectory_steps": len(path) - 1, "trajectory_length": length, "shortest_path_length": shortest,
                "success": float(success), "oracle_success": float(oracle_error < self.error_margin),
                "spl": float(success) * shortest / max(shortest, length) if shortest > 0 else float(success)}

    def score(self, results: list[dict]) -> tuple[dict, list[dict]]:
        rows = [self.score_item(r["instr_id"], r["trajectory"]) for r in results if r["instr_id"] in self.instr_ids]
        missing = self.instr_ids - {r["instr_id"] for r in rows}
        if missing:
            raise ValueError(f"trajectories not provided for {len(missing)} instruction ids, e.g. {sorted(missing)[:3]}")
        keys = ("nav_error", "oracle_error", "trajectory_steps", "trajectory_length", "success",
                "oracle_success", "spl")
        summary = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
        summary = {"length": len(rows), "error_margin": self.error_margin,
                   **{("success_rate" if k == "success" else "oracle_rate" if k == "oracle_success" else k): v
                      for k, v in summary.items()}}
        return summary, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--split", action="append", required=True)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--error-margin", type=float, default=3.0, help="success radius in metres (R2R: 3.0)")
    ap.add_argument("--out", type=Path, default=None, help="write per-episode rows + summary JSON here")
    args = ap.parse_args()
    summary, rows = Evaluation(args.pack, args.split, args.error_margin).score(json.loads(args.results.read_text()))
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "episodes": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
