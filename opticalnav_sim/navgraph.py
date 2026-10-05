"""Matterport3D connectivity graphs and MatterSim's navigable-location rule.

The loader and ``navigable`` mirror Matterport3DSimulator (src/lib/NavGraph.cpp,
Simulator::populateNavigable) so candidate lists, their order and the
rel_heading / rel_elevation values match what R2R agents expect.
"""
from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ViewPoint:
    viewpointId: str
    ix: int
    x: float
    y: float
    z: float
    rel_heading: float = 0.0
    rel_elevation: float = 0.0
    rel_distance: float = 0.0


@dataclass
class NavGraph:
    scan: str
    ids: list[str]
    pos: list[tuple[float, float, float]]
    included: list[bool]
    adjacent: list[list[int]] = field(default_factory=list)

    @classmethod
    def from_connectivity(cls, scan: str, rows: list[dict]) -> "NavGraph":
        ids = [r["image_id"] for r in rows]
        pos = [(float(r["pose"][3]), float(r["pose"][7]), float(r["pose"][11])) for r in rows]
        inc = [bool(r["included"]) for r in rows]
        adj = [[j for j, ok in enumerate(r["unobstructed"]) if ok and j != i and inc[j]] for i, r in enumerate(rows)]
        return cls(scan, ids, pos, inc, adj)

    @classmethod
    def load(cls, connectivity_dir: str | Path, scan: str) -> "NavGraph":
        path = Path(connectivity_dir) / f"{scan}_connectivity.json"
        return cls.from_connectivity(scan, json.loads(path.read_text()))

    def index(self, viewpoint_id: str) -> int:
        try:
            return self._index[viewpoint_id]
        except AttributeError:
            self._index = {v: i for i, v in enumerate(self.ids)}
            return self._index[viewpoint_id]

    def viewpoint(self, ix: int) -> ViewPoint:
        x, y, z = self.pos[ix]
        return ViewPoint(self.ids[ix], ix, x, y, z)

    def navigable(self, ix: int, heading: float, elevation: float, hfov: float,
                  restricted: bool = True) -> list[ViewPoint]:
        """Index 0 is the current viewpoint; the rest sorted by angle from image centre."""
        ah = math.pi / 2.0 - heading
        cx, cy = math.cos(ah), math.sin(ah)
        cos_half = math.cos(hfov / 2.0)
        px, py, pz = self.pos[ix]
        out = []
        for i in self.adjacent[ix]:
            tx, ty, tz = self.pos[i][0] - px, self.pos[i][1] - py, self.pos[i][2] - pz
            dist = math.sqrt(tx * tx + ty * ty + tz * tz)
            horiz = math.hypot(tx, ty)
            rel_el = math.atan2(tz, horiz) - elevation
            cos_angle = (tx * cx + ty * cy) / horiz if horiz > 0 else 1.0
            if restricted and cos_angle < cos_half:
                continue
            rel_h = math.atan2(tx * cy - ty * cx, tx * cx + ty * cy)
            out.append(ViewPoint(self.ids[i], i, *self.pos[i], rel_h, rel_el, dist))
        out.sort(key=lambda v: math.hypot(v.rel_heading, v.rel_elevation))
        return [self.viewpoint(ix)] + out

    def distance(self, a: int, b: int) -> float:
        return math.dist(self.pos[a], self.pos[b])

    def shortest_paths(self, source: str) -> tuple[dict[str, float], dict[str, str]]:
        """Dijkstra from ``source`` with Euclidean edge weights (R2R's load_nav_graphs)."""
        src = self.index(source)
        dist, parent, heap = {src: 0.0}, {}, [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v in self.adjacent[u]:
                nd = d + self.distance(u, v)
                if nd < dist.get(v, math.inf):
                    dist[v], parent[v] = nd, u
                    heapq.heappush(heap, (nd, v))
        return ({self.ids[k]: v for k, v in dist.items()},
                {self.ids[k]: self.ids[v] for k, v in parent.items()})

    def shortest_path(self, source: str, goal: str) -> list[str]:
        _, parent = self.shortest_paths(source)
        if goal != source and goal not in parent:
            return []
        path = [goal]
        while path[-1] != source:
            path.append(parent[path[-1]])
        return path[::-1]
