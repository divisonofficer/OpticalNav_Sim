"""OpticalNav episode files (robomituba ``episodes/<split>/<episode_id>.json``).

An episode's observation at step i is the dataset camera at
``(path_nodes[i], path_headings[i])``: the support pose the agent stands on and
its discrete heading, the same key the export's ``index.jsonl`` joins on
(``vp_id``, ``heading_id``). ``adaptive_navigation_support_v4`` episodes list one
node and heading per step; older ``viewpoint_graph`` episodes keep them in
``timesteps[i].extras``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from . import frames


@dataclass
class Step:
    index: int
    node_id: str
    heading_id: str
    action: str

    @property
    def yaw_deg(self) -> float:
        return float(self.heading_id.split("_", 1)[1])

    @property
    def frame_key(self) -> tuple[str, str]:
        return self.node_id, self.heading_id


@dataclass
class Episode:
    episode_id: str
    scene_id: str
    split: str
    steps: list[Step]
    raw: dict = field(repr=False)
    path: Path | None = None

    def frame_id(self, step: Step) -> str:
        return frame_id(self.scene_id, step.node_id, step.heading_id)

    def unique_frames(self) -> list[tuple[str, str]]:
        """(node_id, heading_id) in first-visit order; steps that do not move or turn share a frame."""
        seen: dict[tuple[str, str], None] = {}
        for s in self.steps:
            seen.setdefault(s.frame_key, None)
        return list(seen)


def frame_id(scene_id: str, node_id: str, heading_id: str) -> str:
    return f"{scene_id}_{node_id}_{heading_id}"


def heading_id(yaw_deg: float) -> str:
    return f"h_{int(round(yaw_deg)) % 360:03d}"


def parse(raw: Mapping, path: Path | None = None) -> Episode:
    actions = list(raw.get("actions") or [t.get("action") for t in raw.get("timesteps", [])])
    nodes, headings = list(raw.get("path_nodes") or []), list(raw.get("path_headings") or [])
    if not (nodes and len(nodes) == len(headings) == len(actions)):
        # viewpoint_graph episodes: path_nodes lists distinct viewpoints; the per-step state is in timesteps
        ts = raw.get("timesteps") or []
        nodes = [(t.get("extras") or {}).get("node_id") for t in ts]
        headings = [(t.get("extras") or {}).get("heading_id") for t in ts]
        actions = [t.get("action") for t in ts]
        if not ts or None in nodes or None in headings:
            raise ValueError(f"{path or raw.get('episode_id')}: no per-step node and heading")
    steps = [Step(i, n, h, a) for i, (n, h, a) in enumerate(zip(nodes, headings, actions))]
    return Episode(raw["episode_id"], raw["scene_id"], raw.get("split", ""), steps, dict(raw), path)


def load(path: str | Path) -> Episode:
    path = Path(path)
    return parse(json.loads(path.read_text()), path)


def camera_to_world(node_xy_r2r: tuple[float, float], yaw_deg: float, mount: Mapping | None) -> np.ndarray:
    """The dataset rig camera at a support pose and heading (index.jsonl camera_to_world, as 4x4)."""
    return frames.camera_at_viewpoint(node_xy_r2r[0], node_xy_r2r[1],
                                      frames.heading_from_yaw(math.radians(yaw_deg)), 0.0, mount)


def base_pose(node_xy_r2r: tuple[float, float], yaw_deg: float, mount: Mapping | None) -> np.ndarray:
    """The robot base pose the dataset manifests store: the rig at the camera's height, no mount offset."""
    mount = dict(mount or {})
    height = float((mount.get("xyz_m") or [0.0, 1.0, 0.0])[1])
    x, y = frames.dataset_xy(*node_xy_r2r)
    return frames.rig_camera_to_world(x, y, math.radians(yaw_deg), {"xyz_m": [0.0, height, 0.0],
                                                                   "rpy_deg": mount.get("rpy_deg")})


def flat(matrix: np.ndarray) -> list[float]:
    """4x4 -> the export's flat camera_to_world (inverse of frames.legacy_flat_to_matrix)."""
    return [float(v) for v in np.asarray(matrix, dtype=np.float64).T.reshape(-1)]
