"""Generate OpticalNav episodes on a scene's navigation support graph.

The graph (robomituba ``navigation_support_graph.json``, schema
``opticalnav.adaptive_navigation_support.v4``) is a state machine: a state is a
support pose and one of 24 headings (``support_pose_00001:h_045``), and its
transitions are ``turn_left`` / ``turn_right`` (15 degrees) and ``move_forward``
(0.25 m along a collision-checked lane). Generated episodes follow those
transitions only and use the v4 episode schema, so replay, the export readers and
training code take them like dataset episodes.

Route choice differs from robomituba's (blueprint routes, ``global_room_length_balance_v1``,
usually 5-60% longer than the shortest route): a generated episode is the fewest-action
route from the start state to the goal state, with start and goal drawn at random under
a seed and a path length range. ``metadata.episode_selection_policy`` says so.
"""
from __future__ import annotations

import heapq
import math
import random
from collections import defaultdict, deque

POLICY = "opticalnav_sim_fewest_actions_v1"
INSTRUCTION = "Navigate safely to the destination."


class SupportGraph:
    def __init__(self, raw: dict):
        if raw.get("profile_id") != "adaptive_navigation_support_v4":
            raise ValueError(f"unsupported support graph profile {raw.get('profile_id')!r}")
        self.raw = raw
        self.scene_id = raw["scene_id"]
        self.xy = {p["motion_pose_id"]: (float(p["position"][0]), float(p["position"][1])) for p in raw["poses"]}
        self.headings: dict[str, list[str]] = defaultdict(list)
        for s in raw["states"]:
            self.headings[s["motion_pose_id"]].append(s["heading_id"])
        for h in self.headings.values():
            h.sort()
        self.next: dict[str, list[tuple[str, str, str]]] = defaultdict(list)  # state -> (state, action, transition)
        for t in raw["transitions"]:
            self.next[t["source_state_id"]].append((t["target_state_id"], t["action"], t["transition_id"]))
        self.lanes: dict[str, list[tuple[str, float]]] = defaultdict(list)  # pose -> (pose, metres), forward lanes
        for e in raw["edges"]:
            if e.get("action") == "move_forward":
                self.lanes[e["source"]].append((e["target"], float(e["distance_m"])))

    def lane_distances(self, pose: str) -> dict[str, float]:
        """Metres along forward lanes from ``pose`` to every reachable pose (Dijkstra)."""
        dist, heap = {pose: 0.0}, [(0.0, pose)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, w in self.lanes[u]:
                if d + w < dist.get(v, math.inf):
                    dist[v] = d + w
                    heapq.heappush(heap, (d + w, v))
        return dist

    def route(self, start: str, goal: str) -> list[tuple[str, str, str]] | None:
        """Fewest-action route between two states: [(state, action, transition_id)] ending with (goal, 'stop', '')."""
        prev: dict[str, tuple[str, str, str] | None] = {start: None}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if u == goal:
                steps = [(u, "stop", "")]
                while prev[u] is not None:
                    p, action, tid = prev[u]
                    steps.append((p, action, tid))
                    u = p
                return steps[::-1]
            for v, action, tid in self.next[u]:
                if v not in prev:
                    prev[v] = (u, action, tid)
                    queue.append(v)
        return None


def _pose(graph: SupportGraph, state: str) -> list[float]:
    pose, heading = state.split(":")
    x, y = graph.xy[pose]
    return [x, y, math.radians(float(heading.split("_", 1)[1]))]


def episode(graph: SupportGraph, route: list[tuple[str, str, str]], episode_id: str, split: str, seed: int) -> dict:
    states = [s for s, _, _ in route]
    nodes = [s.split(":")[0] for s in states]
    path_distance_m = sum(math.dist(graph.xy[a], graph.xy[b]) for a, b in zip(nodes, nodes[1:]) if a != b)
    headings = [s.split(":")[1] for s in states]
    trajectory = [_pose(graph, s) for s in states]
    timesteps = []
    for i, (state, action, tid) in enumerate(route):
        extras = {"motion_profile": graph.raw["profile_id"], "motion_state_id": state, "heading_id": headings[i],
                  **({"transition_id": tid} if tid else {}), "available_heading_ids": graph.headings[nodes[i]]}
        timesteps.append({"timestep_index": i, "timestamp": float(i), "agent_pose": trajectory[i], "action": action,
                          "collision": False, "hazard_collision": False, "observation_bundle_ref": None,
                          "extras": extras})
    free = math.dist(trajectory[0][:2], trajectory[-1][:2])
    return {
        "episode_id": episode_id, "scene_id": graph.scene_id, "split": split,
        "start_pose": trajectory[0], "goal_pose": trajectory[-1], "goal_region": nodes[-1],
        "natural_language_instruction": INSTRUCTION,
        "trajectory": trajectory, "actions": [a for _, a, _ in route], "timesteps": timesteps,
        "metadata": {
            "motion_profile": graph.raw["profile_id"], "profile_digest": graph.raw.get("profile_digest"),
            "forward_step_m": graph.raw.get("forward_step_m"), "forward_tolerance_m": graph.raw.get("forward_tolerance_m"),
            "min_link_distance_m": graph.raw.get("min_link_distance_m"),
            "support_graph_ref": "navigation_support_graph.json",
            "episode_selection_policy": POLICY, "selection_seed": seed,
            "path_distance_m": round(path_distance_m, 4), "free_space_distance_m": round(free, 4),
            "detour_ratio": round(path_distance_m / free, 4) if free > 0 else None,
            "generator": "opticalnav_sim",
        },
        "schema_version": "0.1", "navigation_mode": graph.raw["profile_id"],
        "graph_id": graph.raw.get("place_graph_id"), "start_node": nodes[0], "goal_node": nodes[-1],
        "path_nodes": nodes, "path_headings": headings, "observation_refs": [], "instructions": [],
    }


def generate(graph: SupportGraph, count: int, split: str = "train", seed: int = 0, min_m: float = 3.0,
             max_m: float = 30.0, goal_heading: str = "random", id_prefix: str = "sim",
             max_tries: int | None = None) -> list[dict]:
    """``count`` episodes with lane distance in [min_m, max_m] and distinct (start pose, goal pose) pairs.
    ``goal_heading``: 'random' (turn to a drawn heading at the goal, like dataset episodes) or 'arrive'
    (stop facing the way the agent came in)."""
    rng = random.Random(seed)
    poses = sorted(p for p in graph.headings if graph.lanes[p])
    out, used = [], set()
    tries = 0
    max_tries = max_tries or 50 * count
    while len(out) < count and tries < max_tries:
        tries += 1
        start_pose = rng.choice(poses)
        reach = graph.lane_distances(start_pose)
        goals = [p for p, d in reach.items() if min_m <= d <= max_m and (start_pose, p) not in used]
        if not goals:
            continue
        goal_pose = rng.choice(goals)
        start = f"{start_pose}:{rng.choice(graph.headings[start_pose])}"
        if goal_heading == "random":
            route = graph.route(start, f"{goal_pose}:{rng.choice(graph.headings[goal_pose])}")
        else:  # arrive: the first state reached at the goal pose
            route = min((r for r in (graph.route(start, f"{goal_pose}:{h}") for h in graph.headings[goal_pose]) if r),
                        key=len, default=None)
        if route is None:
            continue
        used.add((start_pose, goal_pose))
        eid = f"{graph.scene_id}_{split}_{id_prefix}_{len(out) + 1:06d}"
        out.append(episode(graph, route, eid, split, seed))
    return out
