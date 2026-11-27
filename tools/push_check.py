#!/usr/bin/env python3
"""Check a scene's push physics offline (opticalnav_sim/hw/physics.py): how the world is built, how far its free bodies
drift at rest, and what a robot driving into an object does to it and to what is around it.

    python3.10 tools/push_check.py SCENE_DIR [--objects reachable|movable] [--push OBJECT_ID:SIDE[:METRES]] ...

SIDE is where the robot comes from (east, west, north or south of the object's box, in the simulator's world: x east,
y north); it starts 0.2 m off that face and drives through it at ``--speed`` until it is METRES (default 0.3) past it,
then stands 1.5 s. Every push starts from the scene as built. Reads physics.json and the scene's meshes, writes nothing.

Measured on blend_REF_CS02 (13 reachable objects and the carton riding on the cart; a Ranger Mini, 0.4 m):
    python3.10 tools/push_check.py packs/blend-cache/scenes/blend_REF_CS02 \\
        --push CS02_service_cart_shelf:east:0.68 --push CS02_service_cart_shelf:south:0.4 \\
        --push CS02_r003_shelf_transport_case_case.001:south:0.15 --push CS02_stool_chrome_stem:east:0.47
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # many small products (physics_tag.py)
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opticalnav_sim.hw import physics, robots  # noqa: E402

SIDES = {"east": (1, 180.0), "west": (-1, 0.0), "north": (2, -90.0), "south": (-2, 90.0)}   # (axis sign, robot yaw)


def turned(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))))


def settle(built, foot, height: float, seconds: float = 5.0, awake: bool = False) -> None:
    import mujoco
    world = physics.PushWorld(built.objects, built.statics, foot, height, built.floor_z, built.floor_mu, None, dict(built.info))
    if awake:   # every body simulated (no sleeping): the drift the contacts themselves give
        world.model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_SLEEP)
    start = world.world_poses()
    away = (-1e3, 0.0, built.floor_z or 0.0, 0.0)    # the robot far off: the drift alone
    world.step(away, 0.05)
    t = time.perf_counter()
    for _ in range(int(seconds / 0.05)):
        world.step(away, 0.05)
    ms = (time.perf_counter() - t) / (seconds / 0.05) * 1000.0
    now = world.world_poses()
    drift = sorted(((float(np.linalg.norm(now[i][:3, 3] - start[i][:3, 3])), turned(start[i], now[i]), i) for i in world.ids),
                   reverse=True)
    print(f"at rest {seconds:.0f} s{' (all awake)' if awake else ''}: {ms:.2f} ms a 50 ms step, "
          f"{world.awake()} of {len(world.ids)} awake; "
          f"drift max {drift[0][0] * 1000:.3f} mm, {max(d[1] for d in drift):.3f} deg")
    for d in drift[:3]:
        print(f"    {d[0] * 1000:8.3f} mm {d[1]:6.3f} deg  {d[2]}")
    if awake:
        return
    settled = world.info["settled_mm"]
    big = sorted(((v, k) for k, v in settled.items() if v > 3.0), reverse=True)
    print(f"settled at build (vs physics.json): median {np.median(list(settled.values())):.1f} mm; over 3 mm: "
          + (", ".join(f"{k} {v:.1f}" for v, k in big[:8]) or "none") + (f" (+{len(big) - 8})" if len(big) > 8 else ""))
    if world.info.get("held"):
        print(f"held asleep where the scene has them (they do not stand as their hulls): {', '.join(world.info['held'])}")


def push(built, foot, height: float, target: str, side: str, past: float, speed: float) -> None:
    world = physics.PushWorld(built.objects, built.statics, foot, height, built.floor_z, built.floor_mu, None, dict(built.info))
    if target not in world.ids:
        raise SystemExit(f"{target} is not a free body here ({len(world.ids)}: {', '.join(world.ids[:6])}, ...)")
    axis, yaw = SIDES[side]
    k = world.ids.index(target)
    p0 = world.world_poses()
    v = world.objects[k].vertices @ p0[target][:3, :3].T + p0[target][:3, 3]
    lo, hi = v.min(0), v.max(0)
    a = abs(axis) - 1
    face = hi[a] if axis > 0 else lo[a]
    half = float(np.abs(foot[:, 0]).max())
    centre = (lo + hi) / 2
    pos = centre.copy()
    pos[a] = face + math.copysign(half + 0.2, axis)
    step = np.zeros(3)
    step[a] = -math.copysign(speed * 0.05, axis)
    z = built.floor_z or 0.0
    ticks = int(round((0.2 + past) / (speed * 0.05)))
    seen, times, z_lo, z_hi, lag, stall = set(), [], {}, {}, 0.0, 0
    for i in range(ticks + int(1.5 / 0.05)):
        if i < ticks:
            pos += step
        t = time.perf_counter()
        world.step((pos[0], pos[1], z, yaw), 0.05)
        times.append(time.perf_counter() - t)
        seen |= set(world.contacts())
        lag, stall = max(lag, world.lag), stall + (world.stalled > 0)
        for i_, P in world.world_poses().items():
            z_lo[i_], z_hi[i_] = min(z_lo.get(i_, 1e9), P[2, 3]), max(z_hi.get(i_, -1e9), P[2, 3])
    p1 = world.world_poses()
    t = np.array(times) * 1000.0
    print(f"push {target} from the {side}, {past:.2f} m past its face at {speed} m/s: touched {sorted(seen) or 'nothing'}; "
          f"step {np.mean(t):.2f} ms mean, {t.max():.2f} max; collider lag max {lag * 100:.1f} cm, {stall} stalled steps")
    rows = sorted(((float(np.linalg.norm(p1[i][:3, 3] - p0[i][:3, 3])), i) for i in world.ids), reverse=True)
    for d, i in rows[:4]:
        if d < 1e-4:
            break
        dv = (p1[i][:3, 3] - p0[i][:3, 3]) * 100
        print(f"    {i[:46]:46s} {d * 100:6.1f} cm ({dv[0]:+.1f}, {dv[1]:+.1f}, {dv[2]:+.1f}), "
              f"turned {turned(p0[i], p1[i]):5.1f} deg, "
              f"z {(z_lo[i] - p0[i][2, 3]) * 100:+.1f}..{(z_hi[i] - p0[i][2, 3]) * 100:+.1f} cm on the way")
    rest = [d for d, i in rows if i not in seen and i != target]
    print(f"    not touched by the robot (what rides on the pushed body too): {len(rest)}, the most any moved "
          f"{max(rest or [0.0]) * 1000:.2f} mm")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--objects", default="reachable", choices=("reachable", "movable"))
    ap.add_argument("--push", action="append", default=[], help="OBJECT_ID:SIDE[:METRES] (repeatable)")
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--robot-height", type=float, default=0.4)
    args = ap.parse_args()
    foot = robots.RangerMini2().footprint()
    t = time.monotonic()
    built = physics.build(args.scene_dir, args.objects)
    world = physics.PushWorld(built.objects, built.statics, foot, args.robot_height, built.floor_z, built.floor_mu, None,
                              dict(built.info))
    info = {k: v for k, v in world.info.items() if k not in ("settled_mm",)}
    print(f"built in {time.monotonic() - t:.1f} s: " + json.dumps(info))
    settle(built, foot, args.robot_height)
    settle(built, foot, args.robot_height, awake=True)
    for spec in args.push:
        target, side, *rest = spec.split(":")
        push(built, foot, args.robot_height, target, side, float(rest[0]) if rest else 0.3, args.speed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
