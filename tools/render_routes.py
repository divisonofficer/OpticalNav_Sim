#!/usr/bin/env python3
"""Render the camera poses of route files through a running render server, in the jn1027 render kit's file format.

    python3 tools/render_routes.py --server http://127.0.0.1:18770 --pack packs/opticalnav-sim-v1 \
        --routes <jn1027_render_kit>/routes --out <dir> [--variant glassmirror] [--spp 1024] \
        [--scene SCENE ...] [--limit N]

A route file is ``{route id: {"poses": [[x, y, yaw_rad], ...], ...}}`` (the dataset floor frame: x, y = Mitsuba X, Z).
For every pose, as the kit's own worker (kit/render_kit.py) does it: the scene camera with its rig mount, the
variant and spp given, seed ``crc32("<route id>/<k>") % 40000 * 100003``, polar mode, and

* ``<out>/images/<route id>/rgb/NNNN.jpg``: the server's RGB preview, JPEG quality 95,
* ``<out>/images/<route id>/pol/NNNN.png``: R = DoLP (0-1 -> 0-255), G = AoLP (0-180 degrees -> 0-255), B = s0 over its
  99th percentile (clipped), from the channel means of s0, s1, s2.

A frame is done when both files exist: a rerun skips them (and is safe to stop at any time). Unlike the kit's worker
this uses the server that is already running, with whatever options it was started with (``--default-sky``,
``--lamp-power``, ``--fill-gain``, ``--alpha-power``: the looks of this repository's renderer), after one 16 spp render
of each scene (a first render of a scene at 1024 spp ended a freshly started server once). ``--limit`` renders only
the first N missing frames per scene (a pilot). Prints the seconds per frame every 100 frames.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opticalnav_sim import frames  # noqa: E402
from opticalnav_sim.client import RenderClient  # noqa: E402


def done(rgb: Path) -> bool:
    pol = rgb.parent.parent / "pol" / (rgb.stem + ".png")
    return rgb.is_file() and rgb.stat().st_size > 0 and pol.is_file() and pol.stat().st_size > 0


def encode_pol(out: dict) -> np.ndarray:
    s0, s1, s2 = (np.asarray(out[c], np.float32) for c in ("s0", "s1", "s2"))
    s0, s1, s2 = (a.mean(axis=-1) if a.ndim == 3 else a for a in (s0, s1, s2))
    s0, s1, s2 = (np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0) for a in (s0, s1, s2))
    dolp = np.clip(np.sqrt(s1 ** 2 + s2 ** 2) / np.maximum(s0, 1e-6), 0, 1)
    aolp = (np.degrees(0.5 * np.arctan2(s2, s1)) % 180) / 180
    return np.stack([dolp * 255, aolp * 255, np.clip(s0 / max(np.percentile(s0, 99), 1e-6), 0, 1) * 255], -1).astype(np.uint8)


def render_scene(client: RenderClient, pack: Path, scene: str, routes: dict, out: Path, variant: str, spp: int,
                 limit: int | None) -> int:
    camera = json.loads((pack / "scenes" / scene / "scene.json").read_text())["camera"]
    todo = [(rid, k, p) for rid in sorted(routes) for k, p in enumerate(routes[rid]["poses"])
            if not done(out / "images" / rid / "rgb" / f"{k:04d}.jpg")]
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        print(f"{scene}: nothing left", flush=True)
        return 0

    def view(pose, seed, n):
        x, y, yaw = pose
        return {"scan": scene, "variant": variant, "camera_to_world": frames.rig_camera_to_world(x, y, yaw, camera["mount"]),
                "width": camera["width"], "height": camera["height"], "hfov_deg": camera["hfov_deg"], "spp": n, "seed": seed,
                "mode": "polar"}

    client.render([view(todo[0][2], 0, 16)])  # the scene loads and records at a small spp first
    t0, n = time.time(), 0
    for rid, k, pose in todo:
        rgb = out / "images" / rid / "rgb" / f"{k:04d}.jpg"
        pol = out / "images" / rid / "pol" / f"{k:04d}.png"
        result = client.render([view(pose, zlib.crc32(f"{rid}/{k}".encode()) % 40000 * 100003, spp)])[0]
        pol.parent.mkdir(parents=True, exist_ok=True)
        rgb.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(encode_pol(result)).save(str(pol) + ".part", "PNG")
        os.replace(str(pol) + ".part", pol)
        Image.fromarray(result["rgb"]).save(str(rgb) + ".part", "JPEG", quality=95)
        os.replace(str(rgb) + ".part", rgb)  # the rgb file lands last: a frame is done when it exists
        n += 1
        if n % 100 == 0:
            print(f"{scene}: {n}/{len(todo)} frames, {(time.time() - t0) / n:.2f} s/frame", flush=True)
    print(f"{scene}: {n} frames in {time.time() - t0:.0f} s ({(time.time() - t0) / max(n, 1):.2f} s/frame)", flush=True)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:18770")
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--routes", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--variant", default="glassmirror")
    ap.add_argument("--spp", type=int, default=1024)
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--limit", type=int, default=None, help="only the first N missing frames of each scene")
    args = ap.parse_args()
    client = RenderClient(args.server, timeout=3600.0)
    scenes = args.scene or sorted(p.stem for p in args.routes.glob("*.json"))
    total = 0
    for scene in scenes:
        routes = json.loads((args.routes / f"{scene}.json").read_text())
        total += render_scene(client, args.pack, scene, routes, args.out, args.variant, args.spp, args.limit)
    print(f"done: {total} frames", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
