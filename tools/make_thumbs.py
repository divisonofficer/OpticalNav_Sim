#!/usr/bin/env python3
"""Pictures for the viewer's scene gallery: each served scene's start view, rendered small by the render server.

    python3 tools/make_thumbs.py [--server URL] [--scene SCENE ...] [--redo] [--limit N] [--spp 64]

The viewer shows a scene's first finished frame once it has been opened, and the bird's-eye map of a scene converted from a .blend; this
fills the rest (the packs' scenes) in one go, 320 x 240 into ``~/.cache/opticalnav_sim/thumbs/<scene>.jpg``. Every scene is loaded by the
server (some seconds to a minute each: the server's other work waits), so run it when the GPU is free. Scenes that fail are skipped.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opticalnav_sim import MatterSim, frames  # noqa: E402
from opticalnav_sim.client import RenderClient  # noqa: E402
from opticalnav_sim.gui import THUMBS, thumb_file  # noqa: E402


def start_pose(client: RenderClient, scene: str, mount: dict):
    """The camera of the scene's first viewpoint (its connectivity graph) or its first spawn point (scene.json)."""
    meta = client.meta(scene)
    try:
        rows = client.connectivity(scene)
        row = next(r for r in rows if r["included"])
        return frames.camera_at_viewpoint(float(row["pose"][3]), float(row["pose"][7]), 0.0, 0.0, mount)
    except Exception:  # noqa: BLE001 - no graph: the spawn points
        s = (meta.get("spawn") or [None])[0]
        if s is None:
            raise
        return frames.camera_at_position(s["position"], math.radians(float(s.get("heading_deg", 0.0))), 0.0, mount)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=os.environ.get("OPTICALNAV_SIM_URL", "http://127.0.0.1:18770"))
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--redo", action="store_true", help="also the scenes that have a picture")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--spp", type=int, default=64)
    args = ap.parse_args()
    client = RenderClient(args.server)
    scenes = args.scene or client.scans()
    todo = [s for s in scenes if args.redo or not thumb_file(s).is_file()]
    todo = todo[:args.limit] if args.limit else todo
    print(f"{len(todo)} of {len(scenes)} scenes without a picture", flush=True)
    sim = MatterSim.Simulator()
    sim.setDatasetPath(args.server)
    sim.setVariant("base")
    sim.setRenderMode("polar")
    sim.setStokesDtype("float32")
    sim.setCameraResolution(320, 240)
    sim.setRenderSpp(args.spp)
    sim.initialize()
    from PIL import Image
    THUMBS.mkdir(parents=True, exist_ok=True)
    for n, scene in enumerate(todo, 1):
        t0 = time.time()
        try:
            mount = client.meta(scene)["camera"]["mount"]
            s0 = np.asarray(sim.renderCamera(scene, start_pose(client, scene, mount))["s0"], np.float32)
            lum = s0.mean(-1)
            scale = 0.8 / max(float(np.percentile(lum, 95)), 1e-4)  # the bright part at about 0.8, then gamma: a picture, not a measurement
            img = (np.clip(s0 * scale, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
            Image.fromarray(img).save(thumb_file(scene), quality=85)
            print(f"[{n}/{len(todo)}] {scene} {time.time() - t0:.0f} s", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[{n}/{len(todo)}] {scene} skipped: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
