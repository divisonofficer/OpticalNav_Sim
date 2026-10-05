#!/usr/bin/env python3
"""Re-render the dataset reference frames stored in a pack and compare them.

    python tools/check_parity.py --pack packs/opticalnav-v0.2 --scene SCENE [--variant base] [--spp 256] [--server URL]

Each pack scene keeps a few dataset frames per variant under
``scenes/<scene>/reference/<variant>/<node>__<heading>/`` (manifest, Stokes NPZ,
preview). The camera comes from the MatterSim-style state (viewpoint + heading),
not from the stored matrix, so a pass also proves the frame conventions.
Reports render latency, camera error and S0 / S1 / S2 agreement. Below the
dataset's 1024 spp the residual is Monte Carlo noise and shrinks with spp.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opticalnav_sim import MatterSim, frames  # noqa: E402
from opticalnav_sim.stokes import LUMA  # noqa: E402


def block(a: np.ndarray, k: int = 8) -> np.ndarray:
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--variant", action="append", default=None, help="default: every variant with references")
    ap.add_argument("--limit", type=int, default=1, help="reference views per variant")
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--server", default=None)
    args = ap.parse_args()

    meta = json.loads((args.pack / "scenes" / args.scene / "scene.json").read_text())
    refs = [r for r in meta.get("reference_views", []) if not args.variant or r["variant"] in args.variant]
    sim = MatterSim.Simulator()
    if args.server:
        sim.setDatasetPath(args.server)
    sim.setNavGraphPath(str(args.pack / "connectivity"))
    sim.setRenderSpp(args.spp)
    sim.setStokesDtype("float32")
    sim.initialize()
    seen: dict[str, int] = {}
    for ref in refs:
        if seen.get(ref["variant"], 0) >= args.limit:
            continue
        seen[ref["variant"]] = seen.get(ref["variant"], 0) + 1
        folder = args.pack / "scenes" / args.scene / ref["path"]
        stored = np.load(folder / "stokes_data.npz")
        heading = frames.heading_from_yaw(math.radians(int(ref["heading_id"].split("_")[1])))
        sim.newEpisode([f"{args.scene}__{ref['variant']}"], [ref["node_id"]], [heading], [0.0])
        state = sim.getState()[0]
        manifest = json.loads((folder / "manifest.json").read_text())
        want = frames.legacy_flat_to_matrix(manifest["camera_specs"][0]["camera_to_world"])
        row = {"variant": ref["variant"], "view": f"{ref['node_id']}:{ref['heading_id']}", "spp": args.spp,
               "camera_max_abs_err": float(np.abs(state.camera_to_world - want).max()),
               "server_render_s": round(sim._client.last_render_seconds, 2)}
        for k in ("s0", "s1", "s2"):
            got, ref_k = state.stokes[k].astype(np.float32) @ LUMA, stored[k].astype(np.float32) @ LUMA
            row[f"{k}_mean_sim_vs_dataset"] = [round(float(got.mean()), 5), round(float(ref_k.mean()), 5)]
            row[f"{k}_corr_8x8"] = round(float(np.corrcoef(block(got).ravel(), block(ref_k).ravel())[0, 1]), 4)
        s0, r0 = state.stokes["s0"].astype(np.float32) @ LUMA, stored["s0"].astype(np.float32) @ LUMA
        row["s0_mean_abs_rel_err"] = round(float(np.abs(s0 - r0).mean() / r0.mean()), 4)
        print(json.dumps(row), flush=True)
    if not seen:
        print(f"no reference views for {args.scene} {args.variant or ''}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
