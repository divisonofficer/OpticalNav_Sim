#!/usr/bin/env python3
"""Frame rate and quality of each render mode, through the MatterSim API.

    python tools/benchmark_modes.py --pack packs/opticalnav-v0.2 --scene SCENE \
        [--modes polar,rgb] [--spp 128,256,512,1024,2048] [--denoise off,on] [--frames 3] [--budget-s 300]

Every frame renders one of the scene's dataset reference views (a different
viewpoint and heading each time, as during navigation) and is compared with
the stored 1024-spp dataset frame: ``s0_err`` is mean |S0 - S0_dataset| / mean
S0_dataset (radiance vs S0 in rgb mode). The first frame of each setting is a
warm-up (tracing, dr.freeze recording) and is reported separately.

``--budget-s`` skips the rest of a mode once one frame takes longer than this,
so a slow build still finishes; skipped settings are listed, not estimated.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opticalnav_sim import MatterSim, frames  # noqa: E402
from opticalnav_sim.stokes import LUMA  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--variant", default="base")
    ap.add_argument("--server", default=None)
    ap.add_argument("--modes", default="polar,rgb")
    ap.add_argument("--spp", default="128,256,512,1024,2048")
    ap.add_argument("--denoise", default="off,on")
    ap.add_argument("--frames", type=int, default=3, help="timed frames per setting (after one warm-up)")
    ap.add_argument("--budget-s", type=float, default=300.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    meta = json.loads((args.pack / "scenes" / args.scene / "scene.json").read_text())
    refs = [r for r in meta.get("reference_views", []) if r["variant"] == args.variant]
    if not refs:
        raise SystemExit(f"{args.scene} has no {args.variant} reference views in the pack")
    stored = [np.load(args.pack / "scenes" / args.scene / r["path"] / "stokes_data.npz") for r in refs]
    sim = MatterSim.Simulator()
    if args.server:
        sim.setDatasetPath(args.server)
    sim.setNavGraphPath(str(args.pack / "connectivity"))
    sim.setVariant(args.variant)
    sim.initialize()
    info = sim._client.info()
    print(f"server: modes={info['modes']} freeze={info['drjit_freeze']} spp_chunk={info.get('spp_chunk')}", flush=True)

    rows, skipped = [], []
    for mode in args.modes.split(","):
        over_budget = False
        for spp in (int(s) for s in args.spp.split(",")):
            for denoise in (d == "on" for d in args.denoise.split(",")):
                setting = {"mode": mode, "spp": spp, "denoise": denoise}
                if over_budget:
                    skipped.append(setting)
                    continue
                sim.setRenderMode(mode)
                sim.setRenderSpp(spp)
                sim.setDenoiser(denoise)
                times, server, errs = [], [], []
                for k in range(args.frames + 1):
                    ref, gt = refs[k % len(refs)], stored[k % len(refs)]
                    heading = frames.heading_from_yaw(math.radians(int(ref["heading_id"].split("_")[1])))
                    started = time.perf_counter()
                    sim.newEpisode([args.scene], [ref["node_id"]], [heading], [0.0])
                    elapsed = time.perf_counter() - started
                    st = sim.getState()[0]
                    got = st.radiance.astype(np.float32) @ LUMA
                    want = gt["s0"].astype(np.float32) @ LUMA
                    err = float(np.abs(got - want).mean() / want.mean())
                    if k == 0:
                        setting["warmup_s"] = round(elapsed, 3)
                    else:
                        times.append(elapsed)
                        server.append(dict(sim._client.last_timing))
                        errs.append(err)
                    if elapsed > args.budget_s:
                        over_budget = True
                        break
                if times:
                    mean = sum(times) / len(times)
                    keys = sorted({k for s in server for k in s if k.endswith("_s")})
                    setting.update({
                        "fps": round(1.0 / mean, 3), "frame_ms": round(1000 * mean, 1),
                        **{f"{k}_ms": round(1000 * sum(s.get(k, 0.0) for s in server) / len(server), 1) for k in keys},
                        "s0_err": round(sum(errs) / len(errs), 4), "frames": len(times)})
                    if "gpu_s_ms" in setting:
                        # with dr.freeze the per-frame tracing, codegen and compile disappear: what remains is the
                        # GPU work plus denoise and post-processing (replay overhead not included)
                        est = setting["gpu_s_ms"] + setting["denoise_s_ms"] + setting["post_s_ms"]
                        setting["freeze_estimate_ms"] = round(est, 1)
                        setting["freeze_estimate_fps"] = round(1000.0 / est, 2) if est > 0 else None
                    rows.append(setting)
                else:
                    skipped.append(setting)
                print(json.dumps(setting), flush=True)

    print("\n| mode | spp | denoise | fps | frame ms | trace ms | compile ms | GPU ms | denoise ms | post ms | "
          "freeze est. fps | S0 err | warm-up s |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['mode']} | {r['spp']} | {'on' if r['denoise'] else 'off'} | {r['fps']} | {r['frame_ms']} | "
              f"{r.get('trace_s_ms', '-')} | {r.get('compile_s_ms', '-')} | {r.get('gpu_s_ms', '-')} | {r['denoise_s_ms']} | "
              f"{r['post_s_ms']} | {r.get('freeze_estimate_fps', '-')} | {r['s0_err']} | {r.get('warmup_s')} |")
    if skipped:
        print("\nskipped (over --budget-s): " + ", ".join(
            f"{s['mode']}/{s['spp']}/{'dn' if s['denoise'] else 'raw'}" for s in skipped))
    if args.out:
        args.out.write_text(json.dumps({"server": info, "scene": args.scene, "variant": args.variant,
                                        "rows": rows, "skipped": skipped}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
