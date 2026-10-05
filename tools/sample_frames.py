#!/usr/bin/env python3
"""Save example frames for a report: one dataset reference view rendered per setting.

    python tools/sample_frames.py --pack packs/opticalnav-v0.2 --scene SCENE --server URL \
        --mode polar --spp 128 --denoise off,on --out samples/

Writes JPEGs of the RGB preview and, in polar mode, DoLP (black to red, fixed 0-0.2
range) and AoLP (hue, brightness DoLP) maps for the simulator frames and for the
stored dataset frame of the same view, plus samples.json with the per-image S0
error and timing.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opticalnav_sim import MatterSim, frames, stokes  # noqa: E402


def save(img: np.ndarray, path: Path) -> None:
    Image.fromarray(img).save(path, quality=88)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--mode", default="polar")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--spp", default="128")
    ap.add_argument("--denoise", default="off,on")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tag", default="sim", help="suffix of the raw .npz, e.g. the Mitsuba variant")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((args.pack / "scenes" / args.scene / "scene.json").read_text())
    ref = next(r for r in meta["reference_views"] if r["variant"] == args.variant)
    folder = args.pack / "scenes" / args.scene / ref["path"]
    stored = np.load(folder / "stokes_data.npz")
    s0r, s1r, s2r = (stored[k].astype(np.float32) for k in ("s0", "s1", "s2"))
    record_path = args.out / "samples.json"
    record = json.loads(record_path.read_text()) if record_path.is_file() else {"view": ref, "frames": []}
    if not (args.out / "dataset_rgb.jpg").is_file():
        save(np.asarray(Image.open(folder / "polar_rgb_preview.png").convert("RGB")), args.out / "dataset_rgb.jpg")
        dr = stokes.derived(s0r, s1r, s2r)
        save(stokes.dolp_rgb(dr["dolp"]), args.out / "dataset_dolp.jpg")
        save(stokes.aolp_rgb(dr["aolp_deg"], dr["dolp"]), args.out / "dataset_aolp.jpg")

    sim = MatterSim.Simulator()
    sim.setDatasetPath(args.server)
    sim.setNavGraphPath(str(args.pack / "connectivity"))
    sim.setVariant(args.variant)
    sim.setRenderMode(args.mode)
    sim.setStokesDtype("float32")
    sim.initialize()
    heading = frames.heading_from_yaw(math.radians(int(ref["heading_id"].split("_")[1])))
    for spp in (int(s) for s in args.spp.split(",")):
        for denoise in (d == "on" for d in args.denoise.split(",")):
            sim.setRenderSpp(spp)
            sim.setDenoiser(denoise)
            sim.newEpisode([args.scene], [ref["node_id"]], [heading], [0.0])
            st = sim.getState()[0]
            stem = f"{args.mode}_{spp}{'_dn' if denoise else ''}"
            save(np.ascontiguousarray(st.rgb[:, :, ::-1]), args.out / f"{stem}_rgb.jpg")
            got = st.radiance.astype(np.float32) @ stokes.LUMA
            want = s0r @ stokes.LUMA
            diff = np.abs(got - want)
            clip = np.quantile(want, 0.99)
            entry = {"mode": args.mode, "spp": spp, "denoise": denoise, "rgb": f"{stem}_rgb.jpg",
                     "s0_err": round(float(diff.mean() / want.mean()), 4),
                     # robust views of the same error: median pixel, and with the brightest 1% clipped
                     "s0_err_median": round(float(np.median(diff) / want.mean()), 4),
                     "s0_err_clipped": round(float(np.abs(np.minimum(got, clip) - np.minimum(want, clip)).mean() / want.mean()), 4),
                     "top1pct_share": round(float(np.sort(diff.ravel())[-diff.size // 100:].sum() / diff.sum()), 3),
                     "timing": sim._client.last_timing}
            np.savez_compressed(args.out / f"{stem}_{args.tag}.npz", **(
                {k: v.astype(np.float16) for k, v in st.stokes.items()} if st.stokes is not None
                else {"radiance": st.radiance.astype(np.float16)}))
            if st.stokes is not None:
                d = stokes.derived(st.stokes["s0"].astype(np.float32), st.stokes["s1"].astype(np.float32),
                                   st.stokes["s2"].astype(np.float32))
                save(stokes.dolp_rgb(d["dolp"]), args.out / f"{stem}_dolp.jpg")
                save(stokes.aolp_rgb(d["aolp_deg"], d["dolp"]), args.out / f"{stem}_aolp.jpg")
                entry["dolp"], entry["aolp"] = f"{stem}_dolp.jpg", f"{stem}_aolp.jpg"
                ref_dolp = stokes.derived(s0r, s1r, s2r)["dolp"]
                entry["dolp_mae"] = round(float(np.abs(d["dolp"] - ref_dolp).mean()), 4)
            record["frames"] = [f for f in record["frames"] if f.get("rgb") != entry["rgb"]] + [entry]
            print(json.dumps(entry), flush=True)
    record_path.write_text(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
