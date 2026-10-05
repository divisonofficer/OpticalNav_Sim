#!/usr/bin/env python3
"""Re-render exported OpticalNav episodes and write them as an export-style bundle.

    python tools/replay_episode.py --pack packs/opticalnav-v0.2 --server URL \
        --episodes <bundle dir | bundle.zip | bundle.zip.part000 | robomituba project | pack | episode.json> \
        [--scene S] [--split val_unseen] [--limit 2] \
        --variants base,perturbed --spp 64 --out runs/replay \
        [--exposure scene|episode|auto|fixed:<exposure>[,<white>]] [--image-format jpeg|png|none] [--hdr npz,exr|none] \
        [--compare <same kinds of source>]

Every step renders the dataset camera at (path_nodes[i], path_headings[i]); steps
that share a node and heading share a frame. The output has the export's layout
(index.jsonl, images/, polarization_raw/, episodes/, graph/, dataset_meta.json;
see opticalnav_sim.bundle), so code that reads exported bundles reads it as is.

--compare scores each frame against the dataset's own render of the same view,
from an export bundle's polarization_raw/ (the usual case: robomituba prunes raw
renders after export), a project that still has them, or a pack's reference
frames. It writes compare.jsonl and prints per-episode S0 error, S0 8x8 block
correlation and DoLP error. Sources are read by opticalnav_sim.sources; split
uploads (bundle.zip.partNNN) are read in place without joining or unzipping.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opticalnav_sim import episodes, stokes  # noqa: E402
from opticalnav_sim.bundle import BundleWriter  # noqa: E402
from opticalnav_sim.client import RenderClient  # noqa: E402
from opticalnav_sim.navgraph import NavGraph  # noqa: E402
from opticalnav_sim.sources import open_source  # noqa: E402
from opticalnav_sim.tonemap import Tone  # noqa: E402

def block(a: np.ndarray, k: int = 8) -> np.ndarray:
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def score(got: dict, want: dict) -> dict:
    g0, w0 = (np.asarray(x["s0"], np.float32) @ stokes.LUMA for x in (got, want))
    gd = stokes.derived(*(np.asarray(got[k], np.float32) for k in ("s0", "s1", "s2")))["dolp"]
    wd = stokes.derived(*(np.asarray(want[k], np.float32) for k in ("s0", "s1", "s2")))["dolp"]
    return {"s0_err": float(np.abs(g0 - w0).mean() / max(float(w0.mean()), 1e-9)),
            "s0_corr_8x8": float(np.corrcoef(block(g0).ravel(), block(w0).ravel())[0, 1]),
            "dolp_mae": float(np.abs(gd - wd).mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--server", default=None)
    ap.add_argument("--episodes", required=True, type=Path, help="episode source (see opticalnav_sim.sources)")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--split", action="append", default=None)
    ap.add_argument("--limit", type=int, default=None, help="episodes to replay")
    ap.add_argument("--max-steps", type=int, default=None, help="replay only the first N steps of each episode")
    ap.add_argument("--variants", default="base")
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--exposure", default="scene")
    ap.add_argument("--image-format", default="jpeg", choices=["jpeg", "png", "none"])
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--modalities", default="polar_rgb_preview,dop,aolp")
    ap.add_argument("--hdr", default="npz", help="comma list of npz, exr; or none")
    ap.add_argument("--compare", type=Path, default=None)
    args = ap.parse_args()

    source = open_source(args.episodes)
    eps = source.episodes(args.scene, args.split)[: args.limit]
    if not eps:
        raise SystemExit(f"no episodes in {args.episodes} ({source.kind})")
    reference = open_source(args.compare) if args.compare else None
    print(f"[source] {len(eps)} episode(s) from {source.kind} {args.episodes}"
          + (f"; comparing with {reference.kind} {args.compare}" if reference else ""), flush=True)
    client = RenderClient(args.server)
    available = client.info()["scans"]
    variants = args.variants.split(",")
    hdr = () if args.hdr == "none" else tuple(args.hdr.split(","))
    tone_scene = eps[0].scene_id
    tone = Tone.parse(args.exposure, args.pack / "scenes" / tone_scene, variants[0])
    writer = BundleWriter(args.out, modalities=args.modalities.split(","), image_format=args.image_format,
                          jpeg_quality=args.jpeg_quality, hdr=hdr, tone=tone)
    compare_log = (args.out / "compare.jsonl").open("w") if args.compare else None
    graphs, metas = {}, {}
    started = time.time()
    frames_rendered = 0
    for ep in eps:
        scene = ep.scene_id
        if scene not in available:
            raise SystemExit(f"{scene} is not on the render server")
        if tone.mode == "episode":
            tone.params = None
        if scene not in graphs:
            graphs[scene] = NavGraph.load(args.pack / "connectivity", scene)
            metas[scene] = json.loads((args.pack / "scenes" / scene / "scene.json").read_text())
            for src in (open_source(args.pack), source, reference):
                data = src.graph(scene) if src else None
                if data:
                    writer.add_graph(scene, data)
                    break
        graph, cam = graphs[scene], metas[scene]["camera"]
        steps = ep.steps[: args.max_steps] if args.max_steps else ep.steps
        keys = list(dict.fromkeys(s.frame_key for s in steps))
        writer.add_episode(ep.raw)
        for variant in variants:
            if variant not in available[scene]:
                print(f"[skip] {scene} has no {variant} variant on this server", flush=True)
                continue
            scores = []
            for node, heading in keys:
                fid = episodes.frame_id(scene, node, heading)
                if writer.has(variant, fid):
                    continue
                x, y, _ = graph.pos[graph.index(node)]
                yaw = float(heading.split("_", 1)[1])
                c2w = episodes.camera_to_world((x, y), yaw, cam["mount"])
                view = {"scan": scene, "variant": variant, "camera_to_world": c2w, "width": cam["width"],
                        "height": cam["height"], "hfov_deg": cam["hfov_deg"], "spp": args.spp, "seed": args.seed,
                        "mode": "polar", "preview": False}
                out = client.render([view], "float32")[0]
                render = {"renderer": "opticalnav_sim", "spp": args.spp, "seed": args.seed, "mode": "polar",
                          "denoise": False, "variant": variant, "server_s": round(client.last_render_seconds, 3)}
                writer.add_frame(scene_id=scene, variant=variant, node_id=node, heading_id=heading,
                                 camera_to_world=c2w, base_pose=episodes.base_pose((x, y), yaw, cam["mount"]),
                                 fov_deg=cam["hfov_deg"], resolution=(cam["width"], cam["height"]),
                                 stokes_images=out, render=render)
                frames_rendered += 1
                if reference:
                    want = reference.stokes(scene, variant, node, heading)
                    if want is not None:
                        row = {"episode_id": ep.episode_id, "variant": variant, "frame_id": fid, **score(out, want)}
                        compare_log.write(json.dumps(row) + "\n")
                        scores.append(row)
            msg = f"[replay] {ep.episode_id} {variant}: {len(keys)} frames"
            if scores:
                msg += (f"  S0 err {np.mean([s['s0_err'] for s in scores]):.4f} (max {max(s['s0_err'] for s in scores):.4f})"
                        f"  S0 corr {np.mean([s['s0_corr_8x8'] for s in scores]):.4f}"
                        f"  DoLP MAE {np.mean([s['dolp_mae'] for s in scores]):.4f}  ({len(scores)} compared)")
            print(msg, flush=True)
    meta = writer.close({"replay": {"episodes": [e.episode_id for e in eps], "spp": args.spp, "seed": args.seed,
                                    "variants": variants, "frames_rendered": frames_rendered,
                                    "seconds": round(time.time() - started, 1),
                                    "server": client.url, "renderer": client.info().get("modes")}})
    if compare_log:
        compare_log.close()
    print(f"[done] {meta['frame_count']} frames, {meta['index_rows']} index rows, {meta['episode_count']} episodes "
          f"-> {args.out}  ({time.time() - started:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
