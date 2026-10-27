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

With ``--format exr`` each frame is one half-float (16-bit) OpenEXR file instead, ``<out>/images/<route id>/exr/NNNN.exr``,
with all 16 channels of a polarized render of both bands (RGB and near-infrared): ``S0.R S0.G S0.B`` ... ``S3.R S3.G S3.B``
(the Stokes parameters of the three colour bands, linear radiance) and ``NIR.S0`` ... ``NIR.S3`` (the near-infrared
Stokes parameters, collapsed to one channel), ZIP-compressed, (OpenEXR's Python binding cannot write string attributes, so the scene, route, pose index, seed and spp
are not in the file: they are the route file's, the file name's and ``<out>/render_info.json``'s). ``--preview`` also writes the kit's RGB JPEG next to it. A frame is done when both files exist: a rerun skips them (and is safe to stop at any time). Unlike the kit's worker
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


def done(rgb: Path, fmt: str = "kit") -> bool:
    if fmt == "exr":
        exr = rgb.parent.parent / "exr" / (rgb.stem + ".exr")
        return exr.is_file() and exr.stat().st_size > 0
    pol = rgb.parent.parent / "pol" / (rgb.stem + ".png")
    return rgb.is_file() and rgb.stat().st_size > 0 and pol.is_file() and pol.stat().st_size > 0


def write_exr(path: Path, out: dict) -> None:
    """One half-float EXR with the 12 RGB Stokes channels and the 4 near-infrared ones (OpenEXR's Python binding with
    Imath: ``OutputFile``)."""
    import Imath
    import OpenEXR

    channels = {}
    for i in range(4):
        rgb = np.asarray(out[f"s{i}"], np.float32)
        for c, name in enumerate("RGB"):
            channels[f"S{i}.{name}"] = np.ascontiguousarray(rgb[..., c])
        channels[f"NIR.S{i}"] = np.ascontiguousarray(np.asarray(out[f"nir_s{i}"], np.float32))
    height, width = next(iter(channels.values())).shape
    header = OpenEXR.Header(width, height)
    header["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
    half = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    header["channels"] = {name: half for name in sorted(channels)}
    tmp = str(path) + ".part"
    exr = OpenEXR.OutputFile(tmp, header)
    exr.writePixels({name: a.astype(np.float16).tobytes() for name, a in channels.items()})
    exr.close()
    os.replace(tmp, path)


def encode_pol(out: dict) -> np.ndarray:
    s0, s1, s2 = (np.asarray(out[c], np.float32) for c in ("s0", "s1", "s2"))
    s0, s1, s2 = (a.mean(axis=-1) if a.ndim == 3 else a for a in (s0, s1, s2))
    s0, s1, s2 = (np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0) for a in (s0, s1, s2))
    dolp = np.clip(np.sqrt(s1 ** 2 + s2 ** 2) / np.maximum(s0, 1e-6), 0, 1)
    aolp = (np.degrees(0.5 * np.arctan2(s2, s1)) % 180) / 180
    return np.stack([dolp * 255, aolp * 255, np.clip(s0 / max(np.percentile(s0, 99), 1e-6), 0, 1) * 255], -1).astype(np.uint8)


def render_scene(client: RenderClient, pack: Path, scene: str, routes: dict, out: Path, variant: str, spp: int,
                 limit: int | None, fmt: str = "kit", preview: bool = False) -> int:
    camera = json.loads((pack / "scenes" / scene / "scene.json").read_text())["camera"]
    todo = [(rid, k, p) for rid in sorted(routes) for k, p in enumerate(routes[rid]["poses"])
            if not done(out / "images" / rid / "rgb" / f"{k:04d}.jpg", fmt)]
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        print(f"{scene}: nothing left", flush=True)
        return 0

    def view(pose, seed, n):
        x, y, yaw = pose
        return {"scan": scene, "variant": variant, "camera_to_world": frames.rig_camera_to_world(x, y, yaw, camera["mount"]),
                "width": camera["width"], "height": camera["height"], "hfov_deg": camera["hfov_deg"], "spp": n, "seed": seed,
                "mode": "polar", **({"bands": "both"} if fmt == "exr" else {})}

    client.render([view(todo[0][2], 0, 16)])  # the scene loads and records at a small spp first
    t0, n = time.time(), 0
    for rid, k, pose in todo:
        rgb = out / "images" / rid / "rgb" / f"{k:04d}.jpg"
        pol = out / "images" / rid / "pol" / f"{k:04d}.png"
        seed = zlib.crc32(f"{rid}/{k}".encode()) % 40000 * 100003
        result = client.render([view(pose, seed, spp)], "float32")[0]
        if fmt == "exr":
            exr = out / "images" / rid / "exr" / f"{k:04d}.exr"
            exr.parent.mkdir(parents=True, exist_ok=True)
            write_exr(exr, result)
            if preview:
                rgb.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(result["rgb"]).save(str(rgb) + ".part", "JPEG", quality=95)
                os.replace(str(rgb) + ".part", rgb)
        else:
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
    ap.add_argument("--format", choices=("kit", "exr"), default="kit",
                    help="kit: the kit's JPEG and PNG; exr: one 16-bit EXR with the RGB and near-infrared Stokes channels")
    ap.add_argument("--preview", action="store_true", help="with --format exr, also the RGB JPEG")
    ap.add_argument("--limit", type=int, default=None, help="only the first N missing frames of each scene")
    args = ap.parse_args()
    client = RenderClient(args.server, timeout=3600.0)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.format == "exr":
        (args.out / "render_info.json").write_text(json.dumps({
            "format": "one EXR per frame at images/<route id>/exr/NNNN.exr, half float, ZIP",
            "channels": {"S0.R S0.G S0.B ... S3.R S3.G S3.B": "Stokes parameters of the RGB bands, linear radiance",
                         "NIR.S0 ... NIR.S3": "near-infrared Stokes parameters (collapsed to one channel)"},
            "pose": "routes/<scene>.json poses[k] = [x, y, yaw_rad] (dataset floor frame); frame NNNN is pose k",
            "seed": 'zlib.crc32("<route id>/<k>") % 40000 * 100003', "spp": args.spp, "variant": args.variant,
            "width_height": [512, 384], "server": args.server}, indent=1))
    scenes = args.scene or sorted(p.stem for p in args.routes.glob("*.json"))
    total = 0
    for scene in scenes:
        routes = json.loads((args.routes / f"{scene}.json").read_text())
        total += render_scene(client, args.pack, scene, routes, args.out, args.variant, args.spp, args.limit,
                             args.format, args.preview)
    print(f"done: {total} frames", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
