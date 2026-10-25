#!/usr/bin/env python3
"""Render-test pack scenes on the CPU (no GPU, no server), and pick their start poses from what it sees.

    CUDA_VISIBLE_DEVICES= python3.10 tools/cpu_check.py --pack packs/opticalnav-sim-v1 \
        --project <robomituba>/out/opticalnav/opticalnav-sim-v1 [--scene SCENE ...] [--apply-spawns] [--out DIR]

Mitsuba's ``scalar_rgb`` variant, the server's own scene preparation (lamp_lights_text: lamps glow from their
parts, a light's fixture by the importer's tag) and ``path_nocaustics`` with the softboxes hidden from the
camera, so the picture is the server's without polarization: RGB intensity only, no Stokes, no mirror pool or
flash. Per scene it

* loads base.xml (time and shapes),
* with ``--apply-spawns``, scores every candidate start of the three largest traversable regions
  (build_pack.spawn_candidates) by its least open view (median depth over four headings, capped at 5 m, 0 if the
  view is black) and writes the best of each to scene.json (``spawn``; the old ones under ``spawn_before``), as
  tools/check_spawns.py does through the server,
* renders four headings from each of its three spawn points at ``--size`` and ``--spp``, and checks that the
  images are finite and lit (not black) and that the brightest pixels are not far above the median (fireflies),
* saves a contact sheet ``<out>/cpu_<scene>.jpg`` and prints one line.

Exit code 1 if any scene has a black or non-finite view. A scene the GPU renderer shows differently (polarization,
mirrors, flash) is outside what this checks.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mitsuba as mi  # noqa: E402

mi.set_variant("scalar_rgb")
mi.set_log_level(mi.LogLevel.Warn)
from opticalnav_sim import frames, server  # noqa: E402
import build_pack  # noqa: E402

HEADINGS = (0.0, 90.0, 180.0, 270.0)
OPEN_CAP_M, MIN_LUMA = 5.0, 0.004
STOKES = re.compile(r'\s*<integrator type="stokes">.*?</integrator>\s*</integrator>', re.S)


def load_scene(scene_dir: Path):
    """The scene as the server prepares it (lamps lit from their parts), without the polarized wrapper."""
    text = STOKES.sub("", (scene_dir / "base.xml").read_text(), count=1)
    mi.file_resolver().append(str(scene_dir))

    def mesh_info(kind: str, filename: str):
        mesh = mi.load_dict({"type": kind, "filename": str(scene_dir / filename)})
        area = mesh.surface_area()
        return float(area if isinstance(area, float) else area[0]), float(mesh.bbox().min.y)

    def mesh_bounds(kind: str, body: str):
        body = re.sub(r'<ref\b[^>]*/>|<bsdf\b.*</bsdf>|<bsdf\b[^>]*/>|<emitter\b.*?</emitter>', "", body, flags=re.S)
        body = re.sub(r'(<string name="filename" value=")(?!/)', rf"\g<1>{scene_dir}/", body)
        box = mi.load_string(f'<shape version="3.0.0" type="{kind}">{body}<bsdf type="diffuse"/></shape>').bbox()
        return list(box.min), list(box.max)

    text, _ = server.lamp_lights_text(text, mesh_info, mesh_bounds=mesh_bounds)
    return mi.load_string(text)


def camera(c2w, width: int, height: int, spp: int):
    origin, target, up = frames.lookat(c2w)
    return mi.load_dict({"type": "perspective", "fov": 90.0,
                         "to_world": mi.ScalarTransform4f.look_at(origin=origin, target=target, up=up),
                         "film": {"type": "hdrfilm", "width": width, "height": height, "rfilter": {"type": "box"}},
                         "sampler": {"type": "independent", "sample_count": spp}})


PATH = None
DEPTH = None


def integrators():
    global PATH, DEPTH
    if PATH is None:
        inner = {"type": "path_nocaustics", "max_depth": 6, "hidden": server.FILL_PREFIX}
        PATH = mi.load_dict(inner)
        DEPTH = mi.load_dict({"type": "aov", "aovs": "depth:depth", "integrator": {**inner, "max_depth": 3}})
    return PATH, DEPTH


def view(scene, c2w, size: tuple[int, int], spp: int, aov: bool = False) -> np.ndarray:
    path, depth = integrators()
    return np.array(mi.render(scene, sensor=camera(c2w, size[0], size[1], spp), spp=spp,
                              integrator=depth if aov else path), dtype=np.float32)


def openness(scene, mount: dict, pose: dict) -> float:
    """Least open of the four views of ``pose`` (median depth along the view, capped); 0 if one is black."""
    worst = OPEN_CAP_M
    x, y, z = pose["position"]
    for h in HEADINGS:
        img = view(scene, frames.camera_at_position((x, y, z), math.radians(h), 0.0, mount), (32, 24), 2, aov=True)
        rgb, depth = img[..., :3], img[..., 3]
        ok = np.isfinite(depth) & (depth > 0)
        if float(rgb.mean()) < MIN_LUMA or ok.mean() < 0.5:
            return 0.0
        worst = min(worst, float(np.median(depth[ok])))
    return worst


def pick_spawns(scene, project: Path, scene_dir: Path, meta: dict) -> list[dict]:
    mount, eye = meta["camera"]["mount"], float(meta["spawn"][0]["position"][2])
    chosen = []
    for candidates in build_pack.spawn_candidates(project / "scenes" / scene_dir.name, eye):
        best = max(((openness(scene, mount, p), i) for i, p in enumerate(candidates)), default=(0.0, 0))
        if best[0] > 0:
            chosen.append({**candidates[best[1]], "open_m": round(best[0], 2)})
    return chosen


def check_scene(pack: Path, project: Path, name: str, apply: bool, size, spp: int, out: Path) -> bool:
    scene_dir = (pack / "scenes" / name).resolve()
    meta_path = scene_dir / "scene.json"
    meta = json.loads(meta_path.read_text())
    t0 = time.perf_counter()
    scene = load_scene(scene_dir)
    load_s = time.perf_counter() - t0
    note = ""
    if apply:
        chosen = pick_spawns(scene, project, scene_dir, meta)
        if chosen:
            meta["spawn_before"] = meta.get("spawn_before") or meta["spawn"]
            meta["spawn"] = chosen
            tmp = meta_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(meta, indent=1))
            os.replace(tmp, meta_path)
            note = f", spawns open {[s['open_m'] for s in chosen]} m"
    mount = meta["camera"]["mount"]
    tiles, rows = [], []
    t1 = time.perf_counter()
    for sp in meta["spawn"]:
        for h in HEADINGS:
            img = view(scene, frames.camera_at_position(sp["position"], math.radians(h), 0.0, mount), size, spp)
            lum = img.mean(-1)
            rows.append({"finite": bool(np.isfinite(img).all()), "mean": float(lum.mean()),
                         "black": float((lum < 1e-4).mean()),
                         "hot": float(np.percentile(lum, 99.9)) / max(float(np.median(lum)), 1e-6)})
            tiles.append(img)
    render_s = time.perf_counter() - t1
    scale = float(np.percentile(np.concatenate([t.ravel() for t in tiles]), 99)) or 1.0
    sheet = [np.concatenate([np.clip(t / scale, 0, 1) ** (1 / 2.2) for t in tiles[i:i + 4]], 1) for i in range(0, len(tiles), 4)]
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.concatenate(sheet, 0) * 255).astype(np.uint8)).save(out / f"cpu_{name}.jpg", quality=85)
    bad = [r for r in rows if not r["finite"] or r["black"] > 0.6]
    print(f"[cpu] {name}: loaded {load_s:.0f}s ({len(scene.shapes())} shapes), {len(rows)} views in {render_s:.0f}s, "
          f"mean {np.mean([r['mean'] for r in rows]):.3f}, black views {sum(r['black'] > 0.6 for r in rows)}, "
          f"non-finite {sum(not r['finite'] for r in rows)}, hottest pixel {max(r['hot'] for r in rows):.0f}x median{note}"
          f"{'  FAIL' if bad else ''}", flush=True)
    return not bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--project", required=True, type=Path, help="robomituba project holding scenes/<scene>/authoring_map.json")
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--apply-spawns", action="store_true")
    ap.add_argument("--size", default="192x144")
    ap.add_argument("--spp", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("cpu_check"))
    args = ap.parse_args()
    size = tuple(int(v) for v in args.size.split("x"))
    scenes = args.scene or sorted(p.parent.name for p in (args.pack / "scenes").glob("*/base.xml"))
    ok = all([check_scene(args.pack, args.project.resolve(), s, args.apply_spawns, size, args.spp, args.out) for s in scenes])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
