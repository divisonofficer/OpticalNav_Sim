#!/usr/bin/env python3
"""Look at every object physics.json decides on: contact sheets to check the movable / fixed verdicts by eye.

    python3.10 tools/physics_review.py --pack packs/blend-cache --scene blend_REF_PH33 [--out DIR] [--per-group 4]

For each object of the scene's physics.json (tools/physics_tag.py) a tile: the object in its place, orange in the Solid
view of the scene (pack_solid: the camera of eight around it, and one from above, that sees most of it), beside it the
object alone, filling its half; under them its number and id, the verdict (PUSH: movable and within the robot's reach,
MOVE: movable out of reach, FIX: fixed), the confidence and the rule's reason, and the features the rules read (size,
mass, what it stands on, what it touches). Objects whose label stem, verdict and rule are the same are many in some
scenes (a plant's 600 leaves): ``--per-group`` of them are drawn, the rest listed under the first.

Writes OUT/<scene>/sheet_NNN.jpg (12 tiles a sheet) and OUT/<scene>/objects.csv (one row an object, its group's
representative). The scene's triangles are matched to physics.json's parts by centroid (solid_filters.scene_index), so
a merged scene is drawn as loaded. Needs the GPU (torch + nvdiffrast, as the viewer); reads only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TILE_W, CTX_W, ISO_W, IMG_H, CAPTION_H = 560, 320, 240, 240, 54
COLS, ROWS = 3, 4
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def verdict(o: dict) -> str:
    return "PUSH" if o["movable"] and o["robot_reachable"] else "MOVE" if o["movable"] else "FIX"


def rule_of(reason: str) -> str:
    """The rule that decided, without its numbers (the grouping key)."""
    return re.sub(r"[:(~\d].*", "", reason).strip()[:40]


def stem(label: str) -> str:
    return re.sub(r"[\s_.]*\d+$", "", re.sub(r"\.\d+$", "", label or "")).strip()


def look_at(eye, at) -> np.ndarray:
    eye, at = np.asarray(eye, float), np.asarray(at, float)
    f = at - eye
    f /= max(np.linalg.norm(f), 1e-9)
    up = np.array([0.0, 1.0, 0.0]) if abs(f[1]) < 0.98 else np.array([0.0, 0.0, -1.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, u, -f, eye
    return m


def cameras(lo, hi, hfov: float, context: float):
    """Cameras around a box: eight at 35 degrees up, one from above; ``context`` times the distance that fits it."""
    c, r = (lo + hi) / 2.0, max(float(np.linalg.norm(hi - lo)) / 2.0, 0.05)
    dist = r / math.tan(math.radians(hfov) / 2.0) * context + 0.05
    out = []
    for el, azs in ((35.0, range(0, 360, 45)), (80.0, (0,))):
        for az in azs:
            a, e = math.radians(az), math.radians(el)
            d = np.array([math.cos(e) * math.cos(a), math.sin(e), math.cos(e) * math.sin(a)])
            out.append(look_at(c + d * dist, c))
    return out


def review(scene_dir: Path, out_dir: Path, per_group: int = 4) -> dict:
    from PIL import Image, ImageDraw, ImageFont
    from opticalnav_sim import pack_solid, solid_filters

    meta = json.loads((scene_dir / "scene.json").read_text())
    xml = ((meta.get("variants") or {}).get("base") or {}).get("xml", "base.xml")
    phys = json.loads((scene_dir / "physics.json").read_text())
    objects = phys["objects"]
    solid = pack_solid.PackSolid(scene_dir, xml)
    index = solid_filters.scene_index(scene_dir, xml)
    cen = ((solid.v[solid.f[:, 0]] + solid.v[solid.f[:, 1]] + solid.v[solid.f[:, 2]]) / 3.0).astype(np.float32)
    part = solid_filters.match(index, cen)
    part_obj = np.array([p["object"] for p in index["parts"]] + [-1], np.int64)
    tri_obj = part_obj[part]
    font = ImageFont.truetype(FONT, 11) if Path(FONT).is_file() else ImageFont.load_default()
    bold = ImageFont.truetype(FONT.replace("Sans.ttf", "Sans-Bold.ttf"), 11) if Path(FONT).is_file() else font

    # the groups: same label stem, verdict and rule; the first per_group drawn
    groups: dict = {}
    for k, o in enumerate(objects):
        groups.setdefault((stem(o["label"]), verdict(o), rule_of(o["reason"])), []).append(k)
    order = sorted(groups.items(), key=lambda kv: ({"PUSH": 0, "MOVE": 1, "FIX": 2}[kv[0][1]], kv[0][2], kv[0][0]))
    shown, rep_of = [], {}
    for key, ks in order:
        for k in ks:
            rep_of[k] = ks[0]
        for k in ks[:per_group]:
            shown.append((k, len(ks) - per_group if k == ks[0] and len(ks) > per_group else 0))

    out = out_dir / scene_dir.name
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("sheet_*.jpg"):
        old.unlink()
    tiles, rows = [], []
    for k, more in shown:
        o = objects[k]
        tris = np.nonzero(tri_obj == k)[0]
        tile = Image.new("RGB", (TILE_W, IMG_H + CAPTION_H), (18, 20, 23))
        if len(tris):
            p = solid.v[solid.f[tris].ravel()]
            lo, hi = p.min(0), p.max(0)
            best, best_n = None, -1
            for cam in cameras(lo, hi, 60.0, 2.2):
                try:
                    ids, _ = solid.ids(cam, 128, 96, 60.0)
                except RuntimeError:   # nvdiffrast's clipper overflowing (a camera inside huge triangles): another one
                    continue
                n = int(np.isin(ids[ids >= 0], tris).sum())
                if n > best_n:
                    best, best_n = cam, n
            if best is None:
                best = cameras(lo, hi, 60.0, 2.2)[-1]
            try:
                ctx, _ = solid.render(best, CTX_W, IMG_H, 60.0, tint=tris)
            except RuntimeError:
                ctx = np.zeros((IMG_H, CTX_W, 4), np.float32)
            # alone: from the same side, filling its half
            c = (lo + hi) / 2.0
            d = best[:3, 3] - c
            d /= max(np.linalg.norm(d), 1e-9)
            r = max(float(np.linalg.norm(hi - lo)) / 2.0, 0.02)
            iso, _ = solid.render(look_at(c + d * (r / math.tan(math.radians(20.0)) * 1.05), c), ISO_W, IMG_H, 40.0, subset=tris)
            for img, x in ((ctx, 0), (iso, CTX_W)):
                tile.paste(Image.fromarray((np.clip(img[..., :3], 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)), (x, 0))
            if best_n == 0:
                ImageDraw.Draw(tile).text((6, 6), "hidden from every side", fill=(255, 120, 120), font=font)
        else:
            ImageDraw.Draw(tile).text((10, 10), "no triangle of it found in the scene as loaded", fill=(255, 120, 120), font=font)
        v = verdict(o)
        colour = {"PUSH": (90, 220, 120), "MOVE": (110, 170, 255), "FIX": (200, 200, 200)}[v]
        draw = ImageDraw.Draw(tile)
        sx, sy, sz = o.get("size_m") or (0, 0, 0)
        sup = o.get("support") or {}
        touch = (o.get("touches") or {})
        line1 = f"#{k}  {o['id'][:46]}" + (f"   (+{more} more like it)" if more else "")
        line2 = f"{o['confidence']:.2f}  {o['reason'][:84]}"
        bits = [f"{sx:.2f}x{sy:.2f}x{sz:.2f} m", f"{o['mass_kg']:.1f} kg" if o.get("mass_kg") is not None else "no mass",
                f"on {sup.get('kind')}" + (f" ({', '.join(sup.get('on') or [])[:34]})" if sup.get("on") else ""),
                f"bottom {sup['bottom_z'] - sup['floor_z']:+.2f} m over floor" if sup.get("floor_z") is not None else "no floor",
                f"walls {len(touch.get('wall') or [])}" + (" ceiling" if touch.get("ceiling") else "")]
        draw.text((6, IMG_H + 3), v, fill=colour, font=bold)
        draw.text((48, IMG_H + 3), line1, fill=(230, 230, 230), font=font)
        draw.text((6, IMG_H + 19), line2, fill=(200, 200, 170), font=font)
        draw.text((6, IMG_H + 35), " · ".join(bits)[:100], fill=(160, 175, 190), font=font)
        tiles.append(tile)
    for s in range(0, len(tiles), COLS * ROWS):
        sheet = Image.new("RGB", (COLS * TILE_W + (COLS - 1) * 4, ROWS * (IMG_H + CAPTION_H) + (ROWS - 1) * 4), (40, 44, 48))
        for i, t in enumerate(tiles[s:s + COLS * ROWS]):
            sheet.paste(t, ((i % COLS) * (TILE_W + 4), (i // COLS) * (IMG_H + CAPTION_H + 4)))
        sheet.save(out / f"sheet_{s // (COLS * ROWS):03d}.jpg", quality=88)
    with open(out / "objects.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "id", "label", "verdict", "confidence", "rule", "reason", "class", "keyword", "size_x", "size_y",
                    "size_z", "mass_kg", "material", "support", "support_on", "bottom_over_floor", "walls", "ceiling",
                    "parts", "drawn_triangles", "group_rep", "drawn"])
        drawn = {k for k, _ in shown}
        for k, o in enumerate(objects):
            sup, touch = o.get("support") or {}, o.get("touches") or {}
            w.writerow([k, o["id"], o["label"], verdict(o), o["confidence"], rule_of(o["reason"]), o["reason"], o.get("class"),
                        o.get("keyword"), *(o.get("size_m") or ("", "", "")), o.get("mass_kg"), o.get("material"),
                        sup.get("kind"), "; ".join(sup.get("on") or []),
                        round(sup["bottom_z"] - sup["floor_z"], 3) if sup.get("floor_z") is not None else "",
                        len(touch.get("wall") or []), bool(touch.get("ceiling")), len(o.get("parts") or []),
                        int((tri_obj == k).sum()), rep_of[k], k in drawn])
    print(f"{scene_dir.name}: {len(objects)} objects, {len(tiles)} drawn on {math.ceil(len(tiles) / (COLS * ROWS))} sheets "
          f"-> {out}", flush=True)
    return {"objects": len(objects), "drawn": len(tiles)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", type=Path, action="append", required=True)
    ap.add_argument("--scene", action="append", default=[], help="scene id (repeatable; default: every scene with a physics.json)")
    ap.add_argument("--out", type=Path, default=REPO / "physics_review")
    ap.add_argument("--per-group", type=int, default=4)
    args = ap.parse_args()
    for pack in args.pack:
        for d in sorted((pack / "scenes").glob("*/physics.json")):
            if args.scene and d.parent.name not in args.scene:
                continue
            review(d.parent, args.out, args.per_group)
    return 0


if __name__ == "__main__":
    sys.exit(main())
