#!/usr/bin/env python3
"""Open the closed doors of pack scenes and make the glass partitions clear, in scenes already built.

    python3 tools/open_doors.py --pack packs/opticalnav-sim-v1 [--scene SCENE ...] [--keep-frost] [--dry-run]

Two things Infinigen's offices and clinics have that stop a walk and a view:

* A door leaf that is closed (``LiteDoorFactory_<seed>_.spawn_asset_<n>``, thin across the wall it stands in) fills
  its doorway (``DoorCasingFactory_<seed>_.spawn_asset_<n>``, same seed and number). Its shapes (the leaf, its
  handle and glass panes) are removed, so the doorway is open. A leaf swung open, thin along the wall's own direction,
  and anything between, stays. A door in the building's outer wall stays closed (``--exterior`` opens it too).
* ``GlassPartitionFrost_*``, the frosted privacy film on the glass partitions, is an opaque pplastic band 1.65 m
  high, so one cannot see into the office through the glass next to the door. It is removed (``--keep-frost``
  leaves it).

The scene XMLs are rewritten, the meshes and textures no file names any more are unlinked (build_pack and
compact_assets rebuild them), and scene.json records what was changed under ``doors_opened``. Rebuild the scene to
undo it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compact_assets import json_refs, scene_lock, xml_refs  # noqa: E402

SHAPE = re.compile(r'(\n  <shape type="[a-z_]+" id="([^"]+)">)(.*?)(\n  </shape>)', re.S)
DOOR = re.compile(r"^(\w*Door\w*?)Factory_(\d+)_\.spawn_asset_(\d+)")
CASING = re.compile(r"^DoorCasingFactory_(\d+)_\.spawn_asset_(\d+)")
PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "int": "<i4", "uint": "<u4",
             "short": "<i2", "ushort": "<u2", "char": "i1"}


def bounds(path: Path):
    """(min, max) of a binary little-endian PLY's vertex positions, or None."""
    with path.open("rb") as f:
        head = b""
        while b"end_header\n" not in head:
            block = f.read(4096)
            if not block:
                return None
            head += block
        header, _ = head.split(b"end_header\n", 1)
        text = header.decode("latin-1")
        if "binary_little_endian" not in text:
            return None
        count, fields, vertex = 0, [], False
        for line in text.splitlines():
            parts = line.split()
            if parts[:2] == ["element", "vertex"]:
                count, vertex = int(parts[2]), True
            elif parts[:1] == ["element"]:
                vertex = False
            elif vertex and parts[:1] == ["property"]:
                if parts[1] not in PLY_TYPES:
                    return None
                fields.append((parts[2], PLY_TYPES[parts[1]]))
        f.seek(len(header) + len(b"end_header\n"))
        v = np.fromfile(f, np.dtype(fields), count)
    pos = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    return pos.min(0), pos.max(0)


def world_boxes(scene_dir: Path, text: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Shape id -> world bounding box, for the shapes placed by a translation alone."""
    out = {}
    for _, sid, body, _ in SHAPE.findall(text):
        if "<rotate" in body or "<scale" in body or "<matrix" in body:
            continue
        ref = re.search(r'<string name="filename" value="([^"]+\.ply)"', body)
        if not ref or not (scene_dir / ref.group(1)).is_file():
            continue
        box = bounds(scene_dir / ref.group(1))
        if box is None:
            continue
        move = re.search(r'<translate x="([-\d.e]+)" y="([-\d.e]+)" z="([-\d.e]+)"', body)
        off = np.array([float(v) for v in move.groups()]) if move else np.zeros(3)
        out[sid] = (box[0] + off, box[1] + off)
    return out


def building_edge(boxes: dict):
    """The outer extent (min, max) of the room walls, x and z: a door by it leads outside."""
    walls = [b for sid, b in boxes.items() if sid.endswith(".wall")]
    if not walls:
        return None
    return np.min([b[0] for b in walls], 0), np.max([b[1] for b in walls], 0)


def closed_doors(boxes: dict, exterior: bool = False) -> list[dict]:
    """Doors whose leaf stands closed in the casing: [{key, casing, shapes}]. A door in the building's outer wall
    (its casing within 0.6 m of the walls' extent) stays closed unless ``exterior``."""
    edge = building_edge(boxes)
    leaves: dict[tuple, list[str]] = defaultdict(list)
    for sid in boxes:
        m = DOOR.match(sid)
        if m and "Casing" not in m.group(1):
            leaves[(m.group(2), m.group(3))].append(sid)
    out = []
    for sid, (lo, hi) in boxes.items():
        m = CASING.match(sid)
        if not m or sid not in boxes or (m.group(1), m.group(2)) not in leaves or "__part" in sid:
            continue
        shapes = leaves[(m.group(1), m.group(2))]
        llo = np.min([boxes[s][0] for s in shapes], 0)
        lhi = np.max([boxes[s][1] for s in shapes], 0)
        ext, lext = hi - lo, lhi - llo
        thin = 0 if ext[0] < ext[2] else 2  # the wall's thin axis (the casing is wide along the wall)
        along = 2 - thin
        center_in = all(lo[a] - 0.3 <= (llo[a] + lhi[a]) / 2 <= hi[a] + 0.3 for a in (0, 2))
        outside = edge is not None and any(
            abs((lo[a] + hi[a]) / 2 - edge[0][a]) < 0.6 or abs((lo[a] + hi[a]) / 2 - edge[1][a]) < 0.6 for a in (0, 2))
        if outside and not exterior:
            continue
        if center_in and lext[thin] < 0.25 and lext[along] > 0.6:  # across the doorway: closed
            out.append({"key": f"{m.group(1)}_{m.group(2)}", "casing": sid, "shapes": shapes,
                        "at": [round(float((lo[0] + hi[0]) / 2), 2), round(float((lo[2] + hi[2]) / 2), 2)]})
    return out


def open_scene(scene_dir: Path, clear_frost: bool = True, dry_run: bool = False, exterior: bool = False) -> dict:
    with scene_lock(scene_dir):
        return _open_scene(scene_dir, clear_frost, dry_run, exterior)


def _open_scene(scene_dir: Path, clear_frost: bool, dry_run: bool, exterior: bool = False) -> dict:
    files = sorted(scene_dir.glob("*.xml"))
    text = files[0].read_text()
    doors = closed_doors(world_boxes(scene_dir, text), exterior)
    drop = {s for d in doors for s in d["shapes"]}
    frost = {sid for _, sid, _, _ in SHAPE.findall(text) if clear_frost and sid.startswith("GlassPartitionFrost")}
    drop |= frost
    summary = {"doors_opened": [{"casing": d["casing"][:60], "at": d["at"], "shapes": len(d["shapes"])} for d in doors],
               "frost_removed": len(frost), "dry_run": dry_run}
    if dry_run or not drop:
        return summary
    for f in files:
        t = f.read_text()
        new = SHAPE.sub(lambda m: "" if m.group(2) in drop else m.group(0), t)
        tmp = f.with_suffix(".xml.tmp")
        tmp.write_text(new)
        os.replace(tmp, f)
    kept = xml_refs(scene_dir) | json_refs(scene_dir)
    for ref in {r for r in re.findall(r'assets/[^"\\\s<>]+', text)} - kept:
        (scene_dir / ref).unlink(missing_ok=True)
    meta_path = scene_dir / "scene.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        meta["doors_opened"] = summary
        tmp = meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=1))
        os.replace(tmp, meta_path)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--keep-frost", action="store_true")
    ap.add_argument("--exterior", action="store_true", help="also open the doors in the building's outer wall")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    scenes = args.scene or sorted(p.parent.name for p in (args.pack / "scenes").glob("*/base.xml"))
    for name in scenes:
        s = open_scene((args.pack / "scenes" / name).resolve(), not args.keep_frost, args.dry_run, args.exterior)
        if s["doors_opened"] or s["frost_removed"]:
            print(f"{name}: {len(s['doors_opened'])} closed doors, {s['frost_removed']} frost bands"
                  f"{' [dry run]' if args.dry_run else ' removed'}: {[d['at'] for d in s['doors_opened']]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
