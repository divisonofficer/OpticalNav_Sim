#!/usr/bin/env python3
"""What each light of a pack scene glows from, before anyone sees a square on a ceiling.

    python3 tools/light_audit.py --pack packs/opticalnav-sim-v1 [--pack ...] [--scene SCENE ...] [--strict]

robomituba's importer turns every Infinigen light into a box (a 0.6 m square under a ceiling light, a flat
cube on a desk lamp); the server's lamp_lights_text moves each box's power onto the lamp it belongs to
(server.pair_lamp_lights: by name, then by position). Per scene this prints how many lights paired by name,
how many by the importer's own tag (``metadata.fixture_id``), how many by position (a fixture whose name no rule knows: a new kit's), and the orphans: lights with no
fixture mesh near them, which the server turns into round downlights (ORPHAN_DOWNLIGHT_R). It also counts
the room softboxes, which camera rays pass through (server.FILL_PREFIX, path_nocaustics ``hidden``).

An orphan usually means the importer placed a light by no mesh, or a kit's fixture lies more than 0.1 m
beside or 0.6 m above or below its light. ``--strict`` exits 1 when a scene has one the importer never looked at (no ``fixture_id`` key: an older blend; a light tagged null, found no fixture mesh by the importer's own pass, is reported but passes); tools/build_pack.py
prints the same line for every scene it builds.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opticalnav_sim import server  # noqa: E402


def audit_scene(scene_dir: Path) -> dict:
    """Light pairing of a scene's base.xml: counts, and the orphans' ids."""
    text = (Path(scene_dir) / "base.xml").read_text()
    notes = []
    for note in re.findall(r"<!--opticalnav-obj:(\{.*?\})-->", text):
        try:
            notes.append(json.loads(note))
        except ValueError:
            continue
    shapes = {m.group(2): m for m in re.finditer(r'\n  <shape type="([a-z_]+)" id="([^"]+)">(.*?)\n  </shape>', text, re.S)}
    _, kinds, paired, orphans = server.pair_lamp_lights(notes or server.notes_from_shapes(shapes))
    tagged = sorted(k for k, light in paired.items() if (light.get("metadata") or {}).get("fixture_id") == k)
    by_position = sorted(k for k in paired if kinds[k] == server.FIXTURE and k not in tagged)
    def key(o, name):
        return (o.get("metadata") or {}).get(name, "absent")

    # an orphan the importer tagged with a fixture shares it with other lights (a merged track): a light at its
    # own place by design; tagged null, it has no fixture mesh; untagged, nobody looked (an older blend)
    shared = sorted(o["id"] for o in orphans if key(o, "fixture_id") not in ("absent", None))
    orphans = [o for o in orphans if o["id"] not in shared]
    return {"scene": Path(scene_dir).name, "lights": len(paired) + len(orphans) + len(shared), "shared": len(shared),
            "by_tag": len(tagged), "by_name": len(paired) - len(by_position) - len(tagged),
            "by_position": len(by_position),
            "orphans": sorted(o["id"] for o in orphans),
            "untagged": sorted(o["id"] for o in orphans if key(o, "fixture_id") == "absent"),
            "softboxes": sum(sid.startswith(server.FILL_PREFIX) for sid in shapes)}


def line(row: dict) -> str:
    orphans = f" orphans {len(row['orphans'])}: {', '.join(row['orphans'])}" if row["orphans"] else ""
    return (f"[lights] {row['scene']}: {row['lights']} lights, {row['by_tag']} by the importer's tag, {row['by_name']} by "
            f"name, {row['by_position']} by position, "
            f"{row['shared']} on a fixture shared with other lights; {row['softboxes']} softboxes hidden from the camera{orphans}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", action="append", required=True, type=Path)
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--strict", action="store_true", help="exit 1 if a scene has a light by no fixture")
    args = ap.parse_args()
    bad = 0
    for pack in args.pack:
        for scene_dir in sorted(p.parent for p in (pack / "scenes").glob("*/base.xml")):
            if args.scene and scene_dir.name not in args.scene:
                continue
            row = audit_scene(scene_dir)
            bad += bool(row["untagged"])
            print(line(row), flush=True)
    return 1 if args.strict and bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
