#!/usr/bin/env python3
"""Decide, per scene and without clicking, which objects a robot could push: ``physics.json`` in the scene dir.

    python3.10 tools/physics_tag.py --pack packs/blend-cache [--scene SCENE ...] [--if-needed] [--dry-run] [--list]

opticalnav_sim/physics_objects.py does the work (read the scene object by object, group shapes into objects, measure
them, classify them by rules, a convex hull for each movable one); this tool runs it over a pack's scenes and writes the
result. physics.json carries the rules and thresholds it used, per object its part (shape) ids, label, ``movable``,
``robot_reachable``, confidence, the reason, mass, centre of mass, box, support, and for a movable one a pose, a hull and a
friction, plus ``movable`` (object ids) and ``movable_parts`` (shape ids). tools/merge_meshes.py makes the movable
objects its scope keeps (``keep`` here or scene.json ``physics.keep``: "reachable", the default, "movable" or "none") one
body each, ``dyn_<object>_<n>`` meshes merged per material, and records under ``merge`` which shapes hold each movable
object, so a physics engine can move them as bodies the way the robot's are.

Order: tag, then merge. A merged scene is tagged from its manifest (the parts as they were, merged_meshes.original_text);
merge_meshes.needs_merging sees a changed physics.json (its plan changes) and the merge unmerges and merges again. From
Python (the render server's "on open", not wired in):

    import physics_tag, merge_meshes                       # tools/ on sys.path
    physics_tag.tag_pack(pack, [scene], if_needed=True)     # a text check when up to date
    merge_meshes.merge_pack(pack, [scene], if_needed=True)

``--if-needed`` (needs_tagging): physics.json missing, of another version or rules, or the scene file (as converted, before
any merge) changed since. No Mitsuba: numpy/scipy on the PLY files, one core. Measured: blend_REF_CS02 (4,960 shapes, merged)
15 s, blend_REF_PH46 22 s, blend_REF_PH43 (1,756 shapes, 300 shrub instances: 76 M triangles before thinning) 46 s,
the opticalnav-sim-v1 Infinigen scenes 4 to 10 s; a scene up to date costs the text check (~1 s on CS02).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# numpy's BLAS threads: many small products, and with 32 threads waiting on each one PH46 took 6 min of system time for 25 s
# of work (the same 25 s with one), on a machine whose GPU process needs its cores. Set before numpy is imported.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compact_assets import SceneBusy, scene_lock  # noqa: E402
from opticalnav_sim import physics_objects  # noqa: E402
from opticalnav_sim.physics_objects import PHYSICS, VERSION  # noqa: E402


def read_physics(scene_dir) -> dict | None:
    path = Path(scene_dir) / PHYSICS
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None


def needs_tagging(scene_dir, xml: str = "base.xml") -> bool:
    """Whether physics.json is missing or stale: another version or rules, or the scene file (as converted, before any
    merge) changed. A text check (the scene file and the merge manifest are hashed, no mesh is read)."""
    scene_dir = Path(scene_dir)
    if not (scene_dir / xml).is_file():
        return False
    data = read_physics(scene_dir)
    if data is None or data.get("version") != VERSION or data.get("rules_sha1") != physics_objects.rules_sha1():
        return True
    return data.get("source_sha1") != physics_objects.source_sha1(scene_dir, xml)


def _write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")   # replaced, never written through a hard link
    tmp.write_text(text)
    os.replace(tmp, path)


def tag_scene(scene_dir, xml: str = "base.xml", dry_run: bool = False, show: bool = False) -> dict:
    """Analyse a scene and write its physics.json (not with ``dry_run``). Returns the counts. Holds the scene's lock
    (compact_assets.scene_lock) so a merge does not rewrite the files under it."""
    scene_dir = Path(scene_dir)
    with scene_lock(scene_dir):
        result = physics_objects.analyse(scene_dir, xml)
        # what is not the analysis carries over: the scope a person set ("keep") and what the last merge recorded
        # ("merge": a new movable list changes the merge plan, and that merge writes it again)
        for key in ("keep", "merge"):
            if key in (read_physics(scene_dir) or {}):
                result[key] = read_physics(scene_dir)[key]
        if not dry_run:
            result["at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            _write(scene_dir / PHYSICS, json.dumps(result, separators=(",", ":")))
    c = result["counts"]
    print(f"{scene_dir.name}{' (dry run)' if dry_run else ''}: {c['parts']} shapes -> {c['objects']} objects, "
          f"{c['movable']} movable ({c['robot_reachable']} at the robot's height), {c['seconds']} s", flush=True)
    if show:
        for o in result["objects"]:
            if o["movable"]:
                print(f"  {'R' if o['robot_reachable'] else ' '} {o['label'][:50]:50s} {o['mass_kg']:7.2f} kg  {o['reason']}")
    return c


def tag_pack(pack, scenes=(), if_needed: bool = True, dry_run: bool = False, show: bool = False) -> dict:
    """Tag the scenes of a pack (default: all). Scene id -> counts, None for a scene up to date, "busy" for one another
    process is rewriting."""
    pack = Path(pack)
    scenes = list(scenes) or sorted(p.parent.name for p in (pack / "scenes").glob("*/scene.json"))
    out = {}
    for scene in scenes:
        scene_dir = (pack / "scenes" / scene).resolve()
        if if_needed and not needs_tagging(scene_dir):
            out[scene] = None
            continue
        try:
            out[scene] = tag_scene(scene_dir, dry_run=dry_run, show=show)
        except SceneBusy as exc:
            print(f"skipped: {exc}", flush=True)
            out[scene] = "busy"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--scene", action="append", default=[], help="scene id (repeatable; default: every scene)")
    ap.add_argument("--if-needed", action="store_true", help="skip scenes whose physics.json is up to date (a text check)")
    ap.add_argument("--dry-run", action="store_true", help="analyse and print, write nothing")
    ap.add_argument("--list", action="store_true", help="print the movable objects with their mass and reason")
    args = ap.parse_args()
    done = tag_pack(args.pack, args.scene, args.if_needed, args.dry_run, args.list)
    skipped = [s for s, r in done.items() if r is None]
    if skipped:
        print(f"nothing to do: {len(skipped)} scene(s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
