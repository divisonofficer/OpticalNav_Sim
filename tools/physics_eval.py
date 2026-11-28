#!/usr/bin/env python3
"""Score physics_objects' rules against verdicts reviewed by eye (tools/physics_review.py's sheets, verdicts.json).

    python3 tools/physics_eval.py --pack packs/blend-cache --pack packs/opticalnav-sim-v1 [--review DIR] [--scene S ...] [--show 20]

A reviewed verdict (DIR/<scene>/verdicts.json: per object of the physics.json it was made from, ``should``: PUSH, MOVE,
FIX or UNSURE, and ``grouping``) is a truth for each of that object's parts (shape ids): a new rule set is scored part by
part, so a change in how shapes are grouped into objects is scored too. Objects reviewed UNSURE are left out; so are the
ones whose grouping was wrong (merged or partial: their parts' truth is the reviewer's for the whole, often not each
part's), unless --with-grouping. The analysis runs now (physics_objects.analyse: the rules as the code has them, the
overrides set by hand left out), nothing is written. Prints per scene and in all: agreement, the confusion matrix, and
the parts the rules get wrong (``--show``).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CLASSES = ("PUSH", "MOVE", "FIX")


def verdict(o: dict) -> str:
    return "PUSH" if o["movable"] and o["robot_reachable"] else "MOVE" if o["movable"] else "FIX"


def truth_by_part(scene_dir: Path, review: Path, with_grouping: bool) -> dict:
    """Part id -> (truth, object id reviewed, problem): from docs/physics_truth/<scene>.json (made once from a review,
    per part: it holds when physics.json is made again), else from verdicts.json and the physics.json it was made from."""
    kept = REPO / "docs" / "physics_truth" / f"{scene_dir.name}.json"
    if kept.is_file():
        return {p: (t["should"], t["object"], t.get("problem", "")) for p, t in json.loads(kept.read_text())["parts"].items()
                if t["should"] in CLASSES and (with_grouping or t.get("grouping", "ok") in ("ok", "not_drawn"))}
    rows = json.loads((review / "verdicts.json").read_text())
    by_n = {int(r["n"]): r for r in rows}
    phys = json.loads((scene_dir / "physics.json").read_text())
    out = {}
    for n, o in enumerate(phys["objects"]):
        r = by_n.get(n)
        if r is None or r.get("should") not in CLASSES:
            continue
        if r.get("grouping", "ok") not in ("ok", "not_drawn") and not with_grouping:
            continue
        for p in o["parts"]:
            out[p] = (r["should"], o["id"], r.get("problem", ""))
    return out


def score(scene_dir: Path, review: Path, with_grouping: bool = False) -> dict:
    from opticalnav_sim import physics_objects
    truth = truth_by_part(scene_dir, review, with_grouping)
    meta = json.loads((scene_dir / "scene.json").read_text())
    xml = ((meta.get("variants") or {}).get("base") or {}).get("xml", "base.xml")
    result = physics_objects.analyse(scene_dir, xml)
    got = {}
    for o in result["objects"]:   # a piece of a split shape (``<shape>__p<k>``) scored against its shape's review
        for p in o["parts"]:
            got.setdefault(p, (verdict(o), o))
            base = re.sub(r"__p\d+$", "", p)
            if base != p and base not in got:
                got[base] = (verdict(o), o)
    confusion, wrong, votes = Counter(), [], {}
    for p, (t, oid, problem) in truth.items():
        if p in got:
            votes.setdefault(oid, (t, Counter()))[1][got[p][0]] += 1
    objects = Counter((t, c.most_common(1)[0][0]) for t, c in votes.values())   # per reviewed object: its parts' majority
    for p, (t, oid, problem) in truth.items():
        if p not in got:
            continue
        g, o = got[p]
        confusion[(t, g)] += 1
        if g != t:
            wrong.append({"part": p, "truth": t, "got": g, "object": o["id"], "label": o["label"], "reason": o["reason"],
                          "reviewed_problem": problem})
    # grouping: of the objects reviewed "partial" (a piece of a thing), how many are now in one object with other
    # parts; and how many new objects join two objects reviewed sound (ok) and apart (a join too many)
    kept = REPO / "docs" / "physics_truth" / f"{scene_dir.name}.json"
    graded = json.loads(kept.read_text())["parts"] if kept.is_file() else {}
    old_obj = {p: t["object"] for p, t in graded.items()}
    partial = {t["object"] for t in graded.values() if t.get("grouping") == "partial"}
    sound = {t["object"] for t in graded.values() if t.get("grouping") == "ok"}
    joined, bad = set(), []
    for o in result["objects"]:
        olds = {old_obj[re.sub(r"__p\d+$", "", p)] for p in o["parts"] if re.sub(r"__p\d+$", "", p) in old_obj}
        for x in olds & partial:
            if len(olds) > 1:
                joined.add(x)
        if len(olds & sound) > 1:
            bad.append((o["label"], sorted(olds & sound)[:4]))
    grouping = {"partial": len(partial), "partial_joined": len(joined), "bad_joins": len(bad), "bad": bad[:8]}
    return {"scene": scene_dir.name, "parts": sum(confusion.values()), "grouping": grouping, "agree": sum(v for (t, g), v in confusion.items() if t == g),
            "confusion": confusion, "objects": objects, "wrong": wrong, "seconds": result["counts"]["seconds"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review", type=Path, default=Path("/nonexistent"), help="a review's folder (verdicts.json per scene); "
                    "docs/physics_truth/ is read first")
    ap.add_argument("--pack", type=Path, action="append", required=True)
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--show", type=int, default=15, help="wrong parts listed per scene (distinct objects)")
    ap.add_argument("--with-grouping", action="store_true")
    ap.add_argument("--json", type=Path, help="write the scores here")
    args = ap.parse_args()
    total, total_o, results = Counter(), Counter(), []
    for pack in args.pack:
        for d in sorted((pack / "scenes").glob("*/physics.json")):
            scene = d.parent
            reviewed = (args.review / scene.name / "verdicts.json").is_file() or \
                (REPO / "docs" / "physics_truth" / f"{scene.name}.json").is_file()
            if (args.scene and scene.name not in args.scene) or not reviewed:
                continue
            s = score(scene, args.review / scene.name, args.with_grouping)
            results.append(s)
            total.update(s["confusion"])
            total_o.update(s["objects"])
            no = sum(s["objects"].values())
            ao = sum(v for (t, g), v in s["objects"].items() if t == g)
            print(f"\n{scene.name}: objects {ao}/{no} agree ({100 * ao / max(no, 1):.1f}%), parts {s['agree']}/{s['parts']} "
                  f"({100 * s['agree'] / max(s['parts'], 1):.1f}%), analysis {s['seconds']} s; grouping: "
                  f"{s['grouping']['partial_joined']}/{s['grouping']['partial']} partial joined, {s['grouping']['bad_joins']} bad joins")
            for lab, olds in s["grouping"]["bad"][: args.show]:
                print(f"  BAD JOIN {lab[:40]}: {', '.join(x[:30] for x in olds)}")
            seen = set()
            for w in s["wrong"]:
                if w["object"] in seen or len(seen) >= args.show:
                    continue
                seen.add(w["object"])
                print(f"  {w['truth']:4s} <- {w['got']:4s}  {w['label'][:40]:40s} {w['reason'][:70]}")
    for name, tot in (("objects", total_o), ("parts", total)):
        n = sum(tot.values())
        ok = sum(v for (t, g), v in tot.items() if t == g)
        print(f"\nall {name}: {ok}/{n} agree ({100 * ok / max(n, 1):.1f}%)")
        print("truth \\ rules " + " ".join(f"{c:>6s}" for c in CLASSES))
        for t in CLASSES:
            print(f"{t:13s} " + " ".join(f"{tot[(t, g)]:6d}" for g in CLASSES))
    if args.json:
        args.json.write_text(json.dumps([{**r, "confusion": {f"{t}>{g}": v for (t, g), v in r["confusion"].items()},
                                          "objects": {f"{t}>{g}": v for (t, g), v in r["objects"].items()}}
                                         for r in results], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
