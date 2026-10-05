#!/usr/bin/env python3
"""Build a simulator scene pack from an OpticalNav project (robomituba side).

    python tools/build_pack.py --project /jarvis/project/robomituba/out/opticalnav/opticalnav-v0.2 \
        --out packs/opticalnav-v0.2 [--scene SCENE ...] [--link hard|copy]

Per scene the pack holds what the render server and R2R tooling read:

    connectivity/<scene>_connectivity.json   Matterport3D format, R2R frame (see frames.py)
    tasks/R2R/data/R2R_<split>.json          R2R annotations for every scene in the pack
    scenes/<scene>/scene.json                camera rig, variants, conventions, provenance
    scenes/<scene>/<variant>.xml             the production Stokes scene each variant was rendered from
    scenes/<scene>/assets/...                meshes and textures (hard links by default)

The Stokes XMLs are the files the dataset renderer actually loaded, found
through the current observation pointers' manifests; the newest render
version of each variant wins (the mirror re-render replaced only the views
that see a mirror, and those views were rendered from the newer scene).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opticalnav_sim import frames  # noqa: E402

FILENAME_RE = re.compile(r'(<string name="filename" value=")([^"]+)(")')
POINTER_DIRS = {"base": "observations", "perturbed": "observations_perturbed",
                "active_polar": "observations_perturbed_active_polar"}
SAMPLE = int(os.environ.get("OPTICALNAV_PACK_SAMPLE", "400"))
REFERENCES = 3


def pointer_manifests(scene_dir: Path, project: Path, variant: str):
    pointers = sorted((scene_dir / POINTER_DIRS[variant]).glob("*/*/current.json"))
    random.Random(0).shuffle(pointers)
    for ptr in pointers[:SAMPLE]:
        try:  # pointers and manifests can vanish while a cleanup runs
            payload = json.loads(ptr.read_text())
            yield payload, json.loads((project / payload["bundle_ref"] / "manifest.json").read_text())
        except (OSError, ValueError):
            continue


def staged_scene(manifest: dict) -> str:
    return (manifest.get("artifacts") or [{}])[0].get("timing", {}).get("scene")


def _flash_spec(light: dict) -> dict:
    return {"distance_m": light["distance_m"], "size_world": light["size_world"],
            "radiance": (light.get("extras") or {}).get("radiance"), "emitter": (light.get("extras") or {}).get("emitter"),
            "polarizer_angle_deg": light.get("polarizer_angle_deg", 0.0)}


def pick_variant_sources(scene_dir: Path, project: Path) -> tuple[dict, dict | None, dict]:
    """variant -> {source_xml, source_flash_xml?, flash?, protocol?, render_version_id}; also the camera
    spec and, per variant, up to REFERENCES sampled dataset bundles for parity checks."""
    out, refs, camera = {}, {}, None
    for variant in ("base", "perturbed"):
        rows = list(pointer_manifests(scene_dir, project, variant))
        if not rows:
            continue
        payload, manifest = max(rows, key=lambda r: r[0].get("updated_at", ""))
        camera = camera or manifest["camera_specs"][0]
        newest = [p for p, _ in rows if p["render_version_id"] == payload["render_version_id"]]
        refs[variant] = newest[:REFERENCES]
        out[variant] = {"source_xml": staged_scene(manifest), "render_version_id": payload["render_version_id"],
                        "staged_counts": Counter(staged_scene(m) for _, m in rows).most_common()}
    rows = list(pointer_manifests(scene_dir, project, "active_polar"))
    if rows:
        composed = [(p, m) for p, m in rows if (m.get("extras") or {}).get("active_polar_composition")]
        payload, manifest = max(composed or rows, key=lambda r: r[0].get("updated_at", ""))
        refs["active_polar"] = [p for p, _ in (composed or rows) if p["render_version_id"] == payload["render_version_id"]][:REFERENCES]
        if composed and "perturbed" in out:
            # active = passive (perturbed, path) + flash-only pass (path_nocaustics), summed as linear Stokes
            comp = manifest["extras"]["active_polar_composition"]
            flash_manifest = json.loads((project / comp["flash_bundle_ref"] / "manifest.json").read_text())
            out["active_polar"] = {
                "source_xml": out["perturbed"]["source_xml"], "source_flash_xml": staged_scene(flash_manifest),
                "render_version_id": payload["render_version_id"], "protocol": comp["protocol"],
                "flash": _flash_spec(flash_manifest["extras"]["assist_light"]), "passes": 2}
        elif not composed:
            # older protocol: one pass with the room lights and the camera-aligned flash together
            extras = manifest.get("extras") or {}
            light = extras.get("assist_light") or {}
            out["active_polar"] = {
                "source_xml": staged_scene(manifest), "render_version_id": payload["render_version_id"],
                "protocol": (light.get("extras") or {}).get("protocol") or extras.get("active_polar_protocol"),
                "flash": _flash_spec(light), "passes": 1}
    return out, camera, refs


def infer_sources(scene_dir: Path) -> dict:
    """No observation manifests left: match staged Stokes XMLs to their source scenes by object ids.

    The active scene is the one carrying the camera-aligned flash; base and perturbed are told apart
    by which authored scene (render_scene.xml / render_scene_perturbed.xml) shares the most object ids.
    """
    ids = lambda text: set(re.findall(r'<!--opticalnav-obj:\{"id":"([^"]+)"', text))  # noqa: E731
    staged = sorted((scene_dir / ".staged_mitsuba" / "base").glob("*.xml"), key=lambda p: p.stat().st_mtime)
    sources = {v: ids((scene_dir / f).read_text()) for v, f in (("base", "render_scene.xml"),
                                                                 ("perturbed", "render_scene_perturbed.xml"))
               if (scene_dir / f).is_file()}
    out = {}
    for path in staged:  # oldest first, so the newest match wins
        text = path.read_text()
        if "camera_assist_light" in text:
            continue
        own = ids(text)
        best = max(sources, key=lambda v: len(own & sources[v]) / max(1, len(own | sources[v])), default=None)
        if best:
            out[best] = {"source_xml": str(path), "inferred": "object-id match; observation manifests were deleted"}
    return out


def link(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hard":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def rewrite_xml(src_xml: Path, dst_xml: Path, repo_root: Path, assets: Path, mode: str) -> int:
    text = src_xml.read_text()
    count = 0

    def sub(match):
        nonlocal count
        raw = match.group(2)
        path = Path(raw)
        if not path.is_absolute():
            path = next((p for p in (src_xml.parent / raw, repo_root / raw) if p.is_file()), repo_root / raw)
        if not path.is_file():
            raise FileNotFoundError(f"{src_xml}: asset {raw} not found")
        name = f"{hashlib.sha1(str(path).encode()).hexdigest()[:12]}_{path.name}"
        link(path, assets / name, mode)
        count += 1
        return f"{match.group(1)}assets/{name}{match.group(3)}"

    out = FILENAME_RE.sub(sub, text)
    if '<sensor type="perspective" id="sensor">' not in out:
        raise ValueError(f"{src_xml}: expected a sensor with id='sensor' (production staging stamps it)")
    dst_xml.write_text(out)
    return count


def connectivity(graph: dict, eye_height: float) -> tuple[list[dict], dict[str, tuple[float, float, float]]]:
    poses = sorted(graph["poses"], key=lambda p: p["motion_node_id"])
    index = {p["motion_node_id"]: i for i, p in enumerate(poses)}
    adj = [[False] * len(poses) for _ in poses]
    for edge in graph["edges"]:
        i, j = index.get(edge["source"]), index.get(edge["target"])
        if i is not None and j is not None and i != j:
            adj[i][j] = adj[j][i] = True
    rows, pos = [], {}
    for i, p in enumerate(poses):
        x, y = frames.r2r_xy(p["position"][0], p["position"][1])
        pose = [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, y, 0.0, 0.0, 1.0, eye_height, 0.0, 0.0, 0.0, 1.0]
        rows.append({"image_id": p["motion_node_id"], "pose": pose, "included": True,
                     "unobstructed": adj[i], "height": eye_height})
        pos[p["motion_node_id"]] = (x, y, eye_height)
    return rows, pos


def r2r_entries(scene: str, scene_dir: Path, pos: dict, adjacency: set) -> tuple[dict[str, list], list[str]]:
    by_split, problems = {}, []
    for path in sorted(scene_dir.glob("episodes/*/*.json")):
        ep = json.loads(path.read_text())
        route = [n for k, n in enumerate(ep["path_nodes"]) if k == 0 or n != ep["path_nodes"][k - 1]]
        missing = [(a, b) for a, b in zip(route, route[1:]) if (a, b) not in adjacency]
        if missing:
            problems.append(f"{ep['episode_id']}: {len(missing)} path steps are not graph edges")
            continue
        texts = [ep.get("natural_language_instruction")] + [
            i.get("text") if isinstance(i, dict) else i for i in ep.get("instructions") or []]
        by_split.setdefault(ep["split"], []).append({
            "scan": scene,
            "path": route,
            "heading": frames.heading_from_yaw(float(ep["start_pose"][2])),
            "distance": sum(math.dist(pos[a], pos[b]) for a, b in zip(route, route[1:])),
            "instructions": [t for t in texts if t],
            "episode_id": ep["episode_id"],
            "goal_node": ep.get("goal_node"),
            "actions": len(ep.get("actions") or []),
        })
    return by_split, problems


def build_scene(scene: str, project: Path, repo_root: Path, out: Path, mode: str) -> dict:
    scene_dir = project / "scenes" / scene
    sources, camera, refs = pick_variant_sources(scene_dir, project)
    if not sources:
        sources = infer_sources(scene_dir)
    if not sources:
        raise RuntimeError(f"{scene}: no rendered observations or staged scenes to take the scene state from")
    camera = camera or DEFAULT_CAMERA
    missing = {v: s["source_xml"] for v, s in sources.items()
               if not Path(s["source_xml"]).is_file() or (s.get("source_flash_xml") and not Path(s["source_flash_xml"]).is_file())}
    for variant in missing:  # the staged file was cleaned up after rendering: that variant cannot be reproduced
        sources.pop(variant)
    graph = json.loads((scene_dir / "navigation_support_graph.json").read_text())
    mount = camera["extras"]["robot_mount"]
    eye = float(mount["xyz_m"][1])
    rows, pos = connectivity(graph, eye)
    (out / "connectivity").mkdir(parents=True, exist_ok=True)
    (out / "connectivity" / f"{scene}_connectivity.json").write_text(json.dumps(rows))

    dst = out / "scenes" / scene
    dst.mkdir(parents=True, exist_ok=True)
    variants, written = {}, {}
    for variant, spec in sources.items():
        entry = {k: v for k, v in spec.items() if not k.startswith("source")}
        for key, out_name in (("source_xml", f"{variant}.xml"), ("source_flash_xml", f"{variant}_flash.xml")):
            src = spec.get(key)
            if not src:
                continue
            if src not in written:  # active_polar's passive pass reuses perturbed.xml
                written[src] = (out_name, rewrite_xml(Path(src), dst / out_name, repo_root, dst / "assets", mode))
            entry[key.replace("source_", "")] = written[src][0]
            entry[key] = src
        variants[variant] = entry

    tasks, problems = scene_tasks(scene, project, out, mode)
    references = capture_references(refs, project, dst / "reference", mode)
    meta = {
        "scan": scene,
        "camera": {"sensor_id": camera["camera_id"], "width": camera["resolution"][0],
                   "height": camera["resolution"][1], "hfov_deg": camera["fov_deg"], "mount": mount,
                   "nominal_pitch_rad": frames.nominal_pitch(mount),
                   "dataset_spp": (camera["extras"].get("render") or {}).get("polar_spp")},
        "stokes": {"basis": "world_gravity_y_v1", "reference_axis": [0.0, 1.0, 0.0],
                   "components": ["s0", "s1", "s2", "s3"], "channels": "linear RGB per component"},
        "variants": variants,
        "variants_unavailable": {v: f"staged scene no longer on disk: {p}" for v, p in missing.items()},
        "reference_views": references,
        "graph": {"source": "navigation_support_graph.json", "nodes": len(rows),
                  "edges": sum(sum(r["unobstructed"]) for r in rows) // 2,
                  "heading_count": graph.get("heading_count"), "turn_deg": graph.get("turn_deg"),
                  "forward_step_m": graph.get("forward_step_m")},
        "episodes": {split: len(v) for split, v in tasks.items()},
        "episode_problems": problems,
        "frames": "connectivity/state frame: (x, -y_dataset, height), z-up; heading = pi - dataset_yaw",
    }
    (dst / "scene.json").write_text(json.dumps(meta, indent=1))
    print(f"[pack] {scene}: {len(rows)} nodes, {meta['graph']['edges']} edges, variants {sorted(variants)}, "
          f"assets {sum(n for _, n in written.values())} refs, episodes {meta['episodes']}"
          + (f", {len(problems)} episodes skipped" if problems else ""), flush=True)


DEFAULT_CAMERA = {"camera_id": "polar_cam", "resolution": [512, 384], "fov_deg": 90.0,
                  "extras": {"robot_mount": {"parent_frame": "base_link", "xyz_m": [0.0, 1.5, 0.1], "rpy_deg": [0.0, 0.0, 0.0]},
                             "render": {"polar_spp": 1024}}}


def capture_references(refs: dict, project: Path, dst: Path, mode: str) -> list[dict]:
    """Link a few dataset frames (manifest + Stokes NPZ + preview) per variant for tools/check_parity.py."""
    out = []
    for variant, pointers in refs.items():
        for ptr in pointers:
            bundle = project / ptr["bundle_ref"]
            node, heading = ptr["node_id"], ptr["heading_id"]
            target = dst / variant / f"{node}__{heading}"
            files = [bundle / "manifest.json", bundle / "cameras" / "polar_cam" / "stokes_data.npz",
                     bundle / "cameras" / "polar_cam" / "polar_rgb_preview.png"]
            if not all(f.is_file() for f in files):
                continue
            for f in files:
                link(f, target / f.name, mode)
            out.append({"variant": variant, "node_id": node, "heading_id": heading,
                        "render_version_id": ptr["render_version_id"],
                        "path": str(target.relative_to(dst.parent))})
    return out


def copy_navigation(scene: str, project: Path, out: Path, mode: str) -> int:
    """The scene's navigation support graph (states and transitions; episode generation needs it) and its
    original OpticalNav episodes (step-aligned path_nodes/path_headings; replay needs them) into the pack."""
    src, dst = project / "scenes" / scene, out / "scenes" / scene
    graph = src / "navigation_support_graph.json"
    if graph.is_file():
        link(graph, dst / "navigation_support_graph.json", mode)
    count = 0
    for path in sorted(src.glob("episodes/*/*.json")):
        link(path, dst / "episodes" / path.parent.name / path.name, mode)
        count += 1
    return count


def scene_tasks(scene: str, project: Path, out: Path, mode: str = "hard") -> tuple[dict[str, list], list[str]]:
    """R2R rows for one scene from the project's episodes and the pack's connectivity -> scenes/<scene>/r2r.json."""
    copy_navigation(scene, project, out, mode)
    rows = json.loads((out / "connectivity" / f"{scene}_connectivity.json").read_text())
    pos = {r["image_id"]: (r["pose"][3], r["pose"][7], r["pose"][11]) for r in rows}
    adjacency = {(rows[i]["image_id"], rows[j]["image_id"]) for i in range(len(rows))
                 for j, ok in enumerate(rows[i]["unobstructed"]) if ok}
    tasks, problems = r2r_entries(scene, project / "scenes" / scene, pos, adjacency)
    (out / "scenes" / scene / "r2r.json").write_text(json.dumps({"splits": tasks, "problems": problems}))
    return tasks, problems


def aggregate(out: Path, project: Path) -> dict[str, int]:
    """Pack-level files from every scene in the pack; path ids are deterministic (split, scene, episode)."""
    scans = sorted(p.parent.name for p in (out / "scenes").glob("*/scene.json"))
    tasks: dict[str, list] = {}
    for scene in scans:
        path = out / "scenes" / scene / "r2r.json"
        if not path.is_file():
            print(f"[aggregate] {scene}: no r2r.json yet (rebuild that scene or run --tasks-only)", flush=True)
            continue
        for split, rows in json.loads(path.read_text())["splits"].items():
            tasks.setdefault(split, []).extend(rows)
    data = out / "tasks" / "R2R" / "data"
    data.mkdir(parents=True, exist_ok=True)
    path_id = 0
    for split in sorted(tasks):
        tasks[split].sort(key=lambda r: (r["scan"], r["episode_id"]))
        for row in tasks[split]:
            row["path_id"] = path_id
            path_id += 1
        (data / f"R2R_{split}.json").write_text(json.dumps(tasks[split]))
    (out / "connectivity" / "scans.txt").write_text("\n".join(scans) + "\n")
    counts = {s: len(r) for s, r in tasks.items()}
    (out / "pack.json").write_text(json.dumps({
        "built_at": datetime.now(timezone.utc).isoformat(), "project": str(project), "scans": scans,
        "splits": counts}, indent=1))
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, type=Path)
    ap.add_argument("--repo-root", type=Path, default=None, help="robomituba root (relative XML paths); "
                    "default: three levels above --project")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--scene", action="append", default=[], help="scene id (repeatable; default: dataset.json)")
    ap.add_argument("--link", choices=["hard", "copy"], default="hard")
    ap.add_argument("--tasks-only", action="store_true",
                    help="only rebuild R2R tasks for scenes already in the pack (no observation manifests needed)")
    args = ap.parse_args()
    project = args.project.resolve()
    repo_root = (args.repo_root or project.parents[2]).resolve()
    scenes = args.scene or (
        sorted(p.parent.name for p in (args.out.resolve() / "scenes").glob("*/scene.json")) if args.tasks_only
        else [s["scene_id"] if isinstance(s, dict) else s for s in json.loads((project / "dataset.json").read_text())["scenes"]])
    out = args.out.resolve()
    failed = []
    for scene in scenes:
        try:
            if args.tasks_only:
                scene_tasks(scene, project, out, args.link)
            else:
                build_scene(scene, project, repo_root, out, args.link)
        except Exception as exc:  # noqa: BLE001 - one broken scene must not sink the pack
            failed.append(scene)
            print(f"[skip] {scene}: {type(exc).__name__}: {exc}", flush=True)
    counts = aggregate(out, project)
    print(f"[pack] {len(scenes) - len(failed)}/{len(scenes)} scenes updated -> {out}; pack splits {counts}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
