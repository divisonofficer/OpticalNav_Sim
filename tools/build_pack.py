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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import light_audit  # noqa: E402

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
    # robomituba tools/infinigen/export_room_map.py: room polygons + door/open markers for the bird's-eye map, dataset (x, y) frame
    if (scene_dir / "room_map.json").is_file():
        shutil.copyfile(scene_dir / "room_map.json", dst / "room_map.json")
    print(f"[pack] {scene}: {len(rows)} nodes, {meta['graph']['edges']} edges, variants {sorted(variants)}, "
          f"assets {sum(n for _, n in written.values())} refs, episodes {meta['episodes']}"
          + (f", {len(problems)} episodes skipped" if problems else ""), flush=True)


KEEP_OWN_BSDF = ("mirror_auto_", "glass_auto_", "pool_")  # the server hides these per shape


def share_bsdfs(scene_dst: Path) -> dict:
    """Make every variant XML of a pack scene share identical BSDFs (fewer material instances to trace).

    (a) byte-identical asset files collapse to one name (the duplicates are unlinked from the pack);
    (b) top-level <bsdf>/<texture> nobody references are dropped;
    (c) each distinct inline shape BSDF (hashed on its canonical XML after (a), twosided wrapper included) is
        hoisted to one top-level <bsdf id="sb_<hash>"> and the inline copies become <ref id>.
    Shapes with an emitter and the hideable mirror/glass/pool shapes keep their own BSDF; shape ids and the
    opticalnav-obj comments are kept. Files are rewritten through os.replace, so hard links into another
    pack or the project are never modified."""
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    xmls = sorted(p for p in scene_dst.glob("*.xml") if not p.name.startswith("_"))
    assets = scene_dst / "assets"
    digest = {p.name: hashlib.sha1(p.read_bytes()).hexdigest() for p in assets.iterdir() if p.is_file()}
    keep_name: dict[str, str] = {}
    for name in sorted(digest):
        keep_name.setdefault(digest[name], name)
    alias = {n: keep_name[h] for n, h in digest.items()}

    def canon(el) -> str:
        attrs = sorted((k, v) for k, v in el.attrib.items() if k != "id")
        return f"<{el.tag}{attrs}>" + "".join(canon(c) for c in el if isinstance(c.tag, str)) + f"</{el.tag}>"

    stats = {"files": len(xmls), "asset_files": len(digest), "asset_contents": len(keep_name)}
    for path in xmls:
        root = ET.parse(path, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))).getroot()
        for el in root.iter():  # (a)
            if el.tag == "string" and el.get("name") == "filename" and el.get("value", "").startswith("assets/"):
                el.set("value", "assets/" + alias.get(el.get("value")[7:], el.get("value")[7:]))
        refs = {r.get("id") for r in root.iter("ref")}
        dropped = [c for c in list(root) if c.tag in ("bsdf", "texture") and c.get("id") not in refs]
        for c in dropped:  # (b)
            root.remove(c)
        hoisted, shared_ids, inline_before = {}, set(), 0
        for shape in root.findall("./shape"):  # (c)
            inner = shape.findall("./bsdf")
            inline_before += len(inner)
            if not inner or shape.find("./emitter") is not None or str(shape.get("id", "")).startswith(KEEP_OWN_BSDF):
                continue
            for b in inner:
                key = "sb_" + hashlib.sha1(canon(b).encode()).hexdigest()[:16]
                if key not in hoisted:
                    b.set("id", key)
                    hoisted[key] = b
                idx = list(shape).index(b)
                shape.remove(b)
                shape.insert(idx, ET.Element("ref", {"id": key}))
                shared_ids.add(key)
        first_shape = next((i for i, c in enumerate(root) if c.tag == "shape"), len(root))
        for k, b in enumerate(hoisted.values()):
            root.insert(first_shape + k, b)
        ET.indent(root, space="  ")
        tmp = path.with_name(f"_{path.name}.tmp")
        tmp.write_text(ET.tostring(root, encoding="unicode"))
        os.replace(tmp, path)
        stats[path.name] = {"unreferenced_dropped": len(dropped), "inline_bsdfs_before": inline_before,
                            "shared_bsdfs": len(hoisted)}
    used = {m.group(1) for p in xmls for m in re.finditer(r'value="assets/([^"]+)"', p.read_text())}
    removed = [n for n in digest if n not in used]
    for n in removed:
        (assets / n).unlink()
    stats["asset_files_removed"] = len(removed)
    return stats


def spawn_candidates(scene_dir: Path, eye_height: float, count: int = 3, step: int = 7) -> list[list[dict]]:
    """Start poses for a graph-less scene, per traversable region (the ``count`` largest): grid points ordered by
    their distance from every non-structure object's footprint (farthest first), at camera height, in the
    connectivity frame (x, -y_dataset, z). The first of each list is the default spawn (``spawn_points``);
    tools/check_spawns.py takes the next when a render shows the first inside a wall.
    ponytail: rectangle footprints from the authoring map; a free-space grid would be exact."""
    authoring = json.loads((scene_dir / "authoring_map.json").read_text())
    rooms = sorted((r["geometry"]["bounds"] for r in authoring.get("regions") or []
                    if r.get("type") == "traversable" and (r.get("geometry") or {}).get("bounds")),
                   key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    boxes = []
    for o in authoring.get("objects") or []:
        g, md = o.get("geometry") or {}, o.get("metadata") or {}
        c, s = g.get("center"), g.get("size_m")
        if md.get("kind") == "structure" or not (isinstance(c, list) and isinstance(s, list) and len(c) >= 2 and len(s) >= 3):
            continue
        boxes.append((c[0] - s[0] / 2, c[1] - s[2] / 2, c[0] + s[0] / 2, c[1] + s[2] / 2))

    def clearance(x: float, y: float) -> float:
        return min((math.hypot(max(b[0] - x, 0, x - b[2]), max(b[1] - y, 0, y - b[3])) for b in boxes), default=9.9)

    out = []
    for x0, y0, x1, y1 in rooms[:count]:
        pts = [(x0 + (x1 - x0) * (i + 0.5) / step, y0 + (y1 - y0) * (j + 0.5) / step) for i in range(step) for j in range(step)]
        pts.sort(key=lambda p: -clearance(*p))
        out.append([{"position": [round(x, 3), round(-y, 3), round(eye_height, 3)], "heading_deg": 0.0,
                     "clearance_m": round(clearance(x, y), 2)} for x, y in pts])
    return out


def spawn_points(scene_dir: Path, eye_height: float, count: int = 3, step: int = 7) -> list[dict]:
    """The default start poses: in each of the largest traversable regions, the grid point farthest from every
    non-structure object's footprint (see spawn_candidates)."""
    return [c[0] for c in spawn_candidates(scene_dir, eye_height, count, step)]


def tag_fixtures(base_xml: Path, authoring_map: Path) -> int:
    """Copy ``metadata.fixture_id`` and ``metadata.light`` of the authoring map's lights into the scene file's opticalnav-obj notes.
    apps/import_infinigen_scene.py tags each light with the fixture mesh that holds it, but robomituba's scene
    file writer leaves the note's metadata out, and the server reads only the scene file. Returns the number of
    notes tagged; a map without tags (an older blend) changes nothing."""
    if not authoring_map.is_file():
        return 0
    keep = ("fixture_id", "light")  # the importer's tag, and the Blender light's own parameters (metadata.light)
    tags = {o["id"]: {k: o["metadata"][k] for k in keep if k in o["metadata"]}
            for o in json.loads(authoring_map.read_text()).get("objects") or []
            if o.get("is_emitter") and any(k in (o.get("metadata") or {}) for k in keep)}
    if not tags:
        return 0
    count = 0

    def note(m):
        nonlocal count
        try:
            obj = json.loads(m.group(1))
        except ValueError:
            return m.group(0)
        if obj.get("id") not in tags:
            return m.group(0)
        obj["metadata"] = {**(obj.get("metadata") or {}), **tags[obj["id"]]}
        count += 1
        return f"<!--opticalnav-obj:{json.dumps(obj)}-->"

    text = base_xml.read_text()
    new = re.sub(r"<!--opticalnav-obj:(\{.*?\})-->", note, text)
    if new != text:
        tmp = base_xml.with_name(f"_{base_xml.name}.tmp")
        tmp.write_text(new)
        os.replace(tmp, base_xml)
    return count


def build_staged_scene(scene: str, project: Path, repo_root: Path, out: Path, mode: str) -> dict:
    """A scene staged without rendering (robomituba apps/stage_sim_scene.py writes sim_stage.json): base
    variant only, no graph, episodes or reference frames. Mirrors come from tools/build_perturbation_pool.py."""
    scene_dir = project / "scenes" / scene
    stage = json.loads((scene_dir / "sim_stage.json").read_text())
    src = repo_root / stage["stokes_xml"]
    dst = out / "scenes" / scene
    dst.mkdir(parents=True, exist_ok=True)
    refs = rewrite_xml(src, dst / "base.xml", repo_root, dst / "assets", mode)
    tagged = tag_fixtures(dst / "base.xml", scene_dir / "authoring_map.json")
    if tagged:
        print(f"[tags] {scene}: {tagged} lights carry the importer's fixture_id", flush=True)
    if '<bsdf type="measured_polarized' in (dst / "base.xml").read_text():
        raise ValueError(f"{scene}: measured_polarized BSDF in a staged simulator scene")
    camera, mount = DEFAULT_CAMERA, DEFAULT_CAMERA["extras"]["robot_mount"]
    meta = {
        "scan": scene,
        "camera": {"sensor_id": camera["camera_id"], "width": camera["resolution"][0],
                   "height": camera["resolution"][1], "hfov_deg": camera["fov_deg"], "mount": mount,
                   "nominal_pitch_rad": frames.nominal_pitch(mount), "dataset_spp": None},
        "stokes": {"basis": "world_gravity_y_v1", "reference_axis": stage["stokes_reference_axis"],
                   "components": ["s0", "s1", "s2", "s3"], "channels": "linear RGB per component"},
        "variants": {"base": {"xml": "base.xml", "source_xml": str(src), "protocol": "staged_without_render_v1",
                              "texture_max_resolution": stage.get("texture_max_resolution")}},
        "variants_unavailable": {},
        "reference_views": [],
        "graph": None,
        "episodes": {},
        "episode_problems": [],
        "spawn": spawn_points(scene_dir, float(mount["xyz_m"][1])),
        "staged": stage,
        "frames": "connectivity/state frame: (x, -y_dataset, height), z-up; heading = pi - dataset_yaw",
    }
    (dst / "scene.json").write_text(json.dumps(meta, indent=1))
    # robomituba tools/infinigen/export_room_map.py: room polygons + door/open markers for the bird's-eye map, dataset (x, y) frame
    if (scene_dir / "room_map.json").is_file():
        shutil.copyfile(scene_dir / "room_map.json", dst / "room_map.json")
    print(f"[pack] {scene}: staged base only, {refs} asset refs, no graph or episodes", flush=True)
    return meta


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
    (out / "connectivity").mkdir(parents=True, exist_ok=True)  # a pack of graph-less staged scenes has none yet
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
    ap.add_argument("--share-bsdfs", action="store_true",
                    help="after packing, make each scene's XMLs share identical BSDFs (see share_bsdfs)")
    args = ap.parse_args()
    project = args.project.resolve()
    repo_root = (args.repo_root or project.parents[2]).resolve()
    scenes = args.scene or (
        sorted(p.parent.name for p in (args.out.resolve() / "scenes").glob("*/scene.json")) if args.tasks_only
        else [s["scene_id"] if isinstance(s, dict) else s for s in json.loads((project / "dataset.json").read_text())["scenes"]]
        if (project / "dataset.json").is_file()
        else sorted(p.parent.name for p in (project / "scenes").glob("*/sim_stage.json")))
    out = args.out.resolve()
    failed = []
    for scene in scenes:
        try:
            if (project / "scenes" / scene / "sim_stage.json").is_file():  # staged without rendering
                build_staged_scene(scene, project, repo_root, out, args.link)
            elif args.tasks_only:
                scene_tasks(scene, project, out, args.link)
            else:
                build_scene(scene, project, repo_root, out, args.link)
            if args.share_bsdfs and not args.tasks_only:
                print(f"[share] {scene}: {json.dumps(share_bsdfs(out / 'scenes' / scene))}", flush=True)
            if not args.tasks_only and (out / "scenes" / scene / "base.xml").is_file():  # lights by no fixture
                print(light_audit.line(light_audit.audit_scene(out / "scenes" / scene)), flush=True)
        except Exception as exc:  # noqa: BLE001 - one broken scene must not sink the pack
            failed.append(scene)
            print(f"[skip] {scene}: {type(exc).__name__}: {exc}", flush=True)
    counts = aggregate(out, project)
    print(f"[pack] {len(scenes) - len(failed)}/{len(scenes)} scenes updated -> {out}; pack splits {counts}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
