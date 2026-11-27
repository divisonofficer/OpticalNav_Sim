#!/usr/bin/env python3
"""Merge a pack scene's static meshes of one material into one mesh each, in the scene files themselves.

    PYTHONPATH=<mitsuba build>/python python3.10 tools/merge_meshes.py --pack packs/blend-cache [--scene SCENE ...]
        [--if-needed] [--dry-run] [--keep REGEX ...] [--clear-keep] [--unmerge]

Why. A scene converted from its .blend (tools/blend_to_pack.py) writes one shape per object and material slot: blend_REF_CS02's
base.xml holds 4,964 shapes (screw holes, dial ticks, leaves) over 43 distinct BSDFs. Mitsuba merges them at load (MergeShape)
into a couple of hundred meshes, but each of the 4,964 source shapes took a Dr.Jit registry id first, so the live meshes carry
ids up to ~5,000. With a robot in the scene (instances, shape groups) every nested Shape call of a frozen render carries an
offset table as long as the largest Shape id: ~285 call sites x 5,075 entries rebuilt on every replay, 11 ms a frame. Merged
in the pack, the file names a few hundred shapes and the ids stay small. CS02: 4,964 shapes -> 215 (46 merged meshes of 4,795
parts) in 1.3 s, 25 MB of new PLY; scalar_rgb loads it in 0.12 s instead of 0.81 s, to the same 445,864 triangles.

What merges: a top-level ``ply`` or ``obj`` shape that
  * is no light: no <emitter>, its note's ``is_emitter`` false, not in server.light_groups, no lamp part that
    server.lamp_lights_text may light (server.pair_lamp_lights), no id under a prefix the server looks shapes up by
    (PROTECTED_PREFIXES: lights, the camera flash, the dataset's mirrors and glass, robot bodies);
  * nothing names: no <ref id=...> in a scene file, no string of a JSON file of the scene dir (scene.json,
    perturbation_pool.json, room_map.json, ...), no ``metadata`` value of another shape's note (a light's fixture_id);
  * is the same, byte for byte with its note, in every scene file of the directory (base.xml, perturbed.xml,
    glassmirror.xml, active_polar*.xml): a variant stays "base plus extra shapes" for server.extra_shapes;
  * may not move by hand: its note has no ``"dynamic": true``, and neither scene.json's ``physics.dynamic`` regexes nor
    ``--keep`` match its id or its label (re.search; the label is the Blender object, so one regex keeps all its parts);
  * carries only a filename, a to_world, its BSDF (inline or a <ref>) and face_normals / flip_normals, and its mesh has no
    attribute beyond positions, normals and UVs.
Movable objects (physics.json, tools/physics_tag.py) the scope keeps apart become bodies: each one's shapes merge among
themselves only, one mesh per material, ``dyn_<object id>_<n>`` (a shape alone in its material stays as it is, kept reason
``dynamic_single``). The scope is scene.json ``physics.keep``, else physics.json ``keep``: "reachable" (the default: the
objects the robot's body can meet), "movable" (all), "none" (every object static). The other movable objects merge like
static ones. physics.json ``merge`` then names, per movable object, the shapes that hold it now (kept or not), and the
manifest each merged shape's ``object``. CS02: 215 shapes with nothing kept, 1,620 with every movable shape a shape of its
own (before), 471 with keep "movable" (79 objects in 140 dyn_ meshes and 279 single shapes) and 255 with keep
"reachable" (13 objects, 60 shapes).
Shapes merge when everything apart from the file and the transform is the same: the BSDF text, face_normals, flip_normals and
the note's material name (groundmap names surfaces and their friction by it; on CS02: 57 keys instead of 43, 11 of them alone),
and, after loading, the attribute layout (vertex normals, UVs). Every part is loaded by Mitsuba's own plugin with its to_world,
as the scene load does (normals computed for a PLY without them, the transform applied), concatenated, written with
Mesh.write_ply as ``assets/merged_<sha1>.ply`` and loaded back: positions, faces and UVs bit for bit, normals within 1e-6 (the
loader renormalises). face_normals and flip_normals stay properties of the merged shape. A group that fails stays as it was.

Undo and physics. ``merged_meshes.json`` (scene dir; scene.json's ``mesh_merge.manifest``) keeps, per merged shape, its file,
key and material, and per part its id, note, file, exact text in the scene file, vertex and face range in the merged mesh and
bounding box. ``--unmerge`` puts the original files back byte for byte (a scene file edited since gets every part back in place
of its merged shape). To let an object move, tag the scene (tools/physics_tag.py writes physics.json) or add it to scene.json
``"physics": {"dynamic": [regex, ...]}`` (or its note's ``"dynamic": true``) and run again: a merged scene is first unmerged,
then merged with the new plan (``--if-needed``: only when the plan changed, so a new physics.json is noticed). The
manifest names the original meshes, so tools/compact_assets.py keeps them (it keeps what any JSON names).
opticalnav_sim.merged_meshes.original_text gives a reader the scene object by object (groundmap.scene_surfaces reads it so).

Like compact_assets: files are never written in place (pack assets are hard links), the scene's lock is held, scene.json
records the run (``mesh_merge``), merged files no scene file or JSON names any more are unlinked. ``needs_merging`` is a text
scan (no Mitsuba); merge_pack / merge_scene / unmerge_scene do the work from Python.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compact_assets import SceneBusy, scene_lock  # noqa: E402
from opticalnav_sim import bodies as body_models, server  # noqa: E402
from opticalnav_sim.merged_meshes import DYN_PREFIX, MANIFEST, VERSION, read_manifest, restore_text, sha1_text  # noqa: E402
from opticalnav_sim.physics_objects import PHYSICS  # noqa: E402

MERGED_PREFIX = "merged_"
# A vertex attribute on every movable object's mesh. Mitsuba's loader merges all top-level meshes of one BSDF into one
# (MergeShape) unless a mesh has attributes: with it, an object's mesh stays a mesh of its own under its dyn_ id, which
# the render server moves by rewriting its vertices (and rebuilding its small acceleration structure) as physics
# pushes the object. Without it, 40 of CS02's 60 kept shapes were swallowed by larger meshes of their material.
BODY_ATTRIBUTE = "vertex_body"
MERGED_FILE = re.compile(r"merged_[0-9a-f]{20}\.ply$")
# ids the server finds shapes by (lights and their groups, the flash, the dataset's mirrors and glass walls, robot bodies)
PROTECTED_PREFIXES = ("light_", server.HIDDEN_LIGHT_PREFIX, server.HIDDEN_SPEC_PREFIX, server.FILL_PREFIX,
                      server.ASSIST_PREFIX, "mirror_auto_", "glass_auto_", body_models.BODY_PREFIX)
MESH_TYPES = ("ply", "obj")
FLAGS = ("face_normals", "flip_normals")  # shape properties a merged shape keeps as they are
STANDARD_KEYS = ("faces", "vertex_positions", "vertex_normals", "vertex_texcoords", "silhouette_sampling_weight")
NORMAL_TOLERANCE = 1e-6
NOTE_HEAD = "<!--opticalnav-obj:"

_TAG = re.compile(r'<!--.*?-->|<(/?)([A-Za-z_][\w.:-]*)((?:[^>"]|"[^"]*")*?)(/?)>', re.S)
_REF_ID = re.compile(r'<ref\b[^>]*\bid="([^"]+)"')
_FILENAME = re.compile(r'\s*<string name="filename" value="([^"]*)"\s*/>')
_TO_WORLD = re.compile(r'\s*<transform name="to_world"\s*>.*?</transform>', re.S)


def san(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_") or "material"


# ---------------------------------------------------------------- reading scene files
def top_shapes(text: str) -> list[dict]:
    """The top-level shapes of a scene file (children of <scene>), each with its block: the whitespace and the opticalnav-obj
    note right before it (when only whitespace separates them), then the element. Removing a block leaves the file as if
    the shape had never been written. {"id", "type", "start", "shape_start", "end", "lead", "shape", "inner", "note",
    "note_raw", "note_ok"}; ``note_ok`` False: a note precedes it that is another shape's (moving it would re-attach)."""
    items, depth, open_ = [], 0, None
    for m in _TAG.finditer(text):
        if m.group(0).startswith("<!--"):
            if depth == 1:
                items.append({"kind": "comment", "start": m.start(), "end": m.end()})
            continue
        closing, name, attrs, self_closing = m.groups()
        if closing:
            depth -= 1
            if depth == 1 and open_ is not None:
                open_.update(end=m.end(), close_start=m.start())
                items.append(open_)
                open_ = None
            continue
        if depth == 1:
            el = {"kind": "element", "tag": name, "attrs": attrs, "start": m.start(), "open_end": m.end()}
            if self_closing:
                el.update(end=m.end(), close_start=m.end())
                items.append(el)
            else:
                open_ = el
        if not self_closing:
            depth += 1
    out = []
    for i, it in enumerate(items):
        if it["kind"] != "element" or it["tag"] != "shape":
            continue
        kind = re.search(r'\btype="([^"]*)"', it["attrs"])
        sid = re.search(r'\bid="([^"]*)"', it["attrs"])
        note = raw = None
        head = it["start"]
        prev = items[i - 1] if i else None
        if prev is not None and prev["kind"] == "comment" and not text[prev["end"]:it["start"]].strip() \
                and text.startswith(NOTE_HEAD, prev["start"]):
            raw = text[prev["start"] + len(NOTE_HEAD):prev["end"] - 3]
            try:
                note = json.loads(raw)
            except ValueError:
                note = None
            head = prev["start"]
        start = head  # with the whitespace before it (the previous item ends in '>')
        while start > 0 and text[start - 1].isspace():
            start -= 1
        sid = sid.group(1) if sid else None
        out.append({"id": sid, "type": kind.group(1) if kind else None, "start": start, "shape_start": it["start"],
                    "end": it["end"], "lead": text[start:it["start"]], "shape": text[it["start"]:it["end"]],
                    "inner": text[it["open_end"]:it["close_start"]], "note": note, "note_raw": raw,
                    "note_ok": raw is None or (isinstance(note, dict) and note.get("id") in (None, sid))})
    return out


def scene_xmls(scene_dir: Path) -> list[Path]:
    """The scene files of a directory, base.xml first (it orders the parts and names the merged shapes)."""
    return sorted(scene_dir.glob("*.xml"), key=lambda p: (p.name != "base.xml", p.name))


def _strings(obj, out: set) -> None:
    if isinstance(obj, str):
        out.add(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.add(str(k))
            _strings(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _strings(v, out)


def json_strings(scene_dir: Path) -> set[str]:
    """Every key and string value of the scene dir's JSON files but this tool's manifest and physics.json (it names every
    shape, fixed or not; its ``movable_parts`` are read by ``physics_movable``): an id among them is looked up."""
    out: set[str] = set()
    for f in scene_dir.rglob("*.json"):
        if f.name in (MANIFEST, PHYSICS) and f.parent == scene_dir:
            continue
        try:
            _strings(json.loads(f.read_text()), out)
        except (OSError, ValueError):
            continue
    return out


KEEP_SCOPES = ("reachable", "movable", "none")


def read_physics(scene_dir: Path) -> dict | None:
    try:
        data = json.loads((scene_dir / PHYSICS).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def physics_scope(scene_dir: Path, meta: dict | None) -> str:
    """Which movable objects stay bodies of their own: scene.json ``physics.keep``, else physics.json ``keep``, else
    "reachable" (those the robot's body can meet: the physics step starts with them). "movable": all; "none": none (every
    object merged as a static one, physics.json still listing what is movable)."""
    scope = ((meta or {}).get("physics") or {}).get("keep") or (read_physics(scene_dir) or {}).get("keep") or "reachable"
    if scope not in KEEP_SCOPES:
        raise ValueError(f"{scene_dir.name}: physics keep {scope!r} is none of {KEEP_SCOPES}")
    return scope


def physics_dynamic(scene_dir: Path, meta: dict | None) -> dict[str, str]:
    """Shape id -> its object id, for the movable objects of physics.json (tools/physics_tag.py) the scope keeps apart.
    Each such object's shapes merge among themselves only (one mesh per object and material), as ``dyn_<object>_<n>``."""
    data = read_physics(scene_dir)
    scope = physics_scope(scene_dir, meta)
    if data is None or scope == "none":
        return {}
    return {str(p): str(o["id"]) for o in data.get("objects") or [] if o.get("movable")
            and (scope == "movable" or o.get("robot_reachable")) for p in o.get("parts") or []}


def rest_of(inner: str) -> str:
    """A shape's content without its filename and to_world: the BSDF (or its ref) and the flags."""
    return _TO_WORLD.sub("", _FILENAME.sub("", inner, count=1), count=1)


def plain_properties(rest: str) -> bool:
    """Whether a shape carries nothing but a BSDF (inline or a ref) and FLAGS beside its file and transform."""
    try:
        root = ET.fromstring(f"<x>{rest}</x>")
    except ET.ParseError:
        return False
    bsdfs = 0
    for el in root:
        if el.tag == "bsdf" or (el.tag == "ref" and el.get("name") in (None, "bsdf")):
            bsdfs += 1
        elif not (el.tag == "boolean" and el.get("name") in FLAGS):
            return False
    return bsdfs <= 1


# ---------------------------------------------------------------- the plan (text only)
def _plan(scene_dir: Path, texts: dict[str, str], keep, meta: dict | None) -> dict:
    """Which shapes merge with which, from the unmerged scene files: {"groups": [[id, ...], ...] (base order), "kept":
    Counter of reasons, "shapes": {id: shape of base}, "keys": [key per group], "keep": the regexes used}."""
    names = list(texts)
    base = texts[names[0]]
    shapes = top_shapes(base)
    by_file = {n: {} for n in names}
    dup: set[str] = set()
    for n in names:
        for s in (shapes if n == names[0] else top_shapes(texts[n])):
            if s["id"] in by_file[n]:
                dup.add(s["id"])
            by_file[n][s["id"]] = s["lead"] + s["shape"]
    refs = {r for t in texts.values() for r in _REF_ID.findall(t)}
    named = json_strings(scene_dir)
    noted: set[str] = set()  # ids named by another shape's note (a light's metadata.fixture_id)
    notes = [s["note"] for s in shapes if isinstance(s["note"], dict)]
    for note in notes:
        _strings(note.get("metadata") or {}, noted)
    groups_of_lights = server.light_groups(base)
    lamp_ids: set[str] = set()
    shape_m = {m.group(2): m for m in re.finditer(r'\n  <shape type="([a-z_]+)" id="([^"]+)">(.*?)\n  </shape>', base, re.S)}
    lamps, _, paired, orphans = server.pair_lamp_lights(notes or server.notes_from_shapes(shape_m))
    for o in lamps:
        lamp_ids |= {sid for sid in shape_m if sid == o["id"] or sid.startswith(o["id"] + "__part_")}
    lamp_ids |= {L["id"] for L in list(paired.values()) + list(orphans)}
    physics = [re.compile(r) for r in ((meta or {}).get("physics") or {}).get("dynamic") or []]
    dynamic = physics_dynamic(scene_dir, meta)
    keep_rx = [re.compile(r) for r in keep]

    def matches(rxs, s) -> bool:
        label = str((s["note"] or {}).get("label") or "") if isinstance(s["note"], dict) else ""
        return any(r.search(s["id"]) or (label and r.search(label)) for r in rxs)

    kept: Counter = Counter()
    groups: dict = {}
    for s in shapes:
        sid, note = s["id"], s["note"] if isinstance(s["note"], dict) else {}
        file = _FILENAME.search(s["inner"])
        rest = rest_of(s["inner"])
        if s["type"] not in MESH_TYPES or not sid:
            reason = "not_a_mesh"
        elif sid.startswith((MERGED_PREFIX, DYN_PREFIX)):
            reason = "merged"
        elif "<emitter" in s["inner"] or note.get("is_emitter") or sid in groups_of_lights:
            reason = "emitter"
        elif sid.startswith(PROTECTED_PREFIXES):
            reason = "protected_prefix"
        elif sid in lamp_ids:
            reason = "lamp"
        elif note.get("dynamic"):
            reason = "dynamic_note"
        elif matches(physics, s):
            reason = "physics_list"
        elif matches(keep_rx, s):
            reason = "keep_option"
        elif sid in refs or sid in named or sid in noted:
            reason = "referenced"
        elif sid in dup:
            reason = "duplicate_id"
        elif any(by_file[n].get(sid) != s["lead"] + s["shape"] for n in names[1:]):
            reason = "variant"
        elif not s["note_ok"]:
            reason = "note_layout"
        elif file is None or file.group(1).startswith("/") or ".." in Path(file.group(1)).parts \
                or not (scene_dir / file.group(1)).is_file():
            reason = "file"
        elif not plain_properties(rest):
            reason = "properties"
        else:
            reason = None
        if reason:
            kept[reason] += 1
            continue
        # a kept movable object's shapes merge among themselves: one body, a mesh per material
        key = (" ".join(rest.split()), str(note.get("material") or sid), dynamic.get(sid))
        groups.setdefault(key, []).append(sid)
    out_groups, keys = [], []
    for key, ids in groups.items():
        if len(ids) < 2 and not key[2]:   # a kept object's lone shape still becomes its dyn_ mesh (BODY_ATTRIBUTE)
            kept["single"] += 1
            continue
        out_groups.append(ids)
        keys.append(key)
    return {"groups": out_groups, "keys": keys, "objects": [k[2] for k in keys], "kept": kept,
            "shapes": {s["id"]: s for s in shapes}, "keep": list(keep), "count": len(shapes), "dynamic": dynamic,
            "scope": physics_scope(scene_dir, meta)}


def _original_texts(scene_dir: Path, manifest: dict | None) -> dict[str, str]:
    texts = {f.name: f.read_text() for f in scene_xmls(scene_dir)}
    if manifest is None:
        return texts
    return {n: restore_text(t, manifest, n)[0] for n, t in texts.items()}


def _meta(scene_dir: Path) -> dict | None:
    path = scene_dir / "scene.json"
    return json.loads(path.read_text()) if path.is_file() else None


def needs_merging(scene_dir: Path, keep=None) -> bool:
    """Whether merging would change the scene: no manifest yet and some shapes merge, or the plan of the unmerged files (with
    scene.json's physics list, the notes and ``keep``; None: the regexes the last run used) is not the one merged, or a
    merged file is gone. A text scan (no Mitsuba), well under a second on CS02."""
    scene_dir = Path(scene_dir)
    if not any(scene_dir.glob("*.xml")):
        return False
    manifest = read_manifest(scene_dir)
    if keep is None:
        keep = manifest.get("keep", []) if manifest else []
    plan = _plan(scene_dir, _original_texts(scene_dir, manifest), keep, _meta(scene_dir))
    # physics.json's record of the bodies out of date (another scope, other movable objects), even where the shapes merge
    # the same (the Infinigen scenes: few of their shapes merge at all)
    physics = read_physics(scene_dir)
    if physics is not None:
        record = physics.get("merge") or {}
        kept = {o for o, r in (record.get("objects") or {}).items() if r.get("kept")}
        if record.get("keep") != plan["scope"] or kept != set(plan["dynamic"].values()):
            return True
    if manifest is None:
        return bool(plan["groups"])
    if any(not (scene_dir / g["file"]).is_file() for g in manifest["merged"]):
        return True
    # a manifest the files do not follow (a scene converted again over the merged one): merge it anew
    if any(g["xml"] not in f.read_text() for f in scene_xmls(scene_dir) for g in manifest["merged"]):
        return True
    return plan["groups"] != manifest["plan"] \
        or plan["objects"] != manifest.get("plan_objects", [None] * len(manifest["plan"]))


# ---------------------------------------------------------------- geometry (Mitsuba, scalar_rgb)
_MI = []


def _mitsuba():
    if not _MI:
        import mitsuba as mi

        mi.set_variant("scalar_rgb")
        mi.set_log_level(mi.LogLevel.Warn)
        _MI.append(mi)
    return _MI[0]


def _geometry_xml(scene_dir: Path, kind: str, inner: str) -> str:
    """A shape alone, as the scene loads its geometry: file (absolute), to_world and flags; the material a placeholder."""
    file = _FILENAME.search(inner).group(1)
    to_world = _TO_WORLD.search(inner)
    flags = "".join(f'<boolean name="{f}" value="{m.group(1)}" />' for f in FLAGS
                    for m in [re.search(rf'<boolean name="{f}" value="([^"]+)"\s*/>', inner)] if m)
    path = str((scene_dir / file).resolve()).replace("&", "&amp;").replace('"', "&quot;")
    return (f'<shape version="3.0.0" type="{kind}"><string name="filename" value="{path}" />'
            f'{to_world.group(0) if to_world else ""}{flags}<bsdf type="diffuse" /></shape>')


def load_arrays(scene_dir: Path, kind: str, inner: str) -> dict:
    """A shape's world-space mesh as Mitsuba loads it: positions (N x 3), normals (N x 3 or None), uvs (N x 2 or None), faces
    (M x 3), and ``extra``: parameters beyond those (mesh attributes a BSDF may read: such a shape is not merged)."""
    import numpy as np

    mi = _mitsuba()
    mesh = mi.load_string(_geometry_xml(scene_dir, kind, inner))
    p = mi.traverse(mesh)
    extra = [k for k in p.keys() if k not in STANDARD_KEYS and not k.startswith("bsdf.")]
    return {"positions": np.array(p["vertex_positions"], np.float32).reshape(-1, 3),
            "normals": np.array(p["vertex_normals"], np.float32).reshape(-1, 3) if mesh.has_vertex_normals() else None,
            "uvs": np.array(p["vertex_texcoords"], np.float32).reshape(-1, 2) if mesh.has_vertex_texcoords() else None,
            "faces": np.array(p["faces"], np.uint32).reshape(-1, 3), "extra": extra}


def write_merged(scene_dir: Path, arrays: list[dict], merged_inner: str, body: bool = False) -> tuple[str, dict]:
    """The parts as one mesh in assets/merged_<sha1>.ply (Mesh.write_ply), checked by loading it as the merged shape loads.
    ``body``: a movable object's mesh, given the BODY_ATTRIBUTE vertex attribute so the scene loader leaves it a mesh of
    its own. Returns (file, {"positions", "normals", "uvs", "faces"} of the merged mesh)."""
    import numpy as np

    mi = _mitsuba()
    offsets = np.cumsum([0] + [len(a["positions"]) for a in arrays])
    merged = {"positions": np.concatenate([a["positions"] for a in arrays]),
              "normals": np.concatenate([a["normals"] for a in arrays]) if arrays[0]["normals"] is not None else None,
              "uvs": np.concatenate([a["uvs"] for a in arrays]) if arrays[0]["uvs"] is not None else None,
              "faces": np.concatenate([a["faces"] + np.uint32(o) for a, o in zip(arrays, offsets)])}
    mesh = mi.Mesh("merged", len(merged["positions"]), len(merged["faces"]),
                   has_vertex_normals=merged["normals"] is not None, has_vertex_texcoords=merged["uvs"] is not None)
    p = mi.traverse(mesh)
    for key, name in (("positions", "vertex_positions"), ("normals", "vertex_normals"), ("uvs", "vertex_texcoords"),
                      ("faces", "faces")):
        if merged[key] is not None:
            p[name] = type(p[name])(np.ascontiguousarray(merged[key]).ravel())
    p.update()
    if body:
        mesh.add_attribute(BODY_ATTRIBUTE, 1, [0.0] * len(merged["positions"]))
    assets = scene_dir / "assets"
    tmp = assets / f".merged.{os.getpid()}.ply.tmp"
    try:
        mesh.write_ply(str(tmp))
        back = load_arrays(scene_dir, "ply", _FILENAME.sub(
            f'<string name="filename" value="assets/{tmp.name}" />', merged_inner, count=1))
        bad = [k for k in ("positions", "uvs", "faces")
               if (back[k] is None) != (merged[k] is None) or (back[k] is not None and not np.array_equal(back[k], merged[k]))]
        if (back["normals"] is None) != (merged["normals"] is None) or (merged["normals"] is not None and not np.allclose(
                back["normals"], merged["normals"], rtol=0, atol=NORMAL_TOLERANCE)):
            bad.append("normals")
        if bad:
            raise ValueError(f"the merged PLY differs from its parts in {bad}")
        h = hashlib.sha1(tmp.read_bytes()).hexdigest()[:20]
        name = f"assets/merged_{h}.ply"
        if (scene_dir / name).exists():  # the same merge written before
            tmp.unlink()
        else:
            os.replace(tmp, scene_dir / name)
        return name, merged
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------- merging and unmerging
def _merged_note(mid: str, material: str, parts: list[dict], lo, hi, obj: str | None = None) -> dict:
    """The opticalnav-obj note of a merged shape, in the converter's fields (groundmap reads ``material``; light_groups,
    light_classes and pair_lamp_lights skip it: not an emitter, no lamp label)."""
    blocks = [((p["note"] or {}).get("nav") or {}).get("blocks", True) for p in parts]
    return {"id": mid, "type": "landmark", "label": f"merged:{material}", "placement": "point", "material": material,
            "source_ref": None, "is_emitter": False, "emitter_radiance": None, "emitter_intensity": 1.0, "emitter_shape": None,
            "emitter_polarized": False, "emitter_polarizer_angle_deg": 0.0, "emitter_pattern": None,
            "geometry": {"type": "point", "center": [float((lo[0] + hi[0]) / 2), float((lo[2] + hi[2]) / 2)], "start": None,
                         "end": None, "height_m": None, "thickness_m": None,
                         "size_m": [float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2])],
                         "base_height_m": float(lo[1]), "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0, "scale": None},
            "nav": {"blocks": any(blocks), "hazard_type": None, "include_hazard_mask": False, "goal_candidate": False,
                    "instruction_candidate": False},
            "merged_parts": len(parts), "merged_manifest": MANIFEST, **({"physics_object": obj} if obj else {})}


def _write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")  # replaced, never written through a hard link
    tmp.write_text(text)
    if path.exists():
        os.chmod(tmp, path.stat().st_mode & 0o7777)
    os.replace(tmp, path)


def _unlink_unused_merged(scene_dir: Path) -> list[str]:
    """Merged meshes no scene file or JSON names any more (an earlier plan's, or all after --unmerge)."""
    named = {r for f in scene_dir.glob("*.xml") for r in re.findall(r'value="(assets/[^"]+)"', f.read_text())}
    named |= {r for f in scene_dir.rglob("*.json") for r in re.findall(r'assets/[^"\\\s]+', f.read_text())}
    gone = []
    for p in sorted((scene_dir / "assets").glob("merged_*.ply")):
        if MERGED_FILE.match(p.name) and f"assets/{p.name}" not in named:
            p.unlink()
            gone.append(p.name)
    return gone


def _restore_files(scene_dir: Path, manifest: dict) -> dict:
    exact = {}
    for f in scene_xmls(scene_dir):
        text = f.read_text()
        out, ok = restore_text(text, manifest, f.name)
        exact[f.name] = ok
        if out != text:
            _write(f, out)
        if not ok and any(g["xml"] in text for g in manifest["merged"]):
            print(f"  ! {f.name}: edited since the merge; its parts are back in place of their merged shapes", flush=True)
    return exact


def _record_bodies(scene_dir: Path, plan: dict, merged: list[dict], base_text: str) -> None:
    """physics.json ``merge``: per movable object the shapes of the scene file that now hold its parts. Kept: its own
    ``dyn_`` meshes and the parts left alone (a single shape of a material, an emitter), which a physics step moves by
    their transforms or makes one body of; not kept: the static merged meshes it went into. And the scope used."""
    data = read_physics(scene_dir)
    if data is None:
        return
    holder = {p["id"]: g["id"] for g in merged for p in g["parts"]}
    present = {s["id"] for s in top_shapes(base_text)}
    kept_objects = set(plan["dynamic"].values())
    objects = {}
    for o in data.get("objects") or []:
        if o.get("movable"):
            shapes = list(dict.fromkeys(holder.get(p, p) for p in o.get("parts") or []))
            objects[o["id"]] = {"kept": o["id"] in kept_objects, "shapes": [x for x in shapes if x in present]}
    new = {"keep": plan["scope"], "manifest": MANIFEST if merged else None, "objects": objects}
    if data.get("merge") != new:
        data["merge"] = new
        _write(scene_dir / PHYSICS, json.dumps(data, separators=(",", ":")))


def _forget_bodies(scene_dir: Path) -> None:
    data = read_physics(scene_dir)
    if data is not None and "merge" in data:
        del data["merge"]
        _write(scene_dir / PHYSICS, json.dumps(data, separators=(",", ":")))


def unmerge_scene(scene_dir: Path) -> dict | None:
    """Put a merged scene's files back as they were (scene.json loses ``mesh_merge``; the merged meshes are unlinked).
    {"xmls": {name: exact?}, "unlinked": n}; None when the scene is not merged."""
    scene_dir = Path(scene_dir)
    with scene_lock(scene_dir):
        manifest = read_manifest(scene_dir)
        if manifest is None:
            return None
        exact = _restore_files(scene_dir, manifest)
        (scene_dir / MANIFEST).unlink()
        meta = _meta(scene_dir)
        if meta is not None and "mesh_merge" in meta:
            del meta["mesh_merge"]
            _write(scene_dir / "scene.json", json.dumps(meta, indent=1))
        _forget_bodies(scene_dir)
        gone = _unlink_unused_merged(scene_dir)
    print(f"{scene_dir.name}: unmerged ({sum(exact.values())}/{len(exact)} files exact), {len(gone)} merged meshes unlinked",
          flush=True)
    return {"xmls": exact, "unlinked": len(gone)}


def merge_scene(scene_dir: Path, keep=None, dry_run: bool = False) -> dict:
    """Merge a scene's static meshes (see the module doc). A merged scene is unmerged first and merged with the plan of now.
    ``keep``: id / label regexes kept apart for this and later runs (None: the last run's). Returns the summary scene.json
    records under ``mesh_merge``."""
    scene_dir = Path(scene_dir)
    with scene_lock(scene_dir):
        return _merge_scene(scene_dir, keep, dry_run)


def _merge_scene(scene_dir: Path, keep, dry_run: bool) -> dict:
    started = time.time()
    manifest = read_manifest(scene_dir)
    if keep is None:
        keep = manifest.get("keep", []) if manifest else []
    keep = list(keep)
    texts = _original_texts(scene_dir, manifest)
    meta = _meta(scene_dir)
    plan = _plan(scene_dir, texts, keep, meta)
    if dry_run:
        parts = sum(len(g) for g in plan["groups"])
        summary = {"shapes_before": plan["count"], "shapes_after": plan["count"] - parts + len(plan["groups"]),
                   "merged_groups": len(plan["groups"]), "parts": parts, "kept": dict(sorted(plan["kept"].items()))}
        print(f"{scene_dir.name} (dry run): {summary}", flush=True)
        return summary
    if manifest is not None:  # back to the originals first: the new plan starts from them
        _restore_files(scene_dir, manifest)
        (scene_dir / MANIFEST).unlink()
    names = list(texts)
    shapes = plan["shapes"]
    kept = Counter(plan["kept"])
    used_ids = {s["id"] for t in texts.values() for s in top_shapes(t)}
    merged, triangles, new_bytes = [], 0, 0
    for (rest_key, material, obj), ids in zip(plan["keys"], plan["groups"]):
        arrays = {}
        for sid in ids:
            arrays[sid] = load_arrays(scene_dir, shapes[sid]["type"], shapes[sid]["inner"])
        layouts: dict = {}
        for sid in ids:
            a = arrays[sid]
            if a["extra"]:
                kept["mesh_attributes"] += 1
                continue
            layouts.setdefault((a["normals"] is not None, a["uvs"] is not None), []).append(sid)
        for (has_n, has_uv), sub in layouts.items():
            if len(sub) < 2 and not obj:
                kept["single"] += 1
                continue
            first = shapes[sub[0]]
            if obj:     # the object's own meshes, numbered: dyn_<object>_1, _2, ... (one per material)
                k, mid = 1, f"{DYN_PREFIX}{san(obj)}_1"
                while mid in used_ids:
                    k += 1
                    mid = f"{DYN_PREFIX}{san(obj)}_{k}"
            else:
                k, mid = 1, f"{MERGED_PREFIX}{san(material)}"
                while mid in used_ids:
                    k += 1
                    mid = f"{MERGED_PREFIX}{san(material)}__{k}"
            used_ids.add(mid)
            rest = rest_of(first["inner"])
            placeholder = f'\n    <string name="filename" value="assets/merged.ply" />{rest}'
            try:
                file, mesh = write_merged(scene_dir, [arrays[s] for s in sub], placeholder, body=bool(obj))
            except Exception as exc:  # noqa: BLE001 - the group stays as it was
                print(f"  ! {mid}: {exc} (kept apart)", flush=True)
                kept["merge_failed"] += len(sub)
                continue
            lo, hi = mesh["positions"].min(0), mesh["positions"].max(0)
            parts, v0, f0 = [], 0, 0
            for sid in sub:
                a, s = arrays[sid], shapes[sid]
                nv, nf = len(a["positions"]), len(a["faces"])
                plo, phi = a["positions"].min(0), a["positions"].max(0)
                parts.append({"id": sid, "note": s["note"], "file": _FILENAME.search(s["inner"]).group(1), "type": s["type"],
                              "lead": s["lead"], "shape": s["shape"], "vertex_range": [v0, v0 + nv], "face_range": [f0, f0 + nf],
                              "bbox": [[float(x) for x in plo], [float(x) for x in phi]]})
                v0, f0 = v0 + nv, f0 + nf
            note = _merged_note(mid, material, parts, lo, hi, obj)
            xml = (f"\n  {NOTE_HEAD}{json.dumps(note, separators=(',', ':'))}-->\n  <shape type=\"ply\" id=\"{mid}\">"
                   f'\n    <string name="filename" value="{file}" />{rest}</shape>')
            merged.append({"id": mid, "file": file, "material": material, "object": obj,
                           "key": {"bsdf": rest_key, "flags": {f: bool(re.search(rf'<boolean name="{f}" value="true"', rest))
                                                               for f in FLAGS},
                                   "vertex_normals": has_n, "texcoords": has_uv},
                           "vertices": int(len(mesh["positions"])), "faces": int(len(mesh["faces"])),
                           "bbox": [[float(x) for x in lo], [float(x) for x in hi]], "xml": xml, "parts": parts})
            triangles += int(len(mesh["faces"]))
            new_bytes += (scene_dir / file).stat().st_size
        del arrays

    # the scene files: every part's block out, each merged shape where its first part was
    out_texts, records = {}, {}
    for n in names:
        text = texts[n]
        spans = {s["id"]: (s["start"], s["end"]) for s in top_shapes(text)}
        gone = sorted((spans[p["id"]][0], spans[p["id"]][1], p["id"]) for g in merged for p in g["parts"])
        pieces, offsets, last, removed = [], {}, 0, 0
        for a, b, pid in gone:
            pieces.append(text[last:a])
            offsets[pid] = a - removed
            removed += b - a
            last = b
        residual = "".join(pieces) + text[last:]
        inserts = sorted((offsets[g["parts"][0]["id"]], i) for i, g in enumerate(merged))
        out, last = [], 0
        for at, i in inserts:
            out += [residual[last:at], merged[i]["xml"]]
            last = at
        out.append(residual[last:])
        out_texts[n] = "".join(out)
        records[n] = {"residual_sha1": sha1_text(residual), "original_sha1": sha1_text(text),
                      "parts": [[pid, offsets[pid]] for _, _, pid in gone]}
    for n in names:  # a check before anything is written: the manifest gives every file back
        probe = {"merged": merged, "xmls": records}
        if restore_text(out_texts[n], probe, n) != (texts[n], True):
            raise RuntimeError(f"{scene_dir.name}/{n}: the merged file does not restore exactly")
    after = len(top_shapes(out_texts[names[0]]))
    parts_n = sum(len(g["parts"]) for g in merged)
    summary = {"version": VERSION, "at": datetime.now(timezone.utc).isoformat(), "manifest": MANIFEST,
               "shapes_before": plan["count"], "shapes_after": after, "merged_groups": len(merged), "parts": parts_n,
               "kept": dict(sorted(kept.items())), "keep": keep, "triangles": triangles, "merged_bytes": new_bytes,
               "seconds": round(time.time() - started, 1)}
    if merged:
        manifest_out = {"version": VERSION, "tool": "tools/merge_meshes.py", "at": summary["at"], "keep": keep,
                        "plan": plan["groups"], "plan_objects": plan["objects"], "physics_keep": plan["scope"],
                        "xmls": records, "merged": merged}
        _write(scene_dir / MANIFEST, json.dumps(manifest_out, separators=(",", ":")))
        for n in names:
            if out_texts[n] != texts[n]:
                _write(scene_dir / n, out_texts[n])
    meta = _meta(scene_dir)
    if meta is not None and (merged or "mesh_merge" in meta):
        if merged:
            meta["mesh_merge"] = summary
        else:
            del meta["mesh_merge"]
        _write(scene_dir / "scene.json", json.dumps(meta, indent=1))
    _record_bodies(scene_dir, plan, merged, out_texts[names[0]])
    unlinked = _unlink_unused_merged(scene_dir)
    print(f"{scene_dir.name}: shapes {summary['shapes_before']} -> {after}, {len(merged)} merged meshes of {parts_n} parts "
          f"({triangles} triangles, {new_bytes / 1e6:.1f} MB new), kept {summary['kept']}, "
          f"{len(unlinked)} stale merged files unlinked, {summary['seconds']} s", flush=True)
    return summary


def merge_pack(pack: Path, scenes=(), keep=None, if_needed: bool = True, dry_run: bool = False,
               unmerge: bool = False) -> dict:
    """Merge (or unmerge) the scenes of a pack (default: all). Scene id -> summary, None for a scene skipped (nothing to do),
    or "busy" for one another process is rewriting."""
    pack = Path(pack)
    scenes = list(scenes) or sorted(p.parent.name for p in (pack / "scenes").glob("*/scene.json"))
    out = {}
    for scene in scenes:
        scene_dir = (pack / "scenes" / scene).resolve()
        try:
            if unmerge:
                out[scene] = None if dry_run else unmerge_scene(scene_dir)
                continue
            if if_needed and not needs_merging(scene_dir, keep):
                out[scene] = None
                continue
            if not dry_run:
                _mitsuba()
            out[scene] = merge_scene(scene_dir, keep, dry_run)
        except SceneBusy as exc:
            print(f"skipped: {exc}", flush=True)
            out[scene] = "busy"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--scene", action="append", default=[], help="scene id (repeatable; default: every scene)")
    ap.add_argument("--if-needed", action="store_true", help="skip scenes whose merge is up to date (a text scan)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    ap.add_argument("--keep", action="append", default=None, metavar="REGEX",
                    help="keep shapes whose id or label matches apart (repeatable; remembered for later runs)")
    ap.add_argument("--clear-keep", action="store_true", help="forget the --keep regexes of earlier runs")
    ap.add_argument("--unmerge", action="store_true", help="put the original scene files back")
    args = ap.parse_args()
    keep = [] if args.clear_keep and args.keep is None else args.keep
    if not args.unmerge and not args.dry_run:
        _mitsuba()  # fail before touching anything when Mitsuba is not importable
    done = merge_pack(args.pack, args.scene, keep, args.if_needed, args.dry_run, args.unmerge)
    skipped = [s for s, r in done.items() if r is None]
    if skipped:
        print(f"nothing to do: {len(skipped)} scene(s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
