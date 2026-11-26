"""A scene file as it was before tools/merge_meshes.py merged its static meshes, from the scene's ``merged_meshes.json``.

    text = merged_meshes.original_text(scene_dir, "base.xml")   # the file itself when the scene is not merged

Pure text (no Mitsuba): the tool's ``--unmerge`` and readers that need the scene object by object (groundmap: coplanar
surfaces settle a cell's material by their order in the file) both use it. The parts' own mesh files stay in the scene's
assets as long as the manifest names them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST = "merged_meshes.json"
VERSION = 1


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def read_manifest(scene_dir) -> dict | None:
    path = Path(scene_dir) / MANIFEST
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if data.get("version") != VERSION:
        raise ValueError(f"{path}: manifest version {data.get('version')} (this code reads {VERSION})")
    return data


def restore_text(text: str, manifest: dict, name: str) -> tuple[str, bool]:
    """A scene file as it was before the merge: (text, exact). Exact when everything but the merged shapes is as the merge
    left it (every part goes back to its offset, so the bytes are the original's); otherwise every merged shape found is
    replaced by its parts, in their order (the same scene, its shapes in another order)."""
    residual, found = text, []
    for g in manifest["merged"]:
        i = residual.find(g["xml"])
        if i >= 0:
            found.append(g)
            residual = residual[:i] + residual[i + len(g["xml"]):]
    record = manifest["xmls"].get(name)
    if record is not None and len(found) == len(manifest["merged"]) and sha1_text(residual) == record["residual_sha1"]:
        block = {p["id"]: p["lead"] + p["shape"] for g in manifest["merged"] for p in g["parts"]}
        out, last = [], 0
        for pid, at in record["parts"]:
            out += [residual[last:at], block[pid]]
            last = at
        out.append(residual[last:])
        return "".join(out), True
    for g in found:
        text = text.replace(g["xml"], "".join(p["lead"] + p["shape"] for p in g["parts"]), 1)
    return text, False


def original_text(scene_dir, xml: str = "base.xml") -> str:
    """The scene file ``xml`` as it was before the merge (as it is, when the scene has no manifest)."""
    scene_dir = Path(scene_dir)
    text = (scene_dir / xml).read_text()
    manifest = read_manifest(scene_dir) if 'id="merged_' in text else None
    return text if manifest is None else restore_text(text, manifest, xml)[0]
