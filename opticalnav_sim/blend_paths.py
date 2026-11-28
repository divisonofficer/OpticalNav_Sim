"""Where a scene's .blend is, the copy it is loaded from, and where the files it names went.

    blend_paths.library("/bean/infinigen_invers3r")      # {"blend_REF_PH33": Path(".../ref/REF_PH33/PH33_passed_r002.blend"), ...}
    copy = blend_paths.working_copy(blend, sid)            # ~/.cache/opticalnav_sim/blend_src/<sid>/<name>: what Blender opens
    blend_paths.fix_paths(bpy, origin, moves)              # inside Blender: relative and moved file paths made absolute, in memory

A scene library (``--blend-library``) holds approved scenes as ``<series>/<scene id>/<file>.blend`` (series ``ref``, ``cs``,
``old``), moved there from where they were made; its ``MANIFEST*.json`` files list every move (``moves``: ``source``, the old
path, and ``destination``), the .blend files' and the files they use (textures under ``<scene id>/deps/``). The library's
files are never opened for writing, nor opened at all: the converter and the Cycles backend open a copy in the cache
(``working_copy``, made again when the file changes). A copy elsewhere loses what its relative paths (``//textures/x.png``)
pointed at, so ``fix_paths`` reads them from where the .blend was made (``origin``: its manifest source), and a file that
was moved since from where the manifests say it went.

Pure Python (no bpy at import): the Blender processes load this file by its path.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

SERIES = ("ref", "cs", "old")
CACHE = Path(os.environ.get("OPTICALNAV_SIM_CACHE", Path.home() / ".cache" / "opticalnav_sim")) / "blend_src"


def library(root) -> dict:
    """``blend_<scene id>`` -> its .blend, for every scene folder of the library: the ``*passed*`` file when there is one (the
    approved round), else the newest .blend."""
    root = Path(root).expanduser()
    out = {}
    for series in SERIES:
        for d in sorted((root / series).glob("*")) if (root / series).is_dir() else ():
            files = [f for f in d.glob("*.blend") if f.is_file()]
            if not files:
                continue
            passed = [f for f in files if "passed" in f.name.lower()]
            pick = max(passed or files, key=lambda f: (f.stat().st_mtime_ns, f.name))
            out["blend_" + d.name] = pick
    return out


def moves(root) -> dict:
    """Old path -> new path, from every manifest of the library (a later manifest's move wins)."""
    out = {}
    for f in sorted(Path(root).expanduser().glob("MANIFEST*.json")):
        try:
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        for m in data.get("moves") or () if isinstance(data, dict) else ():
            if isinstance(m, dict) and m.get("source") and m.get("destination"):
                out[str(m["source"])] = str(m["destination"])
    return out


def origin_of(blend, moved: dict) -> Path:
    """Where a library file was made (its manifest source), else the file itself."""
    back = {v: k for k, v in moved.items()}
    return Path(back.get(str(Path(blend)), str(blend)))


def working_copy(blend, sid: str, cache: Path | None = None) -> Path:
    """The copy of ``blend`` that is opened (made, or made again when the file's size or time changed: copy2 keeps the
    time, so an unchanged copy compares equal)."""
    blend = Path(blend)
    folder = Path(cache or CACHE) / sid
    folder.mkdir(parents=True, exist_ok=True)
    dst = folder / blend.name
    st = blend.stat()
    if dst.is_file():
        d = dst.stat()
        if d.st_size == st.st_size and d.st_mtime_ns == st.st_mtime_ns:
            return dst
    tmp = folder / f".{blend.name}.{os.getpid()}.tmp"
    shutil.copy2(blend, tmp)
    os.replace(tmp, dst)
    for old in folder.glob("*.blend"):   # another file of the scene from before (an older round)
        if old != dst:
            old.unlink()
    return dst


def resolve(path: str, origin_dir: Path, moved: dict) -> str | None:
    """A file a .blend names, as an absolute path that exists: relative ones (``//``) from ``origin_dir``, a moved one where
    it went; None when neither is there."""
    if not path:
        return None
    p = Path(os.path.normpath(origin_dir / path[2:])) if path.startswith("//") else Path(path)
    if p.exists():
        return str(p)
    q = moved.get(str(p))
    if q and Path(q).exists():
        return q
    return None


def fix_paths(bpy, origin, moved: dict) -> dict:
    """Inside Blender, after opening a working copy: every image's and library's path made absolute where the original
    file's was (``origin``: the original .blend), moved files followed. In memory only; nothing is saved. Counts."""
    origin_dir = Path(origin).parent
    counts = {"fixed": 0, "missing": 0}
    for item in list(bpy.data.images) + list(bpy.data.libraries):
        fp = getattr(item, "filepath", "") or ""
        if not fp or (getattr(item, "packed_file", None) is not None):
            continue
        if not fp.startswith("//") and Path(fp).exists():
            continue
        found = resolve(fp, origin_dir, moved)
        if found is None:
            counts["missing"] += 1
            continue
        item.filepath = found
        if hasattr(item, "reload"):
            try:
                item.reload()
            except Exception:  # noqa: BLE001 - a library reloads itself; an image that will not reads as before
                pass
        counts["fixed"] += 1
    return counts


def load_into_blender(repo: Path):
    """This module, loaded by its path inside Blender (the package's __init__ imports what Blender's Python may lack)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("opticalnav_blend_paths", str(Path(repo) / "opticalnav_sim" / "blend_paths.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
