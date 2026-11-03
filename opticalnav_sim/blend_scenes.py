"""Scenes served straight from their .blend files.

``python -m opticalnav_sim.server --blend scene.blend`` (or ``--blend-dir DIR``) converts each .blend with tools/blend_to_pack.py into
a scene of the *blend pack* (``packs/blend-cache`` by default) and serves it: the first start of a file takes seconds to minutes
(PH19 20 s, PH40 95 s, PH37 3 min), later starts find the converted scene and go straight on. A scene is converted again when its
.blend (size, modification time) or the converter changed. The scene's ``scene.json`` points at the .blend itself, so the Cycles
renderer (``"renderer": "cycles"``) renders the same file.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONVERTER = REPO / "tools" / "blend_to_pack.py"
DEFAULT_PACK = REPO / "packs" / "blend-cache"
PROGRESS: dict[str, dict] = {}  # scene id -> {"fraction", "stage", "t0"} of a conversion in progress (the server's /v1/status shows it)


def converter_version() -> str:
    return hashlib.sha1(CONVERTER.read_bytes()).hexdigest()[:12]


def scene_id(blend: Path) -> str:
    """blend_<stem>: PH19_passed_r004.blend -> blend_PH19_passed_r004."""
    return "blend_" + (re.sub(r"[^0-9A-Za-z._-]+", "_", blend.stem).strip("_") or "scene")


def parse_spec(spec: str) -> tuple[str, Path]:
    """``ID=PATH`` or ``PATH`` (the id then comes from the file name)."""
    if "=" in spec and not spec.startswith("/"):
        sid, path = spec.split("=", 1)
        return sid, Path(path).expanduser().resolve()
    path = Path(spec).expanduser().resolve()
    return scene_id(path), path


def up_to_date(pack: Path, sid: str, blend: Path) -> bool:
    stamp = pack / "scenes" / sid / ".source.json"
    if not (pack / "scenes" / sid / "base.xml").is_file() or not stamp.is_file():
        return False
    st = blend.stat()
    try:
        old = json.loads(stamp.read_text())
    except ValueError:
        return False
    return old == {"blend": str(blend), "size": st.st_size, "mtime_ns": st.st_mtime_ns, "converter": converter_version()}


def ensure(pack: Path, sid: str, blend: Path, force: bool = False) -> float:
    """Convert ``blend`` into the scene ``sid`` of ``pack`` unless it is already (seconds spent; 0 if nothing to do)."""
    if not blend.is_file():
        raise SystemExit(f"--blend: {blend} is not a file")
    if not force and up_to_date(pack, sid, blend):
        return 0.0
    pack.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[blend] converting {blend.name} -> {sid} (the first time takes seconds to minutes)", flush=True)
    PROGRESS[sid] = {"fraction": 0.01, "stage": "시작", "t0": t0}
    proc = subprocess.Popen([sys.executable, str(CONVERTER), "--pack", str(pack), "--blend", str(blend), "--scene-id", sid, "--link-blend"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail: list[str] = []
    try:
        for line in proc.stdout:
            m = re.match(r"\[progress\] ([0-9.]+) (.*)", line)
            if m:
                PROGRESS[sid] = {"fraction": float(m.group(1)), "stage": m.group(2).strip(), "t0": t0}
            tail = (tail + [line])[-40:]
        proc.wait()
    finally:
        PROGRESS.pop(sid, None)
    if proc.returncode or not (pack / "scenes" / sid / "base.xml").is_file():
        raise SystemExit(f"--blend: converting {blend} failed:\n{''.join(tail)[-1500:]}")
    st = blend.stat()
    (pack / "scenes" / sid / ".source.json").write_text(json.dumps(
        {"blend": str(blend), "size": st.st_size, "mtime_ns": st.st_mtime_ns, "converter": converter_version()}))
    print(f"[blend] {sid} converted in {time.time() - t0:.0f} s", flush=True)
    return time.time() - t0


def listed(dirs: list[str], tsvs: list[str]) -> dict[str, Path]:
    """The .blend files offered in the viewer's scene list (id -> file), not converted until one is chosen: every ``*.blend`` directly in a
    directory, and the ``id`` / ``passed_blend`` columns of robomituba's promoted_ref_scenes tsv (id ``blend_<id>``)."""
    out: dict[str, Path] = {}
    for d in dirs:
        for f in sorted(Path(d).expanduser().glob("*.blend")):
            out[scene_id(f)] = f.resolve()
    for tsv in tsvs:
        rows = [l.split("\t") for l in Path(tsv).expanduser().read_text().splitlines() if l.strip()]
        head = rows[0]
        ci, cp = head.index("id"), head.index("passed_blend")
        for r in rows[1:]:
            if len(r) > max(ci, cp) and r[cp]:
                out["blend_" + re.sub(r"[^0-9A-Za-z._-]+", "_", r[ci])] = Path(r[cp])
    return out


def prepare(specs: list[str], pack: Path | None = None) -> Path:
    """Convert every explicit --blend file now; returns the blend pack to serve."""
    pack = Path(pack) if pack else DEFAULT_PACK
    for sid, blend in (parse_spec(s) for s in specs):
        ensure(pack, sid, blend)
    return pack
