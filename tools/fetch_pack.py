#!/usr/bin/env python3
"""Fetch a shared scene pack (made by tools/export_pack.py), verify it and unpack it.

    python tools/fetch_pack.py --from <source> --out packs/opticalnav-v0.2 [--scene SCENE ...] [--list]

<source> is one of
    https://drive.google.com/drive/folders/<id>   a Google Drive folder shared by link (needs `pip install gdown`)
    remote:path/to/folder                         an rclone remote (needs rclone configured for that remote)
    /path/to/folder                               a local or mounted copy

Only the scenes asked for are downloaded (default: all of them). Each archive's sha256 is checked against
manifest.json before it is unpacked, and pack.json, connectivity/scans.txt and the R2R task files are
rewritten for the scenes present in --out, so a partial pack works with the simulator and the evaluator.
Needs tar and zstd (Ubuntu: apt install zstd).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


class Source:
    def __init__(self, spec: str):
        self.spec = spec
        self.kind = ("drive" if spec.startswith(("http://", "https://")) else
                     "local" if Path(spec).exists() else "rclone")
        self.ids: dict[str, str] = {}
        if self.kind == "drive":
            import gdown  # noqa: PLC0415

            listing = gdown.download_folder(spec, skip_download=True, quiet=True)
            if not listing:
                raise SystemExit(f"could not list {spec} (is it shared with 'anyone with the link'?)")
            self.ids = {f.path.replace("\\", "/"): f.id for f in listing}

    def get(self, rel: str, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if self.kind == "local":
            shutil.copyfile(Path(self.spec) / rel, dst)
        elif self.kind == "rclone":
            subprocess.run(["rclone", "copyto", "--progress", f"{self.spec.rstrip('/')}/{rel}", str(dst)], check=True)
        else:
            import gdown  # noqa: PLC0415

            if rel not in self.ids:
                raise SystemExit(f"{rel} is not in the shared folder")
            if not gdown.download(id=self.ids[rel], output=str(dst), quiet=False):
                raise SystemExit(f"download of {rel} failed")


def unpack(archive: Path, out: Path) -> None:
    zst = subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE)
    tar = subprocess.run(["tar", "-xf", "-", "-C", str(out)], stdin=zst.stdout)
    zst.stdout.close()
    if zst.wait() != 0 or tar.returncode != 0:
        raise SystemExit(f"unpacking {archive.name} failed")


def finish(out: Path, manifest: dict) -> dict[str, int]:
    """pack.json, scans.txt and R2R tasks for the scenes present in out."""
    present = sorted(p.parent.name for p in (out / "scenes").glob("*/scene.json"))
    (out / "connectivity").mkdir(parents=True, exist_ok=True)
    (out / "connectivity" / "scans.txt").write_text("\n".join(present) + "\n")
    counts = {}
    for f in sorted((out / "tasks" / "R2R" / "data").glob("R2R_*.json")):
        rows = [r for r in json.loads(f.read_text()) if r["scan"] in present]
        f.write_text(json.dumps(rows))
        counts[f.stem.replace("R2R_", "")] = len(rows)
    (out / "pack.json").write_text(json.dumps({
        "built_at": manifest.get("built_at"), "fetched_at": datetime.now(timezone.utc).isoformat(),
        "project": f"shared pack {manifest.get('pack')}", "scans": present, "splits": counts}, indent=1))
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="source", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--scene", action="append", default=[], help="scene id (repeatable; default: every scene)")
    ap.add_argument("--list", action="store_true", help="print the scenes on offer and exit")
    ap.add_argument("--keep-archives", action="store_true")
    args = ap.parse_args()
    for tool in ("tar", "zstd"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} not found (Ubuntu: apt install {tool})")

    src = Source(args.source)
    out = args.out.resolve()
    cache = out / ".download"
    src.get("manifest.json", cache / "manifest.json")
    manifest = json.loads((cache / "manifest.json").read_text())
    offered = manifest["scenes"]
    if args.list:
        for scene, s in offered.items():
            notes = "; ".join(filter(None, [
                "inferred: " + ", ".join(s["inferred"]) if s["inferred"] else "",
                "unavailable: " + ", ".join(s["variants_unavailable"]) if s["variants_unavailable"] else ""]))
            print(f"{scene}  {s['bytes'] / 1e9:5.2f} GB ({s['unpacked_bytes'] / 1e9:5.2f} GB unpacked)  "
                  f"variants {','.join(s['variants'])}  {notes}")
        return 0
    wanted = args.scene or list(offered)
    unknown = [s for s in wanted if s not in offered]
    if unknown:
        raise SystemExit(f"not in this pack: {unknown} (see --list)")
    todo = [(s, offered[s]) for s in wanted if not (out / "scenes" / s / "scene.json").is_file()]
    total = sum(e["bytes"] for _, e in todo)
    print(f"[fetch] {len(todo)} of {len(wanted)} scenes to download, {total / 1e9:.1f} GB "
          f"({sum(e['unpacked_bytes'] for _, e in todo) / 1e9:.1f} GB unpacked) -> {out}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    for name, entry in [("tasks", manifest["tasks"])] + todo:
        archive = cache / entry["file"]
        if not (archive.is_file() and sha256(archive) == entry["sha256"]):
            src.get(entry["file"], archive)
            digest = sha256(archive)
            if digest != entry["sha256"]:
                raise SystemExit(f"{entry['file']}: sha256 {digest} does not match the manifest; download again")
        unpack(archive, out)
        if not args.keep_archives:
            archive.unlink()
        print(f"[ok] {name}", flush=True)
    counts = finish(out, manifest)
    print(f"[done] {out}: scenes {len(json.loads((out / 'pack.json').read_text())['scans'])}, R2R splits {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
