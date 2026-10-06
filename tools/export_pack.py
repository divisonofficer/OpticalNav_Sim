#!/usr/bin/env python3
"""Pack a scene pack for sharing: one compressed archive per scene plus a manifest.

    python tools/export_pack.py --pack packs/opticalnav-v0.2 --out packs/share-opticalnav-v0.2 [--verified-only]
        [--scene SCENE ...] [--level 10]

Writes
    <out>/scenes/<scene>.tar.zst    scenes/<scene>/** and connectivity/<scene>_connectivity.json, hard links
                                    stored as files, so an extracted scene stands alone
    <out>/tasks.tar.zst             tasks/R2R/data/R2R_<split>.json for the exported scenes
    <out>/manifest.json             pack name, scenes (variants, inferred / unavailable notes, file, bytes,
                                    sha256, unpacked bytes), splits, tools needed to unpack

Upload <out> as one folder (for example `rclone copy <out> <remote>:<folder>`); the receiving side runs
tools/fetch_pack.py. Needs tar and zstd.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def pack_tar_zst(root: Path, members: list[str], dst: Path, level: int) -> None:
    """tar members (relative to root, hard links stored as files) piped through zstd into dst."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    tar = subprocess.Popen(["tar", "--hard-dereference", "--dereference", "-cf", "-", "-C", str(root), *members],
                           stdout=subprocess.PIPE)
    with tmp.open("wb") as out:
        zst = subprocess.run(["zstd", "-q", f"-{level}", "-T0", "-c"], stdin=tar.stdout, stdout=out)
    tar.stdout.close()
    if tar.wait() != 0 or zst.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"packing {dst.name} failed (tar {tar.returncode}, zstd {zst.returncode})")
    tmp.replace(dst)


def tree_bytes(paths: list[Path]) -> int:
    return sum(f.stat().st_size for p in paths for f in ([p] if p.is_file() else p.rglob("*")) if f.is_file())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--scene", action="append", default=[], help="scene id (repeatable; default: every scene)")
    ap.add_argument("--verified-only", action="store_true",
                    help="skip scenes with a variant matched by object ids (scene.json 'inferred')")
    ap.add_argument("--level", type=int, default=10, help="zstd level")
    ap.add_argument("--jobs", type=int, default=4,
                    help="scenes packed at once; reading many small files from a network share is the bottleneck")
    args = ap.parse_args()
    for tool in ("tar", "zstd"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} not found")

    pack = args.pack.resolve()
    metas = {p.parent.name: json.loads(p.read_text()) for p in sorted((pack / "scenes").glob("*/scene.json"))}
    scenes = args.scene or sorted(metas)
    skipped = []
    if args.verified_only:
        skipped = [s for s in scenes if any(v.get("inferred") for v in metas[s]["variants"].values())]
        scenes = [s for s in scenes if s not in skipped]
    out = args.out.resolve()
    (out / "scenes").mkdir(parents=True, exist_ok=True)
    manifest = {"pack": pack.name, "built_at": datetime.now(timezone.utc).isoformat(), "format": "tar.zst",
                "unpack": "zstd -dc <file> | tar -xf - -C <pack dir>  (or tools/fetch_pack.py)",
                "skipped_unverified": skipped, "scenes": {}}
    started = time.time()
    done = [0]

    def one(scene: str) -> tuple[str, dict]:
        members = [f"scenes/{scene}", f"connectivity/{scene}_connectivity.json"]
        dst = out / "scenes" / f"{scene}.tar.zst"
        raw = tree_bytes([pack / m for m in members])
        t = time.time()
        if not (dst.is_file() and dst.stat().st_mtime > (pack / "scenes" / scene / "scene.json").stat().st_mtime):
            pack_tar_zst(pack, members, dst, args.level)
        meta = metas[scene]
        entry = {
            "file": f"scenes/{scene}.tar.zst", "bytes": dst.stat().st_size, "sha256": sha256(dst),
            "unpacked_bytes": raw, "variants": sorted(meta["variants"]),
            "variants_unavailable": meta.get("variants_unavailable") or {},
            "inferred": {k: v["inferred"] for k, v in meta["variants"].items() if v.get("inferred")},
            "active_polar_needs_path_nocaustics": any(v.get("flash_xml") for v in meta["variants"].values()),
            "episodes": sum(len(r) for r in json.loads((pack / "scenes" / scene / "r2r.json").read_text())["splits"].values())
            if (pack / "scenes" / scene / "r2r.json").is_file() else None,
        }
        done[0] += 1
        print(f"[{done[0]}/{len(scenes)}] {scene}: {raw / 1e9:.2f} GB -> {dst.stat().st_size / 1e9:.2f} GB "
              f"({time.time() - t:.0f} s)", flush=True)
        return scene, entry

    with ThreadPoolExecutor(max(1, args.jobs)) as pool:
        for scene, entry in pool.map(one, scenes):
            manifest["scenes"][scene] = entry

    # tasks for the exported scenes only, so R2R splits never name a scene the receiver lacks
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "tasks" / "R2R" / "data"
        data.mkdir(parents=True)
        splits = {}
        for f in sorted((pack / "tasks" / "R2R" / "data").glob("R2R_*.json")):
            rows = [r for r in json.loads(f.read_text()) if r["scan"] in manifest["scenes"]]
            (data / f.name).write_text(json.dumps(rows))
            splits[f.stem.replace("R2R_", "")] = len(rows)
        pack_tar_zst(Path(tmp), ["tasks"], out / "tasks.tar.zst", args.level)
    manifest["tasks"] = {"file": "tasks.tar.zst", "bytes": (out / "tasks.tar.zst").stat().st_size,
                         "sha256": sha256(out / "tasks.tar.zst"), "splits": splits}
    manifest["bytes"] = sum(s["bytes"] for s in manifest["scenes"].values()) + manifest["tasks"]["bytes"]
    manifest["unpacked_bytes"] = sum(s["unpacked_bytes"] for s in manifest["scenes"].values())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[done] {len(scenes)} scenes, {manifest['unpacked_bytes'] / 1e9:.1f} GB -> {manifest['bytes'] / 1e9:.1f} GB "
          f"in {time.time() - started:.0f} s; splits {splits}; skipped {skipped or 'none'} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
