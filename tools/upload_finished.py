#!/usr/bin/env python3
"""Upload every scene of a long ``render_routes`` run to Google Drive as one tar, as the scene finishes.

    python3 tools/upload_finished.py start  --routes <kit>/routes --out /mnt/d/jn1027_frames --remote gcloud:dataset/opticalnav_sim/jn1027_frames/tar
    python3 tools/upload_finished.py status --routes ... --out ...
    python3 tools/upload_finished.py stop   --out ...

``start`` detaches and checks every ``--interval`` minutes. A scene is finished when all its frames exist, or when the
only ones missing have failed three times (``failed_frames*.json``): those are first given one more try (their failure
counts are reset once; ``render_supervisor.py`` takes them again the next time its renderer restarts), and if they
fail again the scene is uploaded without them and the missing frames are written to ``<scene>.missing.txt`` beside
the tar. The tar (no compression: EXR is compressed already) is written to ``<out>/../jn1027_upload`` with a low
priority, copied with rclone, checked by size against the remote, and deleted locally. Scenes done before this runs
(``--done``, default bookstoreR, uploaded as loose files) are skipped. State: ``<upload dir>/uploaded.json``; log:
``<upload dir>/upload_finished.log``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_routes import load_failed  # noqa: E402
from render_supervisor import frame_counts  # noqa: E402


def missing_frames(routes: Path, out: Path, scene: str) -> list[str]:
    data = json.loads((routes / f"{scene}.json").read_text())
    return [f"{scene}/{rid}/{k}" for rid, v in data.items() for k in range(len(v["poses"]))
            if not (out / "images" / rid / "exr" / f"{k:04d}.exr").is_file()]


def run(args) -> int:
    updir = args.out.parent / "jn1027_upload"
    updir.mkdir(exist_ok=True)
    state_path, stop = updir / "uploaded.json", updir / "STOP_UPLOAD"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    for scene in args.done:
        state.setdefault(scene, "uploaded earlier as loose files")
    state_path.write_text(json.dumps(state, indent=1))
    retried: set[str] = set()
    logf = open(updir / "upload_finished.log", "a", buffering=1)

    def log(msg: str) -> None:
        logf.write(f"{time.strftime('%m-%d %H:%M:%S')} {msg}\n")

    stop.unlink(missing_ok=True)
    scenes = sorted(p.stem for p in args.routes.glob("*.json"))
    log(f"start: {len(scenes)} scenes, {len(state)} already uploaded")
    while not stop.exists():
        for scene in scenes:
            if scene in state or stop.exists():
                continue
            miss = missing_frames(args.routes, args.out, scene)
            if miss:
                failed = load_failed(args.out)
                hopeless = [m for m in miss if failed.get(m, 0) >= 3]
                if len(hopeless) != len(miss):
                    continue  # frames still to render
                if scene not in retried:  # one more try for the frames that failed three times
                    retried.add(scene)
                    for f in args.out.glob("failed_frames*.json"):
                        d = json.loads(f.read_text())
                        for m in hopeless:
                            d.pop(m, None)
                        f.write_text(json.dumps(d))
                    log(f"{scene}: {len(hopeless)} frame(s) failed three times: reset for one more try")
                    continue
                # the retry failed again (three more failures each): the scene goes up without them
            tar = updir / f"{scene}.tar"
            log(f"{scene}: tar ({len(miss)} frames missing)")
            subprocess.run(["bash", "-c", f"cd {args.out} && nice -n 19 ionice -c3 tar -cf {tar}.part images/{scene}_gen_* && mv {tar}.part {tar}"],
                           check=True)
            if miss:
                (updir / f"{scene}.missing.txt").write_text("\n".join(miss) + "\n")
            log(f"{scene}: upload {tar.stat().st_size / 1e9:.2f} GB")
            up = subprocess.run(["rclone", "copy", str(tar), args.remote, "--transfers", "4", "--drive-chunk-size", "256M"],
                                capture_output=True, text=True)
            if miss:
                subprocess.run(["rclone", "copy", str(updir / f"{scene}.missing.txt"), args.remote])
            chk = subprocess.run(["rclone", "lsl", f"{args.remote}/{tar.name}"], capture_output=True, text=True).stdout.split()
            if up.returncode == 0 and chk and int(chk[0]) == tar.stat().st_size:
                state[scene] = {"bytes": tar.stat().st_size, "missing": len(miss), "at": time.strftime("%m-%d %H:%M")}
                state_path.write_text(json.dumps(state, indent=1))
                tar.unlink()
                log(f"{scene}: uploaded and checked, local tar deleted")
            else:
                log(f"{scene}: upload failed ({up.returncode}) or size differs: {up.stderr[-200:]}; the tar stays, retry next round")
        if len(state) >= len(scenes):
            log("all scenes uploaded")
            return 0
        for _ in range(args.interval * 6):
            if stop.exists():
                break
            time.sleep(10)
    log("stopped")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("start", "run", "status", "stop"))
    ap.add_argument("--routes", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--remote", default="gcloud:dataset/opticalnav_sim/jn1027_frames/tar")
    ap.add_argument("--interval", type=int, default=10, help="minutes between checks")
    ap.add_argument("--done", action="append", default=["sim_v2a_bookstoreR_underground_47311301"])
    args = ap.parse_args()
    updir = args.out.parent / "jn1027_upload"
    if args.command == "stop":
        updir.mkdir(exist_ok=True)
        (updir / "STOP_UPLOAD").write_text("stop\n")
        return 0
    if args.command == "status":
        st = json.loads((updir / "uploaded.json").read_text()) if (updir / "uploaded.json").is_file() else {}
        for k, v in st.items():
            what = v if isinstance(v, str) else "%.2f GB, missing %d (%s)" % (v["bytes"] / 1e9, v["missing"], v["at"])
            print(f"{k[7:44]:38s} {what}")
        print(f"{len(st)} scenes uploaded")
        lg = updir / "upload_finished.log"
        print("log:", lg.read_text().splitlines()[-1] if lg.is_file() and lg.read_text() else "-")
        return 0
    if args.routes is None:
        ap.error("--routes is required")
    if args.command == "start":
        rest = [a for a in sys.argv[2:]]
        updir.mkdir(exist_ok=True)
        with open(updir / "upload_finished.out", "ab") as f:
            subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "run", *rest], stdout=f, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        print("started in the background; status: python3 tools/upload_finished.py status --out", args.out)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
