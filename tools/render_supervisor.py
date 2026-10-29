#!/usr/bin/env python3
"""Keep a long ``render_routes.py`` run going: the render server up, the renderer restarted, the work resumed.

    python3 tools/render_supervisor.py start  --routes <kit>/routes --out /mnt/d/jn1027_frames [--format exr] [--preview]
                                              [--gpu 0] [--port 18770] [--scene SCENE ...]
    python3 tools/render_supervisor.py status --routes <kit>/routes --out /mnt/d/jn1027_frames
    python3 tools/render_supervisor.py stop   --out /mnt/d/jn1027_frames

``start`` detaches (closing the terminal does not stop it) and loops until every frame of the route files exists:

1. the render server is started on ``--port`` (the options the renderer is meant to run with are its defaults) when
   nothing answers ``/v1/status`` there; a server that stops answering, or dies (a CUDA error ends the process), is
   killed and started again;
2. ``render_routes.py`` runs against it; when it exits with an error (the server was gone, a frame failed) the server
   is restarted and it runs again: frames already written are skipped, and a frame that has failed three times
   (``failed_frames.json``) is left out;
3. the work stops when ``render_routes.py`` ends cleanly, or when ``<out>/STOP`` exists (``stop`` writes it), or when
   the disk of ``--out`` has less than ``--min-free-gb`` free.

Logs are in ``<out>/logs/`` (``supervisor.log``, ``server.log``, ``render.log``). ``status`` prints the frames done
per scene against the route files, the speed of the last frames and the time left. One supervisor at a time per
``--out`` (a lock file).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MITSUBA = Path("/home/jinnyeong/robomituba-build/mitsuba3-3.9.1")
PYTHON = "/usr/bin/python3.10"


def env(gpu: int | None) -> dict:
    e = dict(os.environ, PYTHONPATH=f"{MITSUBA}/python:{REPO}", LD_LIBRARY_PATH=f"/usr/lib/wsl/lib:{MITSUBA}")
    if gpu is not None:
        e["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return e


def answering(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/status", timeout=10) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def server_pids(port: int) -> list[int]:
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines()
    return [int(l.split(None, 1)[0]) for l in out
            if f"opticalnav_sim.server" in l and f"--port {port}" in l and "ps -eo" not in l and "supervisor" not in l]


def stop_server(port: int, log) -> None:
    for pid in server_pids(port):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for _ in range(60):
        if not server_pids(port):
            return
        time.sleep(1)
    for pid in server_pids(port):
        log(f"server {pid} did not stop: kill -9")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(3)


def start_server(args, log) -> bool:
    stop_server(args.port, log)
    logs = args.out / "logs"
    cmd = [PYTHON, "-u", "-m", "opticalnav_sim.server", "--pack", str(args.pack), "--no-compact", "--port", str(args.port),
           "--modes", "polar", "--max-resident", "1", "--envmaps", str(args.envmaps)]
    with open(logs / "server.log", "ab") as f:
        subprocess.Popen(cmd, cwd=REPO, env=env(args.gpu), stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(120):
        time.sleep(2)
        if answering(args.port):
            log("render server up")
            return True
    log("render server did not come up in 4 minutes")
    return False


def frame_counts(routes: Path, out: Path, scenes: list[str], fmt: str) -> dict[str, tuple[int, int]]:
    ext = "exr/{k:04d}.exr" if fmt == "exr" else "rgb/{k:04d}.jpg"
    counts = {}
    for scene in scenes:
        data = json.loads((routes / f"{scene}.json").read_text())
        total = sum(len(v["poses"]) for v in data.values())
        done = sum(1 for rid, v in data.items() for k in range(len(v["poses"])) if (out / "images" / rid / ext.format(k=k)).is_file())
        counts[scene] = (done, total)
    return counts


def run(args) -> int:
    (args.out / "logs").mkdir(parents=True, exist_ok=True)
    lock = open(args.out / ".supervisor.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("a supervisor already runs on this --out", flush=True)
        return 1
    logf = open(args.out / "logs" / "supervisor.log", "a", buffering=1)

    def log(msg: str) -> None:
        logf.write(f"{time.strftime('%m-%d %H:%M:%S')} {msg}\n")

    stop = args.out / "STOP"
    stop.unlink(missing_ok=True)
    scenes = args.scene or sorted(p.stem for p in args.routes.glob("*.json"))
    log(f"supervisor start: {len(scenes)} scenes, format {args.format}, out {args.out}")
    failures = restarts = last_done = 0
    while not stop.exists():
        if shutil.disk_usage(args.out).free < args.min_free_gb * 1e9:
            log(f"stop: less than {args.min_free_gb} GB free on the disk of {args.out}")
            return 2
        if not answering(args.port) and not start_server(args, log):
            failures += 1
            if failures >= 5:
                log("stop: the server failed to start 5 times")
                return 3
            continue
        cmd = [PYTHON, "-u", str(REPO / "tools/render_routes.py"), "--server", f"http://127.0.0.1:{args.port}",
               "--pack", str(args.pack), "--routes", str(args.routes), "--out", str(args.out), "--format", args.format,
               "--spp", str(args.spp), "--variant", args.variant]
        for scene in scenes:
            cmd += ["--scene", scene]
        if args.preview:
            cmd.append("--preview")
        log("render_routes start")
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        with open(args.out / "logs" / "render.log", "ab") as f:
            proc = subprocess.Popen(cmd, cwd=REPO, env=env(None), stdout=f, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                if stop.exists():  # stop: end the renderer now (finished frames stay)
                    proc.terminate()
                    proc.wait(60)
                    break
                time.sleep(5)
        if stop.exists():
            break
        if proc.returncode == 0:
            counts = frame_counts(args.routes, args.out, scenes, args.format)
            left = sum(t - d for d, t in counts.values())
            log(f"render_routes done; {left} of the frames of the route files are not rendered (failed three times, or --limit)")
            return 0
        counts = frame_counts(args.routes, args.out, scenes, args.format)
        done_now = sum(d for d, _ in counts.values())
        failures = 0 if done_now > last_done else failures + 1  # restarts in a row that made no progress
        last_done = done_now
        restarts += 1
        log(f"render_routes exited {proc.returncode} (restart {restarts}, {failures} without progress, {done_now} frames done): "
            "restarting the server")
        if failures >= args.max_restarts:
            log(f"stop: {failures} restarts in a row without a new frame")
            return 4
        start_server(args, log)
        time.sleep(5)
    log("stopped (STOP file)")
    return 0


def status(args) -> int:
    scenes = args.scene or sorted(p.stem for p in args.routes.glob("*.json"))
    counts = frame_counts(args.routes, args.out, scenes, args.format)
    done, total = (sum(c[i] for c in counts.values()) for i in (0, 1))
    for scene, (d, t) in counts.items():
        print(f"{scene[7:44]:38s} {d:6d} / {t:6d}  {100 * d / max(t, 1):5.1f} %")
    print(f"{'TOTAL':38s} {done:6d} / {total:6d}  {100 * done / max(total, 1):5.1f} %")
    last = None
    log = args.out / "logs" / "render.log"
    if log.is_file():
        for line in log.read_text(errors="replace").splitlines()[-200:]:
            if "s/frame" in line:
                last = line
    if last:
        print("last speed:", last)
        try:
            spf = float(last.rsplit(",", 1)[1].split("s/frame")[0])
            print(f"time left at {spf:.1f} s/frame: {spf * (total - done) / 3600:.0f} h ({spf * (total - done) / 86400:.1f} days)")
        except (ValueError, IndexError):
            pass
    sup = args.out / "logs" / "supervisor.log"
    if sup.is_file():
        print("supervisor:", sup.read_text().splitlines()[-1] if sup.read_text() else "")
    print(f"free on disk: {shutil.disk_usage(args.out).free / 1e9:.0f} GB; failed frames: "
          f"{len(json.loads((args.out / 'failed_frames.json').read_text())) if (args.out / 'failed_frames.json').is_file() else 0}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("start", "run", "status", "stop"))
    ap.add_argument("--routes", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pack", type=Path, default=REPO / "packs/opticalnav-sim-v1")
    ap.add_argument("--envmaps", type=Path, default=REPO / "packs/opticalnav-v0.2/envmaps")
    ap.add_argument("--format", choices=("kit", "exr"), default="exr")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--spp", type=int, default=1024)
    ap.add_argument("--variant", default="glassmirror")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int, default=18770)
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--limit", type=int, default=None, help="a test: only the first N missing frames of each scene")
    ap.add_argument("--max-restarts", type=int, default=10, help="restarts in a row that wrote no new frame before it stops")
    ap.add_argument("--min-free-gb", type=float, default=50.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "stop":
        (args.out / "STOP").write_text("stop\n")
        print("STOP written: the supervisor ends the renderer within seconds (finished frames stay) and exits; "
              "the render server is left running")
        return 0
    if args.routes is None:
        ap.error("--routes is required")
    if args.command == "status":
        return status(args)
    if args.command == "start":  # detach
        (args.out / "logs").mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "run", *sys.argv[2:]]
        with open(args.out / "logs" / "supervisor.out", "ab") as f:
            subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        print(f"started in the background; status: python3 tools/render_supervisor.py status --routes {args.routes} --out {args.out}")
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
