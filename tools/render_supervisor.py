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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_routes import load_failed  # noqa: E402

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
    with open(logs / f"server{'' if args.worker == 0 else '.' + str(args.worker)}.log", "ab") as f:
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
    sfx = "" if args.worker == 0 else f".{args.worker}"  # the files of worker i: supervisor.i.log, render.i.log, ...
    args.port += args.worker
    lock = open(args.out / f".supervisor{sfx}.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("a supervisor already runs on this --out", flush=True)
        return 1
    logf = open(args.out / "logs" / f"supervisor{sfx}.log", "a", buffering=1)

    def log(msg: str) -> None:
        logf.write(f"{time.strftime('%m-%d %H:%M:%S')} {msg}\n")

    stop = args.out / "STOP"
    if args.worker == 0 and not args.keep_stop:
        stop.unlink(missing_ok=True)
    time.sleep(30 * args.worker)  # the workers load their scene one after the other
    scenes = args.scene or sorted(p.stem for p in args.routes.glob("*.json"))
    log(f"supervisor start: {len(scenes)} scenes, format {args.format}, out {args.out}, worker {args.worker}, port {args.port}")
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
               "--spp", str(args.spp), "--variant", args.variant, "--worker", str(args.worker)]
        for scene in scenes:
            cmd += ["--scene", scene]
        if args.preview:
            cmd.append("--preview")
        log("render_routes start")
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        with open(args.out / "logs" / f"render{sfx}.log", "ab") as f:
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
    import re
    speeds = []
    for log in sorted((args.out / "logs").glob("render*.log")):
        last = None
        for line in log.read_text(errors="replace").splitlines()[-300:]:
            if "s/frame" in line:
                last = line
        if last:
            print(f"{log.name}: last speed: {last}")
            m = re.search(r"([\d.]+) s/frame", last)
            if m and float(m.group(1)) > 0:  # a run that rendered nothing logs "0 frames in 0 s (0.00 s/frame)"
                speeds.append(float(m.group(1)))
    hist_path = args.out / "logs" / "status_history.json"
    hist = json.loads(hist_path.read_text()) if hist_path.is_file() else []
    now = time.time()
    if not hist or now - hist[-1][0] > 600:
        hist.append([now, done])
        hist_path.write_text(json.dumps(hist[-500:]))
    measured = None
    old = [h for h in hist if now - h[0] >= 1800]
    if old:
        t0, d0 = old[-1]
        measured = (done - d0) / ((now - t0) / 3600)
        print(f"measured over the last {(now - t0) / 3600:.1f} h: {measured:.0f} frames/h")
    if speeds:
        pure = sum(3600 / v for v in speeds)
        rate = measured or pure
        print(f"renderers {len(speeds)}, frame speed {', '.join(f'{v:.1f}' for v in speeds)} s/frame ({pure:.0f} frames/h without restarts)")
        print(f"time left at {rate:.0f} frames/h: {(total - done) / rate:.0f} h ({(total - done) / rate / 24:.1f} days)")
    for sup in sorted((args.out / "logs").glob("supervisor*.log")):
        text = sup.read_text().splitlines()
        print(f"{sup.name}:", text[-1] if text else "")
    print(f"free on disk: {shutil.disk_usage(args.out).free / 1e9:.0f} GB; failed frames: "
          f"{len(load_failed(args.out))}")
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
    ap.add_argument("--workers", type=int, default=1, help="renderers at once, each with its own server on port "
                    "--port + i; they share the frame list through claim files, so one reloading its scene does not "
                    "stop the others")
    ap.add_argument("--worker", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--keep-stop", action="store_true", help=argparse.SUPPRESS)
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
        rest, skip = [], False
        for a in sys.argv[2:]:
            if skip:
                skip = False
            elif a == "--workers":
                skip = True
            elif not a.startswith("--workers="):
                rest.append(a)
        (args.out / "STOP").unlink(missing_ok=True)
        for i in range(args.workers):
            cmd = [sys.executable, str(Path(__file__).resolve()), "run", "--worker", str(i), "--keep-stop", *rest]
            with open(args.out / "logs" / f"supervisor{'' if i == 0 else '.' + str(i)}.out", "ab") as f:
                subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        print(f"{args.workers} worker(s) started in the background; status: python3 tools/render_supervisor.py status --routes {args.routes} --out {args.out}")
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
