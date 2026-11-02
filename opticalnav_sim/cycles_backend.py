"""Cycles as a render backend: a scene's ``.blend`` rendered by Blender, next to the Mitsuba and raster renderers.

The render server (server.py) asks for it with a view's ``"renderer": "cycles"``. One Blender process per ``.blend``
(cycles_worker.py) keeps the scene loaded and on the GPU; a request carries the camera the simulator's other renderers
use (pose, horizontal field of view, size), a sample count and a seed. The result is the linear RGB radiance of the
Combined pass, so the viewer accumulates passes of it exactly as it does Mitsuba's (a pass of N samples with its own seed,
the mean weighted by N).

The ``.blend`` travels with the scene: ``scene.json`` has ``"blend": {"file": "blend/scene.blend", "origin_offset":
[ox, oy, oz]}`` (tools/attach_blend.py writes it). robomituba's importer put a Blender point ``b`` at the dataset plan
position ``(bx + ox, oy - by, bz + oz)`` (height z), and the simulator's Mitsuba world is ``(x, height, y)`` of the plan, so
``Mitsuba = (bx + ox, bz + oz, oy - by)``: a rotation, nothing mirrored (sim_v2a_REF_PH19's 12 lights check it).
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

BLENDER = Path(os.environ.get("OPTICALNAV_BLENDER",
                              "/jarvis/project/robomituba/modules/infinigen/blender-4.2.0-linux-x64/blender"))
# on WSL the real OptiX / CUDA user-mode libraries (see README: driver-dist) come first, as for the Mitsuba server
DRIVER_DIRS = os.environ.get("OPTICALNAV_DRIVER_DIRS", "/home/jinnyeong/driver-dist:/usr/lib/wsl/lib")
WORKER = Path(__file__).with_name("cycles_worker.py")
OOM_WORDS = ("out of memory", "out of gpu memory", "cuda_error_out_of_memory", "optix_error_memory")


class CyclesError(RuntimeError):
    pass


def mitsuba_to_blender(origin_offset) -> np.ndarray:
    """4x4 matrix taking a Mitsuba world point (x, height, y) to the Blender scene's frame (see the module doc)."""
    ox, oy, oz = (float(v) for v in origin_offset)
    m = np.eye(4)
    m[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    m[:3, 3] = (-ox, oy, -oz)
    return m


def blender_camera(c2w: np.ndarray, origin_offset) -> np.ndarray:
    """The Blender camera matrix of a simulator ``camera_to_world`` (columns right, up, -forward, origin): Blender's
    camera looks down its -Z with +Y up, the same column order."""
    return mitsuba_to_blender(origin_offset) @ np.asarray(c2w, dtype=np.float64).reshape(4, 4)


def lib_dir(blender: Path = BLENDER) -> Path:
    """A directory of soname links for the Blender copy robomituba ships: its lib/ has the versioned libraries only
    (libsycl.so.7.2.0-8 but no libsycl.so.7), so the loader finds none of them. Made once, from what ldd misses."""
    out = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "opticalnav_sim" / "blender_libs"
    stamp = out / ".done"
    if stamp.is_file():
        return out
    out.mkdir(parents=True, exist_ok=True)
    libs = blender.parent / "lib"
    env = dict(os.environ, LD_LIBRARY_PATH=f"{libs}:{out}")
    for _ in range(3):  # a library's own dependencies show up once it is found
        files = [str(blender)] + [str(p) for p in libs.glob("*.so*")]
        text = subprocess.run(["ldd", *files], env=env, capture_output=True, text=True).stdout
        missing = sorted({line.split()[0] for line in text.splitlines() if "not found" in line})
        for name in missing:
            target = next(iter(sorted(libs.glob(name + "*"))), None)
            if target is not None:
                (out / name).unlink(missing_ok=True)
                (out / name).symlink_to(target)
    stamp.write_text("ok")
    return out


class CyclesWorker:
    """One Blender process rendering one ``.blend``; ``render`` is serialised by a lock."""

    def __init__(self, blend: Path, origin_offset, device: str = "OPTIX", blender: Path = BLENDER):
        self.blend, self.origin_offset, self.device = Path(blend), tuple(origin_offset), device
        self.lock = threading.Lock()
        self.load_s = 0.0
        self.info: dict = {}
        if not blender.is_file():
            raise CyclesError(f"no Blender at {blender} (OPTICALNAV_BLENDER)")
        env = dict(os.environ, OPTICALNAV_CYCLES_DEVICE=device,
                   LD_LIBRARY_PATH=f"{DRIVER_DIRS}:{blender.parent / 'lib'}:{lib_dir(blender)}")
        cmd_r, self._cmd_w = os.pipe()
        self._reply_r, reply_w = os.pipe()
        self.log = tempfile.NamedTemporaryFile(prefix="opticalnav_blender_", suffix=".log", mode="w+b")
        t0 = time.perf_counter()
        self.proc = subprocess.Popen([str(blender), "-b", str(self.blend), "--python", str(WORKER), "--", str(cmd_r), str(reply_w)],
                                     env=env, pass_fds=(cmd_r, reply_w), stdout=subprocess.DEVNULL, stderr=self.log)
        os.close(cmd_r)
        os.close(reply_w)
        self._cmd, self._reply = os.fdopen(self._cmd_w, "w"), os.fdopen(self._reply_r, "r")
        self.info = self._read()
        self.load_s = time.perf_counter() - t0

    def _read(self) -> dict:
        line = self._reply.readline()
        if not line:
            self.proc.wait(timeout=5)
            self.log.seek(0)
            raise CyclesError(f"Blender exited ({self.proc.returncode}): {self.log.read().decode(errors='replace')[-800:]}")
        return json.loads(line)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def render(self, c2w: np.ndarray, width: int, height: int, hfov_deg: float, spp: int, seed: int,
               denoise: bool = False, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Linear RGB, float32 (height, width, 3), and the worker's timing. ``options``: Cycles settings for this frame
        (``scene.cycles.<name>``: filter_glossy, sample_clamp_indirect ...); they stay set for the next frames."""
        fd, path = tempfile.mkstemp(prefix="opticalnav_cycles_", suffix=".f32", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
        os.close(fd)
        req = {"op": "render", "matrix": blender_camera(c2w, self.origin_offset).ravel().tolist(), "width": int(width),
               "height": int(height), "hfov_deg": float(hfov_deg), "spp": int(spp), "seed": int(seed),
               "denoise": bool(denoise), "options": options or {}, "out": path}
        try:
            with self.lock:
                self._cmd.write(json.dumps(req) + "\n")
                self._cmd.flush()
                reply = self._read()
            if not reply.get("ok"):
                raise CyclesError(reply.get("error", "render failed"))
            rgba = np.fromfile(path, dtype=np.float32).reshape(int(reply["height"]), int(reply["width"]), 4)
            return np.ascontiguousarray(rgba[..., :3]), reply
        finally:
            Path(path).unlink(missing_ok=True)

    def close(self) -> None:
        with self.lock:
            try:
                self._cmd.write(json.dumps({"op": "quit"}) + "\n")
                self._cmd.flush()
            except (OSError, ValueError):
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            for f in (self._cmd, self._reply):
                try:
                    f.close()
                except OSError:
                    pass


def is_oom(exc: BaseException) -> bool:
    return any(w in str(exc).lower() for w in OOM_WORDS)
