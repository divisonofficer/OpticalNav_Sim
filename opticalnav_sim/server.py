"""OpticalNav render server: resident Mitsuba scenes, polar or RGB renders on request.

    python -m opticalnav_sim.server --pack <pack_dir> [--port 18770] [--preload SCAN[:VARIANT[:MODE]]]

Mitsuba must be importable (PYTHONPATH of a build with an ``*_rgb_polarized``
variant; an ``*_rgb`` variant enables the faster RGB mode). Endpoints (JSON unless noted):

    GET  /v1/info                       server, renderers, variants, resident scenes
    GET  /v1/scans                      scan ids
    GET  /v1/scans/<scan>/meta          rig, intrinsics, variants, conventions
    GET  /v1/scans/<scan>/connectivity  Matterport3D connectivity JSON
    POST /v1/render                     {"views": [...]} -> NPZ (application/octet-stream)

All Mitsuba calls run on the main thread; HTTP requests queue render jobs for it.

A view is ``{"scan", "variant", "camera_to_world" (4x4, dataset convention),
"width", "height", "hfov_deg", "spp", "seed", "mode" ("polar" | "rgb"),
"denoise" (bool)}``. The NPZ holds ``rgb_<i>`` (uint8 HxWx3 preview) plus
``s0_<i>..s3_<i>`` (polar mode) or ``radiance_<i>`` (rgb mode), float16 or
float32 HxWx3. The ``X-Render-Timing`` header gives render / denoise / post seconds.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hmac
import io
import json
import os
import queue
import threading
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from . import frames, stokes

MAX_PIXELS = 2048 * 2048


class Activity:
    """What the render loop is doing now, readable from HTTP threads while it works (GET /v1/status)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.what, self.since = "", None

    @contextlib.contextmanager
    def doing(self, what: str):
        with self.lock:
            prev = (self.what, self.since)
            self.what, self.since = what, time.time()
        try:
            yield
        finally:
            with self.lock:
                self.what, self.since = prev

    def snapshot(self) -> dict:
        with self.lock:
            return {"what": self.what, "elapsed_s": round(time.time() - self.since, 1) if self.since else 0.0}


ACTIVITY = Activity()
MAX_SPP = 8192
MAX_VIEWS = 64
MAX_BODY = 1 << 20
MODE_VARIANTS = {"polar": ("cuda_rgb_polarized", "cuda_ad_rgb_polarized"), "rgb": ("cuda_rgb", "cuda_ad_rgb")}
# Without dr.freeze, high spp is rendered in passes of at most this many samples (wavefront memory);
# a polarized path carries Mueller matrices, so its passes are smaller.
# What a frozen render takes as input. "scene": the whole scene, so Dr.Jit walks every texture and mesh
# on each replay (~23 ms on Device 1). "sensor": only the camera, with the static scene captured when the
# function is recorded; a replay then walks the sensor alone. Flash scenes always use "scene" (the flash
# moves with the camera).
FREEZE_INPUT = os.environ.get("OPTICALNAV_SIM_FREEZE_INPUT", "scene")
SPP_CHUNK = {"polar": max(1, int(os.environ.get("OPTICALNAV_SIM_SPP_CHUNK", "64"))),
             "rgb": max(1, int(os.environ.get("OPTICALNAV_SIM_SPP_CHUNK_RGB", "256")))}


def rgb_scene_text(xml: Path) -> str:
    """Production Stokes scene -> plain radiance scene: its nested integrator replaces the stokes wrapper."""
    root = ET.parse(xml).getroot()
    integrator = root.find("./integrator")
    if integrator is not None and integrator.get("type") == "stokes":
        nested = integrator.find("./integrator")
        index = list(root).index(integrator)
        root.remove(integrator)
        root.insert(index, nested)
    return ET.tostring(root, encoding="unicode")


class Resident:
    """One loaded Mitsuba scene, driven only through mi.traverse() between renders."""

    def __init__(self, mi, xml: Path, mode: str, flash: dict | None = None, freeze: bool = False):
        self.mi, self.mode = mi, mode
        started = time.time()
        if mode == "rgb":
            mi.Thread.thread().file_resolver().prepend(str(xml.parent))  # assets are relative to the XML
            self.scene = mi.load_string(rgb_scene_text(xml))
        else:
            self.scene = mi.load_file(str(xml))
        self.load_s = time.time() - started
        keys = list(mi.traverse(self.scene).keys())
        self.k_to_world = next(k for k in keys if k.endswith("sensor.to_world"))
        self.k_fov = next(k for k in keys if k.endswith("sensor.x_fov"))
        self.k_size = next((k for k in keys if k.endswith("sensor.film.size")), None)
        self.flash = flash
        self.k_flash = next((k for k in keys if k.startswith("camera_assist_light") and k.endswith("to_world")), None)
        self.k_polarizer = next((k for k in keys if k.startswith("camera_assist_polarizer") and k.endswith("to_world")), None)
        if flash is not None and self.k_flash is None:
            raise RuntimeError(f"{xml}: flash scene has no camera_assist_light transform")
        # Camera-only updates traverse just the sensor: traversing and updating the whole scene (thousands
        # of textures and meshes) cost ~46 ms per frame on Device 1. A flash scene also moves its flash, so
        # it keeps the whole-scene path.
        self.sensor = self.scene.sensors()[0]
        sensor_keys = set(mi.traverse(self.sensor).keys())
        self.ks_size = "film.size" if "film.size" in sensor_keys else None
        self.size = self.fov = None
        # dr.freeze records the render once per (spp, film size, seed) and replays its kernels, like
        # the production renderer on Device 1; without it every frame re-traces the scene.
        self.freeze = freeze
        self.sensor_input = FREEZE_INPUT == "sensor" and flash is None
        self.frozen: dict[tuple, object] = {}
        self.batch_sensors: dict[tuple, object] = {}

    def render(self, c2w: np.ndarray, width: int, height: int, hfov_deg: float, spp: int, seed: int) -> np.ndarray:
        img, self.breakdown = self.finish(self.launch(c2w, width, height, hfov_deg, spp, seed))
        return img

    def launch(self, c2w: np.ndarray, width: int, height: int, hfov_deg: float, spp: int, seed: int) -> tuple:
        """Set the camera and queue the render on the GPU. Returns a handle for ``finish``, which waits for
        the GPU and reads the image back, so the caller can queue the next view in between (CUDA runs
        the queued kernels in order)."""
        mi = self.mi
        t0 = time.perf_counter()
        origin, target, up = frames.lookat(c2w)
        look = mi.ScalarTransform4f.look_at(origin=origin, target=target, up=up)
        if self.flash is None:
            p = mi.traverse(self.sensor)  # fresh map: a frozen replay swaps the variables an old map holds
            t1 = time.perf_counter()
            p["to_world"] = look
            if self.fov != float(hfov_deg):
                p["x_fov"] = float(hfov_deg)
                self.fov = float(hfov_deg)
            if self.ks_size is not None and self.size != (int(width), int(height)):
                p[self.ks_size] = mi.ScalarVector2u(int(width), int(height))
                self.size = (int(width), int(height))
            p.update()
            t2 = time.perf_counter()
        else:
            p = mi.traverse(self.scene)
            t1 = time.perf_counter()
            p[self.k_to_world] = look
            p[self.k_fov] = float(hfov_deg)
            if self.k_size is not None:
                p[self.k_size] = mi.ScalarVector2u(int(width), int(height))
            d = float(self.flash["distance_m"])
            p[self.k_flash] = mi.ScalarTransform4f(flash_matrix(c2w, self.flash["size_world"], d).tolist())
            if self.k_polarizer is not None:
                p[self.k_polarizer] = mi.ScalarTransform4f(flash_matrix(
                    c2w, self.flash["size_world"], d + 0.01, float(self.flash.get("polarizer_angle_deg") or 0.0)).tolist())
            p.update()
            t2 = time.perf_counter()
        breakdown = {"traverse_s": t1 - t0, "update_s": t2 - t1}
        if not self.freeze:
            # passes of at most SPP_CHUNK samples, averaged by sample count
            total, done, acc, i = int(spp), 0, None, 0
            while done < total:
                n = min(SPP_CHUNK[self.mode], total - done)
                img = np.array(mi.render(self.scene, spp=n, seed=int(seed) + 104729 * i), dtype=np.float64) * n
                acc = img if acc is None else acc + img
                done, i = done + n, i + 1
            return (acc / total).astype(np.float32), breakdown
        import drjit as dr

        key = (int(spp), int(width), int(height))
        if key not in self.frozen:  # spp and film size are baked into each recording
            # the seed is an input rather than a constant, so one recording serves every seed
            # (progressive viewers render the same camera again and again with new seeds); every input
            # changes between frames, so all are recorded opaque rather than as literals
            with ACTIVITY.doing(f"recording {spp} spp {width}x{height}"):
                if self.sensor_input:
                    scene = self.scene
                    frozen = dr.freeze(lambda sn, sd, n=int(spp): mi.render(scene, sensor=sn, spp=n, seed=sd),
                                       backend=dr.JitBackend.CUDA, auto_opaque=False)
                    self._settle(frozen, self.sensor)
                else:
                    frozen = dr.freeze(lambda scene, sd, n=int(spp): mi.render(scene, spp=n, seed=sd),
                                       backend=dr.JitBackend.CUDA, auto_opaque=False)
                    self._settle(frozen, self.scene)
                self.frozen[key] = frozen  # listed by /v1/status only once both recordings are done
            t2 = time.perf_counter()
        source = self.sensor if self.sensor_input else self.scene
        img = self.frozen[key](source, dr.opaque(mi.UInt32, int(seed)))  # replays, launching the kernels
        breakdown["replay_s"] = time.perf_counter() - t2
        return img, breakdown

    def finish(self, handle: tuple) -> tuple[np.ndarray, dict]:
        img, breakdown = handle
        if isinstance(img, np.ndarray):
            return img, breakdown
        import drjit as dr

        t0 = time.perf_counter()
        dr.sync_thread()  # wait for this view's kernels (and any queued before them)
        t1 = time.perf_counter()
        out = np.array(img)
        return out, {**breakdown, "wait_s": t1 - t0, "readback_s": time.perf_counter() - t1}

    def _settle(self, frozen, *args) -> None:
        """Record, then replay once. The second call of a new recording records again (measured on
        Device 1: 79 s, then 71 s, then 0.1 s, with the same or a new seed; pre-seeding the sensor's
        sampler did not prevent it), most likely because the first render leaves scene state such as the
        sampler and film in a new layout. Paying both now keeps later frames at replay speed."""
        import drjit as dr

        for _ in range(2):
            dr.eval(frozen(*args, dr.opaque(self.mi.UInt32, 0)))


    def render_many(self, cams: list, width: int, height: int, hfov_deg: float, spp: int, seed: int) -> list:
        """K views in one render call through a resident ``batch`` sensor.

        Tracing the scene (materials, lights, geometry) dominates a call and does not depend on the
        cameras, so one call for K views pays it once. The camera-aligned flash of an active scene
        follows a single camera, so those views still render one at a time.
        """
        if len(cams) == 1 or self.flash is not None:
            return [self.render(c, width, height, hfov_deg, spp, seed) for c in cams]
        mi, k = self.mi, len(cams)
        looks = []
        for c in cams:
            origin, target, up = frames.lookat(c)
            looks.append(mi.ScalarTransform4f.look_at(origin=origin, target=target, up=up))
        key = (k, int(width), int(height), round(float(hfov_deg), 6))
        if key not in self.batch_sensors:
            spec = {"type": "batch", "film": {"type": "hdrfilm", "width": k * int(width), "height": int(height)},
                    "sampler": {"type": "independent", "sample_count": 1}}
            for i, look in enumerate(looks):
                spec[f"slot{i}"] = {"type": "perspective", "fov": float(hfov_deg), "fov_axis": "x", "to_world": look}
            self.batch_sensors[key] = mi.load_dict(spec)
        sensor = self.batch_sensors[key]
        p = mi.traverse(sensor)
        for i, look in enumerate(looks):
            p[f"slot_{i}.to_world"] = look
        p.update()
        if not self.freeze:
            # the per-pass memory budget is SPP_CHUNK samples for each of the K views together
            per_pass = max(1, SPP_CHUNK[self.mode] // k)
            total, done, acc, i = int(spp), 0, None, 0
            while done < total:
                n = min(per_pass, total - done)
                img = np.array(mi.render(self.scene, sensor=sensor, spp=n, seed=int(seed) + 104729 * i),
                               dtype=np.float64) * n
                acc = img if acc is None else acc + img
                done, i = done + n, i + 1
            img = (acc / total).astype(np.float32)
        else:
            import drjit as dr

            fkey = ("batch",) + key + (int(spp),)
            if fkey not in self.frozen:
                # the sensor and seed are arguments, not closure values: a frozen replay only sees updated
                # slot poses on objects it receives as inputs (production keeps its batch sensor inside the scene)
                self.frozen[fkey] = dr.freeze(
                    lambda scene, sn, sd, n=int(spp): mi.render(scene, sensor=sn, spp=n, seed=sd),
                    backend=dr.JitBackend.CUDA, auto_opaque=False)
                self._settle(self.frozen[fkey], self.scene, sensor)
            img = np.array(self.frozen[fkey](self.scene, sensor, dr.opaque(mi.UInt32, int(seed))))
        w = int(width)
        return [img[:, i * w:(i + 1) * w] for i in range(k)]


def flash_matrix(c2w: np.ndarray, size_world, distance_m: float, roll_deg: float = 0.0) -> np.ndarray:
    """Camera-aligned rectangle (port of multimodal._camera_aligned_rectangle_matrix)."""
    right = c2w[:3, 0] / np.linalg.norm(c2w[:3, 0])
    up = c2w[:3, 1] / np.linalg.norm(c2w[:3, 1])
    forward = -c2w[:3, 2] / np.linalg.norm(c2w[:3, 2])
    if abs(roll_deg) > 1e-6:
        a = np.deg2rad(roll_deg)
        right, up = right * np.cos(a) + up * np.sin(a), -right * np.sin(a) + up * np.cos(a)
        right, up = right / np.linalg.norm(right), up / np.linalg.norm(up)
    m = np.eye(4)
    m[:3, 0] = right * (float(size_world[0]) * 0.5)
    m[:3, 1] = up * (float(size_world[1]) * 0.5)
    m[:3, 2] = forward
    m[:3, 3] = c2w[:3, 3] - forward * float(distance_m)
    return m


class MainThreadExecutor:
    """Runs submitted jobs on the thread that calls run_forever(). Mitsuba keeps its variant and
    file resolver per thread, and loading an RGB scene from a worker thread segfaulted on the OptiX 7
    build, so all Mitsuba work stays on the main thread while HTTP is served from a background one."""

    def __init__(self):
        self.jobs: "queue.Queue[tuple]" = queue.Queue()
        self.main = threading.get_ident()

    def submit(self, fn, *args) -> Future:
        future = Future()
        if threading.get_ident() == self.main:  # start-up work (preload) before the loop runs
            future.set_result(fn(*args))
        else:
            self.jobs.put((fn, args, future, False))
        return future

    def submit_pipelined(self, start, *args) -> Future:
        """``start(*args)`` queues GPU work and returns ``finish()``, which completes the job. The loop starts
        the next queued job before finishing this one, so the CPU work of one view (camera update, kernel
        launch, readback, post-processing) overlaps the GPU work of its neighbour."""
        future = Future()
        self.jobs.put((start, args, future, True))
        return future

    @staticmethod
    def _complete(pending) -> None:
        finish, future = pending
        try:
            future.set_result(finish())
        except BaseException as exc:  # noqa: BLE001 - delivered to the waiting request
            future.set_exception(exc)

    def run_forever(self) -> None:
        pending = None  # (finish, future) of the job whose GPU work is in flight
        while True:
            try:  # with work in flight, take the next job only if it is already waiting
                fn, args, future, pipelined = self.jobs.get(timeout=0.002) if pending else self.jobs.get()
            except queue.Empty:
                self._complete(pending)
                pending = None
                continue
            if fn is None:
                if pending:
                    self._complete(pending)
                return
            if not pipelined:
                if pending:
                    self._complete(pending)
                    pending = None
                try:
                    future.set_result(fn(*args))
                except BaseException as exc:  # noqa: BLE001
                    future.set_exception(exc)
                continue
            try:
                finish = fn(*args)
            except BaseException as exc:  # noqa: BLE001
                future.set_exception(exc)
                continue
            if pending:
                self._complete(pending)
            pending = (finish, future)


class Renderer:
    """Mitsuba work runs on the main thread (see MainThreadExecutor); one process serialises its renders."""

    def __init__(self, pack: Path, variant: str | None, max_resident: int, freeze: str = "auto",
                 rgb_variant: str | None = None, modes: tuple[str, ...] = ("polar",), kernel_history: bool = False):
        import drjit as dr
        import mitsuba as mi

        available = mi.variants()
        wanted = {"polar": variant, "rgb": rgb_variant}
        self.variant_of = {}
        for mode in modes:
            chosen = wanted[mode] or next((v for v in MODE_VARIANTS[mode] if v in available), None)
            if chosen not in available:
                raise SystemExit(f"no Mitsuba variant for {mode} mode (available: {available})")
            self.variant_of[mode] = chosen
        self.mi, self.dr, self.pack = mi, dr, pack
        self.variant = next(iter(self.variant_of.values()))
        mi.set_variant(self.variant)
        # Per-kernel codegen / compile / GPU times split a frame into tracing vs ray tracing, but reading the
        # history synchronises the device, which stops one view's GPU work overlapping the next one's CPU
        # work. Off by default; --kernel-history turns it on for measurements (benchmark_modes, frame_overhead).
        self.kernel_history = bool(kernel_history)
        dr.set_flag(dr.JitFlag.KernelHistory, self.kernel_history)
        # ponytail: one render loop per server process; run one server per GPU to scale out.
        self.executor = MainThreadExecutor()
        self.max_resident = max(1, max_resident)
        self.meta = {p.parent.name: json.loads(p.read_text()) for p in sorted((pack / "scenes").glob("*/scene.json"))}
        self.cache: OrderedDict[tuple[str, str, str], list[Resident]] = OrderedDict()
        self.denoisers: dict[tuple[str, int, int], object] = {}
        self.has_nocaustics = self._has_plugin("path_nocaustics")
        self.freeze = hasattr(dr, "freeze") if freeze == "auto" else freeze == "on"

    def _has_plugin(self, name: str) -> bool:
        try:
            self.mi.load_dict({"type": name})
            return True
        except Exception:  # noqa: BLE001 - absent plugin raises a variant-specific error type
            return False

    def variants(self, scan: str) -> list[str]:
        return [name for name, v in self.meta[scan]["variants"].items()
                if self.has_nocaustics or not v.get("flash_xml")]

    def _on_main(self, mode: str, fn, *args):
        def job():
            if self.mi.variant() != self.variant_of[mode]:
                self.mi.set_variant(self.variant_of[mode])
            return fn(*args)
        return self.executor.submit(job).result()

    def preload(self, scan: str, variant: str, mode: str = "polar") -> float:
        return sum(r.load_s for r in self._on_main(mode, self._resident, scan, variant, mode))

    def _resident(self, scan: str, variant: str, mode: str) -> list[Resident]:
        key = (scan, variant, mode)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        spec = self.meta[scan]["variants"][variant]
        if spec.get("flash_xml") and not self.has_nocaustics:
            raise ValueError(f"variant {variant!r} needs the path_nocaustics integrator, absent from this Mitsuba build")
        while len(self.cache) >= self.max_resident:
            self.cache.popitem(last=False)
            gc.collect()
            # Dr.Jit keeps freed GPU buffers in its allocation cache; hand them back before the next scene
            # loads, or the old scene's memory and the new one's add up (28 of 32.6 GB on Device 1 while a
            # variant switch loaded, with another process holding 8.8 GB) and freeze recordings fail
            self.dr.flush_malloc_cache()
        base = self.pack / "scenes" / scan  # XML asset paths are relative to this directory
        with ACTIVITY.doing(f"loading {scan} {variant}"):
            if spec.get("flash_xml"):  # two passes: passive scene + flash-only scene, summed
                loaded = [Resident(self.mi, base / spec["xml"], mode, freeze=self.freeze),
                          Resident(self.mi, base / spec["flash_xml"], mode, flash=spec["flash"], freeze=self.freeze)]
            else:  # one pass; an older active scene carries its flash, which follows the camera
                loaded = [Resident(self.mi, base / spec["xml"], mode, flash=spec.get("flash"), freeze=self.freeze)]
        self.cache[key] = loaded
        return loaded

    def _denoise(self, mode: str, img: np.ndarray) -> np.ndarray:
        """OptiX AI denoiser on one non-negative HxWx3 radiance image (runs on the mode's worker)."""
        h, w = img.shape[:2]
        key = (mode, w, h)
        if key not in self.denoisers:
            self.denoisers[key] = self.mi.OptixDenoiser(input_size=self.mi.ScalarVector2u(w, h))
        noisy = self.mi.TensorXf(np.ascontiguousarray(np.clip(np.nan_to_num(img), 0.0, None), dtype=np.float32))
        return np.array(self.denoisers[key](noisy))[:, :, :3]

    def _check(self, view: dict) -> tuple:
        """Validate one view; returns (group key, camera_to_world)."""
        scan, variant = view["scan"], view["variant"]
        mode, denoise = view.get("mode", "polar"), bool(view.get("denoise", False))
        if scan not in self.meta:
            raise ValueError(f"unknown scan {scan!r}")
        if variant not in self.meta[scan]["variants"]:
            raise ValueError(f"unknown variant {variant!r} for {scan}")
        if mode not in self.variant_of:
            raise ValueError(f"mode {mode!r} unavailable on this server (have {sorted(self.variant_of)})")
        c2w = np.asarray(view["camera_to_world"], dtype=np.float64).reshape(4, 4)
        width, height = int(view["width"]), int(view["height"])
        spp, seed, hfov = int(view["spp"]), int(view.get("seed", 0)), float(view["hfov_deg"])
        if not (0 < width * height <= MAX_PIXELS and 0 < spp <= MAX_SPP and 1.0 <= hfov <= 170.0):
            raise ValueError("width/height/spp/hfov_deg out of range")
        if not np.all(np.isfinite(c2w)):
            raise ValueError("camera_to_world must be finite")
        return (scan, variant, mode, width, height, hfov, spp, seed, denoise), c2w

    def _start(self, key: tuple, cams: list, submitted: float):
        """Main-thread half one: queue the views on the GPU. Returns the second half (see submit_pipelined)."""
        started = time.perf_counter()
        scan, variant, mode, width, height, hfov, spp, seed, denoise = key
        if self.mi.variant() != self.variant_of[mode]:
            self.mi.set_variant(self.variant_of[mode])
        with ACTIVITY.doing("rendering"):
            residents = self._resident(scan, variant, mode)
            if self.kernel_history:
                self.dr.kernel_history()  # discard kernels launched before this call
            t0 = time.perf_counter()
            if len(cams) == 1:  # the pipelined path
                handles = [r.launch(cams[0], width, height, hfov, spp, seed) for r in residents]
            else:  # batch sensor: completes here
                batch = [r.render_many(cams, width, height, hfov, spp, seed) for r in residents]
            launch_s = time.perf_counter() - t0

        def finish():
            with ACTIVITY.doing("rendering"):
                t1 = time.perf_counter()
                parts: dict[str, float] = {"queue_s": started - submitted, "launch_s": launch_s}
                if len(cams) == 1:
                    done = [r.finish(h) for r, h in zip(residents, handles)]
                    per_resident = [[img] for img, _ in done]
                    breakdowns = [b for _, b in done]
                else:
                    per_resident, breakdowns = batch, [getattr(r, "breakdown", None) or {} for r in residents]
                for b in breakdowns:
                    for k, v in b.items():
                        parts[k] = parts.get(k, 0.0) + v
                render_s = launch_s + time.perf_counter() - t1
                return self._assemble(key, cams, per_resident, parts, render_s)
        return finish

    def _assemble(self, key: tuple, cams: list, per_resident: list, parts: dict, render_s: float) -> tuple:
        scan, variant, mode, width, height, hfov, spp, seed, denoise = key
        t1 = time.perf_counter()
        kernels = {"kernels": 0, "codegen_s": 0.0, "compile_s": 0.0, "gpu_s": 0.0}
        if self.kernel_history:
            history = self.dr.kernel_history()
            kernels = {  # milliseconds in Dr.Jit's history -> seconds
                "kernels": len(history),
                "codegen_s": sum(k.get("codegen_time", 0.0) for k in history) / 1000.0,
                "compile_s": sum(k.get("backend_time", 0.0) for k in history) / 1000.0,
                "gpu_s": sum(k.get("execution_time", 0.0) for k in history) / 1000.0,
            }
        outs = []
        for i in range(len(cams)):
            passes = [views[i] for views in per_resident]
            if mode == "rgb":
                out = {"radiance": sum(p[:, :, :3] for p in passes)}
                if denoise:
                    out["radiance"] = self._denoise(mode, out["radiance"])
            else:
                rgb, s0, s1, s2, s3 = stokes.split_channels(passes[0])
                for extra in passes[1:]:  # active = passive (path) + flash (path_nocaustics), linear Stokes sum
                    _, f0, f1, f2, f3 = stokes.split_channels(extra)
                    s0, s1, s2, s3 = s0 + f0, s1 + f1, s2 + f2, s3 + f3
                    rgb = s0
                if denoise:
                    # An intensity denoiser cannot take signed S1/S2: denoise what linear polarizers at
                    # 0/90/45/135 degrees would see, then recombine (S3 stays as rendered).
                    i0, i90 = self._denoise(mode, (s0 + s1) / 2), self._denoise(mode, (s0 - s1) / 2)
                    i45, i135 = self._denoise(mode, (s0 + s2) / 2), self._denoise(mode, (s0 - s2) / 2)
                    s0, s1, s2 = (i0 + i90 + i45 + i135) / 2, i0 - i90, i45 - i135
                    rgb = s0
                out = {"rgb_src": rgb, "s0": s0, "s1": s1, "s2": s2, "s3": s3}
            outs.append(out)
        kernels.update(parts)
        kernels["split_s"] = time.perf_counter() - t1
        return outs, kernels, render_s, time.perf_counter() - t1

    @staticmethod
    def _finish(mode: str, out: dict, dtype, preview: bool = True) -> dict[str, np.ndarray]:
        """``preview=False`` skips the dataset preview recipe (about 40 ms at 512x384); the client then
        tonemaps the radiance itself (MatterSim.setPreviewEnabled)."""
        if mode == "rgb":
            rad = out["radiance"]
            result = {"radiance": rad.astype(dtype)}
            if preview:
                result["rgb"] = stokes.rgb_preview(rad, rad, np.zeros_like(rad), np.zeros_like(rad), percentile=0.995, blur=0.0)
            return result
        result = {k: out[k].astype(dtype) for k in ("s0", "s1", "s2", "s3")}
        if preview:
            result["rgb"] = stokes.rgb_preview(out["rgb_src"], out["s0"], out["s1"], out["s2"])
        return result

    def render_views(self, views: list[dict], dtype) -> tuple[list[dict[str, np.ndarray]], dict[str, float]]:
        """Consecutive views with the same scan, variant, mode, size, spp, seed and denoiser share one
        render call (see Resident.render_many). Timing sums over calls."""
        checked = [self._check(v) for v in views]
        results, total = [], {}
        i = 0
        while i < len(checked):
            key = checked[i][0]
            j = i + 1
            while j < len(checked) and checked[j][0] == key:
                j += 1
            outs, kernels, render_s, denoise_s = self.executor.submit_pipelined(
                self._start, key, [c for _, c in checked[i:j]], time.perf_counter()).result()
            t0 = time.perf_counter()
            results.extend(self._finish(key[2], out, dtype, bool(views[i + n].get("preview", True)))
                           for n, out in enumerate(outs))
            timing = {"render_s": render_s, "denoise_s": denoise_s if key[8] else 0.0,
                      "post_s": time.perf_counter() - t0, **kernels, "calls": 1,
                      # what is left of the render call once kernels are accounted for: tracing the scene
                      # into JIT IR, plus readback. dr.freeze removes the tracing, codegen and compile parts.
                      "trace_s": (max(0.0, render_s - kernels["codegen_s"] - kernels["compile_s"] - kernels["gpu_s"])
                                  if self.kernel_history else 0.0)}
            for k, v in timing.items():
                total[k] = total.get(k, 0.0) + v
            i = j
        total["views"] = len(views)
        return results, total

    def render(self, view: dict, dtype) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        results, timing = self.render_views([view], dtype)
        return results[0], timing


def make_handler(renderer: Renderer, token: str | None = None):
    pack = renderer.pack

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _authorized(self) -> bool:
            if not token:
                return True
            sent = self.headers.get("Authorization", "")
            if hmac.compare_digest(sent.encode(), f"Bearer {token}".encode()):
                return True
            self._json(401, {"error": "missing or wrong bearer token"})
            return False

        def log_message(self, fmt, *args):  # quieter than the default per-request stderr line
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json", headers: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload) -> None:
            self._send(code, json.dumps(payload).encode())

        def do_GET(self):
            if not self._authorized():
                return
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            try:
                if parts == ["v1", "status"]:  # cheap; answers while a render, load or recording runs
                    resident = list(renderer.cache.items())
                    return self._json(200, {
                        "activity": ACTIVITY.snapshot(), "freeze": renderer.freeze,
                        "resident": [list(k) for k, _ in resident],
                        # freeze recordings per resident scene: [spp, width, height] of each single-view recording
                        "recorded": {"|".join(k): sorted({tuple(f) for r in rs for f in r.frozen if f[0] != "batch"})
                                     for k, rs in resident}})
                if parts == ["v1", "info"]:
                    return self._json(200, {
                        "server": "opticalnav_sim", "mitsuba_variant": renderer.variant,
                        "modes": renderer.variant_of, "path_nocaustics": renderer.has_nocaustics,
                        "drjit_freeze": renderer.freeze, "spp_chunk": SPP_CHUNK,
                        "scans": {s: renderer.variants(s) for s in renderer.meta},
                        "resident": [list(k) + [round(sum(r.load_s for r in v), 1)] for k, v in renderer.cache.items()],
                    })
                if parts == ["v1", "scans"]:
                    return self._json(200, sorted(renderer.meta))
                if len(parts) == 4 and parts[:2] == ["v1", "scans"] and parts[2] in renderer.meta:
                    scan = parts[2]
                    if parts[3] == "meta":
                        meta = dict(renderer.meta[scan])
                        meta["variants_available"] = renderer.variants(scan)
                        return self._json(200, meta)
                    if parts[3] == "connectivity":
                        body = (pack / "connectivity" / f"{scan}_connectivity.json").read_bytes()
                        return self._send(200, body)
                self._json(404, {"error": f"no route {self.path}"})
            except Exception as exc:  # noqa: BLE001 - report, keep serving
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self):
            if not self._authorized():
                return
            if self.path.split("?")[0].rstrip("/") != "/v1/render":
                return self._json(404, {"error": f"no route {self.path}"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise ValueError(f"request body must be 1..{MAX_BODY} bytes")
                req = json.loads(self.rfile.read(length))
                views = req["views"]
                if not 0 < len(views) <= MAX_VIEWS:
                    raise ValueError(f"1..{MAX_VIEWS} views per request")
                dtype = {"float16": np.float16, "float32": np.float32}[req.get("stokes_dtype", "float16")]
                started = time.time()
                results, total = renderer.render_views(views, dtype)
                arrays = {f"{k}_{i}": v for i, result in enumerate(results) for k, v in result.items()}
                t_enc = time.perf_counter()
                buf = io.BytesIO()
                np.savez(buf, **arrays)
                total["encode_s"] = time.perf_counter() - t_enc
                self._send(200, buf.getvalue(), "application/octet-stream",
                           {"X-Render-Seconds": f"{time.time() - started:.4f}",
                            "X-Render-Timing": json.dumps({k: round(v, 4) for k, v in total.items()})})
            except (KeyError, ValueError, TypeError) as exc:
                self._json(400, {"error": f"{type(exc).__name__}: {exc}"})
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, help="scene pack directory (tools/build_pack.py output)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18770)
    ap.add_argument("--variant", default=None,
                    help="polar-mode Mitsuba variant (default: cuda_rgb_polarized, else cuda_ad_rgb_polarized)")
    ap.add_argument("--rgb-variant", default=None, help="rgb-mode variant (default: cuda_rgb, else cuda_ad_rgb)")
    ap.add_argument("--modes", default="polar",
                    help="render modes this process serves: polar, rgb or polar,rgb. Two Mitsuba variants in one "
                         "process crashed on the OptiX 7 build, so run one server per mode there")
    ap.add_argument("--max-resident", type=int, default=2, help="scene/variant/mode triples kept loaded")
    ap.add_argument("--preload", action="append", default=[], help="SCAN[:VARIANT[:MODE]] to load at start (repeatable)")
    ap.add_argument("--freeze", choices=["auto", "on", "off"], default="auto",
                    help="replay renders with dr.freeze (auto: when this Dr.Jit has it)")
    ap.add_argument("--kernel-history", action="store_true",
                    help="record per-kernel codegen/compile/GPU times (X-Render-Timing gpu_s etc.). Reading them "
                         "synchronises the GPU every view, so views no longer overlap; use it for measurements")
    ap.add_argument("--token", default=os.environ.get("OPTICALNAV_SIM_TOKEN"),
                    help="require 'Authorization: Bearer <token>' (default $OPTICALNAV_SIM_TOKEN)")
    args = ap.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.token:
        raise SystemExit("refusing to listen beyond localhost without --token (HTTP has no other access control)")
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    if not modes or any(m not in MODE_VARIANTS for m in modes):
        raise SystemExit(f"--modes must name polar and/or rgb, got {args.modes!r}")
    renderer = Renderer(Path(args.pack).resolve(), args.variant, args.max_resident, args.freeze, args.rgb_variant, modes,
                        kernel_history=args.kernel_history)
    for item in args.preload:
        parts = item.split(":")
        scan, variant, mode = parts[0], (parts[1:2] or ["base"])[0], (parts[2:3] or [modes[0]])[0]
        print(f"[preload] {scan}:{variant}:{mode} in {renderer.preload(scan, variant, mode):.1f}s", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(renderer, args.token))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[serve] http://{args.host}:{args.port}  modes={renderer.variant_of}  scans={len(renderer.meta)}  "
          f"path_nocaustics={renderer.has_nocaustics}  freeze={renderer.freeze}  "
          f"kernel_history={renderer.kernel_history}", flush=True)
    try:
        renderer.executor.run_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
