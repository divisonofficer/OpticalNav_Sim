"""Browser GUI for driving the simulator by hand.

    python -m opticalnav_sim.gui --pack packs/opticalnav-v0.2 --server http://127.0.0.1:18770 [--port 18780]

Open http://127.0.0.1:18780. The page talks to this process, which holds one
MatterSim session and turns its Stokes frames into JPEGs (RGB preview, DoLP,
AoLP, S1/S0, S2/S0) and per-pixel readouts.

Rendering runs on its own, like a game loop: inputs (moves, turns, settings)
only change the simulator state and return at once, while a background loop
keeps ``--depth`` passes of the newest camera in flight on the render server.
Each pass is ``--pass-spp`` samples with its own seed; while the camera stays
put, passes accumulate into a running mean up to ``--target-spp``, and the next
input starts over. With two passes in flight the render server queues the next
view's kernels before reading back the last one, so its CPU work hides behind
the GPU. The page long-polls /api/state for new frames. Needs numpy and Pillow.
"""
from __future__ import annotations

import argparse
import contextlib
import hmac
import io
import json
import math
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

from . import MatterSim, frames, stokes, tonemap
from .client import RenderClient

PAGE = Path(__file__).with_name("gui.html")
ACCUMULATED = ("s0", "s1", "s2", "nir_s0", "nir_s1", "nir_s2")  # summed over passes, shown as their mean
CHANNELS = ("rgb", "dolp", "aolp", "s1", "s2", "nir", "nir_dolp", "nir_aolp")  # the older pane names
PANE_BANDS = ("rgb", "r", "g", "b", "nir")  # rgb: colour image, luminance polarization (the dataset's)
PANE_VIEWS = ("img", "dolp", "aolp", "s1", "s2")
_OLD_PANES = {"rgb": ("rgb", "img"), "dolp": ("rgb", "dolp"), "aolp": ("rgb", "aolp"), "s1": ("rgb", "s1"),
              "s2": ("rgb", "s2"), "nir": ("nir", "img"), "nir_dolp": ("nir", "dolp"), "nir_aolp": ("nir", "aolp")}


def pane_channel(name: str) -> tuple[str, str]:
    """A pane's (band, view): ``"<band>.<view>"`` (PANE_BANDS x PANE_VIEWS, e.g. ``"g.aolp"``) or an older
    name (CHANNELS). KeyError for anything else."""
    if name in _OLD_PANES:
        return _OLD_PANES[name]
    band, _, view = str(name).partition(".")
    if band not in PANE_BANDS or view not in PANE_VIEWS:
        raise KeyError(name)
    return band, view


def channel_image(frame: dict, channel: str, scale: float) -> np.ndarray:
    """uint8 RGB image of one channel; ``scale`` is the DoLP (and |S1/S0|, |S2/S0|) shown at full colour.
    ``rgb`` and ``nir`` are the tonemapped images Session.image puts in the frame."""
    if channel == "rgb":
        return frame["rgb"]
    if channel.startswith("nir"):
        if "nir_grey" not in frame:  # the near-infrared band is not being rendered: a dark pane
            return np.full(frame["s0_luma"].shape + (3,), 24, np.uint8)
        if channel == "nir":
            return frame["nir"]
        n = frame["nir_derived"]
        return stokes.dolp_rgb(n["dolp"], scale) if channel == "nir_dolp" else stokes.aolp_rgb(n["aolp_deg"], n["dolp"], scale)
    d = frame["derived"]
    if channel == "dolp":
        return stokes.dolp_rgb(d["dolp"], scale)
    if channel == "aolp":
        return stokes.aolp_rgb(d["aolp_deg"], d["dolp"], scale)
    if channel in ("s1", "s2"):
        return stokes.signed_rgb(d[f"{channel}_over_s0"] / scale)
    raise KeyError(channel)


def room_map_r2r(data: dict) -> dict:
    """A robomituba room_map.v1 (dataset x, y) in the R2R frame the map draws: (x, -y)."""
    def pt(p):
        return [round(float(p[0]), 3), round(-float(p[1]), 3)]

    rooms = [{"id": r.get("id"), "type": r.get("type") or "room", "area_m2": r.get("area_m2"),
              "centroid": pt(r["centroid"]) if r.get("centroid") else None,
              "polygons": [[pt(p) for p in poly] for poly in r.get("polygons") or ([r["polygon"]] if r.get("polygon") else [])]}
             for r in data.get("rooms", [])]
    links = [{"rooms": c.get("rooms"), "kind": c.get("kind"), "midpoint": pt(c["midpoint"]),
              "width_m": c.get("passage_width_m")} for c in data.get("connections", []) if c.get("midpoint")]
    return {"rooms": rooms, "connections": links}


def jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img)).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class Session:
    """One simulator episode driven by the page, rendered by a background loop.

    ``lock`` serialises changes to the MatterSim state (inputs). ``cv`` guards everything the page and the
    render loop share (current view, accumulation, frames, the published snapshot) and wakes the loop and
    long-polling pages. A change of camera or settings bumps ``gen``; passes still in flight for an older
    ``gen`` are dropped when they arrive.
    """

    FLY_KEYS = ("forward", "right", "up", "yaw_deg", "pitch_deg")

    def __init__(self, sim: MatterSim.Simulator, scans: dict[str, list[str]], pass_spp: int = 16,
                 target_spp: int = 1024, turn_deg: float = 15.0, pitch_deg: float = 10.0, depth: int = 2):
        self.sim, self.scans = sim, scans
        self.pass_spp, self.target_spp = int(pass_spp), int(target_spp)
        self.turn, self.pitch = math.radians(turn_deg), math.radians(pitch_deg)
        self.depth = max(1, int(depth))
        sim.renderingEnabled = False  # MatterSim only tracks state; this session renders (see _view)
        sim.setPreviewEnabled(False)  # the page tonemaps the running mean itself
        self.tones: dict[tuple[str, str], dict] = {}  # fixed display exposure per (scene, variant)
        self.envmaps: list[str] = []  # environment maps the render server offers (main sets them)
        self.room_maps: dict[str, dict | None] = {}  # scene -> floor plan for the map (room_map)
        self.server_info: dict = {}  # the render server's Mitsuba version and variant (main sets them)
        self.nir_available = False  # the render server has the rgb2nir plugin (main sets it)
        self.lock = threading.Lock()
        self.cv = threading.Condition()
        self.version = 0
        self.gen = 0
        self.view: dict | None = None     # render request for the current camera
        self.gen_started = 0.0
        self.issued = self.inflight = self.passes = 0
        self.published = 0                # passes in the newest published frame of this gen
        self.acc: dict[str, np.ndarray] = {}
        self.auto = True                  # accumulate while the camera stays put
        self.error = ""
        self.started = False
        self.pose: dict = {}
        self.frame_meta: dict = {}
        self.frame_id = 0
        self.frames: dict[int, dict] = {}  # the last four, so image requests that lag a frame or two still work
        self.arrivals: deque = deque(maxlen=64)
        self.timings: deque = deque(maxlen=64)  # (arrival, the pass's time split) of the frames shown
        self.trail: list[tuple[float, float]] = []
        self._local = threading.local()
        self._pool = ThreadPoolExecutor(self.depth, thread_name_prefix="pass")
        self._loop_thread = threading.Thread(target=self._loop, name="render-loop", daemon=True)
        self._loop_thread.start()
        self.renderer: dict = {}          # the render server's /v1/status, refreshed every second
        threading.Thread(target=self._watch_renderer, name="renderer-watch", daemon=True).start()

    # --- inputs: change the simulator state, never wait for a render ---
    def run(self, what: str, fn, *args) -> dict:
        with self.lock:
            fn(*args)
        return self.status()

    def start(self, scene: str, viewpoint: str | None = None, heading: float | None = None) -> None:
        graph = self.sim._graph(scene)
        if viewpoint is None:
            viewpoint = next(v for v, ok in zip(graph.ids, graph.included) if ok)
        if heading is None and getattr(graph, "headings", None):  # a spawn point's own heading
            heading = graph.headings[graph.index(viewpoint)]
        h = self.sim.getState()[0].heading if heading is None and self.started else (heading or 0.0)
        have = self.scans.get(scene, [])
        if have and self.sim.variant not in have:  # e.g. the light was on and this scene has no active variant
            nearer = {"active_polar": ["perturbed", "base"], "perturbed": ["base", "active_polar"],
                      "base": ["perturbed", "active_polar"]}.get(self.sim.variant, [])
            self.sim.setVariant(next((v for v in nearer if v in have), have[0]))
        self.trail = []
        self.sim.newEpisode([scene], [viewpoint], [h], [0.0])
        self._moved()

    def goto(self, x: float, y: float) -> None:
        """Jump to the graph viewpoint nearest to (x, y), keeping heading."""
        st = self.sim.getState()[0]
        graph = self.sim._graph(st.scanId)
        near = min((i for i, ok in enumerate(graph.included) if ok),
                   key=lambda i: math.hypot(graph.pos[i][0] - x, graph.pos[i][1] - y))
        self.sim.newEpisode([st.scanId], [graph.ids[near]], [st.heading], [st.elevation])
        self._moved()

    def free_pose(self) -> list[float]:
        """The camera's optical centre [x, y, height] in the connectivity (R2R) frame."""
        st = self.sim.getState()[0]
        if st._free_position is not None:
            return [float(v) for v in st._free_position]
        c = self._camera(st)  # dataset convention: (x, height, y_dataset), and y_r2r = -y_dataset
        return [float(c[0, 3]), -float(c[2, 3]), float(c[1, 3])]

    def fly(self, forward: float = 0.0, right: float = 0.0, up: float = 0.0, yaw_deg: float = 0.0,
            pitch_deg: float = 0.0) -> None:
        """Turn by ``yaw_deg`` / ``pitch_deg``, then move ``forward`` / ``right`` / ``up`` metres along the new
        heading, off the graph (MatterSim.teleport). No collision check: the camera can pass through walls."""
        st = self.sim.getState()[0]
        x, y, z = self.free_pose()
        h = st.heading + math.radians(yaw_deg)
        x += forward * math.sin(h) + right * math.cos(h)  # heading 0 looks along +y; positive turns right
        y += forward * math.cos(h) - right * math.sin(h)
        z = min(max(z + up, 0.2), 3.0)
        e = min(max(st.elevation + math.radians(pitch_deg), -0.9), 0.9)
        self.sim.teleport([st.scanId], [[x, y, z]], [h], [e])
        self._moved()

    def act(self, op: str, value=0.0) -> None:
        st = self.sim.getState()[0]
        if op == "fly":
            return self.fly(**{k: float(v) for k, v in dict(value or {}).items() if k in self.FLY_KEYS})
        if op == "free_off":  # back onto the graph at the viewpoint nearest the camera
            x, y, _ = self.free_pose()
            return self.goto(x, y)
        if op == "forward":  # the candidate nearest the image centre, resolved now
            op, value = "move", 1
        if op == "move":
            index = int(value)
            if not 0 < index < len(st.navigableLocations):
                raise ValueError(f"no candidate {index} here ({len(st.navigableLocations) - 1} visible)")
            self.sim.makeAction([index], [0.0], [0.0])
        elif op == "turn":
            self.sim.makeAction([0], [self.turn * float(value)], [0.0])
        elif op == "about":
            self.sim.makeAction([0], [math.pi], [0.0])
        elif op == "pitch":
            self.sim.makeAction([0], [0.0], [self.pitch * float(value)])
        else:
            raise ValueError(f"unknown action {op!r}")
        self._moved()

    def configure(self, settings: dict) -> None:
        """Variant, lights, environment, pass spp, denoiser restart accumulation; target spp and auto only
        change when it stops."""
        restart = False
        if "variant" in settings:
            scene = self.sim.getState()[0].scanId
            if settings["variant"] not in self.scans.get(scene, []):
                raise ValueError(f"variant {settings['variant']!r} is not available for {scene}")
            self.sim.setVariant(settings["variant"])
            restart = True
        if "lights" in settings:
            self.sim.setIndoorLights(float(settings["lights"]))
            restart = True
        if "mirrors" in settings:  # None: the variant's own; else {"density", "glass", "glass_blocking"}
            m = settings["mirrors"]
            self.sim.setMirrors(None if m is None else m.get("density", 1.0), int((m or {}).get("glass", 0)),
                                bool((m or {}).get("glass_blocking", False)))
            restart = True
        if "lighting" in settings:
            self.sim.setLighting(str(settings["lighting"]))
            restart = True
        if "lamps" in settings:
            self.sim.setLamps(bool(settings["lamps"]))
            restart = True
        if "flash_scale" in settings:
            self.sim.setFlashScale(float(settings["flash_scale"]))
            restart = True
        if "flash_offset" in settings:  # [right, up(, forward)] m from the dataset's flash (MatterSim.setFlashOffset)
            self.sim.setFlashOffset(*(float(v) for v in settings["flash_offset"]))
            restart = True
        if "max_depth" in settings:  # 0: the scene's own (MatterSim.setMaxDepth)
            self.sim.setMaxDepth(int(settings["max_depth"]))
            restart = True
        if "nir_range" in settings:  # [low, high] (MatterSim.setNirRange)
            self.sim.setNirRange(*(float(v) for v in settings["nir_range"]))
            restart = True
        if "active_split" in settings:  # MatterSim.setActiveSplit
            self.sim.setActiveSplit(bool(settings["active_split"]))
            restart = True
        if "caustics" in settings:  # MatterSim.setCaustics
            self.sim.setCaustics(bool(settings["caustics"]))
            restart = True
        if "camera_offset" in settings:  # [right, up(, forward)] m from the scene's mount (MatterSim.setCameraOffset)
            self.sim.setCameraOffset(*(float(v) for v in settings["camera_offset"]))
            restart = True
        if "bands" in settings:
            self.sim.setBands(str(settings["bands"]))
            restart = True
        if "flash_band" in settings:
            self.sim.setFlashBand(str(settings["flash_band"]))
            restart = True
        if {"env", "env_scale", "env_rotate_deg"} & set(settings):
            env = dict(self.sim.env)
            name = str(settings.get("env", env["map"]))
            if name != "original" and name not in self.envmaps:
                raise ValueError(f"unknown environment map {name!r}")
            self.sim.setEnvironment(name, float(settings.get("env_scale", env["scale"])),
                                    float(settings.get("env_rotate_deg", env["rotate_deg"])))
            restart = True
        if "pass_spp" in settings:
            self.pass_spp = int(settings["pass_spp"])
            restart = True
        if settings.get("refit_tone"):  # display only: nothing renders again
            self.refit_tone()
        if "denoise" in settings:
            self.sim.setDenoiser(bool(settings["denoise"]))
            restart = True
        with self.cv:
            if "target_spp" in settings:
                self.target_spp = int(settings["target_spp"])
            if "auto" in settings:
                self.auto = bool(settings["auto"])
            self.version += 1
            self.cv.notify_all()
        if restart and self.started:
            self._moved()

    def refine(self, frame_id: int) -> None:
        """Kept for older pages: the render loop accumulates on its own."""

    # --- the render loop ---
    def _camera(self, st) -> np.ndarray:
        """The camera MatterSim would render for this state (MatterSim._render)."""
        return self.sim._camera(st)

    def _moved(self) -> None:
        """The state changed: describe it, and point the render loop at the new camera."""
        sim = self.sim
        st = sim.getState()[0]
        c2w = self._camera(st)
        st.camera_to_world = c2w
        scene, variant = sim._split(st.scanId)
        view = {"scan": scene, "variant": variant, "camera_to_world": c2w, "width": sim.width, "height": sim.height,
                "hfov_deg": sim._hfov_deg(sim.width, sim.height), "spp": self.pass_spp, "mode": "polar",
                "denoise": False, "preview": False, "s3": False, "lights": sim.lights, "env": dict(sim.env),
                "bands": sim.bands, "flash_band": sim.flash_band, "nir": dict(sim.nir), "lighting": sim.lighting, "lamps": sim.lamps,
                "flash_scale": sim.flash_scale, "flash_offset": list(sim.flash_offset), "max_depth": sim.max_depth,
                "caustics": sim.caustics, "nir_range": list(sim.nir_range), "active_split": sim.active_split,
                "mirrors": dict(sim.mirrors) if sim.mirrors is not None else None}
        pos = st._free_position or (st.location.x, st.location.y)
        if not self.trail or self.trail[-1] != (pos[0], pos[1]):
            self.trail.append((pos[0], pos[1]))
        height = float(c2w[1, 3])
        pose = {
            "scene": st.scanId, "viewpoint": st.location.viewpointId, "step": st.step,
            "heading_deg": round(math.degrees(st.heading), 2), "elevation_deg": round(math.degrees(st.elevation), 2),
            "position": [round(pos[0], 3), round(pos[1], 3)], "free": st._free_position is not None,
            "height_m": round(float(height), 3),
            "candidates": [{"index": i, "viewpoint": c.viewpointId, "x": round(c.x, 3), "y": round(c.y, 3),
                            "rel_heading_deg": round(math.degrees(c.rel_heading), 2),
                            "rel_elevation_deg": round(math.degrees(c.rel_elevation), 2),
                            "distance_m": round(c.rel_distance, 3)}
                           for i, c in enumerate(st.navigableLocations) if i > 0],
            "trail": [[round(x, 3), round(y, 3)] for x, y in self.trail[-400:]],
            "rig": sim.rig(scene),  # the rig editor's camera mount, flash and offsets
            "pack": self.scene_dir(scene).parent.parent.name,  # the scene's pack (several may be served)
        }
        with self.cv:
            self.gen += 1
            self.view, self.gen_started = view, time.time()
            self.issued = self.passes = self.published = 0
            self.acc = {}
            self.error = ""
            self.started = True
            self.pose = pose
            self.version += 1
            self.cv.notify_all()

    def _wants_pass(self) -> bool:
        if self.view is None or self.error or self.inflight >= self.depth:
            return False
        if self.issued == 0:
            return True
        return self.auto and self.issued * self.pass_spp < self.target_spp

    def _loop(self) -> None:
        while True:
            with self.cv:
                while not self._wants_pass():
                    self.cv.wait()
                gen, view, seed = self.gen, self.view, self.issued
                self.issued += 1
                self.inflight += 1
            self._pool.submit(self._pass, gen, view, seed)

    def _client(self) -> RenderClient:
        client = getattr(self._local, "client", None)
        if client is None:  # one connection per worker, so passes travel side by side
            main = self.sim._client
            client = self._local.client = (RenderClient(main.url, token=main.token)
                                          if isinstance(main, RenderClient) else main)
        return client

    def _pass(self, gen: int, view: dict, seed: int) -> None:
        started = time.perf_counter()
        try:
            client = self._client()
            # float32: numpy's float32 -> float16 conversion on the server took 20-29 ms a pass (on the HTTP thread,
            # holding the GIL against the render thread); the bytes it saved cost ~3 ms to send on one machine
            out = client.render([dict(view, seed=int(seed))], "float32")[0]
            timing = dict(getattr(client, "last_timing", {}) or {})
            server_s = float(getattr(client, "last_render_seconds", 0.0) or 0.0)  # the request, on the server
        except Exception as exc:  # noqa: BLE001 - shown on the page; the next input retries
            with self.cv:
                self.inflight -= 1
                if gen == self.gen:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.version += 1
                self.cv.notify_all()
            return
        wall = time.perf_counter() - started  # the render request's round trip
        with self.cv:
            self.inflight -= 1
            self.cv.notify_all()
            if gen != self.gen:
                # the camera moved on while this pass rendered: while moving, still show it if it is newer than
                # what the page has (a game shows the last finished frame), but never accumulate it
                if seed != 0 or gen <= self.frame_meta.get("frame_gen", -1):
                    return
                passes = 1
                mean = {k: np.asarray(out[k], dtype=np.float32) for k in ACCUMULATED if k in out}
            else:
                for k in ACCUMULATED:
                    if k in out:
                        a = np.asarray(out[k], dtype=np.float32)
                        self.acc[k] = a.copy() if k not in self.acc else self.acc[k] + a
                self.passes += 1
                passes = self.passes
                mean = {k: self.acc[k] / passes for k in ACCUMULATED if k in self.acc}
        nir = [mean.get(f"nir_s{k}") for k in range(3)]
        if self.sim.denoise:
            # passes are rendered and accumulated raw; only the mean on screen is denoised (averaging
            # denoised 16 spp passes keeps just what every pass's denoiser agreed on, and smears detail).
            # Both bands in one request: one trip, queued once behind the render server's passes.
            images = {k: mean[k] for k in ("s0", "s1", "s2") if k in mean}
            if nir[0] is not None:
                images.update({f"nir_s{k}": nir[k] for k in range(3)})
            try:
                den = self._client().denoise(images)
                mean.update({k: den[k] for k in ("s0", "s1", "s2") if k in den})
                if nir[0] is not None:
                    nir = [den[f"nir_s{k}"] for k in range(3)]
            except (ConnectionError, OSError, RuntimeError):
                pass  # show the raw mean rather than nothing
        if "s0" not in mean:  # near-infrared only: show it in the RGB pane too, grey
            mean.update({f"s{k}": np.repeat(nir[k][..., None], 3, axis=2) for k in range(3)})
        s0, s1, s2 = mean["s0"], mean["s1"], mean["s2"]
        extra = {}
        if nir[0] is not None:
            extra = {"nir_grey": nir[0], "nir_derived": stokes.derived_mono(nir[0], nir[1], nir[2])}
        # the radiance itself: the page's exposure and the tone fit apply when a pane is sent (Session.image)
        frame = {"s0": s0, "scan": view["scan"], "s0_luma": s0 @ stokes.LUMA, **extra,
                 "derived": stokes.derived(s0, s1, s2), "s1": s1, "s2": s2, "spp": passes * view["spp"], "passes": passes,
                 "variant": view["variant"]}
        # Where a pass's time went (ms). gpu: the server waiting for the GPU after queuing the pass's kernels, the
        # GPU time no CPU work hid (exact kernel times would need a sync per kernel); server_cpu: the rest of its
        # render call plus splitting and encoding; queue: waiting behind other passes on the server's one render
        # thread; transfer: the round trip minus the server's time (network, decoding); viewer: accumulating,
        # denoising, deriving here.
        g = lambda k: 1000.0 * float(timing.get(k, 0.0))
        frame["timing"] = {
            "wall_ms": 1000.0 * (time.perf_counter() - started), "round_trip_ms": 1000.0 * wall,
            "queue_ms": g("queue_s"), "gpu_ms": g("wait_s"),
            "server_cpu_ms": g("render_s") - g("wait_s") + g("split_s") + g("post_s") + g("encode_s"),
            "transfer_ms": max(0.0, 1000.0 * (wall - server_s)), "viewer_ms": 1000.0 * (time.perf_counter() - started - wall)}
        with self.cv:
            if gen == self.gen:
                if passes <= self.published:  # a later pass of this camera got here first
                    return
                self.published = passes
            elif gen <= self.frame_meta.get("frame_gen", -1):  # an older camera, and something newer is shown
                return
            self.frame_id += 1
            self.frames[self.frame_id] = frame
            for old in [k for k in self.frames if k < self.frame_id - 3]:
                del self.frames[old]
            self.arrivals.append(time.time())
            self.timings.append((time.time(), frame["timing"]))
            self.frame_meta = {"frame_id": self.frame_id, "frame_gen": gen, "frame_spp": frame["spp"],
                               "frame_variant": frame["variant"], "passes": passes}
            self.version += 1
            self.cv.notify_all()

    def _tone(self, scene: str, variant: str, radiance: np.ndarray, own_frame: bool = False) -> dict:
        """Fixed display exposure for a scene and variant: robomituba's scene-global extended Reinhard,
        from the pack's 1024 spp dataset frames, so brightness never jumps from frame to frame (a lamp or
        window entering the view). Without stored frames, the first frame rendered decides."""
        key = (scene, variant)
        if key not in self.tones and own_frame:  # no dataset frames for this band: its first frame decides
            self.tones[key] = tonemap.global_params([radiance])
        if key not in self.tones:
            try:
                self.tones[key] = tonemap.scene_params(self.scene_dir(scene), variant)
            except (OSError, ValueError, KeyError):
                self.tones[key] = tonemap.global_params([radiance])
        return self.tones[key]

    # --- what the page reads ---
    def wait(self, after: int, timeout: float) -> None:
        """Block until something changed after ``version`` ``after`` (long poll), or ``timeout``."""
        with self.cv:
            self.cv.wait_for(lambda: self.version > after, timeout=max(0.0, min(timeout, 30.0)))

    def status(self) -> dict:
        sim = self.sim
        with self.cv:
            now = time.time()
            # frames shown in the last 5 s (2 s jumped with every slow pass); the time split as medians over them
            recent = [t for t in self.arrivals if now - t <= 5.0]
            fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 2 and recent[-1] > recent[0] else 0.0
            split = [s for t, s in self.timings if now - t <= 5.0] or [s for _, s in list(self.timings)[-1:]]
            timing = {k: round(float(np.median([s[k] for s in split])), 1) for k in (split[0] if split else {})}
            waiting = (self.frame_meta.get("frame_gen") != self.gen) and self.view is not None and not self.error
            spp_now = self.frame_meta.get("frame_spp", 0) if self.frame_meta.get("frame_gen") == self.gen else 0
            out = {"scans": self.scans, "pass_spp": self.pass_spp, "target_spp": self.target_spp, "auto": self.auto,
                   "denoise": sim.denoise, "variant": sim.variant, "started": self.started,
                   "lights": sim.lights, "env": sim.env, "envmaps": self.envmaps,
                   "bands": sim.bands, "flash_band": sim.flash_band, "nir_available": self.nir_available,
                   "lighting": sim.lighting, "lamps": sim.lamps, "flash_scale": sim.flash_scale,
                   "flash_offset": list(sim.flash_offset), "mirrors": sim.mirrors, "max_depth": sim.max_depth,
                   "caustics": sim.caustics, "nir_range": list(sim.nir_range), "server_info": self.server_info,
                   "active_split": sim.active_split,
                   "has_frame": bool(self.frame_meta),
                   "error": self.error, "version": self.version,
                   "turn_deg": round(math.degrees(self.turn), 3), "pitch_deg": round(math.degrees(self.pitch), 3),
                   "hfov_deg": sim._hfov_deg(sim.width, sim.height), "vfov_deg": math.degrees(sim.vfov),
                   "width": sim.width, "height": sim.height, **self.pose, **self.frame_meta,
                   "done": spp_now >= self.target_spp or (not self.auto and spp_now > 0),
                   "stale": waiting, "fps": round(fps, 1), "in_flight": self.inflight, "timing": timing}
            waited = now - self.gen_started if waiting else 0.0
            info = self.renderer or {}
            out["renderer"] = {"reachable": info.get("reachable"), "activity": self._renderer_status(),
                               "resident": info.get("resident", []), "freeze_available": info.get("freeze"),
                               "ready": self.readiness()}
            # the scene the render server holds, or is still preloading (a page opened during --preload waits for it)
            resident = info.get("resident") or info.get("preloading") or []
            out["default_scene"] = resident[0][0] if resident else None
        server = {"busy": waiting, "what": "render", "waiting": 0, "elapsed_s": round(waited, 1)}
        if waiting and waited > 1.0:
            server["renderer"] = out["renderer"]["activity"]
        out["server"] = server
        return out

    def _watch_renderer(self, period: float = 1.0) -> None:
        """Poll the render server's status (reachable, loaded scenes, freeze recordings, activity) and publish
        it to the page whenever it changes."""
        client = None
        while True:
            try:
                if client is None:
                    main = self.sim._client
                    if not isinstance(main, RenderClient):
                        return  # a test double: nothing to watch
                    client = RenderClient(main.url, timeout=2.0, token=main.token)
                info = {"reachable": True, **client.status()}
            except Exception as exc:  # noqa: BLE001 - down, restarting, or an older server without /v1/status
                client = None
                info = {"reachable": False, "error": f"{type(exc).__name__}"}
            elapsed = (info.get("activity") or {}).pop("elapsed_s", None)
            with self.cv:
                changed = info != self.renderer
                self.renderer = info
                self.renderer_elapsed = elapsed
                if changed:
                    self.version += 1
                    self.cv.notify_all()
            time.sleep(period)

    def _renderer_status(self) -> dict:
        info = self.renderer or {}
        if not info.get("reachable", True):
            return {"error": "unreachable"}
        return {**(info.get("activity") or {}), "elapsed_s": getattr(self, "renderer_elapsed", 0.0) or 0.0}

    def readiness(self) -> dict:
        """Is the current view's scene loaded on the render server, and is its freeze recorded?"""
        info, view = self.renderer or {}, self.view
        if not info.get("reachable") or view is None:
            return {"scene": None, "freeze": None}
        key = f"{view['scan']}|{view['variant']}|{view['mode']}"
        resident = any("|".join(k) == key for k in info.get("resident", []))
        if "recorded" not in info or not info.get("freeze"):  # an older render server does not report recordings
            return {"scene": resident, "freeze": None}
        have, base = info["recorded"].get(key, []), int(info.get("freeze_spp") or 0)
        recorded = [view["spp"], view["width"], view["height"]] in have or (
            base > 0 and view["spp"] % base == 0 and [base, view["width"], view["height"]] in have)  # replayed repeatedly
        return {"scene": resident, "freeze": recorded}

    def _frame(self, frame_id: int) -> dict:
        with self.cv:
            return self.frames[frame_id]

    def wait_frame(self, after_frame: int = 0, timeout: float = 10.0) -> dict:
        """Wait for a frame of the current camera newer than ``after_frame`` (for scripts and tests)."""
        end = time.time() + timeout
        with self.cv:
            while not (self.frame_meta.get("frame_gen") == self.gen and self.frame_id > after_frame) and not self.error:
                left = end - time.time()
                if left <= 0:
                    raise TimeoutError("no frame")
                self.cv.wait(left)
        return self.status()

    def wait_idle(self, timeout: float = 30.0) -> dict:
        """Wait until the current camera reached its target spp (or accumulation is off)."""
        end = time.time() + timeout
        while True:
            st = self.status()
            if st.get("done") or st.get("error"):
                return st
            if time.time() > end:
                raise TimeoutError("not idle")
            self.wait(st["version"], end - time.time())

    POL_CHANNELS = ("luma", "r", "g", "b", "nir")  # probe channels (luma: the dataset's luminance Stokes)

    def _polarization(self, frame: dict, pol: str) -> dict | None:
        """DoLP, AoLP, S1/S0, S2/S0 of one channel: "luma" (the dataset's definition, from luminance Stokes),
        "r", "g", "b" (that channel's own Stokes vector) or "nir". None when the frame has no such band."""
        if pol in ("luma", None, ""):
            return frame["derived"]
        if pol == "nir":
            return frame.get("nir_derived")
        k = "rgb".index(pol)
        return stokes.derived_mono(frame["s0"][..., k], frame["s1"][..., k], frame["s2"][..., k])

    def image(self, frame_id: int, channel: str, scale: float, ev: dict | None = None) -> bytes:
        """A pane's JPEG. ``channel``: a band and a view (pane_channel), e.g. ``"g.aolp"``: the image (the colour
        image for ``rgb``, the band's own S0 in grey for ``r``, ``g``, ``b``, the near-infrared image) or that
        band's DoLP, AoLP, S1/S0, S2/S0 (``rgb``: from luminance Stokes, as the dataset defines them). ``ev``:
        the page's exposure offset per band ({"rgb": stops, "nir": stops}) on top of the scene's tone fit.
        Nothing renders again."""
        band, view = pane_channel(channel)
        frame = self._frame(frame_id)
        ev = ev or {}
        if band == "nir" and "nir_grey" not in frame:  # the near-infrared band is not being rendered
            return jpeg(np.full(frame["s0_luma"].shape + (3,), 24, np.uint8))
        if view == "img":
            if band in ("rgb", "nir"):
                return jpeg(self._tonemapped(frame, band, float(ev.get(band, 0.0))))
            grey = np.repeat(frame["s0"][..., "rgb".index(band), None], 3, axis=2)
            tone = self._tone(frame["scan"], frame["variant"], frame["s0"])
            return jpeg(tonemap.reinhard(grey, tone["exposure"] * 2.0 ** float(ev.get("rgb", 0.0)), tone["white"]))
        derived = self._polarization(frame, "luma" if band == "rgb" else band)
        return jpeg(channel_image(dict(frame, derived=derived), view, scale))

    def _tonemapped(self, frame: dict, band: str, ev: float) -> np.ndarray:
        """Extended Reinhard at the band's tone fit (see _tone), its exposure shifted by ``ev`` stops."""
        if band == "rgb":
            radiance, key = frame["s0"], (frame["scan"], frame["variant"])
        else:
            radiance, key = np.repeat(frame["nir_grey"][..., None], 3, axis=2), (frame["scan"], frame["variant"] + "|nir")
        tone = self._tone(key[0], key[1], radiance, own_frame=band == "nir")
        return tonemap.reinhard(radiance, tone["exposure"] * 2.0 ** ev, tone["white"])

    def refit_tone(self) -> None:
        """Fit both bands' tone to the frame on screen now (instead of the dataset frames, or the band's first
        frame): its exposure puts that frame's own bright percentile at white."""
        with self.cv:
            if not self.frame_meta:
                raise ValueError("no frame to fit the exposure to yet")
            frame = self.frames[self.frame_meta["frame_id"]]
            self.tones[(frame["scan"], frame["variant"])] = tonemap.global_params([frame["s0"]])
            if "nir_grey" in frame:
                grey = np.repeat(frame["nir_grey"][..., None], 3, axis=2)
                self.tones[(frame["scan"], frame["variant"] + "|nir")] = tonemap.global_params([grey])
            self.version += 1
            self.cv.notify_all()

    def probe(self, frame_id: int, x: int, y: int, pol: str = "luma") -> dict:
        f = self._frame(frame_id)
        h, w = f["s0_luma"].shape
        x, y = min(max(int(x), 0), w - 1), min(max(int(y), 0), h - 1)
        d = self._polarization(f, pol) if pol in self.POL_CHANNELS else None
        if d is None:  # an unknown channel, or the near-infrared band not rendered: luminance
            pol, d = "luma", f["derived"]
        s0 = (f["s0_luma"] if pol == "luma" else f["nir_grey"] if pol == "nir" else f["s0"][..., "rgb".index(pol)])[y, x]
        return {"x": x, "y": y, "channel": pol, "s0": float(s0), "dolp": float(d["dolp"][y, x]),
                "aolp_deg": float(d["aolp_deg"][y, x]), "s1_over_s0": float(d["s1_over_s0"][y, x]),
                "s2_over_s0": float(d["s2_over_s0"][y, x])}

    def graph(self, scene: str) -> dict:
        if scene not in self.scans:
            raise KeyError(f"scene {scene!r}")
        with self.lock:
            g = self.sim._graph(scene)
        keep = [i for i, ok in enumerate(g.included) if ok]
        return {"scene": scene, "x": [round(g.pos[i][0], 3) for i in keep], "y": [round(g.pos[i][1], 3) for i in keep],
                "rooms": self.room_map(scene)}

    def scene_dir(self, scene: str) -> Path:
        """The scene's pack directory on this machine: as the render server reports it (it may serve several
        packs), else under the first --pack."""
        with contextlib.suppress(Exception):
            reported = Path(self.sim._scene_meta(scene)["scene_dir"])
            if reported.is_dir():
                return reported
        return Path(self.sim.navGraphPath).parent / "scenes" / scene

    def room_map(self, scene: str) -> dict | None:
        """The scene's floor plan for the map (robomituba tools/infinigen/export_room_map.py, schema
        robomituba.room_map.v1), in the R2R frame the map draws: rooms with type, polygons and centroid, and the
        passages between them. Read from the pack, else from the project the pack was built from; None when
        neither has one."""
        if scene not in self.room_maps:
            scene_dir = self.scene_dir(scene)
            pack = scene_dir.parent.parent
            places = [scene_dir / "room_map.json"]
            try:
                places.append(Path(json.loads((pack / "pack.json").read_text())["project"]) / "scenes" / scene / "room_map.json")
            except (OSError, KeyError, ValueError):
                pass
            path = next((p for p in places if p.is_file()), None)
            self.room_maps[scene] = room_map_r2r(json.loads(path.read_text())) if path else None
        return self.room_maps[scene]


COOKIE = "opticalnav_gui"
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class WebSocket:
    """Just enough RFC 6455 for one browser page: text and binary messages, ping, close (no extensions)."""

    def __init__(self, rfile, wfile):
        self.rfile, self.wfile = rfile, wfile
        self.write_lock = threading.Lock()

    def _read(self, n: int) -> bytes:
        data = self.rfile.read(n)
        if len(data) < n:
            raise ConnectionError("socket closed")
        return data

    def recv(self) -> tuple[int, bytes]:
        """(opcode, payload) of the next message; client frames are always masked."""
        b0, b1 = self._read(2)
        opcode, length = b0 & 0x0F, b1 & 0x7F
        if length == 126:
            length = int.from_bytes(self._read(2), "big")
        elif length == 127:
            length = int.from_bytes(self._read(8), "big")
        mask = self._read(4) if b1 & 0x80 else b"\0\0\0\0"
        data = bytearray(self._read(length))
        for i in range(len(data)):
            data[i] ^= mask[i % 4]
        return opcode, bytes(data)

    def send(self, payload: bytes, opcode: int = 0x1) -> None:
        n = len(payload)
        head = bytes([0x80 | opcode]) + (bytes([n]) if n < 126 else bytes([126]) + n.to_bytes(2, "big") if n < 1 << 16
                                         else bytes([127]) + n.to_bytes(8, "big"))
        with self.write_lock:
            self.wfile.write(head + payload)
            self.wfile.flush()

    def send_json(self, obj) -> None:
        self.send(json.dumps(obj).encode())


def dispatch(session: Session, path: str, body: dict) -> dict:
    """One input, from POST /api/<path> or a socket message: changes the state and returns at once."""
    if path == "start":
        return session.run("start", session.start, body["scene"], body.get("viewpoint"), body.get("heading"))
    if path == "action":
        return session.run(str(body["op"]), session.act, body["op"], body.get("value", 0.0))
    if path == "goto":
        return session.run("goto", session.goto, float(body["x"]), float(body["y"]))
    if path == "configure":
        return session.run("configure", session.configure, body)
    if path == "refine":
        return session.run("pass", session.refine, int(body["frame_id"]))
    raise KeyError(f"no route /api/{path}")


def make_handler(session: Session, token: str | None = None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str, headers: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self, query: dict) -> bool:
            """With a token, a browser opens /?token=<token> once; that sets a cookie the page's own
            requests carry. Scripts may send 'Authorization: Bearer <token>' instead."""
            if not token:
                return True
            jar = SimpleCookie(self.headers.get("Cookie", ""))
            bearer = self.headers.get("Authorization", "")
            sent = [jar[COOKIE].value if COOKIE in jar else "", query.get("token", ""),
                    bearer[7:] if bearer.startswith("Bearer ") else ""]
            if any(hmac.compare_digest(v.encode(), token.encode()) for v in sent):
                return True
            self._json(401, {"error": "open the viewer as /?token=<token> (or send Authorization: Bearer <token>)"})
            return False

        def _json(self, code: int, payload) -> None:
            self._send(code, json.dumps(payload).encode(), "application/json")

        def do_GET(self):
            url = urlsplit(self.path)
            q = {k: v[-1] for k, v in parse_qs(url.query).items()}
            parts = [p for p in url.path.split("/") if p]
            if not self._authorized(q):
                return
            if parts == ["ws"] and self.headers.get("Upgrade", "").lower() == "websocket":
                return self._websocket()
            try:
                if not parts:
                    if token and q.get("token"):  # remember the token, then drop it from the address bar
                        return self._send(303, b"", "text/plain", {
                            "Set-Cookie": f"{COOKIE}={token}; HttpOnly; SameSite=Strict; Path=/", "Location": "/"})
                    return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                if parts == ["api", "state"]:  # ?after=<version>&wait=<s>: long poll until something changes
                    if "after" in q:
                        session.wait(int(q["after"]), float(q.get("wait", 2.0)))
                    return self._json(200, session.status())
                if parts == ["api", "scene_stats"]:  # the scene tab: asks the render server each time
                    return self._json(200, session.sim._client.stats(q["scene"]))
                if parts == ["api", "graph"]:
                    return self._json(200, session.graph(q["scene"]))
                if parts == ["api", "probe"]:
                    return self._json(200, session.probe(int(q["frame"]), int(q["x"]), int(q["y"]), q.get("pol", "luma")))
                if len(parts) == 4 and parts[:2] == ["api", "frame"] and parts[3].endswith(".jpg"):
                    channel = parts[3][:-4]
                    pane_channel(channel)  # KeyError (404) for an unknown pane
                    body = session.image(int(parts[2]), channel, float(q.get("scale", 0.2)),
                                         {"rgb": float(q.get("ev_rgb", 0.0)), "nir": float(q.get("ev_nir", 0.0))})
                    return self._send(200, body, "image/jpeg")
                self._json(404, {"error": f"no route {url.path}"})
            except KeyError as exc:
                self._json(404, {"error": f"not found: {exc}"})
            except (ValueError, TypeError) as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - report to the page instead of a server traceback
                self._json(502, {"error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self):
            url = urlsplit(self.path)
            if not self._authorized({}):
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
                if not url.path.startswith("/api/"):
                    return self._json(404, {"error": f"no route {url.path}"})
                try:
                    out = dispatch(session, url.path[len("/api/"):], body)
                except KeyError as exc:
                    if "no route" in str(exc):
                        return self._json(404, {"error": str(exc).strip("'\"")})
                    raise
                self._json(200, out)
            except (KeyError, ValueError, TypeError) as exc:
                self._json(400, {"error": f"{type(exc).__name__}: {exc}"})
            except Exception as exc:  # noqa: BLE001 - render server failures go to the page
                self._json(502, {"error": f"{type(exc).__name__}: {exc}"})

        def _websocket(self) -> None:
            """GET /ws: the page's one connection. The server pushes {"type": "state"} whenever anything
            changes and, for each new frame, the JPEGs of the panes the page subscribed to as binary messages
            (4-byte frame id, 1-byte pane, JPEG). The page sends {"type": "input", "id", "path", "body"}
            (answered by {"type": "ack"}), {"type": "subscribe", "panes": {"a": "rgb", "b": "dolp"|null},
            "scale"} and {"type": "probe", "frame", "x", "y"}."""
            import base64
            import hashlib

            key = self.headers.get("Sec-WebSocket-Key", "")
            accept = base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode()
            self.send_response(101)
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept)
            self.end_headers()
            self.close_connection = True
            ws = WebSocket(self.rfile, self.wfile)
            sub = {"panes": {"a": "rgb.img", "b": "rgb.dolp"}, "scale": 0.2, "ev": {}, "dirty": True}
            alive = threading.Event()
            alive.set()

            def push() -> None:  # sender: state on every change, pane JPEGs on every new frame
                version, sent_frame = -1, None
                try:
                    while alive.is_set():
                        session.wait(version, 5.0)
                        st = session.status()
                        if st["version"] != version:
                            version = st["version"]
                            ws.send_json({"type": "state", "state": st})
                        frame_id = st.get("frame_id") if st.get("has_frame") else None
                        if frame_id is not None and (frame_id != sent_frame or sub["dirty"]):
                            sub["dirty"] = False
                            sent_frame = frame_id
                            for i, pane in enumerate(("a", "b")):
                                ch = sub["panes"].get(pane)
                                try:
                                    pane_channel(ch)
                                except KeyError:  # no pane (a phone shows one), or an unknown name
                                    ch = None
                                if ch is not None:
                                    try:
                                        img = session.image(frame_id, ch, float(sub["scale"]), sub["ev"])
                                    except KeyError:  # already replaced by a newer frame
                                        break
                                    ws.send(frame_id.to_bytes(4, "big") + bytes([i]) + img, opcode=0x2)
                except (ConnectionError, OSError, ValueError):
                    alive.clear()

            sender = threading.Thread(target=push, name="ws-push", daemon=True)
            sender.start()
            try:
                while alive.is_set():
                    opcode, data = ws.recv()
                    if opcode == 0x8:  # close
                        break
                    if opcode == 0x9:  # ping
                        ws.send(data, opcode=0xA)
                        continue
                    if opcode != 0x1:
                        continue
                    msg = json.loads(data)
                    kind = msg.get("type")
                    if kind == "input":
                        try:
                            dispatch(session, msg["path"], msg.get("body") or {})
                            ws.send_json({"type": "ack", "id": msg.get("id")})
                        except Exception as exc:  # noqa: BLE001 - shown on the page
                            ws.send_json({"type": "ack", "id": msg.get("id"), "error": f"{type(exc).__name__}: {exc}"})
                    elif kind == "subscribe":
                        sub["panes"] = dict(msg.get("panes") or sub["panes"])
                        sub["scale"] = float(msg.get("scale", sub["scale"]))
                        sub["ev"] = {k: float(v) for k, v in dict(msg.get("ev") or {}).items() if k in ("rgb", "nir")}
                        sub["dirty"] = True
                        with session.cv:  # wake the sender to resend this frame's panes
                            session.version += 1
                            session.cv.notify_all()
                    elif kind == "probe":
                        try:
                            ws.send_json({"type": "probe", **session.probe(int(msg["frame"]), int(msg["x"]), int(msg["y"]),
                                                                           str(msg.get("pol", "luma")))})
                        except KeyError:
                            pass
            except (ConnectionError, OSError, ValueError):
                pass
            finally:
                alive.clear()
                with session.cv:
                    session.cv.notify_all()

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path, action="append",
                    help="scene pack (repeatable, as the render server's): graphs are read from the first one's "
                         "connectivity/, the other packs' from the render server")
    ap.add_argument("--server", default=None, help="render server URL (default $OPTICALNAV_SIM_URL)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18780)
    ap.add_argument("--pass-spp", type=int, default=16, help="samples per render pass (moves show one pass)")
    ap.add_argument("--target-spp", type=int, default=1024, help="stop accumulating at this many samples")
    ap.add_argument("--turn-deg", type=float, default=15.0)
    ap.add_argument("--pitch-deg", type=float, default=10.0)
    ap.add_argument("--depth", type=int, default=2, help="passes in flight on the render server (2 overlaps CPU and GPU)")
    ap.add_argument("--token", default=os.environ.get("OPTICALNAV_GUI_TOKEN"),
                    help="required to listen beyond localhost (default $OPTICALNAV_GUI_TOKEN); open /?token=<token>")
    args = ap.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.token:
        raise SystemExit("refusing to listen beyond localhost without --token (anyone could drive the simulator)")

    sim = MatterSim.Simulator()
    if args.server:
        sim.setDatasetPath(args.server)
    sim.setNavGraphPath(str(args.pack[0] / "connectivity"))
    sim.initialize()
    waited = 0
    while True:  # the render server may still be loading its --preload scene (it listens only after that)
        try:
            info = sim._client.info()
            break
        except (ConnectionError, OSError, RuntimeError) as exc:
            if waited % 30 == 0:
                print(f"[gui] waiting for the render server at {sim._client.url} ({type(exc).__name__})", flush=True)
            time.sleep(2)
            waited += 2
    # start on what the render server has loaded, or is preloading (pages start its first scene)
    if info.get("resident") or info.get("preloading"):
        sim.setVariant((info.get("resident") or info["preloading"])[0][1])
    session = Session(sim, info["scans"], args.pass_spp, args.target_spp, args.turn_deg, args.pitch_deg, args.depth)
    session.envmaps = list(info.get("envmaps", []))
    session.server_info = {k: info.get(k) for k in ("mitsuba_version", "mitsuba_variant")}
    session.nir_available = bool(info.get("nir"))
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(session, args.token))
    resident = [r[0] for r in info.get("resident", [])]
    print(f"[gui] http://{args.host}:{args.port}{'/?token=<token>' if args.token else ''}  render server {sim._client.url}  freeze={info.get('drjit_freeze')}  "
          f"resident={resident}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
