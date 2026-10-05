"""Matterport3DSimulator-compatible API over OpticalNav polar renders.

Drop-in for R2R code written against ``import MatterSim``::

    from opticalnav_sim import MatterSim
    sim = MatterSim.Simulator()
    sim.setDatasetPath("http://127.0.0.1:18770")    # render server
    sim.setNavGraphPath("pack/connectivity")        # optional; fetched from the server otherwise
    sim.setDiscretizedViewingAngles(True)
    sim.initialize()
    sim.newEpisode([scan], [viewpoint], [heading], [0])
    state = sim.getState()[0]                      # state.rgb (BGR uint8), state.navigableLocations, ...
    sim.makeAction([1], [0], [0])

Behaviour follows Matterport3DSimulator (src/lib/MatterSim.cpp): heading
snapping, the +/-30 degree discretised actions, the field-of-view candidate
filter and its ordering, and init-only setters. Differences, all documented in
the README:

* ``state.rgb`` is the polar camera's RGB preview (BGR, like MatterSim); the
  polarimetric data is ``state.stokes`` = {"s0", "s1", "s2", "s3"} (float HxWx3).
* ``elevation`` is measured from the dataset camera's pitch (about -8.5
  degrees), so elevation 0 reproduces dataset views exactly.
* The default camera is the dataset's polar camera (512x384, 90 degree HFOV).
* Extensions: ``setVariant``, ``setRenderMode`` ("polar" | "rgb"), ``setRenderSpp``,
  ``setDenoiser``, ``setStokesDtype``, ``teleport`` (free camera poses),
  ``renderCamera`` (raw dataset matrices).
* Depth is not rendered.
"""
from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np

from . import frames, stokes
from .client import RenderClient
from .navgraph import NavGraph, ViewPoint

__all__ = ["Simulator", "SimState", "ViewPoint"]

HEADING_COUNT = 12
ELEVATION_INCREMENT = math.pi / 6.0
VARIANT_SEPARATOR = "__"


class SimState:
    def __init__(self):
        self.scanId = ""
        self.step = 0
        self.rgb = np.zeros((0, 0, 3), np.uint8)
        self.depth = np.zeros((0, 0, 1), np.uint16)
        self.location: ViewPoint | None = None
        self.heading = 0.0
        self.elevation = 0.0
        self.viewIndex = 0
        self.navigableLocations: list[ViewPoint] = []
        # OpticalNav extensions
        self.stokes: dict[str, np.ndarray] | None = None
        self.radiance: np.ndarray | None = None  # linear RGB (= S0 in polar mode)
        self.camera_to_world: np.ndarray | None = None
        self.variant = ""
        self._free_position: tuple[float, float, float] | None = None

    def __repr__(self) -> str:
        loc = self.location.viewpointId if self.location else None
        return (f"SimState(scanId={self.scanId!r}, step={self.step}, location={loc!r}, "
                f"heading={self.heading:.4f}, elevation={self.elevation:.4f}, viewIndex={self.viewIndex}, "
                f"navigable={len(self.navigableLocations)})")


class Simulator:
    def __init__(self):
        self.width, self.height = 512, 384
        self.vfov = 2.0 * math.atan(math.tan(math.radians(90.0) / 2.0) * 384 / 512)
        self.minElevation, self.maxElevation = -0.94, 0.94
        self.navGraphPath = ""
        self.datasetPath = ""
        self.renderingEnabled = True
        self.discretizeViews = False
        self.restrictedNavigation = True
        self.preloadImages = False
        self.renderDepth = False
        self.batchSize = 1
        self.cacheSize = 200
        self.randomSeed = 1
        self.variant = "base"
        self.renderMode = "polar"
        self.denoise = False
        self.preview = True
        self.spp = 64
        self.seed = 0
        self.stokesDtype = "float16"
        self.initialized = False
        self.states: list[SimState] = []
        self._graphs: dict[str, NavGraph] = {}
        self._meta: dict[str, dict] = {}
        self._client: RenderClient | None = None
        self._rng = random.Random(1)
        self._frames = 0
        self._wall = self._render_s = self._server_s = 0.0

    # --- MatterSim setters (init-only ones are ignored after initialize, as in MatterSim) ---
    def setDatasetPath(self, path: str) -> None:
        if not self.initialized:
            self.datasetPath = str(path)

    def setNavGraphPath(self, path: str) -> None:
        if not self.initialized:
            self.navGraphPath = str(path)

    def setRenderingEnabled(self, value: bool) -> None:
        if not self.initialized:
            self.renderingEnabled = bool(value)

    def setCameraResolution(self, width: int, height: int) -> None:
        self.width, self.height = int(width), int(height)

    def setCameraVFOV(self, vfov: float) -> None:
        self.vfov = float(vfov)

    def setElevationLimits(self, min: float, max: float) -> bool:  # noqa: A002 - MatterSim names
        if -math.pi / 2.0 < min < 0.0 < max < math.pi / 2.0:
            self.minElevation, self.maxElevation = float(min), float(max)
            return True
        return False

    def setDiscretizedViewingAngles(self, value: bool) -> None:
        if not self.initialized:
            self.discretizeViews = bool(value)

    def setRestrictedNavigation(self, value: bool) -> None:
        if not self.initialized:
            self.restrictedNavigation = bool(value)

    def setPreloadingEnabled(self, value: bool) -> None:
        if not self.initialized:
            self.preloadImages = bool(value)  # nothing to preload: views are rendered on demand

    def setDepthEnabled(self, value: bool) -> None:
        if value:
            raise NotImplementedError("opticalnav_sim renders polarization only; depth is not available")

    def setBatchSize(self, size: int) -> None:
        if not self.initialized:
            self.batchSize = int(size)

    def setCacheSize(self, size: int) -> None:
        if not self.initialized:
            self.cacheSize = int(size)  # the server owns scene residency (--max-resident)

    def setSeed(self, seed: int) -> None:
        if not self.initialized:
            self.randomSeed = int(seed)

    # --- OpticalNav extensions ---
    def setVariant(self, variant: str) -> None:
        """Default optical variant: 'base', 'perturbed' or 'active_polar'. A scan id
        ``<scene>__<variant>`` overrides it per episode."""
        self.variant = str(variant)

    def setRenderSpp(self, spp: int) -> None:
        """Samples per pixel per frame (dataset renders used 1024)."""
        self.spp = int(spp)

    def setRenderMode(self, mode: str) -> None:
        """'polar' (Stokes S0..S3, the dataset's sensor) or 'rgb' (radiance only, faster)."""
        if mode not in ("polar", "rgb"):
            raise ValueError("render mode must be 'polar' or 'rgb'")
        self.renderMode = mode

    def setDenoiser(self, enabled: bool) -> None:
        """OptiX AI denoiser on each frame (polar mode denoises the 0/45/90/135 degree polarizer images)."""
        self.denoise = bool(enabled)

    def setPreviewEnabled(self, enabled: bool) -> None:
        """False skips the server's dataset preview recipe (~40 ms per frame); state.rgb is then a plain
        tonemap of the radiance made here (stokes.quick_preview)."""
        self.preview = bool(enabled)

    def setRenderSeed(self, seed: int) -> None:
        self.seed = int(seed)

    def setStokesDtype(self, dtype: str) -> None:
        if dtype not in ("float16", "float32"):
            raise ValueError("dtype must be 'float16' or 'float32'")
        self.stokesDtype = dtype

    # --- lifecycle ---
    def initialize(self) -> None:
        if self.initialized:
            return
        if self.renderingEnabled or not self.navGraphPath:
            self._client = RenderClient(self.datasetPath or None)
        self.states = [SimState() for _ in range(self.batchSize)]
        self._rng = random.Random(self.randomSeed)
        self.initialized = True

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
        self.initialized = False

    def _split(self, scan_id: str) -> tuple[str, str]:
        scene, sep, variant = scan_id.partition(VARIANT_SEPARATOR)
        return (scene, variant) if sep and variant else (scan_id, self.variant)

    def _graph(self, scan_id: str) -> NavGraph:
        scene, _ = self._split(scan_id)
        if scene not in self._graphs:
            local = Path(self.navGraphPath) / f"{scene}_connectivity.json" if self.navGraphPath else None
            if local is not None and local.is_file():
                self._graphs[scene] = NavGraph.load(self.navGraphPath, scene)
            else:
                self._graphs[scene] = NavGraph.from_connectivity(scene, self._client.connectivity(scene))
        return self._graphs[scene]

    def _scene_meta(self, scene: str) -> dict:
        if scene not in self._meta:
            self._meta[scene] = self._client.meta(scene)
        return self._meta[scene]

    def _check(self, values, name):
        if len(values) != self.batchSize:
            raise ValueError(f"MatterSim: {name} has {len(values)} entries, batch size is {self.batchSize}")

    # --- MatterSim episode API ---
    def newEpisode(self, scanId, viewpointId, heading, elevation) -> None:
        started = time.perf_counter()
        if not self.initialized:
            self.initialize()
        for values, name in ((scanId, "scanId"), (viewpointId, "viewpointId"), (heading, "heading"),
                             (elevation, "elevation")):
            self._check(values, name)
        self._set_heading_elevation(heading, elevation)
        for state, scan, vp in zip(self.states, scanId, viewpointId):
            graph = self._graph(scan)
            state.step = 0
            state.scanId = scan
            state.variant = self._split(scan)[1]
            state.location = graph.viewpoint(graph.index(vp))
            state._free_position = None
        self._populate_navigable()
        self._render()
        self._wall += time.perf_counter() - started

    def newRandomEpisode(self, scanId) -> None:
        viewpoints, headings = [], []
        if not self.initialized:
            self.initialize()
        for scan in scanId:
            graph = self._graph(scan)
            viewpoints.append(self._rng.choice([v for v, ok in zip(graph.ids, graph.included) if ok]))
            headings.append(self._rng.uniform(0.0, 2.0 * math.pi))
        self.newEpisode(scanId, viewpoints, headings, [0.0] * len(scanId))

    def teleport(self, scanId, position, heading, elevation) -> None:
        """Start an episode with the camera's optical centre at ``position``
        ([x, y, z], connectivity frame, z = height above the floor). The location
        reports the nearest graph viewpoint id; moving to a candidate snaps back
        onto the graph."""
        started = time.perf_counter()
        if not self.initialized:
            self.initialize()
        for values, name in ((scanId, "scanId"), (position, "position"), (heading, "heading"),
                             (elevation, "elevation")):
            self._check(values, name)
        self._set_heading_elevation(heading, elevation)
        for state, scan, pos in zip(self.states, scanId, position):
            graph = self._graph(scan)
            x, y, z = (float(v) for v in pos)
            near = min((i for i, ok in enumerate(graph.included) if ok),
                       key=lambda i: math.hypot(graph.pos[i][0] - x, graph.pos[i][1] - y))
            state.step = 0
            state.scanId = scan
            state.variant = self._split(scan)[1]
            state.location = ViewPoint(graph.ids[near], near, x, y, z)
            state._free_position = (x, y, z)
        self._populate_navigable()
        self._render()
        self._wall += time.perf_counter() - started

    def getState(self) -> list[SimState]:
        return self.states

    def makeAction(self, index, heading, elevation) -> None:
        started = time.perf_counter()
        if not self.initialized:
            raise RuntimeError("MatterSim: newEpisode must be called before makeAction")
        for values, name in ((index, "index"), (heading, "heading"), (elevation, "elevation")):
            self._check(values, name)
        new_h, new_e = [], []
        for i, (state, ix, h, e) in enumerate(zip(self.states, index, heading, elevation)):
            if int(ix) >= len(state.navigableLocations) or int(ix) < 0:
                raise ValueError(f"MatterSim: Invalid action index: {ix} in environment {i} of {self.batchSize}")
            target = state.navigableLocations[int(ix)]
            if int(ix) > 0:
                state._free_position = None
                graph = self._graph(state.scanId)
                target = graph.viewpoint(target.ix)
            state.location = ViewPoint(target.viewpointId, target.ix, target.x, target.y, target.z)
            state.step += 1
            h, e = float(h), float(e)
            if self.discretizeViews:
                h = math.copysign(2.0 * math.pi / HEADING_COUNT, h) if h != 0.0 else 0.0
                e = math.copysign(ELEVATION_INCREMENT, e) if e != 0.0 else 0.0
            new_h.append(state.heading + h)
            new_e.append(state.elevation + e)
        self._set_heading_elevation(new_h, new_e)
        self._populate_navigable()
        self._render()
        self._wall += time.perf_counter() - started

    def renderCamera(self, scanId: str, camera_to_world, *, width=None, height=None, hfov_deg=None) -> dict:
        """Render one dataset-convention camera (e.g. an index.jsonl ``camera_to_world``,
        flat column-major or 4x4). Returns {rgb (RGB uint8), s0..s3}."""
        if not self.initialized:
            self.initialize()
        c2w = np.asarray(camera_to_world, dtype=np.float64)
        c2w = frames.legacy_flat_to_matrix(c2w) if c2w.size == 16 and c2w.ndim == 1 else c2w.reshape(4, 4)
        scene, variant = self._split(scanId)
        w, h = int(width or self.width), int(height or self.height)
        hfov = hfov_deg if hfov_deg is not None else self._hfov_deg(w, h)
        return self._client.render([self._view(scene, variant, c2w, w, h, hfov)], self.stokesDtype)[0]

    # --- internals ---
    def _hfov_deg(self, width: int, height: int) -> float:
        return math.degrees(2.0 * math.atan(math.tan(self.vfov / 2.0) * width / height))

    def _view(self, scene, variant, c2w, width, height, hfov_deg) -> dict:
        return {"scan": scene, "variant": variant, "camera_to_world": c2w, "width": width, "height": height,
                "hfov_deg": hfov_deg, "spp": self.spp, "seed": self.seed, "mode": self.renderMode,
                "denoise": self.denoise, "preview": self.preview}

    def _set_heading_elevation(self, heading, elevation) -> None:
        for state, h, e in zip(self.states, heading, elevation):
            state.heading = frames.wrap(float(h))
            if self.discretizeViews:
                inc = 2.0 * math.pi / HEADING_COUNT
                step = int(math.floor(state.heading / inc + 0.5))  # std::lround for non-negative values
                if step == HEADING_COUNT:
                    step = 0
                state.heading = step * inc
                e = float(e)
                if e < -ELEVATION_INCREMENT / 2.0:
                    state.elevation, state.viewIndex = -ELEVATION_INCREMENT, step
                elif e > ELEVATION_INCREMENT / 2.0:
                    state.elevation, state.viewIndex = ELEVATION_INCREMENT, step + 2 * HEADING_COUNT
                else:
                    state.elevation, state.viewIndex = 0.0, step + HEADING_COUNT
            else:
                state.elevation = max(min(float(e), self.maxElevation), self.minElevation)

    def _populate_navigable(self) -> None:
        hfov = self.vfov * self.width / self.height  # MatterSim's approximation, kept for identical candidates
        for state in self.states:
            graph = self._graph(state.scanId)
            loc = state.location
            if state._free_position is None:
                state.navigableLocations = graph.navigable(loc.ix, state.heading, state.elevation, hfov,
                                                           self.restrictedNavigation)
            else:
                saved = graph.pos[loc.ix]
                graph.pos[loc.ix] = state._free_position
                try:
                    cands = graph.navigable(loc.ix, state.heading, state.elevation, hfov, self.restrictedNavigation)
                finally:
                    graph.pos[loc.ix] = saved
                state.navigableLocations = [loc] + cands[1:]

    def _render(self) -> None:
        if not self.renderingEnabled:
            return
        started = time.perf_counter()
        views = []
        hfov = self._hfov_deg(self.width, self.height)
        for state in self.states:
            scene, variant = self._split(state.scanId)
            mount = self._scene_meta(scene)["camera"]["mount"]
            if state._free_position is None:
                c2w = frames.camera_at_viewpoint(state.location.x, state.location.y, state.heading,
                                                 state.elevation, mount)
            else:
                c2w = frames.camera_at_position(state._free_position, state.heading, state.elevation, mount)
            state.camera_to_world = c2w
            views.append(self._view(scene, variant, c2w, self.width, self.height, hfov))
        results = self._client.render(views, self.stokesDtype)
        for state, out in zip(self.states, results):
            state.stokes = {k: out[k] for k in ("s0", "s1", "s2", "s3")} if "s0" in out else None
            state.radiance = out["s0"] if "s0" in out else out["radiance"]
            rgb = out["rgb"] if "rgb" in out else stokes.quick_preview(state.radiance)
            state.rgb = np.ascontiguousarray(rgb[:, :, ::-1])  # BGR, as MatterSim's cv::Mat
        self._frames += len(self.states)
        self._render_s += time.perf_counter() - started
        self._server_s += self._client.last_render_seconds

    def resetTimers(self) -> None:
        self._frames = 0
        self._wall = self._render_s = self._server_s = 0.0

    def timingInfo(self) -> str:
        fps = self._frames / self._wall if self._wall else 0.0
        return (f"Rendered {self._frames} frames\n"
                f"Wall time: {self._wall * 1000:.1f} ms, ({fps:.2f} fps)\n"
                f"\tRender requests: {self._render_s * 1000:.1f} ms (server {self._server_s * 1000:.1f} ms)\n")
