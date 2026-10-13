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
        self.lights = 1.0  # indoor light level (setIndoorLights)
        self.env = {"map": "original", "scale": 1.0, "rotate_deg": 0.0}  # sky (setEnvironment)
        self.bands = "rgb"  # setBands
        self.lighting = "all"  # setLighting
        self.lamps = False  # setLamps
        self.flash_scale = 1.0  # setFlashScale
        self.flash_offset = (0.0, 0.0, 0.0)  # setFlashOffset
        self.camera_offset = (0.0, 0.0, 0.0)  # setCameraOffset
        self.max_depth = 0  # setMaxDepth: 0 is the scene's own
        self.caustics = True  # setCaustics
        self.nir_range = (0.5, 1.0)  # setNirRange
        self.active_split = True  # setActiveSplit
        self.mirrors = None  # setMirrors: None (the variant's own) or {"density", "glass", "glass_blocking"}
        self.flash_band = "rgb"  # setFlashBand
        self.nir = {}  # setNirScales (server defaults otherwise)
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

    def setActiveLight(self, enabled: bool) -> None:
        """Switch the camera-mounted active polarized light: on renders 'active_polar', off renders
        'perturbed' (the same scene and lights without it). Takes effect from the next render
        (newEpisode, teleport or makeAction, mid-episode too). The render server keeps one resident
        scene for both, so switching loads nothing; the first frame of each records its freeze."""
        self.setVariant("active_polar" if enabled else "perturbed")

    def setBands(self, bands: str) -> None:
        """'rgb' (default), 'nir' or 'both'. The near-infrared band renders the same scene with pseudo-NIR
        materials (robomituba's pseudo_nir_albedo of each base colour) and per-class light levels
        (setNirScales); ``state.nir_stokes`` then holds its linear Stokes S0..S3 as HxW images."""
        if bands not in ("rgb", "nir", "both"):
            raise ValueError("bands must be 'rgb', 'nir' or 'both'")
        self.bands = bands

    def setFlashBand(self, band: str) -> None:
        """Which band the active polarized light emits in: 'rgb' (the dataset's active_polar), 'nir' or 'both'."""
        if band not in ("rgb", "nir", "both"):
            raise ValueError("flash band must be 'rgb', 'nir' or 'both'")
        self.flash_band = band

    def setNirScales(self, led: float | None = None, incandescent: float | None = None,
                     daylight: float | None = None) -> None:
        """Near-infrared radiance relative to visible, per light class (server defaults: LED 0,
        incandescent 0.45, daylight 0.75, robomituba's NIR_PASSIVE_SCALE_POLICY)."""
        for name, value in (("led", led), ("incandescent", incandescent), ("daylight", daylight)):
            if value is not None:
                self.nir[name] = float(value)

    def setMirrors(self, density: float | None = None, glass: int = 0, glass_blocking: bool = False) -> None:
        """Random mirrors and glass walls from the scene's perturbation pool instead of the variant's own:
        ``density`` in robomituba's units (1 is its default, 0 none), ``glass`` glass walls. Glass walls
        block the R2R edges they cross (navigableLocations leaves them out); only walls that leave a detour
        are used unless ``glass_blocking``. ``density=None`` returns to the variant's own. From the next
        render, without loading or recording."""
        self.mirrors = None if density is None else {"density": float(density), "glass": int(glass),
                                                     "glass_blocking": bool(glass_blocking)}

    def blockedEdges(self, scan_id: str) -> set[tuple[str, str]]:
        """R2R viewpoint pairs the current glass walls block (both orders)."""
        if not self.mirrors or not self.mirrors.get("glass"):
            return set()
        scene, _ = self._split(scan_id)
        pool = self._scene_meta(scene).get("perturbation_pool") or {}
        walls = [g for g in pool.get("glass", []) if self.mirrors.get("glass_blocking") or not g["splits_r2r"]]
        pairs = {tuple(e) for g in walls[: self.mirrors["glass"]] for e in g["blocked_edges"]}
        return pairs | {(b, a) for a, b in pairs}

    def setLighting(self, mode: str) -> None:
        """'none' (no lights), 'fixtures' (Infinigen's own lamp and ceiling-light emitters) or 'all' (also the
        room softboxes robomituba's importer added for even fill; the dataset's lighting). Lights a mode leaves
        out are removed from the scene, not dimmed. From the next render, without recording."""
        if mode not in ("none", "fixtures", "all"):
            raise ValueError("lighting must be 'none', 'fixtures' or 'all'")
        self.lighting = mode

    def setFlashScale(self, scale: float) -> None:
        """The camera flash's side times ``scale`` at the same power (radiance / scale^2): a larger light leaves far
        fewer bright outliers (fireflies) through glossy and mirror bounces. 1 is the dataset's flash. From the
        next render, without recording."""
        if not 0.25 <= float(scale) <= 8.0:
            raise ValueError("flash scale must be 0.25..8")
        self.flash_scale = float(scale)

    def setFlashOffset(self, right_m: float, up_m: float, forward_m: float = 0.0) -> None:
        """Move the camera flash from where the dataset put it (0.2 m behind the camera, on its axis) by ``right_m``,
        ``up_m`` and ``forward_m``: the observer's right, world up and the observer's level forward. It rides with
        the camera and keeps facing its view. Off the axis, a glossy surface facing the camera no longer sends the
        flash straight back into it: at 0.2 m up, a view with the light on had as few bright outliers as with it
        off (46 vs 152), at the same brightness. From the next render, without recording."""
        offset = (float(right_m), float(up_m), float(forward_m))
        if not all(abs(v) <= 2.0 for v in offset):
            raise ValueError("flash offset must be within 2 m")
        self.flash_offset = offset

    def setMaxDepth(self, depth: int) -> None:
        """Cap every light path at ``depth`` vertices (Mitsuba's max_depth: 1 shows emitters seen directly, 2 adds
        direct light, 3 and up one more indirect bounce each); 0 renders with the scene's own. A diagnostic: it
        splits where noise and fireflies come from by path length. Each new depth records its freeze once."""
        if not 0 <= int(depth) <= 64:
            raise ValueError("max_depth must be 0..64")
        self.max_depth = int(depth)

    def setCaustics(self, on: bool) -> None:
        """Off: the lights' passes drop caustics as the flash pass always does (path_nocaustics: light reaching an
        emitter past a near-specular bounce that follows a rough one), which removes the bright outliers of
        mirror, glass and glossy paths but also the light such paths carry (a room lit through window glass
        darkens). On (default) is the dataset's rendering. Each setting records its freeze once."""
        self.caustics = bool(on)

    def setActiveSplit(self, on: bool) -> None:
        """With the camera light on, render the lights and the flash on one path (path_nocaustics "split": the
        same image as two paths, about a third less GPU time) instead of two. On by default; the render server
        falls back to two paths when its plugin has no split mode. Each setting records its freeze once."""
        self.active_split = bool(on)

    def setNirRange(self, low: float, high: float) -> None:
        """Map the pseudo near-infrared reflectance onto [``low``, ``high``]. robomituba's formula
        (``dot(max(c, 1 - c), luma)`` of the base colour) spans 0.5 (mid grey) to 1 (black or white): every surface
        reflects at least half the near-infrared, which keeps paths alive through many bounces (grain and
        fireflies). (0.5, 1) is the formula as the dataset used it. From the next render, without recording."""
        if not 0.0 <= float(low) <= float(high) <= 1.0:
            raise ValueError("nir range must satisfy 0 <= low <= high <= 1")
        self.nir_range = (float(low), float(high))

    def setCameraOffset(self, right_m: float, up_m: float, forward_m: float = 0.0) -> None:
        """Move the camera on its rig by ``right_m``, ``up_m`` and ``forward_m`` (the observer's right, world up and
        level forward) from the scene's mount. Raising it tilts the view down as the dataset rig does (it looks at
        0.9 x eye height, 1 m ahead); (0, 0, 0) is the dataset camera. The flash follows. From the next render."""
        offset = (float(right_m), float(up_m), float(forward_m))
        if not all(abs(v) <= 2.0 for v in offset):
            raise ValueError("camera offset must be within 2 m")
        self.camera_offset = offset

    def rig(self, scene: str) -> dict:
        """The camera rig of ``scene`` for an editor: the mount (lateral, height, forward; m), the flash (distance
        behind the camera, size), and the offsets set on top."""
        meta = self._scene_meta(scene)
        mount = (meta.get("camera") or {}).get("mount") or {}
        flash = next((v["flash"] for v in (meta.get("variants") or {}).values() if v.get("flash")), None)
        return {"mount_xyz_m": [float(v) for v in (mount.get("xyz_m") or [0.0, 1.0, 0.0])],
                "nominal_pitch_deg": math.degrees(frames.nominal_pitch(self._mount(scene))),
                "flash": {"distance_m": float(flash["distance_m"]), "size_m": [float(v) for v in flash["size_world"]]}
                if flash else None,
                "camera_offset": list(self.camera_offset), "flash_offset": list(self.flash_offset),
                "flash_scale": self.flash_scale}

    def setLamps(self, on: bool) -> None:
        """Also light the lamps the scene left dark (Infinigen spawns a light with only some of its lamps; the
        server adds one inside every other lamp's shade, at the power of the scene's own lamp lights). On top
        of the lighting mode; off by default, as the dataset rendered. From the next render, without recording."""
        self.lamps = bool(on)

    def setIndoorLights(self, level: float) -> None:
        """Scale every indoor light (the scene's area emitters; not the camera flash or the sky): 1 as built,
        0 all off. From the next render; the server changes the loaded scene in place, without recording."""
        self.lights = float(level)

    def setEnvironment(self, name: str = "original", scale: float = 1.0, rotate_deg: float = 0.0) -> None:
        """The sky seen through windows: 'original' (the scene's own) or an environment map the render server
        offers (``GET /v1/info`` "envmaps"; files in the pack's envmaps folder), times ``scale``, turned
        ``rotate_deg`` about the vertical. From the next render, without recording."""
        self.env = {"map": str(name), "scale": float(scale), "rotate_deg": float(rotate_deg)}

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
            elif self._scene_meta(scene).get("spawn") and not self._scene_meta(scene).get("graph"):
                # packed without a navigation graph: start at its spawn points and move freely
                self._graphs[scene] = NavGraph.from_spawn(scene, self._scene_meta(scene)["spawn"])
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
    def _mount(self, scene: str) -> dict:
        """The scene's camera mount, raised by the camera offset (setCameraOffset)."""
        mount = dict(self._scene_meta(scene)["camera"]["mount"] or {})
        if self.camera_offset[1]:
            xyz = list(mount.get("xyz_m") or [0.0, 1.0, 0.0])
            xyz[1] = float(xyz[1]) + self.camera_offset[1]
            mount["xyz_m"] = xyz
        return mount

    def _camera(self, state) -> np.ndarray:
        """Camera to world of ``state``: the dataset rig (frames.camera_at_viewpoint / camera_at_position) with the
        camera offset on top, its height through the mount (the rig's pitch follows it), the rest along the level
        view."""
        scene = self._split(state.scanId)[0]
        mount = self._mount(scene)
        if state._free_position is None:
            c2w = frames.camera_at_viewpoint(state.location.x, state.location.y, state.heading, state.elevation, mount)
        else:
            x, y, z = state._free_position
            c2w = frames.camera_at_position((x, y, z + self.camera_offset[1]), state.heading, state.elevation, mount)
        right_m, _, forward_m = self.camera_offset
        if right_m or forward_m:
            right = c2w[:3, 0] / np.linalg.norm(c2w[:3, 0])  # level already: the rig pitches about it
            c2w = c2w.copy()
            c2w[:3, 3] += right * right_m + np.cross((0.0, 1.0, 0.0), right) * forward_m
        return c2w

    def _hfov_deg(self, width: int, height: int) -> float:
        return math.degrees(2.0 * math.atan(math.tan(self.vfov / 2.0) * width / height))

    def _view(self, scene, variant, c2w, width, height, hfov_deg) -> dict:
        return {"scan": scene, "variant": variant, "camera_to_world": c2w, "width": width, "height": height,
                "hfov_deg": hfov_deg, "spp": self.spp, "seed": self.seed, "mode": self.renderMode,
                "denoise": self.denoise, "preview": self.preview, "lights": self.lights, "env": dict(self.env),
                "bands": self.bands, "flash_band": self.flash_band, "nir": dict(self.nir), "lighting": self.lighting, "lamps": self.lamps,
                "flash_scale": self.flash_scale, "flash_offset": list(self.flash_offset), "max_depth": self.max_depth,
                "caustics": self.caustics, "nir_range": list(self.nir_range), "active_split": self.active_split,
                "mirrors": dict(self.mirrors) if self.mirrors is not None else None}

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
            blocked = self.blockedEdges(state.scanId)
            if state._free_position is None:
                state.navigableLocations = graph.navigable(loc.ix, state.heading, state.elevation, hfov,
                                                           self.restrictedNavigation)
                if blocked:  # glass walls in the way
                    state.navigableLocations = [state.navigableLocations[0]] + [
                        c for c in state.navigableLocations[1:] if (loc.viewpointId, c.viewpointId) not in blocked]
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
            c2w = self._camera(state)
            state.camera_to_world = c2w
            state.variant = variant  # setVariant / setActiveLight apply from the next render
            views.append(self._view(scene, variant, c2w, self.width, self.height, hfov))
        results = self._client.render(views, self.stokesDtype)
        for state, out in zip(self.states, results):
            state.stokes = {k: out[k] for k in ("s0", "s1", "s2", "s3")} if "s0" in out else None
            state.nir_stokes = {k: out[f"nir_{k}"] for k in ("s0", "s1", "s2", "s3")} if "nir_s0" in out else (
                {"s0": out["nir"]} if "nir" in out else None)
            state.radiance = out["s0"] if "s0" in out else out.get("radiance")
            if state.radiance is None:  # near-infrared only: a grey preview of it
                n = state.nir_stokes["s0"]
                state.radiance = np.repeat(np.asarray(n, dtype=np.float32)[..., None], 3, axis=2)
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
