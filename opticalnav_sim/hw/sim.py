"""A World drawn by the render server: every actor with shapes is a body in the scene, every camera renders through its
lens.

    world = SimWorld(scene="blend_REF_CS02")
    robot = world.spawn(robots.RangerMini2(), at=world.on_floor(17.6, -11.4, yaw_deg=120.0))
    cam = world.spawn(cameras.LucidTDR054S(), attach_to=robot, link="top_plate", mount=Transform((0.2, 0.0, 0.15)))
    robot.spec.apply(robot.state, RangerControl("ackermann", 0.5, 10.0))
    world.tick()
    frame = world.render(cam)        # frame.s0 (H x W x 3), frame.image (uint8), frame.intrinsics, frame.pose
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..bodies import model_key, safe
from ..client import RenderClient
from . import optics, sensor as sensor_model
from .actor import Actor, Spec, World
from .transform import Transform, camera_to_world, optical_pose, pose2d, to_mitsuba, yaw_from_heading


@dataclass
class Frame:
    """One camera image. ``s0``..``s3``: Stokes components, linear radiance, H x W x 3 (RGB), as the lens images them
    (distortion applied). ``image``: the server's 8-bit preview, resampled the same way. ``pose``: world pose of the
    optical frame. ``intrinsics``: of this image (its size, not the sensor's)."""
    time: float
    intrinsics: optics.Intrinsics
    pose: np.ndarray
    image: np.ndarray
    s0: np.ndarray
    s1: np.ndarray | None = None
    s2: np.ndarray | None = None
    s3: np.ndarray | None = None
    depth: np.ndarray | None = None
    timing: dict = field(default_factory=dict)


def _visual_dict(v) -> dict:
    origin = v.origin.matrix()
    if tuple(getattr(v, "scale", (1.0, 1.0, 1.0))) != (1.0, 1.0, 1.0):   # a mesh in other units than metres
        origin = origin @ np.diag([*v.scale, 1.0])
    return {"shape": v.shape, "size": list(v.size), "origin": origin.tolist(),
            "material": [v.material[0], list(v.material[1]), float(v.material[2])], **({"mesh": v.mesh} if v.mesh else {})}


class SimWorld(World):
    def __init__(self, url: str | None = None, scene: str = "", variant: str = "base", dt: float = 0.05, spp: int = 64,
                 client: RenderClient | None = None):
        super().__init__(dt=dt)
        self.client = client or RenderClient(url)
        self.scene, self.variant, self.spp = scene, variant, int(spp)
        self.seed = 0
        self._models: dict[int, tuple[str, dict]] = {}   # id(spec) -> (model name, descriptor)
        self._registered: set[int] = set()               # id(spec) of the models the server has
        self._plans: dict[tuple, optics.RenderPlan] = {}
        self.meta = self.client.meta(scene) if scene else {}

    # --- the scene ------------------------------------------------------------------------------------------------
    def spawn_points(self) -> list[Transform]:
        """The scene's start poses, as camera poses (their height is a camera's, not the floor's: see on_floor)."""
        return [Transform(tuple(float(v) for v in s["position"]),
                          (0.0, 0.0, math.degrees(yaw_from_heading(math.radians(float(s.get("heading_deg", 0.0)))))))
                for s in self.meta.get("spawn") or []]

    def floor_height(self, x: float, y: float, from_z: float) -> float:
        """Height of the first surface under (x, y, from_z): one ray down, through the server's depth render."""
        down = Transform((float(x), float(y), float(from_z)), (0.0, 90.0, 0.0)).matrix()
        view = {"scan": self.scene, "variant": self.variant, "camera_to_world": camera_to_world(down), "width": 16,
                "height": 16, "hfov_deg": 2.0, "spp": 1, "seed": 0, "mode": "polar", "depth": True, "preview": False}
        bodies = self.bodies()
        if bodies:  # the same resident scene as the cameras' views (another would load), its bodies out of the ray's way
            away = np.eye(4)
            away[1, 3] = -1000.0
            view["bodies"] = [dict(b, links={k: away @ m for k, m in b["links"].items()}) for b in bodies]
        z = np.asarray(self.client.render([view])[0]["depth"], dtype=np.float64)
        hit = float(np.median(z[6:10, 6:10]))
        if hit <= 0.0:
            raise ValueError(f"nothing under ({x:.2f}, {y:.2f}, {from_z:.2f}) in {self.scene}")
        return float(from_z) - hit

    def load_ground(self, floor_z: float | None = None, top: float = 1.4, cell: float = 0.05, rebuild: bool = False,
                    around: float = 20.0, center=None):
        """The scene's ground map as this world's ground (heights, obstacles, each surface's friction from the scene's
        own materials: opticalnav_sim.groundmap). Read from ``<scene>/map/ground.npz`` when it fits, else built from the
        scene's files (seconds; needs them on this machine) and saved there. ``floor_z``: the floor to stand on
        (default: the one under the first start pose); ``around``: metres mapped around the scene's start poses, or
        around ``center`` (x, y) when one is given (a robot put down elsewhere: a scene's own map can span a campus).
        Robots then follow the floor, stop at obstacles, and, with ``dynamics``, grip as the surface under each wheel
        allows."""
        from pathlib import Path

        from .. import groundmap
        folder = Path(str(self.meta.get("scene_dir") or ""))
        if not (folder / "base.xml").is_file():
            raise FileNotFoundError(f"the files of {self.scene} are not on this machine ({folder}): build the map where they are "
                                    "(tools/build_ground_map.py) and load it with groundmap.GroundMap.load(...).ground()")
        spawn = (self.meta.get("spawn") or [{"position": [0.0, 0.0, 1.5]}])[0]["position"]
        if floor_z is None:
            probe = groundmap.build(folder, spawn[2] - 1.5, top=0.5, cell=0.25, below=1.0,
                                    bounds=(spawn[0] - 2, spawn[1] - 2, spawn[0] + 2, spawn[1] + 2))
            floor_z = float(np.nanmedian(np.where(probe.material >= 0, probe.heights, np.nan)))
        whole = (self.meta.get("map") or {}).get("bounds")
        bounds = None
        if center is not None:
            bounds = [center[0] - around, center[1] - around, center[0] + around, center[1] + around]
        elif self.meta.get("spawn"):
            p = np.array([s["position"][:2] for s in self.meta["spawn"]])
            bounds = [p[:, 0].min() - around, p[:, 1].min() - around, p[:, 0].max() + around, p[:, 1].max() + around]
        if bounds is not None and whole:
            bounds = [max(bounds[0], whole[0]), max(bounds[1], whole[1]), min(bounds[2], whole[2]), min(bounds[3], whole[3])]
        saved = folder / "map" / "ground.npz"
        gm = None
        if saved.is_file() and not rebuild:
            gm = groundmap.GroundMap.load(saved)
            fits = gm.cell == cell and gm.top >= top
            if center is None:
                fits = fits and abs(gm.floor_z - floor_z) <= 0.02 and gm.meta.get("around") == around and not gm.meta.get("center")
            else:   # a map made for another place does while it holds this one with room to drive (or to the scene's edge)
                rows, cols = gm.heights.shape
                have = [gm.origin[0], gm.origin[1], gm.origin[0] + (cols - 1) * gm.cell, gm.origin[1] + (rows - 1) * gm.cell]
                room = min(8.0, 0.5 * around)
                edge = [max(center[0] - room, bounds[0]), max(center[1] - room, bounds[1]),
                        min(center[0] + room, bounds[2]), min(center[1] + room, bounds[3])]
                fits = fits and abs(gm.floor_z - floor_z) <= 0.25 and have[0] <= edge[0] + cell and have[1] <= edge[1] + cell \
                    and have[2] >= edge[2] - cell and have[3] >= edge[3] - cell
            if not fits:
                gm = None
        if gm is None:
            gm = groundmap.build(folder, floor_z, top, cell, bounds)
            gm.meta["around"] = around
            if center is not None:
                gm.meta["center"] = [float(center[0]), float(center[1])]
            (folder / "map").mkdir(exist_ok=True)
            gm.save(saved)
        self.ground = gm.ground(self.ground.step)
        for a in self.actors:
            if hasattr(a.state, "ground"):
                a.state.ground = self.ground
        return gm

    def on_floor(self, x: float, y: float, yaw_deg: float = 0.0, from_z: float | None = None) -> Transform:
        """A pose standing on the floor at (x, y): by the ground map when one is loaded (``load_ground``), else by a ray
        down through the render server. ``from_z``: where to look down from (default: the scene's first start pose's
        height)."""
        if self.ground.heights is not None and from_z is None:
            z = self.ground.height(x, y, near=self.ground.z)
            return pose2d(x, y, z, yaw_deg)
        if from_z is None:
            from_z = float((self.meta.get("spawn") or [{"position": [0.0, 0.0, 1.5]}])[0]["position"][2])
        return pose2d(x, y, self.floor_height(x, y, from_z), yaw_deg)

    # --- bodies ---------------------------------------------------------------------------------------------------
    def _model(self, spec: Spec) -> tuple[str, dict] | None:
        if not hasattr(spec, "visuals"):
            return None
        if id(spec) not in self._models:
            links = {link: [_visual_dict(v) for v in vs] for link, vs in spec.visuals().items()}
            # named by what is drawn too: two lamps of one model with other colours are two models
            name = f"{safe(getattr(spec, 'model', None) or type(spec).__name__)}_{model_key({'links': links})[:8]}"
            self._models[id(spec)] = (name, links)
        return self._models[id(spec)] if self._models[id(spec)][1] else None   # nothing drawn: not a body

    def bodies(self) -> list[dict]:
        """Every drawn actor as the render server's view field: registers a model the first time it is used."""
        out = []
        for actor in self.actors:
            model = self._model(actor.spec)
            if model is None:
                continue
            name, links = model
            if id(actor.spec) not in self._registered:
                self.client.register_body(name, links)
                self._registered.add(id(actor.spec))
            frames = actor.link_matrices()
            out.append({"id": actor.name, "model": name, "links": {link: to_mitsuba(frames[link]) for link in links}})
        return out

    # --- cameras --------------------------------------------------------------------------------------------------
    def view(self, camera: Actor, spp: int | None = None, scale: float | None = None, oversample: float = 1.0,
             **options) -> tuple[dict, optics.RenderPlan, optics.Intrinsics]:
        """The render server view of a camera actor: the pinhole that holds its lens's rays, and the bodies."""
        k = camera.spec.intrinsics(scale)
        key = (k, float(oversample))
        if key not in self._plans:
            self._plans[key] = optics.render_plan(k, oversample)
        plan = self._plans[key]
        view = {"scan": self.scene, "variant": self.variant, "mode": "polar", "seed": self.seed,
                "camera_to_world": camera_to_world(camera.link_matrix("camera")), "width": plan.width,
                "height": plan.height, "hfov_deg": plan.hfov_deg, "spp": int(spp or self.spp), **options}
        bodies = self.bodies()
        if bodies:
            view["bodies"] = bodies
        return view, plan, k

    def request(self, camera: Actor, spp: int | None = None, scale: float | None = None, oversample: float = 1.0,
                **options) -> dict:
        """What ``render`` asks the server for, with the world as it is now: ``{"view", "plan", "intrinsics", "pose",
        "time"}``. Split from the rendering so a caller that steps the world on another thread reads the poses in one
        short moment (under its own lock) and waits for the image outside it."""
        if camera.spec.kind != "camera":
            raise ValueError(f"{camera.name} is a {camera.spec.kind}, not a camera")
        view, plan, k = self.view(camera, spp, scale, oversample, **options)
        self.seed += 1
        return {"view": view, "plan": plan, "intrinsics": k, "pose": optical_pose(camera.link_matrix("camera")), "time": self.time}

    def frame(self, request: dict, out: dict, components=("s0", "s1", "s2", "s3")) -> Frame:
        """The server's answer to a ``request`` as the camera's Frame (the lens's distortion applied). ``components``:
        the Stokes images to keep (a plain sensor needs S0 only: the others are not resampled)."""
        plan = request["plan"]

        def lens(a):
            return None if a is None else optics.remap(np.asarray(a, dtype=np.float32), plan.map_x, plan.map_y)

        image = np.clip(np.rint(lens(out["rgb"])), 0, 255).astype(np.uint8) if "rgb" in out else None
        s = [lens(out.get(n)) if n in components else None for n in ("s0", "s1", "s2", "s3")]
        return Frame(request["time"], request["intrinsics"], request["pose"], image, s[0], s[1], s[2], s[3],
                     lens(out.get("depth")), dict(self.client.last_timing))

    def render(self, camera: Actor, spp: int | None = None, scale: float | None = None, oversample: float = 1.0,
               stokes_dtype: str = "float32", **options) -> Frame:
        """Render a camera now. ``scale``: share of the sensor's resolution (default: the spec's render_scale).
        ``options``: other view fields of the render server (bands, lights, env, depth, denoise...)."""
        request = self.request(camera, spp, scale, oversample, **options)
        return self.frame(request, self.client.render([request["view"]], stokes_dtype)[0])

    def capture(self, camera: Actor, exposure_s: float | None = None, gain_db: float | None = None, noise: bool = True,
                spp: int | None = None, scale: float | None = None, binning: bool = False, **options) -> tuple:
        """Render a camera and expose its sensor: ``(raw, frame)``. ``raw`` (hw.sensor.Raw) is what the camera delivers:
        its mosaic, noise, saturation and bit depth; ``hw.isp.process(raw)`` is the image.

        The exposure is the camera's own (``camera.state.exposure``: auto exposure, on by default, which follows the
        scene over the frames captured); ``exposure_s`` / ``gain_db`` given here take this one frame with those
        instead and leave the camera's setting alone. ``binning``: the rendered pixel is the mean of the sensor pixels
        it covers (their noise averaged), instead of one pixel of a smaller sensor. ``frame`` keeps the Stokes images
        the raw was made from (the ground truth)."""
        request = self.request(camera, spp, scale, **options)
        return self.expose(camera, request, self.client.render([request["view"]], "float32")[0], exposure_s, gain_db, noise, binning)

    def expose(self, camera: Actor, request: dict, out: dict, exposure_s: float | None = None, gain_db: float | None = None,
               noise: bool = True, binning: bool = False) -> tuple:
        """``capture``'s second half: the server's answer to ``request`` through the camera's sensor."""
        polar = camera.spec.sensor.cfa.startswith("polar_")
        frame = self.frame(request, out, ("s0", "s1", "s2") if polar else ("s0",))
        light = sensor_model.irradiance(frame, camera.spec)   # once: the meter and the sensor both read it
        if exposure_s is None:
            exposure_s, auto_gain = camera.state.exposure.next(frame, camera.spec, light)
            gain_db = auto_gain if gain_db is None else gain_db
        covered = (camera.spec.sensor.width / frame.intrinsics.width) ** 2 if binning else 1.0
        return sensor_model.expose(frame, camera.spec, exposure_s, gain_db or 0.0, noise, seed=self.seed,
                                   pixels_per_sample=covered, light=light), frame
