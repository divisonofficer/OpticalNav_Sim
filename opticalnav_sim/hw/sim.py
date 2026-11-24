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

from ..bodies import safe
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
    return {"shape": v.shape, "size": list(v.size), "origin": v.origin.matrix().tolist(),
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

    def on_floor(self, x: float, y: float, yaw_deg: float = 0.0, from_z: float | None = None) -> Transform:
        """A pose standing on the floor at (x, y). ``from_z``: where to look down from (default: the scene's first
        start pose's height)."""
        if from_z is None:
            from_z = float((self.meta.get("spawn") or [{"position": [0.0, 0.0, 1.5]}])[0]["position"][2])
        return pose2d(x, y, self.floor_height(x, y, from_z), yaw_deg)

    # --- bodies ---------------------------------------------------------------------------------------------------
    def _model(self, spec: Spec) -> tuple[str, dict] | None:
        if not hasattr(spec, "visuals"):
            return None
        if id(spec) not in self._models:
            links = {link: [_visual_dict(v) for v in vs] for link, vs in spec.visuals().items()}
            self._models[id(spec)] = (safe(getattr(spec, "model", None) or type(spec).__name__), links)
        return self._models[id(spec)]

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

    def render(self, camera: Actor, spp: int | None = None, scale: float | None = None, oversample: float = 1.0,
               stokes_dtype: str = "float32", **options) -> Frame:
        """Render a camera now. ``scale``: share of the sensor's resolution (default: the spec's render_scale).
        ``options``: other view fields of the render server (bands, lights, env, depth, denoise...)."""
        if camera.spec.kind != "camera":
            raise ValueError(f"{camera.name} is a {camera.spec.kind}, not a camera")
        view, plan, k = self.view(camera, spp, scale, oversample, **options)
        out = self.client.render([view], stokes_dtype)[0]
        self.seed += 1

        def lens(a):
            return None if a is None else optics.remap(np.asarray(a, dtype=np.float32), plan.map_x, plan.map_y)

        image = np.clip(np.rint(lens(out["rgb"])), 0, 255).astype(np.uint8) if "rgb" in out else None
        return Frame(self.time, k, optical_pose(camera.link_matrix("camera")), image, lens(out.get("s0")),
                     lens(out.get("s1")), lens(out.get("s2")), lens(out.get("s3")), lens(out.get("depth")),
                     dict(self.client.last_timing))

    def capture(self, camera: Actor, exposure_s: float | None = None, gain_db: float = 0.0, noise: bool = True,
                spp: int | None = None, scale: float | None = None, binning: bool = False, **options) -> tuple:
        """Render a camera and expose its sensor: ``(raw, frame)``. ``raw`` (hw.sensor.Raw) is what the camera delivers:
        its mosaic, noise, saturation and bit depth; ``hw.isp.process(raw)`` is the image. ``exposure_s`` None: the
        camera's auto exposure. ``binning``: the rendered pixel is the mean of the sensor pixels it covers (their
        noise averaged), instead of one pixel of a smaller sensor. ``frame`` keeps the Stokes images the raw was made
        from (the ground truth)."""
        frame = self.render(camera, spp, scale, **options)
        covered = (camera.spec.sensor.width / frame.intrinsics.width) ** 2 if binning else 1.0
        return sensor_model.expose(frame, camera.spec, exposure_s, gain_db, noise, seed=self.seed,
                                   pixels_per_sample=covered), frame
