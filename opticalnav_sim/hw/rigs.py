"""Sensor rigs: the structure a robot carries and the cameras on it, placed by a measured calibration.

    rig = rigs.spawn_camera_rig(world, robot, "multi_cam_calibration.yaml", cameras=("left", "12_left"))
    frame = world.render(rig["left"])

The cameras' intrinsics and their poses relative to one another are the calibration's. Where the reference camera
sits on the mast (``reference_mount``) is not in a camera calibration: it is measured on the rig, or estimated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import cameras as camera_models
from .actor import Actor, Spec, World
from .calibration import RigCalibration
from .robots import Visual
from .transform import BODY_TO_OPTICAL, Transform

ALUMINIUM = ("aluminium", (0.62, 0.63, 0.65), 0.5)   # anodized: satin, not a mirror


@dataclass
class Mast(Spec):
    """A frame of T-slot aluminium extrusions (``profile`` square): four posts of ``height`` on a ``footprint`` (along
    x, along y), tied by rungs across the front and the back every ``rung_spacing`` (the ladder of the real rig) and by
    rails along the sides at the bottom, the middle and the top; on top, a bar across the front that the cameras sit on
    (``camera_bar`` long). Links: ``mast`` (the centre of its foot) and ``top`` (the centre of the posts' tops)."""
    kind = "structure"
    height: float = 0.80
    footprint: tuple = (0.20, 0.24)
    profile: float = 0.03
    rung_spacing: float = 0.19
    camera_bar: float = 0.44
    material: tuple = ALUMINIUM

    def link_frames(self, state) -> dict:
        return {"mast": np.eye(4), "top": Transform((0.0, 0.0, self.height)).matrix()}

    def visuals(self) -> dict:
        (lx, ly), h, p, m = self.footprint, self.height, self.profile, self.material
        x, y = lx / 2.0 - p / 2.0, ly / 2.0 - p / 2.0
        along_x, along_y = (0.0, 90.0, 0.0), (90.0, 0.0, 0.0)   # a profile runs along its z: turned to lie along x or y
        bars = [Visual("profile", (p, h), Transform((sx * x, sy * y, h / 2.0)), m) for sx in (-1, 1) for sy in (-1, 1)]
        rungs = max(2, round((h - p) / self.rung_spacing) + 1)
        for z in np.linspace(p / 2.0, h - p / 2.0, rungs):       # the ladder: across the front and the back
            bars += [Visual("profile", (p, ly - 2 * p), Transform((sx * x, 0.0, float(z)), along_y), m) for sx in (-1, 1)]
        for z in (p / 2.0, h / 2.0, h - p / 2.0):                # the sides
            bars += [Visual("profile", (p, lx - 2 * p), Transform((0.0, sy * y, z), along_x), m) for sy in (-1, 1)]
        bars.append(Visual("profile", (p, self.camera_bar), Transform((x + p, 0.0, h - p / 2.0), along_y), m))
        return {"mast": bars}


@dataclass
class ProfileFrame(Spec):
    """A free structure of T-slot aluminium extrusions and plates. ``bars``: each ``{"from": [x, y, z], "to": [x, y, z]}``
    (the profile's centre line, metres; ``"profile"``: another side than the frame's). ``plates``: each ``{"size": [x, y,
    z], "at": [x, y, z]}``. ``points``: named places things mount on, ``{"top": {"xyz": ..., "rpy_deg": ...}}``.
    Links: ``frame`` (the structure's origin) and the points."""
    kind = "structure"
    profile: float = 0.02
    bars: tuple = ()
    plates: tuple = ()
    points: dict = field(default_factory=dict)
    material: tuple = ALUMINIUM
    model: str = "frame"

    def __post_init__(self):
        for b in self.bars:
            if set(b) - {"from", "to", "profile"} or not {"from", "to"} <= set(b):
                raise ValueError(f"a bar is {{from, to[, profile]}}, not {b}")
            if np.linalg.norm(np.subtract(b["to"], b["from"])) < 1e-6:
                raise ValueError(f"a bar of no length: {b}")
        for q in self.plates:
            if set(q) != {"size", "at"}:
                raise ValueError(f"a plate is {{size, at}}, not {q}")

    def link_frames(self, state) -> dict:
        out = {"frame": np.eye(4)}
        for name, at in self.points.items():
            out[name] = Transform(tuple(float(v) for v in at.get("xyz", (0, 0, 0))), tuple(float(v) for v in at.get("rpy_deg", (0, 0, 0)))).matrix()
        return out

    def visuals(self) -> dict:
        vis = []
        for b in self.bars:
            a, z = np.asarray(b["from"], dtype=np.float64), np.asarray(b["to"], dtype=np.float64)
            along = (z - a) / np.linalg.norm(z - a)
            side = np.cross([0.0, 1.0, 0.0] if abs(along[1]) < 0.9 else [1.0, 0.0, 0.0], along)
            side /= np.linalg.norm(side)
            m = np.eye(4)
            m[:3, :3], m[:3, 3] = np.stack([side, np.cross(along, side), along], 1), (a + z) / 2.0
            vis.append(Visual("profile", (float(b.get("profile", self.profile)), float(np.linalg.norm(z - a))), Transform.from_matrix(m), self.material))
        vis += [Visual("box", tuple(float(v) for v in q["size"]), Transform(tuple(float(v) for v in q["at"])), self.material) for q in self.plates]
        return {"frame": vis} if vis else {}


def camera_for(name: str, calibrated, render_scale: float = 0.25):
    """The camera model and recorded image size behind a calibration entry's name, for the rig of
    docs/hardware_platform_plan.md: ``left`` / ``right`` are the LUCID TDR054S pair (read out 2x2 binned, 1440 x 928),
    ``12_*`` the LUCID TRI032S (12-bit, 1024 x 768). Give ``models`` to spawn_camera_rig for other names."""
    if name in ("left", "right"):
        return camera_models.LucidTDR054S(calibration=calibrated.intrinsics(1440, 928), render_scale=render_scale)
    if name.startswith("12_"):
        return camera_models.LucidTRI032S(calibration=calibrated.intrinsics(1024, 768), render_scale=render_scale)
    raise KeyError(f"no camera model known for the calibration entry {name!r}: pass models={{{name!r}: a CameraSpec factory}}")


def camera_mounts(calibration: RigCalibration, names, reference_mount: Transform = Transform()) -> dict:
    """Each camera's mount (its body frame: x along the optical axis) in the frame ``reference_mount`` places the
    reference camera in. The calibration relates optical frames (z forward); a body frame is that, turned."""
    to_body = BODY_TO_OPTICAL
    return {name: Transform.from_matrix(reference_mount.matrix() @ to_body @ calibration.relative(name) @ np.linalg.inv(to_body))
            for name in names}


def spawn_camera_rig(world: World, robot: Actor, calibration, cameras=("left", "12_left"), mast: Mast | None = None,
                     link: str = "top_plate", mast_mount: Transform = Transform(),
                     reference_mount: Transform = Transform((0.03, 0.0, 0.14)), render_scale: float = 0.25,
                     models: dict | None = None) -> dict:
    """A mast on ``robot`` and the calibration's ``cameras`` on its top: ``{"mast": actor, name: actor, ...}``.

    ``calibration``: a RigCalibration or the path of its OpenCV YAML. ``reference_mount``: the reference camera's
    pose on the mast's ``top`` link (default: an estimate for the 80 cm mast, the reference camera 14 cm above the
    plate, the others 9 to 11 cm lower and 5 to 7 cm ahead of it as calibrated). ``models``: name -> a function of
    (CalibratedCamera, render_scale) giving the CameraSpec, for names ``camera_for`` does not know."""
    if not isinstance(calibration, RigCalibration):
        calibration = RigCalibration.load(calibration)
    mounts = camera_mounts(calibration, cameras, reference_mount)
    # every camera's spec before anything is spawned: an unknown name leaves the world as it was
    specs = {name: (models or {}).get(name, lambda c, s, n=name: camera_for(n, c, s))(calibration.cameras[name], render_scale)
             for name in mounts}
    out = {"mast": world.spawn(mast or Mast(), attach_to=robot, link=link, mount=mast_mount, name=f"{robot.name}_mast")}
    for name, spec in specs.items():
        out[name] = world.spawn(spec, attach_to=out["mast"], link="top", mount=mounts[name], name=f"{robot.name}_{name}")
    return out
