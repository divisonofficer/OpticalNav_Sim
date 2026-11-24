"""Sensor rigs: the structure a robot carries and the cameras on it, placed by a measured calibration.

    rig = rigs.spawn_camera_rig(world, robot, "multi_cam_calibration.yaml", cameras=("left", "12_left"))
    frame = world.render(rig["left"])

The cameras' intrinsics and their poses relative to one another are the calibration's. Where the reference camera
sits on the mast (``reference_mount``) is not in a camera calibration: it is measured on the rig, or estimated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import cameras as camera_models
from .actor import Actor, Spec, World
from .calibration import RigCalibration
from .robots import Visual
from .transform import BODY_TO_OPTICAL, Transform

ALUMINIUM = ("metal", (0.80, 0.81, 0.83), 0.35)


@dataclass
class Mast(Spec):
    """An aluminium profile frame: four posts of ``height`` on a ``footprint`` (along x, along y), tied by rails at the
    bottom, the middle and the top, with a plate on top. Links: ``mast`` (the centre of its foot) and ``top`` (the
    centre of the top plate's upper face)."""
    kind = "structure"
    height: float = 0.80
    footprint: tuple = (0.24, 0.28)
    profile: float = 0.03
    plate: float = 0.006
    material: tuple = ALUMINIUM

    def link_frames(self, state) -> dict:
        return {"mast": np.eye(4), "top": Transform((0.0, 0.0, self.height)).matrix()}

    def visuals(self) -> dict:
        (lx, ly), h, p = self.footprint, self.height, self.profile
        x, y = lx / 2.0 - p / 2.0, ly / 2.0 - p / 2.0
        bars = [Visual("box", (p, p, h - self.plate), Transform((sx * x, sy * y, (h - self.plate) / 2.0)), self.material)
                for sx in (-1, 1) for sy in (-1, 1)]
        for z in (p / 2.0, h / 2.0, h - self.plate - p / 2.0):
            bars += [Visual("box", (lx - 2 * p, p, p), Transform((0.0, sy * y, z)), self.material) for sy in (-1, 1)]
            bars += [Visual("box", (p, ly - 2 * p, p), Transform((sx * x, 0.0, z)), self.material) for sx in (-1, 1)]
        bars.append(Visual("box", (lx + 0.04, ly + 0.04, self.plate), Transform((0.0, 0.0, h - self.plate / 2.0)), self.material))
        return {"mast": bars}


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
