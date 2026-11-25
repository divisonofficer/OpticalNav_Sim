"""Robots by name, and what links are drawn with (``Visual``). How a base moves is hw.drive.

AgileX Ranger Mini 2.0: four independently steered and driven wheels. Numbers from the maker's user manual (V2.0.1,
2023.08, 1.2 Specifications and 3.2 CAN protocol): 738 x 500 x 338 mm, wheelbase 494 mm, track 364 mm, clearance
107 mm, kerb weight 64.5 kg, load up to 80 kg, 4 x 350 W drive motors, 1.5 m/s (0.75 m/s with more than 20 degrees of
steering), spin up to 3.259 rad/s, steering command = the inner wheel's angle: up to 0.698 rad (40 degrees) in the
front-and-rear Ackermann mode, 1.571 rad in the oblique mode. Not in the manual, estimated here: the wheel radius
(measured on the model's meshes, 0.100 m), acceleration and steering rate, wheel inertia, tyre behaviour.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .drive import FourWheelSteering, Twist, WheeledState  # noqa: F401 - WheeledState is also known by this name
from .transform import Transform


@dataclass(frozen=True)
class Visual:
    """One drawn primitive of a link, in the link's frame: a ``box`` (size x, y, z), a ``cylinder`` (radius, length;
    its axis is the link's y, a wheel's axle), a ``profile`` (side, length), a ``mesh`` file (``scale``: of the file's
    own units), or a ``light`` (width, height: a rectangle that shines along the origin's z). ``material``: (kind,
    rgb, roughness) with kind "plastic" | "metal" | "rubber" | "aluminium"; a light's is ("emitter", radiance rgb, 0)."""
    shape: str
    size: tuple
    origin: Transform = Transform()
    material: tuple = ("plastic", (0.5, 0.5, 0.5), 0.4)
    mesh: str | None = None
    scale: tuple = (1.0, 1.0, 1.0)


@dataclass
class RangerControl:
    """The Ranger's motion modes, as its remote and its CAN motion command have them. ``ackermann`` (front and rear):
    ``linear`` m/s and ``steering_deg``, the inner wheel's angle (left positive, up to 40 degrees; the rear wheels
    steer the other way). ``oblique``: every wheel at ``steering_deg`` (up to 90), the body keeps its heading.
    ``spin``: ``angular`` rad/s on the spot (counter-clockwise positive). ``lateral``: sideways at ``linear`` m/s
    (positive: left), the oblique mode at 90 degrees. ``Twist`` is the driver's cmd_vel: apply either."""
    mode: str = "ackermann"
    linear: float = 0.0
    steering_deg: float = 0.0
    angular: float = 0.0

    def twist(self, wheelbase: float, track: float = 0.364, limits: "FourWheelSteering | None" = None) -> Twist:
        v, steer, spin = self.linear, self.steering_deg, self.angular
        if limits is not None:  # the chassis's own clamps (manual 3.2)
            top = limits.max_steer_deg if self.mode == "oblique" else limits.max_ackermann_deg
            steer = min(max(steer, -top), top)
            slow = (limits.steering_speed_limit or limits.max_speed) if abs(steer) > limits.steering_slow_deg and self.mode in ("ackermann", "oblique") \
                else limits.max_speed
            v, spin = min(max(v, -slow), slow), min(max(spin, -limits.max_spin), limits.max_spin)
        d = math.radians(steer)
        if self.mode == "ackermann":
            if abs(d) < 1e-9:
                return Twist(v, 0.0, 0.0)
            # the inner wheel at d: the turning centre is abreast of the middle, (L/2) / tan|d| beyond the inner wheels
            radius = (wheelbase / 2.0) / math.tan(abs(d)) + track / 2.0
            return Twist(v, 0.0, math.copysign(v / radius, d * v) if v else 0.0)
        if self.mode == "oblique":
            return Twist(v * math.cos(d), v * math.sin(d), 0.0)
        if self.mode == "spin":
            return Twist(0.0, 0.0, spin)
        if self.mode == "lateral":
            return Twist(0.0, v, 0.0)
        raise ValueError(f"Ranger mode {self.mode!r}: ackermann, oblique, spin or lateral")


WHEELS = ("fl", "fr", "rl", "rr")


@dataclass
class RangerMini2(FourWheelSteering):
    """AgileX Ranger Mini 2.0: the four-wheel steering drive with the maker's numbers (the module's note)."""
    wheelbase: float = 0.494
    track: float = 0.364
    wheel_radius: float = 0.100         # measured on the model's meshes
    wheel_width: float = 0.07           # estimate
    size: tuple = (0.738, 0.500, 0.338)
    clearance: float = 0.107
    kerb_mass: float = 64.5             # kg
    payload: float = 12.0               # kg carried (estimate of a mast with cameras)
    max_payload: float | None = 80.0
    max_speed: float = 1.5              # m/s
    steering_speed_limit: float | None = 0.75  # m/s once the steering command is past ``steering_slow_deg``
    max_spin: float = 3.259             # rad/s
    max_accel: float = 1.5              # m/s^2, estimate
    motor_torque: float = 23.0          # N m at a wheel: 350 W at the top speed's 15 rad/s (derived, not from the manual)
    wheel_inertia: float = 0.02         # kg m^2, estimate
    model_dir: str | None = None        # a converted model (tools/robot_from_usd.py): its meshes are drawn

    @classmethod
    def from_model(cls, model_dir, **kw) -> "RangerMini2":
        """The robot as a converted model has it: its meshes, and the wheelbase, track, wheel size and height measured
        on them."""
        m = json.loads((Path(model_dir) / "model.json").read_text())
        return cls(wheelbase=m["wheelbase"], track=m["track"], wheel_radius=m["wheel_radius"], wheel_width=m["wheel_width"],
                   size=tuple(m["size"]), clearance=m["clearance"], model_dir=str(Path(model_dir).resolve()), **kw)

    def visuals(self) -> dict:
        """Link name -> what is drawn for it: the converted model's meshes, or placeholders with the robot's outer
        dimensions."""
        if self.model_dir and self.link_visuals is None:
            links = json.loads((Path(self.model_dir) / "model.json").read_text())["links"]
            # the model's chassis frame is this spec's base_link (on the floor); its wheel frames are the wheels' centres
            return {("base_link" if link == "chassis" else link):
                    [Visual("mesh", (), Transform(), (v["material"][0], tuple(v["material"][1]), v["material"][2]),
                            str(Path(self.model_dir) / v["mesh"])) for v in visuals]
                    for link, visuals in links.items()}
        return super().visuals()


MODELS = Path(os.environ.get("OPTICALNAV_HARDWARE", Path(__file__).resolve().parents[2] / "packs" / "hardware"))


def ranger_mini(**kw) -> RangerMini2:
    """The Ranger Mini with its converted meshes when they are there (``$OPTICALNAV_HARDWARE/ranger_mini_v3``, default
    ``packs/hardware/ranger_mini_v3``: tools/robot_from_usd.py), else with placeholder shapes."""
    model = MODELS / "ranger_mini_v3"
    return RangerMini2.from_model(model, **kw) if (model / "model.json").is_file() else RangerMini2(**kw)
