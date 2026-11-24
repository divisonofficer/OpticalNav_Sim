"""Mobile bases. A robot spec gives its links (frames others attach to, and what is drawn), and a kinematic model that
turns a control into motion. No contact forces: wheels do not slip unless the model says so.

AgileX Ranger Mini 2.0: four independently steered and driven wheels (4WIS). Dimensions from Weston Robot's guide
(738 x 500 x 338 mm, 107 mm clearance) and a reseller's sheet (wheelbase 494 mm, track 364 mm); the wheel radius, the
speed and steering limits and the drawn shapes are estimates until the vendor's URDF is in (docs/hardware_platform_plan.md, S3).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .actor import Spec
from .transform import Transform, rpy_matrix


@dataclass(frozen=True)
class Visual:
    """One drawn primitive of a link, in the link's frame: a ``box`` (size x, y, z) or a ``cylinder`` (radius, length;
    its axis is the link's y, a wheel's axle), or a ``mesh`` file. ``material``: (kind, rgb, roughness) with kind
    "plastic" | "metal" | "rubber"."""
    shape: str
    size: tuple
    origin: Transform = Transform()
    material: tuple = ("plastic", (0.5, 0.5, 0.5), 0.4)
    mesh: str | None = None


@dataclass
class Twist:
    """Body velocity in the base frame: forward and left (m/s), turn rate (rad/s, counter-clockwise)."""
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0


@dataclass
class RangerControl:
    """The Ranger's motion modes, as its remote and driver have them. ``ackermann``: ``linear`` m/s and
    ``steering_deg`` (front wheels' mean angle; the rear wheels steer the other way). ``oblique``: every wheel at
    ``steering_deg``, the body keeps its heading. ``spin``: ``angular`` rad/s on the spot. ``lateral``: sideways at
    ``linear`` m/s (positive: left). The driver's cmd_vel is ``Twist``: apply either."""
    mode: str = "ackermann"
    linear: float = 0.0
    steering_deg: float = 0.0
    angular: float = 0.0

    def twist(self, wheelbase: float) -> Twist:
        d = math.radians(self.steering_deg)
        if self.mode == "ackermann":  # both axles steer: the turning centre is on the line through the middle
            return Twist(self.linear, 0.0, self.linear * math.tan(d) / (wheelbase / 2.0))
        if self.mode == "oblique":
            return Twist(self.linear * math.cos(d), self.linear * math.sin(d), 0.0)
        if self.mode == "spin":
            return Twist(0.0, 0.0, self.angular)
        if self.mode == "lateral":
            return Twist(0.0, self.linear, 0.0)
        raise ValueError(f"Ranger mode {self.mode!r}: ackermann, oblique, spin or lateral")


@dataclass
class WheeledState:
    command: Twist = field(default_factory=Twist)
    twist: Twist = field(default_factory=Twist)        # what the wheels give now (base frame)
    steer: np.ndarray = None                           # rad per wheel
    speed: np.ndarray = None                           # m/s along each wheel's heading
    roll: np.ndarray = None                            # rad each wheel has turned
    odometry: np.ndarray = None                        # x, y, yaw integrated from the wheels, from the spawn pose


WHEELS = ("fl", "fr", "rl", "rr")


@dataclass
class RangerMini2(Spec):
    kind = "robot"
    wheelbase: float = 0.494
    track: float = 0.364
    wheel_radius: float = 0.09          # estimate
    wheel_width: float = 0.07           # estimate
    size: tuple = (0.738, 0.500, 0.338)
    clearance: float = 0.107
    max_speed: float = 1.5              # m/s, estimate
    max_accel: float = 1.5              # m/s^2 at a wheel, estimate
    max_steer_deg: float = 90.0         # a steering module's range to either side
    max_steer_rate_deg: float = 180.0   # estimate
    steer_tolerance_deg: float = 15.0   # wheels further than this from their angle are turned before the base drives
    model_dir: str | None = None        # a converted model (tools/robot_from_usd.py): its meshes are drawn

    @classmethod
    def from_model(cls, model_dir, **kw) -> "RangerMini2":
        """The robot as a converted model has it: its meshes, and the wheelbase, track, wheel size and height measured
        on them."""
        m = json.loads((Path(model_dir) / "model.json").read_text())
        return cls(wheelbase=m["wheelbase"], track=m["track"], wheel_radius=m["wheel_radius"], wheel_width=m["wheel_width"],
                   size=tuple(m["size"]), clearance=m["clearance"], model_dir=str(Path(model_dir).resolve()), **kw)

    def wheel_xy(self) -> np.ndarray:
        a, b = self.wheelbase / 2.0, self.track / 2.0
        return np.array([[a, b], [a, -b], [-a, b], [-a, -b]])

    def make_state(self) -> WheeledState:
        return WheeledState(steer=np.zeros(4), speed=np.zeros(4), roll=np.zeros(4), odometry=np.zeros(3))

    # --- kinematics -------------------------------------------------------------------------------------------------
    def wheel_targets(self, twist: Twist, steer_now: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Steering angle and signed speed of each wheel for a body twist. A wheel drives either way, so of the two
        ways to make a velocity (angle, speed) and (angle + 180 deg, -speed) the one nearer its present angle within
        the steering range is taken; a wheel that should stand still keeps its angle."""
        xy = self.wheel_xy()
        v = np.stack([twist.vx - twist.wz * xy[:, 1], twist.vy + twist.wz * xy[:, 0]], 1)
        speed = np.hypot(v[:, 0], v[:, 1])
        angle = np.arctan2(v[:, 1], v[:, 0])
        steer, out = steer_now.copy(), np.zeros(4)
        lim = math.radians(self.max_steer_deg) + 1e-9
        for i in range(4):
            if speed[i] < 1e-6:
                continue
            options = [(angle[i], speed[i]), (angle[i] - math.copysign(math.pi, angle[i]), -speed[i])]
            options = [o for o in options if abs(o[0]) <= lim] or [(float(np.clip(angle[i], -lim, lim)), speed[i])]
            steer[i], out[i] = min(options, key=lambda o: abs(o[0] - steer_now[i]))
        return steer, out

    def body_twist(self, steer: np.ndarray, speed: np.ndarray) -> Twist:
        """The body twist the wheels make without slipping (least squares over the four wheels: exact when they agree)."""
        xy = self.wheel_xy()
        a, b = np.zeros((8, 3)), np.zeros(8)
        for i in range(4):
            a[2 * i] = (1.0, 0.0, -xy[i, 1])
            a[2 * i + 1] = (0.0, 1.0, xy[i, 0])
            b[2 * i], b[2 * i + 1] = speed[i] * math.cos(steer[i]), speed[i] * math.sin(steer[i])
        vx, vy, wz = np.linalg.lstsq(a, b, rcond=None)[0]
        return Twist(float(vx), float(vy), float(wz))

    def apply(self, state: WheeledState, control) -> None:
        """Set the command: a ``Twist`` (the driver's cmd_vel) or a ``RangerControl``. Speeds over the limit are scaled."""
        t = control.twist(self.wheelbase) if isinstance(control, RangerControl) else control
        xy = self.wheel_xy()
        fastest = float(np.max(np.hypot(t.vx - t.wz * xy[:, 1], t.vy + t.wz * xy[:, 0])))
        k = min(1.0, self.max_speed / fastest) if fastest > 0.0 else 1.0
        state.command = Twist(t.vx * k, t.vy * k, t.wz * k)

    def step(self, state: WheeledState, pose: Transform, dt: float) -> Transform:
        want_steer, want_speed = self.wheel_targets(state.command, state.steer)
        turn = math.radians(self.max_steer_rate_deg) * dt
        error = want_steer - state.steer
        state.steer = state.steer + np.clip(error, -turn, turn)
        # Wheels far from their angle (a change of mode: forward to sideways) are turned with the base at rest, as the
        # robot does; within ``steer_tolerance_deg`` they drive on, a little slower, while they finish turning.
        off = float(np.max(np.abs(want_steer - state.steer)))
        want_speed = want_speed * (math.cos(off) if off <= math.radians(self.steer_tolerance_deg) else 0.0)
        state.speed = state.speed + np.clip(want_speed - state.speed, -self.max_accel * dt, self.max_accel * dt)
        state.roll = state.roll + state.speed / self.wheel_radius * dt
        state.twist = t = self.body_twist(state.steer, state.speed)
        x, y, z = pose.xyz
        roll, pitch, yaw_deg = pose.rpy_deg
        yaw = math.radians(yaw_deg)
        mid = yaw + 0.5 * t.wz * dt  # midpoint heading: exact on a circle to second order
        dx, dy = (t.vx * math.cos(mid) - t.vy * math.sin(mid)) * dt, (t.vx * math.sin(mid) + t.vy * math.cos(mid)) * dt
        o = state.odometry
        om = o[2] + 0.5 * t.wz * dt
        state.odometry = o + np.array([(t.vx * math.cos(om) - t.vy * math.sin(om)) * dt,
                                       (t.vx * math.sin(om) + t.vy * math.cos(om)) * dt, t.wz * dt])
        return Transform((x + dx, y + dy, z), (roll, pitch, math.degrees(yaw + t.wz * dt)))

    # --- frames and shapes ------------------------------------------------------------------------------------------
    def link_frames(self, state: WheeledState | None) -> dict:
        """``base_link``: on the floor under the centre (x forward). ``chassis``: the body's centre. ``top_plate``: the
        centre of the deck, where equipment is bolted. ``steer_*``: a steering module (turns about z). ``wheel_*``:
        a wheel (turns about its axle, y)."""
        steer = state.steer if state is not None else np.zeros(4)
        roll = state.roll if state is not None else np.zeros(4)
        out = {"base_link": np.eye(4)}
        out["chassis"] = Transform((0.0, 0.0, (self.clearance + self.size[2]) / 2.0)).matrix()
        out["top_plate"] = Transform((0.0, 0.0, self.size[2])).matrix()
        for name, (x, y), s, r in zip(WHEELS, self.wheel_xy(), steer, roll):
            m = np.eye(4)
            m[:3, :3] = rpy_matrix(0.0, 0.0, float(s))
            m[:3, 3] = (x, y, self.wheel_radius)
            out[f"steer_{name}"] = m
            w = m.copy()
            w[:3, :3] = m[:3, :3] @ rpy_matrix(0.0, float(r), 0.0)
            out[f"wheel_{name}"] = w
        return out

    def visuals(self) -> dict:
        """Link name -> what is drawn for it: the converted model's meshes, or placeholders with the robot's outer
        dimensions."""
        if self.model_dir:
            links = json.loads((Path(self.model_dir) / "model.json").read_text())["links"]
            # the model's chassis frame is this spec's base_link (on the floor); its wheel frames are the wheels' centres
            return {("base_link" if link == "chassis" else link):
                    [Visual("mesh", (), Transform(), (v["material"][0], tuple(v["material"][1]), v["material"][2]),
                            str(Path(self.model_dir) / v["mesh"])) for v in visuals]
                    for link, visuals in links.items()}
        L, W, H = self.size
        body_h = H - self.clearance - 0.03
        dark, grey, tyre = ("plastic", (0.05, 0.05, 0.06), 0.35), ("metal", (0.75, 0.76, 0.78), 0.3), ("rubber", (0.03, 0.03, 0.03), 0.6)
        out = {"chassis": [Visual("box", (L - 2.2 * self.wheel_radius, self.track - self.wheel_width - 0.04, body_h),
                                  Transform((0.0, 0.0, -0.015)), dark)],
               "top_plate": [Visual("box", (L - 0.2, W - 0.14, 0.03), Transform((0.0, 0.0, -0.015)), grey)]}
        for name in WHEELS:
            out[f"steer_{name}"] = [Visual("box", (0.06, 0.05, 0.10), Transform((0.0, 0.0, self.wheel_radius + 0.05)), grey)]
            out[f"wheel_{name}"] = [Visual("cylinder", (self.wheel_radius, self.wheel_width), Transform(), tyre),
                                    Visual("box", (0.02, self.wheel_width + 0.004, self.wheel_radius), Transform(), grey)]
        return out


MODELS = Path(os.environ.get("OPTICALNAV_HARDWARE", Path(__file__).resolve().parents[2] / "packs" / "hardware"))


def ranger_mini(**kw) -> RangerMini2:
    """The Ranger Mini with its converted meshes when they are there (``$OPTICALNAV_HARDWARE/ranger_mini_v3``, default
    ``packs/hardware/ranger_mini_v3``: tools/robot_from_usd.py), else with placeholder shapes."""
    model = MODELS / "ranger_mini_v3"
    return RangerMini2.from_model(model, **kw) if (model / "model.json").is_file() else RangerMini2(**kw)
