"""Mobile bases. A robot spec gives its links (frames others attach to, and what is drawn), and a model that turns a
control into motion: kinematic (wheels roll without slipping), or with ``dynamics`` a planar rigid body driven through
its tyres' friction on the ground (hw.ground): wheels spin and slide where the grip runs out.

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
    """The Ranger's motion modes, as its remote and its CAN motion command have them. ``ackermann`` (front and rear):
    ``linear`` m/s and ``steering_deg``, the inner wheel's angle (left positive, up to 40 degrees; the rear wheels
    steer the other way). ``oblique``: every wheel at ``steering_deg`` (up to 90), the body keeps its heading.
    ``spin``: ``angular`` rad/s on the spot (counter-clockwise positive). ``lateral``: sideways at ``linear`` m/s
    (positive: left), the oblique mode at 90 degrees. ``Twist`` is the driver's cmd_vel: apply either."""
    mode: str = "ackermann"
    linear: float = 0.0
    steering_deg: float = 0.0
    angular: float = 0.0

    def twist(self, wheelbase: float, track: float = 0.364, limits: "RangerMini2 | None" = None) -> Twist:
        v, steer, spin = self.linear, self.steering_deg, self.angular
        if limits is not None:  # the chassis's own clamps (manual 3.2)
            top = limits.max_steer_deg if self.mode == "oblique" else limits.max_ackermann_deg
            steer = min(max(steer, -top), top)
            slow = limits.steering_speed_limit if abs(steer) > limits.steering_slow_deg and self.mode in ("ackermann", "oblique") \
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


@dataclass
class WheeledState:
    command: Twist = field(default_factory=Twist)
    twist: Twist = field(default_factory=Twist)        # the body's velocity now (base frame)
    steer: np.ndarray = None                           # rad per wheel
    speed: np.ndarray = None                           # m/s at each wheel's rim, along its heading
    roll: np.ndarray = None                            # rad each wheel has turned
    odometry: np.ndarray = None                        # x, y, yaw integrated from the wheels (their encoders), from the spawn pose
    setpoint: np.ndarray = None                        # the drives' speed setpoints, m/s (dynamics)
    slip: np.ndarray = None                            # m/s each tyre slides over the ground (dynamics)
    mu: np.ndarray = None                              # friction coefficient under each wheel (dynamics)
    ground: object = None                              # hw.ground.Ground, set by the World
    collided: bool = False                             # the last step was stopped by an obstacle
    contact: np.ndarray = None                         # the floor's height under each wheel (with a height map)


WHEELS = ("fl", "fr", "rl", "rr")


@dataclass
class RangerMini2(Spec):
    kind = "robot"
    wheelbase: float = 0.494
    track: float = 0.364
    wheel_radius: float = 0.100         # measured on the model's meshes
    wheel_width: float = 0.07           # estimate
    size: tuple = (0.738, 0.500, 0.338)
    clearance: float = 0.107
    kerb_mass: float = 64.5             # kg
    payload: float = 12.0               # kg carried (estimate of a mast with cameras; the limit is 80)
    max_speed: float = 1.5              # m/s
    steering_speed_limit: float = 0.75  # m/s once the steering command is past ``steering_slow_deg``
    steering_slow_deg: float = 20.0
    max_spin: float = 3.259             # rad/s
    max_ackermann_deg: float = 40.0     # the inner wheel, front-and-rear Ackermann mode
    max_steer_deg: float = 90.0         # a steering module's range to either side (the oblique mode's)
    max_accel: float = 1.5              # m/s^2: the drives' speed ramp, estimate
    max_steer_rate_deg: float = 180.0   # estimate
    steer_tolerance_deg: float = 15.0   # wheels further than this from their angle are turned before the base drives
    model_dir: str | None = None        # a converted model (tools/robot_from_usd.py): its meshes are drawn
    # --- dynamics (off: the wheels never slip) ---
    dynamics: bool = False
    motor_torque: float = 23.0          # N m at a wheel: 350 W at the top speed's 15 rad/s (derived, not from the manual)
    wheel_inertia: float = 0.02         # kg m^2, estimate
    yaw_inertia: float | None = None    # kg m^2 about the vertical axis; None: a uniform box of the robot's size
    slip_speed: float = 0.03            # m/s: below it a tyre's force rises with slip, above it is the friction limit
    drive_gain: float = 40.0            # N m per rad/s of wheel speed error (the drives' speed loop)
    substep: float = 5.0e-4             # s

    @property
    def mass(self) -> float:
        return self.kerb_mass + self.payload

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
        return WheeledState(steer=np.zeros(4), speed=np.zeros(4), roll=np.zeros(4), odometry=np.zeros(3),
                            setpoint=np.zeros(4), slip=np.zeros(4))

    def footprint(self) -> np.ndarray:
        """The body's outline on the floor: four corners (x, y) in the base frame, in order."""
        a, b = self.size[0] / 2.0, self.size[1] / 2.0
        return np.array([[-a, -b], [a, -b], [a, b], [-a, b]])

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
        """Set the command: a ``Twist`` (the driver's cmd_vel) or a ``RangerControl`` (clamped as the chassis clamps its
        CAN command). No wheel is asked for more than the top speed: a faster twist is scaled, its direction kept."""
        t = control.twist(self.wheelbase, self.track, self) if isinstance(control, RangerControl) else control
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
        if self.dynamics:
            return self._step_dynamic(state, pose, dt, want_speed)
        state.speed = state.speed + np.clip(want_speed - state.speed, -self.max_accel * dt, self.max_accel * dt)
        state.roll = state.roll + state.speed / self.wheel_radius * dt
        state.twist = t = self.body_twist(state.steer, state.speed)
        x, y, z = pose.xyz
        roll, pitch, yaw_deg = pose.rpy_deg
        yaw = math.radians(yaw_deg)
        mid = yaw + 0.5 * t.wz * dt  # midpoint heading: exact on a circle to second order
        dx, dy = (t.vx * math.cos(mid) - t.vy * math.sin(mid)) * dt, (t.vx * math.sin(mid) + t.vy * math.cos(mid)) * dt
        self._count(state, t, dt)
        return Transform((x + dx, y + dy, z), (roll, pitch, math.degrees(yaw + t.wz * dt)))

    def _count(self, state: WheeledState, wheels: Twist, dt: float) -> None:
        """Odometry: the pose the wheels' own turning adds up to (what encoders give), slip or not."""
        o = state.odometry
        mid = o[2] + 0.5 * wheels.wz * dt
        state.odometry = o + np.array([(wheels.vx * math.cos(mid) - wheels.vy * math.sin(mid)) * dt,
                                       (wheels.vx * math.sin(mid) + wheels.vy * math.cos(mid)) * dt, wheels.wz * dt])

    def _step_dynamic(self, state: WheeledState, pose: Transform, dt: float, want_speed: np.ndarray) -> Transform:
        """A planar rigid body on four tyres. Each tyre pushes against its slip over the ground (rim speed minus the
        contact point's speed along the wheel, and all of the contact point's speed across it) with a force that grows
        with the slip up to ``mu`` times its load, in the slip's direction (so driving and cornering share one
        friction circle). The drives hold their speed setpoints with a torque limit. Loads are static (a quarter of
        the weight each)."""
        xy, r, g = self.wheel_xy(), self.wheel_radius, 9.81
        mass = self.mass
        inertia = self.yaw_inertia if self.yaw_inertia is not None else mass * (self.size[0] ** 2 + self.size[1] ** 2) / 12.0
        load = mass * g / 4.0
        x, y, z = pose.xyz
        roll, pitch, yaw_deg = pose.rpy_deg
        yaw = math.radians(yaw_deg)
        ground = state.ground
        if ground is not None:  # the grip under each wheel, where it stands at the start of the step
            c0, s0 = math.cos(yaw), math.sin(yaw)
            state.mu = np.array([ground.friction(x + c0 * px - s0 * py, y + s0 * px + c0 * py) for px, py in xy])
        elif state.mu is None:
            state.mu = np.full(4, 0.6)
        grip = state.mu * load
        cs, sn = np.cos(state.steer), np.sin(state.steer)
        vx, vy, wz = state.twist.vx, state.twist.vy, state.twist.wz
        omega = state.speed / r
        n = max(1, int(math.ceil(dt / self.substep)))
        h = dt / n
        ramp = self.max_accel * h
        encoder = np.zeros(3)  # the wheels' own twist, summed
        for _ in range(n):
            state.setpoint = state.setpoint + np.clip(want_speed - state.setpoint, -ramp, ramp)
            cx, cy = vx - wz * xy[:, 1], vy + wz * xy[:, 0]              # the contact points' velocity (base frame)
            along, across = cx * cs + cy * sn, -cx * sn + cy * cs
            slide = np.sqrt((r * omega - along) ** 2 + across ** 2 + self.slip_speed ** 2)
            k = grip / slide                                              # force per unit of slip speed
            torque = np.clip(self.drive_gain * (state.setpoint / r - omega), -self.motor_torque, self.motor_torque)
            # the wheel's speed with the tyre's pull taken at the new speed (stable for stiff tyres at this step)
            omega = (omega + h * (torque + r * k * along) / self.wheel_inertia) / (1.0 + h * r * r * k / self.wheel_inertia)
            f_along, f_across = k * (r * omega - along), -k * across
            fx, fy = f_along * cs - f_across * sn, f_along * sn + f_across * cs
            ax, ay = fx.sum() / mass + wz * vy, fy.sum() / mass - wz * vx
            alpha = float((xy[:, 0] * fy - xy[:, 1] * fx).sum()) / inertia
            vx, vy, wz = vx + h * ax, vy + h * ay, wz + h * alpha
            mid = yaw + 0.5 * wz * h
            x += (vx * math.cos(mid) - vy * math.sin(mid)) * h
            y += (vx * math.sin(mid) + vy * math.cos(mid)) * h
            yaw += wz * h
            state.roll = state.roll + omega * h
            w = self.body_twist(state.steer, r * omega)
            encoder += (w.vx, w.vy, w.wz)
        state.speed = r * omega
        state.twist = Twist(float(vx), float(vy), float(wz))
        state.slip = np.hypot(r * omega - along, across)
        self._count(state, Twist(*(encoder / n)), dt)
        return Transform((x, y, z), (roll, pitch, math.degrees(yaw)))

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
