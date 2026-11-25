"""Mobile bases: what moves a robot. ``Drive`` is the contract a base fulfils towards the world and the viewer;
``WheeledDrive`` is every base on rolling wheels (kinematic, or a planar rigid body on its tyres' friction), and a drive
type says only where its wheels are and how a motion becomes wheel angles and speeds:

    FourWheelSteering    every wheel steers and drives (Ackermann, oblique, sideways, spin): the Ranger Mini
    DifferentialDrive    two driven wheels on one axle and casters: the TurtleBot 4

A new drive type implements ``wheel_xy`` and ``wheel_targets`` (and, where they differ, ``capabilities``, ``intent``,
``limit``, ``link_frames``, ``visuals``). A base that does not roll (legs, tracks) subclasses ``Drive`` itself.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .actor import Spec
from .transform import Transform, rpy_matrix


@dataclass
class Twist:
    """Body velocity in the base frame: forward and left (m/s), turn rate (rad/s, counter-clockwise)."""
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0


@dataclass
class DriveState:
    """What every drive's state has (the world, the viewer and recordings read it)."""
    command: Twist = field(default_factory=Twist)
    twist: Twist = field(default_factory=Twist)        # the body's velocity now (base frame)
    odometry: np.ndarray = field(default_factory=lambda: np.zeros(3))   # x, y, yaw the base itself counts, from where it started
    ground: object = None                              # hw.ground.Ground, set when it enters a world
    collided: bool = False                             # the last step was stopped by an obstacle
    contact: np.ndarray = None                         # the floor's height under each contact point (with a height map)


@dataclass
class WheeledState(DriveState):
    steer: np.ndarray = None                           # rad per wheel (0 for a wheel that does not steer)
    speed: np.ndarray = None                           # m/s at each wheel's rim, along its heading
    roll: np.ndarray = None                            # rad each wheel has turned
    setpoint: np.ndarray = None                        # the drives' speed setpoints, m/s (dynamics)
    slip: np.ndarray = None                            # m/s each tyre slides over the ground (dynamics)
    mu: np.ndarray = None                              # friction coefficient under each wheel (dynamics)


@dataclass
class Drive(Spec):
    """A base that moves over the ground. Subclasses give ``contacts``, ``footprint``, ``apply`` and ``step``; the rest
    has defaults."""
    kind = "robot"

    # --- what a drive type must give --------------------------------------------------------------------------------
    def contacts(self) -> np.ndarray:
        """Where it touches the floor (N x 2, base frame): its height, roll and pitch follow the floor under these."""
        raise NotImplementedError

    def footprint(self) -> np.ndarray:
        """Its outline on the floor (N x 2, a convex polygon in order, base frame)."""
        raise NotImplementedError

    def apply(self, state, control) -> None:
        """Set the command: a ``Twist``, or a control of the drive's own."""
        raise NotImplementedError

    # --- what it may say about itself --------------------------------------------------------------------------------
    def capabilities(self) -> dict:
        """For a driver's controls: which of them mean something. ``steer_deg``: the largest steering angle, or None."""
        return {"type": type(self).__name__, "max_speed": None, "max_spin": None, "lateral": False, "steer_deg": None}

    def intent(self, forward: float = 0.0, turn: float = 0.0, side: float = 0.0, speed: float = 0.5, steer_deg: float = 20.0):
        """What a driver's keys ask for (each -1, 0 or 1; forward, left, left) as a control of this drive."""
        return Twist(forward * speed, 0.0, turn * 1.0)

    def control(self, body: dict):
        """A control from a message: ``{"mode": "twist", vx, vy, wz}`` or ``{"mode": "drive", forward, turn, side,
        speed, steer_deg}``. A drive type adds its own modes."""
        num = lambda k, d=0.0: float(body.get(k, d))  # noqa: E731
        mode = str(body.get("mode", "twist"))
        if mode == "twist":
            return Twist(num("vx"), num("vy"), num("wz"))
        if mode == "drive":
            return self.intent(num("forward"), num("turn"), num("side"), num("speed", 0.5), num("steer_deg", 20.0))
        raise ValueError(f"{type(self).__name__} has no control mode {mode!r}")

    def wheel_report(self, state) -> list[dict]:
        """Per wheel, what a display shows: name, label, steer_deg (None: does not steer), speed, slip, mu."""
        return []

    def stop(self, state, keep: str = "") -> None:
        """An obstacle took the motion: all of it, or what is not in ``keep`` ("turn", "travel")."""
        t = state.twist
        state.twist = Twist(0.0, 0.0, t.wz) if keep == "turn" else Twist(t.vx, t.vy, 0.0) if keep == "travel" else Twist()

    def posture(self, state) -> np.ndarray:
        """Numbers that change when a link moves against the base (degrees): a display redraws when they do."""
        return np.zeros(0)

    def mounts(self) -> dict:
        """Named places equipment goes on: name -> (link, Transform). ``deck``: the top of the robot."""
        first = next(iter(self.link_frames(None)))
        return {"deck": (first, Transform())}

    # --- with the world ----------------------------------------------------------------------------------------------
    def enter(self, state, world) -> None:
        state.ground = world.ground

    def settle(self, state, old: Transform, pose: Transform, ground, _partial: bool = False):
        """The pose after a step, as the ground allows it: stopped where the footprint would cover something higher than
        a step (what is free of the step is still taken: the turn, the travel, or the slide along a wall), else set on
        the floor under the contact points: their heights give the base's height, roll and pitch."""
        if ground.heights is None:
            state.collided = False
            return pose
        yaw = math.radians(pose.rpy_deg[2])
        c, s = math.cos(yaw), math.sin(yaw)
        points = np.asarray(self.contacts(), dtype=np.float64)
        # each contact's height: the floor under it, unless that is more than a step from where it stood (a kerb, a
        # wall, a drop: it does not take that, and the footprint test below stops the robot)
        last = state.contact
        z = np.array([ground.height(pose.xyz[0] + c * px - s * py, pose.xyz[1] + s * px + c * py,
                                    near=None if last is None else float(last[i])) for i, (px, py) in enumerate(points)])
        # the plane through the contact points, in the base's own axes: z = a x + b y + z0
        a, b, z0 = (float(v) for v in np.linalg.lstsq(np.c_[points, np.ones(len(points))], z, rcond=None)[0])
        foot = np.asarray(self.footprint(), dtype=np.float64)
        corners = (np.c_[foot, np.zeros(len(foot)), np.ones(len(foot))] @ pose.matrix().T)[:, :2]
        if ground.blocked(corners, z0, (a * c - b * s, a * s + b * c), pose.xyz[:2]):
            if _partial:
                return None
            moved = math.hypot(pose.xyz[0] - old.xyz[0], pose.xyz[1] - old.xyz[1]) > 1e-9
            turned = abs(pose.rpy_deg[2] - old.rpy_deg[2]) > 1e-9
            parts = []
            if moved and turned:
                parts = [("turn", Transform((old.xyz[0], old.xyz[1], pose.xyz[2]), pose.rpy_deg)),
                         ("travel", Transform(pose.xyz, (pose.rpy_deg[0], pose.rpy_deg[1], old.rpy_deg[2])))]
            if moved:  # along a wall: the step's part along world x, or along world y
                parts += [("slide", Transform((pose.xyz[0], old.xyz[1], pose.xyz[2]), pose.rpy_deg)),
                          ("slide", Transform((old.xyz[0], pose.xyz[1], pose.xyz[2]), pose.rpy_deg))]
            for kind, part in parts:
                if math.hypot(part.xyz[0] - old.xyz[0], part.xyz[1] - old.xyz[1]) < 1e-9 and \
                        abs(part.rpy_deg[2] - old.rpy_deg[2]) < 1e-9:
                    continue
                done = self.settle(state, old, part, ground, _partial=True)
                if done is not None:
                    state.collided = True   # it touched, and gave way
                    if kind != "slide":
                        self.stop(state, keep=kind)
                    return done
            state.collided = True
            self.stop(state)
            return old
        state.collided, state.contact = False, z
        return Transform((pose.xyz[0], pose.xyz[1], z0), (math.degrees(math.atan(b)), -math.degrees(math.atan(a)), pose.rpy_deg[2]))


@dataclass
class WheeledDrive(Drive):
    """A base on rolling wheels, some or all of them steered. ``dynamics`` off: they never slip. On: a planar rigid body
    on its tyres (``_step_dynamic``). Casters carry load and do nothing else."""
    wheel_radius: float = 0.1
    wheel_width: float = 0.05
    size: tuple = (0.6, 0.5, 0.3)       # length, width, height of the body, m
    clearance: float = 0.05
    kerb_mass: float = 20.0             # kg
    payload: float = 0.0                # kg carried (hw.config adds what is mounted)
    max_payload: float | None = None    # kg, the maker's limit
    com_xy: tuple = (0.0, 0.0)          # where the mass is, base frame: the wheels' share of the load
    max_speed: float = 1.0              # m/s
    max_spin: float = 2.0               # rad/s
    max_accel: float = 1.0              # m/s^2: the drives' speed ramp
    max_steer_rate_deg: float = 180.0
    steer_tolerance_deg: float = 15.0   # wheels further than this from their angle are turned before the base drives
    link_visuals: dict | None = None    # link -> [Visual], given instead of the placeholder (a URDF's, a structure file's)
    # --- dynamics ---
    dynamics: bool = False
    motor_torque: float = 10.0          # N m at a wheel
    wheel_inertia: float = 0.01         # kg m^2
    yaw_inertia: float | None = None    # kg m^2 about the vertical axis; None: a uniform box of the robot's size
    slip_speed: float = 0.03            # m/s: below it a tyre's force rises with slip, above it is the friction limit
    drive_gain: float = 40.0            # N m per rad/s of wheel speed error (the drives' speed loop)
    substep: float = 5.0e-4             # s

    # --- the wheels --------------------------------------------------------------------------------------------------
    def wheel_xy(self) -> np.ndarray:
        """The driven wheels' contact points (N x 2, base frame)."""
        raise NotImplementedError

    def wheel_targets(self, twist: Twist, steer_now: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Steering angle and signed speed of each wheel for a body twist."""
        raise NotImplementedError

    def wheel_names(self) -> list[tuple[str, str]]:
        """(link suffix, label) of each wheel."""
        return [(str(i), f"바퀴 {i + 1}") for i in range(len(self.wheel_xy()))]

    def caster_xy(self) -> np.ndarray:
        return np.zeros((0, 2))

    def steers(self) -> bool:
        return False

    def limit(self, t: Twist) -> Twist:
        """A twist as the base can do it: no wheel over the top speed (the direction kept), no faster spin than it has."""
        xy = self.wheel_xy()
        fastest = float(np.max(np.hypot(t.vx - t.wz * xy[:, 1], t.vy + t.wz * xy[:, 0])))
        k = min(1.0, self.max_speed / fastest) if fastest > 0.0 else 1.0
        return Twist(t.vx * k, t.vy * k, t.wz * k)

    # --- Drive ---------------------------------------------------------------------------------------------------------
    @property
    def mass(self) -> float:
        return self.kerb_mass + self.payload

    def contacts(self) -> np.ndarray:
        return np.vstack([self.wheel_xy(), self.caster_xy()])

    def footprint(self) -> np.ndarray:
        a, b = self.size[0] / 2.0, self.size[1] / 2.0
        return np.array([[-a, -b], [a, -b], [a, b], [-a, b]])

    def load_share(self) -> np.ndarray:
        """The share of the weight on each contact point (static, level ground): forces and moments about ``com_xy``
        in balance, as even as that allows. A contact that would have to pull carries nothing."""
        p = self.contacts() - np.asarray(self.com_xy)
        share = np.maximum(np.linalg.pinv(np.vstack([np.ones(len(p)), p.T])) @ np.array([1.0, 0.0, 0.0]), 0.0)
        return share / share.sum()

    def make_state(self) -> WheeledState:
        n = len(self.wheel_xy())
        return WheeledState(steer=np.zeros(n), speed=np.zeros(n), roll=np.zeros(n), setpoint=np.zeros(n), slip=np.zeros(n))

    def to_twist(self, control) -> Twist:
        return control

    def apply(self, state, control) -> None:
        state.command = self.limit(self.to_twist(control))

    def capabilities(self) -> dict:
        return {"type": type(self).__name__, "max_speed": self.max_speed, "max_spin": self.max_spin, "lateral": False, "steer_deg": None}

    def intent(self, forward=0.0, turn=0.0, side=0.0, speed=0.5, steer_deg=20.0):
        return Twist(forward * min(speed, self.max_speed), 0.0, turn * min(1.0, self.max_spin))

    def wheel_report(self, state) -> list[dict]:
        zeros = np.zeros(len(self.wheel_xy()))
        mu, slip = (zeros if v is None else v for v in (state.mu, state.slip))
        return [{"name": n, "label": label, "steer_deg": round(math.degrees(float(state.steer[i])), 1) if self.steers() else None,
                 "speed": round(float(state.speed[i]), 3), "slip": round(float(slip[i]), 3), "mu": round(float(mu[i]), 3)}
                for i, (n, label) in enumerate(self.wheel_names())]

    def stop(self, state, keep: str = "") -> None:
        super().stop(state, keep)
        if not keep:
            state.speed, state.setpoint = np.zeros_like(state.speed), np.zeros_like(state.speed)

    def posture(self, state) -> np.ndarray:
        return np.degrees(state.steer)

    def mounts(self) -> dict:
        return {"deck": ("top_plate", Transform())}

    def body_twist(self, steer: np.ndarray, speed: np.ndarray) -> Twist:
        """The body twist the wheels make without slipping (least squares over the wheels: exact when they agree)."""
        xy = self.wheel_xy()
        n = len(xy)
        a, b = np.zeros((2 * n, 3)), np.zeros(2 * n)
        for i in range(n):
            a[2 * i] = (1.0, 0.0, -xy[i, 1])
            a[2 * i + 1] = (0.0, 1.0, xy[i, 0])
            b[2 * i], b[2 * i + 1] = speed[i] * math.cos(steer[i]), speed[i] * math.sin(steer[i])
        vx, vy, wz = np.linalg.lstsq(a, b, rcond=None)[0]
        return Twist(float(vx), float(vy), float(wz))

    def step(self, state: WheeledState, pose: Transform, dt: float) -> Transform:
        want_steer, want_speed = self.wheel_targets(state.command, state.steer)
        turn = math.radians(self.max_steer_rate_deg) * dt
        state.steer = state.steer + np.clip(want_steer - state.steer, -turn, turn)
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
        """A planar rigid body on its tyres. Each tyre pushes against its slip over the ground (rim speed minus the
        contact point's speed along the wheel, and all of the contact point's speed across it) with a force that grows
        with the slip up to ``mu`` times its load, in the slip's direction (so driving and cornering share one
        friction circle). The drives hold their speed setpoints with a torque limit. Loads are static
        (``load_share``); casters carry theirs without friction."""
        xy, r, g = self.wheel_xy(), self.wheel_radius, 9.81
        mass, n_wheels = self.mass, len(xy)
        inertia = self.yaw_inertia if self.yaw_inertia is not None else mass * (self.size[0] ** 2 + self.size[1] ** 2) / 12.0
        load = mass * g * self.load_share()[:n_wheels]
        x, y, z = pose.xyz
        roll, pitch, yaw_deg = pose.rpy_deg
        yaw = math.radians(yaw_deg)
        ground = state.ground
        if ground is not None:  # the grip under each wheel, where it stands at the start of the step
            c0, s0 = math.cos(yaw), math.sin(yaw)
            state.mu = np.array([ground.friction(x + c0 * px - s0 * py, y + s0 * px + c0 * py) for px, py in xy])
        elif state.mu is None:
            state.mu = np.full(n_wheels, 0.6)
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
            # the wheel's speed with the tyre's pull and the drive's torque taken at the new speed (stable for stiff tyres
            # and for a strong drive on a light wheel); a drive at its torque limit pushes with that
            q, tyre = h / self.wheel_inertia, h * r * r * k / self.wheel_inertia
            free = (omega + q * (self.drive_gain * state.setpoint / r + r * k * along)) / (1.0 + tyre + q * self.drive_gain)
            torque = self.drive_gain * (state.setpoint / r - free)
            pushed = (omega + q * (np.clip(torque, -self.motor_torque, self.motor_torque) + r * k * along)) / (1.0 + tyre)
            omega = np.where(np.abs(torque) <= self.motor_torque, free, pushed)
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

    # --- frames ------------------------------------------------------------------------------------------------------
    def link_frames(self, state: WheeledState | None) -> dict:
        """``base_link``: on the floor under the centre (x forward). ``chassis``: the body's centre. ``top_plate``: the
        centre of its top, where equipment is bolted. ``steer_*`` (a steered wheel's module, turning about z) and
        ``wheel_*`` (a wheel, turning about its axle, y)."""
        n = len(self.wheel_xy())
        steer = state.steer if state is not None else np.zeros(n)
        roll = state.roll if state is not None else np.zeros(n)
        out = {"base_link": np.eye(4)}
        out["chassis"] = Transform((0.0, 0.0, (self.clearance + self.size[2]) / 2.0)).matrix()
        out["top_plate"] = Transform((0.0, 0.0, self.size[2])).matrix()
        for (name, _), (x, y), s, r in zip(self.wheel_names(), self.wheel_xy(), steer, roll):
            m = np.eye(4)
            m[:3, :3] = rpy_matrix(0.0, 0.0, float(s))
            m[:3, 3] = (x, y, self.wheel_radius)
            if self.steers():
                out[f"steer_{name}"] = m
            w = m.copy()
            w[:3, :3] = m[:3, :3] @ rpy_matrix(0.0, float(r), 0.0)
            out[f"wheel_{name}"] = w
        return out


@dataclass
class FourWheelSteering(WheeledDrive):
    """Four independently steered and driven wheels, in the order front left, front right, rear left, rear right."""
    wheelbase: float = 0.5
    track: float = 0.4
    steering_speed_limit: float | None = None   # m/s once the steering command is past ``steering_slow_deg``
    steering_slow_deg: float = 20.0
    max_ackermann_deg: float = 40.0     # the inner wheel, front-and-rear Ackermann mode
    max_steer_deg: float = 90.0         # a steering module's range to either side (the oblique mode's)

    def wheel_xy(self) -> np.ndarray:
        a, b = self.wheelbase / 2.0, self.track / 2.0
        return np.array([[a, b], [a, -b], [-a, b], [-a, -b]])

    def wheel_names(self):
        return [("fl", "앞 왼쪽"), ("fr", "앞 오른쪽"), ("rl", "뒤 왼쪽"), ("rr", "뒤 오른쪽")]

    def steers(self) -> bool:
        return True

    def capabilities(self) -> dict:
        return dict(super().capabilities(), lateral=True, steer_deg=self.max_ackermann_deg)

    def intent(self, forward=0.0, turn=0.0, side=0.0, speed=0.5, steer_deg=20.0):
        from .robots import RangerControl
        if side:
            return RangerControl("lateral", side * min(speed, self.steering_speed_limit or self.max_speed))
        if forward:
            return RangerControl("ackermann", forward * speed, turn * steer_deg)
        return RangerControl("spin", angular=turn * 1.0) if turn else Twist()

    def control(self, body: dict):
        from .robots import RangerControl
        mode = str(body.get("mode", "ackermann"))
        if mode in ("twist", "drive"):
            return super().control(dict(body, mode=mode))
        c = RangerControl(mode, float(body.get("linear", 0.0)), float(body.get("steering_deg", 0.0)), float(body.get("angular", 0.0)))
        c.twist(self.wheelbase, self.track)  # an unknown mode fails here
        return c

    def to_twist(self, control) -> Twist:
        return control if isinstance(control, Twist) else control.twist(self.wheelbase, self.track, self)

    def wheel_targets(self, twist: Twist, steer_now: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A wheel drives either way, so of the two ways to make a velocity (angle, speed) and (angle + 180 deg,
        -speed) the one nearer its present angle within the steering range is taken; a wheel that should stand still
        keeps its angle."""
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

    def visuals(self) -> dict:
        from .robots import Visual
        if self.link_visuals is not None:
            return {link: list(vs) for link, vs in self.link_visuals.items() if vs}
        L, W, H = self.size
        body_h = H - self.clearance - 0.03
        dark, grey, tyre = ("plastic", (0.05, 0.05, 0.06), 0.35), ("metal", (0.75, 0.76, 0.78), 0.3), ("rubber", (0.03, 0.03, 0.03), 0.6)
        out = {"chassis": [Visual("box", (L - 2.2 * self.wheel_radius, self.track - self.wheel_width - 0.04, body_h),
                                  Transform((0.0, 0.0, -0.015)), dark)],
               "top_plate": [Visual("box", (L - 0.2, W - 0.14, 0.03), Transform((0.0, 0.0, -0.015)), grey)]}
        for name, _ in self.wheel_names():
            out[f"steer_{name}"] = [Visual("box", (0.06, 0.05, 0.10), Transform((0.0, 0.0, self.wheel_radius + 0.05)), grey)]
            out[f"wheel_{name}"] = [Visual("cylinder", (self.wheel_radius, self.wheel_width), Transform(), tyre),
                                    Visual("box", (0.02, self.wheel_width + 0.004, self.wheel_radius), Transform(), grey)]
        return out


@dataclass
class DifferentialDrive(WheeledDrive):
    """Two driven wheels on one axle (left, right) through the base frame's origin, and casters. It turns by their
    difference and does not move sideways. ``radius``: a round body's (its footprint), else ``size`` is the outline."""
    track: float = 0.3
    casters: tuple = ((0.12, 0.0),)     # contact points (x, y) that carry load and roll freely any way
    radius: float | None = None

    def wheel_xy(self) -> np.ndarray:
        return np.array([[0.0, self.track / 2.0], [0.0, -self.track / 2.0]])

    def wheel_names(self):
        return [("left", "왼쪽"), ("right", "오른쪽")]

    def caster_xy(self) -> np.ndarray:
        return np.asarray(self.casters, dtype=np.float64).reshape(-1, 2)

    def footprint(self) -> np.ndarray:
        if self.radius is None:
            return super().footprint()
        a = np.arange(16) * (2.0 * math.pi / 16)
        return np.stack([np.cos(a), np.sin(a)], 1) * self.radius / math.cos(math.pi / 16)   # around the circle, not in it

    def limit(self, t: Twist) -> Twist:
        """No sideways motion; the spin up to its limit; then no wheel over the top speed, the curve's radius kept."""
        t = Twist(t.vx, 0.0, max(-self.max_spin, min(self.max_spin, t.wz)))
        fastest = abs(t.vx) + abs(t.wz) * self.track / 2.0
        k = min(1.0, self.max_speed / fastest) if fastest > 0.0 else 1.0
        return Twist(t.vx * k, 0.0, t.wz * k)

    def wheel_targets(self, twist: Twist, steer_now: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        half = self.track / 2.0
        return np.zeros(2), np.array([twist.vx - twist.wz * half, twist.vx + twist.wz * half])

    def visuals(self) -> dict:
        from .robots import Visual
        dark, tyre = ("plastic", (0.05, 0.05, 0.06), 0.4), ("rubber", (0.03, 0.03, 0.03), 0.6)
        wheels = {f"wheel_{n}": [Visual("cylinder", (self.wheel_radius, self.wheel_width), Transform(), tyre)] for n, _ in self.wheel_names()}
        if self.link_visuals is not None:
            return {**wheels, **{link: list(vs) for link, vs in self.link_visuals.items() if vs}}
        h = self.size[2] - self.clearance
        body = Visual("cylinder", (self.radius, h), Transform((0.0, 0.0, 0.0), (90.0, 0.0, 0.0)), dark) if self.radius else \
            Visual("box", (self.size[0], self.size[1], h), Transform(), dark)
        return {"chassis": [body], **wheels}
