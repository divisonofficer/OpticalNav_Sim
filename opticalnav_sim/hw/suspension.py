"""A wheeled base's body on its suspension: how it bounces, pitches and rolls on rough ground.

    spec = robots.ranger_mini(suspension_hz=3.0, suspension_zeta=0.2)     # off (0 Hz): the body sits on the contact plane

Without it (hw.drive.Drive.settle) the body stands on the plane through the floor under its wheels, at once: a bump
lifts it the moment a wheel is over it. With it:

* each wheel touches the ground where its rim does: its centre is as high as the ground lets a circle of its radius be,
  over the stretch ahead of and behind its axle (a step is climbed along an arc, a pothole narrower than the wheel is
  bridged);
* the body is a rigid mass (the robot's and what it carries) on one spring and damper at each wheel: heave, pitch and
  roll, from the springs' and dampers' forces (each at least zero: a wheel that would have to pull leaves the ground),
  integrated in ``substep`` steps. Static sag is left out: at rest the body stands where the plane does.

The numbers are a robot configuration's (``suspension_hz``: the body's bounce frequency on its springs; ``zeta``: the
damping ratio); the Ranger Mini 2.0's are estimates (its manual gives none). Yaw and the travel stay the drive's.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

G = 9.81


@dataclass
class BodyState:
    z: float                  # the body's height at the base origin (m)
    a: float                  # its slope along the base's x (dz/dx: the nose up) ...
    b: float                  # ... and along y (dz/dy)
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))   # d/dt of (z, a, b)
    ground: np.ndarray | None = None                              # each wheel's ground height last step
    accel: float = 0.0        # the body's vertical acceleration over the last step (m/s^2, gravity left out)
    lifted: int = 0           # wheels off the ground at the end of the last step


def rim_heights(ground, points_world: np.ndarray, heading: float, radius: float, near=None, samples: int = 9) -> np.ndarray:
    """Each wheel's ground height as its rim meets the ground (N): the highest h(s) + sqrt(r^2 - s^2) - r over s along
    the heading within the radius (a circle resting on the profile, its lowest point's height)."""
    s = np.linspace(-radius, radius, samples)
    lift = np.sqrt(np.maximum(radius * radius - s * s, 0.0)) - radius
    c, sn = math.cos(heading), math.sin(heading)
    out = np.empty(len(points_world))
    for i, (x, y) in enumerate(points_world):
        nr = None if near is None else float(near[i])
        h = np.array([ground.height(x + c * d, y + sn * d, near=nr) for d in s])
        out[i] = float(np.max(h + lift))
    return out


def advance(state: BodyState, ground_z: np.ndarray, points: np.ndarray, share: np.ndarray, mass: float,
            inertia: tuple, hz: float, zeta: float, dt: float, substep: float = 0.001) -> BodyState:
    """The body after ``dt`` on its springs, the wheels' ground at ``ground_z`` (N, the end of the step; its rate from the
    last step's). ``points``: the wheels in the base frame (N x 2); ``share``: each one's share of the weight;
    ``inertia``: (about x, about y) in kg m^2."""
    n = len(points)
    w = 2.0 * math.pi * hz
    k = mass * share * w * w                       # each wheel's spring: the body bounces at ``hz``
    c = 2.0 * zeta * mass * share * w              # ... and its damper
    g_old = state.ground if state.ground is not None else ground_z
    rate = (ground_z - g_old) / max(dt, 1e-9)
    x = np.array([state.z, state.a, state.b], dtype=np.float64)
    v = state.v.astype(np.float64).copy()
    m_diag = np.array([mass, inertia[1], inertia[0]])
    lever = np.c_[np.ones(n), points[:, 0], points[:, 1]]        # (1, px, py): force -> (heave, pitch, roll)
    steps = max(1, int(math.ceil(dt / substep)))
    h = dt / steps
    z0 = float(x[0])
    v_start = float(v[0])
    forces = np.zeros(n)
    for i in range(steps):
        f = (i + 1) / steps
        gz = g_old + (ground_z - g_old) * f
        at = lever @ x
        vel = lever @ v
        forces = mass * G * share + k * (gz - at) + c * (rate - vel)
        forces = np.maximum(forces, 0.0)
        acc = (lever.T @ forces - np.array([mass * G, 0.0, 0.0])) / m_diag
        v = v + acc * h
        x = x + v * h
    out = BodyState(float(x[0]), float(x[1]), float(x[2]), v, ground_z.copy())
    out.accel = (float(v[0]) - v_start) / max(dt, 1e-9)
    out.lifted = int(np.count_nonzero(forces <= 0.0))
    del z0
    return out
