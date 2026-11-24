"""The scene graph of hardware: actors that hold a spec, attached to one another.

    world = World()
    robot = world.spawn(robots.RangerMini2(), at=pose2d(1.0, 2.0, yaw_deg=90.0))
    cam = world.spawn(cameras.LucidTDR054S(), attach_to=robot, mount=Transform((0.25, 0.0, 0.45), (0.0, 8.0, 0.0)))
    robot.apply_control(...); world.tick(0.05)
    cam.world_matrix()   # follows the robot

Nothing here renders: ``World`` is the kinematic state. ``sim.SimWorld`` (the same interface) sends it to the render server.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ground import Ground
from .transform import Transform


@dataclass
class Spec:
    """What a piece of hardware is. Subclasses add their own fields; ``kind`` names the family ("robot", "camera", ...)."""
    kind = "hardware"

    def make_state(self):
        """The mutable state an actor of this spec carries (a robot's joints and velocity); None for a rigid part."""
        return None

    def step(self, state, pose: Transform, dt: float) -> Transform:
        """Advance ``state`` by ``dt`` seconds; returns the actor's new pose (root actors only)."""
        return pose

    def link_frames(self, state) -> dict:
        """Link name -> 4x4 in the actor's frame. The first is the frame others attach to by default."""
        return {"base_link": np.eye(4)}


class Actor:
    def __init__(self, world: "World", spec: Spec, name: str, parent: "Actor | None" = None, link: str | None = None,
                 mount: Transform = Transform(), pose: Transform = Transform()):
        self.world, self.spec, self.name = world, spec, name
        self.parent, self.link, self.mount = parent, link, mount
        self.pose = pose  # world pose of a root actor (an attached actor's place is its mount)
        self.state = spec.make_state()
        self.children: list[Actor] = []

    def __repr__(self) -> str:
        where = f"on {self.parent.name}" + (f".{self.link}" if self.link else "") if self.parent else "root"
        return f"<Actor {self.name} ({type(self.spec).__name__}, {where})>"

    def world_matrix(self) -> np.ndarray:
        """This actor's frame in the world (4x4)."""
        if self.parent is None:
            return self.pose.matrix()
        return self.parent.link_matrix(self.link) @ self.mount.matrix()

    def link_matrices(self) -> dict:
        """Link name -> world 4x4 of every link of this actor (a rigid part has one)."""
        base = self.world_matrix()
        return {name: base @ m for name, m in self.spec.link_frames(self.state).items()}

    def link_matrix(self, link: str | None = None) -> np.ndarray:
        frames = self.spec.link_frames(self.state)
        if link is None:
            link = next(iter(frames))
        if link not in frames:
            raise KeyError(f"{self.name} has no link {link!r} (links: {', '.join(frames)})")
        return self.world_matrix() @ frames[link]

    def set_pose(self, pose: Transform) -> None:
        if self.parent is not None:
            raise ValueError(f"{self.name} is attached to {self.parent.name}: move that, or change this actor's mount")
        self.pose = pose

    def descendants(self) -> list["Actor"]:
        out = []
        for c in self.children:
            out += [c] + c.descendants()
        return out


@dataclass
class World:
    """Actors and time. ``tick`` advances every root actor's spec (kinematics); attached actors follow by construction."""
    dt: float = 0.05
    time: float = 0.0
    actors: list = field(default_factory=list)
    ground: Ground = field(default_factory=Ground)   # heights, obstacles and friction under the robots

    def spawn(self, spec: Spec, at: Transform | None = None, attach_to: Actor | None = None, mount: Transform | None = None,
              link: str | None = None, name: str | None = None) -> Actor:
        if attach_to is not None and at is not None:
            raise ValueError("give `at` (a world pose) or `attach_to` (+ `mount`), not both")
        if attach_to is None and (mount is not None or link is not None):
            raise ValueError("`mount` and `link` place an actor on another: give `attach_to`")
        if attach_to is not None and attach_to.world is not self:
            raise ValueError(f"{attach_to.name} belongs to another world")
        taken = {a.name for a in self.actors}
        if name is None:
            stem, i = type(spec).__name__.lower(), 0
            while f"{stem}_{i}" in taken:
                i += 1
            name = f"{stem}_{i}"
        elif name in taken:
            raise ValueError(f"an actor named {name!r} exists")
        actor = Actor(self, spec, name, attach_to, link, mount or Transform(), at or Transform())
        if hasattr(actor.state, "ground"):
            actor.state.ground = self.ground
        if attach_to is not None:
            attach_to.link_matrix(link)  # an unknown link fails here, not at the first render
            attach_to.children.append(actor)
        self.actors.append(actor)
        return actor

    def destroy(self, actor: Actor) -> None:
        """Remove an actor and everything attached to it."""
        for a in [actor] + actor.descendants():
            self.actors.remove(a)
        if actor.parent is not None:
            actor.parent.children.remove(actor)

    def _on_ground(self, actor: Actor, pose: Transform, _partial: bool = False) -> Transform:
        """A wheeled base's pose after a step, as the ground allows it: stopped where its footprint would cover
        something higher than a step (it keeps its place, its speed is dropped), else set on the floor under its
        wheels: their heights give the base's height and its roll and pitch."""
        import math

        import numpy as np

        spec, state, ground = actor.spec, actor.state, self.ground
        if ground.heights is None:
            state.collided = False
            return pose
        yaw = math.radians(pose.rpy_deg[2])
        c, s = math.cos(yaw), math.sin(yaw)
        wheels = spec.wheel_xy()
        # each wheel's contact height: the floor under it, unless that is more than a step from where the wheel stood
        # (a kerb, a wall, a drop: the wheel does not take it, and the footprint test below stops the robot)
        last = getattr(state, "contact", None)
        z = np.array([ground.height(pose.xyz[0] + c * px - s * py, pose.xyz[1] + s * px + c * py,
                                    near=None if last is None else float(last[i])) for i, (px, py) in enumerate(wheels)])
        # the plane through the four contact points, in the base's own axes: z = a x + b y + z0
        a, b, z0 = (float(v) for v in np.linalg.lstsq(np.c_[wheels, np.ones(4)], z, rcond=None)[0])
        corners = (np.c_[spec.footprint(), np.zeros(4), np.ones(4)] @ pose.matrix().T)[:, :2]
        if ground.blocked(corners, z0, (a * c - b * s, a * s + b * c), pose.xyz[:2]):
            # Blocked as a whole: the part of the step that is free is still taken (the turn without the travel, or
            # the travel without the turn), so a robot against a wall can turn away or slide along it; if neither
            # is free it stays, and its speed is dropped.
            if not _partial:
                old = actor.pose
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
                    done = self._on_ground(actor, part, _partial=True)
                    if done is not None:
                        state.collided = True   # it touched, and gave way
                        t = state.twist
                        if kind != "slide":
                            state.twist = type(t)(0.0, 0.0, t.wz) if kind == "turn" else type(t)(t.vx, t.vy, 0.0)
                        return done
            if _partial:
                return None
            state.collided = True
            state.speed, state.setpoint = np.zeros(4), np.zeros(4)
            state.twist = type(state.twist)()
            return actor.pose
        state.collided, state.contact = False, z
        return Transform((pose.xyz[0], pose.xyz[1], z0), (math.degrees(math.atan(b)), -math.degrees(math.atan(a)), pose.rpy_deg[2]))

    def roots(self) -> list[Actor]:
        return [a for a in self.actors if a.parent is None]

    def of_kind(self, kind: str) -> list[Actor]:
        return [a for a in self.actors if a.spec.kind == kind]

    def tick(self, dt: float | None = None) -> float:
        """Advance by ``dt`` (default: the world's). Returns the new time."""
        dt = self.dt if dt is None else float(dt)
        for a in self.roots():
            pose = a.spec.step(a.state, a.pose, dt)
            a.pose = self._on_ground(a, pose) if hasattr(a.spec, "footprint") else pose
        for a in self.actors:  # attached actors with a state of their own (a pan-tilt head) advance in place
            if a.parent is not None and a.state is not None:
                a.spec.step(a.state, Transform(), dt)
        self.time += dt
        return self.time
