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

    def visuals(self) -> dict:
        """Link name -> [robots.Visual]: what is drawn of it. Empty: nothing."""
        return {}

    def mounts(self) -> dict:
        """Named places other things go on: name -> (link, Transform)."""
        return {}

    def enter(self, state, world: "World") -> None:
        """The actor joins ``world``, or the world's ground changed."""

    def settle(self, state, old: Transform, pose: Transform, ground) -> Transform:
        """The pose a root actor takes after a step from ``old`` to ``pose``, as the ground allows it (hw.drive)."""
        return pose


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
        spec.enter(actor.state, self)
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

    def set_ground(self, ground: Ground) -> None:
        """Another ground under everything that is here."""
        self.ground = ground
        for a in self.actors:
            a.spec.enter(a.state, self)

    def roots(self) -> list[Actor]:
        return [a for a in self.actors if a.parent is None]

    def of_kind(self, kind: str) -> list[Actor]:
        return [a for a in self.actors if a.spec.kind == kind]

    def tick(self, dt: float | None = None) -> float:
        """Advance by ``dt`` (default: the world's). Returns the new time."""
        dt = self.dt if dt is None else float(dt)
        for a in self.roots():
            pose = a.spec.step(a.state, a.pose, dt)
            a.pose = a.spec.settle(a.state, a.pose, pose, self.ground)
        for a in self.actors:  # attached actors with a state of their own (a pan-tilt head) advance in place
            if a.parent is not None and a.state is not None:
                a.spec.step(a.state, Transform(), dt)
        self.time += dt
        return self.time
