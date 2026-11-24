"""Lights a robot carries: a lamp on a link, lighting the scene from wherever the robot stands.

    lamp = world.spawn(lights.AreaLight(size=(0.06, 0.02), color=(1.0, 0.9, 0.8), radiance=40.0),
                       attach_to=robot, link="base_link", mount=Transform((0.37, 0.15, 0.22)))

Drawn as a housing with a rectangle that emits along the light's x (its body frame, like a camera's). In the render
server it is a shape of its own, moved with its link each frame (an instance cannot emit: bodies.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .actor import Spec
from .robots import Visual
from .transform import Transform


@dataclass
class AreaLight(Spec):
    """A rectangular lamp. ``size``: width (along the light's y) and height (along its z), metres. ``color`` times
    ``radiance`` is what it emits (the scene's radiance units: its own lamps' ``radiance`` values are the scale).
    ``enabled``: False keeps the housing and emits nothing."""
    kind = "light"
    size: tuple = (0.06, 0.02)
    color: tuple = (1.0, 1.0, 1.0)
    radiance: float = 40.0
    enabled: bool = True
    depth: float = 0.02          # the housing behind the lamp's face
    model: str = "light"

    def link_frames(self, state) -> dict:
        return {"light": np.eye(4)}

    def emitted(self) -> tuple:
        return tuple(float(c) * float(self.radiance) for c in self.color) if self.enabled else (0.0, 0.0, 0.0)

    def visuals(self) -> dict:
        w, h = self.size
        housing = Visual("box", (self.depth, w + 0.008, h + 0.008), Transform((-self.depth / 2.0 - 0.001, 0.0, 0.0)),
                         ("plastic", (0.03, 0.03, 0.03), 0.4))
        out = [housing]
        if any(self.emitted()):   # the face: a rectangle whose normal (its z) is the light's x
            out.append(Visual("light", (h, w), Transform((0.0, 0.0, 0.0), (0.0, 90.0, 0.0)), ("emitter", self.emitted(), 0.0)))
        return {"light": out}
