"""The ground a robot drives on: its height, what blocks the way, and how much grip each surface gives.

    ground = Ground(mu=0.6)
    ground.add_region([(2.0, -1.0), (4.0, -1.0), (4.0, 1.0), (2.0, 1.0)], mu=0.15)     # a wet patch
    world = World(ground=ground)

A ``World`` asks it three things each tick: the friction coefficient under each wheel, the floor's height under each
wheel (the base follows it: height, roll, pitch), and whether the robot's footprint at its next pose is clear.

Heights come from a map (``set_height_map``: the highest surface below the robot's top, on a grid): flat at ``z``
without one. The render server will fill it from ray queries of the scene (docs/hardware_platform_plan.md, M4a); a
synthetic map does for tests and for scenes laid out by hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

def friction_of(material: str, default: float = 0.6) -> float:
    """The coefficient a material's name stands for (``"Gray speckled epoxy floor"`` -> 0.6, ``"apron_wet"`` -> 0.41):
    opticalnav_sim.groundmap.material_friction, by the name alone."""
    from ..groundmap import material_friction
    mu, why = material_friction(material)
    return default if why.startswith("default") and "x" not in why else mu


def _inside(polygon: np.ndarray, x: float, y: float) -> bool:
    px, py = polygon[:, 0], polygon[:, 1]
    qx, qy = np.roll(px, -1), np.roll(py, -1)
    cross = (py > y) != (qy > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        at = px + (y - py) * (qx - px) / (qy - py)
    return bool(np.count_nonzero(cross & (x < at)) % 2)


@dataclass
class Ground:
    mu: float = 0.6                  # friction coefficient where nothing else says
    z: float = 0.0                   # the floor's height where there is no height map
    step: float = 0.03               # a rise the wheels take; anything higher under the footprint blocks
    regions: list = field(default_factory=list)       # (polygon N x 2, mu): later ones win
    materials: dict = field(default_factory=dict)     # material name -> mu, over the keyword table
    heights: np.ndarray | None = None                 # rows along y, columns along x; NaN: not seen
    material_map: np.ndarray | None = None            # per cell, an index into ``material_names`` (-1: none): the scene's own
    material_names: list = field(default_factory=list)
    origin: tuple = (0.0, 0.0)       # world (x, y) of the map's cell [0, 0] centre
    cell: float = 0.05

    # --- friction -------------------------------------------------------------------------------------------------
    def add_region(self, polygon, mu: float) -> None:
        """A patch of other grip: a polygon in world (x, y), its friction coefficient."""
        poly = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        if len(poly) < 3 or not 0.0 <= float(mu) <= 2.0:
            raise ValueError("a region is a polygon of at least three points and a friction coefficient of 0..2")
        self.regions.append((poly, float(mu)))

    def set_material(self, material: str, mu: float) -> None:
        self.materials[str(material)] = float(mu)

    def set_material_map(self, material_map, names, table: dict | None = None) -> None:
        """The scene's surface materials on the height map's grid (opticalnav_sim.groundmap), and their coefficients.
        ``set_material`` afterwards overrides one of them."""
        self.material_map, self.material_names = np.asarray(material_map, dtype=np.int64), list(names)
        for name, mu in (table or {}).items():
            self.materials.setdefault(name, float(mu))

    def material_at(self, x: float, y: float) -> str | None:
        """The scene material under a point, when the map knows it."""
        if self.material_map is None:
            return None
        i, j = self._index(x, y)
        if not (0 <= i < self.material_map.shape[0] and 0 <= j < self.material_map.shape[1]):
            return None
        k = int(self.material_map[i, j])
        return self.material_names[k] if k >= 0 else None

    def friction(self, x: float, y: float, material: str | None = None) -> float:
        """Friction coefficient at a point: the last region that holds it, else its material's (given, or the scene's
        there), else ``mu``."""
        for poly, mu in reversed(self.regions):
            if _inside(poly, float(x), float(y)):
                return mu
        if material is None:
            material = self.material_at(x, y)
        if material is not None:
            return self.materials[material] if material in self.materials else friction_of(material, self.mu)
        return self.mu

    # --- heights --------------------------------------------------------------------------------------------------
    def set_height_map(self, heights, origin, cell: float) -> None:
        """``heights[i, j]``: the highest surface under the robot's top at (origin x + j cell, origin y + i cell)."""
        self.heights, self.origin, self.cell = np.asarray(heights, dtype=np.float64), (float(origin[0]), float(origin[1])), float(cell)

    def _index(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        j = np.rint((np.asarray(x, dtype=np.float64) - self.origin[0]) / self.cell).astype(np.int64)
        i = np.rint((np.asarray(y, dtype=np.float64) - self.origin[1]) / self.cell).astype(np.int64)
        return i, j

    def height(self, x: float, y: float, near: float | None = None) -> float:
        """The floor's height at a point. With a map: the cell's height, or ``near`` (the height the robot stands at)
        when the cell holds something the wheels do not climb, or was not seen."""
        if self.heights is None:
            return self.z
        i, j = self._index(x, y)
        if not (0 <= i < self.heights.shape[0] and 0 <= j < self.heights.shape[1]):
            return self.z if near is None else near
        h = float(self.heights[i, j])
        if np.isnan(h) or (near is not None and abs(h - near) > self.step):
            return self.z if near is None else near
        return h

    def blocked(self, corners, floor: float, slope=(0.0, 0.0), at=(0.0, 0.0)) -> bool:
        """Is anything more than a step above the floor under the footprint (4 x 2 world corners, in order)? The floor
        is the plane the robot stands on: height ``floor`` at the point ``at``, rising by ``slope`` (per metre along
        world x and y), so a ramp is floor and a kerb on it is not. Outside the map, and where it was not seen,
        counts as blocked: a robot does not drive off what is known."""
        if self.heights is None:
            return False
        c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        along, across = c[1] - c[0], c[3] - c[0]
        n, m = (max(2, int(np.ceil(np.linalg.norm(v) / (self.cell * 0.5))) + 1) for v in (along, across))
        s, t = np.meshgrid(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, m))
        pts = c[0] + s[..., None] * along + t[..., None] * across
        i, j = self._index(pts[..., 0], pts[..., 1])
        ok = (i >= 0) & (i < self.heights.shape[0]) & (j >= 0) & (j < self.heights.shape[1])
        if not ok.all():
            return True
        h = self.heights[i, j]
        plane = floor + slope[0] * (pts[..., 0] - at[0]) + slope[1] * (pts[..., 1] - at[1])
        # a cell holds the height at its centre: half a cell of the slope is not an obstacle
        slack = self.step + 0.5 * self.cell * (abs(slope[0]) + abs(slope[1]))
        return bool(np.any(np.isnan(h) | (h > plane + slack)))
