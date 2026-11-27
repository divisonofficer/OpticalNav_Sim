"""opticalnav_sim/hw/physics.py on a tiny synthetic scene: a crate on a floor (and a wall), a robot pushing it. The convex
pieces of a fixed part, settling, a push that slides the crate and stops, nothing moving without contact, the robot
stalling against a jam instead of throwing what it pushes, and the pose deltas a renderer moves meshes by."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("scipy")

from opticalnav_sim import physics_objects as po  # noqa: E402
from opticalnav_sim.hw import physics  # noqa: E402
from test_physics_tag import Scene, box_mesh  # noqa: E402

FOOT = np.array([[-0.369, -0.25], [0.369, -0.25], [0.369, 0.25], [-0.369, 0.25]])   # a Ranger Mini
HALF = 0.369


def crate_scene(root, wall_x: float = 1.0):
    """A floor (top at 0), a 0.4 m pine crate at sim x 0..0.4, y -0.4..0, and a wall from sim x ``wall_x`` (Mitsuba
    axes in the files: Y up, sim y = -Z)."""
    s = Scene(root)
    s.box("floor", [-4, -0.1, -4], [4, 0, 4], label="room floor", material="Gray epoxy floor")
    s.box("crate", [0, 0, 0], [0.4, 0.4, 0.4], label="crate", material="Pine wood")
    s.box("east_wall", [wall_x, 0, -3], [wall_x + 0.2, 2.0, 3], label="east wall", material="Wall paint")
    scene = s.write()
    (scene / po.PHYSICS).write_text(json.dumps(po.analyse(scene)))
    return scene


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    return crate_scene(tmp_path_factory.mktemp("push"))


def drive(world, x0: float, speed: float, seconds: float, y: float = -0.2, still: float = 0.0, dt: float = 0.05):
    """The robot driving along +x from x0 (its centre) for ``seconds``, then standing ``still``. The contacts seen."""
    seen = set()
    x = x0
    for _ in range(int(round(seconds / dt))):
        x += speed * dt
        world.step((x, y, 0.0, 0.0), dt)
        seen |= set(world.contacts())
    for _ in range(int(round(still / dt))):
        world.step((x, y, 0.0, 0.0), dt)
        seen |= set(world.contacts())
    return seen, x


def test_clip_and_convex_pieces():
    # a box: one piece, a box geom
    v, f = box_mesh([0, 0, 0], [1.0, 0.2, 0.5])
    tri = v[f].astype(float)
    pieces = physics.convex_pieces(tri, tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0))
    assert len(pieces) == 1
    hv, h = pieces[0]
    assert physics._as_box(hv, float(h.volume), 0.97) is not None
    # clipping keeps what is inside, the cut faces' area exactly
    half = physics.clip_box(tri, [0, 0, 0], [0.5, 0.2, 0.5])
    assert half.reshape(-1, 3).max(0)[0] == pytest.approx(0.5)
    assert po._area(half) == pytest.approx(0.8, rel=1e-9)   # half its long sides (0.7 m^2) and its near end (0.1), no far end
    # an L (a post on a board, one mesh): its hull would fill the corner; the pieces leave it open
    a, fa = box_mesh([0, 0, 0], [1.0, 0.05, 0.5])
    b, fb = box_mesh([0, 0, 0], [0.05, 1.0, 0.5])
    tri = np.concatenate([a[fa], b[fb]]).astype(float)
    pieces = physics.convex_pieces(tri, tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0))
    assert len(pieces) >= 2
    corner = np.array([0.6, 0.6, 0.25])
    assert not any(np.all(h.equations[:, :3] @ corner + h.equations[:, 3] <= 1e-9) for _, h in pieces)


def test_box_hull_inertia():
    v, f = box_mesh([-0.5, -0.25, -0.1], [0.5, 0.25, 0.1])
    inertia = physics.hull_inertia(v.astype(float), f, 6.0)
    expect = 6.0 / 12.0 * np.array([0.5 ** 2 + 0.2 ** 2, 1.0 ** 2 + 0.2 ** 2, 1.0 ** 2 + 0.5 ** 2])
    assert np.allclose(np.diag(inertia), expect, rtol=1e-6) and np.allclose(inertia - np.diag(np.diag(inertia)), 0.0, atol=1e-9)


def test_scene_build_settles_and_nothing_moves_without_contact(scene):
    world = physics.PushWorld.from_scene(scene, FOOT, 0.4)
    assert world.ids == ["crate"] and world.info["free"] == 1
    assert world.info["floor_z"] == pytest.approx(0.0) and world.info["static"] >= 1   # the plane and the wall
    # the robot 10 cm short of the crate for 5 s: nothing touches it, nothing moves
    first = world.step((-HALF - 0.1, -0.2, 0.0, 0.0), 0.05)
    assert set(first) <= {"crate"}                         # the settling, if it moved the crate (it does not here)
    for _ in range(100):
        assert world.step((-HALF - 0.1, -0.2, 0.0, 0.0), 0.05) == {}
    assert world.contacts() == []
    delta = world.poses()["crate"]
    assert np.abs(delta[:3, 3]).max() < 1e-3 and np.allclose(delta[:3, :3], np.eye(3), atol=1e-3)


def test_robot_pushes_the_crate_it_slides_and_stops(scene):
    world = physics.PushWorld.from_scene(scene, FOOT, 0.4)
    start = world.world_poses()["crate"].copy()
    # from 0.2 m short of the crate, 0.5 m past its face at 0.5 m/s, then still for 1.5 s
    seen, x = drive(world, -HALF - 0.2, 0.5, 1.4, still=1.5)
    assert seen == {"crate"}
    now = world.world_poses()["crate"]
    moved = now[:3, 3] - start[:3, 3]
    # pushed 0.5 m, then sliding on: v^2 / (2 mu g) = 2.1 cm at mu 0.6
    assert 0.48 < moved[0] < 0.56 and abs(moved[1]) < 0.02 and abs(moved[2]) < 0.005
    tilt = math.degrees(math.acos(min(1.0, now[2, 2])))
    assert tilt < 2.0
    # at rest: the same pose a second later, and it no longer touches the robot (it slid off its face)
    again = world.step((x, -0.2, 0.0, 0.0), 1.0)
    assert again == {} and world.contacts() == []
    # the delta moves the crate's original vertices to where it is
    delta = world.poses()["crate"]
    o = world.objects[0]
    original = o.vertices + o.initial[:3, 3]
    current = (now[:3, :3] @ o.vertices.T).T + now[:3, 3]
    assert np.allclose((delta[:3, :3] @ original.T).T + delta[:3, 3], current, atol=1e-9)


def test_jammed_against_a_wall_the_robot_stalls(tmp_path):
    scene = crate_scene(tmp_path, wall_x=0.6)            # 20 cm of room behind the crate
    world = physics.PushWorld.from_scene(scene, FOOT, 0.4)
    start = world.world_poses()["crate"].copy()
    z_max = -np.inf
    x = -HALF - 0.2
    for _ in range(26):                                   # 0.65 m at 0.5 m/s: 0.45 m into a crate with 0.2 m of room
        x += 0.025
        world.step((x, -0.2, 0.0, 0.0), 0.05)
        z_max = max(z_max, world.world_poses()["crate"][2, 3])
    now = world.world_poses()["crate"]
    assert 0.17 < now[0, 3] - start[0, 3] < 0.21            # at the wall, not through it
    assert z_max - start[2, 3] < 0.01 and now[2, 2] > 0.999   # not thrown up, not tipped
    assert 0.2 < world.lag < world.settings["lag_m"] and world.stalled > 0.5   # the collider stopped, the drive went on
    for _ in range(4):                                    # lag_m behind: the collider jumps to the robot, through the crate
        x += 0.025
        world.step((x, -0.2, 0.0, 0.0), 0.05)
    assert world.lag == 0.0 and world._ghost and world.contacts() == []
    assert np.allclose(world.world_poses()["crate"], now, atol=2e-3)


def test_a_jump_passes_through_until_clear(scene):
    world = physics.PushWorld.from_scene(scene, FOOT, 0.4)
    # put down on the crate: no contact, nothing thrown
    world.step((0.2, -0.2, 0.0, 0.0), 0.05)
    assert world.contacts() == [] and world._ghost
    world.step((0.2, -0.2, 0.0, 0.0), 0.05)
    assert np.abs(world.poses()["crate"][:3, 3]).max() < 1e-3
    # away from it again: collides once clear
    world.step((-2.0, -0.2, 0.0, 0.0), 0.05)
    world.step((-2.0, -0.2, 0.0, 0.0), 0.05)
    assert not world._ghost
