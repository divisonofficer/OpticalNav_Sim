"""The hardware layer without a GPU. Run: python -m pytest tests -q"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opticalnav_sim import bodies, frames, hw  # noqa: E402
from opticalnav_sim.hw import Transform, cameras, optics, rigs, robots  # noqa: E402
from opticalnav_sim.hw.calibration import RigCalibration, load_opencv_yaml  # noqa: E402
from opticalnav_sim.hw.transform import yaw_from_heading  # noqa: E402


def test_transform_composes_and_inverts():
    a, b = Transform((1.0, 2.0, 0.5), (10.0, -20.0, 75.0)), Transform((0.2, 0.0, 0.4), (0.0, 8.0, -30.0))
    assert np.allclose((a @ b).matrix(), a.matrix() @ b.matrix(), atol=1e-12)
    assert np.allclose((a @ a.inverse()).matrix(), np.eye(4), atol=1e-12)
    up = Transform((0.0, 0.0, 0.0), (0.0, -90.0, 30.0))  # straight up: roll and yaw are one turn
    assert np.allclose(Transform.from_matrix(up.matrix()).matrix(), up.matrix(), atol=1e-9)
    # ROS axes: a positive pitch looks down, a positive yaw turns left
    assert Transform(rpy_deg=(0.0, 30.0, 0.0)).matrix()[2, 0] < 0.0
    assert Transform(rpy_deg=(0.0, 0.0, 30.0)).matrix()[1, 0] > 0.0


def test_camera_matrix_matches_the_simulators_free_camera():
    """A level camera at an R2R position and MatterSim heading is frames.camera_at_position's camera."""
    for heading_deg in (0.0, 37.0, 190.0, 331.0):
        h, pos = math.radians(heading_deg), (3.0, -2.0, 1.4)
        ours = hw.camera_to_world(Transform(pos, (0.0, 0.0, math.degrees(yaw_from_heading(h)))).matrix())
        ref = frames.camera_at_position(pos, h, -frames.nominal_pitch(None), None)
        assert np.allclose(ours, ref, atol=1e-12)
    # a link mesh keeps its ROS axes: its x (forward) at yaw 0 is the world's x, which is Mitsuba's X; its z is up (Y)
    m = hw.to_mitsuba(Transform((1.0, 2.0, 3.0)).matrix())
    assert np.allclose(m[:3, 3], (1.0, 3.0, -2.0)) and np.allclose(m[:3, 0], (1, 0, 0)) and np.allclose(m[:3, 2], (0, 1, 0))


def test_attached_actors_follow_and_links_carry_them():
    world = hw.World(dt=0.02)
    robot = world.spawn(robots.RangerMini2(), at=hw.pose2d(1.0, 2.0, 0.0, 90.0))
    cam = world.spawn(cameras.LucidTRI032S(), attach_to=robot, link="top_plate", mount=Transform((0.2, 0.0, 0.1)))
    depth = world.spawn(cameras.LucidTRI032S(), attach_to=cam, mount=Transform((0.0, 0.05, 0.0)), name="second")
    # yaw 90: the robot's forward is the world's +y; 0.2 m ahead on a deck 0.338 m up
    assert np.allclose(cam.world_matrix()[:3, 3], (1.0, 2.2, 0.438), atol=1e-9)
    assert np.allclose(depth.world_matrix()[:3, 3], (0.95, 2.2, 0.438), atol=1e-9)  # 0.05 m to the camera's left
    before = cam.world_matrix()[:3, 3] - robot.world_matrix()[:3, 3]
    robot.spec.apply(robot.state, hw.RangerControl("ackermann", 0.5, 15.0))
    for _ in range(100):
        world.tick()
    assert np.linalg.norm(robot.pose.xyz[:2] - np.array([1.0, 2.0])) > 0.5
    rel = np.linalg.inv(robot.world_matrix()) @ cam.world_matrix()
    assert np.allclose(rel[:3, 3], (0.2, 0.0, 0.438), atol=1e-9) and np.linalg.norm(before) > 0
    with pytest.raises(KeyError):
        world.spawn(cameras.LucidTRI032S(), attach_to=robot, link="no_such_link")
    with pytest.raises(ValueError):
        cam.set_pose(Transform())
    world.destroy(cam)
    assert [a.name for a in world.actors] == [robot.name]  # what was on the camera went with it


def test_ranger_modes_follow_their_kinematics():
    spec = robots.RangerMini2()

    def run(control, seconds=4.0, dt=0.005):
        world = hw.World(dt=dt)
        r = world.spawn(spec, at=hw.pose2d(0.0, 0.0))
        spec.apply(r.state, control)
        for _ in range(int(seconds / dt)):
            world.tick()
        return r

    # Ackermann: both axles steer, the turning centre is abreast of the middle: radius (L / 2) / tan(steer)
    r = run(hw.RangerControl("ackermann", 0.5, 20.0))
    radius = (spec.wheelbase / 2.0) / math.tan(math.radians(20.0))
    assert math.isclose(r.state.twist.wz, 0.5 / radius, rel_tol=1e-6) and abs(r.state.twist.vy) < 1e-9
    steer = np.degrees(r.state.steer)
    assert steer[0] > steer[1] > 0 and np.allclose(steer[2:], -steer[:2])  # inner wheel turned more; rear mirrors front
    # oblique: every wheel at the angle, no turn
    r = run(hw.RangerControl("oblique", 0.4, 30.0))
    assert np.allclose(np.degrees(r.state.steer), 30.0) and abs(r.state.twist.wz) < 1e-9
    assert math.isclose(math.degrees(math.atan2(r.pose.xyz[1], r.pose.xyz[0])), 30.0, abs_tol=0.5)
    assert abs(r.pose.rpy_deg[2]) < 1e-6
    # spin: on the spot
    r = run(hw.RangerControl("spin", angular=1.0))
    assert math.isclose(r.state.twist.wz, 1.0, rel_tol=1e-6) and np.hypot(*r.pose.xyz[:2]) < 1e-6
    # lateral: sideways, wheels at 90 degrees
    r = run(hw.RangerControl("lateral", 0.3))
    assert np.allclose(np.abs(np.degrees(r.state.steer)), 90.0) and r.pose.xyz[1] > 1.0
    assert abs(r.pose.xyz[0]) < 2e-3  # the wheels are turned sideways before the base moves
    # the driver's cmd_vel, over the speed limit: scaled, direction kept
    r = run(hw.Twist(10.0, 0.0, 0.0), seconds=3.0)
    assert math.isclose(r.state.twist.vx, spec.max_speed, rel_tol=1e-6)
    # wheels roll as far as the robot went; odometry is the pose from the start
    assert np.allclose(r.state.roll * spec.wheel_radius, r.pose.xyz[0], rtol=1e-6)
    assert np.allclose(r.state.odometry[:2], r.pose.xyz[:2], atol=1e-9)


def test_wheels_wait_for_their_steering():
    """A new direction is taken at the steering rate: the base does not jump sideways."""
    spec = robots.RangerMini2(max_steer_rate_deg=90.0)
    world = hw.World(dt=0.01)
    r = world.spawn(spec, at=hw.pose2d(0.0, 0.0))
    spec.apply(r.state, hw.RangerControl("lateral", 0.3))
    world.tick()
    assert np.max(np.abs(np.degrees(r.state.steer))) <= 0.9 + 1e-9 and abs(r.state.twist.vy) < 1e-3
    for _ in range(60):  # 0.6 s at 90 degrees a second: 54 degrees, still turning, the base at rest
        world.tick()
    assert np.hypot(*r.pose.xyz[:2]) < 1e-9
    for _ in range(300):
        world.tick()
    assert math.isclose(r.state.twist.vy, 0.3, rel_tol=1e-6)


def test_cameras_and_lens_reproduce_the_datasheets():
    # the lens model on its 2/3 inch format gives the datasheet's angle of view (74.7 x 58.1 degrees)
    two_thirds = cameras.SensorSpec("2/3 inch", 8800, 6600, 1.0, 8)
    k = cameras.CameraSpec(sensor=two_thirds, lens=cameras.FujinonHF6XA5M()).full_intrinsics()
    v = k.rays(np.array([[k.cx, -0.5], [k.cx, k.height - 0.5]]))[:, 1]
    assert abs(k.hfov_deg() - 74.7) < 0.1 and abs(math.degrees(math.atan(v[1]) - math.atan(v[0])) - 58.1) < 0.1
    assert abs(cameras.tv_distortion_k1(-2.85, 6.0, (8.8, 6.6)) - k.dist[0]) < 0.005  # its TV distortion says the same
    for make, size in ((cameras.LucidTDR054S, (2880, 1860)), (cameras.LucidTRI032S, (2048, 1536)),
                       (cameras.FlirBFS51S5PC, (2448, 2048))):
        cam = make()
        full, small = cam.full_intrinsics(), cam.intrinsics()
        assert (full.width, full.height) == size and small.width % 2 == 0 and small.height % 2 == 0
        assert abs(small.hfov_deg() - full.hfov_deg()) < 0.05  # the same camera, fewer pixels
        # a pixel's ray is the same at both sizes: the centre of the top left 4x4 block is the small image's first pixel
        assert np.allclose(full.rays(np.array([[1.5, 1.5]])), small.rays(np.array([[0.0, 0.0]])), atol=2e-3)
    assert cameras.FlirBFS51S5PC().sensor.cfa == "polar_RGGB" and len(cameras.FlirBFS51S5PC().sensor.polarizer_deg) == 4
    with pytest.raises(ValueError):  # a 1/3 inch lens does not cover the sensor
        cameras.CameraSpec(sensor=cameras.IMX490, lens=cameras.LensSpec("small", 6.0, 2.0, 6.0))
    with pytest.raises(ValueError):
        cameras.LucidTRI032S(render_scale=0.0)


def test_distortion_matches_opencv():
    cv2 = pytest.importorskip("cv2")
    k = optics.Intrinsics(1920, 1200, 1500.0, 1490.0, 950.0, 610.0, (-0.21, 0.07, 0.001, -0.0007, -0.01))
    pts = np.random.default_rng(0).uniform([-0.55, -0.35, 1.0], [0.55, 0.35, 3.0], (500, 3))
    ref, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k.matrix(), np.array(k.dist))
    ref = ref.reshape(-1, 2)
    assert np.abs(k.project(pts) - ref).max() < 1e-9
    rays = k.rays(ref)  # the inverse: back to the points' directions
    assert np.abs(rays - pts[:, :2] / pts[:, 2:3]).max() < 1e-9
    assert np.abs(rays - cv2.undistortPoints(ref.reshape(-1, 1, 2), k.matrix(), np.array(k.dist)).reshape(-1, 2)).max() < 1e-4


def test_render_plan_holds_every_ray_and_resamples_a_pinhole_into_the_lens():
    """An ideal pinhole image of a checker target, resampled by the plan, is the distorted camera's image of it."""
    k = cameras.LucidTDR054S().intrinsics(0.125)
    plan = optics.render_plan(k)
    assert plan.width % 2 == 0 and plan.height % 2 == 0 and plan.width > k.width  # barrel distortion: a wider pinhole
    assert plan.map_x.min() >= 0.0 and plan.map_x.max() <= plan.width - 1.0
    assert plan.map_y.min() >= 0.0 and plan.map_y.max() <= plan.height - 1.0
    f = plan.width / (2.0 * math.tan(math.radians(plan.hfov_deg) / 2.0))  # the pinhole's focal length in its pixels
    assert math.isclose(f, max(k.fx, k.fy), rel_tol=1e-9)

    def target(x, y):  # a smooth pattern on the plane z = 1, as a function of the normalized coordinates
        return np.sin(9.0 * x) * np.cos(7.0 * y)

    u, v = np.meshgrid(np.arange(plan.width), np.arange(plan.height))
    pinhole = target((u - plan.width / 2.0 + 0.5) / f, (v - plan.height / 2.0 + 0.5) / f).astype(np.float32)
    got = optics.remap(pinhole, plan.map_x, plan.map_y)
    uu, vv = np.meshgrid(np.arange(k.width, dtype=np.float64), np.arange(k.height, dtype=np.float64))
    rays = k.rays(np.stack([uu.ravel(), vv.ravel()], 1))
    want = target(rays[:, 0], rays[:, 1]).reshape(k.height, k.width)
    assert np.abs(got - want).max() < 2e-3  # bilinear resampling of a smooth image
    rgb = optics.remap(np.stack([pinhole] * 3, -1), plan.map_x, plan.map_y)
    assert rgb.shape == (k.height, k.width, 3) and np.allclose(rgb[..., 1], got)


def test_body_models_become_shapegroups_and_instances(tmp_path):
    spec = robots.RangerMini2()
    links = {link: [{"shape": v.shape, "size": list(v.size), "origin": v.origin.matrix().tolist(),
                     "material": [v.material[0], list(v.material[1]), v.material[2]]} for v in vs]
             for link, vs in spec.visuals().items()}
    model = bodies.build_model("ranger", {"links": links}, tmp_path)
    assert set(model["links"]) == set(spec.visuals()) and set(model["links"]) <= set(spec.link_frames(None))
    again = bodies.build_model("ranger", {"links": links}, tmp_path)
    assert again["key"] == model["key"]
    links["chassis"][0]["size"][0] += 0.01
    assert bodies.build_model("ranger", {"links": links}, tmp_path)["key"] != model["key"]  # a changed model is another
    text = bodies.scene_xml({"ranger": model}, [("robot_0", "ranger"), ("robot 1", "ranger")])
    assert text.count('type="shapegroup"') == len(model["links"])  # two robots share the meshes
    assert text.count('type="instance"') == 2 * len(model["links"])
    assert f'id="{bodies.instance_id("robot 1", "wheel_fl")}"' in text and "body_robot_1__wheel_fl" in text
    import xml.etree.ElementTree as ET
    root = ET.fromstring(f"<scene>{text}</scene>")
    files = [s.get("value") for s in root.iter("string") if s.get("name") == "filename"]
    assert files and all(Path(f).is_file() for f in files)
    # a wheel: closed, about the link's y axis, with its radius and width
    v, f = bodies.cylinder(0.09, 0.07)
    assert np.isclose(np.abs(v[:, 1]).max(), 0.035) and np.isclose(np.hypot(v[:, 0], v[:, 2]).max(), 0.09)
    edges = {}
    for a, b, c in f:
        for e in ((a, b), (b, c), (c, a)):
            edges[tuple(sorted(map(tuple, np.round(v[list(e)], 6))))] = edges.get(tuple(sorted(map(tuple, np.round(v[list(e)], 6)))), 0) + 1
    assert set(edges.values()) == {2}  # watertight (by position: the rims' vertices are doubled)
    vb, fb = bodies.box((1.0, 2.0, 3.0))
    n = np.cross(vb[fb[:, 1]] - vb[fb[:, 0]], vb[fb[:, 2]] - vb[fb[:, 0]])
    assert np.all(np.einsum("ij,ij->i", n, vb[fb].mean(1)) > 0)  # faces wound outwards
    with pytest.raises(ValueError):
        bodies.bsdf_xml(("wood", (0.5, 0.5, 0.5), 0.4))
    assert "roughconductor" in bodies.bsdf_xml(("metal", (0.8, 0.8, 0.8), 0.3)) and "pplastic" in bodies.bsdf_xml(("rubber", (0, 0, 0), 0.6))


RIG_YAML = """%YAML:1.0
---
t_ref_transform_in_ref: !!opencv-matrix
   rows: 4
   cols: 4
   dt: d
   data: [ 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
       1. ]
k_ref: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 910.5, 0., 501.9, 0., 910.3, 408.5, 0., 0., 1. ]
d_ref: !!opencv-matrix
   rows: 1
   cols: 12
   dt: d
   data: [ -0.115, -0.0118, 0.0088, -0.0036, 3.203, 0.0814, -0.2285,
       3.604, 0.0034, 0.0053, -0.0118, -0.0007 ]
t_left_transform_in_ref: !!opencv-matrix
   rows: 4
   cols: 4
   dt: d
   data: [ 1., 0., 0., -0.10, 0., 1., 0., 0.09, 0., 0., 1., 0.07, 0., 0., 0., 1. ]
k_left: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 1032.6, 0., 696.9, 0., 1032.1, 463.5, 0., 0., 1. ]
d_left: !!opencv-matrix
   rows: 1
   cols: 5
   dt: d
   data: [ -0.09, 0.10, -0.002, -0.009, -0.45 ]
"""


def test_rational_and_thin_prism_distortion_matches_opencv_and_inverts():
    """A 12-coefficient calibration (the rig's): projection as OpenCV's, and every pixel's ray found back."""
    cv2 = pytest.importorskip("cv2")
    d = (-0.115, -0.0118, 0.0088, -0.0036, 3.203, 0.0814, -0.2285, 3.604, 0.0034, 0.0053, -0.0118, -0.0007)
    k = optics.Intrinsics(1024, 768, 910.5, 910.3, 501.9, 408.5, d)
    pts = np.random.default_rng(1).uniform([-0.6, -0.45, 1.0], [0.6, 0.45, 2.0], (400, 3))
    ref, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k.matrix(), np.array(d))
    assert np.abs(k.project(pts) - ref.reshape(-1, 2)).max() < 1e-9
    u, v = np.meshgrid(np.arange(0.0, 1024.0, 7.0), np.arange(0.0, 768.0, 7.0))
    px = np.stack([u.ravel(), v.ravel()], 1)
    rays = k.rays(px)
    assert np.abs(k.project(np.c_[rays, np.ones(len(rays))]) - px).max() < 1e-7
    with pytest.raises(ValueError):  # a tilted sensor is another model
        optics.distort(np.zeros((1, 2)), d + (0.01, 0.0))


def test_rig_calibration_places_the_cameras(tmp_path):
    path = tmp_path / "rig.yaml"
    path.write_text(RIG_YAML)
    assert load_opencv_yaml(path)["d_ref"].shape == (1, 12)
    cal = RigCalibration.load(path)
    assert cal.reference == "ref" and sorted(cal.cameras) == ["left", "ref"]
    assert np.allclose(cal.relative("ref", "left")[:3, 3], (0.10, -0.09, -0.07))
    k = cal.cameras["left"].intrinsics(1440, 928)
    assert (k.width, k.fx, k.cx, len(k.dist)) == (1440, 1032.6, 696.9, 5)
    # optical frame: x right, y down, z forward. 10 cm to the left, 9 cm lower, 7 cm ahead of the reference camera
    mounts = rigs.camera_mounts(cal, ["ref", "left"], Transform((0.03, 0.0, 0.14)))
    assert np.allclose(mounts["ref"].xyz, (0.03, 0.0, 0.14)) and np.allclose(mounts["left"].xyz, (0.10, 0.10, 0.05))
    assert np.allclose(mounts["left"].rpy_deg, 0.0, atol=1e-9)

    world = hw.World()
    robot = world.spawn(robots.RangerMini2(), at=hw.pose2d(2.0, 1.0, 0.0, 90.0))
    spec = {"ref": lambda c, s: cameras.LucidTRI032S(calibration=c.intrinsics(1024, 768), render_scale=s),
            "left": lambda c, s: cameras.LucidTDR054S(calibration=c.intrinsics(1440, 928), render_scale=s)}
    rig = rigs.spawn_camera_rig(world, robot, path, cameras=("left", "ref"), models=spec)
    assert set(rig) == {"mast", "left", "ref"} and rig["left"].parent is rig["mast"] and rig["mast"].parent is robot
    # deck 0.338 m + mast 0.80 m + mount; the robot faces +y, so its left is -x
    assert np.allclose(rig["ref"].world_matrix()[:3, 3], (2.0, 1.03, 0.338 + 0.80 + 0.14), atol=1e-9)
    assert np.allclose(rig["left"].world_matrix()[:3, 3], (1.9, 1.10, 0.338 + 0.80 + 0.05), atol=1e-9)
    small = rig["left"].spec.intrinsics()   # a quarter of the sensor's width, the recorded image's shape
    assert (small.width, small.height) == (720, 464) and math.isclose(small.fx, 1032.6 / 2.0)
    assert rig["ref"].spec.image_size() == (512, 384) and rig["left"].spec.image_size(0.5) == (1440, 928)
    with pytest.raises(KeyError):
        rigs.spawn_camera_rig(world, robot, path, cameras=("left", "ref"))  # no model known for "ref": say which
    assert len(world.actors) == 4  # and nothing was spawned by the failed call
    mast = rigs.Mast(height=0.8)
    boxes = mast.visuals()["mast"]
    assert len(boxes) == 4 + 3 * 4 + 1 and max(v.origin.xyz[2] + v.size[2] / 2.0 for v in boxes) == pytest.approx(0.8)
