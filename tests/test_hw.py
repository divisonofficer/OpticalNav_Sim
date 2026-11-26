"""The hardware layer without a GPU. Run: python -m pytest tests -q"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opticalnav_sim import bodies, frames, groundmap, hw  # noqa: E402
from opticalnav_sim.hw import Transform, cameras, isp, optics, rigs, robots, sensor  # noqa: E402
from opticalnav_sim.hw.sim import Frame  # noqa: E402
from opticalnav_sim.hw.transform import optical_pose  # noqa: E402
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

    # Ackermann, front and rear: the command is the inner wheel's angle (manual 3.2); the turning centre is abreast of
    # the middle, (L / 2) / tan(angle) beyond the inner wheels
    r = run(hw.RangerControl("ackermann", 0.5, 20.0))
    radius = (spec.wheelbase / 2.0) / math.tan(math.radians(20.0)) + spec.track / 2.0
    assert math.isclose(r.state.twist.wz, 0.5 / radius, rel_tol=1e-6) and abs(r.state.twist.vy) < 1e-9
    steer = np.degrees(r.state.steer)
    assert math.isclose(steer[0], 20.0, abs_tol=1e-6) and steer[0] > steer[1] > 0 and np.allclose(steer[2:], -steer[:2])
    # the chassis's own limits: 40 degrees of inner wheel, half speed past 20 degrees; at full lock the body's outer
    # corner sweeps the manual's minimum turning radius (810 mm)
    full = hw.RangerControl("ackermann", 1.5, 75.0).twist(spec.wheelbase, spec.track, spec)
    assert math.isclose(full.vx, 0.75) and math.isclose(full.vx / full.wz, (spec.wheelbase / 2) / math.tan(math.radians(40.0)) + spec.track / 2)
    centre = full.vx / full.wz
    assert abs(math.hypot(centre + spec.size[1] / 2.0, spec.size[0] / 2.0) - 0.810) < 0.01
    assert hw.RangerControl("ackermann", 1.5, 15.0).twist(spec.wheelbase, spec.track, spec).vx == 1.5
    assert hw.RangerControl("ackermann", 0.5, -20.0).twist(spec.wheelbase, spec.track).wz < 0 < hw.RangerControl("ackermann", -0.5, -20.0).twist(spec.wheelbase, spec.track).wz
    assert hw.RangerControl("spin", angular=9.0).twist(spec.wheelbase, spec.track, spec).wz == spec.max_spin
    assert spec.mass == 64.5 + spec.payload and spec.size == (0.738, 0.500, 0.338)
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
    bars = mast.visuals()["mast"]
    posts = [v for v in bars if v.origin.rpy_deg == (0.0, 0.0, 0.0)]
    assert all(v.shape == "profile" for v in bars) and len(posts) == 4 and all(v.size == (0.03, 0.8) for v in posts)
    assert len(bars) == 4 + 2 * 5 + 2 * 3 + 1  # posts, five rungs front and back, three rails a side, the camera bar
    # a T-slot section: inside the square, a groove in the middle of each face, material at the corners and the core
    v, f = bodies.profile(0.03, 0.5)
    assert np.allclose(np.abs(v[:, :2]).max(0), 0.015) and np.isclose(np.abs(v[:, 2]).max(), 0.25)
    on_face = v[np.isclose(v[:, 0], 0.015)]
    assert np.abs(on_face[:, 1]).min() >= 0.004 - 1e-9  # nothing on the face inside the 8 mm slot
    model = bodies.build_model("mast", {"links": {"mast": [{"shape": "profile", "size": [0.03, 0.8], "material": list(rigs.ALUMINIUM)}]}}, tmp_path)
    assert 'value="Al"' in model["links"]["mast"] and "blendbsdf" in model["links"]["mast"]
    assert 'name="face_normals" value="true"' in model["links"]["mast"]  # flat faces: not the smooth normals of a bare PLY
    wheel = bodies.build_model("w", {"links": {"w": [{"shape": "cylinder", "size": [0.1, 0.05]}]}}, tmp_path)
    assert "face_normals" not in wheel["links"]["w"]                      # a wheel's side stays round


def _frame(spec, s0, s1=None, s2=None, pose=None):
    zero = np.zeros_like(s0)
    return Frame(0.0, spec.intrinsics(), np.eye(4) if pose is None else pose, None, s0.astype(np.float32),
                 (zero if s1 is None else s1).astype(np.float32), (zero if s2 is None else s2).astype(np.float32), zero)


def test_demosaic_returns_a_smooth_image_and_keeps_flat_colours():
    h, w = 96, 128
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([0.5 + 0.4 * np.sin(x / 23.0) * np.cos(y / 31.0), 0.5 + 0.4 * np.cos(x / 37.0 + 1.0),
                    0.4 + 0.3 * np.sin((x + y) / 29.0)], -1)
    for cfa in ("RGGB", "BGGR", "GRBG", "GBRG"):
        colour, _ = sensor.mosaic(cfa, h, w)
        for method in ("malvar", "bilinear"):
            got = isp.demosaic(img[y, x, colour], cfa, method)
            assert np.abs(got - img)[4:-4, 4:-4].max() < 0.01, (cfa, method)
            flat = np.ones((h, w, 3)) * (0.2, 0.5, 0.8)
            assert np.allclose(isp.demosaic(flat[y, x, colour], cfa, method), flat, atol=1e-12), (cfa, method)
    colour, angle = sensor.mosaic("polar_RGGB", 8, 8, (90.0, 45.0, 135.0, 0.0))
    assert colour[:4, :4].tolist() == [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 2, 2], [1, 1, 2, 2]]  # colours in 2x2 blocks
    assert angle[:2, :2].tolist() == [[90.0, 45.0], [135.0, 0.0]] and np.array_equal(angle[:2, :2], angle[2:4, 4:6])


def test_sensor_follows_the_emva_linear_model():
    """Mean and variance of a flat field: variance = K^2 (electrons + read noise^2); saturation at the full well."""
    spec = cameras.LucidTRI032S()
    s, k = spec.sensor, spec.intrinsics()
    gain = 4095.0 / s.full_well_e   # counts per electron
    dark = sensor.expose(_frame(spec, np.zeros((k.height, k.width, 3))), spec, exposure_s=0.01, vignette=False, seed=1)
    assert dark.data.mean() / gain < 1.0 and dark.data.max() / gain < 6 * s.read_noise_e  # no light: read noise, cut at 0
    for radiance, electrons in ((0.1, 1000.0), (0.4, 4000.0)):
        raw = sensor.expose(_frame(spec, np.full((k.height, k.width, 3), radiance)), spec, exposure_s=0.01, vignette=False, seed=1)
        green = raw.data[0::2, 1::2].astype(np.float64)
        assert abs(green.mean() / gain - electrons) < 0.01 * electrons + 0.3
        assert abs(green.var() - gain ** 2 * (electrons + s.read_noise_e ** 2)) < 0.05 * gain ** 2 * (electrons + 10.0)
    raw = sensor.expose(_frame(spec, np.full((k.height, k.width, 3), 5.0)), spec, exposure_s=0.01, vignette=False)
    # the well is full: every pixel at the top, less the readout's noise
    assert raw.data.dtype == np.uint16 and raw.data.max() == 4095 and raw.data.min() > 4095 - 6 * s.read_noise_e * gain
    # twice the exposure, or twice the gain: twice the counts; four pixels averaged: half the noise
    f = _frame(spec, np.full((k.height, k.width, 3), 0.1))
    a = sensor.expose(f, spec, 0.005, vignette=False, noise=False).data.mean()
    assert np.isclose(sensor.expose(f, spec, 0.010, vignette=False, noise=False).data.mean(), 2.0 * a, rtol=1e-3)
    assert np.isclose(sensor.expose(f, spec, 0.005, gain_db=6.0206, vignette=False, noise=False).data.mean(), 2.0 * a, rtol=1e-3)
    one = sensor.expose(f, spec, 0.01, vignette=False, seed=5).data.std()
    four = sensor.expose(f, spec, 0.01, vignette=False, seed=5, pixels_per_sample=4.0).data.std()
    assert 1.8 < one / four < 2.2
    # the lens dims the corners (cos^4), auto exposure puts the bright end near the top without clipping it away
    v = sensor.vignetting(k)
    assert v.max() <= 1.0 and 0.3 < v[0, 0] < 0.5 and np.isclose(v[k.height // 2, k.width // 2], 1.0, atol=1e-3)
    assert np.allclose(sensor.vignetting(k, 2.0), np.sqrt(v))
    flat = sensor.expose(_frame(spec, np.full((k.height, k.width, 3), 0.3)), spec, 0.01, noise=False)
    corrected = isp.linear_rgb(flat, isp.Isp(shading=4.0))
    assert np.ptp(corrected[8:-8, 8:-8]) < 0.01  # the ISP's shading correction undoes it
    grad = np.repeat(np.linspace(0.01, 1.0, k.width)[None, :, None], k.height, 0).repeat(3, 2)
    t = sensor.auto_exposure(_frame(spec, grad), spec)
    raw = sensor.expose(_frame(spec, grad), spec, t, noise=False, vignette=True)
    assert abs(raw.linear().mean() - 0.27) < 0.01  # the meter's target: the image's mean at 0.27 of the range
    assert np.array_equal(sensor.expose(_frame(spec, grad), spec, noise=False).data, raw.data)  # no exposure given: that one


def test_auto_exposure_follows_the_scene_over_frames():
    """On by default; a camera carried from a bright view to a dark one takes a few frames, time first, then gain."""
    spec = cameras.LucidTRI032S()
    k = spec.intrinsics()
    world = hw.World()
    cam = world.spawn(spec, at=Transform((0.0, 0.0, 1.0)))
    ae = cam.state.exposure
    assert ae.auto and ae.exposure_s is None and world.spawn(spec, at=Transform()).state.exposure is not ae
    bright, dark = (_frame(spec, np.full((k.height, k.width, 3), v)) for v in (0.5, 0.02))
    t0, g0 = ae.next(bright, spec)
    assert abs(ae.level(bright, spec, t0, g0) - ae.target) < 1e-3 and g0 == 0.0  # the first frame: already settled
    assert ae.next(bright, spec) == (t0, g0)                                      # and it stays there
    seen = [ae.next(dark, spec) for _ in range(12)]
    levels = [ae.level(dark, spec, t, g) for t, g in seen]
    assert seen[0] == (t0, g0) and levels[0] < 0.05 * ae.target        # the frame after the change is still at the old setting
    assert all(b >= a for a, b in zip(levels, levels[1:])) and levels[2] < 0.5 * ae.target  # it climbs, not jumps
    assert abs(levels[-1] - ae.target) < 0.02
    t_end, g_end = seen[-1]
    assert t_end == ae.exposure_limits[1] and 0.0 < g_end <= ae.gain_limits[1]  # the time ran out at its limit: gain made up the rest
    night = _frame(spec, np.full((k.height, k.width, 3), 1e-6))
    for _ in range(30):
        ae.next(night, spec)
    assert (ae.exposure_s, ae.gain_db) == (ae.exposure_limits[1], ae.gain_limits[1])  # both at their limits, no further
    # manual: what is set is what is used
    ae.auto, ae.exposure_s, ae.gain_db = False, 0.002, 3.0
    assert ae.next(bright, spec) == (0.002, 3.0) and ae.next(dark, spec) == (0.002, 3.0)
    # an HDR sensor meters its most sensitive readouts: the highlights are left to the others
    hdr = cameras.LucidTDR054S()
    kh = hdr.intrinsics()
    f = _frame(hdr, np.full((kh.height, kh.width, 3), 0.3))
    raw = sensor.expose(f, hdr, vignette=False, noise=False)
    assert 0.0005 < raw.linear().mean() < 0.05 and np.mean(raw.data >= raw.white) == 0.0


def test_hdr_sensor_is_linear_across_its_readouts():
    """IMX490: one linear 24-bit value from four readouts, with the published noise floor, range and best SNR."""
    spec = cameras.LucidTDR054S()
    s, k = spec.sensor, spec.intrinsics()
    top = s.readouts[-1][1] / s.readouts[-1][0]
    assert abs(20 * math.log10(top / s.readouts[0][2]) - 123.6) < 0.1 and abs(10 * math.log10(s.readouts[-1][1]) - 50.8) < 0.1
    snr = {}
    for electrons in (10.0, 2000.0, 25000.0, 800000.0):
        radiance = electrons / (sensor.responsivity(spec) * 0.01)
        raw = sensor.expose(_frame(spec, np.full((k.height, k.width, 3), radiance)), spec, exposure_s=0.01, vignette=False, seed=2)
        assert raw.data.dtype == np.uint32 and raw.white == 2 ** 24 - 1
        got = raw.linear()[0::2, 1::2] * top
        assert abs(got.mean() / electrons - 1.0) < 0.01
        snr[electrons] = 20 * math.log10(got.mean() / got.std())
    assert abs(snr[2000.0] - 33.0) < 0.5 and abs(snr[800000.0] - 50.7) < 0.5
    assert snr[25000.0] < snr[2000.0] + 4.0  # the small photodiode has taken over: fewer electrons, the known dip


def test_polarization_sensor_recovers_the_stokes_field_in_its_own_frame():
    spec = cameras.FlirBFS51S5PC(render_scale=0.25)
    k = spec.intrinsics()
    h, w = k.height, k.width
    y, x = np.mgrid[0:h, 0:w]
    s0 = np.stack([0.6 + 0.2 * np.sin(x / 60.0), 0.5 + 0.2 * np.cos(y / 50.0), 0.4 + 0.1 * np.sin((x + y) / 70.0)], -1)
    dolp, aolp = 0.3 + 0.2 * np.sin(x / 90.0)[..., None], np.radians(20.0 + 40.0 * np.cos(y / 80.0))[..., None]
    s1, s2 = s0 * dolp * np.cos(2 * aolp), s0 * dolp * np.sin(2 * aolp)
    d = (spec.sensor.extinction_ratio - 1.0) / (spec.sensor.extinction_ratio + 1.0)
    for roll in (0.0, 25.0):
        pose = optical_pose(Transform((0.0, 0.0, 1.5), (roll, 0.0, 0.0)).matrix())
        psi = sensor.sensor_frame_rotation(k, pose)
        assert abs(math.degrees(psi[h // 2, w // 2]) + roll) < 0.2  # on the axis: the camera's roll, nothing else
        raw = sensor.expose(_frame(spec, s0, s1, s2, pose), spec, exposure_s=0.004, noise=False, vignette=False)
        out = isp.polarization(raw)
        g1, g2 = sensor.to_sensor_frame(s1, s2, psi)
        want_dolp = (np.sqrt(g1 ** 2 + g2 ** 2) / s0)[::2, ::2] * d
        want_aolp = (np.degrees(0.5 * np.arctan2(g2, g1)) % 180.0)[::2, ::2]
        assert out["s0"].shape == (h // 2, w // 2, 3) and sorted(out["i"]) == [0.0, 45.0, 90.0, 135.0]
        assert np.abs(out["dolp"] - want_dolp)[4:-4, 4:-4].mean() < 0.005
        assert (np.abs((out["aolp_deg"] - want_aolp + 90.0) % 180.0 - 90.0))[4:-4, 4:-4].mean() < 0.6
    # unpolarized light: every angle sees half of it
    raw = sensor.expose(_frame(spec, s0), spec, exposure_s=0.004, noise=False, vignette=False)
    assert isp.polarization(raw)["dolp"][4:-4, 4:-4].max() < 0.02
    with pytest.raises(ValueError):
        isp.polarization(sensor.expose(_frame(cameras.LucidTRI032S(), np.ones((384, 512, 3))), cameras.LucidTRI032S(), 0.001))


def test_isp_makes_an_image_and_keeps_saturated_areas_neutral():
    spec = cameras.LucidTRI032S()
    k = spec.intrinsics()
    y, x = np.mgrid[0:k.height, 0:k.width]
    scene = np.stack([0.30 + 0.0 * x, 0.20 + 0.0 * x, 0.10 + 0.0 * x], -1) * (0.2 + 1.8 * (x / k.width))[..., None]
    scene[:, k.width * 3 // 4:] = 50.0  # a window: far over the well
    raw = sensor.expose(_frame(spec, scene), spec, exposure_s=0.01, vignette=False, noise=False)
    img = isp.process(raw, isp.Isp(white_balance=(0.6, 1.0, 0.8), exposure=1.0))
    assert img.dtype == np.uint8 and img.shape == (k.height, k.width, 3)
    window = img[8:-8, -40:-8].reshape(-1, 3).astype(int)
    assert np.ptp(window, axis=1).max() <= 1  # white balance did not tint the clipped window
    tinted = isp.process(raw, isp.Isp(white_balance=(0.6, 1.0, 0.8), exposure=1.0, clip_highlights=False))[8:-8, -40:-8]
    assert np.ptp(tinted.reshape(-1, 3).astype(int), axis=1).min() > 20  # without it: a green window
    grey = isp.process(raw, isp.Isp(white_balance="gray_world"))[8:-8, 8:k.width // 2].reshape(-1, 3).astype(int)
    assert np.ptp(grey, axis=1).mean() < 6  # gray world: the one-coloured wall comes out neutral
    hdr = cameras.LucidTDR054S()
    kh = hdr.intrinsics()
    big = np.ones((kh.height, kh.width, 3)) * np.logspace(-3, 2, kh.width)[None, :, None]
    out = isp.process(sensor.expose(_frame(hdr, big), hdr, vignette=False), isp.Isp(tone="reinhard"))
    cols = out[kh.height // 2, ::kh.width // 12, 1].astype(int)
    assert np.all(np.diff(cols) >= 0) and cols[-1] - cols[0] > 150  # five decades in one image, in order


G = 9.81


def _dynamic(mu=0.6, ground=None, **kw):
    world = hw.World(dt=0.02, ground=ground or hw.Ground(mu=mu))
    return world, world.spawn(robots.RangerMini2(dynamics=True, **kw), at=hw.pose2d(0.0, 0.0))


def test_wheel_dynamics_keep_the_kinematic_path_when_there_is_grip():
    wk = hw.World(dt=0.02)
    rk = wk.spawn(robots.RangerMini2(), at=hw.pose2d(0.0, 0.0))
    wd, rd = _dynamic(1.0)
    for r in (rk, rd):
        r.spec.apply(r.state, hw.RangerControl("ackermann", 0.8, 15.0))
    for _ in range(250):
        wk.tick(), wd.tick()
    assert math.hypot(rk.pose.xyz[0] - rd.pose.xyz[0], rk.pose.xyz[1] - rd.pose.xyz[1]) < 0.06  # 4 m driven
    assert abs(rk.pose.rpy_deg[2] - rd.pose.rpy_deg[2]) < 1.0 and rd.state.slip.max() < 0.01
    assert np.allclose(rd.state.odometry[:2], rd.pose.xyz[:2], atol=0.01)  # the wheels did not slip: odometry is right
    # standing still stays still
    w, r = _dynamic(0.6)
    for _ in range(100):
        w.tick()
    assert r.pose.xyz == (0.0, 0.0, 0.0) and r.state.twist.vx == 0.0


def test_braking_distance_is_set_by_the_friction_coefficient():
    """Wheels held from v0: the robot slides v0^2 / (2 mu g)."""
    for mu in (0.6, 0.2, 0.1):
        w, r = _dynamic(mu)
        r.spec.apply(r.state, hw.Twist(1.0, 0.0, 0.0))
        for _ in range(150):
            w.tick()
        v0, x0 = r.state.twist.vx, r.pose.xyz[0]
        assert abs(v0 - 1.0) < 1e-3
        r.spec.max_accel = 1000.0  # the drives stop the wheels at once
        r.spec.apply(r.state, hw.Twist())
        for _ in range(400):
            w.tick()
        assert abs(r.state.twist.vx) < 1e-3
        assert abs((r.pose.xyz[0] - x0) / (v0 * v0 / (2.0 * mu * G)) - 1.0) < 0.03, mu


def test_wheels_spin_on_ice_and_odometry_runs_ahead():
    w, r = _dynamic(0.05, max_accel=20.0)
    r.spec.apply(r.state, hw.Twist(1.0, 0.0, 0.0))
    for _ in range(25):
        w.tick()
    assert abs(r.state.twist.vx - 0.05 * G * 0.5) < 0.01          # the ground gives mu g, whatever the wheels do
    assert r.state.speed.mean() > 0.9 and r.state.slip.min() > 0.5   # they are at speed already, sliding
    assert r.state.odometry[0] > 5.0 * r.pose.xyz[0] > 0.0           # encoders count the spin


def test_cornering_is_held_to_the_friction_limit():
    """The centripetal acceleration of the path never exceeds mu g: past it the tyres slide and the commanded circle
    at the commanded speed is not what the robot does."""
    def circle(mu):
        w, r = _dynamic(mu)
        r.spec.apply(r.state, hw.RangerControl("ackermann", 1.0, 15.0))
        pts = []
        for i in range(700):
            w.tick()
            if i >= 500:
                pts.append(r.pose.xyz[:2])
        p = np.array(pts)
        cx, cy, c = np.linalg.lstsq(np.c_[2.0 * p, np.ones(len(p))], (p ** 2).sum(1), rcond=None)[0]
        return math.sqrt(c + cx * cx + cy * cy), math.hypot(r.state.twist.vx, r.state.twist.vy), float(r.state.slip.max())
    asked = (0.494 / 2.0) / math.tan(math.radians(15.0)) + 0.364 / 2.0   # 1.10 m: v^2 / R = 0.91 m/s^2
    radius, speed, slip = circle(0.6)
    assert abs(radius - asked) < 0.02 and abs(speed - 1.0) < 0.01 and slip < 0.02
    radius, speed, slip = circle(0.05)                                    # mu g = 0.49 m/s^2: not enough for it
    assert speed * speed / radius <= 0.05 * G * 1.05 and slip > 0.2
    assert abs(speed * speed / radius - 1.0 / asked) > 0.3               # not the commanded motion


def test_split_grip_turns_the_robot_towards_the_grip_when_braking():
    ground = hw.Ground(mu=0.8)
    ground.add_region([(-50.0, 0.0), (50.0, 0.0), (50.0, 50.0), (-50.0, 50.0)], 0.1)   # the left half: slippery
    w, r = _dynamic(ground=ground)
    r.spec.apply(r.state, hw.Twist(1.0, 0.0, 0.0))
    for _ in range(150):
        w.tick()
    assert r.state.mu.tolist() == [0.1, 0.8, 0.1, 0.8]   # fl, fr, rl, rr: each wheel its own ground
    r.spec.max_accel = 1000.0
    r.spec.apply(r.state, hw.Twist())
    yaw0 = r.pose.rpy_deg[2]
    for _ in range(100):
        w.tick()
    assert r.pose.rpy_deg[2] - yaw0 < -3.0   # clockwise: the right wheels hold, the left ones slide on


def test_ground_friction_from_regions_and_material_names():
    g = hw.Ground(mu=0.6)
    assert g.friction(0.0, 0.0) == 0.6
    g.add_region([(0, 0), (2, 0), (2, 2), (0, 2)], 0.2)
    g.add_region([(1, 1), (3, 1), (3, 3), (1, 3)], 0.9)   # a later region wins where they overlap
    assert g.friction(0.5, 0.5) == 0.2 and g.friction(1.5, 1.5) == 0.9 and g.friction(2.5, 0.5) == 0.6
    assert g.friction(9.0, 9.0, "Gray speckled epoxy floor") == 0.6 and g.friction(9.0, 9.0, "ph11_wet_stain") == 0.33
    g.set_material("ph11_wet_stain", 0.12)
    assert g.friction(9.0, 9.0, "ph11_wet_stain") == 0.12 and g.friction(0.5, 0.5, "ph11_wet_stain") == 0.2
    with pytest.raises(ValueError):
        g.add_region([(0, 0), (1, 1)], 0.5)


def test_base_follows_the_floor_and_stops_at_obstacles():
    """A height map: a flat floor, a ramp, a wall. The base's height, pitch and roll are its wheels' contact plane."""
    cell = 0.05
    x = np.arange(-2.0, 8.0 + 1e-9, cell)
    y = np.arange(-2.0, 2.0 + 1e-9, cell)
    heights = np.zeros((len(y), len(x)))
    heights[:, x >= 2.0] = (x[x >= 2.0] - 2.0) * math.tan(math.radians(8.0))   # an 8 degree ramp from x = 2
    heights[:, x >= 6.0] = 3.0                                                  # a wall at x = 6
    ground = hw.Ground(step=0.03)
    ground.set_height_map(heights, (x[0], y[0]), cell)
    world = hw.World(dt=0.02, ground=ground)
    spec = robots.RangerMini2()
    r = world.spawn(spec, at=hw.pose2d(0.0, 0.0))
    cam = world.spawn(cameras.LucidTRI032S(), attach_to=r, link="top_plate", mount=Transform((0.0, 0.0, 0.5)))
    spec.apply(r.state, hw.Twist(1.0, 0.0, 0.0))
    for _ in range(40):
        world.tick()
    assert abs(r.pose.xyz[2]) < 1e-9 and np.allclose(r.pose.rpy_deg[:2], 0.0, atol=1e-9) and not r.state.collided
    for _ in range(400):
        if r.pose.xyz[0] >= 4.0:
            break
        world.tick()
    assert r.pose.xyz[0] >= 4.0 and not r.state.collided   # a ramp is floor, not an obstacle
    # on the ramp: nose up 8 degrees (a negative pitch), the base at the ramp's height under its centre
    assert abs(r.pose.rpy_deg[1] + 8.0) < 0.6 and abs(r.pose.rpy_deg[0]) < 0.2
    assert abs(r.pose.xyz[2] - (r.pose.xyz[0] - 2.0) * math.tan(math.radians(8.0))) < 0.02
    assert cam.world_matrix()[0, 0] < 1.0 and cam.world_matrix()[2, 0] > 0.1   # what it carries tilts with it
    for _ in range(300):
        world.tick()
    front = r.pose.xyz[0] + spec.size[0] / 2.0 * math.cos(math.radians(8.0))
    assert r.state.collided and 5.8 < front <= 6.0 + cell and r.state.twist.vx == 0.0   # stopped with its nose at the wall
    spec.apply(r.state, hw.Twist(-0.5, 0.0, 0.0))                                       # and it can back away
    for _ in range(50):
        world.tick()
    assert not r.state.collided and r.pose.xyz[0] < 5.5
    # off the map is not driven onto
    edge = world.spawn(robots.RangerMini2(), at=hw.pose2d(0.0, 1.6, 0.0, 90.0), name="edge")
    edge.spec.apply(edge.state, hw.Twist(1.0, 0.0, 0.0))
    for _ in range(100):
        world.tick()
    assert edge.state.collided and edge.pose.xyz[1] < 2.0 - spec.size[0] / 2.0 + cell


def test_material_friction_reads_the_name_then_the_pbr_parameters():
    f = groundmap.material_friction
    assert f("asphalt_dark", 0.9, 0.0) == (0.8, "asphalt 0.80")
    assert f("apron_wet", 0.2, 0.0)[0] == pytest.approx(0.75 * 0.55, abs=1e-3)       # the author's "wet" halves it
    assert f("paint_yellow_wet")[0] < f("paint_yellow")[0]
    assert f("icy_patch", 0.1, 0.0)[0] == 0.10
    assert f("ph27_granite_floor", 0.15, 0.0)[0] < f("ph27_granite_floor", 0.6, 0.0)[0] == 0.55   # a polished finish
    assert f("some_plate", 0.5, 1.0) == (0.45, "metallic 0.45") and f("thing", 0.5, 0.0)[0] == 0.6
    assert f("CS02_r003_corrugated_kraft", 0.5, 0.0)[0] == 0.6                          # "corrugated" is not a rug
    assert "wet" in f("apron_wet", 0.2, 0.0)[1] and "smooth" in f("floor_terrazzo_int", 0.1, 0.0)[1]


def _write_scene(folder, shapes):
    """A scene folder as the converter writes it: base.xml (each shape with its opticalnav-obj comment) and PLY files."""
    (folder / "assets").mkdir(parents=True)
    xml = "<scene version=\"3.0.0\">\n"
    for name, material, alpha, verts, faces in shapes:
        bodies.write_ply(folder / "assets" / f"{name}.ply", np.asarray(verts, np.float32), np.asarray(faces, np.int32))
        xml += (f'<!--opticalnav-obj:{{"id":"{name}","material":"{material}"}}-->\n<shape type="ply" id="{name}">\n'
                f'<string name="filename" value="assets/{name}.ply" />\n<bsdf type="twosided"><bsdf type="pplastic">'
                f'<float name="alpha" value="{alpha}" /></bsdf></bsdf>\n</shape>\n')
    (folder / "base.xml").write_text(xml + "</scene>\n")


def _quad(x0, x1, y0, y1, z):
    """A horizontal rectangle in the world (x, y, z up) as Mitsuba vertices (X, Y up, Z = -y)."""
    return [[x0, z, -y0], [x1, z, -y0], [x1, z, -y1], [x0, z, -y1]], [[0, 1, 2], [0, 2, 3]]


def test_scene_files_of_every_pack_are_read_transforms_primitives_and_instances(tmp_path):
    (tmp_path / "assets").mkdir()
    v, f = bodies.box((0.4, 1.0, 0.4))
    bodies.write_ply(tmp_path / "assets" / "box.ply", np.asarray(v, np.float32), np.asarray(f, np.int32))
    bsdf = '<bsdf type="twosided"><bsdf type="pplastic"><float name="alpha" value="0.25" /></bsdf></bsdf>'
    mesh = '<string name="filename" value="assets/box.ply" />'
    (tmp_path / "base.xml").write_text(f"""<scene version="3.0.0">
      <shape type="rectangle" id="hall_tile"><transform name="to_world"><scale x="5" y="5" /><rotate x="1" angle="-90" />
        <translate x="0" y="0.1" z="0" /></transform>{bsdf}</shape>
      <shape type="ply" id="crate">{mesh}<transform name="to_world"><translate x="2" y="0.6" z="-1" /></transform>{bsdf}</shape>
      <shape type="cube" id="block"><transform name="to_world"><scale value="0.25" /><translate value="-2, 0.35, 0" /></transform>{bsdf}</shape>
      <shape type="shapegroup" id="posts"><shape type="ply" id="post">{mesh}{bsdf}</shape></shape>
      <shape type="instance" id="i"><ref id="posts" /><transform name="to_world">
        <matrix value="1 0 0 -3  0 1 0 0.6  0 0 1 -2  0 0 0 1" /></transform></shape>
      <shape type="rectangle" id="lamp"><transform name="to_world"><translate x="0" y="0.5" z="0" /></transform><emitter type="area" /></shape>
    </scene>""")
    ids = [s["id"] for s in groundmap.scene_surfaces(tmp_path)]
    assert ids == ["hall_tile", "crate", "block", "post", "lamp"]    # (the group itself is drawn nowhere)
    for again in (False, True):                                     # the second time only the meshes near are read
        gm = groundmap.build(tmp_path, floor_z=0.1, top=1.4, cell=0.05, bounds=(-4.0, -4.0, 4.0, 4.0))
        at = lambda x, y: float(gm.heights[int(round((y - gm.origin[1]) / 0.05)), int(round((x - gm.origin[0]) / 0.05))])
        assert at(0.0, 0.0) == pytest.approx(0.1, abs=1e-6)          # the lamp is not in the way
        assert at(2.0, 1.0) == pytest.approx(1.1, abs=1e-4) and at(-2.0, 0.0) == pytest.approx(0.6, abs=1e-4)
        assert at(-3.0, 2.0) == pytest.approx(1.1, abs=1e-4)
    assert gm.ground().material_at(0.0, 0.0) == "hall_tile"


def test_the_floor_under_a_camera_is_not_the_furniture_around_it(tmp_path):
    def floor(folder, eye, *shapes):
        _write_scene(folder, [("f", "hall_tile", 0.25, *_quad(-6, 6, -6, 6, 0.02)), *shapes])
        return groundmap.floor_under(groundmap.floor_probe(folder, eye), eye[0], eye[1])
    shelves = ("s", "wood", 0.25, *_quad(0.5, 6, -6, 6, 1.4))         # most of the 3 m around is a shelf's top
    assert floor(tmp_path / "a", (0.0, 0.0, 1.5), shelves, ("t", "wood", 0.25, *_quad(-6, -0.6, -6, 6, 1.4))) == pytest.approx(0.02)
    table = ("t", "wood", 0.25, *_quad(-0.8, 0.8, -0.5, 0.5, 0.75))   # the camera above a table
    assert floor(tmp_path / "b", (0.0, 0.0, 1.6), table) == pytest.approx(0.02)
    balcony = ("b", "wood", 0.25, *_quad(-6, 0.6, -6, 6, 3.0))        # a storey down beside it is another floor
    assert floor(tmp_path / "c", (0.0, 0.0, 4.5), balcony) == pytest.approx(3.0)
    assert groundmap.floor_probe(tmp_path / "c", (50.0, 50.0, 1.5)) is None


def test_ground_map_from_scene_files_gives_heights_obstacles_and_friction(tmp_path):
    box_v, box_f = bodies.box((0.4, 1.0, 0.4))                      # Mitsuba axes: 0.4 wide, 1.0 tall, 0.4 deep
    box_v = box_v + np.array([3.0, 0.5, 0.0])                       # a crate standing on the floor at x = 3, y = 0
    _write_scene(tmp_path, [("floor", "hall_tile", 0.25, *_quad(-1.0, 5.0, -2.0, 2.0, 0.0)),
                            ("stain", "hall_wet_stain", 0.0025, *_quad(0.5, 1.5, -2.0, 2.0, 0.002)),
                            ("crate", "crate_wood", 0.25, box_v, box_f)])
    surfaces = {s["id"]: s for s in groundmap.scene_surfaces(tmp_path)}
    assert surfaces["floor"]["material"] == "hall_tile" and surfaces["floor"]["roughness"] == pytest.approx(0.5)
    gm = groundmap.build(tmp_path, floor_z=0.0, top=1.4, cell=0.05)
    assert gm.friction["hall_tile"]["mu"] == 0.5 and gm.friction["hall_wet_stain"]["mu"] == pytest.approx(0.33)
    ground = gm.ground()
    assert ground.material_at(0.0, 0.0) == "hall_tile" and ground.material_at(1.0, 0.0) == "hall_wet_stain"
    assert ground.friction(0.0, 0.0) == 0.5 and ground.friction(1.0, 0.5) == pytest.approx(0.33)
    assert abs(ground.height(0.0, 0.0)) < 1e-6 and ground.height(3.0, 0.0) == pytest.approx(1.0, abs=1e-3)   # the crate's top
    gm.save(tmp_path / "ground.npz")
    again = groundmap.GroundMap.load(tmp_path / "ground.npz")
    assert np.array_equal(again.material, gm.material) and again.friction == gm.friction and again.origin == gm.origin
    # a robot on it: the floor's height, each wheel its own surface, stopped by the crate
    world = hw.World(dt=0.02, ground=ground)
    spec = robots.RangerMini2(dynamics=True)
    r = world.spawn(spec, at=hw.pose2d(0.0, 0.0))
    spec.apply(r.state, hw.Twist(0.8, 0.0, 0.0))
    seen = set()
    for _ in range(400):
        world.tick()
        seen.add(tuple(np.round(r.state.mu, 2)))
        if r.state.collided:
            break
    assert (0.33, 0.33, 0.5, 0.5) in seen          # the front wheels on the wet stain, the rear ones still on tile
    assert r.state.collided and 2.3 < r.pose.xyz[0] + spec.size[0] / 2.0 <= 2.8 + 0.05   # its nose at the crate (x = 2.8)
    ground.set_material("hall_wet_stain", 0.05)    # an override: the stain is ice now
    assert ground.friction(1.0, 0.0) == 0.05


def test_robot_routes_sensors_and_commands():
    """The viewer's robot without a render server: its routes, its sensors' switches, its command parsing."""
    from opticalnav_sim import robot_view
    assert robot_view.handles(["api", "robot", "state"]) and not robot_view.handles(["api", "state"]) and not robot_view.handles(["robot"])
    session = robot_view.RobotSession("http://127.0.0.1:1", calibration="")
    st = session.state()
    assert not st["running"] and not st["spawned"] and "robot" not in st
    # a robot's sensors are listed once it stands in a scene; the configuration names them before that
    assert st["sensors"] == [] and [s["name"] for s in st["config"]["sensors"]] == ["hdr", "12bit"]
    on = lambda: [n in session.rendering() for n in ("hdr", "12bit")]  # noqa: E731
    assert on() == [False, False]
    session.set_sensors({"hdr": True})
    assert on() == [True, False]
    with pytest.raises(ValueError):
        session.set_sensors({"lidar": True})
    # a sensor renders while a page shows it (the viewer says which) or a call switched it on; no other does
    session.viewers = lambda: {"12bit"}
    assert session.rendering() == {"hdr", "12bit"} and on() == [True, True]
    session.set_sensors({"hdr": False})
    assert session.rendering() == {"12bit"}
    session.viewers = lambda: set()
    assert session.rendering() == set() and on() == [False, False]
    session.viewers = None
    session.set_sensors({"hdr": True})
    assert session.bodies("scene") == [] and session.pose2d() is None      # nothing spawned: nothing to draw
    assert session.command({"mode": "ackermann", "linear": 0.5, "steering_deg": 20.0}) == {"ok": True}
    control, at = session.control
    assert isinstance(control, hw.RangerControl) and control.steering_deg == 20.0
    session.command({"mode": "twist", "vx": 0.2, "wz": 0.1})
    assert isinstance(session.control[0], hw.Twist) and session.control[0].wz == 0.1
    with pytest.raises(ValueError):
        session.command({"mode": "fly"})
    assert session.frame("hdr") is None
    # a sensor's view takes the editor's look (its lighting, its variant), but is always the path tracer's RGB
    session.scene = "scene"
    view, look = session._sensor_view({"scan": "scene", "variant": "base", "spp": 16},
                                      {"scan": "scene", "variant": "perturbed", "lighting": "all", "bands": "both", "renderer": "raster", "width": 9})
    assert view["variant"] == "perturbed" and view["lighting"] == "all" and view["bands"] == "rgb" and view["renderer"] == "path"
    assert "width" not in view and view["depth"] is False and "perturbed" in look
    # each camera its own mode, whatever the editor's: a plain camera unpolarized where the server can, a polarization
    # camera polarized; a server without the unpolarized mode renders both polarized
    session.modes = ["polar", "rgb"]
    for editor in ("polar", "rgb"):
        plain, _ = session._sensor_view({"scan": "scene", "mode": "polar"}, {"scan": "scene", "mode": editor})
        polar, _ = session._sensor_view({"scan": "scene", "mode": "rgb"}, {"scan": "scene", "mode": editor}, polarized=True)
        assert plain["mode"] == "rgb" and polar["mode"] == "polar"
    # the viewer's global polarization setting: off, every camera unpolarized; unified, every one polarized (one scene)
    session.polar_policy = "off"
    assert [session.camera_mode(p) for p in (False, True)] == ["rgb", "rgb"]
    session.polar_policy = "unified"
    assert [session.camera_mode(p) for p in (False, True)] == ["polar", "polar"]
    session.polar_policy = "split"
    session.modes = ["polar"]
    assert session._sensor_view({"scan": "scene"}, {"scan": "scene", "mode": "polar"})[0]["mode"] == "polar"
    other, _ = session._sensor_view({"scan": "scene", "variant": "base"}, {"scan": "elsewhere", "variant": "perturbed"})
    assert other["variant"] == "base"                                      # the editor shows another scene: not its look

    class Fake:
        sent = None

        def _send(self, code, body, ctype):
            self.sent = (code, ctype, len(body))

        def _json(self, code, payload):
            self.sent = (code, payload)
    h = Fake()
    robot_view.get(h, session, ["api", "robot", "frame", "hdr.jpg"], {})
    assert h.sent[0] == 404
    robot_view.get(h, session, ["api", "robot", "state"], {})
    assert h.sent[0] == 200 and h.sent[1]["sensors"] == [] and "hdr" in session.rendering()
    robot_view.post(h, session, ["api", "robot", "control"], {"mode": "spin", "angular": 1.0})
    assert h.sent == (200, {"ok": True})
    robot_view.post(h, session, ["api", "robot", "sensors"], {"hdr": False, "12bit": True})
    assert h.sent[0] == 200 and session.rendering() == {"12bit"}
    robot_view.post(h, session, ["api", "robot", "spawn"], {})             # no viewer behind this handler
    assert h.sent[0] == 404
    here = Path(robot_view.__file__).parent
    page = (here / "gui.html").read_text()                                 # one viewer: the robot is a tab and panes of it
    # the cameras (the editor's and the robot's) in one list; a robot added from a list of pictures; its parts as a tree
    assert 'id="cam-list"' in page and 'draggable="true"' in page and 'id="robot-add"' in page and 'id="robot-tree"' in page \
        and "/api/robot/preview.png" in page and "/api/robot/" in page
    assert 'id="subs"' in page and 'id="sub-add"' in page and 'id="pane-b"' not in page   # sub viewers: any number, opened and closed
    assert not (here / "robot.html").exists()


def test_blocked_robot_can_turn_or_slide_free():
    """Against a wall the part of a step that is free is still taken: the robot is not glued to what it touched."""
    cell = 0.05
    x = np.arange(-3.0, 3.0 + 1e-9, cell)
    heights = np.zeros((len(x), len(x)))
    heights[:, x >= 1.0] = 2.0                         # a wall along x = 1
    ground = hw.Ground()
    ground.set_height_map(heights, (x[0], x[0]), cell)
    world = hw.World(dt=0.02, ground=ground)
    spec = robots.RangerMini2()
    r = world.spawn(spec, at=hw.pose2d(0.0, 0.0, 0.0, 0.0))
    spec.apply(r.state, hw.Twist(0.6, 0.0, 0.0))
    for _ in range(200):
        world.tick()
    stuck = r.pose.xyz[0]
    assert r.state.collided and stuck + spec.size[0] / 2.0 <= 1.0 + cell
    spec.apply(r.state, hw.Twist(0.6, 0.4, 0.0))       # still pushing into the wall, and sideways: it slides along it
    for _ in range(100):
        world.tick()
    assert r.pose.xyz[1] > 0.3 and abs(r.pose.xyz[0] - stuck) < 0.06
    spec.apply(r.state, hw.Twist(-0.5, 0.0, 0.0))
    for _ in range(60):
        world.tick()
    assert not r.state.collided and r.pose.xyz[0] < stuck - 0.2


# --- robots from configuration files (hw/config.py), URDFs, lights ---------------------------------------------------
URDF = """<?xml version="1.0"?>
<robot name="cart">
  <material name="blue"><color rgba="0.1 0.2 0.8 1"/></material>
  <link name="base_link"><visual><origin xyz="0 0 0.05"/><geometry><box size="0.6 0.3 0.1"/></geometry><material name="blue"/></visual></link>
  {wheels}
</robot>
"""
WHEEL = """
  <link name="{n}_steer_link"><visual><geometry><box size="0.04 0.04 0.08"/></geometry></visual></link>
  <joint name="{n}_steer" type="revolute"><parent link="base_link"/><child link="{n}_steer_link"/>
    <origin xyz="{x} {y} 0"/><axis xyz="0 0 1"/><limit lower="-1.6" upper="1.6" effort="5" velocity="6"/></joint>
  <link name="{n}_wheel_link"><visual><geometry><cylinder radius="0.08" length="0.05"/></geometry>
    <material name="tyre"><color rgba="0.02 0.02 0.02 1"/></material></visual>
    <visual><origin xyz="0 0.03 0"/><geometry><mesh filename="hub.stl" scale="0.001 0.001 0.001"/></geometry></visual></link>
  <joint name="{n}_wheel" type="continuous"><parent link="{n}_steer_link"/><child link="{n}_wheel_link"/>
    <origin xyz="0 0 -0.06"/><axis xyz="0 1 0"/></joint>
"""


def _cart(tmp_path):
    """A robot folder as a user would write one: a URDF with an STL mesh, and its structure file."""
    import struct
    folder = tmp_path / "cart"
    folder.mkdir()
    tri = np.array([[0, 0, 0], [20, 0, 0], [0, 0, 20]], np.float32)            # one triangle, in millimetres
    (folder / "hub.stl").write_bytes(b"\0" * 80 + struct.pack("<I", 1) + struct.pack("<12fH", 0, 1, 0, *tri.ravel(), 0))
    places = {"fl": (0.25, 0.2), "fr": (0.25, -0.2), "rl": (-0.25, 0.2), "rr": (-0.25, -0.2)}
    (folder / "cart.urdf").write_text(URDF.format(wheels="".join(WHEEL.format(n=n, x=x, y=y) for n, (x, y) in places.items())))
    (folder / "robot.yaml").write_text("""
name: cart
label: A cart
base:
  type: four_wheel_steering
  urdf: cart.urdf
  joints: {steering: [fl_steer, fr_steer, rl_steer, rr_steer], wheels: [fl_wheel, fr_wheel, rl_wheel, rr_wheel]}
  geometry: {wheel_radius: 0.08}
  params: {kerb_mass: 20.0, max_speed: 1.0}
parts:
  - {name: lamp, type: light, link: base_link, mount: {xyz: [0.31, 0.0, 0.15]}, params: {size: [0.1, 0.03], color: [1.0, 0.5, 0.0], radiance: 10.0}}
  - {name: mast, type: mast, link: top_plate, params: {height: 0.5}}
  - {name: front, type: camera, model: lucid_tri032s, parent: mast, link: top, label: Front camera,
     mount: {xyz: [0.1, 0.0, 0.05], rpy_deg: [0, 10, 0]}, params: {render_scale: 0.5}}
  - {name: polar, type: camera, model: flir_bfs_51s5pc, parent: mast, link: top, mount: {xyz: [0.1, 0.1, 0.05]}}
""")
    return folder


def test_urdf_links_joints_and_meshes(tmp_path):
    from opticalnav_sim.hw import urdf
    folder = _cart(tmp_path)
    model = urdf.load(folder / "cart.urdf", cache=tmp_path / "cache")
    assert model.root == "base_link" and len(model.joints) == 8 and model.joint("fl_steer").limit == (-1.6, 1.6)
    frames = model.frames()
    assert np.allclose(frames["fl_wheel_link"][:3, 3], [0.25, 0.2, -0.06])
    turned = model.frames({"fl_steer": math.pi / 2})["fl_wheel_link"]
    assert np.allclose(turned[:3, 0], [0.0, 1.0, 0.0]) and np.allclose(turned[:3, 3], [0.25, 0.2, -0.06])   # about z, in place
    box, = model.visuals["base_link"]
    assert box.shape == "box" and box.size == (0.6, 0.3, 0.1) and box.material[1] == (0.1, 0.2, 0.8)
    tyre, hub = model.visuals["fl_wheel_link"]
    assert tyre.shape == "cylinder" and tyre.material[1] == (0.02, 0.02, 0.02)
    assert hub.shape == "mesh" and hub.mesh.endswith(".ply") and hub.scale == (0.001, 0.001, 0.001)   # the STL, converted once
    v, f = groundmap.read_ply(hub.mesh)[:2]
    assert len(f) == 1 and np.allclose(np.asarray(v)[:, :3].max(0), [20.0, 0.0, 20.0])
    with pytest.raises(KeyError):
        model.joint("no_such_joint")


def test_robot_from_a_structure_file(tmp_path, monkeypatch):
    """A folder with a URDF and a robot.yaml is a robot: listed, spawned with its parts on their links, driven."""
    from opticalnav_sim.hw import config
    folder = _cart(tmp_path)
    monkeypatch.setenv("OPTICALNAV_ROBOTS", str(tmp_path))
    names = [r["name"] for r in config.available()]
    assert "cart" in names and "ranger_mini" in names and "turtlebot4" in names              # the one beside it, and the one shipped
    cfg = config.load("cart")
    assert cfg.label == "A cart" and [s["name"] for s in cfg.sensors()] == ["front", "polar"]
    assert cfg.sensors()[0] == {"name": "front", "label": "Front camera", "type": "camera", "model": "LUCID TRI032S"}
    world = hw.World()
    robot = config.spawn(world, cfg, at=hw.pose2d(1.0, 2.0, 0.0, 90.0))
    spec = robot.base.spec
    assert abs(spec.wheelbase - 0.5) < 1e-9 and abs(spec.track - 0.4) < 1e-9 and spec.wheel_radius == 0.08   # measured on the URDF
    assert spec.kerb_mass == 20.0 and spec.max_speed == 1.0
    drawn = spec.visuals()
    assert set(drawn) == {"base_link"} | {f"{k}_{w}" for k in ("steer", "wheel") for w in ("fl", "fr", "rl", "rr")}
    # the body's box: 0.05 above the URDF's root, whose origin is 0.06 above the wheels' centres, themselves 0.08 up
    assert np.allclose(drawn["base_link"][0].origin.xyz, [0.0, 0.0, 0.08 + 0.06 + 0.05])
    assert np.allclose(drawn["wheel_fl"][0].origin.xyz, [0.0, 0.0, 0.0])                                    # a wheel about its own centre
    assert set(robot.parts) == {"lamp", "mast", "front", "polar"} and set(robot.sensors) == {"front", "polar"} and set(robot.lights) == {"lamp"}
    assert robot.sensors["front"].parent is robot.parts["mast"] and robot.sensors["front"].spec.render_scale == 0.5
    assert robot.sensors["polar"].spec.sensor.cfa.startswith("polar_")
    # facing +y: the lamp 0.31 m ahead of the base, the camera on top of the 0.5 m mast on the deck
    assert np.allclose(robot.lights["lamp"].world_matrix()[:3, 3], [1.0, 2.31, 0.15])
    deck = spec.size[2]
    assert np.allclose(robot.sensors["front"].world_matrix()[:3, 3], [1.0, 2.1, deck + 0.5 + 0.05])
    spec.apply(robot.base.state, hw.RangerControl("ackermann", 0.5, 0.0))
    for _ in range(40):
        world.tick()
    assert robot.base.pose.xyz[1] > 2.5 and robot.sensors["front"].world_matrix()[1, 3] > 2.6                # what is on it went with it
    # a wrong entry names itself and leaves the world as it was
    for bad, word in ((("parts", 0, "params"), {"brightness": 3}), (("parts", 2, "model"), "no_such_camera")):
        import copy
        data = copy.deepcopy(cfg.data)
        data[bad[0]][bad[1]][bad[2]] = word
        other = hw.World()
        with pytest.raises(ValueError):
            config.spawn(other, config.RobotConfig("cart", data, cfg.path))
        assert other.actors == []
    with pytest.raises(ValueError):
        config.load({"name": "x", "base": {"type": "hovercraft"}})
    with pytest.raises(ValueError):
        config.load({"name": "x", "base": {"type": "urdf"}, "parts": [{"name": "a", "type": "mast", "parent": "b"}]})
    (tmp_path / "yes.yaml").write_text("name: trap\nbase: {type: four_wheel_steering}\nparts:\n  - {name: m, type: mast, on: base}\n")
    with pytest.raises(ValueError, match="quote"):
        config.load(tmp_path / "yes.yaml")                               # YAML reads the key "on" as true: said, not guessed
    with pytest.raises(FileNotFoundError):
        config.load("no_such_robot")


def test_shipped_ranger_configuration_is_the_rig():
    """The Ranger Mini with its rig comes from its structure file: two lamps in the front face, the mast, two cameras."""
    from opticalnav_sim.hw import config
    cfg = config.load("ranger_mini")
    assert [(p["name"], p["type"]) for p in cfg.parts()] == [("headlight_left", "light"), ("headlight_right", "light"), ("mast", "mast"),
                                                             ("hdr", "camera"), ("12bit", "camera")]
    robot = config.spawn(hw.World(), cfg)
    assert isinstance(robot.base.spec, robots.RangerMini2) and robot.base.spec.dynamics and robot.base.spec.kerb_mass == 64.5
    assert not robot.calibrated and robot.sensors["hdr"].spec.model == "LUCID TDR054S"
    left, right = (robot.lights[n].world_matrix()[:3, 3] for n in ("headlight_left", "headlight_right"))
    assert np.allclose(left, [0.3605, 0.123, 0.28]) and np.allclose(right, [0.3605, -0.123, 0.28])


def test_lights_are_shapes_of_their_own_that_move_with_their_link(tmp_path):
    from opticalnav_sim.hw import lights
    from opticalnav_sim.hw.sim import _visual_dict
    lamp = lights.AreaLight(size=(0.1, 0.04), color=(1.0, 0.5, 0.0), radiance=10.0)
    descriptor = {"links": {link: [_visual_dict(v) for v in vs] for link, vs in lamp.visuals().items()}}
    model = bodies.build_model("lamp", descriptor, cache=tmp_path)
    assert 'type="rectangle"' not in model["links"]["light"] and len(model["lights"]["light"]) == 1    # the group holds the housing only
    text = bodies.scene_xml({"lamp": model}, [("robot_lamp", "lamp")])
    sid = bodies.light_id("robot_lamp", "light", 1)
    assert f'<shape type="rectangle" id="{sid}">' in text and '<rgb name="radiance" value="10, 5, 0" />' in text
    # the link at (1, 2, 3) facing the world's x: the rectangle there, its normal (its z) along x, 0.1 wide (y), 0.04 high (z)
    at = np.eye(4)
    at[:3, 3] = (1.0, 2.0, 3.0)
    (index, m), = bodies.light_poses(model, "light", at)
    assert index == 1 and np.allclose(m[:3, 3], [1.0, 2.0, 3.0]) and np.allclose(m[:3, 2], [1.0, 0.0, 0.0])
    assert np.allclose(np.abs(m[:3, 1]), [0.0, 0.05, 0.0]) and np.allclose(np.abs(m[:3, 0]), [0.0, 0.0, 0.02])
    dark = lights.AreaLight(enabled=False)
    assert [v.shape for v in dark.visuals()["light"]] == ["box"]                                        # off: the housing, no emitter
    with pytest.raises(ValueError):
        bodies.build_model("bare", {"links": {"light": [descriptor["links"]["light"][1]]}}, cache=tmp_path)


def test_a_camera_frame_from_an_unpolarized_render():
    """The server's rgb mode gives radiance alone: a camera's frame takes it as S0, the other Stokes images zero."""
    world = hw.SimWorld(scene="", client=object())
    cam = world.spawn(cameras.LucidTRI032S())
    k = cam.spec.intrinsics()
    plan = optics.render_plan(k)
    rad = np.full((plan.height, plan.width, 3), 0.5, np.float32)
    request = {"plan": plan, "intrinsics": k, "pose": np.eye(4), "time": 0.0}
    world.client = type("C", (), {"last_timing": {}})()
    frame = world.frame(request, {"radiance": rad}, ("s0", "s1", "s2"))
    assert frame.s0.shape == (k.height, k.width, 3) and abs(float(frame.s0[k.height // 2, k.width // 2, 0]) - 0.5) < 1e-6
    assert not frame.s1.any() and not frame.s2.any() and frame.s3 is None


# --- recordings of the viewer's robot (robot_record.py) --------------------------------------------------------------
def test_a_drive_is_logged_and_put_back():
    """A recording logs the robot each tick (pose, wheels, odometry, the editor camera) and replays it exactly."""
    from opticalnav_sim import robot_view
    from opticalnav_sim.hw import config
    session = robot_view.RobotSession("http://127.0.0.1:1", calibration="")
    with pytest.raises(ValueError):
        session.log_start()                                        # no robot yet
    world = hw.World()
    built = config.spawn(world, session.config, at=hw.pose2d(1.0, 2.0, 0.0, 90.0), name="robot")
    session.world, session.robot, session.sensors, session.start_pose = world, built.base, dict(built.sensors), built.base.pose
    session.editor_pose = lambda: ((0.0, 0.0, 1.5, 0.3, -0.2), "track")
    session.log_start()
    assert session.state()["recording"]["phase"] == "logging"
    built.base.spec.apply(built.base.state, hw.RangerControl("ackermann", 0.5, 20.0))
    log = []
    for _ in range(30):
        world.tick()
        log.append(session._entry())
    assert log[-1]["t"] > log[0]["t"] and log[0]["cam"] == (0.0, 0.0, 1.5, 0.3, -0.2) and log[0]["follow"] == "track"
    end_pose, end_steer, cam_end = built.base.pose, built.base.state.steer.copy(), built.sensors["hdr"].world_matrix().copy()
    session.restore(log[10])
    mid = built.sensors["hdr"].world_matrix().copy()
    assert built.base.pose == log[10]["pose"] and not np.allclose(mid, cam_end)                 # the cameras went back with it
    assert np.allclose(built.base.state.steer, log[10]["state"]["steer"])
    session.restore(log[-1])
    assert built.base.pose == end_pose and np.allclose(built.base.state.steer, end_steer) and np.allclose(built.sensors["hdr"].world_matrix(), cam_end)
    built.base.state.steer[:] = 9.0                                # a log's arrays are its own: the live state does not write into it
    assert not np.allclose(log[-1]["state"]["steer"], 9.0)
    # the odometry's place in the world: from the pose the drive started at (facing +y here)
    assert abs(log[-1]["odometry_xy"][0] - 1.0) < 0.5 and log[-1]["odometry_xy"][1] > 2.3
    assert session.log_stop() == [] and session.log is None        # the session's own log held nothing (no real-time thread here)


def test_clean_render_holds_fireflies_back_before_the_denoiser():
    """A still sensor frame: eight passes, a pass's pixel far above the others scaled down, then the denoiser."""
    from opticalnav_sim import robot_view

    class Client:
        seeds, denoised = [], None

        def render(self, views, dtype):
            v = views[0]
            self.seeds.append(v["seed"])
            img = np.full((6, 8, 3), 0.5, np.float32)
            if len(self.seeds) == 3:
                img[2, 3] = 4000.0                                 # one firefly, in one pass
            return [{"radiance": img, "depth": np.ones((6, 8), np.float32)}]

        def denoise(self, images):
            self.denoised = sorted(images)
            return {k: v for k, v in images.items()}
    client = Client()
    out = robot_view.clean_render(client, {"scan": "s", "spp": 64, "seed": 7, "denoise": True}, passes=8)
    assert len(set(client.seeds)) == 8 and client.denoised == ["s0"]
    assert set(out) >= {"radiance", "depth"} and out["radiance"].shape == (6, 8, 3)
    assert float(out["radiance"][2, 3].max()) < 1.0               # 4000 in one pass of eight: a plain mean would give 500
    assert np.allclose(out["radiance"][0, 0], 0.5)


def test_recording_frame_layout_and_follow_modes(tmp_path):
    from opticalnav_sim import gui, robot_record
    robot = {"x": 2.0, "y": 3.0, "z": 0.0, "yaw_deg": 30.0, "vx": 0.4, "vy": 0.0, "wz": 0.1, "command": [0.5, 0.0, 0.1],
             "wheels": [{"name": n, "label": n, "steer_deg": a, "speed": 0.4, "slip": 0.0, "mu": 0.6} for n, a in zip(("fl", "fr", "rl", "rr"), (10.0, 8.0, -10.0, -8.0))], "collided": False,
             "footprint": [[1.6, 2.7], [2.4, 2.7], [2.4, 3.3], [1.6, 3.3]], "material": "epoxy floor"}
    layout = robot_record.Layout("OpticalNav Sim | scene | robot", ["hdr", "12bit"], None, None, size=(960, 540))
    editor = np.full((96, 128, 3), 200, np.uint8)
    images = {"hdr": (np.full((58, 90, 3), 90, np.uint8), "HDR"), "12bit": (np.full((48, 64, 3), 30, np.uint8), "12-bit")}
    frame = np.asarray(layout.frame(1.25, editor, images, robot, [(1.0, 3.0), (2.0, 3.0)], [(1.0, 3.0), (2.1, 3.0)]))
    assert frame.shape == (540, 960, 3)
    assert frame[200, 300].tolist() == [200, 200, 200]            # the editor's view fills the left
    assert abs(int(frame[120, 800, 0]) - 90) <= 2 and abs(int(frame[330, 800, 0]) - 30) <= 2   # the sensors down the right
    for k in range(3):
        layout.frame(k / 20, editor, images, robot, [], []).save(tmp_path / f"frame_{k:05d}.jpg")
    import shutil
    if shutil.which("ffmpeg"):
        robot_record.encode(tmp_path, 20, tmp_path / "out.mp4")
        assert (tmp_path / "out.mp4").stat().st_size > 1000
    # the editor camera following a robot at (2, 3) facing +x: riding behind it, or standing where it is and looking at it
    at = (2.0, 3.0, 0.0, 0.0)
    x, y, z, h, e = gui.Session.follow_camera(gui.Session.FOLLOW, at, (9.0, 9.0, 9.0, 0.0, 0.0))
    assert np.allclose([x, y, z], [2.0 + gui.Session.FOLLOW[0], 3.0, gui.Session.FOLLOW[2]]) and abs(h - math.pi / 2) < 1e-9
    x, y, z, h, e = gui.Session.follow_camera("track", at, (2.0, 0.0, 2.0, 1.0, 0.5))
    assert (x, y, z) == (2.0, 0.0, 2.0) and abs(h) < 1e-9 and abs(e - math.atan2(0.5 - 2.0, 3.0)) < 1e-9     # heading 0 looks along +y


# --- other robots: the differential drive, the TurtleBot 4 ------------------------------------------------------------
def _diff(**kw):
    from opticalnav_sim.hw import drive
    kw = {"track": 0.3, "wheel_radius": 0.05, "max_speed": 1.0, "max_spin": 4.0, "max_accel": 100.0, "casters": ((0.12, 0.0),), "radius": 0.2,
          "size": (0.4, 0.4, 0.3), "kerb_mass": 10.0, **kw}
    world = hw.World(dt=0.02)
    return world, world.spawn(drive.DifferentialDrive(**kw), at=hw.pose2d(0.0, 0.0))


def test_differential_drive_kinematics_and_limits():
    world, r = _diff()
    spec = r.spec
    spec.apply(r.state, hw.Twist(0.5, 0.3, 1.0))                    # sideways is not something it does
    assert r.state.command.vy == 0.0
    for _ in range(100):
        world.tick()
    # an arc of radius v/w = 0.5 m for 2 s: 2 rad
    assert np.allclose([*r.pose.xyz[:2], math.radians(r.pose.rpy_deg[2])], [0.5 * math.sin(2.0), 0.5 * (1 - math.cos(2.0)), 2.0], atol=2e-2)
    assert np.allclose(r.state.speed, [0.5 - 0.15, 0.5 + 0.15])     # left slower in a left turn
    assert np.allclose(r.state.odometry[:2], r.pose.xyz[:2], atol=1e-9)
    spec.apply(r.state, hw.Twist(2.0, 0.0, 4.0))                    # over the top speed: the same curve, slower
    assert abs(r.state.command.vx / r.state.command.wz - 0.5) < 1e-9 and abs(r.state.command.vx + r.state.command.wz * 0.15 - 1.0) < 1e-9
    assert [w["steer_deg"] for w in spec.wheel_report(r.state)] == [None, None] and not spec.capabilities()["lateral"]
    assert spec.control({"mode": "drive", "forward": 1, "turn": 0, "side": 1, "speed": 0.4}) == hw.Twist(0.4, 0.0, 0.0)
    with pytest.raises(ValueError):
        spec.control({"mode": "ackermann", "linear": 0.5})          # a Ranger's mode
    assert "steer_left" not in spec.link_frames(None) and {"wheel_left", "wheel_right"} <= set(spec.visuals())


def test_differential_drive_dynamics_and_loads():
    # statics: the mass 0.03 m ahead of the axle and the caster 0.12 m ahead carry 3/4 and 1/4
    world, r = _diff(com_xy=(0.03, 0.0), dynamics=True, motor_torque=50.0)
    assert np.allclose(r.spec.load_share(), [0.375, 0.375, 0.25])
    world.ground.mu = 0.4
    r.spec.apply(r.state, hw.Twist(1.0))
    for _ in range(150):
        world.tick()
    assert abs(r.state.twist.vx - 1.0) < 0.03
    # wheels held: only they brake (0.75 of the weight), the caster rolls: v^2 / (2 mu g 0.75)
    x0, v0 = r.pose.xyz[0], r.state.twist.vx
    r.spec.max_accel = 1e6
    r.spec.apply(r.state, hw.Twist())
    for _ in range(150):
        world.tick()
    assert abs((r.pose.xyz[0] - x0) - v0 ** 2 / (2 * 0.4 * 9.81 * 0.75)) < 0.03 and abs(r.state.twist.vx) < 0.02


def test_round_base_on_a_map_follows_the_floor_and_stops_at_things():
    x = np.arange(-3.0, 3.0 + 1e-9, 0.05)
    heights = np.zeros((len(x), len(x)))
    heights[:, x >= 1.5] = 1.0                                       # a wall
    ground = hw.Ground()
    ground.set_height_map(heights, (-3.0, -3.0), 0.05)
    world, r = _diff()
    world.set_ground(ground)
    assert r.state.ground is ground
    r.spec.apply(r.state, hw.Twist(0.5))
    for _ in range(300):
        world.tick()
    assert r.state.collided and 1.20 < r.pose.xyz[0] <= 1.5 - 0.2 + 0.03 and np.allclose(r.state.speed, 0.0)
    r.spec.apply(r.state, hw.Twist(0.0, 0.0, 1.0))                   # against the wall it still turns: it is round
    for _ in range(50):
        world.tick()
    assert r.pose.rpy_deg[2] > 30.0
    ramp = hw.Ground(step=0.06)                                      # 5 degrees up along x: three contacts make the plane
    ramp.set_height_map(np.tile(np.tan(math.radians(5.0)) * (x + 3.0), (len(x), 1)), (-3.0, -3.0), 0.05)
    world, r = _diff()
    world.set_ground(ramp)
    r.state.contact = None
    world.tick()
    assert abs(r.pose.rpy_deg[1] + 5.0) < 1.0 and abs(r.pose.rpy_deg[0]) < 0.5


def test_turtlebot4_and_rigs_on_either_robot():
    from opticalnav_sim.hw import config, drive, sensing
    world = hw.World()
    tb = config.spawn(world, "turtlebot4")
    assert isinstance(tb.base.spec, drive.DifferentialDrive) and tb.base.spec.max_speed == 0.31 and tb.config.rig == ""
    assert set(tb.sensors) == {"lidar", "depth", "oak_rgb"} and isinstance(tb.sensors["lidar"].spec, sensing.Lidar)
    assert np.allclose(tb.sensors["lidar"].world_matrix()[:3, 3], [-0.04, 0.0, 0.1929])
    cam = tb.sensors["oak_rgb"].spec
    assert abs(cam.full_intrinsics().hfov_deg() - 69.0) < 0.5 if callable(cam.full_intrinsics().hfov_deg) else abs(cam.full_intrinsics().hfov_deg - 69.0) < 0.5
    # the same rig on both: on each one's own deck
    for robot, deck in (("turtlebot4", 0.347), ("ranger_mini", None)):
        r = config.spawn(hw.World(), robot, rig="hdr_12bit_frame")
        deck = deck or r.base.spec.size[2]
        assert np.allclose(r.sensors["hdr"].world_matrix()[:3, 3], [0.12, 0.05, deck + 0.27 + 0.02])
        assert abs(r.base.spec.payload - 1.4) < 1e-9 and r.parts["frame"].spec.visuals()["frame"][0].shape == "profile"
    assert abs(config.height(config.load("turtlebot4")) - 0.347) < 0.01          # its sensor plate, the LiDAR and camera are under it
    assert 0.63 < config.height(config.load("turtlebot4", rig="hdr_12bit_frame")) < 0.70 and 1.15 < config.height("ranger_mini") < 1.35
    heavy = config.load("turtlebot4", rig="hdr_12bit_mast")
    assert heavy.warnings() and not config.load("ranger_mini", rig="hdr_12bit_mast").warnings()
    assert config.load("ranger_mini", rig="none").sensors() == [] and config.load("ranger_mini").rig == "hdr_12bit_mast"
    # a rig changed where it is used: a part's numbers, a part left out
    cfg = config.load("turtlebot4", rig="hdr_12bit_frame", edits={"hdr": {"mount": {"rpy_deg": [0, 10, 0]}}, "12bit": None})
    r = config.spawn(hw.World(), cfg)
    assert "12bit" not in r.sensors and np.allclose(r.sensors["hdr"].mount.xyz, (0.06, 0.05, 0.02)) and r.sensors["hdr"].mount.rpy_deg == (0.0, 10.0, 0.0)
    with pytest.raises(ValueError):
        config.load("turtlebot4", rig="hdr_12bit_frame", edits={"hrd": {}})
    with pytest.raises(FileNotFoundError, match="hdr_12bit_frame"):
        config.load("turtlebot4", rig="hdr_12bit_fram")


def test_profile_frame_bars_stand_where_they_are_written():
    f = rigs.ProfileFrame(profile=0.02, bars=({"from": [0, 0, 0], "to": [0, 0, 0.5]}, {"from": [0.1, -0.2, 0.3], "to": [0.1, 0.2, 0.3]}),
                          points={"top": {"xyz": [0, 0, 0.5]}})
    for v, (a, b) in zip(f.visuals()["frame"], (((0, 0, 0), (0, 0, 0.5)), ((0.1, -0.2, 0.3), (0.1, 0.2, 0.3)))):
        m = v.origin.matrix()
        ends = [m @ [0, 0, s * v.size[1] / 2, 1] for s in (-1, 1)]
        assert np.allclose([e[:3] for e in ends], [a, b], atol=1e-9) and abs(np.linalg.det(m[:3, :3]) - 1.0) < 1e-9
    assert np.allclose(f.link_frames(None)["top"][:3, 3], [0, 0, 0.5])
    with pytest.raises(ValueError):
        rigs.ProfileFrame(bars=({"from": [0, 0, 0], "too": [0, 0, 1]},))


def test_configuration_mistakes_are_named():
    """Nothing a file says is passed over: an unknown key, at any level, names the nearest known one."""
    from opticalnav_sim.hw import config
    base = {"type": "differential", "params": {"track": 0.3}}
    cam = {"name": "c", "type": "camera", "model": "lucid_tri032s"}
    for data, word in (({"bse": base, "base": base}, "base"), ({"base": {**base, "parms": {}}}, "params"), ({"base": {"type": "diferential"}}, "differential"),
                       ({"base": base, "parts": [{**cam, "mnt": {"xyz": [0, 0, 0]}}]}, "mount"), ({"base": base, "parts": [{**cam, "mount": {"xzy": [0, 0, 0]}}]}, "xyz"),
                       ({"base": base, "parts": [{**cam, "name": "m", "type": "mast", "mode": 1}]}, "model|has:")):
        with pytest.raises(ValueError, match=word):
            config.load({"name": "x", **data})
    for part, word in (({**cam, "params": {"render_scal": 0.5}}, "render_scale"), ({**cam, "isp": {"tonee": "srgb"}}, "tone"), ({**cam, "exposure": {"autoo": True}}, "auto"),
                       ({**cam, "sensor": "imx26"}, "imx265"), ({**cam, "model": "lucid_tri32s"}, "lucid_tri032s"), ({"name": "l", "type": "lidar", "params": {"sample": 3}}, "samples")):
        with pytest.raises(ValueError, match=word):
            config.spawn(hw.World(), {"name": "x", "base": base, "parts": [part]})
    # what a camera's entry says about its picture is used
    r = config.spawn(hw.World(), {"name": "x", "base": base, "parts": [{**cam, "isp": {"tone": "linear"}, "exposure": {"auto": False, "exposure_s": 0.01}}]})
    assert r.sensors["c"].spec.processing().tone == "linear" and r.sensors["c"].state.exposure.exposure_s == 0.01 and not r.sensors["c"].state.exposure.auto


def test_extends_and_broken_files(tmp_path, monkeypatch):
    from opticalnav_sim.hw import config
    monkeypatch.setenv("OPTICALNAV_ROBOTS", str(tmp_path))
    (tmp_path / "tb_lab").mkdir()
    (tmp_path / "tb_lab" / "robot.yaml").write_text("name: tb_lab\nextends: turtlebot4\nlabel: Lab\nbase: {params: {max_speed: 0.46}}\nremove: [oak_rgb]\n"
                                                    "parts:\n  - {name: lidar, params: {samples: 720}}\n  - {name: cam, type: camera, model: lucid_tri032s, link: deck}\nrig: polar_frame\nedits: {polar: {label: P}}\n")
    (tmp_path / "rigs").mkdir()
    (tmp_path / "rigs" / "mine.yaml").write_text("kind: rig\nlabel: Mine\nmass: 0.2\nparts:\n  - {name: c2, type: camera, model: lucid_tri032s}\n")
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "robot.yaml").write_text("name: bad\nbase: {type: differential, prams: {}}\n")
    cfg = config.load("tb_lab")
    assert [s["name"] for s in cfg.sensors()] == ["lidar", "depth", "cam", "polar"] and cfg.label == "Lab" and cfg.rig == "polar_frame"
    r = config.spawn(hw.World(), cfg)
    assert r.base.spec.max_speed == 0.46 and r.base.spec.track == 0.233 and r.sensors["lidar"].spec.samples == 720 and r.sensors["lidar"].spec.range_m == (0.15, 12.0)
    assert cfg.parts()[-1]["label"] == "P" and config.load("tb_lab", rig="mine").sensors()[-1]["name"] == "c2"   # the file's edits are of its own rig
    listed = {c["name"]: c for c in config.available()}
    assert "params" in listed["bad"]["error"] and "mine" in [g["name"] for g in config.available_rigs()]
    (tmp_path / "a.yaml").write_text("name: a\nextends: b\nbase: {}\n")
    (tmp_path / "b.yaml").write_text("name: b\nextends: a\nbase: {}\n")
    with pytest.raises(ValueError, match="circle"):
        config.load("a")


# --- sensors other than cameras ----------------------------------------------------------------------------------------
class _Room:
    """A render server of an empty room, 8 x 6 m, floor at 0: exact depth along the axis, for depth views."""
    def __init__(self):
        self.views = []

    def register_body(self, name, links):
        return {}

    def render(self, views, dtype="float32"):
        out = []
        for v in views:
            assert v["renderer"] == "depth"
            self.views.append(v)
            c2w = np.asarray(v["camera_to_world"])                   # dataset convention: x right, y up, looks along -z; y up world
            w, h = v["width"], v["height"]
            t = math.tan(math.radians(v["hfov_deg"]) / 2)
            u, r = np.meshgrid((2 * (np.arange(w) + 0.5) / w - 1) * t, (1 - 2 * (np.arange(h) + 0.5) / h) * t * h / w)
            d = np.stack([u, r, -np.ones_like(u)], -1) @ c2w[:3, :3].T
            o = c2w[:3, 3]
            # the world's x, y, z here are Mitsuba's x, height, -y
            best = np.full(u.shape, np.inf)
            for axis, planes in ((0, (-4.0, 4.0)), (2, (-3.0, 3.0)), (1, (0.0, 3.0))):
                for at in planes:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        s = (at - o[axis]) / d[..., axis]
                    best = np.where((s > 1e-6) & (s < best), s, best)
            out.append({"depth": best.astype(np.float32)})
        return out


def test_lidar_and_depth_camera_measure_a_room():
    from opticalnav_sim.hw import sensing
    room = _Room()
    world = hw.SimWorld(scene="", client=room)
    base = world.spawn(hw.drive.DifferentialDrive(), at=hw.pose2d(1.0, 0.5, 0.0, 30.0))
    lidar = world.spawn(sensing.Lidar(samples=360, elevations_deg=(0.0, 10.0), resolution_px=1024, range_m=(0.1, 50.0)), attach_to=base, link="base_link", mount=Transform((0.0, 0.0, 0.5)))
    got = world.sense(lidar)
    assert got.modality == "lidar" and len(room.views) == 4 and got.data["ranges"].shape == (2, 360) and got.picture.shape[2] == 3
    # each direction to the walls x = -4, 4 and y = -3, 3
    az = np.radians(got.data["azimuth_deg"] + 30.0)
    with np.errstate(divide="ignore"):
        wall = np.minimum(np.where(np.cos(az) > 0, (4.0 - 1.0) / np.cos(az), (-4.0 - 1.0) / np.cos(az)), np.where(np.sin(az) > 0, (3.0 - 0.5) / np.sin(az), (-3.0 - 0.5) / np.sin(az)))
    assert np.abs(got.data["ranges"][0] - wall).max() < 0.04 and np.abs(got.data["ranges"][1] * math.cos(math.radians(10.0)) - wall).max() < 0.05
    front = got.data["points"][np.argmin(np.abs(got.data["azimuth_deg"]))]   # straight ahead, in the lidar's frame
    assert front[0] > 2.0 and abs(front[1]) < 0.05
    far = world.spawn(sensing.Lidar(range_m=(0.15, 3.0)), attach_to=base, link="base_link", mount=Transform((0.0, 0.0, 0.5)))
    r = world.sense(far).data["ranges"]
    assert (r[r > 0] <= 3.0).all() and (r == 0).any()                # out of range: no return
    cam = world.spawn(sensing.DepthCamera(width=640, height=400, hfov_deg=80.0, range_m=(0.2, 20.0)), attach_to=base, link="base_link", mount=Transform((0.0, 0.0, 0.5), (0.0, 20.0, 0.0)))
    got = world.sense(cam)
    z = got.data["depth"]
    assert z.shape == (200, 320) and got.data["valid"].all()
    # the floor 0.5 m under a camera pitched 20 degrees down: straight ahead at 0.5 / sin 20
    assert abs(z[100, 160] - min(0.5 / math.sin(math.radians(20.0)), 99)) < 0.02
    noisy = world.spawn(sensing.DepthCamera(width=640, height=400, hfov_deg=80.0, range_m=(0.2, 20.0), baseline_m=0.075, disparity_noise_px=0.2), attach_to=base, link="base_link", mount=Transform((0.0, 0.0, 0.5), (0.0, 20.0, 0.0)))
    e = world.sense(noisy).data["depth"] - z
    sigma = z ** 2 * 0.2 / (noisy.spec.focal_px(640) * 0.075)
    assert abs(float((e / sigma).std()) - 1.0) < 0.05 and np.array_equal(world.sense(noisy).data["depth"], world.sense(noisy).data["depth"])
    with pytest.raises(ValueError):
        world.request(lidar)                                         # a camera's request is a camera's


def test_robot_session_exchanges_robots_and_keeps_rigs(tmp_path, monkeypatch):
    from opticalnav_sim import robot_view
    from opticalnav_sim.hw import config, robots as robot_models
    monkeypatch.setattr(robot_models, "MODELS", tmp_path)
    session = robot_view.RobotSession("http://127.0.0.1:1", calibration="")
    assert session.state()["config"]["rig"] == "hdr_12bit_mast"
    session.use(rig="hdr_12bit_frame")
    assert session.state()["config"]["name"] == "ranger_mini" and session.state()["config"]["rig"] == "hdr_12bit_frame"
    session.use("turtlebot4")
    assert [s["name"] for s in session.state()["config"]["sensors"]] == ["lidar", "depth", "oak_rgb"]
    session.command({"mode": "drive", "forward": 1, "speed": 0.2})
    assert session.control[0] == hw.Twist(0.2, 0.0, 0.0)
    with pytest.raises(FileNotFoundError):
        session.use(rig="nothing")
    assert session.state()["config"]["name"] == "turtlebot4"         # a wrong one changes nothing
    text = (config.BUILTIN / "rigs" / "polar_frame.yaml").read_text().replace("0.2]}]", "0.4]}]")
    session.save_rig("tall_post", text)
    assert session.state()["config"]["rig"] == "tall_post" and (tmp_path / "rigs" / "tall_post.yaml").read_text() == text
    for name, body in (("polar_frame", text), ("bad name", text), ("broken", text.replace("profile:", "profle:")), ("broken", "kind: rig\nparts: [{name: c, type: camera, model: nope}]")):
        with pytest.raises((ValueError, KeyError)):
            session.save_rig(name, body)
    assert not (tmp_path / "rigs" / "broken.yaml").exists()
    # a robot that stands somewhere is exchanged there
    started = []
    session.robot = type("A", (), {"pose": hw.pose2d(3.0, 4.0, 0.1, 50.0)})()
    session.scene = "scene"
    monkeypatch.setattr(session, "_start", lambda scene, build: started.append((scene, build)))
    monkeypatch.setattr(session, "_build_at", lambda run, scene, pose: started.append(pose))
    session.use("ranger_mini")
    started[0][1](1, "scene")
    assert started[0][0] == "scene" and started[1].xyz[:2] == (3.0, 4.0)


def test_a_light_wheel_with_a_strong_drive_is_stable():
    """The TurtleBot 4's numbers: its drive's gain over its wheel's inertia is far past what an explicit step takes."""
    from opticalnav_sim.hw import config
    for command, want in ((hw.Twist(0, 0, 1.0), (0, 0, 1.0)), (hw.Twist(0.3), (0.3, 0, 0)), (hw.Twist(0.2, 0, 0.5), (0.2, 0, 0.5))):
        world = hw.World()
        r = config.spawn(world, "turtlebot4")
        r.base.spec.apply(r.base.state, command)
        for _ in range(60):
            world.tick()
        t = r.base.state.twist
        assert np.allclose([t.vx, t.vy, t.wz], want, atol=0.01) and np.abs(r.base.state.slip).max() < 0.005


def test_robot_tree_and_preview():
    """A configuration as a tree of its hardware (base, wheels, mounts, parts, cameras) and its picture, without a server."""
    from opticalnav_sim.hw import preview, tree
    t = tree.config_tree("ranger_mini", "hdr_12bit_mast")
    kids = {c["id"]: c for c in t["children"]}
    assert t["kind"] == "base" and [w["kind"] for w in kids["base/wheels"]["children"]] == ["wheel"] * 4
    mast = kids["base@deck"]["children"][0]
    assert kids["base@deck"]["kind"] == "mount" and mast["kind"] == "mast" and mast["rig"]
    cams = mast["children"][0]["children"]
    assert [(c["sensor"], c["kind"]) for c in cams] == [("hdr", "camera"), ("12bit", "camera")]
    t = tree.config_tree("turtlebot4", "none")
    assert [w["label"] for w in t["children"][0]["children"]] == ["왼쪽", "오른쪽"]
    ids = {n["id"] for c in t["children"] for n in [c, *c["children"]]}
    assert {"lidar", "depth", "oak_rgb"} <= ids and not any(n.startswith("frame") for n in ids)
    png = preview.render(hw.config.spawn(hw.World(), "turtlebot4", rig="hdr_12bit_frame"), size=(80, 60))
    from PIL import Image
    import io
    im = np.asarray(Image.open(io.BytesIO(png)))
    assert im.shape == (60, 80, 3) and (np.abs(im.astype(int) - preview.BACKGROUND).sum(-1) > 30).mean() > 0.05   # something drawn


def test_sensor_render_settings_are_each_cameras_own():
    """A robot camera's own pass and target spp, accumulation and resolution share (the viewer's camera tab sets them)."""
    from opticalnav_sim import robot_view
    session = robot_view.RobotSession("http://127.0.0.1:1", calibration="")
    robot = hw.config.spawn(hw.World(), "ranger_mini", rig="hdr_12bit_mast")
    session.sensors = dict(robot.sensors)
    s = session.sensor_settings("hdr", "polar")
    assert (s["pass_spp"], s["target_spp"], s["accumulate"]) == (16, 256, True) and 0 < s["scale"] <= 1
    assert session.sensor_settings("hdr", "rgb")["target_spp"] == 1024        # unpolarized: a pass costs less
    s = session.set_sensor_settings("hdr", {"pass_spp": 64, "accumulate": False, "scale": 0.5})
    assert (s["pass_spp"], s["accumulate"], s["scale"]) == (64, False, 0.5) and robot.sensors["hdr"].spec.render_scale == 0.5
    assert session.sensor_settings("12bit", "polar")["pass_spp"] == 16       # the other camera keeps its own
    for bad in ({"pass_spp": 0}, {"scale": 1.5}, {"spp": 4}):
        with pytest.raises(ValueError):
            session.set_sensor_settings("hdr", bad)
    with pytest.raises(ValueError):
        session.set_sensor_settings("nope", {"pass_spp": 8})
    assert session.sensor_settings("12bit", "polar")["isp"] is True             # a camera's sensor model and ISP: on
    assert session.set_sensor_settings("12bit", {"isp": False})["isp"] is False
    assert session.sensor_settings("hdr", "polar")["denoise"] is True              # each camera's own denoiser switch
    assert session.set_sensor_settings("hdr", {"denoise": False})["denoise"] is False
    assert session.sensor_settings("12bit", "polar")["denoise"] is True


def test_camera_without_isp_is_its_render_through_the_lens():
    """ISP off: the render with the lens's distortion and a display tone, no sensor model; the same size as with it."""
    from opticalnav_sim import robot_view
    from opticalnav_sim.hw.sim import SimWorld

    class Stub:
        last_timing = {}

        def register_body(self, *a, **k):
            return {}

        def meta(self, *a):
            return {}
    world = SimWorld(scene="", spp=16, client=Stub())
    robot = hw.config.spawn(world, "ranger_mini", rig="hdr_12bit_mast")
    session = robot_view.RobotSession("http://127.0.0.1:1", calibration="")
    session.world, session.sensors = world, dict(robot.sensors)
    actor = robot.sensors["12bit"]
    request = actor.spec.request(world, actor, spp=16, scale=0.25)
    v = request["views"][0]
    out = {"radiance": np.random.default_rng(0).lognormal(-1.0, 1.0, (v["height"], v["width"], 3)).astype(np.float32)}
    off = session._read("12bit", actor, request, [out], isp=False)
    on = session._read("12bit", actor, request, [out], isp=True)
    assert off.picture.dtype == np.uint8 and off.picture.shape == on.picture.shape
    assert "ISP" in off.info["format"] and 30 < off.picture.mean() < 225


def test_camera_reading_in_a_worker_process_is_the_same_as_here():
    """hw.offload: the sensor model and ISP in a worker process give the raw and picture of the reading done here (and
    of SimWorld.expose + isp.process, the way it was); a worker that dies leaves the reading to this process."""
    import os
    import signal

    from opticalnav_sim.hw import offload
    from opticalnav_sim.hw.sim import SimWorld

    class Stub:
        last_timing = {}

        def register_body(self, *a, **k):
            return {}

        def meta(self, *a):
            return {}
    world = SimWorld(scene="", spp=16, client=Stub())
    robot = hw.config.spawn(world, "ranger_mini", rig="hdr_12bit_mast")
    pool = offload.Pool(1)
    try:
        assert pool.start(timeout=60.0), pool.off
        rng = np.random.default_rng(0)
        for name in ("hdr", "12bit"):
            actor = robot.sensors[name]
            spec = actor.spec
            request = spec.request(world, actor, spp=16, scale=0.25)
            v = request["views"][0]
            out = {"radiance": rng.lognormal(-1.0, 1.0, (v["height"], v["width"], 3)).astype(np.float32)}
            spec.read(world, actor, request, [out])            # the auto exposure started (it settles on its first frame)
            snap, seed = spec.snapshot(actor.state), world.seed

            def again():
                spec.restore(actor.state, snap)
                world.seed = seed
            raw0, _ = world.expose(actor, request, out)
            picture0 = isp.process(raw0, spec.processing())
            exposure0 = (actor.state.exposure.exposure_s, actor.state.exposure.gain_db)
            again()
            here = spec.read(world, actor, request, [out])
            again()
            there = spec.read(world, actor, request, [out], develop=pool.develop)
            assert pool.off is None
            for r in (here, there):
                raw = r.data["raw"]
                assert raw.data.dtype == raw0.data.dtype and np.array_equal(raw.data, raw0.data)
                assert (raw.exposure_s, raw.gain_db, raw.intrinsics, raw.time, raw.meta) == \
                    (raw0.exposure_s, raw0.gain_db, raw0.intrinsics, raw0.time, raw0.meta)
                assert np.array_equal(raw.pose, raw0.pose) and np.array_equal(r.picture, picture0)
            assert here.info == there.info
            assert (actor.state.exposure.exposure_s, actor.state.exposure.gain_db) == exposure0   # moved on the same way
        # an error of the reading itself is raised as without the pool, which goes on
        with pytest.raises(ValueError):   # no light, and no frame to take it from
            pool.develop(spec, None, 0.01, 0.0, 1, 1.0, raw0.intrinsics, raw0.pose, 0.0)
        assert pool.off is None and pool.develop(spec, np.ones((8, 8)), 0.01, 0.0, 1, 1.0, raw0.intrinsics, raw0.pose, 0.0)[1].shape == (8, 8, 3)
        # a worker that dies: the reading is done here, the same, and the pool stays off
        os.kill(pool.pids()[0], signal.SIGKILL)
        again()
        after = spec.read(world, actor, request, [out], develop=pool.develop)
        assert pool.off is not None and np.array_equal(after.picture, here.picture)
        assert np.array_equal(spec.read(world, actor, request, [out], develop=pool.develop).data["raw"].data.shape, raw0.data.shape)
    finally:
        pool.shutdown()
    assert offload.Pool(0).develop is not None and offload.Pool(0).off   # 0 processes: readings here
