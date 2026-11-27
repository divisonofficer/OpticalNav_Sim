"""hw.arm: the MuJoCo Menagerie arms (needs mujoco and the models; skipped without them): their links and drawing, the
tool centre point, inverse kinematics, motion toward targets, and an arm on a configured robot."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opticalnav_sim.hw import arm, config  # noqa: E402
from opticalnav_sim.hw.actor import World  # noqa: E402

if not (arm.MENAGERIE / arm.ARMS["piper"]["xml"]).is_file():
    pytest.skip("MuJoCo Menagerie's models are not installed", allow_module_level=True)


@pytest.mark.parametrize("name", ["piper", "panda"])
def test_arm_draws_moves_and_reaches(name):
    a = arm.MjcfArm(model=name)
    st = a.make_state()
    frames = a.link_frames(st)
    assert next(iter(frames)) == a.root and np.allclose(frames[a.root], np.eye(4))
    vis = a.visuals()
    assert vis and all(Path(v.mesh).is_file() for vs in vis.values() for v in vs)
    # the tool centre point between the fingers, the fingers closing across its z
    assert abs(a.tcp_offset[2, 3]) > 0.05 and abs(float(a.close_axis @ [0.0, 0.0, 1.0])) < 1e-6
    # inverse kinematics back to a pose the arm reached, from home
    q_goal = a.home + np.array([0.3, -0.2, 0.2, 0.15, 0.2, -0.3, 0.1][:len(a.home)])
    q_goal = np.clip(q_goal, a.limits[:, 0], a.limits[:, 1])
    target = a.tcp(st, q_goal)
    q = a.ik(st, target)
    got = a.tcp(st, q)
    assert np.linalg.norm(got[:3, 3] - target[:3, 3]) < 0.002
    assert np.allclose(got[:3, :3], target[:3, :3], atol=0.02)
    with pytest.raises(ValueError):
        far = np.eye(4)
        far[:3, 3] = [3.0, 0.0, 0.0]
        a.ik(st, far, restarts=2)
    # motion: every joint arrives together, at the arm's speed
    st.target = q_goal
    st.grip_target = 0.0
    steps = 0
    while a.moving(st) and steps < 1000:
        a.step(st, None, 0.05)
        steps += 1
    far_joint = float(np.max(np.abs(q_goal - a.home)))
    assert np.allclose(st.q, q_goal) and st.grip == 0.0
    assert steps == max(int(np.ceil(far_joint / (a.speed * 0.05) - 1e-9)), int(np.ceil(1.0 / (a.grip_speed * 0.05) - 1e-9)))


def test_arm_on_the_ranger_rig():
    w = World()
    robot = config.spawn(w, "ranger_mini", rig="mast_ouster_piper")
    arm_actor = robot.parts["arm"]
    assert isinstance(arm_actor.spec, arm.MjcfArm) and "ouster" in robot.sensors
    base = arm_actor.world_matrix()
    assert abs(base[0, 3] - 0.26) < 1e-9 and base[2, 3] > 0.3   # on the deck, ahead of the mast
