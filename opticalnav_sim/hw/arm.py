"""Robot arms from MuJoCo models (MJCF): an AgileX PiPER or a Franka Emika Panda on a robot, its joints moved.

    part = {"name": "arm", "type": "arm", "params": {"model": "piper"}}          # a robot's or a rig's part
    arm = world.spawn(MjcfArm(model="panda"), attach_to=robot, link="deck", mount=Transform((0.26, 0, 0)))
    q = arm.spec.ik(arm.state, target)          # joint angles that put the tool centre point at ``target`` (4x4, arm frame)
    arm.state.target = q; arm.state.grip_target = 0.0     # move there; close the gripper
    world.tick()                                # the joints go toward their targets at the arm's joint speed

The models are MuJoCo Menagerie's (github.com/google-deepmind/mujoco_menagerie: ``agilex_piper`` MIT, ``franka_emika_panda``
Apache-2.0), read where the ``robot_descriptions`` package keeps them (``$OPTICALNAV_MENAGERIE``, default
``~/.cache/robot_descriptions/mujoco_menagerie``); nothing of them is copied into this repository. MuJoCo compiles the
model: its bodies are the arm's links, its visual geoms (group 2) what is drawn, merged per link and colour into PLY
files under ``~/.cache/opticalnav_sim/arms`` (the PiPER's 140 visual meshes become 20). Forward kinematics and the
Jacobian for the inverse kinematics are MuJoCo's own (mj_kinematics, mj_jac) on a model of the arm alone.

What is simulated: joint positions moving toward their targets at a joint speed (all joints arriving together), the
gripper's fingers likewise; no dynamics, no torque limits, no self-collision. Holding an object is the caller's (the
pick scenario fixes the object to the tool centre point while the gripper is closed on it: tools/arm_pick_demo.py).
"""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .actor import Spec

MENAGERIE = Path(os.environ.get("OPTICALNAV_MENAGERIE", Path.home() / ".cache" / "robot_descriptions" / "mujoco_menagerie"))
CACHE = Path.home() / ".cache" / "opticalnav_sim" / "arms"

# label, the MJCF under MENAGERIE, its arm joints (base to wrist), its finger joints (name -> sign of opening), a
# finger's travel open (m), the link the tool centre point is on and that point's place (the finger pads' middle, closed:
# measured on the model's pad geoms by ``_tcp``), the home pose (the model's "home" keyframe), a pose for driving
# (``travel``, ours: clear of a mast 0.16 m behind the arm's base and of what the robot drives up to: the PiPER folded,
# the hand 0.29 m ahead; the Panda's home turned to the robot's left), the maker's reach and payload
ARMS = {
    "piper": dict(label="AgileX PiPER", xml="agilex_piper/piper.xml", joints=[f"joint{i}" for i in range(1, 7)],
                  fingers={"joint7": 1.0, "joint8": -1.0}, finger_open=0.035, tcp_body="link6",
                  pads=("link7", "link8"), home=(0.0, 1.57, -1.3485, 0.0, 0.0, 0.0), travel=(0.0, 2.62, -0.84, 0.0, 0.0, 0.0),
                  reach_m=0.626, payload_kg=1.5),
    "panda": dict(label="Franka Emika Panda", xml="franka_emika_panda/panda.xml", joints=[f"joint{i}" for i in range(1, 8)],
                  fingers={"finger_joint1": 1.0, "finger_joint2": 1.0}, finger_open=0.04, tcp_body="hand",
                  pads=("left_finger", "right_finger"), home=(0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853),
                  travel=(1.5708, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853), reach_m=0.855, payload_kg=3.0),
}
_LOADED: dict = {}


def _load(name: str):
    """The compiled MuJoCo model of an arm (cached): (model, xml path)."""
    import mujoco
    if name not in ARMS:
        raise ValueError(f"arm {name!r}: one of {sorted(ARMS)}")
    path = MENAGERIE / ARMS[name]["xml"]
    if not path.is_file():
        raise FileNotFoundError(f"{path} is not there: install the models (pip install robot_descriptions, then load "
                                f"robot_descriptions.{name}_mj_description once) or set OPTICALNAV_MENAGERIE")
    if name not in _LOADED:
        _LOADED[name] = (mujoco.MjModel.from_xml_path(str(path)), path)
    return _LOADED[name]


def _quat_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _rotvec(r: np.ndarray) -> np.ndarray:
    """The rotation vector (axis times angle) of a 3x3 rotation."""
    c = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    a = math.acos(c)
    if a < 1e-9:
        return np.zeros(3)
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if a > math.pi - 1e-4:   # near half a turn: the axis from the symmetric part
        w, vecs = np.linalg.eigh((r + r.T) / 2.0)
        axis = vecs[:, int(np.argmax(w))]
        return axis * a
    return v / (2.0 * math.sin(a)) * a


def _axis_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


@dataclass
class ArmState:
    q: np.ndarray                 # arm joint positions (rad)
    target: np.ndarray            # where they go
    grip: float = 1.0             # the gripper's opening, 0 closed .. 1 open
    grip_target: float = 1.0
    holding: str | None = None    # what it holds (the caller's: an object id)


@dataclass
class MjcfArm(Spec):
    """An arm of ARMS on a robot (module docstring). ``speed``: the fastest joint's speed (rad/s) on a move;
    ``grip_speed``: the gripper's, in openings a second. Links: the MJCF's bodies, the first its base."""
    kind = "arm"
    model: str = "piper"
    speed: float = 1.0
    grip_speed: float = 1.5
    drawn: bool = True

    def __post_init__(self):
        import mujoco
        self.info = ARMS[self.model] if self.model in ARMS else None
        m, self.path = _load(self.model)
        self._m, self._d = m, mujoco.MjData(m)
        self.label = self.info["label"]
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.info["joints"]]
        self._qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self._dofs = np.array([m.jnt_dofadr[j] for j in jid])
        self.limits = np.array([m.jnt_range[j] for j in jid])
        self._fingers = [(int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]), s)
                         for j, s in self.info["fingers"].items()]
        self._bodies = [(b, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)) for b in range(1, m.nbody)]
        self.root = self._bodies[0][1]
        self._tcp_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.info["tcp_body"])
        self.home = np.array(self.info["home"], dtype=np.float64)
        self.travel = np.clip(np.array(self.info["travel"], dtype=np.float64), self.limits[:, 0], self.limits[:, 1])
        self.tcp_offset = self._tcp()
        self.close_axis = self._close_axis()

    # --- kinematics ---------------------------------------------------------------------------------------------------
    def _pose(self, q, grip: float = 1.0) -> None:
        import mujoco
        d = self._d
        d.qpos[:] = 0.0
        d.qpos[self._qadr] = q
        for adr, s in self._fingers:
            d.qpos[adr] = s * float(grip) * self.info["finger_open"]
        mujoco.mj_kinematics(self._m, d)

    def _body(self, b: int) -> np.ndarray:
        out = np.eye(4)
        out[:3, :3] = self._d.xmat[b].reshape(3, 3)
        out[:3, 3] = self._d.xpos[b]
        return out

    def _tcp(self) -> np.ndarray:
        """The tool centre point on the TCP link (4x4: its place, the link's axes): the middle of the finger pads, closed."""
        import mujoco
        m = self._m
        self._pose(self.home, 0.0)
        pads = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in self.info["pads"]]
        pts = []
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) in pads:
                pts.append(self._d.geom_xpos[g])
        # the pads' far half: where an object is held (the fingers' tips, not their roots at the hand)
        pts = np.array(pts)
        link = self._body(self._tcp_body)
        local = (pts - link[:3, 3]) @ link[:3, :3]
        far = local[:, 2] >= np.median(local[:, 2])
        out = np.eye(4)
        out[:3, 3] = local[far].mean(0)
        out[0, 3] = out[1, 3] = 0.0     # on the link's axis: between the fingers
        return out

    def _close_axis(self) -> np.ndarray:
        """The direction the fingers close along, in the tool centre point's frame (unit; across its z)."""
        import mujoco
        m = self._m
        pads = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in self.info["pads"]]
        self._pose(self.home, 1.0)
        tcp = self._body(self._tcp_body) @ self.tcp_offset
        a, b = (self._body(p)[:3, 3] for p in pads)
        self._pose(self.home, 0.0)
        a0, b0 = (self._body(p)[:3, 3] for p in pads)
        d = ((a - b) - (a0 - b0)) @ tcp[:3, :3]       # how the fingers part as they open
        d[2] = 0.0
        return d / np.linalg.norm(d)

    def link_frames(self, state) -> dict:
        q, grip = (state.q, state.grip) if state is not None else (self.home, 1.0)
        self._pose(q, grip)
        return {name: self._body(b) for b, name in self._bodies}

    def tcp(self, state, q=None) -> np.ndarray:
        """The tool centre point in the arm's frame (4x4: z along the fingers, y across them... as the TCP link's axes)."""
        self._pose(state.q if q is None else q, state.grip)
        return self._body(self._tcp_body) @ self.tcp_offset

    def _solve(self, q, target, grip, iters, rot_weight, tol, rest=None, rest_gain: float = 0.2):
        """Damped least squares toward ``target``; ``rest``: joint angles the redundant joints (a 7-joint arm's elbow) are
        drawn back to, in the Jacobian's null space (they would drift along a long path)."""
        import mujoco
        m, d = self._m, self._d
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        lam = 0.02
        for _ in range(iters):
            self._pose(q, grip)
            t = self._body(self._tcp_body) @ self.tcp_offset
            ep = target[:3, 3] - t[:3, 3]
            er = _rotvec(target[:3, :3] @ t[:3, :3].T)
            if np.linalg.norm(ep) < tol and np.linalg.norm(er) < 10 * tol:
                break
            e = np.concatenate([ep, rot_weight * er])
            mujoco.mj_comPos(m, d)   # the joints' motion axes, which mj_jac reads
            mujoco.mj_jac(m, d, jacp, jacr, t[:3, 3], self._tcp_body)
            J = np.vstack([jacp[:, self._dofs], rot_weight * jacr[:, self._dofs]])
            JJ = np.linalg.solve(J @ J.T + lam * lam * np.eye(6), J)      # (J J^T + lam^2)^-1 J
            dq = JJ.T @ e
            if rest is not None and J.shape[1] > 6:
                dq += (np.eye(J.shape[1]) - J.T @ JJ) @ (rest_gain * (np.asarray(rest) - q))
            step = np.max(np.abs(dq))
            if step > 0.2:
                dq *= 0.2 / step
            q = np.clip(q + dq, self.limits[:, 0], self.limits[:, 1])
        self._pose(q, grip)
        t = self._body(self._tcp_body) @ self.tcp_offset
        return q, float(np.linalg.norm(target[:3, 3] - t[:3, 3])), float(np.degrees(np.linalg.norm(_rotvec(target[:3, :3] @ t[:3, :3].T))))

    def ik(self, state, target: np.ndarray, q0=None, iters: int = 300, rot_weight: float = 0.5, tol: float = 1e-4,
           restarts: int = 24) -> np.ndarray:
        """Joint angles that put the tool centre point at ``target`` (4x4, arm frame), within the joint limits: damped least
        squares from ``q0`` (default: where the arm is), then from home and from seeded random poses until one gets within
        2 mm and 1 degree (the nearest to ``q0`` of those). Raises when none gets within 5 mm and 3 degrees."""
        target = np.asarray(target, dtype=np.float64)
        start = np.array(state.q if q0 is None else q0, dtype=np.float64)
        q, dp, da = self._solve(start.copy(), target, state.grip, iters, rot_weight, tol)
        if dp < 0.002 and da < 1.0:
            return q
        # a long way from where it is: the target walked there from the tool's pose now, each step solved from the last
        # (a straight solve to a pose turned far from the present one stalls at a joint limit)
        self._pose(start, state.grip)
        now = self._body(self._tcp_body) @ self.tcp_offset
        rv = _rotvec(target[:3, :3] @ now[:3, :3].T)
        q = start.copy()
        for k in range(1, 13):
            f = k / 12.0
            step = np.eye(4)
            angle = np.linalg.norm(rv) * f
            step[:3, :3] = (_axis_rot(rv / max(np.linalg.norm(rv), 1e-12), angle) if angle > 1e-9 else np.eye(3)) @ now[:3, :3]
            step[:3, 3] = now[:3, 3] + (target[:3, 3] - now[:3, 3]) * f
            q, dp, da = self._solve(q, step, state.grip, iters, rot_weight, tol)
        if dp < 0.002 and da < 1.0:
            return q
        rng = np.random.default_rng(7)
        seeds = [self.home] + [rng.uniform(self.limits[:, 0], self.limits[:, 1]) for _ in range(restarts)]
        best = None
        for k, q in enumerate(seeds):
            q, dp, da = self._solve(np.array(q), target, state.grip, iters, rot_weight, tol)
            good = dp < 0.002 and da < 1.0
            score = (0 if good else 1, float(np.abs(q - start).sum()) if good else dp + da / 100.0)
            if best is None or score < best[0]:
                best = (score, q, dp, da)
            if good and k < 1:   # from home: the natural one
                break
        _, q, dp, da = best
        if dp > 0.005 or da > 3.0:
            raise ValueError(f"{self.label}: no joint angles reach that pose ({dp * 1000:.0f} mm, {da:.1f} degrees off)")
        return q

    # --- motion -------------------------------------------------------------------------------------------------------
    def make_state(self):
        return ArmState(self.home.copy(), self.home.copy())

    def step(self, state, pose, dt: float):
        dq = state.target - state.q
        far = float(np.max(np.abs(dq))) if len(dq) else 0.0
        if far > 1e-9:
            state.q = state.q + dq * min(1.0, self.speed * dt / far)   # every joint arrives with the farthest
        dg = state.grip_target - state.grip
        state.grip += math.copysign(min(abs(dg), self.grip_speed * dt), dg)
        return pose

    def moving(self, state) -> bool:
        return bool(np.max(np.abs(state.target - state.q)) > 1e-4 or abs(state.grip_target - state.grip) > 1e-4)

    # --- drawing ------------------------------------------------------------------------------------------------------
    def visuals(self) -> dict:
        """Each link's visual geoms, merged per colour into PLY files (in the link's frame)."""
        if not self.drawn:
            return {}
        if getattr(self, "_visuals", None) is None:
            self._visuals = self._build_visuals()
        return self._visuals

    def _build_visuals(self) -> dict:
        from ..bodies import write_ply
        from .robots import Visual
        from .transform import Transform
        m = self._m
        stamp = hashlib.sha1(f"{self.path.resolve()}|{self.path.stat().st_mtime_ns}|1".encode()).hexdigest()[:10]
        folder = CACHE / f"{self.model}-{stamp}"
        folder.mkdir(parents=True, exist_ok=True)
        out = {}
        for b, name in self._bodies:
            groups: dict = {}
            for g in range(m.ngeom):
                if int(m.geom_bodyid[g]) != b or int(m.geom_group[g]) != 2 or int(m.geom_type[g]) != 7:   # visual meshes
                    continue
                mid = int(m.geom_dataid[g])
                rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
                key = tuple(round(float(c), 3) for c in rgba[:3])
                va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
                v = m.mesh_vert[va:va + vn] @ _quat_matrix(m.geom_quat[g]).T + m.geom_pos[g]
                f = m.mesh_face[fa:fa + fn]
                groups.setdefault(key, []).append((v, f))
            visuals = []
            for k, (rgb, parts) in enumerate(sorted(groups.items())):
                off = np.cumsum([0] + [len(v) for v, _ in parts])
                v = np.concatenate([p[0] for p in parts]).astype(np.float32)
                f = np.concatenate([p[1] + o for p, o in zip(parts, off)]).astype(np.int32)
                path = folder / f"{name}_{k}.ply"
                if not path.is_file():
                    write_ply(path, v, f)
                bright = max(rgb)
                kind = "plastic" if bright > 0.3 else "rubber"
                visuals.append(Visual("mesh", (), Transform(), (kind, tuple(min(0.95, c) for c in rgb), 0.35 if bright > 0.3 else 0.6),
                                      str(path)))
            if visuals:
                out[name] = visuals
        return out

    def summary(self) -> dict:
        return {"model": self.label, "joints": len(self.home), "reach_m": self.info["reach_m"], "payload_kg": self.info["payload_kg"]}
