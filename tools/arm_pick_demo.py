#!/usr/bin/env python3
"""An arm on the Ranger Mini picks a binder off a shelf: the scenario, rendered by the render server into an MP4.

    python3.10 tools/arm_pick_demo.py --arm piper --out piper_pick.mp4      # or --arm panda; --dry-run: no rendering

The robot (configs/ranger_mini with the rig mast_ouster_<arm>: the mast, its cameras and an Ouster OS1-128, the arm on
the deck in front) faces the north rack of blend_REF_CS02. The target is a binder standing at the end of a row on the
rack's bottom shelf (physics.json's CS02_r003_shelf_manuals_binder.005: 2 cm thick, its top 2 cm over its neighbours',
4 cm under the shelf above). What happens, phase by phase (``plan``):

1. the robot drives up to the rack (the arm at home);
2. the arm swings its open gripper in front of the binder's top edge, pointing into the shelf (joint-space move to the
   inverse kinematics' answer);
3. it goes straight in, 4 cm (the tool centre point on a line: inverse kinematics each frame, from the last answer);
4. the gripper closes on the binder's 2.6 cm; from here the binder is held (hw.physics.PushWorld.hold: it goes where the
   tool centre point takes it, and collides with nothing);
5. the arm lifts it 1.5 cm (the shelf above leaves 4);
6. the robot backs off 0.45 m: the binder slides out of the shelf;
7. the arm raises it to show it.

Each frame: the scene path-traced from a camera beside the robot (``--passes`` passes of ``--spp`` samples averaged with
fireflies held back, no denoiser; ``--denoise on``: one pass, OptiX-denoised; the first frame's tone for all), and
the Ouster's scan (hw.ouster: its noise model) drawn as its 3D point view, beside it. What is not simulated: forces, the
arm's dynamics and collisions (the fingers can touch the neighbouring binder), slip in the grasp.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opticalnav_sim import hw, stokes, tonemap  # noqa: E402
from opticalnav_sim.hw import Transform, config  # noqa: E402
from opticalnav_sim.hw.physics import PushWorld  # noqa: E402
from opticalnav_sim.hw.transform import camera_to_world  # noqa: E402

TARGET = "CS02_r003_shelf_manuals_binder.005"
REACH = {"piper": 0.60, "panda": 0.42}
# the cameras the scenario can be seen from: (eye, aim), each relative to the grasp point (world, m)
CAMERAS = {"side": ((1.2, -1.6, 1.0), (-0.1, -0.7, -0.15)), "left": ((-1.5, -1.7, 0.9), (0.1, -0.6, -0.15)),
           "close": ((0.75, -0.95, 0.45), (-0.05, -0.35, -0.05))}     # the grasp's distance ahead of the arm's base (inside each arm's horizontal reach)
ARM_MOUNT_X = 0.26                         # the arm on the deck, ahead of the robot's centre (the rig's mount)


def look_at(eye, target) -> np.ndarray:
    """A camera pose (x forward, z up) at ``eye`` looking at ``target``."""
    eye, f = np.asarray(eye, float), np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    left = np.cross([0.0, 0.0, 1.0], f)
    left /= np.linalg.norm(left)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = f, left, np.cross(f, left), eye
    return m


def held_mean(passes: np.ndarray) -> np.ndarray:
    """The mean of a pixel's passes (P x H x W x 3) with fireflies held back, as the viewer does: where a pixel's
    brightest pass is over four times the median of its passes, that pass is left out."""
    lum = passes @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    top = np.argmax(lum, axis=0)
    med = np.median(lum, axis=0)
    fire = np.take_along_axis(lum, top[None], 0)[0] > 4.0 * np.maximum(med, 1e-6)
    total = passes.sum(0)
    drop = np.take_along_axis(passes, top[None, ..., None].repeat(3, -1), 0)[0]
    return np.where(fire[..., None], (total - drop) / (len(passes) - 1), total / len(passes))


def held_stokes(passes: list) -> tuple:
    """S0, S1, S2 (H x W x 3 each) from passes of them with fireflies held back as in ``held_mean``: the pass left out
    of a pixel (by its S0) is left out of all three."""
    s0 = np.stack([p["s0"] for p in passes])
    lum = s0 @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    top = np.argmax(lum, axis=0)
    fire = np.take_along_axis(lum, top[None], 0)[0] > 4.0 * np.maximum(np.median(lum, axis=0), 1e-6)
    out = []
    for key in ("s0", "s1", "s2"):
        a = np.stack([p[key] for p in passes])
        total = a.sum(0)
        drop = np.take_along_axis(a, top[None, ..., None].repeat(3, -1), 0)[0]
        out.append(np.where(fire[..., None], (total - drop) / (len(passes) - 1), total / len(passes)))
    return tuple(out)


def smooth(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


class Scenario:
    def __init__(self, arm: str, server: str, scene: str = "blend_REF_CS02"):
        self.world = hw.SimWorld(server, scene=scene)
        self.scene_dir = Path(self.world.meta["scene_dir"])
        physics = json.loads((self.scene_dir / "physics.json").read_text())
        self.obj = next(o for o in physics["objects"] if o["id"] == TARGET)
        lo, hi = (np.array(v, float) for v in self.obj["bbox"])
        self.thick = float(hi[0] - lo[0])
        # the grasp: in the middle of the binder's thickness, 2.5 cm into its front, 1.2 cm under its top
        self.grasp_p = np.array([(lo[0] + hi[0]) / 2.0, lo[1] + 0.025, hi[2] - 0.012])
        self.floor = float(self.obj["support"].get("floor_z") or 0.18) if self.obj.get("support") else 0.18
        self.robot = config.spawn(self.world, "ranger_mini", rig=f"mast_ouster_{arm}")
        self.arm = self.robot.parts["arm"]
        self.spec = self.arm.spec
        # the robot faces the rack (+y), the arm's base REACH behind the grasp
        self.yaw = 90.0
        self.stand = np.array([self.grasp_p[0], self.grasp_p[1] - REACH[arm] - ARM_MOUNT_X])
        self.set_robot(self.stand[1] - 1.0)
        base = self.robot.base.spec
        self.push = PushWorld.from_scene(self.scene_dir, np.asarray(base.footprint())[:, :2],
                                         float(getattr(base, "size", (0, 0, 0.3))[2]) + 0.1)
        self.rel = None    # the binder's frame in the tool centre point's, once held

    # --- poses ----------------------------------------------------------------------------------------------------------
    def set_robot(self, y: float) -> None:
        self.robot.base.pose = Transform((float(self.stand[0]), float(y), self.floor), (0.0, 0.0, self.yaw))

    def grasp_world(self, dy: float = 0.0, dz: float = 0.0) -> np.ndarray:
        """The tool centre point's pose for the grasp (world): pointing into the shelf (+y), the fingers closing across the
        binder (along x)."""
        close = self.spec.close_axis          # in the TCP frame
        z = np.array([0.0, 1.0, 0.0])
        rot_tcp = np.c_[np.cross(close, [0.0, 0.0, 1.0]), close, [0.0, 0.0, 1.0]]   # TCP axes: (close x z, close, z)
        rot_world = np.c_[np.cross([1.0, 0.0, 0.0], z), [1.0, 0.0, 0.0], z]          # those axes wanted in the world
        m = np.eye(4)
        m[:3, :3] = rot_world @ rot_tcp.T
        m[:3, 3] = self.grasp_p + [0.0, dy, dz]
        return m

    def in_arm(self, world_pose: np.ndarray) -> np.ndarray:
        return np.linalg.inv(self.arm.world_matrix()) @ world_pose

    def tcp_world(self) -> np.ndarray:
        return self.arm.world_matrix() @ self.spec.tcp(self.arm.state)

    def ik(self, world_pose, q0=None) -> np.ndarray:
        return self.spec.ik(self.arm.state, self.in_arm(world_pose), q0=q0)

    def follow(self, world_pose, q, phase: str = "", rot_weight: float = 0.5, rest=None) -> np.ndarray:
        """IK along a path: from ``q``; where the path leaves what the arm reaches, it stays where it got to."""
        # only from where the arm is (another branch of the solutions would be a jump): a local solve
        q_new, dp, da = self.spec._solve(np.array(q, dtype=np.float64), self.in_arm(world_pose), self.arm.state.grip,
                                         300, rot_weight, 1e-4, rest=q if rest is None else rest)
        if dp < 0.003 and (da < 2.0 or rot_weight < 0.5):
            return q_new
        if phase not in self.__dict__.setdefault("_warned", set()):
            print(f"[pick] {phase}: the arm does not reach on ({dp * 1000:.0f} mm, {da:.1f} degrees off): it stops there", flush=True)
            self._warned.add(phase)
        return q

    # --- the plan -------------------------------------------------------------------------------------------------------
    def plan(self, fps: float) -> list:
        """Each frame's (phase label, robot y, arm joint angles, gripper opening, held?)."""
        st, spec = self.arm.state, self.spec
        frames = []
        grip_closed = min(1.0, (self.thick - 0.002) / (2.0 * spec.info["finger_open"]))   # a little squeeze
        home = spec.travel.copy()          # folded while it drives
        y_stand = float(self.stand[1])

        def hold(label, seconds, y, q, grip, held):
            for _ in range(int(round(seconds * fps))):
                frames.append((label, y, q.copy(), grip, held))

        # 1. drive up
        n = int(2.5 * fps)
        for i in range(n):
            frames.append(("1 drive to the rack", y_stand - 1.0 + smooth((i + 1) / n), home.copy(), 1.0, False))
        hold("1 drive to the rack", 0.4, y_stand, home, 1.0, False)
        self.set_robot(y_stand)
        st.q, st.grip = home.copy(), 1.0
        # 2. swing to the pre-grasp, 4 cm before the binder
        q_pre = self.ik(self.grasp_world(dy=-0.06), q0=home)
        n = int(2.5 * fps)
        for i in range(n):
            frames.append(("2 swing in front of the binder", y_stand, home + (q_pre - home) * smooth((i + 1) / n), 1.0, False))
        # 3. straight in
        n, q = int(1.5 * fps), q_pre
        for i in range(n):
            q = self.follow(self.grasp_world(dy=-0.06 * (1.0 - smooth((i + 1) / n))), q, "in")
            frames.append(("3 in to the binder", y_stand, q, 1.0, False))
        q_grasp = q
        # 4. close
        n = int(0.8 * fps)
        for i in range(n):
            frames.append(("4 close the gripper", y_stand, q_grasp, 1.0 + (grip_closed - 1.0) * smooth((i + 1) / n), False))
        hold("4 close the gripper", 0.3, y_stand, q_grasp, grip_closed, True)
        # 5. lift 1.5 cm
        n, q = int(0.8 * fps), q_grasp
        for i in range(n):
            q = self.follow(self.grasp_world(dz=0.015 * smooth((i + 1) / n)), q, "lift")
            frames.append(("5 lift it off the shelf", y_stand, q, grip_closed, True))
        q_lift = q
        # 6. back off: the binder comes out
        n = int(2.5 * fps)
        for i in range(n):
            frames.append(("6 back off: it slides out", y_stand - 0.45 * smooth((i + 1) / n), q_lift, grip_closed, True))
        y_back = y_stand - 0.45
        # 7. raise it and tip it up toward the camera: a pose 15 cm higher, the gripper pitched up 34 degrees, reached by a
        # joint-space move (clear of the shelf now)
        self.set_robot(y_back)
        st.q = q_lift
        lifted = self.in_arm(self.tcp_world())
        q_show = None
        for tilt in (-0.6, 0.6, -0.3, 0.3, 0.0):   # the tilt each arm's wrist allows there (a pitch about the arm's y)
            show = lifted.copy()
            show[2, 3] += 0.15
            c, s_ = math.cos(tilt), math.sin(tilt)
            show[:3, :3] = np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]]) @ show[:3, :3]
            try:
                q_show = spec.ik(st, show, q0=q_lift, rot_weight=0.3)
                break
            except ValueError:
                continue
        if q_show is None:
            raise SystemExit(f"{spec.label} cannot raise the binder from there")
        n = int(2.5 * fps)
        for i in range(n):
            frames.append(("7 raise it", y_back, q_lift + (q_show - q_lift) * smooth((i + 1) / n), grip_closed, True))
        hold("7 raise it", 1.2, y_back, q_show, grip_closed, True)
        return frames

    # --- a frame --------------------------------------------------------------------------------------------------------
    def apply(self, frame, dt: float) -> None:
        label, y, q, grip, held = frame
        self.set_robot(y)
        st = self.arm.state
        st.q, st.target, st.grip, st.grip_target = q.copy(), q.copy(), grip, grip
        if held:
            tcp = self.tcp_world()
            if self.rel is None:   # the moment it closes: the binder's frame in the gripper's
                self.rel = np.linalg.inv(tcp) @ self.push.world_poses()[TARGET]
                st.holding = TARGET
            self.push.hold(TARGET, tcp @ self.rel)
        self.push.step(self.robot.base.pose, dt)
        # every movable object, moved or not: a frozen render records again when the set of objects it is given changes
        self.world.objects_view = {oid: np.asarray(m).tolist() for oid, m in self.push.poses().items()}
        self.world.time += dt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", choices=("piper", "panda"), default="piper")
    ap.add_argument("--server", default="http://127.0.0.1:18770")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--spp", type=int, default=64, help="samples a pass")
    ap.add_argument("--denoise", choices=("on", "off"), default="off",
                    help="off: --passes passes of --spp each, averaged with fireflies held back (no denoiser blotches)")
    ap.add_argument("--passes", type=int, default=8)
    ap.add_argument("--polar", action="store_true",
                    help="polarized: the frame shows S0, DoLP and AoLP (the viewer's colours) and the Ouster's 3D view, "
                         "each at half the --size")
    ap.add_argument("--lidar-every", type=int, default=1, help="scan the Ouster every this many frames")
    ap.add_argument("--camera", choices=sorted(CAMERAS), default="side")
    ap.add_argument("--dry-run", action="store_true", help="plan only: positions and joint angles, nothing rendered")
    ap.add_argument("--frames", default=None, help="render only these frames (a:b)")
    args = ap.parse_args()
    sc = Scenario(args.arm, args.server)
    t0 = time.time()
    frames = sc.plan(args.fps)
    print(f"[pick] {sc.spec.label}: {len(frames)} frames planned in {time.time() - t0:.2f} s; grasp at {sc.grasp_p.round(3)}, "
          f"robot at y {sc.stand[1]:.3f} (front {sc.stand[1] + 0.369:.3f}, the shelf's front -7.62)", flush=True)
    if args.dry_run:
        for k in (0, len(frames) // 2, len(frames) - 1):
            print(k, frames[k][0], np.round(frames[k][2], 3))
        return 0
    from PIL import Image, ImageDraw, ImageFont
    W, H = (int(v) for v in args.size.split("x"))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    g = sc.grasp_p
    eye, aim = CAMERAS[args.camera]
    cam = look_at(g + np.array(eye), g + np.array(aim))
    try:
        tone = tonemap.scene_params(sc.scene_dir)
    except (ValueError, OSError):
        tone = None    # a .blend scene has no reference frames: the first frame sets the exposure for all
    ouster = sc.robot.sensors["ouster"]
    out_dir = Path(tempfile.mkdtemp(prefix="arm_pick_"))
    rng = range(len(frames))
    if args.frames:
        a, b = (int(v) for v in args.frames.split(":"))
        rng = range(a, b)
    lidar_img = None
    dt = 1.0 / args.fps
    for k in rng:
        t1 = time.time()
        sc.apply(frames[k], dt)
        view = {"scan": sc.world.scene, "variant": "base", "mode": "rgb", "seed": k, "spp": args.spp,
                "denoise": args.denoise == "on", "preview": False, "camera_to_world": camera_to_world(cam), "width": W,
                "height": H, "hfov_deg": 66.0, "bodies": sc.world.bodies()}
        if sc.world.objects_view:
            view["objects"] = sc.world.objects_view
        if args.polar:
            view.update(mode="polar", width=W // 2, height=H // 2, s3=False, bands="rgb")
            if args.denoise == "on":
                o = sc.world.client.render([view], "float32")[0]
                s0, s1, s2 = o["s0"], o["s1"], o["s2"]
            else:
                s0, s1, s2 = held_stokes([sc.world.client.render([dict(view, seed=k * 1009 + p)], "float32")[0]
                                          for p in range(args.passes)])
            rad = s0
        elif args.denoise == "on":
            rad = sc.world.client.render([view], "float32")[0]["radiance"]
        else:
            passes = [sc.world.client.render([dict(view, seed=k * 1009 + p)], "float32")[0]["radiance"]
                      for p in range(args.passes)]
            rad = held_mean(np.stack(passes))
        if tone is None:
            tone = tonemap.global_params([rad])
        img = tonemap.reinhard(rad, tone["exposure"], tone["white"])
        if lidar_img is None or k % args.lidar_every == 0:
            rd = sc.world.sense(ouster)
            lidar_img = rd.picture
        side_w = W // 2
        cloud = Image.fromarray(lidar_img[:576]).resize((side_w, side_w * 9 // 16))
        strips = Image.fromarray(lidar_img[580:]).resize((side_w, max(1, round(lidar_img[580:].shape[0] * side_w / lidar_img.shape[1]))))
        if args.polar:   # S0 | DoLP over AoLP | the Ouster's 3D view
            pol = stokes.derived(s0, s1, s2)
            canvas = Image.new("RGB", (W, H + 40), (12, 14, 17))
            panels = [(Image.fromarray(img), "S0 (intensity)"),
                      (Image.fromarray(stokes.dolp_rgb(pol["dolp"], 0.2)), "DoLP  0 to 0.2 (black to red)"),
                      (Image.fromarray(stokes.aolp_rgb(pol["aolp_deg"], pol["dolp"], 0.2)), "AoLP  hue 0-180 deg, brightness DoLP"),
                      (cloud.resize((W // 2, H // 2)), f"{ouster.spec.model_label}: 3D view (noise model)")]
            dp = ImageDraw.Draw(canvas)
            for i, (im, title) in enumerate(panels):
                x, y = (i % 2) * (W // 2), 40 + (i // 2) * (H // 2)
                canvas.paste(im.resize((W // 2, H // 2)), (x, y))
                dp.rectangle((x, y, x + 9 * len(title) + 16, y + 24), fill=(0, 0, 0))
                dp.text((x + 8, y + 3), title, fill=(235, 235, 235), font=small)
        else:
            canvas = Image.new("RGB", (W + side_w, H), (12, 14, 17))
            canvas.paste(Image.fromarray(img), (0, 0))
            canvas.paste(cloud, (W, 0))
            canvas.paste(strips, (W, cloud.height + 6))
        d = ImageDraw.Draw(canvas)
        label = frames[k][0]
        d.rectangle((0, 0, W, 40), fill=(0, 0, 0))
        d.text((12, 8), f"{sc.spec.label} on Ranger Mini 2.0  ·  {label}", fill=(240, 240, 240), font=font)
        y0 = cloud.height + strips.height + 20
        if args.polar:   # the state, in the header's right half
            d.text((W - 560, 10), f"t {k / args.fps:5.2f} s  gripper {100 * frames[k][3]:.0f}%  "
                                  f"{'holding' if frames[k][4] else ''}", fill=(200, 200, 200), font=small)
            canvas.save(out_dir / f"{k:04d}.png")
            print(f"[pick] frame {k + 1}/{len(frames)} {time.time() - t1:.2f} s  {label}", flush=True)
            continue
        info = [f"{ouster.spec.model_label} (noise model): 3D view, then range / reflectivity",
                f"t = {k / args.fps:5.2f} s   gripper {100 * frames[k][3]:.0f}% open   {'holding the binder' if frames[k][4] else ''}",
                "joints (deg): " + "  ".join(f"{math.degrees(v):.0f}" for v in frames[k][2])]
        for i, line in enumerate(info):
            d.text((W + 12, y0 + 24 * i), line, fill=(200, 200, 200), font=small)
        canvas.save(out_dir / f"{k:04d}.png")
        print(f"[pick] frame {k + 1}/{len(frames)} {time.time() - t1:.2f} s  {label}", flush=True)
    if args.out:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(args.fps), "-start_number", str(rng.start),
                        "-i", str(out_dir / "%04d.png"), "-c:v", "libx264", "-crf", "22", "-preset", "slow",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out], check=True)
        print(f"[pick] wrote {args.out} (frames in {out_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
