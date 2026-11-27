#!/usr/bin/env python3
"""Driving test on rough ground: the Ranger Mini on its suspension over a dirt strip, a speed bump and a door
threshold, in an outdoor scene (blend_REF_PH45: a clinic's parking lot).

    python3.10 tools/rough_drive_test.py --plots out.png                 # simulate at 0.5, 1.0, 1.5 m/s: plots and numbers
    python3.10 tools/rough_drive_test.py --video out.mp4 --speed 1.0     # and render the run (the render server)

The course lies along +x on the lot's free lane (y -509, x 492.5 to 506.5), laid on the asphalt (heights added to the
scene's ground map; drawn as bodies):

* a dirt strip (x 493.5 to 499.5, 1.6 m wide): a mound 4 cm thick with ruts where the wheels run, potholes down to the
  asphalt, lumps and stones (seeded noise);
* a rubber speed bump (x 501.5): 6 cm high, 40 cm across, a parabola;
* a door threshold (x 503.5): 2.5 cm high, its top 6 cm, its ramps 2 cm (steep: what a wheel feels as a step).

The robot is configs/ranger_mini with the rig hdr_12bit_mast_ouster (the mast and its cameras 0.8 m up: what shakes),
its body on hw.suspension's springs (3 Hz, damping ratio 0.15: estimates, lightly damped), its wheels' ground the rim's
(hw.suspension.rim_heights), what it climbs up to 8 cm (Ground.step; the Ranger's clearance is 10.7 cm). Logged per tick
(100 Hz): pose, pitch, roll, the body's vertical acceleration, wheels off the ground, the HDR camera's pose (its angular
rate and how far it sways from its path).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opticalnav_sim import bodies, hw, tonemap  # noqa: E402
from opticalnav_sim.hw import Transform, config  # noqa: E402
from opticalnav_sim.hw.actor import Spec  # noqa: E402
from opticalnav_sim.hw.drive import Twist  # noqa: E402
from opticalnav_sim.hw.ground import Ground  # noqa: E402
from opticalnav_sim.hw.robots import Visual  # noqa: E402
from opticalnav_sim.hw.transform import camera_to_world  # noqa: E402

SCENE = "blend_REF_PH45"
LANE_Y, X0, X1 = -509.0, 492.5, 506.5
LOT_Z = 6.0
CELL = 0.01
FEATURES = {"dirt": (493.5, 499.5), "speed bump": (501.3, 501.7), "threshold": (503.45, 503.55)}
CACHE = Path.home() / ".cache" / "opticalnav_sim" / "course"


def course(seed: int = 3) -> dict:
    """The course's heights over the asphalt (m, on a CELL grid): each feature apart, for drawing, and their sum."""
    rng = np.random.default_rng(seed)
    xs = np.arange(X0 - 1.5, X1 + 1.5 + CELL / 2, CELL)      # the map around the course too: the robot's footprint
    ys = np.arange(LANE_Y - 1.4, LANE_Y + 1.4 + CELL / 2, CELL)
    X, Y = np.meshgrid(xs, ys)
    out = {"x": xs, "y": ys}
    # the dirt strip: a 4 cm mound, ramped over 0.4 m at its ends and 0.15 m at its sides
    a, b = FEATURES["dirt"]
    edge = np.clip(np.minimum(X - a, b - X) / 0.4, 0, 1) * np.clip((0.8 - np.abs(Y - LANE_Y)) / 0.15, 0, 1)
    dirt = 0.04 * edge
    # lumps: smooth noise, a few cm
    for _ in range(90):
        cx, cy, r, h = rng.uniform(a, b), LANE_Y + rng.uniform(-0.75, 0.75), rng.uniform(0.08, 0.35), rng.uniform(-0.015, 0.025)
        dirt += h * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * r * r)) * edge
    # ruts where the wheels run (the track is 0.364 m): 1 to 3 cm deep, wandering
    for side in (-0.182, 0.182):
        wander = 0.04 * np.sin((X - a) * 1.7 + side * 9) + 0.02 * np.sin((X - a) * 4.3)
        depth = 0.01 + 0.02 * (0.5 + 0.5 * np.sin((X - a) * 2.3 + side * 5))
        dirt -= depth * np.exp(-((Y - LANE_Y - side - wander) ** 2) / (2 * 0.05 ** 2)) * edge
    # potholes down to the asphalt, and stones
    for cx in (494.6, 496.3, 497.9, 498.8):
        cy, r = LANE_Y + rng.choice([-0.18, 0.18]) + rng.uniform(-0.05, 0.05), rng.uniform(0.12, 0.2)
        dirt = np.minimum(dirt, np.where((X - cx) ** 2 + (Y - cy) ** 2 < r * r, 0.0 + 0.6 * ((X - cx) ** 2 + (Y - cy) ** 2) / r, 1.0))
    for _ in range(40):
        cx, cy, r = rng.uniform(a + 0.3, b - 0.3), LANE_Y + rng.uniform(-0.6, 0.6), rng.uniform(0.02, 0.045)
        dirt = np.maximum(dirt, (0.04 + r) * np.clip(1 - ((X - cx) ** 2 + (Y - cy) ** 2) / (r * r), 0, 1) ** 0.5 * edge)
    out["dirt"] = np.clip(dirt, 0.0, None) * (edge > 0)
    # the speed bump: a parabola 6 cm high, 0.4 m across, 2.4 m long
    c = (FEATURES["speed bump"][0] + FEATURES["speed bump"][1]) / 2
    out["speed bump"] = 0.06 * np.clip(1 - ((X - c) / 0.2) ** 2, 0, None) * (np.abs(Y - LANE_Y) < 1.2)
    # the threshold: 2.5 cm, a flat top 6 cm, ramps 2 cm
    c = (FEATURES["threshold"][0] + FEATURES["threshold"][1]) / 2
    out["threshold"] = 0.025 * np.clip((0.05 - np.abs(X - c)) / 0.02, 0, 1) * (np.abs(Y - LANE_Y) < 1.2)
    out["total"] = out["dirt"] + out["speed bump"] + out["threshold"]
    return out


def mesh_of(heights: np.ndarray, xs: np.ndarray, ys: np.ndarray, floor: float = -0.005, every: int = 2):
    """A height field's top surface (where it stands over the asphalt) and its skirt down to ``floor``, as a mesh."""
    h = heights[::every, ::every]
    x, y = xs[::every], ys[::every]
    ny, nx = h.shape
    on = h > 0.001
    idx = -np.ones((ny, nx), np.int64)
    verts = []
    for i in range(ny):
        for j in range(nx):
            if on[max(0, i - 1):i + 2, max(0, j - 1):j + 2].any():
                idx[i, j] = len(verts)
                verts.append((x[j], y[i], h[i, j] if on[i, j] else floor))
    faces = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            q = idx[i, j], idx[i, j + 1], idx[i + 1, j + 1], idx[i + 1, j]
            if min(q) >= 0:
                faces += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
    return np.array(verts, np.float32), np.array(faces, np.int32)


@dataclass
class Course(Spec):
    """The course drawn: a body at the world origin (its heights over the lot)."""
    kind = "structure"
    meshes: tuple = ()
    model: str = "rough_course"

    def link_frames(self, state) -> dict:
        return {"course": np.eye(4)}

    def visuals(self) -> dict:
        return {"course": list(self.meshes)}


def course_spec(c: dict) -> Course:
    CACHE.mkdir(parents=True, exist_ok=True)
    looks = {"dirt": ("rubber", (0.30, 0.22, 0.15), 0.95), "speed bump": ("rubber", (0.85, 0.66, 0.08), 0.6),
             "threshold": ("aluminium", (0.75, 0.75, 0.77), 0.35)}
    out = []
    for name, look in looks.items():
        v, f = mesh_of(c[name], c["x"], c["y"])
        v[:, 2] += LOT_Z
        path = CACHE / f"{name.replace(' ', '_')}.ply"
        bodies.write_ply(path, v, f)
        out.append(Visual("mesh", (), Transform(), look, str(path)))
    return Course(meshes=tuple(out))


def ground_of(scene_dir: Path, c: dict) -> Ground:
    """The scene's ground map with the course's heights added, at the course's resolution there."""
    z = np.load(scene_dir / "map" / "ground_1.3m.npz", allow_pickle=True)
    info = json.loads(str(z["info"]))
    h0, (ox, oy), cell = z["heights"].astype(np.float64), info["origin"], float(info["cell"])
    # a fine map over the course (1 cm), the scene's 5 cm one elsewhere: resample the scene's onto 1 cm over the lane
    xs, ys = c["x"], c["y"]
    j = np.clip(np.rint((xs - ox) / cell).astype(int), 0, h0.shape[1] - 1)
    i = np.clip(np.rint((ys - oy) / cell).astype(int), 0, h0.shape[0] - 1)
    base = h0[np.ix_(i, j)]
    base = np.where(np.isnan(base), LOT_Z, base)
    g = Ground(step=0.08)
    g.set_height_map(base + c["total"], (xs[0], ys[0]), CELL)
    return g


def run(speed: float, server: str | None, dt: float = 0.01, video: bool = False):
    """Drive the course at ``speed``: the world, and the log (one row a tick)."""
    world = hw.SimWorld(server, scene=SCENE) if server else hw.World(dt=dt)
    world.dt = dt
    c = course()
    scene_dir = Path(world.meta["scene_dir"]) if server else Path("packs/blend-cache/scenes") / SCENE
    world.set_ground(ground_of(scene_dir, c))
    robot = config.spawn(world, "ranger_mini", at=Transform((X0 + 0.2, LANE_Y, LOT_Z), (0.0, 0.0, 0.0)),
                         rig="hdr_12bit_mast_ouster")
    base = robot.base
    base.spec.suspension_hz, base.spec.suspension_zeta = 3.0, 0.15
    if video:
        world.spawn(course_spec(c), at=Transform(), name="course")
    cam = robot.sensors["hdr"]
    return world, robot, cam, c


def simulate(speed: float, server=None, dt: float = 0.01) -> np.ndarray:
    world, robot, cam, _ = run(speed, server, dt)
    base = robot.base
    rows = []
    while base.pose.xyz[0] < X1 - 0.6:
        base.spec.apply(base.state, Twist(speed, 0.0, 0.0))
        world.tick()
        m = cam.link_matrix("camera")
        b = base.state.body
        rows.append([world.time, base.pose.xyz[0], base.pose.xyz[2], base.pose.rpy_deg[0], base.pose.rpy_deg[1],
                     0.0 if b is None else b.accel, 0 if b is None else b.lifted, *m[:3, 3], *m[:3, 0]])
        if world.time > 60:
            break
    return np.array(rows)


def summary(log: np.ndarray, speed: float) -> dict:
    t, x, z, roll, pitch, acc, lifted = (log[:, i] for i in range(7))
    cam, fwd = log[:, 7:10], log[:, 10:13]
    rate = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", fwd[1:], fwd[:-1]), -1, 1))) / np.diff(t)
    # how far the camera sways from its own path: its height against that height smoothed over half a second
    pad = np.pad(cam[:, 2], 25, mode="edge")
    sway = cam[:, 2] - np.convolve(pad, np.ones(51) / 51, mode="valid")
    out = {"speed_m_s": speed}
    for name, (a, b) in {"flat": (X0, 493.3), **FEATURES}.items():
        sel = (x > a - 0.5) & (x < b + 1.0)
        if name == "flat":
            sel = x < 493.2
        if not sel.any():
            continue
        out[name] = {"pitch_deg": [round(float(pitch[sel].min()), 1), round(float(pitch[sel].max()), 1)],
                     "roll_deg": [round(float(roll[sel].min()), 1), round(float(roll[sel].max()), 1)],
                     "accel_peak_m_s2": round(float(np.abs(acc[sel]).max()), 1),
                     "camera_rate_peak_deg_s": round(float(rate[sel[1:]].max()), 0),
                     "camera_sway_cm": round(float(np.abs(sway[sel]).max()) * 100, 1),
                     "wheels_off_ground": int(lifted[sel].max())}
    return out


def plots(logs: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for speed, log in logs.items():
        x = log[:, 1]
        cam, fwd, t = log[:, 7:10], log[:, 10:13], log[:, 0]
        rate = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", fwd[1:], fwd[:-1]), -1, 1))) / np.diff(t)
        axes[0].plot(x, log[:, 4], label=f"{speed} m/s")
        axes[1].plot(x, log[:, 3], label=f"{speed} m/s")
        axes[2].plot(x, log[:, 5], label=f"{speed} m/s", lw=0.8)
        axes[3].plot(x[1:], rate, label=f"{speed} m/s", lw=0.8)
    for ax, title in zip(axes, ("body pitch (deg)", "body roll (deg)", "body vertical acceleration (m/s^2)",
                                "HDR camera angular rate (deg/s)")):
        ax.set_ylabel(title, fontsize=9)
        ax.grid(alpha=0.3)
        for name, (a, b) in FEATURES.items():
            ax.axvspan(a, b, color={"dirt": "#c8a878", "speed bump": "#f0c020", "threshold": "#a0a0a8"}[name], alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("x along the course (m): dirt strip 493.5-499.5, speed bump 501.5, threshold 503.5")
    fig.suptitle("Ranger Mini 2.0 on its suspension (3 Hz, damping 0.15): blend_REF_PH45 test course")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def video(server: str, speed: float, out: str, fps: float = 15.0, spp: int = 64, passes: int = 4, size=(960, 540)) -> None:
    """The run rendered: a camera following beside the robot, and the HDR camera's view (a pinhole at its pose; no
    sensor model), each pass averaged with fireflies held back (no denoiser), and the pitch over the course so far."""
    from PIL import Image, ImageDraw, ImageFont
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from arm_pick_demo import held_mean, look_at
    world, robot, cam, c = run(speed, server, 0.01, video=True)
    base = robot.base
    W, H = size
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    folder = Path(tempfile.mkdtemp(prefix="rough_drive_"))
    tone, k, rows = None, 0, []
    per_frame = max(1, int(round(1.0 / fps / world.dt)))
    while base.pose.xyz[0] < X1 - 0.6:
        for _ in range(per_frame):
            base.spec.apply(base.state, Twist(speed, 0.0, 0.0))
            world.tick()
            rows.append((base.pose.xyz[0], base.pose.rpy_deg[1], base.pose.rpy_deg[0]))
        t0 = time.time()
        p = base.pose.xyz
        chase = look_at((p[0] - 1.6, p[1] - 2.3, LOT_Z + 1.3), (p[0] + 0.3, p[1], LOT_Z + 0.35))
        views = [(camera_to_world(chase), W, H, 62.0), (camera_to_world(cam.link_matrix("camera")), W // 2, H // 2, 75.0)]
        imgs = []
        for c2w, w, h, fov in views:
            v = {"scan": SCENE, "variant": "base", "mode": "rgb", "spp": spp, "denoise": False, "preview": False,
                 "camera_to_world": c2w, "width": w, "height": h, "hfov_deg": fov, "bodies": world.bodies()}
            rad = held_mean(np.stack([world.client.render([dict(v, seed=k * 31 + i)], "float32")[0]["radiance"]
                                      for i in range(passes)]))
            if tone is None:
                tone = tonemap.global_params([rad])
            imgs.append(Image.fromarray(tonemap.reinhard(rad, tone["exposure"], tone["white"])))
        canvas = Image.new("RGB", (W + W // 2, H), (12, 14, 17))
        canvas.paste(imgs[0], (0, 0))
        canvas.paste(imgs[1], (W, 0))
        # the pitch and roll so far, under the camera's view
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(W / 2 / 100, H / 2 / 100), dpi=100)
        r = np.array(rows)
        ax.plot(r[:, 0], r[:, 1], label="pitch", lw=1.2)
        ax.plot(r[:, 0], r[:, 2], label="roll", lw=1.2)
        for name, (a, b) in FEATURES.items():
            ax.axvspan(a, b, color={"dirt": "#c8a878", "speed bump": "#f0c020", "threshold": "#a0a0a8"}[name], alpha=0.3)
        ax.set_xlim(X0, X1)
        ax.set_ylim(-12, 12)
        ax.set_ylabel("deg", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.canvas.draw()
        canvas.paste(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]), (W, H // 2))
        plt.close(fig)
        d = ImageDraw.Draw(canvas)
        where = next((n for n, (a, b) in FEATURES.items() if a - 0.4 < p[0] < b + 0.6), "asphalt")
        d.rectangle((0, 0, W, 34), fill=(0, 0, 0))
        d.text((10, 6), f"Ranger Mini 2.0 at {speed:.1f} m/s  ·  {where}  ·  pitch {base.pose.rpy_deg[1]:+.1f}°  roll {base.pose.rpy_deg[0]:+.1f}°",
               fill=(240, 240, 240), font=font)
        d.rectangle((W, 0, W + 220, 22), fill=(0, 0, 0))
        d.text((W + 8, 3), "HDR camera (on the mast)", fill=(230, 230, 230), font=small)
        canvas.save(folder / f"{k:04d}.png")
        print(f"[rough] frame {k} x {p[0]:.2f} {time.time() - t0:.1f} s", flush=True)
        k += 1
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", str(folder / "%04d.png"), "-c:v", "libx264",
                    "-crf", "21", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)
    print(f"[rough] wrote {out} ({k} frames in {folder})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--server", default="http://127.0.0.1:18770")
    ap.add_argument("--plots", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--passes", type=int, default=4)
    args = ap.parse_args()
    if args.plots or args.json:
        logs = {s: simulate(s) for s in (0.5, 1.0, 1.5)}
        result = [summary(log, s) for s, log in logs.items()]
        print(json.dumps(result, indent=1))
        if args.json:
            Path(args.json).write_text(json.dumps(result, indent=1))
        if args.plots:
            plots(logs, args.plots)
    if args.video:
        video(args.server, args.speed, args.video, passes=args.passes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
