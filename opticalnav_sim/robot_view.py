"""The viewer's robot mode: drive a robot with its camera rig through a scene, see what its cameras deliver.

Served by the viewer (gui.py) beside its own pages: ``/robot`` is the page, ``/api/robot/*`` its data. One
``RobotSession`` holds a hw.SimWorld (the Ranger Mini with wheel dynamics, a mast, the rig's cameras, a chase camera),
steps it in real time and renders its cameras in turn through the render server:

    GET  /api/robot/state                  pose, velocity, odometry, slip and friction per wheel, cameras, messages
    GET  /api/robot/frame/<camera>.jpg     that camera's latest image (through its sensor model and ISP)
    GET  /api/robot/map.png                the scene's ground map: friction (red slippery .. green grippy), obstacles
    POST /api/robot/start    {"scene", "x", "y", "yaw_deg"}     (x, y optional: the scene's first start pose)
    POST /api/robot/control  {"mode", "linear", "steering_deg", "angular"}    held for 0.5 s (the chassis's own timeout)
    POST /api/robot/stop
"""
from __future__ import annotations

import io
import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np

CONTROL_TIMEOUT_S = 0.5      # the Ranger's CAN command timeout: no command for this long stops it
PAGE = Path(__file__).with_name("robot.html")


def _jpeg(image: np.ndarray, quality: int = 82) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


class RobotSession:
    def __init__(self, server_url: str | None = None, calibration: str | None = None, client=None):
        self.server_url, self.client = server_url, client
        self.calibration = calibration or os.environ.get("OPTICALNAV_RIG_CALIBRATION")
        self.lock = threading.RLock()
        self.world = None
        self.robot = None
        self.cameras: dict = {}          # name -> actor
        self.frames: dict = {}           # name -> {"jpeg", "seq", "spp", "exposure_ms", "gain_db", "size", "seconds"}
        self.message, self.error = "", ""
        self.scene = ""
        self.running = False
        self.thread = None
        self.control = None              # (hw control, time it arrived)
        self.trail: list = []            # [x, y] of the robot
        self.odo_trail: list = []        # [x, y] as its wheels count it
        self.spawn = None                # the pose odometry starts from
        self.map_png, self.map_info = None, None
        self.sim_rate = 0.0
        self.moved_at = 0.0             # when the robot last moved (a camera standing still gets more samples)

    # --- lifecycle ------------------------------------------------------------------------------------------------
    def start(self, scene: str, x: float | None = None, y: float | None = None, yaw_deg: float | None = None) -> dict:
        self.stop()
        with self.lock:
            self.scene, self.message, self.error = scene, "starting", ""
            self.frames, self.trail, self.odo_trail, self.map_png, self.map_info = {}, [], [], None, None
            self.running = True
        self.thread = threading.Thread(target=self._run, args=(scene, x, y, yaw_deg), daemon=True)
        self.thread.start()
        return self.state()

    def stop(self) -> None:
        self.running = False
        thread, self.thread = self.thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=30.0)

    def _say(self, text: str) -> None:
        with self.lock:
            self.message = text

    def _build(self, scene: str, x, y, yaw_deg) -> None:
        from . import hw
        from .hw import Transform, cameras, rigs, robots

        self._say("connecting to the render server")
        world = hw.SimWorld(self.server_url, scene=scene, spp=16, client=self.client) if self.client is None else \
            hw.SimWorld(scene=scene, spp=16, client=self.client)
        self._say("ground map: heights, obstacles and friction from the scene's files")
        gm = world.load_ground()
        self._map(gm)
        spawn = (world.meta.get("spawn") or [{"position": [0.0, 0.0, 1.5], "heading_deg": 0.0}])[0]
        if x is None or y is None:
            x, y = float(spawn["position"][0]), float(spawn["position"][1])
        if yaw_deg is None:
            yaw_deg = math.degrees(hw.transform.yaw_from_heading(math.radians(float(spawn.get("heading_deg", 0.0)))))
        x, y = self._free_spot(world, robots.ranger_mini(), float(x), float(y), float(yaw_deg))
        start = world.on_floor(x, y, float(yaw_deg))
        robot = world.spawn(robots.ranger_mini(dynamics=True), at=start, name="robot")
        cams = {}
        if self.calibration and Path(self.calibration).is_file():
            rig = rigs.spawn_camera_rig(world, robot, self.calibration, cameras=("left", "12_left"))
            cams["hdr"], cams["12bit"] = rig["left"], rig["12_left"]
        else:  # no calibration at hand: the same two cameras by their datasheet models, side by side on the mast
            mast = world.spawn(rigs.Mast(), attach_to=robot, link="top_plate", name="robot_mast")
            cams["hdr"] = world.spawn(cameras.LucidTDR054S(), attach_to=mast, link="top", mount=Transform((0.10, 0.04, 0.05)), name="robot_hdr")
            cams["12bit"] = world.spawn(cameras.LucidTRI032S(), attach_to=mast, link="top", mount=Transform((0.10, -0.06, 0.05)), name="robot_12bit")
        # behind and above the robot, looking down at it: not drawn, it is the viewer's eye
        cams["chase"] = world.spawn(cameras.LucidTRI032S(render_scale=0.2, drawn=False), attach_to=robot,
                                    mount=Transform((-1.9, 0.0, 1.7), (0.0, 24.0, 0.0)), name="robot_chase")
        with self.lock:
            self.world, self.robot, self.cameras, self.spawn = world, robot, cams, start

    @staticmethod
    def _free_spot(world, spec, x: float, y: float, yaw_deg: float) -> tuple[float, float]:
        """(x, y) or the nearest place around it where the robot's footprint is clear (a start pose is a camera's:
        it may stand over a table)."""
        ground = world.ground
        if ground.heights is None:
            return x, y
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        foot = spec.footprint()

        def clear(px, py):
            corners = np.c_[px + c * foot[:, 0] - s * foot[:, 1], py + s * foot[:, 0] + c * foot[:, 1]]
            floor = ground.height(px, py, near=ground.z)
            return abs(floor - ground.z) <= 0.5 and not ground.blocked(corners, floor)
        for radius in np.arange(0.0, 6.01, 0.25):
            for angle in np.arange(0.0, 2.0 * math.pi, math.pi / 8.0) if radius else (0.0,):
                px, py = x + radius * math.cos(angle), y + radius * math.sin(angle)
                if clear(px, py):
                    return float(px), float(py)
        return x, y

    def _map(self, gm) -> None:
        """The ground map as a picture (north up) and where it lies in the world."""
        mu = gm.mu()
        t = np.clip((mu - 0.1) / 0.8, 0.0, 1.0)
        img = np.stack([0.85 - 0.6 * t, 0.25 + 0.55 * t, 0.2 + 0.0 * t], -1)
        img[np.isnan(mu)] = (0.12, 0.14, 0.16)
        stand = np.where(gm.material >= 0, gm.heights, np.nan)
        img[(gm.heights - gm.floor_z > 0.5) & ~np.isnan(gm.heights)] = (0.03, 0.03, 0.04)
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray((img[::-1] * 255).astype(np.uint8)).save(buf, "PNG")
        rows, cols = gm.heights.shape
        del stand
        with self.lock:
            self.map_png = buf.getvalue()
            self.map_info = {"x0": gm.origin[0] - gm.cell / 2, "y0": gm.origin[1] - gm.cell / 2, "x1": gm.origin[0] + (cols - 0.5) * gm.cell,
                             "y1": gm.origin[1] + (rows - 0.5) * gm.cell, "cell": gm.cell,
                             "materials": {m: v for m, v in sorted(gm.friction.items(), key=lambda kv: kv[1]["mu"])}}

    # --- the loops ------------------------------------------------------------------------------------------------
    def _simulate(self) -> None:
        """The robot's own time: the command (or none: the chassis stops 0.5 s after the last one), a tick of the
        world every ``dt`` of real time, the trail. Never waits for an image."""
        from . import hw
        world, robot = self.world, self.robot
        last, ticks, t_rate = time.monotonic(), 0, time.monotonic()
        while self.running:
            now = time.monotonic()
            steps = min(int((now - last) / world.dt), 20)   # real time, at most a second behind
            if steps:
                with self.lock:
                    control = self.control
                    robot.spec.apply(robot.state, hw.Twist() if control is None or now - control[1] > CONTROL_TIMEOUT_S else control[0])
                    for _ in range(steps):
                        world.tick()
                    last += steps * world.dt if steps < 20 else (now - last)
                    ticks += steps
                    p, t = robot.pose, robot.state.twist
                    if abs(t.vx) + abs(t.vy) + abs(t.wz) > 1e-3:
                        self.moved_at = now
                    if not self.trail or math.hypot(p.xyz[0] - self.trail[-1][0], p.xyz[1] - self.trail[-1][1]) > 0.02:
                        self.trail.append([round(p.xyz[0], 3), round(p.xyz[1], 3)])
                        o = robot.state.odometry
                        c, s = math.cos(math.radians(self.spawn.rpy_deg[2])), math.sin(math.radians(self.spawn.rpy_deg[2]))
                        self.odo_trail.append([round(self.spawn.xyz[0] + c * o[0] - s * o[1], 3),
                                               round(self.spawn.xyz[1] + s * o[0] + c * o[1], 3)])
                        del self.trail[:-4000], self.odo_trail[:-4000]
            if now - t_rate > 1.0:
                self.sim_rate, ticks, t_rate = ticks / (now - t_rate), 0, now
            time.sleep(max(0.0, world.dt * 0.5 - (time.monotonic() - now)))

    def _run(self, scene, x, y, yaw_deg) -> None:
        from .hw import isp
        try:
            self._build(scene, x, y, yaw_deg)
        except Exception as exc:  # noqa: BLE001 - shown on the page
            with self.lock:
                self.error, self.running = f"{type(exc).__name__}: {exc}", False
            return
        world = self.world
        order, turn = list(self.cameras), 0
        self.moved_at = time.monotonic()
        threading.Thread(target=self._simulate, daemon=True).start()
        self._say("loading the scene with the robot in it (the first frame of each camera records, about half a minute)")
        while self.running:
            name = order[turn % len(order)]
            turn += 1
            # standing still for a while: more samples, once per camera (a clean image to look at); moving: few
            # samples, denoised on the server
            settled = time.monotonic() - self.moved_at > 0.6
            if settled and self.frames.get(name, {}).get("still"):
                time.sleep(0.02)
                continue
            try:
                t0 = time.monotonic()
                camera = self.cameras[name]
                with self.lock:  # the poses of this moment
                    request = world.request(camera, spp=256 if settled else 16, denoise=not settled, preview=False)
                out = world.client.render([request["view"]], "float32")[0]
                raw, _ = world.expose(camera, request, out)
                image = isp.process(raw, isp.Isp(tone="reinhard" if raw.bit_depth > 16 else "srgb"))
                with self.lock:
                    seq = self.frames.get(name, {}).get("seq", 0) + 1
                    self.frames[name] = {"jpeg": _jpeg(image), "seq": seq, "spp": 256 if settled else 16, "still": settled,
                                         "exposure_ms": round(raw.exposure_s * 1e3, 3), "gain_db": round(raw.gain_db, 2),
                                         "size": [int(image.shape[1]), int(image.shape[0])], "seconds": round(time.monotonic() - t0, 3),
                                         "model": camera.spec.model, "format": f"{raw.cfa} {raw.bit_depth}-bit"}
                    self.error = ""
                    if len(self.frames) == len(order):
                        self.message = ""
            except Exception as exc:  # noqa: BLE001 - the render server went away (a restart): say so, try again
                with self.lock:
                    self.error = f"{type(exc).__name__}: {str(exc)[:200]}"
                world._registered.clear()   # a restarted server knows no body models
                time.sleep(2.0)

    # --- the page's calls -----------------------------------------------------------------------------------------
    def command(self, body: dict) -> dict:
        from . import hw
        mode = str(body.get("mode", "ackermann"))
        if mode == "twist":
            control = hw.Twist(float(body.get("vx", 0.0)), float(body.get("vy", 0.0)), float(body.get("wz", 0.0)))
        else:
            control = hw.RangerControl(mode, float(body.get("linear", 0.0)), float(body.get("steering_deg", 0.0)), float(body.get("angular", 0.0)))
            control.twist(0.494)  # an unknown mode fails here, as a 400
        with self.lock:
            self.control = (control, time.monotonic())
        return {"ok": True}

    def state(self) -> dict:
        with self.lock:
            out = {"scene": self.scene, "running": self.running, "message": self.message, "error": self.error,
                   "cameras": {n: {k: v for k, v in f.items() if k != "jpeg"} for n, f in self.frames.items()},
                   "map": self.map_info, "sim_rate": round(self.sim_rate, 1)}
            robot = self.robot
            if robot is None:
                return out
            st, p = robot.state, robot.pose
            spec = robot.spec
            c, s = math.cos(math.radians(p.rpy_deg[2])), math.sin(math.radians(p.rpy_deg[2]))
            foot = spec.footprint()
            out["robot"] = {
                "x": round(p.xyz[0], 3), "y": round(p.xyz[1], 3), "z": round(p.xyz[2], 3), "yaw_deg": round(p.rpy_deg[2], 2),
                "roll_deg": round(p.rpy_deg[0], 2), "pitch_deg": round(p.rpy_deg[1], 2),
                "vx": round(st.twist.vx, 3), "vy": round(st.twist.vy, 3), "wz": round(st.twist.wz, 3),
                "command": [round(st.command.vx, 3), round(st.command.vy, 3), round(st.command.wz, 3)],
                "steer_deg": [round(float(v), 1) for v in np.degrees(st.steer)],
                "wheel_speed": [round(float(v), 3) for v in st.speed],
                "slip": [round(float(v), 3) for v in (st.slip if st.slip is not None else np.zeros(4))],
                "mu": [round(float(v), 3) for v in (st.mu if st.mu is not None else np.zeros(4))],
                "collided": bool(st.collided),
                "footprint": [[round(p.xyz[0] + c * fx - s * fy, 3), round(p.xyz[1] + s * fx + c * fy, 3)] for fx, fy in foot],
                "material": self.world.ground.material_at(p.xyz[0], p.xyz[1]) if self.world is not None else None}
            out["trail"], out["odometry_trail"] = self.trail[-1500:], self.odo_trail[-1500:]
            return out

    def frame(self, name: str) -> bytes | None:
        with self.lock:
            f = self.frames.get(name)
            return None if f is None else f["jpeg"]


def get(handler, robot: RobotSession, parts: list, query: dict) -> None:
    """GET of /robot and /api/robot/*, answered through the viewer's handler."""
    if parts == ["robot"]:
        return handler._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
    tail = parts[2:]
    if tail == ["state"]:
        return handler._json(200, robot.state())
    if tail == ["map.png"]:
        with robot.lock:
            png = robot.map_png
        return handler._send(200, png, "image/png") if png else handler._json(404, {"error": "no map yet"})
    if len(tail) == 2 and tail[0] == "frame" and tail[1].endswith(".jpg"):
        body = robot.frame(tail[1][:-4])
        return handler._send(200, body, "image/jpeg") if body else handler._json(404, {"error": "no frame yet"})
    if tail == ["scenes"]:
        client = robot.client
        if client is None:
            from .client import RenderClient
            client = RenderClient(robot.server_url)
        return handler._json(200, {"scenes": sorted(client.scans())})
    handler._json(404, {"error": "no such robot route"})


def post(handler, robot: RobotSession, parts: list, body: dict) -> None:
    tail = parts[2:]
    if tail == ["start"]:
        opt = lambda k: None if body.get(k) is None else float(body[k])  # noqa: E731
        return handler._json(200, robot.start(str(body["scene"]), opt("x"), opt("y"), opt("yaw_deg")))
    if tail == ["control"]:
        return handler._json(200, robot.command(body))
    if tail == ["stop"]:
        robot.stop()
        return handler._json(200, robot.state())
    handler._json(404, {"error": "no such robot route"})


def handles(parts: list) -> bool:
    return parts[:1] == ["robot"] or parts[:2] == ["api", "robot"]
