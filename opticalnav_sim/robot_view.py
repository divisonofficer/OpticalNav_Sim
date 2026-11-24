"""The viewer's robot: a robot with its camera rig put into the scene the viewer shows, driven from the page.

The viewer (gui.py) stays the editor: its camera renders the scene with the robot in it (``bodies`` gives its view the
robot's links as they stand), and each of the robot's cameras is a sensor pane the page switches on and off. One
``RobotSession`` holds a hw.SimWorld (the Ranger Mini with wheel dynamics, the mast, the rig's cameras), steps it in
real time, and renders the sensors that are switched on through the render server, with the editor's lighting:

    GET  /api/robot/state                  pose, velocity, odometry, slip and friction per wheel, sensors, messages
    GET  /api/robot/frame/<sensor>.jpg     that sensor's latest image (through its sensor model and ISP)
    GET  /api/robot/map.png                the ground map around the robot: friction (red slippery .. green grippy), obstacles
    POST /api/robot/spawn                  put the robot on the floor in front of the editor camera
    POST /api/robot/despawn
    POST /api/robot/sensors  {"hdr": true, "12bit": false}       which sensor panes render
    POST /api/robot/follow   {"on": true}                        the editor camera rides behind the robot
    POST /api/robot/control  {"mode", "linear", "steering_deg", "angular"}    held for 0.5 s (the chassis's own timeout)
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
SENSOR_LABELS = {"hdr": "HDR 카메라", "12bit": "12-bit 카메라"}
# what a sensor's view takes from the editor's: the scene as it is lit and dressed there (and so the same resident
# scene on the render server: another variant or lighting would not load a second one, but these keep one picture)
LOOK = ("variant", "lights", "env", "flash_band", "nir", "lighting", "lamps", "flash_scale", "flash_offset", "max_depth",
        "rr_depth", "caustics", "nir_range", "active_split", "mirrors")


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
        self.view_source = None          # () -> the editor's current view (dict) or None: the viewer sets it
        self.world = None
        self.robot = None
        self.cameras: dict = {}          # sensor name -> actor
        self.enabled: set = set()        # the sensors that render
        self.frames: dict = {}           # name -> {"jpeg", "seq", "spp", "exposure_ms", "gain_db", "size", "seconds"}
        self.message, self.error = "", ""
        self.scene = ""
        self.run_id = 0                  # a spawn's threads stop when it moves on
        self.running = False
        self.control = None              # (hw control, time it arrived)
        self.trail: list = []            # [x, y] of the robot
        self.odo_trail: list = []        # [x, y] as its wheels count it
        self.start_pose = None           # the pose odometry starts from
        self.map_png, self.map_info = None, None
        self.sim_rate = 0.0
        self.moved_at = 0.0              # when the robot last moved (a camera standing still gets more samples)
        self.pose_seq = 0                # grows whenever a link has moved: the viewer redraws its own view
        self._bodies: list = []          # the robot's links as the render server's view field, as of pose_seq

    # --- lifecycle ------------------------------------------------------------------------------------------------
    def spawn(self, scene: str, eye, heading: float = 0.0, elevation: float = 0.0) -> dict:
        """Put the robot into ``scene`` on the floor in front of a camera: ``eye`` its (x, y, height), ``heading`` and
        ``elevation`` in radians as MatterSim counts them (heading 0 looks along +y, positive turns right)."""
        self.despawn()
        with self.lock:
            self.run_id += 1
            self.scene, self.message, self.error = scene, "로봇을 준비하는 중", ""
            self.frames, self.trail, self.odo_trail, self.map_png, self.map_info = {}, [], [], None, None
            self.running = True
            run = self.run_id
        threading.Thread(target=self._run, args=(run, scene, tuple(float(v) for v in eye), float(heading), float(elevation)),
                         name="robot", daemon=True).start()
        return self.state()

    def despawn(self) -> None:
        with self.lock:
            had = self.robot is not None
            self.run_id += 1
            self.running = False
            self.world = self.robot = None
            self.cameras, self.frames, self._bodies, self.control = {}, {}, [], None
            self.message = ""
            if had:
                self.pose_seq += 1

    def _alive(self, run: int) -> bool:
        return self.running and self.run_id == run

    def _say(self, text: str) -> None:
        with self.lock:
            self.message = text

    def _build(self, run: int, scene: str, eye, heading: float, elevation: float) -> None:
        from . import groundmap, hw
        from .hw import Transform, cameras, rigs, robots

        self._say("렌더 서버에 연결하는 중")
        world = hw.SimWorld(self.server_url, scene=scene, spp=16) if self.client is None else \
            hw.SimWorld(scene=scene, spp=16, client=self.client)
        # the floor under the camera: the scene's own surfaces below it, around where it stands
        self._say("바닥 찾는 중 (씬 파일에서)")
        folder = Path(str(world.meta.get("scene_dir") or ""))
        if not (folder / "base.xml").is_file():
            raise FileNotFoundError(f"이 씬의 파일이 뷰어가 도는 기계에 없습니다 ({folder}): 지면 지도를 만들 수 없습니다")
        try:
            probe = groundmap.build(folder, eye[2] - 0.05, top=0.0, cell=0.25, below=40.0,
                                    bounds=(eye[0] - 1.5, eye[1] - 1.5, eye[0] + 1.5, eye[1] + 1.5))
            seen = probe.heights[probe.material >= 0]
        except ValueError:
            seen = ()
        if not len(seen):
            raise ValueError("카메라 아래에 바닥이 없습니다: 바닥 위로 옮긴 뒤 다시 스폰하세요")
        floor_z = float(np.median(seen))
        # where the camera looks at the floor, between 1.5 and 8 m ahead (2.5 m when it looks level or up)
        ahead = (eye[2] - floor_z) / math.tan(-elevation) if elevation < -0.05 else 2.5
        ahead = min(max(ahead, 1.5), 8.0)
        x, y = eye[0] + ahead * math.sin(heading), eye[1] + ahead * math.cos(heading)
        yaw_deg = math.degrees(hw.transform.yaw_from_heading(heading))
        self._say("지면 지도 만드는 중: 높이·장애물·마찰 (씬 파일의 재질에서, 처음 한 번)")
        gm = world.load_ground(floor_z=floor_z, center=(x, y))
        self._map(gm)
        x, y = self._free_spot(world, robots.ranger_mini(), x, y, yaw_deg)
        start = world.on_floor(x, y, yaw_deg)
        robot = world.spawn(robots.ranger_mini(dynamics=True), at=start, name="robot")
        cams = {}
        if self.calibration and Path(self.calibration).is_file():
            rig = rigs.spawn_camera_rig(world, robot, self.calibration, cameras=("left", "12_left"))
            cams["hdr"], cams["12bit"] = rig["left"], rig["12_left"]
        else:  # no calibration at hand: the same two cameras by their datasheet models, side by side on the mast
            mast = world.spawn(rigs.Mast(), attach_to=robot, link="top_plate", name="robot_mast")
            cams["hdr"] = world.spawn(cameras.LucidTDR054S(), attach_to=mast, link="top", mount=Transform((0.10, 0.04, 0.05)), name="robot_hdr")
            cams["12bit"] = world.spawn(cameras.LucidTRI032S(), attach_to=mast, link="top", mount=Transform((0.10, -0.06, 0.05)), name="robot_12bit")
        self._say("로봇 모델을 렌더 서버에 올리는 중")
        bodies = world.bodies()   # registers the models
        with self.lock:
            if self.run_id != run:
                return
            self.world, self.robot, self.cameras, self.start_pose = world, robot, cams, start
            self._bodies = bodies
            self.pose_seq += 1
            self.message = "로봇이 든 씬을 올리는 중 (처음 한 번, 씬에 따라 30초쯤)"

    @staticmethod
    def _free_spot(world, spec, x: float, y: float, yaw_deg: float) -> tuple[float, float]:
        """(x, y) or the nearest place around it where the robot's footprint is clear (in front of a camera may be a
        table)."""
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
        img[(gm.heights - gm.floor_z > 0.5) & ~np.isnan(gm.heights)] = (0.03, 0.03, 0.04)
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray((img[::-1] * 255).astype(np.uint8)).save(buf, "PNG")
        rows, cols = gm.heights.shape
        with self.lock:
            self.map_png = buf.getvalue()
            self.map_info = {"x0": gm.origin[0] - gm.cell / 2, "y0": gm.origin[1] - gm.cell / 2, "x1": gm.origin[0] + (cols - 0.5) * gm.cell,
                             "y1": gm.origin[1] + (rows - 0.5) * gm.cell, "cell": gm.cell, "floor_z": round(float(gm.floor_z), 3),
                             "materials": {m: v for m, v in sorted(gm.friction.items(), key=lambda kv: kv[1]["mu"])}}

    # --- the loops ------------------------------------------------------------------------------------------------
    def _simulate(self, run: int) -> None:
        """The robot's own time: the command (or none: the chassis stops 0.5 s after the last one), a tick of the
        world every ``dt`` of real time, the trail. Never waits for an image."""
        from . import hw
        world, robot = self.world, self.robot
        last, ticks, t_rate = time.monotonic(), 0, time.monotonic()
        drawn = None
        while self._alive(run):
            now = time.monotonic()
            steps = min(int((now - last) / world.dt), 20)   # real time, at most a second behind
            if steps:
                with self.lock:
                    if self.run_id != run:
                        return
                    control = self.control
                    robot.spec.apply(robot.state, hw.Twist() if control is None or now - control[1] > CONTROL_TIMEOUT_S else control[0])
                    for _ in range(steps):
                        world.tick()
                    last += steps * world.dt if steps < 20 else (now - last)
                    ticks += steps
                    p, t = robot.pose, robot.state.twist
                    if abs(t.vx) + abs(t.vy) + abs(t.wz) > 1e-3:
                        self.moved_at = now
                    # the links as they stand now, for the viewer's own camera: a new sequence number when one moved
                    stand = np.r_[p.xyz, p.rpy_deg, np.degrees(robot.state.steer)]
                    if drawn is None or np.abs(stand[:3] - drawn[:3]).max() > 5e-4 or np.abs(stand[3:] - drawn[3:]).max() > 0.05:
                        drawn = stand
                        self._bodies = world.bodies()
                        self.pose_seq += 1
                    if not self.trail or math.hypot(p.xyz[0] - self.trail[-1][0], p.xyz[1] - self.trail[-1][1]) > 0.02:
                        self.trail.append([round(p.xyz[0], 3), round(p.xyz[1], 3)])
                        o = robot.state.odometry
                        c, s = math.cos(math.radians(self.start_pose.rpy_deg[2])), math.sin(math.radians(self.start_pose.rpy_deg[2]))
                        self.odo_trail.append([round(self.start_pose.xyz[0] + c * o[0] - s * o[1], 3),
                                               round(self.start_pose.xyz[1] + s * o[0] + c * o[1], 3)])
                        del self.trail[:-4000], self.odo_trail[:-4000]
            if now - t_rate > 1.0:
                self.sim_rate, ticks, t_rate = ticks / (now - t_rate), 0, now
            time.sleep(max(0.0, world.dt * 0.5 - (time.monotonic() - now)))

    def _sensor_view(self, request: dict, template: dict | None) -> tuple[dict, str]:
        """The sensor's view with the editor's look, and that look as a key (a still image is of one look)."""
        view = request["view"]
        if template is not None and template.get("scan") == self.scene:
            for k in LOOK:
                if k in template:
                    view[k] = template[k]
        view["bands"], view["renderer"], view["depth"] = "rgb", "path", False
        return view, json.dumps({k: view.get(k) for k in LOOK}, sort_keys=True, default=str)

    def _run(self, run: int, scene: str, eye, heading: float, elevation: float) -> None:
        from .hw import isp
        try:
            self._build(run, scene, eye, heading, elevation)
        except Exception as exc:  # noqa: BLE001 - shown on the page
            with self.lock:
                if self.run_id == run:
                    self.error, self.running, self.message = f"{type(exc).__name__}: {exc}", False, ""
            return
        world = self.world
        if world is None:
            return
        self.moved_at = time.monotonic()
        threading.Thread(target=self._simulate, args=(run,), name="robot-sim", daemon=True).start()
        turn = 0
        while self._alive(run):
            with self.lock:
                order = [n for n in self.cameras if n in self.enabled]
            if not order:
                time.sleep(0.05)
                continue
            name = order[turn % len(order)]
            turn += 1
            template = self.view_source() if self.view_source is not None else None
            # standing still for a while: more samples, once per camera (a clean image to look at); moving: few
            # samples, denoised on the server
            settled = time.monotonic() - self.moved_at > 0.6
            try:
                t0 = time.monotonic()
                with self.lock:  # the poses of this moment
                    if self.run_id != run:
                        return
                    camera = self.cameras[name]
                    request = world.request(camera, spp=256 if settled else 16, denoise=not settled, preview=False)
                    view, look = self._sensor_view(request, template)
                    seq_at = self.pose_seq
                    have = self.frames.get(name, {})
                if settled and have.get("still") and have.get("look") == look and have.get("pose_seq") == seq_at:
                    time.sleep(0.03)
                    continue
                out = world.client.render([view], "float32")[0]
                raw, _ = world.expose(camera, request, out)
                image = isp.process(raw, isp.Isp(tone="reinhard" if raw.bit_depth > 16 else "srgb"))
                with self.lock:
                    if self.run_id != run:
                        return
                    seq = self.frames.get(name, {}).get("seq", 0) + 1
                    self.frames[name] = {"jpeg": _jpeg(image), "seq": seq, "spp": 256 if settled else 16, "still": settled,
                                         "look": look, "pose_seq": seq_at,
                                         "exposure_ms": round(raw.exposure_s * 1e3, 3), "gain_db": round(raw.gain_db, 2),
                                         "size": [int(image.shape[1]), int(image.shape[0])], "seconds": round(time.monotonic() - t0, 3),
                                         "model": camera.spec.model, "format": f"{raw.cfa} {raw.bit_depth}-bit"}
                    self.error = ""
                    if self.message.startswith("로봇이 든 씬"):
                        self.message = ""
            except Exception as exc:  # noqa: BLE001 - the render server went away (a restart): say so, try again
                with self.lock:
                    if self.run_id == run:
                        self.error = f"{type(exc).__name__}: {str(exc)[:200]}"
                self.reregister()
                time.sleep(2.0)

    # --- the viewer's calls ---------------------------------------------------------------------------------------
    def bodies(self, scene: str) -> list:
        """The robot's links for a view of ``scene`` (the render server's ``bodies`` field): none elsewhere."""
        with self.lock:
            return list(self._bodies) if self.robot is not None and scene == self.scene else []

    def loaded(self) -> None:
        """The viewer's own camera has drawn the scene with the robot in it."""
        with self.lock:
            if self.robot is not None and self.message.startswith("로봇이 든 씬"):
                self.message = ""

    def reregister(self) -> None:
        """A restarted render server knows no body models: send them again with the next view."""
        with self.lock:
            world = self.world
            if world is None:
                return
            world._registered.clear()
            try:
                self._bodies = world.bodies()
                self.pose_seq += 1
            except Exception:  # noqa: BLE001 - still away: the next failure tries again
                pass

    def pose2d(self):
        """(x, y, z, yaw in radians) of the robot's base, or None."""
        with self.lock:
            if self.robot is None:
                return None
            p = self.robot.pose
            return float(p.xyz[0]), float(p.xyz[1]), float(p.xyz[2]), math.radians(float(p.rpy_deg[2]))

    def set_sensors(self, wanted: dict) -> None:
        with self.lock:
            for name, on in wanted.items():
                if name not in SENSOR_LABELS:
                    raise ValueError(f"no sensor {name!r} (have: {', '.join(SENSOR_LABELS)})")
                (self.enabled.add if on else self.enabled.discard)(name)
                if not on:
                    self.frames.pop(name, None)

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
            out = {"scene": self.scene, "running": self.running, "spawned": self.robot is not None, "message": self.message,
                   "error": self.error, "map": self.map_info, "sim_rate": round(self.sim_rate, 1),
                   "calibrated": bool(self.calibration and Path(self.calibration).is_file()),
                   "sensors": [{"name": n, "label": label, "enabled": n in self.enabled,
                                "model": self.cameras[n].spec.model if n in self.cameras else "",
                                **{k: v for k, v in self.frames.get(n, {}).items() if k not in ("jpeg", "look")}}
                               for n, label in SENSOR_LABELS.items()]}
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
    """GET of /api/robot/*, answered through the viewer's handler."""
    tail = parts[2:]
    if tail == ["state"]:
        return handler._json(200, robot_state(handler, robot))
    if tail == ["map.png"]:
        with robot.lock:
            png = robot.map_png
        return handler._send(200, png, "image/png") if png else handler._json(404, {"error": "no map yet"})
    if len(tail) == 2 and tail[0] == "frame" and tail[1].endswith(".jpg"):
        body = robot.frame(tail[1][:-4])
        return handler._send(200, body, "image/jpeg") if body else handler._json(404, {"error": "no frame yet"})
    handler._json(404, {"error": "no such robot route"})


def robot_state(handler, robot: RobotSession) -> dict:
    out = robot.state()
    session = getattr(handler, "viewer_session", None)
    if session is not None:
        out["follow"] = session.follow is not None
    return out


def post(handler, robot: RobotSession, parts: list, body: dict) -> None:
    """POST of /api/robot/*. Spawning and following need the viewer's own camera: its session does them."""
    tail = parts[2:]
    session = getattr(handler, "viewer_session", None)
    if tail == ["control"]:
        return handler._json(200, robot.command(body))
    if tail == ["sensors"]:
        robot.set_sensors({str(k): bool(v) for k, v in body.items()})
    elif tail == ["spawn"] and session is not None:
        session.run("robot", session.spawn_robot)
    elif tail == ["despawn"] and session is not None:
        session.run("robot", session.despawn_robot)
    elif tail == ["follow"] and session is not None:
        session.run("robot", session.set_follow, bool(body.get("on")))
    else:
        return handler._json(404, {"error": "no such robot route"})
    handler._json(200, robot_state(handler, robot))


def handles(parts: list) -> bool:
    return parts[:2] == ["api", "robot"]
