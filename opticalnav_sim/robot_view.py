"""The viewer's robot: a robot with its camera rig put into the scene the viewer shows, driven from the page.

The viewer (gui.py) stays the editor: its camera renders the scene with the robot in it (``bodies`` gives its view the
robot's links as they stand), and each of the robot's cameras is a sensor a sub viewer of the page shows. What the
robot is comes from its configuration (hw/config.py: a structure file naming the base, its lights, masts and cameras);
one ``RobotSession`` spawns it into a hw.SimWorld, steps it in real time, and renders the sensors that are switched
on through the render server, with the editor's lighting:

    GET  /api/robot/state                  pose, velocity, odometry, slip and friction per wheel, sensors, messages
    GET  /api/robot/frame/<sensor>.jpg     that sensor's latest image (through its sensor model and ISP)
    GET  /api/robot/map.png                the ground map around the robot: friction (red slippery .. green grippy), obstacles
    GET  /api/robot/configs                the registered robots (hw.config.available) and the one in use
    POST /api/robot/spawn    {"config"}    put the robot (that configuration, or the one in use) on the floor in front
                                           of the editor camera
    POST /api/robot/despawn
    POST /api/robot/sensors  {"hdr": true, "12bit": false}       sensors rendered whoever watches (a script's; a page's
                                                                 sub viewers name theirs in its socket subscription:
                                                                 a sensor no page shows and no call switched on is
                                                                 not rendered)
    POST /api/robot/follow   {"on": true, "mode": "ride"|"track"}   the editor camera rides behind the robot, or stays
                                                                 where it is and turns to keep it in view
    POST /api/robot/record   {"action": "start"}                 log the drive from now on
                             {"action": "stop", "fps", "spp", "sensor_spp", "sensors", "name"}   render what was logged
                             (robot_record.py: every frame at full quality) and write an MP4; "cancel" drops it
    GET  /api/robot/recording/<name>.mp4   a finished recording (state["recording"]["file"] names it)
    POST /api/robot/control  {"mode", "linear", "steering_deg", "angular"}    held for 0.5 s (the chassis's own timeout)
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np

CONTROL_TIMEOUT_S = 0.5      # the Ranger's CAN command timeout: no command for this long stops it
DEFAULT_CONFIG = "ranger_mini_rig"
# what a sensor's view takes from the editor's: the scene as it is lit and dressed there (and so the same resident
# scene on the render server: another variant or lighting would not load a second one, but these keep one picture)
LOOK = ("variant", "lights", "env", "flash_band", "nir", "lighting", "lamps", "flash_scale", "flash_offset", "max_depth",
        "rr_depth", "caustics", "nir_range", "active_split", "mirrors")


def clean_render(client, view: dict, passes: int = 8) -> dict:
    """A view rendered as ``passes`` passes of ``view["spp"]`` samples, their mean with fireflies held back
    (gui.robust_mean: a pass's pixel far above the others' median is scaled down), then denoised: the server's
    answer as one render would give it (``s0``.. or ``radiance``). The denoiser alone leaves a firefly as a white
    speck; the mean alone leaves the grain. For a camera standing still, and for recordings."""
    from .gui import robust_mean
    outs = [client.render([dict(view, seed=int(view.get("seed", 0)) * 1000 + i, denoise=False)], "float32")[0] for i in range(passes)]
    keys = [k for k in ("s0", "s1", "s2", "radiance") if k in outs[0]]
    spp = float(view["spp"])
    acc = {("s0" if k == "radiance" else k): np.stack([np.asarray(o[k], dtype=np.float32) for o in outs]) * spp for k in keys}
    mean = robust_mean(acc, [spp] * passes)
    try:
        den = client.denoise({k: mean[k] for k in ("s0", "s1", "s2") if k in mean})
        mean.update({k: den[k] for k in den if k in mean})
    except (ConnectionError, OSError, RuntimeError):
        pass   # the mean as it is rather than nothing
    out = dict(outs[-1])
    for k in keys:
        out[k] = mean["s0" if k == "radiance" else k]
    return out


def _jpeg(image: np.ndarray, quality: int = 82) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


class RobotSession:
    def __init__(self, server_url: str | None = None, calibration: str | None = None, client=None, config: str | None = None):
        from .hw import config as robot_config
        self.server_url, self.client = server_url, client
        self.calibration = calibration or os.environ.get("OPTICALNAV_RIG_CALIBRATION")
        self.config = robot_config.load(config or os.environ.get("OPTICALNAV_ROBOT") or DEFAULT_CONFIG)   # what the robot is
        self.calibrated = False
        self.lock = threading.RLock()
        self.view_source = None          # () -> the editor's current view (dict) or None: the viewer sets it
        self.viewers = None              # () -> the sensors some page shows now (the viewer sets it): rendered while shown
        self.world = None
        self.robot = None
        self.cameras: dict = {}          # sensor name -> actor
        self.enabled: set = set()        # the sensors switched on by a call (POST /api/robot/sensors): rendered whoever watches
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
        self.editor_pose = None          # () -> the editor camera's (pose, follow) now (the viewer sets it): logged with a drive
        self.log: list | None = None     # a drive being logged for a recording: one entry a tick (log_start)
        self.replaying = False           # a recording renders: the robot's real time and its live sensors hold
        self.recording: dict = {"phase": "idle"}   # logging / rendering / encoding / done / error, with its numbers
        self._bodies: list = []          # the robot's links as the render server's view field, as of pose_seq

    # --- lifecycle ------------------------------------------------------------------------------------------------
    def use(self, config) -> None:
        """Another robot (a registered name, or the path of a structure file) for the next spawn."""
        from .hw import config as robot_config
        cfg = robot_config.load(config)
        with self.lock:
            self.config = cfg
            self.enabled &= {s["name"] for s in cfg.sensors()}

    def labels(self) -> dict:
        return {s["name"]: s["label"] for s in self.config.sensors()}

    def spawn(self, scene: str, eye, heading: float = 0.0, elevation: float = 0.0, config=None) -> dict:
        """Put the robot into ``scene`` on the floor in front of a camera: ``eye`` its (x, y, height), ``heading`` and
        ``elevation`` in radians as MatterSim counts them (heading 0 looks along +y, positive turns right)."""
        self.despawn()
        if config:
            self.use(config)
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

    # --- recordings: a drive logged tick by tick, rendered again by robot_record.render ----------------------------
    LOG_LIMIT = 6000   # ticks: five minutes

    def log_start(self) -> None:
        with self.lock:
            if self.robot is None:
                raise ValueError("no robot to record: spawn one first")
            if self.replaying:
                raise ValueError("a recording is being rendered")
            self.log = []
            self.recording = {"phase": "logging", "ticks": 0, "seconds": 0.0}

    def log_stop(self) -> list:
        with self.lock:
            log, self.log = self.log or [], None
            return log

    def _entry(self) -> dict:
        """The robot as it stands now, to put it back there: its pose, its base's state, where its odometry says it
        is, and the editor camera."""
        robot = self.robot
        st = robot.state
        snap = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in vars(st).items() if k != "ground"} if st is not None else {}
        for k in ("command", "twist"):
            if k in snap:
                snap[k] = type(snap[k])(**vars(snap[k]))
        o = snap.get("odometry")
        o = np.zeros(3) if o is None else o
        c, s = math.cos(math.radians(self.start_pose.rpy_deg[2])), math.sin(math.radians(self.start_pose.rpy_deg[2]))
        cam, follow = self.editor_pose() if self.editor_pose is not None else ((0.0, 0.0, 1.5, 0.0, 0.0), None)
        return {"t": self.world.time, "pose": robot.pose, "state": snap, "cam": cam, "follow": follow,
                "odometry_xy": (self.start_pose.xyz[0] + c * o[0] - s * o[1], self.start_pose.xyz[1] + s * o[0] + c * o[1])}

    def restore(self, entry: dict) -> None:
        """Put the robot back as an entry of a log has it (its pose and its base's state: the wheels' angles too)."""
        robot = self.robot
        robot.pose = entry["pose"]
        for k, v in entry["state"].items():
            setattr(robot.state, k, v.copy() if isinstance(v, np.ndarray) else v)

    def _alive(self, run: int) -> bool:
        return self.running and self.run_id == run

    def _say(self, text: str) -> None:
        with self.lock:
            self.message = text

    def _build(self, run: int, scene: str, eye, heading: float, elevation: float) -> None:
        from . import groundmap, hw
        from .hw import config as robot_config

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
        cfg = self.config
        x, y = self._free_spot(world, robot_config.base_spec(cfg), x, y, yaw_deg)
        start = world.on_floor(x, y, yaw_deg)
        # the robot as its configuration lists it: the base, then its lights, masts and cameras on their links
        built = robot_config.spawn(world, cfg, at=start, name="robot",
                                   calibration=self.calibration if self.calibration and Path(self.calibration).is_file() else None)
        robot, cams = built.base, dict(built.sensors)
        self._say("로봇 모델을 렌더 서버에 올리는 중")
        bodies = world.bodies()   # registers the models
        with self.lock:
            if self.run_id != run:
                return
            self.world, self.robot, self.cameras, self.start_pose = world, robot, cams, start
            self.calibrated = built.calibrated
            self._bodies = bodies
            self.pose_seq += 1
            self.message = "로봇이 든 씬을 올리는 중 (처음 한 번, 씬에 따라 30초쯤)"

    @staticmethod
    def _free_spot(world, spec, x: float, y: float, yaw_deg: float) -> tuple[float, float]:
        """(x, y) or the nearest place around it where the robot's footprint is clear (in front of a camera may be a
        table)."""
        ground = world.ground
        if ground.heights is None or not hasattr(spec, "footprint"):
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
            if self.replaying:   # a recording renders: the robot stands where the recording puts it
                last = now
                time.sleep(0.05)
                continue
            steps = min(int((now - last) / world.dt), 20)   # real time, at most a second behind
            if steps:
                with self.lock:
                    if self.run_id != run:
                        return
                    control = self.control
                    if hasattr(robot.spec, "apply"):   # a base that drives
                        robot.spec.apply(robot.state, hw.Twist() if control is None or now - control[1] > CONTROL_TIMEOUT_S else control[0])
                    for _ in range(steps):
                        world.tick()
                        if self.log is not None:   # a recording logs: every tick, to be rendered again
                            self.log.append(self._entry())
                            self.recording.update(ticks=len(self.log), seconds=round(len(self.log) * world.dt, 2))
                            if len(self.log) >= self.LOG_LIMIT:
                                self.recording["full"] = True
                                break
                    last += steps * world.dt if steps < 20 else (now - last)
                    ticks += steps
                    p, t = robot.pose, getattr(robot.state, "twist", None) or hw.Twist()
                    if abs(t.vx) + abs(t.vy) + abs(t.wz) > 1e-3:
                        self.moved_at = now
                    # the links as they stand now, for the viewer's own camera: a new sequence number when one moved
                    stand = np.r_[p.xyz, p.rpy_deg, np.degrees(getattr(robot.state, "steer", np.zeros(0)))]
                    if drawn is None or np.abs(stand[:3] - drawn[:3]).max() > 5e-4 or np.abs(stand[3:] - drawn[3:]).max() > 0.05:
                        drawn = stand
                        self._bodies = world.bodies()
                        self.pose_seq += 1
                    if not self.trail or math.hypot(p.xyz[0] - self.trail[-1][0], p.xyz[1] - self.trail[-1][1]) > 0.02:
                        self.trail.append([round(p.xyz[0], 3), round(p.xyz[1], 3)])
                        o = getattr(robot.state, "odometry", None)
                        o = np.zeros(3) if o is None else o
                        c, s = math.cos(math.radians(self.start_pose.rpy_deg[2])), math.sin(math.radians(self.start_pose.rpy_deg[2]))
                        self.odo_trail.append([round(self.start_pose.xyz[0] + c * o[0] - s * o[1], 3),
                                               round(self.start_pose.xyz[1] + s * o[0] + c * o[1], 3)])
                        del self.trail[:-4000], self.odo_trail[:-4000]
            if now - t_rate > 1.0:
                self.sim_rate, ticks, t_rate = ticks / (now - t_rate), 0, now
            time.sleep(max(0.0, world.dt * 0.5 - (time.monotonic() - now)))

    def _sensor_view(self, request: dict, template: dict | None, polarized: bool = False) -> tuple[dict, str]:
        """The sensor's view with the editor's look, and that look as a key (a still image is of one look).
        ``polarized``: the sensor measures polarization, so its view is a polarized render whatever the editor's is;
        any other sensor takes the editor's mode (unpolarized RGB when polarization is off there: the same resident
        scene as the editor's view, and several times faster)."""
        view = request["view"]
        if template is not None and template.get("scan") == self.scene:
            for k in LOOK:
                if k in template:
                    view[k] = template[k]
            if not polarized and template.get("mode") in ("polar", "rgb"):
                view["mode"] = template["mode"]
        view["bands"], view["renderer"], view["depth"] = "rgb", "path", False
        return view, json.dumps({k: view.get(k) for k in (*LOOK, "mode")}, sort_keys=True, default=str)

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
            if self.replaying:          # the recording renders the sensors itself
                time.sleep(0.05)
                continue
            wanted = self.rendering()   # a sensor nobody shows is not rendered
            with self.lock:
                order = [n for n in self.cameras if n in wanted]
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
                    view, look = self._sensor_view(request, template, camera.spec.sensor.cfa.startswith("polar_"))
                    # standing still: eight passes, fireflies held back, then the denoiser (clean_render). Unpolarized
                    # a pass costs a fifth and has more outliers: four times the samples
                    still_passes = 8
                    if settled:
                        view["spp"] = 128 if view.get("mode") == "rgb" else 32
                    spp = int(view["spp"]) * (still_passes if settled else 1)
                    seq_at = self.pose_seq
                    have = self.frames.get(name, {})
                if settled and have.get("still") and have.get("look") == look and have.get("pose_seq") == seq_at:
                    time.sleep(0.03)
                    continue
                out = clean_render(world.client, view, still_passes) if settled else world.client.render([view], "float32")[0]
                raw, _ = world.expose(camera, request, out)
                image = isp.process(raw, isp.Isp(tone="reinhard" if raw.bit_depth > 16 else "srgb"))
                with self.lock:
                    if self.run_id != run:
                        return
                    seq = self.frames.get(name, {}).get("seq", 0) + 1
                    self.frames[name] = {"jpeg": _jpeg(image), "seq": seq, "spp": spp, "still": settled,
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
    def rendering(self) -> set:
        """The sensors that render: those a connected page shows in a sub viewer, and those switched on by a call."""
        shown = set()
        if self.viewers is not None:
            with contextlib.suppress(Exception):
                shown = set(self.viewers())
        with self.lock:
            return shown | self.enabled

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
                if name not in self.labels():
                    raise ValueError(f"{self.config.name} has no sensor {name!r} (has: {', '.join(self.labels())})")
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
        rendering = self.rendering()
        with self.lock:
            out = {"scene": self.scene, "running": self.running, "spawned": self.robot is not None, "message": self.message,
                   "error": self.error, "map": self.map_info, "sim_rate": round(self.sim_rate, 1),
                   "calibrated": self.calibrated, "config": self.config.summary(),
                   "recording": {k: v for k, v in self.recording.items() if k != "cancel"},
                   "sensors": [{"name": n, "label": label, "enabled": n in rendering,
                                "model": self.cameras[n].spec.model if n in self.cameras else "",
                                **{k: v for k, v in self.frames.get(n, {}).items() if k not in ("jpeg", "look")}}
                               for n, label in self.labels().items()]}
            for s, listed in zip(out["sensors"], self.config.sensors()):
                s["model"] = s["model"] or listed["model"]
            robot = self.robot
            if robot is None:
                return out
            st, p = robot.state, robot.pose
            spec = robot.spec
            c, s = math.cos(math.radians(p.rpy_deg[2])), math.sin(math.radians(p.rpy_deg[2]))
            if not hasattr(spec, "footprint"):   # a base that does not drive: where it stands
                out["robot"] = {"x": round(p.xyz[0], 3), "y": round(p.xyz[1], 3), "z": round(p.xyz[2], 3), "yaw_deg": round(p.rpy_deg[2], 2),
                                "roll_deg": 0.0, "pitch_deg": 0.0, "vx": 0.0, "vy": 0.0, "wz": 0.0, "command": [0.0, 0.0, 0.0],
                                "steer_deg": [], "wheel_speed": [], "slip": [], "mu": [], "collided": False, "material": None,
                                "footprint": [[round(p.xyz[0] + dx, 3), round(p.xyz[1] + dy, 3)] for dx, dy in ((-.2, -.2), (.2, -.2), (.2, .2), (-.2, .2))]}
                out["trail"], out["odometry_trail"] = self.trail[-1500:], self.odo_trail[-1500:]
                return out
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
    if tail == ["configs"]:
        from .hw import config as robot_config
        return handler._json(200, {"configs": robot_config.available(), "current": robot.config.name})
    if len(tail) == 2 and tail[0] == "recording" and tail[1].endswith(".mp4"):
        from .robot_record import FOLDER
        file = FOLDER / Path(tail[1]).name
        if not file.is_file():
            return handler._json(404, {"error": "no such recording"})
        return handler._send(200, file.read_bytes(), "video/mp4")
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
        out["follow_mode"] = None if session.follow is None else "track" if session.follow == "track" else "ride"
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
        if body.get("config"):
            robot.use(str(body["config"]))   # an unknown robot fails here, as a 400
        session.run("robot", session.spawn_robot)
    elif tail == ["config"]:
        robot.use(str(body["config"]))
    elif tail == ["despawn"] and session is not None:
        session.run("robot", session.despawn_robot)
    elif tail == ["follow"] and session is not None:
        session.run("robot", session.set_follow, bool(body.get("on")), str(body.get("mode") or "ride"))
    elif tail == ["record"] and session is not None:
        session.record(str(body.get("action") or "start"), body)
    else:
        return handler._json(404, {"error": "no such robot route"})
    handler._json(200, robot_state(handler, robot))


def handles(parts: list) -> bool:
    return parts[:2] == ["api", "robot"]
