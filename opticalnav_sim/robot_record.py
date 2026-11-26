"""A recording of the viewer's robot: a drive logged in real time, rendered again frame by frame, written as an MP4.

While a recording logs (robot_view.RobotSession.log_start), every tick of the robot keeps its state and the editor
camera's pose. ``render`` then holds the robot's real time, puts the robot and the editor camera back at each video
frame's moment, and renders what the viewer shows at full quality: the editor's view through the viewer's own session
(its lighting, tone and accumulation, to ``spp`` samples, denoised) and each sensor through its sensor model and ISP.
The frames are laid out as one picture (editor, sensors, the ground map with the robot's trail, its wheels' numbers)
and encoded with ffmpeg. A live drive is noisy (16 samples, as fast as it comes); its recording is not.
"""
from __future__ import annotations

import io
import math
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

FOLDER = Path.home() / ".cache" / "opticalnav_sim" / "recordings"
BG, PANEL, INK, MUTED, ACCENT, WARM, BAD = (13, 17, 19), (22, 28, 31), (228, 233, 235), (140, 153, 160), (63, 201, 212), (240, 130, 80), (239, 107, 107)


HANGUL_FONTS = ("~/.fonts/NotoSansKR-VF.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/mnt/c/Windows/Fonts/malgun.ttf")


def _font(size: int, bold: bool = False, hangul: bool = False):
    """A font of the frame's text. ``hangul``: one that has Korean (a robot's and its sensors' labels may be Korean),
    when the machine has one."""
    from PIL import ImageFont
    plain = (f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",)
    for path in (HANGUL_FONTS if hangul else ()) + plain:
        try:
            font = ImageFont.truetype(str(Path(path).expanduser()), size)
        except OSError:
            continue
        try:   # a variable font starts at its thinnest: its weight by name, where it has one
            font.set_variation_by_name("Bold" if bold else "Medium")
        except (OSError, AttributeError, ValueError):
            pass
        return font
    return ImageFont.load_default()


def _has_hangul_font() -> bool:
    return any(Path(p).expanduser().is_file() for p in HANGUL_FONTS)


def _fit(image: np.ndarray, width: int, height: int):
    """``image`` scaled to fit a box, and where it lies in it."""
    from PIL import Image
    im = Image.fromarray(np.asarray(image, dtype=np.uint8))
    k = min(width / im.width, height / im.height)
    size = (max(1, round(im.width * k)), max(1, round(im.height * k)))
    return im.resize(size, Image.LANCZOS), ((width - size[0]) // 2, (height - size[1]) // 2)


WHEEL_NAMES = {"fl": "front left", "fr": "front right", "rl": "rear left", "rr": "rear right", "left": "left", "right": "right"}


class Layout:
    """One video frame: the editor's view on the left, the sensors down the right, the map over the editor's corner,
    the robot's numbers along the bottom."""

    def __init__(self, title: str, sensors: list, map_png: bytes | None, map_info: dict | None, size=(1920, 1080)):
        from PIL import Image
        self.size, self.title, self.sensors = size, title, sensors
        self.map = Image.open(io.BytesIO(map_png)).convert("RGB") if map_png else None
        self.map_info = map_info
        self.fonts = {"title": _font(24, True, hangul=True), "label": _font(19, True, hangul=True), "small": _font(17), "mono": _font(18),
                      "big": _font(30, True)}
        if not _has_hangul_font():   # no font here draws Korean: leave it out rather than draw boxes
            self.title = " ".join("".join(c for c in self.title if ord(c) < 0x2E80).split())

    def _label(self, d, at: tuple, text: str) -> None:
        """A pane's caption on a dark plate (an outline around thin strokes smears them)."""
        box = d.textbbox(at, text, font=self.fonts["label"])
        d.rectangle((box[0] - 7, box[1] - 5, box[2] + 7, box[3] + 5), fill=(13, 17, 19))
        d.text(at, text, font=self.fonts["label"], fill=INK)

    def _map(self, robot: dict, trail: list, odometry: list, side: int, span: float = 9.0):
        """The ground map around the robot (``span`` metres across), with its trail, its odometry's and its footprint."""
        from PIL import Image, ImageDraw
        m, out = self.map_info, Image.new("RGB", (side, side), (10, 12, 13))
        k = side / span
        to = lambda x, y: (side / 2 + (x - robot["x"]) * k, side / 2 - (y - robot["y"]) * k)  # noqa: E731
        if self.map is not None and m:
            px = self.map.width / (m["x1"] - m["x0"])          # the picture's top row is the map's far (y1) edge
            box = ((robot["x"] - span / 2 - m["x0"]) * px, (m["y1"] - robot["y"] - span / 2) * px,
                   (robot["x"] + span / 2 - m["x0"]) * px, (m["y1"] - robot["y"] + span / 2) * px)
            out = self.map.crop(tuple(round(v) for v in box)).resize((side, side), Image.NEAREST)
        d = ImageDraw.Draw(out)
        if len(odometry) > 1:
            pts = [to(x, y) for x, y in odometry]
            for a, b in list(zip(pts, pts[1:]))[::2]:
                d.line([a, b], fill=(209, 123, 232), width=2)
        if len(trail) > 1:
            d.line([to(x, y) for x, y in trail], fill=ACCENT, width=3)
        d.polygon([to(x, y) for x, y in robot["footprint"]], outline=BAD if robot["collided"] else INK, fill=None, width=2)
        a = math.radians(robot["yaw_deg"])
        d.line([to(robot["x"], robot["y"]), to(robot["x"] + 0.7 * math.cos(a), robot["y"] + 0.7 * math.sin(a))], fill=ACCENT, width=3)
        d.rectangle([0, 0, side - 1, side - 1], outline=(60, 72, 78), width=2)
        return out

    def frame(self, t: float, editor: np.ndarray, sensor_images: dict, robot: dict, trail: list, odometry: list) -> "Image":
        from PIL import Image, ImageDraw
        W, H = self.size
        canvas = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(canvas)
        f = self.fonts
        d.text((20, 14), self.title, font=f["title"], fill=INK)
        d.text((W - 20, 16), f"t = {t:5.2f} s", font=f["title"], fill=ACCENT, anchor="ra")
        top, bottom = 56, H - 128
        left_w = int(W * 0.615)
        # the editor's view
        box = (16, top, left_w - 8, bottom)
        d.rectangle(box, fill=(0, 0, 0))
        im, (ox, oy) = _fit(editor, box[2] - box[0], box[3] - box[1])
        canvas.paste(im, (box[0] + ox, box[1] + oy))
        self._label(d, (box[0] + ox + 12, box[1] + oy + 10), "Editor camera")
        side = min(300, (box[3] - box[1]) // 3)
        canvas.paste(self._map(robot, trail, odometry, side), (box[0] + ox + 12, box[1] + oy + im.height - side - 12))
        # the sensors
        names = [n for n in self.sensors if n in sensor_images]
        if names:
            x0, x1 = left_w + 8, W - 16
            each = (bottom - top - 12 * (len(names) - 1)) // len(names)
            for i, name in enumerate(names):
                y0 = top + i * (each + 12)
                d.rectangle((x0, y0, x1, y0 + each), fill=(0, 0, 0))
                image, meta = sensor_images[name]
                im, (ox, oy) = _fit(image, x1 - x0, each)
                canvas.paste(im, (x0 + ox, y0 + oy))
                self._label(d, (x0 + ox + 12, y0 + oy + 10), meta)
        # the robot's numbers
        y = bottom + 14
        d.rectangle((16, y - 4, W - 16, H - 12), fill=PANEL)
        speed = math.hypot(robot["vx"], robot["vy"])
        d.text((32, y + 4), f"{speed:4.2f} m/s", font=f["big"], fill=INK)
        d.text((32, y + 44), f"yaw rate {robot['wz']:+5.2f} rad/s", font=f["mono"], fill=MUTED)
        d.text((32, y + 70), f"command  vx {robot['command'][0]:+4.2f}  vy {robot['command'][1]:+4.2f}  w {robot['command'][2]:+4.2f}", font=f["mono"], fill=MUTED)
        wheels = robot["wheels"][:4]
        cols = [("wheel", [WHEEL_NAMES.get(w["name"], w["name"]) for w in wheels])]
        if any(w["steer_deg"] is not None for w in wheels):
            cols.append(("steer", [f"{w['steer_deg']:+6.1f} deg" for w in wheels]))
        cols += [("speed", [f"{w['speed']:+5.2f} m/s" for w in wheels]), ("friction", [f"{w['mu']:4.2f}" for w in wheels]),
                 ("slip", [f"{w['slip']:5.3f} m/s" for w in wheels])]
        x = 470
        for head, values in cols:
            d.text((x, y), head, font=f["small"], fill=MUTED)
            for i, v in enumerate(values):
                hot = head == "slip" and wheels[i]["slip"] > 0.05
                d.text((x, y + 23 + 20 * i), v, font=f["mono"], fill=WARM if hot else INK)
            x += 190 if head == "wheel" else 170
        d.text((x + 20, y), "floor", font=f["small"], fill=MUTED)
        d.text((x + 20, y + 23), str(robot.get("material") or "-")[:34], font=f["mono"], fill=INK)
        d.text((x + 20, y + 49), f"pose  {robot['x']:.2f}, {robot['y']:.2f}  yaw {robot['yaw_deg']:.0f} deg", font=f["mono"], fill=MUTED)
        if robot["collided"]:
            d.text((x + 20, y + 75), "blocked by an obstacle", font=f["mono"], fill=BAD)
        return canvas


def encode(folder: Path, fps: float, out: Path) -> None:
    """The frames of ``folder`` (frame_00000.jpg ...) as an H.264 MP4 any browser plays."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed: the frames are in " + str(folder))
    out.parent.mkdir(parents=True, exist_ok=True)
    run = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps:g}", "-i", str(folder / "frame_%05d.jpg"),
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", "-movflags", "+faststart", str(out)],
                         capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {run.stderr.strip()[-300:]}")


def _wait_for_server(client, timeout: float) -> None:
    """Until the render server answers again (a supervisor restarts one that died), or ``timeout`` seconds."""
    end = time.monotonic() + timeout
    while True:
        try:
            client.info()
            return
        except Exception:  # noqa: BLE001 - still away
            if time.monotonic() > end:
                raise
            time.sleep(3.0)


def render(session, robot, log: list, fps: float = 20.0, spp: int = 1024, sensor_spp: int = 512, sensors=None,
           name: str | None = None, keep_frames: bool = False) -> Path:
    """Render a logged drive (robot.log_stop()) and write it as an MP4; returns its path. ``session``: the viewer's
    (gui.Session), ``robot``: its RobotSession. ``spp``: the editor view's samples a frame (accumulated by the
    session with fireflies held back, then denoised); ``sensor_spp``: a sensor's, in eight passes (clean_render).
    ``sensors``: the sensors in the video (default: all the robot has). The robot's real time is held meanwhile;
    ``robot.recording`` tells the progress. Measured on CS02, unpolarized: the denoiser does not remove fireflies, so
    128 samples leave white specks in every frame; 1024 samples with the hold-back are clean."""
    if len(log) < 2:
        raise ValueError("nothing was logged: start the recording, drive, then stop it")
    world, base = robot.world, robot.robot
    if world is None or base is None:
        raise ValueError("the robot is gone")
    names = [n for n in (sensors if sensors is not None else list(robot.sensors)) if n in robot.sensors]
    t0 = log[0]["t"]
    times = np.arange(0.0, log[-1]["t"] - t0 + 1e-9, 1.0 / fps)
    stamps = np.array([e["t"] - t0 for e in log])
    picks = [int(np.argmin(np.abs(stamps - t))) for t in times]
    out = FOLDER / f"{name or time.strftime('robot_%Y%m%d_%H%M%S')}.mp4"
    spp = max(session.pass_spp, int(spp) - int(spp) % session.pass_spp)
    scan = session.sim.getState()[0].scanId
    layout = Layout(f"OpticalNav Sim  |  {robot.scene}  |  {robot.config.label}", names, robot.map_png, robot.map_info)
    labels = robot.labels()
    kept = {"target": session.target_spp, "denoise": session.sim.denoise, "render_on": session.render_on, "follow": session.follow,
            "cam": session.cam_now, "sensors": {n: robot.sensors[n].state for n in names}}
    frames_dir = Path(tempfile.mkdtemp(prefix="opticalnav_rec_"))
    started = time.monotonic()
    robot.replaying = session.rec_busy = True
    try:
        with session.lock:
            session.follow = None
            session.target_spp, session.render_on = spp, True
            session.sim.setDenoiser(True)
        for n in names:   # each sensor starts fresh: a camera's exposure settles on the first frame, then follows the drive
            robot.sensors[n].state = robot.sensors[n].spec.make_state()
        trail, odometry = [], []

        def frame_of(e: dict):
            """Everything of one video frame, rendered: the robot put back at the entry, the editor's view, the sensors."""
            with robot.lock:
                robot.restore(e)
                robot._bodies = world.bodies()
                robot.pose_seq += 1
                state = robot.state()["robot"]
            # the editor camera of that moment: where it was, or where its following of the robot puts it
            at = (state["x"], state["y"], state["z"], math.radians(state["yaw_deg"]))
            x, y, z, h, el = session.follow_camera(e["follow"], at, e["cam"]) if e["follow"] is not None else e["cam"]
            with session.lock:
                session.sim.teleport([scan], [[x, y, z]], [h], [el])
                session._moved()
            st = session.wait_idle(180.0)
            if st.get("error"):
                raise RuntimeError(f"the editor's view failed: {st['error']}")
            editor = np.asarray(session._tonemapped(session._frame(st["frame_id"]), "rgb", 0.0))
            if editor.dtype != np.uint8:
                editor = np.clip(np.rint(editor * (255.0 if editor.max() <= 1.0 else 1.0)), 0, 255).astype(np.uint8)
            images = {}
            for n in names:
                reading, _, _ = robot.sense(n, True, max(1, int(sensor_spp) // 8), session._view_now())
                images[n] = (reading.picture, "  |  ".join([labels.get(n, n), *(v for k, v in reading.info.items() if k in ("model", "exposure", "range"))]))
            return state, editor, images

        retries = 0
        for k, (t, i) in enumerate(zip(times, picks)):
            if robot.recording.get("cancel"):
                raise RuntimeError("cancelled")
            before = {n: robot.sensors[n].spec.snapshot(robot.sensors[n].state) for n in names}   # a frame done twice meters once
            for attempt in range(6):
                try:
                    state, editor, images = frame_of(log[i])
                    break
                except Exception as exc:  # noqa: BLE001 - the render server went away (it restarts): wait for it, the frame again
                    if attempt == 5 or robot.recording.get("cancel"):
                        raise
                    retries += 1
                    robot.recording.update(waiting=f"{type(exc).__name__}: {str(exc)[:120]}", retries=retries)
                    for n, snap in before.items():
                        robot.sensors[n].spec.restore(robot.sensors[n].state, snap)
                    _wait_for_server(world.client, 600.0)
                    robot.reregister()   # a restarted server knows no body models
                    robot.recording.pop("waiting", None)
            trail.append((state["x"], state["y"]))
            odometry.append(tuple(log[i]["odometry_xy"]))
            layout.frame(float(t), editor, images, state, trail, odometry).save(frames_dir / f"frame_{k:05d}.jpg", quality=94)
            done = k + 1
            robot.recording.update(done=done, frames=len(times), seconds=round(time.monotonic() - started, 1),
                                   eta_s=round((time.monotonic() - started) / done * (len(times) - done), 1))
        robot.recording.update(phase="encoding")
        encode(frames_dir, fps, out)
        return out
    finally:
        for n, state in kept["sensors"].items():
            if n in robot.sensors:
                robot.sensors[n].state = state
        with robot.lock:
            robot.restore(log[-1])   # where the drive ended: the robot's real time goes on from there
            if robot.world is not None:
                robot._bodies = robot.world.bodies()
            robot.pose_seq += 1
        with session.lock:
            session.target_spp, session.render_on, session.follow = kept["target"], kept["render_on"], kept["follow"]
            session.sim.setDenoiser(kept["denoise"])
            x, y, z, h, el = kept["cam"]
            session.sim.teleport([scan], [[x, y, z]], [h], [el])
            if session.follow is not None:
                session._ride()
            session._moved()
        robot.replaying = session.rec_busy = False
        if not keep_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)
