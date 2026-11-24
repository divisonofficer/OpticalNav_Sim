"""Robots from configuration files: a base, and what is mounted on it, read from a file instead of written in code.

    cfg = config.load("ranger_mini_rig")                 # a registered robot's name, or the path of its robot.yaml
    robot = config.spawn(world, cfg, at=world.on_floor(2.0, 1.0, yaw_deg=90.0))
    robot.base, robot.parts["mast"], robot.sensors["hdr"], robot.lights

A robot is a folder with a ``robot.yaml`` (or ``.json``): the structure file. It names the base (its drive model, its
meshes or its URDF with the role of each joint, its numbers) and the parts mounted on it (lights, masts, cameras, any
URDF), each on a link of the base or of another part:

    name: my_robot
    label: My robot
    base:
      type: four_wheel_steering            # a drive model (register_base); "urdf": a body that does not drive itself
      urdf: my_robot.urdf                  # its links and meshes ... or  model: <folder with model.json>
      joints: {steering: [fl_steer, fr_steer, rl_steer, rr_steer], wheels: [fl_wheel, fr_wheel, rl_wheel, rr_wheel]}
      geometry: {wheel_radius: 0.1}        # what a URDF's visuals do not say
      params: {dynamics: true, kerb_mass: 64.5}
    parts:
      - {name: headlight, type: light, link: base_link, mount: {xyz: [0.37, 0.0, 0.2]}, params: {radiance: 40}}
      - {name: mast, type: mast, link: top_plate}
      - {name: front, type: camera, model: lucid_tri032s, parent: mast, link: top, label: Front camera,
         mount: {xyz: [0.1, 0.0, 0.05], rpy_deg: [0, 10, 0]}}

Folders are found in ``$OPTICALNAV_ROBOTS`` (a path list), ``packs/hardware`` (``$OPTICALNAV_HARDWARE``) and the
configurations shipped here (``hw/configs``): putting a folder there registers the robot (``available()``; the viewer
lists them). New kinds of part, base and camera are added with ``register_part`` / ``register_base`` /
``register_camera``: a function from the file's entry to a Spec.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from . import cameras as camera_models
from . import lights, rigs, robots, urdf
from .actor import Actor, Spec, World
from .transform import Transform

BUILTIN = Path(__file__).with_name("configs")
FILES = ("robot.yaml", "robot.yml", "robot.json")


# --- the files ----------------------------------------------------------------------------------------------------
def search_path() -> list[Path]:
    extra = [Path(p) for p in os.environ.get("OPTICALNAV_ROBOTS", "").split(os.pathsep) if p]
    return [*extra, Path(robots.MODELS), BUILTIN]


def _read(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(f"{path} is YAML: install PyYAML, or write the configuration as robot.json") from exc
    data = yaml.safe_load(text) or {}

    def plain(node, where):   # YAML 1.1 reads the bare words on / off / yes / no as true and false, as keys too
        if isinstance(node, dict):
            for k, v in node.items():
                if not isinstance(k, str):
                    raise ValueError(f"{path}: the key {k!r} under {where or 'the top'} is not a word (YAML reads on, off, yes "
                                     "and no as true and false: quote it)")
                plain(v, k)
        elif isinstance(node, list):
            for v in node:
                plain(v, where)
    plain(data, "")
    return data


def _find(name: str) -> Path | None:
    for folder in search_path():
        for cand in [folder / name / f for f in FILES] + [folder / f"{name}{ext}" for ext in (".yaml", ".yml", ".json")]:
            if cand.is_file():
                return cand
    return None


@dataclass
class RobotConfig:
    name: str
    data: dict
    path: Path | None = None       # the structure file: relative paths in it start at its folder

    @property
    def label(self) -> str:
        return str(self.data.get("label") or self.name)

    @property
    def folder(self) -> Path:
        return self.path.parent if self.path is not None else Path.cwd()

    def parts(self) -> list[dict]:
        return [dict(p) for p in self.data.get("parts") or []]

    def sensors(self) -> list[dict]:
        """The parts that deliver data (cameras), as the file lists them: ``[{"name", "label", "type", "model"}]``."""
        out = []
        for p in self.parts():
            if p.get("type") == "camera":
                model = str(p.get("model") or "camera")
                out.append({"name": str(p["name"]), "label": str(p.get("label") or p["name"]), "type": "camera",
                            "model": CAMERA_NAMES.get(model, model)})
        return out

    def summary(self) -> dict:
        base = self.data.get("base") or {}
        return {"name": self.name, "label": self.label, "description": str(self.data.get("description") or ""),
                "base": str(base.get("type") or ""), "path": str(self.path or ""),
                "parts": [{"name": str(p.get("name")), "type": str(p.get("type")), "parent": str(p.get("parent") or "base"),
                           "link": str(p.get("link") or ""), "label": str(p.get("label") or p.get("name"))} for p in self.parts()],
                "sensors": self.sensors()}


def load(source) -> RobotConfig:
    """A robot's configuration: by its registered name, the path of its structure file or folder, or a dict."""
    if isinstance(source, RobotConfig):
        return source
    if isinstance(source, dict):
        return _checked(RobotConfig(str(source.get("name") or "robot"), source))
    path = Path(str(source))
    if path.is_dir():
        path = next((path / f for f in FILES if (path / f).is_file()), path / FILES[0])
    if not path.is_file():
        found = _find(str(source))
        if found is None:
            raise FileNotFoundError(f"no robot {str(source)!r}: not a file, and no robot.yaml of that name in "
                                    f"{', '.join(str(p) for p in search_path())} (have: {', '.join(r['name'] for r in available()) or 'none'})")
        path = found
    data = _read(path)
    default = path.parent.name if path.name in FILES else path.stem
    return _checked(RobotConfig(str(data.get("name") or default), data, path.resolve()))


def _checked(cfg: RobotConfig) -> RobotConfig:
    base = cfg.data.get("base")
    if not isinstance(base, dict) or not base.get("type"):
        raise ValueError(f"{cfg.name}: the configuration needs a base with a type ({', '.join(sorted(BASES))})")
    if base["type"] not in BASES:
        raise ValueError(f"{cfg.name}: no base type {base['type']!r} (have: {', '.join(sorted(BASES))})")
    names = ["base"]
    for p in cfg.parts():
        if not p.get("name") or p.get("type") not in PARTS:
            raise ValueError(f"{cfg.name}: a part needs a name and a type of {', '.join(sorted(PARTS))}: {p}")
        if p["name"] in names:
            raise ValueError(f"{cfg.name}: two parts are named {p['name']!r}")
        if str(p.get("parent") or "base") not in names:
            raise ValueError(f"{cfg.name}: part {p['name']!r} is mounted on {p.get('parent')!r}, which is not listed before it")
        names.append(p["name"])
    return cfg


def available() -> list[dict]:
    """Every robot the search path holds (the first of a name wins): their summaries."""
    out, seen = [], set()
    for folder in search_path():
        if not folder.is_dir():
            continue
        files = [f for d in sorted(folder.iterdir()) if d.is_dir() for f in (d / n for n in FILES) if f.is_file()]
        files += [f for f in sorted(folder.iterdir()) if f.is_file() and f.suffix in (".yaml", ".yml", ".json") and f.name not in FILES
                  and f.name != "model.json"]
        for f in files:
            try:
                cfg = load(f)
            except Exception:  # noqa: BLE001 - a file that is not a robot (or a broken one) is not listed
                continue
            if cfg.name not in seen:
                seen.add(cfg.name)
                out.append(cfg.summary())
    return out


# --- what the entries become ----------------------------------------------------------------------------------------
BASES: dict = {}       # type -> (cfg, base entry) -> Spec
PARTS: dict = {}       # type -> (cfg, part entry) -> Spec
CAMERAS: dict = {}     # model -> (**params) -> CameraSpec
CAMERA_NAMES: dict = {}
SENSORS = {"imx490": camera_models.IMX490, "imx265": camera_models.IMX265, "imx250myr": camera_models.IMX250MYR}
LENSES = {"fujinon_hf6xa_5m": camera_models.FujinonHF6XA5M}


def register_base(kind: str, build) -> None:
    BASES[kind] = build


def register_part(kind: str, build) -> None:
    PARTS[kind] = build


def register_camera(model: str, factory, name: str | None = None) -> None:
    """A camera model a configuration names (``model: <model>``): ``factory(**params)`` gives its CameraSpec."""
    CAMERAS[model] = factory
    CAMERA_NAMES[model] = name or model


def _transform(entry) -> Transform:
    entry = entry or {}
    return Transform(tuple(float(v) for v in entry.get("xyz", (0.0, 0.0, 0.0))),
                     tuple(float(v) for v in entry.get("rpy_deg", (0.0, 0.0, 0.0))))


def _params(spec_class, entry: dict, where: str) -> dict:
    """The entry's ``params`` as the dataclass's fields (lists become tuples): an unknown one is named."""
    known = {f.name for f in fields(spec_class)}
    out = {}
    for k, v in (entry.get("params") or {}).items():
        if k not in known:
            raise ValueError(f"{where}: no parameter {k!r} (has: {', '.join(sorted(known))})")
        out[k] = tuple(v) if isinstance(v, list) else v
    return out


def _path(cfg: RobotConfig, name: str, also: Path | None = None) -> Path:
    """A path of the file: as written when absolute, else beside the structure file (or in ``also``)."""
    p = Path(os.path.expandvars(os.path.expanduser(str(name))))
    if p.is_absolute():
        return p
    here = cfg.folder / p
    return here if here.exists() or also is None else also / p


def fws_from_urdf(model: urdf.UrdfModel, joints: dict, geometry: dict, params: dict) -> robots.RangerMini2:
    """A four-wheel-steering base drawn by a URDF. ``joints``: the URDF's steering and wheel joints, each list in the
    order front left, front right, rear left, rear right. Wheelbase and track are measured between the wheel joints;
    ``geometry`` gives what the visuals do not say (``wheel_radius``; optionally wheel_width, size, clearance)."""
    for role in ("steering", "wheels"):
        if len(joints.get(role) or ()) != 4:
            raise ValueError(f"{model.name}: joints.{role} lists four joints (front left, front right, rear left, rear right)")
    frames = model.frames()
    wheel_links = [model.joint(n).child for n in joints["wheels"]]
    steer_links = [model.joint(n).child for n in joints["steering"]]
    c = np.array([frames[link][:3, 3] for link in wheel_links])
    wheelbase = float((c[0, 0] + c[1, 0] - c[2, 0] - c[3, 0]) / 2.0)
    track = float((c[0, 1] + c[2, 1] - c[1, 1] - c[3, 1]) / 2.0)
    if wheelbase <= 0.0 or track <= 0.0:
        raise ValueError(f"{model.name}: the wheel joints are not in the order front left, front right, rear left, rear right "
                         f"(measured wheelbase {wheelbase:.3f} m, track {track:.3f} m)")
    if "wheel_radius" not in geometry:
        raise ValueError(f"{model.name}: geometry.wheel_radius is needed (a URDF's visuals do not give it)")
    radius = float(geometry["wheel_radius"])
    width = float(geometry.get("wheel_width", 0.6 * radius))
    size = tuple(geometry.get("size") or (wheelbase + 2.4 * radius, track + width + 0.06, 3.4 * radius))
    spec = robots.RangerMini2(wheelbase=wheelbase, track=track, wheel_radius=radius, wheel_width=width, size=tuple(float(v) for v in size),
                              clearance=float(geometry.get("clearance", radius)), **params)
    # the URDF's root frame in this base's frame (on the floor under the wheels' centre), then each link's visuals in
    # the frame of the base's link it moves with: the body, a steering module, a wheel
    to_base = Transform((-float(c[:, 0].mean()), -float(c[:, 1].mean()), radius - float(c[:, 2].mean()))).matrix()
    parent = {j.child: j.parent for j in model.joints.values()}
    ours = spec.link_frames(None)

    def moves_with(link: str) -> str:
        at = link
        while at is not None:
            if at in wheel_links:
                return f"wheel_{robots.WHEELS[wheel_links.index(at)]}"
            if at in steer_links:
                return f"steer_{robots.WHEELS[steer_links.index(at)]}"
            at = parent.get(at)
        return "base_link"
    drawn: dict = {}
    for link, visuals in model.visuals.items():
        own = moves_with(link)
        place = np.linalg.inv(ours[own]) @ to_base @ frames[link]
        drawn.setdefault(own, []).extend(
            robots.Visual(v.shape, v.size, Transform.from_matrix(place @ v.origin.matrix()), v.material, v.mesh, v.scale) for v in visuals)
    spec.link_visuals = drawn
    return spec


def _four_wheel_steering(cfg: RobotConfig, entry: dict) -> Spec:
    params = _params(robots.RangerMini2, entry, f"{cfg.name}.base")
    if entry.get("urdf"):
        return fws_from_urdf(urdf.load(_path(cfg, entry["urdf"])), entry.get("joints") or {}, entry.get("geometry") or {}, params)
    geometry = {k: (tuple(v) if isinstance(v, list) else v) for k, v in (entry.get("geometry") or {}).items()}
    model = entry.get("model")
    folder = _path(cfg, model, Path(robots.MODELS)) if model else None
    if folder is not None and (folder / "model.json").is_file():
        spec = robots.RangerMini2.from_model(folder, **params)
        for k, v in geometry.items():
            setattr(spec, k, v)
        return spec
    return robots.RangerMini2(**{**geometry, **params})   # no meshes at hand: placeholder shapes of the same size


@dataclass
class UrdfBody(Spec):
    """A URDF drawn as it is, its joints held at ``positions`` (name -> rad or m): a bracket, a tripod, an arm at rest.
    Its links are the URDF's: parts mount on them by name."""
    kind = "structure"
    urdf: urdf.UrdfModel = None
    positions: dict = field(default_factory=dict)
    model: str = "urdf"

    def link_frames(self, state) -> dict:
        return self.urdf.frames(self.positions)

    def visuals(self) -> dict:
        return {link: list(vs) for link, vs in self.urdf.visuals.items() if vs}


def _urdf_body(cfg: RobotConfig, entry: dict) -> Spec:
    if not entry.get("urdf"):
        raise ValueError(f"{cfg.name}: a part or base of type urdf names its file (urdf: ...)")
    model = urdf.load(_path(cfg, entry["urdf"]))
    return UrdfBody(urdf=model, positions=dict(entry.get("positions") or {}), model=f"urdf_{model.name}")


def _camera(cfg: RobotConfig, entry: dict) -> Spec:
    model = str(entry.get("model") or "")
    if model not in CAMERAS:
        raise ValueError(f"{cfg.name}.{entry.get('name')}: no camera model {model!r} (have: {', '.join(sorted(CAMERAS))}; "
                         "others: hw.config.register_camera)")
    kw = dict(entry.get("params") or {})
    lens = entry.get("lens")
    if lens is not None:   # a registered lens's name, or its numbers (hw.cameras.LensSpec)
        kw["lens"] = LENSES[lens]() if isinstance(lens, str) else camera_models.LensSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in lens.items()})
    sensor = entry.get("sensor")
    spec = CAMERAS[model](**kw)
    if sensor is not None:  # another sensor behind the same housing: a registered one's name, or its numbers
        spec.sensor = SENSORS[sensor] if isinstance(sensor, str) else camera_models.SensorSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in sensor.items()})
        spec.__post_init__()
    return spec


register_base("four_wheel_steering", _four_wheel_steering)
register_base("urdf", _urdf_body)
register_part("light", lambda cfg, e: lights.AreaLight(**_params(lights.AreaLight, e, f"{cfg.name}.{e['name']}")))
register_part("mast", lambda cfg, e: rigs.Mast(**_params(rigs.Mast, e, f"{cfg.name}.{e['name']}")))
register_part("camera", _camera)
register_part("urdf", _urdf_body)
register_camera("lucid_tdr054s", camera_models.LucidTDR054S, "LUCID TDR054S")
register_camera("lucid_tri032s", camera_models.LucidTRI032S, "LUCID TRI032S")
register_camera("flir_bfs_51s5pc", camera_models.FlirBFS51S5PC, "FLIR BFS-U3-51S5PC-C")


# --- building -------------------------------------------------------------------------------------------------------
def base_spec(source) -> Spec:
    """The base of a configuration, not spawned: its footprint and numbers, to look for a place for it."""
    cfg = load(source)
    return BASES[cfg.data["base"]["type"]](cfg, cfg.data["base"])


@dataclass
class Robot:
    """A configuration spawned into a world: its base, every part by name, and its sensors and lights among them."""
    config: RobotConfig
    base: Actor
    parts: dict = field(default_factory=dict)
    sensors: dict = field(default_factory=dict)
    lights: dict = field(default_factory=dict)
    calibrated: bool = False       # the cameras stand as a measured calibration says

    def labels(self) -> dict:
        return {s["name"]: s["label"] for s in self.config.sensors()}


def spawn(world: World, source, at: Transform | None = None, name: str | None = None, calibration=None) -> Robot:
    """Spawn a configured robot: the base at ``at``, each part on its link. ``calibration``: a measured rig calibration
    (OpenCV YAML) for the cameras that name an entry of it (``calibrated: {camera, size}``), instead of the file the
    configuration names; without one that exists, those cameras stand at their ``mount`` with their datasheet lens."""
    cfg = load(source)
    name = name or cfg.name
    entries = cfg.parts()
    # every spec before anything is spawned: a wrong entry leaves the world as it was
    base_spec = BASES[cfg.data["base"]["type"]](cfg, cfg.data["base"])
    specs = {e["name"]: PARTS[e["type"]](cfg, e) for e in entries}
    cal_entry = cfg.data.get("calibration") or {}
    cal_file = calibration or cal_entry.get("file")
    cal_path = _path(cfg, cal_file) if cal_file else None
    measured, mounts = None, {}
    wanted = {e["name"]: e["calibrated"] for e in entries if e["type"] == "camera" and e.get("calibrated")}
    if wanted and cal_path is not None and cal_path.is_file():
        from .calibration import RigCalibration
        measured = RigCalibration.load(cal_path)
        ref = cal_entry.get("reference") or {}
        by_entry = rigs.camera_mounts(measured, [w["camera"] for w in wanted.values()], _transform(ref.get("mount")))
        for part, w in wanted.items():
            size = w.get("size") or (specs[part].sensor.width, specs[part].sensor.height)
            specs[part].calibration = measured.cameras[w["camera"]].intrinsics(int(size[0]), int(size[1]))
            mounts[part] = (str(ref.get("parent") or "base"), ref.get("link"), by_entry[w["camera"]])
    robot = Robot(cfg, world.spawn(base_spec, at=at, name=name), calibrated=measured is not None)
    actors = {"base": robot.base}
    try:
        for e in entries:
            on, link, mount = mounts.get(e["name"], (str(e.get("parent") or "base"), e.get("link"), _transform(e.get("mount"))))
            actor = world.spawn(specs[e["name"]], attach_to=actors[on], link=link, mount=mount, name=f"{name}_{e['name']}")
            actors[e["name"]] = robot.parts[e["name"]] = actor
            if e["type"] == "camera":
                robot.sensors[e["name"]] = actor
            elif e["type"] == "light":
                robot.lights[e["name"]] = actor
    except Exception:
        world.destroy(robot.base)   # what was spawned goes with its base
        raise
    return robot
