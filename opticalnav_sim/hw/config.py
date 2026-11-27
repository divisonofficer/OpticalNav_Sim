"""Robots from configuration files: a base, and what is mounted on it, read from a file instead of written in code.

    cfg = config.load("turtlebot4", rig="hdr_12bit_compact")      # a robot, and the equipment on its deck
    robot = config.spawn(world, cfg, at=world.on_floor(2.0, 1.0, yaw_deg=90.0))
    robot.base, robot.parts["frame"], robot.sensors["hdr"], robot.lights

Two kinds of file. A **robot** (``robot.yaml`` in a folder of its name) is a base and what belongs to it; a **rig**
(``rigs/<name>.yaml``) is equipment, a structure with its cameras, that goes on any robot's ``deck``:

    name: my_robot                                    # rigs/my_rig.yaml
    label: My robot                                   kind: rig
    base:                                             label: Two cameras on a frame
      type: differential        # a drive type        mass: 1.2
      params: {track: 0.233, ...}                     parts:
      mounts: {deck: {link: top_plate}}                 - {name: frame, type: frame, params: {bars: [...], points: {top: {xyz: [0, 0, 0.3]}}}}
    parts:                                              - {name: hdr, type: camera, model: lucid_tdr054s, parent: frame, link: top}
      - {name: lidar, type: lidar, link: deck}
    rig: my_rig                                       # the rig it has unless another is asked for

``extends: other_robot`` starts from another file: its ``base`` with these values over it, its parts (a part of the same
name is changed, a new one added, ``remove: [names]`` takes some out). ``edits: {part: {...}}`` changes parts after the rig
is on: how a robot's own file (or a caller) adjusts a rig. A part is on the base's ``deck`` unless ``parent`` (another
part) and ``link`` say otherwise. An unknown key anywhere is an error that names the nearest known one.

Files are found in ``$OPTICALNAV_ROBOTS`` (a path list), ``packs/hardware`` (``$OPTICALNAV_HARDWARE``) and the ones shipped
(``hw/configs``), robots as ``<name>/robot.yaml`` and rigs as ``rigs/<name>.yaml``. New kinds are one function each, from the
file's entry to a Spec: ``register_base``, ``register_part``, ``register_sensor``, ``register_camera``.
"""
from __future__ import annotations

import copy
import difflib
import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from . import cameras as camera_models
from . import arm, drive, lights, ouster, rigs, robots, sensing, urdf
from .actor import Actor, Spec, World
from .isp import Isp
from .sensor import ExposureControl
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
    plain_keys(data, str(path))
    return data


def plain_keys(node, where: str) -> None:
    """Every key is a word: YAML 1.1 reads the bare words on / off / yes / no as true and false, as keys too."""
    if isinstance(node, dict):
        for k, v in node.items():
            if not isinstance(k, str):
                raise ValueError(f"{where}: the key {k!r} is not a word (YAML reads on, off, yes and no as true and false: quote it)")
            plain_keys(v, where)
    elif isinstance(node, list):
        for v in node:
            plain_keys(v, where)


def _find(name: str) -> Path | None:
    for folder in search_path():
        for cand in [folder / name / f for f in FILES] + [folder / f"{name}{ext}" for ext in (".yaml", ".yml", ".json")]:
            if cand.is_file():
                return cand
    return None


@dataclass
class RobotConfig:
    """A robot's structure as it will be built: ``data`` has ``extends``, the rig and the edits worked in. ``rig``: the
    name of the rig on it ("" for none)."""
    name: str
    data: dict
    path: Path | None = None       # the structure file: relative paths in it start at its folder
    rig: str = ""

    @property
    def label(self) -> str:
        return str(self.data.get("label") or self.name)

    @property
    def folder(self) -> Path:
        return self.path.parent if self.path is not None else Path.cwd()

    def parts(self) -> list[dict]:
        return [dict(p) for p in self.data.get("parts") or []]

    def sensors(self) -> list[dict]:
        """The parts that measure, as the file lists them: ``[{"name", "label", "type", "model"}]``."""
        return [{"name": str(p["name"]), "label": str(p.get("label") or p["name"]), "type": str(p["type"]), "model": _model_of(p)}
                for p in self.parts() if p.get("type") in SENSOR_TYPES]

    def carried_mass(self) -> float:
        return float(sum(float(p.get("mass", 0.0)) for p in self.parts()) + float(self.data.get("rig_mass", 0.0)))

    def warnings(self) -> list[str]:
        """What is built but not as the real robot could be."""
        out = []
        limit = ((self.data.get("base") or {}).get("params") or {}).get("max_payload")
        if limit is not None and self.carried_mass() > float(limit):
            out.append(f"{self.carried_mass():g} kg on it, over its {float(limit):g} kg")
        return out

    def summary(self) -> dict:
        base = self.data.get("base") or {}
        return {"name": self.name, "label": self.label, "description": str(self.data.get("description") or ""),
                "base": str(base.get("type") or ""), "path": str(self.path or ""), "rig": self.rig, "rig_label": str(self.data.get("rig_label") or ""),
                "mass": round(self.carried_mass(), 2), "warnings": self.warnings(),
                "parts": [{"name": str(p.get("name")), "type": str(p.get("type")), "parent": str(p.get("parent") or "base"),
                           "link": str(p.get("link") or ("deck" if not p.get("parent") else "")), "label": str(p.get("label") or p.get("name")),
                           "rig": bool(p.get("_rig"))} for p in self.parts()],
                "sensors": self.sensors()}


# --- checks -----------------------------------------------------------------------------------------------------------
TOP_KEYS = ("name", "label", "description", "base", "parts", "calibration", "extends", "remove", "rig", "edits")
RIG_KEYS = ("kind", "name", "label", "description", "mass", "parts", "calibration")
PART_KEYS = ("name", "type", "label", "parent", "link", "mount", "mass", "params")
BASE_KEYS = ("type", "params", "mounts", "visuals", "mass")
MOUNT_KEYS = ("xyz", "rpy_deg")


def _known(entry, keys, where: str) -> None:
    """An entry has no key but these (and the marks this module leaves, which start with "_")."""
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: expected a mapping, got {entry!r}")
    for k in entry:
        if str(k).startswith("_") or k in keys:
            continue
        near = difflib.get_close_matches(str(k), [str(x) for x in keys], 1)
        raise ValueError(f"{where}: unknown key {k!r}" + (f" (did you mean {near[0]!r}?)" if near else f" (has: {', '.join(sorted(map(str, keys)))})"))


def _merge(old, new):
    """``new`` over ``old``: mappings key by key, anything else replaced."""
    if not (isinstance(old, dict) and isinstance(new, dict)):
        return copy.deepcopy(new)
    out = copy.deepcopy(old)
    for k, v in new.items():
        out[k] = _merge(out[k], v) if k in out else copy.deepcopy(v)
    return out


def _merge_parts(old: list, new: list) -> list:
    out = [copy.deepcopy(p) for p in old]
    at = {p.get("name"): i for i, p in enumerate(out)}
    for p in new:
        if p.get("name") in at:
            out[at[p["name"]]] = _merge(out[at[p["name"]]], p)
        else:
            out.append(copy.deepcopy(p))
    return out


def rig_files() -> dict:
    """name -> file of every rig on the search path (the first of a name wins)."""
    out: dict = {}
    for folder in search_path():
        for f in sorted((folder / "rigs").glob("*")) if (folder / "rigs").is_dir() else ():
            if f.suffix in (".yaml", ".yml", ".json"):
                out.setdefault(f.stem, f)
    return out


def load_rig(source) -> tuple[str, dict, Path | None]:
    """(name, data, file) of a rig: a registered name, a file, or a mapping."""
    if isinstance(source, dict):
        name, data, path = str(source.get("name") or "rig"), copy.deepcopy(source), None
    else:
        path = Path(str(source))
        if not path.is_file():
            files = rig_files()
            if str(source) not in files:
                near = difflib.get_close_matches(str(source), list(files), 1)
                raise FileNotFoundError(f"no rig {str(source)!r}" + (f" (did you mean {near[0]!r}?)" if near else f" (have: {', '.join(sorted(files)) or 'none'})"))
            path = files[str(source)]
        data, name = _read(path), path.stem
    _known(data, RIG_KEYS, f"rig {name}")
    if data.get("kind") != "rig":
        raise ValueError(f"rig {name}: the file says kind: {data.get('kind')!r}, not rig")
    for p in data.get("parts") or []:
        _check_part(p, f"rig {name}")
    return name, data, path


def _check_part(p: dict, where: str) -> None:
    if not isinstance(p, dict) or not p.get("name") or p.get("type") not in PARTS:
        have = ", ".join(sorted(PARTS))
        raise ValueError(f"{where}: a part needs a name and a type of {have}: {p}")
    _known(p, (*PART_KEYS, *PART_EXTRA.get(p["type"], ())), f"{where}.{p['name']}")
    if p.get("mount") is not None:
        _known(p["mount"], MOUNT_KEYS, f"{where}.{p['name']}.mount")


def load(source, rig=None, edits: dict | None = None) -> RobotConfig:
    """A robot's configuration: by its registered name, the path of its structure file or folder, or a mapping. ``rig``:
    another rig than the file's (a name, a file or a mapping; "" or "none": no rig). ``edits``: {part: changes} on top."""
    if isinstance(source, RobotConfig):
        return source
    data, path = _raw(source)
    name = str(data.get("name") or (path.parent.name if path is not None and path.name in FILES else path.stem if path is not None else "robot"))
    return _resolve(name, data, path, rig, edits)


def _raw(source) -> tuple[dict, Path | None]:
    if isinstance(source, dict):
        return copy.deepcopy(source), None
    path = Path(str(source))
    if path.is_dir():
        path = next((path / f for f in FILES if (path / f).is_file()), path / FILES[0])
    if not path.is_file():
        found = _find(str(source))
        if found is None:
            raise FileNotFoundError(f"no robot {str(source)!r}: not a file, and no robot.yaml of that name in "
                                    f"{', '.join(str(p) for p in search_path())} (have: {', '.join(r['name'] for r in available()) or 'none'})")
        path = found
    return _read(path), path.resolve()


def _extended(name: str, data: dict, path: Path | None, seen: tuple = ()) -> dict:
    """``data`` over the file it extends. Every part remembers the folder its relative paths start at."""
    _known(data, TOP_KEYS, name)
    own = copy.deepcopy(data)
    for p in own.get("parts") or []:
        if isinstance(p, dict) and path is not None:
            p.setdefault("_folder", str(path.parent))
    if path is not None and isinstance(own.get("base"), dict):
        own["base"].setdefault("_folder", str(path.parent))
    parent = own.pop("extends", None)
    if not parent:
        own.pop("remove", None)
        return own
    if str(parent) in seen:
        raise ValueError(f"{name}: extends goes in a circle ({' -> '.join((*seen, str(parent)))})")
    raw, parent_path = _raw(parent)
    if "type" in (own.get("base") or {}) and own["base"]["type"] != (raw.get("base") or {}).get("type", own["base"]["type"]):
        raise ValueError(f"{name}: extends {parent}, whose base is {raw['base'].get('type')!r}, with another base type {own['base']['type']!r}: "
                         "write that robot's file instead")
    old = _extended(str(parent), raw, parent_path, (*seen, str(parent)))
    drop = own.pop("remove", None) or []
    for d in drop:
        if d not in [p.get("name") for p in old.get("parts") or []]:
            raise ValueError(f"{name}: remove names {d!r}, which {parent} has not")
    out = _merge({k: v for k, v in old.items() if k not in ("parts", "name")}, {k: v for k, v in own.items() if k != "parts"})
    out["parts"] = _merge_parts([p for p in old.get("parts") or [] if p.get("name") not in drop], own.get("parts") or [])
    return out


def _resolve(name: str, data: dict, path: Path | None, rig, edits) -> RobotConfig:
    data = _extended(name, data, path)
    base = data.get("base")
    if not isinstance(base, dict) or not base.get("type"):
        raise ValueError(f"{name}: the configuration needs a base with a type ({', '.join(sorted(BASES))})")
    if base["type"] not in BASES:
        near = difflib.get_close_matches(str(base["type"]), list(BASES), 1)
        raise ValueError(f"{name}: no base type {base['type']!r}" + (f" (did you mean {near[0]!r}?)" if near else f" (have: {', '.join(sorted(BASES))})"))
    _known(base, (*BASE_KEYS, *BASE_EXTRA.get(base["type"], ())), f"{name}.base")
    for m, at in (base.get("mounts") or {}).items():
        _known(at, ("link", "mount"), f"{name}.base.mounts.{m}")
        _known(at.get("mount") or {}, MOUNT_KEYS, f"{name}.base.mounts.{m}.mount")
    _known(data.get("calibration") or {}, ("file", "reference"), f"{name}.calibration")
    for p in data.get("parts") or []:
        _check_part(p, name)
    # the rig: its parts after the robot's own
    wanted = data.pop("rig", None) if rig is None else (data.pop("rig", None), rig)[1]
    rig_name = ""
    if wanted and wanted != "none":
        rig_name, r, rig_path = load_rig(wanted)
        for p in copy.deepcopy(r.get("parts") or []):
            if p["name"] in [q["name"] for q in data.get("parts") or []]:
                raise ValueError(f"{name}: the rig {rig_name} and the robot both have a part {p['name']!r}")
            p["_rig"] = True
            if rig_path is not None:
                p["_folder"] = str(rig_path.parent)
            data.setdefault("parts", []).append(p)
        if r.get("calibration"):
            _known(r["calibration"], ("file", "reference"), f"rig {rig_name}.calibration")
            data["calibration"] = copy.deepcopy(r["calibration"])
        data["rig_mass"], data["rig_label"] = float(r.get("mass", 0.0)), str(r.get("label") or rig_name)
    for part, change in {**(data.pop("edits", None) or {}), **(edits or {})}.items():
        at = next((i for i, p in enumerate(data.get("parts") or []) if p["name"] == part), None)
        if at is None:
            if (edits and part in edits) or rig is None:
                raise ValueError(f"{name}: edits change {part!r}, which is not a part of it")
            continue   # the file's edits of its own rig, and another rig (or none) is on
        if change is None:
            data["parts"].pop(at)
            continue
        data["parts"][at] = _merge(data["parts"][at], change)
        _check_part(data["parts"][at], f"{name}.edits")
    cfg = RobotConfig(name, data, path, rig_name)
    names = ["base"]
    for p in cfg.parts():
        if p["name"] in names:
            raise ValueError(f"{name}: two parts are named {p['name']!r}")
        if str(p.get("parent") or "base") not in names:
            raise ValueError(f"{name}: part {p['name']!r} is mounted on {p.get('parent')!r}, which is not listed before it")
        names.append(p["name"])
    return cfg


def available() -> list[dict]:
    """Every robot the search path holds (the first of a name wins): their summaries. A file that does not load is listed
    with its ``error``."""
    out, seen = [], set()
    for folder in search_path():
        if not folder.is_dir():
            continue
        files = [f for d in sorted(folder.iterdir()) if d.is_dir() for f in (d / n for n in FILES) if f.is_file()]
        files += [f for f in sorted(folder.iterdir()) if f.is_file() and f.suffix in (".yaml", ".yml", ".json") and f.name not in FILES
                  and f.name != "model.json"]
        for f in files:
            fallback = f.parent.name if f.name in FILES else f.stem
            try:
                summary = load(f).summary()
            except Exception as exc:  # noqa: BLE001 - a broken file is listed with what is wrong in it
                summary = {"name": fallback, "label": fallback, "error": str(exc), "path": str(f), "sensors": [], "parts": []}
            if summary["name"] not in seen:
                seen.add(summary["name"])
                out.append(summary)
    return out


def available_rigs() -> list[dict]:
    out = []
    for name, f in sorted(rig_files().items()):
        try:
            _, data, _ = load_rig(f)
            out.append({"name": name, "label": str(data.get("label") or name), "description": str(data.get("description") or ""),
                        "mass": float(data.get("mass", 0.0)), "path": str(f)})
        except Exception as exc:  # noqa: BLE001
            out.append({"name": name, "label": name, "error": str(exc), "path": str(f)})
    return out


# --- what the entries become ----------------------------------------------------------------------------------------
BASES: dict = {}       # type -> (cfg, base entry) -> Spec
BASE_EXTRA: dict = {}  # type -> the keys of its own an entry may have
PARTS: dict = {}       # type -> (cfg, part entry) -> Spec
PART_EXTRA: dict = {}
SENSOR_TYPES: set = set()   # the part types that measure
CAMERAS: dict = {}     # model -> (**params) -> CameraSpec
CAMERA_NAMES: dict = {}
SENSORS = {"imx490": camera_models.IMX490, "imx265": camera_models.IMX265, "imx250myr": camera_models.IMX250MYR}
LENSES = {"fujinon_hf6xa_5m": camera_models.FujinonHF6XA5M}


def register_base(kind: str, build, keys=()) -> None:
    """A base type: ``build(cfg, entry) -> Spec``. ``keys``: the entry's own keys (type, params, mounts, visuals aside)."""
    BASES[kind], BASE_EXTRA[kind] = build, tuple(keys)


def register_part(kind: str, build, keys=()) -> None:
    PARTS[kind], PART_EXTRA[kind] = build, tuple(keys)


def register_sensor(kind: str, build, keys=()) -> None:
    """A part that measures: its spec is a hw.sensing.Sensor."""
    register_part(kind, build, keys)
    SENSOR_TYPES.add(kind)


def _model_of(entry: dict) -> str:
    model = str(entry.get("model") or (entry.get("params") or {}).get("model") or entry.get("type"))
    return CAMERA_NAMES.get(model, model)


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
    _known(entry.get("params") or {}, sorted(known), f"{where}.params")
    return {k: tuple(v) if isinstance(v, list) else v for k, v in (entry.get("params") or {}).items()}


def _path(cfg: RobotConfig, name: str, also: Path | None = None, entry: dict | None = None) -> Path:
    """A path of the file: as written when absolute, else beside the file the entry comes from (or in ``also``)."""
    p = Path(os.path.expandvars(os.path.expanduser(str(name))))
    if p.is_absolute():
        return p
    here = Path((entry or {}).get("_folder") or cfg.folder) / p
    return here if here.exists() or also is None else also / p


def fws_from_urdf(model: urdf.UrdfModel, joints: dict, geometry: dict, params: dict, spec_class=drive.FourWheelSteering) -> drive.FourWheelSteering:
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
    spec = spec_class(wheelbase=wheelbase, track=track, wheel_radius=radius, wheel_width=width, size=tuple(float(v) for v in size),
                              clearance=float(geometry.get("clearance", radius)), **params)
    # the URDF's root frame in this base's frame: on the floor under the wheels' centre
    to_base = Transform((-float(c[:, 0].mean()), -float(c[:, 1].mean()), radius - float(c[:, 2].mean()))).matrix()
    moving = {link: f"wheel_{n}" for link, n in zip(wheel_links, robots.WHEELS)}
    moving.update({link: f"steer_{n}" for link, n in zip(steer_links, robots.WHEELS)})
    spec.link_visuals = _urdf_visuals(model, spec, to_base, moving)
    return spec


def diff_from_urdf(model: urdf.UrdfModel, joints: dict, geometry: dict, params: dict) -> drive.DifferentialDrive:
    """A differential base drawn by a URDF. ``joints.wheels``: the left and the right wheel joint; the track is measured
    between them. ``geometry``: ``wheel_radius`` (and wheel_width, size, clearance)."""
    if len(joints.get("wheels") or ()) != 2:
        raise ValueError(f"{model.name}: joints.wheels lists two joints (left, right)")
    if "wheel_radius" not in geometry:
        raise ValueError(f"{model.name}: geometry.wheel_radius is needed (a URDF's visuals do not give it)")
    frames = model.frames()
    links = [model.joint(n).child for n in joints["wheels"]]
    c = np.array([frames[link][:3, 3] for link in links])
    if c[0, 1] <= c[1, 1]:
        raise ValueError(f"{model.name}: the wheel joints are not in the order left, right")
    radius = float(geometry["wheel_radius"])
    kw = {"track": float(c[0, 1] - c[1, 1]), **{k: (tuple(v) if isinstance(v, list) else v) for k, v in geometry.items()}, **params}
    spec = drive.DifferentialDrive(**kw)
    to_base = Transform((-float(c[:, 0].mean()), -float(c[:, 1].mean()), radius - float(c[:, 2].mean()))).matrix()
    spec.link_visuals = _urdf_visuals(model, spec, to_base, dict(zip(links, ("wheel_left", "wheel_right"))))
    return spec


def _urdf_visuals(model: urdf.UrdfModel, spec, to_base: np.ndarray, moving: dict) -> dict:
    """The URDF's visuals in the frames of the base's links. ``moving``: URDF link -> the base's link it is; a link
    hanging from one of those moves with it, any other with ``base_link``."""
    frames, parent, ours = model.frames(), {j.child: j.parent for j in model.joints.values()}, spec.link_frames(None)

    def moves_with(link: str) -> str:
        at = link
        while at is not None:
            if at in moving:
                return moving[at]
            at = parent.get(at)
        return "base_link"
    drawn: dict = {}
    for link, visuals in model.visuals.items():
        own = moves_with(link)
        place = np.linalg.inv(ours[own]) @ to_base @ frames[link]
        drawn.setdefault(own, []).extend(
            robots.Visual(v.shape, v.size, Transform.from_matrix(place @ v.origin.matrix()), v.material, v.mesh, v.scale) for v in visuals)
    return drawn


VISUAL_KEYS = ("link", "shape", "size", "xyz", "rpy_deg", "axis", "material", "color", "roughness", "mesh", "scale")


def _file_visuals(cfg: RobotConfig, entry: dict) -> dict | None:
    """A base's ``visuals``: shapes of its body, when there is no mesh. ``axis: z`` stands a cylinder up."""
    if not entry.get("visuals"):
        return None
    out: dict = {}
    for k, v in enumerate(entry["visuals"]):
        where = f"{cfg.name}.base.visuals[{k}]"
        _known(v, VISUAL_KEYS, where)
        if v.get("shape") not in ("box", "cylinder", "profile", "mesh"):
            raise ValueError(f"{where}: shape is box, cylinder, profile or mesh")
        at = _transform(v)
        if v.get("axis") == "z":
            at = Transform.from_matrix(at.matrix() @ Transform((0, 0, 0), (90.0, 0.0, 0.0)).matrix())
        mesh = str(urdf.drawable_mesh(_path(cfg, v["mesh"], entry=entry))) if v.get("mesh") else None
        out.setdefault(str(v.get("link", "base_link")), []).append(robots.Visual(
            v["shape"], tuple(float(x) for x in v.get("size", ())), at,
            (str(v.get("material", "plastic")), tuple(float(x) for x in v.get("color", (0.1, 0.1, 0.1))), float(v.get("roughness", 0.4))), mesh,
            tuple(float(x) for x in v.get("scale", (1.0, 1.0, 1.0)))))
    return out


def _wheeled(spec_class, from_urdf):
    """A base of a wheeled drive type: its numbers from ``params``, drawn by a URDF, by ``visuals``, or as placeholders."""
    def build(cfg: RobotConfig, entry: dict) -> Spec:
        params = _params(spec_class, entry, f"{cfg.name}.base")
        params.setdefault("payload", 0.0)
        if entry.get("urdf"):
            spec = from_urdf(urdf.load(_path(cfg, entry["urdf"], entry=entry)), entry.get("joints") or {}, entry.get("geometry") or {}, params)
        else:
            spec = spec_class(**{**{k: (tuple(v) if isinstance(v, list) else v) for k, v in (entry.get("geometry") or {}).items()}, **params})
        spec.payload += cfg.carried_mass()
        drawn = _file_visuals(cfg, entry)
        if drawn is not None:
            spec.link_visuals = drawn
        return spec
    return build


def _ranger_mini2(cfg: RobotConfig, entry: dict) -> Spec:
    params = _params(robots.RangerMini2, entry, f"{cfg.name}.base")
    model = entry.get("model")
    folder = _path(cfg, model, Path(robots.MODELS), entry) if model else None
    if "payload" not in params:
        params["payload"] = cfg.carried_mass()
    if folder is not None and (folder / "model.json").is_file():
        return robots.RangerMini2.from_model(folder, **params)
    return robots.RangerMini2(**params)   # no meshes at hand: placeholder shapes of the same size


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
    model = urdf.load(_path(cfg, entry["urdf"], entry=entry))
    return UrdfBody(urdf=model, positions=dict(entry.get("positions") or {}), model=f"urdf_{model.name}")


def _numbers(cls, entry: dict, where: str):
    known = [f.name for f in fields(cls)]
    _known(entry, known, where)
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in entry.items()})


def _camera(cfg: RobotConfig, entry: dict) -> Spec:
    """A camera: a registered ``model`` with its ``params``, optionally with another ``lens`` or image ``sensor`` (a
    registered one's name, or its numbers), its ``isp`` (hw.isp.Isp) and ``exposure`` (hw.sensor.ExposureControl)."""
    where = f"{cfg.name}.{entry.get('name')}"
    model = str(entry.get("model") or "")
    if model not in CAMERAS:
        near = difflib.get_close_matches(model, list(CAMERAS), 1)
        raise ValueError(f"{where}: no camera model {model!r}" + (f" (did you mean {near[0]!r}?)" if near else f" (have: {', '.join(sorted(CAMERAS))}; others: hw.config.register_camera)"))
    kw = _params(camera_models.CameraSpec, entry, where)
    for key, known, cls in (("lens", LENSES, camera_models.LensSpec), ("sensor", SENSORS, camera_models.SensorSpec)):
        v = entry.get(key)
        if isinstance(v, str):
            if v not in known:
                raise ValueError(f"{where}: no {key} {v!r} (have: {', '.join(sorted(known))}; or give its numbers)")
            kw[key] = known[v]() if callable(known[v]) else known[v]
        elif v is not None:
            kw[key] = _numbers(cls, v, f"{where}.{key}")
    chip = kw.pop("sensor", None)
    if entry.get("isp") is not None:
        kw["isp"] = _numbers(Isp, entry["isp"], f"{where}.isp")
    if entry.get("exposure") is not None:
        _known(entry["exposure"], [f.name for f in fields(ExposureControl)], f"{where}.exposure")
        kw["exposure"] = dict(entry["exposure"])
    spec = CAMERAS[model](**kw)
    if chip is not None:  # another image sensor behind the same housing
        spec.sensor = chip
        spec.__post_init__()
    return spec


def _plain(cls):
    return lambda cfg, e: cls(**_params(cls, e, f"{cfg.name}.{e['name']}"))


register_base("four_wheel_steering", _wheeled(drive.FourWheelSteering, fws_from_urdf), ("urdf", "joints", "geometry"))
register_base("differential", _wheeled(drive.DifferentialDrive, diff_from_urdf), ("urdf", "joints", "geometry"))
register_base("ranger_mini2", _ranger_mini2, ("model",))
register_base("urdf", _urdf_body, ("urdf", "positions"))
register_part("light", _plain(lights.AreaLight))
register_part("mast", _plain(rigs.Mast))
register_part("frame", _plain(rigs.ProfileFrame))
register_part("urdf", _urdf_body, ("urdf", "positions"))
register_sensor("camera", _camera, ("model", "lens", "sensor", "isp", "exposure", "calibrated"))
register_sensor("depth", _plain(sensing.DepthCamera))
register_sensor("lidar", _plain(sensing.Lidar))
register_sensor("ouster", _plain(ouster.OusterLidar))
register_part("arm", _plain(arm.MjcfArm))


def _oak_d_pro_rgb(**kw):
    """The colour camera of a Luxonis OAK-D Pro (Sony IMX378, 12 MP, HFOV 69 degrees: its spec sheet). The sensor's
    electrical numbers and the distortion are not published there: estimates, no distortion."""
    chip = camera_models.SensorSpec("Sony IMX378", 4056, 3040, 1.55, 10, shutter="rolling", full_well_e=4500.0, read_noise_e=2.0, sensitivity=0.2,
                                    note="full well, read noise and sensitivity are estimates")
    lens = camera_models.LensSpec("OAK-D Pro colour lens", 4.57, 1.8, 8.0, barrel_mm=(8.0, 4.0), note="focal length from HFOV 69 degrees")
    return camera_models.CameraSpec(**{"sensor": chip, "lens": lens, "model": "OAK-D Pro (colour)", "render_scale": 0.15,
                                       "body_mm": (12.0, 12.0, 6.0), **kw})


register_camera("oak_d_pro_rgb", _oak_d_pro_rgb, "OAK-D Pro (colour)")
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


def height(source) -> float:
    """How tall a configured robot is with what is on it (m above the floor): what must be free above its path."""
    robot = spawn(World(), source)
    top = float(getattr(robot.base.spec, "size", (0.0, 0.0, 0.0))[2])
    for actor in [robot.base, *robot.parts.values()]:
        frames = actor.link_matrices()
        for link, visuals in actor.spec.visuals().items():
            for v in visuals:
                if v.shape == "mesh":   # a base's size speaks for its meshes
                    continue
                half = np.array({"box": v.size, "cylinder": (2 * v.size[0], v.size[-1], 2 * v.size[0]), "profile": (v.size[0], v.size[0], v.size[-1]),
                                 "light": (*v.size, 0.0)}[v.shape], dtype=np.float64) / 2.0
                m = frames[link] @ v.origin.matrix()
                top = max(top, float(m[2, 3] + np.abs(m[2, :3]) @ half))
        top = max(top, *(float(m[2, 3]) for m in frames.values()))
    return top


def _places(cfg: RobotConfig, spec: Spec) -> dict:
    """The base's named mounts: its drive's own, and the file's."""
    out = dict(spec.mounts())
    for name, at in (cfg.data["base"].get("mounts") or {}).items():
        out[name] = (at.get("link") or out.get(name, (None,))[0], _transform(at.get("mount")))
    return out


def spawn(world: World, source, at: Transform | None = None, name: str | None = None, calibration=None, rig=None) -> Robot:
    """Spawn a configured robot: the base at ``at``, each part on its link. ``calibration``: a measured rig calibration
    (OpenCV YAML) for the cameras that name an entry of it (``calibrated: {camera, size}``), instead of the file the
    configuration names; without one that exists, those cameras stand at their ``mount`` with their datasheet lens."""
    cfg = load(source, rig)
    name = name or cfg.name
    entries = cfg.parts()
    # every spec before anything is spawned: a wrong entry leaves the world as it was
    base = BASES[cfg.data["base"]["type"]](cfg, cfg.data["base"])
    specs = {e["name"]: PARTS[e["type"]](cfg, e) for e in entries}
    places = _places(cfg, base)

    def seat(parent: str, link, mount: Transform) -> tuple[str, str | None, Transform]:
        """A part's parent, link and mount, with the base's named mounts worked in (a part on the base is on its deck)."""
        if parent != "base":
            return parent, link, mount
        link = link or ("deck" if "deck" in places else None)
        if link in places:
            on, place = places[link]
            return parent, on, Transform.from_matrix(place.matrix() @ mount.matrix())
        return parent, link, mount
    cal_entry = cfg.data.get("calibration") or {}
    cal_file = calibration or cal_entry.get("file")
    cal_path = _path(cfg, cal_file) if cal_file else None
    measured, seats = None, {}
    wanted = {e["name"]: e["calibrated"] for e in entries if e.get("calibrated")}
    for part, w in wanted.items():
        _known(w, ("camera", "size"), f"{cfg.name}.{part}.calibrated")
    if wanted and cal_path is not None and cal_path.is_file():
        from .calibration import RigCalibration
        measured = RigCalibration.load(cal_path)
        ref = cal_entry.get("reference") or {}
        _known(ref, ("parent", "link", "mount"), f"{cfg.name}.calibration.reference")
        by_entry = rigs.camera_mounts(measured, [w["camera"] for w in wanted.values()], _transform(ref.get("mount")))
        for part, w in wanted.items():
            size = w.get("size") or (specs[part].sensor.width, specs[part].sensor.height)
            specs[part].calibration = measured.cameras[w["camera"]].intrinsics(int(size[0]), int(size[1]))
            seats[part] = seat(str(ref.get("parent") or "base"), ref.get("link"), by_entry[w["camera"]])
    robot = Robot(cfg, world.spawn(base, at=at, name=name), calibrated=measured is not None)
    actors = {"base": robot.base}
    try:
        for e in entries:
            on, link, mount = seats.get(e["name"]) or seat(str(e.get("parent") or "base"), e.get("link"), _transform(e.get("mount")))
            actor = world.spawn(specs[e["name"]], attach_to=actors[on], link=link, mount=mount, name=f"{name}_{e['name']}")
            actors[e["name"]] = robot.parts[e["name"]] = actor
            if isinstance(actor.spec, sensing.Sensor):
                robot.sensors[e["name"]] = actor
            elif actor.spec.kind == "light":
                robot.lights[e["name"]] = actor
    except Exception:
        world.destroy(robot.base)   # what was spawned goes with its base
        raise
    return robot
