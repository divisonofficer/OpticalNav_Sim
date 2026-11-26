"""A configured robot as a tree, the way its hardware hangs together: the base, its wheels, the places on it (a deck,
a link) and the parts mounted there, then what is mounted on those parts, down to the cameras.

    tree = hw.tree.robot_tree(robot)        # a spawned hw.config.Robot (in the viewer's world, or an offline World())
    # {"id": "base", "label": "Ranger Mini 2.0", "kind": "base", "detail": "four_wheel_steering", "children": [
    #    {"id": "base/wheels", "kind": "group", "children": [{"id": "base/wheel/fl", "kind": "wheel", ...}, ...]},
    #    {"id": "base@deck", "kind": "mount", "children": [{"id": "mast", "kind": "mast", "children": [
    #        {"id": "mast@top", "kind": "link", "children": [{"id": "hdr", "kind": "camera", "sensor": "hdr", ...}]}]}]}]}

Every node: ``id`` (unique), ``label``, ``kind`` (base, group, wheel, mount, link, or a part's type: camera, depth, lidar,
light, mast, frame, urdf), ``detail`` (a line about it), ``children``; a part also ``model``, ``mass``, ``rig`` (it came with
the rig, not the robot) and, for a sensor, ``sensor`` (its name: what the viewer shows).
"""
from __future__ import annotations

from . import config as robot_config
from . import sensing


def _node(id_: str, label: str, kind: str, detail: str = "", **extra) -> dict:
    return {"id": id_, "label": label, "kind": kind, "detail": detail, "children": [], **extra}


def robot_tree(robot: "robot_config.Robot") -> dict:
    cfg = robot.config
    base = robot.base
    spec = base.spec
    base_entry = cfg.data.get("base") or {}
    root = _node("base", cfg.label, "base", str(base_entry.get("type") or type(spec).__name__),
                 model=type(spec).__name__, mass=float(getattr(spec, "kerb_mass", 0.0) or base_entry.get("mass") or 0.0))
    if hasattr(spec, "wheel_names"):   # a wheeled drive: its wheels, steered or not
        steered = hasattr(spec, "max_steer_deg")
        wheels = _node("base/wheels", "바퀴", "group", f"{len(spec.wheel_names())}개")
        for name, label in spec.wheel_names():
            wheels["children"].append(_node(f"base/wheel/{name}", label, "wheel", "조향 + 구동" if steered else "구동",
                                            radius=float(getattr(spec, "wheel_radius", 0.0))))
        root["children"].append(wheels)
    places = robot_config._places(cfg, spec)
    nodes = {"base": root}
    anchors: dict[str, dict] = {}

    def anchor(owner: str, link: str | None) -> dict:
        """The node under which something mounted on ``owner`` at ``link`` hangs: a named place (the deck), a link."""
        key = f"{owner}@{link}" if link else owner
        if not link:
            return nodes[owner]
        if key not in anchors:
            kind = "mount" if owner == "base" and link in places else "link"
            detail = f"→ {places[link][0]}" if kind == "mount" and places[link][0] else ""
            anchors[key] = _node(key, link, kind, detail)
            nodes[owner]["children"].append(anchors[key])
        return anchors[key]
    for e in cfg.parts():
        name, parent = str(e["name"]), str(e.get("parent") or "base")
        link = e.get("link") or ("deck" if parent == "base" and "deck" in places else None)
        actor = robot.parts.get(name)
        part_spec = actor.spec if actor is not None else None
        detail = robot_config._model_of(e) if e.get("type") in robot_config.SENSOR_TYPES else ""
        node = _node(name, str(e.get("label") or name), str(e["type"]), detail, model=robot_config._model_of(e),
                     mass=float(e.get("mass", 0.0)), rig=bool(e.get("_rig")))
        if part_spec is not None and isinstance(part_spec, sensing.Sensor):
            node["sensor"] = name
        nodes[name] = node
        anchor(parent, link)["children"].append(node)
    return root


def config_tree(source, rig=None) -> dict:
    """The tree of a configuration that stands nowhere yet (spawned into a world of its own)."""
    from .actor import World
    return robot_tree(robot_config.spawn(World(), source, rig=rig))
