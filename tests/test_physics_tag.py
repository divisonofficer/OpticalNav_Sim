"""opticalnav_sim/physics_objects.py and tools/physics_tag.py on small synthetic scenes: grouping shapes into objects,
volume and mass, the classification of typical cases, physics.json, and tools/merge_meshes.py keeping movable objects
apart (that part needs Mitsuba scalar_rgb and is skipped without it)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import merge_meshes as mm  # noqa: E402
import physics_tag  # noqa: E402
from opticalnav_sim import bodies  # noqa: E402
from opticalnav_sim import physics_objects as po  # noqa: E402

BSDF = '    <bsdf type="diffuse">\n      <rgb name="reflectance" value="0.5 0.5 0.5" />\n    </bsdf>\n'


def box_mesh(lo, hi):
    """A closed box, faces outward (Mitsuba axes: Y up)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    v = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4],
                  [1, 5, 7], [1, 7, 3]])
    if po.mesh_volume(v[f]) < 0:
        f = f[:, ::-1]
    return v, f


class Scene:
    """A scene dir written shape by shape: each a box in its own PLY with an opticalnav-obj note."""

    def __init__(self, root: Path, name: str = "toy"):
        self.dir = root / "scenes" / name
        (self.dir / "assets").mkdir(parents=True)
        self.shapes = []

    def box(self, sid: str, lo, hi, label: str | None = None, material: str = "Oak wood", **note) -> str:
        v, f = box_mesh(lo, hi)
        bodies.write_ply(self.dir / "assets" / f"{sid}.ply", v.astype(np.float32), f)
        n = {"id": sid, "type": "landmark", "label": label or sid, "material": material, "is_emitter": False,
             "geometry": {"center": [0, 0], "size_m": [1, 1, 1], "base_height_m": 0.0}, "nav": {"blocks": True}, **note}
        head = f"\n  <!--opticalnav-obj:{json.dumps(n, separators=(',', ':'))}-->"
        self.shapes.append(f'{head}\n  <shape type="ply" id="{sid}">\n'
                           f'    <string name="filename" value="assets/{sid}.ply" />\n{BSDF}  </shape>')
        return sid

    def write(self) -> Path:
        (self.dir / "base.xml").write_text('<scene version="3.0.0">\n  <integrator type="path" />' + "".join(self.shapes)
                                           + "\n</scene>\n")
        (self.dir / "scene.json").write_text(json.dumps({"scan": self.dir.name, "variants": {"base": {"xml": "base.xml"}}}))
        return self.dir


def room(root: Path) -> Path:
    """A floor, a wall, a stool (seat on a leg), a table (top on four legs) with a cup on it, a tall shelf, a box hung on
    the wall, two crates of one label stem far apart, and a chair whose parts share a note ``object``."""
    s = Scene(root)
    s.box("floor", [-5, -0.1, -5], [5, 0, 5], label="room floor", material="Gray epoxy floor")
    s.box("north_wall", [-5, 0, -5.2], [5, 3, -5], label="north wall", material="Wall paint")
    s.box("stool_seat", [1, 0.45, 1], [1.35, 0.5, 1.35], label="stool seat", material="Upholstery fabric")
    s.box("stool_leg", [1.15, 0, 1.15], [1.2, 0.45, 1.2], label="stool leg", material="Chrome steel")
    s.box("table_top", [-2, 0.72, 0], [-0.8, 0.76, 0.8], label="table top")
    for k, (x, z) in enumerate([(-2, 0), (-0.85, 0), (-2, 0.75), (-0.85, 0.75)]):
        s.box(f"table_leg_{k}", [x, 0, z], [x + 0.05, 0.72, z + 0.05], label=f"table leg.00{k}")
    s.box("cup", [-1.5, 0.76, 0.3], [-1.42, 0.86, 0.38], label="coffee cup")   # oak, as the table: it would merge
    s.box("shelf", [2, 0, -5], [3, 2, -4.6], label="tall shelf")
    s.box("wall_box", [-1, 1.5, -5], [-0.7, 1.8, -4.8], label="fuse box", material="Painted steel")
    s.box("crate_a", [3, 0, 2], [3.4, 0.3, 2.4], label="crate.001", material="Pine wood")
    s.box("crate_a_lid", [3, 0.3, 2], [3.4, 0.32, 2.4], label="crate.001", material="Pine wood")
    s.box("crate_b", [-3, 0, 3], [-2.6, 0.3, 3.4], label="crate.002", material="Pine wood")
    s.box("chair_seat", [0, 0.4, 3], [0.45, 0.45, 3.45], label="seat cushion", material="Fabric", object="Chair_01")
    s.box("chair_base", [0.2, 0, 3.2], [0.25, 0.4, 3.25], label="pedestal", material="Steel", object="Chair_01")
    return s.write()


def by_part(result: dict) -> dict:
    return {p: o for o in result["objects"] for p in o["parts"]}


def test_words_classes_and_stems():
    assert po.label_class("CS02_service cart shelf") == ("movable", "cart")
    assert po.label_class("CS02_r008_round stool upholstered seat") == ("movable", "stool")
    assert po.label_class("CS02_bench tool driver handle") == ("movable", "tool")
    assert po.label_class("CS02_wired bench unit 0 case") == ("movable", "case")
    assert po.label_class("CS02_west bench top") == ("furniture", "bench")
    assert po.label_class("CS02_bench measurement cable") == (None, None)
    assert po.label_class("CardboardBoxFactory(106).spawn_asset(106)") == ("movable", "box")
    assert po.label_class("CS02_north wall") == ("structure", "wall")
    assert po.label_class("CS02_underbed cable tray") == ("structure", "cable tray")
    assert po.label_stem("CS02_north rack punched slot.003") == "CS02_north rack punched slot"
    assert po.label_stem("CS08_mullion_10") == "CS08_mullion"
    assert po.asset_key("CS02_r008_round stool upholstered seat") == "CS02_r008"
    assert po.asset_key("CS02_cart caster") is None
    assert po.material_of("Brushed aluminium") == "aluminium" and po.material_of("", "mystery") == po.DEFAULT_MATERIAL


def test_volume_closedness_and_mass_of_a_box(tmp_path):
    v, f = box_mesh([0, 0, 0], [0.5, 0.5, 0.5])
    tri = v[f]
    assert po.mesh_volume(tri) == pytest.approx(0.125)
    assert po.closed_fraction(tri) == 1.0 and po.closed_fraction(tri[:-2]) < 1.0
    s = Scene(tmp_path)
    s.box("floor", [-2, -0.1, -2], [2, 0, 2], label="floor")
    s.box("solid", [0, 0, 0], [0.5, 0.5, 0.5], label="block", material="Oak wood")
    s.box("open", [1, 0, 0], [1.5, 0.5, 0.5], label="crate", material="Oak wood")
    scene = s.write()
    # the open crate: the same box without its top, written over the first file
    v2, f2 = box_mesh([1, 0, 0], [1.5, 0.5, 0.5])
    keep = [k for k, t in enumerate(f2) if not np.allclose(v2[t][:, 1], 0.5)]
    bodies.write_ply(scene / "assets" / "open.ply", v2.astype(np.float32), f2[keep])
    parts = po.read_parts(scene)
    idx = {p["id"]: i for i, p in enumerate(parts)}
    po._part_flags(parts, po.RULES)
    solid = po._mass(parts, [idx["solid"]], po.Body(parts, [idx["solid"]], po.RULES))
    assert solid["volume_method"] == "closed mesh" and solid["volume_m3"] == pytest.approx(0.125, rel=1e-4)
    # wood is 600 kg/m^3; a closed 0.5 m box of it weighs the lighter of solid (75 kg) and an 18 mm shell (16.2 kg)
    assert solid["mass_kg"] == pytest.approx(1.5 * 0.018 * 600, rel=1e-3)
    assert np.allclose(solid["com"], [0.25, 0.25, 0.25], atol=1e-6)
    hollow = po._mass(parts, [idx["open"]], po.Body(parts, [idx["open"]], po.RULES))
    assert hollow["volume_method"] == "half the convex hull"
    assert hollow["mass_kg"] == pytest.approx(1.25 * 0.018 * 600, rel=1e-3)


def test_grouping_by_note_object_label_stem_and_support(tmp_path):
    scene = room(tmp_path)
    parts = po.read_parts(scene)
    groups = [sorted(parts[i]["id"] for i in g) for g in po.group(parts)]
    assert ["stool_leg", "stool_seat"] in groups                       # a seat on its leg: one body
    assert ["table_leg_0", "table_leg_1", "table_leg_2", "table_leg_3", "table_top"] in groups
    assert ["cup"] in groups                                           # on the table, not part of it
    assert ["crate_a", "crate_a_lid"] in groups and ["crate_b"] in groups   # one stem, two places: two crates
    assert ["chair_base", "chair_seat"] in groups                      # the author's parent
    assert ["floor"] in groups and ["north_wall"] in groups


def test_classification_of_typical_cases(tmp_path):
    scene = room(tmp_path)
    result = po.analyse(scene)
    obj = by_part(result)
    assert not obj["north_wall"]["movable"] and obj["north_wall"]["reason"].startswith("structure")
    assert not obj["floor"]["movable"]
    stool = obj["stool_seat"]
    assert stool["movable"] and stool["robot_reachable"] and "'stool'" in stool["reason"]
    assert stool["support"]["kind"] == "floor"
    cup = obj["cup"]
    assert cup["movable"] and not cup["robot_reachable"] and cup["support"]["kind"] == "object"
    assert cup["support"]["on"] == ["table top"]
    assert not obj["shelf"]["movable"]                                 # heavy (69 kg of oak) before the shelf word
    wall_box = obj["wall_box"]
    assert not wall_box["movable"] and "nothing under it" in wall_box["reason"] and wall_box["touches"]["wall"] == ["north wall"]
    assert not obj["table_top"]["movable"]                             # 1.2 x 0.8 m: too wide to be a free object
    assert obj["crate_a"]["movable"] and obj["crate_a"]["robot_reachable"]
    assert obj["chair_seat"]["movable"]                                # free-standing, low, light (no keyword)


def test_physics_json_and_needs_tagging(tmp_path):
    scene = room(tmp_path)
    assert physics_tag.needs_tagging(scene)
    counts = physics_tag.tag_scene(scene)
    data = json.loads((scene / po.PHYSICS).read_text())
    assert data["version"] == po.VERSION and data["rules"]["max_mass_kg"] == po.RULES["max_mass_kg"]
    assert data["counts"] == {k: data["counts"][k] for k in counts} and counts["objects"] == len(data["objects"])
    movable = [o for o in data["objects"] if o["movable"]]
    assert data["movable"] == [o["id"] for o in movable]
    assert sorted(data["movable_parts"]) == sorted(p for o in movable for p in o["parts"])
    for o in movable:
        hull = o["hull"]
        assert 4 <= len(hull["vertices"]) <= po.RULES["hull_vertices"] and o["mass_kg"] > 0
        v = np.array(hull["vertices"]) + o["pose"]["position"]
        lo, hi = np.array(o["bbox"][0]), np.array(o["bbox"][1])
        assert np.all(v >= lo - 1e-3) and np.all(v <= hi + 1e-3)    # the hull in the sim frame, around the pose
        f = np.array(hull["faces"])
        n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        assert np.all((n * (v[f].mean(1) - v.mean(0))).sum(1) > 0)    # faces outward
        assert 0.0 < o["friction"]["mu"] <= 1.0 and o["pose"]["quat_wxyz"] == [1.0, 0.0, 0.0, 0.0]
    stool = by_part(data)["stool_seat"]
    assert stool["bbox"][0][2] == pytest.approx(0.0) and stool["bbox"][1][2] == pytest.approx(0.5)   # z up in the sim frame
    assert not physics_tag.needs_tagging(scene)
    (scene / "base.xml").write_text((scene / "base.xml").read_text().replace('id="cup"', 'id="mug"'))
    assert physics_tag.needs_tagging(scene)


def _mitsuba():
    pytest.importorskip("mitsuba")
    try:
        return mm._mitsuba()
    except Exception as exc:  # noqa: BLE001 - a build without scalar_rgb
        pytest.skip(f"no scalar_rgb Mitsuba: {exc}")


def shape_ids(scene: Path) -> set:
    return {s["id"] for s in mm.top_shapes((scene / "base.xml").read_text())}


def merged_of(scene: Path) -> dict:
    """Merged shape id -> (its physics object, its part ids)."""
    return {g["id"]: (g.get("object"), [p["id"] for p in g["parts"]]) for g in mm.read_manifest(scene)["merged"]}


def set_keep(scene: Path, scope: str) -> None:
    meta = json.loads((scene / "scene.json").read_text())
    meta["physics"] = {"keep": scope}
    (scene / "scene.json").write_text(json.dumps(meta))


def test_merge_makes_one_body_per_kept_object_and_follows_the_scope(tmp_path):
    _mitsuba()
    scene = room(tmp_path)
    original = (scene / "base.xml").read_bytes()
    physics_tag.tag_scene(scene)
    data = json.loads((scene / po.PHYSICS).read_text())
    obj = by_part(data)
    reach = {o["id"] for o in data["objects"] if o["movable"] and o["robot_reachable"]}
    assert {obj["crate_a"]["id"], obj["stool_seat"]["id"]} <= reach and obj["cup"]["id"] not in reach
    # keep "reachable" (the default): the crate's two pine shapes become one body mesh; the cup (on the table, out of
    # the robot's reach) merges with the oak table like any static shape
    summary = mm.merge_scene(scene)
    merged = merged_of(scene)
    crate = obj["crate_a"]["id"]
    assert merged[f"dyn_{crate}_1"] == (crate, ["crate_a", "crate_a_lid"])
    static = [m for m, (o, parts) in merged.items() if o is None and "cup" in parts]
    assert static and static[0].startswith("merged_")
    assert summary["kept"]["dynamic_single"] >= 3                       # crate_b, the stool's fabric and chrome parts
    record = json.loads((scene / po.PHYSICS).read_text())["merge"]
    assert record["keep"] == "reachable"
    assert record["objects"][crate] == {"kept": True, "shapes": [f"dyn_{crate}_1"]}
    assert record["objects"][obj["stool_seat"]["id"]] == {"kept": True, "shapes": ["stool_seat", "stool_leg"]}
    assert record["objects"][obj["cup"]["id"]] == {"kept": False, "shapes": static}
    assert set(record["objects"][crate]["shapes"]) <= shape_ids(scene)
    assert not mm.needs_merging(scene) and not physics_tag.needs_tagging(scene)
    # read from the merged files (dyn_ shapes too), the scene gives the same objects
    again = po.analyse(scene)
    assert again["movable_parts"] == data["movable_parts"]
    assert [o["mass_kg"] for o in again["objects"]] == pytest.approx([o["mass_kg"] for o in data["objects"]])
    # tagged again, physics.json keeps what the merge recorded
    physics_tag.tag_scene(scene)
    assert json.loads((scene / po.PHYSICS).read_text())["merge"] == record
    # keep "movable": the cup is a body of its own too (one shape: left alone), merged anew
    set_keep(scene, "movable")
    assert mm.needs_merging(scene)
    mm.merge_scene(scene)
    assert "cup" in shape_ids(scene) and not any("cup" in parts for _, parts in merged_of(scene).values())
    assert json.loads((scene / po.PHYSICS).read_text())["merge"]["objects"][obj["cup"]["id"]]["kept"]
    # physics.json changes (the cup is fixed now): merged with the table again
    data = json.loads((scene / po.PHYSICS).read_text())
    for o in data["objects"]:
        if "cup" in o["parts"]:
            o["movable"] = False
    (scene / po.PHYSICS).write_text(json.dumps(data))
    assert mm.needs_merging(scene)
    mm.merge_scene(scene)
    assert "cup" not in shape_ids(scene)
    # keep "none": every object static, the crate's shapes in the pine mesh of the scene with the other crate
    set_keep(scene, "none")
    assert mm.needs_merging(scene)
    mm.merge_scene(scene)
    assert not any(m.startswith("dyn_") for m in merged_of(scene))
    pine = [parts for m, (o, parts) in merged_of(scene).items() if "crate_a" in parts]
    assert pine and set(pine[0]) == {"crate_a", "crate_a_lid", "crate_b"}
    assert not mm.needs_merging(scene)
    # and back, byte for byte
    mm.unmerge_scene(scene)
    assert (scene / "base.xml").read_bytes() == original
    assert "merge" not in json.loads((scene / po.PHYSICS).read_text())
