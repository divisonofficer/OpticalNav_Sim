"""tools/merge_meshes.py on a small synthetic scene. The plan needs no Mitsuba; merging does (scalar_rgb), and those tests
are skipped when it is not importable (PYTHONPATH=<mitsuba build>/python)."""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import merge_meshes as mm  # noqa: E402
from opticalnav_sim import bodies, groundmap, server  # noqa: E402

BSDF_A = ('      <bsdf type="twosided">\n      <bsdf type="pplastic">\n        <rgb name="diffuse_reflectance" value="0.2 0.2 0.2" />\n'
          '        <float name="alpha" value="0.04" />\n      </bsdf>\n      </bsdf>\n')
BSDF_B = '    <bsdf type="diffuse">\n      <rgb name="reflectance" value="0.6 0.5 0.4" />\n    </bsdf>\n'
BSDF_C = '    <bsdf type="roughconductor">\n      <float name="alpha" value="0.1" />\n    </bsdf>\n'


def ply_with_attributes(path: Path, v, f, n, uv) -> None:
    """A binary PLY with normals and UVs, as tools/blend_to_pack.py writes them."""
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\n"
            "property float nx\nproperty float ny\nproperty float nz\nproperty float u\nproperty float v\n"
            f"element face {len(f)}\nproperty list uchar uint vertex_indices\nend_header\n").encode()
    face = np.empty(len(f), dtype=[("n", "u1"), ("i", "<u4", 3)])
    face["n"], face["i"] = 3, f
    path.write_bytes(head + np.concatenate([v, n, uv], 1).astype("<f4").tobytes() + face.tobytes())


def note(sid: str, material: str, **extra) -> str:
    n = {"id": sid, "type": "landmark", "label": extra.pop("label", sid), "material": material, "is_emitter": False,
         "geometry": {"center": [0, 0], "size_m": [1, 1, 1], "base_height_m": 0.0}, "nav": {"blocks": True}, **extra}
    return f"\n  <!--opticalnav-obj:{json.dumps(n, separators=(',', ':'))}-->"


def shape(sid: str, file: str, bsdf: str, material: str, transform: str = "", emitter: str = "", **extra) -> str:
    return (note(sid, material, **extra) + f'\n  <shape type="ply" id="{sid}">\n    <string name="filename" value="{file}" />\n'
            f"{transform}{bsdf}{emitter}  </shape>")


def make_scene(root: Path) -> Path:
    """base.xml, perturbed.xml (base + a mirror cube) and glassmirror.xml (base, the tv's material changed)."""
    scene = root / "scenes" / "toy"
    (scene / "assets").mkdir(parents=True)
    quad_v = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], np.float32)
    quad_f = np.array([[0, 2, 1], [0, 3, 2]])
    for i in range(12):
        bodies.write_ply(scene / "assets" / f"m{i}.ply", quad_v + np.float32([i, 0.01 * i, 0]), quad_f)
    wall_v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    up = np.tile([0.0, 0.0, 1.0], (4, 1))
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    ply_with_attributes(scene / "assets" / "w0.ply", wall_v + np.float32([0, 0, 5]), quad_f, up, uv)
    ply_with_attributes(scene / "assets" / "w1.ply", wall_v + np.float32([2, 0, 5]), quad_f, up, uv)
    moved = ('    <transform name="to_world">\n      <rotate y="1" angle="90" />\n      <scale value="2" />\n'
             '      <translate x="1" y="0.5" z="3" />\n    </transform>\n')
    light = '    <emitter type="area">\n      <rgb name="radiance" value="5 5 5" />\n    </emitter>\n'
    shapes = [
        shape("floor_a", "assets/m0.ply", BSDF_A, "Floor_epoxy"),
        shape("floor_b", "assets/m1.ply", BSDF_A, "Floor_epoxy"),
        shape("floor_moved", "assets/m2.ply", BSDF_A, "Floor_epoxy", transform=moved),
        shape("floor_other_name", "assets/m3.ply", BSDF_A, "Rug"),  # same BSDF, another material name: alone
        shape("wall_a", "assets/w0.ply", BSDF_B, "Paint"),
        shape("wall_b", "assets/w1.ply", BSDF_B, "Paint"),
        shape("wall_c", "assets/m4.ply", BSDF_B, "Paint"),  # no UVs: another layout, alone after loading
        shape("lamp_glow", "assets/m5.ply", BSDF_B, "Glow", emitter=light, is_emitter=True),
        shape("chair__part_0_", "assets/m6.ply", BSDF_C, "Steel", dynamic=True),
        shape("box_1", "assets/m7.ply", BSDF_C, "Steel", label="cardboard box"),
        shape("crate_1", "assets/m8.ply", BSDF_C, "Steel"),
        shape("shelf", "assets/m9.ply", BSDF_C, "Steel"),
        shape("mirror_auto_0", "assets/m10.ply", BSDF_C, "Steel"),
        shape("tv", "assets/m11.ply", BSDF_C, "Steel"),
        shape("steel_a", "assets/m10.ply", BSDF_C, "Steel"),
        shape("steel_b", "assets/m11.ply", BSDF_C, "Steel"),
    ]
    rect = ('\n  <shape type="rectangle" id="light_hidden_0_panel">\n    <emitter type="area">\n'
            '      <rgb name="radiance" value="1 1 1" />\n    </emitter>\n  </shape>')
    head = ('<scene version="3.0.0">\n  <integrator type="path">\n    <integer name="max_depth" value="4" />\n  </integrator>\n'
            '  <sensor type="perspective" id="sensor">\n    <float name="fov" value="60" />\n  </sensor>\n')
    base = head + "".join(shapes) + rect + "\n</scene>\n"
    (scene / "base.xml").write_text(base)
    cube = '\n  <shape type="cube" id="mirror_auto_9">\n    <bsdf type="conductor" />\n  </shape>'
    (scene / "perturbed.xml").write_text(base.replace("\n</scene>", cube + "\n</scene>"))
    tv = base.index('id="tv"')
    (scene / "glassmirror.xml").write_text(base[:tv] + base[tv:].replace(BSDF_C, BSDF_B, 1))
    (scene / "scene.json").write_text(json.dumps({"scan": "toy", "physics": {"dynamic": ["^crate_"]},
                                                  "variants": {"base": {"xml": "base.xml"}}}, indent=1))
    (scene / "room_map.json").write_text(json.dumps({"rooms": [{"objects": ["shelf"]}]}))
    return scene


def surfaces(scene: Path) -> list[tuple]:
    return [tuple(None if v is None else np.asarray(v).tolist() if k == "matrix" else v for k, v in sorted(s.items()))
            for s in groundmap.scene_surfaces(scene)]


def ids(text: str) -> list[str]:
    return [s["id"] for s in mm.top_shapes(text)]


def test_merge_plan_groups_and_exclusions(tmp_path):
    scene = make_scene(tmp_path)
    texts = {f.name: f.read_text() for f in mm.scene_xmls(scene)}
    plan = mm._plan(scene, texts, ["box"], json.loads((scene / "scene.json").read_text()))
    assert plan["groups"] == [["floor_a", "floor_b", "floor_moved"], ["wall_a", "wall_b", "wall_c"], ["steel_a", "steel_b"]]
    assert dict(plan["kept"]) == {"single": 1, "emitter": 1, "dynamic_note": 1, "keep_option": 1, "physics_list": 1,
                                  "referenced": 1, "protected_prefix": 1, "variant": 1, "not_a_mesh": 1}
    assert mm.needs_merging(scene)
    blocks = mm.top_shapes(texts["base.xml"])  # each block: whitespace, its own note, the shape; out, nothing else moves
    assert all(b["note"]["id"] == b["id"] for b in blocks if b["note"])
    cut = texts["base.xml"]
    for b in reversed(blocks):
        cut = cut[:b["start"]] + cut[b["end"]:]
    assert "<shape" not in cut and "opticalnav-obj" not in cut and cut.rstrip().endswith("</scene>")


def _mitsuba():
    pytest.importorskip("mitsuba")
    try:
        return mm._mitsuba()
    except Exception as exc:  # noqa: BLE001 - a build without scalar_rgb
        pytest.skip(f"no scalar_rgb Mitsuba: {exc}")


def test_merge_geometry_manifest_idempotence_and_unmerge(tmp_path):
    _mitsuba()
    scene = make_scene(tmp_path)
    original = {f.name: f.read_bytes() for f in mm.scene_xmls(scene)}
    before = {"groups": server.light_groups(original["base.xml"].decode()),
              "extra": server.extra_shapes(original["perturbed.xml"].decode(), original["base.xml"].decode()),
              "surfaces": surfaces(scene)}
    parts_alone = {}
    for s in mm.top_shapes(original["base.xml"].decode()):
        if s["type"] == "ply":
            parts_alone[s["id"]] = mm.load_arrays(scene, "ply", s["inner"])
    summary = mm.merge_scene(scene, keep=["box"])
    base = (scene / "base.xml").read_text()
    assert summary["shapes_before"] == 17 and summary["merged_groups"] == 3 and summary["parts"] == 7
    assert summary["shapes_after"] == 17 - 7 + 3 and summary["kept"]["single"] == 2  # floor_other_name, wall_c
    assert ids(base) == ["merged_Floor_epoxy", "floor_other_name", "merged_Paint", "wall_c", "lamp_glow", "chair__part_0_",
                         "box_1", "crate_1", "shelf", "mirror_auto_0", "tv", "merged_Steel", "light_hidden_0_panel"]
    for name in ("perturbed.xml", "glassmirror.xml"):  # every variant merged alike
        assert ids((scene / name).read_text())[:4] == ids(base)[:4]

    manifest = json.loads((scene / mm.MANIFEST).read_text())
    meta = json.loads((scene / "scene.json").read_text())
    assert meta["mesh_merge"]["manifest"] == mm.MANIFEST and meta["mesh_merge"]["parts"] == 7
    for g in manifest["merged"]:
        merged = mm.load_arrays(scene, "ply", f'<string name="filename" value="{g["file"]}" />')
        assert g["file"].startswith("assets/merged_") and g["xml"] in base
        assert g["vertices"] == len(merged["positions"]) and g["faces"] == len(merged["faces"])
        for p in g["parts"]:
            alone = parts_alone[p["id"]]
            v0, v1 = p["vertex_range"]
            f0, f1 = p["face_range"]
            assert np.array_equal(merged["positions"][v0:v1], alone["positions"])
            assert np.array_equal(merged["faces"][f0:f1] - v0, alone["faces"])
            assert np.allclose(merged["normals"][v0:v1], alone["normals"], rtol=0, atol=1e-6)
            assert (merged["uvs"] is None) == (alone["uvs"] is None)
            if alone["uvs"] is not None:
                assert np.array_equal(merged["uvs"][v0:v1], alone["uvs"])
            assert np.allclose(p["bbox"], [alone["positions"].min(0), alone["positions"].max(0)])
            assert p["note"]["id"] == p["id"] and f'value="{p["file"]}"' in p["shape"] and p["shape"] in original["base.xml"].decode()
    # the to_world is in the vertices: groundmap's reading of the transform agrees
    floor = next(g for g in manifest["merged"] if g["id"] == "merged_Floor_epoxy")
    moved = next(p for p in floor["parts"] if p["id"] == "floor_moved")
    m = groundmap._to_world(ET.fromstring(moved["shape"]))
    raw = groundmap.read_ply(scene / "assets" / "m2.ply")[0]
    merged_floor = groundmap.read_ply(scene / floor["file"])[0]
    assert np.allclose(merged_floor[slice(*moved["vertex_range"])], raw @ m[:3, :3].T + m[:3, 3], atol=1e-5)
    # the merged note, and what the server and groundmap read of the scene
    shapes = {s["id"]: s for s in mm.top_shapes(base)}
    n = shapes["merged_Paint"]["note"]
    assert n["material"] == "Paint" and n["label"] == "merged:Paint" and n["is_emitter"] is False and n["merged_parts"] == 2
    assert server.light_groups(base) == before["groups"]
    assert server.extra_shapes((scene / "perturbed.xml").read_text(), base) == before["extra"] == ("mirror_auto_9",)
    assert surfaces(scene) == before["surfaces"]  # read through the manifest, object by object
    import compact_assets
    assert {f"assets/m{i}.ply" for i in (0, 1, 2)} <= compact_assets.json_refs(scene)  # compaction keeps the parts' files

    # a second run changes nothing
    snapshot = {f.name: f.read_bytes() for f in scene.iterdir() if f.is_file()}
    assert not mm.needs_merging(scene)
    assert mm.merge_pack(tmp_path, ["toy"], if_needed=True) == {"toy": None}
    assert snapshot == {f.name: f.read_bytes() for f in scene.iterdir() if f.is_file()}

    # an object that may now move: unmerged and merged again without it; its merged mesh rewritten, the old one gone
    meta["physics"]["dynamic"].append("^floor_b$")
    (scene / "scene.json").write_text(json.dumps(meta, indent=1))
    assert mm.needs_merging(scene)
    mm.merge_scene(scene)
    base = (scene / "base.xml").read_text()
    assert "floor_b" in ids(base) and "merged_Floor_epoxy" in ids(base) and "box_1" in ids(base)  # --keep remembered
    assert not (scene / floor["file"]).exists()

    # back to the original files, byte for byte; merged meshes unlinked, scene.json without the record
    assert mm.unmerge_scene(scene)["xmls"] == {"base.xml": True, "glassmirror.xml": True, "perturbed.xml": True}
    assert {f.name: f.read_bytes() for f in mm.scene_xmls(scene)} == original
    assert not (scene / mm.MANIFEST).exists() and not list((scene / "assets").glob("merged_*"))
    assert "mesh_merge" not in json.loads((scene / "scene.json").read_text())


def test_unmerge_after_an_edit_puts_the_parts_in_place(tmp_path):
    _mitsuba()
    scene = make_scene(tmp_path)
    before = sorted(ids((scene / "base.xml").read_text()))
    mm.merge_scene(scene, keep=[])
    text = (scene / "base.xml").read_text()
    (scene / "base.xml").write_text(text.replace("</scene>", "  <!-- edited -->\n</scene>"))  # a later tool's change
    assert mm.unmerge_scene(scene)["xmls"]["base.xml"] is False
    after = (scene / "base.xml").read_text()
    assert sorted(ids(after)) == before and "<!-- edited -->" in after and "merged_" not in after


def test_a_scene_written_again_over_a_merged_one_is_merged_anew(tmp_path):
    _mitsuba()
    scene = make_scene(tmp_path)
    original = {f.name: f.read_text() for f in mm.scene_xmls(scene)}
    mm.merge_scene(scene, keep=[])
    merged = {f.name: f.read_text() for f in mm.scene_xmls(scene)}
    for name, text in original.items():  # the converter writes its files again; the old manifest stays behind
        (scene / name).write_text(text)
    assert mm.needs_merging(scene)
    mm.merge_scene(scene)
    assert {f.name: f.read_text() for f in mm.scene_xmls(scene)} == merged
    assert not mm.needs_merging(scene)
