"""The Solid preview's filters: one property of the scene's surfaces shown in colour, the rest in grey.

    view["renderer"], view["solid_filter"] = "solid", "movable"      # the render server (raster.render_solid)

Each filter colours what it is about and greys the rest, over the Solid preview's studio light, so the scene's shapes
stay readable. The colours and their meanings are here, plain data, for the renderer (raster.py) and the viewer's
legend (gui.py) alike.

* ``movable``   physics.json's decision for the object a surface belongs to (tools/physics_tag.py)
* ``bsdf``      the kind of BSDF the path tracer gives it: diffuse, coated (plastic), metal, metal blended with a
                coat, mirror, glass (drawn over what is behind it), emitter
* ``roughness`` the GGX alpha of its specular layer (textured or not; a surface with no specular layer is grey)
* ``metallic``  the conductor's share of its BSDF (Infinigen's metallic weight, a blendbsdf of plastic and conductor)
* ``facing``    which side the camera sees: the back of a one-sided BSDF renders black in the path tracer
* ``lights``    emitters and the window panes the sky comes in by
* ``objects``   one colour per object (physics.json's grouping of the scene's shapes)
* ``materials`` one colour per BSDF of the loaded scene

The object filters need each triangle's object. A pack scene is merged (tools/merge_meshes.py) and Mitsuba's loader
merges what it can further, so the loaded meshes no longer say which object a triangle came from; ``scene_index``
reads the scene object by object (physics_objects.read_parts: merged parts from their slice of the merged mesh) and
keeps each triangle's centroid, and the renderer finds its triangles' objects by nearest centroid.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

NAMES = ("movable", "bsdf", "roughness", "metallic", "facing", "lights", "objects", "materials")
OBJECT_FILTERS = ("movable", "objects")   # need the scene's objects (scene_index)
MATCH_M = 0.002   # a loaded triangle's centroid this close to a part's triangle's is that part's
CACHE = Path(os.environ.get("OPTICALNAV_SIM_CACHE", Path.home() / ".cache" / "opticalnav_sim")) / "solid_index"
INDEX_VERSION = 1

# legend colours (sRGB, as the viewer shows them over a surface lit head-on); the renderer draws their linear values
DIM = "#8a8a8a"
COLORS = {
    "movable": [("reachable", "#ff8a1f", "로봇이 미는 물체 (닿는 높이)"), ("rider", "#f2d03b", "그 위에 얹혀 같이 움직임"),
                ("movable", "#4f8cff", "움직일 수 있음 (닿지 않음)"), ("fixed", DIM, "고정"), ("unknown", "#5a5a5a", "물체 정보 없음")],
    "bsdf": [("diffuse", "#d9cdb4", "diffuse"), ("coated", "#5cc26a", "plastic (코팅)"), ("metal", "#4f8cff", "금속 (conductor)"),
             ("blend", "#b46bff", "금속+코팅 섞임"), ("mirror", "#2fe6f0", "거울"), ("glass", "#7fd8ff", "유리 (앞에 겹침)"),
             ("emitter", "#ffe14d", "발광체")],
    "facing": [("front", "#8fa6c8", "앞면"), ("back_two", "#f0b45a", "뒷면, 양면 재질 (문제없음)"),
               ("back_one", "#ff2a2a", "뒷면, 단면 재질 (경로추적에서 검게 나옴)")],
    "lights": [("emitter", "#ffe14d", "발광체"), ("window", "#7fd8ff", "창 (하늘빛이 드는 유리)"), ("other", DIM, "그 밖")],
}
RAMPS = {"roughness": ("GGX alpha", 0.0, 1.0, "매끈", "거침"), "metallic": ("금속 비율", 0.0, 1.0, "비금속", "금속")}
RAMP_STOPS = ["#30123b", "#4686fb", "#1be5b5", "#a4fc3c", "#fbb938", "#e4460a"]   # turbo, low to high
LABELS = {"movable": "움직이는 물체", "bsdf": "BSDF 종류", "roughness": "거칠기", "metallic": "금속도", "facing": "앞면 / 뒷면",
          "lights": "광원", "objects": "오브젝트 ID", "materials": "재질 ID"}


def srgb_linear(hex_colour: str) -> tuple[float, float, float]:
    c = [int(hex_colour[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c)


def colour(name: str, key: str) -> tuple[float, float, float]:
    return srgb_linear(next(c for k, c, _ in COLORS[name] if k == key))


def legend() -> dict:
    """For the viewer: each filter's label and its colours (categories) or ramp."""
    out = {}
    for name in NAMES:
        entry = {"label": LABELS[name]}
        if name in COLORS:
            entry["items"] = [{"key": k, "colour": c, "label": text} for k, c, text in COLORS[name]]
        elif name in RAMPS:
            title, lo, hi, lo_text, hi_text = RAMPS[name]
            entry["ramp"] = {"title": title, "low": lo, "high": hi, "low_label": lo_text, "high_label": hi_text,
                             "stops": RAMP_STOPS}
        else:
            entry["note"] = "색은 임의 (같은 색 = 같은 " + ("물체" if name == "objects" else "재질") + ")"
        out[name] = entry
    return out


def check(name) -> str | None:
    if name in (None, "", "none"):
        return None
    if name not in NAMES:
        raise ValueError(f"solid_filter must be one of {NAMES} (or none)")
    return str(name)


# ---------------------------------------------------------------- the scene's objects
def _stamp(scene_dir: Path, xml: str) -> str:
    parts = []
    for name in (xml, "physics.json", "merged_meshes.json"):
        p = scene_dir / name
        with_stat = p.stat() if p.is_file() else None
        parts.append(f"{name}:{with_stat.st_size}:{with_stat.st_mtime_ns}" if with_stat else f"{name}:-")
    return hashlib.sha1(f"{INDEX_VERSION}|{scene_dir.resolve()}|{'|'.join(parts)}".encode()).hexdigest()[:16]


def _object_rows(scene_dir: Path) -> tuple[dict, list]:
    """physics.json: part id -> object number, and per object {id, label, class, movable, reachable, kept, mass_kg,
    reason} (empty without the file)."""
    path = scene_dir / "physics.json"
    if not path.is_file():
        return {}, []
    data = json.loads(path.read_text())
    kept = {oid for oid, r in ((data.get("merge") or {}).get("objects") or {}).items() if r.get("kept")}
    of, rows = {}, []
    for o in data.get("objects", []):
        for p in o.get("parts", []):
            of[p] = len(rows)
        rows.append({"id": o["id"], "label": o.get("label"), "class": o.get("class"), "movable": bool(o.get("movable")),
                     "reachable": bool(o.get("robot_reachable")), "kept": o["id"] in kept, "mass_kg": o.get("mass_kg"),
                     "reason": o.get("reason")})
    return of, rows


def scene_index(scene_dir, xml: str = "base.xml") -> dict:
    """Every object triangle of the scene: ``centroids`` (T x 3, Mitsuba's world), ``part`` (T, the part's number),
    ``parts`` ([{id, label, material, object}], object: physics.json's object number or -1) and ``objects`` (see
    _object_rows). Cached on disk by the scene files' sizes and times."""
    from . import physics_objects

    scene_dir = Path(scene_dir)
    key = _stamp(scene_dir, xml)
    path = CACHE / f"{scene_dir.name}-{key}.npz"
    meta_path = path.with_suffix(".json")
    if path.is_file() and meta_path.is_file():
        with np.load(path) as z:
            meta = json.loads(meta_path.read_text())
            return {"centroids": z["centroids"], "part": z["part"], **meta}
    t0 = time.monotonic()
    of, rows = _object_rows(scene_dir)
    cents, owner, parts = [], [], []
    for p in physics_objects.read_parts(scene_dir, xml):
        v, f = p["verts"], p["faces"]
        if not len(f):
            continue
        cents.append(((v[f[:, 0]] + v[f[:, 1]] + v[f[:, 2]]) / 3.0).astype(np.float32))
        owner.append(np.full(len(f), len(parts), np.int32))
        parts.append({"id": p["id"], "label": p["label"], "material": p["material"], "object": of.get(p["id"], -1)})
    centroids = np.concatenate(cents) if cents else np.zeros((0, 3), np.float32)
    part = np.concatenate(owner) if owner else np.zeros(0, np.int32)
    meta = {"parts": parts, "objects": rows, "seconds": round(time.monotonic() - t0, 2)}
    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
        np.savez(tmp, centroids=centroids, part=part)
        os.replace(tmp, path)
        meta_path.write_text(json.dumps(meta))
    except OSError:
        pass
    return {"centroids": centroids, "part": part, **meta}


def match(index: dict, centroids: np.ndarray, tol: float = MATCH_M) -> np.ndarray:
    """Each loaded triangle's part number (by its centroid; -1 when no part's triangle is within ``tol``)."""
    from scipy.spatial import cKDTree

    out = np.full(len(centroids), -1, np.int32)
    if not len(index["centroids"]) or not len(centroids):
        return out
    tree = cKDTree(index["centroids"])
    d, i = tree.query(centroids, distance_upper_bound=tol, workers=-1)
    ok = np.isfinite(d)
    out[ok] = index["part"][i[ok]]
    return out
