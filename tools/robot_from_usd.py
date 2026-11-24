#!/usr/bin/env python3
"""A robot's drawn model from its USD: one mesh file per link and material, in the link's own frame.

    python3 tools/robot_from_usd.py --usd <robomituba>/assets/robots/ranger_mini_v3/ranger_mini_rokace.usdc \
        --out packs/hardware/ranger_mini_v3

Reads the USD in Blender (the copy the Cycles backend uses), takes the wheeled base's links (``base_link``, four
``*_steer_link``, four ``*_wheel_link``), and writes ``<out>/model.json`` and PLY files (positions and normals):

* ``chassis``: the body, in the frame on the floor under the centre of the four wheels (x forward, z up);
* ``steer_fl`` ...: a steering module, in the frame at its wheel's centre (it turns about z);
* ``wheel_fl`` ...: a wheel, in the frame at its centre (it rolls about y).

``opticalnav_sim.hw.robots.RangerMini2(model_dir=<out>)`` draws these instead of its placeholder boxes, with the
wheelbase, track, wheel radius and height measured here. Meshes over ``--max-triangles`` are decimated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

WHEELS = ("fl", "fr", "rl", "rr")


def inside_blender(cfg: dict) -> None:
    import bpy
    import numpy as np

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=cfg["usd"])
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    objects = {o.name: o for o in bpy.data.objects}
    for name in ["base_link"] + [f"{w}_{k}_link" for w in WHEELS for k in ("steer", "wheel")]:
        if name not in objects:
            raise SystemExit(f"{cfg['usd']}: no {name} (links found: {sorted(n for n in objects if n.endswith('_link'))})")

    def meshes_of(link):  # the meshes directly under a link (a wheel's are not its steering module's)
        return [c for c in link.children if c.type == "MESH" and not c.hide_render]

    def world_points(obj):
        m = np.array(obj.matrix_world)
        v = np.array([p.co[:] for p in obj.data.vertices], dtype=np.float64)
        return (np.c_[v, np.ones(len(v))] @ m.T)[:, :3]

    # a wheel is named by where it is (front: +x, left: +y), not by the file's name for it (this asset's rear pair is
    # named the other way round)
    at = {w: np.array(objects[f"{w}_wheel_link"].matrix_world.translation[:]) for w in WHEELS}
    mid = np.mean(list(at.values()), axis=0)
    usd = {("f" if p[0] > mid[0] else "r") + ("l" if p[1] > mid[1] else "r"): w for w, p in at.items()}
    if sorted(usd) != sorted(WHEELS):
        raise SystemExit(f"the four wheels are not at four corners: {at}")
    wheel_pts = {w: np.concatenate([world_points(o) for o in meshes_of(objects[f"{usd[w]}_wheel_link"])]) for w in WHEELS}
    ground = min(float(p[:, 2].min()) for p in wheel_pts.values())
    centres = {w: at[usd[w]] for w in WHEELS}
    radius = float(np.mean([centres[w][2] - ground for w in WHEELS]))
    width = float(np.mean([np.ptp(p[:, 1]) for p in wheel_pts.values()]))
    wheelbase = float(np.mean([centres["fl"][0] - centres["rl"][0], centres["fr"][0] - centres["rr"][0]]))
    track = float(np.mean([centres["fl"][1] - centres["fr"][1], centres["rl"][1] - centres["rr"][1]]))
    floor = np.eye(4)
    floor[:3, 3] = (mid[0], mid[1], ground)  # the model's base frame in the USD's world

    def frame_at(point):
        m = np.eye(4)
        m[:3, 3] = point
        return m

    frames = {"chassis": floor}
    for w in WHEELS:
        frames[f"steer_{w}"] = frame_at(centres[w])
        frames[f"wheel_{w}"] = frame_at(centres[w])
    sources = {"chassis": meshes_of(objects["base_link"])}
    for w in WHEELS:
        sources[f"steer_{w}"] = meshes_of(objects[f"{usd[w]}_steer_link"])
        sources[f"wheel_{w}"] = meshes_of(objects[f"{usd[w]}_wheel_link"])

    def material_of(mat):
        kind, rgb, rough = "plastic", (0.5, 0.5, 0.5), 0.5
        node = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None) if mat and mat.use_nodes else None
        if node is not None:
            rgb = tuple(round(float(c), 4) for c in node.inputs["Base Color"].default_value[:3])
            rough = round(float(node.inputs["Roughness"].default_value), 3)
            kind = "metal" if float(node.inputs["Metallic"].default_value) >= 0.5 else "plastic"
        if kind == "plastic" and max(rgb) < 0.03:  # tyres and black trim
            kind, rgb, rough = "rubber", (0.03, 0.03, 0.03), max(rough, 0.6)
        return kind, rgb, rough

    def write_ply(path, pos, nrm, tris):
        head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pos)}\nproperty float x\nproperty float y\n"
                f"property float z\nproperty float nx\nproperty float ny\nproperty float nz\nelement face {len(tris)}\n"
                f"property list uchar int vertex_indices\nend_header\n")
        rows = np.zeros(len(tris), dtype=[("n", "u1"), ("i", "<i4", (3,))])
        rows["n"], rows["i"] = 3, tris
        path.write_bytes(head.encode() + np.concatenate([pos, nrm], 1).astype("<f4").tobytes() + rows.tobytes())

    dg = bpy.context.evaluated_depsgraph_get()
    links, total, bounds, clearance = {}, 0, [np.full(3, 1e9), np.full(3, -1e9)], 1e9
    for link, objs in sources.items():
        to_link = np.linalg.inv(frames[link])
        parts: dict[tuple, list] = {}  # material -> [(positions, normals, triangles)]
        for obj in objs:
            count = sum(len(p.vertices) - 2 for p in obj.data.polygons)
            if count > cfg["max_triangles"]:
                mod = obj.modifiers.new("decimate", "DECIMATE")
                mod.ratio = cfg["max_triangles"] / count
                dg = bpy.context.evaluated_depsgraph_get()
                print(f"[robot] {link}: {count} triangles decimated to about {cfg['max_triangles']}", flush=True)
            mesh = obj.evaluated_get(dg).to_mesh()
            mesh.calc_loop_triangles()
            m = to_link @ np.array(obj.matrix_world)
            n_mat = np.linalg.inv(m[:3, :3]).T
            co = np.array([v.co[:] for v in mesh.vertices], dtype=np.float64)
            co = (np.c_[co, np.ones(len(co))] @ m.T)[:, :3]
            normals = np.array([n.vector[:] for n in mesh.corner_normals], dtype=np.float64) @ n_mat.T
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            loop_vert = np.array([l.vertex_index for l in mesh.loops], dtype=np.int64)
            tri_loops = np.array([t.loops[:] for t in mesh.loop_triangles], dtype=np.int64).reshape(-1, 3)
            tri_mat = np.array([t.material_index for t in mesh.loop_triangles], dtype=np.int64)
            slots = [s.material for s in obj.material_slots] or [None]
            for slot in sorted(set(tri_mat.tolist())):
                key = material_of(slots[min(slot, len(slots) - 1)])
                loops = tri_loops[tri_mat == slot]
                # one vertex per (position, normal): smooth where the model is smooth, creased where it is creased
                corner = np.c_[loop_vert[loops.ravel()], np.round(normals[loops.ravel()], 3)]
                _, first, inverse = np.unique(corner, axis=0, return_index=True, return_inverse=True)
                ids = loops.ravel()[first]
                parts.setdefault(key, []).append((co[loop_vert[ids]], normals[ids], inverse.reshape(-1, 3)))
            obj.evaluated_get(dg).to_mesh_clear()
        visuals = []
        for i, (key, chunks) in enumerate(sorted(parts.items(), key=lambda kv: -sum(len(c[2]) for c in kv[1]))):
            pos = np.concatenate([c[0] for c in chunks])
            nrm = np.concatenate([c[1] for c in chunks])
            offsets = np.cumsum([0] + [len(c[0]) for c in chunks[:-1]])
            tris = np.concatenate([c[2] + o for c, o in zip(chunks, offsets)])
            file = f"{link}_{i}.ply"
            write_ply(out / file, pos, nrm, tris)
            visuals.append({"shape": "mesh", "mesh": file, "material": [key[0], list(key[1]), key[2]], "triangles": int(len(tris))})
            total += len(tris)
            world = (np.c_[pos, np.ones(len(pos))] @ (np.linalg.inv(floor) @ frames[link]).T)[:, :3]
            if link == "chassis":  # the body's lowest point above the floor
                clearance = min(clearance, float(world[:, 2].min()))
            bounds = [np.minimum(bounds[0], world.min(0)), np.maximum(bounds[1], world.max(0))]
        links[link] = visuals
    size = bounds[1] - bounds[0]
    model = {"name": cfg["name"], "source": cfg["usd"], "frame": "x forward, y left, z up; the chassis frame is on the floor under the wheels' centre",
             "wheelbase": round(wheelbase, 4), "track": round(track, 4), "wheel_radius": round(radius, 4),
             "wheel_width": round(width, 4), "size": [round(float(v), 4) for v in size],
             "clearance": round(float(clearance), 4),
             "bounds": [[round(float(v), 4) for v in bounds[0]], [round(float(v), 4) for v in bounds[1]]],
             "triangles": int(total), "links": links}
    (out / "model.json").write_text(json.dumps(model, indent=1))
    print(f"[robot] {cfg['name']}: {len(links)} links, {total} triangles, wheelbase {wheelbase:.3f} m, track {track:.3f} m, "
          f"wheel radius {radius:.3f} m, size {size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f} m -> {out}", flush=True)


def main() -> int:
    import argparse
    import os
    import subprocess

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from opticalnav_sim import cycles_backend as cb

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usd", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--name", default=None, help="model name (default: the output folder's)")
    ap.add_argument("--max-triangles", type=int, default=150000, help="a mesh with more is decimated to this many")
    args = ap.parse_args()
    if not args.usd.is_file():
        raise SystemExit(f"no USD at {args.usd}")
    cfg = {"usd": str(args.usd.resolve()), "out": str(args.out.resolve()), "name": args.name or args.out.name,
           "max_triangles": int(args.max_triangles)}
    env = dict(os.environ, LD_LIBRARY_PATH=f"{cb.DRIVER_DIRS}:{cb.BLENDER.parent / 'lib'}:{cb.lib_dir()}")
    run = subprocess.run([str(cb.BLENDER), "-b", "--python", str(Path(__file__).resolve()), "--", json.dumps(cfg)], env=env,
                         capture_output=True, text=True)
    for line in run.stdout.splitlines():
        if line.startswith("[robot]"):
            print(line)
    if run.returncode or not (args.out / "model.json").is_file():
        print(run.stdout[-2000:], run.stderr[-2000:], file=sys.stderr)
        raise SystemExit("the conversion failed in Blender")
    return 0


if __name__ == "__main__":
    try:
        import bpy  # noqa: F401
    except ImportError:
        raise SystemExit(main())
    inside_blender(json.loads(sys.argv[sys.argv.index("--") + 1]))
