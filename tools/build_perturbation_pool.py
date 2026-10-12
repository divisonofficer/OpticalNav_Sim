#!/usr/bin/env python3
"""Build each pack scene's pool of candidate mirrors and glass walls with robomituba's optical perturbation.

    python tools/build_perturbation_pool.py --pack packs/opticalnav-v0.2 [--scene SCENE ...]
        [--robomituba /jarvis/project/robomituba] [--density 4.0] [--glass 6] [--seed 0]

robomituba's ``navigation_dataset.optical_perturbation`` places the perturbed variant's mirrors and glass
walls: mirrors on wall runs taken from the Infinigen wall meshes (count per room = wall length x 0.07 x
density, kept away from doors, windows and nav edges), glass walls across the longest graph edges whose
removal leaves a detour. Here it runs unchanged at a high density, on the scene's authoring map and wall
meshes and on the pack's R2R graph (its viewpoints are the support poses; connectivity (x, y) is authoring
(x, -y)), and the result is kept as a pool next to the scene:

    scenes/<scene>/perturbation_pool.json
        mirrors: [{id, rank, geometry, roughness, xml}]   rank: the order a density adds them in
        glass:   [{id, rank, splits_r2r, geometry, blocked_edges, xml}]  blocked_edges: R2R viewpoint pairs it
                 cuts; splits_r2r: blocking them leaves no detour (walls that do not come first in rank)
        bsdf_xml: shared BSDFs the shapes reference

The render server loads the whole pool into the resident scene, hidden, and shows a density's share of it
by rank (``"mirrors": {"density": d, "glass": k}`` on a view), so changing the density edits memory and
records nothing. Shapes are cubes converted as robomituba's render daemon does (_wall_shape_xml_element),
centred at half their height: like the dataset's perturbed scenes, the mirrors stand on the floor
(``base_height_m`` is not applied). ``--mounted`` applies it.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path


def r2r_graph(pack: Path, scene: str) -> tuple[dict, dict[str, tuple[float, float]], list[tuple[str, str, str]]]:
    """The pack's R2R graph in the authoring frame: (graph json for the generator, node xy, edges)."""
    conn = json.loads((pack / "connectivity" / f"{scene}_connectivity.json").read_text())
    pos = {r["image_id"]: (float(r["pose"][3]), -float(r["pose"][7])) for r in conn if r.get("included", True)}
    edges, seen = [], set()
    for r in conn:
        a = r["image_id"]
        for j, ok in enumerate(r.get("unobstructed", [])):
            b = conn[j]["image_id"]
            if ok and a != b and a in pos and b in pos and (b, a) not in seen:
                seen.add((a, b))
                edges.append((f"{a}|{b}", a, b))
    graph = {"nodes": [{"node_id": n, "position": list(xy)} for n, xy in pos.items()],
             "edges": [{"edge_id": e, "source": a, "target": b} for e, a, b in edges]}
    return graph, pos, edges


def cube_xml(obj: dict, bsdf_id: str, mounted: bool) -> str:
    """robomituba render_daemon._wall_shape_xml_element: a cube between the line's ends."""
    g = obj["geometry"]
    (x1, z1), (x2, z2) = g["start"], g["end"]
    length = math.hypot(x2 - x1, z2 - z1)
    h, t = float(g.get("height_m") or 2.4), float(g.get("thickness_m") or 0.08)
    cy = h / 2 + (float(g.get("base_height_m") or 0.0) if mounted else 0.0)
    angle = math.degrees(math.atan2(z2 - z1, x2 - x1))
    return (f'\n  <shape type="cube" id="{obj["id"]}">\n    <transform name="to_world">\n'
            f'      <scale x="{length / 2:.6f}" y="{h / 2:.6f}" z="{t / 2:.6f}" />\n'
            f'      <rotate y="1" angle="{angle:.4f}" />\n'
            f'      <translate x="{(x1 + x2) / 2:.6f}" y="{cy:.6f}" z="{(z1 + z2) / 2:.6f}" />\n'
            f'    </transform>\n    <ref id="{bsdf_id}" />\n  </shape>')


def build(pack: Path, scene: str, robomituba: Path, density: float, glass_max: int, seed: int, mounted: bool) -> dict:
    sys.path.insert(0, str(robomituba / "modules" / "navigation_dataset" / "src"))
    from navigation_dataset import optical_perturbation as op  # noqa: PLC0415
    from navigation_dataset.object_footprint import object_footprint  # noqa: PLC0415

    meta = json.loads((pack / "scenes" / scene / "scene.json").read_text())
    src = next((Path(v["source_xml"]) for v in meta["variants"].values() if v.get("source_xml")), None)
    scene_dir = src.parents[2] if src is not None else robomituba / "out" / "opticalnav" / pack.name / "scenes" / scene
    authoring = json.loads((scene_dir / "authoring_map.json").read_text())
    walls = op._wall_segments_from_wall_meshes(scene_dir)
    wall_source = "wall_mesh_body_band"
    if not walls:
        walls, wall_source = op._rect_segments(op._room_rects(authoring)), "room_rect_fallback"
    openings = op._opening_points(authoring)
    graph, pos, edges = r2r_graph(pack, scene)
    world_edges = [(pos[a], pos[b]) for _, a, b in edges]
    rng = random.Random(seed)
    mirrors = op.place_mirrors(walls, openings, world_edges, rng=rng, density=density)
    # Glass walls need a graph with detours: robomituba's viewpoint graph (place_graph.json), not the R2R
    # support graph, whose 0.25 m lanes any 1.4 m wall cuts in two. The R2R edges each one blocks are then
    # found geometrically.
    place = json.loads((scene_dir / "place_graph.json").read_text()) if (scene_dir / "place_graph.json").is_file() else graph
    glass, _ = op.place_glass_walls(authoring, place, rng=rng, max_walls=glass_max)
    op.assign_mirror_roughness(mirrors, seed, str(authoring.get("scene_id") or scene))

    order = random.Random(f"pool-order:{seed}:{scene}")
    ranks = list(range(len(mirrors)))
    order.shuffle(ranks)
    # Every pool shape gets its own BSDF, a millionth apart: the loader merges meshes that share an (equal)
    # BSDF, and a pool shape merged into, say, the scene's window glass would rewrite that whole mesh on
    # every density change (850k vertices in a Device 1 apartment).
    bsdfs = []
    out_m, out_g = [], []
    for i, (rank, m) in enumerate(zip(ranks, mirrors)):
        sid = "pool_" + m["id"]
        alpha = float((m.get("material_params") or {}).get("roughness", 0.0)) + 1e-6 * (i + 1)
        bsdfs.append(f'  <bsdf type="twosided" id="pool_bsdf_{sid}"><bsdf type="roughconductor">'
                     f'<string name="material" value="Ag" /><float name="alpha" value="{alpha:.7f}" /></bsdf></bsdf>')
        out_m.append({"id": sid, "rank": rank, "geometry": m["geometry"], "roughness": alpha,
                      "xml": cube_xml(dict(m, id=sid), f"pool_bsdf_{sid}", mounted)})
    nodes_xy = {n: xy for n, xy in pos.items()}
    base_parts = op._num_components(set(pos), [(a, b) for _, a, b in edges])
    for g in glass:  # placement order: longest edges first
        sid = "pool_" + g["id"]
        fp = object_footprint(dict(g["geometry"], type="line"), margin=float(g["geometry"].get("thickness_m") or 0.06))
        pairs = [[a, b] for _, a, b in edges if fp is not None and op._edge_crosses_footprint(nodes_xy[a], nodes_xy[b], fp)]
        # robomituba's rule (a detour must remain) checked on the R2R graph too: its 0.25 m lanes rarely
        # leave one, so a wall that blocks a lane usually cuts part of the scene off. Kept, but flagged.
        alone = {tuple(p) for p in pairs}
        splits = op._num_components(set(pos), [(a, b) for _, a, b in edges if (a, b) not in alone]) != base_parts
        bsdfs.append(f'  <bsdf type="dielectric" id="pool_bsdf_{sid}"><float name="int_ior" '
                     f'value="{1.5 + 1e-6 * (len(out_g) + 1):.7f}" /></bsdf>')
        out_g.append({"id": sid, "rank": 0, "splits_r2r": splits, "geometry": g["geometry"], "blocked_edges": pairs,
                      "xml": cube_xml(dict(g, id=sid), f"pool_bsdf_{sid}", mounted)})
    out_g.sort(key=lambda g: g["splits_r2r"])  # walls that leave the R2R graph whole come first
    for rank, g in enumerate(out_g):
        g["rank"] = rank
    blocked = {tuple(e) for g in out_g for e in g["blocked_edges"]}
    parts_before = op._num_components(set(pos), [(a, b) for _, a, b in edges])
    parts_after = op._num_components(set(pos), [(a, b) for _, a, b in edges if (a, b) not in blocked and (b, a) not in blocked])
    return {"version": "opticalnav-sim-perturbation-pool-v1", "scene": scene,
            "r2r_components": [parts_before, parts_after],
            "source": "robomituba navigation_dataset.optical_perturbation", "seed": seed, "density_max": density,
            "glass_max": glass_max, "wall_source": wall_source, "mounted": mounted,
            "mirrors": sorted(out_m, key=lambda m: m["rank"]), "glass": out_g, "bsdf_xml": "\n".join(bsdfs)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--robomituba", type=Path, default=Path("/jarvis/project/robomituba"))
    ap.add_argument("--density", type=float, default=4.0, help="robomituba mirror density of the full pool")
    ap.add_argument("--glass", type=int, default=6, help="most glass walls in the pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mounted", action="store_true", help="mount mirrors at base_height_m (the dataset does not)")
    args = ap.parse_args()
    pack = args.pack.resolve()
    scenes = args.scene or sorted(p.parent.name for p in (pack / "scenes").glob("*/scene.json"))
    for scene in scenes:
        try:
            pool = build(pack, scene, args.robomituba, args.density, args.glass, args.seed, args.mounted)
        except (OSError, KeyError, ValueError) as exc:
            print(f"[skip] {scene}: {type(exc).__name__}: {exc}")
            continue
        (pack / "scenes" / scene / "perturbation_pool.json").write_text(json.dumps(pool, indent=1))
        print(f"[pool] {scene}: {len(pool['mirrors'])} mirrors ({pool['wall_source']}), {len(pool['glass'])} glass walls "
              f"({sum(not g['splits_r2r'] for g in pool['glass'])} leave the R2R graph whole) "
              f"cutting {sum(len(g['blocked_edges']) for g in pool['glass'])} R2R edges "
              f"(R2R graph parts {pool['r2r_components'][0]} -> {pool['r2r_components'][1]} with all of them)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
