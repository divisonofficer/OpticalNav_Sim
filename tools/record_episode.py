#!/usr/bin/env python3
"""Record agents running in the simulator to an MP4 (every frame is a live render).

    python tools/record_episode.py --pack packs/opticalnav-v0.2 --scene SCENE --server URL \
        --mode polar --spp 64 --agents 1 --max-steps 15 --out demo_polar.mp4

Agents follow the shortest path to their episode's goal through the MatterSim
API (discretised 30 degree turns). With ``--agents N`` the simulator runs N
episodes as one batch, so every step is a single render call for all N views.
Each video frame shows the agents' RGB preview (plus DoLP and AoLP in polar mode), a
bird's-eye map of the graph with planned and travelled paths, and the step,
action and server render time.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opticalnav_sim import MatterSim, stokes  # noqa: E402
from opticalnav_sim.navgraph import NavGraph  # noqa: E402

COLORS = [(63, 201, 212), (240, 130, 80), (166, 142, 240), (76, 201, 148)]
BG, PANEL, INK, MUTED = (15, 19, 21), (22, 28, 31), (228, 233, 235), (140, 153, 160)


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except OSError:
        return ImageFont.load_default()


def polar_images(st) -> tuple[Image.Image, Image.Image]:
    """DoLP (black to red at 0.2) and AoLP (hue around the colour circle, brightness DoLP / 0.2)."""
    d = stokes.derived(*(st.stokes[k].astype(np.float32) for k in ("s0", "s1", "s2")))
    return Image.fromarray(stokes.dolp_rgb(d["dolp"], 0.2)), Image.fromarray(stokes.aolp_rgb(d["aolp_deg"], d["dolp"], 0.2))


class Map:
    def __init__(self, graph: NavGraph, size: int):
        xs = np.array([p[0] for p in graph.pos])
        ys = np.array([p[1] for p in graph.pos])
        self.size, self.graph = size, graph
        pad = 18
        span = max(xs.max() - xs.min(), ys.max() - ys.min(), 1e-6)
        self.scale = (size - 2 * pad) / span
        self.ox = pad - xs.min() * self.scale + (size - 2 * pad - (xs.max() - xs.min()) * self.scale) / 2
        self.oy = pad + ys.max() * self.scale + (size - 2 * pad - (ys.max() - ys.min()) * self.scale) / 2
        self.base = Image.new("RGB", (size, size), PANEL)
        draw = ImageDraw.Draw(self.base)
        for i, nbrs in enumerate(graph.adjacent):
            for j in nbrs:
                if j > i:
                    draw.line([self.xy(graph.pos[i]), self.xy(graph.pos[j])], fill=(50, 60, 66), width=1)

    def xy(self, p) -> tuple[float, float]:
        return self.ox + p[0] * self.scale, self.oy - p[1] * self.scale  # R2R frame: +y up on the map

    def draw(self, agents: list[dict]) -> Image.Image:
        img = self.base.copy()
        draw = ImageDraw.Draw(img)
        for a in agents:
            plan = [self.xy(self.graph.pos[self.graph.index(v)]) for v in a["plan"]]
            if len(plan) > 1:
                draw.line(plan, fill=(96, 120, 132), width=1)
            gx, gy = plan[-1]
            draw.ellipse([gx - 6, gy - 6, gx + 6, gy + 6], outline=(242, 181, 68), width=2)
            trail = [self.xy(p) for p in a["trail"]]
            if len(trail) > 1:
                draw.line(trail, fill=a["color"], width=3)
            x, y = trail[-1]
            h = a["heading"]  # MatterSim heading: 0 = +y (up on the map), positive turns right
            tip = (x + 13 * math.sin(h), y - 13 * math.cos(h))
            left = (x + 7 * math.sin(h - 2.5), y - 7 * math.cos(h - 2.5))
            right = (x + 7 * math.sin(h + 2.5), y - 7 * math.cos(h + 2.5))
            draw.polygon([tip, left, right], fill=a["color"])
        return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--split", default="val_unseen")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--mode", default="polar", choices=["polar", "rgb"])
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--agents", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    rows = [r for r in json.loads((args.pack / "tasks" / "R2R" / "data" / f"R2R_{args.split}.json").read_text())
            if r["scan"] == args.scene]
    rows.sort(key=lambda r: r["distance"])
    episodes = rows[: args.agents]
    graph = NavGraph.load(args.pack / "connectivity", args.scene)
    sim = MatterSim.Simulator()
    sim.setDatasetPath(args.server)
    sim.setNavGraphPath(str(args.pack / "connectivity"))
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(len(episodes))
    sim.setVariant(args.variant)
    sim.setRenderMode(args.mode)
    sim.setRenderSpp(args.spp)
    sim.initialize()
    sim.newEpisode([args.scene] * len(episodes), [e["path"][0] for e in episodes],
                   [e["heading"] for e in episodes], [0.0] * len(episodes))
    agents = [{"plan": graph.shortest_path(e["path"][0], e["path"][-1]), "goal": e["path"][-1],
               "color": COLORS[i % len(COLORS)], "trail": [], "done": False, "action": "start",
               "episode": e["episode_id"]} for i, e in enumerate(episodes)]

    view_w, view_h = 384, 288
    per_agent_w = view_w * (3 if args.mode == "polar" else 1)
    cols = 1 if len(agents) == 1 or args.mode == "polar" else 2
    rws = math.ceil(len(agents) / cols)
    map_size = max(rws * view_h, 360)
    W, H = cols * per_agent_w + map_size, 56 + max(rws * view_h, map_size)
    W, H = W + W % 2, H + H % 2
    bev = Map(graph, map_size)
    tmp = Path(tempfile.mkdtemp(prefix="opticalnav_rec_"))
    f_big, f_small, f_tag = font(17, True), font(13), font(12, True)
    started = time.time()
    try:
        for step in range(args.max_steps + 1):
            states = sim.getState()
            for a, st in zip(agents, states):
                a["trail"].append((st.location.x, st.location.y))
                a["heading"] = st.heading
            frame = Image.new("RGB", (W, H), BG)
            draw = ImageDraw.Draw(frame)
            timing = sim._client.last_timing
            draw.text((16, 9), f"OpticalNav polar simulator  ·  {args.scene}  ·  {args.variant}  ·  {args.mode} {args.spp} spp",
                      font=f_big, fill=INK)
            draw.text((16, 33), f"step {step}  ·  {len(agents)} agent(s) in one render call  ·  call {timing.get('render_s', 0):.1f} s "
                      f"(trace {timing.get('trace_s', 0):.1f} s, GPU {timing.get('gpu_s', 0):.2f} s)  ·  live MatterSim API",
                      font=f_small, fill=MUTED)
            for i, (a, st) in enumerate(zip(agents, states)):
                x0, y0 = (i % cols) * per_agent_w, 56 + (i // cols) * view_h
                rgb = Image.fromarray(np.ascontiguousarray(st.rgb[:, :, ::-1])).resize((view_w, view_h))
                frame.paste(rgb, (x0, y0))
                if args.mode == "polar":
                    dolp, aolp = polar_images(st)
                    frame.paste(dolp.resize((view_w, view_h)), (x0 + view_w, y0))
                    frame.paste(aolp.resize((view_w, view_h)), (x0 + 2 * view_w, y0))
                    draw.text((x0 + view_w + 8, y0 + 6), "DoLP 0-0.2", font=f_tag, fill=INK)
                    draw.text((x0 + 2 * view_w + 8, y0 + 6), "AoLP hue 0-180°, brightness DoLP", font=f_tag, fill=INK)
                draw.rectangle([x0 + 6, y0 + 6, x0 + 18, y0 + 18], fill=a["color"])
                label = f"{a['action']}  ·  {st.location.viewpointId}  ·  {math.degrees(st.heading):.0f}°"
                draw.text((x0 + 24, y0 + 5), label, font=f_tag, fill=INK)
            frame.paste(bev.draw(agents), (cols * per_agent_w, 56))
            frame.save(tmp / f"f{step:04d}.png")
            print(f"step {step}: render {timing.get('render_s', 0):.1f}s  elapsed {time.time() - started:.0f}s", flush=True)
            if all(a["done"] for a in agents) or step == args.max_steps:
                break
            actions = []
            for a, st in zip(agents, states):
                here = st.location.viewpointId
                if a["done"] or here == a["goal"]:
                    a["done"], a["action"] = True, "stop"
                    actions.append((0, 0, 0))
                    continue
                if here not in a["plan"]:
                    a["plan"] = graph.shortest_path(here, a["goal"])
                nxt = a["plan"][a["plan"].index(here) + 1]
                pick = next((k for k, loc in enumerate(st.navigableLocations[1:], 1) if loc.viewpointId == nxt), None)
                if pick is not None:
                    a["action"] = "forward"
                    actions.append((pick, 0, 0))
                else:
                    a["action"] = "turn right"
                    actions.append((0, 1, 0))
            sim.makeAction([a[0] for a in actions], [a[1] for a in actions], [a[2] for a in actions])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["/usr/bin/ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", str(tmp / "f%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(args.out)],
                       check=True)
        Image.open(tmp / "f0000.png").convert("RGB").save(args.out.with_suffix(".jpg"), quality=85)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {args.out} ({step + 1} frames, {time.time() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
