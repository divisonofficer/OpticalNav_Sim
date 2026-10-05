"""Browser GUI for driving the simulator by hand.

    python -m opticalnav_sim.gui --pack packs/opticalnav-v0.2 --server http://127.0.0.1:18770 [--port 18780]

Open http://127.0.0.1:18780. The page talks to this process, which holds one
MatterSim session and turns its Stokes frames into JPEGs (RGB preview, DoLP,
AoLP, S1/S0, S2/S0) and per-pixel readouts.

Rendering is progressive: every render is one pass of ``--pass-spp`` samples.
A move shows its first pass at once; while the camera stays put, passes with
new seeds accumulate into a running mean up to ``--target-spp``, and the next
key press starts over. The seed is an input of the server's freeze recording,
so one pass spp costs one recording (about 70 s, the first time) however many
passes run. Needs numpy and Pillow.
"""
from __future__ import annotations

import argparse
import hmac
import io
import json
import math
import os
import sys
import threading
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

from . import MatterSim, stokes

PAGE = Path(__file__).with_name("gui.html")
CHANNELS = ("rgb", "dolp", "aolp", "s1", "s2")


def channel_image(frame: dict, channel: str, scale: float) -> np.ndarray:
    """uint8 RGB image of one channel; ``scale`` is the DoLP (and |S1/S0|, |S2/S0|) shown at full colour."""
    if channel == "rgb":
        return frame["rgb"]
    d = frame["derived"]
    if channel == "dolp":
        return stokes.dolp_rgb(d["dolp"], scale)
    if channel == "aolp":
        return stokes.aolp_rgb(d["aolp_deg"], d["dolp"], scale)
    if channel in ("s1", "s2"):
        return stokes.signed_rgb(d[f"{channel}_over_s0"] / scale)
    raise KeyError(channel)


def jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img)).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class Session:
    """One simulator episode driven by the page. Calls are serialised by ``lock``."""

    """Every render is one pass of ``pass_spp`` samples. A move shows its first pass at once; while the
    camera stays put, ``refine`` adds passes with new seeds and shows the running mean, up to
    ``target_spp``. One spp value means one freeze recording on the render server."""

    def __init__(self, sim: MatterSim.Simulator, scans: dict[str, list[str]], pass_spp: int = 16,
                 target_spp: int = 1024, turn_deg: float = 15.0, pitch_deg: float = 10.0):
        self.sim, self.scans = sim, scans
        self.pass_spp, self.target_spp = int(pass_spp), int(target_spp)
        self.turn, self.pitch = math.radians(turn_deg), math.radians(pitch_deg)
        self.lock = threading.Lock()
        self.frame_id = 0
        self.frames: dict[int, dict] = {}  # the last four, so image requests that lag a pass or two still work
        self.acc: dict[str, np.ndarray] = {}  # Stokes sums of this camera's passes
        self.passes = 0
        self.trail: list[tuple[float, float]] = []
        self.started = False
        self.error = ""
        sim.setRenderSpp(self.pass_spp)
        sim.setRenderSeed(0)
        sim.setPreviewEnabled(False)  # the page tonemaps the running mean itself

    # --- episode ---
    def start(self, scene: str, viewpoint: str | None = None, heading: float | None = None) -> dict:
        graph = self.sim._graph(scene)
        if viewpoint is None:
            viewpoint = next(v for v, ok in zip(graph.ids, graph.included) if ok)
        h = self.sim.getState()[0].heading if heading is None and self.started else (heading or 0.0)
        self.trail = []
        return self._run(lambda: self.sim.newEpisode([scene], [viewpoint], [h], [0.0]))

    def goto(self, x: float, y: float) -> dict:
        """Jump to the graph viewpoint nearest to (x, y), keeping heading."""
        st = self.sim.getState()[0]
        graph = self.sim._graph(st.scanId)
        near = min((i for i, ok in enumerate(graph.included) if ok),
                   key=lambda i: math.hypot(graph.pos[i][0] - x, graph.pos[i][1] - y))
        return self._run(lambda: self.sim.newEpisode([st.scanId], [graph.ids[near]], [st.heading], [st.elevation]))

    def act(self, op: str, value: float = 0.0) -> dict:
        st = self.sim.getState()[0]
        if op == "forward":  # the candidate nearest the image centre, resolved now (queued keys may lag the page)
            op, value = "move", 1
        if op == "move":
            index = int(value)
            if not 0 < index < len(st.navigableLocations):
                raise ValueError(f"no candidate {index} here ({len(st.navigableLocations) - 1} visible)")
            return self._run(lambda: self.sim.makeAction([index], [0.0], [0.0]))
        if op == "turn":
            return self._run(lambda: self.sim.makeAction([0], [self.turn * float(value)], [0.0]))
        if op == "about":
            return self._run(lambda: self.sim.makeAction([0], [math.pi], [0.0]))
        if op == "pitch":
            return self._run(lambda: self.sim.makeAction([0], [0.0], [self.pitch * float(value)]))
        raise ValueError(f"unknown action {op!r}")

    def configure(self, settings: dict) -> dict:
        """Variant, pass spp, target spp, denoiser. Anything but the target restarts accumulation."""
        restart = False
        if "variant" in settings:
            scene = self.sim.getState()[0].scanId
            if settings["variant"] not in self.scans.get(scene, []):
                raise ValueError(f"variant {settings['variant']!r} is not available for {scene}")
            self.sim.setVariant(settings["variant"])
            restart = True
        if "pass_spp" in settings:
            self.pass_spp = int(settings["pass_spp"])
            self.sim.setRenderSpp(self.pass_spp)
            restart = True
        if "target_spp" in settings:
            self.target_spp = int(settings["target_spp"])
        if "denoise" in settings:
            self.sim.setDenoiser(bool(settings["denoise"]))
            restart = True
        if not restart:
            return self.state()
        self.passes = 0
        return self._pass()

    def refine(self, frame_id: int) -> dict:
        """Add one pass to the current camera, unless the viewer has moved on or the target is reached."""
        if frame_id != self.frame_id or self.passes * self.pass_spp >= self.target_spp:
            return self.state()
        return self._pass()

    # --- frames ---
    def _run(self, call) -> dict:
        """A camera change: render its first pass (seed 0) through the MatterSim API."""
        started = time.perf_counter()
        self.sim.setRenderSeed(0)
        try:
            call()
        except Exception as exc:
            self.error = str(exc)
            raise
        st = self.sim.getState()[0]
        self.started, self.error = True, ""
        pos = st._free_position or (st.location.x, st.location.y)
        if not self.trail or self.trail[-1] != (pos[0], pos[1]):
            self.trail.append((pos[0], pos[1]))
        self.passes = 0
        self._accumulate(st.stokes, time.perf_counter() - started)
        return self.state()

    def _pass(self) -> dict:
        """One more pass of the current camera; pass k uses seed k, so passes are independent samples."""
        st = self.sim.getState()[0]
        started = time.perf_counter()
        self.sim.setRenderSeed(self.passes)
        try:
            out = self.sim.renderCamera(st.scanId, st.camera_to_world)
        except Exception as exc:
            self.error = str(exc)
            raise
        finally:
            self.sim.setRenderSeed(0)
        self.error = ""
        self._accumulate(out, time.perf_counter() - started)
        return self.state()

    def _accumulate(self, stk: dict, wall_s: float) -> None:
        if self.passes == 0:
            self.acc = {k: stk[k].astype(np.float32) for k in ("s0", "s1", "s2")}
        else:
            for k in self.acc:
                self.acc[k] += stk[k]
        self.passes += 1
        s0, s1, s2 = (self.acc[k] / self.passes for k in ("s0", "s1", "s2"))
        timing = dict(getattr(self.sim._client, "last_timing", {}) or {})
        self.frame_id += 1
        self.frames[self.frame_id] = {"rgb": stokes.quick_preview(s0), "s0_luma": s0 @ stokes.LUMA,
                                      "derived": stokes.derived(s0, s1, s2), "spp": self.passes * self.pass_spp,
                                      "passes": self.passes, "variant": self.sim.variant,
                                      "wall_ms": 1000.0 * wall_s,
                                      "server_ms": 1000.0 * float(timing.get("render_s", 0.0)),
                                      "gpu_ms": 1000.0 * float(timing.get("gpu_s", 0.0))}
        for old in [k for k in self.frames if k < self.frame_id - 3]:
            del self.frames[old]

    def image(self, frame_id: int, channel: str, scale: float) -> bytes:
        return jpeg(channel_image(self.frames[frame_id], channel, scale))

    def probe(self, frame_id: int, x: int, y: int) -> dict:
        f = self.frames[frame_id]
        h, w = f["s0_luma"].shape
        x, y = min(max(int(x), 0), w - 1), min(max(int(y), 0), h - 1)
        d = f["derived"]
        return {"x": x, "y": y, "s0": float(f["s0_luma"][y, x]), "dolp": float(d["dolp"][y, x]),
                "aolp_deg": float(d["aolp_deg"][y, x]), "s1_over_s0": float(d["s1_over_s0"][y, x]),
                "s2_over_s0": float(d["s2_over_s0"][y, x])}

    # --- page data ---
    def graph(self, scene: str) -> dict:
        g = self.sim._graph(scene)
        keep = [i for i, ok in enumerate(g.included) if ok]
        return {"scene": scene, "x": [round(g.pos[i][0], 3) for i in keep], "y": [round(g.pos[i][1], 3) for i in keep]}

    def state(self) -> dict:
        sim = self.sim
        base = {"scans": self.scans, "pass_spp": self.pass_spp, "target_spp": self.target_spp,
                "denoise": sim.denoise, "variant": sim.variant, "started": self.started, "error": self.error,
                "turn_deg": round(math.degrees(self.turn), 3), "pitch_deg": round(math.degrees(self.pitch), 3),
                "hfov_deg": sim._hfov_deg(sim.width, sim.height), "vfov_deg": math.degrees(sim.vfov),
                "width": sim.width, "height": sim.height}
        if not self.started:
            return base
        st = sim.getState()[0]
        f = self.frames[self.frame_id]
        pos = st._free_position or (st.location.x, st.location.y)
        return dict(base, **{
            "frame_id": self.frame_id, "frame_spp": f["spp"], "frame_variant": f["variant"],
            "passes": f["passes"], "done": f["spp"] >= self.target_spp,
            "timing": {k: round(f[k], 1) for k in ("wall_ms", "server_ms", "gpu_ms")},
            "scene": st.scanId, "viewpoint": st.location.viewpointId, "step": st.step,
            "heading_deg": round(math.degrees(st.heading), 2), "elevation_deg": round(math.degrees(st.elevation), 2),
            "position": [round(pos[0], 3), round(pos[1], 3)],
            "candidates": [{"index": i, "viewpoint": c.viewpointId, "x": round(c.x, 3), "y": round(c.y, 3),
                            "rel_heading_deg": round(math.degrees(c.rel_heading), 2),
                            "rel_elevation_deg": round(math.degrees(c.rel_elevation), 2),
                            "distance_m": round(c.rel_distance, 3)}
                           for i, c in enumerate(st.navigableLocations) if i > 0],
            "trail": [[round(x, 3), round(y, 3)] for x, y in self.trail[-400:]],
        })


COOKIE = "opticalnav_gui"


def make_handler(session: Session, token: str | None = None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str, headers: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self, query: dict) -> bool:
            """With a token, a browser opens /?token=<token> once; that sets a cookie the page's own
            requests carry. Scripts may send 'Authorization: Bearer <token>' instead."""
            if not token:
                return True
            jar = SimpleCookie(self.headers.get("Cookie", ""))
            bearer = self.headers.get("Authorization", "")
            sent = [jar[COOKIE].value if COOKIE in jar else "", query.get("token", ""),
                    bearer[7:] if bearer.startswith("Bearer ") else ""]
            if any(hmac.compare_digest(v.encode(), token.encode()) for v in sent):
                return True
            self._json(401, {"error": "open the viewer as /?token=<token> (or send Authorization: Bearer <token>)"})
            return False

        def _json(self, code: int, payload) -> None:
            self._send(code, json.dumps(payload).encode(), "application/json")

        def do_GET(self):
            url = urlsplit(self.path)
            q = {k: v[-1] for k, v in parse_qs(url.query).items()}
            parts = [p for p in url.path.split("/") if p]
            if not self._authorized(q):
                return
            try:
                if not parts:
                    if token and q.get("token"):  # remember the token, then drop it from the address bar
                        return self._send(303, b"", "text/plain", {
                            "Set-Cookie": f"{COOKIE}={token}; HttpOnly; SameSite=Strict; Path=/", "Location": "/"})
                    return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                if parts == ["api", "state"]:
                    with session.lock:
                        return self._json(200, session.state())
                if parts == ["api", "graph"]:
                    with session.lock:
                        return self._json(200, session.graph(q["scene"]))
                if parts == ["api", "probe"]:
                    with session.lock:
                        return self._json(200, session.probe(int(q["frame"]), int(q["x"]), int(q["y"])))
                if len(parts) == 4 and parts[:2] == ["api", "frame"] and parts[3].endswith(".jpg"):
                    channel = parts[3][:-4]
                    if channel not in CHANNELS:
                        raise KeyError(channel)
                    with session.lock:
                        body = session.image(int(parts[2]), channel, float(q.get("scale", 0.2)))
                    return self._send(200, body, "image/jpeg")
                self._json(404, {"error": f"no route {url.path}"})
            except KeyError as exc:
                self._json(404, {"error": f"not found: {exc}"})
            except (ValueError, TypeError) as exc:
                self._json(400, {"error": str(exc)})

        def do_POST(self):
            url = urlsplit(self.path)
            if not self._authorized({}):
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
                with session.lock:
                    if url.path == "/api/start":
                        out = session.start(body["scene"], body.get("viewpoint"), body.get("heading"))
                    elif url.path == "/api/action":
                        out = session.act(body["op"], body.get("value", 0.0))
                    elif url.path == "/api/goto":
                        out = session.goto(float(body["x"]), float(body["y"]))
                    elif url.path == "/api/configure":
                        out = session.configure(body)
                    elif url.path == "/api/refine":
                        out = session.refine(int(body["frame_id"]))
                    else:
                        return self._json(404, {"error": f"no route {url.path}"})
                self._json(200, out)
            except (KeyError, ValueError, TypeError) as exc:
                self._json(400, {"error": f"{type(exc).__name__}: {exc}"})
            except Exception as exc:  # noqa: BLE001 - render server failures go to the page
                self._json(502, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--server", default=None, help="render server URL (default $OPTICALNAV_SIM_URL)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18780)
    ap.add_argument("--pass-spp", type=int, default=16, help="samples per render pass (moves show one pass)")
    ap.add_argument("--target-spp", type=int, default=1024, help="stop accumulating at this many samples")
    ap.add_argument("--turn-deg", type=float, default=15.0)
    ap.add_argument("--pitch-deg", type=float, default=10.0)
    ap.add_argument("--token", default=os.environ.get("OPTICALNAV_GUI_TOKEN"),
                    help="required to listen beyond localhost (default $OPTICALNAV_GUI_TOKEN); open /?token=<token>")
    args = ap.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.token:
        raise SystemExit("refusing to listen beyond localhost without --token (anyone could drive the simulator)")

    sim = MatterSim.Simulator()
    if args.server:
        sim.setDatasetPath(args.server)
    sim.setNavGraphPath(str(args.pack / "connectivity"))
    sim.initialize()
    info = sim._client.info()
    session = Session(sim, info["scans"], args.pass_spp, args.target_spp, args.turn_deg, args.pitch_deg)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(session, args.token))
    resident = [r[0] for r in info.get("resident", [])]
    print(f"[gui] http://{args.host}:{args.port}{'/?token=<token>' if args.token else ''}  render server {sim._client.url}  freeze={info.get('drjit_freeze')}  "
          f"resident={resident}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
