"""Checks that need no GPU. Run: python -m pytest tests -q   (or python tests/test_sim.py)

The pose-parity test reads the dataset reference frames stored in a scene pack
($OPTICALNAV_PACK, default packs/opticalnav-v0.2) and is skipped when none exists.
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opticalnav_sim import MatterSim, frames, server  # noqa: E402
from opticalnav_sim.client import RenderClient  # noqa: E402
from opticalnav_sim.navgraph import NavGraph  # noqa: E402

PACK = Path(os.environ.get("OPTICALNAV_PACK", Path(__file__).resolve().parents[1] / "packs" / "opticalnav-v0.2"))
MOUNT = {"xyz_m": [0.0, 1.5, 0.1], "rpy_deg": [0.0, 0.0, 0.0]}


def _rows(points, edges, z=1.5):
    n = len(points)
    adj = [[False] * n for _ in range(n)]
    for a, b in edges:
        adj[a][b] = adj[b][a] = True
    return [{"image_id": f"v{i}", "pose": [1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1],
             "included": True, "unobstructed": adj[i], "height": z} for i, (x, y) in enumerate(points)]


def test_pose_parity_with_dataset_manifests():
    """Viewpoint + heading in the R2R frame must reproduce the dataset camera matrices of the pack's reference frames."""
    scenes = sorted(PACK.glob("scenes/*/scene.json")) if PACK.is_dir() else []
    checked = 0
    for meta_path in scenes:
        meta = json.loads(meta_path.read_text())
        if not meta.get("reference_views"):
            continue
        graph = NavGraph.load(PACK / "connectivity", meta["scan"])
        for ref in meta["reference_views"]:
            manifest = json.loads((meta_path.parent / ref["path"] / "manifest.json").read_text())
            spec = manifest["camera_specs"][0]
            loc = graph.viewpoint(graph.index(ref["node_id"]))
            heading = frames.heading_from_yaw(math.radians(int(ref["heading_id"].split("_")[1])))
            got = frames.camera_at_viewpoint(loc.x, loc.y, heading, 0.0, spec["extras"]["robot_mount"])
            want = frames.legacy_flat_to_matrix(spec["camera_to_world"])
            assert np.abs(got - want).max() < 1e-9, (meta["scan"], ref, np.abs(got - want).max())
            checked += 1
    if not checked:
        print("skip: no pack with reference views")
        return
    print(f"pose parity: {checked} dataset reference views reproduced from connectivity + heading")


def test_frames_mattersim_conventions():
    # heading 0 looks along +y of the R2R frame and image-right is +x (MatterSim: positive heading turns right)
    c2w = frames.camera_at_position([0.0, 0.0, 1.5], 0.0, 0.0, MOUNT)
    forward_dataset = -c2w[:3, 2]
    assert abs(forward_dataset[0]) < 1e-12 and forward_dataset[2] < 0  # dataset y decreases == R2R y increases
    assert c2w[0, 0] > 0.99  # image right = +x
    right = frames.camera_at_position([0.0, 0.0, 1.5], math.pi / 2, 0.0, MOUNT)
    assert -right[0, 2] > 0.98  # heading +90 deg (turn right) faces +x
    up = frames.camera_at_position([0.0, 0.0, 1.5], 0.0, 0.3, MOUNT)
    assert -up[1, 2] > -c2w[1, 2]  # positive elevation looks up
    assert abs(frames.yaw_from_heading(frames.heading_from_yaw(1.234)) - 1.234) < 1e-12


def test_navigable_matches_mattersim_rules():
    # v0 at origin; v1 ahead (+y), v2 to the right (+x), v3 behind (-y)
    g = NavGraph.from_connectivity("s", _rows([(0, 0), (0, 2), (2, 0), (0, -2)], [(0, 1), (0, 2), (0, 3)]))
    hfov = 0.8 * 640 / 480
    cands = g.navigable(0, 0.0, 0.0, hfov, restricted=True)
    assert [c.viewpointId for c in cands] == ["v0", "v1"]
    allc = g.navigable(0, 0.0, 0.0, hfov, restricted=False)
    assert [c.viewpointId for c in allc] == ["v0", "v1", "v2", "v3"]
    assert math.isclose(allc[2].rel_heading, math.pi / 2)  # right of the camera is positive
    assert math.isclose(allc[1].rel_distance, 2.0)
    assert g.shortest_path("v1", "v2") == ["v1", "v0", "v2"]


def test_flash_matrix_matches_production_staging():
    # production stages the flash for an identity camera; these are the matrices it wrote into the XML
    light = server.flash_matrix(np.eye(4), [0.12, 0.08], 0.2)
    polarizer = server.flash_matrix(np.eye(4), [0.12, 0.08], 0.21, 0.0)
    want = np.array([[0.06, 0, 0, 0], [0, 0.04, 0, 0], [0, 0, -1, 0.2], [0, 0, 0, 1]])
    assert np.abs(light - want).max() < 1e-7
    assert abs(polarizer[2, 3] - 0.21) < 1e-12
    rolled = server.flash_matrix(np.eye(4), [0.12, 0.08], 0.2, 90.0)
    assert np.abs(rolled[:3, 0] - [0, 0.06, 0]).max() < 1e-9  # roll turns image-right into up


class FakeClient:
    def __init__(self, rows):
        self.rows, self.views, self.last_render_seconds = rows, [], 0.0

    def connectivity(self, scan):
        return self.rows

    def meta(self, scan):
        return {"camera": {"mount": MOUNT}}

    def render(self, views, dtype):
        self.views.extend(views)
        out = []
        for v in views:
            rgb = np.zeros((v["height"], v["width"], 3), np.uint8)
            rgb[..., 0] = 255  # pure red in RGB
            s = np.zeros((v["height"], v["width"], 3), np.float16)
            out.append({"rgb": rgb, "s0": s, "s1": s, "s2": s, "s3": s})
        return out

    def close(self):
        pass


def test_simulator_discretized_actions_follow_mattersim():
    sim = MatterSim.Simulator()
    sim.setDiscretizedViewingAngles(True)
    sim.setCameraResolution(64, 48)
    sim.initialize()
    sim._client = FakeClient(_rows([(0, 0), (0, 2), (2, 0)], [(0, 1), (0, 2)]))
    sim.newEpisode(["scene"], ["v0"], [0.1], [0.0])
    st = sim.getState()[0]
    assert st.heading == 0.0 and st.viewIndex == 12 and st.rgb.shape == (48, 64, 3)
    assert st.radiance is st.stokes["s0"] and sim._client.views[-1]["mode"] == "polar"
    assert st.rgb[0, 0].tolist() == [0, 0, 255]  # BGR like MatterSim
    assert [c.viewpointId for c in st.navigableLocations] == ["v0", "v1"]
    sim.makeAction([0], [5.0], [0.0])  # any positive heading = one 30 degree right turn
    st = sim.getState()[0]
    assert math.isclose(st.heading, math.pi / 6) and st.viewIndex == 13 and st.step == 1
    sim.makeAction([0], [0.0], [1.0])
    assert sim.getState()[0].viewIndex == 25
    for _ in range(2):
        sim.makeAction([0], [1.0], [0.0])
    sim.makeAction([0], [0.0], [-1.0])
    st = sim.getState()[0]
    assert math.isclose(st.heading, math.pi / 2) and st.navigableLocations[1].viewpointId == "v2"
    sim.makeAction([1], [0.0], [0.0])
    assert sim.getState()[0].location.viewpointId == "v2"
    try:
        sim.makeAction([5], [0.0], [0.0])
        raise AssertionError("invalid index accepted")
    except ValueError:
        pass
    # the camera sent to the renderer is the dataset rig camera at that viewpoint and heading
    view = sim._client.views[-1]
    want = frames.camera_at_viewpoint(2.0, 0.0, math.pi / 2, 0.0, MOUNT)
    assert np.abs(np.asarray(view["camera_to_world"]) - want).max() < 1e-12
    assert view["variant"] == "base"
    sim.newEpisode(["scene__active_polar"], ["v0"], [0.0], [0.0])
    assert sim._client.views[-1]["variant"] == "active_polar"
    sim.teleport(["scene"], [[0.3, 0.1, 1.2]], [0.0], [0.0])
    st = sim.getState()[0]
    assert st.location.viewpointId == "v0" and abs(st.camera_to_world[1, 3] - 1.2) < 1e-12


class FakeRenderer:
    pack = Path(".")
    variant, has_nocaustics, freeze, cache = "fake", False, False, {}
    variant_of = {"polar": "fake", "rgb": "fake_rgb"}
    meta = {"scene": {"variants": {"base": {}}}}

    def variants(self, scan):
        return ["base"]

    def render_views(self, views, dtype):
        out = []
        for view in views:
            if view["scan"] != "scene":
                raise ValueError("unknown scan")
            h, w = view["height"], view["width"]
            keys = ("radiance",) if view.get("mode") == "rgb" else ("s0", "s1", "s2", "s3")
            out.append({"rgb": np.full((h, w, 3), 7, np.uint8), **{k: np.full((h, w, 3), 0.5, dtype) for k in keys}})
        return out, {"render_s": 0.01, "denoise_s": 0.0, "post_s": 0.0, "views": len(views)}


def test_http_roundtrip():
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(FakeRenderer()))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = RenderClient(f"http://127.0.0.1:{httpd.server_address[1]}")
        assert client.info()["mitsuba_variant"] == "fake"
        view = {"scan": "scene", "variant": "base", "camera_to_world": np.eye(4), "width": 8, "height": 6,
                "hfov_deg": 90.0, "spp": 4, "seed": 0}
        out = client.render([view, view], "float32")
        assert len(out) == 2 and out[1]["s2"].dtype == np.float32 and out[0]["rgb"].shape == (6, 8, 3)
        assert client.last_timing["render_s"] > 0
        rgb_out = client.render([dict(view, mode="rgb")])
        assert set(rgb_out[0]) == {"rgb", "radiance"}
        try:
            client.render([dict(view, scan="nope")])
            raise AssertionError("bad scan accepted")
        except RuntimeError as exc:
            assert "400" in str(exc)
    finally:
        httpd.shutdown()
    secured = server.ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(FakeRenderer(), token="s3cret"))
    threading.Thread(target=secured.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{secured.server_address[1]}"
        try:
            RenderClient(url, token="wrong").info()
            raise AssertionError("wrong token accepted")
        except RuntimeError as exc:
            assert "401" in str(exc)
        assert RenderClient(url, token="s3cret").info()["mitsuba_variant"] == "fake"
    finally:
        secured.shutdown()


def test_gui_session_and_routes():
    from urllib.request import Request, urlopen

    from opticalnav_sim import gui

    sim = MatterSim.Simulator()
    sim.setCameraResolution(64, 48)
    sim.initialize()
    sim._client = FakeClient(_rows([(0, 0), (0, 2), (2, 0)], [(0, 1), (0, 2)]))
    session = gui.Session(sim, {"scene": ["base", "perturbed"]}, pass_spp=16, target_spp=48)
    st = session.start("scene", "v0", 0.0)
    assert st["frame_spp"] == 16 and st["passes"] == 1 and not st["done"]
    assert [c["viewpoint"] for c in st["candidates"]] == ["v1"]
    first = sim._client.views[-1]
    assert first["spp"] == 16 and first["seed"] == 0 and first["preview"] is False
    for want in (2, 3):  # passes keep the spp (one freeze recording) and change the seed
        st = session.refine(st["frame_id"])
        assert st["passes"] == want and sim._client.views[-1]["seed"] == want - 1 and sim._client.views[-1]["spp"] == 16
    assert st["done"] and st["frame_spp"] == 48
    assert session.refine(st["frame_id"])["passes"] == 3  # target reached: no more passes
    assert session.refine(st["frame_id"] - 1)["frame_id"] == st["frame_id"]  # stale refine is a no-op
    st = session.act("forward")
    assert st["passes"] == 1 and sim._client.views[-1]["seed"] == 0  # a move starts over
    assert st["viewpoint"] == "v1" and st["step"] == 1 and st["trail"] == [[0.0, 0.0], [0.0, 2.0]]
    st = session.act("turn", 6)  # 6 x 15 degrees
    assert abs(st["heading_deg"] - 90.0) < 1e-6
    st = session.configure({"variant": "perturbed"})
    assert sim._client.views[-1]["variant"] == "perturbed" and st["frame_variant"] == "perturbed"
    st = session.goto(1.9, 0.1)
    assert st["viewpoint"] == "v2" and st["step"] == 0
    for ch in gui.CHANNELS:
        assert session.image(st["frame_id"], ch, 0.2)[:2] == b"\xff\xd8"  # JPEG
    assert session.probe(st["frame_id"], 999, -5)["x"] == 63

    httpd = gui.ThreadingHTTPServer(("127.0.0.1", 0), gui.make_handler(session))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        assert b"OpticalNav Sim Viewer" in urlopen(base + "/").read()
        post = Request(base + "/api/action", data=json.dumps({"op": "pitch", "value": 1}).encode(),
                       headers={"Content-Type": "application/json"})
        st = json.loads(urlopen(post).read())
        assert abs(st["elevation_deg"] - 10.0) < 1e-6
        assert urlopen(f"{base}/api/frame/{st['frame_id']}/aolp.jpg").read()[:2] == b"\xff\xd8"
        assert len(json.loads(urlopen(base + "/api/graph?scene=scene").read())["x"]) == 3
        try:
            urlopen(Request(base + "/api/action", data=b'{"op": "move", "value": 7}'))
            raise AssertionError("invalid candidate accepted")
        except Exception as exc:  # urllib raises HTTPError for 400
            assert "400" in str(exc)
    finally:
        httpd.shutdown()

    locked = gui.ThreadingHTTPServer(("127.0.0.1", 0), gui.make_handler(session, token="s3cret"))
    threading.Thread(target=locked.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{locked.server_address[1]}"
        for path in ("/", "/api/state", f"/api/state?token=wrong"):
            try:
                urlopen(base + path)
                raise AssertionError(f"{path} served without the token")
            except Exception as exc:
                assert "401" in str(exc)
        assert json.loads(urlopen(Request(base + "/api/state", headers={"Authorization": "Bearer s3cret"})).read())["started"]
        assert json.loads(urlopen(Request(base + "/api/state", headers={"Cookie": "opticalnav_gui=s3cret"})).read())["started"]
        # /?token= sets the cookie and redirects to / (urllib follows the redirect but keeps no cookies)
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", locked.server_address[1])
        conn.request("GET", "/?token=s3cret")
        resp = conn.getresponse()
        assert resp.status == 303 and "opticalnav_gui=s3cret" in resp.getheader("Set-Cookie") and resp.getheader("Location") == "/"
        conn.close()
    finally:
        locked.shutdown()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
