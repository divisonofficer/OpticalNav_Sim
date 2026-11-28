"""server.split_emitter_meshes and view_emitter_weights: a long light mesh cut into pieces, each an emitter of its own
(the same light: the same total area and radiance), and weights that favour the lights near the camera."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_a_long_light_strip_becomes_pieces(tmp_path, monkeypatch):
    mi = pytest.importorskip("mitsuba")
    try:
        mi.set_variant("scalar_rgb")
    except Exception as exc:  # noqa: BLE001 - a build without scalar_rgb
        pytest.skip(str(exc))
    from opticalnav_sim import bodies, server
    monkeypatch.setattr(server, "SPLIT_CACHE", tmp_path / "cache")
    (tmp_path / "assets").mkdir()
    # a 6 m x 0.04 m strip of two triangles, and a 0.5 m panel (short: left whole)
    v = np.array([[0, 2.5, 0], [6, 2.5, 0], [6, 2.5, 0.04], [0, 2.5, 0.04]], np.float32)
    bodies.write_ply(tmp_path / "assets" / "strip.ply", v, np.array([[0, 2, 1], [0, 3, 2]]))
    bodies.write_ply(tmp_path / "assets" / "panel.ply", v * [0.08, 1, 12], np.array([[0, 2, 1], [0, 3, 2]]))
    emit = '<emitter type="area"><rgb name="radiance" value="5, 5, 5" /></emitter>'
    note = '<!--opticalnav-obj:{"id":"strip_led","label":"strip","is_emitter":true}-->'
    text = (f'<scene version="3.0.0">\n  {note}\n  <shape type="ply" id="strip_led">\n    <string name="filename" value="assets/strip.ply" />\n'
            f'    {emit}\n  </shape>\n  <shape type="ply" id="panel">\n    <string name="filename" value="assets/panel.ply" />\n    {emit}\n  </shape>\n</scene>')
    out, n = server.split_emitter_meshes(mi, text, tmp_path)
    assert n == 6 and 'id="panel"' in out and 'id="strip_led"' not in out and '"id":"strip_led__e0"' in out
    mi.file_resolver().append(str(tmp_path))
    scene = mi.load_string(out)
    pieces = [s for s in scene.shapes() if s.id().startswith("strip_led__e")]
    assert len(pieces) == 6 and all(s.emitter() is not None for s in pieces)
    assert sum(float(s.surface_area()) for s in pieces) == pytest.approx(6 * 0.04, rel=1e-4)
    again, n2 = server.split_emitter_meshes(mi, text, tmp_path)   # from the cache, the same
    assert again == out and n2 == 6
    # weights: a near light over a far one of the same power; the mean 1; no light under 5 % of the mean
    w = server.view_emitter_weights({"a": 1.0, "b": 1.0, "c": 1.0},
                                    {"a": (np.array([1.0, 0, 0]), 0.1), "b": (np.array([20.0, 0, 0]), 0.1),
                                     "c": (np.array([200.0, 0, 0]), 0.1)}, np.zeros(3))
    assert w["a"] > 10 * w["b"] and np.mean(list(w.values())) == pytest.approx(1.0, rel=0.1) and min(w.values()) > 0
