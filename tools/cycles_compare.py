#!/usr/bin/env python3
"""Render a pose of a pack scene through the running server and set it beside a Cycles image, at Blender's own
view transform (AgX, exposure 0), so the two are comparable in brightness.

    python3 tools/cycles_compare.py --server http://127.0.0.1:18770 --pack packs/opticalnav-sim-v1 \
        --scene sim_v2a_cafeS_lphigh_47312401 --origin 9.793 1.5 4.835 --forward 0 -0.148 0.989 \
        --cycles report_assets/cycles/lphigh_viewQ.png --out /tmp/q.jpg [--spp 256]

The poses are the Mitsuba frame's (Y up). The viewer's own tone curve fits its exposure to each frame, so its
pictures say nothing about absolute brightness against a Cycles render; this script writes the radiance through
AgX at exposure 1, the approximation of Blender's AgX base transform used by three.js (Rec.2020 inset, log2
between -12.47 and 4.03 EV, a polynomial contrast curve). Prints the mean of both pictures (display and
linear) and the ratio of their linear means.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opticalnav_sim import MatterSim  # noqa: E402

SRGB_TO_REC2020 = np.array([[0.6274, 0.3293, 0.0433], [0.0691, 0.9195, 0.0114], [0.0164, 0.0880, 0.8956]])
REC2020_TO_SRGB = np.array([[1.6605, -0.5876, -0.0728], [-0.1246, 1.1329, -0.0083], [-0.0182, -0.1006, 1.1187]])
INSET = np.array([[0.856627153315983, 0.0951212405381588, 0.0482516061458583],
                  [0.137318972929847, 0.761241990602591, 0.101439036467306],
                  [0.11189821299995, 0.0767994186031903, 0.811302368396859]])
OUTSET = np.array([[1.1271005818144368, -0.1413297634984383, -0.14132976349843826],
                   [-0.11060664309660323, 1.157823702216272, -0.11060664309660294],
                   [-0.016493938717834573, -0.016493938717834573, 1.2519364065950405]])
OUTSET = OUTSET.T  # rows sum to 1, so grey stays grey
MIN_EV, MAX_EV = -12.47393, 4.026069


def agx(radiance: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """Linear sRGB radiance -> display-linear sRGB, three.js's AgX approximation."""
    x = np.clip(np.nan_to_num(radiance.astype(np.float64)), 0.0, 1e4) * exposure
    x = np.maximum(x @ SRGB_TO_REC2020.T @ INSET.T, 1e-10)
    x = np.clip((np.log2(x) - MIN_EV) / (MAX_EV - MIN_EV), 0.0, 1.0)
    x2 = x * x
    x4 = x2 * x2
    x = (15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232)
    x = np.maximum(x @ OUTSET.T, 0.0) ** 2.2
    return np.clip(x @ REC2020_TO_SRGB.T, 0.0, 1.0)


def srgb(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 1e-9), 1 / 2.4) - 0.055)


def lin(a: np.ndarray) -> np.ndarray:
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def c2w(origin, forward) -> np.ndarray:
    f = np.asarray(forward, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    m = np.stack([right, up, -f, np.asarray(origin, float)], axis=1)
    return np.vstack([m, [0.0, 0.0, 0.0, 1.0]])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:18770")
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--origin", nargs=3, type=float, required=True)
    ap.add_argument("--forward", nargs=3, type=float, required=True)
    ap.add_argument("--cycles", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--env", help="an environment map of the server's --envmaps, instead of the scene's own sky")
    ap.add_argument("--env-scale", type=float, default=1.0, help="its radiance scale (0 darkens the sky)")
    args = ap.parse_args()
    sim = MatterSim.Simulator()
    sim.setDatasetPath(args.server)
    sim.setNavGraphPath(str(args.pack / "connectivity"))
    sim.setVariant("base")
    sim.setRenderMode("polar")
    sim.setStokesDtype("float32")
    sim.setCameraResolution(512, 384)
    sim.setRenderSpp(args.spp)
    if args.env:
        sim.setEnvironment(args.env, args.env_scale, 0.0)
    sim.initialize()
    r = sim.renderCamera(args.scene, c2w(args.origin, args.forward))
    ours = agx(np.asarray(r["s0"], np.float32))
    cycles = np.asarray(Image.open(args.cycles).convert("RGB").resize((512, 384))).astype(np.float64) / 255.0
    a = lin(cycles)
    print(f"Cycles: display mean {cycles.mean():.3f}, linear {a.mean():.3f}")
    print(f"ours  : display mean {srgb(ours).mean():.3f}, linear {ours.mean():.3f}; Cycles/ours linear {a.mean() / max(ours.mean(), 1e-9):.2f}")
    sheet = Image.new("RGB", (1024, 384))
    for j, (img, label) in enumerate(((cycles, "Cycles"), (srgb(ours), "ours, AgX exposure 1"))):
        im = Image.fromarray(np.rint(np.clip(img, 0, 1) * 255).astype(np.uint8))
        ImageDraw.Draw(im).text((4, 2), label, fill="white")
        sheet.paste(im, (j * 512, 0))
    sheet.save(args.out, quality=90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
