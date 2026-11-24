#!/usr/bin/env python3
"""Stock Mitsuba reproducer of the CUDA error 700 (ILLEGAL_ADDRESS) that killed render servers; no opticalnav code.

    python3.10 tools/ser_crash_repro.py [--minutes 10] [--ser on|off] [--kind polar|rgb] [--partner busy|idle|none]

Two processes render the cornell box on one GPU: a polarized one (the victim) and a partner that keeps the GPU busy with
plain RGB renders. With shader execution reordering on (Dr.Jit's default on RTX 40 / 50) the polarized process dies of
error 700 within minutes; alone, next to an idle CUDA context, or as RGB it did not. Uses cuda_rgbn_polarized when the
build has it (RGB + near-infrared lanes: it died), else cuda_rgb_polarized (three lanes: not seen to die in 2700 renders).
A useful check after a driver update: ten minutes without a crash with --ser on is the sign the driver no longer does it.
"""
import argparse
import subprocess
import sys
import time


def victim(minutes: float, ser: bool, polar: bool) -> None:
    import drjit as dr
    import mitsuba as mi
    dr.set_flag(dr.JitFlag.ShaderExecutionReordering, ser)
    mi.set_variant(next(v for v in (("cuda_rgbn_polarized", "cuda_rgb_polarized", "cuda_ad_rgb_polarized") if polar
                                    else ("cuda_rgb", "cuda_ad_rgb")) if v in mi.variants()))
    scene = mi.cornell_box()
    scene["sensor"]["film"]["width"], scene["sensor"]["film"]["height"] = 1024, 768
    if polar:
        scene["integrator"] = {"type": "stokes", "nested": {"type": "path", "max_depth": 8}}
    scene, n, end = mi.load_dict(scene), 0, time.time() + 60 * minutes
    while time.time() < end:
        dr.eval(mi.render(scene, spp=64, seed=n))
        dr.sync_thread()
        n += 1
    print(f"{mi.variant()} SER {ser}: {n} renders", flush=True)


def idle(minutes: float) -> None:
    import drjit as dr
    from drjit.cuda import Float
    dr.eval(dr.arange(Float, 10))
    time.sleep(60 * minutes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--ser", choices=["on", "off"], default="on")
    ap.add_argument("--kind", choices=["polar", "rgb"], default="polar")
    ap.add_argument("--partner", choices=["busy", "idle", "none"], default="busy")
    ap.add_argument("--role", choices=["victim", "busy", "idle"], help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.role:
        idle(args.minutes) if args.role == "idle" else victim(args.minutes, args.ser == "on", args.role == "victim" and args.kind == "polar")
        return 0
    start = lambda role, minutes: subprocess.Popen([sys.executable, __file__, "--role", role, "--minutes", str(minutes), "--ser", args.ser, "--kind", args.kind],  # noqa: E731
                                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    partner = None if args.partner == "none" else start(args.partner, args.minutes + 0.2)
    end, crashes = time.time() + 60 * args.minutes, 0
    while time.time() < end - 5:
        p = start("victim", (end - time.time()) / 60)
        out = p.communicate()[0]
        if p.returncode:
            crashes += 1
            print(time.strftime("%H:%M:%S"), f"crash {crashes} (exit {p.returncode})", flush=True)
            if p.returncode == 1:   # an exception of its own (no such variant, no OptiX), not the abort of error 700
                print("the process could not run: start it by hand with --role victim to see why")
                break
        else:
            print(out.strip(), flush=True)
    if partner is not None:
        partner.kill()
    print(f"{crashes} crashes in {args.minutes:g} minutes (SER {args.ser}, {args.kind}, partner {args.partner})")
    return 1 if crashes else 0


if __name__ == "__main__":
    raise SystemExit(main())
