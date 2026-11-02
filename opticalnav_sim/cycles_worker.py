"""Cycles render worker: runs inside Blender (``blender -b scene.blend --python cycles_worker.py -- CMD_FD REPLY_FD``).

The scene is loaded once. Each request is one JSON line on the command pipe, answered by one JSON line on the reply
pipe (Blender's own stdout is noisy, so the replies have their own pipe). The image is a float32 RGBA file the
server reads and deletes. Not importable outside Blender (``bpy``); cycles_backend.py starts it.

Requests: ``{"op": "render", "matrix": [16] (Blender camera to world, row major), "width", "height", "hfov_deg",
"spp", "seed", "denoise", "device", "out"}``, ``{"op": "ping"}``, ``{"op": "quit"}``.
"""
import json
import os
import sys
import time
import traceback

import bpy
import numpy as np
from mathutils import Matrix


EXR = os.path.join("/dev/shm" if os.path.isdir("/dev/shm") else "/tmp", f"opticalnav_cycles_{os.getpid()}.exr")


def setup(device: str) -> None:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = device
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == device
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "GPU"
    sc.cycles.use_adaptive_sampling = False  # every pass the spp asked for, as Mitsuba's
    sc.cycles.time_limit = 0.0
    sc.render.use_persistent_data = True  # the scene stays on the GPU between frames
    sc.render.use_motion_blur = False
    sc.render.use_compositing = False  # the Combined pass as rendered, not whatever the blend's node tree makes of it
    sc.render.use_sequencer = False
    sc.render.film_transparent = False
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_depth = "32"
    sc.view_settings.view_transform = "Standard"  # linear values only are read; the viewer tonemaps them
    cam = bpy.data.objects.new("OPTICALNAV_CAM", bpy.data.cameras.new("OPTICALNAV_CAM"))
    sc.collection.objects.link(cam)
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.sensor_width = 36.0
    cam.data.clip_start, cam.data.clip_end = 0.02, 1000.0
    sc.camera = cam


def render(req: dict) -> dict:
    sc = bpy.context.scene
    cam = sc.camera
    t0 = time.perf_counter()
    sc.render.resolution_x, sc.render.resolution_y = int(req["width"]), int(req["height"])
    cam.data.angle = float(req["hfov_deg"]) * np.pi / 180.0
    cam.matrix_world = Matrix(np.asarray(req["matrix"], dtype=np.float64).reshape(4, 4).tolist())  # a plain list is ignored
    sc.cycles.samples = int(req["spp"])
    sc.cycles.seed = int(req["seed"]) & 0x7FFFFFFF
    sc.cycles.use_animated_seed = False
    for name, value in (req.get("options") or {}).items():  # Cycles settings of this frame (sc.cycles.<name>), e.g. filter_glossy
        setattr(sc.cycles, name, value)
    sc.cycles.use_denoising = bool(req.get("denoise"))
    if sc.cycles.use_denoising:
        sc.cycles.denoiser = "OPTIX"
    # background Blender keeps no Render Result in memory: the frame goes through a float EXR (scene linear), read back
    sc.render.filepath = EXR
    sc.render.use_file_extension = False
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(EXR, check_existing=False)
    try:
        w, h = image.size
        pixels = np.empty(w * h * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
    finally:
        bpy.data.images.remove(image)
        os.unlink(EXR)
    pixels.reshape(h, w, 4)[::-1].tofile(req["out"])  # Blender's rows run bottom to top
    return {"ok": True, "width": w, "height": h, "render_s": round(time.perf_counter() - t0, 4)}


def main() -> None:
    cmd_fd, reply_fd = (int(v) for v in sys.argv[sys.argv.index("--") + 1:][:2])
    cmd, reply = os.fdopen(cmd_fd, "r"), os.fdopen(reply_fd, "w")
    device = os.environ.get("OPTICALNAV_CYCLES_DEVICE", "OPTIX")
    setup(device)
    # the first GPU render compiles the kernels for this card (minutes, once; Blender caches them): done before "ready"
    render({"width": 32, "height": 24, "hfov_deg": 60.0, "spp": 1, "seed": 0,
            "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1.5, 0, 0, 0, 1], "out": os.devnull})
    reply.write(json.dumps({"ok": True, "ready": True, "device": device, "scene": bpy.data.filepath,
                            "objects": len(bpy.data.objects), "lights": sum(o.type == "LIGHT" for o in bpy.data.objects)}) + "\n")
    reply.flush()
    for line in cmd:
        req = json.loads(line)
        try:
            if req["op"] == "quit":
                break
            out = {"ok": True} if req["op"] == "ping" else render(req)
        except Exception as exc:  # noqa: BLE001 - reported to the server, which decides what to do
            out = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1500:]}
        reply.write(json.dumps(out) + "\n")
        reply.flush()


main()
