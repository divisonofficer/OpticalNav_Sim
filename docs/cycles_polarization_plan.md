# Polarized Cycles: Mitsuba's polarimetric BSDFs in Blender's Cycles

Goal: render a `.blend` with its own node graphs (procedural textures, groups, every shader the author used) **and** polarization
(Stokes S0..S3), so the simulator no longer needs `tools/blend_to_pack.py`'s translation to Mitsuba for polarized output. Speed is
not the aim: a frame in the Mitsuba path (about 2 s at 1024 spp) is the reference.

Base: Blender `v4.2.0` (the shipped binary's hash `a51f2935` is this tag's), built from source in `~/blender-polar/src`, with the
CUDA toolkit 12.8 (Blackwell sm_120) and OptiX headers. Mitsuba 3.9.1's polarized variants (`cuda_rgbn_polarized`) and our measured
Blender/Mitsuba conventions (README) are the reference for every milestone.

## What polarization needs from a path tracer

Mitsuba traces backwards from the camera. Each interaction is a 4x4 Mueller matrix per colour channel, the light is a Stokes vector
(unpolarized emission: (L, 0, 0, 0)), and the camera sees `M_n ... M_1 S_light`. Between interactions the matrices are rotated into the
local reference frame of the next ray (the Stokes basis: the plane of incidence, `rotate_stokes_basis`). Cycles keeps one RGB
`throughput` and one RGB `BsdfEval`; it has no frame to carry. So:

| Cycles today | Polarized |
|---|---|
| `IntegratorState.path.throughput` (`PackedSpectrum`) | + a Mueller product per channel (3 x 16 floats), + the ray's Stokes basis vector |
| `BsdfEval` (spectrum per closure, summed) | + Mueller matrix per closure evaluation, summed, in the right frame |
| `shade_surface` direct light: `integrate_surface_direct_light` writes `unshadowed_throughput` to the shadow state | the Mueller of the light sample, applied to the light's Stokes vector after the shadow test |
| film: `film_write_pass_spectrum` | S0..S3 passes (`Stokes.S0` ... `Stokes.S3`) next to Combined |
| closures: `bsdf_eval`, `bsdf_sample`, one scalar weight each | each closure type also returns its Mueller matrix |

Mapping of Blender shaders to Mitsuba's polarized BSDFs (the same as the converter: Principled values, not names):
diffuse lobe = depolarizing (`pplastic`'s diffuse part), GGX glossy on a dielectric = Fresnel Mueller of the microfacet normal
(`pplastic`'s specular part, `roughdielectric`), metals = complex Fresnel (`roughconductor`, eta / k from the base colour as in the
converter), glass = `dielectric` / `roughdielectric`. Closures with no polarimetric model (hair, SSS, volumes, sheen, toon)
depolarize and pass through.

## Milestones (each ends with a number against Mitsuba)

**M0. Baseline build.** Blender 4.2.0 from source, CUDA + OptiX, Blackwell; renders PH19 like the shipped binary (median radiance
within noise). Libraries: `lib/linux_x64` (git LFS), OptiX headers from `NVIDIA/optix-dev`.

**M1. Mueller library** (`kernel/polarization/`): Mueller of rotation, ideal polarizer, depolarizer; Fresnel for dielectric (real eta)
and conductor (complex eta, k) as Mueller matrices (`mueller::specular_reflection`, `specular_transmission` in Mitsuba);
the Baek et al. microfacet pBRDF. Ported from Mitsuba's `mitsuba/core/mueller.h` and `pplastic` / `roughconductor` sources (BSD-3;
Cycles is Apache-2.0: compatible, keep the notices). Host-side unit tests against Mitsuba's own functions (same inputs, same
matrices, 1e-5).

**M2. CPU integrator, diffuse + GGX + glass.** The Mueller product in the path state, rotation between segments, direct light with
Mueller, the film passes. First test: a closed room with one lamp (the synthetic room of the light calibration), then PH19's
corridor: S0 within 3% and DoLP / AoLP within 0.03 of Mitsuba at 1024 spp on the same poses.

**M3. Principled mapping.** Blend of the closures Principled BSDF makes (diffuse, specular, coat, transmission, metallic) into the
right Mueller sum; Mix Shader weights; Emission unpolarized; Alpha (transparent) as an identity. Check on PH33 (metals, glass,
brick, tile) against Mitsuba.

**M4. GPU (CUDA / OptiX).** Wavefront state grows by 60 floats a path (about 4 GB for 16 M paths in flight: lower the state
count or compress to half); shadow state gets the light's Mueller. Same numbers as M2 / M3 on the GPU.

**M5. Blender side.** `scene.cycles.use_polarization`, passes in the Python API, the render server's Cycles worker
(`opticalnav_sim/cycles_worker.py`) reads S0..S3 (multilayer EXR) and fills `s1..s3` of the reply, the viewer shows DoLP / AoLP for
`renderer: "cycles"`.

## Known hard parts
* MIS between light and BSDF sampling: the weights are scalars, the contributions are Mueller-weighted: keep the scalar pdfs as they are
  and apply Mueller only to the value, as Mitsuba does.
* The light tree and path guiding sample with scalar estimates: unchanged (they only choose a light).
* Russian roulette on the scalar `throughput`: keep it (the Mueller product is carried beside it).
* Memory: per path 48 floats of Mueller (RGB) plus the Stokes basis; half precision for the product is a fallback.
* Denoising: OptiX / OIDN take S0 only; S1..S3 are not denoised (the viewer denoises the running mean as before).
* Cross-checking needs Mitsuba and Cycles on the same scene: `tools/blend_to_pack.py` gives that, and the measured lighting and
  projection conventions (README) already agree to within 0.85 to 1.5 times in median radiance.

## Status
M0 in progress: source cloned (`~/blender-polar/src`, v4.2.0), libraries downloading.
