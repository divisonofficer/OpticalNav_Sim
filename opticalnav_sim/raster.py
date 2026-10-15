"""Rasterized preview of a resident scene: nvdiffrast's CUDA rasterizer on the meshes, textures and lights the
loaded Mitsuba scene holds, for moving about quickly (one image a frame, no noise).

Everything comes from the scene in GPU memory (DLPack), never from the scene file: hidden shapes (collapsed or
lifted by Resident._write_look), the pool's mirrors, the lamps' glowing parts and the light levels of the Look
are the ones the path tracer sees.

* Geometry: every mesh but the camera-assist shapes and null BSDFs, one vertex and one triangle buffer; glass
  (``dielectric``) is drawn as a second, see-through layer.
* Materials: per BSDF, up to two leaves (``blendbsdf`` and its weight; ``twosided``, ``normalmap`` and other
  wrappers are looked through; ``normalmap`` is applied). A leaf is plastic or diffuse (its base colour under a
  clear coat of its ior) or a conductor (its specular colour times the Fresnel reflectance at normal incidence
  of its eta, k, per lane), with its GGX alpha.
  Textures go into one buffer at their own size (at most TEX_MAX a side) with their mip chains, sampled
  trilinearly with repeat (sRGB decoded unless raw, as Mitsuba's ``bitmap`` reads them): no shimmer far away.
* Near-infrared: the fourth lane, as the RGBN variant has it: rgb2nir's pseudo near-infrared of a base colour
  mapped onto the Look's range, a conductor's eta, k lane, a light's luminance times its near-infrared level.
* Lights: each emitter as a point at its centroid carrying its power (radiance x area; one-sided when flat),
  with a cube shadow map (SHADOW a side) made the first time it is lit and kept until the geometry changes,
  filtered into soft shadows as wide as the light (PCSS; a CUDA kernel, raster_kernels);
  the camera flash as a one-sided light behind the camera, without shadows; the windows that see the sky as
  lights of the sky's mean radiance. Metal-rough shading: Lambert under GGX (Smith, Schlick) with each pixel's
  roughness (Mitsuba's ``alpha``, textured or not) and metallic weight (the ``blendbsdf`` of plastic and
  conductor). Indirect light comes from a grid of light probes baked with Mitsuba's ray tracing when the
  lights or the geometry change (raster_gi), diffuse and specular (Karis' analytic environment BRDF),
  darkened in corners and crevices by screen-space ambient occlusion; without them, a flat ambient term (a
  share of the frame's mean direct irradiance plus some of the sky).
* Polarization, one reflection deep, with Mitsuba's own Mueller matrices: a light's GGX reflection is the
  Fresnel reflection (mueller::specular_reflection, complex for a conductor) in the plane of the microfacet,
  of unpolarized light or of the flash's (polarized_area, linear in its canonical basis); a pplastic's diffuse
  light leaves through its coat's Fresnel transmission (p-polarized); the glossy indirect light is the
  Fresnel reflection about the normal, less on a rough surface; glass transmits and reflects by its Fresnel
  s and p. Each is turned into the camera's basis as the stokes integrator does (across the ray and the
  scene's reference axis). Indirect light, the sky and the lights' glow are unpolarized; no circular part.
* Output: Stokes images S0, S1, S2 in R, G, B and near-infrared lanes (S3 zero), which the server hands on
  as the path tracer's bands, and the depth along the view axis.
"""
from __future__ import annotations

import math
import time
from types import SimpleNamespace

import sys

import numpy as np
import torch
import torch.utils.cpp_extension
import nvdiffrast.torch as drt

from . import raster_gi, raster_kernels

# nvdiffrast 0.3.3 builds its plugin with torch.utils.cpp_extension.load, then imports it by name from sys.path,
# where torch 2.12's load no longer puts its build directory: "No module named 'nvdiffrast_plugin'"
_build = torch.utils.cpp_extension._get_build_directory("nvdiffrast_plugin", False)
if _build not in sys.path:
    sys.path.insert(0, _build)

TEX_MAX = 1024     # a texture's longest side in the preview's buffer (2048 structure textures are halved)
MIP_LEVELS = 11    # a texture's mip chain: TEX_MAX halved down to one texel
SHADOW = 512       # cube shadow map side
CHUNK = 4096       # triangles per culling chunk
PIXEL_CHUNK = 1 << 17  # pixels lit at a time (lights x pixels x 3 floats each)
AA_MODES = ("off", "edge", "ssaa2")
WINDOW_TRI = 0.05  # m^2: a glass triangle this large is part of a window pane (_windows)
WINDOW_NEAR = 0.1  # m: a window's shadow map starts this far out (its frame and wall cast none)
WINDOW_INSET = 0.3  # m: a window's light sits this far into the room (see _cubes)
GRAZE = 0.1        # a light whose cosine with the surface's own normal is below this fades out (see _lights)
NEAR, FAR = 0.05, 200.0
SHADOW_NEAR, SHADOW_FAR = 0.02, 100.0
NIR_WEIGHTS = (0.229, 0.587, 0.114)     # rgb2nir's pseudo near-infrared (robomituba's pseudo_nir_albedo)
LUMA = (0.2126, 0.7152, 0.0722)         # an RGBN variant's near-infrared lane of a plain colour
PLASTIC_F0 = 0.04
AMBIENT_DIRECT = 0.25  # without probes: share of the frame's mean direct irradiance added as ambient
AMBIENT_SKY = 0.03     # share of the sky's mean radiance added as ambient
GLASS_T = 0.9          # glass: what passes through (a window's light)
MIRROR_ALPHA = 0.05    # a conductor this smooth (GGX alpha) is a mirror: it reflects the scene (see _mirrors)
MAX_MIRRORS = 4        # mirror planes reflected in a frame, the largest on screen
MIN_MIRROR_PIXELS = 16
MIRROR_OFFSET = 0.005  # m: a mirror's reflected view starts this far in front of it
MIN_ALPHA = 0.04       # GGX alpha floor: a mirror's highlight from a point light would be a single pixel
SOFT_SEARCH, SOFT_TAPS = 8, 12  # soft shadows: taps looking for what casts, then taps filtering (see _visibility)
SOFT_MAX = 0.3          # m: the widest penumbra
AO_SAMPLES = 12         # ambient occlusion: points tried in the hemisphere over a pixel (see _occlusion)
AO_RADIUS = 0.5         # m: how far around a point counts as its surroundings
AO_BIAS = 0.03          # m: nearer than this in front of a sample does not occlude it
AO_POWER = 2.0          # the occlusion squared: the ambient term stands for all indirect light, darker in corners
PROBES_ON = True        # light probes (raster_gi) for indirect light; False: the flat ambient term
PROFILE = bool(__import__("os").environ.get("OPTICALNAV_RASTER_PROFILE"))
COLOR_SLOTS = ("diffuse_reflectance", "specular_reflectance", "reflectance")
KIND_PLASTIC, KIND_CONDUCTOR, KIND_NONE = 0, 1, 2

# cube faces: (axis, sign, s axis, t axis); the shadow maps are drawn and read with this one table
FACES = [(0, 1.0, (0, 0, -1), (0, -1, 0)), (0, -1.0, (0, 0, 1), (0, -1, 0)),
         (1, 1.0, (1, 0, 0), (0, 0, 1)), (1, -1.0, (1, 0, 0), (0, 0, -1)),
         (2, 1.0, (1, 0, 0), (0, -1, 0)), (2, -1.0, (-1, 0, 0), (0, -1, 0))]


def _srgb_lut(device) -> torch.Tensor:
    c = torch.arange(256, dtype=torch.float32, device=device) / 255.0
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _disk(count: int, device) -> torch.Tensor:
    """``count`` points spread evenly over the unit disk (a Vogel spiral): count x 2."""
    i = torch.arange(count, dtype=torch.float32, device=device)
    r, a = ((i + 0.5) / count).sqrt(), i * 2.399963
    return torch.stack([r * torch.cos(a), r * torch.sin(a)], dim=1)


def _hemisphere(count: int, device) -> torch.Tensor:
    """``count`` fixed points in the unit hemisphere about +z, more of them near the centre: count x 3."""
    rng = np.random.RandomState(7)
    d = rng.normal(size=(count, 3))
    d[:, 2] = np.abs(d[:, 2]) + 0.15  # none along the surface itself
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    scale = 0.1 + 0.9 * (np.arange(count) / count) ** 2
    return torch.tensor(d * scale[:, None] * rng.uniform(0.5, 1.0, (count, 1)), dtype=torch.float32, device=device)


def _fresnel(cos: torch.Tensor, eta: torch.Tensor, k: torch.Tensor) -> tuple:
    """Mitsuba's mueller::specular_reflection for light arriving outside at ``cos`` on a surface of complex index
    eta + i k (broadcast): its a, b and c cos(delta), the terms of linear light."""
    e = torch.complex(eta.float(), -k.float().abs())
    c = cos.clamp(0, 1).to(torch.complex64)
    ct = torch.sqrt(1 - (1 / e) ** 2 * (1 - c * c))
    ct = torch.where(ct.imag > 0, ct.conj(), ct)
    a_s = (c - e * ct) / (c + e * ct)
    a_p = (e * c - ct) / (e * c + ct)
    matched = (k == 0) & ((eta - 1).abs() < 1e-6)
    rs, rp = a_s.abs() ** 2, a_p.abs() ** 2
    cc = (rs * rp).sqrt()
    cos_d = torch.where(cc > 0, (a_p * a_s.conj()).real / cc.clamp_min(1e-12), 0.0)
    a, b, cd = 0.5 * (rs + rp), 0.5 * (rs - rp), cc * cos_d
    return (torch.where(matched, 0.0, a), torch.where(matched, 0.0, b), torch.where(matched, 0.0, cd))


def _fresnel_rs_rp(cos: torch.Tensor, eta: torch.Tensor) -> tuple:
    """The s and p reflectances (|a_s|^2, |a_p|^2) of a dielectric of ior ``eta`` at ``cos``."""
    a, b, _ = _fresnel(cos, eta, torch.zeros_like(eta))
    return a + b, a - b


def _leaves_fresnel(cos: torch.Tensor, spec: torch.Tensor) -> tuple:
    """The a, b and c cos(delta) (P x 4 each) of a pixel's two leaves' Fresnel reflections at ``cos`` (P),
    weighted by their shares and tints (``spec``: P x 26, see RasterScene._surface)."""
    fa = fb = fc = 0.0
    for j in range(2):
        a, b, c = _fresnel(cos[:, None], spec[:, 2 + 4 * j:6 + 4 * j], spec[:, 10 + 4 * j:14 + 4 * j])
        w = spec[:, j:j + 1] * spec[:, 18 + 4 * j:22 + 4 * j]
        fa, fb, fc = fa + w * a, fb + w * b, fc + w * c
    return fa, fb, fc


def _reference(view: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """The camera's Stokes basis for light reaching it along ``view`` (P x 3, toward the camera): across the
    camera ray and the reference axis ``up``, as the stokes integrator turns its result."""
    ref = torch.cross(-view, up[None].expand_as(view), dim=1)
    return ref / ref.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _turn(forward: torch.Tensor, frm: torch.Tensor, to: torch.Tensor) -> tuple:
    """sin and cos (P each) of twice the signed angle that turns Stokes basis ``frm`` into ``to`` about
    ``forward`` (mueller::rotate_stokes_basis: S1' = c S1 + s S2, S2' = -s S1 + c S2)."""
    frm = frm / frm.norm(dim=1, keepdim=True).clamp_min(1e-12)
    to = to / to.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cos = (frm * to).sum(1).clamp(-1, 1)
    sin = (forward * torch.cross(frm, to, dim=1)).sum(1)
    th = torch.atan2(sin, cos)
    return torch.sin(2 * th), torch.cos(2 * th)


def _frame(n: torch.Tensor, angle: torch.Tensor | None = None) -> tuple:
    """Two unit tangents (P x 3 each) across normals ``n``, turned by ``angle`` (P) about them."""
    a = torch.where((n[:, 0].abs() < 0.9)[:, None], torch.tensor([1.0, 0.0, 0.0], device=n.device),
                    torch.tensor([0.0, 1.0, 0.0], device=n.device))
    t = torch.cross(n, a, dim=1)
    t = t / t.norm(dim=1, keepdim=True).clamp_min(1e-12)
    b = torch.cross(n, t, dim=1)
    if angle is not None:
        c, s = torch.cos(angle)[:, None], torch.sin(angle)[:, None]
        t, b = c * t + s * b, c * b - s * t
    return t, b


class Leaf:
    """One BSDF of a material: its kind, base colour (atlas layer, constant), whether rgb2nir wraps the colour,
    reflectance at normal incidence per lane (a conductor's from eta, k; a plastic's from its ior), the complex
    index of its specular surface per lane (eta, k: a plastic's coat, k 0; a plain diffuse's 1, no coat) and
    GGX alpha (atlas layer, constant), as Mitsuba's ``alpha`` holds it."""

    def __init__(self, kind=KIND_PLASTIC, color=None, wrapped=True, f0=(PLASTIC_F0,) * 4, rough=0.5, rough_tex=None,
                 eta=(1.5,) * 4, k=(0.0,) * 4):
        self.kind, self.color, self.wrapped, self.f0, self.rough = kind, color, wrapped, f0, rough
        self.rough_tex, self.eta, self.k = rough_tex, eta, k


class RasterScene:
    """The resident scene, rasterized. ``render`` draws one view; the geometry follows the scene's hidden
    shapes (``sync``)."""

    def __init__(self, resident):
        import drjit as dr
        t0 = time.perf_counter()
        self.resident, self.mi, self.dr = resident, resident.mi, dr
        self.dev = torch.device("cuda")
        self.ctx = drt.RasterizeCudaContext(device=self.dev)
        self.lut = _srgb_lut(self.dev)
        dr.eval()
        dr.sync_thread()
        from .server import ASSIST_PREFIX
        self.textures: dict[int, int] = {}  # Dr.Jit variable index of a texture's data -> atlas layer
        self._layers: list[torch.Tensor] = []
        self._srgb: list[bool] = []
        self._uv: list[np.ndarray] = []
        materials, mat_of = [], {}
        shapes, self.meshes = [], []
        for s in resident.scene.shapes():
            sid = s.id()
            if sid.startswith(ASSIST_PREFIX) or not s.is_mesh():
                continue
            b = s.bsdf()
            if str(b).startswith("Null"):
                continue
            key = id(b)
            if key not in mat_of:
                mat_of[key] = len(materials)
                materials.append((b, self._material(b)))
            shapes.append((s, sid, mat_of[key]))
        self.shape_ids = [sid for _, sid, _ in shapes]
        self.shapes = [s for s, _, _ in shapes]
        self.shape_mat = torch.tensor([m for _, _, m in shapes], dtype=torch.long, device=self.dev)
        self._build_materials([m for _, m in materials])
        self._build_geometry()
        self._build_lights()
        self.hidden = resident.look.hidden
        self.face_s = torch.tensor([f[2] for f in FACES], dtype=torch.float32, device=self.dev)
        self.face_t = torch.tensor([f[3] for f in FACES], dtype=torch.float32, device=self.dev)
        g = (2 * (torch.arange(SHADOW, device=self.dev) + 0.5) / SHADOW - 1)
        rows, cols = torch.meshgrid(g, g, indexing="ij")
        axes = torch.eye(3, device=self.dev)
        self.cube_dirs = torch.stack([cols[..., None] * self.face_s[i] + rows[..., None] * self.face_t[i] + sign * axes[axis]
                                      for i, (axis, sign, _, _) in enumerate(FACES)])  # 6 x S x S x 3, unnormalized
        self.search_taps, self.soft_taps = _disk(SOFT_SEARCH, self.dev), _disk(SOFT_TAPS, self.dev)
        self.ao_kernel = _hemisphere(AO_SAMPLES, self.dev)
        self.probes, self.geometry_version = None, 0
        self.env_key, self.env = None, None
        self._dirs_key = None
        self.build_s = time.perf_counter() - t0
        print(f"[raster] {len(self.shapes)} meshes, {self.n_tris} triangles ({self.n_glass} glass), "
              f"{len(materials)} materials, {self.t_w.shape[0]} textures ({self.atlas_flat.numel() / 1e6:.0f} MB), {self.n_emit} lights, {self.l_pos.shape[0] - self.n_emit} window panes, "
              f"{self.build_s:.1f} s", flush=True)

    # --- materials ---
    def _texture(self, params, prefix: str, srgb: bool):
        """(atlas layer, None) for a bitmap, (None, rgbn values) for a constant, (None, None) if absent;
        and whether rgb2nir wraps it."""
        keys = params.keys()
        wrapped = f"{prefix}.nested.data" in keys or f"{prefix}.nested.value" in keys
        p = f"{prefix}.nested" if wrapped else prefix
        if f"{p}.data" in keys:
            data = params[f"{p}.data"]
            index = data.array.index
            if index not in self.textures:
                t = torch.from_dlpack(data.array).view(*data.shape)
                img = t.to(torch.float32)
                if img.dim() == 2:
                    img = img[..., None]
                img = img[..., :3]
                if img.shape[-1] == 1:
                    img = img.expand(-1, -1, 3)
                k = max(1, -(-max(img.shape[:2]) // TEX_MAX))  # a power of two for Mitsuba's 2048s
                if k > 1:
                    img = torch.nn.functional.interpolate(img.permute(2, 0, 1)[None], size=(img.shape[0] // k, img.shape[1] // k),
                                                          mode="area")[0].permute(1, 2, 0)
                if data.array.__class__.__name__.startswith("Float"):  # linear float data: kept linear, as 8 bits
                    img = (img.clamp(0, 1) * 255.0)
                    srgb_layer = False
                else:
                    srgb_layer = srgb
                self.textures[index] = len(self._layers)
                self._layers.append(img.round().clamp(0, 255).to(torch.uint8).contiguous())
                self._srgb.append(srgb_layer)
                m = np.eye(3, dtype=np.float32)
                with_uv = f"{p}.to_uv"
                if with_uv in keys:
                    m = np.array(params[with_uv].matrix, dtype=np.float32).reshape(3, 3)
                self._uv.append(m[:2])
            return self.textures[index], None, wrapped
        if f"{p}.value" in keys:
            v = np.array(params[f"{p}.value"], dtype=np.float32).ravel()
            rgb = np.resize(v, 3) if v.size < 3 else v[:3]
            return None, rgb, wrapped
        return None, None, wrapped

    def _leaf(self, params, prefix: str) -> Leaf:
        keys = list(params.keys())
        has = lambda name: any(k == prefix + name or k.startswith(prefix + name + ".") for k in keys)
        value = lambda name: (np.array(params[prefix + name + ".value"] if prefix + name + ".value" in keys
                                       else params[prefix + name], np.float32).ravel()
                              if prefix + name + ".value" in keys or prefix + name in keys else None)
        rough, rough_tex = 0.5, None
        if has("alpha"):
            layer, const, _ = self._texture(params, prefix + "alpha", False)
            rough_tex, rough = layer, float(const[0]) if const is not None else 0.5
        if has("diffuse_reflectance"):
            tex, const, wrapped = self._texture(params, prefix + "diffuse_reflectance", True)
            eta = value("eta")
            ior = float(eta[0]) if eta is not None else 1.5
            f0 = ((ior - 1) / (ior + 1)) ** 2
            return Leaf(KIND_PLASTIC, (tex, const), wrapped, (f0,) * 4, rough, rough_tex, eta=(ior,) * 4)
        if has("k") or has("specular_reflectance"):
            tex, const, wrapped = (self._texture(params, prefix + "specular_reflectance", True)
                                   if has("specular_reflectance") else (None, None, True))
            eta, k = value("eta"), value("k")
            eta = np.resize(eta, 4) if eta is not None else np.zeros(4, np.float32)
            k = np.resize(k, 4) if k is not None else np.ones(4, np.float32)
            f0 = ((eta - 1) ** 2 + k ** 2) / ((eta + 1) ** 2 + k ** 2)
            return Leaf(KIND_CONDUCTOR, (tex, const), wrapped, tuple(float(x) for x in f0),
                        rough if has("alpha") else 0.0, rough_tex,
                        eta=tuple(float(x) for x in eta), k=tuple(float(x) for x in k))
        if has("reflectance"):
            tex, const, wrapped = self._texture(params, prefix + "reflectance", True)
            return Leaf(KIND_PLASTIC, (tex, const), wrapped, f0=(0.0,) * 4, rough=1.0, eta=(1.0,) * 4)
        if has("eta"):
            return Leaf(KIND_NONE)  # dielectric: the glass layer
        return Leaf(KIND_PLASTIC, (None, np.full(3, 0.5, np.float32)), True)

    def _node(self, params, prefix: str, nmap=None):
        """(weight texture or constant, leaf 0, leaf 1, normal map layer or None) of the BSDF at ``prefix``."""
        keys = list(params.keys())
        starts = lambda p: any(k.startswith(prefix + p) for k in keys)
        if starts("normalmap.") and starts("nested_bsdf."):
            layer = self._texture(params, prefix + "normalmap", False)[0]
            return self._node(params, prefix + "nested_bsdf.", layer if nmap is None else nmap)
        if starts("bsdf_0.") and starts("weight"):
            w = self._texture(params, prefix + "weight", False)
            a = self._node(params, prefix + "bsdf_0.", nmap)
            b = self._node(params, prefix + "bsdf_1.", nmap)
            return (w[0], None if w[1] is None else float(w[1][0])), a[1], b[1], a[3] if a[3] is not None else b[3]
        for wrapper in ("brdf_0.", "nested_bsdf."):
            if starts(wrapper):
                return self._node(params, prefix + wrapper, nmap)
        leaf = self._leaf(params, prefix)
        return (None, 0.0), leaf, leaf, nmap

    def _material(self, bsdf):
        params = self.mi.traverse(bsdf)
        node = self._node(params, "")
        glass = all(leaf.kind == KIND_NONE for leaf in node[1:3])
        return node, glass

    def _build_materials(self, mats) -> None:
        M, dev = len(mats), self.dev
        self.m_glass = torch.tensor([g for _, g in mats], dtype=torch.bool, device=dev)
        wtex = np.full(M, -1, np.int64)
        wconst = np.zeros(M, np.float32)
        kind = np.zeros((M, 2), np.int64)
        ctex = np.full((M, 2), -1, np.int64)
        cconst = np.ones((M, 2, 3), np.float32)
        wrapped = np.ones((M, 2), bool)
        f0 = np.zeros((M, 2, 4), np.float32)
        rough = np.ones((M, 2), np.float32)
        rtex = np.full((M, 2), -1, np.int64)
        eta = np.ones((M, 2, 4), np.float32)
        kk = np.zeros((M, 2, 4), np.float32)
        ntex = np.full(M, -1, np.int64)
        for i, ((w, a, b, nmap), _) in enumerate(mats):
            ntex[i] = -1 if nmap is None else nmap
            wtex[i] = -1 if w[0] is None else w[0]
            wconst[i] = 0.0 if w[1] is None else w[1]
            for j, leaf in enumerate((a, b)):
                kind[i, j] = leaf.kind
                if leaf.color is not None:
                    tex, const = leaf.color
                    ctex[i, j] = -1 if tex is None else tex
                    if const is not None:
                        cconst[i, j] = const
                wrapped[i, j] = leaf.wrapped
                f0[i, j] = leaf.f0
                rough[i, j] = leaf.rough
                rtex[i, j] = -1 if leaf.rough_tex is None else leaf.rough_tex
                eta[i, j], kk[i, j] = leaf.eta, leaf.k
        t = lambda a, dt=None: torch.as_tensor(a, device=dev, dtype=dt)
        self.m_wtex, self.m_wconst, self.m_kind, self.m_ctex = t(wtex), t(wconst), t(kind), t(ctex)
        self.m_cconst, self.m_wrapped, self.m_f0, self.m_ntex = t(cconst), t(wrapped), t(f0), t(ntex)
        self.m_alpha, self.m_atex = t(rough), t(rtex)
        self.m_eta, self.m_k = t(eta), t(kk)
        # a mirror: wholly a smooth conductor (no textured roughness or blend)
        mirror = np.zeros(M, bool)
        for i in range(M):
            for j in range(2):
                share = (wconst[i] if j == 1 else 1 - wconst[i]) if wtex[i] < 0 else 0.0
                mirror[i] |= (kind[i, j] == KIND_CONDUCTOR and rough[i, j] < MIRROR_ALPHA and rtex[i, j] < 0
                              and share > 0.99)
        self.m_mirror = t(mirror)
        layers = self._layers or [torch.zeros((1, 1, 3), dtype=torch.uint8, device=dev)]
        srgb = self._srgb or [False]
        # every texture's mip levels, one after another: [T, MIP_LEVELS] offsets and sizes (a short chain repeats
        # its last level)
        flat, start = [], 0
        off = np.zeros((len(layers), MIP_LEVELS), np.int64)
        hs, ws = np.ones((len(layers), MIP_LEVELS), np.int64), np.ones((len(layers), MIP_LEVELS), np.int64)
        counts = []
        for i, (layer, sr) in enumerate(zip(layers, srgb)):
            chain = self._mips(layer, sr)
            counts.append(len(chain))
            for lv in range(MIP_LEVELS):
                img = chain[min(lv, len(chain) - 1)]
                off[i, lv], hs[i, lv], ws[i, lv] = start if lv < len(chain) else off[i, len(chain) - 1], img.shape[0], img.shape[1]
                if lv < len(chain):
                    flat.append(img.reshape(-1, 3))
                    start += img.shape[0] * img.shape[1]
        self.atlas_flat = torch.cat(flat)  # every texture's texels, one after another
        self.t_off, self.t_h, self.t_w = t(off), t(hs), t(ws)
        self.t_levels = t(np.array(counts, np.int64))
        self.t_srgb = t(np.array(srgb, bool))
        self.t_uv = t(np.array(self._uv or [np.eye(3, dtype=np.float32)[:2]], np.float32))
        del self._layers

    def _mips(self, layer: torch.Tensor, srgb: bool) -> list:
        """A texture (H x W x 3, uint8) and its halvings down to one texel, each the mean of the one above it in
        linear values (an sRGB texture decoded, averaged, encoded again)."""
        chain = [layer]
        img = self.lut[layer.long()] if srgb else layer.float() / 255.0
        while max(img.shape[:2]) > 1 and len(chain) < MIP_LEVELS:
            h, w = max(1, img.shape[0] // 2), max(1, img.shape[1] // 2)
            img = torch.nn.functional.interpolate(img.permute(2, 0, 1)[None], size=(h, w), mode="area")[0].permute(1, 2, 0)
            enc = torch.where(img <= 0.0031308, img * 12.92, 1.055 * img.clamp_min(1e-12) ** (1 / 2.4) - 0.055) if srgb else img
            chain.append((enc * 255.0).round().clamp(0, 255).to(torch.uint8).contiguous())
        return chain

    # --- geometry ---
    def _buffers(self):
        """Every kept mesh's positions (V x 3), as the scene holds them now."""
        return [torch.from_dlpack(s.vertex_positions_buffer()).view(-1, 3) for s in self.shapes]

    def _build_geometry(self) -> None:
        dev = self.dev
        pos = self._buffers()
        normals, uvs, tris, tri_shape = [], [], [], []
        first = 0
        for i, (s, p) in enumerate(zip(self.shapes, pos)):
            n = p.shape[0]
            normals.append(torch.from_dlpack(s.vertex_normals_buffer()).view(-1, 3) if s.has_vertex_normals()
                           else torch.zeros((n, 3), device=dev))
            uvs.append(torch.from_dlpack(s.vertex_texcoords_buffer()).view(-1, 2) if s.has_vertex_texcoords()
                       else torch.zeros((n, 2), device=dev))
            f = torch.from_dlpack(s.faces_buffer()).view(-1, 3).to(torch.int64)
            tris.append(f + first)
            tri_shape.append(torch.full((f.shape[0],), i, dtype=torch.int32, device=dev))
            first += n
        self.counts = [p.shape[0] for p in pos]
        self.n_verts = first
        self.pos_h = torch.cat([torch.cat(pos), torch.ones((first, 1), device=dev)], dim=1).contiguous()
        self.attr = torch.cat([torch.cat(normals), torch.cat(uvs)], dim=1).contiguous()  # normal, uv
        tris, tri_shape = torch.cat(tris).to(torch.int32), torch.cat(tri_shape)
        glass = self.m_glass[self.shape_mat[tri_shape.long()]]
        self.tri_o, self.shape_o = tris[~glass].contiguous(), tri_shape[~glass].contiguous()
        self.tri_g, self.shape_g = tris[glass].contiguous(), tri_shape[glass].contiguous()
        self.n_tris, self.n_glass = int(tris.shape[0]), int(self.tri_g.shape[0])
        if self.tri_o.shape[0] >= 1 << 24:
            raise RuntimeError("more triangles than a float32 triangle id holds exactly (2^24)")
        self._build_chunks()

    def _build_chunks(self) -> None:
        """The opaque triangles in Morton order of their centroids, cut into CHUNK-triangle chunks: nearby
        triangles share a chunk, and a frame rasterizes only the chunks whose boxes meet its view."""
        tri = self.tri_o.long()
        cen = (self.pos_h[tri[:, 0], :3] + self.pos_h[tri[:, 1], :3] + self.pos_h[tri[:, 2], :3]) / 3.0
        finite = torch.isfinite(cen).all(1)
        lo = torch.where(finite[:, None], cen, math.inf).amin(0)
        hi = torch.where(finite[:, None], cen, -math.inf).amax(0)
        q = ((cen - lo) / (hi - lo).clamp_min(1e-6) * 1023).nan_to_num(0).long().clamp(0, 1023)

        def spread(x):  # 10 bits -> every third of 30
            x = (x | (x << 16)) & 0x030000FF
            x = (x | (x << 8)) & 0x0300F00F
            x = (x | (x << 4)) & 0x030C30C3
            return (x | (x << 2)) & 0x09249249

        order = (spread(q[:, 0]) | (spread(q[:, 1]) << 1) | (spread(q[:, 2]) << 2)).argsort()
        n = order.shape[0]
        count = -(-n // CHUNK)
        pad = torch.cat([order, order[-1:].expand(count * CHUNK - n)])  # the last triangle again: drawn twice, same
        self.chunk_tri = self.tri_o[pad].view(count, CHUNK, 3).contiguous()
        self.chunk_shape = self.shape_o[pad].view(count, CHUNK).contiguous()
        self._chunk_boxes()

    def _chunk_boxes(self) -> None:
        lo, hi = [], []
        for part in self.chunk_tri.split(256):
            v = self.pos_h[part.long().view(-1), :3].view(part.shape[0], -1, 3)
            lo.append(v.amin(1))
            hi.append(v.amax(1))
        lo, hi = torch.cat(lo), torch.cat(hi)
        corners = [torch.stack([(hi if (k >> a) & 1 else lo)[:, a] for a in range(3)], dim=1) for k in range(8)]
        self.chunk_corners = torch.cat([torch.stack(corners, dim=1), torch.ones((lo.shape[0], 8, 1), device=self.dev)],
                                       dim=2)  # C x 8 x 4

    def _in_view(self, mvp: torch.Tensor) -> tuple:
        """The opaque triangles (and their shapes) of the chunks whose boxes the view's frustum meets."""
        c = self.chunk_corners @ mvp.T  # C x 8 x 4
        w = c[..., 3:4]
        outside = torch.cat([c[..., :3] < -w, c[..., :3] > w], dim=-1).all(dim=1).any(dim=1)  # all 8 beyond one plane
        keep = torch.nonzero(~outside)[:, 0]
        return self.chunk_tri[keep].view(-1, 3), self.chunk_shape[keep].view(-1)

    def _build_lights(self) -> None:
        """The scene's indoor emitters (shape index -> light index), their radiance and levels' keys."""
        r = self.resident
        self.light_shape, self.light_le, self.light_level = [], [], []
        shape_light = np.full(len(self.shapes), -1, np.int64)
        for i, (s, sid) in enumerate(zip(self.shapes, self.shape_ids)):
            if s.emitter() is None:
                continue
            # the emitter's own parameters: the scene's traverse lists a radiance texture shared by several
            # emitters (equal ones are merged on loading) under the first of them only
            ps = self.mi.traverse(s.emitter())
            key = next((k for k in ("radiance.nested.value", "radiance.value") if k in ps), None)
            if key is None:
                print(f"[raster] {sid}: an emitter without a plain radiance, left out", flush=True)
                continue
            le = np.resize(np.array(ps[key], dtype=np.float32).ravel(), 3)
            group = r.groups.get(sid, "fixture")
            shape_light[i] = len(self.light_shape)
            self.light_shape.append(i)
            self.light_le.append(le)
            self.light_level.append((f"light_rgb_{group}", f"light_nir_{group}_{r.classes.get(sid, 'led')}"))
        self.shape_light = torch.as_tensor(shape_light, device=self.dev)
        self.cubes, self.cube_made = None, set()  # cube shadow maps (_cubes)
        self._light_geometry()

    def _light_geometry(self) -> None:
        """Each light's centroid, area and mean normal from its triangles as the scene holds them now: the indoor
        emitters, then the windows (``_windows``)."""
        starts = np.concatenate([[0], np.cumsum(self.counts)])
        cs, areas, ns, flat = [], [], [], []
        for i in self.light_shape:
            s = self.shapes[i]
            p = self.pos_h[starts[i]:starts[i + 1], :3]
            f = torch.from_dlpack(s.faces_buffer()).view(-1, 3).long()
            a, b, c = p[f[:, 0]], p[f[:, 1]], p[f[:, 2]]
            cr = torch.cross(b - a, c - a, dim=1)
            area = 0.5 * cr.norm(dim=1)
            total = float(area.sum())
            cs.append(((a + b + c) / 3.0 * area[:, None]).sum(0) / max(total, 1e-12))
            mean_n = 0.5 * cr.sum(0) / max(total, 1e-12)
            areas.append(total)
            ns.append(mean_n / max(float(mean_n.norm()), 1e-12))
            flat.append(float(mean_n.norm()) > 0.95)
        windows = self._windows()
        self.n_emit = len(cs)
        for c, n, area in windows:
            cs.append(torch.as_tensor(c, device=self.dev, dtype=torch.float32))
            ns.append(torch.as_tensor(n, device=self.dev, dtype=torch.float32))
            areas.append(area)
            flat.append(True)
        dev = self.dev
        count = len(cs)
        self.l_pos = torch.stack(cs) if cs else torch.zeros((0, 3), device=dev)
        self.l_area = torch.tensor(areas, device=dev, dtype=torch.float32)
        self.l_n = torch.stack(ns) if ns else torch.zeros((0, 3), device=dev)
        self.l_flat = torch.tensor(flat, dtype=torch.bool, device=dev)
        self.l_two = torch.arange(count, device=dev) >= self.n_emit  # windows let the sky in either way
        # added to the squared distance from a light's point: a disk's on its axis, E = L A / (d^2 + A / pi), for a
        # panel or pane (a point's 1 / d^2 near a window was up to twice too bright), a sphere's size for a shade
        self.l_r2 = (self.l_area / torch.where(self.l_flat, math.pi, 4 * math.pi)).clamp_min(1e-4)
        # a light's radius, which sets its penumbras: a disk's for a panel or pane, a sphere's for a shade or bulb
        self.l_radius = (self.l_area / torch.where(self.l_flat, math.pi, 4 * math.pi)).sqrt().clamp_max(0.5)
        # nothing nearer than this casts a light's shadow (see _cubes); a window's own frame and wall neither
        near = torch.where(self.l_flat, SHADOW_NEAR, (0.6 * self.l_area.sqrt()).clamp(SHADOW_NEAR, 0.5))
        self.l_near = torch.where(self.l_two, WINDOW_NEAR, near).cpu()
        self.l_sky = torch.zeros(count)  # a window's share of directions that leave the scene (set with its cube)
        if self.cubes is not None and self.cubes.shape[0] != max(1, count):
            self.cubes = None
        self.cube_made = set()

    def _windows(self) -> list:
        """The glass panes the sky can come in by: large glass triangles (WINDOW_TRI m^2 or more) grouped into
        panes (parallel, in one plane, near each other; a pane's two faces are one). [(centre, normal, area)].
        Whether a pane looks outside is told later by its cube map (l_sky): a glass wall between rooms sees walls
        everywhere and lets no sky in."""
        if not self.n_glass:
            return []
        t = self.tri_g.long()
        pa, pb, pc = self.pos_h[t[:, 0], :3], self.pos_h[t[:, 1], :3], self.pos_h[t[:, 2], :3]
        cr = torch.cross(pb - pa, pc - pa, dim=1)
        area = 0.5 * cr.norm(dim=1)
        big = torch.nonzero(area >= WINDOW_TRI)[:, 0]
        if not big.numel():
            return []
        cen = ((pa + pb + pc) / 3.0)[big].cpu().numpy().astype(np.float64)
        nrm = (cr[big] / (2 * area[big, None])).cpu().numpy().astype(np.float64)
        ar = area[big].cpu().numpy().astype(np.float64)
        panes = []  # [sum area * centre, sum area, normal, (areas of the faces along +normal, along -normal)]
        for i in np.argsort(-ar):
            for pane in panes:
                c = pane[0] / pane[1]
                if abs(nrm[i] @ pane[2]) > 0.99 and abs(pane[2] @ (cen[i] - c)) < 0.06 and np.linalg.norm(cen[i] - c) < 1.5:
                    pane[0] = pane[0] + ar[i] * cen[i]
                    pane[1] += ar[i]
                    pane[3][0 if nrm[i] @ pane[2] > 0 else 1] += ar[i]
                    break
            else:
                panes.append([ar[i] * cen[i], ar[i], nrm[i], [ar[i], 0.0]])
        return [(p[0] / p[1], p[2], max(p[3])) for p in panes]

    def sync(self, hidden: frozenset) -> bool:
        """Follow the scene's hidden shapes: positions read again, shadows dropped. True if anything changed."""
        if hidden == self.hidden:
            return False
        self.dr.eval()
        self.dr.sync_thread()
        self.pos_h[:, :3] = torch.cat(self._buffers())
        self._chunk_boxes()
        self._light_geometry()
        self.hidden = hidden
        self.geometry_version += 1
        return True

    # --- shadows ---
    def _cubes(self, lights: list[int]) -> None:
        """Make the cube shadow maps (distance along each face's axis from a light to the nearest caster: not the
        lights themselves, not glass; inf where nothing is) of ``lights`` not made since the geometry changed."""
        if self.cubes is None:
            self.cubes = torch.full((max(1, self.l_pos.shape[0]), 6, SHADOW, SHADOW), math.inf, device=self.dev)
        if not hasattr(self, "tri_cast"):
            keep = self.shape_light[self.shape_o.long()] < 0
            self.tri_cast = self.tri_o[keep].contiguous()
        tris = self.tri_cast
        for light in lights:
            if light in self.cube_made:
                continue
            self._draw_cube(light, tris)
            if light >= self.n_emit:
                # a window: drawn at the pane first, which tells the share of directions leaving the scene (the
                # sky it lets in) and which side is outside (more of them there). Its light then moves
                # WINDOW_INSET into the room and shines that way only: from the pane itself it grazed the wall
                # the window is in, and that wall's shadow test striped it
                free = torch.isinf(self.cubes[light])
                along = (self.cube_dirs @ self.l_n[light]) > 0
                self.l_sky[light] = float(free.float().mean())
                inward = self.l_n[light] * (-1.0 if bool((free & along).sum() > (free & ~along).sum()) else 1.0)
                self.l_n[light] = inward
                self.l_pos[light] = self.l_pos[light] + WINDOW_INSET * inward
                self.l_two[light] = False
                self._draw_cube(light, tris)
            self.cube_made.add(light)

    def _draw_cube(self, light: int, tris: torch.Tensor) -> None:
        """Draw one light's cube shadow map (see _cubes)."""
        f = SHADOW_FAR
        # nothing nearer than the light's own size casts its shadow (the near plane): the light leaves the
        # whole glowing part, not the point standing in for it, so a lamp's holder and pole do not cut a cone
        # of shadow out of the floor (a lamp's other parts cannot be told by id: the loader merges meshes)
        n = float(self.l_near[light])
        c = self.l_pos[light]
        for i, (axis, sign, s_ax, t_ax) in enumerate(FACES):
            m = torch.zeros((4, 4), device=self.dev)
            m[0, :3], m[1, :3] = self.face_s[i], self.face_t[i]
            m[3, axis] = sign
            m[2, axis] = sign * (f + n) / (f - n)
            m[0, 3], m[1, 3] = -(m[0, :3] @ c), -(m[1, :3] @ c)
            m[3, 3] = -sign * c[axis]
            m[2, 3] = -sign * c[axis] * (f + n) / (f - n) - 2 * f * n / (f - n)
            clip = (self.pos_h @ m.T)[None].contiguous()
            rast, _ = drt.rasterize(self.ctx, clip, tris, (SHADOW, SHADOW))
            depth = 2 * f * n / ((f + n) - rast[0, :, :, 2] * (f - n))
            self.cubes[light, i] = torch.where(rast[0, :, :, 3] > 0, depth, math.inf)

    def _visibility(self, lights: torch.Tensor, x: torch.Tensor, n: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """How much of each of ``lights`` (A) points ``x`` (P x 3, normals ``n``) see, by their cube maps: A x P,
        0 to 1. Soft shadows as PCSS makes them: taps over the surface around a point (a disk turned by ``angle``,
        P) look for what casts its shadow; the penumbra is as wide as the light's radius times how much farther
        the point is than the caster, and as many taps across that width give the share lit. The shadow
        sharpens where the caster touches the receiver, as an area light's does. A CUDA kernel (raster_kernels);
        without one, hard shadows."""
        k = raster_kernels.kernels()
        if k is None:
            ma, _, limit = self._cube_test(lights, self.l_pos[lights][:, None], x[None], n)
            return (ma <= limit).float()
        return k.soft_shadows(self.cubes, self.l_pos.contiguous(), self.l_radius.contiguous(), lights.contiguous(),
                              x.contiguous(), n.contiguous(), angle.contiguous(), self.face_s, self.face_t,
                              self.search_taps, self.soft_taps, SOFT_MAX)

    def _cube_test(self, lights, light_pos, x, n) -> tuple:
        """Points ``x`` (A x P x 3 or 1 x P x 3) against ``lights``' cube maps: their distance along the face's
        axis, the caster's stored there and the most a lit point's can be (A x P each). The margin grows as the
        light grazes the surface (a texel then spans more depth): no stripes of shadow acne where a window's light
        runs along its wall."""
        d = x - light_pos
        ad = d.abs()
        axis = ad.argmax(dim=-1, keepdim=True)
        ma = ad.gather(-1, axis)[..., 0]
        face = axis[..., 0] * 2 + (d.gather(-1, axis)[..., 0] <= 0).long()  # FACES' order: +x, -x, +y, -y, +z, -z
        inv = 1.0 / ma.clamp_min(1e-9)
        col = (((d * self.face_s[face]).sum(-1) * inv + 1) * (0.5 * SHADOW)).long().clamp(0, SHADOW - 1)
        row = (((d * self.face_t[face]).sum(-1) * inv + 1) * (0.5 * SHADOW)).long().clamp(0, SHADOW - 1)
        stored = self.cubes.view(-1)[((lights[:, None] * 6 + face) * SHADOW + row) * SHADOW + col]
        cos = ((d * n[None]).sum(-1).abs() * d.norm(dim=-1).clamp_min(1e-9).reciprocal()).clamp(0.05, 1.0)
        slope = ((1 - cos * cos).sqrt() / cos).clamp_max(10.0)  # tan of the angle from the normal
        return ma, stored, stored * 1.02 + 0.04 + ma * slope * (2.0 / SHADOW)

    # --- shading ---
    def _sample(self, layer: torch.Tensor, uv: torch.Tensor, duv: torch.Tensor) -> torch.Tensor:
        """Trilinear, repeating lookups (P x 3, linear) of textures ``layer`` (P, >= 0) at ``uv`` (P x 2): the
        mip level from how far the uv moves a pixel (``duv``, P x 4: d/dx, then d/dy), bilinear in the two
        levels about it and blended, as a GPU's sampler does without anisotropy. A CUDA kernel (raster_kernels),
        else torch."""
        k = raster_kernels.kernels()
        if k is not None:
            return k.texture_sample(layer.contiguous(), uv.contiguous(), duv.contiguous(), self.t_uv, self.t_w,
                                    self.t_h, self.t_off, self.t_levels, self.t_srgb, self.atlas_flat, self.lut)
        m = self.t_uv[layer]  # P x 2 x 3
        u = m[:, 0, 0] * uv[:, 0] + m[:, 0, 1] * uv[:, 1] + m[:, 0, 2]
        v = m[:, 1, 0] * uv[:, 0] + m[:, 1, 1] * uv[:, 1] + m[:, 1, 2]
        w0, h0 = self.t_w[layer, 0].float(), self.t_h[layer, 0].float()
        span = None
        for k in (0, 2):  # texels a pixel moves across x, then y: the longer sets the level
            du = (m[:, 0, 0] * duv[:, k] + m[:, 0, 1] * duv[:, k + 1]) * w0
            dv = (m[:, 1, 0] * duv[:, k] + m[:, 1, 1] * duv[:, k + 1]) * h0
            d = du * du + dv * dv
            span = d if span is None else torch.maximum(span, d)
        lod = (0.5 * torch.log2(span.clamp_min(1e-12))).clamp(0.0, None)
        top = (self.t_levels[layer] - 1).float()
        lod = torch.minimum(lod, top)
        l0 = lod.floor()
        f = (lod - l0)[:, None]
        l0 = l0.long()
        l1 = torch.minimum(l0 + 1, top.long())
        return self._bilinear(layer, l0, u, v) * (1 - f) + self._bilinear(layer, l1, u, v) * f

    def _bilinear(self, layer: torch.Tensor, level: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Bilinear, repeating lookups (P x 3, linear) of mip ``level`` of textures ``layer`` at ``u, v``."""
        tw, th = self.t_w[layer, level], self.t_h[layer, level]
        x, y = u * tw - 0.5, v * th - 0.5
        x0, y0 = torch.floor(x), torch.floor(y)
        fx, fy = (x - x0)[:, None], (y - y0)[:, None]
        x0, y0 = torch.remainder(x0.long(), tw), torch.remainder(y0.long(), th)
        x1, y1 = torch.remainder(x0 + 1, tw), torch.remainder(y0 + 1, th)
        base = self.t_off[layer, level]
        srgb = self.t_srgb[layer][:, None]

        def tap(yy, xx):
            c = self.atlas_flat[base + yy * tw + xx].long()
            return torch.where(srgb, self.lut[c], c.float() / 255.0)

        return (tap(y0, x0) * (1 - fx) + tap(y0, x1) * fx) * (1 - fy) + (tap(y1, x0) * (1 - fx) + tap(y1, x1) * fx) * fy

    def _lookup(self, tex: torch.Tensor, uv: torch.Tensor, duv: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        """Atlas layer ``tex`` (P; -1: none) at ``uv``, else ``fallback`` (P x 3). Every pixel is sampled (layer 0
        for none) and the result selected: no host sync, as a masked lookup would need."""
        return torch.where((tex >= 0)[:, None], self._sample(tex.clamp_min(0), uv, duv), fallback)

    def _nir(self, rgb: torch.Tensor, wrapped: torch.Tensor, nir_range) -> torch.Tensor:
        c = rgb.clamp(0, 1)
        w = torch.tensor(NIR_WEIGHTS, device=self.dev)
        pseudo = (torch.maximum(c, 1 - c) * w).sum(1)
        lo, hi = float(nir_range[0]), float(nir_range[1])
        mapped = lo + (hi - lo) * (pseudo - 0.5) * 2.0
        return torch.where(wrapped, mapped, rgb @ torch.tensor(LUMA, device=self.dev))

    def _surface(self, mat: torch.Tensor, uv: torch.Tensor, duv: torch.Tensor, nir_range):
        """Per pixel, the metal-rough parameters of its material's (up to two, blended) BSDFs: diffuse albedo and
        reflectance at normal incidence (P x 4 lanes) and GGX alpha (P). A plastic's base colour is diffuse under
        a clear coat of its ior; a conductor's tints its reflectance and has no diffuse part, so a ``blendbsdf``
        of the two (Infinigen's metallic, as robomituba converts it) is the metallic weight.

        And for polarization: each leaf's specular surface (P x 26: the two leaves' shares, then their eta, k and
        tint per lane: a conductor's colour, 1 for a plastic's coat, 0 for glass) and the plastic coat's ior (P)
        the diffuse light leaves through."""
        P = mat.shape[0]
        w = self._lookup(self.m_wtex[mat], uv, duv, self.m_wconst[mat][:, None].expand(-1, 3))[:, 0]
        kd = torch.zeros((P, 4), device=self.dev)
        f0 = torch.zeros((P, 4), device=self.dev)
        alpha = torch.zeros(P, device=self.dev)
        shares, etas, ks, tints = [], [], [], []
        coat, coat_w = torch.zeros(P, device=self.dev), torch.zeros(P, device=self.dev)
        for j in range(2):
            share = (w if j == 1 else 1 - w)[:, None]
            kind = self.m_kind[mat, j]
            color = self._lookup(self.m_ctex[mat, j], uv, duv, self.m_cconst[mat, j])
            lanes = torch.cat([color, self._nir(color, self.m_wrapped[mat, j], nir_range)[:, None]], dim=1)
            leaf_f0 = self.m_f0[mat, j]
            plastic = (kind == KIND_PLASTIC)[:, None]
            conductor = (kind == KIND_CONDUCTOR)[:, None]
            kd = kd + share * torch.where(plastic, lanes * (1 - leaf_f0), 0.0)
            f0 = f0 + share * torch.where(plastic, leaf_f0, torch.where(conductor, lanes * leaf_f0, 0.0))
            a = self._lookup(self.m_atex[mat, j], uv, duv, self.m_alpha[mat, j][:, None].expand(-1, 3))[:, 0]
            alpha = alpha + share[:, 0] * a
            shares.append(share * (kind != KIND_NONE)[:, None])
            etas.append(self.m_eta[mat, j])
            ks.append(self.m_k[mat, j])
            tints.append(torch.where(conductor, lanes, 1.0))
            coat = coat + share[:, 0] * plastic[:, 0] * self.m_eta[mat, j, 0]
            coat_w = coat_w + share[:, 0] * plastic[:, 0]
        spec = torch.cat(shares + etas + ks + tints, dim=1)
        coat = torch.where(coat_w > 0, coat / coat_w.clamp_min(1e-6), 1.0)
        return kd, f0, alpha.clamp(MIN_ALPHA, 1.0), spec, coat

    def _normal_map(self, n, mat, uv, duv, corners, pa, pb, pc) -> torch.Tensor:
        """Shading normals (P x 3) with each material's normal map applied as Mitsuba's ``normalmap`` does: the
        texture's 2c - 1 in the frame of the shading normal, dp/du (made orthogonal to it) and their cross."""
        nt = self.m_ntex[mat]
        ua, ub, uc = (self.attr[corners[:, k], 3:5] for k in range(3))
        e1, e2 = ub - ua, uc - ua
        det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        ok = (nt >= 0) & (det.abs() > 1e-12)
        dpdu = ((pb - pa) * e2[:, 1:2] - (pc - pa) * e1[:, 1:2]) / torch.where(ok, det, 1.0)[:, None]
        s = dpdu - n * (n * dpdu).sum(1, keepdim=True)
        ls = s.norm(dim=1, keepdim=True)
        ok = ok & (ls[:, 0] > 1e-12)
        s = s / ls.clamp_min(1e-12)
        t = torch.cross(n, s, dim=1)
        c = self._sample(nt.clamp_min(0), uv, duv) * 2.0 - 1.0
        m = c[:, 0:1] * s + c[:, 1:2] * t + c[:, 2:3] * n
        m = m / m.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return torch.where(ok[:, None], m, n)

    def _environment(self, look):
        r = self.resident
        if r.env_original is None:
            return None
        if self.env_key != look.env:
            img = r.env_original if look.env == "original" else r.envs(look.env)
            self.env = torch.as_tensor(np.ascontiguousarray(img, dtype=np.float32), device=self.dev)
            self.env_mean = self.env.mean(dim=(0, 1))
            self.env_key = look.env
        return self.env

    @staticmethod
    def _lights(pos, normal, mode, power, r2, vis, x, n, n_geo, view, nv, a2, f0) -> tuple:
        """Irradiance (P x 4) and GGX radiance (P x 4) at points ``x`` with normals ``n`` seen along ``view``
        from lights (rows: position, normal, how they emit, power, squared size; visibility A x P)."""
        to = pos[:, None] - x[None]  # A x P x 3
        dd = (to * to).sum(-1)
        w = to * dd.clamp_min(1e-8).rsqrt()[..., None]
        cos_s = (w * n[None]).sum(-1).clamp_min(0)
        facing = -(w * normal[:, None]).sum(-1)
        cos_l = torch.where(mode[:, None] == 1, facing.abs(), torch.where(mode[:, None] == 0, facing.clamp_min(0), 0.25))
        # a light the surface itself (not its normal map) barely faces fades out: there the shadow test is least
        # sure (stripes of acne), and the light hardly counts
        fade = ((w * n_geo[None]).sum(-1) * (1.0 / GRAZE)).clamp(0, 1)
        g = cos_l * cos_s * fade / (dd + r2[:, None]) * vis  # A x P: irradiance per unit power
        h = w + view[None]
        h = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        nh = (h * n[None]).sum(-1).clamp_min(0)
        dist = a2 / (math.pi * (nh * nh * (a2 - 1) + 1) ** 2)  # GGX
        vis_g = 0.5 / (cos_s * (nv * nv * (1 - a2) + a2).sqrt()
                       + nv[None] * (cos_s * cos_s * (1 - a2) + a2).sqrt()).clamp_min(1e-6)  # Smith, height-correlated
        lobe = g * dist * vis_g
        schlick = (1 - (h * view[None]).sum(-1).clamp(0, 1)) ** 5
        return g.T @ power, f0 * (lobe.T @ power) + (1 - f0) * ((lobe * schlick).T @ power)

    def _sky_mean(self, look) -> torch.Tensor:
        """The sky's mean radiance (4 lanes)."""
        if self._environment(look) is None:
            return torch.zeros(4, device=self.dev)
        nir = (self.env_mean @ torch.tensor(LUMA, device=self.dev))[None] * float(dict(look.nir).get("daylight", 0.0))
        return torch.cat([self.env_mean, nir]) * float(look.env_scale)

    def _sky(self, d: torch.Tensor, look) -> torch.Tensor:
        """Sky radiance (P x 4) in directions ``d`` (P x 3), as Mitsuba's envmap maps them."""
        env = self._environment(look)
        if env is None:
            return torch.zeros((d.shape[0], 4), device=self.dev)
        a = math.radians(-float(look.env_rotate_deg))  # into the map's frame: the inverse rotation about y
        x = d[:, 0] * math.cos(a) + d[:, 2] * math.sin(a)
        z = -d[:, 0] * math.sin(a) + d[:, 2] * math.cos(a)
        u = torch.atan2(x, -z) / (2 * math.pi)
        u = u - torch.floor(u)
        v = torch.acos(d[:, 1].clamp(-1, 1)) / math.pi
        h, w = env.shape[:2]
        rgb = env[(v * h).long().clamp(0, h - 1), (u * w).long().clamp(0, w - 1)] * float(look.env_scale)
        nir = rgb @ torch.tensor(LUMA, device=self.dev) * float(dict(look.nir).get("daylight", 0.0))
        return torch.cat([rgb, nir[:, None]], dim=1)

    def _ray_dirs(self, c2w: torch.Tensor, width: int, height: int, hfov_deg: float) -> torch.Tensor:
        key = (width, height, round(float(hfov_deg), 6))
        if self._dirs_key != key:
            t = math.tan(math.radians(float(hfov_deg)) / 2.0)
            u = (2.0 * (torch.arange(width, device=self.dev) + 0.5) / width - 1.0) * t
            v = (1.0 - 2.0 * (torch.arange(height, device=self.dev) + 0.5) / height) * t * height / width
            vv, uu = torch.meshgrid(v, u, indexing="ij")
            self._dirs = torch.stack([uu, vv, -torch.ones_like(uu)], dim=-1).view(-1, 3)
            self._dirs_key = key
        d = self._dirs @ c2w[:3, :3].T
        return d / d.norm(dim=1, keepdim=True)

    def render(self, c2w: np.ndarray, width: int, height: int, hfov_deg: float, look, follow: bool,
               aa: str = "off") -> tuple:
        """One view: ([S0, S1, S2, S3], depth): Stokes images H x W x 4 (R, G, B, near-infrared lanes; None for
        zero) and the depth H x W along the view axis (0 where nothing is), numpy float32. ``aa``: "off" (one
        sample a pixel), "edge" (nvdiffrast's antialias: silhouettes blended by their coverage) or "ssaa2"
        (2 x 2 samples a pixel, averaged: also textures, highlights and shadow edges; four times the shading)."""
        if aa not in AA_MODES:
            raise ValueError(f"raster antialiasing is one of {AA_MODES}")
        try:
            k = 2 if aa == "ssaa2" else 1
            film, depth = self._render(c2w, int(width) * k, int(height) * k, hfov_deg, look, follow, edge=aa == "edge")
            if k > 1:
                film = film.view(int(height), k, int(width), k, 12).mean(dim=(1, 3))
                depth = depth[::k, ::k]  # one sample a pixel: depth is not averaged across an edge
            film = film.cpu().numpy()
            return [np.ascontiguousarray(film[..., 4 * i:4 * i + 4]) for i in range(3)] + [None], \
                depth.contiguous().cpu().numpy()
        except Exception:
            import traceback
            traceback.print_exc()
            raise

    def _render(self, c2w, width, height, hfov_deg, look, follow, edge=False):
        prof = PROFILE and []
        self._tick = (lambda name: (torch.cuda.synchronize(), prof.append((name, time.perf_counter())))) \
            if prof is not False else (lambda name: None)
        tick = self._tick
        tick("start")
        dev = self.dev
        width, height = int(width), int(height)
        m = torch.as_tensor(np.asarray(c2w, dtype=np.float64), device=dev, dtype=torch.float32)
        proj = self._projection(hfov_deg, width, height)
        surf = self._rasterize(m, proj, width, height)
        tick("rasterize")
        frame = self._frame_lights(look, follow, c2w, m)
        tick("lights of the frame")
        stokes, spec_pol, ambient = self._shade(surf, look, frame)
        P_all = width * height
        film = torch.zeros((P_all, 12), device=dev)  # S0, S1, S2: 4 lanes each
        depth = torch.zeros(P_all, device=dev)
        film[surf.hit] = stokes
        depth[surf.hit] = surf.zbuf[surf.hit]
        dirs = self._ray_dirs(m, width, height, hfov_deg)
        miss = torch.nonzero(surf.tid <= 0)[:, 0]
        if miss.numel():
            film[miss, :4] = self._sky(dirs[miss], look)
        self._mirrors(film, surf, spec_pol, look, frame, m, proj, dirs)
        tick("mirrors")
        sky_mean = self._sky_mean(look)
        if self.n_glass:
            rg, _ = drt.rasterize(self.ctx, surf.clip, self.tri_g, (height, width))
            gid = rg[0, :, :, 3].reshape(-1)
            gz = rg[0, :, :, 2].reshape(-1)
            front = (gid > 0) & ((surf.tid <= 0) | (gz < surf.rast[0, :, :, 2].reshape(-1)))
            g = torch.nonzero(front)[:, 0]
            if g.numel():
                film[g] = self._through_glass(film[g], -dirs[g], self._glass_normal(gid[g].long() - 1, rg, g, dirs[g]),
                                              frame.up, (ambient + math.pi * sky_mean)[None] / math.pi)
                w_inv = 2 * FAR * NEAR / ((FAR + NEAR) - gz[g] * (FAR - NEAR))
                depth[g] = w_inv
        tick("sky, glass")
        film = film.view(1, height, width, 12)
        if edge:
            film = drt.antialias(film.contiguous(), surf.rast, surf.clip, surf.tri_v)
            tick("antialias")
        result = film[0], depth.view(height, width)
        if prof:
            print("[raster-profile] " + ", ".join(f"{b[0]} {1000 * (b[1] - a[1]):.1f}" for a, b in zip(prof, prof[1:]))
                  + f" ms; {surf.hit.shape[0]} pixels hit", flush=True)
        return result

    def _projection(self, hfov_deg: float, width: int, height: int) -> torch.Tensor:
        t = math.tan(math.radians(float(hfov_deg)) / 2.0)
        aspect = width / height
        return torch.tensor([[1 / t, 0, 0, 0], [0, -aspect / t, 0, 0],  # row 0 at the top
                             [0, 0, -(FAR + NEAR) / (FAR - NEAR), -2 * FAR * NEAR / (FAR - NEAR)], [0, 0, -1, 0]],
                            device=self.dev, dtype=torch.float32)

    @staticmethod
    def _oblique(proj: torch.Tensor, plane: torch.Tensor) -> torch.Tensor:
        """``proj`` with its near plane moved onto ``plane`` (view space, 4: what lies on its positive side is
        kept), Lengyel's oblique frustum: a mirror's reflected view sees nothing behind the mirror."""
        inv = torch.linalg.inv(proj)
        clip = plane @ inv  # the plane in clip space: the frustum's far corner opposite it
        one = torch.ones_like(plane[0])
        q = inv @ torch.stack([torch.sign(clip[0]), torch.sign(clip[1]), one, one])
        c = plane * (2.0 / (plane @ q))
        out = proj.clone()
        out[2] = c - proj[3]
        return out

    def _rasterize(self, m: torch.Tensor, proj: torch.Tensor, width: int, height: int, plane=None):
        """The opaque surfaces seen by a camera (``m``: camera to world, a reflected one's too; ``plane``: a near
        plane in world space, 4, keeping its positive side): per hit pixel its point, normals, uv and their screen
        derivatives, material and shape, and the frame's buffers (rasterization, depth along the view axis)."""
        dev = self.dev
        w2c = torch.linalg.inv(m)
        if plane is not None:
            proj = self._oblique(proj, plane @ m)
        mvp = proj @ w2c
        clip = (self.pos_h @ mvp.T)[None].contiguous()
        tri_v, shape_v = self._in_view(mvp)
        rast, rast_db = drt.rasterize(self.ctx, clip, tri_v, (height, width))
        tid = rast[0, :, :, 3].reshape(-1)
        hit = torch.nonzero(tid > 0)[:, 0]
        tri = tid[hit].long() - 1
        bary = rast[0, :, :, :2].reshape(-1, 2)[hit]
        corners = tri_v[tri].long()  # P x 3 vertex indices
        b0, b1 = bary[:, 0:1], bary[:, 1:2]
        b2 = 1 - b0 - b1
        lerp = lambda a: a[corners[:, 0]] * b0 + a[corners[:, 1]] * b1 + a[corners[:, 2]] * b2
        x = lerp(self.pos_h[:, :3])
        attr = lerp(self.attr)
        n_sh, uv = attr[:, :3], attr[:, 3:5]
        # how far the uv moves a pixel across, then down (the barycentrics' screen derivatives): the mip level
        db = rast_db[0].reshape(-1, 4)[hit]  # du/dx, du/dy, dv/dx, dv/dy of the first two barycentrics
        uv0, uv1, uv2 = (self.attr[corners[:, k], 3:5] for k in range(3))
        e0, e1 = uv0 - uv2, uv1 - uv2
        duv = torch.cat([e0 * db[:, 0:1] + e1 * db[:, 2:3], e0 * db[:, 1:2] + e1 * db[:, 3:4]], dim=1)
        pa, pb, pc = (self.pos_h[corners[:, k], :3] for k in range(3))
        n_geo = torch.cross(pb - pa, pc - pa, dim=1)
        n_geo = n_geo / n_geo.norm(dim=1, keepdim=True).clamp_min(1e-12)
        eye = m[:3, 3]
        view = eye - x
        view = view / view.norm(dim=1, keepdim=True).clamp_min(1e-9)
        n_geo = torch.where((n_geo * view).sum(1, keepdim=True) < 0, -n_geo, n_geo)  # both sides face the camera
        ln = n_sh.norm(dim=1, keepdim=True)
        n = torch.where(ln > 1e-6, n_sh / ln.clamp_min(1e-12), n_geo)
        n = torch.where((n * n_geo).sum(1, keepdim=True) < 0, -n, n)
        shape = shape_v[tri].long()
        forward = -m[:3, 2]  # a reflected camera's too: it looks out of the mirror
        zbuf = torch.full((width * height,), math.inf, device=dev)
        zbuf[hit] = ((x - eye) * forward).sum(1)
        return SimpleNamespace(hit=hit, tid=tid, x=x, n=n, n_geo=n_geo, uv=uv, duv=duv, corners=corners, pa=pa, pb=pb,
                               pc=pc, shape=shape, mat=self.shape_mat[shape], view=view, eye=eye, forward=forward,
                               zbuf=zbuf, mvp=mvp, rast=rast, clip=clip, tri_v=tri_v, width=width, height=height)

    @staticmethod
    def _subset(surf, sel: torch.Tensor):
        """``surf`` (from _rasterize) for its hit pixels ``sel`` only."""
        per = ("hit", "x", "n", "n_geo", "uv", "duv", "corners", "pa", "pb", "pc", "shape", "mat", "view")
        return SimpleNamespace(**{k: (v[sel] if k in per else v) for k, v in vars(surf).items()})

    def _frame_lights(self, look, follow: bool, c2w, m: torch.Tensor):
        """The lights of a frame: each light's radiance per lane at the Look's levels, those lit (with shadow
        maps made), the light probes baked for them, the camera flash, and the rows the shading takes (position,
        normal, how they emit: 0 one-sided, 1 both ways, 2 every way, a lamp's shade; power per lane; the squared
        size added to the squared distance; polarized or not, and the frame of a polarized one)."""
        dev = self.dev
        levels = look.uniforms()
        L = self.l_pos.shape[0]
        lanes = np.zeros((max(L, 1), 4), np.float32)
        for li, (le, (rgb_u, nir_u)) in enumerate(zip(self.light_le, self.light_level)):
            lr, lnir = levels.get(rgb_u, 0.0), levels.get(nir_u, 0.0)
            lanes[li] = (le[0] * lr, le[1] * lr, le[2] * lr, float(np.dot(le, LUMA)) * lnir)
        if L > self.n_emit:  # windows: the sky's mean radiance through the glass, by the share of sky each sees
            self._cubes(list(range(self.n_emit, L)))
            sky = self._sky_mean(look).cpu().numpy() * GLASS_T
            for li in range(self.n_emit, L):
                lanes[li] = sky * min(1.0, 2.0 * float(self.l_sky[li]))
        lanes_t = torch.as_tensor(lanes, device=dev)
        active = [li for li in range(L) if lanes[li].any()]
        rows, idx = [], None
        if active:
            self._cubes(active)
            idx = torch.as_tensor(active, device=dev)
            mode = torch.where(self.l_two[idx], 1, torch.where(self.l_flat[idx], 0, 2))
            rows.append((self.l_pos[idx], self.l_n[idx], mode, lanes_t[idx] * self.l_area[idx][:, None],
                         self.l_r2[idx], torch.zeros(len(active), device=dev),
                         torch.zeros((len(active), 10), device=dev)))
        probes = self._indirect(look, lanes, (idx, *rows[0][:5]) if active else None)
        r = self.resident
        flash = None
        if follow and r.flash is not None and r.k_flash_radiance is not None and (look.flash_rgb or look.flash_nir):
            # the camera flash: one-sided, facing the view, no shadows (it sits behind the camera)
            le = np.resize(r.radiance[r.k_flash_radiance][1].astype(np.float32), 3)
            from .server import flash_matrix, level_offset
            size = [float(v) * float(look.flash_scale) for v in r.flash["size_world"]]
            fm = flash_matrix(np.asarray(c2w, np.float64), size, float(r.flash["distance_m"]))
            pos = fm[:3, 3] + (level_offset(c2w, look.flash_offset) if any(look.flash_offset) else 0.0)
            area = float(r.flash["size_world"][0]) * float(r.flash["size_world"][1])
            on_rgb, on_nir = float(bool(look.flash_rgb)), float(bool(look.flash_nir))
            power = torch.tensor([[le[0] * on_rgb, le[1] * on_rgb, le[2] * on_rgb, float(np.dot(le, LUMA)) * on_nir]],
                                 device=dev, dtype=torch.float32) * area
            f = lambda v: torch.as_tensor(np.asarray(v, np.float32)[None], device=dev)
            # a polarized_area flash: linear along Mitsuba's canonical basis in the rectangle's own frame (its
            # unnormalized dp/du, dp/dv as Mitsuba's rectangle keeps them), turned by the polarizer's angle
            polarized = str(r.flash.get("emitter") or "polarized_area") == "polarized_area"
            frame = np.concatenate([fm[:3, 0] * 2, fm[:3, 1] * 2, fm[:3, 2],
                                    [math.radians(float(r.flash.get("polarizer_angle_deg") or 0.0))]])
            rows.append((f(pos), f(fm[:3, 2]), torch.zeros(1, dtype=torch.long, device=dev), power,
                         torch.full((1,), 1e-4, device=dev), torch.full((1,), float(polarized), device=dev), f(frame)))
            # the rectangle itself, seen in a mirror (its radiance: the same power from a flash_scale larger side)
            flash = SimpleNamespace(pos=f(pos)[0], s=f(fm[:3, 0])[0], t=f(fm[:3, 1])[0], n=f(fm[:3, 2])[0],
                                    radiance=power[0] / (area * float(look.flash_scale) ** 2), polarized=polarized,
                                    frame=f(frame)[0])
        axis = (getattr(r, "depth", None) or {}).get("own", {}).get("reference_axis")
        up = torch.as_tensor(axis, device=dev, dtype=torch.float32) if axis is not None else m[:3, 1].contiguous()
        return SimpleNamespace(lanes_t=lanes_t, active=active, idx=idx, probes=probes, up=up, flash=flash,
                               rows=[torch.cat(c).contiguous() for c in zip(*rows)] if rows else None)

    def _shade(self, surf, look, frame):
        """Stokes S0, S1, S2 (P x 12, the camera's basis about ``surf.view``) leaving ``surf``'s points toward its
        camera; and the materials' polarization data (P x 26, see _surface) and the flat ambient term (4)."""
        dev, tick = self.dev, self._tick
        x, n_geo, view, mat, hit = surf.x, surf.n_geo, surf.view, surf.mat, surf.hit
        kd, f0, alpha, spec_pol, coat = self._surface(mat, surf.uv, surf.duv, look.nir_range)
        n = self._normal_map(surf.n, mat, surf.uv, surf.duv, surf.corners, surf.pa, surf.pb, surf.pc)
        tick("materials")
        P = hit.shape[0]
        # a turn per pixel, tiled over 4 x 4 pixels, for the taps of soft shadows and ambient occlusion
        bayer = torch.tensor([0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5], device=dev, dtype=torch.float32)
        angle = bayer[(hit // surf.width % 4) * 4 + hit % surf.width % 4] * (2 * math.pi / 16)
        ao = self._occlusion(x, n_geo, hit, angle, surf.zbuf, surf.mvp, surf.eye, surf.forward, surf.width, surf.height)
        tick("occlusion")
        lid = self.shape_light[surf.shape]
        emission = torch.where((lid >= 0)[:, None], frame.lanes_t[lid.clamp_min(0)], 0.0)
        lit = x + n_geo * 0.03  # off the surface, against shadow acne
        diffuse = torch.zeros((P, 4), device=dev)  # irradiance
        stokes = torch.zeros((P, 3, 4), device=dev)  # the lights' GGX radiance: S0, S1, S2 in the camera's basis
        kernels = raster_kernels.kernels()
        nv = (n * view).sum(1).clamp_min(1e-4)
        a2 = (alpha * alpha)[None]
        if frame.rows is not None:
            pos, normal, mode, power, r2, pol, frames = frame.rows
            for first in range(0, P, PIXEL_CHUNK):  # lights x pixels at a time: a 2x2 frame would hold gigabytes
                sl = slice(first, min(P, first + PIXEL_CHUNK))
                vis = torch.ones((pos.shape[0], sl.stop - sl.start), device=dev)
                if frame.active:  # the lights with cube maps come first (the flash, last, casts no shadow)
                    vis[:len(frame.active)] = self._visibility(frame.idx, lit[sl], n_geo[sl], angle[sl])
                if kernels is not None:  # polarized: Mitsuba's Mueller matrices of each reflection (raster_kernels)
                    o = kernels.shade_lights(x[sl].contiguous(), n[sl].contiguous(), n_geo[sl].contiguous(),
                                             view[sl].contiguous(), a2[0, sl].contiguous(), spec_pol[sl].contiguous(),
                                             vis.contiguous(), pos, normal, mode, power, r2, pol, frames, frame.up, GRAZE)
                    diffuse[sl], stokes[sl] = o[:, :4], o[:, 4:].view(-1, 3, 4)
                else:  # S0 alone, Schlick's Fresnel
                    diffuse[sl], stokes[sl, 0] = self._lights(pos, normal, mode, power, r2, vis, x[sl], n[sl], n_geo[sl],
                                                              view[sl], nv[sl], a2[:, sl], f0[sl])
        tick("lights")
        sky_mean = self._sky_mean(look)
        ambient = AMBIENT_DIRECT * diffuse.mean(0) + AMBIENT_SKY * math.pi * sky_mean  # glass's reflection; no probes
        occluded = (ao ** AO_POWER)[:, None]
        if frame.probes is not None:  # the probes' irradiance about the normal, and about the mirror direction
            mirror = 2 * (n * view).sum(1, keepdim=True) * n - view
            irr, irr_mirror = frame.probes.irradiance(x, n, mirror, n_geo)
            shade, gloss = irr * occluded, irr_mirror * occluded
        else:
            shade = gloss = ambient[None] * occluded
        tick("indirect")
        rough = alpha.sqrt()
        r0 = rough[:, None] * torch.tensor([-1.0, -0.0275, -0.572, 0.022], device=dev) + \
            torch.tensor([1.0, 0.0425, 1.04, -0.04], device=dev)
        a004 = torch.minimum(r0[:, 0] ** 2, torch.exp2(-9.28 * nv)) * r0[:, 0] + r0[:, 1]
        env_a, env_b = -1.04 * a004 + r0[:, 2], 1.04 * a004 + r0[:, 3]
        diffused = kd / math.pi * (diffuse + shade)
        glossy = (f0 * env_a[:, None] + env_b[:, None]) * gloss / math.pi
        radiance = diffused + stokes[:, 0] + glossy + emission
        # polarization of the unpolarized light reflected about the normal: the diffuse part leaves through the
        # plastic's coat (pplastic: Fresnel transmission out, p-polarized), the glossy one is its Fresnel
        # reflection (each leaf, per lane); both in the basis across the plane of the normal and the view
        rs, rp = _fresnel_rs_rp(nv, coat)
        out_diffuse = (rp - rs) / (2 - rs - rp).clamp_min(1e-6)
        fa, fb, _ = _leaves_fresnel(nv, spec_pol)
        # a rough surface's glossy light comes from a lobe of mirror directions, each in its own plane: their
        # polarization partly cancels (here as much as the GGX roughness, sqrt alpha)
        across = out_diffuse[:, None] * diffused + fb / fa.clamp_min(1e-6) * glossy * (1 - rough)[:, None]
        sn, cn = _turn(view, torch.cross(n, view, dim=1), _reference(view, frame.up))
        s1 = stokes[:, 1] + cn[:, None] * across
        s2 = stokes[:, 2] - sn[:, None] * across
        tick("polarization")
        return torch.cat([radiance, s1, s2], dim=1), spec_pol, ambient

    def _mirrors(self, film, surf, spec_pol, look, frame, m, proj, dirs) -> None:
        """Into ``film`` (H*W x 12), what the mirrors seen reflect: per mirror plane (the pixels of one mirror
        shape facing one way, the MAX_MIRRORS largest), the scene drawn and shaded from the camera reflected in
        it (nothing behind it: an oblique near plane), the sky where nothing is, then the mirror's Fresnel
        reflection (mueller::specular_reflection of its conductor, per lane) from the plane of incidence into
        the camera's basis. One bounce: a mirror seen in a mirror is shaded as a surface."""
        mirror = self.m_mirror[surf.mat]
        if not bool(mirror.any()):
            return
        sel = torch.nonzero(mirror)[:, 0]
        q = torch.round(surf.n_geo[sel] * 2).long() + 2
        key = surf.shape[sel] * 125 + q[:, 0] * 25 + q[:, 1] * 5 + q[:, 2]
        groups, inverse, counts = torch.unique(key, return_inverse=True, return_counts=True)
        order = torch.argsort(counts, descending=True)[:MAX_MIRRORS]
        for gi in order.tolist():
            if int(counts[gi]) < MIN_MIRROR_PIXELS:
                continue
            g = sel[inverse == gi]
            nm = surf.n_geo[g].mean(0)
            nm = nm / nm.norm().clamp_min(1e-12)
            d = (surf.x[g] @ nm).mean()
            R = torch.eye(3, device=self.dev) - 2 * torch.outer(nm, nm)
            mr = torch.eye(4, device=self.dev)
            mr[:3, :3] = R @ m[:3, :3]
            mr[:3, 3] = R @ m[:3, 3] + 2 * d * nm
            # what is reflected: the camera's side of the mirror only (n . p > d, a little off the glass)
            plane = torch.cat([nm, -(d + MIRROR_OFFSET)[None]])
            refl = self._rasterize(mr, proj, surf.width, surf.height, plane)
            pix = surf.hit[g]
            at = torch.full((surf.width * surf.height,), -1, dtype=torch.long, device=self.dev)
            at[refl.hit] = torch.arange(refl.hit.shape[0], device=self.dev)
            j = at[pix]
            seen = j >= 0
            k_in = -(dirs[pix] @ R.T)  # light comes along the reflected camera ray, reversed
            S = torch.zeros((pix.shape[0], 12), device=self.dev)
            if bool(seen.any()):
                S[seen] = self._shade(self._subset(refl, j[seen]), look, frame)[0]
            if bool((~seen).any()):
                S[~seen, :4] = self._sky(-k_in[~seen], look)
            if frame.flash is not None:  # the camera flash's rectangle, when the mirror shows it nearer than the rest
                far = torch.full((pix.shape[0],), math.inf, device=self.dev)
                far[seen] = (refl.x[j[seen]] - surf.x[g][seen]).norm(dim=1)
                lit = self._flash_seen(frame.flash, surf.x[g], -k_in, far, frame.up)
                if lit is not None:
                    at_flash, flash_stokes = lit
                    S[at_flash] = flash_stokes
            # the mirror's Fresnel reflection, from the reflected camera's basis into the real one's
            out = -dirs[pix]
            s_ax = torch.cross(nm.expand_as(k_in), k_in, dim=1)
            sn, cn = _turn(k_in, _reference(k_in, frame.up), s_ax)
            s1, s2 = S[:, 4:8], S[:, 8:]
            p1, p2 = cn[:, None] * s1 + sn[:, None] * s2, -sn[:, None] * s1 + cn[:, None] * s2
            fa, fb, fc = _leaves_fresnel((out @ nm).abs().clamp(0, 1), spec_pol[g])
            o0, o1, o2 = fa * S[:, :4] + fb * p1, fb * S[:, :4] + fa * p1, fc * p2
            sn, cn = _turn(out, s_ax, _reference(out, frame.up))
            film[pix] = torch.cat([o0, cn[:, None] * o1 + sn[:, None] * o2, -sn[:, None] * o1 + cn[:, None] * o2], dim=1)

    def _flash_seen(self, fl, origin: torch.Tensor, d: torch.Tensor, far: torch.Tensor, up: torch.Tensor):
        """Where rays (``origin``, ``d``: P x 3) meet the flash's rectangle from its lit side before ``far``:
        (mask P, Stokes S0, S1, S2 there, Q x 12, in the basis _reference(-d, up)), or None. Its light is linear
        along Mitsuba's canonical basis of its way out in the rectangle's frame (polarized_area)."""
        denom = d @ fl.n
        t = ((fl.pos - origin) @ fl.n) / torch.where(denom.abs() > 1e-9, denom, -1e-9)
        p = origin + t[:, None] * d - fl.pos
        u = (p @ fl.s) / (fl.s @ fl.s)
        v = (p @ fl.t) / (fl.t @ fl.t)
        mask = (denom < 0) & (t > 0) & (t < far) & (u.abs() <= 1) & (v.abs() <= 1)
        if not bool(mask.any()):
            return None
        k = -d[mask]  # the light's way: from the flash
        Q = k.shape[0]
        out = torch.zeros((Q, 12), device=self.dev)
        out[:, :4] = fl.radiance[None]
        if fl.polarized:
            fs, ft, fn, th = fl.frame[0:3], fl.frame[3:6], fl.frame[6:9], fl.frame[9]
            loc = torch.stack([k @ fs, k @ ft, k @ fn], dim=1)
            sign = torch.where(loc[:, 2] >= 0, 1.0, -1.0)
            ca = -1.0 / (sign + loc[:, 2])
            cb = loc[:, 0] * loc[:, 1] * ca
            sl = torch.stack([1 + sign * loc[:, 0] ** 2 * ca, sign * cb, -sign * loc[:, 0]], dim=1)  # coordinate_system
            sl = torch.cos(th) * sl + torch.sin(th) * torch.cross(loc, sl, dim=1)
            e = sl[:, 0:1] * fs + sl[:, 1:2] * ft + sl[:, 2:3] * fn
            e = e - (e * k).sum(1, keepdim=True) * k
            sn, cn = _turn(k, e, _reference(k, up))  # (1, 1, 0) along e, in the reference basis
            out[:, 4:8] = fl.radiance[None] * cn[:, None]
            out[:, 8:] = -fl.radiance[None] * sn[:, None]
        return mask, out

    def _indirect(self, look, lanes: np.ndarray, light_rows):
        """The light probes (raster_gi), baked again when the lights, the sky or the geometry changed; None
        without Mitsuba's ray tracing (the flat ambient term then)."""
        if self.probes is False or not PROBES_ON:
            return None
        if self.probes is None:
            try:
                self.probes = raster_gi.Probes(self)
            except Exception as exc:  # noqa: BLE001 - the preview goes on with its flat ambient term
                print(f"[raster] no light probes ({type(exc).__name__}: {exc}): flat ambient light", flush=True)
                self.probes = False
                return None
        key = (lanes.tobytes(), look.env, float(look.env_scale), float(look.env_rotate_deg),
               float(dict(look.nir).get("daylight", 0.0)), tuple(look.nir_range), self.geometry_version)
        self.probes.bake(key, light_rows, lambda d: self._sky(d, look), look.nir_range)
        return self.probes

    def _occlusion(self, x, n, hit, angle, depth, mvp, eye, forward, width, height) -> torch.Tensor:
        """Screen-space ambient occlusion (P, 1 open to 0 shut): of AO_SAMPLES points in the hemisphere over each
        hit point (normal ``n``, turned by ``angle``), the share that something nearer the camera hides, within
        AO_RADIUS. The tiled turn is then averaged out over 4 x 4 pixels, not across an edge in depth (``depth``,
        H*W along the view axis, inf where nothing is). A CUDA kernel (raster_kernels), else the same in torch."""
        kernels = raster_kernels.kernels()
        if kernels is not None:
            return kernels.occlusion(x.contiguous(), n.contiguous(), hit, angle.contiguous(), depth,
                                     mvp.contiguous(), eye.contiguous(), forward.contiguous(), self.ao_kernel,
                                     width, height, AO_RADIUS, AO_BIAS, NEAR)
        t, b = _frame(n, angle)
        k = self.ao_kernel
        z0 = depth[hit]
        hidden = torch.zeros(x.shape[0], device=self.dev)
        for kx, ky, kz in k.tolist():
            s = x + AO_RADIUS * (kx * t + ky * b + kz * n)
            c = s @ mvp[:, :3].T + mvp[:, 3]
            wc = c[:, 3].clamp_min(1e-6)
            col = ((c[:, 0] / wc + 1) * (0.5 * width)).long()
            row = ((c[:, 1] / wc + 1) * (0.5 * height)).long()
            inside = (c[:, 3] > NEAR) & (col >= 0) & (col < width) & (row >= 0) & (row < height)
            scene = depth[(row.clamp(0, height - 1) * width + col.clamp(0, width - 1))]
            sz = ((s - eye) * forward).sum(1)
            near = (AO_RADIUS / (z0 - scene).abs().clamp_min(1e-6)).clamp_max(1.0)  # far in front: another object
            hidden = hidden + torch.where(inside & (scene < sz - AO_BIAS), near, 0.0)
        ao = torch.ones(height * width, device=self.dev)
        ao[hit] = 1 - hidden / AO_SAMPLES
        ao, z = ao.view(1, 1, height, width), depth.view(1, 1, height, width)
        pad = lambda a: torch.nn.functional.pad(a, (2, 1, 2, 1), mode="replicate")
        ap, zp = pad(ao), pad(z)
        total, weight = torch.zeros_like(ao), torch.zeros_like(ao)
        for dy in range(4):
            for dx in range(4):
                zz = zp[..., dy:dy + height, dx:dx + width]
                w = ((zz - z).abs() < 0.05 * z.clamp_max(1e6)).float()
                total = total + w * ap[..., dy:dy + height, dx:dx + width]
                weight = weight + w
        return (total / weight.clamp_min(1.0)).view(-1)[hit]

    def _through_glass(self, behind: torch.Tensor, out: torch.Tensor, normal: torch.Tensor, up: torch.Tensor,
                       reflected: torch.Tensor) -> torch.Tensor:
        """Stokes S0, S1, S2 (P x 12, the camera's basis) seen through a pane of glass (ior 1.5) along ``out``
        (toward the camera; the pane's normal ``normal``): what lies behind through both faces' Fresnel
        transmission, plus the unpolarized ``reflected`` radiance (P x 4 or 1 x 4) by its Fresnel reflection,
        in the basis across the plane of incidence."""
        cos = (out * normal).sum(1).abs().clamp(0, 1)
        rs, rp = _fresnel_rs_rp(cos, torch.full_like(cos, 1.5))
        ts, tp = (1 - rs) ** 2, (1 - rp) ** 2  # two faces
        A, B, C = ((ts + tp) / 2)[:, None], ((ts - tp) / 2)[:, None], (ts * tp).sqrt()[:, None]
        ra, rb = ((rs + rp) / 2)[:, None], ((rs - rp) / 2)[:, None]
        ref = _reference(out, up)
        sn, cn = _turn(out, ref, torch.cross(normal, out, dim=1))
        sn, cn = sn[:, None], cn[:, None]
        s0, s1, s2 = behind[:, :4], behind[:, 4:8], behind[:, 8:]
        p1, p2 = cn * s1 + sn * s2, -sn * s1 + cn * s2  # into the pane's basis
        q0 = A * s0 + B * p1 + ra * reflected
        q1 = B * s0 + A * p1 + rb * reflected
        q2 = C * p2
        return torch.cat([q0, cn * q1 - sn * q2, sn * q1 + cn * q2], dim=1)  # and back

    def _glass_normal(self, tri: torch.Tensor, rg, pix, dirs) -> torch.Tensor:
        corners = self.tri_g[tri].long()
        pa, pb, pc = (self.pos_h[corners[:, k], :3] for k in range(3))
        n = torch.cross(pb - pa, pc - pa, dim=1)
        return n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)
