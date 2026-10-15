"""Indirect light of the rasterized preview (raster.py): a grid of light probes, baked when the scene or its lights
change, in place of a flat ambient term.

* Bake: every probe casts PROBE_RAYS rays (Mitsuba's ray_intersect, the scene as the path tracer holds it); a ray
  goes on through glass and invisible shapes and ends on a surface, which is lit as the preview lights it (its
  direct light, with soft shadows) plus, after the first bounce, the probes' light from the bounce before; a ray
  that leaves the scene brings the sky, unless through a window. A light's own glow and the sky through glass
  are not taken: the preview lights by them directly (its lights and windows), and they would count twice. The radiance each probe receives is kept as order-2
  spherical harmonics per lane (R, G, B, near-infrared), and the distances it sees as a small octahedral map.
* Lookup: a point's irradiance is the eight probes around it, blended trilinearly, each weighted down when it
  sits behind the surface or when something lies between them (the Chebyshev test of its distances, as DDGI
  does): no light leaking through a wall from the room next door.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

PROBE_SPACING = 0.5   # m between probes (wider when the scene would need more than PROBE_MAX)
PROBE_MAX = 32768
PROBE_RAYS = 256
PROBE_BOUNCES = 2     # indirect bounces baked
PROBE_OCT = 8         # side of a probe's octahedral distance map
PROBE_CHUNK = 512     # probes traced and lit at a time
PASS_THROUGH = 3      # glass or invisible shapes a ray goes through at most
INSIDE = 0.25         # a probe whose rays hit this share of back faces or more is inside a wall or an object
SH_BAND = (math.pi, 2 * math.pi / 3, math.pi / 4)  # cosine lobe per SH band: radiance to irradiance


def _sh(d: torch.Tensor) -> torch.Tensor:
    """Real spherical harmonics to order 2 of unit directions ``d`` (... x 3): ... x 9."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    return torch.stack([torch.full_like(x, 0.282095), 0.488603 * y, 0.488603 * z, 0.488603 * x,
                        1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1), 1.092548 * x * z,
                        0.546274 * (x * x - y * y)], dim=-1)


def _fibonacci(count: int, device) -> torch.Tensor:
    """``count`` directions spread evenly over the sphere: count x 3."""
    i = torch.arange(count, dtype=torch.float32, device=device) + 0.5
    z = 1 - 2 * i / count
    r = (1 - z * z).clamp_min(0).sqrt()
    a = i * 2.399963
    return torch.stack([r * torch.cos(a), r * torch.sin(a), z], dim=1)


def _oct(d: torch.Tensor) -> torch.Tensor:
    """Octahedral coordinates in [0, 1]^2 of unit directions ``d`` (... x 3)."""
    p = d[..., :2] / d.abs().sum(-1, keepdim=True).clamp_min(1e-12)
    flip = (1 - p.flip(-1).abs()) * torch.where(p >= 0, 1.0, -1.0)
    p = torch.where(d[..., 2:3] < 0, flip, p)
    return p * 0.5 + 0.5


class Probes:
    """The light probes of one RasterScene (see the module's doc); ``bake`` when its key changes, ``irradiance``
    per pixel."""

    def __init__(self, scene):
        self.scene, self.dev = scene, scene.dev
        mi, dr = scene.mi, scene.dr
        # the grid over the scene's geometry (its lights and glass included: rooms are bounded by their walls)
        p = scene.pos_h[::max(1, scene.pos_h.shape[0] // 1_000_000), :3]
        lo, hi = torch.quantile(p, 0.001, dim=0), torch.quantile(p, 0.999, dim=0)
        extent = (hi - lo).clamp_min(1e-3)
        step = max(PROBE_SPACING, float((extent.prod() / PROBE_MAX) ** (1 / 3)))
        dims = ((extent / step).floor().long() + 1).clamp_min(1)
        while int(dims.prod()) > PROBE_MAX:
            step *= 1.05
            dims = ((extent / step).floor().long() + 1).clamp_min(1)
        self.lo, self.step, self.dims = lo + 0.5 * (extent - (dims - 1).float() * step), step, dims
        g = torch.stack(torch.meshgrid(*(torch.arange(int(n), device=self.dev) for n in dims), indexing="ij"), -1)
        self.pos = self.lo + g.reshape(-1, 3).float() * step  # G x 3, x fastest last: index (ix * ny + iy) * nz + iz
        G = self.pos.shape[0]
        self.dirs = _fibonacci(PROBE_RAYS, self.dev)
        self.basis = _sh(self.dirs)  # N x 9
        # a probe's distance map: each texel the mean over the rays near its direction (weights N x OCT^2)
        t = (torch.arange(PROBE_OCT, device=self.dev, dtype=torch.float32) + 0.5) / PROBE_OCT * 2 - 1
        tv, tu = torch.meshgrid(t, t, indexing="ij")
        c = torch.stack([tu, tv, 1 - tu.abs() - tv.abs()], -1).reshape(-1, 3)
        neg = c[:, 2] < 0
        c[neg, :2] = (1 - c[neg].flip(-1)[:, 1:].abs()) * torch.where(c[neg, :2] >= 0, 1.0, -1.0)
        c = c / c.norm(dim=1, keepdim=True)
        w = (self.dirs @ c.T).clamp_min(0) ** 40
        self.texel_w = w / w.sum(0, keepdim=True).clamp_min(1e-12)
        self.max_dist = 1.5 * step * math.sqrt(3)
        self.sh = torch.zeros((G, 9, 4), device=self.dev)
        self.mean = torch.full((G, PROBE_OCT * PROBE_OCT), self.max_dist, device=self.dev)
        self.mean2 = self.mean * self.mean
        self.valid = torch.ones(G, device=self.dev)
        self.key = None
        self.bake_s = 0.0
        # Mitsuba's shape registry ids -> the preview's shape index (-1: a shape it does not draw)
        ids = [int(dr.reinterpret_array(mi.UInt32, mi.ShapePtr(s))[0]) for s in scene.shapes]
        table = torch.full((max(ids + [0]) + 1,), -1, dtype=torch.long, device=self.dev)
        table[torch.tensor(ids, device=self.dev)] = torch.arange(len(ids), device=self.dev)
        self.shape_of = table

    def bake(self, key, light_rows, sky, nir_range) -> None:
        """Trace and light every probe's rays PROBE_BOUNCES times over (see the module's doc)."""
        if key == self.key:
            return
        t0 = time.perf_counter()
        sc = self.scene
        G = self.pos.shape[0]
        for bounce in range(PROBE_BOUNCES):
            sh = torch.zeros_like(self.sh)
            for first in range(0, G, PROBE_CHUNK):
                pr = slice(first, min(G, first + PROBE_CHUNK))
                radiance, dist, back = self._trace(self.pos[pr], light_rows, sky, nir_range, bounce > 0)
                n = radiance.shape[0] // PROBE_RAYS
                rad = radiance.view(n, PROBE_RAYS, 4)
                sh[pr] = torch.einsum("gnl,nk->gkl", rad, self.basis) * (4 * math.pi / PROBE_RAYS)
                if bounce == 0:
                    d = dist.view(n, PROBE_RAYS).clamp_max(self.max_dist)
                    self.mean[pr] = d @ self.texel_w
                    self.mean2[pr] = (d * d) @ self.texel_w
                    self.valid[pr] = (back.view(n, PROBE_RAYS).float().mean(1) < INSIDE).float()
            self.sh = sh
        self.key = key
        torch.cuda.synchronize()
        self.bake_s = time.perf_counter() - t0
        print(f"[raster] indirect light: {G} probes {tuple(int(x) for x in self.dims)} every {self.step:.2f} m, "
              f"{PROBE_BOUNCES} bounces, {self.bake_s:.2f} s", flush=True)

    def _trace(self, origins, light_rows, sky, nir_range, indirect):
        """The radiance (P x 4) each of the probes' rays brings back, how far its first hit is (P) and whether
        that hit is a back face (P)."""
        from .raster import GLASS_T
        sc, mi, dr = self.scene, self.scene.mi, self.scene.dr
        n = origins.shape[0]
        o = origins[:, None].expand(n, PROBE_RAYS, 3).reshape(-1, 3).contiguous()
        d = self.dirs[None].expand(n, PROBE_RAYS, 3).reshape(-1, 3).contiguous()
        P = o.shape[0]
        through = torch.ones((P, 1), device=self.dev)
        first = None
        hit_idx = torch.full((P,), -1, dtype=torch.long, device=self.dev)
        hp, uv, ng, ns = (torch.zeros((P, k), device=self.dev) for k in (3, 2, 3, 3))
        todo = torch.arange(P, device=self.dev)
        start = o
        for step in range(PASS_THROUGH + 1):
            torch.cuda.synchronize()  # Mitsuba reads the rays on its own stream
            f = lambda t: mi.Float(t.contiguous())
            ray = mi.Ray3f(mi.Point3f(f(start[:, 0]), f(start[:, 1]), f(start[:, 2])),
                           mi.Vector3f(f(d[todo, 0]), f(d[todo, 1]), f(d[todo, 2])))
            si = sc.resident.scene.ray_intersect(ray)
            sid = dr.reinterpret_array(mi.UInt32, si.shape)
            dr.eval(si.t, sid, si.p, si.uv, si.n, si.sh_frame.n)
            dr.sync_thread()
            g = lambda a: torch.from_dlpack(a).clone()
            t = g(si.t)
            ok = torch.isfinite(t)
            shape = torch.where(ok, self.shape_of[g(sid).long().clamp(0, self.shape_of.shape[0] - 1)], -1)
            if first is None:
                first = torch.where(ok, t, math.inf)
                back = ok & ((torch.stack([g(si.n.x), g(si.n.y), g(si.n.z)], 1) * d).sum(1) > 0)
            p = torch.stack([g(si.p.x), g(si.p.y), g(si.p.z)], 1)
            mat = sc.shape_mat[shape.clamp_min(0)]
            onward = ok & ((shape < 0) | sc.m_glass[mat])  # glass, or a shape the preview does not draw
            stop = ok & ~onward
            idx = todo[stop]
            hit_idx[idx] = shape[stop]
            hp[idx] = p[stop]
            uv[idx] = torch.stack([g(si.uv.x), g(si.uv.y)], 1)[stop]
            ng[idx] = torch.stack([g(si.n.x), g(si.n.y), g(si.n.z)], 1)[stop]
            ns[idx] = torch.stack([g(si.sh_frame.n.x), g(si.sh_frame.n.y), g(si.sh_frame.n.z)], 1)[stop]
            glass = onward & (shape >= 0)
            through[todo[glass]] *= GLASS_T
            todo, start = todo[onward], p[onward] + d[todo[onward]] * 1e-3
            if not todo.numel():
                break
        radiance = torch.zeros((P, 4), device=self.dev)
        # the sky through glass is the windows' light, which the preview lights directly: not again here
        miss = (hit_idx < 0) & (through[:, 0] == 1.0)
        if bool(miss.any()):
            radiance[miss] = sky(d[miss])
        miss = hit_idx < 0
        hit = torch.nonzero(~miss)[:, 0]
        if hit.numel():
            radiance[hit] = self._shade(hit_idx[hit], hp[hit], uv[hit], ng[hit], ns[hit], d[hit], light_rows,
                                        nir_range, indirect)
        if todo.numel():  # still in glass after PASS_THROUGH: nothing
            radiance[todo] = 0.0
        return radiance * through, first, back

    def _shade(self, shape, x, uv, ng, ns, d, light_rows, nir_range, indirect):
        """Diffuse radiance (P x 4) leaving hit points toward their rays' origins."""
        sc = self.scene
        P = x.shape[0]
        flip = lambda v: torch.where((v * d).sum(1, keepdim=True) > 0, -v, v)  # facing the ray
        ng = flip(ng / ng.norm(dim=1, keepdim=True).clamp_min(1e-12))
        ln = ns.norm(dim=1, keepdim=True)
        n = torch.where(ln > 1e-6, ns / ln.clamp_min(1e-12), ng)
        n = torch.where((n * ng).sum(1, keepdim=True) < 0, -n, n)
        mat = sc.shape_mat[shape]
        duv = torch.full((P, 4), 0.02, device=self.dev)  # a coarse mip: a texture's colour on average
        kd = sc._surface(mat, uv, duv, nir_range)[0]
        irr = torch.zeros((P, 4), device=self.dev)
        if light_rows is not None:
            idx, pos, normal, mode, power, r2 = light_rows
            vis = sc._visibility(idx, x + ng * 0.03, ng, torch.zeros(P, device=self.dev))
            view = -d
            irr, _ = sc._lights(pos, normal, mode, power, r2, vis, x, n, ng, view, (n * view).sum(1).clamp_min(1e-4),
                                torch.ones((1, P), device=self.dev), torch.zeros((P, 4), device=self.dev))
        if indirect:
            irr = irr + self.irradiance(x, n, None, ng)[0]
        return kd / math.pi * irr  # not a light's own glow: the preview lights by it directly

    def _distances(self, probe: torch.Tensor, u: torch.Tensor) -> tuple:
        """The mean distance and mean squared distance (P each) probes ``probe`` see in directions ``u``
        (P x 3), bilinear in their octahedral maps."""
        c = _oct(u) * PROBE_OCT - 0.5
        c0 = c.floor()
        f = c - c0
        c0 = c0.long()
        mu = torch.zeros(u.shape[0], device=self.dev)
        mu2 = torch.zeros_like(mu)
        for dy in (0, 1):
            for dx in (0, 1):
                tx = (c0[:, 0] + dx).clamp(0, PROBE_OCT - 1)
                ty = (c0[:, 1] + dy).clamp(0, PROBE_OCT - 1)
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
                texel = ty * PROBE_OCT + tx
                mu = mu + w * self.mean[probe, texel]
                mu2 = mu2 + w * self.mean2[probe, texel]
        return mu, mu2

    def irradiance(self, x: torch.Tensor, n: torch.Tensor, r: torch.Tensor | None = None,
                   n_geo: torch.Tensor | None = None) -> tuple:
        """Irradiance (P x 4) at points ``x`` with normals ``n`` from the probes around them, and (if ``r``, P x 3
        unit directions are given) the same probes' irradiance about ``r``, for a glossy reflection. Which probes
        count is told by the surface's own normal ``n_geo`` (default ``n``), not a normal map's: else the blend
        would change from pixel to pixel with the map, in blotches. A CUDA kernel (raster_kernels), else torch."""
        n_geo = n if n_geo is None else n_geo
        from . import raster_kernels
        kernels = raster_kernels.kernels()
        if kernels is not None:
            grid = torch.cat([self.lo, torch.tensor([self.step], device=self.dev)]).float()
            empty = torch.zeros((0, 3), device=self.dev)
            out = kernels.probe_irradiance(x.contiguous(), n.contiguous(), n_geo.contiguous(),
                                           empty if r is None else r.contiguous(), grid, self.dims.contiguous(),
                                           self.sh.contiguous(), self.mean.contiguous(), self.mean2.contiguous(),
                                           self.valid.contiguous(), PROBE_OCT)
            return out[0], (out[1] if r is not None else None)
        nx, ny, nz = (int(v) for v in self.dims)
        q = x + n_geo * (0.25 * self.step)  # off the surface: a probe just behind it is not "in front"
        g = (q - self.lo) / self.step
        base = g.floor().long()
        frac = g - base.float()
        total = torch.zeros((x.shape[0], 9, 4), device=self.dev)
        wsum = torch.zeros((x.shape[0], 1), device=self.dev)
        for corner in range(8):
            o = torch.tensor([(corner >> 2) & 1, (corner >> 1) & 1, corner & 1], device=self.dev)
            c = base + o
            inside = ((c >= 0) & (c < self.dims)).all(1)
            c = torch.minimum(c.clamp_min(0), self.dims - 1)
            i = (c[:, 0] * ny + c[:, 1]) * nz + c[:, 2]
            tri = torch.where(o.bool(), frac, 1 - frac).prod(1)
            to = q - self.pos[i]
            dist = to.norm(dim=1).clamp_min(1e-6)
            u = to / dist[:, None]
            front = ((-u * n_geo).sum(1) + 1) * 0.5  # the probe in front of the surface counts most
            w = tri * (front * front + 0.2)
            mu, mu2 = self._distances(i, u)
            var = (mu2 - mu * mu).abs().clamp_min(1e-4)
            cheb = var / (var + (dist - mu).clamp_min(0) ** 2)
            w = w * torch.where(dist <= mu, 1.0, cheb ** 3).clamp_min(1e-3) * (inside.float() * self.valid[i])
            total = total + w[:, None, None] * self.sh[i]
            wsum = wsum + w[:, None]
        sh = total / wsum[:, :, None].clamp_min(1e-9)  # no probe at all: dark (a point inside the walls)
        band = torch.tensor([SH_BAND[0]] + [SH_BAND[1]] * 3 + [SH_BAND[2]] * 5, device=self.dev)
        eval_at = lambda dirs: torch.einsum("pk,pkl->pl", _sh(dirs) * band, sh).clamp_min(0)
        return eval_at(n), (eval_at(r) if r is not None else None)
