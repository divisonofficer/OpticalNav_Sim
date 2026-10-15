"""CUDA kernels of the rasterized preview (raster.py), built once with torch's cpp_extension and cached:
``soft_shadows`` is the PCSS filter, a thread a light and pixel; in torch, its 20 taps of a dozen tensor ops each
took 60 ms a frame, here under one. ``kernels()`` is None where they cannot be built (no compiler): the preview
then falls back to hard shadows."""
from __future__ import annotations

import torch
import torch.utils.cpp_extension

_CPP = """
torch::Tensor soft_shadows(torch::Tensor cubes, torch::Tensor light_pos, torch::Tensor radius, torch::Tensor lights,
                           torch::Tensor x, torch::Tensor n, torch::Tensor angle, torch::Tensor face_s,
                           torch::Tensor face_t, torch::Tensor search, torch::Tensor taps, double widest);
torch::Tensor occlusion(torch::Tensor x, torch::Tensor n, torch::Tensor hit, torch::Tensor angle, torch::Tensor depth,
                        torch::Tensor mvp, torch::Tensor eye, torch::Tensor forward, torch::Tensor kernel,
                        int64_t width, int64_t height, double radius, double bias, double near);
torch::Tensor probe_irradiance(torch::Tensor x, torch::Tensor n, torch::Tensor n_geo, torch::Tensor r,
                               torch::Tensor grid, torch::Tensor dims, torch::Tensor sh, torch::Tensor mean,
                               torch::Tensor mean2, torch::Tensor valid, int64_t side);
torch::Tensor texture_sample(torch::Tensor layer, torch::Tensor uv, torch::Tensor duv, torch::Tensor to_uv,
                             torch::Tensor tw, torch::Tensor th, torch::Tensor off, torch::Tensor levels,
                             torch::Tensor srgb, torch::Tensor atlas, torch::Tensor lut);
torch::Tensor shade_lights(torch::Tensor x, torch::Tensor n, torch::Tensor ng, torch::Tensor view, torch::Tensor a2,
                           torch::Tensor spec, torch::Tensor vis, torch::Tensor lpos, torch::Tensor lnorm,
                           torch::Tensor lmode, torch::Tensor lpower, torch::Tensor lr2, torch::Tensor lpol,
                           torch::Tensor lframe, torch::Tensor up, double graze);
"""

_CUDA = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

struct Test { float ma, stored, limit; };

// a point against a light's cube map, as RasterScene._cube_test does it
__device__ __forceinline__ Test cube_test(const float* __restrict__ cubes, int light, int side, float3 d, float3 n,
                                          const float* __restrict__ fs, const float* __restrict__ ft) {
    float ax = fabsf(d.x), ay = fabsf(d.y), az = fabsf(d.z);
    int axis = (ax >= ay && ax >= az) ? 0 : (ay >= az ? 1 : 2);
    float ma = axis == 0 ? ax : (axis == 1 ? ay : az);
    float along = axis == 0 ? d.x : (axis == 1 ? d.y : d.z);
    int face = axis * 2 + (along <= 0.f ? 1 : 0);
    float inv = 1.f / fmaxf(ma, 1e-9f);
    float s = (d.x * fs[face * 3] + d.y * fs[face * 3 + 1] + d.z * fs[face * 3 + 2]) * inv;
    float t = (d.x * ft[face * 3] + d.y * ft[face * 3 + 1] + d.z * ft[face * 3 + 2]) * inv;
    int col = min(max((int)((s + 1.f) * (0.5f * side)), 0), side - 1);
    int row = min(max((int)((t + 1.f) * (0.5f * side)), 0), side - 1);
    float stored = cubes[(((long)light * 6 + face) * side + row) * side + col];
    float len = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
    float c = fminf(fmaxf(fabsf(d.x * n.x + d.y * n.y + d.z * n.z) / fmaxf(len, 1e-9f), 0.05f), 1.f);
    float slope = fminf(sqrtf(1.f - c * c) / c, 10.f);
    return {ma, stored, stored * 1.02f + 0.04f + ma * slope * (2.f / side)};
}

__global__ void soft_kernel(const float* __restrict__ cubes, int side, const float* __restrict__ light_pos,
                            const float* __restrict__ radius, const long* __restrict__ lights, int A,
                            const float* __restrict__ x, const float* __restrict__ n, const float* __restrict__ angle,
                            int P, const float* __restrict__ fs, const float* __restrict__ ft,
                            const float* __restrict__ search, int n_search, const float* __restrict__ taps,
                            int n_taps, float widest, float* __restrict__ out) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)A * P) return;
    int a = i / P, p = i % P;
    int light = (int)lights[a];
    float3 lp = make_float3(light_pos[light * 3], light_pos[light * 3 + 1], light_pos[light * 3 + 2]);
    float3 xp = make_float3(x[p * 3], x[p * 3 + 1], x[p * 3 + 2]);
    float3 np = make_float3(n[p * 3], n[p * 3 + 1], n[p * 3 + 2]);
    // tangents across the normal, turned by the pixel's angle (raster._frame)
    float3 h = fabsf(np.x) < 0.9f ? make_float3(1.f, 0.f, 0.f) : make_float3(0.f, 1.f, 0.f);
    float3 t = make_float3(np.y * h.z - np.z * h.y, np.z * h.x - np.x * h.z, np.x * h.y - np.y * h.x);
    float tl = rsqrtf(fmaxf(t.x * t.x + t.y * t.y + t.z * t.z, 1e-24f));
    t = make_float3(t.x * tl, t.y * tl, t.z * tl);
    float3 b = make_float3(np.y * t.z - np.z * t.y, np.z * t.x - np.x * t.z, np.x * t.y - np.y * t.x);
    float ca = cosf(angle[p]), sa = sinf(angle[p]);
    float3 t2 = make_float3(ca * t.x + sa * b.x, ca * t.y + sa * b.y, ca * t.z + sa * b.z);
    float3 b2 = make_float3(ca * b.x - sa * t.x, ca * b.y - sa * t.y, ca * b.z - sa * t.z);
    float3 d0 = make_float3(xp.x - lp.x, xp.y - lp.y, xp.z - lp.z);
    Test centre = cube_test(cubes, light, side, d0, np, fs, ft);
    float r = radius[light];
    float reach = fminf(r, widest);
    int found = 0;
    float caster = 0.f;
    for (int k = 0; k < n_search; ++k) {
        float u = search[2 * k] * reach, v = search[2 * k + 1] * reach;
        float3 d = make_float3(d0.x + u * t2.x + v * b2.x, d0.y + u * t2.y + v * b2.y, d0.z + u * t2.z + v * b2.z);
        Test q = cube_test(cubes, light, side, d, np, fs, ft);
        if (q.ma > q.limit) { ++found; caster += q.stored; }
    }
    float lit;
    if (!found) {
        lit = centre.ma <= centre.limit ? 1.f : 0.f;
    } else {
        caster /= found;
        float w = fminf(fmaxf(r * (centre.ma - caster) / fmaxf(caster, 1e-3f), 0.f), widest);
        int seen = 0;
        for (int k = 0; k < n_taps; ++k) {
            float u = taps[2 * k] * w, v = taps[2 * k + 1] * w;
            float3 d = make_float3(d0.x + u * t2.x + v * b2.x, d0.y + u * t2.y + v * b2.y, d0.z + u * t2.z + v * b2.z);
            Test q = cube_test(cubes, light, side, d, np, fs, ft);
            seen += q.ma <= q.limit;
        }
        lit = (float)seen / n_taps;
    }
    out[i] = lit;
}

torch::Tensor soft_shadows(torch::Tensor cubes, torch::Tensor light_pos, torch::Tensor radius, torch::Tensor lights,
                           torch::Tensor x, torch::Tensor n, torch::Tensor angle, torch::Tensor face_s,
                           torch::Tensor face_t, torch::Tensor search, torch::Tensor taps, double widest) {
    int A = lights.size(0), P = x.size(0);
    auto out = torch::empty({A, P}, x.options());
    long total = (long)A * P;
    if (total == 0) return out;
    int threads = 256;
    long blocks = (total + threads - 1) / threads;
    soft_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        cubes.data_ptr<float>(), cubes.size(2), light_pos.data_ptr<float>(), radius.data_ptr<float>(),
        lights.data_ptr<long>(), A, x.data_ptr<float>(), n.data_ptr<float>(), angle.data_ptr<float>(), P,
        face_s.data_ptr<float>(), face_t.data_ptr<float>(), search.data_ptr<float>(), search.size(0),
        taps.data_ptr<float>(), taps.size(0), (float)widest, out.data_ptr<float>());
    return out;
}

// screen-space ambient occlusion of the hit pixels, as RasterScene._occlusion describes it: raw into an image
__global__ void occlusion_kernel(const float* __restrict__ x, const float* __restrict__ n, const long* __restrict__ hit,
                                 const float* __restrict__ angle, const float* __restrict__ depth,
                                 const float* __restrict__ mvp, const float* __restrict__ eye_fwd,
                                 const float* __restrict__ kernel, int K, int W, int H, float radius, float bias,
                                 float near, int P, float* __restrict__ raw) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    float3 xp = make_float3(x[p * 3], x[p * 3 + 1], x[p * 3 + 2]);
    float3 np = make_float3(n[p * 3], n[p * 3 + 1], n[p * 3 + 2]);
    float3 h = fabsf(np.x) < 0.9f ? make_float3(1.f, 0.f, 0.f) : make_float3(0.f, 1.f, 0.f);
    float3 t = make_float3(np.y * h.z - np.z * h.y, np.z * h.x - np.x * h.z, np.x * h.y - np.y * h.x);
    float tl = rsqrtf(fmaxf(t.x * t.x + t.y * t.y + t.z * t.z, 1e-24f));
    t = make_float3(t.x * tl, t.y * tl, t.z * tl);
    float3 b = make_float3(np.y * t.z - np.z * t.y, np.z * t.x - np.x * t.z, np.x * t.y - np.y * t.x);
    float ca = cosf(angle[p]), sa = sinf(angle[p]);
    float3 t2 = make_float3(ca * t.x + sa * b.x, ca * t.y + sa * b.y, ca * t.z + sa * b.z);
    float3 b2 = make_float3(ca * b.x - sa * t.x, ca * b.y - sa * t.y, ca * b.z - sa * t.z);
    float3 eye = make_float3(eye_fwd[0], eye_fwd[1], eye_fwd[2]), fwd = make_float3(eye_fwd[3], eye_fwd[4], eye_fwd[5]);
    long pix = hit[p];
    float z0 = depth[pix];
    float hidden = 0.f;
    for (int k = 0; k < K; ++k) {
        float kx = kernel[3 * k] * radius, ky = kernel[3 * k + 1] * radius, kz = kernel[3 * k + 2] * radius;
        float3 s = make_float3(xp.x + kx * t2.x + ky * b2.x + kz * np.x, xp.y + kx * t2.y + ky * b2.y + kz * np.y,
                               xp.z + kx * t2.z + ky * b2.z + kz * np.z);
        float cx = mvp[0] * s.x + mvp[1] * s.y + mvp[2] * s.z + mvp[3];
        float cy = mvp[4] * s.x + mvp[5] * s.y + mvp[6] * s.z + mvp[7];
        float cw = mvp[12] * s.x + mvp[13] * s.y + mvp[14] * s.z + mvp[15];
        if (cw <= near) continue;
        int col = (int)floorf((cx / cw + 1.f) * (0.5f * W)), row = (int)floorf((cy / cw + 1.f) * (0.5f * H));
        if (col < 0 || col >= W || row < 0 || row >= H) continue;
        float scene = depth[(long)row * W + col];
        float sz = (s.x - eye.x) * fwd.x + (s.y - eye.y) * fwd.y + (s.z - eye.z) * fwd.z;
        if (scene < sz - bias) hidden += fminf(radius / fmaxf(fabsf(z0 - scene), 1e-6f), 1.f);
    }
    raw[pix] = 1.f - hidden / K;
}

// the tiled turn averaged out over 4 x 4 pixels, not across an edge in depth
__global__ void blur_kernel(const float* __restrict__ raw, const float* __restrict__ depth, const long* __restrict__ hit,
                            int W, int H, int P, float* __restrict__ out) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    long pix = hit[p];
    int row = pix / W, col = pix % W;
    float z = depth[pix], total = 0.f, weight = 0.f;
    for (int dy = -2; dy <= 1; ++dy)
        for (int dx = -2; dx <= 1; ++dx) {
            int r = min(max(row + dy, 0), H - 1), c = min(max(col + dx, 0), W - 1);
            float zz = depth[(long)r * W + c];
            if (fabsf(zz - z) < 0.05f * z) { total += raw[(long)r * W + c]; weight += 1.f; }
        }
    out[p] = weight > 0.f ? total / weight : 1.f;
}

torch::Tensor occlusion(torch::Tensor x, torch::Tensor n, torch::Tensor hit, torch::Tensor angle, torch::Tensor depth,
                        torch::Tensor mvp, torch::Tensor eye, torch::Tensor forward, torch::Tensor kernel,
                        int64_t width, int64_t height, double radius, double bias, double near) {
    int P = x.size(0);
    auto raw = torch::ones({height * width}, x.options());
    auto out = torch::empty({P}, x.options());
    if (P == 0) return out;
    auto eye_fwd = torch::cat({eye, forward}).contiguous();  // stays on the GPU: no wait for the frame so far
    auto stream = c10::cuda::getCurrentCUDAStream();
    int threads = 256, blocks = (P + threads - 1) / threads;
    occlusion_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(), n.data_ptr<float>(), hit.data_ptr<long>(), angle.data_ptr<float>(), depth.data_ptr<float>(),
        mvp.data_ptr<float>(), eye_fwd.data_ptr<float>(), kernel.data_ptr<float>(), kernel.size(0), width, height, (float)radius,
        (float)bias, (float)near, P, raw.data_ptr<float>());
    blur_kernel<<<blocks, threads, 0, stream>>>(raw.data_ptr<float>(), depth.data_ptr<float>(), hit.data_ptr<long>(),
                                                width, height, P, out.data_ptr<float>());
    return out;
}

// the light probes' irradiance at a point, as raster_gi.Probes.irradiance describes it: eight probes about the
// point, weighted (trilinear, in front of the surface, Chebyshev on their distances, valid), their order-2
// spherical harmonics blended and evaluated about the normal and the mirror direction
__device__ __forceinline__ float2 oct_coords(float3 d) {
    float l = fabsf(d.x) + fabsf(d.y) + fabsf(d.z);
    float px = d.x / fmaxf(l, 1e-12f), py = d.y / fmaxf(l, 1e-12f);
    if (d.z < 0.f) {
        float qx = (1.f - fabsf(py)) * (px >= 0.f ? 1.f : -1.f), qy = (1.f - fabsf(px)) * (py >= 0.f ? 1.f : -1.f);
        px = qx; py = qy;
    }
    return make_float2(px * 0.5f + 0.5f, py * 0.5f + 0.5f);
}

__device__ __forceinline__ void sh_basis(float3 d, float* y) {
    y[0] = 0.282095f; y[1] = 0.488603f * d.y; y[2] = 0.488603f * d.z; y[3] = 0.488603f * d.x;
    y[4] = 1.092548f * d.x * d.y; y[5] = 1.092548f * d.y * d.z; y[6] = 0.315392f * (3.f * d.z * d.z - 1.f);
    y[7] = 1.092548f * d.x * d.z; y[8] = 0.546274f * (d.x * d.x - d.y * d.y);
}

__global__ void probe_kernel(const float* __restrict__ x, const float* __restrict__ n, const float* __restrict__ ng,
                             const float* __restrict__ r, int has_r, const float* __restrict__ grid,
                             const long* __restrict__ dims, const float* __restrict__ sh,
                             const float* __restrict__ mean, const float* __restrict__ mean2,
                             const float* __restrict__ valid, int side, int P, float* __restrict__ out) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    int nx = dims[0], ny = dims[1], nz = dims[2];
    float step = grid[3];
    float3 g = make_float3(ng[p * 3], ng[p * 3 + 1], ng[p * 3 + 2]);
    float3 q = make_float3(x[p * 3] + g.x * 0.25f * step, x[p * 3 + 1] + g.y * 0.25f * step,
                           x[p * 3 + 2] + g.z * 0.25f * step);
    float gx = (q.x - grid[0]) / step, gy = (q.y - grid[1]) / step, gz = (q.z - grid[2]) / step;
    int bx = (int)floorf(gx), by = (int)floorf(gy), bz = (int)floorf(gz);
    float fx = gx - bx, fy = gy - by, fz = gz - bz;
    float acc[36];
    for (int k = 0; k < 36; ++k) acc[k] = 0.f;
    float wsum = 0.f;
    for (int corner = 0; corner < 8; ++corner) {
        int ox = (corner >> 2) & 1, oy = (corner >> 1) & 1, oz = corner & 1;
        int cx = bx + ox, cy = by + oy, cz = bz + oz;
        if (cx < 0 || cy < 0 || cz < 0 || cx >= nx || cy >= ny || cz >= nz) continue;
        long i = ((long)cx * ny + cy) * nz + cz;
        if (valid[i] <= 0.f) continue;
        float tri = (ox ? fx : 1.f - fx) * (oy ? fy : 1.f - fy) * (oz ? fz : 1.f - fz);
        float3 pp = make_float3(grid[0] + cx * step, grid[1] + cy * step, grid[2] + cz * step);
        float3 to = make_float3(q.x - pp.x, q.y - pp.y, q.z - pp.z);
        float dist = fmaxf(sqrtf(to.x * to.x + to.y * to.y + to.z * to.z), 1e-6f);
        float3 u = make_float3(to.x / dist, to.y / dist, to.z / dist);
        float front = (-(u.x * g.x + u.y * g.y + u.z * g.z) + 1.f) * 0.5f;
        float w = tri * (front * front + 0.2f);
        float2 oc = oct_coords(u);
        float cxo = oc.x * side - 0.5f, cyo = oc.y * side - 0.5f;
        int tx0 = (int)floorf(cxo), ty0 = (int)floorf(cyo);
        float ftx = cxo - tx0, fty = cyo - ty0, mu = 0.f, mu2 = 0.f;
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) {
                int tx = min(max(tx0 + dx, 0), side - 1), ty = min(max(ty0 + dy, 0), side - 1);
                float bw = (dx ? ftx : 1.f - ftx) * (dy ? fty : 1.f - fty);
                mu += bw * mean[i * side * side + ty * side + tx];
                mu2 += bw * mean2[i * side * side + ty * side + tx];
            }
        float var = fmaxf(fabsf(mu2 - mu * mu), 1e-4f);
        float over = fmaxf(dist - mu, 0.f);
        float cheb = var / (var + over * over);
        w *= fmaxf(dist <= mu ? 1.f : cheb * cheb * cheb, 1e-3f);
        for (int k = 0; k < 36; ++k) acc[k] += w * sh[i * 36 + k];
        wsum += w;
    }
    float inv = 1.f / fmaxf(wsum, 1e-9f);
    const float band[9] = {3.141593f, 2.094395f, 2.094395f, 2.094395f, 0.785398f, 0.785398f, 0.785398f, 0.785398f, 0.785398f};
    float y[9];
    for (int which = 0; which < 1 + has_r; ++which) {
        const float* dir = which ? r : n;
        sh_basis(make_float3(dir[p * 3], dir[p * 3 + 1], dir[p * 3 + 2]), y);
        for (int l = 0; l < 4; ++l) {
            float e = 0.f;
            for (int k = 0; k < 9; ++k) e += band[k] * y[k] * acc[k * 4 + l];
            out[(which * (long)P + p) * 4 + l] = fmaxf(e * inv, 0.f);
        }
    }
}

torch::Tensor probe_irradiance(torch::Tensor x, torch::Tensor n, torch::Tensor n_geo, torch::Tensor r,
                               torch::Tensor grid, torch::Tensor dims, torch::Tensor sh, torch::Tensor mean,
                               torch::Tensor mean2, torch::Tensor valid, int64_t side) {
    int P = x.size(0);
    int has_r = r.numel() > 0 ? 1 : 0;
    auto out = torch::empty({1 + has_r, P, 4}, x.options());
    if (P == 0) return out;
    int threads = 128, blocks = (P + threads - 1) / threads;
    probe_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), n.data_ptr<float>(), n_geo.data_ptr<float>(), has_r ? r.data_ptr<float>() : nullptr,
        has_r, grid.data_ptr<float>(), dims.data_ptr<long>(), sh.data_ptr<float>(), mean.data_ptr<float>(),
        mean2.data_ptr<float>(), valid.data_ptr<float>(), (int)side, P, out.data_ptr<float>());
    return out;
}

// trilinear, repeating texture lookups, as RasterScene._sample does them: the mip level from the uv's screen
// derivatives, bilinear in the two levels about it
__device__ __forceinline__ float3 bilinear(const unsigned char* __restrict__ atlas, const float* __restrict__ lut,
                                           bool srgb, long base, int w, int h, float u, float v) {
    float x = u * w - 0.5f, y = v * h - 0.5f;
    float x0f = floorf(x), y0f = floorf(y), fx = x - x0f, fy = y - y0f;
    long x0 = (long)x0f, y0 = (long)y0f;
    x0 = ((x0 % w) + w) % w; y0 = ((y0 % h) + h) % h;
    long x1 = (x0 + 1) % w, y1 = (y0 + 1) % h;
    float3 c = make_float3(0.f, 0.f, 0.f);
    long xs[2] = {x0, x1}, ys[2] = {y0, y1};
    for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i) {
            float wgt = (i ? fx : 1.f - fx) * (j ? fy : 1.f - fy);
            const unsigned char* t = atlas + (base + ys[j] * w + xs[i]) * 3;
            float r = srgb ? lut[t[0]] : t[0] / 255.f, g = srgb ? lut[t[1]] : t[1] / 255.f, b = srgb ? lut[t[2]] : t[2] / 255.f;
            c.x += wgt * r; c.y += wgt * g; c.z += wgt * b;
        }
    return c;
}

__global__ void texture_kernel(const long* __restrict__ layer, const float* __restrict__ uv, const float* __restrict__ duv,
                               const float* __restrict__ to_uv, const long* __restrict__ tw, const long* __restrict__ th,
                               const long* __restrict__ off, const long* __restrict__ levels, const bool* __restrict__ srgb,
                               const unsigned char* __restrict__ atlas, const float* __restrict__ lut, int L, int P,
                               float* __restrict__ out) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    long t = layer[p];
    const float* m = to_uv + t * 6;
    float a = uv[p * 2], b = uv[p * 2 + 1];
    float u = m[0] * a + m[1] * b + m[2], v = m[3] * a + m[4] * b + m[5];
    float w0 = (float)tw[t * L], h0 = (float)th[t * L], span = 0.f;
    for (int k = 0; k < 4; k += 2) {
        float du = (m[0] * duv[p * 4 + k] + m[1] * duv[p * 4 + k + 1]) * w0;
        float dv = (m[3] * duv[p * 4 + k] + m[4] * duv[p * 4 + k + 1]) * h0;
        span = fmaxf(span, du * du + dv * dv);
    }
    float top = (float)(levels[t] - 1);
    float lod = fminf(fmaxf(0.5f * log2f(fmaxf(span, 1e-12f)), 0.f), top);
    int l0 = (int)floorf(lod), l1 = min(l0 + 1, (int)top);
    float f = lod - l0;
    bool s = srgb[t];
    float3 c0 = bilinear(atlas, lut, s, off[t * L + l0], tw[t * L + l0], th[t * L + l0], u, v);
    float3 c1 = f > 0.f ? bilinear(atlas, lut, s, off[t * L + l1], tw[t * L + l1], th[t * L + l1], u, v) : c0;
    out[p * 3] = c0.x * (1.f - f) + c1.x * f;
    out[p * 3 + 1] = c0.y * (1.f - f) + c1.y * f;
    out[p * 3 + 2] = c0.z * (1.f - f) + c1.z * f;
}

torch::Tensor texture_sample(torch::Tensor layer, torch::Tensor uv, torch::Tensor duv, torch::Tensor to_uv,
                             torch::Tensor tw, torch::Tensor th, torch::Tensor off, torch::Tensor levels,
                             torch::Tensor srgb, torch::Tensor atlas, torch::Tensor lut) {
    int P = layer.size(0);
    auto out = torch::empty({P, 3}, uv.options());
    if (P == 0) return out;
    int threads = 256, blocks = (P + threads - 1) / threads;
    texture_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        layer.data_ptr<long>(), uv.data_ptr<float>(), duv.data_ptr<float>(), to_uv.data_ptr<float>(),
        tw.data_ptr<long>(), th.data_ptr<long>(), off.data_ptr<long>(), levels.data_ptr<long>(), srgb.data_ptr<bool>(),
        atlas.data_ptr<unsigned char>(), lut.data_ptr<float>(), (int)tw.size(1), P, out.data_ptr<float>());
    return out;
}

// --- polarized direct light ---
struct Cx { float r, i; };
__device__ __forceinline__ Cx cmul(Cx a, Cx b) { return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r}; }
__device__ __forceinline__ Cx cdiv(Cx a, Cx b) {
    float d = b.r * b.r + b.i * b.i;
    return {(a.r * b.r + a.i * b.i) / d, (a.i * b.r - a.r * b.i) / d};
}
__device__ __forceinline__ Cx csqrt(Cx z) {
    float m = sqrtf(z.r * z.r + z.i * z.i);
    return {sqrtf(fmaxf(0.5f * (m + z.r), 0.f)), copysignf(sqrtf(fmaxf(0.5f * (m - z.r), 0.f)), z.i)};
}

// Mitsuba's mueller::specular_reflection (fresnel_polarized, Verdet's sign of a_p) for cos_theta_i >= 0 and a
// complex index eta + i k: its a, b and c cos(delta) (the S3 terms are not needed: no circular light here)
__device__ float3 fresnel_abc(float ci, float eta, float k) {
    if (k == 0.f && fabsf(eta - 1.f) < 1e-6f) return make_float3(0.f, 0.f, 0.f);
    Cx e = {eta, -fabsf(k)};  // kappa negative, as Mitsuba's polarized Fresnel takes it
    Cx rcp = cdiv({1.f, 0.f}, e), r2 = cmul(rcp, rcp);
    float s2 = 1.f - ci * ci;
    Cx ct = csqrt({1.f - r2.r * s2, -r2.i * s2});
    if (ct.i > 0.f) ct.i = -ct.i;
    Cx ect = cmul(e, ct);
    Cx as = cdiv({ci - ect.r, -ect.i}, {ci + ect.r, ect.i});
    Cx ap = cdiv({e.r * ci - ct.r, e.i * ci - ct.i}, {e.r * ci + ct.r, e.i * ci + ct.i});
    float rs = as.r * as.r + as.i * as.i, rp = ap.r * ap.r + ap.i * ap.i, c = sqrtf(rs * rp);
    float cosd = c > 0.f ? (ap.r * as.r + ap.i * as.i) / c : 0.f;
    return make_float3(0.5f * (rs + rp), 0.5f * (rs - rp), c * cosd);
}

__device__ __forceinline__ float3 f3(const float* p) { return make_float3(p[0], p[1], p[2]); }
__device__ __forceinline__ float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ float3 cross3(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ __forceinline__ float3 unit3(float3 a) {
    float l = rsqrtf(fmaxf(dot3(a, a), 1e-30f));
    return make_float3(a.x * l, a.y * l, a.z * l);
}
// sin, cos of twice the signed angle that turns Stokes basis `from` into `to` about `fwd` (rotate_stokes_basis)
__device__ __forceinline__ float2 basis_turn(float3 fwd, float3 from, float3 to) {
    float th = acosf(fminf(fmaxf(dot3(from, to), -1.f), 1.f));
    if (dot3(fwd, cross3(from, to)) < 0.f) th = -th;
    float s, c;
    sincosf(2.f * th, &s, &c);
    return make_float2(s, c);
}
// a unit vector across `fwd` (where a reflection's plane is not defined)
__device__ __forceinline__ float3 any_across(float3 fwd) {
    float3 h = fabsf(fwd.x) < 0.9f ? make_float3(1.f, 0.f, 0.f) : make_float3(0.f, 1.f, 0.f);
    return unit3(cross3(fwd, h));
}

__global__ void shade_kernel(const float* __restrict__ x, const float* __restrict__ n, const float* __restrict__ ng,
                             const float* __restrict__ view, const float* __restrict__ a2v,
                             const float* __restrict__ spec, const float* __restrict__ vis,
                             const float* __restrict__ lpos, const float* __restrict__ lnorm,
                             const long* __restrict__ lmode, const float* __restrict__ lpower,
                             const float* __restrict__ lr2, const float* __restrict__ lpol,
                             const float* __restrict__ lframe, const float* __restrict__ up, float graze,
                             int A, int P, float* __restrict__ out) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    float3 X = f3(x + p * 3), N = f3(n + p * 3), G = f3(ng + p * 3), V = f3(view + p * 3);
    float a2 = a2v[p], nv = fmaxf(dot3(N, V), 1e-4f);
    const float* sp = spec + (long)p * 26;  // shares[2], eta[2][4], k[2][4], tint[2][4]
    float3 xref = cross3(make_float3(-V.x, -V.y, -V.z), f3(up));
    xref = dot3(xref, xref) > 1e-12f ? unit3(xref) : any_across(V);
    float E[4] = {0.f, 0.f, 0.f, 0.f}, S[3][4] = {{0.f}};
    for (int a = 0; a < A; ++a) {
        float v = vis[(long)a * P + p];
        if (v <= 0.f) continue;
        float3 to = make_float3(lpos[a * 3] - X.x, lpos[a * 3 + 1] - X.y, lpos[a * 3 + 2] - X.z);
        float dd = dot3(to, to);
        float3 w = unit3(to);
        float cos_s = fmaxf(dot3(w, N), 0.f);
        float facing = -dot3(w, f3(lnorm + a * 3));
        long mode = lmode[a];
        float cos_l = mode == 1 ? fabsf(facing) : (mode == 0 ? fmaxf(facing, 0.f) : 0.25f);
        float fade = fminf(fmaxf(dot3(w, G) / graze, 0.f), 1.f);
        float g = cos_l * cos_s * fade / (dd + lr2[a]) * v;
        if (g <= 0.f) continue;
        for (int l = 0; l < 4; ++l) E[l] += g * lpower[a * 4 + l];
        if (sp[0] <= 0.f && sp[1] <= 0.f) continue;
        float3 H = unit3(make_float3(w.x + V.x, w.y + V.y, w.z + V.z));
        float nh = fmaxf(dot3(H, N), 0.f), q = nh * nh * (a2 - 1.f) + 1.f;
        float D = a2 / (3.14159265f * q * q);
        float vg = 0.5f / fmaxf(cos_s * sqrtf(nv * nv * (1.f - a2) + a2) + nv * sqrtf(cos_s * cos_s * (1.f - a2) + a2), 1e-6f);
        float lobe = g * D * vg;
        float ch = fminf(fmaxf(dot3(H, V), 0.f), 1.f);
        // the light's Stokes vector in the reflection's basis (across the plane of H and the light's way in)
        float3 k_in = make_float3(-w.x, -w.y, -w.z);
        float3 s_ax = cross3(H, k_in);
        s_ax = dot3(s_ax, s_ax) > 1e-12f ? unit3(s_ax) : any_across(V);
        float S1i = 0.f, S2i = 0.f;
        if (lpol[a] > 0.f) {  // polarized_area: linear along Mitsuba's canonical basis of the way out, in its frame
            const float* fr = lframe + a * 10;
            float3 fs = f3(fr), ft = f3(fr + 3), fn = f3(fr + 6);
            float3 loc = make_float3(dot3(k_in, fs), dot3(k_in, ft), dot3(k_in, fn));
            float sign = copysignf(1.f, loc.z), ca = -1.f / (sign + loc.z), cb = loc.x * loc.y * ca;
            float3 sl = make_float3(1.f + sign * loc.x * loc.x * ca, sign * cb, -sign * loc.x);  // coordinate_system
            float th = fr[9];
            if (th != 0.f) {
                float3 o = cross3(loc, sl);
                float st, ct;
                sincosf(th, &st, &ct);
                sl = make_float3(ct * sl.x + st * o.x, ct * sl.y + st * o.y, ct * sl.z + st * o.z);
            }
            float3 e = make_float3(sl.x * fs.x + sl.y * ft.x + sl.z * fn.x, sl.x * fs.y + sl.y * ft.y + sl.z * fn.y,
                                   sl.x * fs.z + sl.y * ft.z + sl.z * fn.z);
            float ek = dot3(e, k_in);
            e = unit3(make_float3(e.x - ek * k_in.x, e.y - ek * k_in.y, e.z - ek * k_in.z));
            float2 t = basis_turn(k_in, e, s_ax);  // (1, 1, 0) along e, in s_ax's basis
            S1i = t.y;
            S2i = -t.x;
        }
        float2 r = basis_turn(V, s_ax, xref);  // the way out: to the camera's reference basis
        for (int l = 0; l < 4; ++l) {
            float fa = 0.f, fb = 0.f, fc = 0.f;
            for (int j = 0; j < 2; ++j) {
                float share = sp[j];
                if (share <= 0.f) continue;
                float tint = sp[18 + j * 4 + l];
                float3 f = fresnel_abc(ch, sp[2 + j * 4 + l], sp[10 + j * 4 + l]);
                fa += share * tint * f.x; fb += share * tint * f.y; fc += share * tint * f.z;
            }
            float scale = lobe * lpower[a * 4 + l];
            float o0 = fa + fb * S1i, o1 = fb + fa * S1i, o2 = fc * S2i;
            S[0][l] += scale * o0;
            S[1][l] += scale * (r.y * o1 + r.x * o2);
            S[2][l] += scale * (-r.x * o1 + r.y * o2);
        }
    }
    float* o = out + (long)p * 16;
    for (int l = 0; l < 4; ++l) { o[l] = E[l]; o[4 + l] = S[0][l]; o[8 + l] = S[1][l]; o[12 + l] = S[2][l]; }
}

torch::Tensor shade_lights(torch::Tensor x, torch::Tensor n, torch::Tensor ng, torch::Tensor view, torch::Tensor a2,
                           torch::Tensor spec, torch::Tensor vis, torch::Tensor lpos, torch::Tensor lnorm,
                           torch::Tensor lmode, torch::Tensor lpower, torch::Tensor lr2, torch::Tensor lpol,
                           torch::Tensor lframe, torch::Tensor up, double graze) {
    int P = x.size(0), A = lpos.size(0);
    auto out = torch::zeros({P, 16}, x.options());
    if (P == 0 || A == 0) return out;
    int threads = 128, blocks = (P + threads - 1) / threads;
    shade_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), n.data_ptr<float>(), ng.data_ptr<float>(), view.data_ptr<float>(), a2.data_ptr<float>(),
        spec.data_ptr<float>(), vis.data_ptr<float>(), lpos.data_ptr<float>(), lnorm.data_ptr<float>(),
        lmode.data_ptr<long>(), lpower.data_ptr<float>(), lr2.data_ptr<float>(), lpol.data_ptr<float>(),
        lframe.data_ptr<float>(), up.data_ptr<float>(), (float)graze, A, P, out.data_ptr<float>());
    return out;
}
"""

_module = None
_failed = False


def kernels():
    """The built module (cached under torch's extension directory), or None if it cannot be built."""
    global _module, _failed
    if _module is None and not _failed:
        try:
            _module = torch.utils.cpp_extension.load_inline(
                "opticalnav_raster_kernels", _CPP, cuda_sources=_CUDA, functions=["soft_shadows", "occlusion", "probe_irradiance", "texture_sample", "shade_lights"],
                extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
        except Exception as exc:  # noqa: BLE001 - no compiler, no headers: hard shadows instead
            print(f"[raster] CUDA kernels not built ({type(exc).__name__}: {str(exc)[:200]}): hard shadows", flush=True)
            _failed = True
    return _module
