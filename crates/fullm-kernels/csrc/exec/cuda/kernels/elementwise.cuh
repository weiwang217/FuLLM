// SPDX-License-Identifier: Apache-2.0
// kernels/elementwise.cuh - norms (rmsnorm, the fused add + norm and MoE reduce + norm),
// elementwise ops, row copies, the embedding gather.
//
// Part of cuda_backend.cu's translation unit, included there after the CUDA and standard
// headers, in the order common, formats, gemv, convert, router, elementwise, copy, kimi, mla,
// attention, sampling, moe (each file uses only those before it). Kernels and their device
// helpers: no state.
#pragma once

namespace fullm {
namespace {

// ------------------------------------------------------------------ elementwise and norms

__global__ void rmsnorm_kernel(float *y, const float *x, const float *__restrict__ w, int64_t n, int64_t stride,
                               float eps) {
    __shared__ float red[32];
    pdl_trigger();
    const float *xr = x + static_cast<int64_t>(blockIdx.x) * stride;
    float *yr = y + static_cast<int64_t>(blockIdx.x) * stride;
    float ss = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        ss = fmaf(v, v, ss);
    }
    ss = block_sum(ss, red);
    const float inv = 1.0f / sqrtf(ss / static_cast<float>(n) + eps);
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) yr[i] = w[i] * (xr[i] * inv);
}

// four consecutive elements of a row of T (fp32, or bf16 bits) as floats
__device__ __forceinline__ float4 row4(const float *r, int c) { return reinterpret_cast<const float4 *>(r)[c]; }
__device__ __forceinline__ float4 row4(const uint16_t *r, int c) {
    const uint2 u = reinterpret_cast<const uint2 *>(r)[c];
    return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xFFFF0000u), __uint_as_float(u.y << 16),
                       __uint_as_float(u.y & 0xFFFF0000u));
}
__device__ __forceinline__ float row_at(const float *r, int64_t i) { return r[i]; }
__device__ __forceinline__ float row_at(const uint16_t *r, int64_t i) {
    return __uint_as_float(static_cast<uint32_t>(r[i]) << 16);
}

// A thread's PER consecutive elements of a row as PER / 4 chunks of 4 (chunks >= n_chunks read as
// zeros), and the stores back (fp32, or bf16 bits, 4 at a time).
template <int PER, typename T>
__device__ __forceinline__ void load_row(const T *r, int chunk0, int n_chunks, float (&v)[PER]) {
#pragma unroll
    for (int q = 0; q < PER / 4; ++q) {
        const int c = chunk0 + q;
        const float4 f = c < n_chunks ? row4(r, c) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        v[4 * q] = f.x, v[4 * q + 1] = f.y, v[4 * q + 2] = f.z, v[4 * q + 3] = f.w;
    }
}
__device__ __forceinline__ void store4(float *r, int c, float4 v) { reinterpret_cast<float4 *>(r)[c] = v; }
__device__ __forceinline__ void store4(uint16_t *r, int c, float4 v) {
    uint2 u;
    u.x = static_cast<uint32_t>(f2bf(v.x)) | (static_cast<uint32_t>(f2bf(v.y)) << 16);
    u.y = static_cast<uint32_t>(f2bf(v.z)) | (static_cast<uint32_t>(f2bf(v.w)) << 16);
    reinterpret_cast<uint2 *>(r)[c] = u;
}

// rmsnorm_kernel with the row in registers, PER consecutive elements per thread
// (i = PER * threadIdx.x + j; n % 4 == 0, n <= PER * blockDim): chunked loads and stores. The
// per-thread order and the block tree fix the bits of the sum of squares, so this and
// add_rmsnorm_kernel<PER> agree bitwise whenever the launchers pick the same PER (norm_per).
template <int PER>
__global__ void rmsnorm_vec_kernel(float *y, const float *x, const float *__restrict__ w, int64_t n, int64_t stride,
                                   float eps) {
    __shared__ float red[32];
    const int nc = static_cast<int>(n >> 2), c0 = threadIdx.x * (PER / 4);
    float wv[PER];
    load_row<PER>(w, c0, nc, wv);
    pdl_trigger();
    const float *xr = x + static_cast<int64_t>(blockIdx.x) * stride;
    float *yr = y + static_cast<int64_t>(blockIdx.x) * stride;
    float v[PER];
    load_row<PER>(xr, c0, nc, v);
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < PER; ++j) ss = fmaf(v[j], v[j], ss);
    ss = block_sum(ss, red);
    const float inv = 1.0f / sqrtf(ss / static_cast<float>(n) + eps);
#pragma unroll
    for (int q = 0; q < PER / 4; ++q)
        if (c0 + q < nc)
            store4(yr, c0 + q,
                   make_float4(wv[4 * q] * (v[4 * q] * inv), wv[4 * q + 1] * (v[4 * q + 1] * inv),
                               wv[4 * q + 2] * (v[4 * q + 2] * inv), wv[4 * q + 3] * (v[4 * q + 3] * inv)));
}

// A pair's row: contrib[p], or wts[p] * dn[inv[p]] when the rows were left in expert order
// (inv[p] < 0: not routed there, so contrib). The same bits as the scatter it replaces.
template <typename T>
__device__ __forceinline__ float pair_val(const float *contrib, const T *dn, const int32_t *inv, const float *wts,
                                          int64_t p, int64_t width, int64_t c) {
    if (inv) {
        const int32_t r = inv[p];
        if (r >= 0) return wts[p] * row_at(dn, static_cast<int64_t>(r) * width + c);
    }
    return contrib[p * width + c];
}

constexpr int kMaxReduceK = 16;  // pairs per token the fused reduce handles

// h += d, then y = rmsnorm(h): add_kernel then rmsnorm_kernel in one pass, the same bits
// (the same per-thread element order and block tree at kThreads threads).
// PER > 0: the row is held in registers, PER consecutive elements per thread
// (i = PER * threadIdx.x + j; n % 4 == 0, n <= PER * blockDim), loaded and stored 4 at a time
// (a warp's request is 256-512 contiguous bytes; the strided per-element form ran the MoE reduce
// at 1.8 TB/s). The same per-element arithmetic as the loop; the sum of squares has the bits of
// rmsnorm_vec_kernel<PER> (the launchers pick PER by norm_per for both).
// k > 0: d is the experts' rows [rows*k][n] and the delta is moe_reduce_kernel's sum (j order,
// then + addend), so h, y have the bits of moe_reduce then add_rmsnorm.
// y16 (optional): y as bf16 too (the next GEMMs' operand)
// RED: the k > 0 (MoE reduce) path; its own instantiation, since its unrolled pair loads cost
// 72 registers (3 blocks per SM) and the plain add + norm is bandwidth-bound at full occupancy.
// DT: d's element type (fp32, or the bf16 twin a gemm16 left); y null: only y16 is written.
// CL > 1 (sm90, the plain chunked form): a row's kThreads threads over a cluster of CL blocks of
// kThreads / CL, the sum of squares through cluster_sum: the bits of CL == 1, the row's loads and
// stores spread over CL SMs (a GEMV-sized step has a row an SM otherwise).
template <int PER, bool RED = false, typename T = float, typename DT = float, int CL = 1>
__global__ void add_rmsnorm_kernel(float *h, const DT *d, float *y, const float *__restrict__ w, int64_t n,
                                   float eps, int k, const float *addend, const T *dn, const int32_t *pinv,
                                   const float *wts, uint16_t *y16) {
    __shared__ float red[32];
    static_assert(PER % 4 == 0, "chunks of 4");
    static_assert(CL == 1 || (PER > 0 && !RED), "clusters: the plain chunked form only");
    constexpr int NQ = PER > 0 ? PER / 4 : 1;
    // CL > 1: the row's threads split over a cluster of CL blocks (rank r: threads [r * blockDim, ...))
    const int64_t row = blockIdx.x / CL;
    const int tr = static_cast<int>((blockIdx.x % CL) * blockDim.x + threadIdx.x);
    const int nc = static_cast<int>(n >> 2), c0 = tr * NQ;  // the thread's chunks [c0, c0 + NQ)
    float wv[PER > 0 ? PER : 1];  // the norm weights load while the previous kernel finishes
    if constexpr (PER > 0) load_row<PER>(w, c0, nc, wv);
    pdl_wait();
    pdl_trigger();
    float *hr = h + row * n;
    const DT *dr = d + row * n;
    float *yr = y ? y + row * n : nullptr;
    float ss = 0.0f;
    if constexpr (PER > 0) {
        float v[PER];
        if constexpr (RED) {
            // the row's k pairs: their rows (dn through the inverse list, or contrib) and
            // weights once per block; then every pair's loads in flight together
            __shared__ int32_t srow[kMaxReduceK];
            __shared__ float sw[kMaxReduceK];
            if (threadIdx.x < k) {
                const int64_t p = row * k + threadIdx.x;
                const int32_t r = pinv ? pinv[p] : -1;
                srow[threadIdx.x] = r;
                sw[threadIdx.x] = r >= 0 ? wts[p] : 1.0f;  // x * 1.0f is x: the same bits as no weight
            }
            __syncthreads();
            float a[PER];
#pragma unroll
            for (int j = 0; j < PER; ++j) a[j] = 0.0f;
#pragma unroll 4
            for (int jj = 0; jj < k; ++jj) {
                const int32_t r = srow[jj];
                const float wj = sw[jj];
                float t[PER];
                if (r >= 0) load_row<PER>(dn + static_cast<int64_t>(r) * n, c0, nc, t);
                else load_row<PER>(reinterpret_cast<const float *>(d) + (row * k + jj) * n, c0, nc, t);
#pragma unroll
                for (int j = 0; j < PER; ++j) a[j] = a[j] + wj * t[j];  // weight, then the j-order sum
            }
            if (addend) {
                float ad[PER];
                load_row<PER>(addend + row * n, c0, nc, ad);
#pragma unroll
                for (int j = 0; j < PER; ++j) a[j] = a[j] + ad[j];
            }
            load_row<PER>(hr, c0, nc, v);
#pragma unroll
            for (int j = 0; j < PER; ++j) v[j] = v[j] + a[j];  // chunks past n: 0 + 0
        } else {
            float t[PER];
            load_row<PER>(hr, c0, nc, v);
            load_row<PER>(dr, c0, nc, t);
#pragma unroll
            for (int j = 0; j < PER; ++j) v[j] = v[j] + t[j];
        }
#pragma unroll
        for (int q = 0; q < NQ; ++q)
            if (c0 + q < nc) store4(hr, c0 + q, make_float4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]));
#pragma unroll
        for (int j = 0; j < PER; ++j) ss = fmaf(v[j], v[j], ss);  // zeros past n leave ss as it is
        ss = cluster_sum<CL>(ss, red);
        const float inv = 1.0f / sqrtf(ss / static_cast<float>(n) + eps);
        uint16_t *y16r = y16 ? y16 + row * n : nullptr;
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            if (c0 + q >= nc) continue;
            const float4 yv = make_float4(wv[4 * q] * (v[4 * q] * inv), wv[4 * q + 1] * (v[4 * q + 1] * inv),
                                          wv[4 * q + 2] * (v[4 * q + 2] * inv), wv[4 * q + 3] * (v[4 * q + 3] * inv));
            if (yr) store4(yr, c0 + q, yv);
            if (y16r) store4(y16r, c0 + q, yv);
        }
        return;
    }
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = hr[i] + row_at(dr, i);
        hr[i] = v;
        ss = fmaf(v, v, ss);
    }
    ss = block_sum(ss, red);
    const float inv = 1.0f / sqrtf(ss / static_cast<float>(n) + eps);
    if (yr)
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) yr[i] = w[i] * (hr[i] * inv);
}

// One warp per contiguous vector of d floats (row r, head h is vector r*heads + h).
__global__ void l2norm_kernel(float *x, int64_t nvec, int64_t d, float eps) {
    const int64_t vec = static_cast<int64_t>(blockIdx.x) * 8 + (threadIdx.x >> 5);
    if (vec >= nvec) return;
    const int lane = threadIdx.x & 31;
    float *p = x + vec * d;
    float ss = 0.0f;
    for (int64_t i = lane; i < d; i += 32) ss = fmaf(p[i], p[i], ss);
    ss = warp_sum(ss);
    const float inv = 1.0f / sqrtf(ss + eps);
    for (int64_t i = lane; i < d; i += 32) p[i] *= inv;
}

__global__ void situ_kernel(float *y, const float *g, const float *u, int64_t n, float b1, float b2) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const float gv = g[i];
        const float a = b1 * tanhf(gv / b1) * sigm(gv);  // sigmoid sees the UNCAPPED gate
        const float uu = b2 * tanhf(u[i] / b2);
        y[i] = a * uu;
    }
}

__global__ void silu_kernel(float *y, const float *g, const float *u, int64_t n) {
    pdl_trigger();
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const float gv = g[i];
        y[i] = gv * sigm(gv) * u[i];
    }
}

// The W8A8 GEMM's scales applied after the matmul: y[m][n] = acc[m][n] * sa[n] * sb[m] (sa the
// weight rows' scales, sb the tokens'), for a cuBLASLt that has no outer-vector scale mode (sm_120:
// scalar scales only) and so ran the e4m3 x e4m3 matmul with unit scales into fp32. OUT float: in
// place (acc == y, lda == ldy); uint16_t: rounded to bf16 once, as cuBLASLt's own epilogue would.
// V4: N % 4 == 0 and 16-byte aligned rows, four elements a thread.
template <typename OUT, bool V4>
__global__ void fp8_post_scale_kernel(OUT *y, int64_t ldy, const float *acc, int64_t lda, const float *__restrict__ sa,
                                      const float *__restrict__ sb, int64_t M, int64_t N) {
    constexpr int W = V4 ? 4 : 1;
    const int64_t nw = N / W;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < M * nw;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t m = i / nw, n = (i - m * nw) * W;
        const float s = sb[m];
        if constexpr (V4) {
            const float4 v = *reinterpret_cast<const float4 *>(acc + m * lda + n);
            const float4 a = *reinterpret_cast<const float4 *>(sa + n);
            const float r0 = v.x * (s * a.x), r1 = v.y * (s * a.y), r2 = v.z * (s * a.z), r3 = v.w * (s * a.w);
            if constexpr (std::is_same_v<OUT, float>) {
                *reinterpret_cast<float4 *>(y + m * ldy + n) = make_float4(r0, r1, r2, r3);
            } else {
                uint2 o;
                o.x = static_cast<uint32_t>(f2bf(r0)) | (static_cast<uint32_t>(f2bf(r1)) << 16);
                o.y = static_cast<uint32_t>(f2bf(r2)) | (static_cast<uint32_t>(f2bf(r3)) << 16);
                *reinterpret_cast<uint2 *>(y + m * ldy + n) = o;
            }
        } else {
            const float r = acc[m * lda + n] * (s * sa[n]);
            if constexpr (std::is_same_v<OUT, float>) y[m * ldy + n] = r;
            else y[m * ldy + n] = f2bf(r);
        }
    }
}

// x's rows as e4m3 with a scale each (amax / 448: the per-token dynamic scaling of a W8A8 GEMM), a
// block a row. IN: fp32 rows, or their bf16 twin.
template <typename IN>
__global__ void act_quant_e4m3_kernel(uint8_t *q, float *scale, const IN *x, int64_t K) {
    __shared__ float red[32];
    const IN *xr = x + static_cast<int64_t>(blockIdx.x) * K;
    uint8_t *qr = q + static_cast<int64_t>(blockIdx.x) * K;
    float a = 0.0f;
    for (int64_t i = threadIdx.x; i < K; i += blockDim.x) a = fmaxf(a, fabsf(row_at(xr, i)));
    a = warp_max(a);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) red[wid] = a;
    __syncthreads();
    if (wid == 0) {
        float r = lane < static_cast<int>(blockDim.x >> 5) ? red[lane] : 0.0f;
        r = warp_max(r);
        if (lane == 0) red[0] = r;
    }
    __syncthreads();
    const float s = fmaxf(red[0], 1e-30f) / 448.0f, inv = 1.0f / s;
    if (threadIdx.x == 0) scale[blockIdx.x] = s;
    for (int64_t i = threadIdx.x; i < K; i += blockDim.x) qr[i] = f2e4m3(row_at(xr, i) * inv);
}

// act_quant_e4m3_kernel with the row in registers, one pass over x: PER consecutive elements a
// thread (the row's K <= PER * blockDim), loaded 16 bytes at a time and stored 8 at a time (rows
// of IN 16-byte aligned, K a multiple of 8). The same bits as the loop above.
template <typename IN, int PER>
__global__ void act_quant_vec_kernel(uint8_t *q, float *scale, const IN *x, int64_t K) {
    constexpr int VEC = 16 / static_cast<int>(sizeof(IN));  // 8 bf16 or 4 fp32 a load
    static_assert(PER % 8 == 0 && PER % VEC == 0, "chunks of 8");
    __shared__ float red[32];
    const IN *xr = x + static_cast<int64_t>(blockIdx.x) * K;
    uint8_t *qr = q + static_cast<int64_t>(blockIdx.x) * K;
    const int64_t c0 = static_cast<int64_t>(threadIdx.x) * PER;
    float v[PER];
    float a = 0.0f;
#pragma unroll
    for (int j = 0; j < PER; j += VEC) {
        if (c0 + j < K) {
            const uint4 u = *reinterpret_cast<const uint4 *>(xr + c0 + j);
            if constexpr (VEC == 8) {
                const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                for (int t = 0; t < 4; ++t)
                    v[j + 2 * t] = __uint_as_float(w[t] << 16), v[j + 2 * t + 1] = __uint_as_float(w[t] & 0xffff0000u);
            } else {
                v[j] = __uint_as_float(u.x), v[j + 1] = __uint_as_float(u.y), v[j + 2] = __uint_as_float(u.z),
                v[j + 3] = __uint_as_float(u.w);
            }
        } else {
#pragma unroll
            for (int t = 0; t < VEC; ++t) v[j + t] = 0.0f;
        }
#pragma unroll
        for (int t = 0; t < VEC; ++t) a = fmaxf(a, fabsf(v[j + t]));
    }
    a = warp_max(a);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) red[wid] = a;
    __syncthreads();
    if (wid == 0) {
        float r = lane < static_cast<int>(blockDim.x >> 5) ? red[lane] : 0.0f;
        r = warp_max(r);
        if (lane == 0) red[0] = r;
    }
    __syncthreads();
    const float s = fmaxf(red[0], 1e-30f) / 448.0f, inv = 1.0f / s;
    if (threadIdx.x == 0) scale[blockIdx.x] = s;
#pragma unroll
    for (int j = 0; j < PER; j += 8) {
        if (c0 + j >= K) break;
        uint32_t lo = 0, hi = 0;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            lo |= static_cast<uint32_t>(f2e4m3(v[j + t] * inv)) << (8 * t);
            hi |= static_cast<uint32_t>(f2e4m3(v[j + 4 + t] * inv)) << (8 * t);
        }
        *reinterpret_cast<uint2 *>(qr + c0 + j) = make_uint2(lo, hi);
    }
}

// ---- the split dense MLP's big steps (CudaBackend::split_mlp): int4 weights as e4m3, activations
// as e4m3 with a scale per row per 128 columns, for cuBLASLt's block-scaled fp8 GEMM

// n int4 matrices (int4_group.h: [rows][cols] each, fragment or row-major layout) as e4m3: matrix e
// at dst + e * step, rows ld bytes apart. The integers q in [-8, 7] are exact in e4m3 (the group
// scales go to cuBLASLt as they are); a nibble's byte through a 16-entry table (two byte_perms). A
// thread a 32-bit word of q; one launch for the n matrices.
constexpr int kI4Max = 16;
struct I4Mats {
    const uint32_t *q[kI4Max];
};
__device__ __forceinline__ uint32_t i4_e4m3(uint32_t v, int nib) {
    // e4m3 of q = nibble - 8: -8 .. -1 (0xD0 0xCE 0xCC 0xCA 0xC8 0xC4 0xC0 0xB8), 0 .. 7 (0x00 0x38 0x40 0x44 0x48 0x4A 0x4C 0x4E)
    const uint32_t n = (v >> (4 * nib)) & 15u, sel = n & 7u;
    const uint32_t lo = __byte_perm(0xCACCCED0u, 0xB8C0C4C8u, sel), hi = __byte_perm(0x44403800u, 0x4E4C4A48u, sel);
    return (n & 8u ? hi : lo) & 0xffu;
}
// (matrix e: blockIdx.y. A thread takes four words, 16 bytes: one word a thread left too few bytes
// in flight, ~1.5 TB/s. In the fragment layout they are one lane's four MMA steps, 16 contiguous
// columns of each of its two rows; row-major, 32 columns of one row.)
__device__ __forceinline__ uint32_t i4_rowA(uint32_t v) { return i4_e4m3(v, 0) | i4_e4m3(v, 4) << 8 | i4_e4m3(v, 2) << 16 | i4_e4m3(v, 6) << 24; }
__device__ __forceinline__ uint32_t i4_rowB(uint32_t v) { return i4_e4m3(v, 1) | i4_e4m3(v, 5) << 8 | i4_e4m3(v, 3) << 16 | i4_e4m3(v, 7) << 24; }
__device__ __forceinline__ uint32_t i4_lo(uint32_t v) { return i4_e4m3(v, 0) | i4_e4m3(v, 2) << 8 | i4_e4m3(v, 4) << 16 | i4_e4m3(v, 6) << 24; }
__device__ __forceinline__ uint32_t i4_hi(uint32_t v) { return i4_e4m3(v, 1) | i4_e4m3(v, 3) << 8 | i4_e4m3(v, 5) << 16 | i4_e4m3(v, 7) << 24; }
__global__ void int4_to_e4m3_kernel(uint8_t *dst, int64_t step, int64_t ld, I4Mats src, int64_t rows, int64_t cols, int frag) {
    const uint32_t quads = static_cast<uint32_t>(rows * cols / 32), nch = static_cast<uint32_t>(cols / 64),
                   qpr = static_cast<uint32_t>(cols / 32);
    const uint4 *q = reinterpret_cast<const uint4 *>(src.q[blockIdx.y]);
    uint8_t *d = dst + static_cast<int64_t>(blockIdx.y) * step;
    for (uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < quads; c += gridDim.x * blockDim.x) {
        const uint4 v = __ldg(q + c);
        if (frag) {  // quad c: lane (g, t) of chunk ch of 16-row tile T, steps 0..3
            const uint32_t lane = c & 31, chl = c >> 5, g = lane >> 2, t = lane & 3, T = chl / nch, ch = chl - T * nch;
            const int64_t ra = 16 * int64_t(T) + 2 * g, k0 = 64 * int64_t(ch) + 16 * t;
            *reinterpret_cast<uint4 *>(d + ra * ld + k0) = make_uint4(i4_rowA(v.x), i4_rowA(v.y), i4_rowA(v.z), i4_rowA(v.w));
            *reinterpret_cast<uint4 *>(d + (ra + 1) * ld + k0) = make_uint4(i4_rowB(v.x), i4_rowB(v.y), i4_rowB(v.z), i4_rowB(v.w));
        } else {  // row-major: row c / (cols / 32), columns 32j .. 32j + 31 (a word's 8 at nibbles 0 2 4 6 1 3 5 7)
            const uint32_t r = c / qpr;
            uint8_t *p = d + int64_t(r) * ld + int64_t(c - r * qpr) * 32;
            *reinterpret_cast<uint4 *>(p) = make_uint4(i4_lo(v.x), i4_hi(v.x), i4_lo(v.y), i4_hi(v.y));
            *reinterpret_cast<uint4 *>(p + 16) = make_uint4(i4_lo(v.z), i4_hi(v.z), i4_lo(v.w), i4_hi(v.w));
        }
    }
}

// x [M][K] (fp32, or its bf16 twin) as e4m3 with a scale per row per 128 columns (amax / 448),
// scales [K / 128][M]: cuBLASLt's VEC128 layout. A warp a (row, group), four values a lane.
template <typename IN>
__global__ void act_quant_g128_kernel(uint8_t *q, float *scale, const IN *x, int64_t M, int64_t K) {
    const int64_t G = K / 128, items = M * G;
    const int lane = threadIdx.x & 31;
    for (int64_t it = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5; it < items;
         it += (static_cast<int64_t>(gridDim.x) * blockDim.x) >> 5) {
        const int64_t m = it / G, g = it % G, c = m * K + g * 128 + 4 * lane;
        float v[4];
        if constexpr (sizeof(IN) == 2) {
            const uint2 u = *reinterpret_cast<const uint2 *>(x + c);
            v[0] = __uint_as_float(u.x << 16), v[1] = __uint_as_float(u.x & 0xffff0000u);
            v[2] = __uint_as_float(u.y << 16), v[3] = __uint_as_float(u.y & 0xffff0000u);
        } else {
            const float4 f = *reinterpret_cast<const float4 *>(x + c);
            v[0] = f.x, v[1] = f.y, v[2] = f.z, v[3] = f.w;
        }
        float a = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
        a = warp_max(a);
        const float s = fmaxf(a, 1e-30f) / 448.0f, inv = 1.0f / s;
        if (lane == 0) scale[g * M + m] = s;
        *reinterpret_cast<uint32_t *>(q + c) = static_cast<uint32_t>(f2e4m3(v[0] * inv)) | static_cast<uint32_t>(f2e4m3(v[1] * inv)) << 8 |
                                               static_cast<uint32_t>(f2e4m3(v[2] * inv)) << 16 | static_cast<uint32_t>(f2e4m3(v[3] * inv)) << 24;
    }
}

// The GLU of interleaved bf16 gate / up pairs d [M][2J] (column 2j gate j, 2j + 1 up j) as e4m3
// [M][J] with a scale per row per 128 outputs (scales [J / 128][M]). A warp a (row, group).
__global__ void glu_quant_g128_kernel(uint8_t *q, float *scale, const uint16_t *d, int64_t M, int64_t J) {
    const int64_t G = J / 128, items = M * G;
    const int lane = threadIdx.x & 31;
    for (int64_t it = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5; it < items;
         it += (static_cast<int64_t>(gridDim.x) * blockDim.x) >> 5) {
        const int64_t m = it / G, g = it % G, j = g * 128 + 4 * lane;
        const uint4 u = *reinterpret_cast<const uint4 *>(d + m * 2 * J + 2 * j);  // 4 (gate, up) pairs
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
        float v[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const float gv = __uint_as_float(w[t] << 16), uv = __uint_as_float(w[t] & 0xffff0000u);
            v[t] = gv * sigm(gv) * uv;
        }
        float a = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
        a = warp_max(a);
        const float s = fmaxf(a, 1e-30f) / 448.0f, inv = 1.0f / s;
        if (lane == 0) scale[g * M + m] = s;
        *reinterpret_cast<uint32_t *>(q + m * J + j) = static_cast<uint32_t>(f2e4m3(v[0] * inv)) | static_cast<uint32_t>(f2e4m3(v[1] * inv)) << 8 |
                                                       static_cast<uint32_t>(f2e4m3(v[2] * inv)) << 16 | static_cast<uint32_t>(f2e4m3(v[3] * inv)) << 24;
    }
}

// silu_kernel over bf16 gate and up rows ld apart (a fused GEMM's [rows][ld] result, or two
// [rows][width] twins), into bf16 y [rows][width]: the same per-element arithmetic on the
// bf16-rounded inputs. VEC elements a thread (8: 16-byte accesses; width, ld multiples of 8).
template <int VEC>
__global__ void silu_b16_kernel(uint16_t *y, const uint16_t *g, const uint16_t *u, int64_t ld, int64_t width, int64_t n) {
    pdl_trigger();
    const int64_t wv = width / VEC;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n / VEC;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t r = i / wv, c = (i - r * wv) * VEC, src = r * ld + c;
        if constexpr (VEC == 8) {
            const uint4 gv = *reinterpret_cast<const uint4 *>(g + src), uv = *reinterpret_cast<const uint4 *>(u + src);
            auto two = [](uint32_t g2, uint32_t u2) {
                const float g0 = __uint_as_float(g2 << 16), g1 = __uint_as_float(g2 & 0xffff0000u);
                const float u0 = __uint_as_float(u2 << 16), u1 = __uint_as_float(u2 & 0xffff0000u);
                return static_cast<uint32_t>(f2bf(g0 * sigm(g0) * u0)) | (static_cast<uint32_t>(f2bf(g1 * sigm(g1) * u1)) << 16);
            };
            *reinterpret_cast<uint4 *>(y + i * VEC) = make_uint4(two(gv.x, uv.x), two(gv.y, uv.y), two(gv.z, uv.z), two(gv.w, uv.w));
        } else {
            const float gv = bf2f(g[src]);
            y[i] = f2bf(gv * sigm(gv) * bf2f(u[src]));
        }
    }
}

__global__ void add_kernel(float *y, const float *x, int64_t n) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        y[i] += x[i];
}

__global__ void sigmoid_mul_kernel(float *y, const float *g, int64_t n) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        y[i] *= sigm(g[i]);
}

__global__ void copy_rows_kernel(float *dst, int64_t dst_stride, int64_t dst_off, const float *src, int64_t rows,
                                 int64_t n) {
    const int64_t total = rows * n;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t r = i / n, j = i % n;
        dst[r * dst_stride + dst_off + j] = src[i];
    }
}


// One block per row: dst[r] = table[ids[r]] as fp32 (bf16 or f32 table), 4 elements per thread.
// dst[r] = table[idx[r]] (int32)
__global__ void gather_ids_kernel(int32_t *dst, const int32_t *__restrict__ table, const int32_t *__restrict__ idx,
                                  int64_t n) {
    const int64_t r = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r < n) dst[r] = table[idx[r]];
}

template <int WF>
__global__ void embed_rows_kernel(float *dst, const void *__restrict__ table, const int32_t *__restrict__ ids,
                                  int64_t n) {
    pdl_wait();  // the ids were just uploaded
    pdl_trigger();
    const int64_t r = blockIdx.x, id = ids[r];
    float *d = dst + r * n;
    for (int64_t i = static_cast<int64_t>(threadIdx.x) * 4; i < n; i += static_cast<int64_t>(blockDim.x) * 4) {
        float4 v;
        if constexpr (WF == 0) {
            v = *reinterpret_cast<const float4 *>(static_cast<const float *>(table) + id * n + i);
        } else {
            const uint2 u = *reinterpret_cast<const uint2 *>(static_cast<const uint16_t *>(table) + id * n + i);
            v = make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xFFFF0000u), __uint_as_float(u.y << 16),
                            __uint_as_float(u.y & 0xFFFF0000u));
        }
        *reinterpret_cast<float4 *>(d + i) = v;
    }
}

}  // namespace
}  // namespace fullm
