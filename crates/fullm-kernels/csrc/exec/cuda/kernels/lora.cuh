// SPDX-License-Identifier: Apache-2.0
// kernels/lora.cuh - LoRA adapters, many at once (Backend::lora): each row takes its own adapter
// (a slot per row, -1 none) up to its adapter's rank, for a group of targets that read one x
// (q, k, v; gate, up; o; down). Part of cuda_backend.cu's translation unit, after common.cuh.
//
// Two launches a group. lora_shrink_parts: a block takes four rows and a part of in, and every
// row of the group's A over that part, so x is read once and A once per four rows; its partial
// sums go to part[p][row][k]. lora_expand_parts: a block takes four rows and 256 outputs of one
// target, sums the parts in a fixed order (a row's result does not depend on its neighbours or on
// the split), and adds scale B xa to y where the backend keeps it: fp32, or a bf16 matrix (the
// fused q/k/v or gate/up result, a lazy twin) updated in place. Rows sharing an adapter read its A
// and B once; every load is 16 bytes. A long run of rows on one adapter outside a CUDA graph goes
// to cuBLAS instead (CudaBackend::lora_rows).
#pragma once

namespace fullm {
namespace {

constexpr int kLoraRows = 4;
constexpr int kLoraJobs = 3;

struct LoraShrinkJobs {
    const uint16_t *a[kLoraJobs];
    int64_t a_stride[kLoraJobs];
    const int32_t *ranks[kLoraJobs];
    int n;
};

struct LoraOut {
    float *y;        // fp32 rows, or
    uint16_t *y16;   // bf16 rows (row stride ld)
    int64_t ld, out;
    const uint16_t *b;  // [slots][out][ldb]
    int64_t ldb;
    const float *scale;
    const int32_t *ranks;
};

struct LoraExpandJobs {
    LoraOut t[kLoraJobs];
    int blk0[kLoraJobs + 1];  // job t's blocks: [blk0[t], blk0[t + 1])
    int n;
};

__device__ __forceinline__ void bf16x8(const uint4 w, float (&f)[8]) {
    const uint32_t u[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) f[2 * e] = __uint_as_float(u[e] << 16), f[2 * e + 1] = __uint_as_float(u[e] & 0xffff0000u);
}
__device__ __forceinline__ void ld8(const float *p, float (&v)[8]) {
    const float4 a = *reinterpret_cast<const float4 *>(p), b = *reinterpret_cast<const float4 *>(p + 4);
    v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
}
__device__ __forceinline__ void ld8(const uint16_t *p, float (&v)[8]) { bf16x8(*reinterpret_cast<const uint4 *>(p), v); }
__device__ __forceinline__ float dot8(const float (&w)[8], const float (&v)[8]) {
    float s = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) s = fmaf(w[e], v[e], s);
    return s;
}

// the rows' common slot among those that have one: -1 none has one, -2 they differ
__device__ __forceinline__ int lora_common(const int (&s)[kLoraRows]) {
    int common = -1;
#pragma unroll
    for (int j = 0; j < kLoraRows; ++j)
        if (s[j] >= 0) common = common == -1 || common == s[j] ? s[j] : -2;
    return common;
}

// grid (P, ceil(N / kLoraRows), KS), 256 threads. Block (p, g, z): rows [4g, 4g + 4), x's
// 256-wide chunks [p C, p C + C), rows [z Kc, z Kc + Kc) of the jobs' stacked A (job k / rank,
// its row k % rank), warp w taking every 8th from the w-th, skipping rows past an adapter's rank. in % 256 == 0, x and every A 16-byte
// aligned (the host checks). XT: float, or uint16_t for x's bf16 twin.
template <typename XT>
__global__ void __launch_bounds__(256) lora_shrink_parts(float *part, const XT *x, int64_t N, int64_t in, int rank,
                                                         LoraShrinkJobs J, const int32_t *slot, int C, int Kc) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t r0 = static_cast<int64_t>(blockIdx.y) * kLoraRows;
    const int64_t i0 = static_cast<int64_t>(blockIdx.x) * C * 256, i1 = min(in, i0 + static_cast<int64_t>(C) * 256);
    const int Kt = J.n * rank;
    int s0[kLoraRows];
#pragma unroll
    for (int j = 0; j < kLoraRows; ++j) s0[j] = r0 + j < N ? slot[r0 + j] : -1;
    const int c0 = lora_common(s0);
    if (c0 == -1) return;
    const int kb = static_cast<int>(blockIdx.z) * Kc, k1 = min(Kt, kb + Kc);
    if (c0 >= 0) {  // the rows share an adapter: four of its A rows at a time, x read once for them
        for (int k0 = kb + warp; k0 < k1; k0 += 32) {
            const uint16_t *ap[4];
            bool on[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int k = k0 + 8 * q, t = k / rank, kk = k % rank;
                on[q] = k < k1 && kk < J.ranks[t][c0];
                ap[q] = on[q] ? J.a[t] + c0 * J.a_stride[t] + static_cast<int64_t>(kk) * in : nullptr;
            }
            if (!(on[0] || on[1] || on[2] || on[3])) continue;  // the same for the whole warp
            float acc[4][kLoraRows] = {};
#pragma unroll 2
            for (int64_t i = i0 + lane * 8; i < i1; i += 256) {
                float w[4][8] = {};
#pragma unroll
                for (int q = 0; q < 4; ++q)
                    if (on[q]) bf16x8(*reinterpret_cast<const uint4 *>(ap[q] + i), w[q]);
#pragma unroll
                for (int j = 0; j < kLoraRows; ++j)
                    if (s0[j] >= 0) {
                        float v[8];
                        ld8(x + (r0 + j) * in + i, v);
#pragma unroll
                        for (int q = 0; q < 4; ++q) acc[q][j] += dot8(w[q], v);
                    }
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                if (!on[q]) continue;
#pragma unroll
                for (int j = 0; j < kLoraRows; ++j) acc[q][j] = warp_sum(acc[q][j]);
                if (lane == 0)
#pragma unroll
                    for (int j = 0; j < kLoraRows; ++j)
                        if (s0[j] >= 0) part[(static_cast<int64_t>(blockIdx.x) * N + r0 + j) * Kt + k0 + 8 * q] = acc[q][j];
            }
        }
        return;
    }
    for (int k = kb + warp; k < k1; k += 8) {  // rows on different adapters: a row of A at a time
        const int t = k / rank, kk = k % rank;
        int s[kLoraRows];
#pragma unroll
        for (int j = 0; j < kLoraRows; ++j) s[j] = s0[j] >= 0 && kk < J.ranks[t][s0[j]] ? s0[j] : -1;
        const int common = lora_common(s);
        if (common == -1) continue;  // the same for the whole warp
        float acc[kLoraRows] = {};
        if (common >= 0) {  // one A row for every row
            const uint16_t *ak = J.a[t] + common * J.a_stride[t] + static_cast<int64_t>(kk) * in;
#pragma unroll 2
            for (int64_t i = i0 + lane * 8; i < i1; i += 256) {
                float w[8];
                bf16x8(*reinterpret_cast<const uint4 *>(ak + i), w);
#pragma unroll
                for (int j = 0; j < kLoraRows; ++j)
                    if (s[j] >= 0) {
                        float v[8];
                        ld8(x + (r0 + j) * in + i, v);
                        acc[j] += dot8(w, v);
                    }
            }
        } else {
#pragma unroll
            for (int j = 0; j < kLoraRows; ++j) {
                if (s[j] < 0) continue;
                const uint16_t *ak = J.a[t] + s[j] * J.a_stride[t] + static_cast<int64_t>(kk) * in;
#pragma unroll 2
                for (int64_t i = i0 + lane * 8; i < i1; i += 256) {
                    float w[8], v[8];
                    bf16x8(*reinterpret_cast<const uint4 *>(ak + i), w);
                    ld8(x + (r0 + j) * in + i, v);
                    acc[j] += dot8(w, v);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < kLoraRows; ++j) acc[j] = warp_sum(acc[j]);
        if (lane == 0)
#pragma unroll
            for (int j = 0; j < kLoraRows; ++j)
                if (s[j] >= 0) part[(static_cast<int64_t>(blockIdx.x) * N + r0 + j) * Kt + k] = acc[j];
    }
}

// grid (blk0[n], ceil(N / kLoraRows)), 256 threads: block x of job t adds its 256 outputs for
// the block's rows, their xa (the P parts summed, zero past each row's rank) in shared memory.
// rank % 8 == 0, ldb % 8 == 0, every B 16-byte aligned (the host checks).
__global__ void __launch_bounds__(256) lora_expand_parts(const float *part, int P, int64_t N, int rank,
                                                         LoraExpandJobs E, const int32_t *slot) {
    extern __shared__ float sxa[];  // [kLoraRows][rank]
    int t = 0;
    while (t + 1 < E.n && static_cast<int>(blockIdx.x) >= E.blk0[t + 1]) ++t;
    const LoraOut &J = E.t[t];
    const int Kt = E.n * rank;
    const int64_t r0 = static_cast<int64_t>(blockIdx.y) * kLoraRows;
    int s[kLoraRows];
#pragma unroll
    for (int j = 0; j < kLoraRows; ++j) s[j] = r0 + j < N ? slot[r0 + j] : -1;
    const int common = lora_common(s);
    if (common == -1) return;  // the block's rows have no adapter (the same for every thread)
    for (int i = threadIdx.x; i < kLoraRows * rank; i += blockDim.x) {
        const int j = i / rank, k = i % rank;
        float v = 0.0f;
        if (s[j] >= 0 && k < J.ranks[s[j]])
#pragma unroll 8
            for (int p = 0; p < P; ++p) v += part[(static_cast<int64_t>(p) * N + r0 + j) * Kt + t * rank + k];
        sxa[i] = v;
    }
    __syncthreads();
    const int64_t o = static_cast<int64_t>(blockIdx.x - E.blk0[t]) * blockDim.x + threadIdx.x;
    if (o >= J.out) return;
    float acc[kLoraRows] = {};
    if (common >= 0) {
        const uint16_t *bo = J.b + (static_cast<int64_t>(common) * J.out + o) * J.ldb;
        const int kr = J.ranks[common];  // B's columns past it are zero, as are those of sxa
#pragma unroll 4
        for (int k = 0; k < kr; k += 8) {
            float w[8];
            bf16x8(*reinterpret_cast<const uint4 *>(bo + k), w);
#pragma unroll
            for (int j = 0; j < kLoraRows; ++j)
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[j] = fmaf(w[e], sxa[j * rank + k + e], acc[j]);
        }
    } else {
#pragma unroll
        for (int j = 0; j < kLoraRows; ++j) {
            if (s[j] < 0) continue;
            const uint16_t *bo = J.b + (static_cast<int64_t>(s[j]) * J.out + o) * J.ldb;
            const int kr = J.ranks[s[j]];
#pragma unroll 4
            for (int k = 0; k < kr; k += 8) {
                float w[8];
                bf16x8(*reinterpret_cast<const uint4 *>(bo + k), w);
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[j] = fmaf(w[e], sxa[j * rank + k + e], acc[j]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < kLoraRows; ++j) {
        if (s[j] < 0) continue;
        const float d = J.scale[s[j]] * acc[j];
        if (J.y16) {
            uint16_t *p = J.y16 + (r0 + j) * J.ld + o;
            *p = f2bf(bf2f(*p) + d);
        } else {
            J.y[(r0 + j) * J.ld + o] += d;
        }
    }
}

// any shape (one job, fp32 x and y): one block a row, its warps over A's rows
__global__ void lora_shrink_kernel(float *xa, int64_t ldxa, const float *x, int64_t in, const uint16_t *a,
                                   int64_t a_stride, const int32_t *ranks, int rank, const int32_t *slot) {
    const int64_t row = blockIdx.x;
    const int s = slot[row];
    if (s < 0) return;
    const float *xr = x + row * in;
    const uint16_t *A = a + s * a_stride;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5, kr = min(ranks[s], rank);
    for (int k = warp; k < kr; k += nw) {
        const uint16_t *ak = A + static_cast<int64_t>(k) * in;
        float acc = 0.0f;
        for (int64_t i = lane; i < in; i += 32) acc = fmaf(bf2f(ak[i]), xr[i], acc);
        acc = warp_sum(acc);
        if (lane == 0) xa[row * ldxa + k] = acc;
    }
}

// any shape (one job, fp32 y): grid (output chunks, rows), a thread an output of a row
__global__ void lora_expand_kernel(float *y, const float *xa, int64_t ldxa, int64_t out, const uint16_t *b, int64_t ldb,
                                   int rank, const int32_t *ranks, const float *scale, const int32_t *slot) {
    const int64_t row = blockIdx.y;
    const int s = slot[row];
    if (s < 0) return;
    const int64_t o = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (o >= out) return;
    const uint16_t *B = b + (static_cast<int64_t>(s) * out + o) * ldb;
    const float *xr = xa + row * ldxa;
    const int kr = min(ranks[s], rank);
    float acc = 0.0f;
    for (int k = 0; k < kr; ++k) acc = fmaf(bf2f(B[k]), xr[k], acc);
    y[row * out + o] += scale[s] * acc;
}

}  // namespace
}  // namespace fullm
