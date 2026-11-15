// SPDX-License-Identifier: Apache-2.0
// kernels/attention.cuh - dense (GQA) attention: FlashAttention-3's row metadata, RoPE, the decode
// kernels (scalar and tensor-core, bf16 and fp8 caches), the prefill kernel, the fp8 cache's
// staging, the fused q/k RoPE + append.
//
// Part of cuda_backend.cu's translation unit, included there after the CUDA and standard
// headers, in the order common, formats, gemv, convert, router, elementwise, copy, kimi, mla,
// attention, sampling, moe (each file uses only those before it). Kernels and their device
// helpers: no state.
#pragma once

namespace fullm {
namespace {

// FlashAttention-3's per-sequence metadata, [cu_seqlens_q n + 1][seqused_k n][slot n], for the
// rows [0, n_rows) of a step taken as one-query sequences (the decoding rows): cu_seqlens_q[i] =
// i, seqused_k = the row's position + 1, its slot; from the step's device arrays, so a captured
// graph recomputes it at every replay. The chunks' entries (rows beyond) come from the host.
__global__ void fa3_rows_meta_kernel(int32_t *meta, const int32_t *__restrict__ tok_pos,
                                     const int32_t *__restrict__ tok_slot, int n_rows, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n_rows) meta[i] = i;
    if (i < n_rows) {
        meta[n + 1 + i] = tok_pos[i] + 1;
        meta[2 * n + 1 + i] = tok_slot[i];
    }
}


// ------------------------------------------------------------------ dense attention (Llama / Qwen)

// grid (token), threads over heads x D/2: rotate-half RoPE in place (angle in fp32).
__global__ void rope_kernel(float *x, const int32_t *__restrict__ tok_pos, const float *__restrict__ inv_freq,
                            int64_t heads, int64_t D, int64_t stride) {
    const int64_t t = blockIdx.x, half = D / 2;
    const float pos = static_cast<float>(tok_pos[t]);
    for (int64_t i = threadIdx.x; i < heads * half; i += blockDim.x) {
        const int64_t h = i / half, j = i % half;
        const float ang = pos * inv_freq[j];
        const float c = cosf(ang), sn = sinf(ang);
        float *v = x + t * stride + h * D;
        const float a = v[j], b = v[j + half];
        v[j] = a * c - b * sn;
        v[j + half] = b * c + a * sn;
    }
}

// Flash-decoding over a paged K/V cache, grouped-query: grid (token, kv head, split), one warp
// per query head of the group (they share each staged K/V tile). Splits follow attn_split
// (a function of the token's own position). With cnt, the last split block of a (token, kv
// head) to finish merges the partials (mla_combine_kernel's arithmetic, in split order) and
// resets its counter; without it, mla_combine_kernel merges them. Either way a token's result
// never depends on the rest of a small batch. DR = D / 32 dims per lane.
template <typename KT, int DR>
__global__ void gqa_attn_kernel(float *out, float *part, unsigned *cnt, const float *q, const KT *__restrict__ kc,
                                const KT *__restrict__ vc, const int32_t *__restrict__ tok_slot,
                                const int32_t *__restrict__ tok_pos, const int32_t *__restrict__ pt,
                                int64_t max_pages, int64_t page, int Hq, int Hkv, float scale, int max_splits) {
    constexpr int D = 32 * DR, TILE = DR <= 4 ? 32 : 16;
    __shared__ float tk[TILE][D], tv[TILE][D];
    __shared__ float fs[32][kAttnMaxSplits];
    __shared__ bool last;
    pdl_wait();
    pdl_trigger();
    const int64_t t = blockIdx.x;
    const int kh = blockIdx.y, sp = blockIdx.z, G = Hq / Hkv;
    const int64_t pos = tok_pos[t];
    int64_t S;
    int nsplit;
    attn_split(pos + 1, max_splits, S, nsplit);
    if (sp >= nsplit) return;  // block-uniform
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = kh * G + warp;
    const int32_t *ptab = pt + static_cast<int64_t>(tok_slot[t]) * max_pages;
    const int64_t start = static_cast<int64_t>(sp) * S, end = imin(start + S, pos + 1);
    const int64_t W = static_cast<int64_t>(Hkv) * D;
    float qr[DR], acc[DR];
#pragma unroll
    for (int c = 0; c < DR; ++c) qr[c] = q[(t * Hq + h) * D + lane + 32 * c] * scale, acc[c] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (int64_t p0 = start; p0 < end; p0 += TILE) {
        const int np = static_cast<int>(imin(TILE, end - p0));
        __syncthreads();
        // stage the tile: each warp its positions warp, warp + G, ..., four at a time so their
        // page lookups, then their loads, are in flight together
        for (int b = warp; b < np; b += 4 * G) {
            int64_t row[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int pp = b + j * G;
                const int64_t p = p0 + pp;
                row[j] = pp < np ? (static_cast<int64_t>(ptab[p / page]) * page + p % page) * W + kh * D : -1;
            }
            float kr[4][DR], vr[4][DR];
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int c = 0; c < DR; ++c) {
                    kr[j][c] = row[j] >= 0 ? ldst(kc + row[j] + lane + 32 * c) : 0.0f;
                    vr[j][c] = row[j] >= 0 ? ldst(vc + row[j] + lane + 32 * c) : 0.0f;
                }
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if (row[j] >= 0)
#pragma unroll
                    for (int c = 0; c < DR; ++c) tk[b + j * G][lane + 32 * c] = kr[j][c], tv[b + j * G][lane + 32 * c] = vr[j][c];
        }
        __syncthreads();
        for (int g = 0; g < np; g += 4) {
            const int nv = min(4, np - g);  // rows past np hold stale data: computed, never used
            float d[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float a = 0.0f;
#pragma unroll
                for (int c = 0; c < DR; ++c) a = fmaf(qr[c], tk[g + i][lane + 32 * c], a);
                d[i] = a;
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)
#pragma unroll
                for (int i = 0; i < 4; ++i) d[i] += __shfl_xor_sync(0xffffffffu, d[i], o);
            float mn = m;
#pragma unroll
            for (int i = 0; i < 4; ++i)
                if (i < nv) mn = fmaxf(mn, d[i]);
            const float corr = expf(m - mn);
            float pe[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) pe[i] = i < nv ? expf(d[i] - mn) : 0.0f;
            l = l * corr + pe[0] + pe[1] + pe[2] + pe[3];
#pragma unroll
            for (int c = 0; c < DR; ++c) {
                float o = acc[c] * corr;
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    if (i < nv) o = fmaf(pe[i], tv[g + i][lane + 32 * c], o);
                acc[c] = o;
            }
            m = mn;
        }
    }
    float *orow = out + (t * Hq + h) * D;
    if (nsplit == 1) {
        const float inv = 1.0f / l;
#pragma unroll
        for (int c = 0; c < DR; ++c) orow[lane + 32 * c] = acc[c] * inv;
        return;
    }
    constexpr int64_t sl = D + 2;
    const float *base = part + (t * Hq + h) * gridDim.z * sl;
    float *pp = part + ((t * Hq + h) * gridDim.z + sp) * sl;
#pragma unroll
    for (int c = 0; c < DR; ++c) pp[lane + 32 * c] = acc[c];
    if (lane == 0) pp[D] = m, pp[D + 1] = l;
    if (!cnt) return;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned *ct = cnt + t * Hkv + kh;
        last = atomicAdd(ct, 1u) == static_cast<unsigned>(nsplit - 1);
        if (last) *ct = 0;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    // lanes take the splits' (m, l) in parallel; every sum has a fixed order for a given nsplit
    float *f = fs[warp];
    float ms[2], ls[2], M = -INFINITY;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int s2 = lane + 32 * j;
        ms[j] = s2 < nsplit ? __ldcg(base + s2 * sl + D) : -INFINITY;
        ls[j] = s2 < nsplit ? __ldcg(base + s2 * sl + D + 1) : 0.0f;
        M = fmaxf(M, ms[j]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
    float lp = 0.0f;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int s2 = lane + 32 * j;
        if (s2 < nsplit) {
            const float e = expf(ms[j] - M);
            f[s2] = e;
            lp = fmaf(ls[j], e, lp);
        }
    }
    const float inv = 1.0f / warp_sum(lp);
    __syncwarp();
    float a[DR];
#pragma unroll
    for (int c = 0; c < DR; ++c) a[c] = 0.0f;
#pragma unroll 4
    for (int s2 = 0; s2 < nsplit; ++s2) {
        const float e = f[s2];
#pragma unroll
        for (int c = 0; c < DR; ++c) a[c] = fmaf(__ldcg(base + s2 * sl + lane + 32 * c), e, a[c]);
    }
#pragma unroll
    for (int c = 0; c < DR; ++c) orow[lane + 32 * c] = a[c] * inv;
}

// Prefill attention on the tensor cores (bf16 activations): grid (query tile, kv head), one warp
// per query head of the group, each warp's 16 rows the tile's tokens (<= 16 consecutive
// positions of one sequence). K/V tiles of kPfKv positions are staged in shared memory as bf16
// and shared by the group; S = Q K^T and O += P V are m16n8k16 MMAs (Q and P rounded to bf16,
// as FlashAttention does), with an online softmax in fp32 and the causal mask per row.
// gqa_prefill_kernel (bf16 caches): the tiles stream through a two-stage cp.async ring (the
// next tile lands while this one is computed) and the K fragments come by ldmatrix.
// gqa_prefill_f32_kernel (fp32 caches): plain loads, rows rounded to bf16 as they are staged.
constexpr int kPfKv = 64, kPfStages = 2;
template <int D>
constexpr int pf_stage_elems() {
    return 2 * kPfKv * (D + 8);  // K rows then V rows, padded to a conflict-free pitch
}
template <int D>
constexpr size_t pf_smem() {
    return sizeof(uint16_t) * kPfStages * pf_stage_elems<D>();
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void *p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}
__device__ __forceinline__ void cp_async16_zfill(void *smem, const void *gmem, bool valid) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const int n = valid ? 16 : 0;  // 0: the 16 bytes are zero-filled, nothing is read
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(a), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void ldsm_x4_trans(uint32_t (&r)[4], const void *p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

template <int D>
__global__ void __launch_bounds__(256)
    gqa_prefill_kernel(float *out, const float *q, const __nv_bfloat16 *__restrict__ kc,
                       const __nv_bfloat16 *__restrict__ vc, const int32_t *__restrict__ tiles,
                       const int32_t *__restrict__ tok_slot, const int32_t *__restrict__ tok_pos,
                       const int32_t *__restrict__ pt, int64_t max_pages, int64_t page, int Hq, int Hkv,
                       float scale_log2) {
    constexpr int LD = D + 8, SE = pf_stage_elems<D>();
    extern __shared__ __align__(16) uint16_t pf_smem_[];
    pdl_wait();
    pdl_trigger();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int G = Hq / Hkv, kh = blockIdx.y, h = kh * G + warp;
    const int64_t first = tiles[2 * blockIdx.x];
    const int count = tiles[2 * blockIdx.x + 1];
    const int64_t pos0 = tok_pos[first], last = pos0 + count - 1;
    const int32_t *ptab = pt + static_cast<int64_t>(tok_slot[first]) * max_pages;
    const int64_t W = static_cast<int64_t>(Hkv) * D;

    // Q as A fragments: rows g and g + 8 of the tile, k-step kk
    uint32_t qa[D / 16][4];
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk) {
        auto at = [&](int r, int c) -> float2 {
            if (r >= count) return make_float2(0.f, 0.f);
            return *reinterpret_cast<const float2 *>(q + ((first + r) * Hq + h) * D + kk * 16 + c);
        };
        const float2 x0 = at(g, 2 * t), x1 = at(g + 8, 2 * t), x2 = at(g, 2 * t + 8), x3 = at(g + 8, 2 * t + 8);
        qa[kk][0] = bf16x2(x0.x, x0.y), qa[kk][1] = bf16x2(x1.x, x1.y);
        qa[kk][2] = bf16x2(x2.x, x2.y), qa[kk][3] = bf16x2(x3.x, x3.y);
    }
    float o[D / 8][4];
#pragma unroll
    for (int j = 0; j < D / 8; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;  // rows g, g + 8
    const int64_t qp0 = pos0 + g, qp1 = pos0 + g + 8;

    // K/V tiles 0 .. last / kPfKv: the block's threads stride over the tile's 16-byte chunks
    // (D / 8 per row), K and V alike; rows past `last` are zero-filled
    const int ntile = static_cast<int>(last / kPfKv) + 1;
    constexpr int CPR = D / 8;
    auto issue = [&](int i) {
        uint16_t *ks = pf_smem_ + (i % kPfStages) * SE, *vs = ks + kPfKv * LD;
        const int64_t kv0 = static_cast<int64_t>(i) * kPfKv;
        for (int c = threadIdx.x; c < kPfKv * CPR; c += blockDim.x) {
            const int r = c / CPR, cc = c % CPR;
            const int64_t p = kv0 + r;
            const bool valid = p <= last;
            const int64_t row = valid ? (static_cast<int64_t>(ptab[p / page]) * page + p % page) * W + kh * D : 0;
            cp_async16_zfill(ks + r * LD + cc * 8, kc + row + cc * 8, valid);
            cp_async16_zfill(vs + r * LD + cc * 8, vc + row + cc * 8, valid);
        }
    };
    issue(0);
    cp_async_commit();
    for (int i = 0; i < ntile; ++i) {
        cp_async_wait<0>();  // tile i has landed (this thread's copies)
        __syncthreads();     // everyone's; and tile i - 1 is no longer read
        if (i + 1 < ntile) issue(i + 1);
        cp_async_commit();
        const uint16_t *ks = pf_smem_ + (i % kPfStages) * SE, *vs = ks + kPfKv * LD;
        const int64_t kv0 = static_cast<int64_t>(i) * kPfKv;
        // S = Q K^T: 16 rows x kPfKv positions; K fragments by ldmatrix, two n-tiles per load
        float sc[kPfKv / 8][4];
#pragma unroll
        for (int j = 0; j < kPfKv / 8; ++j) sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.0f;
#pragma unroll
        for (int kk = 0; kk < D / 16; ++kk)
#pragma unroll
            for (int jp = 0; jp < kPfKv / 16; ++jp) {
                uint32_t bb[4];
                ldsm_x4(bb, ks + (jp * 16 + (lane & 7) + ((lane >> 4) << 3)) * LD + kk * 16 + ((lane >> 3) & 1) * 8);
                mma_bf16(sc[2 * jp], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], bb[0], bb[1]);
                mma_bf16(sc[2 * jp + 1], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], bb[2], bb[3]);
            }
        // causal mask and online softmax (base 2), rows g (c0, c1) and g + 8 (c2, c3)
        float mx0 = m0, mx1 = m1;
#pragma unroll
        for (int j = 0; j < kPfKv / 8; ++j) {
            const int64_t kp = kv0 + j * 8 + 2 * t;
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                sc[j][e] = kp + e <= qp0 ? sc[j][e] * scale_log2 : -INFINITY;
                sc[j][2 + e] = kp + e <= qp1 ? sc[j][2 + e] * scale_log2 : -INFINITY;
                mx0 = fmaxf(mx0, sc[j][e]), mx1 = fmaxf(mx1, sc[j][2 + e]);
            }
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
        }
        const float b0m = mx0 == -INFINITY ? 0.0f : mx0, b1m = mx1 == -INFINITY ? 0.0f : mx1;
        const float c0 = exp2f(m0 - b0m), c1 = exp2f(m1 - b1m);
        m0 = mx0, m1 = mx1;
        float s0 = 0.0f, s1 = 0.0f;
        uint32_t pa[kPfKv / 16][4];
#pragma unroll
        for (int j = 0; j < kPfKv / 8; ++j) {
            const float p0 = exp2f(sc[j][0] - b0m), p1 = exp2f(sc[j][1] - b0m);
            const float p2 = exp2f(sc[j][2] - b1m), p3 = exp2f(sc[j][3] - b1m);
            s0 += p0 + p1, s1 += p2 + p3;
            pa[j / 2][(j & 1) * 2] = bf16x2(p0, p1);  // C of n-tiles 2kk, 2kk + 1 is A of k-step kk
            pa[j / 2][(j & 1) * 2 + 1] = bf16x2(p2, p3);
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            s0 += __shfl_xor_sync(0xffffffffu, s0, off);
            s1 += __shfl_xor_sync(0xffffffffu, s1, off);
        }
        l0 = l0 * c0 + s0, l1 = l1 * c1 + s1;
#pragma unroll
        for (int j = 0; j < D / 8; ++j) o[j][0] *= c0, o[j][1] *= c0, o[j][2] *= c1, o[j][3] *= c1;
        // O += P V: V fragments by ldmatrix.trans, two n-tiles (16 dims) per load
#pragma unroll
        for (int kk = 0; kk < kPfKv / 16; ++kk)
#pragma unroll
            for (int dj = 0; dj < D / 8; dj += 2) {
                uint32_t vb[4];
                ldsm_x4_trans(vb, vs + (kk * 16 + (lane & 15)) * LD + (dj + (lane >> 4)) * 8);
                mma_bf16(o[dj], pa[kk][0], pa[kk][1], pa[kk][2], pa[kk][3], vb[0], vb[1]);
                mma_bf16(o[dj + 1], pa[kk][0], pa[kk][1], pa[kk][2], pa[kk][3], vb[2], vb[3]);
            }
    }
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < D / 8; ++j) {
        if (g < count)
            *reinterpret_cast<float2 *>(out + ((first + g) * Hq + h) * D + j * 8 + 2 * t) =
                make_float2(o[j][0] * i0, o[j][1] * i0);
        if (g + 8 < count)
            *reinterpret_cast<float2 *>(out + ((first + g + 8) * Hq + h) * D + j * 8 + 2 * t) =
                make_float2(o[j][2] * i1, o[j][3] * i1);
    }
}

template <typename KT, int D>
__global__ void __launch_bounds__(256)
    gqa_prefill_f32_kernel(float *out, const float *q, const KT *__restrict__ kc, const KT *__restrict__ vc,
                       const int32_t *__restrict__ tiles, const int32_t *__restrict__ tok_slot,
                       const int32_t *__restrict__ tok_pos, const int32_t *__restrict__ pt, int64_t max_pages,
                       int64_t page, int Hq, int Hkv, float scale_log2) {
    constexpr int LD = D + 8;  // padded rows: conflict-free fragment reads
    __shared__ __align__(16) uint16_t ks[kPfKv][LD], vs[kPfKv][LD];
    pdl_wait();
    pdl_trigger();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int G = Hq / Hkv, kh = blockIdx.y, h = kh * G + warp;
    const int64_t first = tiles[2 * blockIdx.x];
    const int count = tiles[2 * blockIdx.x + 1];
    const int64_t pos0 = tok_pos[first], last = pos0 + count - 1;
    const int32_t *ptab = pt + static_cast<int64_t>(tok_slot[first]) * max_pages;
    const int64_t W = static_cast<int64_t>(Hkv) * D;

    // Q as A fragments: rows g and g + 8 of the tile, k-step ks
    uint32_t qa[D / 16][4];
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk) {
        auto at = [&](int r, int c) -> float2 {
            if (r >= count) return make_float2(0.f, 0.f);
            return *reinterpret_cast<const float2 *>(q + ((first + r) * Hq + h) * D + kk * 16 + c);
        };
        const float2 x0 = at(g, 2 * t), x1 = at(g + 8, 2 * t), x2 = at(g, 2 * t + 8), x3 = at(g + 8, 2 * t + 8);
        qa[kk][0] = bf16x2(x0.x, x0.y), qa[kk][1] = bf16x2(x1.x, x1.y);
        qa[kk][2] = bf16x2(x2.x, x2.y), qa[kk][3] = bf16x2(x3.x, x3.y);
    }
    float o[D / 8][4];
#pragma unroll
    for (int j = 0; j < D / 8; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;  // rows g, g + 8
    const int64_t qp0 = pos0 + g, qp1 = pos0 + g + 8;

    for (int64_t kv0 = 0; kv0 <= last; kv0 += kPfKv) {
        __syncthreads();  // the previous tile is no longer read
        // stage K and V rows kv0 .. kv0 + kPfKv - 1 (rows past `last` are zero)
        constexpr int CH = D / 8;  // 16-byte bf16 chunks per row
        for (int c = threadIdx.x; c < kPfKv * CH; c += blockDim.x) {
            const int r = c / CH, cc = c % CH;
            const int64_t p = kv0 + r;
            uint4 kv[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
            if (p <= last) {
                const int64_t row = (static_cast<int64_t>(ptab[p / page]) * page + p % page) * W + kh * D + cc * 8;
                if constexpr (std::is_same_v<KT, float>) {
                    const float4 *k4 = reinterpret_cast<const float4 *>(kc + row), *v4 = reinterpret_cast<const float4 *>(vc + row);
                    const float4 a = k4[0], b = k4[1], e = v4[0], f = v4[1];
                    kv[0] = make_uint4(bf16x2(a.x, a.y), bf16x2(a.z, a.w), bf16x2(b.x, b.y), bf16x2(b.z, b.w));
                    kv[1] = make_uint4(bf16x2(e.x, e.y), bf16x2(e.z, e.w), bf16x2(f.x, f.y), bf16x2(f.z, f.w));
                } else {
                    kv[0] = *reinterpret_cast<const uint4 *>(kc + row);
                    kv[1] = *reinterpret_cast<const uint4 *>(vc + row);
                }
            }
            *reinterpret_cast<uint4 *>(&ks[r][cc * 8]) = kv[0];
            *reinterpret_cast<uint4 *>(&vs[r][cc * 8]) = kv[1];
        }
        __syncthreads();
        // S = Q K^T: 16 rows x kPfKv positions (n-tiles of 8)
        float sc[kPfKv / 8][4];
#pragma unroll
        for (int j = 0; j < kPfKv / 8; ++j) {
            sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.0f;
#pragma unroll
            for (int kk = 0; kk < D / 16; ++kk) {
                const uint32_t b0 = *reinterpret_cast<const uint32_t *>(&ks[j * 8 + g][kk * 16 + 2 * t]);
                const uint32_t b1 = *reinterpret_cast<const uint32_t *>(&ks[j * 8 + g][kk * 16 + 2 * t + 8]);
                mma_bf16(sc[j], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b0, b1);
            }
        }
        // causal mask and online softmax (base 2), rows g (c0, c1) and g + 8 (c2, c3)
        float mx0 = m0, mx1 = m1;
#pragma unroll
        for (int j = 0; j < kPfKv / 8; ++j) {
            const int64_t kp = kv0 + j * 8 + 2 * t;
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                sc[j][e] = kp + e <= qp0 ? sc[j][e] * scale_log2 : -INFINITY;
                sc[j][2 + e] = kp + e <= qp1 ? sc[j][2 + e] * scale_log2 : -INFINITY;
                mx0 = fmaxf(mx0, sc[j][e]), mx1 = fmaxf(mx1, sc[j][2 + e]);
            }
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
        }
        const float b0m = mx0 == -INFINITY ? 0.0f : mx0, b1m = mx1 == -INFINITY ? 0.0f : mx1;
        const float c0 = exp2f(m0 - b0m), c1 = exp2f(m1 - b1m);
        m0 = mx0, m1 = mx1;
        float s0 = 0.0f, s1 = 0.0f;
        uint32_t pa[kPfKv / 16][4];
#pragma unroll
        for (int j = 0; j < kPfKv / 8; ++j) {
            const float p0 = exp2f(sc[j][0] - b0m), p1 = exp2f(sc[j][1] - b0m);
            const float p2 = exp2f(sc[j][2] - b1m), p3 = exp2f(sc[j][3] - b1m);
            s0 += p0 + p1, s1 += p2 + p3;
            // C fragments of n-tiles 2kk, 2kk + 1 are the A fragment of k-step kk
            pa[j / 2][(j & 1) * 2] = bf16x2(p0, p1);
            pa[j / 2][(j & 1) * 2 + 1] = bf16x2(p2, p3);
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            s0 += __shfl_xor_sync(0xffffffffu, s0, off);
            s1 += __shfl_xor_sync(0xffffffffu, s1, off);
        }
        l0 = l0 * c0 + s0, l1 = l1 * c1 + s1;
#pragma unroll
        for (int j = 0; j < D / 8; ++j) o[j][0] *= c0, o[j][1] *= c0, o[j][2] *= c1, o[j][3] *= c1;
        // O += P V: V fragments by ldmatrix.trans, two n-tiles (16 dims) per load
#pragma unroll
        for (int kk = 0; kk < kPfKv / 16; ++kk) {
#pragma unroll
            for (int dj = 0; dj < D / 8; dj += 2) {
                uint32_t vb[4];
                const int r = kk * 16 + (lane & 15), cdim = (dj + (lane >> 4)) * 8;
                ldsm_x4_trans(vb, &vs[r][cdim]);
                mma_bf16(o[dj], pa[kk][0], pa[kk][1], pa[kk][2], pa[kk][3], vb[0], vb[1]);
                mma_bf16(o[dj + 1], pa[kk][0], pa[kk][1], pa[kk][2], pa[kk][3], vb[2], vb[3]);
            }
        }
    }
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < D / 8; ++j) {
        if (g < count)
            *reinterpret_cast<float2 *>(out + ((first + g) * Hq + h) * D + j * 8 + 2 * t) =
                make_float2(o[j][0] * i0, o[j][1] * i0);
        if (g + 8 < count)
            *reinterpret_cast<float2 *>(out + ((first + g + 8) * Hq + h) * D + j * 8 + 2 * t) =
                make_float2(o[j][2] * i1, o[j][3] * i1);
    }
}

// Decode attention on the tensor cores (bf16 cache and activations): grid (token, kv head,
// split) as gqa_attn_kernel, but the block's kAtcWarps warps each stream their own 16-position
// tiles of the split through a per-warp cp.async ring (K and V rows in 16-byte chunks, rows
// past the split zero). Per tile S = Q K^T with the kv group's heads as the MMA rows (padded to
// 16; Q rounded to bf16 once) and O += P V, m16n8k16 MMAs with the online softmax in fp32
// (base 2). The warps' partials merge in warp order; the block then writes its split partial
// (m in natural units, l) or the result, and the last split block combines, as gqa_attn_kernel.
// A token's result depends only on its own position's splits and the fixed tile order.
constexpr int kAtcWarps = 4, kAtcStages = 3, kAtcTile = 16;
constexpr int kAtcMaxSplits = 32;  // this kernel's split policy: fewer, longer splits
// fp8 cache: half as many again (each split's prologue and warp merge were a third of its
// instructions at 16 tiles per split)
constexpr int kAtc8MaxSplits = 16;
// fp8 cache: P is carried as p * 2^(ev - E + kP8) <= 2^kP8, inside fp16's range, so a weight keeps
// fp16's precision down to p * 2^(ev - E) ~ 2^-(14 + kP8): without the offset the small weights of
// positions whose values lie far below the stream's largest exponent fall into fp16's subnormals
constexpr int kP8 = 14;
template <int DR>
constexpr int atc_stage_elems() {
    return 2 * kAtcTile * (32 * DR + 8);  // K rows then V rows, padded to a conflict-free pitch
}
// F8 (fp8 cache, kv_fp8.h): a stage is the tile's e4m3 key rows (pitch LK8), value rows (LV8),
// then its 16 key and 16 value exponent bytes; the pitches make the lanes' 16-byte reads below
// conflict-free
template <int DR>
constexpr int atc8_lk() { return (32 * DR) % 128 == 64 ? 32 * DR : 32 * DR + 64; }
template <int DR>
constexpr int atc8_stage_bytes() { return kAtcTile * (atc8_lk<DR>() + 32 * DR + 16) + 32; }
// ST: stages of each warp's ring. kAtcStages (3) where the device's dynamic shared memory allows
// it (the H100's 227 KB); 2 for the bf16 cache at head_dim 128 on devices with less (consumer
// Blackwell, sm_120: 99 KB, where the 3-stage ring is 102 KB). The same bits either way: the
// stage count only sets how many tiles are in flight.
template <int DR, bool F8 = false, int ST = kAtcStages>
constexpr size_t atc_smem() {
    const size_t ring = F8 ? size_t(kAtcWarps) * ST * atc8_stage_bytes<DR>()
                           : sizeof(uint16_t) * kAtcWarps * ST * atc_stage_elems<DR>();
    const size_t merge = sizeof(float) * kAtcWarps * 16 * (32 * DR + 2);
    return ring > merge ? ring : merge;
}

// F8: the fp8 cache. Per tile, S = Q K^T and O += P V on fp16 MMAs (e4m3 converts exactly; Q
// rounded to fp16 once): K's k order is permuted so a lane's 16-byte run of a key row holds four
// MMA steps' B values (Q's A fragments permuted alike), and V's dims so a lane's 16-byte runs of
// four value rows give its B values by byte permutes (n-tile dj, column n is dim n * D/8 + dj;
// the warp's partials are written back in natural order). Scores are scaled by their keys'
// powers of two. P is scaled by the values' powers of two relative to the largest seen so far,
// E (a running maximum like the softmax's: O is rescaled when it grows), so it stays in fp16's
// range (times 2^kP8: see there); O is multiplied by 2^(E - 127 - kP8) before the warps merge.
template <int DR, bool F8 = false, int ST = kAtcStages>
__global__ void __launch_bounds__(kAtcWarps * 32, 2)
    gqa_attn_tc_kernel(float *out, float *part, unsigned *cnt, const float *q, const void *__restrict__ kc_,
                       const void *__restrict__ vc_, const int32_t *__restrict__ tok_slot,
                       const int32_t *__restrict__ tok_pos, const int32_t *__restrict__ pt, int64_t max_pages,
                       int64_t page, int Hq, int Hkv, float scale_log2, int max_splits, int zs, int min_len) {
    constexpr int D = 32 * DR, LD = D + 8, SE = atc_stage_elems<DR>();
    constexpr int LK8 = atc8_lk<DR>(), LV8 = D + 16, SB8 = atc8_stage_bytes<DR>(), VB = D / 8;
    const auto *kc = static_cast<const __nv_bfloat16 *>(kc_), *vc = static_cast<const __nv_bfloat16 *>(vc_);
    const auto *kc8 = static_cast<const uint8_t *>(kc_), *vc8 = static_cast<const uint8_t *>(vc_);
    extern __shared__ __align__(16) uint16_t atc_smem_[];
    __shared__ float fs[kAtcWarps][kAttnMaxSplits];
    __shared__ bool last;
    pdl_wait();
    pdl_trigger();
    const int64_t t = blockIdx.x;
    const int kh = blockIdx.y, G = Hq / Hkv;
    int sp = blockIdx.z;
    const int64_t pos = tok_pos[t];
    int64_t S;
    int nsplit;
    attn_split(pos + 1, max_splits, S, nsplit, min_len);
    if (sp >= nsplit) return;  // block-uniform
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
    const int32_t *ptab = pt + static_cast<int64_t>(tok_slot[t]) * max_pages;
    const int64_t W = static_cast<int64_t>(Hkv) * D;
    const int h0 = kh * G;  // the group's first query head

    // Q as A fragments: rows g and g + 8 are heads g and g + 8 of the group (zero past G). F8:
    // fp16, step kk = 4h + s's k-slots 2tq, 2tq + 1 | 2tq + 8, 2tq + 9 are dims c0, c0 + 1 | c0 + 2,
    // c0 + 3 with c0 = 64h + 16tq + 4s
    uint32_t qa[D / 16][4];
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk) {
        auto at = [&](int r, int c) -> float2 {
            if (r >= G) return make_float2(0.f, 0.f);
            return *reinterpret_cast<const float2 *>(q + (t * Hq + h0 + r) * D + c);
        };
        if constexpr (F8) {
            const int c0 = 64 * (kk / 4) + 16 * tq + 4 * (kk % 4);
            const float2 x0 = at(g, c0), x1 = at(g + 8, c0), x2 = at(g, c0 + 2), x3 = at(g + 8, c0 + 2);
            qa[kk][0] = f16x2(x0.x, x0.y), qa[kk][1] = f16x2(x1.x, x1.y);
            qa[kk][2] = f16x2(x2.x, x2.y), qa[kk][3] = f16x2(x3.x, x3.y);
        } else {
            const int c0 = kk * 16 + 2 * tq;
            const float2 x0 = at(g, c0), x1 = at(g + 8, c0), x2 = at(g, c0 + 8), x3 = at(g + 8, c0 + 8);
            qa[kk][0] = bf16x2(x0.x, x0.y), qa[kk][1] = bf16x2(x1.x, x1.y);
            qa[kk][2] = bf16x2(x2.x, x2.y), qa[kk][3] = bf16x2(x3.x, x3.y);
        }
    }
    // the block's splits: blockIdx.z, + gridDim.z, ... (a graph's grid has a fixed depth; fewer
    // blocks than splits keeps short contexts from launching thousands of empty blocks). Each
    // split's partial and the merge are as with one block per split: the same bits.
    for (; sp < nsplit; sp += static_cast<int>(gridDim.z)) {
    if (sp != static_cast<int>(blockIdx.z)) __syncthreads();  // the previous split's merge read the rings
    const int64_t start = static_cast<int64_t>(sp) * S, end = imin(start + S, pos + 1);
    // this warp's tiles: warp, warp + kAtcWarps, ... of the split's ceil((end - start) / 16)
    const int ntile = static_cast<int>((end - start + kAtcTile - 1) / kAtcTile);
    const int mine = warp < ntile ? (ntile - warp + kAtcWarps - 1) / kAtcWarps : 0;
    uint16_t *ring = atc_smem_ + warp * ST * SE;
    uint8_t *ring8 = reinterpret_cast<uint8_t *>(atc_smem_) + warp * ST * SB8;
    constexpr int CPR = D / 8;  // 16-byte chunks per K or V row
    // A tile is 16-aligned, so with pages of a multiple of 16 positions (fp8 requires it) it lies in
    // one page: one page-table lookup per tile. Per row otherwise, in 32 bits (positions < 2^31): a
    // 64-bit division per 16-byte chunk was most of this kernel's instructions (~1000 per tile).
    const bool tile_page = page % kAtcTile == 0;
    const int pg32 = static_cast<int>(page);
    auto issue = [&](int i) {  // tile i of mine into stage i % ST
        const int64_t p0 = start + static_cast<int64_t>(warp + i * kAtcWarps) * kAtcTile;
        const int lp0 = static_cast<int>(p0) / pg32, r0 = static_cast<int>(p0) - lp0 * pg32;
        const int64_t pg0 = ptab[lp0];
        if constexpr (F8) {
            // the tile's exponents are one 16-byte run of its page
            uint8_t *ks = ring8 + (i % ST) * SB8, *vs = ks + kAtcTile * LK8, *es = vs + kAtcTile * LV8;
            const int64_t base = kv8_row(pg0, r0, kh, page, Hkv, D);
            constexpr int CPR8 = D / 16;
#pragma unroll
            for (int j = 0; j < kAtcTile * CPR8 / 32; ++j) {
                const int c = lane + 32 * j, r = c / CPR8, cc = c % CPR8;
                const bool valid = p0 + r < end;
                const int64_t row = base + static_cast<int64_t>(r * Hkv * D + cc * 16);
                cp_async16_zfill(ks + r * LK8 + cc * 16, kc8 + (valid ? row : 0), valid);
                cp_async16_zfill(vs + r * LV8 + cc * 16, vc8 + (valid ? row : 0), valid);
            }
            if (lane < 2) cp_async16(es + 16 * lane, (lane ? vc8 : kc8) + kv8_exp(pg0, r0, kh, page, Hkv, D));
        } else {
            uint16_t *ks = ring + (i % ST) * SE, *vs = ks + kAtcTile * LD;
            const int64_t base = (pg0 * page + r0) * W + kh * D;
#pragma unroll
            for (int j = 0; j < kAtcTile * CPR / 32; ++j) {  // the lane's chunks: c = lane + 32 j
                const int c = lane + 32 * j, r = c / CPR, cc = c % CPR;
                const int pr = static_cast<int>(p0) + r;
                const bool valid = pr < end;
                int64_t row = base + static_cast<int64_t>(r) * W;
                if (!tile_page && valid) {
                    const int lp = pr / pg32;
                    row = (static_cast<int64_t>(ptab[lp]) * page + (pr - lp * pg32)) * W + kh * D;
                }
                cp_async16_zfill(ks + r * LD + cc * 8, kc + (valid ? row : 0) + cc * 8, valid);
                cp_async16_zfill(vs + r * LD + cc * 8, vc + (valid ? row : 0) + cc * 8, valid);
            }
        }
    };
    float o[D / 8][4];
#pragma unroll
    for (int j = 0; j < D / 8; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;  // rows g, g + 8
    int E = 0;  // F8: the largest value exponent so far (O is held in units of 2^(E - 127))
#pragma unroll
    for (int st = 0; st < ST - 1; ++st) {
        if (st < mine) issue(st);
        cp_async_commit();
    }
    for (int i = 0; i < mine; ++i) {
        cp_async_wait<ST - 2>();  // tile i has landed (this lane's copies)
        __syncwarp();                     // every lane's; and tile i - 1 is no longer read
        const int nx = i + ST - 1;
        if (nx < mine) issue(nx);
        cp_async_commit();
        const uint16_t *ks = ring + (i % ST) * SE, *vs = ks + kAtcTile * LD;
        const uint8_t *ks8 = ring8 + (i % ST) * SB8, *vs8 = ks8 + kAtcTile * LK8, *es8 = vs8 + kAtcTile * LV8;
        const int64_t p0 = start + static_cast<int64_t>(warp + i * kAtcWarps) * kAtcTile;
        // S = Q K^T: n-tiles of positions 0-7 and 8-15
        float sc[2][4];
        sc[0][0] = sc[0][1] = sc[0][2] = sc[0][3] = sc[1][0] = sc[1][1] = sc[1][2] = sc[1][3] = 0.0f;
        if constexpr (F8) {
#pragma unroll
            for (int ni = 0; ni < 2; ++ni)
#pragma unroll
                for (int h = 0; h < D / 64; ++h) {
                    const uint4 kw = *reinterpret_cast<const uint4 *>(ks8 + (ni * 8 + g) * LK8 + 64 * h + 16 * tq);
                    const uint32_t ww[4] = {kw.x, kw.y, kw.z, kw.w};
#pragma unroll
                    for (int st = 0; st < 4; ++st) {
                        const uint32_t(&a)[4] = qa[4 * h + st];
                        mma_f16(sc[ni], a[0], a[1], a[2], a[3], e4m3x2_f16x2(ww[st]), e4m3x2_hi_f16x2(ww[st]));
                    }
                }
        } else {
#pragma unroll
            for (int kk = 0; kk < D / 16; ++kk) {
                uint32_t bb[4];
                ldsm_x4(bb, ks + ((lane & 7) + ((lane >> 4) << 3)) * LD + kk * 16 + ((lane >> 3) & 1) * 8);
                mma_bf16(sc[0], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], bb[0], bb[1]);
                mma_bf16(sc[1], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], bb[2], bb[3]);
            }
        }
        // F8: the lane's positions' exponents, keys and values: (n-tile ni, e) is byte 8 ni + 2tq + e
        uint32_t ekp[2] = {0, 0}, evp[2] = {0, 0};
        if constexpr (F8) {
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                ekp[ni] = *reinterpret_cast<const uint16_t *>(es8 + 8 * ni + 2 * tq);
                evp[ni] = *reinterpret_cast<const uint16_t *>(es8 + 16 + 8 * ni + 2 * tq);
            }
        }
        // mask past the split, online softmax: rows g (sc[.][0..1]) and g + 8 (sc[.][2..3])
        float mx0 = m0, mx1 = m1;
        int et = 0;  // F8: the tile's largest value exponent (valid positions)
#pragma unroll
        for (int ni = 0; ni < 2; ++ni)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const bool ok = p0 + ni * 8 + 2 * tq + e < end;
                float f = scale_log2;
                if constexpr (F8) {
                    f *= pow2_biased(static_cast<int>((ekp[ni] >> (8 * e)) & 0xFFu));
                    if (ok) et = max(et, static_cast<int>((evp[ni] >> (8 * e)) & 0xFFu));
                }
                sc[ni][e] = ok ? sc[ni][e] * f : -INFINITY;
                sc[ni][2 + e] = ok ? sc[ni][2 + e] * f : -INFINITY;
                mx0 = fmaxf(mx0, sc[ni][e]), mx1 = fmaxf(mx1, sc[ni][2 + e]);
            }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
        }
        const float b0m = mx0 == -INFINITY ? 0.0f : mx0, b1m = mx1 == -INFINITY ? 0.0f : mx1;
        const float c0 = exp2f(m0 - b0m), c1 = exp2f(m1 - b1m);
        m0 = mx0, m1 = mx1;
        float oc0 = c0, oc1 = c1;  // O's rescale: the softmax's, and for F8 E's
        if constexpr (F8) {
#pragma unroll
            for (int off = 1; off <= 2; off <<= 1) et = max(et, __shfl_xor_sync(0xffffffffu, et, off));
            const int En = max(E, et);
            const float fe = pow2_biased(127 + E - En);
            oc0 *= fe, oc1 *= fe, E = En;
        }
        uint32_t pa[4];
        float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            const float p00 = exp2f(sc[ni][0] - b0m), p01 = exp2f(sc[ni][1] - b0m);
            const float p10 = exp2f(sc[ni][2] - b1m), p11 = exp2f(sc[ni][3] - b1m);
            s0 += p00 + p01, s1 += p10 + p11;
            // n-tile ni is k-slots 8 ni + 2tq, + 1 of the A fragment
            if constexpr (F8) {  // times the values' powers of two over 2^(E - 127 - kP8)
                const float v0 = pow2_biased(127 + kP8 + static_cast<int>(evp[ni] & 0xFFu) - E);
                const float v1 = pow2_biased(127 + kP8 + static_cast<int>((evp[ni] >> 8) & 0xFFu) - E);
                pa[2 * ni] = f16x2_in_range(p00 * v0, p01 * v1);  // in [0, 1]: no clamp
                pa[2 * ni + 1] = f16x2_in_range(p10 * v0, p11 * v1);
            } else {
                pa[2 * ni] = bf16x2(p00, p01);
                pa[2 * ni + 1] = bf16x2(p10, p11);
            }
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            s0 += __shfl_xor_sync(0xffffffffu, s0, off);
            s1 += __shfl_xor_sync(0xffffffffu, s1, off);
        }
        l0 = l0 * c0 + s0, l1 = l1 * c1 + s1;
        // O's rescale. F8: skipped when no lane's factor changed (times 1: the same bits); once the
        // running maxima settle that is most tiles, and it was 64 of the tile's ~530 instructions
        // (the bf16 tile measured no gain from it)
        if (!F8 || __any_sync(0xffffffffu, oc0 != 1.0f || oc1 != 1.0f)) {
#pragma unroll
            for (int j = 0; j < D / 8; ++j) o[j][0] *= oc0, o[j][1] *= oc0, o[j][2] *= oc1, o[j][3] *= oc1;
        }
        if constexpr (F8) {
            // O += P V: the lane's value rows 2tq, 2tq + 1 (b0) and 2tq + 8, 2tq + 9 (b1), bytes
            // g * VB .. + VB; byte permutes pair the rows' bytes per dim, two dims per word
            constexpr int NW = VB / 4;  // words per row run
            uint32_t r[4][NW];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint8_t *src = vs8 + (2 * tq + (k & 1) + 8 * (k >> 1)) * LV8 + g * VB;
                if constexpr (NW == 4) {
                    const uint4 u = *reinterpret_cast<const uint4 *>(src);
                    r[k][0] = u.x, r[k][1] = u.y, r[k][2] = u.z, r[k][3] = u.w;
                } else {
                    const uint2 u = *reinterpret_cast<const uint2 *>(src);
                    r[k][0] = u.x, r[k][1] = u.y;
                }
            }
#pragma unroll
            for (int w = 0; w < NW; ++w) {
                const uint32_t lo01 = __byte_perm(r[0][w], r[1][w], 0x5140), hi01 = __byte_perm(r[0][w], r[1][w], 0x7362);
                const uint32_t lo23 = __byte_perm(r[2][w], r[3][w], 0x5140), hi23 = __byte_perm(r[2][w], r[3][w], 0x7362);
                mma_f16(o[4 * w], pa[0], pa[1], pa[2], pa[3], e4m3x2_f16x2(lo01), e4m3x2_f16x2(lo23));
                mma_f16(o[4 * w + 1], pa[0], pa[1], pa[2], pa[3], e4m3x2_hi_f16x2(lo01), e4m3x2_hi_f16x2(lo23));
                mma_f16(o[4 * w + 2], pa[0], pa[1], pa[2], pa[3], e4m3x2_f16x2(hi01), e4m3x2_f16x2(hi23));
                mma_f16(o[4 * w + 3], pa[0], pa[1], pa[2], pa[3], e4m3x2_hi_f16x2(hi01), e4m3x2_hi_f16x2(hi23));
            }
        } else {
            // O += P V: V fragments by ldmatrix.trans, two n-tiles (16 dims) per load
#pragma unroll
            for (int dj = 0; dj < D / 8; dj += 2) {
                uint32_t vb[4];
                ldsm_x4_trans(vb, vs + (lane & 15) * LD + (dj + (lane >> 4)) * 8);
                mma_bf16(o[dj], pa[0], pa[1], pa[2], pa[3], vb[0], vb[1]);
                mma_bf16(o[dj + 1], pa[0], pa[1], pa[2], pa[3], vb[2], vb[3]);
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();  // the rings are free: the warps' partials go there, [warp][16 heads][D + 2]
    float *mg = reinterpret_cast<float *>(atc_smem_);
    constexpr int ML = D + 2;
    if constexpr (F8) {  // O in natural units, its dims in natural order
        const float fo = pow2_biased(E - kP8);
#pragma unroll
        for (int j = 0; j < D / 8; ++j) {
            float *r0 = mg + (warp * 16 + g) * ML + j, *r1 = r0 + 8 * ML;
            r0[2 * tq * VB] = o[j][0] * fo, r0[(2 * tq + 1) * VB] = o[j][1] * fo;
            r1[2 * tq * VB] = o[j][2] * fo, r1[(2 * tq + 1) * VB] = o[j][3] * fo;
        }
    } else {
#pragma unroll
        for (int j = 0; j < D / 8; ++j) {
            float *r0 = mg + (warp * 16 + g) * ML + j * 8 + 2 * tq, *r1 = r0 + 8 * ML;
            r0[0] = o[j][0], r0[1] = o[j][1], r1[0] = o[j][2], r1[1] = o[j][3];
        }
    }
    if (tq == 0) {
        mg[(warp * 16 + g) * ML + D] = m0, mg[(warp * 16 + g) * ML + D + 1] = l0;
        mg[(warp * 16 + g + 8) * ML + D] = m1, mg[(warp * 16 + g + 8) * ML + D + 1] = l1;
    }
    __syncthreads();
    // merge the warps (fixed order), one warp per head; m is kept in natural units for the combine
    constexpr float kLn2 = 0.6931471805599453f;
    for (int h = warp; h < G; h += kAtcWarps) {
        float M = -INFINITY;
#pragma unroll
        for (int w = 0; w < kAtcWarps; ++w) M = fmaxf(M, mg[(w * 16 + h) * ML + D]);
        float f[kAtcWarps], l = 0.0f;
#pragma unroll
        for (int w = 0; w < kAtcWarps; ++w) {
            const float mw = mg[(w * 16 + h) * ML + D];
            f[w] = mw == -INFINITY ? 0.0f : exp2f(mw - M);
            l = fmaf(mg[(w * 16 + h) * ML + D + 1], f[w], l);
        }
        float *orow = out + (t * Hq + h0 + h) * D;
        float *pp = part + ((t * Hq + h0 + h) * zs + sp) * static_cast<int64_t>(ML);
#pragma unroll
        for (int c = 0; c < DR; ++c) {
            const int d = lane + 32 * c;
            float a = 0.0f;
#pragma unroll
            for (int w = 0; w < kAtcWarps; ++w) a = fmaf(mg[(w * 16 + h) * ML + d], f[w], a);
            if (nsplit == 1) orow[d] = a / l;
            else pp[d] = a;
        }
        if (nsplit > 1 && lane == 0) pp[D] = M * kLn2, pp[D + 1] = l;
    }
    if (nsplit == 1 || !cnt) continue;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned *ct = cnt + t * Hkv + kh;
        last = atomicAdd(ct, 1u) == static_cast<unsigned>(nsplit - 1);
        if (last) *ct = 0;
    }
    __syncthreads();
    if (!last) continue;
    __threadfence();
    // the last split block merges every split, as gqa_attn_kernel does
    float *f = fs[warp];
    for (int h = warp; h < G; h += kAtcWarps) {
        const float *base = part + (t * Hq + h0 + h) * zs * static_cast<int64_t>(ML);
        float ms[2], ls[2], M = -INFINITY;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int s2 = lane + 32 * j;
            ms[j] = s2 < nsplit ? __ldcg(base + s2 * ML + D) : -INFINITY;
            ls[j] = s2 < nsplit ? __ldcg(base + s2 * ML + D + 1) : 0.0f;
            M = fmaxf(M, ms[j]);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, off));
        float lp = 0.0f;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int s2 = lane + 32 * j;
            if (s2 < nsplit) {
                const float e = expf(ms[j] - M);
                f[s2] = e;
                lp = fmaf(ls[j], e, lp);
            }
        }
        const float inv = 1.0f / warp_sum(lp);
        __syncwarp();
        float a[DR];
#pragma unroll
        for (int c = 0; c < DR; ++c) a[c] = 0.0f;
        for (int s2 = 0; s2 < nsplit; ++s2) {
            const float e = f[s2];
#pragma unroll
            for (int c = 0; c < DR; ++c) a[c] = fmaf(__ldcg(base + s2 * ML + lane + 32 * c), e, a[c]);
        }
        float *orow = out + (t * Hq + h0 + h) * D;
#pragma unroll
        for (int c = 0; c < DR; ++c) orow[lane + 32 * c] = a[c] * inv;
        __syncwarp();
    }
    return;  // every split is done: this was the block's last
    }
}

// K/V rows 0 .. n-1 of one sequence from the paged cache into contiguous [n][W] rows (bf16);
// rows n .. gridDim.x-1 are zero (padding).
__global__ void kv_gather_kernel(__nv_bfloat16 *kd, __nv_bfloat16 *vd, const __nv_bfloat16 *__restrict__ kc,
                                 const __nv_bfloat16 *__restrict__ vc, const int32_t *__restrict__ ptab, int64_t page,
                                 int64_t W, int64_t n) {
    const int64_t p = blockIdx.x;
    uint4 *kt = reinterpret_cast<uint4 *>(kd + p * W), *vt = reinterpret_cast<uint4 *>(vd + p * W);
    if (p >= n) {
        for (int c = threadIdx.x; c < W / 8; c += blockDim.x) kt[c] = vt[c] = make_uint4(0, 0, 0, 0);
        return;
    }
    const int64_t row = (static_cast<int64_t>(ptab[p / page]) * page + p % page) * W;
    const uint4 *ks = reinterpret_cast<const uint4 *>(kc + row), *vs = reinterpret_cast<const uint4 *>(vc + row);
    for (int c = threadIdx.x; c < W / 8; c += blockDim.x) kt[c] = ks[c], vt[c] = vs[c];
}

// fp8 cache pages (kv_fp8.h) as bf16 pages ([page][Hkv][D]) for the prefill attention: block (j, 0)
// converts staging page j's keys, (j, 1) its values, from physical page ptab[src[j]]. Exact: an
// e4m3 value times a power of two is a bf16.
__global__ void kv8_dequant_kernel(__nv_bfloat16 *kd, __nv_bfloat16 *vd, const uint8_t *__restrict__ kc,
                                   const uint8_t *__restrict__ vc, const int32_t *__restrict__ src,
                                   const int32_t *__restrict__ ptab, int64_t page, int64_t Hkv, int64_t D) {
    pdl_wait();
    pdl_trigger();
    const int64_t j = blockIdx.x;
    const uint8_t *pool = blockIdx.y ? vc : kc;
    __nv_bfloat16 *dst = (blockIdx.y ? vd : kd) + j * page * Hkv * D;
    const int64_t pg = ptab[src[j]];
    const uint8_t *vals = pool + kv8_row(pg, 0, 0, page, Hkv, D), *ex = pool + kv8_exp(pg, 0, 0, page, Hkv, D);
    for (int64_t i = threadIdx.x; i < page * Hkv * D / 8; i += blockDim.x) {
        const int64_t e0 = i * 8, r = e0 / (Hkv * D), h = (e0 / D) % Hkv;
        const float sc = pow2_biased(ex[h * page + r]);
        const uint2 b = *reinterpret_cast<const uint2 *>(vals + e0);
        uint32_t w[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t pair = e4m3x2_f16x2((k < 2 ? b.x : b.y) >> (16 * (k & 1)));
            const float2 f = __half22float2(*reinterpret_cast<const __half2 *>(&pair));
            w[k] = bf16x2(f.x * sc, f.y * sc);
        }
        *reinterpret_cast<uint4 *>(dst + e0) = make_uint4(w[0], w[1], w[2], w[3]);
    }
}


// q/k RMSNorm + RoPE + cache append: grid (token, head blocks of 8 warps), one warp per head
// of the Hq + Hkv: query heads are normalised and rotated in place, key heads normalised,
// rotated and written to the cache with their values. Lane holds dims lane + 32c (DR = D/32);
// a RoPE pair (j, j + D/2) is in one lane when D >= 64, across lanes 16 apart when D = 32.
// IT: the q/k/v input type (fp32, or the bf16 twins the GEMMs left); q null: only q16 is written.
// KT uint8_t: the fp8 cache (kv_fp8.h): each key and value vector's exponent from its warp-wide
// largest magnitude, the e4m3 values, and the exponents into the page's exponent rows.
// KT __nv_fp8_e4m3: plain e4m3 pages (DType::FP8E4M3), every value at scale 1, saturating.
template <int DR, typename KT, typename IT>
__global__ void qk_rope_append_kernel(float *q, const IT *qi, const IT *ki, const IT *vi, KT *kc, KT *vc,
                                      const float *__restrict__ qn, const float *__restrict__ kn, float eps,
                                      const float *__restrict__ inv_freq, const int32_t *__restrict__ tok_slot,
                                      const int32_t *__restrict__ tok_pos, const int32_t *__restrict__ pt,
                                      int64_t max_pages, int64_t page, int Hq, int Hk, uint16_t *q16, int64_t ldq,
                                      int64_t ldkv, uint8_t *q8) {
    constexpr int D = 32 * DR;
    pdl_wait();
    pdl_trigger();
    const int64_t t = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int hh = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (hh >= Hq + Hk) return;  // warp-uniform
    const bool isq = hh < Hq;
    const int h = isq ? hh : hh - Hq;
    float *qrow = q ? q + (t * Hq + h) * D : nullptr;
    const IT *src = isq ? qi + t * ldq + h * D : ki + t * ldkv + h * D;  // the inputs' rows: ldq, ldkv apart
    float x[DR];
#pragma unroll
    for (int c = 0; c < DR; ++c) x[c] = row_at(src, lane + 32 * c);
    const float *nw = isq ? qn : kn;
    if (nw) {
        float ss = 0.0f;
#pragma unroll
        for (int c = 0; c < DR; ++c) ss = fmaf(x[c], x[c], ss);
        const float inv = 1.0f / sqrtf(warp_sum(ss) / static_cast<float>(D) + eps);
#pragma unroll
        for (int c = 0; c < DR; ++c) x[c] = nw[lane + 32 * c] * (x[c] * inv);
    }
    const float pos = static_cast<float>(tok_pos[t]);
    if constexpr (DR >= 2) {
#pragma unroll
        for (int c = 0; c < DR / 2; ++c) {
            const float ang = pos * inv_freq[lane + 32 * c];
            const float cs = cosf(ang), sn = sinf(ang), a = x[c], b = x[c + DR / 2];
            x[c] = a * cs - b * sn;
            x[c + DR / 2] = b * cs + a * sn;
        }
    } else {  // D = 32
        const float ang = pos * inv_freq[lane & 15];
        const float cs = cosf(ang), sn = sinf(ang), other = __shfl_xor_sync(0xffffffffu, x[0], 16);
        x[0] = lane < 16 ? x[0] * cs - other * sn : x[0] * cs + other * sn;
    }
    if (isq) {
        uint16_t *q16r = q16 ? q16 + (t * Hq + h) * D : nullptr;
        uint8_t *q8r = q8 ? q8 + (t * Hq + h) * D : nullptr;
#pragma unroll
        for (int c = 0; c < DR; ++c) {
            if (qrow) qrow[lane + 32 * c] = x[c];
            if (q16r) q16r[lane + 32 * c] = f2bf(x[c]);
            if (q8r) q8r[lane + 32 * c] = f2e4m3(x[c]);
        }
        return;
    }
    const int64_t p = tok_pos[t];
    const int64_t pg = pt[static_cast<int64_t>(tok_slot[t]) * max_pages + p / page];
    const IT *vr = vi + t * ldkv + h * D;
    if constexpr (std::is_same_v<KT, uint8_t>) {
        float v[DR], ak = 0.0f, av = 0.0f;
#pragma unroll
        for (int c = 0; c < DR; ++c) {
            v[c] = row_at(vr, lane + 32 * c);
            ak = fmaxf(ak, fabsf(x[c])), av = fmaxf(av, fabsf(v[c]));
        }
        const uint32_t ek = kv8_exp_dev(warp_max(ak)), ev = kv8_exp_dev(warp_max(av));
        const float ik = pow2_biased(254 - static_cast<int>(ek)), iv = pow2_biased(254 - static_cast<int>(ev));
        const int64_t row = kv8_row(pg, p % page, h, page, Hk, D);
#pragma unroll
        for (int c = 0; c < DR; ++c) {
            kc[row + lane + 32 * c] = f2e4m3(x[c] * ik);
            vc[row + lane + 32 * c] = f2e4m3(v[c] * iv);
        }
        if (lane == 0) {
            const int64_t ex = kv8_exp(pg, p % page, h, page, Hk, D);
            kc[ex] = static_cast<uint8_t>(ek), vc[ex] = static_cast<uint8_t>(ev);
        }
    } else if constexpr (std::is_same_v<KT, __nv_fp8_e4m3>) {
        const int64_t row = (pg * page + p % page) * (static_cast<int64_t>(Hk) * D) + static_cast<int64_t>(h) * D;
        auto *k8 = reinterpret_cast<uint8_t *>(kc), *v8 = reinterpret_cast<uint8_t *>(vc);
#pragma unroll
        for (int c = 0; c < DR; ++c) {
            k8[row + lane + 32 * c] = f2e4m3(x[c]);
            v8[row + lane + 32 * c] = f2e4m3(row_at(vr, lane + 32 * c));
        }
    } else {
        const int64_t row = (pg * page + p % page) * (static_cast<int64_t>(Hk) * D) + static_cast<int64_t>(h) * D;
#pragma unroll
        for (int c = 0; c < DR; ++c) {
            stst(kc + row + lane + 32 * c, x[c]);
            stst(vc + row + lane + 32 * c, row_at(vr, lane + 32 * c));
        }
    }
}

}  // namespace
}  // namespace fullm
