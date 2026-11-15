// SPDX-License-Identifier: Apache-2.0
// kernels/kda_chunk.cuh - Kimi Delta Attention over a long segment (a prompt) in chunks
// (Backend::kda_chunked). Part of cuda_backend.cu's translation unit, after kimi.cuh.
//
// kda_fused_kernel walks a segment's tokens one after another (per token: S *= alpha row-wise;
// u = S^T k; S += k (beta (v - u))^T; o = S^T q), which a prompt of 2048 tokens makes 2048
// dependent steps a layer (22.9 ms on an H100, 80% of Kimi Linear's prefill). The same updates
// in their chunk-parallel form (the gated delta rule's, Yang et al. 2024, as flash-linear-attention
// computes it, with a decay per key channel; NOTICE): within a
// chunk of C tokens, from the state S0 that enters it, with g_t the cumulative log decay since
// the chunk's start (g <= 0, so every exponential below is of a value <= 0),
//   L[t][s] = sum_i k_t[i] k_s[i] exp(g_t[i] - g_s[i])        (s < t)
//   M[t][s] = sum_i q_t[i] k_s[i] exp(g_t[i] - g_s[i])        (s <= t)
//   U = T diag(beta) V,  Wk = T diag(beta) (exp(g) * K),  T = (I + diag(beta) L)^-1 (forward substitution)
//   P = exp(g) * Q - M Wk,  Oi = M U,  Kb[s] = k_s * exp(g_C - g_s)
// every chunk at once (kda_chunk_prep_tc_kernel), then chunk after chunk (kda_chunk_scan_tc_kernel):
//   W = U - Wk S0,  O = P S0 + Oi,  S = diag(exp(g_C)) S0 + Kb^T W
// whose columns of S (the value dimension) are independent: blocks take slices of them. The
// q / k / v short convolutions, the L2 norms, the gates and the output norm and gate are
// kda_fused_kernel's arithmetic, per token in parallel (in kda_chunk_prep_tc_kernel, and
// kda_out_seg_kernel). fp32 values throughout (the products 3xTF32 on the tensor cores); the
// results equal the recurrence's up to the order of the sums.
#pragma once

namespace fullm {
namespace {

constexpr int kKdaChunk = 32;  // tokens a chunk
constexpr int kKdaSlice = 32;  // value columns a scan block
// The chunk kernels keep the cumulative log decay in base 2 (g * log2 e): every decay is an exp2f,
// cheaper than expf (no range reduction by ln 2). g_C in the scratch is base 2.
constexpr float kLog2e = 1.4426950408889634f;

// The convolution histories after a segment: its last KH inputs (oldest first), per channel.
template <int KH>
__global__ void kda_conv_hist_kernel(float *conv, int64_t cstride, int64_t hist, const float *qp, const float *kp,
                                     const float *vp, const int32_t *__restrict__ seg, int64_t P) {
    const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= P) return;
    const int s = blockIdx.y;
    const int64_t slot = seg[3 * s], off = seg[3 * s + 1], len = seg[3 * s + 2];
    const float *xs[3] = {qp, kp, vp};
    for (int m = 0; m < 3; ++m) {
        float *hp = conv + slot * cstride + m * hist + c * KH;
        float nb[KH];
#pragma unroll
        for (int i = 0; i < KH; ++i) {
            const int64_t tt = len - KH + i;
            nb[i] = tt >= 0 ? xs[m][(off + tt) * P + c] : hp[KH + tt];
        }
#pragma unroll
        for (int i = 0; i < KH; ++i) hp[i] = nb[i];
    }
}

// The first chunk of segment s among the step's chunks (the scratch is indexed by chunk, so it
// grows with the step's tokens, not with segments x the longest segment).
template <int C>
__device__ __forceinline__ int64_t kda_chunk_base(const int32_t *__restrict__ seg, int s) {
    int64_t b = 0;
    for (int i = 0; i < s; ++i) b += (seg[3 * i + 2] + C - 1) / C;
    return b;
}

// Chunk after chunk for one (value slice, head, segment): the slice of S in shared memory, read
// from and written back to the slot's state.
template <int D, int C, int VS>
constexpr int kda_scan_smem() {
    return (D * VS + 3 * C * (D + 1) + 3 * C * VS + D) * 4;
}
template <int D, int C, int VS, typename ST>
__global__ void __launch_bounds__(256)
    kda_chunk_scan_kernel(float *o, const float *scr, ST *state, int64_t sstride, const int32_t *__restrict__ seg,
                          int H) {
    extern __shared__ float sm[];
    constexpr int R = D + 1, NT = 256;
    float *S = sm, *sW = S + D * VS, *sP = sW + C * R, *sK = sP + C * R, *sU = sK + C * R, *sO = sU + C * VS,
          *sWv = sO + C * VS, *sg = sWv + C * VS;
    const int j0 = blockIdx.x * VS, h = blockIdx.y, s = blockIdx.z, tid = threadIdx.x;
    const int64_t slot = seg[3 * s], off = seg[3 * s + 1], len = seg[3 * s + 2];
    const int64_t P = static_cast<int64_t>(H) * D;
    ST *Sg = state + slot * sstride + static_cast<int64_t>(h) * D * D;
    for (int idx = tid; idx < D * VS; idx += NT) S[idx] = ldst(Sg + static_cast<int64_t>(idx / VS) * D + j0 + idx % VS);
    const int nch = static_cast<int>((len + C - 1) / C);
    const int64_t cb = kda_chunk_base<C>(seg, s);
    for (int c = 0; c < nch; ++c) {
        const float *base = scr + ((cb + c) * H + h) * (5 * C * D + D);
        __syncthreads();  // the previous chunk's readers are done
        for (int idx = tid; idx < C * D; idx += NT) {
            const int t = idx / D, i = idx % D;
            sW[t * R + i] = base[C * D + idx], sP[t * R + i] = base[2 * C * D + idx], sK[t * R + i] = base[4 * C * D + idx];
        }
        for (int idx = tid; idx < C * VS; idx += NT) {
            const int t = idx / VS, jj = idx % VS;
            sU[idx] = base[t * D + j0 + jj], sO[idx] = base[3 * C * D + t * D + j0 + jj];
        }
        for (int i = tid; i < D; i += NT) sg[i] = exp2f(base[5 * C * D + i]);
        __syncthreads();
        // W = U - Wk S, O = P S + Oi
        const int64_t t0 = static_cast<int64_t>(c) * C;
        for (int idx = tid; idx < C * VS; idx += NT) {
            const int t = idx / VS, jj = idx % VS;
            float w = sU[idx], ov = sO[idx];
#pragma unroll 8
            for (int i = 0; i < D; ++i) {
                const float sv = S[i * VS + jj];
                w = fmaf(-sW[t * R + i], sv, w);
                ov = fmaf(sP[t * R + i], sv, ov);
            }
            sWv[idx] = w;
            if (t0 + t < len) o[(off + t0 + t) * P + static_cast<int64_t>(h) * D + j0 + jj] = ov;
        }
        __syncthreads();
        // S = diag(exp(g_C)) S + Kb^T W
        for (int idx = tid; idx < D * VS; idx += NT) {
            const int i = idx / VS, jj = idx % VS;
            float acc = sg[i] * S[idx];
#pragma unroll 8
            for (int t = 0; t < C; ++t) acc = fmaf(sK[t * R + i], sWv[t * VS + jj], acc);
            S[idx] = acc;
        }
    }
    __syncthreads();
    for (int idx = tid; idx < D * VS; idx += NT) stst(Sg + static_cast<int64_t>(idx / VS) * D + j0 + idx % VS, S[idx]);
}

// The same scan on the tensor cores: m16n8k8 TF32 MMAs in the 3xTF32 form (each operand split
// into a TF32 head and a TF32 remainder, products as As*Bb + Ab*Bs + Ab*Bb: ~fp32 accuracy, the
// state compounding over hundreds of chunks), the chunk's matrices double-buffered through
// cp.async (the next chunk lands while this one is computed). 8 warps: W = U - Wk S and O = P S +
// Oi a 16 x 8 tile each per warp; S = diag(exp(g_C)) S + Kb^T W with the warp's rows of S held in
// MMA accumulators across the chunks (also written to shared memory, where the next chunk's
// products read it as an operand).
__device__ __forceinline__ void kda_tf32_split(float x, uint32_t &big, uint32_t &small) {
    uint32_t b, r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(b) : "f"(x));
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x - __uint_as_float(b)));
    big = b, small = r;
}
__device__ __forceinline__ void kda_mma_tf32(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// c += a b, 3xTF32, from the operands already split
__device__ __forceinline__ void kda_mma_3xtf32(float (&c)[4], const uint32_t (&ab)[4], const uint32_t (&as)[4],
                                           const uint32_t (&bb)[2], const uint32_t (&bs)[2]) {
    kda_mma_tf32(c, as, bb);
    kda_mma_tf32(c, ab, bs);
    kda_mma_tf32(c, ab, bb);
}

template <int D, int C, int VS>
struct KdaScanTc {
    static constexpr int LA = D + 4;   // Wk, P rows (A operands read by row: conflict-free)
    static constexpr int LK = D + 8;   // Kb rows (read transposed)
    static constexpr int LV = VS + 8;  // U, Oi, S, W rows
    static constexpr int STAGE = 2 * C * LA + C * LK + 2 * C * LV + D;
    static constexpr int SMEM = (2 * STAGE + D * LV + C * LV) * 4;
};

template <int D, int C, int VS, typename ST>
__global__ void __launch_bounds__(256)
    kda_chunk_scan_tc_kernel(float *o, const float *scr, ST *state, int64_t sstride, const int32_t *__restrict__ seg,
                             int H) {
    using G = KdaScanTc<D, C, VS>;
    static_assert(C == 32 && VS == 32 && D % 32 == 0, "8 warps: 2 x 4 tiles of W and O, D/16 x 4 of S");
    constexpr int LA = G::LA, LK = G::LK, LV = G::LV, MT = D / 16, NTW = MT / 2;
    extern __shared__ __align__(16) float sm[];
    float *sS = sm + 2 * G::STAGE, *sWv = sS + D * LV;
    const int j0 = blockIdx.x * VS, h = blockIdx.y, s = blockIdx.z, tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
    const int64_t slot = seg[3 * s], off = seg[3 * s + 1], len = seg[3 * s + 2];
    const int64_t P = static_cast<int64_t>(H) * D;
    const int nch = static_cast<int>((len + C - 1) / C);
    const int64_t cb = kda_chunk_base<C>(seg, s);
    ST *Sg = state + slot * sstride + static_cast<int64_t>(h) * D * D;

    auto load = [&](int c, int buf) {  // chunk c's Wk, P, Kb, U and Oi slices, g_C
        const float *base = scr + ((cb + c) * H + h) * (5 * C * D + D);
        float *st = sm + buf * G::STAGE;
        float *sW = st, *sP = sW + C * LA, *sK = sP + C * LA, *sU = sK + C * LK, *sO = sU + C * LV, *sg = sO + C * LV;
        for (int x = tid; x < C * D / 4; x += 256) {
            const int t = x / (D / 4), i = (x % (D / 4)) * 4;
            cp_async16(sW + t * LA + i, base + C * D + t * D + i);
            cp_async16(sP + t * LA + i, base + 2 * C * D + t * D + i);
            cp_async16(sK + t * LK + i, base + 4 * C * D + t * D + i);
        }
        for (int x = tid; x < C * VS / 4; x += 256) {
            const int t = x / (VS / 4), j = (x % (VS / 4)) * 4;
            cp_async16(sU + t * LV + j, base + t * D + j0 + j);
            cp_async16(sO + t * LV + j, base + 3 * C * D + t * D + j0 + j);
        }
        for (int x = tid; x < D / 4; x += 256) cp_async16(sg + 4 * x, base + 5 * C * D + 4 * x);
    };

    // the warp's accumulator tiles of S: rows 16 * mw + {g, g + 8}, columns 8 * n + 2 tq + {0, 1}
    const int mw = warp % MT, nw0 = (warp / MT) * NTW;
    float Sr[NTW][4];
#pragma unroll
    for (int q = 0; q < NTW; ++q)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int i = 16 * mw + g + 8 * (e >> 1), j = 8 * (nw0 + q) + 2 * tq + (e & 1);
            Sr[q][e] = ldst(Sg + static_cast<int64_t>(i) * D + j0 + j);
        }
    for (int x = tid; x < D * VS; x += 256) sS[(x / VS) * LV + x % VS] = ldst(Sg + static_cast<int64_t>(x / VS) * D + j0 + x % VS);
    if (nch > 0) load(0, 0);
    cp_async_commit();

    const int mt = warp >> 2, nt = warp & 3;  // the warp's tile of W and O
    for (int c = 0; c < nch; ++c) {
        if (c + 1 < nch) load(c + 1, (c + 1) & 1);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();  // chunk c landed (every thread's copies); S of the previous chunk written
        const float *st = sm + (c & 1) * G::STAGE;
        const float *sW = st, *sP = sW + C * LA, *sK = sP + C * LA, *sU = sK + C * LK, *sO = sU + C * LV, *sg = sO + C * LV;
        // W = U - Wk S (accumulated as Wk S - U), O = P S + Oi
        const int r0 = 16 * mt + g, cc = 8 * nt + 2 * tq;
        float aw[4] = {-sU[r0 * LV + cc], -sU[r0 * LV + cc + 1], -sU[(r0 + 8) * LV + cc], -sU[(r0 + 8) * LV + cc + 1]};
        float ao[4] = {sO[r0 * LV + cc], sO[r0 * LV + cc + 1], sO[(r0 + 8) * LV + cc], sO[(r0 + 8) * LV + cc + 1]};
#pragma unroll 4
        for (int k0 = 0; k0 < D; k0 += 8) {
            uint32_t wb[4], ws[4], pb[4], ps[4], bb[2], bs[2];
            kda_tf32_split(sW[r0 * LA + k0 + tq], wb[0], ws[0]);
            kda_tf32_split(sW[(r0 + 8) * LA + k0 + tq], wb[1], ws[1]);
            kda_tf32_split(sW[r0 * LA + k0 + tq + 4], wb[2], ws[2]);
            kda_tf32_split(sW[(r0 + 8) * LA + k0 + tq + 4], wb[3], ws[3]);
            kda_tf32_split(sP[r0 * LA + k0 + tq], pb[0], ps[0]);
            kda_tf32_split(sP[(r0 + 8) * LA + k0 + tq], pb[1], ps[1]);
            kda_tf32_split(sP[r0 * LA + k0 + tq + 4], pb[2], ps[2]);
            kda_tf32_split(sP[(r0 + 8) * LA + k0 + tq + 4], pb[3], ps[3]);
            kda_tf32_split(sS[(k0 + tq) * LV + 8 * nt + g], bb[0], bs[0]);
            kda_tf32_split(sS[(k0 + tq + 4) * LV + 8 * nt + g], bb[1], bs[1]);
            kda_mma_3xtf32(aw, wb, ws, bb, bs);
            kda_mma_3xtf32(ao, pb, ps, bb, bs);
        }
        sWv[r0 * LV + cc] = -aw[0], sWv[r0 * LV + cc + 1] = -aw[1];
        sWv[(r0 + 8) * LV + cc] = -aw[2], sWv[(r0 + 8) * LV + cc + 1] = -aw[3];
        const int64_t t0 = static_cast<int64_t>(c) * C, col = static_cast<int64_t>(h) * D + j0 + cc;
        if (t0 + r0 < len) *reinterpret_cast<float2 *>(o + (off + t0 + r0) * P + col) = make_float2(ao[0], ao[1]);
        if (t0 + r0 + 8 < len) *reinterpret_cast<float2 *>(o + (off + t0 + r0 + 8) * P + col) = make_float2(ao[2], ao[3]);
        __syncthreads();  // W complete
        // S = diag(exp(g_C)) S + Kb^T W: the warp's rows, its NTW column tiles
        const int i0 = 16 * mw + g;
        const float e0 = exp2f(sg[i0]), e1 = exp2f(sg[i0 + 8]);
#pragma unroll
        for (int q = 0; q < NTW; ++q) Sr[q][0] *= e0, Sr[q][1] *= e0, Sr[q][2] *= e1, Sr[q][3] *= e1;
#pragma unroll
        for (int k0 = 0; k0 < C; k0 += 8) {
            uint32_t ab[4], as[4];
            kda_tf32_split(sK[(k0 + tq) * LK + i0], ab[0], as[0]);
            kda_tf32_split(sK[(k0 + tq) * LK + i0 + 8], ab[1], as[1]);
            kda_tf32_split(sK[(k0 + tq + 4) * LK + i0], ab[2], as[2]);
            kda_tf32_split(sK[(k0 + tq + 4) * LK + i0 + 8], ab[3], as[3]);
#pragma unroll
            for (int q = 0; q < NTW; ++q) {
                uint32_t bb[2], bs[2];
                kda_tf32_split(sWv[(k0 + tq) * LV + 8 * (nw0 + q) + g], bb[0], bs[0]);
                kda_tf32_split(sWv[(k0 + tq + 4) * LV + 8 * (nw0 + q) + g], bb[1], bs[1]);
                kda_mma_3xtf32(Sr[q], ab, as, bb, bs);
            }
        }
#pragma unroll
        for (int q = 0; q < NTW; ++q) {
            const int j = 8 * (nw0 + q) + 2 * tq;
            sS[i0 * LV + j] = Sr[q][0], sS[i0 * LV + j + 1] = Sr[q][1];
            sS[(i0 + 8) * LV + j] = Sr[q][2], sS[(i0 + 8) * LV + j + 1] = Sr[q][3];
        }
        __syncthreads();  // S written, this chunk's buffers free
    }
#pragma unroll
    for (int q = 0; q < NTW; ++q)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int i = 16 * mw + g + 8 * (e >> 1), j = 8 * (nw0 + q) + 2 * tq + (e & 1);
            stst(Sg + static_cast<int64_t>(i) * D + j0 + j, Sr[q][e]);
        }
}

// Every chunk's matrices, a block per (chunk, head, segment), the products on the tensor cores (3xTF32). The
// in-chunk products L and M carry a decay per channel, exp2(g_t - g_s): between the two halves of
// the chunk (t >= 16 > s) it factors through the first row of the second half,
// exp2(g_t - g_16) * exp2(g_16 - g_s), both factors <= 1 (g never rises), so that block is
// (K1 . e1) (K0 . e0)^T, an MMA; the two diagonal 16 x 16 blocks are summed directly. The forward
// substitution keeps four partial sums per row; P = exp2(g) q - M Wk and Oi = M U are MMAs.
template <int D, int C>
struct KdaPrepTc {
    static constexpr int R = D + 4;   // q, k, v, g rows (A operands by row, B by column: conflict-free)
    static constexpr int RW = D + 8;  // U, Wk rows (B operands of P and Oi)
    static constexpr int RM = C + 4;  // L, M rows (A operands)
    static constexpr int SMEM = (4 * C * R + 2 * C * RW + 2 * C * RM + C) * 4;
};

template <int D, int C, int KH>
__global__ void __launch_bounds__(256)
    kda_chunk_prep_tc_kernel(float *scr, const float *qp, const float *kp, const float *vp,
                             const float *__restrict__ wq, const float *__restrict__ wk, const float *__restrict__ wv,
                             const float *conv, int64_t cstride, int64_t hist, const float *z, const float *braw,
                             const float *__restrict__ a_log, const float *__restrict__ dt_bias, float lb, int softplus,
                             float l2eps, float qscale, const int32_t *__restrict__ seg, int H) {
    using G = KdaPrepTc<D, C>;
    static_assert(C == 32 && D % 64 == 0, "two 16-row halves; 4 warps of P and 4 of Oi tiles");
    constexpr int R = G::R, RW = G::RW, RM = G::RM, NT = 256, HB = C / 2;
    extern __shared__ __align__(16) float sm[];
    float *sq = sm, *sk = sq + C * R, *sv = sk + C * R, *sg = sv + C * R;
    float *sU = sg + C * R, *sW = sU + C * RW, *sL = sW + C * RW, *sM = sL + C * RM, *sb = sM + C * RM;
    const int c = blockIdx.x, h = blockIdx.y, s = blockIdx.z, tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, tq = lane & 3;
    const int64_t slot = seg[3 * s], off = seg[3 * s + 1], len = seg[3 * s + 2], t0 = static_cast<int64_t>(c) * C;
    if (t0 >= len) return;
    const int n = static_cast<int>(min(static_cast<int64_t>(C), len - t0));
    const int64_t P = static_cast<int64_t>(H) * D;
    // kda_fused_kernel's per-token arithmetic: the short convolutions (the segment's first KH
    // tokens reading the slot's history, which kda_conv_hist_kernel replaces after this kernel),
    // SiLU, the decay's log (base 2 here) and beta; then the head's L2 norms below
    const float a = expf(a_log[h]);
    for (int idx = tid; idx < C * D; idx += NT) {
        const int t = idx / D, i = idx % D;
        float q = 0.0f, k = 0.0f, v = 0.0f, g = 0.0f;
        if (t < n) {
            const int64_t ts = t0 + t, ch = static_cast<int64_t>(h) * D + i, row = (off + ts) * P + ch;
            const float *hq = conv + slot * cstride + ch * KH;
            auto cv = [&](const float *xp, const float *w, const float *hp) {
                float acc = w[KH] * xp[row];
#pragma unroll
                for (int j = 0; j < KH; ++j) {
                    const int64_t tt = ts - KH + j;
                    acc += w[j] * (tt >= 0 ? xp[(off + tt) * P + ch] : hp[KH + tt]);
                }
                return acc * sigm(acc);
            };
            q = cv(qp, wq + ch * (KH + 1), hq), k = cv(kp, wk + ch * (KH + 1), hq + hist);
            v = cv(vp, wv + ch * (KH + 1), hq + 2 * hist);
            const float zz = z[row] + dt_bias[ch];
            g = (softplus ? -a * softplusf(zz) : lb * sigm(a * zz)) * kLog2e;
        }
        sq[t * R + i] = q, sk[t * R + i] = k, sv[t * R + i] = v, sg[t * R + i] = g;
    }
    for (int idx = tid; idx < C * RM; idx += NT) sL[idx] = 0.0f, sM[idx] = 0.0f;
    for (int t = tid; t < C; t += NT) sb[t] = t < n ? sigm(braw[(off + t0 + t) * H + h]) : 0.0f;
    __syncthreads();
    for (int t = warp; t < C; t += NT / 32) {  // the head's L2 norms, a warp a row; qscale on q
        float qq = 0.0f, kk = 0.0f;
        for (int i = lane; i < D; i += 32) qq = fmaf(sq[t * R + i], sq[t * R + i], qq), kk = fmaf(sk[t * R + i], sk[t * R + i], kk);
        const float iq = 1.0f / sqrtf(warp_sum(qq) + l2eps), ik = 1.0f / sqrtf(warp_sum(kk) + l2eps);
        for (int i = lane; i < D; i += 32) sq[t * R + i] = (sq[t * R + i] * iq) * qscale, sk[t * R + i] *= ik;
    }
    __syncthreads();
    if (tid < D)  // cumulative log decay within the chunk (padding rows add nothing)
        for (int t = 1; t < C; ++t) sg[t * R + tid] += sg[(t - 1) * R + tid];
    __syncthreads();
    // the diagonal blocks: pairs s <= t within each half, summed directly
    constexpr int NP = HB * (HB + 1) / 2;
    for (int p = tid; p < 2 * NP; p += NT) {
        const int half = p / NP, x = p % NP;
        int tl = static_cast<int>((sqrtf(8.0f * x + 1.0f) - 1.0f) * 0.5f);
        while (tl * (tl + 1) / 2 > x) --tl;
        while ((tl + 1) * (tl + 2) / 2 <= x) ++tl;
        const int t = half * HB + tl, u = half * HB + (x - tl * (tl + 1) / 2);
        float l = 0.0f, m = 0.0f;
        for (int i = 0; i < D; ++i) {
            const float ku = sk[u * R + i] * exp2f(sg[t * R + i] - sg[u * R + i]);
            l = fmaf(sk[t * R + i], ku, l);
            m = fmaf(sq[t * R + i], ku, m);
        }
        if (u < t) sL[t * RM + u] = l;
        sM[t * RM + u] = m;
    }
    // the off-diagonal block (rows HB.., columns ..HB): warps 0-1 L's two 16 x 8 tiles, 2-3 M's
    if (warp < 4) {
        const float *src = warp < 2 ? sk : sq;
        const int nt = warp & 1, ra = HB + g8, cb = 8 * nt + g8;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
        for (int k0 = 0; k0 < D; k0 += 8) {
            const int ia = k0 + tq, ib = k0 + tq + 4;
            const float r0 = sg[HB * R + ia], r1 = sg[HB * R + ib];
            uint32_t ab[4], as[4], bb[2], bs[2];
            kda_tf32_split(src[ra * R + ia] * exp2f(sg[ra * R + ia] - r0), ab[0], as[0]);
            kda_tf32_split(src[(ra + 8) * R + ia] * exp2f(sg[(ra + 8) * R + ia] - r0), ab[1], as[1]);
            kda_tf32_split(src[ra * R + ib] * exp2f(sg[ra * R + ib] - r1), ab[2], as[2]);
            kda_tf32_split(src[(ra + 8) * R + ib] * exp2f(sg[(ra + 8) * R + ib] - r1), ab[3], as[3]);
            kda_tf32_split(sk[cb * R + ia] * exp2f(r0 - sg[cb * R + ia]), bb[0], bs[0]);
            kda_tf32_split(sk[cb * R + ib] * exp2f(r1 - sg[cb * R + ib]), bb[1], bs[1]);
            kda_mma_3xtf32(acc, ab, as, bb, bs);
        }
        float *dst = warp < 2 ? sL : sM;
        const int cc = 8 * nt + 2 * tq;
        dst[ra * RM + cc] = acc[0], dst[ra * RM + cc + 1] = acc[1];
        dst[(ra + 8) * RM + cc] = acc[2], dst[(ra + 8) * RM + cc + 1] = acc[3];
    }
    __syncthreads();
    // forward substitution, a column a thread: U's (v) and Wk's (decayed k); four partial sums a row
    if (tid < 2 * D) {
        const bool isu = tid < D;
        const int d = isu ? tid : tid - D;
        float *X = isu ? sU : sW;
        for (int t = 0; t < C; ++t) {
            float a0 = isu ? sv[t * R + d] : sk[t * R + d] * exp2f(sg[t * R + d]), a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            const float *Lr = sL + t * RM;
            int u = 0;
            for (; u + 3 < t; u += 4) {
                a0 = fmaf(-Lr[u], X[u * RW + d], a0);
                a1 = fmaf(-Lr[u + 1], X[(u + 1) * RW + d], a1);
                a2 = fmaf(-Lr[u + 2], X[(u + 2) * RW + d], a2);
                a3 = fmaf(-Lr[u + 3], X[(u + 3) * RW + d], a3);
            }
            for (; u < t; ++u) a0 = fmaf(-Lr[u], X[u * RW + d], a0);
            X[t * RW + d] = sb[t] * ((a0 + a1) + (a2 + a3));
        }
    }
    __syncthreads();
    float *base = scr + ((kda_chunk_base<C>(seg, s) + c) * H + h) * (5 * C * D + D);
    float *gU = base, *gW = base + C * D, *gP = base + 2 * C * D, *gO = base + 3 * C * D, *gK = base + 4 * C * D,
          *gC = base + 5 * C * D;
    // P = exp2(g) q - M Wk (warps 0-3, accumulated as M Wk - exp2(g) q) and Oi = M U (warps 4-7):
    // each warp D/16 tiles of 16 x 8 sharing their rows, so one A fragment of M a k step
    {
        constexpr int NTL = D / 8, TPW = D / 16;
        const bool isp = warp < 4;
        const float *B = isp ? sW : sU;
        float *out = isp ? gP : gO;
        const int idx0 = (warp & 3) * TPW, mt = idx0 / NTL, r0 = 16 * mt + g8;
        float acc[TPW][4];
#pragma unroll
        for (int q = 0; q < TPW; ++q) {
            const int cc = 8 * ((idx0 + q) % NTL) + 2 * tq;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int t = r0 + 8 * (e >> 1), d = cc + (e & 1);
                acc[q][e] = isp ? -sq[t * R + d] * exp2f(sg[t * R + d]) : 0.0f;
            }
        }
#pragma unroll
        for (int k0 = 0; k0 < C; k0 += 8) {
            uint32_t ab[4], as[4];
            kda_tf32_split(sM[r0 * RM + k0 + tq], ab[0], as[0]);
            kda_tf32_split(sM[(r0 + 8) * RM + k0 + tq], ab[1], as[1]);
            kda_tf32_split(sM[r0 * RM + k0 + tq + 4], ab[2], as[2]);
            kda_tf32_split(sM[(r0 + 8) * RM + k0 + tq + 4], ab[3], as[3]);
#pragma unroll
            for (int q = 0; q < TPW; ++q) {
                const int nc = 8 * ((idx0 + q) % NTL) + g8;
                uint32_t bb[2], bs[2];
                kda_tf32_split(B[(k0 + tq) * RW + nc], bb[0], bs[0]);
                kda_tf32_split(B[(k0 + tq + 4) * RW + nc], bb[1], bs[1]);
                kda_mma_3xtf32(acc[q], ab, as, bb, bs);
            }
        }
        const float sgn = isp ? -1.0f : 1.0f;
#pragma unroll
        for (int q = 0; q < TPW; ++q) {
            const int cc = 8 * ((idx0 + q) % NTL) + 2 * tq;
            *reinterpret_cast<float2 *>(out + r0 * D + cc) = make_float2(sgn * acc[q][0], sgn * acc[q][1]);
            *reinterpret_cast<float2 *>(out + (r0 + 8) * D + cc) = make_float2(sgn * acc[q][2], sgn * acc[q][3]);
        }
    }
    for (int idx = tid; idx < C * D; idx += NT) {
        const int t = idx / D, d = idx % D;
        gU[idx] = sU[t * RW + d], gW[idx] = sW[t * RW + d];
        gK[idx] = sk[t * R + d] * exp2f(sg[(C - 1) * R + d] - sg[t * R + d]);
    }
    for (int d = tid; d < D; d += NT) gC[d] = sg[(C - 1) * R + d];
}

// kda_out_kernel over the segments' tokens only (the other rows belong to kda_fused): a warp a
// (token, head): head-wise RMSNorm, then the output gate.
__global__ void kda_out_seg_kernel(float *o, const float *gate, const float *__restrict__ w,
                                   const int32_t *__restrict__ seg, int64_t H, int64_t D, float eps) {
    const int s = blockIdx.y;
    const int64_t off = seg[3 * s + 1], len = seg[3 * s + 2];
    const int64_t vec = static_cast<int64_t>(blockIdx.x) * 8 + (threadIdx.x >> 5);
    if (vec >= len * H) return;
    const int lane = threadIdx.x & 31;
    const int64_t t = vec / H, h = vec % H, base = ((off + t) * H + h) * D;
    float *p = o + base;
    const float *g = gate + base;
    float ss = 0.0f;
    for (int64_t i = lane; i < D; i += 32) ss = fmaf(p[i], p[i], ss);
    ss = warp_sum(ss);
    const float inv = 1.0f / sqrtf(ss / static_cast<float>(D) + eps);
    for (int64_t i = lane; i < D; i += 32) p[i] = (w[i] * (p[i] * inv)) * sigm(g[i]);
}

}  // namespace
}  // namespace fullm
