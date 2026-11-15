// SPDX-License-Identifier: Apache-2.0
// kernels/mla.cuh - gated MLA (absorbed): the latent append, q absorption, the latent attention and
// its combine, unabsorption.
//
// Part of cuda_backend.cu's translation unit, included there after the CUDA and standard
// headers, in the order common, formats, gemv, convert, router, elementwise, copy, kimi, mla,
// attention, sampling, moe (each file uses only those before it). Kernels and their device
// helpers: no state.
#pragma once

namespace fullm {
namespace {

// ------------------------------------------------------------------ Gated MLA (absorbed)

template <typename KT>
__global__ void mla_append_kernel(KT *kv, const float *ckv, const int32_t *__restrict__ tok_slot,
                                  const int32_t *__restrict__ tok_pos, const int32_t *__restrict__ pt,
                                  int64_t max_pages, int64_t page, int64_t width, int64_t N) {
    const int64_t total = N * width;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t t = i / width, e = i % width;
        const int64_t p = tok_pos[t];
        const int64_t pg = pt[static_cast<int64_t>(tok_slot[t]) * max_pages + p / page];
        stst(kv + (pg * page + p % page) * width + e, ckv[i]);
    }
}

constexpr int kTokTile = 8;

// q_abs[t][h][j] = sum_{i<qn} q[t][h][i] * W_UK^h[i][j]. grid = (head, 64-column block,
// 8-token tile); the 8 warps split i into slices, each lane owns columns j and j+32, and the
// slices are summed in warp order. W_UK^h is read once per 8 tokens. A token's result does
// not depend on the other tokens of the call.
constexpr int kAbsorbCols = 64;

__global__ void __launch_bounds__(256)
    mla_absorb_kernel(float *q_abs, const float *q, const uint16_t *__restrict__ kvb, int64_t N, int H, int qn,
                      int qr, int vh, int L) {
    extern __shared__ float sm[];  // q: [kTokTile][qn], then red: [8][kTokTile][kAbsorbCols]
    float *red = sm + kTokTile * qn;
    const int h = blockIdx.x;
    const int j0 = blockIdx.y * kAbsorbCols;
    const int64_t t0 = static_cast<int64_t>(blockIdx.z) * kTokTile;
    const int nt = static_cast<int>(imin(kTokTile, N - t0));
    const int qh = qn + qr, W = L + qr;
    for (int idx = threadIdx.x; idx < kTokTile * qn; idx += blockDim.x) {
        const int tt = idx / qn, i = idx % qn;
        sm[idx] = tt < nt ? q[((t0 + tt) * H + h) * qh + i] : 0.0f;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int slice = (qn + 7) / 8;
    const int i0 = warp * slice, i1 = min(qn, i0 + slice);
    const int ja = j0 + lane, jb = j0 + lane + 32;
    const uint16_t *wk = kvb + static_cast<int64_t>(h) * (qn + vh) * L;
    float acc[kTokTile][2];
#pragma unroll
    for (int tt = 0; tt < kTokTile; ++tt) acc[tt][0] = acc[tt][1] = 0.0f;
    for (int i = i0; i < i1; ++i) {
        const float wa = ja < L ? bf2f(__ldg(wk + static_cast<int64_t>(i) * L + ja)) : 0.0f;
        const float wb = jb < L ? bf2f(__ldg(wk + static_cast<int64_t>(i) * L + jb)) : 0.0f;
#pragma unroll
        for (int tt = 0; tt < kTokTile; ++tt) {
            const float qv = sm[tt * qn + i];
            acc[tt][0] = fmaf(qv, wa, acc[tt][0]);
            acc[tt][1] = fmaf(qv, wb, acc[tt][1]);
        }
    }
#pragma unroll
    for (int tt = 0; tt < kTokTile; ++tt) {
        red[(warp * kTokTile + tt) * kAbsorbCols + lane] = acc[tt][0];
        red[(warp * kTokTile + tt) * kAbsorbCols + lane + 32] = acc[tt][1];
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < nt * kAbsorbCols; idx += blockDim.x) {
        const int tt = idx / kAbsorbCols, c = idx % kAbsorbCols;
        if (j0 + c >= L) continue;
        float v = 0.0f;
        for (int w = 0; w < 8; ++w) v += red[(w * kTokTile + tt) * kAbsorbCols + c];
        q_abs[((t0 + tt) * H + h) * W + j0 + c] = v;
    }
    if (blockIdx.y == 0)
        for (int idx = threadIdx.x; idx < nt * qr; idx += blockDim.x) {
            const int tt = idx / qr, r = idx % qr;
            q_abs[((t0 + tt) * H + h) * W + L + r] = q[((t0 + tt) * H + h) * qh + qn + r];
        }
}

// y[t][h*vh + j] = W_UV^h[j] . o_lat[t][h] for steps above kGemvMaxRows tokens (smaller steps
// run the batched GEMV). One warp per output row, 8 tokens per block.
__global__ void mla_unabsorb_kernel(float *y, const float *o_lat, const uint16_t *__restrict__ kvb, int64_t N,
                                    int H, int qn, int vh, int L) {
    extern __shared__ float sm[];  // [kTokTile][L]
    const int h = blockIdx.x;
    const int64_t t0 = static_cast<int64_t>(blockIdx.y) * kTokTile;
    const int nt = static_cast<int>(imin(kTokTile, N - t0));
    for (int idx = threadIdx.x; idx < kTokTile * L; idx += blockDim.x) {
        const int tt = idx / L, l = idx % L;
        sm[idx] = tt < nt ? o_lat[((t0 + tt) * H + h) * L + l] : 0.0f;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    for (int jr = warp; jr < vh; jr += nw) {
        const uint16_t *wr = kvb + (static_cast<int64_t>(h) * (qn + vh) + qn + jr) * L;
        float acc[kTokTile];
#pragma unroll
        for (int tt = 0; tt < kTokTile; ++tt) acc[tt] = 0.0f;
        for (int l = lane; l < L; l += 32) {
            const float wv = bf2f(wr[l]);
#pragma unroll
            for (int tt = 0; tt < kTokTile; ++tt) acc[tt] = fmaf(wv, sm[tt * L + l], acc[tt]);
        }
#pragma unroll
        for (int tt = 0; tt < kTokTile; ++tt) {
            const float v = warp_sum(acc[tt]);
            if (lane == 0 && tt < nt) y[(t0 + tt) * H * vh + static_cast<int64_t>(h) * vh + jr] = v;
        }
    }
}

constexpr int kAttnHeads = 16;      // warps (heads) per block; they share each staged tile
constexpr int kAttnTile = 16;       // positions per shared-memory tile (one per warp when staging)
constexpr int kAttnMaxW = 1024;     // q/o register arrays hold at most 32 floats per lane
constexpr int kAttnSplitLen = 32;   // shortest split, in positions (a multiple of kAttnTile)
constexpr int kAttnMaxSplits = 64;  // splits per token when the step has <= kGemvMaxRows tokens
static_assert(kAttnHeads == kAttnTile, "staging assigns one position per warp");

// How token t's context of ctx = pos + 1 positions is split: S positions per split (at least
// min_len, a whole number of tiles), nsplit = ceil(ctx / S) <= max_splits. A function of the
// token's own position, max_splits and min_len only, so with those fixed (every step of at most
// kGemvMaxRows tokens) a token's result does not depend on the rest of the batch.
__host__ __device__ __forceinline__ void attn_split(int64_t ctx, int max_splits, int64_t &S, int &nsplit,
                                                    int64_t min_len = kAttnSplitLen) {
    int64_t len = (ctx + max_splits - 1) / max_splits;
    len = (len + kAttnTile - 1) / kAttnTile * kAttnTile;
    S = len > min_len ? len : min_len;
    nsplit = static_cast<int>((ctx + S - 1) / S);
}

// Flash-decoding over the paged latent cache. grid = (token, head group, split). Split sp of
// token t covers positions [sp*S, min((sp+1)*S, pos+1)); blocks past the token's nsplit exit.
// A token with one split writes its normalised result to out; otherwise (o_unnormalised, m,
// l) go to part[t][h][sp] (zsplits = gridDim.z slots per head) and mla_combine_kernel merges
// them in split order. Each warp scores 4 positions at a time: four independent dot
// products and butterflies, then one online-softmax update.
template <typename KT, int RW>
__global__ void __launch_bounds__(kAttnHeads * 32)
    mla_attn_kernel(float *out, float *part, const float *q_abs, const KT *__restrict__ kv,
                    const int32_t *__restrict__ tok_slot, const int32_t *__restrict__ tok_pos,
                    const int32_t *__restrict__ pt, int64_t max_pages, int64_t page, int H, int L, int W, float scale,
                    int max_splits) {
    extern __shared__ float tile[];  // [kAttnTile][W]
    const int64_t t = blockIdx.x;
    const int sp = blockIdx.z;
    const int64_t pos = tok_pos[t];
    int64_t S;
    int nsplit;
    attn_split(pos + 1, max_splits, S, nsplit);
    if (sp >= nsplit) return;  // block-uniform, before any barrier
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = blockIdx.y * kAttnHeads + warp;
    const bool active = h < H;
    const int32_t *ptab = pt + static_cast<int64_t>(tok_slot[t]) * max_pages;
    const int64_t start = static_cast<int64_t>(sp) * S;
    const int64_t end = imin(start + S, pos + 1);

    float qreg[RW], oreg[RW];
#pragma unroll
    for (int c = 0; c < RW; ++c) {
        const int d = lane + 32 * c;
        qreg[c] = (active && d < W) ? q_abs[(t * H + h) * W + d] : 0.0f;
        oreg[c] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;

    for (int64_t p0 = start; p0 < end; p0 += kAttnTile) {  // uniform across the block
        const int np = static_cast<int>(imin(kAttnTile, end - p0));
        __syncthreads();
        if (warp < np) {
            const int64_t p = p0 + warp;
            const KT *row = kv + (static_cast<int64_t>(ptab[p / page]) * page + p % page) * W;
            float *dst = tile + warp * W;
            for (int e = lane; e < W; e += 32) dst[e] = ldst(row + e);
        }
        __syncthreads();
        if (!active) continue;
        for (int g = 0; g < np; g += 4) {
            const int nv = min(4, np - g);  // rows past np hold stale data: computed, never used
            float d[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float *row = tile + (g + i) * W;
                float a = 0.0f;
#pragma unroll
                for (int c = 0; c < RW; ++c) {
                    const int dd = lane + 32 * c;
                    if (dd < W) a = fmaf(qreg[c], row[dd], a);
                }
                d[i] = a;
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)
#pragma unroll
                for (int i = 0; i < 4; ++i) d[i] += __shfl_xor_sync(0xffffffffu, d[i], o);
            float mn = m;
#pragma unroll
            for (int i = 0; i < 4; ++i)
                if (i < nv) mn = fmaxf(mn, d[i] * scale);
            const float corr = expf(m - mn);  // m = -inf before the first position: 0
            float pe[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) pe[i] = i < nv ? expf(d[i] * scale - mn) : 0.0f;
            l = l * corr + pe[0] + pe[1] + pe[2] + pe[3];
#pragma unroll
            for (int c = 0; c < RW; ++c) {
                const int dd = lane + 32 * c;
                if (dd < L) {
                    float o = oreg[c] * corr;
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        if (i < nv) o = fmaf(pe[i], tile[(g + i) * W + dd], o);
                    oreg[c] = o;
                }
            }
            m = mn;
        }
    }
    if (!active) return;
    if (nsplit == 1) {
        const float inv = 1.0f / l;  // the only split starts at position 0, so l > 0
#pragma unroll
        for (int c = 0; c < RW; ++c) {
            const int dd = lane + 32 * c;
            if (dd < L) out[(t * H + h) * L + dd] = oreg[c] * inv;
        }
    } else {
        float *pp = part + ((t * H + h) * gridDim.z + sp) * static_cast<int64_t>(L + 2);
#pragma unroll
        for (int c = 0; c < RW; ++c) {
            const int dd = lane + 32 * c;
            if (dd < L) pp[dd] = oreg[c];
        }
        if (lane == 0) {
            pp[L] = m;
            pp[L + 1] = l;
        }
    }
}

// One block per (token, head) of a token with more than one split; merges in split order.
// Every split holds at least one position, so every m is finite.
__global__ void mla_combine_kernel(float *out, const float *part, const int32_t *__restrict__ tok_pos, int H, int L,
                                   int max_splits, int zsplits, int min_len) {
    __shared__ float f[kAttnMaxSplits + 1];
    pdl_wait();
    pdl_trigger();
    const int64_t th = blockIdx.x;  // t*H + h
    int64_t S;
    int nsplit;
    attn_split(static_cast<int64_t>(tok_pos[th / H]) + 1, max_splits, S, nsplit, min_len);
    if (nsplit == 1) return;  // the attention kernel wrote it
    const int64_t sl = L + 2;
    const float *base = part + th * zsplits * sl;
    if (threadIdx.x == 0) {
        float M = -INFINITY;
        for (int s = 0; s < nsplit; ++s) M = fmaxf(M, base[s * sl + L]);
        float lt = 0.0f;
        for (int s = 0; s < nsplit; ++s) {
            const float fs = expf(base[s * sl + L] - M);
            f[s] = fs;
            lt += base[s * sl + L + 1] * fs;
        }
        f[kAttnMaxSplits] = 1.0f / lt;
    }
    __syncthreads();
    const float inv = f[kAttnMaxSplits];
    for (int d = threadIdx.x; d < L; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < nsplit; ++s) acc = fmaf(base[s * sl + d], f[s], acc);
        out[th * L + d] = acc * inv;
    }
}

// ------------------------------------------------------------------ MLA prefill, un-absorbed
// meta (Backend::MlaPrefill): cu_q [n + 1], cu_k [n + 1], slot [n], first row [n]. The segment
// holding compact row r of a prefix-sum array cu (n + 1 entries, cu[0] = 0): the last i with
// cu[i] <= r.
__device__ __forceinline__ int mla_seg_of(const int32_t *cu, int n, int64_t r) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (cu[mid] <= r) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

// Position pos of a slot from the paged latent cache to key row j: its latent to lat16 [][L] and
// its rope slot to rope16 [][qr], as bf16 (the block's threads over the row).
template <typename KT>
__device__ __forceinline__ void mla_latent_row(uint16_t *lat16, uint16_t *rope16, const KT *__restrict__ kv, int64_t j,
                                               int64_t slot, int64_t pos, const int32_t *__restrict__ pt,
                                               int64_t max_pages, int64_t page, int L, int qr) {
    const int64_t pg = pt[slot * max_pages + pos / page];
    const KT *row = kv + (pg * page + pos % page) * (L + qr);
    for (int e = threadIdx.x; e < L + qr; e += blockDim.x) {
        const uint16_t v = f2bf(ldst(row + e));
        if (e < L) lat16[j * L + e] = v;
        else rope16[j * qr + (e - L)] = v;
    }
}

// Key row j (of total_k, the segments' contexts back to back): position j - cu_k[i] of segment
// i's slot. A block per key row.
template <typename KT>
__global__ void mla_latent_rows_kernel(uint16_t *lat16, uint16_t *rope16, const KT *__restrict__ kv,
                                       const int32_t *__restrict__ meta, int n, const int32_t *__restrict__ pt,
                                       int64_t max_pages, int64_t page, int L, int qr) {
    const int64_t j = blockIdx.x;
    const int32_t *cu_k = meta + n + 1, *slot = meta + 2 * (n + 1);
    const int i = mla_seg_of(cu_k, n, j);
    mla_latent_row(lat16, rope16, kv, j, slot[i], j - cu_k[i], pt, max_pages, page, L, qr);
}

// One chunk of a segment's keys (mla_prefill's chunked path): key row j is position pos0 + j of
// the slot. Block 0 also writes the chunk's meta for the kernels after it, one segment's worth:
// cu_q {0, nq}, cu_k {0, nk}, slot, first row.
template <typename KT>
__global__ void mla_latent_span_kernel(uint16_t *lat16, uint16_t *rope16, const KT *__restrict__ kv, int32_t *meta,
                                       int slot, int64_t pos0, int nq, int nk, int first,
                                       const int32_t *__restrict__ pt, int64_t max_pages, int64_t page, int L, int qr) {
    const int64_t j = blockIdx.x;
    if (j == 0 && threadIdx.x == 0) {
        meta[0] = 0, meta[1] = nq, meta[2] = 0, meta[3] = nk, meta[4] = slot, meta[5] = first;
    }
    mla_latent_row(lat16, rope16, kv, j, slot, pos0 + j, pt, max_pages, page, L, qr);
}

// The rope slot, shared by the heads, into each head's key after its qn expanded channels:
// k[j][h][qn + r] = rope16[j][r] for the hg heads of a pass (k rows of hg * (qn + qr)).
__global__ void mla_rope_keys_kernel(uint16_t *k, const uint16_t *__restrict__ rope16, int64_t total_k, int hg,
                                     int qn, int qr) {
    const int64_t total = total_k * hg * qr;
    for (int64_t x = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; x < total;
         x += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t j = x / (hg * qr);
        const int h = static_cast<int>((x / qr) % hg), r = static_cast<int>(x % qr);
        k[(j * hg + h) * (qn + qr) + qn + r] = rope16[j * qr + r];
    }
}

// Compact row r of the segments' queries <-> step row first[i] + (r - cu_q[i]), columns [col0,
// col0 + width) of rows `stride` wide (a pass's heads): q gathered as bf16, the bf16 output
// scattered back as fp32. A block per row.
__global__ void mla_q_rows_kernel(uint16_t *q16, const float *__restrict__ q, const int32_t *__restrict__ meta,
                                  int n, int64_t stride, int64_t col0, int64_t width) {
    const int64_t r = blockIdx.x;
    const int i = mla_seg_of(meta, n, r);
    const int64_t row = meta[3 * (n + 1) - 1 + i] + (r - meta[i]);  // first[i]: after cu_q, cu_k, slot
    for (int64_t e = threadIdx.x; e < width; e += blockDim.x) q16[r * width + e] = f2bf(q[row * stride + col0 + e]);
}
__global__ void mla_o_rows_kernel(float *y, const uint16_t *__restrict__ o16, const int32_t *__restrict__ meta, int n,
                                  int64_t stride, int64_t col0, int64_t width) {
    const int64_t r = blockIdx.x;
    const int i = mla_seg_of(meta, n, r);
    const int64_t row = meta[3 * (n + 1) - 1 + i] + (r - meta[i]);
    for (int64_t e = threadIdx.x; e < width; e += blockDim.x) y[row * stride + col0 + e] = bf2f(o16[r * width + e]);
}

// The chunked path's merge: a chunk's attention over some of the keys (o16 [nq][hg * vh] bf16,
// each row normalised over the chunk's keys, and its log-sum-exp lse [hg][nq]) into the rows of
// one segment (meta: one segment's worth) in y's columns [col0, col0 + hg * vh), which hold the
// merge of the chunks before it, normalised, with their log-sum-exp in acc [hg][nq]:
// y = (y e^acc + o e^lse) / (e^acc + e^lse), acc = log(e^acc + e^lse). first: y = o, acc = lse.
// A block per row.
__global__ void mla_lse_merge_kernel(float *y, const uint16_t *__restrict__ o16, const float *__restrict__ lse,
                                     float *acc, const int32_t *__restrict__ meta, int64_t stride, int64_t col0, int hg,
                                     int vh, int nq, int first) {
    __shared__ float wy[256], wo[256];
    const int64_t r = blockIdx.x, row = meta[5] + r;
    for (int h = threadIdx.x; h < hg; h += blockDim.x) {
        const float ln = lse[static_cast<int64_t>(h) * nq + r];
        float *a = acc + static_cast<int64_t>(h) * nq + r;
        if (first) {
            wy[h] = 0.0f, wo[h] = 1.0f, *a = ln;
            continue;
        }
        const float lo = *a, mx = fmaxf(lo, ln);
        if (mx == -INFINITY) {  // neither saw a key
            wy[h] = 1.0f, wo[h] = 0.0f;
            continue;
        }
        const float eo = expf(lo - mx), en = expf(ln - mx), s = eo + en;
        wy[h] = eo / s, wo[h] = en / s, *a = mx + logf(s);
    }
    __syncthreads();
    float *yr = y + row * stride + col0;
    const uint16_t *orow = o16 + r * hg * vh;
    for (int e = threadIdx.x; e < hg * vh; e += blockDim.x) {
        const int h = e / vh;
        const float o = bf2f(orow[e]);
        yr[e] = first ? o : fmaf(yr[e], wy[h], o * wo[h]);
    }
}

// gather: dst[i][c] = src[rows[i]][c]; scatter: dst[rows[i]][c] = src[i][c]. A block per row.
__global__ void index_rows_kernel(float *dst, const float *src, const int32_t *__restrict__ rows, int64_t n,
                                  int scatter) {
    const int64_t i = blockIdx.x, r = rows[i];
    const float *s = src + (scatter ? i : r) * n;
    float *d = dst + (scatter ? r : i) * n;
    for (int64_t c = threadIdx.x; c < n; c += blockDim.x) d[c] = s[c];
}

}  // namespace
}  // namespace fullm
