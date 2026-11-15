// SPDX-License-Identifier: Apache-2.0
// kernels/sampling.cuh - argmax over the logits rows, and draws (sample_rows).
//
// Part of cuda_backend.cu's translation unit, included there after the CUDA and standard
// headers, in the order common, formats, gemv, convert, router, elementwise, copy, kimi, mla,
// attention, sampling, moe (each file uses only those before it). Kernels and their device
// helpers: no state.
#pragma once

namespace fullm {
namespace {

// ------------------------------------------------------------------ sampling

// Pass 1, grid (row, slice): each block finds the first index of the largest value of its
// slice of the row (larger value, else smaller index; NaN skipped); any non-finite sets *bad.
// Pass 2, one block per row: the same merge over the slices in order.
constexpr int kArgmaxSlices = 64;

__device__ __forceinline__ void argmax_merge(float &v, int64_t &i, float ov, int64_t oi) {
    if (oi >= 0 && (i < 0 || ov > v || (ov == v && oi < i))) v = ov, i = oi;
}

__device__ void argmax_block(float &bv, int64_t &bi, float *sv, int64_t *si) {
    sv[threadIdx.x] = bv;
    si[threadIdx.x] = bi;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) argmax_merge(sv[threadIdx.x], si[threadIdx.x], sv[threadIdx.x + o], si[threadIdx.x + o]);
        __syncthreads();
    }
    bv = sv[0];
    bi = si[0];
}

__global__ void argmax_slices_kernel(float *pv, int64_t *pi, int32_t *bad, const float *x, int64_t n) {
    __shared__ float sv[kThreads];
    __shared__ int64_t si[kThreads];
    const float *v = x + static_cast<int64_t>(blockIdx.x) * n;
    const int64_t len = (n + kArgmaxSlices - 1) / kArgmaxSlices;
    const int64_t lo = blockIdx.y * len, hi = imin(n, lo + len);
    float bv = -INFINITY;
    int64_t bi = -1;
    bool nonfinite = false;
    for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const float f = v[i];
        nonfinite |= !isfinite(f);
        if (!isnan(f) && (bi < 0 || f > bv)) bv = f, bi = i;
    }
    if (nonfinite) *bad = 1;
    argmax_block(bv, bi, sv, si);
    if (threadIdx.x == 0) {
        pv[blockIdx.x * kArgmaxSlices + blockIdx.y] = bv;
        pi[blockIdx.x * kArgmaxSlices + blockIdx.y] = bi;
    }
}

__global__ void argmax_rows_kernel(int32_t *idx, const float *pv, const int64_t *pi) {
    __shared__ float sv[kArgmaxSlices];
    __shared__ int64_t si[kArgmaxSlices];
    float bv = pv[blockIdx.x * kArgmaxSlices + threadIdx.x];
    int64_t bi = pi[blockIdx.x * kArgmaxSlices + threadIdx.x];
    argmax_block(bv, bi, sv, si);
    if (threadIdx.x == 0) idx[blockIdx.x] = static_cast<int32_t>(bi < 0 ? 0 : bi);
}

// ------------------------------------------------------------------ masks
// Structured output, grid (row, slice): a row with a mask gets -inf at every token the mask does
// not allow; any non-finite value it held sets *bad. Rows without one return at once.
__global__ void mask_rows_kernel(float *x, int32_t *bad, int64_t n, const SampleRow *params, const uint32_t *masks,
                                 int64_t words) {
    const int32_t k = params[blockIdx.x].mask;
    if (k <= 0) return;
    const uint32_t *m = masks + int64_t(k - 1) * words;
    float *v = x + static_cast<int64_t>(blockIdx.x) * n;
    const int64_t len = (n + kArgmaxSlices - 1) / kArgmaxSlices;
    const int64_t lo = blockIdx.y * len, hi = imin(n, lo + len);
    bool nonfinite = false;
    for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const float f = v[i];
        nonfinite |= !isfinite(f);
        if (!(m[i >> 5] >> (i & 31) & 1u)) v[i] = -INFINITY;
    }
    if (nonfinite) *bad = 1;
}

// ------------------------------------------------------------------ draws
// One block of kSampleThreads per row; sample_row_host (backend.cpp) is the definition this
// matches. Warp w owns the contiguous range [w*span, (w+1)*span) of the row and its lanes read it
// 32 tokens at a time (coalesced); every reduction runs in a fixed order (warp butterflies, then
// the 32 warp partials in warp order), so a row's draw is the same whatever else the launch holds.
// Top-k: the k-th largest key by a radix select (integer counts). The candidates then go to
// shared memory, in token order, when they are few: the top-k set when k <= kSampleCap, or, for
// top-p alone, the kSampleCap largest values when they hold top_p of the mass (the nucleus is then
// among them); top-p's binary search and the draw run there. Otherwise top-p's threshold is
// found over the whole row 4 bits a pass, and the draw walks the row. 1/T and exp in double, the
// weights stored as float, sums in double: as the host does.
constexpr int kSampleThreads = 1024;
constexpr int kSampleWarps = kSampleThreads / 32;
constexpr int kSampleCap = 1024;   // candidate sets drawn in shared memory
constexpr int kSampleRoom = 2048;  // ... with room for ties at their threshold

__device__ __forceinline__ uint32_t sample_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

struct SamplePcg32 {
    uint64_t state = 0, inc;
    __device__ SamplePcg32(uint64_t seed, uint64_t stream) : inc((stream << 1) | 1) {
        next();
        state += seed;
        next();
    }
    __device__ uint32_t next() {
        const uint64_t old = state;
        state = old * 6364136223846793005ull + inc;
        const uint32_t x = static_cast<uint32_t>(((old >> 18) ^ old) >> 27), rot = static_cast<uint32_t>(old >> 59);
        return (x >> rot) | (x << ((32 - rot) & 31));
    }
    __device__ double next_f64() {
        const uint64_t hi = uint64_t(next()) << 21, lo = next() >> 11;
        return double(hi | lo) / double(1ull << 53);
    }
};

// every lane gets the same sum (a+b == b+a exactly, so the butterfly agrees across lanes)
__device__ __forceinline__ double sample_warp_sum(double v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// the block's sum: per warp, then every warp adds the 32 partials in warp order
__device__ double sample_block_sum(double v, double *sw) {
    v = sample_warp_sum(v);
    if ((threadIdx.x & 31) == 0) sw[threadIdx.x >> 5] = v;
    __syncthreads();
    const double t = sample_warp_sum(sw[threadIdx.x & 31]);
    __syncthreads();
    return t;
}

// the k-th largest key of the row, 8 bits at a time from the top
__device__ uint32_t sample_select(const float *v, int64_t w0, int64_t w1, int lane, uint32_t k, uint32_t *hist,
                                  uint32_t *bc) {
    uint32_t prefix = 0, mask = 0, rem = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int d = threadIdx.x; d < 256; d += blockDim.x) hist[d] = 0;
        __syncthreads();
        for (int64_t i = w0 + lane; i < w1; i += 32) {
            const uint32_t key = sample_key(v[i]);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            uint32_t above = 0;
            for (int d = 255; d >= 0; --d) {
                if (above + hist[d] >= rem) {
                    prefix |= uint32_t(d) << shift;
                    rem -= above;
                    break;
                }
                above += hist[d];
            }
            bc[0] = prefix;
            bc[1] = rem;
        }
        __syncthreads();
        prefix = bc[0];
        rem = bc[1];
        mask |= 255u << shift;
    }
    return prefix;
}

__global__ void __launch_bounds__(kSampleThreads) sample_rows_kernel(int32_t *idx, int32_t *bad, const float *x,
                                                                     int64_t n, const SampleRow *params, float *wbuf) {
    __shared__ double sw[kSampleWarps];
    __shared__ double sb[kSampleWarps][17];
    __shared__ float smv[kSampleWarps];
    __shared__ int64_t smi[kSampleWarps];
    __shared__ uint32_t hist[256], bc[2], scnt[kSampleWarps];
    __shared__ int32_t cidx[kSampleRoom];
    __shared__ uint32_t ckey[kSampleRoom];
    __shared__ float cw[kSampleRoom];
    __shared__ int s_done;
    const int64_t row = blockIdx.x;
    const float *v = x + row * n;
    float *w = wbuf + row * n;
    const SampleRow p = params[row];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t span = (n + kSampleWarps - 1) / kSampleWarps;
    const int64_t w0 = warp * span < n ? warp * span : n, w1 = w0 + span < n ? w0 + span : n;

    // the argmax (first index of the largest), and any non-finite value (a masked row's -inf
    // are its masked tokens: mask_rows checked the values)
    float bv = 0.0f;
    int64_t bi = -1;
    bool nonfinite = false;
    const bool masked = p.mask > 0 || p.adj_n > 0;
    for (int64_t i = w0 + lane; i < w1; i += 32) {
        const float f = v[i];
        nonfinite |= !isfinite(f) && !(masked && f == -INFINITY);
        if (!isnan(f) && (bi < 0 || f > bv)) bv = f, bi = i;
    }
    if (nonfinite) *bad = 1;
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int64_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
        argmax_merge(bv, bi, ov, oi);
    }
    if (lane == 0) smv[warp] = bv, smi[warp] = bi;
    __syncthreads();
    bv = smv[lane];
    bi = smi[lane];
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int64_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
        argmax_merge(bv, bi, ov, oi);
    }
    const int64_t best = bi < 0 ? 0 : bi;
    // greedy; or a row with nothing allowed (every logit -inf): its first token
    if (!(p.temperature > 0.0f) || n == 1 || bv == -INFINITY) {
        if (threadIdx.x == 0) idx[row] = static_cast<int32_t>(best);
        return;
    }
    const float m = bv;
    const float inv_t = static_cast<float>(1.0 / static_cast<double>(p.temperature));
    const uint32_t kmax = sample_key(m);
    const bool topk = p.top_k > 0 && p.top_k < n, topp = p.top_p < 1.0f;
    const bool small_k = topk && p.top_k <= kSampleCap;

    // lo: the candidates' threshold, top-k's and min-p's (0: none); the compact set holds the keys
    // at or above lo_c
    const uint32_t lo_m = p.min_p_cut < 0.0f ? sample_key(m + p.min_p_cut) : 0u;
    uint32_t lo = lo_m, lo_c = 0;
    bool compact = false;
    if (small_k) {
        lo = lo_c = max(sample_select(v, w0, w1, lane, static_cast<uint32_t>(p.top_k), hist, bc), lo_m);
        compact = true;
    } else if (topk) {
        lo = max(sample_select(v, w0, w1, lane, static_cast<uint32_t>(p.top_k), hist, bc), lo_m);
    } else if (topp) {
        lo_c = max(n > kSampleCap ? sample_select(v, w0, w1, lane, kSampleCap, hist, bc) : 0u, lo_m);
        compact = true;
    }
    // the candidates' weights (stored for the whole-row path) and their mass
    auto weigh = [&]() {
        double s = 0.0;
        for (int64_t i = w0 + lane; i < w1; i += 32) {
            const float f = v[i];
            const float wi =
                sample_key(f) >= lo ? static_cast<float>(exp(static_cast<double>((f - m) * inv_t))) : 0.0f;
            w[i] = wi;
            s += wi;
        }
        return sample_block_sum(s, sw);
    };
    double total = 0.0;
    bool weighed = false;
    if (!small_k) total = weigh(), weighed = true;

    if (compact) {
        // the compact set in token order: each warp's count, then its tokens at its offset
        uint32_t cnt = 0;
        for (int64_t b = w0; b < w1; b += 32) {
            const int64_t i = b + lane;
            cnt += __popc(__ballot_sync(0xffffffffu, i < w1 && sample_key(v[i]) >= lo_c));
        }
        if (lane == 0) scnt[warp] = cnt;
        __syncthreads();
        uint32_t off = 0, C = 0;
        for (int k = 0; k < kSampleWarps; ++k) off += k < warp ? scnt[k] : 0, C += scnt[k];
        if (threadIdx.x == 0) s_done = 0;
        if (C <= kSampleRoom) {
            for (int64_t b = w0; b < w1; b += 32) {
                const int64_t i = b + lane;
                const float f = i < w1 ? v[i] : 0.0f;
                const uint32_t key = sample_key(f);
                const bool in = i < w1 && key >= lo_c;
                const uint32_t ball = __ballot_sync(0xffffffffu, in);
                if (in) {
                    const uint32_t pos = off + __popc(ball & ((1u << lane) - 1u));
                    cidx[pos] = static_cast<int32_t>(i);
                    ckey[pos] = key;
                    cw[pos] = static_cast<float>(exp(static_cast<double>((f - m) * inv_t)));
                }
                off += __popc(ball);
            }
            __syncthreads();
            if (warp == 0) {
                auto cmass = [&](uint32_t t) {
                    double s = 0.0;
                    for (uint32_t j = lane; j < C; j += 32) s += ckey[j] >= t ? static_cast<double>(cw[j]) : 0.0;
                    return sample_warp_sum(s);
                };
                const double held = cmass(lo_c);
                const double all = small_k ? held : total;
                uint32_t t = lo_c;
                bool ok = true;
                if (topp) {
                    const double want = static_cast<double>(p.top_p) * all;
                    if (held >= want) {
                        uint32_t a = lo_c, b2 = kmax;
                        while (a < b2) {
                            const uint32_t mid = a + (b2 - a + 1) / 2;
                            if (cmass(mid) >= want) a = mid;
                            else b2 = mid - 1;
                        }
                        t = a;
                    } else {
                        ok = false;  // the nucleus reaches below the compact set
                    }
                }
                if (ok) {
                    const double target = SamplePcg32(p.seed, p.step).next_f64() * cmass(t);
                    double cum = 0.0;
                    int32_t pick = -1, lastc = -1;
                    for (uint32_t j0 = 0; j0 < C && pick < 0; j0 += 32) {
                        const uint32_t j = j0 + lane;
                        const bool cand = j < C && ckey[j] >= t;
                        const double wv = cand ? static_cast<double>(cw[j]) : 0.0;
                        double incl = wv;
                        for (int o = 1; o < 32; o <<= 1) {
                            const double up = __shfl_up_sync(0xffffffffu, incl, o);
                            if (lane >= o) incl += up;
                        }
                        const uint32_t cross = __ballot_sync(0xffffffffu, wv > 0.0 && cum + incl > target);
                        const uint32_t cands = __ballot_sync(0xffffffffu, cand);
                        if (cross) pick = cidx[j0 + __ffs(cross) - 1];
                        if (cands) lastc = cidx[j0 + 31 - __clz(cands)];
                        cum += __shfl_sync(0xffffffffu, incl, 31);
                    }
                    if (lane == 0) idx[row] = pick >= 0 ? pick : (lastc >= 0 ? lastc : static_cast<int32_t>(best));
                }
                if (lane == 0) s_done = ok ? 1 : 0;
            }
        }
        __syncthreads();
        if (s_done) return;
    }

    // the whole row: top-p's threshold 4 bits a pass, then the draw walks the row
    if (!weighed) total = weigh();
    uint32_t t = lo;
    if (topp) {
        const double want = static_cast<double>(p.top_p) * total;
        uint32_t prefix = 0, himask = 0;
        for (int shift = 28; shift >= 0; shift -= 4) {
            double bk[16], ab = 0.0;
#pragma unroll
            for (int q = 0; q < 16; ++q) bk[q] = 0.0;
            for (int64_t i = w0 + lane; i < w1; i += 32) {
                const float wi = w[i];
                if (wi == 0.0f) continue;
                const uint32_t k = sample_key(v[i]), kh = k & himask;
                if (kh > prefix) {
                    ab += wi;
                } else if (kh == prefix) {
                    const uint32_t d = (k >> shift) & 15u;
#pragma unroll
                    for (int q = 0; q < 16; ++q) bk[q] += d == uint32_t(q) ? static_cast<double>(wi) : 0.0;
                }
            }
#pragma unroll
            for (int q = 0; q < 17; ++q) {
                const double s = sample_warp_sum(q < 16 ? bk[q] : ab);
                if (lane == 0) sb[warp][q] = s;
            }
            __syncthreads();
            double col[17];
#pragma unroll
            for (int q = 0; q < 17; ++q) {
                double s = 0.0;
                for (int k2 = 0; k2 < kSampleWarps; ++k2) s += sb[k2][q];
                col[q] = s;
            }
            __syncthreads();
            // the largest digit whose keys, with those above, hold want
            double cum = col[16];
            int dsel = -1;
#pragma unroll
            for (int q = 15; q >= 0; --q) {
                if (dsel < 0) {
                    cum += col[q];
                    if (cum >= want) dsel = q;
                }
            }
            prefix |= uint32_t(dsel < 0 ? 0 : dsel) << shift;
            himask |= 15u << shift;
        }
        t = prefix > kmax ? kmax : (prefix > lo ? prefix : lo);  // as the host's search, within [lo, kmax]
    }
    // each warp's mass at or above t, those of the warps before it, then the warp where u * mass
    // falls walks its range 32 tokens at a time
    double mine = 0.0;
    int64_t lastw = -1;
    for (int64_t i = w0 + lane; i < w1; i += 32)
        if (sample_key(v[i]) >= t) mine += w[i], lastw = i;
    mine = sample_warp_sum(mine);
    for (int o = 16; o > 0; o >>= 1) {
        const int64_t ol = __shfl_xor_sync(0xffffffffu, lastw, o);
        if (ol > lastw) lastw = ol;
    }
    if (lane == 0) sw[warp] = mine, smi[warp] = lastw;
    if (threadIdx.x == 0) s_done = 0;
    __syncthreads();
    double before = 0.0, sum = 0.0;
    int64_t lastall = -1;
    for (int k = 0; k < kSampleWarps; ++k) {
        before += k < warp ? sw[k] : 0.0;
        sum += sw[k];
        if (smi[k] > lastall) lastall = smi[k];
    }
    const double target = SamplePcg32(p.seed, p.step).next_f64() * sum;
    if (mine > 0.0 && target >= before && target < before + mine) {
        double cum = before;
        int64_t pick = -1;
        for (int64_t b = w0; b < w1 && pick < 0; b += 32) {
            const int64_t i = b + lane;
            const double wv = i < w1 && sample_key(v[i]) >= t ? static_cast<double>(w[i]) : 0.0;
            double incl = wv;
            for (int o = 1; o < 32; o <<= 1) {
                const double up = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += up;
            }
            const uint32_t cross = __ballot_sync(0xffffffffu, wv > 0.0 && cum + incl > target);
            if (cross) pick = b + __ffs(cross) - 1;
            cum += __shfl_sync(0xffffffffu, incl, 31);
        }
        if (lane == 0) {
            idx[row] = static_cast<int32_t>(pick >= 0 ? pick : lastw);
            s_done = 1;
        }
    }
    __syncthreads();
    // u * mass at or past the end (rounding): the last candidate
    if (!s_done && threadIdx.x == 0) idx[row] = static_cast<int32_t>(lastall < 0 ? best : lastall);
}


// ------------------------------------------------------------------ adjustments
// Penalties, logit bias, min_tokens: one block per row, a thread per adjustment (distinct tokens);
// rounded op by op (no fused multiply-add) as adjust_row_host is.
__global__ void adjust_rows_kernel(float *x, int64_t n, const SampleRow *params, const LogitAdj *adj) {
    const SampleRow p = params[blockIdx.x];
    if (p.adj_n <= 0) return;
    float *v = x + static_cast<int64_t>(blockIdx.x) * n;
    for (int j = threadIdx.x; j < p.adj_n; j += blockDim.x) {
        const LogitAdj a = adj[p.adj_off + j];
        float f = __fadd_rn(v[a.token], a.bias);
        f = f > 0.0f ? __fdiv_rn(f, a.rep) : __fmul_rn(f, a.rep);
        v[a.token] = __fsub_rn(f, a.sub);
    }
}

// ------------------------------------------------------------------ log-probabilities
// Rows asking (p.logprobs = k > 0) split in kArgmaxSlices slices, a block each (logprob_slices):
// the slice's largest value, its sum of exp(x - that) in double, and its k - 1 largest (larger
// value, else smaller index) by k - 1 passes over the slice (in L1). Then one block per row
// (logprob_merge): the row's log-sum-exp from the slices in slice order, the drawn token's
// log-probability, and the k - 1 largest of the slices' candidates by the same passes.
struct LogprobPart {
    float m;
    double s;
    float v[kMaxTopLogprobs];
    int32_t i[kMaxTopLogprobs];
};

// the block's best (value, index) of each thread's, on every thread (blockDim a multiple of 32)
__device__ __forceinline__ void lp_block_best(float &bv, int64_t &bi, float *smv, int64_t *smi) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int64_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
        argmax_merge(bv, bi, ov, oi);
    }
    if (lane == 0) smv[warp] = bv, smi[warp] = bi;
    __syncthreads();
    bv = lane < nw ? smv[lane] : -INFINITY;
    bi = lane < nw ? smi[lane] : -1;
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int64_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
        argmax_merge(bv, bi, ov, oi);
    }
    __syncthreads();
}

// the block's sum of each thread's v, in a fixed order, on every thread
__device__ __forceinline__ double lp_block_sum(double v, double *sw) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) sw[warp] = v;
    __syncthreads();
    double t = lane < nw ? sw[lane] : 0.0;
    for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
    __syncthreads();
    return t;
}

constexpr int kLogprobThreads = 256;

__global__ void __launch_bounds__(kLogprobThreads) logprob_slices_kernel(LogprobPart *parts, const float *x, int64_t n,
                                                                         const SampleRow *params) {
    __shared__ float smv[32];
    __shared__ int64_t smi[32];
    __shared__ double sw[32];
    const int64_t row = blockIdx.x;
    const SampleRow p = params[row];
    if (p.logprobs <= 0) return;
    const float *v = x + row * n;
    const int64_t len = (n + kArgmaxSlices - 1) / kArgmaxSlices;
    const int64_t lo = blockIdx.y * len, hi = imin(n, lo + len);
    LogprobPart &out = parts[row * kArgmaxSlices + blockIdx.y];
    float bv = -INFINITY;
    int64_t bi = -1;
    for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x)
        if (bi < 0 || v[i] > bv) bv = v[i], bi = i;
    lp_block_best(bv, bi, smv, smi);
    const float m = bv;
    double s = 0.0;
    if (bi >= 0)
        for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) s += exp(static_cast<double>(v[i] - m));
    s = lp_block_sum(s, sw);
    if (threadIdx.x == 0) out.m = m, out.s = bi >= 0 ? s : 0.0;
    const int k = min(p.logprobs - 1, kMaxTopLogprobs);
    float pv = INFINITY;
    int64_t pi = -1;
    for (int j = 0; j < k; ++j) {
        bv = -INFINITY, bi = -1;
        for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
            const float f = v[i];
            if ((f < pv || (f == pv && i > pi)) && (bi < 0 || f > bv)) bv = f, bi = i;
        }
        lp_block_best(bv, bi, smv, smi);
        if (threadIdx.x == 0) out.v[j] = bv, out.i[j] = static_cast<int32_t>(bi);
        pv = bv, pi = bi;
    }
}

__global__ void __launch_bounds__(kLogprobThreads) logprob_merge_kernel(float *lp, int32_t *ids, const LogprobPart *parts,
                                                                        const float *x, int64_t n, const SampleRow *params,
                                                                        const int32_t *idx) {
    __shared__ float smv[32];
    __shared__ int64_t smi[32];
    __shared__ double s_lse;
    const int64_t row = blockIdx.x;
    const SampleRow p = params[row];
    if (p.logprobs <= 0) return;
    const LogprobPart *pr = parts + row * kArgmaxSlices;
    const int64_t len = (n + kArgmaxSlices - 1) / kArgmaxSlices;
    const int slices = static_cast<int>((n + len - 1) / len);
    if (threadIdx.x == 0) {
        float M = -INFINITY;
        for (int q = 0; q < slices; ++q) M = fmaxf(M, pr[q].m);
        double t = 0.0;
        for (int q = 0; q < slices; ++q) t += pr[q].s * exp(static_cast<double>(pr[q].m - M));
        s_lse = static_cast<double>(M) + log(t);
    }
    __syncthreads();
    const double lse = s_lse;
    float *out = lp + row * (kMaxTopLogprobs + 1);
    int32_t *oid = ids + row * kMaxTopLogprobs;
    if (threadIdx.x == 0) out[0] = static_cast<float>(static_cast<double>(x[row * n + idx[row]]) - lse);
    const int k = min(p.logprobs - 1, kMaxTopLogprobs);
    const int64_t top = min(static_cast<int64_t>(k), n);
    float pv = INFINITY;
    int64_t pi = -1;
    for (int64_t j = 0; j < top; ++j) {
        float bv = -INFINITY;
        int64_t bi = -1;
        for (int c = threadIdx.x; c < slices * k; c += blockDim.x) {
            const LogprobPart &q = pr[c / k];
            const int64_t i = q.i[c % k];  // -1: the slice had no more
            if (i < 0) continue;
            const float f = q.v[c % k];
            if ((f < pv || (f == pv && i > pi)) && (bi < 0 || f > bv || (f == bv && i < bi))) bv = f, bi = i;
        }
        lp_block_best(bv, bi, smv, smi);
        if (threadIdx.x == 0) {
            oid[j] = static_cast<int32_t>(bi);
            out[1 + j] = static_cast<float>(static_cast<double>(bv) - lse);
        }
        pv = bv, pi = bi;
    }
}

}  // namespace
}  // namespace fullm
