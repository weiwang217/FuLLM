// SPDX-License-Identifier: Apache-2.0
// kernels/router.cuh - MoE routing: the routing lists, the route kernel, the fused router (GEMV +
// softmax top-k + lists), the softmax and sigmoid routers.
//
// Part of cuda_backend.cu's translation unit, included there after the CUDA and standard
// headers, in the order common, formats, gemv, convert, router, elementwise, copy, kimi, mla,
// attention, sampling, moe (each file uses only those before it). Kernels and their device
// helpers: no state.
#pragma once

namespace fullm {
namespace {

// Routing lists (moe_route_kernel): meta[0] = distinct resident experts U, meta[1] = passes P
// of kGeRows pairs; exps = meta + 2 ([slots]); start = exps + slots ([slots + 1]): expert u's
// pairs are plist[start[u] .. start[u+1]); pass q is expert pass_u[q]'s pairs from
// plist[pass_b[q]] (pass_u, pass_b: [maxp] each, then plist [pairs]); inv [pairs] is the
// inverse of plist (-1 for a pair of a non-resident expert). Pair order within an expert
// affects no result.
constexpr int kGeRows = 128;  // pairs per pass: the M tile of the prefill expert GEMM
struct MoeLists {
    int32_t *meta;
    int slots, maxp, pairs;
    __host__ __device__ int32_t *exps() const { return meta + 2; }
    __host__ __device__ int32_t *start() const { return meta + 2 + slots; }
    __host__ __device__ int32_t *pass_u() const { return meta + 3 + 2 * slots; }
    __host__ __device__ int32_t *pass_b() const { return meta + 3 + 2 * slots + maxp; }
    __host__ __device__ int32_t *plist() const { return meta + 3 + 2 * slots + 2 * maxp; }
    __host__ __device__ int32_t *inv() const { return meta + 3 + 2 * slots + 2 * maxp + pairs; }
    static int64_t maxp_for(int64_t slots, int64_t pairs) { return slots + (pairs + kGeRows - 1) / kGeRows; }
    static int64_t ints(int64_t slots, int64_t pairs) { return 3 + 2 * slots + 2 * maxp_for(slots, pairs) + 2 * pairs; }
};

// Exclusive scan of three per-thread counters over the block, and their totals: warp shuffles,
// then one warp over the warp sums; two barriers (the Hillis-Steele form took 2 log2(blockDim)).
// wt: 33 int3 of shared memory; blockDim a multiple of 32.
__device__ __forceinline__ int3 block_exscan3(int3 v, int3 *wt, int3 &total) {
    const unsigned full = 0xffffffffu;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = static_cast<int>(blockDim.x >> 5);
    int3 inc = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int a = __shfl_up_sync(full, inc.x, o), b = __shfl_up_sync(full, inc.y, o), c = __shfl_up_sync(full, inc.z, o);
        if (lane >= o) inc.x += a, inc.y += b, inc.z += c;
    }
    if (lane == 31) wt[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        const int3 t = lane < nw ? wt[lane] : make_int3(0, 0, 0);
        int3 ti = t;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int a = __shfl_up_sync(full, ti.x, o), b = __shfl_up_sync(full, ti.y, o), c = __shfl_up_sync(full, ti.z, o);
            if (lane >= o) ti.x += a, ti.y += b, ti.z += c;
        }
        if (lane < nw) wt[lane] = make_int3(ti.x - t.x, ti.y - t.y, ti.z - t.z);  // each warp's offset
        if (lane == 31) wt[32] = ti;  // the lanes past nw added zeros: the totals
    }
    __syncthreads();
    const int3 off = wt[warp];
    total = wt[32];
    return make_int3(off.x + inc.x - v.x, off.y + inc.y - v.y, off.z + inc.z - v.z);
}

// The routing lists of npairs pairs (see MoeLists), built by the calling block, all its threads:
// cnt and pos are [E] ints of shared memory, wt 33 int3. Per expert, its pairs' slots in plist
// (their order within an expert follows the atomics and affects no result).
__device__ void moe_build_lists(MoeLists L, const int32_t *idx, int64_t npairs, const Backend::ExpertW *table, int E,
                                int32_t *cnt, int32_t *pos, int3 *wt) {
    int32_t *exps = L.exps(), *start = L.start(), *pu = L.pass_u(), *pb = L.pass_b(), *plist = L.plist(), *inv = L.inv();
    for (int e = threadIdx.x; e < E; e += blockDim.x) cnt[e] = 0;
    for (int64_t p = threadIdx.x; p < npairs; p += blockDim.x) inv[p] = -1;
    __syncthreads();
    for (int64_t p = threadIdx.x; p < npairs; p += blockDim.x) {
        const int e = idx[p];
        if (table[e].w1.data) atomicAdd(&cnt[e], 1);
    }
    __syncthreads();
    // each thread's run of experts: its pairs, used experts and passes, scanned over the block,
    // then the run is filled in
    const int per = (E + static_cast<int>(blockDim.x) - 1) / static_cast<int>(blockDim.x), e0 = threadIdx.x * per,
              e1 = min(E, e0 + per);
    int32_t c = 0, n = 0, nq = 0;
    for (int e = e0; e < e1; ++e) c += cnt[e], n += cnt[e] > 0, nq += (cnt[e] + kGeRows - 1) / kGeRows;
    int3 tot;
    const int3 ex = block_exscan3(make_int3(c, n, nq), wt, tot);
    int32_t o = ex.x, u = ex.y, q = ex.z;
    for (int e = e0; e < e1; ++e) {
        pos[e] = o;
        if (cnt[e]) {
            exps[u] = e, start[u] = o;
            for (int32_t j = 0; j < cnt[e]; j += kGeRows) pu[q] = u, pb[q] = o + j, ++q;
            ++u;
        }
        o += cnt[e];
    }
    if (threadIdx.x == 0) {
        start[tot.y] = tot.x;
        L.meta[0] = tot.y;
        L.meta[1] = tot.z;
    }
    __syncthreads();
    for (int64_t p = threadIdx.x; p < npairs; p += blockDim.x) {
        const int e = idx[p];
        if (table[e].w1.data) {
            const int32_t i = atomicAdd(&pos[e], 1);
            plist[i] = static_cast<int32_t>(p);
            inv[p] = i;
        }
    }
}

constexpr int kRouteThreads = 1024;  // one block; a prefill step routes tens of thousands of pairs
constexpr int kRouterRegs = 16;      // the fused router's top-k keeps up to 16 * 16 experts in registers
__global__ void __launch_bounds__(kRouteThreads)
    moe_route_kernel(MoeLists L, const int32_t *idx, int64_t npairs, const Backend::ExpertW *table, int E) {
    __shared__ int32_t cnt[1024], pos[1024];
    __shared__ int3 wt[33];
    pdl_wait();
    pdl_trigger();
    moe_build_lists(L, idx, npairs, table, E, cnt, pos, wt);
}

// Router logits + softmax top-k for a decode step (N <= kGemvMaxRows): the logits are the
// tensor-core GEMV of the router (bitwise the gemm path: same tiles, same split), and the
// block that finishes last runs each token's selection on one warp with router_softmax_kernel's
// rules (top-k of the probabilities, ties to the lower index). The last block resets the
// counter for the next call. L.meta set: that block then builds the routing lists of the expert
// call that follows (moe_route_kernel's job, one launch less per layer). XB: the GEMV reads x's
// bf16 twin xb (x rounded to bf16: the bits BX's own rounding gives), half the bytes of fp32 rows.
template <bool TWO, bool BX, bool XB = false>
__global__ void __launch_bounds__(kTcWarps * 32, 2)
    router_fused_kernel(float *logits, unsigned *count, int32_t *idx, float *wout, const void *__restrict__ W,
                        const float *x, int N, int E, int64_t width, int k, int renorm, int S, MoeLists L,
                        const Backend::ExpertW *table, __half *xh, int regs, const uint16_t *xb) {
    extern __shared__ uint4 tc_smem[];
    __shared__ bool last;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const auto logit_out = [&](int m, int64_t n, float v0, float v1) {
        if (n < E) logits[m * E + n] = v0;
        if (n + 1 < E) logits[m * E + n + 1] = v1;
    };
    if constexpr (XB)
        tc_tile<1, TWO, BX>(W, N, width, E, S, blockIdx.x, tc_smem,
                            [&](int m) { return xb + static_cast<int64_t>(m) * width; }, logit_out);
    else
        tc_tile<1, TWO, BX>(W, N, width, E, S, blockIdx.x, tc_smem,
                            [&](int m) { return x + static_cast<int64_t>(m) * width; }, logit_out);
    // xh: x as the e4m3 expert tiles take it (fp16), each block a share of the rows, once here
    // instead of in every tile of every expert (x is valid: the tile above waited for it)
    if (xh) {
        const int64_t n4 = static_cast<int64_t>(N) * width / 4;
        for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4;
             i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
            const float4 v = reinterpret_cast<const float4 *>(x)[i];
            reinterpret_cast<uint2 *>(xh)[i] = make_uint2(f16x2(v.x, v.y), f16x2(v.z, v.w));
        }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(count, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    if (threadIdx.x == 0) *count = 0;
    __threadfence();
    // one token per half-warp (16 lanes, xor shuffles within the half), all of a step's tokens at
    // once; lane j of the half keeps pick j (k <= 16)
    const int half = lane >> 4, hl = lane & 15;
    if (regs && E <= 16 * kRouterRegs) {
        // up to 256 experts: lane hl holds logits hl, hl + 16, ... in registers, all loaded at once
        // (the shared-memory form below waited on each load and re-read shared memory per pick).
        // The same operations in the same order: the same picks and weights.
        for (int base = 2 * warp; base < N; base += 2 * kTcWarps) {  // warp-uniform
            const int m = base + half;
            const bool on = m < N;
            float v[kRouterRegs];
#pragma unroll
            for (int r = 0; r < kRouterRegs; ++r) {
                const int i = hl + 16 * r;
                v[r] = on && i < E ? __ldcg(logits + m * E + i) : -INFINITY;
            }
            float mx = -INFINITY;
#pragma unroll
            for (int r = 0; r < kRouterRegs; ++r)
                if (on && hl + 16 * r < E) mx = fmaxf(mx, v[r]);
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            float z = 0.0f;
#pragma unroll
            for (int r = 0; r < kRouterRegs; ++r)
                if (on && hl + 16 * r < E) z += (v[r] = expf(v[r] - mx));
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
#pragma unroll
            for (int r = 0; r < kRouterRegs; ++r)
                if (on && hl + 16 * r < E) v[r] = v[r] / z;
            float sum = 0.0f, mw = 0.0f;
            int mi = 0;
            for (int j = 0; j < k; ++j) {
                float bv = -INFINITY;
                int bi = INT_MAX;
#pragma unroll
                for (int r = 0; r < kRouterRegs; ++r)
                    if (on && hl + 16 * r < E && v[r] > bv) bv = v[r], bi = hl + 16 * r;  // ascending i
#pragma unroll
                for (int o = 8; o > 0; o >>= 1) {
                    const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                    if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
                }
                if (bi == INT_MAX) bi = 0;
                sum += bv;
                if (hl == j) mw = bv, mi = bi;
#pragma unroll
                for (int r = 0; r < kRouterRegs; ++r)
                    if (on && hl + 16 * r == bi) v[r] = -INFINITY;
            }
            if (on && hl < k) {
                idx[m * k + hl] = mi;
                wout[m * k + hl] = renorm ? mw / sum : mw;
            }
        }
    } else {
    float *p = reinterpret_cast<float *>(tc_smem) + (2 * warp + half) * 1024;  // E <= 1024 per token
    for (int base = 2 * warp; base < N; base += 2 * kTcWarps) {  // warp-uniform
        const int m = base + half;
        const bool on = m < N;
        float mx = -INFINITY;
        if (on)
            for (int i = hl; i < E; i += 16) mx = fmaxf(mx, p[i] = __ldcg(logits + m * E + i));
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        float z = 0.0f;
        if (on)
            for (int i = hl; i < E; i += 16) z += (p[i] = expf(p[i] - mx));
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
        if (on)
            for (int i = hl; i < E; i += 16) p[i] = p[i] / z;
        __syncwarp();
        float sum = 0.0f, mw = 0.0f;
        int mi = 0;
        for (int j = 0; j < k; ++j) {
            float bv = -INFINITY;
            int bi = INT_MAX;
            if (on)
                for (int i = hl; i < E; i += 16)
                    if (p[i] > bv) bv = p[i], bi = i;  // ascending i: ties keep the lower index
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
            }
            if (bi == INT_MAX) bi = 0;
            sum += bv;
            if (hl == j) mw = bv, mi = bi;
            if (on && hl == 0) p[bi] = -INFINITY;
            __syncwarp();
        }
        if (on && hl < k) {
            idx[m * k + hl] = mi;
            wout[m * k + hl] = renorm ? mw / sum : mw;
        }
        __syncwarp();
    }
    }
    if (L.meta) {  // the picks above are this block's writes: visible to it after the barrier
        __syncthreads();
        auto *cnt = reinterpret_cast<int32_t *>(tc_smem);  // the selection's buffers are done with
        moe_build_lists(L, idx, static_cast<int64_t>(N) * k, table, E, cnt, cnt + E, reinterpret_cast<int3 *>(cnt + 2 * E));
    }
}


// One block per token: softmax over the E logits, then k rounds of a block argmax over the
// probabilities (value desc, index asc), weights = probabilities, renormalised if asked.
__global__ void router_softmax_kernel(int32_t *idx, float *wout, const float *logits, int E, int k, int renorm) {
    extern __shared__ float sp[];  // prob[E], then choice[E]
    __shared__ float red[32], rv[32];
    __shared__ int ri[32];
    float *prob = sp, *choice = sp + E;
    const int64_t t = blockIdx.x;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    float mx = -INFINITY;
    for (int e = threadIdx.x; e < E; e += blockDim.x) mx = fmaxf(mx, logits[t * E + e]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if (lane == 0) rv[warp] = mx;
    __syncthreads();
    mx = rv[0];
    for (int w = 1; w < nw; ++w) mx = fmaxf(mx, rv[w]);
    float z = 0.0f;
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        const float x = expf(logits[t * E + e] - mx);
        prob[e] = x;
        z += x;
    }
    z = block_sum(z, red);
    for (int e = threadIdx.x; e < E; e += blockDim.x) choice[e] = prob[e] = prob[e] / z;
    __syncthreads();
    for (int j = 0; j < k; ++j) {
        float bv = -INFINITY;
        int bi = INT_MAX;
        for (int e = threadIdx.x; e < E; e += blockDim.x)
            if (choice[e] > bv || (choice[e] == bv && e < bi)) bv = choice[e], bi = e;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
        }
        if (lane == 0) rv[warp] = bv, ri[warp] = bi;
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = rv[0];
            int i = ri[0];
            for (int w = 1; w < nw; ++w)
                if (rv[w] > b || (rv[w] == b && ri[w] < i)) b = rv[w], i = ri[w];
            if (i == INT_MAX) i = 0;
            idx[t * k + j] = i;
            wout[t * k + j] = prob[i];
            choice[i] = -INFINITY;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && renorm) {
        float s = 0.0f;
        for (int j = 0; j < k; ++j) s += wout[t * k + j];
        for (int j = 0; j < k; ++j) wout[t * k + j] /= s;
    }
}

// A warp per token (8 per block), the row's E <= 32 * EPL logits in registers: softmax by
// warp shuffles, then k rounds of a warp argmax (value desc, index asc), weights =
// probabilities, renormalised if asked. The same per-element math and the same selection as
// router_softmax_kernel (expf(x - max) / z; only z's summation order differs), without its
// block barriers: 59 -> ~5 us for 5900 tokens x 128 experts.
template <int EPL>
__global__ void router_softmax_warp_kernel(int32_t *idx, float *wout, const float *__restrict__ logits, int64_t N, int E,
                                           int k, int renorm) {
    const int64_t t = static_cast<int64_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (t >= N) return;  // warp-uniform
    const int lane = threadIdx.x & 31;
    const float *row = logits + t * E;
    float x[EPL];
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < EPL; ++j) {
        const int e = lane + 32 * j;
        x[j] = e < E ? row[e] : -INFINITY;
        mx = fmaxf(mx, x[j]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float z = 0.0f;
#pragma unroll
    for (int j = 0; j < EPL; ++j) {
        const int e = lane + 32 * j;
        x[j] = e < E ? expf(x[j] - mx) : 0.0f;
        z += x[j];
    }
    z = warp_sum(z);
#pragma unroll
    for (int j = 0; j < EPL; ++j) x[j] = lane + 32 * j < E ? x[j] / z : -INFINITY;  // -inf: never chosen
    float wsum = 0.0f;
    for (int r = 0; r < k; ++r) {
        float bv = -INFINITY;
        int bi = INT_MAX;
#pragma unroll
        for (int j = 0; j < EPL; ++j) {
            const int e = lane + 32 * j;
            if (x[j] > bv || (x[j] == bv && e < bi)) bv = x[j], bi = e;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
        }
        if (bi == INT_MAX) bi = 0;  // every probability -inf (NaN logits): expert 0, as the block kernel
        const int owner = bi & 31, jj = bi >> 5;
        float pv = 0.0f;
        if (lane == owner) {
#pragma unroll
            for (int j = 0; j < EPL; ++j)
                if (j == jj) pv = x[j], x[j] = -INFINITY;
        }
        pv = __shfl_sync(0xffffffffu, pv, owner);
        if (lane == 0) idx[t * k + r] = bi, wout[t * k + r] = pv;
        wsum += pv;  // in choice order, as the block kernel's renormalisation
    }
    if (renorm)
        for (int r = lane; r < k; r += 32) wout[t * k + r] /= wsum;
}


// ------------------------------------------------------------------ MoE routing

// One block per token. k rounds of a block argmax, ordered by (value desc, index asc), so
// the first index wins ties exactly like the CPU scan.
__global__ void router_kernel(int32_t *idx, float *wout, const float *logits, const float *bias, int E, int k,
                              int renorm, float rscale, int biased) {
    extern __shared__ float sm[];  // score[E], choice[E]
    __shared__ float rv[32];
    __shared__ int ri[32];
    float *score = sm;
    float *choice = sm + E;
    const int64_t t = blockIdx.x;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        const float s = sigm(logits[t * E + e]);
        score[e] = s;
        choice[e] = s + (bias ? bias[e] : 0.0f);  // the bias steers SELECTION only
    }
    __syncthreads();
    for (int j = 0; j < k; ++j) {
        float bv = -INFINITY;
        int bi = INT_MAX;
        for (int e = threadIdx.x; e < E; e += blockDim.x) {
            const float c = choice[e];
            if (c > bv || (c == bv && e < bi)) {
                bv = c;
                bi = e;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            rv[warp] = bv;
            ri[warp] = bi;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = rv[0];
            int i = ri[0];
            for (int w = 1; w < nw; ++w)
                if (rv[w] > b || (rv[w] == b && ri[w] < i)) {
                    b = rv[w];
                    i = ri[w];
                }
            if (i == INT_MAX) i = 0;
            idx[t * k + j] = i;
            // the UNBIASED score, unless reproducing a checkpoint that weights by score + bias
            wout[t * k + j] = biased ? score[i] + (bias ? bias[i] : 0.0f) : score[i];
            choice[i] = -INFINITY;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (renorm) {
            float s = 0.0f;
            for (int j = 0; j < k; ++j) s += wout[t * k + j];
            s += 1e-20f;
            for (int j = 0; j < k; ++j) wout[t * k + j] /= s;
        }
        for (int j = 0; j < k; ++j) wout[t * k + j] *= rscale;
    }
}

}  // namespace
}  // namespace fullm
