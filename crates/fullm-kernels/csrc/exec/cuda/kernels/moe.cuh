// SPDX-License-Identifier: Apache-2.0
// kernels/moe.cuh - the routed experts: decode GEMVs and tensor-core tiles (two-kernel and fused),
// the prefill expert GEMM, the CUTLASS paths' gathers, GLUs and quantisers, scatter and reduce.
//
// Part of cuda_backend.cu's translation unit, included there after the CUDA and standard
// headers, in the order common, formats, gemv, convert, router, elementwise, copy, kimi, mla,
// attention, sampling, moe (each file uses only those before it). Kernels and their device
// helpers: no state.
#pragma once

namespace fullm {
namespace {

// ------------------------------------------------------------------ routed experts in HBM
// M = 1 GEMV rows with the GEMV path's chunk walk and FMA order, so a resident expert
// computes each row exactly like a trunk matrix of its format would. A warp computes a SET
// of NR rows (possibly of different matrices) so each x chunk it loads feeds NR rows.
template <int WF>
__device__ __forceinline__ const uint4 *row_ptr(const Weight &w, int64_t row) {
    return reinterpret_cast<const uint4 *>(static_cast<const char *>(w.data) + row * w.cols * WT<WF>::kElem);
}

template <int WF, int NR>
__device__ __forceinline__ void warp_dot_rowset(const uint4 *const (&wr)[NR], int64_t K, const float *__restrict__ x,
                                                float (&acc)[NR]) {
    constexpr int P = 16 / WT<WF>::kElem;
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int r = 0; r < NR; ++r) acc[r] = 0.0f;
    for (int64_t c = lane; c < K / P; c += 32) {
        float v[NR][P];
#pragma unroll
        for (int r = 0; r < NR; ++r) decode16<WF>(__ldg(wr[r] + c), v[r]);
        const float4 *xv = reinterpret_cast<const float4 *>(x + c * P);
#pragma unroll
        for (int q = 0; q < P / 4; ++q) {
            const float4 xx = __ldg(xv + q);
#pragma unroll
            for (int r = 0; r < NR; ++r) {
                acc[r] = fmaf(v[r][4 * q + 0], xx.x, acc[r]);
                acc[r] = fmaf(v[r][4 * q + 1], xx.y, acc[r]);
                acc[r] = fmaf(v[r][4 * q + 2], xx.z, acc[r]);
                acc[r] = fmaf(v[r][4 * q + 3], xx.w, acc[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < NR; ++r) acc[r] = warp_sum(acc[r]);
}

template <int WF>
__device__ __forceinline__ float row_scaled(const Weight &w, int64_t row, float acc) {
    return (WF == 2 || WF == 3) ? acc * __ldg(w.scale + row) : acc;
}

// grid (row blocks of 16, pairs): each warp computes rows r0, r0+1 of gate AND up, so every
// x chunk feeds four weight rows. scratch[p][row] = glu(gate . x_t, up . x_t)
template <int WF>
__global__ void __launch_bounds__(256)
    moe_gateup_kernel(float *scratch, const float *x, const int32_t *idx, int k, const Backend::ExpertW *table,
                      int64_t width, int64_t inter, int silu, float b1, float b2) {
    const int64_t p = blockIdx.y;
    const Backend::ExpertW &e = table[idx[p]];
    if (!e.w1.data) return;  // not resident: the CPU computes this pair
    const int64_t r0 = (static_cast<int64_t>(blockIdx.x) * 8 + (threadIdx.x >> 5)) * 2;
    if (r0 >= inter) return;
    const int nr = r0 + 1 < inter ? 2 : 1;
    const int64_t r1 = r0 + nr - 1;
    const uint4 *const wr[4] = {row_ptr<WF>(e.w1, r0), row_ptr<WF>(e.w1, r1), row_ptr<WF>(e.w3, r0),
                                row_ptr<WF>(e.w3, r1)};
    float acc[4];
    warp_dot_rowset<WF, 4>(wr, width, x + (p / k) * width, acc);
    if ((threadIdx.x & 31) == 0)
        for (int j = 0; j < nr; ++j) {
            const float g = row_scaled<WF>(e.w1, r0 + j, acc[j]);
            const float u = row_scaled<WF>(e.w3, r0 + j, acc[2 + j]);
            scratch[p * inter + r0 + j] = silu ? g * sigm(g) * u : b1 * tanhf(g / b1) * sigm(g) * (b2 * tanhf(u / b2));
        }
}

// grid (row blocks of 32, pairs): contrib[p][row] = w[p] * (down . scratch[p]), four rows per warp
template <int WF>
__global__ void __launch_bounds__(256)
    moe_down_kernel(float *contrib, const float *scratch, const int32_t *idx, const float *w,
                    const Backend::ExpertW *table, int64_t width, int64_t inter) {
    const int64_t p = blockIdx.y;
    const Backend::ExpertW &e = table[idx[p]];
    if (!e.w1.data) return;
    const int64_t r0 = (static_cast<int64_t>(blockIdx.x) * 8 + (threadIdx.x >> 5)) * 4;
    if (r0 >= width) return;
    const int nr = static_cast<int>(imin(4, width - r0));
    const uint4 *wr[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) wr[r] = row_ptr<WF>(e.w2, r0 + (r < nr ? r : 0));
    float acc[4];
    warp_dot_rowset<WF, 4>(wr, inter, scratch + p * inter, acc);
    if ((threadIdx.x & 31) == 0)
        for (int r = 0; r < nr; ++r) contrib[p * width + r0 + r] = w[p] * row_scaled<WF>(e.w2, r0 + r, acc[r]);
}

// ---- resident experts on the tensor cores, grouped by expert: each distinct expert of the call
// is streamed once per 16 of its pairs, those pairs' tokens forming the M rows of the tile. Rows of an MMA are independent, so a pair's result depends neither on grouping nor
// on the batch. Gate and up are stored row-interleaved ([2*inter][width]: gate r, up r, ...),
// so each lane's pair of sums is (gate r, up r) and the GLU happens in the epilogue.
//

// The expert GEMVs on a persistent grid: blocks walk (expert, row block) items in a fixed order;
// an item is one tc_tile (S = 1: each warp streams whole rows) per 16 of the expert's pairs.
// Each block first looks up all its items' weights and pair ranges together (one round of
// dependent loads per kernel, not per item).
struct MoeItem {
    const void *data;
    const float *scale;
    int32_t b0, b1;  // the expert's pairs: plist[b0 .. b1)
};
constexpr int kMaxBlockItems = 32;

// The block's items j = 0..: item blockIdx.x + j * gridDim.x; returns how many. W1: gate/up.
template <bool W1>
__device__ int moe_items(MoeItem *it, MoeLists L, const Backend::ExpertW *table, int bpe) {
    const int n = L.meta[0] * bpe;
    const int mine = n > static_cast<int>(blockIdx.x) ? (n - blockIdx.x + gridDim.x - 1) / gridDim.x : 0;
    for (int j = threadIdx.x; j < min(mine, kMaxBlockItems); j += blockDim.x) {
        const int u = (blockIdx.x + j * gridDim.x) / bpe;
        const Backend::ExpertW &e = table[L.exps()[u]];
        const Weight &w = W1 ? e.w1 : e.w2;
        it[j] = MoeItem{w.data, w.scale, L.start()[u], L.start()[u + 1]};
    }
    __syncthreads();
    return mine;
}
__device__ __forceinline__ MoeItem moe_item(const MoeItem *it, int j, MoeLists L, const Backend::ExpertW *table,
                                            int bpe, bool w1) {
    if (j < kMaxBlockItems) return it[j];
    const int u = (blockIdx.x + j * gridDim.x) / bpe;
    const Weight &w = w1 ? table[L.exps()[u]].w1 : table[L.exps()[u]].w2;
    return MoeItem{w.data, w.scale, L.start()[u], L.start()[u + 1]};
}
// The block's call after the one for pairs [b, b + 16) of item j (item w, row block rb), for the
// tiles' chain (TcChain): the item's next 16 pairs, else item j + 1's first ones. on = 0 (or an
// item without pairs next) ends the chain there: the next call fills its own ring.
__device__ __forceinline__ void moe_chain(TcChain &ch, int on, const MoeItem &w, int32_t b, int rb, int j, int mine,
                                          const MoeItem *items, MoeLists L, const Backend::ExpertW *table, int bpe,
                                          bool w1) {
    ch.next = nullptr;
    if (!(on & 1)) return;
    if (b + 16 < w.b1) {
        ch.next = w.data, ch.next_blk = rb;
    } else if (j + 1 < mine) {
        const MoeItem v = moe_item(items, j + 1, L, table, bpe, w1);
        if (v.b0 < v.b1) ch.next = v.data, ch.next_blk = (blockIdx.x + (j + 1) * gridDim.x) % bpe;
    }
}

// XH (e4m3 weights, bf16-activation mode): x arrives as fp16 rows and the GLU leaves as fp16
// rows (clamped and rounded as the tile would round them), so neither tile converts activations;
// otherwise both are fp32.
template <int WF, bool BX, int NT, bool XH = false>
__global__ void __launch_bounds__(kTcWarps * 32, 2)
    moe_tc_gateup_kernel(void *scratch, const void *x, MoeLists L, int k, const Backend::ExpertW *table,
                         int64_t width, int64_t inter, int silu, float b1, float b2, int chain) {
    extern __shared__ uint4 tc_smem[];
    __shared__ MoeItem items[kMaxBlockItems];
    pdl_wait();
    const int32_t *plist = L.plist();
    const int bpe = static_cast<int>((2 * inter + 8 * kTcWarps * NT - 1) / (8 * kTcWarps * NT));
    const int mine = moe_items<true>(items, L, table, bpe);
    TcChain ch;
    for (int j = 0; j < mine; ++j) {
        const int rb = (blockIdx.x + j * gridDim.x) % bpe;
        const MoeItem w = moe_item(items, j, L, table, bpe, true);
        for (int32_t b = w.b0; b < w.b1; b += 16) {
            const int mt = min(16, w.b1 - b);
            const int32_t *ps = plist + b;
            auto xrow = [&](int m) {
                if constexpr (XH) return static_cast<const __half *>(x) + static_cast<int64_t>(ps[m] / k) * width;
                else return static_cast<const float *>(x) + static_cast<int64_t>(ps[m] / k) * width;
            };
            auto out = [&](int m, int64_t n, float v0, float v1) {
                const float gv = (WF == 2 || WF == 3) ? v0 * __ldg(w.scale + n) : v0;
                const float uv = (WF == 2 || WF == 3) ? v1 * __ldg(w.scale + n + 1) : v1;
                const float g = silu ? gv * sigm(gv) * uv : b1 * tanhf(gv / b1) * sigm(gv) * (b2 * tanhf(uv / b2));
                const int64_t at = static_cast<int64_t>(ps[m]) * inter + n / 2;
                if constexpr (XH) static_cast<__half *>(scratch)[at] = f16c(g);
                else static_cast<float *>(scratch)[at] = g;
            };
            const float *bs = (WF == 4 || WF == 5) ? w.scale : nullptr;
            moe_chain(ch, chain, w, b, rb, j, mine, items, L, table, bpe, true);
            if (mt > 8) tc_tile<WF, true, BX, NT>(w.data, mt, width, 2 * inter, 1, rb, tc_smem, xrow, out, bs, &ch);
            else tc_tile<WF, false, BX, NT>(w.data, mt, width, 2 * inter, 1, rb, tc_smem, xrow, out, bs, &ch);
        }
    }
}

template <int WF, bool BX, int NT, bool XH = false>
__global__ void __launch_bounds__(kTcWarps * 32, 2)
    moe_tc_down_kernel(float *contrib, const void *scratch, MoeLists L, const float *wts,
                       const Backend::ExpertW *table, int64_t width, int64_t inter, int chain) {
    extern __shared__ uint4 tc_smem[];
    __shared__ MoeItem items[kMaxBlockItems];
    const int32_t *plist = L.plist();
    const int bpe = static_cast<int>((width + 8 * kTcWarps * NT - 1) / (8 * kTcWarps * NT));
    const int mine = moe_items<false>(items, L, table, bpe);
    TcChain ch;
    for (int j = 0; j < mine; ++j) {
        const int rb = (blockIdx.x + j * gridDim.x) % bpe;
        const MoeItem w = moe_item(items, j, L, table, bpe, false);
        for (int32_t b = w.b0; b < w.b1; b += 16) {
            const int mt = min(16, w.b1 - b);
            const int32_t *ps = plist + b;
            auto xrow = [&](int m) {
                if constexpr (XH) return static_cast<const __half *>(scratch) + static_cast<int64_t>(ps[m]) * inter;
                else return static_cast<const float *>(scratch) + static_cast<int64_t>(ps[m]) * inter;
            };
            auto out = [&](int m, int64_t n, float v0, float v1) {
                const int64_t p = ps[m];
                float *cr = contrib + p * width;
                if (n < width) cr[n] = wts[p] * ((WF == 2 || WF == 3) ? v0 * __ldg(w.scale + n) : v0);
                if (n + 1 < width) cr[n + 1] = wts[p] * ((WF == 2 || WF == 3) ? v1 * __ldg(w.scale + n + 1) : v1);
            };
            const float *bs = (WF == 4 || WF == 5) ? w.scale : nullptr;
            moe_chain(ch, chain, w, b, rb, j, mine, items, L, table, bpe, false);
            if (mt > 8) tc_tile<WF, true, BX, NT>(w.data, mt, inter, width, 1, rb, tc_smem, xrow, out, bs, &ch);
            else tc_tile<WF, false, BX, NT>(w.data, mt, inter, width, 1, rb, tc_smem, xrow, out, bs, &ch);
        }
    }
}

// The expert slots a call can have (moe_experts rejects more than 1024 experts).
constexpr int kMoeMaxSlots = 1024;

// Gate/up and down as one persistent kernel (the two above, back to back, cost ~14 us of launch,
// list lookups, ring fill and tail each per layer). Item gi < n1 is gate/up item gi (expert
// gi / bpe1, row block gi % bpe1), the rest are down items in the same order. A down item reads
// its expert's GLU rows, which that expert's bpe1 gate/up items write: each raises done[u] when
// finished (after tc_tile's closing barrier), and the down item waits for bpe1 once its first
// weight groups are in flight (TcChain::ready). Blocks take their items in increasing order and
// every gate/up item precedes every down item, so no item a down item waits for is itself
// waiting: with the whole grid resident (it is sized to the resident capacity) nothing
// deadlocks. The last down item of an expert zeroes its two counters for the next call. The
// tiles, their k order and their epilogues are the two kernels': the same bits.
struct MoeFItem {
    const void *data;
    const float *scale;
    int32_t b0, b1;  // the expert's pairs: plist[b0 .. b1)
    int u, rb;       // the expert's list index, the item's row block
};
__device__ __forceinline__ MoeFItem moe_fitem(int gi, MoeLists L, const Backend::ExpertW *table, int bpe1, int bpe2,
                                              int n1) {
    const bool down = gi >= n1;
    const int r = down ? gi - n1 : gi, bpe = down ? bpe2 : bpe1, u = r / bpe;
    const Backend::ExpertW &e = table[L.exps()[u]];
    const Weight &w = down ? e.w2 : e.w1;
    return MoeFItem{w.data, w.scale, L.start()[u], L.start()[u + 1], u, r % bpe};
}

// done: 2 * kMoeMaxSlots counters, zero between calls (per expert: gate/up items finished, down
// items finished)
template <int WF, bool BX, int NT, bool XH = false>
__global__ void __launch_bounds__(kTcWarps * 32, 2)
    moe_tc_fused_kernel(float *contrib, void *scratch, const void *x, MoeLists L, int k, const float *wts,
                        const Backend::ExpertW *table, int64_t width, int64_t inter, int silu, float b1, float b2,
                        int chain, unsigned *done) {
    extern __shared__ uint4 tc_smem[];
    __shared__ MoeFItem items[kMaxBlockItems];
    pdl_wait();
    const int32_t *plist = L.plist();
    const int bpe1 = static_cast<int>((2 * inter + 8 * kTcWarps * NT - 1) / (8 * kTcWarps * NT));
    const int bpe2 = static_cast<int>((width + 8 * kTcWarps * NT - 1) / (8 * kTcWarps * NT));
    const int nu = L.meta[0], n1 = nu * bpe1, n = n1 + nu * bpe2;
    const int bi = static_cast<int>(blockIdx.x), gd = static_cast<int>(gridDim.x);
    const int mine = n > bi ? (n - bi + gd - 1) / gd : 0;
    for (int j = threadIdx.x; j < min(mine, kMaxBlockItems); j += blockDim.x)
        items[j] = moe_fitem(bi + j * gd, L, table, bpe1, bpe2, n1);
    __syncthreads();
    auto item = [&](int j) { return j < kMaxBlockItems ? items[j] : moe_fitem(bi + j * gd, L, table, bpe1, bpe2, n1); };
    unsigned *done1 = done, *done2 = done + kMoeMaxSlots;
    // the block's gate/up items come first (gi < n1), then its down items: two loops, so the
    // phases' live values do not add up (one loop over both had the fp8b kernel spilling)
    const int mine1 = n1 > bi ? min(mine, (n1 - bi + gd - 1) / gd) : 0;
    TcChain ch;
    for (int j = 0; j < mine1; ++j) {
        const MoeFItem w = item(j);
        const float *bs = (WF == 4 || WF == 5) ? w.scale : nullptr;
        ch.ready = nullptr;
        for (int32_t b = w.b0; b < w.b1; b += 16) {
            const int mt = min(16, w.b1 - b);
            const int32_t *ps = plist + b;
            ch.next = nullptr;  // the next call: this item's next 16 pairs, else the next gate/up item
            if (chain & 1) {
                if (b + 16 < w.b1) {
                    ch.next = w.data, ch.next_blk = w.rb;
                } else if (j + 1 < mine1) {
                    const MoeFItem v = item(j + 1);
                    if (v.b0 < v.b1) ch.next = v.data, ch.next_blk = v.rb;
                }
            }
            auto xrow = [&](int m) {
                if constexpr (XH) return static_cast<const __half *>(x) + static_cast<int64_t>(ps[m] / k) * width;
                else return static_cast<const float *>(x) + static_cast<int64_t>(ps[m] / k) * width;
            };
            auto out = [&](int m, int64_t nn, float v0, float v1) {
                const float gv = (WF == 2 || WF == 3) ? v0 * __ldg(w.scale + nn) : v0;
                const float uv = (WF == 2 || WF == 3) ? v1 * __ldg(w.scale + nn + 1) : v1;
                const float g = silu ? gv * sigm(gv) * uv : b1 * tanhf(gv / b1) * sigm(gv) * (b2 * tanhf(uv / b2));
                const int64_t at = static_cast<int64_t>(ps[m]) * inter + nn / 2;
                if constexpr (XH) static_cast<__half *>(scratch)[at] = f16c(g);
                else static_cast<float *>(scratch)[at] = g;
            };
            if (mt > 8) tc_tile<WF, true, BX, NT>(w.data, mt, width, 2 * inter, 1, w.rb, tc_smem, xrow, out, bs, &ch);
            else tc_tile<WF, false, BX, NT>(w.data, mt, width, 2 * inter, 1, w.rb, tc_smem, xrow, out, bs, &ch);
        }
        // tc_tile ended with a block barrier: the block's GLU rows of this item are written
        if (threadIdx.x == 0) red_release_gpu(done1 + w.u, 1u);
    }
    for (int j = mine1; j < mine; ++j) {
        const MoeFItem w = item(j);
        const float *bs = (WF == 4 || WF == 5) ? w.scale : nullptr;
        ch.ready = done1 + w.u, ch.need = static_cast<unsigned>(bpe1);
        for (int32_t b = w.b0; b < w.b1; b += 16) {
            const int mt = min(16, w.b1 - b);
            const int32_t *ps = plist + b;
            ch.next = nullptr;  // the next call: this item's next 16 pairs, else the next down item
            if (chain & 1) {
                if (b + 16 < w.b1) {
                    ch.next = w.data, ch.next_blk = w.rb;
                } else if (j + 1 < mine) {
                    const MoeFItem v = item(j + 1);
                    if (v.b0 < v.b1) ch.next = v.data, ch.next_blk = v.rb;
                }
            }
            auto xrow = [&](int m) {
                if constexpr (XH) return static_cast<const __half *>(scratch) + static_cast<int64_t>(ps[m]) * inter;
                else return static_cast<const float *>(scratch) + static_cast<int64_t>(ps[m]) * inter;
            };
            auto out = [&](int m, int64_t nn, float v0, float v1) {
                const int64_t p = ps[m];
                float *cr = contrib + p * width;
                if (nn < width) cr[nn] = wts[p] * ((WF == 2 || WF == 3) ? v0 * __ldg(w.scale + nn) : v0);
                if (nn + 1 < width) cr[nn + 1] = wts[p] * ((WF == 2 || WF == 3) ? v1 * __ldg(w.scale + nn + 1) : v1);
            };
            if (mt > 8) tc_tile<WF, true, BX, NT, true>(w.data, mt, inter, width, 1, w.rb, tc_smem, xrow, out, bs, &ch);
            else tc_tile<WF, false, BX, NT, true>(w.data, mt, inter, width, 1, w.rb, tc_smem, xrow, out, bs, &ch);
        }
        // the last down item of the expert: every item of it is past its wait
        if (threadIdx.x == 0 && atomicAdd(done2 + w.u, 1u) == static_cast<unsigned>(bpe2 - 1)) {
            atomicExch(done1 + w.u, 0u);
            atomicExch(done2 + w.u, 0u);
        }
    }
}

// ---- prefill experts (bf16 weights and activations, steps above kGemvMaxRows rows): a tiled
// GEMM per (pass of kGeRows pairs, kGeCols weight rows). Both operands are staged in shared
// memory by cp.async (A: the pass's x rows gathered through plist, rows past the pass zero;
// B: the weight rows), kGeK of K per stage, kGeStages in flight; fragments come by ldmatrix
// (rows padded to kGePitch: conflict-free) and the sums are m16n8k16 MMAs in k order. 8 warps
// as 2 (M) x 4 (N), each 64 x 32 of the tile (every fragment feeds 4 MMAs). An output
// element's sum runs over K in the same order whatever its tile or pass, so a pair's rows do
// not depend on the batch.
// GATEUP: A = x16[token(p)], B = w1 ([2*inter][width], gate/up interleaved), GLU -> s16[p].
// else:   A = s16[p], B = w2 ([width][inter]), contrib[p][n] = wts[p] * sum.
constexpr int kGeCols = 128, kGeK = 32, kGePitch = kGeK + 8, kGeStages = 4, kGeThreads = 256;
constexpr int kGeStage = (kGeRows + kGeCols) * kGePitch;  // elements per stage: A rows, then B rows
constexpr size_t kGeSmem = sizeof(uint16_t) * kGeStages * kGeStage;

template <bool GATEUP>
__global__ void __launch_bounds__(kGeThreads, 2)
    moe_gemm_kernel(float *contrib, uint16_t *s16, const uint16_t *__restrict__ x16, MoeLists L, int k,
                    const Backend::ExpertW *table, const float *wts, int64_t width, int64_t inter, int silu, float b1,
                    float b2) {
    extern __shared__ __align__(16) uint16_t ge_smem[];
    pdl_wait();  // the lists, x16 (gate/up) or s16 (down) come from the kernels just before
    pdl_trigger();
    const int64_t K = GATEUP ? width : inter, NR = GATEUP ? 2 * inter : width;
    const int ntiles = static_cast<int>((NR + kGeCols - 1) / kGeCols), nk = static_cast<int>(K / kGeK);
    const int32_t *plist = L.plist(), *pu = L.pass_u(), *pb = L.pass_b(), *exps = L.exps(), *start = L.start();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp >> 2, wn = warp & 3;                // the warp's 64 rows, 32 columns
    const int lr = threadIdx.x >> 2, lc = threadIdx.x & 3;  // this thread stages rows lr, lr + 64, chunk lc
    const int T = L.meta[1] * ntiles;
    for (int item = blockIdx.x; item < T; item += gridDim.x) {
        const int q = item / ntiles, nt = item % ntiles, u = pu[q];
        const int32_t b0 = pb[q];
        const int count = min(kGeRows, start[u + 1] - b0);
        const Backend::ExpertW &ew = table[exps[u]];
        const uint16_t *W = static_cast<const uint16_t *>((GATEUP ? ew.w1 : ew.w2).data);
        // sources of this thread's A rows (zero past the pass) and B rows (clamped past NR)
        bool arow[2];
        const uint16_t *asrc[2], *bsrc[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int r = lr + 64 * j;
            arow[j] = r < count;
            asrc[j] = x16;
            if (arow[j]) {
                const int64_t p = plist[b0 + r];
                asrc[j] = GATEUP ? x16 + (p / k) * width : s16 + p * inter;
            }
            bsrc[j] = W + imin(static_cast<int64_t>(nt) * kGeCols + r, NR - 1) * K;
        }
        auto load = [&](int st, int ks) {
            uint16_t *as = ge_smem + st * kGeStage, *bs = as + kGeRows * kGePitch;
            const int64_t k0 = static_cast<int64_t>(ks) * kGeK + lc * 8;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                cp_async16_zfill(as + (lr + 64 * j) * kGePitch + lc * 8, asrc[j] + (arow[j] ? k0 : 0), arow[j]);
                cp_async16(bs + (lr + 64 * j) * kGePitch + lc * 8, bsrc[j] + k0);
            }
        };
        float c[4][4][4];
#pragma unroll
        for (int mi = 0; mi < 4; ++mi)
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) c[mi][ni][0] = c[mi][ni][1] = c[mi][ni][2] = c[mi][ni][3] = 0.0f;
        __syncthreads();  // the previous item's stages are no longer read
#pragma unroll
        for (int st = 0; st < kGeStages - 1; ++st) {
            if (st < nk) load(st, st);
            cp_async_commit();
        }
        for (int ks = 0; ks < nk; ++ks) {
            cp_async_wait<kGeStages - 2>();  // stage ks has landed (this thread's copies)
            __syncthreads();                 // everyone's; and stage ks - 1 is no longer read
            const int nx = ks + kGeStages - 1;
            if (nx < nk) load(nx % kGeStages, nx);
            cp_async_commit();
            const uint16_t *as = ge_smem + (ks % kGeStages) * kGeStage, *bs = as + kGeRows * kGePitch;
#pragma unroll
            for (int kk = 0; kk < kGeK / 16; ++kk) {
                uint32_t a[4][4], bb[2][4];
#pragma unroll
                for (int mi = 0; mi < 4; ++mi)
                    ldsm_x4(a[mi], as + (wm * 64 + mi * 16 + (lane & 15)) * kGePitch + kk * 16 + (lane >> 4) * 8);
#pragma unroll
                for (int nj = 0; nj < 2; ++nj)
                    ldsm_x4(bb[nj], bs + (wn * 32 + nj * 16 + (lane & 7) + ((lane >> 4) << 3)) * kGePitch + kk * 16 +
                                        ((lane >> 3) & 1) * 8);
#pragma unroll
                for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                    for (int nj = 0; nj < 2; ++nj) {
                        mma_bf16(c[mi][2 * nj], a[mi][0], a[mi][1], a[mi][2], a[mi][3], bb[nj][0], bb[nj][1]);
                        mma_bf16(c[mi][2 * nj + 1], a[mi][0], a[mi][1], a[mi][2], a[mi][3], bb[nj][2], bb[nj][3]);
                    }
            }
        }
        cp_async_wait<0>();
        // epilogue: thread holds rows g, g + 8 of each m-tile at columns 2t, 2t + 1 of each n-tile
#pragma unroll
        for (int mi = 0; mi < 4; ++mi)
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int r = wm * 64 + mi * 16 + g + 8 * half;
                if (r >= count) continue;
                const int64_t p = plist[b0 + r];
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    const int64_t col = static_cast<int64_t>(nt) * kGeCols + wn * 32 + ni * 8 + 2 * t;
                    const float v0 = c[mi][ni][2 * half], v1 = c[mi][ni][2 * half + 1];
                    if constexpr (GATEUP) {  // col: gate row col/2; col + 1: its up row
                        if (col < NR)
                            s16[p * inter + col / 2] =
                                f2bf(silu ? v0 * sigm(v0) * v1 : b1 * tanhf(v0 / b1) * sigm(v0) * (b2 * tanhf(v1 / b2)));
                    } else {
                        float *cr = contrib + p * width;
                        if (col < NR) cr[col] = wts[p] * v0;
                        if (col + 1 < NR) cr[col + 1] = wts[p] * v1;
                    }
                }
            }
    }
}

// ---- the CUTLASS prefill path (cutlass_ops): rows gathered in pair-list order, the GLU on the
// bf16 gate/up output, the weighted scatter of the down output back to pair order. Only the
// routed pairs (start[U] of them: pairs of non-resident experts are not listed) have rows.
__device__ __forceinline__ int32_t moe_routed(MoeLists L) { return L.start()[L.meta[0]]; }

__global__ void moe_gather_rows_kernel(uint16_t *xs, const uint16_t *__restrict__ x16, MoeLists L, int k, int64_t width) {
    const int64_t i = blockIdx.x;
    if (i >= moe_routed(L)) return;
    const int64_t t = L.plist()[i] / k;
    const uint4 *src = reinterpret_cast<const uint4 *>(x16 + t * width);
    uint4 *dst = reinterpret_cast<uint4 *>(xs + i * width);
    for (int c = threadIdx.x; c < width / 8; c += blockDim.x) dst[c] = src[c];
}

// four GLU outputs per thread: 8 interleaved gate/up values in, 4 bf16 out (inter % 4 == 0)
__global__ void moe_glu16_kernel(uint16_t *s16, const uint16_t *__restrict__ g16, MoeLists L, int64_t inter, int silu,
                                 float b1, float b2) {
    const int64_t n4 = static_cast<int64_t>(moe_routed(L)) * inter / 4;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const uint4 u = *reinterpret_cast<const uint4 *>(g16 + i * 8);  // (gate, up) x 4
        const uint32_t pr[4] = {u.x, u.y, u.z, u.w};
        uint16_t out[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float gv = __uint_as_float(pr[q] << 16), uv = __uint_as_float(pr[q] & 0xFFFF0000u);
            out[q] = f2bf(silu ? gv * sigm(gv) * uv : b1 * tanhf(gv / b1) * sigm(gv) * (b2 * tanhf(uv / b2)));
        }
        *reinterpret_cast<uint2 *>(s16 + i * 4) =
            make_uint2(uint32_t(out[0]) | (uint32_t(out[1]) << 16), uint32_t(out[2]) | (uint32_t(out[3]) << 16));
    }
}

// ---- fp8 experts (FP8B) on CUTLASS's blockwise GEMM: the activation rows in pair-list order as
// e4m3, one scale per row per 128 columns (s = amax / 448), stored in the GEMM's packed per-group
// layout: group u's scales are [ncb][M_u] at sfa + start[u] * ncb.
__device__ __forceinline__ int moe_group_of(MoeLists L, int32_t i) {  // u with start[u] <= i < start[u + 1]
    const int32_t *st = L.start();
    int lo = 0, hi = L.meta[0] - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (st[mid] <= i) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}
// A warp's 128-column chunk, 4 values per lane: e4m3 bytes for the lane's 4 columns at dst; the
// chunk's scale is returned (every lane gets it)
__device__ __forceinline__ float quant128(const float (&v)[4], uint8_t *dst) {
    float a = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const float s = a > 0.0f ? a / 448.0f : 1.0f, inv = a > 0.0f ? 448.0f / a : 1.0f;
    uint32_t q = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j)
        q |= static_cast<uint32_t>(__nv_cvt_float_to_fp8(v[j] * inv, __NV_SATFINITE, __NV_E4M3)) << (8 * j);
    *reinterpret_cast<uint32_t *>(dst) = q;
    return s;
}
// grid: a block per pair-list row (rows past the routed ones return); x16: bf16 [N][width]
__global__ void moe_gather_quant_kernel(uint8_t *xs8, float *sfa, const uint16_t *__restrict__ x16, MoeLists L, int k,
                                        int64_t width) {
    const int32_t i = blockIdx.x;
    if (i >= moe_routed(L)) return;
    const int64_t t = L.plist()[i] / k;
    const int u = moe_group_of(L, i);
    const int32_t s0 = L.start()[u], mu = L.start()[u + 1] - s0;
    const int ncb = static_cast<int>(width / 128), lane = threadIdx.x & 31;
    for (int cb = threadIdx.x >> 5; cb < ncb; cb += blockDim.x >> 5) {
        const uint2 r = *reinterpret_cast<const uint2 *>(x16 + t * width + cb * 128 + lane * 4);
        const float v[4] = {__uint_as_float(r.x << 16), __uint_as_float(r.x & 0xFFFF0000u), __uint_as_float(r.y << 16),
                            __uint_as_float(r.y & 0xFFFF0000u)};
        const float sc = quant128(v, xs8 + static_cast<int64_t>(i) * width + cb * 128 + lane * 4);
        if (lane == 0) sfa[static_cast<int64_t>(s0) * ncb + static_cast<int64_t>(cb) * mu + (i - s0)] = sc;
    }
}
// the GLU of the gate/up output (bf16 [rows][2*inter], gate/up interleaved), quantised for the
// down GEMM: s8 e4m3 [rows][inter] and its scales as above (fp32 GLU values, not rounded to bf16)
__global__ void moe_glu_quant_kernel(uint8_t *s8, float *sfa, const uint16_t *__restrict__ g16, MoeLists L, int64_t inter,
                                     int silu, float b1, float b2) {
    const int32_t i = blockIdx.x;
    if (i >= moe_routed(L)) return;
    const int u = moe_group_of(L, i);
    const int32_t s0 = L.start()[u], mu = L.start()[u + 1] - s0;
    const int ncb = static_cast<int>(inter / 128), lane = threadIdx.x & 31;
    for (int cb = threadIdx.x >> 5; cb < ncb; cb += blockDim.x >> 5) {
        const uint4 r = *reinterpret_cast<const uint4 *>(g16 + static_cast<int64_t>(i) * 2 * inter + (cb * 128 + lane * 4) * 2);
        const uint32_t pr[4] = {r.x, r.y, r.z, r.w};  // (gate, up) x 4
        float v[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float gv = __uint_as_float(pr[q] << 16), uv = __uint_as_float(pr[q] & 0xFFFF0000u);
            v[q] = silu ? gv * sigm(gv) * uv : b1 * tanhf(gv / b1) * sigm(gv) * (b2 * tanhf(uv / b2));
        }
        const float sc = quant128(v, s8 + static_cast<int64_t>(i) * inter + cb * 128 + lane * 4);
        if (lane == 0) sfa[static_cast<int64_t>(s0) * ncb + static_cast<int64_t>(cb) * mu + (i - s0)] = sc;
    }
}

template <typename T>
__global__ void moe_scatter_kernel(float *contrib, const T *__restrict__ dn, MoeLists L, const float *__restrict__ wts,
                                   int64_t width) {
    const int64_t i = blockIdx.x;
    if (i >= moe_routed(L)) return;
    const int64_t p = L.plist()[i];
    const float w = wts[p];
    const T *src = dn + i * width;
    float4 *dst = reinterpret_cast<float4 *>(contrib + p * width);
    for (int c = threadIdx.x; c < width / 4; c += blockDim.x) {
        float4 v = row4(src, c);
        v.x *= w, v.y *= w, v.z *= w, v.w *= w;
        dst[c] = v;
    }
}

template <typename T>
__global__ void moe_reduce_kernel(float *acc, const float *contrib, int64_t N, int k, int64_t width,
                                  const float *addend, const T *dn, const int32_t *inv, const float *wts) {
    pdl_wait();
    pdl_trigger();
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < N * width;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t t = i / width, c = i % width;
        float a = 0.0f;
        for (int j = 0; j < k; ++j) a = a + pair_val(contrib, dn, inv, wts, t * k + j, width, c);  // j order, no FMA
        if (addend) a = a + addend[i];
        acc[i] = a;
    }
}

__global__ void count_routes_kernel(uint32_t *counts, const int32_t *idx, int64_t n, int64_t n_counts) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int32_t e = idx[i];
        if (e >= 0 && e < n_counts) atomicAdd(counts + e, 1u);
    }
}

__global__ void scatter_rows_kernel(float *dst, const float *src, const int32_t *rows, int64_t n, int64_t width) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n * width;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        dst[static_cast<int64_t>(rows[i / width]) * width + i % width] = src[i];
}

}  // namespace
}  // namespace fullm
