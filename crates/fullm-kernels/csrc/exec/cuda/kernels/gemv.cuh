// SPDX-License-Identifier: Apache-2.0
// kernels/gemv.cuh - the small-M GEMV paths: scalar and vector warps, the tensor-core tile
// (tc_tile, tc_tile_i4) and its kernels, the grouped GEMV, their launchers, the cuBLAS path's
// helper kernels.
//
// Part of cuda_backend.cu's translation unit, included there after the CUDA and standard
// headers, in the order common, formats, gemv, convert, router, elementwise, copy, kimi, mla,
// attention, sampling, moe (each file uses only those before it). Kernels and their device
// helpers: no state.
#pragma once

namespace fullm {
namespace {

// ------------------------------------------------------------------ GEMV (M <= 16)
// One warp per output row. Lanes walk 16-byte chunks c = lane, lane+32, ...; within a chunk
// elements are accumulated in order with explicit fmaf. That sequence is the same for every
// M and for the vector and scalar paths, so each y[m][n] has one fixed summation order.
// The vector path computes R output rows per warp so every x chunk it loads feeds R rows:
// for 1-byte weights x is 4x the weight bytes, and without the reuse the kernel is bound by
// x loads, not by HBM. R is 4, or 2 when 4 would leave SMs idle (a function of the shape
// only). Each row's FMA sequence (chunk order, then element order) is unchanged by R, so
// results are identical bit for bit.
// x rows are ldx floats apart. blockIdx.y selects one of several independent GEMVs of the
// same shape (MLA's per-head un-absorb): each operand advances by its GemvBatch stride.
// The grouped GEMV runs several matrices, each with its own x and y, in one launch (the
// projections that read one hidden state): blocks are dealt to the jobs in order and every
// row is computed by the same warp routine, so grouping changes no result bit either.
constexpr int kGemvWarps = 8;

struct GemvBatch {
    int64_t w = 0, x = 0, y = 0;  // elements of W, floats of x, floats of y
};

template <int WF, int MAXM, int R, int WPB = kGemvWarps>
__device__ __forceinline__ void gemv_vec_warp(const void *__restrict__ W, const float *__restrict__ scale,
                                              const float *__restrict__ x, int M, int64_t K, int64_t N,
                                              float *__restrict__ y, int64_t ldy, int64_t ldx, int64_t blk) {
    constexpr int E = WT<WF>::kElem;
    constexpr int P = 16 / E;
    const int lane = threadIdx.x & 31;
    const int64_t nchunk = K / P;  // the vector path needs K * E % 16 == 0
    const int64_t nb = (blk * WPB + (threadIdx.x >> 5)) * R;
    if (nb >= N) return;  // warp-uniform
    const int nr = static_cast<int>(imin(R, N - nb));
    float acc[R][MAXM];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < MAXM; ++m) acc[r][m] = 0.0f;
    const uint4 *wr[R];
#pragma unroll
    for (int r = 0; r < R; ++r)
        wr[r] = reinterpret_cast<const uint4 *>(static_cast<const char *>(W) + (nb + (r < nr ? r : 0)) * K * E);
    for (int64_t c = lane; c < nchunk; c += 32) {
        float w[R][P];
#pragma unroll
        for (int r = 0; r < R; ++r) decode16<WF>(__ldg(wr[r] + c), w[r]);
        const int64_t k0 = c * P;
#pragma unroll
        for (int m = 0; m < MAXM; ++m) {
            if (m < M) {
                const float4 *xv = reinterpret_cast<const float4 *>(x + m * ldx + k0);
#pragma unroll
                for (int q = 0; q < P / 4; ++q) {
                    const float4 xx = __ldg(xv + q);
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        acc[r][m] = fmaf(w[r][4 * q + 0], xx.x, acc[r][m]);
                        acc[r][m] = fmaf(w[r][4 * q + 1], xx.y, acc[r][m]);
                        acc[r][m] = fmaf(w[r][4 * q + 2], xx.z, acc[r][m]);
                        acc[r][m] = fmaf(w[r][4 * q + 3], xx.w, acc[r][m]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (r < nr) {  // warp-uniform
            const float sc = (WF == 2 || WF == 3) ? __ldg(scale + nb + r) : 1.0f;
#pragma unroll
            for (int m = 0; m < MAXM; ++m) {
                if (m < M) {
                    const float v = warp_sum(acc[r][m]);
                    if (lane == 0) y[m * ldy + nb + r] = v * sc;
                }
            }
        }
    }
}

template <int WF, int MAXM>
__device__ __forceinline__ void gemv_scalar_warp(const void *__restrict__ W, const float *__restrict__ scale,
                                                 const float *__restrict__ x, int M, int64_t K, int64_t N,
                                                 float *__restrict__ y, int64_t ldy, int64_t ldx, int64_t blk) {
    constexpr int P = 16 / WT<WF>::kElem;
    const int lane = threadIdx.x & 31;
    const int64_t nchunk = (K + P - 1) / P;
    const int64_t n = blk * kGemvWarps + (threadIdx.x >> 5);
    if (n >= N) return;  // warp-uniform
    float acc[MAXM];
#pragma unroll
    for (int m = 0; m < MAXM; ++m) acc[m] = 0.0f;
    for (int64_t c = lane; c < nchunk; c += 32) {
        const int64_t k0 = c * P;
        float w[P];
#pragma unroll
        for (int p = 0; p < P; ++p) w[p] = (k0 + p < K) ? WT<WF>::at(W, n * K + k0 + p) : 0.0f;
#pragma unroll
        for (int m = 0; m < MAXM; ++m) {
            if (m < M) {
#pragma unroll
                for (int p = 0; p < P; ++p)
                    if (k0 + p < K) acc[m] = fmaf(w[p], __ldg(x + m * ldx + k0 + p), acc[m]);
            }
        }
    }
    const float sc = (WF == 2 || WF == 3) ? __ldg(scale + n) : 1.0f;
#pragma unroll
    for (int m = 0; m < MAXM; ++m) {
        if (m < M) {  // warp-uniform
            const float v = warp_sum(acc[m]);
            if (lane == 0) y[m * ldy + n] = v * sc;
        }
    }
}

template <int WF, int MAXM, bool VEC, int R>
__global__ void __launch_bounds__(256)
    gemv_kernel(const void *__restrict__ W, const float *__restrict__ scale, const float *__restrict__ x, int M,
                int64_t K, int64_t N, float *__restrict__ y, int64_t ldy, int64_t ldx, GemvBatch bt) {
    W = static_cast<const char *>(W) + static_cast<int64_t>(blockIdx.y) * bt.w * WT<WF>::kElem;
    x += static_cast<int64_t>(blockIdx.y) * bt.x;
    y += static_cast<int64_t>(blockIdx.y) * bt.y;
    if constexpr (VEC) gemv_vec_warp<WF, MAXM, R>(W, scale, x, M, K, N, y, ldy, ldx, blockIdx.x);
    else gemv_scalar_warp<WF, MAXM>(W, scale, x, M, K, N, y, ldy, ldx, blockIdx.x);
}

// ------------------------------------------------------------------ tensor-core GEMV (M <= 16)
// y[m][n] = sum_k x[m][k] W[n][k] on the tensor cores for 1- and 2-byte weights (bf16, int8,
// fp8: every value is exact in TF32). mma.m16n8k8 TF32 with x = hi + lo: hi is x with the 13
// low bits cleared (exact TF32), lo = x - hi exactly, and the MMA truncates lo to TF32, so
// the products carry ~21 bits of x: ~1e-6 relative at worst. The split is two full-rate ops
// (a cvt.rna would be quarter-rate and bound the kernel).
// M is padded to 16 with zeros, so a row is computed by the same MMAs whatever M is
// (batch-invariant), and a group computes each tile exactly like a single call.
// Each warp computes one n8-tile (8 weight rows) over a share of K: S warps (1, 2, 4 or 8, a
// function of the matrix shape only) split K round-robin in 256-byte row groups and their
// partial tiles are summed in split order, so the result never depends on M or on grouping.
// In a group, lane t reads bytes q*64 + t*16 .. +16 of its row for q = 0..3 (so the 4 lanes
// of a row read whole 32-byte sectors together); sub-chunk q's elements 2s, 2s+1 are MMA step
// s's k-slots t and t+4. The weights
// stream through a per-warp ring of kTcStages groups in shared memory (cp.async, lane-major so
// each lane reads back its own 16-byte copies without bank conflicts): kTcStages-1 groups are
// in flight per warp without holding registers. In a chunk, lane (g =
// lane/4, t = lane%4) holds 16 bytes of row g (elements t*P .. t*P+P-1), and MMA step s uses
// its elements 2s, 2s+1 as k-slots t and t+4: a fixed permutation of k, applied to x alike.
constexpr int kTcWarps = 8;
constexpr int kTcStages = 4;
// dynamic shared memory of a tensor-core block: the weight rings, then the split partials
constexpr size_t kTcSmem = sizeof(uint4) * kTcWarps * kTcStages * 128 + sizeof(float) * kTcWarps * 128;

__device__ __forceinline__ void cp_async16(void *smem, const void *gmem) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(a), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

// bf16 activations (bx): x rounded to bf16, one m16n8k16 MMA per 4 elements of a lane's chunk.
__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// fp16 operands: the 8-bit float weights' path (e4m3 converts to fp16 exactly, in pairs)
__device__ __forceinline__ void mma_f16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// two e4m3 (the low 16 bits; byte 0 the lower k) as an f16x2 register, in one instruction
// (sm89+; below, cuda_fp8.h's conversion: the same values)
__device__ __forceinline__ uint32_t e4m3x2_f16x2(uint32_t pair) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(pair & 0xFFFFu), __NV_E4M3);
    return static_cast<uint32_t>(h.x) | (static_cast<uint32_t>(h.y) << 16);
#else
    uint32_t out;
    const unsigned short in = static_cast<unsigned short>(pair & 0xFFFFu);
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(out) : "h"(in));
    return out;
#endif
}
// the two e4m3 in the high 16 bits of w, as an f16x2 register (the register's half, no shift)
__device__ __forceinline__ uint32_t e4m3x2_hi_f16x2(uint32_t w) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
    return e4m3x2_f16x2(w >> 16);
#else
    uint32_t out;
    asm("{\n .reg .b16 lo, hi;\n mov.b32 {lo, hi}, %1;\n cvt.rn.f16x2.e4m3x2 %0, hi;\n}" : "=r"(out) : "r"(w));
    return out;
#endif
}
// two floats in fp16's range as an f16x2 register (lo in the low half)
__device__ __forceinline__ uint32_t f16x2_in_range(float lo, float hi) {
    const __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&v);
}
// two floats as an f16x2 register, clamped to fp16's range (lo in the low half)
__device__ __forceinline__ uint32_t f16x2(float lo, float hi) {
    const __half2 v = __floats2half2_rn(fminf(fmaxf(lo, -65504.0f), 65504.0f), fminf(fmaxf(hi, -65504.0f), 65504.0f));
    return *reinterpret_cast<const uint32_t *>(&v);
}

// one float as fp16 the way f16x2() rounds it (clamped, nearest even)
__device__ __forceinline__ __half f16c(float v) { return __float2half_rn(fminf(fmaxf(v, -65504.0f), 65504.0f)); }

// two floats as a bf16x2 register: lo in the low half (the lower k)
__device__ __forceinline__ uint32_t bf16x2(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&v);
}

// x -> (hi, lo) for a TF32 MMA: hi exact in TF32, lo = x - hi exact in fp32 (truncated by the MMA)
__device__ __forceinline__ void tf32_split(float x, uint32_t &hi, uint32_t &lo) {
    hi = __float_as_uint(x) & 0xFFFFE000u;
    lo = __float_as_uint(x - __uint_as_float(hi));
}

__device__ __forceinline__ void mma_tf32(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// x loads: the read-only path, or (XC: rows this kernel wrote, read after an acquire) ordinary
// loads, coherent with those writes (ld.global.nc is only for data no one writes during the kernel)
template <bool XC, class T>
__device__ __forceinline__ T ldx(const T *p) {
    if constexpr (XC) return *p;
    else return __ldg(p);
}

// Counters one block raises for others of the same grid: red_release after the block's barrier
// publishes everything the block wrote; wait_count spins (relaxed) until the count is reached,
// then acquires, so the waiting thread's later loads see those writes. Bounded: a grid that is
// not all resident (another process holding SMs) traps instead of hanging.
__device__ __forceinline__ unsigned ld_relaxed_gpu(const unsigned *p) {
    unsigned v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ unsigned ld_acquire_gpu(const unsigned *p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void red_release_gpu(unsigned *p, unsigned v) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void wait_count(const unsigned *p, unsigned need) {
    for (unsigned spins = 0; ld_relaxed_gpu(p) < need; ++spins) {
        if (spins > (1u << 25)) __trap();
        __nanosleep(64);
    }
    (void)ld_acquire_gpu(p);  // reads the last increment: synchronises with every release before it
}

template <int WF, int P, bool XC = false>
__device__ __forceinline__ void load_x(float (&v)[P], const float *p, bool ok) {
#pragma unroll
    for (int q = 0; q < P / 4; ++q) {
        const float4 f = ok ? ldx<XC>(reinterpret_cast<const float4 *>(p) + q) : make_float4(0.f, 0.f, 0.f, 0.f);
        v[4 * q] = f.x, v[4 * q + 1] = f.y, v[4 * q + 2] = f.z, v[4 * q + 3] = f.w;
    }
}

// One pass of this block's 8-row tiles of W ([N][K]) over M <= 16 rows of x. xrow(m) gives
// row m of x; out(m, n, v0, v1) receives the (unscaled) sums for weight rows n (even) and
// n + 1 (which may be >= N). Callable repeatedly within a kernel: it returns with the shared
// memory (kTcSmem bytes) free again.
// NT tiles per warp (S = 1 only): the warp streams NT neighbouring 8-row tiles together, their
// groups alternating in its ring; each tile's sums are exactly those of NT = 1.
// WF 4 (FP8B): bscale holds W's block scales (fp8_block.h); a group's 256 elements are two
// 128-wide K blocks (sub-chunks q 0-1 and 2-3), and a tile's 8 rows lie in one 128-row block, so
// each K block's sums are scaled by one value before they join the tile's sums (out() then
// receives scaled sums).
// ch (TcChain *): consecutive calls of a kernel sharing each warp's weight stream (S = 1 only).
//
// A chain of tc_tile calls (S = 1: each warp owns its ring, and nothing else of the shared memory
// is used): in its last kTcStages-1 iterations, where it would commit empty groups, a call issues
// the next call's first groups, and the next call skips its prologue, so items no longer start
// with the ring's fill latency (with e4m3 weights an expert's down rows are 3 groups: nearly all
// fill). The ring continues at the slot after the call's last group; calls end without draining
// it (the block barrier stays: without it bf16 decode was 0.1 ms/step slower, fp8b no faster). The
// same bytes reach the same MMAs in the same order: the sums are unchanged.
// Every call of a chain has the same K, N, NT (so the same group count, which must be at least
// kTcStages-1 for a call to pass; shorter calls just fill their own rings).
struct TcChain {
    const void *next = nullptr;  // the next call's W, or nullptr: no next call (or don't prefetch)
    int64_t next_blk = 0;        // the next call's blk
    int slot = 0;                // ring slot of this call's first group
    bool primed = false;         // this call's first kTcStages-1 groups are already in flight
    // x is still being written by other blocks: after streaming in its first weights, the call
    // waits until *ready >= need (wait_count) before reading x
    const unsigned *ready = nullptr;
    unsigned need = 0;
};
// p[i % N] for a runtime i, by selects: indexing a register array with it can put the array on
// the stack (the fp8b gate/up tile had 48 bytes of it; without, its decode step went 9.3 -> 8.9 ms
// at 16 tokens before the prefetch)
template <int N, class T>
__device__ __forceinline__ T pick(const T (&p)[N], int64_t i) {
    T r = p[0];
    if constexpr (N > 1) {
        const int k = static_cast<int>(i % N);
#pragma unroll
        for (int j = 1; j < N; ++j) r = k == j ? p[j] : r;
    }
    return r;
}
// INT4G tiles (the expert kernels: S = 1, chained, NT = 2: a warp's 16 rows are one MMA tile).
// The weights are the MMA's A operand and the tokens its B operand ("swap AB"): int4_group.h
// stores each 16-row tile as A fragments, a 64-column chunk being 512 contiguous bytes whose
// lane's 16 are its four steps' fragments, unpacked straight into the A registers; lane (g, t)
// loads token g's (and with TWO token g + 8's) 32 bytes of x per chunk, whose words 2s, 2s + 1 are
// step s's B registers. One MMA per step and 8 tokens (tokens as A rows needed the fragments
// assembled from two x rows: a register move per word, ~30% of the old tile's instructions, which
// left it issue-bound at ~50% of DRAM bandwidth). The sums come out as (tile rows 2g, 2g + 1:
// adjacent matrix rows) x (tokens 2t, 2t + 1), so out() gets a token's row pair as from the other
// tiles. Ring: a group is 256 columns, four chunks = 2 KB per warp, kTcStages of them. Per
// 128-column scale group the sums are folded with the lane's two rows' scales (one float2: the
// rows are adjacent); out() receives scaled sums. The chain (TcChain) works as in tc_tile, its
// groups being these.
template <bool TWO, int NT, bool XC, class XRow, class Out>
__device__ __forceinline__ void tc_tile_i4(const void *__restrict__ W, int M, int64_t K, int64_t N, int64_t blk,
                                           void *smem, XRow xrow, Out out, const float *__restrict__ scale, TcChain *ch) {
    static_assert(NT == 2, "int4: a warp's 16 rows are one MMA tile");
    constexpr int CK = 4, SLOT = CK * 32, STG = kTcStages;  // chunks per group; uint4 per ring slot
    static_assert(STG * SLOT == kTcStages * 128 && (STG & (STG - 1)) == 0, "ring slots wrap by masking");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int64_t ntile = N / 16, tile = blk * kTcWarps + warp, count = K / 256, TB = K * 8;  // TB: bytes per tile
    auto tilep = [&](const void *Wm, int64_t T) -> const uint4 * {  // tiles past N read the last one, unwritten
        return reinterpret_cast<const uint4 *>(static_cast<const char *>(Wm) + imin(T, ntile - 1) * TB) + lane;
    };
    const uint4 *wr = tilep(W, tile), *nwr = wr;
    const int base = ch->slot;
    const bool primed = ch->primed, pass = ch->next != nullptr && count >= STG - 1;
    if (pass) nwr = tilep(ch->next, ch->next_blk * kTcWarps + warp);
    uint4 *ring = static_cast<uint4 *>(smem) + warp * kTcStages * 128;  // [stage][chunk][lane]
    auto at = [&](int64_t i) { return ring + ((base + i) & (STG - 1)) * SLOT + lane; };
    auto copy = [&](uint4 *dst, const uint4 *src, int64_t a) {  // group a: chunks 4a .. 4a + 3
#pragma unroll
        for (int c = 0; c < CK; ++c) cp_async16(dst + c * 32, src + (CK * a + c) * 32);
    };
    if (!primed) {
#pragma unroll
        for (int i = 0; i < STG - 1; ++i) {
            if (i < count) copy(at(i), wr, i);
            cp_async_commit();
        }
    }
    pdl_wait();
    if (ch->ready) wait_count(ch->ready, ch->need);
    const bool ok0 = g < M, ok1 = TWO && g + 8 < M;
    const __half *x0 = ok0 ? xrow(g) + 16 * t : nullptr;
    const __half *x1 = ok1 ? xrow(g + 8) + 16 * t : nullptr;
    const float *srow = scale + imin(tile, ntile - 1) * 16 + 2 * g;  // the lane's rows' scales, group 0
    float c0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, c1[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // tokens 0-7, 8-15
    for (int64_t i = 0; i < count; ++i) {
        if (i + STG - 1 < count) copy(at(i + STG - 1), wr, i + STG - 1);
        else if (pass) copy(at(i + STG - 1), nwr, i + STG - 1 - count);
        cp_async_commit();
        cp_async_wait<STG - 1>();  // group i has landed
        const uint4 *cur = at(i);
#pragma unroll
        for (int q = 0; q < 2; ++q) {  // 128-column scale group 2i + q: chunks 2q, 2q + 1
            const float2 sc = __ldg(reinterpret_cast<const float2 *>(srow + (2 * i + q) * N));
            const uint4 z = make_uint4(0, 0, 0, 0);
            uint4 xa[4], xb[4];  // [chunk][half of the 32-byte run]
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                const int64_t off = i * 256 + q * 128 + (h >> 1) * 64 + (h & 1) * 8;
                xa[h] = ok0 ? ldx<XC>(reinterpret_cast<const uint4 *>(x0 + off)) : z;
                xb[h] = TWO && ok1 ? ldx<XC>(reinterpret_cast<const uint4 *>(x1 + off)) : z;
            }
            float ta[4] = {0.0f, 0.0f, 0.0f, 0.0f}, tb[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
                const uint4 u = cur[(2 * q + cc) * 32];
                const uint32_t ww[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                for (int st = 0; st < 4; ++st) {  // step st: x words 2st, 2st + 1 of the chunk's run
                    uint32_t a[4];
                    int4x8_f16x2(ww[st], a);
                    const uint4 &va = xa[2 * cc + (st >> 1)];
                    mma_f16(ta, a[0], a[1], a[2], a[3], st & 1 ? va.z : va.x, st & 1 ? va.w : va.y);
                    if constexpr (TWO) {
                        const uint4 &vb = xb[2 * cc + (st >> 1)];
                        mma_f16(tb, a[0], a[1], a[2], a[3], st & 1 ? vb.z : vb.x, st & 1 ? vb.w : vb.y);
                    }
                }
            }
            c0[0] = fmaf(sc.x, ta[0], c0[0]), c0[1] = fmaf(sc.x, ta[1], c0[1]);
            c0[2] = fmaf(sc.y, ta[2], c0[2]), c0[3] = fmaf(sc.y, ta[3], c0[3]);
            if constexpr (TWO) {
                c1[0] = fmaf(sc.x, tb[0], c1[0]), c1[1] = fmaf(sc.x, tb[1], c1[1]);
                c1[2] = fmaf(sc.y, tb[2], c1[2]), c1[3] = fmaf(sc.y, tb[3], c1[3]);
            }
        }
    }
    if (!pass) cp_async_wait<0>();
    pdl_trigger();
    if (tile < ntile) {
        const int64_t n = tile * 16 + 2 * g;
        if (2 * t < M) out(2 * t, n, c0[0], c0[2]);
        if (2 * t + 1 < M) out(2 * t + 1, n, c0[1], c0[3]);
        if (TWO && 2 * t + 8 < M) out(2 * t + 8, n, c1[0], c1[2]);
        if (TWO && 2 * t + 9 < M) out(2 * t + 9, n, c1[1], c1[3]);
    }
    ch->slot = static_cast<int>((base + count) & (STG - 1));
    ch->primed = pass;
    __syncthreads();  // the ring is free for the next call
}

// XC: x rows written earlier in this kernel (ldx).
template <int WF, bool TWO, bool BX, int NT = 1, bool XC = false, class XRow, class Out, class Chain = std::nullptr_t>
__device__ __forceinline__ void tc_tile(const void *__restrict__ W, int M, int64_t K, int64_t N, int S, int64_t blk,
                                        void *smem, XRow xrow, Out out, const float *__restrict__ bscale = nullptr,
                                        Chain ch = nullptr) {
    constexpr bool CH = std::is_same_v<Chain, TcChain *>;
    static_assert(CH || std::is_same_v<Chain, std::nullptr_t>, "ch: a TcChain * or nothing");
    if constexpr (WF == 5) {  // int4: its own tile (expert kernels only)
        static_assert(CH && BX, "int4 tiles: the chained expert kernels, 16-bit activations");
        tc_tile_i4<TWO, NT, XC>(W, M, K, N, blk, smem, xrow, out, bscale, ch);
        return;
    } else {
    static_assert((kTcStages & (kTcStages - 1)) == 0, "ring slots wrap by masking");
    constexpr int E = WT<WF>::kElem, P = 16 / E, GE = 16 * P;  // GE: elements per 256-byte row group
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int split = warp % S;
    const int64_t n0 = (blk * (kTcWarps / S) + warp / S) * 8 * NT;
    const int64_t ngroup = K / GE;
    // rows past N (the last block) read row N-1 and are never written
    auto rowp = [&](const void *Wm, int64_t r0) -> const uint4 * {  // the lane's first chunk, tile at row r0
        return reinterpret_cast<const uint4 *>(static_cast<const char *>(Wm) + imin(r0 + g, N - 1) * K * E) + t;
    };
    constexpr int64_t gs = 16;  // 16-byte chunks from one 256-byte group of a row to the next
    const uint4 *wr[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) wr[j] = rowp(W, n0 + 8 * j);
    float c[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.0f;
    static_assert(WF != 4 || 16 * (16 / WT<WF>::kElem) == 2 * 128, "FP8B: a row group is two 128-wide K blocks");
    int64_t trb[NT];  // FP8B: each tile's 128-row block, and the row blocks per K block of the scales
#pragma unroll
    for (int j = 0; j < NT; ++j) trb[j] = imin(n0 + 8 * j, N - 1) / 128;
    const int64_t nrb = (N + 127) / 128;

    uint4 *ring = static_cast<uint4 *>(smem) + warp * kTcStages * 128;  // [stage][q][lane]
    float *red = reinterpret_cast<float *>(static_cast<uint4 *>(smem) + kTcWarps * kTcStages * 128);
    const int64_t count = NT * (split < ngroup ? (ngroup - split + S - 1) / S : 0);  // this warp's groups
    // CH: base = the slot of group 0; pass = this call issues the next one's first groups
    int base = 0;
    bool primed = false, pass = false;
    const uint4 *nwr[NT];  // CH: the next call's rows (S = 1, split 0)
#pragma unroll
    for (int j = 0; j < NT; ++j) nwr[j] = wr[j];
    if constexpr (CH) {
        base = ch->slot;
        primed = ch->primed;
        pass = ch->next != nullptr && count >= kTcStages - 1;
        if (pass) {
            const int64_t m0 = (ch->next_blk * kTcWarps + warp) * 8 * NT;
#pragma unroll
            for (int j = 0; j < NT; ++j) nwr[j] = rowp(ch->next, m0 + 8 * j);
        }
    }
    auto at = [&](int64_t i) { return ring + ((base + i) & (kTcStages - 1)) * 128 + lane; };  // group i's slot
    auto copy = [&](uint4 *dst, const uint4 *src) {
#pragma unroll
        for (int q = 0; q < 4; ++q) cp_async16(dst + q * 32, src + 4 * q);
    };
    auto issue = [&](int64_t i) {  // group i: tile i % NT, row group split + (i / NT) * S
        copy(at(i), pick(wr, i) + gs * (split + i / NT * S));
    };
    auto issue_next = [&](int64_t i) {  // the next call's group i (< kTcStages - 1), after this call's
        copy(at(count + i), pick(nwr, i) + gs * (i / NT));
    };
    if (!primed) {
#pragma unroll
        for (int i = 0; i < kTcStages - 1; ++i) {
            if (i < count) issue(i);
            cp_async_commit();  // empty groups keep the count uniform
        }
    }
    pdl_wait();  // the weights above stream in while the kernel producing x finishes
    if constexpr (CH) {
        if (ch->ready) wait_count(ch->ready, ch->need);  // ... or while other blocks write x
    }
    // x rows given as bf16 (rounded as the BX path rounds them): each 16-byte load is the A fragment
    using XP = decltype(xrow(0));
    constexpr bool XB = std::is_same_v<XP, const uint16_t *>;
    static_assert(!XB || (BX && WF == 1), "bf16 x rows: bf16 weights and activations only");
    // x rows given as fp16 (const __half *, clamped and rounded by their producer exactly as f16x2()
    // does here): the e4m3 weights' fp16 MMAs take them as A fragments, no per-element work
    constexpr bool XH = std::is_same_v<XP, const __half *>;
    static_assert(!XH || (BX && (WF == 3 || WF == 4)), "fp16 x rows: e4m3 weights with 16-bit activations only");
    const bool ok0 = g < M, ok1 = TWO && g + 8 < M;
    const XP x0 = ok0 ? xrow(g) + t * P : nullptr;
    const XP x1 = ok1 ? xrow(g + 8) + t * P : nullptr;
    for (int64_t i0 = 0; i0 < count; i0 += NT) {
#pragma unroll
    for (int jt = 0; jt < NT; ++jt) {
        const int64_t i = i0 + jt;
        if (i + kTcStages - 1 < count) issue(i + kTcStages - 1);
        else if (pass) issue_next(i + kTcStages - 1 - count);
        cp_async_commit();
        cp_async_wait<kTcStages - 1>();  // group i has landed
        const int64_t a = split + i0 / NT * S;
        const uint4 *cur = at(i);
        float tt[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // FP8B: one K block's sums, before its scale
        float(&cc)[4] = WF == 4 ? tt : c[jt];
        auto fold = [&](int64_t kb) {  // FP8B: K block kb's sums, scaled, into the tile's
            if constexpr (WF == 4) {
                const float sc = __ldg(bscale + kb * nrb + trb[jt]);
#pragma unroll
                for (int e = 0; e < 4; ++e) c[jt][e] = fmaf(sc, tt[e], c[jt][e]), tt[e] = 0.0f;
            }
        };
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            if (WF == 4 && q == 2) fold(2 * a);
            if constexpr (XH) {
                // the lane's 16 elements of each row: 32 bytes, words (2j, 2j + 1) = MMA j's A
                // k-slots, the same mapping as the converting path below
                const uint4 z = make_uint4(0, 0, 0, 0);
                const uint4 *pa = reinterpret_cast<const uint4 *>(x0 + a * GE + 4 * q * P);
                const uint4 *pb = reinterpret_cast<const uint4 *>(x1 + a * GE + 4 * q * P);
                const uint4 a0v = ok0 ? ldx<XC>(pa) : z, a1v = ok0 ? ldx<XC>(pa + 1) : z;
                const uint4 b0v = TWO && ok1 ? ldx<XC>(pb) : z, b1v = TWO && ok1 ? ldx<XC>(pb + 1) : z;
                const uint32_t A0[8] = {a0v.x, a0v.y, a0v.z, a0v.w, a1v.x, a1v.y, a1v.z, a1v.w};
                const uint32_t A1[8] = {b0v.x, b0v.y, b0v.z, b0v.w, b1v.x, b1v.y, b1v.z, b1v.w};
                const uint4 u = cur[q * 32];
                const uint32_t ww[4] = {u.x, u.y, u.z, u.w};
                uint32_t wh[8];
#pragma unroll
                for (int i = 0; i < 4; ++i) wh[2 * i] = e4m3x2_f16x2(ww[i]), wh[2 * i + 1] = e4m3x2_f16x2(ww[i] >> 16);
#pragma unroll
                for (int j = 0; j < 4; ++j) mma_f16(cc, A0[2 * j], A1[2 * j], A0[2 * j + 1], A1[2 * j + 1], wh[2 * j], wh[2 * j + 1]);
                continue;
            } else if constexpr (XB) {
                const uint4 z = make_uint4(0, 0, 0, 0);
                const uint4 xa = ok0 ? ldx<XC>(reinterpret_cast<const uint4 *>(x0 + a * GE + 4 * q * P)) : z;
                const uint4 xb = TWO && ok1 ? ldx<XC>(reinterpret_cast<const uint4 *>(x1 + a * GE + 4 * q * P)) : z;
                const uint4 u = cur[q * 32];
                mma_bf16(cc, xa.x, xb.x, xa.y, xb.y, u.x, u.y);
                mma_bf16(cc, xa.z, xb.z, xa.w, xb.w, u.z, u.w);
                continue;
            } else {
            float xa[P], xb[P];
            load_x<WF, P, XC>(xa, ok0 ? x0 + a * GE + 4 * q * P : nullptr, ok0);
            if constexpr (TWO) load_x<WF, P, XC>(xb, ok1 ? x1 + a * GE + 4 * q * P : nullptr, ok1);
            if constexpr (BX) {
                // MMA j takes the lane's elements 4j..4j+3: (4j, 4j+1) as k-slots (2t, 2t+1) and
                // (4j+2, 4j+3) as (2t+8, 2t+9), the same for x and W. bf16 weights are already
                // the B fragment.
                if constexpr (WF == 3 || WF == 4) {
                    // e4m3 weights: each pair to f16x2 in one instruction (exact), x rounded to
                    // fp16 (finer than bf16; clamped to its range), fp16 MMAs; the same k-slots as
                    // the bf16 path below. Decoding each byte to fp32 then repacking had the tile
                    // compute-bound at half the bytes.
                    const uint4 u = cur[q * 32];
                    const uint32_t ww[4] = {u.x, u.y, u.z, u.w};
                    uint32_t wh[P / 2];
#pragma unroll
                    for (int i = 0; i < 4; ++i) wh[2 * i] = e4m3x2_f16x2(ww[i]), wh[2 * i + 1] = e4m3x2_f16x2(ww[i] >> 16);
#pragma unroll
                    for (int j = 0; j < P / 4; ++j) {
                        const uint32_t a0 = f16x2(xa[4 * j], xa[4 * j + 1]), a2 = f16x2(xa[4 * j + 2], xa[4 * j + 3]);
                        uint32_t a1 = 0, a3 = 0;
                        if constexpr (TWO) a1 = f16x2(xb[4 * j], xb[4 * j + 1]), a3 = f16x2(xb[4 * j + 2], xb[4 * j + 3]);
                        mma_f16(cc, a0, a1, a2, a3, wh[2 * j], wh[2 * j + 1]);
                    }
                    continue;
                }
                uint32_t wv[P / 2];
                if constexpr (WF == 1) {
                    const uint4 u = cur[q * 32];
                    wv[0] = u.x, wv[1] = u.y, wv[2] = u.z, wv[3] = u.w;
                } else {
                    float wf[P];
                    decode16<WF>(cur[q * 32], wf);  // int8 / fp8 values are exact in bf16
#pragma unroll
                    for (int j = 0; j < P / 2; ++j) wv[j] = bf16x2(wf[2 * j], wf[2 * j + 1]);
                }
#pragma unroll
                for (int j = 0; j < P / 4; ++j) {
                    const uint32_t a0 = bf16x2(xa[4 * j], xa[4 * j + 1]), a2 = bf16x2(xa[4 * j + 2], xa[4 * j + 3]);
                    uint32_t a1 = 0, a3 = 0;
                    if constexpr (TWO) a1 = bf16x2(xb[4 * j], xb[4 * j + 1]), a3 = bf16x2(xb[4 * j + 2], xb[4 * j + 3]);
                    mma_bf16(cc, a0, a1, a2, a3, wv[2 * j], wv[2 * j + 1]);
                }
                continue;
            }
            float wf[P];
            decode16<WF>(cur[q * 32], wf);
#pragma unroll
            for (int s = 0; s < P / 2; ++s) {
                uint32_t h0, h2, l0, l2, h1 = 0, h3 = 0, l1 = 0, l3 = 0;
                tf32_split(xa[2 * s], h0, l0);
                tf32_split(xa[2 * s + 1], h2, l2);
                if constexpr (TWO) {
                    tf32_split(xb[2 * s], h1, l1);
                    tf32_split(xb[2 * s + 1], h3, l3);
                }
                const uint32_t b0 = __float_as_uint(wf[2 * s]), b1 = __float_as_uint(wf[2 * s + 1]);
                mma_tf32(cc, l0, l1, l2, l3, b0, b1);
                mma_tf32(cc, h0, h1, h2, h3, b0, b1);
            }
            }
        }
        fold(2 * a + 1);
    }
    }
    if (!pass) cp_async_wait<0>();  // else the next call's first groups stay in flight
    pdl_trigger();

    if constexpr (NT == 1) {
        if (S > 1) {
#pragma unroll
            for (int i = 0; i < 4; ++i) red[(warp * 32 + lane) * 4 + i] = c[0][i];
            __syncthreads();
            if (split == 0)
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    for (int q = 1; q < S; ++q) c[0][i] += red[((warp + q) * 32 + lane) * 4 + i];
        }
    }
    if (split == 0) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            if (g < M) out(g, n0 + 8 * j + 2 * t, c[j][0], c[j][1]);
            if (TWO && g + 8 < M) out(g + 8, n0 + 8 * j + 2 * t, c[j][2], c[j][3]);
        }
    }
    if constexpr (CH) {  // the warp's ring carries on, the next call's groups in flight
        ch->slot = static_cast<int>((base + count) & (kTcStages - 1));
        ch->primed = pass;
    }
    __syncthreads();  // the rings and partials are free for the next call
    }
}

// y[m][n] = W[n] . x[m] (times the row scale for int8 / fp8)
template <int WF, bool TWO, bool BX, typename XT = float>
__device__ __forceinline__ void tc_gemv_block(const void *__restrict__ W, const float *__restrict__ scale,
                                              const XT *__restrict__ x, int M, int64_t K, int64_t N,
                                              float *__restrict__ y, int64_t ldy, int64_t ldx, int S, int64_t blk,
                                              void *smem) {
    tc_tile<WF, TWO, BX>(
        W, M, K, N, S, blk, smem, [&](int m) { return x + static_cast<int64_t>(m) * ldx; },
        [&](int m, int64_t n, float v0, float v1) {
            float *yr = y + static_cast<int64_t>(m) * ldy;
            if (n < N) yr[n] = (WF == 2 || WF == 3) ? v0 * __ldg(scale + n) : v0;
            if (n + 1 < N) yr[n + 1] = (WF == 2 || WF == 3) ? v1 * __ldg(scale + n + 1) : v1;
        });
}


// Warps sharing a tile for a matrix of `rows` x K (elements of E bytes): the smallest S that
// puts ~1024 warps to work (one wave of blocks, each warp streaming a long run of the row),
// keeping at least four row groups per warp.
// (FULLM_TC_WARPS and FULLM_TC_GROUPS override the 1024 and the 4, for tuning.)


int tc_split(int64_t rows, int64_t K, int E) {
    static const int target = tc_env("FULLM_TC_WARPS", 1024), min_groups = tc_env("FULLM_TC_GROUPS", 4);
    const int64_t tiles = (rows + 7) / 8, ngroup = K * E / 256;
    int S = 1;
    while (S < kTcWarps && tiles * S < target && ngroup >= min_groups * S) S *= 2;
    return S;
}

// XB: x is bf16 (uint16_t); XH: x is fp16 (e4m3 weights, as x_to_f16_kernel rounds it); else fp32
template <int WF, bool TWO, bool BX, bool XB = false, bool XH = false>
__global__ void __launch_bounds__(kTcWarps * 32, 2)
    tc_gemv_kernel(const void *__restrict__ W, const float *__restrict__ scale, const void *__restrict__ xv, int M,
                   int64_t K, int64_t N, float *__restrict__ y, int64_t ldy, int64_t ldx, GemvBatch bt, int S) {
    extern __shared__ uint4 tc_smem[];
    using XT = std::conditional_t<XB, uint16_t, std::conditional_t<XH, __half, float>>;
    const XT *x = static_cast<const XT *>(xv) + static_cast<int64_t>(blockIdx.y) * bt.x;
    W = static_cast<const char *>(W) + static_cast<int64_t>(blockIdx.y) * bt.w * WT<WF>::kElem;
    y += static_cast<int64_t>(blockIdx.y) * bt.y;
    tc_gemv_block<WF, TWO, BX, XT>(W, scale, x, M, K, N, y, ldy, ldx, S, blockIdx.x, tc_smem);
}


constexpr int kMaxGemvJobs = 8;

struct GemvJob {
    const void *W;
    const float *scale;
    const void *x;  // fp32, or bf16 in the XB kernels
    float *y;
    int64_t K, N, ldy;
    int32_t fmt, block0, split;
};

struct GemvJobs {
    GemvJob j[kMaxGemvJobs];
    int n;
};

// Jobs of 1-2 byte weights, each with its own split; every row is computed as in a single call.
// XB: every job is bf16 weights with bf16 x; XH: every job is fp8 weights with fp16 x
template <bool TWO, bool BX, bool XB = false, bool XH = false>
__global__ void __launch_bounds__(kTcWarps * 32, 2) gemv_group_kernel(const __grid_constant__ GemvJobs jobs, int M) {
    extern __shared__ uint4 tc_smem[];
    int i = 0;
    while (i + 1 < jobs.n && static_cast<int32_t>(blockIdx.x) >= jobs.j[i + 1].block0) ++i;
    const GemvJob &g = jobs.j[i];
    const int64_t blk = static_cast<int64_t>(blockIdx.x) - g.block0;
    if constexpr (XH) {
        tc_gemv_block<3, TWO, true, __half>(g.W, g.scale, static_cast<const __half *>(g.x), M, g.K, g.N, g.y, g.ldy,
                                            g.K, g.split, blk, tc_smem);
    } else if constexpr (XB) {
        tc_gemv_block<1, TWO, true, uint16_t>(g.W, g.scale, static_cast<const uint16_t *>(g.x), M, g.K, g.N, g.y,
                                              g.ldy, g.K, g.split, blk, tc_smem);
    } else {
        const float *x = static_cast<const float *>(g.x);
        switch (g.fmt) {
            case 1: tc_gemv_block<1, TWO, BX>(g.W, g.scale, x, M, g.K, g.N, g.y, g.ldy, g.K, g.split, blk, tc_smem); break;
            case 2: tc_gemv_block<2, TWO, BX>(g.W, g.scale, x, M, g.K, g.N, g.y, g.ldy, g.K, g.split, blk, tc_smem); break;
            default: tc_gemv_block<3, TWO, BX>(g.W, g.scale, x, M, g.K, g.N, g.y, g.ldy, g.K, g.split, blk, tc_smem); break;
        }
    }
}

int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Rows per warp: 4 unless that leaves the GPU under ~90% filled with blocks, then 2.
int gemv_rows_per_warp(int64_t rows, int sms) {
    return ceil_div(rows, 4 * kGemvWarps) * 10 >= int64_t{9} * sms ? 4 : 2;
}

int wfmt_bytes(WFmt f) { return f == WFmt::F32 ? 4 : f == WFmt::BF16 ? 2 : 1; }

// The tensor-core path: 1-2 byte weights, rows a whole number of 256-byte groups, 16-byte x rows.
bool gemv_tc_ok(const Weight &w, const float *x, int64_t ldx) {
    return w.fmt != WFmt::F32 && (w.cols * wfmt_bytes(w.fmt)) % 256 == 0 &&
           reinterpret_cast<uintptr_t>(w.data) % 16 == 0 && reinterpret_cast<uintptr_t>(x) % 16 == 0 && ldx % 4 == 0;
}

bool gemv_vec_ok(const Weight &w, const float *x, int64_t ldx) {
    return (w.cols * wfmt_bytes(w.fmt)) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data) % 16 == 0 &&
           reinterpret_cast<uintptr_t>(x) % 16 == 0 && ldx % 4 == 0;
}

#define SK_BY_M(M, CALL)          \
    do {                          \
        if ((M) <= 1) CALL(1);    \
        else if ((M) <= 2) CALL(2); \
        else if ((M) <= 4) CALL(4); \
        else if ((M) <= 8) CALL(8); \
        else CALL(16);            \
    } while (0)

template <int WF, bool VEC, int R>
void launch_gemv(const Weight &w, const float *x, int M, float *y, int64_t ldy, int64_t ldx, GemvBatch bt,
                 int nbatch, cudaStream_t s) {
    const int64_t rows_per_block = kGemvWarps * (VEC ? R : 1);
    const dim3 grid(static_cast<unsigned>(ceil_div(w.rows, rows_per_block)), static_cast<unsigned>(nbatch)),
        block(32 * kGemvWarps);
#define SK_GEMV(MM) \
    gemv_kernel<WF, MM, VEC, R><<<grid, block, 0, s>>>(w.data, w.scale, x, M, w.cols, w.rows, y, ldy, ldx, bt)
    SK_BY_M(M, SK_GEMV);
#undef SK_GEMV
}


template <int WF, bool TWO, bool BX, bool XB = false, bool XH = false>
void launch_tc(const Weight &w, const void *x, int M, float *y, int64_t ldy, int64_t ldx, GemvBatch bt, int nbatch,
               cudaStream_t s) {
    const int S = tc_split(w.rows * nbatch, w.cols, WT<WF>::kElem);
    const dim3 grid(static_cast<unsigned>(ceil_div(w.rows, 8 * (kTcWarps / S))), static_cast<unsigned>(nbatch));
    launch_pdl(tc_gemv_kernel<WF, TWO, BX, XB, XH>, grid, dim3(kTcWarps * 32), kTcSmem, s, w.data, w.scale, x, M, w.cols,
               w.rows, y, ldy, ldx, bt, S);
}

// w describes ONE matrix of the batch; scaled formats (int8, fp8) are not batched.
// xb: x already as packed bf16 rows (bf16 weights with bf16 activations; one matrix)
// xh: x already as packed fp16 rows (fp8 weights with 16-bit activations; one matrix)
template <int WF>
void dispatch_gemv(const Weight &w, const float *x, int M, float *y, int64_t ldy, cudaStream_t s, int sms, bool bx,
                   int64_t ldx = -1, GemvBatch bt = {}, int nbatch = 1, const uint16_t *xb = nullptr,
                   const __half *xh = nullptr) {
    constexpr int E = WT<WF>::kElem;
    if (ldx < 0) ldx = w.cols;
    if constexpr (WF == 1) {
        if (xb) {
            if (M > 8) launch_tc<1, true, true, true>(w, xb, M, y, ldy, w.cols, bt, 1, s);
            else launch_tc<1, false, true, true>(w, xb, M, y, ldy, w.cols, bt, 1, s);
            return;
        }
    }
    if constexpr (WF == 3) {
        if (xh) {
            if (M > 8) launch_tc<3, true, true, false, true>(w, xh, M, y, ldy, w.cols, bt, 1, s);
            else launch_tc<3, false, true, false, true>(w, xh, M, y, ldy, w.cols, bt, 1, s);
            return;
        }
    }
    if constexpr (WF != 0) {
        if (gemv_tc_ok(w, x, ldx) && (bt.w * E) % 16 == 0 && bt.x % 4 == 0) {
            if (bx) {
                if (M > 8) launch_tc<WF, true, true>(w, x, M, y, ldy, ldx, bt, nbatch, s);
                else launch_tc<WF, false, true>(w, x, M, y, ldy, ldx, bt, nbatch, s);
            } else {
                if (M > 8) launch_tc<WF, true, false>(w, x, M, y, ldy, ldx, bt, nbatch, s);
                else launch_tc<WF, false, false>(w, x, M, y, ldy, ldx, bt, nbatch, s);
            }
            return;
        }
    }
    const bool vec = gemv_vec_ok(w, x, ldx) && (bt.w * E) % 16 == 0 && bt.x % 4 == 0;
    if (!vec) launch_gemv<WF, false, 1>(w, x, M, y, ldy, ldx, bt, nbatch, s);
    else if (gemv_rows_per_warp(w.rows * nbatch, sms) == 4) launch_gemv<WF, true, 4>(w, x, M, y, ldy, ldx, bt, nbatch, s);
    else launch_gemv<WF, true, 2>(w, x, M, y, ldy, ldx, bt, nbatch, s);
}

// ------------------------------------------------------------------ cuBLAS path helpers

template <int WF>
__global__ void to_bf16_kernel(const void *__restrict__ src, int64_t n, uint16_t *__restrict__ dst) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        dst[i] = f2bf(WT<WF>::at(src, i));
}

// An 8-bit [N][K] matrix as bf16 with each row's scale folded in (the cached prefill copy)
template <int WF>
__global__ void scaled_to_bf16_kernel(const void *__restrict__ src, const float *__restrict__ scale, int64_t N,
                                      int64_t K, uint16_t *__restrict__ dst) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < N * K;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        dst[i] = f2bf(WT<WF>::at(src, i) * __ldg(scale + i / K));
}

__global__ void scale_cols_kernel(float *y, int64_t M, int64_t N, int64_t ldy, const float *__restrict__ scale) {
    const int64_t total = M * N;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t m = i / N, n = i % N;
        y[m * ldy + n] *= scale[n];
    }
}

}  // namespace
}  // namespace fullm
