// Part of cuda_backend.cu's translation unit (after mla.cuh, before attention.cuh): the int4 KV cache
// (DType::KV4, kv_int4.h's layout), head_dim 128, pages of 16.
//   kv4_append_row    in qk_rope_append (KT Kv4Rows): a row's key and value (rotated) into the step's
//                     bf16 rows, its slot's side_last, and the ring when its run spans <= 2 pages
//   kv4_page_kernel   quantises the pages a step fills (from its rows and the ring)
//   kv4_ring_kernel   the rows of runs spanning 3+ pages (prompt chunks) into the ring, after the
//                     pages are quantised (they read the ring entries those rows would overwrite)
//   gqa_attn_kv4_kernel  decode attention: a row per block, its own page from the ring
//   kv4_stage_kernel  pages dequantised to bf16 staging pages (prompt chunks: FlashAttention-3)
// The decode kernel is strata's tensor-core decode kernel's frame (gqa_attn_tc_kernel) with
// futransformer's int4 decode lessons: the unpack in the consumer (Marlin's lop3 under f16 1024,
// formats.cuh), keys in MMA-fragment order, the key's scale and offset in the unpack's fma, the
// value's scale on P (relative to the largest seen, so P stays in [0, 1]), the value rotation undone
// once per output row, and a grid of one wave.
#pragma once

#include "../../kv_int4.h"

namespace fullm {
namespace {

constexpr int kKv4VPitch = 80;  // V's 64-byte rows in shared memory: the lanes' 16-byte reads conflict-free
// a decode stage: the K block, the V block's nibble rows (padded), its c pairs
constexpr int kKv4SV = kKv4Block, kKv4SC = kKv4Block + 16 * kKv4VPitch, kKv4Stage = kKv4SC + 256;
constexpr int kKv4Warps = 4, kKv4Stages = 3;
constexpr size_t kv4_smem() {
    const size_t ring = size_t(kKv4Warps) * kKv4Stages * kKv4Stage;
    const size_t merge = sizeof(float) * kKv4Warps * 16 * (128 + 2);
    return ring > merge ? ring : merge;
}

__device__ __forceinline__ uint32_t h2sub(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("sub.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
__device__ __forceinline__ uint32_t h2fma(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
constexpr uint32_t kH1024 = 0x64006400u;  // f16 1024: 1024 + n has n in its mantissa
// a word's nibbles at bits (0, 16) and (4, 20) as two f16x2 (formats.cuh's int4x8_f16x2 unpack, its
// offsets as arguments)
__device__ __forceinline__ void nib2(uint32_t x, uint32_t lo_off, uint32_t hi_fma_c, uint32_t &b0, uint32_t &b1) {
    b0 = h2sub(lop3_and_or(x, 0x000F000Fu, kH1024), lo_off);                // 1024 + n  - lo_off
    b1 = h2fma(lop3_and_or(x, 0x00F000F0u, kH1024), 0x2C002C00u, hi_fma_c);  // (1024 + 16n) / 16 + c
}
__device__ __forceinline__ float h16f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ uint16_t f16h(float f) { return __half_as_ushort(__float2half_rn(f)); }

// x H for a warp's 128 values, d = lane + 32c (bits 0-4 the lane, 5-6 c): the normalised Hadamard
__device__ __forceinline__ void kv4_wht(float (&a)[4], int lane) {
#pragma unroll
    for (int s2 = 1; s2 < 32; s2 <<= 1) {
        const bool up = lane & s2;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float p = __shfl_xor_sync(0xffffffffu, a[c], s2);
            a[c] = up ? p - a[c] : a[c] + p;
        }
    }
    const float x0 = a[0] + a[1], x1 = a[0] - a[1], x2 = a[2] + a[3], x3 = a[2] - a[3];
    constexpr float kH = 0.08838834764831845f;  // 1 / sqrt(128)
    a[0] = (x0 + x2) * kH, a[1] = (x1 + x3) * kH, a[2] = (x0 - x2) * kH, a[3] = (x1 - x3) * kH;
}

// qk_rope_append's int4 mode: KT = Kv4Rows, kc / vc the step's bf16 rows, k4 the pools.
struct Kv4Rows {};
struct Kv4Ring {
    uint8_t *kpool, *vpool;
    int64_t ring_off;
};
// Row t's key x (normed, RoPE'd) and value v of kv head h, d = lane + 32c: the value rotated, both to
// the step's rows; the last row of the slot's run sets side_last; and the row goes into the ring when
// its run spans at most two pages (decoding rows, drafts, small prompts: their writes and the page
// kernel's reads of the ring touch different positions of the same pages; longer runs wait for
// kv4_ring_kernel). Runs are bounded scans: a run 32 rows on is far enough.
__device__ __forceinline__ void kv4_append_row(int64_t t, int h, int lane, const float (&x)[4], float (&v)[4], int Hk,
                                               const int32_t *__restrict__ tok_slot, const int32_t *__restrict__ tok_pos,
                                               __nv_bfloat16 *kr, __nv_bfloat16 *vr, const Kv4Ring &k4) {
    kv4_wht(v, lane);
    const int64_t i0 = (t * Hk + h) * 128;
#pragma unroll
    for (int c = 0; c < 4; ++c) kr[i0 + lane + 32 * c] = __float2bfloat16_rn(x[c]), vr[i0 + lane + 32 * c] = __float2bfloat16_rn(v[c]);
    const int64_t N = gridDim.x;
    const int slot = tok_slot[t];
    const int p = tok_pos[t];
    const bool next = t + 1 < N && tok_slot[t + 1] == slot && tok_pos[t + 1] == p + 1;
    if (h == 0 && lane == 0 && !next) reinterpret_cast<int32_t *>(k4.kpool + k4.ring_off)[slot] = p / kKv4Page;
    int pe = p, ps = p;
    if (next) {
        if (t + 32 < N && tok_slot[t + 32] == slot && tok_pos[t + 32] == p + 32) return;  // spans 3+ pages
        for (int64_t r = t + 1; r < N && tok_slot[r] == slot && tok_pos[r] == pe + 1; ++r) ++pe;
    }
    for (int64_t r = t - 1; r >= 0 && ps > p - 33 && tok_slot[r] == slot && tok_pos[r] == ps - 1; --r) --ps;
    if (pe / kKv4Page - ps / kKv4Page > 1) return;
    const int lp = p / kKv4Page, rr = p % kKv4Page;
    const int Hkv = Hk;
    uint16_t *ks = reinterpret_cast<uint16_t *>(k4.kpool + k4.ring_off + kv4_side(slot, lp & 1, h, Hkv));
    uint16_t *vsd = reinterpret_cast<uint16_t *>(k4.vpool + k4.ring_off + kv4_side(slot, lp & 1, h, Hkv));
#pragma unroll
    for (int c = 0; c < 4; ++c) ks[kv4_kside(rr, lane + 32 * c)] = f16h(x[c]), vsd[kv4_vside(rr, lane + 32 * c)] = f16h(v[c]);
}

// The pages this step fills: block (row t, 4 kv heads), a warp per head; the row whose position
// ends a page (pos % 16 == 15) quantises it, its earlier positions from this step's rows (the rows of
// a sequence's chunk are consecutive) or from the ring. kr / vr: the step's keys (normed, RoPE'd) and
// values (rotated), bf16 [N][Hkv][128].
__global__ void __launch_bounds__(128)
    kv4_page_kernel(uint8_t *kpool, uint8_t *vpool, int64_t ring_off, const __nv_bfloat16 *__restrict__ kr,
                    const __nv_bfloat16 *__restrict__ vr, const int32_t *__restrict__ tok_slot,
                    const int32_t *__restrict__ tok_pos, const int32_t *__restrict__ pt, int64_t max_pages, int N, int Hkv) {
    __shared__ uint8_t nk[4][16][128], nv[4][16][128];
    __shared__ uint16_t pa[4][128], pc[4][128], vs[4][16];
    pdl_wait();
    pdl_trigger();
    const int t = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, h = blockIdx.y * 4 + warp;
    const int pos = tok_pos[t], slot = tok_slot[t];
    if (h >= Hkv || pos % kKv4Page != kKv4Page - 1) return;  // warp-uniform
    const int lp = pos / kKv4Page;
    const int64_t pg = pt[static_cast<int64_t>(slot) * max_pages + lp];
    const uint16_t *ks = reinterpret_cast<const uint16_t *>(kpool + ring_off + kv4_side(slot, lp & 1, h, Hkv));
    const uint16_t *vsd = reinterpret_cast<const uint16_t *>(vpool + ring_off + kv4_side(slot, lp & 1, h, Hkv));
    // keys: lane's channels lane + 32c over the page's 16 positions
    float kx[16][4];
#pragma unroll
    for (int q = 0; q < 16; ++q) {
        const int r = t - (15 - q);
        const bool row = r >= 0 && tok_slot[r] == slot && tok_pos[r] == pos - 15 + q;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int ch = lane + 32 * c;
            kx[q][c] = row ? __bfloat162float(kr[(static_cast<int64_t>(r) * Hkv + h) * 128 + ch]) : h16f(ks[kv4_kside(q, ch)]);
        }
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int ch = lane + 32 * c;
        float mn = kx[0][c], mx = kx[0][c];
#pragma unroll
        for (int q = 1; q < 16; ++q) mn = fminf(mn, kx[q][c]), mx = fmaxf(mx, kx[q][c]);
        const float s = (mx - mn) / 15.0f;
        uint16_t a16 = f16h(s / 16.0f), c16 = f16h(mn - 64.0f * s);
        float af = h16f(a16);
        if (!(af > 0.0f)) a16 = 0, c16 = f16h(mn), af = 0.0f;  // a constant channel (or a scale below f16's)
        const float cf = h16f(c16), inv = af > 0.0f ? 1.0f / (16.0f * af) : 0.0f;
        pa[warp][ch] = a16, pc[warp][ch] = c16;
#pragma unroll
        for (int q = 0; q < 16; ++q) {
            const float n = af > 0.0f ? rintf((kx[q][c] - cf) * inv - 64.0f) : 0.0f;
            nk[warp][q][ch] = static_cast<uint8_t>(fminf(fmaxf(n, 0.0f), 15.0f));
        }
    }
    // values: rotated, a scale per position
#pragma unroll 1
    for (int q = 0; q < 16; ++q) {
        const int r = t - (15 - q);
        const bool row = r >= 0 && tok_slot[r] == slot && tok_pos[r] == pos - 15 + q;
        float v[4];
        if (row) {
#pragma unroll
            for (int c = 0; c < 4; ++c) v[c] = __bfloat162float(vr[(static_cast<int64_t>(r) * Hkv + h) * 128 + lane + 32 * c]);
        } else {
#pragma unroll
            for (int c = 0; c < 4; ++c) v[c] = h16f(vsd[kv4_vside(q, lane + 32 * c)]);
        }
        float am = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
        am = warp_max(am);
        const uint16_t s16 = f16h(am / 7.0f);
        const float sf = h16f(s16), inv = sf > 0.0f ? 1.0f / sf : 0.0f;
        if (lane == 0) vs[warp][q] = s16;
#pragma unroll
        for (int c = 0; c < 4; ++c) nv[warp][q][lane + 32 * c] = static_cast<uint8_t>(fminf(fmaxf(rintf(v[c] * inv), -8.0f), 7.0f) + 8.0f);
    }
    __syncwarp();
    uint8_t *kb = kpool + (pg * Hkv + h) * kKv4Block, *vb = vpool + (pg * Hkv + h) * kKv4Block;
#pragma unroll
    for (int j = 0; j < 8; ++j) {  // key words: (position, run tq, word w)
        const int i = lane + 32 * j, r = i / 16, tq = (i / 4) % 4, w = i % 4;
        uint32_t word = 0;
#pragma unroll
        for (int sub = 0; sub < 2; ++sub)
#pragma unroll
            for (int half = 0; half < 2; ++half)
#pragma unroll
                for (int e = 0; e < 2; ++e)
                    word |= uint32_t(nk[warp][r][(2 * w + sub) * 16 + 8 * half + 2 * tq + e]) << (8 * sub + 4 * half + 16 * e);
        *reinterpret_cast<uint32_t *>(kb + r * 64 + tq * 16 + w * 4) = word;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {  // value words: (row 2g + jj, word nt)
        const int i = lane + 32 * j, row = i / 16, nt = i % 16, g = row / 2, jj = row % 2, d = nt * 8 + g;
        constexpr int off[8] = {0, 1, 8, 9, 2, 3, 10, 11}, bit[8] = {0, 16, 4, 20, 8, 24, 12, 28};
        uint32_t word = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) word |= uint32_t(nv[warp][4 * jj + off[k]][d]) << bit[k];
        *reinterpret_cast<uint32_t *>(vb + row * 64 + nt * 4) = word;
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int ch = lane + 32 * c, o = kv4_kparam(ch);
        *reinterpret_cast<uint16_t *>(kb + kKv4KA + o) = pa[warp][ch];
        *reinterpret_cast<uint16_t *>(vb + kKv4KC + o) = pc[warp][ch];
    }
    if (lane < 16) *reinterpret_cast<uint16_t *>(kb + kKv4VS + 2 * lane) = vs[warp][lane];
}

// The step's rows into the ring (those of runs spanning 3+ pages, which qk_rope_append left; the others
// again, the same values): block (row t, 4 kv heads), a warp per head; rows of pages before
// side_last - 1 are already in the pool (kv4_page_kernel ran first: it read the ring).
__global__ void __launch_bounds__(128)
    kv4_ring_kernel(uint8_t *kpool, uint8_t *vpool, int64_t ring_off, const __nv_bfloat16 *__restrict__ kr,
                    const __nv_bfloat16 *__restrict__ vr, const int32_t *__restrict__ tok_slot,
                    const int32_t *__restrict__ tok_pos, int Hkv) {
    pdl_wait();
    pdl_trigger();
    const int t = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, h = blockIdx.y * 4 + warp;
    if (h >= Hkv) return;
    const int pos = tok_pos[t], slot = tok_slot[t], lp = pos / kKv4Page, r = pos % kKv4Page;
    if (lp < reinterpret_cast<const int32_t *>(kpool + ring_off)[slot] - 1) return;
    uint16_t *ks = reinterpret_cast<uint16_t *>(kpool + ring_off + kv4_side(slot, lp & 1, h, Hkv));
    uint16_t *vsd = reinterpret_cast<uint16_t *>(vpool + ring_off + kv4_side(slot, lp & 1, h, Hkv));
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int ch = lane + 32 * c;
        const int64_t i = (static_cast<int64_t>(t) * Hkv + h) * 128 + ch;
        ks[kv4_kside(r, ch)] = f16h(__bfloat162float(kr[i])), vsd[kv4_vside(r, ch)] = f16h(__bfloat162float(vr[i]));
    }
}

// Staging page j (a prompt chunk's context, kv8_stage's mapping: src[j] = slot * max_pages + page)
// as bf16 [16][Hkv][128] K and V: block (j, kv head), a warp per 4 positions. The ring for page
// side_last, else the pool; values rotated back.
__global__ void __launch_bounds__(128)
    kv4_stage_kernel(__nv_bfloat16 *kd, __nv_bfloat16 *vd, const uint8_t *__restrict__ kpool, const uint8_t *__restrict__ vpool,
                     int64_t ring_off, const int32_t *__restrict__ src, const int32_t *__restrict__ ptab, int64_t max_pages, int Hkv) {
    pdl_wait();
    pdl_trigger();
    const int64_t j = blockIdx.x;
    const int h = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int s = src[j], slot = s / static_cast<int>(max_pages), lp = s % static_cast<int>(max_pages);
    const bool ring = lp == reinterpret_cast<const int32_t *>(kpool + ring_off)[slot];
    const int64_t pg = ptab[s];
    const uint8_t *kb = kpool + (pg * Hkv + h) * kKv4Block, *vb = vpool + (pg * Hkv + h) * kKv4Block;
    const uint16_t *ks = reinterpret_cast<const uint16_t *>(kpool + ring_off + kv4_side(slot, lp & 1, h, Hkv));
    const uint16_t *vsd = reinterpret_cast<const uint16_t *>(vpool + ring_off + kv4_side(slot, lp & 1, h, Hkv));
#pragma unroll 1
    for (int r = warp * 4; r < warp * 4 + 4; ++r) {
        float k[4], v[4];
        const float vsc = ring ? 1.0f : h16f(*reinterpret_cast<const uint16_t *>(kb + kKv4VS + 2 * r));
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int ch = lane + 32 * c;
            if (ring) {
                k[c] = h16f(ks[kv4_kside(r, ch)]), v[c] = h16f(vsd[kv4_vside(r, ch)]);
            } else {
                int bk, bv;
                const uint32_t wk = *reinterpret_cast<const uint32_t *>(kb + kv4_kword(r, ch, bk));
                const uint32_t wv = *reinterpret_cast<const uint32_t *>(vb + kv4_vword(r, ch, bv));
                const int o = kv4_kparam(ch);
                k[c] = (1024.0f + 16.0f * float((wk >> bk) & 15u)) * h16f(*reinterpret_cast<const uint16_t *>(kb + kKv4KA + o)) +
                       h16f(*reinterpret_cast<const uint16_t *>(vb + kKv4KC + o));
                v[c] = vsc * (float((wv >> bv) & 15u) - 8.0f);
            }
        }
        kv4_wht(v, lane);
        const int64_t row = ((j * kKv4Page + r) * Hkv + h) * 128;
#pragma unroll
        for (int c = 0; c < 4; ++c) kd[row + lane + 32 * c] = __float2bfloat16(k[c]), vd[row + lane + 32 * c] = __float2bfloat16(v[c]);
    }
}

// Decode attention over the int4 cache: rows row0 + blockIdx.x, one block per (row, kv head, split).
// With at most 8 query heads per kv head, two consecutive rows of a sequence in one page (draft
// checks, a short prompt's rows) share a block: MMA rows 0-7 the first row's heads, 8-15 the second's
// (those rows are empty otherwise), one pass over the keys for both; the second row's block exits.
// Pairs are taken greedily from the start of each run of consecutive positions.
__global__ void __launch_bounds__(kKv4Warps * 32, 3)
    gqa_attn_kv4_kernel(float *out, float *part, unsigned *cnt, const float *q, const uint8_t *__restrict__ kc,
                        const uint8_t *__restrict__ vc, int64_t ring_off, const int32_t *__restrict__ tok_slot,
                        const int32_t *__restrict__ tok_pos, const int32_t *__restrict__ pt, int64_t max_pages, int row0,
                        int Hq, int Hkv, float scale_log2, int max_splits, int zs, int min_len) {
    constexpr int D = 128, DR = 4, T = kKv4Page, ST = kKv4Stages;
    extern __shared__ __align__(16) uint8_t kv4_smem_[];
    __shared__ float fs[kKv4Warps][kAttnMaxSplits];
    __shared__ bool last;
    pdl_wait();
    pdl_trigger();
    const int64_t t = blockIdx.x + row0;
    const int kh = blockIdx.y, G = Hq / Hkv;
    int sp = blockIdx.z;
    const int64_t pos = tok_pos[t];
    const int slot = tok_slot[t];
    // the pairing: walk this row's run from its start (runs are short: drafts, small prompts)
    bool paired = false;
    if (G <= 8) {
        int64_t r = t;
        while (r > row0 && tok_slot[r - 1] == slot && tok_pos[r - 1] == tok_pos[r] - 1) --r;
        bool follower = false;
        for (; r <= t;) {
            const bool pair = r + 1 < row0 + static_cast<int64_t>(gridDim.x) && tok_slot[r + 1] == slot &&
                              tok_pos[r + 1] == tok_pos[r] + 1 && tok_pos[r] % T != T - 1;
            if (r == t) paired = pair;
            if (pair && r + 1 == t) follower = true;
            r += pair ? 2 : 1;
        }
        if (follower) return;
    }
    const int64_t t2 = paired ? t + 1 : t, pos2 = paired ? pos + 1 : pos;  // the second row (rows g + 8)
    int64_t S;
    int nsplit;
    attn_split(pos2 + 1, max_splits, S, nsplit, min_len);
    if (sp >= nsplit) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
    const int32_t *ptab = pt + static_cast<int64_t>(slot) * max_pages;
    const int h0 = kh * G;
    // the row's own page from the ring when it is there (kv_int4.h)
    const int own = static_cast<int>(pos / T);
    const int side_last = reinterpret_cast<const int32_t *>(kc + ring_off)[slot];
    const int ring_page = own >= side_last - 1 ? own : -1;
    const uint8_t *rk = kc + ring_off + kv4_side(slot, own & 1, kh, Hkv), *rv = vc + ring_off + kv4_side(slot, own & 1, kh, Hkv);

    // MMA row hr's (row, head): the first row's heads, then (paired) the second's
    auto row_of = [&](int hr, int64_t &tr, int &hd) {
        if (paired) tr = hr < 8 ? t : t2, hd = hr & 7;
        else tr = t, hd = hr;
        return hd < G;
    };
    // Q as f16 A fragments (natural k order)
    uint32_t qa[D / 16][4];
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk) {
        auto at = [&](int hr, int c) -> float2 {
            int64_t tr;
            int hd;
            if (!row_of(hr, tr, hd)) return make_float2(0.f, 0.f);
            return *reinterpret_cast<const float2 *>(q + (tr * Hq + h0 + hd) * D + c);
        };
        const int c0 = kk * 16 + 2 * tq;
        const float2 x0 = at(g, c0), x1 = at(g + 8, c0), x2 = at(g, c0 + 8), x3 = at(g + 8, c0 + 8);
        qa[kk][0] = f16x2(x0.x, x0.y), qa[kk][1] = f16x2(x1.x, x1.y);
        qa[kk][2] = f16x2(x2.x, x2.y), qa[kk][3] = f16x2(x3.x, x3.y);
    }
    for (; sp < nsplit; sp += static_cast<int>(gridDim.z)) {
    if (sp != static_cast<int>(blockIdx.z)) __syncthreads();
    // end: the keys of rows 8-15 (the second row when paired); end1 those of rows 0-7
    const int64_t start = static_cast<int64_t>(sp) * S, end = imin(start + S, pos2 + 1), end1 = imin(start + S, pos + 1);
    const int ntile = static_cast<int>((end - start + T - 1) / T);
    const int mine = warp < ntile ? (ntile - warp + kKv4Warps - 1) / kKv4Warps : 0;
    uint8_t *ring = kv4_smem_ + warp * ST * kKv4Stage;
    auto issue = [&](int i) {  // tile i of mine (one page) into stage i % ST; the ring's page is read in place
        const int64_t p0 = start + static_cast<int64_t>(warp + i * kKv4Warps) * T;
        if (p0 / T == ring_page) return;
        const int64_t pg = ptab[p0 / T];
        uint8_t *ks = ring + (i % ST) * kKv4Stage;
        const uint8_t *kg = kc + (pg * Hkv + kh) * kKv4Block, *vg = vc + (pg * Hkv + kh) * kKv4Block;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int c = lane + 32 * j;
            if (c < kKv4Block / 16) cp_async16(ks + 16 * c, kg + 16 * c);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int c = lane + 32 * j;  // 64 chunks: row c / 4, column c % 4
            cp_async16(ks + kKv4SV + (c >> 2) * kKv4VPitch + 16 * (c & 3), vg + 16 * c);
        }
        if (lane < 16) cp_async16(ks + kKv4SC + 16 * lane, vg + kKv4KC + 16 * lane);
    };
    float o[D / 8][4];
#pragma unroll
    for (int j = 0; j < D / 8; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    float vm = 0.0f;  // the largest value scale so far: P is carried as p * s_p / vm
#pragma unroll
    for (int st = 0; st < ST - 1; ++st) {
        if (st < mine) issue(st);
        cp_async_commit();
    }
    for (int i = 0; i < mine; ++i) {
        cp_async_wait<ST - 2>();
        __syncwarp();
        const int nx = i + ST - 1;
        if (nx < mine) issue(nx);
        cp_async_commit();
        const uint8_t *ks = ring + (i % ST) * kKv4Stage;
        const int64_t p0 = start + static_cast<int64_t>(warp + i * kKv4Warps) * T;
        const bool rt = p0 / T == ring_page;  // warp-uniform
        // S = Q k^T by n-tiles of positions 0-7 and 8-15
        float sc[2][4];
        if (!rt) {
            uint32_t ka[16], kc2[16];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint4 x = *reinterpret_cast<const uint4 *>(ks + kKv4KA + tq * 64 + 16 * j);
                const uint4 y = *reinterpret_cast<const uint4 *>(ks + kKv4SC + tq * 64 + 16 * j);
                ka[4 * j] = x.x, ka[4 * j + 1] = x.y, ka[4 * j + 2] = x.z, ka[4 * j + 3] = x.w;
                kc2[4 * j] = y.x, kc2[4 * j + 1] = y.y, kc2[4 * j + 2] = y.z, kc2[4 * j + 3] = y.w;
            }
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                sc[ni][0] = sc[ni][1] = sc[ni][2] = sc[ni][3] = 0.0f;
                const uint4 kw = *reinterpret_cast<const uint4 *>(ks + (ni * 8 + g) * 64 + tq * 16);
                const uint32_t ww[4] = {kw.x, kw.y, kw.z, kw.w};
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const uint32_t x = ww[w];
                    const int p = 4 * w;  // channel pairs of steps 2w (b0, b1) and 2w + 1 (b0, b1)
                    const uint32_t b00 = h2fma(lop3_and_or(x << 4, 0x00F000F0u, kH1024), ka[p], kc2[p]);
                    const uint32_t b01 = h2fma(lop3_and_or(x, 0x00F000F0u, kH1024), ka[p + 1], kc2[p + 1]);
                    const uint32_t b10 = h2fma(lop3_and_or(x >> 4, 0x00F000F0u, kH1024), ka[p + 2], kc2[p + 2]);
                    const uint32_t b11 = h2fma(lop3_and_or(x >> 8, 0x00F000F0u, kH1024), ka[p + 3], kc2[p + 3]);
                    mma_f16(sc[ni], qa[2 * w][0], qa[2 * w][1], qa[2 * w][2], qa[2 * w][3], b00, b01);
                    mma_f16(sc[ni], qa[2 * w + 1][0], qa[2 * w + 1][1], qa[2 * w + 1][2], qa[2 * w + 1][3], b10, b11);
                }
            }
        } else {  // the ring's exact keys, in the same register order
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                sc[ni][0] = sc[ni][1] = sc[ni][2] = sc[ni][3] = 0.0f;
                uint32_t kr[16];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint4 x = __ldcg(reinterpret_cast<const uint4 *>(rk + (ni * 8 + g) * 256 + tq * 64 + 16 * j));
                    kr[4 * j] = x.x, kr[4 * j + 1] = x.y, kr[4 * j + 2] = x.z, kr[4 * j + 3] = x.w;
                }
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    mma_f16(sc[ni], qa[2 * w][0], qa[2 * w][1], qa[2 * w][2], qa[2 * w][3], kr[4 * w], kr[4 * w + 1]);
                    mma_f16(sc[ni], qa[2 * w + 1][0], qa[2 * w + 1][1], qa[2 * w + 1][2], qa[2 * w + 1][3], kr[4 * w + 2], kr[4 * w + 3]);
                }
            }
        }
        // the lane's positions' value scales: (n-tile ni, e) is position 8 ni + 2tq + e (1 in the ring)
        float vsp[2][2];
        float vt = 0.0f;
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            if (rt) {
                vsp[ni][0] = vsp[ni][1] = 1.0f;
            } else {
                const float2 f = __half22float2(*reinterpret_cast<const __half2 *>(ks + kKv4VS + 2 * (8 * ni + 2 * tq)));
                vsp[ni][0] = f.x, vsp[ni][1] = f.y;
            }
        }
        float mx0 = m0, mx1 = m1;
#pragma unroll
        for (int ni = 0; ni < 2; ++ni)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const bool ok = p0 + ni * 8 + 2 * tq + e < end, ok1 = p0 + ni * 8 + 2 * tq + e < end1;
                sc[ni][e] = ok1 ? sc[ni][e] * scale_log2 : -INFINITY;
                sc[ni][2 + e] = ok ? sc[ni][2 + e] * scale_log2 : -INFINITY;
                if (!ok) vsp[ni][e] = 0.0f;
                vt = fmaxf(vt, vsp[ni][e]);
                mx0 = fmaxf(mx0, sc[ni][e]), mx1 = fmaxf(mx1, sc[ni][2 + e]);
            }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
            vt = fmaxf(vt, __shfl_xor_sync(0xffffffffu, vt, off));
        }
        const float b0m = mx0 == -INFINITY ? 0.0f : mx0, b1m = mx1 == -INFINITY ? 0.0f : mx1;
        const float c0 = exp2f(m0 - b0m), c1 = exp2f(m1 - b1m);
        m0 = mx0, m1 = mx1;
        float oc0 = c0, oc1 = c1;
        const float vn = fmaxf(vm, vt);
        if (vn > vm && vm > 0.0f) oc0 *= vm / vn, oc1 *= vm / vn;
        vm = vn;
        const float iv = vm > 0.0f ? 1.0f / vm : 0.0f;
        uint32_t pa[4];
        float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            const float p00 = exp2f(sc[ni][0] - b0m), p01 = exp2f(sc[ni][1] - b0m);
            const float p10 = exp2f(sc[ni][2] - b1m), p11 = exp2f(sc[ni][3] - b1m);
            s0 += p00 + p01, s1 += p10 + p11;
            const float v0 = vsp[ni][0] * iv, v1 = vsp[ni][1] * iv;
            pa[2 * ni] = f16x2_in_range(p00 * v0, p01 * v1);
            pa[2 * ni + 1] = f16x2_in_range(p10 * v0, p11 * v1);
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            s0 += __shfl_xor_sync(0xffffffffu, s0, off);
            s1 += __shfl_xor_sync(0xffffffffu, s1, off);
        }
        l0 = l0 * c0 + s0, l1 = l1 * c1 + s1;
        if (__any_sync(0xffffffffu, oc0 != 1.0f || oc1 != 1.0f)) {
#pragma unroll
            for (int j = 0; j < D / 8; ++j) o[j][0] *= oc0, o[j][1] *= oc0, o[j][2] *= oc1, o[j][3] *= oc1;
        }
        // O += P v: n-tile nt = dims 8 nt + g; pool: the lane's row 2g + (tq >> 1), odd tq at bits 8+;
        // ring: the lane's 32 registers (kv_int4.h)
        if (!rt) {
            const uint8_t *vrow = ks + kKv4SV + (2 * g + (tq >> 1)) * kKv4VPitch;
            uint32_t vw[16];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint4 a = *reinterpret_cast<const uint4 *>(vrow + 16 * j);
                vw[4 * j] = a.x, vw[4 * j + 1] = a.y, vw[4 * j + 2] = a.z, vw[4 * j + 3] = a.w;
            }
            const int sh = (tq & 1) * 8;
#pragma unroll
            for (int nt = 0; nt < D / 8; ++nt) {
                uint32_t b0, b1;
                nib2(vw[nt] >> sh, 0x64086408u, 0xD480D480u, b0, b1);  // n - 8: -1032; (1024 + 16n)/16 - 72
                mma_f16(o[nt], pa[0], pa[1], pa[2], pa[3], b0, b1);
            }
        } else {
            const uint8_t *vrr = rv + (g * 4 + tq) * 128;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint4 a = __ldcg(reinterpret_cast<const uint4 *>(vrr + 16 * j));
                mma_f16(o[2 * j], pa[0], pa[1], pa[2], pa[3], a.x, a.y);
                mma_f16(o[2 * j + 1], pa[0], pa[1], pa[2], pa[3], a.z, a.w);
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();
    float *mg = reinterpret_cast<float *>(kv4_smem_);
    constexpr int ML = D + 2;
#pragma unroll
    for (int j = 0; j < D / 8; ++j) {
        float *r0 = mg + (warp * 16 + g) * ML + j * 8 + 2 * tq, *r1 = r0 + 8 * ML;
        r0[0] = o[j][0] * vm, r0[1] = o[j][1] * vm, r1[0] = o[j][2] * vm, r1[1] = o[j][3] * vm;
    }
    if (tq == 0) {
        mg[(warp * 16 + g) * ML + D] = m0, mg[(warp * 16 + g) * ML + D + 1] = l0;
        mg[(warp * 16 + g + 8) * ML + D] = m1, mg[(warp * 16 + g + 8) * ML + D + 1] = l1;
    }
    __syncthreads();
    constexpr float kLn2 = 0.6931471805599453f;
    for (int h = warp; h < 16; h += kKv4Warps) {
        int64_t tr;
        int hd;
        if (!row_of(h, tr, hd)) continue;
        float M = -INFINITY;
#pragma unroll
        for (int w = 0; w < kKv4Warps; ++w) M = fmaxf(M, mg[(w * 16 + h) * ML + D]);
        float f[kKv4Warps], l = 0.0f;
#pragma unroll
        for (int w = 0; w < kKv4Warps; ++w) {
            const float mw = mg[(w * 16 + h) * ML + D];
            f[w] = mw == -INFINITY ? 0.0f : exp2f(mw - M);
            l = fmaf(mg[(w * 16 + h) * ML + D + 1], f[w], l);
        }
        float a[DR];
#pragma unroll
        for (int c = 0; c < DR; ++c) {
            const int d = lane + 32 * c;
            a[c] = 0.0f;
#pragma unroll
            for (int w = 0; w < kKv4Warps; ++w) a[c] = fmaf(mg[(w * 16 + h) * ML + d], f[w], a[c]);
        }
        kv4_wht(a, lane);  // undo the values' rotation: (o H) H = o
        float *orow = out + (tr * Hq + h0 + hd) * D;
        float *pp = part + ((tr * Hq + h0 + hd) * zs + sp) * static_cast<int64_t>(ML);
#pragma unroll
        for (int c = 0; c < DR; ++c) {
            const int d = lane + 32 * c;
            if (nsplit == 1) orow[d] = a[c] / l;
            else pp[d] = a[c];
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
    float *f = fs[warp];
    for (int h = warp; h < 16; h += kKv4Warps) {
        int64_t tr;
        int hd;
        if (!row_of(h, tr, hd)) continue;
        const float *base = part + (tr * Hq + h0 + hd) * zs * static_cast<int64_t>(ML);
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
        float *orow = out + (tr * Hq + h0 + hd) * D;
#pragma unroll
        for (int c = 0; c < DR; ++c) orow[lane + 32 * c] = a[c] * inv;
        __syncwarp();
    }
    return;
    }
}

}  // namespace
}  // namespace fullm
