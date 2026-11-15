// SPDX-License-Identifier: Apache-2.0
// int4_gemm.cu - see int4_gemm.h.
//
// The GEMM kernel: persistent (one CTA per SM), warp-specialised. A tile is 256 weight rows of
// one expert x one pass (up to 128 of its tokens). Thread 0 (warpgroup 0) is the producer: per
// 128-column scale group (a stage) it loads the tile's weights, sixteen contiguous 1 KB runs
// (sixteen 16-row tiles x two 64-column chunks of int4_group.h's layout), with bulk copies, and
// the pass's 128 x 128 fp16 activation block with two TMA loads (128-byte swizzle), into a 4-stage
// mbarrier ring. Warpgroups 1 and 2 take 128 rows each, as two 64-row blocks (two accumulators).
// Per half stage (one 64-column chunk) a warp reads its two 16-row tiles' 512-byte chunk (one
// 16-byte load per lane each), unpacks each 32-bit word into the four f16x2 registers of one k16
// step's A fragment (the decode tile's int4x8_f16x2), multiplies them by the rows' group scale in
// fp16 (q * s rounded once; the decode tile folds the scale into its fp32 group sums instead),
// and issues eight wgmma.m64n128k16 with A from registers and the activation box from shared
// memory, fp32 accumulators. The fragments of one half are converted while the other half's
// wgmmas run (two register sets, wait_group 1). In small calls a pass of up to 64 (32) tokens runs
// n64 (n32) wgmmas on a 64- (32-) row activation box, in a launch of its own (see run()): a few
// hundred tokens route a dozen or so to each expert, and a 128-token tile spent most of its
// multiply-adds on padding (kbench, 256 tokens: 227 us against 313).
//
// Performance (H100, one layer of Qwen3-30B-A3B's experts): 8192 tokens 1473 us, 1024 tokens
// 326 us (the blockwise-fp8 CUTLASS GEMM: 1255 / 341). The 256-row tile halves the activation
// bytes per multiply-add (the activations are fp16, the weights int4). The GLU uses the fast
// sigmoid, and the producer hands each tile to the consumers through a shared-memory ring with one
// descriptor per activation box. The unpacking (LOP3, fp16 subtract and scale) is the largest
// share left, ~30%. Two alternatives measured no better and are not kept: a W4A8 form (int4 ->
// e4m3 in registers, fp8 wgmma) moves the same bytes per multiply-add; folding the group scale in
// fp32 as the decode tiles do doubles the epilogues and raises the L2 traffic.
//
// The k order. int4_group.h places k-slots 2t, 2t+1, 2t+8, 2t+9 of step s of chunk c at
// columns 64c + 16t + 4s .. +3 (so a decode lane's x for its four steps is one 32-byte run).
// wgmma reads B from shared memory in the canonical k order, so the activation rows are stored
// with every 64-column chunk permuted to match (chunk_pos), by the gather kernel for gate/up
// and by the gate/up epilogue for down.
//
// Epilogues, each warpgroup on its own. A warp's fragment rows g and g + 8 are matrix rows 16T + 2g
// and 16T + 2g + 1: gate and up of one GLU output (w1 is interleaved) or two adjacent outputs (w2).
// gate/up: the GLU in fp32, to fp16, staged in shared memory as [token][64] in chunk order (a
// warpgroup's 128 rows give one 64-column chunk of the down GEMM's activations), then 128-byte
// rows out. down: bf16 pairs staged as [token][128] per half tile, then 256-byte rows out. Rows
// past the pass's tokens are not written.

#include "int4_gemm.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cudaTypedefs.h>

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define SK_I4G 1  // this device pass has wgmma
#else
#define SK_I4G 0
#endif

namespace fullm::int4_gemm {
namespace {

constexpr int kTM = 256;  // weight rows per tile: two consumer warpgroups x two 64-row blocks
constexpr int kTN = 128;  // tokens per tile: one pass (cuda_backend.cu's kGeRows)
constexpr int kKG = 128;  // columns per stage: one scale group
constexpr int kStages = 4;
constexpr int kXBytes = kTN * kKG * 2;       // two 64-column TMA boxes of 128 rows x 128 bytes
constexpr int kWBytes = kTM * kKG / 2;       // sixteen 16-row tiles x two 512-byte chunks
constexpr int kOutBytes = 2 * kTN * 64 * 2;  // gate/up's staging: per warpgroup, 128 tokens x 64 fp16
// the producer warpgroup and two consumer warpgroups; setmaxnreg moves registers from the first to
// the others (40 / 232), which hold two accumulators and two fragment sets each
constexpr int kThreads = 384;
constexpr size_t kSmem = 1024 + kStages * (kXBytes + kWBytes) + kOutBytes + 2 * kStages * sizeof(uint64_t) +
                         kStages * 32;  // + the tile ring (TileInfo per stage)

// The GLU's sigmoid on the fast intrinsics (ex2.approx and an approximate reciprocal, ~4
// instructions against ~40 for expf and a correct division): ncu put the accurate form at ~19% of
// the gate/up kernel's stall samples, in an epilogue the tensor cores sit out. Its ~1e-6 relative
// error vanishes in the fp16 rounding of the result.
__device__ __forceinline__ float sigm(float x) { return __fdividef(1.0f, 1.0f + __expf(-x)); }

// Position within its 64-column chunk at which column c of an activation row is stored: the
// k-slot that meets column c in int4_group.h's fragments (see the top). Step s's slots 2t + e
// and 8 + 2t + e are columns 16t + 4s + e and 16t + 4s + 2 + e.
__host__ __device__ __forceinline__ int chunk_pos(int c) {
    const int o = c >> 1, t = o >> 3, s = (o >> 1) & 3, h = o & 1;
    return 2 * (8 * s + t + 4 * h) + (c & 1);
}

// cuda_backend.cu's int4x8_f16x2 (the decode tile's unpack; keep the two alike): one packed word
// to its 8 values as the four f16x2 registers of an m16n8k16 / wgmma k16 A fragment (Marlin's
// and FasterTransformer's magic-number dequantization, Apache-2.0; NOTICE).
__device__ __forceinline__ uint32_t lop3_and_or(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(d) : "r"(a), "r"(b), "r"(c));  // (a & b) | c
    return d;
}
__device__ __forceinline__ void int4x8_f16x2(uint32_t w, uint32_t (&h)[4]) {
    const uint32_t lo0 = lop3_and_or(w, 0x000F000Fu, 0x64006400u), hi0 = lop3_and_or(w, 0x00F000F0u, 0x64006400u);
    const uint32_t w8 = w >> 8;
    const uint32_t lo1 = lop3_and_or(w8, 0x000F000Fu, 0x64006400u), hi1 = lop3_and_or(w8, 0x00F000F0u, 0x64006400u);
    const __half2 k1032 = __halves2half2(__ushort_as_half(0x6408), __ushort_as_half(0x6408));
    const __half2 k16th = __halves2half2(__ushort_as_half(0x2C00), __ushort_as_half(0x2C00));
    const __half2 km72 = __halves2half2(__ushort_as_half(0xD480), __ushort_as_half(0xD480));
    auto u2h = [](uint32_t u) { return *reinterpret_cast<__half2 *>(&u); };
    auto h2u = [](__half2 v) { return *reinterpret_cast<uint32_t *>(&v); };
    h[0] = h2u(__hsub2(u2h(lo0), k1032));
    h[1] = h2u(__hfma2(u2h(hi0), k16th, km72));
    h[2] = h2u(__hsub2(u2h(lo1), k1032));
    h[3] = h2u(__hfma2(u2h(hi1), k16th, km72));
}
__device__ __forceinline__ uint32_t hmul2u(uint32_t a, __half2 b) {
    __half2 r = __hmul2(*reinterpret_cast<__half2 *>(&a), b);
    return *reinterpret_cast<uint32_t *>(&r);
}

// ---------------------------------------------------------------- sm90 primitives
__device__ __forceinline__ uint32_t smem_u32(const void *p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
// predicated rather than branched around: ptxas serialises wgmma near divergent paths (C7520)
__device__ __forceinline__ void mbar_arrive_if(uint32_t bar, bool pred) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, %1, 0;\n @p mbarrier.arrive.shared::cta.b64 _, [%0];\n}" ::"r"(bar),
        "r"(static_cast<int>(pred))
        : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
    asm volatile(
        "{\n .reg .pred p;\n WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT;\n}" ::"r"(bar),
        "r"(parity)
        : "memory");
}
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap *map, uint32_t bar, int c0, int c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(dst),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(bar), "r"(c0), "r"(c1)
        : "memory");
}
__device__ __forceinline__ void bulk_load(uint32_t dst, const void *src, uint32_t bytes, uint32_t bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
                 "l"(src), "r"(bytes), "r"(bar)
                 : "memory");
}
__device__ __forceinline__ void named_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); }

// Shared-memory descriptor of a K-major operand with 128-byte swizzle (8-row atoms of 1 KB, the
// tile 1 KB aligned): stride byte offset 1 KB between 8-row groups, leading byte offset unused;
// a k16 step within the 128-byte row adds 32 bytes to the start address.
__device__ __forceinline__ uint64_t make_desc(uint32_t addr) {
    uint64_t d = 0;
    d |= static_cast<uint64_t>((addr & 0x3FFFFu) >> 4);
    d |= static_cast<uint64_t>(1) << 16;
    d |= static_cast<uint64_t>(1024 >> 4) << 32;
    d |= static_cast<uint64_t>(1) << 62;
    return d;
}

#define SK_D8(i) \
    "+f"(d[i]), "+f"(d[i + 1]), "+f"(d[i + 2]), "+f"(d[i + 3]), "+f"(d[i + 4]), "+f"(d[i + 5]), "+f"(d[i + 6]), "+f"(d[i + 7])
#define SK_D64 SK_D8(0), SK_D8(8), SK_D8(16), SK_D8(24), SK_D8(32), SK_D8(40), SK_D8(48), SK_D8(56)
#define SK_D32 SK_D8(0), SK_D8(8), SK_D8(16), SK_D8(24)
#define SK_D16 SK_D8(0), SK_D8(8)

// D[64 x NV] (+)= A[64 x 16] B[NV x 16]^T: A from registers (fp16, the k16 fragment), B K-major in
// shared memory; NV / 2 fp32 accumulators per thread. NV: the tile's tokens (128, 64 or 32).
template <int NV>
__device__ __forceinline__ void wgmma_rs(float (&d)[NV / 2], const uint32_t (&a)[4], uint64_t desc, int scale_d);
template <>
__device__ __forceinline__ void wgmma_rs<128>(float (&d)[64], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,"
        "%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,"
        "%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %68, p, 1, 1, 0;\n}"
        : SK_D64
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}
template <>
__device__ __forceinline__ void wgmma_rs<64>(float (&d)[32], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,"
        "%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 0;\n}"
        : SK_D32
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}
template <>
__device__ __forceinline__ void wgmma_rs<32>(float (&d)[16], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, %21, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1, 0;\n}"
        : SK_D16
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}
// The compiler takes a wgmma's register operands as consumed at issue; the hardware reads the A
// fragments and writes the accumulators until wait_group retires it. These pin both at a point
// after the wait: the accumulators' reads stay below it, and the fragments' registers are not
// reused above it (CUTLASS's warpgroup_fence_operand).
__device__ __forceinline__ void fence_acc(float (&d)[64]) { asm volatile("" : SK_D64); }
__device__ __forceinline__ void fence_acc(float (&d)[32]) { asm volatile("" : SK_D32); }
__device__ __forceinline__ void fence_acc(float (&d)[16]) { asm volatile("" : SK_D16); }
#define SK_A4(b, i) "+r"(a[b][i][0]), "+r"(a[b][i][1]), "+r"(a[b][i][2]), "+r"(a[b][i][3])
__device__ __forceinline__ void fence_frag(uint32_t (&a)[2][4][4]) {
    asm volatile("" : SK_A4(0, 0), SK_A4(0, 1), SK_A4(0, 2), SK_A4(0, 3), SK_A4(1, 0), SK_A4(1, 1), SK_A4(1, 2), SK_A4(1, 3));
}
#undef SK_A4
#undef SK_D64
#undef SK_D32
#undef SK_D16
#undef SK_D8

struct Params {
    const int32_t *meta, *exps, *start, *pass_u, *pass_b;
    const Backend::ExpertW *table;
    int w1;    // gate/up (w1, the GLU epilogue) or down (w2)
    int N, K;  // weight rows and columns
    void *out;
    int silu;
    float b1, b2;
    int small;  // passes of <= 64 / 32 tokens on 64- / 32-token tiles (else every tile 128)
};

// A tile's tokens: its pass's m rounded up to 32, 64 or 128 (the producer and the consumers agree)
__device__ __forceinline__ int tile_tokens(int m, int small) { return !small || m > 64 ? 128 : m > 32 ? 64 : 32; }

// One consumer warpgroup's half stage (one 64-column chunk: k16 steps 4h .. 4h + 3) for its two
// 64-row blocks: the lanes' words of the chunk to A fragments, scaled.
__device__ __forceinline__ void frags(const uint8_t *wstage, const int (&lt)[2], int lane, int h, const __half2 (&sa)[2],
                                      const __half2 (&sb)[2], uint32_t (&a)[2][4][4]) {
#pragma unroll
    for (int b = 0; b < 2; ++b) {
        const uint4 u = reinterpret_cast<const uint4 *>(wstage + lt[b] * 1024 + h * 512)[lane];
        const uint32_t ww[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int st = 0; st < 4; ++st) {
            int4x8_f16x2(ww[st], a[b][st]);
            a[b][st][0] = hmul2u(a[b][st][0], sa[b]), a[b][st][1] = hmul2u(a[b][st][1], sb[b]);
            a[b][st][2] = hmul2u(a[b][st][2], sa[b]), a[b][st][3] = hmul2u(a[b][st][3], sb[b]);
        }
    }
}
// ... and its eight wgmmas (four steps x two row blocks) against the chunk's activation box
template <int NV>
__device__ __forceinline__ void mmas(float (&acc)[2][NV / 2], const uint32_t (&a)[2][4][4], uint32_t xbox, bool first) {
    const uint64_t d0 = make_desc(xbox);  // a k16 step is 32 bytes on: +2 in the address field
    wgmma_fence();
#pragma unroll
    for (int st = 0; st < 4; ++st) {
#pragma unroll
        for (int b = 0; b < 2; ++b) wgmma_rs<NV>(acc[b], a[b][st], d0 + 2 * st, !(first && st == 0));
    }
    wgmma_commit();
}

// Consumer warpgroup c's half of the epilogue staging (gate/up's GLU chunk, down's bf16 rows).
__device__ __forceinline__ uint8_t *cs_os(uint8_t *os, int c) { return os + c * (kOutBytes / 2); }

// One tile of NV tokens for consumer warpgroup c: the mainloop over its K stages, then the
// epilogue (rows past m are not written). it: the stages consumed so far (all tiles).
template <int NV>
__device__ __forceinline__ void consume(const Params &p, const uint8_t *xs, const uint8_t *ws, uint8_t *os, uint32_t full0,
                                        uint32_t empty0, int c, int tid, int &it, int r, int row0, int m, const float *wscale,
                                        int nkg) {
    const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int lt[2] = {c * 8 + warp, c * 8 + 4 + warp};  // the warp's 16-row tile in each 64-row block
    float acc[2][NV / 2];
    uint32_t a0[2][4][4], a1[2][4][4];  // the two half stages' fragments: one is converted while the other's wgmmas run
    const float *srow0 = wscale + r * kTM + lt[0] * 16 + 2 * g, *srow1 = wscale + r * kTM + lt[1] * 16 + 2 * g;
    int prev = -1;  // the stage whose second half may still run
    for (int kg = 0; kg < nkg; ++kg, ++it) {
        const int s = it % kStages;
        const uint32_t ph = (it / kStages) & 1;
        const int64_t so = static_cast<int64_t>(kg) * p.N;
        const float2 sc0 = __ldg(reinterpret_cast<const float2 *>(srow0 + so));
        const float2 sc1 = __ldg(reinterpret_cast<const float2 *>(srow1 + so));
        const __half2 sa[2] = {__float2half2_rn(sc0.x), __float2half2_rn(sc1.x)};
        const __half2 sb[2] = {__float2half2_rn(sc0.y), __float2half2_rn(sc1.y)};
        mbar_wait(full0 + 8 * s, ph);
        const uint8_t *wstage = ws + s * kWBytes;
        const uint32_t xa = smem_u32(xs + s * kXBytes);
        frags(wstage, lt, lane, 0, sa, sb, a0);  // a0 is free: the last wait retired its wgmmas
        mmas<NV>(acc, a0, xa, kg == 0);
        if (prev >= 0) {  // the previous stage's second half has retired: a1 and that stage are free
            asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
            fence_frag(a1);
            mbar_arrive_if(empty0 + 8 * prev, tid == 0);
        }
        frags(wstage, lt, lane, 1, sa, sb, a1);
        mmas<NV>(acc, a1, xa + kXBytes / 2, false);
        asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");  // this stage's first half
        fence_frag(a0);
        prev = s;
    }
    wgmma_wait0();
    fence_acc(acc[0]);
    fence_acc(acc[1]);
    fence_frag(a1);
    mbar_arrive_if(empty0 + 8 * prev, tid == 0);
    if (p.w1) {  // a warpgroup's 128 rows are 64 GLU outputs: one 64-column chunk of down's activations
        __half *o = reinterpret_cast<__half *>(cs_os(os, c));  // [token][64], chunk order
        named_sync(1 + c, 128);  // the staging is free: the warpgroup is past the last tile's copy-out
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const int pos = chunk_pos((b * 4 + warp) * 8 + g);
#pragma unroll
            for (int n = 0; n < NV / 8; ++n) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const float gv = acc[b][4 * n + e], uv = acc[b][4 * n + 2 + e];
                    const float v = p.silu ? gv * sigm(gv) * uv
                                           : p.b1 * tanhf(gv / p.b1) * sigm(gv) * (p.b2 * tanhf(uv / p.b2));
                    o[(8 * n + 2 * t + e) * 64 + pos] = __float2half_rn(v);
                }
            }
        }
        named_sync(1 + c, 128);
        const int ld = p.N / 2;  // inter
        __half *dst = static_cast<__half *>(p.out) + static_cast<int64_t>(row0) * ld + r * (kTM / 2) + c * 64;
        for (int i = tid; i < NV * 8; i += 128) {  // NV rows x 8 uint4
            const int row = i >> 3, part = i & 7;
            if (row < m)
                *reinterpret_cast<uint4 *>(dst + static_cast<int64_t>(row) * ld + part * 8) =
                    reinterpret_cast<const uint4 *>(o)[i];
        }
    } else {  // down: the warpgroup's 128 output columns staged as bf16 [64 tokens][128] (a half of
              // the tile at a time; 16-byte chunks XOR-swizzled by the token against bank conflicts),
              // then copied out a 16-byte store per lane: compile-time staging offsets in place of a
              // 64-bit address and a predicate per 4-byte global store (ncu: down's largest line)
        uint32_t *o = reinterpret_cast<uint32_t *>(cs_os(os, c));
        constexpr int TH = NV > 64 ? 64 : NV, H = NV / TH;  // tokens per half, halves
#pragma unroll
        for (int h = 0; h < H; ++h) {
            named_sync(1 + c, 128);  // the staging is free: the warpgroup is past the last copy-out
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                const int wd = (b * 4 + warp) * 8 + g;  // the lane's column pair within the 128
#pragma unroll
                for (int n = 0; n < TH / 8; ++n) {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int tl = 8 * n + 2 * t + e, nn = h * (TH / 8) + n;  // token in the half, sums
                        const __nv_bfloat162 v = __floats2bfloat162_rn(acc[b][4 * nn + e], acc[b][4 * nn + 2 + e]);
                        o[tl * 64 + ((((wd >> 2) ^ tl) & 15) << 2) + (wd & 3)] = *reinterpret_cast<const uint32_t *>(&v);
                    }
                }
            }
            named_sync(1 + c, 128);
            const int rows = min(TH, m - h * TH);
            __nv_bfloat16 *dst =
                static_cast<__nv_bfloat16 *>(p.out) + static_cast<int64_t>(row0 + h * TH) * p.N + r * kTM + c * 128;
            for (int i = tid; i < rows * 16; i += 128) {  // a row: 16 chunks of 16 bytes
                const int row = i >> 4, part = i & 15;
                *reinterpret_cast<uint4 *>(dst + static_cast<int64_t>(row) * p.N + part * 8) =
                    reinterpret_cast<const uint4 *>(o)[row * 16 + ((part ^ row) & 15)];
            }
        }
    }
}

// What the producer tells the consumers about the tile whose first stage sits in a ring slot
// (r < 0: no more tiles). Resolving it themselves cost each tile a chain of four or five dependent
// global loads (pass -> group -> expert -> weight -> scales: ncu's long-scoreboard stalls at the
// down kernel's tile starts, ~5% of it); the producer resolves it anyway, ahead of time.
struct TileInfo {
    int r, row0, m;
    const float *wscale;
};

// NV: the tiles' tokens. With p.small, the launch for NV takes only the passes whose tile_tokens
// is NV (one kernel per width: a kernel holding all three spilled and serialised its wgmmas);
// without, every pass on 128. xmap: the activations' tensor map with boxes of NV rows.
template <int NV>
__global__ void __launch_bounds__(kThreads, 1) gemm_kernel(const __grid_constant__ CUtensorMap xmap, const Params p) {
#if SK_I4G
    extern __shared__ uint8_t smem_raw[];
    uint8_t *smem = reinterpret_cast<uint8_t *>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t{1023});
    uint8_t *xs = smem;                    // [stage][box 2][128 rows][128 bytes], swizzled
    uint8_t *ws = xs + kStages * kXBytes;  // [stage][tile 16][chunk 2][lane 32][16 bytes]
    uint8_t *os = ws + kStages * kWBytes;  // gate/up's staging: [warpgroup 2][token 128][64]
    uint64_t *bars = reinterpret_cast<uint64_t *>(os + kOutBytes);
    volatile TileInfo *tinfo = reinterpret_cast<volatile TileInfo *>(bars + 2 * kStages);  // [stage]
    const uint32_t full0 = smem_u32(bars), empty0 = smem_u32(bars + kStages);
    if (threadIdx.x == 0) {
        for (int s = 0; s < kStages; ++s) {
            mbar_init(full0 + 8 * s, 1);   // the producer's expect_tx (or its end-of-tiles arrive)
            mbar_init(empty0 + 8 * s, 2);  // one arrive per consumer warpgroup
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    const int R = p.N / kTM, nkg = p.K / kKG, total = p.meta[1] * R;
    const int wg = threadIdx.x >> 7, tid = threadIdx.x & 127;
    // the tile's pass, its first row and tokens; false: another width's launch takes it
    auto mine = [&](int tile, int &r, int &u, int &row0, int &m) {
        const int q = tile / R;
        r = tile - q * R, u = p.pass_u[q], row0 = p.pass_b[q], m = min(kTN, p.start[u + 1] - row0);
        return tile_tokens(m, p.small) == NV;
    };
    if (wg == 0) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
        if (tid != 0) return;
        asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&xmap)) : "memory");
        const int64_t tb = static_cast<int64_t>(p.K) * 8;  // bytes of a 16-row tile
        int it = 0;
        for (int tile = blockIdx.x; tile < total; tile += gridDim.x) {
            int r, u, row0, m;
            if (!mine(tile, r, u, row0, m)) continue;
            const Weight &w = p.w1 ? p.table[p.exps[u]].w1 : p.table[p.exps[u]].w2;
            const uint8_t *wt = static_cast<const uint8_t *>(w.data) + static_cast<int64_t>(r) * (kTM / 16) * tb;
            for (int kg = 0; kg < nkg; ++kg, ++it) {
                const int s = it % kStages;
                const uint32_t ph = (it / kStages) & 1, full = full0 + 8 * s;
                mbar_wait(empty0 + 8 * s, ph ^ 1);
                if (kg == 0) {  // the tile's first stage carries its description (released by the arrive below)
                    tinfo[s].r = r, tinfo[s].row0 = row0, tinfo[s].m = m;
                    tinfo[s].wscale = w.scale;
                }
                mbar_expect_tx(full, NV * kKG * 2 + kWBytes);
                const uint32_t xd = smem_u32(xs + s * kXBytes), wd = smem_u32(ws + s * kWBytes);
                tma_load_2d(xd, &xmap, full, kg * kKG, row0);
                tma_load_2d(xd + kXBytes / 2, &xmap, full, kg * kKG + 64, row0);
                for (int i = 0; i < kTM / 16; ++i) bulk_load(wd + i * 1024, wt + i * tb + kg * 1024, 1024, full);
            }
        }
        // the end: the next slot carries r = -1, its phase completed by a plain arrive
        const int s = it % kStages;
        mbar_wait(empty0 + 8 * s, ((it / kStages) & 1) ^ 1);
        tinfo[s].r = -1;
        mbar_arrive(full0 + 8 * s);
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
    const int c = wg - 1;
    int it = 0;
    for (;;) {  // the producer's tiles in its order: wait for each one's first stage, read what it is
        const int s = it % kStages;
        mbar_wait(full0 + 8 * s, (it / kStages) & 1);
        const int r = tinfo[s].r;
        if (r < 0) break;
        consume<NV>(p, xs, ws, os, full0, empty0, c, tid, it, r, tinfo[s].row0, tinfo[s].m,
                    const_cast<const float *>(tinfo[s].wscale), nkg);
    }
#endif
}

// Two columns of x as an f16x2 register, rounded as cuda_backend.cu's f16x2 (clamped to +-65504,
// nearest even: the decode tiles' x): from fp16 (copied) or bf16 rows.
__device__ __forceinline__ uint32_t f16x2_of(float lo, float hi) {
    const __half2 v = __floats2half2_rn(fminf(fmaxf(lo, -65504.0f), 65504.0f), fminf(fmaxf(hi, -65504.0f), 65504.0f));
    return *reinterpret_cast<const uint32_t *>(&v);
}
__device__ __forceinline__ uint32_t pair_f16(const __half *p) { return *reinterpret_cast<const uint32_t *>(p); }
__device__ __forceinline__ uint32_t pair_f16(const uint16_t *p) {
    const uint32_t u = *reinterpret_cast<const uint32_t *>(p);
    return f16x2_of(__uint_as_float(u << 16), __uint_as_float(u & 0xFFFF0000u));
}

// A block per pair-list row (rows past the routed ones return): x's row of the pair's token as
// fp16, each 64-column chunk in chunk_pos order; a thread per 4-byte output unit.
template <typename XT>
__global__ void gather_kernel(__half *xs, const XT *__restrict__ x, const int32_t *meta, const int32_t *start,
                              const int32_t *plist, int k, int width) {
    const int i = blockIdx.x;
    if (i >= start[meta[0]]) return;
    const XT *src = x + static_cast<int64_t>(plist[i] / k) * width;
    uint32_t *dst = reinterpret_cast<uint32_t *>(xs + static_cast<int64_t>(i) * width);
    for (int v = threadIdx.x; v < width / 2; v += blockDim.x) {
        // output unit v = 32 c + 8 s + t + 4 h holds columns 2 (8 t + 2 s + h) .. + 1 of chunk c
        const int pp = v & 7, s = (v >> 3) & 3, t = pp & 3, h = pp >> 2;
        dst[v] = pair_f16(src + 2 * ((v & ~31) + 8 * t + 2 * s + h));
    }
}

__global__ void probe_kernel(int *out) {
    *out = SK_I4G;
}

// ---------------------------------------------------------------- host
void check(cudaError_t e, const char *what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("int4_gemm: ") + what + ": " + cudaGetErrorString(e));
}

PFN_cuTensorMapEncodeTiled_v12000 encode_fn() {
    static PFN_cuTensorMapEncodeTiled_v12000 fn = [] {
        void *f = nullptr;
        cudaDriverEntryPointQueryResult q;
#if CUDART_VERSION >= 12050
        if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &f, 12000, cudaEnableDefault, &q) != cudaSuccess)
            f = nullptr;
#else
        if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &f, cudaEnableDefault, &q) != cudaSuccess) f = nullptr;
#endif
        return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(f);
    }();
    return fn;
}

// fp16 [rows][cols] (cols contiguous), boxes of 64 columns x box_rows rows, 128-byte swizzle; rows
// past the tensor read as zeros. Cached per (pointer, shape, box): the buffers live across calls.
CUtensorMap act_map(const __half *ptr, int64_t rows, int64_t cols, int box_rows) {
    struct Entry {
        const void *ptr;
        int64_t rows, cols;
        int box_rows;
        CUtensorMap map;
    };
    static std::mutex mu;
    static std::vector<Entry> cache;
    std::lock_guard<std::mutex> lock(mu);
    for (const Entry &e : cache)
        if (e.ptr == ptr && e.rows == rows && e.cols == cols && e.box_rows == box_rows) return e.map;
    CUtensorMap m;
    const cuuint64_t dims[2] = {static_cast<cuuint64_t>(cols), static_cast<cuuint64_t>(rows)};
    const cuuint64_t strides[1] = {static_cast<cuuint64_t>(cols) * 2};
    const cuuint32_t box[2] = {64, static_cast<cuuint32_t>(box_rows)}, estride[2] = {1, 1};
    const CUresult r = encode_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, const_cast<__half *>(ptr), dims, strides, box,
                                   estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) throw std::runtime_error("int4_gemm: cuTensorMapEncodeTiled failed (" + std::to_string(r) + ")");
    if (cache.size() >= 64) cache.clear();
    cache.push_back(Entry{ptr, rows, cols, box_rows, m});
    return m;
}

int sm_count() {
    static const int n = [] {
        int dev = 0, v = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev);
        return v > 0 ? v : 132;
    }();
    return n;
}

void run(const __half *act, int64_t rows, const Lists &L, const Backend::ExpertW *table, bool w1, int64_t N, int64_t K,
         void *out, bool silu, float b1, float b2, cudaStream_t s) {
    static const bool attr = [] {
        for (const void *k : {(const void *)gemm_kernel<128>, (const void *)gemm_kernel<64>, (const void *)gemm_kernel<32>})
            check(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmem)), "smem attribute");
        return true;
    }();
    (void)attr;
    // The small tiles pay when nearly every pass is small: then one launch has almost all the work.
    // When passes of every size come (kbench at 512+ tokens), three launches each have a tail, and
    // a tile's unpacking does not shrink with its tokens: 1024 tokens took 502 us against 348 on
    // 128 always. So they run when the call averages at most FULLM_INT4_GEMM_SMALL (24) tokens an
    // expert slot (0: never); the routing never comes back to the host for it.
    static const int small_avg = [] {
        const char *e = std::getenv("FULLM_INT4_GEMM_SMALL");
        return e && *e ? std::atoi(e) : 24;
    }();
    const int small = small_avg > 0 && static_cast<int64_t>(L.pairs) <= static_cast<int64_t>(small_avg) * L.slots ? 1 : 0;
    Params p{L.meta, L.exps, L.start, L.pass_u, L.pass_b, table, w1 ? 1 : 0, static_cast<int>(N), static_cast<int>(K),
             out, silu ? 1 : 0, b1, b2, small};
    const int64_t tiles = static_cast<int64_t>(L.maxp) * (N / kTM);
    const unsigned grid = static_cast<unsigned>(std::min<int64_t>(tiles, sm_count()));
    if (grid == 0) return;
    // one launch per tile width (each skips the others' passes); the routing stays on the device
    gemm_kernel<128><<<grid, kThreads, kSmem, s>>>(act_map(act, rows, K, 128), p);
    if (small) {
        gemm_kernel<64><<<grid, kThreads, kSmem, s>>>(act_map(act, rows, K, 64), p);
        gemm_kernel<32><<<grid, kThreads, kSmem, s>>>(act_map(act, rows, K, 32), p);
    }
    check(cudaGetLastError(), w1 ? "gate/up" : "down");
}

}  // namespace

bool available() {
    static const bool ok = [] {
        int dev = 0, major = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess)
            return false;
        if (major != 9 || !encode_fn()) return false;
        int *d = nullptr, h = 0;
        if (cudaMalloc(&d, sizeof(int)) != cudaSuccess) return false;
        probe_kernel<<<1, 1>>>(d);
        const bool ran = cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess;
        cudaFree(d);
        return ran && h == 1;
    }();
    return ok;
}

bool shape_ok(int64_t width, int64_t inter) {
    return width % kKG == 0 && inter % kKG == 0 && (2 * inter) % kTM == 0 && width % kTM == 0;
}

void gather(__half *xs, const void *x, XType xt, const Lists &L, int k, int64_t width, int64_t pairs, cudaStream_t s) {
    if (pairs <= 0) return;
    const unsigned g = static_cast<unsigned>(pairs);
    const int w = static_cast<int>(width);
    if (xt == XType::F16) gather_kernel<<<g, 128, 0, s>>>(xs, static_cast<const __half *>(x), L.meta, L.start, L.plist, k, w);
    else gather_kernel<<<g, 128, 0, s>>>(xs, static_cast<const uint16_t *>(x), L.meta, L.start, L.plist, k, w);
    check(cudaGetLastError(), "gather");
}

void gate_up(__half *s16, const __half *xs, int64_t rows, const Lists &L, const Backend::ExpertW *table, int64_t width,
             int64_t inter, bool silu, float b1, float b2, cudaStream_t s) {
    run(xs, rows, L, table, true, 2 * inter, width, s16, silu, b1, b2, s);
}

void down(uint16_t *dn, const __half *s16, int64_t rows, const Lists &L, const Backend::ExpertW *table, int64_t width,
          int64_t inter, cudaStream_t s) {
    run(s16, rows, L, table, false, width, inter, dn, true, 0.0f, 0.0f, s);
}

}  // namespace fullm::int4_gemm
