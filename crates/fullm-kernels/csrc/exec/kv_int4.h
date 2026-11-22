// SPDX-License-Identifier: Apache-2.0
// kv_int4.h - the int4 KV cache (DType::KV4, `--kv-dtype int4`): its page layout and the host-side
// decoder the tests use (the CUDA kernels, cuda/kernels/attention_kv4.cuh, quantise and read it).
// head_dim 128, pages of 16 positions. DESIGN 5.4, "int4, from futransformer".
//
// Keys per channel over each page (KIVI): k = s_c (n + z_c), n in 0..15, stored as a = s / 16 and
// c = s (z - 64) in f16 so the kernels' unpack (a nibble under f16 1024 reads 1024 + 16n) is one fma:
// k = (1024 + 16n) a + c. Values rotated by the 128-point Hadamard H (v H), then per position:
// v H = s_p (n - 8), n in 0..15; attention's output comes out times H and is rotated back.
//
// Each layer's K pool and V pool: [n_pages][Hkv][kKv4Block] blocks per (page, kv head):
//   K block: [0, 1024) key nibbles, [1024, 1280) the a pairs, [1280, 1312) the 16 value scales (f16)
//   V block: [0, 1024) value nibbles, [1024, 1280) the c pairs, [1280, 1312) unused
// Key nibbles: position r's 64 bytes are four 16-byte runs, run tq the channels kk*16 + 8 half +
// 2 tq + e (kk = 0..7, half, e = 0, 1) of an MMA lane's B fragments; word w of a run holds k-steps
// 2w and 2w + 1 at bit offsets (8 sub + 4 half + 16 e), sub = kk - 2w. The a and c pairs follow the
// same order: run tq, pair 2 kk + half, element e. Value nibbles: [16 rows 2g + j][16 words nt],
// word nt the dim 8 nt + g at positions 4j + {0, 1, 8, 9, 2, 3, 10, 11} (bits 0, 16, 4, 20, 8, 24,
// 12, 28): the PV B fragments of lanes tq = 2j (bits 0-7, 16-23) and 2j + 1 (bits 8-15, 24-31).
//
// After the pages, each pool has a ring (kv4_ring_bytes): the K pool first an int32 per slot (up to
// kKv4MaxSlots), the page of that slot's last position written in the latest step ("side_last"),
// then for each slot two pages ([2][Hkv][kKv4Side]: entry = page % 2) of exact values in f16: the
// page being filled and the one before it (a step of drafts may cross into the next page and be
// rolled back). K entries: [16 positions][128] in the key run order; V entries rotated (v H),
// [8 g][4 tq][16 nt][2 regs][2]: position 2tq + e at reg 0, 2tq + 8 + e at reg 1. Pages leave the
// ring for the pool, quantised, in the step that fills them.
//
// Which copy a reader takes: a decoding row (and a step's short segments: drafts) reads its own page
// pos / 16 from the ring when that page >= side_last - 1, every other page from the pool; a prompt
// chunk's staging (dequantised to bf16 for FlashAttention-3) reads the ring for page side_last.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define KV4_HD __host__ __device__
#else
#define KV4_HD
#endif

namespace fullm {

constexpr int kKv4Page = 16, kKv4D = 128;
constexpr int kKv4Block = 1312;    // bytes per (page, kv head) in each of the two pools
constexpr int kKv4KA = 1024, kKv4VS = 1280, kKv4KC = 1024;  // offsets: K block's a pairs, value scales; V block's c pairs
constexpr int kKv4Side = 4096;     // bytes per (slot, ring entry, kv head): 16 positions x 128 f16
constexpr int kKv4MaxSlots = 1024; // side_last: one int32 per slot, kKv4MaxSlots * 4 bytes reserved

KV4_HD inline int64_t kv4_page_bytes(int64_t page, int64_t Hkv) { return page / kKv4Page * Hkv * kKv4Block; }
KV4_HD inline int64_t kv4_ring_bytes(int64_t slots, int64_t Hkv) { return int64_t{kKv4MaxSlots} * 4 + slots * 2 * Hkv * kKv4Side; }
// a pool's ring area, after its n_pages pages (side_last at its start in the K pool)
KV4_HD inline int64_t kv4_ring_offset(int64_t n_pages, int64_t Hkv) { return n_pages * Hkv * kKv4Block; }
KV4_HD inline int64_t kv4_side(int64_t slot, int64_t entry, int64_t h, int64_t Hkv) {
    return int64_t{kKv4MaxSlots} * 4 + ((slot * 2 + entry) * Hkv + h) * kKv4Side;
}

// key channel c's place: run tq, k-step kk, half, element e
KV4_HD inline void kv4_kch(int c, int &tq, int &kk, int &half, int &e) {
    kk = c / 16, half = (c / 8) & 1, tq = (c / 2) & 3, e = c & 1;
}
// byte offset of the 32-bit word holding key nibble (r, c) in a K block, and the nibble's bit
KV4_HD inline int kv4_kword(int r, int c, int &bit) {
    int tq, kk, half, e;
    kv4_kch(c, tq, kk, half, e);
    bit = 8 * (kk & 1) + 4 * half + 16 * e;
    return r * 64 + tq * 16 + (kk / 2) * 4;
}
// byte offset of channel c's f16 in the a (K block) or c (V block) pairs, from kKv4KA / kKv4KC
KV4_HD inline int kv4_kparam(int c) {
    int tq, kk, half, e;
    kv4_kch(c, tq, kk, half, e);
    return tq * 64 + (2 * kk + half) * 4 + 2 * e;
}
// byte offset of the word holding value nibble (r, d) in a V block, and its bit
KV4_HD inline int kv4_vword(int r, int d, int &bit) {
    const int g = d % 8, nt = d / 8, j = (r % 8) / 4, rr = r % 4, hi = r / 8;
    bit = 8 * (rr / 2) + 4 * hi + 16 * (rr % 2);
    return (2 * g + j) * 64 + nt * 4;
}
// f16 index of (r, c) in a K ring entry, of (r, d) in a V ring entry
KV4_HD inline int kv4_kside(int r, int c) {
    int tq, kk, half, e;
    kv4_kch(c, tq, kk, half, e);
    return r * 128 + tq * 32 + (2 * kk + half) * 2 + e;
}
KV4_HD inline int kv4_vside(int r, int d) {
    const int g = d % 8, nt = d / 8, tq = (r % 8) / 2, b = r / 8, e = r % 2;
    return ((g * 4 + tq) * 16 + nt) * 4 + b * 2 + e;
}

// ---------------------------------------------------------------- host side (tests)
inline float kv4_f16(uint16_t h) {
    const uint32_t s = uint32_t(h >> 15) << 31, e = (h >> 10) & 31u, m = h & 1023u;
    uint32_t u;
    if (e == 0) {
        const float f = std::ldexp(float(m), -24);
        return s ? -f : f;
    }
    if (e == 31) u = s | 0x7F800000u | (m << 13);
    else u = s | ((e + 112u) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
inline uint16_t kv4_to_f16(float f) {  // round to nearest even, finite range
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u;
    const float a = std::fabs(f);
    if (!(a < 65520.0f)) return uint16_t(s | 0x7BFFu);
    if (a < 6.103515625e-05f) return uint16_t(s | uint32_t(std::nearbyint(a * 16777216.0f)));  // subnormal
    int ex;
    const float m = std::frexp(a, &ex);  // a = m 2^ex, m in [0.5, 1)
    uint32_t mant = uint32_t(std::nearbyint((m * 2.0f - 1.0f) * 1024.0f));
    int e = ex - 1 + 15;
    if (mant == 1024) mant = 0, ++e;
    if (e >= 31) return uint16_t(s | 0x7BFFu);
    return uint16_t(s | uint32_t(e) << 10 | mant);
}
inline uint16_t kv4_ld16(const uint8_t *p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return h;
}
// the key at (position r, channel c) of a page, from its K and V blocks
inline float kv4_key(const uint8_t *kb, const uint8_t *vb, int r, int c) {
    int bit;
    uint32_t w;
    std::memcpy(&w, kb + kv4_kword(r, c, bit), 4);
    const float n = float((w >> bit) & 15u);
    return (1024.0f + 16.0f * n) * kv4_f16(kv4_ld16(kb + kKv4KA + kv4_kparam(c))) + kv4_f16(kv4_ld16(vb + kKv4KC + kv4_kparam(c)));
}
// the rotated value (v H) at (position r, dim d)
inline float kv4_value_rot(const uint8_t *kb, const uint8_t *vb, int r, int d) {
    int bit;
    uint32_t w;
    std::memcpy(&w, vb + kv4_vword(r, d, bit), 4);
    return kv4_f16(kv4_ld16(kb + kKv4VS + 2 * r)) * (float(int((w >> bit) & 15u)) - 8.0f);
}
// x H in place, x of 128 (the normalised Sylvester Hadamard: its own inverse)
inline void kv4_hadamard(float *x) {
    for (int s = 1; s < kKv4D; s <<= 1)
        for (int i = 0; i < kKv4D; ++i)
            if (!(i & s)) {
                const float a = x[i], b = x[i | s];
                x[i] = a + b, x[i | s] = a - b;
            }
    for (int i = 0; i < kKv4D; ++i) x[i] *= 0.08838834764831845f;
}

}  // namespace fullm
