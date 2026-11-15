// SPDX-License-Identifier: Apache-2.0
// kv_fp8.h - the fp8 KV cache (DType::FP8, `--kv-dtype fp8`), host side: its page layout, the
// quantisation rule (the CUDA append kernel applies the same one) and the decoder the tests use.
//
// Each key and each value vector of a (token, kv head) is stored as e4m3 values times a power of
// two, 2^(e - 127), e one byte: the largest exponent k with max|x| * 2^-k <= 448 (e4m3's largest
// finite) is chosen from the bits of max|x| alone, so host and device agree exactly, and the
// values are e4m3(x * 2^-k), round to nearest even, saturating. A power of two keeps every stored
// value exact in bf16 and fp16 (the prefill path dequantises to bf16 for FlashAttention-3, the
// decode kernel converts to fp16 and scales its fp32 sums): both see the same numbers.
//
// A page (page % 16 == 0 positions) of one layer's K (or V) pool: [page][Hkv][D] e4m3 bytes, then
// [Hkv][page] exponent bytes, so a 16-position decode tile's exponents for one head are one
// 16-byte run.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include "fp8_block.h"

#if defined(__CUDACC__)
#define KV8_HD __host__ __device__
#else
#define KV8_HD
#endif

namespace fullm {

KV8_HD inline int64_t kv8_page_bytes(int64_t page, int64_t Hkv, int64_t D) { return page * Hkv * (D + 1); }
// byte offsets of (physical page pg, position p % page = r, kv head h) in a pool: its values, its
// exponent (the CUDA kernels use these too)
KV8_HD inline int64_t kv8_row(int64_t pg, int64_t r, int64_t h, int64_t page, int64_t Hkv, int64_t D) {
    return pg * kv8_page_bytes(page, Hkv, D) + (r * Hkv + h) * D;
}
KV8_HD inline int64_t kv8_exp(int64_t pg, int64_t r, int64_t h, int64_t page, int64_t Hkv, int64_t D) {
    return pg * kv8_page_bytes(page, Hkv, D) + page * Hkv * D + h * page + r;
}

// The stored exponent byte for a vector whose largest magnitude is amax (finite, >= 0): k =
// E - 8, plus one when amax's mantissa exceeds 1.75 (448 = 1.75 * 2^8), clamped to [-127, 128].
inline uint8_t kv8_exponent(float amax) {
    uint32_t u;
    std::memcpy(&u, &amax, 4);
    const int E = static_cast<int>((u >> 23) & 0xFFu);
    if (E == 0) return 0;  // zero or subnormal: the smallest scale
    int k = E - 127 - 8 + ((u & 0x7FFFFFu) > 0x600000u ? 1 : 0);
    k = k < -127 ? -127 : k > 128 ? 128 : k;
    return static_cast<uint8_t>(k + 127);
}
inline float kv8_scale(uint8_t e) { return std::ldexp(1.0f, static_cast<int>(e) - 127); }

// One vector of n values: its exponent byte, and the e4m3 bytes into q.
inline uint8_t kv8_quantize(const float *x, int64_t n, uint8_t *q) {
    float amax = 0.0f;
    for (int64_t i = 0; i < n; ++i) amax = std::fmax(amax, std::fabs(x[i]));
    const uint8_t e = kv8_exponent(amax);
    const float inv = std::ldexp(1.0f, 127 - static_cast<int>(e));
    for (int64_t i = 0; i < n; ++i) q[i] = f32_to_e4m3(x[i] * inv);
    return e;
}

inline float e4m3_to_f32(uint8_t b) {
    const int s = b >> 7, E = (b >> 3) & 15, m = b & 7;
    const float v = E == 0 ? std::ldexp(static_cast<float>(m), -9) : std::ldexp(static_cast<float>(8 + m), E - 10);
    return s ? -v : v;
}

}  // namespace fullm
