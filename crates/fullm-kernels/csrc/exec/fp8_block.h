// SPDX-License-Identifier: Apache-2.0
// fp8_block.h - e4m3 weights with one fp32 scale per 128 x 128 block (WFmt::FP8B), host side:
// the encoder and the block quantiser the executor applies at load (and the selftest, so both
// see the same bytes). The scales are stored k-block major, [ceil(cols/128)][ceil(rows/128)]:
// CUTLASS's MN-major scale layout for the B operand of the sm90 blockwise GEMM, read directly.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace fullm {

constexpr int64_t kFp8Block = 128;
inline int64_t fp8b_blocks(int64_t n) { return (n + kFp8Block - 1) / kFp8Block; }
// scales of a [rows][cols] FP8B matrix
inline int64_t fp8b_scales(int64_t rows, int64_t cols) { return fp8b_blocks(rows) * fp8b_blocks(cols); }
// the scale of element (r, c)
inline int64_t fp8b_scale_index(int64_t rows, int64_t r, int64_t c) {
    return (c / kFp8Block) * fp8b_blocks(rows) + r / kFp8Block;
}

// float -> e4m3 ("fn": no infinities, NaN 0x7F / 0xFF, largest 448), round to nearest even,
// saturating to +-448 (as the GPU's satfinite conversion).
inline uint8_t f32_to_e4m3(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint8_t sign = static_cast<uint8_t>((u >> 24) & 0x80u);
    const float a = std::fabs(f);
    if (a != a) return sign | 0x7Fu;
    if (a >= 448.0f) return sign | 0x7Eu;
    if (a < 0.015625f) {  // below the smallest normal 2^-6: subnormals step 2^-9 (8 of them -> 2^-6)
        const int m = static_cast<int>(std::nearbyint(a * 512.0f));
        return sign | static_cast<uint8_t>(m);  // m = 8 is exponent 1, mantissa 0: 2^-6
    }
    int e = 0;
    const float fr = std::frexp(a, &e);  // a = fr * 2^e, fr in [0.5, 1): exponent field e + 6
    int q = static_cast<int>(std::nearbyint(fr * 16.0f));  // 8 .. 16: the implicit 1 and 3 bits
    int E = e + 6;
    if (q == 16) q = 8, ++E;
    if (E > 15 || (E == 15 && q - 8 == 7)) return sign | 0x7Eu;  // 464 would be the NaN pattern
    return sign | static_cast<uint8_t>((E << 3) | (q - 8));
}

inline float bf16_bits_f32(uint16_t h) {
    const uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// w: bf16 [rows][cols]. Per 128 x 128 block: s = amax / 448 (1 for an all-zero block),
// q = e4m3(w / s), scale[fp8b_scale_index(rows, r, c)] = s. Row blocks on parallel threads; the
// result does not depend on the thread count.
inline void quantize_fp8_blocks(const uint16_t *w, int64_t rows, int64_t cols, uint8_t *q, float *scale) {
    const int64_t nrb = fp8b_blocks(rows), ncb = fp8b_blocks(cols);
    auto row_block = [&](int64_t rb) {
        const int64_t r0 = rb * kFp8Block, r1 = std::min(rows, r0 + kFp8Block);
        for (int64_t cb = 0; cb < ncb; ++cb) {
            const int64_t c0 = cb * kFp8Block, c1 = std::min(cols, c0 + kFp8Block);
            float amax = 0.0f;
            for (int64_t r = r0; r < r1; ++r)
                for (int64_t c = c0; c < c1; ++c) amax = std::max(amax, std::fabs(bf16_bits_f32(w[r * cols + c])));
            const float s = amax > 0.0f ? amax / 448.0f : 1.0f;
            for (int64_t r = r0; r < r1; ++r)
                for (int64_t c = c0; c < c1; ++c) q[r * cols + c] = f32_to_e4m3(bf16_bits_f32(w[r * cols + c]) / s);
            scale[cb * nrb + rb] = s;
        }
    };
    const int64_t T = std::min<int64_t>(nrb, std::max(1u, std::thread::hardware_concurrency()));
    if (T <= 1) {
        for (int64_t rb = 0; rb < nrb; ++rb) row_block(rb);
        return;
    }
    std::vector<std::thread> th;
    for (int64_t t = 0; t < T; ++t)
        th.emplace_back([&, t] {
            for (int64_t rb = t; rb < nrb; rb += T) row_block(rb);
        });
    for (auto &x : th) x.join();
}

}  // namespace fullm
