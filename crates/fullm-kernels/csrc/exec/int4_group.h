// SPDX-License-Identifier: Apache-2.0
// int4_group.h - int4 weights with one fp32 scale per row per 128 columns (WFmt::INT4G), host
// side: the quantiser the executor applies at load (and the selftest), the packing the CUDA
// decode tile reads, and the decoder the CPU backend uses.
//
// Values: symmetric int4, q in [-8, 7], W[n][k] ~ q * scale[(k / 128) * rows + n]. Each group's
// scale is its absolute maximum / 7 (round to nearest). Choosing it instead as the best of a few
// clipped maxima (least squared weight error; FULLM_INT4_CLIP=1) lowers the weights' error but
// raises Qwen3-30B-A3B's perplexity (+2.5% over bf16 against +1.7%), so it is off by default.
//
// Packing (rows % 16 == 0 and cols % 64 == 0: the tensor-core layout). The decode tile takes the
// weights as the MMA's A operand (m16n8k16: 16 weight rows, the tokens as the 8 columns of B), so
// the matrix is stored as A fragments: 16-row tiles T, each a run of 64-column chunks c, a chunk
// 32 lanes x 16 bytes (lane = 4 g + t), a lane's 16 bytes four 32-bit words, word s the fragment
// of MMA step s. Fragment row g is matrix row 16 T + 2 g, row g + 8 is 16 T + 2 g + 1 (a lane's
// two rows are adjacent: gate and up of one GLU output, or two adjacent outputs). Step s's
// k-slots 2t, 2t+1, 2t+8, 2t+9 are columns K0 .. K0+3, K0 = 64 c + 16 t + 4 s, so a lane's x for
// its four steps is one 32-byte run of each token row. Within a word the nibbles are placed so the
// tile's unpacking (mask bits 0-3 / 16-19 and 4-7 / 20-23, the same after a shift by 8) gives
// the fragment's four registers in order: (row A col K0, K0+1), (row B K0, K0+1), (row A K0+2,
// K0+3), (row B K0+2, K0+3), A = 16 T + 2 g, B = A + 1; each nibble holds q + 8.
// Other shapes (the CPU's tiny test models), and every matrix of a backend that asks for it
// (Backend::weight_layout = kInt4RowMajor), are row-major, 8 columns per word, nibble 2j holding
// column j and nibble 2j + 1 column j + 4 (j < 4). The CPU backend asks for kInt4CpuSimd where
// cols % 64 == 0: row-major in 64-column chunks of 32 bytes, byte b holding column b of the chunk
// in its low nibble and column 32 + b in its high nibble, so a 32-byte load, a mask and a shift
// give the chunk's 64 values in column order (no transpose; cpu_kernels.cpp dec_i4s_*). The
// layout travels with the matrix (Weight::layout).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

namespace fullm {

constexpr int64_t kInt4Group = 128;
inline int64_t int4g_bytes(int64_t rows, int64_t cols) { return rows * (cols / 2); }
inline int64_t int4g_groups(int64_t cols) { return (cols + kInt4Group - 1) / kInt4Group; }  // the last may be short
inline int64_t int4g_scales(int64_t rows, int64_t cols) { return rows * int4g_groups(cols); }
inline int64_t int4g_scale_index(int64_t rows, int64_t r, int64_t c) { return (c / kInt4Group) * rows + r; }
inline bool int4g_tiled(int64_t rows, int64_t cols) { return rows % 16 == 0 && cols % 64 == 0; }
// Weight::layout for INT4G: the MMA fragments where the shape allows them, or row-major always
constexpr int32_t kInt4Fragments = 0, kInt4RowMajor = 1, kInt4CpuSimd = 2;
inline bool int4g_frag(int64_t rows, int64_t cols, int32_t layout) { return layout == kInt4Fragments && int4g_tiled(rows, cols); }

// Where value (r, c) of a packed [rows][cols] matrix lives: its 32-bit word's byte offset and its
// nibble (0-7). Rows 2i and 2i + 1 share words, no other rows do.
inline void int4g_locate(int64_t rows, int64_t cols, int64_t r, int64_t c, int64_t &off, int &nib,
                         int32_t layout = kInt4Fragments) {
    static constexpr int kTiled[2][4] = {{0, 4, 2, 6}, {1, 5, 3, 7}};  // [row A / B][column K0 + e]
    static constexpr int kRow[8] = {0, 2, 4, 6, 1, 3, 5, 7};          // [column j of the word]
    if (int4g_frag(rows, cols, layout)) {
        const int64_t T = r / 16, g = (r % 16) / 2, ch = c / 64, t = (c % 64) / 16, s = (c % 16) / 4;
        off = ((T * (cols / 64) + ch) * 32 + g * 4 + t) * 16 + s * 4;
        nib = kTiled[r % 2][c % 4];
    } else if (layout == kInt4CpuSimd) {
        const int64_t byte = r * (cols / 2) + (c / 64) * 32 + (c % 64) % 32;
        off = byte & ~int64_t{3};
        nib = static_cast<int>(byte & 3) * 2 + ((c % 64) >= 32 ? 1 : 0);
    } else {
        off = r * (cols / 2) + (c / 8) * 4;
        nib = kRow[c % 8];
    }
}

// The value at (r, c) of a packed [rows][cols] matrix (without its scale).
inline int int4g_value(const uint8_t *q, int64_t rows, int64_t cols, int64_t r, int64_t c,
                       int32_t layout = kInt4Fragments) {
    int64_t off;
    int nib;
    int4g_locate(rows, cols, r, c, off, nib, layout);
    uint32_t word;
    std::memcpy(&word, q + off, 4);
    return static_cast<int>((word >> (4 * nib)) & 15u) - 8;
}

inline float int4g_bf16_f32(uint16_t h) {
    const uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// w: bf16 [rows][cols], cols a multiple of 8 (the decode tile wants rows % 16 == 0 and whole
// 256-column groups). q: int4g_bytes, packed in layout; scale: int4g_scales floats.
// Threads over row pairs (a pair's words are its own); the result does not depend on their count.
// row(r) for every row, on threads over row pairs (a pair's words are its own).
template <typename F>
inline void int4g_for_rows(int64_t rows, const F &row) {
    const int64_t pairs = (rows + 1) / 2;
    auto pair = [&](int64_t p) {
        for (int64_t r = 2 * p; r < std::min(rows, 2 * p + 2); ++r) row(r);
    };
    const int64_t T = std::min<int64_t>(pairs, std::max(1u, std::thread::hardware_concurrency()));
    if (T <= 1) {
        for (int64_t p = 0; p < pairs; ++p) pair(p);
        return;
    }
    std::vector<std::thread> th;
    for (int64_t t = 0; t < T; ++t)
        th.emplace_back([&, t] {
            for (int64_t p = t; p < pairs; p += T) pair(p);
        });
    for (auto &x : th) x.join();
}

// Values quantised elsewhere (a GPTQ checkpoint's experts: one int8 each in [-8, 7], [rows][cols],
// with their own group scales) packed in layout as quantize_int4_groups packs its own.
inline void pack_int4_values(const int8_t *v, int64_t rows, int64_t cols, uint8_t *q, int32_t layout = kInt4Fragments) {
    std::memset(q, 0, static_cast<size_t>(int4g_bytes(rows, cols)));
    int4g_for_rows(rows, [&](int64_t r) {
        for (int64_t c = 0; c < cols; ++c) {
            int64_t off;
            int nib;
            int4g_locate(rows, cols, r, c, off, nib, layout);
            uint32_t word;
            std::memcpy(&word, q + off, 4);
            word |= static_cast<uint32_t>((v[r * cols + c] + 8) & 15) << (4 * nib);
            std::memcpy(q + off, &word, 4);
        }
    });
}

// bf16 of value x scale (round to nearest even), [rows][cols]: the fp8b copies of pre-quantised
// int4 experts.
inline void dequant_int4_values_bf16(const int8_t *v, const float *scale, int64_t rows, int64_t cols, uint16_t *out) {
    int4g_for_rows(rows, [&](int64_t r) {
        for (int64_t c = 0; c < cols; ++c) {
            const float f = static_cast<float>(v[r * cols + c]) * scale[int4g_scale_index(rows, r, c)];
            uint32_t u;
            std::memcpy(&u, &f, 4);
            u += 0x7FFFu + ((u >> 16) & 1u);
            out[r * cols + c] = static_cast<uint16_t>(u >> 16);
        }
    });
}

inline void quantize_int4_groups(const uint16_t *w, int64_t rows, int64_t cols, uint8_t *q, float *scale,
                                 int32_t layout = kInt4Fragments) {
    static constexpr float kClip[] = {1.0f, 0.95f, 0.9f, 0.85f, 0.8f, 0.75f, 0.7f};
    // FULLM_INT4_CLIP=1: the clip search (see the top of the file); else the absolute maximum only
    static const int n_clip = [] {
        const char *e = std::getenv("FULLM_INT4_CLIP");
        return e && std::string(e) == "1" ? static_cast<int>(std::size(kClip)) : 1;
    }();
    const int64_t ng = int4g_groups(cols);
    std::memset(q, 0, static_cast<size_t>(int4g_bytes(rows, cols)));
    auto row = [&](int64_t r) {
        float v[kInt4Group];
        int qv[kInt4Group], best_q[kInt4Group];
        for (int64_t g = 0; g < ng; ++g) {
            const int64_t gl = std::min(kInt4Group, cols - g * kInt4Group);
            float amax = 0.0f;
            for (int64_t i = 0; i < gl; ++i) {
                v[i] = int4g_bf16_f32(w[r * cols + g * kInt4Group + i]);
                amax = std::max(amax, std::fabs(v[i]));
            }
            float best_s = amax > 0.0f ? amax / 7.0f : 1.0f, best_e = INFINITY;
            for (int ci = 0; ci < n_clip; ++ci) {
                const float s = amax > 0.0f ? kClip[ci] * amax / 7.0f : 1.0f;
                float e = 0.0f;
                for (int64_t i = 0; i < gl; ++i) {
                    const int qq = std::clamp(static_cast<int>(std::nearbyint(v[i] / s)), -8, 7);
                    qv[i] = qq;
                    const float d = v[i] - static_cast<float>(qq) * s;
                    e += d * d;
                }
                if (e < best_e) {
                    best_e = e, best_s = s;
                    std::memcpy(best_q, qv, sizeof(qv));
                }
                if (amax == 0.0f) break;
            }
            scale[g * rows + r] = best_s;
            for (int64_t i = 0; i < gl; ++i) {
                int64_t off;
                int nib;
                int4g_locate(rows, cols, r, g * kInt4Group + i, off, nib, layout);
                uint32_t word;
                std::memcpy(&word, q + off, 4);
                word |= static_cast<uint32_t>(best_q[i] + 8) << (4 * nib);
                std::memcpy(q + off, &word, 4);
            }
        }
    };
    int4g_for_rows(rows, row);
}

}  // namespace fullm
