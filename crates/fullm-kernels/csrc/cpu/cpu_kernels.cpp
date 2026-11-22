// SPDX-License-Identifier: Apache-2.0
// CPU kernels. See fullm_kernels.h for the contract; the short version is:
// row-major [N][K] weights, fp32 activations, no threads, batch-invariant dots.
//
// SHAPE OF EVERY GEMM HERE
//   for each weight row n:   decode it ONCE into an f32 buffer (exact for every format)
//     for each activation row m (four at a time):   y[m][n] = dot(x[m], w) * row_scale
// The decode is exact (bf16 widens, int8 converts, E4M3 and MXFP4 values are representable
// in f32), so the only rounding is in the dot, and the dot's summation order per (m, n) is
// fixed: dot4 runs four rows through exactly the sequence dot runs one through. That is
// the batch-invariance argument: M and the [n0, n1) split decide WHICH dots run, never HOW
// one is summed. Decoding once also means a row is unpacked once per call rather than once
// per activation row, which is what makes the prefill (many tokens per expert) path
// compute-bound instead of decode-bound.
//
// MXFP4 rows are decoded in a GROUP-DEINTERLEAVED order (per 32-element group: the 16 even
// elements, then the 16 odd ones), which falls straight out of the nibble layout, and x is
// permuted the same way once per call; the dot is then an ordinary dot.

#include "fullm_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define SK_X86 1
#endif

namespace {

// ------------------------------------------------------------------ format tables

inline float bf16_to_f32(uint16_t h) {
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// OCP E2M1; bit 3 is the sign.
alignas(64) const float kE2M1[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                     -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

struct Tables {
    float e8m0[256];  // 2^(s-127); 255 is NaN by spec and maps to 0 so one bad byte
                      // cannot poison a whole row.
    float e4m3[256];
    Tables() {
        for (int s = 0; s < 256; ++s) e8m0[s] = s == 255 ? 0.0f : std::ldexp(1.0f, s - 127);
        for (int c = 0; c < 256; ++c) {
            const int sign = c >> 7, exp = (c >> 3) & 0xF, man = c & 0x7;
            float v;
            if (exp == 0xF && man == 0x7) v = 0.0f;  // NaN code
            else if (exp == 0) v = std::ldexp(static_cast<float>(man) / 8.0f, -6);
            else v = std::ldexp(1.0f + static_cast<float>(man) / 8.0f, exp - 7);
            e4m3[c] = sign ? -v : v;
        }
    }
};
const Tables &tables() {
    static const Tables t;
    return t;
}

// The fixed reduction tree for a 16-lane partial sum. Every portable dot uses it.
inline float reduce16(const float a[16]) {
    float b[8];
    for (int i = 0; i < 8; ++i) b[i] = a[i] + a[i + 8];
    float c[4];
    for (int i = 0; i < 4; ++i) c[i] = b[i] + b[i + 4];
    return (c[0] + c[2]) + (c[1] + c[3]);
}

// ------------------------------------------------------------------ dots
// dot(x, w, K) and dot4 (four x rows, one w) with identical per-row arithmetic.

// Never inlined: dot4 and every other caller then run this one compiled body. Inlined copies let
// the compiler contract the tail's `s += x * w` to an fma in one caller and not in another, and a
// row's last bits then depended on whether it went through dot or dot4 (widths off the 16 step).
__attribute__((noinline)) float dot_portable(const float *x, const float *w, int64_t K) {
    float acc[16] = {0};
    int64_t k = 0;
    for (; k + 16 <= K; k += 16)
        for (int j = 0; j < 16; ++j) acc[j] += x[k + j] * w[k + j];
    float s = reduce16(acc);
    for (; k < K; ++k) s += x[k] * w[k];
    return s;
}

void dot4_portable(const float *x0, const float *x1, const float *x2, const float *x3, const float *w, int64_t K,
                   float *out) {
    out[0] = dot_portable(x0, w, K);
    out[1] = dot_portable(x1, w, K);
    out[2] = dot_portable(x2, w, K);
    out[3] = dot_portable(x3, w, K);
}

#ifdef SK_X86

// The SIMD dots' scalar tails are explicit fmas: left to the compiler, `s += x * w` was contracted to
// an fma in one function and not in another, and a row's last bits then depended on whether it went
// through dot or dot4 (widths off the 16 / 32 steps only; bf16_rows_dotted_as_read_match_decoded).
__attribute__((target("avx512f,avx512bw,avx512dq,fma"))) float dot_avx512(const float *x, const float *w, int64_t K) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    int64_t k = 0;
    for (; k + 32 <= K; k += 32) {
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(w + k), _mm512_loadu_ps(x + k), a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(w + k + 16), _mm512_loadu_ps(x + k + 16), a1);
    }
    float s = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
    for (; k < K; ++k) s = std::fma(x[k], w[k], s);
    return s;
}

__attribute__((target("avx512f,avx512bw,avx512dq,fma"))) void dot4_avx512(const float *x0, const float *x1,
                                                                           const float *x2, const float *x3,
                                                                           const float *w, int64_t K, float *out) {
    __m512 a0[4], a1[4];
    for (int r = 0; r < 4; ++r) a0[r] = a1[r] = _mm512_setzero_ps();
    const float *xs[4] = {x0, x1, x2, x3};
    int64_t k = 0;
    for (; k + 32 <= K; k += 32) {
        const __m512 w0 = _mm512_loadu_ps(w + k), w1 = _mm512_loadu_ps(w + k + 16);
        for (int r = 0; r < 4; ++r) {
            a0[r] = _mm512_fmadd_ps(w0, _mm512_loadu_ps(xs[r] + k), a0[r]);
            a1[r] = _mm512_fmadd_ps(w1, _mm512_loadu_ps(xs[r] + k + 16), a1[r]);
        }
    }
    for (int r = 0; r < 4; ++r) {
        float s = _mm512_reduce_add_ps(_mm512_add_ps(a0[r], a1[r]));
        for (int64_t j = k; j < K; ++j) s = std::fma(xs[r][j], w[j], s);
        out[r] = s;
    }
}

__attribute__((target("avx2,fma"))) inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(lo);
    __m128 s = _mm_add_ps(lo, sh);
    sh = _mm_movehl_ps(sh, s);
    return _mm_cvtss_f32(_mm_add_ss(s, sh));
}

__attribute__((target("avx2,fma"))) float dot_avx2(const float *x, const float *w, int64_t K) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int64_t k = 0;
    for (; k + 16 <= K; k += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + k), _mm256_loadu_ps(x + k), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + k + 8), _mm256_loadu_ps(x + k + 8), a1);
    }
    float s = hsum256(_mm256_add_ps(a0, a1));
    for (; k < K; ++k) s = std::fma(x[k], w[k], s);
    return s;
}

__attribute__((target("avx2,fma"))) void dot4_avx2(const float *x0, const float *x1, const float *x2,
                                                   const float *x3, const float *w, int64_t K, float *out) {
    __m256 a0[4], a1[4];
    for (int r = 0; r < 4; ++r) a0[r] = a1[r] = _mm256_setzero_ps();
    const float *xs[4] = {x0, x1, x2, x3};
    int64_t k = 0;
    for (; k + 16 <= K; k += 16) {
        const __m256 w0 = _mm256_loadu_ps(w + k), w1 = _mm256_loadu_ps(w + k + 8);
        for (int r = 0; r < 4; ++r) {
            a0[r] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xs[r] + k), a0[r]);
            a1[r] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xs[r] + k + 8), a1[r]);
        }
    }
    for (int r = 0; r < 4; ++r) {
        float s = hsum256(_mm256_add_ps(a0[r], a1[r]));
        for (int64_t j = k; j < K; ++j) s = std::fma(xs[r][j], w[j], s);
        out[r] = s;
    }
}

#endif  // SK_X86

// ------------------------------------------------------------------ row decoders (exact)

void dec_bf16_portable(const uint16_t *w, int64_t K, float *out) {
    for (int64_t k = 0; k < K; ++k) out[k] = bf16_to_f32(w[k]);
}
void dec_i8_portable(const int8_t *w, int64_t K, float *out) {
    for (int64_t k = 0; k < K; ++k) out[k] = static_cast<float>(w[k]);
}
void dec_fp8_portable(const uint8_t *w, int64_t K, float *out) {
    const float *lut = tables().e4m3;
    for (int64_t k = 0; k < K; ++k) out[k] = lut[w[k]];
}
// Group-deinterleaved: out[g*32 + j] = element 2j, out[g*32 + 16 + j] = element 2j+1.
// int4 rows in the kInt4CpuSimd packing: per 64-column chunk 32 bytes, low nibbles columns 0-31,
// high nibbles 32-63; q - 8 times the 128-group's scale (sc[(k / 128) * rows]).
void dec_i4s_portable(const uint8_t *q, const float *sc, int64_t rows, int64_t K, float *out) {
    for (int64_t c = 0; c < K; c += 64) {
        const float s = sc[(c / 128) * rows];
        const uint8_t *b = q + c / 2;
        for (int j = 0; j < 32; ++j) {
            out[c + j] = static_cast<float>(static_cast<int>(b[j] & 15) - 8) * s;
            out[c + 32 + j] = static_cast<float>(static_cast<int>(b[j] >> 4) - 8) * s;
        }
    }
}

void dec_mx_portable(const uint8_t *p, const uint8_t *sc, int64_t K, float *out) {
    const float *e8 = tables().e8m0;
    for (int64_t g = 0; g < K / 32; ++g) {
        const uint8_t *b = p + g * 16;
        const float s = e8[sc[g]];
        float *o = out + g * 32;
        for (int j = 0; j < 16; ++j) {
            o[j] = kE2M1[b[j] & 0xF] * s;
            o[16 + j] = kE2M1[b[j] >> 4] * s;
        }
    }
}

#ifdef SK_X86

__attribute__((target("avx512f,avx512bw,avx512dq,fma"))) void dec_bf16_avx512(const uint16_t *w, int64_t K,
                                                                               float *out) {
    int64_t k = 0;
    for (; k + 16 <= K; k += 16) {
        const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + k));
        _mm512_storeu_ps(out + k, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16)));
    }
    for (; k < K; ++k) out[k] = bf16_to_f32(w[k]);
}

// bf16 rows dotted as they are read (no decoded row in between): the same two accumulators and
// order as dot_avx512 over a decoded row, so the same bits. rows4: four rows of w (stride K) at
// once, four streams and x loaded once, each row's sum as the single-row form's.
__attribute__((target("avx512f,avx512bw,avx512dq,fma"))) inline __m512 bf16x16_avx512(const uint16_t *p) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(p))), 16));
}
__attribute__((target("avx512f,avx512bw,avx512dq,fma"))) float dot_bf16_avx512(const float *x, const uint16_t *w, int64_t K) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    int64_t k = 0;
    for (; k + 32 <= K; k += 32) {
        a0 = _mm512_fmadd_ps(bf16x16_avx512(w + k), _mm512_loadu_ps(x + k), a0);
        a1 = _mm512_fmadd_ps(bf16x16_avx512(w + k + 16), _mm512_loadu_ps(x + k + 16), a1);
    }
    float s = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
    for (; k < K; ++k) s = std::fma(x[k], bf16_to_f32(w[k]), s);
    return s;
}
__attribute__((target("avx512f,avx512bw,avx512dq,fma"))) void dot_bf16_rows4_avx512(const float *x, const uint16_t *w,
                                                                                     int64_t K, float *out) {
    __m512 a0[4], a1[4];
    for (int r = 0; r < 4; ++r) a0[r] = a1[r] = _mm512_setzero_ps();
    int64_t k = 0;
    for (; k + 32 <= K; k += 32) {
        const __m512 x0 = _mm512_loadu_ps(x + k), x1 = _mm512_loadu_ps(x + k + 16);
        for (int r = 0; r < 4; ++r) {
            a0[r] = _mm512_fmadd_ps(bf16x16_avx512(w + r * K + k), x0, a0[r]);
            a1[r] = _mm512_fmadd_ps(bf16x16_avx512(w + r * K + k + 16), x1, a1[r]);
        }
    }
    for (int r = 0; r < 4; ++r) {
        float s = _mm512_reduce_add_ps(_mm512_add_ps(a0[r], a1[r]));
        for (int64_t j = k; j < K; ++j) s = std::fma(x[j], bf16_to_f32(w[r * K + j]), s);
        out[r] = s;
    }
}

__attribute__((target("avx512f,avx512bw,avx512dq,fma"))) void dec_i8_avx512(const int8_t *w, int64_t K, float *out) {
    int64_t k = 0;
    for (; k + 16 <= K; k += 16) {
        const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w + k));
        _mm512_storeu_ps(out + k, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b)));
    }
    for (; k < K; ++k) out[k] = static_cast<float>(w[k]);
}

__attribute__((target("avx512f,avx512bw,avx512dq,fma"))) void dec_fp8_avx512(const uint8_t *w, int64_t K,
                                                                              float *out) {
    const float *lut = tables().e4m3;
    int64_t k = 0;
    for (; k + 16 <= K; k += 16) {
        const __m512i idx = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w + k)));
        _mm512_storeu_ps(out + k, _mm512_i32gather_ps(idx, lut, 4));
    }
    for (; k < K; ++k) out[k] = lut[w[k]];
}

__attribute__((target("avx512f,avx512bw,avx512dq,fma"))) void dec_mx_avx512(const uint8_t *p, const uint8_t *sc,
                                                                             int64_t K, float *out) {
    const float *e8 = tables().e8m0;
    const __m512 lut = _mm512_load_ps(kE2M1);
    const __m512i lo_mask = _mm512_set1_epi32(0x0F);
    for (int64_t g = 0; g < K / 32; ++g) {
        const __m512i b = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p + g * 16)));
        const __m512 s = _mm512_set1_ps(e8[sc[g]]);
        _mm512_storeu_ps(out + g * 32, _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_and_si512(b, lo_mask), lut), s));
        _mm512_storeu_ps(out + g * 32 + 16, _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(b, 4), lut), s));
    }
}

__attribute__((target("avx2,fma"))) void dec_bf16_avx2(const uint16_t *w, int64_t K, float *out) {
    int64_t k = 0;
    for (; k + 8 <= K; k += 8) {
        const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w + k));
        _mm256_storeu_ps(out + k, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16)));
    }
    for (; k < K; ++k) out[k] = bf16_to_f32(w[k]);
}

// as dot_bf16_avx512 for dot_avx2's two 8-wide accumulators
__attribute__((target("avx2,fma"))) inline __m256 bf16x8_avx2(const uint16_t *p) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p))), 16));
}
__attribute__((target("avx2,fma"))) float dot_bf16_avx2(const float *x, const uint16_t *w, int64_t K) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int64_t k = 0;
    for (; k + 16 <= K; k += 16) {
        a0 = _mm256_fmadd_ps(bf16x8_avx2(w + k), _mm256_loadu_ps(x + k), a0);
        a1 = _mm256_fmadd_ps(bf16x8_avx2(w + k + 8), _mm256_loadu_ps(x + k + 8), a1);
    }
    float s = hsum256(_mm256_add_ps(a0, a1));
    for (; k < K; ++k) s = std::fma(x[k], bf16_to_f32(w[k]), s);
    return s;
}
__attribute__((target("avx2,fma"))) void dot_bf16_rows4_avx2(const float *x, const uint16_t *w, int64_t K, float *out) {
    __m256 a0[4], a1[4];
    for (int r = 0; r < 4; ++r) a0[r] = a1[r] = _mm256_setzero_ps();
    int64_t k = 0;
    for (; k + 16 <= K; k += 16) {
        const __m256 x0 = _mm256_loadu_ps(x + k), x1 = _mm256_loadu_ps(x + k + 8);
        for (int r = 0; r < 4; ++r) {
            a0[r] = _mm256_fmadd_ps(bf16x8_avx2(w + r * K + k), x0, a0[r]);
            a1[r] = _mm256_fmadd_ps(bf16x8_avx2(w + r * K + k + 8), x1, a1[r]);
        }
    }
    for (int r = 0; r < 4; ++r) {
        float s = hsum256(_mm256_add_ps(a0[r], a1[r]));
        for (int64_t j = k; j < K; ++j) s = std::fma(x[j], bf16_to_f32(w[r * K + j]), s);
        out[r] = s;
    }
}

__attribute__((target("avx2,fma"))) void dec_i8_avx2(const int8_t *w, int64_t K, float *out) {
    int64_t k = 0;
    for (; k + 8 <= K; k += 8) {
        const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(w + k));
        _mm256_storeu_ps(out + k, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b)));
    }
    for (; k < K; ++k) out[k] = static_cast<float>(w[k]);
}

__attribute__((target("avx2,fma"))) void dec_fp8_avx2(const uint8_t *w, int64_t K, float *out) {
    const float *lut = tables().e4m3;
    int64_t k = 0;
    for (; k + 8 <= K; k += 8) {
        const __m256i idx = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(w + k)));
        _mm256_storeu_ps(out + k, _mm256_i32gather_ps(lut, idx, 4));
    }
    for (; k < K; ++k) out[k] = lut[w[k]];
}

__attribute__((target("avx2,fma"))) void dec_i4s_avx2(const uint8_t *q, const float *sc, int64_t rows, int64_t K,
                                                      float *out) {
    const __m256i m4 = _mm256_set1_epi8(0x0F), eight = _mm256_set1_epi32(8);
    for (int64_t c = 0; c < K; c += 64) {
        const __m256 s = _mm256_set1_ps(sc[(c / 128) * rows]);
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q + c / 2));
        const __m256i lo = _mm256_and_si256(b, m4), hi = _mm256_and_si256(_mm256_srli_epi16(b, 4), m4);
        // 8 bytes at a time: 32 lanes of lo (columns c .. c + 31), then of hi (c + 32 .. c + 63)
        const __m128i l0 = _mm256_castsi256_si128(lo), l1 = _mm256_extracti128_si256(lo, 1);
        const __m128i h0 = _mm256_castsi256_si128(hi), h1 = _mm256_extracti128_si256(hi, 1);
        const __m128i parts[8] = {l0, _mm_srli_si128(l0, 8), l1, _mm_srli_si128(l1, 8),
                                  h0, _mm_srli_si128(h0, 8), h1, _mm_srli_si128(h1, 8)};
#pragma GCC unroll 8
        for (int j = 0; j < 8; ++j) {
            const __m256i v = _mm256_sub_epi32(_mm256_cvtepu8_epi32(parts[j]), eight);
            _mm256_storeu_ps(out + c + 8 * j, _mm256_mul_ps(_mm256_cvtepi32_ps(v), s));
        }
    }
}

// AVX2 has an 8-entry permute: decode the 3 magnitude bits through it and move the sign
// bit (bit 3) to bit 31.
__attribute__((target("avx2,fma"))) inline __m256 e2m1_avx2(__m256i nib, __m256 lut8) {
    const __m256 mag = _mm256_permutevar8x32_ps(lut8, _mm256_and_si256(nib, _mm256_set1_epi32(7)));
    const __m256i sgn = _mm256_slli_epi32(_mm256_and_si256(nib, _mm256_set1_epi32(8)), 28);
    return _mm256_xor_ps(mag, _mm256_castsi256_ps(sgn));
}

__attribute__((target("avx2,fma"))) void dec_mx_avx2(const uint8_t *p, const uint8_t *sc, int64_t K, float *out) {
    const float *e8 = tables().e8m0;
    const __m256 lut8 = _mm256_loadu_ps(kE2M1);
    for (int64_t g = 0; g < K / 32; ++g) {
        const __m256 s = _mm256_set1_ps(e8[sc[g]]);
        for (int h = 0; h < 2; ++h) {
            const __m256i b =
                _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(p + g * 16 + h * 8)));
            _mm256_storeu_ps(out + g * 32 + h * 8,
                             _mm256_mul_ps(e2m1_avx2(_mm256_and_si256(b, _mm256_set1_epi32(0xF)), lut8), s));
            _mm256_storeu_ps(out + g * 32 + 16 + h * 8, _mm256_mul_ps(e2m1_avx2(_mm256_srli_epi32(b, 4), lut8), s));
        }
    }
}

// Fused MXFP4 decode + dot for a single activation row. The FMAs land in the same lanes
// and the same order as dot_avx512 / dot_avx2 over a dec_mx_* buffer (the decoded values
// are exact), so the result is bit-identical to decode-then-dot. That is what lets the
// M=1 decode path skip the buffer without breaking batch invariance.
__attribute__((target("avx512f,avx512bw,avx512dq,fma"))) float dot_mx_avx512(const float *xd, const uint8_t *p,
                                                                             const uint8_t *sc, int64_t K) {
    const float *e8 = tables().e8m0;
    const __m512 lut = _mm512_load_ps(kE2M1);
    const __m512i lo_mask = _mm512_set1_epi32(0x0F);
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    for (int64_t g = 0; g < K / 32; ++g) {
        const __m512i b = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p + g * 16)));
        const __m512 s = _mm512_set1_ps(e8[sc[g]]);
        const __m512 wl = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_and_si512(b, lo_mask), lut), s);
        const __m512 wh = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(b, 4), lut), s);
        a0 = _mm512_fmadd_ps(wl, _mm512_loadu_ps(xd + g * 32), a0);
        a1 = _mm512_fmadd_ps(wh, _mm512_loadu_ps(xd + g * 32 + 16), a1);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
}

__attribute__((target("avx2,fma"))) float dot_mx_avx2(const float *xd, const uint8_t *p, const uint8_t *sc,
                                                      int64_t K) {
    const float *e8 = tables().e8m0;
    const __m256 lut8 = _mm256_loadu_ps(kE2M1);
    const __m256i m4 = _mm256_set1_epi32(0xF);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    for (int64_t g = 0; g < K / 32; ++g) {
        const __m256 s = _mm256_set1_ps(e8[sc[g]]);
        const __m256i b0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(p + g * 16)));
        const __m256i b1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(p + g * 16 + 8)));
        const float *x = xd + g * 32;
        // buffer order: [evens 0-7][evens 8-15][odds 0-7][odds 8-15]; dot_avx2 takes 16 at a time
        a0 = _mm256_fmadd_ps(_mm256_mul_ps(e2m1_avx2(_mm256_and_si256(b0, m4), lut8), s), _mm256_loadu_ps(x), a0);
        a1 = _mm256_fmadd_ps(_mm256_mul_ps(e2m1_avx2(_mm256_and_si256(b1, m4), lut8), s), _mm256_loadu_ps(x + 8), a1);
        a0 = _mm256_fmadd_ps(_mm256_mul_ps(e2m1_avx2(_mm256_srli_epi32(b0, 4), lut8), s), _mm256_loadu_ps(x + 16), a0);
        a1 = _mm256_fmadd_ps(_mm256_mul_ps(e2m1_avx2(_mm256_srli_epi32(b1, 4), lut8), s), _mm256_loadu_ps(x + 24), a1);
    }
    return hsum256(_mm256_add_ps(a0, a1));
}

#endif  // SK_X86

// ------------------------------------------------------------------ dispatch

enum class Isa { Scalar, Neon, Avx2, Avx512 };

Isa detect() {
#ifdef SK_X86
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq"))
        return Isa::Avx512;
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) return Isa::Avx2;
    return Isa::Scalar;
#elif defined(__aarch64__)
    return Isa::Neon;  // the portable code vectorises to NEON
#else
    return Isa::Scalar;
#endif
}

Isa isa() {
    static const Isa v = detect();
    return v;
}

struct Ops {
    float (*dot)(const float *, const float *, int64_t) = dot_portable;
    void (*dot4)(const float *, const float *, const float *, const float *, const float *, int64_t, float *) =
        dot4_portable;
    void (*bf16)(const uint16_t *, int64_t, float *) = dec_bf16_portable;
    void (*i8)(const int8_t *, int64_t, float *) = dec_i8_portable;
    void (*fp8)(const uint8_t *, int64_t, float *) = dec_fp8_portable;
    void (*mx)(const uint8_t *, const uint8_t *, int64_t, float *) = dec_mx_portable;
    void (*i4s)(const uint8_t *, const float *, int64_t, int64_t, float *) = dec_i4s_portable;
    // fused decode + dot for one row; null where decode-then-dot is just as good
    float (*dot_mx)(const float *, const uint8_t *, const uint8_t *, int64_t) = nullptr;
    // bf16 rows dotted as read, the same bits as bf16 + dot (null: decode, then dot)
    float (*dot_bf16)(const float *, const uint16_t *, int64_t) = nullptr;
    void (*dot_bf16_rows4)(const float *, const uint16_t *, int64_t, float *) = nullptr;
};

const Ops &ops() {
    static const Ops o = [] {
        Ops r;
#ifdef SK_X86
        if (isa() == Isa::Avx512) {
            r.dot = dot_avx512;
            r.dot4 = dot4_avx512;
            r.bf16 = dec_bf16_avx512;
            r.i8 = dec_i8_avx512;
            r.fp8 = dec_fp8_avx512;
            r.mx = dec_mx_avx512;
            r.dot_mx = dot_mx_avx512;
            r.i4s = dec_i4s_avx2;
            r.dot_bf16 = dot_bf16_avx512;
            r.dot_bf16_rows4 = dot_bf16_rows4_avx512;
        } else if (isa() == Isa::Avx2) {
            r.dot = dot_avx2;
            r.dot4 = dot4_avx2;
            r.bf16 = dec_bf16_avx2;
            r.i8 = dec_i8_avx2;
            r.fp8 = dec_fp8_avx2;
            r.mx = dec_mx_avx2;
            r.dot_mx = dot_mx_avx2;
            r.i4s = dec_i4s_avx2;
            r.dot_bf16 = dot_bf16_avx2;
            r.dot_bf16_rows4 = dot_bf16_rows4_avx2;
        }
#endif
        return r;
    }();
    return o;
}

// y[m][n] = dot(x[m], decoded row n) * scale(n), rows of x taken four at a time.
//
// Small M (decode steps): row by row, each decoded into L1 and dotted with every x row at once.
// Large M (prefill, x past L2): NB rows are decoded first (<= 512 KB, in L2 beside the four x
// rows a dot4 streams from L1), and each group of x rows then visits them all before the next
// group: x is read from L3 or DRAM once per NB rows instead of once per row. The dots are the
// same calls in the same k order either way: the same bits (the CPU backend is the reference).
template <typename Decode, typename Scale>
void gemm_decoded(const float *x, int64_t M, int64_t K, int64_t n0, int64_t n1, float *y, int64_t ldy,
                  Decode decode, Scale scale) {
    const Ops &o = ops();
    constexpr int64_t kMaxNB = 64;
    const bool big = M * K * static_cast<int64_t>(sizeof(float)) > (int64_t{1} << 20);
    const int64_t NB = big ? std::clamp<int64_t>((int64_t{512} << 10) / (K * static_cast<int64_t>(sizeof(float))), 1, kMaxNB) : 1;
    thread_local std::vector<float> wbuf;
    if (static_cast<int64_t>(wbuf.size()) < NB * K) wbuf.resize(static_cast<size_t>(NB * K));
    for (int64_t nb = n0; nb < n1; nb += NB) {
        const int64_t ne = std::min(n1, nb + NB);
        const float *rows[kMaxNB];
        float sc[kMaxNB];
        for (int64_t n = nb; n < ne; ++n) {
            rows[n - nb] = decode(n, wbuf.data() + (n - nb) * K);
            sc[n - nb] = scale(n);
        }
        int64_t m = 0;
        for (; m + 4 <= M; m += 4) {
            for (int64_t n = nb; n < ne; ++n) {
                float r[4];
                o.dot4(x + m * K, x + (m + 1) * K, x + (m + 2) * K, x + (m + 3) * K, rows[n - nb], K, r);
                for (int j = 0; j < 4; ++j) y[(m + j) * ldy + n] = r[j] * sc[n - nb];
            }
        }
        for (; m < M; ++m)
            for (int64_t n = nb; n < ne; ++n) y[m * ldy + n] = o.dot(x + m * K, rows[n - nb], K) * sc[n - nb];
    }
}

// E4M3 round-to-nearest-even with saturation to +-448.
uint8_t f32_to_e4m3(float v) {
    const uint8_t sign = std::signbit(v) ? 0x80 : 0x00;
    float a = std::fabs(v);
    if (!(a == a)) return 0;  // NaN input: emit zero rather than a NaN code
    if (a >= 448.0f) return sign | 0x7E;
    // Search the 127 non-negative finite codes; exact and simple, and only ever run at
    // conversion time, never in a forward pass.
    const float *lut = tables().e4m3;
    int lo = 0, hi = 0x7E;
    while (lo < hi) {  // first code with value >= a
        int mid = (lo + hi) / 2;
        if (lut[mid] < a) lo = mid + 1;
        else hi = mid;
    }
    if (lo == 0) return sign;
    const float above = lut[lo], below = lut[lo - 1];
    const float da = above - a, db = a - below;
    int code;
    if (da < db) code = lo;
    else if (db < da) code = lo - 1;
    else code = (lo & 1) ? lo - 1 : lo;  // tie: even mantissa
    return sign | static_cast<uint8_t>(code);
}

}  // namespace

extern "C" {

const char *sk_cpu_isa(void) {
    switch (isa()) {
        case Isa::Avx512: return "avx512";
        case Isa::Avx2: return "avx2";
        case Isa::Neon: return "neon";
        default: return "scalar";
    }
}

const float *sk_fp8_e4m3_table(void) { return tables().e4m3; }

void sk_cpu_gemm_f32(const float *x, int64_t M, int64_t K, const float *W, int64_t n0, int64_t n1, float *y,
                     int64_t ldy) {
    gemm_decoded(
        x, M, K, n0, n1, y, ldy, [&](int64_t n, float *) { return W + n * K; }, [](int64_t) { return 1.0f; });
}

void sk_cpu_gemm_bf16(const float *x, int64_t M, int64_t K, const uint16_t *W, int64_t n0, int64_t n1, float *y,
                      int64_t ldy) {
    const Ops &o = ops();
    // Fewer than four rows of x (a decode step, an expert's few tokens): each row of W dotted as it
    // is read, four rows of W at a time, x's rows over a block of four while it is in L1. That is
    // what gemm_decoded does for these rows (o.dot on each decoded row; the scale is 1), without
    // the decoded copy: the same bits, and the weights streamed once at the memory's rate (the
    // CPU expert pool's bf16 GEMV runs at 66-105 GB/s of 156 on a Sapphire Rapids server).
    if (o.dot_bf16 && M < 4) {
        int64_t n = n0;
        for (; n + 4 <= n1; n += 4)
            for (int64_t m = 0; m < M; ++m) o.dot_bf16_rows4(x + m * K, W + n * K, K, y + m * ldy + n);
        for (; n < n1; ++n)
            for (int64_t m = 0; m < M; ++m) y[m * ldy + n] = o.dot_bf16(x + m * K, W + n * K, K);
        return;
    }
    gemm_decoded(
        x, M, K, n0, n1, y, ldy,
        [&](int64_t n, float *buf) {
            o.bf16(W + n * K, K, buf);
            return static_cast<const float *>(buf);
        },
        [](int64_t) { return 1.0f; });
}

void sk_cpu_gemm_fp8(const float *x, int64_t M, int64_t K, const uint8_t *W, const float *row_scale, int64_t n0,
                     int64_t n1, float *y, int64_t ldy) {
    const Ops &o = ops();
    gemm_decoded(
        x, M, K, n0, n1, y, ldy,
        [&](int64_t n, float *buf) {
            o.fp8(W + n * K, K, buf);
            return static_cast<const float *>(buf);
        },
        [&](int64_t n) { return row_scale[n]; });
}

void sk_cpu_gemm_i8(const float *x, int64_t M, int64_t K, const int8_t *W, const float *row_scale, int64_t n0,
                    int64_t n1, float *y, int64_t ldy) {
    const Ops &o = ops();
    gemm_decoded(
        x, M, K, n0, n1, y, ldy,
        [&](int64_t n, float *buf) {
            o.i8(W + n * K, K, buf);
            return static_cast<const float *>(buf);
        },
        [&](int64_t n) { return row_scale[n]; });
}

void sk_cpu_gemm_i4s(const float *x, int64_t M, int64_t K, const uint8_t *q, const float *scale, int64_t rows,
                     int64_t n0, int64_t n1, float *y, int64_t ldy) {
    const Ops &o = ops();
    gemm_decoded(
        x, M, K, n0, n1, y, ldy,
        [&](int64_t n, float *buf) {
            o.i4s(q + n * (K / 2), scale + n, rows, K, buf);
            return static_cast<const float *>(buf);
        },
        [](int64_t) { return 1.0f; });
}

void sk_cpu_gemm_rows(const float *x, int64_t M, int64_t K, int64_t n0, int64_t n1, float *y, int64_t ldy,
                      sk_cpu_row_decode decode, void *ctx) {
    gemm_decoded(
        x, M, K, n0, n1, y, ldy,
        [&](int64_t n, float *buf) {
            decode(ctx, n, buf);
            return static_cast<const float *>(buf);
        },
        [](int64_t) { return 1.0f; });
}

void sk_cpu_gemm_mxfp4(const float *x, int64_t M, int64_t K, const uint8_t *packed, const uint8_t *scales, int64_t n0,
                       int64_t n1, float *y, int64_t ldy) {
    // Permute x into the group-deinterleaved order once per call; every row reuses it.
    thread_local std::vector<float> xbuf;
    if (static_cast<int64_t>(xbuf.size()) < M * K) xbuf.resize(static_cast<size_t>(M * K));
    for (int64_t m = 0; m < M; ++m) {
        const float *xm = x + m * K;
        float *xd = xbuf.data() + m * K;
        for (int64_t g = 0; g < K / 32; ++g)
            for (int j = 0; j < 16; ++j) {
                xd[g * 32 + j] = xm[g * 32 + 2 * j];
                xd[g * 32 + 16 + j] = xm[g * 32 + 2 * j + 1];
            }
    }
    const Ops &o = ops();
    const int64_t pk = K / 2, sk = K / 32;
    const float *xd = xbuf.data();
    if (o.dot_mx) {
        // Groups of four rows share one decoded weight row; the rest use the fused path,
        // which is bit-identical to decoding (see dot_mx_avx512).
        thread_local std::vector<float> wbuf;
        if (static_cast<int64_t>(wbuf.size()) < K) wbuf.resize(static_cast<size_t>(K));
        for (int64_t n = n0; n < n1; ++n) {
            const uint8_t *pn = packed + n * pk, *sn = scales + n * sk;
            int64_t m = 0;
            if (M >= 4) {
                o.mx(pn, sn, K, wbuf.data());
                for (; m + 4 <= M; m += 4) {
                    float r[4];
                    o.dot4(xd + m * K, xd + (m + 1) * K, xd + (m + 2) * K, xd + (m + 3) * K, wbuf.data(), K, r);
                    for (int j = 0; j < 4; ++j) y[(m + j) * ldy + n] = r[j];
                }
            }
            for (; m < M; ++m) y[m * ldy + n] = o.dot_mx(xd + m * K, pn, sn, K);
        }
        return;
    }
    gemm_decoded(
        xd, M, K, n0, n1, y, ldy,
        [&](int64_t n, float *buf) {
            o.mx(packed + n * pk, scales + n * sk, K, buf);
            return static_cast<const float *>(buf);
        },
        [](int64_t) { return 1.0f; });
}

void sk_cpu_quant_fp8_rows(const uint16_t *W, int64_t K, int64_t n0, int64_t n1, uint8_t *q, float *row_scale) {
    for (int64_t n = n0; n < n1; ++n) {
        const uint16_t *w = W + n * K;
        float amax = 0.0f;
        for (int64_t k = 0; k < K; ++k) amax = std::fmax(amax, std::fabs(bf16_to_f32(w[k])));
        const float s = amax > 0.0f ? amax / 448.0f : 1.0f;
        row_scale[n] = s;
        const float inv = 1.0f / s;
        for (int64_t k = 0; k < K; ++k) q[n * K + k] = f32_to_e4m3(bf16_to_f32(w[k]) * inv);
    }
}

void sk_cpu_quant_i8_rows(const uint16_t *W, int64_t K, int64_t n0, int64_t n1, int8_t *q, float *row_scale) {
    for (int64_t n = n0; n < n1; ++n) {
        const uint16_t *w = W + n * K;
        float amax = 0.0f;
        for (int64_t k = 0; k < K; ++k) amax = std::fmax(amax, std::fabs(bf16_to_f32(w[k])));
        const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
        row_scale[n] = s;
        const float inv = 1.0f / s;
        for (int64_t k = 0; k < K; ++k) {
            float v = std::nearbyint(bf16_to_f32(w[k]) * inv);  // ties to even
            q[n * K + k] = static_cast<int8_t>(std::fmax(-127.0f, std::fmin(127.0f, v)));
        }
    }
}

}  // extern "C"
