// SPDX-License-Identifier: Apache-2.0
// kernels/formats.cuh - the weight formats as the kernels read them (WT), the int4 unpacking,
// 16-byte decodes.
//
// Part of cuda_backend.cu's translation unit, included there after the CUDA and standard
// headers, in the order common, formats, gemv, convert, router, elementwise, copy, kimi, mla,
// attention, sampling, moe (each file uses only those before it). Kernels and their device
// helpers: no state.
#pragma once

namespace fullm {
namespace {

// ------------------------------------------------------------------ weight formats

template <int WF>
struct WT;
template <>
struct WT<0> {  // F32
    static constexpr int kElem = 4;
    __device__ static float at(const void *p, int64_t i) { return __ldg(static_cast<const float *>(p) + i); }
};
template <>
struct WT<1> {  // BF16
    static constexpr int kElem = 2;
    __device__ static float at(const void *p, int64_t i) {
        return bf2f(__ldg(static_cast<const unsigned short *>(p) + i));
    }
};
template <>
struct WT<2> {  // INT8
    static constexpr int kElem = 1;
    __device__ static float at(const void *p, int64_t i) {
        return static_cast<float>(__ldg(static_cast<const signed char *>(p) + i));
    }
};
template <>
struct WT<3> {  // FP8 E4M3
    static constexpr int kElem = 1;
    __device__ static float at(const void *p, int64_t i) {
        return fp8f(__ldg(static_cast<const unsigned char *>(p) + i));
    }
};
template <>
struct WT<4> {  // FP8B: E4M3 with block scales (only the tensor-core tiles apply them)
    static constexpr int kElem = 1;
    __device__ static float at(const void *p, int64_t i) {
        return fp8f(__ldg(static_cast<const unsigned char *>(p) + i));
    }
};
template <>
struct WT<5> {  // INT4G: two per byte (int4_group.h); only the tensor-core tiles read it
    static constexpr int kElem = 1;  // not used for it: tc_tile counts its bytes itself
};

// INT4G: one packed 32-bit word to its 8 values as four f16x2 registers, nibbles (0,4) (1,5)
// (2,6) (3,7) (int4_group.h places them so these are an MMA's A fragment): masked nibbles under
// fp16 1024 (exact), minus 1032; the high nibbles sit 4 bits up, so times 1/16, minus 72. Four
// LOP3s, a shift, four half2 ops. The magic-number dequantization of Marlin (IST-DASLab) and
// FasterTransformer (NVIDIA), both Apache-2.0 (NOTICE).
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

// Decode one 16-byte chunk; element 0 is the lowest address (little-endian).
template <int WF>
__device__ __forceinline__ void decode16(const uint4 r, float *w) {
    const uint32_t u[4] = {r.x, r.y, r.z, r.w};
    if (WF == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) w[i] = __uint_as_float(u[i]);
    } else if (WF == 1) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            w[2 * i] = __uint_as_float(u[i] << 16);
            w[2 * i + 1] = __uint_as_float(u[i] & 0xFFFF0000u);
        }
    } else if (WF == 2) {
        // exact int8 -> f32 in full-rate ops: 1.5 * 2^23 + b has b in the low mantissa bits
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int b = 0; b < 4; ++b)
                w[4 * i + b] = __int_as_float(0x4B400000 + static_cast<int8_t>((u[i] >> (8 * b)) & 0xFFu)) -
                               12582912.0f;
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int b = 0; b < 4; ++b) w[4 * i + b] = fp8f((u[i] >> (8 * b)) & 0xFFu);
    }
}

}  // namespace
}  // namespace fullm
