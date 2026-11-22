// SPDX-License-Identifier: Apache-2.0
// kernels/convert.cuh - activation conversions between fp32, bf16, fp16 and e4m3.
//
// Part of cuda_backend.cu's translation unit, included there after the CUDA and standard
// headers, in the order common, formats, gemv, convert, router, elementwise, copy, kimi, mla,
// attention, sampling, moe (each file uses only those before it). Kernels and their device
// helpers: no state.
#pragma once

namespace fullm {
namespace {

// fp32 activations to fp16 as the e4m3 tiles take them (f16x2: clamped, nearest even)
__global__ void x_to_f16_kernel(__half *__restrict__ dst, const float *src, int64_t n4) {
    pdl_wait();
    pdl_trigger();
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const float4 v = reinterpret_cast<const float4 *>(src)[i];
        reinterpret_cast<uint2 *>(dst)[i] = make_uint2(f16x2(v.x, v.y), f16x2(v.z, v.w));
    }
}

// fp32 activations to bf16 (round to nearest even, as the BX path rounds them), for the
// tensor-core GEMVs of larger steps: half the activation traffic, and the A fragments load whole
// dst [rows][w] fp32 = src [rows][ld] bf16's first w columns (n = rows * w)
__global__ void bf16_strided_to_f32_kernel(float *__restrict__ dst, const uint16_t *src, int64_t ld, int64_t w, int64_t n) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        dst[i] = __uint_as_float(static_cast<uint32_t>(src[(i / w) * ld + i % w]) << 16);
}

// e4m3 queries for FlashAttention-3 over an fp8 cache (scale 1, saturating), from the bf16 twin
// RoPE wrote or from fp32; lo > 0: each value's rounding error too, at dst + lo (e4m3_lo)
__global__ void bf16_to_e4m3_kernel(uint8_t *__restrict__ dst, const uint16_t *src, int64_t n, int64_t lo) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const float v = __uint_as_float(static_cast<uint32_t>(src[i]) << 16);
        const uint8_t h = f2e4m3(v);
        dst[i] = h;
        if (lo) dst[lo + i] = e4m3_lo(v, h);
    }
}
__global__ void f32_to_e4m3_kernel(uint8_t *__restrict__ dst, const float *src, int64_t n, int64_t lo) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const uint8_t h = f2e4m3(src[i]);
        dst[i] = h;
        if (lo) dst[lo + i] = e4m3_lo(src[i], h);
    }
}

__global__ void x_to_bf16_kernel(uint16_t *__restrict__ dst, const float *src, int64_t n4) {
    pdl_wait();
    pdl_trigger();
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const float4 v = reinterpret_cast<const float4 *>(src)[i];
        reinterpret_cast<uint2 *>(dst)[i] = make_uint2(bf16x2(v.x, v.y), bf16x2(v.z, v.w));
    }
}


__global__ void bf16_to_f32_kernel(float *dst, const uint16_t *__restrict__ src, int64_t n4) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const uint2 u = reinterpret_cast<const uint2 *>(src)[i];
        reinterpret_cast<float4 *>(dst)[i] = make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xFFFF0000u),
                                                         __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xFFFF0000u));
    }
}

}  // namespace
}  // namespace fullm
