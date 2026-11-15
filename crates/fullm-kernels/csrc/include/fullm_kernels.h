/* SPDX-License-Identifier: Apache-2.0 */
/*
 * fullm_kernels.h - the C ABI between the Rust engine and the C++/CUDA kernels.
 *
 * CONVENTIONS
 *   - Every weight matrix is row-major [N][K]: N output rows, K inputs, no bias.
 *     y[m][n] = sum_k x[m][k] * W[n][k]. This is the checkpoint's own layout
 *     (torch Linear.weight), so nothing is ever transposed at load.
 *   - Activations are fp32 on the CPU path. The kernels never allocate for the caller's
 *     data and never spawn threads: the Rust side splits work over [n0, n1) row ranges,
 *     so threading, NUMA pinning and scheduling live in one place.
 *   - BATCH INVARIANCE. Each output element is produced by the same dot-product routine
 *     regardless of M, of the [n0, n1) split, or of what else is in the batch. A token's
 *     result therefore does not depend on which other tokens share its step, which is
 *     what lets the engine promise identical output for a sequence whether it runs alone,
 *     in a batch, or with its prefill chunked. (Different ISAs may differ in the last bit;
 *     one machine is always self-consistent.)
 *
 * WEIGHT FORMATS
 *   bf16    uint16 per element, the top half of an f32. Widening is exact.
 *   fp8     OCP E4M3 (FN: no inf, 0x7f/0xff NaN) with ONE f32 scale per output row:
 *           W[n][k] = e4m3(q[n][k]) * scale[n]. The trunk's GPU format.
 *   int8    symmetric int8 with ONE f32 scale per output row: W[n][k] = q[n][k] * scale[n].
 *           Same bytes as fp8 but ~4x lower error on Gaussian-like rows (fp8 keeps 3
 *           mantissa bits whatever the scale; int8 resolves amax/127 uniformly).
 *   mxfp4   OCP MX FP4: packed [N][K/2] bytes, LOW NIBBLE = EVEN element, plus E8M0
 *           scales [N][K/32]: W = E2M1[nib] * 2^(scale-127). Kimi K3 ships its routed
 *           experts this way and they are never widened in memory.
 */
#ifndef FULLM_KERNELS_H
#define FULLM_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* "avx512", "avx2", "neon" or "scalar": the path the CPU kernels dispatch to. */
const char *sk_cpu_isa(void);

void sk_cpu_gemm_f32(const float *x, int64_t M, int64_t K, const float *W,
                     int64_t n0, int64_t n1, float *y, int64_t ldy);

void sk_cpu_gemm_bf16(const float *x, int64_t M, int64_t K, const uint16_t *W,
                      int64_t n0, int64_t n1, float *y, int64_t ldy);

void sk_cpu_gemm_fp8(const float *x, int64_t M, int64_t K, const uint8_t *W,
                     const float *row_scale, int64_t n0, int64_t n1, float *y, int64_t ldy);

void sk_cpu_gemm_i8(const float *x, int64_t M, int64_t K, const int8_t *W,
                    const float *row_scale, int64_t n0, int64_t n1, float *y, int64_t ldy);

/* K must be a multiple of 32 (one MX group). */
// int4 weights in the kInt4CpuSimd packing (int4_group.h), group scales [K / 128][rows]:
// y[m][n] = dot(x[m], dequantised row n) for rows [n0, n1). K % 64 == 0.
void sk_cpu_gemm_i4s(const float *x, int64_t M, int64_t K, const uint8_t *q, const float *scale,
                     int64_t rows, int64_t n0, int64_t n1, float *y, int64_t ldy);
// Any format: decode(ctx, n, out) writes row n as K floats (scales applied); the same SIMD dots.
typedef void (*sk_cpu_row_decode)(void *ctx, int64_t n, float *out);
void sk_cpu_gemm_rows(const float *x, int64_t M, int64_t K, int64_t n0, int64_t n1, float *y,
                      int64_t ldy, sk_cpu_row_decode decode, void *ctx);
void sk_cpu_gemm_mxfp4(const float *x, int64_t M, int64_t K, const uint8_t *packed,
                       const uint8_t *scales, int64_t n0, int64_t n1, float *y, int64_t ldy);

/* Quantise rows [n0, n1) of a bf16 matrix to E4M3 with a per-row absmax scale
 * (scale = amax / 448). Round to nearest even; saturates, never produces NaN. */
void sk_cpu_quant_fp8_rows(const uint16_t *W, int64_t K, int64_t n0, int64_t n1,
                           uint8_t *q, float *row_scale);

/* Quantise rows [n0, n1) of a bf16 matrix to symmetric int8, scale = amax / 127. */
void sk_cpu_quant_i8_rows(const uint16_t *W, int64_t K, int64_t n0, int64_t n1,
                          int8_t *q, float *row_scale);

/* E4M3 -> f32 for all 256 codes (NaN codes decode to 0). */
const float *sk_fp8_e4m3_table(void);

#ifdef __cplusplus
}
#endif

#endif /* FULLM_KERNELS_H */
