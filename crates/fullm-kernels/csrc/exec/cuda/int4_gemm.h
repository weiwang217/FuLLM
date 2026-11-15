// SPDX-License-Identifier: Apache-2.0
// int4_gemm.h - the prefill GEMM of int4 experts (WFmt::INT4G) on Hopper: the same packed
// weights the decode tiles read (int4_group.h), unpacked to fp16 in registers as wgmma's A
// operand, the tokens' fp16 rows as its B operand from shared memory (TMA). So prefill and
// decode run one copy of the weights with the same numbers (fp16 activations, fp32 sums), and
// the experts need no 8-bit copies for prompt-sized steps. See int4_gemm.cu.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "../backend.h"

namespace fullm::int4_gemm {

// Built for sm_90a (wgmma) and the driver exports cuTensorMapEncodeTiled.
bool available();

// The routing lists of a moe_experts call (cuda_backend.cu's MoeLists): meta[0] used groups U,
// meta[1] passes (runs of at most 128 pairs of one group); group u is expert exps[u] over the
// pair-list rows start[u] .. start[u+1); pass q is group pass_u[q]'s rows from pass_b[q]; plist
// maps a pair-list row to its pair (token * k + j). maxp bounds the passes; pairs and slots (the
// expert slots, min(pairs, experts)) are the call's sizes, known on the host.
struct Lists {
    const int32_t *meta, *exps, *start, *pass_u, *pass_b, *plist;
    int maxp, pairs, slots;
};

// Shapes the GEMM takes: K and the weight rows multiples of 128 (gate/up: width and 2 * inter;
// down: inter and width).
bool shape_ok(int64_t width, int64_t inter);

// The routed pairs' rows of x ([tokens][width] of fp16 or bf16) in pair-list order as fp16
// (clamped, nearest even: the decode tiles' x), each 64-column chunk in the order the weights'
// k-slots take (int4_gemm.cu: chunk_pos): xs fp16 [pairs][width].
enum class XType { F16, BF16 };
void gather(__half *xs, const void *x, XType xt, const Lists &L, int k, int64_t width, int64_t pairs, cudaStream_t s);

// gate/up (each expert's w1, gate and up rows interleaved) of xs and its GLU (silu, or the
// clamped form with b1 / b2 as moe_experts has it) into s16: fp16 [pairs][inter], pair-list
// order, chunks in the same k order (the down GEMM's activations). rows: the rows xs's buffer
// holds (at least the pairs; the activation loads read whole 128-row blocks).
void gate_up(__half *s16, const __half *xs, int64_t rows, const Lists &L, const Backend::ExpertW *table,
             int64_t width, int64_t inter, bool silu, float b1, float b2, cudaStream_t s);

// down (w2) of s16 (rows: as above) into dn: bf16 [pairs][width], pair-list order.
void down(uint16_t *dn, const __half *s16, int64_t rows, const Lists &L, const Backend::ExpertW *table, int64_t width,
          int64_t inter, cudaStream_t s);


}  // namespace fullm::int4_gemm
