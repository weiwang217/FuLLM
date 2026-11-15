// SPDX-License-Identifier: Apache-2.0
// cutlass_ops.h - CUTLASS (header-only, sm90) kernels behind plain functions: the grouped bf16
// GEMM of the prefill experts (TMA + wgmma, warp-specialised). Built with the `cutlass` feature
// (FULLM_CUTLASS_DIR); without it available() is false and the callers keep their own kernels.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "../backend.h"

namespace fullm::cutlass_ops {

bool available();

// Bytes of the device argument block for `groups` groups (problem sizes, pointers, strides).
size_t args_bytes(int groups);
// Workspace bytes the grouped GEMM needs for `groups` groups (the tile scheduler's); which: 0 =
// gate/up, 1 = down (each may run a different kernel variant, see cutlass_ops.cu).
size_t workspace_bytes(int groups, bool out_f32, int which);

// Fills the argument block on the device (one small kernel on `s`) for the routed experts of
// moe_experts: group u < meta[0] is expert exps[u] over its pairs start[u] .. start[u+1), whose
// rows of A (bf16 [pairs][K], in pair-list order) and of D ([pairs][N], same order) are
// contiguous; B is the expert's w1 (gate/up, rows interleaved) when w1, else its w2. Groups
// past meta[0] get M = 0. meta: exps = meta + 2, start = exps + groups (MoeLists' layout).
void fill_args(void *args, int groups, const int32_t *meta, const Backend::ExpertW *table, bool w1,
               const uint16_t *A, int64_t K, int64_t N, void *D, bool out_f32, cudaStream_t s);

// D[g] = A[g] B[g]^T for every group (bf16 operands, fp32 accumulation, bf16 or fp32 output).
void run(const void *args, int groups, bool out_f32, int which, void *workspace, size_t workspace_bytes,
         cudaStream_t s);

// ---- fp8 experts (WFmt::FP8B): the sm90 blockwise-scaled grouped GEMM of CUTLASS example 68,
// e4m3 A and B, A scaled per row per 128 of K, B per 128 x 128 block, fp32 accumulation with the
// scales applied per K block, bf16 D; 128 x 128 x 128 tiles, the cooperative schedule.
bool fp8_available();
size_t fp8_args_bytes(int groups);
size_t fp8_workspace_bytes(int groups);
// As fill_args, for A e4m3 [pairs][K] in pair-list order with its scales sfa: group u's are the
// kernel's packed [K/128][M_u] at sfa + start[u] * K/128. B is the expert's w1 or w2 and its
// block scales Weight::scale (fp8_block.h's layout is the kernel's). D: bf16 [pairs][N].
// K and N multiples of 128.
void fill_args_fp8(void *args, int groups, const int32_t *meta, const Backend::ExpertW *table, bool w1, const uint8_t *A,
                   const float *sfa, int64_t K, int64_t N, uint16_t *D, cudaStream_t s);
void run_fp8(const void *args, int groups, void *workspace, size_t workspace_bytes, cudaStream_t s);

// Prefill attention: the CUTLASS Hopper FMHA (FlashAttention-3-class, TMA + wgmma) on one
// sequence chunk of n_q queries at the end of n_k keys, causal (end-aligned), softmax scale
// 1/sqrt(D). Q and O: [n_q][Hq][D] bf16; K and V: [n_k][Hkv][D] bf16 with contiguous rows;
// lse: [Hq][n_q] fp32 scratch. Hq % Hkv == 0; D 64 or 128 (fmha_available says which).
bool fmha_available(int D);
size_t fmha_workspace_bytes(int Hq, int Hkv, int n_q, int n_k, int D);
void fmha(const uint16_t *Q, const uint16_t *K, const uint16_t *V, uint16_t *O, float *lse, int Hq, int Hkv, int n_q,
          int n_k, int D, void *workspace, size_t workspace_bytes, cudaStream_t s);

}  // namespace fullm::cutlass_ops
