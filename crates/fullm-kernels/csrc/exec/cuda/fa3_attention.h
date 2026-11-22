// SPDX-License-Identifier: Apache-2.0
// fa3_attention.h - FlashAttention-3's Hopper forward kernel (vendored under fa3/, BSD-3, see
// fa3/README.md) as the step's attention: every sequence of a step in one launch (varlen: the
// decoding sequences' single tokens and the prompt chunks alike), causal with each sequence's
// queries aligned to the end of its context, GQA heads packed, the key/value cache read in place
// through the page table (no gather, no padding), bf16 in and out, fp32 accumulation; or, over an
// fp8 cache (e4m3 pages, one scale per kv head), e4m3 queries, keys and values on the fp8 tensor
// cores with the output still bf16 (head_dim 128 only). Few
// queries over a long context split the keys across CTAs: the scheduler's metadata kernel
// decides the splits per sequence on the device (so a captured graph adapts to the positions it
// replays with) and a combine kernel merges the partials. Needs the `cutlass` feature (it is
// CuTe code): without it available() is false and the callers keep the CUTLASS FMHA or their
// own kernels.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace fullm::fa3_ops {

bool available(int D);      // head_dim 64 or 128, sm90 only
bool available_fp8(int D);  // the e4m3 kernels: head_dim 128

// int32 count of the device scratch the kernel's scheduler needs for n sequences
int scheduler_ints(int n_seqs);

// The most key splits any sequence of a launch may get (1: no split, no partial buffers): the
// kernel's own heuristic from the longest context and the largest query count.
int max_splits(int n_seqs, int max_q, int max_k, int Hq, int Hkv, int D, int sm_count, bool fp8 = false);

struct Prefill {
    bool fp8;                     // q, k_cache and v_cache e4m3 (else bf16)
    const void *q;                // [total_q][Hq][D]: the sequences' rows, back to back
    const void *q_lo;             // fp8, non-null: q's rounding error (q - e4m3(q)) as e4m3, q's layout;
                                  // the HiLo kernel: S = q K^T + q_lo K^T, and P split likewise in
                                  // registers, O += P V + P_lo V (two GEMMs into one accumulator each)
    uint16_t *o;                  // [total_q][Hq][D] bf16
    float *lse;                   // [Hq][total_q] fp32 scratch (the log-sum-exp per row)
    const void *k_cache;          // [num_pages][page][Hkv][D]
    const void *v_cache;          // same layout
    const float *k_descale;       // fp8: [Hkv] (x = stored * descale), null = 1
    const float *v_descale;
    const int32_t *page_table;    // [slots][max_pages]
    const int32_t *kv_batch_idx;  // [n]: each sequence's slot (its row of page_table)
    const int32_t *cu_seqlens_q;  // [n + 1]: prefix sums of the sequences' query counts
    const int32_t *seqused_k;     // [n]: keys of each sequence (its position after its queries)
    int32_t *scheduler;           // scheduler_ints(n) int32 of device scratch
    int32_t *semaphore;           // non-null: the scheduler's metadata is already in `scheduler` (an
                                  // earlier launch of the same sequences computed it); this zeroed
                                  // int32 is the launch's own tile counter
    float *oaccum;                // splits > 1: [splits][Hq][total_q][D] fp32 partials
    float *lseaccum;              // splits > 1: [splits][Hq][total_q] fp32
    int splits;                   // max_splits(...) of this launch
    int n;                        // sequences
    int total_q;                  // rows of q and o
    int max_q, max_k;             // the largest query count and the longest context
    int Hq, Hkv, D, page, max_pages;
    int64_t num_pages;            // pages in the cache pool
    float scale;                  // softmax scale (1/sqrt(D))
    int sm_count;
};

// Enqueues the scheduler's metadata kernel, the attention kernel and, with splits, the combine.
void prefill(const Prefill &p, cudaStream_t s);

// MLA prefill, un-absorbed (fa3_mla.cu): q and k 192 wide (128 no-rope + 64 rope), v 128, as many
// kv heads as query heads, keys and values contiguous per sequence (expanded from the latent by
// the caller, cu_seqlens_k), causal end-aligned or (a chunk of context before the queries) every
// query over every key, bf16 in and out, no key splits.
bool available_mla();

struct Mla {
    const void *q;                // [total_q] rows of q_row elements, head h at h * 192
    const void *k;                // [total_k] rows of k_row elements, head h at h * 192
    const void *v;                // [total_k] rows of v_row elements, head h at h * 128
    uint16_t *o;                  // [total_q] rows of o_row elements, head h at h * 128
    int64_t q_row, k_row, v_row, o_row;
    float *lse;                   // [H][total_q] fp32 scratch
    const int32_t *cu_seqlens_q;  // [n + 1]
    const int32_t *cu_seqlens_k;  // [n + 1]
    int32_t *scheduler;           // scheduler_ints(n) int32 of device scratch
    int n, total_q, total_k, max_q, max_k, H;
    float scale;
    int sm_count;
    bool causal = true;           // false: no mask (keys all before the queries)
};

// Enqueues the scheduler's metadata kernel and the attention kernel.
void mla(const Mla &p, cudaStream_t s);

}  // namespace fullm::fa3_ops
