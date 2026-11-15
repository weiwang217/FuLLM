// SPDX-License-Identifier: Apache-2.0
// fa3_attention.cu - see fa3_attention.h. Its own translation unit, built by build.rs with
// --use_fast_math (exp2f -> ex2.approx in the softmax, as FlashAttention's own build; the rest of
// the backend keeps IEEE math), -DNDEBUG and the PDL define; without the cutlass feature it
// compiles to stubs. The parameter block mirrors what hopper/flash_api.cpp (the PyTorch binding,
// not vendored) sets up for a varlen, paged, causal forward with the scheduler metadata kernel
// and, for few queries over long contexts, the split-KV kernel and its combine.
#include "fa3_attention.h"

#ifdef FULLM_CUTLASS

// The variants this build never runs: each is a full kernel instantiation and minutes of compile.
#define FLASHATTENTION_DISABLE_LOCAL
#define FLASHATTENTION_DISABLE_SOFTCAP
#define FLASHATTENTION_DISABLE_APPENDKV
#define FLASHATTENTION_DISABLE_SM8x
#define FLASHATTENTION_DISABLE_CLUSTER
#define FLASHATTENTION_DISABLE_FP16
#define FLASHATTENTION_DISABLE_BACKWARD

#include "fa3/flash_fwd_launch_template.h"
#include "fa3/flash_fwd_combine_launch_template.h"
#include "fa3/flash_prepare_scheduler.cu"
#include "fa3/heuristics.h"
#include "fa3/tile_size.h"

#include <algorithm>

namespace fullm::fa3_ops {

bool available(int D) { return D == 64 || D == 128; }
bool available_fp8(int D) { return D == 128; }

namespace {
constexpr int round_up(int x, int m) { return (x + m - 1) / m * m; }
// fp8 launches of decoding rows only (packed query rows per kv head <= 64) take the 64 x 96 tile
bool decode_tile(bool fp8, int max_q, int Hq, int Hkv) { return fp8 && max_q * (Hq / Hkv) <= 64; }
}  // namespace

// flash_api.cpp: num_splits_dynamic, num_m_blocks, varlen_batch_idx (the batches sorted), the
// heads-in-L2 count (head swizzle, causal), then the tile counter: four vectors of b rounded to
// 4, plus the semaphore
int scheduler_ints(int n_seqs) { return 1 + 4 * round_up(n_seqs, 4); }

// flash_api.cpp's get_num_splits for a varlen, causal, paged (non-TMA) forward: with the
// dynamic splits the batch counts as one sequence, so this is the per-sequence upper bound the
// metadata kernel then lowers per sequence.
int max_splits(int /*n_seqs*/, int max_q, int max_k, int Hq, int Hkv, int D, int sm_count, bool fp8) {
    const int elem = fp8 ? 1 : 2;
    const auto tile = tile_size_fwd_sm90(D, D, /*is_causal*/ true, /*is_local*/ false, /*element_size*/ elem,
                                         /*v_colmajor*/ false, /*paged_kv_non_TMA*/ true, /*softcap*/ false,
                                         decode_tile(fp8, max_q, Hq, Hkv));
    const int kBlockM = std::get<0>(tile), kBlockN = std::get<1>(tile);
    const int seqlen_q_packgqa = max_q * (Hq / Hkv);
    const int num_n_blocks = (max_k + kBlockN - 1) / kBlockN;
    const int num_m_blocks = (seqlen_q_packgqa + kBlockM - 1) / kBlockM;
    const int size_one_kv_head = max_k * (D + D) * elem;
    const int total_mblocks = 1 * Hkv * num_m_blocks;
    return num_splits_heuristic(total_mblocks, sm_count, num_n_blocks, num_m_blocks, size_one_kv_head,
                                /*is_causal_or_local*/ true, /*max_splits*/ 128);
}

void prefill(const Prefill &a, cudaStream_t s) {
    Flash_fwd_params p{};
    p.is_bf16 = !a.fp8;
    p.is_e4m3 = a.fp8;
    p.arch = 90;
    p.num_sm = a.sm_count;
    p.b = a.n;
    p.seqlen_q = a.max_q;
    p.seqlen_k = a.max_k;
    p.total_q = a.total_q;
    p.h = a.Hq;
    p.h_k = a.Hkv;
    p.d = a.D;
    p.dv = a.D;
    p.d_rounded = a.D;  // 64 and 128 are their own rounded sizes
    p.dv_rounded = a.D;
    p.seqlen_q_rounded = round_up(a.max_q, 128);
    p.seqlen_k_rounded = round_up(a.max_k, 128);
    // q and o: [total_q][Hq][D]; the batch strides stay 0 with cu_seqlens_q
    p.q_ptr = const_cast<void *>(a.q);
    p.q_row_stride = static_cast<int64_t>(a.Hq) * a.D;
    p.q_head_stride = a.D;
    p.o_ptr = a.o;
    p.o_row_stride = static_cast<int64_t>(a.Hq) * a.D;
    p.o_head_stride = a.D;
    // the cache: [num_pages][page][Hkv][D], the page stride is the "batch" stride of a paged K/V
    p.k_ptr = const_cast<void *>(a.k_cache);
    p.v_ptr = const_cast<void *>(a.v_cache);
    // fp8: one descale per kv head, the same for every sequence (batch stride 0); queries at 1
    p.k_descale_ptr = const_cast<float *>(a.k_descale);
    p.v_descale_ptr = const_cast<float *>(a.v_descale);
    p.k_descale_batch_stride = p.v_descale_batch_stride = 0;
    p.k_descale_head_stride = p.v_descale_head_stride = 1;
    p.k_row_stride = p.v_row_stride = static_cast<int64_t>(a.Hkv) * a.D;
    p.k_head_stride = p.v_head_stride = a.D;
    p.v_dim_stride = 1;
    p.k_batch_stride = p.v_batch_stride = static_cast<int64_t>(a.page) * a.Hkv * a.D;
    p.page_table = const_cast<int32_t *>(a.page_table);
    p.page_table_batch_stride = a.max_pages;
    p.page_size = a.page;
    p.num_pages = static_cast<int>(std::min<int64_t>(a.num_pages, INT32_MAX));
    p.kv_batch_idx = const_cast<int32_t *>(a.kv_batch_idx);
    p.cu_seqlens_q = const_cast<int32_t *>(a.cu_seqlens_q);
    p.seqused_k = const_cast<int32_t *>(a.seqused_k);
    p.softmax_lse_ptr = a.lse;
    p.scale_softmax = a.scale;
    p.p_dropout = 1.0f;  // no dropout
    p.rp_dropout = 1.0f;
    p.p_dropout_in_uint8_t = 255;
    // causal: query i of a sequence sees keys up to seqused_k - count + i (end-aligned)
    p.is_causal = true;
    p.is_local = false;
    p.window_size_left = a.max_k - 1;
    p.window_size_right = 0;
    p.attention_chunk = 0;
    p.num_splits = std::max(1, a.splits);
    p.pagedkv_tma = false;  // the page is 16 tokens, the kernel's key tile 128: TMA paging needs page % 128 == 0
    p.pack_gqa = true;      // required with non-TMA paging and with splits (what flash_api picks for varlen)
    p.prepare_varlen_pdl = a.n <= 992;  // PREPARE_VARLEN_MAX_BATCHES_1CTA
    p.varlen_sort_batches = true;       // !is_local
    p.head_swizzle = true;              // is_causal
    p.skip_scheduler_metadata_computation = a.semaphore != nullptr;
    const int br = round_up(a.n, 4);
    p.num_splits_dynamic_ptr = a.scheduler;
    p.num_m_blocks_ptr = a.scheduler + br;
    p.varlen_batch_idx_ptr = a.scheduler + 2 * br;
    p.num_nheads_in_l2_ptr = a.scheduler + 3 * br;
    p.tile_count_semaphore = a.semaphore ? a.semaphore : a.scheduler + 4 * br;  // zeroed by the metadata kernel, or the caller
    p.tile_count_semaphore_offset = 4 * br;
    if (p.num_splits > 1) {
        // the partials, [splits][Hq][total_q][D] and [splits][Hq][total_q] (flash_api's varlen
        // layout); sequences the metadata kernel leaves at one split write o directly
        p.oaccum_ptr = a.oaccum;
        p.softmax_lseaccum_ptr = a.lseaccum;
        p.oaccum_split_stride = static_cast<int64_t>(a.Hq) * a.total_q * a.D;
        p.oaccum_head_stride = static_cast<int64_t>(a.total_q) * a.D;
        p.oaccum_row_stride = a.D;
        p.lseaccum_split_stride = static_cast<int64_t>(a.Hq) * a.total_q;
        p.lseaccum_head_stride = a.total_q;
        p.is_fp32 = false;
    }

    // run_mha_fwd_ launches the scheduler's metadata kernel itself (skip_scheduler_metadata_computation
    // false), then the attention kernel with PDL after it
    // <Arch, T, kHeadDim, kHeadDimV, Split, PagedKVNonTMA, Has_softcap, PackGQA>
    if (a.fp8) {  // head_dim 128 (available_fp8); the output and the partials' combine stay bf16
        // <..., Split, PagedKVNonTMA, Has_softcap, PackGQA, OneMmaWG>
        const bool dec = decode_tile(true, a.max_q, a.Hq, a.Hkv);
        if (p.num_splits > 1) {
            if (dec) run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, true, true, false, true, true>(p, s);
            else run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, true, true, false, true>(p, s);
            run_mha_fwd_combine_<cutlass::bfloat16_t, float, 128>(p, s, /*enable_pdl*/ true);
        } else {
            if (dec) run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, false, true, false, true, true>(p, s);
            else run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, false, true, false, true>(p, s);
        }
        return;
    }
    if (p.num_splits > 1) {
        if (a.D == 64) run_mha_fwd_<90, cutlass::bfloat16_t, 64, 64, true, true, false, true>(p, s);
        else run_mha_fwd_<90, cutlass::bfloat16_t, 128, 128, true, true, false, true>(p, s);
        if (a.D == 64) run_mha_fwd_combine_<cutlass::bfloat16_t, float, 64>(p, s, /*enable_pdl*/ true);
        else run_mha_fwd_combine_<cutlass::bfloat16_t, float, 128>(p, s, /*enable_pdl*/ true);
    } else {
        if (a.D == 64) run_mha_fwd_<90, cutlass::bfloat16_t, 64, 64, false, true, false, true>(p, s);
        else run_mha_fwd_<90, cutlass::bfloat16_t, 128, 128, false, true, false, true>(p, s);
    }
}

}  // namespace fullm::fa3_ops

#else  // no CUTLASS: stubs

namespace fullm::fa3_ops {
bool available(int) { return false; }
bool available_fp8(int) { return false; }
int scheduler_ints(int) { return 0; }
int max_splits(int, int, int, int, int, int, int, bool) { return 1; }
void prefill(const Prefill &, cudaStream_t) {}
}  // namespace fullm::fa3_ops

#endif
