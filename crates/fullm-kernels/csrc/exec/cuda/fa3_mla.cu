// SPDX-License-Identifier: Apache-2.0
// fa3_mla.cu - see fa3_attention.h (fa3_ops::mla): FlashAttention-3's Hopper forward at head_dim
// 192 / head_dim_v 128 (the shapes of DeepSeek-style MLA un-absorbed: 128 no-rope + 64 rope
// channels for q and k, 128 for v), varlen over contiguous keys and values, causal or not, bf16, no
// splits, no GQA packing (MLA expands a key and a value per query head). Its own translation unit
// so the build compiles it beside fa3_attention.cu; the varlen scheduler's metadata kernel
// (prepare_varlen_num_blocks) is fa3_attention.cu's. Same flags and disables as that unit.
#include "fa3_attention.h"

#ifdef FULLM_CUTLASS

#define FLASHATTENTION_DISABLE_LOCAL
#define FLASHATTENTION_DISABLE_SOFTCAP
#define FLASHATTENTION_DISABLE_APPENDKV
#define FLASHATTENTION_DISABLE_SM8x
#define FLASHATTENTION_DISABLE_CLUSTER
#define FLASHATTENTION_DISABLE_FP16
#define FLASHATTENTION_DISABLE_BACKWARD

#include "fa3/flash_fwd_launch_template.h"

#include <algorithm>

namespace fullm::fa3_ops {

bool available_mla() { return true; }

namespace {
constexpr int round_up(int x, int m) { return (x + m - 1) / m * m; }
}  // namespace

void mla(const Mla &a, cudaStream_t s) {
    Flash_fwd_params p{};
    p.is_bf16 = true;
    p.arch = 90;
    p.num_sm = a.sm_count;
    p.b = a.n;
    p.seqlen_q = a.max_q;
    p.seqlen_k = a.max_k;
    p.total_q = a.total_q;
    p.total_k = a.total_k;
    p.h = p.h_k = a.H;
    p.d = p.d_rounded = 192;
    p.dv = p.dv_rounded = 128;
    p.seqlen_q_rounded = round_up(a.max_q, 128);
    p.seqlen_k_rounded = round_up(a.max_k, 128);
    // rows of the caller's strides, heads 192 (q, k) or 128 (v, o) apart; batch strides 0 (varlen)
    p.q_ptr = const_cast<void *>(a.q);
    p.q_row_stride = a.q_row;
    p.q_head_stride = 192;
    p.k_ptr = const_cast<void *>(a.k);
    p.k_row_stride = a.k_row;
    p.k_head_stride = 192;
    p.v_ptr = const_cast<void *>(a.v);
    p.v_row_stride = a.v_row;
    p.v_head_stride = 128;
    p.v_dim_stride = 1;
    p.o_ptr = a.o;
    p.o_row_stride = a.o_row;
    p.o_head_stride = 128;
    p.cu_seqlens_q = const_cast<int32_t *>(a.cu_seqlens_q);
    p.cu_seqlens_k = const_cast<int32_t *>(a.cu_seqlens_k);
    p.softmax_lse_ptr = a.lse;
    p.scale_softmax = a.scale;
    p.p_dropout = 1.0f;
    p.rp_dropout = 1.0f;
    p.p_dropout_in_uint8_t = 255;
    // causal: query i of a sequence sees keys up to keys - queries + i (end-aligned)
    p.is_causal = a.causal;
    p.is_local = false;
    p.window_size_left = a.causal ? a.max_k - 1 : -1;
    p.window_size_right = a.causal ? 0 : -1;
    p.attention_chunk = 0;
    p.num_splits = 1;
    p.pagedkv_tma = false;
    p.pack_gqa = false;
    p.prepare_varlen_pdl = a.n <= 992;  // PREPARE_VARLEN_MAX_BATCHES_1CTA
    p.varlen_sort_batches = true;
    p.head_swizzle = a.causal;
    p.skip_scheduler_metadata_computation = false;
    const int br = round_up(a.n, 4);
    p.num_splits_dynamic_ptr = a.scheduler;
    p.num_m_blocks_ptr = a.scheduler + br;
    p.varlen_batch_idx_ptr = a.scheduler + 2 * br;
    p.num_nheads_in_l2_ptr = a.scheduler + 3 * br;
    p.tile_count_semaphore = a.scheduler + 4 * br;  // zeroed by the metadata kernel
    p.tile_count_semaphore_offset = 4 * br;
    // <Arch, T, kHeadDim, kHeadDimV, Split, PagedKVNonTMA, Has_softcap, PackGQA>
    run_mha_fwd_<90, cutlass::bfloat16_t, 192, 128, false, false, false, false>(p, s);
}

}  // namespace fullm::fa3_ops

#else  // no CUTLASS: stubs

namespace fullm::fa3_ops {
bool available_mla() { return false; }
void mla(const Mla &, cudaStream_t) {}
}  // namespace fullm::fa3_ops

#endif
