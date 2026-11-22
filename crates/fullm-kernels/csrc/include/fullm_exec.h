/* SPDX-License-Identifier: Apache-2.0 */
/*
 * fullm_exec.h - the executor's C ABI: the one boundary between the Rust control plane
 * and the C++/CUDA data plane.
 *
 * LIFECYCLE
 *   sk_exec_create_k3 -> sk_exec_add_matrix / sk_exec_add_vector (any order, streamed one
 *   layer at a time so host memory never holds the whole trunk) -> sk_exec_set_expert ->
 *   sk_exec_finalize -> sk_exec_step ... -> sk_exec_destroy.
 *
 * OWNERSHIP
 *   Matrices and vectors are COPIED (uploaded) during add_*; the caller may free them on
 *   return. The embedding table and the routed experts are NOT copied: they stay in host
 *   memory owned by the caller (mapped checkpoint or DRAM arena) and must outlive the
 *   executor.
 *
 *   The executor owns the physical cache pools (latent pages, recurrent-state slots); the
 *   caller owns their ALLOCATION (which slot and which pages a sequence uses) and passes
 *   page tables with every step.
 *
 * ERRORS
 *   Functions return 0 on success. On failure they write a message into err (when given)
 *   and return nonzero. C++ exceptions never cross this boundary.
 */
#ifndef FULLM_EXEC_H
#define FULLM_EXEC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sk_exec sk_exec;

enum { SK_BACKEND_CPU = 0, SK_BACKEND_CUDA = 1, SK_BACKEND_TORCH = 2 /* CPU, GEMMs via libtorch */ };
enum {
    SK_DT_F32 = 0,
    SK_DT_BF16 = 1,
    SK_DT_FP8 = 2,     /* KV cache only: e4m3 with a scale per token and kv head (kv_fp8.h) */
    SK_DT_FP8E4M3 = 3, /* KV cache only: plain e4m3 pages, FlashAttention-3's fp8 layout */
    SK_DT_KV4 = 4      /* KV cache only: int4 pages (kv_int4.h) */
};
/* SK_FMT_FP8B: e4m3 with fp32 scales per 128 x 128 block; resident experts only, passed as bf16
 * and quantised by the executor (after it fixes the stored layout). SK_FMT_INT4G: int4 with an
 * fp32 scale per row per 128 columns, the same way (int4_group.h) */
enum { SK_FMT_F32 = 0, SK_FMT_BF16 = 1, SK_FMT_INT8 = 2, SK_FMT_FP8 = 3, SK_FMT_FP8B = 4, SK_FMT_INT4G = 5 };

/* What a backend can do (backend.h's Caps and Policy), for the planner before it builds an
 * executor: sk_backend_caps creates the backend, reads them and destroys it. */
typedef struct {
    char name[16];                  /* the backend's name */
    int32_t memory;                 /* 0 host memory, 1 a device's own memory, 2 unified */
    int32_t graphs;                 /* decode steps replay as graphs */
    int32_t async_copy;             /* host-tier copies overlap compute */
    uint32_t dense_formats;         /* 1 << SK_FMT_* per weight format the dense matrices may take */
    uint32_t expert_formats;        /* ... per format the resident experts may take */
    uint32_t kv_formats;            /* 1 << SK_DT_* per KV cache storage */
    int32_t kv8_page_multiple;      /* fp8 KV: pages hold a multiple of this many positions */
    uint32_t families;              /* model families' ops: 1 GQA, 2 MLA, 4 KDA, 8 Kimi (backend.h FAM_*) */
    int64_t expert_tile_max_rows;   /* policy: int4 experts' decode tiles up to this many rows */
    int64_t expert_gemm_min_rows;   /* policy: int4 experts' prefill GEMM from this many rows */
} sk_caps;

typedef struct {
    int32_t backend;           /* SK_BACKEND_*: the alias used when backend_name is empty */
    int32_t device;            /* CUDA device ordinal */
    int32_t n_slots;           /* recurrent-state slots (concurrent sequences) */
    int32_t n_pages;           /* latent-cache pages per MLA layer */
    int32_t page;              /* positions per page */
    int32_t max_pages_per_seq; /* columns of the page table */
    int32_t max_tokens;        /* tokens per step (sizes the workspace) */
    int32_t max_rows;          /* logits rows per step */
    int32_t kv_dtype;          /* SK_DT_*: attention / latent cache storage */
    int32_t state_dtype;       /* SK_DT_*: KDA recurrent-state storage */
    int32_t cpu_threads;       /* routed-expert pool and CPU backend; 0 = all cores */
    int32_t no_graphs;         /* 1 = never replay CUDA graphs (parity tests, debugging) */
    int32_t bf16_activations;  /* 1 = small-M GEMVs round x to bf16 (as vLLM); 0 = exact fp32 x */
    char backend_name[32];     /* the backend by name (sk_backend_list), e.g. "cpu", "cuda" */
} sk_exec_opts;

typedef struct {
    int32_t hidden, n_layers, vocab;
    float rms_eps;
    int32_t kda_heads, kda_head_dim, conv_k;
    float gate_lb;
    int32_t n_heads, q_lora, kv_lora, qk_nope, qk_rope, v_head, mla_out_gate;
    int32_t n_experts, topk, n_shared, latent, moe_inter;
    float routed_scale;
    int32_t moe_renorm, latent_norm, first_dense, dense_inter, attn_res_block;
    float situ_b1, situ_b2;
    /* Kimi family switches (Kimi K3 = 1,0,0,1,0,0; Kimi Linear = 0,1,1,0,1,1). q_lora == 0
     * means a direct q_proj; mla_out_gate == 0 means no MLA output gate. */
    int32_t attn_res;         /* Block Attention Residuals, else plain pre-norm residuals */
    int32_t decay_softplus;   /* KDA decay -exp(A)*softplus(z), else lb*sigmoid(exp(A)*z) */
    int32_t kda_gate_lowrank; /* KDA output gate g_b(g_a(x)), else full-rank g */
    int32_t latent_moe;       /* routed experts in a latent space (down/up projections) */
    int32_t act_silu;         /* SiLU-GLU, else SiTU-GLU */
    int32_t expert_bf16;      /* bf16 routed experts, else MXFP4 */
    int32_t router_biased;    /* MoE weights from score + bias (HF Kimi Linear), else score */
    float kv_norm_eps;        /* eps of the MLA kv_a_layernorm */
    const uint8_t *layer_is_mla; /* [n_layers], 1 = Gated MLA, 0 = KDA */
} sk_k3_dims;

/* Creates the executor and its backend. Fails if the device cannot be opened. */
sk_exec *sk_exec_create_k3(const sk_k3_dims *dims, const sk_exec_opts *opts, char *err,
                           int err_len);

/* A dense decoder (Llama / Qwen3 family): pre-norm residuals; grouped-query attention with
 * RoPE (rotate-half), after a per-head RMSNorm of q and k when qk_norm (Qwen3); SwiGLU MLP.
 * Tensors, by name:
 *   "embed" [vocab][hidden] (host memory, bf16 or f32, read in place), "lm_head" [vocab][hidden],
 *   vector "final_norm"; per layer "L{i}.": vectors "in_norm", "post_norm" (and "q_norm",
 *   "k_norm" [head_dim] when qk_norm), matrices "q" [heads*head_dim][hidden], "k" and "v"
 *   [kv_heads*head_dim][hidden], "o" [hidden][heads*head_dim], "gate" and "up"
 *   [inter][hidden], "down" [hidden][inter].
 * With n_experts > 0 every layer's MLP is a routed MoE instead (Qwen3-MoE): a router matrix
 * "router" [n_experts][hidden] per layer, softmax over the experts, the top-k weights
 * (renormalised to sum 1 when norm_topk), SwiGLU experts of width moe_inter; the experts are
 * given with sk_exec_reserve_gpu_experts / sk_exec_set_gpu_expert and live in backend memory.
 * The KV cache is paged like the MLA latent cache (opts page, n_pages, max_pages_per_seq). */
typedef struct {
    int32_t n_layers, hidden, n_heads, n_kv_heads, head_dim, inter, vocab;
    float rms_eps;
    int32_t qk_norm;
    const float *inv_freq; /* [head_dim / 2] RoPE inverse frequencies, scaling applied (copied) */
    int32_t n_experts, topk, moe_inter, norm_topk;
} sk_dense_dims;

sk_exec *sk_exec_create_dense(const sk_dense_dims *dims, const sk_exec_opts *opts, char *err,
                              int err_len);

/* Inkling 2 (looped): thinkingmachines/Inkling-2-Mini's text decoder (ink2_exec.cpp; the model is
 * stated in crates/fullm/src/models/inkling2). Layers loop_start..loop_end run `passes` times with
 * shared weights; every layer visit has its own key / value pools and convolution histories.
 * Tensors, by name ("L{i}." per layer; matrices bf16 or f32, [out][in]):
 *   matrices  "embed" [vocab][hidden] (host memory, read in place), "lm_head" [out_vocab][hidden];
 *             per layer "q" [hq*d][hidden], "k" and "v" [hk*d][hidden], "o" [hidden][hq*d], "og"
 *             (the output gate) [hq*d][hidden], "sink" [hq][hidden], "rel" [hq*d_rel][hidden],
 *             "router" [n_experts][hidden]; full layers "k_wp" and "v_wp" [hk*conv_k][hidden];
 *             loop layers "cq_" / "ck_" / "cv_" / "cm_" + "down" [gate_rank][2*hidden], "forget"
 *             and "gain" [hidden][gate_rank] (the message gates of the query, key, value and MLP
 *             inputs)
 *   vectors   "embed_norm", "final_norm", "out_query" [hidden], "msg_queries" [loop layers][4]
 *             [hidden] (query, key, value, mlp); per layer "attn_query" ([hidden], full layers
 *             [3][hidden]; not layer 0), "mlp_query", "post_norm", "q_scale" [hq*d], "rel_scale"
 *             [hq*d_rel], "rel_proj" [d_rel][extent], "attn_conv" and "mlp_conv" [hidden][conv_k],
 *             "router_bias" [n_experts]; sliding layers "in_norm", "k_conv" and "v_conv" [hk*d]
 *             [conv_k]; full layers "q_norm", "k_norm", "v_norm", "k_static" and "v_static"
 *             [conv_k][hk*d]
 * and each layer's experts with sk_exec_set_ink2_experts. Host backends only so far; the cache
 * is f32, or half (kv_dtype SK_DT_BF16: 16 bits a value) where the steps run on a GPU. */
typedef struct {
    int32_t hidden, n_layers, vocab, out_vocab;
    float rms_eps;
    int32_t full_hq, full_hk, full_d; /* heads, kv heads, head width of the full layers */
    int32_t swa_hq, swa_hk, swa_d;    /* ... of the sliding-window layers */
    int32_t window, d_rel, rel_extent;
    float log_n_floor, log_alpha, qk_clip, rel_clip;
    int32_t conv_k, block;
    int32_t n_experts, topk, moe_inter;
    float gate_softcap, up_softcap, route_scale, logit_softcap;
    int32_t loop_start, loop_end, passes, gate_rank;
    const uint8_t *layer_sliding; /* [n_layers], 1 = attends to the last `window` positions */
} sk_ink2_dims;
sk_exec *sk_exec_create_ink2(const sk_ink2_dims *dims, const sk_exec_opts *opts, char *err, int err_len);
/* A layer's experts, read in place (host memory that outlives the executor): w13 [n_experts]
 * [2*moe_inter][hidden] with rows gate 0, up 0, gate 1, ..., w2 [n_experts][hidden][moe_inter].
 * SK_XFMT_NVFP4: two E2M1 codes a byte (the even element in the low nibble), scales e4m3 one per
 * 16 values, factor [n_experts] (amax / outer_scale_denominator). SK_XFMT_MXFP8: e4m3 values,
 * scales one exponent byte (2^(s - 127)) per 32 values. SK_XFMT_BF16: bf16 values, no scales. */
enum { SK_XFMT_NVFP4 = 0, SK_XFMT_MXFP8 = 1, SK_XFMT_BF16 = 2 };
int sk_exec_set_ink2_experts(sk_exec *e, int32_t layer, int32_t fmt, const uint8_t *w13, const uint8_t *w13_scale,
                             const float *w13_factor, const uint8_t *w2, const uint8_t *w2_scale,
                             const float *w2_factor, char *err, int err_len);
/* Inkling 2's draft head (the release's mtp.safetensors, a DSpark draft), before finalize: a block
 * of `block` rows (the last sampled token, then mask rows) through n_layers layers whose attention
 * sees, unmasked, the block and keys / values made from the target's features at every earlier
 * position (the `window` before the block in the layers marked local), then a recurrent head of
 * width `rank` that chains block - 1 tokens. taps: the layer visits, in running order, whose input
 * stream is a feature; the last is the number of visits and names the final norm's output.
 * Tensors (add_matrix, bf16): "mtp.L<i>.q" [hq*d][hidden], ".k", ".v", ".k_ctx", ".v_ctx" [hk*d]
 * [hidden], ".o" [hidden][hq*d], ".gate", ".up" [inter][hidden], ".down" [hidden][inter];
 * "mtp.rnn_s" [3 rank][2 rank] (rnn_proj's state and previous-token columns), "mtp.rnn_x" [3 rank]
 * [2 hidden] (its row and feature columns), "mtp.w1" [vocab][rank], "mtp.hidden_out" [hidden]
 * [rank]. Vectors: "mtp.L<i>.dflare", ".in_norm", ".post_norm" [hidden], ".q_norm", ".k_norm",
 * ".k_norm_ctx" [d]; "mtp.fuse" [(n_layers + 1) * 4] (softmaxed here), "mtp.mask", "mtp.norm",
 * "mtp.m_dflare", "mtp.hidden_out_b" [hidden], "mtp.rnn_b" [3 rank].
 * The head runs where the steps run on a GPU (Metal); elsewhere it is not used. */
typedef struct {
    int32_t n_layers, hq, hk, d, inter, window, block, rank;
    float theta;
    int32_t taps[4];
    uint8_t local[8];
} sk_ink2_mtp;
int sk_exec_set_ink2_mtp(sk_exec *e, const sk_ink2_mtp *cfg, char *err, int err_len);
/* Whether the head drafts (after finalize). */
int sk_exec_ink2_has_draft(const sk_exec *e);
/* After a step (and its sk_exec_drop_tail calls): up to k (< block) drafts for each of n sequences
 * of that step. Sequence i (slot slots[i]) holds pos[i] positions and the token sampled after its
 * last one is next[i]; the drafts are the tokens the head expects after next[i], into out [n][k],
 * each with its probability under the head's logits in conf [n][k] (every draft given the ones
 * before it: the product over a prefix estimates the chance the model agrees with all of it).
 * page_table as a step's. */
int sk_exec_ink2_draft(sk_exec *e, int32_t n, const int32_t *slots, const int32_t *next, const int32_t *pos,
                       const int32_t *page_table, int32_t k, int32_t *out, float *conf, char *err, int err_len);
/* The integer-dot NVFP4 expert rows (ink2_exec.cpp) against the rows decoded to floats, on random
 * data: the largest error over the outputs' RMS; -1 when this build has no integer path. */
double sk_ink2_sdot_error(void);
/* An EAGLE-3 draft head on a dense executor (before finalize). Every step keeps, for each of its
 * tokens, the residual stream entering layers taps[0], taps[1], taps[2] (layer 0: the embedding),
 * side by side; sk_exec_draft runs the head on rows of the last step. The head's layer has the
 * target's attention shape (n_heads, n_kv_heads, head_dim, RoPE): its KV cache is one more layer
 * of the paged cache. Tensors: "eagle.fc" [hidden][3*hidden], "eagle.q" [heads*head_dim]
 * [2*hidden], "eagle.k" and "eagle.v" [kv_heads*head_dim][2*hidden], "eagle.o" [hidden]
 * [heads*head_dim], "eagle.gate" and "eagle.up" [inter][hidden], "eagle.down" [hidden][inter],
 * "eagle.lm_head" [draft_vocab][hidden] (optional when draft_vocab is the vocabulary: the model's lm_head); vectors "eagle.in_norm" (on the token embedding),
 * "eagle.hidden_norm", "eagle.post_norm", "eagle.norm" [hidden], "eagle.d2t" [draft_vocab] (the
 * target id of each draft id, as float). */
typedef struct {
    int32_t taps[3];
    int32_t inter, draft_vocab;
} sk_eagle_dims;
int sk_exec_set_eagle(sk_exec *e, const sk_eagle_dims *dims, char *err, int err_len);

/* The draft call, after a step: per sequence, n_in tokens consumed at positions pos0, pos0+1, ...
 * (the token after each position: t(j+1) at j) with the hidden states the last step tapped at its
 * token rows `rows`, then k drafts (target ids) into drafts[s * k_max ...]. The head's cache is
 * written at every position consumed or drafted: their pages must be in the page table. */
typedef struct {
    int32_t n_seqs;
    const int32_t *slot;       /* [n_seqs] */
    const int32_t *pos0;       /* [n_seqs] */
    const int32_t *n_in;       /* [n_seqs], >= 1 */
    const int32_t *k;          /* [n_seqs], 0..k_max */
    const int32_t *tokens;     /* [sum n_in] */
    const int32_t *rows;       /* [sum n_in] */
    const int32_t *page_table; /* [n_slots][max_pages_per_seq], as a step's */
} sk_draft_plan;
int sk_exec_draft(sk_exec *e, const sk_draft_plan *plan, int32_t k_max, int32_t *drafts, char *err, int err_len);
void sk_exec_destroy(sk_exec *e);

/* Canonical names, per layer prefixed "L<l>.":
 *   global  embed (bf16, host-resident, NOT copied), lm_head, final_norm, fold_out
 *   layer   in_norm post_norm, plus fold_attn fold_mlp with attn_res
 *   KDA     q k v o f_a f_b b q_conv k_conv v_conv a_log dt_bias o_norm, and g (full-rank
 *           gate) or g_a g_b (kda_gate_lowrank)
 *   MLA     kv_a kv_a_norm kv_b o, g when mla_out_gate, and q_a q_a_norm q_b (q_lora > 0)
 *           or q (direct)
 *   dense   d_gate d_up d_down
 *   MoE     router router_bias sh_gate sh_up sh_down, plus down up [latent_norm] with
 *           latent_moe
 * fold_out (global) only with attn_res.
 * scale is [rows] for INT8/FP8 and ignored otherwise. */
int sk_exec_add_matrix(sk_exec *e, const char *name, int32_t fmt, int64_t rows, int64_t cols,
                       const void *data, const float *scale, char *err, int err_len);
int sk_exec_add_vector(sk_exec *e, const char *name, const float *data, int64_t n, char *err,
                       int err_len);
/* Host pointers into the MXFP4 expert store; must outlive the executor. */
int sk_exec_set_expert(sk_exec *e, int32_t layer, int32_t expert, const uint8_t *w1_packed,
                       const uint8_t *w1_scales, const uint8_t *w3_packed,
                       const uint8_t *w3_scales, const uint8_t *w2_packed,
                       const uint8_t *w2_scales);
/* ---- routed experts resident in backend memory (the HBM expert tier) ----
 * Reserve an arena of `bytes` for experts stored as fmt (SK_FMT_BF16, INT8 or FP8), then copy
 * experts into it; pairs routed to a resident expert are computed by the backend, the rest by
 * the CPU expert pool. Host pointers are copied and may be freed on return. Scales are
 * [rows] f32 for INT8/FP8 and must be null for BF16. FP8B and INT4G experts come as bf16 (null
 * scales) and are quantised by the executor; INT4G may come quantised already (a GPTQ
 * checkpoint): one int8 per value in [-8, 7], [rows][cols], with the group scales
 * [ceil(cols/128)][rows] f32 (int4_group.h). Call before finalize. */
int sk_exec_reserve_gpu_experts(sk_exec *e, int32_t fmt, size_t bytes, char *err, int err_len);

/* ---- LoRA adapters, unmerged: many at once (dense executor) ----
 * A target linear layer W of layer l (bits of `targets`: 1 q, 2 k, 4 v, 8 o, 16 gate, 32 up, 64
 * down; the MLP's only on dense models) gives every token of a sequence with an adapter
 * W x + scale B (A x), A [rank][in] and B [out][rank] that adapter's; tokens without one, W x.
 * Reserve `slots` adapters of rank up to max_rank before finalize; set and clear them, and tie
 * sequences to them, between steps. */
int sk_exec_reserve_lora(sk_exec *e, int32_t slots, int32_t max_rank, uint32_t targets, char *err, int err_len);
/* Slot `slot`'s update of layer `layer`'s target (0 q, 1 k, 2 v, 3 o, 4 gate, 5 up, 6 down): a
 * [rank][in] and b [out][rank] bf16 (host; a_len, b_len their element counts, which must be
 * exactly that), rank <= max_rank. */
int sk_exec_set_lora(sk_exec *e, int32_t slot, int32_t layer, int32_t target, const uint16_t *a, size_t a_len,
                     const uint16_t *b, size_t b_len, int32_t rank, float scale, char *err, int err_len);
/* Every target of slot `slot` back to nothing (B x A = 0). */
int sk_exec_clear_lora(sk_exec *e, int32_t slot, char *err, int err_len);
/* The adapter of sequence slot `seq` (-1: none) from the next step on. */
int sk_exec_set_seq_lora(sk_exec *e, int32_t seq, int32_t slot, char *err, int err_len);
int sk_exec_set_gpu_expert(sk_exec *e, int32_t layer, int32_t expert, const void *w1, const float *w1_scale,
                           const void *w3, const float *w3_scale, const void *w2, const float *w2_scale,
                           char *err, int err_len);
/* The quantised-expert cache (experts the executor quantises: SK_FMT_FP8B, SK_FMT_INT4G), after
 * sk_exec_reserve_gpu_experts: *state = 1 when the file at path holds this model's experts
 * (sk_exec_set_gpu_expert then ignores its data pointers: call it for every expert in (layer,
 * expert) order, as when the file was recorded), 0 when this load records it (written to
 * path.tmp, renamed by finalize), -1 when there is no cache for this executor or format. */
int sk_exec_expert_cache(sk_exec *e, const char *path, int32_t *state, char *err, int err_len);
/* Backend bytes finalize will allocate for pools, workspace and staging: what remains of
 * device memory after this (and the trunk) is the budget for resident experts. */
size_t sk_exec_pool_bytes(const sk_exec *e);

/* Checks every tensor is present with the right shape and allocates the pools. */
int sk_exec_finalize(sk_exec *e, char *err, int err_len);

typedef struct {
    int32_t n_tokens;
    const int32_t *tokens;     /* [n_tokens] */
    const int32_t *positions;  /* [n_tokens] absolute positions */
    const int32_t *token_slot; /* [n_tokens] state slot of each token's sequence */
    int32_t n_segs;
    const int32_t *segs;       /* [n_segs][3]: slot, first token row, count */
    const int32_t *page_table; /* [n_slots][max_pages_per_seq], the whole table */
    int32_t n_rows;
    const int32_t *rows;       /* [n_rows] token rows whose logits are wanted */
    int32_t n_reset;
    const int32_t *reset_slots; /* slots to zero (state + conv history) before the step */
} sk_step_plan;

typedef struct {
    double ms_total, ms_gpu_wait, ms_experts, ms_upload, ms_download;
    int64_t expert_bytes;     /* routed-expert bytes read by the CPU this step */
    int64_t distinct_experts; /* computed on the CPU, summed over layers */
    int64_t gpu_pairs;        /* (token, expert) pairs computed by resident experts */
    int64_t cpu_pairs;        /* pairs computed by the CPU expert pool */
} sk_step_stats;

/* logits: host [n_rows][vocab]. stats may be NULL. */
int sk_exec_step(sk_exec *e, const sk_step_plan *plan, float *logits, sk_step_stats *stats,
                 char *err, int err_len);

/* The same step for greedy decoding: tokens[r] = the first index of the largest logit of row r,
 * computed on the device; only n_rows ids cross to the host. Fails like sk_exec_step's caller
 * would on a non-finite logit. */
int sk_exec_step_argmax(sk_exec *e, const sk_step_plan *plan, int32_t *tokens, sk_step_stats *stats,
                        char *err, int err_len);

/* How a row's token is drawn (sk_exec_step_sample, sk_sample_rows_host). temperature <= 0: the
 * argmax. Otherwise softmax(logits / temperature) over the candidates: the values at or above the
 * top_k-th largest (top_k 0 or >= vocab: all; ties at the threshold all stay), then those at or
 * above the largest threshold whose candidates carry at least top_p of the mass (1: all); the
 * token is the one whose cumulative mass, in token order, first passes u * mass, u being the
 * first double of PCG32 seeded (seed, step) (step: the request's index of the token). A function
 * of the row alone: the same on any batch, and the same token a draft is verified against.
 * mask k > 0: the row draws only among the tokens allowed by the step's mask k - 1 (structured
 * output), as if every other logit were -inf; 0: no mask. min_p_cut < 0 (min-p): the candidates
 * are also only the logits at least the row's largest + min_p_cut (temperature x ln(min_p),
 * computed once by the caller so every backend cuts at the same float); 0: off. adj_n > 0: the
 * logits first get the step's adjustments [adj_off, adj_off + adj_n) (sk_logit_adj; after the
 * mask). logprobs k > 0: the drawn token's log-probability and the k - 1 most likely tokens',
 * from the logits as the model gave them (before mask, adjustments and temperature); 0: none.
 * Zero fields leave a row as it was. */
typedef struct sk_sample_row {
    float temperature;
    float top_p;
    int32_t top_k;
    int32_t mask;
    uint64_t seed;
    uint64_t step;
    float min_p_cut;
    int32_t adj_off;
    int32_t adj_n;
    int32_t logprobs;
} sk_sample_row;

/* One token's logit adjustment (penalties, logit bias, min_tokens), in vLLM's order:
 * x = x + bias; x = x > 0 ? x / rep : x * rep; x = x - sub (sub = +inf: never drawn). A row's
 * entries name distinct tokens. */
typedef struct sk_logit_adj {
    int32_t token;
    float bias;
    float rep;
    float sub;
} sk_logit_adj;

#define SK_MAX_TOP_LOGPROBS 20

/* A sampling step's inputs and outputs besides its rows. masks: host, the rows' allowed-token
 * bitmasks, each mask_words words (token i: bit i % 32 of word i / 32; mask_words >=
 * ceil(vocab / 32)), as many as the largest sample[r].mask (NULL when no row has one). adj:
 * host, n_adj entries the rows' adj_off / adj_n index. logprobs / top_ids: host, filled for the
 * rows with logprobs > 0 (NULL when none has): row r's drawn token's log-probability at
 * [r][0] and its top alternatives' at [r][1 ..], their ids at top_ids[r][0 ..]
 * ([rows][SK_MAX_TOP_LOGPROBS + 1] floats, [rows][SK_MAX_TOP_LOGPROBS] ids). */
typedef struct sk_sample_io {
    const uint32_t *masks;
    int64_t mask_words;
    const sk_logit_adj *adj;
    int64_t n_adj;
    float *logprobs;
    int32_t *top_ids;
} sk_sample_io;

/* The step for sampling: tokens[r] drawn from row r's logits as sample[r] says (host array of
 * n_rows), on the device; only the ids (and asked-for log-probabilities) cross to the host.
 * Fails on a non-finite logit. io: NULL, or the masks, adjustments and log-probability
 * buffers above. */
int sk_exec_step_sample(sk_exec *e, const sk_step_plan *plan, const sk_sample_row *sample, const sk_sample_io *io,
                        int32_t *tokens, sk_step_stats *stats, char *err, int err_len);

/* The draw on the host (the reference the backends match): rows of n logits. */
void sk_sample_rows_host(int32_t *tokens, const float *logits, int64_t rows, int64_t n, const sk_sample_row *sample);

/* Copy a slot's recurrent state + conv history to/from host memory (checkpoints, swap).
 * bytes must equal sk_exec_slot_bytes. */
size_t sk_exec_slot_bytes(const sk_exec *e);
int sk_exec_save_slot(sk_exec *e, int32_t slot, void *host_dst, char *err, int err_len);
int sk_exec_load_slot(sk_exec *e, int32_t slot, const void *host_src, char *err, int err_len);
/* After a step: the last n rows of `slot`'s segment in it are void (rejected drafts). An executor
 * whose per-sequence state is more than the paged cache puts it back as it was after the rows
 * before them (Inkling 2: the convolution histories, for up to 16 rows); the others have nothing
 * to do: the cache positions are simply written again. */
int sk_exec_drop_tail(sk_exec *e, int32_t slot, int32_t n, char *err, int err_len);
/* Per-sequence state besides the paged cache, kept by cache page so that a page can start another
 * sequence (the prefix cache): after enable (before finalize), every step keeps, for each page a
 * sequence fills to its last position, the state as it is after that position (Inkling 2: the
 * convolution histories, slot_bytes a page, committed as pages fill); load puts page `page`'s into
 * `slot`, whose next token is then the one after the page. Executors without such state refuse. */
int sk_exec_page_state_enable(sk_exec *e, char *err, int err_len);
int sk_exec_page_state_load(sk_exec *e, int32_t slot, int32_t page, char *err, int err_len);

/* The prefix cache's host tier (paged-cache executors; after finalize). A host page holds one
 * cache page of every paged pool: sk_exec_page_bytes bytes, 0 when the executor has no host tier.
 * sk_exec_copy_pages queues n page copies (to_host: cache page dev_pages[i] -> host page
 * host_pages[i], else the reverse) on a copy stream that starts after the steps already run, and
 * returns a ticket; tickets complete in order. copy_done asks without blocking; copy_order makes
 * the next step wait for a ticket on the device; copy_wait blocks the host. Ticket 0 is none. */
size_t sk_exec_page_bytes(const sk_exec *e);
int sk_exec_reserve_host_pages(sk_exec *e, int64_t n, char *err, int err_len);
int sk_exec_copy_pages(sk_exec *e, int32_t to_host, const int32_t *dev_pages, const int32_t *host_pages, int64_t n,
                       uint64_t *ticket, char *err, int err_len);
int sk_exec_copy_done(sk_exec *e, uint64_t ticket);
int sk_exec_copy_order(sk_exec *e, uint64_t ticket, char *err, int err_len);
int sk_exec_copy_wait(sk_exec *e, uint64_t ticket, char *err, int err_len);

/* Human-readable placement summary; valid until the next call or destroy. */
const char *sk_exec_describe(sk_exec *e);

/* The backends compiled into this build: "name:available" pairs separated by commas (available
 * 1 when the backend can run on this machine, e.g. "cpu:1,cpu-core:1,cuda:0"). Returns the length
 * of the whole list (the output is truncated to len - 1 characters). */
int sk_backend_list(char *out, int len);
/* The capabilities of the named backend on device, with or without bf16 activations. */
int sk_backend_caps(const char *backend, int32_t device, int32_t bf16_activations, sk_caps *out, char *err, int err_len);

/* ------------------------------------------------------------------ device utilities */
int sk_cuda_available(void);
/* name: host buffer. Returns 0 on success. */
int sk_cuda_device_info(int32_t device, char *name, int name_len, size_t *total_mem,
                        size_t *free_mem, int32_t *sm_count, int32_t *cc_major,
                        int32_t *cc_minor);
/* Pinned host<->device copy bandwidth, GB/s. */
int sk_cuda_bench_copy(int32_t device, size_t bytes, int32_t iters, double *h2d_gbps,
                       double *d2h_gbps);
/* Achieved weight-read bandwidth of the gemm path for an [n][k] matrix of fmt with M rows
 * of activations, GB/s of weight bytes. */
int sk_cuda_bench_gemm(int32_t device, int32_t fmt, int64_t n, int64_t k, int64_t m,
                       int32_t iters, double *gbps, double *ms);

/* Op-by-op comparison of the CUDA backend against the CPU backend on random inputs.
 * Writes a report (one line per op: max abs / rel error, PASS/FAIL) and returns the
 * number of failing ops, or -1 if CUDA is unavailable. */
int sk_exec_selftest(int32_t device, char *report, int report_len);
/* The conformance suite (conformance.cpp): every op of the subject backend against the reference
 * backend (both by name; the reference usually "cpu"), skipping the model families and KV formats
 * the subject does not declare. Returns the failure count, -1 when it could not run. */
int sk_conformance(const char *subject, const char *reference, int32_t device, char *report, int report_len);

#ifdef __cplusplus
}
#endif

#endif /* FULLM_EXEC_H */
