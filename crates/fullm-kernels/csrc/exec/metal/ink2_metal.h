// SPDX-License-Identifier: Apache-2.0
// ink2_metal.h - Inkling 2's whole step on an Apple GPU (Metal).
//
// A step (a prompt chunk or one token of every running sequence) is encoded as compute dispatches
// into command buffers that run without the host in between: between two GPU calls the host used
// to cost a millisecond of scheduling plus its own copies, 240 times a prompt chunk. The host
// writes the step's embeddings and tables, commits, waits once, and reads the logits.
//
// Memory is the machine's own (unified): the dense matrices, the KV pools and the convolution
// histories are allocated here (ink2_metal_alloc) and used through plain pointers by the CPU path
// too; the experts' pages of the mapped checkpoint are wrapped, not copied.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fullm_exec.h"

namespace fullm {

struct Ink2Metal;

// The default GPU with the kernels compiled, or null (`why` says: no device, a shader error).
// f32_act: activations in f32 where the kernels hold them as half by default (the checks against
// the float oracle; half costs ~2e-3 of an expert's output).
// f32_kv: the cache pools (the model's and the draft head's keys and values) hold f32, else half.
Ink2Metal *ink2_metal_create(bool f32_act, bool f32_kv, std::string &why);
void ink2_metal_destroy(Ink2Metal *m);
const char *ink2_metal_device(const Ink2Metal *m);

// Zeroed page-aligned memory the GPU sees (freed with the device). Null when it cannot be had.
void *ink2_metal_alloc(Ink2Metal *m, size_t bytes);

// A layer's expert stacks, read in place (the pointers as sk_exec_set_ink2_experts takes them;
// fmt SK_XFMT_NVFP4 or SK_XFMT_MXFP8). False when the pages cannot be wrapped.
bool ink2_metal_add_layer(Ink2Metal *m, int layer, int fmt, int n_experts, int hidden, int inter, const uint8_t *w13,
                          const uint8_t *s13, const float *f13, const uint8_t *w2, const uint8_t *s2, const float *f2);

// A layer as the executor holds it. Matrices are bf16 [rows][cols] in ink2_metal_alloc memory;
// vectors f32 (copied when they are not in such memory); taps [tap][channel].
struct Ink2MetalLayer {
    bool sliding = false, loop = false;
    int hq = 0, hk = 0, d = 0, extent = 0;
    const void *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr, *og = nullptr, *sink = nullptr, *rel = nullptr;
    const void *router = nullptr, *k_wp = nullptr, *v_wp = nullptr;
    const void *c_down[4] = {}, *c_forget[4] = {}, *c_gain[4] = {};
    const float *attn_query = nullptr, *mlp_query = nullptr, *q_scale = nullptr, *rel_scale = nullptr, *rel_proj = nullptr;
    const float *post_norm = nullptr, *router_bias = nullptr, *in_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    const float *v_norm = nullptr;
    const float *attn_conv = nullptr, *mlp_conv = nullptr, *k_conv = nullptr, *v_conv = nullptr, *k_static = nullptr;
    const float *v_static = nullptr;
};

// The model's draft head (DSpark; mtp.safetensors): a block of `block` rows (the last sampled token,
// then mask rows) attends, without a mask, to keys and values made from the target's features at
// every earlier position and to itself, and a small recurrent head turns the mask rows into a
// chain of tokens. Matrices bf16, vectors f32, as the executor holds them.
struct Ink2MetalMtpLayer {
    bool local = false;  // sees the last `window` positions before the block, else all
    const void *q = nullptr, *k = nullptr, *v = nullptr, *k_ctx = nullptr, *v_ctx = nullptr, *o = nullptr;
    const void *gate = nullptr, *up = nullptr, *down = nullptr;
    const float *dflare = nullptr, *in_norm = nullptr, *post_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    const float *k_norm_ctx = nullptr;
};
struct Ink2MetalMtp {
    bool on = false;
    int hq = 0, hk = 0, d = 0, inter = 0, window = 0, block = 0, rank = 0;
    float theta = 0;
    int taps[4] = {};  // the visits (in running order) whose input is a feature; the last: the final norm's output
    std::vector<Ink2MetalMtpLayer> layers;
    const float *alpha = nullptr;  // [layers + 1][4]: each layer's mix of the taps, then the recurrent head's
    const float *mask = nullptr, *norm = nullptr, *m_dflare = nullptr, *rnn_b = nullptr, *fold_b = nullptr;
    // rnn_s [3 rank][2 rank] (state | previous token), rnn_x [3 rank][2 hidden] (row | feature),
    // w1 [vocab][rank], fold [out_vocab][rank] (lm_head x hidden_out; fold_b its bias)
    const void *rnn_s = nullptr, *rnn_x = nullptr, *w1 = nullptr, *fold = nullptr;
    std::vector<float *> dk, dv;  // [layer]: the context's keys and values, pools laid out as the model's
};

struct Ink2MetalModel {
    Ink2MetalMtp mtp;
    sk_ink2_dims d{};
    sk_exec_opts o{};
    std::vector<Ink2MetalLayer> layers;
    const void *lm_head = nullptr;
    const float *final_norm = nullptr, *out_query = nullptr, *msg_queries = nullptr;
    // state, in ink2_metal_alloc memory: per visit the key and value pools ([page][kv head][position
    // in page][head width], f32 or half as the device was created), the convolution histories ([slot][vis_off.back()] floats, a
    // visit's at vis_off[visit]: keys, values, attention output, mlp output, K - 1 rows each)
    std::vector<float *> kpool, vpool;
    float *conv = nullptr;
    std::vector<int64_t> vis_off;
    // what a rejected draft puts back (the executor's drop_tail): [slot][undo_stride] floats, per
    // history the last undo_rows + K - 1 rows of [history, the step's rows], laid out as conv with
    // undo_rows + K - 1 rows where it has K - 1
    float *undo = nullptr;
    int64_t undo_stride = 0, undo_rows = 0;
    // null, or [page][vis_off.back()] floats: the histories after each page's last position, laid
    // out as a slot's (the prefix cache starts sequences from them)
    float *tails = nullptr;
};

// The model for ink2_metal_step. False (why) when a shape is outside what the kernels take: the
// executor then stays on the CPU.
bool ink2_metal_setup(Ink2Metal *m, const Ink2MetalModel &model, std::string &why);

// One step: h0 [n_tokens][hidden] the normalised embeddings; logits [n_rows][out_vocab],
// softcapped. Blocks until the GPU is done. False on a GPU error (the state is then undefined).
bool ink2_metal_step(Ink2Metal *m, const sk_step_plan &pl, const float *h0, float *logits, std::string &why);

// Drafts for n sequences after a step: sequence i (slot slots[i]) holds pos[i] positions, its last
// one was row last_row[i] of the last step, and the token sampled there is next[i], whose
// normalised embedding is emb[i]. k tokens each into out [n][k] (k < the block), and into conf each
// one's probability under the head's softcapped logits. page_table as a step's. False on a GPU error.
bool ink2_metal_draft(Ink2Metal *m, int n, const int32_t *slots, const int32_t *pos, const int32_t *last_row, const int32_t *next,
                      const float *emb, const int32_t *page_table, int k, int32_t *out, float *conf, std::string &why);

// The last step: GPU execution and host wall milliseconds, dispatches encoded.
void ink2_metal_last(const Ink2Metal *m, double *gpu_ms, double *wall_ms, int64_t *dispatches);

}  // namespace fullm
