// SPDX-License-Identifier: Apache-2.0
// dense_exec.cpp - the dense decoder graph (Llama / Qwen3 family) on a Backend.
//
// Per layer: h -> rmsnorm -> q, k, v (one grouped GEMM) -> [per-head RMSNorm of q and k] ->
// RoPE -> append k, v to the paged cache -> causal GQA attention -> o_proj, + h;
// -> rmsnorm -> gate, up (grouped) -> SiLU-GLU -> down, + h; or, for a MoE model, router ->
// softmax top-k -> resident SwiGLU experts -> weighted sum, + h. Each residual add is fused
// with the pre-norm that follows it. The embedding table stays in host memory (rows gathered
// per token); everything else lives in backend memory. Small steps replay CUDA graphs as the
// K3 executor does. Mirrors transformers' Qwen3 / Qwen3-MoE / Llama.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <sstream>
#include <tuple>
#include <vector>

#include "fp8_block.h"
#include <cstdlib>
#include "int4_group.h"
#include "kv_fp8.h"
#include "exec_common.h"
#include "kv_int4.h"

namespace fullm {
namespace {

struct DenseLayer {
    const float *in_norm = nullptr, *post_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    Weight q, k, v, o, gate, up, down, router;
};

class DenseExec final : public ExecIface {
public:
    DenseExec(const sk_dense_dims &d, const sk_exec_opts &o) : d_(d), o_(o) {
        if (d.n_layers <= 0 || d.hidden <= 0 || d.n_heads <= 0 || d.n_kv_heads <= 0 || d.head_dim <= 0 ||
            d.inter <= 0 || d.vocab <= 0)
            throw std::runtime_error("dense dims must all be positive");
        if (d.n_heads % d.n_kv_heads) throw std::runtime_error("n_heads must be a multiple of n_kv_heads");
        if (d.head_dim % 2) throw std::runtime_error("head_dim must be even (RoPE pairs)");
        if (!d.inv_freq) throw std::runtime_error("inv_freq is null");
        if (d.n_experts < 0 || (d.n_experts > 0 && (d.topk <= 0 || d.topk > d.n_experts || d.moe_inter <= 0)))
            throw std::runtime_error("dense dims: bad MoE shape");
        if (o.n_slots <= 0 || o.n_pages <= 0 || o.page <= 0 || o.max_pages_per_seq <= 0 || o.max_tokens <= 0 ||
            o.max_rows <= 0)
            throw std::runtime_error("executor options must all be positive");
        inv_freq_.assign(d.inv_freq, d.inv_freq + d.head_dim / 2);
        d_.inv_freq = nullptr;  // copied
        kv_dt_ = o.kv_dtype == SK_DT_KV4       ? DType::KV4
                 : o.kv_dtype == SK_DT_FP8E4M3 ? DType::FP8E4M3
                 : o.kv_dtype == SK_DT_FP8   ? DType::FP8
                 : o.kv_dtype == SK_DT_BF16  ? DType::BF16
                                             : DType::F32;
        b_ = make_backend(backend_name(o.backend_name, o.backend), o.device, o.cpu_threads);
        b_->set_exact_activations(o.bf16_activations == 0);
        // cuBLAS from 4 rows for a dense model, 8 for a MoE (H100, against the GEMV at 4 rows:
        // Qwen3-8B c=4 557 -> 583 tok/s, Qwen3-30B-A3B int4 939 -> 917)
        b_->set_blas_min_rows(d.n_experts > 0 ? 8 : 4);
        const Caps caps = b_->caps();
        if (!(caps.families & FAM_GQA))
            throw std::runtime_error(std::string("the ") + b_->name() + " backend has no grouped-query attention (dense models)");
        if (!(caps.kv_formats & kv_bit(kv_dt_)))
            throw std::runtime_error(std::string("the ") + b_->name() + " backend has no " +
                                     (kv_dt_ == DType::F32 ? "fp32" : kv_dt_ == DType::BF16 ? "bf16" : kv_dt_ == DType::KV4 ? "int4" : "fp8") +
                                     " KV cache" + (o.bf16_activations == 0 ? " with exact activations" : ""));
        if ((kv_dt_ == DType::FP8 || kv_dt_ == DType::FP8E4M3) && o.page % caps.kv8_page_multiple)
            throw std::runtime_error("--kv-dtype fp8 needs pages of a multiple of " + std::to_string(caps.kv8_page_multiple) +
                                     " positions");
        if (kv_dt_ == DType::KV4 && (o.page != kKv4Page || d.head_dim != kKv4D || o.n_slots > kKv4MaxSlots))
            throw std::runtime_error("--kv-dtype int4: head_dim 128, pages of 16 positions, at most " +
                                     std::to_string(kKv4MaxSlots) + " sequences");
        const char *rows = std::getenv("FULLM_INT4_ROWS");
        int4_rows_ = rows ? std::atoll(rows) : b_->policy().expert_tile_max_rows;
    }

    ~DenseExec() override {
        qc_drop();  // a load that failed before finalize: no partial recording survives
        if (!b_) return;
        host_.drain();  // host-tier copies still reading or writing the pools
        try {
            b_->sync();
        } catch (...) {
        }
        for (auto &kv : graphs_) b_->graph_release(kv.second);
        for (auto &kv : mats_) b_->release(const_cast<void *>(kv.second.data)), b_->release(const_cast<float *>(kv.second.scale));
        for (auto &kv : vecs_) b_->release(kv.second.first);
        sample_scratch_.release(b_.get());
        for (void *p : owned_) b_->release(p);
        for (void *p : host_owned_) b_->host_release(p);
        ws_.reset();
    }

    // ------------------------------------------------------------------ loading
    void add_matrix(const std::string &name, int32_t fmt, int64_t rows, int64_t cols, const void *data,
                    const float *scale) override {
        if (finalized_) throw std::runtime_error("add after finalize");
        if (rows <= 0 || cols <= 0 || !data) throw std::runtime_error(name + ": empty matrix");
        Weight w;
        w.fmt = static_cast<WFmt>(fmt);
        w.rows = rows;
        w.cols = cols;
        if (name == "embed") {
            if (w.fmt != WFmt::BF16 && w.fmt != WFmt::F32) throw std::runtime_error("embed must be bf16 or f32");
            // On a device backend the table lives there and rows are gathered on the device
            // (FULLM_HOST_EMBED=1 keeps it in host memory, for models that barely fit).
            embed_dev_ = !b_->on_host() && !std::getenv("FULLM_HOST_EMBED") && cols % 4 == 0;
            if (!embed_dev_) {
                w.data = data;  // host memory, not copied
                embed_ = w;
                return;
            }
            const size_t bytes = static_cast<size_t>(rows) * cols * (w.fmt == WFmt::F32 ? 4 : 2);
            void *dd = b_->alloc(bytes);
            b_->upload(dd, data, bytes);
            b_->sync();
            w.data = dd;
            embed_ = w;
            weight_bytes_ += bytes;
            return;
        }
        const size_t elem = w.fmt == WFmt::F32 ? 4 : w.fmt == WFmt::BF16 ? 2 : 1;
        void *dd = b_->alloc(static_cast<size_t>(rows) * cols * elem);
        b_->upload(dd, data, static_cast<size_t>(rows) * cols * elem);
        w.data = dd;
        if (w.fmt == WFmt::INT8 || w.fmt == WFmt::FP8) {
            if (!scale) throw std::runtime_error(name + ": 8-bit matrix without row scales");
            float *ds = static_cast<float *>(b_->alloc(rows * 4));
            b_->upload(ds, scale, rows * 4);
            w.scale = ds;
        }
        b_->sync();  // the caller frees its copy on return
        if (mats_.count(name)) throw std::runtime_error(name + ": added twice");
        mats_[name] = w;
        weight_bytes_ += w.bytes();
    }

    void add_vector(const std::string &name, const float *data, int64_t n) override {
        if (finalized_) throw std::runtime_error("add after finalize");
        if (vecs_.count(name)) throw std::runtime_error(name + ": added twice");
        float *dd = static_cast<float *>(b_->alloc(n * 4));
        b_->upload(dd, data, n * 4);
        b_->sync();
        vecs_[name] = {dd, n};
        if (name == "eagle.d2t") {  // drafts are mapped to target ids on the host
            eg_d2t_.resize(size_t(n));
            for (int64_t i = 0; i < n; ++i) eg_d2t_[size_t(i)] = static_cast<int32_t>(std::lround(data[i]));
        }
    }

    void set_eagle(const sk_eagle_dims &e) override {
        if (finalized_) throw std::runtime_error("set_eagle after finalize");
        for (int j = 0; j < 3; ++j)
            if (e.taps[j] < 0 || e.taps[j] >= d_.n_layers)
                throw std::runtime_error("EAGLE-3 tap layer " + std::to_string(e.taps[j]) + " outside 0.." +
                                         std::to_string(d_.n_layers - 1));
        if (e.inter <= 0 || e.draft_vocab <= 0) throw std::runtime_error("EAGLE-3 dims must be positive");
        ed_ = e;
        eagle_ = true;
    }

    // Routed experts live in one backend arena; with moe_gate_up_interleaved the backend wants
    // gate and up as one row-interleaved matrix (see Backend). bytes: the experts in fmt; the
    // int4 experts' fp8b copies, when made, are added here.
    void reserve_lora(int32_t slots, int32_t max_rank, uint32_t targets) override {
        if (finalized_) throw std::runtime_error("reserve_lora after finalize");
        if (lora_slots_) throw std::runtime_error("reserve_lora twice");
        if (slots <= 0 || max_rank <= 0 || max_rank > 512) throw std::runtime_error("reserve_lora: slots > 0, rank in 1..512");
        if (d_.n_experts > 0 && (targets & (16u | 32u | 64u)))
            throw std::runtime_error("LoRA on the MLP of an MoE model: routed experts take a merged adapter only");
        const int64_t E = d_.hidden, Q = int64_t(d_.n_heads) * d_.head_dim, KV = int64_t(d_.n_kv_heads) * d_.head_dim,
                      I = d_.inter;
        const int64_t dims[7][2] = {{E, Q}, {E, KV}, {E, KV}, {Q, E}, {E, I}, {E, I}, {I, E}};  // in, out
        // the pools' rank: a multiple of 8 (the kernels' 16-byte loads; each slot's own rank bounds the work)
        const int32_t rank = (max_rank + 7) / 8 * 8;
        lora_slots_ = slots, lora_rank_ = rank;
        lora_.assign(size_t(d_.n_layers), {});
        for (int l = 0; l < d_.n_layers; ++l)
            for (int g = 0; g < 4; ++g) {
                // a group's A pools stacked per slot ([slots][targets][rank][in]): one GEMM takes them
                int members[3], G = 0;
                for (int t : kLoraGroups[g])
                    if (t >= 0 && (targets >> t & 1u)) members[G++] = t;
                if (!G) continue;
                const int64_t in = dims[members[0]][0];
                const size_t na = size_t(slots) * G * rank * in;
                auto *a = static_cast<uint16_t *>(b_->alloc(na * 2));
                b_->zero(a, na * 2);
                owned_.push_back(a);
                for (int m = 0; m < G; ++m) {
                    LoraTarget &T = lora_[size_t(l)][size_t(members[m])];
                    T.in = in, T.out = dims[members[m]][1];
                    T.a = a + size_t(m) * rank * in, T.a_stride = int64_t(G) * rank * in;
                    const size_t nb = size_t(slots) * T.out * rank;
                    T.b = static_cast<uint16_t *>(b_->alloc(nb * 2));
                    T.scale = static_cast<float *>(b_->alloc(size_t(slots) * 4));
                    T.ranks = static_cast<int32_t *>(b_->alloc(size_t(slots) * 4));
                    b_->zero(T.b, nb * 2), b_->zero(T.scale, size_t(slots) * 4), b_->zero(T.ranks, size_t(slots) * 4);
                    owned_.push_back(T.b), owned_.push_back(T.scale), owned_.push_back(T.ranks);
                    T.scale_h.assign(size_t(slots), 0.0f), T.rank_h.assign(size_t(slots), 0);
                }
            }
        seq_lora_.assign(size_t(o_.n_slots), -1);
        d_row_lora_ = static_cast<int32_t *>(b_->alloc(size_t(o_.max_tokens) * 4));
        d_row_lora_rest_ = static_cast<int32_t *>(b_->alloc(size_t(o_.max_tokens) * 4));
        owned_.push_back(d_row_lora_), owned_.push_back(d_row_lora_rest_);
        b_->sync();
    }

    void set_lora(int32_t slot, int32_t layer, int32_t target, const uint16_t *a, size_t a_len, const uint16_t *b,
                  size_t b_len, int32_t rank, float scale) override {
        if (slot < 0 || slot >= lora_slots_ || layer < 0 || layer >= d_.n_layers || target < 0 || target > 6)
            throw std::runtime_error("set_lora: no slot " + std::to_string(slot) + " / layer " + std::to_string(layer) +
                                     " / target " + std::to_string(target));
        LoraTarget &T = lora_[size_t(layer)][size_t(target)];
        if (!T.a) throw std::runtime_error("set_lora: target " + std::to_string(target) + " was not reserved");
        if (rank <= 0 || rank > lora_rank_)
            throw std::runtime_error("set_lora: rank " + std::to_string(rank) + " (reserved up to " + std::to_string(lora_rank_) + ")");
        if (a_len != size_t(rank) * size_t(T.in) || b_len != size_t(T.out) * size_t(rank))
            throw std::runtime_error("set_lora: target " + std::to_string(target) + " takes A [" + std::to_string(rank) + "][" +
                                     std::to_string(T.in) + "] and B [" + std::to_string(T.out) + "][" + std::to_string(rank) +
                                     "], given " + std::to_string(a_len) + " and " + std::to_string(b_len) + " values");
        // padded to the reserved rank with zeros: A's extra rows, B's extra columns
        std::vector<uint16_t> ha(size_t(lora_rank_) * T.in, 0), hb(size_t(T.out) * lora_rank_, 0);
        std::memcpy(ha.data(), a, size_t(rank) * T.in * 2);
        for (int64_t o = 0; o < T.out; ++o) std::memcpy(&hb[size_t(o) * lora_rank_], b + o * rank, size_t(rank) * 2);
        b_->upload(T.a + size_t(slot) * T.a_stride, ha.data(), ha.size() * 2);
        b_->upload(T.b + size_t(slot) * T.out * lora_rank_, hb.data(), hb.size() * 2);
        b_->upload(T.scale + slot, &scale, 4);
        b_->upload(T.ranks + slot, &rank, 4);
        T.scale_h[size_t(slot)] = scale, T.rank_h[size_t(slot)] = rank;
        lora_active_rank();
        b_->sync();
    }

    void clear_lora(int32_t slot) override {
        if (slot < 0 || slot >= lora_slots_) throw std::runtime_error("clear_lora: no slot " + std::to_string(slot));
        for (auto &layer : lora_)
            for (auto &T : layer)
                if (T.a) {
                    b_->zero(T.a + size_t(slot) * T.a_stride, size_t(lora_rank_) * T.in * 2);
                    b_->zero(T.b + size_t(slot) * T.out * lora_rank_, size_t(T.out) * lora_rank_ * 2);
                    b_->zero(T.scale + slot, 4), b_->zero(T.ranks + slot, 4);
                    T.scale_h[size_t(slot)] = 0.0f, T.rank_h[size_t(slot)] = 0;
                }
        lora_active_rank();
        b_->sync();
    }

    // the loaded adapters' largest rank, up to a multiple of 8: the LoRA work's bound (a rank-16
    // adapter in rank-128 slots does rank-16 work); a graph key holds it
    void lora_active_rank() {
        int32_t r = 0;
        for (const auto &layer : lora_)
            for (const auto &T : layer)
                for (int32_t x : T.rank_h) r = std::max(r, x);
        lora_active_ = std::min(lora_rank_, (r + 7) / 8 * 8);
    }

    void set_seq_lora(int32_t seq, int32_t slot) override {
        if (!lora_slots_) throw std::runtime_error("set_seq_lora: no LoRA slots reserved");
        if (seq < 0 || seq >= o_.n_slots || slot < -1 || slot >= lora_slots_)
            throw std::runtime_error("set_seq_lora: sequence " + std::to_string(seq) + ", slot " + std::to_string(slot));
        seq_lora_[size_t(seq)] = slot;
    }

    void reserve_gpu_experts(int32_t fmt, size_t bytes) override {
        if (finalized_ || gx_arena_) throw std::runtime_error("reserve_gpu_experts: after finalize or twice");
        if (d_.n_experts == 0) throw std::runtime_error("this model has no routed experts");
        if (fmt != SK_FMT_BF16 && fmt != SK_FMT_INT8 && fmt != SK_FMT_FP8 && fmt != SK_FMT_FP8B && fmt != SK_FMT_INT4G)
            throw std::runtime_error("resident experts must be bf16, int8, fp8, fp8b or int4");
        gpu_fmt_ = static_cast<WFmt>(fmt);
        b_->set_expert_format(gpu_fmt_);
        // int4 experts: prefill steps run the backend's int4 GEMM over the same weights when it has
        // one; else an fp8b copy of each expert for them (the blockwise fp8 GEMM).
        // FULLM_INT4_PREFILL_FP8=1 keeps the copies anyway, =0 drops them (no int4 GEMM: prefill on
        // the int4 decode tiles)
        const char *pf = std::getenv("FULLM_INT4_PREFILL_FP8");
        const bool copies = pf ? std::string(pf) == "1" : !b_->int4_prefill_gemm(d_.hidden, d_.moe_inter);
        gx8_on_ = gpu_fmt_ == WFmt::INT4G && copies;
        if (gx8_on_) {  // gate/up interleaved and down, each with its block scales (take_gx aligns all four)
            const int64_t E = d_.hidden, I = d_.moe_inter;
            const int64_t per = 2 * I * E + fp8b_scales(2 * I, E) * 4 + E * I + fp8b_scales(E, I) * 4 + 4 * 256;
            bytes += size_t(per) * size_t(d_.n_layers) * size_t(d_.n_experts);
            gx_host8_.assign(d_.n_layers, std::vector<Backend::ExpertW>(d_.n_experts));
        }
        gx_arena_ = static_cast<char *>(b_->alloc(bytes));
        owned_.push_back(gx_arena_);
        gx_cap_ = bytes;
        gx_host_.assign(d_.n_layers, std::vector<Backend::ExpertW>(d_.n_experts));
    }

    void set_gpu_expert(int32_t layer, int32_t e, const void *w1, const float *s1, const void *w3, const float *s3,
                        const void *w2, const float *s2) override {
        if (!gx_arena_) throw std::runtime_error("set_gpu_expert before reserve_gpu_experts");
        if (layer < 0 || layer >= d_.n_layers || e < 0 || e >= d_.n_experts)
            throw std::runtime_error("set_gpu_expert: no expert " + std::to_string(e) + " in layer " + std::to_string(layer));
        Backend::ExpertW &slot = gx_host_[layer][e];
        if (slot.w1.data) throw std::runtime_error("gpu expert set twice");
        // INT8 / FP8 come quantised with row scales; FP8B comes as bf16 and is quantised here,
        // after the stored layout (gate/up rows interleaved or not) is fixed. INT4G comes as bf16
        // (quantised here) or, with scales, quantised already (a GPTQ checkpoint): one int8 value
        // in [-8, 7] each, [rows][cols], and the group scales [cols/128][rows] (int4_group.h).
        const bool q8 = gpu_fmt_ == WFmt::INT8 || gpu_fmt_ == WFmt::FP8;
        const bool given4 = gpu_fmt_ == WFmt::INT4G && s1;
        auto put = [&](const void *data, const float *scale, int64_t rows, int64_t cols, WFmt as) {
            const bool i4 = as == WFmt::INT4G, fb = as == WFmt::FP8B;
            Weight w;
            w.fmt = as;
            w.rows = rows;
            w.cols = cols;
            if (i4) {  // INT4G: like FP8B, from bf16, after the layout is fixed (int4_group.h), packed as
                       // the backend reads it
                if (cols % 8) throw std::runtime_error("int4 experts need rows of whole 8-value words");
                w.layout = b_->weight_layout(as, rows, cols);
                std::vector<uint8_t> q(size_t(int4g_bytes(rows, cols)));
                std::vector<float> s(size_t(int4g_scales(rows, cols)));
                if (qc_replay_) {
                    qc_read(q.data(), q.size()), qc_read(s.data(), s.size() * 4);
                } else {
                    if (scale) {
                        pack_int4_values(static_cast<const int8_t *>(data), rows, cols, q.data(), w.layout);
                        std::memcpy(s.data(), scale, s.size() * 4);
                    } else {
                        quantize_int4_groups(static_cast<const uint16_t *>(data), rows, cols, q.data(), s.data(),
                                             w.layout);
                    }
                    qc_write(q.data(), q.size()), qc_write(s.data(), s.size() * 4);
                }
                w.data = take_gx(q.size());
                b_->upload(const_cast<void *>(w.data), q.data(), q.size());
                w.scale = static_cast<const float *>(take_gx(s.size() * 4));
                b_->upload(const_cast<float *>(w.scale), s.data(), s.size() * 4);
                b_->sync();
                return w;
            }
            if (fb) {
                std::vector<uint8_t> q(size_t(rows) * cols);
                std::vector<float> s(size_t(fp8b_scales(rows, cols)));
                if (qc_replay_) {
                    qc_read(q.data(), q.size()), qc_read(s.data(), s.size() * 4);
                } else {
                    quantize_fp8_blocks(static_cast<const uint16_t *>(data), rows, cols, q.data(), s.data());
                    qc_write(q.data(), q.size()), qc_write(s.data(), s.size() * 4);
                }
                w.data = take_gx(q.size());
                b_->upload(const_cast<void *>(w.data), q.data(), q.size());
                w.scale = static_cast<const float *>(take_gx(s.size() * 4));
                b_->upload(const_cast<float *>(w.scale), s.data(), s.size() * 4);
                b_->sync();  // q and s go out of scope
                return w;
            }
            const size_t bytes = size_t(rows) * cols * (q8 ? 1 : 2);
            w.data = take_gx(bytes);
            b_->upload(const_cast<void *>(w.data), data, bytes);
            if (q8) {
                if (!scale) throw std::runtime_error("8-bit expert without row scales");
                w.scale = static_cast<const float *>(take_gx(size_t(rows) * 4));
                b_->upload(const_cast<float *>(w.scale), scale, size_t(rows) * 4);
            }
            return w;
        };
        const int64_t E = d_.hidden, I = d_.moe_inter;
        if (qc_ && qc_count_ != int64_t(layer) * d_.n_experts + e) {
            if (qc_replay_) throw std::runtime_error("expert cache: experts given out of (layer, expert) order");
            qc_drop();  // recording needs the replay order: give up on it
        }
        if (qc_replay_) {  // the stored bytes, in the order put() recorded them
            if (b_->moe_gate_up_interleaved(gpu_fmt_, E, I)) {
                slot = Backend::ExpertW{put(nullptr, nullptr, 2 * I, E, gpu_fmt_), Weight{}, put(nullptr, nullptr, E, I, gpu_fmt_)};
                if (gx8_on_)
                    gx_host8_[layer][e] = Backend::ExpertW{put(nullptr, nullptr, 2 * I, E, WFmt::FP8B), Weight{},
                                                           put(nullptr, nullptr, E, I, WFmt::FP8B)};
            } else {
                slot = Backend::ExpertW{put(nullptr, nullptr, I, E, gpu_fmt_), put(nullptr, nullptr, I, E, gpu_fmt_),
                                        put(nullptr, nullptr, E, I, gpu_fmt_)};
            }
        } else if (given4 && b_->moe_gate_up_interleaved(gpu_fmt_, E, I)) {
            // pre-quantised int4: gate and up rows interleaved as below, each row's group scales
            // with it ([group][row]); the fp8b copies from the values times their scales
            const int64_t G = int4g_groups(E);
            std::vector<int8_t> q13(2 * size_t(I) * E);
            std::vector<float> s13(size_t(G) * 2 * I);
            for (int64_t r = 0; r < I; ++r) {
                std::memcpy(q13.data() + (2 * r) * E, static_cast<const int8_t *>(w1) + r * E, size_t(E));
                std::memcpy(q13.data() + (2 * r + 1) * E, static_cast<const int8_t *>(w3) + r * E, size_t(E));
                for (int64_t g = 0; g < G; ++g)
                    s13[g * 2 * I + 2 * r] = s1[g * I + r], s13[g * 2 * I + 2 * r + 1] = s3[g * I + r];
            }
            slot = Backend::ExpertW{put(q13.data(), s13.data(), 2 * I, E, gpu_fmt_), Weight{}, put(w2, s2, E, I, gpu_fmt_)};
            if (gx8_on_) {
                std::vector<uint16_t> b13(q13.size()), b2(size_t(E) * I);
                dequant_int4_values_bf16(q13.data(), s13.data(), 2 * I, E, b13.data());
                dequant_int4_values_bf16(static_cast<const int8_t *>(w2), s2, E, I, b2.data());
                gx_host8_[layer][e] = Backend::ExpertW{put(b13.data(), nullptr, 2 * I, E, WFmt::FP8B), Weight{},
                                                       put(b2.data(), nullptr, E, I, WFmt::FP8B)};
            }
        } else if (b_->moe_gate_up_interleaved(gpu_fmt_, E, I)) {
            const size_t row = size_t(E) * (q8 ? 1 : 2);
            std::vector<uint8_t> w13(2 * size_t(I) * row);
            std::vector<float> s13(q8 ? 2 * size_t(I) : 0);
            for (int64_t r = 0; r < I; ++r) {
                std::memcpy(w13.data() + (2 * r) * row, static_cast<const uint8_t *>(w1) + r * row, row);
                std::memcpy(w13.data() + (2 * r + 1) * row, static_cast<const uint8_t *>(w3) + r * row, row);
                if (q8) s13[2 * r] = s1[r], s13[2 * r + 1] = s3[r];
            }
            slot = Backend::ExpertW{put(w13.data(), q8 ? s13.data() : nullptr, 2 * I, E, gpu_fmt_), Weight{},
                                    put(w2, s2, E, I, gpu_fmt_)};
            if (gx8_on_)  // the same interleaved layout, fp8b
                gx_host8_[layer][e] = Backend::ExpertW{put(w13.data(), nullptr, 2 * I, E, WFmt::FP8B), Weight{},
                                                       put(w2, nullptr, E, I, WFmt::FP8B)};
        } else {
            slot = Backend::ExpertW{put(w1, s1, I, E, gpu_fmt_), put(w3, s3, I, E, gpu_fmt_), put(w2, s2, E, I, gpu_fmt_)};
        }
        if (qc_) ++qc_count_;
        b_->sync();  // the caller frees its copies on return
    }

    // The quantised-expert cache. Quantising every expert to int4 (a 7-way clip search per
    // 128-value group) or fp8b at load takes ~2 minutes for Qwen3-30B-A3B; recorded once to a file
    // (each put()'s bytes in call order after a header of dims, format and version) it is replayed
    // on later loads, and the loader skips reading the experts. The loader names the file after the
    // model's files and the options; the header guards the rest.
    static constexpr uint32_t kQcVersion = 2;  // bump when int4_group.h, fp8_block.h or the interleave change
    struct QcHeader {
        char magic[8];
        uint32_t version, fmt, gx8, il, layout;  // layout: the backend's packing of the experts' matrices
        int32_t n_layers, n_experts, hidden, inter;
        uint64_t record;  // bytes per expert
    };
    QcHeader qc_header() const {
        QcHeader h;
        std::memset(&h, 0, sizeof h);  // padding too: headers compare bytewise
        std::memcpy(h.magic, "fullmQC", 8);
        const int64_t E = d_.hidden, I = d_.moe_inter;
        const bool il = b_->moe_gate_up_interleaved(gpu_fmt_, E, I);
        h.version = kQcVersion, h.fmt = uint32_t(gpu_fmt_), h.gx8 = gx8_on_, h.il = il;
        h.layout = uint32_t(b_->weight_layout(gpu_fmt_, il ? 2 * I : I, E)) | uint32_t(b_->weight_layout(gpu_fmt_, E, I)) << 8;
        h.n_layers = d_.n_layers, h.n_experts = d_.n_experts, h.hidden = d_.hidden, h.inter = d_.moe_inter;
        auto m = [](int64_t rows, int64_t cols, WFmt f) -> uint64_t {
            if (f == WFmt::INT4G) return uint64_t(int4g_bytes(rows, cols) + int4g_scales(rows, cols) * 4);
            return uint64_t(rows * cols + fp8b_scales(rows, cols) * 4);
        };
        h.record = il ? m(2 * I, E, gpu_fmt_) + m(E, I, gpu_fmt_) : 2 * m(I, E, gpu_fmt_) + m(E, I, gpu_fmt_);
        if (il && gx8_on_) h.record += m(2 * I, E, WFmt::FP8B) + m(E, I, WFmt::FP8B);
        return h;
    }
    int expert_cache(const char *path) override {
        if (!gx_arena_ || qc_ || qc_count_) throw std::runtime_error("expert_cache: after reserve_gpu_experts, once");
        if (gpu_fmt_ != WFmt::INT4G && gpu_fmt_ != WFmt::FP8B) return -1;  // nothing quantised here
        const QcHeader want = qc_header();
        const uint64_t size = sizeof want + uint64_t(d_.n_layers) * d_.n_experts * want.record;
        if (FILE *f = std::fopen(path, "rb")) {
            QcHeader h;
            const bool ok = std::fread(&h, sizeof h, 1, f) == 1 && std::memcmp(&h, &want, sizeof h) == 0 &&
                            std::fseek(f, 0, SEEK_END) == 0 && uint64_t(std::ftell(f)) == size &&
                            std::fseek(f, long(sizeof h), SEEK_SET) == 0;
            if (ok) {
                qc_ = f, qc_replay_ = true;
                return 1;
            }
            std::fclose(f);
        }
        qc_path_ = path;
        qc_ = std::fopen((qc_path_ + ".tmp").c_str(), "wb");
        if (!qc_ || std::fwrite(&want, sizeof want, 1, qc_) != 1) {
            qc_drop();
            return -1;
        }
        return 0;
    }
    void qc_read(void *dst, size_t n) {
        if (std::fread(dst, 1, n, qc_) != n) throw std::runtime_error("expert cache: short read");
    }
    void qc_write(const void *src, size_t n) {
        if (qc_ && !qc_replay_ && std::fwrite(src, 1, n, qc_) != n) qc_drop();
    }
    void qc_drop() {  // stop recording (a write failed, or the order is not the replay order)
        if (qc_) std::fclose(qc_);
        qc_ = nullptr;
        if (!qc_path_.empty()) std::remove((qc_path_ + ".tmp").c_str());
        qc_path_.clear();
    }
    void qc_close() {  // finalize: a complete recording becomes the cache file
        if (!qc_) return;
        const bool done = qc_count_ == int64_t(d_.n_layers) * d_.n_experts;
        const bool rec = !qc_replay_;
        std::fclose(qc_);
        qc_ = nullptr;
        if (rec && done) std::rename((qc_path_ + ".tmp").c_str(), qc_path_.c_str());
        else if (rec) std::remove((qc_path_ + ".tmp").c_str());
    }

    size_t pool_bytes() const override { return 0; }

    void finalize() override {
        if (finalized_) throw std::runtime_error("finalize twice");
        qc_close();
        const int64_t E = d_.hidden, Q = int64_t(d_.n_heads) * d_.head_dim, KV = int64_t(d_.n_kv_heads) * d_.head_dim;
        std::vector<std::string> missing;
        auto mat = [&](const std::string &n, int64_t rows, int64_t cols) {
            auto it = mats_.find(n);
            if (it == mats_.end() || it->second.rows != rows || it->second.cols != cols) {
                missing.push_back(n + " [" + std::to_string(rows) + "][" + std::to_string(cols) + "]");
                return Weight{};
            }
            return it->second;
        };
        auto vec = [&](const std::string &n, int64_t len) -> const float * {
            auto it = vecs_.find(n);
            if (it == vecs_.end() || it->second.second != len) {
                missing.push_back(n + " [" + std::to_string(len) + "]");
                return nullptr;
            }
            return it->second.first;
        };
        if (!embed_.data || embed_.rows != d_.vocab || embed_.cols != E) missing.push_back("embed");
        lm_head_ = mat("lm_head", d_.vocab, E);
        final_norm_ = vec("final_norm", E);
        layers_.resize(d_.n_layers);
        for (int l = 0; l < d_.n_layers; ++l) {
            const std::string p = "L" + std::to_string(l) + ".";
            DenseLayer &L = layers_[l];
            L.in_norm = vec(p + "in_norm", E);
            L.post_norm = vec(p + "post_norm", E);
            if (d_.qk_norm) {
                L.q_norm = vec(p + "q_norm", d_.head_dim);
                L.k_norm = vec(p + "k_norm", d_.head_dim);
            }
            L.q = mat(p + "q", Q, E);
            L.k = mat(p + "k", KV, E);
            L.v = mat(p + "v", KV, E);
            L.o = mat(p + "o", E, Q);
            if (d_.n_experts > 0) {
                // no router where every token takes every expert (a dense MLP split into experts):
                // the weights are all 1 (see moe)
                if (d_.topk != d_.n_experts || mats_.count(p + "router")) L.router = mat(p + "router", d_.n_experts, E);
                for (int e = 0; e < d_.n_experts; ++e)
                    if (gx_host_.empty() || !gx_host_[l][e].w1.data) {
                        missing.push_back(p + "expert " + std::to_string(e) + " (every expert must be resident)");
                        break;
                    }
            } else {
                L.gate = mat(p + "gate", d_.inter, E);
                L.up = mat(p + "up", d_.inter, E);
                L.down = mat(p + "down", E, d_.inter);
            }
        }
        if (eagle_) {
            const int64_t I = ed_.inter, Dv = ed_.draft_vocab;
            eg_fc_ = mat("eagle.fc", E, 3 * E);
            eg_.q = mat("eagle.q", Q, 2 * E);
            eg_.k = mat("eagle.k", KV, 2 * E);
            eg_.v = mat("eagle.v", KV, 2 * E);
            eg_.o = mat("eagle.o", E, Q);
            eg_.gate = mat("eagle.gate", I, E);
            eg_.up = mat("eagle.up", I, E);
            eg_.down = mat("eagle.down", E, I);
            // a head over the whole vocabulary without its own lm_head reads the model's (as vLLM shares it)
            eg_head_ = Dv == d_.vocab && !mats_.count("eagle.lm_head") ? lm_head_ : mat("eagle.lm_head", Dv, E);
            eg_.in_norm = vec("eagle.in_norm", E);
            eg_hidden_norm_ = vec("eagle.hidden_norm", E);
            eg_.post_norm = vec("eagle.post_norm", E);
            eg_norm_ = vec("eagle.norm", E);
            vec("eagle.d2t", Dv);
            for (int32_t t : eg_d2t_)
                if (t < 0 || t >= d_.vocab) missing.push_back("eagle.d2t (a target id outside the vocabulary)");
            if (int64_t(eg_d2t_.size()) == Dv) {
                d_d2t_ = static_cast<int32_t *>(b_->alloc(size_t(Dv) * 4));
                owned_.push_back(d_d2t_);
                b_->upload(d_d2t_, eg_d2t_.data(), size_t(Dv) * 4);
                b_->sync();
            }
        }
        if (!missing.empty()) {
            std::string msg = "executor is missing or has mis-shaped tensors (" + std::to_string(missing.size()) + "):";
            for (size_t i = 0; i < missing.size() && i < 12; ++i) msg += "\n  " + missing[i];
            throw std::runtime_error(msg);
        }
        // each layer's q, k and v, and a dense layer's gate and up, back to back in one allocation
        // each (moved, not copied twice; bf16, or fp8 with their row scales back to back too): the
        // device backend then runs the projections of a step on cuBLAS as one GEMM
        if (!b_->on_host()) {
            using Part = std::pair<const char *, Weight *>;
            auto pack = [&](const std::string &p, std::initializer_list<Part> parts) {
                const WFmt f = parts.begin()->second->fmt;
                if (f != WFmt::BF16 && f != WFmt::FP8) return;
                const size_t es = f == WFmt::BF16 ? 2 : 1;
                size_t total = 0, srows = 0;
                for (const Part &w : parts) {
                    if (w.second->fmt != f || (f == WFmt::FP8 && !w.second->scale)) return;
                    total += es * w.second->rows * w.second->cols;
                    srows += size_t(w.second->rows);
                }
                char *blk = static_cast<char *>(b_->alloc(total)), *at = blk;
                float *sblk = f == WFmt::FP8 ? static_cast<float *>(b_->alloc(srows * 4)) : nullptr, *sat = sblk;
                for (const Part &w : parts) {
                    const size_t b = es * w.second->rows * w.second->cols;
                    b_->copy(at, w.second->data, b);
                    at += b;
                    if (sblk) b_->copy(sat, w.second->scale, size_t(w.second->rows) * 4), sat += w.second->rows;
                }
                b_->sync();
                at = blk, sat = sblk;
                for (const Part &w : parts) {
                    Weight &m = mats_[p + w.first];
                    b_->release(const_cast<void *>(m.data));
                    m.data = nullptr;  // the layer's slices below; the blocks are owned_
                    w.second->data = at;
                    at += es * w.second->rows * w.second->cols;
                    if (sblk) {
                        b_->release(const_cast<float *>(m.scale));
                        m.scale = nullptr;
                        w.second->scale = sat;
                        sat += w.second->rows;
                    }
                }
                owned_.push_back(blk);
                if (sblk) owned_.push_back(sblk);
            };
            for (int l = 0; l < d_.n_layers; ++l) {
                const std::string p = "L" + std::to_string(l) + ".";
                DenseLayer &L = layers_[l];
                pack(p, {{"q", &L.q}, {"k", &L.k}, {"v", &L.v}});
                if (d_.n_experts == 0) pack(p, {{"gate", &L.gate}, {"up", &L.up}});
            }
        }
        alloc_pools();
        for (int l = 0; l < d_.n_layers && d_.n_experts > 0; ++l) {
            const size_t bytes = gx_host_[l].size() * sizeof(Backend::ExpertW);
            auto *t = static_cast<Backend::ExpertW *>(b_->alloc(bytes));
            owned_.push_back(t);
            b_->upload(t, gx_host_[l].data(), bytes);
            gx_dev_.push_back(t);
            if (gx8_on_) {
                auto *t8 = static_cast<Backend::ExpertW *>(b_->alloc(bytes));
                owned_.push_back(t8);
                b_->upload(t8, gx_host8_[l].data(), bytes);
                gx_dev8_.push_back(t8);
            }
        }
        graphs_on_ = o_.no_graphs == 0;
        // decode steps (every chunk one token, its row's logits) replay graphs up to this many
        // rows: their attention is the decode kernel's either way (FULLM_DECODE_GRAPH_ROWS)
        if (const char *e = std::getenv("FULLM_DECODE_GRAPH_ROWS")) decode_graph_rows_ = std::atoll(e);
        decode_graph_rows_ = std::min<int64_t>({decode_graph_rows_, o_.max_tokens, o_.max_rows});
        b_->sync();
        finalized_ = true;
    }

    // ------------------------------------------------------------------ the step
    void step(const sk_step_plan &pl, float *logits_out, sk_step_stats *st, int32_t *tokens_out,
              const sk_sample_row *sample, const sk_sample_io *io) override {
        if (!finalized_) throw std::runtime_error("step before finalize");
        const auto t_start = Clock::now();
        sk_step_stats stats{};
        const int64_t N = pl.n_tokens, R = pl.n_rows, E = d_.hidden;
        validate(pl);
        last_n_ = N;

        auto t = Clock::now();
        b_->upload(d_pos_, pl.positions, N * 4);
        b_->upload(d_tslot_, pl.token_slot, N * 4);
        b_->upload(d_pt_, pl.page_table, size_t(o_.n_slots) * o_.max_pages_per_seq * 4);
        lora_step_ = false;
        if (lora_slots_ > 0) {  // each row's adapter: its sequence's
            row_lora_.resize(size_t(N));
            for (int64_t i = 0; i < N; ++i) {
                row_lora_[size_t(i)] = seq_lora_[size_t(pl.token_slot[i])];
                lora_step_ |= row_lora_[size_t(i)] >= 0;
            }
            if (lora_step_) b_->upload(d_row_lora_, row_lora_.data(), N * 4);
        }
        ws_->reset();
        float *h = ws_->take<float>(N * E), *hin = ws_->take<float>(N * E), *tmp = ws_->take<float>(N * E);
        if (embed_dev_) {
            b_->upload(d_tok_, pl.tokens, N * 4);
            b_->embed_rows(h, embed_, d_tok_, N, E);
        } else {
            gather_embedding(pl.tokens, N);
            b_->upload(h, h_host_, N * E * 4);
        }
        stats.ms_upload += ms_since(t);
        const size_t sub = ws_->mark();
        // small steps replay a CUDA graph of forward(); attention is then launched for the
        // longest context so the launch does not follow positions (see K3Exec::step)
        // a decode step: every chunk one token, every row's logits in order (a verify chunk's rows
        // attend as one query tile outside a graph, as decode rows inside one: not graphed above 16)
        bool decode_step = R == N && pl.n_segs == N;
        for (int64_t i = 0; decode_step && i < R; ++i) decode_step = pl.rows[i] == i;
        const bool graphable = graphs_on_ && (N <= kGemvMaxRows || (decode_step && N <= decode_graph_rows_));
        lora_runs(N, graphable);
        int64_t max_pos = 0;
        if (graphable) max_pos = int64_t(o_.max_pages_per_seq) * o_.page - 1;
        else
            for (int64_t i = 0; i < N; ++i) max_pos = std::max<int64_t>(max_pos, pl.positions[i]);
        // the step's attention layout: rows [0, n_dec_) are single tokens, each attending to its
        // own context (every row of a small step, graphed or not: exact for prompt rows too, as
        // the step writes their keys first); a larger step adds its prompt chunks as query tiles
        // and segments
        layout(pl.positions, pl.segs, pl.n_segs, N, graphable);
        if (!graphable) {
            forward(pl, h, hin, tmp, sub, max_pos);
        } else {
            const GraphKey key{N, std::vector<int32_t>(pl.rows, pl.rows + R), dec_fa3_, lora_step_ ? lora_active_ : 0};
            auto it = graphs_.find(key);
            if (it == graphs_.end()) {
                forward(pl, h, hin, tmp, sub, max_pos);  // first sighting: sizes every scratch buffer
                if (graphs_.size() < 1024) graphs_.emplace(key, nullptr);
            } else if (!it->second) {
                bool recorded = false;
                if (b_->graph_begin()) {
                    try {
                        forward(pl, h, hin, tmp, sub, max_pos);
                        it->second = b_->graph_end();
                        recorded = true;
                    } catch (const std::exception &ex) {
                        b_->graph_abort();  // nothing recorded ran: run the step op by op
                        if (capture_failed_.insert(N).second)
                            std::fprintf(stderr, "fullm: a step of %lld tokens runs op by op (graph capture failed: %s)\n",
                                         static_cast<long long>(N), ex.what());
                    }
                }
                if (recorded) b_->graph_launch(it->second);
                else forward(pl, h, hin, tmp, sub, max_pos);
            } else {
                b_->graph_launch(it->second);
            }
        }

        t = Clock::now();
        if (tokens_out) {
            step_tokens(b_.get(), logits_d_, R, d_.vocab, sample, io, d_argmax_, d_sample_, sample_scratch_, argmax_host_,
                        tokens_out);
        } else {
            b_->download(logits_host_, logits_d_, size_t(R) * d_.vocab * 4);
            b_->sync();
            std::memcpy(logits_out, logits_host_, size_t(R) * d_.vocab * 4);
        }
        stats.ms_download += ms_since(t);
        stats.ms_total = ms_since(t_start);
        if (st) *st = stats;
    }

    // EAGLE-3: consume each sequence's accepted rows (token t(j+1) at position j with the hidden
    // the last step tapped at its row), then draft k tokens one row a sequence at a time. The head:
    // h = fc(taps); x = [in_norm(embed(token)) | hidden_norm(h)]; attention over its own cache
    // (pool n_layers, RoPE, no q/k norm); h += o; h += mlp(post_norm(h)); the draft is the argmax
    // of lm_head(norm(h)), mapped to a target id; the next step's hidden is that h, before norm.
    void draft(const sk_draft_plan &dp, int32_t k_max, int32_t *drafts) override {
        if (!eagle_) throw std::runtime_error("no EAGLE-3 head on this executor");
        if (!finalized_) throw std::runtime_error("draft before finalize");
        const int64_t S = dp.n_seqs, E = d_.hidden, Dv = ed_.draft_vocab;
        if (S <= 0) return;
        if (S > o_.n_slots || S > o_.max_rows || k_max < 0) throw std::runtime_error("bad draft plan: sequences");
        auto fail = [](const std::string &m) { throw std::runtime_error("bad draft plan: " + m); };
        const int64_t cap = int64_t(o_.max_pages_per_seq) * o_.page;
        auto paged = [&](int32_t slot, int64_t p) {
            if (p < 0 || p >= cap) return false;
            const int32_t pg = dp.page_table[int64_t(slot) * o_.max_pages_per_seq + p / o_.page];
            return pg >= 0 && pg < o_.n_pages;
        };
        std::vector<int32_t> tok, pos, slot, rows, segs;
        std::vector<int64_t> last(static_cast<size_t>(S));
        int32_t kk = 0;
        for (int64_t s2 = 0, m = 0; s2 < S; ++s2) {
            const int32_t n = dp.n_in[s2], k = dp.k[s2], sl = dp.slot[s2];
            if (n < 1 || k < 0 || k > k_max) fail("n_in or k");
            if (sl < 0 || sl >= o_.n_slots) fail("slot out of range");
            segs.insert(segs.end(), {sl, int32_t(m), n});
            for (int32_t i = 0; i < n; ++i, ++m) {
                const int32_t t = dp.tokens[m], r = dp.rows[m];
                if (t < 0 || t >= d_.vocab) fail("token id out of vocabulary");
                if (r < 0 || r >= last_n_) fail("tapped row outside the last step");
                if (!paged(sl, int64_t(dp.pos0[s2]) + i)) fail("a consumed position has no page");
                tok.push_back(t), pos.push_back(dp.pos0[s2] + i), slot.push_back(sl), rows.push_back(r);
            }
            for (int32_t i = 1; i < k; ++i)
                if (!paged(sl, int64_t(dp.pos0[s2]) + n - 1 + i)) fail("a drafted position has no page");
            last[size_t(s2)] = m - 1;
            kk = std::max(kk, k);
        }
        const int64_t M = int64_t(tok.size());
        if (M > o_.max_tokens) fail("more consumed tokens than a step holds");
        b_->upload(d_pt_, dp.page_table, size_t(o_.n_slots) * o_.max_pages_per_seq * 4);
        ws_->reset();
        float *hl = ws_->take<float>(S * E);  // each sequence's last hidden (before norm)
        float *post = ws_->take<float>(S * E), *lg = ws_->take<float>(S * Dv);
        const size_t base = ws_->mark();

        // the rows consumed: fused taps as hidden, the tokens' embeddings, the layer
        float *x3 = ws_->take<float>(M * 3 * E), *h = ws_->take<float>(M * E), *emb = ws_->take<float>(M * E);
        b_->upload(d_rows_, rows.data(), M * 4);
        b_->embed_rows(x3, aux_w_, d_rows_, M, 3 * E);
        b_->gemm(eg_fc_, x3, M, h, E);
        embed(tok.data(), M, emb);
        eagle_layer(emb, h, M, pos.data(), slot.data(), segs.data(), int32_t(S));
        for (int64_t s2 = 0; s2 < S; ++s2) b_->copy(hl + s2 * E, h + last[size_t(s2)] * E, E * 4);
        // reading only: the work stays queued on the stream (the next step follows it there)
        if (kk == 0) return;

        // Drafting, the drafts kept on the device from step to step and downloaded once: the
        // sequences drafting more than one token first (B of them; every one runs the kk - 1
        // steps, its rows past its k held at its last drafted position, whose cache the next
        // consumption writes again, their drafts dropped), then those drafting one.
        std::vector<int64_t> act;
        for (int64_t s2 = 0; s2 < S; ++s2)
            if (dp.k[s2] > 1) act.push_back(s2);
        const int64_t B = int64_t(act.size());
        for (int64_t s2 = 0; s2 < S; ++s2)
            if (dp.k[s2] == 1) act.push_back(s2);
        const int64_t A = int64_t(act.size());
        ws_->reset(base);
        int32_t *ids = ws_->take<int32_t>(int64_t(kk) * A), *am = ws_->take<int32_t>(A + 1);  // [kk][A]; argmax, bad
        float *hd = ws_->take<float>(A * E), *e1 = ws_->take<float>(A * E);
        const size_t base2 = ws_->mark();
        b_->zero(am + A, 4);
        for (int64_t a = 0; a < A; ++a) b_->copy(hd + a * E, hl + act[size_t(a)] * E, E * 4);
        auto head = [&](int64_t n, int32_t i) {  // drafts i of the first n rows of hd, as target ids
            b_->rmsnorm(post, hd, eg_norm_, n, E, E, d_.rms_eps);
            b_->gemm(eg_head_, post, n, lg, Dv);
            b_->argmax_rows(am, am + A, lg, n, Dv);
            b_->gather_ids(ids + int64_t(i) * A, d_d2t_, am, n);
        };
        // the embeddings of device ids: gathered there (or from the host table on a host backend);
        // a device backend holding the table in host memory takes the ids back first
        std::vector<int32_t> tmp_ids;
        auto embed_ids = [&](float *dst, const int32_t *dev_ids, int64_t n) {
            if (embed_dev_ || b_->on_host()) {
                b_->embed_rows(dst, embed_, dev_ids, n, E);
            } else {
                tmp_ids.resize(static_cast<size_t>(n));
                b_->download(tmp_ids.data(), dev_ids, size_t(n) * 4);
                b_->sync();
                embed(tmp_ids.data(), n, dst);
            }
        };
        head(A, 0);
        std::vector<int32_t> p1(static_cast<size_t>(B)), sl1(static_cast<size_t>(B)), sg1;
        for (int64_t a = 0; a < B; ++a) {
            sl1[size_t(a)] = dp.slot[act[size_t(a)]];
            sg1.insert(sg1.end(), {sl1[size_t(a)], int32_t(a), 1});
        }
        for (int32_t i = 1; i < kk && B > 0; ++i) {
            ws_->reset(base2);
            for (int64_t a = 0; a < B; ++a) {
                const int64_t s2 = act[size_t(a)];
                p1[size_t(a)] = dp.pos0[s2] + dp.n_in[s2] - 1 + std::min<int32_t>(i, dp.k[s2] - 1);
            }
            embed_ids(e1, ids + int64_t(i - 1) * A, B);
            eagle_layer(e1, hd, B, p1.data(), sl1.data(), sg1.data(), int32_t(B));
            head(B, i);
        }
        std::vector<int32_t> got(static_cast<size_t>(kk) * A + 1);
        b_->download(got.data(), ids, size_t(kk) * A * 4);
        b_->download(got.data() + size_t(kk) * A, am + A, 4);
        b_->sync();
        if (got[size_t(kk) * A]) throw std::runtime_error("non-finite draft logits");
        for (int64_t a = 0; a < A; ++a) {
            const int64_t s2 = act[size_t(a)];
            for (int32_t i = 0; i < dp.k[s2]; ++i) drafts[s2 * k_max + i] = got[size_t(i) * A + size_t(a)];
        }
    }

    // The KV cache is paged and positions overwrite it: a sequence has no per-slot state.
    size_t slot_bytes() const override { return 0; }
    void save_slot(int32_t, void *) override {}
    void load_slot(int32_t, const void *) override {}

    // the host tier: a host page is one cache page of every layer's K and V pool
    size_t page_bytes() const override { return host_.page_bytes(); }
    void reserve_host_pages(int64_t n) override { host_.reserve(n); }
    uint64_t copy_pages(bool to_host, const int32_t *dev, const int32_t *host, int64_t n) override {
        return host_.copy(to_host, dev, host, n);
    }
    bool copy_done(uint64_t t) override { return host_.done(t); }
    void copy_order(uint64_t t) override { host_.order(t); }
    void copy_wait(uint64_t t) override { host_.wait(t); }

    const char *describe() override {
        std::ostringstream os;
        const double gb = 1e9;
        os << "dense decoder on " << b_->name() << ": weights " << double(weight_bytes_) / gb << " GB, KV cache "
           << double(kv_bytes() * 2 * d_.n_layers) / gb << " GB (" << (int64_t(o_.n_pages) * o_.page)
           << " positions, "
           << (kv_dt_ == DType::KV4 ? "int4" : kv_dt_ == DType::FP8E4M3 ? "fp8 e4m3" : kv_dt_ == DType::FP8 ? "fp8, per-token scales" : kv_dt_ == DType::BF16 ? "bf16" : "f32")
           << "), "
           << d_.n_heads << "/"
           << d_.n_kv_heads << " heads x " << d_.head_dim << (d_.qk_norm ? ", q/k norm" : "");
        if (d_.n_experts > 0)
            os << ", " << d_.n_experts << " experts top-" << d_.topk << " resident ("
               << (gpu_fmt_ == WFmt::BF16   ? "bf16"
                   : gpu_fmt_ == WFmt::INT8  ? "int8"
                   : gpu_fmt_ == WFmt::INT4G ? (gx8_on_                                            ? "int4, fp8b copies for prefill"
                                                : b_->int4_prefill_gemm(d_.hidden, d_.moe_inter) ? "int4, int4 GEMM for prefill"
                                                                                                 : "int4")
                   : gpu_fmt_ == WFmt::FP8B  ? "fp8b"
                                             : "fp8")
               << ", "
               << double(gx_used_) / gb << " GB)";
        if (graphs_on_ && !b_->on_host())
            os << ", CUDA graphs for steps of <= " << kGemvMaxRows << " tokens and decode steps of <= " << decode_graph_rows_;
        desc_ = os.str();
        return desc_.c_str();
    }

private:
    // The attention layout of N rows (positions, segs [n_segs][3] as a step plan's): rows
    // [0, n_dec_) single tokens, each attending to its own context (every row of a small step,
    // graphed or not: exact for prompt rows too, as the step writes their keys first); a larger
    // step adds its prompt chunks as query tiles and segments.
    void layout(const int32_t *positions, const int32_t *segs, int32_t n_segs, int64_t N, bool graphable) {
        n_tiles_ = 0, n_dec_ = N;
        h_segs_.clear();
        if (!graphable) {
            if (N > kGemvMaxRows) {
                std::vector<int32_t> tiles;
                n_dec_ = 0;
                for (int32_t s = 0; s < n_segs; ++s) {
                    const int32_t first = segs[3 * s + 1], count = segs[3 * s + 2];
                    // the decoding sequences' one-token chunks come first: the decode kernel
                    // takes them as a batch (an FMHA launch per chunk, each padded to 8 queries
                    // over a gathered copy of its whole context, cost a mixed step ~35 ms)
                    if (count == 1 && first == n_dec_) {
                        ++n_dec_;
                        continue;
                    }
                    for (int32_t i = 0; i < count; i += 16) tiles.push_back(first + i), tiles.push_back(std::min(16, count - i));
                    h_segs_.insert(h_segs_.end(), {first, count, positions[first], segs[3 * s]});
                }
                n_tiles_ = int64_t(tiles.size() / 2);
                if (n_tiles_) b_->upload(d_tiles_, tiles.data(), tiles.size() * 4);
            }
        }
        // the decoding rows through the backend's prefill form (FlashAttention-3 with split
        // keys) when they are many, or several and long: it loses at one to four short rows,
        // where its scheduler, combine and conversions cost more than the decode kernel's whole
        // attention (-1-2%), and wins from 16 rows (H100, 300-token replies, c = 16 / 32:
        // Qwen3-8B +5 / +8%, Qwen3-30B-A3B bf16 +5 / +4%) and from 4 rows of ~1k positions each
        // (2000-token replies: c = 4 +4%, c = 16 +15% / +8%). FULLM_DECODE_ATTN=fa3 | tc forces
        // either.
        dec_fa3_ = false;
        if (n_dec_ > 0) {
            static const int mode = [] {
                const char *e = std::getenv("FULLM_DECODE_ATTN");
                return !e ? 0 : std::string(e) == "fa3" ? 1 : std::string(e) == "tc" ? 2 : 0;
            }();
            int64_t ctx = 0;
            for (int64_t i = 0; i < n_dec_; ++i) ctx += positions[i] + 1;
            dec_fa3_ = mode == 1 || (mode == 0 && (n_dec_ >= 16 || (n_dec_ >= 4 && ctx >= 4096)));
            if (kv_dt_ == DType::FP8) dec_fa3_ = false;  // FA3 reads bf16: the fp8 decode kernel takes the rows
            if (kv_dt_ == DType::FP8E4M3) dec_fa3_ = true;  // only FA3 reads plain e4m3 pages
        }
    }

    // Embeddings of host token ids into dst ([n][hidden], backend memory), as a step takes them.
    void embed(const int32_t *tokens, int64_t n, float *dst) {
        const int64_t E = d_.hidden;
        if (embed_dev_) {
            b_->upload(d_tok_, tokens, n * 4);
            b_->embed_rows(dst, embed_, d_tok_, n, E);
        } else {
            gather_embedding(tokens, n);
            b_->upload(dst, h_host_, n * E * 4);
        }
    }

    // The EAGLE-3 layer on n rows (positions, slots, segs [n_segs][3] of them): emb, the tokens'
    // embeddings; h, the hidden in, left holding the layer's output before the final norm.
    void eagle_layer(const float *emb, float *h, int64_t n, const int32_t *pos, const int32_t *slot,
                     const int32_t *segs, int32_t n_segs) {
        const int64_t E = d_.hidden, I = ed_.inter;
        b_->upload(d_pos_, pos, n * 4);
        b_->upload(d_tslot_, slot, n * 4);
        int64_t max_pos = 0;
        for (int64_t r = 0; r < n; ++r) max_pos = std::max<int64_t>(max_pos, pos[r]);
        layout(pos, segs, n_segs, n, false);
        float *en = ws_->take<float>(n * E), *hn = ws_->take<float>(n * E), *x2 = ws_->take<float>(n * 2 * E);
        float *tmp = ws_->take<float>(n * E), *hin = ws_->take<float>(n * E);
        b_->rmsnorm(en, emb, eg_.in_norm, n, E, E, d_.rms_eps);
        b_->rmsnorm(hn, h, eg_hidden_norm_, n, E, E, d_.rms_eps);
        b_->copy_rows(x2, 2 * E, 0, en, n, E);
        b_->copy_rows(x2, 2 * E, E, hn, n, E);
        const size_t m = ws_->mark();
        attention(eg_, d_.n_layers, x2, n, max_pos, tmp, false);
        ws_->reset(m);
        b_->add_rmsnorm(h, tmp, hin, eg_.post_norm, n, E, d_.rms_eps);
        float *g = ws_->take<float>(n * I), *u = ws_->take<float>(n * I), *a = ws_->take<float>(n * I);
        const Backend::GemmJob in[] = {{&eg_.gate, hin, g, I}, {&eg_.up, hin, u, I}};
        b_->gemm_group(in, 2, n);
        b_->silu_glu(a, g, u, n * I);
        b_->gemm(eg_.down, a, n, tmp, E);
        b_->add(h, tmp, n * E);
    }

    // Every layer and the head on the requested rows (logits_d_); device work only.
    void forward(const sk_step_plan &pl, float *h, float *hin, float *tmp, size_t sub, int64_t max_pos) {
        const int64_t N = pl.n_tokens, R = pl.n_rows, E = d_.hidden;
        for (int l = 0; l < d_.n_layers; ++l) {
            const DenseLayer &L = layers_[l];
            // EAGLE-3: the residual stream entering this layer, for the draft head
            if (eagle_)
                for (int j = 0; j < 3; ++j)
                    if (ed_.taps[j] == l) b_->copy_rows(aux_, 3 * E, j * E, h, N, E);
            if (l == 0) b_->rmsnorm(hin, h, L.in_norm, N, E, E, d_.rms_eps);
            ws_->reset(sub);
            attention(L, l, hin, N, max_pos, tmp, d_.qk_norm != 0);
            b_->add_rmsnorm(h, tmp, hin, L.post_norm, N, E, d_.rms_eps);
            ws_->reset(sub);
            if (const float *contrib = mlp(L, hin, N, tmp)) {  // MoE: reduce straight into the residual
                if (l + 1 < d_.n_layers)
                    b_->moe_reduce_add_rmsnorm(h, contrib, N, d_.topk, E, nullptr, hin, layers_[l + 1].in_norm,
                                               d_.rms_eps, tmp);
                else {
                    b_->moe_reduce(tmp, contrib, N, d_.topk, E);
                    b_->add(h, tmp, N * E);
                }
            } else if (l + 1 < d_.n_layers) {
                b_->add_rmsnorm(h, tmp, hin, layers_[l + 1].in_norm, N, E, d_.rms_eps);
            } else {
                b_->add(h, tmp, N * E);
            }
        }
        bool in_order = R == N;  // every row's logits, in order: the norm reads them in place
        for (int64_t i = 0; in_order && i < R; ++i) in_order = pl.rows[i] == i;
        if (in_order) {
            b_->rmsnorm(fin_, h, final_norm_, R, E, E, d_.rms_eps);
        } else {
            for (int64_t i = 0; i < R; ++i) b_->copy(fin_ + i * E, h + int64_t(pl.rows[i]) * E, E * 4);
            b_->rmsnorm(fin_, fin_, final_norm_, R, E, E, d_.rms_eps);
        }
        b_->gemm(lm_head_, fin_, R, logits_d_, d_.vocab);
    }

    // The layer's attention over pool l (the model's layers, then an EAGLE-3 head's: qk_norm off).
    void attention(const DenseLayer &L, int l, const float *x, int64_t N, int64_t max_pos, float *out, bool qk_norm) {
        const int64_t H = d_.n_heads, Hk = d_.n_kv_heads, D = d_.head_dim, Q = H * D, KV = Hk * D;
        float *q = ws_->take<float>(N * Q), *k = ws_->take<float>(N * KV), *v = ws_->take<float>(N * KV);
        float *o = ws_->take<float>(N * Q);
        // y16_ok: qk_rope_append reads these, so the backend may keep them in its activation precision
        // (not when adapters update them)
        // (an adapter's update goes into them where the backend keeps them)
        const bool lo = lora_step_ && l < d_.n_layers;
        const Backend::GemmJob in[] = {{&L.q, x, q, Q, true}, {&L.k, x, k, KV, true}, {&L.v, x, v, KV, true}};
        b_->gemm_group(in, 3, N);
        if (lo) lora_apply(l, 0, x, {q, k, v}, N);
        if (!b_->qk_rope_append(q, k, v, k_pools_[l], v_pools_[l], kv_dt_, qk_norm ? L.q_norm : nullptr,
                                qk_norm ? L.k_norm : nullptr, d_.rms_eps, d_inv_freq_, d_tslot_, d_pos_, d_pt_,
                                o_.max_pages_per_seq, o_.page, N, H, Hk, D)) {
            if (qk_norm) {  // per head, before RoPE
                b_->rmsnorm(q, q, L.q_norm, N * H, D, D, d_.rms_eps);
                b_->rmsnorm(k, k, L.k_norm, N * Hk, D, D, d_.rms_eps);
            }
            if (kv_dt_ == DType::FP8 || kv_dt_ == DType::FP8E4M3 || kv_dt_ == DType::KV4)
                throw std::runtime_error("fp8 / int4 KV cache: the backend has no fused append");
            b_->rope(q, d_pos_, d_inv_freq_, N, H, D, Q);
            b_->rope(k, d_pos_, d_inv_freq_, N, Hk, D, KV);
            b_->mla_append(k_pools_[l], kv_dt_, k, d_tslot_, d_pos_, d_pt_, o_.max_pages_per_seq, o_.page, KV, N);
            b_->mla_append(v_pools_[l], kv_dt_, v, d_tslot_, d_pos_, d_pt_, o_.max_pages_per_seq, o_.page, KV, N);
        }
        const float scale = 1.0f / std::sqrt(float(D));
        // the backend's prefill form first (FlashAttention-3 takes the chunks and the decoding
        // rows [0, n_dec_) in one launch; the CUTLASS FMHA the chunks only), the decode kernel
        // for whatever it left: the decoding rows, or every row of a step it declined
        const int done = b_->gqa_prefill(o, q, k_pools_[l], v_pools_[l], kv_dt_, d_tiles_, n_tiles_, h_segs_.data(),
                                         int64_t(h_segs_.size() / 4), n_dec_, dec_fa3_, d_tslot_, d_pos_, d_pt_,
                                         o_.max_pages_per_seq, o_.page, H, Hk, D, max_pos, scale);
        if (kv_dt_ == DType::FP8E4M3 && done != 2)
            throw std::runtime_error("fp8 KV cache: FlashAttention-3 (the cutlass feature, head_dim 128) reads it");
        if (done == 0)
            b_->gqa_attention(o, q, k_pools_[l], v_pools_[l], kv_dt_, d_tslot_, d_pos_, d_pt_, o_.max_pages_per_seq,
                              o_.page, N, H, Hk, D, max_pos, scale);
        else if (done == 1 && n_dec_ > 0)
            b_->gqa_attention(o, q, k_pools_[l], v_pools_[l], kv_dt_, d_tslot_, d_pos_, d_pt_, o_.max_pages_per_seq,
                              o_.page, n_dec_, H, Hk, D, max_pos, scale);
        b_->gemm16(L.o, o, N, out, d_.hidden);  // only the add + norm reads out
        if (lo) lora_apply(l, 1, o, {out}, N);
    }

    // y_t += the row's adapter's scale B (A x) for group g's targets of layer l (those reserved;
    // ys: the group's outputs in order)
    void lora_apply(int l, int g, const float *x, std::initializer_list<float *> ys, int64_t N) {
        Backend::LoraJob jobs[3];
        int32_t idx[3];
        int n = 0, m = 0;
        for (float *y : ys) {
            const int t = kLoraGroups[g][m++];
            const LoraTarget &T = lora_[size_t(l)][size_t(t)];
            if (!T.a) continue;
            jobs[n] = Backend::LoraJob{T.a, T.a_stride, T.b, T.scale, T.ranks, y, T.out, lora_rank_};
            idx[n++] = t;
        }
        if (!n || lora_active_ <= 0) return;
        const int64_t in = lora_[size_t(l)][size_t(idx[0])].in;
        if (lora_rest_) b_->lora(jobs, n, x, N, in, lora_active_, lora_rest_, ws_->take<float>(N * n * lora_rank_));
        for (const LoraRun &u : lora_runs_) {
            int32_t r[3];
            float sc[3];
            bool any = false;
            for (int i = 0; i < n; ++i) {
                const LoraTarget &T = lora_[size_t(l)][size_t(idx[i])];
                r[i] = T.rank_h[size_t(u.slot)], sc[i] = T.scale_h[size_t(u.slot)];
                any = any || r[i] > 0;
            }
            if (any) b_->lora_rows(jobs, n, x, N, u.row0, u.len, in, lora_active_, u.slot, r, sc);
        }
    }

    // The step's LoRA rows: outside a CUDA graph, runs of rows on one adapter long enough for the
    // backend's GEMM form (lora_rows: a prompt's tokens, or many sequences on one adapter), the
    // other rows by the per-row kernels (lora_shrink / lora_expand); in a graph every row by the
    // kernels, whose slots are read on the device.
    void lora_runs(int64_t N, bool graphable) {
        lora_runs_.clear();
        lora_rest_ = lora_step_ ? d_row_lora_ : nullptr;
        const int64_t min = b_->lora_rows_min();
        if (!lora_step_ || graphable || min <= 0 || N < min) return;
        row_lora_rest_.assign(row_lora_.begin(), row_lora_.end());
        bool rest = false;
        for (int64_t i = 0; i < N;) {
            int64_t j = i + 1;
            while (j < N && row_lora_[size_t(j)] == row_lora_[size_t(i)]) ++j;
            if (row_lora_[size_t(i)] >= 0) {
                if (j - i >= min) {
                    lora_runs_.push_back({i, j - i, row_lora_[size_t(i)]});
                    std::fill(row_lora_rest_.begin() + i, row_lora_rest_.begin() + j, -1);
                } else {
                    rest = true;
                }
            }
            i = j;
        }
        if (lora_runs_.empty()) return;
        lora_rest_ = nullptr;
        if (rest) {
            b_->upload(d_row_lora_rest_, row_lora_rest_.data(), N * 4);
            lora_rest_ = d_row_lora_rest_;
        }
    }

    // Dense: the MLP output into out (returns null). MoE: returns the experts' [N*topk] weighted
    // rows, for the caller to reduce.
    const float *mlp(const DenseLayer &L, const float *x, int64_t N, float *out) {
        if (d_.n_experts > 0) {
            // a routerless split dense MLP: the backend's big-step path when it takes the step
            const int l = static_cast<int>(&L - layers_.data());
            if (!L.router.data && b_->split_mlp(out, x, N, gx_host_[l].data(), d_.n_experts, gpu_fmt_, d_.hidden, d_.moe_inter))
                return nullptr;
            return moe(L, x, N);
        }
        const int64_t I = d_.inter;
        float *g = ws_->take<float>(N * I), *u = ws_->take<float>(N * I), *a = ws_->take<float>(N * I);
        // y16_ok: only silu_glu reads these (the backend may hand it one bf16 matrix, and a bf16 a
        // to the down projection; an adapter's update goes in where the backend keeps them)
        // the model's layers only (an EAGLE-3 head's layer is not one of them)
        const bool own = &L >= layers_.data() && &L < layers_.data() + layers_.size();
        const int l = own ? static_cast<int>(&L - layers_.data()) : -1;
        const bool lo = lora_step_ && own;
        const Backend::GemmJob in[] = {{&L.gate, x, g, I, true}, {&L.up, x, u, I, true}};
        b_->gemm_group(in, 2, N);
        if (lo) lora_apply(l, 2, x, {g, u}, N);
        b_->silu_glu(a, g, u, N * I);
        b_->gemm(L.down, a, N, out, d_.hidden);
        if (lo) lora_apply(l, 3, a, {out}, N);
        return nullptr;
    }

    // softmax router -> top-k resident experts (one launch group) -> their weighted rows
    const float *moe(const DenseLayer &L, const float *x, int64_t N) {
        const int64_t E = d_.hidden, k = d_.topk, NK = N * k, X = d_.n_experts;
        const int l = static_cast<int>(&L - layers_.data());
        float *logits = ws_->take<float>(N * X), *w = ws_->take<float>(NK);
        float *contrib = ws_->take<float>(NK * E), *scratch = ws_->take<float>(NK * d_.moe_inter);
        // int4 experts: prefill-sized steps take their fp8b copies (the blockwise fp8 GEMM); the int4
        // tiles take any count and read half the bytes (118 vs 240 us a layer at 32 rows, kbench), so
        // a busy decode step stays on them up to int4_rows_ rows
        const bool pf8 = gx8_on_ && N > int4_rows_;
        const Backend::ExpertW *table = pf8 ? gx_dev8_[l] : gx_dev_[l];
        const WFmt fmt = pf8 ? WFmt::FP8B : gpu_fmt_;
        if (!L.router.data) {  // every token takes every expert at weight 1 (a dense MLP split into experts)
            b_->moe_experts(contrib, x, d_idx_all_, d_w_one_, N, k, table, X, fmt, E, d_.moe_inter, true, 0.0f, 0.0f, scratch);
            return contrib;
        }
        if (!b_->router_softmax_fused(d_idx_, w, L.router, x, N, k, d_.norm_topk != 0, table, X)) {
            b_->gemm(L.router, x, N, logits, X);
            b_->router_softmax_topk(d_idx_, w, logits, N, X, k, d_.norm_topk != 0);
        }
        b_->moe_experts(contrib, x, d_idx_, w, N, k, table, X, fmt, E, d_.moe_inter, true, 0.0f, 0.0f, scratch);
        return contrib;
    }

    void *take_gx(size_t bytes) {
        const size_t a = (bytes + 255) & ~size_t(255);
        if (gx_used_ + a > gx_cap_) throw std::runtime_error("expert arena full (reserve more)");
        void *p = gx_arena_ + gx_used_;
        gx_used_ += a;
        return p;
    }

    // one layer's K (or V) pool, in the backend's page layout
    size_t kv_bytes() const { return size_t(o_.n_pages) * b_->kv_page_bytes(kv_dt_, o_.page, d_.n_kv_heads, d_.head_dim); }

    void alloc_pools() {
        // one allocation, K and V of each layer in turn at a fixed stride: a cache page of every
        // pool is then one strided block, which the host tier copies with one 2D copy per page
        // (an EAGLE-3 head's layer is one more, after the model's)
        const int kv_layers = d_.n_layers + (eagle_ ? 1 : 0);
        // (the int4 cache: each pool's ring after its pages, zeroed: its stale entries are read masked)
        const size_t extra = b_->kv_extra_bytes(kv_dt_, o_.n_slots, d_.n_kv_heads, d_.head_dim);
        const size_t stride = (kv_bytes() + extra + 255) & ~size_t(255);
        char *base = static_cast<char *>(b_->alloc(stride * 2 * kv_layers));
        owned_.push_back(base);
        for (int l = 0; l < kv_layers; ++l) {
            k_pools_.push_back(base + stride * (2 * l));
            v_pools_.push_back(base + stride * (2 * l + 1));
            if (extra) b_->zero(base + stride * (2 * l) + kv_bytes(), extra), b_->zero(base + stride * (2 * l + 1) + kv_bytes(), extra);
        }
        b_->set_kv_pages(o_.n_pages);
        std::vector<void *> pools;
        for (int l = 0; l < kv_layers; ++l) pools.push_back(k_pools_[l]), pools.push_back(v_pools_[l]);
        host_.init(b_.get(), pools, kv_bytes() / o_.n_pages, o_.n_pages);
        const int64_t N = o_.max_tokens, E = d_.hidden, R = o_.max_rows, Vv = d_.vocab;
        d_pos_ = static_cast<int32_t *>(b_->alloc(N * 4));
        d_tslot_ = static_cast<int32_t *>(b_->alloc(N * 4));
        d_tok_ = static_cast<int32_t *>(b_->alloc(N * 4));
        d_tiles_ = static_cast<int32_t *>(b_->alloc(N * 2 * 4));  // <= one tile per token
        d_pt_ = static_cast<int32_t *>(b_->alloc(size_t(o_.n_slots) * o_.max_pages_per_seq * 4));
        d_inv_freq_ = static_cast<float *>(b_->alloc(inv_freq_.size() * 4));
        b_->upload(d_inv_freq_, inv_freq_.data(), inv_freq_.size() * 4);
        logits_d_ = static_cast<float *>(b_->alloc(R * Vv * 4));
        fin_ = static_cast<float *>(b_->alloc(R * E * 4));
        d_argmax_ = static_cast<int32_t *>(b_->alloc((R + 1) * 4));  // [R] ids, then the non-finite flag
        d_sample_ = static_cast<SampleRow *>(b_->alloc(R * sizeof(SampleRow)));
        owned_.push_back(d_sample_);
        d_idx_ = static_cast<int32_t *>(b_->alloc(std::max<int64_t>(1, N * d_.topk) * 4));
        for (void *p : {(void *)d_pos_, (void *)d_tslot_, (void *)d_pt_, (void *)d_inv_freq_, (void *)logits_d_,
                        (void *)fin_, (void *)d_argmax_, (void *)d_idx_, (void *)d_tiles_, (void *)d_tok_})
            owned_.push_back(p);
        if (d_.n_experts > 0 && d_.topk == d_.n_experts && !layers_.empty() && !layers_[0].router.data) {
            // the routerless MLP's fixed routing: token t's j-th pair is expert j at weight 1
            const int64_t NK = N * d_.topk;
            std::vector<int32_t> idx(static_cast<size_t>(NK));
            for (int64_t i = 0; i < NK; ++i) idx[size_t(i)] = int32_t(i % d_.topk);
            const std::vector<float> one(size_t(NK), 1.0f);
            d_idx_all_ = static_cast<int32_t *>(b_->alloc(NK * 4));
            d_w_one_ = static_cast<float *>(b_->alloc(NK * 4));
            b_->upload(d_idx_all_, idx.data(), NK * 4);
            b_->upload(d_w_one_, one.data(), NK * 4);
            b_->sync();
            owned_.push_back(d_idx_all_), owned_.push_back(d_w_one_);
            // every expert gets all the step's rows: the int4 GEMM pays from 16 (H100, Qwen3-8B)
            b_->set_int4_gemm_rows(16);
        }
        h_host_ = host<float>(N * E);
        logits_host_ = host<float>(R * Vv);
        argmax_host_ = host<int32_t>(R + 1);
        auto f = [](int64_t n) { return (size_t(n) * 4 + 255) & ~size_t(255); };
        const int64_t Q = int64_t(d_.n_heads) * d_.head_dim, KV = int64_t(d_.n_kv_heads) * d_.head_dim;
        const int64_t NK = N * d_.topk;
        const size_t attn = 2 * f(N * Q) + 2 * f(N * KV),
                     mlp = d_.n_experts > 0 ? f(N * d_.n_experts) + f(NK) + f(NK * d_.hidden) + f(NK * d_.moe_inter)
                                            : 3 * f(N * d_.inter);
        size_t draft = 0;
        if (eagle_) {
            aux_ = static_cast<float *>(b_->alloc(size_t(N) * 3 * E * 4));
            d_rows_ = static_cast<int32_t *>(b_->alloc(N * 4));
            owned_.push_back(aux_);
            owned_.push_back(d_rows_);
            aux_w_.fmt = WFmt::F32, aux_w_.rows = N, aux_w_.cols = 3 * E, aux_w_.data = aux_;
            // a draft call: the fused hidden and its gathered taps, the embedding, its norm, the
            // hidden's norm, the layer's input, its output, the norm after attention, the MLP, and
            // the heads' rows ([S] each, S <= the slots)
            const int64_t S = o_.n_slots, Dv = ed_.draft_vocab;
            draft = f(N * 3 * E) + 6 * f(N * E) + f(N * 2 * E) + std::max(attn, 3 * f(N * ed_.inter)) + 3 * f(S * E) +
                    f(S * Dv) + 4096;
        }
        ws_ = std::make_unique<Arena>();
        const size_t lora = lora_rank_ > 0 ? 2 * f(N * 3 * lora_rank_) : 0;  // lora_apply's xa, up to two a layer half
        ws_->init(b_.get(), std::max(3 * f(N * E) + std::max(attn, mlp) + lora + 4096, draft));
    }

    template <typename T>
    T *host(int64_t n) {
        void *p = b_->host_alloc(std::max<int64_t>(1, n) * sizeof(T));
        host_owned_.push_back(p);
        return static_cast<T *>(p);
    }

    void validate(const sk_step_plan &pl) const {
        auto fail = [](const std::string &m) { throw std::runtime_error("bad step plan: " + m); };
        if (pl.n_tokens <= 0 || pl.n_tokens > o_.max_tokens) fail("n_tokens outside 1.." + std::to_string(o_.max_tokens));
        if (pl.n_rows <= 0 || pl.n_rows > o_.max_rows) fail("n_rows outside 1.." + std::to_string(o_.max_rows));
        const int64_t cap = int64_t(o_.max_pages_per_seq) * o_.page;
        for (int64_t i = 0; i < pl.n_tokens; ++i) {
            if (pl.tokens[i] < 0 || pl.tokens[i] >= d_.vocab) fail("token id out of vocabulary");
            if (pl.positions[i] < 0 || pl.positions[i] >= cap) fail("position beyond the page table");
            if (pl.token_slot[i] < 0 || pl.token_slot[i] >= o_.n_slots) fail("slot out of range");
            const int32_t pg = pl.page_table[int64_t(pl.token_slot[i]) * o_.max_pages_per_seq + pl.positions[i] / o_.page];
            if (pg < 0 || pg >= o_.n_pages) fail("page id out of range for a token's position");
        }
        // as the Kimi executor's: layout() reads each segment's tokens by its offset and length
        int64_t covered = 0;
        for (int s = 0; s < pl.n_segs; ++s) {
            const int32_t slot = pl.segs[3 * s], off = pl.segs[3 * s + 1], len = pl.segs[3 * s + 2];
            if (slot < 0 || slot >= o_.n_slots || off < 0 || len <= 0 || int64_t(off) + len > pl.n_tokens) fail("segment");
            for (int32_t j = off; j < off + len; ++j)
                if (pl.token_slot[j] != slot) fail("segment slot disagrees with token_slot");
            covered += len;
        }
        if (covered != pl.n_tokens) fail("segments do not cover every token exactly once");
        for (int64_t i = 0; i < pl.n_rows; ++i)
            if (pl.rows[i] < 0 || pl.rows[i] >= pl.n_tokens) fail("logits row out of range");
    }

    void gather_embedding(const int32_t *tokens, int64_t N) {
        const int64_t E = d_.hidden;
        for (int64_t t = 0; t < N; ++t) {
            float *dst = h_host_ + t * E;
            if (embed_.fmt == WFmt::F32) {
                std::memcpy(dst, static_cast<const float *>(embed_.data) + int64_t(tokens[t]) * E, E * 4);
            } else {
                const uint16_t *src = static_cast<const uint16_t *>(embed_.data) + int64_t(tokens[t]) * E;
                for (int64_t i = 0; i < E; ++i) {
                    const uint32_t u = uint32_t(src[i]) << 16;
                    std::memcpy(dst + i, &u, 4);
                }
            }
        }
    }

    sk_dense_dims d_;
    sk_exec_opts o_;
    std::vector<float> inv_freq_;
    DType kv_dt_ = DType::BF16;
    FILE *qc_ = nullptr;       // the quantised-expert cache being replayed or recorded
    bool qc_replay_ = false;
    std::string qc_path_;      // recording: the final name (written to .tmp)
    int64_t qc_count_ = 0;     // experts replayed or recorded
    std::unique_ptr<Backend> b_;
    std::map<std::string, Weight> mats_;
    std::map<std::string, std::pair<float *, int64_t>> vecs_;
    // an EAGLE-3 head (set_eagle): its dims, layer, fc, head, norms and draft -> target ids; the
    // residual rows every step taps ([max_tokens][3 * hidden]), as a table for embed_rows
    bool eagle_ = false;
    sk_eagle_dims ed_{};
    DenseLayer eg_;
    Weight eg_fc_, eg_head_, aux_w_;
    const float *eg_hidden_norm_ = nullptr, *eg_norm_ = nullptr;
    std::vector<int32_t> eg_d2t_;
    int32_t *d_d2t_ = nullptr;  // the same on the device
    float *aux_ = nullptr;
    int32_t *d_rows_ = nullptr;
    int64_t last_n_ = 0;  // the last step's tokens
    std::vector<void *> owned_, host_owned_;
    size_t weight_bytes_ = 0;
    bool finalized_ = false;
    Weight embed_, lm_head_;
    const float *final_norm_ = nullptr;
    std::vector<DenseLayer> layers_;
    std::vector<void *> k_pools_, v_pools_;
    HostPages host_;  // after b_: destroyed before it
    int32_t *d_pos_ = nullptr, *d_tslot_ = nullptr, *d_pt_ = nullptr, *d_argmax_ = nullptr;
    SampleRow *d_sample_ = nullptr;
    SampleScratch sample_scratch_;  // masks, adjustments, log-probabilities (step_tokens)
    int32_t *d_tiles_ = nullptr;  // prefill query tiles of the current step (gqa_prefill)
    std::vector<int32_t> h_segs_;  // and its sequence chunks: first row, count, first position, slot
    int64_t n_dec_ = 0;            // leading one-token chunks of the step (decoding sequences)
    bool dec_fa3_ = false;         // they go through the backend's prefill form (see step)
    int32_t *d_tok_ = nullptr;    // the step's token ids (device-side embedding gather)
    bool embed_dev_ = false;      // the embedding table is on the device
    int64_t n_tiles_ = 0;
    float *d_inv_freq_ = nullptr, *logits_d_ = nullptr, *fin_ = nullptr;
    float *h_host_ = nullptr, *logits_host_ = nullptr;
    int32_t *argmax_host_ = nullptr;
    std::unique_ptr<Arena> ws_;
    std::string desc_;
    int32_t *d_idx_ = nullptr;
    int32_t *d_idx_all_ = nullptr;  // the routerless MLP's fixed routing (see finalize, moe)
    float *d_w_one_ = nullptr;
    // resident experts: arena, per-layer tables (host copy, then device)
    WFmt gpu_fmt_ = WFmt::BF16;
    char *gx_arena_ = nullptr;
    size_t gx_cap_ = 0, gx_used_ = 0;
    std::vector<std::vector<Backend::ExpertW>> gx_host_;
    std::vector<Backend::ExpertW *> gx_dev_;
    // int4 experts' fp8b copies (prefill steps), when on
    bool gx8_on_ = false;
    // the most rows a step runs on the int4 tiles before the fp8b copies (FULLM_INT4_ROWS)
    int64_t int4_rows_ = 64;  // Policy::expert_tile_max_rows, or FULLM_INT4_ROWS
    std::vector<std::vector<Backend::ExpertW>> gx_host8_;
    std::vector<Backend::ExpertW *> gx_dev8_;
    // CUDA graphs of forward(), keyed by what the recorded launches depend on
    struct GraphKey {
        int64_t n_tokens;
        std::vector<int32_t> rows;
        bool dec_fa3;  // the decoding rows' attention kernel (see step)
        int32_t lora;  // some row has an adapter: the loaded adapters' bound rank (lora_active_), else 0
        bool operator<(const GraphKey &o) const {
            return std::tie(n_tokens, rows, dec_fa3, lora) < std::tie(o.n_tokens, o.rows, o.dec_fa3, o.lora);
        }
    };
    // ------------------------------------------------------------------ LoRA (unmerged)
    // per layer and target (q k v o gate up down): A [slots][rank][in], B [slots][out][rank] bf16,
    // scale [slots]; a sequence slot's adapter, and each step's per-row adapters (host, device)
    struct LoraTarget {
        uint16_t *a = nullptr, *b = nullptr;  // slot s's A at a + s * a_stride (its group's pools stacked)
        int64_t a_stride = 0;
        float *scale = nullptr;
        int32_t *ranks = nullptr;      // [slots] each slot's adapter's rank here (0: none)
        std::vector<float> scale_h;    // the same on the host
        std::vector<int32_t> rank_h;
        int64_t in = 0, out = 0;
    };
    // targets that read one x, adapted together (Backend::lora's jobs): q k v | o | gate up | down
    static constexpr int kLoraGroups[4][3] = {{0, 1, 2}, {3, -1, -1}, {4, 5, -1}, {6, -1, -1}};
    struct LoraRun {  // rows [row0, row0 + len) on one adapter, for Backend::lora_rows
        int64_t row0, len;
        int32_t slot;
    };
    int32_t lora_slots_ = 0, lora_rank_ = 0, lora_active_ = 0;
    std::vector<std::array<LoraTarget, 7>> lora_;
    std::vector<int32_t> seq_lora_, row_lora_, row_lora_rest_;
    int32_t *d_row_lora_ = nullptr, *d_row_lora_rest_ = nullptr;
    bool lora_step_ = false;
    std::vector<LoraRun> lora_runs_;          // the step's long runs (outside a graph)
    const int32_t *lora_rest_ = nullptr;      // the slots of every other row, when some has one (else null)
    std::map<GraphKey, void *> graphs_;
    bool graphs_on_ = true;
    int64_t decode_graph_rows_ = 256;
    std::set<int64_t> capture_failed_;  // step sizes whose capture failed (said once each)
};

}  // namespace

std::unique_ptr<ExecIface> make_dense_exec(const sk_dense_dims &d, const sk_exec_opts &o) {
    return std::make_unique<DenseExec>(d, o);
}

}  // namespace fullm
