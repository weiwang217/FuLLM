// SPDX-License-Identifier: Apache-2.0
// k3_exec.cpp - the Kimi K3 forward graph and the executor C ABI (fullm_exec.h).
//
// The graph is written once against Backend and runs unchanged on the CPU backend (the
// oracle check on any machine) and the CUDA backend. It reproduces
// crates/fullm/src/models/kimi_k3/cpu.rs, the Rust reference, statement for statement:
// read the two side by side.
//
// PLACEMENT. Every trunk tensor lives in backend memory. The embedding table stays in host
// memory (one row gathered per token). Routed experts stay in host memory and are computed
// by the CPU ExpertPool: per MoE layer the backend computes the router and the latent
// down-projection, hands z and the routing decisions to the host, runs the shared expert
// while the CPU works, and takes the weighted expert sum back.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "exec_common.h"
#include "fp8_block.h"
#include "int4_group.h"

namespace fullm {
namespace {

struct Layer {
    bool mla = false, dense = false;
    int kda_i = -1, mla_i = -1;
    const float *in_norm, *post_norm, *fold_attn, *fold_mlp;
    // KDA (g full-rank, or g_a/g_b low-rank)
    Weight q, k, v, g, g_a, g_b, o, f_a, f_b, b;
    const float *q_conv, *k_conv, *v_conv, *a_log, *dt_bias, *o_norm;
    // MLA (q_a/q_b LoRA, or mq direct)
    Weight q_a, q_b, mq, kv_a, kv_b, mo, mg;
    bool has_gate = false;
    const float *q_a_norm, *kv_a_norm;
    // dense FFN
    Weight d_gate, d_up, d_down;
    // MoE
    Weight router, down, up, sh_gate, sh_up, sh_down;
    const float *router_bias, *latent_norm;
    bool latent = false;
};

class K3Exec final : public ExecIface {
public:
    K3Exec(const sk_k3_dims &d, const sk_exec_opts &o) : d_(d), o_(o) {
        if (!d.layer_is_mla) throw std::runtime_error("layer_is_mla is null");
        is_mla_.assign(d.layer_is_mla, d.layer_is_mla + d.n_layers);
        if (o.n_slots <= 0 || o.n_pages <= 0 || o.page <= 0 || o.max_pages_per_seq <= 0 || o.max_tokens <= 0 ||
            o.max_rows <= 0)
            throw std::runtime_error("executor options must all be positive");
        if (o.kv_dtype == SK_DT_FP8 || o.kv_dtype == SK_DT_FP8E4M3)
            throw std::runtime_error("--kv-dtype fp8 is for dense GQA models only");
        kv_dt_ = o.kv_dtype == SK_DT_BF16 ? DType::BF16 : DType::F32;
        st_dt_ = o.state_dtype == SK_DT_BF16 ? DType::BF16 : DType::F32;
        b_ = make_backend(backend_name(o.backend_name, o.backend), o.device, o.cpu_threads);
        {  // the families this model needs: Kimi's ops always, MLA and KDA per its layers
            const uint32_t have = b_->caps().families;
            const bool mla = std::find(is_mla_.begin(), is_mla_.end(), 1) != is_mla_.end();
            const bool kda = std::find(is_mla_.begin(), is_mla_.end(), 0) != is_mla_.end();
            const uint32_t need = FAM_KIMI | (mla ? FAM_MLA : 0u) | (kda ? FAM_KDA : 0u);
            if ((have & need) != need)
                throw std::runtime_error(std::string("the ") + b_->name() + " backend lacks this model's ops (" +
                                         ((need & ~have & FAM_MLA) ? "MLA " : "") + ((need & ~have & FAM_KDA) ? "KDA " : "") +
                                         ((need & ~have & FAM_KIMI) ? "Kimi" : "") + ")");
        }
        b_->set_exact_activations(o.bf16_activations == 0);
        ExpertDims ed;
        ed.latent = d.latent;
        ed.inter = d.moe_inter;
        ed.topk = d.topk;
        ed.n_experts = d.n_experts;
        ed.b1 = d.situ_b1;
        ed.b2 = d.situ_b2;
        ed.silu = d.act_silu != 0;
        ed.bf16 = d.expert_bf16 != 0;
        experts_ = std::make_unique<ExpertPool>(o.cpu_threads, ed);
        cpu_bf16_ = ed.bf16;
        expert_table_.assign(d.n_layers, std::vector<ExpertPtrs>());
        for (int l = 0; l < d.n_layers; ++l)
            if (l >= d.first_dense) expert_table_[l].assign(d.n_experts, ExpertPtrs{});
        maxs_ = d.attn_res ? d.n_layers / d.attn_res_block + 2 : 1;
    }

    ~K3Exec() {
        place_stop();  // before anything it writes into goes
        if (!b_) return;
        try {
            b_->sync();
        } catch (...) {
        }
        write_route_stats();
        for (auto &kv : graphs_) b_->graph_release(kv.second.exec);
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
        if (name == "embed") {  // stays in host memory, not copied
            if (w.fmt != WFmt::BF16 && w.fmt != WFmt::F32) throw std::runtime_error("embed must be bf16 or f32");
            w.data = data;
            embed_ = w;
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
        float *dd = static_cast<float *>(b_->alloc(n * 4));
        b_->upload(dd, data, n * 4);
        b_->sync();
        if (vecs_.count(name)) throw std::runtime_error(name + ": added twice");
        vecs_[name] = {dd, n};
        weight_bytes_ += n * 4;
    }

    void set_expert(int32_t layer, int32_t e, const ExpertPtrs &p) override {
        if (layer < 0 || layer >= d_.n_layers || expert_table_[layer].empty() || e < 0 || e >= d_.n_experts)
            throw std::runtime_error("set_expert: no routed expert " + std::to_string(e) + " in layer " +
                                     std::to_string(layer));
        expert_table_[layer][e] = p;
    }

    // ------------------------------------------------------------------ resident experts
    void reserve_gpu_experts(int32_t fmt, size_t bytes) override {
        if (finalized_) throw std::runtime_error("reserve_gpu_experts after finalize");
        if (gx_arena_) throw std::runtime_error("gpu expert arena reserved twice");
        if (fmt != SK_FMT_BF16 && fmt != SK_FMT_INT8 && fmt != SK_FMT_FP8 && fmt != SK_FMT_FP8B && fmt != SK_FMT_INT4G)
            throw std::runtime_error("resident experts must be bf16, int8, fp8, fp8b or int4");
        gpu_fmt_ = static_cast<WFmt>(fmt);
        b_->set_expert_format(gpu_fmt_);
        gx_arena_ = static_cast<char *>(b_->alloc(bytes));
        owned_.push_back(gx_arena_);
        gx_cap_ = bytes;
    }

    void set_gpu_expert(int32_t layer, int32_t e, const void *w1, const float *s1, const void *w3, const float *s3,
                        const void *w2, const float *s2) override {
        if (!gx_arena_) throw std::runtime_error("set_gpu_expert before reserve_gpu_experts");
        if (layer < 0 || layer >= d_.n_layers || expert_table_[layer].empty() || e < 0 || e >= d_.n_experts)
            throw std::runtime_error("set_gpu_expert: no routed expert " + std::to_string(e) + " in layer " +
                                     std::to_string(layer));
        if (gx_host_.empty()) {
            gx_host_.assign(d_.n_layers, std::vector<Backend::ExpertW>());
            gx_count_.assign(d_.n_layers, 0);
        }
        auto &tab = gx_host_[layer];
        if (tab.empty()) tab.assign(d_.n_experts, Backend::ExpertW{});
        if (tab[e].w1.data) throw std::runtime_error("gpu expert set twice");
        // INT8 / FP8 come quantised with row scales. FP8B (blockwise e4m3, the prefill's CUTLASS
        // GEMM) and INT4G come as bf16 and are quantised here, after the stored layout (gate / up
        // rows interleaved or not) is fixed, as the dense executor does.
        const bool q8 = gpu_fmt_ == WFmt::INT8 || gpu_fmt_ == WFmt::FP8;
        auto put = [&](const void *data, const float *scale, int64_t rows, int64_t cols) {
            Weight w;
            w.fmt = gpu_fmt_;
            w.rows = rows;
            w.cols = cols;
            if (gpu_fmt_ == WFmt::FP8B || gpu_fmt_ == WFmt::INT4G) {
                std::vector<uint8_t> q;
                std::vector<float> s;
                if (gpu_fmt_ == WFmt::INT4G) {
                    if (cols % 8) throw std::runtime_error("int4 experts need rows of whole 8-value words");
                    w.layout = b_->weight_layout(WFmt::INT4G, rows, cols);
                    q.resize(size_t(int4g_bytes(rows, cols))), s.resize(size_t(int4g_scales(rows, cols)));
                    quantize_int4_groups(static_cast<const uint16_t *>(data), rows, cols, q.data(), s.data(), w.layout);
                } else {
                    q.resize(size_t(rows) * cols), s.resize(size_t(fp8b_scales(rows, cols)));
                    quantize_fp8_blocks(static_cast<const uint16_t *>(data), rows, cols, q.data(), s.data());
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
                if (!scale) throw std::runtime_error("8-bit resident expert without row scales");
                w.scale = static_cast<const float *>(take_gx(size_t(rows) * 4));
                b_->upload(const_cast<float *>(w.scale), scale, size_t(rows) * 4);
            }
            return w;
        };
        const int64_t L = d_.latent, I = d_.moe_inter;
        if (b_->moe_gate_up_interleaved(gpu_fmt_, L, I)) {
            // gate and up as one row-interleaved [2I][L] matrix (see Backend::moe_gate_up_interleaved)
            const size_t row = size_t(L) * (q8 ? 1 : 2);
            std::vector<uint8_t> w13(2 * size_t(I) * row);
            std::vector<float> s13(q8 ? 2 * size_t(I) : 0);
            for (int64_t r = 0; r < I; ++r) {
                std::memcpy(w13.data() + (2 * r) * row, static_cast<const uint8_t *>(w1) + r * row, row);
                std::memcpy(w13.data() + (2 * r + 1) * row, static_cast<const uint8_t *>(w3) + r * row, row);
                if (q8) s13[2 * r] = s1[r], s13[2 * r + 1] = s3[r];
            }
            tab[e] = Backend::ExpertW{put(w13.data(), q8 ? s13.data() : nullptr, 2 * I, L), Weight{}, put(w2, s2, L, I)};
        } else {
            tab[e] = Backend::ExpertW{put(w1, s1, I, L), put(w3, s3, I, L), put(w2, s2, L, I)};
        }
        gx_count_[layer]++;
        b_->sync();  // the caller frees its copies on return
    }

    void *take_gx(size_t bytes) {
        const size_t a = (bytes + 255) & ~size_t(255);
        if (gx_used_ + a > gx_cap_) throw std::runtime_error("gpu expert arena full (reserve more)");
        void *p = gx_arena_ + gx_used_;
        gx_used_ += a;
        return p;
    }

    // What alloc_pools() will allocate, computable before finalize.
    size_t pool_bytes() const override {
        int nk = 0, nm = 0;
        for (int l = 0; l < d_.n_layers; ++l) (is_mla_[l] ? nm : nk)++;
        const size_t S = size_t(o_.n_slots), N = size_t(o_.max_tokens), R = size_t(o_.max_rows);
        const size_t slot_s = size_t(nk) * kda_s_elems() * dt_size(st_dt_), slot_c = size_t(nk) * kda_conv_floats() * 4;
        const size_t kv = size_t(nm) * o_.n_pages * o_.page * width() * dt_size(kv_dt_);
        const size_t plan = N * 8 + 3 * S * 4 + S * o_.max_pages_per_seq * 4 + N * d_.topk * 4;
        const size_t head = R * d_.vocab * 4 + R * d_.hidden * 4;
        const size_t tables = size_t(d_.n_layers) * d_.n_experts * sizeof(Backend::ExpertW);
        // the long-prompt kernels' scratch (chunked KDA, MLA un-absorbed: one buffer, the larger
        // need), for a step of max_tokens
        const size_t prompts = std::max(nk > 0 ? b_->kda_chunked_scratch(int64_t(N), int64_t(S), d_.kda_heads, d_.kda_head_dim) : 0,
                                        nm > 0 ? b_->mla_prefill_scratch(int64_t(N), d_.n_heads, width()) : 0);
        return S * (slot_s + slot_c) + kv + plan + head + tables + workspace_bytes(o_.max_tokens) + prompts + (64 << 20);
    }

    void finalize() override {
        std::vector<std::string> missing;
        const int64_t E = d_.hidden, H = d_.n_heads, P = int64_t(d_.kda_heads) * d_.kda_head_dim;
        const int64_t D = d_.kda_head_dim, qh = d_.qk_nope + d_.qk_rope, SI = int64_t(d_.moe_inter) * d_.n_shared;
        auto M = [&](const std::string &n, int64_t r, int64_t c) -> Weight {
            auto it = mats_.find(n);
            if (it == mats_.end()) {
                missing.push_back(n);
                return Weight{};
            }
            if (it->second.rows != r || it->second.cols != c) {
                missing.push_back(n + " (shape " + std::to_string(it->second.rows) + "x" +
                                  std::to_string(it->second.cols) + ", want " + std::to_string(r) + "x" +
                                  std::to_string(c) + ")");
            }
            return it->second;
        };
        auto V = [&](const std::string &n, int64_t len) -> const float * {
            auto it = vecs_.find(n);
            if (it == vecs_.end()) {
                missing.push_back(n);
                return nullptr;
            }
            if (it->second.second != len) missing.push_back(n + " (length " + std::to_string(it->second.second) + ", want " + std::to_string(len) + ")");
            return it->second.first;
        };
        if (!embed_.data || embed_.rows != d_.vocab || embed_.cols != E) missing.push_back("embed");
        lm_head_ = M("lm_head", d_.vocab, E);
        final_norm_ = V("final_norm", E);
        if (d_.attn_res) fold_out_ = V("fold_out", E);
        int nk = 0, nm = 0;
        layers_.resize(d_.n_layers);
        for (int l = 0; l < d_.n_layers; ++l) {
            Layer &L = layers_[l];
            const std::string p = "L" + std::to_string(l) + ".";
            L.mla = is_mla_[l] != 0;
            L.dense = l < d_.first_dense;
            L.in_norm = V(p + "in_norm", E);
            L.post_norm = V(p + "post_norm", E);
            if (d_.attn_res) {
                L.fold_attn = V(p + "fold_attn", E);
                L.fold_mlp = V(p + "fold_mlp", E);
            }
            if (L.mla) {
                L.mla_i = nm++;
                if (d_.q_lora > 0) {
                    L.q_a = M(p + "q_a", d_.q_lora, E);
                    L.q_a_norm = V(p + "q_a_norm", d_.q_lora);
                    L.q_b = M(p + "q_b", H * qh, d_.q_lora);
                } else {
                    L.mq = M(p + "q", H * qh, E);
                }
                L.kv_a = M(p + "kv_a", d_.kv_lora + d_.qk_rope, E);
                L.kv_a_norm = V(p + "kv_a_norm", d_.kv_lora);
                L.kv_b = M(p + "kv_b", H * (d_.qk_nope + d_.v_head), d_.kv_lora);
                L.mo = M(p + "o", E, H * d_.v_head);
                L.has_gate = d_.mla_out_gate != 0;
                if (L.has_gate) L.mg = M(p + "g", H * d_.v_head, E);
            } else {
                L.kda_i = nk++;
                L.q = M(p + "q", P, E);
                L.k = M(p + "k", P, E);
                L.v = M(p + "v", P, E);
                if (d_.kda_gate_lowrank) {
                    L.g_a = M(p + "g_a", D, E);
                    L.g_b = M(p + "g_b", P, D);
                } else {
                    L.g = M(p + "g", P, E);
                }
                L.o = M(p + "o", E, P);
                L.f_a = M(p + "f_a", D, E);
                L.f_b = M(p + "f_b", P, D);
                L.b = M(p + "b", d_.kda_heads, E);
                L.q_conv = V(p + "q_conv", P * d_.conv_k);
                L.k_conv = V(p + "k_conv", P * d_.conv_k);
                L.v_conv = V(p + "v_conv", P * d_.conv_k);
                L.a_log = V(p + "a_log", d_.kda_heads);
                L.dt_bias = V(p + "dt_bias", P);
                L.o_norm = V(p + "o_norm", D);
            }
            if (L.dense) {
                L.d_gate = M(p + "d_gate", d_.dense_inter, E);
                L.d_up = M(p + "d_up", d_.dense_inter, E);
                L.d_down = M(p + "d_down", E, d_.dense_inter);
            } else {
                L.router = M(p + "router", d_.n_experts, E);
                L.router_bias = V(p + "router_bias", d_.n_experts);
                L.latent = d_.latent_moe != 0;
                if (L.latent) {
                    L.down = M(p + "down", d_.latent, E);
                    L.up = M(p + "up", E, d_.latent);
                    if (d_.latent_norm) L.latent_norm = V(p + "latent_norm", d_.latent);
                }
                L.sh_gate = M(p + "sh_gate", SI, E);
                L.sh_up = M(p + "sh_up", SI, E);
                L.sh_down = M(p + "sh_down", E, SI);
                for (int e = 0; e < d_.n_experts; ++e)
                    if (!expert_table_[l][e].p1) {
                        missing.push_back(p + "expert " + std::to_string(e));
                        break;
                    }
            }
        }
        if (!missing.empty()) {
            std::string msg = "executor is missing or has mis-shaped tensors (" + std::to_string(missing.size()) + "):";
            for (size_t i = 0; i < missing.size() && i < 12; ++i) msg += "\n  " + missing[i];
            throw std::runtime_error(msg);
        }
        n_kda_ = nk;
        n_mla_ = nm;
        alloc_pools();
        gx_dev_.assign(d_.n_layers, nullptr);
        resident_.assign(d_.n_layers, std::vector<uint8_t>());
        for (int l = 0; l < d_.n_layers && !gx_host_.empty(); ++l) {
            if (gx_host_[l].empty()) continue;
            const size_t bytes = gx_host_[l].size() * sizeof(Backend::ExpertW);
            auto *t = static_cast<Backend::ExpertW *>(b_->alloc(bytes));
            owned_.push_back(t);
            b_->upload(t, gx_host_[l].data(), bytes);
            gx_dev_[l] = t;
            resident_[l].resize(d_.n_experts);
            for (int e = 0; e < d_.n_experts; ++e) resident_[l][e] = gx_host_[l][e].w1.data != nullptr;
        }
        all_resident_ = !gx_count_.empty();
        for (int l = 0; all_resident_ && l < d_.n_layers; ++l)
            if (!layers_[l].dense && gx_count_[l] != d_.n_experts) all_resident_ = false;
        graphs_on_ = o_.no_graphs == 0 && !route_stats_;  // counting routes reads them back every layer
        // experts placed by routing frequency: with some but not all experts in GPU memory, bf16
        // experts on the CPU to copy or quantise from, and a resident format a swap can write
        const char *pe = std::getenv("FULLM_EXPERT_PLACE"), *ps = std::getenv("FULLM_EXPERT_SWAPS");
        place_every_ = pe ? std::atoll(pe) : 128;
        place_swaps_ = ps ? std::atoll(ps) : 128;
        const char *pf = std::getenv("FULLM_EXPERT_FREEZE"), *pt = std::getenv("FULLM_EXPERT_THAW");
        const char *pd = std::getenv("FULLM_EXPERT_DECAY");
        place_freeze_ = pf ? std::atoll(pf) : 3;
        place_thaw_ = pt ? std::max<int64_t>(1, std::atoll(pt)) : 4;
        place_decay_ = pd ? std::clamp(std::atof(pd), 0.0, 0.99) : 0.8;
        place_on_ = place_every_ > 0 && place_swaps_ > 0 && !all_resident_ && !gx_count_.empty() && cpu_bf16_ &&
                    !b_->on_host() &&
                    (gpu_fmt_ == WFmt::BF16 || gpu_fmt_ == WFmt::FP8B || gpu_fmt_ == WFmt::INT4G);
        if (place_on_) {
            const size_t nb = size_t(d_.n_layers) * d_.n_experts * 4;
            d_route_n_ = static_cast<uint32_t *>(b_->alloc(nb));
            owned_.push_back(d_route_n_);
            b_->zero(d_route_n_, nb);
            place_on_ = b_->count_routes(d_route_n_, d_idx_, 0, d_.n_experts);
            place_score_.assign(size_t(d_.n_layers) * d_.n_experts, 0.0);
            if (place_on_) {
                place_stage_ = host<Backend::ExpertW>(int64_t(d_.n_layers) * d_.n_experts);
                place_thread_ = std::thread([this] { place_worker(); });
            }
        }
        b_->sync();
        finalized_ = true;
    }

    // ------------------------------------------------------------------ state pools
    int64_t kda_s_elems() const { return int64_t(d_.kda_heads) * d_.kda_head_dim * d_.kda_head_dim; }
    int64_t kda_conv_floats() const { return 3 * int64_t(d_.kda_heads) * d_.kda_head_dim * (d_.conv_k - 1); }
    int64_t width() const { return d_.kv_lora + d_.qk_rope; }

    size_t slot_s_bytes() const { return n_kda_ * kda_s_elems() * dt_size(st_dt_); }
    size_t slot_conv_bytes() const { return n_kda_ * kda_conv_floats() * 4; }
    size_t slot_bytes() const override { return slot_s_bytes() + slot_conv_bytes(); }

    void alloc_pools() {
        const int64_t S = o_.n_slots;
        s_pool_ = b_->alloc(std::max<size_t>(1, S * slot_s_bytes()));
        conv_pool_ = static_cast<float *>(b_->alloc(std::max<size_t>(1, S * slot_conv_bytes())));
        owned_.push_back(s_pool_);
        owned_.push_back(conv_pool_);
        b_->zero(s_pool_, S * slot_s_bytes());
        b_->zero(conv_pool_, S * slot_conv_bytes());
        const size_t kv_bytes = size_t(o_.n_pages) * o_.page * width() * dt_size(kv_dt_);
        for (int i = 0; i < n_mla_; ++i) {
            void *p = b_->alloc(kv_bytes);
            owned_.push_back(p);
            kv_pools_.push_back(p);
        }
        // plan arrays on the backend
        const int64_t N = o_.max_tokens;
        d_pos_ = static_cast<int32_t *>(b_->alloc(N * 4));
        d_tslot_ = static_cast<int32_t *>(b_->alloc(N * 4));
        d_segs_ = static_cast<int32_t *>(b_->alloc(3 * std::max(1, o_.n_slots) * 4));
        d_segs_long_ = static_cast<int32_t *>(b_->alloc(3 * std::max(1, o_.n_slots) * 4));
        d_segs_short_ = static_cast<int32_t *>(b_->alloc(3 * std::max(1, o_.n_slots) * 4));
        owned_.push_back(d_segs_long_), owned_.push_back(d_segs_short_);
        d_mla_meta_ = static_cast<int32_t *>(b_->alloc((4 * int64_t(std::max(1, o_.n_slots)) + 2) * 4));
        d_short_rows_ = static_cast<int32_t *>(b_->alloc(3 * N * 4));
        d_short_tslot_ = d_short_rows_ + N, d_short_pos_ = d_short_tslot_ + N;
        owned_.push_back(d_mla_meta_), owned_.push_back(d_short_rows_);
        d_pt_ = static_cast<int32_t *>(b_->alloc(size_t(o_.n_slots) * o_.max_pages_per_seq * 4));
        d_idx_ = static_cast<int32_t *>(b_->alloc(N * d_.topk * 4));
        for (void *p : {(void *)d_pos_, (void *)d_tslot_, (void *)d_segs_, (void *)d_pt_, (void *)d_idx_}) owned_.push_back(p);
        // host staging (pinned on CUDA)
        const int64_t E = d_.hidden, L = d_.latent, R = o_.max_rows, Vv = d_.vocab;
        h_host_ = host<float>(N * E);
        z_host_ = host<float>(N * L);
        acc_host_ = host<float>(N * L);
        w_host_ = host<float>(N * d_.topk);
        idx_host_ = host<int32_t>(N * d_.topk);
        logits_host_ = host<float>(R * Vv);
        compact_host_ = host<float>(N * d_.topk * L);
        sel_host_ = host<int32_t>(N * d_.topk);
        argmax_host_ = host<int32_t>(R + 1);
        logits_d_ = static_cast<float *>(b_->alloc(R * Vv * 4));
        fin_ = static_cast<float *>(b_->alloc(R * E * 4));
        d_argmax_ = static_cast<int32_t *>(b_->alloc((R + 1) * 4));  // [R] ids, then the non-finite flag
        d_sample_ = static_cast<SampleRow *>(b_->alloc(R * sizeof(SampleRow)));
        owned_.push_back(d_sample_);
        owned_.push_back(logits_d_);
        owned_.push_back(fin_);
        owned_.push_back(d_argmax_);
        ws_ = std::make_unique<Arena>();
        ws_->init(b_.get(), workspace_bytes(N));
    }

    template <typename T>
    T *host(int64_t n) {
        void *p = b_->host_alloc(std::max<int64_t>(1, n) * sizeof(T));
        host_owned_.push_back(p);
        return static_cast<T *>(p);
    }

    size_t workspace_bytes(int64_t N) const {
        const int64_t E = d_.hidden, H = d_.n_heads, P = int64_t(d_.kda_heads) * d_.kda_head_dim;
        const int64_t D = d_.kda_head_dim, qh = d_.qk_nope + d_.qk_rope, W = width();
        const int64_t SI = int64_t(d_.moe_inter) * d_.n_shared, Hk = d_.kda_heads;
        auto f = [](int64_t n) { return (size_t(n) * 4 + 255) & ~size_t(255); };
        const size_t persistent = 4 * f(N * E) + f(N * maxs_ * E);
        const size_t kda = 10 * f(N * P) + 2 * f(N * D) + 2 * f(N * Hk);
        const size_t mla = f(N * d_.q_lora) + 2 * f(N * H * qh) + f(N * W) + f(N * H * W) + f(N * H * d_.kv_lora) +
                           3 * f(N * H * d_.v_head);  // + the other rows' q and y beside mla_prefill
        const size_t dense = 3 * f(N * d_.dense_inter);
        const int64_t NK = N * d_.topk;
        const size_t moe = f(N * d_.n_experts) + f(N * d_.topk) + 2 * f(N * d_.latent) + 3 * f(N * SI) + f(N * E) +
                           2 * f(NK * d_.latent) + f(NK * d_.moe_inter) + f(NK);  // contrib, compact, scratch, sel
        return persistent + std::max({kda, mla, dense, moe}) + 4096;
    }

    // ------------------------------------------------------------------ the step
    // logits_out: [n_rows][vocab]; or, with tokens_out, only each row's argmax.
    void step(const sk_step_plan &pl, float *logits_out, sk_step_stats *st, int32_t *tokens_out,
              const sk_sample_row *sample, const sk_sample_io *io) override {
        if (!finalized_) throw std::runtime_error("step before finalize");
        const auto t_start = Clock::now();
        sk_step_stats stats{};
        const int64_t N = pl.n_tokens, R = pl.n_rows, E = d_.hidden;
        validate(pl);

        // zero the recurrent state of sequences starting fresh
        for (int i = 0; i < pl.n_reset; ++i) {
            const int64_t s = pl.reset_slots[i];
            b_->zero(static_cast<char *>(s_pool_) + s * slot_s_bytes(), slot_s_bytes());
            b_->zero(reinterpret_cast<char *>(conv_pool_) + s * slot_conv_bytes(), slot_conv_bytes());
        }

        auto t = Clock::now();
        b_->upload(d_pos_, pl.positions, N * 4);
        b_->upload(d_tslot_, pl.token_slot, N * 4);
        b_->upload(d_segs_, pl.segs, size_t(pl.n_segs) * 3 * 4);
        // KDA: the step's prompts long enough for the chunked form, apart from the rest
        n_long_ = n_short_ = 0;
        if (n_kda_ > 0 && b_->kda_chunk_min() > 0) {
            segs_long_.clear(), segs_short_.clear();
            for (int64_t i = 0; i < pl.n_segs; ++i) {
                auto &dst = pl.segs[3 * i + 2] >= b_->kda_chunk_min() ? segs_long_ : segs_short_;
                dst.insert(dst.end(), pl.segs + 3 * i, pl.segs + 3 * i + 3);
            }
            n_long_ = int64_t(segs_long_.size() / 3), n_short_ = int64_t(segs_short_.size() / 3);
            if (n_long_ > 0) {
                b_->upload(d_segs_long_, segs_long_.data(), segs_long_.size() * 4);
                if (n_short_ > 0) b_->upload(d_segs_short_, segs_short_.data(), segs_short_.size() * 4);
            }
        }
        plan_mla(pl);
        b_->upload(d_pt_, pl.page_table, size_t(o_.n_slots) * o_.max_pages_per_seq * 4);
        gather_embedding(pl.tokens, N);
        ws_->reset();
        float *h = ws_->take<float>(N * E), *pref = ws_->take<float>(N * E);
        float *hin = ws_->take<float>(N * E), *tmp = ws_->take<float>(N * E);
        float *blocks = ws_->take<float>(N * maxs_ * E);
        b_->upload(h, h_host_, N * E * 4);
        stats.ms_upload += ms_since(t);
        const size_t sub = ws_->mark();
        const Bufs bufs{h, pref, hin, tmp, blocks, sub};

        // Small steps with no host work in the middle replay a CUDA graph of the forward pass,
        // keyed by everything the recorded launches depend on. Attention is then launched for
        // the longest possible context so the launch does not follow the step's positions
        // (its results never depend on that bound).
        const bool graphable = all_resident_ && graphs_on_ && N <= kGemvMaxRows;
        int64_t max_pos = 0;
        if (graphable) max_pos = int64_t(o_.max_pages_per_seq) * o_.page - 1;
        else
            for (int64_t i = 0; i < N; ++i) max_pos = std::max<int64_t>(max_pos, pl.positions[i]);

        sk_step_stats fs{};
        if (!graphable) {
            forward(pl, bufs, max_pos, fs);
        } else {
            const GraphKey key{N, pl.n_segs, std::vector<int32_t>(pl.rows, pl.rows + R)};
            auto it = graphs_.find(key);
            if (it == graphs_.end()) {
                // first sighting: run directly, which also grows every scratch buffer to size
                forward(pl, bufs, max_pos, fs);
                if (graphs_.size() < kMaxGraphs) graphs_.emplace(key, Graph{nullptr, fs});
            } else if (!it->second.exec) {
                // Record this shape. A capture can be invalidated from outside (another thread's
                // device-wide synchronisation, e.g. cudaFree): then nothing recorded has run, so
                // the step runs op by op and the shape is recorded again at its next sighting.
                bool recorded = false;
                if (b_->graph_begin()) {
                    try {
                        forward(pl, bufs, max_pos, fs);
                        it->second.exec = b_->graph_end();
                        recorded = true;
                    } catch (const std::exception &) {
                        b_->graph_abort();
                        fs = sk_step_stats{};
                        ++graph_failures_;
                    }
                }
                if (recorded) b_->graph_launch(it->second.exec);
                else forward(pl, bufs, max_pos, fs);
            } else {
                b_->graph_launch(it->second.exec);
                fs = it->second.counters;
            }
        }
        stats.gpu_pairs += fs.gpu_pairs;
        stats.cpu_pairs += fs.cpu_pairs;
        stats.distinct_experts += fs.distinct_experts;
        stats.expert_bytes += fs.expert_bytes;
        stats.ms_experts += fs.ms_experts;
        stats.ms_gpu_wait += fs.ms_gpu_wait;

        t = Clock::now();
        if (tokens_out) {
            // the argmax or the draw on the device, R ids (and a non-finite flag) to the host
            step_tokens(b_.get(), logits_d_, R, d_.vocab, sample, io, d_argmax_, d_sample_, sample_scratch_, argmax_host_,
                        tokens_out);
        } else {
            b_->download(logits_host_, logits_d_, size_t(R) * d_.vocab * 4);
            b_->sync();
            std::memcpy(logits_out, logits_host_, size_t(R) * d_.vocab * 4);
        }
        stats.ms_download += ms_since(t);
        if (place_on_) place_tick();  // between steps: the device is idle
        stats.ms_total = ms_since(t_start);
        if (st) *st = stats;
    }

    // Workspace of one step: the residual stream and its scratch, then the mark that every
    // layer's own scratch is taken above.
    struct Bufs {
        float *h, *pref, *hin, *tmp, *blocks;
        size_t sub;
    };

    // Every layer, the output aggregation and the head on the requested rows: logits_d_.
    // Queues device work only (unless experts run on the CPU), so it can be graph-recorded.
    void forward(const sk_step_plan &pl, const Bufs &bf, int64_t max_pos, sk_step_stats &stats) {
        const int64_t N = pl.n_tokens, R = pl.n_rows, E = d_.hidden;
        float *h = bf.h, *pref = bf.pref, *hin = bf.hin, *tmp = bf.tmp, *blocks = bf.blocks;
        const size_t sub = bf.sub;
        int64_t nb = 0;
        for (int l = 0; l < d_.n_layers; ++l) {
            const Layer &L = layers_[l];
            if (!d_.attn_res) {  // plain pre-norm residuals (Kimi Linear)
                // each residual add is fused with the pre-norm that follows it
                if (l == 0) b_->rmsnorm(hin, h, L.in_norm, N, E, E, d_.rms_eps);
                ws_->reset(sub);
                if (L.mla) mla(L, hin, N, max_pos, tmp, pl);
                else kda(L, hin, N, tmp, pl);
                b_->add_rmsnorm(h, tmp, hin, L.post_norm, N, E, d_.rms_eps);
                ws_->reset(sub);
                if (L.dense) dense(L, hin, N, tmp);
                else moe(L, l, hin, N, tmp, stats);
                if (l + 1 < d_.n_layers) b_->add_rmsnorm(h, tmp, hin, layers_[l + 1].in_norm, N, E, d_.rms_eps);
                else b_->add(h, tmp, N * E);
                continue;
            }
            b_->copy(pref, h, N * E * 4);
            // AttnRes before attention, only once snapshots exist. It REPLACES h.
            if (nb > 0) b_->attn_res(h, blocks, nb, maxs_, pref, L.fold_attn, N, E, d_.rms_eps);
            // Block boundary: snapshot the running residual, then CLEAR it.
            bool have_prefix = true;
            if (l % d_.attn_res_block == 0) {
                b_->copy_rows(blocks, maxs_ * E, nb * E, pref, N, E);
                ++nb;
                have_prefix = false;
            }
            b_->rmsnorm(hin, h, L.in_norm, N, E, E, d_.rms_eps);
            ws_->reset(sub);
            if (L.mla) mla(L, hin, N, max_pos, tmp, pl);
            else kda(L, hin, N, tmp, pl);
            if (have_prefix) b_->add(pref, tmp, N * E);
            else b_->copy(pref, tmp, N * E * 4);

            // AttnRes before the FFN: unconditional (layer 0 already pushed a snapshot).
            b_->attn_res(h, blocks, nb, maxs_, pref, L.fold_mlp, N, E, d_.rms_eps);
            b_->rmsnorm(hin, h, L.post_norm, N, E, E, d_.rms_eps);
            ws_->reset(sub);
            if (L.dense) dense(L, hin, N, tmp);
            else moe(L, l, hin, N, tmp, stats);
            b_->add(pref, tmp, N * E);
            b_->copy(h, pref, N * E * 4);
        }

        // output aggregation for every row (AttnRes models), then the head on the requested rows
        const float *fin_src = h;
        if (d_.attn_res) {
            b_->attn_res(tmp, blocks, nb, maxs_, h, fold_out_, N, E, d_.rms_eps);
            fin_src = tmp;
        }
        for (int64_t i = 0; i < R; ++i) b_->copy(fin_ + i * E, fin_src + int64_t(pl.rows[i]) * E, E * 4);
        b_->rmsnorm(fin_, fin_, final_norm_, R, E, E, d_.rms_eps);
        b_->gemm(lm_head_, fin_, R, logits_d_, d_.vocab);
    }

    void validate(const sk_step_plan &pl) const {
        auto fail = [](const std::string &m) { throw std::runtime_error("bad step plan: " + m); };
        if (pl.n_tokens <= 0 || pl.n_tokens > o_.max_tokens) fail("n_tokens " + std::to_string(pl.n_tokens) + " outside 1.." + std::to_string(o_.max_tokens));
        if (pl.n_rows <= 0 || pl.n_rows > o_.max_rows) fail("n_rows " + std::to_string(pl.n_rows) + " outside 1.." + std::to_string(o_.max_rows));
        if (pl.n_segs <= 0 || pl.n_segs > o_.n_slots) fail("n_segs");
        const int64_t cap = int64_t(o_.max_pages_per_seq) * o_.page;
        for (int64_t i = 0; i < pl.n_tokens; ++i) {
            if (pl.tokens[i] < 0 || pl.tokens[i] >= d_.vocab) fail("token id out of vocabulary");
            if (pl.positions[i] < 0 || pl.positions[i] >= cap) fail("position beyond the page table");
            if (pl.token_slot[i] < 0 || pl.token_slot[i] >= o_.n_slots) fail("slot out of range");
            const int32_t pg = pl.page_table[int64_t(pl.token_slot[i]) * o_.max_pages_per_seq + pl.positions[i] / o_.page];
            if (pg < 0 || pg >= o_.n_pages) fail("page id out of range for a token's position");
        }
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
        for (int i = 0; i < pl.n_reset; ++i)
            if (pl.reset_slots[i] < 0 || pl.reset_slots[i] >= o_.n_slots) fail("reset slot out of range");
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

    // ------------------------------------------------------------------ KDA
    void kda(const Layer &L, const float *x, int64_t N, float *out, const sk_step_plan &pl) {
        const int64_t H = d_.kda_heads, D = d_.kda_head_dim, P = H * D, K = d_.conv_k;
        float *qp = ws_->take<float>(N * P), *kp = ws_->take<float>(N * P), *vp = ws_->take<float>(N * P);
        float *q = ws_->take<float>(N * P), *k = ws_->take<float>(N * P), *v = ws_->take<float>(N * P);
        float *z = ws_->take<float>(N * P), *alpha = ws_->take<float>(N * P);
        float *gate = ws_->take<float>(N * P), *o = ws_->take<float>(N * P);
        float *fa = ws_->take<float>(N * D), *braw = ws_->take<float>(N * H), *beta = ws_->take<float>(N * H);
        float *ga = ws_->take<float>(N * D);

        // every projection of x in one group, then the second halves of the low-rank ones
        const bool lowrank = d_.kda_gate_lowrank != 0;  // output gate g_b(g_a(x)) or g(x)
        const Backend::GemmJob in[] = {{&L.q, x, qp, P},   {&L.k, x, kp, P},
                                       {&L.v, x, vp, P},   {&L.b, x, braw, H},
                                       {&L.f_a, x, fa, D}, lowrank ? Backend::GemmJob{&L.g_a, x, ga, D}
                                                                   : Backend::GemmJob{&L.g, x, gate, P}};
        b_->gemm_group(in, 6, N);
        const Backend::GemmJob lr[] = {{&L.f_b, fa, z, P}, {&L.g_b, ga, gate, P}};
        b_->gemm_group(lr, lowrank ? 2 : 1, N);

        const int64_t hist = P * (K - 1);
        float *conv = conv_pool_ + int64_t(L.kda_i) * 3 * hist;
        const int64_t conv_stride = int64_t(n_kda_) * 3 * hist;
        char *s = static_cast<char *>(s_pool_) + int64_t(L.kda_i) * kda_s_elems() * dt_size(st_dt_);
        Backend::KdaFused f;
        f.o = o;
        f.qp = qp, f.kp = kp, f.vp = vp;
        f.wq = L.q_conv, f.wk = L.k_conv, f.wv = L.v_conv;
        f.conv_state = conv, f.conv_stride = conv_stride;
        f.z = z, f.braw = braw, f.a_log = L.a_log, f.dt_bias = L.dt_bias;
        f.lb = d_.gate_lb, f.softplus = d_.decay_softplus != 0;
        f.gate = gate, f.o_norm = L.o_norm;
        f.out_eps = d_.rms_eps, f.l2_eps = 1e-6f, f.qscale = 1.0f / std::sqrt(float(D));
        f.state = s, f.state_dtype = st_dt_, f.state_stride = int64_t(n_kda_) * kda_s_elems();
        f.seg = d_segs_, f.nseg = pl.n_segs, f.H = H, f.D = D, f.K = K;
        if (n_long_ > 0) {  // prompts in chunks, the other segments token by token
            Backend::KdaFused fl = f;
            fl.seg = d_segs_long_, fl.nseg = n_long_;
            if (b_->kda_chunked(fl, segs_long_.data())) {
                Backend::KdaFused fs = f;
                fs.seg = d_segs_short_, fs.nseg = n_short_;
                if (n_short_ > 0 && !b_->kda_fused(fs)) throw std::runtime_error("kda_chunked without kda_fused");
                b_->gemm(L.o, o, N, out, d_.hidden);
                return;
            }
        }
        if (b_->kda_fused(f)) {
            b_->gemm(L.o, o, N, out, d_.hidden);
            return;
        }
        b_->kda_conv(q, qp, L.q_conv, conv, conv_stride, 0, d_segs_, pl.n_segs, P, K);
        b_->kda_conv(k, kp, L.k_conv, conv, conv_stride, hist, d_segs_, pl.n_segs, P, K);
        b_->kda_conv(v, vp, L.v_conv, conv, conv_stride, 2 * hist, d_segs_, pl.n_segs, P, K);
        b_->l2norm_heads(q, N, H, D, 1e-6f);
        b_->l2norm_heads(k, N, H, D, 1e-6f);
        b_->kda_gates(alpha, beta, z, braw, L.a_log, L.dt_bias, N, H, D, d_.gate_lb, d_.decay_softplus != 0);
        b_->kda_recurrence(o, q, k, v, alpha, beta, s, st_dt_, int64_t(n_kda_) * kda_s_elems(), d_segs_, pl.n_segs, H,
                           D, 1.0f / std::sqrt(float(D)));
        // head-wise RMSNorm, THEN the gate, THEN o_proj (MLA gates with no norm)
        b_->kda_out(o, gate, L.o_norm, N, H, D, d_.rms_eps);
        b_->gemm(L.o, o, N, out, d_.hidden);
    }

    // ------------------------------------------------------------------ MLA
    // The step's segments of at least mla_prefill_min tokens take mla_prefill (their keys and
    // values expanded from the latent: worth it for many queries, never for a decode), whatever
    // their context; the other tokens' rows (decodes, short prefills) keep the absorbed kernels,
    // compacted.
    void plan_mla(const sk_step_plan &pl) {
        n_mla_long_ = n_mla_short_ = 0;
        const int64_t min = b_->mla_prefill_min();
        if (n_mla_ == 0 || min <= 0) return;
        mla_cu_q_.assign(1, 0), mla_cu_k_.assign(1, 0), mla_slot_.clear(), mla_first_.clear();
        short_rows_.clear(), short_tslot_.clear(), short_pos_.clear();
        mla_max_q_ = mla_max_k_ = 0, short_max_pos_ = 0;
        for (int64_t i = 0; i < pl.n_segs; ++i) {
            const int32_t slot = pl.segs[3 * i], first = pl.segs[3 * i + 1], count = pl.segs[3 * i + 2];
            if (count <= 0) continue;
            const int32_t keys = pl.positions[first + count - 1] + 1;
            if (count >= min) {
                mla_cu_q_.push_back(mla_cu_q_.back() + count), mla_cu_k_.push_back(mla_cu_k_.back() + keys);
                mla_slot_.push_back(slot), mla_first_.push_back(first);
                mla_max_q_ = std::max<int64_t>(mla_max_q_, count), mla_max_k_ = std::max<int64_t>(mla_max_k_, keys);
            } else {
                for (int32_t r = first; r < first + count; ++r) {
                    short_rows_.push_back(r), short_tslot_.push_back(pl.token_slot[r]);
                    short_pos_.push_back(pl.positions[r]);
                    short_max_pos_ = std::max<int64_t>(short_max_pos_, pl.positions[r]);
                }
            }
        }
        n_mla_long_ = int64_t(mla_slot_.size());
        if (n_mla_long_ == 0) return;
        n_mla_short_ = int64_t(short_rows_.size());
        std::vector<int32_t> &meta = mla_meta_;
        meta.assign(mla_cu_q_.begin(), mla_cu_q_.end());
        meta.insert(meta.end(), mla_cu_k_.begin(), mla_cu_k_.end());
        meta.insert(meta.end(), mla_slot_.begin(), mla_slot_.end());
        meta.insert(meta.end(), mla_first_.begin(), mla_first_.end());
        b_->upload(d_mla_meta_, meta.data(), meta.size() * 4);
        if (n_mla_short_ > 0) {
            b_->upload(d_short_rows_, short_rows_.data(), n_mla_short_ * 4);
            b_->upload(d_short_tslot_, short_tslot_.data(), n_mla_short_ * 4);
            b_->upload(d_short_pos_, short_pos_.data(), n_mla_short_ * 4);
        }
    }

    void mla(const Layer &L, const float *x, int64_t N, int64_t max_pos, float *out, const sk_step_plan &) {
        const int64_t H = d_.n_heads, qn = d_.qk_nope, qr = d_.qk_rope, vh = d_.v_head, lat = d_.kv_lora;
        const int64_t qh = qn + qr, W = width();
        float *ql = ws_->take<float>(N * std::max(1, d_.q_lora)), *q = ws_->take<float>(N * H * qh);
        float *ckv = ws_->take<float>(N * W), *q_abs = ws_->take<float>(N * H * W);
        float *o_lat = ws_->take<float>(N * H * lat), *y = ws_->take<float>(N * H * vh);
        float *g = ws_->take<float>(N * H * vh);

        const Backend::GemmJob in[] = {
            d_.q_lora > 0 ? Backend::GemmJob{&L.q_a, x, ql, d_.q_lora} : Backend::GemmJob{&L.mq, x, q, H * qh},
            {&L.kv_a, x, ckv, W}};
        b_->gemm_group(in, 2, N);
        if (d_.q_lora > 0) {
            b_->rmsnorm(ql, ql, L.q_a_norm, N, d_.q_lora, d_.q_lora, d_.rms_eps);
            b_->gemm(L.q_b, ql, N, q, H * qh);
        }
        // the norm covers the latent only, never the rope slot
        b_->rmsnorm(ckv, ckv, L.kv_a_norm, N, lat, W, d_.kv_norm_eps);
        void *kv = kv_pools_[L.mla_i];
        b_->mla_append(kv, kv_dt_, ckv, d_tslot_, d_pos_, d_pt_, o_.max_pages_per_seq, o_.page, W, N);
        // scale over the FULL head width, rope slots included; NoPE: no rotation anywhere
        const float scale = 1.0f / std::sqrt(float(qh));
        Backend::MlaPrefill m{};
        m.y = y, m.q = q, m.kv = kv, m.kv_dtype = kv_dt_, m.kv_b = &L.kv_b;
        m.page_table = d_pt_, m.max_pages = o_.max_pages_per_seq, m.page = o_.page;
        m.meta = d_mla_meta_, m.meta_host = mla_meta_.data();
        m.n = n_mla_long_, m.total_q = mla_cu_q_.back(), m.total_k = mla_cu_k_.back();
        m.max_q = mla_max_q_, m.max_k = mla_max_k_;
        m.H = H, m.qn = qn, m.qr = qr, m.vh = vh, m.kv_lora = lat, m.scale = scale;
        if (n_mla_long_ > 0 && b_->mla_prefill(m)) {
            if (const int64_t Ns = n_mla_short_; Ns > 0) {  // the other rows, absorbed
                float *qs = ws_->take<float>(Ns * H * qh), *ys = ws_->take<float>(Ns * H * vh);
                b_->index_rows(qs, q, d_short_rows_, Ns, H * qh, false);
                b_->mla_absorb_q(q_abs, qs, L.kv_b, Ns, H, qn, qr, vh);
                b_->mla_attention(o_lat, q_abs, kv, kv_dt_, d_short_tslot_, d_short_pos_, d_pt_,
                                  o_.max_pages_per_seq, o_.page, Ns, H, lat, qr, short_max_pos_, scale);
                b_->mla_unabsorb(ys, o_lat, L.kv_b, Ns, H, qn, vh);
                b_->index_rows(y, ys, d_short_rows_, Ns, H * vh, true);
            }
        } else {
            b_->mla_absorb_q(q_abs, q, L.kv_b, N, H, qn, qr, vh);
            b_->mla_attention(o_lat, q_abs, kv, kv_dt_, d_tslot_, d_pos_, d_pt_, o_.max_pages_per_seq, o_.page, N, H,
                              lat, qr, max_pos, scale);
            b_->mla_unabsorb(y, o_lat, L.kv_b, N, H, qn, vh);
        }
        if (L.has_gate) {  // gate BEFORE o_proj, no norm
            b_->gemm(L.mg, x, N, g, H * vh);
            b_->sigmoid_mul(y, g, N * H * vh);
        }
        b_->gemm(L.mo, y, N, out, d_.hidden);
    }

    // ------------------------------------------------------------------ FFN
    void dense(const Layer &L, const float *x, int64_t N, float *out) {
        const int64_t DI = d_.dense_inter;
        float *g = ws_->take<float>(N * DI), *u = ws_->take<float>(N * DI), *a = ws_->take<float>(N * DI);
        const Backend::GemmJob in[] = {{&L.d_gate, x, g, DI}, {&L.d_up, x, u, DI}};
        b_->gemm_group(in, 2, N);
        glu(a, g, u, N * DI);
        b_->gemm(L.d_down, a, N, out, d_.hidden);
    }

    // Stable LatentMoE: route on the FULL width -> down-project -> routed experts in latent
    // space (CPU) -> RMSNorm the AGGREGATE -> up-project -> + shared expert on the ORIGINAL
    // input, unweighted.
    void moe(const Layer &L, int layer, const float *x, int64_t N, float *out, sk_step_stats &st) {
        const int64_t E = d_.hidden, Lt = d_.latent, k = d_.topk, SI = int64_t(d_.moe_inter) * d_.n_shared;
        float *logits = ws_->take<float>(N * d_.n_experts), *w = ws_->take<float>(N * k);
        // K3 runs the experts in a latent space; Kimi Linear at full width on x itself.
        float *zl = L.latent ? ws_->take<float>(N * Lt) : nullptr, *acc = ws_->take<float>(N * Lt);
        const float *z = L.latent ? zl : x;
        float *sg = ws_->take<float>(N * SI), *su = ws_->take<float>(N * SI), *sa = ws_->take<float>(N * SI);
        float *so = ws_->take<float>(N * E);

        // the router, the latent down-projection and the shared expert's gate/up all read x
        Backend::GemmJob in[4] = {{&L.router, x, logits, d_.n_experts},
                                  {&L.sh_gate, x, sg, SI},
                                  {&L.sh_up, x, su, SI}};
        int nin = 3;
        if (L.latent) in[nin++] = {&L.down, x, zl, Lt};
        b_->gemm_group(in, nin, N);
        b_->router_topk(d_idx_, w, logits, L.router_bias, N, d_.n_experts, k, d_.moe_renorm != 0, d_.routed_scale,
                        d_.router_biased != 0);
        if (route_stats_) count_routes(layer, N);
        if (place_on_) b_->count_routes(d_route_n_ + int64_t(layer) * d_.n_experts, d_idx_, N * k, d_.n_experts);
        // At full width the reduce writes the layer output and adds the shared expert itself.
        float *red = L.latent ? acc : out;
        const float *red_add = L.latent ? nullptr : so;

        // Pairs whose expert is resident in backend memory run there; the rest on the CPU pool.
        // Rows from both land in one [pair][width] buffer and are summed in selection order.
        const int64_t NK = N * k, I = d_.moe_inter;
        const int n_res = gx_count_.empty() ? 0 : gx_count_[layer];
        const bool pool = n_res < d_.n_experts;  // some pair may run on the CPU pool
        // The pool's inputs go to the host first, ahead of the resident experts and the shared
        // expert on the stream: the host waits for the router's outputs only, and the GPU's experts
        // run while the CPU's do. (Queued after the resident experts, the copies had the CPU wait
        // for them: 3.0 ms a step at one token, 11.7 at 16, Kimi Linear with ~82% resident.)
        const float *zh = z, *wh = w;
        const int32_t *ih = d_idx_;
        void *ev = nullptr;
        if (pool && !b_->on_host()) {
            b_->download(z_host_, z, N * Lt * 4);
            b_->download(idx_host_, d_idx_, N * k * 4);
            b_->download(w_host_, w, N * k * 4);
            ev = b_->record_event();
            zh = z_host_, wh = w_host_, ih = idx_host_;
        }
        float *contrib = nullptr;
        if (n_res > 0) {
            contrib = ws_->take<float>(NK * Lt);
            float *msc = ws_->take<float>(NK * I);
            b_->moe_experts(contrib, z, d_idx_, w, N, k, gx_dev_[layer], d_.n_experts, gpu_fmt_, Lt, I,
                            d_.act_silu != 0, d_.situ_b1, d_.situ_b2, msc);
        }
        int64_t active = 0;
        if (!pool) {
            // every expert resident: no host round trip at all
            shared(L, sg, su, sa, so, N);
            b_->moe_reduce(red, contrib, N, k, Lt, red_add);
            st.gpu_pairs += NK;
        } else {
            shared(L, sg, su, sa, so, N);  // on the device: queued behind the resident experts, overlaps the pool
            if (ev) {
                auto t = Clock::now();
                b_->wait_event(ev);
                b_->release_event(ev);
                st.ms_gpu_wait += ms_since(t);
            }
            auto t = Clock::now();
            if (n_res == 0) {
                float *acc_out = b_->on_host() ? acc : acc_host_;
                active = experts_->run(expert_table_[layer], zh, N, ih, wh, acc_out);
                if (!b_->on_host()) b_->upload(acc, acc_host_, N * Lt * 4);
                st.cpu_pairs += NK;
            } else {
                int64_t nsel = 0;
                for (int64_t p = 0; p < NK; ++p)
                    if (!resident_[layer][ih[p]]) sel_host_[nsel++] = static_cast<int32_t>(p);
                active = experts_->run_pairs(expert_table_[layer], zh, N, ih, wh, sel_host_, nsel, compact_host_);
                if (b_->on_host()) {
                    b_->scatter_rows(contrib, compact_host_, sel_host_, nsel, Lt);
                } else if (nsel > 0) {
                    float *cd = ws_->take<float>(NK * Lt);
                    int32_t *sd = ws_->take<int32_t>(NK);
                    b_->upload(cd, compact_host_, nsel * Lt * 4);
                    b_->upload(sd, sel_host_, nsel * 4);
                    b_->scatter_rows(contrib, cd, sd, nsel, Lt);
                }
                b_->moe_reduce(red, contrib, N, k, Lt, red_add);
                st.cpu_pairs += nsel;
                st.gpu_pairs += NK - nsel;
            }
            st.ms_experts += ms_since(t);
        }
        st.distinct_experts += active;
        st.expert_bytes += active * experts_->expert_bytes();

        if (L.latent) {
            if (L.latent_norm) b_->rmsnorm(acc, acc, L.latent_norm, N, Lt, Lt, d_.rms_eps);
            b_->gemm(L.up, acc, N, out, E);
            b_->add(out, so, N * E);
        } else if (n_res == 0) {  // the CPU pool summed into acc
            b_->copy(out, acc, N * E * 4);
            b_->add(out, so, N * E);
        }
    }

    // ------------------------------------------------------------------ expert placement
    // Every place_every_ steps (FULLM_EXPERT_PLACE; 0: never) the device's route counts decay into
    // place_score_ (x place_decay_ each time, then the window's counts added), and per layer the most
    // routed experts outside GPU memory take the slots of the least routed inside, while the one
    // coming in scores over 1.25x the one going out plus 4 (hysteresis); at most place_swaps_ a
    // time (FULLM_EXPERT_SWAPS), the largest gains first. The new experts' bytes are prepared on
    // the CPU pool's threads (copied, gate / up interleaved, or quantised as at load) and written
    // over the old ones' in place; the tables the kernels read are uploaded, so recorded graphs
    // stay valid. FULLM_EXPERT_LOG=1: one line each time on stderr.
    struct HostExpert {
        std::vector<uint8_t> w[3];  // ExpertW order: w1 (or gate / up interleaved), w3, w2
        std::vector<float> s[3];
    };
    HostExpert host_expert(int l, int e, bool interleaved) const {
        const ExpertPtrs &p = expert_table_[l][e];
        const int64_t Lt = d_.latent, I = d_.moe_inter;
        HostExpert h;
        auto enc = [&](int j, const uint16_t *src, int64_t rows, int64_t cols) {
            if (gpu_fmt_ == WFmt::BF16) {
                h.w[j].assign(reinterpret_cast<const uint8_t *>(src), reinterpret_cast<const uint8_t *>(src + rows * cols));
            } else if (gpu_fmt_ == WFmt::FP8B) {
                h.w[j].resize(size_t(rows * cols)), h.s[j].resize(size_t(fp8b_scales(rows, cols)));
                quantize_fp8_blocks(src, rows, cols, h.w[j].data(), h.s[j].data());
            } else {
                h.w[j].resize(size_t(int4g_bytes(rows, cols))), h.s[j].resize(size_t(int4g_scales(rows, cols)));
                quantize_int4_groups(src, rows, cols, h.w[j].data(), h.s[j].data(),
                                     b_->weight_layout(WFmt::INT4G, rows, cols));
            }
        };
        const auto *g = static_cast<const uint16_t *>(static_cast<const void *>(p.p1));
        const auto *u = static_cast<const uint16_t *>(static_cast<const void *>(p.p3));
        const auto *dn = static_cast<const uint16_t *>(static_cast<const void *>(p.p2));
        if (interleaved) {
            std::vector<uint16_t> w13(size_t(2 * I * Lt));
            for (int64_t r = 0; r < I; ++r) {
                std::memcpy(w13.data() + (2 * r) * Lt, g + r * Lt, size_t(Lt) * 2);
                std::memcpy(w13.data() + (2 * r + 1) * Lt, u + r * Lt, size_t(Lt) * 2);
            }
            enc(0, w13.data(), 2 * I, Lt);
        } else {
            enc(0, g, I, Lt);
            enc(1, u, I, Lt);
        }
        enc(2, dn, Lt, I);
        return h;
    }
    // At a step boundary (the device idle): install the experts a finished fill brought in, then,
    // every place_every_ steps with no fill outstanding, choose the next swaps, take the outgoing
    // experts out of the tables (their pairs go to the CPU from the next step) and hand their
    // slots to the placer thread, which writes the incoming experts there on the copy stream while
    // the steps go on. Nothing waits for a copy; a slot is in no table while it is written.
    void place_tick() {
        if (!place_pending_.empty()) {
            void *ev = nullptr;
            std::string err;
            bool busy = true;
            {
                std::lock_guard<std::mutex> lk(place_mu_);
                busy = place_busy_, ev = place_event_, err = place_err_;
            }
            if (!err.empty()) throw std::runtime_error("expert placement: " + err);
            if (busy || !b_->event_done(ev)) return;
            const auto t_in = Clock::now();
            std::vector<uint8_t> changed(size_t(d_.n_layers), 0);
            for (const PlaceSwap &sw : place_pending_) {
                gx_host_[sw.l][sw.in] = sw.slot;
                resident_[sw.l][sw.in] = 1;
                changed[size_t(sw.l)] = 1;
            }
            upload_tables(changed);
            b_->release_event(ev);
            {
                std::lock_guard<std::mutex> lk(place_mu_);
                place_event_ = nullptr;
            }
            place_total_swaps_ += int64_t(place_pending_.size());
            if (std::getenv("FULLM_EXPERT_LOG"))
                std::fprintf(stderr,
                             "expert placement: %zu experts in (%lld in all) %.1f ms after their slots were freed, "
                             "installed in %.2f ms; the window's CPU routes %.1f%% -> %.1f%%\n",
                             place_pending_.size(), static_cast<long long>(place_total_swaps_), ms_since(place_t0_),
                             ms_since(t_in), place_before_, place_after_);
            place_pending_.clear();
            return;  // the next choice after the next window
        }
        if (++place_steps_ < place_every_) return;
        place_steps_ = 0;
        const auto t_choose = Clock::now();
        const int64_t E = d_.n_experts, Ln = d_.n_layers;
        std::vector<uint32_t> c(size_t(Ln * E));
        b_->download(c.data(), d_route_n_, c.size() * 4);
        b_->sync();
        b_->zero(d_route_n_, c.size() * 4);
        for (size_t i = 0; i < c.size(); ++i) place_score_[i] = place_decay_ * place_score_[i] + c[i];
        // the CPU's share of the routes counted in w (this window's, or the decayed scores), now or
        // once some swaps are in
        auto share = [&](const auto &w, const std::vector<PlaceSwap> *with) {
            std::vector<uint8_t> res(size_t(Ln * E), 0);
            for (int l = 0; l < Ln; ++l)
                for (int e = 0; e < E && !gx_host_[l].empty(); ++e) res[size_t(l) * E + e] = gx_host_[l][e].w1.data != nullptr;
            if (with)
                for (const PlaceSwap &sw : *with) res[size_t(sw.l) * E + sw.in] = 1, res[size_t(sw.l) * E + sw.out] = 0;
            double cpu = 0, all = 0;
            for (int l = 0; l < Ln; ++l)
                for (int e = 0; e < E && !gx_host_[l].empty(); ++e) {
                    const double v = double(w[size_t(l) * E + e]);
                    all += v;
                    if (!res[size_t(l) * E + e]) cpu += v;
                }
            return all > 0 ? 100.0 * cpu / all : 0.0;
        };
        const bool log = std::getenv("FULLM_EXPERT_LOG") != nullptr;
        // Frozen (the placement converged): no swaps, so a prompt's tokens stay the same run after
        // run; the counts still come in, and place_thaw_ windows in a row whose CPU share has
        // clearly risen (over the frozen share + 5 points and 2x: the traffic moved, not a burst)
        // thaw it (FULLM_EXPERT_THAW).
        if (place_frozen_) {
            const double now = share(c, nullptr);
            place_drift_ = now > place_frozen_share_ + 5.0 && now > 2.0 * place_frozen_share_ ? place_drift_ + 1 : 0;
            if (place_drift_ < place_thaw_) return;
            place_frozen_ = false, place_quiet_ = 0, place_drift_ = 0;
            if (log)
                std::fprintf(stderr, "expert placement: thawed, the window's CPU routes %.1f%% (frozen at %.1f%%)\n", now,
                             place_frozen_share_);
        }
        std::vector<PlaceSwap> cand;
        std::vector<double> gain;
        for (int l = 0; l < Ln; ++l) {
            if (gx_host_[l].empty()) continue;
            const double *sc = place_score_.data() + int64_t(l) * E;
            std::vector<int> res, non;
            for (int e = 0; e < E; ++e) (gx_host_[l][e].w1.data ? res : non).push_back(e);
            std::sort(non.begin(), non.end(), [&](int a, int b) { return sc[a] > sc[b] || (sc[a] == sc[b] && a < b); });
            std::sort(res.begin(), res.end(), [&](int a, int b) { return sc[a] < sc[b] || (sc[a] == sc[b] && a < b); });
            for (size_t j = 0; j < std::min(res.size(), non.size()); ++j) {
                const double a = sc[non[j]], b = sc[res[j]];
                if (a <= 1.25 * b + 4.0) break;
                cand.push_back({l, non[j], res[j], Backend::ExpertW{}});
                gain.push_back(a - b);
            }
        }
        std::vector<size_t> order(cand.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t x, size_t y) {
            return gain[x] > gain[y] || (gain[x] == gain[y] && (cand[x].l < cand[y].l || (cand[x].l == cand[y].l && cand[x].in < cand[y].in)));
        });
        if (int64_t(order.size()) > place_swaps_) order.resize(size_t(place_swaps_));
        std::vector<PlaceSwap> job;
        for (size_t i : order) job.push_back(cand[i]);
        const double before = share(c, nullptr), after = share(c, &job);
        // Quiet: the swaps would move under a point of the routes, judged on the decayed scores (one
        // window's counts are noisy). After place_freeze_ quiet windows in a row (FULLM_EXPERT_FREEZE;
        // 0: never) the placement freezes, these swaps not made.
        const double sb = share(place_score_, nullptr), sa = share(place_score_, &job);
        place_quiet_ = sb - sa < 1.0 ? place_quiet_ + 1 : 0;
        if (place_freeze_ > 0 && place_quiet_ >= place_freeze_) {
            // the baseline a drift is judged against: the decayed scores' share, not this window's
            // (a burst frozen in as the baseline would hold the thaw off)
            place_frozen_ = true, place_frozen_share_ = sb;
            if (log)
                std::fprintf(stderr,
                             "expert placement: frozen after %lld quiet windows, the CPU routes %.1f%% (decayed), %.1f%% "
                             "(this window)\n",
                             static_cast<long long>(place_quiet_), sb, before);
            return;
        }
        if (job.empty()) return;
        place_before_ = before, place_after_ = after;
        // the outgoing experts leave the tables now: their slots are free for the copy stream
        std::vector<uint8_t> changed(size_t(Ln), 0);
        for (PlaceSwap &sw : job) {
            sw.slot = gx_host_[sw.l][sw.out];
            gx_host_[sw.l][sw.out] = Backend::ExpertW{};
            resident_[sw.l][sw.out] = 0;
            changed[size_t(sw.l)] = 1;
        }
        upload_tables(changed);
        b_->sync();  // the device's tables no longer name the slots before the copies start
        place_t0_ = Clock::now();
        if (log)
            std::fprintf(stderr, "expert placement: %zu swaps chosen, their slots freed in %.2f ms (the step's wait)\n",
                         job.size(), ms_since(t_choose));
        place_pending_ = job;
        {
            std::lock_guard<std::mutex> lk(place_mu_);
            place_job_ = std::move(job);
            place_busy_ = true;
        }
        place_cv_.notify_one();
    }
    // The changed layers' tables, through pinned staging (a pageable copy could queue behind the
    // placer thread's).
    void upload_tables(const std::vector<uint8_t> &changed) {
        const int64_t E = d_.n_experts;
        for (int l = 0; l < d_.n_layers; ++l) {
            if (!changed[size_t(l)]) continue;
            Backend::ExpertW *st = place_stage_ + int64_t(l) * E;
            std::copy(gx_host_[l].begin(), gx_host_[l].end(), st);
            b_->upload(gx_dev_[l], st, size_t(E) * sizeof(Backend::ExpertW));
        }
    }
    // The placer thread: each job's incoming experts written into their slots on the copy stream
    // (bf16 straight from the experts' host bytes, gate / up interleaved by pitched copies; fp8b
    // and int4 quantised here first), then an event the step boundary polls.
    void place_worker() {
        const bool inter = b_->moe_gate_up_interleaved(gpu_fmt_, d_.latent, d_.moe_inter);
        const size_t Lt = size_t(d_.latent), I = size_t(d_.moe_inter);
        for (;;) {
            std::vector<PlaceSwap> job;
            {
                std::unique_lock<std::mutex> lk(place_mu_);
                place_cv_.wait(lk, [&] { return place_stop_ || !place_job_.empty(); });
                if (place_stop_) return;
                job.swap(place_job_);
            }
            void *ev = nullptr;
            std::string err;
            try {
                for (const PlaceSwap &sw : job) {
                    const ExpertPtrs &p = expert_table_[sw.l][sw.in];
                    if (gpu_fmt_ == WFmt::BF16) {
                        const size_t row = Lt * 2;
                        if (inter) {
                            b_->upload_async(const_cast<void *>(sw.slot.w1.data), 2 * row, p.p1, row, row, I);
                            b_->upload_async(static_cast<char *>(const_cast<void *>(sw.slot.w1.data)) + row, 2 * row, p.p3,
                                             row, row, I);
                        } else {
                            b_->upload_async(const_cast<void *>(sw.slot.w1.data), I * row, p.p1, I * row, I * row, 1);
                            b_->upload_async(const_cast<void *>(sw.slot.w3.data), I * row, p.p3, I * row, I * row, 1);
                        }
                        b_->upload_async(const_cast<void *>(sw.slot.w2.data), Lt * I * 2, p.p2, Lt * I * 2, Lt * I * 2, 1);
                    } else {
                        const HostExpert h = host_expert(sw.l, sw.in, inter);
                        const Weight *dst[3] = {&sw.slot.w1, &sw.slot.w3, &sw.slot.w2};
                        for (int j = 0; j < 3; ++j) {
                            if (h.w[j].empty()) continue;
                            b_->upload_async(const_cast<void *>(dst[j]->data), h.w[j].size(), h.w[j].data(), h.w[j].size(),
                                             h.w[j].size(), 1);
                            if (!h.s[j].empty())
                                b_->upload_async(const_cast<float *>(dst[j]->scale), h.s[j].size() * 4, h.s[j].data(),
                                                 h.s[j].size() * 4, h.s[j].size() * 4, 1);
                        }
                    }
                }
                ev = b_->record_copy_event();
            } catch (const std::exception &e) {
                err = e.what();
            }
            std::lock_guard<std::mutex> lk(place_mu_);
            place_event_ = ev, place_err_ = err, place_busy_ = false;
        }
    }
    void place_stop() {
        if (!place_thread_.joinable()) return;
        {
            std::lock_guard<std::mutex> lk(place_mu_);
            place_stop_ = true;
        }
        place_cv_.notify_one();
        place_thread_.join();
    }

    // FULLM_ROUTE_STATS=<file>: how often each (layer, expert) is routed to, over every step run op
    // by op (it turns graphs off), written as "layer,expert,count" lines when the
    // executor closes: the measurement behind choosing which experts live in GPU memory.
    void count_routes(int layer, int64_t N) {
        const int64_t k = d_.topk;
        std::vector<int32_t> idx(size_t(N * k));
        b_->download(idx.data(), d_idx_, idx.size() * 4);
        b_->sync();
        if (route_counts_.empty()) route_counts_.assign(size_t(d_.n_layers) * d_.n_experts, 0);
        for (int32_t e : idx)
            if (e >= 0 && e < d_.n_experts) ++route_counts_[size_t(layer) * d_.n_experts + e];
    }
    void write_route_stats() const {
        if (!route_stats_ || route_counts_.empty()) return;
        FILE *f = std::fopen(route_stats_, "w");
        if (!f) return;
        for (int l = 0; l < d_.n_layers; ++l)
            for (int e = 0; e < d_.n_experts; ++e)
                if (route_counts_[size_t(l) * d_.n_experts + e])
                    std::fprintf(f, "%d,%d,%lld\n", l, e, static_cast<long long>(route_counts_[size_t(l) * d_.n_experts + e]));
        std::fclose(f);
    }

    // The family's gated activation: SiLU-GLU (Kimi Linear) or SiTU-GLU (K3).
    void glu(float *y, const float *g, const float *u, int64_t n) {
        if (d_.act_silu) b_->silu_glu(y, g, u, n);
        else b_->situ_glu(y, g, u, n, d_.situ_b1, d_.situ_b2);
    }

    // The shared expert after its gate/up (computed with the router): so = down(glu(sg, su)).
    void shared(const Layer &L, const float *sg, const float *su, float *sa, float *so, int64_t N) {
        const int64_t SI = int64_t(d_.moe_inter) * d_.n_shared;
        glu(sa, sg, su, N * SI);
        b_->gemm(L.sh_down, sa, N, so, d_.hidden);
    }

    // ------------------------------------------------------------------ slots
    void save_slot(int32_t slot, void *dst) override {
        check_slot(slot);
        b_->download(dst, static_cast<char *>(s_pool_) + slot * slot_s_bytes(), slot_s_bytes());
        b_->download(static_cast<char *>(dst) + slot_s_bytes(), reinterpret_cast<char *>(conv_pool_) + slot * slot_conv_bytes(),
                     slot_conv_bytes());
        b_->sync();
    }
    void load_slot(int32_t slot, const void *src) override {
        check_slot(slot);
        b_->upload(static_cast<char *>(s_pool_) + slot * slot_s_bytes(), src, slot_s_bytes());
        b_->upload(reinterpret_cast<char *>(conv_pool_) + slot * slot_conv_bytes(),
                   static_cast<const char *>(src) + slot_s_bytes(), slot_conv_bytes());
        b_->sync();
    }
    void check_slot(int32_t s) const {
        if (!finalized_ || s < 0 || s >= o_.n_slots) throw std::runtime_error("slot out of range");
    }

    const char *describe() override {
        std::ostringstream os;
        const double gb = 1e9;
        os << "kimi_k3 executor on " << b_->name() << ": weights " << double(weight_bytes_) / gb << " GB, "
           << o_.n_slots << " slots x " << double(slot_bytes()) / gb << " GB state ("
           << (st_dt_ == DType::BF16 ? "bf16" : "f32") << "), latent cache " << n_mla_ << " x "
           << double(size_t(o_.n_pages) * o_.page * width() * dt_size(kv_dt_)) / gb << " GB ("
           << (int64_t(o_.n_pages) * o_.page) << " positions, " << (kv_dt_ == DType::BF16 ? "bf16" : "f32")
           << "), workspace " << double(workspace_bytes(o_.max_tokens)) / gb << " GB for " << o_.max_tokens
           << " tokens/step, " << experts_->threads() << " CPU expert threads";
        if (gx_used_) {
            int64_t n = 0;
            for (int c : gx_count_) n += c;
            os << ", " << n << " experts resident in " << b_->name() << " ("
               << (gpu_fmt_ == WFmt::BF16 ? "bf16" : gpu_fmt_ == WFmt::INT8 ? "int8" : "fp8") << ", "
               << double(gx_used_) / gb << " GB)";
            if (all_resident_ && graphs_on_ && !b_->on_host())
                os << ", CUDA graphs for steps of <= " << kGemvMaxRows << " tokens";
            if (place_on_) os << ", placed by routing frequency every " << place_every_ << " steps";
        }
        desc_ = os.str();
        return desc_.c_str();
    }

private:
    sk_k3_dims d_;
    sk_exec_opts o_;
    std::vector<uint8_t> is_mla_;
    DType kv_dt_ = DType::BF16, st_dt_ = DType::BF16;
    std::unique_ptr<Backend> b_;
    std::unique_ptr<ExpertPool> experts_;
    std::vector<std::vector<ExpertPtrs>> expert_table_;
    std::map<std::string, Weight> mats_;
    std::map<std::string, std::pair<float *, int64_t>> vecs_;
    std::vector<void *> owned_, host_owned_;
    size_t weight_bytes_ = 0;
    bool finalized_ = false;
    int64_t maxs_ = 0;
    int n_kda_ = 0, n_mla_ = 0;

    Weight embed_, lm_head_;
    const float *final_norm_ = nullptr, *fold_out_ = nullptr;
    std::vector<Layer> layers_;

    void *s_pool_ = nullptr;
    float *conv_pool_ = nullptr;
    std::vector<void *> kv_pools_;
    int32_t *d_pos_ = nullptr, *d_tslot_ = nullptr, *d_segs_ = nullptr, *d_pt_ = nullptr, *d_idx_ = nullptr;
    // KDA: the step's segments of at least kda_chunk_min tokens and the rest (host, device)
    std::vector<int32_t> segs_long_, segs_short_;
    int32_t *d_segs_long_ = nullptr, *d_segs_short_ = nullptr;
    int64_t n_long_ = 0, n_short_ = 0;
    // MLA (plan_mla): the long segments' cu_q, cu_k, slots and first rows (host; mla_meta_ and
    // d_mla_meta_ the four back to back), the other tokens' rows, slots and positions (host, device)
    std::vector<int32_t> mla_cu_q_{0}, mla_cu_k_{0}, mla_slot_, mla_first_, mla_meta_, short_rows_, short_tslot_, short_pos_;
    int32_t *d_mla_meta_ = nullptr, *d_short_rows_ = nullptr, *d_short_tslot_ = nullptr, *d_short_pos_ = nullptr;
    int64_t n_mla_long_ = 0, n_mla_short_ = 0, mla_max_q_ = 0, mla_max_k_ = 0, short_max_pos_ = 0;
    float *h_host_ = nullptr, *z_host_ = nullptr, *acc_host_ = nullptr, *w_host_ = nullptr, *logits_host_ = nullptr;
    int32_t *idx_host_ = nullptr, *argmax_host_ = nullptr, *d_argmax_ = nullptr;
    SampleRow *d_sample_ = nullptr;
    SampleScratch sample_scratch_;  // masks, adjustments, log-probabilities (step_tokens)
    float *logits_d_ = nullptr, *fin_ = nullptr;
    float *compact_host_ = nullptr;
    int32_t *sel_host_ = nullptr;
    // resident experts: arena, per-layer tables (host copy and device), per-expert flags
    WFmt gpu_fmt_ = WFmt::BF16;
    char *gx_arena_ = nullptr;
    size_t gx_cap_ = 0, gx_used_ = 0;
    std::vector<std::vector<Backend::ExpertW>> gx_host_;
    std::vector<Backend::ExpertW *> gx_dev_;
    std::vector<int> gx_count_;
    std::vector<std::vector<uint8_t>> resident_;
    std::unique_ptr<Arena> ws_;
    std::string desc_;
    // CUDA graphs of whole forward passes (see step)
    struct GraphKey {
        int64_t n_tokens;
        int32_t n_segs;
        std::vector<int32_t> rows;
        bool operator<(const GraphKey &o) const {
            return std::tie(n_tokens, n_segs, rows) < std::tie(o.n_tokens, o.n_segs, o.rows);
        }
    };
    struct Graph {
        void *exec;              // null until the second sighting records it
        sk_step_stats counters;  // the expert counters of one such step
    };
    static constexpr size_t kMaxGraphs = 256;
    std::map<GraphKey, Graph> graphs_;
    bool all_resident_ = false, graphs_on_ = true;
    const char *route_stats_ = std::getenv("FULLM_ROUTE_STATS");  // count_routes
    // expert placement (place_tick, place_worker)
    struct PlaceSwap {
        int l, in, out;
        Backend::ExpertW slot;  // the outgoing expert's bytes in GPU memory, for the incoming one
    };
    std::vector<PlaceSwap> place_pending_;  // the swaps being filled (main thread)
    std::vector<PlaceSwap> place_job_;      // handed to the placer thread (under place_mu_)
    std::thread place_thread_;
    std::mutex place_mu_;
    std::condition_variable place_cv_;
    bool place_busy_ = false, place_stop_ = false;  // under place_mu_
    void *place_event_ = nullptr;                   // the copy stream after a job (under place_mu_)
    std::string place_err_;                         // under place_mu_
    Backend::ExpertW *place_stage_ = nullptr;       // pinned: the tables on their way up
    Clock::time_point place_t0_;
    double place_before_ = 0, place_after_ = 0;
    bool place_on_ = false, cpu_bf16_ = false, place_frozen_ = false;
    int64_t place_freeze_ = 3, place_quiet_ = 0;  // quiet windows before freezing; quiet in a row
    int64_t place_thaw_ = 4, place_drift_ = 0;    // drifted windows before thawing; drifted in a row
    double place_decay_ = 0.8;                    // the scores' memory per window (FULLM_EXPERT_DECAY)
    double place_frozen_share_ = 0;               // the CPU's share of routes when it froze
    int64_t place_every_ = 0, place_swaps_ = 0, place_steps_ = 0, place_total_swaps_ = 0;
    uint32_t *d_route_n_ = nullptr;   // the device's route counts since the last placement, [layer][expert]
    std::vector<double> place_score_;  // decayed counts, [layer][expert]
    std::vector<int64_t> route_counts_;                             // [layer][expert]
    int64_t graph_failures_ = 0;  // captures that failed and ran op by op instead
};

void put_err(char *err, int n, const std::string &m) {
    if (err && n > 0) {
        std::strncpy(err, m.c_str(), size_t(n) - 1);
        err[n - 1] = 0;
    }
}

}  // namespace

std::unique_ptr<ExecIface> make_k3_exec(const sk_k3_dims &d, const sk_exec_opts &o) {
    return std::make_unique<K3Exec>(d, o);
}

}  // namespace fullm

struct sk_exec {
    std::unique_ptr<fullm::ExecIface> ex;
};

using fullm::put_err;

#define SK_TRY(body)                          \
    try {                                     \
        body;                                 \
        return 0;                             \
    } catch (const std::exception &ex) {      \
        put_err(err, err_len, ex.what());     \
        return 1;                             \
    } catch (...) {                           \
        put_err(err, err_len, "unknown error"); \
        return 1;                             \
    }

template <class Make>
static sk_exec *create(Make make, char *err, int err_len) {
    try {
        auto e = std::make_unique<sk_exec>();
        e->ex = make();
        return e.release();
    } catch (const std::exception &ex) {
        put_err(err, err_len, ex.what());
    } catch (...) {
        put_err(err, err_len, "unknown error");
    }
    return nullptr;
}

extern "C" {

sk_exec *sk_exec_create_k3(const sk_k3_dims *dims, const sk_exec_opts *opts, char *err, int err_len) {
    return create([&] { return fullm::make_k3_exec(*dims, *opts); }, err, err_len);
}

sk_exec *sk_exec_create_dense(const sk_dense_dims *dims, const sk_exec_opts *opts, char *err, int err_len) {
    return create([&] { return fullm::make_dense_exec(*dims, *opts); }, err, err_len);
}

void sk_exec_destroy(sk_exec *e) { delete e; }

int sk_exec_add_matrix(sk_exec *e, const char *name, int32_t fmt, int64_t rows, int64_t cols, const void *data,
                       const float *scale, char *err, int err_len) {
    SK_TRY(e->ex->add_matrix(name, fmt, rows, cols, data, scale))
}

int sk_exec_add_vector(sk_exec *e, const char *name, const float *data, int64_t n, char *err, int err_len) {
    SK_TRY(e->ex->add_vector(name, data, n))
}

int sk_exec_set_expert(sk_exec *e, int32_t layer, int32_t expert, const uint8_t *w1p, const uint8_t *w1s,
                       const uint8_t *w3p, const uint8_t *w3s, const uint8_t *w2p, const uint8_t *w2s) {
    try {
        e->ex->set_expert(layer, expert, fullm::ExpertPtrs{w1p, w1s, w3p, w3s, w2p, w2s});
        return 0;
    } catch (...) {
        return 1;
    }
}

int sk_exec_finalize(sk_exec *e, char *err, int err_len) { SK_TRY(e->ex->finalize()) }

int sk_exec_reserve_gpu_experts(sk_exec *e, int32_t fmt, size_t bytes, char *err, int err_len) {
    SK_TRY(e->ex->reserve_gpu_experts(fmt, bytes))
}

int sk_exec_reserve_lora(sk_exec *e, int32_t slots, int32_t max_rank, uint32_t targets, char *err, int err_len) {
    SK_TRY(e->ex->reserve_lora(slots, max_rank, targets))
}

int sk_exec_set_lora(sk_exec *e, int32_t slot, int32_t layer, int32_t target, const uint16_t *a, size_t a_len,
                     const uint16_t *b, size_t b_len, int32_t rank, float scale, char *err, int err_len) {
    SK_TRY(e->ex->set_lora(slot, layer, target, a, a_len, b, b_len, rank, scale))
}

int sk_exec_clear_lora(sk_exec *e, int32_t slot, char *err, int err_len) { SK_TRY(e->ex->clear_lora(slot)) }

int sk_exec_set_seq_lora(sk_exec *e, int32_t seq, int32_t slot, char *err, int err_len) {
    SK_TRY(e->ex->set_seq_lora(seq, slot))
}

int sk_exec_set_gpu_expert(sk_exec *e, int32_t layer, int32_t expert, const void *w1, const float *w1s, const void *w3,
                           const float *w3s, const void *w2, const float *w2s, char *err, int err_len) {
    SK_TRY(e->ex->set_gpu_expert(layer, expert, w1, w1s, w3, w3s, w2, w2s))
}

int sk_exec_expert_cache(sk_exec *e, const char *path, int32_t *state, char *err, int err_len) {
    SK_TRY(*state = e->ex->expert_cache(path))
}

size_t sk_exec_pool_bytes(const sk_exec *e) { return e->ex->pool_bytes(); }

int sk_exec_step(sk_exec *e, const sk_step_plan *plan, float *logits, sk_step_stats *stats, char *err, int err_len) {
    SK_TRY(e->ex->step(*plan, logits, stats, nullptr))
}

int sk_exec_step_argmax(sk_exec *e, const sk_step_plan *plan, int32_t *tokens, sk_step_stats *stats, char *err,
                        int err_len) {
    SK_TRY(e->ex->step(*plan, nullptr, stats, tokens))
}

int sk_exec_step_sample(sk_exec *e, const sk_step_plan *plan, const sk_sample_row *sample, const sk_sample_io *io,
                        int32_t *tokens, sk_step_stats *stats, char *err, int err_len) {
    SK_TRY(e->ex->step(*plan, nullptr, stats, tokens, sample, io))
}

int sk_exec_set_eagle(sk_exec *e, const sk_eagle_dims *dims, char *err, int err_len) { SK_TRY(e->ex->set_eagle(*dims)) }

int sk_exec_draft(sk_exec *e, const sk_draft_plan *plan, int32_t k_max, int32_t *drafts, char *err, int err_len) {
    SK_TRY(e->ex->draft(*plan, k_max, drafts))
}

size_t sk_exec_slot_bytes(const sk_exec *e) { return e->ex->slot_bytes(); }

int sk_exec_save_slot(sk_exec *e, int32_t slot, void *dst, char *err, int err_len) {
    SK_TRY(e->ex->save_slot(slot, dst))
}

int sk_exec_load_slot(sk_exec *e, int32_t slot, const void *src, char *err, int err_len) {
    SK_TRY(e->ex->load_slot(slot, src))
}

size_t sk_exec_page_bytes(const sk_exec *e) { return e->ex->page_bytes(); }

int sk_exec_reserve_host_pages(sk_exec *e, int64_t n, char *err, int err_len) { SK_TRY(e->ex->reserve_host_pages(n)) }

int sk_exec_copy_pages(sk_exec *e, int32_t to_host, const int32_t *dev_pages, const int32_t *host_pages, int64_t n,
                       uint64_t *ticket, char *err, int err_len) {
    SK_TRY(*ticket = e->ex->copy_pages(to_host != 0, dev_pages, host_pages, n))
}

int sk_exec_copy_done(sk_exec *e, uint64_t ticket) {
    try {
        return e->ex->copy_done(ticket) ? 1 : 0;
    } catch (...) {
        return -1;
    }
}

int sk_exec_copy_order(sk_exec *e, uint64_t ticket, char *err, int err_len) { SK_TRY(e->ex->copy_order(ticket)) }

int sk_exec_copy_wait(sk_exec *e, uint64_t ticket, char *err, int err_len) { SK_TRY(e->ex->copy_wait(ticket)) }

const char *sk_exec_describe(sk_exec *e) { return e->ex->describe(); }

}  // extern "C"
