// SPDX-License-Identifier: Apache-2.0
// ink2_exec.cpp - the Inkling 2 (looped) forward graph on the executor ABI (fullm_exec.h).
//
// The model is crates/fullm/src/models/inkling2/cpu.rs (the Rust reference; tools/inkling2_ref.py
// states it in torch), run here a step at a time over many sequences: the dense matrices through
// Backend::gemm, the model's own pieces (attention residuals, the short convolutions, the
// attention with its position bias and sink, the message gates, the routed experts) in this file,
// on host memory. A backend with memory of its own is refused until those pieces are backend ops.
//
// PLACEMENT. The dense matrices are copied into backend memory in the format given (bf16). The
// embedding table and the routed experts stay where the caller mapped them: the experts are read
// in place in the release's formats (NVFP4: two E2M1 codes a byte with an e4m3 scale per 16 values
// and a factor per expert; MXFP8: e4m3 with a power of two per 32), a row decoded at a time
// into L1 and dotted with every token routed to its expert. Nothing of the 20 GB is widened.
//
// STATE. Per layer visit (layers 6-17 are visited three times a token: 48 visits), a paged pool
// of keys and one of values ([page][kv head][position in page][head width], f32: a kv head's keys
// of a page are contiguous rows), and per sequence slot the last K - 1 inputs of the visit's four
// short convolutions.
//
// A row's result depends on its own sequence only (every dot is the CPU kernels' row routine), so
// a sequence gives the same logits alone, batched, or with its prompt in chunks.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "exec_common.h"
#include "fullm_kernels.h"
#include "threadpool.h"
#if defined(__APPLE__)
#include "metal/ink2_metal.h"
#define INK2_METAL 1
#else
#define INK2_METAL 0
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace fullm {
namespace {

// One layer's experts, a stack of n_experts matrices [rows][cols] read in place.
struct XStack {
    int fmt = -1;  // SK_XFMT_*
    const uint8_t *w = nullptr, *s = nullptr;
    const float *f = nullptr;  // NVFP4: amax / outer_scale_denominator per expert
    int64_t rows = 0, cols = 0;
};

struct XRow {
    const XStack *x;
    int64_t e;
};

// Row n of expert e as cols floats.
void decode_row(void *ctx, int64_t n, float *out) {
    const XRow &r = *static_cast<const XRow *>(ctx);
    const XStack &x = *r.x;
    const int64_t K = x.cols, row = r.e * x.rows + n;
    const float *e4 = sk_fp8_e4m3_table();
    if (x.fmt == SK_XFMT_NVFP4) {
        const uint8_t *codes = x.w + row * (K / 2), *sc = x.s + row * (K / 16);
        const float f = x.f[r.e];
#if defined(__aarch64__)
        // E2M1 times two, by code: the block's scale carries the half
        static const int8_t lut2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
        const int8x16_t lut = vld1q_s8(lut2);
        const uint8x8_t m4 = vdup_n_u8(0xF);
        for (int64_t b = 0; b < K / 16; ++b) {
            const uint8x8_t by = vld1_u8(codes + 8 * b);
            const uint8x8_t lo = vand_u8(by, m4), hi = vshr_n_u8(by, 4);
            const int8x16_t v = vqtbl1q_s8(lut, vcombine_u8(vzip1_u8(lo, hi), vzip2_u8(lo, hi)));
            const float32x4_t s = vdupq_n_f32(e4[sc[b]] * f * 0.5f);
            const int16x8_t a = vmovl_s8(vget_low_s8(v)), c = vmovl_high_s8(v);
            float *o = out + 16 * b;
            vst1q_f32(o, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a))), s));
            vst1q_f32(o + 4, vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(a)), s));
            vst1q_f32(o + 8, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(c))), s));
            vst1q_f32(o + 12, vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(c)), s));
        }
#else
        static const float e2m1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                       -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
        for (int64_t b = 0; b < K / 16; ++b) {
            const float s = e4[sc[b]] * f;
            for (int i = 0; i < 8; ++i) {
                const uint8_t by = codes[8 * b + i];
                out[16 * b + 2 * i] = e2m1[by & 0xF] * s;
                out[16 * b + 2 * i + 1] = e2m1[by >> 4] * s;
            }
        }
#endif
    } else if (x.fmt == SK_XFMT_MXFP8) {
        const uint8_t *w = x.w + row * K, *sc = x.s + row * (K / 32);
        for (int64_t b = 0; b < K / 32; ++b) {
            const float s = std::ldexp(1.0f, int(sc[b]) - 127);
            for (int i = 0; i < 32; ++i) out[32 * b + i] = e4[w[32 * b + i]] * s;
        }
    } else {  // bf16
        const uint16_t *w = reinterpret_cast<const uint16_t *>(x.w) + row * K;
        for (int64_t i = 0; i < K; ++i) {
            const uint32_t u = uint32_t(w[i]) << 16;
            std::memcpy(out + i, &u, 4);
        }
    }
}

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
#define INK2_SDOT 1
// A row of activations as two int8 vectors, x ~ s1 * hi + s2 * lo (s1 = amax / 127, s2 = s1 / 254:
// the pair resolves amax * 1.6e-5), for the integer dots below.
void quant_hilo(const float *x, int64_t K, int8_t *hi, int8_t *lo, float *s1, float *s2) {
    float amax = 0;
    for (int64_t i = 0; i < K; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float a = amax > 0 ? amax / 127.0f : 1.0f, b = a / 254.0f, ia = 1.0f / a, ib = 1.0f / b;
    for (int64_t i = 0; i < K; ++i) {
        const float h = std::nearbyint(x[i] * ia);
        hi[i] = int8_t(h);
        lo[i] = int8_t(std::nearbyint((x[i] - h * a) * ib));
    }
    *s1 = a, *s2 = b;
}

// NVFP4 rows [n0, n1) of expert e against M quantised rows of x, without decoding to floats: a
// block's 16 codes become 16 small integers (E2M1 times two) by one table lookup, dotted with x's
// int8 pairs by SDOT, and the block's scale applied to the four partial sums. y[m][n].
void nvfp4_rows(const XStack &x, int64_t e, int64_t n0, int64_t n1, const int8_t *hi, const int8_t *lo, const float *s1,
                const float *s2, int64_t M, float *y, int64_t ldy) {
    static const int8_t lut2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const int8x16_t lut = vld1q_s8(lut2);
    const uint8x8_t m4 = vdup_n_u8(0xF);
    const float *e4 = sk_fp8_e4m3_table();
    const int64_t K = x.cols, B = K / 16;
    const float f = x.f[e] * 0.5f;
    const int32x4_t zero = vdupq_n_s32(0);
    for (int64_t n = n0; n < n1; ++n) {
        const int64_t row = e * x.rows + n;
        const uint8_t *codes = x.w + row * (K / 2), *sc = x.s + row * B;
        if (M == 1) {
            float32x4_t ah = vdupq_n_f32(0), al = vdupq_n_f32(0);
            for (int64_t b = 0; b < B; ++b) {
                const uint8x8_t by = vld1_u8(codes + 8 * b);
                const uint8x8_t l4 = vand_u8(by, m4), h4 = vshr_n_u8(by, 4);
                const int8x16_t v = vqtbl1q_s8(lut, vcombine_u8(vzip1_u8(l4, h4), vzip2_u8(l4, h4)));
                const float32x4_t sb = vdupq_n_f32(e4[sc[b]]);
                ah = vfmaq_f32(ah, vcvtq_f32_s32(vdotq_s32(zero, v, vld1q_s8(hi + 16 * b))), sb);
                al = vfmaq_f32(al, vcvtq_f32_s32(vdotq_s32(zero, v, vld1q_s8(lo + 16 * b))), sb);
            }
            y[n] = f * (s1[0] * vaddvq_f32(ah) + s2[0] * vaddvq_f32(al));
            continue;
        }
        // several tokens: the row's integers once, then each token's dots
        thread_local std::vector<int8_t> vbuf;
        thread_local std::vector<float> sbuf;
        vbuf.resize(size_t(K)), sbuf.resize(size_t(B));
        for (int64_t b = 0; b < B; ++b) {
            const uint8x8_t by = vld1_u8(codes + 8 * b);
            const uint8x8_t l4 = vand_u8(by, m4), h4 = vshr_n_u8(by, 4);
            vst1q_s8(vbuf.data() + 16 * b, vqtbl1q_s8(lut, vcombine_u8(vzip1_u8(l4, h4), vzip2_u8(l4, h4))));
            sbuf[size_t(b)] = e4[sc[b]];
        }
        for (int64_t m = 0; m < M; ++m) {
            const int8_t *xh = hi + m * K, *xl = lo + m * K;
            float32x4_t ah = vdupq_n_f32(0), al = vdupq_n_f32(0);
            for (int64_t b = 0; b < B; ++b) {
                const int8x16_t v = vld1q_s8(vbuf.data() + 16 * b);
                const float32x4_t sb = vdupq_n_f32(sbuf[size_t(b)]);
                ah = vfmaq_f32(ah, vcvtq_f32_s32(vdotq_s32(zero, v, vld1q_s8(xh + 16 * b))), sb);
                al = vfmaq_f32(al, vcvtq_f32_s32(vdotq_s32(zero, v, vld1q_s8(xl + 16 * b))), sb);
            }
            y[m * ldy + n] = f * (s1[m] * vaddvq_f32(ah) + s2[m] * vaddvq_f32(al));
        }
    }
}
#else
#define INK2_SDOT 0
#endif

struct ILayer {
    bool sliding = false, loop = false;
    int64_t hq = 0, hk = 0, d = 0, extent = 0;
    Weight q, k, v, o, og, sink, rel, router, k_wp, v_wp;
    Weight c_down[4], c_forget[4], c_gain[4];  // message gates: query, key, value, mlp
    const float *attn_query = nullptr, *mlp_query = nullptr, *q_scale = nullptr, *rel_scale = nullptr;
    const float *rel_proj = nullptr, *post_norm = nullptr, *router_bias = nullptr;
    const float *in_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *v_norm = nullptr;
    // convolution taps as [tap][channel]: static ones oldest input first, the full layers' key /
    // value ones by lag (0 = the current row)
    std::vector<float> attn_conv, mlp_conv, k_conv, v_conv, k_static, v_static;
    XStack x13, x2;
};

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float softcap(float x, float cap) { return std::tanh(x / cap) * cap; }

void unorm(float *x, int64_t n, float eps) {
    double ss = 0;
    for (int64_t i = 0; i < n; ++i) ss += double(x[i]) * x[i];
    const float s = float(1.0 / std::sqrt(ss / double(n) + eps));
    for (int64_t i = 0; i < n; ++i) x[i] *= s;
}

void l2_clip(float *x, int64_t n, float thr) {
    double ss = 0;
    for (int64_t i = 0; i < n; ++i) ss += double(x[i]) * x[i];
    const float s = std::min(1.0f, thr / (float(std::sqrt(ss)) + 1e-6f));
    for (int64_t i = 0; i < n; ++i) x[i] *= s;
}

void softmax(float *x, int64_t n) {
    float m = x[0];
    for (int64_t i = 1; i < n; ++i) m = std::max(m, x[i]);
    float s = 0;
    for (int64_t i = 0; i < n; ++i) x[i] = std::exp(x[i] - m), s += x[i];
    const float inv = 1.0f / s;
    for (int64_t i = 0; i < n; ++i) x[i] *= inv;
}

using Rows = std::vector<float>;  // [N][width]

class Ink2Exec final : public ExecIface {
public:
    Ink2Exec(const sk_ink2_dims &d, const sk_exec_opts &o) : d_(d), o_(o) {
        if (!d.layer_sliding) throw std::runtime_error("layer_sliding is null");
        sliding_.assign(d.layer_sliding, d.layer_sliding + d.n_layers);
        d_.layer_sliding = nullptr;
        if (o.n_slots <= 0 || o.n_pages <= 0 || o.page <= 0 || o.max_pages_per_seq <= 0 || o.max_tokens <= 0 ||
            o.max_rows <= 0)
            throw std::runtime_error("executor options must all be positive");
        if (d.loop_start <= 0 || d.loop_start > d.loop_end || d.loop_end > d.n_layers || d.block <= 0 ||
            d.loop_start % d.block || d.loop_end % d.block || d.passes < 1 || d.conv_k < 2)
            throw std::runtime_error("inkling2: loop, block or kernel out of range");
        if (d.full_hq % d.full_hk || d.swa_hq % d.swa_hk) throw std::runtime_error("inkling2: heads per kv head");
        b_ = make_backend(backend_name(o.backend_name, o.backend), o.device, o.cpu_threads);
        if (!b_->on_host())
            throw std::runtime_error(std::string("inkling2_loop runs on host backends only so far (not ") + b_->name() + ")");
        b_->set_exact_activations(true);
        pool_ = std::make_unique<ThreadPool>(o.cpu_threads);
        n_loop_ = d.loop_end - d.loop_start;
        visits_ = d.n_layers + (d.passes - 1) * n_loop_;
#if INK2_METAL
        // The steps on the Mac's GPU (metal/ink2_metal.h). FULLM_INK2_METAL=0: the CPU path below;
        // FULLM_INK2_METAL_F32=1: f32 where the GPU holds activations as half (the oracle's checks).
        const char *me = std::getenv("FULLM_INK2_METAL");
        if (!(me && *me == '0')) {
            std::string why;
            metal_ = ink2_metal_create(gpu_f32_, o.kv_dtype == SK_DT_F32, why);
            kv_half_ = metal_ && o.kv_dtype != SK_DT_F32;
            if (!metal_) std::fprintf(stderr, "fullm: inkling2 stays on the CPU (%s)\n", why.c_str());
        }
#endif
    }

    ~Ink2Exec() {
        for (void *p : owned_) b_->release(p);
        sample_scratch_.release(b_.get());
#if INK2_METAL
        if (metal_) ink2_metal_destroy(metal_);  // with it the memory it handed out
#endif
    }

    // ------------------------------------------------------------------ loading
    void add_matrix(const std::string &name, int32_t fmt, int64_t rows, int64_t cols, const void *data,
                    const float *) override {
        if (finalized_) throw std::runtime_error("add after finalize");
        if (rows <= 0 || cols <= 0 || !data) throw std::runtime_error(name + ": empty matrix");
        if (fmt != SK_FMT_BF16 && fmt != SK_FMT_F32) throw std::runtime_error(name + ": inkling2 matrices are bf16 or f32");
        Weight w;
        w.fmt = static_cast<WFmt>(fmt);
        w.rows = rows;
        w.cols = cols;
        if (name == "embed") {  // stays in host memory, not copied
            w.data = data;
            embed_ = w;
            return;
        }
        const size_t bytes = size_t(rows) * cols * (w.fmt == WFmt::F32 ? 4 : 2);
        void *dd = alloc(bytes);
        b_->upload(dd, data, bytes);
        w.data = dd;
        if (mats_.count(name)) throw std::runtime_error(name + ": added twice");
        mats_[name] = w;
        weight_bytes_ += bytes;
    }

    void add_vector(const std::string &name, const float *data, int64_t n) override {
        if (finalized_) throw std::runtime_error("add after finalize");
        float *dd = static_cast<float *>(alloc(size_t(n) * 4));
        b_->upload(dd, data, size_t(n) * 4);
        if (vecs_.count(name)) throw std::runtime_error(name + ": added twice");
        vecs_[name] = {dd, n};
        weight_bytes_ += size_t(n) * 4;
    }

    void set_ink2_experts(int32_t layer, int32_t fmt, const uint8_t *w13, const uint8_t *s13, const float *f13,
                          const uint8_t *w2, const uint8_t *s2, const float *f2) override {
        if (finalized_) throw std::runtime_error("experts after finalize");
        if (layer < 0 || layer >= d_.n_layers) throw std::runtime_error("set_ink2_experts: layer out of range");
        const int64_t H = d_.hidden, I = d_.moe_inter;
        const bool nv = fmt == SK_XFMT_NVFP4, mx = fmt == SK_XFMT_MXFP8;
        if (!nv && !mx && fmt != SK_XFMT_BF16) throw std::runtime_error("set_ink2_experts: unknown format");
        if (!w13 || !w2 || ((nv || mx) && (!s13 || !s2)) || (nv && (!f13 || !f2)))
            throw std::runtime_error("set_ink2_experts: missing data for the format");
        if ((nv && (H % 16 || I % 16)) || (mx && (H % 32 || I % 32)))
            throw std::runtime_error("set_ink2_experts: widths must be whole scale blocks");
        if (xs_.empty()) xs_.resize(size_t(d_.n_layers));
        xs_[size_t(layer)] = {XStack{fmt, w13, s13, f13, 2 * I, H}, XStack{fmt, w2, s2, f2, H, I}};
#if INK2_METAL
        if (metal_ && (nv || mx) &&
            !ink2_metal_add_layer(metal_, layer, fmt, d_.n_experts, int(H), int(I), w13, s13, f13, w2, s2, f2))
            std::fprintf(stderr, "fullm: layer %d's experts could not be mapped for the GPU\n", layer);
#endif
        const size_t per = nv ? 1 : mx ? 2 : 4;  // half-bytes a value
        expert_bytes_ += size_t(d_.n_experts) * size_t(3 * I * H) * per / 2;
    }

    void finalize() override {
        std::vector<std::string> missing;
        const int64_t H = d_.hidden, K = d_.conv_k, R = d_.gate_rank;
        auto M = [&](const std::string &n, int64_t r, int64_t c) -> Weight {
            auto it = mats_.find(n);
            if (it == mats_.end()) {
                missing.push_back(n);
                return Weight{};
            }
            if (it->second.rows != r || it->second.cols != c)
                missing.push_back(n + " (shape " + std::to_string(it->second.rows) + "x" + std::to_string(it->second.cols) +
                                  ", want " + std::to_string(r) + "x" + std::to_string(c) + ")");
            return it->second;
        };
        auto V = [&](const std::string &n, int64_t len) -> const float * {
            auto it = vecs_.find(n);
            if (it == vecs_.end()) {
                missing.push_back(n);
                return nullptr;
            }
            if (it->second.second != len)
                missing.push_back(n + " (length " + std::to_string(it->second.second) + ", want " + std::to_string(len) + ")");
            return it->second.first;
        };
        // a [channel][tap] convolution weight as [tap][channel]
        auto taps = [&](const std::string &n, int64_t C) {
            std::vector<float> t(size_t(K * C));
            if (const float *w = V(n, C * K))
                for (int64_t c = 0; c < C; ++c)
                    for (int64_t j = 0; j < K; ++j) t[size_t(j * C + c)] = w[c * K + j];
            return t;
        };
        if (!embed_.data || embed_.rows != d_.vocab || embed_.cols != H) missing.push_back("embed");
        lm_head_ = M("lm_head", d_.out_vocab, H);
        embed_norm_ = V("embed_norm", H);
        final_norm_ = V("final_norm", H);
        out_query_ = V("out_query", H);
        msg_queries_ = V("msg_queries", n_loop_ * 4 * H);
        layers_.resize(size_t(d_.n_layers));
        for (int l = 0; l < d_.n_layers; ++l) {
            ILayer &L = layers_[size_t(l)];
            const std::string p = "L" + std::to_string(l) + ".";
            L.sliding = sliding_[size_t(l)] != 0;
            L.loop = l >= d_.loop_start && l < d_.loop_end;
            L.hq = L.sliding ? d_.swa_hq : d_.full_hq;
            L.hk = L.sliding ? d_.swa_hk : d_.full_hk;
            L.d = L.sliding ? d_.swa_d : d_.full_d;
            L.extent = L.sliding ? d_.window : d_.rel_extent;
            const int64_t Cq = L.hq * L.d, Ck = L.hk * L.d;
            L.q = M(p + "q", Cq, H);
            L.k = M(p + "k", Ck, H);
            L.v = M(p + "v", Ck, H);
            L.o = M(p + "o", H, Cq);
            L.og = M(p + "og", Cq, H);
            L.sink = M(p + "sink", L.hq, H);
            L.rel = M(p + "rel", L.hq * d_.d_rel, H);
            L.router = M(p + "router", d_.n_experts, H);
            if (l > 0) L.attn_query = V(p + "attn_query", L.sliding ? H : 3 * H);
            L.mlp_query = V(p + "mlp_query", H);
            L.q_scale = V(p + "q_scale", Cq);
            L.rel_scale = V(p + "rel_scale", L.hq * d_.d_rel);
            L.rel_proj = V(p + "rel_proj", d_.d_rel * L.extent);
            L.post_norm = V(p + "post_norm", H);
            L.router_bias = V(p + "router_bias", d_.n_experts);
            L.attn_conv = taps(p + "attn_conv", H);
            L.mlp_conv = taps(p + "mlp_conv", H);
            if (L.sliding) {
                L.in_norm = V(p + "in_norm", H);
                L.k_conv = taps(p + "k_conv", Ck);
                L.v_conv = taps(p + "v_conv", Ck);
            } else {
                L.q_norm = V(p + "q_norm", H);
                L.k_norm = V(p + "k_norm", H);
                L.v_norm = V(p + "v_norm", H);
                L.k_wp = M(p + "k_wp", L.hk * K, H);
                L.v_wp = M(p + "v_wp", L.hk * K, H);
                for (auto kv : {std::make_pair("k_static", &L.k_static), std::make_pair("v_static", &L.v_static)})
                    if (const float *w = V(p + kv.first, K * Ck)) kv.second->assign(w, w + K * Ck);
            }
            if (L.loop) {
                static const char *const g[4] = {"cq", "ck", "cv", "cm"};
                for (int i = 0; i < 4; ++i) {
                    L.c_down[i] = M(p + g[i] + "_down", R, 2 * H);
                    L.c_forget[i] = M(p + g[i] + "_forget", H, R);
                    L.c_gain[i] = M(p + g[i] + "_gain", H, R);
                }
            }
            if (xs_.size() <= size_t(l) || xs_[size_t(l)].first.fmt < 0) missing.push_back(p + "experts");
            else L.x13 = xs_[size_t(l)].first, L.x2 = xs_[size_t(l)].second;
        }
        if (!missing.empty()) {
            std::string m = "inkling2: missing or misshapen tensors:";
            for (size_t i = 0; i < missing.size() && i < 12; ++i) m += " " + missing[i];
            if (missing.size() > 12) m += " ... (" + std::to_string(missing.size()) + " in all)";
            throw std::runtime_error(m);
        }
        // pools: keys and values per visit, convolution histories per slot
        vis_off_.assign(size_t(visits_) + 1, 0);
        for (int v = 0; v < visits_; ++v) {
            const ILayer &L = layers_[size_t(layer_of(v))];
            vis_off_[size_t(v) + 1] = vis_off_[size_t(v)] + (2 * L.hk * L.d + 2 * H) * (K - 1);
            const size_t n = size_t(o_.n_pages) * o_.page * L.hk * L.d;
            // (half on the GPU unless the cache was asked for in f32: the steps then run there only)
            const size_t eb = kv_half_ ? 2 : 4;
            kpool_.push_back(static_cast<float *>(alloc(n * eb)));
            vpool_.push_back(static_cast<float *>(alloc(n * eb)));
            std::memset(kpool_.back(), 0, n * eb), std::memset(vpool_.back(), 0, n * eb);
            kv_bytes_ += 2 * n * eb;
        }
        conv_n_ = size_t(o_.n_slots) * vis_off_.back();
        conv_ = static_cast<float *>(alloc(conv_n_ * 4));
        std::memset(conv_, 0, conv_n_ * 4);
        // what a rejected draft needs (drop_tail): per history, the last kUndo + K - 1 rows of [history, step]
        undo_stride_ = size_t(vis_off_.back()) / size_t(K - 1) * size_t(K - 1 + kUndo);
        undo_ = static_cast<float *>(alloc(size_t(o_.n_slots) * undo_stride_ * 4));
        // the histories after each page's last position (the prefix cache): committed as pages fill
        if (tails_on_) tails_ = static_cast<float *>(alloc(size_t(o_.n_pages) * size_t(vis_off_.back()) * 4));
        last_n_.assign(size_t(o_.n_slots), 0);
        last_kept_.assign(size_t(o_.n_slots), 0), last_first_.assign(size_t(o_.n_slots), 0), last_step_.assign(size_t(o_.n_slots), 0);
        setup_gpu();
        if ((w8_dense_ || w8_head_) && !gpu_)
            throw std::runtime_error("inkling2: int8 matrices are the GPU path's; this model stays on the CPU (--trunk bf16, no --head)");
        if (kv_half_ && !gpu_)
            throw std::runtime_error("inkling2: the 16-bit cache is the GPU path's; this model stays on the CPU (--kv-dtype f32)");
        logits_.resize(size_t(o_.max_rows) * d_.out_vocab);
        ids_.resize(size_t(o_.max_rows) + 1);
        hids_.resize(size_t(o_.max_rows) + 1);
        sample_.resize(size_t(o_.max_rows));
        finalized_ = true;
    }

    size_t pool_bytes() const override { return 0; }
    size_t slot_bytes() const override { return finalized_ ? size_t(vis_off_.back()) * 4 : 0; }
    void save_slot(int32_t slot, void *dst) override {
        check_slot(slot);
        std::memcpy(dst, conv_ + size_t(slot) * vis_off_.back(), slot_bytes());
    }
    void load_slot(int32_t slot, const void *src) override {
        check_slot(slot);
        std::memcpy(conv_ + size_t(slot) * vis_off_.back(), src, slot_bytes());
        last_n_[size_t(slot)] = 0;
    }

    void set_ink2_mtp(const sk_ink2_mtp &c) override {
        if (finalized_) throw std::runtime_error("draft head after finalize");
        if (c.n_layers < 1 || c.n_layers > 8 || c.hk < 1 || c.hq % c.hk || c.d % 4 || c.d % 2 || c.inter % 4 || c.rank % 4 ||
            c.block < 2 || c.block > 16 || c.window < 1 || c.theta <= 1)
            throw std::runtime_error("draft head: a dimension out of range");
        if (c.taps[3] != visits_ || !(0 <= c.taps[0] && c.taps[0] < c.taps[1] && c.taps[1] < c.taps[2] && c.taps[2] < visits_))
            throw std::runtime_error("draft head: its features are not layer visits of this model in order, then the final norm");
        mtp_ = c, mtp_set_ = true;
    }
    bool ink2_has_draft() const override { return mtp_on_; }
    void ink2_int8(bool dense, bool head) override {
        if (finalized_) throw std::runtime_error("ink2_int8 after finalize");
        w8_dense_ = dense, w8_head_ = head;
    }

    // Drafts after a step: see sk_exec_ink2_draft.
    void ink2_draft(int32_t n, const int32_t *slots, const int32_t *next, const int32_t *pos, const int32_t *page_table, int32_t k,
                    int32_t *out, float *conf) override {
        if (!mtp_on_) throw std::runtime_error("no draft head");
#if INK2_METAL
        const int64_t H = d_.hidden;
        std::vector<int32_t> rows(static_cast<size_t>(n));
        Rows emb(size_t(n) * size_t(H));
        for (int i = 0; i < n; ++i) {
            check_slot(slots[i]);
            const size_t s = size_t(slots[i]);
            if (last_step_[s] != step_no_ || last_kept_[s] <= 0) throw std::runtime_error("draft: the sequence was not in the last step");
            if (next[i] < 0 || next[i] >= d_.vocab || pos[i] <= 0) throw std::runtime_error("draft: token or position out of range");
            rows[size_t(i)] = last_first_[s] + last_kept_[s] - 1;
            embed_row(next[i], emb.data() + int64_t(i) * H);
        }
        emb = rms(emb, embed_norm_);
        std::string why;
        if (!ink2_metal_draft(metal_, n, slots, pos, rows.data(), next, emb.data(), page_table, k, out, conf, why))
            throw std::runtime_error("inkling2 draft head on Metal: " + why);
#else
        (void)n, (void)slots, (void)next, (void)pos, (void)page_table, (void)k, (void)out, (void)conf;
#endif
    }

    void page_state_enable() override {
        if (finalized_) throw std::runtime_error("page state after finalize");
        tails_on_ = true;
    }

    // The slot continues after `page`: its histories are the ones kept when the page was filled.
    void page_state_load(int32_t slot, int32_t page) override {
        check_slot(slot);
        if (!tails_ || page < 0 || page >= o_.n_pages) throw std::runtime_error("page_state_load: no state for the page");
        std::memcpy(conv_ + size_t(slot) * vis_off_.back(), tails_ + size_t(page) * vis_off_.back(), slot_bytes());
        last_n_[size_t(slot)] = 0;
    }

    // The last `dropped` rows of the slot's segment in the last step were rejected drafts: every
    // history becomes the K - 1 rows before them (the cache positions are written again later).
    void drop_tail(int32_t slot, int32_t dropped) override {
        check_slot(slot);
        const int64_t n = last_n_[size_t(slot)], K1 = d_.conv_k - 1, H = d_.hidden;
        if (dropped <= 0) return;
        if (dropped >= n || dropped > kUndo)
            throw std::runtime_error("drop_tail: " + std::to_string(dropped) + " rows of a segment of " + std::to_string(n) +
                                     " (at most " + std::to_string(kUndo) + ", and not all)");
        const int64_t m = std::min<int64_t>(n, kUndo) + K1, from = (n - dropped) - (n + K1 - m);  // the first kept-history row in the saved rows
        for (int v = 0; v < visits_; ++v) {
            const ILayer &L = layers_[size_t(layer_of(v))];
            const int64_t Ck = L.hk * L.d, width[4] = {Ck, Ck, H, H};
            for (int w = 0; w < 4; ++w)
                std::memcpy(conv_state(slot, v, w, Ck), undo_state(slot, v, w, Ck) + from * width[w], size_t(K1 * width[w]) * 4);
        }
        last_kept_[size_t(slot)] = int32_t(n - dropped);
        last_n_[size_t(slot)] = 0;
    }

    const char *describe() override {
        std::ostringstream os;
        os << "inkling2_loop on " << b_->name() << " (" << sk_cpu_isa() << "): " << d_.n_layers << " layers, layers "
           << d_.loop_start << ".." << d_.loop_end << " x" << d_.passes << " (" << visits_ << " visits a token), dense weights "
           << double(weight_bytes_) / 1e9 << " GB, " << d_.n_experts << " experts top-" << d_.topk << " read in place ("
           << double(expert_bytes_) / 1e9 << " GB), KV cache " << double(kv_bytes_) / 1e9 << (kv_half_ ? " GB f16 (" : " GB f32 (")
           << int64_t(o_.n_pages) * o_.page << " positions), " << pool_->size() << " threads";
        if (w8_dense_ || w8_head_)
            os << ", int8 with a scale per row: " << (w8_dense_ ? "the layers' projections" : "") << (w8_dense_ && w8_head_ ? ", " : "")
               << (w8_head_ ? "the lm_head" : "");
        if (mtp_on_) os << ", draft head (" << mtp_.n_layers << " layers, blocks of " << mtp_.block << ")";
        if (tails_)
            os << ", prefix cache: the histories kept by page (" << double(vis_off_.back()) * 4 / 1e6 << " MB a page, "
               << double(o_.n_pages) * double(vis_off_.back()) * 4 / 1e9 << " GB when every page is filled)";
#if INK2_METAL
        if (gpu_) os << ", steps on " << ink2_metal_device(metal_) << " (Metal" << (gpu_f32_ ? ", f32 activations" : "") << ")";
#endif
        desc_ = os.str();
        return desc_.c_str();
    }

    // ------------------------------------------------------------------ the step
    void step(const sk_step_plan &pl, float *logits_out, sk_step_stats *st, int32_t *tokens_out,
              const sk_sample_row *sample, const sk_sample_io *io) override {
        if (!finalized_) throw std::runtime_error("step before finalize");
        const auto t0 = Clock::now();
        validate(pl);
        pl_ = &pl;
        small_ = pl.n_tokens <= 16;
        ++step_no_;
        for (int i = 0; i < pl.n_segs; ++i) {
            const size_t s = size_t(pl.segs[3 * i]);
            last_n_[s] = last_kept_[s] = pl.segs[3 * i + 2];
            last_first_[s] = pl.segs[3 * i + 1], last_step_[s] = step_no_;
        }
        const int64_t N = pl.n_tokens, R = pl.n_rows, H = d_.hidden;
        for (int i = 0; i < pl.n_reset; ++i) {
            float *c = conv_ + size_t(pl.reset_slots[i]) * vis_off_.back();
            std::fill(c, c + vis_off_.back(), 0.0f);
        }
        // the embeddings, normalised
        Rows h(size_t(N * H));
        rows_for(N, [&](int64_t r) { embed_row(pl.tokens[r], h.data() + r * H); }, H);
        h = rms(h, embed_norm_);
#if INK2_METAL
        if (gpu_) {
            // the whole step on the GPU (logits softcapped there)
            std::string why;
            if (!ink2_metal_step(metal_, pl, h.data(), logits_.data(), why)) throw std::runtime_error("inkling2 on Metal: " + why);
            finish(pl, logits_out, st, tokens_out, sample, io, t0);
            return;
        }
#endif
        std::vector<Rows> blocks;
        run_layers(0, d_.loop_start, 0, h, blocks, nullptr, nullptr);
        const Rows pre_h = h;
        const std::vector<Rows> pre_blocks = blocks;
        run_layers(d_.loop_start, d_.loop_end, 0, h, blocks, nullptr, nullptr);
        for (int p = 1; p < d_.passes; ++p) {
            // the previous pass's block states past the one it started with, and its output
            const std::vector<Rows> prev_blocks(blocks.begin() + std::min(pre_blocks.size() + 1, blocks.size()), blocks.end());
            const Rows prev_h = std::move(h);
            h = pre_h, blocks = pre_blocks;
            run_layers(d_.loop_start, d_.loop_end, p, h, blocks, &prev_blocks, &prev_h);
        }
        run_layers(d_.loop_end, d_.n_layers, 0, h, blocks, nullptr, nullptr);
        Rows fin = rms(attn_res(out_query_, 1, blocks, h)[0], final_norm_);
        // the wanted rows' logits, softcapped
        Rows sel(size_t(R * H));
        for (int64_t i = 0; i < R; ++i) std::memcpy(sel.data() + i * H, fin.data() + int64_t(pl.rows[i]) * H, size_t(H) * 4);
        const int64_t V = d_.out_vocab;
        b_->gemm(lm_head_, sel.data(), R, logits_.data(), V);
        const float cap = d_.logit_softcap;
        const int64_t piece = 4096, pieces = (R * V + piece - 1) / piece;
        rows_for(pieces, [&](int64_t c) {
            float *x = logits_.data() + c * piece;
            const int64_t n = std::min(piece, R * V - c * piece);
            for (int64_t i = 0; i < n; ++i) x[i] = softcap(x[i], cap);
        }, 16 * piece);
        finish(pl, logits_out, st, tokens_out, sample, io, t0);
    }

private:
    // the logits to the caller or through the sampler, and the step's accounts
    void finish(const sk_step_plan &pl, float *logits_out, sk_step_stats *st, int32_t *tokens_out, const sk_sample_row *sample,
                const sk_sample_io *io, Clock::time_point t0) {
        const int64_t R = pl.n_rows, V = d_.out_vocab;
        if (tokens_out) {
            step_tokens(b_.get(), logits_.data(), R, V, sample, io, ids_.data(), sample_.data(), sample_scratch_, hids_.data(),
                        tokens_out);
        } else {
            std::memcpy(logits_out, logits_.data(), size_t(R * V) * 4);
        }
        if (st) {
            *st = sk_step_stats{};
            st->ms_total = ms_since(t0);
            st->ms_experts = ms_experts_;
        }
#if INK2_METAL
        if (prof_ && gpu_) {
            double g, w;
            int64_t n;
            ink2_metal_last(metal_, &g, &w, &n);
            std::fprintf(stderr, "ink2 step: %d rows, %.1f ms: GPU %.1f ms in %lld dispatches (encode + wait %.1f)\n", pl.n_tokens,
                         ms_since(t0), g, static_cast<long long>(n), w);
        }
#endif
        if (prof_ && !gpu_) {
            std::fprintf(stderr, "ink2 step: %d rows, %.1f ms: experts nvfp4 %.1f, mxfp8/bf16 %.1f, attention %.1f, rest %.1f (GEMMs in all: %.1f)\n",
                         pl.n_tokens, ms_since(t0), ms_nv_, ms_mx_, ms_attn_, ms_since(t0) - ms_nv_ - ms_mx_ - ms_attn_, ms_gemm_);
            std::fprintf(stderr, "   attention: GEMMs + scale/clip %.0f, k/v convolutions %.0f, k/v into the pools %.0f, scoring %.0f, gate + o %.0f\n",
                         ms_ph_[0], ms_ph_[1], ms_ph_[2], ms_ph_[4], ms_ph_[5]);
        }
        for (double &x : ms_ph_) x = 0;
        ms_experts_ = ms_nv_ = ms_mx_ = ms_attn_ = ms_gemm_ = 0;
        pl_ = nullptr;
    }

    void embed_row(int64_t t, float *o) const {
        const int64_t H = d_.hidden;
        if (embed_.fmt == WFmt::F32) {
            std::memcpy(o, static_cast<const float *>(embed_.data) + t * H, size_t(H) * 4);
        } else {
            const uint16_t *w = static_cast<const uint16_t *>(embed_.data) + t * H;
            for (int64_t i = 0; i < H; ++i) {
                const uint32_t u = uint32_t(w[i]) << 16;
                std::memcpy(o + i, &u, 4);
            }
        }
    }

    // host memory: the GPU's (it sees it in place) when there is one
    void *alloc(size_t bytes) {
#if INK2_METAL
        if (metal_) {
            if (void *p = ink2_metal_alloc(metal_, bytes)) return p;
            throw std::runtime_error("inkling2: no memory for the GPU");
        }
#endif
        void *p = b_->alloc(bytes);
        owned_.push_back(p);
        return p;
    }

#if INK2_METAL
    // The draft head's tensors for the GPU, its context pools, and the recurrent head's output
    // folded through the model's head: lm_head (h + hidden_out z) = lm_head h + fold z + fold_b.
    bool setup_mtp(Ink2MetalMtp &T, std::string &why) {
        const sk_ink2_mtp &c = mtp_;
        const int64_t H = d_.hidden, Cq = int64_t(c.hq) * c.d, Ck = int64_t(c.hk) * c.d, R = c.rank, V = d_.out_vocab;
        bool ok = true;
        auto M = [&](const std::string &n, int64_t r, int64_t cols) -> const void * {
            auto it = mats_.find(n);
            if (it == mats_.end() || it->second.rows != r || it->second.cols != cols || it->second.fmt != WFmt::BF16) {
                if (ok) why = n + " is missing, not bf16 or misshapen";
                ok = false;
                return nullptr;
            }
            return it->second.data;
        };
        auto Vv = [&](const std::string &n, int64_t len) -> const float * {
            auto it = vecs_.find(n);
            if (it == vecs_.end() || it->second.second != len) {
                if (ok) why = n + " is missing or misshapen";
                ok = false;
                return nullptr;
            }
            return it->second.first;
        };
        T.hq = c.hq, T.hk = c.hk, T.d = c.d, T.inter = c.inter, T.window = c.window, T.block = c.block, T.rank = c.rank, T.theta = c.theta;
        for (int i = 0; i < 4; ++i) T.taps[i] = c.taps[i];
        for (int i = 0; i < c.n_layers; ++i) {
            const std::string p = "mtp.L" + std::to_string(i) + ".";
            Ink2MetalMtpLayer L;
            L.local = c.local[i] != 0;
            L.q = M(p + "q", Cq, H), L.k = M(p + "k", Ck, H), L.v = M(p + "v", Ck, H), L.k_ctx = M(p + "k_ctx", Ck, H);
            L.v_ctx = M(p + "v_ctx", Ck, H), L.o = M(p + "o", H, Cq), L.gate = M(p + "gate", c.inter, H), L.up = M(p + "up", c.inter, H);
            L.down = M(p + "down", H, c.inter);
            L.dflare = Vv(p + "dflare", H), L.in_norm = Vv(p + "in_norm", H), L.post_norm = Vv(p + "post_norm", H);
            L.q_norm = Vv(p + "q_norm", c.d), L.k_norm = Vv(p + "k_norm", c.d), L.k_norm_ctx = Vv(p + "k_norm_ctx", c.d);
            T.layers.push_back(L);
        }
        T.rnn_s = M("mtp.rnn_s", 3 * R, 2 * R), T.rnn_x = M("mtp.rnn_x", 3 * R, 2 * H), T.w1 = M("mtp.w1", d_.vocab, R);
        const void *hout = M("mtp.hidden_out", H, R);
        T.mask = Vv("mtp.mask", H), T.norm = Vv("mtp.norm", H), T.m_dflare = Vv("mtp.m_dflare", H), T.rnn_b = Vv("mtp.rnn_b", 3 * R);
        const float *hb = Vv("mtp.hidden_out_b", H), *fuse = Vv("mtp.fuse", int64_t(c.n_layers + 1) * 4);
        if (!ok) return false;
        // each consumer's mix of the four features
        float *alpha = static_cast<float *>(alloc(size_t(c.n_layers + 1) * 16));
        for (int r = 0; r <= c.n_layers; ++r) {
            float row[4];
            std::memcpy(row, fuse + 4 * r, 16);
            softmax(row, 4);
            std::memcpy(alpha + 4 * r, row, 16);
        }
        T.alpha = alpha;
        // fold [V][R] = lm_head x hidden_out, fold_b = lm_head x its bias: R + 1 rows through the head
        {
            const size_t nH = size_t(H), nV = size_t(V), nR = size_t(R);
            Rows x((nR + 1) * nH), y((nR + 1) * nV);
            const uint16_t *w = static_cast<const uint16_t *>(hout);
            for (size_t j = 0; j < nH; ++j) {
                for (size_t r = 0; r < nR; ++r) {
                    const uint32_t u = uint32_t(w[j * nR + r]) << 16;
                    std::memcpy(&x[r * nH + j], &u, 4);
                }
                x[nR * nH + j] = hb[j];
            }
            b_->gemm(lm_head_, x.data(), R + 1, y.data(), V);
            uint16_t *fold = static_cast<uint16_t *>(alloc(nV * nR * 2));
            float *fb = static_cast<float *>(alloc(nV * 4));
            rows_for(V, [&](int64_t v) {
                for (size_t r = 0; r < nR; ++r) {
                    uint32_t u;
                    std::memcpy(&u, &y[r * nV + size_t(v)], 4);
                    fold[size_t(v) * nR + r] = uint16_t((u + 0x7FFF + ((u >> 16) & 1)) >> 16);  // nearest even
                }
                fb[v] = y[nR * nV + size_t(v)];
            }, 4 * R);
            T.fold = fold, T.fold_b = fb;
        }
        for (int i = 0; i < c.n_layers; ++i) {
            const size_t n = size_t(o_.n_pages) * size_t(o_.page) * size_t(Ck);
            const size_t eb = kv_half_ ? 2 : 4;
            T.dk.push_back(static_cast<float *>(alloc(n * eb))), T.dv.push_back(static_cast<float *>(alloc(n * eb)));
            kv_bytes_ += 2 * n * eb;
        }
        T.on = true;
        return true;
    }
#endif

    void setup_gpu() {
#if INK2_METAL
        if (!metal_) return;
        Ink2MetalModel M;
        M.d = d_, M.o = o_;
        for (const ILayer &L : layers_) {
            Ink2MetalLayer G;
            G.sliding = L.sliding, G.loop = L.loop, G.hq = int(L.hq), G.hk = int(L.hk), G.d = int(L.d), G.extent = int(L.extent);
            G.q = L.q.data, G.k = L.k.data, G.v = L.v.data, G.o = L.o.data, G.og = L.og.data, G.sink = L.sink.data, G.rel = L.rel.data;
            G.router = L.router.data, G.k_wp = L.k_wp.data, G.v_wp = L.v_wp.data;
            for (int i = 0; i < 4; ++i) G.c_down[i] = L.c_down[i].data, G.c_forget[i] = L.c_forget[i].data, G.c_gain[i] = L.c_gain[i].data;
            G.attn_query = L.attn_query, G.mlp_query = L.mlp_query, G.q_scale = L.q_scale, G.rel_scale = L.rel_scale;
            G.rel_proj = L.rel_proj, G.post_norm = L.post_norm, G.router_bias = L.router_bias, G.in_norm = L.in_norm;
            G.q_norm = L.q_norm, G.k_norm = L.k_norm, G.v_norm = L.v_norm;
            G.attn_conv = L.attn_conv.data(), G.mlp_conv = L.mlp_conv.data(), G.k_conv = L.k_conv.data(), G.v_conv = L.v_conv.data();
            G.k_static = L.k_static.data(), G.v_static = L.v_static.data();
            M.layers.push_back(G);
        }
        M.lm_head = lm_head_.data, M.final_norm = final_norm_, M.out_query = out_query_, M.msg_queries = msg_queries_;
        M.kpool = kpool_, M.vpool = vpool_, M.conv = conv_, M.vis_off = vis_off_;
        M.undo = undo_, M.undo_stride = int64_t(undo_stride_), M.undo_rows = kUndo;
        M.tails = tails_;
        M.w8_dense = w8_dense_, M.w8_head = w8_head_;
        std::string head_why;
        if (mtp_set_ && !setup_mtp(M.mtp, head_why)) std::fprintf(stderr, "fullm: inkling2's draft head is not used (%s)\n", head_why.c_str());
        bool bf16 = lm_head_.fmt == WFmt::BF16;
        for (auto &kv : mats_) bf16 = bf16 && kv.second.fmt == WFmt::BF16;
        std::string why = "a dense matrix is not bf16";
        gpu_ = bf16 && ink2_metal_setup(metal_, M, why);
        mtp_on_ = gpu_ && M.mtp.on;
        if (!gpu_) std::fprintf(stderr, "fullm: inkling2 stays on the CPU (%s)\n", why.c_str());
#endif
    }


    int layer_of(int v) const { return v < d_.n_layers ? v : d_.loop_start + (v - d_.n_layers) % n_loop_; }
    int visit(int l, int p) const { return p == 0 ? l : d_.n_layers + (p - 1) * n_loop_ + (l - d_.loop_start); }
    void check_slot(int32_t s) const {
        if (!finalized_ || s < 0 || s >= o_.n_slots) throw std::runtime_error("slot out of range");
    }

    void validate(const sk_step_plan &pl) const {
        auto fail = [](const char *m) { throw std::runtime_error(std::string("bad step plan: ") + m); };
        if (pl.n_tokens <= 0 || pl.n_tokens > o_.max_tokens) fail("token count");
        if (pl.n_rows <= 0 || pl.n_rows > o_.max_rows) fail("row count");
        const int64_t cap = int64_t(o_.max_pages_per_seq) * o_.page;
        for (int i = 0; i < pl.n_tokens; ++i) {
            if (pl.tokens[i] < 0 || pl.tokens[i] >= d_.vocab) fail("token id out of vocabulary");
            const int32_t s = pl.token_slot[i], p = pl.positions[i];
            if (s < 0 || s >= o_.n_slots) fail("slot out of range");
            if (p < 0 || p >= cap) fail("position out of range");
            const int32_t pg = pl.page_table[int64_t(s) * o_.max_pages_per_seq + p / o_.page];
            if (pg < 0 || pg >= o_.n_pages) fail("a position has no page");
        }
        int64_t covered = 0;
        for (int i = 0; i < pl.n_segs; ++i) {
            const int32_t s = pl.segs[3 * i], first = pl.segs[3 * i + 1], n = pl.segs[3 * i + 2];
            if (s < 0 || s >= o_.n_slots || first != covered || n <= 0 || first + n > pl.n_tokens) fail("segments");
            for (int j = 0; j < n; ++j)
                if (pl.token_slot[first + j] != s || pl.positions[first + j] != pl.positions[first] + j) fail("segment rows");
            covered += n;
        }
        if (covered != pl.n_tokens) fail("segments do not cover the tokens");
        for (int i = 0; i < pl.n_rows; ++i)
            if (pl.rows[i] < 0 || pl.rows[i] >= pl.n_tokens) fail("logits row out of range");
        for (int i = 0; i < pl.n_reset; ++i)
            if (pl.reset_slots[i] < 0 || pl.reset_slots[i] >= o_.n_slots) fail("reset slot out of range");
    }

    // f(i) for i in [0, n): on the pool, or on the caller when the whole job is about `work`
    // multiply-adds an item and too small to be worth waking the pool for (a decode step has
    // some thirty such loops a layer visit; woken for each, the threads spent more time starting
    // than computing)
    void rows_for(int64_t n, const std::function<void(int64_t)> &f, int64_t work = int64_t(1) << 30) {
        if (n <= 1 || n * work < (int64_t(1) << 17)) {
            for (int64_t i = 0; i < n; ++i) f(i);
            return;
        }
        pool_->parallel_for(n, f, small_);
    }

    Rows gemm(const Weight &w, const Rows &x) {
        const int64_t n = int64_t(x.size()) / w.cols;
        Rows y(size_t(n * w.rows));
        const auto tg = Clock::now();
        struct Timer {
            double &acc;
            Clock::time_point t;
            ~Timer() { acc += ms_since(t); }
        } timer{ms_gemm_, tg};
        b_->gemm(w, x.data(), n, y.data(), w.rows);
        return y;
    }

    struct GemmJob {
        const Weight *w;
        const Rows *x;
        Rows *y;
    };
    void gemm_many(const std::vector<GemmJob> &jobs) {
        for (const GemmJob &j : jobs) *j.y = gemm(*j.w, *j.x);
    }

    Rows rms(const Rows &x, const float *w) {
        Rows y(x.size());
        b_->rmsnorm(y.data(), x.data(), w, int64_t(x.size()) / d_.hidden, d_.hidden, d_.hidden, d_.rms_eps);
        return y;
    }

    // Block Attention Residuals: each of nq queries attends, at every row alike, over the finished
    // blocks and the running sum of the current one (RMS-normalised, no gain).
    std::vector<Rows> attn_res(const float *queries, int nq, const std::vector<Rows> &blocks, const Rows &partial) {
        const int64_t H = d_.hidden, N = int64_t(partial.size()) / H, n = int64_t(blocks.size()) + 1;
        std::vector<Rows> out(size_t(nq), Rows(partial.size(), 0.0f));
        const float sc = 1.0f / std::sqrt(float(H));
        rows_for(N, [&](int64_t r) {
            thread_local std::vector<float> st, p;
            st.resize(size_t(n * H)), p.resize(size_t(n));
            for (int64_t i = 0; i < n; ++i) {
                const Rows &src = i + 1 < n ? blocks[size_t(i)] : partial;
                std::memcpy(st.data() + i * H, src.data() + r * H, size_t(H) * 4);
                unorm(st.data() + i * H, H, 1e-6f);
            }
            for (int qi = 0; qi < nq; ++qi) {
                sk_cpu_gemm_f32(queries + qi * H, 1, H, st.data(), 0, n, p.data(), n);
                for (int64_t i = 0; i < n; ++i) p[size_t(i)] *= sc;
                softmax(p.data(), n);
                float *o = out[size_t(qi)].data() + r * H;
                for (int64_t i = 0; i < n; ++i) {
                    const float pi = p[size_t(i)], *s = st.data() + i * H;
                    for (int64_t j = 0; j < H; ++j) o[j] += pi * s[j];
                }
            }
        }, n * H * (nq + 1));
        return out;
    }

    // Message gates `which[i]` (0 query, 1 key, 2 value, 3 mlp) on hs[i] and msgs[which[i]]:
    // (1 - forget) * h + gain * message, both gates tanh of a low-rank map of [norm(h) | norm(m)].
    // The gates' GEMMs go out together (two batches for all of them).
    std::vector<Rows> gates(const ILayer &L, const std::vector<int> &which, const std::vector<const Rows *> &hs,
                            const std::vector<Rows> &msgs) {
        const int64_t H = d_.hidden, N = int64_t(hs[0]->size()) / H;
        const size_t n = which.size();
        std::vector<Rows> cat(n, Rows(size_t(N * 2 * H))), g(n), forget(n), gain(n);
        for (size_t i = 0; i < n; ++i) {
            const Rows &h = *hs[i], &msg = msgs[size_t(which[i])];
            rows_for(N, [&](int64_t r) {
                float *c = cat[i].data() + r * 2 * H;
                std::memcpy(c, h.data() + r * H, size_t(H) * 4);
                std::memcpy(c + H, msg.data() + r * H, size_t(H) * 4);
                unorm(c, H, d_.rms_eps), unorm(c + H, H, d_.rms_eps);
            }, 4 * H);
        }
        std::vector<GemmJob> jobs;
        for (size_t i = 0; i < n; ++i) jobs.push_back({&L.c_down[which[i]], &cat[i], &g[i]});
        gemm_many(jobs);
        jobs.clear();
        for (size_t i = 0; i < n; ++i) {
            jobs.push_back({&L.c_forget[which[i]], &g[i], &forget[i]});
            jobs.push_back({&L.c_gain[which[i]], &g[i], &gain[i]});
        }
        gemm_many(jobs);
        for (size_t i = 0; i < n; ++i) {
            const Rows &h = *hs[i], &msg = msgs[size_t(which[i])];
            Rows &f = forget[i];
            const Rows &gn = gain[i];
            rows_for(N, [&](int64_t r) {
                for (int64_t x = r * H; x < (r + 1) * H; ++x)
                    f[size_t(x)] = (1.0f - std::tanh(f[size_t(x)])) * h[size_t(x)] + std::tanh(gn[size_t(x)]) * msg[size_t(x)];
            }, 16 * H);
        }
        return forget;
    }

    float *conv_state(int slot, int vis, int which, int64_t Ck) {
        const int64_t H = d_.hidden, K1 = d_.conv_k - 1;
        const int64_t off[4] = {0, Ck * K1, 2 * Ck * K1, 2 * Ck * K1 + H * K1};
        return conv_ + size_t(slot) * vis_off_.back() + vis_off_[size_t(vis)] + off[which];
    }

    float *undo_state(int slot, int vis, int which, int64_t Ck) {
        const int64_t H = d_.hidden, K1 = d_.conv_k - 1, rows = K1 + kUndo;
        const int64_t off[4] = {0, Ck * rows, 2 * Ck * rows, 2 * Ck * rows + H * rows};
        return undo_ + size_t(slot) * undo_stride_ + size_t(vis_off_[size_t(vis)]) / size_t(K1) * size_t(rows) + off[which];
    }

    // The short convolutions over each sequence's rows, in place: `taps` [tap][channel]. Static
    // (dyn null): tap j weighs the input K - 1 - j rows back, then (out + x) / sqrt(K + 1).
    // Dynamic (the full layers' keys and values): tap `lag` weighs the input lag rows back by
    // taps[lag][c] + dyn[row][head * K + lag], then (out + x) / sqrt(2K + 1).
    void sconv(Rows &x, const std::vector<float> &taps, const float *dyn, int heads, int vis, int which, int64_t C, int64_t Ck) {
        const int64_t K = d_.conv_k, K1 = K - 1;
        const float norm = 1.0f / std::sqrt(float(dyn ? 2 * K + 1 : K + 1));
        const sk_step_plan &pl = *pl_;
        Rows y(x.size());
        for (int s = 0; s < pl.n_segs; ++s) {
            const int64_t first = pl.segs[3 * s + 1], n = pl.segs[3 * s + 2];
            float *hist = conv_state(pl.segs[3 * s], vis, which, Ck);
            ctx_.resize(size_t((n + K1) * C));
            std::memcpy(ctx_.data(), hist, size_t(K1 * C) * 4);
            std::memcpy(ctx_.data() + K1 * C, x.data() + first * C, size_t(n * C) * 4);
            std::memcpy(hist, ctx_.data() + n * C, size_t(K1 * C) * 4);
            if (tails_) {  // rows that end a page: the history after them, by page
                const int slot = pl.segs[3 * s];
                const int64_t pos0 = pl.positions[first], pg = o_.page;
                for (int64_t t = pg - 1 - pos0 % pg; t < n; t += pg) {
                    const int64_t page = pl.page_table[int64_t(slot) * o_.max_pages_per_seq + (pos0 + t) / pg];
                    const int64_t H = d_.hidden, off[4] = {0, Ck * K1, 2 * Ck * K1, 2 * Ck * K1 + H * K1};
                    std::memcpy(tails_ + size_t(page) * vis_off_.back() + vis_off_[size_t(vis)] + off[which],
                                ctx_.data() + (t + 1) * C, size_t(K1 * C) * 4);
                }
            }
            if (n >= 2) {  // may end in drafts: the rows a rejection puts back
                const int64_t m = std::min<int64_t>(n, kUndo) + K1;
                std::memcpy(undo_state(pl.segs[3 * s], vis, which, Ck), ctx_.data() + (n + K1 - m) * C, size_t(m * C) * 4);
            }
            rows_for(n, [&](int64_t t) {
                float *o = y.data() + (first + t) * C;
                const float *xr = x.data() + (first + t) * C;
                std::fill(o, o + C, 0.0f);
                for (int64_t j = 0; j < K; ++j) {
                    // static: tap j on ctx row t + j; dynamic: lag j on ctx row t + K - 1 - j
                    const float *c = ctx_.data() + (dyn ? t + K1 - j : t + j) * C, *w = taps.data() + j * C;
                    if (!dyn) {
                        for (int64_t i = 0; i < C; ++i) o[i] += w[i] * c[i];
                    } else {
                        const int64_t per = C / heads;
                        const float *dw = dyn + (first + t) * heads * K;
                        for (int hd = 0; hd < heads; ++hd) {
                            const float d = dw[hd * K + j];
                            for (int64_t i = hd * per; i < (hd + 1) * per; ++i) o[i] += (w[i] + d) * c[i];
                        }
                    }
                }
                for (int64_t i = 0; i < C; ++i) o[i] = (o[i] + xr[i]) * norm;
            }, C * (K + 1));
        }
        x.swap(y);
    }

    float *kv_rows(float *pool, const ILayer &L, int slot, int64_t pos, int64_t kh) const {
        const int64_t page = pl_->page_table[int64_t(slot) * o_.max_pages_per_seq + pos / o_.page];
        return pool + ((page * L.hk + kh) * o_.page + pos % o_.page) * L.d;
    }

    Rows attention(const ILayer &L, int vis, const Rows &xq, const Rows &xk, const Rows &xv) {
        const sk_step_plan &pl = *pl_;
        const int64_t H = d_.hidden, N = pl.n_tokens, Hq = L.hq, Hk = L.hk, D = L.d, G = Hq / Hk;
        const int64_t Cq = Hq * D, Ck = Hk * D, dr = d_.d_rel, ext = L.extent;
        const bool full = !L.sliding, same = &xq == &xk;
        (void)H;
        (void)same;
        auto tp = Clock::now();
        auto lap = [&](int i) {
            ms_ph_[i] += ms_since(tp);
            tp = Clock::now();
        };
        Rows q, rel, sink, og, k, v;
        gemm_many({{&L.q, &xq, &q}, {&L.og, &xq, &og}, {&L.k, &xk, &k}, {&L.v, &xv, &v}, {&L.rel, &xq, &rel}, {&L.sink, &xq, &sink}});
        rows_for(N, [&](int64_t r) {
            float *qr = q.data() + r * Cq, *rr = rel.data() + r * Hq * dr;
            for (int64_t i = 0; i < Cq; ++i) qr[i] *= L.q_scale[i];
            for (int64_t i = 0; i < Hq * dr; ++i) rr[i] *= L.rel_scale[i];
            for (int64_t hd = 0; hd < Hq; ++hd) l2_clip(qr + hd * D, D, d_.qk_clip), l2_clip(rr + hd * dr, dr, d_.rel_clip);
            if (full) {  // queries and relative states grow with the position past the floor
                const float tau = 1.0f + d_.log_alpha * std::log(std::max(1.0f, float(pl.positions[r] + 1) / d_.log_n_floor));
                for (int64_t i = 0; i < Cq; ++i) qr[i] *= tau;
                for (int64_t i = 0; i < Hq * dr; ++i) rr[i] *= tau;
            }
        }, 4 * Cq);
        lap(0);
        if (full) {
            const Rows dk = gemm(L.k_wp, xk), dv = gemm(L.v_wp, same ? xk : xv);
            sconv(k, L.k_static, dk.data(), int(Hk), vis, 0, Ck, Ck);
            sconv(v, L.v_static, dv.data(), int(Hk), vis, 1, Ck, Ck);
        } else {
            sconv(k, L.k_conv, nullptr, 0, vis, 0, Ck, Ck);
            sconv(v, L.v_conv, nullptr, 0, vis, 1, Ck, Ck);
        }
        lap(1);
        // keys clipped, values normalised, into the pools (every row before any row attends: a
        // chunk's rows read each other's)
        float *kp = kpool_[size_t(vis)], *vp = vpool_[size_t(vis)];
        rows_for(N, [&](int64_t r) {
            for (int64_t kh = 0; kh < Hk; ++kh) {
                float *kr = k.data() + r * Ck + kh * D, *vr = v.data() + r * Ck + kh * D;
                l2_clip(kr, D, d_.qk_clip);
                unorm(vr, D, d_.rms_eps);
                std::memcpy(kv_rows(kp, L, pl.token_slot[r], pl.positions[r], kh), kr, size_t(D) * 4);
                std::memcpy(kv_rows(vp, L, pl.token_slot[r], pl.positions[r], kh), vr, size_t(D) * 4);
            }
        }, 4 * Ck);
        lap(2);
        Rows out(size_t(N * Cq), 0.0f);
        const float scale = 1.0f / std::sqrt(float(D));
        const int64_t page = o_.page;
        lap(3);
        int64_t max_ctx = 0;
        for (int64_t r = 0; r < N; ++r) max_ctx = std::max<int64_t>(max_ctx, pl.positions[r] + 1);
        if (!full) max_ctx = std::min<int64_t>(max_ctx, d_.window);
        rows_for(
            N * Hk,
            [&](int64_t i) {
                const int64_t r = i / Hk, kh = i % Hk, pos = pl.positions[r];
                const int slot = pl.token_slot[r];
                const int64_t lo = full ? 0 : std::max<int64_t>(0, pos + 1 - d_.window), T = pos + 1 - lo;
                thread_local std::vector<float> sc, bias;
                sc.resize(size_t(G * (T + 1))), bias.resize(size_t(ext));
                const float *qv = q.data() + r * Cq + kh * G * D;
                // the G query heads of this kv head against its keys, a page at a time
                for (int64_t j = lo; j <= pos;) {
                    const int64_t n = std::min(pos + 1, (j / page + 1) * page) - j;
                    sk_cpu_gemm_f32(qv, G, D, kv_rows(kp, L, slot, j, kh), 0, n, sc.data() + (j - lo), T + 1);
                    j += n;
                }
                for (int64_t g = 0; g < G; ++g) {
                    const int64_t hd = kh * G + g;
                    float *s = sc.data() + g * (T + 1);
                    // the bias by backward distance: the relative state's mix of the trained profiles
                    const int64_t nb = std::min(ext, T);
                    std::fill(bias.begin(), bias.begin() + nb, 0.0f);
                    const float *rv = rel.data() + (r * Hq + hd) * dr;
                    for (int64_t a = 0; a < dr; ++a) {
                        const float ra = rv[a], *pr = L.rel_proj + a * ext;
                        for (int64_t dd = 0; dd < nb; ++dd) bias[size_t(dd)] += ra * pr[dd];
                    }
                    for (int64_t t = 0; t < T; ++t) {
                        const int64_t dist = T - 1 - t;
                        s[t] = s[t] * scale + (dist < ext ? bias[size_t(dist)] : 0.0f);
                    }
                    s[T] = sink[size_t(r * Hq + hd)];  // takes probability mass, adds nothing
                    softmax(s, T + 1);
                    float *o = out.data() + r * Cq + hd * D;
                    for (int64_t j = lo; j <= pos;) {
                        const int64_t n = std::min(pos + 1, (j / page + 1) * page) - j;
                        const float *vr = kv_rows(vp, L, slot, j, kh);
                        for (int64_t t = 0; t < n; ++t) {
                            const float pj = s[j - lo + t], *vv = vr + t * D;
                            for (int64_t x = 0; x < D; ++x) o[x] += pj * vv[x];
                        }
                        j += n;
                    }
                }
            },
            G * (2 * max_ctx * D + dr * std::min(ext, max_ctx)));
        lap(4);
        rows_for(N, [&](int64_t r) {
            for (int64_t i = r * Cq; i < (r + 1) * Cq; ++i) out[size_t(i)] *= sigmoidf(og[size_t(i)]);
        }, 8 * Cq);
        Rows res = gemm(L.o, out);
        lap(5);
        return res;
    }

    // The routed experts: sigmoid router, the top-k by score + bias with weights from the scores,
    // each expert's tokens gathered and run through its matrices (rows decoded in place).
    Rows moe(const ILayer &L, const Rows &x) {
        const auto t0 = Clock::now();
        const int64_t H = d_.hidden, N = int64_t(x.size()) / H, E = d_.n_experts, k = d_.topk, I = d_.moe_inter;
        const Rows logits = gemm(L.router, x);
        std::vector<int32_t> idx(size_t(N * k));
        std::vector<float> wt(size_t(N * k));
        rows_for(N, [&](int64_t r) {
            thread_local std::vector<float> sc;
            thread_local std::vector<int32_t> ord;
            sc.resize(size_t(E)), ord.resize(size_t(E));
            for (int64_t e = 0; e < E; ++e) sc[size_t(e)] = sigmoidf(logits[size_t(r * E + e)]), ord[size_t(e)] = int32_t(e);
            std::partial_sort(ord.begin(), ord.begin() + k, ord.end(), [&](int32_t a, int32_t b) {
                const float sa = sc[size_t(a)] + L.router_bias[a], sb = sc[size_t(b)] + L.router_bias[b];
                return sa > sb || (sa == sb && a < b);
            });
            float sum = 1e-20f;
            for (int64_t j = 0; j < k; ++j) sum += sc[size_t(ord[size_t(j)])];
            for (int64_t j = 0; j < k; ++j) {
                idx[size_t(r * k + j)] = ord[size_t(j)];
                wt[size_t(r * k + j)] = sc[size_t(ord[size_t(j)])] / sum * d_.route_scale;
            }
        }, 16 * E);
        // pairs by expert: start[e] .. start[e + 1] index `pairs` (pair p = row * k + j)
        std::vector<int64_t> start(size_t(E) + 1, 0), pairs(size_t(N * k)), slot_of(size_t(N * k));
        for (int64_t p = 0; p < N * k; ++p) start[size_t(idx[size_t(p)]) + 1]++;
        for (int64_t e = 0; e < E; ++e) start[size_t(e) + 1] += start[size_t(e)];
        {
            std::vector<int64_t> at(start.begin(), start.end() - 1);
            for (int64_t p = 0; p < N * k; ++p) slot_of[size_t(p)] = at[size_t(idx[size_t(p)])], pairs[size_t(at[size_t(idx[size_t(p)])]++)] = p;
        }
        std::vector<int64_t> used;
        for (int64_t e = 0; e < E; ++e)
            if (start[size_t(e) + 1] > start[size_t(e)]) used.push_back(e);
        Rows xg(size_t(N * k * H)), y13(size_t(N * k * 2 * I)), act(size_t(N * k * I)), y2(size_t(N * k * H));
        rows_for(N * k, [&](int64_t i) {
            std::memcpy(xg.data() + i * H, x.data() + (pairs[size_t(i)] / k) * H, size_t(H) * 4);
        }, H / 4);
        // tasks: (expert, a range of the matrix's rows); each row is decoded once and dotted with
        // every token of the expert
        struct Task {
            int64_t e, n0, n1;
        };
        auto tasks_for = [&](int64_t rows) {
            std::vector<Task> t;
            const int64_t threads = pool_->size(small_);
            for (int64_t e : used) {
                const int64_t m = start[size_t(e) + 1] - start[size_t(e)];
                // about a dozen ranges a thread over the step, none below 32 rows
                const int64_t want = std::max<int64_t>(1, threads * 12 / int64_t(used.size()));
                const int64_t stepn = std::max<int64_t>(32, std::min<int64_t>(rows, (rows + want - 1) / want));
                (void)m;
                for (int64_t n0 = 0; n0 < rows; n0 += stepn) t.push_back({e, n0, std::min(rows, n0 + stepn)});
            }
            return t;
        };
        // NVFP4 on a CPU with integer dots: the tokens quantised to int8 pairs once, the experts'
        // rows dotted as integers (nvfp4_rows); else rows decoded to floats
        const bool sdot = INK2_SDOT && L.x13.fmt == SK_XFMT_NVFP4 && sdot_on_;
        std::vector<int8_t> qh, ql;
        std::vector<float> q1, q2;
        auto quantise = [&](const Rows &src, int64_t width) {
            (void)src, (void)width;
#if INK2_SDOT
            qh.resize(size_t(N * k * width)), ql.resize(size_t(N * k * width)), q1.resize(size_t(N * k)), q2.resize(size_t(N * k));
            rows_for(N * k, [&](int64_t i) {
                quant_hilo(src.data() + i * width, width, qh.data() + i * width, ql.data() + i * width, &q1[size_t(i)], &q2[size_t(i)]);
            }, 4 * width);
#endif
        };
        auto run = [&](const XStack &xs, const Rows &src, int64_t width, Rows &dst, int64_t rows) {
            if (sdot) quantise(src, width);
            const std::vector<Task> ts = tasks_for(rows);
            pool_->parallel_for(
                int64_t(ts.size()),
                [&](int64_t i) {
                    const Task &t = ts[size_t(i)];
                    const int64_t s = start[size_t(t.e)], m = start[size_t(t.e) + 1] - s;
#if INK2_SDOT
                    if (sdot) {
                        nvfp4_rows(xs, t.e, t.n0, t.n1, qh.data() + s * width, ql.data() + s * width, q1.data() + s, q2.data() + s,
                                   m, dst.data() + s * rows, rows);
                        return;
                    }
#endif
                    XRow ctx{&xs, t.e};
                    sk_cpu_gemm_rows(src.data() + s * width, m, width, t.n0, t.n1, dst.data() + s * rows, rows, decode_row, &ctx);
                },
                small_);
        };
        run(L.x13, xg, H, y13, 2 * I);
        const float gc = d_.gate_softcap, uc = d_.up_softcap;
        rows_for(N * k, [&](int64_t i) {
            const float *gu = y13.data() + i * 2 * I;
            float *a = act.data() + i * I;
            for (int64_t j = 0; j < I; ++j) a[j] = softcap(gu[2 * j], gc) * sigmoidf(gu[2 * j]) * softcap(gu[2 * j + 1], uc);
        }, 24 * I);
        run(L.x2, act, I, y2, H);
        // each row's experts in selection order
        Rows out(size_t(N * H), 0.0f);
        rows_for(N, [&](int64_t r) {
            float *o = out.data() + r * H;
            for (int64_t j = 0; j < k; ++j) {
                const float w = wt[size_t(r * k + j)], *y = y2.data() + slot_of[size_t(r * k + j)] * H;
                for (int64_t i = 0; i < H; ++i) o[i] += w * y[i];
            }
        }, k * H);
        const double ms = ms_since(t0);
        ms_experts_ += ms;
        (L.x13.fmt == SK_XFMT_NVFP4 ? ms_nv_ : ms_mx_) += ms;
        return out;
    }

    // Layer l's pass p: h in, h out; blocks: the finished blocks' states; msgs: the four messages
    // (query, key, value, mlp) in passes after the first.
    Rows layer(int l, int p, const Rows &h, const std::vector<Rows> &blocks, const std::vector<Rows> *msgs) {
        const ILayer &L = layers_[size_t(l)];
        const int64_t H = d_.hidden, vis = visit(l, p);
        std::vector<Rows> ins;
        if (l > 0) ins = attn_res(L.attn_query, L.sliding ? 1 : 3, blocks, h);
        else ins.push_back(h);
        Rows a;
        const auto ta = Clock::now();
        if (L.sliding) {
            if (!msgs) {
                const Rows x = rms(ins[0], L.in_norm);
                a = attention(L, int(vis), x, x, x);
            } else {
                const std::vector<Rows> gd = gates(L, {0, 1, 2}, {&ins[0], &ins[0], &ins[0]}, *msgs);
                const Rows xq = rms(gd[0], L.in_norm), xk = rms(gd[1], L.in_norm), xv = rms(gd[2], L.in_norm);
                a = attention(L, int(vis), xq, xk, xv);
            }
        } else {
            if (ins.size() == 1) ins.resize(3, ins[0]);
            if (msgs) ins = gates(L, {0, 1, 2}, {&ins[0], &ins[1], &ins[2]}, *msgs);
            const Rows xq = rms(ins[0], L.q_norm), xk = rms(ins[1], L.k_norm), xv = rms(ins[2], L.v_norm);
            a = attention(L, int(vis), xq, xk, xv);
        }
        ins.clear();
        ms_attn_ += ms_since(ta);
        sconv(a, L.attn_conv, nullptr, 0, int(vis), 2, H, L.hk * L.d);
        // a block starts: the stream so far becomes a block state and the running sum restarts
        const bool boundary = l % d_.block == 0;
        std::vector<Rows> with;
        const std::vector<Rows> *b2 = &blocks;
        if (boundary) {
            with = blocks;
            with.push_back(h);
            b2 = &with;
        } else {
            for (size_t i = 0; i < a.size(); ++i) a[i] += h[i];
        }
        Rows m = std::move(attn_res(L.mlp_query, 1, *b2, a)[0]);
        if (msgs) m = std::move(gates(L, {3}, {&m}, *msgs)[0]);
        m = moe(L, rms(m, L.post_norm));
        sconv(m, L.mlp_conv, nullptr, 0, int(vis), 3, H, L.hk * L.d);
        for (size_t i = 0; i < a.size(); ++i) a[i] += m[i];
        return a;
    }

    // Layers [l0, l1) in pass p; prev: the previous pass's block states and output, the source of
    // this pass's messages.
    void run_layers(int l0, int l1, int p, Rows &h, std::vector<Rows> &blocks, const std::vector<Rows> *prev_blocks,
                    const Rows *prev_h) {
        for (int l = l0; l < l1; ++l) {
            std::vector<Rows> msgs;
            if (prev_h) msgs = attn_res(msg_queries_ + int64_t(l - d_.loop_start) * 4 * d_.hidden, 4, *prev_blocks, *prev_h);
            Rows nh = layer(l, p, h, blocks, prev_h ? &msgs : nullptr);
            if (l % d_.block == 0) blocks.push_back(std::move(h));
            h = std::move(nh);
        }
    }

    sk_ink2_dims d_;
    sk_exec_opts o_;
    std::vector<uint8_t> sliding_;
    std::unique_ptr<Backend> b_;
    std::unique_ptr<ThreadPool> pool_;
    int n_loop_ = 0, visits_ = 0;
    bool finalized_ = false, small_ = false;
    const bool gpu_f32_ = [] {
        const char *e = std::getenv("FULLM_INK2_METAL_F32");
        return e && *e == '1';
    }();
    // FULLM_INK2_SDOT=0: NVFP4 experts by rows decoded to floats (exact fp32 activations)
    const bool sdot_on_ = [] {
        const char *e = std::getenv("FULLM_INK2_SDOT");
        return !(e && *e == '0');
    }();
    std::map<std::string, Weight> mats_;
    std::map<std::string, std::pair<float *, int64_t>> vecs_;
    std::vector<std::pair<XStack, XStack>> xs_;
    Weight embed_, lm_head_;
    const float *embed_norm_ = nullptr, *final_norm_ = nullptr, *out_query_ = nullptr, *msg_queries_ = nullptr;
    std::vector<ILayer> layers_;
    std::vector<int64_t> vis_off_;               // each visit's convolution histories within a slot (floats)
    std::vector<float *> kpool_, vpool_;  // [visit]
    float *conv_ = nullptr, *undo_ = nullptr, *tails_ = nullptr;  // tails_: [page][a slot's histories]
    bool tails_on_ = false;
    size_t conv_n_ = 0, undo_stride_ = 0;  // floats a slot
    static constexpr int64_t kUndo = 16;   // rows of a step a rejection can take back
    std::vector<int32_t> last_n_;          // [slot]: rows of its segment in the last step
    // [slot]: of its segment in its last step, the rows kept (after drop_tail), the first row, the step
    std::vector<int32_t> last_kept_, last_first_;
    std::vector<uint64_t> last_step_;
    uint64_t step_no_ = 0;
    sk_ink2_mtp mtp_{};
    bool mtp_set_ = false, mtp_on_ = false;  // on: the head drafts (the steps run on the GPU)
    std::vector<void *> owned_;  // the backend's allocations (none when the GPU hands out the memory)
    std::vector<float> ctx_, logits_;
    std::vector<int32_t> ids_, hids_;
    std::vector<SampleRow> sample_;
    SampleScratch sample_scratch_;
    const sk_step_plan *pl_ = nullptr;
    size_t weight_bytes_ = 0, expert_bytes_ = 0, kv_bytes_ = 0;
#if INK2_METAL
    Ink2Metal *metal_ = nullptr;
    bool w8_dense_ = false, w8_head_ = false;  // int8 matrices on the GPU path (ink2_int8)
    bool gpu_ = false;      // steps run on it
    bool kv_half_ = false;  // the pools hold half (the GPU path only)
#endif
    double ms_experts_ = 0, ms_nv_ = 0, ms_mx_ = 0, ms_attn_ = 0, ms_gemm_ = 0;
    double ms_ph_[6] = {0, 0, 0, 0, 0, 0};  // attention's phases (FULLM_INK2_PROF)
    const bool prof_ = std::getenv("FULLM_INK2_PROF") != nullptr;
    std::string desc_;
};

}  // namespace

std::unique_ptr<ExecIface> make_ink2_exec(const sk_ink2_dims &d, const sk_exec_opts &o) {
    return std::make_unique<Ink2Exec>(d, o);
}

// The integer NVFP4 rows against the same rows decoded to floats and dotted in double, on random
// codes, scales and activations: the largest error over the outputs' RMS (-1: no integer path in
// this build).
double ink2_sdot_error() {
#if INK2_SDOT
    const int64_t R = 24, K = 2304, M = 5;
    uint32_t state = 12345;
    auto rnd = [&] { return state = state * 1664525u + 1013904223u, (state >> 8) & 0xFFFF; };
    const size_t nK = static_cast<size_t>(K), nM = static_cast<size_t>(M), nR = static_cast<size_t>(R);
    std::vector<uint8_t> codes(nR * nK / 2), sc(nR * nK / 16);
    for (auto &c : codes) c = uint8_t(rnd());
    for (auto &c : sc) c = uint8_t(0x30 + rnd() % 0x20);  // e4m3 scales around 0.1 .. 2
    const float factor[1] = {0.037f};
    const XStack xs{SK_XFMT_NVFP4, codes.data(), sc.data(), factor, R, K};
    std::vector<float> x(nM * nK), y(nM * nR), row(nK), s1(nM), s2(nM);
    for (int64_t i = 0; i < M * K; ++i) {
        const float u = float(rnd()) / 65536.0f - 0.5f;
        x[size_t(i)] = u * (i % 97 == 0 ? 40.0f : 2.0f);  // a few outliers, as activations have
    }
    std::vector<int8_t> hi(nM * nK), lo(nM * nK);
    for (int64_t m = 0; m < M; ++m) quant_hilo(x.data() + m * K, K, hi.data() + m * K, lo.data() + m * K, &s1[size_t(m)], &s2[size_t(m)]);
    double worst = 0, ss = 0;
    for (int64_t pass = 0; pass < 2; ++pass) {  // one token at a time, then all (the two forms)
        if (pass == 0)
            for (int64_t m = 0; m < M; ++m)
                nvfp4_rows(xs, 0, 0, R, hi.data() + m * K, lo.data() + m * K, &s1[size_t(m)], &s2[size_t(m)], 1, y.data() + m * R, R);
        else nvfp4_rows(xs, 0, 0, R, hi.data(), lo.data(), s1.data(), s2.data(), M, y.data(), R);
        for (int64_t n = 0; n < R; ++n) {
            XRow ctx{&xs, 0};
            decode_row(&ctx, n, row.data());
            for (int64_t m = 0; m < M; ++m) {
                double ref = 0;
                for (int64_t i = 0; i < K; ++i) ref += double(row[size_t(i)]) * x[size_t(m * K + i)];
                worst = std::max(worst, std::fabs(ref - y[size_t(m * R + n)]));
                ss += ref * ref;
            }
        }
    }
    return worst / std::sqrt(ss / double(2 * R * M));
#else
    return -1.0;
#endif
}

}  // namespace fullm

extern "C" double sk_ink2_sdot_error(void) { return fullm::ink2_sdot_error(); }
