// SPDX-License-Identifier: Apache-2.0
// cpu_backend.cpp - the Backend interface on host memory.
//
// Two jobs: run the model graph on any machine (so the C++ graph is checked against the
// torch oracle without a GPU), and serve as the reference the CUDA backend's self test
// compares against op by op. Arithmetic mirrors crates/fullm/src/ops.rs: double
// accumulators for norms, the same op order, the fullm-kernels SIMD dots for matmuls.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "fp8_block.h"
#include "int4_group.h"
#include "backend.h"
#include "fullm_kernels.h"
#include "threadpool.h"

namespace fullm {

size_t Weight::bytes() const {
    const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    switch (fmt) {
        case WFmt::F32: return n * 4;
        case WFmt::BF16: return n * 2;
        case WFmt::INT8:
        case WFmt::FP8: return n + static_cast<size_t>(rows) * 4;
        case WFmt::FP8B: return n + static_cast<size_t>(fp8b_scales(rows, cols)) * 4;
        case WFmt::INT4G: return static_cast<size_t>(int4g_bytes(rows, cols)) + static_cast<size_t>(int4g_scales(rows, cols)) * 4;
    }
    return 0;
}

namespace {

inline float sigm(float x) { return 1.0f / (1.0f + std::exp(-x)); }
// torch softplus (beta 1, threshold 20)
inline float softplus_(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

inline float bf16_to_f32(uint16_t h) {
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline uint16_t f32_to_bf16(float v) {  // round to nearest even
    uint32_t u;
    std::memcpy(&u, &v, 4);
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x7fffffu)) return static_cast<uint16_t>((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

// Decode row r of any weight format into out[cols].
void weight_row(const Weight &w, int64_t r, float *out) {
    const int64_t c = w.cols;
    switch (w.fmt) {
        case WFmt::F32:
            std::memcpy(out, static_cast<const float *>(w.data) + r * c, c * 4);
            break;
        case WFmt::BF16: {
            const uint16_t *p = static_cast<const uint16_t *>(w.data) + r * c;
            for (int64_t i = 0; i < c; ++i) out[i] = bf16_to_f32(p[i]);
            break;
        }
        case WFmt::INT8: {
            const int8_t *p = static_cast<const int8_t *>(w.data) + r * c;
            for (int64_t i = 0; i < c; ++i) out[i] = static_cast<float>(p[i]) * w.scale[r];
            break;
        }
        case WFmt::FP8: {
            const uint8_t *p = static_cast<const uint8_t *>(w.data) + r * c;
            const float *t = sk_fp8_e4m3_table();
            for (int64_t i = 0; i < c; ++i) out[i] = t[p[i]] * w.scale[r];
            break;
        }
        case WFmt::FP8B: {
            const uint8_t *p = static_cast<const uint8_t *>(w.data) + r * c;
            const float *t = sk_fp8_e4m3_table();
            for (int64_t i = 0; i < c; ++i) out[i] = t[p[i]] * w.scale[fp8b_scale_index(w.rows, r, i)];
            break;
        }
        case WFmt::INT4G: {
            const uint8_t *p = static_cast<const uint8_t *>(w.data);
            for (int64_t i = 0; i < c; ++i)
                out[i] = static_cast<float>(int4g_value(p, w.rows, c, r, i, w.layout)) * w.scale[int4g_scale_index(w.rows, r, i)];
            break;
        }
    }
}

// Output rows [n0, n1) of any weight format, through the batch-invariant SIMD dots.
void gemm_range(const Weight &w, const float *x, int64_t M, int64_t n0, int64_t n1, float *y, int64_t ldy) {
    const int64_t K = w.cols;
    switch (w.fmt) {
        case WFmt::F32: sk_cpu_gemm_f32(x, M, K, static_cast<const float *>(w.data), n0, n1, y, ldy); break;
        case WFmt::BF16: sk_cpu_gemm_bf16(x, M, K, static_cast<const uint16_t *>(w.data), n0, n1, y, ldy); break;
        case WFmt::INT8:
            sk_cpu_gemm_i8(x, M, K, static_cast<const int8_t *>(w.data), w.scale, n0, n1, y, ldy);
            break;
        case WFmt::FP8:
            sk_cpu_gemm_fp8(x, M, K, static_cast<const uint8_t *>(w.data), w.scale, n0, n1, y, ldy);
            break;
        case WFmt::INT4G:
            if (w.layout == kInt4CpuSimd) {
                sk_cpu_gemm_i4s(x, M, K, static_cast<const uint8_t *>(w.data), w.scale, w.rows, n0, n1, y, ldy);
                break;
            }
            [[fallthrough]];
        case WFmt::FP8B: {
            // each row decoded with its scales (scalar), then the same SIMD dots as every format
            Weight wc = w;
            sk_cpu_gemm_rows(
                x, M, K, n0, n1, y, ldy,
                [](void *ctx, int64_t n, float *out) { weight_row(*static_cast<const Weight *>(ctx), n, out); }, &wc);
            break;
        }
    }
}

class CpuBackend final : public Backend {
public:
    explicit CpuBackend(int threads, GemmHook hook = nullptr, const char *name = "cpu")
        : pool_(threads), hook_(hook), name_(name) {}

    const char *name() const override { return name_; }
    // int4 matrices in the packing dec_i4s_* reads (int4_group.h); odd widths row-major (scalar)
    int32_t weight_layout(WFmt fmt, int64_t, int64_t cols) const override {
        return fmt == WFmt::INT4G ? (cols % 64 == 0 ? kInt4CpuSimd : kInt4RowMajor) : 0;
    }
    bool on_host() const override { return true; }
    Caps caps() const override {
        Caps c;
        c.memory = MemKind::Host;
        // every format, dense or expert (fp8b and int4 decoded per row); fp32 and bf16 KV
        c.dense_formats = c.expert_formats = fmt_bit(WFmt::F32) | fmt_bit(WFmt::BF16) | fmt_bit(WFmt::INT8) | fmt_bit(WFmt::FP8) | fmt_bit(WFmt::FP8B) | fmt_bit(WFmt::INT4G);
        c.kv_formats = kv_bit(DType::F32) | kv_bit(DType::BF16);
        c.families = FAM_ALL;
        return c;
    }

    void *alloc(size_t bytes) override {
        void *p = nullptr;
        if (posix_memalign(&p, 64, bytes ? bytes : 64) != 0 || !p)
            throw std::runtime_error("cpu backend: cannot allocate " + std::to_string(bytes) + " bytes");
        return p;
    }
    void release(void *p) override { std::free(p); }
    void *host_alloc(size_t bytes) override { return alloc(bytes); }
    void host_release(void *p) override { std::free(p); }
    void upload(void *dst, const void *src, size_t n) override { if (n) std::memcpy(dst, src, n); }
    void download(void *dst, const void *src, size_t n) override { if (n) std::memcpy(dst, src, n); }
    void copy(void *dst, const void *src, size_t n) override { if (n) std::memmove(dst, src, n); }
    void zero(void *p, size_t n) override { if (n) std::memset(p, 0, n); }
    void sync() override {}
    void *record_event() override { return nullptr; }
    void wait_event(void *) override {}
    void release_event(void *) override {}
    size_t free_memory() const override { return SIZE_MAX; }

    // ------------------------------------------------------------------ dense
    // A step of up to kSmallRows rows is memory-bound (a GEMV per matrix): on a hybrid CPU its ops
    // run on the performance cores only (ThreadPool, perf_only). The GEMM sees the step's rows
    // first in every layer and sets small_rows_ for the ops that follow it (norms, attention,
    // elementwise), which do not all know the step's rows.
    static constexpr int64_t kSmallRows = 16;
    void gemm(const Weight &w, const float *x, int64_t M, float *y, int64_t ldy) override {
        const int64_t N = w.rows, K = w.cols;
        if (M <= 0 || N <= 0) return;
        if (hook_ && hook_(w, x, M, y, ldy)) return;
        small_rows_ = M <= kSmallRows;
        const int64_t threads = pool_.size(small_rows_);
        const int64_t step = std::max<int64_t>(16, (N + threads * 4 - 1) / (threads * 4));
        const int64_t tasks = (N + step - 1) / step;
        (void)K;
        pool_.parallel_for(
            tasks,
            [&](int64_t i) {
                const int64_t n0 = i * step, n1 = std::min(N, n0 + step);
                gemm_range(w, x, M, n0, n1, y, ldy);
            },
            small_rows_);
    }

    // ------------------------------------------------------------------ elementwise
    void rmsnorm(float *y, const float *x, const float *w, int64_t rows, int64_t n, int64_t stride,
                 float eps) override {
        rows_for(rows, [&](int64_t r) {
            const float *xr = x + r * stride;
            float *yr = y + r * stride;
            double ss = 0;
            for (int64_t i = 0; i < n; ++i) ss += static_cast<double>(xr[i]) * xr[i];
            const float inv = static_cast<float>(1.0 / std::sqrt(ss / n + eps));
            for (int64_t i = 0; i < n; ++i) yr[i] = w[i] * (xr[i] * inv);
        });
    }

    void l2norm_heads(float *x, int64_t rows, int64_t heads, int64_t d, float eps) override {
        rows_for(rows * heads, [&](int64_t i) {
            float *v = x + i * d;
            double ss = 0;
            for (int64_t j = 0; j < d; ++j) ss += static_cast<double>(v[j]) * v[j];
            const float inv = static_cast<float>(1.0 / std::sqrt(ss + eps));
            for (int64_t j = 0; j < d; ++j) v[j] *= inv;
        });
    }

    void situ_glu(float *y, const float *gate, const float *up, int64_t n, float b1, float b2) override {
        elems_for(n, [&](int64_t i) {
            const float g = gate[i];
            const float a = b1 * std::tanh(g / b1) * sigm(g);
            y[i] = a * (b2 * std::tanh(up[i] / b2));
        });
    }

    void silu_glu(float *y, const float *gate, const float *up, int64_t n) override {
        elems_for(n, [&](int64_t i) {
            const float g = gate[i];
            y[i] = g * sigm(g) * up[i];
        });
    }

    void add(float *y, const float *x, int64_t n) override {
        elems_for(n, [&](int64_t i) { y[i] += x[i]; });
    }

    void sigmoid_mul(float *y, const float *g, int64_t n) override {
        elems_for(n, [&](int64_t i) { y[i] *= sigm(g[i]); });
    }

    void copy_rows(float *dst, int64_t dst_stride, int64_t dst_off, const float *src, int64_t rows,
                   int64_t n) override {
        for (int64_t r = 0; r < rows; ++r) std::memcpy(dst + r * dst_stride + dst_off, src + r * n, n * 4);
    }

    void attn_res(float *out, const float *blocks, int64_t nb, int64_t maxs, const float *pref,
                  const float *fold, int64_t rows, int64_t n, float eps) override {
        rows_for(rows, [&](int64_t r) {
            const int64_t ns = nb + 1;
            std::vector<const float *> src(ns);
            for (int64_t b = 0; b < nb; ++b) src[b] = blocks + (r * maxs + b) * n;
            src[nb] = pref + r * n;
            std::vector<float> score(ns);
            for (int64_t s = 0; s < ns; ++s) {
                const float *v = src[s];
                double ss = 0;
                for (int64_t i = 0; i < n; ++i) ss += static_cast<double>(v[i]) * v[i];
                const float inv = static_cast<float>(1.0 / std::sqrt(ss / n + eps));
                double acc = 0;
                for (int64_t i = 0; i < n; ++i) acc += static_cast<double>((v[i] * inv) * fold[i]);
                score[s] = static_cast<float>(acc);
            }
            float m = -INFINITY;
            for (float s : score) m = std::max(m, s);
            double z = 0;
            for (float &s : score) {
                s = std::exp(s - m);
                z += s;
            }
            float *o = out + r * n;
            for (int64_t i = 0; i < n; ++i) o[i] = 0.0f;
            for (int64_t s = 0; s < ns; ++s) {
                const float p = static_cast<float>(score[s] / z);
                for (int64_t i = 0; i < n; ++i) o[i] += p * src[s][i];
            }
        });
    }

    // ------------------------------------------------------------------ KDA
    void kda_conv(float *y, const float *x, const float *w, float *conv_state, int64_t conv_stride,
                  int64_t hist_off, const int32_t *seg, int64_t nseg, int64_t C, int64_t K) override {
        const int64_t hist = K - 1;
        for (int64_t s = 0; s < nseg; ++s) {
            const int64_t slot = seg[3 * s], off = seg[3 * s + 1], len = seg[3 * s + 2];
            float *st = conv_state + slot * conv_stride + hist_off;
            rows_for(C, [&](int64_t c) {
                float buf[64];
                float *h = st + c * hist;
                for (int64_t j = 0; j < hist; ++j) buf[j] = h[j];
                const float *wc = w + c * K;
                for (int64_t t = 0; t < len; ++t) {
                    const float cur = x[(off + t) * C + c];
                    float acc = wc[hist] * cur;
                    for (int64_t j = 0; j < hist; ++j) acc += wc[j] * buf[j];
                    for (int64_t j = 0; j + 1 < hist; ++j) buf[j] = buf[j + 1];
                    if (hist > 0) buf[hist - 1] = cur;
                    y[(off + t) * C + c] = acc * sigm(acc);
                }
                for (int64_t j = 0; j < hist; ++j) h[j] = buf[j];
            });
        }
    }

    void kda_gates(float *alpha, float *beta, const float *z, const float *braw, const float *a_log,
                   const float *dt_bias, int64_t N, int64_t H, int64_t D, float lb, bool softplus) override {
        rows_for(N, [&](int64_t t) {
            for (int64_t h = 0; h < H; ++h) {
                const float a = std::exp(a_log[h]);
                for (int64_t d = 0; d < D; ++d) {
                    const int64_t i = h * D + d;
                    const float zz = z[t * H * D + i] + dt_bias[i];
                    const float g = softplus ? -a * softplus_(zz) : lb * sigm(a * zz);
                    alpha[t * H * D + i] = std::exp(g);
                }
                beta[t * H + h] = sigm(braw[t * H + h]);
            }
        });
    }

    void kda_recurrence(float *o, const float *q, const float *k, const float *v, const float *alpha,
                        const float *beta, void *state, DType sdt, int64_t state_stride,
                        const int32_t *seg, int64_t nseg, int64_t H, int64_t D, float qscale) override {
        const int64_t P = H * D;
        if (D > 512) throw std::runtime_error("kda_recurrence: head_dim above 512");
        pool_.parallel_for(nseg * H, [&](int64_t task) {
            const int64_t s = task / H, h = task % H;
            const int64_t slot = seg[3 * s], off = seg[3 * s + 1], len = seg[3 * s + 2];
            std::vector<float> S(D * D);
            const int64_t base = slot * state_stride + h * D * D;
            if (sdt == DType::F32) {
                std::memcpy(S.data(), static_cast<float *>(state) + base, D * D * 4);
            } else {
                const uint16_t *p = static_cast<uint16_t *>(state) + base;
                for (int64_t i = 0; i < D * D; ++i) S[i] = bf16_to_f32(p[i]);
            }
            float u[512], qs[512];
            for (int64_t t = off; t < off + len; ++t) {
                const float *qt = q + t * P + h * D, *kt = k + t * P + h * D, *vt = v + t * P + h * D;
                const float *at = alpha + t * P + h * D;
                const float b = beta[t * H + h];
                for (int64_t j = 0; j < D; ++j) u[j] = 0.0f;
                for (int64_t i = 0; i < D; ++i) {  // decay, then read along k
                    float *row = S.data() + i * D;
                    for (int64_t j = 0; j < D; ++j) {
                        row[j] *= at[i];
                        u[j] += kt[i] * row[j];
                    }
                }
                for (int64_t j = 0; j < D; ++j) u[j] = b * (vt[j] - u[j]);
                for (int64_t i = 0; i < D; ++i) qs[i] = qt[i] * qscale;
                float *ot = o + t * P + h * D;
                for (int64_t j = 0; j < D; ++j) ot[j] = 0.0f;
                for (int64_t i = 0; i < D; ++i) {  // delta write, output from the UPDATED state
                    float *row = S.data() + i * D;
                    for (int64_t j = 0; j < D; ++j) {
                        row[j] += kt[i] * u[j];
                        ot[j] += qs[i] * row[j];
                    }
                }
            }
            if (sdt == DType::F32) {
                std::memcpy(static_cast<float *>(state) + base, S.data(), D * D * 4);
            } else {
                uint16_t *p = static_cast<uint16_t *>(state) + base;
                for (int64_t i = 0; i < D * D; ++i) p[i] = f32_to_bf16(S[i]);
            }
        }, small_rows_);
    }

    void kda_out(float *o, const float *gate, const float *o_norm, int64_t N, int64_t H, int64_t D,
                 float eps) override {
        rows_for(N * H, [&](int64_t i) {
            float *v = o + i * D;
            const float *g = gate + i * D;
            double ss = 0;
            for (int64_t j = 0; j < D; ++j) ss += static_cast<double>(v[j]) * v[j];
            const float inv = static_cast<float>(1.0 / std::sqrt(ss / D + eps));
            for (int64_t j = 0; j < D; ++j) v[j] = o_norm[j] * (v[j] * inv) * sigm(g[j]);
        });
    }

    // ------------------------------------------------------------------ MLA
    void mla_append(void *kv, DType kdt, const float *ckv, const int32_t *tok_slot, const int32_t *tok_pos,
                    const int32_t *page_table, int64_t max_pages, int64_t page, int64_t width,
                    int64_t N) override {
        for (int64_t t = 0; t < N; ++t) {
            const int64_t at = row_of(tok_slot[t], tok_pos[t], page_table, max_pages, page) * width;
            if (kdt == DType::F32) {
                std::memcpy(static_cast<float *>(kv) + at, ckv + t * width, width * 4);
            } else {
                uint16_t *p = static_cast<uint16_t *>(kv) + at;
                for (int64_t i = 0; i < width; ++i) p[i] = f32_to_bf16(ckv[t * width + i]);
            }
        }
    }

    void mla_absorb_q(float *q_abs, const float *q, const Weight &kv_b, int64_t N, int64_t H, int64_t qn,
                      int64_t qr, int64_t vh) override {
        const int64_t lat = kv_b.cols, qh = qn + qr, wa = lat + qr;
        pool_.parallel_for(N * H, [&](int64_t i) {
            const int64_t t = i / H, h = i % H;
            const float *qt = q + (t * H + h) * qh;
            float *out = q_abs + (t * H + h) * wa;
            std::vector<float> row(lat);
            for (int64_t j = 0; j < lat; ++j) out[j] = 0.0f;
            for (int64_t r = 0; r < qn; ++r) {
                weight_row(kv_b, h * (qn + vh) + r, row.data());
                for (int64_t j = 0; j < lat; ++j) out[j] += qt[r] * row[j];
            }
            for (int64_t j = 0; j < qr; ++j) out[lat + j] = qt[qn + j];
        }, small_rows_);
    }

    void mla_attention(float *o_lat, const float *q_abs, const void *kv, DType kdt, const int32_t *tok_slot,
                       const int32_t *tok_pos, const int32_t *page_table, int64_t max_pages, int64_t page,
                       int64_t N, int64_t H, int64_t lat, int64_t qr, int64_t /*max_pos*/,
                       float scale) override {
        const int64_t width = lat + qr;
        pool_.parallel_for(N * H, [&](int64_t i) {
            const int64_t t = i / H, h = i % H;
            const int64_t pos = tok_pos[t];
            const float *qa = q_abs + (t * H + h) * width;
            std::vector<float> sc(pos + 1), ent(width);
            float m = -INFINITY;
            for (int64_t p = 0; p <= pos; ++p) {
                load_row(kv, kdt, row_of(tok_slot[t], p, page_table, max_pages, page) * width, width, ent.data());
                float d = 0.0f;
                for (int64_t j = 0; j < lat; ++j) d += qa[j] * ent[j];
                for (int64_t j = 0; j < qr; ++j) d += qa[lat + j] * ent[lat + j];
                sc[p] = d * scale;
                m = std::max(m, sc[p]);
            }
            double z = 0;
            for (auto &s : sc) {
                s = std::exp(s - m);
                z += s;
            }
            float *o = o_lat + (t * H + h) * lat;
            for (int64_t j = 0; j < lat; ++j) o[j] = 0.0f;
            for (int64_t p = 0; p <= pos; ++p) {
                const float pr = static_cast<float>(sc[p] / z);
                load_row(kv, kdt, row_of(tok_slot[t], p, page_table, max_pages, page) * width, lat, ent.data());
                for (int64_t j = 0; j < lat; ++j) o[j] += pr * ent[j];
            }
        }, small_rows_);
    }

    void rope(float *x, const int32_t *tok_pos, const float *inv_freq, int64_t N, int64_t heads, int64_t D,
              int64_t stride) override {
        const int64_t half = D / 2;
        rows_for(N, [&](int64_t t) {
            const float pos = static_cast<float>(tok_pos[t]);
            for (int64_t i = 0; i < half; ++i) {
                const float ang = pos * inv_freq[i];
                const float c = std::cos(ang), sn = std::sin(ang);
                for (int64_t h = 0; h < heads; ++h) {
                    float *v = x + t * stride + h * D;
                    const float a = v[i], b = v[i + half];
                    v[i] = a * c - b * sn;
                    v[i + half] = b * c + a * sn;
                }
            }
        });
    }

    void gqa_attention(float *o, const float *q, const void *kc, const void *vc, DType kdt, const int32_t *tok_slot,
                       const int32_t *tok_pos, const int32_t *page_table, int64_t max_pages, int64_t page, int64_t N,
                       int64_t Hq, int64_t Hkv, int64_t D, int64_t /*max_pos*/, float scale) override {
        const int64_t W = Hkv * D, group = Hq / Hkv;
        pool_.parallel_for(N * Hq, [&](int64_t i) {
            const int64_t t = i / Hq, h = i % Hq, kh = h / group;
            const int64_t pos = tok_pos[t];
            const float *qv = q + (t * Hq + h) * D;
            std::vector<float> sc(pos + 1), ent(D);
            float m = -INFINITY;
            for (int64_t p = 0; p <= pos; ++p) {
                load_row(kc, kdt, row_of(tok_slot[t], p, page_table, max_pages, page) * W + kh * D, D, ent.data());
                float d = 0.0f;
                for (int64_t j = 0; j < D; ++j) d += qv[j] * ent[j];
                sc[p] = d * scale;
                m = std::max(m, sc[p]);
            }
            double z = 0;
            for (auto &v : sc) {
                v = std::exp(v - m);
                z += v;
            }
            float *ov = o + (t * Hq + h) * D;
            for (int64_t j = 0; j < D; ++j) ov[j] = 0.0f;
            for (int64_t p = 0; p <= pos; ++p) {
                const float pr = static_cast<float>(sc[p] / z);
                load_row(vc, kdt, row_of(tok_slot[t], p, page_table, max_pages, page) * W + kh * D, D, ent.data());
                for (int64_t j = 0; j < D; ++j) ov[j] += pr * ent[j];
            }
        }, small_rows_);
    }

    void mla_unabsorb(float *y, const float *o_lat, const Weight &kv_b, int64_t N, int64_t H, int64_t qn,
                      int64_t vh) override {
        const int64_t lat = kv_b.cols;
        pool_.parallel_for(N * H, [&](int64_t i) {
            const int64_t t = i / H, h = i % H;
            const float *ol = o_lat + (t * H + h) * lat;
            std::vector<float> row(lat);
            for (int64_t j = 0; j < vh; ++j) {
                weight_row(kv_b, h * (qn + vh) + qn + j, row.data());
                float acc = 0.0f;
                for (int64_t l = 0; l < lat; ++l) acc += row[l] * ol[l];
                y[t * H * vh + h * vh + j] = acc;
            }
        }, small_rows_);
    }

    // ------------------------------------------------------------------ router
    void router_softmax_topk(int32_t *idx, float *w, const float *logits, int64_t N, int64_t E, int64_t k,
                             bool renorm) override {
        rows_for(N, [&](int64_t t) {
            const float *l = logits + t * E;
            float m = -INFINITY;
            for (int64_t e = 0; e < E; ++e) m = std::max(m, l[e]);
            std::vector<float> p(E);
            float z = 0.0f;
            for (int64_t e = 0; e < E; ++e) z += (p[e] = std::exp(l[e] - m));
            for (int64_t e = 0; e < E; ++e) p[e] /= z;
            float sum = 0.0f;
            for (int64_t j = 0; j < k; ++j) {
                int64_t best = -1;
                for (int64_t e = 0; e < E; ++e)
                    if (p[e] >= 0.0f && (best < 0 || p[e] > p[best])) best = e;
                idx[t * k + j] = static_cast<int32_t>(best);
                w[t * k + j] = p[best];
                sum += p[best];
                p[best] = -1.0f;  // taken
            }
            if (renorm)
                for (int64_t j = 0; j < k; ++j) w[t * k + j] /= sum;
        });
    }

    void embed_rows(float *dst, const Weight &table, const int32_t *ids, int64_t N, int64_t n) override {
        for (int64_t t = 0; t < N; ++t) {
            float *d = dst + t * n;
            const int64_t row = int64_t(ids[t]) * n;
            if (table.fmt == WFmt::F32) {
                std::memcpy(d, static_cast<const float *>(table.data) + row, size_t(n) * 4);
            } else if (table.fmt == WFmt::BF16) {
                const uint16_t *p = static_cast<const uint16_t *>(table.data) + row;
                for (int64_t i = 0; i < n; ++i) d[i] = bf16_to_f32(p[i]);
            } else {
                throw std::runtime_error("embed_rows: the table must be bf16 or f32");
            }
        }
    }
    void gather_ids(int32_t *dst, const int32_t *table, const int32_t *idx, int64_t n) override {
        for (int64_t r = 0; r < n; ++r) dst[r] = table[idx[r]];
    }
    void argmax_rows(int32_t *idx, int32_t *bad, const float *x, int64_t rows, int64_t n) override {
        for (int64_t r = 0; r < rows; ++r) {
            const float *v = x + r * n;
            int64_t best = -1;
            for (int64_t i = 0; i < n; ++i) {
                if (!std::isfinite(v[i])) *bad = 1;
                if (!std::isnan(v[i]) && (best < 0 || v[i] > v[best])) best = i;
            }
            idx[r] = static_cast<int32_t>(best < 0 ? 0 : best);
        }
    }
    void router_topk(int32_t *idx, float *w, const float *logits, const float *bias, int64_t N, int64_t E,
                     int64_t k, bool renorm, float scale, bool biased) override {
        rows_for(N, [&](int64_t t) {
            std::vector<float> score(E), choice(E);
            for (int64_t e = 0; e < E; ++e) {
                score[e] = sigm(logits[t * E + e]);
                choice[e] = score[e] + (bias ? bias[e] : 0.0f);
            }
            float sum = 0.0f;
            for (int64_t j = 0; j < k; ++j) {
                int64_t best = 0;
                float bv = -INFINITY;
                for (int64_t e = 0; e < E; ++e)
                    if (choice[e] > bv) { bv = choice[e]; best = e; }
                idx[t * k + j] = static_cast<int32_t>(best);
                const float wt = biased ? score[best] + (bias ? bias[best] : 0.0f) : score[best];
                w[t * k + j] = wt;
                sum += wt;
                choice[best] = -INFINITY;
            }
            if (renorm) {
                const float s = sum + 1e-20f;
                for (int64_t j = 0; j < k; ++j) w[t * k + j] /= s;
            }
            for (int64_t j = 0; j < k; ++j) w[t * k + j] *= scale;
        });
    }

    // ------------------------------------------------------------------ resident experts
    // FP8B experts are stored like the GPU stores them (gate/up rows interleaved in w1, block
    // scales over that matrix), so both backends compute from the same quantised bytes
    bool moe_gate_up_interleaved(WFmt fmt, int64_t, int64_t) const override {
        return fmt == WFmt::FP8B || fmt == WFmt::INT4G;
    }
    // Prefill steps run int4 experts as they are: the decoded-row GEMM takes any count of rows, so
    // the executor keeps no fp8b copies for them (half the memory of the copies, and the CPU's
    // fp8b row decode is scalar).
    bool int4_prefill_gemm(int64_t, int64_t) const override { return true; }

    // The pairs bucketed by expert and each expert's tokens as one GEMM (M = its pairs), so a
    // prefill step runs the blocked GEMM over the expert's rows instead of a GEMV per pair (a dense
    // MLP split into int4 experts). Tasks:
    // (expert, 128-row chunk of inter) for gate/up with the GLU, then (expert, 128-row chunk of
    // width) for down with the routing weight, each over all the expert's tokens. The dots run in
    // the same k order whatever the batch (gemm_decoded): a pair's rows do not depend on it.
    void moe_experts(float *contrib, const float *x, const int32_t *idx, const float *w, int64_t N, int64_t k,
                     const ExpertW *table, int64_t n_experts, WFmt, int64_t width, int64_t inter, bool silu, float b1,
                     float b2, float *scratch) override {
        const int64_t pairs = N * k, chunk = 128;
        // counting sort of the resident pairs by expert, stable (a token's pairs stay in order)
        std::vector<int64_t> start(static_cast<size_t>(n_experts) + 1, 0), order(static_cast<size_t>(pairs));
        auto resident = [&](int64_t p) { return idx[p] >= 0 && idx[p] < n_experts && table[idx[p]].w1.data; };
        for (int64_t p = 0; p < pairs; ++p)
            if (resident(p)) ++start[static_cast<size_t>(idx[p]) + 1];
        for (int64_t e = 0; e < n_experts; ++e) start[static_cast<size_t>(e) + 1] += start[static_cast<size_t>(e)];
        std::vector<int64_t> cursor(start.begin(), start.end() - 1);
        for (int64_t p = 0; p < pairs; ++p)
            if (resident(p)) order[static_cast<size_t>(cursor[static_cast<size_t>(idx[p])]++)] = p;
        const int64_t used = start[static_cast<size_t>(n_experts)];
        if (used == 0) return;
        // x rows per expert: x itself when the expert has every token (the sort keeps them in order),
        // else its tokens' rows gathered
        bool gather = false;
        for (int64_t e = 0; e < n_experts && !gather; ++e) gather = start[e + 1] - start[e] != 0 && start[e + 1] - start[e] != N;
        if (gather) {
            if (static_cast<int64_t>(xg_.size()) < used * width) xg_.resize(static_cast<size_t>(used * width));
            rows_for(used, [&](int64_t i) {
                std::memcpy(xg_.data() + i * width, x + (order[static_cast<size_t>(i)] / k) * width, static_cast<size_t>(width) * 4);
            });
        }
        auto x_of = [&](int64_t e) { return gather ? xg_.data() + start[e] * width : x; };
        // y of a chunk's GEMM: gemm_range writes y[m * ldy + n] for its columns n in [n0, n1), so a
        // [c][cw] result goes in a buffer of c * cw + n0 floats from its start (ldy = cw): row m's
        // columns at m * cw + n, never a pointer before the buffer
        const int64_t ti = (inter + chunk - 1) / chunk, tw = (width + chunk - 1) / chunk;
        pool_.parallel_for(n_experts * ti, [&](int64_t task) {
            const int64_t e = task / ti, n0 = (task % ti) * chunk, n1 = std::min(inter, n0 + chunk), cw = n1 - n0;
            const int64_t c = start[e + 1] - start[e];
            if (c == 0) return;
            const ExpertW &ex = table[e];
            const float *xe = x_of(e);
            thread_local std::vector<float> g, u;
            if (!ex.w3.data) {  // interleaved: w1 row 2n is gate n, row 2n + 1 up n
                g.resize(static_cast<size_t>(c * 2 * cw + 2 * n0));
                gemm_range(ex.w1, xe, c, 2 * n0, 2 * n1, g.data(), 2 * cw);
            } else {
                g.resize(static_cast<size_t>(c * cw + n0));
                u.resize(static_cast<size_t>(c * cw + n0));
                gemm_range(ex.w1, xe, c, n0, n1, g.data(), cw);
                gemm_range(ex.w3, xe, c, n0, n1, u.data(), cw);
            }
            for (int64_t m = 0; m < c; ++m) {
                float *a = scratch + (start[e] + m) * inter;
                for (int64_t n = n0; n < n1; ++n) {
                    const float gv = ex.w3.data ? g[static_cast<size_t>(m * cw + n)] : g[static_cast<size_t>(m * 2 * cw + 2 * n)];
                    const float uv = ex.w3.data ? u[static_cast<size_t>(m * cw + n)] : g[static_cast<size_t>(m * 2 * cw + 2 * n + 1)];
                    a[n] = silu ? gv * sigm(gv) * uv : b1 * std::tanh(gv / b1) * sigm(gv) * (b2 * std::tanh(uv / b2));
                }
            }
        }, small_rows_);
        // down rows, weighted
        pool_.parallel_for(n_experts * tw, [&](int64_t task) {
            const int64_t e = task / tw, n0 = (task % tw) * chunk, n1 = std::min(width, n0 + chunk), cw = n1 - n0;
            const int64_t c = start[e + 1] - start[e];
            if (c == 0) return;
            thread_local std::vector<float> d;
            d.resize(static_cast<size_t>(c * cw + n0));
            gemm_range(table[e].w2, scratch + start[e] * inter, c, n0, n1, d.data(), cw);
            for (int64_t m = 0; m < c; ++m) {
                const int64_t p = order[static_cast<size_t>(start[e] + m)];
                for (int64_t n = n0; n < n1; ++n) contrib[p * width + n] = w[p] * d[static_cast<size_t>(m * cw + n)];
            }
        }, small_rows_);
    }

    void moe_reduce(float *acc, const float *contrib, int64_t N, int64_t k, int64_t width,
                    const float *addend) override {
        rows_for(N, [&](int64_t t) {
            float *a = acc + t * width;
            for (int64_t i = 0; i < width; ++i) a[i] = 0.0f;
            for (int64_t j = 0; j < k; ++j) {
                const float *c = contrib + (t * k + j) * width;
                for (int64_t i = 0; i < width; ++i) a[i] = a[i] + c[i];
            }
            if (addend)
                for (int64_t i = 0; i < width; ++i) a[i] = a[i] + addend[t * width + i];
        });
    }

    void scatter_rows(float *dst, const float *src, const int32_t *rows, int64_t n, int64_t width) override {
        for (int64_t i = 0; i < n; ++i) std::memcpy(dst + int64_t(rows[i]) * width, src + i * width, width * 4);
    }

private:
    template <typename F>
    void rows_for(int64_t rows, F &&f) {
        const std::function<void(int64_t)> fn = f;
        pool_.parallel_for(rows, fn, small_rows_);
    }

    template <typename F>
    void elems_for(int64_t n, F &&f) {
        const int64_t chunk = 1 << 14;
        const int64_t tasks = (n + chunk - 1) / chunk;
        pool_.parallel_for(tasks, [&](int64_t i) {
            const int64_t a = i * chunk, b = std::min(n, a + chunk);
            for (int64_t j = a; j < b; ++j) f(j);
        });
    }

    static int64_t row_of(int64_t slot, int64_t pos, const int32_t *pt, int64_t max_pages, int64_t page) {
        return static_cast<int64_t>(pt[slot * max_pages + pos / page]) * page + pos % page;
    }

    static void load_row(const void *kv, DType kdt, int64_t at, int64_t n, float *out) {
        if (kdt == DType::F32) {
            std::memcpy(out, static_cast<const float *>(kv) + at, n * 4);
        } else {
            const uint16_t *p = static_cast<const uint16_t *>(kv) + at;
            for (int64_t i = 0; i < n; ++i) out[i] = bf16_to_f32(p[i]);
        }
    }

    ThreadPool pool_;
    bool small_rows_ = false;  // the current step has <= kSmallRows rows (set by gemm)
    std::vector<float> xg_;    // moe_experts: the experts' token rows gathered
    GemmHook hook_ = nullptr;
    const char *name_ = "cpu";
};

}  // namespace

std::unique_ptr<Backend> make_cpu_backend(int threads) { return std::make_unique<CpuBackend>(threads); }

std::unique_ptr<Backend> make_cpu_backend_with_gemm(int threads, GemmHook gemm, const char *name) {
    return std::make_unique<CpuBackend>(threads, gemm, name);
}

#ifndef FULLM_TORCH
std::unique_ptr<Backend> make_torch_backend(int) {
    throw std::runtime_error("this build has no torch backend (build with --features torch and FULLM_TORCH_DIR)");
}
#endif

}  // namespace fullm
