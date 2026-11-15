// SPDX-License-Identifier: Apache-2.0
// core_backend.cpp - "cpu-core": the CPU backend cut down to what a new device implements first:
// the runtime, the core ops and the GQA family, nothing else. Every fast path
// (gemm_group, add_rmsnorm, qk_rope_append, gqa_prefill, the fused router, moe_experts, ...) falls
// to Backend's base form or to the executor's own composition. Dense and MoE models run on it;
// the tests compare it with the full CPU backend, which checks those base forms and that a
// backend with only the core runs the models.

#include "backend.h"
#include "int4_group.h"

namespace fullm {

namespace {

class CoreBackend final : public Backend {
public:
    explicit CoreBackend(int threads) : c_(make_cpu_backend(threads)) {}

    const char *name() const override { return "cpu-core"; }
    bool on_host() const override { return true; }
    Caps caps() const override {
        Caps c = c_->caps();
        c.families = FAM_GQA;  // the core and the dense models' attention only
        c.expert_formats = c.dense_formats;  // moe_experts' base form computes with gemm
        return c;
    }
    void set_exact_activations(bool exact) override { c_->set_exact_activations(exact); }
    // int4 experts row-major (the CPU decoder reads either packing, by the matrix's layout): a
    // backend picking its own layout, against the full backend's MMA fragments in the tests
    int32_t weight_layout(WFmt fmt, int64_t, int64_t) const override { return fmt == WFmt::INT4G ? kInt4RowMajor : 0; }

    // the runtime
    void *alloc(size_t bytes) override { return c_->alloc(bytes); }
    void release(void *p) override { c_->release(p); }
    void *host_alloc(size_t bytes) override { return c_->host_alloc(bytes); }
    void host_release(void *p) override { c_->host_release(p); }
    void upload(void *dst, const void *src, size_t n) override { c_->upload(dst, src, n); }
    void download(void *dst, const void *src, size_t n) override { c_->download(dst, src, n); }
    void copy(void *dst, const void *src, size_t n) override { c_->copy(dst, src, n); }
    void zero(void *p, size_t n) override { c_->zero(p, n); }
    void sync() override { c_->sync(); }
    void *record_event() override { return c_->record_event(); }
    void wait_event(void *ev) override { c_->wait_event(ev); }
    void release_event(void *ev) override { c_->release_event(ev); }
    size_t free_memory() const override { return c_->free_memory(); }

    // the core ops
    void gemm(const Weight &w, const float *x, int64_t M, float *y, int64_t ldy) override { c_->gemm(w, x, M, y, ldy); }
    void rmsnorm(float *y, const float *x, const float *w, int64_t rows, int64_t n, int64_t stride, float eps) override {
        c_->rmsnorm(y, x, w, rows, n, stride, eps);
    }
    void silu_glu(float *y, const float *g, const float *u, int64_t n) override { c_->silu_glu(y, g, u, n); }
    void add(float *y, const float *x, int64_t n) override { c_->add(y, x, n); }
    void copy_rows(float *dst, int64_t dst_stride, int64_t dst_off, const float *src, int64_t rows, int64_t n) override {
        c_->copy_rows(dst, dst_stride, dst_off, src, rows, n);
    }
    void mla_append(void *kv, DType dt, const float *ckv, const int32_t *slot, const int32_t *pos, const int32_t *pt,
                    int64_t max_pages, int64_t page, int64_t width, int64_t N) override {
        c_->mla_append(kv, dt, ckv, slot, pos, pt, max_pages, page, width, N);
    }
    void embed_rows(float *dst, const Weight &table, const int32_t *ids, int64_t N, int64_t n) override {
        c_->embed_rows(dst, table, ids, N, n);
    }
    void argmax_rows(int32_t *idx, int32_t *bad, const float *x, int64_t rows, int64_t n) override {
        c_->argmax_rows(idx, bad, x, rows, n);
    }
    void gather_ids(int32_t *dst, const int32_t *table, const int32_t *idx, int64_t n) override {
        c_->gather_ids(dst, table, idx, n);
    }
    void router_softmax_topk(int32_t *idx, float *w, const float *logits, int64_t N, int64_t E, int64_t k,
                             bool renorm) override {
        c_->router_softmax_topk(idx, w, logits, N, E, k, renorm);
    }
    void moe_reduce(float *acc, const float *contrib, int64_t N, int64_t k, int64_t width, const float *addend) override {
        c_->moe_reduce(acc, contrib, N, k, width, addend);
    }
    void scatter_rows(float *dst, const float *src, const int32_t *rows, int64_t n, int64_t width) override {
        c_->scatter_rows(dst, src, rows, n, width);
    }

    // the GQA family
    void rope(float *x, const int32_t *pos, const float *inv_freq, int64_t N, int64_t heads, int64_t D,
              int64_t stride) override {
        c_->rope(x, pos, inv_freq, N, heads, D, stride);
    }
    void gqa_attention(float *o, const float *q, const void *kc, const void *vc, DType dt, const int32_t *slot,
                       const int32_t *pos, const int32_t *pt, int64_t max_pages, int64_t page, int64_t N, int64_t Hq,
                       int64_t Hkv, int64_t D, int64_t max_pos, float scale) override {
        c_->gqa_attention(o, q, kc, vc, dt, slot, pos, pt, max_pages, page, N, Hq, Hkv, D, max_pos, scale);
    }

private:
    std::unique_ptr<Backend> c_;
};

}  // namespace

std::unique_ptr<Backend> make_cpu_core_backend(int threads) { return std::make_unique<CoreBackend>(threads); }

}  // namespace fullm
