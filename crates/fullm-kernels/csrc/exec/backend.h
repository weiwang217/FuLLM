// SPDX-License-Identifier: Apache-2.0
// backend.h - the device abstraction the C++ model graphs are written against.
//
// A model's forward pass is written ONCE, against this interface. The CPU backend runs it
// on host memory (and is how the graph is checked against the torch oracle on any
// machine); the CUDA backend runs it on an H100. The op-level self test compares the two
// op by op on random inputs, so a kernel bug is found at the kernel, not as a wrong token.
//
// CONVENTIONS
//   - "Backend memory" is host memory for the CPU backend and device memory for CUDA.
//     Every pointer an op takes is backend memory unless named host_*.
//   - Ops are asynchronous and ordered (one stream). Host-visible results need sync()
//     or an event wait.
//   - Activations are fp32, row-major [rows][width]. Weights are row-major [N][K].
//   - Index arrays (segments, token slots/positions, page tables, top-k ids) are int32.
//   - DETERMINISM: no op uses floating-point atomics; in calls of at most kGemvMaxRows rows
//     no reduction order depends on how many rows share the call. Above that, two CUDA ops
//     are run-to-run deterministic but not batch-invariant: gemm (cuBLAS, bf16 inputs) and
//     mla_attention (its split count is bounded by the partials buffer).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fullm {

// FP8B: e4m3 with one fp32 scale per 128 x 128 block (fp8_block.h), resident experts only
// INT4G: int4 with one fp32 scale per row per 128 columns, packed for the decode tiles
// (int4_group.h), resident experts only
enum class WFmt : int32_t { F32 = 0, BF16 = 1, INT8 = 2, FP8 = 3, FP8B = 4, INT4G = 5 };
// FP8: the dense GQA KV cache only (kv_fp8.h's layout, a power-of-two scale per token and kv
// head; CUDA backend with bf16 activations). FP8E4M3: the same cache as plain e4m3 pages,
// [page][Hkv][D] at one scale (1) for every token: FlashAttention-3's fp8 layout, which it reads
// for the prefill chunks and the decoding rows alike.
enum class DType : int32_t { F32 = 0, BF16 = 1, FP8 = 2, FP8E4M3 = 3 };

// Rows at or below this take the exact-fp32 GEMV path on every backend.
constexpr int64_t kGemvMaxRows = 16;

// What the device behind a backend can do. Rust's planner reads it before it builds an executor
// (sk_backend_caps), and the executors check requests against it rather than naming a backend.
enum class MemKind : int32_t { Host = 0, Discrete = 1, Unified = 2 };
struct Caps {
    MemKind memory = MemKind::Host;
    bool graphs = false;            // graph_begin/graph_end replay a decode step
    bool async_copy = false;        // copy_pages overlaps compute (copy engines)
    uint32_t dense_formats = 0;     // fmt_bit(f) per weight format gemm() takes
    uint32_t expert_formats = 0;    // ... per format moe_experts() takes for resident experts
    uint32_t kv_formats = 0;        // kv_bit(d) per KV cache layout the attention ops read
    int32_t kv8_page_multiple = 1;  // the fp8 KV layouts' pages hold a multiple of this many positions
    uint32_t families = 0;          // FAM_* bits: the model families' ops this backend implements
    // fp8 weights' GEMMs above the GEMV's rows (bf16 activations) may quantize x's rows to e4m3, a
    // scale each (amax / 448): W8A8, not the weights' bf16 copies (where, the backend's choice)
    bool fp8_w8a8 = false;
    // ... for small matrices (under 2^24 elements) at decode-sized steps (under 512 rows) too; else
    // those take the weights' bf16 copies (the CUDA backend on devices with memory to spare)
    bool fp8_w8a8_small = false;
};
// The ops of a model family: a backend without one runs no model that needs
// it, and the executor says so when it is built. GQA: rope, gqa_attention (dense models). MLA:
// mla_absorb_q, mla_attention, mla_unabsorb. KDA: kda_conv, kda_gates, kda_recurrence, kda_out,
// l2norm_heads, sigmoid_mul. KIMI: situ_glu, attn_res, router_topk (Kimi's MLP, residuals, router).
enum : uint32_t { FAM_GQA = 1, FAM_MLA = 2, FAM_KDA = 4, FAM_KIMI = 8, FAM_ALL = 15 };
constexpr uint32_t fmt_bit(WFmt f) { return 1u << static_cast<int>(f); }
constexpr uint32_t kv_bit(DType d) { return 1u << static_cast<int>(d); }

// The values the shared code tunes by, chosen per device; the environment variables named here
// still override them where they are read.
struct Policy {
    int64_t expert_tile_max_rows = 64;  // int4 experts: steps of up to this many rows take the decode tiles (FULLM_INT4_ROWS; CUDA 96)
    int64_t expert_gemm_min_rows = 64;  // int4 experts: calls of this many rows or more take the prefill GEMM (FULLM_INT4_GEMM_ROWS)
};

// A weight matrix in backend memory, row-major [rows][cols].
struct Weight {
    WFmt fmt = WFmt::F32;
    int64_t rows = 0, cols = 0;
    const void *data = nullptr;
    // INT8 and FP8: [rows], W[n][k] = q[n][k] * scale[n]. FP8B: [ceil(cols/128)][ceil(rows/128)],
    // W[n][k] = q[n][k] * scale[(k/128) * ceil(rows/128) + n/128] (fp8_block.h). INT4G:
    // [cols/128][rows], W[n][k] = q(n, k) * scale[(k/128) * rows + n] (int4_group.h).
    const float *scale = nullptr;
    // The packing within fmt, chosen by the backend that reads the matrix (weight_layout): for
    // INT4G kInt4Fragments (the MMA fragments where the shape allows) or kInt4RowMajor
    // (int4_group.h); 0 for every other format.
    int32_t layout = 0;
    size_t bytes() const;
};

// A row's sampling (sk_sample_row's layout; see fullm_exec.h for the draw).
struct SampleRow {
    float temperature;
    float top_p;
    int32_t top_k;
    int32_t mask;  // 0: none; k: the step's mask k - 1 (mask_rows)
    uint64_t seed;
    uint64_t step;
    float min_p_cut;         // < 0: min-p's cut below the row's largest logit; 0: off
    int32_t adj_off, adj_n;  // the row's logit adjustments (adjust_rows)
    int32_t logprobs;        // 0: none; k: the drawn token's and k - 1 alternatives' (logprob_rows)
};

// A token's logit adjustment (sk_logit_adj): x = x + bias; x = x > 0 ? x / rep : x * rep; x -= sub.
struct LogitAdj {
    int32_t token;
    float bias;
    float rep;
    float sub;
};

constexpr int kMaxTopLogprobs = 20;

// The draw for one row of n logits on the host: the definition every backend's sample_rows
// matches. Assumes finite logits (the executors refuse a step with any other).
int32_t sample_row_host(const float *x, int64_t n, const SampleRow &p);
// The adjustments of one row on the host (adjust_rows' definition).
void adjust_row_host(float *x, const SampleRow &p, const LogitAdj *adj);
// One row's log-probabilities (logprob_rows' definition): lp kMaxTopLogprobs + 1, ids kMaxTopLogprobs.
void logprob_row_host(float *lp, int32_t *ids, const float *x, int64_t n, int32_t picked, int32_t k);

class Backend {
public:
    virtual ~Backend() = default;
    virtual const char *name() const = 0;
    virtual bool on_host() const = 0;
    // The device's capabilities and tuning (Caps, Policy). caps() may depend on
    // set_exact_activations: CUDA's fp8 KV layouts need 16-bit activations.
    virtual Caps caps() const = 0;
    virtual Policy policy() const { return Policy{}; }

    // ---------------------------------------------------------------- memory
    virtual void *alloc(size_t bytes) = 0;  // throws std::runtime_error on failure
    virtual void release(void *p) = 0;
    virtual void *host_alloc(size_t bytes) = 0;  // pinned on CUDA
    virtual void host_release(void *p) = 0;
    virtual void upload(void *dst, const void *host_src, size_t n) = 0;
    virtual void download(void *host_dst, const void *src, size_t n) = 0;
    virtual void copy(void *dst, const void *src, size_t n) = 0;
    virtual void zero(void *p, size_t n) = 0;
    virtual void sync() = 0;
    // An event marks a point in the op stream; wait_event blocks the HOST until every op
    // queued before it has finished. On the CPU backend both are no-ops.
    virtual void *record_event() = 0;
    virtual void wait_event(void *ev) = 0;
    virtual void release_event(void *ev) = 0;
    virtual size_t free_memory() const = 0;
    // Page copies for the prefix cache's host tier, on a copy stream of their own (in order
    // among themselves) that starts after `after` (an event on the main stream, from
    // record_event; may be null). Page dev_pages[i] of each paged pool (pools[s]:
    // [n_pages][seg_bytes]) to or from host page host_pages[i] (host_alloc memory,
    // [n_pools][seg_bytes]). May be called from another thread than the main stream's (it must
    // not touch the main stream). record_copy_event marks the copy stream; event_done asks
    // without blocking; stream_wait makes the main stream's later ops wait for an event without
    // blocking the host. The defaults copy synchronously through upload / download, which is all
    // the CPU backend needs.
    virtual void copy_pages(bool to_host, void *const *pools, int n_pools, size_t seg_bytes,
                            const int32_t *dev_pages, void *const *host_pages, int64_t n, void *after) {
        if (after) wait_event(after);
        for (int64_t i = 0; i < n; ++i)
            for (int s = 0; s < n_pools; ++s) {
                char *dev = static_cast<char *>(pools[s]) + size_t(dev_pages[i]) * seg_bytes;
                char *host = static_cast<char *>(host_pages[i]) + size_t(s) * seg_bytes;
                if (to_host)
                    download(host, dev, seg_bytes);
                else
                    upload(dev, host, seg_bytes);
            }
    }
    // Host bytes to device memory on that copy stream, `height` rows of `width` bytes (source rows
    // `spitch` apart, destination rows `dpitch` apart; height 1 for a plain copy). From pageable
    // memory the call returns once the bytes are staged, so the source may then be reused. Like
    // copy_pages it may be called from another thread and never touches the main stream; a
    // record_copy_event after it marks its completion. The default copies synchronously.
    virtual void upload_async(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height) {
        for (size_t r = 0; r < height; ++r)
            upload(static_cast<char *>(dst) + r * dpitch, static_cast<const char *>(src) + r * spitch, width);
    }
    virtual void *record_copy_event() { return record_event(); }
    virtual bool event_done(void *ev) {
        if (ev) wait_event(ev);
        return true;
    }
    virtual void stream_wait(void *) {}
    // Graphs: the ops queued between graph_begin and graph_end are recorded into a replayable
    // graph instead of run (CUDA graphs). Pointers and launch arguments are baked in, so a
    // graph may only be replayed where the same ops would be queued again, and nothing in the
    // recorded span may synchronise with the host. A backend without graphs returns false
    // from graph_begin and the caller runs the ops directly.
    virtual bool graph_begin() { return false; }
    // Exact (default): small-M GEMVs keep x in fp32 (on CUDA: TF32 hi + lo). Otherwise x is
    // rounded to bf16 for them, as cuBLAS and vLLM do; still deterministic and batch-invariant.
    virtual void set_exact_activations(bool) {}
    // With bf16 activations, steps of at least this many rows may take the backend's BLAS for the
    // dense GEMMs: the executor's choice for its model (a backend may ignore it).
    virtual void set_blas_min_rows(int64_t) {}
    // int4 experts: steps of more than this many tokens may take the backend's int4 GEMM instead of
    // its decode tiles (the executor's choice for its routing; a backend may ignore it)
    virtual void set_int4_gemm_rows(int64_t) {}
    virtual void *graph_end() { return nullptr; }
    virtual void graph_abort() {}  // after a failure between begin and end
    virtual void graph_launch(void *) {}
    virtual void graph_release(void *) {}
    // counts[idx[i]] += 1 for i < n, entries outside [0, n_counts) skipped: how often each expert
    // is routed to, kept on the device (graph-recordable; the executor places experts by it).
    // Returns false when the backend has no such op.
    virtual bool count_routes(uint32_t * /*counts*/, const int32_t * /*idx*/, int64_t /*n*/, int64_t /*n_counts*/) {
        return false;
    }

    // ---------------------------------------------------------------- dense
    // y[m*ldy + n] = sum_k x[m*K + k] * W[n][k], m < M, n < w.rows, K = w.cols.
    virtual void gemm(const Weight &w, const float *x, int64_t M, float *y, int64_t ldy) = 0;
    // Several gemms with the same M, each with its own weight, x and y (the projections that
    // read one hidden state). The results are those of the separate calls bit for bit; a
    // backend may run the group as one launch.
    struct GemmJob {
        const Weight *w;
        const float *x;
        float *y;
        int64_t ldy;
        /// The result is only read by the backend's own next op (q/k/v into qk_rope_append): the
        /// backend may keep it in its activation precision and never write y (see gemm16).
        bool y16_ok = false;
    };
    virtual void gemm_group(const GemmJob *jobs, int n, int64_t M) {
        for (int i = 0; i < n; ++i) gemm(*jobs[i].w, jobs[i].x, M, jobs[i].y, jobs[i].ldy);
    }
    /// gemm whose result only the backend's own next op reads (the o projection into the
    /// add + norm): a backend may keep it in its activation precision and never write y.
    virtual void gemm16(const Weight &w, const float *x, int64_t M, float *y, int64_t ldy) { gemm(w, x, M, y, ldy); }
    // The page count of the key/value pools (the paged attention kernels describe the whole pool).
    virtual void set_kv_pages(int64_t) {}
    // Bytes of one page of a layer's K (or V) cache: page positions x Hkv heads x D in layout dt.
    // The executor sizes its pools by it, so a backend whose layout differs says what it takes;
    // the base form (backend.cpp) is the layouts of backend.h's DType.
    virtual size_t kv_page_bytes(DType dt, int64_t page, int64_t Hkv, int64_t D) const;
    // The format of the resident routed experts (the backend may prepare their operands for it).
    virtual void set_expert_format(WFmt) {}

    // ---------------------------------------------------------------- elementwise
    // y = w * x / sqrt(mean(x[:n]^2) + eps) over the first n of each row; stride is the row
    // length in floats. y == x allowed. Elements past n are untouched.
    virtual void rmsnorm(float *y, const float *x, const float *w, int64_t rows, int64_t n,
                         int64_t stride, float eps) = 0;
    // x[r][h*d .. h*d+d] *= 1/sqrt(sum(sq) + eps), in place.
    virtual void l2norm_heads(float * /*x*/, int64_t /*rows*/, int64_t /*heads*/, int64_t /*d*/, float /*eps*/) { unsupported("l2norm_heads"); }
    // y = b1*tanh(g/b1)*sigmoid(g) * b2*tanh(u/b2); the sigmoid sees the UNCAPPED g.
    virtual void situ_glu(float * /*y*/, const float * /*gate*/, const float * /*up*/, int64_t /*n*/, float /*b1*/,
                          float /*b2*/) { unsupported("situ_glu"); }
    // y = silu(g) * u = g * sigmoid(g) * u (Kimi Linear, DeepSeek-style experts).
    virtual void silu_glu(float *y, const float *gate, const float *up, int64_t n) = 0;
    // LoRA, many adapters at once, for a group of targets that read one x (q, k, v; gate, up; or
    // one alone). Job t, slot s: A at a + s * a_stride ([ldb][in] bf16), B at b + s * out * ldb
    // ([out][ldb] bf16; ldb: the pools' rank), scale[s], ranks[s] (the adapter's rank: A's rows
    // and B's columns past it are zero; 0 for an empty slot). rank: at least every ranks[s] (the
    // work's bound; at most ldb). lora: every row r with s = slot[r] >= 0 gets
    // y_t[r] += scale_t[s] B_ts (A_ts x[r]), sums in fp32; rows with slot -1 are left alone. y_t
    // [N][out] is an earlier GEMM's result, wherever the backend left it (fp32, or a bf16 form the
    // backend updates in place). xa: scratch of N * n * rank floats the backend may use.
    struct LoraJob {
        const uint16_t *a = nullptr;
        int64_t a_stride = 0;
        const uint16_t *b = nullptr;
        const float *scale = nullptr;
        const int32_t *ranks = nullptr;
        float *y = nullptr;
        int64_t out = 0;
        int64_t ldb = 0;
    };
    static constexpr int kMaxLoraJobs = 3;
    virtual void lora(const LoraJob *jobs, int n, const float *x, int64_t N, int64_t in, int64_t rank,
                      const int32_t *slot, float *xa);
    // The same for rows [row0, row0 + M) of the step's N, all on one slot (the caller's run of
    // rows on one adapter), M at least lora_rows_min() (0: the backend has no such form, use lora);
    // r[t] / scale[t]: the slot's rank and scale for job t, on the host. The backend may round x and
    // A x to bf16 (tensor-core GEMMs). Jobs whose A pools sit back to back (job t's at
    // a_0 + t * ldb * in, as dense_exec stacks a group's) may take one GEMM.
    virtual int64_t lora_rows_min() const { return 0; }
    virtual void lora_rows(const LoraJob * /*jobs*/, int /*n*/, const float * /*x*/, int64_t /*N*/, int64_t /*row0*/,
                           int64_t /*M*/, int64_t /*in*/, int64_t /*rank*/, int32_t /*slot*/, const int32_t * /*r*/,
                           const float * /*scale*/) {
        unsupported("lora_rows");
    }
    virtual void add(float *y, const float *x, int64_t n) = 0;          // y += x
    // h += d, then y = rmsnorm(h) (row stride n): the residual add and the next pre-norm. The
    // same bits as add then rmsnorm; a backend may fuse them.
    virtual void add_rmsnorm(float *h, const float *d, float *y, const float *w, int64_t rows, int64_t n,
                             float eps) {
        add(h, d, rows * n);
        rmsnorm(y, h, w, rows, n, n, eps);
    }
    virtual void sigmoid_mul(float * /*y*/, const float * /*g*/, int64_t /*n*/) { unsupported("sigmoid_mul"); }  // y *= sigmoid(g)
    // dst[r*dst_stride + dst_off + i] = src[r*n + i]
    virtual void copy_rows(float *dst, int64_t dst_stride, int64_t dst_off, const float *src,
                           int64_t rows, int64_t n) = 0;
    // AttnRes, every row r: sources are blocks[r*maxs*n + b*n] for b < nb, then pref[r*n].
    //   score_s = dot(src_s * rsqrt(mean(src_s^2) + eps), fold); out[r] = sum softmax(score)_s * src_s
    // out must not alias blocks or pref.
    virtual void attn_res(float * /*out*/, const float * /*blocks*/, int64_t /*nb*/, int64_t /*maxs*/,
                          const float * /*pref*/, const float * /*fold*/, int64_t /*rows*/, int64_t /*n*/,
                          float /*eps*/) { unsupported("attn_res"); }

    // ---------------------------------------------------------------- Kimi Delta Attention
    // Segments: seg[3*i + 0] = state slot, seg[3*i + 1] = first token row, seg[3*i + 2] = count.
    // Tokens of a segment are consecutive rows, in position order.
    //
    // ShortConv with fused SiLU. x, y: [N][C] (y must not alias x); w: [C][K], w[K-1] on the
    // CURRENT input. History of slot s: conv_state + s*conv_stride + hist_off, [C][K-1] fp32,
    // oldest first, updated in place.
    virtual void kda_conv(float * /*y*/, const float * /*x*/, const float * /*w*/, float * /*conv_state*/,
                          int64_t /*conv_stride*/, int64_t /*hist_off*/, const int32_t * /*seg*/, int64_t /*nseg*/,
                          int64_t /*C*/, int64_t /*K*/) { unsupported("kda_conv"); }
    // With zz = z[t][h*D+d] + dt_bias[h*D+d] and a = exp(a_log[h]) (PER HEAD):
    //   softplus == false (K3):          alpha = exp(lb * sigmoid(a * zz))
    //   softplus == true  (Kimi Linear): alpha = exp(-a * softplus(zz))   (lb unused)
    // beta[t][h] = sigmoid(braw[t][h]).
    virtual void kda_gates(float * /*alpha*/, float * /*beta*/, const float * /*z*/, const float * /*braw*/,
                           const float * /*a_log*/, const float * /*dt_bias*/, int64_t /*N*/, int64_t /*H*/,
                           int64_t /*D*/, float /*lb*/, bool /*softplus*/) { unsupported("kda_gates"); }
    // The delta rule, positions in order within each segment. State of (slot, head) is the
    // D*D elements at state + slot*state_stride + h*D*D (elements of state_dtype),
    // row-major [dk][dv]. Per token: S[i][:] *= alpha[i]; u = S^T k; S += k (beta (v - u))^T;
    // o = S^T (q * qscale). Compute is fp32; a bf16 state is widened on load and rounded
    // once at the end of each segment.
    virtual void kda_recurrence(float * /*o*/, const float * /*q*/, const float * /*k*/, const float * /*v*/,
                                const float * /*alpha*/, const float * /*beta*/, void * /*state*/,
                                DType /*state_dtype*/, int64_t /*state_stride*/, const int32_t * /*seg*/,
                                int64_t /*nseg*/, int64_t /*H*/, int64_t /*D*/, float /*qscale*/) { unsupported("kda_recurrence"); }
    // o[t][h] = rmsnorm(o[t][h], o_norm, eps) * sigmoid(gate[t][h]), in place.
    virtual void kda_out(float * /*o*/, const float * /*gate*/, const float * /*o_norm*/, int64_t /*N*/, int64_t /*H*/,
                         int64_t /*D*/, float /*eps*/) { unsupported("kda_out"); }
    // Everything from the projections to o_proj's input in one call: per segment, per head,
    //   q, k, v = kda_conv(qp, kp, vp) (histories at hist_off 0, hist, 2*hist; hist = H*D*(K-1));
    //   l2norm_heads(q), l2norm_heads(k) (eps l2_eps); kda_gates(z, braw); kda_recurrence;
    //   kda_out(o, gate, o_norm, out_eps).
    // The same results as that sequence up to summation order. Returns false when the backend
    // has no fused form (the caller then runs the sequence).
    struct KdaFused {
        float *o;
        const float *qp, *kp, *vp, *wq, *wk, *wv;
        float *conv_state;
        int64_t conv_stride;
        const float *z, *braw, *a_log, *dt_bias;
        float lb;
        bool softplus;
        const float *gate, *o_norm;
        float out_eps, l2_eps, qscale;
        void *state;
        DType state_dtype;
        int64_t state_stride;
        const int32_t *seg;
        int64_t nseg, H, D, K;
    };
    virtual bool kda_fused(const KdaFused &) { return false; }
    // kda_fused over long segments (prompts) in the delta rule's chunk-parallel form: the same
    // results up to the order of the sums. seg_host: the same [nseg][3] list on the host (the
    // chunk counts). Returns false when the backend has no chunked form (the caller runs
    // kda_fused). kda_chunk_min: segments at least this long are worth it (0: never).
    virtual bool kda_chunked(const KdaFused &, const int32_t * /*seg_host*/) { return false; }
    virtual int64_t kda_chunk_min() const { return 0; }
    // the device scratch kda_chunked may hold for a step of n_tokens in up to n_segs segments
    // (the executor's memory plan counts it; kda_chunked and mla_prefill may share one buffer,
    // so the plan takes the larger of the two)
    virtual size_t kda_chunked_scratch(int64_t /*n_tokens*/, int64_t /*n_segs*/, int64_t /*H*/, int64_t /*D*/) const {
        return 0;
    }

    // ---------------------------------------------------------------- Gated MLA (absorbed)
    // One layer's latent cache: [n_pages][page][width] of kv_dtype, width = kv_lora + qr.
    // Position p of slot s is row p % page of page page_table[s*max_pages + p/page].
    //
    // Write ckv[t] ([N][width] fp32: normalised latent, then the raw rope slot).
    virtual void mla_append(void *kv, DType kv_dtype, const float *ckv, const int32_t *tok_slot,
                            const int32_t *tok_pos, const int32_t *page_table, int64_t max_pages,
                            int64_t page, int64_t width, int64_t N) = 0;
    // q: [N][H][qn+qr]; kv_b: bf16 Weight [H*(qn+vh)][kv_lora] (per head qn rows of W_UK
    // then vh rows of W_UV). q_abs: [N][H][kv_lora+qr] = (W_UK^T q_nope, q_rope).
    virtual void mla_absorb_q(float * /*q_abs*/, const float * /*q*/, const Weight &/*kv_b*/, int64_t /*N*/,
                              int64_t /*H*/, int64_t /*qn*/, int64_t /*qr*/, int64_t /*vh*/) { unsupported("mla_absorb_q"); }
    // Causal attention of token t over positions 0..tok_pos[t] of its own slot, in latent
    // space: o_lat[t][h] = sum_p softmax_p(scale * q_abs[t][h] . kv[p]) * kv[p][:kv_lora].
    // max_pos is the largest tok_pos in the batch.
    virtual void mla_attention(float * /*o_lat*/, const float * /*q_abs*/, const void * /*kv*/, DType /*kv_dtype*/,
                               const int32_t * /*tok_slot*/, const int32_t * /*tok_pos*/,
                               const int32_t * /*page_table*/, int64_t /*max_pages*/, int64_t /*page*/,
                               int64_t /*N*/, int64_t /*H*/, int64_t /*kv_lora*/, int64_t /*qr*/,
                               int64_t /*max_pos*/, float /*scale*/) { unsupported("mla_attention"); }
    // y[t][h*vh + j] = W_UV^h[j] . o_lat[t][h]
    virtual void mla_unabsorb(float * /*y*/, const float * /*o_lat*/, const Weight &/*kv_b*/, int64_t /*N*/,
                              int64_t /*H*/, int64_t /*qn*/, int64_t /*vh*/) { unsupported("mla_unabsorb"); }
    // MLA over long segments (prompts), un-absorbed: per head, k = (W_UK^h c, rope slot) and
    // v = W_UV^h c expanded from the latent cache over each segment's whole context, then causal
    // attention with (qn + qr)-wide q and k and vh-wide v on the tensor cores: what mla_absorb_q,
    // mla_attention and mla_unabsorb compute, up to rounding (bf16 operands, fp32 sums). Writes
    // y's rows of the segments only. meta (device int32; meta_host the same on the host): cu_q
    // [n + 1] (prefix sums of the segments' token counts), cu_k [n + 1] (of their key counts: last
    // position + 1), slot [n], first row [n]. Any context length: a backend may expand the keys in
    // chunks and merge the chunks' attention. Returns false when the backend has no such kernel
    // for these shapes.
    struct MlaPrefill {
        float *y;        // [N][H * vh]
        const float *q;  // [N][H][qn + qr]
        const void *kv;  // the layer's latent cache
        DType kv_dtype;
        const Weight *kv_b;
        const int32_t *page_table;
        int64_t max_pages, page;
        const int32_t *meta, *meta_host;
        int64_t n, total_q, total_k, max_q, max_k;
        int64_t H, qn, qr, vh, kv_lora;
        float scale;
        // tests: chunk_keys > 0 forces chunks of this many keys (chunk_heads > 0: passes of this
        // many heads), chunk_keys < 0 one launch a pass when it fits
        int64_t chunk_keys, chunk_heads;
    };
    virtual bool mla_prefill(const MlaPrefill &) { return false; }
    virtual int64_t mla_prefill_min() const { return 0; }  // segments at least this long take it (0: never)
    // the device scratch mla_prefill may hold for a step of n_tokens (latent rows `width` wide)
    virtual size_t mla_prefill_scratch(int64_t /*n_tokens*/, int64_t /*H*/, int64_t /*width*/) const { return 0; }
    // gather: dst[i*n + c] = src[rows[i]*n + c]; scatter: dst[rows[i]*n + c] = src[i*n + c]
    virtual void index_rows(float * /*dst*/, const float * /*src*/, const int32_t * /*rows*/, int64_t /*count*/,
                            int64_t /*n*/, bool /*scatter*/) { unsupported("index_rows"); }

    // ---------------------------------------------------------------- dense attention (Llama / Qwen)
    // Rotate-half RoPE, in place: x rows of `stride` floats hold `heads` vectors of D from 0;
    // for i < D/2, angle = tok_pos[t] * inv_freq[i] (fp32), a = x[i], b = x[i + D/2]:
    //   x[i] = a cos - b sin, x[i + D/2] = b cos + a sin.
    virtual void rope(float *, const int32_t * /*tok_pos*/, const float * /*inv_freq*/, int64_t /*N*/,
                      int64_t /*heads*/, int64_t /*D*/, int64_t /*stride*/) {
        unsupported("rope");
    }
    // Causal grouped-query attention over a paged cache. kc, vc: one layer's keys and values,
    // [n_pages][page][Hkv][D] of kv_dtype, written by mla_append with width Hkv*D. Token t
    // attends to positions 0..tok_pos[t] of its slot; query head h reads kv head h / (Hq/Hkv).
    // o, q: [N][Hq][D]; scores scaled by `scale`; max_pos is the largest tok_pos.
    virtual void gqa_attention(float *, const float * /*q*/, const void * /*kc*/, const void * /*vc*/, DType,
                               const int32_t * /*tok_slot*/, const int32_t * /*tok_pos*/,
                               const int32_t * /*page_table*/, int64_t /*max_pages*/, int64_t /*page*/,
                               int64_t /*N*/, int64_t /*Hq*/, int64_t /*Hkv*/, int64_t /*D*/, int64_t /*max_pos*/,
                               float /*scale*/) {
        unsupported("gqa_attention");
    }

    // The step's attention in its prefill form (bf16 tensor cores). Rows [0, n_dec) are the
    // decoding sequences' single tokens (each attends to its own context: positions and slots in
    // tok_pos / tok_slot); the rest are prompt chunks: tiles [n_tiles][2] = (first token row,
    // count <= 16), each tile's tokens one sequence's consecutive positions, and h_segs, in HOST
    // memory, [n_segs][4] = (first token row, count, first position, slot) per chunk (null when
    // unknown). max_pos: the largest position of the step (an upper bound is fine). rows: the
    // backend may take the n_dec rows too (else it leaves them to gqa_attention). Returns 2
    // when every row was done, 1 when only the chunks were (the caller runs gqa_attention on
    // the n_dec rows), 0 when nothing was (no such kernel for this shape or precision).
    virtual int gqa_prefill(float * /*o*/, const float * /*q*/, const void * /*kc*/, const void * /*vc*/, DType,
                            const int32_t * /*tiles*/, int64_t /*n_tiles*/, const int32_t * /*h_segs*/,
                            int64_t /*n_segs*/, int64_t /*n_dec*/, bool /*rows*/, const int32_t * /*tok_slot*/,
                            const int32_t * /*tok_pos*/, const int32_t * /*page_table*/, int64_t /*max_pages*/,
                            int64_t /*page*/, int64_t /*Hq*/, int64_t /*Hkv*/, int64_t /*D*/, int64_t /*max_pos*/,
                            float /*scale*/) {
        return 0;
    }

    // ---------------------------------------------------------------- MoE routing
    // Sigmoid scores, selection by score + bias (first index wins ties), weights from the
    // UNBIASED scores, renormalised if renorm, times scale. idx/w: [N][k] in selection order.
    // biased: weights from score + bias instead (the released Kimi Linear HF code's behaviour).
    virtual void router_topk(int32_t * /*idx*/, float * /*w*/, const float * /*logits*/, const float * /*bias*/,
                             int64_t /*N*/, int64_t /*E*/, int64_t /*k*/, bool /*renorm*/, float /*scale*/,
                             bool /*biased*/) { unsupported("router_topk"); }

    // dst[i] = row ids[i] of table ([vocab][n], bf16 or f32) as fp32; ids on the device.
    virtual void embed_rows(float * /*dst*/, const Weight & /*table*/, const int32_t * /*ids*/, int64_t /*N*/,
                            int64_t /*n*/) {
        unsupported("embed_rows");
    }
    // dst[r] = table[idx[r]] for r < n, int32 on the device (an EAGLE-3 head's draft ids to the
    // target's, its drafts kept on the device between draft steps).
    virtual void gather_ids(int32_t * /*dst*/, const int32_t * /*table*/, const int32_t * /*idx*/, int64_t /*n*/) {
        unsupported("gather_ids");
    }

    // ---------------------------------------------------------------- sampling
    // idx[r] = the first index of the largest of x[r][0..n) (NaN never wins). Any non-finite
    // value sets *bad = 1 (it is never cleared here).
    virtual void argmax_rows(int32_t *idx, int32_t *bad, const float *x, int64_t rows, int64_t n) = 0;
    // Structured output: every row r with p[r].mask = k > 0 gets -inf where masks[k - 1] (words
    // words, token i at bit i % 32 of word i / 32) does not allow the token; any non-finite value
    // of such a row before that sets *bad = 1. Other rows are left alone. p, masks on the device.
    virtual void mask_rows(float *x, int32_t *bad, int64_t rows, int64_t n, const SampleRow *p, const uint32_t *masks,
                           int64_t words);
    // Every row r with p[r].adj_n > 0 gets its adjustments adj[p[r].adj_off ..] (LogitAdj, distinct
    // tokens) in place. p, adj on the device.
    virtual void adjust_rows(float *x, int64_t rows, int64_t n, const SampleRow *p, const LogitAdj *adj);
    // Every row r with p[r].logprobs = k > 0: lp[r][0] = log-softmax(x[r])[idx[r]], then the k - 1
    // largest of the row (larger value, else smaller index first): their log-probabilities at
    // lp[r][1 ..] and ids at ids[r][0 ..]; lp rows kMaxTopLogprobs + 1 wide, ids rows kMaxTopLogprobs.
    // The base form copies to the host (logprob_row_host).
    virtual void logprob_rows(float *lp, int32_t *ids, const float *x, int64_t rows, int64_t n, const SampleRow *p,
                              const int32_t *idx);
    // idx[r] = the token drawn from row r as p[r] says (sk_sample_row in fullm_exec.h: the
    // argmax at temperature <= 0; else top-k, then top-p by value thresholds, inverse CDF in
    // token order with PCG32(seed, step); min-p cuts with p.min_p_cut); p on the device. *bad as
    // argmax_rows, except that a row with a mask or adjustments may hold -inf (mask_rows,
    // adjust_rows). The base form copies the rows to the host and draws there (sample_row_host):
    // a backend's kernel must match it.
    virtual void sample_rows(int32_t *idx, int32_t *bad, const float *x, int64_t rows, int64_t n,
                             const SampleRow *p);

    // Fused attention prologue (Llama / Qwen): for every token, the per-head RMSNorm of q and k
    // (when q_norm / k_norm are given), RoPE on both (as rope), q in place, then k and v appended
    // to the paged cache (as mla_append with width Hkv*D). Returns false when the backend has no
    // fused form (the caller runs those ops).
    virtual bool qk_rope_append(float * /*q*/, const float * /*k*/, const float * /*v*/, void * /*kc*/,
                                void * /*vc*/, DType, const float * /*q_norm*/, const float * /*k_norm*/,
                                float /*eps*/, const float * /*inv_freq*/, const int32_t * /*tok_slot*/,
                                const int32_t * /*tok_pos*/, const int32_t * /*page_table*/, int64_t /*max_pages*/,
                                int64_t /*page*/, int64_t /*N*/, int64_t /*Hq*/, int64_t /*Hkv*/, int64_t /*D*/) {
        return false;
    }
    // Resident expert weights (see moe_experts).
    struct ExpertW {
        Weight w1, w3, w2;
    };
    // Router GEMV (bf16 W [E][width], fp32 x) + router_softmax_topk in one call. Returns false
    // when the backend has no fused form for this shape. table / n_experts (optional): those of
    // the moe_experts call that follows with the same idx and N; the backend may prepare that
    // call's routing then (it must not rely on it if any other call comes in between).
    virtual bool router_softmax_fused(int32_t * /*idx*/, float * /*w*/, const Weight & /*W*/, const float * /*x*/,
                                      int64_t /*N*/, int64_t /*k*/, bool /*renorm*/, const ExpertW * /*table*/ = nullptr,
                                      int64_t /*n_experts*/ = 0) {
        return false;
    }

    // Softmax router (Qwen3-MoE): p = softmax(logits[t]) over E; the k largest p (first index
    // wins ties) in selection order; w = p, divided by their sum if renorm. idx/w: [N][k].
    virtual void router_softmax_topk(int32_t *, float *, const float * /*logits*/, int64_t /*N*/, int64_t /*E*/,
                                     int64_t /*k*/, bool /*renorm*/) {
        unsupported("router_softmax_topk");
    }

    // ---------------------------------------------------------------- routed experts in backend memory
    // table: [n_experts] of ExpertW in backend memory; an entry with w1.data == nullptr is not
    // resident here (the CPU expert pool computes it). For every pair p = t*k + j whose expert
    // idx[p] is resident:
    //     contrib[p][:] = w[p] * down( glu( gate(x[t]), up(x[t]) ) )
    // with gate/up [inter][width], down [width][inter], all resident experts in weight format
    // fmt; pairs whose expert is not resident are left untouched. glu is SiLU-GLU (silu) or
    // SiTU-GLU (b1, b2). scratch: >= N*k*inter floats of backend memory.
    // n_experts (<= 1024): entries of table.
    // The base form (backend.cpp) composes it from core ops: each resident expert's rows gathered,
    // gemm gate and up (separate matrices: moe_gate_up_interleaved stays false), the GLU, gemm
    // down, the routing weights applied on the host. Correct on any backend, not fast.
    virtual void moe_experts(float *contrib, const float *x, const int32_t *idx, const float *w,
                             int64_t N, int64_t k, const ExpertW *table, int64_t n_experts, WFmt fmt,
                             int64_t width, int64_t inter, bool silu, float b1, float b2, float *scratch);
    // A dense MLP held as n experts every token takes at weight 1 (expert e: the e-th slice of the
    // intermediate dimension, its gate / up rows interleaved in w1 and its down columns in w2;
    // host_table: the experts' Weights on the host): out [N][width] = the MLP of x by the
    // backend's own path for big steps. False: not taken (nothing done; run moe_experts).
    virtual bool split_mlp(float * /*out*/, const float * /*x*/, int64_t /*N*/, const ExpertW * /*host_table*/,
                           int64_t /*n*/, WFmt /*fmt*/, int64_t /*width*/, int64_t /*inter*/) {
        return false;
    }
    // When true for a format and shape, the table's experts must hold gate and up as ONE
    // [2*inter][width] matrix in w1 with rows interleaved (gate 0, up 0, gate 1, up 1, ...;
    // scales likewise) and w3 empty: the backend then computes the GLU from adjacent rows.
    virtual bool moe_gate_up_interleaved(WFmt, int64_t /*width*/, int64_t /*inter*/) const { return false; }
    // The packing this backend reads a [rows][cols] matrix of fmt in (Weight::layout): the
    // executor quantises into it, and the expert cache records it. The base form is 0: INT4G as
    // the MMA fragments where the shape allows (what the CUDA decode tile reads).
    virtual int32_t weight_layout(WFmt /*fmt*/, int64_t /*rows*/, int64_t /*cols*/) const { return 0; }
    // Whether moe_experts runs prompt-sized steps of INT4G experts on a GEMM of its own over the
    // same weights (else they need another copy, e.g. fp8b, for those steps to be fast).
    virtual bool int4_prefill_gemm(int64_t /*width*/, int64_t /*inter*/) const { return false; }
    // A backend may leave moe_experts' routed rows in its own order until the moe_reduce that
    // consumes contrib; this materialises them in contrib (pair order) for any other reader.
    virtual void moe_materialize(float * /*contrib*/) {}
    // acc[t] = sum_{j<k} contrib[t*k + j], in j order (deterministic whatever computed the rows),
    // then + addend[t] when addend is given (the shared expert).
    virtual void moe_reduce(float *acc, const float *contrib, int64_t N, int64_t k, int64_t width,
                            const float *addend = nullptr) = 0;
    // moe_reduce into the residual, then the next norm: h += reduce(contrib) (+ addend);
    // y = rmsnorm(h) * w. The same bits as moe_reduce then add_rmsnorm (tmp: [N][width] scratch
    // for backends without the fused form).
    virtual void moe_reduce_add_rmsnorm(float *h, const float *contrib, int64_t N, int64_t k, int64_t width,
                                        const float *addend, float *y, const float *w, float eps, float *tmp) {
        moe_reduce(tmp, contrib, N, k, width, addend);
        add_rmsnorm(h, tmp, y, w, N, width, eps);
    }
    // dst[rows[i]] = src[i] for i < n (rows of `width` floats): merges rows computed elsewhere.
    virtual void scatter_rows(float *dst, const float *src, const int32_t *rows, int64_t n,
                              int64_t width) = 0;

protected:
    [[noreturn]] void unsupported(const char *op) const;
};

inline void Backend::unsupported(const char *op) const {
    throw std::runtime_error(std::string(name()) + ": " + op + " is not implemented on this backend yet");
}

std::unique_ptr<Backend> make_cpu_backend(int threads);
// The CPU backend with its gemm replaced where `gemm` returns true (otherwise the native
// kernels run): how the torch backend plugs libtorch's GEMM in.
using GemmHook = bool (*)(const Weight &w, const float *x, int64_t M, float *y, int64_t ldy);
std::unique_ptr<Backend> make_cpu_backend_with_gemm(int threads, GemmHook gemm, const char *name);
// The CPU backend computing every bf16 GEMM with libtorch in bf16 (at::linear: x rounded to
// bf16, bf16 output), as vLLM does on a CPU; everything else as the CPU backend. Throws when
// built without the `torch` feature.
std::unique_ptr<Backend> make_torch_backend(int threads);
// Throws when built without CUDA or when the device cannot be opened.
std::unique_ptr<Backend> make_cuda_backend(int device);
// The CPU backend reduced to the runtime, the core ops and the GQA family (core_backend.cpp).
std::unique_ptr<Backend> make_cpu_core_backend(int threads);
// The backends compiled into this build, by name: available() says whether
// one can run on this machine; create() builds it. An executor names its backend in
// sk_exec_opts.backend_name; the old SK_BACKEND_* numbers are aliases of "cpu", "cuda", "torch".
struct BackendEntry {
    const char *name;
    bool (*available)(int device);
    std::unique_ptr<Backend> (*create)(int device, int cpu_threads);
};
const std::vector<BackendEntry> &backend_registry();
std::unique_ptr<Backend> make_backend(const std::string &name, int device, int cpu_threads);
// sk_exec_opts' backend: its name, else the alias of its number
std::string backend_name(const char *name, int kind);
bool cuda_available();

}  // namespace fullm
