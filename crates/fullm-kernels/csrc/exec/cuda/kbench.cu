// SPDX-License-Identifier: Apache-2.0
// kbench.cu - kernel micro-benchmarks: one backend op at a time on synthetic device data of real
// shapes (default Qwen3-30B-A3B), timed over many calls. No model, no load: seconds per case,
// for iterating on a kernel (and for `ncu` on one launch) without the minutes a model takes.
//
// spec: cases separated by ';', each "op key=value ...":
//   attn  n=16 ctx=8000 kv=bf16|fp8 fa3=0|1 hq=32 hkv=4 d=128 page=16 iters=50
//         decode attention of n rows, each over its own ctx positions (fa3=1: bf16 rows through
//         the prefill form, FlashAttention-3, as the executor sends many long rows)
//   moe   n=16 fmt=bf16|fp8b|int4 e=128 k=8 width=2048 inter=768 iters=20
//         the resident experts (moe_experts) for n tokens, top-k of e random experts each
//   gemv  n=1 fmt=bf16|int8|fp8 rows=19456 cols=2560 iters=200
//         the dense trunk's GEMV over a random matrix (default Qwen3-4B's gate/up)
//   mla   nq=8192 ctx=100000 segs=1 h=32 page=16 keys=0 heads=0 iters=3
//         MLA prefill un-absorbed (mla_prefill, Kimi Linear's shapes: latent 512 + rope 64, q/k 192,
//         v 128, bf16 cache) of segs segments, each nq queries ending a ctx-key context; keys > 0
//         forces chunks of that many keys (heads > 0 passes of that many heads), keys=-1 one launch
//         a pass (when it fits)
//   pagecopy pages=2048 layers=48 hkv=4 d=128 page=16 iters=5 separate=0
//         the host tier's page copies (copy_pages, bf16 K and V pools of every layer) to pinned
//         host pages and back, on the copy stream; separate=1 allocates each pool on its own
//         (batched copies; FULLM_COPY_KERNEL=1 the kernel, FULLM_COPY_BLOCKS its grid)
// Values are random (fp8 / int4 bit patterns without NaNs); the results are not checked here.

#include "../backend.h"
#include "../kv_fp8.h"
#include "fullm_exec.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using fullm::Backend;
using fullm::DType;
using fullm::Weight;
using fullm::WFmt;

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    float uni() { return static_cast<float>((next() >> 40) * (1.0 / 16777216.0)) * 2.0f - 1.0f; }
};

uint16_t bf16_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// device buffers released with the case
struct Mem {
    Backend &be;
    std::vector<void *> ptrs;
    explicit Mem(Backend &b) : be(b) {}
    Mem(const Mem &) = delete;
    Mem &operator=(const Mem &) = delete;
    ~Mem() {
        try {
            be.sync();
        } catch (...) {
        }
        for (void *p : ptrs) be.release(p);
    }
    void *alloc(size_t n) {
        void *p = be.alloc(n ? n : 4);
        ptrs.push_back(p);
        return p;
    }
    template <class T>
    T *up(const std::vector<T> &h) {
        T *p = static_cast<T *>(alloc(h.size() * sizeof(T)));
        if (!h.empty()) be.upload(p, h.data(), h.size() * sizeof(T));
        return p;
    }
};

using Params = std::map<std::string, std::string>;
int64_t geti(const Params &p, const char *k, int64_t d) {
    auto it = p.find(k);
    return it == p.end() ? d : std::stoll(it->second);
}
std::string gets(const Params &p, const char *k, const char *d) {
    auto it = p.find(k);
    return it == p.end() ? d : it->second;
}

// ms per call of f over iters calls (after warm-up calls), the backend synchronised around them
template <class F>
double time_ms(Backend &be, int64_t iters, F &&f) {
    for (int i = 0; i < 3; ++i) f();
    be.sync();
    const auto t0 = std::chrono::steady_clock::now();
    for (int64_t i = 0; i < iters; ++i) f();
    be.sync();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / double(iters);
}

std::string attn(Backend &be, const Params &p) {
    const int64_t n = geti(p, "n", 16), ctx = geti(p, "ctx", 8000), Hq = geti(p, "hq", 32), Hk = geti(p, "hkv", 4),
                  D = geti(p, "d", 128), page = geti(p, "page", 16), iters = geti(p, "iters", 50);
    const std::string kv = gets(p, "kv", "bf16");
    const bool f8 = kv == "fp8", fa3 = geti(p, "fa3", 0) != 0;
    if (!f8 && kv != "bf16") throw std::runtime_error("attn: kv=bf16|fp8");
    const int64_t mp = (ctx + page - 1) / page, n_pages = n * mp;
    Mem m(be);
    Rng rng(7);
    std::vector<int32_t> pt(static_cast<size_t>(n * mp)), slot(static_cast<size_t>(n)), pos(static_cast<size_t>(n));
    for (int64_t i = 0; i < n * mp; ++i) pt[static_cast<size_t>(i)] = static_cast<int32_t>(i);  // slot s: pages s*mp ..
    for (int64_t s = 0; s < n; ++s) slot[static_cast<size_t>(s)] = static_cast<int32_t>(s), pos[static_cast<size_t>(s)] = static_cast<int32_t>(ctx - 1);
    // one page of random contents, copied to every page
    size_t pb;
    std::vector<uint8_t> pg;
    if (f8) {
        pb = static_cast<size_t>(fullm::kv8_page_bytes(page, Hk, D));
        pg.resize(pb);
        for (int64_t i = 0; i < page * Hk * D; ++i) pg[static_cast<size_t>(i)] = static_cast<uint8_t>(rng.next() & 0xF7u);  // no NaN
        for (int64_t i = page * Hk * D; i < static_cast<int64_t>(pb); ++i) pg[static_cast<size_t>(i)] = static_cast<uint8_t>(118 + (rng.next() % 8));
    } else {
        pb = static_cast<size_t>(page * Hk * D * 2);
        pg.resize(pb);
        for (int64_t i = 0; i < page * Hk * D; ++i) {
            const uint16_t h = bf16_of(rng.uni());
            std::memcpy(pg.data() + 2 * i, &h, 2);
        }
    }
    auto pool = [&] {
        char *d = static_cast<char *>(m.alloc(pb * static_cast<size_t>(n_pages)));
        for (int64_t i = 0; i < n_pages; ++i) be.upload(d + i * static_cast<int64_t>(pb), pg.data(), pb);
        return d;
    };
    void *kc = pool(), *vc = pool();
    std::vector<float> q(static_cast<size_t>(n * Hq * D));
    for (float &x : q) x = 0.3f * rng.uni();
    const float *qd = m.up(q);
    float *o = static_cast<float *>(m.alloc(sizeof(float) * n * Hq * D));
    const int32_t *ptd = m.up(pt), *sd = m.up(slot), *pd = m.up(pos);
    be.set_exact_activations(false);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const DType dt = f8 ? DType::FP8 : DType::BF16;
    const double ms = time_ms(be, iters, [&] {
        if (fa3 && !f8 &&
            be.gqa_prefill(o, qd, kc, vc, dt, nullptr, 0, nullptr, 0, n, true, sd, pd, ptd, mp, page, Hq, Hk, D, ctx - 1, scale) == 2)
            return;
        be.gqa_attention(o, qd, kc, vc, dt, sd, pd, ptd, mp, page, n, Hq, Hk, D, ctx - 1, scale);
    });
    be.set_exact_activations(true);
    const double bytes = double(n) * double(ctx) * double(Hk) * double(D) * 2.0 * (f8 ? 1.0 : 2.0);
    char b[256];
    std::snprintf(b, sizeof b, "attn n=%lld ctx=%lld kv=%s%s: %.1f us, %.0f GB/s of K/V\n", static_cast<long long>(n),
                  static_cast<long long>(ctx), kv.c_str(), fa3 && !f8 ? " (fa3 rows)" : "", ms * 1e3, bytes / (ms * 1e6));
    return b;
}

std::string mla(Backend &be, const Params &p) {
    const int64_t nq = geti(p, "nq", 8192), ctx = geti(p, "ctx", 100000), segs = geti(p, "segs", 1), H = geti(p, "h", 32),
                  page = geti(p, "page", 16), iters = geti(p, "iters", 3);
    const int64_t L = 512, qr = 64, qn = 128, vh = 128, W = L + qr, QK = qn + qr, N = segs * nq;
    if (nq < 1 || ctx < nq) throw std::runtime_error("mla: 1 <= nq <= ctx");
    be.set_exact_activations(false);  // mla_prefill runs with 16-bit activations
    if (be.mla_prefill_min() <= 0) {
        be.set_exact_activations(true);
        throw std::runtime_error("mla: no mla_prefill on this backend");
    }
    const int64_t mp = (ctx + page - 1) / page, n_pages = segs * mp;
    Mem m(be);
    Rng rng(11);
    std::vector<int32_t> pt(static_cast<size_t>(segs * mp));
    for (int64_t i = 0; i < segs * mp; ++i) pt[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    // one page of random latent rows, copied to every page
    std::vector<uint16_t> pg(static_cast<size_t>(page * W));
    for (uint16_t &h : pg) h = bf16_of(rng.uni());
    char *kv = static_cast<char *>(m.alloc(static_cast<size_t>(n_pages * page * W * 2)));
    for (int64_t i = 0; i < n_pages; ++i) be.upload(kv + i * page * W * 2, pg.data(), pg.size() * 2);
    std::vector<uint16_t> wb(static_cast<size_t>(H * (qn + vh) * L));
    for (uint16_t &h : wb) h = bf16_of(0.05f * rng.uni());
    Weight w;
    w.fmt = WFmt::BF16, w.rows = H * (qn + vh), w.cols = L, w.data = m.up(wb);
    std::vector<float> q(static_cast<size_t>(N * H * QK));
    for (float &x : q) x = 0.3f * rng.uni();
    float *y = static_cast<float *>(m.alloc(sizeof(float) * N * H * vh));
    std::vector<int32_t> meta;
    for (int64_t i = 0; i <= segs; ++i) meta.push_back(static_cast<int32_t>(i * nq));
    for (int64_t i = 0; i <= segs; ++i) meta.push_back(static_cast<int32_t>(i * ctx));
    for (int64_t i = 0; i < segs; ++i) meta.push_back(static_cast<int32_t>(i));
    for (int64_t i = 0; i < segs; ++i) meta.push_back(static_cast<int32_t>(i * nq));
    Backend::MlaPrefill a{};
    a.y = y, a.q = m.up(q), a.kv = kv, a.kv_dtype = DType::BF16, a.kv_b = &w;
    a.page_table = m.up(pt), a.max_pages = mp, a.page = page, a.meta = m.up(meta), a.meta_host = meta.data();
    a.n = segs, a.total_q = N, a.total_k = segs * ctx, a.max_q = nq, a.max_k = ctx;
    a.H = H, a.qn = qn, a.qr = qr, a.vh = vh, a.kv_lora = L, a.scale = 1.0f / std::sqrt(static_cast<float>(QK));
    a.chunk_keys = geti(p, "keys", 0), a.chunk_heads = geti(p, "heads", 0);
    bool ok = true;
    const double ms = time_ms(be, iters, [&] { ok = ok && be.mla_prefill(a); });
    be.set_exact_activations(true);
    if (!ok) throw std::runtime_error("mla: mla_prefill declined these shapes");
    // the attention's multiply-adds: each query over its keys (causal), q.k 192 and p.v 128 wide
    const double flops = 2.0 * double(segs) * double(H) * double(nq) * (double(ctx) - double(nq) / 2.0) * double(QK + vh);
    char b[256];
    std::snprintf(b, sizeof b, "mla nq=%lld ctx=%lld segs=%lld keys=%lld heads=%lld: %.2f ms, %.0f TFLOPS of attention\n",
                  static_cast<long long>(nq), static_cast<long long>(ctx), static_cast<long long>(segs),
                  static_cast<long long>(a.chunk_keys), static_cast<long long>(a.chunk_heads), ms, flops / (ms * 1e9));
    return b;
}

std::string moe(Backend &be, const Params &p) {
    const int64_t n = geti(p, "n", 16), E = geti(p, "e", 128), k = geti(p, "k", 8), W = geti(p, "width", 2048),
                  I = geti(p, "inter", 768), iters = geti(p, "iters", 20);
    const std::string f = gets(p, "fmt", "int4");
    const WFmt fmt = f == "bf16" ? WFmt::BF16 : f == "fp8b" ? WFmt::FP8B : f == "int4" ? WFmt::INT4G : WFmt::F32;
    if (fmt == WFmt::F32) throw std::runtime_error("moe: fmt=bf16|fp8b|int4");
    if (!be.moe_gate_up_interleaved(fmt, W, I)) throw std::runtime_error("moe: no tensor-core expert path for this shape");
    be.set_expert_format(fmt);
    Mem m(be);
    Rng rng(11);
    // one expert's random matrices, copied to every expert (distinct memory, as the real arena)
    auto bytes_of = [&](int64_t rows, int64_t cols) -> size_t {
        if (fmt == WFmt::BF16) return static_cast<size_t>(rows * cols * 2);
        if (fmt == WFmt::FP8B) return static_cast<size_t>(rows * cols);
        return static_cast<size_t>(rows * cols / 2);
    };
    auto scales_of = [&](int64_t rows, int64_t cols) -> size_t {
        if (fmt == WFmt::FP8B) return static_cast<size_t>(((rows + 127) / 128) * ((cols + 127) / 128));
        if (fmt == WFmt::INT4G) return static_cast<size_t>(rows * ((cols + 127) / 128));
        return 0;
    };
    auto host = [&](int64_t rows, int64_t cols) {
        std::vector<uint8_t> h(bytes_of(rows, cols));
        if (fmt == WFmt::BF16) {
            for (size_t i = 0; i < h.size() / 2; ++i) {
                const uint16_t v = bf16_of(0.05f * rng.uni());
                std::memcpy(h.data() + 2 * i, &v, 2);
            }
        } else {
            for (auto &b : h) b = static_cast<uint8_t>(rng.next() & (fmt == WFmt::FP8B ? 0xF7u : 0xFFu));
        }
        return h;
    };
    const auto h1 = host(2 * I, W), h2 = host(W, I);
    std::vector<float> s1(scales_of(2 * I, W), 0.01f), s2(scales_of(W, I), 0.01f);
    std::vector<Backend::ExpertW> table(static_cast<size_t>(E));
    for (int64_t e = 0; e < E; ++e) {
        auto mk = [&](const std::vector<uint8_t> &d, const std::vector<float> &s, int64_t rows, int64_t cols) {
            Weight w;
            w.fmt = fmt, w.rows = rows, w.cols = cols;
            w.data = m.up(d);
            if (!s.empty()) w.scale = m.up(s);
            return w;
        };
        table[static_cast<size_t>(e)] = Backend::ExpertW{mk(h1, s1, 2 * I, W), Weight{}, mk(h2, s2, W, I)};
    }
    const Backend::ExpertW *td = m.up(table);
    // each token's k distinct random experts, softmax-like weights
    std::vector<int32_t> idx(static_cast<size_t>(n * k));
    std::vector<float> wts(static_cast<size_t>(n * k), 1.0f / static_cast<float>(k));
    std::vector<char> used(static_cast<size_t>(E));
    int64_t distinct = 0;
    std::vector<char> any(static_cast<size_t>(E), 0);
    for (int64_t t = 0; t < n; ++t) {
        std::fill(used.begin(), used.end(), 0);
        for (int64_t j = 0; j < k; ++j) {
            int64_t e;
            do e = static_cast<int64_t>(rng.next() % static_cast<uint64_t>(E));
            while (used[static_cast<size_t>(e)]);
            used[static_cast<size_t>(e)] = 1;
            idx[static_cast<size_t>(t * k + j)] = static_cast<int32_t>(e);
            if (!any[static_cast<size_t>(e)]) any[static_cast<size_t>(e)] = 1, ++distinct;
        }
    }
    std::vector<float> x(static_cast<size_t>(n * W));
    for (float &v : x) v = rng.uni();
    const float *xd = m.up(x), *wd = m.up(wts);
    const int32_t *id = m.up(idx);
    float *contrib = static_cast<float *>(m.alloc(sizeof(float) * n * k * W));
    float *scratch = static_cast<float *>(m.alloc(sizeof(float) * n * k * I));
    be.set_exact_activations(false);
    const double ms = time_ms(be, iters, [&] {
        be.moe_experts(contrib, xd, id, wd, n, k, td, E, fmt, W, I, true, 0.0f, 0.0f, scratch);
    });
    be.set_exact_activations(true);
    const double bytes = double(distinct) * double(bytes_of(2 * I, W) + bytes_of(W, I));
    char b[256];
    std::snprintf(b, sizeof b, "moe n=%lld fmt=%s: %.1f us, %lld experts, %.0f GB/s of weights\n", static_cast<long long>(n),
                  f.c_str(), ms * 1e3, static_cast<long long>(distinct), bytes / (ms * 1e6));
    return b;
}

std::string pagecopy(Backend &be, const Params &p) {
    const int64_t pages = geti(p, "pages", 2048), L = geti(p, "layers", 48), Hk = geti(p, "hkv", 4), D = geti(p, "d", 128),
                  page = geti(p, "page", 16), iters = geti(p, "iters", 5);
    const size_t seg = static_cast<size_t>(page * Hk * D * 2);
    const int n_pools = static_cast<int>(2 * L);
    Mem m(be);
    // one allocation at a fixed stride, as the dense executor lays its pools out (separate=1:
    // one allocation per pool, the batched-copy path)
    std::vector<void *> pools(static_cast<size_t>(n_pools));
    const size_t stride = (seg * static_cast<size_t>(pages) + 255) & ~size_t(255);
    if (geti(p, "separate", 0)) {
        for (auto &q : pools) q = m.alloc(seg * static_cast<size_t>(pages));
    } else {
        char *base = static_cast<char *>(m.alloc(stride * n_pools));
        for (int s = 0; s < n_pools; ++s) pools[static_cast<size_t>(s)] = base + stride * s;
    }
    char *host = static_cast<char *>(be.host_alloc(seg * n_pools * static_cast<size_t>(pages)));
    std::vector<void *> hp(static_cast<size_t>(pages));
    std::vector<int32_t> dev(static_cast<size_t>(pages));
    for (int64_t i = 0; i < pages; ++i) {
        hp[static_cast<size_t>(i)] = host + static_cast<size_t>(i) * seg * n_pools;
        dev[static_cast<size_t>(i)] = static_cast<int32_t>(pages - 1 - i);  // scattered, as a real load is
    }
    auto run = [&](bool to_host) {
        be.copy_pages(to_host, pools.data(), n_pools, seg, dev.data(), hp.data(), pages, nullptr);
        void *ev = be.record_copy_event();
        be.wait_event(ev);
        be.release_event(ev);
    };
    auto ms_of = [&](bool to_host) {
        run(to_host);
        const auto t0 = std::chrono::steady_clock::now();
        for (int64_t i = 0; i < iters; ++i) run(to_host);
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / double(iters);
    };
    const double d2h = ms_of(true), h2d = ms_of(false);
    be.host_release(host);
    const double bytes = double(seg) * n_pools * double(pages);
    char b[256];
    std::snprintf(b, sizeof b, "pagecopy %lld pages (%.2f GB): to host %.1f ms (%.1f GB/s), back %.1f ms (%.1f GB/s)\n",
                  static_cast<long long>(pages), bytes / 1e9, d2h, bytes / (d2h * 1e6), h2d, bytes / (h2d * 1e6));
    return b;
}

// gemv n=1 fmt=bf16|int8|fp8 rows=19456 cols=2560 iters=200: the dense trunk's GEMV (gemm with n rows
// of x; n <= 16 is the tensor-core GEMV, above it cuBLAS / the fp8 GEMM) over a random matrix of the
// given shape (default Qwen3-4B's gate/up). Reports the op's time and the weight bytes per second.
std::string gemv(Backend &be, const Params &p) {
    const int64_t n = geti(p, "n", 1), rows = geti(p, "rows", 19456), cols = geti(p, "cols", 2560),
                  iters = geti(p, "iters", 200);
    const std::string fmt = gets(p, "fmt", "fp8");
    if (rows <= 0 || cols <= 0 || cols % 16) throw std::runtime_error("gemv: rows > 0, cols a multiple of 16");
    Mem m(be);
    Rng rng(11);
    Weight w;
    w.rows = rows, w.cols = cols;
    const size_t ne = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    std::vector<float> scale(static_cast<size_t>(rows));
    for (float &s : scale) s = 0.002f + 0.001f * (rng.uni() + 1.0f);
    double wbytes;
    if (fmt == "bf16") {
        std::vector<uint16_t> h(ne);
        for (uint16_t &v : h) v = bf16_of(0.05f * rng.uni());
        w.fmt = WFmt::BF16, w.data = m.up(h), wbytes = 2.0 * double(ne);
    } else if (fmt == "int8") {
        std::vector<int8_t> q(ne);
        for (int8_t &v : q) v = static_cast<int8_t>(static_cast<int>(rng.next() % 255) - 127);
        w.fmt = WFmt::INT8, w.data = m.up(q), w.scale = m.up(scale), wbytes = double(ne);
    } else if (fmt == "fp8") {
        std::vector<uint8_t> q(ne);
        for (uint8_t &v : q) v = static_cast<uint8_t>(rng.next() & 0xF7u);  // no NaN (0x7F / 0xFF)
        w.fmt = WFmt::FP8, w.data = m.up(q), w.scale = m.up(scale), wbytes = double(ne);
    } else {
        throw std::runtime_error("gemv: fmt=bf16|int8|fp8");
    }
    std::vector<float> x(static_cast<size_t>(n * cols));
    for (float &v : x) v = rng.uni();
    const float *xd = m.up(x);
    float *y = static_cast<float *>(m.alloc(sizeof(float) * static_cast<size_t>(n * rows)));
    be.set_exact_activations(false);
    const double ms = time_ms(be, iters, [&] { be.gemm(w, xd, n, y, rows); });
    be.set_exact_activations(true);
    char b[256];
    std::snprintf(b, sizeof b, "gemv n=%lld fmt=%s %lldx%lld: %.1f us, %.0f GB/s of weights\n", static_cast<long long>(n),
                  fmt.c_str(), static_cast<long long>(rows), static_cast<long long>(cols), ms * 1000.0,
                  wbytes / (ms * 1e-3) / 1e9);
    return b;
}

const char *kDefault =
    "attn n=16 ctx=8000 kv=bf16; attn n=16 ctx=8000 kv=bf16 fa3=1; attn n=16 ctx=8000 kv=fp8; attn n=1 ctx=8000 kv=fp8;"
    "moe n=16 fmt=int4; moe n=16 fmt=fp8b; moe n=1 fmt=int4";

}  // namespace

extern "C" int sk_exec_kbench(int32_t device, const char *spec, char *report, int report_len) {
    std::string out;
    int rc = 0;
    try {
        if (!fullm::cuda_available()) throw std::runtime_error("CUDA is not available");
        auto be = fullm::make_cuda_backend(device);
        std::stringstream all(spec && *spec ? spec : kDefault);
        std::string one;
        while (std::getline(all, one, ';')) {
            std::stringstream ss(one);
            std::string op, kv;
            if (!(ss >> op)) continue;
            Params p;
            while (ss >> kv) {
                const auto eq = kv.find('=');
                if (eq == std::string::npos) throw std::runtime_error("kbench: expected key=value, got " + kv);
                p[kv.substr(0, eq)] = kv.substr(eq + 1);
            }
            try {
                if (op == "attn") out += attn(*be, p);
                else if (op == "moe") out += moe(*be, p);
                else if (op == "pagecopy") out += pagecopy(*be, p);
                else if (op == "gemv") out += gemv(*be, p);
                else if (op == "mla") out += mla(*be, p);
                else throw std::runtime_error("unknown op " + op + " (attn, moe, pagecopy, gemv, mla)");
            } catch (const std::exception &e) {
                out += op + ": " + e.what() + "\n";
                rc = 1;
            }
        }
    } catch (const std::exception &e) {
        out += std::string("kbench: ") + e.what() + "\n";
        rc = -1;
    }
    if (report && report_len > 0) {
        const size_t n = std::min(out.size(), static_cast<size_t>(report_len) - 1);
        std::memcpy(report, out.data(), n);
        report[n] = 0;
    }
    return rc;
}
