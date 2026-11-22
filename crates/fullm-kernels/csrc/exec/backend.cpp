// SPDX-License-Identifier: Apache-2.0
// backend.cpp - Backend's base forms of the fast paths that have no simpler fallback in the
// executors: built from core ops only, so a new backend runs every model
// family it declares before it has a kernel of its own for them. Correct, not fast; the
// conformance suite checks each backend's own form against these.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <vector>

#include "backend.h"
#include "kv_int4.h"
#include "kv_fp8.h"

namespace fullm {

namespace {

// PCG-XSH-RR 64/32, seeded as runtime/sampler.rs's Pcg32::new(seed, stream).
struct Pcg32 {
    uint64_t state = 0, inc;
    Pcg32(uint64_t seed, uint64_t stream) : inc((stream << 1) | 1) {
        next();
        state += seed;
        next();
    }
    uint32_t next() {
        const uint64_t old = state;
        state = old * 6364136223846793005ull + inc;
        const uint32_t x = static_cast<uint32_t>(((old >> 18) ^ old) >> 27), rot = static_cast<uint32_t>(old >> 59);
        return (x >> rot) | (x << ((32 - rot) & 31));
    }
    double next_f64() {
        const uint64_t hi = uint64_t(next()) << 21, lo = next() >> 11;
        return double(hi | lo) / double(1ull << 53);
    }
};

// Larger float, larger key (finite values).
uint32_t fkey(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

}  // namespace

int32_t sample_row_host(const float *x, int64_t n, const SampleRow &p) {
    if (n <= 0) return 0;
    int64_t best = 0;
    for (int64_t i = 1; i < n; ++i)
        if (x[i] > x[best]) best = i;
    // a row with nothing allowed (every logit -inf) draws its first token
    if (!(p.temperature > 0.0f) || n == 1 || x[best] == -INFINITY) return static_cast<int32_t>(best);
    const float m = x[best], inv_t = static_cast<float>(1.0 / static_cast<double>(p.temperature));
    // top-k: the k-th largest value; every value equal to it stays
    uint32_t lo = 0;
    if (p.top_k > 0 && p.top_k < n) {
        std::vector<uint32_t> keys(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) keys[size_t(i)] = fkey(x[i]);
        std::nth_element(keys.begin(), keys.begin() + (p.top_k - 1), keys.end(), std::greater<uint32_t>());
        lo = keys[size_t(p.top_k - 1)];
    }
    // min-p: the values at least the largest + the cut
    if (p.min_p_cut < 0.0f) lo = std::max(lo, fkey(m + p.min_p_cut));
    std::vector<float> w(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        w[size_t(i)] = fkey(x[i]) >= lo ? static_cast<float>(std::exp(static_cast<double>((x[i] - m) * inv_t))) : 0.0f;
    auto mass = [&](uint32_t t) {
        double s = 0.0;
        for (int64_t i = 0; i < n; ++i)
            if (fkey(x[i]) >= t) s += w[size_t(i)];
        return s;
    };
    // top-p: the largest threshold whose candidates hold at least top_p of the mass
    uint32_t t = lo;
    if (p.top_p < 1.0f) {
        const double want = static_cast<double>(p.top_p) * mass(lo);
        uint32_t a = lo, b = fkey(m);
        while (a < b) {
            const uint32_t mid = a + (b - a + 1) / 2;
            if (mass(mid) >= want) a = mid;
            else b = mid - 1;
        }
        t = a;
    }
    double r = Pcg32(p.seed, p.step).next_f64() * mass(t);
    int64_t last = best;
    for (int64_t i = 0; i < n; ++i) {
        if (fkey(x[i]) < t) continue;
        last = i;
        r -= w[size_t(i)];
        if (r < 0.0) return static_cast<int32_t>(i);
    }
    return static_cast<int32_t>(last);
}

void adjust_row_host(float *x, const SampleRow &p, const LogitAdj *adj) {
    for (int32_t j = 0; j < p.adj_n; ++j) {
        const LogitAdj &a = adj[p.adj_off + j];
        float v = x[a.token] + a.bias;  // one rounding a statement: no fused multiply-add
        v = v > 0.0f ? v / a.rep : v * a.rep;
        v = v - a.sub;
        x[a.token] = v;
    }
}

void logprob_row_host(float *lp, int32_t *ids, const float *x, int64_t n, int32_t picked, int32_t k) {
    float m = -INFINITY;
    for (int64_t i = 0; i < n; ++i) m = std::max(m, x[i]);
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) s += std::exp(static_cast<double>(x[i] - m));
    const double lse = static_cast<double>(m) + std::log(s);
    lp[0] = static_cast<float>(static_cast<double>(x[picked]) - lse);
    const int64_t top = std::min<int64_t>(std::min<int64_t>(k - 1, kMaxTopLogprobs), n);
    if (top <= 0) return;
    std::vector<int32_t> order(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) order[size_t(i)] = static_cast<int32_t>(i);
    std::partial_sort(order.begin(), order.begin() + top, order.end(),
                      [&](int32_t a, int32_t b) { return x[a] > x[b] || (x[a] == x[b] && a < b); });
    for (int64_t j = 0; j < top; ++j) {
        ids[j] = order[size_t(j)];
        lp[1 + j] = static_cast<float>(static_cast<double>(x[order[size_t(j)]]) - lse);
    }
}

void Backend::adjust_rows(float *x, int64_t rows, int64_t n, const SampleRow *p, const LogitAdj *adj) {
    if (rows <= 0) return;
    std::vector<SampleRow> hp(static_cast<size_t>(rows));
    download(hp.data(), p, hp.size() * sizeof(SampleRow));
    sync();
    int64_t total = 0;
    for (const SampleRow &s : hp) total = std::max<int64_t>(total, int64_t(s.adj_off) + s.adj_n);
    if (total == 0) return;
    std::vector<LogitAdj> ha(static_cast<size_t>(total));
    download(ha.data(), adj, ha.size() * sizeof(LogitAdj));
    std::vector<float> h(static_cast<size_t>(n));
    sync();
    for (int64_t r = 0; r < rows; ++r) {
        if (hp[size_t(r)].adj_n <= 0) continue;
        download(h.data(), x + r * n, h.size() * 4);
        sync();
        adjust_row_host(h.data(), hp[size_t(r)], ha.data());
        upload(x + r * n, h.data(), h.size() * 4);
    }
    sync();
}

void Backend::logprob_rows(float *lp, int32_t *ids, const float *x, int64_t rows, int64_t n, const SampleRow *p,
                           const int32_t *idx) {
    if (rows <= 0) return;
    std::vector<SampleRow> hp(static_cast<size_t>(rows));
    std::vector<int32_t> hi(static_cast<size_t>(rows));
    download(hp.data(), p, hp.size() * sizeof(SampleRow));
    download(hi.data(), idx, hi.size() * 4);
    // rows and entries not asked for keep what they held, as on a backend that writes only the rest
    std::vector<float> h(static_cast<size_t>(n)), hl(size_t(rows) * (kMaxTopLogprobs + 1));
    std::vector<int32_t> hid(size_t(rows) * kMaxTopLogprobs);
    download(hl.data(), lp, hl.size() * 4);
    download(hid.data(), ids, hid.size() * 4);
    sync();
    for (int64_t r = 0; r < rows; ++r) {
        if (hp[size_t(r)].logprobs <= 0) continue;
        download(h.data(), x + r * n, h.size() * 4);
        sync();
        logprob_row_host(&hl[size_t(r) * (kMaxTopLogprobs + 1)], &hid[size_t(r) * kMaxTopLogprobs], h.data(), n,
                         hi[size_t(r)], hp[size_t(r)].logprobs);
    }
    upload(lp, hl.data(), hl.size() * 4);
    upload(ids, hid.data(), hid.size() * 4);
    sync();
}

void Backend::lora(const LoraJob *jobs, int n, const float *x, int64_t N, int64_t in, int64_t rank,
                   const int32_t *slot, float * /*xa*/) {
    if (N <= 0 || n <= 0) return;
    std::vector<int32_t> hs(static_cast<size_t>(N));
    download(hs.data(), slot, hs.size() * 4);
    sync();
    int32_t slots = 0;
    for (int32_t s : hs) slots = std::max(slots, s + 1);
    if (slots == 0) return;
    std::vector<float> hx(size_t(N) * in);
    download(hx.data(), x, hx.size() * 4);
    sync();
    auto f = [](uint16_t h) {
        uint32_t u = uint32_t(h) << 16;
        float w;
        std::memcpy(&w, &u, 4);
        return w;
    };
    for (int t = 0; t < n; ++t) {
        const LoraJob &J = jobs[t];
        const int64_t out = J.out;
        std::vector<float> hy(size_t(N) * out), hsc(static_cast<size_t>(slots)), xa(static_cast<size_t>(rank));
        std::vector<int32_t> hr(static_cast<size_t>(slots));
        std::vector<uint16_t> ha(size_t(slots) * J.a_stride), hb(size_t(slots) * out * J.ldb);
        download(hy.data(), J.y, hy.size() * 4);
        download(ha.data(), J.a, ((slots - 1) * J.a_stride + rank * in) * 2);
        download(hb.data(), J.b, hb.size() * 2);
        download(hsc.data(), J.scale, hsc.size() * 4);
        download(hr.data(), J.ranks, hr.size() * 4);
        sync();
        for (int64_t r = 0; r < N; ++r) {
            const int32_t s = hs[size_t(r)];
            if (s < 0) continue;
            const int64_t kr = std::min<int64_t>(hr[size_t(s)], rank);
            const uint16_t *A = &ha[size_t(s * J.a_stride)], *B = &hb[size_t(s) * out * J.ldb];
            for (int64_t k = 0; k < kr; ++k) {
                float acc = 0.0f;
                for (int64_t i = 0; i < in; ++i) acc += f(A[k * in + i]) * hx[size_t(r * in + i)];
                xa[size_t(k)] = acc;
            }
            for (int64_t o = 0; o < out; ++o) {
                float acc = 0.0f;
                for (int64_t k = 0; k < kr; ++k) acc += f(B[o * J.ldb + k]) * xa[size_t(k)];
                hy[size_t(r * out + o)] += hsc[size_t(s)] * acc;
            }
        }
        upload(J.y, hy.data(), hy.size() * 4);
        sync();
    }
}

void Backend::mask_rows(float *x, int32_t *bad, int64_t rows, int64_t n, const SampleRow *p, const uint32_t *masks,
                        int64_t words) {
    if (rows <= 0) return;
    std::vector<SampleRow> hp(static_cast<size_t>(rows));
    download(hp.data(), p, hp.size() * sizeof(SampleRow));
    sync();
    int32_t n_masks = 0;
    for (const SampleRow &s : hp) n_masks = std::max(n_masks, s.mask);
    if (n_masks == 0) return;
    std::vector<uint32_t> hm(size_t(n_masks) * size_t(words));
    std::vector<float> h(static_cast<size_t>(n));
    download(hm.data(), masks, hm.size() * 4);
    sync();
    bool nonfinite = false;
    for (int64_t r = 0; r < rows; ++r) {
        if (hp[size_t(r)].mask <= 0) continue;
        const uint32_t *m = &hm[size_t(hp[size_t(r)].mask - 1) * size_t(words)];
        download(h.data(), x + r * n, h.size() * 4);
        sync();
        for (int64_t i = 0; i < n; ++i) {
            nonfinite |= !std::isfinite(h[size_t(i)]);
            if (!(m[i >> 5] >> (i & 31) & 1u)) h[size_t(i)] = -INFINITY;
        }
        upload(x + r * n, h.data(), h.size() * 4);
    }
    if (nonfinite) {
        const int32_t one = 1;
        upload(bad, &one, 4);
    }
    sync();
}

void Backend::sample_rows(int32_t *idx, int32_t *bad, const float *x, int64_t rows, int64_t n, const SampleRow *p) {
    if (rows <= 0) return;
    std::vector<float> h(static_cast<size_t>(rows * n));
    std::vector<SampleRow> hp(static_cast<size_t>(rows));
    std::vector<int32_t> out(static_cast<size_t>(rows));
    download(h.data(), x, h.size() * 4);
    download(hp.data(), p, hp.size() * sizeof(SampleRow));
    sync();
    bool nonfinite = false;
    for (int64_t r = 0; r < rows; ++r) {
        const bool masked = hp[size_t(r)].mask > 0 || hp[size_t(r)].adj_n > 0;
        for (int64_t i = 0; i < n; ++i) {
            const float v = h[size_t(r * n + i)];
            nonfinite |= !std::isfinite(v) && !(masked && v == -INFINITY);
        }
    }
    if (nonfinite) {
        const int32_t one = 1;
        upload(bad, &one, 4);
    }
    for (int64_t r = 0; r < rows; ++r) out[size_t(r)] = nonfinite ? 0 : sample_row_host(&h[size_t(r * n)], n, hp[size_t(r)]);
    upload(idx, out.data(), out.size() * 4);
    sync();
}

size_t Backend::kv_extra_bytes(DType dt, int64_t slots, int64_t Hkv, int64_t) const {
    return dt == DType::KV4 ? static_cast<size_t>(kv4_ring_bytes(slots, Hkv)) : 0;
}

size_t Backend::kv_page_bytes(DType dt, int64_t page, int64_t Hkv, int64_t D) const {
    switch (dt) {
        case DType::FP8: return static_cast<size_t>(kv8_page_bytes(page, Hkv, D));  // values, then each row's exponents
        case DType::FP8E4M3: return static_cast<size_t>(page * Hkv * D);
        case DType::BF16: return static_cast<size_t>(page * Hkv * D) * 2;
        case DType::KV4: return static_cast<size_t>(kv4_page_bytes(page, Hkv));
        case DType::F32: break;
    }
    return static_cast<size_t>(page * Hkv * D) * 4;
}

void Backend::moe_experts(float *contrib, const float *x, const int32_t *idx, const float *w, int64_t N, int64_t k,
                          const ExpertW *table, int64_t n_experts, WFmt /*fmt*/, int64_t width, int64_t inter, bool silu,
                          float b1, float b2, float * /*scratch*/) {
    const int64_t pairs = N * k;
    if (pairs <= 0 || n_experts <= 0) return;
    // the routing and the table on the host
    std::vector<int32_t> hidx(static_cast<size_t>(pairs));
    std::vector<float> hw(static_cast<size_t>(pairs));
    std::vector<ExpertW> tab(static_cast<size_t>(n_experts));
    download(hidx.data(), idx, sizeof(int32_t) * pairs);
    download(hw.data(), w, sizeof(float) * pairs);
    download(tab.data(), table, sizeof(ExpertW) * n_experts);
    sync();
    std::vector<std::vector<int32_t>> by(static_cast<size_t>(n_experts));
    size_t most = 0;
    for (int64_t p = 0; p < pairs; ++p) {
        const int32_t e = hidx[static_cast<size_t>(p)];
        if (e < 0 || e >= n_experts || !tab[static_cast<size_t>(e)].w1.data) continue;  // not resident here
        by[static_cast<size_t>(e)].push_back(static_cast<int32_t>(p));
        most = std::max(most, by[static_cast<size_t>(e)].size());
    }
    if (!most) return;
    // one expert's rows at a time: X [m][width], gate G and up U [m][inter] (the GLU into G), D [m][width]
    const size_t m_max = most;
    float *X = static_cast<float *>(alloc(sizeof(float) * m_max * width));
    float *G = static_cast<float *>(alloc(sizeof(float) * m_max * inter));
    float *U = static_cast<float *>(alloc(sizeof(float) * m_max * inter));
    float *A = static_cast<float *>(alloc(sizeof(float) * m_max * inter));
    float *D = static_cast<float *>(alloc(sizeof(float) * m_max * width));
    std::vector<float> hd(m_max * static_cast<size_t>(width));
    try {
        for (int64_t e = 0; e < n_experts; ++e) {
            const std::vector<int32_t> &ps = by[static_cast<size_t>(e)];
            if (ps.empty()) continue;
            const ExpertW &ew = tab[static_cast<size_t>(e)];
            if (!ew.w3.data) unsupported("moe_experts (base form) on gate/up interleaved experts");
            const int64_t m = static_cast<int64_t>(ps.size());
            for (int64_t i = 0; i < m; ++i)  // token t = p / k
                copy(X + i * width, x + (ps[static_cast<size_t>(i)] / k) * width, sizeof(float) * width);
            gemm(ew.w1, X, m, G, inter);
            gemm(ew.w3, X, m, U, inter);
            if (silu) silu_glu(A, G, U, m * inter);
            else situ_glu(A, G, U, m * inter, b1, b2);
            gemm(ew.w2, A, m, D, width);
            // contrib[p] = w[p] * D[i], on the host
            download(hd.data(), D, sizeof(float) * m * width);
            sync();
            for (int64_t i = 0; i < m; ++i) {
                const int32_t p = ps[static_cast<size_t>(i)];
                float *row = hd.data() + i * width;
                const float wp = hw[static_cast<size_t>(p)];
                for (int64_t c = 0; c < width; ++c) row[c] = wp * row[c];
                upload(contrib + static_cast<int64_t>(p) * width, row, sizeof(float) * width);
            }
        }
        sync();
    } catch (...) {
        release(X), release(G), release(U), release(A), release(D);
        throw;
    }
    release(X), release(G), release(U), release(A), release(D);
}

}  // namespace fullm
