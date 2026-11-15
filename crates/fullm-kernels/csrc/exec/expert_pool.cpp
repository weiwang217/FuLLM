// SPDX-License-Identifier: Apache-2.0
#include "expert_pool.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "fullm_kernels.h"

namespace fullm {

namespace {
// Rows of an expert matrix per task, at most: a few hundred KB of MXFP4 each, enough to amortise
// the task. Fewer when the step has few experts here (rows_per_task).
constexpr int64_t kRowsPerTask = 128;
// A phase that moves fewer bytes than this runs on the caller: a parallel_for's wake-up and join
// cost more than copying a decode step's few rows.
constexpr int64_t kSerialBytes = int64_t{256} << 10;

inline float sigm(float x) { return 1.0f / (1.0f + std::exp(-x)); }
}  // namespace

ExpertPool::ExpertPool(int threads, ExpertDims dims) : pool_(threads), d_(dims) {
    if (!d_.bf16 && (d_.latent % 32 || d_.inter % 32))
        throw std::runtime_error("expert widths must be multiples of the MXFP4 group (32)");
}

int64_t ExpertPool::expert_bytes() const {
    const int64_t params = 3 * d_.latent * d_.inter;
    return d_.bf16 ? 2 * params : params / 2 + params / 32;
}

namespace {
// y[m][n0..n1) = x[m] . W[n] for one expert matrix in either storage format.
inline void expert_gemm(bool bf16, const float *x, int64_t M, int64_t K, const uint8_t *p, const uint8_t *s, int64_t n0,
                        int64_t n1, float *y, int64_t ldy) {
    if (bf16) sk_cpu_gemm_bf16(x, M, K, reinterpret_cast<const uint16_t *>(p), n0, n1, y, ldy);
    else sk_cpu_gemm_mxfp4(x, M, K, p, s, n0, n1, y, ldy);
}
}  // namespace

int64_t ExpertPool::run(const std::vector<ExpertPtrs> &experts, const float *z, int64_t N, const int32_t *idx,
                        const float *w, float *acc) {
    const int64_t L = d_.latent, k = d_.topk, pairs = N * k;
    all_.resize(pairs);
    for (int64_t p = 0; p < pairs; ++p) all_[p] = static_cast<int32_t>(p);
    rows_.resize(pairs * L);
    const int64_t active = run_pairs(experts, z, N, idx, w, all_.data(), pairs, rows_.data());
    // (0 + r0) + r1 + ... in selection order: the same additions moe_reduce performs, so a
    // layer gives the same bits whether its experts were all here or split with the GPU.
    pool_.parallel_for(N, [&](int64_t t) {
        float *a = acc + t * L;
        for (int64_t i = 0; i < L; ++i) a[i] = 0.0f;
        for (int64_t j = 0; j < k; ++j) {
            const float *r = rows_.data() + (t * k + j) * L;
            for (int64_t i = 0; i < L; ++i) a[i] = a[i] + r[i];
        }
    });
    return active;
}

int64_t ExpertPool::run_pairs(const std::vector<ExpertPtrs> &experts, const float *z, int64_t N, const int32_t *idx,
                              const float *w, const int32_t *sel, int64_t nsel, float *rows) {
    const int64_t L = d_.latent, I = d_.inter, k = d_.topk, E = d_.n_experts;
    if (nsel == 0) return 0;

    // CSR of the selected pairs by expert; slot r in the CSR order is selected pair order_[r].
    first_.assign(E + 1, 0);
    for (int64_t i = 0; i < nsel; ++i) {
        const int64_t p = sel[i];
        if (p < 0 || p >= N * k) throw std::runtime_error("selected pair out of range");
        const int32_t e = idx[p];
        if (e < 0 || e >= E) throw std::runtime_error("expert id " + std::to_string(e) + " out of range");
        first_[e + 1]++;
    }
    for (int64_t e = 0; e < E; ++e) first_[e + 1] += first_[e];
    order_.resize(nsel);
    {
        std::vector<int32_t> fill(first_.begin(), first_.end() - 1);
        for (int64_t i = 0; i < nsel; ++i) order_[fill[idx[sel[i]]]++] = static_cast<int32_t>(i);
    }
    std::vector<int32_t> active;
    for (int64_t e = 0; e < E; ++e)
        if (first_[e + 1] > first_[e]) {
            const ExpertPtrs &x = experts[e];
            if (!x.p1 || !x.p3 || !x.p2 || (!d_.bf16 && (!x.s1 || !x.s3 || !x.s2)))
                throw std::runtime_error("routed expert " + std::to_string(e) + " has no weights bound");
            active.push_back(static_cast<int32_t>(e));
        }

    if (static_cast<int64_t>(x_.size()) < nsel * L) {
        x_.resize(nsel * L);
        out_.resize(nsel * L);
    }
    if (static_cast<int64_t>(g_.size()) < nsel * I) {
        g_.resize(nsel * I);
        u_.resize(nsel * I);
    }
    auto gather = [&](int64_t r) {
        const int64_t t = sel[order_[r]] / k;
        std::memcpy(x_.data() + r * L, z + t * L, L * 4);
    };
    if (nsel * L * 4 < kSerialBytes) {
        for (int64_t r = 0; r < nsel; ++r) gather(r);
    } else {
        pool_.parallel_for(nsel, gather);
    }

    // Rows per task: about four tasks a thread over the step's experts (a decode step's eight
    // experts in 128-row tasks gave 26 threads 64 tasks: three rounds, the last half idle), in
    // multiples of 16, between 16 and kRowsPerTask. A row's result does not depend on its task.
    const int64_t n_act = static_cast<int64_t>(active.size());
    auto rows_per_task = [&](int64_t rows) {
        const int64_t want = 4 * static_cast<int64_t>(pool_.size());
        const int64_t r = (n_act * rows + want - 1) / want;
        return std::clamp<int64_t>((r + 15) / 16 * 16, 16, kRowsPerTask);
    };

    // Phase 1: gate and up, then the activation in place of gate (g_ becomes the activation).
    const int64_t rows_i = rows_per_task(I), tiles_i = (I + rows_i - 1) / rows_i;
    pool_.parallel_for(n_act * tiles_i, [&](int64_t task) {
        const int64_t e = active[task / tiles_i], n0 = (task % tiles_i) * rows_i;
        const int64_t n1 = std::min(I, n0 + rows_i);
        const int64_t r0 = first_[e], m = first_[e + 1] - r0;
        const ExpertPtrs &x = experts[e];
        float *g = g_.data() + r0 * I, *u = u_.data() + r0 * I;
        const float *xin = x_.data() + r0 * L;
        expert_gemm(d_.bf16, xin, m, L, x.p1, x.s1, n0, n1, g, I);
        expert_gemm(d_.bf16, xin, m, L, x.p3, x.s3, n0, n1, u, I);
        for (int64_t r = 0; r < m; ++r)
            for (int64_t n = n0; n < n1; ++n) {
                const float gv = g[r * I + n];
                if (d_.silu) {
                    g[r * I + n] = gv * sigm(gv) * u[r * I + n];
                } else {
                    const float a = d_.b1 * std::tanh(gv / d_.b1) * sigm(gv);
                    g[r * I + n] = a * (d_.b2 * std::tanh(u[r * I + n] / d_.b2));
                }
            }
    });

    // Phase 2: down projection.
    const int64_t rows_l = rows_per_task(L), tiles_l = (L + rows_l - 1) / rows_l;
    pool_.parallel_for(n_act * tiles_l, [&](int64_t task) {
        const int64_t e = active[task / tiles_l], n0 = (task % tiles_l) * rows_l;
        const int64_t n1 = std::min(L, n0 + rows_l);
        const int64_t r0 = first_[e], m = first_[e + 1] - r0;
        const ExpertPtrs &x = experts[e];
        expert_gemm(d_.bf16, g_.data() + r0 * I, m, I, x.p2, x.s2, n0, n1, out_.data() + r0 * L, L);
    });

    // Phase 3: weighted rows back in selection order.
    auto scatter = [&](int64_t r) {
        const int64_t i = order_[r];
        const float wp = w[sel[i]];
        const float *d = out_.data() + r * L;
        float *dst = rows + i * L;
        for (int64_t c = 0; c < L; ++c) dst[c] = wp * d[c];
    };
    if (nsel * L * 4 < kSerialBytes) {
        for (int64_t r = 0; r < nsel; ++r) scatter(r);
    } else {
        pool_.parallel_for(nsel, scatter);
    }
    return static_cast<int64_t>(active.size());
}

}  // namespace fullm
