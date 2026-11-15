// SPDX-License-Identifier: Apache-2.0
// expert_pool.h - routed experts computed on the CPU, next to the bytes.
//
// At decode a Kimi K3 token touches 25.8 GB of MXFP4 experts. Reading them in host DRAM
// at hundreds of GB/s beats copying them over PCIe at ~50 GB/s, so the GPU sends the
// latent activations and routing decisions here and receives the weighted sum back.
//
// The work for one layer runs in three parallel phases:
//   1. gate/up: tasks over (expert, block of intermediate rows), all tokens of the expert
//   2. down:    tasks over (expert, block of latent rows)
//   3. reduce:  per token, sum_j w_j * d_j in SELECTION order
// Each (token, choice) has its own output row, so the reduction order never depends on
// how experts were scheduled or what else shared the step: the result is deterministic
// and batch-invariant, like the rest of the CPU path.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "threadpool.h"

namespace fullm {

// MXFP4: p* are packed nibbles and s* E8M0 scales. bf16: p* are the bf16 elements and s* null.
struct ExpertPtrs {
    const uint8_t *p1 = nullptr, *s1 = nullptr;  // gate [inter][width]
    const uint8_t *p3 = nullptr, *s3 = nullptr;  // up   [inter][width]
    const uint8_t *p2 = nullptr, *s2 = nullptr;  // down [width][inter]
};

// width is the latent (Kimi K3) or the hidden size (Kimi Linear).
struct ExpertDims {
    int64_t latent = 0, inter = 0, topk = 0, n_experts = 0;
    float b1 = 4.0f, b2 = 25.0f;  // SiTU-GLU caps
    bool silu = false;            // SiLU-GLU instead of SiTU-GLU
    bool bf16 = false;            // bf16 experts instead of MXFP4
};

class ExpertPool {
public:
    ExpertPool(int threads, ExpertDims dims);
    int threads() const { return pool_.size(); }
    ThreadPool &pool() { return pool_; }

    // acc[N][latent] = sum_j w[t][j] * expert_{idx[t][j]}(z[t]), for one layer's experts.
    // Returns the number of distinct experts touched. Throws if an expert pointer is null.
    int64_t run(const std::vector<ExpertPtrs> &experts, const float *z, int64_t N,
                const int32_t *idx, const float *w, float *acc);

    // The pairs p = t*k + j listed in sel[0..nsel) (ascending): rows[i] = w[p] *
    // expert_{idx[p]}(z[p / k]). The building block when some experts live on the GPU.
    int64_t run_pairs(const std::vector<ExpertPtrs> &experts, const float *z, int64_t N,
                      const int32_t *idx, const float *w, const int32_t *sel, int64_t nsel,
                      float *rows);

    int64_t expert_bytes() const;

private:
    ThreadPool pool_;
    ExpertDims d_;
    // grow-only scratch: [N*topk][latent] inputs/outputs, [N*topk][inter] gate/up/act
    std::vector<float> x_, g_, u_, out_;
    std::vector<int32_t> order_, first_;  // pairs sorted by expert, CSR offsets
    std::vector<int32_t> all_;            // 0..N*k for run()
    std::vector<float> rows_;             // per-pair rows for run()
};

}  // namespace fullm
