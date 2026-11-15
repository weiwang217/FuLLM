// SPDX-License-Identifier: Apache-2.0
// torch_ops.cpp - the torch backend: the CPU backend with every bf16 GEMM computed by
// libtorch, the way vLLM computes it on a CPU (at::linear on bf16 x and W, bf16 out). Built
// only with the `torch` feature, against the libtorch of a Python torch install
// (FULLM_TORCH_DIR), so a comparison with vLLM can run the very same GEMM kernels.

#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>

#include "backend.h"

namespace fullm {
namespace {

bool torch_gemm(const Weight &w, const float *x, int64_t M, float *y, int64_t ldy) {
    if (w.fmt != WFmt::BF16) return false;  // other formats: the native kernels
    c10::InferenceMode guard;
    const at::Tensor W = at::from_blob(const_cast<void *>(w.data), {w.rows, w.cols}, at::kBFloat16);
    const at::Tensor X = at::from_blob(const_cast<float *>(x), {M, w.cols}, at::kFloat).to(at::kBFloat16);
    at::Tensor Y = at::from_blob(y, {M, w.rows}, {ldy, 1}, at::kFloat);
    Y.copy_(at::linear(X, W));
    return true;
}

}  // namespace

std::unique_ptr<Backend> make_torch_backend(int threads) {
    return make_cpu_backend_with_gemm(threads, torch_gemm, "cpu+torch");
}

}  // namespace fullm
