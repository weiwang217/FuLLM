// SPDX-License-Identifier: Apache-2.0
// cuda_stub.cpp - compiled instead of csrc/exec/cuda/*.cu when the `cuda` feature is off,
// so a CPU-only build links and reports CUDA as unavailable rather than failing.

#include <stdexcept>

#include "backend.h"
#include "fullm_exec.h"

namespace fullm {

std::unique_ptr<Backend> make_cuda_backend(int) {
    throw std::runtime_error("this build of fullm has no CUDA backend (rebuild with --features cuda)");
}

bool cuda_available() { return false; }

}  // namespace fullm

extern "C" {

int sk_cuda_available(void) { return 0; }

int sk_cuda_device_info(int32_t, char *name, int name_len, size_t *, size_t *, int32_t *, int32_t *, int32_t *) {
    if (name && name_len > 0) name[0] = 0;
    return -1;
}

int sk_cuda_bench_copy(int32_t, size_t, int32_t, double *, double *) { return -1; }

int sk_cuda_bench_gemm(int32_t, int32_t, int64_t, int64_t, int64_t, int32_t, double *, double *) { return -1; }

int sk_exec_kbench(int32_t, const char *, char *report, int report_len) {
    if (report && report_len > 0) report[0] = 0;
    return -1;
}

}  // extern "C"
