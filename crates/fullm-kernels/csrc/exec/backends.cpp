// SPDX-License-Identifier: Apache-2.0
// backends.cpp - the backends compiled into this build, by name, and the C ABI's queries of them
// (sk_backend_list, sk_backend_caps). A new backend adds one entry
// here, under its build feature.

#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

#include "backend.h"
#include "fullm_exec.h"

namespace fullm {

namespace {

bool always(int) { return true; }
std::unique_ptr<Backend> create_cpu(int, int threads) { return make_cpu_backend(threads); }
std::unique_ptr<Backend> create_cpu_core(int, int threads) { return make_cpu_core_backend(threads); }
std::unique_ptr<Backend> create_cuda(int device, int) { return make_cuda_backend(device); }
bool cuda_present(int) { return cuda_available(); }
#ifdef FULLM_TORCH
std::unique_ptr<Backend> create_torch(int, int threads) { return make_torch_backend(threads); }
#endif

}  // namespace

const std::vector<BackendEntry> &backend_registry() {
    static const std::vector<BackendEntry> list = {
        {"cpu", always, create_cpu},
        // the CPU backend with only the core ops: what a new device implements first (and the
        // check of Backend's base forms)
        {"cpu-core", always, create_cpu_core},
#ifdef FULLM_TORCH
        {"torch", always, create_torch},
#endif
        // the CUDA entry is always listed: a build without CUDA reports it unavailable
        {"cuda", cuda_present, create_cuda},
    };
    return list;
}

std::unique_ptr<Backend> make_backend(const std::string &name, int device, int cpu_threads) {
    for (const BackendEntry &e : backend_registry())
        if (name == e.name) return e.create(device, cpu_threads);
    std::string known;
    for (const BackendEntry &e : backend_registry()) known += std::string(known.empty() ? "" : ", ") + e.name;
    throw std::runtime_error("unknown backend \"" + name + "\" (this build has: " + known + ")");
}

std::string backend_name(const char *name, int kind) {
    if (name && *name) return std::string(name, strnlen(name, 32));
    switch (kind) {
        case SK_BACKEND_CUDA: return "cuda";
        case SK_BACKEND_TORCH: return "torch";
        default: return "cpu";
    }
}

}  // namespace fullm

extern "C" void sk_sample_rows_host(int32_t *tokens, const float *logits, int64_t rows, int64_t n,
                                    const sk_sample_row *sample) {
    for (int64_t r = 0; r < rows; ++r)
        tokens[r] = fullm::sample_row_host(logits + r * n, n, reinterpret_cast<const fullm::SampleRow &>(sample[r]));
}

extern "C" int sk_backend_list(char *out, int len) {
    std::string s;
    for (const fullm::BackendEntry &e : fullm::backend_registry()) {
        bool ok = false;
        try {
            ok = e.available(0);
        } catch (...) {
        }
        s += std::string(s.empty() ? "" : ",") + e.name + (ok ? ":1" : ":0");
    }
    if (out && len > 0) std::snprintf(out, static_cast<size_t>(len), "%s", s.c_str());
    return static_cast<int>(s.size());
}

extern "C" int sk_backend_caps(const char *backend, int32_t device, int32_t bf16_activations, sk_caps *out, char *err,
                               int err_len) {
    try {
        if (!out) throw std::runtime_error("sk_backend_caps: out is null");
        auto b = fullm::make_backend(backend ? backend : "", device, 1);
        b->set_exact_activations(bf16_activations == 0);
        const fullm::Caps c = b->caps();
        const fullm::Policy p = b->policy();
        std::memset(out, 0, sizeof *out);
        std::snprintf(out->name, sizeof out->name, "%s", backend);  // the registry name (name() describes the device)
        out->memory = static_cast<int32_t>(c.memory);
        out->graphs = c.graphs ? 1 : 0;
        out->async_copy = c.async_copy ? 1 : 0;
        out->dense_formats = c.dense_formats;
        out->expert_formats = c.expert_formats;
        out->kv_formats = c.kv_formats;
        out->kv8_page_multiple = c.kv8_page_multiple;
        out->families = c.families;
        out->expert_tile_max_rows = p.expert_tile_max_rows;
        out->expert_gemm_min_rows = p.expert_gemm_min_rows;
        return 0;
    } catch (const std::exception &e) {
        if (err && err_len > 0) std::snprintf(err, static_cast<size_t>(err_len), "%s", e.what());
    } catch (...) {
        if (err && err_len > 0) std::snprintf(err, static_cast<size_t>(err_len), "unknown error");
    }
    return 1;
}
