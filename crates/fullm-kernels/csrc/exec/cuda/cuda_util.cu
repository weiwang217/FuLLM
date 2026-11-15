// SPDX-License-Identifier: Apache-2.0
// cuda_util.cu - device queries and micro-benchmarks behind fullm_exec.h.
//
// These feed `fullm bench`: the planner's effective rates (PCIe copy bandwidth, achieved
// GEMV weight bandwidth per format) are replaced by what this device actually sustains.

#include "../backend.h"
#include "fullm_exec.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <vector>

namespace {

struct XorShift {
    uint64_t s;
    explicit XorShift(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    float uni() { return static_cast<float>((next() >> 40) * (1.0 / 16777216.0)) * 2.0f - 1.0f; }
};

int elem_bytes(int32_t fmt) {
    switch (fmt) {
        case SK_FMT_F32: return 4;
        case SK_FMT_BF16: return 2;
        case SK_FMT_INT8:
        case SK_FMT_FP8: return 1;
        default: return 0;
    }
}

}  // namespace

extern "C" {

int sk_cuda_available(void) { return fullm::cuda_available() ? 1 : 0; }

int sk_cuda_device_info(int32_t device, char *name, int name_len, size_t *total_mem, size_t *free_mem,
                        int32_t *sm_count, int32_t *cc_major, int32_t *cc_minor) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return 1;
    if (cudaSetDevice(device) != cudaSuccess) return 1;
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) != cudaSuccess) return 1;
    if (name && name_len > 0) {
        std::strncpy(name, prop.name, static_cast<size_t>(name_len) - 1);
        name[name_len - 1] = 0;
    }
    if (total_mem) *total_mem = t;
    if (free_mem) *free_mem = f;
    if (sm_count) *sm_count = prop.multiProcessorCount;
    if (cc_major) *cc_major = prop.major;
    if (cc_minor) *cc_minor = prop.minor;
    return 0;
}

int sk_cuda_bench_copy(int32_t device, size_t bytes, int32_t iters, double *h2d_gbps, double *d2h_gbps) {
    if (bytes == 0 || iters <= 0) return 1;
    if (cudaSetDevice(device) != cudaSuccess) return 1;
    void *host = nullptr, *dev = nullptr;
    cudaStream_t s = nullptr;
    cudaEvent_t a = nullptr, b = nullptr;
    int rc = 1;
    do {
        if (cudaHostAlloc(&host, bytes, cudaHostAllocPortable) != cudaSuccess) break;
        if (cudaMalloc(&dev, bytes) != cudaSuccess) break;
        if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) break;
        if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) break;
        std::memset(host, 1, bytes);
        // warm up both directions once
        cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(host, dev, bytes, cudaMemcpyDeviceToHost, s);
        if (cudaStreamSynchronize(s) != cudaSuccess) break;
        float ms = 0.0f;
        cudaEventRecord(a, s);
        for (int i = 0; i < iters; ++i) cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, s);
        cudaEventRecord(b, s);
        if (cudaEventSynchronize(b) != cudaSuccess) break;
        cudaEventElapsedTime(&ms, a, b);
        if (h2d_gbps) *h2d_gbps = static_cast<double>(bytes) * iters / (ms / 1e3) / 1e9;
        cudaEventRecord(a, s);
        for (int i = 0; i < iters; ++i) cudaMemcpyAsync(host, dev, bytes, cudaMemcpyDeviceToHost, s);
        cudaEventRecord(b, s);
        if (cudaEventSynchronize(b) != cudaSuccess) break;
        cudaEventElapsedTime(&ms, a, b);
        if (d2h_gbps) *d2h_gbps = static_cast<double>(bytes) * iters / (ms / 1e3) / 1e9;
        rc = cudaGetLastError() == cudaSuccess ? 0 : 1;
    } while (false);
    if (a) cudaEventDestroy(a);
    if (b) cudaEventDestroy(b);
    if (s) cudaStreamDestroy(s);
    if (dev) cudaFree(dev);
    if (host) cudaFreeHost(host);
    return rc;
}

int sk_cuda_bench_gemm(int32_t device, int32_t fmt, int64_t n, int64_t k, int64_t m, int32_t iters, double *gbps,
                       double *ms) {
    const int eb = elem_bytes(fmt);
    if (eb == 0 || n <= 0 || k <= 0 || m <= 0 || iters <= 0) return 1;
    try {
        auto be = fullm::make_cuda_backend(device);
        XorShift rng(1234);
        const size_t wbytes = static_cast<size_t>(n) * k * eb;
        std::vector<uint8_t> hw(wbytes);
        if (fmt == SK_FMT_F32) {
            float *f = reinterpret_cast<float *>(hw.data());
            for (size_t i = 0; i < static_cast<size_t>(n) * k; ++i) f[i] = rng.uni() * 0.02f;
        } else if (fmt == SK_FMT_BF16) {
            uint16_t *h = reinterpret_cast<uint16_t *>(hw.data());
            for (size_t i = 0; i < static_cast<size_t>(n) * k; ++i) {
                float v = rng.uni() * 0.02f;
                uint32_t u;
                std::memcpy(&u, &v, 4);
                h[i] = static_cast<uint16_t>(u >> 16);
            }
        } else {
            for (size_t i = 0; i < wbytes; ++i) {
                uint8_t b = static_cast<uint8_t>(rng.next());
                if (fmt == SK_FMT_FP8 && (b & 0x7F) == 0x7F) b = 0x38;  // no NaN codes
                hw[i] = b;
            }
        }
        std::vector<float> hs(static_cast<size_t>(n), 0.01f);
        std::vector<float> hx(static_cast<size_t>(m) * k);
        for (float &v : hx) v = rng.uni();

        void *wd = be->alloc(wbytes);
        float *sd = static_cast<float *>(be->alloc(hs.size() * sizeof(float)));
        float *xd = static_cast<float *>(be->alloc(hx.size() * sizeof(float)));
        float *yd = static_cast<float *>(be->alloc(static_cast<size_t>(m) * n * sizeof(float)));
        be->upload(wd, hw.data(), wbytes);
        be->upload(sd, hs.data(), hs.size() * sizeof(float));
        be->upload(xd, hx.data(), hx.size() * sizeof(float));
        be->sync();

        fullm::Weight w;
        w.fmt = static_cast<fullm::WFmt>(fmt);
        w.rows = n;
        w.cols = k;
        w.data = wd;
        w.scale = (fmt == SK_FMT_INT8 || fmt == SK_FMT_FP8) ? sd : nullptr;
        for (int i = 0; i < 2; ++i) be->gemm(w, xd, m, yd, n);
        be->sync();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) be->gemm(w, xd, m, yd, n);
        be->sync();
        const auto t1 = std::chrono::steady_clock::now();
        const double per = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        if (ms) *ms = per;
        if (gbps) *gbps = static_cast<double>(w.bytes()) / (per / 1e3) / 1e9;
        be->release(wd);
        be->release(sd);
        be->release(xd);
        be->release(yd);
        return 0;
    } catch (const std::exception &) {
        return 1;
    }
}

}  // extern "C"
