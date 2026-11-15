// SPDX-License-Identifier: Apache-2.0
// kernels/copy.cuh - the host tier's page copy kernel (the last fallback of copy_pages).
//
// Part of cuda_backend.cu's translation unit, included there after the CUDA and standard
// headers, in the order common, formats, gemv, convert, router, elementwise, copy, kimi, mla,
// attention, sampling, moe (each file uses only those before it). Kernels and their device
// helpers: no state.
#pragma once

namespace fullm {
namespace {

// The host tier's page copies: (page, pool) segments of seg bytes between the paged pools and
// pinned host pages, on the copy engines. A kernel copying over UVA reaches the same PCIe rate,
// but its blocks hold their SMs for the whole copy, and the persistent prefill kernels
// (FlashAttention-3, the grouped GEMMs), which need whole SMs, wait for it (measured: +200-360 ms
// on prefill steps while sessions load). When the pools sit at one stride (the dense executor
// allocates them so), a page is one 2D copy; otherwise its segments go as a batch
// (cudaMemcpyBatchAsync, CUDA 12.8+), which costs the host ~0.7 us a segment (measured: 5 s over a
// 16-session run). The kernel stays as the last fallback (FULLM_COPY_KERNEL=1): a launch carries
// up to kCopyPages pages in its parameters, four 16-byte loads in flight per thread.
constexpr int kCopyPages = 128;
struct PageCopyArgs {
    char *const *pools;  // device array [n_pools]
    char *host[kCopyPages];
    int32_t dev[kCopyPages];
    int32_t n_pools, n, to_host;
    uint64_t seg;
};

__global__ void __launch_bounds__(256) kv_page_copy_kernel(const PageCopyArgs a) {
    const int64_t segs = int64_t(a.n) * a.n_pools, n16 = int64_t(a.seg / 16);
    for (int64_t g = blockIdx.x; g < segs; g += gridDim.x) {
        const int i = int(g / a.n_pools), s = int(g % a.n_pools);
        char *dev = a.pools[s] + uint64_t(a.dev[i]) * a.seg;
        char *host = a.host[i] + uint64_t(s) * a.seg;
        const char *src = a.to_host ? dev : host;
        char *dst = a.to_host ? host : dev;
        const uint4 *s4 = reinterpret_cast<const uint4 *>(src);
        uint4 *d4 = reinterpret_cast<uint4 *>(dst);
        int64_t k = threadIdx.x;
        for (; k + 3 * blockDim.x < n16; k += 4 * blockDim.x) {
            const uint4 v0 = s4[k], v1 = s4[k + blockDim.x], v2 = s4[k + 2 * blockDim.x], v3 = s4[k + 3 * blockDim.x];
            d4[k] = v0, d4[k + blockDim.x] = v1, d4[k + 2 * blockDim.x] = v2, d4[k + 3 * blockDim.x] = v3;
        }
        for (; k < n16; k += blockDim.x) d4[k] = s4[k];
        for (int64_t b = n16 * 16 + threadIdx.x; b < int64_t(a.seg); b += blockDim.x) dst[b] = src[b];
    }
}

}  // namespace
}  // namespace fullm
