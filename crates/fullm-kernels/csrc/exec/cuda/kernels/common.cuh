// SPDX-License-Identifier: Apache-2.0
// kernels/common.cuh - device helpers shared by every kernel family: errors, conversions, warp and
// block reductions, programmatic dependent launch, the launch helpers and the environment knobs.
//
// Part of cuda_backend.cu's translation unit, included there after the CUDA and standard
// headers, in the order common, formats, gemv, convert, router, elementwise, copy, kimi, mla,
// attention, sampling, moe (each file uses only those before it). Kernels and their device
// helpers: no state.
#pragma once

namespace fullm {
namespace {

[[noreturn]] void cuda_fail(const char *what, const char *detail) {
    throw std::runtime_error(std::string("cuda backend: ") + what + ": " + detail);
}

#define SK_CUDA(call, what)                                              \
    do {                                                                 \
        cudaError_t sk_err_ = (call);                                    \
        if (sk_err_ != cudaSuccess) cuda_fail((what), cudaGetErrorString(sk_err_)); \
    } while (0)

#define SK_LAUNCH(what) SK_CUDA(cudaGetLastError(), (what))

#define SK_BLAS(call, what)                                                         \
    do {                                                                            \
        cublasStatus_t sk_bst_ = (call);                                            \
        if (sk_bst_ != CUBLAS_STATUS_SUCCESS)                                       \
            cuda_fail((what), ("cuBLAS status " + std::to_string((int)sk_bst_)).c_str()); \
    } while (0)

constexpr int kThreads = 256;

unsigned grid1d(int64_t n, int threads = kThreads) {
    int64_t b = (n + threads - 1) / threads;
    return (unsigned)std::min<int64_t>(std::max<int64_t>(b, 1), 1 << 20);
}

// ------------------------------------------------------------------ device helpers

__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float(static_cast<uint32_t>(h) << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ int64_t imin(int64_t a, int64_t b) { return a < b ? a : b; }

// E4M3 (FN). The NaN codes 0x7f/0xff decode to 0, matching the CPU table.
__device__ __forceinline__ float fp8f(uint32_t b) {
    if ((b & 0x7Fu) == 0x7Fu) return 0.0f;
    __nv_fp8_e4m3 v;
    v.__x = static_cast<__nv_fp8_storage_t>(b);
    return static_cast<float>(v);
}

__device__ __forceinline__ float ldst(const float *p) { return *p; }
__device__ __forceinline__ float ldst(const __nv_bfloat16 *p) { return __bfloat162float(*p); }
__device__ __forceinline__ void stst(float *p, float v) { *p = v; }
__device__ __forceinline__ void stst(__nv_bfloat16 *p, float v) { *p = __float2bfloat16_rn(v); }

// Butterfly: every lane ends with the same bits (each step adds the same pair).
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// fp8 KV cache (kv_fp8.h): the exponent byte for a vector's largest magnitude (the host rule, from
// its bits), 2^(e - 127) and 2^(127 - e) as floats (e in 1 .. 253; e = 0 only for a zero vector)
__device__ __forceinline__ uint32_t kv8_exp_dev(float amax) {
    const uint32_t u = __float_as_uint(amax), E = (u >> 23) & 0xFFu;
    if (E == 0) return 0;
    const int k = static_cast<int>(E) - 135 + ((u & 0x7FFFFFu) > 0x600000u ? 1 : 0);
    return static_cast<uint32_t>(min(max(k, -127), 128) + 127);
}
__device__ __forceinline__ float pow2_biased(int e) {  // 2^(e - 127); 0 below the normal range
    return e >= 1 ? __uint_as_float(static_cast<uint32_t>(min(e, 254)) << 23) : 0.0f;
}
__device__ __forceinline__ uint8_t f2e4m3(float v) {
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}

// Programmatic dependent launch (Hopper). A kernel launched with the PDL attribute may start
// while its predecessor in the stream is still running: it may only touch data no earlier
// kernel writes (weights) until pdl_wait(), which returns once the predecessor has completed
// and its writes are visible. pdl_trigger() lets the successor start early. Both are no-ops
// for kernels launched without the attribute, and compiled out below sm90 (no griddepcontrol;
// the backend never sets the attribute there: KernelTable::pdl).
__device__ __forceinline__ void pdl_wait() {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_trigger() {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// blockDim.x must be a multiple of 32. red: >= 32 floats of shared memory.
__device__ float block_sum(float v, float *red);

// block_sum for a row split across a cluster of CL blocks (sm90; CL == 1: block_sum): block rank
// r holds the row's warps [r * nw, (r + 1) * nw). Every warp's sum goes to red[its warp in the
// row] in every block of the cluster; each block then reduces red as block_sum does for one block
// of CL * nw warps: the same bits.
template <int CL>
__device__ __forceinline__ float cluster_sum(float v, float *red) {
    if constexpr (CL == 1) {
        return block_sum(v, red);
    } else {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
        namespace cg = cooperative_groups;
        cg::cluster_group cl = cg::this_cluster();
        const int lane = threadIdx.x & 31, nw = static_cast<int>(blockDim.x >> 5);
        const int gw = static_cast<int>(cl.block_rank()) * nw + static_cast<int>(threadIdx.x >> 5);
        v = warp_sum(v);
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < CL; ++r) cl.map_shared_rank(red, r)[gw] = v;
        }
        cl.sync();  // every block's writes landed; nothing touches another block's red after this
        if (threadIdx.x < 32) {
            float s = lane < CL * nw ? red[lane] : 0.0f;
            s = warp_sum(s);
            if (lane == 0) red[0] = s;
        }
        __syncthreads();
        return red[0];
#else
        return 0.0f;  // never launched below sm90
#endif
    }
}

__device__ float block_sum(float v, float *red) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int nw = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    __syncthreads();  // red may still be read by a previous call
    if (lane == 0) red[wid] = v;
    __syncthreads();
    if (wid == 0) {
        float r = lane < nw ? red[lane] : 0.0f;
        r = warp_sum(r);
        if (lane == 0) red[0] = r;
    }
    __syncthreads();
    return red[0];
}


int tc_env(const char *name, int dflt) {
    const char *v = std::getenv(name);
    return v && std::atoi(v) > 0 ? std::atoi(v) : dflt;
}
// An on/off knob: unset or empty = dflt, "0" = off, anything else = on (tc_env cannot turn a
// default-on knob off: it reads "0" as unset).
bool env_on(const char *name, bool dflt) {
    const char *v = std::getenv(name);
    if (!v || !*v) return dflt;
    return std::strcmp(v, "0") != 0;
}

// Weight-streaming kernels are launched with programmatic dependent launch (see pdl_wait):
// their weights stream in while the previous kernel finishes. FULLM_NO_PDL=1 turns it off.
bool g_pdl = true;

// launch_pdl in clusters of `cluster` blocks along x (sm90; 1: none)
template <typename... KArgs, typename... Args>
void launch_pdl_cluster(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, int cluster,
                        Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = s;
    cudaLaunchAttribute attr[2];
    unsigned n = 0;
    if (g_pdl) {
        attr[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[n++].val.programmaticStreamSerializationAllowed = 1;
    }
    if (cluster > 1) {
        attr[n].id = cudaLaunchAttributeClusterDimension;
        attr[n].val.clusterDim.x = static_cast<unsigned>(cluster);
        attr[n].val.clusterDim.y = 1;
        attr[n++].val.clusterDim.z = 1;
    }
    cfg.attrs = attr;
    cfg.numAttrs = n;
    SK_CUDA(cudaLaunchKernelEx(&cfg, kernel, args...), "launch (pdl)");
}

template <typename... KArgs, typename... Args>
void launch_pdl(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, Args... args) {
    launch_pdl_cluster(kernel, grid, block, smem, s, 1, args...);
}

}  // namespace
}  // namespace fullm
