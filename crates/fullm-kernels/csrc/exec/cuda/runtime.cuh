// SPDX-License-Identifier: Apache-2.0
// runtime.cuh - the CUDA runtime under the CUDA backend: the device, its
// stream, memory, events, graphs, the host tier's copies and the grow-on-demand scratch, plus the
// KernelTable (which of the architecture-specific kernels this device runs). CudaBackend
// (cuda_backend.cu) derives from it and keeps the ops; what the runtime does that the ops care
// about reaches them through four hooks (step_begin, on_release, before_read, before_write).
//
// Part of cuda_backend.cu's translation unit, included there after kernels/*.cuh.
#pragma once

#include <mutex>

namespace fullm {
namespace {

struct Scratch {
    void *p = nullptr;
    size_t n = 0;
};

// The kernels this device runs where the choice is by architecture: chosen once from its compute
// capability and what this build compiled. The Hopper paths (FlashAttention-3, CUTLASS's sm90
// grouped GEMMs and FMHA, the int4 wgmma GEMM) are sm90a code: major 9 only. Programmatic
// dependent launch needs sm90 or later. FULLM_CUDA_ARCH (e.g. 80) stands in for the compute
// capability: with it below 90 every Hopper path is off and the generic kernels run, on the H100
// itself.
struct KernelTable {
    int cc = 0;                 // compute capability, major * 10 + minor (or FULLM_CUDA_ARCH)
    bool pdl = false;           // programmatic dependent launch (FULLM_NO_PDL=1: off)
    bool fa3[2] = {};           // FlashAttention-3 at head_dim 64, 128 (the cutlass feature)
    bool fa3_mla = false;       // FlashAttention-3 at head_dim 192 / 128 (MLA prefill, un-absorbed)
    bool fa3_fp8 = false;       // its e4m3 kernels (head_dim 128)
    bool fmha[2] = {};          // CUTLASS's example-88 FMHA at head_dim 64, 128
    bool cutlass_bf16 = false;  // the grouped GEMMs of the prefill experts: bf16
    bool cutlass_fp8 = false;   // ... blockwise fp8
    bool int4_gemm = false;     // the int4 experts' prefill GEMM (FULLM_NO_INT4_GEMM=1: off)

    static KernelTable make(const cudaDeviceProp &prop) {
        KernelTable t;
        const char *a = std::getenv("FULLM_CUDA_ARCH");
        t.cc = a && std::atoi(a) > 0 ? std::atoi(a) : prop.major * 10 + prop.minor;
        const bool hopper = t.cc / 10 == 9 && prop.major == 9;
        const char *np = std::getenv("FULLM_NO_PDL");
        t.pdl = t.cc >= 90 && prop.major >= 9 && !(np && std::string(np) == "1");
        t.fa3[0] = hopper && fa3_ops::available(64);
        t.fa3[1] = hopper && fa3_ops::available(128);
        t.fa3_fp8 = hopper && fa3_ops::available_fp8(128);
        t.fa3_mla = hopper && fa3_ops::available_mla();
        t.fmha[0] = hopper && cutlass_ops::fmha_available(64);
        t.fmha[1] = hopper && cutlass_ops::fmha_available(128);
        t.cutlass_bf16 = hopper && cutlass_ops::available();
        t.cutlass_fp8 = hopper && cutlass_ops::fp8_available();
        t.int4_gemm = hopper && !env_on("FULLM_NO_INT4_GEMM", false) && int4_gemm::available();
        return t;
    }
    bool fa3_at(int64_t D) const { return D == 64 ? fa3[0] : D == 128 ? fa3[1] : false; }
    bool fa3_fp8_at(int64_t D) const { return fa3_fp8 && D == 128; }
    bool fmha_at(int64_t D) const { return D == 64 ? fmha[0] : D == 128 ? fmha[1] : false; }
    // e.g. "sm90: fa3 fa3-fp8 fmha cutlass cutlass-fp8 int4-gemm pdl"
    std::string describe() const {
        std::string s = "sm" + std::to_string(cc) + ":";
        if (fa3[0] || fa3[1]) s += " fa3";
        if (fa3_fp8) s += " fa3-fp8";
        if (fa3_mla) s += " fa3-mla";
        if (fmha[0] || fmha[1]) s += " fmha";
        if (cutlass_bf16) s += " cutlass";
        if (cutlass_fp8) s += " cutlass-fp8";
        if (int4_gemm) s += " int4-gemm";
        if (pdl) s += " pdl";
        if (s.back() == ':') s += " generic";
        return s;
    }
};

class CudaRuntime : public Backend {
public:
    explicit CudaRuntime(int device) : dev_(device) {
        int count = 0;
        SK_CUDA(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        if (device < 0 || device >= count)
            cuda_fail("open", ("device " + std::to_string(device) + " of " + std::to_string(count)).c_str());
        SK_CUDA(cudaSetDevice(device), "cudaSetDevice");
        cudaDeviceProp prop;
        SK_CUDA(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
        sm_count_ = prop.multiProcessorCount;
        smem_optin_ = prop.sharedMemPerBlockOptin;
        mem_total_ = prop.totalGlobalMem;
        kt_ = KernelTable::make(prop);
        name_ = std::string("cuda:") + prop.name + " (" + kt_.describe() + ")";
        SK_CUDA(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking), "cudaStreamCreate");
    }

    ~CudaRuntime() override {
        cudaSetDevice(dev_);
        if (s_) cudaStreamSynchronize(s_);
        for (void *p : retired_) cudaFree(p);
        if (cs_) cudaStreamSynchronize(cs_), cudaStreamDestroy(cs_);
        if (d_copy_pools_) cudaFree(d_copy_pools_);
        if (s_) cudaStreamDestroy(s_);
    }

    const char *name() const override { return name_.c_str(); }
    bool on_host() const override { return false; }

    // ---------------------------------------------------------------- memory
    void *alloc(size_t bytes) override {
        bind();
        void *p = nullptr;
        SK_CUDA(cudaMalloc(&p, std::max<size_t>(bytes, 1)), "cudaMalloc");
        return p;
    }
    // Never throws: a release can run in a destructor while the process (and the driver) is
    // shutting down, when freeing cannot fail usefully.
    void release(void *p) override {
        if (!p) return;
        if (cudaSetDevice(dev_) == cudaSuccess) {
            on_release(p);
            cudaFree(p);
        }
        cudaGetLastError();
    }
    void *host_alloc(size_t bytes) override {
        bind();
        void *p = nullptr;
        SK_CUDA(cudaHostAlloc(&p, std::max<size_t>(bytes, 1), cudaHostAllocPortable), "cudaHostAlloc");
        return p;
    }
    void host_release(void *p) override {
        if (!p) return;
        if (cudaSetDevice(dev_) == cudaSuccess) cudaFreeHost(p);
        cudaGetLastError();
    }
    void upload(void *dst, const void *host_src, size_t n) override {
        if (!n) return;
        step_begin();  // a step begins with its uploads
        bind();
        SK_CUDA(cudaMemcpyAsync(dst, host_src, n, cudaMemcpyHostToDevice, s_), "upload");
    }
    void download(void *host_dst, const void *src, size_t n) override {
        if (!n) return;
        bind();
        before_read(src, n);
        SK_CUDA(cudaMemcpyAsync(host_dst, src, n, cudaMemcpyDeviceToHost, s_), "download");
    }
    void copy(void *dst, const void *src, size_t n) override {
        before_write(dst);
        if (!n) return;
        bind();
        before_read(src, n);
        SK_CUDA(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, s_), "copy");
    }
    void zero(void *p, size_t n) override {
        if (!n) return;
        bind();
        SK_CUDA(cudaMemsetAsync(p, 0, n, s_), "zero");
    }
    void sync() override {
        bind();
        SK_CUDA(cudaStreamSynchronize(s_), "sync");
    }
    void *record_event() override {
        bind();
        cudaEvent_t ev;
        SK_CUDA(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
        SK_CUDA(cudaEventRecord(ev, s_), "cudaEventRecord");
        return ev;
    }
    void wait_event(void *ev) override {
        bind();
        SK_CUDA(cudaEventSynchronize(static_cast<cudaEvent_t>(ev)), "cudaEventSynchronize");
    }
    void release_event(void *ev) override {
        if (!ev) return;
        bind();
        SK_CUDA(cudaEventDestroy(static_cast<cudaEvent_t>(ev)), "cudaEventDestroy");
    }
    // the host tier: on a copy stream of its own, after `after` (HostPages' worker thread calls
    // this: nothing here may touch s_)
    void copy_pages(bool to_host, void *const *pools, int n_pools, size_t seg_bytes, const int32_t *dev_pages,
                    void *const *host_pages, int64_t n, void *after) override {
        if (n <= 0) return;
        bind();
        if (!cs_) {
            std::lock_guard<std::mutex> lk(cs_mu_);
            if (!cs_) SK_CUDA(cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking), "cudaStreamCreate (copies)");
            int mp = 0;
            SK_CUDA(cudaDeviceGetAttribute(&mp, cudaDevAttrMaxPitch, dev_), "cudaDeviceGetAttribute");
            max_pitch_ = size_t(mp);
        }
        if (copy_kernel_ && (copy_pools_.size() != size_t(n_pools) || !std::equal(copy_pools_.begin(), copy_pools_.end(), pools))) {
            if (d_copy_pools_) {
                SK_CUDA(cudaStreamSynchronize(cs_), "sync (copies)");
                cudaFree(d_copy_pools_);
            }
            copy_pools_.assign(pools, pools + n_pools);
            SK_CUDA(cudaMalloc(&d_copy_pools_, sizeof(void *) * n_pools), "cudaMalloc");
            SK_CUDA(cudaMemcpy(d_copy_pools_, copy_pools_.data(), sizeof(void *) * n_pools, cudaMemcpyHostToDevice),
                    "upload pools");
        }
        if (after) SK_CUDA(cudaStreamWaitEvent(cs_, static_cast<cudaEvent_t>(after), 0), "cudaStreamWaitEvent");
        const size_t stride = n_pools > 1 ? size_t(static_cast<char *>(pools[1]) - static_cast<char *>(pools[0])) : seg_bytes;
        bool strided = !copy_kernel_ && stride >= seg_bytes && stride <= max_pitch_;
        for (int s = 2; s < n_pools && strided; ++s)
            strided = static_cast<char *>(pools[s]) == static_cast<char *>(pools[0]) + stride * s;
        if (strided) {
            // page i: n_pools rows of seg bytes, stride apart on the device, back to back on the host
            for (int64_t i = 0; i < n; ++i) {
                char *dev = static_cast<char *>(pools[0]) + size_t(dev_pages[i]) * seg_bytes;
                char *host = static_cast<char *>(host_pages[i]);
                if (to_host)
                    SK_CUDA(cudaMemcpy2DAsync(host, seg_bytes, dev, stride, seg_bytes, n_pools, cudaMemcpyDeviceToHost, cs_),
                            "cudaMemcpy2DAsync (to host)");
                else
                    SK_CUDA(cudaMemcpy2DAsync(dev, stride, host, seg_bytes, seg_bytes, n_pools, cudaMemcpyHostToDevice, cs_),
                            "cudaMemcpy2DAsync (from host)");
            }
            return;
        }
#if CUDART_VERSION >= 12080
        if (!copy_kernel_) {
            // one copy per (page, pool) segment, in batches the driver hands to the copy engines
            const size_t total = size_t(n) * n_pools, batch = std::min<size_t>(total, 16384);
            copy_dst_.resize(batch), copy_src_.resize(batch), copy_size_.assign(batch, seg_bytes);
            cudaMemcpyAttributes attr{};
            attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
            attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;
            size_t attr_idx = 0, k = 0;
            for (int64_t i = 0; i < n; ++i)
                for (int s = 0; s < n_pools; ++s) {
                    char *dev = static_cast<char *>(pools[s]) + size_t(dev_pages[i]) * seg_bytes;
                    char *host = static_cast<char *>(host_pages[i]) + size_t(s) * seg_bytes;
                    copy_dst_[k] = to_host ? host : dev;
                    copy_src_[k] = to_host ? dev : host;
                    if (++k == batch || (i == n - 1 && s == n_pools - 1)) {
                        SK_CUDA(cudaMemcpyBatchAsync(copy_dst_.data(), copy_src_.data(), copy_size_.data(), k, &attr,
                                                     &attr_idx, 1, cs_),
                                "cudaMemcpyBatchAsync");
                        k = 0;
                    }
                }
            return;
        }
#endif
        PageCopyArgs a{};
        a.pools = reinterpret_cast<char *const *>(d_copy_pools_);
        a.n_pools = n_pools;
        a.to_host = to_host;
        a.seg = seg_bytes;
        for (int64_t first = 0; first < n; first += kCopyPages) {
            a.n = static_cast<int32_t>(std::min<int64_t>(kCopyPages, n - first));
            for (int i = 0; i < a.n; ++i) {
                a.dev[i] = dev_pages[first + i];
                a.host[i] = static_cast<char *>(host_pages[first + i]);
            }
            const int64_t segs = int64_t(a.n) * n_pools;
            kv_page_copy_kernel<<<static_cast<unsigned>(std::min<int64_t>(segs, copy_blocks_)), 256, 0, cs_>>>(a);
            SK_CUDA(cudaGetLastError(), "kv_page_copy_kernel");
        }
    }
    void upload_async(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height) override {
        if (!width || !height) return;
        bind();
        {
            std::lock_guard<std::mutex> lk(cs_mu_);
            if (!cs_) SK_CUDA(cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking), "cudaStreamCreate (copies)");
        }
        if (height == 1)
            SK_CUDA(cudaMemcpyAsync(dst, src, width, cudaMemcpyHostToDevice, cs_), "upload_async");
        else
            SK_CUDA(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyHostToDevice, cs_),
                    "upload_async (2d)");
    }
    void *record_copy_event() override {
        bind();
        cudaEvent_t ev;
        SK_CUDA(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
        SK_CUDA(cudaEventRecord(ev, cs_ ? cs_ : s_), "cudaEventRecord");
        return ev;
    }
    bool event_done(void *ev) override {
        if (!ev) return true;
        bind();
        const cudaError_t e = cudaEventQuery(static_cast<cudaEvent_t>(ev));
        if (e == cudaErrorNotReady) return false;
        SK_CUDA(e, "cudaEventQuery");
        return true;
    }
    void stream_wait(void *ev) override {
        if (!ev) return;
        bind();
        SK_CUDA(cudaStreamWaitEvent(s_, static_cast<cudaEvent_t>(ev), 0), "cudaStreamWaitEvent");
    }
    size_t free_memory() const override {
        cudaSetDevice(dev_);
        size_t f = 0, t = 0;
        SK_CUDA(cudaMemGetInfo(&f, &t), "cudaMemGetInfo");
        return f;
    }
    bool graph_begin() override {
        bind();
        SK_CUDA(cudaStreamBeginCapture(s_, cudaStreamCaptureModeThreadLocal), "graph capture");
        return true;
    }
    void *graph_end() override {
        cudaGraph_t g = nullptr;
        SK_CUDA(cudaStreamEndCapture(s_, &g), "graph capture end");
        cudaGraphExec_t ex = nullptr;
        const cudaError_t e = cudaGraphInstantiate(&ex, g, 0);
        cudaGraphDestroy(g);
        if (e != cudaSuccess) cuda_fail("graph instantiate", cudaGetErrorString(e));
        ++graphs_live_;
        return ex;
    }
    void graph_abort() override {
        cudaGraph_t g = nullptr;
        if (cudaStreamEndCapture(s_, &g) == cudaSuccess && g) cudaGraphDestroy(g);
        cudaGetLastError();  // clear the capture's error state
    }
    void graph_launch(void *g) override {
        bind();
        SK_CUDA(cudaGraphLaunch(static_cast<cudaGraphExec_t>(g), s_), "graph launch");
    }
    void graph_release(void *g) override {
        if (!g) return;
        cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(g));
        --graphs_live_;
    }

protected:
    // The ops' state that follows the runtime: a step's first upload starts a new step, released
    // memory takes its derived copies along, a buffer the runtime reads (download, copy source)
    // must hold its values, one it overwrites (copy destination) loses what was derived from it.
    virtual void step_begin() {}
    virtual void on_release(void * /*p*/) {}
    virtual void before_read(const void * /*p*/, size_t /*n*/) {}
    virtual void before_write(void * /*p*/) {}

    void bind() const { SK_CUDA(cudaSetDevice(dev_), "cudaSetDevice"); }

    // Grow-on-demand scratch. Growing waits for queued work that may still read the old
    // buffer before freeing it; while graphs exist the old buffer is kept instead (a recorded
    // launch may still point at it) and freed with the backend.
    void *ensure(Scratch &s, size_t bytes) {
        if (bytes <= s.n) return s.p;
        const size_t nb = std::max(bytes, s.n + s.n / 4);  // grow by >= 25% to avoid churn
        SK_CUDA(cudaStreamSynchronize(s_), "scratch sync");
        if (s.p && graphs_live_ > 0) retired_.push_back(s.p);
        else if (s.p) SK_CUDA(cudaFree(s.p), "scratch free");
        s.p = nullptr;
        s.n = 0;
        void *p = nullptr;
        SK_CUDA(cudaMalloc(&p, nb), "scratch alloc");
        s.p = p;
        s.n = nb;
        return p;
    }

    int dev_;
    int sm_count_ = 132;
    size_t smem_optin_ = 227 * 1024;  // the largest dynamic shared memory a block may opt into
    size_t mem_total_ = size_t(80) << 30;  // the device's memory
    std::string name_;
    KernelTable kt_;
    cudaStream_t s_ = nullptr;
    int64_t graphs_live_ = 0;
    std::vector<void *> retired_;

private:
    // the host tier's copies: their stream, the pools as the kernel indexes them, and the copy
    // kernel's grid (FULLM_COPY_BLOCKS)
    cudaStream_t cs_ = nullptr;
    std::mutex cs_mu_;  // the copy stream is created by whichever thread copies first
    std::vector<void *> copy_pools_;
    void *d_copy_pools_ = nullptr;
    const int copy_blocks_ = std::max(1, std::getenv("FULLM_COPY_BLOCKS") ? std::atoi(std::getenv("FULLM_COPY_BLOCKS")) : 32);
    const bool copy_kernel_ = env_on("FULLM_COPY_KERNEL", false);
    size_t max_pitch_ = 0;  // the largest pitch a 2D copy takes
    std::vector<void *> copy_dst_, copy_src_;  // cudaMemcpyBatchAsync's arrays
    std::vector<size_t> copy_size_;
};

}  // namespace
}  // namespace fullm
