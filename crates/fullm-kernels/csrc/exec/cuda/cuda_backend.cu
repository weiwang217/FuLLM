// SPDX-License-Identifier: Apache-2.0
// cuda_backend.cu - the CUDA implementation of fullm::Backend.
//
// Layout: kernels/*.cuh hold the kernels by family (common, formats, gemv, convert, router,
// elementwise, copy, kimi, kda_chunk, mla, attention, sampling, moe, lora), runtime.cuh the
// runtime (device, stream, memory, events, graphs, host-tier copies, scratch) and the
// KernelTable, this file the ops (CudaBackend over CudaRuntime): one translation unit. The
// Hopper libraries are units of their own, compiled for sm_90a only (fa3_attention.cu,
// cutlass_ops.cu, int4_gemm.cu); everything here also compiles for earlier architectures
// (FULLM_CUDA_ARCHS), where the KernelTable keeps to the generic kernels.
//
// Every kernel here has a CPU twin behind the same interface, and conformance.cpp compares
// them op by op. The semantics contract is backend.h; the numerics follow the Rust
// reference (crates/fullm/src/ops.rs, models/kimi_k3/cpu.rs).
//
// DETERMINISM. No floating-point atomics anywhere. Reductions use fixed trees (warp
// butterflies, block trees in a fixed order). The GEMV path (M <= kGemvMaxRows; with bf16 activations
// below FULLM_BLAS_MIN_ROWS, 8, but for fp8 weights) sums each
// output element in an order that does not depend on M, so a row's result is the same in
// any batch. The cuBLAS path (M > kGemvMaxRows) is run-to-run deterministic only.

#include "../backend.h"
#include "../int4_group.h"
#include "../kv_fp8.h"
#include "cutlass_ops.h"
#include "int4_gemm.h"
#include "fa3_attention.h"

#include <cooperative_groups.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <map>
#include <tuple>
#include <unordered_map>
#include <vector>


#include "kernels/common.cuh"
#include "kernels/formats.cuh"
#include "kernels/gemv.cuh"
#include "kernels/convert.cuh"
#include "kernels/router.cuh"
#include "kernels/elementwise.cuh"
#include "kernels/copy.cuh"
#include "kernels/kimi.cuh"
#include "kernels/kda_chunk.cuh"
#include "kernels/mla.cuh"
#include "kernels/attention.cuh"
#include "kernels/sampling.cuh"
#include "kernels/moe.cuh"
#include "kernels/lora.cuh"
#include "runtime.cuh"

namespace fullm {
namespace {

// int4 experts: steps of up to this many rows run the decode tiles, larger ones the int4 GEMM
// (FULLM_INT4_ROWS, FULLM_INT4_GEMM_ROWS; see i4gemm_rows_)
constexpr int64_t kInt4TileRows = 96;

// ------------------------------------------------------------------ the backend
// The ops over the runtime (runtime.cuh): the kernels of kernels/*.cuh, cuBLAS, and the libraries
// of the other units (FlashAttention-3, CUTLASS, the int4 GEMM), each chosen by kt_ where the
// architecture decides.

class CudaBackend final : public CudaRuntime {
public:
    explicit CudaBackend(int device) : CudaRuntime(device) {
        SK_BLAS(cublasCreate(&blas_), "cublasCreate");
        SK_BLAS(cublasSetStream(blas_, s_), "cublasSetStream");
        SK_BLAS(cublasSetMathMode(blas_, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
        // cuBLASLt for the fp8 GEMMs (sm89+: e4m3 tensor cores); its workspace fixed, graph-safe
        if (kt_.cc >= 89 && env_on("FULLM_FP8_GEMM", true) && cublasLtCreate(&lt_) == CUBLAS_STATUS_SUCCESS &&
            cudaMalloc(&lt_ws_, kLtWorkspace) != cudaSuccess) {
            cublasLtDestroy(lt_);
            lt_ = nullptr, lt_ws_ = nullptr;
        }
        g_pdl = kt_.pdl;
        const int tc = static_cast<int>(kTcSmem);
        for (auto *k : {(const void *)tc_gemv_kernel<1, false, false>, (const void *)tc_gemv_kernel<1, true, false>,
                        (const void *)tc_gemv_kernel<2, false, false>, (const void *)tc_gemv_kernel<2, true, false>,
                        (const void *)tc_gemv_kernel<3, false, false>, (const void *)tc_gemv_kernel<3, true, false>,
                        (const void *)tc_gemv_kernel<1, false, true>, (const void *)tc_gemv_kernel<1, true, true>,
                        (const void *)tc_gemv_kernel<2, false, true>, (const void *)tc_gemv_kernel<2, true, true>,
                        (const void *)tc_gemv_kernel<3, false, true>, (const void *)tc_gemv_kernel<3, true, true>,
                        (const void *)gemv_group_kernel<false, false>, (const void *)gemv_group_kernel<true, false>,
                        (const void *)gemv_group_kernel<false, true>, (const void *)gemv_group_kernel<true, true>,
                        (const void *)gemv_group_kernel<false, true, true>, (const void *)gemv_group_kernel<true, true, true>,
                        (const void *)tc_gemv_kernel<1, false, true, true>, (const void *)tc_gemv_kernel<1, true, true, true>,
                        (const void *)tc_gemv_kernel<3, false, true, false, true>, (const void *)tc_gemv_kernel<3, true, true, false, true>,
                        (const void *)gemv_group_kernel<false, true, false, true>, (const void *)gemv_group_kernel<true, true, false, true>,
                        (const void *)moe_tc_gateup_kernel<1, false, 1>, (const void *)moe_tc_gateup_kernel<2, false, 1>,
                        (const void *)moe_tc_gateup_kernel<3, false, 1>, (const void *)moe_tc_down_kernel<1, false, 1>,
                        (const void *)moe_tc_down_kernel<2, false, 1>, (const void *)moe_tc_down_kernel<3, false, 1>,
                        (const void *)moe_tc_gateup_kernel<1, true, 1>, (const void *)moe_tc_gateup_kernel<2, true, 1>,
                        (const void *)moe_tc_gateup_kernel<3, true, 1>, (const void *)moe_tc_down_kernel<1, true, 1>,
                        (const void *)moe_tc_down_kernel<2, true, 1>, (const void *)moe_tc_down_kernel<3, true, 1>,
                        (const void *)moe_tc_gateup_kernel<1, false, 2>, (const void *)moe_tc_gateup_kernel<2, false, 2>,
                        (const void *)moe_tc_gateup_kernel<3, false, 2>, (const void *)moe_tc_down_kernel<1, false, 2>,
                        (const void *)moe_tc_down_kernel<2, false, 2>, (const void *)moe_tc_down_kernel<3, false, 2>,
                        (const void *)moe_tc_gateup_kernel<1, true, 2>, (const void *)moe_tc_gateup_kernel<2, true, 2>,
                        (const void *)moe_tc_gateup_kernel<3, true, 2>, (const void *)moe_tc_down_kernel<1, true, 2>,
                        (const void *)moe_tc_down_kernel<2, true, 2>, (const void *)moe_tc_down_kernel<3, true, 2>,
                        (const void *)moe_tc_gateup_kernel<3, true, 1, true>, (const void *)moe_tc_down_kernel<3, true, 1, true>, (const void *)moe_tc_gateup_kernel<3, true, 2, true>, (const void *)moe_tc_down_kernel<3, true, 2, true>, (const void *)moe_tc_gateup_kernel<4, true, 1, true>, (const void *)moe_tc_down_kernel<4, true, 1, true>, (const void *)moe_tc_gateup_kernel<4, true, 2, true>, (const void *)moe_tc_down_kernel<4, true, 2, true>,
                        (const void *)moe_tc_gateup_kernel<4, false, 1>, (const void *)moe_tc_down_kernel<4, false, 1>,
                        (const void *)moe_tc_gateup_kernel<4, true, 1>, (const void *)moe_tc_down_kernel<4, true, 1>,
                        (const void *)moe_tc_gateup_kernel<4, false, 2>, (const void *)moe_tc_down_kernel<4, false, 2>,
                        (const void *)moe_tc_gateup_kernel<4, true, 2>, (const void *)moe_tc_down_kernel<4, true, 2>,
                        (const void *)moe_tc_fused_kernel<1, false, 1>, (const void *)moe_tc_fused_kernel<1, false, 2>,
                        (const void *)moe_tc_fused_kernel<1, true, 1>, (const void *)moe_tc_fused_kernel<1, true, 2>,
                        (const void *)moe_tc_fused_kernel<2, false, 1>, (const void *)moe_tc_fused_kernel<2, false, 2>,
                        (const void *)moe_tc_fused_kernel<2, true, 1>, (const void *)moe_tc_fused_kernel<2, true, 2>,
                        (const void *)moe_tc_fused_kernel<3, false, 1>, (const void *)moe_tc_fused_kernel<3, false, 2>,
                        (const void *)moe_tc_fused_kernel<3, true, 1, true>, (const void *)moe_tc_fused_kernel<3, true, 2, true>,
                        (const void *)moe_tc_fused_kernel<4, false, 1>, (const void *)moe_tc_fused_kernel<4, false, 2>,
                        (const void *)moe_tc_fused_kernel<4, true, 1, true>, (const void *)moe_tc_fused_kernel<4, true, 2, true>,
                        (const void *)moe_tc_fused_kernel<5, true, 2, true>,
                        (const void *)moe_tc_gateup_kernel<5, true, 2, true>, (const void *)moe_tc_down_kernel<5, true, 2, true>,
                        (const void *)router_fused_kernel<false, false>, (const void *)router_fused_kernel<true, false>,
                        (const void *)router_fused_kernel<false, true>, (const void *)router_fused_kernel<true, true>,
                        (const void *)router_fused_kernel<false, true, true>, (const void *)router_fused_kernel<true, true, true>})
            SK_CUDA(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, tc), "tensor-core smem");
        // resident blocks per SM of the expert kernels: their persistent grids fill exactly that
        int occ = 0;
        SK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, moe_tc_gateup_kernel<1, true, 1>, kTcWarps * 32,
                                                              kTcSmem),
                "occupancy");
        moe_blocks_ = sm_count_ * tc_env("FULLM_MOE_BLOCKS_PER_SM", std::max(occ, 1));
        // at most this many blocks (the selftest's small grid: many items per block)
        moe_blocks_ = std::min(moe_blocks_, tc_env("FULLM_MOE_GRID", moe_blocks_));
        // the fused expert kernel's blocks wait on each other: never more than can be resident
        int occ_f = INT_MAX;
        for (auto *k : {(const void *)moe_tc_fused_kernel<1, true, 1>, (const void *)moe_tc_fused_kernel<1, false, 2>,
                        (const void *)moe_tc_fused_kernel<4, true, 2, true>, (const void *)moe_tc_fused_kernel<4, false, 1>,
                        (const void *)moe_tc_fused_kernel<5, true, 2, true>}) {
            int o = 0;
            SK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, k, kTcWarps * 32, kTcSmem), "occupancy");
            occ_f = std::min(occ_f, o);
        }
        moe_fused_blocks_ = std::min(moe_blocks_, sm_count_ * occ_f);
        attn_blocks_ = tc_env("FULLM_ATTN_BLOCKS", 4 * sm_count_);
        i4gemm_ok_ = kt_.int4_gemm;
        for (auto *k : {(const void *)moe_gemm_kernel<true>, (const void *)moe_gemm_kernel<false>})
            SK_CUDA(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kGeSmem)),
                    "expert GEMM smem");
        SK_CUDA(cudaFuncSetAttribute(gqa_prefill_kernel<64>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(pf_smem<64>())),
                "prefill attention smem");
        SK_CUDA(cudaFuncSetAttribute(gqa_prefill_kernel<128>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(pf_smem<128>())),
                "prefill attention smem");
        SK_CUDA(cudaFuncSetAttribute(gqa_attn_tc_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(atc_smem<2>())),
                "attention smem");
        // the bf16 ring at head_dim 128: 3 stages (102 KB) where the device allows, else 2 (68 KB)
        atc_stages_ = atc_smem<4>() <= smem_optin_ ? kAtcStages : 2;
        if (atc_stages_ == kAtcStages)
            SK_CUDA(cudaFuncSetAttribute(gqa_attn_tc_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(atc_smem<4>())),
                    "attention smem");
        else
            SK_CUDA(cudaFuncSetAttribute(gqa_attn_tc_kernel<4, false, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(atc_smem<4, false, 2>())),
                    "attention smem");
        SK_CUDA(cudaFuncSetAttribute(gqa_attn_tc_kernel<2, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(atc_smem<2, true>())),
                "attention smem");
        SK_CUDA(cudaFuncSetAttribute(gqa_attn_tc_kernel<4, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(atc_smem<4, true>())),
                "attention smem");
        SK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, moe_gemm_kernel<true>, kGeThreads, kGeSmem),
                "occupancy");
        ge_blocks_ = sm_count_ * std::max(occ, 1);
        SK_CUDA(cudaMalloc(&router_count_, sizeof(unsigned)), "cudaMalloc");
        SK_CUDA(cudaMemset(router_count_, 0, sizeof(unsigned)), "cudaMemset");
        SK_CUDA(cudaMalloc(&moe_done_, sizeof(unsigned) * 2 * kMoeMaxSlots), "cudaMalloc");
        SK_CUDA(cudaMemset(moe_done_, 0, sizeof(unsigned) * 2 * kMoeMaxSlots), "cudaMemset");
        SK_CUDA(cudaMalloc(&attn_count_, sizeof(unsigned) * kAttnCounters), "cudaMalloc");
        SK_CUDA(cudaMemset(attn_count_, 0, sizeof(unsigned) * kAttnCounters), "cudaMemset");
    }

    ~CudaBackend() override {
        cudaSetDevice(dev_);
        cudaStreamSynchronize(s_);
        for (Scratch *s : {&wbf_, &xbf_, &part_, &moe_route_, &argmax_part_, &sample_w_, &lp_part_, &router_, &xb16_, &moe16_, &moe_xs_,
                           &moe_g16_, &moe_dn_, &cutlass_args_, &cutlass_ws_, &fmha_k_, &fmha_v_, &fmha_q_, &fmha_o_,
                           &fmha_lse_, &fmha_ws_, &twin_y_, &twin_q_, &fa3_meta_, &fa3_sched_, &fa3_lse_, &fa3_oacc_, &fa3_lseacc_, &fa3_q8_, &fa3_sems_, &qkv_buf_, &twin_a_,
                           &moe_sfa_, &xh16_, &xh_trunk_, &kv8_k_, &kv8_v_, &kv8_src_, &kv8_pt_, &xq8_, &sx8_, &lt_acc_,
                           &sm_w13_, &sm_s13_, &sm_w2_, &sm_s2_, &sm_xq_, &sm_sx_, &sm_d13_, &sm_aq_, &sm_sa_, &lora_xa_, &lora_part_, &prompt_,
                           &mla_lse_, &mla_acc_, &mla_sched_})
            if (s->p) cudaFree(s->p);
        for (auto &kv : w8_bf16_) cudaFree(kv.second);
        if (router_count_) cudaFree(router_count_);
        if (moe_done_) cudaFree(moe_done_);
        if (attn_count_) cudaFree(attn_count_);
        if (blas_) cublasDestroy(blas_);
        for (auto &kv : lt_plans_) {
            LtPlan &p = kv.second;
            if (p.la) cublasLtMatrixLayoutDestroy(p.la);
            if (p.lb) cublasLtMatrixLayoutDestroy(p.lb);
            if (p.lc) cublasLtMatrixLayoutDestroy(p.lc);
            if (p.op) cublasLtMatmulDescDestroy(p.op);
        }
        if (lt_ws_) cudaFree(lt_ws_);
        if (lt_) cublasLtDestroy(lt_);
    }
    Caps caps() const override {
        Caps c;
        c.memory = MemKind::Discrete;
        c.graphs = true;
        c.async_copy = true;
        c.dense_formats = fmt_bit(WFmt::F32) | fmt_bit(WFmt::BF16) | fmt_bit(WFmt::INT8) | fmt_bit(WFmt::FP8);  // fp8b and int4: resident experts only
        c.expert_formats = fmt_bit(WFmt::BF16) | fmt_bit(WFmt::INT8) | fmt_bit(WFmt::FP8) | fmt_bit(WFmt::FP8B) |
                           fmt_bit(WFmt::INT4G);
        c.kv_formats = kv_bit(DType::F32) | kv_bit(DType::BF16);
        if (bx_) {  // the fp8 caches' kernels take 16-bit activations
            c.kv_formats |= kv_bit(DType::FP8);
            if (kt_.fa3_fp8) c.kv_formats |= kv_bit(DType::FP8E4M3);  // FA3 (head_dim 128)
        }
        c.kv8_page_multiple = 16;
        c.families = FAM_ALL;
        c.fp8_w8a8 = lt_ != nullptr;
        c.fp8_w8a8_small = lt_ != nullptr && fp8_min_elems_ <= 0;
        return c;
    }
    Policy policy() const override {
        Policy p;
        p.expert_tile_max_rows = kInt4TileRows;
        p.expert_gemm_min_rows = i4gemm_rows_;
        return p;
    }

    void set_kv_pages(int64_t n) override { kv_pages_ = n; }
    void set_expert_format(WFmt f) override { expert_fmt_ = f; }
    void set_exact_activations(bool exact) override { bx_ = !exact; }
    void set_blas_min_rows(int64_t rows) override {
        if (!std::getenv("FULLM_BLAS_MIN_ROWS")) blas_min_rows_ = std::max<int64_t>(1, rows);
    }
    void set_int4_gemm_rows(int64_t rows) override {
        if (!std::getenv("FULLM_INT4_GEMM_ROWS")) i4gemm_rows_ = std::max<int64_t>(1, rows);
    }

    // ---------------------------------------------------------------- dense
    void gemm(const Weight &w, const float *x, int64_t M, float *y, int64_t ldy) override {
        if (M <= 0 || w.rows <= 0) return;
        twin_clear(y);
        if (ldy < w.rows) cuda_fail("gemm", "ldy < rows");
        if ((w.fmt == WFmt::INT8 || w.fmt == WFmt::FP8) && !w.scale) cuda_fail("gemm", "8-bit weight without scales");
        if (w.fmt == WFmt::FP8B || w.fmt == WFmt::INT4G) cuda_fail("gemm", "fp8b and int4 are for resident experts only");
        bind();
        if (!blas_rows(w, M)) {
            const int m = static_cast<int>(M);
            // bf16 weights read x's bf16 rows when a producer left them (attention's output); every
            // other path reads fp32, written now if x was left bf16-only
            const uint16_t *xb = w.fmt == WFmt::BF16 ? xb_rows(w, x, M) : nullptr;
            if (!xb) materialize(x);
            switch (w.fmt) {
                case WFmt::F32: dispatch_gemv<0>(w, x, m, y, ldy, s_, sm_count_, bx_); break;
                case WFmt::BF16:
                    dispatch_gemv<1>(w, x, m, y, ldy, s_, sm_count_, bx_, -1, {}, 1, xb);
                    break;
                case WFmt::INT8: dispatch_gemv<2>(w, x, m, y, ldy, s_, sm_count_, bx_); break;
                case WFmt::FP8:
                    dispatch_gemv<3>(w, x, m, y, ldy, s_, sm_count_, bx_, -1, {}, 1, nullptr, xh_rows(w, x, M));
                    break;
                case WFmt::FP8B:
                case WFmt::INT4G: cuda_fail("gemm", "fp8b and int4 are for resident experts only"); break;
            }
            SK_LAUNCH("gemm/gemv");
            return;
        }
        if (LtPlan *p8 = fp8_plan(w, M, ldy, false)) return gemm_fp8(w, x, M, y, nullptr, ldy, *p8);
        gemm_blas(w, x, M, y, ldy);
    }
    // GEMV rows with bf16 activations: x's bf16 twin when a producer wrote one (x rounded: the
    // same bits), else for xb_min_ or more rows x rounded to bf16 once, into scratch (null when
    // not applicable: the GEMV then rounds x itself, to the same bits).
    const uint16_t *xb_rows(const Weight &w, const float *x, int64_t M) {
        if (!bx_ || blas_rows(w, M) || w.fmt != WFmt::BF16 || !gemv_tc_ok(w, x, w.cols)) return nullptr;
        if (const uint16_t *t = twin_get(x, M, w.cols)) return t;
        if (M < xb_min_) return nullptr;
        const int64_t n4 = M * w.cols / 4;
        auto *xb = static_cast<uint16_t *>(ensure(xb16_, sizeof(uint16_t) * M * w.cols));
        launch_pdl(x_to_bf16_kernel, dim3(static_cast<unsigned>(std::min<int64_t>(ceil_div(n4, kThreads), 4 * sm_count_))),
                   dim3(kThreads), 0, s_, xb, x, n4);
        return xb;
    }
    // ... and for fp8 weights, x as fp16 (rounded as the tile's f16x2 rounds it: the same bits),
    // so the tile's MMAs take it whole and only the weights are converted
    const __half *xh_rows(const Weight &w, const float *x, int64_t M) {
        if (!bx_ || M < xb_min_ || blas_rows(w, M) || w.fmt != WFmt::FP8 || !gemv_tc_ok(w, x, w.cols)) return nullptr;
        materialize(x);
        const int64_t n4 = M * w.cols / 4;
        auto *xh = static_cast<__half *>(ensure(xh_trunk_, sizeof(__half) * M * w.cols));
        launch_pdl(x_to_f16_kernel, dim3(static_cast<unsigned>(std::min<int64_t>(ceil_div(n4, kThreads), 4 * sm_count_))),
                   dim3(kThreads), 0, s_, xh, x, n4);
        return xh;
    }
    void gemm_group(const GemmJob *jobs, int n, int64_t M) override {
        if (n <= 0 || M <= 0) return;
        bool blas = true;  // cuBLAS when it takes every job (fp8 weights keep the GEMV to 16 rows)
        for (int i = 0; i < n; ++i) twin_clear(jobs[i].y), blas = blas && blas_rows(*jobs[i].w, M);
        if (blas) {  // cuBLAS: x rounded to bf16 once for the jobs that share it
            bind();
            // q, k and v (or a dense MLP's gate and up) whose bf16 weights sit back to back (dense_exec
            // lays them out so) and that only the backend's next op reads (qk_rope_append, silu_glu):
            // one GEMM over them, its bf16 result read in place (launches fewer per layer; k and v
            // alone are a few rows; gate and up as one GEMM: Qwen3-8B c=16 2.78 -> 2.65 ms a step)
            int nf = n >= 3 && bx_ && fusable(jobs, 3) ? 3 : n == 2 && bx_ && fusable(jobs, 2) ? 2 : 0;
            Weight wf;
            LtPlan *pf = nullptr;  // fp8 stacks: the fp8 GEMM's plan for the stacked rows, else the stack's bf16 copy
            if (nf) {
                wf = *jobs[0].w;
                for (int i = 1; i < nf; ++i) wf.rows += jobs[i].w->rows;
                if (wf.fmt == WFmt::FP8 && !(pf = fp8_plan(wf, M, wf.rows, true)) && !w8_bf16(wf)) nf = 0;
            }
            if (nf) {
                if (qkv_.b) qkv_materialize();  // one fused result at a time
                auto *yf = static_cast<uint16_t *>(ensure(qkv_buf_, sizeof(uint16_t) * M * wf.rows));
                if (pf) gemm_fp8(wf, jobs[0].x, M, nullptr, yf, wf.rows, *pf);
                else gemm_blas16_out16(wf, x_bf16(jobs[0].x, M, wf.cols), M, yf, wf.rows);
                qkv_ = QkvFused{jobs[0].y, jobs[1].y, nf == 3 ? jobs[2].y : nullptr, M, wf.rows, jobs[0].w->rows,
                                jobs[1].w->rows, nf == 3 ? jobs[2].w->rows : 0, yf};
                jobs += nf, n -= nf;
            }
            const float *xb_src = nullptr;  // the x that xb_group_ holds as bf16
            int64_t xb_cols = 0;
            for (int i = 0; i < n; ++i) {
                const Weight &w = *jobs[i].w;
                if (w.rows <= 0) continue;
                if (jobs[i].ldy < w.rows) cuda_fail("gemm", "ldy < rows");
                if ((w.fmt == WFmt::INT8 || w.fmt == WFmt::FP8) && !w.scale) cuda_fail("gemm", "8-bit weight without scales");
                if (w.fmt == WFmt::FP8B || w.fmt == WFmt::INT4G) cuda_fail("gemm", "fp8b and int4 are for resident experts only");
                // a result its consumer takes as bf16: written as the (lazy) twin only
                const bool want16 = jobs[i].y16_ok && i < 4 && jobs[i].ldy == w.rows;
                LtPlan *p8 = fp8_plan(w, M, jobs[i].ldy, want16);
                uint16_t *y16 = want16 && (w.fmt == WFmt::BF16 || p8 || w8_bf16(w))
                                    ? twin_out_lazy(twin_g_[i], jobs[i].y, M, w.rows)
                                    : nullptr;
                if (p8 && want16 && !y16) p8 = fp8_plan(w, M, jobs[i].ldy, false);  // no twin: fp32 out
                if (p8) {
                    gemm_fp8(w, jobs[i].x, M, y16 ? nullptr : jobs[i].y, y16, jobs[i].ldy, *p8);
                    continue;
                }
                if (jobs[i].x != xb_src || w.cols != xb_cols)
                    xb_group_ = x_bf16(jobs[i].x, M, w.cols), xb_src = jobs[i].x, xb_cols = w.cols;
                if (y16) gemm_blas16_out16(w, xb_group_, M, y16, jobs[i].ldy);
                else gemm_blas16(w, xb_group_, M, jobs[i].y, jobs[i].ldy);
            }
            return;
        }
        // Tensor-core-eligible jobs share one launch; the rest (f32 weights such as the router,
        // odd shapes, steps above kGemvMaxRows) run as single calls, which compute the same bits.
        GemvJobs g{};
        int64_t blocks = 0;
        for (int i = 0; i < n; ++i) {
            const Weight &w = *jobs[i].w;
            const bool ok = !blas_rows(w, M) && g.n < kMaxGemvJobs && w.rows > 0 && gemv_tc_ok(w, jobs[i].x, w.cols) &&
                            jobs[i].ldy >= w.rows && (w.fmt == WFmt::BF16 || w.scale);
            if (!ok) {
                gemm(w, jobs[i].x, M, jobs[i].y, jobs[i].ldy);
                continue;
            }
            const int S = tc_split(w.rows, w.cols, wfmt_bytes(w.fmt));
            g.j[g.n++] = GemvJob{w.data, w.scale, jobs[i].x, jobs[i].y, w.cols, w.rows, jobs[i].ldy,
                                 static_cast<int32_t>(w.fmt), static_cast<int32_t>(blocks), S};
            blocks += ceil_div(w.rows, 8 * (kTcWarps / S));
        }
        if (g.n == 0) return;
        bind();
        const int m = static_cast<int>(M);
        const unsigned grid = static_cast<unsigned>(blocks), threads = kTcWarps * 32;
        // one bf16 copy of x when every job is bf16 weights on the same x
        bool same = true;
        for (int i = 0; i < g.n; ++i) same = same && g.j[i].fmt == 1 && g.j[i].x == g.j[0].x && g.j[i].K == g.j[0].K;
        Weight w0;  // the shape the bf16 copy is made for
        w0.fmt = WFmt::BF16, w0.rows = g.j[0].N, w0.cols = g.j[0].K, w0.data = g.j[0].W;
        if (const uint16_t *xb = same ? xb_rows(w0, static_cast<const float *>(g.j[0].x), M) : nullptr) {
            for (int i = 0; i < g.n; ++i) g.j[i].x = xb;
            if (m > 8) launch_pdl(gemv_group_kernel<true, true, true>, dim3(grid), dim3(threads), kTcSmem, s_, g, m);
            else launch_pdl(gemv_group_kernel<false, true, true>, dim3(grid), dim3(threads), kTcSmem, s_, g, m);
            SK_LAUNCH("gemm_group");
            return;
        }
        // ... or one fp16 copy when every job is fp8 weights on the same x
        bool same8 = true;
        for (int i = 0; i < g.n; ++i) same8 = same8 && g.j[i].fmt == 3 && g.j[i].x == g.j[0].x && g.j[i].K == g.j[0].K;
        Weight w8 = w0;
        w8.fmt = WFmt::FP8;
        if (const __half *xh = same8 ? xh_rows(w8, static_cast<const float *>(g.j[0].x), M) : nullptr) {
            for (int i = 0; i < g.n; ++i) g.j[i].x = xh;
            if (m > 8) launch_pdl(gemv_group_kernel<true, true, false, true>, dim3(grid), dim3(threads), kTcSmem, s_, g, m);
            else launch_pdl(gemv_group_kernel<false, true, false, true>, dim3(grid), dim3(threads), kTcSmem, s_, g, m);
            SK_LAUNCH("gemm_group");
            return;
        }
        for (int i = 0; i < g.n; ++i) materialize(static_cast<const float *>(g.j[i].x));
        if (bx_) {
            if (m > 8) launch_pdl(gemv_group_kernel<true, true>, dim3(grid), dim3(threads), kTcSmem, s_, g, m);
            else launch_pdl(gemv_group_kernel<false, true>, dim3(grid), dim3(threads), kTcSmem, s_, g, m);
        } else {
            if (m > 8) launch_pdl(gemv_group_kernel<true, false>, dim3(grid), dim3(threads), kTcSmem, s_, g, m);
            else launch_pdl(gemv_group_kernel<false, false>, dim3(grid), dim3(threads), kTcSmem, s_, g, m);
        }
        SK_LAUNCH("gemm_group");
    }

    // ---------------------------------------------------------------- elementwise
    void rmsnorm(float *y, const float *x, const float *w, int64_t rows, int64_t n, int64_t stride,
                 float eps) override {
        materialize(x);
        materialize(w);
        twin_clear(y);
        if (rows <= 0 || n <= 0) return;
        bind();
        const dim3 g(static_cast<unsigned>(rows)), b(kThreads);
        const bool vec = stride % 4 == 0 && aligned16(x) && aligned16(y) && aligned16(w);
        switch (vec ? norm_per(n) : 0) {
        case 4: rmsnorm_vec_kernel<4><<<g, b, 0, s_>>>(y, x, w, n, stride, eps); break;
        case 8: rmsnorm_vec_kernel<8><<<g, b, 0, s_>>>(y, x, w, n, stride, eps); break;
        case 16: rmsnorm_vec_kernel<16><<<g, b, 0, s_>>>(y, x, w, n, stride, eps); break;
        default: rmsnorm_kernel<<<g, b, 0, s_>>>(y, x, w, n, stride, eps);
        }
        SK_LAUNCH("rmsnorm");
    }
    // PER of the vectorised norm kernels for a row of n (4-wide chunks, the row in registers):
    // 0 = the scalar loop. One rule for rmsnorm, add_rmsnorm and the fused MoE reduce, so the
    // sums of squares (and the selftest's bitwise comparisons) agree.
    static int norm_per(int64_t n) {
        if (n % 4) return 0;
        if (n <= 4 * kThreads) return 4;
        if (n <= 8 * kThreads) return 8;
        if (n <= 16 * kThreads) return 16;
        return 0;
    }
    // the chunked kernels' 16-byte (fp32) and 8-byte (bf16) accesses need aligned rows: the base
    // pointers (cudaMalloc gives 256) and, for rmsnorm, a stride that is a multiple of 4 floats
    static bool aligned16(const void *p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }
    static bool aligned8(const void *p) { return (reinterpret_cast<uintptr_t>(p) & 7) == 0; }
    void add_rmsnorm(float *h, const float *d, float *y, const float *w, int64_t rows, int64_t n,
                     float eps) override {
        if (rows <= 0 || n <= 0) return;
        twin_clear(h);
        twin_clear(y);
        bind();
        const float *none = nullptr;
        const int32_t *noi = nullptr;
        // d through its bf16 twin when gemm16 left only that; y bf16-only (lazy) in the bf16 prefill
        const uint16_t *d16 = twin_get(d, rows, n);
        if (!d16) materialize(d);
        uint16_t *y16 = n <= 16 * kThreads ? twin_out_lazy(twin_y_, y, rows, n) : nullptr;
        float *yo = y16 ? nullptr : y;
        // GEMV-sized steps: y written both ways, its bf16 rows registered as the twin, which the next
        // GEMV (xb_rows) and the fused router read instead of rounding x themselves (the same bits)
        if (!y16 && bx_ && rows <= kGemvMaxRows && n <= 16 * kThreads && (rows * n) % 4 == 0) {
            y16 = static_cast<uint16_t *>(ensure(twin_y_, sizeof(uint16_t) * rows * n));
            twin_set(y, rows, n, y16);
        }
        const bool vec = aligned16(h) && aligned16(w) && (!yo || aligned16(yo)) && (d16 ? aligned8(d16) : aligned16(d)) &&
                         (!y16 || aligned8(y16));
        const int per = vec ? norm_per(n) : 0;
        // few wide rows (a row a block would leave most SMs idle): each row over a cluster of 4
        // blocks up to sm_count / 4 rows, of 2 up to sm_count / 2; the same bits (cluster_sum).
        // H100, Qwen3-8B (rows of 4096) c=4/8/16/32 +1.4/+1.6/+1.4/+0.6% (nsys at 16: 4.4 -> 3.5 us
        // a call); rows of 2048 (Qwen3-30B-A3B) lost 0.5-1.8%, so PER 16 only.
        int cl = 1;
        if (per == 16 && kt_.cc >= 90) {
            if (norm_cluster_ >= 4 && rows * 4 <= sm_count_) cl = 4;
            else if (norm_cluster_ >= 2 && rows * 2 <= sm_count_) cl = 2;
        }
#define SK_ARN(PER, DT, DP, CL)                                                                                          \
    launch_pdl_cluster(add_rmsnorm_kernel<PER, false, float, DT, CL>, dim3(static_cast<unsigned>(rows * CL)),           \
                       dim3(kThreads / CL), 0, s_, CL, h, DP, yo, w, n, eps, 0, none, none, noi, none, y16)
#define SK_ARN_CL(PER, DT, DP) \
    (cl == 4 ? SK_ARN(PER, DT, DP, 4) : cl == 2 ? SK_ARN(PER, DT, DP, 2) : SK_ARN(PER, DT, DP, 1))
        switch (per) {
        case 4:
            if (d16) SK_ARN_CL(4, uint16_t, d16);
            else SK_ARN_CL(4, float, d);
            break;
        case 8:
            if (d16) SK_ARN_CL(8, uint16_t, d16);
            else SK_ARN_CL(8, float, d);
            break;
        case 16:
            if (d16) SK_ARN_CL(16, uint16_t, d16);
            else SK_ARN_CL(16, float, d);
            break;
        default:
            if (d16) SK_ARN(0, uint16_t, d16, 1);
            else SK_ARN(0, float, d, 1);
        }
#undef SK_ARN_CL
#undef SK_ARN
        SK_LAUNCH("add_rmsnorm");
    }
    void moe_reduce_add_rmsnorm(float *h, const float *contrib, int64_t N, int64_t k, int64_t width,
                                const float *addend, float *y, const float *w, float eps, float *tmp) override {
        if (N <= 0 || width <= 0) return;
        // the fused kernel is the chunked form only (the base pointers are the backend's own buffers)
        if (norm_per(width) == 0 || k > kMaxReduceK || !aligned16(h) || !aligned16(w) || !aligned16(contrib) ||
            (addend && !aligned16(addend)) || (y && !aligned16(y)))
            return Backend::moe_reduce_add_rmsnorm(h, contrib, N, k, width, addend, y, w, eps, tmp);
        bind();
        twin_clear(h);
        twin_clear(y);
        const dim3 g(static_cast<unsigned>(N)), b(kThreads);
        const int kk = static_cast<int>(k);
        const Sorted so = take_sorted(contrib);
        uint16_t *y16 = twin_out_lazy(twin_y_, y, N, width);
        float *yo = y16 ? nullptr : y;  // bf16-only when the twin exists
        // GEMV-sized steps: the next layer's q/k/v GEMV takes this output as bf16 (xb_rows), so it
        // is written both ways, the bf16 rows registered as the twin (as in add_rmsnorm)
        if (!y16 && y && bx_ && N <= kGemvMaxRows && (N * width) % 4 == 0) {
            y16 = static_cast<uint16_t *>(ensure(twin_y_, sizeof(uint16_t) * N * width));
            twin_set(y, N, width, y16);
        }
        const int per = norm_per(width);
#define SK_RED(PER, T, DN) \
    launch_pdl(add_rmsnorm_kernel<PER, true, T>, g, b, 0, s_, h, contrib, yo, w, width, eps, kk, addend, DN, so.inv, so.wts, y16)
        if (so.dn16) {
            auto *dn = static_cast<const uint16_t *>(so.dn);
            if (per == 4) SK_RED(4, uint16_t, dn);
            else if (per == 8) SK_RED(8, uint16_t, dn);
            else SK_RED(16, uint16_t, dn);
        } else {
            auto *dn = static_cast<const float *>(so.dn);
            if (per == 4) SK_RED(4, float, dn);
            else if (per == 8) SK_RED(8, float, dn);
            else SK_RED(16, float, dn);
#undef SK_RED
        }
        SK_LAUNCH("moe_reduce_add_rmsnorm");
    }
    bool kda_fused(const KdaFused &f) override {
        if (f.K != 4 || f.nseg <= 0) return false;
        const int64_t hist = f.H * f.D * (f.K - 1);
        const dim3 grid(static_cast<unsigned>(f.H), static_cast<unsigned>(f.nseg));
        bind();
#define SK_KF(DD, ST)                                                                                                \
    kda_fused_kernel<DD, 3, ST><<<grid, DD, 0, s_>>>(                                                                \
        f.o, f.qp, f.kp, f.vp, f.wq, f.wk, f.wv, f.conv_state, f.conv_stride, hist, f.z, f.braw, f.a_log, f.dt_bias, \
        f.lb, f.softplus ? 1 : 0, f.gate, f.o_norm, f.out_eps, f.l2_eps, f.qscale, static_cast<ST *>(f.state),     \
        f.state_stride, f.seg, static_cast<int>(f.H))
#define SK_KF_D(DD)                                     \
    if (f.state_dtype == DType::F32) SK_KF(DD, float); \
    else SK_KF(DD, __nv_bfloat16)
        switch (f.D) {
            case 16: SK_KF_D(16); break;
            case 32: SK_KF_D(32); break;
            case 64: SK_KF_D(64); break;
            case 128: SK_KF_D(128); break;
            default: return false;
        }
#undef SK_KF_D
#undef SK_KF
        SK_LAUNCH("kda_fused");
        return true;
    }
    // A prompt's KDA in chunks (kernels/kda_chunk.cuh): every chunk's matrices at once (the
    // convolutions, norms and gates per token included), the slots' convolution histories, the
    // chunks in order over slices of the state's columns, the output norm and gate. Prompts below
    // kda_chunk_min run kda_fused (a step's segments are split by the executor). The tensor-core
    // kernels need sm80; the scratch is shared with mla_prefill (prompt_).
    int64_t kda_chunk_min() const override { return kt_.cc >= 80 ? kda_chunk_min_ : 0; }
    size_t kda_chunked_scratch(int64_t n_tokens, int64_t n_segs, int64_t H, int64_t D) const override {
        if (kda_chunk_min() <= 0) return 0;
        const int64_t C = kKdaChunk, segs = std::min(n_segs, std::max<int64_t>(1, n_tokens / kda_chunk_min_));
        return sizeof(float) * (n_tokens / C + segs) * H * (5 * C * D + D);
    }
    bool kda_chunked(const KdaFused &f, const int32_t *seg_host) override {
        if (f.K != 4 || f.nseg <= 0 || !seg_host || kda_chunk_min() <= 0 || (f.D != 64 && f.D != 128)) return false;
        constexpr int C = kKdaChunk, VS = kKdaSlice;
        int64_t maxlen = 0, chunks = 0;
        for (int64_t i = 0; i < f.nseg; ++i)
            maxlen = std::max<int64_t>(maxlen, seg_host[3 * i + 2]), chunks += ceil_div(int64_t{seg_host[3 * i + 2]}, int64_t(C));
        if (maxlen <= 0) return true;
        const int64_t P = f.H * f.D, maxch = ceil_div(maxlen, int64_t(C)), hist = P * (f.K - 1);
        bind();
        auto *scr = static_cast<float *>(ensure(prompt_, sizeof(float) * chunks * f.H * (5 * C * f.D + f.D)));
        const unsigned ns = static_cast<unsigned>(f.nseg), H = static_cast<unsigned>(f.H);
#define SK_KC(DD)                                                                                                       \
    {                                                                                                                   \
        constexpr int ss = kda_scan_smem<DD, C, VS>(), ts = KdaScanTc<DD, C, VS>::SMEM, pt = KdaPrepTc<DD, C>::SMEM;  \
        static const bool attr = [] {                                                                                   \
            cudaFuncSetAttribute(kda_chunk_prep_tc_kernel<DD, C, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, pt);  \
            cudaFuncSetAttribute(kda_chunk_scan_kernel<DD, C, VS, float>, cudaFuncAttributeMaxDynamicSharedMemorySize, ss); \
            cudaFuncSetAttribute(kda_chunk_scan_kernel<DD, C, VS, __nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize, ss); \
            cudaFuncSetAttribute(kda_chunk_scan_tc_kernel<DD, C, VS, float>, cudaFuncAttributeMaxDynamicSharedMemorySize, ts); \
            cudaFuncSetAttribute(kda_chunk_scan_tc_kernel<DD, C, VS, __nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize, ts); \
            return true;                                                                                                \
        }();                                                                                                            \
        (void)attr;                                                                                                     \
        kda_chunk_prep_tc_kernel<DD, C, 3><<<dim3(static_cast<unsigned>(maxch), H, ns), 256, pt, s_>>>(               \
            scr, f.qp, f.kp, f.vp, f.wq, f.wk, f.wv, f.conv_state, f.conv_stride, hist, f.z, f.braw, f.a_log,          \
            f.dt_bias, f.lb, f.softplus ? 1 : 0, f.l2_eps, f.qscale, f.seg, static_cast<int>(f.H));                     \
        kda_conv_hist_kernel<3><<<dim3(static_cast<unsigned>(ceil_div(P, int64_t(256))), ns), 256, 0, s_>>>(            \
            f.conv_state, f.conv_stride, hist, f.qp, f.kp, f.vp, f.seg, P);                                             \
        const dim3 sg(DD / VS, H, ns);                                                                                  \
        if (kda_scan_tc_ && f.state_dtype == DType::F32)                                                                \
            kda_chunk_scan_tc_kernel<DD, C, VS, float><<<sg, 256, ts, s_>>>(f.o, scr, static_cast<float *>(f.state),   \
                                                                            f.state_stride, f.seg, static_cast<int>(f.H)); \
        else if (kda_scan_tc_)                                                                                          \
            kda_chunk_scan_tc_kernel<DD, C, VS, __nv_bfloat16><<<sg, 256, ts, s_>>>(                                    \
                f.o, scr, static_cast<__nv_bfloat16 *>(f.state), f.state_stride, f.seg, static_cast<int>(f.H));        \
        else if (f.state_dtype == DType::F32)                                                                           \
            kda_chunk_scan_kernel<DD, C, VS, float><<<sg, 256, ss, s_>>>(f.o, scr, static_cast<float *>(f.state),      \
                                                                         f.state_stride, f.seg, static_cast<int>(f.H)); \
        else                                                                                                            \
            kda_chunk_scan_kernel<DD, C, VS, __nv_bfloat16><<<sg, 256, ss, s_>>>(                                       \
                f.o, scr, static_cast<__nv_bfloat16 *>(f.state), f.state_stride, f.seg, static_cast<int>(f.H));        \
    }
        if (f.D == 64) SK_KC(64) else SK_KC(128)
#undef SK_KC
        kda_out_seg_kernel<<<dim3(static_cast<unsigned>(ceil_div(maxlen * f.H, int64_t(8))), ns), 256, 0, s_>>>(
            f.o, f.gate, f.o_norm, f.seg, f.H, f.D, f.out_eps);
        SK_LAUNCH("kda_chunked");
        return true;
    }
    void l2norm_heads(float *x, int64_t rows, int64_t heads, int64_t d, float eps) override {
        const int64_t nvec = rows * heads;
        if (nvec <= 0 || d <= 0) return;
        bind();
        l2norm_kernel<<<static_cast<unsigned>((nvec + 7) / 8), 256, 0, s_>>>(x, nvec, d, eps);
        SK_LAUNCH("l2norm_heads");
    }
    void situ_glu(float *y, const float *gate, const float *up, int64_t n, float b1, float b2) override {
        materialize(gate);
        materialize(up);
        if (n <= 0) return;
        bind();
        situ_kernel<<<grid1d(n), kThreads, 0, s_>>>(y, gate, up, n, b1, b2);
        SK_LAUNCH("situ_glu");
    }
    void silu_glu(float *y, const float *gate, const float *up, int64_t n) override {
        if (silu_glu16(y, gate, up, n)) return;
        materialize(gate);
        materialize(up);
        twin_clear(y);
        if (n <= 0) return;
        bind();
        silu_kernel<<<grid1d(n), kThreads, 0, s_>>>(y, gate, up, n);
        SK_LAUNCH("silu_glu");
    }
    // gate and up as bf16 only (gemm_group's fused [M][ld] matrix, or both GEMMs' lazy twins): y
    // written as a bf16-only twin, which the down projection's GEMM reads as its operand (x_bf16,
    // xb_rows); an fp32 reader materializes it. False: not applicable, nothing done.
    bool silu_glu16(float *y, const float *gate, const float *up, int64_t n) {
        if (!bx_ || n <= 0) return false;
        const uint16_t *g16 = nullptr, *u16 = nullptr;
        int64_t M = 0, I = 0, ld = 0;
        const bool fused = qkv_.b && qkv_.q == gate && qkv_.k == up && !qkv_.v && qkv_.nq == qkv_.nk && qkv_.M * qkv_.nq == n;
        if (fused) {
            g16 = qkv_.b, u16 = qkv_.b + qkv_.nq, M = qkv_.M, I = qkv_.nq, ld = qkv_.ld;
        } else {
            for (const auto &t : twins_)
                if (t.x == gate && t.lazy && t.M * t.K == n) g16 = t.b, M = t.M, I = t.K;
            if (g16) u16 = twin_lazy(up, M, I);
            ld = I;
        }
        if (!g16 || !u16 || (M * I) % 4) return false;
        bind();
        twin_clear(y);
        auto *y16 = static_cast<uint16_t *>(ensure(twin_a_, sizeof(uint16_t) * M * I));
        if (I % 8 == 0 && ld % 8 == 0 && aligned16(g16) && aligned16(u16) && aligned16(y16))
            silu_b16_kernel<8><<<grid1d(M * I / 8), kThreads, 0, s_>>>(y16, g16, u16, ld, I, M * I);
        else
            silu_b16_kernel<1><<<grid1d(M * I), kThreads, 0, s_>>>(y16, g16, u16, ld, I, M * I);
        SK_LAUNCH("silu_glu16");
        // consumed: gate and up are not read again (their twins' slots freed)
        if (fused) qkv_ = QkvFused{};
        else twin_clear(gate), twin_clear(up);
        twin_set(y, M, I, y16, true);
        return true;
    }
    void add(float *y, const float *x, int64_t n) override {
        materialize(x);
        twin_clear(y);
        if (n <= 0) return;
        bind();
        add_kernel<<<grid1d(n), kThreads, 0, s_>>>(y, x, n);
        SK_LAUNCH("add");
    }
    void sigmoid_mul(float *y, const float *g, int64_t n) override {
        materialize(g);
        if (n <= 0) return;
        bind();
        sigmoid_mul_kernel<<<grid1d(n), kThreads, 0, s_>>>(y, g, n);
        SK_LAUNCH("sigmoid_mul");
    }
    void copy_rows(float *dst, int64_t dst_stride, int64_t dst_off, const float *src, int64_t rows,
                   int64_t n) override {
        materialize(src);
        if (rows <= 0 || n <= 0) return;
        bind();
        copy_rows_kernel<<<grid1d(rows * n), kThreads, 0, s_>>>(dst, dst_stride, dst_off, src, rows, n);
        SK_LAUNCH("copy_rows");
    }
    void attn_res(float *out, const float *blocks, int64_t nb, int64_t maxs, const float *pref, const float *fold,
                  int64_t rows, int64_t n, float eps) override {
        materialize(blocks);
        materialize(pref);
        materialize(fold);
        if (rows <= 0 || n <= 0) return;
        if (nb < 0 || nb + 1 > kMaxSources) cuda_fail("attn_res", "too many sources");
        bind();
        attn_res_kernel<<<static_cast<unsigned>(rows), kThreads, 0, s_>>>(out, blocks, static_cast<int>(nb), maxs, pref,
                                                                         fold, n, eps);
        SK_LAUNCH("attn_res");
    }

    // ---------------------------------------------------------------- KDA
    void kda_conv(float *y, const float *x, const float *w, float *conv_state, int64_t conv_stride, int64_t hist_off,
                  const int32_t *seg, int64_t nseg, int64_t C, int64_t K) override {
        materialize(x);
        materialize(w);
        if (nseg <= 0 || C <= 0) return;
        if (K < 1 || K - 1 > kMaxConvHist) cuda_fail("kda_conv", "kernel size outside 1..9");
        if (nseg > 65535) cuda_fail("kda_conv", "more than 65535 segments");
        bind();
        const dim3 grid(static_cast<unsigned>((C + kThreads - 1) / kThreads), static_cast<unsigned>(nseg));
        kda_conv_kernel<<<grid, kThreads, 0, s_>>>(y, x, w, conv_state, conv_stride, hist_off, seg, C,
                                                    static_cast<int>(K));
        SK_LAUNCH("kda_conv");
    }
    void kda_gates(float *alpha, float *beta, const float *z, const float *braw, const float *a_log,
                   const float *dt_bias, int64_t N, int64_t H, int64_t D, float lb, bool softplus) override {
        materialize(z);
        materialize(braw);
        materialize(a_log);
        materialize(dt_bias);
        if (N <= 0) return;
        bind();
        kda_gates_kernel<<<grid1d(N * H * D), kThreads, 0, s_>>>(alpha, beta, z, braw, a_log, dt_bias, N, H, D, lb,
                                                                  softplus ? 1 : 0);
        SK_LAUNCH("kda_gates");
    }
    void kda_recurrence(float *o, const float *q, const float *k, const float *v, const float *alpha, const float *beta,
                        void *state, DType state_dtype, int64_t state_stride, const int32_t *seg, int64_t nseg,
                        int64_t H, int64_t D, float qscale) override {
        materialize(q);
        materialize(k);
        materialize(v);
        materialize(alpha);
        materialize(beta);
        if (nseg <= 0 || H <= 0) return;
        if (nseg > 65535) cuda_fail("kda_recurrence", "more than 65535 segments");
        bind();
        const dim3 grid(static_cast<unsigned>(H), static_cast<unsigned>(nseg));
        const int h = static_cast<int>(H);
#define SK_REC(DD, ST)                                                                                         \
    kda_rec_kernel<DD, ST><<<grid, DD, 0, s_>>>(o, q, k, v, alpha, beta, static_cast<ST *>(state), state_stride, \
                                                seg, h, qscale)
#define SK_REC_D(DD)                                     \
    if (state_dtype == DType::F32) SK_REC(DD, float);   \
    else SK_REC(DD, __nv_bfloat16);
        switch (D) {
            case 16: SK_REC_D(16) break;
            case 32: SK_REC_D(32) break;
            case 64: SK_REC_D(64) break;
            case 128: SK_REC_D(128) break;
            default: cuda_fail("kda_recurrence", ("unsupported head_dim " + std::to_string(D)).c_str());
        }
#undef SK_REC_D
#undef SK_REC
        SK_LAUNCH("kda_recurrence");
    }
    void kda_out(float *o, const float *gate, const float *o_norm, int64_t N, int64_t H, int64_t D,
                 float eps) override {
        materialize(gate);
        materialize(o_norm);
        const int64_t nvec = N * H;
        if (nvec <= 0) return;
        bind();
        kda_out_kernel<<<static_cast<unsigned>((nvec + 7) / 8), 256, 0, s_>>>(o, gate, o_norm, nvec, D, eps);
        SK_LAUNCH("kda_out");
    }

    // ---------------------------------------------------------------- MLA
    void mla_append(void *kv, DType kv_dtype, const float *ckv, const int32_t *tok_slot, const int32_t *tok_pos,
                    const int32_t *page_table, int64_t max_pages, int64_t page, int64_t width, int64_t N) override {
        materialize(ckv);
        if (N <= 0) return;
        bind();
        if (kv_dtype == DType::F32)
            mla_append_kernel<float><<<grid1d(N * width), kThreads, 0, s_>>>(
                static_cast<float *>(kv), ckv, tok_slot, tok_pos, page_table, max_pages, page, width, N);
        else
            mla_append_kernel<__nv_bfloat16><<<grid1d(N * width), kThreads, 0, s_>>>(
                static_cast<__nv_bfloat16 *>(kv), ckv, tok_slot, tok_pos, page_table, max_pages, page, width, N);
        SK_LAUNCH("mla_append");
    }
    void mla_absorb_q(float *q_abs, const float *q, const Weight &kv_b, int64_t N, int64_t H, int64_t qn, int64_t qr,
                      int64_t vh) override {
        materialize(q);
        if (N <= 0) return;
        check_kvb(kv_b, H, qn, vh, "mla_absorb_q");
        bind();
        const int64_t L = kv_b.cols;
        const dim3 grid(static_cast<unsigned>(H), static_cast<unsigned>((L + kAbsorbCols - 1) / kAbsorbCols),
                        static_cast<unsigned>((N + kTokTile - 1) / kTokTile));
        const size_t smem = sizeof(float) * kTokTile * (qn + 8 * kAbsorbCols);
        if (smem > 48 * 1024)
            SK_CUDA(cudaFuncSetAttribute(mla_absorb_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(smem)),
                    "mla_absorb_q smem");
        mla_absorb_kernel<<<grid, 256, smem, s_>>>(q_abs, q, static_cast<const uint16_t *>(kv_b.data), N,
                                                        static_cast<int>(H), static_cast<int>(qn),
                                                        static_cast<int>(qr), static_cast<int>(vh),
                                                        static_cast<int>(kv_b.cols));
        SK_LAUNCH("mla_absorb_q");
    }
    void mla_attention(float *o_lat, const float *q_abs, const void *kv, DType kv_dtype, const int32_t *tok_slot,
                       const int32_t *tok_pos, const int32_t *page_table, int64_t max_pages, int64_t page, int64_t N,
                       int64_t H, int64_t kv_lora, int64_t qr, int64_t max_pos, float scale) override {
        materialize(q_abs);
        if (N <= 0 || H <= 0) return;
        const int64_t W = kv_lora + qr;
        if (W > kAttnMaxW) cuda_fail("mla_attention", "latent width above 1024");
        bind();
        const int64_t hg = (H + kAttnHeads - 1) / kAttnHeads;
        // Small steps split every token on its own position alone (batch-invariant); larger
        // ones bound the partials buffer instead.
        int max_splits = kAttnMaxSplits;
        if (N > kGemvMaxRows) {
            const int64_t per_split = N * H * (kv_lora + 2) * static_cast<int64_t>(sizeof(float));
            max_splits = static_cast<int>(std::clamp<int64_t>((int64_t{256} << 20) / per_split, 1, kAttnMaxSplits));
        }
        // The split count is not monotonic in position (the split length grows in whole tiles),
        // so size the grid by its bound over every position up to max_pos.
        const int zsplits =
            static_cast<int>(std::min<int64_t>(max_splits, (max_pos + kAttnSplitLen) / kAttnSplitLen));
        float *part = nullptr;
        if (zsplits > 1)
            part = static_cast<float *>(ensure(part_, sizeof(float) * N * H * zsplits * (kv_lora + 2)));

        const size_t smem = sizeof(float) * kAttnTile * W;
        const dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(hg), static_cast<unsigned>(zsplits));
        const dim3 block(kAttnHeads * 32);
        const int h = static_cast<int>(H), L = static_cast<int>(kv_lora), w = static_cast<int>(W);
        auto launch = [&](auto kernel, const auto *kvp) {
            if (smem > 48 * 1024)
                SK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             static_cast<int>(smem)),
                        "mla_attention smem");
            kernel<<<grid, block, smem, s_>>>(o_lat, part, q_abs, kvp, tok_slot, tok_pos, page_table, max_pages, page,
                                              h, L, w, scale, max_splits);
        };
        const bool f32 = kv_dtype == DType::F32;
        const auto *kf = static_cast<const float *>(kv);
        const auto *kb = static_cast<const __nv_bfloat16 *>(kv);
        if (W <= 18 * 32) {  // 576: Kimi K3 and Kimi Linear (512 + 64)
            if (f32) launch(mla_attn_kernel<float, 18>, kf);
            else launch(mla_attn_kernel<__nv_bfloat16, 18>, kb);
        } else {
            if (f32) launch(mla_attn_kernel<float, kAttnMaxW / 32>, kf);
            else launch(mla_attn_kernel<__nv_bfloat16, kAttnMaxW / 32>, kb);
        }
        SK_LAUNCH("mla_attention");
        if (zsplits > 1) {
            mla_combine_kernel<<<static_cast<unsigned>(N * H), 128, 0, s_>>>(o_lat, part, tok_pos, h, L, max_splits,
                                                                            zsplits, kAttnSplitLen);
            SK_LAUNCH("mla_attention/combine");
        }
    }
    void mla_unabsorb(float *y, const float *o_lat, const Weight &kv_b, int64_t N, int64_t H, int64_t qn,
                      int64_t vh) override {
        materialize(o_lat);
        if (N <= 0) return;
        check_kvb(kv_b, H, qn, vh, "mla_unabsorb");
        bind();
        const int64_t L = kv_b.cols;
        if (N <= kGemvMaxRows) {
            // per head h: y[t][h*vh + j] = W_UV^h[j] . o_lat[t][h], the GEMV's row-order sums
            Weight wv = kv_b;
            wv.data = static_cast<const uint16_t *>(kv_b.data) + qn * L;
            wv.rows = vh;
            GemvBatch bt;
            bt.w = (qn + vh) * L;
            bt.x = L;
            bt.y = vh;
            dispatch_gemv<1>(wv, o_lat, static_cast<int>(N), y, H * vh, s_, sm_count_, bx_, H * L, bt, static_cast<int>(H));
            SK_LAUNCH("mla_unabsorb/gemv");
            return;
        }
        const dim3 grid(static_cast<unsigned>(H), static_cast<unsigned>((N + kTokTile - 1) / kTokTile));
        const size_t smem = sizeof(float) * kTokTile * kv_b.cols;
        if (smem > 48 * 1024)
            SK_CUDA(cudaFuncSetAttribute(mla_unabsorb_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(smem)),
                    "mla_unabsorb smem");
        mla_unabsorb_kernel<<<grid, 128, smem, s_>>>(y, o_lat, static_cast<const uint16_t *>(kv_b.data), N,
                                                     static_cast<int>(H), static_cast<int>(qn), static_cast<int>(vh),
                                                     static_cast<int>(kv_b.cols));
        SK_LAUNCH("mla_unabsorb");
    }
    // MLA over long segments, un-absorbed (Backend::mla_prefill): the keys' latent rows gathered
    // from the cache as bf16, then per pass of hg heads (as many as the pass budget takes)
    // k = W_UK^h c and v = W_UV^h c by two strided-batched cuBLAS GEMMs (the heads the batch, the
    // latent shared), the rope slot copied into every head's key, the pass's q columns as bf16,
    // FlashAttention-3 at 192 / 128, and its output into y's columns of the pass. One launch a
    // pass over every segment's whole context when the keys fit (mla_prefill_keys_) and the pass
    // keeps the SMs busy; else segment by segment in chunks of keys (mla_prefill_chunked), merged
    // by their log-sum-exp: any context length in the same scratch.
    // bf16 operands: with exact (fp32) activations the absorbed fp32 kernels stay, as FlashAttention-3 does for GQA
    int64_t mla_prefill_min() const override { return kt_.fa3_mla && bx_ ? mla_prefill_min_ : 0; }
    size_t mla_prefill_scratch(int64_t n_tokens, int64_t H, int64_t width) const override {
        if (mla_prefill_min() <= 0) return 0;
        return sizeof(uint16_t) * mla_prefill_keys_ * width + static_cast<size_t>(mla_prefill_bytes_) +
               2 * sizeof(float) * H * n_tokens + sizeof(int32_t) * 4096;
    }
    // the most heads (a divisor of H) whose pass fits the budget: keys and values, q and o, bf16
    int64_t mla_pass_heads(int64_t H, int64_t Nk, int64_t Nq, int64_t QK, int64_t vh) const {
        int64_t hg = H;
        while (hg > 1 && (Nk + Nq) * hg * (QK + vh) * 2 > mla_prefill_bytes_) {
            int64_t d = hg - 1;
            while (d > 1 && H % d) --d;
            hg = d;
        }
        return hg;
    }
    // the share of the SMs a launch of t tiles keeps busy over its waves
    double wave_fill(int64_t t) const {
        const int64_t w = std::max<int64_t>(1, ceil_div(t, sm_count_));
        return static_cast<double>(t) / static_cast<double>(w * sm_count_);
    }
    bool mla_prefill(const MlaPrefill &m) override {
        if (mla_prefill_min() <= 0 || m.n <= 0 || m.qn + m.qr != 192 || m.vh != 128 || m.kv_b->fmt != WFmt::BF16 ||
            m.kv_b->rows != m.H * (m.qn + m.vh) || m.total_q > INT_MAX || m.H > 256)
            return false;
        const int64_t H = m.H, QK = m.qn + m.qr, vh = m.vh, Nk = m.total_k, Nq = m.total_q;
        const int64_t hg = mla_pass_heads(H, Nk, Nq, QK, vh);
        bool batched = m.chunk_keys <= 0 && m.chunk_heads <= 0 && Nk <= mla_prefill_keys_ &&
                       (Nk + Nq) * hg * (QK + vh) * 2 <= mla_prefill_bytes_;
        if (batched && hg < H && m.meta_host && m.chunk_keys == 0) {
            // a pass of few heads (long contexts) over few query blocks leaves SMs idle: chunks then,
            // each pass of as many heads as fill the SMs
            int64_t mb = 0;
            for (int64_t i = 0; i < m.n; ++i) mb += ceil_div(m.meta_host[i + 1] - m.meta_host[i], 128);
            batched = wave_fill(hg * mb) >= 0.8;
        }
        if (batched) {
            mla_prefill_batched(m, hg);
            return true;
        }
        return m.meta_host && mla_prefill_chunked(m);
    }
    // k16 [nk][hg][QK] (the rope slot copied in) and v16 [nk][hg][vh] for heads [h0, h0 + hg) from
    // nk latent rows lat [nk][L] and their rope slots rope [nk][qr]
    void mla_expand(const MlaPrefill &m, const uint16_t *lat, const uint16_t *rope, int64_t nk, int64_t h0, int64_t hg,
                    uint16_t *k16, uint16_t *v16) {
        const int64_t L = m.kv_lora, qn = m.qn, qr = m.qr, vh = m.vh, QK = qn + qr;
        const auto *wb = static_cast<const uint16_t *>(m.kv_b->data);
        const float one = 1.0f, zero = 0.0f;
        // k[j][h][0..qn) = sum_l W_UK^h[i][l] lat[j][l]; v[j][h][..] likewise with W_UV^h.
        // Column-major: C (qn x nk, ld hg*QK) = W^T (qn x L) . lat (L x nk), batched over heads.
        SK_BLAS(cublasGemmStridedBatchedEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(qn), static_cast<int>(nk),
                                           static_cast<int>(L), &one, wb + h0 * (qn + vh) * L, CUDA_R_16BF,
                                           static_cast<int>(L), (qn + vh) * L, lat, CUDA_R_16BF, static_cast<int>(L), 0,
                                           &zero, k16, CUDA_R_16BF, static_cast<int>(hg * QK), QK,
                                           static_cast<int>(hg), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                "mla_prefill/k");
        SK_BLAS(cublasGemmStridedBatchedEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(vh), static_cast<int>(nk),
                                           static_cast<int>(L), &one, wb + (h0 * (qn + vh) + qn) * L, CUDA_R_16BF,
                                           static_cast<int>(L), (qn + vh) * L, lat, CUDA_R_16BF, static_cast<int>(L), 0,
                                           &zero, v16, CUDA_R_16BF, static_cast<int>(hg * vh), vh,
                                           static_cast<int>(hg), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                "mla_prefill/v");
        const int64_t nr = nk * hg * qr;
        mla_rope_keys_kernel<<<static_cast<unsigned>(std::min<int64_t>(ceil_div(nr, kThreads), 4 * sm_count_)), kThreads, 0, s_>>>(
            k16, rope, nk, static_cast<int>(hg), static_cast<int>(qn), static_cast<int>(qr));
        SK_LAUNCH("mla_prefill/rope");
    }
    // FlashAttention-3 over n sequences (meta: cu_q, cu_k) of hg heads, rows of q16 / k16 hg * QK
    // wide, of v16 / o16 hg * vh
    void mla_fa3(const MlaPrefill &m, const uint16_t *q16, const uint16_t *k16, const uint16_t *v16, uint16_t *o16,
                 float *lse, const int32_t *meta, int32_t *sched, int64_t n, int64_t total_q, int64_t total_k,
                 int64_t max_q, int64_t max_k, int64_t hg, bool causal) {
        const int64_t QK = m.qn + m.qr, vh = m.vh;
        fa3_ops::Mla a{};
        a.q = q16, a.q_row = hg * QK;
        a.k = k16, a.k_row = hg * QK;
        a.v = v16, a.v_row = hg * vh;
        a.o = o16, a.o_row = hg * vh;
        a.lse = lse;
        a.cu_seqlens_q = meta, a.cu_seqlens_k = meta + n + 1;
        a.scheduler = sched;
        a.n = static_cast<int>(n), a.total_q = static_cast<int>(total_q), a.total_k = static_cast<int>(total_k);
        a.max_q = static_cast<int>(max_q), a.max_k = static_cast<int>(max_k), a.H = static_cast<int>(hg);
        a.scale = m.scale, a.sm_count = sm_count_, a.causal = causal;
        fa3_ops::mla(a, s_);
        SK_LAUNCH("mla_prefill/fa3");
    }
    template <typename F>
    void mla_by_kv_dtype(const MlaPrefill &m, F &&f) {
        if (m.kv_dtype == DType::F32) f(static_cast<const float *>(m.kv));
        else f(static_cast<const __nv_bfloat16 *>(m.kv));
    }
    // every segment's whole context in one launch a pass (the keys back to back, cu_k)
    void mla_prefill_batched(const MlaPrefill &m, int64_t hg) {
        const int64_t H = m.H, L = m.kv_lora, qr = m.qr, vh = m.vh, QK = m.qn + qr, Nk = m.total_k, Nq = m.total_q;
        materialize(m.q);
        twin_clear(m.y);
        bind();
        const int64_t lat_elems = (Nk * (L + qr) + 127) / 128 * 128;  // the pass buffers 256-byte aligned after it
        auto *lat = static_cast<uint16_t *>(ensure(prompt_, sizeof(uint16_t) * (lat_elems + (Nk + Nq) * hg * (QK + vh))));
        uint16_t *rope = lat + Nk * L;
        uint16_t *k16 = lat + lat_elems;
        uint16_t *v16 = k16 + Nk * hg * QK, *q16 = v16 + Nk * hg * vh, *o16 = q16 + Nq * hg * QK;
        auto *lse = static_cast<float *>(ensure(mla_lse_, sizeof(float) * hg * Nq));
        auto *sched = static_cast<int32_t *>(ensure(mla_sched_, sizeof(int32_t) * fa3_ops::scheduler_ints(static_cast<int>(m.n))));
        const int n = static_cast<int>(m.n);
        mla_by_kv_dtype(m, [&](auto *kv) {
            mla_latent_rows_kernel<<<static_cast<unsigned>(Nk), 128, 0, s_>>>(lat, rope, kv, m.meta, n, m.page_table,
                                                                             m.max_pages, m.page, static_cast<int>(L),
                                                                             static_cast<int>(qr));
        });
        SK_LAUNCH("mla_prefill/latent");
        for (int64_t h0 = 0; h0 < H; h0 += hg) {
            mla_expand(m, lat, rope, Nk, h0, hg, k16, v16);
            mla_q_rows_kernel<<<static_cast<unsigned>(Nq), 256, 0, s_>>>(q16, m.q, m.meta, n, H * QK, h0 * QK, hg * QK);
            SK_LAUNCH("mla_prefill/q");
            mla_fa3(m, q16, k16, v16, o16, lse, m.meta, sched, n, Nq, Nk, m.max_q, m.max_k, hg, true);
            mla_o_rows_kernel<<<static_cast<unsigned>(Nq), 256, 0, s_>>>(m.y, o16, m.meta, n, H * vh, h0 * vh, hg * vh);
            SK_LAUNCH("mla_prefill/o");
        }
    }
    // A segment's chunking: passes of hg heads, chunks of at most kb keys (at least its queries).
    // hg: the most heads (a divisor of H) whose pass leaves room for chunks of max(nq, 4096) keys
    // in the budget after its q and o; kb: the keys it leaves room for. Many heads a pass keep
    // the SMs busy with few queries (their query blocks times hg tiles) and balance a causal
    // launch's tiles; on an H100 against fewer heads in longer chunks: 512 queries over 100K keys
    // 4.8 ms (32 heads) vs 9.1 (8) and 25.1 (2); 8192 over 104K 28.4 ms (8) vs 31.7 (2).
    struct MlaChunks {
        int64_t hg, kb;
    };
    MlaChunks mla_chunks(const MlaPrefill &m, int64_t nq) const {
        const int64_t H = m.H, head = (m.qn + m.qr + m.vh) * 2;  // a key's k and v (a query's q and o) bytes a head
        auto room = [&](int64_t hg) { return std::min(mla_prefill_keys_, mla_prefill_bytes_ / (hg * head) - nq); };
        int64_t hg = H;
        if (m.chunk_heads > 0) {
            hg = m.chunk_heads;
        } else {
            while (hg > 1 && room(hg) < std::max<int64_t>(nq, 4096)) {
                int64_t d = hg - 1;
                while (d > 1 && H % d) --d;
                hg = d;
            }
        }
        return {hg, m.chunk_keys > 0 ? std::max(m.chunk_keys, nq) : room(hg)};
    }
    // Segment by segment: its last kb keys (its own and the context just before them) in one
    // causal launch, the context before those in nearly equal chunks of at most kb keys with no
    // mask (every query sees every key), each chunk's latent rows gathered and its keys and values
    // expanded for the pass, and its output merged into y by the log-sum-exp (mla_lse_merge_kernel:
    // y holds the merge so far, normalised; acc its log-sum-exp). The scratch is a pass's chunk,
    // whatever the context.
    bool mla_prefill_chunked(const MlaPrefill &m) {
        const int64_t H = m.H, L = m.kv_lora, qr = m.qr, vh = m.vh, QK = m.qn + qr, n = m.n;
        const int32_t *cu_q = m.meta_host, *cu_k = cu_q + n + 1, *slot = cu_k + n + 1, *first = slot + n;
        // every segment's chunking first: one no pass fits leaves the whole call to the absorbed kernels
        chunks_.resize(static_cast<size_t>(n));
        int64_t lat_max = 0, buf_max = 0, lse_max = 0;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t nq = cu_q[i + 1] - cu_q[i], keys = cu_k[i + 1] - cu_k[i];
            const MlaChunks c = chunks_[static_cast<size_t>(i)] = mla_chunks(m, nq);
            if (nq <= 0 || keys < nq || c.hg < 1 || H % c.hg || c.kb < nq) return false;  // kb >= nq: the pass fits
            const int64_t kb = std::min(c.kb, keys);
            lat_max = std::max(lat_max, kb), lse_max = std::max(lse_max, c.hg * nq);
            buf_max = std::max(buf_max, (kb + nq) * c.hg * (QK + vh));
        }
        materialize(m.q);
        twin_clear(m.y);
        bind();
        const int64_t lat_elems = (lat_max * (L + qr) + 127) / 128 * 128;
        auto *lat = static_cast<uint16_t *>(ensure(prompt_, sizeof(uint16_t) * (lat_elems + buf_max)));
        uint16_t *rope = lat + lat_max * L, *k16 = lat + lat_elems;
        auto *lse = static_cast<float *>(ensure(mla_lse_, sizeof(float) * lse_max));
        auto *acc = static_cast<float *>(ensure(mla_acc_, sizeof(float) * lse_max));
        const int si = fa3_ops::scheduler_ints(1);
        auto *sched = static_cast<int32_t *>(ensure(mla_sched_, sizeof(int32_t) * (si + 8)));
        int32_t *cmeta = sched + si;  // the chunk's meta (mla_latent_span_kernel writes it)
        for (int64_t i = 0; i < n; ++i) {
            const int64_t hg = chunks_[static_cast<size_t>(i)].hg, kb = chunks_[static_cast<size_t>(i)].kb;
            const int64_t nq = cu_q[i + 1] - cu_q[i], keys = cu_k[i + 1] - cu_k[i];
            const int64_t kl = std::min(kb, keys), c_last = keys - kl;  // the causal chunk: keys [c_last, keys)
            const int64_t n_nc = ceil_div(c_last, kb), kc = n_nc > 0 ? ceil_div(c_last, n_nc) : 0;
            uint16_t *v16 = k16 + kl * hg * QK, *q16 = v16 + kl * hg * vh, *o16 = q16 + nq * hg * QK;
            for (int64_t h0 = 0; h0 < H; h0 += hg) {
                for (int64_t c = 0; c <= n_nc; ++c) {
                    const bool causal = c == n_nc;
                    const int64_t pos0 = causal ? c_last : c * kc, nk = causal ? kl : std::min(kc, c_last - pos0);
                    if (nk <= 0) continue;
                    mla_by_kv_dtype(m, [&](auto *kv) {
                        mla_latent_span_kernel<<<static_cast<unsigned>(nk), 128, 0, s_>>>(
                            lat, rope, kv, cmeta, slot[i], pos0, static_cast<int>(nq), static_cast<int>(nk), first[i],
                            m.page_table, m.max_pages, m.page, static_cast<int>(L), static_cast<int>(qr));
                    });
                    SK_LAUNCH("mla_prefill/latent");
                    mla_expand(m, lat, rope, nk, h0, hg, k16, v16);
                    if (c == 0) {
                        mla_q_rows_kernel<<<static_cast<unsigned>(nq), 256, 0, s_>>>(q16, m.q, cmeta, 1, H * QK, h0 * QK,
                                                                                    hg * QK);
                        SK_LAUNCH("mla_prefill/q");
                    }
                    mla_fa3(m, q16, k16, v16, o16, lse, cmeta, sched, 1, nq, nk, nq, nk, hg, causal);
                    mla_lse_merge_kernel<<<static_cast<unsigned>(nq), 256, 0, s_>>>(
                        m.y, o16, lse, acc, cmeta, H * vh, h0 * vh, static_cast<int>(hg), static_cast<int>(vh),
                        static_cast<int>(nq), c == 0 ? 1 : 0);
                    SK_LAUNCH("mla_prefill/merge");
                }
            }
        }
        return true;
    }
    void index_rows(float *dst, const float *src, const int32_t *rows, int64_t count, int64_t n, bool scatter) override {
        materialize(src);
        if (scatter) materialize(dst);  // the rows not written keep their values
        else twin_clear(dst);
        if (count <= 0 || n <= 0) return;
        bind();
        index_rows_kernel<<<static_cast<unsigned>(count), 256, 0, s_>>>(dst, src, rows, n, scatter ? 1 : 0);
        SK_LAUNCH("index_rows");
    }

    // ---------------------------------------------------------------- dense attention
    void rope(float *x, const int32_t *tok_pos, const float *inv_freq, int64_t N, int64_t heads, int64_t D,
              int64_t stride) override {
        materialize(inv_freq);
        if (N <= 0) return;
        bind();
        rope_kernel<<<static_cast<unsigned>(N), kThreads, 0, s_>>>(x, tok_pos, inv_freq, heads, D, stride);
        SK_LAUNCH("rope");
    }
    void gqa_attention(float *o, const float *q, const void *kc, const void *vc, DType kv_dtype,
                       const int32_t *tok_slot, const int32_t *tok_pos, const int32_t *page_table, int64_t max_pages,
                       int64_t page, int64_t N, int64_t Hq, int64_t Hkv, int64_t D, int64_t max_pos,
                       float scale) override {
        if (N <= 0) return;
        twin_clear(o);  // o is rewritten here
        materialize(q);  // a mixed step's q may be bf16-only from RoPE
        const int64_t G = Hq / Hkv;
        if (Hq % Hkv || G > 32) cuda_fail("gqa_attention", "need Hq a multiple of Hkv, at most 32 per group");
        if (D % 32 || D / 32 > 8) cuda_fail("gqa_attention", "head_dim must be 32, 64, 128, 192 or 256");
        bind();
        // the same split policy as mla_attention (see there); the tensor-core kernel takes
        // fewer, longer splits (each of its warps streams whole tiles)
        static const bool no_tc = tc_env("FULLM_NO_ATTN_TC", 0) != 0;
        if (kv_dtype == DType::FP8E4M3) cuda_fail("gqa_attention", "plain e4m3 cache: FlashAttention-3 reads it (gqa_prefill)");
        const bool f8 = kv_dtype == DType::FP8;
        if (f8 && (!bx_ || (D != 64 && D != 128) || G > 16 || page % 16))
            cuda_fail("gqa_attention", "fp8 cache: bf16 activations, head_dim 64 or 128, <= 16 heads per group, 16-multiple pages");
        const bool tc = f8 || (!no_tc && bx_ && kv_dtype == DType::BF16 && (D == 64 || D == 128) && G <= 16);
        int max_splits = f8 ? kAtc8MaxSplits : tc ? kAtcMaxSplits : kAttnMaxSplits;
        if (N > kGemvMaxRows) {
            const int64_t per_split = N * Hq * (D + 2) * static_cast<int64_t>(sizeof(float));
            max_splits = static_cast<int>(std::clamp<int64_t>((int64_t{256} << 20) / per_split, 1, int64_t{max_splits}));
        }
        const int zsplits = static_cast<int>(std::min<int64_t>(max_splits, (max_pos + kAttnSplitLen) / kAttnSplitLen));
        float *part = nullptr;
        if (zsplits > 1) part = static_cast<float *>(ensure(part_, sizeof(float) * N * Hq * zsplits * (D + 2)));
        // the split blocks merge their own partials when every (token, kv head) has a counter
        unsigned *cnt = zsplits > 1 && N * Hkv <= kAttnCounters ? attn_count_ : nullptr;
        const dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(Hkv), static_cast<unsigned>(zsplits));
        const int hq = static_cast<int>(Hq), hk = static_cast<int>(Hkv);
        if (tc) {
            const float sl2 = scale * 1.4426950408889634f;
            // blocks walk several splits each: about attn_blocks_ blocks, not N * Hkv * zsplits (a
            // graph sizes zsplits for the longest context: 2048 mostly empty blocks at 16 tokens)
            const int zg = static_cast<int>(std::clamp<int64_t>(ceil_div(attn_blocks_, N * Hkv), 1, zsplits));
            const dim3 gt(static_cast<unsigned>(N), static_cast<unsigned>(Hkv), static_cast<unsigned>(zg));
#define SK_ATC(DR, F8, ST)                                                                                       \
    launch_pdl(gqa_attn_tc_kernel<DR, F8, ST>, gt, dim3(kAtcWarps * 32), atc_smem<DR, F8, ST>(), s_, o, part, cnt, q, \
               kc, vc, tok_slot, tok_pos, page_table, max_pages, page, hq, hk, sl2, max_splits, zsplits, atc_split_len_)
            if (D == 64) {
                if (f8) SK_ATC(2, true, kAtcStages);
                else SK_ATC(2, false, kAtcStages);
            } else {
                if (f8) SK_ATC(4, true, kAtcStages);
                else if (atc_stages_ == kAtcStages) SK_ATC(4, false, kAtcStages);
                else SK_ATC(4, false, 2);
            }
#undef SK_ATC
            SK_LAUNCH("gqa_attention/tc");
            if (zsplits > 1 && !cnt) {
                launch_pdl(mla_combine_kernel, dim3(static_cast<unsigned>(N * Hq)), dim3(128), 0, s_, o, (const float *)part,
                           tok_pos, hq, static_cast<int>(D), max_splits, zsplits, atc_split_len_);
                SK_LAUNCH("gqa_attention/combine");
            }
            return;
        }
        const dim3 block(static_cast<unsigned>(32 * G));
#define SK_GQA(KT, DR)                                                                                            \
    launch_pdl(gqa_attn_kernel<KT, DR>, grid, block, 0, s_, o, part, cnt, q, static_cast<const KT *>(kc),        \
               static_cast<const KT *>(vc), tok_slot, tok_pos, page_table, max_pages, page, hq, hk, scale, max_splits)
#define SK_GQA_D(KT)                    \
    switch (D / 32) {                   \
        case 1: SK_GQA(KT, 1); break;   \
        case 2: SK_GQA(KT, 2); break;   \
        case 3: SK_GQA(KT, 3); break;   \
        case 4: SK_GQA(KT, 4); break;   \
        case 6: SK_GQA(KT, 6); break;   \
        case 8: SK_GQA(KT, 8); break;   \
        default: cuda_fail("gqa_attention", "head_dim"); \
    }
        if (kv_dtype == DType::F32) {
            SK_GQA_D(float)
        } else {
            SK_GQA_D(__nv_bfloat16)
        }
#undef SK_GQA_D
#undef SK_GQA
        SK_LAUNCH("gqa_attention");
        if (zsplits > 1 && !cnt) {
            launch_pdl(mla_combine_kernel, dim3(static_cast<unsigned>(N * Hq)), dim3(128), 0, s_, o, (const float *)part,
                       tok_pos, hq, static_cast<int>(D), max_splits, zsplits, kAttnSplitLen);
            SK_LAUNCH("gqa_attention/combine");
        }
    }
    int gqa_prefill(float *o, const float *q, const void *kc, const void *vc, DType kv_dtype, const int32_t *tiles,
                    int64_t n_tiles, const int32_t *h_segs, int64_t n_segs, int64_t n_dec, bool rows, const int32_t *tok_slot,
                    const int32_t *tok_pos, const int32_t *page_table, int64_t max_pages, int64_t page, int64_t Hq,
                    int64_t Hkv, int64_t D, int64_t max_pos, float scale) override {
        static const bool off = tc_env("FULLM_NO_PREFILL_ATTN", 0) != 0;
        twin_clear(o);  // o is about to be rewritten (the FMHA paths register their own copy)
        const int64_t G = Hq / Hkv;
        if (off || !bx_ || Hq % Hkv || G > 8 || (D != 64 && D != 128)) return 0;  // exact: the fp32 kernel
        bind();
        static const bool no_fmha8 = tc_env("FULLM_NO_FMHA", 0) != 0;
        if (kv_dtype == DType::FP8) {
            // fp8 cache: the chunks' contexts dequantised to bf16 pages (exact: power-of-two
            // scales; the decode kernel sees the same values), the FMHA over them through a staging
            // page table. The decoding rows stay with the fp8 decode kernel (the caller runs it).
            if (no_fmha8 || n_segs <= 0 || !h_segs || page % 16) return 0;  // every row: the decode kernel
            const int64_t ns = kv8_stage(h_segs, n_segs, max_pages, page);
            const size_t pb = sizeof(__nv_bfloat16) * page * Hkv * D;
            auto *kst = static_cast<__nv_bfloat16 *>(ensure(kv8_k_, pb * ns));
            auto *vst = static_cast<__nv_bfloat16 *>(ensure(kv8_v_, pb * ns));
            launch_pdl(kv8_dequant_kernel, dim3(static_cast<unsigned>(ns), 2), dim3(128), 0, s_, kst, vst,
                       static_cast<const uint8_t *>(kc), static_cast<const uint8_t *>(vc), (const int32_t *)kv8_src_.p,
                       page_table, page, Hkv, D);
            SK_LAUNCH("gqa_prefill/fp8 stage");
            const int64_t pool_pages = kv_pages_;
            kv_pages_ = ns;  // the staging pool is the FMHA's page pool
            const int done = fmha_prefill(o, q, kst, vst, h_segs, n_segs, n_dec, false, tok_slot, tok_pos,
                                          (const int32_t *)kv8_pt_.p, max_pages, page, Hq, Hkv, D, max_pos, scale);
            kv_pages_ = pool_pages;
            return done;
        }
        // FlashAttention-3 (the chunks, and the decoding rows when the caller says so) or the
        // CUTLASS FMHA (the chunks) when built in, over a bf16 cache
        static const bool no_fmha = tc_env("FULLM_NO_FMHA", 0) != 0;
        if (kv_dtype == DType::FP8E4M3) {
            // only FlashAttention-3 reads plain e4m3 pages: every chunk and every decoding row here
            if (!kt_.fa3_fp8_at(D))
                cuda_fail("gqa_prefill", "the fp8 cache needs FlashAttention-3 (the cutlass feature) and head_dim 128");
            if (n_segs <= 0 && n_dec <= 0) return 2;
            if (!fa3_prefill(o, q, kc, vc, true, h_segs, n_segs, n_dec, tok_slot, tok_pos, page_table, max_pages, page, Hq,
                             Hkv, D, max_pos, scale))
                cuda_fail("gqa_prefill", "FlashAttention-3 declined an fp8 step");
            return 2;
        }
        if (!no_fmha && kv_dtype == DType::BF16 && (n_segs > 0 || (rows && n_dec > 0))) {
            const int done =
                fmha_prefill(o, q, static_cast<const __nv_bfloat16 *>(kc), static_cast<const __nv_bfloat16 *>(vc), h_segs,
                             n_segs, n_dec, rows, tok_slot, tok_pos, page_table, max_pages, page, Hq, Hkv, D, max_pos,
                             scale);
            if (done) return done;
        }
        if (n_tiles <= 0) return 0;  // no chunks: the decode kernel takes the rows
        materialize(q);  // these kernels read q as fp32
        const dim3 grid(static_cast<unsigned>(n_tiles), static_cast<unsigned>(Hkv)), block(static_cast<unsigned>(32 * G));
        const float sl2 = scale * 1.4426950408889634f;  // softmax in base 2
        const int hq = static_cast<int>(Hq), hk = static_cast<int>(Hkv);
#define SK_PF(KT, DD)                                                                                                 \
    launch_pdl(gqa_prefill_f32_kernel<KT, DD>, grid, block, 0, s_, o, q, static_cast<const KT *>(kc),               \
               static_cast<const KT *>(vc), tiles, tok_slot, tok_pos, page_table, max_pages, page, hq, hk, sl2)
#define SK_PF16(DD)                                                                                                   \
    launch_pdl(gqa_prefill_kernel<DD>, grid, block, pf_smem<DD>(), s_, o, q, static_cast<const __nv_bfloat16 *>(kc), \
               static_cast<const __nv_bfloat16 *>(vc), tiles, tok_slot, tok_pos, page_table, max_pages, page, hq, hk, \
               sl2)
        if (kv_dtype == DType::F32) {
            if (D == 64) SK_PF(float, 64);
            else SK_PF(float, 128);
        } else {
            if (D == 64) SK_PF16(64);
            else SK_PF16(128);
        }
#undef SK_PF16
#undef SK_PF
        SK_LAUNCH("gqa_prefill");
        return n_dec > 0 ? 1 : 2;
    }
    bool qk_rope_append(float *q, const float *k, const float *v, void *kc, void *vc, DType kv_dtype,
                        const float *q_norm, const float *k_norm, float eps, const float *inv_freq,
                        const int32_t *tok_slot, const int32_t *tok_pos, const int32_t *page_table, int64_t max_pages,
                        int64_t page, int64_t N, int64_t Hq, int64_t Hkv, int64_t D) override {
        if (N <= 0) return true;
        if (D % 32 || (D / 32 != 1 && D / 32 != 2 && D / 32 != 4 && D / 32 != 8)) return false;
        bind();
        // the q/k/v GEMMs' bf16 results when they left only those (gemm_group, y16_ok): three twins,
        // or one fused matrix read in place (rows ld apart); else fp32
        const uint16_t *qi = twin_get(q, N, Hq * D), *ki = twin_get(k, N, Hkv * D), *vi = twin_get(v, N, Hkv * D);
        int64_t ldq = Hq * D, ldkv = Hkv * D;
        if (qkv_.b && qkv_.q == q && qkv_.k == k && qkv_.v == v && qkv_.M == N && qkv_.nq == Hq * D &&
            qkv_.nk == Hkv * D && qkv_.nv == Hkv * D) {
            qi = qkv_.b, ki = qkv_.b + qkv_.nq, vi = qkv_.b + qkv_.nq + qkv_.nk, ldq = ldkv = qkv_.ld;
            qkv_ = QkvFused{};  // consumed: q is rewritten below, k and v are not read again
        }
        const bool in16 = qi && ki && vi;
        if (!in16) materialize(q), materialize(k), materialize(v);
        twin_clear(q);
        // the FMHA's Q (its pad rows zeroed below); q's fp32 rows then stay unwritten (lazy)
        uint16_t *q16 = twin_out_lazy(twin_q_, q, N, Hq * D);
        float *qo = q16 ? nullptr : q;
        if (q16) SK_CUDA(cudaMemsetAsync(q16 + N * Hq * D, 0, sizeof(uint16_t) * 8 * Hq * D, s_), "q twin pad");
        // decoding rows over FlashAttention-3's fp8 cache: q also as e4m3, FA3's operand, rounded
        // from the same fp32 values its conversion kernel would read (the same bits, a launch fewer)
        q8_ = Q8{};
        uint8_t *q8 = nullptr;
        if (!q16 && kv_dtype == DType::FP8E4M3 && kt_.fa3_fp8_at(D)) {
            q8 = static_cast<uint8_t *>(ensure(fa3_q8_, N * Hq * D));
            q8_ = Q8{q, N, q8};
        }
        const dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(ceil_div(Hq + Hkv, 8)));
        const int hq = static_cast<int>(Hq), hk = static_cast<int>(Hkv);
#define SK_QKR(DR, KT)                                                                                                     \
    if (in16)                                                                                                              \
        launch_pdl(qk_rope_append_kernel<DR, KT, uint16_t>, grid, dim3(256), 0, s_, qo, qi, ki, vi, static_cast<KT *>(kc), \
                   static_cast<KT *>(vc), q_norm, k_norm, eps, inv_freq, tok_slot, tok_pos, page_table, max_pages, page, hq, \
                   hk, q16, ldq, ldkv, q8);                                                                                \
    else                                                                                                                   \
        launch_pdl(qk_rope_append_kernel<DR, KT, float>, grid, dim3(256), 0, s_, qo, (const float *)q, k, v,               \
                   static_cast<KT *>(kc), static_cast<KT *>(vc), q_norm, k_norm, eps, inv_freq, tok_slot, tok_pos,          \
                   page_table, max_pages, page, hq, hk, q16, int64_t{hq} * D, int64_t{hk} * D, q8)
#define SK_QKR_D(KT)                \
    switch (D / 32) {               \
        case 1: SK_QKR(1, KT); break; \
        case 2: SK_QKR(2, KT); break; \
        case 4: SK_QKR(4, KT); break; \
        default: SK_QKR(8, KT); break; \
    }
        if (kv_dtype == DType::F32) {
            SK_QKR_D(float)
        } else if (kv_dtype == DType::FP8) {
            if (page % 16) cuda_fail("qk_rope_append", "fp8 cache: pages of a multiple of 16 positions");
            SK_QKR_D(uint8_t)
        } else if (kv_dtype == DType::FP8E4M3) {
            SK_QKR_D(__nv_fp8_e4m3)
        } else {
            SK_QKR_D(__nv_bfloat16)
        }
#undef SK_QKR_D
#undef SK_QKR
        SK_LAUNCH("qk_rope_append");
        return true;
    }
    bool router_softmax_fused(int32_t *idx, float *w, const Weight &W, const float *x, int64_t N, int64_t k,
                              bool renorm, const ExpertW *table, int64_t n_experts) override {
        route_ready_ = RouteReady{};
        if (N > kGemvMaxRows || W.fmt != WFmt::BF16 || W.rows > 1024 || W.cols % 8 || k > W.rows || k > 16) return false;
        if (N <= 0) return true;
        bind();
        if (!gemv_tc_ok(W, x, W.cols)) return false;
        auto *logits = static_cast<float *>(ensure(router_, sizeof(float) * N * W.rows));
        const int S = tc_split(W.rows, W.cols, 2);  // as dispatch_gemv: the same logits as gemm
        const dim3 grid(static_cast<unsigned>(ceil_div(W.rows, 8 * (kTcWarps / S)))), blk(kTcWarps * 32);
        const int n = static_cast<int>(N), e = static_cast<int>(W.rows), kk = static_cast<int>(k), rn = renorm ? 1 : 0;
        // the expert call's routing lists too, when its table is given (FULLM_ROUTE_FUSED=0: the
        // route kernel builds them)
        static const bool fuse_route = env_on("FULLM_ROUTE_FUSED", true);
        MoeLists L{};
        if (fuse_route && table && n_experts == W.rows) L = moe_lists(N * k, n_experts);
        // and x as fp16 for e4m3 experts' tiles
        __half *xh = nullptr;
        if (table && bx_ && (expert_fmt_ == WFmt::FP8 || expert_fmt_ == WFmt::FP8B || expert_fmt_ == WFmt::INT4G) &&
            W.cols % 4 == 0)
            xh = static_cast<__half *>(ensure(xh16_, sizeof(__half) * N * W.cols));
        // x's bf16 twin (the add + norm writes one for decode rows): the GEMV reads it, the fp16 copy
        // for the experts still comes from the fp32 rows
        const uint16_t *xb = bx_ ? twin_get(x, N, W.cols) : nullptr;
#define SK_ROUTER(TWO, BX, XB)                                                                                         \
    launch_pdl(router_fused_kernel<TWO, BX, XB>, grid, blk, kTcSmem, s_, logits, router_count_, idx, w, W.data, x, n, e, \
               W.cols, kk, rn, S, L, table, xh, router_regs_ ? 1 : 0, xb)
        if (xb) {
            if (N > 8) SK_ROUTER(true, true, true);
            else SK_ROUTER(false, true, true);
        } else if (bx_) {
            if (N > 8) SK_ROUTER(true, true, false);
            else SK_ROUTER(false, true, false);
        } else {
            if (N > 8) SK_ROUTER(true, false, false);
            else SK_ROUTER(false, false, false);
        }
#undef SK_ROUTER
        SK_LAUNCH("router_softmax_fused");
        route_ready_ = RouteReady{L.meta ? idx : nullptr, table, N * k, n_experts, x, xh, N, W.cols};
        return true;
    }
    // the routing lists' layout for `pairs` pairs over n_experts, in moe_route_: the router's
    // fused build and moe_experts take it from here, so the two agree
    MoeLists moe_lists(int64_t pairs, int64_t n_experts) {
        const int64_t slots = std::min<int64_t>(pairs, n_experts);
        MoeLists L{nullptr, static_cast<int>(slots), static_cast<int>(MoeLists::maxp_for(slots, pairs)), static_cast<int>(pairs)};
        L.meta = static_cast<int32_t *>(ensure(moe_route_, sizeof(int32_t) * MoeLists::ints(slots, pairs)));
        return L;
    }
    void router_softmax_topk(int32_t *idx, float *w, const float *logits, int64_t N, int64_t E, int64_t k,
                             bool renorm) override {
        materialize(logits);
        if (N <= 0) return;
        if (E > 4096 || k < 1 || k > E) cuda_fail("router_softmax_topk", "need 1 <= k <= E <= 4096");
        bind();
        const int e = static_cast<int>(E), kk = static_cast<int>(k), rn = renorm ? 1 : 0;
        const dim3 wgrid(static_cast<unsigned>(ceil_div(N, int64_t{8}))), wblock(256);
        if (E <= 128) router_softmax_warp_kernel<4><<<wgrid, wblock, 0, s_>>>(idx, w, logits, N, e, kk, rn);
        else if (E <= 256) router_softmax_warp_kernel<8><<<wgrid, wblock, 0, s_>>>(idx, w, logits, N, e, kk, rn);
        else if (E <= 512) router_softmax_warp_kernel<16><<<wgrid, wblock, 0, s_>>>(idx, w, logits, N, e, kk, rn);
        else
            router_softmax_kernel<<<static_cast<unsigned>(N), kThreads, sizeof(float) * 2 * E, s_>>>(idx, w, logits, e, kk, rn);
        SK_LAUNCH("router_softmax_topk");
    }

    // ---------------------------------------------------------------- sampling
    void embed_rows(float *dst, const Weight &table, const int32_t *ids, int64_t N, int64_t n) override {
        twin_clear(dst);
        if (N <= 0) return;
        if (n % 4 || reinterpret_cast<uintptr_t>(table.data) % 16) cuda_fail("embed_rows", "row width must be a multiple of 4");
        bind();
        const dim3 g(static_cast<unsigned>(N)), b(256);
        if (table.fmt == WFmt::BF16) launch_pdl(embed_rows_kernel<1>, g, b, 0, s_, dst, table.data, ids, n);
        else if (table.fmt == WFmt::F32) launch_pdl(embed_rows_kernel<0>, g, b, 0, s_, dst, table.data, ids, n);
        else cuda_fail("embed_rows", "table must be bf16 or f32");
        SK_LAUNCH("embed_rows");
    }
    void gather_ids(int32_t *dst, const int32_t *table, const int32_t *idx, int64_t n) override {
        if (n <= 0) return;
        bind();
        gather_ids_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s_>>>(dst, table, idx, n);
        SK_LAUNCH("gather_ids");
    }
    void argmax_rows(int32_t *idx, int32_t *bad, const float *x, int64_t rows, int64_t n) override {
        materialize(x);
        if (rows <= 0) return;
        bind();
        void *part = ensure(argmax_part_, size_t(rows) * kArgmaxSlices * (sizeof(float) + sizeof(int64_t)));
        float *pv = static_cast<float *>(part);
        int64_t *pi = reinterpret_cast<int64_t *>(pv + rows * kArgmaxSlices);
        argmax_slices_kernel<<<dim3(static_cast<unsigned>(rows), kArgmaxSlices), kThreads, 0, s_>>>(pv, pi, bad, x, n);
        argmax_rows_kernel<<<static_cast<unsigned>(rows), kArgmaxSlices, 0, s_>>>(idx, pv, pi);
        SK_LAUNCH("argmax_rows");
    }

    void mask_rows(float *x, int32_t *bad, int64_t rows, int64_t n, const SampleRow *p, const uint32_t *masks,
                   int64_t words) override {
        materialize(x);
        if (rows <= 0) return;
        bind();
        mask_rows_kernel<<<dim3(static_cast<unsigned>(rows), kArgmaxSlices), kThreads, 0, s_>>>(x, bad, n, p, masks, words);
        SK_LAUNCH("mask_rows");
    }

    // ---- LoRA (kernels/lora.cuh) ----
    // Where y's value lives for an update in place: the fused q/k/v (gate/up) bf16 matrix, a lazy
    // bf16 twin (no fp32 rows yet), or y itself (a twin that copies it is dropped: it would go stale).
    struct YLoc {
        float *f = nullptr;
        uint16_t *b = nullptr;
        int64_t ld = 0;
    };
    YLoc y_loc(float *y, int64_t N, int64_t out) {
        if (qkv_has(y)) {
            if (qkv_.M == N) {
                const int64_t col = y == qkv_.q ? 0 : y == qkv_.k ? qkv_.nq : qkv_.nq + qkv_.nk;
                return {nullptr, const_cast<uint16_t *>(qkv_.b) + col, qkv_.ld};
            }
            qkv_materialize();
        }
        for (auto &t : twins_)
            if (t.x == y && t.lazy) {
                if (t.M == N && t.K == out) return {nullptr, const_cast<uint16_t *>(t.b), out};
                materialize(t);
            }
        twin_clear(y);
        return {y, nullptr, out};
    }
    // x's bf16 twin (its N rows), else null with x's fp32 rows made current
    const uint16_t *x_twin(const float *x, int64_t N, int64_t in) {
        for (const auto &t : twins_)
            if (t.x == x && t.M == N && t.K == in) return t.b;
        materialize(x);
        return nullptr;
    }

    void lora(const LoraJob *jobs, int n, const float *x, int64_t N, int64_t in, int64_t rank, const int32_t *slot,
              float *xa) override {
        if (N <= 0 || n <= 0 || rank <= 0) return;
        if (n > kLoraJobs) cuda_fail("lora", "more jobs than kLoraJobs");
        bind();
        const uint16_t *xb = x_twin(x, N, in);
        bool vec = in % 256 == 0 && rank % 8 == 0 && (xb ? aligned16(xb) : aligned16(x));
        for (int t = 0; t < n; ++t)
            vec = vec && aligned16(jobs[t].a) && jobs[t].a_stride % 8 == 0 && aligned16(jobs[t].b) && jobs[t].ldb % 8 == 0;
        if (!vec) {  // odd shapes: a job at a time on fp32 rows
            if (xb) materialize(x);
            for (int t = 0; t < n; ++t) {
                const LoraJob &J = jobs[t];
                if (qkv_has(J.y)) qkv_materialize();
                materialize(J.y);  // y's fp32 rows, any bf16 copy dropped
                twin_clear(J.y);
                lora_shrink_kernel<<<static_cast<unsigned>(N), 256, 0, s_>>>(xa, rank, x, in, J.a, J.a_stride, J.ranks,
                                                                           static_cast<int>(rank), slot);
                lora_expand_kernel<<<dim3(static_cast<unsigned>(ceil_div(J.out, int64_t(256))), static_cast<unsigned>(N)), 256, 0, s_>>>(
                    J.y, xa, rank, J.out, J.b, J.ldb, static_cast<int>(rank), J.ranks, J.scale, slot);
                SK_LAUNCH("lora (any shape)");
            }
            return;
        }
        // a decode step of a few rows: the stacked A rows split over KS blocks (each re-reads x:
        // KS times the row groups at most 32, and at least a row a warp), then in over P parts
        // (at most 8: the expand sums them), until about two waves
        const int64_t groups = ceil_div(N, int64_t(kLoraRows)), chunks = in / 256, Kt = n * rank,
                      waves = 2 * int64_t(sm_count_);
        int64_t KS = std::clamp<int64_t>(ceil_div(int64_t(32), groups), 1, ceil_div(Kt, int64_t(8)));
        const int64_t Kc = ceil_div(Kt, KS);
        KS = ceil_div(Kt, Kc);
        int64_t P = std::clamp<int64_t>(ceil_div(waves, groups * KS), 1, std::min<int64_t>(chunks, 8));
        const int64_t C = ceil_div(chunks, P);
        P = ceil_div(chunks, C);
        auto *part = static_cast<float *>(ensure(lora_part_, sizeof(float) * P * N * Kt));
        LoraShrinkJobs J{};
        LoraExpandJobs E{};
        J.n = E.n = n;
        int blocks = 0;
        for (int t = 0; t < n; ++t) {
            J.a[t] = jobs[t].a, J.a_stride[t] = jobs[t].a_stride, J.ranks[t] = jobs[t].ranks;
            const YLoc L = y_loc(jobs[t].y, N, jobs[t].out);
            E.t[t] = LoraOut{L.f, L.b, L.ld, jobs[t].out, jobs[t].b, jobs[t].ldb, jobs[t].scale, jobs[t].ranks};
            E.blk0[t] = blocks;
            blocks += static_cast<int>(ceil_div(jobs[t].out, int64_t(256)));
        }
        E.blk0[n] = blocks;
        const dim3 sg(static_cast<unsigned>(P), static_cast<unsigned>(groups), static_cast<unsigned>(KS));
        const int c = static_cast<int>(C), kc = static_cast<int>(Kc), rk = static_cast<int>(rank);
        if (xb) lora_shrink_parts<uint16_t><<<sg, 256, 0, s_>>>(part, xb, N, in, rk, J, slot, c, kc);
        else lora_shrink_parts<float><<<sg, 256, 0, s_>>>(part, x, N, in, rk, J, slot, c, kc);
        lora_expand_parts<<<dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(groups)), 256,
                            sizeof(float) * kLoraRows * rank, s_>>>(part, static_cast<int>(P), N, static_cast<int>(rank), E, slot);
        SK_LAUNCH("lora");
    }

    // a long run of rows on one adapter (a prompt's, or many sequences' on it): cuBLAS on bf16
    // operands, the group's A x in one GEMM when its A rows sit back to back (dense_exec stacks
    // them), rounded to bf16, then each B's product added into y where it lives (beta 1)
    int64_t lora_rows_min() const override { return lora_gemm_rows_; }
    void lora_rows(const LoraJob *jobs, int n, const float *x, int64_t N, int64_t row0, int64_t M, int64_t in,
                   int64_t rank, int32_t slot, const int32_t *r, const float *scale) override {
        if (M <= 0 || n <= 0) return;
        // one GEMM for A rows back to back: then each job's A x is ldb (the pool's rank) wide
        const int64_t pool = jobs[0].ldb;
        bool stacked = true;
        for (int t = 1; t < n; ++t) stacked = stacked && jobs[t].a == jobs[0].a + t * pool * in && jobs[t].a_stride == jobs[0].a_stride;
        const int64_t R = stacked ? pool : rank, Kt = n * R;
        if (M > INT_MAX || in > INT_MAX || Kt > INT_MAX) cuda_fail("lora_rows", "dimension above INT_MAX");
        bind();
        const uint16_t *tw = x_twin(x, N, in);
        const uint16_t *xb = tw ? tw + row0 * in : x_bf16(x + row0 * in, M, in);
        auto *xa = static_cast<uint16_t *>(ensure(lora_xa_, sizeof(uint16_t) * M * Kt));
        const float one = 1.0f, zero = 0.0f;
        const int m = static_cast<int>(M), k = static_cast<int>(in), kt = static_cast<int>(Kt);
        for (int t = 0; t < (stacked ? 1 : n); ++t) {
            if (!stacked && r[t] <= 0) continue;
            const int rows = static_cast<int>(stacked ? Kt : R);
            SK_BLAS(cublasGemmEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, rows, m, k, &one, jobs[t].a + slot * jobs[t].a_stride,
                                 CUDA_R_16BF, k, xb, CUDA_R_16BF, k, &zero, xa + t * R, CUDA_R_16BF, kt,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                    "lora_rows/shrink");
        }
        for (int t = 0; t < n; ++t) {
            if (r[t] <= 0) continue;
            const LoraJob &J = jobs[t];
            if (J.out > INT_MAX) cuda_fail("lora_rows", "dimension above INT_MAX");
            const YLoc L = y_loc(J.y, N, J.out);
            void *c = L.b ? static_cast<void *>(L.b + row0 * L.ld) : static_cast<void *>(L.f + row0 * L.ld);
            SK_BLAS(cublasGemmEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(J.out), m, r[t], &scale[t],
                                 J.b + slot * J.out * J.ldb, CUDA_R_16BF, static_cast<int>(J.ldb), xa + t * R, CUDA_R_16BF,
                                 kt, &one, c, L.b ? CUDA_R_16BF : CUDA_R_32F, static_cast<int>(L.ld), CUBLAS_COMPUTE_32F,
                                 CUBLAS_GEMM_DEFAULT),
                    "lora_rows/expand");
        }
        SK_LAUNCH("lora_rows");
    }

    void adjust_rows(float *x, int64_t rows, int64_t n, const SampleRow *p, const LogitAdj *adj) override {
        materialize(x);
        if (rows <= 0) return;
        bind();
        adjust_rows_kernel<<<static_cast<unsigned>(rows), 256, 0, s_>>>(x, n, p, adj);
        SK_LAUNCH("adjust_rows");
    }

    void logprob_rows(float *lp, int32_t *ids, const float *x, int64_t rows, int64_t n, const SampleRow *p,
                      const int32_t *idx) override {
        materialize(x);
        if (rows <= 0) return;
        bind();
        auto *parts = static_cast<LogprobPart *>(ensure(lp_part_, size_t(rows) * kArgmaxSlices * sizeof(LogprobPart)));
        logprob_slices_kernel<<<dim3(static_cast<unsigned>(rows), kArgmaxSlices), kLogprobThreads, 0, s_>>>(parts, x, n, p);
        logprob_merge_kernel<<<static_cast<unsigned>(rows), kLogprobThreads, 0, s_>>>(lp, ids, parts, x, n, p, idx);
        SK_LAUNCH("logprob_rows");
    }

    void sample_rows(int32_t *idx, int32_t *bad, const float *x, int64_t rows, int64_t n, const SampleRow *p) override {
        materialize(x);
        if (rows <= 0) return;
        bind();
        float *w = static_cast<float *>(ensure(sample_w_, size_t(rows) * n * sizeof(float)));
        sample_rows_kernel<<<static_cast<unsigned>(rows), kSampleThreads, 0, s_>>>(idx, bad, x, n, p, w);
        SK_LAUNCH("sample_rows");
    }

    // ---------------------------------------------------------------- MoE routing
    void router_topk(int32_t *idx, float *w, const float *logits, const float *bias, int64_t N, int64_t E, int64_t k,
                     bool renorm, float scale, bool biased) override {
        materialize(logits);
        materialize(bias);
        if (N <= 0) return;
        if (E > 1024 || k < 1 || k > E) cuda_fail("router_topk", "need 1 <= k <= E <= 1024");
        bind();
        router_kernel<<<static_cast<unsigned>(N), kThreads, sizeof(float) * 2 * E, s_>>>(
            idx, w, logits, bias, static_cast<int>(E), static_cast<int>(k), renorm ? 1 : 0, scale, biased ? 1 : 0);
        SK_LAUNCH("router_topk");
    }

    // FlashAttention-3 (fa3_attention.h) over the page table: the step's rows [0, n_rows) as
    // one-query sequences (the decoding rows; 0 when the decode kernel keeps them) and its
    // chunks, all in one launch, any query count, the cache read in place (no gather). The
    // chunks' rows must follow the rows consecutively in segment order (the planner lays them
    // out so); false otherwise.
    bool fa3_prefill(float *o, const float *q, const void *kc, const void *vc, bool kv8, const int32_t *h_segs,
                     int64_t n_segs, int64_t n_rows, const int32_t *tok_slot, const int32_t *tok_pos,
                     const int32_t *page_table, int64_t max_pages, int64_t page, int64_t Hq, int64_t Hkv, int64_t D,
                     int64_t max_pos, float scale) {
        const int64_t QW = Hq * D;
        const int64_t min_first = n_rows > 0 ? 0 : (n_segs > 0 ? h_segs[0] : 0);
        int64_t next = n_rows > 0 ? n_rows : min_first, n_tok = n_rows, max_q = n_rows > 0 ? 1 : 0,
                max_k = n_rows > 0 ? max_pos + 1 : 0, max_slot = 0;
        for (int64_t i = 0; i < n_segs; ++i) {
            const int64_t first = h_segs[4 * i], count = h_segs[4 * i + 1], pos0 = h_segs[4 * i + 2], slot = h_segs[4 * i + 3];
            if (first != next || count <= 0) return false;
            next = first + count;
            n_tok = std::max(n_tok, next);
            max_q = std::max(max_q, count);
            max_k = std::max(max_k, pos0 + count);
            max_slot = std::max(max_slot, slot);
        }
        const int64_t covered = next - min_first;
        const int n = static_cast<int>(n_rows + n_segs);
        if (n <= 0 || covered <= 0) return false;
        // the per-sequence metadata, [cu_seqlens_q n + 1][seqused_k n][slot n]: the rows' part by
        // a kernel from the step's device arrays (once per step; in a captured graph, at every
        // replay), the chunks' part uploaded (once per step, and again within it when they change;
        // never in a graph, whose steps have no chunks). Both every step: a graph replay in between
        // rewrites the buffer with its rows (over an fp8 cache every decode step is one), so a step
        // whose chunks equal the last upload's must still upload them
        auto *meta = static_cast<int32_t *>(ensure(fa3_meta_, sizeof(int32_t) * (3 * n + 1)));
        const bool fresh = fa3_meta_dirty_ || meta != fa3_meta_dev_ || n != fa3_meta_n_;
        if (fresh) {
            fa3_rows_meta_kernel<<<static_cast<unsigned>(ceil_div(n_rows + 1, int64_t{128})), 128, 0, s_>>>(
                meta, tok_pos, tok_slot, static_cast<int>(n_rows), n);
            SK_LAUNCH("fa3/meta");
        }
        if (n_segs > 0) {
            // n_rows, n, then the chunks' cu_seqlens_q (their end rows, relative to the launch's
            // first row), seqused_k and slots
            fa3_meta_host_.assign(static_cast<size_t>(3 * n_segs) + 2, 0);
            fa3_meta_host_[0] = static_cast<int32_t>(n_rows), fa3_meta_host_[1] = n;
            int32_t *cu = fa3_meta_host_.data() + 2, *su = cu + n_segs, *sl = su + n_segs;
            for (int64_t j = 0; j < n_segs; ++j) {
                cu[j] = static_cast<int32_t>(h_segs[4 * j] + h_segs[4 * j + 1] - min_first);
                su[j] = h_segs[4 * j + 2] + h_segs[4 * j + 1];
                sl[j] = h_segs[4 * j + 3];
            }
            if (fresh || fa3_meta_host_ != fa3_meta_last_) {
                // pageable sources: each copy is staged before the call returns
                const size_t b = sizeof(int32_t) * static_cast<size_t>(n_segs);
                SK_CUDA(cudaMemcpyAsync(meta + n_rows + 1, cu, b, cudaMemcpyHostToDevice, s_), "fa3 metadata");
                SK_CUDA(cudaMemcpyAsync(meta + (n + 1) + n_rows, su, b, cudaMemcpyHostToDevice, s_), "fa3 metadata");
                SK_CUDA(cudaMemcpyAsync(meta + (n + 1) + n + n_rows, sl, b, cudaMemcpyHostToDevice, s_), "fa3 metadata");
                fa3_meta_last_ = fa3_meta_host_;
            }
        }
        fa3_meta_dev_ = meta, fa3_meta_n_ = n, fa3_meta_dirty_ = false;
        // q as bf16: RoPE's twin, else converted here (the launch's rows only); over an fp8 cache,
        // as e4m3 (FlashAttention-3's fp8 kernel takes all three operands in e4m3)
        const uint16_t *qtw = twin_get(q, n_tok, QW);
        const uint16_t *q16 = qtw ? qtw + min_first * QW : nullptr;
        const void *qa = q16;
        if (kv8 && !q16 && q8_.q == q && q8_.n >= min_first + covered) {
            qa = q8_.p + min_first * QW;  // qk_rope_append wrote it
        } else if (kv8) {
            auto *q8 = static_cast<uint8_t *>(ensure(fa3_q8_, covered * QW));
            const int64_t n = covered * QW;
            const unsigned g = static_cast<unsigned>(std::min<int64_t>(ceil_div(n, kThreads), 4 * sm_count_));
            if (q16) bf16_to_e4m3_kernel<<<g, kThreads, 0, s_>>>(q8, q16, n);
            else f32_to_e4m3_kernel<<<g, kThreads, 0, s_>>>(q8, q + min_first * QW, n);
            SK_LAUNCH("fa3/q8");
            qa = q8;
        } else if (!q16) {
            auto *qb = static_cast<uint16_t *>(ensure(fmha_q_, sizeof(uint16_t) * covered * QW));
            const int64_t n4 = covered * QW / 4;
            x_to_bf16_kernel<<<static_cast<unsigned>(std::min<int64_t>(ceil_div(n4, kThreads), 4 * sm_count_)), kThreads, 0, s_>>>(
                qb, q + min_first * QW, n4);
            SK_LAUNCH("fa3/q");
            qa = q16 = qb;
        }
        auto *o16 = static_cast<uint16_t *>(ensure(fmha_o_, sizeof(uint16_t) * n_tok * QW));
        auto *lse = static_cast<float *>(ensure(fa3_lse_, sizeof(float) * Hq * covered));
        auto *sched = static_cast<int32_t *>(ensure(fa3_sched_, sizeof(int32_t) * fa3_ops::scheduler_ints(n)));
        // few queries over long contexts split their keys across CTAs: the partials' buffers for
        // the kernel's upper bound (the metadata kernel settles each sequence's count)
        const int splits = fa3_ops::max_splits(n, static_cast<int>(max_q), static_cast<int>(max_k), static_cast<int>(Hq),
                                               static_cast<int>(Hkv), static_cast<int>(D), sm_count_, kv8);
        float *oacc = nullptr, *lseacc = nullptr;
        if (splits > 1) {
            oacc = static_cast<float *>(ensure(fa3_oacc_, sizeof(float) * splits * Hq * covered * D));
            lseacc = static_cast<float *>(ensure(fa3_lseacc_, sizeof(float) * splits * Hq * covered));
        }
        // the scheduler's metadata once per step: every layer launches over the same sequences, so
        // the first launch computes it (and zeroes the later launches' tile counters), the rest reuse
        int32_t *sem = nullptr;
        const std::array<int64_t, 6> key{n, covered, max_q, max_k, kv8 ? 1 : 0, reinterpret_cast<int64_t>(sched)};
        if (fa3_calls_ > 0 && fa3_calls_ < kFa3Sems && key == fa3_sched_key_) {
            sem = static_cast<int32_t *>(fa3_sems_.p) + fa3_calls_;
        } else {
            ensure(fa3_sems_, sizeof(int32_t) * kFa3Sems);
            SK_CUDA(cudaMemsetAsync(fa3_sems_.p, 0, sizeof(int32_t) * kFa3Sems, s_), "fa3 tile counters");
            fa3_calls_ = 0;
        }
        fa3_sched_key_ = key, ++fa3_calls_;
        fa3_ops::Prefill a{};
        a.semaphore = sem;
        a.fp8 = kv8;
        a.q = qa;
        a.o = o16 + min_first * QW;
        a.lse = lse;
        a.oaccum = oacc;
        a.lseaccum = lseacc;
        a.splits = splits;
        a.k_cache = kc;
        a.v_cache = vc;
        a.page_table = page_table;
        a.kv_batch_idx = meta + 2 * n + 1;
        a.cu_seqlens_q = meta;
        a.seqused_k = meta + n + 1;
        a.scheduler = sched;
        a.n = n;
        a.total_q = static_cast<int>(covered);
        a.max_q = static_cast<int>(max_q);
        a.max_k = static_cast<int>(max_k);
        a.Hq = static_cast<int>(Hq);
        a.Hkv = static_cast<int>(Hkv);
        a.D = static_cast<int>(D);
        a.page = static_cast<int>(page);
        a.max_pages = static_cast<int>(max_pages);
        a.num_pages = kv_pages_ > 0 ? kv_pages_ : max_pages * (max_slot + 1);  // the selftest sets no pool size
        a.scale = scale;
        a.sm_count = sm_count_;
        fa3_ops::prefill(a, s_);
        SK_LAUNCH("fa3");
        // every row of o written here: o stays bf16-only, the o GEMM (or GEMV) takes the twin; a
        // mixed step keeps the fp32 rows for the decode kernel's part
        const bool o_lazy = min_first == 0 && covered == n_tok && (n_tok * QW) % 4 == 0;
        if (!o_lazy) {
            const int64_t n4 = covered * QW / 4;
            bf16_to_f32_kernel<<<static_cast<unsigned>(std::min<int64_t>(ceil_div(n4, kThreads), 4 * sm_count_)), kThreads, 0, s_>>>(
                o + min_first * QW, o16 + min_first * QW, n4);
            SK_LAUNCH("fa3/o");
        }
        twin_set(o, n_tok, QW, o16, o_lazy);  // the o GEMM reads these rows
        return true;
    }
    // The step's attention over its decoding rows [0, n_dec) and its sequence chunks (h_segs:
    // first row, count, first position, slot), q as bf16, the bf16 output registered as o's
    // twin. FlashAttention-3 when built in (fa3_prefill: the chunks, and the rows too when
    // `take_rows`, the caller's choice); else the CUTLASS FMHA (see
    // cutlass_ops.h) over the chunks only: a launch per chunk over its K/V rows gathered
    // contiguous from the pages, queries padded to a multiple of 8, the output back to fp32, and
    // only the 1/sqrt(D) scale. FULLM_FMHA=cutlass forces the latter. Returns 2 when every row
    // was done, 1 when the chunks were and the rows were not, 0 when nothing was.
    int fmha_prefill(float *o, const float *q, const __nv_bfloat16 *kc, const __nv_bfloat16 *vc, const int32_t *h_segs,
                     int64_t n_segs, int64_t n_dec, bool take_rows, const int32_t *tok_slot, const int32_t *tok_pos,
                     const int32_t *page_table, int64_t max_pages, int64_t page, int64_t Hq, int64_t Hkv, int64_t D,
                     int64_t max_pos, float scale) {
        static const bool want_fa3 = [] {
            const char *e = std::getenv("FULLM_FMHA");
            return !(e && std::string(e) == "cutlass");
        }();
        if (want_fa3 && kt_.fa3_at(D)) {
            const int64_t rows = take_rows ? n_dec : 0;
            if ((rows > 0 || n_segs > 0) && fa3_prefill(o, q, kc, vc, false, h_segs, n_segs, rows, tok_slot, tok_pos,
                                                        page_table, max_pages, page, Hq, Hkv, D, max_pos, scale))
                return rows == n_dec ? 2 : 1;
        }
        if (n_segs <= 0 || !kt_.fmha_at(D) ||
            std::fabs(scale * std::sqrt(static_cast<float>(D)) - 1.0f) >= 1e-3f)
            return 0;
        // The kernel wants the query count a multiple of 8: pad the queries, and the keys by the
        // same amount so the end-aligned causal offset (keys - queries) is unchanged and no real
        // query sees a pad key; pad rows are zero and the pad outputs are dropped.
        auto pad8 = [](int64_t n) { return (n + 7) & ~int64_t{7}; };
        const int64_t W = Hkv * D, QW = Hq * D;
        int64_t max_k = 0, max_q = 0;
        for (int64_t i = 0; i < n_segs; ++i) {
            const int64_t nq = pad8(h_segs[4 * i + 1]);
            max_q = std::max<int64_t>(max_q, nq);
            max_k = std::max<int64_t>(max_k, h_segs[4 * i + 2] + nq);
        }
        auto *kd = static_cast<__nv_bfloat16 *>(ensure(fmha_k_, sizeof(__nv_bfloat16) * max_k * W));
        auto *vd = static_cast<__nv_bfloat16 *>(ensure(fmha_v_, sizeof(__nv_bfloat16) * max_k * W));
        int64_t n_tok = 0;  // rows of o this step (the chunks are consecutive)
        for (int64_t i = 0; i < n_segs; ++i) n_tok = std::max<int64_t>(n_tok, h_segs[4 * i] + h_segs[4 * i + 1]);
        const uint16_t *qtw = twin_get(q, n_tok, QW);  // RoPE wrote q as bf16 already (pad rows zero)
        auto *q16 = qtw ? nullptr : static_cast<uint16_t *>(ensure(fmha_q_, sizeof(uint16_t) * max_q * QW));
        auto *o16 = static_cast<uint16_t *>(ensure(fmha_o_, sizeof(uint16_t) * (n_tok + 8) * QW));  // + pad rows
        auto *lse = static_cast<float *>(ensure(fmha_lse_, sizeof(float) * max_q * Hq));
        const int hq = static_cast<int>(Hq), hk = static_cast<int>(Hkv), d = static_cast<int>(D);
        size_t ws_bytes = 0;
        for (int64_t i = 0; i < n_segs; ++i) {
            const int64_t nq = pad8(h_segs[4 * i + 1]);
            ws_bytes = std::max(ws_bytes, cutlass_ops::fmha_workspace_bytes(hq, hk, static_cast<int>(nq),
                                                                            static_cast<int>(h_segs[4 * i + 2] + nq), d));
        }
        void *ws = ensure(fmha_ws_, std::max<size_t>(ws_bytes, 16));
        // every row of o written here (no decoding rows in the step): o stays bf16-only, the o
        // GEMM takes the twin; a mixed step keeps the fp32 rows for the decode kernel's part
        int64_t covered = 0, min_first = n_tok;
        for (int64_t i = 0; i < n_segs; ++i) covered += h_segs[4 * i + 1], min_first = std::min<int64_t>(min_first, h_segs[4 * i]);
        const bool o_lazy = min_first == 0 && covered == n_tok && n_tok > kGemvMaxRows && (n_tok * QW) % 4 == 0;
        for (int64_t i = 0; i < n_segs; ++i) {
            const int64_t first = h_segs[4 * i], n_q = h_segs[4 * i + 1], n_k = h_segs[4 * i + 2] + n_q, slot = h_segs[4 * i + 3];
            const int64_t nq = pad8(n_q), nk = n_k + (nq - n_q);
            kv_gather_kernel<<<static_cast<unsigned>(nk), 128, 0, s_>>>(kd, vd, kc, vc, page_table + slot * max_pages, page, W,
                                                                        n_k);
            SK_LAUNCH("fmha/gather");
            const int64_t n4 = n_q * QW / 4;
            const uint16_t *qs = qtw ? qtw + first * QW : q16;
            if (!qtw) {
                x_to_bf16_kernel<<<static_cast<unsigned>(std::min<int64_t>(ceil_div(n4, kThreads), 4 * sm_count_)), kThreads, 0,
                                   s_>>>(q16, q + first * QW, n4);
                SK_LAUNCH("fmha/q");
                if (nq > n_q) SK_CUDA(cudaMemsetAsync(q16 + n_q * QW, 0, sizeof(uint16_t) * (nq - n_q) * QW, s_), "fmha/q pad");
            }
            cutlass_ops::fmha(qs, reinterpret_cast<const uint16_t *>(kd), reinterpret_cast<const uint16_t *>(vd), o16 + first * QW,
                              lse, hq, hk, static_cast<int>(nq), static_cast<int>(nk), d, ws, ws_bytes, s_);
            if (!o_lazy) {
                bf16_to_f32_kernel<<<static_cast<unsigned>(std::min<int64_t>(ceil_div(n4, kThreads), 4 * sm_count_)), kThreads, 0,
                                     s_>>>(o + first * QW, o16 + first * QW, n4);
                SK_LAUNCH("fmha/o");
            }
        }
        twin_set(o, n_tok, QW, o16, o_lazy);  // the o GEMM takes these rows as they are
        return n_dec > 0 ? 1 : 2;
    }

    // Prefill experts through CUTLASS's grouped GEMM (see cutlass_ops.h): x as bf16, its rows
    // gathered in pair-list order, gate/up -> GLU (bf16) -> down (fp32), scattered back to pair
    // order with the routing weights. A pair's rows depend only on its token and expert.
    void moe_experts_cutlass(float *contrib, const float *x, const float *w, int64_t N, int64_t k, const ExpertW *table,
                             const MoeLists &lists, int64_t width, int64_t inter, bool silu, float b1, float b2) {
        const int64_t pairs = N * k;
        const int groups = lists.slots;
        const uint16_t *x16 = twin_get(x, N, width);  // the add + norm wrote the layer input as bf16 too
        if (!x16) {
            auto *xb = static_cast<uint16_t *>(ensure(xb16_, sizeof(uint16_t) * N * width));
            const int64_t n4 = N * width / 4;
            launch_pdl(x_to_bf16_kernel, dim3(static_cast<unsigned>(std::min<int64_t>(ceil_div(n4, kThreads), 4 * sm_count_))),
                       dim3(kThreads), 0, s_, xb, x, n4);
            x16 = xb;
        }
        auto *xs = static_cast<uint16_t *>(ensure(moe_xs_, sizeof(uint16_t) * pairs * width));
        moe_gather_rows_kernel<<<static_cast<unsigned>(pairs), 128, 0, s_>>>(xs, x16, lists, static_cast<int>(k), width);
        SK_LAUNCH("moe_experts/gather");
        auto *g16 = static_cast<uint16_t *>(ensure(moe_g16_, sizeof(uint16_t) * pairs * 2 * inter));
        void *args = ensure(cutlass_args_, cutlass_ops::args_bytes(groups));
        // the down output as bf16 rows (this path is the bf16-activation mode; halves the GEMM's
        // store and the reduce's read, +4% prefill; FULLM_MOE_DOWN_BF16=0 keeps fp32 rows)
        static const bool down16 = env_on("FULLM_MOE_DOWN_BF16", true);
        const size_t ws_bytes = std::max(cutlass_ops::workspace_bytes(groups, false, 0),
                                         cutlass_ops::workspace_bytes(groups, !down16, 1));
        void *ws = ensure(cutlass_ws_, std::max<size_t>(ws_bytes, 16));
        cutlass_ops::fill_args(args, groups, lists.meta, table, true, xs, width, 2 * inter, g16, false, s_);
        cutlass_ops::run(args, groups, false, 0, ws, ws_bytes, s_);
        auto *s16 = static_cast<uint16_t *>(ensure(moe16_, sizeof(uint16_t) * pairs * inter));
        moe_glu16_kernel<<<grid1d(pairs * inter / 4), kThreads, 0, s_>>>(s16, g16, lists, inter, silu ? 1 : 0, b1, b2);
        SK_LAUNCH("moe_experts/glu");
        void *dn = ensure(moe_dn_, (down16 ? sizeof(uint16_t) : sizeof(float)) * pairs * width);
        cutlass_ops::fill_args(args, groups, lists.meta, table, false, s16, inter, width, dn, !down16, s_);
        cutlass_ops::run(args, groups, !down16, 1, ws, ws_bytes, s_);
        // no scatter: the reduce of this contrib reads dn through inv (pairs of non-resident
        // experts, inv < 0, still come from contrib, where scatter_rows puts them)
        sorted_ = Sorted{contrib, dn, lists.inv(), w, down16};
        sorted_pairs_ = pairs, sorted_width_ = width, sorted_lists_ = lists;
    }
    // FP8B experts through CUTLASS's blockwise-scaled grouped GEMM (cutlass_ops.h): the gathered
    // rows of x quantised to e4m3 (per row per 128 columns), gate/up -> bf16, the GLU quantised
    // the same way, down -> bf16 rows left in expert order for the fused reduce, as the bf16 path.
    // A pair's rows depend only on its token and expert.
    void moe_experts_cutlass_fp8(float *contrib, const float *x, const float *w, int64_t N, int64_t k,
                                 const ExpertW *table, const MoeLists &lists, int64_t width, int64_t inter, bool silu,
                                 float b1, float b2) {
        const int64_t pairs = N * k;
        const int groups = lists.slots;
        const uint16_t *x16 = twin_get(x, N, width);
        if (!x16) {
            auto *xb = static_cast<uint16_t *>(ensure(xb16_, sizeof(uint16_t) * N * width));
            const int64_t n4 = N * width / 4;
            launch_pdl(x_to_bf16_kernel, dim3(static_cast<unsigned>(std::min<int64_t>(ceil_div(n4, kThreads), 4 * sm_count_))),
                       dim3(kThreads), 0, s_, xb, x, n4);
            x16 = xb;
        }
        const int64_t ncb1 = width / 128, ncb2 = inter / 128;
        auto *xs8 = static_cast<uint8_t *>(ensure(moe_xs_, pairs * width));
        auto *sfa = static_cast<float *>(ensure(moe_sfa_, sizeof(float) * pairs * std::max(ncb1, ncb2)));
        moe_gather_quant_kernel<<<static_cast<unsigned>(pairs), 128, 0, s_>>>(xs8, sfa, x16, lists, static_cast<int>(k),
                                                                                width);
        SK_LAUNCH("moe_experts/gather8");
        auto *g16 = static_cast<uint16_t *>(ensure(moe_g16_, sizeof(uint16_t) * pairs * 2 * inter));
        void *args = ensure(cutlass_args_, cutlass_ops::fp8_args_bytes(groups));
        const size_t ws_bytes = cutlass_ops::fp8_workspace_bytes(groups);
        void *ws = ensure(cutlass_ws_, std::max<size_t>(ws_bytes, 16));
        cutlass_ops::fill_args_fp8(args, groups, lists.meta, table, true, xs8, sfa, width, 2 * inter, g16, s_);
        cutlass_ops::run_fp8(args, groups, ws, ws_bytes, s_);
        // the GLU's rows and scales (the gather's are consumed: the GEMM above is ahead in the stream)
        auto *s8 = static_cast<uint8_t *>(ensure(moe16_, pairs * inter));
        moe_glu_quant_kernel<<<static_cast<unsigned>(pairs), 128, 0, s_>>>(s8, sfa, g16, lists, inter, silu ? 1 : 0, b1,
                                                                             b2);
        SK_LAUNCH("moe_experts/glu8");
        auto *dn = static_cast<uint16_t *>(ensure(moe_dn_, sizeof(uint16_t) * pairs * width));
        cutlass_ops::fill_args_fp8(args, groups, lists.meta, table, false, s8, sfa, inter, width, dn, s_);
        cutlass_ops::run_fp8(args, groups, ws, ws_bytes, s_);
        sorted_ = Sorted{contrib, dn, lists.inv(), w, true};
        sorted_pairs_ = pairs, sorted_width_ = width, sorted_lists_ = lists;
    }
    // INT4G prefill on the int4 GEMM (int4_gemm.cu): x as the decode tiles take it, fp16 (clamped,
    // nearest even). The gather reads the router's fp16 copy, or the bf16 twin when x is bf16-only
    // (its fp32 would be that twin widened: the same bits; the materialize and fp16 passes go), or
    // else an fp16 copy made here once: gathering fp32 itself read each token's row k times at 4
    // bytes (kbench, 8192 tokens: 1606 us against 1570). gate/up with the GLU in the epilogue,
    // down -> bf16 rows left in expert order for the fused reduce, as above.
    void moe_experts_int4(float *contrib, const float *x, const __half *xh, const float *w, int64_t N, int64_t k,
                          const ExpertW *table, const MoeLists &lists, int64_t width, int64_t inter, bool silu, float b1,
                          float b2) {
        const int64_t pairs = N * k;
        const uint16_t *x16 = xh ? nullptr : twin_lazy(x, N, width);
        if (!xh && !x16) {
            materialize(x);  // a non-lazy twin is a rounding of x: fp32 then
            auto *c = static_cast<__half *>(ensure(xh16_, sizeof(__half) * N * width));
            const int64_t n4 = N * width / 4;
            launch_pdl(x_to_f16_kernel, dim3(static_cast<unsigned>(std::min<int64_t>(ceil_div(n4, kThreads), 4 * sm_count_))),
                       dim3(kThreads), 0, s_, c, x, n4);
            xh = c;
        }
        auto *xs = static_cast<__half *>(ensure(moe_xs_, sizeof(__half) * pairs * width));
        auto *s16 = static_cast<__half *>(ensure(moe16_, sizeof(__half) * pairs * inter));
        auto *dn = static_cast<uint16_t *>(ensure(moe_dn_, sizeof(uint16_t) * pairs * width));
        const int4_gemm::Lists L{lists.meta,  lists.exps(), lists.start(), lists.pass_u(), lists.pass_b(), lists.plist(),
                                 lists.maxp,  lists.pairs,  lists.slots};
        // the tensor maps cover the buffers (cached per buffer, not per step)
        const int64_t xs_rows = static_cast<int64_t>(moe_xs_.n / (sizeof(__half) * width));
        const int64_t s16_rows = static_cast<int64_t>(moe16_.n / (sizeof(__half) * inter));
        const void *xsrc = xh ? static_cast<const void *>(xh) : static_cast<const void *>(x16);
        const int4_gemm::XType xt = xh ? int4_gemm::XType::F16 : int4_gemm::XType::BF16;
        if (moe_hist_) moe_hist(lists, N);
        int4_gemm::gather(xs, xsrc, xt, L, static_cast<int>(k), width, pairs, s_);
        int4_gemm::gate_up(s16, xs, xs_rows, L, table, width, inter, silu, b1, b2, s_);
        int4_gemm::down(dn, s16, s16_rows, L, table, width, inter, s_);
        SK_LAUNCH("moe_experts/int4 gemm");
        sorted_ = Sorted{contrib, dn, lists.inv(), w, true};
        sorted_pairs_ = pairs, sorted_width_ = width, sorted_lists_ = lists;
    }
    // The routed pairs' rows of the last moe_experts, left in expert order (see above).
    struct Sorted {
        const float *contrib = nullptr;
        const void *dn = nullptr;  // fp32 rows, or bf16 when dn16
        const int32_t *inv = nullptr;
        const float *wts = nullptr;
        bool dn16 = false;
    } sorted_;
    Sorted take_sorted(const float *contrib) {
        const Sorted s = sorted_.contrib == contrib ? sorted_ : Sorted{};
        sorted_ = Sorted{};
        return s;
    }
    void moe_materialize(float *contrib) override {
        const Sorted so = take_sorted(contrib);
        if (!so.dn) return;
        bind();
        if (so.dn16)
            moe_scatter_kernel<uint16_t><<<static_cast<unsigned>(sorted_pairs_), 128, 0, s_>>>(
                contrib, static_cast<const uint16_t *>(so.dn), sorted_lists_, so.wts, sorted_width_);
        else
            moe_scatter_kernel<float><<<static_cast<unsigned>(sorted_pairs_), 128, 0, s_>>>(
                contrib, static_cast<const float *>(so.dn), sorted_lists_, so.wts, sorted_width_);
        SK_LAUNCH("moe_experts/scatter");
    }
    int64_t sorted_pairs_ = 0, sorted_width_ = 0;  // for moe_materialize
    // FULLM_MOE_HIST=1 (a debugging aid: it synchronises every call): the int4 GEMM's routing on
    // stderr, the experts' token counts and the passes per tile size (int4_gemm.cu: tile_tokens)
    const bool moe_hist_ = env_on("FULLM_MOE_HIST", false);
    void moe_hist(const MoeLists &lists, int64_t N) {
        std::vector<int32_t> meta(2), st(static_cast<size_t>(lists.slots) + 1);
        SK_CUDA(cudaMemcpyAsync(meta.data(), lists.meta, 2 * sizeof(int32_t), cudaMemcpyDeviceToHost, s_), "hist");
        SK_CUDA(cudaMemcpyAsync(st.data(), lists.start(), st.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, s_), "hist");
        SK_CUDA(cudaStreamSynchronize(s_), "hist");
        std::vector<int32_t> cnt;
        int tiles[3] = {0, 0, 0};  // passes on 32-, 64-, 128-token tiles
        for (int u = 0; u < meta[0]; ++u) {
            const int32_t c = st[u + 1] - st[u];
            cnt.push_back(c);
            for (int32_t j = 0; j < c; j += 128) {
                const int32_t m = std::min<int32_t>(128, c - j);
                ++tiles[m > 64 ? 2 : m > 32 ? 1 : 0];
            }
        }
        std::sort(cnt.begin(), cnt.end());
        auto q = [&](double f) { return cnt.empty() ? 0 : cnt[std::min(cnt.size() - 1, size_t(f * double(cnt.size())))]; };
        std::fprintf(stderr, "moe_hist N=%lld experts=%d tokens/expert min/p25/p50/p75/max %d/%d/%d/%d/%d passes 32/64/128: %d/%d/%d\n",
                     static_cast<long long>(N), meta[0], q(0.0), q(0.25), q(0.5), q(0.75), cnt.empty() ? 0 : cnt.back(),
                     tiles[0], tiles[1], tiles[2]);
    }
    MoeLists sorted_lists_{};

    // ---------------------------------------------------------------- resident experts
    bool moe_gate_up_interleaved(WFmt fmt, int64_t width, int64_t inter) const override {
        return tc_moe_ok(fmt, width, inter);
    }
    bool int4_prefill_gemm(int64_t width, int64_t inter) const override {
        return i4gemm_ok_ && tc_moe_ok(WFmt::INT4G, width, inter) && int4_gemm::shape_ok(width, inter);
    }
    static bool tc_moe_ok(WFmt fmt, int64_t width, int64_t inter) {
        if (fmt == WFmt::INT4G) return width % 256 == 0 && inter % 256 == 0;  // whole 128-byte row groups
        const int E = wfmt_bytes(fmt);
        return fmt != WFmt::F32 && (width * E) % 256 == 0 && (inter * E) % 256 == 0;
    }
    void moe_experts(float *contrib, const float *x, const int32_t *idx, const float *w, int64_t N, int64_t k,
                     const ExpertW *table, int64_t n_experts, WFmt fmt, int64_t width, int64_t inter, bool silu,
                     float b1, float b2, float *scratch) override {
        const int64_t pairs = N * k;
        sorted_ = Sorted{};
        // the router already built this call's lists (router_softmax_fused with the table)
        const bool routed = route_ready_.idx && route_ready_.idx == idx && route_ready_.table == table &&
                            route_ready_.pairs == pairs && route_ready_.n_experts == n_experts;
        const __half *xh_ready = route_ready_.xh && route_ready_.x == x && route_ready_.N == N && route_ready_.width == width
                                     ? route_ready_.xh
                                     : nullptr;
        route_ready_ = RouteReady{};
        if (pairs <= 0) return;
        if (tc_moe_ok(fmt, width, inter)) {
            if (n_experts > 1024) cuda_fail("moe_experts", "more than 1024 experts");
            bind();
            const int64_t slots = std::min<int64_t>(pairs, n_experts);
            const MoeLists lists = moe_lists(pairs, n_experts);
            if (!routed) {
                // a decode step's few pairs do not pay for 1024 threads
                launch_pdl(moe_route_kernel, dim3(1), dim3(pairs > 4096 ? kRouteThreads : 256), 0, s_, lists, idx, pairs,
                           table, static_cast<int>(n_experts));
                SK_LAUNCH("moe_experts/route");
            }
            // larger steps: two tiles per warp (items of 128 rows), half the items' start-up
            // latency; the same bits. Small ones keep 64-row items, enough blocks for the GPU.
            // (Four tiles per warp were measured slower at 8 and 16 tokens: fewer, longer items
            // load the persistent grid unevenly.)
            // (int4: always two, a warp's 16 rows being its tile's MMA rows)
            const int nt = (pairs >= moe_nt2_pairs_ || fmt == WFmt::INT4G) && (2 * inter) % 128 == 0 && width % 128 == 0 ? 2 : 1;
            // persistent grids: every resident block slot (no blocks for unused expert slots)
            const dim3 g1(static_cast<unsigned>(std::min<int64_t>(moe_blocks_, ceil_div(2 * inter, 64 * nt) * slots))),
                g2(static_cast<unsigned>(std::min<int64_t>(moe_blocks_, ceil_div(width, 64 * nt) * slots))),
                gf(static_cast<unsigned>(std::min<int64_t>(
                    moe_fused_blocks_, (ceil_div(2 * inter, 64 * nt) + ceil_div(width, 64 * nt)) * slots))),
                blk(kTcWarps * 32);
            const bool fused = moe_fused_ && moe_done_ && moe_fused_blocks_ > 0;
            const int kk = static_cast<int>(k), sl = silu ? 1 : 0, chain = moe_prefetch_ ? 1 : 0;
#define SK_TCMOE_NT(WF, BX, NT, XH)                                                                                      \
    if (fused) {                                                                                                         \
        launch_pdl(moe_tc_fused_kernel<WF, BX, NT, XH>, gf, blk, kTcSmem, s_, contrib, (void *)scratch, xin, lists, kk, w, \
                   table, width, inter, sl, b1, b2, chain, moe_done_);                                                    \
    } else {                                                                                                             \
        launch_pdl(moe_tc_gateup_kernel<WF, BX, NT, XH>, g1, blk, kTcSmem, s_, (void *)scratch, xin, lists, kk, table,   \
                   width, inter, sl, b1, b2, chain);                                                                     \
        launch_pdl(moe_tc_down_kernel<WF, BX, NT, XH>, g2, blk, kTcSmem, s_, contrib, (const void *)scratch, lists, w,    \
                   table, width, inter, chain);                                                                          \
    }
#define SK_TCMOE(WF, BX, XH)            \
    if (nt == 2) {                      \
        SK_TCMOE_NT(WF, BX, 2, XH);     \
    } else {                            \
        SK_TCMOE_NT(WF, BX, 1, XH);     \
    }
#define SK_TCMOE_F(WF)             \
    if (bx_) {                     \
        SK_TCMOE(WF, true, false)  \
    } else {                       \
        SK_TCMOE(WF, false, false) \
    }
// e4m3 weights in the bf16-activation mode: fp16 activation rows in and between the tiles
#define SK_TCMOE_H(WF)             \
    if (bx_) {                     \
        SK_TCMOE(WF, true, true)   \
    } else {                       \
        SK_TCMOE(WF, false, false) \
    }
            static const bool no_gemm = tc_env("FULLM_NO_MOE_GEMM", 0) != 0;
            static const bool no_cutlass = tc_env("FULLM_NO_CUTLASS", 0) != 0;
            const bool via_cutlass = fmt == WFmt::BF16 && bx_ && !no_cutlass && kt_.cutlass_bf16 &&
                                     N > kGemvMaxRows && width % 8 == 0 && inter % 8 == 0;
            // FP8B prefill: the blockwise-scaled fp8 GEMM (else the decode tiles, slower above 16 rows)
            const bool via_cutlass8 = fmt == WFmt::FP8B && bx_ && !no_cutlass && kt_.cutlass_fp8 &&
                                      N > kGemvMaxRows && width % 128 == 0 && inter % 128 == 0;
            // INT4G prefill: the int4 GEMM over the same weights (else the decode tiles)
            const bool via_i4 = fmt == WFmt::INT4G && N > i4gemm_rows_ && int4_prefill_gemm(width, inter);
            // the other paths read x as fp32 (CUTLASS and the int4 GEMM's gather take the twin)
            if (!via_cutlass && !via_cutlass8 && !via_i4) materialize(x);
            if (via_cutlass) {
                moe_experts_cutlass(contrib, x, w, N, k, table, lists, width, inter, silu, b1, b2);
            } else if (via_cutlass8) {
                moe_experts_cutlass_fp8(contrib, x, w, N, k, table, lists, width, inter, silu, b1, b2);
            } else if (via_i4) {
                moe_experts_int4(contrib, x, xh_ready, w, N, k, table, lists, width, inter, silu, b1, b2);
            } else if (fmt == WFmt::BF16 && bx_ && !no_gemm && N > kGemvMaxRows && width % kGeCols == 0 &&
                (2 * inter) % kGeCols == 0 && width % kGeK == 0 && inter % kGeK == 0) {
                // prefill: the tiled GEMM over bf16 rows of x (made once here) and of the GLU
                auto *x16 = static_cast<uint16_t *>(ensure(xb16_, sizeof(uint16_t) * N * width));
                const int64_t n4 = N * width / 4;
                launch_pdl(x_to_bf16_kernel, dim3(static_cast<unsigned>(std::min<int64_t>(ceil_div(n4, kThreads), 4 * sm_count_))),
                           dim3(kThreads), 0, s_, x16, x, n4);
                auto *s16 = static_cast<uint16_t *>(ensure(moe16_, sizeof(uint16_t) * pairs * inter));
                const int64_t items1 = ceil_div(2 * inter, kGeCols) * lists.maxp, items2 = ceil_div(width, kGeCols) * lists.maxp;
                const dim3 m1(static_cast<unsigned>(std::min<int64_t>(items1, ge_blocks_))),
                    m2(static_cast<unsigned>(std::min<int64_t>(items2, ge_blocks_))), gb(kGeThreads);
                launch_pdl(moe_gemm_kernel<true>, m1, gb, kGeSmem, s_, contrib, s16, (const uint16_t *)x16, lists, kk,
                           table, w, width, inter, sl, b1, b2);
                launch_pdl(moe_gemm_kernel<false>, m2, gb, kGeSmem, s_, contrib, s16, (const uint16_t *)x16, lists, kk,
                           table, w, width, inter, sl, b1, b2);
            } else {
            // the activations the tiles read: fp32 x, or for e4m3 weights x as fp16 (the router's
            // copy when it made one, else converted here, once)
            const void *xin = x;
            if (((fmt == WFmt::FP8 || fmt == WFmt::FP8B) && bx_) || fmt == WFmt::INT4G) {
                const __half *xh = xh_ready;
                if (!xh) {
                    auto *c = static_cast<__half *>(ensure(xh16_, sizeof(__half) * N * width));
                    const int64_t n4 = N * width / 4;
                    launch_pdl(x_to_f16_kernel, dim3(static_cast<unsigned>(std::min<int64_t>(ceil_div(n4, kThreads), 4 * sm_count_))),
                               dim3(kThreads), 0, s_, c, x, n4);
                    xh = c;
                }
                xin = xh;
            }
            switch (fmt) {
                case WFmt::BF16: SK_TCMOE_F(1) break;
                case WFmt::INT8: SK_TCMOE_F(2) break;
                case WFmt::FP8: SK_TCMOE_H(3) break;
                case WFmt::FP8B: SK_TCMOE_H(4) break;
                case WFmt::INT4G: SK_TCMOE_NT(5, true, 2, true) break;  // fp16 x rows in every mode
                default: break;
            }
            }
#undef SK_TCMOE_H
#undef SK_TCMOE_F
#undef SK_TCMOE
#undef SK_TCMOE_NT
            SK_LAUNCH("moe_experts/tc");
            return;
        }
        if (pairs > 65535) cuda_fail("moe_experts", "more than 65535 (token, expert) pairs in one call");
        const int E = fmt == WFmt::F32 ? 4 : fmt == WFmt::BF16 ? 2 : 1;
        if ((width * E) % 16 || (inter * E) % 16 || width % 4 || inter % 4)
            cuda_fail("moe_experts", "expert widths must give 16-byte rows");
        bind();
        // One block row per (token, expert) pair. Grouping pairs by expert (each expert streamed
        // once) was measured slower at decode sizes: ~1-2 pairs per expert, and the grouping pass
        // and lower x reuse cost more than the ~20% of expert bytes it saves at 16 tokens.
        const dim3 g1(static_cast<unsigned>(ceil_div(inter, 16)), static_cast<unsigned>(pairs));
        const dim3 g2(static_cast<unsigned>(ceil_div(width, 32)), static_cast<unsigned>(pairs));
        const int kk = static_cast<int>(k), sl = silu ? 1 : 0;
#define SK_MOE(WF)                                                                                                 \
    moe_gateup_kernel<WF><<<g1, 256, 0, s_>>>(scratch, x, idx, kk, table, width, inter, sl, b1, b2);           \
    SK_LAUNCH("moe_experts/gate_up");                                                                          \
    moe_down_kernel<WF><<<g2, 256, 0, s_>>>(contrib, scratch, idx, w, table, width, inter);                    \
    SK_LAUNCH("moe_experts/down")
        switch (fmt) {
            case WFmt::F32: SK_MOE(0); break;
            case WFmt::BF16: SK_MOE(1); break;
            case WFmt::INT8: SK_MOE(2); break;
            case WFmt::FP8: SK_MOE(3); break;
            case WFmt::FP8B: cuda_fail("moe_experts", "fp8b experts need rows of whole 256-byte groups"); break;
            case WFmt::INT4G: cuda_fail("moe_experts", "int4 experts need widths of whole 256 columns"); break;
        }
#undef SK_MOE
    }
    void moe_reduce(float *acc, const float *contrib, int64_t N, int64_t k, int64_t width,
                    const float *addend) override {
        materialize(contrib);
        materialize(addend);
        if (N <= 0) return;
        bind();
        const Sorted so = take_sorted(contrib);
        twin_clear(acc);
        if (so.dn16)
            launch_pdl(moe_reduce_kernel<uint16_t>, dim3(grid1d(N * width)), dim3(kThreads), 0, s_, acc, contrib, N,
                       static_cast<int>(k), width, addend, static_cast<const uint16_t *>(so.dn), so.inv, so.wts);
        else
            launch_pdl(moe_reduce_kernel<float>, dim3(grid1d(N * width)), dim3(kThreads), 0, s_, acc, contrib, N,
                       static_cast<int>(k), width, addend, static_cast<const float *>(so.dn), so.inv, so.wts);
        SK_LAUNCH("moe_reduce");
    }
    bool count_routes(uint32_t *counts, const int32_t *idx, int64_t n, int64_t n_counts) override {
        if (n <= 0) return true;
        bind();
        count_routes_kernel<<<static_cast<unsigned>(std::min<int64_t>(ceil_div(n, int64_t(256)), sm_count_)), 256, 0, s_>>>(
            counts, idx, n, n_counts);
        SK_LAUNCH("count_routes");
        return true;
    }
    void scatter_rows(float *dst, const float *src, const int32_t *rows, int64_t n, int64_t width) override {
        materialize(src);
        twin_clear(dst);
        if (n <= 0) return;
        bind();
        scatter_rows_kernel<<<grid1d(n * width), kThreads, 0, s_>>>(dst, src, rows, n, width);
        SK_LAUNCH("scatter_rows");
    }

protected:
    // the runtime's hooks (runtime.cuh)
    void step_begin() override {
        twins_clear_all();  // nothing from the last step is current
        fa3_meta_dirty_ = true;
        fa3_calls_ = 0;  // a new step: its first FlashAttention-3 launch computes the scheduler's metadata
    }
    void on_release(void *p) override {
        for (auto it = w8_bf16_.begin(); it != w8_bf16_.end();) {  // its bf16 copy goes with it
            if (std::get<0>(it->first) == p) {
                cudaFree(it->second);
                it = w8_bf16_.erase(it);
            } else {
                ++it;
            }
        }
    }
    void before_read(const void *p, size_t n) override { materialize_range(p, n); }
    void before_write(void *p) override { twin_clear(p); }

private:

    static void check_kvb(const Weight &w, int64_t H, int64_t qn, int64_t vh, const char *op) {
        if (w.fmt != WFmt::BF16) cuda_fail(op, "kv_b must be bf16");
        if (w.rows != H * (qn + vh)) cuda_fail(op, "kv_b rows != H*(qn+vh)");
    }

    // M > kGemvMaxRows: W (unscaled) and x to bf16, cuBLAS with fp32 output, then the
    // per-row scale on the output columns. int8 and e4m3 values are exact in bf16, so
    // only x is rounded (and f32 weights).
    // bf16 twins: a producer of an fp32 activation (the add + norm for the layer input, RoPE for
    // q, the FMHA for its output) also writes the bf16 copy its consumers want, so they skip the
    // conversion. A twin is valid until its buffer is rewritten: every writer of activations
    // clears the twin of its output, and upload() (the first call of every step) clears them all.
    // lazy: x's fp32 rows were never written (the producer stored only the twin, since every
    // consumer in the bf16-activation prefill takes bf16); materialize() converts the twin back
    // into x when something does read fp32 (a host download, the decode kernels, the fallbacks)
    struct XB16 {
        const float *x = nullptr;
        int64_t M = 0, K = 0;
        const uint16_t *b = nullptr;
        bool lazy = false;
    };
    XB16 twins_[8];
    void twins_clear_all() {
        for (auto &t : twins_) t = XB16{};
        qkv_ = QkvFused{};
    }
    void twin_clear(const void *x) {  // x is about to be rewritten: a lazy twin's value is dropped with it
        if (qkv_has(x)) qkv_materialize();  // the other two may still be read
        for (auto &t : twins_)
            if (t.x == x) t = XB16{};
    }
    // q, k and v of one gemm_group written as one bf16 matrix [M][ld] (columns q | k | v), their
    // fp32 buffers left unwritten: qk_rope_append reads it in place, any other reader materializes.
    // A dense MLP's gate and up the same way (q = gate, k = up, v null), for silu_glu.
    struct QkvFused {
        const float *q = nullptr, *k = nullptr, *v = nullptr;
        int64_t M = 0, ld = 0, nq = 0, nk = 0, nv = 0;
        const uint16_t *b = nullptr;
    } qkv_;
    Scratch qkv_buf_;
    // q's rows as e4m3, written by qk_rope_append for FlashAttention-3 over an fp8 cache
    struct Q8 {
        const float *q = nullptr;
        int64_t n = 0;
        const uint8_t *p = nullptr;
    } q8_;
    bool qkv_has(const void *x) const { return qkv_.b && x && (x == qkv_.q || x == qkv_.k || x == qkv_.v); }
    void qkv_materialize() {
        if (!qkv_.b) return;
        const QkvFused f = qkv_;
        qkv_ = QkvFused{};
        bind();
        const int64_t off[3] = {0, f.nq, f.nq + f.nk}, w[3] = {f.nq, f.nk, f.nv};
        const float *dst[3] = {f.q, f.k, f.v};
        for (int i = 0; i < 3; ++i) {
            const int64_t n = f.M * w[i];
            if (!dst[i] || n <= 0) continue;
            bf16_strided_to_f32_kernel<<<static_cast<unsigned>(std::min<int64_t>(ceil_div(n, kThreads), 4 * sm_count_)), kThreads, 0, s_>>>(
                const_cast<float *>(dst[i]), f.b + off[i], f.ld, w[i], n);
            SK_LAUNCH("qkv/materialize");
        }
    }
    void twin_set(const float *x, int64_t M, int64_t K, const uint16_t *b, bool lazy = false) {
        twin_clear(x);
        for (auto &t : twins_)
            if (!t.x) {
                t = XB16{x, M, K, b, lazy};
                return;
            }
        for (auto &t : twins_)  // full: evict a plain twin (its fp32 is good)
            if (!t.lazy) {
                t = XB16{x, M, K, b, lazy};
                return;
            }
        materialize(twins_[0]);  // every slot lazy: restore one before evicting it
        twins_[0] = XB16{x, M, K, b, lazy};
    }
    const uint16_t *twin_get(const float *x, int64_t M, int64_t K) const {
        for (const auto &t : twins_)
            if (t.x == x && t.M == M && t.K == K) return t.b;
        return nullptr;
    }
    // the twin when x has no fp32 rows yet (they would be the twin widened), else null
    const uint16_t *twin_lazy(const float *x, int64_t M, int64_t K) const {
        if (qkv_has(x)) return nullptr;
        for (const auto &t : twins_)
            if (t.x == x && t.M == M && t.K == K && t.lazy) return t.b;
        return nullptr;
    }
    void materialize(XB16 &t) {
        bind();
        const int64_t n4 = t.M * t.K / 4;
        bf16_to_f32_kernel<<<static_cast<unsigned>(std::min<int64_t>(ceil_div(n4, kThreads), 4 * sm_count_)), kThreads, 0, s_>>>(
            const_cast<float *>(t.x), t.b, n4);
        SK_LAUNCH("materialize");
        t.lazy = false;
    }
    // x's fp32 rows, if a producer left them bf16-only (no-op otherwise: every fp32 reader calls this)
    void materialize(const float *x) {
        if (qkv_has(x)) qkv_materialize();
        for (auto &t : twins_)
            if (t.lazy && t.x == x) materialize(t);
    }
    // any lazy rows inside [p, p + n) bytes
    void materialize_range(const void *p, size_t n) {
        const char *lo = static_cast<const char *>(p), *hi = lo + n;
        if (qkv_.b) {
            const float *b3[3] = {qkv_.q, qkv_.k, qkv_.v};
            const int64_t w3[3] = {qkv_.nq, qkv_.nk, qkv_.nv};
            bool hit = false;
            for (int i = 0; i < 3; ++i) {
                if (!b3[i]) continue;
                const char *tlo = reinterpret_cast<const char *>(b3[i]), *thi = tlo + sizeof(float) * qkv_.M * w3[i];
                hit = hit || (tlo < hi && lo < thi);
            }
            if (hit) qkv_materialize();
        }
        for (auto &t : twins_) {
            if (!t.lazy) continue;
            const char *tlo = reinterpret_cast<const char *>(t.x), *thi = tlo + sizeof(float) * t.M * t.K;
            if (tlo < hi && lo < thi) materialize(t);
        }
    }
    // prefill rows (bf16 activations): the producer writes the twin into this scratch
    uint16_t *twin_out(Scratch &sc, const float *x, int64_t M, int64_t K) {
        // (not below kGemvMaxRows even where cuBLAS takes the rows: steps of up to 16 rows keep their
        // fp32 rows, which the fused router's fp16 copy for the experts and other kernels read)
        if (!bx_ || M <= kGemvMaxRows) return nullptr;
        auto *b = static_cast<uint16_t *>(ensure(sc, sizeof(uint16_t) * (M + 8) * K));  // + pad rows for the FMHA
        twin_set(x, M, K, b);
        return b;
    }
    // the same, for a producer that will then not write x's fp32 rows at all
    uint16_t *twin_out_lazy(Scratch &sc, const float *x, int64_t M, int64_t K) {
        if (!bx_ || M <= kGemvMaxRows || (M * K) % 4) return nullptr;
        auto *b = static_cast<uint16_t *>(ensure(sc, sizeof(uint16_t) * (M + 8) * K));
        twin_set(x, M, K, b, true);
        return b;
    }
    // x as bf16 in the xbf_ scratch (the cuBLAS operand), or its twin
    const uint16_t *x_bf16(const float *x, int64_t M, int64_t K) {
        if (const uint16_t *b = twin_get(x, M, K)) return b;
        uint16_t *xb = static_cast<uint16_t *>(ensure(xbf_, sizeof(uint16_t) * M * K));
        to_bf16_kernel<0><<<grid1d(M * K), kThreads, 0, s_>>>(x, M * K, xb);
        SK_LAUNCH("gemm/x_to_bf16");
        return xb;
    }
    void gemm_blas(const Weight &w, const float *x, int64_t M, float *y, int64_t ldy) {
        gemm_blas16(w, x_bf16(x, M, w.cols), M, y, ldy);
    }
    // the result as bf16 rows (bf16 weights only: the 8-bit scale pass wants fp32)
    // n jobs on one x whose bf16 (or fp8, with their row scales too) weights sit back to back, each
    // result read only by the backend's next op (y16_ok): one GEMM over the stacked weights takes
    // them (an fp8 stack: if the fp8 GEMM does, see gemm_group)
    bool fusable(const GemmJob *j, int n) const {
        const Weight &a = *j[0].w;
        if (a.fmt != WFmt::BF16 && a.fmt != WFmt::FP8) return false;
        const int64_t es = a.fmt == WFmt::BF16 ? 2 : 1;
        const char *next = static_cast<const char *>(a.data);
        const float *next_s = a.scale;
        for (int i = 0; i < n; ++i) {
            const Weight &w = *j[i].w;
            if (w.fmt != a.fmt || !j[i].y16_ok || j[i].x != j[0].x || w.cols != a.cols || j[i].ldy != w.rows ||
                w.data != next || (a.fmt == WFmt::FP8 && (!w.scale || w.scale != next_s)))
                return false;
            next += es * w.rows * w.cols;
            if (next_s) next_s += w.rows;
        }
        return true;
    }
    void gemm_blas16_out16(const Weight &w, const uint16_t *xb, int64_t M, uint16_t *y16, int64_t ldy) {
        const int64_t N = w.rows, K = w.cols;
        if (N > INT_MAX || M > INT_MAX || K > INT_MAX || ldy > INT_MAX) cuda_fail("gemm", "dimension above INT_MAX");
        const void *wb = w.fmt == WFmt::BF16 ? w.data : w8_bf16(w);
        if (!wb) cuda_fail("gemm16", "bf16 weights (or a cached 8-bit copy) only");
        const float one = 1.0f, zero = 0.0f;
        SK_BLAS(cublasGemmEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(N), static_cast<int>(M),
                             static_cast<int>(K), &one, wb, CUDA_R_16BF, static_cast<int>(K), xb, CUDA_R_16BF,
                             static_cast<int>(K), &zero, y16, CUDA_R_16BF, static_cast<int>(ldy), CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT),
                "gemm/cublasGemmEx (bf16 out)");
    }
    void gemm16(const Weight &w, const float *x, int64_t M, float *y, int64_t ldy) override {
        if (M <= 0 || w.rows <= 0) return;
        LtPlan *p8 = bx_ && blas_rows(w, M) && ldy == w.rows ? fp8_plan(w, M, ldy, true) : nullptr;
        if (bx_ && blas_rows(w, M) && (w.fmt == WFmt::BF16 || p8 || w8_bf16(w)) && ldy == w.rows) {
            twin_clear(y);
            bind();
            if (uint16_t *y16 = twin_out_lazy(twin_o16_, y, M, ldy)) {
                if (p8) gemm_fp8(w, x, M, nullptr, y16, ldy, *p8);
                else gemm_blas16_out16(w, x_bf16(x, M, w.cols), M, y16, ldy);
                SK_LAUNCH("gemm16");
                return;
            }
        }
        gemm(w, x, M, y, ldy);
    }
    // ---- fp8 weights above the GEMV's rows: cuBLASLt e4m3 x e4m3 (W8A8), the weight's row scales
    // and x's per-row dynamic scales as outer vectors (CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F),
    // fp32 accumulation (no fast accumulation), bf16 or fp32 out. H100, Qwen3-8B's shapes at 17-256
    // rows: 0.61-0.87x the time of cuBLAS on bf16 weights (gate/up 83 -> 55 us), 0.70-0.75x at 4096
    // rows. The weights need no bf16 copies. FULLM_FP8_GEMM=0: the bf16 copies (w8_bf16).
    static constexpr size_t kLtWorkspace = size_t(32) << 20;
    struct LtPlan {
        cublasLtMatmulDesc_t op = nullptr;
        cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
        cublasLtMatmulAlgo_t algo{};
        bool ok = false;
        // post: the matmul runs with unit scales into fp32 (lc: fp32 [M][N], a row N apart) and
        // fp8_post_scale_kernel applies the row and token scales (and rounds to bf16 when out16):
        // a cuBLASLt without the outer-vector scale mode (sm_120). The shape, for that kernel.
        bool post = false, out16 = false;
        int64_t M = 0, N = 0, ldy = 0;
    };
    // the plan for y[M][ldy] = x w^T with fp8 w (null: not this weight, step or shape), made on the
    // first sighting of the shape (descriptors and cuBLASLt's first heuristic choice) and kept
    LtPlan *fp8_plan(const Weight &w, int64_t M, int64_t ldy, bool out16) {
        if (!lt_ || !bx_ || w.fmt != WFmt::FP8 || !w.scale || !blas_rows(w, M) || w.cols % 16 || ldy % (out16 ? 8 : 4) ||
            M > INT_MAX || w.rows > INT_MAX || w.cols > INT_MAX || ldy > INT_MAX || !aligned16(w.data) || !aligned16(w.scale))
            return nullptr;
        // small matrices keep their bf16 copies at decode sizes: x's quantization (~2 us) and the
        // one fp8 kernel's 128 x 128 tiles cost more than the bytes save there (Qwen3-30B-A3B's
        // attention projections, <= 10.5M elements: c=16-64 -2 to -5%); prefill sizes gain anyway.
        // Only where memory is plentiful (fp8_min_elems_): a copy is twice its fp8 weight's bytes.
        if (w.rows * w.cols < fp8_min_elems_ && M < 512) return nullptr;
        return lt_plan(M, w.rows, w.cols, ldy, out16, false);
    }
    // the plan for y[M][ldy] = x w^T, e4m3 w [N][K] and x [M][K] with outer-vector scales (a row's,
    // a token's) or, vec128, a scale per row (token) per 128 columns of K ([K / 128][N], [K / 128][M])
    LtPlan *lt_plan(int64_t M, int64_t N, int64_t K, int64_t ldy, bool out16, bool vec128) {
        const auto key = std::make_tuple(M, N, K, ldy, out16, vec128);
        auto it = lt_plans_.find(key);
        if (it == lt_plans_.end()) it = lt_plans_.emplace(key, lt_make(M, N, K, ldy, out16, vec128)).first;
        return it->second.ok ? &it->second : nullptr;
    }
    LtPlan lt_make(int64_t M, int64_t N, int64_t K, int64_t ldy, bool out16, bool vec128) {
        // post: scalar (unit) scales and an fp32 result, the scales applied by fp8_post_scale_kernel
        auto attempt = [&](bool post) {
            LtPlan p;
            p.post = post, p.out16 = out16, p.M = M, p.N = N, p.ldy = ldy;
            const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
            const cublasLtMatmulMatrixScale_t vec =
                vec128 ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F : CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
            const cudaDataType_t dt = out16 && !post ? CUDA_R_16BF : CUDA_R_32F;
            const int64_t ldc = post ? N : ldy;
            const int8_t fast = 0;
            // the heuristic wants scale pointers with the outer-vector modes (INVALID_VALUE without):
            // a placeholder until gemm_fp8 sets each call's. Not with scalar scales (null: none).
            const void *ph = lt_ws_;
            cublasLtMatmulPreference_t pref = nullptr;
            cublasLtMatmulHeuristicResult_t h{};
            int found = 0;
            const size_t wsb = kLtWorkspace;
            bool ok = cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS &&
                      cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta) == CUBLAS_STATUS_SUCCESS &&
                      cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb) == CUBLAS_STATUS_SUCCESS &&
                      (post ||
                       (cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &vec, sizeof vec) == CUBLAS_STATUS_SUCCESS &&
                        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &vec, sizeof vec) == CUBLAS_STATUS_SUCCESS &&
                        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &ph, sizeof ph) == CUBLAS_STATUS_SUCCESS &&
                        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &ph, sizeof ph) == CUBLAS_STATUS_SUCCESS)) &&
                      cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast, sizeof fast) == CUBLAS_STATUS_SUCCESS &&
                      cublasLtMatrixLayoutCreate(&p.la, CUDA_R_8F_E4M3, K, N, K) == CUBLAS_STATUS_SUCCESS &&
                      cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_8F_E4M3, K, M, K) == CUBLAS_STATUS_SUCCESS &&
                      cublasLtMatrixLayoutCreate(&p.lc, dt, N, M, ldc) == CUBLAS_STATUS_SUCCESS &&
                      cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS &&
                      cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof wsb) ==
                          CUBLAS_STATUS_SUCCESS &&
                      cublasLtMatmulAlgoGetHeuristic(lt_, p.op, p.la, p.lb, p.lc, p.lc, pref, 1, &h, &found) ==
                          CUBLAS_STATUS_SUCCESS &&
                      found > 0;
            if (pref) cublasLtMatmulPreferenceDestroy(pref);
            if (ok) p.algo = h.algo;
            else lt_free(p);
            p.ok = ok;
            return p;
        };
        LtPlan p;
        if (lt_outer_vec_ || vec128) p = attempt(false);
        // no outer-vector scaling on this device (sm_120's cuBLASLt: scalar scales only; vec128 has
        // no such fallback, its scales vary along K): the scales after the matmul
        if (!p.ok && !vec128) {
            p = attempt(true);
            if (p.ok && lt_outer_vec_) {
                lt_outer_vec_ = false;
                std::fprintf(stderr, "fullm: cuBLASLt fp8 GEMM without outer-vector scales (the scales applied after the matmul)\n");
            }
        }
        if (!p.ok && !lt_warned_) {
            lt_warned_ = true;
            std::fprintf(stderr, "fullm: no cuBLASLt fp8 GEMM for %lld x %lld x %lld%s (fp8 weights take their bf16 copies)\n",
                         static_cast<long long>(M), static_cast<long long>(N), static_cast<long long>(K),
                         vec128 ? " (128-block scales)" : "");
        }
        return p;
    }
    static void lt_free(LtPlan &p) {
        if (p.op) cublasLtMatmulDescDestroy(p.op);
        if (p.la) cublasLtMatrixLayoutDestroy(p.la);
        if (p.lb) cublasLtMatrixLayoutDestroy(p.lb);
        if (p.lc) cublasLtMatrixLayoutDestroy(p.lc);
        p.op = nullptr, p.la = p.lb = p.lc = nullptr;
    }
    // y (fp32) or y16 (bf16), rows ldy apart, = x w^T for an fp8 w through plan p: x's rows to e4m3
    // with a scale each (from its bf16 twin when it has one), then one cuBLASLt matmul
    void gemm_fp8(const Weight &w, const float *x, int64_t M, float *y, uint16_t *y16, int64_t ldy, LtPlan &p) {
        const int64_t K = w.cols;
        bind();
        auto *xq = static_cast<uint8_t *>(ensure(xq8_, static_cast<size_t>(M * K)));
        auto *sx = static_cast<float *>(ensure(sx8_, sizeof(float) * M));
        const uint16_t *xb = twin_get(x, M, K);
        if (!xb) materialize(x);
        act_quant(xq, sx, x, xb, M, K);
        SK_LAUNCH("gemm_fp8/quant");
        lt_run(p, w.data, w.scale, xq, sx, y16 ? static_cast<void *>(y16) : static_cast<void *>(y));
    }
    // one cuBLASLt matmul through plan p: e4m3 a (the weights) and b (x) with their scales, into d
    void lt_run(LtPlan &p, const void *a, const float *sa, const void *b, const float *sb, void *d) {
        const void *pa = sa, *pb = sb;
        const float one = 1.0f, zero = 0.0f;
        if (!p.post) {
            SK_BLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pa, sizeof pa), "fp8 gemm scales");
            SK_BLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &pb, sizeof pb), "fp8 gemm scales");
            SK_BLAS(cublasLtMatmul(lt_, p.op, &one, a, p.la, b, p.lb, &zero, d, p.lc, d, p.lc, &p.algo, lt_ws_, kLtWorkspace, s_),
                    "gemm/cublasLtMatmul (fp8)");
            return;
        }
        // unit scales into fp32: d itself (fp32 out, a row ldy apart: the layout's ld) or a scratch
        // (bf16 out), then the scales (and the rounding) in one pass
        float *acc = static_cast<float *>(d);
        if (p.out16) acc = static_cast<float *>(ensure(lt_acc_, sizeof(float) * p.M * p.N));
        SK_BLAS(cublasLtMatmul(lt_, p.op, &one, a, p.la, b, p.lb, &zero, acc, p.lc, acc, p.lc, &p.algo, lt_ws_, kLtWorkspace, s_),
                "gemm/cublasLtMatmul (fp8, unit scales)");
        const int64_t lda = p.out16 ? p.N : p.ldy;
        const bool v4 = p.N % 4 == 0 && lda % 4 == 0 && p.ldy % 4 == 0 && aligned16(acc) && aligned16(d) && aligned16(sa);
        const unsigned g = grid1d(p.M * (v4 ? p.N / 4 : p.N));
        if (p.out16) {
            if (v4) fp8_post_scale_kernel<uint16_t, true><<<g, kThreads, 0, s_>>>(static_cast<uint16_t *>(d), p.ldy, acc, lda, sa, sb, p.M, p.N);
            else fp8_post_scale_kernel<uint16_t, false><<<g, kThreads, 0, s_>>>(static_cast<uint16_t *>(d), p.ldy, acc, lda, sa, sb, p.M, p.N);
        } else {
            if (v4) fp8_post_scale_kernel<float, true><<<g, kThreads, 0, s_>>>(acc, p.ldy, acc, lda, sa, sb, p.M, p.N);
            else fp8_post_scale_kernel<float, false><<<g, kThreads, 0, s_>>>(acc, p.ldy, acc, lda, sa, sb, p.M, p.N);
        }
        SK_LAUNCH("gemm_fp8/post-scale");
    }

    // A dense MLP split into n int4 experts (dense.rs) on big steps: the int4 GEMM runs at the fp16
    // tensor rate (~69% of it at 1536 rows, Qwen3-8B: 24.6 of a prompt step's 44 ms), and folding
    // the unpacking into a W4A8 form of it did not pay (int4_gemm.cu). Here the unpacking is a pass
    // of its own: the experts' int4 weights as e4m3 (the integers exact) in two matrices, gate / up
    // rows expert after expert and down columns side by side, their group scales as they are
    // (int4_group.h's [K / 128][rows] is cuBLASLt's VEC128 layout); x as e4m3 with a scale per row
    // per 128 columns; gate / up on cuBLASLt's block-scaled fp8 GEMM into bf16, the GLU into e4m3
    // the same way, down into out. No pair rows, no reduce. Steps of split_mlp_rows_ (512) or more.
    bool split_mlp(float *out, const float *x, int64_t N, const ExpertW *ht, int64_t n, WFmt fmt, int64_t width,
                   int64_t inter) override {
        if (fmt != WFmt::INT4G || !lt_ || !bx_ || N < split_mlp_rows_ || N > INT_MAX || n <= 0 || n > kI4Max || width % 128 ||
            inter % 128 || !moe_gate_up_interleaved(fmt, width, inter) || 2 * inter * width / 8 > INT32_MAX)
            return false;
        I4Mats q13{}, q2{};
        for (int64_t e = 0; e < n; ++e) {
            const Weight &a = ht[e].w1, &d = ht[e].w2;
            if (a.fmt != WFmt::INT4G || d.fmt != WFmt::INT4G || a.rows != 2 * inter || a.cols != width || d.rows != width ||
                d.cols != inter || !a.scale || !d.scale || int4g_frag(a.rows, a.cols, a.layout) != int4g_frag(ht[0].w1.rows, ht[0].w1.cols, ht[0].w1.layout) ||
                int4g_frag(d.rows, d.cols, d.layout) != int4g_frag(ht[0].w2.rows, ht[0].w2.cols, ht[0].w2.layout))
                return false;
            q13.q[e] = static_cast<const uint32_t *>(a.data), q2.q[e] = static_cast<const uint32_t *>(d.data);
        }
        const int64_t R = n * 2 * inter, J = n * inter, G = width / 128;
        LtPlan *p1 = lt_plan(N, R, width, R, true, true), *p2 = lt_plan(N, width, J, width, false, true);
        if (!p1 || !p2) return false;
        bind();
        twin_clear(out);
        auto *w13 = static_cast<uint8_t *>(ensure(sm_w13_, size_t(R) * width));
        auto *s13 = static_cast<float *>(ensure(sm_s13_, sizeof(float) * G * R));
        auto *w2 = static_cast<uint8_t *>(ensure(sm_w2_, size_t(width) * J));
        auto *s2 = static_cast<float *>(ensure(sm_s2_, sizeof(float) * (J / 128) * width));
        const Weight &a0 = ht[0].w1, &d0 = ht[0].w2;
        auto blocks = [&](const Weight &w) {
            return dim3(static_cast<unsigned>(std::min<int64_t>(ceil_div(w.rows * w.cols / 32, kThreads), 8 * sm_count_ / n + 1)),
                        static_cast<unsigned>(n));
        };
        int4_to_e4m3_kernel<<<blocks(a0), kThreads, 0, s_>>>(w13, 2 * inter * width, width, q13, a0.rows, a0.cols,
                                                               int4g_frag(a0.rows, a0.cols, a0.layout));
        int4_to_e4m3_kernel<<<blocks(d0), kThreads, 0, s_>>>(w2, inter, J, q2, d0.rows, d0.cols, int4g_frag(d0.rows, d0.cols, d0.layout));
        for (int64_t e = 0; e < n; ++e) {
            const Weight &a = ht[e].w1, &d = ht[e].w2;
            SK_CUDA(cudaMemcpy2DAsync(s13 + e * 2 * inter, sizeof(float) * R, a.scale, sizeof(float) * 2 * inter, sizeof(float) * 2 * inter, G,
                                      cudaMemcpyDeviceToDevice, s_),
                    "split_mlp scales");
            SK_CUDA(cudaMemcpyAsync(s2 + e * (inter / 128) * width, d.scale, sizeof(float) * (inter / 128) * width, cudaMemcpyDeviceToDevice, s_),
                    "split_mlp scales");
        }
        SK_LAUNCH("split_mlp/weights");
        auto *xq = static_cast<uint8_t *>(ensure(sm_xq_, size_t(N) * width));
        auto *sx = static_cast<float *>(ensure(sm_sx_, sizeof(float) * G * N));
        const uint16_t *xb = twin_get(x, N, width);
        if (!xb) materialize(x);
        const unsigned qb = static_cast<unsigned>(std::min<int64_t>(ceil_div(N * G * 32, kThreads), 16 * sm_count_));
        if (xb) act_quant_g128_kernel<uint16_t><<<qb, kThreads, 0, s_>>>(xq, sx, xb, N, width);
        else act_quant_g128_kernel<float><<<qb, kThreads, 0, s_>>>(xq, sx, x, N, width);
        SK_LAUNCH("split_mlp/x");
        auto *d13 = static_cast<uint16_t *>(ensure(sm_d13_, sizeof(uint16_t) * N * R));
        lt_run(*p1, w13, s13, xq, sx, d13);
        auto *aq = static_cast<uint8_t *>(ensure(sm_aq_, size_t(N) * J));
        auto *sa = static_cast<float *>(ensure(sm_sa_, sizeof(float) * (J / 128) * N));
        const unsigned gb = static_cast<unsigned>(std::min<int64_t>(ceil_div(N * (J / 128) * 32, kThreads), 16 * sm_count_));
        glu_quant_g128_kernel<<<gb, kThreads, 0, s_>>>(aq, sa, d13, N, J);
        SK_LAUNCH("split_mlp/glu");
        lt_run(*p2, w2, s2, aq, sa, out);
        return true;
    }
    // x's rows (fp32, or xb: their bf16 twin) as e4m3 with a scale each: the row in registers, one
    // pass, where the row fits 256 threads x 64 and is 16-byte aligned (8 us a call -> ~2 at 32 rows
    // of Qwen3-8B); else the two-pass loop
    void act_quant(uint8_t *xq, float *sx, const float *x, const uint16_t *xb, int64_t M, int64_t K) {
        const int64_t per = (ceil_div(K, kThreads) + 7) / 8 * 8;
        const bool vec = K % 8 == 0 && per <= 64 && aligned16(xq) && (xb ? aligned16(xb) && (K * 2) % 16 == 0 : aligned16(x) && (K * 4) % 16 == 0);
        const dim3 g(static_cast<unsigned>(M)), b(kThreads);
#define SK_AQ(PER)                                                            \
    if (xb) act_quant_vec_kernel<uint16_t, PER><<<g, b, 0, s_>>>(xq, sx, xb, K); \
    else act_quant_vec_kernel<float, PER><<<g, b, 0, s_>>>(xq, sx, x, K)
        switch (vec ? per : 0) {
            case 8: SK_AQ(8); break;
            case 16: SK_AQ(16); break;
            case 24: SK_AQ(24); break;
            case 32: SK_AQ(32); break;
            case 40: SK_AQ(40); break;
            case 48: SK_AQ(48); break;
            case 56: SK_AQ(56); break;
            case 64: SK_AQ(64); break;
            default:
                if (xb) act_quant_e4m3_kernel<uint16_t><<<g, b, 0, s_>>>(xq, sx, xb, K);
                else act_quant_e4m3_kernel<float><<<g, b, 0, s_>>>(xq, sx, x, K);
        }
#undef SK_AQ
    }
    // xb: x already as bf16 [M][K]
    void gemm_blas16(const Weight &w, const uint16_t *xb, int64_t M, float *y, int64_t ldy) {
        const int64_t N = w.rows, K = w.cols;
        if (N > INT_MAX || M > INT_MAX || K > INT_MAX || ldy > INT_MAX) cuda_fail("gemm", "dimension above INT_MAX");
        const void *wb = w.data;
        // 8-bit weights: the cached bf16 copy with the row scales folded in (w8_bf16), so prefill
        // runs as with bf16 weights (no scale pass, bf16 results allowed)
        const bool folded = w8_bf16(w) != nullptr;
        if (folded) {
            wb = w8_bf16(w);
        } else if (w.fmt != WFmt::BF16) {
            uint16_t *s = static_cast<uint16_t *>(ensure(wbf_, sizeof(uint16_t) * N * K));
            switch (w.fmt) {
                case WFmt::F32: to_bf16_kernel<0><<<grid1d(N * K), kThreads, 0, s_>>>(w.data, N * K, s); break;
                case WFmt::INT8: to_bf16_kernel<2><<<grid1d(N * K), kThreads, 0, s_>>>(w.data, N * K, s); break;
                case WFmt::FP8: to_bf16_kernel<3><<<grid1d(N * K), kThreads, 0, s_>>>(w.data, N * K, s); break;
                default: break;
            }
            SK_LAUNCH("gemm/dequant");
            wb = s;
        }
        const float one = 1.0f, zero = 0.0f;
        SK_BLAS(cublasGemmEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(N), static_cast<int>(M),
                             static_cast<int>(K), &one, wb, CUDA_R_16BF, static_cast<int>(K), xb, CUDA_R_16BF,
                             static_cast<int>(K), &zero, y, CUDA_R_32F, static_cast<int>(ldy), CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT),
                "gemm/cublasGemmEx");
        if ((w.fmt == WFmt::INT8 || w.fmt == WFmt::FP8) && !folded) {
            scale_cols_kernel<<<grid1d(M * N), kThreads, 0, s_>>>(y, M, N, ldy, w.scale);
            SK_LAUNCH("gemm/scale");
        }
    }
    // An 8-bit weight's bf16 copy with its row scales folded in, made once and kept (the prefill
    // GEMMs then treat it as a bf16 weight); null for other formats or with FULLM_W8_CACHE=0.
    // The rounding of q * scale to bf16 is the copy's only difference from the 8-bit values.
    const void *w8_bf16(const Weight &w) {
        if ((w.fmt != WFmt::INT8 && w.fmt != WFmt::FP8) || !w8_cache_ || !w.scale) return nullptr;
        const W8Key key{w.data, w.rows, w.cols, static_cast<int>(w.fmt)};
        auto it = w8_bf16_.find(key);
        if (it == w8_bf16_.end()) {
            const int64_t N = w.rows, K = w.cols;
            void *d = nullptr;
            SK_CUDA(cudaMalloc(&d, sizeof(uint16_t) * N * K), "cudaMalloc (8-bit weight as bf16)");
            if (w.fmt == WFmt::INT8)
                scaled_to_bf16_kernel<2><<<grid1d(N * K), kThreads, 0, s_>>>(w.data, w.scale, N, K, static_cast<uint16_t *>(d));
            else
                scaled_to_bf16_kernel<3><<<grid1d(N * K), kThreads, 0, s_>>>(w.data, w.scale, N, K, static_cast<uint16_t *>(d));
            SK_LAUNCH("gemm/dequant once");
            it = w8_bf16_.emplace(key, d).first;
        }
        return it->second;
    }

    int moe_blocks_ = 264;
    cublasHandle_t blas_ = nullptr;
    cublasLtHandle_t lt_ = nullptr;  // the fp8 GEMMs (null: off)
    void *lt_ws_ = nullptr;
    bool lt_warned_ = false;
    bool lt_outer_vec_ = true;  // cuBLASLt's outer-vector scale mode works here (false: LtPlan::post)
    Scratch lt_acc_;            // the post-scaled fp8 GEMM's fp32 result before its bf16 rounding
    // see fp8_plan: below this many elements an fp8 weight's small steps take its bf16 copy. Default
    // 2^24 on devices of 32 GB or more (the H100), 0 (every step the fp8 GEMM, no copies) below:
    // an 8 GB GPU cannot spend 2x the trunk's bytes on copies. FULLM_FP8_GEMM_MIN=0 is 0 here.
    const int64_t fp8_min_elems_ = [this] {
        if (const char *v = std::getenv("FULLM_FP8_GEMM_MIN"); v && *v) return int64_t(std::atoll(v));
        return mem_total_ >= (size_t(32) << 30) ? int64_t(1) << 24 : int64_t(0);
    }();
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, bool, bool>, LtPlan> lt_plans_;
    // split_mlp's buffers: the e4m3 weights and their scales, x and the GLU as e4m3 with theirs, gate / up
    Scratch sm_w13_, sm_s13_, sm_w2_, sm_s2_, sm_xq_, sm_sx_, sm_d13_, sm_aq_, sm_sa_;
    // split_mlp takes steps of at least this many rows (FULLM_SPLIT_MLP_ROWS; see split_mlp)
    const int64_t split_mlp_rows_ = tc_env("FULLM_SPLIT_MLP_ROWS", 512);
    Scratch xq8_, sx8_;  // gemm_fp8's x as e4m3 and its row scales
    // the long-prompt kernels' scratch, one buffer (they never run at once): kda_chunked's matrices
    // per chunk and head; mla_prefill's latent rows, then a pass's keys, values, q and o
    Scratch prompt_;
    // prompts at least this long take kda_chunked (FULLM_KDA_CHUNK_MIN, a positive count;
    // FULLM_KDA_CHUNKED=0: never, every segment token by token)
    const int64_t kda_chunk_min_ = env_on("FULLM_KDA_CHUNKED", true) ? tc_env("FULLM_KDA_CHUNK_MIN", 64) : 0;
    // the chunks' scan on the tensor cores (3xTF32, sm80+); FULLM_KDA_SCAN_TC=0: the fp32 CUDA-core scan
    const bool kda_scan_tc_ = kt_.cc >= 80 && env_on("FULLM_KDA_SCAN_TC", true);
    // mla_prefill (beside prompt_): the log-sum-exp; the chunked path's merged log-sum-exp;
    // FlashAttention-3's scheduler metadata (and a chunk's meta after it); the segments' chunkings
    Scratch mla_lse_, mla_acc_, mla_sched_;
    std::vector<MlaChunks> chunks_;
    const int64_t mla_prefill_min_ = env_on("FULLM_MLA_PREFILL", true) ? tc_env("FULLM_MLA_PREFILL_MIN", 32) : 0;
    const int64_t mla_prefill_bytes_ = int64_t{tc_env("FULLM_MLA_PREFILL_MB", 128)} << 20;  // a pass's buffers
    // the most keys a launch expands (all the segments' in one, or a chunk's)
    const int64_t mla_prefill_keys_ = tc_env("FULLM_MLA_PREFILL_KEYS", 131072);
    Scratch lora_xa_;    // lora_rows' A x, bf16
    Scratch lora_part_;  // lora's partial sums [P][N][n rank]
    // lora_rows takes runs of at least this many rows (FULLM_LORA_GEMM_ROWS; 0: never)
    const int64_t lora_gemm_rows_ = tc_env("FULLM_LORA_GEMM_ROWS", 64);
    Scratch wbf_, xbf_, part_, moe_route_, argmax_part_, router_, xb16_, moe16_, sample_w_, lp_part_;
    Scratch moe_xs_, moe_g16_, moe_dn_, cutlass_args_, cutlass_ws_;  // the CUTLASS prefill path
    Scratch moe_sfa_;  // its fp8 activation scales (FP8B experts)
    Scratch xh16_;     // x as fp16 for the e4m3 expert tiles (router_softmax_fused, moe_experts)
    Scratch xh_trunk_;  // x as fp16 for the fp8 trunk GEMVs (xh_rows)
    // 8-bit trunk weights as bf16, for prefill: by (data, rows, cols, format); dropped when the
    // weight's memory is released (weights are allocated one per matrix)
    using W8Key = std::tuple<const void *, int64_t, int64_t, int>;
    std::map<W8Key, void *> w8_bf16_;
    const bool w8_cache_ = env_on("FULLM_W8_CACHE", true);
    Scratch fmha_k_, fmha_v_, fmha_q_, fmha_o_, fmha_lse_, fmha_ws_;  // the CUTLASS FMHA
    Scratch fa3_meta_, fa3_sched_, fa3_lse_, fa3_oacc_, fa3_lseacc_;  // FlashAttention-3
    // fp8 cache, prefill: bf16 staging pages, their sources and page table (kv8_stage)
    Scratch kv8_k_, kv8_v_, kv8_src_, kv8_pt_;
    std::vector<int32_t> kv8_key_;  // the chunks (and page geometry) the staging tables were built for
    int64_t kv8_ns_ = 0;

    // The staging pages of a step's prefill chunks: every page of each chunk's sequence up to the
    // chunk's end, numbered in chunk order; kv8_src_[j] = the page-table index (slot * max_pages +
    // logical page) of staging page j, kv8_pt_ the staging page table (rows by slot, as the cache's).
    // Built and uploaded when the chunks change (once per step: every layer has the same).
    int64_t kv8_stage(const int32_t *h_segs, int64_t n_segs, int64_t max_pages, int64_t page) {
        std::vector<int32_t> key(h_segs, h_segs + 4 * n_segs);
        key.push_back(static_cast<int32_t>(max_pages)), key.push_back(static_cast<int32_t>(page));
        if (key == kv8_key_) return kv8_ns_;
        std::map<int32_t, int64_t> need;  // slot -> pages
        for (int64_t i = 0; i < n_segs; ++i) {
            int64_t &np = need[h_segs[4 * i + 3]];
            np = std::max<int64_t>(np, ceil_div(int64_t{h_segs[4 * i + 2]} + h_segs[4 * i + 1], page));
        }
        int64_t ns = 0;
        for (const auto &kv : need) ns += kv.second;
        const int32_t max_slot = need.rbegin()->first;
        std::vector<int32_t> src(static_cast<size_t>(ns)), pt(static_cast<size_t>((max_slot + 1) * max_pages));
        int32_t j = 0;
        for (const auto &kv : need)
            for (int64_t lp = 0; lp < kv.second; ++lp, ++j) {
                src[static_cast<size_t>(j)] = static_cast<int32_t>(kv.first * max_pages + lp);
                pt[static_cast<size_t>(kv.first * max_pages + lp)] = j;
            }
        auto *ds = static_cast<int32_t *>(ensure(kv8_src_, sizeof(int32_t) * ns));
        auto *dp = static_cast<int32_t *>(ensure(kv8_pt_, sizeof(int32_t) * pt.size()));
        // pageable sources: each copy is staged before the call returns
        SK_CUDA(cudaMemcpyAsync(ds, src.data(), sizeof(int32_t) * ns, cudaMemcpyHostToDevice, s_), "fp8 staging");
        for (const auto &kv : need)
            SK_CUDA(cudaMemcpyAsync(dp + kv.first * max_pages, pt.data() + kv.first * max_pages, sizeof(int32_t) * kv.second,
                                    cudaMemcpyHostToDevice, s_),
                    "fp8 staging");
        kv8_key_ = std::move(key), kv8_ns_ = ns;
        return ns;
    }
    std::vector<int32_t> fa3_meta_host_, fa3_meta_last_;  // the chunks' metadata, and the step's last upload of it
    void *fa3_meta_dev_ = nullptr;
    int fa3_meta_n_ = -1;          // sequences the device metadata was built for
    bool fa3_meta_dirty_ = true;   // the rows' part must be recomputed (a new step)
    // FlashAttention-3 launches so far this step (the first computes the scheduler's metadata into
    // fa3_sched_, the next kFa3Sems - 1 reuse it, each with its own zeroed tile counter in fa3_sems_)
    static constexpr int kFa3Sems = 128;
    int fa3_calls_ = 0;
    std::array<int64_t, 6> fa3_sched_key_{};  // the launch the metadata was computed for
    Scratch fa3_sems_;
    int64_t kv_pages_ = 0;  // set_kv_pages
    WFmt expert_fmt_ = WFmt::BF16;  // set_expert_format: e4m3 experts get x as fp16 from the router
    Scratch twin_y_, twin_q_;  // bf16 twins written by add + norm and by RoPE
    Scratch fa3_q8_;           // e4m3 queries for FlashAttention-3 over an fp8 cache
    Scratch twin_g_[4], twin_o16_;  // bf16 results of gemm_group (y16_ok jobs) and of gemm16
    Scratch twin_a_;                // silu_glu16's bf16 output (the down projection's operand)
    const uint16_t *xb_group_ = nullptr;  // the bf16 x of the current gemm_group call
    int ge_blocks_ = 264;  // resident blocks of the prefill expert GEMM
    const int64_t xb_min_ = tc_env("FULLM_XB_MIN_ROWS", 8);
    // steps of more than kGemvMaxRows rows take cuBLAS; with bf16 activations, from this many rows
    // already (the executor's set_blas_min_rows: 4 for a dense model, 8 for a MoE): H100 decode,
    // cuBLAS against the GEMV at 8 / 16 rows, Qwen3-8B +2 / +8% (2000-token replies at 16: +7%),
    // Qwen3-30B-A3B int4 -2 / +3% (+2%); at 4 rows Qwen3-8B +5%, the MoE -2%. int8
    // weights too, through their bf16 copies (Qwen3-8B c = 8 / 16: 904 / 1055 -> 1055 / 2089 tok/s;
    // bf16 trunk 1116 / 2168); fp8 weights not: their fp16 tiles read half the bytes (1470 / 2477).
    // The GEMV is batch-invariant, cuBLAS run-to-run deterministic only: FULLM_BLAS_MIN_ROWS=17
    // keeps the GEMV to 16 rows.
    int64_t blas_min_rows_ = tc_env("FULLM_BLAS_MIN_ROWS", 8);
    // add_rmsnorm's largest cluster a row (4, 2 or 1 = off; see add_rmsnorm)
    const int norm_cluster_ = tc_env("FULLM_NORM_CLUSTER", 4);
    bool blas_rows(const Weight &w, int64_t M) const {
        return M > kGemvMaxRows || (bx_ && M >= blas_min_rows_ && w.fmt != WFmt::FP8);
    }
    const int64_t moe_nt2_pairs_ = tc_env("FULLM_MOE_NT2_PAIRS", 32);  // (token, expert) pairs for 2-tile items
    // blocks of the tensor-core decode attention (each walks several splits): about two waves
    int64_t attn_blocks_ = 528;
    int atc_stages_ = kAtcStages;  // its bf16 ring at head_dim 128: 3 stages, or 2 on small-smem devices
    // its shortest split, in positions (a multiple of 16; the other attention kernels keep
    // kAttnSplitLen): contexts up to it are one split, written without partials or a merge
    // the fused router's top-k in registers (FULLM_ROUTER_REGS=0: shared memory; the same bits)
    const bool router_regs_ = env_on("FULLM_ROUTER_REGS", true);
    const int atc_split_len_ = std::max(kAttnSplitLen, tc_env("FULLM_ATTN_SPLIT_LEN", kAttnSplitLen) / 16 * 16);
    // the expert tiles stream the next call's first weight groups during each call's last ones
    // (TcChain; the same bits either way)
    const bool moe_prefetch_ = env_on("FULLM_MOE_PREFETCH", true);
    unsigned *router_count_ = nullptr;  // the router's last-block counter, zero between calls
    unsigned *moe_done_ = nullptr;      // the fused expert kernel's per-expert counters, zero between calls
    int moe_fused_blocks_ = 0;          // its grid: at most the resident capacity
    // gate/up and down in one kernel (FULLM_MOE_FUSED=0: the two kernels; the same bits)
    const bool moe_fused_ = env_on("FULLM_MOE_FUSED", true);
    // INT4G steps of more than i4gemm_rows_ tokens run the int4 GEMM (int4_gemm.cu; off with
    // FULLM_NO_INT4_GEMM=1), smaller ones the decode tiles. With uniform routing the crossover is
    // near 512 tokens, but a served step's routing is skewed and the tiles' busiest expert sets
    // their time, so the default is 96 (speculative decode steps of 65-80 rows stay on the tiles).
    // A dense MLP split into experts that every token takes gives each expert all the step's rows:
    // from 16 there (the executor's set_int4_gemm_rows).
    bool i4gemm_ok_ = false;
    int64_t i4gemm_rows_ = tc_env("FULLM_INT4_GEMM_ROWS", kInt4TileRows);
    // routing lists the fused router built for the moe_experts call that follows (one step's
    // router -> experts pair; any other call in between clears it)
    struct RouteReady {
        const int32_t *idx = nullptr;  // the lists were built (null: only x's fp16 copy, if any)
        const ExpertW *table = nullptr;
        int64_t pairs = 0, n_experts = 0;
        const float *x = nullptr;  // x's fp16 copy, xh: rows [N][width]
        const __half *xh = nullptr;
        int64_t N = 0, width = 0;
    };
    RouteReady route_ready_;
    static constexpr int64_t kAttnCounters = 65536;
    unsigned *attn_count_ = nullptr;  // gqa_attention's per-(token, kv head) counters, zero between calls
    bool bx_ = false;  // tensor-core GEMVs and experts take bf16 activations (see set_exact_activations)
};

}  // namespace

bool cuda_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

std::unique_ptr<Backend> make_cuda_backend(int device) { return std::make_unique<CudaBackend>(device); }

}  // namespace fullm
