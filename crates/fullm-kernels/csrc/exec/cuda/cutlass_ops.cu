// SPDX-License-Identifier: Apache-2.0
// cutlass_ops.cu - see cutlass_ops.h. The grouped GEMM is CUTLASS's sm90 pointer-array kernel
// (TMA loads, wgmma, the cooperative warp-specialised schedule, 128 x 256 x 64 tiles): A and B
// are K-contiguous (A [M][K] row-major, B [N][K] = "TN"), alpha 1, beta 0. Problem sizes and
// pointers live on the device, so the routing never comes back to the host.

#include "cutlass_ops.h"

#ifdef FULLM_CUTLASS

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"

// the Hopper FMHA example (examples/88_hopper_fmha on the include path)
#include "collective/fmha_fusion.hpp"
#include "device/device_universal.hpp"
#include "kernel/fmha_kernel_builder.hpp"

namespace fullm::cutlass_ops {
namespace {

using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M, N, K> per group
using Shape3 = typename ProblemShape::UnderlyingProblemShape;
using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
constexpr int kAlignAB = 128 / cutlass::sizeof_bits<ElementA>::value;

// TN: the N tile (256 by default: ~4% faster than 128 on Qwen3-30B's experts; FULLM_CUTLASS_TN=128
// switches). PP: the ping-pong schedule (two consumer warpgroups alternating on 64-row tiles;
// FULLM_CUTLASS_SCHED=pingpong) instead of the cooperative one (both on a 128-row tile).
// CM: CTAs per cluster along M (2: the B tile is TMA-multicast to both; FULLM_CUTLASS_CLUSTER=2)
template <typename Out, int TN = 128, bool PP = false, int CM = 1>
struct GemmFor {
    using OutT = Out;
    using TileShape = Shape<Int<PP ? 64 : 128>, Int<TN>, _64>;
    using ClusterShape = Shape<Int<CM>, _1, _1>;
    using KernelSchedule = std::conditional_t<PP, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong,
                                              cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>;
    using EpilogueSchedule = std::conditional_t<PP, cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong,
                                                cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>;
    static constexpr int kAlignC = 128 / cutlass::sizeof_bits<Out>::value;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, float, float, Out, LayoutC *, kAlignC, Out, LayoutC *, kAlignC,
        EpilogueSchedule, cutlass::epilogue::fusion::LinearCombination<Out, float>>::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementA, LayoutA *, kAlignAB, ElementB, LayoutB *, kAlignAB,
        float, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    using StrideA = typename Kernel::InternalStrideA;
    using StrideB = typename Kernel::InternalStrideB;
    using StrideC = typename Kernel::InternalStrideC;
    using StrideD = typename Kernel::InternalStrideD;
};

// The strides do not depend on the output type (they count elements): one block layout.
using Ref = GemmFor<float>;
using StrideA = Ref::StrideA;
using StrideB = Ref::StrideB;
using StrideD = Ref::StrideD;
static_assert(sizeof(typename GemmFor<cutlass::bfloat16_t>::StrideD) == sizeof(StrideD), "stride layout");
static_assert(std::is_same_v<typename GemmFor<float, 256>::StrideA, StrideA> &&
                  std::is_same_v<typename GemmFor<float, 256>::StrideD, StrideD>,
              "the tile does not change the strides");

// The variant per projection (which: 0 = gate/up, 1 = down), from the environment:
// FULLM_CUTLASS_TN / FULLM_CUTLASS_DOWN_TN (128 | 256), FULLM_CUTLASS_SCHED /
// FULLM_CUTLASS_DOWN_SCHED (coop | pingpong), FULLM_CUTLASS_CLUSTER (1 | 2, gate/up only). All
// measured within noise of the default on Qwen3-30B's experts.
struct Variant {
    int tn = 256;
    bool pp = false, c2 = false;
};
Variant variant_from(const char *tn_var, const char *sched_var, const char *cluster_var) {
    Variant r;
    const char *tn = std::getenv(tn_var), *sc = std::getenv(sched_var);
    const char *cl = cluster_var ? std::getenv(cluster_var) : nullptr;
    if (tn && std::atoi(tn) == 128) r.tn = 128;
    if (sc && std::string(sc) == "pingpong") r.pp = true;
    if (cl && std::atoi(cl) == 2) r.c2 = true;
    return r;
}
const Variant &variant(int which) {
    static const Variant v[2] = {variant_from("FULLM_CUTLASS_TN", "FULLM_CUTLASS_SCHED", "FULLM_CUTLASS_CLUSTER"),
                                 variant_from("FULLM_CUTLASS_DOWN_TN", "FULLM_CUTLASS_DOWN_SCHED", nullptr)};
    return v[which == 1 ? 1 : 0];
}

// The device argument block: sizes, then the A / B / C / D pointer arrays, then the strides.
struct Block {
    Shape3 *sizes;
    const ElementA **A;
    const ElementB **B;
    const void **C;
    void **D;
    StrideA *sa;
    StrideB *sb;
    StrideD *sd;
};

size_t align16(size_t x) { return (x + 15) & ~size_t{15}; }

Block block_of(void *base, int groups) {
    char *p = static_cast<char *>(base);
    Block b;
    const size_t g = static_cast<size_t>(groups);
    b.sizes = reinterpret_cast<Shape3 *>(p), p += align16(sizeof(Shape3) * g);
    b.A = reinterpret_cast<const ElementA **>(p), p += align16(sizeof(void *) * g);
    b.B = reinterpret_cast<const ElementB **>(p), p += align16(sizeof(void *) * g);
    b.C = reinterpret_cast<const void **>(p), p += align16(sizeof(void *) * g);
    b.D = reinterpret_cast<void **>(p), p += align16(sizeof(void *) * g);
    b.sa = reinterpret_cast<StrideA *>(p), p += align16(sizeof(StrideA) * g);
    b.sb = reinterpret_cast<StrideB *>(p), p += align16(sizeof(StrideB) * g);
    b.sd = reinterpret_cast<StrideD *>(p);
    return b;
}

__global__ void fill_kernel(Block b, int groups, const int32_t *__restrict__ meta, const Backend::ExpertW *table,
                            int w1, const ElementA *A, int64_t K, int64_t N, char *D, int d_elem) {
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= groups) return;
    const int U = meta[0];
    const int32_t *exps = meta + 2, *start = exps + groups;
    int M = 0;
    int64_t a0 = 0;
    const void *bp = nullptr;
    if (u < U) {
        M = start[u + 1] - start[u];
        a0 = start[u];
        const Weight &w = w1 ? table[exps[u]].w1 : table[exps[u]].w2;
        bp = w.data;
    }
    if (!bp) bp = A;  // an empty group still carries a valid pointer
    b.sizes[u] = Shape3{M, static_cast<int>(N), static_cast<int>(K)};
    b.A[u] = A + a0 * K;
    b.B[u] = static_cast<const ElementB *>(bp);
    b.C[u] = nullptr;
    b.D[u] = D + a0 * N * d_elem;
    b.sa[u] = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, static_cast<int>(K), 1));
    b.sb[u] = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(static_cast<int>(N), static_cast<int>(K), 1));
    b.sd[u] = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, static_cast<int>(N), 1));
}

template <class G>
typename G::Gemm::Arguments make_args(const Block &b, int groups) {
    using Out = typename G::OutT;
    static const cutlass::KernelHardwareInfo hw = [] {
        int dev = 0;
        cudaGetDevice(&dev);
        return cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename G::Kernel>(dev);
    }();
    typename G::Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
                                     {groups, b.sizes, nullptr},
                                     {b.A, b.sa, b.B, b.sb},
                                     {{}, reinterpret_cast<const Out **>(b.C), reinterpret_cast<typename G::StrideC *>(b.sd),
                                      reinterpret_cast<Out **>(b.D), b.sd},
                                     hw};
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.beta = 0.0f;
    return args;
}

void check(cutlass::Status st, const char *what) {
    if (st != cutlass::Status::kSuccess)
        throw std::runtime_error(std::string("cutlass ") + what + ": " + cutlass::cutlassGetStatusString(st));
}

template <class G>
void run_t(const Block &b, int groups, void *ws, size_t ws_bytes, cudaStream_t s) {
    typename G::Gemm gemm;
    auto args = make_args<G>(b, groups);
    if (G::Gemm::get_workspace_size(args) > ws_bytes) throw std::runtime_error("cutlass grouped gemm: workspace too small");
    check(gemm.can_implement(args), "can_implement");
    check(gemm.initialize(args, ws, s), "initialize");
    check(gemm.run(s), "run");
}

// ---- fp8 experts: CUTLASS example 68's blockwise-scaled grouped GEMM. The kernel rebuilds each
// group's scale layouts from its (M, N, K) (packed: A's [K/128][M], B's [K/128][N/128]), so the
// layout arrays only have to be valid; they are filled alike.
// PP: the ping-pong schedule (two consumer warpgroups alternating on whole tiles, one's
// per-K-block scale promotion under the other's MMAs) instead of the cooperative one; CN: CTAs
// per cluster along N (the A tile multicast). FULLM_CUTLASS_FP8 = coop2 (default) | coop |
// pingpong picks one; the layouts and strides are the same for all. Measured, 8000-token
// prefill of Qwen3-30B-A3B: coop2 57.6k tok/s, coop 56.4k, pingpong 41.1k.
template <bool PP = false, int CN = 1>
struct Fp8GemmT {
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using Out = cutlass::bfloat16_t;
    using TileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, Int<CN>, _1>;
    using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128>;
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
    static constexpr int kAlignAB = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int kAlignC = 128 / cutlass::sizeof_bits<Out>::value;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, float, float, Out, LayoutC *, kAlignC, Out, LayoutC *, kAlignC,
        std::conditional_t<PP, cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong,
                           cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>,
        cutlass::epilogue::fusion::LinearCombination<Out, float>>::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementA, cute::tuple<LayoutA *, LayoutSFA *>, kAlignAB,
        ElementB, cute::tuple<LayoutB *, LayoutSFB *>, kAlignAB, float, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        std::conditional_t<PP, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8Blockwise,
                           cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8Blockwise>>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    using StrideA = typename Kernel::InternalStrideA;
    using StrideB = typename Kernel::InternalStrideB;
    using StrideC = typename Kernel::InternalStrideC;
    using StrideD = typename Kernel::InternalStrideD;
};
using Fp8Gemm = Fp8GemmT<>;  // the block layout's types (the same for every variant)
static_assert(std::is_same_v<Fp8GemmT<true>::StrideA, Fp8Gemm::StrideA> &&
                  std::is_same_v<Fp8GemmT<true>::StrideD, Fp8Gemm::StrideD> &&
                  std::is_same_v<Fp8GemmT<false, 2>::StrideB, Fp8Gemm::StrideB>,
              "fp8 variants share the argument block");
int fp8_variant() {  // 0 coop, 1 pingpong, 2 coop with 2-CTA clusters along N (default)
    static const int v = [] {
        const char *e = std::getenv("FULLM_CUTLASS_FP8");
        if (!e) return 2;
        const std::string s(e);
        return s == "pingpong" ? 1 : s == "coop" ? 0 : 2;
    }();
    return v;
}
template <class F>
auto with_fp8_variant(F &&f) {
    switch (fp8_variant()) {
        case 1: return f(Fp8GemmT<true>{});
        case 2: return f(Fp8GemmT<false, 2>{});
        default: return f(Fp8GemmT<>{});
    }
}

struct Block8 {
    Shape3 *sizes;
    const Fp8Gemm::ElementA **A;
    const Fp8Gemm::ElementB **B;
    const void **C;
    void **D;
    Fp8Gemm::StrideA *sa;
    Fp8Gemm::StrideB *sb;
    Fp8Gemm::StrideD *sd;
    const float **sfa;
    const float **sfb;
    Fp8Gemm::LayoutSFA *lsfa;
    Fp8Gemm::LayoutSFB *lsfb;
};

Block8 block8_of(void *base, int groups) {
    char *p = static_cast<char *>(base);
    Block8 b;
    const size_t g = static_cast<size_t>(groups);
    b.sizes = reinterpret_cast<Shape3 *>(p), p += align16(sizeof(Shape3) * g);
    b.A = reinterpret_cast<const Fp8Gemm::ElementA **>(p), p += align16(sizeof(void *) * g);
    b.B = reinterpret_cast<const Fp8Gemm::ElementB **>(p), p += align16(sizeof(void *) * g);
    b.C = reinterpret_cast<const void **>(p), p += align16(sizeof(void *) * g);
    b.D = reinterpret_cast<void **>(p), p += align16(sizeof(void *) * g);
    b.sa = reinterpret_cast<Fp8Gemm::StrideA *>(p), p += align16(sizeof(Fp8Gemm::StrideA) * g);
    b.sb = reinterpret_cast<Fp8Gemm::StrideB *>(p), p += align16(sizeof(Fp8Gemm::StrideB) * g);
    b.sd = reinterpret_cast<Fp8Gemm::StrideD *>(p), p += align16(sizeof(Fp8Gemm::StrideD) * g);
    b.sfa = reinterpret_cast<const float **>(p), p += align16(sizeof(void *) * g);
    b.sfb = reinterpret_cast<const float **>(p), p += align16(sizeof(void *) * g);
    b.lsfa = reinterpret_cast<Fp8Gemm::LayoutSFA *>(p), p += align16(sizeof(Fp8Gemm::LayoutSFA) * g);
    b.lsfb = reinterpret_cast<Fp8Gemm::LayoutSFB *>(p);
    return b;
}

__global__ void fill8_kernel(Block8 b, int groups, const int32_t *__restrict__ meta, const Backend::ExpertW *table, int w1,
                             const Fp8Gemm::ElementA *A, const float *sfa, int64_t K, int64_t N, Fp8Gemm::Out *D) {
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= groups) return;
    const int U = meta[0];
    const int32_t *exps = meta + 2, *start = exps + groups;
    int M = 0;
    int64_t a0 = 0;
    const void *bp = nullptr;
    const float *bs = nullptr;
    if (u < U) {
        M = start[u + 1] - start[u];
        a0 = start[u];
        const Weight &w = w1 ? table[exps[u]].w1 : table[exps[u]].w2;
        bp = w.data, bs = w.scale;
    }
    if (!bp) bp = A;  // an empty group still carries valid pointers
    if (!bs) bs = sfa;
    const int n = static_cast<int>(N), kk = static_cast<int>(K), kb = (kk + 127) / 128;
    b.sizes[u] = Shape3{M, n, kk};
    b.A[u] = A + a0 * K;
    b.B[u] = static_cast<const Fp8Gemm::ElementB *>(bp);
    b.C[u] = nullptr;
    b.D[u] = D + a0 * N;
    b.sfa[u] = sfa + a0 * kb;
    b.sfb[u] = bs;
    b.sa[u] = cutlass::make_cute_packed_stride(Fp8Gemm::StrideA{}, cute::make_shape(M, kk, 1));
    b.sb[u] = cutlass::make_cute_packed_stride(Fp8Gemm::StrideB{}, cute::make_shape(n, kk, 1));
    b.sd[u] = cutlass::make_cute_packed_stride(Fp8Gemm::StrideD{}, cute::make_shape(M, n, 1));
    b.lsfa[u] = Fp8Gemm::ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(M, n, kk, 1));
    b.lsfb[u] = Fp8Gemm::ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(M, n, kk, 1));
}

template <class G>
typename G::Gemm::Arguments make_args8(const Block8 &b, int groups) {
    using Out = typename G::Out;
    static const cutlass::KernelHardwareInfo hw = [] {
        int dev = 0;
        cudaGetDevice(&dev);
        return cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename G::Kernel>(dev);
    }();
    typename G::Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
                                     {groups, b.sizes, nullptr},
                                     {b.A, b.sa, b.B, b.sb, b.sfa, b.lsfa, b.sfb, b.lsfb},
                                     {{}, reinterpret_cast<const Out **>(b.C), reinterpret_cast<typename G::StrideC *>(b.sd),
                                      reinterpret_cast<Out **>(b.D), b.sd},
                                     hw};
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.beta = 0.0f;
    return args;
}

// Causal attention with the queries at the END of the key range (n_k >= n_q): chunk query i sees
// keys j <= i + (n_k - n_q). Adapted from the CausalFusion of CUTLASS example 88 (BSD-3-Clause;
// NOTICE), which puts the queries at the start.
struct EndCausalFusion : cutlass::fmha::collective::DefaultFusion {
    using Base = cutlass::fmha::collective::DefaultFusion;
    template <class BlkCoord, class TileShape, class ProblemSize>
    CUTLASS_DEVICE int get_trip_count(BlkCoord const &blk, TileShape const &tile, ProblemSize const &ps) {
        const int offset = get<3>(ps) - get<2>(ps);
        const int max_k = Base::get_trip_count(blk, tile, ps);
        const int max_q = ceil_div((get<0>(blk) + 1) * get<0>(tile) + offset, get<1>(tile));
        return std::min(max_k, max_q);
    }
    template <class BlkCoord, class TileShape, class ProblemSize>
    CUTLASS_DEVICE int get_masked_trip_count(BlkCoord const &blk, TileShape const &tile, ProblemSize const &ps) {
        // the k blocks that can hold this q block's diagonal (one more when the offset is not a
        // multiple of the k tile); masking a block that needs none is harmless
        return std::min(get_trip_count(blk, tile, ps), ceil_div(get<0>(tile), get<1>(tile)) + 1);
    }
    template <class BlkCoord, class TileShape, class ProblemSize>
    CUTLASS_DEVICE int get_unmasked_trip_count(BlkCoord const &blk, TileShape const &tile, ProblemSize const &ps) {
        return get_trip_count(blk, tile, ps) - get_masked_trip_count(blk, tile, ps);
    }
    template <class AccQK, class IndexQK, class ProblemSize>
    CUTLASS_DEVICE void before_softmax(AccQK &acc_qk, IndexQK const &index_qk, ProblemSize const &ps) {
        const int offset = get<3>(ps) - get<2>(ps);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(acc_qk); i++) {
            auto pos = index_qk(i);
            if (get<0>(pos) + offset < get<1>(pos)) acc_qk(i) = -INFINITY;
        }
    }
};

// TK: the key tile (128; 256 compiles but is not offered, see fmha_tk256). The example's
// IndividualTileScheduler runs one CTA per (query tile, batch, head) in ascending query-tile
// order (a longest-first order measured no better).
template <int D, int TK = 128>
struct FmhaFor {
    using Element = cutlass::bfloat16_t;
    using TileShape = Shape<_128, Int<TK>, Int<D>>;
    using Stride = cute::tuple<int, _1, cute::tuple<int, int>>;  // rows, columns, (batch, head)
    using Policy = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using Kernel = typename cutlass::fmha::kernel::FmhaBuilder<Element, float, float, TileShape, Stride, Stride, Stride,
                                                               EndCausalFusion, Policy>::Kernel;
    using Op = cutlass::device::Universal<Kernel>;
};

// The kv heads are the batch mode and the group's heads the head mode: Q and O [n][Hq][D] with
// head kh * G + g, K and V [n][Hkv][D] shared by the group (head stride 0).
// A 128 x 256 key tile (FmhaFor<128, 256>) compiles but fails in the kernel's initialize on this
// CUTLASS (Error Internal), so it is not offered; the 128 x 128 tile measures ~450 TFLOPS.
bool fmha_tk256() { return false; }

template <int D, int TK = 128>
typename FmhaFor<D, TK>::Op::Arguments fmha_args(const uint16_t *Q, const uint16_t *K, const uint16_t *V, uint16_t *O,
                                                 float *lse, int Hq, int Hkv, int n_q, int n_k) {
    using F = FmhaFor<D, TK>;
    using E = typename F::Element;
    static const cutlass::KernelHardwareInfo hw = [] {
        int dev = 0;
        cudaGetDevice(&dev);
        return cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename F::Kernel>(dev);
    }();
    const int G = Hq / Hkv;
    typename F::Stride sQ = make_stride(Hq * D, _1{}, make_stride(G * D, D));
    typename F::Stride sK = make_stride(Hkv * D, _1{}, make_stride(D, 0));
    cute::tuple<_1, cute::tuple<int, int>> sL = make_stride(_1{}, make_stride(G * n_q, n_q));
    return typename F::Op::Arguments{cute::make_tuple(Hkv, G, n_q, n_k, D),
                                     {reinterpret_cast<const E *>(Q), sQ, reinterpret_cast<const E *>(K), sK,
                                      reinterpret_cast<const E *>(V), sK},
                                     {reinterpret_cast<E *>(O), sQ, lse, sL},
                                     hw};
}

template <int D, int TK = 128>
void fmha_t(const uint16_t *Q, const uint16_t *K, const uint16_t *V, uint16_t *O, float *lse, int Hq, int Hkv, int n_q,
            int n_k, void *ws, size_t ws_bytes, cudaStream_t s) {
    using F = FmhaFor<D, TK>;
    typename F::Op op;
    auto args = fmha_args<D, TK>(Q, K, V, O, lse, Hq, Hkv, n_q, n_k);
    if (F::Op::get_workspace_size(args) > ws_bytes) throw std::runtime_error("cutlass fmha: workspace too small");
    check(op.can_implement(args), "fmha can_implement");
    check(op.initialize(args, ws, s), "fmha initialize");
    check(op.run(s), "fmha run");
}

}  // namespace

bool fmha_available(int D) { return D == 64 || D == 128; }

size_t fmha_workspace_bytes(int Hq, int Hkv, int n_q, int n_k, int D) {
    if (D == 64) return FmhaFor<64>::Op::get_workspace_size(fmha_args<64>(nullptr, nullptr, nullptr, nullptr, nullptr, Hq, Hkv, n_q, n_k));
    if (fmha_tk256())
        return FmhaFor<128, 256>::Op::get_workspace_size(fmha_args<128, 256>(nullptr, nullptr, nullptr, nullptr, nullptr, Hq, Hkv, n_q, n_k));
    return FmhaFor<128>::Op::get_workspace_size(fmha_args<128>(nullptr, nullptr, nullptr, nullptr, nullptr, Hq, Hkv, n_q, n_k));
}

void fmha(const uint16_t *Q, const uint16_t *K, const uint16_t *V, uint16_t *O, float *lse, int Hq, int Hkv, int n_q,
          int n_k, int D, void *ws, size_t ws_bytes, cudaStream_t s) {
    if (D == 64) fmha_t<64>(Q, K, V, O, lse, Hq, Hkv, n_q, n_k, ws, ws_bytes, s);
    else if (D == 128 && fmha_tk256()) fmha_t<128, 256>(Q, K, V, O, lse, Hq, Hkv, n_q, n_k, ws, ws_bytes, s);
    else if (D == 128) fmha_t<128>(Q, K, V, O, lse, Hq, Hkv, n_q, n_k, ws, ws_bytes, s);
    else throw std::runtime_error("cutlass fmha: head_dim must be 64 or 128");
}

bool available() { return true; }

size_t args_bytes(int groups) {
    const size_t g = static_cast<size_t>(groups);
    return align16(sizeof(Shape3) * g) + 4 * align16(sizeof(void *) * g) + align16(sizeof(StrideA) * g) +
           align16(sizeof(StrideB) * g) + align16(sizeof(StrideD) * g);
}

// f(GemmFor<Out, ...>{}) for the compiled variant v selects
template <typename Out, class F>
auto with_variant(const Variant &v, F &&f) {
    if (v.pp) return f(GemmFor<Out, 256, true>{});
    if (v.c2) return f(GemmFor<Out, 256, false, 2>{});
    if (v.tn == 128) return f(GemmFor<Out, 128>{});
    return f(GemmFor<Out, 256>{});
}

size_t workspace_bytes(int groups, bool out_f32, int which) {
    Block b{};
    auto ws = [&](auto g) {
        using G = decltype(g);
        return G::Gemm::get_workspace_size(make_args<G>(b, groups));
    };
    return out_f32 ? with_variant<float>(variant(which), ws) : with_variant<cutlass::bfloat16_t>(variant(which), ws);
}

void fill_args(void *args, int groups, const int32_t *meta, const Backend::ExpertW *table, bool w1, const uint16_t *A,
               int64_t K, int64_t N, void *D, bool out_f32, cudaStream_t s) {
    const Block b = block_of(args, groups);
    fill_kernel<<<(groups + 127) / 128, 128, 0, s>>>(b, groups, meta, table, w1 ? 1 : 0,
                                                     reinterpret_cast<const ElementA *>(A), K, N, static_cast<char *>(D),
                                                     out_f32 ? 4 : 2);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("cutlass fill_args: ") + cudaGetErrorString(e));
}

void run(const void *args, int groups, bool out_f32, int which, void *workspace, size_t workspace_bytes,
         cudaStream_t s) {
    const Block b = block_of(const_cast<void *>(args), groups);
    auto go = [&](auto g) { run_t<decltype(g)>(b, groups, workspace, workspace_bytes, s); };
    if (out_f32) with_variant<float>(variant(which), go);
    else with_variant<cutlass::bfloat16_t>(variant(which), go);
}

bool fp8_available() { return true; }

size_t fp8_args_bytes(int groups) {
    const size_t g = static_cast<size_t>(groups);
    return align16(sizeof(Shape3) * g) + 6 * align16(sizeof(void *) * g) + align16(sizeof(Fp8Gemm::StrideA) * g) +
           align16(sizeof(Fp8Gemm::StrideB) * g) + align16(sizeof(Fp8Gemm::StrideD) * g) +
           align16(sizeof(Fp8Gemm::LayoutSFA) * g) + align16(sizeof(Fp8Gemm::LayoutSFB) * g);
}

size_t fp8_workspace_bytes(int groups) {
    Block8 b{};
    return with_fp8_variant([&](auto g) {
        using G = decltype(g);
        return G::Gemm::get_workspace_size(make_args8<G>(b, groups));
    });
}

void fill_args_fp8(void *args, int groups, const int32_t *meta, const Backend::ExpertW *table, bool w1, const uint8_t *A,
                   const float *sfa, int64_t K, int64_t N, uint16_t *D, cudaStream_t s) {
    const Block8 b = block8_of(args, groups);
    fill8_kernel<<<(groups + 127) / 128, 128, 0, s>>>(b, groups, meta, table, w1 ? 1 : 0,
                                                      reinterpret_cast<const Fp8Gemm::ElementA *>(A), sfa, K, N,
                                                      reinterpret_cast<Fp8Gemm::Out *>(D));
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("cutlass fill_args_fp8: ") + cudaGetErrorString(e));
}

void run_fp8(const void *args, int groups, void *workspace, size_t workspace_bytes, cudaStream_t s) {
    const Block8 b = block8_of(const_cast<void *>(args), groups);
    with_fp8_variant([&](auto g) {
        using G = decltype(g);
        typename G::Gemm gemm;
        auto a = make_args8<G>(b, groups);
        if (G::Gemm::get_workspace_size(a) > workspace_bytes)
            throw std::runtime_error("cutlass fp8 grouped gemm: workspace too small");
        check(gemm.can_implement(a), "fp8 can_implement");
        check(gemm.initialize(a, workspace, s), "fp8 initialize");
        check(gemm.run(s), "fp8 run");
        return 0;
    });
}

}  // namespace fullm::cutlass_ops

#else  // no CUTLASS in this build

namespace fullm::cutlass_ops {
bool available() { return false; }
size_t args_bytes(int) { return 0; }
size_t workspace_bytes(int, bool, int) { return 0; }
void fill_args(void *, int, const int32_t *, const Backend::ExpertW *, bool, const uint16_t *, int64_t, int64_t, void *,
               bool, cudaStream_t) {}
void run(const void *, int, bool, int, void *, size_t, cudaStream_t) {}
bool fp8_available() { return false; }
size_t fp8_args_bytes(int) { return 0; }
size_t fp8_workspace_bytes(int) { return 0; }
void fill_args_fp8(void *, int, const int32_t *, const Backend::ExpertW *, bool, const uint8_t *, const float *, int64_t,
                   int64_t, uint16_t *, cudaStream_t) {}
void run_fp8(const void *, int, void *, size_t, cudaStream_t) {}
bool fmha_available(int) { return false; }
size_t fmha_workspace_bytes(int, int, int, int, int) { return 0; }
void fmha(const uint16_t *, const uint16_t *, const uint16_t *, uint16_t *, float *, int, int, int, int, int, void *, size_t,
          cudaStream_t) {}
}  // namespace fullm::cutlass_ops

#endif
