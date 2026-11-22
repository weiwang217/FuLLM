// SPDX-License-Identifier: Apache-2.0
// conformance.cpp - every Backend op of a backend (the subject) against a reference backend (the
// CPU's), on identical random inputs: `fullm selftest --backend NAME`.
// The groups and cases of model families or KV formats the subject does not declare in its Caps
// are skipped. sk_exec_selftest is the CUDA backend against the CPU's.
//
// The CPU backend is the reference (it is itself checked against the torch oracle through
// the K3 graph). A failure here names the op and the shape, which is the point: a kernel
// bug should be found at the kernel, not as a different token 90 layers later.
//
// Tolerances are relative to the largest reference magnitude in each output:
//   exact-fp32 paths (GEMV, norms, recurrences)  ~1e-5 .. 1e-4  (summation order only)
//   cuBLAS bf16 path (M > kGemvMaxRows)          2e-2           (x and f32 W rounded to bf16)
//   bf16 state / cache                           2e-3 .. 1e-2   (one rounding may differ)
// Integer outputs (router indices) must match exactly.

#include "fp8_block.h"
#include "int4_group.h"
#include "kv_fp8.h"
#include "kv_int4.h"
#include "backend.h"
#include "fullm_exec.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace {

using fullm::Backend;
using fullm::DType;
using fullm::Weight;
using fullm::WFmt;
using fullm::fp8b_scales;
using fullm::quantize_fp8_blocks;
using fullm::int4g_bytes;
using fullm::int4g_scales;
using fullm::quantize_int4_groups;
using fullm::kv8_page_bytes;
using fullm::kv8_row;
using fullm::kv8_exp;
using fullm::kv_bit;
using fullm::FAM_GQA;
using fullm::FAM_KDA;
using fullm::FAM_KIMI;
using fullm::FAM_MLA;

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    float uni() { return static_cast<float>((next() >> 40) * (1.0 / 16777216.0)) * 2.0f - 1.0f; }  // [-1, 1)
    float range(float a, float b) { return a + (b - a) * (0.5f * uni() + 0.5f); }
    std::vector<float> vec(size_t n, float scale) {
        std::vector<float> v(n);
        for (float &x : v) x = uni() * scale;
        return v;
    }
};

uint16_t bf16_rne(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x7fffffu)) return static_cast<uint16_t>((u >> 16) | 0x40);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

float bf16f(uint16_t h) {
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// Buffers on one backend, released when the test ends.
struct Arena {
    Backend &be;
    std::vector<void *> ptrs;
    explicit Arena(Backend &b) : be(b) {}
    Arena(const Arena &) = delete;
    Arena &operator=(const Arena &) = delete;
    ~Arena() {
        try {
            be.sync();
        } catch (...) {
        }
        for (void *p : ptrs) {
            try {
                be.release(p);
            } catch (...) {
            }
        }
    }
    template <class T>
    T *up(const std::vector<T> &h) {
        const size_t bytes = h.size() * sizeof(T);
        T *p = static_cast<T *>(be.alloc(bytes ? bytes : 4));
        ptrs.push_back(p);
        if (bytes) be.upload(p, h.data(), bytes);
        return p;
    }
    void *zeros(size_t bytes) {
        void *p = be.alloc(bytes ? bytes : 4);
        ptrs.push_back(p);
        if (bytes) be.zero(p, bytes);
        return p;
    }
    template <class T>
    std::vector<T> down(const T *p, size_t n) {
        std::vector<T> h(n);
        if (n) be.download(h.data(), p, n * sizeof(T));
        be.sync();
        return h;
    }
};

struct Report {
    std::string text;
    int fails = 0;
    void line(const std::string &name, double mabs, double mrel, double tol) {
        const bool ok = std::isfinite(mrel) && mrel <= tol;
        char b[256];
        std::snprintf(b, sizeof b, "%-40s max_abs %.3e max_rel %.3e tol %.0e %s\n", name.c_str(), mabs, mrel, tol,
                      ok ? "PASS" : "FAIL");
        text += b;
        if (!ok) ++fails;
    }
    void error(const std::string &name, const char *msg) {
        text += name + " ERROR " + msg + "\n";
        ++fails;
    }
};

void compare(Report &r, const std::string &name, const std::vector<float> &ref, const std::vector<float> &got,
             double tol) {
    if (ref.size() != got.size()) {
        r.error(name, "output sizes differ");
        return;
    }
    double mag = 0.0, mabs = 0.0;
    bool bad = false;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(ref[i]) || !std::isfinite(got[i])) {
            if (std::isfinite(ref[i]) != std::isfinite(got[i])) bad = true;
            continue;
        }
        mag = std::max(mag, static_cast<double>(std::fabs(ref[i])));
        mabs = std::max(mabs, static_cast<double>(std::fabs(ref[i] - got[i])));
    }
    const double mrel = bad ? INFINITY : mabs / std::max(mag, 1e-30);
    r.line(name, mabs, mrel, tol);
}

using Fn = std::function<std::vector<float>(Backend &)>;

// A case whose fast path the subject does not have (the op returned false: the executor composes
// it instead): skipped, not failed.
struct Absent : std::runtime_error {
    using std::runtime_error::runtime_error;
};

void check(Report &r, Backend &cpu, Backend &gpu, const std::string &name, double tol, const Fn &f) {
    try {
        const std::vector<float> a = f(cpu);
        const std::vector<float> b = f(gpu);
        compare(r, name, a, b, tol);
    } catch (const Absent &e) {
        r.text += name + " skipped: the subject has no " + e.what() + "\n";
    } catch (const std::exception &e) {
        r.error(name, e.what());
    }
}

// What a subject declares (its Caps): the groups and cases it does not are skipped, not failed.
bool has_fam(Backend &b, uint32_t f) { return (b.caps().families & f) != 0; }
// whether the backend keeps an fp8 KV cache (in its 16-bit activation mode, where it has one)
bool has_kv8(Backend &b) {
    b.set_exact_activations(false);
    const bool ok = (b.caps().kv_formats & kv_bit(DType::FP8)) != 0;
    b.set_exact_activations(true);
    return ok;
}
// FlashAttention-3's fp8 cache (plain e4m3 pages), also reported with 16-bit activations only
bool has_fa3_kv8(Backend &b) {
    b.set_exact_activations(false);
    const bool ok = (b.caps().kv_formats & kv_bit(DType::FP8E4M3)) != 0;
    b.set_exact_activations(true);
    return ok;
}

template <class T>
std::vector<float> as_float(const std::vector<T> &v) {
    return std::vector<float>(v.begin(), v.end());
}

std::vector<float> cat(std::vector<float> a, const std::vector<float> &b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

const char *fmt_name(WFmt f) {
    switch (f) {
        case WFmt::F32: return "f32";
        case WFmt::BF16: return "bf16";
        case WFmt::INT8: return "int8";
        case WFmt::FP8: return "fp8";
        case WFmt::FP8B: return "fp8b";
        case WFmt::INT4G: return "int4";
    }
    return "?";
}

// ------------------------------------------------------------------ tests

// A random weight of any format on the host: bytes, plus per-row scales for int8 / fp8.
struct HostWeight {
    WFmt fmt;
    int64_t rows, cols;
    std::vector<uint8_t> data;
    std::vector<float> scale;
    Weight upload(Arena &ar) const {
        Weight w;
        w.fmt = fmt;
        w.rows = rows;
        w.cols = cols;
        w.data = ar.up(data);
        w.scale = scale.empty() ? nullptr : ar.up(scale);
        return w;
    }
};

HostWeight random_weight(WFmt fmt, int64_t rows, int64_t cols, Rng &rng) {
    HostWeight h{fmt, rows, cols, {}, {}};
    std::vector<uint8_t> &w = h.data;
    const size_t ne = static_cast<size_t>(rows * cols);
    switch (fmt) {
        case WFmt::F32: {
            w.resize(ne * 4);
            std::vector<float> f = rng.vec(ne, 1.0f);
            std::memcpy(w.data(), f.data(), w.size());
            break;
        }
        case WFmt::BF16: {
            w.resize(ne * 2);
            for (size_t i = 0; i < ne; ++i) {
                const uint16_t b = bf16_rne(rng.uni());
                std::memcpy(w.data() + 2 * i, &b, 2);
            }
            break;
        }
        case WFmt::INT8:
            w.resize(ne);
            for (size_t i = 0; i < ne; ++i)
                w[i] = static_cast<uint8_t>(static_cast<int8_t>(std::lround(rng.uni() * 127.0f)));
            break;
        case WFmt::FP8:
            w.resize(ne);
            for (size_t i = 0; i < ne; ++i) {
                uint8_t b = static_cast<uint8_t>(rng.next());
                while ((b & 0x7F) == 0x7F) b = static_cast<uint8_t>(rng.next());
                w[i] = b;
            }
            break;
        case WFmt::FP8B:
        case WFmt::INT4G:
            break;  // quantised by the moe cases from bf16 (quantize_fp8_blocks / quantize_int4_groups)
    }
    if (fmt == WFmt::INT8 || fmt == WFmt::FP8) {
        h.scale.resize(static_cast<size_t>(rows));
        for (float &s : h.scale) s = rng.range(0.005f, 0.02f);
    }
    return h;
}

// small: CUDA backends with a 3-block expert grid, the tiles' cross-call prefetch on and off;
// split: the full grid with gate/up and down as two kernels (else one fused kernel)
struct SmallGrid {
    Backend *on = nullptr, *off = nullptr, *split = nullptr;
};

void test_moe_case(Report &r, Backend &cpu, Backend &gpu, int64_t N, int64_t n_exp, int64_t width, int64_t inter,
                   SmallGrid small = {});

// Routed experts held by the backend: a table with some experts resident, each pair's row
// computed or left alone; then the j-order reduce and the row scatter.
void test_moe(Report &r, Backend &cpu, Backend &gpu, SmallGrid small) {
    // 3 tokens over 6 experts (~2 pairs each), and 20 over 3 (~27 each: several 16-row tiles
    // per expert); 64 x 96 runs the CUDA-core kernels, 256 x 256 the tensor-core ones
    test_moe_case(r, cpu, gpu, 3, 6, 64, 96);
    test_moe_case(r, cpu, gpu, 3, 6, 256, 256);
    test_moe_case(r, cpu, gpu, 20, 3, 256, 256);
    // 768 wide: every tile call is >= 3 weight groups (fp8b: exactly 3 at one tile per warp), so
    // the calls chain; 3 tokens give one-tile items, 16 over 3 experts two-tile items of two
    // 16-pair calls. On a 3-block grid each block walks many items (the full grid: about one).
    test_moe_case(r, cpu, gpu, 3, 6, 768, 768, small);
    test_moe_case(r, cpu, gpu, 16, 3, 768, 768, small);
    // above 64 tokens int4 runs the int4 GEMM (int4_gemm.cu): ~350 pairs per expert, so full
    // 128-pair passes and a partial one each; 768 wide: six scale groups, three 256-row tiles of down
    test_moe_case(r, cpu, gpu, 520, 6, 256, 256);
    test_moe_case(r, cpu, gpu, 520, 6, 768, 768);
    // ~21 and ~53 pairs per expert: the GEMM's 32- and 64-token tiles (520 x 6: its 128-token ones)
    test_moe_case(r, cpu, gpu, 80, 16, 256, 256);
    test_moe_case(r, cpu, gpu, 200, 16, 256, 256);
}

void test_moe_case(Report &r, Backend &cpu, Backend &gpu, int64_t N, int64_t n_exp, int64_t width, int64_t inter,
                   SmallGrid small) {
    const int64_t k = 4;
    Rng rng(4242);
    const auto x = rng.vec(static_cast<size_t>(N * width), 1.0f);
    const auto wts = rng.vec(static_cast<size_t>(N * k), 0.5f);
    std::vector<int32_t> idx(static_cast<size_t>(N * k));
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int32_t>((i * 5 + 1) % n_exp);
    for (WFmt fmt : {WFmt::BF16, WFmt::INT8, WFmt::FP8B, WFmt::INT4G}) {
        // fp8b: the tensor-core tiles and the blockwise GEMM only (rows of whole 256-byte groups);
        // int4: the tiles and the int4 GEMM (widths of whole 256 columns), always with fp16
        // activations
        if ((fmt == WFmt::FP8B || fmt == WFmt::INT4G) && (width % 256 || inter % 256)) continue;
        const bool q16 = fmt == WFmt::BF16 || fmt == WFmt::FP8B || fmt == WFmt::INT4G;  // host data bf16
        for (bool silu : {true, false}) {
            if (!silu && !has_fam(gpu, FAM_KIMI)) continue;  // SiTU-GLU is Kimi's
            const std::string name = std::string("moe_experts/") + fmt_name(fmt) + (silu ? "/silu" : "/situ") + "/N" +
                                     std::to_string(N) + "/" + std::to_string(width) + "x" + std::to_string(inter);
            // host weights: bytes (bf16 or int8) and row scales (int8), per expert and matrix
            struct HostMat {
                std::vector<uint8_t> data;
                std::vector<float> scale;
            };
            std::vector<std::array<HostMat, 3>> hm(static_cast<size_t>(n_exp));
            for (int64_t e = 0; e < n_exp; ++e) {
                Rng er(100 + static_cast<int>(e));
                const int64_t shape[3][2] = {{inter, width}, {inter, width}, {width, inter}};
                for (int m = 0; m < 3; ++m) {
                    const size_t n = static_cast<size_t>(shape[m][0] * shape[m][1]);
                    HostMat &h = hm[static_cast<size_t>(e)][m];
                    if (q16) {  // fp8b / int4: bf16 here, quantised at upload
                        h.data.resize(2 * n);
                        for (size_t i = 0; i < n; ++i) {
                            const uint16_t v = bf16_rne(er.uni() * 0.2f);
                            std::memcpy(h.data.data() + 2 * i, &v, 2);
                        }
                    } else {
                        h.data.resize(n);
                        for (auto &v : h.data) v = static_cast<uint8_t>(static_cast<int8_t>(static_cast<int>(er.uni() * 120.0f)));
                        h.scale = er.vec(static_cast<size_t>(shape[m][0]), 0.002f);
                    }
                }
            }
            // routed: the picks come from the router (CUDA: the fused router), else the fixed idx / wts
            const HostWeight rw = random_weight(WFmt::BF16, n_exp, width, rng);
            // tokens [t0, t0 + n), in reverse order when rev (rows put back in order); routed:
            // picks from the router; bx: bf16 activations
            // fused: the router also builds the expert call's routing lists (else the route kernel)
            auto run = [&](Backend &be, bool routed, bool bx, int64_t t0, int64_t n, bool rev, bool fused) {
                Arena ar(be);
                auto tok = [&](int64_t i) { return rev ? t0 + n - 1 - i : t0 + i; };
                const bool il = be.moe_gate_up_interleaved(fmt, width, inter);
                const size_t row = static_cast<size_t>(width) * (q16 ? 2 : 1);
                auto up = [&](const std::vector<uint8_t> &d, const std::vector<float> &sc, int64_t rows, int64_t cols) {
                    Weight w;
                    w.fmt = fmt;
                    w.rows = rows;
                    w.cols = cols;
                    if (fmt == WFmt::INT4G) {  // as the executor stores them: in the backend's packing
                        std::vector<uint8_t> q(static_cast<size_t>(int4g_bytes(rows, cols)));
                        std::vector<float> s4(static_cast<size_t>(int4g_scales(rows, cols)));
                        w.layout = be.weight_layout(fmt, rows, cols);
                        quantize_int4_groups(reinterpret_cast<const uint16_t *>(d.data()), rows, cols, q.data(), s4.data(),
                                             w.layout);
                        w.data = ar.up(q);
                        w.scale = ar.up(s4);
                        return w;
                    }
                    if (fmt == WFmt::FP8B) {  // as the executor stores them: blocks over the stored matrix
                        std::vector<uint8_t> q(static_cast<size_t>(rows * cols));
                        std::vector<float> s8(static_cast<size_t>(fp8b_scales(rows, cols)));
                        quantize_fp8_blocks(reinterpret_cast<const uint16_t *>(d.data()), rows, cols, q.data(), s8.data());
                        w.data = ar.up(q);
                        w.scale = ar.up(s8);
                        return w;
                    }
                    w.data = ar.up(d);
                    if (!sc.empty()) w.scale = ar.up(sc);
                    return w;
                };
                std::vector<Backend::ExpertW> table(static_cast<size_t>(n_exp));
                for (int64_t e = 0; e < n_exp; ++e) {
                    if (e == 2) continue;  // not resident: its pairs must stay untouched
                    const auto &h = hm[static_cast<size_t>(e)];
                    if (il) {
                        std::vector<uint8_t> d(2 * h[0].data.size());
                        std::vector<float> sc(h[0].scale.empty() ? 0 : 2 * h[0].scale.size());
                        for (int64_t q = 0; q < inter; ++q) {
                            std::memcpy(d.data() + (2 * q) * row, h[0].data.data() + q * row, row);
                            std::memcpy(d.data() + (2 * q + 1) * row, h[1].data.data() + q * row, row);
                            if (!sc.empty()) sc[2 * q] = h[0].scale[q], sc[2 * q + 1] = h[1].scale[q];
                        }
                        table[static_cast<size_t>(e)] =
                            Backend::ExpertW{up(d, sc, 2 * inter, width), Weight{}, up(h[2].data, h[2].scale, width, inter)};
                    } else {
                        table[static_cast<size_t>(e)] =
                            Backend::ExpertW{up(h[0].data, h[0].scale, inter, width), up(h[1].data, h[1].scale, inter, width),
                                             up(h[2].data, h[2].scale, width, inter)};
                    }
                }
                const Backend::ExpertW *td = ar.up(table);
                float *contrib = ar.up(std::vector<float>(static_cast<size_t>(n * k * width), -7.0f));
                float *scratch = ar.up(std::vector<float>(static_cast<size_t>(n * k * inter), 0.0f));
                std::vector<float> xs, ws;
                std::vector<int32_t> is;
                for (int64_t i = 0; i < n; ++i) {
                    xs.insert(xs.end(), x.begin() + tok(i) * width, x.begin() + (tok(i) + 1) * width);
                    is.insert(is.end(), idx.begin() + tok(i) * k, idx.begin() + (tok(i) + 1) * k);
                    ws.insert(ws.end(), wts.begin() + tok(i) * k, wts.begin() + (tok(i) + 1) * k);
                }
                const float *xd = ar.up(xs);
                const int32_t *idd = ar.up(is);
                const float *wd = ar.up(ws);
                if (routed) {
                    const Weight rwd = rw.upload(ar);
                    auto *id = static_cast<int32_t *>(ar.zeros(sizeof(int32_t) * n * k));
                    auto *wq = static_cast<float *>(ar.zeros(sizeof(float) * n * k));
                    if (!be.router_softmax_fused(id, wq, rwd, xd, n, k, true, fused ? td : nullptr, fused ? n_exp : 0)) {
                        auto *lg = static_cast<float *>(ar.zeros(sizeof(float) * n * n_exp));
                        be.gemm(rwd, xd, n, lg, n_exp);
                        be.router_softmax_topk(id, wq, lg, n, n_exp, k, true);
                    }
                    idd = id, wd = wq;
                }
                be.set_exact_activations(!bx);
                be.moe_experts(contrib, xd, idd, wd, n, k, td, n_exp, fmt, width, inter, silu, 4.0f, 25.0f, scratch);
                be.set_exact_activations(true);
                be.moe_materialize(contrib);  // rows a backend left in its own order
                const std::vector<float> got = ar.down(contrib, static_cast<size_t>(n * k * width));
                if (!rev) return got;
                std::vector<float> ord(got.size());
                for (int64_t i = 0; i < n; ++i)  // token tok(i) was row i
                    std::copy(got.begin() + i * k * width, got.begin() + (i + 1) * k * width,
                              ord.begin() + (tok(i) - t0) * k * width);
                return ord;
            };
            // exact activations: the rounding grows as the square root of the dot length (bf16 at
            // 768 wide: 9.6e-6 against the CPU, 1.2e-5 with the router's picks)
            // (int4: its tiles round x to fp16 even here)
            const double tol_exact = fmt == WFmt::INT4G
                                         ? 1e-2
                                         : 1e-5 * std::sqrt(std::max(1.0, static_cast<double>(std::max(width, inter)) / 256.0));
            for (bool routed : {false, true}) {
                // the router's picks against the CPU's: a few tokens of hundreds meet a near-tie that
                // the GPU's and the CPU's logit sums break differently (any format), so only small N
                if (routed && (n_exp < k || N > 32)) continue;
                check(r, cpu, gpu, routed ? name + "/routed" : name, tol_exact,
                      [&](Backend &be) { return run(be, routed, false, 0, N, false, true); });
            }
            if (n_exp >= k) {  // the lists from the router's last block: bitwise the route kernel's
                const std::string nf = name + "/routed (bitwise fused lists vs route kernel)";
                try {
                    compare(r, nf, run(gpu, true, false, 0, N, false, true), run(gpu, true, false, 0, N, false, false), 0.0);
                    if (fmt == WFmt::BF16)
                        compare(r, nf + "/bf16act", run(gpu, true, true, 0, N, false, true),
                                run(gpu, true, true, 0, N, false, false), 0.0);
                } catch (const std::exception &e) {
                    r.error(nf, e.what());
                }
            }
            // bf16 activations. Above 16 tokens the prefill GEMM computes the pairs (its own k
            // order, as cuBLAS has for the trunk): against the exact CPU, and a pair's rows must
            // not depend on where in the batch its token sits. Up to 16: the decode tile,
            // bitwise what each token gets alone.
            if (q16) {
                const std::string nb = name + "/bf16act";
                // fp8b above 16 rows: the blockwise GEMM quantises the activations to e4m3 too
                const double tol = fmt == WFmt::FP8B && N > 16 ? 6e-2 : 1e-2;
                try {
                    compare(r, nb, run(cpu, false, false, 0, N, false, true), run(gpu, false, true, 0, N, false, true), tol);
                    if (N > 16) {
                        compare(r, nb + " (bitwise reversed vs in order)", run(gpu, false, true, 0, N, true, true),
                                run(gpu, false, true, 0, N, false, true), 0.0);
                    } else {
                        std::vector<float> alone;
                        for (int64_t t0 = 0; t0 < N; ++t0) alone = cat(alone, run(gpu, false, true, t0, 1, false, true));
                        compare(r, nb + " (bitwise alone vs batched)", alone, run(gpu, false, true, 0, N, false, true), 0.0);
                    }
                } catch (const std::exception &e) {
                    r.error(nb, e.what());
                }
            }
            // the tiles' chain: blocks walking many items, with and without the prefetch, give
            // the full grid's bits; so do the two kernels the fused one replaces (the decode
            // tiles only: N <= 16)
            if (small.on && small.off && small.split && N <= 16) {
                for (bool bx : {false, true}) {
                    if (bx && !q16) continue;
                    const std::string ns = name + (bx ? "/bf16act" : "") + " (bitwise 3 blocks";
                    try {
                        const std::vector<float> full = run(gpu, false, bx, 0, N, false, true);
                        compare(r, ns + ", prefetch vs full grid)", run(*small.on, false, bx, 0, N, false, true), full, 0.0);
                        compare(r, ns + ", no prefetch vs full grid)", run(*small.off, false, bx, 0, N, false, true), full,
                                0.0);
                        compare(r, name + (bx ? "/bf16act" : "") + " (bitwise fused vs two kernels)",
                                full, run(*small.split, false, bx, 0, N, false, true), 0.0);
                    } catch (const std::exception &e) {
                        r.error(ns + ")", e.what());
                    }
                }
            }
        }
    }
    const auto c = rng.vec(static_cast<size_t>(N * k * width), 1.0f);
    check(r, cpu, gpu, "moe_reduce", 1e-6, [&](Backend &be) {
        Arena ar(be);
        float *acc = ar.up(std::vector<float>(static_cast<size_t>(N * width), 0.0f));
        be.moe_reduce(acc, ar.up(c), N, k, width);
        return ar.down(acc, static_cast<size_t>(N * width));
    });
    const auto addend = rng.vec(static_cast<size_t>(N * width), 1.0f);
    check(r, cpu, gpu, "moe_reduce/addend", 0.0, [&](Backend &be) {
        Arena ar(be);
        float *acc = ar.up(std::vector<float>(static_cast<size_t>(N * width), 0.0f));
        be.moe_reduce(acc, ar.up(c), N, k, width, ar.up(addend));
        return ar.down(acc, static_cast<size_t>(N * width));
    });
    const auto hh = rng.vec(static_cast<size_t>(N * width), 1.0f), nw = rng.vec(static_cast<size_t>(width), 1.0f);
    for (bool add : {false, true}) {
        auto run = [&](Backend &be, bool fused) {
            Arena ar(be);
            float *h = ar.up(hh);
            float *y = static_cast<float *>(ar.zeros(sizeof(float) * N * width));
            float *tmp = static_cast<float *>(ar.zeros(sizeof(float) * N * width));
            const float *ad = add ? ar.up(addend) : nullptr, *cd = ar.up(c), *wd = ar.up(nw);
            if (fused) {
                be.moe_reduce_add_rmsnorm(h, cd, N, k, width, ad, y, wd, 1e-6f, tmp);
            } else {
                be.moe_reduce(tmp, cd, N, k, width, ad);
                be.add_rmsnorm(h, tmp, y, wd, N, width, 1e-6f);
            }
            return cat(ar.down(h, static_cast<size_t>(N * width)), ar.down(y, static_cast<size_t>(N * width)));
        };
        const std::string name = add ? "moe_reduce_add_rmsnorm/addend" : "moe_reduce_add_rmsnorm";
        check(r, cpu, gpu, name, 1e-5, [&](Backend &be) { return run(be, true); });
        try {
            compare(r, name + " (bitwise vs reduce, add_rmsnorm)", run(gpu, false), run(gpu, true), 0.0);
        } catch (const std::exception &e) {
            r.error(name, e.what());
        }
    }
    check(r, cpu, gpu, "gather_ids", 0.0, [&](Backend &be) {
        Arena ar(be);
        std::vector<int32_t> table(300);
        for (int32_t i = 0; i < 300; ++i) table[size_t(i)] = 1000 + 7 * i;
        const std::vector<int32_t> idx = {299, 0, 17, 17, 150};
        auto *dst = static_cast<int32_t *>(ar.zeros(sizeof(int32_t) * idx.size()));
        be.gather_ids(dst, ar.up(table), ar.up(idx), int64_t(idx.size()));
        return as_float(ar.down(dst, idx.size()));
    });
    // route counts on the device (expert placement), against a count on the host where the
    // backend has none (the CPU)
    check(r, cpu, gpu, "count_routes", 0.0, [&](Backend &be) {
        Arena ar(be);
        const std::vector<int32_t> idx = {3, 7, 3, -1, 0, 7, 7, 300, 2, 3};
        std::vector<float> want(8, 0.0f);
        for (int32_t e : idx)
            if (e >= 0 && e < 8) want[size_t(e)] += 1.0f;
        auto *cnt = static_cast<uint32_t *>(ar.zeros(8 * 4));
        if (!be.count_routes(cnt, ar.up(idx), int64_t(idx.size()), 8)) return want;
        be.count_routes(cnt, ar.up(idx), int64_t(idx.size()), 8);  // twice: counts add up
        const auto got = ar.down(cnt, 8);
        std::vector<float> out(8);
        for (size_t i = 0; i < 8; ++i) out[i] = float(got[i]) / 2.0f;
        return out;
    });
    check(r, cpu, gpu, "scatter_rows", 0.0, [&](Backend &be) {
        Arena ar(be);
        float *dst = ar.up(std::vector<float>(static_cast<size_t>(N * k * width), 0.0f));
        const std::vector<int32_t> rows = {5, 0, 9};
        be.scatter_rows(dst, ar.up(std::vector<float>(c.begin(), c.begin() + 3 * width)), ar.up(rows), 3, width);
        return ar.down(dst, static_cast<size_t>(N * k * width));
    });
}


// Projections of shared and separate inputs, mixed formats and shapes, as one group: equal to
// the CPU, and bit-identical to the same gemms issued one by one.
void test_gemm_group(Report &r, Backend &cpu, Backend &gpu) {
    Rng rng(2024);
    struct J {
        WFmt fmt;
        int64_t rows, cols, ldy;
        int x;  // which input
    };
    const J js[] = {{WFmt::F32, 300, 256, 300, 0}, {WFmt::BF16, 1000, 256, 1003, 0}, {WFmt::INT8, 37, 512, 40, 1},
                    {WFmt::FP8, 129, 256, 129, 0}, {WFmt::BF16, 64, 128, 64, 2}};
    const int nj = 5;
    std::vector<HostWeight> ws;
    for (const J &j : js) ws.push_back(random_weight(j.fmt, j.rows, j.cols, rng));
    const int64_t xk[3] = {256, 512, 128};
    for (int64_t M : {1, 5, 16}) {
        std::vector<std::vector<float>> xs;
        for (int64_t k : xk) xs.push_back(rng.vec(static_cast<size_t>(M * k), 1.0f));
        auto run = [&](Backend &be, bool grouped) {
            Arena ar(be);
            std::vector<Weight> w(nj);
            std::vector<float *> y(nj);
            const float *xd[3];
            for (int i = 0; i < 3; ++i) xd[i] = ar.up(xs[i]);
            std::vector<Backend::GemmJob> jobs;
            for (int i = 0; i < nj; ++i) {
                w[i] = ws[i].upload(ar);
                y[i] = ar.up(std::vector<float>(static_cast<size_t>(M * js[i].ldy), 7.0f));
                jobs.push_back({&w[i], xd[js[i].x], y[i], js[i].ldy});
            }
            if (grouped) be.gemm_group(jobs.data(), nj, M);
            else
                for (const auto &j : jobs) be.gemm(*j.w, j.x, M, j.y, j.ldy);
            std::vector<float> out;
            for (int i = 0; i < nj; ++i) out = cat(out, ar.down(y[i], static_cast<size_t>(M * js[i].ldy)));
            return out;
        };
        const std::string name = "gemm_group/M" + std::to_string(M);
        check(r, cpu, gpu, name, 1e-5, [&](Backend &be) { return run(be, true); });
        try {
            compare(r, name + " (bitwise vs separate)", run(gpu, false), run(gpu, true), 0.0);
        } catch (const std::exception &e) {
            r.error(name, e.what());
        }
    }
}

// bf16 activations on the GPU (x rounded to bf16 for the tensor cores): close to the exact CPU at
// 16 rows, batch-invariant on the GEMV (each row alone == the row in a 16-row call, bit for bit;
// 7 rows for bf16 and int8 weights, which cuBLAS takes from FULLM_BLAS_MIN_ROWS, 8) and still
// identical grouped or separate.
// fp8 too: batched (16 rows) the fp8 GEMVs take x as one fp16 copy (xh_rows), alone (1 row) the
// tile rounds it itself; bitwise equal
void test_bf16_activations(Report &r, Backend &cpu, Backend &gpu) {
    gpu.set_exact_activations(false);
    Rng rng(4321);
    for (WFmt fmt : {WFmt::BF16, WFmt::INT8, WFmt::FP8}) {
        const HostWeight hw = random_weight(fmt, 1000, 512, rng);
        const int64_t M = 16;
        const auto x = rng.vec(static_cast<size_t>(M * 512), 1.0f);
        auto run = [&](Backend &be, int64_t m0, int64_t m) {
            Arena ar(be);
            const Weight w = hw.upload(ar);
            float *y = static_cast<float *>(ar.zeros(sizeof(float) * m * 1000));
            be.gemm(w, ar.up(std::vector<float>(x.begin() + m0 * 512, x.begin() + (m0 + m) * 512)), m, y, 1000);
            return ar.down(y, static_cast<size_t>(m * 1000));
        };
        const std::string name = std::string("bf16 activations/gemm/") + fmt_name(fmt);
        try {
            compare(r, name, run(cpu, 0, M), run(gpu, 0, M), 1e-2);
            const int64_t Mg = fmt == WFmt::FP8 ? M : 7;  // the GEMV's rows (cuBLAS takes bf16 / int8 from 8)
            std::vector<float> rows;
            for (int64_t m = 0; m < Mg; ++m) rows = cat(rows, run(gpu, m, 1));
            compare(r, name + " (bitwise rows alone vs batched)", rows, run(gpu, 0, Mg), 0.0);
        } catch (const std::exception &e) {
            r.error(name, e.what());
        }
    }
    {  // q/k/v-like: three fp8 jobs on one x share its fp16 copy
        const int64_t M = 16, K = 512;
        const int64_t rows[3] = {1024, 256, 256};
        std::vector<HostWeight> hw;
        for (int64_t n : rows) hw.push_back(random_weight(WFmt::FP8, n, K, rng));
        const auto x = rng.vec(static_cast<size_t>(M * K), 1.0f);
        auto run = [&](Backend &be, int64_t m0, int64_t m) {
            Arena ar(be);
            const float *xd = ar.up(std::vector<float>(x.begin() + m0 * K, x.begin() + (m0 + m) * K));
            Weight w[3];
            float *y[3];
            Backend::GemmJob jobs[3];
            for (int i = 0; i < 3; ++i) {
                w[i] = hw[static_cast<size_t>(i)].upload(ar);
                y[i] = static_cast<float *>(ar.zeros(sizeof(float) * m * rows[i]));
                jobs[i] = Backend::GemmJob{&w[i], xd, y[i], rows[i]};
            }
            be.gemm_group(jobs, 3, m);
            std::vector<float> out;
            for (int i = 0; i < 3; ++i) out = cat(out, ar.down(y[i], static_cast<size_t>(m * rows[i])));
            return out;
        };
        const std::string name = "bf16 activations/gemm_group/fp8";
        try {
            compare(r, name, run(cpu, 0, M), run(gpu, 0, M), 1e-2);
            std::vector<float> alone;
            for (int64_t m = 0; m < M; ++m) alone = cat(alone, run(gpu, m, 1));
            // rows alone come out job by job: reorder the batched result to match
            const std::vector<float> b = run(gpu, 0, M);
            std::vector<float> bo;
            for (int64_t m = 0; m < M; ++m) {
                int64_t off = 0;
                for (int i = 0; i < 3; ++i) {
                    bo.insert(bo.end(), b.begin() + off + m * rows[i], b.begin() + off + (m + 1) * rows[i]);
                    off += M * rows[i];
                }
            }
            compare(r, name + " (bitwise rows alone vs batched)", alone, bo, 0.0);
        } catch (const std::exception &e) {
            r.error(name, e.what());
        }
    }
    gpu.set_exact_activations(true);
}

// fp8 weights above the GEMV's rows with bf16 activations: with Caps::fp8_w8a8 the CUDA backend's
// cuBLASLt e4m3 GEMM where it takes them (matrices of >= 16M elements, or steps of >= 512 rows),
// x's rows quantized to e4m3 with a scale each (amax / 448), checked closely against the exact CPU
// on x rounded the same way (the scales and layouts) and loosely against exact x; otherwise the
// weights' bf16 copies (bf16 x). A 4096 x 4096 matrix at 40 rows singly (fp32 out) and as gemm16
// (bf16 out); a back-to-back q/k/v-like stack, scales back to back too, in one gemm_group at 600
// rows (one GEMM, bf16 out) and at 40 (the stack's bf16 copy).
void test_fp8_gemm(Report &r, Backend &cpu, Backend &gpu) {
    gpu.set_exact_activations(false);
    const bool w8a8 = gpu.caps().fp8_w8a8;
    Rng rng(5150);
    // x as the GPU rounds it for these rows: e4m3 with a scale a row (W8A8), or bf16
    auto rounded = [&](const std::vector<float> &x, int64_t M, int64_t K, bool e4m3) {
        std::vector<float> xq(x.size());
        for (int64_t m = 0; m < M; ++m) {
            float a = 0.0f;
            for (int64_t k = 0; k < K; ++k) a = std::max(a, std::fabs(x[static_cast<size_t>(m * K + k)]));
            const float s = std::max(a, 1e-30f) / 448.0f, inv = 1.0f / s;
            for (int64_t k = 0; k < K; ++k) {
                const size_t i = static_cast<size_t>(m * K + k);
                xq[i] = e4m3 ? fullm::e4m3_to_f32(fullm::f32_to_e4m3(x[i] * inv)) * s : bf16f(bf16_rne(x[i]));
            }
        }
        return xq;
    };
    {  // one large matrix at decode-sized steps
        const int64_t M = 40, N = 4096, K = 4096;
        const HostWeight hw = random_weight(WFmt::FP8, N, K, rng);
        const auto x = rng.vec(static_cast<size_t>(M * K), 1.0f);
        const auto xq = rounded(x, M, K, w8a8);
        auto single = [&](Backend &be, const std::vector<float> &xin, bool g16) {
            Arena ar(be);
            const Weight w = hw.upload(ar);
            auto *y = static_cast<float *>(ar.zeros(sizeof(float) * M * N));
            if (g16) be.gemm16(w, ar.up(xin), M, y, N);
            else be.gemm(w, ar.up(xin), M, y, N);
            return ar.down(y, static_cast<size_t>(M * N));
        };
        const std::string m = std::string(w8a8 ? " (W8A8)" : " (bf16 copies)") + " 4096x4096 M=40";
        try {
            // (Hopper's e4m3 MMAs keep fewer accumulator bits than fp32: ~4e-4 of the output scale)
            compare(r, "fp8 gemm" + m, single(cpu, xq, false), single(gpu, x, false), w8a8 ? 1e-3 : 1e-2);
            compare(r, "fp8 gemm16, bf16 out" + m, single(cpu, xq, true), single(gpu, x, true), 1e-2);
            compare(r, "fp8 gemm vs exact x" + m, single(cpu, x, false), single(gpu, x, false), 6e-2);
        } catch (const std::exception &e) {
            r.error("fp8 gemm" + m, e.what());
        }
    }
    const int64_t K = 512, rows[3] = {1024, 256, 256};
    std::vector<HostWeight> hw;
    for (int64_t n : rows) hw.push_back(random_weight(WFmt::FP8, n, K, rng));
    // small matrices: W8A8 at prefill sizes, the copy below (unless the backend quantizes there too)
    for (int64_t M : {int64_t{600}, int64_t{40}}) {
        const bool e4m3 = w8a8 && (M >= 512 || gpu.caps().fp8_w8a8_small);
        const auto x = rng.vec(static_cast<size_t>(M * K), 1.0f);
        const auto xq = rounded(x, M, K, e4m3);
        auto stacked = [&](Backend &be, const std::vector<float> &xin) {
            Arena ar(be);
            size_t tb = 0, ts = 0;
            for (const auto &h : hw) tb += h.data.size(), ts += h.scale.size();
            auto *blk = static_cast<uint8_t *>(ar.zeros(tb));
            auto *sblk = static_cast<float *>(ar.zeros(sizeof(float) * ts));
            const float *xd = ar.up(xin);
            Weight w[3];
            float *y[3];
            Backend::GemmJob jobs[3];
            size_t ob = 0, os = 0;
            for (int i = 0; i < 3; ++i) {
                be.upload(blk + ob, hw[i].data.data(), hw[i].data.size());
                be.upload(sblk + os, hw[i].scale.data(), sizeof(float) * hw[i].scale.size());
                w[i].fmt = WFmt::FP8, w[i].rows = rows[i], w[i].cols = K, w[i].data = blk + ob, w[i].scale = sblk + os;
                ob += hw[i].data.size(), os += hw[i].scale.size();
                y[i] = static_cast<float *>(ar.zeros(sizeof(float) * M * rows[i]));
                jobs[i] = Backend::GemmJob{&w[i], xd, y[i], rows[i], true};
            }
            be.gemm_group(jobs, 3, M);
            std::vector<float> out;
            for (int i = 0; i < 3; ++i) out = cat(out, ar.down(y[i], static_cast<size_t>(M * rows[i])));
            return out;
        };
        const std::string m = std::string(e4m3 ? " (W8A8)" : " (bf16 copy)") + " M=" + std::to_string(M);
        try {
            compare(r, "fp8 gemm_group, stacked" + m, stacked(cpu, xq), stacked(gpu, x), 1e-2);
        } catch (const std::exception &e) {
            r.error("fp8 gemm_group" + m, e.what());
        }
    }
    gpu.set_exact_activations(true);
}

// A dense MLP split into n int4 experts (dense.rs) on a big step through Backend::split_mlp, where
// the backend has it: against the exact CPU on the same roundings (x and the GLU as e4m3 with a
// scale per row per 128 columns, the gate / up result as bf16: the scales' layouts and the
// experts' order), and loosely against the exact MLP of the int4 weights.
void test_split_mlp(Report &r, Backend &gpu) {
    gpu.set_exact_activations(false);
    Rng rng(777);
    const int64_t n = 2, E = 256, I = 256, N = 600, R = 2 * I, J = n * I;
    struct Host {
        std::vector<uint8_t> q;
        std::vector<float> s;
        int32_t layout;
    };
    auto quant = [&](int64_t rows, int64_t cols) {
        std::vector<uint16_t> w(static_cast<size_t>(rows * cols));
        for (auto &v : w) v = bf16_rne(0.05f * rng.uni());
        Host h{std::vector<uint8_t>(static_cast<size_t>(int4g_bytes(rows, cols))),
               std::vector<float>(static_cast<size_t>(int4g_scales(rows, cols))), gpu.weight_layout(WFmt::INT4G, rows, cols)};
        quantize_int4_groups(w.data(), rows, cols, h.q.data(), h.s.data(), h.layout);
        return h;
    };
    std::vector<Host> w1, w2;
    for (int64_t e = 0; e < n; ++e) w1.push_back(quant(R, E)), w2.push_back(quant(E, I));
    auto deq = [](const Host &h, int64_t rows, int64_t cols, int64_t rr, int64_t c) {
        return static_cast<double>(fullm::int4g_value(h.q.data(), rows, cols, rr, c, h.layout)) * h.s[static_cast<size_t>((c / 128) * rows + rr)];
    };
    const auto x = rng.vec(static_cast<size_t>(N * E), 1.0f);
    auto e4 = [](const float *v, int64_t len, float *out) {  // per 128: amax / 448, e4m3, back
        for (int64_t g = 0; g < len / 128; ++g) {
            float a = 0.0f;
            for (int64_t i = 0; i < 128; ++i) a = std::max(a, std::fabs(v[g * 128 + i]));
            const float s = std::max(a, 1e-30f) / 448.0f, inv = 1.0f / s;
            for (int64_t i = 0; i < 128; ++i) out[g * 128 + i] = fullm::e4m3_to_f32(fullm::f32_to_e4m3(v[g * 128 + i] * inv)) * s;
        }
    };
    auto reference = [&](bool rounded) {
        std::vector<float> out(static_cast<size_t>(N * E)), xr(static_cast<size_t>(E)), a(static_cast<size_t>(J)), ar(static_cast<size_t>(J));
        for (int64_t m = 0; m < N; ++m) {
            const float *xm = x.data() + m * E;
            if (rounded) e4(xm, E, xr.data());
            else std::copy(xm, xm + E, xr.begin());
            for (int64_t e = 0; e < n; ++e)
                for (int64_t i = 0; i < I; ++i) {
                    double g = 0.0, u = 0.0;
                    for (int64_t k = 0; k < E; ++k) g += deq(w1[e], R, E, 2 * i, k) * xr[k], u += deq(w1[e], R, E, 2 * i + 1, k) * xr[k];
                    const float gf = rounded ? bf16f(bf16_rne(static_cast<float>(g))) : static_cast<float>(g);
                    const float uf = rounded ? bf16f(bf16_rne(static_cast<float>(u))) : static_cast<float>(u);
                    a[static_cast<size_t>(e * I + i)] = gf * (1.0f / (1.0f + std::exp(-gf))) * uf;
                }
            if (rounded) e4(a.data(), J, ar.data());
            else ar = a;
            for (int64_t o = 0; o < E; ++o) {
                double acc = 0.0;
                for (int64_t e = 0; e < n; ++e)
                    for (int64_t i = 0; i < I; ++i) acc += deq(w2[e], E, I, o, i) * ar[static_cast<size_t>(e * I + i)];
                out[static_cast<size_t>(m * E + o)] = static_cast<float>(acc);
            }
        }
        return out;
    };
    const std::string name = "split_mlp (int4 experts, big step)";
    try {
        Arena ar(gpu);
        std::vector<Backend::ExpertW> table(static_cast<size_t>(n));
        for (int64_t e = 0; e < n; ++e) {
            auto up = [&](const Host &h, int64_t rows, int64_t cols) {
                Weight w;
                w.fmt = WFmt::INT4G, w.rows = rows, w.cols = cols, w.layout = h.layout;
                w.data = ar.up(h.q), w.scale = ar.up(h.s);
                return w;
            };
            table[static_cast<size_t>(e)] = Backend::ExpertW{up(w1[static_cast<size_t>(e)], R, E), Weight{}, up(w2[static_cast<size_t>(e)], E, I)};
        }
        auto *out = static_cast<float *>(ar.zeros(sizeof(float) * N * E));
        if (!gpu.split_mlp(out, ar.up(x), N, table.data(), n, WFmt::INT4G, E, I)) {
            r.text += name + " SKIPPED (not on this backend)\n";
        } else {
            const auto got = ar.down(out, static_cast<size_t>(N * E));
            // (a GLU value near an e4m3 rounding boundary may land on the other side of it, the
            // GEMM's sums not those of the CPU: one input off by a step; ~1.2e-2 of the output scale here)
            compare(r, name, reference(true), got, 3e-2);
            compare(r, name + " vs exact", reference(false), got, 6e-2);
        }
    } catch (const std::exception &e) {
        r.error(name, e.what());
    }
    gpu.set_exact_activations(true);
}

// A dense MLP with bf16 activations on the GPU, as dense_exec runs it (gate and up y16_ok, then
// silu_glu, then down): bf16 gate and up back to back in one allocation (cuBLAS sizes: one GEMM,
// its bf16 result read by silu_glu, whose bf16-only output the down GEMM takes), or int8 (their
// bf16 twins above 16 rows). Close to the exact CPU at GEMV, small cuBLAS and prefill sizes, the
// SiLU-GLU output's fp32 rows included (downloaded: materialized from the twin).
void test_mlp_bf16(Report &r, Backend &cpu, Backend &gpu) {
    gpu.set_exact_activations(false);
    Rng rng(9876);
    const int64_t E = 256, I = 384;
    for (WFmt fmt : {WFmt::BF16, WFmt::INT8}) {
        const HostWeight hg = random_weight(fmt, I, E, rng), hu = random_weight(fmt, I, E, rng),
                         hd = random_weight(fmt, E, I, rng);
        for (int64_t M : {4, 12, 40}) {
            const auto x = rng.vec(static_cast<size_t>(M * E), 1.0f);
            auto run = [&](Backend &be) {
                Arena ar(be);
                Weight g = hg.upload(ar), u = hu.upload(ar);
                const Weight d = hd.upload(ar);
                if (fmt == WFmt::BF16) {  // back to back
                    const size_t b = hg.data.size();
                    auto *both = static_cast<uint8_t *>(ar.zeros(2 * b));
                    be.upload(both, hg.data.data(), b);
                    be.upload(both + b, hu.data.data(), b);
                    g.data = both, u.data = both + b;
                }
                const float *xd = ar.up(x);
                auto *gy = static_cast<float *>(ar.zeros(sizeof(float) * M * I));
                auto *uy = static_cast<float *>(ar.zeros(sizeof(float) * M * I));
                auto *a = static_cast<float *>(ar.zeros(sizeof(float) * M * I));
                auto *y = static_cast<float *>(ar.zeros(sizeof(float) * M * E));
                const Backend::GemmJob jobs[] = {{&g, xd, gy, I, true}, {&u, xd, uy, I, true}};
                be.gemm_group(jobs, 2, M);
                be.silu_glu(a, gy, uy, M * I);
                be.gemm(d, a, M, y, E);
                std::vector<float> out = ar.down(y, static_cast<size_t>(M * E));
                const std::vector<float> av = ar.down(a, static_cast<size_t>(M * I));
                out.insert(out.end(), av.begin(), av.end());
                return out;
            };
            const std::string name = std::string("bf16 activations/mlp/") + fmt_name(fmt) + " M=" + std::to_string(M);
            try {
                compare(r, name, run(cpu), run(gpu), 2e-2);
            } catch (const std::exception &e) {
                r.error(name, e.what());
            }
        }
    }
    gpu.set_exact_activations(true);
}

void test_gemm(Report &r, Backend &cpu, Backend &gpu) {
    const int64_t N = 37, ldy = N + 3;
    for (WFmt fmt : {WFmt::F32, WFmt::BF16, WFmt::INT8, WFmt::FP8}) {
        for (int64_t K : {256, 200}) {
            Rng rng(1000 + static_cast<int>(fmt) * 10 + K);
            const HostWeight hw = random_weight(fmt, N, K, rng);
            for (int64_t M : {1, 7, 16, 40}) {
                const std::vector<float> x = rng.vec(static_cast<size_t>(M * K), 1.0f);
                const std::vector<float> y0(static_cast<size_t>(M * ldy), 7.0f);  // sentinel past N
                const std::string name = std::string("gemm/") + fmt_name(fmt) + "/K" + std::to_string(K) + "/M" +
                                         std::to_string(M);
                check(r, cpu, gpu, name, M <= fullm::kGemvMaxRows ? 1e-5 : 2e-2, [&](Backend &be) {
                    Arena ar(be);
                    const Weight wt = hw.upload(ar);
                    const float *xd = ar.up(x);
                    float *yd = ar.up(y0);
                    be.gemm(wt, xd, M, yd, ldy);
                    return ar.down(yd, y0.size());
                });
            }
        }
    }
}

void test_elementwise(Report &r, Backend &cpu, Backend &gpu) {
    Rng rng(7);
    {  // rmsnorm, strided, in place and out of place
        const int64_t rows = 5, n = 100, stride = 130;
        const auto x = rng.vec(static_cast<size_t>(rows * stride), 3.0f);
        const auto w = rng.vec(static_cast<size_t>(n), 1.0f);
        check(r, cpu, gpu, "rmsnorm/in_place", 1e-5, [&](Backend &be) {
            Arena ar(be);
            float *xd = ar.up(x);
            const float *wd = ar.up(w);
            be.rmsnorm(xd, xd, wd, rows, n, stride, 1e-5f);
            return ar.down(xd, x.size());
        });
        const std::vector<float> y0(x.size(), -3.0f);
        check(r, cpu, gpu, "rmsnorm/out_of_place", 1e-5, [&](Backend &be) {
            Arena ar(be);
            const float *xd = ar.up(x);
            const float *wd = ar.up(w);
            float *yd = ar.up(y0);
            be.rmsnorm(yd, xd, wd, rows, n, stride, 1e-5f);
            return ar.down(yd, y0.size());
        });
    }
    {
        const auto x = rng.vec(3 * 4 * 32, 2.0f);
        if (has_fam(gpu, FAM_KDA)) check(r, cpu, gpu, "l2norm_heads", 1e-5, [&](Backend &be) {
            Arena ar(be);
            float *xd = ar.up(x);
            be.l2norm_heads(xd, 3, 4, 32, 1e-6f);
            return ar.down(xd, x.size());
        });
    }
    {  // SiTU with both caps saturated somewhere
        std::vector<float> g = rng.vec(1000, 1000.0f), u = rng.vec(1000, 1000.0f);
        for (size_t i = 0; i < 500; ++i) {
            g[i] *= 0.003f;
            u[i] *= 0.003f;
        }
        if (has_fam(gpu, FAM_KIMI)) check(r, cpu, gpu, "situ_glu", 1e-5, [&](Backend &be) {
            Arena ar(be);
            const float *gd = ar.up(g);
            const float *ud = ar.up(u);
            float *yd = ar.up(std::vector<float>(1000, 0.0f));
            be.situ_glu(yd, gd, ud, 1000, 4.0f, 25.0f);
            return ar.down(yd, 1000);
        });
        check(r, cpu, gpu, "silu_glu", 1e-5, [&](Backend &be) {
            Arena ar(be);
            const float *gd = ar.up(g);
            const float *ud = ar.up(u);
            float *yd = ar.up(std::vector<float>(1000, 0.0f));
            be.silu_glu(yd, gd, ud, 1000);
            return ar.down(yd, 1000);
        });
    }
    {
        const auto a = rng.vec(1000, 1.0f), b = rng.vec(1000, 5.0f);
        check(r, cpu, gpu, "add", 1e-6, [&](Backend &be) {
            Arena ar(be);
            float *ad = ar.up(a);
            be.add(ad, ar.up(b), 1000);
            return ar.down(ad, 1000);
        });
        if (has_fam(gpu, FAM_KDA)) check(r, cpu, gpu, "sigmoid_mul", 1e-6, [&](Backend &be) {
            Arena ar(be);
            float *ad = ar.up(a);
            be.sigmoid_mul(ad, ar.up(b), 1000);
            return ar.down(ad, 1000);
        });
    }
    {
        const auto src = rng.vec(4 * 20, 1.0f);
        const std::vector<float> dst0(4 * 50, 9.0f);
        check(r, cpu, gpu, "copy_rows", 0.0, [&](Backend &be) {
            Arena ar(be);
            float *dd = ar.up(dst0);
            be.copy_rows(dd, 50, 7, ar.up(src), 4, 20);
            return ar.down(dd, dst0.size());
        });
    }
    for (int64_t nb : {0, 3}) {
        const int64_t rows = 3, n = 300, maxs = 6;
        const auto blocks = rng.vec(static_cast<size_t>(rows * maxs * n), 2.0f);
        const auto pref = rng.vec(static_cast<size_t>(rows * n), 2.0f);
        const auto fold = rng.vec(static_cast<size_t>(n), 0.1f);
        const std::vector<float> out0(static_cast<size_t>(rows * n), 5.0f);
        if (has_fam(gpu, FAM_KIMI)) check(r, cpu, gpu, "attn_res/nb" + std::to_string(nb), 1e-5, [&](Backend &be) {
            Arena ar(be);
            float *od = ar.up(out0);
            be.attn_res(od, ar.up(blocks), nb, maxs, ar.up(pref), ar.up(fold), rows, n, 1e-5f);
            return ar.down(od, out0.size());
        });
    }
}

void test_kda(Report &r, Backend &cpu, Backend &gpu) {
    Rng rng(99);
    {  // ShortConv over two segments with carried history at an offset
        const int64_t C = 70, K = 4, hist = K - 1, off = 10, stride = off + C * hist + 5, N = 8;
        const std::vector<int32_t> seg = {1, 0, 3, 0, 3, 5};
        const auto x = rng.vec(static_cast<size_t>(N * C), 1.5f);
        const auto w = rng.vec(static_cast<size_t>(C * K), 0.5f);
        const auto st = rng.vec(static_cast<size_t>(2 * stride), 1.0f);
        check(r, cpu, gpu, "kda_conv", 1e-5, [&](Backend &be) {
            Arena ar(be);
            float *yd = ar.up(std::vector<float>(static_cast<size_t>(N * C), 0.0f));
            float *sd = ar.up(st);
            be.kda_conv(yd, ar.up(x), ar.up(w), sd, stride, off, ar.up(seg), 2, C, K);
            return cat(ar.down(yd, static_cast<size_t>(N * C)), ar.down(sd, st.size()));
        });
    }
    {
        const int64_t N = 4, H = 3, D = 16;
        const auto z = rng.vec(static_cast<size_t>(N * H * D), 2.0f);
        const auto braw = rng.vec(static_cast<size_t>(N * H), 3.0f);
        const auto alog = rng.vec(static_cast<size_t>(H), 1.0f);
        const auto dtb = rng.vec(static_cast<size_t>(H * D), 0.5f);
        for (bool sp : {false, true}) {
            check(r, cpu, gpu, sp ? "kda_gates/softplus" : "kda_gates/lower_bound", 1e-5, [&](Backend &be) {
                Arena ar(be);
                float *ad = ar.up(std::vector<float>(static_cast<size_t>(N * H * D), 0.0f));
                float *bd = ar.up(std::vector<float>(static_cast<size_t>(N * H), 0.0f));
                be.kda_gates(ad, bd, ar.up(z), ar.up(braw), ar.up(alog), ar.up(dtb), N, H, D, -5.0f, sp);
                return cat(ar.down(ad, static_cast<size_t>(N * H * D)), ar.down(bd, static_cast<size_t>(N * H)));
            });
        }
    }
    for (int64_t D : {16, 128}) {
        for (DType dt : {DType::F32, DType::BF16}) {
            const int64_t H = D == 16 ? 3 : 2, N = 6, stride = H * D * D + 7;
            const std::vector<int32_t> seg = {1, 0, 1, 0, 1, 5};
            const float qs = 1.0f / std::sqrt(static_cast<float>(D));
            const auto q = rng.vec(static_cast<size_t>(N * H * D), 1.7f * qs);
            const auto k = rng.vec(static_cast<size_t>(N * H * D), 1.7f * qs);
            const auto v = rng.vec(static_cast<size_t>(N * H * D), 1.0f);
            std::vector<float> alpha(static_cast<size_t>(N * H * D)), beta(static_cast<size_t>(N * H));
            for (float &a : alpha) a = rng.range(0.5f, 1.0f);
            for (float &b : beta) b = rng.range(0.0f, 1.0f);
            const auto s0 = rng.vec(static_cast<size_t>(2 * stride), 0.1f);
            std::vector<uint16_t> s0b(s0.size());
            for (size_t i = 0; i < s0.size(); ++i) s0b[i] = bf16_rne(s0[i]);
            const std::string name = "kda_recurrence/D" + std::to_string(D) + (dt == DType::F32 ? "/f32" : "/bf16");
            check(r, cpu, gpu, name, dt == DType::F32 ? 1e-4 : 1e-2, [&](Backend &be) {
                Arena ar(be);
                float *od = ar.up(std::vector<float>(static_cast<size_t>(N * H * D), 0.0f));
                void *sd = dt == DType::F32 ? static_cast<void *>(ar.up(s0)) : static_cast<void *>(ar.up(s0b));
                be.kda_recurrence(od, ar.up(q), ar.up(k), ar.up(v), ar.up(alpha), ar.up(beta), sd, dt, stride,
                                  ar.up(seg), 2, H, D, qs);
                std::vector<float> out = ar.down(od, static_cast<size_t>(N * H * D));
                if (dt == DType::F32) return cat(out, ar.down(static_cast<float *>(sd), s0.size()));
                const auto sb = ar.down(static_cast<uint16_t *>(sd), s0b.size());
                std::vector<float> sf(sb.size());
                for (size_t i = 0; i < sb.size(); ++i) sf[i] = bf16f(sb[i]);
                return cat(out, sf);
            });
        }
    }
    {
        const int64_t N = 4, H = 3, D = 16;
        const auto o = rng.vec(static_cast<size_t>(N * H * D), 2.0f);
        const auto g = rng.vec(static_cast<size_t>(N * H * D), 3.0f);
        const auto w = rng.vec(static_cast<size_t>(D), 1.0f);
        check(r, cpu, gpu, "kda_out", 1e-5, [&](Backend &be) {
            Arena ar(be);
            float *od = ar.up(o);
            be.kda_out(od, ar.up(g), ar.up(w), N, H, D, 1e-5f);
            return ar.down(od, o.size());
        });
    }
}

// The fused KDA mixer against the CPU backend's op sequence (conv x3, L2 norms, gates,
// recurrence, output norm and gate): output, conv histories and recurrent state, over a
// prefill segment and a decode segment.
void test_kda_fused(Report &r, Backend &cpu, Backend &gpu) {
    Rng rng(515);
    for (int64_t D : {16, 128}) {
        for (DType dt : {DType::F32, DType::BF16}) {
            for (bool sp : {true, false}) {
                const int64_t H = 3, K = 4, P = H * D, N = 6, hist = P * (K - 1);
                const int64_t cstride = 3 * hist + 5, sstride = H * D * D + 3;
                const std::vector<int32_t> seg = {1, 0, 5, 0, 5, 1};  // slot 1: 5 tokens; slot 0: 1
                const auto qp = rng.vec(static_cast<size_t>(N * P), 1.0f), kp = rng.vec(static_cast<size_t>(N * P), 1.0f);
                const auto vp = rng.vec(static_cast<size_t>(N * P), 1.0f);
                const auto wq = rng.vec(static_cast<size_t>(P * K), 0.5f), wk = rng.vec(static_cast<size_t>(P * K), 0.5f);
                const auto wv = rng.vec(static_cast<size_t>(P * K), 0.5f);
                const auto conv0 = rng.vec(static_cast<size_t>(2 * cstride), 1.0f);
                const auto z = rng.vec(static_cast<size_t>(N * P), 2.0f), braw = rng.vec(static_cast<size_t>(N * H), 3.0f);
                const auto alog = rng.vec(static_cast<size_t>(H), 1.0f), dtb = rng.vec(static_cast<size_t>(P), 0.5f);
                const auto gate = rng.vec(static_cast<size_t>(N * P), 3.0f), onorm = rng.vec(static_cast<size_t>(D), 1.0f);
                const auto s0 = rng.vec(static_cast<size_t>(2 * sstride), 0.1f);
                std::vector<uint16_t> s0b(s0.size());
                for (size_t i = 0; i < s0.size(); ++i) s0b[i] = bf16_rne(s0[i]);
                const float qs = 1.0f / std::sqrt(static_cast<float>(D)), eps = 1e-5f, lb = -5.0f;
                auto run = [&](Backend &be, bool fused) {
                    Arena ar(be);
                    float *o = ar.up(std::vector<float>(static_cast<size_t>(N * P), 0.0f));
                    float *conv = ar.up(conv0);
                    void *st = dt == DType::F32 ? static_cast<void *>(ar.up(s0)) : static_cast<void *>(ar.up(s0b));
                    const float *qpd = ar.up(qp), *kpd = ar.up(kp), *vpd = ar.up(vp), *wqd = ar.up(wq), *wkd = ar.up(wk),
                                *wvd = ar.up(wv), *zd = ar.up(z), *bd = ar.up(braw), *ad = ar.up(alog), *dd = ar.up(dtb),
                                *gd = ar.up(gate), *nd = ar.up(onorm);
                    const int32_t *sg = ar.up(seg);
                    if (fused) {
                        Backend::KdaFused f;
                        f.o = o, f.qp = qpd, f.kp = kpd, f.vp = vpd, f.wq = wqd, f.wk = wkd, f.wv = wvd;
                        f.conv_state = conv, f.conv_stride = cstride;
                        f.z = zd, f.braw = bd, f.a_log = ad, f.dt_bias = dd, f.lb = lb, f.softplus = sp;
                        f.gate = gd, f.o_norm = nd, f.out_eps = eps, f.l2_eps = 1e-6f, f.qscale = qs;
                        f.state = st, f.state_dtype = dt, f.state_stride = sstride;
                        f.seg = sg, f.nseg = 2, f.H = H, f.D = D, f.K = K;
                        if (!be.kda_fused(f)) throw Absent("kda_fused");
                    } else {
                        float *q = ar.up(std::vector<float>(static_cast<size_t>(N * P), 0.0f));
                        float *k = ar.up(std::vector<float>(static_cast<size_t>(N * P), 0.0f));
                        float *v = ar.up(std::vector<float>(static_cast<size_t>(N * P), 0.0f));
                        float *al = ar.up(std::vector<float>(static_cast<size_t>(N * P), 0.0f));
                        float *be_ = ar.up(std::vector<float>(static_cast<size_t>(N * H), 0.0f));
                        be.kda_conv(q, qpd, wqd, conv, cstride, 0, sg, 2, P, K);
                        be.kda_conv(k, kpd, wkd, conv, cstride, hist, sg, 2, P, K);
                        be.kda_conv(v, vpd, wvd, conv, cstride, 2 * hist, sg, 2, P, K);
                        be.l2norm_heads(q, N, H, D, 1e-6f);
                        be.l2norm_heads(k, N, H, D, 1e-6f);
                        be.kda_gates(al, be_, zd, bd, ad, dd, N, H, D, lb, sp);
                        be.kda_recurrence(o, q, k, v, al, be_, st, dt, sstride, sg, 2, H, D, qs);
                        be.kda_out(o, gd, nd, N, H, D, eps);
                    }
                    std::vector<float> out = cat(ar.down(o, static_cast<size_t>(N * P)), ar.down(conv, conv0.size()));
                    if (dt == DType::F32) return cat(out, ar.down(static_cast<float *>(st), s0.size()));
                    const auto sb = ar.down(static_cast<uint16_t *>(st), s0b.size());
                    std::vector<float> sf(sb.size());
                    for (size_t i = 0; i < sb.size(); ++i) sf[i] = bf16f(sb[i]);
                    return cat(out, sf);
                };
                const std::string name = "kda_fused/D" + std::to_string(D) + (dt == DType::F32 ? "/f32" : "/bf16") +
                                         (sp ? "/softplus" : "/lower_bound");
                try {
                    compare(r, name, run(cpu, false), run(gpu, true), dt == DType::F32 ? 1e-4 : 1e-2);
                } catch (const Absent &e) {
                    r.text += name + " skipped: the subject has no " + e.what() + "\n";
                } catch (const std::exception &e) {
                    r.error(name, e.what());
                }
            }
        }
    }
}

// KDA over prompts in chunks (kda_chunked) and the step's other segments token by token
// (kda_fused), as the executor splits a step, against the CPU's token-by-token sequence:
// segments of several chunks, one with a partial last chunk, one under kda_chunk_min, one token.
void test_kda_chunked(Report &r, Backend &cpu, Backend &gpu) {
    Rng rng(616);
    for (int64_t D : {64, 128}) {
        for (DType dt : {DType::F32, DType::BF16}) {
            for (bool sp : {true, false}) {
                const int64_t H = 3, K = 4, P = H * D, hist = P * (K - 1), slots = 5;
                // (slot, offset, length): 100 tokens, 37, 70, 1, and 700 (22 chunks: the state carried far)
                const std::vector<int32_t> seg = {2, 0, 100, 0, 100, 37, 1, 137, 70, 3, 207, 1, 4, 208, 700};
                const int64_t N = 908, nseg = 5;
                const int64_t cstride = 3 * hist + 5, sstride = H * D * D + 3;
                const auto qp = rng.vec(static_cast<size_t>(N * P), 1.0f), kp = rng.vec(static_cast<size_t>(N * P), 1.0f);
                const auto vp = rng.vec(static_cast<size_t>(N * P), 1.0f);
                const auto wq = rng.vec(static_cast<size_t>(P * K), 0.5f), wk = rng.vec(static_cast<size_t>(P * K), 0.5f);
                const auto wv = rng.vec(static_cast<size_t>(P * K), 0.5f);
                const auto conv0 = rng.vec(static_cast<size_t>(slots * cstride), 1.0f);
                const auto z = rng.vec(static_cast<size_t>(N * P), 2.0f), braw = rng.vec(static_cast<size_t>(N * H), 3.0f);
                const auto alog = rng.vec(static_cast<size_t>(H), 1.0f), dtb = rng.vec(static_cast<size_t>(P), 0.5f);
                const auto gate = rng.vec(static_cast<size_t>(N * P), 3.0f), onorm = rng.vec(static_cast<size_t>(D), 1.0f);
                const auto s0 = rng.vec(static_cast<size_t>(slots * sstride), 0.1f);
                std::vector<uint16_t> s0b(s0.size());
                for (size_t i = 0; i < s0.size(); ++i) s0b[i] = bf16_rne(s0[i]);
                const float qs = 1.0f / std::sqrt(static_cast<float>(D)), eps = 1e-5f, lb = -5.0f;
                auto run = [&](Backend &be, bool chunked) {
                    Arena ar(be);
                    float *o = ar.up(std::vector<float>(static_cast<size_t>(N * P), 0.0f));
                    float *conv = ar.up(conv0);
                    void *st = dt == DType::F32 ? static_cast<void *>(ar.up(s0)) : static_cast<void *>(ar.up(s0b));
                    const float *qpd = ar.up(qp), *kpd = ar.up(kp), *vpd = ar.up(vp), *wqd = ar.up(wq), *wkd = ar.up(wk),
                                *wvd = ar.up(wv), *zd = ar.up(z), *bd = ar.up(braw), *ad = ar.up(alog), *dd = ar.up(dtb),
                                *gd = ar.up(gate), *nd = ar.up(onorm);
                    if (chunked) {
                        const int64_t cmin = be.kda_chunk_min();
                        if (cmin <= 0) throw Absent("kda_chunked");
                        std::vector<int32_t> lng, sht;
                        for (int64_t i = 0; i < nseg; ++i) {
                            auto &dst = seg[size_t(3 * i + 2)] >= cmin ? lng : sht;
                            dst.insert(dst.end(), seg.begin() + 3 * i, seg.begin() + 3 * i + 3);
                        }
                        Backend::KdaFused f;
                        f.o = o, f.qp = qpd, f.kp = kpd, f.vp = vpd, f.wq = wqd, f.wk = wkd, f.wv = wvd;
                        f.conv_state = conv, f.conv_stride = cstride;
                        f.z = zd, f.braw = bd, f.a_log = ad, f.dt_bias = dd, f.lb = lb, f.softplus = sp;
                        f.gate = gd, f.o_norm = nd, f.out_eps = eps, f.l2_eps = 1e-6f, f.qscale = qs;
                        f.state = st, f.state_dtype = dt, f.state_stride = sstride;
                        f.H = H, f.D = D, f.K = K;
                        f.seg = ar.up(lng), f.nseg = int64_t(lng.size() / 3);
                        if (lng.empty() || !be.kda_chunked(f, lng.data())) throw Absent("kda_chunked");
                        if (!sht.empty()) {
                            f.seg = ar.up(sht), f.nseg = int64_t(sht.size() / 3);
                            if (!be.kda_fused(f)) throw Absent("kda_fused");
                        }
                    } else {
                        const int32_t *sg = ar.up(seg);
                        float *q = ar.up(std::vector<float>(static_cast<size_t>(N * P), 0.0f));
                        float *k = ar.up(std::vector<float>(static_cast<size_t>(N * P), 0.0f));
                        float *v = ar.up(std::vector<float>(static_cast<size_t>(N * P), 0.0f));
                        float *al = ar.up(std::vector<float>(static_cast<size_t>(N * P), 0.0f));
                        float *be_ = ar.up(std::vector<float>(static_cast<size_t>(N * H), 0.0f));
                        be.kda_conv(q, qpd, wqd, conv, cstride, 0, sg, nseg, P, K);
                        be.kda_conv(k, kpd, wkd, conv, cstride, hist, sg, nseg, P, K);
                        be.kda_conv(v, vpd, wvd, conv, cstride, 2 * hist, sg, nseg, P, K);
                        be.l2norm_heads(q, N, H, D, 1e-6f);
                        be.l2norm_heads(k, N, H, D, 1e-6f);
                        be.kda_gates(al, be_, zd, bd, ad, dd, N, H, D, lb, sp);
                        be.kda_recurrence(o, q, k, v, al, be_, st, dt, sstride, sg, nseg, H, D, qs);
                        be.kda_out(o, gd, nd, N, H, D, eps);
                    }
                    std::vector<float> out = cat(ar.down(o, static_cast<size_t>(N * P)), ar.down(conv, conv0.size()));
                    if (dt == DType::F32) return cat(out, ar.down(static_cast<float *>(st), s0.size()));
                    const auto sb = ar.down(static_cast<uint16_t *>(st), s0b.size());
                    std::vector<float> sf(sb.size());
                    for (size_t i = 0; i < sb.size(); ++i) sf[i] = bf16f(sb[i]);
                    return cat(out, sf);
                };
                const std::string name = "kda_chunked/D" + std::to_string(D) + (dt == DType::F32 ? "/f32" : "/bf16") +
                                         (sp ? "/softplus" : "/lower_bound");
                try {
                    compare(r, name, run(cpu, false), run(gpu, true), dt == DType::F32 ? 2e-4 : 1e-2);
                } catch (const Absent &e) {
                    r.text += name + " skipped: the subject has no " + e.what() + "\n";
                } catch (const std::exception &e) {
                    r.error(name, e.what());
                }
            }
        }
    }
}

// add + rmsnorm fused: equal to the CPU, and bit-identical to the separate CUDA kernels.
void test_add_rmsnorm(Report &r, Backend &cpu, Backend &gpu) {
    Rng rng(808);
    const int64_t rows = 5, n = 2304;
    const auto h = rng.vec(static_cast<size_t>(rows * n), 2.0f), d = rng.vec(static_cast<size_t>(rows * n), 1.0f);
    const auto w = rng.vec(static_cast<size_t>(n), 1.0f);
    auto run = [&](Backend &be, bool fused) {
        Arena ar(be);
        float *hd = ar.up(h), *yd = ar.up(std::vector<float>(h.size(), 0.0f));
        if (fused) be.add_rmsnorm(hd, ar.up(d), yd, ar.up(w), rows, n, 1e-6f);
        else {
            be.add(hd, ar.up(d), rows * n);
            be.rmsnorm(yd, hd, ar.up(w), rows, n, n, 1e-6f);
        }
        return cat(ar.down(hd, h.size()), ar.down(yd, h.size()));
    };
    try {
        compare(r, "add_rmsnorm", run(cpu, false), run(gpu, true), 1e-5);
        compare(r, "add_rmsnorm (bitwise vs add, rmsnorm)", run(gpu, false), run(gpu, true), 0.0);
    } catch (const std::exception &e) {
        r.error("add_rmsnorm", e.what());
    }
    // few rows go through a cluster of blocks a row (the CUDA backend), many through a block a row:
    // a row's h and y bit for bit the same either way
    for (int64_t wn : {int64_t{2304}, int64_t{4096}}) {
        const int64_t many = 200, few = 5;
        const auto hb = rng.vec(static_cast<size_t>(many * wn), 2.0f), db = rng.vec(static_cast<size_t>(many * wn), 1.0f);
        const auto wb = rng.vec(static_cast<size_t>(wn), 1.0f);
        auto rows_of = [&](int64_t m) {
            Arena ar(gpu);
            const size_t ne = static_cast<size_t>(m * wn);
            float *hd = ar.up(std::vector<float>(hb.begin(), hb.begin() + ne));
            float *yd = ar.up(std::vector<float>(ne, 0.0f));
            gpu.add_rmsnorm(hd, ar.up(std::vector<float>(db.begin(), db.begin() + ne)), yd, ar.up(wb), m, wn, 1e-6f);
            std::vector<float> hh = ar.down(hd, ne), yy = ar.down(yd, ne);
            hh.resize(static_cast<size_t>(few * wn)), yy.resize(static_cast<size_t>(few * wn));
            return cat(hh, yy);
        };
        const std::string name = "add_rmsnorm (bitwise " + std::to_string(few) + " rows vs " + std::to_string(many) + ", n " +
                                 std::to_string(wn) + ")";
        try {
            compare(r, name, rows_of(many), rows_of(few), 0.0);
        } catch (const std::exception &e) {
            r.error(name, e.what());
        }
    }
}

void test_mla(Report &r, Backend &cpu, Backend &gpu) {
    Rng rng(4242);
    // Cache: 2 slots, 600 and 38 positions, pages scattered through the pool.
    const int64_t H = 20, L = 128, qr = 32, W = L + qr, page = 16, max_pages = 40, n_pages = 48;
    const int64_t lens[2] = {600, 38};
    std::vector<int32_t> perm(n_pages);
    std::iota(perm.begin(), perm.end(), 0);
    for (int64_t i = n_pages - 1; i > 0; --i) std::swap(perm[i], perm[rng.next() % (i + 1)]);
    std::vector<int32_t> pt(2 * max_pages, 0);
    int used = 0;
    for (int s = 0; s < 2; ++s)
        for (int64_t p = 0; p < (lens[s] + page - 1) / page; ++p) pt[s * max_pages + p] = perm[used++];
    std::vector<int32_t> aslot, apos;
    for (int s = 0; s < 2; ++s)
        for (int64_t p = 0; p < lens[s]; ++p) {
            aslot.push_back(s);
            apos.push_back(static_cast<int32_t>(p));
        }
    const int64_t napp = static_cast<int64_t>(aslot.size());
    const auto ckv = rng.vec(static_cast<size_t>(napp * W), 1.0f);

    struct Q {
        const char *name;
        std::vector<int32_t> slot, pos;
    };
    std::vector<Q> sets;
    sets.push_back({"decode", {0, 0, 1}, {599, 300, 37}});
    Q pre{"prefill", {}, {}};
    for (int32_t p = 0; p < 70; ++p) {
        pre.slot.push_back(0);
        pre.pos.push_back(p);
    }
    sets.push_back(pre);

    for (const Q &qs : sets) {
        const int64_t N = static_cast<int64_t>(qs.slot.size());
        const int32_t max_pos = *std::max_element(qs.pos.begin(), qs.pos.end());
        const auto qa = rng.vec(static_cast<size_t>(N * H * W), 0.3f);
        for (DType dt : {DType::F32, DType::BF16}) {
            const size_t esz = dt == DType::F32 ? 4 : 2;
            const size_t kv_elems = static_cast<size_t>(n_pages * page * W);
            const std::string name =
                std::string("mla_append+attention/") + qs.name + (dt == DType::F32 ? "/f32" : "/bf16");
            check(r, cpu, gpu, name, dt == DType::F32 ? 1e-4 : 2e-3, [&](Backend &be) {
                Arena ar(be);
                void *kv = ar.zeros(kv_elems * esz);
                const int32_t *ptd = ar.up(pt);
                be.mla_append(kv, dt, ar.up(ckv), ar.up(aslot), ar.up(apos), ptd, max_pages, page, W, napp);
                float *od = ar.up(std::vector<float>(static_cast<size_t>(N * H * L), 0.0f));
                be.mla_attention(od, ar.up(qa), kv, dt, ar.up(qs.slot), ar.up(qs.pos), ptd, max_pages, page, N, H, L,
                                 qr, max_pos, 0.1f);
                std::vector<float> out = ar.down(od, static_cast<size_t>(N * H * L));
                if (dt == DType::F32) return cat(out, ar.down(static_cast<float *>(kv), kv_elems));
                const auto kb = ar.down(static_cast<uint16_t *>(kv), kv_elems);
                std::vector<float> kf(kb.size());
                for (size_t i = 0; i < kb.size(); ++i) kf[i] = bf16f(kb[i]);
                return cat(out, kf);
            });
        }
    }

    // absorb / un-absorb through a bf16 kv_b
    const int64_t N = 11, Hh = 5, qn = 24, qr2 = 8, vh = 16, L2 = 64;
    std::vector<uint16_t> kvb(static_cast<size_t>(Hh * (qn + vh) * L2));
    for (uint16_t &h : kvb) h = bf16_rne(rng.uni() * 0.2f);
    const auto q = rng.vec(static_cast<size_t>(N * Hh * (qn + qr2)), 1.0f);
    const auto olat = rng.vec(static_cast<size_t>(N * Hh * L2), 1.0f);
    auto weight = [&](Arena &ar) {
        Weight w;
        w.fmt = WFmt::BF16;
        w.rows = Hh * (qn + vh);
        w.cols = L2;
        w.data = ar.up(kvb);
        return w;
    };
    check(r, cpu, gpu, "mla_absorb_q", 1e-5, [&](Backend &be) {
        Arena ar(be);
        const Weight w = weight(ar);
        float *qd = ar.up(std::vector<float>(static_cast<size_t>(N * Hh * (L2 + qr2)), 0.0f));
        be.mla_absorb_q(qd, ar.up(q), w, N, Hh, qn, qr2, vh);
        return ar.down(qd, static_cast<size_t>(N * Hh * (L2 + qr2)));
    });
    check(r, cpu, gpu, "mla_unabsorb", 1e-5, [&](Backend &be) {
        Arena ar(be);
        const Weight w = weight(ar);
        float *yd = ar.up(std::vector<float>(static_cast<size_t>(N * Hh * vh), 0.0f));
        be.mla_unabsorb(yd, ar.up(olat), w, N, Hh, qn, vh);
        return ar.down(yd, static_cast<size_t>(N * Hh * vh));
    });
}

// Steps of at most kGemvMaxRows tokens split each token's context by its own position, so its
// attention output is bit-identical alone and in any small batch. The 2600-position context
// takes the split length past its minimum (more than 64 splits of 32).
void test_mla_invariance(Report &r, Backend &cpu, Backend &gpu) {
    Rng rng(777);
    const int64_t H = 20, L = 128, qr = 32, W = L + qr, page = 16, len = 2600;
    const int64_t max_pages = (len + page - 1) / page;
    std::vector<int32_t> pt(static_cast<size_t>(max_pages));
    std::iota(pt.begin(), pt.end(), 0);
    std::reverse(pt.begin(), pt.end());
    const auto cache = rng.vec(static_cast<size_t>(max_pages * page * W), 1.0f);
    const std::vector<int32_t> pos = {2599, 1000, 37, 2047, 2048, 511, 0, 31, 32, 33};
    const auto qa = rng.vec(pos.size() * static_cast<size_t>(H * W), 0.3f);
    auto run = [&](Backend &be, const std::vector<size_t> &which) {
        Arena ar(be);
        const int64_t n = static_cast<int64_t>(which.size());
        std::vector<int32_t> slot(which.size(), 0), p;
        std::vector<float> q;
        for (size_t i : which) {
            p.push_back(pos[i]);
            q.insert(q.end(), qa.begin() + static_cast<int64_t>(i) * H * W, qa.begin() + static_cast<int64_t>(i + 1) * H * W);
        }
        float *od = static_cast<float *>(ar.zeros(sizeof(float) * n * H * L));
        be.mla_attention(od, ar.up(q), ar.up(cache), DType::F32, ar.up(slot), ar.up(p), ar.up(pt), max_pages, page, n,
                         H, L, qr, *std::max_element(p.begin(), p.end()), 0.1f);
        return ar.down(od, static_cast<size_t>(n * H * L));
    };
    std::vector<size_t> all(pos.size());
    std::iota(all.begin(), all.end(), 0);
    check(r, cpu, gpu, "mla_attention/long-context", 1e-4, [&](Backend &be) { return run(be, all); });
    try {
        const auto batch = run(gpu, all);
        std::vector<float> alone;
        for (size_t i : all) alone = cat(alone, run(gpu, {i}));
        compare(r, "mla_attention/batch-invariant (bitwise)", alone, batch, 0.0);
    } catch (const std::exception &e) {
        r.error("mla_attention/batch-invariant", e.what());
    }
}

// MLA un-absorbed over long segments (mla_prefill, on the tensor cores in bf16) with the other
// rows through the absorbed kernels compacted (index_rows), as the Kimi executor runs a step:
// against the absorbed path over every row. Segments: 100 new tokens after 50 cached positions,
// 70 from position 0, a decode at position 30, 5 tokens after 10 (those two absorbed) and 40
// after 300. Each as the backend chooses, then in chunks of 64 keys merged by their log-sum-exp
// (the 300-position context in six), then also in passes of 2 heads.
void test_mla_prefill(Report &r, Backend &cpu, Backend &gpu) {
    Rng rng(5151);
    const int64_t H = 6, qn = 128, qr = 64, vh = 128, L = 512, W = L + qr, QK = qn + qr, page = 16;
    const int64_t max_pages = 22, n_pages = 48;
    // (slot, first row, count, first position)
    const std::vector<std::array<int32_t, 4>> segs = {
        {1, 0, 100, 50}, {2, 100, 1, 30}, {0, 101, 70, 0}, {3, 171, 5, 10}, {4, 176, 40, 300}};
    const int64_t N = 216, last_pos = 339;
    std::vector<int32_t> perm(n_pages);
    std::iota(perm.begin(), perm.end(), 0);
    for (int64_t i = n_pages - 1; i > 0; --i) std::swap(perm[i], perm[rng.next() % (i + 1)]);
    std::vector<int32_t> pt(5 * max_pages, 0), aslot, apos, tslot(N), tpos(N);
    int used = 0;
    for (const auto &sg : segs) {
        const int32_t end = sg[3] + sg[2];  // every position of the slot up to the segment's last
        for (int64_t p = 0; p < (end + page - 1) / page; ++p) pt[sg[0] * max_pages + p] = perm[used++];
        for (int32_t p = 0; p < end; ++p) aslot.push_back(sg[0]), apos.push_back(p);
        for (int32_t t = 0; t < sg[2]; ++t) tslot[sg[1] + t] = sg[0], tpos[sg[1] + t] = sg[3] + t;
    }
    const int64_t napp = static_cast<int64_t>(aslot.size());
    const auto ckv = rng.vec(static_cast<size_t>(napp * W), 1.0f);
    const auto q = rng.vec(static_cast<size_t>(N * H * QK), 1.0f);
    std::vector<uint16_t> kvb(static_cast<size_t>(H * (qn + vh) * L));
    for (uint16_t &h : kvb) h = bf16_rne(rng.uni() * 0.08f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(QK));
    for (DType dt : {DType::F32, DType::BF16}) {
        auto run = [&](Backend &be, bool un, int64_t chunk_keys, int64_t chunk_heads) {
            Arena ar(be);
            const size_t esz = dt == DType::F32 ? 4 : 2;
            void *kv = ar.zeros(static_cast<size_t>(n_pages * page * W) * esz);
            const int32_t *ptd = ar.up(pt);
            be.mla_append(kv, dt, ar.up(ckv), ar.up(aslot), ar.up(apos), ptd, max_pages, page, W, napp);
            Weight w;
            w.fmt = WFmt::BF16, w.rows = H * (qn + vh), w.cols = L, w.data = ar.up(kvb);
            const float *qd = ar.up(q);
            float *y = ar.up(std::vector<float>(static_cast<size_t>(N * H * vh), 0.0f));
            float *qa = ar.up(std::vector<float>(static_cast<size_t>(N * H * W), 0.0f));
            float *ol = ar.up(std::vector<float>(static_cast<size_t>(N * H * L), 0.0f));
            if (!un) {
                be.mla_absorb_q(qa, qd, w, N, H, qn, qr, vh);
                be.mla_attention(ol, qa, kv, dt, ar.up(tslot), ar.up(tpos), ptd, max_pages, page, N, H, L, qr, last_pos, scale);
                be.mla_unabsorb(y, ol, w, N, H, qn, vh);
                return ar.down(y, static_cast<size_t>(N * H * vh));
            }
            // the subject's bf16-activation mode, where it runs mla_prefill (exact mode keeps the fp32 kernels)
            struct Bf16Mode {
                Backend &b;
                explicit Bf16Mode(Backend &x) : b(x) { b.set_exact_activations(false); }
                ~Bf16Mode() { b.set_exact_activations(true); }
            } mode(be);
            const int64_t min = be.mla_prefill_min();
            if (min <= 0) throw Absent("mla_prefill");
            std::vector<int32_t> cu_q{0}, cu_k{0}, slot, first, rows, ss, sp;
            int64_t max_q = 0, max_k = 0, max_pos = 0;
            for (const auto &sg : segs) {
                if (sg[2] >= min) {
                    cu_q.push_back(cu_q.back() + sg[2]), cu_k.push_back(cu_k.back() + sg[3] + sg[2]);
                    slot.push_back(sg[0]), first.push_back(sg[1]);
                    max_q = std::max<int64_t>(max_q, sg[2]), max_k = std::max<int64_t>(max_k, sg[3] + sg[2]);
                } else {
                    for (int32_t t = 0; t < sg[2]; ++t)
                        rows.push_back(sg[1] + t), ss.push_back(sg[0]), sp.push_back(sg[3] + t);
                    max_pos = std::max<int64_t>(max_pos, sg[3] + sg[2] - 1);
                }
            }
            std::vector<int32_t> meta(cu_q);
            meta.insert(meta.end(), cu_k.begin(), cu_k.end());
            meta.insert(meta.end(), slot.begin(), slot.end());
            meta.insert(meta.end(), first.begin(), first.end());
            Backend::MlaPrefill m{};
            m.y = y, m.q = qd, m.kv = kv, m.kv_dtype = dt, m.kv_b = &w;
            m.page_table = ptd, m.max_pages = max_pages, m.page = page, m.meta = ar.up(meta), m.meta_host = meta.data();
            m.n = int64_t(slot.size()), m.total_q = cu_q.back(), m.total_k = cu_k.back(), m.max_q = max_q, m.max_k = max_k;
            m.H = H, m.qn = qn, m.qr = qr, m.vh = vh, m.kv_lora = L, m.scale = scale;
            m.chunk_keys = chunk_keys, m.chunk_heads = chunk_heads;
            if (slot.size() != 3 || !be.mla_prefill(m)) throw Absent("mla_prefill");
            const int64_t Ns = int64_t(rows.size());
            const int32_t *rd = ar.up(rows);
            float *qs = ar.up(std::vector<float>(static_cast<size_t>(Ns * H * QK), 0.0f));
            float *ys = ar.up(std::vector<float>(static_cast<size_t>(Ns * H * vh), 0.0f));
            be.index_rows(qs, qd, rd, Ns, H * QK, false);
            be.mla_absorb_q(qa, qs, w, Ns, H, qn, qr, vh);
            be.mla_attention(ol, qa, kv, dt, ar.up(ss), ar.up(sp), ptd, max_pages, page, Ns, H, L, qr, max_pos, scale);
            be.mla_unabsorb(ys, ol, w, Ns, H, qn, vh);
            be.index_rows(y, ys, rd, Ns, H * vh, true);
            return ar.down(y, static_cast<size_t>(N * H * vh));
        };
        const std::string base = std::string("mla_prefill/") + (dt == DType::F32 ? "f32" : "bf16");
        std::vector<float> want;
        try {
            want = run(cpu, false, 0, 0);
        } catch (const std::exception &e) {
            r.error(base, e.what());
            continue;
        }
        const struct {
            const char *name;
            int64_t keys, heads;
        } ways[] = {{"", 0, 0}, {"/chunks", 64, 0}, {"/chunks+passes", 64, 2}};
        for (const auto &w : ways) {
            const std::string name = base + w.name;
            try {
                compare(r, name, want, run(gpu, true, w.keys, w.heads), 2e-2);
            } catch (const Absent &e) {
                r.text += name + " skipped: the subject has no " + e.what() + "\n";
            } catch (const std::exception &e) {
                r.error(name, e.what());
            }
        }
    }
}

// fp8 KV pools (kv_fp8.h) from fp32 [n_pages][page][Hk][D] keys or values (deq: the values the pool
// holds, in the fp32 layout), and the values an fp8 pool holds
void kv8_pool(const std::vector<float> &c, int64_t n_pages, int64_t page, int64_t Hk, int64_t D, std::vector<uint8_t> &pool,
              std::vector<float> &deq) {
    pool.assign(static_cast<size_t>(n_pages * kv8_page_bytes(page, Hk, D)), 0);
    deq.assign(c.size(), 0.0f);
    for (int64_t pg = 0; pg < n_pages; ++pg)
        for (int64_t r = 0; r < page; ++r)
            for (int64_t h = 0; h < Hk; ++h) {
                const int64_t at = ((pg * page + r) * Hk + h) * D;
                uint8_t *q = pool.data() + kv8_row(pg, r, h, page, Hk, D);
                const uint8_t e = fullm::kv8_quantize(c.data() + at, D, q);
                pool[static_cast<size_t>(kv8_exp(pg, r, h, page, Hk, D))] = e;
                for (int64_t i = 0; i < D; ++i) deq[static_cast<size_t>(at + i)] = fullm::e4m3_to_f32(q[i]) * fullm::kv8_scale(e);
            }
}
std::vector<float> kv8_values(const std::vector<uint8_t> &pool, int64_t n_pages, int64_t page, int64_t Hk, int64_t D) {
    std::vector<float> v(static_cast<size_t>(n_pages * page * Hk * D));
    for (int64_t pg = 0; pg < n_pages; ++pg)
        for (int64_t r = 0; r < page; ++r)
            for (int64_t h = 0; h < Hk; ++h) {
                const float sc = fullm::kv8_scale(pool[static_cast<size_t>(kv8_exp(pg, r, h, page, Hk, D))]);
                for (int64_t i = 0; i < D; ++i)
                    v[static_cast<size_t>(((pg * page + r) * Hk + h) * D + i)] =
                        fullm::e4m3_to_f32(pool[static_cast<size_t>(kv8_row(pg, r, h, page, Hk, D) + i)]) * sc;
            }
    return v;
}

// Dense attention (Llama / Qwen): RoPE, grouped-query attention over a paged cache (decode,
// prefill, a long context; f32 and bf16 cache; bit-identical alone and batched), and the
// softmax router.
// loop: a CUDA backend whose decode attention runs one block per (token, kv head), walking every split
void test_dense_attention(Report &r, Backend &cpu, Backend &gpu, Backend *loop = nullptr) {
    Rng rng(9090);
    {
        const int64_t N = 5, H = 3, D = 64, stride = H * D + 8;
        const auto x = rng.vec(static_cast<size_t>(N * stride), 1.0f);
        const std::vector<int32_t> pos = {0, 1, 17, 300, 4095};
        std::vector<float> inv(D / 2);
        for (int64_t i = 0; i < D / 2; ++i) inv[i] = 1.0f / std::pow(1e6f, float(2 * i) / float(D));
        check(r, cpu, gpu, "rope", 1e-5, [&](Backend &be) {
            Arena ar(be);
            float *xd = ar.up(x);
            be.rope(xd, ar.up(pos), ar.up(inv), N, H, D, stride);
            return ar.down(xd, x.size());
        });
    }
    const int64_t Hq = 8, Hk = 2, D = 128, page = 16, len0 = 2600, len1 = 40;
    const int64_t mp = (len0 + page - 1) / page, W = Hk * D;
    std::vector<int32_t> pt(2 * mp, 0);
    for (int64_t i = 0; i < mp; ++i) pt[i] = static_cast<int32_t>(mp - 1 - i);          // slot 0, reversed
    for (int64_t i = 0; i < (len1 + page - 1) / page; ++i) pt[mp + i] = static_cast<int32_t>(mp + i);  // slot 1
    const int64_t n_pages = mp + (len1 + page - 1) / page;
    const auto kc = rng.vec(static_cast<size_t>(n_pages * page * W), 1.0f), vc = rng.vec(static_cast<size_t>(n_pages * page * W), 1.0f);
    std::vector<uint16_t> kb(kc.size()), vb(vc.size());
    for (size_t i = 0; i < kc.size(); ++i) kb[i] = bf16_rne(kc[i]), vb[i] = bf16_rne(vc[i]);
    struct Q {
        const char *name;
        std::vector<int32_t> slot, pos;
    };
    std::vector<Q> sets = {{"decode", {0, 1, 0, 0}, {2599, 39, 31, 2047}}, {"prefill", {}, {}}};
    for (int32_t p = 0; p < 40; ++p) sets[1].slot.push_back(1), sets[1].pos.push_back(p);
    for (const Q &qs : sets) {
        const int64_t N = static_cast<int64_t>(qs.slot.size());
        const auto q = rng.vec(static_cast<size_t>(N * Hq * D), 0.3f);
        for (DType dt : {DType::F32, DType::BF16}) {
            auto run = [&](Backend &be, const std::vector<size_t> &which) {
                Arena ar(be);
                std::vector<int32_t> sl, ps;
                std::vector<float> qq;
                for (size_t i : which) {
                    sl.push_back(qs.slot[i]), ps.push_back(qs.pos[i]);
                    qq.insert(qq.end(), q.begin() + int64_t(i) * Hq * D, q.begin() + int64_t(i + 1) * Hq * D);
                }
                const int64_t n = static_cast<int64_t>(which.size());
                const void *k = dt == DType::F32 ? static_cast<const void *>(ar.up(kc)) : static_cast<const void *>(ar.up(kb));
                const void *v = dt == DType::F32 ? static_cast<const void *>(ar.up(vc)) : static_cast<const void *>(ar.up(vb));
                float *o = static_cast<float *>(ar.zeros(sizeof(float) * n * Hq * D));
                be.gqa_attention(o, ar.up(qq), k, v, dt, ar.up(sl), ar.up(ps), ar.up(pt), mp, page, n, Hq, Hk, D,
                                 *std::max_element(ps.begin(), ps.end()), 0.088f);
                return ar.down(o, static_cast<size_t>(n * Hq * D));
            };
            std::vector<size_t> all(static_cast<size_t>(N));
            std::iota(all.begin(), all.end(), 0);
            const std::string name = std::string("gqa_attention/") + qs.name + (dt == DType::F32 ? "/f32" : "/bf16");
            check(r, cpu, gpu, name, 1e-4, [&](Backend &be) { return run(be, all); });
            if (dt == DType::BF16) {  // the rows as one-query sequences of the prefill form (FlashAttention-3)
                check(r, cpu, gpu, name + "/rows", 2e-2, [&](Backend &be) {
                    Arena ar(be);
                    const void *k = ar.up(kb), *v = ar.up(vb);
                    float *o = static_cast<float *>(ar.zeros(sizeof(float) * N * Hq * D));
                    const float *qd = ar.up(q);
                    const int32_t *sd = ar.up(qs.slot), *pd = ar.up(qs.pos), *ptd = ar.up(pt);
                    const int64_t max_pos = *std::max_element(qs.pos.begin(), qs.pos.end());
                    be.set_exact_activations(false);
                    const int done = be.gqa_prefill(o, qd, k, v, dt, nullptr, 0, nullptr, 0, N, true, sd, pd, ptd, mp, page, Hq, Hk, D,
                                                    max_pos, 0.088f);
                    be.set_exact_activations(true);
                    if (done == 0) be.gqa_attention(o, qd, k, v, dt, sd, pd, ptd, mp, page, N, Hq, Hk, D, max_pos, 0.088f);
                    return ar.down(o, static_cast<size_t>(N * Hq * D));
                });
            }
            if (N <= 16) {
                try {
                    std::vector<float> alone;
                    for (size_t i : all) alone = cat(alone, run(gpu, {i}));
                    compare(r, name + " (bitwise alone vs batched)", alone, run(gpu, all), 0.0);
                } catch (const std::exception &e) {
                    r.error(name, e.what());
                }
            }
            if (dt == DType::BF16 && N <= 16) {  // the tensor-core decode kernel (bf16 activations)
                const std::string nt = name + "/tc";
                try {
                    gpu.set_exact_activations(false);
                    const std::vector<float> batched = run(gpu, all);
                    std::vector<float> alone;
                    for (size_t i : all) alone = cat(alone, run(gpu, {i}));
                    gpu.set_exact_activations(true);
                    compare(r, nt, run(cpu, all), batched, 2e-2);
                    compare(r, nt + " (bitwise alone vs batched)", alone, batched, 0.0);
                    if (loop) {
                        loop->set_exact_activations(false);
                        const std::vector<float> walked = run(*loop, all);
                        loop->set_exact_activations(true);
                        compare(r, nt + " (bitwise all splits in one block vs a block per split)", walked, batched, 0.0);
                    }
                } catch (const std::exception &e) {
                    gpu.set_exact_activations(true);
                    if (loop) loop->set_exact_activations(true);
                    r.error(nt, e.what());
                }
            }
        }
    }
    if (has_kv8(gpu)) {  // the fp8 cache: the decode kernel and the prefill form (bf16 staging pages)
       // against the fp32 attention over the values the fp8 pools hold
        std::vector<uint8_t> k8, v8;
        std::vector<float> kq, vq;
        kv8_pool(kc, n_pages, page, Hk, D, k8, kq);
        // values whose magnitudes span 2^-8 .. 2^8 across positions: the P scaling by the values'
        // powers of two relative to the largest (small weights must keep their precision)
        std::vector<float> vw = vc;
        for (int64_t i = 0; i < n_pages * page * Hk; ++i) {
            const float m = std::ldexp(1.0f, static_cast<int>(std::lround(rng.uni() * 8.0f)));
            for (int64_t d = 0; d < D; ++d) vw[static_cast<size_t>(i * D + d)] *= m;
        }
        kv8_pool(vw, n_pages, page, Hk, D, v8, vq);
        for (const Q &qs : sets) {
            const int64_t N = static_cast<int64_t>(qs.slot.size());
            const auto q = rng.vec(static_cast<size_t>(N * Hq * D), 0.3f);
            auto run = [&](Backend &be, const std::vector<size_t> &which, bool f8) {
                Arena ar(be);
                std::vector<int32_t> sl, ps;
                std::vector<float> qq;
                for (size_t i : which) {
                    sl.push_back(qs.slot[i]), ps.push_back(qs.pos[i]);
                    qq.insert(qq.end(), q.begin() + int64_t(i) * Hq * D, q.begin() + int64_t(i + 1) * Hq * D);
                }
                const int64_t n = static_cast<int64_t>(which.size());
                const void *k = f8 ? static_cast<const void *>(ar.up(k8)) : static_cast<const void *>(ar.up(kq));
                const void *v = f8 ? static_cast<const void *>(ar.up(v8)) : static_cast<const void *>(ar.up(vq));
                float *o = static_cast<float *>(ar.zeros(sizeof(float) * n * Hq * D));
                be.set_exact_activations(!f8);
                be.gqa_attention(o, ar.up(qq), k, v, f8 ? DType::FP8 : DType::F32, ar.up(sl), ar.up(ps), ar.up(pt), mp, page, n,
                                 Hq, Hk, D, *std::max_element(ps.begin(), ps.end()), 0.088f);
                be.set_exact_activations(true);
                return ar.down(o, static_cast<size_t>(n * Hq * D));
            };
            std::vector<size_t> all(static_cast<size_t>(N));
            std::iota(all.begin(), all.end(), 0);
            const std::string name = std::string("gqa_attention/") + qs.name + "/fp8";
            try {
                const std::vector<float> batched = run(gpu, all, true);
                compare(r, name, run(cpu, all, false), batched, 2e-2);
                if (N <= 16) {
                    std::vector<float> alone;
                    for (size_t i : all) alone = cat(alone, run(gpu, {i}, true));
                    compare(r, name + " (bitwise alone vs batched)", alone, batched, 0.0);
                    if (loop) compare(r, name + " (bitwise all splits in one block vs a block per split)", run(*loop, all, true),
                                      batched, 0.0);
                }
            } catch (const std::exception &e) {
                gpu.set_exact_activations(true);
                r.error(name, e.what());
            }
        }
        // prefill chunks: their contexts staged as bf16 pages for the FMHA (else the decode kernel)
        const float pscale = 1.0f / std::sqrt(static_cast<float>(D));
        for (const auto &ps : {std::array<int32_t, 3>{1, 0, 40}, std::array<int32_t, 3>{0, 2560, 40}}) {
            const int32_t slot = ps[0], p0 = ps[1], n = ps[2];
            const auto q = rng.vec(static_cast<size_t>(n * Hq * D), 0.3f);
            std::vector<int32_t> sl(static_cast<size_t>(n), slot), pos(static_cast<size_t>(n)), tiles;
            for (int32_t i = 0; i < n; ++i) pos[static_cast<size_t>(i)] = p0 + i;
            for (int32_t i = 0; i < n; i += 16) tiles.push_back(i), tiles.push_back(std::min(16, n - i));
            const std::string name = std::string("gqa_prefill/") + (p0 ? "long" : "short") + "/fp8";
            auto run = [&](Backend &be, bool f8) {
                Arena ar(be);
                const void *k = f8 ? static_cast<const void *>(ar.up(k8)) : static_cast<const void *>(ar.up(kq));
                const void *v = f8 ? static_cast<const void *>(ar.up(v8)) : static_cast<const void *>(ar.up(vq));
                const DType dt = f8 ? DType::FP8 : DType::F32;
                float *o = static_cast<float *>(ar.zeros(sizeof(float) * n * Hq * D));
                const float *qd = ar.up(q);
                const int32_t *sd = ar.up(sl), *pd = ar.up(pos), *ptd = ar.up(pt);
                be.set_exact_activations(!f8);
                const std::vector<int32_t> segs = {0, n, p0, slot};
                const bool done = be.gqa_prefill(o, qd, k, v, dt, ar.up(tiles), int64_t(tiles.size() / 2), segs.data(), 1, 0, false,
                                                 sd, pd, ptd, mp, page, Hq, Hk, D, p0 + n - 1, pscale) != 0;
                if (!done) be.gqa_attention(o, qd, k, v, dt, sd, pd, ptd, mp, page, n, Hq, Hk, D, p0 + n - 1, pscale);
                be.set_exact_activations(true);
                return ar.down(o, static_cast<size_t>(n * Hq * D));
            };
            try {
                compare(r, name, run(cpu, false), run(gpu, true), 2e-2);
            } catch (const std::exception &e) {
                gpu.set_exact_activations(true);
                r.error(name, e.what());
            }
        }
    }
    if (has_fa3_kv8(gpu)) {
        // FlashAttention-3's fp8 cache (plain e4m3 pages, scale 1), its fp8 kernel with Q and P each
        // split in two e4m3 (the value and its rounding error): the decoding rows (the 64-row tile),
        // prefill chunks, and both in one launch, against the fp32 attention over the values the
        // pools hold
        std::vector<uint8_t> k8(kc.size()), v8(vc.size());
        std::vector<float> kq(kc.size()), vq(vc.size());
        for (size_t i = 0; i < kc.size(); ++i) {
            k8[i] = fullm::f32_to_e4m3(kc[i]), kq[i] = fullm::e4m3_to_f32(k8[i]);
            v8[i] = fullm::f32_to_e4m3(vc[i]), vq[i] = fullm::e4m3_to_f32(v8[i]);
        }
        auto run = [&](Backend &be, bool f8, const std::vector<float> &q, const std::vector<int32_t> &sl,
                       const std::vector<int32_t> &pos, int64_t nd, float sc, int64_t nx = 0) {
            // nd decoding rows, then (nx > 0) a draft check of nx rows, then a chunk
            Arena ar(be);
            const int64_t n = static_cast<int64_t>(sl.size());
            const void *k = f8 ? static_cast<const void *>(ar.up(k8)) : static_cast<const void *>(ar.up(kq));
            const void *v = f8 ? static_cast<const void *>(ar.up(v8)) : static_cast<const void *>(ar.up(vq));
            const DType dt = f8 ? DType::FP8E4M3 : DType::F32;
            float *o = static_cast<float *>(ar.zeros(sizeof(float) * n * Hq * D));
            const float *qd = ar.up(q);
            const int32_t *sd = ar.up(sl), *pd = ar.up(pos), *ptd = ar.up(pt);
            const int64_t max_pos = *std::max_element(pos.begin(), pos.end());
            be.set_exact_activations(!f8);
            int done = 0;
            if (nd == n) {
                done = be.gqa_prefill(o, qd, k, v, dt, nullptr, 0, nullptr, 0, n, true, sd, pd, ptd, mp, page, Hq, Hk, D, max_pos, sc);
            } else {
                std::vector<int32_t> tiles;
                for (int64_t i = nd; i < n; i += 16) tiles.push_back(int32_t(i)), tiles.push_back(int32_t(std::min<int64_t>(16, n - i)));
                std::vector<int32_t> segs;
                if (nx > 0) segs = {int32_t(nd), int32_t(nx), pos[size_t(nd)], sl[size_t(nd)]};
                const int64_t c0 = nd + nx;
                segs.insert(segs.end(), {int32_t(c0), int32_t(n - c0), pos[size_t(c0)], sl[size_t(c0)]});
                done = be.gqa_prefill(o, qd, k, v, dt, ar.up(tiles), int64_t(tiles.size() / 2), segs.data(), int64_t(segs.size() / 4), nd, true, sd, pd,
                                      ptd, mp, page, Hq, Hk, D, max_pos, sc);
            }
            if (!done) {
                if (f8) throw std::runtime_error("FlashAttention-3 declined the fp8 cache");
                be.gqa_attention(o, qd, k, v, dt, sd, pd, ptd, mp, page, n, Hq, Hk, D, max_pos, sc);
            }
            be.set_exact_activations(true);
            return ar.down(o, static_cast<size_t>(n * Hq * D));
        };
        const float pscale = 1.0f / std::sqrt(static_cast<float>(D));
        // queries large enough for peaked attention (score spread ~2, as models have): the scores'
        // error from q rounded once to e4m3 then shows (4-7% here; 1-2% with q split, P not)
        const float qs = 6.0f;
        const std::vector<int32_t> dslot = {0, 1, 0, 0}, dpos = {2599, 39, 31, 2047};
        const auto qd = rng.vec(static_cast<size_t>(dslot.size()) * Hq * D, qs);
        const std::string dn = "gqa_attention/decode/fp8-e4m3 (FA3, split q, P)";
        try {
            compare(r, dn, run(cpu, false, qd, dslot, dpos, 4, pscale), run(gpu, true, qd, dslot, dpos, 4, pscale), 2.5e-2);
        } catch (const std::exception &e) {
            gpu.set_exact_activations(true);
            r.error(dn, e.what());
        }
        // a chunk of 40 at 0 and at 2560; the decoding rows with a chunk of 160 at 2400 (long enough
        // for the bf16 staging pages: two launches); the same with a draft check of 8 rows between
        // (slot 1 from 40: it stays on the e4m3 pages); a chunk of 300 alone (staged, one launch)
        for (const auto &ps : {std::array<int32_t, 5>{1, 0, 40, 0, 0}, std::array<int32_t, 5>{0, 2560, 40, 0, 0},
                               std::array<int32_t, 5>{1, 2400, 160, 1, 0}, std::array<int32_t, 5>{0, 2400, 160, 1, 8},
                               std::array<int32_t, 5>{0, 2100, 300, 0, 0}}) {
            const int32_t slot = ps[0], p0 = ps[1], nc = ps[2];
            const int64_t nx = ps[4], nd = ps[3] ? int64_t(dslot.size()) : 0, n = nd + nx + nc;
            const auto q = rng.vec(static_cast<size_t>(n * Hq * D), qs);
            std::vector<int32_t> sl(static_cast<size_t>(n), slot), pos(static_cast<size_t>(n));
            for (int64_t i = 0; i < nd; ++i) sl[size_t(i)] = dslot[size_t(i)], pos[size_t(i)] = dpos[size_t(i)];
            for (int64_t i = 0; i < nx; ++i) sl[size_t(nd + i)] = 1, pos[size_t(nd + i)] = 40 + int32_t(i);
            for (int32_t i = 0; i < nc; ++i) pos[size_t(nd + nx + i)] = p0 + i;
            const std::string name = std::string("gqa_prefill/") +
                                     (nx ? "mixed+drafts" : nd ? "mixed" : nc >= 128 ? "staged" : p0 ? "long" : "short") +
                                     "/fp8-e4m3 (FA3, split q, P)";
            try {
                compare(r, name, run(cpu, false, q, sl, pos, nd, pscale, nx), run(gpu, true, q, sl, pos, nd, pscale, nx), 2.5e-2);
            } catch (const std::exception &e) {
                gpu.set_exact_activations(true);
                r.error(name, e.what());
            }
        }
    }
    {  // the tensor-core prefill kernel (bf16 activations) against the fp32 attention
        struct PS {
            const char *name;
            int32_t slot, p0, n;
        };
        // the exact 1/sqrt(D), the scale models use (0.088 is 0.5% off for D = 128)
        const float pscale = 1.0f / std::sqrt(static_cast<float>(D));
        for (const PS &ps : {PS{"gqa_prefill/short", 1, 0, 40}, PS{"gqa_prefill/long", 0, 2560, 40}}) {
            const auto q = rng.vec(static_cast<size_t>(ps.n * Hq * D), 0.3f);
            std::vector<int32_t> sl(static_cast<size_t>(ps.n), ps.slot), pos(static_cast<size_t>(ps.n)), tiles;
            for (int32_t i = 0; i < ps.n; ++i) pos[static_cast<size_t>(i)] = ps.p0 + i;
            for (int32_t i = 0; i < ps.n; i += 16) tiles.push_back(i), tiles.push_back(std::min(16, ps.n - i));
            for (DType dt : {DType::F32, DType::BF16}) {
                const std::string name = std::string(ps.name) + (dt == DType::F32 ? "/f32" : "/bf16");
                check(r, cpu, gpu, name, 2e-2, [&](Backend &be) {
                    Arena ar(be);
                    const void *k = dt == DType::F32 ? static_cast<const void *>(ar.up(kc)) : static_cast<const void *>(ar.up(kb));
                    const void *v = dt == DType::F32 ? static_cast<const void *>(ar.up(vc)) : static_cast<const void *>(ar.up(vb));
                    float *o = static_cast<float *>(ar.zeros(sizeof(float) * ps.n * Hq * D));
                    const float *qd = ar.up(q);
                    const int32_t *sd = ar.up(sl), *pd = ar.up(pos), *ptd = ar.up(pt);
                    be.set_exact_activations(false);
                    const std::vector<int32_t> segs = {0, ps.n, ps.p0, ps.slot};
                    const bool done = be.gqa_prefill(o, qd, k, v, dt, ar.up(tiles), int64_t(tiles.size() / 2), segs.data(), 1, 0, false,
                                                     sd, pd, ptd, mp, page, Hq, Hk, D, ps.p0 + ps.n - 1, pscale) != 0;
                    be.set_exact_activations(true);
                    if (!done)
                        be.gqa_attention(o, qd, k, v, dt, sd, pd, ptd, mp, page, ps.n, Hq, Hk, D, ps.p0 + ps.n - 1, pscale);
                    return ar.down(o, static_cast<size_t>(ps.n * Hq * D));
                });
            }
        }
    }
    {  // head_dim 64: the other instantiation of the tensor-core prefill and decode kernels (bf16 cache)
        const int64_t Hq2 = 4, Hk2 = 2, D2 = 64, len = 70, mp2 = (len + page - 1) / page, W2 = Hk2 * D2;
        std::vector<int32_t> pt2(static_cast<size_t>(mp2));
        for (int64_t i = 0; i < mp2; ++i) pt2[static_cast<size_t>(i)] = static_cast<int32_t>(mp2 - 1 - i);
        const auto kc2 = rng.vec(static_cast<size_t>(mp2 * page * W2), 1.0f), vc2 = rng.vec(static_cast<size_t>(mp2 * page * W2), 1.0f);
        std::vector<uint16_t> kb2(kc2.size()), vb2(vc2.size());
        for (size_t i = 0; i < kc2.size(); ++i) kb2[i] = bf16_rne(kc2[i]), vb2[i] = bf16_rne(vc2[i]);
        const auto q2 = rng.vec(static_cast<size_t>(len * Hq2 * D2), 0.3f);
        std::vector<int32_t> sl(static_cast<size_t>(len), 0), ps(static_cast<size_t>(len)), tiles;
        for (int32_t i = 0; i < len; ++i) ps[static_cast<size_t>(i)] = i;
        for (int32_t i = 0; i < len; i += 16) tiles.push_back(i), tiles.push_back(std::min<int32_t>(16, len - i));
        check(r, cpu, gpu, "gqa_prefill/d64/bf16", 2e-2, [&](Backend &be) {
            Arena ar(be);
            const void *k = ar.up(kb2), *v = ar.up(vb2);
            float *o = static_cast<float *>(ar.zeros(sizeof(float) * len * Hq2 * D2));
            const float *qd = ar.up(q2);
            const int32_t *sd = ar.up(sl), *pd = ar.up(ps), *ptd = ar.up(pt2);
            be.set_exact_activations(false);
            const std::vector<int32_t> segs = {0, static_cast<int32_t>(len), 0, 0};
            const bool done = be.gqa_prefill(o, qd, k, v, DType::BF16, ar.up(tiles), int64_t(tiles.size() / 2), segs.data(), 1, 0, false,
                                             sd, pd, ptd, mp2, page, Hq2, Hk2, D2, len - 1, 0.125f) != 0;
            be.set_exact_activations(true);
            if (!done) be.gqa_attention(o, qd, k, v, DType::BF16, sd, pd, ptd, mp2, page, len, Hq2, Hk2, D2, len - 1, 0.125f);
            return ar.down(o, static_cast<size_t>(len * Hq2 * D2));
        });
        const std::vector<int32_t> dsl = {0, 0, 0}, dps = {69, 30, 5};
        const std::vector<float> dq(q2.begin(), q2.begin() + 3 * Hq2 * D2);
        check(r, cpu, gpu, "gqa_attention/d64/bf16/tc", 2e-2, [&](Backend &be) {
            Arena ar(be);
            const void *k = ar.up(kb2), *v = ar.up(vb2);
            float *o = static_cast<float *>(ar.zeros(sizeof(float) * 3 * Hq2 * D2));
            be.set_exact_activations(false);
            be.gqa_attention(o, ar.up(dq), k, v, DType::BF16, ar.up(dsl), ar.up(dps), ar.up(pt2), mp2, page, 3, Hq2, Hk2, D2,
                             69, 0.125f);
            be.set_exact_activations(true);
            return ar.down(o, static_cast<size_t>(3 * Hq2 * D2));
        });
        std::vector<uint8_t> k8, v8;
        std::vector<float> kq, vq;
        kv8_pool(kc2, mp2, page, Hk2, D2, k8, kq);
        kv8_pool(vc2, mp2, page, Hk2, D2, v8, vq);
        if (has_kv8(gpu)) try {
            auto run = [&](Backend &be, bool f8) {
                Arena ar(be);
                const void *k = f8 ? static_cast<const void *>(ar.up(k8)) : static_cast<const void *>(ar.up(kq));
                const void *v = f8 ? static_cast<const void *>(ar.up(v8)) : static_cast<const void *>(ar.up(vq));
                float *o = static_cast<float *>(ar.zeros(sizeof(float) * 3 * Hq2 * D2));
                be.set_exact_activations(!f8);
                be.gqa_attention(o, ar.up(dq), k, v, f8 ? DType::FP8 : DType::F32, ar.up(dsl), ar.up(dps), ar.up(pt2), mp2, page, 3,
                                 Hq2, Hk2, D2, 69, 0.125f);
                be.set_exact_activations(true);
                return ar.down(o, static_cast<size_t>(3 * Hq2 * D2));
            };
            compare(r, "gqa_attention/d64/fp8", run(cpu, false), run(gpu, true), 2e-2);
        } catch (const std::exception &e) {
            gpu.set_exact_activations(true);
            r.error("gqa_attention/d64/fp8", e.what());
        }
    }
    {
        const int64_t N = 6, E = 128, k = 8;
        const auto logits = rng.vec(static_cast<size_t>(N * E), 3.0f);
        for (bool renorm : {true, false}) {
            check(r, cpu, gpu, renorm ? "router_softmax_topk/renorm" : "router_softmax_topk", 1e-5, [&](Backend &be) {
                Arena ar(be);
                int32_t *id = static_cast<int32_t *>(ar.zeros(sizeof(int32_t) * N * k));
                float *wd = static_cast<float *>(ar.zeros(sizeof(float) * N * k));
                be.router_softmax_topk(id, wd, ar.up(logits), N, E, k, renorm);
                return cat(as_float(ar.down(id, static_cast<size_t>(N * k))), ar.down(wd, static_cast<size_t>(N * k)));
            });
        }
    }
}

// The draws: the subject's sample_rows token for token against the reference's (whose base form
// is sample_row_host, the definition), at a small vocabulary and at Qwen3's, over every path of
// the CUDA kernel: greedy; top-k in shared memory (with top-p, alone, k = 1); top-p alone with the
// nucleus among the 1024 largest (peaked rows) and beyond them (flat rows); ties at the top-k
// threshold past shared memory (integer-valued rows); top-k above 1024; no filter.
// LoRA, many adapters at once: rows of three adapters and none, shrink then expand onto a y that
// holds values already, at the shapes of Qwen3-8B's q (4096 -> 4096) and down (12288 -> 4096).
// LoRA groups (Backend::lora): jobs reading one x, their A pools stacked per slot as dense_exec
// lays them out, slot 1's adapter of rank 8 in rank-16 pools
struct LoraCase {
    int64_t in, rank = 16, slots = 3;
    std::vector<int64_t> outs;
    std::vector<int32_t> ranks = {16, 8, 16};
    std::vector<float> scale = {2.0f, 0.5f, 1.25f};
    std::vector<uint16_t> a;                // [slots][n][rank][in]
    std::vector<std::vector<uint16_t>> b;   // per job [slots][out][rank]
    LoraCase(int64_t in_, std::vector<int64_t> outs_, Rng &rng, std::vector<int32_t> ranks_ = {16, 8, 16})
        : in(in_), outs(std::move(outs_)), ranks(std::move(ranks_)) {
        const int64_t n = static_cast<int64_t>(outs.size());
        a.assign(size_t(slots * n * rank * in), 0);
        b.resize(size_t(n));
        for (int64_t t = 0; t < n; ++t) b[size_t(t)].assign(size_t(slots * outs[size_t(t)] * rank), 0);
        for (int64_t s = 0; s < slots; ++s)
            for (int64_t t = 0; t < n; ++t)
                for (int64_t k = 0; k < ranks[size_t(s)]; ++k) {
                    for (int64_t i = 0; i < in; ++i) a[size_t(((s * n + t) * rank + k) * in + i)] = bf16_rne(rng.uni() * 0.05f);
                    for (int64_t o = 0; o < outs[size_t(t)]; ++o)
                        b[size_t(t)][size_t((s * outs[size_t(t)] + o) * rank + k)] = bf16_rne(rng.uni() * 0.05f);
                }
    }
    // the jobs on be (uploaded into ar), outputs ys
    std::vector<Backend::LoraJob> jobs(Arena &ar, const std::vector<float *> &ys) const {
        const int64_t n = static_cast<int64_t>(outs.size());
        const uint16_t *da = ar.up(a);
        const int32_t *dr = ar.up(ranks);
        const float *ds = ar.up(scale);
        std::vector<Backend::LoraJob> js;
        for (int64_t t = 0; t < n; ++t)
            js.push_back({da + t * rank * in, n * rank * in, ar.up(b[size_t(t)]), ds, dr, ys[size_t(t)], outs[size_t(t)], rank});
        return js;
    }
};

// The int4 KV cache (kv_int4.h): steps written through qk_rope_append (prompt chunks, decoding rows,
// a short prompt and drafts as rows, drafts crossing a page, a rollback rewritten, a mixed step),
// each step's attention (gqa_prefill: chunks over bf16 staging, the rest by the decode kernel) against
// the fp32 attention over the values the cache holds for each row (decoded on the host: the ring for
// a row's own page per kv_int4.h's rule, else the pool), side_last, and the pool against what was
// written (the quantiser).
void test_kv4(Report &r, Backend &gpu) {
    using namespace fullm;
    gpu.set_exact_activations(false);
    const bool has = (gpu.caps().kv_formats & kv_bit(DType::KV4)) != 0;
    gpu.set_exact_activations(true);
    if (!has) return;
    constexpr int64_t Hq = 16, Hk = 2, D = 128, page = 16, slots = 3, mp = 16, n_pages = slots * mp + 2, G = Hq / Hk;
    const size_t pb = gpu.kv_page_bytes(DType::KV4, page, Hk, D), extra = gpu.kv_extra_bytes(DType::KV4, slots, Hk, D);
    const size_t pool_bytes = pb * n_pages + extra;
    const int64_t ro = kv4_ring_offset(n_pages, Hk);
    Arena ar(gpu);
    auto *kp = static_cast<uint8_t *>(ar.zeros(pool_bytes)), *vp = static_cast<uint8_t *>(ar.zeros(pool_bytes));
    gpu.set_kv_pages(n_pages);
    std::vector<int32_t> pt(size_t(slots * mp));
    for (int64_t s2 = 0; s2 < slots; ++s2)
        for (int64_t i = 0; i < mp; ++i) pt[size_t(s2 * mp + i)] = int32_t(s2 * mp + (mp - 1 - i));  // reversed: indexing errors show
    const int32_t *ptd = ar.up(pt);
    // RoPE as models have it: q's rows the kernels attend with are rotated (a launch reading q's stale
    // fp32 rows instead of RoPE's output shows), the keys rotated before they are quantised
    std::vector<float> inv(D / 2);
    for (int64_t j = 0; j < D / 2; ++j) inv[size_t(j)] = std::pow(10000.0f, -2.0f * float(j) / float(D));
    const float *invd = ar.up(inv);
    auto rope = [&](float *x, int32_t p) {  // pairs (j, j + D/2), as qk_rope_append rotates
        for (int64_t j = 0; j < D / 2; ++j) {
            const float a = x[j], b = x[j + D / 2], ang = float(p) * inv[size_t(j)];
            x[j] = a * std::cos(ang) - b * std::sin(ang), x[j + D / 2] = b * std::cos(ang) + a * std::sin(ang);
        }
    };
    Rng rng(4242);
    auto bfr = [](float x) {
        const uint32_t u = uint32_t(bf16_rne(x)) << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    };
    std::vector<float> koff(size_t(Hk * D));
    for (float &x : koff) x = rng.uni();  // per-channel offsets, as real keys have
    std::vector<float> tk(size_t(slots * mp * page * Hk * D)), tv(tk.size());  // what each (slot, position) holds
    std::vector<int32_t> len(slots, 0);
    struct Step {
        const char *name;
        std::vector<std::array<int32_t, 2>> rows;  // (slot, position), the planner's order
        int64_t n_dec;
        std::vector<int32_t> segs;  // (first row, count, first position, slot)
    };
    auto run_rows = [](int32_t slot, int32_t p0, int32_t n) {
        std::vector<std::array<int32_t, 2>> v;
        for (int32_t i = 0; i < n; ++i) v.push_back({slot, p0 + i});
        return v;
    };
    auto cat = [](std::vector<std::array<int32_t, 2>> a, const std::vector<std::array<int32_t, 2>> &b) {
        a.insert(a.end(), b.begin(), b.end());
        return a;
    };
    std::vector<Step> steps;
    steps.push_back({"chunks + short prompt", cat(cat(run_rows(0, 0, 40), run_rows(1, 0, 21)), run_rows(2, 0, 10)), 0,
                     {0, 40, 0, 0, 40, 21, 0, 1, 61, 10, 0, 2}});
    steps.push_back({"decoding rows", {{0, 40}, {1, 21}, {2, 10}}, 3, {}});
    steps.push_back({"drafts filling a page", cat({{1, 22}}, run_rows(0, 41, 7)), 1, {1, 7, 41, 0}});
    steps.push_back({"drafts across a page", run_rows(1, 23, 13), 0, {0, 13, 23, 1}});
    steps.push_back({"after a rollback", {{1, 28}}, 1, {}});
    steps.push_back({"mixed", cat(cat({{2, 11}}, run_rows(0, 48, 52)), run_rows(1, 29, 12)), 1, {1, 52, 48, 0, 53, 12, 29, 1}});
    steps.push_back({"decoding rows again", {{0, 100}, {1, 41}, {2, 12}}, 3, {}});
    const float scale = 1.0f / std::sqrt(float(D));
    for (const Step &st : steps) {
        const std::string name = std::string("kv4/") + st.name;
        try {
            const int64_t N = int64_t(st.rows.size()), n_segs = int64_t(st.segs.size() / 4);
            std::vector<int32_t> sl(static_cast<size_t>(N)), ps(static_cast<size_t>(N));
            int32_t max_pos = 0;
            for (int64_t i = 0; i < N; ++i) sl[size_t(i)] = st.rows[size_t(i)][0], ps[size_t(i)] = st.rows[size_t(i)][1], max_pos = std::max(max_pos, ps[size_t(i)]);
            std::vector<float> q(size_t(N * Hq * D)), k(size_t(N * Hk * D)), v(k.size());
            for (float &x : q) x = 3.0f * rng.uni();
            for (size_t i = 0; i < k.size(); ++i) k[i] = bfr(koff[i % size_t(Hk * D)] + 0.5f * rng.uni()), v[i] = bfr(rng.uni());
            std::vector<float> qrot = q, krot = k;  // what the kernels see: rotated (keys in bf16 rows)
            for (int64_t i = 0; i < N; ++i) {
                for (int64_t h = 0; h < Hq; ++h) rope(qrot.data() + (i * Hq + h) * D, ps[size_t(i)]);
                for (int64_t h = 0; h < Hk; ++h) {
                    float *kr = krot.data() + (i * Hk + h) * D;
                    rope(kr, ps[size_t(i)]);
                    for (int64_t d = 0; d < D; ++d) kr[d] = bfr(kr[d]);
                }
            }
            std::vector<int32_t> tiles;
            for (int64_t j = 0; j < n_segs; ++j)
                for (int32_t i = 0; i < st.segs[size_t(4 * j + 1)]; i += 16)
                    tiles.push_back(st.segs[size_t(4 * j)] + i), tiles.push_back(std::min(16, st.segs[size_t(4 * j + 1)] - i));
            float *qd = ar.up(q);
            const int32_t *sd = ar.up(sl), *pd = ar.up(ps), *td = ar.up(tiles.empty() ? std::vector<int32_t>{0} : tiles);
            float *o = static_cast<float *>(ar.zeros(sizeof(float) * size_t(N * Hq * D)));
            gpu.set_exact_activations(false);
            if (!gpu.qk_rope_append(qd, ar.up(k), ar.up(v), kp, vp, DType::KV4, nullptr, nullptr, 1e-6f, invd, sd, pd, ptd, mp, page, N, Hq,
                                    Hk, D))
                throw std::runtime_error("no fused append");
            const int done = gpu.gqa_prefill(o, qd, kp, vp, DType::KV4, td, int64_t(tiles.size() / 2), n_segs ? st.segs.data() : nullptr,
                                             n_segs, st.n_dec, true, sd, pd, ptd, mp, page, Hq, Hk, D, max_pos, scale);
            gpu.set_exact_activations(true);
            if (done != 2) throw std::runtime_error("gqa_prefill left rows");
            const auto got = ar.down(o, size_t(N * Hq * D));
            const auto kh = ar.down(kp, pool_bytes), vh = ar.down(vp, pool_bytes);
            for (int64_t i = 0; i < N; ++i) {
                const int32_t s2 = sl[size_t(i)], p = ps[size_t(i)];
                for (int64_t h = 0; h < Hk; ++h)
                    for (int64_t d = 0; d < D; ++d) {
                        const size_t o2 = (size_t(s2 * mp * page + p) * Hk + size_t(h)) * D + size_t(d);
                        tk[o2] = krot[size_t((i * Hk + h) * D + d)], tv[o2] = v[size_t((i * Hk + h) * D + d)];
                    }
                len[size_t(s2)] = std::max(len[size_t(s2)], p + 1);
            }
            // side_last: each slot's last row's page
            std::vector<int32_t> side(slots);
            std::memcpy(side.data(), kh.data() + ro, sizeof(int32_t) * slots);
            for (int64_t i = 0; i < N; ++i)
                if (i + 1 == N || sl[size_t(i + 1)] != sl[size_t(i)] || ps[size_t(i + 1)] != ps[size_t(i)] + 1)
                    if (side[size_t(sl[size_t(i)])] != ps[size_t(i)] / page) throw std::runtime_error("side_last differs");
            // the staged span: from the first segment longer than a page to the last
            int64_t r0 = -1, r1 = -1;
            for (int64_t j = 0; j < n_segs; ++j)
                if (st.segs[size_t(4 * j + 1)] > page) {
                    if (r0 < 0) r0 = st.segs[size_t(4 * j)];
                    r1 = st.segs[size_t(4 * j)] + st.segs[size_t(4 * j + 1)];
                }
            // the values row i sees at (slot, position x), head h: the ring or the pool
            auto seen = [&](int64_t i, int32_t s2, int32_t x, int64_t h, float *kx, float *vx) {
                const int32_t p = ps[size_t(i)], lp = x / int32_t(page), rr = x % int32_t(page);
                const bool staged = i >= r0 && i < r1;
                const bool ring = staged ? lp == side[size_t(s2)] : (lp == p / int32_t(page) && lp >= side[size_t(s2)] - 1);
                if (ring) {
                    const uint8_t *ke = kh.data() + ro + kv4_side(s2, lp & 1, h, Hk), *ve = vh.data() + ro + kv4_side(s2, lp & 1, h, Hk);
                    for (int c = 0; c < D; ++c) kx[c] = kv4_f16(kv4_ld16(ke + 2 * kv4_kside(rr, c))), vx[c] = kv4_f16(kv4_ld16(ve + 2 * kv4_vside(rr, c)));
                } else {
                    const int64_t pg = pt[size_t(s2 * mp + lp)];
                    const uint8_t *kb = kh.data() + (pg * Hk + h) * kKv4Block, *vb = vh.data() + (pg * Hk + h) * kKv4Block;
                    for (int c = 0; c < D; ++c) kx[c] = kv4_key(kb, vb, rr, c), vx[c] = kv4_value_rot(kb, vb, rr, c);
                }
                kv4_hadamard(vx);
            };
            std::vector<float> ref(size_t(N * Hq * D), 0.0f);
            std::vector<float> kx(D), vx(D);
            for (int64_t i = 0; i < N; ++i) {
                const int32_t s2 = sl[size_t(i)], p = ps[size_t(i)];
                for (int64_t h = 0; h < Hk; ++h) {
                    std::vector<float> K(size_t((p + 1) * D)), V(K.size());
                    for (int32_t x = 0; x <= p; ++x) {
                        seen(i, s2, x, h, kx.data(), vx.data());
                        std::copy(kx.begin(), kx.end(), K.begin() + x * D), std::copy(vx.begin(), vx.end(), V.begin() + x * D);
                    }
                    for (int64_t gq = 0; gq < G; ++gq) {
                        const float *qr = qrot.data() + (i * Hq + h * G + gq) * D;
                        std::vector<double> sc(size_t(p + 1));
                        double mx = -1e300, l = 0.0;
                        for (int32_t x = 0; x <= p; ++x) {
                            double a = 0.0;
                            for (int c = 0; c < D; ++c) a += double(qr[c]) * K[size_t(x * D + c)];
                            sc[size_t(x)] = a * scale, mx = std::max(mx, sc[size_t(x)]);
                        }
                        for (auto &x2 : sc) x2 = std::exp(x2 - mx), l += x2;
                        float *orow = ref.data() + (i * Hq + h * G + gq) * D;
                        for (int32_t x = 0; x <= p; ++x)
                            for (int c = 0; c < D; ++c) orow[c] += float(sc[size_t(x)] / l * V[size_t(x * D + c)]);
                    }
                }
            }
            compare(r, name, ref, got, 2e-2);
        } catch (const std::exception &e) {
            gpu.set_exact_activations(true);
            r.error(name, e.what());
            return;
        }
    }
    // the quantiser: every page each slot has filled, decoded, against what was written
    try {
        const auto kh = ar.down(kp, pool_bytes), vh = ar.down(vp, pool_bytes);
        std::vector<float> rk, gk, rv, gv;
        for (int64_t s2 = 0; s2 < slots; ++s2)
            for (int64_t lp = 0; lp < len[size_t(s2)] / page; ++lp)
                for (int64_t h = 0; h < Hk; ++h) {
                    const int64_t pg = pt[size_t(s2 * mp + lp)];
                    const uint8_t *kb = kh.data() + (pg * Hk + h) * kKv4Block, *vb = vh.data() + (pg * Hk + h) * kKv4Block;
                    for (int rr = 0; rr < page; ++rr) {
                        float vr[D];
                        for (int c = 0; c < D; ++c) vr[c] = kv4_value_rot(kb, vb, rr, c);
                        kv4_hadamard(vr);
                        for (int c = 0; c < D; ++c) {
                            const size_t o2 = (size_t(s2 * mp * page + lp * page + rr) * Hk + size_t(h)) * D + size_t(c);
                            rk.push_back(tk[o2]), gk.push_back(kv4_key(kb, vb, rr, c)), rv.push_back(tv[o2]), gv.push_back(vr[c]);
                        }
                    }
                }
        compare(r, "kv4/quantised keys (int4 per channel over a page)", rk, gk, 0.1);
        compare(r, "kv4/quantised values (rotated, int4 per position)", rv, gv, 0.5);  // random values: amax / 14 a value, spread back by the rotation
    } catch (const std::exception &e) {
        r.error("kv4/quantiser", e.what());
    }
}

void test_lora(Report &r, Backend &cpu, Backend &gpu) {
    Rng rng(4321);
    // rows: mixed slots (every group of four differs), runs of one slot (a decode step's sequences
    // on one adapter: the groups share A and B), one row, a big decode step
    struct Rows {
        const char *name;
        int64_t N;
        int pattern;
    };
    auto slots_of = [](int64_t N, int pattern) {
        std::vector<int32_t> slot(static_cast<size_t>(N));
        for (int64_t i = 0; i < N; ++i)
            slot[size_t(i)] = pattern == 0 ? (i % 5 == 4 ? -1 : int32_t(i % 3)) : (i % 13 == 12 ? -1 : int32_t(i / 8 % 3));
        return slot;
    };
    struct Shape {
        const char *name;
        int64_t in;
        std::vector<int64_t> outs;
    };
    // q/k/v-like, down-like, an odd shape (the any-shape kernels), and adapters of rank 8 in
    // rank-16 pools with the work bound at 8 (rank: the loaded adapters' largest)
    const Shape shapes[] = {{"qkv", 4096, {4096, 1024, 1024}}, {"down", 12288, {4096}}, {"odd", 64, {40, 24}},
                            {"qkv-r8", 4096, {4096, 1024, 1024}}};
    for (const Shape &sh : shapes) {
        const bool r8 = std::string(sh.name) == "qkv-r8";
        const LoraCase lc(sh.in, sh.outs, rng, r8 ? std::vector<int32_t>{8, 8, 0} : std::vector<int32_t>{16, 8, 16});
        const int64_t bound = r8 ? 8 : lc.rank;
        for (const Rows c : {Rows{"mixed", 37, 0}, Rows{"runs", 37, 1}, Rows{"one", 1, 1}, Rows{"256", 256, 1}}) {
            const int64_t N = c.N, n = static_cast<int64_t>(sh.outs.size());
            const auto slot = slots_of(N, c.pattern);
            const auto x = rng.vec(size_t(N * sh.in), 1.0f);
            std::vector<std::vector<float>> y0;
            for (int64_t o : sh.outs) y0.push_back(rng.vec(size_t(N * o), 1.0f));
            check(r, cpu, gpu, std::string("lora/") + sh.name + "/" + c.name, 1e-5, [&](Backend &be) {
                Arena ar(be);
                std::vector<float *> ys;
                for (const auto &v : y0) ys.push_back(ar.up(v));
                const auto js = lc.jobs(ar, ys);
                float *xa = static_cast<float *>(ar.zeros(sizeof(float) * size_t(N * n * lc.rank)));
                be.lora(js.data(), int(n), ar.up(x), N, sh.in, bound, ar.up(slot), xa);
                std::vector<float> out;
                for (int64_t t = 0; t < n; ++t) out = cat(out, ar.down(ys[size_t(t)], size_t(N * sh.outs[size_t(t)])));
                return out;
            });
        }
    }
    // lora_rows (the subject's GEMM form for a run of rows on one adapter) against the per-row
    // form: rows [5, 285) of 300 on slot 1; the updates compared, x and A x rounded to bf16
    {
        const LoraCase lc(4096, {4096, 1024, 1024}, rng);
        const int64_t N = 300, M = 280, row0 = 5, n = 3;
        const auto x = rng.vec(size_t(N * lc.in), 1.0f);
        std::vector<std::vector<float>> y0;
        for (int64_t o : lc.outs) y0.push_back(rng.vec(size_t(N * o), 1.0f));
        std::vector<int32_t> slot(static_cast<size_t>(N), -1);
        for (int64_t i = row0; i < row0 + M; ++i) slot[size_t(i)] = 1;
        check(r, cpu, gpu, "lora_rows/qkv", 1e-2, [&](Backend &be) {
            Arena ar(be);
            std::vector<float *> ys;
            for (const auto &v : y0) ys.push_back(ar.up(v));
            const auto js = lc.jobs(ar, ys);
            if (&be == &cpu) {
                float *xa = static_cast<float *>(ar.zeros(sizeof(float) * size_t(N * n * lc.rank)));
                be.lora(js.data(), int(n), ar.up(x), N, lc.in, lc.rank, ar.up(slot), xa);
            } else {
                if (be.lora_rows_min() <= 0 || be.lora_rows_min() > M) throw Absent("lora_rows");
                const int32_t rk[3] = {lc.ranks[1], lc.ranks[1], lc.ranks[1]};
                const float sc[3] = {lc.scale[1], lc.scale[1], lc.scale[1]};
                be.lora_rows(js.data(), int(n), ar.up(x), N, row0, M, lc.in, lc.rank, 1, rk, sc);
            }
            std::vector<float> d;
            for (int64_t t = 0; t < n; ++t) {
                std::vector<float> v = ar.down(ys[size_t(t)], size_t(N * lc.outs[size_t(t)]));
                for (size_t i = 0; i < v.size(); ++i) v[i] -= y0[size_t(t)][i];
                d = cat(d, v);
            }
            return d;
        });
    }
    // on a GEMM's bf16 result: q/k/v from one stacked bf16 weight with y16_ok (the GPU keeps them
    // as one fused bf16 matrix), then the adapters, per row and as a run; against the exact CPU
    gpu.set_exact_activations(false);
    {
        const int64_t N = 96, K = 256, n = 3;
        const std::vector<int64_t> outs = {512, 128, 128};
        const LoraCase lc(K, outs, rng);
        const HostWeight hw = random_weight(WFmt::BF16, 768, K, rng);
        const auto x = rng.vec(size_t(N * K), 1.0f);
        std::vector<int32_t> slot(static_cast<size_t>(N));
        for (int64_t i = 0; i < N; ++i) slot[size_t(i)] = i < 70 ? 2 : int32_t(i % 4) - 1;
        for (bool rows : {false, true}) {
            check(r, cpu, gpu, std::string("lora/on fused bf16 q|k|v") + (rows ? " (run)" : ""), 1e-2, [&](Backend &be) {
                // every upload first: an upload begins a step, which drops the GEMM's bf16 result
                Arena ar(be);
                const Weight w = hw.upload(ar);
                Weight wq[3];
                float *ys[3];
                Backend::GemmJob gj[3];
                const float *xd = ar.up(x);
                int64_t off = 0;
                for (int t = 0; t < 3; ++t) {
                    wq[t] = w, wq[t].rows = outs[size_t(t)];
                    wq[t].data = static_cast<const uint16_t *>(w.data) + off * K;
                    off += outs[size_t(t)];
                    ys[t] = static_cast<float *>(ar.zeros(sizeof(float) * size_t(N * outs[size_t(t)])));
                    gj[t] = Backend::GemmJob{&wq[t], xd, ys[t], outs[size_t(t)], true};
                }
                const auto js = lc.jobs(ar, {ys[0], ys[1], ys[2]});
                float *xa = static_cast<float *>(ar.zeros(sizeof(float) * size_t(N * n * lc.rank)));
                const bool run = rows && &be == &gpu && be.lora_rows_min() > 0 && be.lora_rows_min() <= 70;
                std::vector<int32_t> rest = slot;
                if (run)
                    for (int64_t i = 0; i < 70; ++i) rest[size_t(i)] = -1;
                const int32_t *ds = ar.up(rest);
                be.gemm_group(gj, 3, N);
                if (run) {
                    const int32_t rk[3] = {lc.ranks[2], lc.ranks[2], lc.ranks[2]};
                    const float sc[3] = {lc.scale[2], lc.scale[2], lc.scale[2]};
                    be.lora_rows(js.data(), int(n), xd, N, 0, 70, K, lc.rank, 2, rk, sc);
                }
                be.lora(js.data(), int(n), xd, N, K, lc.rank, ds, xa);
                std::vector<float> out;
                for (int t = 0; t < 3; ++t) out = cat(out, ar.down(ys[t], size_t(N * outs[size_t(t)])));
                return out;
            });
        }
    }
    gpu.set_exact_activations(true);
}

void test_sample(Report &r, Backend &cpu, Backend &gpu) {
    Rng rng(777);
    struct Spec {
        float t, p;
        int32_t k;
        int shape;  // 0 flat, 1 peaked, 2 integer-valued
    };
    const Spec specs[] = {{0.0f, 1.0f, 0, 0},   {0.7f, 0.8f, 20, 0},  {0.6f, 0.95f, 0, 1}, {1.0f, 1.0f, 0, 0},
                          {0.3f, 0.5f, 1, 0},   {1.3f, 0.99f, 50, 0}, {0.7f, 1.0f, 5, 0},  {0.6f, 0.99f, 0, 0},
                          {0.9f, 0.9f, 5, 2},   {0.8f, 0.9f, 5000, 0}, {0.6f, 0.95f, 0, 2}, {1.0f, 0.7f, -1, 1}};
    for (int64_t n : {1000, 151936}) {
        const int64_t R = static_cast<int64_t>(sizeof specs / sizeof specs[0]);
        std::vector<float> logits = rng.vec(static_cast<size_t>(R * n), 4.0f);
        std::vector<fullm::SampleRow> p(static_cast<size_t>(R));
        for (int64_t i = 0; i < R; ++i) {
            const Spec &sp = specs[i];
            float *row = &logits[size_t(i * n)];
            if (sp.shape == 1)  // a few dozen values well above the rest
                for (int j = 0; j < 40; ++j) row[size_t((j * 7919 + 13) % n)] += 12.0f + 0.25f * float(j % 5);
            if (sp.shape == 2)
                for (int64_t j = 0; j < n; ++j) row[j] = std::round(row[j]);
            const int32_t k = sp.k < 0 ? static_cast<int32_t>(n) : std::min<int32_t>(sp.k, static_cast<int32_t>(n));
            p[size_t(i)] = fullm::SampleRow{sp.t, sp.p, k, 0, 1000 + uint64_t(i), uint64_t(7 * i), 0.0f, 0, 0, 0};
        }
        check(r, cpu, gpu, "sample_rows/" + std::to_string(n), 0.0, [&](Backend &be) {
            Arena ar(be);
            int32_t *id = static_cast<int32_t *>(ar.zeros(sizeof(int32_t) * static_cast<size_t>(R + 1)));
            be.sample_rows(id, id + R, ar.up(logits), R, n, ar.up(p));
            return as_float(ar.down(id, static_cast<size_t>(R + 1)));
        });
        // structured output: every other row drawn among a random tenth of the vocabulary (one
        // row: a single token), mask_rows then sample_rows; no masked token may win, and the
        // masked rows' -inf are not non-finite logits
        const int64_t words = (n + 31) / 32;
        std::vector<fullm::SampleRow> pm = p;
        int32_t nm = 0;
        for (int64_t i = 1; i < R; i += 2) pm[size_t(i)].mask = ++nm;
        std::vector<uint32_t> masks(size_t(nm) * size_t(words), 0u);
        for (uint32_t &w : masks)
            for (int b = 0; b < 32; ++b)
                if (rng.uni() < -0.8f) w |= 1u << b;
        std::fill(masks.begin(), masks.begin() + words, 0u);
        masks[size_t(n / 2 / 32)] = 1u << (n / 2 % 32);
        check(r, cpu, gpu, "sample_rows/masked/" + std::to_string(n), 0.0, [&](Backend &be) {
            Arena ar(be);
            int32_t *id = static_cast<int32_t *>(ar.zeros(sizeof(int32_t) * static_cast<size_t>(R + 1)));
            float *x = ar.up(logits);
            const fullm::SampleRow *dp = ar.up(pm);
            be.mask_rows(x, id + R, R, n, dp, ar.up(masks), words);
            be.sample_rows(id, id + R, x, R, n, dp);
            const std::vector<int32_t> got = ar.down(id, static_cast<size_t>(R + 1));
            if (got[size_t(R)]) throw std::runtime_error("masked rows reported non-finite logits");
            for (int64_t i = 0; i < R; ++i) {
                const int32_t k = pm[size_t(i)].mask, t = got[size_t(i)];
                if (k > 0 && !(masks[size_t(k - 1) * size_t(words) + size_t(t >> 5)] >> (t & 31) & 1u))
                    throw std::runtime_error("a masked token was drawn");
            }
            return as_float(got);
        });
        // the rest of the sampling step (step_tokens): adjustments (penalties, logit bias,
        // min_tokens' +inf, on the masked rows too) after the masks, min-p, and the
        // log-probabilities of the logits as given (a copy) for 0..20 alternatives
        std::vector<fullm::SampleRow> pa = pm;
        std::vector<fullm::LogitAdj> adj;
        for (int64_t i = 0; i < R; ++i) {
            fullm::SampleRow &s = pa[size_t(i)];
            if (i % 3 != 2) {
                s.adj_off = static_cast<int32_t>(adj.size());
                std::vector<int32_t> seen;
                for (int j = 0; j < 40; ++j) {
                    const int32_t t = static_cast<int32_t>(rng.next() % uint64_t(n));
                    if (std::find(seen.begin(), seen.end(), t) != seen.end()) continue;
                    seen.push_back(t);
                    const float rep = j % 4 == 0 ? 1.0f : rng.range(0.5f, 2.0f);
                    const float sub = j % 10 == 0 ? INFINITY : rng.range(-1.0f, 2.0f);
                    adj.push_back(fullm::LogitAdj{t, rng.range(-3.0f, 3.0f), rep, sub});
                }
                // the row's largest logit penalised hard: the draw moves
                const int64_t top = std::max_element(&logits[size_t(i * n)], &logits[size_t(i * n)] + n) - &logits[size_t(i * n)];
                if (std::find(seen.begin(), seen.end(), int32_t(top)) == seen.end())
                    adj.push_back(fullm::LogitAdj{static_cast<int32_t>(top), 0.0f, 1.0f, 30.0f});
                s.adj_n = static_cast<int32_t>(adj.size()) - s.adj_off;
            }
            if (i % 4 == 1 && s.temperature > 0.0f) s.min_p_cut = static_cast<float>(double(s.temperature) * std::log(0.05));
            s.logprobs = i % 5 == 0 ? 0 : 1 + static_cast<int32_t>(i % 21);
        }
        auto adjusted = [&](Backend &be, int what) {
            Arena ar(be);
            int32_t *id = static_cast<int32_t *>(ar.zeros(sizeof(int32_t) * static_cast<size_t>(R + 1)));
            float *x = ar.up(logits);
            const float *raw = ar.up(logits);
            const fullm::SampleRow *dp = ar.up(pa);
            be.mask_rows(x, id + R, R, n, dp, ar.up(masks), words);
            be.adjust_rows(x, R, n, dp, ar.up(adj));
            be.sample_rows(id, id + R, x, R, n, dp);
            if (what == 0) {
                const std::vector<int32_t> got = ar.down(id, static_cast<size_t>(R + 1));
                if (got[size_t(R)]) throw std::runtime_error("adjusted rows reported non-finite logits");
                return as_float(got);
            }
            auto *lp = static_cast<float *>(ar.zeros(sizeof(float) * size_t(R) * (fullm::kMaxTopLogprobs + 1)));
            auto *ids = static_cast<int32_t *>(ar.zeros(sizeof(int32_t) * size_t(R) * fullm::kMaxTopLogprobs));
            be.logprob_rows(lp, ids, raw, R, n, dp, id);
            if (what == 1) return ar.down(lp, size_t(R) * (fullm::kMaxTopLogprobs + 1));
            return as_float(ar.down(ids, size_t(R) * fullm::kMaxTopLogprobs));
        };
        check(r, cpu, gpu, "sample_rows/adjusted/" + std::to_string(n), 0.0, [&](Backend &be) { return adjusted(be, 0); });
        check(r, cpu, gpu, "logprob_rows/" + std::to_string(n), 1e-6, [&](Backend &be) { return adjusted(be, 1); });
        check(r, cpu, gpu, "logprob_rows/ids/" + std::to_string(n), 0.0, [&](Backend &be) { return adjusted(be, 2); });
    }
}

// The fused decode ops (CUDA) against the op sequences they replace (CPU): the attention
// prologue (q/k RMSNorm, RoPE, k/v cache append) and the router GEMV + softmax top-k.
// alt: a CUDA backend whose fused router selects in shared memory (the same bits)
void test_fused_dense(Report &r, Backend &cpu, Backend &gpu, Backend *alt = nullptr) {
    Rng rng(4242);
    const int64_t Hq = 6, Hk = 2, page = 16, mp = 4;
    const std::vector<int32_t> slot = {0, 1, 0, 1, 1}, pos = {0, 3, 17, 40, 63};
    std::vector<int32_t> pt(2 * mp);
    for (int64_t i = 0; i < 2 * mp; ++i) pt[i] = static_cast<int32_t>(2 * mp - 1 - i);
    const int64_t N = static_cast<int64_t>(slot.size()), rows = 2 * mp * page;
    for (int64_t D : {32, 64, 128}) {
        std::vector<float> inv(D / 2);
        for (int64_t i = 0; i < D / 2; ++i) inv[i] = 1.0f / std::pow(1e6f, float(2 * i) / float(D));
        const auto q = rng.vec(static_cast<size_t>(N * Hq * D), 1.0f), k = rng.vec(static_cast<size_t>(N * Hk * D), 1.0f),
                   v = rng.vec(static_cast<size_t>(N * Hk * D), 1.0f);
        const auto qn = rng.vec(static_cast<size_t>(D), 1.0f), kn = rng.vec(static_cast<size_t>(D), 1.0f);
        for (DType dt : {DType::F32, DType::BF16})
            for (bool norm : {true, false}) {
                const std::string name = "qk_rope_append/D" + std::to_string(D) + (dt == DType::F32 ? "/f32" : "/bf16") +
                                         (norm ? "/norm" : "");
                check(r, cpu, gpu, name, dt == DType::F32 ? 1e-5 : 1e-2, [&](Backend &be) {
                    Arena ar(be);
                    const size_t cb = static_cast<size_t>(rows * Hk * D) * (dt == DType::F32 ? 4 : 2);
                    float *qd = ar.up(q), *kd = ar.up(k);
                    const float *vd = ar.up(v), *qnd = norm ? ar.up(qn) : nullptr, *knd = norm ? ar.up(kn) : nullptr;
                    void *kc = ar.zeros(cb), *vc = ar.zeros(cb);
                    const int32_t *sl = ar.up(slot), *ps = ar.up(pos), *ptd = ar.up(pt);
                    const float *invd = ar.up(inv);
                    if (!be.qk_rope_append(qd, kd, vd, kc, vc, dt, qnd, knd, 1e-6f, invd, sl, ps, ptd, mp, page, N, Hq,
                                           Hk, D)) {
                        if (norm) {
                            be.rmsnorm(qd, qd, qnd, N * Hq, D, D, 1e-6f);
                            be.rmsnorm(kd, kd, knd, N * Hk, D, D, 1e-6f);
                        }
                        be.rope(qd, ps, invd, N, Hq, D, Hq * D);
                        be.rope(kd, ps, invd, N, Hk, D, Hk * D);
                        be.mla_append(kc, dt, kd, sl, ps, ptd, mp, page, Hk * D, N);
                        be.mla_append(vc, dt, vd, sl, ps, ptd, mp, page, Hk * D, N);
                    }
                    auto cache = [&](const void *c) {
                        const size_t n = static_cast<size_t>(rows * Hk * D);
                        if (dt == DType::F32) return ar.down(static_cast<const float *>(c), n);
                        std::vector<float> f;
                        for (uint16_t h : ar.down(static_cast<const uint16_t *>(c), n)) f.push_back(bf16f(h));
                        return f;
                    };
                    return cat(cat(ar.down(qd, q.size()), cache(kc)), cache(vc));
                });
            }
    }
    for (int64_t D : {32, 64, 128}) {
        if (!has_kv8(gpu)) break;  // the fp8 append: subjects with an fp8 cache
        std::vector<float> inv(D / 2);
        for (int64_t i = 0; i < D / 2; ++i) inv[i] = 1.0f / std::pow(1e6f, float(2 * i) / float(D));
        const auto q = rng.vec(static_cast<size_t>(N * Hq * D), 1.0f), k = rng.vec(static_cast<size_t>(N * Hk * D), 1.0f),
                   v = rng.vec(static_cast<size_t>(N * Hk * D), 0.05f);
        const auto kn = rng.vec(static_cast<size_t>(D), 1.0f);
        const int64_t n_pages = 2 * mp;
        auto run = [&](DType dt) {  // the caches' values
            Arena ar(gpu);
            const size_t cb = dt == DType::FP8 ? static_cast<size_t>(n_pages * kv8_page_bytes(page, Hk, D))
                                               : static_cast<size_t>(rows * Hk * D) * 4;
            float *qd = ar.up(q), *kd = ar.up(k);
            void *kc = ar.zeros(cb), *vc = ar.zeros(cb);
            gpu.set_exact_activations(dt == DType::F32);
            const bool ok = gpu.qk_rope_append(qd, kd, ar.up(v), kc, vc, dt, nullptr, ar.up(kn), 1e-6f, ar.up(inv), ar.up(slot),
                                               ar.up(pos), ar.up(pt), mp, page, N, Hq, Hk, D);
            gpu.set_exact_activations(true);
            if (!ok) throw std::runtime_error("no fused append");
            if (dt == DType::F32) {
                return std::make_pair(ar.down(static_cast<const float *>(kc), static_cast<size_t>(rows * Hk * D)),
                                      ar.down(static_cast<const float *>(vc), static_cast<size_t>(rows * Hk * D)));
            }
            const auto kb = ar.down(static_cast<const uint8_t *>(kc), cb), vb = ar.down(static_cast<const uint8_t *>(vc), cb);
            return std::make_pair(kv8_values(kb, n_pages, page, Hk, D), kv8_values(vb, n_pages, page, Hk, D));
        };
        const std::string name = "qk_rope_append/D" + std::to_string(D) + "/fp8";
        try {
            const auto f = run(DType::F32), g = run(DType::FP8);
            std::vector<uint8_t> pool;
            std::vector<float> kw, vw;
            kv8_pool(f.first, n_pages, page, Hk, D, pool, kw);
            kv8_pool(f.second, n_pages, page, Hk, D, pool, vw);
            compare(r, name + " (bitwise vs the host rule on the fp32 append)", cat(kw, vw), cat(g.first, g.second), 0.0);
        } catch (const std::exception &e) {
            gpu.set_exact_activations(true);
            r.error(name, e.what());
        }
    }
    for (int64_t width : {256, 2048}) {
        const int64_t E = 128, k = 8;
        const HostWeight hw = random_weight(WFmt::BF16, E, width, rng);
        for (int64_t n : {1, 6, 16}) {
            const auto x = rng.vec(static_cast<size_t>(n * width), 1.0f);
            for (bool renorm : {true, false}) {
                auto run = [&](Backend &be, bool want_idx) {
                    Arena ar(be);
                    const Weight wt = hw.upload(ar);
                    const float *xd = ar.up(x);
                    int32_t *id = static_cast<int32_t *>(ar.zeros(sizeof(int32_t) * n * k));
                    float *wd = static_cast<float *>(ar.zeros(sizeof(float) * n * k));
                    if (!be.router_softmax_fused(id, wd, wt, xd, n, k, renorm)) {
                        float *lg = static_cast<float *>(ar.zeros(sizeof(float) * n * E));
                        be.gemm(wt, xd, n, lg, E);
                        be.router_softmax_topk(id, wd, lg, n, E, k, renorm);
                    }
                    if (want_idx) return as_float(ar.down(id, static_cast<size_t>(n * k)));
                    return ar.down(wd, static_cast<size_t>(n * k));
                };
                const std::string name = "router_softmax_fused/K" + std::to_string(width) + "/N" + std::to_string(n) +
                                         (renorm ? "/renorm" : "");
                check(r, cpu, gpu, name + "/idx", 0.0, [&](Backend &be) { return run(be, true); });
                check(r, cpu, gpu, name + "/w", 1e-5, [&](Backend &be) { return run(be, false); });
                if (alt) {
                    try {
                        compare(r, name + " (bitwise top-k in registers vs shared memory)",
                                cat(run(gpu, true), run(gpu, false)), cat(run(*alt, true), run(*alt, false)), 0.0);
                    } catch (const std::exception &e) {
                        r.error(name, e.what());
                    }
                }
            }
        }
    }
}

void test_router(Report &r, Backend &cpu, Backend &gpu) {
    Rng rng(31337);
    struct Case {
        const char *name;
        int64_t N, E, k;
        bool renorm;
        float scale;
        std::vector<float> logits, bias;
    };
    std::vector<Case> cases;
    {
        Case c{"router/e896_k16", 5, 896, 16, true, 2.5f, rng.vec(5 * 896, 4.0f), rng.vec(896, 0.1f)};
        cases.push_back(c);
    }
    {
        Case c{"router/all_ties", 2, 64, 8, false, 1.0f, std::vector<float>(2 * 64, 0.0f), std::vector<float>(64, 0.0f)};
        cases.push_back(c);
    }
    {  // repeated values scattered through the row
        Case c{"router/some_ties", 3, 128, 8, true, 1.0f, {}, std::vector<float>(128, 0.0f)};
        c.logits.resize(3 * 128);
        for (float &v : c.logits) v = static_cast<float>(static_cast<int>(rng.next() % 5));
        cases.push_back(c);
    }
    for (const Case &c : cases) {
        auto run = [&](Backend &be, bool want_idx) {
            Arena ar(be);
            int32_t *id = ar.up(std::vector<int32_t>(static_cast<size_t>(c.N * c.k), -1));
            float *wd = ar.up(std::vector<float>(static_cast<size_t>(c.N * c.k), 0.0f));
            be.router_topk(id, wd, ar.up(c.logits), ar.up(c.bias), c.N, c.E, c.k, c.renorm, c.scale, false);
            if (want_idx) return as_float(ar.down(id, static_cast<size_t>(c.N * c.k)));
            return ar.down(wd, static_cast<size_t>(c.N * c.k));
        };
        check(r, cpu, gpu, std::string(c.name) + "/idx", 0.0, [&](Backend &be) { return run(be, true); });
        check(r, cpu, gpu, std::string(c.name) + "/w", 1e-6, [&](Backend &be) { return run(be, false); });
    }
}

// Sets an environment variable for its lifetime (restores the old value, or unsets it).
struct EnvOverride {
    std::string name, old;
    bool had;
    EnvOverride(const char *n, const char *v) : name(n) {
        const char *o = std::getenv(n);
        had = o != nullptr;
        if (o) old = o;
        setenv(n, v, 1);
    }
    ~EnvOverride() {
        if (had) setenv(name.c_str(), old.c_str(), 1);
        else unsetenv(name.c_str());
    }
    EnvOverride(const EnvOverride &) = delete;
    EnvOverride &operator=(const EnvOverride &) = delete;
};

// The suite on subject against reference (backends by name). The CUDA backend also gets its
// tuning variants (the knobs are read when a backend is made): expert grids of 3 blocks with the
// prefetch on and off, two-kernel experts, one-block decode attention, the router's top-k in
// shared memory.
int conformance(const char *subject_name, const char *reference_name, int32_t device, std::string &text) {
    Report r;
    // the int4 GEMM's small tiles in every call averaging up to 64 tokens an expert (24 when
    // serving), so the moe cases reach its 32-, 64- and 128-token kernels
    const EnvOverride i4small("FULLM_INT4_GEMM_SMALL", "64");
    const std::string sub_name = subject_name ? subject_name : "cuda", ref_name = reference_name ? reference_name : "cpu";
    for (const fullm::BackendEntry &e : fullm::backend_registry())
        if (sub_name == e.name && !e.available(device)) {
            text = "the " + sub_name + " backend cannot run here\n";
            return -1;
        }
    auto cpu = fullm::make_backend(ref_name, device, 0);
    auto gpu = fullm::make_backend(sub_name, device, 0);
    std::unique_ptr<fullm::Backend> small_on, small_off, split;
    if (sub_name == "cuda") {
        EnvOverride fused("FULLM_MOE_FUSED", "1");
        {
            EnvOverride grid("FULLM_MOE_GRID", "3"), pf("FULLM_MOE_PREFETCH", "1"), ab("FULLM_ATTN_BLOCKS", "1"),
                rr("FULLM_ROUTER_REGS", "0");
            small_on = fullm::make_backend("cuda", device, 0);
            EnvOverride pf0("FULLM_MOE_PREFETCH", "0");
            small_off = fullm::make_backend("cuda", device, 0);
        }
        EnvOverride two("FULLM_MOE_FUSED", "0");
        split = fullm::make_backend("cuda", device, 0);
    }
    r.text = std::string("selftest: ") + gpu->name() + " vs " + cpu->name() + "\n";
    const uint32_t fam = gpu->caps().families;
    std::string skipped;
    test_gemm(r, *cpu, *gpu);
    test_elementwise(r, *cpu, *gpu);
    test_sample(r, *cpu, *gpu);
    test_lora(r, *cpu, *gpu);
    if (fam & FAM_KDA) test_kda(r, *cpu, *gpu), test_kda_fused(r, *cpu, *gpu), test_kda_chunked(r, *cpu, *gpu);
    else skipped += " kda";
    if (fam & FAM_MLA) test_mla(r, *cpu, *gpu), test_mla_invariance(r, *cpu, *gpu), test_mla_prefill(r, *cpu, *gpu);
    else skipped += " mla";
    test_add_rmsnorm(r, *cpu, *gpu);
    if (fam & FAM_GQA) test_dense_attention(r, *cpu, *gpu, small_on.get()), test_kv4(r, *gpu);
    else skipped += " gqa";
    test_gemm_group(r, *cpu, *gpu);
    test_bf16_activations(r, *cpu, *gpu);
    test_mlp_bf16(r, *cpu, *gpu);
    test_fp8_gemm(r, *cpu, *gpu);
    test_split_mlp(r, *gpu);
    if (fam & FAM_KIMI) test_router(r, *cpu, *gpu);
    else skipped += " kimi";
    if (fam & FAM_GQA) test_fused_dense(r, *cpu, *gpu, small_on.get());
    test_moe(r, *cpu, *gpu, SmallGrid{small_on.get(), small_off.get(), split.get()});
    if (!skipped.empty()) r.text += "skipped (families the subject does not declare):" + skipped + "\n";
    r.text += r.fails ? std::to_string(r.fails) + " FAILED\n" : "all ops PASS\n";
    text = r.text;
    return r.fails;
}

}  // namespace

static int report_out(int rc, const std::string &text, char *report, int report_len) {
    if (report && report_len > 0) {
        const size_t n = std::min(text.size(), static_cast<size_t>(report_len) - 1);
        std::memcpy(report, text.data(), n);
        report[n] = 0;
    }
    return rc;
}

extern "C" int sk_conformance(const char *subject, const char *reference, int32_t device, char *report, int report_len) {
    std::string text;
    int rc = -1;
    try {
        rc = conformance(subject, reference, device, text);
    } catch (const std::exception &e) {
        text += std::string("selftest aborted: ") + e.what() + "\n";
        rc = -1;
    }
    return report_out(rc, text, report, report_len);
}

extern "C" int sk_exec_selftest(int32_t device, char *report, int report_len) {
    if (!fullm::cuda_available()) return report_out(-1, "CUDA is not available\n", report, report_len);
    return sk_conformance("cuda", "cpu", device, report, report_len);
}
