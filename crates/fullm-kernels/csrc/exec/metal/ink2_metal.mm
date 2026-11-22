// SPDX-License-Identifier: Apache-2.0
// ink2_metal.mm - see ink2_metal.h. The graph is ink2_exec.cpp's (the CPU path), dispatch by dispatch.
//
// What shapes the kernels (measured on an M3 Pro):
//   - a token's weights are 5.4 GB to sweep (bf16 matrices 2.6, experts 1.8, the output head 0.9)
//     and the GPU sweeps 110 GB/s: decode is bound by memory, so every matrix is read once, in the
//     format it is stored in (bf16 two to a word, NVFP4 / MXFP8 decoded as read);
//   - a dispatch costs ~1.5 us, but one thread walks ~12 ns an element: no kernel loops over more
//     than a few hundred elements in a thread. Norms and softmaxes over long rows are two or three
//     dispatches (partial sums by chunk, then the use);
//   - tiles: a thread takes one matrix row against up to TT rows of activations (TT = 16; 8 for an
//     expert's last few tokens; 1 for a lone token), accumulating in registers. Weights a 32-bit
//     word at a time, activations as half4, four multiply-adds a dot(): 2.5 TFLOPS. Any larger
//     local array spills (a 4 x 8 tile ran at a tenth).
// Activations entering a matrix are half (normalised rows, softcapped GLU outputs: far inside its
// range), sums and everything else f32. The values of a prompt chunk's attention are multiplied
// by MPS (P V); every other result of a row depends on its own sequence only.

#import <Foundation/Foundation.h>
#import <QuartzCore/QuartzCore.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <unistd.h>

#include "fullm_kernels.h"
#include "ink2_metal.h"

namespace fullm {

namespace {

NSString *const kSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

// TT (rows of activations a thread takes), ACT / ACT4 (activations: half or float) and KV / KV4 (the
// cache pools' element: half or float) are given at compile time
struct P { uint a[16]; float f[4]; };

#define W2F(a, b) as_type<float4>(uint4((a) << 16, (a) & 0xFFFF0000u, (b) << 16, (b) & 0xFFFF0000u))
inline float hsum(float4 v) { return v.x + v.y + v.z + v.w; }
inline float bf(ushort u) { return as_type<float>(uint(u) << 16); }
inline float sigm(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float4 sigm4(float4 x) { return 1.0f / (1.0f + exp(-x)); }

// the row of a (slot, position, kv head) in a pool [page][kv head][position in page]
inline uint kvrow(device const int *pt, uint slot, uint pos, uint kh, uint Hk, uint page, uint maxp) {
    return (uint(pt[slot * maxp + pos / page]) * Hk + kh) * page + pos % page;
}

// ---- dense matrices -------------------------------------------------------------------------
// y[s + t][n] = W[n] . x[s + t]: thread (n, tile). a: K, ldy, N; f0: softcap (0: none). The
// products in the activations' type (half: 11 bits each, against bf16's 8), the sums in float.
kernel void gemm(constant P &p [[buffer(0)]], device const uint *w [[buffer(1)]], device const ACT4 *x [[buffer(2)]],
                 device float *y [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    const uint K4 = p.a[0] / 4, ldy = p.a[1], N = p.a[2], s = gid.y * TT, wo = gid.x * (p.a[0] / 2);
    float acc[TT];
    for (uint t = 0; t < TT; ++t) acc[t] = 0.0f;
    for (uint i = 0; i < K4; ++i) {
        const uint a = w[wo + 2 * i], b = w[wo + 2 * i + 1];
        const ACT4 wv = ACT4(W2F(a, b));
        const uint xo = s * K4 + i;
        for (uint t = 0; t < TT; ++t) {
            const float4 pr = float4(wv * x[xo + t * K4]);
            acc[t] += (pr.x + pr.y) + (pr.z + pr.w);
        }
    }
    const uint m = min(uint(TT), N - s);
    const float cap = p.f[0];
    for (uint t = 0; t < m; ++t) y[(s + t) * ldy + gid.x] = cap > 0.0f ? tanh(acc[t] / cap) * cap : acc[t];
}

// A few tiles of activations: a thread walks ~12 ns an element, and a few tiles leave the GPU a
// thread or so a matrix row, so the row is cut into chunks of 256 columns (thread (n, chunk, tile)
// -> part[tile][n][chunk][t]) and gsumt adds them up. a: K, nc, -, rows
kernel void gemmc(constant P &p [[buffer(0)]], device const uint *w [[buffer(1)]], device const ACT4 *x [[buffer(2)]],
                  device float *part [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) {
    const uint K4 = p.a[0] / 4, c0 = gid.y * 64, c1 = min(K4, c0 + 64), wo = gid.x * (p.a[0] / 2), xb = gid.z * TT * K4;
    float acc[TT];
    for (uint t = 0; t < TT; ++t) acc[t] = 0.0f;
    for (uint i = c0; i < c1; ++i) {
        const uint a = w[wo + 2 * i], b = w[wo + 2 * i + 1];
        const ACT4 wv = ACT4(W2F(a, b));
        for (uint t = 0; t < TT; ++t) {
            const float4 pr = float4(wv * x[xb + i + t * K4]);
            acc[t] += (pr.x + pr.y) + (pr.z + pr.w);
        }
    }
    const uint o = ((gid.z * p.a[3] + gid.x) * p.a[1] + gid.y) * TT;
    for (uint t = 0; t < TT; ++t) part[o + t] = acc[t];
}
// y[r][n] = the sum of row n's chunks for activation row r: thread (n, r). a[1]: nc, a[2]: ldy,
// a[3]: rows; f0: softcap
kernel void gsumt(constant P &p [[buffer(0)]], device const float *part [[buffer(1)]], device float *y [[buffer(2)]],
                  uint2 gid [[thread_position_in_grid]]) {
    const uint o = (((gid.y / TT) * p.a[3] + gid.x) * p.a[1]) * TT + gid.y % TT;
    float s = 0.0f;
    for (uint c = 0; c < p.a[1]; ++c) s += part[o + c * TT];
    y[gid.y * p.a[2] + gid.x] = p.f[0] > 0.0f ? tanh(s / p.f[0]) * p.f[0] : s;
}

// ---- norms ------------------------------------------------------------------------------------
// part[r][c] = the sum of squares of chunk c (256 wide) of x[r]: thread (c, r). a: W, nc
kernel void sspart(constant P &p [[buffer(0)]], device const float4 *x [[buffer(1)]], device float *part [[buffer(2)]],
                   uint2 gid [[thread_position_in_grid]]) {
    const uint W4 = p.a[0] / 4, c0 = gid.x * 64, c1 = min(W4, c0 + 64);
    float4 s = float4(0.0f);
    for (uint j = c0; j < c1; ++j) {
        const float4 v = x[gid.y * W4 + j];
        s += v * v;
    }
    part[gid.y * p.a[1] + gid.x] = hsum(s);
}

// y = x * rsqrt(mean(x^2) + eps) * w, as activations: thread (j / 4, r). a: W, nc; f0: eps
kernel void rms_act(constant P &p [[buffer(0)]], device const float4 *x [[buffer(1)]], device const float *part [[buffer(2)]],
                    device const float4 *w [[buffer(3)]], device ACT4 *y [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    const uint W4 = p.a[0] / 4, nc = p.a[1];
    float ss = 0.0f;
    for (uint c = 0; c < nc; ++c) ss += part[gid.y * nc + c];
    const float s = rsqrt(ss / float(p.a[0]) + p.f[0]);
    y[gid.y * W4 + gid.x] = ACT4(x[gid.y * W4 + gid.x] * s * w[gid.x]);
}

// ---- block attention residuals ----------------------------------------------------------------
#define PICK(i) ((i) == 0 ? s0 : (i) == 1 ? s1 : (i) == 2 ? s2 : (i) == 3 ? s3 : (i) == 4 ? s4 : s5)
// part[r * n + i][c] = {sum of squares, dot with each query} of chunk c of source i's row r:
// thread (c, r * n + i). a: H, n, nq, nc
kernel void ares_part(constant P &p [[buffer(0)]], device const float4 *qv [[buffer(1)]], device float *part [[buffer(2)]],
                      device const float4 *s0 [[buffer(3)]], device const float4 *s1 [[buffer(4)]],
                      device const float4 *s2 [[buffer(5)]], device const float4 *s3 [[buffer(6)]],
                      device const float4 *s4 [[buffer(7)]], device const float4 *s5 [[buffer(8)]],
                      uint2 gid [[thread_position_in_grid]]) {
    const uint H4 = p.a[0] / 4, n = p.a[1], nq = p.a[2], nc = p.a[3];
    const uint r = gid.y / n, i = gid.y % n, c0 = gid.x * 64, c1 = min(H4, c0 + 64);
    device const float4 *s = PICK(i);
    const uint q1 = min(1u, nq - 1) * H4, q2 = min(2u, nq - 1) * H4, q3 = min(3u, nq - 1) * H4;
    float4 ss = float4(0.0f), d0 = float4(0.0f), d1 = float4(0.0f), d2 = float4(0.0f), d3 = float4(0.0f);
    for (uint j = c0; j < c1; ++j) {
        const float4 v = s[r * H4 + j];
        ss += v * v;
        d0 += v * qv[j];
        d1 += v * qv[q1 + j];
        d2 += v * qv[q2 + j];
        d3 += v * qv[q3 + j];
    }
    const uint o = (gid.y * nc + gid.x) * 5;
    part[o] = hsum(ss), part[o + 1] = hsum(d0), part[o + 2] = hsum(d1), part[o + 3] = hsum(d2), part[o + 4] = hsum(d3);
}

// wts[r][q][i] = softmax_i(query q . norm(source i) / sqrt(H)) / rms(source i): thread (q, r)
kernel void ares_wts(constant P &p [[buffer(0)]], device const float *part [[buffer(1)]], device float *wts [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]]) {
    const uint n = p.a[1], nq = p.a[2], nc = p.a[3], r = gid.y, q = gid.x;
    float lg[6], rm[6];
    float mx = -INFINITY;
    for (uint i = 0; i < n; ++i) {
        float ss = 0.0f, d = 0.0f;
        for (uint c = 0; c < nc; ++c) {
            const uint o = ((r * n + i) * nc + c) * 5;
            ss += part[o], d += part[o + 1 + q];
        }
        rm[i] = rsqrt(ss / float(p.a[0]) + 1e-6f);
        lg[i] = d * rm[i] * p.f[0];
        mx = max(mx, lg[i]);
    }
    float sum = 0.0f;
    for (uint i = 0; i < n; ++i) lg[i] = exp(lg[i] - mx), sum += lg[i];
    for (uint i = 0; i < n; ++i) wts[(r * nq + q) * 8 + i] = lg[i] / sum * rm[i];
}

// out q = sum_i wts[r][q][i] * source i: thread (j / 4, r)
kernel void ares_mix(constant P &p [[buffer(0)]], device const float *wts [[buffer(1)]],
                     device const float4 *s0 [[buffer(2)]], device const float4 *s1 [[buffer(3)]],
                     device const float4 *s2 [[buffer(4)]], device const float4 *s3 [[buffer(5)]],
                     device const float4 *s4 [[buffer(6)]], device const float4 *s5 [[buffer(7)]],
                     device float4 *o0 [[buffer(8)]], device float4 *o1 [[buffer(9)]], device float4 *o2 [[buffer(10)]],
                     device float4 *o3 [[buffer(11)]], uint2 gid [[thread_position_in_grid]]) {
    const uint H4 = p.a[0] / 4, n = p.a[1], nq = p.a[2], r = gid.y, at = r * H4 + gid.x;
    const uint w0 = r * nq * 8, w1 = (r * nq + min(1u, nq - 1)) * 8, w2 = (r * nq + min(2u, nq - 1)) * 8, w3 = (r * nq + min(3u, nq - 1)) * 8;
    float4 a0 = float4(0.0f), a1 = float4(0.0f), a2 = float4(0.0f), a3 = float4(0.0f);
    for (uint i = 0; i < n; ++i) {
        const float4 v = PICK(i)[at];
        a0 += wts[w0 + i] * v, a1 += wts[w1 + i] * v, a2 += wts[w2 + i] * v, a3 += wts[w3 + i] * v;
    }
    o0[at] = a0;
    if (nq > 1) o1[at] = a1;
    if (nq > 2) o2[at] = a2;
    if (nq > 3) o3[at] = a3;
}

// ---- message gates ----------------------------------------------------------------------------
// cat[row] = [norm(h) | norm(msg)] as activations (the gate's low-rank map is then a matrix like
// any other): thread (j / 4, row). a: H, nc; f0: eps
kernel void gate_cat(constant P &p [[buffer(0)]], device const float4 *h [[buffer(1)]], device const float4 *msg [[buffer(2)]],
                     device const float *hp [[buffer(3)]], device const float *mp [[buffer(4)]], device ACT4 *cat [[buffer(5)]],
                     uint2 gid [[thread_position_in_grid]]) {
    const uint H = p.a[0], H4 = H / 4, nc = p.a[1], row = gid.y;
    float ssh = 0.0f, ssm = 0.0f;
    for (uint c = 0; c < nc; ++c) ssh += hp[row * nc + c], ssm += mp[row * nc + c];
    cat[row * 2 * H4 + gid.x] = ACT4(h[row * H4 + gid.x] * rsqrt(ssh / float(H) + p.f[0]));
    cat[row * 2 * H4 + H4 + gid.x] = ACT4(msg[row * H4 + gid.x] * rsqrt(ssm / float(H) + p.f[0]));
}

// out = (1 - tanh(Wf g)) h + tanh(Wg g) msg: thread (j, row). a: H, R
kernel void gate_mix(constant P &p [[buffer(0)]], device const ushort *wf [[buffer(1)]], device const ushort *wg [[buffer(2)]],
                     device const float *g [[buffer(3)]], device const float *h [[buffer(4)]], device const float *msg [[buffer(5)]],
                     device float *out [[buffer(6)]], uint2 gid [[thread_position_in_grid]]) {
    const uint H = p.a[0], R = p.a[1], row = gid.y, j = gid.x;
    float f = 0.0f, gn = 0.0f;
    if (R % 4 == 0) {
        float4 f4 = float4(0.0f), g4 = float4(0.0f);
        for (uint r = 0; r < R; r += 4) {
            const float4 gr = float4(g[row * R + r], g[row * R + r + 1], g[row * R + r + 2], g[row * R + r + 3]);
            f4 += float4(bf(wf[j * R + r]), bf(wf[j * R + r + 1]), bf(wf[j * R + r + 2]), bf(wf[j * R + r + 3])) * gr;
            g4 += float4(bf(wg[j * R + r]), bf(wg[j * R + r + 1]), bf(wg[j * R + r + 2]), bf(wg[j * R + r + 3])) * gr;
        }
        f = hsum(f4), gn = hsum(g4);
    } else {
        for (uint r = 0; r < R; ++r) {
            const float gr = g[row * R + r];
            f += bf(wf[j * R + r]) * gr, gn += bf(wg[j * R + r]) * gr;
        }
    }
    out[row * H + j] = (1.0f - tanh(f)) * h[row * H + j] + tanh(gn) * msg[row * H + j];
}

// in -> activations: thread (j / 4, row). a: W
kernel void toact(constant P &p [[buffer(0)]], device const float4 *x [[buffer(1)]], device ACT4 *y [[buffer(2)]],
                  uint2 gid [[thread_position_in_grid]]) {
    y[gid.y * (p.a[0] / 4) + gid.x] = ACT4(x[gid.y * (p.a[0] / 4) + gid.x]);
}
// out = (1 - tanh(f)) h + tanh(gn) msg: thread (j / 4, row). a: H
kernel void gate_fin(constant P &p [[buffer(0)]], device const float4 *f [[buffer(1)]], device const float4 *gn [[buffer(2)]],
                     device const float4 *h [[buffer(3)]], device const float4 *msg [[buffer(4)]], device float4 *out [[buffer(5)]],
                     uint2 gid [[thread_position_in_grid]]) {
    const uint at = gid.y * (p.a[0] / 4) + gid.x;
    out[at] = (1.0f - tanh(f[at])) * h[at] + tanh(gn[at]) * msg[at];
}

// ---- attention --------------------------------------------------------------------------------
// queries scaled, clipped per head, grown with the position (tau), as activations by kv head:
// qh[kh][row * G + g]. thread (hq, row). a: D, G, Hq, N, use_tau; f0: clip
kernel void qpost(constant P &p [[buffer(0)]], device const float4 *q [[buffer(1)]], device const float4 *sc [[buffer(2)]],
                  device const float *tau [[buffer(3)]], device ACT4 *qh [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    const uint D4 = p.a[0] / 4, G = p.a[1], Hq = p.a[2], N = p.a[3], hq = gid.x, row = gid.y, base = (row * Hq + hq) * D4;
    float4 ss = float4(0.0f);
    for (uint i = 0; i < D4; ++i) {
        const float4 v = q[base + i] * sc[hq * D4 + i];
        ss += v * v;
    }
    float s = min(1.0f, p.f[0] / (sqrt(hsum(ss)) + 1e-6f));
    if (p.a[4] != 0) s *= tau[row];
    const uint ob = ((hq / G) * N * G + row * G + hq % G) * D4;
    for (uint i = 0; i < D4; ++i) qh[ob + i] = ACT4(q[base + i] * sc[hq * D4 + i] * s);
}

// the relative states scaled, clipped, grown, in place: thread (hq, row). a: dr, Hq, use_tau; f0: clip
kernel void relpost(constant P &p [[buffer(0)]], device float4 *rel [[buffer(1)]], device const float4 *sc [[buffer(2)]],
                    device const float *tau [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    const uint R4 = p.a[0] / 4, Hq = p.a[1], hq = gid.x, row = gid.y, base = (row * Hq + hq) * R4;
    float4 ss = float4(0.0f);
    for (uint i = 0; i < R4; ++i) {
        const float4 v = rel[base + i] * sc[hq * R4 + i];
        ss += v * v;
    }
    float s = min(1.0f, p.f[0] / (sqrt(hsum(ss)) + 1e-6f));
    if (p.a[2] != 0) s *= tau[row];
    for (uint i = 0; i < R4; ++i) rel[base + i] = rel[base + i] * sc[hq * R4 + i] * s;
}

// B[row * Hq + hq][dist] = rel . proj[:, dist]: thread (dist / 4, row * Hq + hq). a: dr, ext, ldB
kernel void relbias(constant P &p [[buffer(0)]], device const float *rel [[buffer(1)]], device const float *proj [[buffer(2)]],
                    device float4 *B [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    const uint dr = p.a[0], ext = p.a[1], last = dr * ext - 1;
    float4 s = float4(0.0f);
    for (uint a = 0; a < dr; ++a) {
        const uint o = a * ext + gid.x * 4;
        s += rel[gid.y * dr + a] * float4(proj[min(o, last)], proj[min(o + 1, last)], proj[min(o + 2, last)], proj[min(o + 3, last)]);
    }
    B[gid.y * (p.a[2] / 4) + gid.x] = s;
}

// The short convolution over each sequence's rows: y = (sum_j w_j * ctx + x) * norm (+ add), ctx
// the sequence's K - 1 rows before the step (hist) then the step's. Static: tap j on ctx row
// t + j; dynamic: lag j on ctx row t + K - 1 - j with weight taps[j] + dyn[row][head][j].
// thread (c / 4, row). a: C, K, dynamic, heads, slot stride, state offset, add; f0: norm
kernel void sconv(constant P &p [[buffer(0)]], device const float4 *x [[buffer(1)]], device float4 *y [[buffer(2)]],
                  device const float4 *taps [[buffer(3)]], device const float *dyn [[buffer(4)]],
                  device const float4 *add [[buffer(5)]], device const float4 *hist [[buffer(6)]],
                  device const uint4 *rowinfo [[buffer(7)]], uint2 gid [[thread_position_in_grid]]) {
    const uint C4 = p.a[0] / 4, K = p.a[1], K1 = K - 1, row = gid.y, c4 = gid.x;
    const uint4 ri = rowinfo[row];
    const uint first = ri.z, t = row - first, hb = (ri.x * p.a[4] + p.a[5]) / 4;
    const bool dy = p.a[2] != 0;
    const uint hd = dy ? (c4 * 4) / (p.a[0] / p.a[3]) : 0;
    float4 o = float4(0.0f);
    for (uint j = 0; j < K; ++j) {
        const uint ci = dy ? t + K1 - j : t + j;
        const float4 v = ci >= K1 ? x[(first + ci - K1) * C4 + c4] : hist[hb + ci * C4 + c4];
        float4 w = taps[j * C4 + c4];
        if (dy) w += dyn[row * p.a[3] * K + hd * K + j];
        o += w * v;
    }
    o = (o + x[row * C4 + c4]) * p.f[0];
    if (p.a[6] != 0) o += add[row * C4 + c4];
    y[row * C4 + c4] = o;
}

// a sequence's history after the step: the last K - 1 rows of [hist, x]. thread (c / 4, segment).
// a: C, K, slot stride, state offset
kernel void hist_update(constant P &p [[buffer(0)]], device const float4 *x [[buffer(1)]], device float4 *hist [[buffer(2)]],
                        device const uint4 *segs [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    const uint C4 = p.a[0] / 4, K1 = p.a[1] - 1;
    const uint4 sg = segs[gid.y];
    const uint hb = (sg.x * p.a[2] + p.a[3]) / 4, first = sg.y, n = sg.z;
    for (uint j = 0; j < K1; ++j)
        hist[hb + j * C4 + gid.x] = n + j >= K1 ? x[(first + n + j - K1) * C4 + gid.x] : hist[hb + (j + n) * C4 + gid.x];
}

// The rows a rejected draft puts back: the last min(n, U) + K - 1 rows of [hist, x] of each
// sequence with two rows or more (before hist_update). thread (c / 4, segment).
// a: C, K, slot stride, state offset, undo stride, undo offset, U
kernel void hist_save(constant P &p [[buffer(0)]], device const float4 *x [[buffer(1)]], device const float4 *hist [[buffer(2)]],
                      device const uint4 *segs [[buffer(3)]], device float4 *undo [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
    const uint C4 = p.a[0] / 4, K1 = p.a[1] - 1;
    const uint4 sg = segs[gid.y];
    const uint first = sg.y, n = sg.z;
    if (n < 2) return;
    const uint hb = (sg.x * p.a[2] + p.a[3]) / 4, ub = (sg.x * p.a[4] + p.a[5]) / 4, m = min(n, p.a[6]) + K1, r0 = n + K1 - m;
    for (uint j = 0; j < m; ++j) {
        const uint row = r0 + j;
        undo[ub + j * C4 + gid.x] = row >= K1 ? x[(first + row - K1) * C4 + gid.x] : hist[hb + row * C4 + gid.x];
    }
}

// The history after each row that ends a cache page, by page (before hist_update): the K - 1 rows
// of [hist, x] up to it. ends: [slot, first row of its segment, row, page]. thread (c / 4, end).
// a: C, K, slot stride, state offset
kernel void tail_save(constant P &p [[buffer(0)]], device const float4 *x [[buffer(1)]], device const float4 *hist [[buffer(2)]],
                      device const uint4 *ends [[buffer(3)]], device float4 *tails [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
    const uint C4 = p.a[0] / 4, K1 = p.a[1] - 1;
    const uint4 e = ends[gid.y];
    const uint hb = (e.x * p.a[2] + p.a[3]) / 4, tb = (e.w * p.a[2] + p.a[3]) / 4, t = e.z - e.y;
    for (uint j = 0; j < K1; ++j) {
        const uint row = t + 1 + j;  // in [hist, x]
        tails[tb + j * C4 + gid.x] = row >= K1 ? x[(e.y + row - K1) * C4 + gid.x] : hist[hb + row * C4 + gid.x];
    }
}

// keys clipped (mode 0: f0 the threshold) or values normalised (mode 1: f0 eps) per kv head, into
// the pool: thread (kh, row). a: D, Hk, page, maxp, mode
kernel void kvstore(constant P &p [[buffer(0)]], device const float4 *src [[buffer(1)]], device KV4 *pool [[buffer(2)]],
                    device const uint4 *rowinfo [[buffer(3)]], device const int *pt [[buffer(4)]],
                    uint2 gid [[thread_position_in_grid]]) {
    const uint D4 = p.a[0] / 4, Hk = p.a[1], kh = gid.x, row = gid.y, base = (row * Hk + kh) * D4;
    float4 ss = float4(0.0f);
    for (uint i = 0; i < D4; ++i) ss += src[base + i] * src[base + i];
    const float s = p.a[4] == 0 ? min(1.0f, p.f[0] / (sqrt(hsum(ss)) + 1e-6f)) : rsqrt(hsum(ss) / float(p.a[0]) + p.f[0]);
    const uint4 ri = rowinfo[row];
    const uint ob = kvrow(pt, ri.x, ri.y, kh, Hk, p.a[2], p.a[3]) * D4;
    for (uint i = 0; i < D4; ++i) pool[ob + i] = KV4(src[base + i] * s);
}

// The scoring of one sequence's n rows (positions pos0 ..) against its keys k0 .. k0 + nk.
// a: G, nk, ldT, ext, window, pos0, k0, Hq, ldB, first, nch, nG, D, slot, page, maxp; f0: scale;
// f1: Hk, f2: N * G (the queries' rows a kv head) as floats
struct Rng { uint lo, hi; };
inline Rng rng(uint pos, uint k0, uint window) {
    Rng r;
    r.hi = pos - k0;
    r.lo = (window > 0 && pos + 1 > window + k0) ? pos + 1 - window - k0 : 0;
    return r;
}

// S[kh][i][t] = q_i . k_t: thread (t, tile of TT queries, kh)
kernel void attn_qk(constant P &p [[buffer(0)]], device const KV4 *pool [[buffer(1)]], device const ACT4 *qh [[buffer(2)]],
                    device float *S [[buffer(3)]], device const int *pt [[buffer(4)]], uint3 gid [[thread_position_in_grid]]) {
    const uint G = p.a[0], ldT = p.a[2], k0 = p.a[6], first = p.a[9], nG = p.a[11], D4 = p.a[12] / 4;
    const uint Hk = uint(p.f[1]), NG = uint(p.f[2]), kh = gid.z, s = gid.y * TT;
    const uint ka = kvrow(pt, p.a[13], k0 + gid.x, kh, Hk, p.a[14], p.a[15]) * D4;
    const uint qb = (kh * NG + first * G + s) * D4;
    float acc[TT];
    for (uint t = 0; t < TT; ++t) acc[t] = 0.0f;
    for (uint i = 0; i < D4; ++i) {
        const ACT4 kv = ACT4(pool[ka + i]);
        for (uint t = 0; t < TT; ++t) {
            const float4 pr = float4(kv * qh[qb + i + t * D4]);
            acc[t] += (pr.x + pr.y) + (pr.z + pr.w);
        }
    }
    const uint m = min(uint(TT), nG - s);
    for (uint t = 0; t < m; ++t) S[(kh * nG + s + t) * ldT + gid.x] = acc[t];
}

// scores scaled, the bias by distance added, each chunk's maximum: thread (chunk of 64 keys, i, kh)
kernel void sm_a(constant P &p [[buffer(0)]], device float *S [[buffer(1)]], device const float *B [[buffer(2)]],
                 device float *cm [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) {
    const uint G = p.a[0], nk = p.a[1], ldT = p.a[2], ext = p.a[3], Hq = p.a[7], nch = p.a[10], nG = p.a[11];
    const uint i = gid.y, kh = gid.z, r = i / G, hq = kh * G + i % G;
    const Rng g = rng(p.a[5] + r, p.a[6], p.a[4]);
    const uint base = (kh * nG + i) * ldT, bo = ((p.a[9] + r) * Hq + hq) * p.a[8];
    const uint t0 = max(g.lo, gid.x * 64), t1 = min(g.hi + 1, min(nk, gid.x * 64 + 64));
    float mx = -INFINITY;
    for (uint t = t0; t < t1; ++t) {
        const uint dist = g.hi - t;
        float v = S[base + t] * p.f[0];
        if (dist < ext) v += B[bo + dist];
        S[base + t] = v;
        mx = max(mx, v);
    }
    cm[(kh * nG + i) * nch + gid.x] = mx;
}

// exp(score - max) (0 outside the row's keys), each chunk's sum: thread (chunk, i, kh)
kernel void sm_b(constant P &p [[buffer(0)]], device float *S [[buffer(1)]], device const float *sink [[buffer(2)]],
                 device const float *cm [[buffer(3)]], device float *cs [[buffer(4)]], uint3 gid [[thread_position_in_grid]]) {
    const uint G = p.a[0], nk = p.a[1], ldT = p.a[2], Hq = p.a[7], nch = p.a[10], nG = p.a[11];
    const uint i = gid.y, kh = gid.z, r = i / G, hq = kh * G + i % G;
    const Rng g = rng(p.a[5] + r, p.a[6], p.a[4]);
    const uint base = (kh * nG + i) * ldT, ci = (kh * nG + i) * nch;
    float mx = sink[(p.a[9] + r) * Hq + hq];
    for (uint c = 0; c < nch; ++c) mx = max(mx, cm[ci + c]);
    const uint c0 = gid.x * 64, c1 = min(nk, c0 + 64);
    float sum = 0.0f;
    for (uint t = c0; t < c1; ++t) {
        float e = 0.0f;
        if (t >= g.lo && t <= g.hi) e = exp(S[base + t] - mx), sum += e;
        S[base + t] = e;
    }
    cs[ci + gid.x] = sum;
}

// den[kh][i] = 1 / (the sink's share + the sum): the sink takes probability and adds nothing.
// thread (i, kh)
kernel void sm_den(constant P &p [[buffer(0)]], device const float *sink [[buffer(1)]], device const float *cm [[buffer(2)]],
                   device const float *cs [[buffer(3)]], device float *den [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    const uint G = p.a[0], Hq = p.a[7], nch = p.a[10], nG = p.a[11];
    const uint i = gid.x, kh = gid.y, r = i / G, hq = kh * G + i % G, ci = (kh * nG + i) * nch;
    const float sk = sink[(p.a[9] + r) * Hq + hq];
    float mx = sk, sum = 0.0f;
    for (uint c = 0; c < nch; ++c) mx = max(mx, cm[ci + c]), sum += cs[ci + c];
    den[kh * nG + i] = 1.0f / (exp(sk - mx) + sum);
}

// the sequence's values k0 .. as contiguous rows per kv head (for MPS): thread (d / 4, t, kh).
// a[1]: nk, a[2]: floats a row of vg
kernel void kv_gather(constant P &p [[buffer(0)]], device const KV4 *pool [[buffer(1)]], device const int *pt [[buffer(2)]],
                      device float4 *vg [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) {
    const uint D4 = p.a[12] / 4, Hk = uint(p.f[1]);
    vg[(gid.z * p.a[1] + gid.y) * (p.a[2] / 4) + gid.x] =
        float4(pool[kvrow(pt, p.a[13], p.a[6] + gid.y, gid.z, Hk, p.a[14], p.a[15]) * D4 + gid.x]);
}

// xo[row][hq] = den * O[kh][i] * sigmoid(og), as activations (O = E V with E the unnormalised
// exp scores): thread (d / 4, i, kh). a[2]: floats a row of O
kernel void ogate(constant P &p [[buffer(0)]], device const float4 *O [[buffer(1)]], device const float4 *og [[buffer(2)]],
                  device ACT4 *xo [[buffer(3)]], device const float *den [[buffer(4)]], uint3 gid [[thread_position_in_grid]]) {
    const uint G = p.a[0], Hq = p.a[7], nG = p.a[11], D4 = p.a[12] / 4;
    const uint i = gid.y, kh = gid.z, at = ((p.a[9] + i / G) * Hq + kh * G + i % G) * D4 + gid.x;
    xo[at] = ACT4(O[(kh * nG + i) * (p.a[2] / 4) + gid.x] * den[kh * nG + i] * sigm4(og[at]));
}

// part[kh][i][chunk] = den * sum over the chunk's keys of e * v: thread (chunk, i * D/4 + d/4, kh)
kernel void pv1(constant P &p [[buffer(0)]], device const float *S [[buffer(1)]], device const KV4 *pool [[buffer(2)]],
                device const int *pt [[buffer(3)]], device const float *den [[buffer(4)]], device float4 *part [[buffer(5)]],
                uint3 gid [[thread_position_in_grid]]) {
    const uint nk = p.a[1], ldT = p.a[2], k0 = p.a[6], nch = p.a[10], nG = p.a[11], D4 = p.a[12] / 4;
    const uint Hk = uint(p.f[1]), i = gid.y / D4, d4 = gid.y % D4, kh = gid.z, base = (kh * nG + i) * ldT;
    const uint page = p.a[14], t1 = min(nk, gid.x * 64 + 64);
    float4 acc = float4(0.0f);
    for (uint t = gid.x * 64; t < t1;) {
        const uint pos = k0 + t, in_page = pos % page, cnt = min(t1 - t, page - in_page);
        const uint vb = ((uint(pt[p.a[13] * p.a[15] + pos / page]) * Hk + kh) * page + in_page) * D4 + d4;
        for (uint j = 0; j < cnt; ++j) acc += S[base + t + j] * float4(pool[vb + j * D4]);
        t += cnt;
    }
    part[((kh * nG + i) * nch + gid.x) * D4 + d4] = acc * den[kh * nG + i];
}

// xo[row][hq] = (sum of the chunks) * sigmoid(og), as activations: thread (d / 4, i, kh)
kernel void pv2(constant P &p [[buffer(0)]], device const float4 *part [[buffer(1)]], device const float4 *og [[buffer(2)]],
                device ACT4 *xo [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) {
    const uint G = p.a[0], Hq = p.a[7], nch = p.a[10], nG = p.a[11], D4 = p.a[12] / 4;
    const uint i = gid.y, kh = gid.z, at = ((p.a[9] + i / G) * Hq + kh * G + i % G) * D4 + gid.x;
    float4 acc = float4(0.0f);
    for (uint c = 0; c < nch; ++c) acc += part[((kh * nG + i) * nch + c) * D4 + gid.x];
    xo[at] = ACT4(acc * sigm4(og[at]));
}


// ---- the draft head (DSpark: a block of mask rows attending to features of the target) ----------
// taps[row][t] = src[row] (the target's stream entering a visit): thread (j / 4, row). a: H, t
kernel void tapcopy(constant P &p [[buffer(0)]], device const float4 *src [[buffer(1)]], device float4 *taps [[buffer(2)]],
                    uint2 gid [[thread_position_in_grid]]) {
    const uint H4 = p.a[0] / 4;
    taps[(gid.y * 4 + p.a[1]) * H4 + gid.x] = src[gid.y * H4 + gid.x];
}
kernel void tapcopy_act(constant P &p [[buffer(0)]], device const ACT4 *src [[buffer(1)]], device float4 *taps [[buffer(2)]],
                        uint2 gid [[thread_position_in_grid]]) {
    const uint H4 = p.a[0] / 4;
    taps[(gid.y * 4 + p.a[1]) * H4 + gid.x] = float4(src[gid.y * H4 + gid.x]);
}
// The taps of rows[i] mixed by alpha and RMS-normalised with gain w, as activations: mixpart the
// sums of squares by chunk (thread (c, i)), mixapply the rows (thread (j / 4, i)). a: H, nc; f0: eps
inline float4 tapmix(device const float4 *taps, device const float *al, uint row, uint H4, uint j) {
    return taps[(row * 4) * H4 + j] * al[0] + taps[(row * 4 + 1) * H4 + j] * al[1] + taps[(row * 4 + 2) * H4 + j] * al[2] +
           taps[(row * 4 + 3) * H4 + j] * al[3];
}
kernel void mixpart(constant P &p [[buffer(0)]], device const float4 *taps [[buffer(1)]], device const float *al [[buffer(2)]],
                    device const uint *rows [[buffer(3)]], device float *part [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    const uint H4 = p.a[0] / 4, c0 = gid.x * 64, c1 = min(H4, c0 + 64), row = rows[gid.y];
    float4 s = float4(0.0f);
    for (uint j = c0; j < c1; ++j) {
        const float4 v = tapmix(taps, al, row, H4, j);
        s += v * v;
    }
    part[gid.y * p.a[1] + gid.x] = hsum(s);
}
kernel void mixapply(constant P &p [[buffer(0)]], device const float4 *taps [[buffer(1)]], device const float *al [[buffer(2)]],
                     device const uint *rows [[buffer(3)]], device const float *part [[buffer(4)]],
                     device const float4 *w [[buffer(5)]], device ACT4 *out [[buffer(6)]], uint2 gid [[thread_position_in_grid]]) {
    const uint H4 = p.a[0] / 4, nc = p.a[1];
    float ss = 0.0f;
    for (uint c = 0; c < nc; ++c) ss += part[gid.y * nc + c];
    out[gid.y * H4 + gid.x] = ACT4(tapmix(taps, al, rows[gid.y], H4, gid.x) * rsqrt(ss / float(p.a[0]) + p.f[0]) * w[gid.x]);
}

// A head's vector normalised (RMS, gain w [D]) and rotated by its position (RoPE, halves): the
// angle of pair i at position pos is pos * theta^(-i / (D/2)).
// normrope_k: f32 [row][heads][D] -> dst, the same layout (a[4] = 0), or -> the pool's row of
// (slot, position) (a[4] = 1; rowinfo = [slot, position, ..]). thread (head, row).
// a: D, heads, page, maxp, to pool; f0: eps, f1: log(theta)
#define NORMROPE                                                                                  \
    const uint D = p.a[0], hw = D / 2, hd = gid.x, row = gid.y, base = (row * p.a[1] + hd) * D; \
    float ss = 0.0f;                                                                              \
    for (uint i = 0; i < D; ++i) ss += src[base + i] * src[base + i];                             \
    const float inv = rsqrt(ss / float(D) + p.f[0]);                                              \
    const float pos = float(rowinfo[row].y);
kernel void normrope_k(constant P &p [[buffer(0)]], device const float *src [[buffer(1)]], device const float *w [[buffer(2)]],
                       device const uint4 *rowinfo [[buffer(3)]], device const int *pt [[buffer(4)]], device float *dst [[buffer(5)]],
                       device KV *pool [[buffer(6)]], uint2 gid [[thread_position_in_grid]]) {
    NORMROPE
    const bool tp = p.a[4] != 0;
    const uint ob = tp ? kvrow(pt, rowinfo[row].x, rowinfo[row].y, hd, p.a[1], p.a[2], p.a[3]) * D : base;
    for (uint i = 0; i < hw; ++i) {
        const float ang = pos * exp(-p.f[1] * float(i) / float(hw)), c = cos(ang), sn = sin(ang);
        const float x1 = src[base + i] * inv * w[i], x2 = src[base + hw + i] * inv * w[hw + i];
        const float o1 = x1 * c - x2 * sn, o2 = x2 * c + x1 * sn;
        if (tp) pool[ob + i] = KV(o1), pool[ob + hw + i] = KV(o2);
        else dst[ob + i] = o1, dst[ob + hw + i] = o2;
    }
}
// normrope_q: the queries as activations by kv head, qh[kh][row * G + g]. a: D, heads, G, N
kernel void normrope_q(constant P &p [[buffer(0)]], device const float *src [[buffer(1)]], device const float *w [[buffer(2)]],
                       device const uint4 *rowinfo [[buffer(3)]], device ACT *qh [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    NORMROPE
    const uint G = p.a[2], ob = ((hd / G) * p.a[3] * G + row * G + hd % G) * D;
    for (uint i = 0; i < hw; ++i) {
        const float ang = pos * exp(-p.f[1] * float(i) / float(hw)), c = cos(ang), sn = sin(ang);
        const float x1 = src[base + i] * inv * w[i], x2 = src[base + hw + i] * inv * w[hw + i];
        qh[ob + i] = ACT(x1 * c - x2 * sn), qh[ob + hw + i] = ACT(x2 * c + x1 * sn);
    }
}
// values as they are into the pool: thread (d / 4, kh, row). a: D, Hk, page, maxp
kernel void kvcopy(constant P &p [[buffer(0)]], device const float4 *src [[buffer(1)]], device KV4 *pool [[buffer(2)]],
                   device const uint4 *rowinfo [[buffer(3)]], device const int *pt [[buffer(4)]], uint3 gid [[thread_position_in_grid]]) {
    const uint D4 = p.a[0] / 4, Hk = p.a[1];
    const uint4 ri = rowinfo[gid.z];
    pool[kvrow(pt, ri.x, ri.y, gid.y, Hk, p.a[2], p.a[3]) * D4 + gid.x] = KV4(src[(gid.z * Hk + gid.y) * D4 + gid.x]);
}
// The block's own keys after the context's in the scores: S[kh][i][nk + j] = q_i . k_j.
// thread (j, i, kh). a as the scoring's (a[1] = the context's keys + the block's, a[9] = the
// block's first row); f3: the block's rows
kernel void dblock_qk(constant P &p [[buffer(0)]], device const ACT *qh [[buffer(1)]], device const float *kb [[buffer(2)]],
                      device float *S [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) {
    const uint G = p.a[0], ldT = p.a[2], first = p.a[9], nG = p.a[11], D = p.a[12], B = uint(p.f[3]);
    const uint Hk = uint(p.f[1]), NG = uint(p.f[2]), j = gid.x, i = gid.y, kh = gid.z;
    const uint qb = (kh * NG + first * G + i) * D, kbase = ((first + j) * Hk + kh) * D;
    float acc = 0.0f;
    for (uint d = 0; d < D; ++d) acc += float(qh[qb + d]) * kb[kbase + d];
    S[(kh * nG + i) * ldT + (p.a[1] - B) + j] = acc;
}
// xo[row][hq] = the context's part (pv1's chunks) + den * sum_j e_j v_j over the block's rows, as
// activations: thread (d / 4, i, kh). a[10]: the context's chunks
kernel void dpv2(constant P &p [[buffer(0)]], device const float4 *part [[buffer(1)]], device const float *S [[buffer(2)]],
                 device const float4 *vb [[buffer(3)]], device const float *den [[buffer(4)]], device ACT4 *xo [[buffer(5)]],
                 uint3 gid [[thread_position_in_grid]]) {
    const uint G = p.a[0], ldT = p.a[2], Hq = p.a[7], first = p.a[9], nch = p.a[10], nG = p.a[11], D4 = p.a[12] / 4;
    const uint Hk = uint(p.f[1]), B = uint(p.f[3]), i = gid.y, kh = gid.z, nk = p.a[1] - B;
    float4 acc = float4(0.0f);
    for (uint c = 0; c < nch; ++c) acc += part[((kh * nG + i) * nch + c) * D4 + gid.x];
    float4 blk = float4(0.0f);
    for (uint j = 0; j < B; ++j) blk += S[(kh * nG + i) * ldT + nk + j] * vb[((first + j) * Hk + kh) * D4 + gid.x];
    xo[((first + i / G) * Hq + kh * G + i % G) * D4 + gid.x] = ACT4(acc + blk * den[kh * nG + i]);
}
// out = a + b: thread (j / 4, row). a: W
kernel void add2(constant P &p [[buffer(0)]], device const float4 *a [[buffer(1)]], device const float4 *b [[buffer(2)]],
                 device float4 *out [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    const uint at = gid.y * (p.a[0] / 4) + gid.x;
    out[at] = a[at] + b[at];
}
// act = silu(gate) * up: thread (j / 4, row). a: W
kernel void silu_mul(constant P &p [[buffer(0)]], device const float4 *g [[buffer(1)]], device const float4 *u [[buffer(2)]],
                     device ACT4 *act [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    const uint at = gid.y * (p.a[0] / 4) + gid.x;
    act[at] = ACT4(g[at] * sigm4(g[at]) * u[at]);
}
// cat[r] = [h of the block row sel[r] | f[r / k]] (the recurrent head's inputs of a draft):
// thread (j / 4, r). a: H, k
kernel void dcat(constant P &p [[buffer(0)]], device const ACT4 *h [[buffer(1)]], device const uint *sel [[buffer(2)]],
                 device const ACT4 *f [[buffer(3)]], device ACT4 *cat [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    const uint H4 = p.a[0] / 4;
    cat[gid.y * 2 * H4 + gid.x] = h[sel[gid.y] * H4 + gid.x];
    cat[gid.y * 2 * H4 + H4 + gid.x] = f[(gid.y / p.a[1]) * H4 + gid.x];
}
// The recurrent head, draft i of every block: u = Ws [state | w1[prev]] + cond: thread (r, s).
// a: rank, k, i, 3 * rank
kernel void mk_step(constant P &p [[buffer(0)]], device const ushort *ws [[buffer(1)]], device const float *state [[buffer(2)]],
                    device const ushort *w1 [[buffer(3)]], device const uint *prev [[buffer(4)]], device const float *cond [[buffer(5)]],
                    device float *u [[buffer(6)]], device const float *rb [[buffer(7)]], uint2 gid [[thread_position_in_grid]]) {
    const uint R = p.a[0], r = gid.x, sq = gid.y, tok = prev[sq];
    float4 acc = float4(0.0f);
    for (uint c = 0; c < R; c += 4) {
        acc += float4(bf(ws[r * 2 * R + c]), bf(ws[r * 2 * R + c + 1]), bf(ws[r * 2 * R + c + 2]), bf(ws[r * 2 * R + c + 3])) *
               float4(state[sq * R + c], state[sq * R + c + 1], state[sq * R + c + 2], state[sq * R + c + 3]);
        acc += float4(bf(ws[r * 2 * R + R + c]), bf(ws[r * 2 * R + R + c + 1]), bf(ws[r * 2 * R + R + c + 2]), bf(ws[r * 2 * R + R + c + 3])) *
               float4(bf(w1[tok * R + c]), bf(w1[tok * R + c + 1]), bf(w1[tok * R + c + 2]), bf(w1[tok * R + c + 3]));
    }
    u[sq * p.a[3] + r] = hsum(acc) + cond[(sq * p.a[1] + p.a[2]) * p.a[3] + r] + rb[r];
}
// state = g state + (1 - g) tanh(cand), latent = tanh(out): thread (c, s). a: rank, -, -, 3 * rank
kernel void mk_gate(constant P &p [[buffer(0)]], device const float *u [[buffer(1)]], device float *state [[buffer(2)]],
                    device ACT *latent [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    const uint R = p.a[0], c = gid.x, sq = gid.y;
    const float g = sigm(u[sq * p.a[3] + c]);
    state[sq * R + c] = g * state[sq * R + c] + (1.0f - g) * tanh(u[sq * p.a[3] + R + c]);
    latent[sq * R + c] = ACT(tanh(u[sq * p.a[3] + 2 * R + c]));
}
// The draft: argmax over the vocabulary of base[s * k + i] + low[s] + bias. mk_max1 a chunk of 256
// (thread (chunk, s)), mk_max2 the chunks (thread (s)): the token into prev and out[s * k + i],
// and its probability under the softcapped logits into conf (how far to trust the chain).
// a: V, k, i, chunks; f0: the logits' softcap
kernel void mk_max1(constant P &p [[buffer(0)]], device const float *base [[buffer(1)]], device const float *low [[buffer(2)]],
                    device const float *bias [[buffer(3)]], device float *cv [[buffer(4)]], device uint *ci [[buffer(5)]],
                    device float *cs [[buffer(6)]], uint2 gid [[thread_position_in_grid]]) {
    const uint V = p.a[0], sq = gid.y, v0 = gid.x * 256, v1 = min(V, v0 + 256), bo = (sq * p.a[1] + p.a[2]) * V;
    float best = -INFINITY, sum = 0.0f;
    uint at = v0;
    for (uint v = v0; v < v1; ++v) {
        const float x = base[bo + v] + low[sq * V + v] + bias[v];
        if (x > best) best = x, at = v;
        sum += exp(tanh(x / p.f[0]) * p.f[0] - p.f[0]);  // (capped logits are at most the cap)
    }
    cv[sq * p.a[3] + gid.x] = best, ci[sq * p.a[3] + gid.x] = at, cs[sq * p.a[3] + gid.x] = sum;
}
kernel void mk_max2(constant P &p [[buffer(0)]], device const float *cv [[buffer(1)]], device const uint *ci [[buffer(2)]],
                    device uint *prev [[buffer(3)]], device uint *out [[buffer(4)]], device const float *cs [[buffer(5)]],
                    device float *conf [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    float best = -INFINITY, sum = 0.0f;
    uint at = 0;
    for (uint c = 0; c < p.a[3]; ++c) {
        if (cv[gid * p.a[3] + c] > best) best = cv[gid * p.a[3] + c], at = ci[gid * p.a[3] + c];
        sum += cs[gid * p.a[3] + c];
    }
    prev[gid] = at, out[gid * p.a[1] + p.a[2]] = at;
    conf[gid * p.a[1] + p.a[2]] = exp(tanh(best / p.f[0]) * p.f[0] - p.f[0]) / sum;
}

// ---- routing ----------------------------------------------------------------------------------
// expert e's rank by sigmoid(logit) + bias (ties: the lower index); the first k into idx.
// thread (e, row). a: E, k
kernel void route_rank(constant P &p [[buffer(0)]], device const float *lg [[buffer(1)]], device const float *bias [[buffer(2)]],
                       device uint *idx [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    const uint E = p.a[0], e = gid.x, row = gid.y;
    const float me = sigm(lg[row * E + e]) + bias[e];
    uint rank = 0;
    for (uint o = 0; o < E; ++o) {
        const float so = sigm(lg[row * E + o]) + bias[o];
        rank += (so > me || (so == me && o < e)) ? 1u : 0u;
    }
    if (rank < p.a[1]) idx[row * p.a[1] + rank] = e;
}

// wt[row][j] = score_j / sum of the chosen scores * route_scale: thread (j, row). a: E, k; f0: scale
kernel void route_wt(constant P &p [[buffer(0)]], device const float *lg [[buffer(1)]], device const uint *idx [[buffer(2)]],
                     device float *wt [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    const uint E = p.a[0], k = p.a[1], row = gid.y;
    float sum = 1e-20f;
    for (uint j = 0; j < k; ++j) sum += sigm(lg[row * E + idx[row * k + j]]);
    wt[row * k + gid.x] = sigm(lg[row * E + idx[row * k + gid.x]]) / sum * p.f[0];
}

// The (row, choice) pairs by expert, in four dispatches of short threads. An expert's tokens go in
// tiles of 16, its last ones in the smallest tile that holds them (12, 8, 4 or 2): a tile's kernel
// computes all its tokens, so a tile 9 of 16 full wastes as much as it uses. Items are [expert,
// first x row, first y row, count], by tile class in cnt[5] = {16, 12, 8, 4, 2}.
// xcount: cc[e][c] = pairs of chunk c (64 pairs) routed to e. thread (c, e). a: pairs, E, k, nch
kernel void xcount(constant P &p [[buffer(0)]], device const uint *idx [[buffer(1)]], device uint *cc [[buffer(2)]],
                   uint2 gid [[thread_position_in_grid]]) {
    uint n = 0;
    for (uint i = gid.x * 64; i < min(p.a[0], gid.x * 64 + 64); ++i) n += idx[i] == gid.y ? 1u : 0u;
    cc[gid.y * p.a[3] + gid.x] = n;
}
// xprefix: cc[e][c] <- the pairs of e before chunk c; tot[e]. thread (e)
kernel void xprefix(constant P &p [[buffer(0)]], device uint *cc [[buffer(1)]], device uint *tot [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    uint run = 0;
    for (uint c = 0; c < p.a[3]; ++c) {
        const uint v = cc[gid * p.a[3] + c];
        cc[gid * p.a[3] + c] = run;
        run += v;
    }
    tot[gid] = run;
}
inline uint4 tiles_of(uint c) {  // {tiles of 16, class of the last tile (1..4 = 12, 8, 4, 2; 0 none), -, -}
    const uint rem = c % 16;
    uint n16 = c / 16, cls = 0;
    if (rem > 12) n16 += 1;
    else if (rem > 8) cls = 1;
    else if (rem > 4) cls = 2;
    else if (rem > 2) cls = 3;
    else if (rem > 0) cls = 4;
    return uint4(n16, cls, 0, 0);
}
// xstart: start[e], and e's items. thread (e)
kernel void xstart(constant P &p [[buffer(0)]], device const uint *tot [[buffer(1)]], device uint *start [[buffer(2)]],
                   device uint *it16 [[buffer(3)]], device uint *it12 [[buffer(4)]], device uint *it8 [[buffer(5)]],
                   device uint *it4 [[buffer(6)]], device uint *it2 [[buffer(7)]], device uint *cnt [[buffer(8)]],
                   uint gid [[thread_position_in_grid]]) {
    uint st = 0, b16 = 0, b12 = 0, b8 = 0, b4 = 0, b2 = 0;
    for (uint e = 0; e < gid; ++e) {
        const uint4 t = tiles_of(tot[e]);
        st += tot[e], b16 += t.x;
        b12 += t.y == 1 ? 1u : 0u, b8 += t.y == 2 ? 1u : 0u, b4 += t.y == 3 ? 1u : 0u, b2 += t.y == 4 ? 1u : 0u;
    }
    start[gid] = st;
    const uint c = tot[gid];
    const uint4 t = tiles_of(c);
    uint s = st;
    for (uint f = 0; f < t.x; ++f) {
        const uint m = min(16u, st + c - s);
        it16[4 * (b16 + f)] = gid, it16[4 * (b16 + f) + 1] = s, it16[4 * (b16 + f) + 2] = s, it16[4 * (b16 + f) + 3] = m;
        s += m;
    }
    const uint m = st + c - s;
    if (t.y == 1) it12[4 * b12] = gid, it12[4 * b12 + 1] = s, it12[4 * b12 + 2] = s, it12[4 * b12 + 3] = m;
    if (t.y == 2) it8[4 * b8] = gid, it8[4 * b8 + 1] = s, it8[4 * b8 + 2] = s, it8[4 * b8 + 3] = m;
    if (t.y == 3) it4[4 * b4] = gid, it4[4 * b4 + 1] = s, it4[4 * b4 + 2] = s, it4[4 * b4 + 3] = m;
    if (t.y == 4) it2[4 * b2] = gid, it2[4 * b2 + 1] = s, it2[4 * b2 + 2] = s, it2[4 * b2 + 3] = m;
    if (gid + 1 == p.a[1]) {
        cnt[0] = b16 + t.x, cnt[1] = b12 + (t.y == 1 ? 1u : 0u), cnt[2] = b8 + (t.y == 2 ? 1u : 0u);
        cnt[3] = b4 + (t.y == 3 ? 1u : 0u), cnt[4] = b2 + (t.y == 4 ? 1u : 0u);
    }
}
// xslot: slot_of[pair], prow[slot] = the pair's row. thread (pair)
kernel void xslot(constant P &p [[buffer(0)]], device const uint *idx [[buffer(1)]], device const uint *cc [[buffer(2)]],
                  device const uint *start [[buffer(3)]], device uint *slot_of [[buffer(4)]], device uint *prow [[buffer(5)]],
                  uint gid [[thread_position_in_grid]]) {
    const uint e = idx[gid], c = gid / 64;
    uint rank = 0;
    for (uint i = c * 64; i < gid; ++i) rank += idx[i] == e ? 1u : 0u;
    const uint s = start[e] + cc[e * p.a[3] + c] + rank;
    slot_of[gid] = s, prow[s] = gid / p.a[2];
}

// a few rows: every pair an item of its own, in pair order (x13 reads the row's x, x2 the pair's
// GLU output). thread (pair). a: k
kernel void pairs(constant P &p [[buffer(0)]], device const uint *idx [[buffer(1)]], device uint *ita [[buffer(2)]],
                  device uint *itb [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    ita[4 * gid] = idx[gid], ita[4 * gid + 1] = gid / p.a[0], ita[4 * gid + 2] = gid, ita[4 * gid + 3] = 1;
    itb[4 * gid] = idx[gid], itb[4 * gid + 1] = gid, itb[4 * gid + 2] = gid, itb[4 * gid + 3] = 1;
}

// dst[i] = src[rows[i]] (activations): thread (j / 4, i). a: W
kernel void rowgather(constant P &p [[buffer(0)]], device const ACT4 *src [[buffer(1)]], device const uint *rows [[buffer(2)]],
                      device ACT4 *dst [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    const uint W4 = p.a[0] / 4;
    dst[gid.y * W4 + gid.x] = src[rows[gid.y] * W4 + gid.x];
}

// ---- experts ----------------------------------------------------------------------------------
// One thread per (row n of an expert's matrix, item): the row is read and decoded once for the
// item's tokens. NVFP4: a byte is two E2M1 codes (the even element in the low nibble), an e4m3
// scale per 16 values, a factor per expert. MXFP8: e4m3 values, 2^(s - 127) per 32. The _b forms
// read bytes (stacks not on a 4-byte boundary).
// a: K, rows, ldy, which count, a fixed item count (0: cnt[which])
constant float e2m1[16] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};

#define XHEAD                                                                                              \
    if (gid.y >= (p.a[4] != 0 ? p.a[4] : cnt[p.a[3]])) return;                                             \
    const uint e = items[4 * gid.y], xs = items[4 * gid.y + 1], ys = items[4 * gid.y + 2], m = items[4 * gid.y + 3]; \
    const uint n = gid.x, K = p.a[0], row = e * p.a[1] + n, K4 = K / 4;                                    \
    float acc[TT];                                                                                         \
    for (uint t = 0; t < TT; ++t) acc[t] = 0.0f;
#define XARGS(WT)                                                                                          \
    constant P &p [[buffer(0)]], device const WT *w [[buffer(1)]], device const uchar *sc [[buffer(2)]],   \
    device const float *fac [[buffer(3)]], device const ACT4 *x [[buffer(4)]], device float *y [[buffer(5)]], \
    device const uint *items [[buffer(6)]], device const float *lut [[buffer(7)]], device const uint *cnt [[buffer(8)]], \
    uint2 gid [[thread_position_in_grid]]

kernel void xrows_nv(XARGS(uint)) {
    XHEAD
    const uint wo = row * (K / 8), so = row * (K / 16);
    for (uint b = 0; b < K / 16; ++b) {
        const float scale = lut[sc[so + b]];
        for (uint h = 0; h < 2; ++h) {
            uint c = w[wo + 2 * b + h];
            for (uint i = 0; i < 2; ++i) {
                // the products in the activations' type (the codes are exact in half), the block's
                // scale in float: folded into half weights it overflows (6 * 448 * x)
                const ACT4 wv = ACT4(float4(e2m1[c & 15u], e2m1[(c >> 4) & 15u], e2m1[(c >> 8) & 15u], e2m1[(c >> 12) & 15u]));
                c >>= 16;
                const uint xo = xs * K4 + 4 * b + 2 * h + i;
                for (uint t = 0; t < TT; ++t) {
                    const float4 pr = float4(wv * x[xo + t * K4]);
                    acc[t] += scale * ((pr.x + pr.y) + (pr.z + pr.w));
                }
            }
        }
    }
    const float f = fac[e];
    for (uint t = 0; t < m; ++t) y[(ys + t) * p.a[2] + n] = acc[t] * f;
}

kernel void xrows_mx(XARGS(uint)) {
    XHEAD
    const uint wo = row * (K / 4), so = row * (K / 32);
    for (uint b = 0; b < K / 32; ++b) {
        const float scale = exp2(float(sc[so + b]) - 127.0f);
        for (uint i = 0; i < 8; ++i) {
            const uint c = w[wo + 8 * b + i];
            const float4 wv = float4(lut[c & 255u], lut[(c >> 8) & 255u], lut[(c >> 16) & 255u], lut[c >> 24]) * scale;
            const uint xo = xs * K4 + 8 * b + i;
            for (uint t = 0; t < TT; ++t) acc[t] += dot(wv, float4(x[xo + t * K4]));
        }
    }
    for (uint t = 0; t < m; ++t) y[(ys + t) * p.a[2] + n] = acc[t] * fac[e];
}

kernel void xrows_nv_b(XARGS(uchar)) {
    XHEAD
    const uint wo = row * (K / 2), so = row * (K / 16);
    for (uint b = 0; b < K / 16; ++b) {
        const float scale = lut[sc[so + b]];
        for (uint i = 0; i < 4; ++i) {
            const uint b0 = w[wo + 8 * b + 2 * i], b1 = w[wo + 8 * b + 2 * i + 1];
            const float4 wv = float4(e2m1[b0 & 15u], e2m1[b0 >> 4], e2m1[b1 & 15u], e2m1[b1 >> 4]) * scale;
            const uint xo = xs * K4 + 4 * b + i;
            for (uint t = 0; t < TT; ++t) acc[t] += dot(wv, float4(x[xo + t * K4]));
        }
    }
    const float f = fac[e];
    for (uint t = 0; t < m; ++t) y[(ys + t) * p.a[2] + n] = acc[t] * f;
}

kernel void xrows_mx_b(XARGS(uchar)) {
    XHEAD
    const uint wo = row * K, so = row * (K / 32);
    for (uint b = 0; b < K / 32; ++b) {
        const float scale = exp2(float(sc[so + b]) - 127.0f);
        for (uint i = 0; i < 8; ++i) {
            const uint o = wo + 32 * b + 4 * i;
            const float4 wv = float4(lut[w[o]], lut[w[o + 1]], lut[w[o + 2]], lut[w[o + 3]]) * scale;
            const uint xo = xs * K4 + 8 * b + i;
            for (uint t = 0; t < TT; ++t) acc[t] += dot(wv, float4(x[xo + t * K4]));
        }
    }
    for (uint t = 0; t < m; ++t) y[(ys + t) * p.a[2] + n] = acc[t] * fac[e];
}

// gu [pairs][2 * I] (gate i, up i adjacent) -> act [pairs][I]: thread (i, pair). a: I; f0, f1: caps
kernel void glu(constant P &p [[buffer(0)]], device const float *gu [[buffer(1)]], device ACT *act [[buffer(2)]],
                uint2 gid [[thread_position_in_grid]]) {
    const uint I = p.a[0], i = gid.x, r = gid.y;
    const float g = gu[r * 2 * I + 2 * i], u = gu[r * 2 * I + 2 * i + 1];
    act[r * I + i] = ACT(tanh(g / p.f[0]) * p.f[0] * sigm(g) * (tanh(u / p.f[1]) * p.f[1]));
}

// out[row] = sum_j wt[row][j] * y2[the pair's slot]: thread (c / 4, row). a: H, k, identity slots
kernel void combine(constant P &p [[buffer(0)]], device const float4 *y2 [[buffer(1)]], device const float *wt [[buffer(2)]],
                    device const uint *slot_of [[buffer(3)]], device float4 *out [[buffer(4)]],
                    uint2 gid [[thread_position_in_grid]]) {
    const uint H4 = p.a[0] / 4, k = p.a[1], row = gid.y;
    float4 acc = float4(0.0f);
    for (uint j = 0; j < k; ++j) {
        const uint s = p.a[2] != 0 ? row * k + j : slot_of[row * k + j];
        acc += wt[row * k + j] * y2[s * H4 + gid.x];
    }
    out[row * H4 + gid.x] = acc;
}
)MSL";

struct PC {
    uint32_t a[16] = {};
    float f[4] = {};
};

struct BR {
    id<MTLBuffer> b = nil;
    NSUInteger off = 0;
    BR() {}
    BR(id<MTLBuffer> buf, NSUInteger o = 0) : b(buf), off(o) {}
};

struct Stack {
    id<MTLBuffer> w = nil, s = nil, f = nil;
    NSUInteger w_off = 0, s_off = 0;
};

struct LayerBufs {
    int fmt = -1;
    bool words = false;  // the stacks start on 4-byte boundaries: the kernels read words
    Stack x13, x2;
};

constexpr int kTiles[6] = {16, 12, 8, 4, 2, 1};
constexpr int kPad = 16;  // rows of padding after activations: a tile reads all its rows

}  // namespace

struct Ink2Metal {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;
    bool f32 = false;
    size_t asz = 2, kvsz = 2, page = 16384;  // bytes an activation, a cache element
    std::string name;
    std::unordered_map<std::string, id<MTLComputePipelineState>> ps;  // "kernel" or "kernel@tile"
    std::map<uintptr_t, std::pair<id<MTLBuffer>, size_t>> allocs;      // ink2_metal_alloc: base -> (buffer, bytes)
    std::map<const void *, id<MTLBuffer>> copies;                      // small host vectors, by pointer
    std::map<int, LayerBufs> xl;
    id<MTLBuffer> lut = nil;
    Ink2MetalModel M;
    bool ready = false;
    // the step being encoded
    id<MTLCommandBuffer> cb = nil;
    id<MTLComputeCommandEncoder> enc = nil;
    std::vector<id<MTLCommandBuffer>> sent;
    const sk_step_plan *pl = nullptr;
    int64_t N = 0, dispatches = 0;
    size_t arena_rows = 0;
    std::vector<id<MTLBuffer>> arena_free;
    std::unordered_map<std::string, id<MTLBuffer>> scratch;
    std::map<std::tuple<int64_t, int64_t, int64_t>, MPSMatrixMultiplication *> mms;
    double gpu_ms = 0, wall_ms = 0;
    // FULLM_INK2_PROF=2: every section committed and waited for, its GPU time by label
    bool sections = false, per_kernel = false;  // PROF=3: a section a dispatch, by kernel
    std::string last;
    // a sequence's runs of this many rows or more multiply their values by MPS (FULLM_INK2_METAL_PV_MIN)
    int64_t pv_min = 32;
    std::map<std::string, double> sec_ms;
    std::string why;
    bool failed = false;
};

namespace {

using FB = std::shared_ptr<BR>;  // an [N][hidden] f32 buffer of the step, returned to the arena when dropped

FB take(Ink2Metal *m) {
    id<MTLBuffer> b = nil;
    if (!m->arena_free.empty()) {
        b = m->arena_free.back();
        m->arena_free.pop_back();
    } else {
        b = [m->dev newBufferWithLength:m->arena_rows * size_t(m->M.d.hidden) * 4 options:MTLResourceStorageModeShared];
        if (b == nil) m->failed = true, m->why = "Metal buffer allocation failed";
    }
    return FB(new BR(b), [m](BR *r) {
        if (r->b != nil) m->arena_free.push_back(r->b);
        delete r;
    });
}

// a named buffer of the step, at least `bytes`
BR sb(Ink2Metal *m, const char *name, size_t bytes) {
    id<MTLBuffer> b = m->scratch[name];
    if (b == nil || b.length < bytes) {
        b = [m->dev newBufferWithLength:std::max<size_t>(bytes + bytes / 4, 1 << 14) options:MTLResourceStorageModeShared];
        if (b == nil) m->failed = true, m->why = "Metal buffer allocation failed";
        m->scratch[name] = b;
    }
    return BR(b);
}

// host memory as a buffer: inside an ink2_metal_alloc block, or copied once (by its pointer)
BR ref(Ink2Metal *m, const void *p, size_t bytes) {
    if (!p) return BR();
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    auto it = m->allocs.upper_bound(a);
    if (it != m->allocs.begin()) {
        --it;
        if (a >= it->first && a + bytes <= it->first + it->second.second) return BR(it->second.first, NSUInteger(a - it->first));
    }
    auto c = m->copies.find(p);
    if (c != m->copies.end() && c->second.length >= bytes) return BR(c->second);
    id<MTLBuffer> b = [m->dev newBufferWithBytes:p length:std::max<size_t>(bytes, 16) options:MTLResourceStorageModeShared];
    if (b == nil) m->failed = true, m->why = "Metal buffer allocation failed";
    m->copies[p] = b;
    return BR(b);
}

id<MTLComputePipelineState> pipe(Ink2Metal *m, const char *name, int tile = 0) {
    std::string key = name;
    if (tile) key += "@" + std::to_string(tile);
    m->last = key;
    return m->ps[key];
}

void begin(Ink2Metal *m) {
    m->cb = [m->queue commandBuffer];
    m->enc = [m->cb computeCommandEncoder];
}

// the encoded work to the GPU; the step goes on in a new command buffer
void flush(Ink2Metal *m, bool more) {
    [m->enc endEncoding];
    [m->cb commit];
    m->sent.push_back(m->cb);
    m->cb = nil, m->enc = nil;
    if (more) begin(m);
}

bool wait_all(Ink2Metal *m) {
    bool ok = true;
    for (id<MTLCommandBuffer> cb : m->sent) {
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            ok = false;
            m->why = std::string("Metal command buffer: ") + (cb.error ? cb.error.localizedDescription.UTF8String : "failed");
        }
        m->gpu_ms += (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
    }
    m->sent.clear();
    return ok;
}

// FULLM_INK2_PROF=2: what was encoded since the last mark ran as `label`
void mark(Ink2Metal *m, const char *label, bool kernel = false) {
    if (!m->sections || kernel != m->per_kernel) return;
    const double before = m->gpu_ms;
    flush(m, true);
    wait_all(m);
    m->sec_ms[label] += m->gpu_ms - before;
}

void run(Ink2Metal *m, id<MTLComputePipelineState> ps, const PC &p, std::initializer_list<BR> bufs, NSUInteger x, NSUInteger y = 1,
         NSUInteger z = 1) {
    if (x == 0 || y == 0 || z == 0 || m->failed) return;
    id<MTLComputeCommandEncoder> e = m->enc;
    [e setComputePipelineState:ps];
    [e setBytes:&p length:sizeof(p) atIndex:0];
    NSUInteger i = 1;
    for (const BR &b : bufs) {
        if (b.b != nil) [e setBuffer:b.b offset:b.off atIndex:i];
        ++i;
    }
    // a thread group as wide as the grid allows, the rest of the SIMD width down the rows
    const NSUInteger w = ps.threadExecutionWidth, gx = std::min(x, w), gy = std::max<NSUInteger>(1, std::min(y, w / gx));
    [e dispatchThreads:MTLSizeMake(x, y, z) threadsPerThreadgroup:MTLSizeMake(gx, gy, 1)];
    m->dispatches++;
    if (m->per_kernel) {
        const std::string label = m->last;
        mark(m, label.c_str(), true);
    }
}

NSUInteger row_bytes(NSUInteger cols) { return [MPSMatrixDescriptor rowBytesForColumns:cols dataType:MPSDataTypeFloat32]; }

MPSMatrix *matrix(BR b, NSUInteger rows, NSUInteger cols, NSUInteger rb) {
    return [[MPSMatrix alloc] initWithBuffer:b.b
                                      offset:b.off
                                  descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols rowBytes:rb dataType:MPSDataTypeFloat32]];
}

// ------------------------------------------------------------------------------------- the graph
struct Step {
    Step(Ink2Metal *m_, const sk_step_plan &pl_, int64_t n, int64_t h, int64_t k) : m(m_), pl(pl_), N(n), H(h), K(k) {}
    Ink2Metal *m;
    const sk_step_plan &pl;
    int64_t N, H, K;
    BR rowinfo, segs, pt, tau;
    int64_t max_keys = 0;  // the longest context a row of the step sees
    bool drafts = false;   // a sequence has two rows or more: they may end in drafts
    BR taps, ident;        // with a draft head: the features [N][4][H] of the step's rows; 0 .. N - 1
    int exec_i = 0;        // visits run so far
    BR ends;               // the rows that end a cache page (kept for the prefix cache)
    int64_t n_ends = 0;

    const sk_ink2_dims &d() const { return m->M.d; }
    size_t asz() const { return m->asz; }

    // y [N][rows] = x [N][cols] W^T
    void gemm(const void *w, int64_t rows, int64_t cols, BR x, BR y, int64_t n, float cap = 0.0f) {
        PC p;
        p.a[0] = uint32_t(cols), p.a[1] = uint32_t(rows), p.a[2] = uint32_t(n), p.f[0] = cap;
        BR W = ref(m, w, size_t(rows * cols) * 2);
        if (n == 1 || (n <= 64 && rows < 16384)) {
            // a few tiles: chunks of the columns, then their sums. (For a lone row, one thread a
            // whole matrix row was 10% slower even for the 3072-row matrices; for 2 to 64 rows the
            // unchunked tiles ran at a third of their rate. The output head has rows enough.)
            const int t = n == 1 ? 1 : n <= 2 ? 2 : n <= 4 ? 4 : n <= 8 ? 8 : n <= 12 ? 12 : 16;
            const int64_t nc = (cols + 255) / 256, tiles = (n + t - 1) / t;
            BR part = sb(m, "gemm_part", size_t(tiles * rows * nc * t) * 4);
            p.a[1] = uint32_t(nc), p.a[2] = uint32_t(rows), p.a[3] = uint32_t(rows);
            run(m, pipe(m, "gemmc", t), p, {W, x, part}, NSUInteger(rows), NSUInteger(nc), NSUInteger(tiles));
            run(m, pipe(m, "gsumt", t), p, {part, y}, NSUInteger(rows), NSUInteger(n));
            return;
        }
        run(m, pipe(m, "gemm", 16), p, {W, x, y}, NSUInteger(rows), NSUInteger((n + 15) / 16));
    }

    BR vec(const float *p, int64_t n) { return ref(m, p, size_t(n) * 4); }

    BR ss(BR x, const char *name) {
        const int64_t nc = (H + 255) / 256;
        BR part = sb(m, name, size_t(N * nc) * 4);
        PC p;
        p.a[0] = uint32_t(H), p.a[1] = uint32_t(nc);
        run(m, pipe(m, "sspart"), p, {x, part}, NSUInteger(nc), NSUInteger(N));
        return part;
    }

    // x normalised with gain w, as activations
    void rms(BR x, const float *w, BR out) {
        BR part = ss(x, "rms_part");
        PC p;
        p.a[0] = uint32_t(H), p.a[1] = uint32_t((H + 255) / 256), p.f[0] = d().rms_eps;
        run(m, pipe(m, "rms_act"), p, {x, part, vec(w, H), out}, NSUInteger(H / 4), NSUInteger(N));
    }

    std::vector<FB> ares(const float *queries, int nq, const std::vector<FB> &blocks, const FB &partial) {
        const int64_t n = int64_t(blocks.size()) + 1, nc = (H + 255) / 256;
        std::vector<BR> src;
        for (const FB &b : blocks) src.push_back(*b);
        src.push_back(*partial);
        while (src.size() < 6) src.push_back(src.back());
        BR part = sb(m, "ares_part", size_t(N * n * nc) * 5 * 4), wts = sb(m, "ares_wts", size_t(N * nq) * 8 * 4);
        std::vector<FB> out;
        for (int i = 0; i < nq; ++i) out.push_back(take(m));
        PC p;
        p.a[0] = uint32_t(H), p.a[1] = uint32_t(n), p.a[2] = uint32_t(nq), p.a[3] = uint32_t(nc);
        p.f[0] = 1.0f / std::sqrt(float(H));
        run(m, pipe(m, "ares_part"), p, {vec(queries, nq * H), part, src[0], src[1], src[2], src[3], src[4], src[5]}, NSUInteger(nc),
            NSUInteger(N * n));
        run(m, pipe(m, "ares_wts"), p, {part, wts}, NSUInteger(nq), NSUInteger(N));
        run(m, pipe(m, "ares_mix"), p,
            {wts, src[0], src[1], src[2], src[3], src[4], src[5], *out[0], *out[size_t(std::min(1, nq - 1))],
             *out[size_t(std::min(2, nq - 1))], *out[size_t(std::min(3, nq - 1))]},
            NSUInteger(H / 4), NSUInteger(N));
        return out;
    }

    FB gate(const Ink2MetalLayer &L, int which, const FB &h, const FB &msg) {
        const int64_t R = d().gate_rank, nc = (H + 255) / 256;
        BR hp = ss(*h, "gate_hp"), mp = ss(*msg, "gate_mp"), g = sb(m, "gate_g", size_t(N * R) * 4);
        BR cat = sb(m, "gate_cat", size_t(N + kPad) * size_t(2 * H) * asz());
        FB out = take(m);
        PC p;
        p.a[0] = uint32_t(H), p.a[1] = uint32_t(nc), p.f[0] = d().rms_eps;
        run(m, pipe(m, "gate_cat"), p, {*h, *msg, hp, mp, cat}, NSUInteger(H / 4), NSUInteger(N));
        gemm(L.c_down[which], R, 2 * H, cat, g, N);
        p.a[1] = uint32_t(R);
        if (N > 1 && R % 4 == 0) {  // the two gates as matrices over the step's rows
            BR ga = sb(m, "gate_ga", size_t(N + kPad) * size_t(R) * asz()), f = sb(m, "gate_f", size_t(N * H) * 4);
            BR gn = sb(m, "gate_gn", size_t(N * H) * 4);
            PC t;
            t.a[0] = uint32_t(R);
            run(m, pipe(m, "toact"), t, {g, ga}, NSUInteger(R / 4), NSUInteger(N));
            gemm(L.c_forget[which], H, R, ga, f, N), gemm(L.c_gain[which], H, R, ga, gn, N);
            run(m, pipe(m, "gate_fin"), p, {f, gn, *h, *msg, *out}, NSUInteger(H / 4), NSUInteger(N));
        } else {
            run(m, pipe(m, "gate_mix"), p,
                {ref(m, L.c_forget[which], size_t(H * R) * 2), ref(m, L.c_gain[which], size_t(H * R) * 2), g, *h, *msg, *out},
                NSUInteger(H), NSUInteger(N));
        }
        return out;
    }

    // state `which` of a visit: 0 keys, 1 values, 2 attention output, 3 mlp output
    uint32_t state_off(int vis, int which, int64_t Ck) const {
        const int64_t K1 = K - 1;
        const int64_t off[4] = {0, Ck * K1, 2 * Ck * K1, 2 * Ck * K1 + H * K1};
        return uint32_t(m->M.vis_off[size_t(vis)] + off[which]);
    }

    void sconv(BR x, BR y, const float *taps, BR dyn, int heads, int vis, int which, int64_t C, int64_t Ck, BR add) {
        PC p;
        p.a[0] = uint32_t(C), p.a[1] = uint32_t(K), p.a[2] = dyn.b != nil, p.a[3] = uint32_t(std::max(1, heads));
        p.a[4] = uint32_t(m->M.vis_off.back()), p.a[5] = state_off(vis, which, Ck), p.a[6] = add.b != nil;
        p.f[0] = 1.0f / std::sqrt(float(dyn.b != nil ? 2 * K + 1 : K + 1));
        BR conv = ref(m, m->M.conv, 4);
        run(m, pipe(m, "sconv"), p, {x, y, vec(taps, K * C), dyn.b != nil ? dyn : x, add.b != nil ? add : x, conv, rowinfo}, NSUInteger(C / 4),
            NSUInteger(N));
        if (n_ends > 0)
            run(m, pipe(m, "tail_save"), p2(C, vis, which, Ck), {x, conv, ends, ref(m, m->M.tails, 4)}, NSUInteger(C / 4),
                NSUInteger(n_ends));
        if (drafts) {
            PC u = p2(C, vis, which, Ck);
            const int64_t rows = K - 1 + m->M.undo_rows;
            u.a[4] = uint32_t(m->M.undo_stride), u.a[5] = uint32_t(int64_t(u.a[3]) / (K - 1) * rows), u.a[6] = uint32_t(m->M.undo_rows);
            run(m, pipe(m, "hist_save"), u, {x, conv, segs, ref(m, m->M.undo, 4)}, NSUInteger(C / 4), NSUInteger(pl.n_segs));
        }
        run(m, pipe(m, "hist_update"), p2(C, vis, which, Ck), {x, conv, segs}, NSUInteger(C / 4), NSUInteger(pl.n_segs));
    }
    PC p2(int64_t C, int vis, int which, int64_t Ck) const {
        PC p;
        p.a[0] = uint32_t(C), p.a[1] = uint32_t(K), p.a[2] = uint32_t(m->M.vis_off.back()), p.a[3] = state_off(vis, which, Ck);
        return p;
    }

    // the attention's output before `o`, as activations in xo [N][Cq]
    void attention(const Ink2MetalLayer &L, int vis, BR xq, BR xk, BR xv, BR xo) {
        const int64_t Hq = L.hq, Hk = L.hk, D = L.d, G = Hq / Hk, Cq = Hq * D, Ck = Hk * D, dr = d().d_rel, ext = L.extent;
        const bool full = !L.sliding;
        BR q = sb(m, "q", size_t(N * Cq) * 4), og = sb(m, "og", size_t(N * Cq) * 4), k = sb(m, "k", size_t(N * Ck) * 4);
        BR v = sb(m, "v", size_t(N * Ck) * 4), rel = sb(m, "rel", size_t(N * Hq * dr) * 4), sink = sb(m, "sink", size_t(N * Hq) * 4);
        BR kc = sb(m, "kc", size_t(N * Ck) * 4), vc = sb(m, "vc", size_t(N * Ck) * 4);
        BR qh = sb(m, "qh", size_t(Hk * N * G + kPad) * size_t(D) * asz());
        gemm(L.q, Cq, H, xq, q, N), gemm(L.og, Cq, H, xq, og, N), gemm(L.k, Ck, H, xk, k, N), gemm(L.v, Ck, H, xv, v, N);
        gemm(L.rel, Hq * dr, H, xq, rel, N), gemm(L.sink, Hq, H, xq, sink, N);
        mark(m, "attention: matrices");
        PC p;
        p.a[0] = uint32_t(D), p.a[1] = uint32_t(G), p.a[2] = uint32_t(Hq), p.a[3] = uint32_t(N), p.a[4] = full, p.f[0] = d().qk_clip;
        run(m, pipe(m, "qpost"), p, {q, vec(L.q_scale, Cq), tau, qh}, NSUInteger(Hq), NSUInteger(N));
        p = PC();
        p.a[0] = uint32_t(dr), p.a[1] = uint32_t(Hq), p.a[2] = full, p.f[0] = d().rel_clip;
        run(m, pipe(m, "relpost"), p, {rel, vec(L.rel_scale, Hq * dr), tau}, NSUInteger(Hq), NSUInteger(N));
        const int64_t nb = std::min<int64_t>(ext, max_keys), ldB = (nb + 3) / 4 * 4;
        BR B = sb(m, "bias", size_t(N * Hq * ldB) * 4);
        p = PC();
        p.a[0] = uint32_t(dr), p.a[1] = uint32_t(ext), p.a[2] = uint32_t(ldB);
        run(m, pipe(m, "relbias"), p, {rel, vec(L.rel_proj, dr * ext), B}, NSUInteger(ldB / 4), NSUInteger(N * Hq));
        if (full) {
            BR dk = sb(m, "dk", size_t(N * Hk * K) * 4), dv = sb(m, "dv", size_t(N * Hk * K) * 4);
            gemm(L.k_wp, Hk * K, H, xk, dk, N), gemm(L.v_wp, Hk * K, H, xv, dv, N);
            sconv(k, kc, L.k_static, dk, int(Hk), vis, 0, Ck, Ck, BR());
            sconv(v, vc, L.v_static, dv, int(Hk), vis, 1, Ck, Ck, BR());
        } else {
            sconv(k, kc, L.k_conv, BR(), 0, vis, 0, Ck, Ck, BR());
            sconv(v, vc, L.v_conv, BR(), 0, vis, 1, Ck, Ck, BR());
        }
        const size_t pool_bytes = size_t(m->M.o.n_pages) * size_t(m->M.o.page) * size_t(Ck) * m->kvsz;
        BR kp = ref(m, m->M.kpool[size_t(vis)], pool_bytes), vp = ref(m, m->M.vpool[size_t(vis)], pool_bytes);
        p = PC();
        p.a[0] = uint32_t(D), p.a[1] = uint32_t(Hk), p.a[2] = uint32_t(m->M.o.page), p.a[3] = uint32_t(m->M.o.max_pages_per_seq);
        p.a[4] = 0, p.f[0] = d().qk_clip;
        run(m, pipe(m, "kvstore"), p, {kc, kp, rowinfo, pt}, NSUInteger(Hk), NSUInteger(N));
        p.a[4] = 1, p.f[0] = d().rms_eps;
        run(m, pipe(m, "kvstore"), p, {vc, vp, rowinfo, pt}, NSUInteger(Hk), NSUInteger(N));
        mark(m, "attention: q, k, v");
        // a sequence's rows in runs of 256 at most (the scores of a run are [rows][keys])
        for (int s = 0; s < pl.n_segs; ++s)
          for (int64_t at = 0; at < pl.segs[3 * s + 2]; at += 256) {
            const int64_t slot = pl.segs[3 * s], first = pl.segs[3 * s + 1] + at, n = std::min<int64_t>(256, pl.segs[3 * s + 2] - at);
            const int64_t pos0 = pl.positions[first];
            const int64_t k0 = full ? 0 : std::max<int64_t>(0, pos0 + 1 - d().window), nk = pos0 + n - k0, nG = n * G;
            const int64_t ldT = int64_t(row_bytes(NSUInteger(nk)) / 4), nch = (nk + 63) / 64, ldD = int64_t(row_bytes(NSUInteger(D)) / 4);
            BR S = sb(m, "S", size_t(Hk * nG * ldT) * 4), cm = sb(m, "cm", size_t(Hk * nG * nch) * 4);
            BR cs = sb(m, "cs", size_t(Hk * nG * nch) * 4), den = sb(m, "den", size_t(Hk * nG) * 4);
            PC a;
            a.a[0] = uint32_t(G), a.a[1] = uint32_t(nk), a.a[2] = uint32_t(ldT), a.a[3] = uint32_t(nb);
            a.a[4] = uint32_t(full ? 0 : d().window), a.a[5] = uint32_t(pos0), a.a[6] = uint32_t(k0), a.a[7] = uint32_t(Hq);
            a.a[8] = uint32_t(ldB), a.a[9] = uint32_t(first), a.a[10] = uint32_t(nch), a.a[11] = uint32_t(nG), a.a[12] = uint32_t(D);
            a.a[13] = uint32_t(slot), a.a[14] = uint32_t(m->M.o.page), a.a[15] = uint32_t(m->M.o.max_pages_per_seq);
            a.f[0] = 1.0f / std::sqrt(float(D)), a.f[1] = float(Hk), a.f[2] = float(N * G);
            const int qt = nG <= 4 ? 4 : 16;
            run(m, pipe(m, "attn_qk", qt), a, {kp, qh, S, pt}, NSUInteger(nk), NSUInteger((nG + qt - 1) / qt), NSUInteger(Hk));
            run(m, pipe(m, "sm_a"), a, {S, B, cm}, NSUInteger(nch), NSUInteger(nG), NSUInteger(Hk));
            run(m, pipe(m, "sm_b"), a, {S, sink, cm, cs}, NSUInteger(nch), NSUInteger(nG), NSUInteger(Hk));
            run(m, pipe(m, "sm_den"), a, {sink, cm, cs, den}, NSUInteger(nG), NSUInteger(Hk));
            if (n >= m->pv_min) {  // a prompt chunk: P V by MPS over the values gathered into rows
                BR vg = sb(m, "vg", size_t(Hk * nk * ldD) * 4), O = sb(m, "O", size_t(Hk * nG * ldD) * 4);
                PC gp = a;
                gp.a[2] = uint32_t(ldD);
                run(m, pipe(m, "kv_gather"), gp, {vp, pt, vg}, NSUInteger(D / 4), NSUInteger(nk), NSUInteger(Hk));
                [m->enc endEncoding];
                const auto key = std::make_tuple(nG, nk, D);
                MPSMatrixMultiplication *mm = nil;
                auto it = m->mms.find(key);
                if (it == m->mms.end()) {
                    if (m->mms.size() > 512) m->mms.clear();
                    mm = [[MPSMatrixMultiplication alloc] initWithDevice:m->dev transposeLeft:NO transposeRight:NO resultRows:NSUInteger(nG) resultColumns:NSUInteger(D) interiorColumns:NSUInteger(nk) alpha:1.0 beta:0.0];
                    m->mms[key] = mm;
                } else {
                    mm = it->second;
                }
                for (int64_t kh = 0; kh < Hk; ++kh)
                    [mm encodeToCommandBuffer:m->cb
                                   leftMatrix:matrix(BR(S.b, NSUInteger(kh * nG * ldT) * 4), NSUInteger(nG), NSUInteger(nk), NSUInteger(ldT) * 4)
                                  rightMatrix:matrix(BR(vg.b, NSUInteger(kh * nk * ldD) * 4), NSUInteger(nk), NSUInteger(D), NSUInteger(ldD) * 4)
                                 resultMatrix:matrix(BR(O.b, NSUInteger(kh * nG * ldD) * 4), NSUInteger(nG), NSUInteger(D), NSUInteger(ldD) * 4)];
                m->enc = [m->cb computeCommandEncoder];
                run(m, pipe(m, "ogate"), gp, {O, og, xo, den}, NSUInteger(D / 4), NSUInteger(nG), NSUInteger(Hk));
            } else {
                BR part = sb(m, "pv_part", size_t(Hk * nG * nch * D) * 4);
                run(m, pipe(m, "pv1"), a, {S, vp, pt, den, part}, NSUInteger(nch), NSUInteger(nG * D / 4), NSUInteger(Hk));
                run(m, pipe(m, "pv2"), a, {part, og, xo}, NSUInteger(D / 4), NSUInteger(nG), NSUInteger(Hk));
            }
        }
        mark(m, "attention: scoring");
    }

    void xrows(const LayerBufs &X, const Stack &st, int tile, BR x, BR y, int64_t Kc, int64_t rows, BR items, BR cnt, int which,
               int64_t fixed, int64_t max_items) {
        PC p;
        p.a[0] = uint32_t(Kc), p.a[1] = uint32_t(rows), p.a[2] = uint32_t(rows), p.a[3] = uint32_t(which), p.a[4] = uint32_t(fixed);
        const char *fn = X.fmt == SK_XFMT_NVFP4 ? (X.words ? "xrows_nv" : "xrows_nv_b") : (X.words ? "xrows_mx" : "xrows_mx_b");
        run(m, pipe(m, fn, tile), p, {BR(st.w, st.w_off), BR(st.s, st.s_off), BR(st.f), x, y, items, BR(m->lut), cnt}, NSUInteger(rows),
            NSUInteger(max_items));
    }

    // the routed experts of xm (activations) -> [N][H]
    FB moe(const Ink2MetalLayer &L, int li, BR xm) {
        const int64_t E = d().n_experts, k = d().topk, I = d().moe_inter, Pn = N * k;
        const LayerBufs &X = m->xl[li];
        BR lg = sb(m, "route_lg", size_t(N * E) * 4), idx = sb(m, "route_idx", size_t(Pn) * 4), wt = sb(m, "route_wt", size_t(Pn) * 4);
        gemm(L.router, E, H, xm, lg, N);
        PC p;
        p.a[0] = uint32_t(E), p.a[1] = uint32_t(k), p.f[0] = d().route_scale;
        run(m, pipe(m, "route_rank"), p, {lg, vec(L.router_bias, E), idx}, NSUInteger(E), NSUInteger(N));
        run(m, pipe(m, "route_wt"), p, {lg, idx, wt}, NSUInteger(k), NSUInteger(N));
        BR gu = sb(m, "gu", size_t(Pn + kPad) * size_t(2 * I) * 4), act = sb(m, "act", size_t(Pn + kPad) * size_t(I) * asz());
        BR y2 = sb(m, "y2", size_t(Pn + kPad) * size_t(H) * 4), cnt = sb(m, "cnt", 32);
        BR slot_of = sb(m, "slot_of", size_t(Pn) * 4);
        PC gl;
        gl.a[0] = uint32_t(I), gl.f[0] = d().gate_softcap, gl.f[1] = d().up_softcap;
        const bool sorted = N >= 16;  // (from 2 rows on: no faster, a draft step's rows share few experts)
        mark(m, "router");
        if (sorted) {
            // by expert: each expert's rows read once for its tokens
            const int64_t nch = (Pn + 63) / 64;
            // the most items a class can have: {16, 12, 8, 4, 2} hold at least {13, 9, 5, 3, 1} tokens
            static const int tile[5] = {16, 12, 8, 4, 2}, least[5] = {13, 9, 5, 3, 1};
            int64_t most[5];
            BR it[5];
            static const char *const names[5] = {"x_it16", "x_it12", "x_it8", "x_it4", "x_it2"};
            for (int c = 0; c < 5; ++c) {
                most[c] = c == 0 ? Pn / least[c] + 1 : std::min<int64_t>(E, Pn / least[c] + 1);
                it[c] = sb(m, names[c], size_t(most[c] + 1) * 16);
            }
            BR cc = sb(m, "x_cc", size_t(E * nch) * 4), tot = sb(m, "x_tot", size_t(E) * 4), start = sb(m, "x_start", size_t(E) * 4);
            BR prow = sb(m, "x_prow", size_t(Pn) * 4), xg = sb(m, "xg", size_t(Pn + kPad) * size_t(H) * asz());
            PC g;
            g.a[0] = uint32_t(Pn), g.a[1] = uint32_t(E), g.a[2] = uint32_t(k), g.a[3] = uint32_t(nch);
            run(m, pipe(m, "xcount"), g, {idx, cc}, NSUInteger(nch), NSUInteger(E));
            run(m, pipe(m, "xprefix"), g, {cc, tot}, NSUInteger(E));
            run(m, pipe(m, "xstart"), g, {tot, start, it[0], it[1], it[2], it[3], it[4], cnt}, NSUInteger(E));
            run(m, pipe(m, "xslot"), g, {idx, cc, start, slot_of, prow}, NSUInteger(Pn));
            PC rg;
            rg.a[0] = uint32_t(H);
            run(m, pipe(m, "rowgather"), rg, {xm, prow, xg}, NSUInteger(H / 4), NSUInteger(Pn));
            for (int c = 0; c < 5; ++c) xrows(X, X.x13, tile[c], xg, gu, H, 2 * I, it[c], cnt, c, 0, most[c]);
            run(m, pipe(m, "glu"), gl, {gu, act}, NSUInteger(I), NSUInteger(Pn));
            for (int c = 0; c < 5; ++c) xrows(X, X.x2, tile[c], act, y2, I, H, it[c], cnt, c, 0, most[c]);
        } else {
            BR ita = sb(m, "x_ita", size_t(Pn) * 16), itb = sb(m, "x_itb", size_t(Pn) * 16);
            PC g;
            g.a[0] = uint32_t(k);
            run(m, pipe(m, "pairs"), g, {idx, ita, itb}, NSUInteger(Pn));
            xrows(X, X.x13, 1, xm, gu, H, 2 * I, ita, cnt, 0, Pn, Pn);
            run(m, pipe(m, "glu"), gl, {gu, act}, NSUInteger(I), NSUInteger(Pn));
            xrows(X, X.x2, 1, act, y2, I, H, itb, cnt, 0, Pn, Pn);
        }
        FB out = take(m);
        PC c;
        c.a[0] = uint32_t(H), c.a[1] = uint32_t(k), c.a[2] = sorted ? 0 : 1;
        run(m, pipe(m, "combine"), c, {y2, wt, slot_of, *out}, NSUInteger(H / 4), NSUInteger(N));
        mark(m, X.fmt == SK_XFMT_NVFP4 ? "experts nvfp4" : "experts mxfp8");
        return out;
    }

    // rows[0 .. n) of the taps mixed by alpha's row and normalised with gain w, as activations
    void mixnorm(BR rows, int64_t n, int row, const float *w, BR out) {
        const int64_t nc = (H + 255) / 256;
        BR part = sb(m, "mix_part", size_t(n * nc) * 4), al = ref(m, m->M.mtp.alpha + 4 * row, 16);
        PC p;
        p.a[0] = uint32_t(H), p.a[1] = uint32_t(nc), p.f[0] = d().rms_eps;
        run(m, pipe(m, "mixpart"), p, {taps, al, rows, part}, NSUInteger(nc), NSUInteger(n));
        run(m, pipe(m, "mixapply"), p, {taps, al, rows, part, vec(w, H), out}, NSUInteger(H / 4), NSUInteger(n));
    }

    // The draft head's context: every row's keys and values, per draft layer, from the row's
    // features, into the head's pools at the row's position (a rejected draft's are written again).
    void observe() {
        const Ink2MetalMtp &T = m->M.mtp;
        const int64_t Ck = int64_t(T.hk) * T.d;
        const size_t pool_bytes = size_t(m->M.o.n_pages) * size_t(m->M.o.page) * size_t(Ck) * m->kvsz;
        BR ctx = sb(m, "dctx", size_t(N + kPad) * size_t(H) * asz()), kc = sb(m, "dkc", size_t(N * Ck) * 4);
        BR vc = sb(m, "dvc", size_t(N * Ck) * 4);
        for (size_t i = 0; i < T.layers.size(); ++i) {
            const Ink2MetalMtpLayer &L = T.layers[i];
            mixnorm(ident, N, int(i), L.dflare, ctx);
            gemm(L.k_ctx, Ck, H, ctx, kc, N), gemm(L.v_ctx, Ck, H, ctx, vc, N);
            PC p;
            p.a[0] = uint32_t(T.d), p.a[1] = uint32_t(T.hk), p.a[2] = uint32_t(m->M.o.page), p.a[3] = uint32_t(m->M.o.max_pages_per_seq);
            p.a[4] = 1, p.f[0] = d().rms_eps, p.f[1] = std::log(T.theta);
            run(m, pipe(m, "normrope_k"), p, {kc, vec(L.k_norm_ctx, T.d), rowinfo, pt, kc, ref(m, T.dk[i], pool_bytes)},
                NSUInteger(T.hk), NSUInteger(N));
            run(m, pipe(m, "kvcopy"), p, {vc, ref(m, T.dv[i], pool_bytes), rowinfo, pt}, NSUInteger(T.d / 4), NSUInteger(T.hk),
                NSUInteger(N));
        }
        mark(m, "draft head: context");
    }

    void tap(BR src, int t, bool act) {
        PC p;
        p.a[0] = uint32_t(H), p.a[1] = uint32_t(t);
        run(m, pipe(m, act ? "tapcopy_act" : "tapcopy"), p, {src, taps}, NSUInteger(H / 4), NSUInteger(N));
    }

    int visit(int l, int p) const { return p == 0 ? l : d().n_layers + (p - 1) * (d().loop_end - d().loop_start) + (l - d().loop_start); }

    FB layer(int l, int ps, const FB &h, const std::vector<FB> &blocks, const std::vector<FB> *msgs) {
        const Ink2MetalLayer &L = m->M.layers[size_t(l)];
        const int vis = visit(l, ps);
        const size_t hb = size_t(N + kPad) * size_t(H) * asz();
        std::vector<FB> ins;
        if (l > 0) ins = ares(L.attn_query, L.sliding ? 1 : 3, blocks, h);
        else ins.push_back(h);
        mark(m, "residual attention");
        BR xq = sb(m, "xq", hb), xk = xq, xv = xq;
        if (L.sliding && !msgs) {
            rms(*ins[0], L.in_norm, xq);
        } else {
            if (ins.size() == 1) ins.resize(3, ins[0]);
            if (msgs)
                for (int i = 0; i < 3; ++i) ins[size_t(i)] = gate(L, i, ins[size_t(i)], (*msgs)[size_t(i)]);
            xk = sb(m, "xk", hb), xv = sb(m, "xv", hb);
            rms(*ins[0], L.sliding ? L.in_norm : L.q_norm, xq);
            rms(*ins[1], L.sliding ? L.in_norm : L.k_norm, xk);
            rms(*ins[2], L.sliding ? L.in_norm : L.v_norm, xv);
        }
        ins.clear();
        mark(m, "gates, norms");
        const int64_t Cq = int64_t(L.hq) * L.d, Ck = int64_t(L.hk) * L.d;
        BR xo = sb(m, "xo", size_t(N + kPad) * size_t(Cq) * asz());
        attention(L, vis, xq, xk, xv, xo);
        FB a0 = take(m), a = take(m);
        gemm(L.o, H, Cq, xo, *a0, N);
        mark(m, "attention: matrices");
        // a block starts: the stream so far becomes a block state and the running sum restarts
        const bool boundary = l % d().block == 0;
        sconv(*a0, *a, L.attn_conv, BR(), 0, vis, 2, H, Ck, boundary ? BR() : *h);
        a0.reset();
        std::vector<FB> b2 = blocks;
        if (boundary) b2.push_back(h);
        FB mm = ares(L.mlp_query, 1, b2, a)[0];
        mark(m, "residual attention");
        if (msgs) mm = gate(L, 3, mm, (*msgs)[3]);
        BR xm = sb(m, "xm", hb);
        rms(*mm, L.post_norm, xm);
        mm.reset();
        mark(m, "gates, norms");
        FB y = moe(L, l, xm);
        FB out = take(m);
        sconv(*y, *out, L.mlp_conv, BR(), 0, vis, 3, H, Ck, *a);
        mark(m, "convolutions");
        return out;
    }

    void run_layers(int l0, int l1, int ps, FB &h, std::vector<FB> &blocks, const std::vector<FB> *prev_blocks, const FB *prev_h) {
        for (int l = l0; l < l1; ++l) {
            std::vector<FB> msgs;
            if (prev_h) msgs = ares(m->M.msg_queries + int64_t(l - d().loop_start) * 4 * H, 4, *prev_blocks, *prev_h);
            if (m->M.mtp.on)
                for (int t = 0; t < 3; ++t)
                    if (m->M.mtp.taps[t] == exec_i) tap(*h, t, false);
            ++exec_i;
            FB nh = layer(l, ps, h, blocks, prev_h ? &msgs : nullptr);
            if (l % d().block == 0) blocks.push_back(h);
            h = nh;
            // the GPU starts on these layers while the next are encoded
            if (!m->sections) flush(m, true);
        }
    }
};

}  // namespace

Ink2Metal *ink2_metal_create(bool f32_act, bool f32_kv, std::string &why) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (dev == nil) {
            why = "no Metal device";
            return nullptr;
        }
        NSError *err = nil;
        auto *m = new Ink2Metal;
        m->dev = dev;
        m->queue = [dev newCommandQueue];
        m->page = size_t(getpagesize());
        m->f32 = f32_act, m->asz = f32_act ? 4 : 2, m->kvsz = f32_kv ? 4 : 2;
        static const char *const plain[] = {"sspart", "rms_act", "ares_part", "ares_wts", "ares_mix", "gate_cat", "gate_mix", "gate_fin", "toact", "qpost",
                                            "relpost", "relbias", "sconv", "hist_save", "tail_save", "hist_update", "tapcopy", "tapcopy_act", "mixpart", "mixapply",
                                            "normrope_k", "normrope_q", "kvcopy", "dblock_qk", "dpv2", "add2", "silu_mul", "dcat", "mk_step",
                                            "mk_gate", "mk_max1", "mk_max2", "kvstore", "sm_a", "sm_b", "sm_den",
                                            "kv_gather", "ogate", "pv1", "pv2", "route_rank", "route_wt", "xcount", "xprefix", "xstart", "xslot", "pairs", "rowgather",
                                            "glu", "combine"};
        static const char *const tiled[] = {"gemm", "gemmc", "gsumt", "attn_qk", "xrows_nv", "xrows_mx", "xrows_nv_b", "xrows_mx_b"};
        bool ok = m->queue != nil;
        for (int ti = 0; ti < 6 && ok; ++ti) {
            MTLCompileOptions *opt = [MTLCompileOptions new];
            opt.preprocessorMacros = @{@"TT" : @(kTiles[ti]), @"ACT" : (f32_act ? @"float" : @"half"), @"ACT4" : (f32_act ? @"float4" : @"half4"),
                                       @"KV" : (f32_kv ? @"float" : @"half"), @"KV4" : (f32_kv ? @"float4" : @"half4")};
            id<MTLLibrary> lib = [dev newLibraryWithSource:kSource options:opt error:&err];
            auto add = [&](const char *fn, const std::string &key) {
                id<MTLFunction> f = lib == nil ? nil : [lib newFunctionWithName:@(fn)];
                id<MTLComputePipelineState> ps = f == nil ? nil : [dev newComputePipelineStateWithFunction:f error:&err];
                if (ps == nil) ok = false;
                else m->ps[key] = ps;
            };
            if (ti == 0)
                for (const char *fn : plain) add(fn, fn);
            for (const char *fn : tiled) add(fn, std::string(fn) + "@" + std::to_string(kTiles[ti]));
        }
        if (!ok) {
            why = std::string("Metal kernels: ") + (err ? err.localizedDescription.UTF8String : "unknown error");
            delete m;
            return nullptr;
        }
        m->lut = [dev newBufferWithBytes:sk_fp8_e4m3_table() length:256 * sizeof(float) options:MTLResourceStorageModeShared];
        m->name = dev.name.UTF8String;
        const char *pr = std::getenv("FULLM_INK2_PROF");
        m->sections = pr && (*pr == '2' || *pr == '3');
        m->per_kernel = pr && *pr == '3';
        if (const char *e = std::getenv("FULLM_INK2_METAL_PV_MIN")) m->pv_min = std::max<int64_t>(1, std::atoll(e));
        return m;
    }
}

void ink2_metal_destroy(Ink2Metal *m) { delete m; }
const char *ink2_metal_device(const Ink2Metal *m) { return m->name.c_str(); }

void *ink2_metal_alloc(Ink2Metal *m, size_t bytes) {
    @autoreleasepool {
        const size_t len = (std::max<size_t>(bytes, 16) + m->page - 1) & ~(m->page - 1);
        id<MTLBuffer> b = [m->dev newBufferWithLength:len options:MTLResourceStorageModeShared];
        if (b == nil) return nullptr;
        m->allocs[reinterpret_cast<uintptr_t>(b.contents)] = {b, len};
        return b.contents;
    }
}

namespace {

// The pages around [p, p + bytes) of a mapping as a buffer that copies nothing; off: p's offset in it.
id<MTLBuffer> wrap(Ink2Metal *m, const void *p, size_t bytes, NSUInteger *off) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p), base = a & ~uintptr_t(m->page - 1);
    const size_t len = ((a + bytes + m->page - 1) & ~uintptr_t(m->page - 1)) - base;
    *off = NSUInteger(a - base);
    return [m->dev newBufferWithBytesNoCopy:reinterpret_cast<void *>(base) length:len options:MTLResourceStorageModeShared deallocator:nil];
}

}  // namespace

bool ink2_metal_add_layer(Ink2Metal *m, int layer, int fmt, int n_experts, int hidden, int inter, const uint8_t *w13,
                          const uint8_t *s13, const float *f13, const uint8_t *w2, const uint8_t *s2, const float *f2) {
    @autoreleasepool {
        if (fmt != SK_XFMT_NVFP4 && fmt != SK_XFMT_MXFP8) return false;
        const bool nv = fmt == SK_XFMT_NVFP4;
        const size_t E = size_t(n_experts), H = size_t(hidden), I = size_t(inter);
        const size_t n13 = E * 2 * I * H, n2 = E * H * I;
        LayerBufs L;
        L.fmt = fmt;
        L.x13.w = wrap(m, w13, nv ? n13 / 2 : n13, &L.x13.w_off);
        L.x13.s = wrap(m, s13, nv ? n13 / 16 : n13 / 32, &L.x13.s_off);
        L.x2.w = wrap(m, w2, nv ? n2 / 2 : n2, &L.x2.w_off);
        L.x2.s = wrap(m, s2, nv ? n2 / 16 : n2 / 32, &L.x2.s_off);
        std::vector<float> one(E, 1.0f);
        L.x13.f = [m->dev newBufferWithBytes:(nv ? f13 : one.data()) length:E * 4 options:MTLResourceStorageModeShared];
        L.x2.f = [m->dev newBufferWithBytes:(nv ? f2 : one.data()) length:E * 4 options:MTLResourceStorageModeShared];
        if (L.x13.w == nil || L.x13.s == nil || L.x2.w == nil || L.x2.s == nil || L.x13.f == nil || L.x2.f == nil) return false;
        L.words = (reinterpret_cast<uintptr_t>(w13) % 4 == 0) && (reinterpret_cast<uintptr_t>(w2) % 4 == 0) && H % 32 == 0 && I % 32 == 0;
        m->xl[layer] = L;
        return true;
    }
}

bool ink2_metal_setup(Ink2Metal *m, const Ink2MetalModel &M, std::string &why) {
    const sk_ink2_dims &d = M.d;
    auto bad = [&](const char *s) {
        why = s;
        return false;
    };
    if (d.hidden % 4 || d.moe_inter % 4 || d.d_rel % 4 || d.full_d % 4 || d.swa_d % 4) return bad("a width is not a multiple of 4");
    if (d.n_layers / d.block + 1 > 6) return bad("more than 6 block states");
    if (d.passes > 1 && d.gate_rank <= 0) return bad("no gate rank");
    for (int l = 0; l < d.n_layers; ++l)
        if (!m->xl.count(l)) return bad("a layer's experts are not NVFP4 or MXFP8, or could not be mapped");
    if (!M.conv || M.kpool.size() != M.vpool.size() || M.vis_off.empty()) return bad("no state");
    m->M = M;
    m->M.d.layer_sliding = nullptr;
    m->ready = true;
    return true;
}

bool ink2_metal_step(Ink2Metal *m, const sk_step_plan &pl, const float *h0, float *logits, std::string &why) {
    @autoreleasepool {
        if (!m->ready) {
            why = "the model is not set up";
            return false;
        }
        const CFTimeInterval wall0 = CACurrentMediaTime();
        const sk_ink2_dims &d = m->M.d;
        const sk_exec_opts &o = m->M.o;
        const int64_t N = pl.n_tokens, R = pl.n_rows, H = d.hidden, V = d.out_vocab;
        m->failed = false, m->gpu_ms = 0, m->dispatches = 0, m->pl = &pl, m->N = N;
        if (size_t(N + kPad) > m->arena_rows) m->arena_free.clear(), m->arena_rows = size_t(N + kPad);
        Step s(m, pl, N, H, d.conv_k);
        // the step's tables
        s.rowinfo = sb(m, "rowinfo", size_t(N) * 16), s.segs = sb(m, "segs", size_t(pl.n_segs) * 16);
        s.pt = sb(m, "pt", size_t(o.n_slots) * size_t(o.max_pages_per_seq) * 4), s.tau = sb(m, "tau", size_t(N) * 4);
        BR rows = sb(m, "rows", size_t(R) * 4);
        if (m->failed) {
            why = m->why;
            return false;
        }
        {
            uint32_t *ri = static_cast<uint32_t *>(s.rowinfo.b.contents), *sg = static_cast<uint32_t *>(s.segs.b.contents);
            float *tau = static_cast<float *>(s.tau.b.contents);
            for (int i = 0; i < pl.n_segs; ++i) {
                const int32_t slot = pl.segs[3 * i], first = pl.segs[3 * i + 1], n = pl.segs[3 * i + 2];
                sg[4 * i] = uint32_t(slot), sg[4 * i + 1] = uint32_t(first), sg[4 * i + 2] = uint32_t(n), sg[4 * i + 3] = 0;
                s.drafts = s.drafts || (n >= 2 && m->M.undo != nullptr);
                for (int j = 0; j < n; ++j) {
                    const int r = first + j;
                    ri[4 * r] = uint32_t(slot), ri[4 * r + 1] = uint32_t(pl.positions[r]), ri[4 * r + 2] = uint32_t(first), ri[4 * r + 3] = uint32_t(i);
                    tau[r] = 1.0f + d.log_alpha * std::log(std::max(1.0f, float(pl.positions[r] + 1) / d.log_n_floor));
                    s.max_keys = std::max<int64_t>(s.max_keys, pl.positions[r] + 1);
                }
            }
            if (m->M.tails) {
                std::vector<uint32_t> ends;
                for (int i = 0; i < pl.n_segs; ++i) {
                    const int64_t slot = pl.segs[3 * i], first = pl.segs[3 * i + 1], n = pl.segs[3 * i + 2];
                    const int64_t pos0 = pl.positions[first], pg = o.page;
                    for (int64_t t = pg - 1 - pos0 % pg; t < n; t += pg)
                        ends.insert(ends.end(), {uint32_t(slot), uint32_t(first), uint32_t(first + t),
                                                 uint32_t(pl.page_table[slot * o.max_pages_per_seq + (pos0 + t) / pg])});
                }
                s.n_ends = int64_t(ends.size() / 4);
                s.ends = sb(m, "ends", ends.size() * 4 + 16);
                if (s.ends.b != nil) std::memcpy(s.ends.b.contents, ends.data(), ends.size() * 4);
            }
            std::memcpy(s.pt.b.contents, pl.page_table, size_t(o.n_slots) * size_t(o.max_pages_per_seq) * 4);
            uint32_t *rw = static_cast<uint32_t *>(rows.b.contents);
            for (int64_t i = 0; i < R; ++i) rw[i] = uint32_t(pl.rows[i]);
        }
        if (m->M.mtp.on) {
            s.taps = sb(m, "taps", size_t(N) * 4 * size_t(H) * 4), s.ident = sb(m, "ident", size_t(N) * 4);
            if (s.ident.b != nil)
                for (int64_t i = 0; i < N; ++i) static_cast<uint32_t *>(s.ident.b.contents)[i] = uint32_t(i);
        }
        begin(m);
        FB h = take(m);
        if (m->failed) {
            why = m->why;
            return false;
        }
        std::memcpy(h->b.contents, h0, size_t(N * H) * 4);
        std::vector<FB> blocks;
        s.run_layers(0, d.loop_start, 0, h, blocks, nullptr, nullptr);
        const FB pre_h = h;
        const std::vector<FB> pre_blocks = blocks;
        s.run_layers(d.loop_start, d.loop_end, 0, h, blocks, nullptr, nullptr);
        for (int p = 1; p < d.passes; ++p) {
            // the previous pass's block states past the one it started with, and its output
            const std::vector<FB> prev_blocks(blocks.begin() + std::min(pre_blocks.size() + 1, blocks.size()), blocks.end());
            const FB prev_h = h;
            h = pre_h, blocks = pre_blocks;
            s.run_layers(d.loop_start, d.loop_end, p, h, blocks, &prev_blocks, &prev_h);
        }
        s.run_layers(d.loop_end, d.n_layers, 0, h, blocks, nullptr, nullptr);
        FB fin = s.ares(m->M.out_query, 1, blocks, h)[0];
        BR xf = sb(m, "xf", size_t(N + kPad) * size_t(H) * m->asz), xsel = sb(m, "xsel", size_t(R + kPad) * size_t(H) * m->asz);
        BR lg = sb(m, "logits", size_t(R * V) * 4);
        s.rms(*fin, m->M.final_norm, xf);
        if (m->M.mtp.on) {
            s.tap(xf, 3, true);
            s.observe();
        }
        PC rg;
        rg.a[0] = uint32_t(H);
        run(m, pipe(m, "rowgather"), rg, {xf, rows, xsel}, NSUInteger(H / 4), NSUInteger(R));
        s.gemm(m->M.lm_head, V, H, xsel, lg, R, d.logit_softcap);
        mark(m, "output head");
        flush(m, false);
        const bool ok = wait_all(m) && !m->failed;
        if (ok) std::memcpy(logits, lg.b.contents, size_t(R * V) * 4);
        else why = m->why;
        m->wall_ms = (CACurrentMediaTime() - wall0) * 1e3;
        if (m->sections) {
            std::string line = "   GPU sections (ms):";
            char buf[96];
            for (auto &kv : m->sec_ms) {
                std::snprintf(buf, sizeof buf, " %s %.1f;", kv.first.c_str(), kv.second);
                line += buf;
            }
            std::fprintf(stderr, "%s\n", line.c_str());
            m->sec_ms.clear();
        }
        m->pl = nullptr;
        return ok;
    }
}

bool ink2_metal_draft(Ink2Metal *m, int n, const int32_t *slots, const int32_t *pos, const int32_t *last_row, const int32_t *next,
                      const float *emb, const int32_t *page_table, int k, int32_t *out, float *conf, std::string &why) {
    @autoreleasepool {
        const Ink2MetalMtp &T = m->M.mtp;
        if (!m->ready || !T.on || n <= 0 || k <= 0 || k >= T.block) {
            why = "no draft head, or a draft count outside its block";
            return false;
        }
        const CFTimeInterval wall0 = CACurrentMediaTime();
        const sk_ink2_dims &d = m->M.d;
        const sk_exec_opts &o = m->M.o;
        const int64_t B = T.block, S = n, N = S * B, H = d.hidden, V = d.out_vocab, R = T.rank, I = T.inter;
        const int64_t Hq = T.hq, Hk = T.hk, D = T.d, G = Hq / Hk, Cq = Hq * D, Ck = Hk * D, SK = S * k;
        static const sk_step_plan none{};
        m->failed = false, m->gpu_ms = 0, m->dispatches = 0;
        if (size_t(N + kPad) > m->arena_rows) m->arena_free.clear(), m->arena_rows = size_t(N + kPad);
        Step s(m, none, N, H, d.conv_k);
        s.taps = sb(m, "taps", 16);  // the last step's
        s.rowinfo = sb(m, "d_rowinfo", size_t(N) * 16), s.pt = sb(m, "pt", size_t(o.n_slots) * size_t(o.max_pages_per_seq) * 4);
        BR lrows = sb(m, "d_lrows", size_t(S) * 4), sel = sb(m, "d_sel", size_t(SK) * 4), prev = sb(m, "d_prev", size_t(S) * 4);
        BR toks = sb(m, "d_toks", size_t(SK) * 4), state = sb(m, "d_state", size_t(S * R) * 4), neg = sb(m, "d_neg", size_t(N * Hq) * 4);
        FB x = take(m);
        if (m->failed) {
            why = m->why;
            return false;
        }
        {
            uint32_t *ri = static_cast<uint32_t *>(s.rowinfo.b.contents);
            float *xr = static_cast<float *>(x->b.contents);
            for (int64_t q = 0; q < S; ++q) {
                for (int64_t j = 0; j < B; ++j) {
                    const int64_t r = q * B + j;
                    ri[4 * r] = uint32_t(slots[q]), ri[4 * r + 1] = uint32_t(pos[q] + j), ri[4 * r + 2] = uint32_t(q * B), ri[4 * r + 3] = uint32_t(q);
                    std::memcpy(xr + r * H, j == 0 ? emb + q * H : T.mask, size_t(H) * 4);
                }
                static_cast<uint32_t *>(lrows.b.contents)[q] = uint32_t(last_row[q]);
                static_cast<uint32_t *>(prev.b.contents)[q] = uint32_t(next[q]);
                for (int i = 0; i < k; ++i) static_cast<uint32_t *>(sel.b.contents)[q * k + i] = uint32_t(q * B + 1 + i);
            }
            std::memcpy(s.pt.b.contents, page_table, size_t(o.n_slots) * size_t(o.max_pages_per_seq) * 4);
            std::memset(state.b.contents, 0, size_t(S * R) * 4);
            float *ng = static_cast<float *>(neg.b.contents);
            for (int64_t i = 0; i < N * Hq; ++i) ng[i] = -1e30f;
        }
        begin(m);
        // the recurrent head's feature: of each sequence's last position
        BR fr = sb(m, "d_fr", size_t(S + kPad) * size_t(H) * m->asz);
        s.mixnorm(lrows, S, int(T.layers.size()), T.m_dflare, fr);
        const size_t hb = size_t(N + kPad) * size_t(H) * m->asz, pool_bytes = size_t(o.n_pages) * size_t(o.page) * size_t(Ck) * m->kvsz;
        BR xa = sb(m, "xq", hb), q = sb(m, "q", size_t(N * Cq) * 4), kb = sb(m, "k", size_t(N * Ck) * 4), vb = sb(m, "v", size_t(N * Ck) * 4);
        BR qh = sb(m, "qh", size_t(Hk * N * G + kPad) * size_t(D) * m->asz), xo = sb(m, "xo", size_t(N + kPad) * size_t(Cq) * m->asz);
        BR gt = sb(m, "d_gate", size_t(N * I) * 4), up = sb(m, "d_up", size_t(N * I) * 4), act = sb(m, "d_act", size_t(N + kPad) * size_t(I) * m->asz);
        FB sum = x;
        for (size_t li = 0; li < T.layers.size(); ++li) {
            const Ink2MetalMtpLayer &L = T.layers[li];
            s.rms(*sum, L.in_norm, xa);
            s.gemm(L.q, Cq, H, xa, q, N), s.gemm(L.k, Ck, H, xa, kb, N), s.gemm(L.v, Ck, H, xa, vb, N);
            PC p;
            p.a[0] = uint32_t(D), p.a[1] = uint32_t(Hq), p.a[2] = uint32_t(G), p.a[3] = uint32_t(N), p.f[0] = d.rms_eps, p.f[1] = std::log(T.theta);
            run(m, pipe(m, "normrope_q"), p, {q, s.vec(L.q_norm, D), s.rowinfo, qh}, NSUInteger(Hq), NSUInteger(N));
            p.a[1] = uint32_t(Hk), p.a[2] = uint32_t(o.page), p.a[3] = uint32_t(o.max_pages_per_seq), p.a[4] = 0;
            run(m, pipe(m, "normrope_k"), p, {kb, s.vec(L.k_norm, D), s.rowinfo, s.pt, kb, kb}, NSUInteger(Hk), NSUInteger(N));
            BR kp = ref(m, T.dk[li], pool_bytes), vp = ref(m, T.dv[li], pool_bytes);
            for (int64_t sq = 0; sq < S; ++sq) {
                const int64_t p0 = pos[sq], lo = L.local ? std::max<int64_t>(0, p0 - T.window) : 0, nk = p0 - lo, nt = nk + B, nG = B * G;
                const int64_t ldT = int64_t(row_bytes(NSUInteger(nt)) / 4), nch = (nt + 63) / 64, nchc = (nk + 63) / 64;
                BR Sc = sb(m, "S", size_t(Hk * nG * ldT) * 4), cm = sb(m, "cm", size_t(Hk * nG * nch) * 4), cs = sb(m, "cs", size_t(Hk * nG * nch) * 4);
                BR den = sb(m, "den", size_t(Hk * nG) * 4), part = sb(m, "pv_part", size_t(Hk * nG * std::max<int64_t>(1, nchc) * D) * 4);
                PC a;
                a.a[0] = uint32_t(G), a.a[1] = uint32_t(nt), a.a[2] = uint32_t(ldT), a.a[3] = 0, a.a[4] = 0, a.a[5] = uint32_t(lo + nt);
                a.a[6] = uint32_t(lo), a.a[7] = uint32_t(Hq), a.a[8] = 1, a.a[9] = uint32_t(sq * B), a.a[10] = uint32_t(nch), a.a[11] = uint32_t(nG);
                a.a[12] = uint32_t(D), a.a[13] = uint32_t(slots[sq]), a.a[14] = uint32_t(o.page), a.a[15] = uint32_t(o.max_pages_per_seq);
                a.f[0] = 1.0f / std::sqrt(float(D)), a.f[1] = float(Hk), a.f[2] = float(N * G), a.f[3] = float(B);
                if (nk > 0) run(m, pipe(m, "attn_qk", 16), a, {kp, qh, Sc, s.pt}, NSUInteger(nk), NSUInteger((nG + 15) / 16), NSUInteger(Hk));
                run(m, pipe(m, "dblock_qk"), a, {qh, kb, Sc}, NSUInteger(B), NSUInteger(nG), NSUInteger(Hk));
                run(m, pipe(m, "sm_a"), a, {Sc, neg, cm}, NSUInteger(nch), NSUInteger(nG), NSUInteger(Hk));
                run(m, pipe(m, "sm_b"), a, {Sc, neg, cm, cs}, NSUInteger(nch), NSUInteger(nG), NSUInteger(Hk));
                run(m, pipe(m, "sm_den"), a, {neg, cm, cs, den}, NSUInteger(nG), NSUInteger(Hk));
                PC c = a;
                c.a[1] = uint32_t(nk), c.a[10] = uint32_t(nchc);
                if (nk > 0) run(m, pipe(m, "pv1"), c, {Sc, vp, s.pt, den, part}, NSUInteger(nchc), NSUInteger(nG * D / 4), NSUInteger(Hk));
                PC e = a;
                e.a[10] = uint32_t(nchc);
                run(m, pipe(m, "dpv2"), e, {part, Sc, vb, den, xo}, NSUInteger(D / 4), NSUInteger(nG), NSUInteger(Hk));
            }
            FB at = take(m), s2 = take(m), mo = take(m), nx = take(m);
            PC w;
            w.a[0] = uint32_t(H);
            s.gemm(L.o, H, Cq, xo, *at, N);
            run(m, pipe(m, "add2"), w, {*sum, *at, *s2}, NSUInteger(H / 4), NSUInteger(N));
            s.rms(*s2, L.post_norm, xa);
            s.gemm(L.gate, I, H, xa, gt, N), s.gemm(L.up, I, H, xa, up, N);
            PC wi;
            wi.a[0] = uint32_t(I);
            run(m, pipe(m, "silu_mul"), wi, {gt, up, act}, NSUInteger(I / 4), NSUInteger(N));
            s.gemm(L.down, H, I, act, *mo, N);
            run(m, pipe(m, "add2"), w, {*s2, *mo, *nx}, NSUInteger(H / 4), NSUInteger(N));
            sum = nx;
        }
        // the mask rows drafted: their logits through the model's head, and the recurrent head's inputs
        BR xh = sb(m, "xf", hb), xsel = sb(m, "xsel", size_t(SK + kPad) * size_t(H) * m->asz), base = sb(m, "d_base", size_t(SK * V) * 4);
        BR cat = sb(m, "d_cat", size_t(SK + kPad) * size_t(2 * H) * m->asz), cond = sb(m, "d_cond", size_t(SK * 3 * R) * 4);
        s.rms(*sum, T.norm, xh);
        PC g;
        g.a[0] = uint32_t(H), g.a[1] = uint32_t(k);
        run(m, pipe(m, "rowgather"), g, {xh, sel, xsel}, NSUInteger(H / 4), NSUInteger(SK));
        s.gemm(m->M.lm_head, V, H, xsel, base, SK);
        run(m, pipe(m, "dcat"), g, {xh, sel, fr, cat}, NSUInteger(H / 4), NSUInteger(SK));
        s.gemm(T.rnn_x, 3 * R, 2 * H, cat, cond, SK);
        // the chain: each draft from the state and the draft before it
        const int64_t chunks = (V + 255) / 256;
        BR u = sb(m, "d_u", size_t(S * 3 * R) * 4), lat = sb(m, "d_lat", size_t(S + kPad) * size_t(R) * m->asz), low = sb(m, "d_low", size_t(S * V) * 4);
        BR cv = sb(m, "d_cv", size_t(S * chunks) * 4), ci = sb(m, "d_ci", size_t(S * chunks) * 4);
        BR csum = sb(m, "d_cs", size_t(S * chunks) * 4), cf = sb(m, "d_conf", size_t(SK) * 4);
        for (int i = 0; i < k; ++i) {
            PC r;
            r.a[0] = uint32_t(R), r.a[1] = uint32_t(k), r.a[2] = uint32_t(i), r.a[3] = uint32_t(3 * R);
            run(m, pipe(m, "mk_step"), r,
                {ref(m, T.rnn_s, size_t(3 * R * 2 * R) * 2), state, ref(m, T.w1, 4), prev, cond, u, s.vec(T.rnn_b, 3 * R)}, NSUInteger(3 * R),
                NSUInteger(S));
            run(m, pipe(m, "mk_gate"), r, {u, state, lat}, NSUInteger(R), NSUInteger(S));
            s.gemm(T.fold, V, R, lat, low, S);
            PC x2;
            x2.a[0] = uint32_t(V), x2.a[1] = uint32_t(k), x2.a[2] = uint32_t(i), x2.a[3] = uint32_t(chunks), x2.f[0] = d.logit_softcap;
            run(m, pipe(m, "mk_max1"), x2, {base, low, s.vec(T.fold_b, V), cv, ci, csum}, NSUInteger(chunks), NSUInteger(S));
            run(m, pipe(m, "mk_max2"), x2, {cv, ci, prev, toks, csum, cf}, NSUInteger(S));
        }
        mark(m, "draft head");
        flush(m, false);
        const bool ok = wait_all(m) && !m->failed;
        if (ok) std::memcpy(out, toks.b.contents, size_t(SK) * 4), std::memcpy(conf, cf.b.contents, size_t(SK) * 4);
        else why = m->why;
        if (std::getenv("FULLM_INK2_PROF"))
            std::fprintf(stderr, "ink2 draft: %d sequences x %d tokens, %.1f ms: GPU %.1f ms in %lld dispatches\n", n, k,
                         (CACurrentMediaTime() - wall0) * 1e3, m->gpu_ms, static_cast<long long>(m->dispatches));
        return ok;
    }
}

void ink2_metal_last(const Ink2Metal *m, double *gpu_ms, double *wall_ms, int64_t *dispatches) {
    *gpu_ms = m->gpu_ms, *wall_ms = m->wall_ms, *dispatches = m->dispatches;
}

}  // namespace fullm
