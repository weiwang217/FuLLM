// SPDX-License-Identifier: Apache-2.0
// kernels/kimi.cuh - the Kimi family: block attention residuals and Kimi Delta Attention.
//
// Part of cuda_backend.cu's translation unit, included there after the CUDA and standard
// headers, in the order common, formats, gemv, convert, router, elementwise, copy, kimi, mla,
// attention, sampling, moe (each file uses only those before it). Kernels and their device
// helpers: no state.
#pragma once

namespace fullm {
namespace {

constexpr int kMaxSources = 32;

// One block per row. Sources: blocks[r][0..nb) then pref[r].
__global__ void attn_res_kernel(float *out, const float *blocks, int nb, int64_t maxs, const float *pref,
                                const float *__restrict__ fold, int64_t n, float eps) {
    __shared__ float red[32];
    __shared__ float prob[kMaxSources];
    const int64_t r = blockIdx.x;
    const int ns = nb + 1;
    for (int s = 0; s < ns; ++s) {
        const float *v = s < nb ? blocks + (r * maxs + s) * n : pref + r * n;
        float ss = 0.0f;
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) ss = fmaf(v[i], v[i], ss);
        ss = block_sum(ss, red);
        const float inv = 1.0f / sqrtf(ss / static_cast<float>(n) + eps);
        float d = 0.0f;
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) d = fmaf(v[i] * inv, fold[i], d);
        d = block_sum(d, red);
        if (threadIdx.x == 0) prob[s] = d;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = prob[0];
        for (int s = 1; s < ns; ++s) m = fmaxf(m, prob[s]);
        float z = 0.0f;
        for (int s = 0; s < ns; ++s) {
            prob[s] = expf(prob[s] - m);
            z += prob[s];
        }
        for (int s = 0; s < ns; ++s) prob[s] /= z;
    }
    __syncthreads();
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < ns; ++s) {
            const float *v = s < nb ? blocks + (r * maxs + s) * n : pref + r * n;
            acc = fmaf(prob[s], v[i], acc);
        }
        out[r * n + i] = acc;
    }
}

// ------------------------------------------------------------------ Kimi Delta Attention

constexpr int kMaxConvHist = 8;

// Thread per (segment, channel); positions in order within the segment.
__global__ void kda_conv_kernel(float *y, const float *x, const float *__restrict__ w, float *conv, int64_t stride,
                                int64_t hist_off, const int32_t *__restrict__ seg, int64_t C, int K) {
    const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int s = blockIdx.y;
    const int64_t slot = seg[3 * s], off = seg[3 * s + 1], len = seg[3 * s + 2];
    const int hist = K - 1;
    float *hp = conv + slot * stride + hist_off + c * hist;
    float buf[kMaxConvHist];
    for (int j = 0; j < hist; ++j) buf[j] = hp[j];
    const float *wc = w + c * K;
    for (int64_t t = 0; t < len; ++t) {
        const float cur = x[(off + t) * C + c];
        float acc = wc[hist] * cur;  // w[K-1] multiplies the CURRENT input
        for (int j = 0; j < hist; ++j) acc += wc[j] * buf[j];
        for (int j = 0; j + 1 < hist; ++j) buf[j] = buf[j + 1];
        if (hist > 0) buf[hist - 1] = cur;
        y[(off + t) * C + c] = acc * sigm(acc);  // SiLU, fused
    }
    for (int j = 0; j < hist; ++j) hp[j] = buf[j];
}

// torch softplus (beta 1, threshold 20)
__device__ __forceinline__ float softplusf(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

__global__ void kda_gates_kernel(float *alpha, float *beta, const float *z, const float *braw,
                                 const float *__restrict__ a_log, const float *__restrict__ dt_bias, int64_t N,
                                 int64_t H, int64_t D, float lb, int softplus) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t i0 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t HD = H * D;
    for (int64_t i = i0; i < N * HD; i += stride) {
        const int64_t hd = i % HD;
        const float a = expf(a_log[hd / D]);  // PER HEAD
        const float zz = z[i] + dt_bias[hd];
        const float g = softplus ? -a * softplusf(zz) : lb * sigm(a * zz);
        alpha[i] = expf(g);
    }
    for (int64_t i = i0; i < N * H; i += stride) beta[i] = sigm(braw[i]);
}

// One block per (head, segment); thread j owns column j of S, all D rows in registers.
// The delta rule then needs no cross-thread reduction: u_j, the write and o_j are all
// column-local. Order per token: decay -> read u = S^T k -> delta write -> o = S^T q.
template <int D, typename ST>
__global__ void __launch_bounds__(D)
    kda_rec_kernel(float *o, const float *q, const float *k, const float *v, const float *alpha, const float *beta,
                   ST *state, int64_t sstride, const int32_t *__restrict__ seg, int H, float qscale) {
    __shared__ float sq[D], sk[D], sa[D];
    const int h = blockIdx.x, s = blockIdx.y, j = threadIdx.x;
    const int64_t slot = seg[3 * s], off = seg[3 * s + 1], len = seg[3 * s + 2];
    ST *S = state + slot * sstride + static_cast<int64_t>(h) * D * D;
    float col[D];
#pragma unroll
    for (int i = 0; i < D; ++i) col[i] = ldst(S + static_cast<int64_t>(i) * D + j);
    const int64_t HD = static_cast<int64_t>(H) * D;
    for (int64_t t = 0; t < len; ++t) {
        const int64_t row = (off + t) * HD + static_cast<int64_t>(h) * D;
        __syncthreads();  // previous token's readers are done with the staging arrays
        sq[j] = q[row + j] * qscale;
        sk[j] = k[row + j];
        sa[j] = alpha[row + j];
        __syncthreads();
        const float vj = v[row + j];
        const float b = beta[(off + t) * H + h];
        float u = 0.0f;
#pragma unroll
        for (int i = 0; i < D; ++i) {
            col[i] *= sa[i];
            u += sk[i] * col[i];
        }
        const float wv = b * (vj - u);
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < D; ++i) {
            col[i] += sk[i] * wv;
            acc += sq[i] * col[i];  // read from the UPDATED state
        }
        o[row + j] = acc;
    }
#pragma unroll
    for (int i = 0; i < D; ++i) stst(S + static_cast<int64_t>(i) * D + j, col[i]);
}

// One warp per (token, head) vector: head-wise RMSNorm, then the output gate.
__global__ void kda_out_kernel(float *o, const float *gate, const float *__restrict__ w, int64_t nvec, int64_t D,
                               float eps) {
    const int64_t vec = static_cast<int64_t>(blockIdx.x) * 8 + (threadIdx.x >> 5);
    if (vec >= nvec) return;
    const int lane = threadIdx.x & 31;
    float *p = o + vec * D;
    const float *g = gate + vec * D;
    float ss = 0.0f;
    for (int64_t i = lane; i < D; i += 32) ss = fmaf(p[i], p[i], ss);
    ss = warp_sum(ss);
    const float inv = 1.0f / sqrtf(ss / static_cast<float>(D) + eps);
    for (int64_t i = lane; i < D; i += 32) p[i] = (w[i] * (p[i] * inv)) * sigm(g[i]);
}

// Sum over the D threads of a head block, the same value on every thread. D <= 32: one masked
// butterfly; otherwise warp butterflies combined in warp order through red (>= D/32 floats).
template <int D>
__device__ __forceinline__ float head_sum(float v, float *red) {
    if constexpr (D <= 32) {
        constexpr unsigned mask = D == 32 ? 0xffffffffu : (1u << D) - 1u;
#pragma unroll
        for (int o = D / 2; o > 0; o >>= 1) v += __shfl_xor_sync(mask, v, o);
        return v;
    } else {
        v = warp_sum(v);
        __syncthreads();  // red may still be read by the previous call
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
        __syncthreads();
        float t = 0.0f;
#pragma unroll
        for (int w = 0; w < D / 32; ++w) t += red[w];
        return t;
    }
}

// One ShortConv tap sequence for one channel (kda_conv_kernel's arithmetic), SiLU fused.
template <int KH>
__device__ __forceinline__ float conv_step(float cur, float (&buf)[KH], const float *__restrict__ w) {
    float acc = w[KH] * cur;  // w[K-1] multiplies the CURRENT input
#pragma unroll
    for (int j = 0; j < KH; ++j) acc += w[j] * buf[j];
#pragma unroll
    for (int j = 0; j + 1 < KH; ++j) buf[j] = buf[j + 1];
    buf[KH - 1] = cur;
    return acc * sigm(acc);
}

// The whole KDA token mixer for one (head, segment): thread j owns channel h*D + j of q, k
// and v (their ShortConv histories), the head's L2 norms, the gates, column j of the state
// (kda_rec_kernel's delta rule) and the output norm and gate. Replaces 8 launches per layer.
template <int D, int KH, typename ST>
__global__ void __launch_bounds__(D)
    kda_fused_kernel(float *o, const float *qp, const float *kp, const float *vp, const float *__restrict__ wq,
                     const float *__restrict__ wk, const float *__restrict__ wv, float *conv, int64_t cstride,
                     int64_t hist, const float *z, const float *braw, const float *__restrict__ a_log,
                     const float *__restrict__ dt_bias, float lb, int softplus, const float *gate,
                     const float *__restrict__ o_norm, float out_eps, float l2eps, float qscale, ST *state,
                     int64_t sstride, const int32_t *__restrict__ seg, int H) {
    __shared__ float sq[D], sk[D], sa[D], red[32];
    pdl_trigger();
    const int h = blockIdx.x, s = blockIdx.y, j = threadIdx.x;
    const int64_t slot = seg[3 * s], off = seg[3 * s + 1], len = seg[3 * s + 2];
    const int64_t P = static_cast<int64_t>(H) * D, c = static_cast<int64_t>(h) * D + j;
    float *hq = conv + slot * cstride + c * KH, *hk = hq + hist, *hv = hq + 2 * hist;
    float bq[KH], bk[KH], bv[KH];
#pragma unroll
    for (int i = 0; i < KH; ++i) bq[i] = hq[i], bk[i] = hk[i], bv[i] = hv[i];
    const float *cq = wq + c * (KH + 1), *ck = wk + c * (KH + 1), *cv = wv + c * (KH + 1);
    const float a = expf(a_log[h]), dtb = dt_bias[c], wn = o_norm[j];
    ST *S = state + slot * sstride + static_cast<int64_t>(h) * D * D;
    float col[D];
#pragma unroll
    for (int i = 0; i < D; ++i) col[i] = ldst(S + static_cast<int64_t>(i) * D + j);
    for (int64_t t = 0; t < len; ++t) {
        const int64_t row = (off + t) * P + c;
        __syncthreads();  // the previous token's readers are done with the staging arrays
        const float q = conv_step<KH>(qp[row], bq, cq);
        const float k = conv_step<KH>(kp[row], bk, ck);
        const float v = conv_step<KH>(vp[row], bv, cv);
        const float iq = 1.0f / sqrtf(head_sum<D>(q * q, red) + l2eps);
        const float ik = 1.0f / sqrtf(head_sum<D>(k * k, red) + l2eps);
        const float zz = z[row] + dtb;
        const float g = softplus ? -a * softplusf(zz) : lb * sigm(a * zz);
        sq[j] = (q * iq) * qscale;
        sk[j] = k * ik;
        sa[j] = expf(g);
        __syncthreads();
        const float b = sigm(braw[(off + t) * H + h]);
        float u = 0.0f;
#pragma unroll
        for (int i = 0; i < D; ++i) {
            col[i] *= sa[i];
            u += sk[i] * col[i];
        }
        const float wvv = b * (v - u);
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < D; ++i) {
            col[i] += sk[i] * wvv;
            acc += sq[i] * col[i];  // read from the UPDATED state
        }
        const float io = 1.0f / sqrtf(head_sum<D>(acc * acc, red) / static_cast<float>(D) + out_eps);
        o[row] = (wn * (acc * io)) * sigm(gate[row]);
    }
#pragma unroll
    for (int i = 0; i < KH; ++i) hq[i] = bq[i], hk[i] = bk[i], hv[i] = bv[i];
#pragma unroll
    for (int i = 0; i < D; ++i) stst(S + static_cast<int64_t>(i) * D + j, col[i]);
}

}  // namespace
}  // namespace fullm
