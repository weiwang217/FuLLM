// SPDX-License-Identifier: Apache-2.0
//! The small fp32 operators of the CPU backend. The matmuls live in fullm-kernels;
//! everything here is elementwise or a short reduction.
//!
//! Each function cites the behaviour it must reproduce. Several of these have a
//! plausible-looking wrong version that still produces fluent text, and those are the
//! comments worth reading.

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// torch's softplus (beta 1, threshold 20): log(1 + e^x), and x itself above 20.
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// `y = w * x / sqrt(mean(x^2) + eps)`. eps is inside the root.
pub fn rmsnorm(y: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let n = x.len();
    let ss: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let inv = (1.0 / (ss / n as f64 + eps as f64).sqrt()) as f32;
    for i in 0..n {
        y[i] = w[i] * (x[i] * inv);
    }
}

pub fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let n = x.len();
    let ss: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let inv = (1.0 / (ss / n as f64 + eps as f64).sqrt()) as f32;
    for i in 0..n {
        x[i] = w[i] * (x[i] * inv);
    }
}

/// `x / sqrt(sum(x^2) + eps)`: a SUM of squares, unlike RMSNorm's mean. q and k only.
pub fn l2norm_inplace(x: &mut [f32], eps: f32) {
    let ss: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let inv = (1.0 / (ss + eps as f64).sqrt()) as f32;
    for v in x {
        *v *= inv;
    }
}

/// SiTU-GLU over `[gate | up]` of width `2n`:
///   a = b1 * tanh(gate / b1) * sigmoid(gate)   -- the sigmoid sees the UNCAPPED gate
///   u = b2 * tanh(up / b2)
///   y = a * u                                   -- |y| <= b1 * b2
pub fn situ_glu(y: &mut [f32], gate: &[f32], up: &[f32], b1: f32, b2: f32) {
    for i in 0..y.len() {
        let g = gate[i];
        let a = b1 * (g / b1).tanh() * sigmoid(g);
        let u = b2 * (up[i] / b2).tanh();
        y[i] = a * u;
    }
}

/// SiLU-GLU over separate gate and up halves: y = silu(gate) * up (Kimi Linear, DeepSeek).
pub fn silu_glu(y: &mut [f32], gate: &[f32], up: &[f32]) {
    for i in 0..y.len() {
        let g = gate[i];
        y[i] = g * sigmoid(g) * up[i];
    }
}

/// The family's gated activation: SiTU-GLU (K3) or SiLU-GLU (Kimi Linear).
pub fn glu(y: &mut [f32], gate: &[f32], up: &[f32], silu: bool, b1: f32, b2: f32) {
    if silu {
        silu_glu(y, gate, up)
    } else {
        situ_glu(y, gate, up, b1, b2)
    }
}

/// Causal depthwise conv over T steps with fused SiLU, carrying K-1 inputs per channel.
///
/// `x`/`y` are `[T][C]`, `w` is `[C][K]` with `w[K-1]` on the CURRENT input, `state` is
/// `[C][K-1]` oldest first and is updated in place.
pub fn shortconv(y: &mut [f32], x: &[f32], w: &[f32], state: &mut [f32], c: usize, k: usize, t: usize) {
    let hist = k - 1;
    let mut buf = vec![0.0f32; hist];
    for ch in 0..c {
        buf.copy_from_slice(&state[ch * hist..(ch + 1) * hist]);
        let wc = &w[ch * k..(ch + 1) * k];
        for s in 0..t {
            let cur = x[s * c + ch];
            let mut acc = wc[hist] * cur;
            for j in 0..hist {
                acc += wc[j] * buf[j];
            }
            if hist > 0 {
                buf.copy_within(1.., 0);
                buf[hist - 1] = cur;
            }
            y[s * c + ch] = acc * sigmoid(acc);
        }
        state[ch * hist..(ch + 1) * hist].copy_from_slice(&buf);
    }
}

/// One KDA delta-rule step for one head. `s` is `[dk][dv]` row-major.
/// The ORDER is load-bearing (fla/ops/kda/naive.py:59-63 of flash-linear-attention, MIT; NOTICE):
///   decay rows by alpha -> read u = S^T k -> write S += k (beta (v - u))^T -> o = S^T q,
/// the output reading the ALREADY UPDATED state. `q` arrives pre-scaled by dk^-0.5.
pub fn kda_step(s: &mut [f32], o: &mut [f32], q: &[f32], k: &[f32], v: &[f32], alpha: &[f32], beta: f32) {
    let dk = q.len();
    let dv = v.len();
    assert!(dv <= 512, "kda_step: head_dim {dv} exceeds the stack buffer");
    let mut u = [0.0f32; 512];
    let u = &mut u[..dv];
    for i in 0..dk {
        let row = &mut s[i * dv..(i + 1) * dv];
        let a = alpha[i];
        let ki = k[i];
        for j in 0..dv {
            row[j] *= a;
            u[j] += ki * row[j];
        }
    }
    for j in 0..dv {
        u[j] = beta * (v[j] - u[j]);
    }
    o.fill(0.0);
    for i in 0..dk {
        let row = &mut s[i * dv..(i + 1) * dv];
        let ki = k[i];
        let qi = q[i];
        for j in 0..dv {
            row[j] += ki * u[j];
            o[j] += qi * row[j];
        }
    }
}

/// AttnRes: softmax over sources of `dot(rmsnorm_noweight(src), fold)`, applied to the RAW
/// sources. `fold = norm.weight * proj.weight`. `srcs` are the block snapshots followed by
/// the current prefix sum.
pub fn attn_res(out: &mut [f32], srcs: &[&[f32]], fold: &[f32], eps: f32) {
    let n = out.len();
    let mut score = Vec::with_capacity(srcs.len());
    for v in srcs {
        let ss: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let inv = (1.0 / (ss / n as f64 + eps as f64).sqrt()) as f32;
        let mut acc = 0.0f64;
        for i in 0..n {
            acc += ((v[i] * inv) * fold[i]) as f64;
        }
        score.push(acc as f32);
    }
    let m = score.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut z = 0.0f64;
    for s in &mut score {
        *s = (*s - m).exp();
        z += *s as f64;
    }
    out.fill(0.0);
    for (v, s) in srcs.iter().zip(&score) {
        let p = (*s as f64 / z) as f32;
        for i in 0..n {
            out[i] += p * v[i];
        }
    }
}

/// Sigmoid router with a selection-only bias. Returns `(experts, weights)` in selection
/// order. Weights come from the UNBIASED scores; using the biased ones still routes to
/// the same experts and silently perturbs the mixture.
pub fn route(
    logits: &[f32],
    bias: &[f32],
    topk: usize,
    renorm: bool,
    scale: f32,
    biased: bool,
) -> (Vec<u32>, Vec<f32>) {
    let score: Vec<f32> = logits.iter().map(|&l| sigmoid(l)).collect();
    let mut choice: Vec<f32> = score.iter().zip(bias).map(|(s, b)| s + b).collect();
    let mut idx = Vec::with_capacity(topk);
    let mut w = Vec::with_capacity(topk);
    for _ in 0..topk {
        // first index wins ties: deterministic, independent of batch
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (e, &c) in choice.iter().enumerate() {
            if c > bv {
                bv = c;
                best = e;
            }
        }
        idx.push(best as u32);
        // `biased` reproduces checkpoints whose reference code weights by score + bias
        w.push(if biased { score[best] + bias[best] } else { score[best] });
        choice[best] = f32::NEG_INFINITY;
    }
    if renorm {
        let s: f32 = w.iter().sum::<f32>() + 1e-20;
        for x in &mut w {
            *x /= s;
        }
    }
    for x in &mut w {
        *x *= scale;
    }
    (idx, w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn situ_is_bounded_and_uses_uncapped_sigmoid() {
        let mut y = [0.0f32; 2];
        situ_glu(&mut y, &[1000.0, 0.5], &[1000.0, 0.5], 4.0, 25.0);
        assert!((y[0] - 100.0).abs() < 1e-3);
        let a = 4.0 * (0.5f32 / 4.0).tanh() * sigmoid(0.5);
        assert!((y[1] - a * 25.0 * (0.5f32 / 25.0).tanh()).abs() < 1e-6);
    }

    #[test]
    fn router_bias_selects_but_does_not_weight() {
        let logits = [0.0, 1.0, 2.0];
        let bias = [10.0, 0.0, 0.0];
        let (idx, w) = route(&logits, &bias, 2, false, 1.0, false);
        assert_eq!(idx, vec![0, 2]);
        assert!((w[0] - 0.5).abs() < 1e-6, "expert 0 weight is its unbiased score");
        assert!((w[1] - sigmoid(2.0)).abs() < 1e-6);
    }
}
