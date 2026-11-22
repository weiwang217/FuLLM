// SPDX-License-Identifier: Apache-2.0
//! The reference backend for Inkling 2 (looped): f32 throughout, one sequence's chunk at a time,
//! every step written as tools/inkling2_ref.py writes it. It reproduces that torch oracle within
//! fp32 rounding and is what the executor backends are checked against; it is not a serving path
//! (it holds every weight as f32 and attends token by token).
//!
//! STATE PER SEQUENCE, per layer visit (48 for the released model): the keys and values of every
//! position (after their convolution, clip and norm: what attention reads), and the last K - 1
//! inputs of the visit's four short convolutions (keys, values, attention output, MLP output).
//!
//! A sequence's rows depend on nothing else in the step, so its logits are the same alone,
//! batched, or with its prompt split into chunks.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};

use super::{Ink2Config, PREFIX};
use crate::models::{Capacity, LoadOptions, Model, SeqChunk, SeqId};
use crate::ops::sigmoid;
use crate::st::{Dtype, SafeTensors};

/// E2M1 (NVFP4's 4-bit float) by code.
const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

/// The decoder's tensors as f32, by checkpoint name without the prefix.
struct Weights {
    t: HashMap<String, Vec<f32>>,
}

impl Weights {
    fn get(&self, name: &str) -> &[f32] {
        self.t.get(name).unwrap_or_else(|| panic!("tensor {name} was checked at load")).as_slice()
    }
}

/// An expert matrix stack ([E][rows][cols]) from the release's formats: bf16 / f32 as stored;
/// NVFP4 (`_amax` present): two E2M1 codes a byte, the even element in the low nibble, an e4m3 scale
/// per 16 values, and amax / outer_scale_denominator per expert; MXFP8: e4m3 values with a power
/// of two, 2^(scale - 127), per 32.
fn load_experts(st: &SafeTensors, name: &str, e: usize, rows: usize, cols: usize) -> Result<Vec<f32>> {
    let w = st.need(name)?;
    let e4m3 = fullm_kernels::fp8_e4m3_table();
    let out = match w.dtype() {
        Dtype::F32 | Dtype::BF16 | Dtype::F16 => w.to_f32()?,
        Dtype::U8 => {
            let scale = st.need(&format!("{name}_scale"))?;
            let amax = st.need(&format!("{name}_amax"))?.to_f32()?;
            let den = st.need(&format!("{name}_outer_scale_denominator"))?.to_f32()?;
            ensure!(scale.dtype() == Dtype::F8E4M3, "{name}_scale: expected F8_E4M3, found {:?}", scale.dtype());
            ensure!(cols % 16 == 0 && w.bytes.len() == e * rows * cols / 2, "{name}: NVFP4 codes of {:?}", w.shape());
            ensure!(
                scale.bytes.len() == e * rows * cols / 16 && amax.len() == e && den.len() == 1,
                "{name}: NVFP4 scales"
            );
            let mut out = vec![0.0f32; e * rows * cols];
            for (i, o) in out.iter_mut().enumerate() {
                let b = w.bytes[i / 2];
                let code = if i % 2 == 0 { b & 0xF } else { b >> 4 };
                *o = E2M1[code as usize] * e4m3[scale.bytes[i / 16] as usize] * (amax[i / (rows * cols)] / den[0]);
            }
            out
        }
        Dtype::F8E4M3 => {
            let scale = st.need(&format!("{name}_scale"))?;
            ensure!(scale.dtype() == Dtype::U8, "{name}_scale: expected U8 exponents, found {:?}", scale.dtype());
            ensure!(cols % 32 == 0 && w.bytes.len() == e * rows * cols, "{name}: MXFP8 values of {:?}", w.shape());
            ensure!(scale.bytes.len() == e * rows * cols / 32, "{name}: MXFP8 scales");
            let mut out = vec![0.0f32; e * rows * cols];
            for (i, o) in out.iter_mut().enumerate() {
                *o = e4m3[w.bytes[i] as usize] * 2.0f32.powi(scale.bytes[i / 32] as i32 - 127);
            }
            out
        }
        d => bail!("{name}: unsupported expert dtype {d:?}"),
    };
    ensure!(out.len() == e * rows * cols, "{name}: {} values, expected [{e}, {rows}, {cols}]", out.len());
    Ok(out)
}

fn bind(st: &SafeTensors, c: &Ink2Config) -> Result<Weights> {
    let mut t = HashMap::new();
    let h = c.hidden;
    let mut plain = |name: String, n: usize| -> Result<()> {
        let v = st.need(&format!("{PREFIX}{name}"))?.to_f32()?;
        ensure!(v.len() == n, "{name}: {} values, expected {n}", v.len());
        t.insert(name, v);
        Ok(())
    };
    plain("embed.weight".into(), c.vocab * h)?;
    plain("embed_norm.weight".into(), h)?;
    plain("unembed.weight".into(), c.vocab * h)?;
    plain("norm.weight".into(), h)?;
    plain("attn_res.output_query".into(), h)?;
    let nl = c.loop_end - c.loop_start;
    plain("middle_cycle_cross_pass_queries".into(), 2 * nl * h)?;
    plain("middle_cycle_cross_pass_kv_queries".into(), 2 * nl * h)?;
    for l in 0..c.n_layers {
        let (hq, hk, d) = c.heads(l);
        let p = format!("layers.{l}.");
        if l > 0 {
            // layer l's attention-input queries are stored one index down
            plain(format!("attn_res.attn_queries.{}", l - 1), if c.sliding[l] { h } else { 3 * h })?;
        }
        plain(format!("attn_res.mlp_queries.{l}"), h)?;
        plain(format!("{p}attn.wq_du.weight"), hq * d * h)?;
        plain(format!("{p}attn.wk_dv.weight"), hk * d * h)?;
        plain(format!("{p}attn.wv_dv.weight"), hk * d * h)?;
        plain(format!("{p}attn.wo_ud.weight"), h * hq * d)?;
        plain(format!("{p}attn.gate_attn.weight"), hq * d * h)?;
        plain(format!("{p}attn.attn_sink_proj.weight"), hq * h)?;
        plain(format!("{p}attn.wr_du.weight"), hq * c.d_rel * h)?;
        plain(format!("{p}attn.q_l2_clip.scale"), hq * d)?;
        plain(format!("{p}attn.rel_l2_clip.scale"), hq * c.d_rel)?;
        plain(format!("{p}attn.rel_logits_proj.proj"), c.d_rel * if c.sliding[l] { c.window } else { c.rel_extent })?;
        if c.sliding[l] {
            plain(format!("{p}attn.k_sconv.weight"), hk * d * c.conv_k)?;
            plain(format!("{p}attn.v_sconv.weight"), hk * d * c.conv_k)?;
            plain(format!("{p}attn_norm.weight"), h)?;
        } else {
            for kv in ["k", "v"] {
                plain(format!("{p}attn.{kv}_sconv.static_w"), c.conv_k * hk * d)?;
                plain(format!("{p}attn.{kv}_sconv.weight_proj"), hk * c.conv_k * h)?;
            }
            for n in ["q", "k", "v"] {
                plain(format!("{p}{n}_input_norm.weight"), h)?;
            }
        }
        plain(format!("{p}attn_sconv.weight"), h * c.conv_k)?;
        plain(format!("{p}mlp_sconv.weight"), h * c.conv_k)?;
        plain(format!("{p}mlp_norm.weight"), h)?;
        plain(format!("{p}mlp.gate.weight"), c.n_experts * h)?;
        plain(format!("{p}mlp.gate.bias"), c.n_experts)?;
        if c.in_loop(l) {
            for g in ["attn", "k", "v", "mlp"] {
                plain(format!("{p}controllers.{g}.down_weight"), c.gate_rank * 2 * h)?;
                plain(format!("{p}controllers.{g}.forget_weight"), h * c.gate_rank)?;
                plain(format!("{p}controllers.{g}.gain_weight"), h * c.gate_rank)?;
            }
        }
    }
    for l in 0..c.n_layers {
        let p = format!("layers.{l}.mlp.experts.");
        // w13: rows gate 0, up 0, gate 1, up 1, ...
        let w13 = load_experts(st, &format!("{PREFIX}{p}w13_weight"), c.n_experts, 2 * c.moe_inter, h)?;
        let w2 = load_experts(st, &format!("{PREFIX}{p}w2_weight"), c.n_experts, h, c.moe_inter)?;
        t.insert(format!("{p}w13_weight"), w13);
        t.insert(format!("{p}w2_weight"), w2);
    }
    Ok(Weights { t })
}

// ------------------------------------------------------------------ pieces (rows are [n][width])
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// y[r] = W x[r], W [n_out][n_in].
fn linear(x: &[f32], w: &[f32], n_in: usize) -> Vec<f32> {
    let n_out = w.len() / n_in;
    let mut y = Vec::with_capacity(x.len() / n_in * n_out);
    for xr in x.chunks_exact(n_in) {
        y.extend(w.chunks_exact(n_in).map(|wr| dot(wr, xr)));
    }
    y
}

/// RMS normalisation without a gain, each `width` values.
fn unorm(x: &mut [f32], width: usize, eps: f32) {
    for r in x.chunks_exact_mut(width) {
        let s = (r.iter().map(|v| v * v).sum::<f32>() / width as f32 + eps).sqrt().recip();
        r.iter_mut().for_each(|v| *v *= s);
    }
}

fn rms(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mut y = x.to_vec();
    unorm(&mut y, w.len(), eps);
    for r in y.chunks_exact_mut(w.len()) {
        r.iter_mut().zip(w).for_each(|(v, g)| *v *= g);
    }
    y
}

/// Scale each `width` values down to L2 norm `thr` when longer.
fn l2_clip(x: &mut [f32], width: usize, thr: f32) {
    for r in x.chunks_exact_mut(width) {
        let n = r.iter().map(|v| v * v).sum::<f32>().sqrt();
        let s = (thr / (n + 1e-6)).min(1.0);
        r.iter_mut().for_each(|v| *v *= s);
    }
}

fn softmax(x: &mut [f32]) {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    x.iter_mut().for_each(|v| *v /= s);
}

fn softcap(x: f32, cap: f32) -> f32 {
    (x / cap).tanh() * cap
}

/// Block Attention Residuals: each of the queries ([nq][h]) attends, at every row alike, over the
/// finished blocks and the running sum of the current one (all RMS-normalised, no gain). Returns
/// one [rows][h] per query.
fn attn_res(queries: &[f32], blocks: &[Vec<f32>], partial: &[f32], h: usize) -> Vec<Vec<f32>> {
    let (nq, rows, n) = (queries.len() / h, partial.len() / h, blocks.len() + 1);
    let mut out = vec![vec![0.0f32; rows * h]; nq];
    let mut st = vec![0.0f32; n * h];
    let mut p = vec![0.0f32; n];
    for r in 0..rows {
        for (i, b) in blocks.iter().map(|b| b.as_slice()).chain(std::iter::once(partial)).enumerate() {
            st[i * h..(i + 1) * h].copy_from_slice(&b[r * h..(r + 1) * h]);
        }
        unorm(&mut st, h, 1e-6);
        for (q, o) in queries.chunks_exact(h).zip(out.iter_mut()) {
            for i in 0..n {
                p[i] = dot(&st[i * h..(i + 1) * h], q) / (h as f32).sqrt();
            }
            softmax(&mut p);
            let o = &mut o[r * h..(r + 1) * h];
            for i in 0..n {
                o.iter_mut().zip(&st[i * h..(i + 1) * h]).for_each(|(a, b)| *a += p[i] * b);
            }
        }
    }
    out
}

/// The inputs a convolution sees for rows `x` ([rows][c]): the state (the k - 1 rows before, zeros
/// at a sequence's start) followed by x. Leaves the state holding the last k - 1 of them.
fn conv_context(x: &[f32], state: &mut Vec<f32>, c: usize, k: usize) -> Vec<f32> {
    if state.is_empty() {
        state.resize((k - 1) * c, 0.0);
    }
    let mut ctx = std::mem::take(state);
    ctx.extend_from_slice(x);
    *state = ctx[ctx.len() - (k - 1) * c..].to_vec();
    ctx
}

/// Short causal convolution per channel: out[t] = sum_j w[c][j] x[t - (k-1) + j] (the last tap is
/// the current row), then (out + x) / sqrt(k + 1). w [c][k].
fn sconv_static(x: &[f32], w: &[f32], state: &mut Vec<f32>, c: usize, k: usize) -> Vec<f32> {
    let ctx = conv_context(x, state, c, k);
    let norm = ((k + 1) as f32).sqrt();
    let mut y = vec![0.0f32; x.len()];
    for (t, yr) in y.chunks_exact_mut(c).enumerate() {
        for (ch, o) in yr.iter_mut().enumerate() {
            let mut a = 0.0;
            for j in 0..k {
                a += w[ch * k + j] * ctx[(t + j) * c + ch];
            }
            *o = (a + x[t * c + ch]) / norm;
        }
    }
    y
}

/// The full layers' key / value convolution: tap `lag` (0 = the current row) weighs x[t - lag] by
/// static_w[lag][c] + a per-row, per-head weight projected from the layer input (`dyn_w`:
/// [rows][heads * k]); then (out + x) / sqrt(2k + 1).
fn sconv_dynamic(
    x: &[f32],
    dyn_w: &[f32],
    static_w: &[f32],
    state: &mut Vec<f32>,
    c: usize,
    k: usize,
    heads: usize,
) -> Vec<f32> {
    let ctx = conv_context(x, state, c, k);
    let norm = ((2 * k + 1) as f32).sqrt();
    let per = c / heads;
    let mut y = vec![0.0f32; x.len()];
    for (t, yr) in y.chunks_exact_mut(c).enumerate() {
        for (ch, o) in yr.iter_mut().enumerate() {
            let mut a = 0.0;
            for lag in 0..k {
                let wt = static_w[lag * c + ch] + dyn_w[t * heads * k + (ch / per) * k + lag];
                a += wt * ctx[(t + k - 1 - lag) * c + ch];
            }
            *o = (a + x[t * c + ch]) / norm;
        }
    }
    y
}

#[derive(Default)]
struct Visit {
    /// [positions][kv heads * head width]
    k: Vec<f32>,
    v: Vec<f32>,
    /// keys, values, attention output, MLP output: each the last conv_k - 1 inputs
    conv: [Vec<f32>; 4],
}

struct SeqState {
    used: bool,
    len: usize,
    visits: Vec<Visit>,
}

pub struct Ink2Cpu {
    cfg: Ink2Config,
    w: Weights,
    seqs: Vec<SeqState>,
    cache_tokens: usize,
    max_context: usize,
    page: usize,
}

impl Ink2Cpu {
    pub fn load(dir: &Path, config: &serde_json::Value, opts: &LoadOptions) -> Result<Self> {
        let cfg = Ink2Config::from_json(config)?;
        ensure!(opts.max_seqs > 0 && opts.page > 0, "max_seqs and page must be positive");
        let st = SafeTensors::open_dir(dir)?;
        let w = bind(&st, &cfg).context("binding the Inkling 2 checkpoint")?;
        let max_context =
            if opts.max_context > 0 { opts.max_context.min(opts.cache_tokens) } else { opts.cache_tokens };
        Ok(Ink2Cpu {
            seqs: (0..opts.max_seqs).map(|_| SeqState { used: false, len: 0, visits: Vec::new() }).collect(),
            cache_tokens: opts.cache_tokens,
            max_context,
            page: opts.page,
            cfg,
            w,
        })
    }

    pub fn config(&self) -> &Ink2Config {
        &self.cfg
    }

    fn pages(&self, len: usize) -> usize {
        len.div_ceil(self.page)
    }

    fn used_pages(&self) -> usize {
        self.seqs.iter().filter(|s| s.used).map(|s| self.pages(s.len)).sum()
    }

    /// (1 - forget) * h + gain * message, both gates tanh of a low-rank map of [norm(h) | norm(m)].
    fn gate(&self, l: usize, name: &str, hs: &[f32], msg: &[f32]) -> Vec<f32> {
        let (c, h) = (&self.cfg, self.cfg.hidden);
        let p = format!("layers.{l}.controllers.{name}.");
        let (mut a, mut b) = (hs.to_vec(), msg.to_vec());
        unorm(&mut a, h, c.rms_eps);
        unorm(&mut b, h, c.rms_eps);
        let mut cat = Vec::with_capacity(2 * hs.len());
        for (x, y) in a.chunks_exact(h).zip(b.chunks_exact(h)) {
            cat.extend_from_slice(x);
            cat.extend_from_slice(y);
        }
        let g = linear(&cat, self.w.get(&format!("{p}down_weight")), 2 * h);
        let forget = linear(&g, self.w.get(&format!("{p}forget_weight")), c.gate_rank);
        let gain = linear(&g, self.w.get(&format!("{p}gain_weight")), c.gate_rank);
        (0..hs.len()).map(|i| (1.0 - forget[i].tanh()) * hs[i] + gain[i].tanh() * msg[i]).collect()
    }

    /// Layer l's attention for rows at positions pos0.., reading and extending visit `vis`'s cache.
    #[allow(clippy::too_many_arguments)]
    fn attention(&self, l: usize, vis: &mut Visit, xq: &[f32], xk: &[f32], xv: &[f32], pos0: usize) -> Vec<f32> {
        let (c, w, h) = (&self.cfg, &self.w, self.cfg.hidden);
        let p = format!("layers.{l}.attn.");
        let (hq, hk, d) = c.heads(l);
        let (rows, full, g, kw) = (xq.len() / h, !c.sliding[l], hq / hk, hk * d);
        // queries and relative states: scaled, clipped, and in the full layers scaled up with position
        let mut q = linear(xq, w.get(&format!("{p}wq_du.weight")), h);
        let mut rel = linear(xq, w.get(&format!("{p}wr_du.weight")), h);
        let (qs, rs) = (w.get(&format!("{p}q_l2_clip.scale")), w.get(&format!("{p}rel_l2_clip.scale")));
        q.chunks_exact_mut(hq * d).for_each(|r| r.iter_mut().zip(qs).for_each(|(a, s)| *a *= s));
        rel.chunks_exact_mut(hq * c.d_rel).for_each(|r| r.iter_mut().zip(rs).for_each(|(a, s)| *a *= s));
        l2_clip(&mut q, d, c.qk_clip);
        l2_clip(&mut rel, c.d_rel, c.rel_clip);
        if full {
            for r in 0..rows {
                let tau = 1.0 + c.log_alpha * (((pos0 + r + 1) as f32) / c.log_n_floor).max(1.0).ln();
                q[r * hq * d..(r + 1) * hq * d].iter_mut().for_each(|a| *a *= tau);
                rel[r * hq * c.d_rel..(r + 1) * hq * c.d_rel].iter_mut().for_each(|a| *a *= tau);
            }
        }
        let proj = w.get(&format!("{p}rel_logits_proj.proj")); // [d_rel][extent]
        let extent = proj.len() / c.d_rel;
        // keys and values: projection, short convolution, keys clipped, values normalised
        let mut k = linear(xk, w.get(&format!("{p}wk_dv.weight")), h);
        let mut v = linear(xv, w.get(&format!("{p}wv_dv.weight")), h);
        if full {
            let dk = linear(xk, w.get(&format!("{p}k_sconv.weight_proj")), h);
            let dv = linear(xv, w.get(&format!("{p}v_sconv.weight_proj")), h);
            k = sconv_dynamic(&k, &dk, w.get(&format!("{p}k_sconv.static_w")), &mut vis.conv[0], kw, c.conv_k, hk);
            v = sconv_dynamic(&v, &dv, w.get(&format!("{p}v_sconv.static_w")), &mut vis.conv[1], kw, c.conv_k, hk);
        } else {
            k = sconv_static(&k, w.get(&format!("{p}k_sconv.weight")), &mut vis.conv[0], kw, c.conv_k);
            v = sconv_static(&v, w.get(&format!("{p}v_sconv.weight")), &mut vis.conv[1], kw, c.conv_k);
        }
        l2_clip(&mut k, d, c.qk_clip);
        unorm(&mut v, d, c.rms_eps);
        vis.k.extend_from_slice(&k);
        vis.v.extend_from_slice(&v);
        let sink = linear(xq, w.get(&format!("{p}attn_sink_proj.weight")), h); // [rows][hq]
        let gate = linear(xq, w.get(&format!("{p}gate_attn.weight")), h);
        let mut out = vec![0.0f32; rows * hq * d];
        let mut sc = Vec::new();
        let mut bias = vec![0.0f32; extent];
        for r in 0..rows {
            let pos = pos0 + r;
            let lo = if full { 0 } else { (pos + 1).saturating_sub(c.window) };
            for hd in 0..hq {
                let qv = &q[(r * hq + hd) * d..(r * hq + hd + 1) * d];
                let rv = &rel[(r * hq + hd) * c.d_rel..(r * hq + hd + 1) * c.d_rel];
                for (dist, b) in bias.iter_mut().enumerate() {
                    *b = (0..c.d_rel).map(|i| rv[i] * proj[i * extent + dist]).sum();
                }
                let kh = hd / g;
                sc.clear();
                for j in lo..=pos {
                    let dist = pos - j;
                    let s = dot(qv, &vis.k[(j * hk + kh) * d..(j * hk + kh + 1) * d]) / (d as f32).sqrt();
                    sc.push(s + if dist < extent { bias[dist] } else { 0.0 });
                }
                // the sink takes probability mass and adds nothing
                sc.push(sink[r * hq + hd]);
                softmax(&mut sc);
                let o = &mut out[(r * hq + hd) * d..(r * hq + hd + 1) * d];
                for (j, pj) in (lo..=pos).zip(&sc) {
                    o.iter_mut().zip(&vis.v[(j * hk + kh) * d..(j * hk + kh + 1) * d]).for_each(|(a, b)| *a += pj * b);
                }
            }
        }
        out.iter_mut().zip(&gate).for_each(|(a, g)| *a *= sigmoid(*g));
        linear(&out, w.get(&format!("{p}wo_ud.weight")), hq * d)
    }

    fn moe(&self, l: usize, x: &[f32]) -> Vec<f32> {
        let (c, w, h) = (&self.cfg, &self.w, self.cfg.hidden);
        let p = format!("layers.{l}.mlp.");
        let (gw, gb) = (w.get(&format!("{p}gate.weight")), w.get(&format!("{p}gate.bias")));
        let (w13, w2) = (w.get(&format!("{p}experts.w13_weight")), w.get(&format!("{p}experts.w2_weight")));
        let i2 = 2 * c.moe_inter;
        let mut out = vec![0.0f32; x.len()];
        for (xr, or) in x.chunks_exact(h).zip(out.chunks_exact_mut(h)) {
            let scores: Vec<f32> = gw.chunks_exact(h).map(|wr| sigmoid(dot(wr, xr))).collect();
            // the top-k by score + bias (ties to the lower index), weights from the scores
            let mut idx: Vec<usize> = (0..c.n_experts).collect();
            idx.sort_by(|&a, &b| (scores[b] + gb[b]).total_cmp(&(scores[a] + gb[a])).then(a.cmp(&b)));
            idx.truncate(c.topk);
            let sum: f32 = idx.iter().map(|&e| scores[e]).sum::<f32>() + 1e-20;
            for &e in &idx {
                let wt = scores[e] / sum * c.route_scale;
                let gu = linear(xr, &w13[e * i2 * h..(e + 1) * i2 * h], h);
                let a: Vec<f32> = gu
                    .chunks_exact(2)
                    .map(|gu| softcap(gu[0], c.gate_softcap) * sigmoid(gu[0]) * softcap(gu[1], c.up_softcap))
                    .collect();
                let y = linear(&a, &w2[e * h * c.moe_inter..(e + 1) * h * c.moe_inter], c.moe_inter);
                or.iter_mut().zip(&y).for_each(|(o, v)| *o += wt * v);
            }
        }
        out
    }

    /// Layer l's pass p on rows hs; blocks: the finished blocks' states; msgs: the four messages
    /// (query, key, value, mlp) in passes after the first.
    #[allow(clippy::too_many_arguments)]
    fn layer(
        &self,
        l: usize,
        p: usize,
        hs: &[f32],
        blocks: &[Vec<f32>],
        msgs: Option<&[Vec<f32>]>,
        pos0: usize,
        st: &mut SeqState,
    ) -> Vec<f32> {
        let (c, w, h) = (&self.cfg, &self.w, self.cfg.hidden);
        let pre = format!("layers.{l}.");
        let vis = &mut st.visits[c.visit(l, p)];
        let mut ins = if l > 0 {
            attn_res(w.get(&format!("attn_res.attn_queries.{}", l - 1)), blocks, hs, h)
        } else {
            vec![hs.to_vec()]
        };
        let names = ["attn", "k", "v"];
        let (xq, xk, xv);
        if c.sliding[l] {
            let nw = w.get(&format!("{pre}attn_norm.weight"));
            match msgs {
                None => {
                    xq = rms(&ins[0], nw, c.rms_eps);
                    xk = xq.clone();
                    xv = xq.clone();
                }
                Some(m) => {
                    let x: Vec<Vec<f32>> =
                        (0..3).map(|i| rms(&self.gate(l, names[i], &ins[0], &m[i]), nw, c.rms_eps)).collect();
                    let mut it = x.into_iter();
                    (xq, xk, xv) = (it.next().unwrap(), it.next().unwrap(), it.next().unwrap());
                }
            }
        } else {
            if ins.len() == 1 {
                ins = vec![ins[0].clone(), ins[0].clone(), ins[0].clone()];
            }
            if let Some(m) = msgs {
                ins = (0..3).map(|i| self.gate(l, names[i], &ins[i], &m[i])).collect();
            }
            xq = rms(&ins[0], w.get(&format!("{pre}q_input_norm.weight")), c.rms_eps);
            xk = rms(&ins[1], w.get(&format!("{pre}k_input_norm.weight")), c.rms_eps);
            xv = rms(&ins[2], w.get(&format!("{pre}v_input_norm.weight")), c.rms_eps);
        }
        let a = self.attention(l, vis, &xq, &xk, &xv, pos0);
        let a = sconv_static(&a, w.get(&format!("{pre}attn_sconv.weight")), &mut vis.conv[2], h, c.conv_k);
        // a block starts: the stream so far becomes a block state and the running sum restarts
        let boundary = l % c.block == 0;
        let mut with: Vec<Vec<f32>>;
        let (blocks2, resid): (&[Vec<f32>], Vec<f32>) = if boundary {
            with = blocks.to_vec();
            with.push(hs.to_vec());
            (&with, a)
        } else {
            (blocks, hs.iter().zip(&a).map(|(x, y)| x + y).collect())
        };
        let mut m = attn_res(w.get(&format!("attn_res.mlp_queries.{l}")), blocks2, &resid, h).remove(0);
        if let Some(ms) = msgs {
            m = self.gate(l, "mlp", &m, &ms[3]);
        }
        let m = self.moe(l, &rms(&m, w.get(&format!("{pre}mlp_norm.weight")), c.rms_eps));
        let m = sconv_static(&m, w.get(&format!("{pre}mlp_sconv.weight")), &mut vis.conv[3], h, c.conv_k);
        resid.iter().zip(&m).map(|(x, y)| x + y).collect()
    }

    /// Layers `layers` in pass p; prev: the previous pass's block states and output, the source
    /// of this pass's messages.
    #[allow(clippy::too_many_arguments)]
    fn run_layers(
        &self,
        layers: std::ops::Range<usize>,
        p: usize,
        mut hs: Vec<f32>,
        mut blocks: Vec<Vec<f32>>,
        prev: Option<(&[Vec<f32>], &[f32])>,
        pos0: usize,
        st: &mut SeqState,
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        let (c, w, h) = (&self.cfg, &self.w, self.cfg.hidden);
        for l in layers {
            let msgs = prev.map(|(pb, ph)| {
                let i = l - c.loop_start;
                let qm = &w.get("middle_cycle_cross_pass_queries")[2 * i * h..(2 * i + 2) * h]; // query, mlp
                let kv = &w.get("middle_cycle_cross_pass_kv_queries")[2 * i * h..(2 * i + 2) * h]; // key, value
                let q4 = [&qm[..h], &kv[..h], &kv[h..], &qm[h..]].concat();
                attn_res(&q4, pb, ph, h)
            });
            let nh = self.layer(l, p, &hs, &blocks, msgs.as_deref(), pos0, st);
            if l % c.block == 0 {
                blocks.push(hs);
            }
            hs = nh;
        }
        (hs, blocks)
    }

    /// Logits of the wanted rows of `tokens` appended to sequence `seq`.
    fn run(&self, tokens: &[u32], st: &mut SeqState, all: bool) -> Vec<f32> {
        let (c, w, h) = (&self.cfg, &self.w, self.cfg.hidden);
        let pos0 = st.len;
        let emb = w.get("embed.weight");
        let mut x = Vec::with_capacity(tokens.len() * h);
        for &t in tokens {
            x.extend_from_slice(&emb[t as usize * h..(t as usize + 1) * h]);
        }
        let hs = rms(&x, w.get("embed_norm.weight"), c.rms_eps);
        let (hs, blocks) = self.run_layers(0..c.loop_start, 0, hs, Vec::new(), None, pos0, st);
        let (pre_h, pre_blocks) = (hs.clone(), blocks.clone());
        let (mut hs, mut blocks) = self.run_layers(c.loop_start..c.loop_end, 0, hs, blocks, None, pos0, st);
        for p in 1..c.passes {
            // the previous pass's block states past the one it started with, and its output
            let prev_blocks = blocks[(pre_blocks.len() + 1).min(blocks.len())..].to_vec();
            let prev_h = hs;
            (hs, blocks) = self.run_layers(
                c.loop_start..c.loop_end,
                p,
                pre_h.clone(),
                pre_blocks.clone(),
                Some((&prev_blocks, &prev_h)),
                pos0,
                st,
            );
        }
        let (hs, blocks) = self.run_layers(c.loop_end..c.n_layers, 0, hs, blocks, None, pos0, st);
        let fin = attn_res(w.get("attn_res.output_query"), &blocks, &hs, h).remove(0);
        let fin = rms(&fin, w.get("norm.weight"), c.rms_eps);
        let rows = if all { &fin[..] } else { &fin[fin.len() - h..] };
        let un = &w.get("unembed.weight")[..c.out_vocab * h];
        linear(rows, un, h).into_iter().map(|v| softcap(v, c.logit_softcap)).collect()
    }
}

impl Model for Ink2Cpu {
    fn arch(&self) -> &'static str {
        "inkling2_loop"
    }

    fn vocab(&self) -> usize {
        self.cfg.out_vocab
    }

    fn max_context(&self) -> usize {
        self.max_context
    }

    fn capacity(&self) -> Capacity {
        Capacity {
            max_seqs: self.seqs.len(),
            free_seqs: self.seqs.iter().filter(|s| !s.used).count(),
            free_cache_tokens: (self.pages(self.cache_tokens).saturating_sub(self.used_pages())) * self.page,
            prefix_cached_tokens: 0,
            host_cached_tokens: 0,
        }
    }

    fn alloc_seq(&mut self) -> Option<SeqId> {
        let id = self.seqs.iter().position(|s| !s.used)?;
        let visits = (0..self.cfg.visits()).map(|_| Visit::default()).collect();
        self.seqs[id] = SeqState { used: true, len: 0, visits };
        Some(id)
    }

    fn free_seq(&mut self, seq: SeqId) {
        if let Some(s) = self.seqs.get_mut(seq) {
            *s = SeqState { used: false, len: 0, visits: Vec::new() };
        }
    }

    fn page_tokens(&self) -> usize {
        self.page
    }

    fn can_grow(&self, seq: SeqId, n: usize) -> bool {
        let s = &self.seqs[seq];
        s.len + n <= self.max_context
            && self.used_pages() - self.pages(s.len) + self.pages(s.len + n) <= self.pages(self.cache_tokens)
    }

    fn forward(&mut self, batch: &[SeqChunk<'_>]) -> Result<Vec<Vec<f32>>> {
        let mut seen = vec![false; self.seqs.len()];
        let mut grow = 0;
        for ch in batch {
            ensure!(ch.seq < self.seqs.len() && self.seqs[ch.seq].used, "sequence {} is not allocated", ch.seq);
            ensure!(!std::mem::replace(&mut seen[ch.seq], true), "sequence {} appears twice in one step", ch.seq);
            ensure!(!ch.tokens.is_empty(), "empty chunk for sequence {}", ch.seq);
            let len = self.seqs[ch.seq].len;
            ensure!(ch.pos == len, "sequence {} is at {len}, chunk starts at {}", ch.seq, ch.pos);
            ensure!(ch.draft == 0, "this model takes no draft tokens (its convolution states cannot roll back)");
            ensure!(ch.tokens.iter().all(|&t| (t as usize) < self.cfg.vocab), "token id out of vocabulary");
            ensure!(len + ch.tokens.len() <= self.max_context, "sequence {} would pass the context limit", ch.seq);
            grow += self.pages(len + ch.tokens.len()) - self.pages(len);
        }
        ensure!(
            self.used_pages() + grow <= self.pages(self.cache_tokens),
            "KV cache exhausted: need {grow} more pages of {}",
            self.pages(self.cache_tokens)
        );
        let mut out = Vec::with_capacity(batch.len());
        for ch in batch {
            let mut st = std::mem::replace(&mut self.seqs[ch.seq], SeqState { used: true, len: 0, visits: Vec::new() });
            let logits = self.run(ch.tokens, &mut st, ch.all_logits);
            st.len += ch.tokens.len();
            self.seqs[ch.seq] = st;
            out.push(logits);
        }
        Ok(out)
    }

    fn describe(&self) -> String {
        let c = &self.cfg;
        format!(
            "inkling2_loop reference on CPU (f32): {} layers, layers {}..{} x{} ({} visits a token), {} experts top-{}, {} seq slots, {} cached positions",
            c.n_layers,
            c.loop_start,
            c.loop_end,
            c.passes,
            c.visits(),
            c.n_experts,
            c.topk,
            self.seqs.len(),
            self.cache_tokens
        )
    }
}
