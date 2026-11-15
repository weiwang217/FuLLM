// SPDX-License-Identifier: Apache-2.0
//! The CPU backend for Kimi K3: a batched, incremental forward over many sequences.
//!
//! It is three things at once:
//!   - the executable reference every other backend is checked against (it reproduces the
//!     torch oracle of tools/make_tiny_oracle.py within fp32 rounding);
//!   - the production path for the routed experts on hosts whose DRAM holds them, where
//!     computing next to the data beats copying 25.8 GB/token over PCIe;
//!   - the overflow path in exact-bf16 mode, for trunk layers that do not fit in HBM.
//!
//! STATE PER SEQUENCE
//!   KDA layers: a fixed `[H][D][D]` delta-rule state plus three ShortConv histories,
//!   one slot per sequence, updated in place.
//!   MLA layers: the normalised latent (kv_lora) and the unrotated rope slot (qk_rope) per
//!   position, in fixed-size pages. Attention is computed in the ABSORBED form: queries
//!   are projected into latent space through W_UK, values read back out through W_UV, so
//!   the cache holds 576 floats per position per layer instead of 96 x 256.
//!
//! BATCH INVARIANCE
//!   Every per-token quantity is produced by the same sequence of operations whatever
//!   else is in the step: matmuls are row-wise dots (fullm-kernels), the KDA recurrence
//!   and MLA attention run per token in position order, and routed-expert outputs are
//!   reduced per token in selection order. A sequence therefore yields bit-identical
//!   logits alone, batched, or with its prefill split into chunks. The tests assert it.

use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, ensure, Result};
use rayon::prelude::*;

use super::experts::{ExpertStore, MmapExperts};
use super::weights::{bind, Attn, DenseW, Ffn, KdaGate, KdaW, LayerW, MlaQuery, MlaW, MoeW, TrunkW};
use super::K3Config;
use crate::models::{Capacity, LoadOptions, Model, SeqChunk, SeqId};
use crate::ops;
use crate::st::SafeTensors;
use crate::tensor::{linear, linear_new, linear_ref, Precision, SendPtr};

struct SeqState {
    used: bool,
    len: usize,
    pages: Vec<u32>,
}

pub struct K3Cpu {
    cfg: K3Config,
    w: TrunkW,
    experts: Arc<dyn ExpertStore>,
    trunk: Precision,
    kda_of: Vec<Option<usize>>,
    mla_of: Vec<Option<usize>>,
    /// `[seq][kda_layer][state_floats]`
    kda_state: Vec<f32>,
    /// `[mla_layer][page][page_len][kv_lora + qk_rope]`
    kv: Vec<f32>,
    page: usize,
    n_pages: usize,
    free_pages: Vec<u32>,
    seqs: Vec<SeqState>,
}

/// A chunk's rows in the flattened step.
#[derive(Clone, Copy)]
struct Span {
    seq: SeqId,
    pos: usize,
    off: usize,
    len: usize,
}

impl K3Cpu {
    pub fn load(dir: &Path, config: &serde_json::Value, opts: &LoadOptions) -> Result<Self> {
        let mut cfg = K3Config::from_json(config)?;
        if let Some(n) = opts.layers {
            cfg.truncate(n);
        }
        cfg.router_weights_biased = opts.router_weights_biased;
        let st = SafeTensors::open_dir(dir)?;
        let w = bind(&st, &cfg, opts.trunk)?;
        let experts: Arc<dyn ExpertStore> = Arc::new(MmapExperts::new(st.clone(), &cfg)?);
        Self::new(cfg, w, experts, opts)
    }

    pub fn new(cfg: K3Config, w: TrunkW, experts: Arc<dyn ExpertStore>, opts: &LoadOptions) -> Result<Self> {
        ensure!(opts.max_seqs > 0 && opts.page > 0, "max_seqs and page must be positive");
        let mut kda_of = vec![None; cfg.n_layers];
        let mut mla_of = vec![None; cfg.n_layers];
        let (mut nk, mut nm) = (0, 0);
        for l in 0..cfg.n_layers {
            if cfg.is_mla(l) {
                mla_of[l] = Some(nm);
                nm += 1;
            } else {
                kda_of[l] = Some(nk);
                nk += 1;
            }
        }
        let n_pages = opts.cache_tokens.div_ceil(opts.page).max(1);
        let kda_state = vec![0.0; opts.max_seqs * nk * cfg.kda_state_floats()];
        let kv = vec![0.0; nm * n_pages * opts.page * cfg.mla_cache_width()];
        Ok(K3Cpu {
            trunk: opts.trunk,
            kda_of,
            mla_of,
            kda_state,
            kv,
            page: opts.page,
            n_pages,
            free_pages: (0..n_pages as u32).rev().collect(),
            seqs: (0..opts.max_seqs).map(|_| SeqState { used: false, len: 0, pages: Vec::new() }).collect(),
            cfg,
            w,
            experts,
        })
    }

    pub fn config(&self) -> &K3Config {
        &self.cfg
    }

    fn pages_for(&self, len: usize) -> usize {
        len.div_ceil(self.page)
    }
}

impl Model for K3Cpu {
    fn arch(&self) -> &'static str {
        match self.cfg.family {
            super::config::Family::K3 => "kimi_k3",
            super::config::Family::Linear => "kimi_linear",
        }
    }

    fn vocab(&self) -> usize {
        self.cfg.vocab
    }

    fn max_context(&self) -> usize {
        self.n_pages * self.page
    }

    fn capacity(&self) -> Capacity {
        Capacity {
            max_seqs: self.seqs.len(),
            free_seqs: self.seqs.iter().filter(|s| !s.used).count(),
            free_cache_tokens: self.free_pages.len() * self.page,
            prefix_cached_tokens: 0,
            host_cached_tokens: 0,
        }
    }

    fn alloc_seq(&mut self) -> Option<SeqId> {
        let id = self.seqs.iter().position(|s| !s.used)?;
        self.seqs[id] = SeqState { used: true, len: 0, pages: Vec::new() };
        let sf = self.cfg.kda_state_floats() * self.cfg.n_kda();
        self.kda_state[id * sf..(id + 1) * sf].fill(0.0);
        Some(id)
    }

    fn free_seq(&mut self, seq: SeqId) {
        if let Some(s) = self.seqs.get_mut(seq) {
            if s.used {
                self.free_pages.append(&mut s.pages);
                s.used = false;
                s.len = 0;
            }
        }
    }

    fn page_tokens(&self) -> usize {
        self.page
    }

    fn can_grow(&self, seq: SeqId, n: usize) -> bool {
        let s = &self.seqs[seq];
        self.pages_for(s.len + n).saturating_sub(s.pages.len()) <= self.free_pages.len()
    }

    fn forward(&mut self, batch: &[SeqChunk<'_>]) -> Result<Vec<Vec<f32>>> {
        // Validate the whole batch and reserve pages before touching any state.
        let mut spans = Vec::with_capacity(batch.len());
        let mut off = 0;
        let mut seen = vec![false; self.seqs.len()];
        for ch in batch {
            ensure!(ch.seq < self.seqs.len() && self.seqs[ch.seq].used, "sequence {} is not allocated", ch.seq);
            ensure!(!std::mem::replace(&mut seen[ch.seq], true), "sequence {} appears twice in one step", ch.seq);
            ensure!(!ch.tokens.is_empty(), "empty chunk for sequence {}", ch.seq);
            ensure!(
                ch.pos == self.seqs[ch.seq].len,
                "sequence {} is at {}, chunk starts at {}",
                ch.seq,
                self.seqs[ch.seq].len,
                ch.pos
            );
            ensure!(ch.tokens.iter().all(|&t| (t as usize) < self.cfg.vocab), "token id out of vocabulary");
            spans.push(Span { seq: ch.seq, pos: ch.pos, off, len: ch.tokens.len() });
            off += ch.tokens.len();
        }
        let need: usize =
            batch.iter().map(|ch| self.pages_for(ch.pos + ch.tokens.len()) - self.seqs[ch.seq].pages.len()).sum();
        ensure!(need <= self.free_pages.len(), "KV cache exhausted: need {need} pages, {} free", self.free_pages.len());
        for ch in batch {
            let want = self.pages_for(ch.pos + ch.tokens.len());
            while self.seqs[ch.seq].pages.len() < want {
                let p = self.free_pages.pop().expect("reserved above");
                self.seqs[ch.seq].pages.push(p);
            }
        }

        let tokens: Vec<u32> = batch.iter().flat_map(|ch| ch.tokens.iter().copied()).collect();
        let rows: Vec<usize> = batch
            .iter()
            .zip(&spans)
            .flat_map(|(ch, sp)| {
                if ch.all_logits {
                    (sp.off..sp.off + sp.len).collect::<Vec<_>>()
                } else {
                    vec![sp.off + sp.len - 1]
                }
            })
            .collect();

        let logits = self.run(&tokens, &spans, &rows)?;

        for sp in &spans {
            self.seqs[sp.seq].len += sp.len;
        }
        let v = self.cfg.vocab;
        let mut out = Vec::with_capacity(batch.len());
        let mut r = 0;
        for ch in batch {
            let n = if ch.all_logits { ch.tokens.len() } else { 1 };
            out.push(logits[r * v..(r + n) * v].to_vec());
            r += n;
        }
        Ok(out)
    }

    fn describe(&self) -> String {
        format!(
            "kimi_k3 on CPU ({}): trunk {:?} {:.2} GB, experts {}, {} seq slots, {} cached positions",
            fullm_kernels::cpu_isa(),
            self.trunk,
            self.w.layer_bytes() as f64 / 1e9,
            self.experts.describe(),
            self.seqs.len(),
            self.n_pages * self.page
        )
    }
}

impl K3Cpu {
    fn run(&mut self, tokens: &[u32], spans: &[Span], rows: &[usize]) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let e = c.hidden;
        let n = tokens.len();
        let maxs = c.max_sources();

        // embedding: gather bf16 rows
        let mut h = vec![0.0f32; n * e];
        for (t, &id) in tokens.iter().enumerate() {
            self.w.embed.row_f32(id as usize, &mut h[t * e..(t + 1) * e]);
        }

        // AttnRes block snapshots, [token][source][hidden]
        let mut blocks = vec![0.0f32; n * maxs * e];
        let mut n_blocks = 0usize;
        let mut pref = vec![0.0f32; n * e];
        let mut hin = vec![0.0f32; n * e];
        let mut tmp = vec![0.0f32; n * e];

        for l in 0..c.n_layers {
            let lw = &self.w.layers[l];
            let mut ctx = AttnCtx {
                kda_of: &self.kda_of,
                mla_of: &self.mla_of,
                kda_state: &mut self.kda_state,
                kv: &mut self.kv,
                page: self.page,
                n_pages: self.n_pages,
                seqs: &self.seqs,
            };
            let Some(folds) = &lw.folds else {
                // Plain pre-norm residuals (Kimi Linear).
                norm_rows(&mut hin, &h, &lw.in_norm, e, c.rms_eps);
                attention(c, lw, l, &mut ctx, &hin, spans, &mut tmp);
                h.iter_mut().zip(&tmp).for_each(|(a, b)| *a += b);
                norm_rows(&mut hin, &h, &lw.post_norm, e, c.rms_eps);
                ffn(c, lw, self.experts.as_ref(), l, &hin, n, &mut tmp)?;
                h.iter_mut().zip(&tmp).for_each(|(a, b)| *a += b);
                continue;
            };
            pref.copy_from_slice(&h);

            // AttnRes before attention, only once snapshots exist. It REPLACES h.
            if n_blocks > 0 {
                attn_res_rows(&mut h, &blocks, n_blocks, maxs, &pref, &folds.attn, e, c.rms_eps);
            }
            // Block boundary: snapshot the running residual, then CLEAR it.
            let mut have_prefix = true;
            if l % c.attn_res_block == 0 {
                for t in 0..n {
                    blocks[(t * maxs + n_blocks) * e..(t * maxs + n_blocks + 1) * e]
                        .copy_from_slice(&pref[t * e..(t + 1) * e]);
                }
                n_blocks += 1;
                have_prefix = false;
            }

            norm_rows(&mut hin, &h, &lw.in_norm, e, c.rms_eps);
            attention(c, lw, l, &mut ctx, &hin, spans, &mut tmp);
            if have_prefix {
                pref.iter_mut().zip(&tmp).for_each(|(p, x)| *p += x);
            } else {
                pref.copy_from_slice(&tmp);
            }

            // AttnRes before the FFN: unconditional (layer 0 already pushed a snapshot).
            attn_res_rows(&mut h, &blocks, n_blocks, maxs, &pref, &folds.mlp, e, c.rms_eps);
            norm_rows(&mut hin, &h, &lw.post_norm, e, c.rms_eps);
            ffn(c, lw, self.experts.as_ref(), l, &hin, n, &mut tmp)?;
            pref.iter_mut().zip(&tmp).for_each(|(p, x)| *p += x);
            h.copy_from_slice(&pref);
        }

        // Output aggregation (AttnRes models) and head, for the requested rows only.
        let m = rows.len();
        let mut fin = vec![0.0f32; m * e];
        for (i, &t) in rows.iter().enumerate() {
            let mut agg = h[t * e..(t + 1) * e].to_vec();
            if let Some(fold_out) = &self.w.fold_out {
                let mut srcs: Vec<&[f32]> =
                    (0..n_blocks).map(|b| &blocks[(t * maxs + b) * e..(t * maxs + b + 1) * e]).collect();
                srcs.push(&h[t * e..(t + 1) * e]);
                ops::attn_res(&mut agg, &srcs, fold_out, c.rms_eps);
            }
            ops::rmsnorm(&mut fin[i * e..(i + 1) * e], &agg, &self.w.final_norm, c.rms_eps);
        }
        Ok(linear_new(&fin, m, &self.w.lm_head))
    }
}

/// The per-sequence state an attention layer reads and writes.
struct AttnCtx<'a> {
    kda_of: &'a [Option<usize>],
    mla_of: &'a [Option<usize>],
    kda_state: &'a mut [f32],
    kv: &'a mut [f32],
    page: usize,
    n_pages: usize,
    seqs: &'a [SeqState],
}

fn attention(c: &K3Config, lw: &LayerW, l: usize, ctx: &mut AttnCtx<'_>, x: &[f32], spans: &[Span], out: &mut [f32]) {
    match &lw.attn {
        Attn::Kda(w) => {
            let ki = ctx.kda_of[l].expect("kda layer");
            kda_layer(c, w, ki, x, spans, ctx.kda_state, out);
        }
        Attn::Mla(w) => {
            let mi = ctx.mla_of[l].expect("mla layer");
            let kv =
                KvView { data: ctx.kv, page: ctx.page, n_pages: ctx.n_pages, width: c.mla_cache_width(), layer: mi };
            mla_layer(c, w, kv, ctx.seqs, x, spans, out);
        }
    }
}

fn ffn(
    c: &K3Config,
    lw: &LayerW,
    experts: &dyn ExpertStore,
    l: usize,
    x: &[f32],
    n: usize,
    out: &mut [f32],
) -> Result<()> {
    match &lw.ffn {
        Ffn::Dense(w) => {
            dense_ffn(c, w, x, n, out);
            Ok(())
        }
        Ffn::Moe(w) => moe(c, w, experts, l, x, n, out),
    }
}

#[allow(clippy::too_many_arguments)]
fn attn_res_rows(
    h: &mut [f32],
    blocks: &[f32],
    nb: usize,
    maxs: usize,
    pref: &[f32],
    fold: &[f32],
    e: usize,
    eps: f32,
) {
    h.par_chunks_mut(e).enumerate().for_each(|(t, out)| {
        let mut srcs: Vec<&[f32]> = (0..nb).map(|b| &blocks[(t * maxs + b) * e..(t * maxs + b + 1) * e]).collect();
        srcs.push(&pref[t * e..(t + 1) * e]);
        ops::attn_res(out, &srcs, fold, eps);
    });
}

fn norm_rows(y: &mut [f32], x: &[f32], w: &[f32], e: usize, eps: f32) {
    y.par_chunks_mut(e).zip(x.par_chunks(e)).for_each(|(yr, xr)| ops::rmsnorm(yr, xr, w, eps));
}

// ------------------------------------------------------------------------------- KDA

fn kda_layer(c: &K3Config, w: &KdaW, ki: usize, x: &[f32], spans: &[Span], pool: &mut [f32], out: &mut [f32]) {
    let n = x.len() / c.hidden;
    let (hh, d, p, kk) = (c.kda_heads, c.kda_head_dim, c.kda_proj(), c.conv_k);
    let hist = kk - 1;

    // 1. projections, batched over every token in the step
    let qp = linear_new(x, n, &w.q);
    let kp = linear_new(x, n, &w.k);
    let vp = linear_new(x, n, &w.v);
    let bt = linear_new(x, n, &w.b);
    let fa = linear_new(x, n, &w.f_a);
    let z = linear_new(&fa, n, &w.f_b);
    let gate = match &w.gate {
        KdaGate::Full(g) => linear_new(x, n, g),
        KdaGate::LowRank { a, b } => linear_new(&linear_new(x, n, a), n, b),
    };

    let mut q = vec![0.0f32; n * p];
    let mut k = vec![0.0f32; n * p];
    let mut v = vec![0.0f32; n * p];
    let mut alpha = vec![0.0f32; n * p];
    let mut o = vec![0.0f32; n * p];

    let sf = c.kda_state_floats();
    let n_kda = c.n_kda();
    for sp in spans {
        let st = &mut pool[(sp.seq * n_kda + ki) * sf..(sp.seq * n_kda + ki + 1) * sf];
        let (s_mat, conv) = st.split_at_mut(hh * d * d);
        let rows = sp.off * p..(sp.off + sp.len) * p;

        // 2. ShortConv with fused SiLU, carrying K-1 raw inputs per channel
        let (cq, rest) = conv.split_at_mut(p * hist);
        let (ck, cv) = rest.split_at_mut(p * hist);
        ops::shortconv(&mut q[rows.clone()], &qp[rows.clone()], &w.q_conv, cq, p, kk, sp.len);
        ops::shortconv(&mut k[rows.clone()], &kp[rows.clone()], &w.k_conv, ck, p, kk, sp.len);
        ops::shortconv(&mut v[rows.clone()], &vp[rows.clone()], &w.v_conv, cv, p, kk, sp.len);

        for t in sp.off..sp.off + sp.len {
            // 3. L2 norm on q and k only, per head
            for h in 0..hh {
                ops::l2norm_inplace(&mut q[t * p + h * d..t * p + (h + 1) * d], 1e-6);
                ops::l2norm_inplace(&mut k[t * p + h * d..t * p + (h + 1) * d], 1e-6);
            }
            // 5. decay, A_log PER HEAD:
            //    K3:          g = lb * sigmoid(exp(A_log[h]) * (z + dt_bias))
            //    Kimi Linear: g = -exp(A_log[h]) * softplus(z + dt_bias)
            for h in 0..hh {
                let a = w.a_log[h].exp();
                for j in 0..d {
                    let i = h * d + j;
                    let zi = z[t * p + i] + w.dt_bias[i];
                    let g = if c.decay_softplus { -a * ops::softplus(zi) } else { c.gate_lb * ops::sigmoid(a * zi) };
                    alpha[t * p + i] = g.exp();
                }
            }
        }

        // 6. the recurrence, heads in parallel, positions in order within a head
        let qscale = 1.0 / (d as f32).sqrt();
        let op = SendPtr(o.as_mut_ptr());
        let (q, k, v, alpha, bt) = (&q, &k, &v, &alpha, &bt);
        s_mat.par_chunks_mut(d * d).enumerate().for_each(|(h, s)| {
            let mut qs = vec![0.0f32; d];
            for t in sp.off..sp.off + sp.len {
                let r = t * p + h * d..t * p + (h + 1) * d;
                for (dst, &src) in qs.iter_mut().zip(&q[r.clone()]) {
                    *dst = src * qscale;
                }
                let beta = ops::sigmoid(bt[t * hh + h]);
                // SAFETY: head h owns o[t][h*d..(h+1)*d] for every t.
                let orow = unsafe { std::slice::from_raw_parts_mut(op.get().add(t * p + h * d), d) };
                ops::kda_step(s, orow, &qs, &k[r.clone()], &v[r.clone()], &alpha[r], beta);
            }
        });
    }

    // 7-9. head-wise RMSNorm, THEN the output gate, THEN o_proj (MLA gates without a norm)
    o.par_chunks_mut(p).enumerate().for_each(|(t, row)| {
        for h in 0..hh {
            ops::rmsnorm_inplace(&mut row[h * d..(h + 1) * d], &w.o_norm, c.rms_eps);
        }
        for i in 0..p {
            row[i] *= ops::sigmoid(gate[t * p + i]);
        }
    });
    linear(&o, n, &w.o, out);
}

// ------------------------------------------------------------------------------- MLA

struct KvView<'a> {
    data: &'a mut [f32],
    page: usize,
    n_pages: usize,
    width: usize,
    layer: usize,
}

impl KvView<'_> {
    fn index(&self, pages: &[u32], pos: usize) -> usize {
        let pg = pages[pos / self.page] as usize;
        ((self.layer * self.n_pages + pg) * self.page + pos % self.page) * self.width
    }
}

fn mla_layer(c: &K3Config, w: &MlaW, kv: KvView<'_>, seqs: &[SeqState], x: &[f32], spans: &[Span], out: &mut [f32]) {
    let n = x.len() / c.hidden;
    let hh = c.n_heads;
    let (qn, qr, vh, lat) = (c.qk_nope, c.qk_rope, c.v_head, c.kv_lora);
    let qh = qn + qr;
    let width = lat + qr;
    let scale = 1.0 / (qh as f32).sqrt(); // over the FULL head width, rope slots included

    let q = match &w.q {
        MlaQuery::Lora { a, norm, b } => {
            let mut ql = linear_new(x, n, a);
            ql.par_chunks_mut(c.q_lora).for_each(|r| ops::rmsnorm_inplace(r, norm, c.rms_eps));
            linear_new(&ql, n, b)
        }
        MlaQuery::Direct(qp) => linear_new(x, n, qp),
    };
    let mut ckv = linear_new(x, n, &w.kv_a);
    // the norm covers the latent only, never the rope slot
    ckv.par_chunks_mut(width).for_each(|r| ops::rmsnorm_inplace(&mut r[..lat], &w.kv_a_norm, c.kv_norm_eps));

    // append this step's positions to each sequence's pages
    for sp in spans {
        let pages = &seqs[sp.seq].pages;
        for t in 0..sp.len {
            let at = kv.index(pages, sp.pos + t);
            kv.data[at..at + width].copy_from_slice(&ckv[(sp.off + t) * width..(sp.off + t + 1) * width]);
        }
    }
    let kvd: &[f32] = kv.data;

    // (token, head) pairs in parallel; causal over each sequence's own positions
    let mut y = vec![0.0f32; n * hh * vh];
    let tok_pos: Vec<(usize, usize)> =
        spans.iter().flat_map(|sp| (0..sp.len).map(move |t| (sp.seq, sp.pos + t))).collect();
    y.par_chunks_mut(vh).enumerate().for_each(|(i, yo)| {
        let (t, h) = (i / hh, i % hh);
        let (seq, pos) = tok_pos[t];
        let pages = &seqs[seq].pages;
        let qt = &q[(t * hh + h) * qh..(t * hh + h + 1) * qh];
        let (q_nope, q_rope) = qt.split_at(qn);

        // absorb: q_lat = W_UK^T q_nope
        let base = h * (qn + vh);
        let mut q_lat = vec![0.0f32; lat];
        let mut row = vec![0.0f32; lat];
        for (j, &qv) in q_nope.iter().enumerate() {
            w.kv_b.row_f32(base + j, &mut row);
            for (a, b) in q_lat.iter_mut().zip(&row) {
                *a += qv * b;
            }
        }
        let mut sc = vec![0.0f32; pos + 1];
        let mut m = f32::NEG_INFINITY;
        for (s, slot) in sc.iter_mut().enumerate() {
            let at = kv.index(pages, s);
            let ent = &kvd[at..at + width];
            let mut d = 0.0f32;
            for (a, b) in q_lat.iter().zip(&ent[..lat]) {
                d += a * b;
            }
            // the rope slot is UNROTATED but still scored, the same 64 values for every head
            for (a, b) in q_rope.iter().zip(&ent[lat..]) {
                d += a * b;
            }
            *slot = d * scale;
            m = m.max(*slot);
        }
        let mut zsum = 0.0f64;
        for s in sc.iter_mut() {
            *s = (*s - m).exp();
            zsum += *s as f64;
        }
        let mut o_lat = vec![0.0f32; lat];
        for (s, &pr) in sc.iter().enumerate() {
            let pr = (pr as f64 / zsum) as f32;
            let at = kv.index(pages, s);
            for (a, b) in o_lat.iter_mut().zip(&kvd[at..at + lat]) {
                *a += pr * b;
            }
        }
        // un-absorb: o = W_UV o_lat
        fullm_kernels::gemm(&o_lat, 1, lat, w.kv_b.rows_ref(base + qn, base + qn + vh), vh, 0, vh, yo, vh);
    });

    // gate BEFORE o_proj, with no norm
    if let Some(g) = &w.g {
        let gate = linear_new(x, n, g);
        y.par_iter_mut().zip(gate.par_iter()).for_each(|(a, &b)| *a *= ops::sigmoid(b));
    }
    linear(&y, n, &w.o, out);
}

// ------------------------------------------------------------------------------- FFN

fn dense_ffn(c: &K3Config, w: &DenseW, x: &[f32], n: usize, out: &mut [f32]) {
    let g = linear_new(x, n, &w.gate);
    let u = linear_new(x, n, &w.up);
    let mut a = vec![0.0f32; n * c.dense_inter];
    ops::glu(&mut a, &g, &u, c.act_silu, c.situ_b1, c.situ_b2);
    linear(&a, n, &w.down, out);
}

/// Stable LatentMoE. Order (modeling_kimi_linear.py:815-838):
///   route on the FULL width -> down-project to the latent -> weighted sum of the selected
///   experts in latent space -> RMSNorm the AGGREGATE -> up-project -> add the shared
///   expert, computed on the ORIGINAL input and added UNWEIGHTED.
fn moe(
    c: &K3Config,
    w: &MoeW,
    experts: &dyn ExpertStore,
    layer: usize,
    x: &[f32],
    n: usize,
    out: &mut [f32],
) -> Result<()> {
    let (l, inter, k) = (c.latent, c.moe_inter, c.topk);

    let logits = linear_new(x, n, &w.gate);
    let routes: Vec<(Vec<u32>, Vec<f32>)> = logits
        .par_chunks(c.n_experts)
        .map(|lg| ops::route(lg, &w.bias, k, c.moe_renorm, c.routed_scale, c.router_weights_biased))
        .collect();
    // K3 routes experts in a latent space; Kimi Linear runs them at full width.
    let z = match &w.latent {
        Some(lt) => linear_new(x, n, &lt.down),
        None => x.to_vec(),
    };

    // Group (token, slot) pairs by expert: each expert's weights are read once per step.
    let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); c.n_experts];
    for (t, (idx, _)) in routes.iter().enumerate() {
        for (j, &ex) in idx.iter().enumerate() {
            by_expert[ex as usize].push((t, j));
        }
    }
    let active: Vec<(usize, &Vec<(usize, usize)>)> =
        by_expert.iter().enumerate().filter(|(_, v)| !v.is_empty()).collect();
    experts.prefetch(layer, &active.iter().map(|(e, _)| *e as u32).collect::<Vec<_>>());

    // Each (token, slot) gets its own output row so the per-token reduction below can run
    // in selection order no matter how experts were scheduled.
    let mut contrib = vec![0.0f32; n * k * l];
    let cp = SendPtr(contrib.as_mut_ptr());
    let (zr, routes_r) = (&z, &routes);
    active.par_iter().try_for_each(|&(ex, list)| -> Result<()> {
        let er = experts.get(layer, ex)?;
        let m = list.len();
        let mut xin = vec![0.0f32; m * l];
        for (r, &(t, _)) in list.iter().enumerate() {
            xin[r * l..(r + 1) * l].copy_from_slice(&zr[t * l..(t + 1) * l]);
        }
        let mut g = vec![0.0f32; m * inter];
        let mut u = vec![0.0f32; m * inter];
        linear_ref(&xin, m, l, er.w1, inter, &mut g);
        linear_ref(&xin, m, l, er.w3, inter, &mut u);
        let mut a = vec![0.0f32; m * inter];
        ops::glu(&mut a, &g, &u, c.act_silu, c.situ_b1, c.situ_b2);
        let mut d = vec![0.0f32; m * l];
        linear_ref(&a, m, inter, er.w2, l, &mut d);
        for (r, &(t, j)) in list.iter().enumerate() {
            let wj = routes_r[t].1[j];
            // SAFETY: (t, j) is unique across all experts.
            let dst = unsafe { std::slice::from_raw_parts_mut(cp.get().add((t * k + j) * l), l) };
            for (o, &v) in dst.iter_mut().zip(&d[r * l..(r + 1) * l]) {
                *o = wj * v;
            }
        }
        Ok(())
    })?;

    let mut acc = vec![0.0f32; n * l];
    acc.par_chunks_mut(l).enumerate().for_each(|(t, a)| {
        for j in 0..k {
            for (o, &v) in a.iter_mut().zip(&contrib[(t * k + j) * l..(t * k + j + 1) * l]) {
                *o += v;
            }
        }
        if let Some(norm) = w.latent.as_ref().and_then(|lt| lt.norm.as_ref()) {
            ops::rmsnorm_inplace(a, norm, c.rms_eps);
        }
    });
    match &w.latent {
        Some(lt) => linear(&acc, n, &lt.up, out),
        None => out.copy_from_slice(&acc),
    }

    let si = c.moe_inter * c.n_shared;
    let sg = linear_new(x, n, &w.sh_gate);
    let su = linear_new(x, n, &w.sh_up);
    let mut sa = vec![0.0f32; n * si];
    ops::glu(&mut sa, &sg, &su, c.act_silu, c.situ_b1, c.situ_b2);
    let sd = linear_new(&sa, n, &w.sh_down);
    out.par_iter_mut().zip(sd.par_iter()).for_each(|(o, &s)| *o += s);
    if out.iter().any(|v| !v.is_finite()) {
        bail!("non-finite activation after MoE layer {layer}");
    }
    Ok(())
}
