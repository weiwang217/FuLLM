// SPDX-License-Identifier: Apache-2.0
//! Inkling 2 (looped) on the C++ executor (ink2_exec.cpp): the serving path.
//!
//! The control plane (which slot and cache pages a sequence owns, the step plan) is `Paged`, as
//! for the other executors. The dense matrices are copied into the executor as bf16; the
//! embedding table and the experts stay in the mapped checkpoint and are read in place, the
//! experts in the formats the release stores them in (NVFP4, MXFP8): the 20 GB file is the
//! model's memory.

use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, bail, ensure, Context, Result};
use fullm_kernels::exec::{Executor, ExpertFmt, Fmt, Ink2Dims, Ink2Mtp};

use super::{Ink2Config, PREFIX};
use crate::models::paged::{Geometry, Paged};
use crate::models::{BackendKind, DraftReq, LoadOptions, Model};
use crate::st::{Dtype, SafeTensors};
use crate::tensor::Precision;

pub struct Ink2Exec {
    cfg: Ink2Config,
    // drops first: the executor reads the mapping and the factors below in place
    paged: Paged,
    backend: BackendKind,
    _factors: Vec<Vec<f32>>,
    _st: Arc<SafeTensors>,
    desc: String,
    /// Tokens a draft call may ask of the head (its block less the anchor); 0: no head.
    draft_max: usize,
    /// A chain of drafts is kept while the product of its probabilities under the head stays above
    /// this (FULLM_INK2_DRAFT_CONF; 0: every draft, and the head never sits out): a draft row costs about a third of a step here (its eight
    /// experts in each of 48 visits are read for it alone), so a draft kept less often loses time.
    draft_conf: f32,
    /// By sequence: the head's record (`DraftRecord`).
    draft_rec: Vec<DraftRecord>,
}

/// What a draft row and a call of the head cost, in single-row steps (M3 Pro: 20 and 17-21 ms
/// against 66): a kept draft saves one.
const DRAFT_ROW_COST: f32 = 0.3;
const DRAFT_CALL_COST: f32 = 0.25;

/// A sequence's record with the draft head: the head sits out `wait` calls after its drafts stop
/// paying (a chain it did not trust itself, or an average gain below zero), twice as many each
/// time in a row, up to 8.
#[derive(Clone, Copy, Default)]
struct DraftRecord {
    /// The sequence's length at the last call (a shorter one: another sequence in the slot).
    len: usize,
    /// Drafts were handed out at the last call: the step after it tells what they were worth.
    pending: bool,
    /// Steps gained per call, averaged over the recent ones.
    gain: f32,
    wait: u32,
    backoff: u32,
}

impl DraftRecord {
    fn sit_out(&mut self) {
        self.backoff = (2 * self.backoff).clamp(1, 8);
        self.wait = self.backoff;
        self.gain = 0.0;
    }
}

/// The release's draft head (mtp.safetensors beside the model, `mtp_config` in its config; a
/// DSpark draft): its tensors into the executor as "mtp.*". Ok(false): the checkpoint has none.
fn load_mtp(ex: &mut Executor, st: &SafeTensors, config: &serde_json::Value, c: &Ink2Config) -> Result<bool> {
    let Some(m) = config.get("mtp_config") else { return Ok(false) };
    if st.get("fuse_weight").is_none() {
        return Ok(false);
    }
    let u = |k: &str| -> Result<usize> {
        m.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).with_context(|| format!("mtp_config: missing {k}"))
    };
    let ids = |k: &str| -> Result<Vec<usize>> {
        Ok(m.get(k)
            .and_then(|x| x.as_array())
            .with_context(|| format!("mtp_config: missing {k}"))?
            .iter()
            .filter_map(|x| x.as_u64().map(|x| x as usize))
            .collect())
    };
    let kind = m.get("markov_head_type").and_then(|x| x.as_str()).unwrap_or("");
    ensure!(kind == "rnn_hidden", "mtp_config: markov head {kind:?} (only rnn_hidden is implemented)");
    let (h, nl, hq, hk, d) =
        (c.hidden, u("num_hidden_layers")?, u("num_attention_heads")?, u("num_key_value_heads")?, u("head_dim")?);
    let (inter, rank, block) = (u("intermediate_size")?, u("markov_rank")?, u("block_size")?);
    ensure!(u("hidden_size")? == h, "mtp_config: hidden size differs from the model's");
    let (taps, local) = (ids("target_layer_ids")?, ids("local_layer_ids")?);
    ensure!(taps.len() == 4 && nl <= 8, "mtp_config: expected 4 target_layer_ids and at most 8 layers");
    let mut cfg = Ink2Mtp {
        n_layers: nl as i32,
        hq: hq as i32,
        hk: hk as i32,
        d: d as i32,
        inter: inter as i32,
        window: u("sliding_window_size")? as i32,
        block: block as i32,
        rank: rank as i32,
        theta: m.get("rope_theta").and_then(|x| x.as_f64()).unwrap_or(1e6) as f32,
        ..Default::default()
    };
    for (i, &t) in taps.iter().enumerate() {
        cfg.taps[i] = t as i32;
    }
    for &l in local.iter().filter(|&&l| l < 8) {
        cfg.local[l] = 1;
    }
    ex.set_ink2_mtp(&cfg).map_err(|e| anyhow!(e))?;
    let mt = |ex: &mut Executor, from: &str, to: &str, rows: usize, cols: usize| -> Result<()> {
        let v = st.need(from)?;
        ensure!(
            v.dtype() == Dtype::BF16 && v.shape() == [rows, cols],
            "{from}: {:?} {:?}, expected bf16 [{rows}, {cols}]",
            v.dtype(),
            v.shape()
        );
        ex.add_matrix(to, Fmt::Bf16, rows, cols, v.bytes, None).map_err(|e| anyhow!(e))
    };
    let vc = |ex: &mut Executor, from: &str, to: &str, n: usize| -> Result<()> {
        let v = st.need(from)?.to_f32()?;
        ensure!(v.len() == n, "{from}: {} values, expected {n}", v.len());
        ex.add_vector(to, &v).map_err(|e| anyhow!(e))
    };
    for i in 0..nl {
        let (s, p) = (format!("layers.{i}."), format!("mtp.L{i}."));
        mt(ex, &format!("{s}self_attn.q_proj.weight"), &format!("{p}q"), hq * d, h)?;
        mt(ex, &format!("{s}self_attn.k_proj.weight"), &format!("{p}k"), hk * d, h)?;
        mt(ex, &format!("{s}self_attn.v_proj.weight"), &format!("{p}v"), hk * d, h)?;
        mt(ex, &format!("{s}self_attn.k_proj_ctx.weight"), &format!("{p}k_ctx"), hk * d, h)?;
        mt(ex, &format!("{s}self_attn.v_proj_ctx.weight"), &format!("{p}v_ctx"), hk * d, h)?;
        mt(ex, &format!("{s}self_attn.o_proj.weight"), &format!("{p}o"), h, hq * d)?;
        mt(ex, &format!("{s}mlp.gate_proj.weight"), &format!("{p}gate"), inter, h)?;
        mt(ex, &format!("{s}mlp.up_proj.weight"), &format!("{p}up"), inter, h)?;
        mt(ex, &format!("{s}mlp.down_proj.weight"), &format!("{p}down"), h, inter)?;
        vc(ex, &format!("{s}dflare_norm.weight"), &format!("{p}dflare"), h)?;
        vc(ex, &format!("{s}input_layernorm.weight"), &format!("{p}in_norm"), h)?;
        vc(ex, &format!("{s}post_attention_layernorm.weight"), &format!("{p}post_norm"), h)?;
        vc(ex, &format!("{s}self_attn.q_norm.weight"), &format!("{p}q_norm"), d)?;
        vc(ex, &format!("{s}self_attn.k_norm.weight"), &format!("{p}k_norm"), d)?;
        vc(ex, &format!("{s}self_attn.k_norm_ctx.weight"), &format!("{p}k_norm_ctx"), d)?;
    }
    mt(ex, "markov_head.markov_w1.weight", "mtp.w1", c.vocab, rank)?;
    mt(ex, "markov_head.hidden_out.weight", "mtp.hidden_out", h, rank)?;
    {
        // rnn_proj reads [state | previous token | row | feature]: the first two apart
        let v = st.need("markov_head.rnn_proj.weight")?;
        let cols = 2 * rank + 2 * h;
        ensure!(
            v.dtype() == Dtype::BF16 && v.shape() == [3 * rank, cols],
            "markov_head.rnn_proj.weight: {:?}",
            v.shape()
        );
        let (mut a, mut b) = (Vec::with_capacity(3 * rank * 2 * rank * 2), Vec::with_capacity(3 * rank * 2 * h * 2));
        for r in 0..3 * rank {
            let row = &v.bytes[r * cols * 2..(r + 1) * cols * 2];
            a.extend_from_slice(&row[..4 * rank]);
            b.extend_from_slice(&row[4 * rank..]);
        }
        ex.add_matrix("mtp.rnn_s", Fmt::Bf16, 3 * rank, 2 * rank, &a, None).map_err(|e| anyhow!(e))?;
        ex.add_matrix("mtp.rnn_x", Fmt::Bf16, 3 * rank, 2 * h, &b, None).map_err(|e| anyhow!(e))?;
    }
    vc(ex, "fuse_weight", "mtp.fuse", (nl + 1) * 4)?;
    vc(ex, "mask_embed", "mtp.mask", h)?;
    vc(ex, "norm.weight", "mtp.norm", h)?;
    vc(ex, "markov_head.dflare_norm.weight", "mtp.m_dflare", h)?;
    vc(ex, "markov_head.hidden_out.bias", "mtp.hidden_out_b", h)?;
    vc(ex, "markov_head.rnn_proj.bias", "mtp.rnn_b", 3 * rank)?;
    Ok(true)
}

fn dims(c: &Ink2Config) -> Ink2Dims {
    let i = |v: usize| v as i32;
    let h = |t: (usize, usize, usize)| (i(t.0), i(t.1), i(t.2));
    Ink2Dims {
        hidden: i(c.hidden),
        n_layers: i(c.n_layers),
        vocab: i(c.vocab),
        out_vocab: i(c.out_vocab),
        rms_eps: c.rms_eps,
        full: h(c.full_heads),
        swa: h(c.swa_heads),
        window: i(c.window),
        d_rel: i(c.d_rel),
        rel_extent: i(c.rel_extent),
        log_n_floor: c.log_n_floor,
        log_alpha: c.log_alpha,
        qk_clip: c.qk_clip,
        rel_clip: c.rel_clip,
        conv_k: i(c.conv_k),
        block: i(c.block),
        n_experts: i(c.n_experts),
        topk: i(c.topk),
        moe_inter: i(c.moe_inter),
        gate_softcap: c.gate_softcap,
        up_softcap: c.up_softcap,
        route_scale: c.route_scale,
        logit_softcap: c.logit_softcap,
        loop_start: i(c.loop_start),
        loop_end: i(c.loop_end),
        passes: i(c.passes),
        gate_rank: i(c.gate_rank),
        layer_sliding: c.sliding.iter().map(|&s| s as u8).collect(),
    }
}

/// The checkpoint tensor `name` (without the prefix) as executor matrix `to`, its first `rows` rows.
fn mat(ex: &mut Executor, st: &SafeTensors, name: &str, to: &str, rows: usize, cols: usize) -> Result<()> {
    let v = st.need(&format!("{PREFIX}{name}"))?;
    let (fmt, size) = match v.dtype() {
        Dtype::BF16 => (Fmt::Bf16, 2),
        Dtype::F32 => (Fmt::F32, 4),
        d => bail!("{name}: expected BF16 or F32, found {d:?}"),
    };
    ensure!(v.numel() >= rows * cols && v.numel() % cols == 0, "{name}: {:?}, expected [{rows}, {cols}]", v.shape());
    ex.add_matrix(to, fmt, rows, cols, &v.bytes[..rows * cols * size], None).map_err(|e| anyhow!(e))
}

fn vec(ex: &mut Executor, st: &SafeTensors, name: &str, to: &str, n: usize) -> Result<()> {
    let v = st.need(&format!("{PREFIX}{name}"))?.to_f32()?;
    ensure!(v.len() == n, "{name}: {} values, expected {n}", v.len());
    ex.add_vector(to, &v).map_err(|e| anyhow!(e))
}

impl Ink2Exec {
    pub fn load(dir: &Path, config: &serde_json::Value, opts: &LoadOptions) -> Result<Self> {
        let cfg = Ink2Config::from_json(config)?;
        let c = &cfg;
        let geo = Geometry::new(opts)?;
        let st = SafeTensors::open_dir(dir)?;
        let mut ex =
            Executor::new_ink2(&dims(c), &geo.exec_opts(opts)).map_err(|e| anyhow!("creating the executor: {e}"))?;
        let t0 = std::time::Instant::now();
        let h = c.hidden;

        let emb = st.need(&format!("{PREFIX}embed.weight"))?;
        let efmt = match emb.dtype() {
            Dtype::BF16 => Fmt::Bf16,
            Dtype::F32 => Fmt::F32,
            d => bail!("embed.weight: expected BF16 or F32, found {d:?}"),
        };
        ensure!(emb.numel() == c.vocab * h, "embed.weight: {:?}", emb.shape());
        // SAFETY: the mapping is held by Self and outlives the executor (field order).
        unsafe { ex.add_embedding(efmt, c.vocab, h, emb.bytes.as_ptr()) }.map_err(|e| anyhow!(e))?;
        mat(&mut ex, &st, "unembed.weight", "lm_head", c.out_vocab, h)?;
        vec(&mut ex, &st, "embed_norm.weight", "embed_norm", h)?;
        vec(&mut ex, &st, "norm.weight", "final_norm", h)?;
        vec(&mut ex, &st, "attn_res.output_query", "out_query", h)?;
        {
            // the release stores [query, mlp] and [key, value] per loop layer; the executor takes
            // query, key, value, mlp
            let nl = c.loop_end - c.loop_start;
            let qm = st.need(&format!("{PREFIX}middle_cycle_cross_pass_queries"))?.to_f32()?;
            let kv = st.need(&format!("{PREFIX}middle_cycle_cross_pass_kv_queries"))?.to_f32()?;
            ensure!(qm.len() == 2 * nl * h && kv.len() == 2 * nl * h, "cross-pass queries: unexpected sizes");
            let mut m = Vec::with_capacity(4 * nl * h);
            for i in 0..nl {
                m.extend_from_slice(&qm[2 * i * h..(2 * i + 1) * h]);
                m.extend_from_slice(&kv[2 * i * h..(2 * i + 2) * h]);
                m.extend_from_slice(&qm[(2 * i + 1) * h..(2 * i + 2) * h]);
            }
            ex.add_vector("msg_queries", &m).map_err(|e| anyhow!(e))?;
        }

        let mut factors = Vec::new();
        // FULLM_INK2_PREFAULT=0: leave the experts' pages to fault in as tokens reach them
        let prefault = std::env::var("FULLM_INK2_PREFAULT").map_or(true, |v| v != "0");
        let mut touched = 0u64;
        // The experts' pages are wired (mlock): read in place from the mapped file, they are
        // otherwise unmapped again behind the reader, and a token then pays soft faults for the
        // ~1.5 GB of experts it touches (M3 Pro, 6 threads: the experts' share of a decode step
        // 114 ms, 34 ms wired). FULLM_INK2_MLOCK=0: not wired (the pages stay evictable).
        let mlock = std::env::var("FULLM_INK2_MLOCK").map_or(true, |v| v != "0");
        let mut locked_ok = true;
        for l in 0..c.n_layers {
            let (hq, hk, d) = c.heads(l);
            let (s, p) = (format!("layers.{l}."), format!("L{l}."));
            let mut m = |from: &str, to: &str, rows: usize, cols: usize| {
                mat(&mut ex, &st, &format!("{s}{from}"), &format!("{p}{to}"), rows, cols)
                    .with_context(|| format!("layer {l}"))
            };
            m("attn.wq_du.weight", "q", hq * d, h)?;
            m("attn.wk_dv.weight", "k", hk * d, h)?;
            m("attn.wv_dv.weight", "v", hk * d, h)?;
            m("attn.wo_ud.weight", "o", h, hq * d)?;
            m("attn.gate_attn.weight", "og", hq * d, h)?;
            m("attn.attn_sink_proj.weight", "sink", hq, h)?;
            m("attn.wr_du.weight", "rel", hq * c.d_rel, h)?;
            m("mlp.gate.weight", "router", c.n_experts, h)?;
            if !c.sliding[l] {
                m("attn.k_sconv.weight_proj", "k_wp", hk * c.conv_k, h)?;
                m("attn.v_sconv.weight_proj", "v_wp", hk * c.conv_k, h)?;
            }
            if c.in_loop(l) {
                for (g, to) in [("attn", "cq"), ("k", "ck"), ("v", "cv"), ("mlp", "cm")] {
                    m(&format!("controllers.{g}.down_weight"), &format!("{to}_down"), c.gate_rank, 2 * h)?;
                    m(&format!("controllers.{g}.forget_weight"), &format!("{to}_forget"), h, c.gate_rank)?;
                    m(&format!("controllers.{g}.gain_weight"), &format!("{to}_gain"), h, c.gate_rank)?;
                }
            }
            let mut v = |from: &str, to: &str, n: usize| {
                vec(&mut ex, &st, from, &format!("{p}{to}"), n).with_context(|| format!("layer {l}"))
            };
            if l > 0 {
                // layer l's attention-input queries are stored one index down
                v(&format!("attn_res.attn_queries.{}", l - 1), "attn_query", if c.sliding[l] { h } else { 3 * h })?;
            }
            v(&format!("attn_res.mlp_queries.{l}"), "mlp_query", h)?;
            v(&format!("{s}mlp_norm.weight"), "post_norm", h)?;
            v(&format!("{s}attn.q_l2_clip.scale"), "q_scale", hq * d)?;
            v(&format!("{s}attn.rel_l2_clip.scale"), "rel_scale", hq * c.d_rel)?;
            let extent = if c.sliding[l] { c.window } else { c.rel_extent };
            v(&format!("{s}attn.rel_logits_proj.proj"), "rel_proj", c.d_rel * extent)?;
            v(&format!("{s}attn_sconv.weight"), "attn_conv", h * c.conv_k)?;
            v(&format!("{s}mlp_sconv.weight"), "mlp_conv", h * c.conv_k)?;
            v(&format!("{s}mlp.gate.bias"), "router_bias", c.n_experts)?;
            if c.sliding[l] {
                v(&format!("{s}attn_norm.weight"), "in_norm", h)?;
                v(&format!("{s}attn.k_sconv.weight"), "k_conv", hk * d * c.conv_k)?;
                v(&format!("{s}attn.v_sconv.weight"), "v_conv", hk * d * c.conv_k)?;
            } else {
                for n in ["q", "k", "v"] {
                    v(&format!("{s}{n}_input_norm.weight"), &format!("{n}_norm"), h)?;
                }
                v(&format!("{s}attn.k_sconv.static_w"), "k_static", c.conv_k * hk * d)?;
                v(&format!("{s}attn.v_sconv.static_w"), "v_static", c.conv_k * hk * d)?;
            }

            // the experts, in place
            let stack = |which: &str,
                         rows: usize,
                         cols: usize|
             -> Result<(ExpertFmt, &[u8], Option<&[u8]>, Option<Vec<f32>>)> {
                let name = format!("{PREFIX}{s}mlp.experts.{which}_weight");
                let w = st.need(&name)?;
                let n = c.n_experts * rows * cols;
                Ok(match w.dtype() {
                    Dtype::U8 => {
                        let sc = st.need(&format!("{name}_scale"))?;
                        let amax = st.need(&format!("{name}_amax"))?.to_f32()?;
                        let den = st.need(&format!("{name}_outer_scale_denominator"))?.to_f32()?;
                        ensure!(sc.dtype() == Dtype::F8E4M3, "{name}_scale: expected F8_E4M3");
                        ensure!(
                            w.bytes.len() == n / 2 && sc.bytes.len() == n / 16,
                            "{name}: NVFP4 sizes {:?}",
                            w.shape()
                        );
                        ensure!(amax.len() == c.n_experts && den.len() == 1 && den[0] != 0.0, "{name}: NVFP4 factors");
                        (ExpertFmt::Nvfp4, w.bytes, Some(sc.bytes), Some(amax.iter().map(|a| a / den[0]).collect()))
                    }
                    Dtype::F8E4M3 => {
                        let sc = st.need(&format!("{name}_scale"))?;
                        ensure!(sc.dtype() == Dtype::U8, "{name}_scale: expected U8 exponents");
                        ensure!(w.bytes.len() == n && sc.bytes.len() == n / 32, "{name}: MXFP8 sizes {:?}", w.shape());
                        (ExpertFmt::Mxfp8, w.bytes, Some(sc.bytes), None)
                    }
                    Dtype::BF16 => {
                        ensure!(w.bytes.len() == n * 2, "{name}: bf16 size {:?}", w.shape());
                        (ExpertFmt::Bf16, w.bytes, None, None)
                    }
                    d => bail!("{name}: unsupported expert dtype {d:?}"),
                })
            };
            let (f13, w13, s13, x13) = stack("w13", 2 * c.moe_inter, h)?;
            let (f2, w2, s2, x2) = stack("w2", h, c.moe_inter)?;
            ensure!(f13 == f2, "layer {l}: gate/up and down experts in different formats");
            // SAFETY: the mapping and the factor vectors are held by Self and outlive the executor;
            // the sizes were checked against the model's shapes above.
            unsafe { ex.set_ink2_experts(l, f13, w13, s13, x13.as_deref(), w2, s2, x2.as_deref()) }
                .map_err(|e| anyhow!("layer {l}: {e}"))?;
            if mlock {
                for b in [w13, w2] {
                    let page = 16384usize;
                    let p0 = (b.as_ptr() as usize) & !(page - 1);
                    let len = b.as_ptr() as usize + b.len() - p0;
                    // SAFETY: the range lies within the mapping; a failure only leaves it unlocked.
                    if unsafe { libc::mlock(p0 as *const libc::c_void, len) } != 0 {
                        locked_ok = false;
                    }
                }
            }
            // not wired: at least map the pages now, or the first tokens fault in the pages of
            // every expert they are first routed to
            if prefault && !mlock {
                use rayon::prelude::*;
                for b in [w13, w2] {
                    touched += b
                        .par_chunks(1 << 20)
                        .map(|c| c.iter().step_by(4096).map(|&x| x as u64).sum::<u64>())
                        .sum::<u64>();
                }
            }
            factors.extend(x13);
            factors.extend(x2);
            if l % 8 == 0 || l + 1 == c.n_layers {
                tracing::info!("layer {}/{} loaded ({:.1}s)", l + 1, c.n_layers, t0.elapsed().as_secs_f64());
            }
        }
        std::hint::black_box(touched);
        if mlock && !locked_ok {
            tracing::warn!("could not wire the experts' pages (mlock failed: see ulimit -l); decoding will be slower");
        }
        // --trunk int8 / --head int8: the dense matrices as int8 with a scale per row (the GPU path)
        let w8 = |p: Precision, flag: &str| match p {
            Precision::Bf16 => Ok(false),
            Precision::Int8 => Ok(true),
            p => Err(anyhow!("{flag} {p:?}: Inkling 2's dense matrices are bf16 or int8")),
        };
        let (dense, head) = (w8(opts.trunk, "--trunk")?, w8(opts.head.unwrap_or(Precision::Bf16), "--head")?);
        if dense || head {
            ex.ink2_int8(dense, head).map_err(|e| anyhow!(e))?;
        }
        // FULLM_INK2_MTP=0: without the draft head
        if std::env::var("FULLM_INK2_MTP").map_or(true, |v| v != "0") {
            load_mtp(&mut ex, &st, config, c).context("the draft head (mtp.safetensors)")?;
        }
        // the prefix cache: a page starts another sequence with the convolution histories kept for it
        let cache = !opts.no_prefix_cache;
        if cache {
            ex.page_state_enable().map_err(|e| anyhow!("keeping the histories by page: {e}"))?;
        }
        ex.finalize().map_err(|e| anyhow!("finalizing the executor: {e}"))?;
        let desc = ex.describe();
        let draft_max = if ex.ink2_has_draft() {
            let b = config.get("mtp_config").and_then(|m| m.get("block_size")).and_then(|x| x.as_u64());
            b.map_or(0, |b| b as usize - 1)
        } else {
            0
        };
        let mut paged = Paged::new(ex, cfg.out_vocab, &geo, opts, true)?;
        paged.set_page_state(cache);
        Ok(Ink2Exec {
            paged,
            backend: opts.backend.clone(),
            _factors: factors,
            _st: st,
            desc,
            cfg,
            draft_max,
            draft_conf: std::env::var("FULLM_INK2_DRAFT_CONF").ok().and_then(|v| v.parse().ok()).unwrap_or(0.35),
            draft_rec: vec![DraftRecord::default(); opts.max_seqs],
        })
    }

    pub fn config(&self) -> &Ink2Config {
        &self.cfg
    }
}

impl Model for Ink2Exec {
    fn arch(&self) -> &'static str {
        "inkling2_loop"
    }

    fn vocab(&self) -> usize {
        self.cfg.out_vocab
    }

    fn describe(&self) -> String {
        format!("{} [{:?} backend]", self.desc, self.backend)
    }

    /// Drafts are taken: a rejected one's cache positions are written again, and the executor
    /// puts the convolution histories back (`Paged::accept` -> `drop_tail`; up to 16 a chunk).
    fn speculation(&self) -> bool {
        true
    }

    fn draft_head(&self) -> bool {
        self.draft_max > 0
    }

    /// The model's own draft head, after a step and its `accept`s. Its context (keys and values
    /// from the target's features) is written by the steps themselves, so only the sequences that
    /// ask for drafts cost anything here.
    fn draft(&mut self, reqs: &[DraftReq]) -> Result<Vec<Vec<u32>>> {
        ensure!(self.draft_max > 0, "this model has no draft head");
        let mut out = vec![Vec::new(); reqs.len()];
        let (mut who, mut slots, mut next, mut pos) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut kmax = 0;
        for (i, r) in reqs.iter().enumerate() {
            if r.k == 0 {
                continue;
            }
            let len = self.paged.seq_len(r.seq);
            let rec = &mut self.draft_rec[r.seq];
            if len < rec.len {
                *rec = DraftRecord::default();
            }
            rec.len = len;
            // (FULLM_INK2_DRAFT_CONF=0: every draft, every call)
            if std::mem::take(&mut rec.pending) && self.draft_conf > 0.0 {
                let (d, kept) = self.paged.last_draft(r.seq);
                let gain = kept as f32 - DRAFT_ROW_COST * d as f32 - DRAFT_CALL_COST;
                rec.gain = 0.7 * rec.gain + 0.3 * gain;
                if rec.gain < 0.0 {
                    rec.sit_out();
                } else if kept > 0 {
                    rec.backoff = 0;
                }
            }
            if rec.wait > 0 {
                rec.wait -= 1;
                continue;
            }
            // (as many as the cache can take after the token itself)
            let mut k = r.k.min(self.draft_max);
            while k > 0 && !self.paged.can_grow(r.seq, 1 + k) {
                k -= 1;
            }
            if k == 0 {
                continue;
            }
            who.push((i, k));
            slots.push(r.seq as i32);
            next.push(r.next as i32);
            pos.push(self.paged.seq_len(r.seq) as i32);
            kmax = kmax.max(k);
        }
        if who.is_empty() {
            return Ok(out);
        }
        let table = self.paged.page_table().to_vec();
        let (toks, conf) =
            self.paged.ex.ink2_draft(&slots, &next, &pos, &table, kmax).map_err(|e| anyhow!("draft head: {e}"))?;
        for (j, (i, k)) in who.into_iter().enumerate() {
            let mut p = 1.0f32;
            let keep = conf[j * kmax..j * kmax + k]
                .iter()
                .take_while(|&&c| {
                    p *= c;
                    p >= self.draft_conf
                })
                .count();
            out[i] = toks[j * kmax..j * kmax + keep].iter().map(|&t| t as u32).collect();
            let rec = &mut self.draft_rec[reqs[i].seq];
            rec.pending = keep > 0;
            if keep == 0 && self.draft_conf > 0.0 {
                rec.sit_out();
            }
        }
        Ok(out)
    }

    crate::delegate_paged!(paged, own_draft);
}
