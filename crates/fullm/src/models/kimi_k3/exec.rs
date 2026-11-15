// SPDX-License-Identifier: Apache-2.0
//! Kimi K3 on the C++ executor (CPU or CUDA backend): the production path.
//!
//! The control plane lives here: which slot and which cache pages each sequence owns,
//! which slots start fresh, and the flat StepPlan the executor runs. The executor owns the
//! physical pools and every byte of arithmetic.
//!
//! Weights are streamed in one layer at a time (bind, convert, copy, drop), so host memory
//! never holds more than one layer of the trunk. The embedding table stays in host memory
//! (owned here), and routed experts stay where the expert store maps them.

use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, bail, ensure, Context, Result};
use fullm_kernels::exec::{self, Executor, Fmt, K3Dims};
use fullm_kernels::WeightRef;

use super::experts::{ExpertFmt, ExpertStore, MmapExperts};
use super::weights::{bind_globals, bind_layer, Attn, Ffn, KdaGate, MlaQuery};
use super::K3Config;
use crate::models::paged::{Geometry, Paged};
use crate::models::{BackendKind, LoadOptions, Model};
use crate::st::SafeTensors;
use crate::tensor::{Mat, MatData, Precision};

pub struct K3Exec {
    cfg: K3Config,
    // drops first: the executor reads the embedding and the experts below in place
    paged: Paged,
    backend: BackendKind,
    _embed: Mat,
    _experts: Arc<MmapExperts>,
    _st: Arc<SafeTensors>,
    desc: String,
}

fn dims(c: &K3Config) -> K3Dims {
    let i = |v: usize| v as i32;
    K3Dims {
        hidden: i(c.hidden),
        n_layers: i(c.n_layers),
        vocab: i(c.vocab),
        rms_eps: c.rms_eps,
        kda_heads: i(c.kda_heads),
        kda_head_dim: i(c.kda_head_dim),
        conv_k: i(c.conv_k),
        gate_lb: c.gate_lb,
        n_heads: i(c.n_heads),
        q_lora: i(c.q_lora),
        kv_lora: i(c.kv_lora),
        qk_nope: i(c.qk_nope),
        qk_rope: i(c.qk_rope),
        v_head: i(c.v_head),
        mla_out_gate: c.mla_out_gate,
        n_experts: i(c.n_experts),
        topk: i(c.topk),
        n_shared: i(c.n_shared),
        latent: i(c.latent),
        moe_inter: i(c.moe_inter),
        routed_scale: c.routed_scale,
        moe_renorm: c.moe_renorm,
        latent_norm: c.latent_norm,
        first_dense: i(c.first_dense),
        dense_inter: i(c.dense_inter),
        attn_res_block: i(c.attn_res_block),
        situ_b1: c.situ_b1,
        situ_b2: c.situ_b2,
        attn_res: c.attn_res,
        decay_softplus: c.decay_softplus,
        kda_gate_lowrank: c.kda_gate_lowrank,
        latent_moe: c.latent_moe,
        act_silu: c.act_silu,
        expert_bf16: c.expert_bf16,
        router_biased: c.router_weights_biased,
        kv_norm_eps: c.kv_norm_eps,
        layer_is_mla: (0..c.n_layers).map(|l| c.is_mla(l) as u8).collect(),
    }
}

impl K3Exec {
    pub fn load(dir: &Path, config: &serde_json::Value, opts: &LoadOptions) -> Result<Self> {
        let mut cfg = K3Config::from_json(config)?;
        if let Some(n) = opts.layers {
            cfg.truncate(n);
        }
        cfg.router_weights_biased = opts.router_weights_biased;
        let geo = Geometry::new(opts)?;
        let st = SafeTensors::open_dir(dir)?;
        let experts = Arc::new(MmapExperts::new(st.clone(), &cfg)?);
        let mut ex =
            Executor::new_k3(&dims(&cfg), &geo.exec_opts(opts)).map_err(|e| anyhow!("creating the executor: {e}"))?;

        let t0 = std::time::Instant::now();
        let g = bind_globals(&st, &cfg)?;
        let (fmt, bytes, _) = g.embed.raw();
        // SAFETY: the embedding Mat is moved into Self and outlives the executor (field order:
        // `ex` drops first).
        unsafe { ex.add_embedding(fmt, g.embed.rows, g.embed.cols, bytes.as_ptr()) }.map_err(|e| anyhow!(e))?;
        add_mat(&mut ex, "lm_head", &g.lm_head)?;
        add_vec(&mut ex, "final_norm", &g.final_norm)?;
        if let Some(f) = &g.fold_out {
            add_vec(&mut ex, "fold_out", f)?;
        }

        for l in 0..cfg.n_layers {
            let lw = bind_layer(&st, &cfg, l, opts.trunk).with_context(|| format!("binding layer {l}"))?;
            let p = format!("L{l}.");
            add_vec(&mut ex, &format!("{p}in_norm"), &lw.in_norm)?;
            add_vec(&mut ex, &format!("{p}post_norm"), &lw.post_norm)?;
            if let Some(f) = &lw.folds {
                add_vec(&mut ex, &format!("{p}fold_attn"), &f.attn)?;
                add_vec(&mut ex, &format!("{p}fold_mlp"), &f.mlp)?;
            }
            match &lw.attn {
                Attn::Kda(w) => {
                    for (n, m) in [
                        ("q", &w.q),
                        ("k", &w.k),
                        ("v", &w.v),
                        ("o", &w.o),
                        ("f_a", &w.f_a),
                        ("f_b", &w.f_b),
                        ("b", &w.b),
                    ] {
                        add_mat(&mut ex, &format!("{p}{n}"), m)?;
                    }
                    match &w.gate {
                        KdaGate::Full(g) => add_mat(&mut ex, &format!("{p}g"), g)?,
                        KdaGate::LowRank { a, b } => {
                            add_mat(&mut ex, &format!("{p}g_a"), a)?;
                            add_mat(&mut ex, &format!("{p}g_b"), b)?;
                        }
                    }
                    for (n, v) in [
                        ("q_conv", &w.q_conv),
                        ("k_conv", &w.k_conv),
                        ("v_conv", &w.v_conv),
                        ("a_log", &w.a_log),
                        ("dt_bias", &w.dt_bias),
                        ("o_norm", &w.o_norm),
                    ] {
                        add_vec(&mut ex, &format!("{p}{n}"), v)?;
                    }
                }
                Attn::Mla(w) => {
                    for (n, m) in [("kv_a", &w.kv_a), ("kv_b", &w.kv_b), ("o", &w.o)] {
                        add_mat(&mut ex, &format!("{p}{n}"), m)?;
                    }
                    match &w.q {
                        MlaQuery::Lora { a, norm, b } => {
                            add_mat(&mut ex, &format!("{p}q_a"), a)?;
                            add_mat(&mut ex, &format!("{p}q_b"), b)?;
                            add_vec(&mut ex, &format!("{p}q_a_norm"), norm)?;
                        }
                        MlaQuery::Direct(q) => add_mat(&mut ex, &format!("{p}q"), q)?,
                    }
                    if let Some(gm) = &w.g {
                        add_mat(&mut ex, &format!("{p}g"), gm)?;
                    }
                    add_vec(&mut ex, &format!("{p}kv_a_norm"), &w.kv_a_norm)?;
                }
            }
            match &lw.ffn {
                Ffn::Dense(w) => {
                    for (n, m) in [("d_gate", &w.gate), ("d_up", &w.up), ("d_down", &w.down)] {
                        add_mat(&mut ex, &format!("{p}{n}"), m)?;
                    }
                }
                Ffn::Moe(w) => {
                    for (n, m) in
                        [("router", &w.gate), ("sh_gate", &w.sh_gate), ("sh_up", &w.sh_up), ("sh_down", &w.sh_down)]
                    {
                        add_mat(&mut ex, &format!("{p}{n}"), m)?;
                    }
                    add_vec(&mut ex, &format!("{p}router_bias"), &w.bias)?;
                    if let Some(lt) = &w.latent {
                        add_mat(&mut ex, &format!("{p}down"), &lt.down)?;
                        add_mat(&mut ex, &format!("{p}up"), &lt.up)?;
                        if let Some(norm) = &lt.norm {
                            add_vec(&mut ex, &format!("{p}latent_norm"), norm)?;
                        }
                    }
                    for e in 0..cfg.n_experts {
                        let ptrs = experts.get(l, e)?.raw_ptrs();
                        // SAFETY: the expert store (and its mapping) is held by Self.
                        unsafe { ex.set_expert(l, e, ptrs) }.map_err(|e| anyhow!(e))?;
                    }
                }
            }
            if l % 8 == 0 || l + 1 == cfg.n_layers {
                tracing::info!("layer {}/{} loaded ({:.1}s)", l + 1, cfg.n_layers, t0.elapsed().as_secs_f64());
            }
        }
        let placed = place_gpu_experts(&mut ex, &cfg, experts.as_ref(), opts)?;
        if placed > 0 {
            tracing::info!(
                "{placed} of {} routed experts resident in {:?} memory ({:?}) ({:.1}s)",
                cfg.n_moe() * cfg.n_experts,
                opts.backend,
                opts.gpu_expert_format,
                t0.elapsed().as_secs_f64()
            );
        }
        ex.finalize().map_err(|e| anyhow!("finalizing the executor: {e}"))?;
        let desc = format!("{} | experts {} | trunk {:?}", ex.describe(), experts.describe(), opts.trunk);

        Ok(K3Exec {
            // no prefix cache: a sequence's KDA state is not in its pages
            paged: Paged::new(ex, cfg.vocab, &geo, opts, false)?,
            backend: opts.backend.clone(),
            _embed: g.embed,
            _experts: experts,
            _st: st,
            desc,
            cfg,
        })
    }
}

/// The HBM expert tier: copy as many routed experts as the budget allows into backend memory,
/// spread evenly over the MoE layers. Only bf16-expert checkpoints qualify (MXFP4 experts
/// would have to be widened, and for Kimi K3 the HBM holds ~1% of them anyway). Returns how
/// many experts were placed.
fn place_gpu_experts(ex: &mut Executor, c: &K3Config, store: &MmapExperts, opts: &LoadOptions) -> Result<usize> {
    use crate::models::GpuExperts;
    use rayon::prelude::*;
    if opts.gpu_experts == GpuExperts::Off {
        return Ok(0);
    }
    if store.format() != ExpertFmt::Bf16 {
        if let GpuExperts::Gb(_) = opts.gpu_experts {
            tracing::warn!(
                "--gpu-experts needs bf16 experts; this checkpoint's are {:?}, keeping them on the CPU",
                store.format()
            );
        }
        return Ok(0);
    }
    let (w, i) = (c.latent, c.moe_inter);
    let b = |n: usize| n.div_ceil(128);
    let (fmt, per) = match opts.gpu_expert_format {
        Precision::Bf16 => (Fmt::Bf16, 3 * w * i * 2),
        Precision::Int8 => (Fmt::Int8, 3 * w * i + (2 * i + w) * 4),
        Precision::Fp8 => (Fmt::Fp8, 3 * w * i + (2 * i + w) * 4),
        // blockwise e4m3 (prefill on the CUTLASS blockwise GEMM) and int4 groups: given as bf16,
        // quantised by the executor (block / group scales as the dense executor counts them)
        Precision::Fp8Block => (Fmt::Fp8Block, 3 * w * i + (2 * b(i) * b(w) + b(w) * b(i)) * 4),
        Precision::Int4Group => (Fmt::Int4Group, 3 * w * i / 2 + (2 * i * b(w) + w * b(i)) * 4),
        Precision::F32 => bail!("resident experts are bf16, int8, fp8, fp8b or int4"),
    };
    let per = per + 6 * 256; // the arena aligns every matrix and scale vector
    let budget = match opts.gpu_experts {
        GpuExperts::Gb(gb) => (gb * 1e9) as usize,
        // "auto": a device with memory of its own holds what fits
        GpuExperts::Auto
            if exec::backend_caps(opts.backend.exec_name(), opts.device, opts.bf16_activations)
                .is_ok_and(|c| c.memory == exec::MemKind::Discrete) =>
        {
            let free = exec::device_info(opts.device).map(|d| d.free_bytes).unwrap_or(0);
            free.saturating_sub(ex.pool_bytes()).saturating_sub(2 << 30)
        }
        _ => return Ok(0), // "auto" means GPU memory; a CPU run opts in with an explicit size
    };
    let moe_layers: Vec<usize> = (0..c.n_layers).filter(|&l| !c.is_dense(l)).collect();
    let n_fit = (budget / per).min(moe_layers.len() * c.n_experts);
    if n_fit == 0 {
        return Ok(0);
    }
    ex.reserve_gpu_experts(fmt, n_fit * per).map_err(|e| anyhow!(e))?;

    // One expert's three matrices in the resident format.
    struct Prepared<'a> {
        w: [std::borrow::Cow<'a, [u8]>; 3],
        s: [Option<Vec<f32>>; 3],
    }
    fn bf16_bytes(r: &WeightRef<'_>) -> Result<&'static [u16]> {
        match *r {
            // SAFETY of the 'static: the slice points into the store's mapping, which the caller
            // keeps alive for the executor's lifetime; it is only read during this function.
            WeightRef::Bf16(d) => Ok(unsafe { std::slice::from_raw_parts(d.as_ptr(), d.len()) }),
            _ => bail!("expected bf16 expert weights"),
        }
    }
    let prepare = |l: usize, e: usize| -> Result<Prepared<'static>> {
        let r = store.get(l, e)?;
        let mats = [(bf16_bytes(&r.w1)?, i, w), (bf16_bytes(&r.w3)?, i, w), (bf16_bytes(&r.w2)?, w, i)];
        let mut out_w: Vec<std::borrow::Cow<'static, [u8]>> = Vec::with_capacity(3);
        let mut out_s = Vec::with_capacity(3);
        for (data, rows, cols) in mats {
            match fmt {
                Fmt::Bf16 | Fmt::Fp8Block | Fmt::Int4Group => {
                    // SAFETY: plain u16 data viewed as bytes for the copy into the executor.
                    let b = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
                    out_w.push(std::borrow::Cow::Borrowed(b));
                    out_s.push(None);
                }
                _ => {
                    let mut q = vec![0u8; rows * cols];
                    let mut sc = vec![0f32; rows];
                    if fmt == Fmt::Int8 {
                        // SAFETY: i8 and u8 have the same layout.
                        let qi = unsafe { std::slice::from_raw_parts_mut(q.as_mut_ptr() as *mut i8, q.len()) };
                        fullm_kernels::quant_i8_rows(data, cols, 0, rows, qi, &mut sc);
                    } else {
                        fullm_kernels::quant_fp8_rows(data, cols, 0, rows, &mut q, &mut sc);
                    }
                    out_w.push(std::borrow::Cow::Owned(q));
                    out_s.push(Some(sc));
                }
            }
        }
        let mut wi = out_w.into_iter();
        let mut si = out_s.into_iter();
        Ok(Prepared {
            w: [wi.next().unwrap(), wi.next().unwrap(), wi.next().unwrap()],
            s: [si.next().unwrap(), si.next().unwrap(), si.next().unwrap()],
        })
    };

    let (q, r) = (n_fit / moe_layers.len(), n_fit % moe_layers.len());
    for (li, &l) in moe_layers.iter().enumerate() {
        let ids: Vec<usize> = (0..q + usize::from(li < r)).collect();
        for chunk in ids.chunks(32) {
            let prepared = chunk.par_iter().map(|&e| prepare(l, e)).collect::<Result<Vec<_>>>()?;
            for (&e, p) in chunk.iter().zip(&prepared) {
                // SAFETY: prepare() built each matrix at the reserved format and the model's shapes
                unsafe {
                    ex.set_gpu_expert(
                        l,
                        e,
                        &p.w[0],
                        p.s[0].as_deref(),
                        &p.w[1],
                        p.s[1].as_deref(),
                        &p.w[2],
                        p.s[2].as_deref(),
                    )
                }
                .map_err(|err| anyhow!(err))?;
            }
        }
    }
    Ok(n_fit)
}

fn add_mat(ex: &mut Executor, name: &str, m: &Mat) -> Result<()> {
    let (fmt, bytes, scale) = m.raw();
    ensure!(!matches!(m.data, MatData::F32(_)) || fmt == Fmt::F32);
    ex.add_matrix(name, fmt, m.rows, m.cols, bytes, scale).map_err(|e| anyhow!(e))
}

fn add_vec(ex: &mut Executor, name: &str, v: &[f32]) -> Result<()> {
    ex.add_vector(name, v).map_err(|e| anyhow!(e))
}

impl Model for K3Exec {
    fn arch(&self) -> &'static str {
        match self.cfg.family {
            super::config::Family::K3 => "kimi_k3",
            super::config::Family::Linear => "kimi_linear",
        }
    }

    fn vocab(&self) -> usize {
        self.cfg.vocab
    }

    fn describe(&self) -> String {
        format!("{} [{:?} backend]", self.desc, self.backend)
    }

    crate::delegate_paged!(paged);
}
