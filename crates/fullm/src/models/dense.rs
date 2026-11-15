// SPDX-License-Identifier: Apache-2.0
//! Dense-attention decoders: the Llama / Qwen3 family (pre-norm, grouped-query attention with
//! RoPE, Qwen3's per-head q/k RMSNorm, SwiGLU MLP) and Qwen3-MoE (the MLP a softmax-routed
//! set of SwiGLU experts, all resident in backend memory), on the C++ executor
//! (`dense_exec.cpp`).
//!
//! Supported `model_type`s: `qwen3`, `qwen3_moe`, `llama`. Attention or MLP biases (Qwen2),
//! mixed dense/MoE layer stacks and RoPE scalings other than none and `llama3` are refused
//! rather than run wrong.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, bail, ensure, Context, Result};
use fullm_kernels::exec::{DenseDims, Executor};
use serde_json::Value;

use crate::models::lora::Adapter;
use crate::models::paged::{Geometry, Paged};
use crate::models::{BackendKind, LoadOptions, Model};
use crate::st::SafeTensors;
use crate::tensor::{Mat, MatData, Precision};

#[derive(Clone, Debug)]
pub struct DenseConfig {
    pub model_type: String,
    pub n_layers: usize,
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub qk_norm: bool,
    pub tie_word_embeddings: bool,
    /// `[head_dim / 2]`, scaling applied, computed in f32 as transformers does.
    pub inv_freq: Vec<f32>,
    /// Qwen3-MoE: experts per layer (0 = dense), experts per token, expert width.
    pub n_experts: usize,
    pub topk: usize,
    pub moe_inter: usize,
    pub norm_topk: bool,
}

impl DenseConfig {
    pub fn from_json(v: &Value) -> Result<Self> {
        let mt = v.get("model_type").and_then(Value::as_str).unwrap_or("").to_string();
        ensure!(mt == "qwen3" || mt == "qwen3_moe" || mt == "llama", "dense decoder: unsupported model_type {mt:?}");
        let u = |k: &str| -> Result<usize> {
            v.get(k).and_then(Value::as_u64).map(|x| x as usize).with_context(|| format!("config: missing {k}"))
        };
        let f = |k: &str, d: f64| v.get(k).and_then(Value::as_f64).unwrap_or(d);
        let b = |k: &str| v.get(k).and_then(Value::as_bool).unwrap_or(false);
        ensure!(!b("attention_bias") && !b("mlp_bias"), "dense decoder: attention / MLP biases are not supported");
        ensure!(!b("use_sliding_window"), "dense decoder: sliding-window attention is not supported");
        let act = v.get("hidden_act").and_then(Value::as_str).unwrap_or("silu");
        ensure!(act == "silu", "dense decoder: activation {act:?} (only silu)");
        let hidden = u("hidden_size")?;
        let n_heads = u("num_attention_heads")?;
        let head_dim = v.get("head_dim").and_then(Value::as_u64).map(|x| x as usize).unwrap_or(hidden / n_heads);
        ensure!(head_dim % 2 == 0, "head_dim must be even");
        // transformers 5 writes rope_parameters {rope_theta, rope_type, ...} instead
        let rp = v.get("rope_parameters").filter(|r| r.is_object());
        let theta =
            rp.and_then(|r| r.get("rope_theta")).and_then(Value::as_f64).unwrap_or_else(|| f("rope_theta", 10000.0))
                as f32;
        let inv_freq = rope_inv_freq(head_dim, theta, v.get("rope_scaling").filter(|r| !r.is_null()).or(rp))?;
        let moe = mt == "qwen3_moe";
        if moe {
            let step = v.get("decoder_sparse_step").and_then(Value::as_u64).unwrap_or(1);
            let dense_layers = v.get("mlp_only_layers").and_then(Value::as_array).map_or(0, |a| a.len());
            ensure!(step == 1 && dense_layers == 0, "qwen3_moe: mixed dense/MoE layer stacks are not supported");
        }
        Ok(DenseConfig {
            n_experts: if moe { u("num_experts").or_else(|_| u("num_local_experts"))? } else { 0 },
            topk: if moe { u("num_experts_per_tok")? } else { 0 },
            moe_inter: if moe { u("moe_intermediate_size")? } else { 0 },
            norm_topk: moe && b("norm_topk_prob"),
            n_layers: u("num_hidden_layers")?,
            hidden,
            n_heads,
            n_kv_heads: v.get("num_key_value_heads").and_then(Value::as_u64).map(|x| x as usize).unwrap_or(n_heads),
            head_dim,
            inter: u("intermediate_size")?,
            vocab: u("vocab_size")?,
            rms_eps: f("rms_norm_eps", 1e-6) as f32,
            qk_norm: mt == "qwen3" || mt == "qwen3_moe",
            tie_word_embeddings: b("tie_word_embeddings"),
            inv_freq,
            model_type: mt,
        })
    }
}

/// transformers' `_compute_default_rope_parameters` and `_compute_llama3_parameters`, in f32.
pub(crate) fn rope_inv_freq(dim: usize, theta: f32, scaling: Option<&Value>) -> Result<Vec<f32>> {
    let mut inv: Vec<f32> = (0..dim / 2).map(|i| 1.0 / theta.powf((2 * i) as f32 / dim as f32)).collect();
    let Some(s) = scaling.filter(|s| !s.is_null()) else { return Ok(inv) };
    let kind = s.get("rope_type").or_else(|| s.get("type")).and_then(Value::as_str).unwrap_or("default");
    match kind {
        "default" => Ok(inv),
        "llama3" => {
            let g = |k: &str| {
                s.get(k).and_then(Value::as_f64).map(|x| x as f32).with_context(|| format!("rope_scaling.{k}"))
            };
            let (factor, lo, hi, old) =
                (g("factor")?, g("low_freq_factor")?, g("high_freq_factor")?, g("original_max_position_embeddings")?);
            let (lo_wl, hi_wl) = (old / lo, old / hi);
            for f in inv.iter_mut() {
                let wl = 2.0 * std::f32::consts::PI / *f;
                let scaled = if wl > lo_wl { *f / factor } else { *f };
                let smooth = (old / wl - lo) / (hi - lo);
                let medium = !(wl < hi_wl) && !(wl > lo_wl);
                *f = if medium { (1.0 - smooth) * scaled / factor + smooth * scaled } else { scaled };
            }
            Ok(inv)
        }
        other => bail!("dense decoder: RoPE scaling {other:?} is not supported"),
    }
}

pub struct DenseModel {
    cfg: DenseConfig,
    // drops first: the executor reads the embedding below in place
    paged: Paged,
    backend: BackendKind,
    _embed: Mat,
    _st: Arc<SafeTensors>,
    desc: String,
}

impl DenseModel {
    pub fn load(dir: &Path, config: &Value, opts: &LoadOptions) -> Result<Self> {
        let mut cfg = DenseConfig::from_json(config)?;
        if let Some(n) = opts.layers {
            cfg.n_layers = cfg.n_layers.min(n);
        }
        ensure!(
            opts.backend != BackendKind::Reference,
            "dense decoders run on the executor: use --backend cpu or cuda"
        );
        let geo = Geometry::new(opts)?;
        let st = SafeTensors::open_dir(dir)?;
        let adapter = match &opts.lora_merge {
            Some(p) => Some(Adapter::load(p).with_context(|| format!("the LoRA adapter {}", p.display()))?),
            None => None,
        };
        let lora = adapter.as_ref();
        // a dense model with --gpu-expert-format int4: its MLP split along the intermediate dimension
        // into S experts that every token takes at weight 1, no router (the MLP is the sum of its
        // slices' down projections), so it runs the int4 experts' kernels (decode tiles, prefill GEMM,
        // the quantised-expert cache) with S experts' worth of parallel work; the attention stays at
        // --trunk. S: 8 (FULLM_DENSE_MLP_SPLIT), halved until the slices are whole 256-column groups:
        // the CUDA int4 kernels' unit (128-byte rows of nibbles; tc_moe_ok, int4_gemm::shape_ok), below
        // which a step has no int4 or fp8b path there (Qwen3-0.6B's 3072: 4 slices of 768; Qwen3-4B's
        // 9728: 2 of 4864).
        let mlp_expert = cfg.n_experts == 0 && opts.gpu_expert_format == Precision::Int4Group;
        let gptq = match &opts.int4_checkpoint {
            Some(d) => {
                ensure!(
                    opts.gpu_expert_format == Precision::Int4Group,
                    "--int4-checkpoint goes with --gpu-expert-format int4"
                );
                ensure!(cfg.n_experts > 0, "--int4-checkpoint: for routed experts only (this model is dense)");
                Some(Gptq::open(d)?)
            }
            None => None,
        };
        if mlp_expert {
            let mut s =
                std::env::var("FULLM_DENSE_MLP_SPLIT").ok().and_then(|v| v.parse().ok()).unwrap_or(8usize).max(1);
            while s > 1 && cfg.inter % (s * 256) != 0 {
                s /= 2;
            }
            (cfg.n_experts, cfg.topk, cfg.moe_inter, cfg.norm_topk) = (s, s, cfg.inter / s, false);
        }
        let i = |x: usize| x as i32;
        let dims = DenseDims {
            n_layers: i(cfg.n_layers),
            hidden: i(cfg.hidden),
            n_heads: i(cfg.n_heads),
            n_kv_heads: i(cfg.n_kv_heads),
            head_dim: i(cfg.head_dim),
            inter: i(cfg.inter),
            vocab: i(cfg.vocab),
            rms_eps: cfg.rms_eps,
            qk_norm: cfg.qk_norm,
            inv_freq: cfg.inv_freq.clone(),
            n_experts: i(cfg.n_experts),
            topk: i(cfg.topk),
            moe_inter: i(cfg.moe_inter),
            norm_topk: cfg.norm_topk,
        };
        ensure!(
            opts.trunk != Precision::Fp8Block && opts.trunk != Precision::Int4Group,
            "--trunk fp8b / int4: for the resident experts only (--gpu-expert-format)"
        );
        let mut ex =
            Executor::new_dense(&dims, &geo.exec_opts(opts)).map_err(|e| anyhow!("creating the executor: {e}"))?;

        let (e, q, kv) = (cfg.hidden, cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim);
        let embed = mat(&st, lora, "model.embed_tokens.weight", cfg.vocab, e, Precision::Bf16)?;
        let (fmt, bytes, _) = embed.raw();
        // SAFETY: the embedding Mat is moved into Self and outlives the executor (`paged` drops first).
        unsafe { ex.add_embedding(fmt, embed.rows, embed.cols, bytes.as_ptr()) }.map_err(|e| anyhow!(e))?;
        let head = if cfg.tie_word_embeddings { "model.embed_tokens.weight" } else { "lm_head.weight" };
        // tied: the head's own delta (an adapter may train the two apart), on its own copy
        let head_lora = if cfg.tie_word_embeddings { lora.filter(|a| a.has("lm_head.weight")) } else { lora };
        let head_p = match opts.head {
            Some(p) => {
                ensure!(matches!(p, Precision::Bf16 | Precision::Int8 | Precision::Fp8), "--head: bf16, int8 or fp8");
                p
            }
            None => Precision::Bf16,
        };
        let head_mat = match head_lora {
            Some(a) if cfg.tie_word_embeddings => {
                let mut m = mat(&st, None, head, cfg.vocab, e, Precision::Bf16)?;
                if let MatData::Bf16(w) = &mut m.data {
                    a.merge_bf16("lm_head.weight", w, cfg.vocab, e)?;
                }
                m.convert(head_p)
            }
            _ => mat(&st, lora, head, cfg.vocab, e, head_p)?,
        };
        add_mat(&mut ex, "lm_head", &head_mat)?;
        add_vec(&mut ex, "final_norm", &vector(&st, "model.norm.weight", e)?)?;
        // replay: the experts come from the quantised-expert cache (FULLM_QCACHE), not the checkpoint
        let mut replay = false;
        if cfg.n_experts > 0 {
            reserve_experts(&mut ex, &cfg, opts)?;
            if let Some(path) = expert_cache_path(
                dir,
                &cfg,
                opts.gpu_expert_format,
                opts.int4_checkpoint.as_deref(),
                opts.lora_merge.as_deref(),
            ) {
                replay = ex.expert_cache(&path).map_err(|e| anyhow!(e))? == Some(true);
                tracing::info!("expert cache {}: {}", path.display(), if replay { "replaying" } else { "recording" });
            }
        }
        for l in 0..cfg.n_layers {
            let (src, dst) = (format!("model.layers.{l}."), format!("L{l}."));
            add_vec(&mut ex, &format!("{dst}in_norm"), &vector(&st, &format!("{src}input_layernorm.weight"), e)?)?;
            add_vec(
                &mut ex,
                &format!("{dst}post_norm"),
                &vector(&st, &format!("{src}post_attention_layernorm.weight"), e)?,
            )?;
            if cfg.qk_norm {
                for n in ["q_norm", "k_norm"] {
                    add_vec(
                        &mut ex,
                        &format!("{dst}{n}"),
                        &vector(&st, &format!("{src}self_attn.{n}.weight"), cfg.head_dim)?,
                    )?;
                }
            }
            let attn = [
                ("q", "self_attn.q_proj", q, e),
                ("k", "self_attn.k_proj", kv, e),
                ("v", "self_attn.v_proj", kv, e),
                ("o", "self_attn.o_proj", e, q),
            ];
            let mlp = [
                ("gate", "mlp.gate_proj", cfg.inter, e),
                ("up", "mlp.up_proj", cfg.inter, e),
                ("down", "mlp.down_proj", e, cfg.inter),
            ];
            let dense: &[(&str, &str, usize, usize)] = if cfg.n_experts > 0 { &[] } else { &mlp };
            for &(n, name, rows, cols) in attn.iter().chain(dense) {
                let m = mat(&st, lora, &format!("{src}{name}.weight"), rows, cols, opts.trunk)?;
                add_mat(&mut ex, &format!("{dst}{n}"), &m)?;
            }
            if cfg.n_experts > 0 {
                // the router stays bf16, as vLLM keeps it (none for a split dense MLP)
                if !mlp_expert {
                    let r = mat(&st, lora, &format!("{src}mlp.gate.weight"), cfg.n_experts, e, Precision::Bf16)?;
                    add_mat(&mut ex, &format!("{dst}router"), &r)?;
                }
                if replay {
                    // the cache was recorded with the adapter merged (its name is in the key)
                    if let Some(a) = lora {
                        a.mark_used(|w| {
                            w.starts_with(&format!("{src}mlp.")) && w.contains(".experts.")
                                || (mlp_expert && w.starts_with(&format!("{src}mlp.")))
                        });
                    }
                    for x in 0..cfg.n_experts {
                        // SAFETY: the cache holds these experts (replay), so the empty slices are not read
                        unsafe { ex.set_gpu_expert(l, x, &[], None, &[], None, &[], None) }.map_err(|e| anyhow!(e))?;
                    }
                } else {
                    load_experts(&mut ex, &st, &cfg, l, opts.gpu_expert_format, mlp_expert, gptq.as_ref(), lora)?;
                }
            }
            if l % 8 == 0 || l + 1 == cfg.n_layers {
                tracing::info!("layer {}/{} loaded", l + 1, cfg.n_layers);
            }
        }
        // LoRA adapters served unmerged: q/k/v/o, and a dense MLP's gate/up/down (routed experts
        // take a merged adapter only)
        let lora_slots = if opts.max_loras > 0 {
            let targets = if cfg.n_experts == 0 { 0x7f } else { 0x0f };
            let rank = if opts.max_lora_rank > 0 { opts.max_lora_rank } else { 64 };
            ex.reserve_lora(opts.max_loras, rank, targets).map_err(|e| anyhow!(e))?;
            opts.max_loras
        } else {
            0
        };
        let head = match &opts.spec_model {
            Some(d) => Some(super::eagle::attach(&mut ex, d, &cfg)?),
            None => None,
        };
        if let Some(a) = lora {
            a.check_all_used()?;
            tracing::info!("LoRA adapter {} merged into the weights", a.dir.display());
        }
        ex.finalize().map_err(|e| anyhow!("finalizing the executor: {e}"))?;
        let mut desc = format!("{} | {} | trunk {:?}", ex.describe(), cfg.model_type, opts.trunk);
        if let Some(h) = &head {
            desc += &format!(" | {h}");
        }
        let mut paged = Paged::new(ex, cfg.vocab, &geo, opts, true)?;
        paged.set_draft_head(head.is_some());
        paged.set_lora_slots(lora_slots);
        Ok(DenseModel { paged, backend: opts.backend.clone(), _embed: embed, _st: st, desc, cfg })
    }
}

/// The arena for every routed expert in `--gpu-expert-format` (bf16, int8, fp8, fp8b or int4).
fn reserve_experts(ex: &mut Executor, c: &DenseConfig, opts: &LoadOptions) -> Result<()> {
    use fullm_kernels::exec::Fmt;
    let (w, i) = (c.hidden, c.moe_inter);
    let b = |n: usize| n.div_ceil(128);
    let (fmt, per) = match opts.gpu_expert_format {
        Precision::Bf16 => (Fmt::Bf16, 3 * w * i * 2),
        Precision::Int8 => (Fmt::Int8, 3 * w * i + (2 * i + w) * 4),
        Precision::Fp8 => (Fmt::Fp8, 3 * w * i + (2 * i + w) * 4),
        // block scales: gate and up (separate or interleaved, at most 2 b(i) row blocks), down
        Precision::Fp8Block => (Fmt::Fp8Block, 3 * w * i + (2 * b(i) * b(w) + b(w) * b(i)) * 4),
        // two values per byte; a scale per row per 128 columns: gate and up (2 i rows of w), down
        // (the executor adds the fp8b copies for prefill when the backend has no int4 GEMM)
        Precision::Int4Group => (Fmt::Int4Group, 3 * w * i / 2 + (2 * i * b(w) + w * b(i)) * 4),
        Precision::F32 => bail!("resident experts are bf16, int8, fp8, fp8b or int4"),
    };
    let per = per + 6 * 256; // the arena aligns every matrix and scale vector
    ex.reserve_gpu_experts(fmt, c.n_layers * c.n_experts * per).map_err(|e| anyhow!(e))
}

/// The quantised-expert cache file for this model and `--gpu-expert-format` when FULLM_QCACHE
/// names a directory and the executor quantises the experts (fp8b, int4): named after the
/// checkpoint's files (names, sizes, modification times), the format, the layer count and the
/// options that change what is stored; the executor checks the dims and its own format version.
fn expert_cache_path(
    dir: &Path,
    c: &DenseConfig,
    p: Precision,
    gptq: Option<&Path>,
    lora: Option<&Path>,
) -> Option<PathBuf> {
    let root = std::env::var_os("FULLM_QCACHE")?;
    if !matches!(p, Precision::Fp8Block | Precision::Int4Group) {
        return None;
    }
    // the fp8b copies come with the int4 experts unless the int4 GEMM runs prefill: both switches
    let mut key = format!(
        "{}|{p:?}|{}|{}|{}",
        dir.canonicalize().ok()?.display(),
        c.n_layers,
        std::env::var("FULLM_INT4_PREFILL_FP8").unwrap_or_default(),
        std::env::var("FULLM_NO_INT4_GEMM").unwrap_or_default()
    );
    // the int4 scales' rule (int4_group.h): every key names it, so a cache from the other rule is not replayed
    let clip = std::env::var("FULLM_INT4_CLIP").is_ok_and(|v| v == "1");
    key += &match gptq {
        Some(g) => format!("|gptq {}", g.canonicalize().ok()?.display()),
        None if clip => "|clip".into(),
        None => "|absmax".into(),
    };
    if let Some(l) = lora {
        key += &format!("|lora {}", l.canonicalize().ok()?.display());
    }
    for d in std::iter::once(dir).chain(gptq).chain(lora) {
        let mut files: Vec<PathBuf> = std::fs::read_dir(d)
            .ok()?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|f| f.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        for f in files {
            let m = std::fs::metadata(&f).ok()?;
            key += &format!("|{}:{}:{:?}", f.file_name()?.to_string_lossy(), m.len(), m.modified().ok());
        }
    }
    // 64-bit FNV-1a
    let h = key.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3));
    std::fs::create_dir_all(&root).ok()?;
    Some(PathBuf::from(root).join(format!("experts-{}-{h:016x}.bin", format!("{p:?}").to_lowercase())))
}

/// An adapter weight's (layer, executor target): q k v o gate up down -> 0..6.
fn lora_target(name: &str) -> Option<(usize, usize)> {
    let rest = name.strip_prefix("model.layers.")?;
    let (l, m) = rest.split_once('.')?;
    let t = match m {
        "self_attn.q_proj.weight" => 0,
        "self_attn.k_proj.weight" => 1,
        "self_attn.v_proj.weight" => 2,
        "self_attn.o_proj.weight" => 3,
        "mlp.gate_proj.weight" => 4,
        "mlp.up_proj.weight" => 5,
        "mlp.down_proj.weight" => 6,
        _ => return None,
    };
    Some((l.parse().ok()?, t))
}

fn bf16_bits(x: f32) -> u16 {
    let u = x.to_bits();
    ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16
}

/// A GPTQ checkpoint's int4 experts (`--int4-checkpoint`): 4 bits, symmetric, groups of 128
/// along the input, no act-order, which are INT4G's own values and scales (W = q x scale, q in
/// [-8, 7], a scale per output row per 128 inputs), so they go to the executor as they are.
struct Gptq {
    st: Arc<SafeTensors>,
    /// Every group's stored zero point (symmetric: all alike): 7 in GPTQ's v1 format (it stores
    /// zero - 1), 8 in v2.
    zero_stored: u32,
}

impl Gptq {
    fn open(dir: &Path) -> Result<Self> {
        let read =
            |f: &str| -> Option<Value> { serde_json::from_str(&std::fs::read_to_string(dir.join(f)).ok()?).ok() };
        let qc = read("config.json")
            .and_then(|c| c.get("quantization_config").cloned())
            .or_else(|| read("quantize_config.json"))
            .with_context(|| {
                format!("{}: no quantization_config (config.json) or quantize_config.json", dir.display())
            })?;
        let method = qc["quant_method"].as_str().unwrap_or("gptq");
        ensure!(method == "gptq", "{}: quant_method {method:?}, not gptq", dir.display());
        ensure!(
            qc["bits"] == 4 && qc["group_size"] == 128 && qc["sym"] == true && qc["desc_act"] != true,
            "{}: fullm's int4 experts take GPTQ with bits 4, group_size 128, sym, no desc_act; this one is {qc}",
            dir.display()
        );
        let v2 = qc["checkpoint_format"].as_str() == Some("gptq_v2");
        Ok(Gptq { st: SafeTensors::open_dir(dir)?, zero_stored: if v2 { 8 } else { 7 } })
    }

    /// Linear layer `name` ([rows = outputs][cols = inputs]): qweight [cols/8][rows] I32 (input
    /// k of output n at bits 4 (k % 8) of word [k / 8][n]), scales [groups][rows], qzeros
    /// [groups][rows/8] (all the symmetric zero), g_idx (k / 128 when present).
    fn mat(&self, name: &str, rows: usize, cols: usize) -> Result<Mat> {
        use crate::st::Dtype;
        use rayon::prelude::*;
        ensure!(rows % 8 == 0 && cols % 8 == 0, "{name}: [{rows}][{cols}] is not in whole 8-value words");
        let groups = cols.div_ceil(128);
        let need = |part: &str, shape: &[usize]| {
            let v = self.st.need(&format!("{name}.{part}"))?;
            ensure!(v.shape() == shape, "{name}.{part}: shape {:?}, expected {shape:?}", v.shape());
            Ok(v)
        };
        let qw = need("qweight", &[cols / 8, rows])?;
        let qz = need("qzeros", &[groups, rows / 8])?;
        ensure!(qw.dtype() == Dtype::I32 && qz.dtype() == Dtype::I32, "{name}: qweight / qzeros are not I32");
        let z = self.zero_stored * 0x1111_1111;
        ensure!(
            qz.bytes.chunks_exact(4).all(|c| u32::from_le_bytes(c.try_into().unwrap()) == z),
            "{name}: zero points other than the symmetric {} (an asymmetric checkpoint)",
            self.zero_stored
        );
        if let Some(gi) = self.st.get(&format!("{name}.g_idx")) {
            ensure!(
                gi.to_i64()?.iter().enumerate().all(|(k, &g)| g == (k / 128) as i64),
                "{name}: g_idx is not k / 128 (act-order groups are not supported)"
            );
        }
        let scale = need("scales", &[groups, rows])?.to_f32()?;
        let words = qw.bytes;
        let mut q = vec![0i8; rows * cols];
        q.par_chunks_mut(cols).enumerate().for_each(|(n, row)| {
            for kw in 0..cols / 8 {
                let o = (kw * rows + n) * 4;
                let w = u32::from_le_bytes(words[o..o + 4].try_into().unwrap());
                for j in 0..8 {
                    row[kw * 8 + j] = ((w >> (4 * j)) & 15) as i8 - 8;
                }
            }
        });
        Ok(Mat { rows, cols, data: MatData::Int4 { q, scale } })
    }
}

/// Every expert of layer `l` into the arena (quantised in parallel when 8-bit).
/// (`mlp`: a dense layer's MLP, expert x its x-th slice of the intermediate dimension: gate and up
/// rows, down columns)
fn load_experts(
    ex: &mut Executor,
    st: &SafeTensors,
    c: &DenseConfig,
    l: usize,
    p: Precision,
    mlp: bool,
    gptq: Option<&Gptq>,
    lora: Option<&Adapter>,
) -> Result<()> {
    use rayon::prelude::*;
    let (w, i) = (c.hidden, c.moe_inter);
    let mats: Vec<[Mat; 3]> = if mlp {
        let n = |m: &str| format!("model.layers.{l}.mlp.{m}.weight");
        let full = i * c.n_experts;
        let (g, u, d) = (
            mat(st, lora, &n("gate_proj"), full, w, Precision::Bf16)?,
            mat(st, lora, &n("up_proj"), full, w, Precision::Bf16)?,
            mat(st, lora, &n("down_proj"), w, full, Precision::Bf16)?,
        );
        let (MatData::Bf16(g), MatData::Bf16(u), MatData::Bf16(d)) = (&g.data, &u.data, &d.data) else {
            bail!("the dense MLP loads as bf16");
        };
        (0..c.n_experts)
            .into_par_iter()
            .map(|x| -> Result<[Mat; 3]> {
                let rows = |m: &Vec<u16>| m[x * i * w..(x + 1) * i * w].to_vec();
                let cols: Vec<u16> =
                    (0..w).flat_map(|r| d[r * full + x * i..r * full + (x + 1) * i].iter().copied()).collect();
                Ok([
                    Mat::bf16(i, w, rows(g)).convert(p),
                    Mat::bf16(i, w, rows(u)).convert(p),
                    Mat::bf16(w, i, cols).convert(p),
                ])
            })
            .collect::<Result<_>>()?
    } else if let Some(g) = gptq {
        (0..c.n_experts)
            .into_par_iter()
            .map(|x| -> Result<[Mat; 3]> {
                let n = |m: &str| format!("model.layers.{l}.mlp.experts.{x}.{m}");
                Ok([g.mat(&n("gate_proj"), i, w)?, g.mat(&n("up_proj"), i, w)?, g.mat(&n("down_proj"), w, i)?])
            })
            .collect::<Result<_>>()?
    } else {
        (0..c.n_experts)
            .into_par_iter()
            .map(|x| -> Result<[Mat; 3]> {
                let n = |m: &str| format!("model.layers.{l}.mlp.experts.{x}.{m}.weight");
                Ok([
                    mat(st, lora, &n("gate_proj"), i, w, p)?,
                    mat(st, lora, &n("up_proj"), i, w, p)?,
                    mat(st, lora, &n("down_proj"), w, i, p)?,
                ])
            })
            .collect::<Result<_>>()?
    };
    for (x, m) in mats.iter().enumerate() {
        let (_, w1, s1) = m[0].raw();
        let (_, w3, s3) = m[1].raw();
        let (_, w2, s2) = m[2].raw();
        // SAFETY: mat() built each matrix at the reserved format and the model's [rows][cols]
        unsafe { ex.set_gpu_expert(l, x, w1, s1, w3, s3, w2, s2) }.map_err(|e| anyhow!(e))?;
    }
    Ok(())
}

/// A `[rows][cols]` matrix from the checkpoint, with `lora`'s delta for it added (`--lora-merge`),
/// stored at precision `p` (bf16 kept as is).
fn mat(st: &SafeTensors, lora: Option<&Adapter>, name: &str, rows: usize, cols: usize, p: Precision) -> Result<Mat> {
    let v = st.need(name)?;
    ensure!(v.numel() == rows * cols, "{name}: {} elements, config predicts [{rows}][{cols}]", v.numel());
    let mut w = v.to_bf16().with_context(|| name.to_string())?;
    if let Some(a) = lora {
        a.merge_bf16(name, &mut w, rows, cols)?;
    }
    Ok(Mat::bf16(rows, cols, w).convert(p))
}

fn vector(st: &SafeTensors, name: &str, n: usize) -> Result<Vec<f32>> {
    let v = st.need(name)?;
    ensure!(v.numel() == n, "{name}: {} elements, config predicts {n}", v.numel());
    v.to_f32().with_context(|| name.to_string())
}

fn add_mat(ex: &mut Executor, name: &str, m: &Mat) -> Result<()> {
    let (fmt, bytes, scale) = m.raw();
    ensure!(!matches!(m.data, MatData::F32(_)));
    ex.add_matrix(name, fmt, m.rows, m.cols, bytes, scale).map_err(|e| anyhow!(e))
}

fn add_vec(ex: &mut Executor, name: &str, v: &[f32]) -> Result<()> {
    ex.add_vector(name, v).map_err(|e| anyhow!(e))
}

impl Model for DenseModel {
    fn set_lora(&mut self, slot: usize, a: &Adapter) -> Result<()> {
        ensure!(slot < self.paged.lora_slots(), "no LoRA slot {slot} (serve --max-loras)");
        let ex = self.paged.executor();
        ex.clear_lora(slot).map_err(|e| anyhow!(e))?;
        for (name, d) in a.deltas() {
            let (l, target) = lora_target(name).with_context(|| {
                format!("{name}: unmerged adapters update q/k/v/o and a dense MLP's gate/up/down only (merge it with --lora-merge)")
            })?;
            ensure!(l < self.cfg.n_layers, "{name}: no layer {l}");
            ensure!(
                target < 4 || self.cfg.n_experts == 0,
                "{name}: an MoE model's experts take a merged adapter only (--lora-merge)"
            );
            let bf = |v: &[f32]| v.iter().map(|&x| bf16_bits(x)).collect::<Vec<u16>>();
            ex.set_lora(slot, l, target, &bf(&d.a), &bf(&d.b), d.rank, d.scale).map_err(|e| anyhow!("{name}: {e}"))?;
        }
        Ok(())
    }
    fn clear_lora(&mut self, slot: usize) -> Result<()> {
        ensure!(slot < self.paged.lora_slots(), "no LoRA slot {slot}");
        self.paged.executor().clear_lora(slot).map_err(|e| anyhow!(e))
    }

    fn arch(&self) -> &'static str {
        match self.cfg.model_type.as_str() {
            "qwen3" => "qwen3",
            "qwen3_moe" => "qwen3_moe",
            _ => "llama",
        }
    }

    fn vocab(&self) -> usize {
        self.cfg.vocab
    }

    fn describe(&self) -> String {
        format!("{} [{:?} backend]", self.desc, self.backend)
    }

    // attention is the only per-sequence state: a rejected draft's cache is written over
    fn speculation(&self) -> bool {
        true
    }

    crate::delegate_paged!(paged);
}
