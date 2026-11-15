// SPDX-License-Identifier: Apache-2.0
//! EAGLE-3 draft heads (vLLM's `Eagle3LlamaForCausalLM`, checkpoints named
//! `LlamaForCausalLMEagle3`): one Llama decoder layer over the target's tapped hidden states,
//! trained by the user and loaded beside a dense target (`--spec-model DIR`). fullm only runs it.
//!
//! The head reads the residual stream entering three target layers (`eagle_aux_hidden_state_layer_ids`,
//! else layers 2, L/2 and L-3), in ascending layer order as vLLM concatenates them, fuses them with
//! `fc`, and runs its layer on [input_layernorm(embed(token)) | hidden_norm(fused)] with the
//! target's attention shape; its KV cache is one more layer of the target's paged cache. Its
//! vocabulary may be a subset of the target's: `d2t` maps each draft id to a target id
//! (target = id + d2t[id]). It shares the target's embedding, and the target's lm_head too when
//! it drafts over the whole vocabulary (as vLLM does). Variants vLLM switches on in the config
//! (`norm_before_residual`, `norm_before_fc`, `fc_norm`, `norm_output`, several layers, a fused
//! input other than three hidden states) are refused rather than run wrong.

use std::path::Path;

use anyhow::{anyhow, bail, ensure, Context, Result};
use fullm_kernels::exec::{EagleDims, Executor};
use serde_json::Value;

use crate::models::dense::{rope_inv_freq, DenseConfig};
use crate::st::SafeTensors;
use crate::tensor::{Mat, Precision};

/// The head in `dir` onto `ex` (a dense executor before `finalize`), for target `t`. Returns a
/// description of it.
pub fn attach(ex: &mut Executor, dir: &Path, t: &DenseConfig) -> Result<String> {
    let path = dir.join("config.json");
    let raw = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let v: Value = serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
    let dims = config(&v, t).with_context(|| format!("EAGLE-3 head {}", dir.display()))?;
    ex.set_eagle(&dims).map_err(|e| anyhow!("EAGLE-3 head: {e}"))?;
    if dir.join("pytorch_model.bin").exists()
        && !std::fs::read_dir(dir)?.flatten().any(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
    {
        bail!(
            "EAGLE-3 head {}: pytorch_model.bin only; fullm reads safetensors (convert it: \
             safetensors.torch.save_file(torch.load(path, weights_only=True), \"model.safetensors\"))",
            dir.display()
        );
    }
    let st = SafeTensors::open_dir(dir)?;
    let (e, q, kv) = (t.hidden, t.n_heads * t.head_dim, t.n_kv_heads * t.head_dim);
    let (i, dv) = (dims.inter as usize, dims.draft_vocab as usize);
    let layer = |n: &str| -> Result<String> {
        // "midlayer." in the published heads; vLLM maps it to "layers.0."
        ["midlayer.", "layers.0.", "model.layers.0.", "model.midlayer."]
            .iter()
            .map(|p| format!("{p}{n}"))
            .find(|name| st.get(name).is_some())
            .with_context(|| format!("EAGLE-3 head is missing tensor midlayer.{n}"))
    };
    let top = |n: &str| -> Result<String> {
        [n.to_string(), format!("model.{n}")]
            .into_iter()
            .find(|name| st.get(name).is_some())
            .with_context(|| format!("EAGLE-3 head is missing tensor {n}"))
    };
    let mats = [
        ("eagle.fc", top("fc.weight")?, e, 3 * e),
        ("eagle.q", layer("self_attn.q_proj.weight")?, q, 2 * e),
        ("eagle.k", layer("self_attn.k_proj.weight")?, kv, 2 * e),
        ("eagle.v", layer("self_attn.v_proj.weight")?, kv, 2 * e),
        ("eagle.o", layer("self_attn.o_proj.weight")?, e, q),
        ("eagle.gate", layer("mlp.gate_proj.weight")?, i, e),
        ("eagle.up", layer("mlp.up_proj.weight")?, i, e),
        ("eagle.down", layer("mlp.down_proj.weight")?, e, i),
    ];
    // the head stays bf16, as vLLM runs it
    for (dst, src, rows, cols) in &mats {
        let m = mat(&st, src, *rows, *cols)?;
        let (fmt, bytes, scale) = m.raw();
        ex.add_matrix(dst, fmt, m.rows, m.cols, bytes, scale).map_err(|e| anyhow!(e))?;
    }
    let vecs = [
        ("eagle.in_norm", layer("input_layernorm.weight")?),
        ("eagle.hidden_norm", layer("hidden_norm.weight")?),
        ("eagle.post_norm", layer("post_attention_layernorm.weight")?),
        ("eagle.norm", top("norm.weight")?),
    ];
    for (dst, src) in &vecs {
        let x = st.need(src)?;
        ensure!(x.numel() == e, "{src}: {} elements, the target's hidden size is {e}", x.numel());
        // rounded to bf16 like the matrices: vLLM casts every parameter of the head to the model's dtype
        let v: Vec<f32> = x.to_bf16()?.into_iter().map(|b| f32::from_bits(u32::from(b) << 16)).collect();
        ex.add_vector(dst, &v).map_err(|e| anyhow!(e))?;
    }
    let d2t: Vec<f32> = if dv == t.vocab {
        // the target's lm_head (the executor binds it when the head brings none)
        (0..dv).map(|x| x as f32).collect()
    } else {
        let m = mat(&st, "lm_head.weight", dv, e)?;
        let (fmt, bytes, scale) = m.raw();
        ex.add_matrix("eagle.lm_head", fmt, m.rows, m.cols, bytes, scale).map_err(|e| anyhow!(e))?;
        let d = st.need("d2t").context("a head over part of the vocabulary maps its ids with d2t")?;
        ensure!(d.numel() == dv, "d2t: {} elements, draft_vocab_size is {dv}", d.numel());
        let off = d.to_i64()?;
        let ids: Vec<i64> = off.iter().enumerate().map(|(x, o)| x as i64 + o).collect();
        ensure!(ids.iter().all(|&x| (0..t.vocab as i64).contains(&x)), "d2t maps a draft id outside the vocabulary");
        ids.into_iter().map(|x| x as f32).collect()
    };
    ex.add_vector("eagle.d2t", &d2t).map_err(|e| anyhow!(e))?;
    let [a, b, c] = dims.taps;
    Ok(format!("EAGLE-3 head (taps {a},{b},{c}; vocab {dv}; mlp {i})"))
}

/// The head's executor dims from its config, after checking it fits the target.
fn config(v: &Value, t: &DenseConfig) -> Result<EagleDims> {
    let u = |k: &str| v.get(k).and_then(Value::as_u64).map(|x| x as usize);
    let need = |k: &str| u(k).with_context(|| format!("config: missing {k}"));
    let flag = |k: &str| v.get(k).and_then(Value::as_bool).unwrap_or(false);
    let ec = v.get("eagle_config").filter(|x| x.is_object());
    let eflag = |k: &str| ec.and_then(|c| c.get(k)).and_then(Value::as_bool);

    let archs: Vec<&str> = v
        .get("architectures")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    ensure!(
        archs.iter().any(|a| matches!(*a, "LlamaForCausalLMEagle3" | "Eagle3LlamaForCausalLM")),
        "architectures {archs:?}: not an EAGLE-3 Llama head (LlamaForCausalLMEagle3)"
    );
    ensure!(need("num_hidden_layers")? == 1, "heads of more than one layer are not supported");
    let hidden = need("hidden_size")?;
    let heads = need("num_attention_heads")?;
    let kv_heads = u("num_key_value_heads").unwrap_or(heads);
    let head_dim = u("head_dim").unwrap_or(hidden / heads.max(1));
    let shape = (hidden, u("target_hidden_size").unwrap_or(hidden), heads, kv_heads, head_dim);
    let want = (t.hidden, t.hidden, t.n_heads, t.n_kv_heads, t.head_dim);
    ensure!(
        shape == want,
        "(hidden, target hidden, heads, kv heads, head_dim) {shape:?}, the target's {want:?}: the head's layer \
         shares the target's attention shape"
    );
    let vocab = need("vocab_size")?;
    ensure!(vocab == t.vocab, "vocab_size {vocab}, the target's {}", t.vocab);
    let draft_vocab = u("draft_vocab_size").unwrap_or(vocab);
    ensure!((1..=vocab).contains(&draft_vocab), "draft_vocab_size {draft_vocab}");
    let eps = v.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6) as f32;
    ensure!(eps == t.rms_eps, "rms_norm_eps {eps}, the target's {}: the executor uses one", t.rms_eps);
    let act = v.get("hidden_act").and_then(Value::as_str).unwrap_or("silu");
    ensure!(act == "silu", "activation {act:?} (only silu)");
    ensure!(!flag("attention_bias") && !flag("mlp_bias"), "attention / MLP biases are not supported");
    // as DenseConfig reads it: transformers 5 writes rope_parameters {rope_theta, rope_type, ...}
    let rp = v.get("rope_parameters").filter(|r| r.is_object());
    let theta =
        rp.and_then(|r| r.get("rope_theta")).or(v.get("rope_theta")).and_then(Value::as_f64).unwrap_or(10000.0) as f32;
    let inv = rope_inv_freq(head_dim, theta, v.get("rope_scaling").filter(|r| !r.is_null()).or(rp))?;
    ensure!(inv == t.inv_freq, "RoPE (theta {theta}, scaling) differs from the target's: the executor uses one");
    for k in ["norm_before_residual", "norm_before_fc", "fc_norm", "norm_output"] {
        ensure!(!eflag(k).unwrap_or(flag(k)), "{k}: this EAGLE-3 variant is not supported");
    }
    ensure!(eflag("use_aux_hidden_state").unwrap_or(true), "heads that read only the last hidden state (EAGLE-1/2)");
    ensure!(u("num_aux_hidden_states").unwrap_or(3) == 3, "num_aux_hidden_states: only three tapped layers");

    let ids = ec.and_then(|c| c.get("eagle_aux_hidden_state_layer_ids")).or(v.get("eagle_aux_hidden_state_layer_ids"));
    let l = t.n_layers;
    let mut taps: Vec<usize> = match ids.and_then(Value::as_array) {
        Some(a) => a.iter().map(|x| x.as_u64().map(|x| x as usize).context("a tap layer id")).collect::<Result<_>>()?,
        None => vec![2, l / 2, l.saturating_sub(3)],
    };
    // vLLM sorts and deduplicates them; the fused input is three hidden states
    taps.sort_unstable();
    taps.dedup();
    ensure!(taps.len() == 3, "tap layers {taps:?}: three distinct layers");
    ensure!(taps[2] < l, "tap layer {} past the target's {l} layers", taps[2]);
    let i32_ = |x: usize| x as i32;
    Ok(EagleDims {
        taps: [i32_(taps[0]), i32_(taps[1]), i32_(taps[2])],
        inter: i32_(need("intermediate_size")?),
        draft_vocab: i32_(draft_vocab),
    })
}

fn mat(st: &SafeTensors, name: &str, rows: usize, cols: usize) -> Result<Mat> {
    let v = st.need(name)?;
    ensure!(v.numel() == rows * cols, "{name}: {} elements, the head predicts [{rows}][{cols}]", v.numel());
    Ok(Mat::bf16(rows, cols, v.to_bf16().with_context(|| name.to_string())?).convert(Precision::Bf16))
}
