// SPDX-License-Identifier: Apache-2.0
//! Configuration of the Kimi hybrid family: Kimi K3 and Kimi Linear. Read strictly: an
//! absent field is an error, never a default.
//!
//! Both models share one skeleton (KDA layers interleaved with NoPE MLA layers, a sigmoid-
//! routed MoE, one dense layer first) and differ in a handful of switches, each recorded
//! below as an explicit field so the forward passes branch on facts, not on model names:
//!
//! | switch            | Kimi K3                         | Kimi Linear                      |
//! |-------------------|---------------------------------|----------------------------------|
//! | KDA decay         | lb * sigmoid(exp(A) * z)        | -exp(A) * softplus(z)            |
//! | KDA output gate   | full-rank g_proj                | low-rank g_b(g_a(x))             |
//! | MLA query         | q_a -> norm -> q_b (LoRA)       | one q_proj                       |
//! | MLA output gate   | yes                             | no                               |
//! | MLA kv norm eps   | rms_norm_eps                    | 1e-6 (KimiRMSNorm default)       |
//! | residual          | Block Attention Residuals       | plain pre-norm                   |
//! | MoE width         | latent projection + norm        | full hidden width                |
//! | activation        | SiTU-GLU                        | SiLU-GLU                         |
//! | routed experts    | MXFP4                           | bf16                             |
//!
//! Kimi K3: two layouts describe this model. The released config.json nests the language model
//! under `text_config` and the KDA/MLA map under `text_config.linear_attn_config`; the
//! test fixtures are flat and shorten a few names. Both are accepted, every missing key is
//! reported at once, and nothing is guessed: a reader that defaulted `full_attn_layers`
//! to empty would run all 93 layers as KDA and still print fluent text.

use anyhow::{bail, Result};
use serde::Serialize;
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Family {
    K3,
    Linear,
}

#[derive(Clone, Debug, Serialize)]
pub struct K3Config {
    pub family: Family,
    pub hidden: usize,
    pub n_layers: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub tie_word_embeddings: bool,

    // Kimi Delta Attention
    pub kda_heads: usize,
    pub kda_head_dim: usize,
    pub conv_k: usize,
    pub gate_lb: f32,

    // Gated MLA (NoPE: the rope slots exist and are cached, but are never rotated)
    pub n_heads: usize,
    pub q_lora: usize,
    pub kv_lora: usize,
    pub qk_nope: usize,
    pub qk_rope: usize,
    pub v_head: usize,
    pub mla_out_gate: bool,

    // Stable LatentMoE
    pub n_experts: usize,
    pub topk: usize,
    pub n_shared: usize,
    pub latent: usize,
    pub moe_inter: usize,
    pub routed_scale: f32,
    pub moe_renorm: bool,
    pub latent_norm: bool,

    pub first_dense: usize,
    pub dense_inter: usize,
    pub attn_res_block: usize,

    // SiTU-GLU
    pub situ_b1: f32,
    pub situ_b2: f32,

    /// ONE-based, as both config shapes store it.
    pub full_attn_layers: Vec<usize>,

    // Family switches (see the table above).
    /// Decay g = -exp(A) * softplus(z) instead of lb * sigmoid(exp(A) * z).
    pub decay_softplus: bool,
    /// KDA output gate is g_b(g_a(x)) (rank head_dim) instead of a full-rank g_proj.
    pub kda_gate_lowrank: bool,
    /// Block Attention Residuals; false means plain pre-norm residuals.
    pub attn_res: bool,
    /// Routed experts run in a latent space (down/up projections, aggregate norm).
    pub latent_moe: bool,
    /// SiLU-GLU instead of SiTU-GLU.
    pub act_silu: bool,
    /// Routed experts are stored bf16 rather than MXFP4.
    pub expert_bf16: bool,
    /// eps of the MLA kv_a_layernorm.
    pub kv_norm_eps: f32,
    /// Take the MoE combining weights from the BIASED scores. DeepSeek-V3, Kimi K3 and vLLM
    /// use the unbiased scores (false); the released Kimi Linear modeling_kimi.py adds the
    /// bias in place to a view of the scores and so uses the biased ones. A load option.
    pub router_weights_biased: bool,
    /// Tensor-name prefix of the layers and the lm_head's full name.
    pub prefix: String,
    pub lm_head_name: String,
}

struct Src<'a> {
    txt: Option<&'a Value>,
    lin: Option<&'a Value>,
    root: &'a Value,
    missing: Vec<&'static str>,
}

impl<'a> Src<'a> {
    fn find(&self, names: &[&str], lin_first: bool) -> Option<&'a Value> {
        let order: [Option<&Value>; 3] =
            if lin_first { [self.lin, self.txt, Some(self.root)] } else { [self.txt, self.lin, Some(self.root)] };
        for o in order.into_iter().flatten() {
            for n in names {
                if let Some(v) = o.get(*n) {
                    return Some(v);
                }
            }
        }
        None
    }
    fn usize_(&mut self, primary: &'static str, alias: Option<&str>, lin_first: bool) -> usize {
        let names: Vec<&str> = std::iter::once(primary).chain(alias).collect();
        match self.find(&names, lin_first).and_then(|v| v.as_f64()) {
            Some(x) if x >= 0.0 => x as usize,
            _ => {
                self.missing.push(primary);
                0
            }
        }
    }
    fn f32_(&mut self, primary: &'static str, alias: Option<&str>) -> f32 {
        let names: Vec<&str> = std::iter::once(primary).chain(alias).collect();
        match self.find(&names, false).and_then(|v| v.as_f64()) {
            Some(x) => x as f32,
            None => {
                self.missing.push(primary);
                0.0
            }
        }
    }
    /// Genuinely optional booleans with a real default.
    fn bool_(&self, name: &str, dflt: bool) -> bool {
        match self.find(&[name], false) {
            Some(Value::Bool(b)) => *b,
            Some(Value::Number(n)) => n.as_f64() != Some(0.0),
            _ => dflt,
        }
    }
}

impl K3Config {
    pub fn from_json(root: &Value) -> Result<Self> {
        // The released K3 config says kimi_k3 at the top (its text_config says kimi_linear);
        // the tiny K3 fixture has no model_type at all. Only a top-level kimi_linear is Kimi Linear.
        if root.get("model_type").and_then(|v| v.as_str()) == Some("kimi_linear") {
            return Self::from_json_linear(root);
        }
        Self::from_json_k3(root)
    }

    fn from_json_k3(root: &Value) -> Result<Self> {
        let txt = root.get("text_config");
        let base = txt.unwrap_or(root);
        let lin = base.get("linear_attn_config");
        let mut s = Src { txt, lin, root, missing: Vec::new() };

        let fal = s
            .find(&["full_attn_layers"], true)
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(|x| x.as_u64().map(|v| v as usize)).collect::<Option<Vec<_>>>());
        let full_attn_layers = match fal {
            Some(Some(v)) if !v.is_empty() => v,
            _ => {
                s.missing.push("full_attn_layers");
                Vec::new()
            }
        };

        let rms_eps = s.f32_("rms_norm_eps", None);
        let c = K3Config {
            family: Family::K3,
            hidden: s.usize_("hidden_size", None, false),
            n_layers: s.usize_("num_hidden_layers", None, false),
            vocab: s.usize_("vocab_size", None, false),
            rms_eps,
            tie_word_embeddings: s.bool_("tie_word_embeddings", false),
            // KDA fields live in linear_attn_config in the released shape; search it first
            // so a text_config-level num_heads/head_dim can never be read in their place.
            kda_heads: s.usize_("num_heads", Some("kda_num_heads"), true),
            kda_head_dim: s.usize_("head_dim", Some("kda_head_dim"), true),
            conv_k: s.usize_("short_conv_kernel_size", None, true),
            gate_lb: s.f32_("gate_lower_bound", None),
            n_heads: s.usize_("num_attention_heads", None, false),
            q_lora: s.usize_("q_lora_rank", None, false),
            kv_lora: s.usize_("kv_lora_rank", None, false),
            qk_nope: s.usize_("qk_nope_head_dim", None, false),
            qk_rope: s.usize_("qk_rope_head_dim", None, false),
            v_head: s.usize_("v_head_dim", None, false),
            mla_out_gate: s.bool_("mla_use_output_gate", true),
            n_experts: s.usize_("num_experts", None, false),
            topk: s.usize_("num_experts_per_token", None, false),
            n_shared: s.usize_("num_shared_experts", None, false),
            latent: s.usize_("routed_expert_hidden_size", None, false),
            moe_inter: s.usize_("moe_intermediate_size", None, false),
            routed_scale: s.f32_("routed_scaling_factor", None),
            moe_renorm: s.bool_("moe_renormalize", true),
            latent_norm: s.bool_("latent_moe_use_norm", true),
            first_dense: s.usize_("first_k_dense_replace", None, false),
            dense_inter: s.usize_("intermediate_size", None, false),
            attn_res_block: s.usize_("attn_res_block_size", None, false),
            situ_b1: s.f32_("activation_situ_beta", Some("situ_beta")),
            situ_b2: s.f32_("activation_situ_linear_beta", Some("situ_linear_beta")),
            full_attn_layers,
            decay_softplus: false,
            kda_gate_lowrank: false,
            attn_res: true,
            latent_moe: true,
            act_silu: false,
            expert_bf16: false,
            kv_norm_eps: rms_eps,
            router_weights_biased: false,
            prefix: "language_model.model.".into(),
            lm_head_name: "language_model.lm_head.weight".into(),
        };
        if !s.missing.is_empty() {
            bail!(
                "config is missing {} required field(s): {}. Refusing to substitute defaults: a config \
                 this reader cannot fully understand would silently describe a different model.",
                s.missing.len(),
                s.missing.join(", ")
            );
        }
        c.validate()?;
        Ok(c)
    }

    /// Kimi Linear (moonshotai/Kimi-Linear-48B-A3B-Instruct, configuration_kimi.py). Every
    /// switch it does not support is refused rather than ignored.
    fn from_json_linear(root: &Value) -> Result<Self> {
        let lin = root.get("linear_attn_config");
        let mut s = Src { txt: None, lin, root, missing: Vec::new() };
        let layers = |s: &Src, key: &str| -> Option<Vec<usize>> {
            s.find(&[key], true)?.as_array()?.iter().map(|x| x.as_u64().map(|v| v as usize)).collect()
        };
        let full_attn_layers = layers(&s, "full_attn_layers").unwrap_or_else(|| {
            s.missing.push("linear_attn_config.full_attn_layers");
            Vec::new()
        });
        let kda_layers = layers(&s, "kda_layers").unwrap_or_else(|| {
            s.missing.push("linear_attn_config.kda_layers");
            Vec::new()
        });
        let str_ = |k: &str| root.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
        let hidden = s.usize_("hidden_size", None, false);
        let rms_eps = s.f32_("rms_norm_eps", None);
        let c = K3Config {
            family: Family::Linear,
            hidden,
            n_layers: s.usize_("num_hidden_layers", None, false),
            vocab: s.usize_("vocab_size", None, false),
            rms_eps,
            tie_word_embeddings: s.bool_("tie_word_embeddings", false),
            kda_heads: s.usize_("num_heads", None, true),
            kda_head_dim: s.usize_("head_dim", None, true),
            conv_k: s.usize_("short_conv_kernel_size", None, true),
            gate_lb: 0.0,
            n_heads: s.usize_("num_attention_heads", None, false),
            q_lora: 0,
            kv_lora: s.usize_("kv_lora_rank", None, false),
            qk_nope: s.usize_("qk_nope_head_dim", None, false),
            qk_rope: s.usize_("qk_rope_head_dim", None, false),
            v_head: s.usize_("v_head_dim", None, false),
            mla_out_gate: false,
            n_experts: s.usize_("num_experts", None, false),
            topk: s.usize_("num_experts_per_token", None, false),
            n_shared: s.usize_("num_shared_experts", None, false),
            latent: hidden,
            moe_inter: s.usize_("moe_intermediate_size", None, false),
            routed_scale: s.f32_("routed_scaling_factor", None),
            moe_renorm: s.bool_("moe_renormalize", true),
            latent_norm: false,
            first_dense: s.usize_("first_k_dense_replace", None, false),
            dense_inter: s.usize_("intermediate_size", None, false),
            attn_res_block: 0,
            situ_b1: 0.0,
            situ_b2: 0.0,
            full_attn_layers,
            decay_softplus: true,
            kda_gate_lowrank: true,
            attn_res: false,
            latent_moe: false,
            act_silu: true,
            expert_bf16: true,
            kv_norm_eps: 1e-6,
            router_weights_biased: false,
            prefix: "model.".into(),
            lm_head_name: "lm_head.weight".into(),
        };
        if !s.missing.is_empty() {
            bail!("Kimi Linear config is missing {} required field(s): {}", s.missing.len(), s.missing.join(", "));
        }
        // Switches this implementation does not have must not be silently ignored.
        let mut unsupported = Vec::new();
        if !root.get("q_lora_rank").is_some_and(|v| v.is_null()) {
            unsupported.push("q_lora_rank must be null".to_string());
        }
        if root.get("mla_use_nope").and_then(|v| v.as_bool()) != Some(true) {
            unsupported.push("mla_use_nope must be true".into());
        }
        if str_("hidden_act") != "silu" {
            unsupported.push(format!("hidden_act {:?}", str_("hidden_act")));
        }
        if str_("moe_router_activation_func") != "sigmoid" {
            unsupported.push(format!("moe_router_activation_func {:?}", str_("moe_router_activation_func")));
        }
        let int = |k: &str| root.get(k).and_then(|v| v.as_u64());
        if int("num_expert_group").unwrap_or(1) != 1 || int("topk_group").unwrap_or(1) != 1 {
            unsupported.push("grouped routing (num_expert_group/topk_group != 1)".into());
        }
        if int("moe_layer_freq").unwrap_or(1) != 1 {
            unsupported.push("moe_layer_freq != 1".into());
        }
        if int("num_key_value_heads").is_some_and(|k| k as usize != c.n_heads) {
            unsupported.push("num_key_value_heads != num_attention_heads".into());
        }
        // kda_layers and full_attn_layers must partition the layers exactly
        let mut covered = vec![0u8; c.n_layers + 1];
        for &l in kda_layers.iter().chain(&c.full_attn_layers) {
            if l == 0 || l > c.n_layers {
                unsupported.push(format!("layer index {l} outside 1..{}", c.n_layers));
            } else {
                covered[l] += 1;
            }
        }
        if covered[1..].iter().any(|&n| n != 1) {
            unsupported.push("kda_layers and full_attn_layers do not partition the layers".into());
        }
        if !unsupported.is_empty() {
            bail!("unsupported Kimi Linear config: {}", unsupported.join("; "));
        }
        c.validate()?;
        Ok(c)
    }

    fn validate(&self) -> Result<()> {
        if self.n_layers == 0 || self.hidden == 0 || self.vocab == 0 {
            bail!("non-positive layers/hidden/vocab");
        }
        if self.full_attn_layers.len() >= self.n_layers {
            bail!("every layer is marked full-attention; no KDA layers remain");
        }
        if let Some(&bad) = self.full_attn_layers.iter().find(|&&l| l < 1 || l > self.n_layers) {
            bail!("full_attn_layers entry {bad} is outside 1..{} (the list is ONE-based)", self.n_layers);
        }
        if self.topk == 0 || self.topk > self.n_experts {
            bail!("top-{} of {} experts", self.topk, self.n_experts);
        }
        if self.attn_res && self.attn_res_block == 0 {
            bail!("attn_res_block_size is 0");
        }
        if self.conv_k == 0 {
            bail!("short_conv_kernel_size is 0");
        }
        if !self.expert_bf16 && (!self.latent.is_multiple_of(32) || !self.moe_inter.is_multiple_of(32)) {
            bail!("routed expert widths {}x{} are not multiples of the MXFP4 group (32)", self.latent, self.moe_inter);
        }
        Ok(())
    }

    /// The released Kimi K3 values (its config.json's). Used
    /// to plan a deployment without a checkpoint on hand; never used to fill gaps.
    pub fn reference() -> Self {
        let n_layers = 93;
        let mut fa: Vec<usize> = (4..=n_layers).filter(|i| i % 4 == 0).collect();
        fa.push(n_layers);
        K3Config {
            family: Family::K3,
            hidden: 7168,
            n_layers,
            vocab: 163_840,
            rms_eps: 1e-5,
            tie_word_embeddings: false,
            kda_heads: 96,
            kda_head_dim: 128,
            conv_k: 4,
            gate_lb: -5.0,
            n_heads: 96,
            q_lora: 1536,
            kv_lora: 512,
            qk_nope: 128,
            qk_rope: 64,
            v_head: 128,
            mla_out_gate: true,
            n_experts: 896,
            topk: 16,
            n_shared: 2,
            latent: 3584,
            moe_inter: 3072,
            routed_scale: 1.0,
            moe_renorm: true,
            latent_norm: true,
            first_dense: 1,
            dense_inter: 33792,
            attn_res_block: 12,
            situ_b1: 4.0,
            situ_b2: 25.0,
            full_attn_layers: fa,
            decay_softplus: false,
            kda_gate_lowrank: false,
            attn_res: true,
            latent_moe: true,
            act_silu: false,
            expert_bf16: false,
            kv_norm_eps: 1e-5,
            router_weights_biased: false,
            prefix: "language_model.model.".into(),
            lm_head_name: "language_model.lm_head.weight".into(),
        }
    }

    /// Keep only the first `n` layers: a debugging aid for running real weights from a
    /// partial download. The result is NOT the model; every backend computes the same
    /// truncated graph, which is what makes it useful for comparing them.
    pub fn truncate(&mut self, n: usize) {
        if n < self.n_layers {
            self.n_layers = n;
            self.full_attn_layers.retain(|&l| l <= n);
        }
    }

    /// `layer` is ZERO-based; the config list is ONE-based.
    pub fn is_mla(&self, layer: usize) -> bool {
        self.full_attn_layers.contains(&(layer + 1))
    }
    pub fn is_dense(&self, layer: usize) -> bool {
        layer < self.first_dense
    }
    /// Upper bound on AttnRes block snapshots, plus one slot for the running prefix.
    pub fn max_sources(&self) -> usize {
        if self.attn_res {
            self.n_layers / self.attn_res_block + 2
        } else {
            1
        }
    }
    pub fn kda_proj(&self) -> usize {
        self.kda_heads * self.kda_head_dim
    }
    pub fn n_mla(&self) -> usize {
        (0..self.n_layers).filter(|&l| self.is_mla(l)).count()
    }
    pub fn n_kda(&self) -> usize {
        self.n_layers - self.n_mla()
    }
    pub fn n_moe(&self) -> usize {
        self.n_layers - self.first_dense.min(self.n_layers)
    }
    /// Latent + shared rope slot: the width of one cached MLA position per layer.
    pub fn mla_cache_width(&self) -> usize {
        self.kv_lora + self.qk_rope
    }
    /// fp32 floats of recurrent state per KDA layer per sequence: S plus 3 conv histories.
    pub fn kda_state_floats(&self) -> usize {
        self.kda_heads * self.kda_head_dim * self.kda_head_dim + 3 * self.kda_proj() * (self.conv_k - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_layer_map() {
        let c = K3Config::reference();
        assert_eq!(c.n_mla(), 24);
        assert_eq!(c.n_kda(), 69);
        assert!(c.is_mla(91) && c.is_mla(92), "the last two layers are both MLA");
        assert!(c.is_mla(3) && !c.is_mla(4));
    }

    #[test]
    fn missing_fields_are_all_reported() {
        let v: Value = serde_json::json!({"hidden_size": 8});
        let e = K3Config::from_json(&v).unwrap_err().to_string();
        assert!(
            e.contains("full_attn_layers") && e.contains("num_hidden_layers") && e.contains("activation_situ_beta")
        );
    }

    #[test]
    fn nested_shape_reads_linear_attn_config() {
        let v: Value = serde_json::json!({"text_config": {
            "hidden_size": 64, "num_hidden_layers": 4, "vocab_size": 32, "rms_norm_eps": 1e-5,
            "num_attention_heads": 2, "q_lora_rank": 16, "kv_lora_rank": 8, "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 4, "v_head_dim": 8, "num_experts": 4, "num_experts_per_token": 2,
            "num_shared_experts": 1, "routed_expert_hidden_size": 32, "moe_intermediate_size": 32,
            "routed_scaling_factor": 1.0, "first_k_dense_replace": 1, "intermediate_size": 64,
            "attn_res_block_size": 2, "activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0,
            "head_dim": 999,
            "linear_attn_config": {"num_heads": 2, "head_dim": 8, "short_conv_kernel_size": 4,
                                   "gate_lower_bound": -5.0, "full_attn_layers": [4]}}});
        let c = K3Config::from_json(&v).unwrap();
        assert_eq!(c.kda_head_dim, 8, "KDA head_dim must come from linear_attn_config");
        assert_eq!(c.full_attn_layers, vec![4]);
    }
}
