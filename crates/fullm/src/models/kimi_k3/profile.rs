// SPDX-License-Identifier: Apache-2.0
//! Byte-level profile of a Kimi K3 deployment, derived from the config alone.
//!
//! The planner never guesses sizes: every figure here is a product of config fields, and
//! the reference config reproduces the released checkpoint's published totals (108.81 GB
//! bf16 trunk, 17,547,264 bytes per MXFP4 expert, 56.74B always-active parameters). The
//! tests pin those numbers so a formula error cannot slip through as a plausible plan.

use super::K3Config;

/// Parameter counts by placement class. "q8" = the projections an 8-bit trunk quantises;
/// "keep16" = the bf16 matrices that stay 16-bit in every trunk mode (low-rank pieces
/// feeding attention scores or the decay gate, and the router); "vec" = f32 vectors.
#[derive(Clone, Copy, Debug, Default)]
pub struct ParamClasses {
    pub q8: u64,
    pub keep16: u64,
    pub vec: u64,
    /// Output rows of the q8 matrices (one f32 scale each when quantised).
    pub q8_rows: u64,
}

impl ParamClasses {
    fn add(&mut self, o: ParamClasses) {
        self.q8 += o.q8;
        self.keep16 += o.keep16;
        self.vec += o.vec;
        self.q8_rows += o.q8_rows;
    }
    pub fn total(&self) -> u64 {
        self.q8 + self.keep16 + self.vec
    }
}

#[derive(Clone, Debug)]
pub struct K3Profile {
    /// Per layer, trunk parameters (everything but routed experts).
    pub layers: Vec<ParamClasses>,
    pub embed_params: u64,
    pub lm_head_params: u64,
    pub n_moe_layers: usize,
    pub n_experts: usize,
    pub topk: usize,
    pub expert_params: u64,
    /// MXFP4: half a byte per weight plus one E8M0 byte per 32.
    pub expert_bytes: u64,
    /// Per sequence, every KDA layer's fp32 recurrent state plus conv histories.
    pub kda_state_bytes: u64,
    /// Per cached position across all MLA layers, at the given element size.
    pub mla_floats_per_token: u64,
    /// FLOPs per token for one routed expert (gate + up + down).
    pub expert_flops: u64,
    pub hidden: usize,
}

impl K3Profile {
    pub fn new(c: &K3Config) -> Self {
        let e = c.hidden as u64;
        let p = c.kda_proj() as u64;
        let d = c.kda_head_dim as u64;
        let h = c.n_heads as u64;
        let qh = (c.qk_nope + c.qk_rope) as u64;
        let (lat, inter) = (c.latent as u64, c.moe_inter as u64);
        let si = inter * c.n_shared as u64;

        let mut layers = Vec::with_capacity(c.n_layers);
        for l in 0..c.n_layers {
            let mut pc = ParamClasses { vec: if c.attn_res { 6 * e } else { 2 * e }, ..Default::default() };
            if c.is_mla(l) {
                let v = h * c.v_head as u64;
                let ql = c.q_lora as u64;
                // LoRA query (q_a, q_b) or one direct q_proj
                let (q_params, q_rows) =
                    if ql > 0 { (ql * e + h * qh * ql, ql + h * qh) } else { (h * qh * e, h * qh) };
                pc.add(ParamClasses {
                    q8: q_params + e * v + if c.mla_out_gate { v * e } else { 0 },
                    q8_rows: q_rows + e + if c.mla_out_gate { v } else { 0 },
                    keep16: (c.kv_lora + c.qk_rope) as u64 * e + h * (c.qk_nope + c.v_head) as u64 * c.kv_lora as u64,
                    vec: ql + c.kv_lora as u64,
                });
            } else {
                // q, k, v, o (+ a full-rank gate) are 8-bit candidates; a low-rank gate stays 16-bit
                let (gq, gk) = if c.kda_gate_lowrank { (0, d * e + p * d) } else { (p * e, 0) };
                pc.add(ParamClasses {
                    q8: 3 * p * e + gq + e * p,
                    q8_rows: 3 * p + if c.kda_gate_lowrank { 0 } else { p } + e,
                    keep16: d * e + p * d + c.kda_heads as u64 * e + gk,
                    vec: 3 * p * c.conv_k as u64 + d + p + d,
                });
            }
            if c.is_dense(l) {
                let di = c.dense_inter as u64;
                pc.add(ParamClasses { q8: 3 * di * e, q8_rows: 2 * di + e, ..Default::default() });
            } else {
                let latent = if c.latent_moe { 2 * lat * e } else { 0 };
                pc.add(ParamClasses {
                    q8: latent + 3 * si * e,
                    q8_rows: if c.latent_moe { lat + e } else { 0 } + 2 * si + e,
                    keep16: c.n_experts as u64 * e,
                    vec: c.n_experts as u64 + if c.latent_moe { lat } else { 0 },
                });
            }
            layers.push(pc);
        }

        let expert_params = 3 * lat * inter;
        K3Profile {
            layers,
            embed_params: c.vocab as u64 * e,
            lm_head_params: if c.tie_word_embeddings { 0 } else { c.vocab as u64 * e },
            n_moe_layers: c.n_moe(),
            n_experts: c.n_experts,
            topk: c.topk,
            expert_params,
            expert_bytes: if c.expert_bf16 { 2 * expert_params } else { expert_params / 2 + expert_params / 32 },
            kda_state_bytes: (c.n_kda() * c.kda_state_floats() * 4) as u64,
            mla_floats_per_token: (c.n_mla() * c.mla_cache_width()) as u64,
            expert_flops: 2 * expert_params,
            hidden: c.hidden,
        }
    }

    pub fn trunk(&self) -> ParamClasses {
        let mut t = ParamClasses::default();
        for l in &self.layers {
            t.add(*l);
        }
        t
    }

    /// Bytes of one layer's trunk at the given storage of the q8 class (1 = int8/fp8 with
    /// per-row f32 scales, 2 = bf16). keep16 is always bf16, vectors always f32.
    pub fn layer_bytes(&self, l: usize, q8_bytes: u64) -> u64 {
        let p = &self.layers[l];
        let scales = if q8_bytes == 1 { p.q8_rows * 4 } else { 0 };
        p.q8 * q8_bytes + scales + p.keep16 * 2 + p.vec * 4
    }

    pub fn trunk_bytes(&self, q8_bytes: u64) -> u64 {
        (0..self.layers.len()).map(|l| self.layer_bytes(l, q8_bytes)).sum()
    }

    pub fn total_experts(&self) -> u64 {
        (self.n_moe_layers * self.n_experts) as u64
    }

    pub fn all_expert_bytes(&self) -> u64 {
        self.total_experts() * self.expert_bytes
    }

    /// Expected distinct experts per layer when `tokens` tokens each pick top-k uniformly.
    /// K3's router is trained with Quantile Balancing to flatten usage, which is what makes
    /// this the right model and an expert LRU nearly useless below ~190 GB.
    pub fn unique_experts(&self, tokens: f64) -> f64 {
        let n = self.n_experts as f64;
        n * (1.0 - (1.0 - self.topk as f64 / n).powf(tokens))
    }

    /// Always-active parameters (trunk + embed + lm_head), as the shard headers sum them.
    pub fn always_active_params(&self) -> u64 {
        self.trunk().total() + self.embed_params + self.lm_head_params
    }

    /// Parameters touched by one token.
    pub fn active_params(&self) -> u64 {
        self.trunk().total() + self.lm_head_params + (self.n_moe_layers * self.topk) as u64 * self.expert_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_profile_matches_released_checkpoint() {
        let p = K3Profile::new(&K3Config::reference());
        assert_eq!(p.expert_bytes, 17_547_264, "one MXFP4 expert");
        assert_eq!(p.expert_params, 33_030_144);
        assert_eq!(p.total_experts(), 92 * 896);
        // 56,743,648,000 always-active parameters, summed from the released shards' headers.
        let aa = p.always_active_params() as f64;
        assert!((aa - 56_743_648_000.0).abs() / aa < 0.001, "always-active params {aa}");
        // bf16 trunk 108.81 GB (vectors are f32 here, which the released files mostly are not,
        // so allow a little slack above).
        let bf16 = p.trunk_bytes(2) as f64 / 1e9;
        assert!((108.7..109.2).contains(&bf16), "bf16 trunk {bf16} GB");
        let active = p.active_params() as f64 / 1e9;
        assert!((103.0..106.0).contains(&active), "~104B active, got {active}");
        assert_eq!(p.kda_state_bytes, 69 * (96 * 128 * 128 + 3 * 12288 * 3) * 4);
        // 25.83 GB of routed experts per token: 92 MoE layers x top-16
        let per_tok = (92 * 16) as f64 * p.expert_bytes as f64 / 1e9;
        assert!((per_tok - 25.83).abs() < 0.01);
    }
}
