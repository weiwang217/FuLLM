// SPDX-License-Identifier: Apache-2.0
//! Inkling 2 (looped) architecture, from a released config.json (its `text_config`).

use anyhow::{ensure, Context, Result};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct Ink2Config {
    pub hidden: usize,
    pub n_layers: usize,
    /// Embedding rows; the unembedding is stored padded to this many.
    pub vocab: usize,
    /// Logits per row (the unembedding's first rows).
    pub out_vocab: usize,
    pub rms_eps: f32,

    /// Query heads, key/value heads and head width of the layers attending to the whole context,
    /// and of the sliding-window layers.
    pub full_heads: (usize, usize, usize),
    pub swa_heads: (usize, usize, usize),
    /// Per layer: attends to the last `window` positions only.
    pub sliding: Vec<bool>,
    pub window: usize,
    /// Width of the relative states, and how far back the full layers' position bias reaches (the
    /// sliding layers': their window).
    pub d_rel: usize,
    pub rel_extent: usize,
    /// Full layers scale queries and relative states by 1 + alpha ln(max((pos + 1) / n_floor, 1)).
    pub log_n_floor: f32,
    pub log_alpha: f32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    /// Taps of the short convolutions.
    pub conv_k: usize,
    /// Layers per attention-residual block.
    pub block: usize,

    pub n_experts: usize,
    pub topk: usize,
    pub moe_inter: usize,
    pub gate_softcap: f32,
    pub up_softcap: f32,
    pub route_scale: f32,
    pub logit_softcap: f32,

    /// Layers `loop_start..loop_end` run `passes` times; the message gates' rank.
    pub loop_start: usize,
    pub loop_end: usize,
    pub passes: usize,
    pub gate_rank: usize,
}

impl Ink2Config {
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let t = v.get("text_config").unwrap_or(v);
        let u = |k: &str| -> Result<usize> {
            t.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).with_context(|| format!("config: missing {k}"))
        };
        let u_or = |k: &str, d: usize| t.get(k).and_then(|x| x.as_u64()).map_or(d, |x| x as usize);
        let f_or = |k: &str, d: f64| t.get(k).and_then(|x| x.as_f64()).unwrap_or(d) as f32;
        let n_layers = u("num_hidden_layers")?;
        let full = (u("num_attention_heads")?, u("num_key_value_heads")?, u("head_dim")?);
        let swa = (
            u_or("swa_num_attention_heads", full.0),
            u_or("swa_num_key_value_heads", full.1),
            u_or("swa_head_dim", full.2),
        );
        // the release lists the sliding layers; transformers' default is every sixth layer full
        let sliding: Vec<bool> = match t.get("local_layer_ids").and_then(|x| x.as_array()) {
            Some(ids) => {
                let ids: Vec<u64> = ids.iter().filter_map(|x| x.as_u64()).collect();
                (0..n_layers).map(|l| ids.contains(&(l as u64))).collect()
            }
            None => (0..n_layers).map(|l| (l + 1) % 6 != 0).collect(),
        };
        let lp = t.get("loop_config");
        let lu = |k: &str, flat: &str, d: usize| {
            lp.and_then(|l| l.get(k)).and_then(|x| x.as_u64()).map_or_else(|| u_or(flat, d), |x| x as usize)
        };
        let vocab = u("vocab_size")?;
        let c = Ink2Config {
            hidden: u("hidden_size")?,
            n_layers,
            vocab,
            out_vocab: u_or("unpadded_vocab_size", vocab),
            rms_eps: f_or("rms_norm_eps", 1e-6),
            full_heads: full,
            swa_heads: swa,
            sliding,
            window: u_or("sliding_window_size", 1024),
            d_rel: u_or("d_rel", 16),
            rel_extent: u_or("rel_extent", 1024),
            log_n_floor: f_or("log_scaling_n_floor", 8192.0),
            log_alpha: f_or("log_scaling_alpha", 0.1),
            qk_clip: f_or("qk_l2_clip_threshold", 22.627416997969522),
            rel_clip: f_or("rel_l2_clip_threshold", 8.0),
            conv_k: u_or("sconv_kernel_size", u_or("conv_kernel_size", 4)),
            block: u_or("n_layers_per_attn_res_block", 6),
            n_experts: u("n_routed_experts")?,
            topk: u("num_experts_per_tok")?,
            moe_inter: u_or("intermediate_size", u_or("moe_intermediate_size", 0)),
            gate_softcap: f_or("mlp_gate_softcap", 4.0),
            up_softcap: f_or("mlp_up_softcap", 25.0),
            route_scale: f_or("route_scale", 8.0),
            logit_softcap: f_or("final_logit_softcapping", 30.0),
            loop_start: lu("core_start", "loop_layer_index_start", 6),
            loop_end: lu("core_end", "loop_layer_index_end", 18),
            passes: lu("num_passes", "loop_num_passes", 3),
            gate_rank: lu("gate_rank", "loop_gate_rank", 64),
        };
        ensure!(c.moe_inter > 0, "config: missing intermediate_size");
        ensure!(c.out_vocab <= c.vocab, "config: unpadded_vocab_size above vocab_size");
        ensure!(c.block > 0 && c.conv_k >= 2 && c.passes >= 1, "config: block, kernel or passes out of range");
        ensure!(
            c.loop_start > 0 && c.loop_start <= c.loop_end && c.loop_end <= c.n_layers,
            "config: loop layers {}..{} of {}",
            c.loop_start,
            c.loop_end,
            c.n_layers
        );
        ensure!(
            c.loop_start % c.block == 0 && c.loop_end % c.block == 0,
            "config: the loop must start and end on attention-residual block boundaries"
        );
        ensure!(
            c.full_heads.0 % c.full_heads.1 == 0 && c.swa_heads.0 % c.swa_heads.1 == 0,
            "config: heads per kv head"
        );
        ensure!(
            t.get("n_shared_experts").and_then(|x| x.as_u64()).unwrap_or(0) == 0,
            "config: shared experts are not supported"
        );
        if let Some(l) = lp {
            ensure!(
                !l.get("is_varloop").and_then(|x| x.as_bool()).unwrap_or(false),
                "config: the variable-loop variant is not supported"
            );
        }
        Ok(c)
    }

    /// (query heads, kv heads, head width) of layer `l`.
    pub fn heads(&self, l: usize) -> (usize, usize, usize) {
        if self.sliding[l] {
            self.swa_heads
        } else {
            self.full_heads
        }
    }

    pub fn in_loop(&self, l: usize) -> bool {
        self.loop_start <= l && l < self.loop_end
    }

    /// Layer visits a token: every layer once, the loop's again per extra pass.
    pub fn visits(&self) -> usize {
        self.n_layers + (self.passes - 1) * (self.loop_end - self.loop_start)
    }

    /// The cache layer `l` uses in pass `p`: its own index in the first pass (and outside the
    /// loop), then the loop's layers again after the model's, pass by pass.
    pub fn visit(&self, l: usize, p: usize) -> usize {
        if p == 0 {
            l
        } else {
            self.n_layers + (p - 1) * (self.loop_end - self.loop_start) + (l - self.loop_start)
        }
    }
}
