// SPDX-License-Identifier: Apache-2.0
//! Binds the trunk of a Kimi hybrid model (Kimi K3, Kimi Linear): everything except the
//! routed experts.
//!
//! Every tensor's element count is checked against the config before a byte is used: a
//! shape the config did not predict means the two disagree, and every kernel downstream
//! would read the wrong strides while producing plausible numbers.
//!
//! PRECISION POLICY. Small vectors that kernels read elementwise (norms, conv taps, A_log,
//! dt_bias, AttnRes projections) are widened to f32; the router keeps its on-disk dtype.
//! Large matrices keep the checkpoint's bf16, or become per-row 8-bit (int8 or FP8 E4M3)
//! when the trunk policy asks for it. 8-bit is applied only to the big, well-conditioned
//! projections; the low-rank pieces that feed attention scores or the gates (f_a, f_b,
//! g_a, g_b, b_proj, kv_a, kv_b) stay bf16. They are a few percent of the bytes and the
//! place where rounding is least forgiving.

use anyhow::{bail, Result};

use super::K3Config;
use crate::st::{Dtype, SafeTensors};
use crate::tensor::{Mat, Precision};

/// The KDA output gate: full rank (K3) or low rank g_b(g_a(x)) (Kimi Linear).
pub enum KdaGate {
    Full(Mat),
    LowRank { a: Mat, b: Mat },
}

pub struct KdaW {
    pub q: Mat,
    pub k: Mat,
    pub v: Mat,
    pub gate: KdaGate,
    pub o: Mat,
    pub f_a: Mat,
    pub f_b: Mat,
    pub b: Mat,
    pub q_conv: Vec<f32>,
    pub k_conv: Vec<f32>,
    pub v_conv: Vec<f32>,
    /// PER HEAD (K3 ships head_dim floats zero-padded past num_heads; Kimi Linear ships H).
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub o_norm: Vec<f32>,
}

/// The MLA query path: LoRA q_a -> norm -> q_b (K3) or one projection (Kimi Linear).
pub enum MlaQuery {
    Lora { a: Mat, norm: Vec<f32>, b: Mat },
    Direct(Mat),
}

pub struct MlaW {
    pub q: MlaQuery,
    pub kv_a: Mat,
    pub kv_a_norm: Vec<f32>,
    /// `[H * (qk_nope + v_head)][kv_lora]`: per head, qk_nope rows of W_UK then v_head rows
    /// of W_UV. The absorbed attention reads them as two separate matrices.
    pub kv_b: Mat,
    pub o: Mat,
    pub g: Option<Mat>,
}

/// K3's latent space around the routed experts.
pub struct LatentW {
    pub down: Mat,
    pub up: Mat,
    pub norm: Option<Vec<f32>>,
}

pub struct MoeW {
    /// Router `[n_experts][hidden]`, at its on-disk dtype. Selection is too discrete to round.
    pub gate: Mat,
    pub bias: Vec<f32>,
    pub latent: Option<LatentW>,
    pub sh_gate: Mat,
    pub sh_up: Mat,
    pub sh_down: Mat,
}

pub struct DenseW {
    pub gate: Mat,
    pub up: Mat,
    pub down: Mat,
}

pub enum Attn {
    Kda(Box<KdaW>),
    Mla(Box<MlaW>),
}

pub enum Ffn {
    Dense(Box<DenseW>),
    Moe(Box<MoeW>),
}

/// AttnRes scoring vectors for one layer, folded at load: norm.weight * proj.weight.
pub struct ResFolds {
    pub attn: Vec<f32>,
    pub mlp: Vec<f32>,
}

pub struct LayerW {
    pub in_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    /// Present when the model uses Block Attention Residuals.
    pub folds: Option<ResFolds>,
    pub attn: Attn,
    pub ffn: Ffn,
}

pub struct TrunkW {
    /// Kept bf16 always; one row is gathered per token.
    pub embed: Mat,
    pub lm_head: Mat,
    pub final_norm: Vec<f32>,
    pub fold_out: Option<Vec<f32>>,
    pub layers: Vec<LayerW>,
}

struct Binder<'a> {
    st: &'a SafeTensors,
    trunk: Precision,
    errors: Vec<String>,
}

impl Binder<'_> {
    fn vec(&mut self, name: &str, want: usize) -> Vec<f32> {
        self.vec_take(name, &[want], want)
    }

    /// Read a vector whose element count is one of `wants`, keeping the first `take`.
    fn vec_take(&mut self, name: &str, wants: &[usize], take: usize) -> Vec<f32> {
        let Some(v) = self.st.get(name) else {
            self.errors.push(format!("missing {name}"));
            return vec![0.0; take];
        };
        if !wants.contains(&v.numel()) || v.numel() < take {
            self.errors.push(format!("{name}: {} elements, config predicts {wants:?}", v.numel()));
            return vec![0.0; take];
        }
        match v.to_f32() {
            Ok(mut x) => {
                if x[take..].iter().any(|&t| t != 0.0) {
                    self.errors.push(format!("{name}: padding past element {take} is not zero"));
                }
                x.truncate(take);
                x
            }
            Err(e) => {
                self.errors.push(e.to_string());
                vec![0.0; take]
            }
        }
    }

    /// A large matrix. `q8_ok` marks the projections an 8-bit trunk policy may quantise.
    fn mat(&mut self, name: &str, rows: usize, cols: usize, q8_ok: bool) -> Mat {
        let Some(v) = self.st.get(name) else {
            self.errors.push(format!("missing {name}"));
            return Mat::f32(0, 0, vec![]);
        };
        if v.numel() != rows * cols {
            self.errors.push(format!("{name}: {} elements, config predicts [{rows}][{cols}]", v.numel()));
            return Mat::f32(0, 0, vec![]);
        }
        match v.to_bf16() {
            Ok(w) => {
                let m = Mat::bf16(rows, cols, w);
                if q8_ok {
                    m.convert(self.trunk)
                } else {
                    m
                }
            }
            Err(e) => {
                self.errors.push(e.to_string());
                Mat::f32(0, 0, vec![])
            }
        }
    }

    /// A matrix kept at its on-disk dtype (F32 stays F32, BF16 stays BF16): the router,
    /// whose selection is too discrete to round and whose bf16 copy is already exact.
    fn mat_native(&mut self, name: &str, rows: usize, cols: usize) -> Mat {
        match self.st.get(name).map(|v| v.dtype()) {
            Some(Dtype::BF16) => self.mat(name, rows, cols, false),
            _ => {
                let v = self.vec(name, rows * cols);
                Mat::f32(rows, cols, v)
            }
        }
    }

    fn finish<T>(self, value: T) -> Result<T> {
        if self.errors.is_empty() {
            return Ok(value);
        }
        let n = self.errors.len();
        let shown: Vec<_> = self.errors.into_iter().take(12).collect();
        bail!("checkpoint does not match the config ({n} problem(s)):\n  {}", shown.join("\n  "))
    }
}

/// Everything outside the layers.
pub struct GlobalsW {
    pub embed: Mat,
    pub lm_head: Mat,
    pub final_norm: Vec<f32>,
    pub fold_out: Option<Vec<f32>>,
}

pub fn bind_globals(st: &SafeTensors, c: &K3Config) -> Result<GlobalsW> {
    let mut b = Binder { st, trunk: Precision::Bf16, errors: Vec::new() };
    let e = c.hidden;
    let p = &c.prefix;
    let embed = b.mat(&format!("{p}embed_tokens.weight"), c.vocab, e, false);
    let lm_head = if c.tie_word_embeddings {
        b.mat(&format!("{p}embed_tokens.weight"), c.vocab, e, false)
    } else {
        b.mat(&c.lm_head_name, c.vocab, e, false)
    };
    let final_norm = b.vec(&format!("{p}norm.weight"), e);
    let fold_out = c.attn_res.then(|| {
        fold(
            &b.vec(&format!("{p}output_attn_res_norm.weight"), e),
            &b.vec(&format!("{p}output_attn_res_proj.weight"), e),
        )
    });
    b.finish(GlobalsW { embed, lm_head, final_norm, fold_out })
}

/// One layer's trunk. Binding a layer at a time keeps host memory at one layer when the
/// weights are streamed into an executor.
pub fn bind_layer(st: &SafeTensors, c: &K3Config, l: usize, trunk: Precision) -> Result<LayerW> {
    let mut b = Binder { st, trunk, errors: Vec::new() };
    let e = c.hidden;
    let lp = format!("{}layers.{l}.", c.prefix);
    let in_norm = b.vec(&format!("{lp}input_layernorm.weight"), e);
    let post_norm = b.vec(&format!("{lp}post_attention_layernorm.weight"), e);
    let folds = c.attn_res.then(|| ResFolds {
        attn: fold(
            &b.vec(&format!("{lp}self_attention_res_norm.weight"), e),
            &b.vec(&format!("{lp}self_attention_res_proj.weight"), e),
        ),
        mlp: fold(&b.vec(&format!("{lp}mlp_res_norm.weight"), e), &b.vec(&format!("{lp}mlp_res_proj.weight"), e)),
    });

    let a = format!("{lp}self_attn.");
    let attn = if c.is_mla(l) {
        let qh = c.qk_nope + c.qk_rope;
        let h = c.n_heads;
        let q = if c.q_lora > 0 {
            MlaQuery::Lora {
                a: b.mat(&format!("{a}q_a_proj.weight"), c.q_lora, e, true),
                norm: b.vec(&format!("{a}q_a_layernorm.weight"), c.q_lora),
                b: b.mat(&format!("{a}q_b_proj.weight"), h * qh, c.q_lora, true),
            }
        } else {
            MlaQuery::Direct(b.mat(&format!("{a}q_proj.weight"), h * qh, e, true))
        };
        Attn::Mla(Box::new(MlaW {
            q,
            kv_a: b.mat(&format!("{a}kv_a_proj_with_mqa.weight"), c.kv_lora + c.qk_rope, e, false),
            kv_a_norm: b.vec(&format!("{a}kv_a_layernorm.weight"), c.kv_lora),
            kv_b: b.mat(&format!("{a}kv_b_proj.weight"), h * (c.qk_nope + c.v_head), c.kv_lora, false),
            o: b.mat(&format!("{a}o_proj.weight"), e, h * c.v_head, true),
            g: c.mla_out_gate.then(|| b.mat(&format!("{a}g_proj.weight"), h * c.v_head, e, true)),
        }))
    } else {
        let pp = c.kda_proj();
        let d = c.kda_head_dim;
        let gate = if c.kda_gate_lowrank {
            KdaGate::LowRank {
                a: b.mat(&format!("{a}g_a_proj.weight"), d, e, false),
                b: b.mat(&format!("{a}g_b_proj.weight"), pp, d, false),
            }
        } else {
            KdaGate::Full(b.mat(&format!("{a}g_proj.weight"), pp, e, true))
        };
        Attn::Kda(Box::new(KdaW {
            q: b.mat(&format!("{a}q_proj.weight"), pp, e, true),
            k: b.mat(&format!("{a}k_proj.weight"), pp, e, true),
            v: b.mat(&format!("{a}v_proj.weight"), pp, e, true),
            gate,
            o: b.mat(&format!("{a}o_proj.weight"), e, pp, true),
            f_a: b.mat(&format!("{a}f_a_proj.weight"), d, e, false),
            f_b: b.mat(&format!("{a}f_b_proj.weight"), pp, d, false),
            b: b.mat(&format!("{a}b_proj.weight"), c.kda_heads, e, false),
            q_conv: b.vec(&format!("{a}q_conv1d.weight"), pp * c.conv_k),
            k_conv: b.vec(&format!("{a}k_conv1d.weight"), pp * c.conv_k),
            v_conv: b.vec(&format!("{a}v_conv1d.weight"), pp * c.conv_k),
            a_log: b.vec_take(&format!("{a}A_log"), &[d, c.kda_heads], c.kda_heads),
            dt_bias: b.vec(&format!("{a}dt_bias"), pp),
            o_norm: b.vec(&format!("{a}o_norm.weight"), d),
        }))
    };

    let ffn = if c.is_dense(l) {
        let m = format!("{lp}mlp.");
        Ffn::Dense(Box::new(DenseW {
            gate: b.mat(&format!("{m}gate_proj.weight"), c.dense_inter, e, true),
            up: b.mat(&format!("{m}up_proj.weight"), c.dense_inter, e, true),
            down: b.mat(&format!("{m}down_proj.weight"), e, c.dense_inter, true),
        }))
    } else {
        let m = format!("{lp}block_sparse_moe.");
        let si = c.moe_inter * c.n_shared;
        let latent = c.latent_moe.then(|| LatentW {
            down: b.mat(&format!("{m}routed_expert_down_proj.weight"), c.latent, e, true),
            up: b.mat(&format!("{m}routed_expert_up_proj.weight"), e, c.latent, true),
            norm: c.latent_norm.then(|| b.vec(&format!("{m}routed_expert_norm.weight"), c.latent)),
        });
        Ffn::Moe(Box::new(MoeW {
            gate: b.mat_native(&format!("{m}gate.weight"), c.n_experts, e),
            bias: b.vec(&format!("{m}gate.e_score_correction_bias"), c.n_experts),
            latent,
            sh_gate: b.mat(&format!("{m}shared_experts.gate_proj.weight"), si, e, true),
            sh_up: b.mat(&format!("{m}shared_experts.up_proj.weight"), si, e, true),
            sh_down: b.mat(&format!("{m}shared_experts.down_proj.weight"), e, si, true),
        }))
    };
    b.finish(LayerW { in_norm, post_norm, folds, attn, ffn })
}

pub fn bind(st: &SafeTensors, c: &K3Config, trunk: Precision) -> Result<TrunkW> {
    let g = bind_globals(st, c)?;
    let layers = (0..c.n_layers).map(|l| bind_layer(st, c, l, trunk)).collect::<Result<Vec<_>>>()?;
    Ok(TrunkW { embed: g.embed, lm_head: g.lm_head, final_norm: g.final_norm, fold_out: g.fold_out, layers })
}

fn fold(norm: &[f32], proj: &[f32]) -> Vec<f32> {
    norm.iter().zip(proj).map(|(a, b)| a * b).collect()
}

impl TrunkW {
    /// Bytes held by the transformer layers (excluding embed/lm_head).
    pub fn layer_bytes(&self) -> usize {
        let mut n = 0;
        for l in &self.layers {
            let mut add = |m: &Mat| n += m.nbytes();
            match &l.attn {
                Attn::Kda(k) => {
                    [&k.q, &k.k, &k.v, &k.o, &k.f_a, &k.f_b, &k.b].into_iter().for_each(&mut add);
                    match &k.gate {
                        KdaGate::Full(g) => add(g),
                        KdaGate::LowRank { a, b } => {
                            add(a);
                            add(b)
                        }
                    }
                }
                Attn::Mla(m) => {
                    [&m.kv_a, &m.kv_b, &m.o].into_iter().for_each(&mut add);
                    match &m.q {
                        MlaQuery::Lora { a, b, .. } => {
                            add(a);
                            add(b)
                        }
                        MlaQuery::Direct(q) => add(q),
                    }
                    if let Some(g) = &m.g {
                        add(g)
                    }
                }
            }
            match &l.ffn {
                Ffn::Dense(d) => [&d.gate, &d.up, &d.down].into_iter().for_each(&mut add),
                Ffn::Moe(m) => {
                    [&m.gate, &m.sh_gate, &m.sh_up, &m.sh_down].into_iter().for_each(&mut add);
                    if let Some(lt) = &m.latent {
                        add(&lt.down);
                        add(&lt.up)
                    }
                }
            }
        }
        n
    }
}
