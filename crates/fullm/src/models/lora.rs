// SPDX-License-Identifier: Apache-2.0
//! LoRA adapters in PEFT's format (adapter_config.json + adapter_model.safetensors): each target
//! linear layer W [out][in] gains scale x B [out][r] x A [r][in], scale = lora_alpha / r
//! (lora_alpha / sqrt(r) with use_rslora; alpha_pattern / rank_pattern per module). Embedding
//! adapters (`lora_embedding_A` [r][vocab], `lora_embedding_B` [hidden][r]) add (B A)^T to the
//! table. `--lora-merge` adds every delta to its weight at load, in f32, before the weight is
//! rounded or quantised to its storage format: the model then runs at the base model's speed.
//!
//! Refused rather than misread: DoRA, trained biases, `modules_to_save`, `fan_in_fan_out`, and
//! any adapter tensor no weight of the model took (a name the loader does not use, experts that
//! come pre-quantised). `base_layer.weight` copies (PEFT saves the frozen embedding and head of a
//! model whose vocabulary was resized) are skipped; a delta for a resized vocabulary covers the
//! first rows of the model's table.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Mutex;

use anyhow::{bail, ensure, Context, Result};
use rayon::prelude::*;
use serde_json::Value;

use crate::st::SafeTensors;

/// One weight's low-rank update: W[o][i] += scale x sum_k b[o][k] a[k][i].
pub struct Delta {
    pub rank: usize,
    /// [rank][cols]
    pub a: Vec<f32>,
    /// [rows][rank]
    pub b: Vec<f32>,
    pub scale: f32,
    pub rows: usize,
    pub cols: usize,
}

pub struct Adapter {
    pub dir: std::path::PathBuf,
    /// The base model's weight name ("model.layers.0.self_attn.q_proj.weight") -> its delta.
    deltas: HashMap<String, Delta>,
    used: Mutex<HashSet<String>>,
}

impl Adapter {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("adapter_config.json");
        let cfg: Value = serde_json::from_str(
            &std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?,
        )
        .with_context(|| format!("parsing {}", path.display()))?;
        let kind = cfg.get("peft_type").and_then(Value::as_str).unwrap_or("LORA");
        ensure!(kind == "LORA", "{}: peft_type {kind:?}, not LORA", dir.display());
        ensure!(cfg.get("use_dora") != Some(&Value::Bool(true)), "{}: DoRA adapters are not supported", dir.display());
        ensure!(
            cfg.get("fan_in_fan_out") != Some(&Value::Bool(true)),
            "{}: fan_in_fan_out adapters are not supported",
            dir.display()
        );
        let bias = cfg.get("bias").and_then(Value::as_str).unwrap_or("none");
        ensure!(bias == "none", "{}: trained biases (bias = {bias:?}) are not supported", dir.display());
        let saved = cfg.get("modules_to_save").filter(|v| !v.is_null() && v.as_array().is_none_or(|a| !a.is_empty()));
        ensure!(saved.is_none(), "{}: modules_to_save ({}) are not supported", dir.display(), saved.unwrap());
        let alpha = cfg.get("lora_alpha").and_then(Value::as_f64).context("adapter_config.json: no lora_alpha")?;
        let rslora = cfg.get("use_rslora") == Some(&Value::Bool(true));
        let alpha_pattern: Vec<(String, f64)> = cfg
            .get("alpha_pattern")
            .and_then(Value::as_object)
            .map(|m| m.iter().filter_map(|(k, v)| v.as_f64().map(|a| (k.clone(), a))).collect())
            .unwrap_or_default();

        let st = SafeTensors::open_dir(dir)?;
        // module -> (A, B) as given, and whether it is an embedding (A [r][vocab], B [hidden][r])
        let mut parts: HashMap<String, (Option<(Vec<f32>, Vec<usize>)>, Option<(Vec<f32>, Vec<usize>)>, bool)> =
            HashMap::new();
        let mut skipped = 0;
        for name in st.names() {
            let rest = name.strip_prefix("base_model.model.").unwrap_or(name);
            let rest = rest.replace(".default.", ".").replace(".default", "");
            let (module, which, embedding) = if let Some(m) = rest.strip_suffix(".lora_A.weight") {
                (m, 'A', false)
            } else if let Some(m) = rest.strip_suffix(".lora_B.weight") {
                (m, 'B', false)
            } else if let Some(m) = rest.strip_suffix(".lora_embedding_A") {
                (m, 'A', true)
            } else if let Some(m) = rest.strip_suffix(".lora_embedding_B") {
                (m, 'B', true)
            } else if rest.ends_with(".base_layer.weight") {
                skipped += 1;
                continue;
            } else {
                bail!("{}: adapter tensor {name:?} is not a LoRA A / B matrix", dir.display());
            };
            let v = st.need(name)?;
            let t = (v.to_f32()?, v.shape().to_vec());
            ensure!(t.1.len() == 2, "{name}: shape {:?} is not a matrix", t.1);
            let e = parts.entry(module.to_string()).or_insert((None, None, embedding));
            if which == 'A' {
                e.0 = Some(t);
            } else {
                e.1 = Some(t);
            }
        }
        if skipped > 0 {
            tracing::info!("{}: skipped {skipped} frozen base-layer copies (base_layer.weight)", dir.display());
        }
        let mut deltas = HashMap::new();
        for (module, (a, b, embedding)) in parts {
            let ((a, sa), (b, sb)) = (
                a.with_context(|| format!("{module}: no lora_A"))?,
                b.with_context(|| format!("{module}: no lora_B"))?,
            );
            ensure!(sa.len() == 2 && sb.len() == 2, "{module}: lora_A is {sa:?}, lora_B {sb:?}: matrices expected");
            ensure!(sa[0] > 0, "{module}: rank 0");
            // as W[o][i] += scale x sum_k B[o][k] A[k][i]: an embedding's (B A)^T is A^T B^T
            let (a, b, rank, rows, cols) = if embedding {
                let (r, vocab, hidden) = (sa[0], sa[1], sb[0]);
                ensure!(sb[1] == r, "{module}: lora_embedding_B is {sb:?}, lora_embedding_A {sa:?}");
                (transpose(&b, hidden, r), transpose(&a, r, vocab), r, vocab, hidden)
            } else {
                let (r, cols, rows) = (sa[0], sa[1], sb[0]);
                ensure!(sb[1] == r, "{module}: lora_B is {sb:?}, lora_A {sa:?}");
                (a, b, r, rows, cols)
            };
            let al = alpha_pattern
                .iter()
                .find(|(k, _)| module == *k || module.ends_with(&format!(".{k}")))
                .map_or(alpha, |&(_, a)| a);
            let scale = if rslora { al / (rank as f64).sqrt() } else { al / rank as f64 } as f32;
            deltas.insert(format!("{module}.weight"), Delta { rank, a, b, scale, rows, cols });
        }
        ensure!(!deltas.is_empty(), "{}: the adapter has no LoRA matrices", dir.display());
        Ok(Adapter { dir: dir.to_path_buf(), deltas, used: Mutex::new(HashSet::new()) })
    }

    pub fn deltas(&self) -> impl Iterator<Item = (&String, &Delta)> {
        self.deltas.iter()
    }

    /// The largest rank of its matrices.
    pub fn max_rank(&self) -> usize {
        self.deltas.values().map(|d| d.rank).max().unwrap_or(0)
    }

    pub fn has(&self, weight: &str) -> bool {
        self.deltas.contains_key(weight)
    }

    /// Count the deltas of the weights `which` picks as merged (they came from a cache recorded
    /// with them merged).
    pub fn mark_used(&self, which: impl Fn(&str) -> bool) {
        let mut used = self.used.lock().unwrap();
        for k in self.deltas.keys().filter(|k| which(k)) {
            used.insert(k.clone());
        }
    }

    /// Add `weight`'s delta (if the adapter has one) to `w`, bf16 [rows][cols], in f32 and back
    /// to bf16 (nearest even). A delta of fewer rows (a resized vocabulary's embedding or head)
    /// covers the first rows. True when it had one.
    pub fn merge_bf16(&self, weight: &str, w: &mut [u16], rows: usize, cols: usize) -> Result<bool> {
        let Some(d) = self.deltas.get(weight) else { return Ok(false) };
        ensure!(
            d.cols == cols
                && d.rows <= rows
                && (d.rows == rows || weight.contains("embed_tokens") || weight.starts_with("lm_head")),
            "{weight}: the adapter's delta is [{}][{}], the model's weight [{rows}][{cols}]",
            d.rows,
            d.cols
        );
        w[..d.rows * cols].par_chunks_mut(cols).enumerate().for_each(|(o, row)| {
            let mut acc: Vec<f32> = row.iter().map(|&h| f32::from_bits((h as u32) << 16)).collect();
            for k in 0..d.rank {
                let c = d.scale * d.b[o * d.rank + k];
                if c != 0.0 {
                    for (x, &a) in acc.iter_mut().zip(&d.a[k * cols..(k + 1) * cols]) {
                        *x += c * a;
                    }
                }
            }
            for (h, x) in row.iter_mut().zip(acc) {
                *h = bf16_rne(x);
            }
        });
        self.used.lock().unwrap().insert(weight.to_string());
        Ok(true)
    }

    /// Every delta was added to some weight (else the adapter does not fit what was loaded).
    pub fn check_all_used(&self) -> Result<()> {
        let used = self.used.lock().unwrap();
        let mut left: Vec<&String> = self.deltas.keys().filter(|k| !used.contains(*k)).collect();
        left.sort();
        ensure!(
            left.is_empty(),
            "{}: {} adapter matrices fit no weight of this model as loaded (e.g. {}; pre-quantised experts cannot take a merged adapter)",
            self.dir.display(),
            left.len(),
            left[..left.len().min(3)].iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
        );
        Ok(())
    }
}

fn transpose(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0f32; m.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = m[r * cols + c];
        }
    }
    t
}

fn bf16_rne(x: f32) -> u16 {
    let u = x.to_bits();
    if x.is_nan() {
        return ((u >> 16) | 0x40) as u16;
    }
    ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A PEFT adapter directory with the given tensors (f32) and config.
    pub(crate) fn write_adapter(dir: &Path, cfg: &Value, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("adapter_config.json"), cfg.to_string()).unwrap();
        let (mut header, mut data) = (serde_json::Map::new(), Vec::new());
        for (name, shape, v) in tensors {
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            header.insert(name.clone(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [data.len(), data.len() + bytes.len()]}));
            data.extend(bytes);
        }
        let mut h = Value::Object(header).to_string().into_bytes();
        h.resize(h.len().div_ceil(8) * 8, b' ');
        let mut f = (h.len() as u64).to_le_bytes().to_vec();
        f.extend(h);
        f.extend(data);
        std::fs::write(dir.join("adapter_model.safetensors"), f).unwrap();
    }

    #[test]
    fn merges_linear_and_embedding_deltas() {
        let dir = std::env::temp_dir().join(format!("fullm_lora_unit_{}", std::process::id()));
        // q: [3][2] gets alpha/r = 4/2 = 2 x B A; the embedding [vocab 3][hidden 2] gets 2 x (B A)^T
        let cfg = serde_json::json!({"peft_type": "LORA", "r": 2, "lora_alpha": 4, "bias": "none"});
        let a = vec![1.0, 0.0, 0.0, 1.0]; // [2][2] identity
        let b = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5]; // [3][2]
        let ea = vec![1.0, 0.0, 2.0, 0.0, 1.0, 0.0]; // [r 2][vocab 3]
        let eb = vec![1.0, 0.0, 0.0, 1.0]; // [hidden 2][r 2]
        write_adapter(
            &dir,
            &cfg,
            &[
                ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".into(), vec![2, 2], a),
                ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".into(), vec![3, 2], b),
                ("base_model.model.model.embed_tokens.lora_embedding_A".into(), vec![2, 3], ea),
                ("base_model.model.model.embed_tokens.lora_embedding_B".into(), vec![2, 2], eb),
            ],
        );
        let ad = Adapter::load(&dir).unwrap();
        let bf = |x: f32| (x.to_bits() >> 16) as u16;
        let back = |h: u16| f32::from_bits((h as u32) << 16);
        let mut q: Vec<u16> = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0].iter().map(|&x| bf(x)).collect();
        assert!(ad.merge_bf16("model.layers.0.self_attn.q_proj.weight", &mut q, 3, 2).unwrap());
        assert_eq!(q.iter().map(|&h| back(h)).collect::<Vec<_>>(), vec![2.0, 0.0, 1.0, 3.0, 1.0, 1.0]);
        assert!(ad.check_all_used().is_err(), "the embedding's delta was not used yet");
        let mut e: Vec<u16> = vec![bf(0.0); 6];
        assert!(ad.merge_bf16("model.embed_tokens.weight", &mut e, 3, 2).unwrap());
        // (B A)^T: row v = 2 x (A[0][v], A[1][v])
        assert_eq!(e.iter().map(|&h| back(h)).collect::<Vec<_>>(), vec![2.0, 0.0, 0.0, 2.0, 4.0, 0.0]);
        ad.check_all_used().unwrap();
        assert!(!ad.merge_bf16("model.layers.0.self_attn.k_proj.weight", &mut q, 3, 2).unwrap());
        // refused: DoRA
        write_adapter(&dir, &serde_json::json!({"r": 2, "lora_alpha": 4, "use_dora": true}), &[]);
        assert!(format!("{:#}", Adapter::load(&dir).err().unwrap()).contains("DoRA"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
