// SPDX-License-Identifier: Apache-2.0
//! LoRA adapters on the tiny Qwen3-MoE. Merged at load (`--lora-merge`): an adapter of zeros leaves
//! the logits as they were; a real one moves them through attention and the routed experts; an
//! adapter matrix no weight takes is refused. Unmerged (slots, many at once): the logits of the
//! merged adapter up to its bf16 rounding; a base sequence in the same step as one with an adapter
//! gets the base model's logits; through the engine, base and two adapters at once each give their
//! solo reply, and an adapter never takes the base model's prefix cache.

use std::path::{Path, PathBuf};

use fullm::models::{self, BackendKind, CacheDtype, LoadOptions, SeqChunk};
use fullm::runtime::{Engine, EngineConfig, Event, Request, SamplingParams};
use fullm::tensor::Precision;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny_qwen3_moe")
}

fn write_adapter(dir: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
    std::fs::create_dir_all(dir).unwrap();
    let cfg = serde_json::json!({"peft_type": "LORA", "r": 4, "lora_alpha": 8, "bias": "none", "target_modules": ["q_proj", "o_proj", "gate_proj", "down_proj"]});
    std::fs::write(dir.join("adapter_config.json"), cfg.to_string()).unwrap();
    let (mut header, mut data) = (serde_json::Map::new(), Vec::new());
    for (name, shape, v) in tensors {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [data.len(), data.len() + bytes.len()]}),
        );
        data.extend(bytes);
    }
    let mut h = serde_json::Value::Object(header).to_string().into_bytes();
    h.resize(h.len().div_ceil(8) * 8, b' ');
    let mut f = (h.len() as u64).to_le_bytes().to_vec();
    f.extend(h);
    f.extend(data);
    std::fs::write(dir.join("adapter_model.safetensors"), f).unwrap();
}

/// q_proj and o_proj of every layer, gate_proj and down_proj of expert 1 of every layer; B scaled by `b`.
fn adapter(b: f32, extra: bool) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture().join("config.json")).unwrap()).unwrap();
    let n = |k: &str| cfg[k].as_u64().unwrap() as usize;
    let (hidden, q, inter, layers) = (
        n("hidden_size"),
        n("num_attention_heads") * n("head_dim"),
        n("moe_intermediate_size"),
        n("num_hidden_layers"),
    );
    let r = 4;
    let mut seed = 7u64;
    let mut rnd = |len: usize, s: f32| -> Vec<f32> {
        (0..len)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * s
            })
            .collect()
    };
    let mut t = Vec::new();
    for l in 0..layers {
        for (m, rows, cols) in [
            (format!("model.layers.{l}.self_attn.q_proj"), q, hidden),
            (format!("model.layers.{l}.self_attn.o_proj"), hidden, q),
            (format!("model.layers.{l}.mlp.experts.1.gate_proj"), inter, hidden),
            (format!("model.layers.{l}.mlp.experts.1.down_proj"), hidden, inter),
        ] {
            t.push((format!("base_model.model.{m}.lora_A.weight"), vec![r, cols], rnd(r * cols, 0.5)));
            t.push((format!("base_model.model.{m}.lora_B.weight"), vec![rows, r], rnd(rows * r, b)));
        }
    }
    if extra {
        t.push((
            "base_model.model.model.layers.0.self_attn.nope_proj.lora_A.weight".into(),
            vec![r, hidden],
            rnd(r * hidden, 0.5),
        ));
        t.push((
            "base_model.model.model.layers.0.self_attn.nope_proj.lora_B.weight".into(),
            vec![hidden, r],
            rnd(hidden * r, 0.5),
        ));
    }
    t
}

fn opts(backend: BackendKind, lora: Option<PathBuf>, slots: usize) -> LoadOptions {
    LoadOptions {
        trunk: Precision::Bf16,
        max_seqs: 4,
        cache_tokens: 512,
        page: 4,
        backend,
        kv_dtype: CacheDtype::F32,
        step_tokens: 64,
        max_rows: 64,
        max_context: 128,
        threads: 4,
        lora_merge: lora,
        max_loras: slots,
        max_lora_rank: if slots > 0 { 8 } else { 0 },
        ..Default::default()
    }
}

/// Attention only (an MoE model's experts take a merged adapter only): q, k, v, o of every layer.
fn attention_adapter(seed0: u64) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture().join("config.json")).unwrap()).unwrap();
    let n = |k: &str| cfg[k].as_u64().unwrap() as usize;
    let (hidden, q, kv, layers) = (
        n("hidden_size"),
        n("num_attention_heads") * n("head_dim"),
        n("num_key_value_heads") * n("head_dim"),
        n("num_hidden_layers"),
    );
    let r = 4;
    let mut seed = seed0;
    let mut rnd = |len: usize, s: f32| -> Vec<f32> {
        (0..len)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * s
            })
            .collect()
    };
    let mut t = Vec::new();
    for l in 0..layers {
        for (m, rows, cols) in
            [("q_proj", q, hidden), ("k_proj", kv, hidden), ("v_proj", kv, hidden), ("o_proj", hidden, q)]
        {
            let m = format!("model.layers.{l}.self_attn.{m}");
            t.push((format!("base_model.model.{m}.lora_A.weight"), vec![r, cols], rnd(r * cols, 0.5)));
            t.push((format!("base_model.model.{m}.lora_B.weight"), vec![rows, r], rnd(rows * r, 0.5)));
        }
    }
    t
}

fn backends() -> Vec<BackendKind> {
    let mut b = vec![BackendKind::CPU];
    if fullm_kernels::exec::cuda_available() {
        b.push(BackendKind::CUDA);
    }
    b
}

const PROMPT: [u32; 10] = [3, 14, 15, 92, 65, 35, 89, 79, 32, 38];

#[test]
fn lora_served_unmerged() {
    let root = std::env::temp_dir().join(format!("fullm_lora_slots_{}", std::process::id()));
    let (a1, a2) = (root.join("a1"), root.join("a2"));
    write_adapter(&a1, &attention_adapter(11));
    write_adapter(&a2, &attention_adapter(22));
    for be in backends() {
        let exact = be == BackendKind::CPU; // CUDA rounds a step with adapters' q/k/v apart (fp32, not bf16)
        let run1 = |m: &mut Box<dyn models::Model>, lora: Option<usize>| {
            let s = m.alloc_seq().unwrap();
            m.set_seq_lora(s, lora).unwrap();
            let out = m
                .forward(&[SeqChunk { seq: s, pos: 0, tokens: &PROMPT, all_logits: true, draft: 0 }])
                .unwrap()
                .remove(0);
            m.free_seq(s);
            out
        };
        let mut base_m = models::load(&fixture(), &opts(be.clone(), None, 0)).unwrap();
        let base = run1(&mut base_m, None);
        drop(base_m);
        let mut merged_m = models::load(&fixture(), &opts(be.clone(), Some(a1.clone()), 0)).unwrap();
        let merged = run1(&mut merged_m, None);
        drop(merged_m);
        let mut m = models::load(&fixture(), &opts(be.clone(), None, 2)).unwrap();
        let ad1 = models::lora::Adapter::load(&a1).unwrap();
        m.set_lora(1, &ad1).unwrap();
        let un = run1(&mut m, Some(1));
        let d = rel(&un, &merged);
        assert!(d < 2e-2 && rel(&un, &base) > 1e-3, "{be:?}: unmerged vs merged rel-L2 {d:.2e}");
        // one step, two sequences: the adapter's and the base model's
        let (s1, s2) = (m.alloc_seq().unwrap(), m.alloc_seq().unwrap());
        m.set_seq_lora(s1, Some(1)).unwrap();
        let out = m
            .forward(&[
                SeqChunk { seq: s1, pos: 0, tokens: &PROMPT, all_logits: true, draft: 0 },
                SeqChunk { seq: s2, pos: 0, tokens: &PROMPT, all_logits: true, draft: 0 },
            ])
            .unwrap();
        let (with, without) = (&out[0], &out[1]);
        if exact {
            assert!(with == &un && without == &base, "{be:?}: a mixed step changed a row");
        } else {
            assert!(rel(with, &un) < 1e-2 && rel(without, &base) < 1e-2, "{be:?}: a mixed step moved a row");
        }
        m.free_seq(s1);
        m.free_seq(s2);
        // cleared: the base model again
        m.clear_lora(1).unwrap();
        let cleared = run1(&mut m, Some(1));
        assert!(
            rel(&cleared, &base) < if exact { 1e-6 } else { 1e-2 },
            "{be:?}: a cleared slot still moved the logits"
        );
        // an MoE model's experts: merged only
        let ex_dir = root.join("experts");
        write_adapter(&ex_dir, &adapter(0.5, false));
        let e = m.set_lora(0, &models::lora::Adapter::load(&ex_dir).unwrap()).err().expect("experts unmerged accepted");
        assert!(format!("{e:#}").contains("merged"), "{e:#}");
        drop(m);

        // through the engine: base and two adapters at once, each its solo reply; no cache across adapters
        let mut m = models::load(&fixture(), &opts(be.clone(), None, 2)).unwrap();
        m.set_lora(0, &models::lora::Adapter::load(&a1).unwrap()).unwrap();
        m.set_lora(1, &models::lora::Adapter::load(&a2).unwrap()).unwrap();
        let h = Engine::new(m, EngineConfig::default()).spawn().unwrap();
        let run = |lora: Option<usize>| -> (Vec<u32>, usize) {
            let p = SamplingParams { max_tokens: 8, lora, ..Default::default() };
            let mut rx = h.submit(Request { prompt: PROMPT.to_vec(), params: p }).unwrap();
            let mut toks = Vec::new();
            loop {
                match rx.blocking_recv().unwrap() {
                    Event::Token(t) => toks.push(t),
                    Event::Finished { usage, .. } => return (toks, usage.cached_tokens),
                    _ => {}
                }
            }
        };
        let solo: Vec<(Vec<u32>, usize)> = [None, Some(0), Some(1)].into_iter().map(run).collect();
        assert_eq!(solo[1].1, 0, "{be:?}: the adapter's first request took the base model's cache");
        assert!(solo[0].0 != solo[1].0 || solo[1].0 != solo[2].0, "{be:?}: the adapters changed nothing");
        let rxs: Vec<_> = [None, Some(0), Some(1), Some(0)]
            .into_iter()
            .map(|lora| {
                h.submit(Request {
                    prompt: PROMPT.to_vec(),
                    params: SamplingParams { max_tokens: 8, lora, ..Default::default() },
                })
                .unwrap()
            })
            .collect();
        for (i, mut rx) in rxs.into_iter().enumerate() {
            let mut toks = Vec::new();
            loop {
                match rx.blocking_recv().unwrap() {
                    Event::Token(t) => toks.push(t),
                    Event::Finished { .. } => break,
                    _ => {}
                }
            }
            let want = &solo[[0, 1, 2, 1][i]].0;
            if exact {
                assert_eq!(&toks, want, "{be:?}: request {i} differs from its solo run");
            }
        }
        assert!(h
            .submit(Request { prompt: PROMPT.to_vec(), params: SamplingParams { lora: Some(5), ..Default::default() } })
            .is_err());
    }
    std::fs::remove_dir_all(&root).ok();
}

fn logits(lora: Option<PathBuf>) -> anyhow::Result<Vec<f32>> {
    let opts = LoadOptions {
        trunk: Precision::Bf16,
        max_seqs: 2,
        cache_tokens: 256,
        page: 4,
        backend: BackendKind::CPU,
        kv_dtype: CacheDtype::F32,
        step_tokens: 64,
        max_rows: 64,
        max_context: 128,
        threads: 4,
        lora_merge: lora,
        ..Default::default()
    };
    let mut m = models::load(&fixture(), &opts)?;
    let s = m.alloc_seq().unwrap();
    let toks: Vec<u32> = vec![3, 14, 15, 92, 65, 35, 89, 79, 32, 38];
    Ok(m.forward(&[SeqChunk { seq: s, pos: 0, tokens: &toks, all_logits: true, draft: 0 }])?.remove(0))
}

fn rel(a: &[f32], b: &[f32]) -> f32 {
    let d: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
    d / b.iter().map(|y| y * y).sum::<f32>().sqrt()
}

#[test]
fn lora_merged_at_load() {
    let root = std::env::temp_dir().join(format!("fullm_lora_{}", std::process::id()));
    let base = logits(None).unwrap();
    let zero = root.join("zero");
    write_adapter(&zero, &adapter(0.0, false));
    assert_eq!(logits(Some(zero)).unwrap(), base, "an adapter of zeros changed the logits");
    let real = root.join("real");
    write_adapter(&real, &adapter(0.5, false));
    let d = rel(&logits(Some(real)).unwrap(), &base);
    assert!(d > 1e-3, "the adapter did not reach the logits (rel-L2 {d:.2e})");
    let bad = root.join("bad");
    write_adapter(&bad, &adapter(0.5, true));
    let e = logits(Some(bad)).err().expect("an adapter matrix for no weight was accepted");
    assert!(format!("{e:#}").contains("nope_proj"), "{e:#}");
    std::fs::remove_dir_all(&root).ok();
}
