// SPDX-License-Identifier: Apache-2.0
//! Qwen3-MoE (the dense-attention executor with routed experts) against transformers on a tiny
//! random checkpoint (tools/make_tiny_qwen3_moe.py): every logit row of each prompt, greedy
//! tokens, and prefill chunkings against each other. Prompts go in chunks of at most 16 tokens,
//! the exact (GEMV) path: above that the CUDA trunk runs cuBLAS on bf16 activations.

/// Prefill `tokens` in chunks of `chunk`, returning every row's logits.
fn prefill(m: &mut dyn Model, s: usize, tokens: &[u32], chunk: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for (i, ch) in tokens.chunks(chunk).enumerate() {
        out.extend(
            m.forward(&[SeqChunk { seq: s, pos: i * chunk, tokens: ch, all_logits: true, draft: 0 }])
                .unwrap()
                .remove(0),
        );
    }
    out
}

use std::path::PathBuf;

use fullm::models::{self, BackendKind, CacheDtype, LoadOptions, Model, SeqChunk};
use fullm::tensor::Precision;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny_qwen3_moe")
}

struct Case {
    prompt: Vec<u32>,
    logits: Vec<Vec<f32>>,
    greedy: Vec<u32>,
}

fn cases() -> Vec<Case> {
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture().join("oracle.json")).unwrap()).unwrap();
    let ids =
        |x: &serde_json::Value| x.as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect::<Vec<_>>();
    v["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| Case {
            prompt: ids(&c["prompt"]),
            greedy: ids(&c["greedy"]),
            logits: c["logits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect())
                .collect(),
        })
        .collect()
}

fn backends() -> Vec<BackendKind> {
    let mut b = vec![BackendKind::CPU];
    if fullm_kernels::exec::cuda_available() {
        b.push(BackendKind::CUDA);
    }
    b
}

fn load(backend: BackendKind, fmt: Precision) -> Box<dyn Model> {
    let opts = LoadOptions {
        trunk: Precision::Bf16,
        max_seqs: 4,
        cache_tokens: 1024,
        page: 4,
        backend,
        kv_dtype: CacheDtype::F32,
        step_tokens: 64,
        max_rows: 64,
        max_context: 256,
        threads: 4,
        gpu_expert_format: fmt,
        ..Default::default()
    };
    models::load(&fixture(), &opts).expect("load tiny qwen3_moe")
}

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
    num / b.iter().map(|y| y * y).sum::<f32>().sqrt().max(1e-30)
}

fn argmax(v: &[f32]) -> u32 {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b }) as u32
}

#[test]
fn qwen3_moe_matches_transformers() {
    for be in backends() {
        let mut m = load(be.clone(), Precision::Bf16);
        let mut worst = 0.0f32;
        for c in cases() {
            let s = m.alloc_seq().unwrap();
            let out = prefill(m.as_mut(), s, &c.prompt, 16);
            let v = m.vocab();
            let errs: Vec<f32> =
                c.logits.iter().enumerate().map(|(r, want)| rel_l2(&out[r * v..(r + 1) * v], want)).collect();
            if std::env::var("QWEN3_MOE_DEBUG").is_ok() {
                eprintln!(
                    "{be:?} prompt len {}: per-row rel-L2 {:?}",
                    c.prompt.len(),
                    errs.iter().map(|e| format!("{e:.1e}")).collect::<Vec<_>>()
                );
            }
            worst = errs.iter().fold(worst, |w, &e| w.max(e));
            // greedy continuation, one token per step
            let (mut pos, mut next) = (c.prompt.len(), argmax(&out[(c.prompt.len() - 1) * v..]));
            let mut got = Vec::new();
            for _ in 0..c.greedy.len() {
                got.push(next);
                let l = m
                    .forward(&[SeqChunk { seq: s, pos, tokens: &[next], all_logits: false, draft: 0 }])
                    .unwrap()
                    .remove(0);
                pos += 1;
                next = argmax(&l);
            }
            m.free_seq(s);
            assert_eq!(got, c.greedy, "{be:?}: greedy tokens differ from transformers");
        }
        eprintln!("{be:?}: worst rel-L2 vs transformers {worst:.2e}, greedy identical");
        assert!(worst < 1e-4, "{be:?}: logits differ from transformers: {worst}");
    }
}

/// Resident experts as block-scaled fp8 (`--gpu-expert-format fp8b`), quantised by the executor at
/// load over the stored matrices. On the CPU backend (the CUDA tiles need expert rows of whole
/// 256-byte groups, which this model's 64-wide experts are not): the logits stay close to
/// transformers', differ from the bf16 experts' (something was quantised), and prefill chunking
/// is still bit-identical.
#[test]
fn qwen3_moe_fp8b_experts() {
    let c = &cases()[1];
    let run = |fmt: Precision, chunk: usize| {
        let mut m = load(BackendKind::CPU, fmt);
        let s = m.alloc_seq().unwrap();
        let out = prefill(m.as_mut(), s, &c.prompt, chunk);
        m.free_seq(s);
        out
    };
    let (b, f, f5) = (run(Precision::Bf16, 16), run(Precision::Fp8Block, 16), run(Precision::Fp8Block, 5));
    let want: Vec<f32> = c.logits.iter().flatten().copied().collect();
    let (e_f, e_b, d) = (rel_l2(&f, &want), rel_l2(&b, &want), rel_l2(&f, &b));
    eprintln!("fp8b experts: rel-L2 vs transformers {e_f:.2e} (bf16 experts {e_b:.2e}), vs bf16 experts {d:.2e}");
    assert!(d > 0.0, "fp8b experts gave the bf16 experts' logits: nothing was quantised");
    assert!(e_f < 5e-2, "fp8b experts too far from transformers: {e_f}");
    assert!(f == f5, "fp8b experts: chunking changed the logits");
}

#[test]
fn qwen3_moe_int4_experts() {
    let c = &cases()[1];
    let run = |fmt: Precision, chunk: usize| {
        let mut m = load(BackendKind::CPU, fmt);
        let s = m.alloc_seq().unwrap();
        let out = prefill(m.as_mut(), s, &c.prompt, chunk);
        m.free_seq(s);
        out
    };
    let (b, q, q5) = (run(Precision::Bf16, 16), run(Precision::Int4Group, 16), run(Precision::Int4Group, 5));
    let want: Vec<f32> = c.logits.iter().flatten().copied().collect();
    let (e_q, e_b, d) = (rel_l2(&q, &want), rel_l2(&b, &want), rel_l2(&q, &b));
    eprintln!("int4 experts: rel-L2 vs transformers {e_q:.2e} (bf16 experts {e_b:.2e}), vs bf16 experts {d:.2e}");
    assert!(d > 0.0, "int4 experts gave the bf16 experts' logits: nothing was quantised");
    assert!(e_q < 2e-1, "int4 experts too far from transformers: {e_q}");
    assert!(q == q5, "int4 experts: chunking changed the logits");
}

/// A GPTQ checkpoint's experts (`--int4-checkpoint`) are taken as they are: a checkpoint holding
/// exactly the values and scales the executor's own int4 rounding gives (GPTQ v1 layout: qweight
/// packed along the inputs, zero points stored as 7, g_idx k / 128; scales in F32 so nothing is
/// rounded on the way) yields the logits of `--gpu-expert-format int4`, bit for bit; and a
/// checkpoint with asymmetric zero points is refused.
#[test]
fn qwen3_moe_gptq_int4_experts() {
    let st = fullm::st::SafeTensors::open_dir(&fixture()).unwrap();
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture().join("config.json")).unwrap()).unwrap();
    let n = |k: &str| cfg[k].as_u64().unwrap() as usize;
    let (w, i, layers, experts) =
        (n("hidden_size"), n("moe_intermediate_size"), n("num_hidden_layers"), n("num_local_experts"));
    // name -> (dtype, shape, little-endian bytes)
    let mut tensors: Vec<(String, &str, Vec<usize>, Vec<u8>)> = Vec::new();
    let le32 = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    for l in 0..layers {
        for x in 0..experts {
            for (m, rows, cols) in [("gate_proj", i, w), ("up_proj", i, w), ("down_proj", w, i)] {
                let name = format!("model.layers.{l}.mlp.experts.{x}.{m}");
                let b = st.need(&format!("{name}.weight")).unwrap().to_bf16().unwrap();
                let groups = cols.div_ceil(128);
                // int4_group.h's rounding: a group's absolute maximum / 7, nearest (ties to even)
                let mut q = vec![0i32; rows * cols];
                let mut scale = vec![0f32; groups * rows];
                for r in 0..rows {
                    for g in 0..groups {
                        let ks = g * 128..((g + 1) * 128).min(cols);
                        let v = |k: usize| f32::from_bits((b[r * cols + k] as u32) << 16);
                        let amax = ks.clone().fold(0f32, |a, k| a.max(v(k).abs()));
                        let s = if amax > 0.0 { amax / 7.0 } else { 1.0 };
                        scale[g * rows + r] = s;
                        for k in ks {
                            q[r * cols + k] = ((v(k) / s).round_ties_even() as i32).clamp(-8, 7);
                        }
                    }
                }
                let mut qweight = vec![0u32; cols / 8 * rows];
                for r in 0..rows {
                    for k in 0..cols {
                        qweight[(k / 8) * rows + r] |= ((q[r * cols + k] + 8) as u32) << (4 * (k % 8));
                    }
                }
                let g_idx: Vec<u32> = (0..cols).map(|k| (k / 128) as u32).collect();
                tensors.push((format!("{name}.qweight"), "I32", vec![cols / 8, rows], le32(&qweight)));
                tensors.push((
                    format!("{name}.qzeros"),
                    "I32",
                    vec![groups, rows / 8],
                    le32(&vec![0x7777_7777; groups * rows / 8]),
                ));
                tensors.push((
                    format!("{name}.scales"),
                    "F32",
                    vec![groups, rows],
                    scale.iter().flat_map(|s| s.to_le_bytes()).collect(),
                ));
                tensors.push((format!("{name}.g_idx"), "I32", vec![cols], le32(&g_idx)));
            }
        }
    }
    let write = |dir: &std::path::Path, tensors: &[(String, &str, Vec<usize>, Vec<u8>)]| {
        std::fs::create_dir_all(dir).unwrap();
        let (mut header, mut data) = (serde_json::Map::new(), Vec::new());
        for (name, dt, shape, bytes) in tensors {
            header.insert(name.clone(), serde_json::json!({"dtype": dt, "shape": shape, "data_offsets": [data.len(), data.len() + bytes.len()]}));
            data.extend_from_slice(bytes);
        }
        let mut h = serde_json::Value::Object(header).to_string().into_bytes();
        h.resize(h.len().div_ceil(8) * 8, b' ');
        let mut f = (h.len() as u64).to_le_bytes().to_vec();
        f.extend(h);
        f.extend(data);
        std::fs::write(dir.join("model.safetensors"), f).unwrap();
        let qc = serde_json::json!({"bits": 4, "group_size": 128, "sym": true, "desc_act": false, "quant_method": "gptq", "checkpoint_format": "gptq"});
        std::fs::write(dir.join("config.json"), serde_json::json!({"quantization_config": qc}).to_string()).unwrap();
    };
    let dir = std::env::temp_dir().join(format!("fullm_gptq_{}", std::process::id()));
    write(&dir, &tensors);
    let c = &cases()[1];
    let run = |gptq: Option<PathBuf>| -> anyhow::Result<Vec<f32>> {
        let opts = LoadOptions {
            trunk: Precision::Bf16,
            max_seqs: 4,
            cache_tokens: 1024,
            page: 4,
            backend: BackendKind::CPU,
            kv_dtype: CacheDtype::F32,
            step_tokens: 64,
            max_rows: 64,
            max_context: 256,
            threads: 4,
            gpu_expert_format: Precision::Int4Group,
            int4_checkpoint: gptq,
            ..Default::default()
        };
        let mut m = models::load(&fixture(), &opts)?;
        let s = m.alloc_seq().unwrap();
        Ok(prefill(m.as_mut(), s, &c.prompt, 16))
    };
    let (own, gptq) = (run(None).unwrap(), run(Some(dir.clone())).unwrap());
    assert!(
        own == gptq,
        "GPTQ experts holding the int4 rounding's values: logits differ (rel-L2 {:.2e})",
        rel_l2(&gptq, &own)
    );
    // the checkpoint's scales are the ones used: layer 0's down projections 1.5x, other logits
    let mut alt = tensors.clone();
    for t in alt.iter_mut().filter(|t| t.0.starts_with("model.layers.0.") && t.0.ends_with("down_proj.scales")) {
        let s: Vec<u8> =
            t.3.chunks_exact(4).flat_map(|c| (f32::from_le_bytes(c.try_into().unwrap()) * 1.5).to_le_bytes()).collect();
        t.3 = s;
    }
    let alt_dir = dir.join("alt");
    write(&alt_dir, &alt);
    let d = rel_l2(&run(Some(alt_dir)).unwrap(), &own);
    assert!(d > 1e-3, "scaled GPTQ scales left the logits as they were ({d:.2e}): the checkpoint was not read");
    // a zero point of 8 stored (v1: 9) somewhere: asymmetric, refused
    let mut bad = tensors.clone();
    let z = bad.iter_mut().find(|t| t.0.ends_with("experts.3.up_proj.qzeros")).unwrap();
    z.3[0] = 0x78;
    let bad_dir = dir.join("asym");
    write(&bad_dir, &bad);
    let e = run(Some(bad_dir)).err().expect("an asymmetric GPTQ checkpoint loaded");
    assert!(format!("{e:#}").contains("zero points"), "{e:#}");
    std::fs::remove_dir_all(&dir).ok();
}

/// bf16 activations on CUDA (the serving default): steps of 12 rows (cuBLAS for the dense GEMMs
/// from 8 rows, the fused router, the experts) give the logits of the same prompt fed a token at a
/// time (the GEMV) up to bf16 rounding. Steps of 8-16 rows once left the add + norm's rows
/// bf16-only, and the router and experts read stale fp32 rows (rel-L2 0.39, 21 of 39 argmax).
/// bf16 experts: the CUDA int4 tiles want widths of whole 256 columns, more than this model's.
#[test]
fn qwen3_moe_bf16_activations_mid_size_steps() {
    if !fullm_kernels::exec::cuda_available() {
        return;
    }
    let c = &cases()[1];
    for fmt in [Precision::Bf16] {
        let run = |chunk: usize| {
            let opts = LoadOptions {
                trunk: Precision::Bf16,
                max_seqs: 4,
                cache_tokens: 1024,
                page: 4,
                backend: BackendKind::CUDA,
                kv_dtype: CacheDtype::Bf16,
                step_tokens: 64,
                max_rows: 64,
                max_context: 256,
                threads: 4,
                gpu_expert_format: fmt,
                bf16_activations: true,
                ..Default::default()
            };
            let mut m = models::load(&fixture(), &opts).expect("load tiny qwen3_moe");
            let s = m.alloc_seq().unwrap();
            prefill(m.as_mut(), s, &c.prompt, chunk)
        };
        let (one, mid) = (run(1), run(12));
        let err = rel_l2(&mid, &one);
        let v = one.len() / c.prompt.len();
        let same = (0..c.prompt.len())
            .filter(|&r| argmax(&one[r * v..(r + 1) * v]) == argmax(&mid[r * v..(r + 1) * v]))
            .count();
        eprintln!(
            "{fmt:?} experts, bf16 activations: 12-row steps vs 1-row, rel-L2 {err:.2e}, argmax {same} of {}",
            c.prompt.len()
        );
        assert!(
            err < 5e-2 && same * 10 >= c.prompt.len() * 9,
            "{fmt:?}: 12-row steps diverge (rel-L2 {err}, {same} argmax agree)"
        );
    }
}

#[test]
fn qwen3_moe_prefill_chunking_is_bit_identical() {
    for be in backends() {
        let mut m = load(be.clone(), Precision::Bf16);
        let c = &cases()[1];
        let s = m.alloc_seq().unwrap();
        let a = prefill(m.as_mut(), s, &c.prompt, 16);
        m.free_seq(s);
        let s = m.alloc_seq().unwrap();
        let b = prefill(m.as_mut(), s, &c.prompt, 5);
        eprintln!("{be:?}: prefill in chunks of 16 vs 5: rel-L2 {:.2e}", rel_l2(&a, &b));
        assert!(a == b, "{be:?}: chunking changed the logits");
    }
}

/// A prefix-cache hit: a second sequence with the same prompt starts from the first one's full
/// pages and computes only the rest, and its logits are the first one's, bit for bit (the exact
/// path: chunks of at most 16 tokens). A prompt that diverges inside a page reuses only the
/// pages before it.
#[test]
fn qwen3_moe_prefix_hit_is_bit_identical() {
    for be in backends() {
        let mut m = load(be.clone(), Precision::Bf16);
        let (c, v) = (&cases()[1], m.vocab());
        let (s, cached) = m.alloc_seq_cached(&c.prompt).unwrap();
        assert_eq!(cached, 0, "{be:?}: nothing is cached yet");
        let cold = prefill(m.as_mut(), s, &c.prompt, 16);
        m.free_seq(s);
        let free = m.capacity().free_cache_tokens;

        let (s, cached) = m.alloc_seq_cached(&c.prompt).unwrap();
        assert_eq!(cached, (c.prompt.len() - 1) / 4 * 4, "{be:?}: every full page but the last token's");
        let mut warm = Vec::new();
        for (i, ch) in c.prompt[cached..].chunks(16).enumerate() {
            let pos = cached + i * 16;
            warm.extend(
                m.forward(&[SeqChunk { seq: s, pos, tokens: ch, all_logits: true, draft: 0 }]).unwrap().remove(0),
            );
        }
        assert!(warm == cold[cached * v..], "{be:?}: a prefix hit changed the logits");
        m.free_seq(s);
        assert_eq!(m.capacity().free_cache_tokens, free, "{be:?}: pages leaked");

        let mut other = c.prompt.clone();
        other[10] += 1;
        let (s, cached) = m.alloc_seq_cached(&other).unwrap();
        assert_eq!(cached, 8, "{be:?}: the pages before the first difference");
        m.free_seq(s);
        eprintln!("{be:?}: prefix hit bit-identical to the cold prefill");
    }
}

/// The engine serves a repeated prompt and a follow-up turn from the prefix cache with the same
/// tokens as a cold engine, and reports the cached prompt tokens.
#[test]
fn qwen3_moe_engine_prefix_cache() {
    use fullm::runtime::{Engine, EngineConfig, Event, Request, SamplingParams};
    let c = &cases()[1];
    let run = |h: &fullm::runtime::EngineHandle, prompt: &[u32]| {
        let params = SamplingParams { max_tokens: 8, ..Default::default() };
        let mut rx = h.submit(Request { prompt: prompt.to_vec(), params }).unwrap();
        let mut toks = Vec::new();
        loop {
            match rx.blocking_recv().unwrap() {
                Event::Logprobs(_) => {}
                Event::Token(t) => toks.push(t),
                Event::Finished { usage, .. } => return (toks, usage.cached_tokens),
            }
        }
    };
    for be in backends() {
        let (cold_h, warm_h) = (
            Engine::new(load(be.clone(), Precision::Bf16), EngineConfig { step_tokens: 16, ..Default::default() })
                .spawn()
                .unwrap(),
            Engine::new(load(be.clone(), Precision::Bf16), EngineConfig { step_tokens: 16, ..Default::default() })
                .spawn()
                .unwrap(),
        );
        let (first, cached) = run(&warm_h, &c.prompt);
        assert_eq!(cached, 0);
        let (again, cached) = run(&warm_h, &c.prompt);
        assert_eq!((again.as_slice(), cached), (first.as_slice(), 36), "{be:?}: repeated prompt");
        // the next turn: the prompt, the reply, more user tokens
        let turn: Vec<u32> = c.prompt.iter().chain(&first).chain(&[5, 6, 7]).copied().collect();
        let (warm, cached) = run(&warm_h, &turn);
        let (cold, _) = run(&cold_h, &turn);
        assert_eq!(warm, cold, "{be:?}: follow-up turn");
        // the first request fed the prompt and 7 of its 8 tokens: 46 positions, 11 full pages
        assert_eq!(cached, 44, "{be:?}: follow-up turn's cached tokens");
        let stats = warm_h.stats();
        assert_eq!((stats.prompt_tokens, stats.cached_tokens), (39 + 39 + 50, 80));
        eprintln!("{be:?}: engine prefix cache: repeated prompt and follow-up turn served from cache");
    }
}

/// Two sequences that fill the same pages in one step share one copy afterwards (the second
/// switches to the first's), keep their logits, and the second one's own continuation still
/// enters the cache under the shared pages.
#[test]
fn qwen3_moe_prefix_pages_filled_twice_are_shared() {
    let (c, x) = (&cases()[1], [5u32, 6, 7, 8, 9, 10, 11, 12]);
    let px: Vec<u32> = c.prompt.iter().chain(&x).copied().collect();
    let mut cold = load(BackendKind::CPU, Precision::Bf16);
    let s = cold.alloc_seq().unwrap();
    let want = prefill(cold.as_mut(), s, &px, 16);

    let mut m = load(BackendKind::CPU, Precision::Bf16);
    let v = m.vocab();
    let (sa, sb) = (m.alloc_seq_cached(&c.prompt).unwrap().0, m.alloc_seq_cached(&c.prompt).unwrap().0);
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for (i, ch) in c.prompt.chunks(16).enumerate() {
        let pos = i * 16;
        let out = m
            .forward(&[
                SeqChunk { seq: sa, pos, tokens: ch, all_logits: true, draft: 0 },
                SeqChunk { seq: sb, pos, tokens: ch, all_logits: true, draft: 0 },
            ])
            .unwrap();
        a.extend_from_slice(&out[0]);
        b.extend_from_slice(&out[1]);
    }
    assert_eq!(m.capacity().prefix_cached_tokens, 36, "one copy of the 9 full pages");
    b.extend(
        m.forward(&[SeqChunk { seq: sb, pos: c.prompt.len(), tokens: &x, all_logits: true, draft: 0 }])
            .unwrap()
            .remove(0),
    );
    assert!(a == want[..c.prompt.len() * v] && b == want, "sharing a page changed the logits");
    m.free_seq(sa);
    m.free_seq(sb);
    let (s, cached) = m.alloc_seq_cached(&[px.as_slice(), &[1]].concat()).unwrap();
    assert_eq!(cached, px.len() / 4 * 4, "the continuation of the sequence that switched pages is cached");
    m.free_seq(s);
}

fn engine_opts(prefix: bool, cache_tokens: usize) -> LoadOptions {
    LoadOptions {
        trunk: Precision::Bf16,
        max_seqs: 4,
        cache_tokens,
        page: 4,
        backend: BackendKind::CPU,
        kv_dtype: CacheDtype::F32,
        step_tokens: 64,
        max_rows: 64,
        max_context: 256,
        threads: 4,
        no_prefix_cache: !prefix,
        ..Default::default()
    }
}

/// Submit every prompt at once (greedy, `n` tokens); each request's tokens, cached prompt tokens
/// and finish reason.
fn serve_all(
    opts: LoadOptions,
    step_tokens: usize,
    reqs: &[(Vec<u32>, usize)],
) -> (Vec<(Vec<u32>, usize, fullm::runtime::FinishReason)>, fullm::runtime::Stats) {
    use fullm::runtime::{EngineConfig, SamplingParams};
    let reqs: Vec<_> =
        reqs.iter().map(|(p, n)| (p.clone(), SamplingParams { max_tokens: *n, ..Default::default() })).collect();
    serve_with(opts, EngineConfig { step_tokens, ..Default::default() }, &reqs)
}

fn serve_with(
    opts: LoadOptions,
    cfg: fullm::runtime::EngineConfig,
    reqs: &[(Vec<u32>, fullm::runtime::SamplingParams)],
) -> (Vec<(Vec<u32>, usize, fullm::runtime::FinishReason)>, fullm::runtime::Stats) {
    use fullm::runtime::{Engine, Event, Request};
    let h = Engine::new(models::load(&fixture(), &opts).unwrap(), cfg).spawn().unwrap();
    let rxs: Vec<_> = reqs
        .iter()
        .map(|(p, params)| h.submit(Request { prompt: p.clone(), params: params.clone() }).unwrap())
        .collect();
    let out = rxs
        .into_iter()
        .map(|mut rx| {
            let mut toks = Vec::new();
            loop {
                match rx.blocking_recv().unwrap() {
                    Event::Logprobs(_) => {}
                    Event::Token(t) => toks.push(t),
                    Event::Finished { reason, usage } => return (toks, usage.cached_tokens, reason),
                }
            }
        })
        .collect();
    (out, h.stats())
}

/// Identical prompts submitted together: the first computes the shared pages, the others wait
/// for them instead of computing them again, and every request gets the cold engine's tokens.
#[test]
fn qwen3_moe_engine_waits_for_a_shared_prefill() {
    let c = &cases()[1];
    let reqs = vec![(c.prompt.clone(), 6); 3];
    let (warm, stats) = serve_all(engine_opts(true, 1024), 16, &reqs);
    let (cold, _) = serve_all(engine_opts(false, 1024), 16, &reqs);
    // they wait while at least two shared pages are still to come: after the first's second
    // 16-token step, 32 of the 36 shareable positions are cached and they start
    let cached: Vec<usize> = warm.iter().map(|r| r.1).collect();
    assert_eq!(cached, [0, 32, 32]);
    assert_eq!(stats.cached_tokens, 64);
    for (w, c) in warm.iter().zip(&cold) {
        assert_eq!(w.0, c.0, "waiting for a shared prefill changed the tokens");
    }
}

/// Many requests with shared prefixes of every length in a cache far too small for all of them:
/// pages are shared, switched and evicted, no step runs out of cache, and every request gets the
/// tokens an engine without the prefix cache gives it.
#[test]
fn qwen3_moe_engine_prefix_cache_under_pressure() {
    let base = &cases()[1].prompt;
    let cut = [39, 30, 20, 39, 12, 39, 25, 39, 8, 33, 39, 17, 39, 2];
    let reqs: Vec<(Vec<u32>, usize)> = cut
        .iter()
        .enumerate()
        .map(|(k, &n)| {
            let tail = [k as u32 + 1, 7, 9];
            (base[..n].iter().chain(&tail[..k % 3 + 1]).copied().collect(), 4 + k % 5)
        })
        .collect();
    // 24 pages of 4 positions; the longest request may need 13
    let (warm, stats) = serve_all(engine_opts(true, 96), 8, &reqs);
    let (cold, _) = serve_all(engine_opts(false, 96), 8, &reqs);
    for (i, (w, c)) in warm.iter().zip(&cold).enumerate() {
        assert_eq!(w.2, fullm::runtime::FinishReason::Length, "request {i}: {:?}", w.2);
        assert_eq!(w.0, c.0, "request {i}: the prefix cache changed the tokens");
    }
    assert!(stats.cached_tokens > 0, "nothing was served from the cache");
    eprintln!("under pressure: {} of {} prompt tokens from the cache", stats.cached_tokens, stats.prompt_tokens);
}

/// Optimistic admission in a cache too small for what the requests generate: steps run short,
/// the newest request is preempted and resumes from its cached pages, and every request (one of
/// them sampled with a seed) gets exactly the tokens of an unpressured, never-preempting engine.
#[test]
fn qwen3_moe_engine_preempts_and_resumes() {
    use fullm::runtime::{Admission, EngineConfig, FinishReason, SamplingParams};
    let base = &cases()[1].prompt;
    let stop = vec![u32::MAX - 1]; // never sampled: the optimistic estimate applies, all run to max_tokens
    let reqs: Vec<(Vec<u32>, SamplingParams)> = (0..8)
        .map(|k| {
            let prompt: Vec<u32> = base[..12 + 2 * k].iter().chain(&[k as u32 + 3]).copied().collect();
            let mut p = SamplingParams { max_tokens: 40, stop_token_ids: stop.clone(), ..Default::default() };
            if k % 3 == 1 {
                (p.temperature, p.top_p, p.seed) = (0.8, 0.95, k as u64);
            }
            (prompt, p)
        })
        .collect();
    let cfg = |admission| EngineConfig { step_tokens: 8, admission, ..Default::default() };
    let (want, calm) = serve_with(engine_opts(true, 1024), cfg(Admission::WorstCase), &reqs);
    assert_eq!(calm.preemptions, 0);
    // pages of 4 positions; each request ends at 52-66 positions (13-17 pages)
    let mut total = 0;
    for cache in [72, 96, 128] {
        let (got, stats) = serve_with(engine_opts(true, cache), cfg(Admission::Optimistic), &reqs);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.2, FinishReason::Length, "cache {cache}, request {i}: {:?}", g.2);
            assert_eq!(g.0, w.0, "cache {cache}, request {i}: preemption changed the tokens");
        }
        eprintln!(
            "cache {cache}: preempted {} times, {} positions computed again; prefix cache served {} of {} prompt tokens",
            stats.preemptions, stats.recomputed_tokens, stats.cached_tokens, stats.prompt_tokens
        );
        total += stats.preemptions;
    }
    assert!(total >= 3, "the cache rarely ran short: {total} preemptions");
}

/// The host tier on every backend (on CUDA: the copy stream and its kernel): a prompt's pages
/// written through, dropped from the device by a second prompt that needs every page, loaded
/// back, and the rest of the prompt computed on them gives the cold prefill's logits bit for bit.
#[test]
fn qwen3_moe_host_tier_round_trip_is_bit_identical() {
    for be in backends() {
        let opts =
            LoadOptions { backend: be.clone(), cache_tokens: 64, host_cache_tokens: 1024, ..engine_opts(true, 64) };
        let mut m = models::load(&fixture(), &opts).expect("load tiny qwen3_moe");
        let (c, v) = (&cases()[1], m.vocab());
        let (s, _) = m.alloc_seq_cached(&c.prompt).unwrap();
        let cold = prefill(m.as_mut(), s, &c.prompt, 16);
        m.free_seq(s);
        // 64 positions: all 16 pages, so every page of the first prompt leaves the device
        let other: Vec<u32> = (0..64).map(|i| (i * 7 + 3) % 200).collect();
        let (s, _) = m.alloc_seq_cached(&other).unwrap();
        prefill(m.as_mut(), s, &other, 16);
        m.free_seq(s);
        assert_eq!(m.capacity().host_cached_tokens, 36 + 64, "{be:?}: both prompts' full pages on the host");

        let (s, cached) = m.alloc_seq_cached(&c.prompt).unwrap();
        assert_eq!(cached, 0, "{be:?}: nothing of the first prompt is resident");
        assert_eq!(m.host_cached(s, &c.prompt), 36, "{be:?}");
        assert_eq!(m.load_cached(s, &c.prompt, 36).unwrap(), 36, "{be:?}");
        if !m.seq_ready(s) {
            m.wait_copies().unwrap();
        }
        assert!(m.seq_ready(s), "{be:?}: loads done after waiting");
        let mut warm = Vec::new();
        for (i, ch) in c.prompt[36..].chunks(16).enumerate() {
            let pos = 36 + i * 16;
            warm.extend(
                m.forward(&[SeqChunk { seq: s, pos, tokens: ch, all_logits: true, draft: 0 }]).unwrap().remove(0),
            );
        }
        assert!(warm == cold[36 * v..], "{be:?}: a hit loaded from the host changed the logits");
        m.free_seq(s);
        assert_eq!(m.flush_prefix_cache(), 36 + 64, "{be:?}: the flush drops both tiers");
        assert_eq!((m.capacity().free_cache_tokens, m.capacity().host_cached_tokens), (64, 0), "{be:?}");
        eprintln!("{be:?}: host tier round trip bit-identical");
    }
}

/// The engine over a host tier on every backend: requests whose pages other requests pushed off
/// the device load them back (asynchronously on CUDA) and get the cold engine's tokens.
#[test]
fn qwen3_moe_engine_host_tier() {
    let base = &cases()[1].prompt;
    let reqs: Vec<(Vec<u32>, usize)> = (0..6)
        .map(|k| {
            let mut p: Vec<u32> = base.iter().map(|&t| (t + 17 * (k as u32 % 3)) % 200).collect();
            p.extend([k as u32 + 1, 5]);
            (p, 6)
        })
        .collect();
    for be in backends() {
        let opts = |host| LoadOptions { backend: be.clone(), host_cache_tokens: host, ..engine_opts(true, 96) };
        let (cold, _) = serve_all(LoadOptions { no_prefix_cache: true, ..opts(0) }, 16, &reqs);
        // twice through: the second round's prompts (three distinct prefixes) were evicted
        let twice: Vec<_> = reqs.iter().chain(&reqs).cloned().collect();
        let (warm, stats) = serve_all(opts(4096), 16, &twice);
        for (i, w) in warm.iter().enumerate() {
            assert_eq!(w.0, cold[i % reqs.len()].0, "{be:?}: request {i}");
        }
        assert!(stats.host_loaded_tokens > 0, "{be:?}: nothing loaded from the host");
        eprintln!(
            "{be:?}: engine host tier: {} of {} prompt tokens cached, {} loaded",
            stats.cached_tokens, stats.prompt_tokens, stats.host_loaded_tokens
        );
    }
}

/// "cpu-core" (the CPU backend with only the runtime, the core ops and the GQA family: what a new
/// device implements first) runs the model through the executor's compositions
/// and Backend's base forms (moe_experts gathers each expert's rows and calls gemm). Its logits
/// match the full CPU backend's, bit for bit with bf16 and int4 experts; its greedy tokens with
/// bf16 experts are transformers'. fp8b experts are quantised differently: the full backend
/// stores gate and up interleaved (moe_gate_up_interleaved), so its 128 x 128 scale blocks mix
/// gate and up rows, while cpu-core keeps them apart; both stay close to transformers.
#[test]
fn core_backend_matches_cpu() {
    let core = BackendKind::Executor("cpu-core".into());
    let all = cases();
    for fmt in [Precision::Bf16, Precision::Fp8Block, Precision::Int4Group] {
        let mut full = load(BackendKind::CPU, fmt);
        let mut m = load(core.clone(), fmt);
        let mut worst = 0.0f32;
        for c in &all {
            let (s, sf) = (m.alloc_seq().unwrap(), full.alloc_seq().unwrap());
            let (a, b) = (prefill(m.as_mut(), s, &c.prompt, 16), prefill(full.as_mut(), sf, &c.prompt, 16));
            if fmt == Precision::Fp8Block {
                let want: Vec<f32> = c.logits.iter().flatten().copied().collect();
                assert!(rel_l2(&a, &want) < 5e-2, "cpu-core fp8b: too far from transformers");
            } else {
                worst = worst.max(rel_l2(&a, &b));
            }
            let v = m.vocab();
            let (mut pos, mut next) = (c.prompt.len(), argmax(&a[(c.prompt.len() - 1) * v..]));
            let mut got = Vec::new();
            for _ in 0..c.greedy.len() {
                got.push(next);
                let l = m
                    .forward(&[SeqChunk { seq: s, pos, tokens: &[next], all_logits: false, draft: 0 }])
                    .unwrap()
                    .remove(0);
                pos += 1;
                next = argmax(&l);
            }
            if fmt == Precision::Bf16 {
                assert_eq!(got, c.greedy, "cpu-core: greedy tokens differ from transformers");
            }
            m.free_seq(s);
            full.free_seq(sf);
        }
        if fmt != Precision::Fp8Block {
            eprintln!("cpu-core {fmt:?}: worst rel-L2 vs the CPU backend {worst:.2e}");
        }
        assert!(worst == 0.0, "cpu-core {fmt:?}: logits differ from the CPU backend: {worst}");
    }
}
