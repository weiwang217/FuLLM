// SPDX-License-Identifier: Apache-2.0
//! The C++ executor against the torch oracle and against the Rust reference.
//!
//! Every check runs on the CPU backend here and again on the CUDA backend when this build
//! has one and a GPU is present (`cargo test --features cuda`). With f32 state and cache the
//! executor must reproduce the oracle like the reference does; with bf16 (the vLLM-style
//! default) the drift is reported and loosely bounded.
//!
//! On CUDA, steps of up to 16 tokens take the exact-fp32 GEMV path; longer prefills use
//! cuBLAS with bf16 inputs, so the teacher-forcing test there uses a looser tolerance.

use std::path::PathBuf;

use fullm::models::{self, BackendKind, CacheDtype, LoadOptions, Model, SeqChunk};
use fullm::tensor::Precision;

struct Case {
    prompt: Vec<u32>,
    tf_logits: Vec<Vec<f32>>,
    greedy: Vec<u32>,
}

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny_k3")
}

fn cases() -> Vec<Case> {
    let raw = std::fs::read_to_string(fixture().join("oracle.json")).expect("oracle.json");
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let ids =
        |x: &serde_json::Value| x.as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect::<Vec<_>>();
    v["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| Case {
            prompt: ids(&c["prompt"]),
            tf_logits: c["tf_logits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect())
                .collect(),
            greedy: ids(&c["greedy"]),
        })
        .collect()
}

fn backends() -> Vec<BackendKind> {
    let mut b = vec![BackendKind::CPU];
    if fullm_kernels::exec::cuda_available() {
        b.push(BackendKind::CUDA);
    } else {
        eprintln!("CUDA not available: executor tests run on the CPU backend only");
    }
    b
}

fn load(backend: BackendKind, dt: CacheDtype, step_tokens: usize) -> Box<dyn Model> {
    let opts = LoadOptions {
        trunk: Precision::Bf16,
        max_seqs: 4,
        cache_tokens: 512,
        page: 4,
        backend,
        state_dtype: dt,
        kv_dtype: dt,
        step_tokens,
        max_rows: 64,
        max_context: 256,
        threads: 4,
        ..Default::default()
    };
    models::load(&fixture(), &opts).expect("load tiny checkpoint on the executor")
}

fn argmax(v: &[f32]) -> u32 {
    let mut b = 0;
    for i in 1..v.len() {
        if v[i] > v[b] {
            b = i;
        }
    }
    b as u32
}

fn run_greedy(m: &mut dyn Model, prompt: &[u32], chunk: usize, steps: usize) -> (Vec<u32>, Vec<Vec<f32>>) {
    let s = m.alloc_seq().unwrap();
    let mut pos = 0;
    let mut last = Vec::new();
    for c in prompt.chunks(chunk) {
        last = m.forward(&[SeqChunk { seq: s, pos, tokens: c, all_logits: false, draft: 0 }]).unwrap().remove(0);
        pos += c.len();
    }
    let mut toks = Vec::new();
    let mut rows = vec![last.clone()];
    let mut next = argmax(&last);
    for _ in 0..steps {
        toks.push(next);
        let l = m.forward(&[SeqChunk { seq: s, pos, tokens: &[next], all_logits: false, draft: 0 }]).unwrap().remove(0);
        pos += 1;
        next = argmax(&l);
        rows.push(l);
    }
    m.free_seq(s);
    (toks, rows)
}

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
    num / b.iter().map(|y| y * y).sum::<f32>().sqrt().max(1e-30)
}

#[test]
fn executor_teacher_forcing_matches_torch() {
    for be in backends() {
        let mut m = load(be.clone(), CacheDtype::F32, 64);
        let tol = if be == BackendKind::CUDA { 5e-2 } else { 1e-4 };
        for case in cases() {
            let s = m.alloc_seq().unwrap();
            let out = m
                .forward(&[SeqChunk { seq: s, pos: 0, tokens: &case.prompt, all_logits: true, draft: 0 }])
                .unwrap()
                .remove(0);
            m.free_seq(s);
            let v = m.vocab();
            let mut worst = 0.0f32;
            for (t, want) in case.tf_logits.iter().enumerate() {
                worst = worst.max(rel_l2(&out[t * v..(t + 1) * v], want));
            }
            eprintln!("{be:?}: prompt len {} worst rel-L2 logit error {worst:.2e}", case.prompt.len());
            assert!(worst < tol, "{be:?}: logits drift {worst:e} from torch");
        }
    }
}

#[test]
fn executor_greedy_matches_torch() {
    for be in backends() {
        // prompts are <= 12 tokens, so every step stays on the exact GEMV path
        let mut m = load(be.clone(), CacheDtype::F32, 16);
        for case in cases() {
            let (toks, _) = run_greedy(m.as_mut(), &case.prompt, 16, case.greedy.len());
            assert_eq!(toks, case.greedy, "{be:?}: greedy continuation differs from torch");
        }
    }
}

#[test]
fn executor_batching_and_chunking_are_bit_identical() {
    for be in backends() {
        let cs = cases();
        let steps = 6;
        let mut m = load(be.clone(), CacheDtype::F32, 16);
        let alone: Vec<_> = cs.iter().map(|c| run_greedy(m.as_mut(), &c.prompt, 16, steps)).collect();
        for (c, (toks, rows)) in cs.iter().zip(&alone) {
            let (t2, r2) = run_greedy(m.as_mut(), &c.prompt, 3, steps);
            assert_eq!(&t2, toks, "{be:?}: chunked prefill changed tokens");
            assert!(rows.iter().zip(&r2).all(|(a, b)| a == b), "{be:?}: chunked prefill changed logits bits");
        }
        // A prefills alone; then A's first decode shares a step with B's prefill (a mixed
        // prefill + decode batch); then both decode together. Every logits row must equal the
        // corresponding row from the runs above, bit for bit.
        let (a, b) = (&cs[0], &cs[1]);
        let (sa, sb) = (m.alloc_seq().unwrap(), m.alloc_seq().unwrap());
        let la = m
            .forward(&[SeqChunk { seq: sa, pos: 0, tokens: &a.prompt, all_logits: false, draft: 0 }])
            .unwrap()
            .remove(0);
        assert!(la == alone[0].1[0], "{be:?}: prefill logits differ");
        let mut next = [[argmax(&la)], [0u32]];
        let out = m
            .forward(&[
                SeqChunk { seq: sa, pos: a.prompt.len(), tokens: &next[0], all_logits: false, draft: 0 },
                SeqChunk { seq: sb, pos: 0, tokens: &b.prompt, all_logits: false, draft: 0 },
            ])
            .unwrap();
        assert!(out[0] == alone[0].1[1], "{be:?}: decode sharing a step with a prefill differs");
        assert!(out[1] == alone[1].1[0], "{be:?}: prefill sharing a step with a decode differs");
        let mut pos = [a.prompt.len() + 1, b.prompt.len()];
        next = [[argmax(&out[0])], [argmax(&out[1])]];
        for step in 1..steps {
            let out = m
                .forward(&[
                    SeqChunk { seq: sa, pos: pos[0], tokens: &next[0], all_logits: false, draft: 0 },
                    SeqChunk { seq: sb, pos: pos[1], tokens: &next[1], all_logits: false, draft: 0 },
                ])
                .unwrap();
            assert!(out[0] == alone[0].1[step + 1], "{be:?}: batched decode differs (seq A, step {step})");
            assert!(out[1] == alone[1].1[step], "{be:?}: batched decode differs (seq B, step {step})");
            next = [[argmax(&out[0])], [argmax(&out[1])]];
            pos = [pos[0] + 1, pos[1] + 1];
        }
        m.free_seq(sa);
        m.free_seq(sb);
    }
}

#[test]
fn executor_bf16_state_and_cache_drift() {
    for be in backends() {
        let mut exact = load(be.clone(), CacheDtype::F32, 16);
        let mut half = load(be.clone(), CacheDtype::Bf16, 16);
        let mut worst = 0.0f32;
        let (mut agree, mut total) = (0, 0);
        for case in cases() {
            let (t1, r1) = run_greedy(exact.as_mut(), &case.prompt, 16, 8);
            let (t2, r2) = run_greedy(half.as_mut(), &case.prompt, 16, 8);
            // Compare logits only while both runs have seen the same tokens: row i is the
            // prediction after i generated tokens, so rows 0..=prefix are like for like.
            let prefix = t1.iter().zip(&t2).take_while(|(a, b)| a == b).count();
            for (a, b) in r2.iter().zip(&r1).take(prefix + 1) {
                worst = worst.max(rel_l2(a, b));
            }
            agree += prefix;
            total += t1.len();
        }
        eprintln!("{be:?}: bf16 state+cache vs f32: worst rel-L2 logit error {worst:.2e}, greedy prefix agreement {agree}/{total}");
        assert!(worst < 0.5, "{be:?}: bf16 state drifts {worst}");
    }
}

#[test]
fn cuda_selftest() {
    if !fullm_kernels::exec::cuda_available() {
        eprintln!("CUDA not available: skipping the op self test");
        return;
    }
    let (fails, report) = fullm_kernels::exec::selftest(0);
    eprintln!("{report}");
    assert_eq!(fails, 0, "{fails} CUDA op(s) disagree with the CPU backend");
}

/// A backend without the families this model needs (cpu-core has the GQA family only) is refused
/// when the executor is built, with the missing families named.
#[test]
fn k3_refused_without_its_families() {
    let opts = LoadOptions {
        trunk: Precision::Bf16,
        max_seqs: 2,
        cache_tokens: 256,
        page: 4,
        backend: BackendKind::Executor("cpu-core".into()),
        kv_dtype: CacheDtype::F32,
        step_tokens: 16,
        max_rows: 16,
        max_context: 128,
        threads: 2,
        ..Default::default()
    };
    let err = match models::load(&fixture(), &opts) {
        Ok(_) => panic!("cpu-core must refuse Kimi K3"),
        Err(e) => format!("{e:#}"),
    };
    assert!(err.contains("cpu-core backend lacks this model's ops") && err.contains("MLA"), "{err}");
}
