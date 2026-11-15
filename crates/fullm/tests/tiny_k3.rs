// SPDX-License-Identifier: Apache-2.0
//! End-to-end gate: the CPU backend against the torch oracle on the tiny Kimi K3.
//!
//! The tiny checkpoint has the released tensor graph and on-disk formats (BF16 trunk,
//! MXFP4 experts, f32 small vectors), 13 layers so AttnRes completes several blocks, and
//! 4 MLA + 9 KDA layers. tools/make_tiny_oracle.py regenerates it.
//!
//! Checked, in order of what they exercise:
//!   1. teacher forcing: logits at every prompt position match torch (one prefill)
//!   2. greedy decode: 16 tokens match torch's full-recompute greedy exactly
//!   3. two sequences decoded in the same steps: same tokens as alone, bit-identical logits
//!   4. chunked prefill (3 tokens at a time): bit-identical to one-shot prefill
//!   5. FP8 trunk: agreement with the exact greedy continuation (reported, loosely bounded)

use std::path::PathBuf;

use fullm::models::{self, LoadOptions, Model, SeqChunk};
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
    let raw =
        std::fs::read_to_string(fixture().join("oracle.json")).expect("oracle.json; run tools/make_tiny_oracle.py");
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    v["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| Case {
            prompt: c["prompt"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect(),
            tf_logits: c["tf_logits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect())
                .collect(),
            greedy: c["greedy"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect(),
        })
        .collect()
}

fn load(trunk: Precision) -> Box<dyn Model> {
    let opts = LoadOptions { trunk, max_seqs: 4, cache_tokens: 512, page: 4, ..Default::default() };
    models::load(&fixture(), &opts).expect("load tiny checkpoint")
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

/// Prefill `prompt` in chunks of `chunk`, then greedy-decode `steps`. Returns the tokens
/// and every logits row produced for the last prompt position and each decode step.
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

#[test]
fn teacher_forced_logits_match_torch() {
    let mut m = load(Precision::Bf16);
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
            let got = &out[t * v..(t + 1) * v];
            let scale = want.iter().fold(0.0f32, |a, x| a.max(x.abs()));
            for (g, w) in got.iter().zip(want) {
                worst = worst.max((g - w).abs() / scale);
            }
            assert_eq!(argmax(got), argmax(want), "argmax differs at position {t}");
        }
        eprintln!("prompt len {}: worst relative logit error {worst:.2e}", case.prompt.len());
        assert!(worst < 1e-4, "logits drift {worst:e} from the torch reference");
    }
}

#[test]
fn greedy_decode_matches_torch() {
    let mut m = load(Precision::Bf16);
    for case in cases() {
        let (toks, _) = run_greedy(m.as_mut(), &case.prompt, usize::MAX, case.greedy.len());
        assert_eq!(toks, case.greedy, "greedy continuation differs from torch");
    }
}

#[test]
fn batching_and_chunking_are_bit_identical() {
    let cs = cases();
    let steps = 8;
    let mut m = load(Precision::Bf16);
    let alone: Vec<_> = cs.iter().map(|c| run_greedy(m.as_mut(), &c.prompt, usize::MAX, steps)).collect();

    // chunked prefill, 3 tokens per step
    for (c, (toks, rows)) in cs.iter().zip(&alone) {
        let (t2, r2) = run_greedy(m.as_mut(), &c.prompt, 3, steps);
        assert_eq!(&t2, toks);
        assert!(rows.iter().zip(&r2).all(|(a, b)| a == b), "chunked prefill changed logits bits");
    }

    // both sequences in every step: prefills together, then decodes together
    let ids: Vec<_> = cs.iter().map(|_| m.alloc_seq().unwrap()).collect();
    let mut pos: Vec<usize> = vec![0; cs.len()];
    let batch: Vec<SeqChunk> = cs
        .iter()
        .zip(&ids)
        .map(|(c, &s)| SeqChunk { seq: s, pos: 0, tokens: &c.prompt, all_logits: false, draft: 0 })
        .collect();
    let mut last = m.forward(&batch).unwrap();
    for (p, c) in pos.iter_mut().zip(&cs) {
        *p = c.prompt.len();
    }
    for step in 0..steps {
        for (i, (_, rows)) in alone.iter().enumerate() {
            assert!(last[i] == rows[step], "batched logits differ (seq {i}, step {step})");
        }
        let next: Vec<[u32; 1]> = last.iter().map(|l| [argmax(l)]).collect();
        let batch: Vec<SeqChunk> = ids
            .iter()
            .zip(&next)
            .zip(&pos)
            .map(|((&s, t), &p)| SeqChunk { seq: s, pos: p, tokens: t, all_logits: false, draft: 0 })
            .collect();
        last = m.forward(&batch).unwrap();
        pos.iter_mut().for_each(|p| *p += 1);
    }
    for s in ids {
        m.free_seq(s);
    }
}

/// 8-bit trunk formats against the exact model: relative logit error over every prompt
/// position, and teacher-forced agreement with the exact greedy continuation (so one early
/// miss does not cascade). The tiny model's random weights give flat logits, so agreement
/// is pessimistic here; the logit error is the comparable number.
fn quant_quality(p: Precision) -> (f32, usize, usize) {
    let mut m = load(p);
    let (mut worst, mut agree, mut total) = (0.0f32, 0, 0);
    for case in cases() {
        let mut ids = case.prompt.clone();
        ids.extend_from_slice(&case.greedy);
        let s = m.alloc_seq().unwrap();
        let out =
            m.forward(&[SeqChunk { seq: s, pos: 0, tokens: &ids, all_logits: true, draft: 0 }]).unwrap().remove(0);
        m.free_seq(s);
        let v = m.vocab();
        for (t, want) in case.tf_logits.iter().enumerate() {
            let scale = want.iter().fold(0.0f32, |a, x| a.max(x.abs()));
            let num: f32 = out[t * v..(t + 1) * v].iter().zip(want).map(|(g, w)| (g - w) * (g - w)).sum::<f32>().sqrt();
            let den: f32 = want.iter().map(|w| w * w).sum::<f32>().sqrt();
            worst = worst.max(num / den.max(scale * 1e-6));
        }
        for (i, &want) in case.greedy.iter().enumerate() {
            let t = case.prompt.len() - 1 + i;
            total += 1;
            agree += (argmax(&out[t * v..(t + 1) * v]) == want) as usize;
        }
    }
    (worst, agree, total)
}

/// Per-matrix output error of each 8-bit format on the checkpoint's real bf16 matrices.
/// This is the test that can fail on a broken quantiser; the end-to-end one below cannot
/// tell a bug from the fixture's chaos.
#[test]
fn eight_bit_formats_per_matrix() {
    use fullm::st::{Dtype, SafeTensors};
    use fullm::tensor::{linear_new, Mat};
    let st = SafeTensors::open_dir(&fixture()).unwrap();
    let (mut worst_i8, mut worst_f8, mut n) = (0.0f32, 0.0f32, 0);
    for name in st.names() {
        let v = st.get(name).unwrap();
        if v.dtype() != Dtype::BF16 || v.shape().len() != 2 || name.contains("embed") || name.contains("lm_head") {
            continue;
        }
        let (r, c) = (v.shape()[0], v.shape()[1]);
        let w = v.to_bf16().unwrap();
        let x: Vec<f32> = (0..4 * c).map(|i| (((i * 2654435761usize) % 1000) as f32 - 500.0) / 300.0).collect();
        let exact = linear_new(&x, 4, &Mat::bf16(r, c, w.clone()));
        let rel = |m: Mat| {
            let y = linear_new(&x, 4, &m);
            let num: f32 = y.iter().zip(&exact).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
            num / exact.iter().map(|b| b * b).sum::<f32>().sqrt()
        };
        worst_i8 = worst_i8.max(rel(Mat::bf16(r, c, w.clone()).to_int8()));
        worst_f8 = worst_f8.max(rel(Mat::bf16(r, c, w).to_fp8()));
        n += 1;
    }
    eprintln!("{n} matrices: worst output rel-L2 error int8 {worst_i8:.4}, fp8 {worst_f8:.4}");
    assert!(worst_i8 < 0.01, "int8 per-row error {worst_i8}");
    assert!(worst_f8 < 0.04, "fp8 per-row error {worst_f8}");
}

/// End to end, the tiny model amplifies any weight perturbation enormously (random
/// weights, SiTU caps saturated on purpose), so absolute numbers here say nothing about a
/// trained model. What must hold is the ranking: int8 strictly closer to exact than fp8.
#[test]
fn eight_bit_trunk_end_to_end_ranking() {
    let (e_i8, a_i8, n) = quant_quality(Precision::Int8);
    let (e_f8, a_f8, _) = quant_quality(Precision::Fp8);
    eprintln!("int8 trunk: worst rel-L2 logit error {e_i8:.3e}, teacher-forced agreement {a_i8}/{n}");
    eprintln!("fp8  trunk: worst rel-L2 logit error {e_f8:.3e}, teacher-forced agreement {a_f8}/{n}");
    assert!(e_i8 < e_f8, "int8 should track the exact model more closely than fp8");
}
