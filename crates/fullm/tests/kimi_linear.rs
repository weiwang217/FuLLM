// SPDX-License-Identifier: Apache-2.0
//! Kimi Linear (tiny, released layout) against the torch oracle from
//! tools/make_tiny_kimi_linear.py, on every backend: the Rust reference, the C++ executor
//! on the CPU, and on CUDA when this build has it and a GPU is present.
//!
//! The model differs from K3 in its decay gate (softplus), low-rank KDA output gate, direct
//! MLA query with a 1e-6 kv norm, plain residuals, full-width SiLU MoE and bf16 experts;
//! these tests are what check every one of those switches end to end.
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
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny_kimi_linear")
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
    let mut b = vec![BackendKind::Reference, BackendKind::CPU];
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
fn kimi_linear_teacher_forcing_matches_torch() {
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
fn kimi_linear_greedy_matches_torch() {
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
fn kimi_linear_batching_and_chunking_are_bit_identical() {
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
fn kimi_linear_bf16_state_and_cache_drift() {
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

fn load_with_gpu_experts(backend: BackendKind, gb: Option<f64>, fmt: Precision, no_graphs: bool) -> Box<dyn Model> {
    let opts = LoadOptions {
        trunk: Precision::Bf16,
        max_seqs: 4,
        cache_tokens: 512,
        page: 4,
        backend,
        state_dtype: CacheDtype::F32,
        kv_dtype: CacheDtype::F32,
        step_tokens: 16,
        max_rows: 64,
        max_context: 256,
        threads: 4,
        gpu_experts: gb.map_or(fullm::models::GpuExperts::Off, fullm::models::GpuExperts::Gb),
        gpu_expert_format: fmt,
        no_cuda_graphs: no_graphs,
        ..Default::default()
    };
    models::load(&fixture(), &opts).expect("load with resident experts")
}

/// The HBM expert tier. Pairs routed to resident experts are computed by the backend, the rest
/// by the CPU pool, and the rows are summed in selection order either way.
///   - CPU backend, bf16: both paths run the same kernels in the same order, so splitting must
///     not change one bit (half resident and all resident).
///   - CUDA backend, bf16: resident rows use the GPU's GEMV; close to the pool, same tokens.
///   - int8: quantisation moves the tiny model a lot (random weights amplify any change), so
///     the check is CUDA int8 against CPU int8 on the same quantised weights; the quality of
///     int8 experts is measured on the real model (`fullm ppl`).
#[test]
fn kimi_linear_resident_experts_match_cpu_pool() {
    let per = 3 * 128 * 48 * 2 + 6 * 256; // one tiny expert in the arena; 8 per layer, 7 MoE layers
    let run_all = |m: &mut dyn Model| cases().iter().map(|c| run_greedy(m, &c.prompt, 16, 6)).collect::<Vec<_>>();
    let cmp = |a: &[(Vec<u32>, Vec<Vec<f32>>)], b: &[(Vec<u32>, Vec<Vec<f32>>)]| {
        let (mut worst, mut identical, mut same_tokens) = (0.0f32, true, true);
        for ((ta, ra), (tb, rb)) in a.iter().zip(b) {
            identical &= ra == rb;
            same_tokens &= ta == tb;
            let prefix = ta.iter().zip(tb).take_while(|(x, y)| x == y).count();
            for (x, y) in ra.iter().zip(rb).take(prefix + 1) {
                worst = worst.max(rel_l2(x, y));
            }
        }
        (worst, identical, same_tokens)
    };

    let pool = run_all(load_with_gpu_experts(BackendKind::CPU, None, Precision::Bf16, false).as_mut());
    for (label, gb) in [("half resident", (per * 28) as f64 / 1e9), ("all resident", 1.0)] {
        let got = run_all(load_with_gpu_experts(BackendKind::CPU, Some(gb), Precision::Bf16, false).as_mut());
        let (worst, identical, _) = cmp(&got, &pool);
        eprintln!("Cpu {label} bf16: worst rel-L2 vs pool {worst:.2e}, bit-identical {identical}");
        assert!(identical, "{label}: splitting experts between backend and pool changed bits");
    }
    let cpu_int8 = run_all(load_with_gpu_experts(BackendKind::CPU, Some(1.0), Precision::Int8, false).as_mut());
    let (w8, _, _) = cmp(&cpu_int8, &pool);
    eprintln!("Cpu all resident int8: worst rel-L2 vs bf16 pool {w8:.2e} (tiny-model sensitivity, not a bound)");
    // fp8b and int4: given as bf16, interleaved and quantised by the executor (prefill on their
    // GEMMs on CUDA, whose 256-column tiles this model is too narrow for: checked on the real one)
    for fmt in [Precision::Fp8Block, Precision::Int4Group] {
        let got = run_all(load_with_gpu_experts(BackendKind::CPU, Some(1.0), fmt, false).as_mut());
        assert!(got.iter().all(|(_, r)| r.iter().flatten().all(|v| v.is_finite())), "{fmt:?}: non-finite logits");
        let (wq, _, _) = cmp(&got, &pool);
        eprintln!("Cpu all resident {fmt:?}: worst rel-L2 vs bf16 pool {wq:.2e}");
    }

    if fullm_kernels::exec::cuda_available() {
        for (label, gb) in [("half resident", (per * 28) as f64 / 1e9), ("all resident", 1.0)] {
            let got = run_all(load_with_gpu_experts(BackendKind::CUDA, Some(gb), Precision::Bf16, false).as_mut());
            let (worst, _, same) = cmp(&got, &pool);
            eprintln!("Cuda {label} bf16: worst rel-L2 vs CPU pool {worst:.2e}, same tokens {same}");
            assert!(worst < 1e-4 && same, "Cuda {label}: {worst}");
        }
        let got = run_all(load_with_gpu_experts(BackendKind::CUDA, Some(1.0), Precision::Int8, false).as_mut());
        let (worst, _, same) = cmp(&got, &cpu_int8);
        eprintln!("Cuda all resident int8 vs Cpu int8: worst rel-L2 {worst:.2e}, same tokens {same}");
        assert!(worst < 1e-4 && same, "Cuda int8: {worst}");
    }
}

/// CUDA graphs: small steps with every expert resident replay a recorded forward pass. The
/// replays must give the logits of running the same steps op by op, bit for bit: in chunked
/// prefill, in batched decode, and while positions move on (the context crosses attention
/// split boundaries inside one recorded graph).
#[test]
fn kimi_linear_cuda_graphs_are_bit_identical() {
    if !fullm_kernels::exec::cuda_available() {
        eprintln!("no CUDA device: skipped");
        return;
    }
    let run = |no_graphs: bool| {
        let mut m = load_with_gpu_experts(BackendKind::CUDA, Some(1.0), Precision::Bf16, no_graphs);
        let cs = cases();
        let seqs: Vec<_> = cs.iter().map(|_| m.alloc_seq().unwrap()).collect();
        let mut pos = vec![0usize; cs.len()];
        let (mut out, mut next) = (Vec::new(), Vec::new());
        for (i, c) in cs.iter().enumerate() {
            let mut last = Vec::new();
            for ch in c.prompt.chunks(5) {
                last = m
                    .forward(&[SeqChunk { seq: seqs[i], pos: pos[i], tokens: ch, all_logits: false, draft: 0 }])
                    .unwrap()
                    .remove(0);
                pos[i] += ch.len();
            }
            next.push(argmax(&last));
            out.push(last);
        }
        for _ in 0..40 {
            let toks: Vec<[u32; 1]> = next.iter().map(|&t| [t]).collect();
            let chunks: Vec<SeqChunk> = (0..seqs.len())
                .map(|i| SeqChunk { seq: seqs[i], pos: pos[i], tokens: &toks[i], all_logits: false, draft: 0 })
                .collect();
            let rows = m.forward(&chunks).unwrap();
            for (i, r) in rows.iter().enumerate() {
                pos[i] += 1;
                next[i] = argmax(r);
            }
            out.extend(rows);
        }
        (out, *pos.iter().max().unwrap())
    };
    let (eager, longest) = run(true);
    let (graphs, _) = run(false);
    assert_eq!(eager.len(), graphs.len());
    let differ = eager.iter().zip(&graphs).filter(|(a, b)| a != b).count();
    eprintln!("CUDA graphs vs op by op: {} logit rows, {differ} differ (context up to {longest})", eager.len());
    assert_eq!(differ, 0, "CUDA graph replay changed logits");
    assert!(longest > 32, "the test must cross an attention split boundary");
}

/// Greedy steps take the argmax on the device: the same tokens as argmax over the returned
/// logits, on every executor backend, for single and batched steps.
#[test]
fn kimi_linear_device_argmax_matches_host() {
    let mut bes = vec![BackendKind::CPU];
    if fullm_kernels::exec::cuda_available() {
        bes.push(BackendKind::CUDA);
    }
    for be in bes {
        let mut a = load(be.clone(), CacheDtype::F32, 64);
        let mut b = load(be.clone(), CacheDtype::F32, 64);
        let cs = cases();
        let sa: Vec<_> = cs.iter().map(|_| a.alloc_seq().unwrap()).collect();
        let sb: Vec<_> = cs.iter().map(|_| b.alloc_seq().unwrap()).collect();
        let chunks = |s: &[usize]| -> Vec<SeqChunk> {
            cs.iter()
                .enumerate()
                .map(|(i, c)| SeqChunk { seq: s[i], pos: 0, tokens: &c.prompt, all_logits: false, draft: 0 })
                .collect()
        };
        let want: Vec<u32> = a.forward(&chunks(&sa)).unwrap().iter().map(|l| argmax(l)).collect();
        let got = b.forward_greedy(&chunks(&sb)).unwrap();
        assert_eq!(want, got, "{be:?}: device argmax differs from the host's");
        let mut pos: Vec<usize> = cs.iter().map(|c| c.prompt.len()).collect();
        let mut next = want;
        for _ in 0..6 {
            let toks: Vec<[u32; 1]> = next.iter().map(|&t| [t]).collect();
            let mk = |s: &[usize]| -> Vec<SeqChunk> {
                (0..s.len())
                    .map(|i| SeqChunk { seq: s[i], pos: pos[i], tokens: &toks[i], all_logits: false, draft: 0 })
                    .collect()
            };
            let want: Vec<u32> = a.forward(&mk(&sa)).unwrap().iter().map(|l| argmax(l)).collect();
            let got = b.forward_greedy(&mk(&sb)).unwrap();
            assert_eq!(want, got, "{be:?}: device argmax differs from the host's in decode");
            for p in pos.iter_mut() {
                *p += 1;
            }
            next = got;
        }
        eprintln!("{be:?}: device argmax == host argmax over prefill + 6 batched decode steps");
    }
}
