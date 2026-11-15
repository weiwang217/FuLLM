// SPDX-License-Identifier: Apache-2.0
//! Sampling parameters beyond temperature / top-k / top-p through the engine, on the host sampler
//! (the tiny Kimi K3 on the reference backend: logits to the host) and on an executor's
//! `sample_rows` (the tiny Qwen3-MoE on the CPU executor: the device path in the backends' base
//! forms; and on CUDA when there is a GPU: its kernels): logit bias, min_tokens, penalties, min-p,
//! and log-probabilities.

use std::path::PathBuf;

use fullm::models::{self, BackendKind, LoadOptions};
use fullm::runtime::sampler::TokenLogprobs;
use fullm::runtime::{Engine, EngineConfig, EngineHandle, Event, FinishReason, Request, SamplingParams};
use fullm::tensor::Precision;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

/// (name, engine, a prompt): the host path and the executor's path.
fn engines() -> Vec<(&'static str, EngineHandle, Vec<u32>)> {
    let load = |dir: &str, backend: BackendKind| {
        let opts = LoadOptions {
            trunk: Precision::Bf16,
            max_seqs: 4,
            cache_tokens: 512,
            page: 4,
            backend,
            max_rows: 64,
            step_tokens: 64,
            max_context: 256,
            threads: 4,
            ..Default::default()
        };
        let m = models::load(&fixtures().join(dir), &opts).unwrap();
        Engine::new(m, EngineConfig::default()).spawn().unwrap()
    };
    let prompt = vec![3, 14, 15, 92, 65, 35, 89, 79];
    let mut v = vec![
        ("host (tiny_k3, reference)", load("tiny_k3", BackendKind::Reference), prompt.clone()),
        ("executor (tiny_qwen3_moe, cpu)", load("tiny_qwen3_moe", BackendKind::CPU), prompt.clone()),
    ];
    if fullm_kernels::exec::cuda_available() {
        v.push(("executor (tiny_qwen3_moe, cuda)", load("tiny_qwen3_moe", BackendKind::CUDA), prompt));
    }
    v
}

struct Out {
    tokens: Vec<u32>,
    lps: Vec<TokenLogprobs>,
    reason: FinishReason,
}

fn run(h: &EngineHandle, prompt: &[u32], params: SamplingParams) -> Out {
    let mut rx = h.submit(Request { prompt: prompt.to_vec(), params }).unwrap();
    let (mut tokens, mut lps) = (Vec::new(), Vec::new());
    loop {
        match rx.blocking_recv().expect("engine alive") {
            Event::Logprobs(l) => lps.push(l),
            Event::Token(t) => tokens.push(t),
            Event::Finished { reason, .. } => return Out { tokens, lps, reason },
        }
    }
}

fn greedy(n: usize) -> SamplingParams {
    SamplingParams { max_tokens: n, ..Default::default() }
}

#[test]
fn sampling_parameters_through_the_engine() {
    for (name, h, prompt) in engines() {
        let base = run(&h, &prompt, greedy(12));
        assert_eq!(base.tokens.len(), 12, "{name}");
        assert!(base.lps.is_empty(), "{name}: no log-probabilities unless asked");

        // logit bias: +100 on one token makes it every greedy pick
        let x = (base.tokens[0] + 7) % 200 + 20;
        let biased = run(&h, &prompt, SamplingParams { logit_bias: vec![(x, 100.0)], ..greedy(6) });
        assert_eq!(biased.tokens, vec![x; 6], "{name}: logit bias");

        // min_tokens: a stop id the bias makes most likely is held back until min_tokens
        let held = run(
            &h,
            &prompt,
            SamplingParams { logit_bias: vec![(x, 100.0)], stop_token_ids: vec![x], min_tokens: 5, ..greedy(20) },
        );
        assert_eq!(held.tokens.len(), 5, "{name}: min_tokens ({:?})", held.tokens);
        assert!(!held.tokens.contains(&x), "{name}: the stop id drawn before min_tokens");
        assert_eq!(held.reason, FinishReason::Stop, "{name}");

        // penalties: a token the output holds loses to the others
        let rep = run(
            &h,
            &prompt,
            SamplingParams { logit_bias: vec![(x, 3.0)], frequency_penalty: 2.0, presence_penalty: 2.0, ..greedy(8) },
        );
        let free = run(&h, &prompt, SamplingParams { logit_bias: vec![(x, 3.0)], ..greedy(8) });
        assert_ne!(rep.tokens, free.tokens, "{name}: frequency / presence penalties changed nothing");
        // repetition penalty: the biased token wins the first step, then its (positive) logit is
        // divided by 50 once the output holds it
        let pinned = run(&h, &prompt, SamplingParams { logit_bias: vec![(x, 5.0)], ..greedy(4) });
        assert_eq!(pinned.tokens, vec![x; 4], "{name}");
        let rp = run(&h, &prompt, SamplingParams { logit_bias: vec![(x, 5.0)], repetition_penalty: 50.0, ..greedy(4) });
        assert_eq!(rp.tokens[0], x, "{name}");
        assert_ne!(rp.tokens[1], x, "{name}: repetition penalty changed nothing ({:?})", rp.tokens);

        // min-p 1 at temperature 1: the most likely token only, the greedy reply
        let mp = run(&h, &prompt, SamplingParams { temperature: 1.0, min_p: 1.0, seed: 3, ..greedy(12) });
        assert_eq!(mp.tokens, base.tokens, "{name}: min_p 1");

        // log-probabilities: greedy draws are each row's most likely token
        let lp = run(&h, &prompt, SamplingParams { logprobs: Some(4), ..greedy(12) });
        assert_eq!(lp.tokens, base.tokens, "{name}: asking for log-probabilities changed the reply");
        assert_eq!(lp.lps.len(), lp.tokens.len(), "{name}");
        for (t, l) in lp.tokens.iter().zip(&lp.lps) {
            assert_eq!(l.top.len(), 4, "{name}");
            assert_eq!(l.top[0].0, *t, "{name}: greedy token is not the most likely");
            assert!((l.top[0].1 - l.logprob).abs() < 1e-6, "{name}");
            assert!(l.top.windows(2).all(|w| w[0].1 >= w[1].1), "{name}: alternatives out of order");
            let mass: f32 = l.top.iter().map(|x| x.1.exp()).sum();
            assert!(l.logprob <= 0.0 && mass <= 1.0 + 1e-5, "{name}: {l:?}");
        }
        // they are of the model's logits: a bias does not move them
        let lpb = run(&h, &prompt, SamplingParams { logprobs: Some(2), logit_bias: vec![(x, 100.0)], ..greedy(1) });
        assert_eq!(lpb.tokens, vec![x], "{name}");
        assert_eq!(lpb.lps[0].top[0].0, base.tokens[0], "{name}: log-probabilities after the bias");
    }
}
