// SPDX-License-Identifier: Apache-2.0
//! Speculative decoding (n-gram drafts) end to end on the tiny Qwen3-MoE.
//!
//! The CPU backend is batch-invariant, so a greedy request's tokens with drafts must be exactly
//! its tokens without: whatever was drafted and however much of it the model accepted. The
//! prefix cache must hold only accepted tokens (continuations of finished requests hit it and
//! still match), and a stop id or max_tokens inside accepted drafts must end a request where
//! decoding token by token would.

use std::path::PathBuf;

use fullm::models::{self, BackendKind, CacheDtype, LoadOptions};
use fullm::runtime::{Engine, EngineConfig, EngineHandle, Event, FinishReason, Request, SamplingParams, Stats};
use fullm::tensor::Precision;
use tokio::sync::mpsc::UnboundedReceiver;

const PAGE: usize = 4;
/// A stop id outside the vocabulary: the request runs to max_tokens.
const NEVER: u32 = u32::MAX - 1;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny_qwen3_moe")
}

fn engine(spec_tokens: usize, prefix: bool) -> EngineHandle {
    engine_with(spec_tokens, prefix, true)
}

fn engine_with(spec_tokens: usize, prefix: bool, spec_global: bool) -> EngineHandle {
    let opts = LoadOptions {
        trunk: Precision::Bf16,
        max_seqs: 6,
        cache_tokens: 4096,
        page: PAGE,
        backend: BackendKind::CPU,
        kv_dtype: CacheDtype::F32,
        step_tokens: 32,
        max_rows: 64,
        max_context: 512,
        threads: 2,
        no_prefix_cache: !prefix,
        ..Default::default()
    };
    let m = models::load(&fixture(), &opts).expect("load tiny qwen3_moe");
    let cfg = EngineConfig { step_tokens: 32, spec_tokens, spec_ngram: (1, 3), spec_global, ..Default::default() };
    Engine::new(m, cfg).spawn().unwrap()
}

type Req = (Vec<u32>, SamplingParams);

fn greedy(n: usize) -> SamplingParams {
    SamplingParams { max_tokens: n, stop_token_ids: vec![NEVER], ..Default::default() }
}

#[derive(Clone, Debug, PartialEq)]
struct Out {
    tokens: Vec<u32>,
    reason: FinishReason,
    cached: usize,
}

fn finish(rx: &mut UnboundedReceiver<Event>) -> Out {
    let mut tokens = Vec::new();
    loop {
        match rx.blocking_recv().expect("engine alive") {
            Event::Logprobs(_) => {}
            Event::Token(t) => tokens.push(t),
            Event::Finished { reason, usage } => return Out { tokens, reason, cached: usage.cached_tokens },
        }
    }
}

/// Submit every request at once, then collect them in order.
fn run_all(h: &EngineHandle, reqs: &[Req]) -> Vec<Out> {
    let mut rxs: Vec<_> =
        reqs.iter().map(|r| h.submit(Request { prompt: r.0.clone(), params: r.1.clone() }).unwrap()).collect();
    rxs.iter_mut().map(finish).collect()
}

/// Prompts that repeat themselves (a phrase several times, then its start), so drafts are found
/// from the first decode on.
fn prompts(n: usize, gen: usize) -> Vec<Req> {
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    (0..n)
        .map(|i| {
            let phrase: Vec<u32> = (0..10 + i)
                .map(|_| {
                    x ^= x >> 12;
                    x ^= x << 25;
                    x ^= x >> 27;
                    (x.wrapping_mul(0x2545_f491_4f6c_dd1d) % 250) as u32 + 1
                })
                .collect();
            let mut p: Vec<u32> = phrase.iter().copied().cycle().take(phrase.len() * 3).collect();
            p.extend_from_slice(&phrase[..3]);
            (p, greedy(gen))
        })
        .collect()
}

/// The engine's stats once its last step is counted (a request's end reaches the client before
/// the step that ended it updates them).
fn settled(h: &EngineHandle) -> Stats {
    let t = std::time::Instant::now();
    loop {
        let s = h.stats();
        if s.running == 0 && s.waiting == 0 {
            return s;
        }
        assert!(t.elapsed() < std::time::Duration::from_secs(30), "the engine never went idle: {s:?}");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

fn tokens(outs: &[Out]) -> Vec<Vec<u32>> {
    outs.iter().map(|o| o.tokens.clone()).collect()
}

#[test]
fn drafts_change_no_token() {
    let reqs = prompts(6, 48);
    for prefix in [false, true] {
        let want = run_all(&engine(0, prefix), &reqs);
        let h = engine(4, prefix);
        let got = run_all(&h, &reqs);
        assert_eq!(tokens(&got), tokens(&want), "prefix cache {prefix}");
        let s = settled(&h);
        assert!(s.spec_drafted > 0 && s.spec_accepted > 0, "drafts were verified and some kept: {s:?}");
        // one request alone: every accepted draft is a step saved
        let solo = engine(4, prefix);
        let one = run_all(&solo, &reqs[..1]);
        assert_eq!(one[0].tokens, want[0].tokens);
        let s = settled(&solo);
        let decode_steps = s.decode_tokens;
        assert_eq!(decode_steps + s.spec_accepted, want[0].tokens.len() as u64 - 1, "{s:?}");
    }
}

#[test]
fn continuations_hit_the_cache_and_still_match() {
    let reqs = prompts(4, 40);
    let h = engine(4, true);
    let first = run_all(&h, &reqs);
    // each prompt continued with its reply and two more tokens: the reply's pages, filled by
    // accepted drafts, come from the prefix cache
    let next: Vec<Req> =
        reqs.iter().zip(&first).map(|((p, _), o)| ([p.as_slice(), &o.tokens, &[7, 9]].concat(), greedy(24))).collect();
    let got = run_all(&h, &next);
    assert!(got.iter().all(|o| o.cached >= next[0].0.len() / 2), "continuations hit: {got:?}");
    let want = run_all(&engine(0, false), &next);
    assert_eq!(tokens(&got), tokens(&want));
}

#[test]
fn stops_and_max_tokens_inside_drafts() {
    let reqs = prompts(4, 40);
    let base = run_all(&engine(0, false), &reqs);
    // a stop id: the tenth token each request generates
    let stopped: Vec<Req> = reqs
        .iter()
        .zip(&base)
        .map(|((p, _), o)| (p.clone(), SamplingParams { stop_token_ids: vec![o.tokens[9]], ..greedy(40) }))
        .collect();
    // max_tokens from 1 to 6
    let short: Vec<Req> = (0..6).map(|k| (reqs[k % reqs.len()].0.clone(), greedy(k + 1))).collect();
    for batch in [&stopped, &short] {
        let want = run_all(&engine(0, false), batch);
        let got = run_all(&engine(4, false), batch);
        assert_eq!(got, want);
    }
    let got = run_all(&engine(4, false), &stopped);
    assert!(got.iter().all(|o| o.reason == FinishReason::Stop && o.tokens.len() <= 9), "{got:?}");
}

#[test]
fn sampled_replies_with_drafts_equal_those_without() {
    // the sampler draws a position's token from a stream keyed by the position, not by the draws
    // made before it: a seeded request generates the same tokens with drafts, whichever it kept
    let reqs: Vec<Req> = prompts(6, 48)
        .into_iter()
        .enumerate()
        .map(|(i, (p, _))| {
            let (temperature, top_k) = if i % 2 == 0 { (0.3, 0) } else { (0.7, 20) };
            (p, SamplingParams { temperature, top_p: 0.9, top_k, seed: 1000 + i as u64, ..greedy(48) })
        })
        .collect();
    for prefix in [false, true] {
        let want = run_all(&engine(0, prefix), &reqs);
        let h = engine(4, prefix);
        let got = run_all(&h, &reqs);
        assert_eq!(tokens(&got), tokens(&want), "prefix cache {prefix}");
        let s = settled(&h);
        assert!(s.spec_drafted > 0 && s.spec_accepted > 0, "drafts were verified and some kept: {s:?}");
    }
}

#[test]
fn drafts_from_other_replies_change_no_token() {
    // a prompt that never repeats itself, sent five times one after another: from the fourth on,
    // three replies before it say what comes next (the request's own context has no match at first)
    let prompt: Vec<u32> = (0..40u32).map(|i| (i * 37 + 11) % 251 + 1).collect();
    let run = |global: bool| {
        let h = engine_with(4, false, global);
        let outs: Vec<Vec<u32>> =
            (0..5).map(|_| run_all(&h, &[(prompt.clone(), greedy(40))])[0].tokens.clone()).collect();
        (outs, settled(&h))
    };
    let (want, _) = run(false);
    let (got, s) = run(true);
    assert_eq!(got, want);
    assert!(want.iter().all(|o| *o == want[0]), "greedy repeats itself");
    assert!(s.spec_drafted_global > 0 && s.spec_accepted_global > 0, "drafts came from other replies: {s:?}");
}
