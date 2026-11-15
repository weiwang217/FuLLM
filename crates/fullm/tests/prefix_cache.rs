// SPDX-License-Identifier: Apache-2.0
//! The prefix cache, optimistic admission and preemption end to end, on the tiny Qwen3-MoE.
//!
//! The CPU backend is batch-invariant, so a request's tokens must be exactly those of the same
//! request run alone on an engine with neither a prefix cache nor preemption: whatever ran beside
//! it, whatever the cache held, however often it was preempted. Every test checks that, and that
//! nothing leaks: once the engine is idle and flushed, every page and sequence slot is free.
//! The host tier's tests run the same checks with a device cache too small for the workload, so
//! hits come back from host memory (and, with a copy lag, requests wait for their loads).

use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use fullm::models::{self, BackendKind, CacheDtype, LoadOptions};
use fullm::runtime::{
    Admission, Engine, EngineConfig, EngineHandle, Event, FinishReason, Request, SamplingParams, Stats,
};
use fullm::server::{router, AppState};
use fullm::tensor::Precision;
use fullm::tokenizer::ByteTokenizer;
use tokio::sync::mpsc::UnboundedReceiver;

const PAGE: usize = 4;
/// A stop id outside the vocabulary: the request counts as one that may stop (the optimistic
/// estimate applies) and runs to max_tokens.
const NEVER: u32 = u32::MAX - 1;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny_qwen3_moe")
}

#[derive(Clone, Copy)]
struct Setup {
    cache: usize,
    max_seqs: usize,
    prefix: bool,
    admission: Admission,
    step_tokens: usize,
    max_context: usize,
    conservativeness: f64,
    /// Host-tier positions (0 = none), and polls a load reports in flight.
    host: usize,
    lag: usize,
}

impl Setup {
    fn new(cache: usize) -> Self {
        Setup {
            cache,
            max_seqs: 6,
            prefix: true,
            admission: Admission::Optimistic,
            step_tokens: 12,
            max_context: 256,
            conservativeness: 1.0,
            host: 0,
            lag: 0,
        }
    }

    fn with_context(self, max_context: usize) -> Self {
        Setup { max_context, ..self }
    }

    fn spawn(&self) -> EngineHandle {
        let opts = LoadOptions {
            trunk: Precision::Bf16,
            max_seqs: self.max_seqs,
            cache_tokens: self.cache,
            page: PAGE,
            backend: BackendKind::CPU,
            kv_dtype: CacheDtype::F32,
            step_tokens: self.step_tokens,
            max_rows: self.max_seqs.max(64),
            max_context: self.max_context,
            threads: 2,
            no_prefix_cache: !self.prefix,
            host_cache_tokens: self.host,
            host_copy_lag: self.lag,
            ..Default::default()
        };
        let m = models::load(&fixture(), &opts).expect("load tiny qwen3_moe");
        let cfg = EngineConfig {
            step_tokens: self.step_tokens,
            admission: self.admission,
            conservativeness: self.conservativeness,
            ..Default::default()
        };
        Engine::new(m, cfg).spawn().unwrap()
    }
}

type Req = (Vec<u32>, SamplingParams);

fn greedy(n: usize) -> SamplingParams {
    SamplingParams { max_tokens: n, stop_token_ids: vec![NEVER], ..Default::default() }
}

fn submit(h: &EngineHandle, r: &Req) -> UnboundedReceiver<Event> {
    h.submit(Request { prompt: r.0.clone(), params: r.1.clone() }).unwrap()
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

fn one(h: &EngineHandle, r: &Req) -> Out {
    finish(&mut submit(h, r))
}

/// Submit every request at once, then collect them in order.
fn run_all(h: &EngineHandle, reqs: &[Req]) -> Vec<Out> {
    let mut rxs: Vec<_> = reqs.iter().map(|r| submit(h, r)).collect();
    rxs.iter_mut().map(finish).collect()
}

/// An engine that runs each request alone, with neither prefix cache nor preemption: the reference.
fn reference(max_context: usize) -> EngineHandle {
    let cache = 4096.max(max_context);
    Setup { cache, max_seqs: 1, prefix: false, admission: Admission::WorstCase, step_tokens: 64, ..Setup::new(0) }
        .with_context(max_context)
        .spawn()
}

fn solo(reqs: &[Req], max_context: usize) -> Vec<Out> {
    let h = reference(max_context);
    reqs.iter().map(|r| one(&h, r)).collect()
}

/// Wait for the engine to go idle (a cancelled request leaves at its next token), flush the prefix
/// cache, and check that every page and slot is free. Returns the stats from before the flush.
fn settle(h: &EngineHandle, s: &Setup) -> Stats {
    let t = Instant::now();
    loop {
        let st = h.stats();
        if st.running == 0 && st.waiting == 0 {
            break;
        }
        assert!(t.elapsed() < Duration::from_secs(60), "the engine never went idle: {st:?}");
        thread::sleep(Duration::from_millis(2));
    }
    let before = h.stats();
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    rt.block_on(h.flush_cache()).unwrap();
    let after = h.stats();
    assert_eq!(after.capacity.free_cache_tokens, s.cache.div_ceil(PAGE) * PAGE, "pages leaked: {after:?}");
    assert_eq!(after.capacity.prefix_cached_tokens, 0, "the flush left cached pages");
    assert_eq!(after.capacity.host_cached_tokens, 0, "the flush left host pages");
    assert_eq!(after.capacity.free_seqs, s.max_seqs, "sequence slots leaked");
    before
}

/// xorshift64*: deterministic randomness without a dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn tokens(&mut self, n: usize) -> Vec<u32> {
        (0..n).map(|_| self.below(256) as u32).collect()
    }
}

#[test]
fn hits_at_every_prompt_length() {
    let setup = Setup::new(1024);
    let h = setup.spawn();
    let mut rng = Rng(11);
    let reqs: Vec<Req> = (1..=14).map(|len| (rng.tokens(len), greedy(5))).collect();
    let want = solo(&reqs, 256);
    for (r, w) in reqs.iter().zip(&want) {
        for round in 0..2 {
            let got = one(&h, r);
            assert_eq!(got.tokens, w.tokens, "length {}, round {round}", r.0.len());
            // a hit covers whole pages and leaves the last prompt token to compute
            let expect = if round == 0 { 0 } else { (r.0.len() - 1) / PAGE * PAGE };
            assert_eq!(got.cached, expect, "length {}, round {round}", r.0.len());
        }
    }
    settle(&h, &setup);
}

#[test]
fn prompts_diverging_at_every_offset() {
    let setup = Setup::new(1024);
    let h = setup.spawn();
    let mut rng = Rng(12);
    let base = rng.tokens(24);
    one(&h, &(base.clone(), greedy(3)));
    let reqs: Vec<Req> = (0..base.len())
        .map(|k| {
            let mut v = base.clone();
            v[k] = (v[k] + 1) % 256;
            (v, greedy(4))
        })
        .collect();
    let want = solo(&reqs, 256);
    for (k, (r, w)) in reqs.iter().zip(&want).enumerate() {
        let got = one(&h, r);
        assert_eq!(got.tokens, w.tokens, "diverging at {k}");
        assert_eq!(got.cached, (k / PAGE).min((base.len() - 1) / PAGE) * PAGE, "diverging at {k}");
    }
    settle(&h, &setup);
}

/// Prompts of 1500-2048 tokens over many chunks: a hit across hundreds of pages, a prompt just
/// short of the context (it stops at the limit), and one exactly as long as the context.
#[test]
fn long_prompts_and_the_context_limit() {
    let setup = Setup { cache: 8192, max_seqs: 4, step_tokens: 64, max_context: 2048, ..Setup::new(0) };
    let h = setup.spawn();
    let mut rng = Rng(21);
    let p1 = rng.tokens(1500);
    let p2 = [p1.clone(), rng.tokens(300)].concat();
    let near = rng.tokens(2040);
    let full = rng.tokens(2048);
    let reqs: Vec<Req> =
        vec![(p1, greedy(8)), (p2, greedy(8)), (near.clone(), greedy(20)), (near, greedy(20)), (full, greedy(5))];
    let want = solo(&reqs, 2048);
    let got: Vec<Out> = reqs.iter().map(|r| one(&h, r)).collect();
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!((&g.tokens, &g.reason), (&w.tokens, &w.reason), "request {i}");
    }
    let cached: Vec<usize> = got.iter().map(|g| g.cached).collect();
    assert_eq!(cached, [0, 1500, 0, 2036, 0]);
    assert_eq!(got[2].reason, FinishReason::ContextLimit);
    assert_eq!((got[4].tokens.len(), &got[4].reason), (1, &FinishReason::ContextLimit));
    settle(&h, &setup);
}

/// Three sessions of four turns over a shared 900-token system prompt, run concurrently in a cache
/// smaller than all of them (contexts reach ~1500 tokens each): later turns hit their own history
/// unless pressure evicted it, and every turn gets the reference tokens.
/// Three sessions of four turns over a 900-token system prompt, each turn's prompt the previous
/// one plus the reply and 120 new tokens; run concurrently, every turn checked against the
/// reference. Returns the stats and the prompt tokens served from the cache.
fn sessions(setup: &Setup) -> (Stats, usize) {
    let mut rng = Rng(22);
    let system = rng.tokens(900);
    // each turn's prompt depends on the previous reply: build them from the reference engine
    let refh = reference(2048);
    let sessions: Vec<Vec<(Req, Out)>> = (0..3)
        .map(|_| {
            let mut prompt = [system.clone(), rng.tokens(40)].concat();
            (0..4)
                .map(|_| {
                    let r = (prompt.clone(), greedy(8));
                    let w = one(&refh, &r);
                    prompt = [prompt.as_slice(), &w.tokens, &rng.tokens(120)].concat();
                    (r, w)
                })
                .collect()
        })
        .collect();
    let h = setup.spawn();
    let threads: Vec<_> = sessions
        .into_iter()
        .map(|turns| {
            let h = h.clone();
            thread::spawn(move || {
                let mut cached = 0;
                for (t, (r, w)) in turns.iter().enumerate() {
                    let got = one(&h, r);
                    assert_eq!((&got.tokens, &got.reason), (&w.tokens, &w.reason), "turn {t}");
                    cached += got.cached;
                }
                cached
            })
        })
        .collect();
    let cached: usize = threads.into_iter().map(|t| t.join().unwrap()).sum();
    (settle(&h, setup), cached)
}

#[test]
fn long_sessions_concurrently_under_pressure() {
    let setup = Setup { cache: 3000, max_seqs: 4, step_tokens: 64, max_context: 2048, ..Setup::new(0) };
    let (st, cached) = sessions(&setup);
    eprintln!(
        "long sessions: {cached} of {} prompt tokens from the cache, {} preemptions",
        st.prompt_tokens, st.preemptions
    );
    assert!(cached * 2 > st.prompt_tokens as usize, "most of each turn should be a hit");
}

/// Sixteen identical prompts at once: the first computes the prefix, the others wait for it and
/// then hit, so the prompt is not computed sixteen times.
#[test]
fn a_burst_of_identical_prompts_computes_the_prefix_once() {
    let setup = Setup { cache: 2048, max_seqs: 16, step_tokens: 16, ..Setup::new(0) };
    let h = setup.spawn();
    let p = Rng(31).tokens(61);
    let reqs = vec![(p, greedy(6)); 16];
    let want = solo(&reqs[..1], 256).remove(0);
    let got = run_all(&h, &reqs);
    assert!(got.iter().all(|g| g.tokens == want.tokens));
    assert_eq!(got[0].cached, 0);
    assert!(got[1..].iter().all(|g| g.cached >= 48), "{:?}", got.iter().map(|g| g.cached).collect::<Vec<_>>());
    let st = settle(&h, &setup);
    assert!(st.prefill_tokens <= 61 + 15 * 13, "prefilled {} tokens", st.prefill_tokens);
}

/// A request as long as the whole cache (context = cache): two of them take turns and each stops
/// at the context limit.
#[test]
fn requests_as_long_as_the_cache() {
    let setup = Setup { cache: 64, max_seqs: 3, step_tokens: 8, max_context: 64, ..Setup::new(0) };
    let h = setup.spawn();
    let mut rng = Rng(41);
    let reqs: Vec<Req> = (0..3).map(|_| (rng.tokens(30), greedy(100))).collect();
    let want = solo(&reqs, 64);
    let got = run_all(&h, &reqs);
    for (g, w) in got.iter().zip(&want) {
        assert_eq!((&g.tokens, &g.reason), (&w.tokens, &FinishReason::ContextLimit));
    }
    settle(&h, &setup);
}

struct Job {
    req: Req,
    arrive_ms: u64,
    /// Drop the receiver after this many tokens (the client went away).
    cancel_after: Option<usize>,
}

/// Requests over three shared system prompts (some repeated exactly), 1-24 tokens each (some
/// sampled with seeds, some with a stop id that may fire), arriving over ~40 ms, a fifth of them
/// cancelled after 0-3 tokens.
fn workload(seed: u64, n: usize) -> Vec<Job> {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let systems = [rng.tokens(20), rng.tokens(37), rng.tokens(52)];
    let mut prompts: Vec<Vec<u32>> = Vec::new();
    (0..n)
        .map(|_| {
            let prompt = if !prompts.is_empty() && rng.below(10) == 0 {
                prompts[rng.below(prompts.len())].clone()
            } else {
                let (system, tail) = (rng.below(3), 1 + rng.below(30));
                [systems[system].clone(), rng.tokens(tail)].concat()
            };
            prompts.push(prompt.clone());
            let mut p = greedy(1 + rng.below(24));
            if rng.below(10) < 3 {
                p.stop_token_ids.push(rng.below(256) as u32);
            }
            if rng.below(3) == 0 {
                (p.temperature, p.top_p, p.top_k, p.seed) = (0.7, 0.9, 20, rng.next());
            }
            let cancel_after = (rng.below(5) == 0).then(|| rng.below(4));
            Job { req: (prompt, p), arrive_ms: rng.below(40) as u64, cancel_after }
        })
        .collect()
}

enum Got {
    Done(Out),
    Cancelled(Vec<u32>),
}

/// Every job on its own thread: arrive, submit, read (or give up after `cancel_after` tokens).
fn run_jobs(h: &EngineHandle, jobs: &[Job]) -> Vec<Got> {
    let threads: Vec<_> = jobs
        .iter()
        .map(|j| {
            let (h, req, arrive, cancel) = (h.clone(), j.req.clone(), j.arrive_ms, j.cancel_after);
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(arrive));
                let mut rx = submit(&h, &req);
                let Some(k) = cancel else { return Got::Done(finish(&mut rx)) };
                let mut tokens = Vec::new();
                while tokens.len() < k {
                    match rx.blocking_recv().expect("engine alive") {
                        Event::Logprobs(_) => {}
                        Event::Token(t) => tokens.push(t),
                        Event::Finished { .. } => break,
                    }
                }
                Got::Cancelled(tokens)
            })
        })
        .collect();
    threads.into_iter().map(|t| t.join().unwrap()).collect()
}

fn check_jobs(got: &[Got], want: &[Out], label: &str) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        match g {
            Got::Done(o) => {
                assert_eq!((&o.tokens, &o.reason), (&w.tokens, &w.reason), "{label}: request {i}");
            }
            Got::Cancelled(t) => assert!(w.tokens.starts_with(t), "{label}: cancelled request {i} diverged"),
        }
    }
}

fn reqs(jobs: &[Job]) -> Vec<Req> {
    jobs.iter().map(|j| j.req.clone()).collect()
}

/// The main concurrency test: random arrivals, shared and repeated prompts, sampled and stopping
/// requests, cancellations, in a tight cache (constant eviction), a roomy one, and the tight one
/// with very optimistic admission (constant preemption, cancellations of preempted requests).
#[test]
fn random_arrivals_cancellations_and_preemption() {
    let (mut preemptions, mut cached) = (0, 0);
    for seed in 1..=3 {
        let jobs = workload(seed, 45);
        let want = solo(&reqs(&jobs), 256);
        for (cache, conservativeness) in [(160, 1.0), (400, 1.0), (160, 0.05), (120, 0.02)] {
            let setup = Setup { conservativeness, ..Setup::new(cache) };
            let h = setup.spawn();
            let got = run_jobs(&h, &jobs);
            check_jobs(&got, &want, &format!("seed {seed}, cache {cache}, conservativeness {conservativeness}"));
            let st = settle(&h, &setup);
            eprintln!(
                "seed {seed}, cache {cache}, conservativeness {conservativeness}: {} preemptions ({} positions \
                 recomputed), {} of {} prompt tokens cached",
                st.preemptions, st.recomputed_tokens, st.cached_tokens, st.prompt_tokens
            );
            preemptions += st.preemptions;
            cached += st.cached_tokens;
        }
    }
    assert!(preemptions >= 5 && cached > 0, "preemptions {preemptions}, cached {cached}");
}

#[test]
fn worst_case_admission_never_preempts() {
    let jobs = workload(4, 45);
    let want = solo(&reqs(&jobs), 256);
    let setup = Setup { admission: Admission::WorstCase, ..Setup::new(160) };
    let h = setup.spawn();
    check_jobs(&run_jobs(&h, &jobs), &want, "worst case");
    let st = settle(&h, &setup);
    assert_eq!(st.preemptions, 0);
    assert!(st.cached_tokens > 0);
}

/// Without a prefix cache a preempted request recomputes its whole context: still the same tokens
/// (admission made very optimistic so that preemption is certain).
#[test]
fn preemption_without_a_prefix_cache() {
    let mut preemptions = 0;
    for seed in [5, 6] {
        let jobs = workload(seed, 45);
        let want = solo(&reqs(&jobs), 256);
        let setup = Setup { prefix: false, conservativeness: 0.05, ..Setup::new(160) };
        let h = setup.spawn();
        check_jobs(&run_jobs(&h, &jobs), &want, "no prefix cache");
        let st = settle(&h, &setup);
        assert_eq!(st.cached_tokens, 0);
        preemptions += st.preemptions;
    }
    assert!(preemptions > 0, "the cache never ran short");
}

/// Flushing the cache over and over while requests run only drops what no request holds.
#[test]
fn flushing_while_requests_run() {
    let jobs = workload(7, 45);
    let want = solo(&reqs(&jobs), 256);
    let setup = Setup::new(400);
    let h = setup.spawn();
    let stop = Arc::new(AtomicBool::new(false));
    let flusher = {
        let (h, stop) = (h.clone(), stop.clone());
        thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
            let mut n = 0;
            while !stop.load(Ordering::Relaxed) {
                rt.block_on(h.flush_cache()).unwrap();
                n += 1;
                thread::sleep(Duration::from_micros(300));
            }
            n
        })
    };
    let got = run_jobs(&h, &jobs);
    stop.store(true, Ordering::Relaxed);
    let flushes = flusher.join().unwrap();
    check_jobs(&got, &want, "flushing");
    settle(&h, &setup);
    assert!(flushes > 10, "{flushes} flushes");
}

/// Minimal blocking HTTP/1.1 client: the response's status and raw body.
fn http(addr: std::net::SocketAddr, method: &str, path: &str, body: &str) -> (u16, String) {
    let mut s = std::net::TcpStream::connect(addr).unwrap();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut raw = String::new();
    s.read_to_string(&mut raw).unwrap();
    let status = raw[9..12].parse().unwrap();
    (status, raw.split_once("\r\n\r\n").map(|(_, b)| b.to_string()).unwrap_or_default())
}

#[test]
fn http_cached_tokens_flush_and_metrics() {
    let setup = Setup::new(1024);
    let h = setup.spawn();
    let state = Arc::new(AppState {
        engine: h.clone(),
        tok: Arc::new(ByteTokenizer),
        model_name: "tiny".into(),
        special_tokens_in_prompts: false,
        sampling_defaults: Default::default(),
        grammars: Default::default(),
        loras: Default::default(),
        runtime_lora: false,
        api_key: None,
    });
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
    let addr = rt.block_on(async {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(l, router(state)).await.unwrap() });
        addr
    });
    let prompt = serde_json::to_string(&Rng(51).tokens(23)).unwrap();
    let complete = |stream: bool| {
        let body = format!(r#"{{"prompt": {prompt}, "max_tokens": 3, "temperature": 0, "stream": {stream}}}"#);
        let (st, b) = http(addr, "POST", "/v1/completions", &body);
        assert_eq!(st, 200, "{b}");
        let usage = if stream {
            b.lines()
                .filter_map(|l| l.strip_prefix("data: "))
                .filter(|d| *d != "[DONE]")
                .map(|d| serde_json::from_str::<serde_json::Value>(d).unwrap())
                .find_map(|v| v.get("usage").cloned())
                .expect("a streamed chunk with usage")
        } else {
            serde_json::from_str::<serde_json::Value>(&b).unwrap()["usage"].clone()
        };
        assert_eq!(usage["prompt_tokens"], 23);
        usage["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap()
    };
    assert_eq!(complete(false), 0);
    assert_eq!(complete(false), 20, "every whole page before the last token");
    assert_eq!(complete(true), 20);
    let (st, b) = http(addr, "POST", "/flush_cache", "");
    assert_eq!(st, 200, "{b}");
    let flushed = serde_json::from_str::<serde_json::Value>(&b).unwrap()["flushed_tokens"].as_u64().unwrap();
    assert!(flushed >= 20, "{b}");
    let (st, b) = http(addr, "POST", "/reset_prefix_cache", "");
    assert_eq!((st, b.as_str()), (200, r#"{"flushed_tokens":0}"#));
    assert_eq!(complete(false), 0, "the flush emptied the cache");
    let (st, b) = http(addr, "GET", "/metrics", "");
    assert_eq!(st, 200);
    let e = &serde_json::from_str::<serde_json::Value>(&b).unwrap()["engine"];
    assert_eq!((e["prompt_tokens"].as_u64(), e["cached_tokens"].as_u64()), (Some(92), Some(40)));
    assert_eq!(e["preemptions"], 0);
    assert!(e["output_ratio"].as_f64().is_some() && e["recomputed_tokens"].is_u64());
    settle(&h, &setup);
}

/// Heavy preemption: admission made very optimistic (conservativeness 0.05: outputs estimated at
/// ~4% of max_tokens), half the requests generating 40-60 tokens from short prompts, half
/// prefilling 50-80-token prompts over several steps (victims are often mid-prefill), in small
/// caches. Requests are preempted again and again, mid-prefill and mid-decode, and every one gets
/// the reference tokens.
#[test]
fn heavy_preemption_mid_prefill_and_mid_decode() {
    let reqs = heavy_reqs();
    let want = solo(&reqs, 256);
    let mut total = 0;
    for cache in [96, 120, 144, 200] {
        let setup = Setup { step_tokens: 8, max_seqs: 8, conservativeness: 0.05, ..Setup::new(cache) };
        let h = setup.spawn();
        let got = run_all(&h, &reqs);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!((&g.tokens, &g.reason), (&w.tokens, &w.reason), "cache {cache}: request {i}");
        }
        let st = settle(&h, &setup);
        eprintln!(
            "heavy, cache {cache}: {} preemptions, {} positions recomputed, {} of {} prompt tokens cached",
            st.preemptions, st.recomputed_tokens, st.cached_tokens, st.prompt_tokens
        );
        total += st.preemptions;
    }
    assert!(total >= 5, "only {total} preemptions");
}

/// Seven long generations from a short prompt and seven short ones from a long prompt, over two
/// system prompts, some sampled.
fn heavy_reqs() -> Vec<Req> {
    let mut rng = Rng(61);
    let systems = [rng.tokens(12), rng.tokens(30)];
    (0..14)
        .map(|i| {
            let (prompt, n) = if i % 2 == 0 {
                let tail = 1 + rng.below(10);
                ([systems[0].clone(), rng.tokens(tail)].concat(), 40 + rng.below(21))
            } else {
                let tail = 20 + rng.below(30);
                ([systems[1].clone(), rng.tokens(tail)].concat(), 2 + rng.below(6))
            };
            let mut p = greedy(n);
            if i % 3 == 0 {
                (p.temperature, p.top_k, p.seed) = (0.8, 30, rng.next());
            }
            (prompt, p)
        })
        .collect()
}

// ------------------------------------------------------------------ the host tier

/// A prompt whose device pages other requests evicted comes back from host memory: the same
/// tokens, every whole page cached, the positions counted as loaded; without the host tier the
/// same sequence of requests recomputes it.
#[test]
fn the_host_tier_serves_what_the_device_evicted() {
    let mut rng = Rng(71);
    let a: Req = (rng.tokens(61), greedy(4));
    let others: Vec<Req> = (0..3).map(|_| (rng.tokens(61), greedy(4))).collect();
    let want = solo(std::slice::from_ref(&a), 256);
    for host in [0, 1024] {
        let setup = Setup { host, ..Setup::new(96) };
        let h = setup.spawn();
        assert_eq!(one(&h, &a).tokens, want[0].tokens);
        for r in &others {
            one(&h, r);
        }
        let again = one(&h, &a);
        assert_eq!(again.tokens, want[0].tokens, "host {host}");
        let st = settle(&h, &setup);
        if host > 0 {
            assert_eq!(again.cached, 60, "every whole page from the host");
            assert!(st.host_loaded_tokens >= 56, "loaded {}", st.host_loaded_tokens);
        } else {
            assert!(again.cached < 60 && st.host_loaded_tokens == 0, "cached {}", again.cached);
        }
    }
}

/// The long sessions with a device cache that holds about one of them: with the host tier the
/// turns still hit (most of every prompt), without it they mostly recompute.
#[test]
fn long_sessions_through_the_host_tier() {
    let base = Setup { cache: 1600, max_seqs: 4, step_tokens: 64, max_context: 2048, ..Setup::new(0) };
    let (st0, cached0) = sessions(&base);
    let (st1, cached1) = sessions(&Setup { host: 8192, ..base });
    let (st2, cached2) = sessions(&Setup { host: 8192, lag: 2, ..base });
    eprintln!(
        "long sessions, device 1600: {cached0} of {} cached without the host tier; {cached1} of {} with it \
         ({} loaded); {cached2} with a copy lag ({} loaded)",
        st0.prompt_tokens, st1.prompt_tokens, st1.host_loaded_tokens, st2.host_loaded_tokens
    );
    assert!(cached1 > cached0 && cached1 * 4 > st1.prompt_tokens as usize * 3, "the host tier should serve most turns");
    assert!(st1.host_loaded_tokens > 0 && st2.host_loaded_tokens > 0);
}

/// The random workload in a tight device cache over a host tier: requests load, wait for their
/// loads (copy lag), get preempted while loading or after, are cancelled; every one gets the
/// reference tokens and nothing leaks on either tier.
#[test]
fn random_arrivals_with_a_host_tier() {
    let (mut loaded, mut preemptions) = (0, 0);
    for seed in 1..=3 {
        let jobs = workload(seed, 45);
        let want = solo(&reqs(&jobs), 256);
        for (cache, host, lag, conservativeness) in [(160, 1024, 0, 1.0), (160, 1024, 3, 1.0), (120, 400, 1, 0.02)] {
            let setup = Setup { host, lag, conservativeness, ..Setup::new(cache) };
            let h = setup.spawn();
            let label =
                format!("seed {seed}, cache {cache}, host {host}, lag {lag}, conservativeness {conservativeness}");
            check_jobs(&run_jobs(&h, &jobs), &want, &label);
            let st = settle(&h, &setup);
            eprintln!(
                "{label}: {} of {} prompt tokens cached, {} loaded, {} preemptions",
                st.cached_tokens, st.prompt_tokens, st.host_loaded_tokens, st.preemptions
            );
            loaded += st.host_loaded_tokens;
            preemptions += st.preemptions;
        }
    }
    assert!(loaded > 0 && preemptions > 0, "loaded {loaded}, preemptions {preemptions}");
}

/// Heavy preemption over a host tier with a copy lag: preempted requests resume from pages the
/// device dropped, loading them back while other requests step.
#[test]
fn heavy_preemption_through_the_host_tier() {
    let reqs = heavy_reqs();
    let want = solo(&reqs, 256);
    let (mut preemptions, mut loaded) = (0, 0);
    for (cache, lag) in [(96, 0), (96, 2), (120, 1), (144, 3)] {
        let setup = Setup { step_tokens: 8, max_seqs: 8, conservativeness: 0.05, host: 1024, lag, ..Setup::new(cache) };
        let h = setup.spawn();
        let got = run_all(&h, &reqs);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!((&g.tokens, &g.reason), (&w.tokens, &w.reason), "cache {cache}, lag {lag}: request {i}");
        }
        let st = settle(&h, &setup);
        eprintln!(
            "heavy through the host, cache {cache}, lag {lag}: {} preemptions, {} recomputed, {} loaded",
            st.preemptions, st.recomputed_tokens, st.host_loaded_tokens
        );
        preemptions += st.preemptions;
        loaded += st.host_loaded_tokens;
    }
    assert!(preemptions >= 5 && loaded > 0, "preemptions {preemptions}, loaded {loaded}");
}

/// A host tier no larger than the device cache is refused at load.
#[test]
fn a_host_tier_must_exceed_the_device_cache() {
    let opts = LoadOptions {
        max_seqs: 2,
        cache_tokens: 256,
        page: PAGE,
        backend: BackendKind::CPU,
        kv_dtype: CacheDtype::F32,
        threads: 1,
        host_cache_tokens: 256,
        ..Default::default()
    };
    let err = models::load(&fixture(), &opts).err().expect("refused");
    assert!(format!("{err:#}").contains("must exceed"), "{err:#}");
}
