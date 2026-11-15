// SPDX-License-Identifier: Apache-2.0
//! The serving layer against the tiny Kimi K3 and its torch oracle.
//!
//!   1. concurrent greedy requests with a small step budget, so prefills are chunked and
//!      interleave with other sequences' decodes, still reproduce the oracle exactly
//!   2. a seeded sampling request is reproducible under different concurrent load
//!   3. a short prompt submitted behind a long one starts first with PrefillOrder::ShortFirst, after
//!      the long one's prefill with PrefillOrder::Arrival, and both replies are the ones each gets
//!      alone; under a flood of short prompts the long one still gets a quarter of every step
//!   4. HTTP: /v1/completions (JSON and SSE), /v1/chat/completions, /v1/models, a 400, runtime
//!      LoRA loading refused unless allowed

use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use fullm::models::{self, BackendKind, LoadOptions};
use fullm::runtime::{
    Admission, Engine, EngineConfig, EngineHandle, Event, FinishReason, PrefillOrder, Request, SamplingParams,
};
use fullm::server::{router, AppState};
use fullm::tensor::Precision;
use fullm::tokenizer::ByteTokenizer;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny_k3")
}

fn oracle() -> Vec<(Vec<u32>, Vec<u32>)> {
    let raw = std::fs::read_to_string(fixture().join("oracle.json")).expect("oracle.json");
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let ids =
        |x: &serde_json::Value| x.as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect::<Vec<_>>();
    v["cases"].as_array().unwrap().iter().map(|c| (ids(&c["prompt"]), ids(&c["greedy"]))).collect()
}

fn engine(step_tokens: usize) -> EngineHandle {
    let page = 4;
    let model = models::load(
        &fixture(),
        &LoadOptions { trunk: Precision::Bf16, max_seqs: 4, cache_tokens: 256, page, ..Default::default() },
    )
    .unwrap();
    Engine::new(model, EngineConfig { step_tokens, ..Default::default() }).spawn().unwrap()
}

fn run(h: &EngineHandle, prompt: Vec<u32>, params: SamplingParams) -> (Vec<u32>, FinishReason) {
    let mut rx = h.submit(Request { prompt, params }).unwrap();
    let mut toks = Vec::new();
    loop {
        match rx.blocking_recv().expect("engine alive") {
            Event::Logprobs(_) => {}
            Event::Token(t) => toks.push(t),
            Event::Finished { reason, .. } => return (toks, reason),
        }
    }
}

fn greedy(n: usize) -> SamplingParams {
    SamplingParams { max_tokens: n, ..Default::default() }
}

#[test]
fn concurrent_chunked_requests_match_oracle() {
    let cases = oracle();
    let h = engine(5);
    // Submit everything first so the engine sees them together; the third repeats the
    // first prompt and so shares nothing with it but the weights.
    let mut jobs: Vec<_> = cases
        .iter()
        .map(|(p, g)| (h.submit(Request { prompt: p.clone(), params: greedy(g.len()) }).unwrap(), g.clone()))
        .collect();
    jobs.push((
        h.submit(Request { prompt: cases[0].0.clone(), params: greedy(cases[0].1.len()) }).unwrap(),
        cases[0].1.clone(),
    ));
    for (i, (mut rx, want)) in jobs.into_iter().enumerate() {
        let mut got = Vec::new();
        let reason = loop {
            match rx.blocking_recv().unwrap() {
                Event::Logprobs(_) => {}
                Event::Token(t) => got.push(t),
                Event::Finished { reason, usage } => {
                    assert_eq!(usage.completion_tokens, want.len());
                    break reason;
                }
            }
        };
        assert_eq!(reason, FinishReason::Length);
        assert_eq!(got, want, "request {i} diverged from the oracle");
    }
    let s = h.stats();
    assert!(
        s.steps > 0
            && s.decode_tokens > 0
            && s.prefill_tokens as usize >= cases.iter().map(|c| c.0.len()).sum::<usize>()
    );
}

#[test]
fn short_prompts_go_ahead_of_long_ones() {
    // a request's tokens and the time of its first one
    fn collect(mut rx: tokio::sync::mpsc::UnboundedReceiver<Event>) -> (Instant, Vec<u32>) {
        let (mut first, mut toks) = (None, Vec::new());
        loop {
            match rx.blocking_recv().expect("engine alive") {
                Event::Logprobs(_) => {}
                Event::Token(t) => {
                    first.get_or_insert_with(Instant::now);
                    toks.push(t);
                }
                Event::Finished { reason, .. } => {
                    assert_eq!(reason, FinishReason::Length);
                    return (first.expect("a token before the end"), toks);
                }
            }
        }
    }
    let start = |order| {
        let model = models::load(
            &fixture(),
            &LoadOptions { trunk: Precision::Bf16, max_seqs: 4, cache_tokens: 512, page: 4, ..Default::default() },
        )
        .unwrap();
        Engine::new(model, EngineConfig { step_tokens: 8, prefill_order: order, ..Default::default() }).spawn().unwrap()
    };
    // 30 steps of prefill, then a prompt that fits in one
    let long: Vec<u32> = (0..240).map(|i| 1 + i % 200).collect();
    let short = vec![5, 6, 7, 8];
    let solo = start(PrefillOrder::Arrival);
    let want_long = run(&solo, long.clone(), greedy(8)).0;
    let want_short = run(&solo, short.clone(), greedy(8)).0;
    for order in [PrefillOrder::ShortFirst, PrefillOrder::Arrival] {
        let h = start(order);
        let rl = h.submit(Request { prompt: long.clone(), params: greedy(8) }).unwrap();
        let rs = h.submit(Request { prompt: short.clone(), params: greedy(8) }).unwrap();
        let rl = std::thread::spawn(move || collect(rl));
        let (t_short, got_short) = collect(rs);
        let (t_long, got_long) = rl.join().unwrap();
        assert_eq!(t_short < t_long, order == PrefillOrder::ShortFirst, "{order:?}");
        // the order changes where the long prompt's chunks end, never what either request generates
        assert_eq!(got_long, want_long, "{order:?}: the long reply");
        assert_eq!(got_short, want_short, "{order:?}: the short reply");
    }
}

#[test]
fn a_long_prompt_is_not_starved_by_short_ones() {
    let model = models::load(
        &fixture(),
        &LoadOptions { trunk: Precision::Bf16, max_seqs: 8, cache_tokens: 1024, page: 4, ..Default::default() },
    )
    .unwrap();
    let h = Engine::new(model, EngineConfig { step_tokens: 16, ..Default::default() }).spawn().unwrap();
    let mut long = h.submit(Request { prompt: (0..240).map(|i| 1 + i % 200).collect(), params: greedy(1) }).unwrap();
    // 600 prompts of 4 tokens: 7 at a time beside the long one want 28 tokens a step, and four of
    // them would fill a step exactly; they may take three quarters of it
    let shorts: Vec<_> = (0..600u32)
        .map(|i| h.submit(Request { prompt: vec![9, 1 + i % 200, 3, 4], params: greedy(1) }).unwrap())
        .collect();
    while let Event::Token(_) = long.blocking_recv().expect("engine alive") {}
    let steps = h.stats().steps;
    // at least 4 of every step's 16 tokens: 60 steps for its 240 (the 600 short ones alone take 150)
    assert!(steps <= 240 / 4, "the long prompt took {steps} steps");
    drop(shorts);
}

#[test]
fn seeded_sampling_is_reproducible_under_load() {
    let cases = oracle();
    let params =
        SamplingParams { max_tokens: 12, temperature: 0.9, top_p: 0.95, top_k: 40, seed: 7, ..Default::default() };
    let h = engine(4);
    let (alone, _) = run(&h, cases[0].0.clone(), params.clone());
    // the same request again, now sharing every step with a greedy neighbour
    let mut neighbour = h.submit(Request { prompt: cases[1].0.clone(), params: greedy(12) }).unwrap();
    let (busy, _) = run(&h, cases[0].0.clone(), params.clone());
    while let Some(ev) = neighbour.blocking_recv() {
        if matches!(ev, Event::Finished { .. }) {
            break;
        }
    }
    assert_eq!(alone, busy);
    assert_eq!(alone.len(), 12);
    let other = SamplingParams { seed: 8, ..params };
    let (diff, _) = run(&h, cases[0].0.clone(), other);
    assert_ne!(alone, diff, "a different seed should change a 12-token sample");
}

#[test]
fn stop_ids_and_validation() {
    let cases = oracle();
    let h = engine(512);
    let want = &cases[0].1;
    let p = SamplingParams { max_tokens: 16, stop_token_ids: vec![want[3]], ..Default::default() };
    let (got, reason) = run(&h, cases[0].0.clone(), p);
    assert_eq!(reason, FinishReason::Stop);
    assert_eq!(got, want[..3].to_vec(), "the stop id is not emitted");
    assert!(h.submit(Request { prompt: vec![], params: greedy(1) }).is_err());
    assert!(h.submit(Request { prompt: vec![999], params: greedy(1) }).is_err());
    assert!(h.submit(Request { prompt: vec![1; 300], params: greedy(1) }).is_err(), "longer than the cache");
}

/// Preemption on a model whose sequences carry recurrent state and so have no prefix cache: in a
/// cache too small for what four requests generate, optimistic admission (forced) runs short, the
/// newest request is preempted, and on readmission its KDA state restarts from zero and its whole
/// context is computed again. Every request still reproduces the oracle, on the reference model
/// and on the executor.
#[test]
fn preempted_requests_restart_their_recurrent_state() {
    let cases = oracle();
    for backend in [BackendKind::Reference, BackendKind::CPU] {
        let model = models::load(
            &fixture(),
            &LoadOptions {
                trunk: Precision::Bf16,
                max_seqs: 4,
                cache_tokens: 48,
                page: 4,
                backend: backend.clone(),
                ..Default::default()
            },
        )
        .unwrap();
        let h =
            Engine::new(model, EngineConfig { step_tokens: 5, admission: Admission::Optimistic, ..Default::default() })
                .spawn()
                .unwrap();
        let jobs: Vec<_> = (0..4)
            .map(|i| {
                let (prompt, want) = &cases[i % 2];
                let params =
                    SamplingParams { max_tokens: want.len(), stop_token_ids: vec![u32::MAX - 1], ..Default::default() };
                (h.submit(Request { prompt: prompt.clone(), params }).unwrap(), want.clone())
            })
            .collect();
        for (i, (mut rx, want)) in jobs.into_iter().enumerate() {
            let mut got = Vec::new();
            loop {
                match rx.blocking_recv().unwrap() {
                    Event::Logprobs(_) => {}
                    Event::Token(t) => got.push(t),
                    Event::Finished { reason, .. } => {
                        assert_eq!(reason, FinishReason::Length, "{backend:?} request {i}");
                        break;
                    }
                }
            }
            assert_eq!(got, want, "{backend:?} request {i} diverged from the oracle");
        }
        let s = h.stats();
        eprintln!("{backend:?}: {} preemptions, {} positions recomputed", s.preemptions, s.recomputed_tokens);
        assert!(s.preemptions > 0, "{backend:?}: the cache never ran short");
        assert_eq!(s.cached_tokens, 0, "no prefix cache for a recurrent model");
    }
}

/// Minimal blocking HTTP/1.1 client: the response's status line and raw body.
fn http(addr: std::net::SocketAddr, method: &str, path: &str, body: &str) -> (u16, String) {
    http_with(addr, method, path, body, "")
}

/// `http` with extra header lines (each ending in \r\n).
fn http_with(addr: std::net::SocketAddr, method: &str, path: &str, body: &str, headers: &str) -> (u16, String) {
    let mut s = std::net::TcpStream::connect(addr).unwrap();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n{headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut raw = String::new();
    s.read_to_string(&mut raw).unwrap();
    let status = raw[9..12].parse().unwrap();
    let body = raw.split_once("\r\n\r\n").map(|(_, b)| b.to_string()).unwrap_or_default();
    (status, body)
}

#[test]
fn http_api_smoke() {
    let cases = oracle();
    let h = engine(512);
    let state = Arc::new(AppState {
        engine: h.clone(),
        tok: Arc::new(ByteTokenizer),
        model_name: "tiny-k3".into(),
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

    let prompt = serde_json::to_string(&cases[0].0).unwrap();
    let (st, body) =
        http(addr, "POST", "/v1/completions", &format!(r#"{{"prompt": {prompt}, "max_tokens": 4, "temperature": 0}}"#));
    assert_eq!(st, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "text_completion");
    let ids: Vec<u32> =
        v["choices"][0]["token_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    assert_eq!(ids, cases[0].1[..4].to_vec());
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert_eq!(v["usage"]["completion_tokens"], 4);
    assert_eq!(v["usage"]["prompt_tokens"], cases[0].0.len());

    // a body past axum's 2 MB default reaches the handler, which refuses the prompt for its length
    // (400): the body limit follows the context window, so long prompts are not cut off at HTTP (413)
    let big = "a".repeat(3 << 20);
    let (st, body) = http(addr, "POST", "/v1/completions", &format!(r#"{{"prompt": "{big}", "max_tokens": 1}}"#));
    assert_eq!(st, 400, "{}", &body[..body.len().min(300)]);

    // a stop id past the vocabulary is refused (with min_tokens it would index past the logits);
    // max_tokens far past the context is served, the context ending the reply
    let (st, body) =
        http(addr, "POST", "/v1/completions", r#"{"prompt": "hi", "min_tokens": 1, "stop_token_ids": [4000000000]}"#);
    assert_eq!(st, 400, "{body}");
    let (st, body) = http(
        addr,
        "POST",
        "/v1/completions",
        r#"{"prompt": "hi", "max_tokens": 18446744073709551615, "min_tokens": 2}"#,
    );
    assert_eq!(st, 200, "{body}");

    // serve --api-key: 401 without the key or with a wrong one, served with it; /health open
    let keyed = Arc::new(AppState {
        engine: h,
        tok: Arc::new(ByteTokenizer),
        model_name: "tiny-k3".into(),
        special_tokens_in_prompts: false,
        sampling_defaults: Default::default(),
        grammars: Default::default(),
        loras: Default::default(),
        runtime_lora: false,
        api_key: Some("sk-test".into()),
    });
    let kaddr = rt.block_on(async {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(l, router(keyed)).await.unwrap() });
        addr
    });
    let req = r#"{"prompt": "hi", "max_tokens": 2}"#;
    assert_eq!(http(kaddr, "POST", "/v1/completions", req).0, 401);
    assert_eq!(http_with(kaddr, "POST", "/v1/completions", req, "Authorization: Bearer sk-wrong\r\n").0, 401);
    assert_eq!(http_with(kaddr, "POST", "/v1/completions", req, "Authorization: Bearer sk-test\r\n").0, 200);
    assert_eq!(http(kaddr, "GET", "/v1/models", "").0, 401);
    assert_eq!(http(kaddr, "GET", "/health", "").0, 200);

    let (st, body) = http(
        addr,
        "POST",
        "/v1/completions",
        &format!(r#"{{"prompt": {prompt}, "max_tokens": 4, "temperature": 0, "stream": true}}"#),
    );
    assert_eq!(st, 200);
    let streamed: Vec<u32> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .flat_map(|d| {
            let v: serde_json::Value = serde_json::from_str(d).unwrap();
            v["choices"][0]["token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(streamed, cases[0].1[..4].to_vec());
    assert!(body.contains("data: [DONE]"), "{body}");

    let (st, body) = http(
        addr,
        "POST",
        "/v1/chat/completions",
        r#"{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3, "temperature": 0, "stream": true}"#,
    );
    assert_eq!(st, 200);
    assert!(
        body.contains("chat.completion.chunk")
            && body.contains("\"finish_reason\":\"length\"")
            && body.contains("data: [DONE]"),
        "{body}"
    );

    // stop strings: the text ends before the first one, and the request stops there (cancelled,
    // not run to max_tokens), whole or streamed; vLLM's include_stop_str_in_output keeps it
    let ask = |extra: &str| {
        let (st, body) = http(
            addr,
            "POST",
            "/v1/completions",
            &format!(r#"{{"prompt": {prompt}, "max_tokens": 64, "temperature": 0{extra}}}"#),
        );
        assert_eq!(st, 200, "{body}");
        body
    };
    let whole = |body: &str| serde_json::from_str::<serde_json::Value>(body).unwrap()["choices"][0].clone();
    let full = whole(&ask(""))["text"].as_str().unwrap().to_string();
    let chars: Vec<char> = full.chars().collect();
    assert!(chars.len() >= 8, "{full:?}");
    let stop: String = chars[2..4].iter().collect();
    let want = &full[..full.find(&stop).unwrap()];
    let stop_json = serde_json::to_string(&stop).unwrap();
    let c = whole(&ask(&format!(r#", "stop": [{stop_json}, "never generated"]"#)));
    assert_eq!(c["text"].as_str().unwrap(), want);
    assert_eq!((c["finish_reason"].as_str(), c["stop_reason"].as_str()), (Some("stop"), Some(stop.as_str())));
    assert!(c["token_ids"].as_array().unwrap().len() < 64);
    let c = whole(&ask(&format!(r#", "stop": {stop_json}, "include_stop_str_in_output": true"#)));
    assert_eq!(c["text"].as_str().unwrap(), format!("{want}{stop}"));
    let body = ask(&format!(r#", "stop": {stop_json}, "stream": true"#));
    let chunks: Vec<serde_json::Value> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();
    let text: String = chunks.iter().filter_map(|v| v["choices"][0]["text"].as_str()).collect();
    assert_eq!(text, want);
    let last = &chunks.last().unwrap()["choices"][0];
    assert_eq!((last["finish_reason"].as_str(), last["stop_reason"].as_str()), (Some("stop"), Some(stop.as_str())));
    assert!(chunks.last().unwrap()["usage"]["completion_tokens"].as_u64().unwrap() < 64);
    for bad in [r#""stop": [""]"#, r#""stop": 5"#] {
        let (st, body) = http(addr, "POST", "/v1/completions", &format!(r#"{{"prompt": [1, 2], {bad}}}"#));
        assert_eq!(st, 400, "{bad}: {body}");
    }

    let (st, body) = http(addr, "GET", "/v1/models", "");
    assert_eq!(st, 200);
    assert!(body.contains("tiny-k3"));

    let (st, body) = http(addr, "POST", "/v1/completions", r#"{"prompt": [1, 2], "top_p": 0}"#);
    assert_eq!(st, 400, "{body}");
    assert!(body.contains("invalid_request_error"));
    let (st, _) = http(addr, "POST", "/v1/completions", "not json");
    assert_eq!(st, 400);
    // runtime LoRA loading: off unless the server allows it (a client would pick the path)
    for path in ["/v1/load_lora_adapter", "/v1/unload_lora_adapter"] {
        let (st, body) = http(addr, "POST", path, r#"{"lora_name": "x", "lora_path": "/etc"}"#);
        assert_eq!(st, 403, "{path}: {body}");
        assert!(body.contains("--allow-runtime-lora"), "{body}");
    }
}
