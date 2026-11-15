// SPDX-License-Identifier: Apache-2.0
//! EAGLE-3 draft heads on the tiny Qwen3-MoE against an independent torch oracle
//! (tools/make_tiny_eagle3.py, following vLLM's Eagle3LlamaForCausalLM): the drafts after a whole
//! prompt, after a prompt fed in chunks (with and without drafting between them), for several
//! sequences in one call with different draft counts, and after a verify step that accepted
//! part of its drafts. Every oracle argmax wins by a margin, so the drafts must match exactly.
//! Then decoding step by step against the same context fed at once, and the engine: the head's
//! drafts change no token (greedy or seeded), with the prefix cache and through preemptions.

use std::path::PathBuf;

use fullm::models::{self, BackendKind, CacheDtype, DraftReq, LoadOptions, Model, SeqChunk};
use fullm::runtime::{Engine, EngineConfig, EngineHandle, Event, Request, SamplingParams, Stats};
use fullm::tensor::Precision;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

struct Case {
    prompt: Vec<u32>,
    next: u32,
    drafts: Vec<u32>,
}

struct Verify {
    prompt: Vec<u32>,
    step: Vec<u32>,
    accepted: usize,
    bonus: u32,
    drafts: Vec<u32>,
}

fn oracle() -> (Vec<Case>, Verify) {
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixtures().join("tiny_eagle3/oracle.json")).unwrap()).unwrap();
    let ids = |x: &serde_json::Value| x.as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect();
    let cases = v["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| Case {
            prompt: ids(&c["prompt"]),
            next: c["next"].as_u64().unwrap() as u32,
            drafts: ids(&c["drafts"]),
        })
        .collect();
    let w = &v["verify"];
    let verify = Verify {
        prompt: ids(&w["prompt"]),
        step: ids(&w["step"]),
        accepted: w["accepted"].as_u64().unwrap() as usize,
        bonus: w["bonus"].as_u64().unwrap() as u32,
        drafts: ids(&w["drafts"]),
    };
    (cases, verify)
}

fn backends() -> Vec<BackendKind> {
    let mut b = vec![BackendKind::CPU];
    if fullm_kernels::exec::cuda_available() {
        b.push(BackendKind::CUDA);
    }
    b
}

fn load(backend: BackendKind) -> Box<dyn Model> {
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
        spec_model: Some(fixtures().join("tiny_eagle3")),
        ..Default::default()
    };
    let m = models::load(&fixtures().join("tiny_qwen3_moe"), &opts).expect("load tiny qwen3_moe + EAGLE-3 head");
    assert!(m.draft_head());
    m
}

fn argmax(row: &[f32]) -> u32 {
    fullm::runtime::sampler::argmax(row)
}

/// Prefill `prompt` into a fresh sequence in chunks of `chunk`, feeding the head after each
/// (drafting `between` tokens after the inner chunks); then `k` drafts after `next`.
fn prefill_draft(
    m: &mut dyn Model,
    prompt: &[u32],
    next: u32,
    chunk: usize,
    between: usize,
    k: usize,
) -> (usize, Vec<u32>) {
    let s = m.alloc_seq().unwrap();
    let mut pos = 0;
    let mut out = Vec::new();
    for ch in prompt.chunks(chunk) {
        let logits = m.forward(&[SeqChunk { seq: s, pos, tokens: ch, all_logits: false, draft: 0 }]).unwrap();
        pos += ch.len();
        let last = pos == prompt.len();
        if last {
            assert_eq!(argmax(&logits[0]), next, "the target's token after the prompt");
        }
        let (nx, kk) = if last { (next, k) } else { (prompt[pos], between) };
        out = m.draft(&[DraftReq { seq: s, next: nx, k: kk }]).unwrap().remove(0);
        assert_eq!(out.len(), kk);
    }
    (s, out)
}

#[test]
fn drafts_match_the_oracle() {
    let (cases, verify) = oracle();
    for be in backends() {
        let mut m = load(be.clone());
        for c in &cases {
            for (chunk, between) in [(64, 0), (4, 0), (3, 2), (5, 5)] {
                let (s, d) = prefill_draft(m.as_mut(), &c.prompt, c.next, chunk, between, 5);
                assert_eq!(d, c.drafts, "{be}: prompt {:?} in chunks of {chunk}, {between} drafts between", c.prompt);
                m.free_seq(s);
            }
        }

        // the verify step: [next, d1..d4] at once, `accepted` drafts kept, then the bonus token
        let v = &verify;
        let (s, _) = prefill_draft(m.as_mut(), &v.prompt, v.step[0], 64, 0, 5);
        let n = v.step.len() - 1;
        m.forward(&[SeqChunk { seq: s, pos: v.prompt.len(), tokens: &v.step, all_logits: true, draft: n }]).unwrap();
        m.accept(s, v.accepted).unwrap();
        let d = m.draft(&[DraftReq { seq: s, next: v.bonus, k: 5 }]).unwrap().remove(0);
        assert_eq!(d, v.drafts, "{be}: after a verify step keeping {} of {n} drafts", v.accepted);
        m.free_seq(s);
    }
}

#[test]
fn several_sequences_in_one_call() {
    let (cases, _) = oracle();
    for be in backends() {
        let mut m = load(be.clone());
        let seqs: Vec<usize> = cases.iter().map(|_| m.alloc_seq().unwrap()).collect();
        let batch: Vec<SeqChunk<'_>> = cases
            .iter()
            .zip(&seqs)
            .map(|(c, &s)| SeqChunk { seq: s, pos: 0, tokens: &c.prompt, all_logits: false, draft: 0 })
            .collect();
        let logits = m.forward(&batch).unwrap();
        for (c, l) in cases.iter().zip(&logits) {
            assert_eq!(argmax(l), c.next);
        }
        // a draft count per sequence (one drafts none: the head only reads its positions)
        let ks = [5, 0, 2];
        let reqs: Vec<DraftReq> =
            cases.iter().zip(&seqs).zip(ks).map(|((c, &s), k)| DraftReq { seq: s, next: c.next, k }).collect();
        let out = m.draft(&reqs).unwrap();
        for ((c, d), k) in cases.iter().zip(&out).zip(ks) {
            assert_eq!(d[..], c.drafts[..k], "{be}: prompt {:?}, {k} drafts", c.prompt);
        }
        // the one that drafted none then drafts from the same positions (they are in its cache)
        let (c, s) = (&cases[1], seqs[1]);
        let d = m.draft(&[DraftReq { seq: s, next: c.next, k: 5 }]).unwrap().remove(0);
        assert_eq!(d, c.drafts, "{be}: a second draft call after the same step");
    }
}

#[test]
fn refuses_misuse() {
    let (cases, _) = oracle();
    let mut m = load(BackendKind::CPU);
    let c = &cases[0];
    let s = m.alloc_seq().unwrap();
    let t = m.alloc_seq().unwrap();
    m.forward(&[SeqChunk { seq: s, pos: 0, tokens: &c.prompt, all_logits: false, draft: 0 }]).unwrap();
    // a sequence that was not in the last step, a token outside the vocabulary
    assert!(m.draft(&[DraftReq { seq: t, next: 1, k: 2 }]).is_err());
    assert!(m.draft(&[DraftReq { seq: s, next: 1 << 20, k: 2 }]).is_err());
    // drafts not yet accepted
    let n = c.prompt.len();
    m.forward(&[SeqChunk { seq: s, pos: n, tokens: &[c.next, 7, 8], all_logits: true, draft: 2 }]).unwrap();
    assert!(m.draft(&[DraftReq { seq: s, next: 1, k: 2 }]).is_err());
    m.accept(s, 0).unwrap();
    assert_eq!(m.draft(&[DraftReq { seq: s, next: 1, k: 2 }]).unwrap()[0].len(), 2);
}

/// Decoding with drafts and partial accepts, the head fed after every step, drafts what the head
/// drafts after the same context fed whole into a fresh sequence: it read every position once,
/// with the right token.
#[test]
fn step_by_step_equals_all_at_once() {
    let (cases, _) = oracle();
    for be in backends() {
        let mut m = load(be.clone());
        let c = &cases[0];
        let (s, mut drafts) = prefill_draft(m.as_mut(), &c.prompt, c.next, 4, 3, 4);
        let mut ctx = c.prompt.clone();
        let mut next = c.next;
        let mut x = 7u64;
        for step in 0..10 {
            // [next, drafts...]: keep a pseudo-random count of them (accept takes any), then a token
            let mut toks = vec![next];
            toks.extend_from_slice(&drafts);
            let n = toks.len() - 1;
            m.forward(&[SeqChunk { seq: s, pos: ctx.len(), tokens: &toks, all_logits: true, draft: n }]).unwrap();
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let keep = (x >> 33) as usize % (n + 1);
            m.accept(s, keep).unwrap();
            ctx.extend_from_slice(&toks[..1 + keep]);
            next = ((x >> 40) % 256) as u32;
            let k = 1 + step % 4;
            drafts = m.draft(&[DraftReq { seq: s, next, k }]).unwrap().remove(0);

            let f = m.alloc_seq().unwrap();
            m.forward(&[SeqChunk { seq: f, pos: 0, tokens: &ctx, all_logits: false, draft: 0 }]).unwrap();
            let want = m.draft(&[DraftReq { seq: f, next, k }]).unwrap().remove(0);
            m.free_seq(f);
            assert_eq!(drafts, want, "{be}: step {step}, context {}", ctx.len());
        }
        m.free_seq(s);
    }
}

const NEVER: u32 = u32::MAX - 1;

fn engine(head: bool, spec_tokens: usize, prefix: bool, cache_tokens: usize) -> EngineHandle {
    engine_with(head, spec_tokens, prefix, cache_tokens, 1.0)
}

/// `conservativeness` well below 1 admits more than fits: preemptions.
fn engine_with(
    head: bool,
    spec_tokens: usize,
    prefix: bool,
    cache_tokens: usize,
    conservativeness: f64,
) -> EngineHandle {
    let opts = LoadOptions {
        trunk: Precision::Bf16,
        max_seqs: 6,
        cache_tokens,
        page: 4,
        backend: BackendKind::CPU,
        kv_dtype: CacheDtype::F32,
        step_tokens: 32,
        max_rows: 64,
        max_context: 512,
        threads: 2,
        no_prefix_cache: !prefix,
        spec_model: head.then(|| fixtures().join("tiny_eagle3")),
        ..Default::default()
    };
    let m = models::load(&fixtures().join("tiny_qwen3_moe"), &opts).expect("load tiny qwen3_moe");
    // n-grams of 5 to 6 tokens: the head drafts for most decodes
    let cfg = EngineConfig {
        step_tokens: 32,
        spec_tokens,
        spec_ngram: (5, 6),
        eagle_tokens: 3,
        conservativeness,
        ..Default::default()
    };
    Engine::new(m, cfg).spawn().unwrap()
}

type Req = (Vec<u32>, SamplingParams);

fn greedy(n: usize) -> SamplingParams {
    SamplingParams { max_tokens: n, stop_token_ids: vec![NEVER], ..Default::default() }
}

fn run_all(h: &EngineHandle, reqs: &[Req]) -> Vec<(Vec<u32>, usize)> {
    let mut rxs: Vec<_> =
        reqs.iter().map(|r| h.submit(Request { prompt: r.0.clone(), params: r.1.clone() }).unwrap()).collect();
    rxs.iter_mut()
        .map(|rx| {
            let mut toks = Vec::new();
            loop {
                match rx.blocking_recv().expect("engine alive") {
                    Event::Logprobs(_) => {}
                    Event::Token(t) => toks.push(t),
                    Event::Finished { usage, .. } => return (toks, usage.cached_tokens),
                }
            }
        })
        .collect()
}

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

/// Prompts of 9 to 69 tokens (the longer ones prefill over several 32-token steps).
fn prompts(n: usize, gen: usize) -> Vec<Req> {
    let mut x = 0x2545_f491_4f6c_dd1du64;
    (0..n)
        .map(|i| {
            let p = (0..9 + 12 * i)
                .map(|_| {
                    x ^= x >> 12;
                    x ^= x << 25;
                    x ^= x >> 27;
                    (x.wrapping_mul(0x9e37_79b9_7f4a_7c15) % 255) as u32 + 1
                })
                .collect();
            (p, greedy(gen))
        })
        .collect()
}

fn toks(outs: &[(Vec<u32>, usize)]) -> Vec<Vec<u32>> {
    outs.iter().map(|o| o.0.clone()).collect()
}

#[test]
fn engine_head_drafts_change_no_token() {
    let greedy_reqs = prompts(6, 40);
    let sampled: Vec<Req> = greedy_reqs
        .iter()
        .enumerate()
        .map(|(i, (p, _))| {
            let (temperature, top_k) = if i % 2 == 0 { (0.3, 0) } else { (0.8, 20) };
            (p.clone(), SamplingParams { temperature, top_p: 0.9, top_k, seed: 77 + i as u64, ..greedy(40) })
        })
        .collect();
    for reqs in [&greedy_reqs, &sampled] {
        for prefix in [false, true] {
            let want = run_all(&engine(false, 0, prefix, 4096), reqs);
            let h = engine(true, 4, prefix, 4096);
            let got = run_all(&h, reqs);
            assert_eq!(toks(&got), toks(&want), "prefix cache {prefix}");
            let s = settled(&h);
            assert!(s.spec_drafted_eagle > 0, "the head drafted: {s:?}");
            assert!(s.spec_accepted_eagle <= s.spec_drafted_eagle && s.spec_accepted >= s.spec_accepted_eagle);
        }
    }
}

#[test]
fn engine_head_with_cache_hits_and_preemptions() {
    let reqs = prompts(5, 32);
    let want = run_all(&engine(false, 0, false, 4096), &reqs);
    // a cache too small for all of them at once: requests are preempted and resume from their pages
    let h = engine_with(true, 4, true, 192, 0.05);
    let got = run_all(&h, &reqs);
    assert_eq!(toks(&got), toks(&want));
    let s = settled(&h);
    assert!(s.preemptions > 0 && s.spec_drafted_eagle > 0, "{s:?}");
    // each continued with its reply: the reply's pages come from the prefix cache, the head's
    // cache with them
    let next: Vec<Req> =
        reqs.iter().zip(&got).map(|((p, _), o)| ([p.as_slice(), &o.0, &[7, 9]].concat(), greedy(16))).collect();
    let h = engine(true, 4, true, 4096);
    let first = run_all(&h, &reqs);
    assert_eq!(toks(&first), toks(&want));
    let cont = run_all(&h, &next);
    assert!(cont.iter().all(|o| o.1 > 0), "continuations hit the cache");
    assert_eq!(toks(&cont), toks(&run_all(&engine(false, 0, false, 4096), &next)));
}

/// The engine's model, watched: it tracks each sequence's context from the steps and accepts the
/// engine makes, and at every draft call checks the drafts against a second model that is fed
/// the same context whole (through the head chunk by chunk, as `prefill_draft` does) and the same
/// `next`; and that each `next` is the first token of that sequence's next chunk. The head's cache
/// then was fed every position, with the right token, whatever the steps were.
struct Spy {
    inner: Box<dyn Model>,
    shadow: Box<dyn Model>,
    ctx: Vec<Vec<u32>>,
    pending: Vec<Option<Vec<u32>>>,
    next: Vec<Option<u32>>,
    log: std::sync::Arc<std::sync::Mutex<(usize, Vec<String>)>>,
}

impl Spy {
    fn steps(&mut self, batch: &[SeqChunk<'_>]) {
        for ch in batch {
            let mut errs = Vec::new();
            if ch.pos != self.ctx[ch.seq].len() {
                errs.push(format!("seq {}: chunk at {}, context {}", ch.seq, ch.pos, self.ctx[ch.seq].len()));
            }
            if let Some(n) = self.next[ch.seq].take() {
                if n != ch.tokens[0] {
                    errs.push(format!(
                        "seq {}: the head was told {n} comes next, the chunk starts {}",
                        ch.seq, ch.tokens[0]
                    ));
                }
            }
            if ch.draft == 0 {
                self.ctx[ch.seq].extend_from_slice(ch.tokens);
                self.pending[ch.seq] = None;
            } else {
                self.pending[ch.seq] = Some(ch.tokens.to_vec());
            }
            self.log.lock().unwrap().1.extend(errs);
        }
    }

    fn check(&mut self, r: &DraftReq, got: &[u32]) {
        let ctx = self.ctx[r.seq].clone();
        let m = self.shadow.as_mut();
        let s = m.alloc_seq().unwrap();
        let mut want = Vec::new();
        for (i, ch) in ctx.chunks(32).enumerate() {
            m.forward(&[SeqChunk { seq: s, pos: 32 * i, tokens: ch, all_logits: false, draft: 0 }]).unwrap();
            let end = 32 * i + ch.len();
            let (nx, k) = if end == ctx.len() { (r.next, got.len()) } else { (ctx[end], 0) };
            want = m.draft(&[DraftReq { seq: s, next: nx, k }]).unwrap().remove(0);
        }
        m.free_seq(s);
        let mut log = self.log.lock().unwrap();
        log.0 += 1;
        if want != got {
            log.1.push(format!("seq {} at {}: drafts {got:?}, fed whole {want:?}", r.seq, ctx.len()));
        }
    }
}

impl Model for Spy {
    fn arch(&self) -> &'static str {
        self.inner.arch()
    }
    fn vocab(&self) -> usize {
        self.inner.vocab()
    }
    fn max_context(&self) -> usize {
        self.inner.max_context()
    }
    fn capacity(&self) -> models::Capacity {
        self.inner.capacity()
    }
    fn alloc_seq(&mut self) -> Option<usize> {
        let s = self.inner.alloc_seq()?;
        self.ctx[s].clear();
        Some(s)
    }
    fn alloc_seq_cached(&mut self, prompt: &[u32]) -> Option<(usize, usize)> {
        let (s, n) = self.inner.alloc_seq_cached(prompt)?;
        self.ctx[s] = prompt[..n].to_vec();
        Some((s, n))
    }
    fn has_prefix_cache(&self) -> bool {
        self.inner.has_prefix_cache()
    }
    fn host_cached(&self, seq: usize, prompt: &[u32]) -> usize {
        self.inner.host_cached(seq, prompt)
    }
    fn load_cached(&mut self, seq: usize, prompt: &[u32], n: usize) -> anyhow::Result<usize> {
        let added = self.inner.load_cached(seq, prompt, n)?;
        let have = self.ctx[seq].len();
        self.ctx[seq] = prompt[..have + added].to_vec();
        Ok(added)
    }
    fn seq_ready(&mut self, seq: usize) -> bool {
        self.inner.seq_ready(seq)
    }
    fn wait_copies(&mut self) -> anyhow::Result<()> {
        self.inner.wait_copies()
    }
    fn last_step(&self) -> Option<fullm_kernels::exec::StepStats> {
        self.inner.last_step()
    }
    fn flush_prefix_cache(&mut self) -> usize {
        self.inner.flush_prefix_cache()
    }
    fn free_seq(&mut self, seq: usize) {
        self.inner.free_seq(seq);
        self.ctx[seq].clear();
        self.next[seq] = None;
        self.pending[seq] = None;
    }
    fn can_grow(&self, seq: usize, n: usize) -> bool {
        self.inner.can_grow(seq, n)
    }
    fn page_tokens(&self) -> usize {
        self.inner.page_tokens()
    }
    fn forward(&mut self, batch: &[SeqChunk<'_>]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.steps(batch);
        self.inner.forward(batch)
    }
    fn forward_greedy(&mut self, batch: &[SeqChunk<'_>]) -> anyhow::Result<Vec<u32>> {
        self.steps(batch);
        self.inner.forward_greedy(batch)
    }
    fn forward_flat(&mut self, batch: &[SeqChunk<'_>], out: &mut Vec<f32>) -> anyhow::Result<()> {
        self.steps(batch);
        self.inner.forward_flat(batch, out)
    }
    fn device_sampling(&self) -> bool {
        self.inner.device_sampling()
    }
    fn forward_sample(
        &mut self,
        batch: &[SeqChunk<'_>],
        sample: &[models::SampleRow],
        io: &mut models::SampleIo<'_>,
    ) -> anyhow::Result<Vec<u32>> {
        self.steps(batch);
        self.inner.forward_sample(batch, sample, io)
    }
    fn speculation(&self) -> bool {
        self.inner.speculation()
    }
    fn accept(&mut self, seq: usize, n: usize) -> anyhow::Result<()> {
        let t = self.pending[seq].take().expect("accept after a chunk with drafts");
        self.ctx[seq].extend_from_slice(&t[..1 + n]);
        self.inner.accept(seq, n)
    }
    fn draft_head(&self) -> bool {
        self.inner.draft_head()
    }
    fn draft(&mut self, reqs: &[DraftReq]) -> anyhow::Result<Vec<Vec<u32>>> {
        let out = self.inner.draft(reqs)?;
        for (r, d) in reqs.iter().zip(&out) {
            self.next[r.seq] = Some(r.next);
            if !d.is_empty() {
                self.check(r, d);
            }
        }
        Ok(out)
    }
    fn describe(&self) -> String {
        self.inner.describe()
    }
}

type SpyLog = std::sync::Arc<std::sync::Mutex<(usize, Vec<String>)>>;

fn spy_opts() -> LoadOptions {
    LoadOptions {
        trunk: Precision::Bf16,
        max_seqs: 6,
        cache_tokens: 4096,
        page: 4,
        backend: BackendKind::CPU,
        kv_dtype: CacheDtype::F32,
        step_tokens: 32,
        max_rows: 64,
        max_context: 512,
        threads: 2,
        spec_model: Some(fixtures().join("tiny_eagle3")),
        ..Default::default()
    }
}

fn spy_cfg() -> EngineConfig {
    EngineConfig { step_tokens: 32, spec_tokens: 4, spec_ngram: (5, 6), eagle_tokens: 3, ..Default::default() }
}

/// An engine over a spied model loaded with `opts` (the shadow: the same model without a prefix
/// cache or host tier).
fn spied(opts: &LoadOptions, cfg: EngineConfig) -> (EngineHandle, SpyLog) {
    let target = fixtures().join("tiny_qwen3_moe");
    let shadow = LoadOptions {
        no_prefix_cache: true,
        host_cache_tokens: 0,
        cache_tokens: 4096,
        max_seqs: 2,
        max_context: 1024,
        step_tokens: 32,
        ..opts.clone()
    };
    let log = SpyLog::default();
    let n = opts.max_seqs;
    let spy = Spy {
        inner: models::load(&target, opts).unwrap(),
        shadow: models::load(&target, &shadow).unwrap(),
        ctx: vec![Vec::new(); n],
        pending: vec![None; n],
        next: vec![None; n],
        log: log.clone(),
    };
    (Engine::new(Box::new(spy), cfg).spawn().unwrap(), log)
}

fn assert_clean(log: &SpyLog, what: &str, min_checked: usize) {
    let (checked, errs) = &*log.lock().unwrap();
    assert!(errs.is_empty(), "{what}: {} of {checked} draft calls wrong:\n{}", errs.len(), errs.join("\n"));
    assert!(*checked >= min_checked, "{what}: {checked} draft calls checked");
}

/// The reference replies: an engine without speculation.
fn reference(opts: &LoadOptions, reqs: &[Req]) -> Vec<Vec<u32>> {
    let opts = LoadOptions { spec_model: None, ..opts.clone() };
    let h = Engine::new(
        models::load(&fixtures().join("tiny_qwen3_moe"), &opts).unwrap(),
        EngineConfig { step_tokens: 32, ..Default::default() },
    )
    .spawn()
    .unwrap();
    toks(&run_all(&h, reqs))
}

#[test]
fn engine_feeds_the_head_every_position() {
    let reqs = prompts(6, 40);
    let sampled: Vec<Req> = reqs
        .iter()
        .enumerate()
        .map(|(i, (p, _))| {
            (p.clone(), SamplingParams { temperature: 0.8, top_p: 0.9, seed: 5 + i as u64, ..greedy(40) })
        })
        .collect();
    // the last: the head drafts for every decode (no n-gram drafts)
    for (prefix, own) in [(false, true), (true, true), (true, false)] {
        let opts = LoadOptions { no_prefix_cache: !prefix, ..spy_opts() };
        let (h, log) = spied(&opts, EngineConfig { spec_own: own, ..spy_cfg() });
        let first = run_all(&h, &reqs);
        run_all(&h, &sampled);
        // continuations: their cached pages carry the head's cache
        let next: Vec<Req> =
            reqs.iter().zip(&first).map(|((p, _), o)| ([p.as_slice(), &o.0, &[7, 9]].concat(), greedy(16))).collect();
        run_all(&h, &next);
        let st = settled(&h);
        assert_clean(&log, &format!("prefix cache {prefix}, own drafts {own}"), 50);
        if !own {
            assert_eq!(st.spec_drafted, st.spec_drafted_eagle, "only the head drafts: {st:?}");
        }
    }
}

/// Pages the device dropped come back from the host tier with the head's cache in them (written
/// through after the head read them).
#[test]
fn engine_head_with_the_host_tier() {
    let opts = LoadOptions { cache_tokens: 256, host_cache_tokens: 8192, ..spy_opts() };
    let reqs = prompts(6, 24);
    let (h, log) = spied(&opts, spy_cfg());
    let first = run_all(&h, &reqs);
    // others push the first ones' pages off the device
    let others: Vec<Req> = prompts(12, 16).into_iter().skip(6).collect();
    run_all(&h, &others);
    let next: Vec<Req> =
        reqs.iter().zip(&first).map(|((p, _), o)| ([p.as_slice(), &o.0, &[3]].concat(), greedy(20))).collect();
    let got = run_all(&h, &next);
    let s = settled(&h);
    assert!(s.host_loaded_tokens > 0, "loads from the host tier: {s:?}");
    assert_eq!(toks(&got), reference(&opts, &next));
    assert_clean(&log, "host tier", 50);
}

/// Requests that run into the context: the head's drafts stop at its end.
#[test]
fn engine_head_at_the_context_limit() {
    let opts = LoadOptions { max_context: 96, cache_tokens: 2048, ..spy_opts() };
    let reqs: Vec<Req> = prompts(6, 64).into_iter().map(|(p, sp)| (p[..p.len().min(80)].to_vec(), sp)).collect();
    let (h, log) = spied(&opts, spy_cfg());
    let got = run_all(&h, &reqs);
    settled(&h);
    assert_eq!(toks(&got), reference(&opts, &reqs));
    assert!(got.iter().any(|o| o.0.len() < 64), "some requests reached the context limit");
    assert_clean(&log, "context limit", 30);
}

/// Many requests over few slots, greedy and sampled, sharing prefixes that end on and off the page
/// grid (pages are shared only with the token after them: the head read each with it), some
/// preempted for a cache too small for them all.
#[test]
fn engine_head_many_requests_shared_prefixes_and_preemption() {
    let base: Vec<u32> = (0..48u32).map(|i| (i * 53 + 17) % 251 + 1).collect();
    let mut reqs: Vec<Req> = Vec::new();
    for (i, cut) in [8usize, 9, 16, 17, 23, 24, 31, 32, 40, 47].into_iter().enumerate() {
        let mut p = base[..cut].to_vec();
        p.extend((0..5 + i as u32).map(|j| (j * 31 + i as u32 * 7) % 250 + 3));
        let sp = if i % 3 == 0 {
            SamplingParams { temperature: 0.7, top_p: 0.9, top_k: 20, seed: 100 + i as u64, ..greedy(30) }
        } else {
            greedy(30)
        };
        reqs.push((p, sp));
    }
    reqs.extend(prompts(6, 30));
    for (cache, conservativeness) in [(4096, 1.0), (256, 0.05)] {
        let opts = LoadOptions { cache_tokens: cache, ..spy_opts() };
        let (h, log) = spied(&opts, EngineConfig { conservativeness, ..spy_cfg() });
        let got = run_all(&h, &reqs);
        let s = settled(&h);
        assert_eq!(toks(&got), reference(&opts, &reqs), "cache {cache}");
        assert!(s.cached_tokens > 0, "shared prefixes hit: {s:?}");
        if cache < 1000 {
            assert!(s.preemptions > 0, "{s:?}");
        }
        assert_clean(&log, &format!("cache {cache}"), 100);
    }
}

/// How a scenario file (tools/make_eagle3_scenarios.py) ran against torch's drafts.
#[derive(Debug, Default)]
struct Tally {
    requests: usize,
    /// Draft positions compared (a chain stops at its first difference).
    drafts: usize,
    chains_exact: usize,
    /// First differences where torch's top-2 margin was under the tolerance (a near-tie).
    near_ties: usize,
    wrong: Vec<String>,
    /// Prompts that found pages in the prefix cache, and the positions they found.
    hits: usize,
    cached: usize,
}

/// `sfx`: "" for torch's drafts with an f32 cache, "_bf16" for a bf16 one.
fn compare(t: &mut Tally, at: &str, got: &[u32], r: &serde_json::Value, tol: f32, sfx: &str) {
    let ids = |x: &serde_json::Value| -> Vec<u32> {
        x.as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect()
    };
    let (want, k) = (ids(&r[format!("expect{sfx}")]), r["k"].as_u64().unwrap() as usize);
    let margins: Vec<f32> =
        r[format!("margins{sfx}")].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    t.requests += 1;
    if got.len() != k {
        t.wrong.push(format!("{at}: {} drafts of {k}", got.len()));
        return;
    }
    for i in 0..k {
        t.drafts += 1;
        if got[i] != want[i] {
            if margins[i] < tol {
                t.near_ties += 1;
            } else {
                t.wrong.push(format!("{at}: drafts {got:?}, torch {want:?} (margins {margins:?})"));
            }
            return;
        }
    }
    t.chains_exact += 1;
}

/// Replay every scenario of `path` on models from `load` (a fresh one per scenario when `fresh`,
/// else one with its prefix cache flushed between scenarios), comparing each draft call with
/// torch's: a difference counts as a near-tie where torch's margin is under `tol`.
fn run_scenarios(load: &dyn Fn() -> Box<dyn Model>, path: &std::path::Path, tol: f32, fresh: bool, sfx: &str) -> Tally {
    use std::collections::HashMap;
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let ids = |x: &serde_json::Value| -> Vec<u32> {
        x.as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect()
    };
    let step = v["step_tokens"].as_u64().unwrap() as usize;
    let mut t = Tally::default();
    let mut m = load();
    for (si, sc) in v["scenarios"].as_array().unwrap().iter().enumerate() {
        if fresh && si > 0 {
            m = load();
        } else {
            m.flush_prefix_cache();
        }
        let mut seq: HashMap<String, usize> = HashMap::new();
        let mut len: HashMap<String, usize> = HashMap::new();
        for (oi, op) in sc["ops"].as_array().unwrap().iter().enumerate() {
            let name = |x: &serde_json::Value| x.as_str().unwrap().to_string();
            let at = format!("scenario {si} op {oi}");
            match op["op"].as_str().unwrap() {
                "alloc" => {
                    seq.insert(name(&op["seq"]), m.alloc_seq().expect("a free slot"));
                    len.insert(name(&op["seq"]), 0);
                }
                "alloc_prefill" => {
                    let p = ids(&op["prompt"]);
                    let (s, c) = m.alloc_seq_cached(&p).expect("a free slot");
                    if c > 0 {
                        t.hits += 1;
                        t.cached += c;
                    }
                    // the rest in chunks of `step` tokens, the head reading each
                    let mut pos = c;
                    while pos < p.len() {
                        let end = (pos + step).min(p.len());
                        m.forward_greedy(&[SeqChunk {
                            seq: s,
                            pos,
                            tokens: &p[pos..end],
                            all_logits: false,
                            draft: 0,
                        }])
                        .unwrap();
                        if end < p.len() {
                            m.draft(&[DraftReq { seq: s, next: p[end], k: 0 }]).unwrap();
                        }
                        pos = end;
                    }
                    seq.insert(name(&op["seq"]), s);
                    len.insert(name(&op["seq"]), p.len());
                    let k = op["k"].as_u64().unwrap() as usize;
                    let next = op["next"].as_u64().unwrap() as u32;
                    let got = m.draft(&[DraftReq { seq: s, next, k }]).unwrap().remove(0);
                    compare(&mut t, &at, &got, op, tol, sfx);
                }
                "step" => {
                    let chunks: Vec<(String, Vec<u32>, usize)> = op["chunks"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|c| (name(&c["seq"]), ids(&c["tokens"]), c["draft"].as_u64().unwrap() as usize))
                        .collect();
                    let batch: Vec<SeqChunk<'_>> = chunks
                        .iter()
                        .map(|(n, toks, d)| SeqChunk {
                            seq: seq[n],
                            pos: len[n],
                            tokens: toks,
                            all_logits: *d > 0,
                            draft: *d,
                        })
                        .collect();
                    m.forward_greedy(&batch).unwrap_or_else(|e| panic!("{at}: {e:#}"));
                    for (n, toks, d) in &chunks {
                        *len.get_mut(n).unwrap() += toks.len() - d;
                    }
                }
                "accept" => {
                    let n = op["n"].as_u64().unwrap() as usize;
                    m.accept(seq[&name(&op["seq"])], n).unwrap();
                    *len.get_mut(&name(&op["seq"])).unwrap() += n;
                }
                "draft" => {
                    let rs = op["reqs"].as_array().unwrap();
                    let reqs: Vec<DraftReq> = rs
                        .iter()
                        .map(|r| DraftReq {
                            seq: seq[&name(&r["seq"])],
                            next: r["next"].as_u64().unwrap() as u32,
                            k: r["k"].as_u64().unwrap() as usize,
                        })
                        .collect();
                    let out = m.draft(&reqs).unwrap_or_else(|e| panic!("{at}: {e:#}"));
                    for (r, got) in rs.iter().zip(&out) {
                        compare(&mut t, &format!("{at} {}", name(&r["seq"])), got, r, tol, sfx);
                    }
                }
                "free" => {
                    m.free_seq(seq.remove(&name(&op["seq"])).unwrap());
                    len.remove(&name(&op["seq"]));
                }
                other => panic!("{at}: unknown op {other}"),
            }
        }
        for (_, s) in seq.drain() {
            m.free_seq(s);
        }
    }
    t
}

/// A scenario file's tally: torch's drafts exactly (but at near-ties), except where the CUDA
/// backend ran steps of more than 16 rows: cuBLAS then takes bf16 activations (as vLLM computes),
/// which moves logits by more than near-ties; there at least 95% of chains still match.
fn check_tally(t: &Tally, be: &BackendKind, path: &std::path::Path, what: &str) {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let wide_steps = v["step_tokens"].as_u64().unwrap() > 16;
    if *be == BackendKind::CUDA && wide_steps {
        assert!(t.chains_exact * 100 >= 95 * t.requests, "{what}: {t:?}");
    } else {
        assert!(t.wrong.is_empty(), "{what}: {} wrong:\n{}", t.wrong.len(), t.wrong.join("\n"));
        assert!(t.near_ties * 100 <= t.requests, "{what}: {} near-ties in {} requests", t.near_ties, t.requests);
    }
}

fn tiny_load(be: BackendKind, page: usize, kv: CacheDtype, prefix: bool) -> impl Fn() -> Box<dyn Model> {
    tiny_load_head(be, page, kv, prefix, "tiny_eagle3")
}

fn tiny_load_head(
    be: BackendKind,
    page: usize,
    kv: CacheDtype,
    prefix: bool,
    head: &'static str,
) -> impl Fn() -> Box<dyn Model> {
    move || {
        let opts = LoadOptions {
            trunk: Precision::Bf16,
            max_seqs: 12,
            cache_tokens: 8192,
            page,
            backend: be.clone(),
            kv_dtype: kv,
            step_tokens: 64,
            max_rows: 64,
            max_context: 1024,
            threads: 4,
            no_prefix_cache: !prefix,
            spec_model: Some(fixtures().join(head)),
            ..Default::default()
        };
        models::load(&fixtures().join("tiny_qwen3_moe"), &opts).unwrap()
    }
}

/// Random scenarios (every op the engine makes, in random orders and sizes, prefix-cache hits
/// included; up to 6 sequences alive, up to 10 in the wide set, steps of at most 16 tokens in
/// the last) against torch, with an f32 cache: every draft as torch's but at near-ties
/// (`check_tally`).
#[test]
fn scenarios_match_torch() {
    for (file, page, prefix) in [
        ("scenarios.json", 4, true),
        ("scenarios.json", 16, true),
        ("scenarios.json", 4, false),
        ("scenarios_wide.json", 4, true),
        ("scenarios_wide.json", 16, false),
        ("scenarios_16.json", 4, true),
    ] {
        let path = fixtures().join("tiny_eagle3").join(file);
        for be in backends() {
            let t = run_scenarios(&tiny_load(be.clone(), page, CacheDtype::F32, prefix), &path, 1e-3, true, "");
            eprintln!("{be} {file} page {page} prefix {prefix}: {t:?}");
            check_tally(&t, &be, &path, &format!("{be} {file} page {page}"));
            assert!(t.requests > 1000 && (!prefix || t.hits > 100 / page), "{t:?}");
        }
    }
}

/// The same with a bf16 cache (the default), against torch rounding every layer's keys and
/// values to bf16 as the cache stores them.
#[test]
fn scenarios_with_a_bf16_cache() {
    for (file, be) in ["scenarios.json", "scenarios_wide.json", "scenarios_16.json"]
        .into_iter()
        .flat_map(|f| backends().into_iter().map(move |b| (f, b)))
    {
        let path = fixtures().join("tiny_eagle3").join(file);
        // (0.002: torch's bf16 rounding of keys computed in another order moves near-ties that far)
        let t = run_scenarios(&tiny_load(be.clone(), 4, CacheDtype::Bf16, true), &path, 2e-3, true, "_bf16");
        eprintln!("{be} {file} bf16 cache: {t:?}");
        check_tally(&t, &be, &path, &format!("{be} {file} bf16 cache"));
    }
}

/// A head over the whole vocabulary (the target's lm_head, its own ignored as vLLM does; no
/// d2t), fp16 tensors named "layers.0.", a transformers-5 config with the default taps.
#[test]
fn full_vocabulary_head_matches_torch() {
    let path = fixtures().join("tiny_eagle3_full/scenarios.json");
    for be in backends() {
        for (kv, sfx) in [(CacheDtype::F32, ""), (CacheDtype::Bf16, "_bf16")] {
            let load = tiny_load_head(be.clone(), 4, kv, true, "tiny_eagle3_full");
            assert!(load().describe().contains("taps 0,1,2; vocab 256"), "{}", load().describe());
            let t = run_scenarios(&load, &path, 1e-3, true, sfx);
            eprintln!("{be} {kv:?}: {t:?}");
            check_tally(&t, &be, &path, &format!("{be} {kv:?}"));
            assert!(t.requests > 500, "{be}: {t:?}");
        }
    }
}

/// Heads fullm cannot run as vLLM would are refused at load, saying why.
#[test]
fn refuses_heads_it_cannot_run() {
    let src = fixtures().join("tiny_eagle3");
    let base: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(src.join("config.json")).unwrap()).unwrap();
    let edit = |f: &dyn Fn(&mut serde_json::Value)| {
        let mut c = base.clone();
        f(&mut c);
        c
    };
    let set = |k: &'static str, v: serde_json::Value| move |c: &mut serde_json::Value| c[k] = v.clone();
    let cases: Vec<(serde_json::Value, &str)> = vec![
        (edit(&set("architectures", serde_json::json!(["LlamaForCausalLM"]))), "not an EAGLE-3"),
        (edit(&set("num_hidden_layers", 2.into())), "more than one layer"),
        (edit(&set("head_dim", 16.into())), "attention shape"),
        (edit(&set("num_key_value_heads", 4.into())), "attention shape"),
        (edit(&set("vocab_size", 300.into())), "vocab_size"),
        (edit(&set("rope_theta", 10000.0.into())), "RoPE"),
        (edit(&set("rms_norm_eps", 1e-5.into())), "rms_norm_eps"),
        (edit(&set("hidden_act", "gelu".into())), "activation"),
        (edit(&set("norm_before_fc", true.into())), "norm_before_fc"),
        (edit(&|c| c["eagle_config"]["norm_before_residual"] = true.into()), "norm_before_residual"),
        (edit(&|c| c["eagle_config"]["use_aux_hidden_state"] = false.into()), "EAGLE-1/2"),
        (edit(&set("num_aux_hidden_states", 4.into())), "three tapped"),
        (
            edit(&|c| c["eagle_config"]["eagle_aux_hidden_state_layer_ids"] = serde_json::json!([1, 1, 2])),
            "three distinct",
        ),
        (
            edit(&|c| c["eagle_config"]["eagle_aux_hidden_state_layer_ids"] = serde_json::json!([0, 1, 3])),
            "past the target",
        ),
        (edit(&set("draft_vocab_size", 100.into())), "lm_head.weight"),
        (edit(&set("intermediate_size", 64.into())), "mlp"),
    ];
    let dir = std::env::temp_dir().join(format!("fullm-eagle3-refuse-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(src.join("model.safetensors"), dir.join("model.safetensors")).unwrap();
    let opts = LoadOptions {
        backend: BackendKind::CPU,
        kv_dtype: CacheDtype::F32,
        max_seqs: 2,
        cache_tokens: 256,
        page: 4,
        max_context: 128,
        step_tokens: 32,
        max_rows: 32,
        threads: 2,
        spec_model: Some(dir.clone()),
        ..Default::default()
    };
    let load = || models::load(&fixtures().join("tiny_qwen3_moe"), &opts).map(|_| ()).map_err(|e| format!("{e:#}"));
    for (cfg, want) in &cases {
        std::fs::write(dir.join("config.json"), serde_json::to_string(cfg).unwrap()).unwrap();
        let err = load().expect_err(&format!("loads with {cfg}"));
        assert!(err.contains(want), "expected {want:?} in: {err}");
    }
    // the unedited config loads
    std::fs::write(dir.join("config.json"), serde_json::to_string(&base).unwrap()).unwrap();
    load().unwrap();
    // a pytorch_model.bin head: a pointer to the conversion
    std::fs::remove_file(dir.join("model.safetensors")).unwrap();
    std::fs::write(dir.join("pytorch_model.bin"), b"not read").unwrap();
    assert!(load().unwrap_err().contains("convert it"));
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A real model and head against torch on scenarios from `make_eagle3_scenarios.py --natural`:
/// FULLM_EAGLE_REAL="TARGET_DIR|HEAD_DIR|SCENARIOS.json" (FULLM_EAGLE_TOL: the near-tie margin,
/// default 0.02; FULLM_EAGLE_BACKEND: cpu or cuda, default cpu). f32 and bf16 caches, each
/// against its torch reference; prints the tallies.
#[test]
#[ignore]
fn real_model_matches_torch() {
    let Ok(spec) = std::env::var("FULLM_EAGLE_REAL") else {
        eprintln!("FULLM_EAGLE_REAL is not set");
        return;
    };
    let parts: Vec<&str> = spec.split('|').collect();
    let (target, head, path) = (PathBuf::from(parts[0]), PathBuf::from(parts[1]), PathBuf::from(parts[2]));
    let tol: f32 = std::env::var("FULLM_EAGLE_TOL").ok().and_then(|t| t.parse().ok()).unwrap_or(0.02);
    let backend: BackendKind = std::env::var("FULLM_EAGLE_BACKEND").unwrap_or_else(|_| "cpu".into()).parse().unwrap();
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let step = v["step_tokens"].as_u64().unwrap() as usize;
    for (kv, sfx) in [(CacheDtype::F32, ""), (CacheDtype::Bf16, "_bf16")] {
        let opts = LoadOptions {
            trunk: Precision::Bf16,
            max_seqs: 8,
            cache_tokens: 32768,
            page: 16,
            backend: backend.clone(),
            kv_dtype: kv,
            step_tokens: step.max(64),
            max_rows: 64,
            max_context: 8192,
            threads: 0,
            spec_model: Some(head.clone()),
            ..Default::default()
        };
        let model = std::cell::RefCell::new(Some(models::load(&target, &opts).unwrap()));
        // one model for every scenario (the prefix cache flushed between them)
        let load = || model.borrow_mut().take().expect("loaded once");
        let t = run_scenarios(&load, &path, tol, false, sfx);
        eprintln!(
            "{kv:?}: {} requests, {} drafts compared, {} chains as torch, {} near-ties (margin < {tol}), {} wrong, \
             {} prefix hits ({} positions)",
            t.requests,
            t.drafts,
            t.chains_exact,
            t.near_ties,
            t.wrong.len(),
            t.hits,
            t.cached
        );
        for w in &t.wrong {
            eprintln!("   {w}");
        }
    }
}

/// An executor destroyed right after it was created (as when a head is refused at load) never
/// hangs: the CPU thread pool once missed its stop when torn down before its threads first ran.
/// Thousands of executors dropped as soon as they exist, then 100 refused loads, under a watchdog.
#[test]
fn executors_torn_down_right_after_creation() {
    use fullm::models::paged::Geometry;
    use fullm_kernels::exec::{DenseDims, Executor};
    let src = fixtures().join("tiny_eagle3");
    let dir = std::env::temp_dir().join(format!("fullm-eagle3-teardown-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(src.join("model.safetensors"), dir.join("model.safetensors")).unwrap();
    let mut cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(src.join("config.json")).unwrap()).unwrap();
    cfg["num_hidden_layers"] = 2.into();
    std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let d = dir.clone();
    std::thread::spawn(move || {
        let opts = LoadOptions {
            backend: BackendKind::CPU,
            kv_dtype: CacheDtype::F32,
            max_seqs: 2,
            cache_tokens: 64,
            page: 4,
            max_context: 64,
            step_tokens: 16,
            max_rows: 16,
            threads: 4,
            spec_model: Some(d),
            ..Default::default()
        };
        let g = Geometry::new(&opts).unwrap();
        let dims = DenseDims {
            n_layers: 1,
            hidden: 64,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 16,
            inter: 64,
            vocab: 32,
            rms_eps: 1e-6,
            qk_norm: false,
            inv_freq: vec![1.0; 8],
            n_experts: 0,
            topk: 0,
            moe_inter: 0,
            norm_topk: false,
        };
        for _ in 0..3000 {
            drop(Executor::new_dense(&dims, &g.exec_opts(&opts)).unwrap());
        }
        for _ in 0..100 {
            assert!(models::load(&fixtures().join("tiny_qwen3_moe"), &opts).is_err());
        }
        tx.send(()).unwrap();
    });
    rx.recv_timeout(std::time::Duration::from_secs(120)).expect("an executor's teardown hung");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Adaptive source choice: on replies that copy their prompts (n-gram drafts keep many tokens) a
/// random head (it keeps next to nothing) drafts only where nothing matches and on its trials,
/// keeping nearly as many tokens as n-gram-first; no token changes, whatever drafts.
#[test]
fn adaptive_choice_learns_which_source_keeps_more() {
    use fullm::runtime::SpecChoice;
    // a phrase three times, then its start: n-gram matches from the first decode on
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    let reqs: Vec<Req> = (0..6)
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
            (p, greedy(64))
        })
        .collect();
    let opts = spy_opts();
    let want = reference(&opts, &reqs);
    let run = |choice: SpecChoice, own: bool| {
        let cfg = EngineConfig { spec_ngram: (2, 4), spec_choice: choice, spec_own: own, ..spy_cfg() };
        let (h, log) = spied(&opts, cfg);
        let got = run_all(&h, &reqs);
        let st = settled(&h);
        assert_eq!(toks(&got), want, "{choice:?} own {own}");
        assert_clean(&log, &format!("{choice:?}"), 20);
        st
    };
    let first = run(SpecChoice::NgramFirst, true);
    let adaptive = run(SpecChoice::Adaptive, true);
    let head = run(SpecChoice::Adaptive, false);
    eprintln!(
        "kept per draft step (chunks with drafts, of them the head's): ngram-first {} ({}, {}), adaptive {} ({}, {}), \
         head only {} ({}, {})",
        first.spec_accepted,
        first.spec_chunks,
        first.spec_chunks_eagle,
        adaptive.spec_accepted,
        adaptive.spec_chunks,
        adaptive.spec_chunks_eagle,
        head.spec_accepted,
        head.spec_chunks,
        head.spec_chunks_eagle
    );
    assert!(head.spec_accepted * 4 < first.spec_accepted, "the random head keeps little");
    // beyond the steps with no match (where n-gram-first takes the head too): a trial per 16 decode steps
    let trials = adaptive.decode_tokens / 16 + reqs.len() as u64;
    assert!(
        adaptive.spec_chunks_eagle <= first.spec_chunks_eagle + trials,
        "the head drafts where nothing matches and on trials: {adaptive:?} vs {first:?}"
    );
    assert!(adaptive.spec_accepted * 10 >= first.spec_accepted * 9, "{adaptive:?} vs {first:?}");
}
