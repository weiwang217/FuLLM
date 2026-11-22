// SPDX-License-Identifier: Apache-2.0
//! The engine: continuous batching over a `Model` on a dedicated thread.
//!
//! EACH STEP is one `forward` over one batch: every decoding sequence contributes its next
//! token, and prefilling sequences contribute chunks of their prompt under a per-step
//! token budget. Long prompts therefore never stall decodes, and a sequence's prefill may
//! span many steps; the backend's batch invariance makes that invisible in the output.
//!
//! CACHE ACCOUNTING. Every running request is expected to take the pages between what it holds
//! and its expected end; a request is admitted (FIFO) only when the sum of those, its own
//! included, fits in the pages a sequence can take now (free ones, and those only the prefix
//! cache keeps). Two expectations (`Admission`):
//!
//!   - worst case: the prompt plus all of max_tokens. Taking a page lowers both sides alike and
//!     finishing only lowers the sum, so a step never runs out of pages and nothing is preempted.
//!   - optimistic (SGLang's): the remaining output estimated at a fraction of min(max_tokens,
//!     4096), the fraction decaying from 0.7 towards 0.1 and re-estimated from what requests
//!     really generated whenever a step runs short. Requests that set max_tokens high but stop
//!     early then no longer hold the cache back.
//!
//! PREFILL ORDER. With `PrefillOrder::ShortFirst` (the default), the prompts that finish within
//! the step take its budget first, in admission order, and the rest goes to the others in
//! admission order: a short prompt that arrives behind a long one starts in the next step instead
//! of after the long one's whole prefill. While a prompt that cannot finish waits, the short ones
//! stop at three quarters of the budget, so the oldest of the long ones still gets a quarter of
//! every step however many short ones arrive.
//!
//! DECODE SHARE (`EngineConfig::decode_share`, `mixed_step_tokens`; off by default). A prompt
//! chunk rides in the step of the running decodes, which is free where a step's time barely grows
//! with its tokens (an H100: ~100 ms for 2048) and not where it grows with them (Inkling 2 on an
//! Apple GPU: 66 ms for one row, 4.3 s for 1024, so a request that was streaming gets one token
//! per chunk of another's prompt). With a share s, after a step that carried prompt tokens beside
//! decodes, the decodes get steps of their own for s / (1 - s) times as long as that step took,
//! and such a step takes at most `mixed_step_tokens` prompt tokens: the streams keep about s of
//! their speed, in pauses no longer than one such step, and the prompts take 1 / (1 - s) as long.
//!
//! SPECULATION (`EngineConfig::spec_tokens`). A decoding sequence whose context repeats its last
//! few tokens (`spec::Ngram`) feeds its next token plus up to `spec_tokens` drafts: the tokens
//! that followed the earlier occurrence. The step returns the model's output after each of them,
//! and the token the request would generate there is drawn as it would be without drafts (the
//! argmax, or the sampler's draw for that position: its random stream is keyed by the position,
//! not by the draws made). The drafts are kept up to the first that differs from its draw, then
//! that draw, so one step yields 1 to drafts + 1 tokens, the same ones decoding token by token
//! would on a backend whose rows do not depend on the batch, seeded sampling included; a draft d
//! is kept with probability p(d), as speculative sampling keeps a one-token draft. The rejected
//! drafts' cache is simply written over later (`Model::accept`). Drafts go in only while a step's
//! decode tokens stay within `spec_step_tokens`. A request whose own context has no match takes
//! drafts from what other requests' replies said after the same few tokens (`spec::Global`, fed
//! with every finished reply; `spec_global`).
//!
//! EAGLE-3 (a model with a draft head, `Model::draft_head`). After every step the head reads the
//! positions each chunk added (`Model::draft`: every chunk, so its cache covers every context)
//! and drafts up to `eagle_tokens` tokens for the next step of the decoding requests it is chosen
//! for; they are verified as n-gram drafts are. Which source drafts (`SpecChoice::Adaptive`): a
//! request without an n-gram match takes the head's; one with a match takes whichever has kept
//! more tokens per step for it lately (an average per source, started from fixed values: nothing
//! carries over from other requests, so the same request makes the same choices and, where the
//! numerics depend on how a step groups positions (the fp8 cache), comes out the same; the
//! match's are capped by its length). Each source gets a trial when it has not drafted for the
//! request in `TRIAL_EVERY` decode steps (the match first: it costs no head work; so a request's
//! first match is tried), so both averages follow the reply between copying and writing; a source
//! never tried would never be found better. Other replies' continuations come last.
//! (`SpecChoice::NgramFirst`: any match before the head.)
//!
//! PREEMPTION. Each step gives every decode its page first, then prompt chunks what is left (a
//! chunk shrinks to the pages there are, SGLang's rule). When a decode finds no page, or the
//! step could run nothing, the newest running request is preempted (vLLM's choice: the least
//! work lost): its pages are released, and it returns to the front of the queue with what it
//! generated (already streamed). Readmitted, it prefills its prompt and those tokens again,
//! from its own pages wherever the prefix cache still holds them (they entered it as they
//! filled), and samples on. Seeds and the sampler's step count carry over, and a resumed prefill
//! is a chunk boundary, so the output is the same as without the preemption wherever chunked
//! prefill is bit-identical.
//!
//! PREFIX CACHE. A request starts from the longest prefix of its prompt that the model still
//! caches (`Model::alloc_seq_cached`) and prefills only the rest. Pages it shares are already
//! held, so a prefix counts once however many requests share it. A request whose prefix another
//! one is still prefilling waits until those pages are in the cache instead of computing them
//! a second time (SGLang's in-batch prefix caching); requests behind it are not held up.
//!
//! HOST TIER. Past what the device holds, the model's host tier may hold more of the prefix
//! (`Model::host_cached`). Admission counts those pages like any others the request will take;
//! once admitted it loads them (`Model::load_cached`) and its prefill skips them, but it runs
//! only when the copies are done (`Model::seq_ready`): the other requests keep stepping
//! meanwhile, and a step with nothing else to run waits for the first of them to finish.
//!
//! FAILURE. If `forward` errors, every request in that batch is failed and its sequence
//! freed: the backend may have written part of the step's state, so continuing would be
//! silently wrong. Requests outside the batch are untouched.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{ensure, Context, Result};
use serde::Serialize;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

use super::sampler::{adjustments, logprobs, sample, LogitAdj, SamplingParams, TokenLogprobs, MAX_TOP_LOGPROBS};
use super::spec::{Global, Ngram};
use crate::models::{Capacity, DraftReq, Model, SampleIo, SampleRow, SeqChunk, SeqId};

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// max_tokens reached.
    Length,
    /// A stop id was sampled.
    Stop,
    /// The model's context is full.
    ContextLimit,
    /// The step failed; the message says why.
    Error(String),
}

impl FinishReason {
    /// The OpenAI `finish_reason` string.
    pub fn as_openai(&self) -> &'static str {
        match self {
            FinishReason::Length | FinishReason::ContextLimit => "length",
            FinishReason::Stop => "stop",
            FinishReason::Error(_) => "error",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// Prompt tokens whose cache came from the prefix cache instead of being computed.
    pub cached_tokens: usize,
}

#[derive(Clone, Debug)]
pub enum Event {
    /// The log-probabilities of the token the next `Token` carries (requests that ask for them).
    Logprobs(TokenLogprobs),
    Token(u32),
    Finished {
        reason: FinishReason,
        usage: Usage,
    },
}

pub struct Request {
    pub prompt: Vec<u32>,
    pub params: SamplingParams,
}

/// A grammar a request's output must follow (structured output, forced tool calls): every
/// position it samples has its logits masked by it, and it is told each token taken. The engine
/// gives such a request no drafts and samples it on the host.
pub trait Constraint: Send {
    /// The next position's allowed tokens into `words`, a bitmask over the vocabulary (token i:
    /// bit i % 32 of word i / 32; zeroed by the caller), never empty; false when the position is
    /// free (the constraint has not begun), `words` then untouched. The engine asks every
    /// constrained request of a step at once, in parallel, before its forward.
    fn allowed(&mut self, words: &mut [u32]) -> bool;
    /// The token taken at that position.
    fn advance(&mut self, tok: u32);
    /// Nothing may follow: the request ends there.
    fn done(&self) -> bool;
}

/// Ends a request from outside (the server, when a stop string completes): its next sampled
/// token is dropped and it finishes with `FinishReason::Stop` and its usage, as at a stop id.
#[derive(Clone, Debug, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    fn is_set(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Tokens per step across all chunks; decodes always go in, prefill fills the rest.
    pub step_tokens: usize,
    pub admission: Admission,
    /// Scales the optimistic output ratio's start and floor (SGLang's `--schedule-conservativeness`):
    /// above 1 admits fewer requests and preempts less, below 1 the reverse.
    pub conservativeness: f64,
    pub prefill_order: PrefillOrder,
    /// Speculative decoding: the most drafts a decoding sequence verifies per step (0: off),
    /// found by n-gram lookup in its own context with n from `spec_ngram.1` down to `.0`, on
    /// models that can (`Model::speculation`).
    pub spec_tokens: usize,
    pub spec_ngram: (usize, usize),
    /// Drafts go in only while a step's decode tokens, drafts included, stay within this.
    pub spec_step_tokens: usize,
    /// Draft lengths follow each request's acceptance (`draft_len`); off: every match drafts its
    /// full share.
    pub spec_by_acceptance: bool,
    /// Draft from finished replies (across requests) when a request's own context has no match.
    pub spec_global: bool,
    /// With a draft head: the most tokens it drafts for a request per step.
    pub eagle_tokens: usize,
    /// Draft from a request's own context (n-gram matches); off: only the head and other replies.
    pub spec_own: bool,
    /// With a draft head: how a request with an n-gram match picks its drafts' source.
    pub spec_choice: SpecChoice,
    /// While requests decode: the share of the time kept for steps without prompt chunks (0: none;
    /// see the module comment, DECODE SHARE).
    pub decode_share: f32,
    /// While requests decode: the most prompt tokens a step takes (0: the step's budget).
    pub mixed_step_tokens: usize,
}

/// How a decoding request whose own context has an n-gram match picks between that match and the
/// draft head (see the module comment).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum SpecChoice {
    /// Whichever has kept more tokens per step for it lately; the head tried now and then.
    #[default]
    Adaptive,
    /// The match, always.
    NgramFirst,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            step_tokens: 512,
            admission: Admission::Auto,
            conservativeness: 1.0,
            prefill_order: PrefillOrder::ShortFirst,
            spec_tokens: 0,
            spec_ngram: (2, 4),
            spec_step_tokens: 256,
            spec_by_acceptance: true,
            spec_global: true,
            eagle_tokens: 4,
            spec_own: true,
            spec_choice: SpecChoice::Adaptive,
            decode_share: 0.0,
            mixed_step_tokens: 0,
        }
    }
}

/// Which prompts a step's prefill budget goes to first (see the module comment).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum PrefillOrder {
    /// Prompts that finish within the step first, then the rest in admission order; while a
    /// longer prompt waits, the short ones stop at three quarters of the budget.
    #[default]
    ShortFirst,
    /// Admission order (vLLM's and SGLang's): a long prompt ahead takes every step's budget until
    /// it is done, and the prompts behind it wait.
    Arrival,
}

/// How admission counts the output a request has yet to generate (see the module comment).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Admission {
    /// Optimistic when the model keeps a prefix cache (a preempted request then resumes from its
    /// cached pages), worst case otherwise.
    #[default]
    Auto,
    /// The output estimated (SGLang's rule); a step that runs short preempts.
    Optimistic,
    /// All of max_tokens: nothing is ever preempted.
    WorstCase,
}

/// The optimistic estimate of a request's remaining output: `ratio` x min(left, OUT_CLIP), the
/// ratio decaying from RATIO_INIT to RATIO_MIN (both times the conservativeness, at most 1) over
/// RATIO_DECAY_STEPS decoding steps; after a preemption, (generated + RETRACT_STEPS per request) /
/// max_tokens over the running requests (SGLang's constants and rule).
const OUT_CLIP: usize = 4096;
const RATIO_INIT: f64 = 0.7;
const RATIO_MIN_FACTOR: f64 = 0.14;
const RATIO_DECAY_STEPS: f64 = 600.0;
const RETRACT_STEPS: usize = 20;

/// Adaptive draft sources: the weight of a step's kept drafts in a request's average per source;
/// the averages before its first step; how many decode steps a source goes without drafting for a
/// request before it gets a trial.
const KEPT_ALPHA: f32 = 0.3;
const KEPT_START: [f32; 2] = [1.5, 1.0];
const TRIAL_EVERY: u32 = 16;

/// Draft lengths by acceptance (n-gram drafts, a request's own and other replies'). A request
/// keeps, per source, decayed counts of the drafts it kept and of the steps a draft failed in
/// (`ACCEPT_DECAY` a drafted step); a = kept / (kept + failures), from `ACCEPT_PRIOR`, is the chance
/// a draft holds, and it drafts while a^k stays above a floor: `ACCEPT_FLOOR` once
/// `ACCEPT_BUSY` requests decode (a row then costs about what it yields below that), in proportion
/// with fewer, where a step's cost barely grows with its rows. At least one draft, which keeps the
/// counts following the reply between copying and writing. Measured on an H100 (Qwen3-30B-A3B,
/// int4 experts): a request keeps all 8 drafts in 59% of its drafted steps and none in 21% on agent
/// sessions, none in 73% on fresh text, and the step before predicts the next (86% kept after a
/// full keep, 24% after none).
const ACCEPT_DECAY: f32 = 0.8;
const ACCEPT_PRIOR: [f32; 2] = [2.0, 1.0];
const ACCEPT_FLOOR: f32 = 0.3;
const ACCEPT_BUSY: usize = 8;

/// How many drafts a request with these counts ([kept, failures]) takes at this floor.
fn draft_len(acc: [f32; 2], floor: f32) -> usize {
    let a = (acc[0] + ACCEPT_PRIOR[0]) / (acc[0] + acc[1] + ACCEPT_PRIOR[0] + ACCEPT_PRIOR[1]);
    if a >= 0.999 || floor <= 0.0 {
        return usize::MAX;
    }
    ((floor.ln() / a.ln()) as usize).max(1)
}

/// The global continuations' longest n-gram, and how many n-grams they hold before starting over
/// (~50 bytes each).
const GLOBAL_MAX_N: usize = 8;
const GLOBAL_CAP: usize = 1 << 21;

#[derive(Clone, Debug, Default, Serialize)]
pub struct Stats {
    pub steps: u64,
    pub prefill_tokens: u64,
    pub decode_tokens: u64,
    /// Prompt tokens of admitted requests, and those served by the prefix cache.
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    /// Positions loaded from the host tier (part of `cached_tokens` on first admissions).
    pub host_loaded_tokens: u64,
    /// Requests preempted (each time); positions their resumptions computed again (the rest came
    /// from the prefix cache); the admission's current output ratio (1 when it counts the worst
    /// case).
    pub preemptions: u64,
    pub recomputed_tokens: u64,
    pub output_ratio: f64,
    /// Speculation: draft tokens verified, and those the model agreed with; of them, the drafts
    /// taken from other requests' replies.
    pub spec_drafted: u64,
    pub spec_accepted: u64,
    pub spec_drafted_global: u64,
    pub spec_accepted_global: u64,
    /// Of them, the draft head's (EAGLE-3).
    pub spec_drafted_eagle: u64,
    pub spec_accepted_eagle: u64,
    /// Chunks that carried drafts, and those whose drafts were the head's (vLLM's mean
    /// acceptance length is 1 + accepted / chunks).
    pub spec_chunks: u64,
    pub spec_chunks_eagle: u64,
    /// Exponential moving average of step wall time.
    pub step_ms_ema: f64,
    pub running: usize,
    pub waiting: usize,
    pub finished: u64,
    pub failed: u64,
    pub capacity: Capacity,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelInfo {
    pub arch: String,
    pub vocab: usize,
    pub max_context: usize,
    pub description: String,
    /// LoRA adapter slots (unmerged adapters; 0: none).
    pub lora_slots: usize,
}

enum Cmd {
    Submit(Pending),
    FlushCache(tokio::sync::oneshot::Sender<usize>),
    /// An adapter into a slot (None: clear it; refused while a request runs with it).
    Lora(usize, Option<Arc<crate::models::lora::Adapter>>, tokio::sync::oneshot::Sender<Result<(), String>>),
}

struct Pending {
    id: u64,
    req: Request,
    tx: UnboundedSender<Event>,
    cancel: Cancel,
    submitted: Instant,
    /// A preempted request: what it generated (streamed already), and its prompt tokens served
    /// by the prefix cache at its first admission (the usage it reports).
    generated: Vec<u32>,
    cached: Option<usize>,
    constraint: Option<Box<dyn Constraint>>,
}

impl Pending {
    /// What its sequence must hold before it samples: the prompt, and what it generated.
    fn context(&self) -> Vec<u32> {
        [self.req.prompt.as_slice(), &self.generated].concat()
    }
}

struct Live {
    id: u64,
    seq: SeqId,
    prompt: Vec<u32>,
    params: SamplingParams,
    /// Tokens fed to the model so far: its length for this sequence.
    fed: usize,
    /// Prefill up to here, then sample: the prompt, plus the tokens generated before a
    /// preemption.
    target: usize,
    /// Prompt tokens found in the prefix cache at (first) admission.
    cached: usize,
    generated: Vec<u32>,
    tx: UnboundedSender<Event>,
    cancel: Cancel,
    done: bool,
    /// When the request was submitted, until its first step (the trace's queue time).
    submitted: Option<Instant>,
    /// Its context indexed for drafts, when it speculates.
    ngram: Option<Ngram>,
    /// The draft head's proposal for its next step (EAGLE-3).
    eagle: Vec<u32>,
    /// Drafts kept per step lately (an average), when they came from its own n-gram match [0] and
    /// from the head [1]; its decode steps since each last drafted for it.
    kept: [f32; 2],
    since: [u32; 2],
    /// Its n-gram drafts' record (`draft_len`): [kept, failures], for its own matches [0] and for
    /// other replies' continuations [1].
    acc: [[f32; 2]; 2],
    /// The grammar its output follows (no drafts then; sampled on the host).
    constraint: Option<Box<dyn Constraint>>,
    /// Penalties (`SamplingParams::adjusts`): how often it generated each token, and its
    /// prompt's distinct tokens when the repetition penalty applies.
    counts: HashMap<u32, u32>,
    prompt_set: Vec<u32>,
}

impl Live {
    /// The token at position `p` of its sequence: the prompt, then what it generated.
    fn token(&self, p: usize) -> u32 {
        if p < self.prompt.len() {
            self.prompt[p]
        } else {
            self.generated[p - self.prompt.len()]
        }
    }
}

/// One sequence's part of a step.
struct Chunk {
    /// Index into `running`.
    i: usize,
    pos: usize,
    tokens: Vec<u32>,
    /// Sample from its last row afterwards.
    samples: bool,
    /// Its last `draft` tokens are drafts to verify, and where they came from.
    draft: usize,
    src: Src,
}

/// Where a chunk's drafts came from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Src {
    /// Its own context (or no drafts).
    Own,
    /// Other requests' replies.
    Global,
    /// The draft head.
    Eagle,
}

/// Cheap to clone; shared by every HTTP handler. When the last clone goes, the engine finishes
/// the requests it holds and its thread is joined (`Joiner`), so the model is torn down before
/// the process ends (a CUDA model torn down while the process exits crashed it).
#[derive(Clone)]
pub struct EngineHandle {
    tx: mpsc::Sender<Cmd>,
    stats: Arc<Mutex<Stats>>,
    next_id: Arc<AtomicU64>,
    info: Arc<ModelInfo>,
    /// Dropped after `tx` (field order): the last handle's sender is gone when it joins.
    _thread: Arc<Joiner>,
}

/// The engine thread, joined when the last handle drops.
struct Joiner(Mutex<Option<std::thread::JoinHandle<()>>>);

impl Drop for Joiner {
    fn drop(&mut self) {
        if let Some(t) = self.0.lock().ok().and_then(|mut g| g.take()) {
            if t.thread().id() != std::thread::current().id() {
                let _ = t.join();
            }
        }
    }
}

impl EngineHandle {
    /// Queue a request. Malformed requests are refused here, before they reach the step.
    pub fn submit(&self, req: Request) -> Result<UnboundedReceiver<Event>> {
        self.submit_cancellable(req).map(|(rx, _)| rx)
    }

    /// `submit`, with a handle that ends the request early.
    pub fn submit_cancellable(&self, req: Request) -> Result<(UnboundedReceiver<Event>, Cancel)> {
        self.submit_constrained(req, None)
    }

    /// `submit_cancellable`, its output held to `constraint` when given.
    pub fn submit_constrained(
        &self,
        mut req: Request,
        constraint: Option<Box<dyn Constraint>>,
    ) -> Result<(UnboundedReceiver<Event>, Cancel)> {
        let info = &self.info;
        ensure!(!req.prompt.is_empty(), "prompt is empty");
        ensure!(
            req.params.lora.is_none_or(|s| s < info.lora_slots),
            "no LoRA slot {:?} ({} reserved)",
            req.params.lora,
            info.lora_slots
        );
        ensure!(
            req.prompt.iter().all(|&t| (t as usize) < info.vocab),
            "prompt has a token id >= vocab size {}",
            info.vocab
        );
        ensure!(
            req.prompt.len() <= info.max_context,
            "prompt is {} tokens; the context holds {}",
            req.prompt.len(),
            info.max_context
        );
        ensure!(req.params.max_tokens >= 1, "max_tokens must be at least 1");
        // no reply outgrows the context, which ends it first: so the page estimates never overflow
        req.params.max_tokens = req.params.max_tokens.min(info.max_context);
        req.params.min_tokens = req.params.min_tokens.min(req.params.max_tokens);
        ensure!(
            req.params.logit_bias.iter().all(|&(t, _)| (t as usize) < info.vocab),
            "logit_bias has a token id >= vocab size {}",
            info.vocab
        );
        ensure!(req.params.temperature.is_finite() && req.params.temperature >= 0.0, "temperature must be >= 0");
        ensure!(req.params.top_p > 0.0 && req.params.top_p <= 1.0, "top_p must be in (0, 1]");
        let (tx, rx) = unbounded_channel();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let submitted = Instant::now();
        let cancel = Cancel::default();
        let p =
            Pending { id, req, tx, cancel: cancel.clone(), submitted, generated: Vec::new(), cached: None, constraint };
        self.tx.send(Cmd::Submit(p)).ok().context("engine thread has stopped")?;
        Ok((rx, cancel))
    }

    /// Adapter `a` into LoRA slot `slot` (None: clear it, refused while a request runs with it),
    /// between steps.
    pub async fn set_lora(&self, slot: usize, a: Option<Arc<crate::models::lora::Adapter>>) -> Result<()> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.tx.send(Cmd::Lora(slot, a, tx)).ok().context("engine thread has stopped")?;
        rx.await.context("engine thread has stopped")?.map_err(|e| anyhow::anyhow!(e))
    }

    /// Drop the prefix cache's pages that no request holds (SGLang's /flush_cache, vLLM's
    /// /reset_prefix_cache). Returns the positions freed.
    pub async fn flush_cache(&self) -> Result<usize> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.tx.send(Cmd::FlushCache(tx)).ok().context("engine thread has stopped")?;
        rx.await.context("engine thread has stopped")
    }

    pub fn stats(&self) -> Stats {
        self.stats.lock().map(|s| s.clone()).unwrap_or_default()
    }

    pub fn info(&self) -> &ModelInfo {
        &self.info
    }
}

pub struct Engine {
    model: Box<dyn Model>,
    cfg: EngineConfig,
    /// `cfg.spec_tokens` > 0 on a model that can speculate; the continuations of finished replies.
    spec: bool,
    global: Option<Global>,
    /// Speculating with a draft head (EAGLE-3); FULLM_EAGLE_CONSUME_ONLY (measurement): the head
    /// reads every chunk but never drafts.
    eagle: bool,
    eagle_consume_only: bool,
    /// The model draws sampled rows on the device; else the last sampled step's logits,
    /// `[rows][vocab]` (kept for its memory), go to the sampler.
    device_sampling: bool,
    logits: Vec<f32>,
    /// A step's log-probabilities from the device (`SampleIo`), kept for their memory.
    lp_buf: Vec<f32>,
    lp_ids: Vec<i32>,
    waiting: VecDeque<Pending>,
    running: Vec<Live>,
    /// Total cache positions, and how many are promised to admitted requests.
    cache_total: usize,
    /// Whether the model keeps a prefix cache (requests may then wait for a shared prefix).
    prefix_cache: bool,
    /// `cfg.admission` resolved (never Auto), and the optimistic output ratio with its start and
    /// floor.
    admission: Admission,
    ratio: f64,
    ratio_init: f64,
    ratio_min: f64,
    /// The model's cache page, for page-rounded admission accounting.
    page: usize,
    max_context: usize,
    stats: Arc<Mutex<Stats>>,
    /// FULLM_STEP_TRACE: log every step's timing (the gap since the previous step, planning,
    /// the model call and the executor's own upload/download within it, post-processing).
    trace: bool,
    last_end: Option<Instant>,
    /// With a decode share: until when the running decodes step without prompt chunks.
    decode_until: Option<Instant>,
    /// For the trace: requests the last plan skipped while they load from the host tier, and the
    /// time it spent waiting for copies.
    loading: usize,
    copy_wait_ms: f64,
    /// For the trace: the pages the first request admission turned away needed, and the running
    /// requests' expected pages beside it (0 when nothing was turned away).
    head_need: usize,
}

impl Engine {
    pub fn new(model: Box<dyn Model>, cfg: EngineConfig) -> Self {
        let cap = model.capacity();
        let max_context = model.max_context();
        let page = model.page_tokens().max(1);
        let prefix_cache = model.has_prefix_cache();
        let ratio_init = (RATIO_INIT * cfg.conservativeness).min(1.0);
        let admission = match cfg.admission {
            Admission::Auto if prefix_cache => Admission::Optimistic,
            Admission::Auto => Admission::WorstCase,
            a => a,
        };
        let spec = cfg.spec_tokens > 0 && model.speculation();
        let device_sampling = model.device_sampling();
        let global = (spec && cfg.spec_global).then(|| Global::new(cfg.spec_ngram.0, GLOBAL_MAX_N, GLOBAL_CAP));
        if cfg.spec_tokens > 0 && !spec {
            tracing::warn!("speculative decoding is off: {} keeps state that drafts cannot roll back", model.arch());
        }
        let eagle = spec && model.draft_head() && cfg.eagle_tokens > 0;
        if model.draft_head() && !eagle {
            tracing::warn!("the draft head is loaded but speculation is off: it drafts nothing");
        }
        Engine {
            model,
            cfg,
            global,
            spec,
            eagle,
            eagle_consume_only: std::env::var_os("FULLM_EAGLE_CONSUME_ONLY").is_some(),
            device_sampling,
            logits: Vec::new(),
            lp_buf: Vec::new(),
            lp_ids: Vec::new(),
            admission,
            ratio: ratio_init,
            ratio_init,
            ratio_min: ratio_init * RATIO_MIN_FACTOR,
            waiting: VecDeque::new(),
            running: Vec::new(),
            cache_total: cap.free_cache_tokens,
            prefix_cache,
            page,
            max_context,
            stats: Arc::new(Mutex::new(Stats { capacity: cap, output_ratio: 1.0, ..Default::default() })),
            trace: std::env::var_os("FULLM_STEP_TRACE").is_some(),
            last_end: None,
            decode_until: None,
            loading: 0,
            copy_wait_ms: 0.0,
            head_need: 0,
        }
    }

    /// Move the engine onto its own thread and return the handle that feeds it. The thread
    /// exits once every handle is dropped and the work in flight has finished.
    pub fn spawn(self) -> Result<EngineHandle> {
        let info = Arc::new(ModelInfo {
            arch: self.model.arch().to_string(),
            vocab: self.model.vocab(),
            max_context: self.max_context.min(self.cache_total),
            description: self.model.describe(),
            lora_slots: self.model.lora_slots(),
        });
        let (tx, rx) = mpsc::channel();
        let stats = self.stats.clone();
        let t = std::thread::Builder::new()
            .name("fullm-engine".into())
            .spawn(move || self.run(rx))
            .context("spawning the engine thread")?;
        let _thread = Arc::new(Joiner(Mutex::new(Some(t))));
        Ok(EngineHandle { tx, stats, next_id: Arc::new(AtomicU64::new(1)), info, _thread })
    }

    fn run(mut self, rx: mpsc::Receiver<Cmd>) {
        let mut open = true;
        loop {
            if self.running.is_empty() && self.waiting.is_empty() {
                if !open {
                    return;
                }
                match rx.recv() {
                    Ok(cmd) => self.handle(cmd),
                    Err(_) => return,
                }
            }
            loop {
                match rx.try_recv() {
                    Ok(cmd) => self.handle(cmd),
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        open = false;
                        break;
                    }
                }
            }
            if !self.step() {
                // Nothing could run (e.g. waiting on capacity that only a finishing request
                // frees, which cannot happen with nothing running). Do not spin.
                if let Ok(cmd) = rx.recv_timeout(Duration::from_millis(5)) {
                    self.handle(cmd);
                }
            }
        }
    }

    fn handle(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::Submit(p) => self.waiting.push_back(p),
            Cmd::FlushCache(tx) => {
                let n = self.model.flush_prefix_cache();
                // the stats first: a caller that reads them on the reply sees the flush
                self.publish();
                let _ = tx.send(n);
            }
            Cmd::Lora(slot, a, tx) => {
                let busy = self.running.iter().any(|l| l.params.lora == Some(slot))
                    || self.waiting.iter().any(|p| p.req.params.lora == Some(slot));
                let r = if busy {
                    Err(format!("LoRA slot {slot} has requests running"))
                } else {
                    match a {
                        Some(a) => self.model.set_lora(slot, &a),
                        None => self.model.clear_lora(slot),
                    }
                    .map_err(|e| format!("{e:#}"))
                };
                // the slot's old cache must not serve the new adapter: its keys carry the slot
                if r.is_ok() {
                    self.model.flush_prefix_cache();
                    self.publish();
                }
                let _ = tx.send(r);
            }
        }
    }

    fn pages(&self, n: usize) -> usize {
        n.div_ceil(self.page)
    }

    /// The length a request is expected to reach from `len` (what it holds, or must prefill):
    /// plus all of its remaining output under worst-case admission or when it has no stop ids
    /// (it runs to max_tokens), else the optimistic estimate; within the context.
    fn expected_end(&self, len: usize, params: &SamplingParams, generated: usize) -> usize {
        let left = params.max_tokens.saturating_sub(generated);
        let out = if self.admission == Admission::WorstCase || params.stop_token_ids.is_empty() {
            left
        } else {
            ((left.min(OUT_CLIP) as f64 * self.ratio).ceil() as usize).clamp(left.min(1), left)
        };
        (len + out).min(self.max_context)
    }

    /// Pages a running request is expected to take still.
    fn expected_pages(&self, l: &Live) -> usize {
        let end = self.expected_end(l.target.max(l.fed), &l.params, l.generated.len());
        self.pages(end).saturating_sub(self.pages(l.fed))
    }

    fn admit(&mut self) {
        self.waiting.retain(|p| !p.tx.is_closed());
        // a request cancelled while it waits (after a preemption) ends here
        let mut k = 0;
        while k < self.waiting.len() {
            if !self.waiting[k].cancel.is_set() {
                k += 1;
                continue;
            }
            let p = self.waiting.remove(k).expect("index in range");
            let usage = Usage {
                prompt_tokens: p.req.prompt.len(),
                completion_tokens: p.generated.len(),
                cached_tokens: p.cached.unwrap_or(0),
            };
            let _ = p.tx.send(Event::Finished { reason: FinishReason::Stop, usage });
        }
        self.head_need = 0;
        let mut expected: usize = self.running.iter().map(|l| self.expected_pages(l)).sum();
        let mut i = 0;
        while i < self.waiting.len() {
            let p = &self.waiting[i];
            let ctx = p.context();
            // the prefix cache's keys: the tokens, or under an adapter the adapter's (models::lora_key)
            let lora = p.req.params.lora;
            let key: Vec<u32> = match lora {
                Some(_) => ctx.iter().map(|&t| crate::models::lora_key(t, lora)).collect(),
                None => ctx.clone(),
            };
            let Some((seq, mut cached)) = self.model.alloc_seq_cached(&key) else { break };
            let host = self.model.host_cached(seq, &key);
            if self.prefilling_shared(&key, lora, cached + host) {
                self.model.free_seq(seq);
                i += 1;
                continue;
            }
            // the pages loaded from the host are taken like the pages computed
            let need = self.pages(self.expected_end(ctx.len(), &p.req.params, p.generated.len())) - self.pages(cached);
            let avail = self.model.capacity().free_cache_tokens / self.page;
            if expected + need > avail || !self.model.can_grow(seq, ctx.len() - cached) {
                self.model.free_seq(seq);
                self.head_need = need + expected;
                break;
            }
            let loaded = match self.model.load_cached(seq, &key, host).and_then(|n| {
                self.model.set_seq_lora(seq, lora)?;
                Ok(n)
            }) {
                Ok(n) => n,
                Err(e) => {
                    let p = self.waiting.remove(i).expect("index in range");
                    self.model.free_seq(seq);
                    tracing::error!(request = p.id, "{e:#}");
                    let usage = Usage {
                        prompt_tokens: p.req.prompt.len(),
                        completion_tokens: p.generated.len(),
                        cached_tokens: p.cached.unwrap_or(0),
                    };
                    let _ = p.tx.send(Event::Finished { reason: FinishReason::Error(format!("{e:#}")), usage });
                    continue;
                }
            };
            cached += loaded;
            expected += need - self.pages(loaded);
            let p = self.waiting.remove(i).expect("index in range");
            tracing::debug!(
                request = p.id,
                seq,
                context = ctx.len(),
                cached,
                loaded,
                resumed = p.cached.is_some(),
                "admitted"
            );
            if let Ok(mut s) = self.stats.lock() {
                if p.cached.is_none() {
                    s.prompt_tokens += p.req.prompt.len() as u64;
                    s.cached_tokens += cached as u64;
                } else {
                    s.recomputed_tokens += (ctx.len() - cached) as u64;
                }
                s.host_loaded_tokens += loaded as u64;
            }
            // drafts cannot be verified against draws whose logits depend on the drafts themselves
            // (penalties), nor against a grammar
            let ngram = (self.spec && p.constraint.is_none() && !p.req.params.adjusts()).then(|| {
                let mut g = Ngram::new(self.cfg.spec_ngram.0, self.cfg.spec_ngram.1);
                g.push(&ctx);
                g
            });
            let mut counts = HashMap::new();
            if p.req.params.adjusts() {
                for &t in &p.generated {
                    *counts.entry(t).or_insert(0) += 1;
                }
            }
            let prompt_set = if p.req.params.repetition_penalty != 1.0 {
                let mut s = p.req.prompt.clone();
                s.sort_unstable();
                s.dedup();
                s
            } else {
                Vec::new()
            };
            self.running.push(Live {
                id: p.id,
                seq,
                prompt: p.req.prompt,
                params: p.req.params,
                fed: cached,
                target: ctx.len(),
                cached: p.cached.unwrap_or(cached),
                generated: p.generated,
                tx: p.tx,
                cancel: p.cancel,
                done: false,
                submitted: Some(p.submitted),
                ngram,
                eagle: Vec::new(),
                kept: KEPT_START,
                // its first match is tried
                since: [TRIAL_EVERY, 0],
                acc: [[0.0; 2]; 2],
                constraint: p.constraint,
                counts,
                prompt_set,
            });
        }
    }

    /// This step's chunks: every decode, then prompt chunks in `PrefillOrder` within the token
    /// budget and the pages left, skipping requests still loading from the host tier. A decode
    /// that finds no page, or a step that could run nothing, preempts the newest request and plans
    /// again; a step with nothing to run but loads waits for them. Also returns whether it
    /// preempted.
    fn plan(&mut self) -> (Vec<Chunk>, bool) {
        let mut preempted = false;
        self.copy_wait_ms = 0.0;
        loop {
            let (running, model) = (&self.running, &mut self.model);
            let ready: Vec<bool> = running.iter().map(|l| l.fed >= l.target || model.seq_ready(l.seq)).collect();
            self.loading = ready.iter().filter(|r| !**r).count();
            let mut avail = self.model.capacity().free_cache_tokens / self.page;
            let mut plan = Vec::new();
            let mut short = false;
            for (i, l) in self.running.iter().enumerate() {
                if l.fed >= l.target {
                    let need = self.pages(l.fed + 1) - self.pages(l.fed);
                    if need > avail {
                        short = true;
                        break;
                    }
                    avail -= need;
                    let last = *l.generated.last().expect("a decoding sequence has sampled");
                    plan.push(Chunk { i, pos: l.fed, tokens: vec![last], samples: true, draft: 0, src: Src::Own });
                }
            }
            if !short {
                if self.spec {
                    avail = self.add_drafts(&mut plan, avail);
                }
                let decodes: usize = plan.iter().map(|c| c.tokens.len()).sum();
                let mut budget = self.cfg.step_tokens.saturating_sub(decodes);
                if decodes == 0 {
                    budget = budget.max(1);
                } else if self.cfg.decode_share > 0.0 && self.decode_until.is_some_and(|t| Instant::now() < t) {
                    budget = 0; // the decodes' own steps after a prompt chunk
                } else if self.cfg.mixed_step_tokens > 0 {
                    budget = budget.min(self.cfg.mixed_step_tokens);
                }
                // each prompt's tokens still to feed (0 for decodes and loads), and its chunk
                let rest: Vec<usize> = (self.running.iter().zip(&ready))
                    .map(|(l, &r)| if r { l.target.saturating_sub(l.fed) } else { 0 })
                    .collect();
                let mut take = vec![0usize; rest.len()];
                // the most tokens the pages left can hold
                let room = |i: usize, avail: usize| {
                    let fed = self.running[i].fed;
                    (self.pages(fed) + avail) * self.page - fed
                };
                if self.cfg.prefill_order == PrefillOrder::ShortFirst {
                    let reserve = if rest.iter().any(|&r| r > budget) { budget / 4 } else { 0 };
                    let mut first = budget - reserve;
                    for (i, &r) in rest.iter().enumerate() {
                        if r > 0 && r <= first && r <= room(i, avail) {
                            first -= r;
                            budget -= r;
                            let fed = self.running[i].fed;
                            avail -= self.pages(fed + r) - self.pages(fed);
                            take[i] = r;
                        }
                    }
                }
                for (i, &r) in rest.iter().enumerate() {
                    if budget == 0 {
                        break;
                    }
                    if take[i] > 0 {
                        continue;
                    }
                    let n = r.min(budget).min(room(i, avail));
                    if n == 0 {
                        continue;
                    }
                    budget -= n;
                    let fed = self.running[i].fed;
                    avail -= self.pages(fed + n) - self.pages(fed);
                    take[i] = n;
                }
                for (i, &n) in take.iter().enumerate().filter(|(_, n)| **n > 0) {
                    let l = &self.running[i];
                    let tokens = (l.fed..l.fed + n).map(|p| l.token(p)).collect();
                    plan.push(Chunk { i, pos: l.fed, tokens, samples: l.fed + n == l.target, draft: 0, src: Src::Own });
                }
                short = plan.is_empty();
            }
            if plan.is_empty() && ready.contains(&false) {
                let t = Instant::now();
                let waited = self.model.wait_copies();
                self.copy_wait_ms += t.elapsed().as_secs_f64() * 1e3;
                if let Err(e) = waited {
                    tracing::error!("{e:#}");
                    return (plan, preempted);
                }
                continue;
            }
            if !short || self.running.len() <= 1 {
                return (plan, preempted);
            }
            self.preempt_newest();
            preempted = true;
        }
    }

    /// Each decode's draft share: `spec_tokens`, or an equal part of `spec_step_tokens` past the
    /// decodes' own tokens when that is less (at least one).
    fn draft_share(&self, decodes: usize) -> usize {
        let spare = self.cfg.spec_step_tokens.saturating_sub(decodes);
        self.cfg.spec_tokens.min((spare / decodes.max(1)).max(1))
    }

    /// Drafts for the decodes: their own context's match, else the draft head's proposal, else
    /// other replies' continuation; within `spec_step_tokens` decode tokens (an equal share
    /// each), their max_tokens and context, and the pages left (returned). The chunks with
    /// drafts move behind the single-token ones, which the executor batches at the front of a
    /// step.
    fn add_drafts(&self, plan: &mut [Chunk], mut avail: usize) -> usize {
        let mut spare = self.cfg.spec_step_tokens.saturating_sub(plan.len());
        let share = self.draft_share(plan.len());
        let floor = if self.cfg.spec_by_acceptance {
            ACCEPT_FLOOR * (plan.len() as f32 / ACCEPT_BUSY as f32).min(1.0)
        } else {
            0.0
        };
        for c in plan.iter_mut() {
            if spare == 0 {
                break;
            }
            let l = &self.running[c.i];
            let Some(g) = &l.ngram else { continue };
            // the drafts and the pick after them stay within max_tokens and the context
            let left = (l.params.max_tokens.saturating_sub(l.generated.len() + 1))
                .min(self.max_context.saturating_sub(l.fed + 1));
            let want = share.min(spare).min(left);
            // the head's proposal when it was chosen for this step (`run_head`); else its own
            // context; else what other replies said after the same tokens
            let own = if self.cfg.spec_own && l.eagle.is_empty() { g.propose(want) } else { &[] };
            let src = if want == 0 {
                Src::Own
            } else if !l.eagle.is_empty() {
                Src::Eagle
            } else if !own.is_empty() {
                Src::Own
            } else {
                Src::Global
            };
            let other = match (&self.global, src) {
                (Some(gl), Src::Global) => gl.propose(g.tokens(), want),
                _ => Vec::new(),
            };
            let d = match src {
                Src::Own => &own[..own.len().min(draft_len(l.acc[0], floor))],
                Src::Eagle => &l.eagle[..l.eagle.len().min(want)],
                Src::Global => &other[..other.len().min(draft_len(l.acc[1], floor))],
            };
            // the most the pages left can hold past the chunk's own token
            let room = (self.pages(l.fed + 1) + avail) * self.page - (l.fed + 1);
            let d = &d[..d.len().min(room)];
            if d.is_empty() {
                continue;
            }
            avail -= self.pages(l.fed + 1 + d.len()) - self.pages(l.fed + 1);
            spare -= d.len();
            c.tokens.extend_from_slice(d);
            c.draft = d.len();
            c.src = src;
        }
        plan.sort_by_key(|c| c.draft > 0);
        avail
    }

    fn preempt_newest(&mut self) {
        let l = self.running.pop().expect("more than one running");
        self.model.free_seq(l.seq);
        tracing::info!(
            request = l.id,
            context = l.fed,
            generated = l.generated.len(),
            running = self.running.len(),
            "cache full: preempted the newest request"
        );
        self.waiting.push_front(Pending {
            id: l.id,
            req: Request { prompt: l.prompt, params: l.params },
            constraint: l.constraint,
            tx: l.tx,
            cancel: l.cancel,
            submitted: Instant::now(),
            generated: l.generated,
            cached: Some(l.cached),
        });
        // what the running requests really generate, plus some steps ahead
        let (made, most) =
            self.running.iter().fold((0, 0), |(m, x), r| (m + r.generated.len(), x + r.params.max_tokens));
        self.ratio = ((made + RETRACT_STEPS * self.running.len()) as f64 / (most + 1) as f64).min(1.0);
        if let Ok(mut s) = self.stats.lock() {
            s.preemptions += 1;
        }
    }

    /// Whether a running request is still prefilling at least two more pages of `prompt` than
    /// the cache holds (`cached`): once it has, they are a hit.
    fn prefilling_shared(&self, prompt: &[u32], lora: Option<usize>, cached: usize) -> bool {
        if !self.prefix_cache {
            return false;
        }
        // a hit leaves the last prompt token to compute (`prompt` keyed for `lora`)
        let usable = &prompt[..prompt.len() - 1];
        self.running.iter().any(|l| {
            l.fed < l.prompt.len() && l.params.lora == lora && {
                let common = l
                    .prompt
                    .iter()
                    .map(|&t| crate::models::lora_key(t, lora))
                    .zip(usable)
                    .take_while(|(a, b)| a == *b)
                    .count();
                let common = common / self.page * self.page;
                common >= cached + 2 * self.page && l.fed < common
            }
        })
    }

    /// One scheduling step. Returns whether any work ran.
    fn step(&mut self) -> bool {
        let t_enter = Instant::now();
        let gap_ms = self.last_end.map_or(0.0, |e| t_enter.duration_since(e).as_secs_f64() * 1e3);
        self.admit();
        if self.running.is_empty() {
            self.publish();
            return false;
        }

        let (plan, preempted) = self.plan();
        if plan.is_empty() {
            // one request alone always fits unless the cache is smaller than its next chunk
            for l in &mut self.running {
                finish(l, FinishReason::Error("the cache cannot hold this request's next step".into()));
                self.model.free_seq(l.seq);
            }
            self.running.clear();
            self.publish();
            return false;
        }
        // the longest wait from submission to this (first) step among the sequences starting now
        let mut queue_ms = 0.0f64;
        for c in &plan {
            if let Some(s) = self.running[c.i].submitted.take() {
                queue_ms = queue_ms.max(t_enter.duration_since(s).as_secs_f64() * 1e3);
            }
        }
        // positions the step's attention reads
        let ctx: usize = plan.iter().map(|c| c.pos + c.tokens.len()).sum();
        let (n_prefill, n_decode) = plan.iter().fold((0u64, 0u64), |(p, d), c| {
            if self.running[c.i].fed >= self.running[c.i].target {
                (p, d + 1)
            } else {
                (p + c.tokens.len() as u64, d)
            }
        });
        if !preempted && n_decode > 0 {
            self.ratio = (self.ratio - (self.ratio_init - self.ratio_min) / RATIO_DECAY_STEPS).max(self.ratio_min);
        }

        let t0 = Instant::now();
        let mut mask_ms = 0.0;
        let batch: Vec<SeqChunk> = plan
            .iter()
            .map(|c| SeqChunk {
                seq: self.running[c.i].seq,
                pos: c.pos,
                tokens: &c.tokens,
                all_logits: c.draft > 0,
                draft: c.draft,
            })
            .collect();
        // Greedy steps take only each row's argmax from the model (found on the device when the
        // backend can); any sampled request needs the logits.
        let constrained = plan.iter().any(|c| c.samples && self.running[c.i].constraint.is_some());
        // penalties, logit bias, min_tokens, log-probabilities: the sampler's rows, not argmax
        let special = plan.iter().any(|c| {
            let p = &self.running[c.i].params;
            c.samples && (p.adjusts() || p.logprobs.is_some())
        });
        let greedy =
            !constrained && !special && plan.iter().all(|c| !c.samples || self.running[c.i].params.temperature <= 0.0);
        // Structured output: each constrained chunk's allowed tokens for its sampled row, a
        // bitmask (mask_of[chunk] = k: masks' k-1-th; 0: none), worked out in parallel across
        // requests. The device draws within them (or the host, masking the logits).
        let words = self.model.vocab().div_ceil(32);
        let mut mask_of = vec![0i32; plan.len()];
        let mut masks: Vec<u32> = Vec::new();
        if constrained {
            let t = Instant::now();
            let which: Vec<usize> = (0..plan.len())
                .filter(|&ci| plan[ci].samples && self.running[plan[ci].i].constraint.is_some())
                .collect();
            let mut by_request: Vec<Option<&mut Box<dyn Constraint>>> =
                self.running.iter_mut().map(|l| l.constraint.as_mut()).collect();
            let mut cons: Vec<&mut Box<dyn Constraint>> =
                which.iter().map(|&ci| by_request[plan[ci].i].take().expect("one chunk a request")).collect();
            masks.resize(cons.len() * words, 0);
            use rayon::prelude::*;
            let active: Vec<bool> =
                cons.par_iter_mut().zip(masks.par_chunks_mut(words)).map(|(g, w)| g.allowed(w)).collect();
            for (k, (&ci, on)) in which.iter().zip(active).enumerate() {
                debug_assert_eq!(plan[ci].draft, 0, "constrained requests take no drafts");
                if on {
                    mask_of[ci] = k as i32 + 1;
                }
            }
            mask_ms = t.elapsed().as_secs_f64() * 1e3;
        }
        // each row's draw as the sampler would make it, when the device samples: its request's
        // settings, and the index of the token it yields (a chunk with drafts yields one per row)
        // (the host sampler reads the same rows: their adjustments and log-probability counts)
        let mut adj: Vec<LogitAdj> = Vec::new();
        let vocab = self.model.vocab();
        let mut want_lp = false;
        let srows: Vec<SampleRow> = if greedy {
            Vec::new()
        } else {
            let mut rows = Vec::with_capacity(batch.len());
            for (ci, c) in plan.iter().enumerate() {
                let l = &self.running[c.i];
                let (p, t) = (&l.params, if c.samples { l.params.temperature } else { 0.0 });
                for r in 0..if c.draft > 0 { c.tokens.len() } else { 1 } {
                    let mut row = SampleRow::new(t, p.top_p, p.top_k, p.seed, (l.generated.len() + r) as u64);
                    row.mask = mask_of[ci];
                    if c.samples {
                        row.min_p_cut = p.min_p_cut();
                        if let Some(k) = p.logprobs {
                            row.logprobs = 1 + k.min(MAX_TOP_LOGPROBS) as i32;
                            want_lp = true;
                        }
                        if p.adjusts() {
                            debug_assert_eq!(c.draft, 0, "requests with penalties take no drafts");
                            let off = adj.len();
                            adjustments(p, &l.prompt_set, &l.counts, l.generated.len() + r, vocab, &mut adj);
                            (row.adj_off, row.adj_n) = (off as i32, (adj.len() - off) as i32);
                        }
                    }
                    rows.push(row);
                }
            }
            rows
        };
        let result = if greedy {
            self.model.forward_greedy(&batch).map(Picks::Tokens)
        } else if self.device_sampling {
            let n = srows.len();
            if want_lp {
                self.lp_buf.resize(n * (MAX_TOP_LOGPROBS + 1), 0.0);
                self.lp_ids.resize(n * MAX_TOP_LOGPROBS, 0);
            }
            let mut io = SampleIo {
                masks: &masks,
                mask_words: words,
                adj: &adj,
                logprobs: want_lp.then(|| (&mut self.lp_buf[..], &mut self.lp_ids[..])),
            };
            self.model.forward_sample(&batch, &srows, &mut io).map(Picks::Tokens)
        } else {
            self.model.forward_flat(&batch, &mut self.logits).map(|_| Picks::Logits)
        };
        drop(batch);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let t_post = Instant::now();

        let mut failed = 0;
        let spec_drafted: u64 = plan.iter().map(|c| c.draft as u64).sum();
        let drafted_from = |s: Src| plan.iter().filter(|c| c.src == s).map(|c| c.draft as u64).sum::<u64>();
        let (spec_drafted_global, spec_drafted_eagle) = (drafted_from(Src::Global), drafted_from(Src::Eagle));
        let spec_chunks = plan.iter().filter(|c| c.draft > 0).count() as u64;
        let spec_chunks_eagle = plan.iter().filter(|c| c.draft > 0 && c.src == Src::Eagle).count() as u64;
        let (mut spec_accepted, mut spec_accepted_global, mut spec_accepted_eagle) = (0u64, 0u64, 0u64);
        // the token after each chunk's last kept position (for the draft head); None: failed
        let mut after: Vec<Option<u32>> = vec![None; plan.len()];
        // for the trace: request:drafted:kept of each chunk with drafts
        let mut spec_pairs = String::new();
        match result {
            Err(e) => {
                let msg = format!("{e:#}");
                tracing::error!("step failed, failing {} request(s): {msg}", plan.len());
                for c in &plan {
                    let l = &mut self.running[c.i];
                    finish(l, FinishReason::Error(msg.clone()));
                    failed += 1;
                }
            }
            Ok(picks) => {
                let vocab = self.model.vocab();
                // one pick or logits row per row: every row of a chunk with drafts, the last of others
                let mut row = 0;
                for (ci, c) in plan.iter().enumerate() {
                    let max_ctx = self.max_context;
                    let rows = if c.draft > 0 { c.tokens.len() } else { 1 };
                    let r0 = row;
                    row += rows;
                    let l = &mut self.running[c.i];
                    if c.draft == 0 {
                        l.fed += c.tokens.len();
                        if !c.samples {
                            after[ci] = Some(l.token(l.fed));
                            continue;
                        }
                        let (tok, lp) = match &picks {
                            Picks::Tokens(t) => (t[r0], device_logprobs(&self.lp_buf, &self.lp_ids, &srows, r0)),
                            Picks::Logits => {
                                let row = &mut self.logits[r0 * vocab..(r0 + 1) * vocab];
                                // log-probabilities are of the logits as the model gave them
                                let raw = l.params.logprobs.map(|_| row.to_vec());
                                if let Some(k) = (mask_of[ci] as usize).checked_sub(1) {
                                    apply_mask(&masks[k * words..(k + 1) * words], row);
                                }
                                let a = &srows[r0];
                                for x in &adj[a.adj_off as usize..(a.adj_off + a.adj_n) as usize] {
                                    row[x.token as usize] = x.apply(row[x.token as usize]);
                                }
                                let tok = sample(row, &l.params, l.generated.len());
                                (tok, raw.zip(l.params.logprobs).map(|(raw, k)| logprobs(&raw, tok, k)))
                            }
                        };
                        after[ci] = Some(tok);
                        if let Some(g) = &mut l.constraint {
                            g.advance(tok);
                        }
                        emit(l, &[tok], lp.as_slice(), max_ctx);
                        if !l.done && l.constraint.as_ref().is_some_and(|g| g.done()) {
                            finish(l, FinishReason::Stop);
                        }
                        continue;
                    }
                    // the token the request generates after each of the chunk's tokens, drawn for
                    // its position as without drafts: keep the drafts up to the first that differs
                    // from its draw, then that draw (the rows past it are never drawn)
                    let drafts = &c.tokens[1..];
                    let mut n = 0;
                    let mut lps: Vec<TokenLogprobs> = Vec::new();
                    let next = loop {
                        let r = r0 + n;
                        let (tok, lp) = match &picks {
                            Picks::Tokens(t) => (t[r], device_logprobs(&self.lp_buf, &self.lp_ids, &srows, r)),
                            Picks::Logits => {
                                let row = &self.logits[r * vocab..(r + 1) * vocab];
                                let tok = sample(row, &l.params, l.generated.len() + n);
                                (tok, l.params.logprobs.map(|k| logprobs(row, tok, k)))
                            }
                        };
                        lps.extend(lp);
                        if n < drafts.len() && tok == drafts[n] {
                            n += 1;
                        } else {
                            break tok;
                        }
                    };
                    l.fed += 1 + n;
                    spec_accepted += n as u64;
                    if self.trace {
                        use std::fmt::Write;
                        let _ = write!(spec_pairs, "{}:{}:{},", l.id, c.draft, n);
                    }
                    let source = match c.src {
                        Src::Global => {
                            spec_accepted_global += n as u64;
                            None
                        }
                        Src::Eagle => {
                            spec_accepted_eagle += n as u64;
                            Some(1)
                        }
                        Src::Own => Some(0),
                    };
                    if let Some(x) = source {
                        l.kept[x] += KEPT_ALPHA * (n as f32 - l.kept[x]);
                    }
                    if c.src != Src::Eagle {
                        let a = &mut l.acc[(c.src == Src::Global) as usize];
                        *a = [ACCEPT_DECAY * a[0] + n as f32, ACCEPT_DECAY * a[1] + (n < c.draft) as u8 as f32];
                    }
                    if let Err(e) = self.model.accept(l.seq, n) {
                        tracing::error!(request = l.id, "{e:#}");
                        finish(l, FinishReason::Error(format!("{e:#}")));
                        failed += 1;
                        continue;
                    }
                    let mut toks = drafts[..n].to_vec();
                    toks.push(next);
                    after[ci] = Some(next);
                    emit(l, &toks, &lps, max_ctx);
                }
            }
        }
        let t_draft = Instant::now();
        if self.eagle {
            self.run_head(&plan, &after);
        }
        let draft_ms = t_draft.elapsed().as_secs_f64() * 1e3;

        let mut finished = 0;
        for l in self.running.iter().filter(|l| l.done) {
            self.model.free_seq(l.seq);
            if let Some(g) = &mut self.global {
                g.add(&l.prompt, &l.generated);
            }
            finished += 1;
        }
        self.running.retain(|l| !l.done);
        if self.running.is_empty() {
            self.ratio = self.ratio_init;
        }
        if let Ok(mut s) = self.stats.lock() {
            s.steps += 1;
            s.prefill_tokens += n_prefill;
            s.decode_tokens += n_decode;
            s.spec_drafted += spec_drafted;
            s.spec_accepted += spec_accepted;
            s.spec_drafted_global += spec_drafted_global;
            s.spec_accepted_global += spec_accepted_global;
            s.spec_drafted_eagle += spec_drafted_eagle;
            s.spec_accepted_eagle += spec_accepted_eagle;
            s.spec_chunks += spec_chunks;
            s.spec_chunks_eagle += spec_chunks_eagle;
            s.step_ms_ema = if s.steps == 1 { ms } else { 0.9 * s.step_ms_ema + 0.1 * ms };
        }
        if self.cfg.decode_share > 0.0 && n_prefill > 0 && n_decode > 0 {
            let s = self.cfg.decode_share.min(0.95) as f64;
            self.decode_until = Some(Instant::now() + std::time::Duration::from_secs_f64(ms * 1e-3 * s / (1.0 - s)));
        }
        if let Ok(mut s) = self.stats.lock() {
            s.finished += finished - failed.min(finished);
            s.failed += failed;
        }
        self.publish();
        if self.trace {
            let ex = self.model.last_step();
            let (ex_total, ex_up, ex_down) = ex.map_or((0.0, 0.0, 0.0), |s| (s.ms_total, s.ms_upload, s.ms_download));
            let plan_ms = t0.duration_since(t_enter).as_secs_f64() * 1e3;
            let post_ms = t_post.elapsed().as_secs_f64() * 1e3;
            tracing::info!(
                "step gap_ms={gap_ms:.3} plan_ms={plan_ms:.3} mask_ms={mask_ms:.3} fwd_ms={ms:.3} draft_ms={draft_ms:.3} post_ms={post_ms:.3} \
                 ex_total={ex_total:.3} ex_up={ex_up:.3} ex_down={ex_down:.3} prefill={n_prefill} decode={n_decode} \
                 drafted={spec_drafted} accepted={spec_accepted} \
                 queue_ms={queue_ms:.3} ctx={ctx} running={} waiting={} loading={} copy_wait_ms={:.3} free={} head_need={} pairs={spec_pairs}",
                self.running.len(),
                self.waiting.len(),
                self.loading,
                self.copy_wait_ms,
                self.model.capacity().free_cache_tokens,
                self.head_need
            );
        }
        self.last_end = Some(Instant::now());
        true
    }

    /// The draft head after a step: it reads every chunk's kept positions (`after`: the token
    /// after each chunk's last, None where the chunk failed) and drafts for the decoding requests
    /// it is chosen for (their next step's share, within max_tokens and the context): those with
    /// no n-gram match, and per `SpecChoice` those with one. A failed call leaves no drafts; the
    /// requests go on without.
    fn run_head(&mut self, plan: &[Chunk], after: &[Option<u32>]) {
        let decodes = self.running.iter().filter(|l| !l.done && l.fed >= l.target).count();
        let ngram_share = self.draft_share(decodes);
        let share = self.cfg.eagle_tokens.min(ngram_share);
        // trials: a decode step without a source drafting counts towards its next
        for c in plan {
            let l = &mut self.running[c.i];
            if c.pos < l.prompt.len() {
                continue;
            }
            for (x, src) in [(0, Src::Own), (1, Src::Eagle)] {
                if c.draft > 0 && c.src == src {
                    l.since[x] = 0;
                } else {
                    l.since[x] = l.since[x].saturating_add(1);
                }
            }
        }
        let mut reqs = Vec::with_capacity(plan.len());
        let mut who = Vec::with_capacity(plan.len());
        for (c, next) in plan.iter().zip(after) {
            let Some(next) = *next else { continue };
            let l = &mut self.running[c.i];
            l.eagle.clear();
            let k = if l.done
                || l.fed < l.target
                || self.eagle_consume_only
                || l.constraint.is_some()
                || l.params.adjusts()
            {
                0
            } else {
                let left = (l.params.max_tokens.saturating_sub(l.generated.len() + 1))
                    .min(self.max_context.saturating_sub(l.fed + 1));
                let k = share.min(left);
                let matched = match (&l.ngram, self.cfg.spec_own) {
                    (Some(g), true) => g.propose(ngram_share.min(left)).len(),
                    _ => 0,
                };
                let head = matched == 0
                    || match self.cfg.spec_choice {
                        SpecChoice::NgramFirst => false,
                        SpecChoice::Adaptive => head_drafts(l.kept, l.since, matched),
                    };
                if head {
                    k
                } else {
                    0
                }
            };
            reqs.push(DraftReq { seq: l.seq, next, k });
            who.push(c.i);
        }
        match self.model.draft(&reqs) {
            Ok(out) => {
                for (i, d) in who.into_iter().zip(out) {
                    self.running[i].eagle = d;
                }
            }
            Err(e) => tracing::error!("draft head: {e:#}"),
        }
    }

    fn publish(&self) {
        if let Ok(mut s) = self.stats.lock() {
            s.running = self.running.len();
            s.waiting = self.waiting.len();
            s.output_ratio = if self.admission == Admission::WorstCase { 1.0 } else { self.ratio };
            s.capacity = self.model.capacity();
        }
    }
}

/// Adaptive choice for a request whose context has a match of `matched` tokens: whether the head
/// drafts its next step. Trials first (the match's before the head's), then the better average of
/// drafts kept per step (the match's capped by its length).
fn head_drafts(kept: [f32; 2], since: [u32; 2], matched: usize) -> bool {
    if since[0] >= TRIAL_EVERY {
        return false;
    }
    since[1] >= TRIAL_EVERY || kept[1] > kept[0].min(matched as f32)
}

fn finish(l: &mut Live, reason: FinishReason) {
    let usage = Usage { prompt_tokens: l.prompt.len(), completion_tokens: l.generated.len(), cached_tokens: l.cached };
    let _ = l.tx.send(Event::Finished { reason, usage });
    l.done = true;
}

/// Send the tokens a step produced for `l`, in order, until one ends it (a cancel, a stop id,
/// max_tokens, a client gone); then, if it goes on, the context limit.
/// `lps`: each token's log-probabilities when the request asks for them (else empty).
fn emit(l: &mut Live, toks: &[u32], lps: &[TokenLogprobs], max_ctx: usize) {
    for (i, &tok) in toks.iter().enumerate() {
        if l.cancel.is_set() {
            finish(l, FinishReason::Stop); // its token is not sent, nor counted
            return;
        }
        l.generated.push(tok);
        if l.params.adjusts() {
            *l.counts.entry(tok).or_insert(0) += 1;
        }
        if let Some(g) = &mut l.ngram {
            g.push(&[tok]);
        }
        if l.params.stop_token_ids.contains(&tok) {
            finish(l, FinishReason::Stop);
            return;
        }
        if let Some(lp) = lps.get(i) {
            let _ = l.tx.send(Event::Logprobs(lp.clone()));
        }
        if l.tx.send(Event::Token(tok)).is_err() {
            l.done = true; // client went away
            return;
        }
        if l.generated.len() >= l.params.max_tokens {
            finish(l, FinishReason::Length);
            return;
        }
    }
    if l.fed + 1 > max_ctx {
        finish(l, FinishReason::ContextLimit);
    }
}

/// What a step returned: each row's greedy token, or the rows' logits (in `Engine::logits`) for
/// the sampler.
/// Row `r`'s log-probabilities from the device's buffers, when it asked for them.
fn device_logprobs(lp: &[f32], ids: &[i32], rows: &[SampleRow], r: usize) -> Option<TokenLogprobs> {
    let k = rows.get(r)?.logprobs;
    if k <= 0 {
        return None;
    }
    let (l, i) = (&lp[r * (MAX_TOP_LOGPROBS + 1)..], &ids[r * MAX_TOP_LOGPROBS..]);
    Some(TokenLogprobs { logprob: l[0], top: (0..k as usize - 1).map(|j| (i[j] as u32, l[1 + j])).collect() })
}

/// `logits` (a vocabulary row) with the tokens `words` (`Constraint::allowed`) does not allow at
/// -inf: 32 tokens a word, a word of none setting its run at once (most of a grammar's words).
fn apply_mask(words: &[u32], logits: &mut [f32]) {
    for (w, chunk) in words.iter().zip(logits.chunks_mut(32)) {
        match *w {
            u32::MAX => {}
            0 => chunk.fill(f32::NEG_INFINITY),
            bits => {
                for (b, l) in chunk.iter_mut().enumerate() {
                    if bits >> b & 1 == 0 {
                        *l = f32::NEG_INFINITY;
                    }
                }
            }
        }
    }
}

enum Picks {
    Tokens(Vec<u32>),
    Logits,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_lengths_follow_acceptance() {
        // a new request drafts a little, more with every draft kept, the full share after two steps
        assert_eq!(draft_len([0.0, 0.0], ACCEPT_FLOOR), 2);
        let mut acc = [0.0f32; 2];
        let mut step = |k: usize, n: usize| {
            acc = [ACCEPT_DECAY * acc[0] + n as f32, ACCEPT_DECAY * acc[1] + (n < k) as u8 as f32];
            draft_len(acc, ACCEPT_FLOOR)
        };
        let k = step(2, 2);
        assert!(k >= 4, "{k} drafts after 2 of 2 kept");
        assert!(step(k.min(8), k.min(8)) >= 8);
        for _ in 0..10 {
            step(8, 8);
        }
        // a copying request that starts writing: back to one draft within a few steps, never none
        let ks: Vec<usize> = (0..12).map(|_| step(8, 0)).collect();
        assert!(ks[0] >= 8 && ks[11] == 1 && ks.windows(2).all(|w| w[1] <= w[0]), "{ks:?}");
        // and back up once drafts hold again
        assert!(step(1, 1) >= 1);
        for _ in 0..6 {
            step(4, 4);
        }
        assert!(step(8, 8) >= 8);
        // fresh text (drafts rarely kept) stays at one; few decodes (a lower floor) draft longer;
        // no floor: the full share
        assert_eq!(draft_len([0.3, 3.0], ACCEPT_FLOOR), 1);
        assert!(draft_len([2.0, 2.0], ACCEPT_FLOOR / 8.0) > draft_len([2.0, 2.0], ACCEPT_FLOOR));
        assert_eq!(draft_len([0.0, 5.0], 0.0), usize::MAX);
    }

    #[test]
    fn adaptive_choice_tries_both_sources_then_keeps_the_better() {
        let t = TRIAL_EVERY;
        // a request's first match is tried, however much better the head looks
        assert!(!head_drafts([0.1, 3.0], [t, 0], 8));
        // the better average, the match's capped by its length
        assert!(!head_drafts([3.0, 1.5], [0, 3], 8));
        assert!(head_drafts([3.0, 1.5], [0, 3], 1));
        assert!(head_drafts([0.4, 1.0], [0, 3], 8));
        // a source that has not drafted for TRIAL_EVERY steps gets a trial: the match's first
        assert!(head_drafts([3.0, 0.2], [2, t], 8));
        assert!(!head_drafts([0.1, 3.0], [t, 2], 8));
        assert!(!head_drafts([0.1, 3.0], [t + 5, t + 5], 8));
    }
}
