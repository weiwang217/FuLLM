// SPDX-License-Identifier: Apache-2.0
//! `fullm` command line: serve, generate, plan, info, bench, selftest.

use std::io::Write;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};

use fullm::models::{self, BackendKind, CacheDtype, LoadOptions};
use fullm::runtime::{Engine, EngineConfig, Event, Request, SamplingParams};
use fullm::server::{self, AppState};
use fullm::tensor::Precision;
use fullm::tokenizer::{load_tokenizer_with, Detokenizer, Tokenizer};

#[derive(Parser)]
#[command(name = "fullm", version, about = "A single-GPU LLM inference engine")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Serve the OpenAI-compatible HTTP API.
    Serve(ServeArgs),
    /// Generate from one prompt and report throughput.
    Generate(GenerateArgs),
    /// Plan a deployment: placement, memory and expected speed.
    Plan(fullm::planner::PlanArgs),
    /// Load a model and describe where everything lives.
    Info(ModelArgs),
    /// The executor backends of this build: whether each can run here, and what it can do.
    Backends {
        /// The device to ask each backend about.
        #[arg(long, default_value_t = 0)]
        device: i32,
    },
    /// Measure this machine (and optionally a model): copy, GEMV, expert and DRAM rates,
    /// end-to-end prefill and decode.
    Bench(fullm::bench::BenchArgs),
    /// Run one prompt through two backends and compare logits (e.g. reference vs cuda on
    /// the first N layers of the real checkpoint).
    Compare(fullm::bench::CompareArgs),
    /// Teacher-forced perplexity of a text file.
    Ppl(fullm::bench::PplArgs),
    /// Print the token ids of a text (or of a one-message chat) under a model's tokenizer.
    Tokenize {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        text: String,
        /// Render `text` with the chat template instead: a JSON array of messages, or else the
        /// text of one user message.
        #[arg(long)]
        chat: bool,
        /// With --chat: thinking off (Qwen3's enable_thinking=false).
        #[arg(long)]
        no_thinking: bool,
        /// With --chat: this Jinja chat template (a file) instead of the model's.
        #[arg(long)]
        chat_template: Option<PathBuf>,
    },
    /// The conformance suite: every op of a backend against a reference backend (the CPU's) on
    /// random inputs, skipping the model families and KV formats the backend does not declare.
    Selftest {
        #[arg(long, default_value_t = 0)]
        device: i32,
        /// The backend under test (default: auto, the GPU when there is one).
        #[arg(long)]
        backend: Option<BackendKind>,
        /// The backend it is compared with.
        #[arg(long, default_value = "cpu")]
        reference: String,
    },
    /// Time single CUDA ops on synthetic data of real shapes (no model): cases separated by ';',
    /// e.g. "attn n=16 ctx=8000 kv=fp8; moe n=16 fmt=int4" (see crates/fullm-kernels/csrc/exec/cuda/kbench.cu).
    Kbench {
        /// the cases (empty: a default set)
        #[arg(default_value = "")]
        spec: String,
        #[arg(long, default_value_t = 0)]
        device: i32,
    },
}

#[derive(Args, Clone)]
struct ModelArgs {
    /// Checkpoint directory (config.json + *.safetensors).
    #[arg(long)]
    model: PathBuf,
    /// Storage of the large trunk matrices: bf16 (exact), int8 or fp8.
    #[arg(long, default_value = "bf16", value_parser = parse_precision)]
    trunk: Precision,
    /// Storage of the lm_head: bf16 (default) or int8 / fp8. A tied head is otherwise a bf16 copy
    /// of the embedding table (Qwen3-4B: 0.78 GB, 18% of a decode step's bytes on an 8 GB GPU).
    #[arg(long, value_parser = parse_precision)]
    head: Option<Precision>,
    /// Concurrent sequences.
    #[arg(long, default_value_t = 8)]
    max_seqs: usize,
    /// Cached positions across all sequences.
    #[arg(long, default_value_t = 16 * 1024)]
    cache_tokens: usize,
    /// Positions per KV page.
    #[arg(long, default_value_t = 16)]
    page: usize,
    /// Tokens per engine step (decodes plus prefill chunks). Larger steps prefill long prompts
    /// in fewer, longer steps (vLLM's default is 2048, 8192 on 80 GB GPUs); the executor's
    /// workspace grows with it, which comes out of the budget for resident experts.
    #[arg(long, default_value_t = 2048)]
    step_tokens: usize,
    /// How admission counts a request's output still to come: auto (optimistic with the prefix
    /// cache, worst-case otherwise), optimistic (estimated as SGLang does; a step that runs out of
    /// cache preempts the newest request, which resumes from its cached pages) or worst-case (all
    /// of max_tokens: never preempts).
    #[arg(long, value_enum, default_value = "auto")]
    admission: fullm::runtime::Admission,
    /// Scales the optimistic admission's output estimate (SGLang's flag of the same name): above 1
    /// admits fewer requests and preempts less, below 1 the reverse.
    #[arg(long, default_value_t = 1.0)]
    schedule_conservativeness: f64,
    /// Which prompts each step's prefill budget goes to first: short-first (prompts that finish
    /// within the step, so a short one does not wait out a long one's prefill; a long one keeps a
    /// quarter of every step) or arrival (admission order, as vLLM and SGLang).
    #[arg(long, value_enum, default_value = "short-first")]
    prefill_order: fullm::runtime::PrefillOrder,
    /// While requests decode, the share of the time (0 to 0.95) kept for steps without prompt
    /// chunks: after a step that carried another request's prompt, the decodes step alone for
    /// share / (1 - share) times as long. 0 (the default): a prompt chunk in every step, right
    /// where a step's time barely grows with its tokens (an H100). Where it grows with them
    /// (Inkling 2 on a Mac: 66 ms for a token alone, 4.3 s beside a 1024-token chunk) a stream
    /// otherwise gets one token per chunk; 0.5 with --mixed-step-tokens 256 keeps it at half
    /// speed with pauses of ~1.5 s, and the prompt takes twice as long.
    #[arg(long, default_value_t = 0.0)]
    decode_share: f32,
    /// While requests decode, the most prompt tokens a step takes (0: the step's whole budget).
    /// Smaller chunks pause the streams for less and prefill less efficiently.
    #[arg(long, default_value_t = 0)]
    mixed_step_tokens: usize,
    /// Speculative decoding: the most draft tokens a decoding request verifies per step (0:
    /// off). Drafts are the tokens that followed an earlier occurrence of the context's last
    /// n tokens (n-gram lookup, no draft model); a step keeps those the model agrees with. On an
    /// H100, 8 beats 4 by 20-47% on replies that copy their context and costs 2-4% where little
    /// is copied; many decodes share --spec-step-tokens, so each then gets fewer. Default: 8 on a
    /// GPU backend, where a step is bandwidth-bound and verifying drafts is nearly free; 0 on a
    /// CPU backend (reference, cpu, cpu-core, torch), where a step is compute-bound and each draft
    /// costs a token's compute (Qwen3-0.6B on an M3 Pro: 61.0 tok/s per stream without drafts,
    /// 48.8 with); 8 on either with --spec-model.
    #[arg(long)]
    spec_tokens: Option<usize>,
    /// Drafts go in only while a step's decode tokens, drafts included, stay within this, shared
    /// equally by the decodes (32 decodes: 7 drafts each, within --spec-tokens). On an H100 big
    /// steps still pay: agent users at 32 (n-gram drafts) went 1805 -> 3082 tok/s on Qwen3-8B
    /// and 2346 -> 3330 on Qwen3-30B-A3B from 80 to 256.
    #[arg(long, default_value_t = 256)]
    spec_step_tokens: usize,
    /// Every n-gram match drafts its full share. By default a request's draft length follows how
    /// its recent drafts held (short after rejected ones), which keeps drafts from slowing fresh
    /// text at concurrency.
    #[arg(long)]
    spec_fixed: bool,
    /// Speculation only from a request's own context: no drafts from what other requests' replies
    /// said after the same tokens.
    #[arg(long)]
    no_spec_global: bool,
    /// The n-gram lengths speculation matches, longest first: MIN:MAX.
    #[arg(long, default_value = "2:4", value_parser = parse_span)]
    spec_ngram: (usize, usize),
    /// An EAGLE-3 draft head for a model with GQA attention, Qwen3 dense or MoE (a checkpoint
    /// directory, e.g. a head trained for
    /// it such as AngelSlim/Qwen3-a3B_eagle3; fullm only runs it, the head is trained
    /// elsewhere). Off by default. Not recommended for agent workloads: their replies copy their
    /// context, n-gram drafts already keep most of what they propose, and the head's own work
    /// each step made them 6-8% slower (H100, 16 users). It helps thinking and freshly written
    /// text at low concurrency (+9-16%).
    #[arg(long)]
    spec_model: Option<PathBuf>,
    /// The most tokens the draft head proposes for a request per step (within --spec-tokens and
    /// its share of --spec-step-tokens).
    #[arg(long, default_value_t = 4)]
    spec_eagle_tokens: usize,
    /// No drafts from a request's own context (n-gram matches): only the draft head's and other
    /// replies' (measuring the head alone).
    #[arg(long)]
    no_spec_ngram: bool,
    /// With a draft head, how a request whose context has an n-gram match picks its drafts:
    /// adaptive (whichever source has kept more tokens per step for it lately, the head tried
    /// every 16 steps) or ngram-first (the match always).
    #[arg(long, value_enum, default_value = "adaptive")]
    spec_choice: fullm::runtime::SpecChoice,
    /// reference (Rust CPU oracle), auto, or a backend of this build by name (`fullm backends`:
    /// cpu, cuda, ...). Default: auto, the first usable backend with memory of its own, else cpu.
    #[arg(long)]
    backend: Option<BackendKind>,
    /// CUDA device ordinal.
    #[arg(long, default_value_t = 0)]
    device: i32,
    /// Recurrent-state storage (auto = model dtype, bf16; as vLLM).
    #[arg(long, value_enum, default_value = "auto")]
    state_dtype: CacheDtype,
    /// Attention / latent cache storage (auto = bf16; fp8 = the backend's best fp8 cache, half the
    /// bytes, changes numerics: FlashAttention-3's e4m3 pages on CUDA; fp8-token = e4m3 with a
    /// scale per token and kv head; int4 = 0.64 bytes a value, for when the cache is what limits
    /// the server: 1.8x fp8's throughput once fp8's positions no longer hold the sessions,
    /// 15-20% slower than fp8 while they do (agent sessions with drafts); `fullm backends` lists
    /// each backend's caches).
    #[arg(long, value_enum, default_value = "auto")]
    kv_dtype: CacheDtype,
    /// Longest sequence in positions (capped at --cache-tokens: one sequence can hold the whole
    /// KV cache, no more).
    #[arg(long, default_value_t = 32 * 1024)]
    max_context: usize,
    /// CPU threads for routed experts (0 = all cores).
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Bind only the first N layers (debugging real weights from a partial download).
    #[arg(long)]
    layers: Option<usize>,
    /// MoE combining weights from score + bias (the released Kimi Linear HF code) instead of
    /// the unbiased score (DeepSeek-V3, Kimi K3, vLLM).
    #[arg(long)]
    router_weights_biased: bool,
    /// Routed experts resident in GPU memory (the HBM tier): off, auto (as many as fit after
    /// the trunk, state and cache) or a size in GB. Applies to bf16-expert models.
    #[arg(long, default_value = "auto", value_parser = fullm::models::parse_gpu_experts)]
    gpu_experts: fullm::models::GpuExperts,
    /// Storage of GPU-resident experts: bf16 (exact) or int8 / fp8 (half the memory), fp8b,
    /// int4. A dense model with int4: its MLP as one int4 expert (the attention at --trunk).
    #[arg(long, default_value = "bf16", value_parser = parse_precision)]
    gpu_expert_format: Precision,
    /// With --gpu-expert-format int4: take the routed experts from this GPTQ checkpoint (4 bits,
    /// symmetric, groups of 128, no act-order, e.g. Qwen/Qwen3-30B-A3B-GPTQ-Int4) instead of
    /// rounding the model's bf16 experts; the rest of the model still comes from --model.
    #[arg(long)]
    int4_checkpoint: Option<PathBuf>,
    /// A Jinja chat template (a file) for the chat endpoint instead of the model's
    /// (tokenizer_config.json's chat_template, or chat_template.jinja).
    #[arg(long)]
    chat_template: Option<PathBuf>,
    /// A LoRA adapter (PEFT's format: adapter_config.json + adapter_model.safetensors) merged into
    /// the weights at load: one fine-tune served at the base model's speed (before any
    /// --trunk / --gpu-expert-format rounding).
    #[arg(long)]
    lora_merge: Option<PathBuf>,
    /// A LoRA adapter served unmerged, beside the base model and other adapters: NAME=DIR
    /// (PEFT's format), repeatable. A request picks it by `model: NAME`.
    #[arg(long = "lora")]
    loras: Vec<String>,
    /// LoRA adapter slots (default: the --lora count). More leave room for
    /// /v1/load_lora_adapter.
    #[arg(long, default_value_t = 0)]
    max_loras: usize,
    /// The largest LoRA rank a slot holds (default: the largest --lora's, else 64).
    #[arg(long, default_value_t = 0)]
    max_lora_rank: usize,
    /// Run every step op by op instead of replaying CUDA graphs of small steps.
    #[arg(long)]
    no_cuda_graphs: bool,
    /// Free each request's cache pages when it ends instead of keeping them for later requests
    /// that start with the same tokens (the prefix cache; dense GQA models).
    #[arg(long)]
    no_prefix_cache: bool,
    /// Back the prefix cache with this many positions of pinned host memory (more than
    /// --cache-tokens; 0 = off): pages evicted from the GPU stay in host memory and load back
    /// over PCIe when a request starts with them, instead of being computed again.
    #[arg(long, default_value_t = 0)]
    host_cache_tokens: usize,
    /// Activations of small-batch GEMVs: bf16 (as vLLM; faster) or exact (fp32).
    #[arg(long, value_enum, default_value = "bf16")]
    activations: fullm::models::Activations,
}

#[derive(Args)]
struct ServeArgs {
    #[command(flatten)]
    model: ModelArgs,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    /// Name reported by /v1/models; defaults to the directory name.
    #[arg(long)]
    served_name: Option<String>,
    /// Parse control-token spellings (`<|im_start|>` ...) in /v1/completions text prompts, as vLLM
    /// does. Off by default: completion text is ordinary text, so a prompt assembled from user
    /// input cannot carry control tokens; token-id prompts need no flag. Chat messages through a
    /// Jinja chat template (Qwen3, Llama) are tokenized after rendering, as vLLM and SGLang do,
    /// so a spelling in message content becomes the control token there; Kimi's chat keeps message
    /// text ordinary.
    #[arg(long)]
    special_tokens_in_prompts: bool,
    /// Serve vLLM's /v1/load_lora_adapter and /v1/unload_lora_adapter: any client may then have
    /// the server read an adapter from any path it can read (off: 403; vLLM keeps them off unless
    /// VLLM_ALLOW_RUNTIME_LORA_UPDATING is set).
    #[arg(long)]
    allow_runtime_lora: bool,
    /// Require `Authorization: Bearer KEY` on every route but /health (as vLLM's --api-key);
    /// default: the FULLM_API_KEY environment variable, else none.
    #[arg(long)]
    api_key: Option<String>,
    /// Sampling values a request leaves unset: `auto` takes the model's generation_config.json (as
    /// vLLM does), `none` OpenAI's (temperature 1, no top-p / top-k cut).
    #[arg(long, default_value = "auto", value_parser = ["auto", "none"])]
    generation_config: String,
}

#[derive(Args)]
struct GenerateArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// Prompt text (encoded with the model's tokenizer).
    #[arg(long, conflicts_with = "ids")]
    prompt: Option<String>,
    /// Prompt as comma-separated token ids.
    #[arg(long, value_delimiter = ',')]
    ids: Option<Vec<u32>>,
    #[arg(long, default_value_t = 64)]
    max_tokens: usize,
    /// 0 is greedy.
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    #[arg(long, default_value_t = 0)]
    top_k: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Keep going past the tokenizer's end-of-turn ids.
    #[arg(long)]
    ignore_eos: bool,
}

fn parse_precision(s: &str) -> Result<Precision, String> {
    match s {
        "bf16" => Ok(Precision::Bf16),
        "int8" => Ok(Precision::Int8),
        "fp8" => Ok(Precision::Fp8),
        "fp8b" => Ok(Precision::Fp8Block),
        "int4" => Ok(Precision::Int4Group),
        "f32" => Ok(Precision::F32),
        _ => Err(format!("unknown precision {s:?} (bf16, int8, fp8, fp8b, int4, f32)")),
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "fullm=info".into()))
        .with_writer(std::io::stderr)
        .init();
    match Cli::parse().cmd {
        Cmd::Serve(a) => serve(a),
        Cmd::Generate(a) => generate(a),
        Cmd::Plan(a) => fullm::planner::run(&a),
        Cmd::Info(a) => info(a),
        Cmd::Backends { device } => backends(device),
        Cmd::Bench(a) => fullm::bench::run(&a),
        Cmd::Compare(a) => fullm::bench::compare(&a),
        Cmd::Ppl(a) => fullm::bench::ppl(&a),
        Cmd::Selftest { device, backend, reference } => selftest(device, backend, &reference),
        Cmd::Kbench { spec, device } => {
            let (rc, report) = fullm_kernels::exec::kbench(device, &spec);
            print!("{report}");
            match rc {
                -1 => bail!("CUDA is not available in this build or on this machine"),
                0 => Ok(()),
                _ => bail!("kbench: a case failed"),
            }
        }
        Cmd::Tokenize { model, text, chat, no_thinking, chat_template } => {
            tokenize(&model, &text, chat, no_thinking, chat_template.as_deref())
        }
    }
}

fn selftest(device: i32, backend: Option<BackendKind>, reference: &str) -> Result<()> {
    let subject = backend.unwrap_or_else(BackendKind::auto);
    if subject == BackendKind::Reference {
        bail!("the reference forward is the oracle, not a backend under test");
    }
    let (fails, report) = fullm_kernels::exec::conformance(subject.exec_name(), reference, device);
    print!("{report}");
    match fails {
        -1 => bail!("the suite could not run (see above)"),
        0 => {
            println!("selftest: all ops PASS");
            Ok(())
        }
        n => bail!("selftest: {n} op(s) FAILED"),
    }
}

/// The draft tokens per decode: --spec-tokens, else its default on `backend` (see the flag).
fn spec_tokens(a: &ModelArgs, backend: &BackendKind) -> usize {
    a.spec_tokens.unwrap_or(if a.spec_model.is_some() || !backend.host_memory(a.device) { 8 } else { 0 })
}

fn load_opts(a: &ModelArgs) -> LoadOptions {
    let backend = a.backend.clone().unwrap_or_else(BackendKind::auto);
    let spec_tokens = spec_tokens(a, &backend);
    LoadOptions {
        trunk: a.trunk,
        max_seqs: a.max_seqs,
        cache_tokens: a.cache_tokens,
        page: a.page,
        backend,
        device: a.device,
        state_dtype: a.state_dtype,
        kv_dtype: a.kv_dtype,
        step_tokens: a.step_tokens,
        // a step's rows: every decode, every draft (within spec_step_tokens), a row per prompt chunk
        max_rows: a.max_seqs.max(64).max(if spec_tokens > 0 { a.spec_step_tokens + a.max_seqs } else { 0 }),
        max_context: a.max_context,
        threads: a.threads,
        layers: a.layers,
        router_weights_biased: a.router_weights_biased,
        gpu_experts: a.gpu_experts,
        gpu_expert_format: a.gpu_expert_format,
        int4_checkpoint: a.int4_checkpoint.clone(),
        lora_merge: a.lora_merge.clone(),
        max_loras: a.max_loras.max(a.loras.len()),
        max_lora_rank: a.max_lora_rank,
        head: a.head,
        no_cuda_graphs: a.no_cuda_graphs,
        bf16_activations: a.activations == fullm::models::Activations::Bf16,
        no_prefix_cache: a.no_prefix_cache,
        host_cache_tokens: a.host_cache_tokens,
        host_copy_lag: 0,
        spec_model: a.spec_model.clone(),
    }
}

/// Load the model, its tokenizer and its LoRA adapters (in slots 0..), then start the engine
/// thread. Returns the slots' adapter names too.
fn start(a: &ModelArgs) -> Result<(fullm::runtime::EngineHandle, Arc<dyn Tokenizer>, Vec<Option<String>>)> {
    let t0 = Instant::now();
    let mut adapters = Vec::new();
    for spec in &a.loras {
        let (name, dir) = spec.split_once('=').with_context(|| format!("--lora {spec:?}: expected NAME=DIR"))?;
        let ad = models::lora::Adapter::load(Path::new(dir)).with_context(|| format!("--lora {name}"))?;
        adapters.push((name.to_string(), ad));
    }
    let mut opts = load_opts(a);
    if opts.max_lora_rank == 0 && opts.max_loras > 0 {
        opts.max_lora_rank = adapters.iter().map(|(_, ad)| ad.max_rank()).max().unwrap_or(64);
    }
    let mut model = models::load(&a.model, &opts).with_context(|| format!("loading {}", a.model.display()))?;
    let mut names = vec![None; opts.max_loras];
    for (slot, (name, ad)) in adapters.iter().enumerate() {
        model.set_lora(slot, ad).with_context(|| format!("--lora {name}"))?;
        names[slot] = Some(name.clone());
        tracing::info!("LoRA adapter {name} in slot {slot} (rank {})", ad.max_rank());
    }
    tracing::info!("loaded in {:.1}s: {}", t0.elapsed().as_secs_f64(), model.describe());
    let tok: Arc<dyn Tokenizer> = Arc::from(load_tokenizer_with(&a.model, model.vocab(), a.chat_template.as_deref())?);
    let spec_tokens = spec_tokens(a, &opts.backend);
    let source = match (a.spec_tokens, &a.spec_model) {
        (Some(_), _) => "given".to_string(),
        (None, Some(_)) => "default with --spec-model".to_string(),
        (None, None) => format!("default on {}", opts.backend),
    };
    tracing::info!("speculative decoding: --spec-tokens {spec_tokens} ({source})");
    if a.kv_dtype == CacheDtype::Int4 {
        tracing::info!(
            "--kv-dtype int4: a capacity setting (1.5x fp8's positions); slower than fp8 while fp8's cache holds the sessions"
        );
    }
    let engine = Engine::new(
        model,
        EngineConfig {
            step_tokens: a.step_tokens,
            admission: a.admission,
            conservativeness: a.schedule_conservativeness,
            prefill_order: a.prefill_order,
            spec_tokens,
            spec_ngram: a.spec_ngram,
            spec_step_tokens: a.spec_step_tokens,
            spec_by_acceptance: !a.spec_fixed,
            spec_global: !a.no_spec_global,
            eagle_tokens: a.spec_eagle_tokens,
            spec_own: !a.no_spec_ngram,
            spec_choice: a.spec_choice,
            decode_share: a.decode_share,
            mixed_step_tokens: a.mixed_step_tokens,
        },
    )
    .spawn()?;
    Ok((engine, tok, names))
}

/// MIN:MAX, 1 <= MIN <= MAX.
fn parse_span(s: &str) -> std::result::Result<(usize, usize), String> {
    let (a, b) = s.split_once(':').ok_or_else(|| format!("{s:?}: expected MIN:MAX"))?;
    let (a, b) =
        (a.trim().parse::<usize>().map_err(|e| e.to_string())?, b.trim().parse::<usize>().map_err(|e| e.to_string())?);
    if a == 0 || a > b {
        return Err(format!("{s:?}: expected 1 <= MIN <= MAX"));
    }
    Ok((a, b))
}

fn served_name(dir: &Path) -> String {
    dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into())
}

fn serve(a: ServeArgs) -> Result<()> {
    let (engine, tok, loras) = start(&a.model)?;
    let addr: SocketAddr = format!("{}:{}", a.host, a.port).parse().context("--host/--port")?;
    let sampling_defaults = if a.generation_config == "auto" {
        server::SamplingDefaults::from_generation_config(&a.model.model)
    } else {
        server::SamplingDefaults::default()
    };
    tracing::info!(
        "sampling defaults for unset request fields: {sampling_defaults:?} (--generation-config {})",
        a.generation_config
    );
    let state = Arc::new(AppState {
        engine,
        tok,
        model_name: a.served_name.unwrap_or_else(|| served_name(&a.model.model)),
        special_tokens_in_prompts: a.special_tokens_in_prompts,
        sampling_defaults,
        grammars: Default::default(),
        loras: std::sync::Mutex::new(loras),
        runtime_lora: a.allow_runtime_lora,
        api_key: a.api_key.or_else(|| std::env::var("FULLM_API_KEY").ok()).filter(|k| !k.is_empty()),
    });
    tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(server::serve(state, addr))
}

fn generate(a: GenerateArgs) -> Result<()> {
    let (engine, tok, _) = start(&a.model)?;
    let prompt = match (&a.prompt, &a.ids) {
        (Some(p), None) => tok.encode(p),
        (None, Some(ids)) => ids.clone(),
        _ => bail!("give exactly one of --prompt or --ids"),
    };
    let stop = if a.ignore_eos { Vec::new() } else { tok.stop_ids() };
    let params = SamplingParams {
        max_tokens: a.max_tokens,
        temperature: a.temperature,
        top_p: a.top_p,
        top_k: a.top_k,
        seed: a.seed,
        stop_token_ids: stop,
        ..Default::default()
    };
    let n_prompt = prompt.len();
    let t0 = Instant::now();
    let mut rx = engine.submit(Request { prompt, params })?;
    let mut detok = Detokenizer::default();
    let (mut first, mut ids) = (None, Vec::new());
    let mut out = std::io::stdout().lock();
    let (reason, usage) = loop {
        match rx.blocking_recv() {
            Some(Event::Token(t)) => {
                first.get_or_insert_with(|| t0.elapsed());
                ids.push(t);
                if !tok.is_special(t) {
                    write!(out, "{}", detok.push(&tok.decode_bytes(&[t])))?;
                    out.flush()?;
                }
            }
            Some(Event::Finished { reason, usage }) => break (reason, usage),
            Some(Event::Logprobs(_)) => {}
            None => bail!("engine stopped"),
        }
    };
    writeln!(out, "{}", detok.finish())?;
    let total = t0.elapsed().as_secs_f64();
    let ttft = first.map(|d| d.as_secs_f64()).unwrap_or(total);
    let decode_s = (total - ttft).max(1e-9);
    eprintln!(
        "\n[{}] prompt {n_prompt} tok, generated {} tok | first token {:.3}s | decode {:.2} tok/s | total {:.2}s\nids: {:?}",
        reason.as_openai(),
        usage.completion_tokens,
        ttft,
        (ids.len().saturating_sub(1)) as f64 / decode_s,
        total,
        ids
    );
    if let fullm::runtime::FinishReason::Error(e) = reason {
        bail!("generation failed: {e}");
    }
    Ok(())
}

/// `fullm backends`: each backend of this build with its capabilities.
fn backends(device: i32) -> Result<()> {
    use fullm_kernels::exec::{backend_caps, backends, Fmt};
    let fmts = |bits: u32| -> String {
        let all = [Fmt::F32, Fmt::Bf16, Fmt::Int8, Fmt::Fp8, Fmt::Fp8Block, Fmt::Int4Group];
        all.iter()
            .filter(|f| bits & (1 << **f as i32) != 0)
            .map(|f| format!("{f:?}").to_lowercase())
            .collect::<Vec<_>>()
            .join(" ")
    };
    let kvs = |bits: u32| -> String {
        ["f32", "bf16", "fp8-token", "fp8", "int4"]
            .iter()
            .enumerate()
            .filter(|(i, _)| bits & (1 << i) != 0)
            .map(|(_, n)| *n)
            .collect::<Vec<_>>()
            .join(" ")
    };
    let fams = |bits: u32| -> String {
        ["gqa", "mla", "kda", "kimi"]
            .iter()
            .enumerate()
            .filter(|(i, _)| bits & (1 << i) != 0)
            .map(|(_, n)| *n)
            .collect::<Vec<_>>()
            .join(" ")
    };
    for (name, ok) in backends() {
        if !ok {
            println!("{name}: not usable here");
            continue;
        }
        match backend_caps(&name, device, true) {
            Ok(c) => println!(
                "{name}: {:?} memory, graphs {}, async copies {}\n  model families: {}\n  dense weights: {}\n  resident experts: {}\n  KV cache: {} (fp8 pages: multiples of {})\n  policy: int4 decode tiles up to {} rows, int4 GEMM from {} rows",
                c.memory, c.graphs, c.async_copy, fams(c.families), fmts(c.dense_formats), fmts(c.expert_formats), kvs(c.kv_formats),
                c.kv8_page_multiple, c.expert_tile_max_rows, c.expert_gemm_min_rows
            ),
            Err(e) => println!("{name}: {e}"),
        }
    }
    Ok(())
}

fn tokenize(
    dir: &std::path::Path,
    text: &str,
    chat: bool,
    no_thinking: bool,
    template: Option<&std::path::Path>,
) -> Result<()> {
    let raw = std::fs::read_to_string(dir.join("config.json")).context("reading config.json")?;
    let v: serde_json::Value = serde_json::from_str(&raw)?;
    let vocab = v.get("vocab_size").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
    let tok = load_tokenizer_with(dir, vocab, template)?;
    let ids = if chat {
        // a JSON array of messages, an object {"messages": [...], "tools": [...]}, or one user message
        let (msgs, tools): (Vec<fullm::tokenizer::ChatMessage>, Vec<serde_json::Value>) = match text
            .trim_start()
            .chars()
            .next()
        {
            Some('[') => (serde_json::from_str(text).context("--text: a JSON array of messages")?, Vec::new()),
            Some('{') => {
                let v: serde_json::Value =
                    serde_json::from_str(text).context("--text: {\"messages\": [...], \"tools\": [...]}")?;
                let msgs = serde_json::from_value(v.get("messages").cloned().unwrap_or_default())
                    .context("--text: messages")?;
                let tools = v.get("tools").and_then(|t| t.as_array()).cloned().unwrap_or_default();
                (msgs, tools)
            }
            _ => (
                vec![fullm::tokenizer::ChatMessage { role: "user".into(), content: text.into(), ..Default::default() }],
                Vec::new(),
            ),
        };
        let thinking = !no_thinking && tok.soft_thinking(&msgs) != Some(false);
        let opts = fullm::tokenizer::ChatOptions { thinking, tools, ..Default::default() };
        tok.render_chat(&msgs, &opts)?
    } else {
        tok.encode(text)
    };
    println!("{}", ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","));
    Ok(())
}

fn info(a: ModelArgs) -> Result<()> {
    let model = models::load(&a.model, &load_opts(&a))?;
    println!("{}", model.describe());
    println!("max context: {} positions", model.max_context());
    println!("capacity: {}", serde_json::to_string_pretty(&model.capacity())?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    /// Every subcommand's flags parse into their declared types (clap checks the value-parser
    /// output type only when an argument is read, so a mismatch shows up at run time).
    #[test]
    fn cli_parses() {
        Cli::command().debug_assert();
        for argv in [
            "fullm serve --model m --gpu-experts auto --gpu-expert-format int8",
            "fullm serve --model m --gpu-experts 12.5",
            "fullm generate --model m --gpu-experts off --prompt hi",
            "fullm info --model m",
            "fullm bench --gpu-experts auto",
            "fullm compare --model m --gpu-experts 4",
            "fullm ppl --model m --text t --gpu-experts auto --gpu-expert-format fp8",
        ] {
            let cli = Cli::try_parse_from(argv.split(' ')).unwrap_or_else(|e| panic!("{argv}: {e}"));
            let got = match cli.cmd {
                Cmd::Serve(a) => a.model.gpu_experts,
                Cmd::Generate(a) => a.model.gpu_experts,
                Cmd::Info(a) => a.gpu_experts,
                Cmd::Bench(a) => a.gpu_experts,
                Cmd::Compare(a) => a.gpu_experts,
                Cmd::Ppl(a) => a.gpu_experts,
                _ => unreachable!(),
            };
            let want = argv.split("--gpu-experts ").nth(1).map(|v| v.split(' ').next().unwrap());
            let want = want.map_or(models::GpuExperts::Auto, |v| models::parse_gpu_experts(v).unwrap());
            assert_eq!(got, want, "{argv}");
        }
    }

    /// --spec-tokens defaults to 0 on a CPU backend and 8 on a GPU one or with a draft head; a
    /// given value always holds.
    #[test]
    fn spec_tokens_default() {
        let args = |argv: &str| match Cli::try_parse_from(argv.split(' ')).unwrap().cmd {
            Cmd::Serve(a) => a.model,
            _ => unreachable!(),
        };
        for backend in [BackendKind::Reference, BackendKind::CPU] {
            assert_eq!(spec_tokens(&args("fullm serve --model m"), &backend), 0, "{backend}");
            assert_eq!(spec_tokens(&args("fullm serve --model m --spec-model h"), &backend), 8, "{backend}");
            assert_eq!(spec_tokens(&args("fullm serve --model m --spec-tokens 3"), &backend), 3, "{backend}");
        }
        // CUDA here or not (a backend that cannot run here keeps the GPU default)
        assert_eq!(spec_tokens(&args("fullm serve --model m"), &BackendKind::CUDA), 8);
        assert_eq!(spec_tokens(&args("fullm serve --model m --spec-tokens 0"), &BackendKind::CUDA), 0);
    }
}
