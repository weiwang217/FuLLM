// SPDX-License-Identifier: Apache-2.0
//! The model interface the engine schedules against, and the architecture registry.
//!
//! A `Model` owns its weights AND its per-sequence state pools (recurrent slots, paged KV),
//! because their shapes are architecture-specific: Kimi K3 carries a fixed-size KDA state
//! per sequence plus a latent KV cache for 24 of 93 layers, a dense GQA model carries only
//! KV. The engine sees sequences as opaque ids with a capacity check.

pub mod dense;
pub mod eagle;
pub mod kimi_k3;
pub mod lora;
pub mod paged;
pub mod prefix;

use std::borrow::Cow;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};

use crate::tensor::Precision;

pub type SeqId = usize;

pub use fullm_kernels::exec::{LogitAdj, SampleIo, SampleRow, MAX_TOP_LOGPROBS};

/// A token as the prefix cache keys it for a sequence running adapter slot `lora`: its own id for
/// the base model, the slot in the bits above 24 otherwise (vocabularies stay under 2^24), so a
/// cache filled under one adapter never serves another.
pub fn lora_key(t: u32, lora: Option<usize>) -> u32 {
    lora.map_or(t, |s| t | ((s as u32 + 1) << 24))
}

/// One sequence's slice of a step: `tokens` occupy positions `pos..pos + tokens.len()`.
/// `pos` must equal the sequence's current length; models refuse anything else.
pub struct SeqChunk<'a> {
    pub seq: SeqId,
    pub pos: usize,
    pub tokens: &'a [u32],
    /// Logits for every position (teacher forcing, verification) or only the last.
    pub all_logits: bool,
    /// The last `draft` tokens are speculative (`Model::speculation`): their cache is written,
    /// but the sequence grows by them only as far as `Model::accept` says after the step.
    pub draft: usize,
}

/// One sequence of an EAGLE-3 draft call (`Model::draft`): `next` is the token after the last
/// position its chunk of the last step added (the one sampled there); up to `k` drafts follow it.
#[derive(Clone, Copy, Debug)]
pub struct DraftReq {
    pub seq: SeqId,
    pub next: u32,
    pub k: usize,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Capacity {
    pub max_seqs: usize,
    pub free_seqs: usize,
    /// Positions of attention cache no sequence holds (free, or kept by the prefix cache).
    pub free_cache_tokens: usize,
    /// Positions the prefix cache keeps on the device, held by running sequences or not.
    pub prefix_cached_tokens: usize,
    /// Positions the prefix cache's host tier keeps (every cached page, resident or not).
    pub host_cached_tokens: usize,
}

pub trait Model: Send {
    fn arch(&self) -> &'static str;
    fn vocab(&self) -> usize;
    fn max_context(&self) -> usize;
    fn capacity(&self) -> Capacity;
    /// Reserve per-sequence state. `None` when every slot is taken.
    fn alloc_seq(&mut self) -> Option<SeqId>;
    /// `alloc_seq`, starting with the longest prefix of `prompt` whose cache another sequence
    /// left behind: returns the sequence and its length, the positions already in place (fewer
    /// than `prompt.len()`: the last prompt token is always computed, for its logits).
    fn alloc_seq_cached(&mut self, _prompt: &[u32]) -> Option<(SeqId, usize)> {
        Some((self.alloc_seq()?, 0))
    }
    /// Whether `alloc_seq_cached` can find cached prefixes.
    fn has_prefix_cache(&self) -> bool {
        false
    }
    /// Drop the prefix cache's pages that no sequence holds. Returns their positions.
    fn flush_prefix_cache(&mut self) -> usize {
        0
    }
    /// The positions of `prompt` right after what `seq` holds (fresh from `alloc_seq_cached`)
    /// that the prefix cache's host tier could load instead of computing. Changes nothing.
    fn host_cached(&self, _seq: SeqId, _prompt: &[u32]) -> usize {
        0
    }
    /// Extend `seq` by up to `n` of those positions, loaded from host memory into cache it takes
    /// now (as much as `n` positions of new cache would). Returns the positions added; the
    /// sequence may not run until `seq_ready`.
    fn load_cached(&mut self, _seq: SeqId, _prompt: &[u32], _n: usize) -> Result<usize> {
        Ok(0)
    }
    /// Whether `seq`'s loads from the host tier have completed (asks without blocking).
    fn seq_ready(&mut self, _seq: SeqId) -> bool {
        true
    }
    /// Block until a sequence waiting for its loads from the host tier can run (the one that
    /// has waited longest; the others keep loading).
    fn wait_copies(&mut self) -> Result<()> {
        Ok(())
    }
    fn free_seq(&mut self, seq: SeqId);
    /// Whether `seq` can grow by `n` more positions with the cache that is free now.
    fn can_grow(&self, seq: SeqId, n: usize) -> bool;
    /// Positions per cache page: the unit the cache is granted in.
    fn page_tokens(&self) -> usize;
    /// Run one step over a batch of chunks from distinct sequences. Returns, per chunk,
    /// `[rows][vocab]` logits flattened, where rows is `tokens.len()` or 1.
    fn forward(&mut self, batch: &[SeqChunk<'_>]) -> Result<Vec<Vec<f32>>>;
    /// The step for greedy decoding: the argmax of each row's logits (the token `forward` +
    /// `sampler::argmax` would pick), the rows as `forward` lays them out: one per chunk, or
    /// every position of a chunk that asks for all logits. Backends that can, find them on the
    /// device and never move the logits.
    fn forward_greedy(&mut self, batch: &[SeqChunk<'_>]) -> Result<Vec<u32>> {
        let v = self.vocab();
        Ok(self.forward(batch)?.iter().flat_map(|l| l.chunks(v).map(crate::runtime::sampler::argmax)).collect())
    }
    /// `forward` with the rows laid out flat in `out` (reused across steps): `[rows][vocab]` in
    /// batch order. A server step moves every sampled row's logits; one buffer kept warm spares
    /// it a fresh multi-megabyte allocation and a copy per chunk each step.
    fn forward_flat(&mut self, batch: &[SeqChunk<'_>], out: &mut Vec<f32>) -> Result<()> {
        out.clear();
        for l in self.forward(batch)? {
            out.extend_from_slice(&l);
        }
        Ok(())
    }
    /// Whether `forward_sample` draws on the device.
    fn device_sampling(&self) -> bool {
        false
    }
    /// The step for sampling: each row's token drawn as `sample[row]` says (rows as `forward`
    /// lays them out), on the device, so only the ids (and asked-for log-probabilities) cross to
    /// the host. `io`: the allowed-token bitmasks the rows' `mask` fields name (structured
    /// output; `mask_words` = `vocab().div_ceil(32)`), the logit adjustments their `adj_off` /
    /// `adj_n` index, and the log-probability buffers (`SampleIo`).
    fn forward_sample(
        &mut self,
        _batch: &[SeqChunk<'_>],
        _sample: &[SampleRow],
        _io: &mut SampleIo<'_>,
    ) -> Result<Vec<u32>> {
        anyhow::bail!("this model draws on the host (forward_flat and the sampler)")
    }
    /// Whether chunks may carry draft tokens (speculative decoding): the sequence's state is
    /// only its attention cache, so a rejected draft's positions are simply written again.
    fn speculation(&self) -> bool {
        false
    }
    /// After a step whose chunk for `seq` carried drafts: the first `n` were right. The
    /// sequence grows by them (and the pages they fill enter the prefix cache); the rest are
    /// dropped.
    fn accept(&mut self, _seq: SeqId, _n: usize) -> Result<()> {
        Ok(())
    }
    /// Whether `draft` runs: an EAGLE-3 head is loaded (`LoadOptions::spec_model`).
    fn draft_head(&self) -> bool {
        false
    }
    /// EAGLE-3, after a step and its `accept`s: for each request, the head reads the positions
    /// its sequence's chunk added (their target hidden states, each with the token after it),
    /// then proposes up to `k` tokens to follow `next` (fewer when the cache cannot take them).
    /// The head keeps a cache of its own over the whole context, so every chunk of every step
    /// passes through here, drafting or not (`k` = 0).
    fn draft(&mut self, _reqs: &[DraftReq]) -> Result<Vec<Vec<u32>>> {
        bail!("this model has no draft head")
    }
    /// Human-readable description of where the weights and state live.
    fn describe(&self) -> String;
    /// LoRA adapter slots (unmerged adapters, many at once; 0: none).
    fn lora_slots(&self) -> usize {
        0
    }
    /// Adapter `a` into slot `slot`, replacing what it held.
    fn set_lora(&mut self, _slot: usize, _a: &lora::Adapter) -> Result<()> {
        anyhow::bail!("this model serves no LoRA adapters (serve --max-loras)")
    }
    /// Slot `slot` back to the base model.
    fn clear_lora(&mut self, _slot: usize) -> Result<()> {
        anyhow::bail!("this model serves no LoRA adapters")
    }
    /// The adapter sequence `seq` runs with from its next step (None: the base model). Its
    /// prefix-cache keys are the adapter's (`lora_key`): a cache filled under one adapter (or none)
    /// never serves another.
    fn set_seq_lora(&mut self, _seq: SeqId, slot: Option<usize>) -> Result<()> {
        anyhow::ensure!(slot.is_none(), "this model serves no LoRA adapters");
        Ok(())
    }
    /// Timing and expert traffic of the last step, when the backend measures them.
    fn last_step(&self) -> Option<fullm_kernels::exec::StepStats> {
        None
    }
}

/// Activations of small-batch GEMVs (the GPU's tensor-core GEMV path).
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
pub enum Activations {
    /// Rounded to bf16, as cuBLAS and vLLM compute: faster.
    Bf16,
    /// Kept fp32 (TF32 hi + lo on the tensor cores): the reference results.
    Exact,
}

/// Which implementation runs the model: the Rust reference, or the C++ executor on one of the
/// backends compiled into this build, by name (`fullm backends` lists them: cpu, cuda, torch,
/// ...). On the command line: `reference`, `auto` (the first usable backend with memory of its
/// own, else cpu) or a backend's name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// The Rust reference forward on the CPU (the oracle every other backend is checked against).
    Reference,
    /// The C++ executor on the named backend.
    Executor(Cow<'static, str>),
}

impl BackendKind {
    /// The C++ executor on the CPU.
    pub const CPU: Self = BackendKind::Executor(Cow::Borrowed("cpu"));
    /// The C++ executor with the trunk on a CUDA GPU (routed experts in GPU memory or on the CPU).
    pub const CUDA: Self = BackendKind::Executor(Cow::Borrowed("cuda"));
    /// The C++ executor on the CPU, every bf16 GEMM through libtorch (as vLLM on a CPU); needs a
    /// build with the `torch` feature.
    pub const TORCH: Self = BackendKind::Executor(Cow::Borrowed("torch"));

    /// The executor backend's name (the reference runs none: "cpu" sizes its options).
    pub fn exec_name(&self) -> &str {
        match self {
            BackendKind::Reference => "cpu",
            BackendKind::Executor(n) => n,
        }
    }

    /// The first usable backend with memory of its own (a GPU), in the build's order, else the CPU.
    pub fn auto() -> Self {
        use fullm_kernels::exec::{backend_caps, backends, MemKind};
        for (name, ok) in backends() {
            if ok && backend_caps(&name, 0, true).is_ok_and(|c| c.memory != MemKind::Host) {
                return BackendKind::Executor(Cow::Owned(name));
            }
        }
        Self::CPU
    }

    /// Whether the backend computes in host memory (the reference, cpu, cpu-core, torch): a step
    /// there is compute-bound, not bandwidth-bound as on a GPU. False for a backend that cannot
    /// say (not usable here: loading it fails anyway).
    pub fn host_memory(&self, device: i32) -> bool {
        use fullm_kernels::exec::{backend_caps, MemKind};
        match self {
            BackendKind::Reference => true,
            BackendKind::Executor(n) => backend_caps(n, device, true).is_ok_and(|c| c.memory == MemKind::Host),
        }
    }
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendKind::Reference => f.write_str("reference"),
            BackendKind::Executor(n) => f.write_str(n),
        }
    }
}

impl std::str::FromStr for BackendKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "reference" => Ok(BackendKind::Reference),
            "auto" => Ok(BackendKind::auto()),
            name => {
                // a backend of this build, usable here or not (creating it says why it cannot run)
                let list = fullm_kernels::exec::backends();
                match list.iter().find(|(n, _)| n == name) {
                    Some(_) => Ok(BackendKind::Executor(Cow::Owned(name.to_string()))),
                    None => {
                        let names: Vec<&str> = list.iter().map(|(n, _)| n.as_str()).collect();
                        Err(format!("unknown backend {name:?}: reference, auto or one of {}", names.join(", ")))
                    }
                }
            }
        }
    }
}

impl Serialize for BackendKind {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for BackendKind {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Ok(if s == "reference" { BackendKind::Reference } else { BackendKind::Executor(Cow::Owned(s)) })
    }
}

/// Storage of recurrent state and the attention cache (vLLM's `--mamba-ssm-cache-dtype` and
/// `--kv-cache-dtype` analogues). `Auto` follows the model dtype (bf16).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum CacheDtype {
    Auto,
    Bf16,
    F32,
    /// The backend's best fp8 cache (half the bytes of bf16; changes numerics, as vLLM's and
    /// SGLang's fp8 caches do): on CUDA with bf16 activations and the cutlass feature, e4m3 pages at
    /// one scale, FlashAttention-3's fp8 layout, which it reads with e4m3 queries on the fp8 tensor
    /// cores for prefill and decode alike (dense GQA models, head_dim 128); else Fp8Token where the
    /// backend has that (`fullm backends` lists each backend's caches).
    Fp8,
    /// e4m3 with a power-of-two scale per (token, kv head), read by fullm's own decode kernel
    /// and dequantised to bf16 for the prefill's FlashAttention-3.
    Fp8Token,
    /// int4 pages, 0.64 bytes a value (1.56x fp8's positions): keys per channel over each page of 16
    /// (KIVI), values rotated by a Hadamard and scaled per position, the page being filled kept
    /// exact (fullm's own decode kernel; prompt chunks dequantised to bf16 for FlashAttention-3).
    /// CUDA with bf16 activations and the cutlass feature, dense GQA models with head_dim 128,
    /// pages of 16. A capacity setting, not a speed one: on an H100 (Qwen3-30B-A3B, agent sessions
    /// with n-gram drafts) 1.8x fp8 at 32 users when both get the same 8 GB (fp8's positions no
    /// longer hold the sessions), and 15-20% slower than fp8 where fp8's cache holds them (its decode
    /// kernel takes a sequence's draft checks two rows a block). As accurate for agents as bf16.
    Int4,
}

impl CacheDtype {
    pub fn is_bf16(self) -> bool {
        self != CacheDtype::F32
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoadOptions {
    /// Storage of the large trunk matrices. Fp8 reproduces the GPU's resident format.
    pub trunk: Precision,
    /// Concurrent sequences (recurrent-state slots).
    pub max_seqs: usize,
    /// Total cached positions across all sequences.
    pub cache_tokens: usize,
    /// Positions per KV page.
    pub page: usize,
    pub backend: BackendKind,
    /// CUDA device ordinal.
    pub device: i32,
    /// Recurrent-state storage (executor backends; the reference is always f32).
    pub state_dtype: CacheDtype,
    /// Latent-cache storage (executor backends; the reference is always f32).
    pub kv_dtype: CacheDtype,
    /// Most tokens one step may carry (sizes the executor workspace).
    pub step_tokens: usize,
    /// Most logits rows one step may return.
    pub max_rows: usize,
    /// Longest sequence, in positions.
    pub max_context: usize,
    /// CPU threads for the expert pool and CPU backend (0 = all cores).
    pub threads: usize,
    /// Bind only the first N layers (debugging real weights from a partial download).
    pub layers: Option<usize>,
    /// MoE combining weights from the biased scores (see K3Config::router_weights_biased).
    pub router_weights_biased: bool,
    /// Routed experts resident in backend memory (the HBM tier).
    pub gpu_experts: GpuExperts,
    /// Storage of resident experts: Bf16 (exact, from bf16 checkpoints), Int8 or Fp8.
    pub gpu_expert_format: Precision,
    /// Run every step op by op instead of replaying CUDA graphs of small steps.
    #[serde(default)]
    pub no_cuda_graphs: bool,
    /// Small-M GEMVs round activations to bf16 (as vLLM does) instead of keeping them fp32.
    #[serde(default)]
    pub bf16_activations: bool,
    /// Free every page when its sequence ends instead of keeping full pages for later requests
    /// with the same prefix (models whose per-sequence state is all paged cache).
    #[serde(default)]
    pub no_prefix_cache: bool,
    /// Back the prefix cache with this many positions of pinned host memory (more than
    /// `cache_tokens`; 0 = none): pages the device evicts stay there and load back on a hit.
    #[serde(default)]
    pub host_cache_tokens: usize,
    /// Tests: a sequence counts as still loading from the host tier for this many more polls.
    #[serde(default)]
    pub host_copy_lag: usize,
    /// An EAGLE-3 draft head (a checkpoint directory) for a dense target: `Model::draft`.
    #[serde(default)]
    pub spec_model: Option<std::path::PathBuf>,
    /// With `gpu_expert_format` int4: the routed experts' values and scales from this GPTQ
    /// checkpoint (4 bits, symmetric, groups of 128, no act-order: Qwen's GPTQ-Int4 releases)
    /// instead of rounding the model's bf16 experts; everything else still comes from the model.
    #[serde(default)]
    pub int4_checkpoint: Option<std::path::PathBuf>,
    /// A LoRA adapter (PEFT's format) added to the weights at load (`lora::Adapter`): the model
    /// serves the fine-tune at the base model's speed.
    #[serde(default)]
    pub lora_merge: Option<std::path::PathBuf>,
    /// LoRA adapters served unmerged, many at once: slots (0: none) and their largest rank.
    #[serde(default)]
    pub max_loras: usize,
    #[serde(default)]
    pub max_lora_rank: usize,
    /// Storage of the lm_head (None: bf16). A tied head is otherwise a bf16 copy of the embedding
    /// table: Qwen3-4B's is 0.78 GB, 18% of a decode step's bytes on an 8 GB GPU.
    #[serde(default)]
    pub head: Option<Precision>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions {
            trunk: Precision::Bf16,
            max_seqs: 8,
            cache_tokens: 16 * 1024,
            page: 16,
            backend: BackendKind::Reference,
            device: 0,
            state_dtype: CacheDtype::Auto,
            kv_dtype: CacheDtype::Auto,
            step_tokens: 512,
            max_rows: 64,
            max_context: 32 * 1024,
            threads: 0,
            layers: None,
            router_weights_biased: false,
            gpu_experts: GpuExperts::Off,
            gpu_expert_format: Precision::Bf16,
            no_cuda_graphs: false,
            bf16_activations: false,
            no_prefix_cache: false,
            host_cache_tokens: 0,
            host_copy_lag: 0,
            spec_model: None,
            int4_checkpoint: None,
            lora_merge: None,
            max_loras: 0,
            max_lora_rank: 0,
            head: None,
        }
    }
}

/// How many routed experts to keep in backend memory (the HBM tier).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuExperts {
    Off,
    /// As many as fit in free GPU memory.
    Auto,
    /// This many GB.
    Gb(f64),
}

/// `--gpu-experts`: "off" | "auto" | a size in GB.
pub fn parse_gpu_experts(s: &str) -> Result<GpuExperts, String> {
    match s {
        "off" | "none" | "0" => Ok(GpuExperts::Off),
        "auto" => Ok(GpuExperts::Auto),
        n => n.parse::<f64>().map(GpuExperts::Gb).map_err(|_| format!("expected off, auto or GB, got {n:?}")),
    }
}

/// Architectures this build can serve, keyed by how a config identifies itself.
pub fn detect_arch(config: &serde_json::Value) -> Result<&'static str> {
    let mt = config
        .get("model_type")
        .or_else(|| config.get("text_config").and_then(|t| t.get("model_type")))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let txt = config.get("text_config").unwrap_or(config);
    let has_kda = txt.get("linear_attn_config").is_some() || config.get("kda_num_heads").is_some();
    if mt.starts_with("kimi_k3") || mt == "kimi_linear" || has_kda {
        return Ok("kimi_k3");
    }
    if mt == "qwen3" || mt == "qwen3_moe" || mt == "llama" {
        return Ok("dense");
    }
    bail!("unsupported architecture (model_type {mt:?}); this build serves: kimi_k3, kimi_linear, qwen3, qwen3_moe, llama")
}

pub fn load(dir: &Path, opts: &LoadOptions) -> Result<Box<dyn Model>> {
    let cfg_path = dir.join("config.json");
    let raw = std::fs::read_to_string(&cfg_path).with_context(|| format!("reading {}", cfg_path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&raw).with_context(|| format!("parsing {}", cfg_path.display()))?;
    let arch = detect_arch(&v)?;
    ensure!(
        opts.int4_checkpoint.is_none() || arch == "dense",
        "--int4-checkpoint: for Qwen3-MoE's routed experts only"
    );
    ensure!(opts.lora_merge.is_none() || arch == "dense", "--lora-merge: for Qwen3 and Llama models only");
    match arch {
        "kimi_k3" => match opts.backend {
            BackendKind::Reference => Ok(Box::new(kimi_k3::cpu::K3Cpu::load(dir, &v, opts)?)),
            BackendKind::Executor(_) => Ok(Box::new(kimi_k3::exec::K3Exec::load(dir, &v, opts)?)),
        },
        "dense" => Ok(Box::new(dense::DenseModel::load(dir, &v, opts)?)),
        other => bail!("no backend for {other}"),
    }
}
