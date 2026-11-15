// SPDX-License-Identifier: Apache-2.0
//! `fullm bench`: measure the rates the planner assumes, then the model end to end.
//!
//! The planner's presets are effective rates someone else measured. On the target box these
//! replace them: DRAM read bandwidth, CPU MXFP4 expert throughput (bandwidth-bound at
//! decode, compute-bound at prefill), pinned PCIe copies, and the GPU GEMV/GEMM rates at
//! Kimi K3's real matrix shapes. With `--model`, it also times prefill and decode and
//! reports where each step's time went.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use fullm_kernels::exec::{self, Fmt};
use fullm_kernels::WeightRef;
use rayon::prelude::*;
use serde::Serialize;

use crate::models::{self, BackendKind, CacheDtype, LoadOptions, SeqChunk};
use crate::tensor::Precision;

#[derive(clap::Args, Debug, Clone)]
pub struct BenchArgs {
    /// Also load this model and time prefill and decode.
    #[arg(long)]
    pub model: Option<PathBuf>,
    #[arg(long)]
    pub backend: Option<BackendKind>,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    #[arg(long, default_value = "int8", value_parser = parse_precision)]
    pub trunk: Precision,
    /// Storage of the lm_head: bf16 (default) or int8 / fp8 (see serve --head).
    #[arg(long, value_parser = parse_precision)]
    pub head: Option<Precision>,
    #[arg(long, value_enum, default_value = "auto")]
    pub state_dtype: CacheDtype,
    #[arg(long, value_enum, default_value = "auto")]
    pub kv_dtype: CacheDtype,
    /// Prompts cut from this text file (a different stretch per sequence) instead of synthetic
    /// token ids: realistic routing for MoE models, whose batch cost depends on how many
    /// distinct experts the batch's tokens pick.
    #[arg(long)]
    pub text: Option<PathBuf>,
    /// Prompt length per sequence.
    #[arg(long, default_value_t = 256)]
    pub prompt: usize,
    /// Decode steps.
    #[arg(long, default_value_t = 16)]
    pub decode: usize,
    /// Sequences decoded together.
    #[arg(long, default_value_t = 1)]
    pub batch: usize,
    #[arg(long, default_value_t = 512)]
    pub step_tokens: usize,
    #[arg(long, default_value_t = 0)]
    pub threads: usize,
    #[arg(long)]
    pub layers: Option<usize>,
    /// Routed experts resident in GPU memory: off, auto (as many as fit) or a size in GB.
    #[arg(long, default_value = "auto", value_parser = crate::models::parse_gpu_experts)]
    pub gpu_experts: crate::models::GpuExperts,
    /// Storage of GPU-resident experts.
    #[arg(long, default_value = "bf16", value_parser = parse_precision)]
    pub gpu_expert_format: Precision,
    /// With --gpu-expert-format int4: the routed experts from this GPTQ checkpoint.
    #[arg(long)]
    pub int4_checkpoint: Option<std::path::PathBuf>,
    /// A LoRA adapter (PEFT's format) merged into the weights at load.
    #[arg(long)]
    pub lora_merge: Option<std::path::PathBuf>,
    /// Run every step op by op instead of replaying CUDA graphs of small steps.
    #[arg(long)]
    pub no_cuda_graphs: bool,
    /// Activations of small-batch GEMVs: bf16 (as vLLM; faster) or exact (fp32).
    #[arg(long, value_enum, default_value = "bf16")]
    pub activations: crate::models::Activations,
    /// Skip the hardware microbenchmarks.
    #[arg(long)]
    pub skip_hw: bool,
    #[arg(long)]
    pub json: bool,
}

fn parse_precision(s: &str) -> Result<Precision, String> {
    match s {
        "bf16" => Ok(Precision::Bf16),
        "int8" => Ok(Precision::Int8),
        "fp8" => Ok(Precision::Fp8),
        "fp8b" => Ok(Precision::Fp8Block),
        "int4" => Ok(Precision::Int4Group),
        _ => Err(format!("unknown precision {s:?}")),
    }
}

#[derive(Serialize, Default)]
struct Report {
    host: Option<HostRates>,
    gpu: Option<GpuRates>,
    model: Option<ModelRates>,
}

#[derive(Serialize)]
struct HostRates {
    ram_gb: f64,
    dram_read_gbps: f64,
    cpu_isa: String,
    threads: usize,
    /// MXFP4 expert GEMV, one token per expert: bytes of expert weights per second.
    expert_gbps_decode: f64,
    /// MXFP4 expert GEMM, 64 tokens per expert: FLOP/s (the prefill regime).
    expert_tflops_prefill: f64,
}

#[derive(Serialize)]
struct GpuRates {
    name: String,
    total_gb: f64,
    sm_count: i32,
    cc: (i32, i32),
    h2d_gbps: f64,
    d2h_gbps: f64,
    gemv: Vec<GemmRate>,
}

#[derive(Serialize)]
struct GemmRate {
    what: String,
    fmt: String,
    n: i64,
    k: i64,
    m: i64,
    gbps: f64,
    ms: f64,
}

#[derive(Serialize)]
struct ModelRates {
    description: String,
    load_s: f64,
    /// The warm-up prefill (a cold first forward) on a scratch sequence, not in prefill_s.
    warmup_s: f64,
    prefill_tokens: usize,
    prefill_s: f64,
    prefill_tok_s: f64,
    batch: usize,
    decode_steps: usize,
    decode_ms_per_step: f64,
    decode_tok_s_total: f64,
    /// From the executor's last decode step, when it reports stats.
    last_step_ms: Option<f64>,
    last_step_experts_ms: Option<f64>,
    last_step_gpu_wait_ms: Option<f64>,
    last_step_expert_gb: Option<f64>,
    last_step_distinct_experts: Option<i64>,
}

pub fn run(a: &BenchArgs) -> Result<()> {
    let mut rep = Report::default();
    if !a.skip_hw {
        rep.host = Some(host_rates());
        if exec::cuda_available() {
            rep.gpu = gpu_rates(a.device);
        }
    }
    if let Some(dir) = &a.model {
        rep.model = Some(model_rates(dir, a)?);
    }
    if a.json {
        println!("{}", serde_json::to_string_pretty(&rep)?);
    } else {
        print!("{}", render(&rep));
    }
    Ok(())
}

fn host_rates() -> HostRates {
    let h = crate::hw::probe_host();
    let (gbps, tflops) = expert_rates();
    HostRates {
        ram_gb: h.ram_gb,
        dram_read_gbps: h.dram_gbps,
        cpu_isa: fullm_kernels::cpu_isa().to_string(),
        threads: rayon::current_num_threads(),
        expert_gbps_decode: gbps,
        expert_tflops_prefill: tflops,
    }
}

/// Synthetic experts at Kimi K3's shape (latent 3584, inter 3072), enough of them to blow
/// through the last-level cache, multiplied the way the expert pool does it.
fn expert_rates() -> (f64, f64) {
    let (l, i) = (3584usize, 3072usize);
    let n_exp = 48;
    let mat = |rows: usize, cols: usize, seed: usize| -> (Vec<u8>, Vec<u8>) {
        let packed: Vec<u8> = (0..rows * cols / 2).map(|j| ((j * 2654435761 + seed) >> 7) as u8).collect();
        let scales: Vec<u8> = (0..rows * cols / 32).map(|j| 118 + ((j + seed) % 8) as u8).collect();
        (packed, scales)
    };
    let experts: Vec<[(Vec<u8>, Vec<u8>); 3]> =
        (0..n_exp).into_par_iter().map(|e| [mat(i, l, e), mat(i, l, e + 1), mat(l, i, e + 2)]).collect();
    let bytes_per_expert: usize = experts[0].iter().map(|(p, s)| p.len() + s.len()).sum();

    let run = |m: usize| -> f64 {
        let x: Vec<f32> = (0..m * l).map(|j| ((j % 97) as f32 - 48.0) / 50.0).collect();
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t = Instant::now();
            experts.par_iter().for_each(|ex| {
                let (w1, w3, w2) = (&ex[0], &ex[1], &ex[2]);
                let chunks = 24;
                let mut g = vec![0f32; m * i];
                let mut u = vec![0f32; m * i];
                let step = i.div_ceil(chunks);
                for c in 0..chunks {
                    let (n0, n1) = (c * step, ((c + 1) * step).min(i));
                    fullm_kernels::gemm(
                        &x,
                        m,
                        l,
                        WeightRef::Mxfp4 { packed: &w1.0, scales: &w1.1 },
                        i,
                        n0,
                        n1,
                        &mut g,
                        i,
                    );
                    fullm_kernels::gemm(
                        &x,
                        m,
                        l,
                        WeightRef::Mxfp4 { packed: &w3.0, scales: &w3.1 },
                        i,
                        n0,
                        n1,
                        &mut u,
                        i,
                    );
                }
                let mut d = vec![0f32; m * l];
                fullm_kernels::gemm(&g, m, i, WeightRef::Mxfp4 { packed: &w2.0, scales: &w2.1 }, l, 0, l, &mut d, l);
                std::hint::black_box(&d);
            });
            best = best.min(t.elapsed().as_secs_f64());
        }
        best
    };
    let t1 = run(1);
    let t64 = run(64);
    let flops = 2.0 * 3.0 * (l * i) as f64 * 64.0 * n_exp as f64;
    ((n_exp * bytes_per_expert) as f64 / t1 / 1e9, flops / t64 / 1e12)
}

fn gpu_rates(dev: i32) -> Option<GpuRates> {
    let info = exec::device_info(dev)?;
    let (h2d, d2h) = exec::bench_copy(dev, 256 << 20, 10).unwrap_or((0.0, 0.0));
    // Kimi K3 shapes: KDA q/k/v/g [12288][7168], o [7168][12288], shared gate [6144][7168],
    // lm_head [163840][7168] (bf16), and a prefill-sized M on the cuBLAS path.
    let cases: &[(&str, Fmt, i64, i64, i64)] = &[
        ("kda q_proj", Fmt::Int8, 12288, 7168, 1),
        ("kda q_proj", Fmt::Fp8, 12288, 7168, 1),
        ("kda q_proj", Fmt::Bf16, 12288, 7168, 1),
        ("kda q_proj", Fmt::Int8, 12288, 7168, 8),
        ("kda q_proj", Fmt::Int8, 12288, 7168, 16),
        ("kda o_proj", Fmt::Int8, 7168, 12288, 1),
        ("shared gate", Fmt::Int8, 6144, 7168, 1),
        ("lm_head", Fmt::Bf16, 163840, 7168, 1),
        ("kda q_proj prefill", Fmt::Int8, 12288, 7168, 2048),
    ];
    let gemv = cases
        .iter()
        .filter_map(|&(what, fmt, n, k, m)| {
            let (gbps, ms) = exec::bench_gemm(dev, fmt, n, k, m, 20)?;
            Some(GemmRate { what: what.into(), fmt: format!("{fmt:?}"), n, k, m, gbps, ms })
        })
        .collect();
    Some(GpuRates {
        name: info.name,
        total_gb: info.total_bytes as f64 / 1e9,
        sm_count: info.sm_count,
        cc: info.compute_capability,
        h2d_gbps: h2d,
        d2h_gbps: d2h,
        gemv,
    })
}

fn model_rates(dir: &std::path::Path, a: &BenchArgs) -> Result<ModelRates> {
    let backend = a.backend.clone().unwrap_or_else(BackendKind::auto);
    let opts = LoadOptions {
        trunk: a.trunk,
        max_seqs: a.batch.max(1) + 1, // + the warm-up sequence
        cache_tokens: (a.prompt + a.decode + 64) * (a.batch.max(1) + 1),
        page: 16,
        backend,
        device: a.device,
        state_dtype: a.state_dtype,
        kv_dtype: a.kv_dtype,
        step_tokens: a.step_tokens,
        max_rows: a.batch.max(64),
        max_context: a.prompt + a.decode + 64,
        threads: a.threads,
        layers: a.layers,
        router_weights_biased: false,
        gpu_experts: a.gpu_experts,
        gpu_expert_format: a.gpu_expert_format,
        int4_checkpoint: a.int4_checkpoint.clone(),
        lora_merge: a.lora_merge.clone(),
        max_loras: 0,
        max_lora_rank: 0,
        head: a.head,
        no_cuda_graphs: a.no_cuda_graphs,
        bf16_activations: a.activations == crate::models::Activations::Bf16,
        no_prefix_cache: true,
        host_cache_tokens: 0,
        host_copy_lag: 0,
        spec_model: None,
    };
    let t = Instant::now();
    let mut m = models::load(dir, &opts).with_context(|| format!("loading {}", dir.display()))?;
    let load_s = t.elapsed().as_secs_f64();
    let v = m.vocab();
    // A different prompt per sequence: identical ones would decode identical tokens and route
    // every sequence to the same experts, making a batch look as cheap as one sequence.
    let text_ids = match &a.text {
        Some(f) => {
            let tok = crate::tokenizer::load_tokenizer(dir, v)?;
            let ids = tok.encode(&std::fs::read_to_string(f).with_context(|| format!("reading {}", f.display()))?);
            anyhow::ensure!(ids.len() > a.prompt, "{}: fewer than {} tokens", f.display(), a.prompt + 1);
            Some(ids)
        }
        None => None,
    };
    let prompt_of = |s: usize| -> Vec<u32> {
        match &text_ids {
            // stretches spread over the text, wrapping
            Some(ids) => (0..a.prompt).map(|i| ids[(s * 997 + i) % ids.len()]).collect(),
            None => (0..a.prompt).map(|i| ((i * 7919 + 13 + s * 104_729) % v) as u32).collect(),
        }
    };

    let seqs: Vec<usize> =
        (0..a.batch.max(1)).map(|_| m.alloc_seq().context("no free sequence slot")).collect::<Result<_>>()?;
    // A warm-up prefill on a scratch sequence first: the first forward pays one-time costs (the
    // lazy loading of every kernel's module, cuBLAS heuristics, scratch allocation: ~60 ms for an
    // 8000-token step on an H100) that a server pays once at startup, not per request.
    let t = Instant::now();
    {
        let w = m.alloc_seq().context("no free sequence slot for the warm-up")?;
        let prompt = prompt_of(a.batch.max(1));
        let mut pos = 0;
        for c in prompt.chunks(a.step_tokens) {
            m.forward(&[SeqChunk { seq: w, pos, tokens: c, all_logits: false, draft: 0 }])?;
            pos += c.len();
        }
        m.free_seq(w);
    }
    let warmup_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let mut last = Vec::new();
    for &s in &seqs {
        let mut pos = 0;
        let prompt = prompt_of(s);
        for c in prompt.chunks(a.step_tokens) {
            last.push(m.forward(&[SeqChunk { seq: s, pos, tokens: c, all_logits: false, draft: 0 }])?.remove(0));
            pos += c.len();
        }
    }
    let prefill_s = t.elapsed().as_secs_f64();

    let mut next: Vec<[u32; 1]> = seqs.iter().map(|&s| [((s * 31 + 1) % v) as u32]).collect();
    let t = Instant::now();
    for step in 0..a.decode {
        let batch: Vec<SeqChunk> = seqs
            .iter()
            .zip(&next)
            .map(|(&s, tok)| SeqChunk { seq: s, pos: a.prompt + step, tokens: tok, all_logits: false, draft: 0 })
            .collect();
        // greedy, as the server runs it: the argmax on the device
        let toks = m.forward_greedy(&batch)?;
        drop(batch);
        for (n, t) in next.iter_mut().zip(toks) {
            n[0] = t;
        }
    }
    let dec_s = t.elapsed().as_secs_f64();
    let st = m.last_step();
    for s in seqs {
        m.free_seq(s);
    }
    let steps = a.decode.max(1);
    Ok(ModelRates {
        description: m.describe(),
        load_s,
        warmup_s,
        prefill_tokens: a.prompt * a.batch.max(1),
        prefill_s,
        prefill_tok_s: (a.prompt * a.batch.max(1)) as f64 / prefill_s,
        batch: a.batch.max(1),
        decode_steps: a.decode,
        decode_ms_per_step: dec_s * 1e3 / steps as f64,
        decode_tok_s_total: (a.decode * a.batch.max(1)) as f64 / dec_s,
        last_step_ms: st.map(|s| s.ms_total),
        last_step_experts_ms: st.map(|s| s.ms_experts),
        last_step_gpu_wait_ms: st.map(|s| s.ms_gpu_wait),
        last_step_expert_gb: st.map(|s| s.expert_bytes as f64 / 1e9),
        last_step_distinct_experts: st.map(|s| s.distinct_experts),
    })
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

fn render(r: &Report) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    if let Some(h) = &r.host {
        let _ = writeln!(s, "host: {:.0} GB RAM, {} threads, {}", h.ram_gb, h.threads, h.cpu_isa);
        let _ = writeln!(s, "  DRAM read                  {:>8.1} GB/s   (planner --dram-gbps)", h.dram_read_gbps);
        let _ = writeln!(s, "  MXFP4 experts, decode      {:>8.1} GB/s of expert weights", h.expert_gbps_decode);
        let _ = writeln!(
            s,
            "  MXFP4 experts, 64 tok/exp  {:>8.2} TFLOP/s  (planner --cpu-tflops)",
            h.expert_tflops_prefill
        );
    }
    if let Some(g) = &r.gpu {
        let _ = writeln!(s, "gpu: {} ({:.0} GB, {} SMs, sm_{}{})", g.name, g.total_gb, g.sm_count, g.cc.0, g.cc.1);
        let _ = writeln!(s, "  pinned H2D {:.1} GB/s, D2H {:.1} GB/s   (planner --pcie-gbps)", g.h2d_gbps, g.d2h_gbps);
        for c in &g.gemv {
            let _ = writeln!(
                s,
                "  {:<20} {:>5} [{}x{}] M={:<5} {:>8.1} GB/s  {:>8.3} ms",
                c.what, c.fmt, c.n, c.k, c.m, c.gbps, c.ms
            );
        }
    }
    if let Some(m) = &r.model {
        let _ = writeln!(s, "model: {}", m.description);
        let _ = writeln!(
            s,
            "  load {:.1} s, warm-up {:.2} s (a cold first forward of {} tokens)",
            m.load_s,
            m.warmup_s,
            m.prefill_tokens / m.batch
        );
        let _ =
            writeln!(s, "  prefill {} tokens in {:.2} s = {:.1} tok/s", m.prefill_tokens, m.prefill_s, m.prefill_tok_s);
        let _ = writeln!(
            s,
            "  decode batch {} x {} steps: {:.1} ms/step = {:.2} tok/s total",
            m.batch, m.decode_steps, m.decode_ms_per_step, m.decode_tok_s_total
        );
        if let (Some(t), Some(e), Some(w), Some(gb), Some(n)) = (
            m.last_step_ms,
            m.last_step_experts_ms,
            m.last_step_gpu_wait_ms,
            m.last_step_expert_gb,
            m.last_step_distinct_experts,
        ) {
            let _ = writeln!(
                s,
                "  last step {t:.1} ms: CPU experts {e:.1} ms ({n} experts, {gb:.2} GB, {:.0} GB/s), waiting on GPU {w:.1} ms",
                if e > 0.0 { gb / (e / 1e3) } else { 0.0 }
            );
        }
    }
    s
}

// ------------------------------------------------------------------------------- compare

/// `fullm compare`: the same prompt through two backends, logits side by side. With
/// `--layers N` on a partial download this checks the GPU kernels at Kimi K3's real
/// dimensions against the Rust reference without the full 1.56 TB.
#[derive(clap::Args, Debug, Clone)]
pub struct CompareArgs {
    #[arg(long)]
    pub model: PathBuf,
    /// The two backends to compare.
    #[arg(long, default_value = "reference")]
    pub a: BackendKind,
    #[arg(long, default_value = "cuda")]
    pub b: BackendKind,
    #[arg(long, default_value = "bf16", value_parser = parse_precision)]
    pub trunk: Precision,
    #[arg(long, value_enum, default_value = "f32")]
    pub state_dtype: CacheDtype,
    #[arg(long, value_enum, default_value = "f32")]
    pub kv_dtype: CacheDtype,
    #[arg(long)]
    pub layers: Option<usize>,
    /// Routed experts resident in GPU memory: off, auto (as many as fit) or a size in GB.
    #[arg(long, default_value = "auto", value_parser = crate::models::parse_gpu_experts)]
    pub gpu_experts: crate::models::GpuExperts,
    /// Storage of GPU-resident experts.
    #[arg(long, default_value = "bf16", value_parser = parse_precision)]
    pub gpu_expert_format: Precision,
    /// With --gpu-expert-format int4: the routed experts from this GPTQ checkpoint.
    #[arg(long)]
    pub int4_checkpoint: Option<std::path::PathBuf>,
    /// A LoRA adapter (PEFT's format) merged into the weights at load.
    #[arg(long)]
    pub lora_merge: Option<std::path::PathBuf>,
    /// Run every step op by op instead of replaying CUDA graphs of small steps.
    #[arg(long)]
    pub no_cuda_graphs: bool,
    /// Activations of small-batch GEMVs: bf16 (as vLLM; faster) or exact (fp32).
    #[arg(long, value_enum, default_value = "bf16")]
    pub activations: crate::models::Activations,
    /// Prompt token ids; default: a fixed 12-token prompt.
    #[arg(long, value_delimiter = ',')]
    pub ids: Option<Vec<u32>>,
    /// Greedy decode steps after the prompt.
    #[arg(long, default_value_t = 4)]
    pub decode: usize,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    #[arg(long, default_value_t = 0)]
    pub threads: usize,
}

pub fn compare(a: &CompareArgs) -> Result<()> {
    let ids: Vec<u32> = a.ids.clone().unwrap_or_else(|| vec![3, 7, 11, 5, 9, 200, 17, 4, 99, 150, 23, 61]);
    let opts = |backend| LoadOptions {
        trunk: a.trunk,
        max_seqs: 1,
        cache_tokens: ids.len() + a.decode + 64,
        page: 16,
        backend,
        device: a.device,
        state_dtype: a.state_dtype,
        kv_dtype: a.kv_dtype,
        step_tokens: ids.len().max(16),
        max_rows: 64,
        max_context: ids.len() + a.decode + 64,
        threads: a.threads,
        layers: a.layers,
        router_weights_biased: false,
        gpu_experts: a.gpu_experts,
        gpu_expert_format: a.gpu_expert_format,
        int4_checkpoint: a.int4_checkpoint.clone(),
        lora_merge: a.lora_merge.clone(),
        max_loras: 0,
        max_lora_rank: 0,
        head: None,
        no_cuda_graphs: a.no_cuda_graphs,
        bf16_activations: a.activations == crate::models::Activations::Bf16,
        no_prefix_cache: true,
        host_cache_tokens: 0,
        host_copy_lag: 0,
        spec_model: None,
    };
    // One model at a time: two copies of a large partial model may not fit together.
    let run = |backend: BackendKind| -> Result<(Vec<Vec<f32>>, Vec<u32>, f64)> {
        let t = Instant::now();
        let mut m = models::load(&a.model, &opts(backend.clone())).with_context(|| format!("loading on {backend}"))?;
        eprintln!("{backend:?}: loaded in {:.1}s: {}", t.elapsed().as_secs_f64(), m.describe());
        let s = m.alloc_seq().context("no slot")?;
        let t = Instant::now();
        let mut rows =
            vec![m.forward(&[SeqChunk { seq: s, pos: 0, tokens: &ids, all_logits: false, draft: 0 }])?.remove(0)];
        let mut toks = Vec::new();
        for i in 0..a.decode {
            let next = argmax(rows.last().unwrap());
            toks.push(next);
            rows.push(
                m.forward(&[SeqChunk { seq: s, pos: ids.len() + i, tokens: &[next], all_logits: false, draft: 0 }])?
                    .remove(0),
            );
        }
        Ok((rows, toks, t.elapsed().as_secs_f64()))
    };
    let (ra, ta, sa) = run(a.a.clone())?;
    let (rb, tb, sb) = run(a.b.clone())?;
    println!("{:?}: {:.2}s, greedy {:?}", a.a, sa, ta);
    println!("{:?}: {:.2}s, greedy {:?}", a.b, sb, tb);
    let mut worst = 0.0f32;
    for (i, (x, y)) in ra.iter().zip(&rb).enumerate() {
        let num: f32 = x.iter().zip(y).map(|(p, q)| (p - q) * (p - q)).sum::<f32>().sqrt();
        let den: f32 = x.iter().map(|p| p * p).sum::<f32>().sqrt().max(1e-30);
        let top = (argmax(x), argmax(y));
        println!("  step {i}: rel-L2 {:.3e}, argmax {} vs {}", num / den, top.0, top.1);
        worst = worst.max(num / den);
    }
    println!(
        "worst rel-L2 logit difference {worst:.3e}; greedy tokens {}",
        if ta == tb { "IDENTICAL" } else { "DIFFER" }
    );
    Ok(())
}

// ------------------------------------------------------------------------------- ppl

/// `fullm ppl`: teacher-forced perplexity of a text file. Used to settle questions a logit
/// diff cannot, such as which of two plausible conventions a checkpoint was trained with.
#[derive(clap::Args, Debug, Clone)]
pub struct PplArgs {
    #[arg(long)]
    pub model: PathBuf,
    /// Plain-text file to score.
    #[arg(long)]
    pub text: PathBuf,
    #[arg(long)]
    pub backend: Option<BackendKind>,
    #[arg(long, default_value = "bf16", value_parser = parse_precision)]
    pub trunk: Precision,
    /// Storage of the lm_head: bf16 (default) or int8 / fp8 (see serve --head).
    #[arg(long, value_parser = parse_precision)]
    pub head: Option<Precision>,
    #[arg(long, value_enum, default_value = "f32")]
    pub state_dtype: CacheDtype,
    #[arg(long, value_enum, default_value = "f32")]
    pub kv_dtype: CacheDtype,
    /// Score at most this many tokens.
    #[arg(long, default_value_t = 2048)]
    pub max_tokens: usize,
    /// Tokens per forward step.
    #[arg(long, default_value_t = 256)]
    pub chunk: usize,
    /// Routed experts resident in GPU memory: off, auto (as many as fit) or a size in GB.
    #[arg(long, default_value = "auto", value_parser = crate::models::parse_gpu_experts)]
    pub gpu_experts: crate::models::GpuExperts,
    /// Storage of GPU-resident experts.
    #[arg(long, default_value = "bf16", value_parser = parse_precision)]
    pub gpu_expert_format: Precision,
    /// With --gpu-expert-format int4: the routed experts from this GPTQ checkpoint.
    #[arg(long)]
    pub int4_checkpoint: Option<std::path::PathBuf>,
    /// A LoRA adapter (PEFT's format) merged into the weights at load.
    #[arg(long)]
    pub lora_merge: Option<std::path::PathBuf>,
    /// Run every step op by op instead of replaying CUDA graphs of small steps.
    #[arg(long)]
    pub no_cuda_graphs: bool,
    /// Activations of small-batch GEMVs: bf16 (as vLLM; faster) or exact (fp32).
    #[arg(long, value_enum, default_value = "bf16")]
    pub activations: crate::models::Activations,
    /// MoE combining weights from score + bias (the released Kimi Linear HF code).
    #[arg(long)]
    pub router_weights_biased: bool,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    #[arg(long, default_value_t = 0)]
    pub threads: usize,
}

pub fn ppl(a: &PplArgs) -> Result<()> {
    let backend = a.backend.clone().unwrap_or_else(BackendKind::auto);
    let text = std::fs::read_to_string(&a.text).with_context(|| format!("reading {}", a.text.display()))?;
    let raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(a.model.join("config.json"))?)?;
    let vocab = raw.get("vocab_size").or_else(|| raw.get("text_config").and_then(|t| t.get("vocab_size")));
    let tok = crate::tokenizer::load_tokenizer(&a.model, vocab.and_then(|v| v.as_u64()).unwrap_or(0) as usize)?;
    let mut ids = tok.encode(&text);
    ids.truncate(a.max_tokens);
    anyhow::ensure!(ids.len() >= 2, "text is too short to score");
    let opts = LoadOptions {
        trunk: a.trunk,
        max_seqs: 1,
        cache_tokens: ids.len() + 64,
        page: 16,
        backend,
        device: a.device,
        state_dtype: a.state_dtype,
        kv_dtype: a.kv_dtype,
        step_tokens: a.chunk,
        max_rows: a.chunk,
        max_context: ids.len() + 64,
        threads: a.threads,
        layers: None,
        router_weights_biased: a.router_weights_biased,
        gpu_experts: a.gpu_experts,
        gpu_expert_format: a.gpu_expert_format,
        int4_checkpoint: a.int4_checkpoint.clone(),
        lora_merge: a.lora_merge.clone(),
        max_loras: 0,
        max_lora_rank: 0,
        head: a.head,
        no_cuda_graphs: a.no_cuda_graphs,
        bf16_activations: a.activations == crate::models::Activations::Bf16,
        no_prefix_cache: true,
        host_cache_tokens: 0,
        host_copy_lag: 0,
        spec_model: None,
    };
    let mut m = models::load(&a.model, &opts)?;
    let v = m.vocab();
    let s = m.alloc_seq().context("no slot")?;
    let t = Instant::now();
    let (mut nll, mut n) = (0.0f64, 0usize);
    let mut pos = 0;
    for chunk in ids.chunks(a.chunk) {
        let out = m.forward(&[SeqChunk { seq: s, pos, tokens: chunk, all_logits: true, draft: 0 }])?.remove(0);
        for i in 0..chunk.len() {
            let target = pos + i + 1;
            if target >= ids.len() {
                break;
            }
            let row = &out[i * v..(i + 1) * v];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let lse = mx + row.iter().map(|&x| ((x as f64) - mx).exp()).sum::<f64>().ln();
            nll += lse - row[ids[target] as usize] as f64;
            n += 1;
        }
        pos += chunk.len();
    }
    let mean = nll / n as f64;
    println!(
        "{} tokens scored in {:.1}s | router weights {} | mean NLL {:.4} | perplexity {:.3}",
        n,
        t.elapsed().as_secs_f64(),
        if a.router_weights_biased { "BIASED" } else { "unbiased" },
        mean,
        mean.exp()
    );
    Ok(())
}
