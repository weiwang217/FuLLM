// SPDX-License-Identifier: Apache-2.0
//! Deployment planner: where every byte lives, and what that costs per token.
//!
//! THE MODEL OF THE MACHINE
//!   Three memories (HBM, host DRAM, NVMe) and two compute sites (GPU, CPU). The trunk is
//!   re-read in full every step, so it belongs in HBM; routed experts are touched sparsely
//!   and live wherever capacity allows. A routed expert can be computed where it already
//!   sits (CPU, reading DRAM at hundreds of GB/s) or copied to the GPU over PCIe (~50
//!   GB/s). Both draw on the same DRAM bandwidth, so for bandwidth-bound decode the CPU
//!   path wins; for compute-bound prefill the GPU takes the share PCIe can feed.
//!
//! THE MODEL OF THE WORKLOAD
//!   Routing is modelled as uniform: with top-k of n per token, B tokens touch
//!   n(1-(1-k/n)^B) distinct experts per layer. Kimi K3 trains its router with Quantile
//!   Balancing precisely to make usage flat, so this is the honest model for K3. Models with
//!   skewed routing do better than this predicts (the CUDA executor places the most routed
//!   experts in GPU memory).
//!
//! What comes out is an ESTIMATE from effective rates. `fullm bench` on the target box
//! replaces the presets with measured rates; until then treat the numbers as the shape of
//! the answer (what bounds each regime, where the crossovers are), not a promise.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use serde::Serialize;

use crate::hw::{self, Gpu, Host};
use crate::models::kimi_k3::profile::K3Profile;
use crate::models::kimi_k3::K3Config;

/// Storage of the trunk's large projections on the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum TrunkMode {
    /// Weight-only int8, per-row scales. ~0.6-0.9% output error per matrix.
    Int8,
    /// E4M3, per-row scales. Same bytes as int8, ~4x the error on Gaussian-like rows.
    Fp8,
    /// The checkpoint's own bf16. Exact; overflow layers run on the CPU or stream.
    Bf16,
}

impl TrunkMode {
    fn q8_bytes(self) -> u64 {
        if self == TrunkMode::Bf16 {
            2
        } else {
            1
        }
    }
}

/// What the planner needs to know about a model. Implemented per architecture.
pub trait Profile {
    fn name(&self) -> String;
    fn n_layers(&self) -> usize;
    fn layer_bytes(&self, layer: usize, mode: TrunkMode) -> u64;
    fn layer_params(&self, layer: usize) -> u64;
    /// GPU-resident head (lm_head at bf16).
    fn head_bytes(&self) -> u64;
    fn head_params(&self) -> u64;
    /// Host-resident embedding table (one row gathered per token).
    fn embed_bytes(&self) -> u64;
    /// Fixed-size recurrent state per sequence (0 for pure-attention models).
    fn state_bytes_per_seq(&self) -> u64;
    /// Attention cache per position, all layers.
    fn kv_bytes_per_token(&self) -> u64;
    fn moe(&self) -> Option<MoeShape>;
    /// Activation workspace for a prefill chunk.
    fn workspace_bytes(&self, chunk: usize) -> u64;
    /// Layers whose attention is a sequential recurrence (cost per position in prefill v1).
    fn recurrent_layers(&self) -> usize;
    /// Softmax-attention FLOPs for one query token attending over `ctx` positions.
    fn attn_flops(&self, ctx: f64) -> f64;
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct MoeShape {
    pub layers: usize,
    pub experts: usize,
    pub topk: usize,
    pub expert_bytes: u64,
    pub expert_flops: u64,
}

impl MoeShape {
    fn unique(&self, tokens: f64) -> f64 {
        let n = self.experts as f64;
        n * (1.0 - (1.0 - self.topk as f64 / n).powf(tokens))
    }
    fn all_bytes(&self) -> u64 {
        (self.layers * self.experts) as u64 * self.expert_bytes
    }
}

pub struct K3 {
    pub cfg: K3Config,
    pub p: K3Profile,
}

impl Profile for K3 {
    fn name(&self) -> String {
        let total = self.p.always_active_params() + self.p.total_experts() * self.p.expert_params;
        let family =
            if self.cfg.family == crate::models::kimi_k3::config::Family::Linear { "Kimi Linear" } else { "Kimi K3" };
        if total >= 1_000_000_000_000 {
            format!("{family} ({:.2}T params, {} layers)", total as f64 / 1e12, self.cfg.n_layers)
        } else {
            format!("{family} ({:.1}B params, {} layers)", total as f64 / 1e9, self.cfg.n_layers)
        }
    }
    fn n_layers(&self) -> usize {
        self.cfg.n_layers
    }
    fn layer_bytes(&self, l: usize, mode: TrunkMode) -> u64 {
        self.p.layer_bytes(l, mode.q8_bytes())
    }
    fn layer_params(&self, l: usize) -> u64 {
        self.p.layers[l].total()
    }
    fn head_bytes(&self) -> u64 {
        2 * if self.p.lm_head_params > 0 { self.p.lm_head_params } else { self.p.embed_params }
    }
    fn head_params(&self) -> u64 {
        self.head_bytes() / 2
    }
    fn embed_bytes(&self) -> u64 {
        2 * self.p.embed_params
    }
    fn state_bytes_per_seq(&self) -> u64 {
        self.p.kda_state_bytes
    }
    fn kv_bytes_per_token(&self) -> u64 {
        2 * self.p.mla_floats_per_token // bf16 latent + rope slot
    }
    fn moe(&self) -> Option<MoeShape> {
        Some(MoeShape {
            layers: self.p.n_moe_layers,
            experts: self.p.n_experts,
            topk: self.p.topk,
            expert_bytes: self.p.expert_bytes,
            expert_flops: self.p.expert_flops,
        })
    }
    fn workspace_bytes(&self, chunk: usize) -> u64 {
        let c = &self.cfg;
        let n = chunk as u64;
        let act = n * c.hidden as u64 * 4 * (8 + c.max_sources() as u64);
        let kda = n * c.kda_proj() as u64 * 4 * 7;
        let dense = n * c.dense_inter as u64 * 4 * 3;
        let dequant = (c.dense_inter * c.hidden) as u64 * 2; // largest 8-bit matrix, widened
        let staging = 2 * 64 * self.p.expert_bytes; // double-buffered GPU-side experts
        act + kda.max(dense) + dequant + staging
    }
    fn recurrent_layers(&self) -> usize {
        self.cfg.n_kda()
    }
    fn attn_flops(&self, ctx: f64) -> f64 {
        // absorbed MLA: scores over latent+rope (576), values in latent space (512)
        let c = &self.cfg;
        let per_pos = 2.0 * (c.mla_cache_width() + c.kv_lora) as f64;
        (c.n_mla() * c.n_heads) as f64 * ctx * per_pos
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Placement {
    pub mode: TrunkMode,
    pub gpu_layers: usize,
    pub cpu_layers: usize,
    pub gpu_weights_gb: f64,
    pub cpu_trunk_gb: f64,
    pub workspace_gb: f64,
    pub state_slots: usize,
    pub state_gb: f64,
    pub kv_tokens: u64,
    pub kv_gb: f64,
    pub hbm_free_gb: f64,
    pub experts_total_gb: f64,
    pub experts_in_dram_gb: f64,
    pub experts_dram_frac: f64,
    pub host_free_gb: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct DecodeRow {
    pub batch: usize,
    pub unique_per_layer: f64,
    pub expert_gb: f64,
    pub gpu_ms: f64,
    pub cpu_trunk_ms: f64,
    pub expert_ms: f64,
    pub step_ms: f64,
    pub tok_s_per_seq: f64,
    pub tok_s_total: f64,
    pub gpu_share_of_experts: f64,
    pub bound: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct PrefillRow {
    pub tokens: usize,
    pub gpu_s: f64,
    pub expert_s: f64,
    pub total_s: f64,
    pub tok_s: f64,
    pub gpu_share_of_experts: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Plan {
    pub model: String,
    pub gpu: Gpu,
    pub host: Host,
    pub placement: Placement,
    pub decode: Vec<DecodeRow>,
    pub prefill: Vec<PrefillRow>,
    pub notes: Vec<String>,
}

pub struct Request {
    pub mode: TrunkMode,
    pub max_seqs: Option<usize>,
    /// Average context the decode estimate assumes.
    pub context: usize,
    pub prefill_chunk: usize,
    pub prefill_tokens: Vec<usize>,
    pub batches: Vec<usize>,
}

const GB: f64 = 1e9;
/// Per-step GPU fixed cost with CUDA graphs: ~2k kernels at ~1.5 us.
const GPU_STEP_OVERHEAD_MS: f64 = 3.0;
/// One GPU->CPU->GPU handoff per MoE layer per step (through pinned host memory).
const HANDOFF_MS: f64 = 0.02;
/// Sequential KDA recurrence in prefill v1, per position per layer.
const RECURRENT_US_PER_POS: f64 = 2.0;

pub fn plan(m: &dyn Profile, gpu: &Gpu, host: &Host, rq: &Request) -> Result<Plan> {
    let mut notes = Vec::new();
    let nl = m.n_layers();

    // ---- HBM: workspace and head first, then as many trunk layers as fit, then state ----
    let workspace = m.workspace_bytes(rq.prefill_chunk) as f64;
    let min_ctx = (2.0 * m.state_bytes_per_seq() as f64 + 64.0 * 1024.0 * m.kv_bytes_per_token() as f64).max(1.0 * GB);
    let hbm = gpu.hbm_gb * GB - gpu.reserved_gb * GB;
    let mut budget = hbm - workspace - m.head_bytes() as f64 - min_ctx;
    if budget <= 0.0 {
        bail!("{} cannot hold even the workspace and head of {}", gpu.name, m.name());
    }
    let mut gpu_layers = 0;
    let mut gpu_w = m.head_bytes() as f64;
    let mut cpu_trunk = 0.0;
    let mut cpu_params = 0u64;
    let mut gpu_params = m.head_params();
    for l in 0..nl {
        let b = m.layer_bytes(l, rq.mode) as f64;
        if b <= budget && gpu_layers == l {
            budget -= b;
            gpu_w += b;
            gpu_layers += 1;
            gpu_params += m.layer_params(l);
        } else {
            cpu_trunk += b;
            cpu_params += m.layer_params(l);
        }
    }
    let cpu_layers = nl - gpu_layers;
    if cpu_layers > 0 && rq.mode != TrunkMode::Bf16 {
        notes.push(format!("{cpu_layers} layers do not fit in HBM even at 8 bits; they run on the CPU"));
    }
    if cpu_layers > 0 && rq.mode == TrunkMode::Bf16 {
        notes.push(format!(
            "exact bf16: {gpu_layers}/{nl} layers resident on the GPU, {cpu_layers} ({:.1} GB) computed on the CPU from DRAM",
            cpu_trunk / GB
        ));
    }

    // ---- recurrent state slots and paged KV with what is left ----
    let rest = hbm - workspace - gpu_w;
    let state = m.state_bytes_per_seq() as f64;
    let kvt = m.kv_bytes_per_token() as f64;
    let slots = match rq.max_seqs {
        Some(s) => s,
        None => {
            // half of what is left to state, capped where batching stops paying for itself
            let s = if state > 0.0 { ((rest * 0.5) / state).floor() as usize } else { 256 };
            s.clamp(1, 64)
        }
    };
    let state_gb = slots as f64 * state;
    if state_gb > rest {
        bail!(
            "{slots} sequence slots need {:.1} GB of state but only {:.1} GB of HBM is left",
            state_gb / GB,
            rest / GB
        );
    }
    let kv_tokens = ((rest - state_gb) / kvt).floor().max(0.0) as u64;
    if (kv_tokens as f64) < rq.context as f64 {
        notes.push(format!("KV pool holds {kv_tokens} positions, below one {}-token context", rq.context));
    }

    // ---- host DRAM: embed, CPU trunk layers, pinned staging, then experts ----
    let moe = m.moe();
    let staging = 4.0 * GB;
    let host_avail = host.ram_gb * GB - host.os_reserve_gb * GB - m.embed_bytes() as f64 - cpu_trunk - staging;
    if host_avail < 0.0 {
        bail!("{} has too little RAM for the embedding, CPU-side trunk and staging", host.name);
    }
    let (exp_total, exp_dram) = match moe {
        Some(mo) => {
            let t = mo.all_bytes() as f64;
            (t, t.min(host_avail))
        }
        None => (0.0, 0.0),
    };
    let dram_frac = if exp_total > 0.0 { exp_dram / exp_total } else { 1.0 };
    if moe.is_some() && dram_frac < 1.0 {
        notes.push(format!(
            "{:.0}% of routed experts fit in DRAM; the rest stream from NVMe at {:.0} GB/s on every miss",
            dram_frac * 100.0,
            host.nvme_gbps
        ));
    }

    let placement = Placement {
        mode: rq.mode,
        gpu_layers,
        cpu_layers,
        gpu_weights_gb: gpu_w / GB,
        cpu_trunk_gb: cpu_trunk / GB,
        workspace_gb: workspace / GB,
        state_slots: slots,
        state_gb: state_gb / GB,
        kv_tokens,
        kv_gb: kv_tokens as f64 * kvt / GB,
        hbm_free_gb: (rest - state_gb - kv_tokens as f64 * kvt) / GB,
        experts_total_gb: exp_total / GB,
        experts_in_dram_gb: exp_dram / GB,
        experts_dram_frac: dram_frac,
        host_free_gb: (host_avail - exp_dram) / GB,
    };

    // ---- decode ----
    let mut decode = Vec::new();
    for &b in rq.batches.iter().filter(|&&b| b <= slots) {
        let bf = b as f64;
        let mem_ms = (gpu_w + bf * 2.0 * state + bf * rq.context as f64 * kvt) / (gpu.hbm_gbps * GB) * 1e3;
        let flop_ms = 2.0 * gpu_params as f64 * bf / (gpu.bf16_tflops * 1e12) * 1e3;
        let gpu_ms = mem_ms.max(flop_ms) + GPU_STEP_OVERHEAD_MS;

        // CPU-side trunk layers (exact mode): read from DRAM every step, or stream over PCIe
        let cpu_trunk_ms = if cpu_layers > 0 {
            let on_cpu =
                (cpu_trunk / (host.dram_gbps * GB)).max(2.0 * cpu_params as f64 * bf / (host.cpu_tflops * 1e12)) * 1e3;
            let streamed = cpu_trunk / (gpu.pcie_gbps * GB) * 1e3;
            on_cpu.min(streamed)
        } else {
            0.0
        };

        let (uniq, exp_gb, exp_ms, share) = match moe {
            Some(mo) => {
                let u = mo.unique(bf);
                let bytes = mo.layers as f64 * u * mo.expert_bytes as f64;
                let flops = bf * (mo.topk * mo.layers) as f64 * mo.expert_flops as f64;
                let (t, g) = expert_time(bytes, flops, dram_frac, host, gpu, cpu_trunk_ms / 1e3);
                (u, bytes / GB, t * 1e3, g)
            }
            None => (0.0, 0.0, 0.0, 0.0),
        };
        let handoff = moe.map_or(0.0, |mo| mo.layers as f64 * HANDOFF_MS);
        let step = gpu_ms + exp_ms + cpu_trunk_ms + handoff;
        let bound = if exp_ms >= gpu_ms.max(cpu_trunk_ms) {
            if dram_frac < 1.0 {
                "experts (NVMe)"
            } else {
                "experts (DRAM)"
            }
        } else if cpu_trunk_ms > gpu_ms {
            "CPU trunk layers"
        } else {
            "GPU trunk"
        };
        decode.push(DecodeRow {
            batch: b,
            unique_per_layer: uniq,
            expert_gb: exp_gb,
            gpu_ms,
            cpu_trunk_ms,
            expert_ms: exp_ms,
            step_ms: step,
            tok_s_per_seq: 1e3 / step,
            tok_s_total: bf * 1e3 / step,
            gpu_share_of_experts: share,
            bound,
        });
    }

    // ---- prefill (one sequence, chunked) ----
    let mut prefill = Vec::new();
    for &t in &rq.prefill_tokens {
        let mut gpu_s = 0.0;
        let mut exp_s = 0.0;
        let mut share_acc = 0.0;
        let mut done = 0usize;
        while done < t {
            let c = rq.prefill_chunk.min(t - done);
            let cf = c as f64;
            let ctx = done as f64 + cf / 2.0;
            // trunk: compute-bound GEMMs, weights read once per chunk, attention over context
            let mut g = (2.0 * gpu_params as f64 * cf / (gpu.bf16_tflops * 1e12)).max(gpu_w / (gpu.hbm_gbps * GB));
            g += cf * m.attn_flops(ctx) / (gpu.bf16_tflops * 1e12);
            g += cf * m.recurrent_layers() as f64 * RECURRENT_US_PER_POS * 1e-6;
            if cpu_layers > 0 {
                g += cpu_trunk / (gpu.pcie_gbps * GB); // overflow layers streamed per chunk
            }
            gpu_s += g;
            if let Some(mo) = moe {
                let bytes = mo.layers as f64 * mo.unique(cf) * mo.expert_bytes as f64;
                let flops = cf * (mo.topk * mo.layers) as f64 * mo.expert_flops as f64;
                let (e, share) = expert_time(bytes, flops, dram_frac, host, gpu, 0.0);
                exp_s += e;
                share_acc += share * cf;
            }
            done += c;
        }
        let total = gpu_s + exp_s;
        prefill.push(PrefillRow {
            tokens: t,
            gpu_s,
            expert_s: exp_s,
            total_s: total,
            tok_s: t as f64 / total,
            gpu_share_of_experts: share_acc / t as f64,
        });
    }

    if moe.is_some() {
        notes.push("decode assumes uniform routing (K3's Quantile Balancing); an expert LRU adds little until it holds most experts".into());
    }
    if m.recurrent_layers() > 0 {
        notes.push(format!(
            "prefill assumes the v1 sequential recurrence (~{RECURRENT_US_PER_POS} us/position/layer); the chunked KDA kernel removes most of that term"
        ));
    }
    Ok(Plan { model: m.name(), gpu: gpu.clone(), host: host.clone(), placement, decode, prefill, notes })
}

/// Best split of one step's routed-expert work between CPU (in place) and GPU (over PCIe).
///
/// `bytes` are distinct expert bytes touched, `flops` the routed-expert FLOPs; a fraction
/// `dram_frac` of the bytes is DRAM-resident, the rest arrives from NVMe into DRAM first.
/// `dram_busy_s` is DRAM time already spoken for by CPU trunk layers this step. Returns
/// (seconds, GPU share of the work).
fn expert_time(bytes: f64, flops: f64, dram_frac: f64, host: &Host, gpu: &Gpu, dram_busy_s: f64) -> (f64, f64) {
    let res = bytes * dram_frac;
    let miss = bytes - res;
    let nvme = miss / (host.nvme_gbps * GB);
    let mut best = (f64::INFINITY, 0.0);
    for i in 0..=20 {
        let g = i as f64 / 20.0;
        // CPU computes (1-g) of the work in place; GPU receives g of the bytes over PCIe.
        let cpu = (1.0 - g) * flops / (host.cpu_tflops * 1e12);
        let pcie = g * bytes / (gpu.pcie_gbps * GB);
        // every expert byte is read from DRAM once (by a core or by the DMA engine);
        // NVMe misses are also written into DRAM first
        let dram = (bytes + miss) / (host.dram_gbps * GB) + dram_busy_s;
        let t = cpu.max(pcie).max(dram).max(nvme);
        if t < best.0 - 1e-12 {
            best = (t, g);
        }
    }
    best
}

// ------------------------------------------------------------------------------- CLI

#[derive(clap::Args, Debug, Clone)]
pub struct PlanArgs {
    /// Model directory with config.json; omit to plan the reference Kimi K3.
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// GPU preset: h100-sxm, h100-pcie, h100-nvl, h200, a100-80.
    #[arg(long, default_value = "h100-sxm")]
    pub gpu: String,
    /// Host preset: genoa-2t, spr-2t, ws-512g, cloud-1gpu, or "probe" to measure this machine.
    #[arg(long, default_value = "genoa-2t")]
    pub host: String,
    /// Trunk storage on the GPU.
    #[arg(long, value_enum, default_value = "int8")]
    pub trunk: TrunkMode,
    #[arg(long)]
    pub host_ram_gb: Option<f64>,
    #[arg(long)]
    pub dram_gbps: Option<f64>,
    #[arg(long)]
    pub cpu_tflops: Option<f64>,
    #[arg(long)]
    pub nvme_gbps: Option<f64>,
    #[arg(long)]
    pub pcie_gbps: Option<f64>,
    #[arg(long)]
    pub hbm_gb: Option<f64>,
    /// Sequence slots (recurrent state in HBM). Default: sized from free HBM.
    #[arg(long)]
    pub max_seqs: Option<usize>,
    /// Average context length assumed for decode.
    #[arg(long, default_value_t = 8192)]
    pub context: usize,
    /// Prefill tokens per layer-major pass. Every pass re-reads (nearly) all routed experts
    /// once, so larger passes amortise the 1.45 TB sweep; the cost is HBM workspace.
    #[arg(long, default_value_t = 8192)]
    pub prefill_chunk: usize,
    /// Print the plan as JSON.
    #[arg(long)]
    pub json: bool,
}

pub fn run(args: &PlanArgs) -> Result<()> {
    let cfg = match &args.model {
        Some(dir) => {
            let raw = std::fs::read_to_string(dir.join("config.json"))
                .with_context(|| format!("reading {}/config.json", dir.display()))?;
            K3Config::from_json(&serde_json::from_str(&raw)?)?
        }
        None => K3Config::reference(),
    };
    let model = K3 { p: K3Profile::new(&cfg), cfg };
    let mut gpu = hw::gpu_preset(&args.gpu)
        .with_context(|| format!("unknown GPU {:?}; presets: {}", args.gpu, hw::GPU_PRESETS.join(", ")))?;
    let mut host = if args.host == "probe" {
        hw::probe_host()
    } else {
        hw::host_preset(&args.host)
            .with_context(|| format!("unknown host {:?}; presets: {}, probe", args.host, hw::HOST_PRESETS.join(", ")))?
    };
    if let Some(v) = args.hbm_gb {
        gpu.hbm_gb = v;
    }
    if let Some(v) = args.pcie_gbps {
        gpu.pcie_gbps = v;
    }
    if let Some(v) = args.host_ram_gb {
        host.ram_gb = v;
    }
    if let Some(v) = args.dram_gbps {
        host.dram_gbps = v;
    }
    if let Some(v) = args.cpu_tflops {
        host.cpu_tflops = v;
    }
    if let Some(v) = args.nvme_gbps {
        host.nvme_gbps = v;
    }
    let rq = Request {
        mode: args.trunk,
        max_seqs: args.max_seqs,
        context: args.context,
        prefill_chunk: args.prefill_chunk,
        prefill_tokens: vec![1024, 8192, 32768],
        batches: vec![1, 2, 4, 8, 16, 32, 64],
    };
    let p = plan(&model, &gpu, &host, &rq)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&p)?);
    } else {
        print!("{}", render(&p));
    }
    Ok(())
}

pub fn render(p: &Plan) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let pl = &p.placement;
    let _ = writeln!(s, "{}", p.model);
    let _ = writeln!(
        s,
        "  GPU   {}: {:.0} GB HBM @ {:.0} GB/s, {:.0} TFLOPS bf16, PCIe {:.0} GB/s",
        p.gpu.name, p.gpu.hbm_gb, p.gpu.hbm_gbps, p.gpu.bf16_tflops, p.gpu.pcie_gbps
    );
    let _ = writeln!(
        s,
        "  host  {}: {:.0} GB DRAM @ {:.0} GB/s, {:.1} TFLOPS on experts, NVMe {:.0} GB/s",
        p.host.name, p.host.ram_gb, p.host.dram_gbps, p.host.cpu_tflops, p.host.nvme_gbps
    );
    let _ = writeln!(s, "\nPlacement (trunk {:?})", pl.mode);
    let _ = writeln!(
        s,
        "  HBM   weights {:.1} GB ({} of {} layers + lm_head) | workspace {:.1} | state {} x {:.2} = {:.1} | KV {:.1} GB = {} positions | free {:.1}",
        pl.gpu_weights_gb,
        pl.gpu_layers,
        pl.gpu_layers + pl.cpu_layers,
        pl.workspace_gb,
        pl.state_slots,
        if pl.state_slots > 0 { pl.state_gb / pl.state_slots as f64 } else { 0.0 },
        pl.state_gb,
        pl.kv_gb,
        pl.kv_tokens,
        pl.hbm_free_gb
    );
    let _ = writeln!(
        s,
        "  DRAM  experts {:.0} of {:.0} GB resident ({:.0}%) | CPU trunk layers {:.1} GB | spare {:.0} GB",
        pl.experts_in_dram_gb,
        pl.experts_total_gb,
        pl.experts_dram_frac * 100.0,
        pl.cpu_trunk_gb,
        pl.host_free_gb
    );
    let _ = writeln!(s, "\nDecode (avg context per sequence as given)");
    let _ = writeln!(s, "  batch  uniq/layer  expert GB   GPU ms  CPU-trunk ms  experts ms   step ms  tok/s/seq  tok/s total  GPU share  bound");
    for r in &p.decode {
        let _ = writeln!(
            s,
            "  {:>5}  {:>10.0}  {:>9.1}  {:>7.1}  {:>12.1}  {:>10.1}  {:>8.1}  {:>9.2}  {:>11.1}  {:>8.0}%  {}",
            r.batch,
            r.unique_per_layer,
            r.expert_gb,
            r.gpu_ms,
            r.cpu_trunk_ms,
            r.expert_ms,
            r.step_ms,
            r.tok_s_per_seq,
            r.tok_s_total,
            r.gpu_share_of_experts * 100.0,
            r.bound
        );
    }
    let _ = writeln!(s, "\nPrefill (one sequence)");
    let _ = writeln!(s, "  tokens    GPU s  experts s   total s     tok/s  GPU share of experts");
    for r in &p.prefill {
        let _ = writeln!(
            s,
            "  {:>6}  {:>7.1}  {:>9.1}  {:>8.1}  {:>8.0}  {:>8.0}%",
            r.tokens,
            r.gpu_s,
            r.expert_s,
            r.total_s,
            r.tok_s,
            r.gpu_share_of_experts * 100.0
        );
    }
    if !p.notes.is_empty() {
        let _ = writeln!(s, "\nNotes");
        for n in &p.notes {
            let _ = writeln!(s, "  - {n}");
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k3() -> K3 {
        let cfg = K3Config::reference();
        K3 { p: K3Profile::new(&cfg), cfg }
    }

    fn rq(mode: TrunkMode) -> Request {
        Request {
            mode,
            max_seqs: None,
            context: 8192,
            prefill_chunk: 2048,
            prefill_tokens: vec![8192],
            batches: vec![1, 8, 32],
        }
    }

    #[test]
    fn int8_trunk_is_fully_resident_on_h100() {
        let p = plan(
            &k3(),
            &hw::gpu_preset("h100-sxm").unwrap(),
            &hw::host_preset("genoa-2t").unwrap(),
            &rq(TrunkMode::Int8),
        )
        .unwrap();
        assert_eq!(p.placement.cpu_layers, 0);
        assert!(p.placement.gpu_weights_gb < 62.0, "{}", p.placement.gpu_weights_gb);
        assert_eq!(p.placement.experts_dram_frac, 1.0);
        // B=1 decode on a DRAM-resident host is bound by expert bytes, not the GPU
        assert_eq!(p.decode[0].bound, "experts (DRAM)");
        assert!((p.decode[0].expert_gb - 25.83).abs() < 0.05);
    }

    #[test]
    fn exact_bf16_overflows_h100_but_fits_h200() {
        let host = hw::host_preset("genoa-2t").unwrap();
        let p = plan(&k3(), &hw::gpu_preset("h100-sxm").unwrap(), &host, &rq(TrunkMode::Bf16)).unwrap();
        assert!(p.placement.cpu_layers > 20, "bf16 trunk cannot be resident in 80 GB");
        let p = plan(&k3(), &hw::gpu_preset("h200").unwrap(), &host, &rq(TrunkMode::Bf16)).unwrap();
        assert_eq!(p.placement.cpu_layers, 0, "141 GB holds the bf16 trunk");
    }

    #[test]
    fn small_host_streams_from_nvme() {
        let p = plan(
            &k3(),
            &hw::gpu_preset("h100-sxm").unwrap(),
            &hw::host_preset("cloud-1gpu").unwrap(),
            &rq(TrunkMode::Int8),
        )
        .unwrap();
        assert!(p.placement.experts_dram_frac < 0.15);
        assert_eq!(p.decode[0].bound, "experts (NVMe)");
        // and the decode step is then seconds, not milliseconds
        assert!(p.decode[0].step_ms > 1000.0);
    }

    #[test]
    #[ignore]
    fn print_reference_plans() {
        for (g, h, m) in [
            ("h100-sxm", "genoa-2t", TrunkMode::Int8),
            ("h100-sxm", "genoa-2t", TrunkMode::Bf16),
            ("h100-sxm", "ws-512g", TrunkMode::Int8),
            ("h100-sxm", "cloud-1gpu", TrunkMode::Int8),
        ] {
            let mut r = rq(m);
            r.batches = vec![1, 4, 16, 32];
            r.prefill_tokens = vec![1024, 8192, 32768];
            let p = plan(&k3(), &hw::gpu_preset(g).unwrap(), &hw::host_preset(h).unwrap(), &r).unwrap();
            println!("{}\n", render(&p));
        }
    }

    #[test]
    fn unique_experts_saturate() {
        let mo = k3().moe().unwrap();
        assert!((mo.unique(1.0) - 16.0).abs() < 1e-9);
        assert!(mo.unique(4096.0) > 895.0);
    }
}
