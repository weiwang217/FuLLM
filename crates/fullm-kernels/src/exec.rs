// SPDX-License-Identifier: Apache-2.0
//! Rust side of the executor ABI (`csrc/include/fullm_exec.h`): the one boundary between
//! the control plane and the C++/CUDA data plane. One `step` call per engine step.

use std::ffi::{c_char, CStr, CString};
use std::ptr::NonNull;

#[repr(C)]
struct SkExec {
    _opaque: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ExecOpts {
    pub backend: i32,
    pub device: i32,
    pub n_slots: i32,
    pub n_pages: i32,
    pub page: i32,
    pub max_pages_per_seq: i32,
    pub max_tokens: i32,
    pub max_rows: i32,
    pub kv_dtype: i32,
    pub state_dtype: i32,
    pub cpu_threads: i32,
    /// 1 = never replay CUDA graphs (parity tests, debugging).
    pub no_graphs: i32,
    /// 1 = small-M GEMVs round activations to bf16 (as vLLM does); 0 = exact fp32.
    pub bf16_activations: i32,
    /// The backend by name (`backends()`), NUL-padded; `backend` is the alias when it is empty.
    pub backend_name: [c_char; 32],
}

impl ExecOpts {
    /// Names the backend (truncated to 31 bytes).
    pub fn with_backend(mut self, name: &str) -> Self {
        self.backend_name = [0; 32];
        for (d, s) in self.backend_name.iter_mut().zip(name.bytes().take(31)) {
            *d = s as c_char;
        }
        self
    }
}

pub const BACKEND_CPU: i32 = 0;
pub const BACKEND_CUDA: i32 = 1;
pub const BACKEND_TORCH: i32 = 2;
pub const DT_F32: i32 = 0;
pub const DT_BF16: i32 = 1;
pub const DT_FP8: i32 = 2;
pub const DT_FP8E4M3: i32 = 3;
/// The int4 KV cache (kv_int4.h): keys per channel over each page of 16, values rotated + per position.
pub const DT_KV4: i32 = 4;
/// Caps::families bits (backend.h FAM_*).
pub const FAM_GQA: u32 = 1;
pub const FAM_MLA: u32 = 2;
pub const FAM_KDA: u32 = 4;
pub const FAM_KIMI: u32 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Fmt {
    F32 = 0,
    Bf16 = 1,
    Int8 = 2,
    Fp8 = 3,
    /// E4M3 with one f32 scale per 128 x 128 block; given as bf16, quantised by the executor.
    Fp8Block = 4,
    /// int4 with one f32 scale per row per 128 columns; given as bf16, quantised by the executor.
    Int4Group = 5,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct SkK3Dims {
    hidden: i32,
    n_layers: i32,
    vocab: i32,
    rms_eps: f32,
    kda_heads: i32,
    kda_head_dim: i32,
    conv_k: i32,
    gate_lb: f32,
    n_heads: i32,
    q_lora: i32,
    kv_lora: i32,
    qk_nope: i32,
    qk_rope: i32,
    v_head: i32,
    mla_out_gate: i32,
    n_experts: i32,
    topk: i32,
    n_shared: i32,
    latent: i32,
    moe_inter: i32,
    routed_scale: f32,
    moe_renorm: i32,
    latent_norm: i32,
    first_dense: i32,
    dense_inter: i32,
    attn_res_block: i32,
    situ_b1: f32,
    situ_b2: f32,
    attn_res: i32,
    decay_softplus: i32,
    kda_gate_lowrank: i32,
    latent_moe: i32,
    act_silu: i32,
    expert_bf16: i32,
    router_biased: i32,
    kv_norm_eps: f32,
    layer_is_mla: *const u8,
}

/// A dense decoder (Llama / Qwen3 family); see `sk_dense_dims` in fullm_exec.h.
#[derive(Clone, Debug)]
pub struct DenseDims {
    pub n_layers: i32,
    pub hidden: i32,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub inter: i32,
    pub vocab: i32,
    pub rms_eps: f32,
    /// Per-head RMSNorm of q and k before RoPE (Qwen3).
    pub qk_norm: bool,
    /// `[head_dim / 2]` RoPE inverse frequencies, scaling applied.
    pub inv_freq: Vec<f32>,
    /// 0: dense MLPs; otherwise every layer is a routed MoE (Qwen3-MoE).
    pub n_experts: i32,
    pub topk: i32,
    pub moe_inter: i32,
    /// Renormalise the top-k softmax weights to sum 1.
    pub norm_topk: bool,
}

#[repr(C)]
struct SkDenseDims {
    n_layers: i32,
    hidden: i32,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    inter: i32,
    vocab: i32,
    rms_eps: f32,
    qk_norm: i32,
    inv_freq: *const f32,
    n_experts: i32,
    topk: i32,
    moe_inter: i32,
    norm_topk: i32,
}

/// Kimi K3 dimensions, as the executor needs them.
#[derive(Clone, Debug, Default)]
pub struct K3Dims {
    pub hidden: i32,
    pub n_layers: i32,
    pub vocab: i32,
    pub rms_eps: f32,
    pub kda_heads: i32,
    pub kda_head_dim: i32,
    pub conv_k: i32,
    pub gate_lb: f32,
    pub n_heads: i32,
    pub q_lora: i32,
    pub kv_lora: i32,
    pub qk_nope: i32,
    pub qk_rope: i32,
    pub v_head: i32,
    pub mla_out_gate: bool,
    pub n_experts: i32,
    pub topk: i32,
    pub n_shared: i32,
    pub latent: i32,
    pub moe_inter: i32,
    pub routed_scale: f32,
    pub moe_renorm: bool,
    pub latent_norm: bool,
    pub first_dense: i32,
    pub dense_inter: i32,
    pub attn_res_block: i32,
    pub situ_b1: f32,
    pub situ_b2: f32,
    pub attn_res: bool,
    pub decay_softplus: bool,
    pub kda_gate_lowrank: bool,
    pub latent_moe: bool,
    pub act_silu: bool,
    pub expert_bf16: bool,
    pub router_biased: bool,
    pub kv_norm_eps: f32,
    pub layer_is_mla: Vec<u8>,
}

#[repr(C)]
struct SkStepPlan {
    n_tokens: i32,
    tokens: *const i32,
    positions: *const i32,
    token_slot: *const i32,
    n_segs: i32,
    segs: *const i32,
    page_table: *const i32,
    n_rows: i32,
    rows: *const i32,
    n_reset: i32,
    reset_slots: *const i32,
}

/// One step's work. All slices are borrowed for the call only.
pub struct StepPlan<'a> {
    pub tokens: &'a [i32],
    pub positions: &'a [i32],
    pub token_slot: &'a [i32],
    /// `[n_segs][3]`: slot, first token row, count.
    pub segs: &'a [i32],
    /// `[n_slots][max_pages_per_seq]`
    pub page_table: &'a [i32],
    pub rows: &'a [i32],
    pub reset_slots: &'a [i32],
}

/// An EAGLE-3 draft head on a dense executor (`sk_eagle_dims`): the layers whose incoming
/// residual stream it reads, its MLP width and its vocabulary.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EagleDims {
    pub taps: [i32; 3],
    pub inter: i32,
    pub draft_vocab: i32,
}

#[repr(C)]
struct SkDraftPlan {
    n_seqs: i32,
    slot: *const i32,
    pos0: *const i32,
    n_in: *const i32,
    k: *const i32,
    tokens: *const i32,
    rows: *const i32,
    page_table: *const i32,
}

/// One sequence of a draft call: `tokens` (t(j+1)) consumed at positions `pos0..`, with the
/// hidden states the last step tapped at its token rows `rows`, then `k` drafts.
pub struct DraftSeq<'a> {
    pub slot: i32,
    pub pos0: i32,
    pub tokens: &'a [i32],
    pub rows: &'a [i32],
    pub k: usize,
}

/// How a row's token is drawn (`sk_sample_row`): the argmax at temperature <= 0; otherwise
/// softmax(logits / temperature) over the values at or above the top_k-th largest (0: all; ties
/// stay) and at least the largest + `min_p_cut` (min-p), then those at or above the largest
/// threshold holding top_p of the mass, the token whose cumulative mass in token order first
/// passes u * mass, u being PCG32(seed, step)'s first double. A function of the row alone.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SampleRow {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    /// 0: none; k: the row draws only among the tokens of the step's mask k - 1 (`step_sample`).
    pub mask: i32,
    pub seed: u64,
    pub step: u64,
    /// Min-p as a cut below the row's largest logit: temperature x ln(min_p) (< 0); 0: off.
    pub min_p_cut: f32,
    /// The row's logit adjustments: the step's `adj[adj_off .. adj_off + adj_n]`.
    pub adj_off: i32,
    pub adj_n: i32,
    /// 0: none; k: the drawn token's log-probability and the k - 1 most likely tokens'.
    pub logprobs: i32,
}

impl SampleRow {
    pub fn new(temperature: f32, top_p: f32, top_k: usize, seed: u64, step: u64) -> Self {
        SampleRow { temperature, top_p, top_k: top_k.min(i32::MAX as usize) as i32, seed, step, ..Default::default() }
    }
}

/// A token's logit adjustment (`sk_logit_adj`; penalties, logit bias, min_tokens), in vLLM's
/// order: x + bias, then divided by rep when positive (else multiplied), then less sub
/// (+inf: never drawn).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LogitAdj {
    pub token: i32,
    pub bias: f32,
    pub rep: f32,
    pub sub: f32,
}

impl LogitAdj {
    /// `x` adjusted, rounded as the backends round it.
    pub fn apply(&self, x: f32) -> f32 {
        let v = x + self.bias;
        let v = if v > 0.0 { v / self.rep } else { v * self.rep };
        v - self.sub
    }
}

/// Most top alternatives a row's log-probabilities carry (`SK_MAX_TOP_LOGPROBS`).
pub const MAX_TOP_LOGPROBS: usize = 20;

#[repr(C)]
struct SkSampleIo {
    masks: *const u32,
    mask_words: i64,
    adj: *const LogitAdj,
    n_adj: i64,
    logprobs: *mut f32,
    top_ids: *mut i32,
}

/// A sampling step's inputs and outputs besides its rows (`step_sample`): the allowed-token
/// bitmasks the rows' `mask` fields name (`mask_words` u32 each, token i at bit i % 32 of word
/// i / 32), the adjustments their `adj_off` / `adj_n` index, and when a row asks for them the
/// log-probabilities out: row r's drawn token's at `logprobs[r * (MAX_TOP_LOGPROBS + 1)]`, its
/// alternatives' after it, their ids at `top_ids[r * MAX_TOP_LOGPROBS ..]`.
#[derive(Default)]
pub struct SampleIo<'a> {
    pub masks: &'a [u32],
    pub mask_words: usize,
    pub adj: &'a [LogitAdj],
    pub logprobs: Option<(&'a mut [f32], &'a mut [i32])>,
}

/// The draw on the host, as the backends make it (`sample_row_host`): rows of `n` logits.
pub fn sample_rows_host(logits: &[f32], n: usize, sample: &[SampleRow]) -> Vec<i32> {
    assert!(n > 0 && logits.len() == n * sample.len(), "{} logits for {} rows of {n}", logits.len(), sample.len());
    let mut out = vec![0i32; sample.len()];
    unsafe { sk_sample_rows_host(out.as_mut_ptr(), logits.as_ptr(), sample.len() as i64, n as i64, sample.as_ptr()) };
    out
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct StepStats {
    pub ms_total: f64,
    pub ms_gpu_wait: f64,
    pub ms_experts: f64,
    pub ms_upload: f64,
    pub ms_download: f64,
    pub expert_bytes: i64,
    pub distinct_experts: i64,
    pub gpu_pairs: i64,
    pub cpu_pairs: i64,
}

extern "C" {
    fn sk_exec_create_k3(dims: *const SkK3Dims, opts: *const ExecOpts, err: *mut c_char, n: i32) -> *mut SkExec;
    fn sk_exec_create_dense(dims: *const SkDenseDims, opts: *const ExecOpts, err: *mut c_char, n: i32) -> *mut SkExec;
    fn sk_exec_destroy(e: *mut SkExec);
    fn sk_exec_add_matrix(
        e: *mut SkExec,
        name: *const c_char,
        fmt: i32,
        rows: i64,
        cols: i64,
        data: *const u8,
        scale: *const f32,
        err: *mut c_char,
        n: i32,
    ) -> i32;
    fn sk_exec_add_vector(
        e: *mut SkExec,
        name: *const c_char,
        data: *const f32,
        len: i64,
        err: *mut c_char,
        n: i32,
    ) -> i32;
    fn sk_exec_set_expert(
        e: *mut SkExec,
        layer: i32,
        expert: i32,
        w1p: *const u8,
        w1s: *const u8,
        w3p: *const u8,
        w3s: *const u8,
        w2p: *const u8,
        w2s: *const u8,
    ) -> i32;
    fn sk_exec_finalize(e: *mut SkExec, err: *mut c_char, n: i32) -> i32;
    fn sk_exec_set_eagle(e: *mut SkExec, dims: *const EagleDims, err: *mut c_char, n: i32) -> i32;
    fn sk_exec_draft(
        e: *mut SkExec,
        plan: *const SkDraftPlan,
        k_max: i32,
        drafts: *mut i32,
        err: *mut c_char,
        n: i32,
    ) -> i32;
    fn sk_exec_reserve_gpu_experts(e: *mut SkExec, fmt: i32, bytes: usize, err: *mut c_char, n: i32) -> i32;
    fn sk_exec_reserve_lora(e: *mut SkExec, slots: i32, max_rank: i32, targets: u32, err: *mut c_char, n: i32) -> i32;
    fn sk_exec_set_lora(
        e: *mut SkExec,
        slot: i32,
        layer: i32,
        target: i32,
        a: *const u16,
        a_len: usize,
        b: *const u16,
        b_len: usize,
        rank: i32,
        scale: f32,
        err: *mut c_char,
        n: i32,
    ) -> i32;
    fn sk_exec_clear_lora(e: *mut SkExec, slot: i32, err: *mut c_char, n: i32) -> i32;
    fn sk_exec_set_seq_lora(e: *mut SkExec, seq: i32, slot: i32, err: *mut c_char, n: i32) -> i32;
    fn sk_exec_expert_cache(e: *mut SkExec, path: *const c_char, state: *mut i32, err: *mut c_char, n: i32) -> i32;
    fn sk_exec_set_gpu_expert(
        e: *mut SkExec,
        layer: i32,
        expert: i32,
        w1: *const u8,
        w1s: *const f32,
        w3: *const u8,
        w3s: *const f32,
        w2: *const u8,
        w2s: *const f32,
        err: *mut c_char,
        n: i32,
    ) -> i32;
    fn sk_exec_pool_bytes(e: *const SkExec) -> usize;
    fn sk_exec_step(
        e: *mut SkExec,
        plan: *const SkStepPlan,
        logits: *mut f32,
        stats: *mut StepStats,
        err: *mut c_char,
        n: i32,
    ) -> i32;
    fn sk_exec_step_argmax(
        e: *mut SkExec,
        plan: *const SkStepPlan,
        tokens: *mut i32,
        stats: *mut StepStats,
        err: *mut c_char,
        n: i32,
    ) -> i32;
    fn sk_exec_step_sample(
        e: *mut SkExec,
        plan: *const SkStepPlan,
        sample: *const SampleRow,
        io: *const SkSampleIo,
        tokens: *mut i32,
        stats: *mut StepStats,
        err: *mut c_char,
        n: i32,
    ) -> i32;
    fn sk_sample_rows_host(tokens: *mut i32, logits: *const f32, rows: i64, n: i64, sample: *const SampleRow);
    fn sk_exec_slot_bytes(e: *const SkExec) -> usize;
    fn sk_exec_save_slot(e: *mut SkExec, slot: i32, dst: *mut u8, err: *mut c_char, n: i32) -> i32;
    fn sk_exec_load_slot(e: *mut SkExec, slot: i32, src: *const u8, err: *mut c_char, n: i32) -> i32;
    fn sk_exec_page_bytes(e: *const SkExec) -> usize;
    fn sk_exec_reserve_host_pages(e: *mut SkExec, n: i64, err: *mut c_char, len: i32) -> i32;
    fn sk_exec_copy_pages(
        e: *mut SkExec,
        to_host: i32,
        dev_pages: *const i32,
        host_pages: *const i32,
        n: i64,
        ticket: *mut u64,
        err: *mut c_char,
        len: i32,
    ) -> i32;
    fn sk_exec_copy_done(e: *mut SkExec, ticket: u64) -> i32;
    fn sk_exec_copy_order(e: *mut SkExec, ticket: u64, err: *mut c_char, len: i32) -> i32;
    fn sk_exec_copy_wait(e: *mut SkExec, ticket: u64, err: *mut c_char, len: i32) -> i32;
    fn sk_exec_describe(e: *mut SkExec) -> *const c_char;

    fn sk_backend_list(out: *mut c_char, len: i32) -> i32;
    fn sk_backend_caps(
        backend: *const c_char,
        device: i32,
        bf16_activations: i32,
        out: *mut SkCaps,
        err: *mut c_char,
        n: i32,
    ) -> i32;
    fn sk_cuda_available() -> i32;
    fn sk_cuda_device_info(
        dev: i32,
        name: *mut c_char,
        name_len: i32,
        total: *mut usize,
        free: *mut usize,
        sms: *mut i32,
        ccmaj: *mut i32,
        ccmin: *mut i32,
    ) -> i32;
    fn sk_cuda_bench_copy(dev: i32, bytes: usize, iters: i32, h2d: *mut f64, d2h: *mut f64) -> i32;
    fn sk_cuda_bench_gemm(dev: i32, fmt: i32, n: i64, k: i64, m: i64, iters: i32, gbps: *mut f64, ms: *mut f64) -> i32;
    fn sk_exec_selftest(dev: i32, report: *mut c_char, len: i32) -> i32;
    fn sk_conformance(subject: *const c_char, reference: *const c_char, dev: i32, report: *mut c_char, len: i32)
        -> i32;
    fn sk_exec_kbench(dev: i32, spec: *const c_char, report: *mut c_char, len: i32) -> i32;
}

#[repr(C)]
struct SkCaps {
    name: [c_char; 16],
    memory: i32,
    graphs: i32,
    async_copy: i32,
    dense_formats: u32,
    expert_formats: u32,
    kv_formats: u32,
    kv8_page_multiple: i32,
    families: u32,
    expert_tile_max_rows: i64,
    expert_gemm_min_rows: i64,
}

/// Where a backend's buffers live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemKind {
    Host,
    Discrete,
    Unified,
}

/// What a backend can do on a device (backend.h's Caps and Policy), read before an executor is
/// built: the planner checks requests against it instead of naming backends.
#[derive(Clone, Debug)]
pub struct Caps {
    pub name: String,
    pub memory: MemKind,
    pub graphs: bool,
    pub async_copy: bool,
    /// `1 << Fmt` per weight format the dense matrices may take.
    pub dense_formats: u32,
    /// `1 << Fmt` per format the resident experts may take.
    pub expert_formats: u32,
    /// `1 << DT_*` per KV cache storage.
    pub kv_formats: u32,
    /// fp8 KV: pages hold a multiple of this many positions.
    pub kv8_page_multiple: usize,
    /// The model families' ops: FAM_GQA (dense models), FAM_MLA, FAM_KDA, FAM_KIMI.
    pub families: u32,
    /// Policy: int4 experts' decode tiles take steps up to this many rows.
    pub expert_tile_max_rows: i64,
    /// Policy: int4 experts' prefill GEMM takes calls from this many rows.
    pub expert_gemm_min_rows: i64,
}

impl Caps {
    pub fn kv(&self, dt: i32) -> bool {
        self.kv_formats & (1 << dt) != 0
    }
    pub fn expert(&self, f: Fmt) -> bool {
        self.expert_formats & (1 << f as i32) != 0
    }
}

/// The executor backends compiled into this build, in registry order, and whether each can run on
/// this machine.
pub fn backends() -> Vec<(String, bool)> {
    let mut buf = vec![0 as c_char; 1024];
    let n = unsafe { sk_backend_list(buf.as_mut_ptr(), buf.len() as i32) };
    if n as usize >= buf.len() {
        buf = vec![0 as c_char; n as usize + 1];
        unsafe { sk_backend_list(buf.as_mut_ptr(), buf.len() as i32) };
    }
    err_string(&buf)
        .split(',')
        .filter_map(|e| e.rsplit_once(':').map(|(name, ok)| (name.to_string(), ok == "1")))
        .collect()
}

/// The capabilities of the named backend on `device`, with or without bf16 activations.
pub fn backend_caps(backend: &str, device: i32, bf16_activations: bool) -> Result<Caps, String> {
    let name = CString::new(backend).map_err(|_| format!("bad backend name {backend:?}"))?;
    let mut c = std::mem::MaybeUninit::<SkCaps>::zeroed();
    let mut err = err_buf();
    let rc = unsafe {
        sk_backend_caps(
            name.as_ptr(),
            device,
            bf16_activations as i32,
            c.as_mut_ptr(),
            err.as_mut_ptr(),
            ERR_LEN as i32,
        )
    };
    if rc != 0 {
        return Err(err_string(&err));
    }
    let c = unsafe { c.assume_init() };
    Ok(Caps {
        name: err_string(&c.name),
        memory: match c.memory {
            1 => MemKind::Discrete,
            2 => MemKind::Unified,
            _ => MemKind::Host,
        },
        graphs: c.graphs != 0,
        async_copy: c.async_copy != 0,
        dense_formats: c.dense_formats,
        expert_formats: c.expert_formats,
        kv_formats: c.kv_formats,
        kv8_page_multiple: c.kv8_page_multiple.max(1) as usize,
        families: c.families,
        expert_tile_max_rows: c.expert_tile_max_rows,
        expert_gemm_min_rows: c.expert_gemm_min_rows,
    })
}

const ERR_LEN: usize = 2048;

fn err_buf() -> Vec<c_char> {
    vec![0; ERR_LEN]
}

fn err_string(b: &[c_char]) -> String {
    unsafe { CStr::from_ptr(b.as_ptr()) }.to_string_lossy().into_owned()
}

pub struct Executor {
    raw: NonNull<SkExec>,
    rows_cap: usize,
    /// n_slots x max_pages_per_seq: the page table every step and draft must hold
    page_cap: usize,
}

// The executor is used from one thread at a time (the engine thread); it owns its streams.
unsafe impl Send for Executor {}

impl Drop for Executor {
    fn drop(&mut self) {
        unsafe { sk_exec_destroy(self.raw.as_ptr()) }
    }
}

impl Executor {
    fn wrap(raw: NonNull<SkExec>, opts: &ExecOpts) -> Self {
        let page_cap = opts.n_slots.max(0) as usize * opts.max_pages_per_seq.max(0) as usize;
        Executor { raw, rows_cap: opts.max_rows.max(0) as usize, page_cap }
    }

    pub fn new_dense(d: &DenseDims, opts: &ExecOpts) -> Result<Self, String> {
        if d.inv_freq.len() * 2 != d.head_dim as usize {
            return Err("inv_freq must have head_dim / 2 entries".into());
        }
        let raw = SkDenseDims {
            n_layers: d.n_layers,
            hidden: d.hidden,
            n_heads: d.n_heads,
            n_kv_heads: d.n_kv_heads,
            head_dim: d.head_dim,
            inter: d.inter,
            vocab: d.vocab,
            rms_eps: d.rms_eps,
            qk_norm: d.qk_norm as i32,
            inv_freq: d.inv_freq.as_ptr(),
            n_experts: d.n_experts,
            topk: d.topk,
            moe_inter: d.moe_inter,
            norm_topk: d.norm_topk as i32,
        };
        let mut err = err_buf();
        // SAFETY: raw (and inv_freq, copied by the callee) outlive the call.
        let p = unsafe { sk_exec_create_dense(&raw, opts, err.as_mut_ptr(), ERR_LEN as i32) };
        NonNull::new(p).map(|raw| Executor::wrap(raw, opts)).ok_or_else(|| err_string(&err))
    }

    pub fn new_k3(d: &K3Dims, opts: &ExecOpts) -> Result<Self, String> {
        if d.layer_is_mla.len() != d.n_layers as usize {
            return Err("layer_is_mla must have n_layers entries".into());
        }
        let raw = SkK3Dims {
            hidden: d.hidden,
            n_layers: d.n_layers,
            vocab: d.vocab,
            rms_eps: d.rms_eps,
            kda_heads: d.kda_heads,
            kda_head_dim: d.kda_head_dim,
            conv_k: d.conv_k,
            gate_lb: d.gate_lb,
            n_heads: d.n_heads,
            q_lora: d.q_lora,
            kv_lora: d.kv_lora,
            qk_nope: d.qk_nope,
            qk_rope: d.qk_rope,
            v_head: d.v_head,
            mla_out_gate: d.mla_out_gate as i32,
            n_experts: d.n_experts,
            topk: d.topk,
            n_shared: d.n_shared,
            latent: d.latent,
            moe_inter: d.moe_inter,
            routed_scale: d.routed_scale,
            moe_renorm: d.moe_renorm as i32,
            latent_norm: d.latent_norm as i32,
            first_dense: d.first_dense,
            dense_inter: d.dense_inter,
            attn_res_block: d.attn_res_block,
            situ_b1: d.situ_b1,
            situ_b2: d.situ_b2,
            attn_res: d.attn_res as i32,
            decay_softplus: d.decay_softplus as i32,
            kda_gate_lowrank: d.kda_gate_lowrank as i32,
            latent_moe: d.latent_moe as i32,
            act_silu: d.act_silu as i32,
            expert_bf16: d.expert_bf16 as i32,
            router_biased: d.router_biased as i32,
            kv_norm_eps: d.kv_norm_eps,
            layer_is_mla: d.layer_is_mla.as_ptr(),
        };
        let mut err = err_buf();
        let p = unsafe { sk_exec_create_k3(&raw, opts, err.as_mut_ptr(), ERR_LEN as i32) };
        match NonNull::new(p) {
            Some(raw) => Ok(Executor::wrap(raw, opts)),
            None => Err(err_string(&err)),
        }
    }

    fn check(rc: i32, err: &[c_char]) -> Result<(), String> {
        if rc == 0 {
            Ok(())
        } else {
            Err(err_string(err))
        }
    }

    /// Copy a matrix into the executor. `data` holds `rows*cols` elements of `fmt`.
    pub fn add_matrix(
        &mut self,
        name: &str,
        fmt: Fmt,
        rows: usize,
        cols: usize,
        data: &[u8],
        scale: Option<&[f32]>,
    ) -> Result<(), String> {
        let elem = match fmt {
            Fmt::F32 => 4,
            Fmt::Bf16 => 2,
            Fmt::Int8 | Fmt::Fp8 => 1,
            Fmt::Fp8Block | Fmt::Int4Group => {
                return Err(format!("{name}: block-scaled fp8 and int4 are for resident experts only"))
            }
        };
        if rows.checked_mul(cols).and_then(|n| n.checked_mul(elem)).is_none_or(|n| data.len() < n) {
            return Err(format!("{name}: {} bytes for [{rows}][{cols}] {fmt:?}", data.len()));
        }
        if matches!(fmt, Fmt::Int8 | Fmt::Fp8) && scale.is_none_or(|s| s.len() < rows) {
            return Err(format!("{name}: 8-bit matrix needs {rows} row scales"));
        }
        if name == "embed" {
            return Err("use add_embedding for the host-resident embedding table".into());
        }
        let c = CString::new(name).map_err(|e| e.to_string())?;
        let mut err = err_buf();
        let rc = unsafe {
            sk_exec_add_matrix(
                self.raw.as_ptr(),
                c.as_ptr(),
                fmt as i32,
                rows as i64,
                cols as i64,
                data.as_ptr(),
                scale.map_or(std::ptr::null(), |s| s.as_ptr()),
                err.as_mut_ptr(),
                ERR_LEN as i32,
            )
        };
        Self::check(rc, &err)
    }

    /// Register the embedding table, which stays in host memory and is not copied.
    ///
    /// # Safety
    /// `data` must stay valid and unmodified for the executor's lifetime.
    pub unsafe fn add_embedding(&mut self, fmt: Fmt, rows: usize, cols: usize, data: *const u8) -> Result<(), String> {
        let c = CString::new("embed").unwrap();
        let mut err = err_buf();
        let rc = sk_exec_add_matrix(
            self.raw.as_ptr(),
            c.as_ptr(),
            fmt as i32,
            rows as i64,
            cols as i64,
            data,
            std::ptr::null(),
            err.as_mut_ptr(),
            ERR_LEN as i32,
        );
        Self::check(rc, &err)
    }

    pub fn add_vector(&mut self, name: &str, data: &[f32]) -> Result<(), String> {
        let c = CString::new(name).map_err(|e| e.to_string())?;
        let mut err = err_buf();
        let rc = unsafe {
            sk_exec_add_vector(
                self.raw.as_ptr(),
                c.as_ptr(),
                data.as_ptr(),
                data.len() as i64,
                err.as_mut_ptr(),
                ERR_LEN as i32,
            )
        };
        Self::check(rc, &err)
    }

    /// Bind one routed expert: w1, w3, w2 as (packed, scales) host pointers.
    ///
    /// # Safety
    /// Every pointer must stay valid for the executor's lifetime (mapped checkpoint or arena).
    pub unsafe fn set_expert(&mut self, layer: usize, expert: usize, w: [*const u8; 6]) -> Result<(), String> {
        let rc = sk_exec_set_expert(self.raw.as_ptr(), layer as i32, expert as i32, w[0], w[1], w[2], w[3], w[4], w[5]);
        if rc == 0 {
            Ok(())
        } else {
            Err(format!("cannot bind expert {expert} of layer {layer}"))
        }
    }

    /// Device memory finalize() will take for pools, workspace and staging.
    pub fn pool_bytes(&self) -> usize {
        unsafe { sk_exec_pool_bytes(self.raw.as_ptr()) }
    }

    /// Reserve backend memory for `bytes` of resident experts stored as `fmt`.
    /// LoRA adapters served unmerged (dense executor): `slots` adapters of rank up to
    /// `max_rank` on the targets of `targets` (bits: 1 q, 2 k, 4 v, 8 o, 16 gate, 32 up, 64
    /// down). Before `finalize`.
    pub fn reserve_lora(&mut self, slots: usize, max_rank: usize, targets: u32) -> Result<(), String> {
        let mut err = err_buf();
        let rc = unsafe {
            sk_exec_reserve_lora(
                self.raw.as_ptr(),
                slots as i32,
                max_rank as i32,
                targets,
                err.as_mut_ptr(),
                ERR_LEN as i32,
            )
        };
        Self::check(rc, &err)
    }

    /// Slot `slot`'s update of layer `layer`'s target (0 q, 1 k, 2 v, 3 o, 4 gate, 5 up, 6
    /// down): `a` [rank][in], `b` [out][rank] bf16.
    #[allow(clippy::too_many_arguments)]
    pub fn set_lora(
        &mut self,
        slot: usize,
        layer: usize,
        target: usize,
        a: &[u16],
        b: &[u16],
        rank: usize,
        scale: f32,
    ) -> Result<(), String> {
        let mut err = err_buf();
        let rc = unsafe {
            sk_exec_set_lora(
                self.raw.as_ptr(),
                slot as i32,
                layer as i32,
                target as i32,
                a.as_ptr(),
                a.len(),
                b.as_ptr(),
                b.len(),
                rank as i32,
                scale,
                err.as_mut_ptr(),
                ERR_LEN as i32,
            )
        };
        Self::check(rc, &err)
    }

    pub fn clear_lora(&mut self, slot: usize) -> Result<(), String> {
        let mut err = err_buf();
        let rc = unsafe { sk_exec_clear_lora(self.raw.as_ptr(), slot as i32, err.as_mut_ptr(), ERR_LEN as i32) };
        Self::check(rc, &err)
    }

    /// The adapter of sequence slot `seq` (None: the base model) from the next step on.
    pub fn set_seq_lora(&mut self, seq: usize, slot: Option<usize>) -> Result<(), String> {
        let mut err = err_buf();
        let s = slot.map_or(-1, |s| s as i32);
        let rc = unsafe { sk_exec_set_seq_lora(self.raw.as_ptr(), seq as i32, s, err.as_mut_ptr(), ERR_LEN as i32) };
        Self::check(rc, &err)
    }

    pub fn reserve_gpu_experts(&mut self, fmt: Fmt, bytes: usize) -> Result<(), String> {
        let mut err = err_buf();
        let rc = unsafe {
            sk_exec_reserve_gpu_experts(self.raw.as_ptr(), fmt as i32, bytes, err.as_mut_ptr(), ERR_LEN as i32)
        };
        Self::check(rc, &err)
    }

    /// The quantised-expert cache at `path` (experts the executor quantises: fp8b, int4), after
    /// `reserve_gpu_experts`: `Some(true)` when the file holds this model's experts (then call
    /// `set_gpu_expert` for every expert in (layer, expert) order with empty slices: the data comes
    /// from the file), `Some(false)` when this load records it (finalize puts it in place), `None`
    /// when there is no cache for this executor or format.
    pub fn expert_cache(&mut self, path: &std::path::Path) -> Result<Option<bool>, String> {
        let c = std::ffi::CString::new(path.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
        let mut state = -1i32;
        let mut err = err_buf();
        let rc = unsafe {
            sk_exec_expert_cache(self.raw.as_ptr(), c.as_ptr(), &mut state, err.as_mut_ptr(), ERR_LEN as i32)
        };
        Self::check(rc, &err)?;
        Ok(match state {
            1 => Some(true),
            0 => Some(false),
            _ => None,
        })
    }

    /// Copy one expert (w1, w3 `[inter][width]`, w2 `[width][inter]` in the reserved format,
    /// with per-row scales for 8-bit; int4 as bf16, or as int8 values in [-8, 7] with their
    /// group scales `[cols/128][rows]`) into backend memory.
    ///
    /// # Safety
    /// The executor reads each slice and scale array to the size its reserved format and the
    /// model's dimensions give, without their lengths: they must hold at least that much (empty
    /// slices only when `expert_cache` returned `Some(true)`).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn set_gpu_expert(
        &mut self,
        layer: usize,
        expert: usize,
        w1: &[u8],
        s1: Option<&[f32]>,
        w3: &[u8],
        s3: Option<&[f32]>,
        w2: &[u8],
        s2: Option<&[f32]>,
    ) -> Result<(), String> {
        let sp = |s: Option<&[f32]>| s.map_or(std::ptr::null(), |v| v.as_ptr());
        let mut err = err_buf();
        let rc = unsafe {
            sk_exec_set_gpu_expert(
                self.raw.as_ptr(),
                layer as i32,
                expert as i32,
                w1.as_ptr(),
                sp(s1),
                w3.as_ptr(),
                sp(s3),
                w2.as_ptr(),
                sp(s2),
                err.as_mut_ptr(),
                ERR_LEN as i32,
            )
        };
        Self::check(rc, &err)
    }

    pub fn finalize(&mut self) -> Result<(), String> {
        let mut err = err_buf();
        let rc = unsafe { sk_exec_finalize(self.raw.as_ptr(), err.as_mut_ptr(), ERR_LEN as i32) };
        Self::check(rc, &err)
    }

    /// Attach an EAGLE-3 head (before its tensors are added and `finalize`).
    pub fn set_eagle(&mut self, d: &EagleDims) -> Result<(), String> {
        let mut err = err_buf();
        let rc = unsafe { sk_exec_set_eagle(self.raw.as_ptr(), d, err.as_mut_ptr(), ERR_LEN as i32) };
        Self::check(rc, &err)
    }

    /// The EAGLE-3 draft call after a step: each sequence's drafts (target ids), `seqs[s].k` of
    /// them. `page_table` must hold every position consumed or drafted.
    pub fn draft(&mut self, seqs: &[DraftSeq<'_>], page_table: &[i32]) -> Result<Vec<Vec<i32>>, String> {
        if seqs.is_empty() {
            return Ok(Vec::new());
        }
        if seqs.iter().any(|s| s.tokens.is_empty() || s.tokens.len() != s.rows.len()) {
            return Err("draft: every sequence consumes at least one token, with a tapped row each".into());
        }
        if page_table.len() < self.page_cap {
            return Err(format!(
                "draft: page table holds {} entries, the executor reads {}",
                page_table.len(),
                self.page_cap
            ));
        }
        let k_max = seqs.iter().map(|s| s.k).max().unwrap_or(0);
        let (slot, pos0): (Vec<i32>, Vec<i32>) = seqs.iter().map(|s| (s.slot, s.pos0)).unzip();
        let n_in: Vec<i32> = seqs.iter().map(|s| s.tokens.len() as i32).collect();
        let k: Vec<i32> = seqs.iter().map(|s| s.k as i32).collect();
        let tokens: Vec<i32> = seqs.iter().flat_map(|s| s.tokens.iter().copied()).collect();
        let rows: Vec<i32> = seqs.iter().flat_map(|s| s.rows.iter().copied()).collect();
        let plan = SkDraftPlan {
            n_seqs: seqs.len() as i32,
            slot: slot.as_ptr(),
            pos0: pos0.as_ptr(),
            n_in: n_in.as_ptr(),
            k: k.as_ptr(),
            tokens: tokens.as_ptr(),
            rows: rows.as_ptr(),
            page_table: page_table.as_ptr(),
        };
        let mut out = vec![0i32; seqs.len() * k_max];
        let mut err = err_buf();
        let rc = unsafe {
            sk_exec_draft(self.raw.as_ptr(), &plan, k_max as i32, out.as_mut_ptr(), err.as_mut_ptr(), ERR_LEN as i32)
        };
        Self::check(rc, &err)?;
        Ok(seqs.iter().enumerate().map(|(i, s)| out[i * k_max..i * k_max + s.k].to_vec()).collect())
    }

    /// Run one step. `logits` receives `[rows.len()][vocab]`.
    pub fn step(&mut self, p: &StepPlan<'_>, logits: &mut [f32]) -> Result<StepStats, String> {
        if p.rows.len() > self.rows_cap || logits.is_empty() || !logits.len().is_multiple_of(p.rows.len().max(1)) {
            return Err("logits buffer does not match the requested rows".into());
        }
        let raw = self.raw_plan(p)?;
        let mut stats = StepStats::default();
        let mut err = err_buf();
        let rc = unsafe {
            sk_exec_step(self.raw.as_ptr(), &raw, logits.as_mut_ptr(), &mut stats, err.as_mut_ptr(), ERR_LEN as i32)
        };
        Self::check(rc, &err).map(|_| stats)
    }

    /// The step for greedy decoding: `tokens[r]` = argmax of logits row r, computed on the
    /// device (only the ids cross to the host).
    pub fn step_argmax(&mut self, p: &StepPlan<'_>, tokens: &mut [i32]) -> Result<StepStats, String> {
        if p.rows.len() > self.rows_cap || tokens.len() != p.rows.len() {
            return Err("token buffer does not match the requested rows".into());
        }
        let raw = self.raw_plan(p)?;
        let mut stats = StepStats::default();
        let mut err = err_buf();
        let rc = unsafe {
            sk_exec_step_argmax(
                self.raw.as_ptr(),
                &raw,
                tokens.as_mut_ptr(),
                &mut stats,
                err.as_mut_ptr(),
                ERR_LEN as i32,
            )
        };
        Self::check(rc, &err).map(|_| stats)
    }

    /// The step for sampling: `tokens[r]` drawn from logits row r as `sample[r]` says, on the
    /// device (only the ids cross to the host). `masks`: the allowed-token bitmasks the rows'
    /// `mask` fields name, `words` u32 each (token i: bit i % 32 of word i / 32; `words` covers
    /// the vocabulary); empty when no row has one.
    pub fn step_sample(
        &mut self,
        p: &StepPlan<'_>,
        sample: &[SampleRow],
        io: &mut SampleIo<'_>,
        tokens: &mut [i32],
    ) -> Result<StepStats, String> {
        let rows = p.rows.len();
        if rows > self.rows_cap || tokens.len() != rows || sample.len() != rows {
            return Err("token or sampling buffer does not match the requested rows".into());
        }
        let n_masks = sample.iter().map(|s| s.mask.max(0) as usize).max().unwrap_or(0);
        if io.masks.len() < n_masks * io.mask_words {
            return Err(format!("{n_masks} sampling mask(s) of {} words, {} given", io.mask_words, io.masks.len()));
        }
        if sample.iter().any(|s| s.adj_n > 0 && (s.adj_off.max(0) + s.adj_n) as usize > io.adj.len()) {
            return Err("a row's logit adjustments are outside the step's".into());
        }
        let wants_lp = sample.iter().any(|s| s.logprobs > 0);
        let (lp, ids) = match io.logprobs.as_mut() {
            Some((l, i)) if l.len() >= rows * (MAX_TOP_LOGPROBS + 1) && i.len() >= rows * MAX_TOP_LOGPROBS => {
                (l.as_mut_ptr(), i.as_mut_ptr())
            }
            Some(_) => return Err("log-probability buffers smaller than the rows".into()),
            None if wants_lp => return Err("log-probabilities asked with no buffers".into()),
            None => (std::ptr::null_mut(), std::ptr::null_mut()),
        };
        let skio = SkSampleIo {
            masks: if n_masks > 0 { io.masks.as_ptr() } else { std::ptr::null() },
            mask_words: io.mask_words as i64,
            adj: if io.adj.is_empty() { std::ptr::null() } else { io.adj.as_ptr() },
            n_adj: io.adj.len() as i64,
            logprobs: lp,
            top_ids: ids,
        };
        let raw = self.raw_plan(p)?;
        let mut stats = StepStats::default();
        let mut err = err_buf();
        let rc = unsafe {
            sk_exec_step_sample(
                self.raw.as_ptr(),
                &raw,
                sample.as_ptr(),
                &skio,
                tokens.as_mut_ptr(),
                &mut stats,
                err.as_mut_ptr(),
                ERR_LEN as i32,
            )
        };
        Self::check(rc, &err).map(|_| stats)
    }

    fn raw_plan(&self, p: &StepPlan<'_>) -> Result<SkStepPlan, String> {
        let n = p.tokens.len();
        if p.positions.len() != n || p.token_slot.len() != n || !p.segs.len().is_multiple_of(3) {
            return Err("step plan arrays disagree in length".into());
        }
        if p.page_table.len() < self.page_cap {
            return Err(format!(
                "page table holds {} entries, the executor reads {}",
                p.page_table.len(),
                self.page_cap
            ));
        }
        Ok(SkStepPlan {
            n_tokens: n as i32,
            tokens: p.tokens.as_ptr(),
            positions: p.positions.as_ptr(),
            token_slot: p.token_slot.as_ptr(),
            n_segs: (p.segs.len() / 3) as i32,
            segs: p.segs.as_ptr(),
            page_table: p.page_table.as_ptr(),
            n_rows: p.rows.len() as i32,
            rows: p.rows.as_ptr(),
            n_reset: p.reset_slots.len() as i32,
            reset_slots: p.reset_slots.as_ptr(),
        })
    }

    pub fn slot_bytes(&self) -> usize {
        unsafe { sk_exec_slot_bytes(self.raw.as_ptr()) }
    }

    pub fn save_slot(&mut self, slot: usize, dst: &mut [u8]) -> Result<(), String> {
        if dst.len() != self.slot_bytes() {
            return Err("checkpoint buffer has the wrong size".into());
        }
        let mut err = err_buf();
        let rc = unsafe {
            sk_exec_save_slot(self.raw.as_ptr(), slot as i32, dst.as_mut_ptr(), err.as_mut_ptr(), ERR_LEN as i32)
        };
        Self::check(rc, &err)
    }

    pub fn load_slot(&mut self, slot: usize, src: &[u8]) -> Result<(), String> {
        if src.len() != self.slot_bytes() {
            return Err("checkpoint buffer has the wrong size".into());
        }
        let mut err = err_buf();
        let rc = unsafe {
            sk_exec_load_slot(self.raw.as_ptr(), slot as i32, src.as_ptr(), err.as_mut_ptr(), ERR_LEN as i32)
        };
        Self::check(rc, &err)
    }

    /// Bytes of one host page (a cache page of every paged pool), after `finalize`; 0 when the
    /// executor has no host tier.
    pub fn page_bytes(&self) -> usize {
        unsafe { sk_exec_page_bytes(self.raw.as_ptr()) }
    }

    /// Pin `n` host pages for the prefix cache's host tier.
    pub fn reserve_host_pages(&mut self, n: usize) -> Result<(), String> {
        let mut err = err_buf();
        let rc = unsafe { sk_exec_reserve_host_pages(self.raw.as_ptr(), n as i64, err.as_mut_ptr(), ERR_LEN as i32) };
        Self::check(rc, &err)
    }

    /// Queue copies of cache page `dev[i]` to host page `host[i]` (or back, `to_host` false) after
    /// the steps already run. Returns their ticket; tickets complete in order.
    pub fn copy_pages(&mut self, to_host: bool, dev: &[i32], host: &[i32]) -> Result<u64, String> {
        if dev.len() != host.len() {
            return Err("page lists differ in length".into());
        }
        let mut ticket = 0u64;
        let mut err = err_buf();
        let rc = unsafe {
            sk_exec_copy_pages(
                self.raw.as_ptr(),
                to_host as i32,
                dev.as_ptr(),
                host.as_ptr(),
                dev.len() as i64,
                &mut ticket,
                err.as_mut_ptr(),
                ERR_LEN as i32,
            )
        };
        Self::check(rc, &err).map(|_| ticket)
    }

    /// Whether copy `ticket` has completed (does not block). A failed query counts as done: the
    /// next step reports the device's error.
    pub fn copy_done(&mut self, ticket: u64) -> bool {
        unsafe { sk_exec_copy_done(self.raw.as_ptr(), ticket) != 0 }
    }

    /// The next step waits on the device for copy `ticket` (the host does not).
    pub fn copy_order(&mut self, ticket: u64) -> Result<(), String> {
        let mut err = err_buf();
        let rc = unsafe { sk_exec_copy_order(self.raw.as_ptr(), ticket, err.as_mut_ptr(), ERR_LEN as i32) };
        Self::check(rc, &err)
    }

    /// Block until copy `ticket` has completed.
    pub fn copy_wait(&mut self, ticket: u64) -> Result<(), String> {
        let mut err = err_buf();
        let rc = unsafe { sk_exec_copy_wait(self.raw.as_ptr(), ticket, err.as_mut_ptr(), ERR_LEN as i32) };
        Self::check(rc, &err)
    }

    pub fn describe(&mut self) -> String {
        unsafe { CStr::from_ptr(sk_exec_describe(self.raw.as_ptr())) }.to_string_lossy().into_owned()
    }
}

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub total_bytes: usize,
    pub free_bytes: usize,
    pub sm_count: i32,
    pub compute_capability: (i32, i32),
}

pub fn cuda_available() -> bool {
    unsafe { sk_cuda_available() != 0 }
}

pub fn device_info(device: i32) -> Option<DeviceInfo> {
    let mut name = vec![0 as c_char; 256];
    let (mut total, mut free, mut sms, mut maj, mut min) = (0usize, 0usize, 0i32, 0i32, 0i32);
    let rc = unsafe {
        sk_cuda_device_info(device, name.as_mut_ptr(), 256, &mut total, &mut free, &mut sms, &mut maj, &mut min)
    };
    (rc == 0).then(|| DeviceInfo {
        name: err_string(&name),
        total_bytes: total,
        free_bytes: free,
        sm_count: sms,
        compute_capability: (maj, min),
    })
}

/// Pinned host<->device copy bandwidth in GB/s: (h2d, d2h).
pub fn bench_copy(device: i32, bytes: usize, iters: i32) -> Option<(f64, f64)> {
    let (mut h, mut d) = (0.0, 0.0);
    (unsafe { sk_cuda_bench_copy(device, bytes, iters, &mut h, &mut d) } == 0).then_some((h, d))
}

/// Weight-read bandwidth of the gemm path: (GB/s, ms per call).
pub fn bench_gemm(device: i32, fmt: Fmt, n: i64, k: i64, m: i64, iters: i32) -> Option<(f64, f64)> {
    let (mut g, mut ms) = (0.0, 0.0);
    (unsafe { sk_cuda_bench_gemm(device, fmt as i32, n, k, m, iters, &mut g, &mut ms) } == 0).then_some((g, ms))
}

/// Kernel micro-benchmarks on synthetic data (csrc/exec/cuda/kbench.cu; empty spec: the
/// default cases): (0, 1 if a case failed, or -1 if no CUDA; report).
pub fn kbench(device: i32, spec: &str) -> (i32, String) {
    let c = std::ffi::CString::new(spec).unwrap_or_default();
    let mut buf = vec![0 as c_char; 64 * 1024];
    let rc = unsafe { sk_exec_kbench(device, c.as_ptr(), buf.as_mut_ptr(), buf.len() as i32) };
    (rc, err_string(&buf))
}

/// CUDA ops vs CPU ops on random inputs: (failures or -1 if no CUDA, report).
pub fn selftest(device: i32) -> (i32, String) {
    let mut buf = vec![0 as c_char; 64 * 1024];
    let rc = unsafe { sk_exec_selftest(device, buf.as_mut_ptr(), buf.len() as i32) };
    (rc, err_string(&buf))
}

/// The conformance suite: every op of the `subject` backend against the `reference` backend (by
/// name), skipping what the subject's Caps do not declare. (failures or -1, the report)
pub fn conformance(subject: &str, reference: &str, device: i32) -> (i32, String) {
    let (Ok(s), Ok(r)) = (CString::new(subject), CString::new(reference)) else {
        return (-1, "bad backend name\n".into());
    };
    let mut buf = vec![0 as c_char; 256 * 1024];
    let rc = unsafe { sk_conformance(s.as_ptr(), r.as_ptr(), device, buf.as_mut_ptr(), buf.len() as i32) };
    (rc, err_string(&buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_caps() {
        let c = backend_caps("cpu", 0, true).expect("the CPU backend is always there");
        assert_eq!(c.name, "cpu");
        assert_eq!(c.memory, MemKind::Host);
        assert!(!c.graphs);
        assert!(c.kv(DT_F32) && c.kv(DT_BF16) && !c.kv(DT_FP8) && !c.kv(DT_FP8E4M3));
        for f in [Fmt::Bf16, Fmt::Int8, Fmt::Fp8, Fmt::Fp8Block, Fmt::Int4Group] {
            assert!(c.expert(f), "{f:?}");
        }
        assert_eq!((c.expert_tile_max_rows, c.expert_gemm_min_rows), (64, 64));
        assert_eq!(c.families, FAM_GQA | FAM_MLA | FAM_KDA | FAM_KIMI);
        let core = backend_caps("cpu-core", 0, true).expect("cpu-core is always there");
        assert_eq!((core.name.as_str(), core.families), ("cpu-core", FAM_GQA));
        assert!(backend_caps("no-such-device", 0, true).is_err());
        let list = backends();
        assert_eq!(list.first(), Some(&("cpu".to_string(), true)), "{list:?}");
        assert!(list.iter().any(|(n, _)| n == "cuda"), "{list:?}");
    }

    /// The conformance suite on cpu-core (the core ops only, every fast path from Backend's base
    /// forms) against the CPU backend: what a new device's first run looks like.
    #[test]
    fn cpu_core_conformance() {
        let (fails, report) = conformance("cpu-core", "cpu", 0);
        assert_eq!(fails, 0, "{report}");
        assert!(report.contains("skipped (families the subject does not declare): kda mla kimi"), "{report}");
    }
}
