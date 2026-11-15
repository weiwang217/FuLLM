// SPDX-License-Identifier: Apache-2.0
//! What every executor-backed model shares: sequence slots, cache pages granted per sequence
//! (the paged KV or latent cache) from a pool that keeps full pages as a prefix cache
//! (`prefix.rs`), the step plan sent to the executor, and the `Model` step methods built on it.
//!
//! HOST TIER COPIES. With `--host-cache-tokens`, full pages are written through to host pages
//! after the step that filled them, and pages the device dropped are loaded back when a sequence
//! starts with them. The executor runs both on a copy stream, in order, each call a ticket. Two
//! hazards are left for the step: reading a page whose load is in flight (each sequence waits for
//! the loads of the pages it holds, `SeqState::ready`), and overwriting a page a copy still reads
//! or writes (a page taken for new positions waits for its last copy). The engine schedules a
//! sequence only once its loads are done (`seq_ready`), so the step rarely waits at all.

use anyhow::{anyhow, bail, ensure, Context, Result};
use fullm_kernels::exec::{self, DraftSeq, ExecOpts, Executor, StepPlan, StepStats};

use crate::models::prefix::{Insert, PagePool};
use crate::models::{CacheDtype, Capacity, DraftReq, LoadOptions, SampleIo, SampleRow, SeqChunk, SeqId};

#[derive(Default)]
struct SeqState {
    used: bool,
    /// Zero the recurrent state before this sequence's next step.
    fresh: bool,
    len: usize,
    pages: Vec<u32>,
    /// With the prefix cache: the tokens at positions 0..len, and how many leading pages are in
    /// the cache (held by this sequence).
    tokens: Vec<u32>,
    shared: usize,
    /// The copy ticket its held pages wait for (their loads from the host tier).
    ready: u64,
    /// The drafts of its last step, until `accept` (with the prefix cache: their tokens).
    draft: usize,
    draft_tokens: Vec<u32>,
    /// Tests: `seq_ready` answers no this many more times after a load.
    lag: usize,
    /// Its adapter slot (`Model::set_seq_lora`): what `tokens` keys the prefix cache with.
    lora: Option<usize>,
}

/// Cache geometry derived from the load options.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub page: usize,
    pub n_pages: usize,
    /// Columns of the page table: the longest sequence, in pages.
    pub max_pages: usize,
    pub max_rows: usize,
    /// The KV cache layout (DT_*): --kv-dtype resolved against the backend's caps.
    pub kv_dt: i32,
}

impl Geometry {
    pub fn new(opts: &LoadOptions) -> Result<Self> {
        ensure!(
            opts.max_seqs > 0 && opts.page > 0 && opts.step_tokens > 0,
            "max_seqs, page and step_tokens must be positive"
        );
        ensure!(
            !matches!(opts.state_dtype, CacheDtype::Fp8 | CacheDtype::Fp8Token),
            "fp8 is a KV cache format, not a recurrent-state one"
        );
        let mut kv_dt = match opts.kv_dtype {
            CacheDtype::F32 => exec::DT_F32,
            CacheDtype::Fp8 => exec::DT_FP8E4M3,
            CacheDtype::Fp8Token => exec::DT_FP8,
            CacheDtype::Auto | CacheDtype::Bf16 => exec::DT_BF16,
        };
        if matches!(opts.kv_dtype, CacheDtype::Fp8 | CacheDtype::Fp8Token) {
            // the backend says which fp8 layouts it has (and on which pages), not its name: fp8 is
            // its best (FlashAttention-3's pages, else a scale per token), fp8-token that one
            let caps = exec::backend_caps(opts.backend.exec_name(), opts.device, opts.bf16_activations)
                .map_err(|e| anyhow!("{e}"))?;
            if opts.kv_dtype == CacheDtype::Fp8 && !caps.kv(exec::DT_FP8E4M3) && caps.kv(exec::DT_FP8) {
                kv_dt = exec::DT_FP8;
            }
            if !caps.kv(kv_dt) {
                let flag = if opts.kv_dtype == CacheDtype::Fp8 { "fp8" } else { "fp8-token" };
                // only the activation mode in the way: say so; else what this backend's caches are
                let with_bf16 = !opts.bf16_activations
                    && exec::backend_caps(opts.backend.exec_name(), opts.device, true).is_ok_and(|c| c.kv(kv_dt));
                if with_bf16 {
                    bail!("--kv-dtype {flag}: the {} backend's fp8 caches need --activations bf16", caps.name);
                }
                let have: Vec<&str> = [
                    (exec::DT_F32, "f32"),
                    (exec::DT_BF16, "bf16"),
                    (exec::DT_FP8, "fp8-token"),
                    (exec::DT_FP8E4M3, "fp8"),
                ]
                .iter()
                .filter(|(d, _)| caps.kv(*d))
                .map(|(_, n)| *n)
                .collect();
                bail!(
                    "--kv-dtype {flag}: the {} backend has no fp8 cache (its caches: {}; `fullm backends` lists every backend's)",
                    caps.name,
                    have.join(", ")
                );
            }
            ensure!(
                opts.page % caps.kv8_page_multiple == 0,
                "--kv-dtype fp8 needs pages of a multiple of {} positions",
                caps.kv8_page_multiple
            );
        }
        let page = opts.page;
        let n_pages = opts.cache_tokens.div_ceil(page).max(1);
        let max_ctx = opts.max_context.min(n_pages * page).max(page);
        Ok(Geometry {
            page,
            n_pages,
            max_pages: max_ctx.div_ceil(page),
            max_rows: opts.max_rows.max(opts.max_seqs),
            kv_dt,
        })
    }

    /// The executor options for this geometry.
    pub fn exec_opts(&self, opts: &LoadOptions) -> ExecOpts {
        ExecOpts {
            backend: exec::BACKEND_CPU, // the alias, unused: backend_name below
            device: opts.device,
            n_slots: opts.max_seqs as i32,
            n_pages: self.n_pages as i32,
            page: self.page as i32,
            max_pages_per_seq: self.max_pages as i32,
            max_tokens: opts.step_tokens as i32,
            max_rows: self.max_rows as i32,
            kv_dtype: self.kv_dt,
            state_dtype: if opts.state_dtype.is_bf16() { exec::DT_BF16 } else { exec::DT_F32 },
            cpu_threads: opts.threads as i32,
            no_graphs: opts.no_cuda_graphs as i32,
            bf16_activations: opts.bf16_activations as i32,
            backend_name: [0; 32],
        }
        .with_backend(opts.backend.exec_name())
    }
}

pub struct Paged {
    pub ex: Executor,
    vocab: usize,
    page: usize,
    max_pages: usize,
    step_tokens: usize,
    max_rows: usize,
    pool: PagePool,
    seqs: Vec<SeqState>,
    /// `[slot][max_pages]`, uploaded whole with every step.
    page_table: Vec<i32>,
    last: Option<StepStats>,
    /// By device page: the ticket of the last host-tier copy into or out of it, and of the last
    /// load into it.
    copied: Vec<u64>,
    loaded: Vec<u64>,
    copy_lag: usize,
    /// An EAGLE-3 head on the executor (`draft`), and the last step's layout it reads.
    draft_head: bool,
    last_plan: PlanBufs,
    /// With a head: the sequences whose filled pages enter the prefix cache once the head has
    /// written its cache for them (the draft call, or before anything else reads the cache).
    entering: Vec<SeqId>,
    /// LoRA adapter slots on the executor (0: none).
    lora_slots: usize,
}

impl Paged {
    /// `prefix_cache`: keep full pages for later sequences with the same prefix (models whose
    /// whole per-sequence state is the paged cache; see `prefix.rs`), backed by host pages when
    /// `opts.host_cache_tokens` asks for them.
    pub fn new(mut ex: Executor, vocab: usize, g: &Geometry, opts: &LoadOptions, prefix_cache: bool) -> Result<Self> {
        let cache = prefix_cache && !opts.no_prefix_cache;
        let n_host = if cache { opts.host_cache_tokens.div_ceil(g.page) } else { 0 };
        if opts.host_cache_tokens > 0 && !cache {
            tracing::warn!("--host-cache-tokens ignored: this model runs without a prefix cache");
        }
        let pool = if n_host > 0 {
            let bytes = ex.page_bytes();
            ensure!(bytes > 0, "--host-cache-tokens: this model's executor has no host tier");
            ensure!(
                n_host > g.n_pages,
                "--host-cache-tokens ({}) must exceed the device cache ({} positions): the host tier holds every \
                 cached page",
                opts.host_cache_tokens,
                g.n_pages * g.page
            );
            let t = std::time::Instant::now();
            ex.reserve_host_pages(n_host).map_err(|e| anyhow!("pinning the host cache: {e}"))?;
            tracing::info!(
                "host cache: {} positions, {:.1} GB pinned in {:.1} s",
                n_host * g.page,
                (n_host * bytes) as f64 / 1e9,
                t.elapsed().as_secs_f64()
            );
            PagePool::with_host(g.n_pages, n_host, g.page)
        } else {
            PagePool::new(g.n_pages, g.page, cache)
        };
        Ok(Paged {
            ex,
            vocab,
            page: g.page,
            max_pages: g.max_pages,
            step_tokens: opts.step_tokens,
            max_rows: g.max_rows,
            pool,
            seqs: (0..opts.max_seqs).map(|_| SeqState::default()).collect(),
            page_table: vec![0; opts.max_seqs * g.max_pages],
            last: None,
            copied: vec![0; g.n_pages],
            loaded: vec![0; g.n_pages],
            copy_lag: opts.host_copy_lag,
            draft_head: false,
            last_plan: PlanBufs::default(),
            entering: Vec::new(),
            lora_slots: 0,
        })
    }

    /// The executor carries an EAGLE-3 head: `draft` runs, and the prefix cache keys each page
    /// with the token after it (the head's cache of its last position read that token).
    pub fn set_draft_head(&mut self, on: bool) {
        self.draft_head = on;
        if on && self.pool.enabled() {
            self.pool.key_next_token();
        }
    }

    /// Tokens a prefix-cache key holds past its page (1 with a draft head).
    fn key_extra(&self) -> usize {
        self.pool.key_len() - self.page
    }

    pub fn draft_head(&self) -> bool {
        self.draft_head
    }

    pub fn max_context(&self) -> usize {
        self.max_pages * self.page
    }

    pub fn capacity(&self) -> Capacity {
        Capacity {
            max_seqs: self.seqs.len(),
            free_seqs: self.seqs.iter().filter(|s| !s.used).count(),
            free_cache_tokens: self.pool.available() * self.page,
            prefix_cached_tokens: self.pool.cached() * self.page,
            host_cached_tokens: self.pool.host_cached() * self.page,
        }
    }

    pub fn alloc_seq(&mut self) -> Option<SeqId> {
        Some(self.alloc_seq_cached(&[])?.0)
    }

    /// A sequence starting with the longest cached prefix of `prompt` in whole pages, leaving at
    /// least the last prompt token to compute. Returns it and its length.
    pub fn alloc_seq_cached(&mut self, prompt: &[u32]) -> Option<(SeqId, usize)> {
        self.enter_pending_logged();
        let id = self.seqs.iter().position(|s| !s.used)?;
        let n = (prompt.len().saturating_sub(1) / self.page).min(self.max_pages);
        // (a key's token past its page is at most the prompt's last)
        let pages = self.pool.lookup(&prompt[..(n * self.page + self.key_extra()).min(prompt.len())]);
        let len = pages.len() * self.page;
        for (i, &pg) in pages.iter().enumerate() {
            self.page_table[id * self.max_pages + i] = pg as i32;
        }
        let tokens = if self.pool.enabled() { prompt[..len].to_vec() } else { Vec::new() };
        let ready = pages.iter().map(|&pg| self.loaded[pg as usize]).max().unwrap_or(0);
        self.seqs[id] =
            SeqState { used: true, fresh: true, len, shared: pages.len(), pages, tokens, ready, ..Default::default() };
        if self.lora_slots > 0 {
            // a slot's last sequence may have run an adapter: this one runs the base model until told
            self.ex.set_seq_lora(id, None).ok()?;
        }
        Some((id, len))
    }

    /// The positions of `prompt` right after what fresh sequence `seq` holds that only the host
    /// tier has (whole pages, the last prompt token left to compute).
    pub fn host_cached(&self, seq: SeqId, prompt: &[u32]) -> usize {
        let s = &self.seqs[seq];
        let end = self.cacheable(prompt);
        if !self.pool.host_tier() || s.len != s.shared * self.page || s.len >= end || prompt[..s.len] != s.tokens {
            return 0;
        }
        self.pool.peek(s.pages.last().copied(), &prompt[s.len..end + self.key_extra()]) * self.page
    }

    /// Extend fresh sequence `seq` by up to `n` positions of `prompt` from the host tier (as
    /// `host_cached` counts them), loading them into device pages. Returns the positions added.
    pub fn load_cached(&mut self, seq: SeqId, prompt: &[u32], n: usize) -> Result<usize> {
        if self.host_cached(seq, prompt) == 0 {
            return Ok(0);
        }
        let (page, end, extra) = (self.page, self.cacheable(prompt), self.key_extra());
        let s = &self.seqs[seq];
        let got = self.pool.load(s.pages.last().copied(), &prompt[s.len..end + extra], n / page);
        let (mut dev, mut host) = (Vec::new(), Vec::new());
        let s = &mut self.seqs[seq];
        for &(pg, from) in &got {
            self.page_table[seq * self.max_pages + s.pages.len()] = pg as i32;
            s.pages.push(pg);
            match from {
                Some(h) => {
                    dev.push(pg as i32);
                    host.push(h as i32);
                }
                None => s.ready = s.ready.max(self.loaded[pg as usize]),
            }
        }
        let added = got.len() * page;
        s.shared += got.len();
        s.tokens.extend_from_slice(&prompt[s.len..s.len + added]);
        s.len += added;
        if !dev.is_empty() {
            let t0 = std::time::Instant::now();
            let t = self.ex.copy_pages(false, &dev, &host).map_err(|e| anyhow!("loading from the host cache: {e}"))?;
            tracing::debug!(target: "fullm::copies", pages = dev.len(), ticket = t, issue_ms = t0.elapsed().as_secs_f64() * 1e3, "load");
            for &pg in &dev {
                self.copied[pg as usize] = t;
                self.loaded[pg as usize] = t;
            }
            s.ready = t;
            s.lag = self.copy_lag;
        }
        Ok(added)
    }

    /// Whether `seq`'s pages hold their cache: the loads it waits for have completed.
    pub fn seq_ready(&mut self, seq: SeqId) -> bool {
        let s = &mut self.seqs[seq];
        if s.lag > 0 {
            s.lag -= 1;
            return false;
        }
        s.ready == 0 || self.ex.copy_done(s.ready)
    }

    /// Block until the sequence that has waited longest for its loads can run: copies complete
    /// in ticket order, so its ticket is the smallest still pending, and the others keep loading.
    pub fn wait_copies(&mut self) -> Result<()> {
        let mut first: Option<usize> = None;
        for i in 0..self.seqs.len() {
            let s = &self.seqs[i];
            if !s.used || (s.lag == 0 && (s.ready == 0 || self.ex.copy_done(s.ready))) {
                continue;
            }
            if first.is_none_or(|f| self.seqs[f].ready > s.ready) {
                first = Some(i);
            }
        }
        let Some(i) = first else { return Ok(()) };
        self.seqs[i].lag = 0;
        self.ex.copy_wait(self.seqs[i].ready).map_err(|e| anyhow!("host cache copy failed: {e}"))
    }

    /// The prefix of `prompt` a cache hit may cover: whole pages, short of the last token.
    fn cacheable(&self, prompt: &[u32]) -> usize {
        (prompt.len().saturating_sub(1) / self.page).min(self.max_pages) * self.page
    }

    pub fn has_prefix_cache(&self) -> bool {
        self.pool.enabled()
    }

    /// Drop the cached pages no sequence holds; returns their positions.
    pub fn flush_prefix_cache(&mut self) -> usize {
        self.enter_pending_logged();
        self.pool.flush() * self.page
    }

    /// LoRA slots the executor reserved (`reserve_lora`).
    pub fn lora_slots(&self) -> usize {
        self.lora_slots
    }

    pub fn set_lora_slots(&mut self, n: usize) {
        self.lora_slots = n;
    }

    pub fn set_seq_lora(&mut self, seq: SeqId, slot: Option<usize>) -> Result<()> {
        if self.lora_slots == 0 {
            ensure!(slot.is_none(), "this model serves no LoRA adapters (serve --max-loras)");
            return Ok(());
        }
        ensure!(slot.is_none_or(|s| s < self.lora_slots), "no LoRA slot {slot:?}");
        self.ex.set_seq_lora(seq, slot).map_err(|e| anyhow!(e))?;
        self.seqs[seq].lora = slot;
        Ok(())
    }

    pub fn executor(&mut self) -> &mut Executor {
        &mut self.ex
    }

    pub fn free_seq(&mut self, seq: SeqId) {
        self.enter_pending_logged();
        if let Some(s) = self.seqs.get_mut(seq) {
            if s.used {
                let s = std::mem::take(s);
                self.pool.release(&s.pages[..s.shared]);
                for &pg in &s.pages[s.shared..] {
                    self.pool.give_back(pg);
                }
            }
        }
    }

    pub fn can_grow(&self, seq: SeqId, n: usize) -> bool {
        let s = &self.seqs[seq];
        let want = self.pages_for(s.len + n);
        want <= self.max_pages && want.saturating_sub(s.pages.len()) <= self.pool.available()
    }

    pub fn page_tokens(&self) -> usize {
        self.page
    }

    pub fn last_step(&self) -> Option<StepStats> {
        self.last
    }

    pub fn forward(&mut self, batch: &[SeqChunk<'_>]) -> Result<Vec<Vec<f32>>> {
        let plan = self.plan(batch)?;
        let v = self.vocab;
        let mut logits = vec![0.0f32; plan.rows.len() * v];
        let stats = self
            .ex
            .step(&plan.as_step(&self.page_table), &mut logits)
            .map_err(|e| anyhow!("executor step failed: {e}"))?;
        self.commit(batch, stats)?;
        self.last_plan = plan;
        if logits.iter().fold(false, |bad, x| bad | !x.is_finite()) {
            bail!("non-finite logits from the executor");
        }
        let mut out = Vec::with_capacity(batch.len());
        let mut r = 0;
        for ch in batch {
            let k = if ch.all_logits { ch.tokens.len() } else { 1 };
            out.push(logits[r * v..(r + k) * v].to_vec());
            r += k;
        }
        Ok(out)
    }

    pub fn forward_flat(&mut self, batch: &[SeqChunk<'_>], out: &mut Vec<f32>) -> Result<()> {
        let plan = self.plan(batch)?;
        // the executor writes every row: growing fills only the new part, shrinking keeps the memory
        out.resize(plan.rows.len() * self.vocab, 0.0);
        let stats =
            self.ex.step(&plan.as_step(&self.page_table), out).map_err(|e| anyhow!("executor step failed: {e}"))?;
        self.commit(batch, stats)?;
        self.last_plan = plan;
        if out.iter().fold(false, |bad, x| bad | !x.is_finite()) {
            bail!("non-finite logits from the executor");
        }
        Ok(())
    }

    pub fn forward_sample(
        &mut self,
        batch: &[SeqChunk<'_>],
        sample: &[SampleRow],
        io: &mut SampleIo<'_>,
    ) -> Result<Vec<u32>> {
        let plan = self.plan(batch)?;
        let mut toks = vec![0i32; plan.rows.len()];
        let stats = self
            .ex
            .step_sample(&plan.as_step(&self.page_table), sample, io, &mut toks)
            .map_err(|e| anyhow!("executor step failed: {e}"))?;
        self.commit(batch, stats)?;
        self.last_plan = plan;
        Ok(toks.into_iter().map(|t| t as u32).collect())
    }

    pub fn forward_greedy(&mut self, batch: &[SeqChunk<'_>]) -> Result<Vec<u32>> {
        let plan = self.plan(batch)?;
        let mut toks = vec![0i32; plan.rows.len()];
        let stats = self
            .ex
            .step_argmax(&plan.as_step(&self.page_table), &mut toks)
            .map_err(|e| anyhow!("executor step failed: {e}"))?;
        self.commit(batch, stats)?;
        self.last_plan = plan;
        Ok(toks.into_iter().map(|t| t as u32).collect())
    }

    fn pages_for(&self, len: usize) -> usize {
        len.div_ceil(self.page)
    }

    /// Grant `seq` pages from the pool (the caller checked there are enough) until it holds
    /// `want`. Returns the copy ticket the new pages wait for (their last host-tier copy).
    fn grow_to(&mut self, seq: SeqId, want: usize) -> u64 {
        let mut after = 0;
        while self.seqs[seq].pages.len() < want {
            let pg = self.pool.take().expect("reserved by the caller");
            after = after.max(self.copied[pg as usize]);
            let s = &mut self.seqs[seq];
            self.page_table[seq * self.max_pages + s.pages.len()] = pg as i32;
            s.pages.push(pg);
        }
        after
    }

    /// EAGLE-3 (`Model::draft`): each request's sequence feeds the head the positions its chunk of
    /// the last step added (after `accept`), each with the token after it (`next` after the last),
    /// then drafts up to `k` tokens to follow `next`, as many as the cache can take.
    pub fn draft(&mut self, reqs: &[DraftReq]) -> Result<Vec<Vec<u32>>> {
        ensure!(self.draft_head, "this model has no draft head");
        if reqs.is_empty() {
            return Ok(Vec::new());
        }
        struct Item {
            seq: SeqId,
            pos0: usize,
            tokens: Vec<i32>,
            rows: Vec<i32>,
            k: usize,
        }
        let lp = &self.last_plan;
        let mut seen = vec![false; self.seqs.len()];
        let mut items = Vec::with_capacity(reqs.len());
        for r in reqs {
            ensure!(r.seq < self.seqs.len() && self.seqs[r.seq].used, "sequence {} is not allocated", r.seq);
            ensure!(!std::mem::replace(&mut seen[r.seq], true), "sequence {} appears twice in one draft call", r.seq);
            ensure!((r.next as usize) < self.vocab, "token id out of vocabulary");
            let s = &self.seqs[r.seq];
            ensure!(s.draft == 0, "sequence {}: the last step's drafts are not yet accepted", r.seq);
            let c = lp
                .segs
                .chunks_exact(3)
                .find(|c| c[0] == r.seq as i32)
                .with_context(|| format!("sequence {} was not in the last step", r.seq))?;
            let (off, count) = (c[1] as usize, c[2] as usize);
            let pos0 = lp.positions[off] as usize;
            let n = s.len.checked_sub(pos0).filter(|n| (1..=count).contains(n)).with_context(|| {
                format!("sequence {} is at {}; its chunk of the last step began at {pos0}", r.seq, s.len)
            })?;
            let mut tokens = lp.tokens[off + 1..off + n].to_vec();
            tokens.push(r.next as i32);
            items.push(Item { seq: r.seq, pos0, tokens, rows: (off..off + n).map(|x| x as i32).collect(), k: r.k });
        }
        // draft i >= 2 writes the head's cache at len + i - 2, after the positions the sequence holds
        let mut after = 0;
        for it in &mut items {
            let have = self.seqs[it.seq].pages.len();
            let len = self.seqs[it.seq].len;
            let fits = |k: usize| {
                let want = self.pages_for(len + k.saturating_sub(1));
                want <= self.max_pages && want.saturating_sub(have) <= self.pool.available()
            };
            while it.k > 1 && !fits(it.k) {
                it.k -= 1;
            }
            let want = self.pages_for(len + it.k.saturating_sub(1));
            after = after.max(self.grow_to(it.seq, want));
        }
        if after > 0 {
            self.ex.copy_order(after).map_err(|e| anyhow!("ordering the draft after host cache copies: {e}"))?;
        }
        let seqs: Vec<DraftSeq<'_>> = items
            .iter()
            .map(|it| DraftSeq {
                slot: it.seq as i32,
                pos0: it.pos0 as i32,
                tokens: &it.tokens,
                rows: &it.rows,
                k: it.k,
            })
            .collect();
        let out = self.ex.draft(&seqs, &self.page_table).map_err(|e| anyhow!("draft head failed: {e}"));
        self.enter_pending()?;
        Ok(out?.into_iter().map(|d| d.into_iter().map(|t| t as u32).collect()).collect())
    }

    /// Validate a batch, grant it cache pages and lay out its step plan.
    fn plan(&mut self, batch: &[SeqChunk<'_>]) -> Result<PlanBufs> {
        self.enter_pending()?;
        let n: usize = batch.iter().map(|c| c.tokens.len()).sum();
        let rows_wanted: usize = batch.iter().map(|c| if c.all_logits { c.tokens.len() } else { 1 }).sum();
        ensure!(n > 0, "empty step");
        ensure!(n <= self.step_tokens, "step carries {n} tokens; the executor was sized for {}", self.step_tokens);
        ensure!(rows_wanted <= self.max_rows, "step asks for {rows_wanted} logits rows; at most {}", self.max_rows);
        let mut seen = vec![false; self.seqs.len()];
        for ch in batch {
            ensure!(ch.seq < self.seqs.len() && self.seqs[ch.seq].used, "sequence {} is not allocated", ch.seq);
            ensure!(!std::mem::replace(&mut seen[ch.seq], true), "sequence {} appears twice in one step", ch.seq);
            ensure!(!ch.tokens.is_empty(), "empty chunk for sequence {}", ch.seq);
            ensure!(
                ch.pos == self.seqs[ch.seq].len,
                "sequence {} is at {}, chunk starts at {}",
                ch.seq,
                self.seqs[ch.seq].len,
                ch.pos
            );
            ensure!(ch.tokens.iter().all(|&t| (t as usize) < self.vocab), "token id out of vocabulary");
            ensure!(
                ch.draft == 0 || (ch.draft < ch.tokens.len() && ch.all_logits),
                "sequence {}: {} drafts in a chunk of {} (a chunk with drafts asks for all logits)",
                ch.seq,
                ch.draft,
                ch.tokens.len()
            );
            ensure!(self.seqs[ch.seq].draft == 0, "sequence {} has drafts not yet accepted", ch.seq);
            ensure!(
                self.pages_for(ch.pos + ch.tokens.len()) <= self.max_pages,
                "sequence {} exceeds the maximum context",
                ch.seq
            );
        }
        let need: usize = batch
            .iter()
            .map(|ch| self.pages_for(ch.pos + ch.tokens.len()).saturating_sub(self.seqs[ch.seq].pages.len()))
            .sum();
        ensure!(need <= self.pool.available(), "cache exhausted: need {need} pages, {} free", self.pool.available());
        // the step runs after the loads its sequences read and the copies of the pages it writes
        let (mut read_after, mut write_after) = (0, 0);
        for ch in batch {
            let want = self.pages_for(ch.pos + ch.tokens.len());
            read_after = read_after.max(self.seqs[ch.seq].ready);
            write_after = write_after.max(self.grow_to(ch.seq, want));
        }
        let after = read_after.max(write_after);
        if after > 0 {
            if tracing::enabled!(target: "fullm::copies", tracing::Level::DEBUG) && !self.ex.copy_done(after) {
                let pending_read = read_after > 0 && !self.ex.copy_done(read_after);
                tracing::debug!(target: "fullm::copies", read_after, write_after, pending_read, "step waits for copies");
            }
            self.ex.copy_order(after).map_err(|e| anyhow!("ordering the step after host cache copies: {e}"))?;
        }

        let mut tokens = Vec::with_capacity(n);
        let mut positions = Vec::with_capacity(n);
        let mut token_slot = Vec::with_capacity(n);
        let mut segs = Vec::with_capacity(3 * batch.len());
        let mut rows = Vec::with_capacity(rows_wanted);
        let mut reset = Vec::new();
        for ch in batch {
            let off = tokens.len();
            segs.extend_from_slice(&[ch.seq as i32, off as i32, ch.tokens.len() as i32]);
            for (i, &t) in ch.tokens.iter().enumerate() {
                tokens.push(t as i32);
                positions.push((ch.pos + i) as i32);
                token_slot.push(ch.seq as i32);
            }
            if ch.all_logits {
                rows.extend((off..off + ch.tokens.len()).map(|r| r as i32));
            } else {
                rows.push((off + ch.tokens.len() - 1) as i32);
            }
            if self.seqs[ch.seq].fresh {
                reset.push(ch.seq as i32);
            }
        }

        Ok(PlanBufs { tokens, positions, token_slot, segs, rows, reset })
    }

    /// After a successful step: the sequences advanced (their drafts wait for `accept`), and the
    /// pages they filled enter the prefix cache (and are written through to the host tier).
    fn commit(&mut self, batch: &[SeqChunk<'_>], stats: StepStats) -> Result<()> {
        self.last = Some(stats);
        let enabled = self.pool.enabled();
        for ch in batch {
            let s = &mut self.seqs[ch.seq];
            let keep = ch.tokens.len() - ch.draft;
            s.len += keep;
            s.fresh = false;
            s.draft = ch.draft;
            if enabled {
                let lora = s.lora;
                s.tokens.extend(ch.tokens[..keep].iter().map(|&t| crate::models::lora_key(t, lora)));
                s.draft_tokens.clear();
                s.draft_tokens.extend(ch.tokens[keep..].iter().map(|&t| crate::models::lora_key(t, lora)));
            }
        }
        self.enter_after_draft(batch.iter().map(|c| c.seq))
    }

    /// The pages `seqs` filled enter the prefix cache: now, or with a head once it has written
    /// its cache for them (a page in the cache, or written through to the host tier, carries
    /// every layer's cache, the head's too).
    fn enter_after_draft(&mut self, seqs: impl Iterator<Item = SeqId>) -> Result<()> {
        if !self.draft_head {
            return self.enter_cache(seqs);
        }
        if self.pool.enabled() {
            for q in seqs {
                if !self.entering.contains(&q) {
                    self.entering.push(q);
                }
            }
        }
        Ok(())
    }

    fn enter_pending(&mut self) -> Result<()> {
        if self.entering.is_empty() {
            return Ok(());
        }
        let seqs = std::mem::take(&mut self.entering);
        let live: Vec<SeqId> = seqs.into_iter().filter(|&q| self.seqs[q].used).collect();
        self.enter_cache(live.into_iter())
    }

    /// `enter_pending` where an error has no caller to go to (a failed write-through only costs
    /// the host tier those pages).
    fn enter_pending_logged(&mut self) {
        if let Err(e) = self.enter_pending() {
            tracing::error!("prefix cache: {e:#}");
        }
    }

    /// The first `n` drafts of `seq`'s last step were right: it grows by them, and the pages
    /// they fill enter the prefix cache. Positions past them hold the rejected drafts' cache,
    /// written again as the sequence reaches them.
    pub fn accept(&mut self, seq: SeqId, n: usize) -> Result<()> {
        ensure!(seq < self.seqs.len() && self.seqs[seq].used, "sequence {seq} is not allocated");
        let s = &mut self.seqs[seq];
        ensure!(n <= s.draft, "sequence {seq}: accepting {n} of {} drafts", s.draft);
        s.len += n;
        s.draft = 0;
        if self.pool.enabled() {
            let d = std::mem::take(&mut s.draft_tokens);
            s.tokens.extend_from_slice(&d[..n]);
            s.draft_tokens = d;
            s.draft_tokens.clear();
        }
        self.enter_after_draft(std::iter::once(seq))
    }

    /// The pages `seqs` have filled since they last entered the prefix cache enter it now (and
    /// are written through to the host tier).
    fn enter_cache(&mut self, seqs: impl Iterator<Item = SeqId>) -> Result<()> {
        if !self.pool.enabled() {
            return Ok(());
        }
        let (page, extra) = (self.page, self.key_extra());
        let (mut dev, mut host) = (Vec::new(), Vec::new());
        for seq in seqs {
            let s = &mut self.seqs[seq];
            // (with a draft head, once the sequence also holds the token after the page)
            while (s.shared + 1) * page + extra <= s.len {
                let i = s.shared;
                let parent = i.checked_sub(1).map(|j| s.pages[j]);
                match self.pool.insert(parent, &s.tokens[i * page..(i + 1) * page + extra], s.pages[i]) {
                    Insert::Added(to) => {
                        if let Some(h) = to {
                            dev.push(s.pages[i] as i32);
                            host.push(h as i32);
                        }
                    }
                    Insert::Shared(have) => {
                        // another sequence filled the same page first: use its copy, free ours
                        self.pool.give_back(s.pages[i]);
                        s.pages[i] = have;
                        s.ready = s.ready.max(self.loaded[have as usize]);
                        self.page_table[seq * self.max_pages + i] = have as i32;
                    }
                    Insert::Full => break,
                }
                s.shared += 1;
            }
        }
        if !dev.is_empty() {
            let t0 = std::time::Instant::now();
            let t = self.ex.copy_pages(true, &dev, &host).map_err(|e| anyhow!("host cache write-through: {e}"))?;
            tracing::debug!(target: "fullm::copies", pages = dev.len(), ticket = t, issue_ms = t0.elapsed().as_secs_f64() * 1e3, "save");
            for &pg in &dev {
                self.copied[pg as usize] = t;
            }
        }
        Ok(())
    }
}

/// The host arrays of one step plan (borrowing the page table at call time).
#[derive(Default)]
struct PlanBufs {
    tokens: Vec<i32>,
    positions: Vec<i32>,
    token_slot: Vec<i32>,
    segs: Vec<i32>,
    rows: Vec<i32>,
    reset: Vec<i32>,
}

impl PlanBufs {
    fn as_step<'a>(&'a self, page_table: &'a [i32]) -> StepPlan<'a> {
        StepPlan {
            tokens: &self.tokens,
            positions: &self.positions,
            token_slot: &self.token_slot,
            segs: &self.segs,
            page_table,
            rows: &self.rows,
            reset_slots: &self.reset,
        }
    }
}

/// The `Model` methods a `Paged` field answers; `$field` names it in the implementing type.
#[macro_export]
macro_rules! delegate_paged {
    ($field:ident) => {
        fn max_context(&self) -> usize {
            self.$field.max_context()
        }
        fn capacity(&self) -> $crate::models::Capacity {
            self.$field.capacity()
        }
        fn alloc_seq(&mut self) -> Option<$crate::models::SeqId> {
            self.$field.alloc_seq()
        }
        fn alloc_seq_cached(&mut self, prompt: &[u32]) -> Option<($crate::models::SeqId, usize)> {
            self.$field.alloc_seq_cached(prompt)
        }
        fn has_prefix_cache(&self) -> bool {
            self.$field.has_prefix_cache()
        }
        fn flush_prefix_cache(&mut self) -> usize {
            self.$field.flush_prefix_cache()
        }
        fn host_cached(&self, seq: $crate::models::SeqId, prompt: &[u32]) -> usize {
            self.$field.host_cached(seq, prompt)
        }
        fn load_cached(&mut self, seq: $crate::models::SeqId, prompt: &[u32], n: usize) -> anyhow::Result<usize> {
            self.$field.load_cached(seq, prompt, n)
        }
        fn seq_ready(&mut self, seq: $crate::models::SeqId) -> bool {
            self.$field.seq_ready(seq)
        }
        fn wait_copies(&mut self) -> anyhow::Result<()> {
            self.$field.wait_copies()
        }
        fn free_seq(&mut self, seq: $crate::models::SeqId) {
            self.$field.free_seq(seq)
        }
        fn can_grow(&self, seq: $crate::models::SeqId, n: usize) -> bool {
            self.$field.can_grow(seq, n)
        }
        fn page_tokens(&self) -> usize {
            self.$field.page_tokens()
        }
        fn forward(&mut self, batch: &[$crate::models::SeqChunk<'_>]) -> anyhow::Result<Vec<Vec<f32>>> {
            self.$field.forward(batch)
        }
        fn forward_greedy(&mut self, batch: &[$crate::models::SeqChunk<'_>]) -> anyhow::Result<Vec<u32>> {
            self.$field.forward_greedy(batch)
        }
        fn forward_flat(&mut self, batch: &[$crate::models::SeqChunk<'_>], out: &mut Vec<f32>) -> anyhow::Result<()> {
            self.$field.forward_flat(batch, out)
        }
        fn lora_slots(&self) -> usize {
            self.$field.lora_slots()
        }
        fn set_seq_lora(&mut self, seq: $crate::models::SeqId, slot: Option<usize>) -> anyhow::Result<()> {
            self.$field.set_seq_lora(seq, slot)
        }
        fn device_sampling(&self) -> bool {
            true
        }
        fn forward_sample(
            &mut self,
            batch: &[$crate::models::SeqChunk<'_>],
            sample: &[$crate::models::SampleRow],
            io: &mut $crate::models::SampleIo<'_>,
        ) -> anyhow::Result<Vec<u32>> {
            self.$field.forward_sample(batch, sample, io)
        }
        fn accept(&mut self, seq: $crate::models::SeqId, n: usize) -> anyhow::Result<()> {
            self.$field.accept(seq, n)
        }
        fn draft_head(&self) -> bool {
            self.$field.draft_head()
        }
        fn draft(&mut self, reqs: &[$crate::models::DraftReq]) -> anyhow::Result<Vec<Vec<u32>>> {
            self.$field.draft(reqs)
        }
        fn last_step(&self) -> Option<fullm_kernels::exec::StepStats> {
            self.$field.last_step()
        }
    };
}
