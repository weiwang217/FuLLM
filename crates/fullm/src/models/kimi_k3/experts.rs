// SPDX-License-Identifier: Apache-2.0
//! Routed-expert storage. Experts are multiplied straight from their stored form: Kimi K3
//! ships MXFP4 (widening one would turn 17.5 MB into 132 MB, and a token touches 1,472 of
//! them); Kimi Linear ships bf16.
//!
//! `ExpertStore` is the seam between the model and the memory hierarchy. The mmap store
//! below serves experts from the checkpoint's own shards (page cache = DRAM tier); another
//! tier (a pinned DRAM arena, NVMe) would implement the same trait.

use std::sync::Arc;

use anyhow::{bail, Result};
use fullm_kernels::WeightRef;

use super::K3Config;
use crate::st::{Dtype, SafeTensors};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertFmt {
    /// `[rows][cols/2]` packed nibbles (low = even element) + `[rows][cols/32]` E8M0 scales.
    Mxfp4,
    Bf16,
}

/// gate (`w1`, `[inter][width]`), up (`w3`, `[inter][width]`), down (`w2`, `[width][inter]`),
/// where width is the latent (K3) or the hidden size (Kimi Linear).
#[derive(Clone, Copy)]
pub struct ExpertRef<'a> {
    pub w1: WeightRef<'a>,
    pub w3: WeightRef<'a>,
    pub w2: WeightRef<'a>,
}

impl ExpertRef<'_> {
    /// (data, scales) host pointers of w1, w3, w2, for the executor. Scales are null for bf16.
    pub fn raw_ptrs(&self) -> [*const u8; 6] {
        fn p(w: &WeightRef<'_>) -> (*const u8, *const u8) {
            match *w {
                WeightRef::Mxfp4 { packed, scales } => (packed.as_ptr(), scales.as_ptr()),
                WeightRef::Bf16(d) => (d.as_ptr() as *const u8, std::ptr::null()),
                _ => (std::ptr::null(), std::ptr::null()),
            }
        }
        let (a, b, c) = (p(&self.w1), p(&self.w3), p(&self.w2));
        [a.0, a.1, b.0, b.1, c.0, c.1]
    }
}

pub trait ExpertStore: Send + Sync {
    /// The expert's bytes, valid for as long as `&self` is borrowed.
    fn get(&self, layer: usize, expert: usize) -> Result<ExpertRef<'_>>;
    /// Hint that these experts are needed next; stores with a slower tier start reads.
    fn prefetch(&self, _layer: usize, _experts: &[u32]) {}
    fn format(&self) -> ExpertFmt;
    fn bytes_per_expert(&self) -> usize;
    fn describe(&self) -> String;
}

#[derive(Clone, Copy)]
struct Span {
    ptr: *const u8,
    len: usize,
}

impl Span {
    const NULL: Span = Span { ptr: std::ptr::null(), len: 0 };
}

/// Serves experts zero-copy from the mapped checkpoint shards.
pub struct MmapExperts {
    _st: Arc<SafeTensors>,
    fmt: ExpertFmt,
    /// `[layer][expert]` -> w1, w3, w2 as (data, scales). Empty for dense layers.
    index: Vec<Vec<[Span; 6]>>,
    bytes_per_expert: usize,
}

// The spans point into read-only mappings owned by `_st`.
unsafe impl Send for MmapExperts {}
unsafe impl Sync for MmapExperts {}

impl MmapExperts {
    pub fn new(st: Arc<SafeTensors>, c: &K3Config) -> Result<Self> {
        let (width, inter) = (c.latent, c.moe_inter);
        let first_moe = (0..c.n_layers).find(|&l| !c.is_dense(l));
        let base =
            |layer: usize, e: usize, w: &str| format!("{}layers.{layer}.block_sparse_moe.experts.{e}.{w}.", c.prefix);
        // The format is a fact about the checkpoint, checked against what the config expects.
        let fmt = match first_moe {
            Some(l) if st.get(&format!("{}weight_packed", base(l, 0, "w1"))).is_some() => ExpertFmt::Mxfp4,
            _ => ExpertFmt::Bf16,
        };
        if (fmt == ExpertFmt::Bf16) != c.expert_bf16 {
            bail!("routed experts are stored as {fmt:?}, but this model's config expects the other format");
        }
        // (name, rows, cols): logical shapes
        let shapes = [("w1", inter, width), ("w3", inter, width), ("w2", width, inter)];
        let mut index = Vec::with_capacity(c.n_layers);
        let mut problems = Vec::new();
        let mut bytes_per_expert = 0;
        for layer in 0..c.n_layers {
            if c.is_dense(layer) {
                index.push(Vec::new());
                continue;
            }
            let mut per = Vec::with_capacity(c.n_experts);
            for e in 0..c.n_experts {
                let mut spans = [Span::NULL; 6];
                let mut total = 0;
                for (k, (w, rows, cols)) in shapes.iter().enumerate() {
                    let parts: &[(&str, usize, usize)] = match fmt {
                        ExpertFmt::Mxfp4 => {
                            &[("weight_packed", 0, rows * cols / 2), ("weight_scale", 1, rows * cols / 32)]
                        }
                        ExpertFmt::Bf16 => &[("weight", 0, rows * cols * 2)],
                    };
                    for &(suffix, slot, want) in parts {
                        let name = format!("{}{suffix}", base(layer, e, w));
                        match st.get(&name) {
                            Some(v) if v.bytes.len() == want => {
                                if fmt == ExpertFmt::Bf16
                                    && (v.dtype() != Dtype::BF16 || v.bytes.as_ptr() as usize % 2 != 0)
                                {
                                    problems.push(format!("{name}: expected aligned BF16, found {:?}", v.dtype()));
                                }
                                spans[2 * k + slot] = Span { ptr: v.bytes.as_ptr(), len: v.bytes.len() };
                                total += v.bytes.len();
                            }
                            Some(v) => problems.push(format!("{name}: {} bytes, expected {want}", v.bytes.len())),
                            None => problems.push(format!("missing {name}")),
                        }
                    }
                }
                bytes_per_expert = total;
                per.push(spans);
            }
            index.push(per);
            if problems.len() > 12 {
                break;
            }
        }
        if !problems.is_empty() {
            bail!(
                "routed experts do not match the config:\n  {}",
                problems.into_iter().take(12).collect::<Vec<_>>().join("\n  ")
            );
        }
        Ok(MmapExperts { _st: st, fmt, index, bytes_per_expert })
    }
}

impl ExpertStore for MmapExperts {
    fn get(&self, layer: usize, expert: usize) -> Result<ExpertRef<'_>> {
        let Some(s) = self.index.get(layer).and_then(|v| v.get(expert)) else {
            bail!("no routed expert {expert} in layer {layer}");
        };
        // SAFETY: the spans point into mappings owned by `_st`, alive as long as `self`;
        // bf16 spans were checked for 2-byte alignment at construction.
        let w = |i: usize| -> WeightRef<'_> {
            unsafe {
                match self.fmt {
                    ExpertFmt::Mxfp4 => WeightRef::Mxfp4 {
                        packed: std::slice::from_raw_parts(s[i].ptr, s[i].len),
                        scales: std::slice::from_raw_parts(s[i + 1].ptr, s[i + 1].len),
                    },
                    ExpertFmt::Bf16 => {
                        WeightRef::Bf16(std::slice::from_raw_parts(s[i].ptr as *const u16, s[i].len / 2))
                    }
                }
            }
        };
        Ok(ExpertRef { w1: w(0), w3: w(2), w2: w(4) })
    }

    fn format(&self) -> ExpertFmt {
        self.fmt
    }

    fn bytes_per_expert(&self) -> usize {
        self.bytes_per_expert
    }

    fn describe(&self) -> String {
        format!("mmap (page cache), {:?}, {:.2} MB/expert", self.fmt, self.bytes_per_expert as f64 / 1e6)
    }
}
