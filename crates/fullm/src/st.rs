// SPDX-License-Identifier: Apache-2.0
//! A memory-mapped, multi-shard safetensors reader.
//!
//! Every shard is mapped once; a tensor is a slice of its mapping. Nothing is copied
//! until a caller asks for a widened or aligned copy, so binding the 1.5 TB Kimi K3
//! checkpoint costs address space, not memory.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use memmap2::Mmap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    F8E4M3,
    F8E5M2,
    Bool,
}

impl Dtype {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "F64" => Dtype::F64,
            "F32" => Dtype::F32,
            "F16" => Dtype::F16,
            "BF16" => Dtype::BF16,
            "I64" => Dtype::I64,
            "I32" => Dtype::I32,
            "I16" => Dtype::I16,
            "I8" => Dtype::I8,
            "U8" => Dtype::U8,
            "F8_E4M3" => Dtype::F8E4M3,
            "F8_E5M2" => Dtype::F8E5M2,
            "BOOL" => Dtype::Bool,
            other => bail!("unsupported safetensors dtype {other}"),
        })
    }

    pub fn size(self) -> usize {
        match self {
            Dtype::F64 | Dtype::I64 => 8,
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F16 | Dtype::BF16 | Dtype::I16 => 2,
            Dtype::I8 | Dtype::U8 | Dtype::F8E4M3 | Dtype::F8E5M2 | Dtype::Bool => 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    shard: usize,
    /// Absolute byte range within the shard file.
    pub start: usize,
    pub end: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn nbytes(&self) -> usize {
        self.end - self.start
    }
}

struct Shard {
    path: PathBuf,
    map: Mmap,
}

pub struct SafeTensors {
    shards: Vec<Shard>,
    index: HashMap<String, TensorInfo>,
}

/// A borrowed tensor.
#[derive(Clone, Copy)]
pub struct View<'a> {
    pub name: &'a str,
    pub info: &'a TensorInfo,
    pub bytes: &'a [u8],
}

impl SafeTensors {
    /// Map every `*.safetensors` file in `dir`.
    pub fn open_dir(dir: &Path) -> Result<Arc<Self>> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("reading {}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            bail!("no .safetensors files in {}", dir.display());
        }
        let mut shards = Vec::with_capacity(files.len());
        let mut index = HashMap::new();
        for (si, path) in files.into_iter().enumerate() {
            let f = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
            // SAFETY: checkpoint shards are treated as read-only for the process lifetime.
            let map = unsafe { Mmap::map(&f) }.with_context(|| format!("mapping {}", path.display()))?;
            parse_header(&map, si, &path, &mut index)?;
            shards.push(Shard { path, map });
        }
        Ok(Arc::new(SafeTensors { shards, index }))
    }

    pub fn get(&self, name: &str) -> Option<View<'_>> {
        let (k, info) = self.index.get_key_value(name)?;
        let bytes = &self.shards[info.shard].map[info.start..info.end];
        Some(View { name: k, info, bytes })
    }

    /// Like `get`, but a missing tensor is an error naming the tensor.
    pub fn need(&self, name: &str) -> Result<View<'_>> {
        self.get(name).with_context(|| format!("checkpoint is missing tensor {name}"))
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(|s| s.as_str())
    }

    pub fn shard_path(&self, info: &TensorInfo) -> &Path {
        &self.shards[info.shard].path
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }
}

fn parse_header(map: &[u8], shard: usize, path: &Path, index: &mut HashMap<String, TensorInfo>) -> Result<()> {
    if map.len() < 8 {
        bail!("{}: too short for a safetensors header", path.display());
    }
    let hlen = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
    let data0 = 8usize.checked_add(hlen).filter(|&e| e <= map.len());
    let Some(data0) = data0 else { bail!("{}: header length {hlen} overruns the file", path.display()) };
    let header: serde_json::Value =
        serde_json::from_slice(&map[8..data0]).with_context(|| format!("{}: header is not JSON", path.display()))?;
    let obj = header.as_object().with_context(|| format!("{}: header is not an object", path.display()))?;
    for (name, v) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = Dtype::parse(v["dtype"].as_str().unwrap_or(""))?;
        let shape: Vec<usize> = v["shape"]
            .as_array()
            .with_context(|| format!("{name}: no shape"))?
            .iter()
            .map(|d| d.as_u64().map(|x| x as usize))
            .collect::<Option<_>>()
            .with_context(|| format!("{name}: bad shape"))?;
        let offs = v["data_offsets"].as_array().with_context(|| format!("{name}: no data_offsets"))?;
        let off = |i: usize| offs.get(i).and_then(|x| x.as_u64()).map(|x| x as usize);
        let (a, b) = off(0).zip(off(1)).with_context(|| format!("{name}: data_offsets is not two offsets"))?;
        let (start, end) = (data0.saturating_add(a), data0.saturating_add(b));
        if a > b || end > map.len() {
            bail!("{}: tensor {name} spans {a}..{b}, outside the file", path.display());
        }
        let numel = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)).unwrap_or(usize::MAX);
        if numel.checked_mul(dtype.size()) != Some(b - a) {
            bail!(
                "{}: tensor {name} is {} bytes, shape {:?} of {:?} needs {}",
                path.display(),
                b - a,
                shape,
                dtype,
                numel.saturating_mul(dtype.size())
            );
        }
        if index.insert(name.clone(), TensorInfo { dtype, shape, shard, start, end }).is_some() {
            bail!("tensor {name} appears in more than one shard");
        }
    }
    Ok(())
}

impl View<'_> {
    pub fn numel(&self) -> usize {
        self.info.numel()
    }
    pub fn shape(&self) -> &[usize] {
        &self.info.shape
    }
    pub fn dtype(&self) -> Dtype {
        self.info.dtype
    }

    /// Widen to f32. Exact for F32/BF16/F16.
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        let b = self.bytes;
        Ok(match self.info.dtype {
            Dtype::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            Dtype::BF16 => {
                b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()
            }
            Dtype::F16 => {
                b.chunks_exact(2).map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect()
            }
            d => bail!("{}: cannot widen {d:?} to f32", self.name),
        })
    }

    /// Integer elements (I64 / I32), widened to i64.
    pub fn to_i64(&self) -> Result<Vec<i64>> {
        let b = self.bytes;
        Ok(match self.info.dtype {
            Dtype::I64 => b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect(),
            Dtype::I32 => b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64).collect(),
            d => bail!("{}: expected I64 or I32, found {d:?}", self.name),
        })
    }

    /// The raw bf16 elements, copied into an aligned buffer.
    pub fn to_bf16(&self) -> Result<Vec<u16>> {
        match self.info.dtype {
            Dtype::BF16 => Ok(self.bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()),
            // rounded as vLLM casts a checkpoint to the model's dtype
            Dtype::F32 | Dtype::F16 => Ok(self.to_f32()?.into_iter().map(f32_to_bf16).collect()),
            d => bail!("{}: expected BF16, found {d:?}", self.name),
        }
    }
}

/// Round-to-nearest-even f32 -> bf16 (NaN stays NaN).
pub fn f32_to_bf16(v: f32) -> u16 {
    let u = v.to_bits();
    if v.is_nan() {
        return ((u >> 16) | 0x40) as u16;
    }
    let round = 0x7FFF + ((u >> 16) & 1);
    ((u.wrapping_add(round)) >> 16) as u16
}
