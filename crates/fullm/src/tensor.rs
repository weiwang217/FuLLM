// SPDX-License-Identifier: Apache-2.0
//! Host-side weight matrices and the parallel linear layer used by the CPU backend.

use fullm_kernels::WeightRef;
use rayon::prelude::*;

/// Storage precision of a trunk matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    F32,
    Bf16,
    /// E4M3 with one f32 scale per output row.
    Fp8,
    /// Symmetric int8 with one f32 scale per output row (weight-only).
    Int8,
    /// E4M3 with one f32 scale per 128 x 128 block: resident routed experts only. The matrices
    /// go to the executor as bf16 and it quantises them once their stored layout (gate/up rows
    /// interleaved or not) is fixed, so the blocks are those of the stored matrix.
    #[serde(rename = "fp8b")]
    Fp8Block,
    /// Symmetric int4 with one f32 scale per row per 128 columns: resident routed experts only,
    /// quantised by the executor like Fp8Block (int4_group.h).
    #[serde(rename = "int4")]
    Int4Group,
}

pub enum MatData {
    F32(Vec<f32>),
    Bf16(Vec<u16>),
    Fp8 {
        q: Vec<u8>,
        scale: Vec<f32>,
    },
    Int8 {
        q: Vec<i8>,
        scale: Vec<f32>,
    },
    /// Int4 quantised elsewhere (a GPTQ checkpoint): one value in [-8, 7] per element, and a
    /// scale per row per 128 columns, `[ceil(cols/128)][rows]` (the executor's INT4G). Resident
    /// experts only: the executor packs them.
    Int4 {
        q: Vec<i8>,
        scale: Vec<f32>,
    },
}

/// A row-major `[rows][cols]` weight matrix (torch `Linear.weight` layout).
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: MatData,
}

impl Mat {
    pub fn bf16(rows: usize, cols: usize, w: Vec<u16>) -> Self {
        assert_eq!(w.len(), rows * cols);
        Mat { rows, cols, data: MatData::Bf16(w) }
    }

    pub fn f32(rows: usize, cols: usize, w: Vec<f32>) -> Self {
        assert_eq!(w.len(), rows * cols);
        Mat { rows, cols, data: MatData::F32(w) }
    }

    pub fn precision(&self) -> Precision {
        match self.data {
            MatData::F32(_) => Precision::F32,
            MatData::Bf16(_) => Precision::Bf16,
            MatData::Fp8 { .. } => Precision::Fp8,
            MatData::Int8 { .. } => Precision::Int8,
            MatData::Int4 { .. } => Precision::Int4Group,
        }
    }

    pub fn nbytes(&self) -> usize {
        match &self.data {
            MatData::F32(w) => w.len() * 4,
            MatData::Bf16(w) => w.len() * 2,
            MatData::Fp8 { q, scale } => q.len() + scale.len() * 4,
            MatData::Int8 { q, scale } => q.len() + scale.len() * 4,
            MatData::Int4 { q, scale } => q.len() / 2 + scale.len() * 4,
        }
    }

    /// Re-store a bf16 matrix at `p`. Only bf16 sources convert; anything else is kept
    /// (Fp8Block too: the executor quantises those itself).
    pub fn convert(self, p: Precision) -> Self {
        match p {
            Precision::Fp8 => self.to_fp8(),
            Precision::Int8 => self.to_int8(),
            _ => self,
        }
    }

    /// Re-store a bf16 matrix as per-row-scaled symmetric int8.
    pub fn to_int8(self) -> Self {
        let MatData::Bf16(w) = &self.data else { return self };
        let (rows, cols) = (self.rows, self.cols);
        let mut q = vec![0i8; rows * cols];
        let mut scale = vec![0f32; rows];
        let chunk = 64;
        q.par_chunks_mut(chunk * cols).zip(scale.par_chunks_mut(chunk)).enumerate().for_each(|(i, (qc, sc))| {
            let r0 = i * chunk;
            let r1 = (r0 + chunk).min(rows);
            fullm_kernels::quant_i8_rows(&w[r0 * cols..r1 * cols], cols, 0, r1 - r0, qc, sc);
        });
        Mat { rows, cols, data: MatData::Int8 { q, scale } }
    }

    /// Re-store a bf16 matrix as per-row-scaled E4M3.
    pub fn to_fp8(self) -> Self {
        let MatData::Bf16(w) = &self.data else { return self };
        let (rows, cols) = (self.rows, self.cols);
        let mut q = vec![0u8; rows * cols];
        let mut scale = vec![0f32; rows];
        let chunk = 64;
        q.par_chunks_mut(chunk * cols).zip(scale.par_chunks_mut(chunk)).enumerate().for_each(|(i, (qc, sc))| {
            let r0 = i * chunk;
            let r1 = (r0 + chunk).min(rows);
            let n = r1 - r0;
            fullm_kernels::quant_fp8_rows(&w[r0 * cols..r1 * cols], cols, 0, n, qc, sc);
        });
        Mat { rows, cols, data: MatData::Fp8 { q, scale } }
    }

    pub fn weight(&self) -> WeightRef<'_> {
        match &self.data {
            MatData::F32(w) => WeightRef::F32(w),
            MatData::Bf16(w) => WeightRef::Bf16(w),
            MatData::Fp8 { q, scale } => WeightRef::Fp8 { q, scale },
            MatData::Int8 { q, scale } => WeightRef::Int8 { q, scale },
            MatData::Int4 { .. } => panic!("int4 matrices are resident experts: the executor reads them (raw)"),
        }
    }

    /// The raw storage: executor format tag, element bytes, and per-row scales for 8-bit.
    pub fn raw(&self) -> (fullm_kernels::exec::Fmt, &[u8], Option<&[f32]>) {
        use fullm_kernels::exec::Fmt;
        fn bytes<T>(v: &[T]) -> &[u8] {
            // SAFETY: plain-old-data element types, viewed as bytes for an FFI copy.
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
        }
        match &self.data {
            MatData::F32(w) => (Fmt::F32, bytes(w), None),
            MatData::Bf16(w) => (Fmt::Bf16, bytes(w), None),
            MatData::Fp8 { q, scale } => (Fmt::Fp8, bytes(q), Some(scale)),
            MatData::Int8 { q, scale } => (Fmt::Int8, bytes(q), Some(scale)),
            MatData::Int4 { q, scale } => (Fmt::Int4Group, bytes(q), Some(scale)),
        }
    }

    /// Rows `r0..r1` as their own matrix view.
    pub fn rows_ref(&self, r0: usize, r1: usize) -> WeightRef<'_> {
        let c = self.cols;
        match &self.data {
            MatData::F32(w) => WeightRef::F32(&w[r0 * c..r1 * c]),
            MatData::Bf16(w) => WeightRef::Bf16(&w[r0 * c..r1 * c]),
            MatData::Fp8 { q, scale } => WeightRef::Fp8 { q: &q[r0 * c..r1 * c], scale: &scale[r0..r1] },
            MatData::Int8 { q, scale } => WeightRef::Int8 { q: &q[r0 * c..r1 * c], scale: &scale[r0..r1] },
            MatData::Int4 { .. } => panic!("int4 matrices are resident experts: the executor reads them (raw)"),
        }
    }

    /// Decode row `r` into `out` (len `cols`).
    pub fn row_f32(&self, r: usize, out: &mut [f32]) {
        let c = self.cols;
        match &self.data {
            MatData::F32(w) => out.copy_from_slice(&w[r * c..(r + 1) * c]),
            MatData::Bf16(w) => {
                for (o, &v) in out.iter_mut().zip(&w[r * c..(r + 1) * c]) {
                    *o = f32::from_bits((v as u32) << 16);
                }
            }
            MatData::Fp8 { q, scale } => {
                let t = fullm_kernels::fp8_e4m3_table();
                for (o, &v) in out.iter_mut().zip(&q[r * c..(r + 1) * c]) {
                    *o = t[v as usize] * scale[r];
                }
            }
            MatData::Int8 { q, scale } => {
                for (o, &v) in out.iter_mut().zip(&q[r * c..(r + 1) * c]) {
                    *o = v as f32 * scale[r];
                }
            }
            MatData::Int4 { q, scale } => {
                for (k, (o, &v)) in out.iter_mut().zip(&q[r * c..(r + 1) * c]).enumerate() {
                    *o = v as f32 * scale[(k / 128) * self.rows + r];
                }
            }
        }
    }
}

/// Raw pointer that may cross threads. Every use writes a disjoint region.
#[derive(Clone, Copy)]
pub(crate) struct SendPtr<T>(pub *mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    /// Use this rather than `.0` inside closures: edition-2021 closures capture the field,
    /// which is a bare `*mut T` and not `Send`.
    pub(crate) fn get(self) -> *mut T {
        self.0
    }
}

/// Output rows per parallel task. Small enough to balance, large enough that each task
/// streams a few hundred KB of weights.
fn chunk_rows(n: usize, k: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    let by_threads = n.div_ceil(threads * 4);
    let by_bytes = (256 * 1024 / k.max(1)).max(1);
    by_threads.max(by_bytes.min(n)).max(1).min(n.max(1))
}

/// `y[rows][w_rows] = x[rows][k] . W^T`, parallel over output rows of `w`.
pub fn linear_ref(x: &[f32], rows: usize, k: usize, w: WeightRef<'_>, n: usize, y: &mut [f32]) {
    assert_eq!(y.len(), rows * n);
    if rows == 0 || n == 0 {
        return;
    }
    let step = chunk_rows(n, k);
    let yp = SendPtr(y.as_mut_ptr());
    let ylen = y.len();
    (0..n.div_ceil(step)).into_par_iter().for_each(|i| {
        let n0 = i * step;
        let n1 = (n0 + step).min(n);
        // SAFETY: each task writes only columns n0..n1 of every row.
        let y = unsafe { std::slice::from_raw_parts_mut(yp.get(), ylen) };
        fullm_kernels::gemm(x, rows, k, w, n, n0, n1, y, n);
    });
}

pub fn linear(x: &[f32], rows: usize, w: &Mat, y: &mut [f32]) {
    linear_ref(x, rows, w.cols, w.weight(), w.rows, y)
}

/// Convenience: allocate the output.
pub fn linear_new(x: &[f32], rows: usize, w: &Mat) -> Vec<f32> {
    let mut y = vec![0.0; rows * w.rows];
    linear(x, rows, w, &mut y);
    y
}
