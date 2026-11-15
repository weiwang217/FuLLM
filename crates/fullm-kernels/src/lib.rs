// SPDX-License-Identifier: Apache-2.0
//! Safe wrappers over the C++ kernels in `csrc/`.
//!
//! The kernels never thread; callers split the output rows `[n0, n1)` across their own
//! workers. Every wrapper checks the slice lengths it is about to hand to C, because an
//! out-of-bounds read in a kernel produces plausible numbers rather than a crash.

use std::ffi::CStr;

pub mod exec;

mod ffi {
    extern "C" {
        pub fn sk_cpu_isa() -> *const std::os::raw::c_char;
        pub fn sk_fp8_e4m3_table() -> *const f32;
        pub fn sk_cpu_gemm_f32(x: *const f32, m: i64, k: i64, w: *const f32, n0: i64, n1: i64, y: *mut f32, ldy: i64);
        pub fn sk_cpu_gemm_bf16(x: *const f32, m: i64, k: i64, w: *const u16, n0: i64, n1: i64, y: *mut f32, ldy: i64);
        pub fn sk_cpu_gemm_fp8(
            x: *const f32,
            m: i64,
            k: i64,
            w: *const u8,
            row_scale: *const f32,
            n0: i64,
            n1: i64,
            y: *mut f32,
            ldy: i64,
        );
        pub fn sk_cpu_gemm_i8(
            x: *const f32,
            m: i64,
            k: i64,
            w: *const i8,
            row_scale: *const f32,
            n0: i64,
            n1: i64,
            y: *mut f32,
            ldy: i64,
        );
        pub fn sk_cpu_gemm_mxfp4(
            x: *const f32,
            m: i64,
            k: i64,
            packed: *const u8,
            scales: *const u8,
            n0: i64,
            n1: i64,
            y: *mut f32,
            ldy: i64,
        );
        pub fn sk_cpu_quant_fp8_rows(w: *const u16, k: i64, n0: i64, n1: i64, q: *mut u8, row_scale: *mut f32);
        pub fn sk_cpu_quant_i8_rows(w: *const u16, k: i64, n0: i64, n1: i64, q: *mut i8, row_scale: *mut f32);
    }
}

/// The SIMD path the CPU kernels dispatch to on this machine.
pub fn cpu_isa() -> &'static str {
    unsafe { CStr::from_ptr(ffi::sk_cpu_isa()) }.to_str().unwrap_or("?")
}

/// E4M3 code -> f32 for all 256 codes (NaN codes decode to 0).
pub fn fp8_e4m3_table() -> &'static [f32; 256] {
    unsafe { &*(ffi::sk_fp8_e4m3_table() as *const [f32; 256]) }
}

/// A borrowed weight matrix, row-major `[n][k]`.
#[derive(Clone, Copy)]
pub enum WeightRef<'a> {
    F32(&'a [f32]),
    Bf16(&'a [u16]),
    /// E4M3 codes plus one scale per output row.
    Fp8 {
        q: &'a [u8],
        scale: &'a [f32],
    },
    /// Symmetric int8 plus one scale per output row.
    Int8 {
        q: &'a [i8],
        scale: &'a [f32],
    },
    /// OCP MX FP4: `[n][k/2]` packed nibbles (low = even element), `[n][k/32]` E8M0 scales.
    Mxfp4 {
        packed: &'a [u8],
        scales: &'a [u8],
    },
}

impl WeightRef<'_> {
    fn check(&self, n: usize, k: usize) {
        let ok = match *self {
            WeightRef::F32(w) => w.len() >= n * k,
            WeightRef::Bf16(w) => w.len() >= n * k,
            WeightRef::Fp8 { q, scale } => q.len() >= n * k && scale.len() >= n,
            WeightRef::Int8 { q, scale } => q.len() >= n * k && scale.len() >= n,
            WeightRef::Mxfp4 { packed, scales } => {
                k.is_multiple_of(32) && packed.len() >= n * k / 2 && scales.len() >= n * k / 32
            }
        };
        assert!(ok, "weight buffer too small for [{n}][{k}]");
    }
}

/// `y[m*ldy + j] = x[m] . W[j]` for `j` in `n0..n1`, `m` in `0..rows`.
///
/// `n_total` is the matrix's full row count (for bounds checking); `x` is `[rows][k]`.
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    x: &[f32],
    rows: usize,
    k: usize,
    w: WeightRef<'_>,
    n_total: usize,
    n0: usize,
    n1: usize,
    y: &mut [f32],
    ldy: usize,
) {
    assert!(n0 <= n1 && n1 <= n_total && n1 <= ldy);
    assert!(x.len() >= rows * k, "x too small");
    assert!(rows == 0 || y.len() >= (rows - 1) * ldy + n1, "y too small");
    w.check(n_total, k);
    if rows == 0 || n0 == n1 {
        return;
    }
    let (xp, m, kk, n0i, n1i, yp, ld) =
        (x.as_ptr(), rows as i64, k as i64, n0 as i64, n1 as i64, y.as_mut_ptr(), ldy as i64);
    unsafe {
        match w {
            WeightRef::F32(w) => ffi::sk_cpu_gemm_f32(xp, m, kk, w.as_ptr(), n0i, n1i, yp, ld),
            WeightRef::Bf16(w) => ffi::sk_cpu_gemm_bf16(xp, m, kk, w.as_ptr(), n0i, n1i, yp, ld),
            WeightRef::Fp8 { q, scale } => {
                ffi::sk_cpu_gemm_fp8(xp, m, kk, q.as_ptr(), scale.as_ptr(), n0i, n1i, yp, ld)
            }
            WeightRef::Int8 { q, scale } => {
                ffi::sk_cpu_gemm_i8(xp, m, kk, q.as_ptr(), scale.as_ptr(), n0i, n1i, yp, ld)
            }
            WeightRef::Mxfp4 { packed, scales } => {
                ffi::sk_cpu_gemm_mxfp4(xp, m, kk, packed.as_ptr(), scales.as_ptr(), n0i, n1i, yp, ld)
            }
        }
    }
}

/// Quantise rows `n0..n1` of a bf16 `[n][k]` matrix to E4M3 with per-row absmax scales.
pub fn quant_fp8_rows(w: &[u16], k: usize, n0: usize, n1: usize, q: &mut [u8], scale: &mut [f32]) {
    assert!(w.len() >= n1 * k && q.len() >= n1 * k && scale.len() >= n1 && n0 <= n1);
    unsafe {
        ffi::sk_cpu_quant_fp8_rows(w.as_ptr(), k as i64, n0 as i64, n1 as i64, q.as_mut_ptr(), scale.as_mut_ptr())
    }
}

/// Quantise rows `n0..n1` of a bf16 `[n][k]` matrix to symmetric int8, per-row scales.
pub fn quant_i8_rows(w: &[u16], k: usize, n0: usize, n1: usize, q: &mut [i8], scale: &mut [f32]) {
    assert!(w.len() >= n1 * k && q.len() >= n1 * k && scale.len() >= n1 && n0 <= n1);
    unsafe { ffi::sk_cpu_quant_i8_rows(w.as_ptr(), k as i64, n0 as i64, n1 as i64, q.as_mut_ptr(), scale.as_mut_ptr()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16(v: f32) -> u16 {
        (v.to_bits() >> 16) as u16
    }

    #[test]
    fn gemm_formats_agree() {
        let (n, k) = (5usize, 96usize);
        let wf: Vec<f32> = (0..n * k).map(|i| bf16_round(((i * 37 % 101) as f32 - 50.0) / 64.0)).collect();
        let wb: Vec<u16> = wf.iter().map(|&v| bf16(v)).collect();
        let x: Vec<f32> = (0..2 * k).map(|i| ((i * 13 % 29) as f32 - 14.0) / 8.0).collect();
        let mut y1 = vec![0.0; 2 * n];
        let mut y2 = vec![0.0; 2 * n];
        gemm(&x, 2, k, WeightRef::F32(&wf), n, 0, n, &mut y1, n);
        gemm(&x, 2, k, WeightRef::Bf16(&wb), n, 0, n, &mut y2, n);
        for (a, b) in y1.iter().zip(&y2) {
            assert!((a - b).abs() <= 1e-4 * a.abs().max(1.0), "{a} vs {b}");
        }
    }

    fn bf16_round(v: f32) -> f32 {
        f32::from_bits(v.to_bits() & 0xFFFF_0000)
    }

    #[test]
    fn mxfp4_nibble_order_and_scale() {
        // One row, one group: element i has code (i % 16); scale byte 128 => x2.
        let k = 32;
        let mut packed = vec![0u8; 16];
        for (i, b) in packed.iter_mut().enumerate() {
            let even = (2 * i % 16) as u8;
            let odd = ((2 * i + 1) % 16) as u8;
            *b = even | (odd << 4);
        }
        let scales = vec![128u8];
        let e2m1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
        for probe in 0..k {
            let mut x = vec![0.0f32; k];
            x[probe] = 1.0;
            let mut y = [0.0f32];
            gemm(&x, 1, k, WeightRef::Mxfp4 { packed: &packed, scales: &scales }, 1, 0, 1, &mut y, 1);
            assert_eq!(y[0], 2.0 * e2m1[probe % 16], "element {probe}");
        }
    }

    #[test]
    fn mxfp4_batch_invariance_across_paths() {
        // M=5 runs rows 0-3 through the shared-decode dot4 path and row 4 through the
        // fused single-row path; every row must equal its own M=1 result bit for bit.
        let (n, k) = (9usize, 96usize);
        let packed: Vec<u8> = (0..n * k / 2).map(|i| ((i * 2654435761usize) >> 9) as u8).collect();
        let scales: Vec<u8> = (0..n * k / 32).map(|i| 120 + (i % 9) as u8).collect();
        let w = WeightRef::Mxfp4 { packed: &packed, scales: &scales };
        let x: Vec<f32> = (0..5 * k).map(|i| ((i * 37 % 101) as f32 - 50.0) / 17.0).collect();
        let mut all = vec![0.0; 5 * n];
        gemm(&x, 5, k, w, n, 0, n, &mut all, n);
        for m in 0..5 {
            let mut one = vec![0.0; n];
            gemm(&x[m * k..(m + 1) * k], 1, k, w, n, 0, n, &mut one, n);
            assert_eq!(&all[m * n..(m + 1) * n], &one[..], "row {m} differs between batch paths");
        }
        for fmt in 0..3 {
            let qi: Vec<i8> = (0..n * k).map(|i| ((i * 31 % 251) as i32 - 125) as i8).collect();
            let qf: Vec<u8> = (0..n * k).map(|i| ((i * 13 % 255) as u8) & 0xF7).collect();
            let wb: Vec<u16> = (0..n * k).map(|i| bf16(((i * 7 % 61) as f32 - 30.0) / 9.0)).collect();
            let sc: Vec<f32> = (0..n).map(|i| 0.01 + i as f32 * 0.003).collect();
            let wr = match fmt {
                0 => WeightRef::Int8 { q: &qi, scale: &sc },
                1 => WeightRef::Fp8 { q: &qf, scale: &sc },
                _ => WeightRef::Bf16(&wb),
            };
            gemm(&x, 5, k, wr, n, 0, n, &mut all, n);
            for m in 0..5 {
                let mut one = vec![0.0; n];
                gemm(&x[m * k..(m + 1) * k], 1, k, wr, n, 0, n, &mut one, n);
                assert_eq!(&all[m * n..(m + 1) * n], &one[..], "format {fmt} row {m}");
            }
        }
    }

    #[test]
    fn fp8_roundtrip_is_close() {
        let (n, k) = (3usize, 64usize);
        let wf: Vec<f32> = (0..n * k).map(|i| ((i * 7919 % 1000) as f32 - 500.0) / 300.0).collect();
        let wb: Vec<u16> = wf.iter().map(|&v| bf16(v)).collect();
        let mut q = vec![0u8; n * k];
        let mut s = vec![0f32; n];
        quant_fp8_rows(&wb, k, 0, n, &mut q, &mut s);
        let t = fp8_e4m3_table();
        assert_eq!(t[0x7E], 448.0);
        assert_eq!(t[0x38], 1.0);
        for r in 0..n {
            for c in 0..k {
                let orig = f32::from_bits((wb[r * k + c] as u32) << 16);
                let back = t[q[r * k + c] as usize] * s[r];
                // 3 mantissa bits: relative error <= 2^-4 for normals, plus subnormal floor
                assert!((orig - back).abs() <= orig.abs() / 16.0 + s[r] * 2f32.powi(-9), "{orig} vs {back}");
            }
        }
    }

    #[test]
    fn batch_invariance() {
        // Row results must not depend on how many rows share the call or on the n split.
        let (n, k) = (7usize, 160usize);
        let wb: Vec<u16> = (0..n * k).map(|i| bf16(((i * 31 % 97) as f32 - 48.0) / 50.0)).collect();
        let x: Vec<f32> = (0..4 * k).map(|i| ((i * 17 % 53) as f32 - 26.0) / 9.0).collect();
        let mut all = vec![0.0; 4 * n];
        gemm(&x, 4, k, WeightRef::Bf16(&wb), n, 0, n, &mut all, n);
        for m in 0..4 {
            let mut one = vec![0.0; n];
            gemm(&x[m * k..(m + 1) * k], 1, k, WeightRef::Bf16(&wb), n, 0, 3, &mut one, n);
            gemm(&x[m * k..(m + 1) * k], 1, k, WeightRef::Bf16(&wb), n, 3, n, &mut one, n);
            assert_eq!(&all[m * n..(m + 1) * n], &one[..], "row {m}");
        }
    }

    #[test]
    fn bf16_rows_dotted_as_read_match_decoded() {
        // Under four rows of x the bf16 GEMM dots each weight row as it reads it (four rows of W at
        // a time, then single rows); from four rows up it decodes a row once for every x row. Every
        // x row must come out the same either way: M = 1, 2, 3 against the rows of an M = 4 call,
        // with widths that leave a tail to the SIMD steps (16 and 32 values) and W rows that leave
        // some over the four-row blocks.
        for (n, k) in [(7usize, 160usize), (11, 2304 + 37), (5, 37), (9, 1024)] {
            let wb: Vec<u16> = (0..n * k).map(|i| bf16(((i * 31 % 97) as f32 - 48.0) / 50.0)).collect();
            let x: Vec<f32> = (0..4 * k).map(|i| ((i * 17 % 53) as f32 - 26.0) / 9.0).collect();
            let mut four = vec![0.0; 4 * n];
            gemm(&x, 4, k, WeightRef::Bf16(&wb), n, 0, n, &mut four, n);
            for m in 1..4 {
                let mut few = vec![0.0; m * n];
                gemm(&x[..m * k], m, k, WeightRef::Bf16(&wb), n, 0, n, &mut few, n);
                assert_eq!(&four[..m * n], &few[..], "M = {m}, {n} x {k}");
            }
        }
    }
}
