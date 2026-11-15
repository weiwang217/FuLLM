// SPDX-License-Identifier: Apache-2.0
//! Hardware descriptions the planner works from.
//!
//! Every figure is an EFFECTIVE rate, what a well-written kernel or I/O path sustains,
//! not a datasheet peak. Presets carry conservative published or commonly measured values;
//! `probe_host` measures this machine's DRAM bandwidth and memory; anything can be
//! overridden from the CLI. The planner's output is only as good as these numbers, which
//! is why it prints them next to every plan.

use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct Gpu {
    pub name: String,
    pub hbm_gb: f64,
    /// Sustained HBM read bandwidth for streaming GEMV.
    pub hbm_gbps: f64,
    /// Sustained dense bf16 tensor-core throughput (dequant-in-mainloop GEMM).
    pub bf16_tflops: f64,
    /// Host -> device copy rate from pinned memory.
    pub pcie_gbps: f64,
    /// CUDA context, cuBLAS workspaces, allocator slack.
    pub reserved_gb: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Host {
    pub name: String,
    pub ram_gb: f64,
    /// Sustained multi-socket read bandwidth with NUMA-local placement.
    pub dram_gbps: f64,
    /// Sustained CPU throughput on MXFP4 expert GEMMs (fp32 FMA or AMX after decode).
    pub cpu_tflops: f64,
    /// Sequential read bandwidth of the device array holding the checkpoint.
    pub nvme_gbps: f64,
    /// Memory left for the OS, page tables and the process itself.
    pub os_reserve_gb: f64,
}

pub fn gpu_preset(name: &str) -> Option<Gpu> {
    let g = |name: &str, hbm, bw, tf, pcie| Gpu {
        name: name.into(),
        hbm_gb: hbm,
        hbm_gbps: bw,
        bf16_tflops: tf,
        pcie_gbps: pcie,
        reserved_gb: 1.5,
    };
    Some(match name {
        // datasheet 3.35 TB/s, 989 TF dense bf16; ~80% / ~60% sustained
        "h100-sxm" | "h100" => g("H100 SXM 80GB", 80.0, 2700.0, 600.0, 50.0),
        "h100-pcie" => g("H100 PCIe 80GB", 80.0, 1650.0, 450.0, 50.0),
        "h100-nvl" => g("H100 NVL 94GB", 94.0, 3100.0, 600.0, 50.0),
        "h200" => g("H200 141GB", 141.0, 3800.0, 600.0, 50.0),
        "a100-80" => g("A100 SXM 80GB", 80.0, 1600.0, 250.0, 22.0),
        _ => return None,
    })
}

pub fn host_preset(name: &str) -> Option<Host> {
    let h = |name: &str, ram, dram, tf, nvme| Host {
        name: name.into(),
        ram_gb: ram,
        dram_gbps: dram,
        cpu_tflops: tf,
        nvme_gbps: nvme,
        os_reserve_gb: 48.0,
    };
    Some(match name {
        // 2x EPYC 9654, 24 ch DDR5-4800 (921 GB/s peak), AVX-512 (double-pumped) fp32 FMA
        "genoa-2t" => h("2x EPYC 9654, 2.3 TB DDR5", 2304.0, 600.0, 6.0, 25.0),
        // 2x Xeon 8480+, 16 ch DDR5-4800 (614 GB/s peak), AMX bf16
        "spr-2t" => h("2x Xeon 8480+, 2 TB DDR5", 2048.0, 420.0, 30.0, 25.0),
        // 1x Threadripper PRO / Xeon W, 8 ch DDR5, 4x Gen4 NVMe RAID0
        "ws-512g" => h("workstation, 512 GB DDR5", 512.0, 220.0, 3.0, 25.0),
        // a typical 1x H100 cloud VM: 26 vCPU, ~240 GB, one local/network NVMe volume
        "cloud-1gpu" => h("cloud 1x H100 VM, 240 GB", 240.0, 90.0, 1.0, 6.0),
        _ => return None,
    })
}

pub const GPU_PRESETS: &[&str] = &["h100-sxm", "h100-pcie", "h100-nvl", "h200", "a100-80"];
pub const HOST_PRESETS: &[&str] = &["genoa-2t", "spr-2t", "ws-512g", "cloud-1gpu"];

/// Measure this machine: total memory, and a multi-threaded read bandwidth (a sum over a
/// buffer much larger than cache). CPU throughput and NVMe bandwidth are not measured
/// here; pass them on the command line.
pub fn probe_host() -> Host {
    let ram_gb = total_ram_bytes() as f64 / 1e9;
    let dram_gbps = measure_read_gbps((ram_gb * 0.05).clamp(0.5, 4.0));
    Host {
        name: format!("this machine ({} threads, {})", rayon::current_num_threads(), fullm_kernels::cpu_isa()),
        ram_gb,
        dram_gbps,
        cpu_tflops: 0.05 * rayon::current_num_threads() as f64,
        nvme_gbps: 3.0,
        os_reserve_gb: (ram_gb * 0.1).clamp(4.0, 64.0),
    }
}

fn total_ram_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    unsafe {
        let mut v: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let name = c"hw.memsize";
        if libc::sysctlbyname(name.as_ptr(), &mut v as *mut u64 as *mut _, &mut len, std::ptr::null_mut(), 0) == 0 {
            return v;
        }
        0
    }
    #[cfg(not(target_os = "macos"))]
    unsafe {
        let pages = libc::sysconf(libc::_SC_PHYS_PAGES);
        let size = libc::sysconf(libc::_SC_PAGE_SIZE);
        if pages > 0 && size > 0 {
            (pages as u64) * (size as u64)
        } else {
            0
        }
    }
}

fn measure_read_gbps(gb: f64) -> f64 {
    use rayon::prelude::*;
    let n = (gb * 1e9 / 8.0) as usize;
    let buf: Vec<u64> = (0..n as u64).into_par_iter().map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();
    let mut best = 0.0f64;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        let s: u64 = buf.par_chunks(1 << 16).map(|c| c.iter().fold(0u64, |a, &x| a ^ x)).reduce(|| 0, |a, b| a ^ b);
        let dt = t.elapsed().as_secs_f64();
        std::hint::black_box(s);
        best = best.max(n as f64 * 8.0 / dt / 1e9);
    }
    best
}
