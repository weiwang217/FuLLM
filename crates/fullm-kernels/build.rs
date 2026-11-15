// SPDX-License-Identifier: Apache-2.0
// Builds the C++ CPU kernels and executor always, and the CUDA backend when the `cuda`
// feature is on (otherwise a stub that reports CUDA as unavailable).
//
// CUDA builds target sm_90a (H100) by default; override with FULLM_CUDA_ARCHS, e.g.
// "90a" or "80;90a" (a fat binary: the backend's generic kernels for each; the Hopper units,
// FlashAttention-3, CUTLASS's sm90 kernels and the int4 GEMM, are sm_90a only). CUDA_HOME (default
// /usr/local/cuda) locates nvcc and the libraries.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let csrc = PathBuf::from("csrc");
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-env-changed=FULLM_CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    let cuda = env::var_os("CARGO_FEATURE_CUDA").is_some();
    let torch = env::var_os("CARGO_FEATURE_TORCH").is_some();
    let cutlass = env::var_os("CARGO_FEATURE_CUTLASS").is_some();
    println!("cargo:rerun-if-env-changed=FULLM_TORCH_DIR");
    println!("cargo:rerun-if-env-changed=FULLM_CUTLASS_DIR");

    let mut cpp = cc::Build::new();
    cpp.cpp(true)
        .std("c++17")
        .opt_level(3)
        .include(csrc.join("include"))
        .include(csrc.join("exec"))
        .file(csrc.join("cpu/cpu_kernels.cpp"))
        .file(csrc.join("exec/cpu_backend.cpp"))
        .file(csrc.join("exec/backends.cpp"))
        .file(csrc.join("exec/backend.cpp"))
        .file(csrc.join("exec/core_backend.cpp"))
        .file(csrc.join("exec/conformance.cpp"))
        .file(csrc.join("exec/expert_pool.cpp"))
        .file(csrc.join("exec/k3_exec.cpp"))
        .file(csrc.join("exec/dense_exec.cpp"))
        .flag_if_supported("-fno-math-errno")
        .flag_if_supported("-pthread")
        .warnings(true);
    if !cuda {
        cpp.file(csrc.join("exec/cuda_stub.cpp"));
    }
    if torch {
        cpp.define("FULLM_TORCH", None);
    }
    cpp.compile("fullm_cpu");

    if torch {
        build_torch(&csrc);
    }

    if cuda {
        build_cuda(&csrc, cutlass);
    }
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=dylib=pthread");
    }
}

// The torch backend links the libtorch of a Python torch install: FULLM_TORCH_DIR is its
// package directory (python -c 'import torch, os; print(os.path.dirname(torch.__file__))').
fn build_torch(csrc: &Path) {
    let t = env::var("FULLM_TORCH_DIR").expect("the torch feature needs FULLM_TORCH_DIR (the torch package directory)");
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .opt_level(3)
        .include(csrc.join("include"))
        .include(csrc.join("exec"))
        .include(format!("{t}/include"))
        .include(format!("{t}/include/torch/csrc/api/include"))
        .define("FULLM_TORCH", None)
        .file(csrc.join("exec/torch_ops.cpp"))
        .warnings(false)
        .compile("fullm_torch");
    println!("cargo:rustc-link-search=native={t}/lib");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=c10");
    // for dependents' build scripts (DEP_FULLM_KERNELS_TORCH_LIB): an rpath to libtorch
    println!("cargo:torch_lib={t}/lib");
}

// The `cutlass` feature compiles csrc/exec/cuda/cutlass_ops.cu against a CUTLASS checkout
// (FULLM_CUTLASS_DIR, header-only): the sm90 grouped GEMM of the prefill experts.
//
// Every csrc/exec/cuda/*.cu is its own single-file library (no relocatable device code: no file
// calls another's device functions; with more than one file per build cc compiled --device-c,
// under which ptxas serialised CUTLASS's wgmma, C7509), built in parallel, and kept in an object
// cache keyed by a hash of the file, every local header it includes (recursively), the flags and
// nvcc's version: a change to one kernel file recompiles that file only (FlashAttention-3 and
// CUTLASS take minutes and rarely change). Cache: FULLM_OBJ_CACHE, default <target>/fullm-objcache.
fn build_cuda(csrc: &Path, cutlass: bool) {
    let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let archs = env::var("FULLM_CUDA_ARCHS").unwrap_or_else(|_| "90a".into());
    println!("cargo:rerun-if-env-changed=FULLM_OBJ_CACHE");
    let cutlass_dir = if cutlass {
        let d = env::var("FULLM_CUTLASS_DIR")
            .expect("the cutlass feature needs FULLM_CUTLASS_DIR (a CUTLASS checkout, header-only)");
        assert!(
            Path::new(&d).join("include/cutlass/cutlass.h").is_file(),
            "FULLM_CUTLASS_DIR={d} is not a CUTLASS checkout (no include/cutlass/cutlass.h)"
        );
        Some(d)
    } else {
        None
    };

    // the flags of every unit; FlashAttention-3 adds its own (see below). hopper: an sm_90a-only
    // unit (wgmma, TMA, CUTLASS's sm90 collectives), compiled for 90a whatever the list says
    let base = |fa3: bool, hopper: bool| -> cc::Build {
        let mut b = cc::Build::new();
        // debug(false): with debug info on, cc passes nvcc -G, which disables every device
        // optimisation. -lineinfo below keeps source lines for profilers without that cost.
        b.cuda(true)
            .debug(false)
            .cudart("shared")
            .cargo_metadata(false) // link directives below, in a fixed order
            .include(csrc.join("include"))
            .include(csrc.join("exec"))
            .include(format!("{cuda_home}/include"))
            .flag("-std=c++17")
            .flag("-O3")
            .flag("--expt-relaxed-constexpr")
            .flag("-lineinfo")
            .flag("-Xcompiler=-fPIC")
            // CUTLASS's device asserts are calls to __assertfail: ptxas then serialises every wgmma
            // of the kernel (C7509). fullm's own kernels use no assert.
            .define("NDEBUG", None);
        if fa3 {
            // FlashAttention-3 builds with --use_fast_math (exp2f -> ex2.approx in the softmax; the
            // rest of the backend keeps IEEE math) and the PDL define
            b.flag("--expt-extended-lambda").flag("--use_fast_math").define("CUTLASS_ENABLE_GDC_FOR_SM90", None);
        }
        if let Some(dir) = &cutlass_dir {
            b.include(format!("{dir}/include"))
                .include(format!("{dir}/tools/util/include"))
                .define("FULLM_CUTLASS", None);
            if !fa3 {
                b.include(format!("{dir}/examples/88_hopper_fmha")); // the Hopper FMHA lives in an example
            }
            b.flag("-Xcudafe=--diag_suppress=177"); // CUTLASS headers: unused functions
        }
        if hopper {
            b.flag("-gencode=arch=compute_90a,code=sm_90a");
        } else {
            for a in archs.split(';').filter(|s| !s.is_empty()) {
                b.flag(format!("-gencode=arch=compute_{a},code=sm_{a}"));
            }
        }
        b
    };

    let dir = csrc.join("exec/cuda");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("csrc/exec/cuda")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "cu"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "the cuda feature needs csrc/exec/cuda/*.cu");
    // link order: callers before callees (the backend calls the CUTLASS and FA3 libraries; the
    // selftest and the C ABI helpers call the backend)
    let rank = |p: &Path| match p.file_stem().and_then(|s| s.to_str()) {
        Some("cutlass_ops") | Some("int4_gemm") => 2,
        Some("fa3_mla") => 3, // calls fa3_attention's scheduler metadata kernel
        Some("fa3_attention") => 4,
        Some("cuda_backend") => 1,
        _ => 0,
    };
    files.sort_by_key(|p| rank(p));

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cache = env::var("FULLM_OBJ_CACHE").map(PathBuf::from).unwrap_or_else(|_| {
        // OUT_DIR = <target>/<profile>/build/<pkg>-<hash>/out
        out.ancestors().nth(4).map(|t| t.join("fullm-objcache")).unwrap_or_else(|| out.join("objcache"))
    });
    std::fs::create_dir_all(&cache).expect("object cache directory");
    let nvcc_version = std::process::Command::new(format!("{cuda_home}/bin/nvcc"))
        .arg("--version")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default();
    let include_dirs = [csrc.join("include"), csrc.join("exec"), dir.clone()];
    // the CUTLASS checkout's version and commit: a `git pull` there recompiles what includes it
    let cutlass_id = cutlass_dir.as_deref().map(cutlass_identity).unwrap_or_default();

    let libs: Vec<String> = std::thread::scope(|sc| {
        let handles: Vec<_> = files
            .iter()
            .map(|f| {
                let stem = f.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                let fa3 = matches!(stem, "fa3_attention" | "fa3_mla");
                let b = base(fa3, matches!(stem, "fa3_attention" | "fa3_mla" | "cutlass_ops" | "int4_gemm"));
                let (cache, out, nvcc_version, include_dirs, cutlass_id) =
                    (&cache, &out, &nvcc_version, &include_dirs, &cutlass_id);
                sc.spawn(move || {
                    let stem = f.file_stem().unwrap().to_string_lossy().into_owned();
                    let lib = format!("fullm_cu_{stem}");
                    let mut h = Fnv::new();
                    h.add(format!("{:?}", b.get_compiler().args()).as_bytes());
                    h.add(nvcc_version.as_bytes());
                    h.add(cutlass_id.as_bytes());
                    for src in local_includes(f, include_dirs) {
                        h.add(src.to_string_lossy().as_bytes());
                        h.add(&std::fs::read(&src).unwrap_or_default());
                    }
                    let cached = cache.join(format!("lib{lib}-{:016x}.a", h.0));
                    let built = out.join(format!("lib{lib}.a"));
                    if is_archive(&cached) && std::fs::copy(&cached, &built).is_ok() {
                        println!("cargo:warning=fullm: {stem}.cu from the object cache");
                    } else {
                        let _ = std::fs::remove_file(&cached); // a truncated entry (a machine stopped mid-write)
                        let t = std::time::Instant::now();
                        let mut b = b;
                        b.file(f);
                        b.compile(&lib);
                        println!("cargo:warning=fullm: compiled {stem}.cu in {:.0} s", t.elapsed().as_secs_f64());
                        // written whole before it is renamed in: synced, so a power cut leaves no empty entry
                        let tmp = cache.join(format!("lib{lib}-{:016x}.a.tmp", h.0));
                        let synced = std::fs::copy(&built, &tmp).is_ok()
                            && std::fs::File::open(&tmp).and_then(|f| f.sync_all()).is_ok();
                        if synced {
                            let _ = std::fs::rename(&tmp, &cached);
                        }
                        prune(cache, &lib, &cached);
                    }
                    lib
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("a CUDA unit failed to compile")).collect()
    });
    println!("cargo:rustc-link-search=native={}", out.display());
    for lib in &libs {
        println!("cargo:rustc-link-lib=static={lib}");
    }

    println!("cargo:rustc-link-search=native={cuda_home}/lib64");
    println!("cargo:rustc-link-search=native={cuda_home}/lib64/stubs");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

// 64-bit FNV-1a: stable across toolchains (the cache outlives a compiler update of Rust's hasher)
struct Fnv(u64);
impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn add(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
        self.0 = (self.0 ^ 0xff).wrapping_mul(0x0100_0000_01b3); // separate the pieces
    }
}

// A CUTLASS checkout's include/cutlass/version.h and its git HEAD (the commit, through a ref when
// HEAD names one): what the object cache keys on, beside the include path in the flags.
fn cutlass_identity(dir: &str) -> String {
    let d = Path::new(dir);
    let read = |p: PathBuf| std::fs::read_to_string(p).unwrap_or_default();
    let mut id = read(d.join("include/cutlass/version.h"));
    let head = read(d.join(".git/HEAD"));
    id.push_str(&head);
    if let Some(r) = head.trim().strip_prefix("ref: ") {
        id.push_str(&read(d.join(".git").join(r)));
    }
    id
}

// f and every file it includes with #include "..." that resolves locally (next to the including
// file, or in one of dirs), recursively, sorted; system and CUTLASS headers are covered by the
// flags (their paths) and nvcc's version
fn local_includes(f: &Path, dirs: &[PathBuf]) -> Vec<PathBuf> {
    let mut seen = std::collections::BTreeSet::new();
    let mut stack = vec![f.to_path_buf()];
    while let Some(p) = stack.pop() {
        let p = p.canonicalize().unwrap_or(p);
        if !seen.insert(p.clone()) {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&p) else { continue };
        for line in text.lines() {
            let l = line.trim_start();
            let Some(rest) = l.strip_prefix("#include") else { continue };
            let rest = rest.trim_start();
            let Some(name) = rest.strip_prefix('"').and_then(|r| r.split('"').next()) else { continue };
            let here = p.parent().map(|d| d.join(name));
            let found = here.into_iter().chain(dirs.iter().map(|d| d.join(name))).find(|c| c.is_file());
            if let Some(c) = found {
                stack.push(c);
            }
        }
    }
    seen.into_iter().collect()
}

// a static library: the ar magic and more than the header (a cut-off write leaves an empty file)
fn is_archive(p: &Path) -> bool {
    use std::io::Read;
    let mut magic = [0u8; 8];
    std::fs::metadata(p).is_ok_and(|m| m.len() > 8)
        && std::fs::File::open(p).and_then(|mut f| f.read_exact(&mut magic)).is_ok()
        && &magic == b"!<arch>\n"
}

// keep the three newest cached builds of lib (and the current one)
fn prune(cache: &Path, lib: &str, keep: &Path) {
    let prefix = format!("lib{lib}-");
    let mut v: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(cache)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with(&prefix) && n.ends_with(".a")))
        .filter_map(|p| std::fs::metadata(&p).and_then(|m| m.modified()).ok().map(|t| (t, p)))
        .collect();
    v.sort();
    let n = v.len();
    for (_, p) in v.into_iter().take(n.saturating_sub(3)) {
        if p != keep {
            let _ = std::fs::remove_file(p);
        }
    }
}
