// SPDX-License-Identifier: Apache-2.0
// With the `torch` feature, binaries find libtorch where fullm-kernels linked it.
fn main() {
    if let Ok(lib) = std::env::var("DEP_FULLM_KERNELS_TORCH_LIB") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
    }
}
