# FlashAttention-3 (forward, Hopper), vendored

The forward-pass headers of FlashAttention-3 from
https://github.com/Dao-AILab/flash-attention, directory `hopper/`, commit
`2ed4a1f0fc0e36a6ca7b861bfbf86ebb82c4c592` (2026-11-04), under their BSD-3 license, unmodified but for the three changes below
(`LICENSE`; the `AUTHORS` file it refers to is upstream's). Only what `flash_fwd_launch_template.h` includes, plus `flash_prepare_scheduler.cu`
(the varlen scheduler's metadata kernel), the split-KV combine (`flash_fwd_combine_*.h`),
`heuristics.h` and `tile_size.h`; the PyTorch binding (`flash_api.cpp`), the backward pass and
the per-head-dim instantiation files are not here.

Three changes. First, `tile_size_fwd_sm90` takes `one_mma_wg` and `run_mha_fwd_` / `run_flash_fwd` a
trailing `OneMmaWG` (default false, declared in `flash.h`), which selects a 64 x 96 tile with one MMA warpgroup, the decode
tile vLLM's and SGLang's FlashAttention builds launch for decoding rows (their kernel names show it):
with GQA packing a decoding sequence has only the group's query rows per kv head (8 for Qwen3), so
the 128-row tile computes mostly empty rows and its CTAs are too few and too large to fill the GPU.
Second, `prepare_varlen_num_blocks_kernel` (flash_prepare_scheduler.cu) chooses the dynamic split
count among six candidates (no split, the original rule, the work spread over 1-4 rounds of the
persistent grid's CTAs) by rounds x blocks per tile + splits: the original rule could leave a last
round nearly empty (22 decoding sequences x 4 kv heads x 2 splits = 176 tiles on 132 SMs).
Third, `CollectiveMainloopFwdSm90` takes a trailing `HiLo` (and `run_flash_fwd` / `run_mha_fwd_`
one, `Flash_fwd_params::q_lo_ptr` its second query): the fp8 kernel with Q and P each as two e4m3,
the value and its rounding error. The second query is loaded with the first under the same
barrier and multiplied with the same key tile into the same accumulator (S = Q K^T + Q_lo K^T);
P's error is computed in registers after its conversion and multiplied with the same value tile
(O += P V + P_lo V): two GEMMs more per tile, Q and P seen to ~2^-8 instead of e4m3's 2^-4,
which is what the fp8 kernel loses against 16-bit Q and P (DESIGN 5.4).

`../fa3_attention.cu` is FuLLM's binding: it fills `Flash_fwd_params` the way `flash_api.cpp`
does for a varlen, paged, causal bf16 forward and instantiates
`run_mha_fwd_<90, bf16, 128, 128, Split, PagedKVNonTMA=true, softcap=false, PackGQA=true>` for
both `Split` values (and head_dim 64), the same with `cutlass::float_e4m3_t` at head_dim 128 (the fp8
KV cache: e4m3 queries, keys and values, bf16 output), with and without `OneMmaWG` and `HiLo`, plus the bf16
combine kernels: few queries over a long
context (decoding rows) get their keys split across CTAs, the count decided per sequence on the
device by the metadata kernel from `flash_api.cpp`'s `get_num_splits` upper bound. It is
compiled on its own by `build.rs` with `--use_fast_math`, `-DNDEBUG` and
`-DCUTLASS_ENABLE_GDC_FOR_SM90` (FlashAttention's own flags), with the
`FLASHATTENTION_DISABLE_*` defines for the variants FuLLM never runs (local, softcap, append-KV,
sm8x, clusters, fp16) to keep the compile to a few kernels. `../fa3_mla.cu`, a unit of its own
with the same flags (so the two compile side by side), instantiates
`run_mha_fwd_<90, bf16, 192, 128, Split=false, PagedKVNonTMA=false, softcap=false, PackGQA=false>`
for MLA prefill un-absorbed (keys and values expanded from the latent, contiguous, varlen through
`cu_seqlens_k`); it calls `fa3_attention.cu`'s scheduler metadata kernel.

Why this and not the CUTLASS example-88 FMHA: measured on an H100 at ~5900 tokens, FA3 runs
586 TFLOPS against the example's 436, reads the key/value cache through the page table (the
example needs a contiguous copy of each sequence's keys and values: a gather kernel per chunk),
takes any query count (the example wants a multiple of 8) and does every chunk of a step in one
launch. vLLM and SGLang call the same kernel through `flash_attn_varlen_func`.
