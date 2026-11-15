# FuLLM

FuLLM is a single-GPU LLM inference engine written in Rust and C++/CUDA. It aims at two things:

- **Offer optimal performance when the model fits in one GPU with custom CUDA kernel backends.** The workloads are fine-tuned small and
  mid-size models, agent traffic (multi-turn tool calling, structured output, shared prefixes) and
  many LoRA adapters on one base.
- **Serve fast when the model does not fit.** Routed experts are split between GPU memory and host
  DRAM and computed where they live, rather than streamed over PCIe. The hottest experts, by routing
  frequency, are moved into GPU memory while serving.

The engine does inference only: you bring a trained or fine-tuned checkpoint. It targets NVIDIA H100
(sm90), with RTX 50 laptops (sm120) and CPUs (x86 AVX-512 / AVX2, Apple silicon) as the other
backends.

It serves an OpenAI-compatible API with these features:

- chat and completions, streamed;
- tool calls (Qwen3);
- `response_format` json_schema, enforced on the device;
- n-gram and EAGLE-3 speculative decoding;
- a radix prefix cache with a host-memory tier;
- multi-LoRA;
- sampling parameters, log-probabilities and stop strings.

## Models

| model | precisions | notes |
|---|---|---|
| Qwen3 dense (0.6B-8B), Llama-style dense | bf16, fp8 / int8 trunk, int4 MLP | CUDA, CPU; LoRA; EAGLE-3 heads |
| Qwen3-30B-A3B (MoE) | experts bf16 / fp8b / int8 / int4, GPTQ int4 checkpoints; KV bf16 / fp8 | for agents: fp8b experts + bf16 KV |
| Kimi Linear 48B-A3B (KDA + MLA hybrid) | experts bf16 / fp8b / int8 / int4 | in HBM at 8 bits or below; HBM / CPU split at bf16 |
| Kimi K3 | MXFP4 experts in DRAM | the first layers verified against a reference; full model needs a large host |

## Architecture

```
            OpenAI API (chat, completions, tools, json_schema, LoRA, metrics)
                                   |
   Rust control plane: crates/fullm
     scheduler (continuous batching, chunked prefill, preemption), radix prefix cache + DRAM tier,
     speculative decoding (n-gram, EAGLE-3), sampling, tokenizer + chat templates, planner
                                   |  one C call per step (fullm_exec.h): tokens, positions, pages
   C++/CUDA executor: crates/fullm-kernels
     model graphs: dense (Qwen3 / Llama), Qwen3-MoE, Kimi (KDA + MLA + MoE)
     CUDA: tensor-core GEMVs, CUTLASS grouped GEMMs (bf16, blockwise fp8, int4), FlashAttention-3
       (paged, fp8 KV, MLA prefill at 192/128), chunked KDA prefill on the tensor cores, fused
       KDA decode, MoE routing, LoRA, device sampling and grammar masks, CUDA graphs
     CPU: AVX-512 / AVX2 / NEON kernels, the routed-expert pool (bf16, MXFP4)
                                   |
   memory tiers: HBM (trunk, KV, hot experts) <-> host DRAM (cold experts, KV tier) <-> disk
```

The Rust side decides what runs in a step: which requests, which tokens, and which pages of the
paged KV cache. The executor runs the whole forward pass for that step. When some of a MoE model's
experts live in host memory, each layer splits its (token, expert) pairs between the resident
experts on the GPU and the CPU pool. The pool reads each expert once per layer and step, in place
in DRAM. Each pair has its own output row, reduced in selection order, so results are deterministic
and independent of the batch.

The split mode places experts by routing frequency. Routes are counted on the device every step.
Every 128 steps the most routed experts outside GPU memory take the slots of the least routed ones
inside. A placer thread fills the slots on a copy stream, so the step waits about 1 ms. Placement
freezes once it converges, so outputs repeat run after run; it thaws when the traffic shifts.

## Setup

### Build

```bash
# CPU only (macOS or Linux): every backend except CUDA
cargo build --release
cargo test --release                       # the test suite (CPU backends, tiny fixtures)

# H100 (CUDA 12.x / 13.x toolkit). CUTLASS is header-only; this is the commit FuLLM is tested
# with (version 4.8.1, an untagged commit on main); FlashAttention-3 and the grouped GEMMs depend on it.
git init -q ~/cutlass
git -C ~/cutlass fetch -q --depth 1 https://github.com/NVIDIA/cutlass 37b77875c76384f175a88381ba04e865eb2715c6
git -C ~/cutlass checkout -q FETCH_HEAD
export FULLM_CUTLASS_DIR=~/cutlass
cargo build --release --features cuda,cutlass -p fullm
target/release/fullm selftest              # every CUDA op against the CPU backend (~640 checks)
cargo test --release --features cuda,cutlass -p fullm

# RTX 50 laptop (sm_120)
FULLM_CUDA_ARCHS=120 cargo build --release --features cuda
```

### Serve

```bash
# Qwen3-30B-A3B for tool-calling agents: fp8 experts (blockwise), bf16 KV
fullm serve --model /data/qwen3-30b --backend cuda --gpu-expert-format fp8b --kv-dtype bf16 \
  --cache-tokens 196608 --step-tokens 8192 --max-seqs 64 --max-context 32768 --served-name qwen3 --port 8000

# Qwen3-30B-A3B for throughput: int4 experts, fp8 KV
fullm serve --model /data/qwen3-30b --backend cuda --gpu-expert-format int4 --kv-dtype fp8 \
  --cache-tokens 786432 --step-tokens 8192 --max-seqs 64 --served-name qwen3 --port 8000

# Kimi Linear 48B, all experts in HBM at 8 bits
fullm serve --model /data/kimi-linear --backend cuda --gpu-expert-format fp8b --gpu-experts auto \
  --cache-tokens 65536 --max-seqs 64 --served-name kimi-linear --port 8000

# Kimi Linear 48B at bf16 (94 GB of experts): as many experts as fit in HBM, placed by routing
# frequency, the rest on the CPU pool
fullm serve --model /data/kimi-linear --backend cuda --gpu-expert-format bf16 --gpu-experts auto \
  --cache-tokens 65536 --max-seqs 64 --served-name kimi-linear --port 8000

# many LoRA adapters on one base (a request picks one with "model": NAME)
fullm serve --model /data/qwen3-8b --backend cuda --lora sql=/data/lora-sql --lora support=/data/lora-support

# CPU backend (e.g. on a Mac)
fullm serve --model /data/qwen3-0.6b --backend cpu --max-seqs 8 --served-name qwen3 --port 8000
```

Useful flags:

| flag | what it sets |
|---|---|
| `--gpu-expert-format bf16\|fp8b\|int8\|fp8\|int4` | the resident experts' format |
| `--gpu-experts auto\|off\|GB` | how many experts live in GPU memory |
| `--kv-dtype bf16\|fp8\|fp8-token` | the KV cache's format |
| `--trunk bf16\|fp8\|int8` | the dense matrices' format |
| `--step-tokens` | the token budget of a step |
| `--spec-model DIR` | an EAGLE-3 head (n-gram drafts are on by default on GPUs) |
| `--lora-merge DIR` | a LoRA adapter merged into the weights at load |
| `--int4-checkpoint DIR` | GPTQ int4 experts |
| `--activations exact` | fp32 activations for small-batch GEMVs |
| `--no-cuda-graphs` | runs every step op by op |
| `--api-key KEY` | requires `Authorization: Bearer KEY` on every route but `/health` (or `FULLM_API_KEY`) |
| `--allow-runtime-lora` | serves `/v1/load_lora_adapter` and `/v1/unload_lora_adapter` (off by default: with it, any client can have the server read an adapter from a path) |

Environment switches:

| variable | what it does |
|---|---|
| `FULLM_QCACHE=DIR` | caches quantised experts on disk |
| `FULLM_EXPERT_PLACE=0` | keeps a static placement, for bit-identical outputs throughout |
| `FULLM_KDA_CHUNKED=0` / `FULLM_MLA_PREFILL=0` | uses the token-by-token prefill paths |

The other `FULLM_*` variables the code reads are internal tuning and debugging switches. Their
defaults are the tested configuration, and they may change or go away. Some of them slow a server
down (for example `FULLM_MOE_HIST` and `FULLM_ROUTE_STATS`, which synchronise or write files).

### Check and measure

```bash
python3 tools/smoke_test.py --url http://localhost:8000 --model qwen3            # 61 scored cases
python3 tools/bench_openai.py --model qwen3 --tokens 300 --concurrency 1,4,8,16,32
python3 tools/bench_openai.py --model qwen3 --tokens 200 --concurrency 1,4,16 --context 6000 --context-text README.md
python3 tools/agent_loop.py --url http://localhost:8000 --concurrency 16 --duration 120               # tool-calling agents
python3 tools/agent_loop.py --url http://localhost:8000 --suite complex --concurrency 8 --duration 300  # complex flows
python3 tools/bench_agent.py --url http://localhost:8000 --model qwen3 --scenario users --users 16,32
fullm bench --model /data/qwen3-30b --backend cuda --prompt 512 --decode 64 --batch 16 --json
fullm ppl --model /data/qwen3-30b --backend cuda --text FILE --max-tokens 4096
```

`tools/agent_loop.py` drives an agent through `/v1/chat/completions` against a deterministic mock
world. It grades every answer and counts protocol errors. `--suite complex` adds flows that leave
state to check:

- an order placed from a stock search;
- a meeting found across time zones;
- a summary written from three long reports;
- stock counts gathered through failing calls;
- a sub-agent asked for a result that is then emailed;
- a team survey.

## Benchmarks

One H100 80 GB (GCP a3-highgpu-1g, 26 vCPUs, 230 GB DRAM). vLLM
0.33 and SGLang 0.5.23 ran with their defaults, and llama.cpp at e0949d0, with the same client and
settings as FuLLM. All the H100 numbers come from one boot (2026-11-15).

### vs vLLM and SGLang: Qwen3-30B-A3B in one GPU

4-bit experts with fp8 KV: FuLLM int4 against vLLM and SGLang running Qwen's GPTQ-Int4. FuLLM's lead
over vLLM / SGLang:

| workload | FuLLM ahead of vLLM | FuLLM ahead of SGLang |
|---|---|---|
| short replies (300 tokens), c = 1 / 4 / 16 / 32 | +43% / +32% / +20% / +14% | +28% / +29% / +17% / +29% |
| ~6000-word prompts, c = 1 / 4 / 16 | +31% / +14% / +6% | +39% / +55% / +60% |
| agent sessions, 16 / 32 users (bench_agent) | +25% / +19% | +35% / +46% |
| agent sessions with n-gram drafts, 16 / 32 users | +47% / +36% | 2.6x / 3.2x |

bf16:

| workload | FuLLM ahead of vLLM | FuLLM ahead of SGLang |
|---|---|---|
| short replies, c = 1 / 4 | +16% / +14% | +3% / +3% |
| agent sessions with n-gram drafts, 16 / 32 users | +21% / +23% | +39% / +32% |

Qwen3-8B bf16, agent sessions with n-gram drafts at 16 / 32 users: +23% / +13% over vLLM and +46% / +48%
over SGLang.

### vs vLLM: tool-calling agents

Each run is `tools/agent_loop.py`: tasks run to their answers through `/v1/chat/completions` with
tools, thinking off. Neither engine made a protocol error.

| model | FuLLM | vLLM | FuLLM / vLLM |
|---|---|---|---|
| Qwen3-8B bf16, simple tasks, 16 agents | 2111 tasks / min | 1243 | 1.70x |
| Qwen3-30B-A3B bf16, simple tasks, 16 agents | 974 tasks / min | 726 | 1.34x |
| Qwen3-30B-A3B bf16, complex flows (6-13 calls, histories to 12k tokens), 8 agents | 150 tasks / min | 103 | 1.46x |

### vs vLLM: many LoRA adapters

Qwen3-8B bf16, with a rank-16 adapter in rank-32 slots:

| measurement | FuLLM | vLLM `--enable-lora` | FuLLM / vLLM |
|---|---|---|---|
| decode, c = 1 | 130.5 tok/s | 115.0 | +13% |
| prefill with a rank-128 adapter | 33442 tok/s | 25737 | +30% |

### vs vLLM: a model larger than the GPU

Kimi Linear 48B in bf16 has 94 GB of experts. Each run serves 300-token replies; aggregate tok/s:

| concurrency | FuLLM (`--gpu-experts auto`, placed by routing frequency, after a warm-up) | vLLM `--cpu-offload-gb 30` | FuLLM / vLLM |
|---|---|---|---|
| 1 | 151.2 | 11.7 | 13x |
| 4 | 401.9 | 20.5 | 20x |
| 8 | 501.3 | 24.9 | 20x |

### vs llama.cpp: Kimi Linear 48B

llama.cpp is the engine under Ollama; KTransformers has no Kimi Linear support.

All experts in GPU memory (FuLLM fp8b vs llama.cpp Q8_0):

| measurement | FuLLM | llama.cpp | FuLLM / llama.cpp |
|---|---|---|---|
| prefill, ~512-token prompt | 17821 tok/s | 4232 | 4.2x |
| decode, one stream | 246.8 tok/s | 198.8 | 1.24x |
| time to first token, one request | 29 ms | 68 ms | 2.3x sooner |
| server, c = 1 / 4 / 8 / 16 / 32 (aggregate tok/s) | 249 / 658 / 823 / 1177 / 1430 | 184 / 433 / 484 / 545 / 639 | 1.4 / 1.5 / 1.7 / 2.2 / 2.2x |

bf16 experts split between GPU memory and the CPU (FuLLM `auto`, about 80% in HBM, placed by routing
frequency after a warm-up; llama.cpp with 76% of experts on the GPU):

| measurement | FuLLM | llama.cpp | FuLLM / llama.cpp |
|---|---|---|---|
| decode, one stream | 184 tok/s | 97.8 | 1.9x |
| server, c = 1 / 4 / 8 / 16 / 32 (aggregate tok/s) | 151 / 402 / 501 / 696 / 899 | 71 / 127 / 151 / 190 / 32 | 2.1 / 3.2 / 3.3 / 3.7 / 28x |

### Long prompts: Kimi Linear prefill

The KDA layers prefill in chunks on the tensor cores. The MLA layers run un-absorbed on
FlashAttention-3, with long contexts in chunks of keys. With fp8b experts in HBM, a 2048-token prompt
prefills at 33269 tok/s and an 8192-token prompt at 41448 tok/s. A server request with a 3000-word
context gets its first token in 116 ms. With the default settings a 1M-token window works: prompts of
251K / 505K / 1.0M tokens get their first token in 13.8 / 43.4 / 148 s.

### vs vLLM on a CPU

Qwen3-0.6B bf16 on an M3 Pro (vLLM 0.11 CPU), one stream, n-gram drafts off: FuLLM 61.0 tok/s,
vLLM 47.5 (+28%).

## Repository

| path | contents |
|---|---|
| `crates/fullm` | the server, scheduler, prefix cache, speculative decoding, model loaders, CLI |
| `crates/fullm-kernels` | the executor and its kernels |
| `crates/fullm-kernels/csrc/exec/cuda` | CUDA kernels; FlashAttention-3 vendored under `fa3/` (BSD-3) |
| `tools/` | benchmarks, the smoke test, agent drivers, reference implementations |

## License

Apache License 2.0 ([LICENSE](LICENSE)). FlashAttention-3, vendored under
`crates/fullm-kernels/csrc/exec/cuda/fa3/`, keeps its BSD-3-Clause license
([fa3/LICENSE](crates/fullm-kernels/csrc/exec/cuda/fa3/LICENSE)). [NOTICE](NOTICE) lists the
third-party code and techniques FuLLM adapts or follows (tiktoken, CUTLASS, Marlin,
flash-linear-attention and others) and their licenses.
