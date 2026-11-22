// SPDX-License-Identifier: Apache-2.0
//! Inkling 2 (looped): thinkingmachines/Inkling-2-Mini's text decoder. 24 layers with sparse MoE
//! MLPs (128 experts, 8 per token), of which layers 6-17 run three times with shared weights: 48
//! layer visits a token, each with its own attention cache and convolution states.
//!
//! What sets it apart from the dense GQA models (tools/inkling2_ref.py states every step):
//!   - Block Attention Residuals: the stream is cut into blocks of 6 layers; every sub-layer input
//!     is an attention, by a learned query, over the finished blocks and the running sum of the
//!     current one.
//!   - No RoPE. Attention scores get a bias per (query, head, backward distance < 1024) mixed from
//!     16 trained profiles by the query token's hidden state, and a sink logit per (query, head).
//!     Queries and keys are clipped to a maximum L2 norm, values RMS-normalised.
//!   - A short causal convolution (4 taps) over keys, values and each sub-layer's output; in the
//!     layers that attend to the whole context (5, 11, 17, 23) the keys' and values' taps also
//!     depend on the token. The other layers attend to the last 1024 positions.
//!   - The loop: passes 2 and 3 over layers 6-17 restart from the stream as it entered layer 6 and
//!     take "messages" from the pass before (attention over its block states by four learned
//!     queries per layer), mixed into the query, key, value and MLP inputs by low-rank gates.
//!   - Experts in NVFP4 (20 layers) and MXFP8 (4 layers), a SwiGLU with softcaps.
//!
//! `config` reads the architecture; `cpu` is the reference backend (f32, one sequence at a time),
//! checked against the torch oracle of tools/make_tiny_inkling2.py, which is itself checked against
//! the transformers implementation (tools/check_inkling2_ref.py). The vision tower and the drafter
//! (mtp.safetensors) are not served.

pub mod config;
pub mod cpu;
pub mod exec;

pub use config::Ink2Config;

/// Every tensor of the text decoder carries this prefix.
pub const PREFIX: &str = "model.llm.";
