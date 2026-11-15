// SPDX-License-Identifier: Apache-2.0
//! fullm: a single-GPU LLM inference engine (OpenAI-compatible server, scheduler, prefix cache,
//! speculative decoding, LoRA, structured output) over the C++/CUDA executor of `fullm-kernels`.
//!
//! A model's trunk (attention, shared experts, projections) lives in GPU memory; its routed
//! experts live there too when they fit, else as many as fit, placed by routing frequency, with
//! the rest computed by the CPU in host memory.

pub mod bench;
pub mod chat_template;
pub mod hw;
pub mod models;
pub mod ops;
pub mod planner;
pub mod runtime;
pub mod server;
pub mod st;
pub mod tensor;
pub mod tokenizer;
