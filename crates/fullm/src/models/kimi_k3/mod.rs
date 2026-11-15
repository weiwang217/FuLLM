// SPDX-License-Identifier: Apache-2.0
//! Kimi K3: 93 layers (69 Kimi Delta Attention + 24 gated MLA), Stable LatentMoE with 896
//! MXFP4 routed experts (top-16) and 2 shared experts, Block Attention Residuals.
//!
//! `config` reads the architecture, `weights` binds the checkpoint, `experts` serves the
//! routed experts from whatever tier holds them, `cpu` is the reference/CPU backend, and
//! `profile` turns a config into the byte counts the deployment planner works from.

pub mod config;
pub mod cpu;
pub mod exec;
pub mod experts;
pub mod profile;
pub mod weights;

pub use config::K3Config;

/// Every tensor name in the released checkpoint carries this prefix (lm_head excepted).
pub const PREFIX: &str = "language_model.model.";
pub const LM_HEAD: &str = "language_model.lm_head.weight";
