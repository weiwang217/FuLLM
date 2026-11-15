// SPDX-License-Identifier: Apache-2.0
//! OpenAI-compatible HTTP API over an `EngineHandle`.
//!
//!   POST /v1/completions        prompt text or token ids; `stream` for SSE. Control-token spellings in
//!                               text are text unless `serve --special-tokens-in-prompts`
//!   POST /v1/chat/completions   messages (content a string or a list of text parts) through the model's
//!                               chat template; Qwen3's thinking switch is `chat_template_kwargs:
//!                               {"enable_thinking": false}` (or the `thinking` extension), its reasoning
//!                               comes back apart as `reasoning_content`, as vLLM's qwen3 reasoning parser
//!                               returns it
//!                               `tools` / `tool_choice` (Qwen3's <tool_call> blocks come back as
//!                               `tool_calls`), `response_format` json_object / json_schema
//!   Both: vLLM's guided_json / guided_regex / guided_choice / guided_grammar (llguidance, grammar.rs).
//!   Sampling: temperature, top_p, top_k, min_p, seed, presence / frequency / repetition penalties,
//!   logit_bias, min_tokens, max_tokens, stop, stop_token_ids, ignore_eos (out-of-range values are
//!   400s, never ignored); log-probabilities of the model's logits (chat: logprobs + top_logprobs;
//!   completions: logprobs N; at most 20 alternatives). n = 1 only.
//!   Sampling fields a request leaves unset come from generation_config.json (`--generation-config`).
//!   `stop` strings end the text before the first one generated (stop.rs) and cancel the request.
//!   LoRA adapters: `model` names one (`serve --lora NAME=DIR`), /v1/models lists them;
//!   POST /v1/load_lora_adapter, /v1/unload_lora_adapter (vLLM's) with `serve --allow-runtime-lora`
//!   GET  /v1/models  /health  /metrics
//!   POST /flush_cache (alias /reset_prefix_cache)  drop the prefix cache's idle pages
//!
//! Handlers only translate: validation errors become 400s, everything else is the
//! engine's business. Dropping a streaming response drops its receiver, which the engine
//! notices on its next send and frees the sequence.

pub mod grammar;
mod openai;
mod stop;

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};

use crate::runtime::EngineHandle;
use crate::tokenizer::Tokenizer;

pub struct AppState {
    pub engine: EngineHandle,
    pub tok: Arc<dyn Tokenizer>,
    /// The id reported by /v1/models and echoed in responses.
    pub model_name: String,
    /// /v1/completions text prompts parse control-token spellings (`serve --special-tokens-in-prompts`).
    pub special_tokens_in_prompts: bool,
    /// What a request leaves unset samples with.
    pub sampling_defaults: SamplingDefaults,
    /// The structured-output grammar compiler, built on the first request that needs it.
    pub grammars: std::sync::OnceLock<std::result::Result<Arc<grammar::Grammars>, String>>,
    /// The LoRA adapter (its name) in each slot: a request's `model` picks one.
    pub loras: std::sync::Mutex<Vec<Option<String>>>,
    /// /v1/load_lora_adapter and /v1/unload_lora_adapter answer (`serve --allow-runtime-lora`);
    /// else 403.
    pub runtime_lora: bool,
    /// Every route but /health wants `Authorization: Bearer <key>` (`serve --api-key`); None: open.
    pub api_key: Option<String>,
}

/// The sampling values a request does not set: the model's generation_config.json (`serve
/// --generation-config auto`, vLLM's default: Qwen3 samples at temperature 0.6, top-p 0.95, top-k
/// 20), else OpenAI's (temperature 1, no top-p or top-k cut).
#[derive(Clone, Copy, Debug, Default)]
pub struct SamplingDefaults {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
}

impl SamplingDefaults {
    /// The fields generation_config.json has (none when it is absent or unreadable).
    pub fn from_generation_config(dir: &Path) -> Self {
        let Some(v) = std::fs::read_to_string(dir.join("generation_config.json"))
            .ok()
            .and_then(|r| serde_json::from_str::<serde_json::Value>(&r).ok())
        else {
            return Self::default();
        };
        let f = |k: &str| v.get(k).and_then(|x| x.as_f64());
        SamplingDefaults {
            temperature: f("temperature").map(|t| t as f32),
            top_p: f("top_p").map(|p| p as f32),
            top_k: f("top_k").filter(|&k| k > 0.0).map(|k| k as usize),
            min_p: f("min_p").map(|p| p as f32),
            repetition_penalty: f("repetition_penalty").map(|p| p as f32),
        }
    }
}

/// The largest request body: a prompt can hold the whole context window, so the limit follows it
/// (32 bytes a position covers the text, JSON escaping and a multi-turn history), never under 8 MB
/// (axum's default, 2 MB, refused prompts past ~0.5M tokens with a 413).
fn body_limit(max_context: usize) -> usize {
    (max_context.saturating_mul(32)).max(8 << 20)
}

pub fn router(state: Arc<AppState>) -> Router {
    let limit = body_limit(state.engine.info().max_context);
    Router::new()
        .route("/v1/completions", post(openai::completions))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/models", get(openai::models))
        .route("/health", get(|| async { Json(serde_json::json!({"status": "ok"})) }))
        .route("/metrics", get(openai::metrics))
        .route("/flush_cache", post(openai::flush_cache))
        .route("/v1/load_lora_adapter", post(openai::load_lora_adapter))
        .route("/v1/unload_lora_adapter", post(openai::unload_lora_adapter))
        .route("/reset_prefix_cache", post(openai::flush_cache))
        .layer(axum::middleware::from_fn_with_state(state.clone(), require_key))
        .layer(axum::extract::DefaultBodyLimit::max(limit))
        .with_state(state)
}

/// `serve --api-key`: a request without the key gets 401 (OpenAI's error body), /health excepted.
async fn require_key(
    axum::extract::State(st): axum::extract::State<Arc<AppState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(key) = &st.api_key {
        let given = req.headers().get(axum::http::header::AUTHORIZATION).and_then(|v| v.to_str().ok());
        let ok = given.and_then(|v| v.strip_prefix("Bearer ")).is_some_and(|k| same(k.as_bytes(), key.as_bytes()));
        if !ok && req.uri().path() != "/health" {
            let body = serde_json::json!({"error": {"message": "missing or wrong API key (Authorization: Bearer ...)",
                                                     "type": "invalid_request_error", "param": null, "code": null}});
            return (axum::http::StatusCode::UNAUTHORIZED, Json(body)).into_response();
        }
    }
    next.run(req).await
}

/// Equal byte strings, in time independent of where they differ.
fn same(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

pub async fn serve(state: Arc<AppState>, addr: SocketAddr) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await.with_context(|| format!("binding {addr}"))?;
    tracing::info!("serving {} on http://{}", state.model_name, listener.local_addr()?);
    axum::serve(listener, router(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .context("http server")
}

/// An OpenAI-shaped error body.
pub(crate) fn error(status: StatusCode, msg: impl std::fmt::Display) -> Response {
    let kind = if status.is_client_error() { "invalid_request_error" } else { "server_error" };
    (
        status,
        Json(serde_json::json!({"error": {"message": msg.to_string(), "type": kind, "param": null, "code": null}})),
    )
        .into_response()
}
