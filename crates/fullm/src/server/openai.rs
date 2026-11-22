// SPDX-License-Identifier: Apache-2.0
//! Request parsing and response shaping for the OpenAI endpoints.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::{self, Stream, StreamExt};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedReceiver;

use super::grammar::{bounded_whitespace, regex_literal, tool_calls_grammar, Grammars};
use super::stop::StopStrings;
use super::{error, AppState, SamplingDefaults};
use crate::runtime::sampler::TokenLogprobs;
use crate::runtime::{Cancel, Constraint, Event, FinishReason, Request, SamplingParams, Usage};
use crate::tokenizer::{py_json, Channel, ChatMessage, ChatOptions, ChatStream, Detokenizer, Tokenizer};
use llguidance::api::TopLevelGrammar;

/// Sampling fields shared by both endpoints. Unset ones take the server's defaults (the model's
/// generation_config.json, else OpenAI's: temperature 1).
#[derive(Deserialize)]
struct SamplingFields {
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    /// Extension: 0 disables.
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    n: Option<usize>,
    #[serde(default)]
    stream: bool,
    /// Extension: extra ids that end generation, beyond the tokenizer's own.
    #[serde(default)]
    stop_token_ids: Vec<u32>,
    /// Extension: do not stop on the tokenizer's end-of-turn ids.
    #[serde(default)]
    ignore_eos: bool,
    /// A string or a list of them: the text ends before the first one it would contain.
    #[serde(default)]
    stop: Option<Value>,
    /// Extension (vLLM's): keep the stop string at the end of the text.
    #[serde(default)]
    include_stop_str_in_output: bool,
    /// vLLM's guided decoding (at most one): a JSON schema (an object or its text), a regular
    /// expression, a list of strings to choose from, a Lark grammar.
    #[serde(default)]
    guided_json: Option<Value>,
    #[serde(default)]
    guided_regex: Option<String>,
    #[serde(default)]
    guided_choice: Option<Vec<String>>,
    #[serde(default)]
    guided_grammar: Option<String>,
    /// OpenAI's penalties ([-2, 2]) and logit_bias (token id -> [-100, 100]).
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    logit_bias: Option<std::collections::HashMap<String, f32>>,
    /// vLLM's extensions: repetition_penalty (> 0, 1 off), min_p ([0, 1]), min_tokens.
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    min_tokens: Option<usize>,
    /// Chat: true / false (with top_logprobs alternatives); completions: the number of
    /// alternatives (null: none). At most MAX_TOP_LOGPROBS.
    #[serde(default)]
    logprobs: Option<Value>,
    #[serde(default)]
    top_logprobs: Option<usize>,
}

impl SamplingFields {
    /// `chat`: which endpoint's `logprobs` (a flag with top_logprobs, or a count).
    fn params(&self, tok: &dyn Tokenizer, d: &SamplingDefaults, chat: bool) -> Result<SamplingParams, String> {
        if self.n.unwrap_or(1) != 1 {
            return Err("only n = 1 is supported".into());
        }
        let vocab = tok.vocab_size();
        if let Some(&t) = self.stop_token_ids.iter().find(|&&t| t as usize >= vocab) {
            return Err(format!("stop_token_ids: token {t} is outside the vocabulary ({vocab})"));
        }
        let mut stop = if self.ignore_eos { Vec::new() } else { tok.stop_ids() };
        stop.extend(&self.stop_token_ids);
        let range = |name: &str, v: Option<f32>, lo: f32, hi: f32| -> Result<f32, String> {
            match v {
                Some(x) if !(lo..=hi).contains(&x) => Err(format!("{name} must be in [{lo}, {hi}], got {x}")),
                _ => Ok(v.unwrap_or(0.0)),
            }
        };
        let presence_penalty = range("presence_penalty", self.presence_penalty, -2.0, 2.0)?;
        let frequency_penalty = range("frequency_penalty", self.frequency_penalty, -2.0, 2.0)?;
        let min_p = range("min_p", self.min_p.or(d.min_p), 0.0, 1.0)?;
        let repetition_penalty = self.repetition_penalty.or(d.repetition_penalty).unwrap_or(1.0);
        if !(repetition_penalty > 0.0 && repetition_penalty.is_finite()) {
            return Err(format!("repetition_penalty must be > 0, got {repetition_penalty}"));
        }
        let mut logit_bias = Vec::new();
        for (k, &b) in self.logit_bias.iter().flatten() {
            let id: u32 = k.parse().map_err(|_| format!("logit_bias: {k:?} is not a token id"))?;
            if id as usize >= vocab {
                return Err(format!("logit_bias: token {id} is outside the vocabulary ({vocab})"));
            }
            if !(-100.0..=100.0).contains(&b) {
                return Err(format!("logit_bias: {b} for token {id} is outside [-100, 100]"));
            }
            logit_bias.push((id, b));
        }
        logit_bias.sort_unstable_by_key(|&(t, _)| t);
        let max_tokens = self.max_completion_tokens.or(self.max_tokens).unwrap_or(16);
        let min_tokens = self.min_tokens.unwrap_or(0);
        if min_tokens > max_tokens {
            return Err(format!("min_tokens ({min_tokens}) is more than max_tokens ({max_tokens})"));
        }
        let max_lp = crate::runtime::sampler::MAX_TOP_LOGPROBS;
        let logprobs = if chat {
            let on = match &self.logprobs {
                None | Some(Value::Null) => false,
                Some(Value::Bool(b)) => *b,
                Some(_) => return Err("logprobs must be true or false".into()),
            };
            match (on, self.top_logprobs) {
                (false, Some(k)) if k > 0 => return Err("top_logprobs needs logprobs: true".into()),
                (true, k) if k.unwrap_or(0) > max_lp => return Err(format!("top_logprobs: at most {max_lp}")),
                (true, k) => Some(k.unwrap_or(0)),
                (false, _) => None,
            }
        } else {
            match &self.logprobs {
                None | Some(Value::Null) => None,
                Some(v) => match v.as_u64() {
                    Some(k) if k as usize <= max_lp => Some(k as usize),
                    _ => return Err(format!("logprobs must be a count from 0 to {max_lp}")),
                },
            }
        };
        Ok(SamplingParams {
            presence_penalty,
            frequency_penalty,
            repetition_penalty,
            logit_bias,
            min_p,
            min_tokens,
            logprobs,
            lora: None,
            max_tokens,
            temperature: self.temperature.or(d.temperature).unwrap_or(1.0),
            top_p: self.top_p.or(d.top_p).unwrap_or(1.0),
            top_k: self.top_k.or(d.top_k).unwrap_or(0),
            // Unseeded requests still get a definite seed, so a logged one can be replayed.
            seed: self.seed.unwrap_or_else(|| {
                SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
            }),
            stop_token_ids: stop,
        })
    }

    /// The guided-decoding fields as a grammar, when one is set.
    fn guided(&self) -> Result<Option<TopLevelGrammar>, String> {
        let mut gs = Vec::new();
        if let Some(j) = &self.guided_json {
            gs.push(json_schema_grammar(j)?);
        }
        if let Some(r) = &self.guided_regex {
            gs.push(TopLevelGrammar::from_regex(r));
        }
        if let Some(c) = &self.guided_choice {
            if c.is_empty() {
                return Err("guided_choice is empty".into());
            }
            gs.push(TopLevelGrammar::from_regex(
                &c.iter().map(|s| format!("({})", regex_literal(s))).collect::<Vec<_>>().join("|"),
            ));
        }
        if let Some(g) = &self.guided_grammar {
            gs.push(TopLevelGrammar::from_lark(g.clone()));
        }
        match gs.len() {
            0 => Ok(None),
            1 => Ok(gs.pop()),
            _ => Err("at most one of guided_json, guided_regex, guided_choice, guided_grammar".into()),
        }
    }

    fn stop_strings(&self) -> Result<StopStrings, String> {
        let bad = || "stop must be a string or a list of strings".to_string();
        let stops = match &self.stop {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::String(s)) => vec![s.clone()],
            Some(Value::Array(a)) => {
                a.iter().map(|v| v.as_str().map(String::from)).collect::<Option<Vec<_>>>().ok_or_else(bad)?
            }
            Some(_) => return Err(bad()),
        };
        if stops.iter().any(|s| s.is_empty()) {
            return Err("stop strings must not be empty".into());
        }
        if stops.len() > 64 {
            return Err(format!("at most 64 stop strings, got {}", stops.len()));
        }
        Ok(StopStrings::new(stops, self.include_stop_str_in_output))
    }
}

/// A JSON schema (an object, or its text) as a grammar.
fn json_schema_grammar(v: &Value) -> Result<TopLevelGrammar, String> {
    let schema = match v {
        Value::String(s) => serde_json::from_str(s).map_err(|e| format!("the JSON schema does not parse: {e}"))?,
        Value::Object(_) => v.clone(),
        Value::Bool(true) => json!({}),
        _ => return Err("a JSON schema must be an object".into()),
    };
    Ok(TopLevelGrammar::from_json_schema(bounded_whitespace(schema)))
}

/// OpenAI's `response_format`: `{"type": "text"}`, `{"type": "json_object"}` (any object) or
/// `{"type": "json_schema", "json_schema": {"name", "schema", "strict"}}`.
fn response_format_grammar(v: Option<&Value>) -> Result<Option<TopLevelGrammar>, String> {
    let Some(v) = v.filter(|v| !v.is_null()) else { return Ok(None) };
    match v.get("type").and_then(Value::as_str) {
        Some("text") => Ok(None),
        Some("json_object") => {
            Ok(Some(TopLevelGrammar::from_json_schema(bounded_whitespace(json!({"type": "object"})))))
        }
        Some("json_schema") => {
            let s =
                v.get("json_schema").and_then(|j| j.get("schema")).ok_or("response_format json_schema: no schema")?;
            json_schema_grammar(s).map(Some)
        }
        _ => Err("response_format: type must be text, json_object or json_schema".into()),
    }
}

/// The request's constraint for `g` (the grammar compiler is built on first use).
fn constraint(st: &AppState, g: TopLevelGrammar, after: Option<u32>) -> Result<Box<dyn Constraint>, String> {
    let gs = st
        .grammars
        .get_or_init(|| {
            Grammars::new(st.tok.clone(), st.engine.info().vocab).map(Arc::new).map_err(|e| format!("{e:#}"))
        })
        .as_ref()
        .map_err(|e| format!("structured output is unavailable: {e}"))?;
    gs.constraint(g, after).map_err(|e| format!("invalid grammar: {e:#}"))
}

#[derive(Deserialize)]
struct CompletionReq {
    prompt: Value,
    /// The served model's name, or a LoRA adapter's (other names: the base model).
    #[serde(default)]
    model: Option<String>,
    #[serde(flatten)]
    sampling: SamplingFields,
}

#[derive(Deserialize)]
struct ChatReq {
    messages: Vec<ChatMessage>,
    #[serde(default)]
    model: Option<String>,
    /// `low` | `medium` | `high` | `max`.
    #[serde(default)]
    reasoning_effort: Option<String>,
    /// Extension: turn the think channel off.
    #[serde(default)]
    thinking: Option<bool>,
    /// vLLM's and SGLang's way to pass template variables; `enable_thinking` is read (Qwen3).
    #[serde(default)]
    chat_template_kwargs: Option<serde_json::Map<String, Value>>,
    /// OpenAI function definitions: `[{"type": "function", "function": {"name", "description", "parameters"}}]`.
    #[serde(default)]
    tools: Option<Vec<Value>>,
    /// `"auto"` (the default with tools), `"none"`, `"required"`, or `{"type": "function", "function": {"name"}}`.
    #[serde(default)]
    tool_choice: Option<Value>,
    /// Accepted: the model may make several calls in a reply either way.
    #[serde(default)]
    #[allow(dead_code)]
    parallel_tool_calls: Option<bool>,
    /// `{"type": "text" | "json_object" | "json_schema", "json_schema": {"schema": ...}}`.
    #[serde(default)]
    response_format: Option<Value>,
    #[serde(flatten)]
    sampling: SamplingFields,
}

/// How a request may use its tools.
#[derive(Clone, Debug, PartialEq)]
enum ToolChoice {
    None,
    Auto,
    Required,
    Named(String),
}

impl ChatReq {
    /// The request's tools (checked) and how it may use them.
    fn tools(&self) -> Result<(Vec<Value>, ToolChoice), String> {
        let tools = self.tools.clone().unwrap_or_default();
        for (i, t) in tools.iter().enumerate() {
            let ok = t.get("type").and_then(Value::as_str) == Some("function")
                && t.get("function").and_then(|f| f.get("name")).and_then(Value::as_str).is_some_and(|n| !n.is_empty());
            if !ok {
                return Err(format!(
                    "tools[{i}]: expected {{\"type\": \"function\", \"function\": {{\"name\": ...}}}}"
                ));
            }
        }
        let choice = match &self.tool_choice {
            None | Some(Value::Null) => ToolChoice::Auto,
            Some(Value::String(s)) => match s.as_str() {
                "auto" => ToolChoice::Auto,
                "none" => ToolChoice::None,
                "required" => ToolChoice::Required,
                _ => return Err(format!("tool_choice {s:?}: expected auto, none, required or a function")),
            },
            Some(v) => match v.get("function").and_then(|f| f.get("name")).and_then(Value::as_str) {
                Some(n) => ToolChoice::Named(n.to_string()),
                None => {
                    return Err("tool_choice: expected {\"type\": \"function\", \"function\": {\"name\": ...}}".into())
                }
            },
        };
        if tools.is_empty() && matches!(choice, ToolChoice::Required | ToolChoice::Named(_)) {
            return Err("tool_choice requires tools".into());
        }
        if let ToolChoice::Named(n) = &choice {
            if !tools.iter().any(|t| t["function"]["name"].as_str() == Some(n.as_str())) {
                return Err(format!("tool_choice names {n:?}, which is not among the tools"));
            }
        }
        Ok((tools, choice))
    }
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn new_id(prefix: &str) -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    format!("{prefix}-{n:x}")
}

fn usage_json(u: &Usage) -> Value {
    json!({"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
           "total_tokens": u.prompt_tokens + u.completion_tokens,
           "prompt_tokens_details": {"cached_tokens": u.cached_tokens}})
}

fn parse<T: for<'de> Deserialize<'de>>(body: &Bytes) -> Result<T, String> {
    serde_json::from_slice(body).map_err(|e| format!("invalid request body: {e}"))
}

/// A completion prompt's ids: token ids as given, text as ordinary text unless the server was started
/// with `--special-tokens-in-prompts` (then `<|im_start|>` and the like are the control ids, as in vLLM).
fn prompt_ids(v: &Value, tok: &dyn Tokenizer, special: bool) -> Result<Vec<u32>, String> {
    let text = |s: &str| if special { tok.encode_with_special(s) } else { tok.encode(s) };
    match v {
        Value::String(s) => Ok(text(s)),
        Value::Array(a) if a.len() == 1 && a[0].is_string() => Ok(text(a[0].as_str().unwrap_or(""))),
        Value::Array(a) => a
            .iter()
            .map(|x| x.as_u64().filter(|&t| t <= u32::MAX as u64).map(|t| t as u32))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| "prompt must be a string or an array of token ids".into()),
        _ => Err("prompt must be a string or an array of token ids".into()),
    }
}

// ------------------------------------------------------------------------ /v1/completions

pub(super) async fn completions(State(st): State<Arc<AppState>>, body: Bytes) -> Response {
    let mut req: CompletionReq = match parse(&body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let tok = st.tok.clone();
    // tokenized off the async workers: a long prompt takes a blocking thread, not a server thread
    let (t, v, special) = (tok.clone(), std::mem::take(&mut req.prompt), st.special_tokens_in_prompts);
    let prompt = match tokio::task::spawn_blocking(move || prompt_ids(&v, t.as_ref(), special)).await {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => return error(StatusCode::BAD_REQUEST, e),
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, format!("tokenizer: {e}")),
    };
    let (params, stops) = match req
        .sampling
        .params(tok.as_ref(), &st.sampling_defaults, false)
        .map(|p| SamplingParams { lora: lora_slot(&st, req.model.as_deref()), ..p })
        .and_then(|p| Ok((p, req.sampling.stop_strings()?)))
    {
        Ok(p) => p,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let guide = match req.sampling.guided().and_then(|g| g.map(|g| constraint(&st, g, None)).transpose()) {
        Ok(c) => c,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let (rx, cancel) = match st.engine.submit_constrained(Request { prompt, params }, guide) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, format!("{e:#}")),
    };
    let (id, created, model) = (new_id("cmpl"), now(), st.model_name.clone());
    let mut out = Output::new(tok, None, stops, cancel);

    if !req.sampling.stream {
        let (pieces, reason, usage) = drain(rx, &mut out).await;
        if let FinishReason::Error(e) = &reason {
            return error(StatusCode::INTERNAL_SERVER_ERROR, e);
        }
        let text: String = pieces.into_iter().map(|(_, s)| s).collect();
        let (finish, usage) = out.report(&reason, &usage);
        let logprobs = out.completion_logprobs().unwrap_or(Value::Null);
        return Json(json!({
            "id": id, "object": "text_completion", "created": created, "model": model,
            "choices": [{"index": 0, "text": text, "token_ids": out.ids, "logprobs": logprobs,
                         "finish_reason": finish, "stop_reason": out.stop_reason()}],
            "usage": usage_json(&usage),
        }))
        .into_response();
    }

    let chunk = move |text: String, ids: Vec<u32>, finish: Option<&str>, usage: Option<&Usage>| {
        let mut v = json!({
            "id": id, "object": "text_completion", "created": created, "model": model,
            "choices": [{"index": 0, "text": text, "token_ids": ids, "logprobs": null, "finish_reason": finish}],
        });
        if let Some(u) = usage {
            v["usage"] = usage_json(u);
        }
        v
    };
    let events =
        stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|ev| (ev, rx)) }).map(move |ev| -> Vec<Value> {
            match ev {
                Event::Logprobs(lp) => {
                    out.logprobs(lp);
                    Vec::new()
                }
                Event::Token(_) if out.stopped() => Vec::new(), // generated before the cancel took
                Event::Token(t) => {
                    let text: String = out.token(t).into_iter().map(|(_, s)| s).collect();
                    let mut c = chunk(text, vec![t], None, None);
                    if let Some(lp) = out.completion_logprobs() {
                        c["choices"][0]["logprobs"] = lp;
                    }
                    vec![c]
                }
                Event::Finished { reason, usage } => {
                    let mut v = Vec::new();
                    if let FinishReason::Error(e) = &reason {
                        v.push(json!({"error": {"message": e, "type": "server_error"}}));
                    }
                    let text: String = out.finish().into_iter().map(|(_, s)| s).collect();
                    let (finish, usage) = out.report(&reason, &usage);
                    let mut last = chunk(text, vec![], Some(finish), Some(&usage));
                    last["choices"][0]["stop_reason"] = out.stop_reason();
                    if let Some(lp) = out.completion_logprobs() {
                        last["choices"][0]["logprobs"] = lp;
                    }
                    v.push(last);
                    v
                }
            }
        });
    sse(events)
}

// ------------------------------------------------------------------- /v1/chat/completions

pub(super) async fn chat_completions(State(st): State<Arc<AppState>>, body: Bytes) -> Response {
    let mut req: ChatReq = match parse(&body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let tok = st.tok.clone();
    let mut opts = ChatOptions::default();
    if let Some(k) = &req.chat_template_kwargs {
        opts.kwargs = k.clone();
    }
    if let Some(t) = req.chat_template_kwargs.as_ref().and_then(|k| k.get("enable_thinking")).and_then(|v| v.as_bool())
    {
        opts.thinking = t;
    }
    if let Some(t) = req.thinking {
        opts.thinking = t;
    }
    // Qwen3's /no_think, with thinking on: the prompt closes an empty think block, the reply the
    // model gives it (after an opened <think>\n it may reason on, small models more than large)
    if opts.thinking && tok.soft_thinking(&req.messages) == Some(false) {
        opts.thinking = false;
    }
    if let Some(e) = &req.reasoning_effort {
        opts.thinking_effort = e.clone();
    }
    if req.messages.is_empty() {
        return error(StatusCode::BAD_REQUEST, "messages is empty");
    }
    // tools: in the prompt and parsed out of the reply, unless tool_choice is "none"
    let (tools, choice) = match req.tools() {
        Ok(t) => t,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let tool_ids = if choice != ToolChoice::None && !tools.is_empty() {
        match tok.tool_call_ids() {
            Some(t) => {
                opts.tools = tools;
                Some(t)
            }
            None => return error(StatusCode::BAD_REQUEST, "this model's chat format has no tool calls"),
        }
    } else {
        None
    };
    let t_tok = std::time::Instant::now();
    let (t, messages, o) = (tok.clone(), std::mem::take(&mut req.messages), opts.clone());
    let prompt = match tokio::task::spawn_blocking(move || t.render_chat(&messages, &o)).await {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => return error(StatusCode::BAD_REQUEST, format!("{e:#}")),
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, format!("tokenizer: {e}")),
    };
    if step_trace() {
        tracing::info!("tokenized {} tokens in {:.3} ms", prompt.len(), t_tok.elapsed().as_secs_f64() * 1e3);
    }
    let (params, stops) = match req
        .sampling
        .params(tok.as_ref(), &st.sampling_defaults, true)
        .map(|p| SamplingParams { lora: lora_slot(&st, req.model.as_deref()), ..p })
        .and_then(|p| Ok((p, req.sampling.stop_strings()?)))
    {
        Ok(p) => p,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    // stop strings see the text as generated, newlines and all, whole or streamed; a stream keeps
    // the newlines that open each channel (trimming them would hold back its first visible delta, a
    // client's time to first token), a whole reply trims them
    let markers = tok.chat_markers(&opts);
    // the output's grammar: forced tool calls, response_format or a guided field (one of them),
    // from the token after the reasoning closes when the prompt leaves it open
    let mut grammars = Vec::new();
    if let (Some((open, close)), ToolChoice::Required | ToolChoice::Named(_)) = (tool_ids, &choice) {
        let only = if let ToolChoice::Named(n) = &choice { Some(n.as_str()) } else { None };
        grammars.push(tool_calls_grammar(&opts.tools, only, open, close));
    }
    match response_format_grammar(req.response_format.as_ref()).and_then(|r| Ok((r, req.sampling.guided()?))) {
        Ok((r, g)) => grammars.extend(r.into_iter().chain(g)),
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    }
    if grammars.len() > 1 {
        return error(
            StatusCode::BAD_REQUEST,
            "at most one of a forced tool_choice, response_format and the guided fields",
        );
    }
    if markers.channels.is_some() && matches!(choice, ToolChoice::Required | ToolChoice::Named(_)) {
        return error(StatusCode::BAD_REQUEST, "a forced tool_choice is not implemented for this model's chat format");
    }
    // (a reply in messages: from the token that opens the answer's text)
    let after = match markers.channels {
        Some(c) => Some(c.text),
        None => (markers.starts_in_reasoning && markers.think_end.len() == 1).then(|| markers.think_end[0]),
    };
    let guide = match grammars.pop().map(|g| constraint(&st, g, after)).transpose() {
        Ok(c) => c,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let (rx, cancel) = match st.engine.submit_constrained(Request { prompt, params }, guide) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, format!("{e:#}")),
    };
    let (id, created, model) = (new_id("chatcmpl"), now(), st.model_name.clone());
    // (a reply in messages reports the reasoning it has, thinking asked for or not)
    let by_message = markers.channels.is_some();
    let reasons = (opts.thinking && !markers.think_end.is_empty()) || by_message;
    let trim = markers.trim_newlines;
    let mut out = Output::new(tok, Some(ChatStream::new(markers)), stops, cancel);
    out.tools = tool_ids.map(|(open, close)| ToolCalls { open, close, ..Default::default() });

    if !req.sampling.stream {
        let (pieces, reason, usage) = drain(rx, &mut out).await;
        if let FinishReason::Error(e) = &reason {
            return error(StatusCode::INTERNAL_SERVER_ERROR, e);
        }
        let (mut reasoning, mut content) = (String::new(), String::new());
        for (ch, s) in pieces {
            if ch == Channel::Reasoning {
                reasoning += &s
            } else {
                content += &s
            }
        }
        if trim {
            reasoning = reasoning.trim_matches('\n').to_string();
            content = content.trim_start_matches('\n').to_string();
        }
        let calls = out.tools.as_ref().map(|t| t.calls.clone()).unwrap_or_default();
        // the whitespace around calls (the template's, or a forced call's leading run) is no reply
        let content = if content.trim().is_empty() && !calls.is_empty() { Value::Null } else { json!(content) };
        let mut message = json!({"role": "assistant", "content": content});
        if reasons && !(by_message && reasoning.is_empty()) {
            message["reasoning_content"] = json!(reasoning);
        }
        if !calls.is_empty() {
            message["tool_calls"] = json!(calls);
        }
        let (finish, usage) = out.report(&reason, &usage);
        let logprobs = out.chat_logprobs().unwrap_or(Value::Null);
        return Json(json!({
            "id": id, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": message, "logprobs": logprobs, "finish_reason": finish,
                         "stop_reason": out.stop_reason()}],
            "usage": usage_json(&usage),
        }))
        .into_response();
    }

    let chunk = move |delta: Value, finish: Option<&str>, usage: Option<&Usage>| {
        let mut v = json!({
            "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        });
        if let Some(u) = usage {
            v["usage"] = usage_json(u);
        }
        v
    };
    let delta = |ch: Channel, s: String| match ch {
        Channel::Reasoning => json!({"reasoning_content": s}),
        Channel::Content => json!({"content": s}),
    };
    let first = stream::once(futures::future::ready(vec![chunk.clone()(
        json!({"role": "assistant", "content": ""}),
        None,
        None,
    )]));
    let body =
        stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|ev| (ev, rx)) }).map(move |ev| -> Vec<Value> {
            match ev {
                Event::Logprobs(lp) => {
                    out.logprobs(lp);
                    Vec::new()
                }
                Event::Token(_) if out.stopped() => Vec::new(), // generated before the cancel took
                Event::Token(t) => {
                    let mut v: Vec<Value> =
                        out.token(t).into_iter().map(|(ch, s)| chunk(delta(ch, s), None, None)).collect();
                    v.extend(out.new_calls().into_iter().map(|c| chunk(json!({"tool_calls": [c]}), None, None)));
                    // a token's log-probabilities go with the first chunk sent after it (its text
                    // may be held back: a partial character, a possible stop string)
                    if let Some(first) = v.first_mut() {
                        if let Some(lp) = out.chat_logprobs() {
                            first["choices"][0]["logprobs"] = lp;
                        }
                    }
                    v
                }
                Event::Finished { reason, usage } => {
                    let mut v: Vec<Value> =
                        out.finish().into_iter().map(|(ch, s)| chunk(delta(ch, s), None, None)).collect();
                    if let FinishReason::Error(e) = &reason {
                        v.push(json!({"error": {"message": e, "type": "server_error"}}));
                    }
                    let (finish, usage) = out.report(&reason, &usage);
                    let mut last = chunk(json!({}), Some(finish), Some(&usage));
                    last["choices"][0]["stop_reason"] = out.stop_reason();
                    if let Some(lp) = out.chat_logprobs() {
                        last["choices"][0]["logprobs"] = lp;
                    }
                    v.push(last);
                    v
                }
            }
        });
    sse(first.chain(body))
}

// ------------------------------------------------------------------------ the rest

/// FULLM_STEP_TRACE (the engine's per-step trace) also logs the request-side times here.
fn step_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FULLM_STEP_TRACE").is_some())
}

pub(super) async fn models(State(st): State<Arc<AppState>>) -> Response {
    let info = st.engine.info();
    let mut data = vec![json!({
        "id": st.model_name, "object": "model", "created": now(), "owned_by": "fullm",
        "arch": info.arch, "max_model_len": info.max_context,
    })];
    // a slot still loading holds "\0loading NAME": not a model yet
    let names =
        st.loras.lock().map(|l| l.iter().flatten().filter(|n| !n.starts_with('\0')).cloned().collect::<Vec<_>>());
    for name in names.unwrap_or_default() {
        data.push(
            json!({"id": name, "object": "model", "created": now(), "owned_by": "fullm", "parent": st.model_name,
                         "max_model_len": info.max_context}),
        );
    }
    Json(json!({"object": "list", "data": data})).into_response()
}

/// The LoRA slot a request's `model` names (a loaded adapter), else None (the base model).
fn lora_slot(st: &AppState, model: Option<&str>) -> Option<usize> {
    let m = model.filter(|m| !m.starts_with('\0'))?;
    st.loras.lock().ok()?.iter().position(|n| n.as_deref() == Some(m))
}

#[derive(Deserialize)]
struct LoraReq {
    lora_name: String,
    #[serde(default)]
    lora_path: Option<String>,
}

/// vLLM's /v1/load_lora_adapter: {lora_name, lora_path} into a free slot.
pub(super) async fn load_lora_adapter(State(st): State<Arc<AppState>>, body: Bytes) -> Response {
    if !st.runtime_lora {
        return error(
            StatusCode::FORBIDDEN,
            "loading and unloading LoRA adapters at runtime is off (serve --allow-runtime-lora)",
        );
    }
    let req: LoraReq = match parse(&body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let Some(path) = req.lora_path.clone() else { return error(StatusCode::BAD_REQUEST, "lora_path is missing") };
    // a slot, held by the name while it loads
    let slot = {
        let Ok(mut l) = st.loras.lock() else { return error(StatusCode::INTERNAL_SERVER_ERROR, "registry poisoned") };
        if req.lora_name == st.model_name || l.iter().any(|n| n.as_deref() == Some(&req.lora_name)) {
            return error(StatusCode::BAD_REQUEST, format!("the model {:?} is already loaded", req.lora_name));
        }
        let Some(slot) = l.iter().position(Option::is_none) else {
            return error(
                StatusCode::BAD_REQUEST,
                format!("every LoRA slot ({}) is taken (serve --max-loras)", l.len()),
            );
        };
        l[slot] = Some(format!("\u{0}loading {}", req.lora_name));
        slot
    };
    let free = |st: &AppState| {
        if let Ok(mut l) = st.loras.lock() {
            l[slot] = None;
        }
    };
    let loaded =
        tokio::task::spawn_blocking(move || crate::models::lora::Adapter::load(std::path::Path::new(&path))).await;
    let adapter = match loaded {
        Ok(Ok(a)) => Arc::new(a),
        Ok(Err(e)) => {
            free(&st);
            return error(StatusCode::BAD_REQUEST, format!("{e:#}"));
        }
        Err(e) => {
            free(&st);
            return error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string());
        }
    };
    if let Err(e) = st.engine.set_lora(slot, Some(adapter)).await {
        free(&st);
        return error(StatusCode::BAD_REQUEST, format!("{e:#}"));
    }
    if let Ok(mut l) = st.loras.lock() {
        l[slot] = Some(req.lora_name.clone());
    }
    (StatusCode::OK, format!("Success: LoRA adapter '{}' added successfully.", req.lora_name)).into_response()
}

/// vLLM's /v1/unload_lora_adapter: {lora_name} (refused while requests run with it).
pub(super) async fn unload_lora_adapter(State(st): State<Arc<AppState>>, body: Bytes) -> Response {
    if !st.runtime_lora {
        return error(
            StatusCode::FORBIDDEN,
            "loading and unloading LoRA adapters at runtime is off (serve --allow-runtime-lora)",
        );
    }
    let req: LoraReq = match parse(&body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let Some(slot) = lora_slot(&st, Some(&req.lora_name)) else {
        return error(StatusCode::NOT_FOUND, format!("no LoRA adapter {:?}", req.lora_name));
    };
    if let Err(e) = st.engine.set_lora(slot, None).await {
        return error(StatusCode::BAD_REQUEST, format!("{e:#}"));
    }
    if let Ok(mut l) = st.loras.lock() {
        l[slot] = None;
    }
    (StatusCode::OK, format!("Success: LoRA adapter '{}' removed successfully.", req.lora_name)).into_response()
}

pub(super) async fn metrics(State(st): State<Arc<AppState>>) -> Response {
    Json(json!({"engine": st.engine.stats(), "model": st.engine.info()})).into_response()
}

pub(super) async fn flush_cache(State(st): State<Arc<AppState>>) -> Response {
    match st.engine.flush_cache().await {
        Ok(n) => Json(json!({"flushed_tokens": n})).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

/// Bytes of the ids that are text, i.e. everything but control tokens.
fn visible_bytes(tok: &dyn Tokenizer, ids: &[u32]) -> Vec<u8> {
    let vis: Vec<u32> = ids.iter().copied().filter(|&t| !tok.is_special(t)).collect();
    tok.decode_bytes(&vis)
}

/// A reply's function calls (Qwen3: `<tool_call>` JSON `</tool_call>` blocks in the answer channel,
/// recognised on their ids): taken out of the text, parsed as each block closes into OpenAI's
/// `tool_calls` entries. A block that is not a call's JSON goes back into the text, as vLLM's
/// parser does.
#[derive(Default)]
struct ToolCalls {
    open: u32,
    close: u32,
    /// The ids of the block being read.
    inside: Option<Vec<u32>>,
    calls: Vec<Value>,
    /// Calls already streamed.
    sent: usize,
    /// Answer text that is only whitespace so far, held back (`Output::hold_whitespace`).
    held: String,
}

/// OpenAI's entry for one call block, or None when it is not `{"name": ..., "arguments": ...}`.
fn tool_call(text: &str) -> Option<Value> {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let v: Value = serde_json::from_str(text.trim()).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    // (Inkling 2 writes `args`)
    let args = match v.get("arguments").or_else(|| v.get("args")) {
        Some(Value::String(s)) => s.clone(),
        Some(a) => py_json(a),
        None => "{}".to_string(),
    };
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Some(
        json!({"id": format!("call_{nanos:016x}{n:08x}"), "type": "function", "function": {"name": name, "arguments": args}}),
    )
}

/// A request's tokens on their way out: through the chat parser (a completion: plain text), then
/// the stop strings. When one completes, the request is cancelled and its tokens up to there are
/// what it reports.
struct Output {
    tok: Arc<dyn Tokenizer>,
    chat: Option<ChatStream>,
    detok: Detokenizer,
    stops: StopStrings,
    cancel: Cancel,
    /// The tokens taken: all of them, or those up to the one that completed a stop string.
    ids: Vec<u32>,
    /// A chat reply's function calls, when the request has tools.
    tools: Option<ToolCalls>,
    /// Log-probabilities (requests that ask): the next token's, then each taken token's, and
    /// how many a stream has sent; the text offset after the sent ones (completions).
    lp_next: Option<TokenLogprobs>,
    lps: Vec<(u32, TokenLogprobs)>,
    lp_sent: usize,
    lp_offset: usize,
}

impl Output {
    fn new(tok: Arc<dyn Tokenizer>, chat: Option<ChatStream>, stops: StopStrings, cancel: Cancel) -> Self {
        Output {
            tok,
            chat,
            detok: Detokenizer::default(),
            stops,
            cancel,
            ids: Vec::new(),
            tools: None,
            lp_next: None,
            lps: Vec::new(),
            lp_sent: 0,
            lp_offset: 0,
        }
    }

    /// The log-probabilities of the token the next `token` call takes.
    fn logprobs(&mut self, lp: TokenLogprobs) {
        self.lp_next = Some(lp);
    }

    /// A token's text and bytes, as logprobs report it.
    fn lp_token(&self, id: u32) -> (String, Vec<u8>) {
        let b = self.tok.decode_bytes(&[id]);
        (String::from_utf8_lossy(&b).into_owned(), b)
    }

    /// Chat's `logprobs.content` entries of the taken tokens not yet given out.
    fn chat_logprobs(&mut self) -> Option<Value> {
        if self.lp_sent == self.lps.len() {
            return None;
        }
        let entry = |o: &Self, id: u32, lp: f32| {
            let (t, b) = o.lp_token(id);
            json!({"token": t, "logprob": lp, "bytes": b})
        };
        let content: Vec<Value> = self.lps[self.lp_sent..]
            .iter()
            .map(|(id, lp)| {
                let mut e = entry(self, *id, lp.logprob);
                e["top_logprobs"] = Value::Array(lp.top.iter().map(|&(t, l)| entry(self, t, l)).collect());
                e
            })
            .collect();
        self.lp_sent = self.lps.len();
        Some(json!({"content": content}))
    }

    /// Completions' `logprobs` object for the taken tokens not yet given out: tokens,
    /// token_logprobs, top_logprobs (the alternatives and the drawn token, as vLLM gives them),
    /// text_offset (characters into the completion).
    fn completion_logprobs(&mut self) -> Option<Value> {
        if self.lp_sent == self.lps.len() {
            return None;
        }
        let (mut tokens, mut lps, mut tops, mut offs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in self.lp_sent..self.lps.len() {
            let (id, lp) = &self.lps[i];
            let (t, _) = self.lp_token(*id);
            let mut top = serde_json::Map::new();
            for &(a, l) in &lp.top {
                top.insert(self.lp_token(a).0, json!(l));
            }
            top.entry(t.clone()).or_insert(json!(lp.logprob));
            offs.push(self.lp_offset);
            self.lp_offset += t.chars().count();
            tokens.push(t);
            lps.push(lp.logprob);
            tops.push(Value::Object(top));
        }
        self.lp_sent = self.lps.len();
        Some(json!({"tokens": tokens, "token_logprobs": lps, "top_logprobs": tops, "text_offset": offs}))
    }

    /// Calls completed since the last time, as streaming deltas (`index` the call's place in the reply).
    fn new_calls(&mut self) -> Vec<Value> {
        let Some(t) = &mut self.tools else { return Vec::new() };
        let fresh: Vec<Value> = t.calls[t.sent..]
            .iter()
            .enumerate()
            .map(|(j, c)| {
                let mut c = c.clone();
                c["index"] = json!(t.sent + j);
                c
            })
            .collect();
        t.sent = t.calls.len();
        fresh
    }

    /// A tool-call id and what follows it, while a block is open: Some(text to emit) when the id
    /// was the block's (an unparsable block returns its text), None when it is ordinary output.
    fn tool_token(&mut self, t: u32) -> Option<Vec<(Channel, String)>> {
        let tc = self.tools.as_mut()?;
        if let Some(buf) = &mut tc.inside {
            if t != tc.close {
                buf.push(t);
                return Some(Vec::new());
            }
            let ids = std::mem::take(buf);
            tc.inside = None;
            let text = String::from_utf8_lossy(&self.tok.decode_bytes(&ids)).into_owned();
            return Some(match tool_call(&text) {
                Some(c) => {
                    tc.calls.push(c);
                    Vec::new()
                }
                None => vec![(Channel::Content, format!("<tool_call>{text}</tool_call>"))],
            });
        }
        if t == tc.open && self.chat.as_ref().is_some_and(|c| c.tool_ok()) {
            tc.inside = Some(Vec::new());
            return Some(Vec::new());
        }
        None
    }

    fn stopped(&self) -> bool {
        self.stops.matched.is_some()
    }

    fn token(&mut self, t: u32) -> Vec<(Channel, String)> {
        let lp = self.lp_next.take();
        if self.stopped() {
            return Vec::new();
        }
        self.ids.push(t);
        if let Some(lp) = lp {
            self.lps.push((t, lp));
        }
        if let Some(text) = self.tool_token(t) {
            return self.release(text);
        }
        let pieces = match &mut self.chat {
            Some(c) => c.push(self.tok.as_ref(), t),
            None => vec![(Channel::Content, self.detok.push(&visible_bytes(self.tok.as_ref(), &[t])))],
        };
        let out = self.release(pieces);
        if self.stopped() {
            self.cancel.cancel();
        }
        out
    }

    fn finish(&mut self) -> Vec<(Channel, String)> {
        if self.stopped() {
            return Vec::new();
        }
        let mut pieces = match &mut self.chat {
            Some(c) => c.finish(self.tok.as_ref()),
            None => vec![(Channel::Content, self.detok.finish())],
        };
        // a call block the reply did not close (out of tokens): its text, as generated
        if let Some(ids) = self.tools.as_mut().and_then(|t| t.inside.take()) {
            pieces.push((
                Channel::Content,
                format!("<tool_call>{}", String::from_utf8_lossy(&self.tok.decode_bytes(&ids))),
            ));
        }
        let mut out = self.release(pieces);
        let tail = self.stops.finish();
        out.extend(self.hold_whitespace(tail));
        // whitespace at the end of a reply without calls is its text's; around calls it is none
        if let Some(tc) = self.tools.as_mut().filter(|t| t.calls.is_empty() && !t.held.is_empty()) {
            out.push((Channel::Content, std::mem::take(&mut tc.held)));
        }
        out
    }

    fn release(&mut self, pieces: Vec<(Channel, String)>) -> Vec<(Channel, String)> {
        let out = pieces.into_iter().flat_map(|(ch, s)| self.stops.push(ch, &s)).collect();
        self.hold_whitespace(out)
    }

    /// With tools, answer text that is only whitespace waits for text after it (and goes out with
    /// it); the newlines before, between and after call blocks then never reach the client, as
    /// with vLLM's parsers. A streaming client that sends them back would have the template
    /// render them into the history (Qwen3: `assistant\n\n\n<tool_call>`), which no training
    /// conversation looks like: Qwen3-30B-A3B then skipped tools it used otherwise.
    fn hold_whitespace(&mut self, out: Vec<(Channel, String)>) -> Vec<(Channel, String)> {
        let Some(tc) = self.tools.as_mut() else { return out };
        let mut res = Vec::with_capacity(out.len());
        for (ch, s) in out {
            if ch != Channel::Content {
                res.push((ch, s));
            } else if s.trim().is_empty() {
                tc.held.push_str(&s);
            } else {
                res.push((ch, std::mem::take(&mut tc.held) + &s));
            }
        }
        res
    }

    /// The finish reason and usage to report: stopped at a stop string, with the tokens up to it.
    /// A reply that made calls and ended on its own reports `tool_calls`.
    fn report(&self, reason: &FinishReason, usage: &Usage) -> (&'static str, Usage) {
        if self.stopped() {
            ("stop", Usage { completion_tokens: self.ids.len(), ..*usage })
        } else if *reason == FinishReason::Stop && self.tools.as_ref().is_some_and(|t| !t.calls.is_empty()) {
            ("tool_calls", *usage)
        } else {
            (reason.as_openai(), *usage)
        }
    }

    /// vLLM's `stop_reason`: the stop string that ended the text, else null.
    fn stop_reason(&self) -> Value {
        self.stops.matched.as_ref().map_or(Value::Null, |s| json!(s))
    }
}

/// A whole reply: every event of the request through `out`.
async fn drain(mut rx: UnboundedReceiver<Event>, out: &mut Output) -> (Vec<(Channel, String)>, FinishReason, Usage) {
    let mut pieces = Vec::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            Event::Logprobs(lp) => out.logprobs(lp),
            Event::Token(t) => pieces.extend(out.token(t)),
            Event::Finished { reason, usage } => {
                pieces.extend(out.finish());
                return (pieces, reason, usage);
            }
        }
    }
    (pieces, FinishReason::Error("engine stopped before the request finished".into()), Usage::default())
}

/// Frame batches of JSON chunks as SSE, then `data: [DONE]`.
fn sse<S>(chunks: S) -> Response
where
    S: Stream<Item = Vec<Value>> + Send + 'static,
{
    let events = chunks
        .flat_map(stream::iter)
        .map(|v| Ok::<_, Infallible>(SseEvent::default().data(v.to_string())))
        .chain(stream::once(futures::future::ready(Ok(SseEvent::default().data("[DONE]")))));
    Sse::new(events).keep_alive(KeepAlive::default()).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_call_blocks_parse() {
        // arguments as an object (written as vLLM's parser writes them) or a string (kept)
        let c = tool_call("\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"n\": 2}}\n").unwrap();
        assert_eq!(c["type"], "function");
        assert_eq!(c["function"]["name"], "get_weather");
        assert_eq!(c["function"]["arguments"], "{\"city\": \"Paris\", \"n\": 2}");
        assert!(c["id"].as_str().unwrap().starts_with("call_"));
        let s = tool_call(r#"{"name": "f", "arguments": "{\"x\":1}"}"#).unwrap();
        assert_eq!(s["function"]["arguments"], "{\"x\":1}");
        assert_eq!(tool_call(r#"{"name": "f"}"#).unwrap()["function"]["arguments"], "{}");
        // not a call: no name, not JSON
        assert!(tool_call(r#"{"arguments": {}}"#).is_none());
        assert!(tool_call("get_weather(Paris)").is_none());
    }

    #[test]
    fn whitespace_around_calls_is_no_text() {
        let tok: Arc<dyn Tokenizer> = Arc::new(crate::tokenizer::ByteTokenizer);
        // bytes as ids; 1 and 2 stand in for <tool_call> and </tool_call>
        let run = |ids: &[u32]| -> (String, usize) {
            let chat = Some(ChatStream::new(crate::tokenizer::ChatMarkers::default()));
            let mut out = Output::new(tok.clone(), chat, StopStrings::new(Vec::new(), false), Cancel::default());
            out.tools = Some(ToolCalls { open: 1, close: 2, ..Default::default() });
            let mut text = String::new();
            for &t in ids {
                out.token(t).into_iter().for_each(|(_, s)| text += &s);
            }
            out.finish().into_iter().for_each(|(_, s)| text += &s);
            (text, out.tools.as_ref().unwrap().calls.len())
        };
        let call = |name: &str| {
            [vec![1], tok.encode(&format!("{{\"name\": \"{name}\", \"arguments\": {{}}}}")), vec![2]].concat()
        };
        let nl = tok.encode("\n");
        assert_eq!(run(&[nl.clone(), call("f"), nl.clone(), call("g"), nl.clone()].concat()), (String::new(), 2));
        assert_eq!(run(&[tok.encode("Let me check.\n"), call("f")].concat()), ("Let me check.".into(), 1));
        assert_eq!(run(&[call("f"), tok.encode("\n Done.")].concat()), ("\n Done.".into(), 1));
        // a reply without calls keeps its whitespace
        assert_eq!(run(&tok.encode("Hi \n")), ("Hi \n".into(), 0));
    }

    #[test]
    fn tool_choice_checks() {
        let req = |v: &str| serde_json::from_str::<ChatReq>(v).unwrap();
        let tools = r#""tools": [{"type": "function", "function": {"name": "f"}}]"#;
        let m = r#""messages": [{"role": "user", "content": "q"}]"#;
        assert_eq!(req(&format!("{{{m}, {tools}}}")).tools().unwrap().1, ToolChoice::Auto);
        assert_eq!(req(&format!("{{{m}, {tools}, \"tool_choice\": \"none\"}}")).tools().unwrap().1, ToolChoice::None);
        assert_eq!(
            req(&format!(
                "{{{m}, {tools}, \"tool_choice\": {{\"type\": \"function\", \"function\": {{\"name\": \"f\"}}}}}}"
            ))
            .tools()
            .unwrap()
            .1,
            ToolChoice::Named("f".into())
        );
        assert!(req(&format!(
            "{{{m}, {tools}, \"tool_choice\": {{\"type\": \"function\", \"function\": {{\"name\": \"g\"}}}}}}"
        ))
        .tools()
        .is_err());
        assert!(req(&format!("{{{m}, \"tool_choice\": \"required\"}}")).tools().is_err());
        assert!(req(&format!("{{{m}, \"tools\": [{{\"type\": \"function\"}}]}}")).tools().is_err());
    }
}
