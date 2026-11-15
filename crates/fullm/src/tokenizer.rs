// SPDX-License-Identifier: Apache-2.0
//! Tokenizers, chat rendering and streaming detokenization.
//!
//! Two implementations:
//!   - `ByteTokenizer`: token == byte, for the 256-entry test models.
//!   - `KimiTokenizer`: byte-level BPE merging by rank, as tiktoken does. Built from Kimi's
//!     released `tiktoken.model` (base64 token, rank per line) and `tokenizer_config.json`
//!     (special tokens) with the split regex read out of `tokenization_kimi.py`; or from a
//!     Hugging Face `tokenizer.json` whose vocabulary ids are its merge ranks (Qwen, Llama 3:
//!     both tiktoken-derived), with the pre-tokenizer's split regex. (Its NFC normaliser is not
//!     applied: text that is not already NFC may tokenize differently.)
//!
//! SPECIAL TOKENS ARE ONLY RECOGNISED IN STRUCTURAL SEGMENTS. User text that spells
//! `<|end_of_msg|>` encodes as ordinary BPE pieces; recognising it as the control id would
//! let a message close its own turn. Every text-facing `encode` here is ordinary-only. The
//! exception is a Jinja chat template's output (`render_chat` with a template: Qwen3, Llama),
//! tokenized with control tokens recognised because it carries the template's own markers;
//! message text that spells one becomes it there, as in vLLM and SGLang.
//!
//! Chat rendering for Kimi follows the official XTML template (as kimi-k3-in-c renders it,
//! https://github.com/FareedKhan-dev/kimi-k3-in-c): `<|open|>tag attrs<|sep|> body <|close|>tag<|sep|>` with each message
//! terminated by `<|end_of_msg|>`, assistant turns split into think/response channels.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use base64::Engine as _;
use serde::Deserialize;

#[derive(Clone, Debug, Default, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    /// A string, OpenAI's list of parts (the text parts, joined by newlines as vLLM does), or null.
    #[serde(default, deserialize_with = "de_content")]
    pub content: String,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    /// An assistant turn's function calls (OpenAI's `tool_calls`).
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCallMsg>>,
    /// A `tool` message's call (accepted; Qwen3's template does not render it).
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

/// One entry of an assistant message's `tool_calls`.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ToolCallMsg {
    #[serde(default)]
    pub id: Option<String>,
    pub function: ToolFunction,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    /// OpenAI sends a JSON string; an object is accepted too.
    #[serde(default)]
    pub arguments: serde_json::Value,
}

/// JSON as Python's `json.dumps` writes it by default (what the chat templates' `tojson` and
/// vLLM's tool parsers produce): `", "` and `": "` separators, keys in the order given, non-ASCII
/// as is.
pub fn py_json(v: &serde_json::Value) -> String {
    crate::chat_template::py_dumps(v, None, ", ", ": ", false)
}

fn de_content<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<String, D::Error> {
    use serde::de::Error;
    match serde_json::Value::deserialize(d)? {
        serde_json::Value::Null => Ok(String::new()),
        serde_json::Value::String(s) => Ok(s),
        serde_json::Value::Array(parts) => {
            let mut texts = Vec::with_capacity(parts.len());
            for p in &parts {
                match (p.get("type").and_then(|t| t.as_str()), p.get("text").and_then(|t| t.as_str())) {
                    (Some("text"), Some(t)) => texts.push(t),
                    (kind, _) => {
                        return Err(D::Error::custom(format!("unsupported content part {:?}", kind.unwrap_or(""))))
                    }
                }
            }
            Ok(texts.join("\n"))
        }
        v => Err(D::Error::custom(format!("content must be a string, a list of parts or null, not {v}"))),
    }
}

#[derive(Clone, Debug)]
pub struct ChatOptions {
    /// Open the think channel (Kimi). When off, the generation prompt opens `response`.
    pub thinking: bool,
    /// `low`, `medium`, `high` or `max`.
    pub thinking_effort: String,
    /// The request's `tools` (OpenAI function definitions, as given), for the template's tools block.
    pub tools: Vec<serde_json::Value>,
    /// The request's `chat_template_kwargs`, for a Jinja template (`enable_thinking` is `thinking`).
    pub kwargs: serde_json::Map<String, serde_json::Value>,
}

impl Default for ChatOptions {
    fn default() -> Self {
        ChatOptions { thinking: true, thinking_effort: "max".into(), tools: Vec::new(), kwargs: serde_json::Map::new() }
    }
}

/// Token-id sequences that move a chat stream between channels. Empty markers never match.
#[derive(Clone, Debug, Default)]
pub struct ChatMarkers {
    /// Start of reasoning, recognised before any answer text (Qwen3 opens its turn with `<think>`).
    pub think_start: Vec<u32>,
    /// End of reasoning, start of the visible answer.
    pub think_end: Vec<u32>,
    /// End of the answer (the turn's closing tags).
    pub reply_end: Vec<u32>,
    /// Whether generation starts inside the reasoning channel.
    pub starts_in_reasoning: bool,
    /// A whole reply drops the newlines around each channel (Qwen3: `<think>\n...\n</think>\n\n` answer).
    /// ChatStream passes them on, so a stream and the stop strings see the text as generated.
    pub trim_newlines: bool,
}

pub trait Tokenizer: Send + Sync {
    /// Ordinary text; special-token spellings are NOT recognised.
    fn encode(&self, text: &str) -> Vec<u32>;
    /// Text in which special-token spellings ARE the control ids: only for a prompt the client
    /// wrote whole (`serve --special-tokens-in-prompts`), never for text inside a chat message.
    fn encode_with_special(&self, text: &str) -> Vec<u32> {
        self.encode(text)
    }
    /// Raw bytes of the given ids (special tokens contribute their spelling).
    fn decode_bytes(&self, ids: &[u32]) -> Vec<u8>;
    fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }
    /// Control tokens: never rendered as text in API output.
    fn is_special(&self, id: u32) -> bool;
    /// The prompt for a conversation, ending with the generation prompt.
    fn render_chat(&self, msgs: &[ChatMessage], opts: &ChatOptions) -> Result<Vec<u32>>;
    /// How generated ids split into reasoning and answer for `opts`.
    fn chat_markers(&self, opts: &ChatOptions) -> ChatMarkers;
    /// A switch for thinking in the messages themselves (Qwen3's `/think` and `/no_think`), when
    /// the model has one: the last one given, or None.
    fn soft_thinking(&self, _msgs: &[ChatMessage]) -> Option<bool> {
        None
    }
    /// Ids that end a turn.
    fn stop_ids(&self) -> Vec<u32>;
    /// The ids that open and close a function call in a reply (Qwen3's `<tool_call>`,
    /// `</tool_call>`), when the chat format has them.
    fn tool_call_ids(&self) -> Option<(u32, u32)> {
        None
    }
    fn vocab_size(&self) -> usize;
}

/// Pick the tokenizer for a model directory: Kimi when `tiktoken.model` is present, bytes
/// for a vocabulary of at most 256, otherwise an error naming what is missing.
pub fn load_tokenizer(dir: &Path, vocab: usize) -> Result<Box<dyn Tokenizer>> {
    load_tokenizer_with(dir, vocab, None)
}

/// `load_tokenizer`, the chat rendered with the Jinja template in `template` (a file) instead of
/// the model's (`serve --chat-template`).
pub fn load_tokenizer_with(dir: &Path, vocab: usize, template: Option<&Path>) -> Result<Box<dyn Tokenizer>> {
    let template = match template {
        Some(p) => Some(std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?),
        None => None,
    };
    if dir.join("tiktoken.model").exists() {
        let mut t = KimiTokenizer::load(dir)?;
        if let Some(src) = template {
            t.use_template(&src, &tokenizer_config(dir))?;
        }
        return Ok(Box::new(t));
    }
    if dir.join("tokenizer.json").exists() {
        return Ok(Box::new(KimiTokenizer::load_hf_json(dir, template)?));
    }
    if vocab <= 256 {
        return Ok(Box::new(ByteTokenizer));
    }
    bail!("{}: no tiktoken.model, and a {vocab}-entry vocabulary is not byte-level", dir.display())
}

// ----------------------------------------------------------------------- streaming text

/// Turns a byte stream into UTF-8 text, holding back an incomplete trailing character
/// until its continuation bytes arrive. Invalid bytes become U+FFFD.
#[derive(Default)]
pub struct Detokenizer {
    pending: Vec<u8>,
}

impl Detokenizer {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        let mut at = 0;
        loop {
            match std::str::from_utf8(&self.pending[at..]) {
                Ok(s) => {
                    out.push_str(s);
                    at = self.pending.len();
                    break;
                }
                Err(e) => {
                    let good = at + e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[at..good]).expect("validated"));
                    match e.error_len() {
                        Some(n) => {
                            out.push('\u{FFFD}');
                            at = good + n;
                        }
                        None => {
                            at = good; // incomplete tail: keep it
                            break;
                        }
                    }
                }
            }
        }
        self.pending.drain(..at);
        out
    }

    /// Whatever is still held back, decoded lossily.
    pub fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    Reasoning,
    Content,
}

/// Routes generated ids to the reasoning or content channel, swallowing the XTML control
/// sequences between them. Ids that might begin a marker are held until it is decided.
pub struct ChatStream {
    markers: ChatMarkers,
    phase: Option<Channel>,
    held: Vec<u32>,
    reasoning: Detokenizer,
    content: Detokenizer,
    /// Whether each channel (reasoning, content) has emitted text yet.
    started: [bool; 2],
}

impl ChatStream {
    pub fn new(markers: ChatMarkers) -> Self {
        let phase = Some(if markers.starts_in_reasoning { Channel::Reasoning } else { Channel::Content });
        ChatStream {
            markers,
            phase,
            held: Vec::new(),
            reasoning: Detokenizer::default(),
            content: Detokenizer::default(),
            started: [false; 2],
        }
    }

    /// In the answer channel with nothing held back (a tool call's ids can be taken out here).
    pub fn in_content(&self) -> bool {
        self.phase == Some(Channel::Content) && self.held.is_empty()
    }

    pub fn push(&mut self, tok: &dyn Tokenizer, id: u32) -> Vec<(Channel, String)> {
        let mut out = Vec::new();
        if self.phase.is_none() {
            return out;
        }
        self.held.push(id);
        loop {
            let Some(phase) = self.phase else {
                self.held.clear();
                break;
            };
            let think_start = phase == Channel::Content && !self.started[1] && !self.markers.think_start.is_empty();
            if think_start && self.held == self.markers.think_start {
                self.phase = Some(Channel::Reasoning);
                self.held.clear();
                break;
            }
            let think_end = phase == Channel::Reasoning && !self.markers.think_end.is_empty();
            if think_end && self.held == self.markers.think_end {
                self.phase = Some(Channel::Content);
                self.held.clear();
                break;
            }
            if !self.markers.reply_end.is_empty() && self.held == self.markers.reply_end {
                self.phase = None;
                self.held.clear();
                break;
            }
            let prefix_of = |m: &Vec<u32>| !m.is_empty() && m.len() > self.held.len() && m.starts_with(&self.held);
            if (think_start && prefix_of(&self.markers.think_start))
                || (think_end && prefix_of(&self.markers.think_end))
                || prefix_of(&self.markers.reply_end)
            {
                break; // undecided: wait for more ids
            }
            if self.held.is_empty() {
                break;
            }
            let first = self.held.remove(0);
            self.emit(tok, phase, first, &mut out);
        }
        out
    }

    /// Flush held ids as text of the current channel (the stream ended mid-marker).
    pub fn finish(&mut self, tok: &dyn Tokenizer) -> Vec<(Channel, String)> {
        let mut out = Vec::new();
        if let Some(phase) = self.phase {
            for id in std::mem::take(&mut self.held) {
                self.emit(tok, phase, id, &mut out);
            }
        }
        for ch in [Channel::Reasoning, Channel::Content] {
            let s = if ch == Channel::Reasoning { self.reasoning.finish() } else { self.content.finish() };
            self.text(ch, s, &mut out);
        }
        out
    }

    fn emit(&mut self, tok: &dyn Tokenizer, phase: Channel, id: u32, out: &mut Vec<(Channel, String)>) {
        if tok.is_special(id) {
            return;
        }
        let d = if phase == Channel::Reasoning { &mut self.reasoning } else { &mut self.content };
        let s = d.push(&tok.decode_bytes(&[id]));
        self.text(phase, s, out);
    }

    fn text(&mut self, ch: Channel, s: String, out: &mut Vec<(Channel, String)>) {
        if !s.is_empty() {
            self.started[(ch == Channel::Content) as usize] = true;
            out.push((ch, s));
        }
    }
}

// ----------------------------------------------------------------------- bytes

/// Token == byte. Chat is a plain "role: content" transcript.
pub struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(u32::from).collect()
    }
    fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        ids.iter().map(|&i| i as u8).collect()
    }
    fn is_special(&self, _id: u32) -> bool {
        false
    }
    fn render_chat(&self, msgs: &[ChatMessage], _opts: &ChatOptions) -> Result<Vec<u32>> {
        let mut s = String::new();
        for m in msgs {
            s.push_str(&format!("{}: {}\n", m.role, m.content));
        }
        s.push_str("assistant: ");
        Ok(self.encode(&s))
    }
    fn chat_markers(&self, _opts: &ChatOptions) -> ChatMarkers {
        ChatMarkers::default()
    }
    fn stop_ids(&self) -> Vec<u32> {
        Vec::new()
    }
    fn vocab_size(&self) -> usize {
        256
    }
}

// ----------------------------------------------------------------------- Kimi

/// The Kimi K2 split pattern (Moonshot AI's, from the models' tokenization_kimi.py; NOTICE), used
/// when tokenization_kimi.py is not beside the vocabulary.
/// Han runs split first, and Han is excluded from the letter classes.
const KIMI_PATTERN: &[&str] = &[
    r"[\p{Han}]+",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\p{N}{1,3}",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"\s*[\r\n]+",
    r"\s+(?!\S)",
    r"\s+",
];

const OPEN: &str = "<|open|>";
const CLOSE: &str = "<|close|>";
const SEP: &str = "<|sep|>";
const EOM: &str = "<|end_of_msg|>";

/// Which chat format a Kimi vocabulary speaks, detected from its control tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatFormat {
    /// Kimi K3: `<|open|>tag<|sep|> ... <|close|>tag<|sep|>`, think/response channels.
    Xtml,
    /// Kimi K2 / Kimi Linear (chat_template.jinja): `<|im_user|>user<|im_middle|>...<|im_end|>`.
    ImTags,
    /// ChatML (Qwen): `<|im_start|>role\n...<|im_end|>\n`.
    ChatMl,
    /// Any other: the model's Jinja chat template (`chat_template`).
    Template,
}

pub struct KimiTokenizer {
    format: ChatFormat,
    ranks: HashMap<Vec<u8>, u32>,
    /// id -> bytes, ordinary and special alike.
    decoder: Vec<Vec<u8>>,
    special: HashMap<String, u32>,
    is_special: Vec<bool>,
    eos: Option<u32>,
    split: fancy_regex::Regex,
    /// ChatML: the model's chat template opens its replies with a reasoning block (Qwen3's
    /// `enable_thinking`, or a generation prompt that ends in `<think>\n` as Qwen3-2507 Thinking's
    /// does). With thinking on, the generation prompt then carries the `<think>\n` the reply would
    /// start with, prefilled with the prompt instead of two decode steps, and generation starts in
    /// the reasoning channel.
    think_prompt: bool,
    /// ChatML: the template has Qwen3's `enable_thinking` switch, and the model the soft switch
    /// (`/think`, `/no_think` in a user or system message) that goes with it.
    soft_switch: bool,
    /// The chat rendered by the Jinja template (`chat_template`), instead of the native renderer.
    template: Option<std::sync::Arc<crate::chat_template::ChatTemplate>>,
    /// Ids that end a turn besides the format's: generation_config.json's eos_token_id.
    extra_stops: Vec<u32>,
}

impl KimiTokenizer {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("tiktoken.model");
        let raw = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let mut ranks = HashMap::new();
        for (ln, line) in raw.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let mut it = line.split_whitespace();
            let (Some(b64), Some(rank)) = (it.next(), it.next()) else {
                bail!("{}:{}: expected `base64 rank`", path.display(), ln + 1);
            };
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(b64)
                .with_context(|| format!("{}:{}: bad base64", path.display(), ln + 1))?;
            let rank: u32 = rank.parse().with_context(|| format!("{}:{}: bad rank", path.display(), ln + 1))?;
            ranks.insert(bytes, rank);
        }

        let cfg_path = dir.join("tokenizer_config.json");
        let mut special = HashMap::new();
        let mut eos = None;
        if let Ok(raw) = std::fs::read_to_string(&cfg_path) {
            let v: serde_json::Value =
                serde_json::from_str(&raw).with_context(|| format!("parsing {}", cfg_path.display()))?;
            if let Some(obj) = v.get("added_tokens_decoder").and_then(|x| x.as_object()) {
                for (id, entry) in obj {
                    let id: u32 = id.parse().with_context(|| format!("added token id {id:?}"))?;
                    if let Some(c) = entry.get("content").and_then(|c| c.as_str()) {
                        special.insert(c.to_string(), id);
                    }
                }
            }
            let eos_name =
                v.get("eos_token").and_then(|e| e.as_str().or_else(|| e.get("content").and_then(|c| c.as_str())));
            eos = eos_name.and_then(|n| special.get(n).copied());
        }

        let pattern = match std::fs::read_to_string(dir.join("tokenization_kimi.py")) {
            Ok(src) => extract_pat_str(&src).context("tokenization_kimi.py has no parseable pat_str")?,
            Err(_) => KIMI_PATTERN.join("|"),
        };
        Self::from_parts(ranks, special, eos, &pattern)
    }

    pub fn from_parts(
        ranks: HashMap<Vec<u8>, u32>,
        special: HashMap<String, u32>,
        eos: Option<u32>,
        pattern: &str,
    ) -> Result<Self> {
        let max_id = ranks.values().chain(special.values()).copied().max().unwrap_or(0) as usize;
        let mut decoder = vec![Vec::new(); max_id + 1];
        let mut is_special = vec![false; max_id + 1];
        for (b, &id) in &ranks {
            decoder[id as usize] = b.clone();
        }
        for (s, &id) in &special {
            decoder[id as usize] = s.as_bytes().to_vec();
            is_special[id as usize] = true;
        }
        let split = fancy_regex::Regex::new(pattern).context("compiling the split pattern")?;
        let format = if special.contains_key(OPEN) {
            ChatFormat::Xtml
        } else if special.contains_key("<|im_start|>") {
            ChatFormat::ChatMl
        } else {
            ChatFormat::ImTags
        };
        Ok(KimiTokenizer {
            format,
            ranks,
            decoder,
            special,
            is_special,
            eos,
            split,
            think_prompt: false,
            soft_switch: false,
            template: None,
            extra_stops: Vec::new(),
        })
    }

    /// From a Hugging Face `tokenizer.json` (byte-level BPE) and `tokenizer_config.json`.
    /// `template`: a Jinja template to use instead of the model's.
    pub fn load_hf_json(dir: &Path, template_override: Option<String>) -> Result<Self> {
        let path = dir.join("tokenizer.json");
        let raw = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
        let model = v.get("model").context("tokenizer.json: no model")?;
        let kind = model.get("type").and_then(|t| t.as_str()).unwrap_or("");
        ensure!(kind == "BPE", "tokenizer.json: model type {kind:?} (only byte-level BPE)");
        let vocab = model.get("vocab").and_then(|x| x.as_object()).context("tokenizer.json: no model.vocab")?;
        let unbyte = byte_level_decoder();
        let mut ranks = HashMap::with_capacity(vocab.len());
        for (tok, id) in vocab {
            let id = id.as_u64().context("tokenizer.json: non-integer vocab id")? as u32;
            let bytes: Option<Vec<u8>> = tok.chars().map(|c| unbyte.get(&c).copied()).collect();
            let bytes = bytes.with_context(|| format!("tokenizer.json: {tok:?} is not byte-level"))?;
            ranks.insert(bytes, id);
        }
        let mut special = HashMap::new();
        for t in v.get("added_tokens").and_then(|x| x.as_array()).into_iter().flatten() {
            if let (Some(c), Some(id)) =
                (t.get("content").and_then(|c| c.as_str()), t.get("id").and_then(|i| i.as_u64()))
            {
                special.insert(c.to_string(), id as u32);
            }
        }
        let pattern = find_split_regex(v.get("pre_tokenizer").unwrap_or(&serde_json::Value::Null))
            .context("tokenizer.json: no Split regex in the pre-tokenizer")?;
        let cfg = tokenizer_config(dir);
        let eos = cfg
            .as_ref()
            .and_then(|c| c.get("eos_token"))
            .and_then(|e| e.as_str().or_else(|| e.get("content").and_then(|c| c.as_str())))
            .and_then(|n| special.get(n).copied());
        // the template: tokenizer_config.json's chat_template (a string, or named templates), else
        // chat_template.jinja
        let template = cfg
            .as_ref()
            .and_then(|c| c.get("chat_template"))
            .and_then(|t| {
                t.as_str().map(str::to_owned).or_else(|| {
                    let named =
                        t.as_array()?.iter().find(|e| e.get("name").and_then(|n| n.as_str()) == Some("default"))?;
                    named.get("template").and_then(|x| x.as_str()).map(str::to_owned)
                })
            })
            .or_else(|| std::fs::read_to_string(dir.join("chat_template.jinja")).ok());
        let forced = template_override.is_some();
        let template = template_override.or(template);
        let mut t = Self::from_parts(ranks, special, eos, &pattern)?;
        // generation_config.json's end ids (Llama 3's <|eot_id|> and <|eom_id|>, Qwen3's <|endoftext|>)
        if let Some(e) = std::fs::read_to_string(dir.join("generation_config.json"))
            .ok()
            .and_then(|r| serde_json::from_str::<serde_json::Value>(&r).ok())
            .and_then(|g| g.get("eos_token_id").cloned())
        {
            let ids: Vec<u32> = match e {
                serde_json::Value::Array(a) => a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect(),
                v => v.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
            };
            t.extra_stops = ids.into_iter().filter(|&i| (i as usize) < t.decoder.len()).collect();
        }
        let src = template.as_deref().unwrap_or("");
        t.soft_switch = src.contains("enable_thinking");
        t.think_prompt = t.soft_switch || src.contains("assistant\\n<think>");
        if let Some(src) = template {
            if t.format == ChatFormat::ImTags {
                t.format = ChatFormat::Template; // the im-tags format is Kimi's tiktoken one: not from tokenizer.json
            }
            // ChatML keeps the native renderer when it renders this template's prompts exactly
            // (it never reads control tokens out of message text); else, and for every other
            // format, the template renders
            let native_ok = !forced
                && t.format == ChatFormat::ChatMl
                && std::env::var("FULLM_CHAT_TEMPLATE").map_or(true, |v| v != "jinja");
            let tpl = crate::chat_template::ChatTemplate::new(
                &src,
                special_name(&cfg, "bos_token"),
                special_name(&cfg, "eos_token"),
            )
            .context("the model's chat template")?;
            let tpl = std::sync::Arc::new(tpl);
            if !(native_ok && t.native_matches(&tpl)) {
                if t.format == ChatFormat::ChatMl {
                    tracing::info!(
                        "chat template: Jinja (the native ChatML renderer does not reproduce this template)"
                    );
                }
                t.template = Some(tpl);
            }
        }
        Ok(t)
    }

    /// The chat rendered by the Jinja template `src` from now on (`--chat-template`).
    fn use_template(&mut self, src: &str, cfg: &Option<serde_json::Value>) -> Result<()> {
        let tpl = crate::chat_template::ChatTemplate::new(
            src,
            special_name(cfg, "bos_token"),
            special_name(cfg, "eos_token"),
        )?;
        self.soft_switch = src.contains("enable_thinking");
        self.think_prompt = self.soft_switch || src.contains("assistant\\n<think>");
        if self.format == ChatFormat::ImTags {
            self.format = ChatFormat::Template;
        }
        self.template = Some(std::sync::Arc::new(tpl));
        Ok(())
    }

    /// Whether the native ChatML renderer gives `tpl`'s prompts token for token over a set of
    /// conversations (system, multi-turn with reasoning, tools, calls and results, thinking on
    /// and off).
    fn native_matches(&self, tpl: &crate::chat_template::ChatTemplate) -> bool {
        check_conversations().iter().all(|(msgs, opts)| {
            let native = self.render_chatml(msgs, opts);
            let jinja = self.render_template(tpl, msgs, opts);
            match (&native, &jinja) {
                (Ok(a), Ok(b)) if a == b => true,
                _ => {
                    tracing::debug!(
                        "native ChatML vs template differ: {:?} / {:?}",
                        native.as_ref().map(|v| self.decode(v)),
                        jinja.as_ref().map(|v| self.decode(v))
                    );
                    false
                }
            }
        })
    }

    /// A conversation through a Jinja template: the text it renders, read with the control
    /// tokens it spells (as transformers tokenizes it), and with thinking on and a template that
    /// opens replies with reasoning, the `<think>\n` the reply would start with (`think_prompt`).
    fn render_template(
        &self,
        tpl: &crate::chat_template::ChatTemplate,
        msgs: &[ChatMessage],
        opts: &ChatOptions,
    ) -> Result<Vec<u32>> {
        ensure!(!msgs.is_empty(), "no messages");
        for (i, m) in msgs.iter().enumerate() {
            ensure!(
                ["system", "user", "assistant", "tool", "developer"].contains(&m.role.as_str()),
                "message {}: unknown role {:?}",
                i + 1,
                m.role
            );
        }
        let messages: Vec<serde_json::Value> = msgs.iter().map(message_json).collect();
        let tools = (!opts.tools.is_empty()).then(|| serde_json::Value::Array(opts.tools.clone()));
        let mut kwargs = opts.kwargs.clone();
        kwargs.insert("enable_thinking".into(), serde_json::Value::Bool(opts.thinking));
        let text = tpl.render(&serde_json::Value::Array(messages), tools.as_ref(), true, &kwargs)?;
        let mut ids = self.encode_with_special(&text);
        if opts.thinking && self.think_prompt && !text.ends_with("<think>\n") {
            if let Ok(ts) = self.special_id("<think>") {
                ids.push(ts);
                ids.extend(self.encode("\n"));
            }
        }
        Ok(ids)
    }

    pub fn format(&self) -> ChatFormat {
        self.format
    }

    /// ChatML as Qwen3's chat template renders it: per message `<|im_start|>` + role + `\n` +
    /// content + `<|im_end|>` + `\n`. An assistant turn's reasoning (its `reasoning_content`, or a
    /// `<think>...</think>` head of its content) is kept as `<think>\n...\n</think>\n\n` only
    /// after the last user query (a user message that is not a `<tool_response>`), else dropped.
    /// Then `<|im_start|>assistant\n`, plus an empty `<think>\n\n</think>\n\n` with thinking off
    /// (the template's `enable_thinking=false`), or the reply's opening `<think>\n` with thinking on
    /// when the template reasons (`think_prompt`; the template itself leaves it to the model, so a
    /// burst of prompts paid two decode steps, each carrying the others' prefill chunks, before a
    /// reply's first visible text). The segments split where the template's control
    /// tokens split its text, so BPE merges as it does on the rendered string.
    fn render_chatml(&self, msgs: &[ChatMessage], opts: &ChatOptions) -> Result<Vec<u32>> {
        validate_chat(msgs)?;
        let (start, end) = (self.special_id("<|im_start|>")?, self.special_id("<|im_end|>")?);
        let think = self.special_id("<think>").ok().zip(self.special_id("</think>").ok());
        let last_query = msgs
            .iter()
            .rposition(|m| {
                m.role == "user"
                    && !(m.content.starts_with("<tool_response>") && m.content.ends_with("</tool_response>"))
            })
            .unwrap_or(msgs.len() - 1);
        let mut ids = Vec::new();
        // Qwen3's tools block: one system turn, the request's system message (if any) first, then
        // the functions (as the template's `tojson` writes them) and the call format. The template's
        // own text is encoded with its control tokens (<tool_call> ...), the messages' without.
        let mut first = 0;
        if !opts.tools.is_empty() {
            ids.push(start);
            ids.extend(self.encode("system\n"));
            if msgs[0].role == "system" {
                ids.extend(self.encode(&format!("{}\n\n", msgs[0].content)));
                first = 1;
            }
            // Qwen3's chat template text (Qwen, Apache-2.0; NOTICE). (Segments split where the
            // pre-tokenizer splits the whole text: after each "\n" that ends ">\n" or "}\n", so BPE
            // merges as on the rendered string.)
            ids.extend(self.encode_with_special(
                "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with \
                 function signatures within <tools></tools> XML tags:\n<tools>\n",
            ));
            for t in &opts.tools {
                ids.extend(self.encode(&format!("{}\n", py_json(t))));
            }
            ids.extend(self.encode_with_special(
                "</tools>\n\nFor each function call, return a json object with function name and arguments within \
                 <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": \
                 <args-json-object>}\n</tool_call>",
            ));
            ids.push(end);
            ids.extend(self.encode("\n"));
        }
        for (i, m) in msgs.iter().enumerate().skip(first) {
            if m.role == "tool" {
                // consecutive tool results share one user turn, each in <tool_response> tags
                if i == 0 || msgs[i - 1].role != "tool" {
                    ids.push(start);
                    ids.extend(self.encode("user"));
                }
                ids.extend(self.encode_with_special("\n<tool_response>\n"));
                ids.extend(self.encode(&format!("{}\n", m.content))); // "}\n" is one piece
                ids.extend(self.encode_with_special("</tool_response>"));
                if i + 1 == msgs.len() || msgs[i + 1].role != "tool" {
                    ids.push(end);
                    ids.extend(self.encode("\n"));
                }
                continue;
            }
            ids.push(start);
            if m.role == "assistant" {
                let (reasoning, content) = split_reasoning(m);
                // the newline between its text and its first call goes with the text ("."\n is one piece)
                let calls = m.tool_calls.as_deref().unwrap_or_default();
                let sep = if !calls.is_empty() && !content.is_empty() { "\n" } else { "" };
                match think {
                    Some((ts, te)) if i > last_query && (i + 1 == msgs.len() || !reasoning.is_empty()) => {
                        ids.extend(self.encode("assistant\n"));
                        ids.push(ts);
                        ids.extend(self.encode(&format!("\n{}\n", reasoning.trim_matches('\n'))));
                        ids.push(te);
                        ids.extend(self.encode(&format!("\n\n{}{sep}", content.trim_start_matches('\n'))));
                    }
                    _ => ids.extend(self.encode(&format!("assistant\n{content}{sep}"))),
                }
                // its calls: <tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>, newline-separated
                for (j, c) in calls.iter().enumerate() {
                    if j > 0 {
                        ids.extend(self.encode("\n"));
                    }
                    let args = match &c.function.arguments {
                        serde_json::Value::String(s) => s.clone(),
                        serde_json::Value::Null => "{}".to_string(),
                        v => py_json(v),
                    };
                    ids.extend(self.encode_with_special("<tool_call>"));
                    ids.extend(
                        self.encode(&format!("\n{{\"name\": \"{}\", \"arguments\": {args}}}\n", c.function.name)),
                    );
                    ids.extend(self.encode_with_special("</tool_call>"));
                }
            } else {
                ids.extend(self.encode(&format!("{}\n{}", m.role, m.content)));
            }
            ids.push(end);
            ids.extend(self.encode("\n"));
        }
        ids.push(start);
        ids.extend(self.encode("assistant\n"));
        match (opts.thinking, think) {
            (false, Some((ts, te))) => {
                ids.push(ts);
                ids.extend(self.encode("\n\n"));
                ids.push(te);
                ids.extend(self.encode("\n\n"));
            }
            (true, Some((ts, _))) if self.think_prompt => {
                ids.push(ts);
                ids.extend(self.encode("\n"));
            }
            _ => {}
        }
        Ok(ids)
    }

    /// chat_template.jinja of Kimi Linear: per message `<|im_{user,assistant,system}|>` +
    /// role + `<|im_middle|>` + content + `<|im_end|>`, then the generation prompt
    /// `<|im_assistant|>assistant<|im_middle|>`. Text segments never parse control tokens.
    fn render_im_tags(&self, msgs: &[ChatMessage]) -> Result<Vec<u32>> {
        validate_chat(msgs)?;
        let (user, asst, sys, mid, end) = (
            self.special_id("<|im_user|>")?,
            self.special_id("<|im_assistant|>")?,
            self.special_id("<|im_system|>")?,
            self.special_id("<|im_middle|>")?,
            self.special_id("<|im_end|>")?,
        );
        let mut ids = Vec::new();
        for m in msgs {
            ids.push(match m.role.as_str() {
                "user" => user,
                "assistant" => asst,
                _ => sys,
            });
            ids.extend(self.encode(&m.role));
            ids.push(mid);
            ids.extend(self.encode(&m.content));
            ids.push(end);
        }
        ids.push(asst);
        ids.extend(self.encode("assistant"));
        ids.push(mid);
        Ok(ids)
    }

    fn special_id(&self, s: &str) -> Result<u32> {
        self.special.get(s).copied().with_context(|| format!("tokenizer has no control token {s}"))
    }

    fn encode_piece(&self, piece: &[u8], out: &mut Vec<u32>) {
        if let Some(&r) = self.ranks.get(piece) {
            out.push(r);
            return;
        }
        let parts = byte_pair_merge(&self.ranks, piece);
        for w in parts.windows(2) {
            let seg = &piece[w[0].0..w[1].0];
            match self.ranks.get(seg) {
                Some(&r) => out.push(r),
                // A vocabulary without some single byte cannot encode it at all; fall back
                // to whatever single bytes exist rather than dropping text silently.
                None => out.extend(seg.iter().filter_map(|b| self.ranks.get(std::slice::from_ref(b)).copied())),
            }
        }
    }
}

/// GPT-2's byte-level alphabet, inverted: the printable character standing for each byte.
fn byte_level_decoder() -> HashMap<char, u8> {
    let mut m = HashMap::with_capacity(256);
    let mut n = 0u32;
    for b in 0u32..256 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        let c = if printable {
            b
        } else {
            n += 1;
            255 + n
        };
        m.insert(char::from_u32(c).expect("valid code point"), b as u8);
    }
    m
}

/// The first `Split` pre-tokenizer's regex, searching nested `Sequence`s.
fn find_split_regex(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::Object(o) => {
            if o.get("type").and_then(|t| t.as_str()) == Some("Split") {
                if let Some(r) = o.get("pattern").and_then(|p| p.get("Regex")).and_then(|r| r.as_str()) {
                    return Some(r.to_string());
                }
            }
            o.values().find_map(find_split_regex)
        }
        serde_json::Value::Array(a) => a.iter().find_map(find_split_regex),
        _ => None,
    }
}

/// Pieces at least this long merge from a heap (`byte_pair_merge_heap`), shorter ones with the
/// scan (`byte_pair_merge_scan`, cheaper for the few parts of a usual word).
const HEAP_MERGE_MIN: usize = 128;

/// BPE: repeatedly join the adjacent pair whose concatenation has the lowest rank (the leftmost
/// of equal ranks). Returns part boundaries (start offsets plus the end).
fn byte_pair_merge(ranks: &HashMap<Vec<u8>, u32>, piece: &[u8]) -> Vec<(usize, u32)> {
    if piece.len() >= HEAP_MERGE_MIN {
        byte_pair_merge_heap(ranks, piece)
    } else {
        byte_pair_merge_scan(ranks, piece)
    }
}

/// tiktoken's merge loop, ported from its `_byte_pair_merge` (MIT; see NOTICE): a scan of every
/// part per merge, O(n^2) in the piece's bytes.
fn byte_pair_merge_scan(ranks: &HashMap<Vec<u8>, u32>, piece: &[u8]) -> Vec<(usize, u32)> {
    const NONE: u32 = u32::MAX;
    let rank_of = |parts: &[(usize, u32)], i: usize| -> u32 {
        if i + 3 < parts.len() {
            ranks.get(&piece[parts[i].0..parts[i + 3].0]).copied().unwrap_or(NONE)
        } else {
            NONE
        }
    };
    let mut parts: Vec<(usize, u32)> = Vec::with_capacity(piece.len() + 1);
    let mut min = (NONE, usize::MAX);
    for i in 0..piece.len().saturating_sub(1) {
        let r = ranks.get(&piece[i..i + 2]).copied().unwrap_or(NONE);
        if r < min.0 {
            min = (r, i);
        }
        parts.push((i, r));
    }
    parts.push((piece.len().saturating_sub(1), NONE));
    parts.push((piece.len(), NONE));
    while min.0 != NONE {
        let i = min.1;
        if i > 0 {
            parts[i - 1].1 = rank_of(&parts, i - 1);
        }
        parts[i].1 = rank_of(&parts, i);
        parts.remove(i + 1);
        min = (NONE, usize::MAX);
        for (j, &(_, r)) in parts[..parts.len() - 1].iter().enumerate() {
            if r < min.0 {
                min = (r, j);
            }
        }
    }
    parts
}

/// The scan's merges in the scan's order, O(n log n): the parts a linked list (part k starts at
/// byte k: merging drops the right part, so a part keeps its start), each part's pair rank in a
/// min-heap keyed (rank, start), entries whose part has since changed skipped when popped. A long
/// run without a pre-tokenizer split (a blob of letters) cannot stall the server.
fn byte_pair_merge_heap(ranks: &HashMap<Vec<u8>, u32>, piece: &[u8]) -> Vec<(usize, u32)> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    const NONE: u32 = u32::MAX;
    let n = piece.len();
    let mut next: Vec<usize> = (1..=n).collect();
    let mut prev: Vec<usize> = (0..n).map(|k| k.wrapping_sub(1)).collect();
    // the rank of part k joined with the part after it
    let pair = |k: usize, next: &[usize]| -> u32 {
        let j = next[k];
        if j >= n {
            return NONE;
        }
        ranks.get(&piece[k..next[j]]).copied().unwrap_or(NONE)
    };
    let mut rank = vec![NONE; n];
    let mut heap = BinaryHeap::with_capacity(n);
    for k in 0..n.saturating_sub(1) {
        rank[k] = pair(k, &next);
        if rank[k] != NONE {
            heap.push(Reverse((rank[k], k)));
        }
    }
    while let Some(Reverse((r, k))) = heap.pop() {
        if rank[k] != r {
            continue; // stale: part k merged, or was dropped (rank NONE), since
        }
        let j = next[k];
        next[k] = next[j];
        if next[k] < n {
            prev[next[k]] = k;
        }
        rank[j] = NONE;
        for p in [Some(k), (prev[k] != usize::MAX).then(|| prev[k])].into_iter().flatten() {
            rank[p] = pair(p, &next);
            if rank[p] != NONE {
                heap.push(Reverse((rank[p], p)));
            }
        }
    }
    let mut parts = Vec::new();
    let mut k = 0;
    while k < n {
        parts.push((k, rank[k]));
        k = next[k];
    }
    parts.push((n, NONE));
    parts
}

/// The `pat_str` of tokenization_kimi.py: either `"|".join([r"""a""", r"""b""", ...])` or
/// a single raw string literal.
fn extract_pat_str(src: &str) -> Option<String> {
    let at = src.find("pat_str")?;
    let rest = &src[at..];
    let eq = rest.find('=')?;
    let rhs = rest[eq + 1..].trim_start();
    if let Some(join) = rhs.strip_prefix("\"|\".join(") {
        let body = &join[join.find('[')? + 1..];
        let end = find_list_end(body)?;
        let lits = parse_literals(&body[..end])?;
        (!lits.is_empty()).then(|| lits.join("|"))
    } else {
        let lits = parse_literals(rhs.lines().next()?)?;
        lits.into_iter().next()
    }
}

fn find_list_end(s: &str) -> Option<usize> {
    // Strings may contain ']' (character classes), so skip over literals while scanning.
    let mut i = 0;
    while i < s.len() {
        if s.as_bytes()[i] == b']' {
            return Some(i);
        }
        i += match starts_literal(&s[i..]) {
            true => parse_one_literal(&s[i..])?.1,
            false => 1,
        };
    }
    None
}

fn starts_literal(s: &str) -> bool {
    s.starts_with('"') || s.starts_with('\'') || s.starts_with("r\"") || s.starts_with("r'")
}

fn parse_literals(s: &str) -> Option<Vec<String>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < s.len() {
        if starts_literal(&s[i..]) {
            let (lit, used) = parse_one_literal(&s[i..])?;
            out.push(lit);
            i += used;
        } else if s.as_bytes()[i] == b'#' {
            i += s[i..].find('\n').unwrap_or(s.len() - i);
        } else {
            i += 1;
        }
    }
    Some(out)
}

/// One Python string literal at the start of `s` (raw or plain, triple or single quoted).
/// Plain literals are only accepted without escapes, since the pattern needs raw strings.
fn parse_one_literal(s: &str) -> Option<(String, usize)> {
    let (raw, body) = match s.strip_prefix('r') {
        Some(b) => (true, b),
        None => (false, s),
    };
    let pre = s.len() - body.len();
    for q in ["\"\"\"", "'''", "\"", "'"] {
        if let Some(inner) = body.strip_prefix(q) {
            let end = inner.find(q)?;
            let lit = &inner[..end];
            if !raw && lit.contains('\\') {
                return None;
            }
            return Some((lit.to_string(), pre + q.len() * 2 + end));
        }
    }
    None
}

impl Tokenizer for KimiTokenizer {
    fn encode_with_special(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // the earliest control-token spelling, the longest at that position
            let hit = self
                .special
                .iter()
                .filter_map(|(s, &id)| rest.find(s.as_str()).map(|at| (at, std::cmp::Reverse(s.len()), id)))
                .min();
            let Some((at, std::cmp::Reverse(len), id)) = hit else {
                out.extend(self.encode(rest));
                break;
            };
            out.extend(self.encode(&rest[..at]));
            out.push(id);
            rest = &rest[at + len..];
        }
        out
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut done = 0;
        for m in self.split.find_iter(text) {
            match m {
                Ok(m) => {
                    self.encode_piece(m.as_str().as_bytes(), &mut out);
                    done = m.end();
                }
                // The regex gave up (its backtracking limit): the rest as one piece, not dropped.
                Err(_) => {
                    self.encode_piece(&text.as_bytes()[done..], &mut out);
                    break;
                }
            }
        }
        out
    }

    fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &i in ids {
            if let Some(b) = self.decoder.get(i as usize) {
                out.extend_from_slice(b);
            }
        }
        out
    }

    fn is_special(&self, id: u32) -> bool {
        self.is_special.get(id as usize).copied().unwrap_or(false)
    }

    fn render_chat(&self, msgs: &[ChatMessage], opts: &ChatOptions) -> Result<Vec<u32>> {
        if let Some(t) = &self.template {
            return self.render_template(t, msgs, opts);
        }
        match self.format {
            ChatFormat::ImTags => return self.render_im_tags(msgs),
            ChatFormat::ChatMl => return self.render_chatml(msgs, opts),
            ChatFormat::Template => bail!("this model has no chat template (give one with --chat-template)"),
            ChatFormat::Xtml => {}
        }
        ensure!(
            ["low", "medium", "high", "max"].contains(&opts.thinking_effort.as_str()),
            "thinking_effort must be low, medium, high or max"
        );
        validate_chat(msgs)?;
        let (open, close, sep, eom) =
            (self.special_id(OPEN)?, self.special_id(CLOSE)?, self.special_id(SEP)?, self.special_id(EOM)?);
        let mut ids = Vec::new();
        // Every txt() is its own segment: BPE never merges across a tag boundary.
        let txt = |ids: &mut Vec<u32>, s: &str| ids.extend(self.encode(s));
        let open_tag = |ids: &mut Vec<u32>, tag: &str, attrs: &str| {
            ids.push(open);
            txt(ids, tag);
            txt(ids, attrs);
            ids.push(sep);
        };
        let close_tag = |ids: &mut Vec<u32>, tag: &str| {
            ids.push(close);
            txt(ids, tag);
            ids.push(sep);
        };
        if opts.thinking {
            open_tag(&mut ids, "message", " role=\"system\" type=\"thinking-effort\"");
            txt(
                &mut ids,
                &format!(
                    "`thinking_effort` guides on how much to think in your thinking channel (not including the response \
                     channel), supported values include `low`, `medium`, `high`, and `max`.\nNow the system is invoked \
                     with `thinking_effort={}`.",
                    opts.thinking_effort
                ),
            );
            close_tag(&mut ids, "message");
            ids.push(eom);
        }
        for m in msgs {
            open_tag(&mut ids, "message", &format!(" role=\"{}\"", m.role));
            if m.role == "assistant" {
                // With thinking off the template drops the think channel entirely.
                if opts.thinking {
                    open_tag(&mut ids, "think", "");
                    txt(&mut ids, m.reasoning_content.as_deref().unwrap_or(""));
                    close_tag(&mut ids, "think");
                }
                open_tag(&mut ids, "response", "");
                txt(&mut ids, &m.content);
                close_tag(&mut ids, "response");
            } else {
                txt(&mut ids, &m.content);
            }
            close_tag(&mut ids, "message");
            ids.push(eom);
        }
        open_tag(&mut ids, "message", " role=\"assistant\"");
        open_tag(&mut ids, if opts.thinking { "think" } else { "response" }, "");
        Ok(ids)
    }

    fn soft_thinking(&self, msgs: &[ChatMessage]) -> Option<bool> {
        if !matches!(self.format, ChatFormat::ChatMl | ChatFormat::Template) || !self.soft_switch {
            return None;
        }
        // Qwen3 follows the latest switch, in a user or system message
        msgs.iter().rev().filter(|m| m.role == "user" || m.role == "system").find_map(|m| {
            match (m.content.rfind("/think"), m.content.rfind("/no_think")) {
                (None, None) => None,
                (on, off) => Some(on > off),
            }
        })
    }

    fn chat_markers(&self, opts: &ChatOptions) -> ChatMarkers {
        if matches!(self.format, ChatFormat::ChatMl | ChatFormat::Template) {
            // Qwen3: the turn opens with <think> unless thinking is off (the prompt then closed an
            // empty one) or the prompt opened it (think_prompt); the answer follows </think>; the
            // turn ends at <|im_end|> (a stop id)
            return match (opts.thinking, self.special_id("<think>"), self.special_id("</think>")) {
                (true, Ok(ts), Ok(te)) => ChatMarkers {
                    think_start: vec![ts],
                    think_end: vec![te],
                    starts_in_reasoning: self.think_prompt,
                    trim_newlines: true,
                    ..Default::default()
                },
                _ => ChatMarkers::default(),
            };
        }
        if self.format != ChatFormat::Xtml {
            return ChatMarkers::default(); // one channel; the turn ends at <|im_end|>
        }
        let (Ok(open), Ok(close), Ok(sep)) = (self.special_id(OPEN), self.special_id(CLOSE), self.special_id(SEP))
        else {
            return ChatMarkers::default();
        };
        let mut think_end = vec![close];
        think_end.extend(self.encode("think"));
        think_end.extend([sep, open]);
        think_end.extend(self.encode("response"));
        think_end.push(sep);
        let mut reply_end = vec![close];
        reply_end.extend(self.encode("response"));
        reply_end.extend([sep, close]);
        reply_end.extend(self.encode("message"));
        reply_end.push(sep);
        ChatMarkers { think_end, reply_end, starts_in_reasoning: opts.thinking, ..Default::default() }
    }

    fn tool_call_ids(&self) -> Option<(u32, u32)> {
        // Qwen's (hermes) <tool_call> blocks, in ChatML or another template that has the tokens
        match self.format {
            ChatFormat::ChatMl | ChatFormat::Template => {
                Some((self.special_id("<tool_call>").ok()?, self.special_id("</tool_call>").ok()?))
            }
            _ => None,
        }
    }

    fn stop_ids(&self) -> Vec<u32> {
        // XTML ends a turn with <|end_of_msg|>, the im-tags format with <|im_end|>; [EOS] for both.
        let mut v: Vec<u32> = [EOM, "<|im_end|>"].iter().filter_map(|t| self.special.get(*t).copied()).collect();
        for e in self.eos.iter().chain(&self.extra_stops) {
            if !v.contains(e) {
                v.push(*e);
            }
        }
        v
    }

    fn vocab_size(&self) -> usize {
        self.decoder.len()
    }
}

fn tokenizer_config(dir: &Path) -> Option<serde_json::Value> {
    std::fs::read_to_string(dir.join("tokenizer_config.json")).ok().and_then(|r| serde_json::from_str(&r).ok())
}

/// A token string of tokenizer_config.json (`bos_token`, `eos_token`: a string or `{content}`).
fn special_name(cfg: &Option<serde_json::Value>, key: &str) -> Option<String> {
    let v = cfg.as_ref()?.get(key)?;
    v.as_str().or_else(|| v.get("content").and_then(|c| c.as_str())).map(str::to_owned)
}

/// A message as a chat template reads it (OpenAI's fields; a call's arguments an object).
fn message_json(m: &ChatMessage) -> serde_json::Value {
    use serde_json::{json, Value};
    let mut o = serde_json::Map::new();
    o.insert("role".into(), json!(m.role));
    o.insert("content".into(), json!(m.content));
    if let Some(r) = &m.reasoning_content {
        o.insert("reasoning_content".into(), json!(r));
    }
    if let Some(calls) = &m.tool_calls {
        let calls: Vec<Value> = calls
            .iter()
            .map(|c| {
                let mut f = serde_json::Map::new();
                f.insert("name".into(), json!(c.function.name));
                f.insert("arguments".into(), crate::chat_template::arguments_value(&c.function.arguments));
                let mut e = serde_json::Map::new();
                if let Some(id) = &c.id {
                    e.insert("id".into(), json!(id));
                }
                e.insert("type".into(), json!("function"));
                e.insert("function".into(), Value::Object(f));
                Value::Object(e)
            })
            .collect();
        o.insert("tool_calls".into(), Value::Array(calls));
    }
    if let Some(id) = &m.tool_call_id {
        o.insert("tool_call_id".into(), json!(id));
    }
    Value::Object(o)
}

/// Conversations the native ChatML renderer must render as a template does to stand in for it.
fn check_conversations() -> Vec<(Vec<ChatMessage>, ChatOptions)> {
    use serde_json::json;
    let m =
        |role: &str, content: &str| ChatMessage { role: role.into(), content: content.into(), ..Default::default() };
    let tools = vec![
        json!({"type": "function", "function": {"name": "get_weather", "description": "Weather of a city (\"now\").",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "e.g. Paris"}}, "required": ["city"]}}}),
    ];
    let call = |name: &str, args: serde_json::Value| ToolCallMsg {
        id: Some("call_1".into()),
        function: ToolFunction { name: name.into(), arguments: args },
    };
    let on = ChatOptions::default();
    let off = ChatOptions { thinking: false, ..Default::default() };
    let with_tools = |o: &ChatOptions| ChatOptions { tools: tools.clone(), ..o.clone() };
    let reasoned = ChatMessage { reasoning_content: Some("Think: 2+2.".into()), ..m("assistant", "It is 4.") };
    let calling = ChatMessage {
        tool_calls: Some(vec![
            call("get_weather", json!({"city": "Paris"})),
            call("get_weather", json!("{\"city\": \"Oslo\"}")),
        ]),
        ..m("assistant", "")
    };
    vec![
        (vec![m("user", "Hi there.")], on.clone()),
        (vec![m("user", "Hi there.")], off.clone()),
        (vec![m("system", "Be brief."), m("user", "Hi.")], on.clone()),
        (vec![m("user", "2+2?"), reasoned.clone(), m("user", "And 3+3?")], on.clone()),
        (vec![m("user", "2+2?"), reasoned, m("user", "And 3+3?")], off.clone()),
        (vec![m("user", "Weather in Paris?")], with_tools(&on)),
        (
            vec![
                m("system", "Use tools."),
                m("user", "Weather in Paris and Oslo?"),
                calling.clone(),
                m("tool", "18C"),
                m("tool", "7C"),
            ],
            with_tools(&on),
        ),
        (
            vec![
                m("user", "Paris?"),
                calling,
                m("tool", "18C"),
                m("tool", "7C"),
                m("assistant", "18C and 7C."),
                m("user", "Thanks."),
            ],
            with_tools(&off),
        ),
    ]
}

/// An assistant message's reasoning and answer as Qwen3's template takes them: its
/// `reasoning_content`, else the part of its content before `</think>` (after `<think>`).
fn split_reasoning(m: &ChatMessage) -> (String, String) {
    if let Some(r) = &m.reasoning_content {
        return (r.clone(), m.content.clone());
    }
    match m.content.split_once("</think>") {
        Some((head, _)) => {
            let reasoning = head.trim_end_matches('\n').rsplit("<think>").next().unwrap_or("").trim_start_matches('\n');
            let answer = m.content.rsplit("</think>").next().unwrap_or("").trim_start_matches('\n');
            (reasoning.to_string(), answer.to_string())
        }
        None => (String::new(), m.content.clone()),
    }
}

/// The template's contract: an optional leading system message, then user and assistant
/// strictly alternating from user; only assistants carry reasoning.
fn validate_chat(msgs: &[ChatMessage]) -> Result<()> {
    let mut want = "user";
    for (i, m) in msgs.iter().enumerate() {
        match m.role.as_str() {
            "system" => ensure!(i == 0, "message {}: a system message must come first", i + 1),
            // tool results answer an assistant turn's calls: one or more in a row, where a user turn could be
            "tool" if want == "user" || (i > 0 && msgs[i - 1].role == "tool") => {
                ensure!(
                    m.reasoning_content.is_none() && m.tool_calls.is_none(),
                    "message {}: a tool message carries a result only",
                    i + 1
                );
                want = "assistant";
            }
            r @ ("user" | "assistant") => {
                ensure!(r == want, "message {}: role {r:?} where {want:?} was expected (roles must alternate)", i + 1);
                ensure!(
                    r == "assistant" || (m.reasoning_content.is_none() && m.tool_calls.is_none()),
                    "message {}: only assistant messages carry reasoning or tool calls",
                    i + 1
                );
                want = if r == "user" { "assistant" } else { "user" };
            }
            r => bail!("message {}: unknown role {r:?}", i + 1),
        }
    }
    ensure!(want == "assistant", "the conversation must end with a user or tool message");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The heap merge gives the scan's parts: random pieces over a 3-letter alphabet, every string
    /// of 2-6 letters ranked at random, so pieces merge many times and ranks tie across positions.
    #[test]
    fn heap_merge_matches_scan() {
        let mut x = 0x9e3779b97f4a7c15u64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut ranks = HashMap::new();
        let mut words: Vec<Vec<u8>> = vec![Vec::new()];
        for _ in 0..6 {
            words = words.iter().flat_map(|w| b"abc".iter().map(move |&c| [w.as_slice(), &[c]].concat())).collect();
            for w in &words {
                if w.len() >= 2 {
                    ranks.insert(w.clone(), (rnd() % 400) as u32);
                }
            }
        }
        for len in [2usize, 3, 7, 50, 127, 128, 300, 1000] {
            for _ in 0..20 {
                let piece: Vec<u8> = (0..len).map(|_| b"abc"[(rnd() % 3) as usize]).collect();
                let a: Vec<usize> = byte_pair_merge_scan(&ranks, &piece).iter().map(|p| p.0).collect();
                let b: Vec<usize> = byte_pair_merge_heap(&ranks, &piece).iter().map(|p| p.0).collect();
                assert_eq!(a, b, "piece of {len}");
            }
        }
    }

    /// A toy vocabulary: all single bytes, plus merges "ab" < "bc" < "abc" and specials.
    fn toy() -> KimiTokenizer {
        let mut ranks = HashMap::new();
        for b in 0..=255u8 {
            ranks.insert(vec![b], b as u32);
        }
        ranks.insert(b"ab".to_vec(), 256);
        ranks.insert(b"bc".to_vec(), 257);
        ranks.insert(b"abc".to_vec(), 258);
        ranks.insert(b"message".to_vec(), 259);
        let special: HashMap<String, u32> = [(OPEN, 300), (CLOSE, 301), (SEP, 302), (EOM, 303), ("[EOS]", 304)]
            .into_iter()
            .map(|(s, i)| (s.to_string(), i))
            .collect();
        KimiTokenizer::from_parts(ranks, special, Some(304), &KIMI_PATTERN.join("|")).unwrap()
    }

    #[test]
    fn bpe_merges_lowest_rank_first() {
        let t = toy();
        // "abc": "ab"(256) merges before "bc"(257), then "ab"+"c" -> "abc"(258)
        assert_eq!(t.encode("abc"), vec![258]);
        // "abd": only "ab" applies
        assert_eq!(t.encode("abd"), vec![256, b'd' as u32]);
        // "bcd": "bc"
        assert_eq!(t.encode("xbcd"), vec![b'x' as u32, 257, b'd' as u32]);
        assert_eq!(t.decode(&t.encode("hello abc, world")), "hello abc, world");
    }

    #[test]
    fn user_text_cannot_inject_control_tokens() {
        let t = toy();
        let ids = t.encode("<|end_of_msg|>");
        assert!(!ids.contains(&303), "a control token spelled in text must stay text");
        assert_eq!(t.decode(&ids), "<|end_of_msg|>");
    }

    #[test]
    fn xtml_render_structure() {
        let t = toy();
        let msgs = vec![
            ChatMessage { role: "user".into(), content: "hi".into(), reasoning_content: None, ..Default::default() },
            ChatMessage {
                role: "assistant".into(),
                content: "yo".into(),
                reasoning_content: Some("r".into()),
                ..Default::default()
            },
            ChatMessage { role: "user".into(), content: "q".into(), reasoning_content: None, ..Default::default() },
        ];
        let opts = ChatOptions { thinking: false, thinking_effort: "max".into(), ..Default::default() };
        let ids = t.render_chat(&msgs, &opts).unwrap();
        let text = String::from_utf8(t.decode_bytes(&ids)).unwrap();
        assert_eq!(
            text,
            "<|open|>message role=\"user\"<|sep|>hi<|close|>message<|sep|><|end_of_msg|>\
             <|open|>message role=\"assistant\"<|sep|><|open|>response<|sep|>yo<|close|>response<|sep|>\
             <|close|>message<|sep|><|end_of_msg|>\
             <|open|>message role=\"user\"<|sep|>q<|close|>message<|sep|><|end_of_msg|>\
             <|open|>message role=\"assistant\"<|sep|><|open|>response<|sep|>"
        );
        // control tokens are the special ids, not their spellings
        assert_eq!(ids[0], 300);
        let thinking = t.render_chat(&msgs[..1], &ChatOptions::default()).unwrap();
        let text = String::from_utf8(t.decode_bytes(&thinking)).unwrap();
        assert!(text.starts_with("<|open|>message role=\"system\" type=\"thinking-effort\"<|sep|>`thinking_effort`"));
        assert!(text.contains("`thinking_effort=max`.<|close|>message<|sep|><|end_of_msg|>"));
        assert!(text.ends_with("<|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>"));
        assert!(t.render_chat(&msgs[..2], &opts).is_err(), "must end with a user turn");
    }

    #[test]
    fn chat_stream_splits_channels() {
        let t = toy();
        let opts = ChatOptions::default();
        let m = t.chat_markers(&opts);
        let mut ids = t.encode("why");
        ids.extend(&m.think_end);
        ids.extend(t.encode("because"));
        ids.extend(&m.reply_end);
        ids.push(303);
        let mut s = ChatStream::new(m);
        let mut r = String::new();
        let mut c = String::new();
        for &id in &ids {
            for (ch, d) in s.push(&t, id) {
                if ch == Channel::Reasoning {
                    r += &d
                } else {
                    c += &d
                }
            }
        }
        for (ch, d) in s.finish(&t) {
            if ch == Channel::Reasoning {
                r += &d
            } else {
                c += &d
            }
        }
        assert_eq!((r.as_str(), c.as_str()), ("why", "because"));
    }

    #[test]
    fn detokenizer_holds_partial_utf8() {
        let mut d = Detokenizer::default();
        let bytes = "é漢".as_bytes();
        assert_eq!(d.push(&bytes[..1]), "");
        assert_eq!(d.push(&bytes[1..3]), "é");
        assert_eq!(d.push(&bytes[3..]), "漢");
        assert_eq!(d.push(&[0xFF, b'a']), "\u{FFFD}a");
    }

    #[test]
    fn pat_str_extraction() {
        let src = "class T:\n    pat_str = \"|\".join(\n        [\n            r\"\"\"[\\p{Han}]+\"\"\",\n            r\"\"\"\\s+(?!\\S)\"\"\",  # tail\n            r\"\"\"[a]]\"\"\",\n        ]\n    )\n";
        assert_eq!(extract_pat_str(src).unwrap(), "[\\p{Han}]+|\\s+(?!\\S)|[a]]");
    }

    /// A toy ChatML (Qwen3) vocabulary: all single bytes and its control tokens.
    fn toy_chatml() -> KimiTokenizer {
        let ranks: HashMap<Vec<u8>, u32> = (0..=255u8).map(|b| (vec![b], b as u32)).collect();
        let special: HashMap<String, u32> =
            [("<|im_start|>", 300), ("<|im_end|>", 301), ("<think>", 302), ("</think>", 303), ("<|endoftext|>", 304)]
                .into_iter()
                .map(|(s, i)| (s.to_string(), i))
                .collect();
        KimiTokenizer::from_parts(ranks, special, Some(304), &KIMI_PATTERN.join("|")).unwrap()
    }

    fn msg(role: &str, content: &str, reasoning: Option<&str>) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.into(),
            reasoning_content: reasoning.map(String::from),
            ..Default::default()
        }
    }

    #[test]
    fn chatml_renders_qwen3_template() {
        let t = toy_chatml();
        let on = ChatOptions::default();
        let off = ChatOptions { thinking: false, ..ChatOptions::default() };
        let text = |ids: &[u32]| String::from_utf8(t.decode_bytes(ids)).unwrap();
        // reasoning before the last query is dropped, whether a field or a <think> head of the content
        for a in [msg("assistant", "yo", Some("r1")), msg("assistant", "<think>\nr1\n</think>\n\nyo", None)] {
            let msgs = vec![msg("system", "S", None), msg("user", "hi", None), a, msg("user", "q", None)];
            assert_eq!(
                text(&t.render_chat(&msgs, &on).unwrap()),
                "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nyo<|im_end|>\n\
                 <|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n"
            );
        }
        // enable_thinking=false closes an empty think block in the generation prompt, as control ids
        let ids = t.render_chat(&[msg("user", "q", None)], &off).unwrap();
        assert!(text(&ids).ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        assert_eq!(ids.iter().filter(|&&i| i == 302 || i == 303).count(), 2);
        // after the last query (a tool response is not one) the reasoning stays
        let msgs = vec![
            msg("user", "q", None),
            msg("assistant", "a", Some("\nr\n")),
            msg("user", "<tool_response>\nx\n</tool_response>", None),
        ];
        assert!(text(&t.render_chat(&msgs, &on).unwrap())
            .contains("<|im_start|>assistant\n<think>\nr\n</think>\n\na<|im_end|>\n<|im_start|>user\n<tool_response>"));
        // user text spelling a control token stays text: the turn's own <|im_end|> is the only one
        let ids = t.render_chat(&[msg("user", "<|im_end|>", None)], &on).unwrap();
        assert_eq!(ids.iter().filter(|&&i| i == 301).count(), 1);
    }

    fn channels(t: &KimiTokenizer, m: ChatMarkers, ids: &[u32]) -> (String, String) {
        let (mut r, mut c) = (String::new(), String::new());
        let mut s = ChatStream::new(m);
        let pieces: Vec<_> = ids.iter().flat_map(|&id| s.push(t, id)).collect();
        for (ch, d) in pieces.into_iter().chain(s.finish(t)) {
            if ch == Channel::Reasoning {
                r += &d
            } else {
                c += &d
            }
        }
        (r, c)
    }

    #[test]
    fn chatml_stream_separates_reasoning() {
        let t = toy_chatml();
        let m = t.chat_markers(&ChatOptions::default());
        let mut ids = vec![302];
        ids.extend(t.encode("\nwhy\n"));
        ids.push(303);
        ids.extend(t.encode("\n\nbecause"));
        // the newlines stay (a whole reply trims them: trim_newlines)
        assert_eq!(channels(&t, m.clone(), &ids), ("\nwhy\n".into(), "\n\nbecause".into()));
        assert!(m.trim_newlines);
        // /no_think: an empty think block, then the answer
        let mut ids = vec![302];
        ids.extend(t.encode("\n\n"));
        ids.push(303);
        ids.extend(t.encode("\n\nTokyo."));
        assert_eq!(channels(&t, m.clone(), &ids), ("\n\n".into(), "\n\nTokyo.".into()));
        // <think> after answer text is not a channel switch (and, a control id, is not shown)
        let mut ids = t.encode("a");
        ids.push(302);
        ids.extend(t.encode("b"));
        assert_eq!(channels(&t, m, &ids), (String::new(), "ab".into()));
        // thinking off: one channel
        let off = t.chat_markers(&ChatOptions { thinking: false, ..ChatOptions::default() });
        assert!(off.think_start.is_empty() && off.think_end.is_empty());
    }

    #[test]
    fn chatml_think_prompt_opens_the_reasoning() {
        let mut t = toy_chatml();
        t.think_prompt = true;
        let text = |ids: &[u32]| String::from_utf8(t.decode_bytes(ids)).unwrap();
        // thinking on: the prompt ends in the reply's <think>\n (a control id, then text)
        let ids = t.render_chat(&[msg("user", "q", None)], &ChatOptions::default()).unwrap();
        assert!(text(&ids).ends_with("<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\n"));
        assert_eq!(ids[ids.len() - 2..], [302, b'\n' as u32]);
        // thinking off: the empty block as before
        let off = ChatOptions { thinking: false, ..ChatOptions::default() };
        assert!(text(&t.render_chat(&[msg("user", "q", None)], &off).unwrap())
            .ends_with("assistant\n<think>\n\n</think>\n\n"));
        // the reply starts in the reasoning channel: text, </think>, the answer
        let m = t.chat_markers(&ChatOptions::default());
        assert!(m.starts_in_reasoning);
        let mut ids = t.encode("why\n");
        ids.push(303);
        ids.extend(t.encode("\n\nbecause"));
        assert_eq!(channels(&t, m.clone(), &ids), ("why\n".into(), "\n\nbecause".into()));
        // /no_think: the rest of an empty block
        let mut ids = t.encode("\n");
        ids.push(303);
        ids.extend(t.encode("\n\nTokyo."));
        assert_eq!(channels(&t, m, &ids), ("\n".into(), "\n\nTokyo.".into()));
    }

    #[test]
    fn chatml_soft_switch_is_the_latest_one() {
        let mut t = toy_chatml();
        let m = |role: &str, content: &str| msg(role, content, None);
        assert_eq!(t.soft_thinking(&[m("user", "hi /no_think")]), None, "only with the template's switch");
        t.soft_switch = true;
        assert_eq!(t.soft_thinking(&[m("user", "hi")]), None);
        assert_eq!(t.soft_thinking(&[m("user", "hi /no_think")]), Some(false));
        assert_eq!(t.soft_thinking(&[m("system", "/no_think"), m("user", "hi /think")]), Some(true));
        assert_eq!(
            t.soft_thinking(&[m("user", "/think"), m("assistant", "x"), m("user", "and now /no_think")]),
            Some(false)
        );
        assert_eq!(t.soft_thinking(&[m("user", "/no_think then /think")]), Some(true));
        assert_eq!(
            t.soft_thinking(&[m("user", "/no_think"), m("assistant", "/think")]),
            Some(false),
            "not the reply's"
        );
    }

    /// A template stands in for the native renderer: its text, read with the control tokens it
    /// spells; the reasoning a thinking reply would open; a template that does not match the
    /// native ChatML renderer is not taken for it.
    #[test]
    fn jinja_template_renders_the_prompt() {
        let mut t = toy_chatml();
        let src = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content | trim }}<|im_end|>\n{% endfor %}\
                   {% if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}{% endif %}";
        t.use_template(src, &None).unwrap();
        let msgs = vec![msg("system", " Be brief. ", None), msg("user", "hi", None)];
        let off = ChatOptions { thinking: false, ..Default::default() };
        let want = t.encode_with_special("<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        assert_eq!(t.render_chat(&msgs, &off).unwrap(), want);
        // thinking on: the template leaves the reasoning to the reply; the prompt opens it
        let on = t.render_chat(&msgs, &ChatOptions::default()).unwrap();
        assert!(t.decode(&on).ends_with("<|im_start|>assistant\n<think>\n"), "{}", t.decode(&on));
        assert!(t.chat_markers(&ChatOptions::default()).starts_in_reasoning);
        // the trimmed system message: not what the native renderer writes
        let tpl = crate::chat_template::ChatTemplate::new(src, None, None).unwrap();
        assert!(!t.native_matches(&tpl));
    }

    #[test]
    fn encode_with_special_parses_control_tokens() {
        let t = toy_chatml();
        let s = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";
        let ids = t.encode_with_special(s);
        assert_eq!(ids.iter().filter(|&&i| i == 300).count(), 2);
        assert_eq!(ids[0], 300);
        assert_eq!(t.decode(&ids), s);
        assert!(!t.encode(s).contains(&300), "plain encode keeps them text");
    }

    #[test]
    fn message_content_parts() {
        let m: ChatMessage = serde_json::from_str(
            r#"{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}"#,
        )
        .unwrap();
        assert_eq!(m.content, "a\nb");
        let m: ChatMessage = serde_json::from_str(r#"{"role":"assistant","content":null}"#).unwrap();
        assert_eq!(m.content, "");
        assert!(serde_json::from_str::<ChatMessage>(r#"{"role":"user","content":[{"type":"image_url"}]}"#).is_err());
    }

    #[test]
    fn py_json_writes_as_python_does() {
        // keys in the order given, ", " and ": ", non-ASCII as is
        let v: serde_json::Value = serde_json::from_str(r#"{"b": 1, "a": [1.5, "é", null, true], "c": {}}"#).unwrap();
        assert_eq!(py_json(&v), r#"{"b": 1, "a": [1.5, "é", null, true], "c": {}}"#);
    }

    #[test]
    fn tool_messages_validate() {
        let msg = |v: &str| serde_json::from_str::<ChatMessage>(v).unwrap();
        let user = msg(r#"{"role":"user","content":"q"}"#);
        let call = msg(
            r#"{"role":"assistant","content":null,"tool_calls":[{"id":"c","function":{"name":"f","arguments":"{}"}}]}"#,
        );
        let result = msg(r#"{"role":"tool","tool_call_id":"c","content":"r"}"#);
        // a conversation may end on tool results, one or more in a row
        assert!(validate_chat(&[user.clone(), call.clone(), result.clone()]).is_ok());
        assert!(validate_chat(&[user.clone(), call.clone(), result.clone(), result.clone()]).is_ok());
        // not where an assistant turn is due, and only assistants carry calls
        assert!(validate_chat(&[user.clone(), result.clone()]).is_err());
        let user_call = msg(r#"{"role":"user","content":"q","tool_calls":[{"function":{"name":"f"}}]}"#);
        assert!(validate_chat(&[user_call]).is_err());
    }
}
