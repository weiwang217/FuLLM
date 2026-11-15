// SPDX-License-Identifier: Apache-2.0
//! Structured output on llguidance: grammars (JSON schema, regular expressions, Lark) over the
//! model's tokenizer, as the engine's `Constraint`. Each sampled position draws only among the
//! matcher's allowed tokens (a bitmask the engine hands the device's sampler); the matcher then
//! takes the token drawn. With a reasoning channel left open by the prompt (Qwen3 thinking), the
//! grammar starts after its closing id.

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{TokEnv, TokRxInfo, TokTrie, TokenizerEnv};
use llguidance::{Matcher, ParserFactory};

use crate::runtime::Constraint;
use crate::tokenizer::Tokenizer;

/// The tokenizer as llguidance reads it: every id's bytes (control tokens, and ids the
/// tokenizer has no entry for, as llguidance's special tokens, which no text matches).
struct Env {
    trie: TokTrie,
    tok: Arc<dyn Tokenizer>,
}

impl TokenizerEnv for Env {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }
    fn tokenize_bytes(&self, s: &[u8]) -> Vec<u32> {
        self.tok.encode(&String::from_utf8_lossy(s))
    }
}

/// The grammar compiler of one model: built once (a trie over the vocabulary), then a matcher
/// per request.
pub struct Grammars {
    factory: ParserFactory,
    eos: u32,
}

impl Grammars {
    /// `vocab`: the model's logits width (it may exceed the tokenizer's ids). The end of a
    /// grammar is the tokenizer's first stop id (Qwen3: `<|im_end|>`).
    pub fn new(tok: Arc<dyn Tokenizer>, vocab: usize) -> Result<Self> {
        let eos = *tok.stop_ids().first().context("the tokenizer has no stop id")?;
        let words: Vec<Vec<u8>> = (0..vocab as u32)
            .map(|id| {
                let b = tok.decode_bytes(&[id]);
                if tok.is_special(id) || b.is_empty() {
                    let name =
                        if b.is_empty() { format!("<[{id}]>") } else { String::from_utf8_lossy(&b).into_owned() };
                    [vec![TokTrie::SPECIAL_TOKEN_MARKER], name.into_bytes()].concat()
                } else {
                    b
                }
            })
            .collect();
        let mut info = TokRxInfo::new(vocab as u32, eos);
        info.tok_end_of_turn = Some(eos);
        let env: TokEnv = Arc::new(Env { trie: TokTrie::from(&info, &words), tok });
        Ok(Grammars { factory: ParserFactory::new_simple(&env)?, eos })
    }

    /// A request's constraint for `g`; `after`: the id that closes an open reasoning channel
    /// (the grammar holds from the token after it), or None (from the first token).
    pub fn constraint(&self, g: TopLevelGrammar, after: Option<u32>) -> Result<Box<dyn Constraint>> {
        let m = Matcher::new(self.factory.create_parser(g));
        if let Some(e) = m.get_error() {
            bail!("{e}");
        }
        Ok(Box::new(Guided { m, active: after.is_none(), after, eos: self.eos }))
    }
}

struct Guided {
    m: Matcher,
    active: bool,
    after: Option<u32>,
    eos: u32,
}

impl Constraint for Guided {
    fn allowed(&mut self, words: &mut [u32]) -> bool {
        if !self.active {
            return false;
        }
        // the matcher failed (it should not), or allows nothing: the end id alone ends the reply
        if let Ok(v) = self.m.compute_mask() {
            let s = v.as_slice();
            let n = s.len().min(words.len());
            words[..n].copy_from_slice(&s[..n]);
        }
        if words.iter().all(|&w| w == 0) {
            words[self.eos as usize / 32] |= 1 << (self.eos % 32);
        }
        true
    }

    fn advance(&mut self, tok: u32) {
        if self.active {
            let _ = self.m.consume_token(tok);
        } else if Some(tok) == self.after {
            self.active = true;
        }
    }

    fn done(&self) -> bool {
        self.active && (self.m.is_stopped() || self.m.is_error())
    }
}

/// JSON's whitespace between tokens, bounded: up to 8 spaces in a line, or a newline and up to 32
/// spaces of indent (no tabs, no blank lines). Pretty-printing stays possible; llguidance's default
/// (any run of whitespace) let Qwen3-1.7B write newlines after its last value until max_tokens, and
/// a looser bound still let it pad with tabs before the closing brace.
const JSON_WHITESPACE: &str = r"[\x20]{1,8}|\x0A[\x20]{0,32}";

/// `schema` with the bounded whitespace, unless it sets llguidance's options itself (`x-guidance`).
pub fn bounded_whitespace(mut schema: serde_json::Value) -> serde_json::Value {
    if let Some(m) = schema.as_object_mut() {
        m.entry("x-guidance").or_insert_with(|| serde_json::json!({"whitespace_pattern": JSON_WHITESPACE}));
    }
    schema
}

/// `text` as a regular expression that matches it literally.
pub fn regex_literal(text: &str) -> String {
    let mut s = String::with_capacity(text.len());
    for c in text.chars() {
        if "\\.+*?()|[]{}^$#&-~".contains(c) {
            s.push('\\');
        }
        s.push(c);
    }
    s
}

/// The grammar of a reply that is function calls only (`tool_choice` required: one or more of
/// any tool; a named function: one call of it), in Qwen3's form: `<tool_call>` (id `open`),
/// `{"name": ..., "arguments": ...}` with the function's parameter schema, `</tool_call>` (id
/// `close`), calls newline-separated; leading whitespace allowed.
pub fn tool_calls_grammar(tools: &[serde_json::Value], only: Option<&str>, open: u32, close: u32) -> TopLevelGrammar {
    let call = |t: &serde_json::Value| {
        let f = &t["function"];
        let params =
            f.get("parameters").filter(|p| p.is_object()).cloned().unwrap_or(serde_json::json!({"type": "object"}));
        serde_json::json!({
            "type": "object",
            "properties": {"name": {"const": f["name"]}, "arguments": params},
            "required": ["name", "arguments"],
            "additionalProperties": false,
        })
    };
    let picked: Vec<serde_json::Value> =
        tools.iter().filter(|t| only.is_none_or(|n| t["function"]["name"].as_str() == Some(n))).map(call).collect();
    let schema =
        bounded_whitespace(if picked.len() == 1 { picked[0].clone() } else { serde_json::json!({"anyOf": picked}) });
    let more = if only.is_some() { "" } else { " (\"\\n\" call)*" };
    TopLevelGrammar::from_lark(format!(
        "start: ws call{more}\nws: /[ \\n]{{0,3}}/\ncall: <[{open}]> \"\\n\" body \"\\n\" <[{close}]>\nbody: %json {schema}\n"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{ChatMarkers, ChatMessage, ChatOptions};

    /// ASCII bytes 0..127 as tokens, `<eos>` (128) and `</think>` (129) as control tokens.
    struct Ascii;
    impl Tokenizer for Ascii {
        fn encode(&self, text: &str) -> Vec<u32> {
            text.bytes().map(u32::from).collect()
        }
        fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
            ids.iter()
                .flat_map(|&i| match i {
                    128 => b"<eos>".to_vec(),
                    129 => b"</think>".to_vec(),
                    i => vec![i as u8],
                })
                .collect()
        }
        fn is_special(&self, id: u32) -> bool {
            id >= 128
        }
        fn render_chat(&self, _: &[ChatMessage], _: &ChatOptions) -> anyhow::Result<Vec<u32>> {
            unimplemented!()
        }
        fn chat_markers(&self, _: &ChatOptions) -> ChatMarkers {
            ChatMarkers::default()
        }
        fn stop_ids(&self) -> Vec<u32> {
            vec![128]
        }
        fn vocab_size(&self) -> usize {
            130
        }
    }

    fn allowed(c: &mut Box<dyn Constraint>) -> Vec<u32> {
        let mut w = vec![0u32; 130usize.div_ceil(32)];
        if !c.allowed(&mut w) {
            return (0..130).collect();
        }
        (0..130).filter(|&i| w[i as usize / 32] >> (i % 32) & 1 == 1).collect()
    }

    #[test]
    fn regex_masks_step_by_step() {
        let g = Grammars::new(Arc::new(Ascii), 130).unwrap();
        let mut c = g.constraint(TopLevelGrammar::from_regex("ab[cd]"), None).unwrap();
        assert_eq!(allowed(&mut c), vec![b'a' as u32]);
        c.advance(b'a' as u32);
        assert_eq!(allowed(&mut c), vec![b'b' as u32]);
        c.advance(b'b' as u32);
        assert_eq!(allowed(&mut c), vec![b'c' as u32, b'd' as u32]);
        c.advance(b'd' as u32);
        // complete, and nothing may follow: the request ends here, without a step for the end id
        assert!(c.done());
        assert_eq!(allowed(&mut c), vec![128]);
    }

    #[test]
    fn grammar_starts_after_the_reasoning() {
        let g = Grammars::new(Arc::new(Ascii), 130).unwrap();
        let mut c = g.constraint(TopLevelGrammar::from_regex("x"), Some(129)).unwrap();
        assert_eq!(allowed(&mut c).len(), 130, "free while reasoning");
        c.advance(b'q' as u32);
        assert_eq!(allowed(&mut c).len(), 130);
        c.advance(129);
        assert_eq!(allowed(&mut c), vec![b'x' as u32]);
    }

    #[test]
    fn json_schema_and_tool_calls_compile() {
        let g = Grammars::new(Arc::new(Ascii), 130).unwrap();
        let schema = serde_json::json!({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]});
        let mut c = g.constraint(TopLevelGrammar::from_json_schema(bounded_whitespace(schema)), None).unwrap();
        assert!(allowed(&mut c).contains(&(b'{' as u32)));
        let tools =
            vec![serde_json::json!({"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}})];
        // ASCII has no call tokens: any two ids stand in
        assert!(g.constraint(tool_calls_grammar(&tools, Some("f"), 126, 127), None).is_ok());
        assert!(g.constraint(TopLevelGrammar::from_json_schema(serde_json::json!({"type": "frob"})), None).is_err());
        assert_eq!(regex_literal("a.b*(c)"), r"a\.b\*\(c\)");
    }
}
