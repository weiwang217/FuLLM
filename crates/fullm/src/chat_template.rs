// SPDX-License-Identifier: Apache-2.0
//! Chat templates in Jinja, rendered as transformers' `apply_chat_template` renders them (and vLLM,
//! which calls it): `trim_blocks` and `lstrip_blocks` on, a trailing newline dropped, Python's
//! string / dict / list methods (minijinja-contrib's pycompat), `tojson` as Python's `json.dumps`
//! with `ensure_ascii=False` (minijinja's own escapes `<` and `>`, which templates put in prompts
//! verbatim), `raise_exception`, `strftime_now`, and `{% generation %}` blocks (transformers'
//! assistant-mask extension) as their plain content. The context: `messages` (OpenAI messages, a
//! call's `arguments` parsed into an object as vLLM does), `tools`, `add_generation_prompt`,
//! `bos_token`, `eos_token`, and the request's `chat_template_kwargs`.

use anyhow::{Context, Result};
use minijinja::syntax::SyntaxConfig;
use minijinja::value::Kwargs;
use minijinja::{Environment, Error, ErrorKind, Value as JValue};
use serde_json::{Map, Value};

pub struct ChatTemplate {
    env: Environment<'static>,
    bos: Option<String>,
    eos: Option<String>,
}

impl ChatTemplate {
    /// `bos` / `eos`: the tokenizer's token strings, as the template sees them.
    pub fn new(source: &str, bos: Option<String>, eos: Option<String>) -> Result<Self> {
        let mut env = Environment::new();
        let mut syntax = SyntaxConfig::builder();
        syntax.trim_blocks(true).lstrip_blocks(true);
        env.set_syntax(syntax.build().context("chat template syntax")?);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson);
        env.add_function("raise_exception", |msg: String| -> Result<JValue, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_function("strftime_now", |fmt: String| chrono::Local::now().format(&fmt).to_string());
        // transformers' `{% generation %}` ... `{% endgeneration %}` only marks the assistant's spans
        // for training masks; it renders its body unchanged, so the tags are dropped
        let source = strip_generation_tags(source);
        env.add_template_owned("chat", source).context("parsing the chat template")?;
        Ok(ChatTemplate { env, bos, eos })
    }

    /// The prompt text for `messages` (OpenAI messages, as JSON), `tools` (OpenAI function
    /// definitions; None when the request has none) and the template's extra arguments.
    pub fn render(
        &self,
        messages: &Value,
        tools: Option<&Value>,
        add_generation_prompt: bool,
        kwargs: &Map<String, Value>,
    ) -> Result<String> {
        let mut ctx = Map::new();
        for (k, v) in kwargs {
            ctx.insert(k.clone(), v.clone());
        }
        ctx.insert("messages".into(), messages.clone());
        ctx.insert("tools".into(), tools.cloned().unwrap_or(Value::Null));
        ctx.insert("add_generation_prompt".into(), Value::Bool(add_generation_prompt));
        ctx.insert("bos_token".into(), self.bos.clone().map_or(Value::String(String::new()), Value::String));
        ctx.insert("eos_token".into(), self.eos.clone().map_or(Value::String(String::new()), Value::String));
        let t = self.env.get_template("chat").expect("added at construction");
        t.render(JValue::from(minijinja::value::Serde(&Value::Object(ctx))))
            .map_err(|e| anyhow::anyhow!("chat template: {e:#}"))
    }
}

/// transformers' `tojson`: `json.dumps(x, ensure_ascii=False, indent=indent,
/// separators=separators, sort_keys=sort_keys)`.
fn tojson(v: JValue, kwargs: Kwargs) -> Result<String, Error> {
    let indent: Option<JValue> = kwargs.get("indent")?;
    let separators: Option<Vec<String>> = kwargs.get("separators")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    let _: Option<bool> = kwargs.get("ensure_ascii")?;
    kwargs.assert_all_used()?;
    let j: Value =
        serde_json::to_value(&v).map_err(|e| Error::new(ErrorKind::InvalidOperation, format!("tojson: {e}")))?;
    let indent = match indent {
        None => None,
        Some(i) if i.is_none() => None,
        Some(i) => match i.as_i64() {
            Some(n) => Some(" ".repeat(n.max(0) as usize)),
            None => Some(i.as_str().unwrap_or_default().to_string()),
        },
    };
    let (item, key) = match &separators {
        Some(s) if s.len() == 2 => (s[0].clone(), s[1].clone()),
        _ if indent.is_some() => (",".into(), ": ".into()),
        _ => (", ".into(), ": ".into()),
    };
    Ok(py_dumps(&j, indent.as_deref(), &item, &key, sort_keys.unwrap_or(false)))
}

/// Python's `json.dumps` with `ensure_ascii=False`: `indent` (None: one line), the item and key
/// separators, keys in the order given (or sorted).
pub fn py_dumps(v: &Value, indent: Option<&str>, item: &str, key: &str, sort: bool) -> String {
    let mut out = String::new();
    dump(v, indent, item, key, sort, 0, &mut out);
    out
}

fn dump(v: &Value, indent: Option<&str>, item: &str, key: &str, sort: bool, level: usize, out: &mut String) {
    let newline = |out: &mut String, level: usize| {
        if let Some(i) = indent {
            out.push('\n');
            out.push_str(&i.repeat(level));
        }
    };
    match v {
        Value::Object(m) if !m.is_empty() => {
            let mut entries: Vec<(&String, &Value)> = m.iter().collect();
            if sort {
                entries.sort_by(|a, b| a.0.cmp(b.0));
            }
            out.push('{');
            for (n, (k, x)) in entries.into_iter().enumerate() {
                if n > 0 {
                    out.push_str(item);
                }
                newline(out, level + 1);
                out.push_str(&Value::String(k.clone()).to_string());
                out.push_str(key);
                dump(x, indent, item, key, sort, level + 1, out);
            }
            newline(out, level);
            out.push('}');
        }
        Value::Array(a) if !a.is_empty() => {
            out.push('[');
            for (n, x) in a.iter().enumerate() {
                if n > 0 {
                    out.push_str(item);
                }
                newline(out, level + 1);
                dump(x, indent, item, key, sort, level + 1, out);
            }
            newline(out, level);
            out.push(']');
        }
        Value::Object(_) => out.push_str("{}"),
        Value::Array(_) => out.push_str("[]"),
        other => out.push_str(&other.to_string()),
    }
}

/// A message's `tool_calls` arguments as the template expects them: the JSON object a client's
/// string holds (vLLM parses them before rendering), else as given.
pub fn arguments_value(a: &Value) -> Value {
    match a {
        Value::String(s) => match serde_json::from_str::<Value>(s) {
            Ok(v @ Value::Object(_)) => v,
            _ => a.clone(),
        },
        Value::Null => Value::Object(Map::new()),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn render(src: &str, msgs: Value, kwargs: Value) -> String {
        let t = ChatTemplate::new(src, Some("<s>".into()), Some("</s>".into())).unwrap();
        t.render(&msgs, None, true, kwargs.as_object().unwrap()).unwrap()
    }

    #[test]
    fn whitespace_and_python_methods_as_transformers() {
        // trim_blocks / lstrip_blocks: block tags leave no blank lines; the trailing newline goes
        let src = "{{ bos_token }}\n{% for m in messages %}\n  {% if m.role == 'user' %}\n[U]{{ m.content.strip() }}\n  {% else %}\n[A]{{ m.content.split('</think>')[-1].lstrip('\\n') }}\n  {% endif %}\n{% endfor %}\n{% if add_generation_prompt %}[A]{% endif %}\n";
        let msgs = json!([{"role": "user", "content": "  hi "}, {"role": "assistant", "content": "x</think>\n\nyo"}]);
        assert_eq!(render(src, msgs, json!({})), "<s>\n[U]hi\n[A]yo\n[A]");
    }

    #[test]
    fn kwargs_namespace_and_exceptions() {
        let src = "{% set ns = namespace(n=0) %}{% for m in messages %}{% set ns.n = ns.n + 1 %}{% endfor %}{{ ns.n }}{% if enable_thinking is defined and enable_thinking is false %}-off{% endif %}";
        assert_eq!(
            render(src, json!([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]), json!({})),
            "2"
        );
        assert_eq!(render(src, json!([]), json!({"enable_thinking": false})), "0-off");
        let t = ChatTemplate::new("{{ raise_exception('no system') }}", None, None).unwrap();
        let e = t.render(&json!([]), None, true, &Map::new()).unwrap_err();
        assert!(format!("{e:#}").contains("no system"), "{e:#}");
    }

    #[test]
    fn tojson_is_pythons_json_dumps() {
        let v = json!({"name": "f", "description": "a < b & \"c\"", "parameters": {"type": "object", "properties": {}, "x": [1, 2.5, null]}});
        let src = "{{ messages[0] | tojson }}|{{ messages[0] | tojson(indent=2) }}";
        let out = render(src, json!([v]), json!({}));
        let (one, ind) = out.split_once('|').unwrap();
        assert_eq!(
            one,
            r#"{"name": "f", "description": "a < b & \"c\"", "parameters": {"type": "object", "properties": {}, "x": [1, 2.5, null]}}"#
        );
        assert_eq!(
            ind,
            "{\n  \"name\": \"f\",\n  \"description\": \"a < b & \\\"c\\\"\",\n  \"parameters\": {\n    \"type\": \"object\",\n    \"properties\": {},\n    \"x\": [\n      1,\n      2.5,\n      null\n    ]\n  }\n}"
        );
        assert_eq!(py_dumps(&json!({"b": 1, "a": "é"}), None, ",", ":", true), r#"{"a":"é","b":1}"#);
    }

    #[test]
    fn generation_blocks_are_their_content() {
        let src = "{% for m in messages %}{% generation %}{{ m.content }}{% endgeneration %}{% endfor %}";
        assert_eq!(render(src, json!([{"role": "assistant", "content": "ok"}]), json!({})), "ok");
    }
}

/// `source` without transformers' generation tags (`{%- generation -%}`, `{% endgeneration %}`,
/// any whitespace control): each replaced by an empty statement with the same whitespace control.
fn strip_generation_tags(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(i) = rest.find("{%") {
        let Some(j) = rest[i..].find("%}") else { break };
        let tag = &rest[i..i + j + 2];
        let inner = tag[2..tag.len() - 2].trim_matches(|c: char| c == '-' || c == '+' || c.is_whitespace());
        out.push_str(&rest[..i]);
        if inner == "generation" || inner == "endgeneration" {
            // keep the whitespace trimming the tag asked for
            let (l, r) = (tag[2..].starts_with('-'), tag[..tag.len() - 2].ends_with('-'));
            out.push_str(&format!(
                "{{%{} if true {}%}}{{%{} endif {}%}}",
                if l { "-" } else { "" },
                "-",
                "-",
                if r { "-" } else { "" }
            ));
        } else {
            out.push_str(tag);
        }
        rest = &rest[i + j + 2..];
    }
    out.push_str(rest);
    out
}
