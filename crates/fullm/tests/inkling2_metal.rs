// SPDX-License-Identifier: Apache-2.0
//! Inkling 2 (looped) with its steps on the Mac's GPU (Metal) against the torch oracle: whole
//! prompts (the prompt-chunk kernels: tiles of 16 and 8, MPS for the values), prompts in pieces,
//! two sequences in one step, and token-by-token decoding (the single-row kernels).
//!
//! A file of its own: the switches are process environment, and the CPU executor's bit-for-bit
//! tests must not see them. One test: the two modes are loaded one after the other.
//!
//! With FULLM_INK2_METAL_F32=1 the GPU holds activations in f32 and must meet the oracle like the
//! CPU path. By default they are half (11 bits against the experts' 4): the tiny random model
//! amplifies that into whole units of logit, so that mode is only run (finite logits).
#![cfg(target_os = "macos")]

use std::path::PathBuf;

use fullm::models::{self, BackendKind, LoadOptions, Model, SeqChunk};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny_inkling2")
}

fn load() -> Box<dyn Model> {
    // the oracle's checks with an f32 cache as well as f32 activations; else the default, half
    let f32 = std::env::var("FULLM_INK2_METAL_F32").is_ok_and(|v| v == "1");
    let opts = LoadOptions {
        kv_dtype: if f32 { fullm::models::CacheDtype::F32 } else { fullm::models::CacheDtype::Auto },
        max_seqs: 2,
        cache_tokens: 256,
        page: 4,
        backend: BackendKind::CPU,
        step_tokens: 64,
        max_rows: 64,
        max_context: 128,
        threads: 4,
        ..Default::default()
    };
    models::load(&fixture(), &opts).expect("load the tiny checkpoint")
}

/// A prompt, its greedy continuation, the oracle's logits at every prompt position.
type Case = (Vec<u32>, Vec<u32>, Vec<Vec<f32>>);

fn argmax(v: &[f32]) -> u32 {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b }) as u32
}

fn worst(got: &[f32], want: &[Vec<f32>], vocab: usize) -> f32 {
    let mut d = 0.0f32;
    for (i, w) in want.iter().enumerate() {
        for (a, b) in got[i * vocab..(i + 1) * vocab].iter().zip(w) {
            d = d.max((a - b).abs());
        }
    }
    d
}

fn check(tol: f32, exact_text: bool) {
    let raw = std::fs::read_to_string(fixture().join("oracle.json")).expect("oracle.json");
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let ids =
        |x: &serde_json::Value| x.as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect::<Vec<_>>();
    let mut m = load();
    if !m.describe().contains("Metal") && std::env::var("FULLM_INK2_METAL").is_err() {
        eprintln!("no Metal device: nothing to check ({})", m.describe());
        return;
    }
    let vocab = m.vocab();
    let cases: Vec<Case> = v["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let want = c["tf_logits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect())
                .collect();
            (ids(&c["prompt"]), ids(&c["greedy"]), want)
        })
        .collect();
    for (prompt, greedy, want) in &cases {
        // the whole prompt in one step
        let s = m.alloc_seq().unwrap();
        let out =
            m.forward(&[SeqChunk { seq: s, pos: 0, tokens: prompt, all_logits: true, draft: 0 }]).unwrap().remove(0);
        let d = worst(&out, want, vocab);
        assert!(d.is_finite() && d < tol, "whole prompt: max |logit diff| {d}");
        // then decoded token by token
        let (mut pos, mut next, mut got) = (prompt.len(), argmax(&out[(prompt.len() - 1) * vocab..]), Vec::new());
        for _ in 0..greedy.len() {
            got.push(next);
            let l =
                m.forward(&[SeqChunk { seq: s, pos, tokens: &[next], all_logits: false, draft: 0 }]).unwrap().remove(0);
            pos += 1;
            next = argmax(&l);
        }
        if exact_text {
            assert_eq!(&got, greedy);
        }
        m.free_seq(s);
        // the prompt in pieces (1, 2, the rest): the histories and the cache carry over
        let s = m.alloc_seq().unwrap();
        let (mut all, mut at) = (Vec::new(), 0);
        for n in [1, 2, prompt.len() - 3] {
            let o = m
                .forward(&[SeqChunk { seq: s, pos: at, tokens: &prompt[at..at + n], all_logits: true, draft: 0 }])
                .unwrap()
                .remove(0);
            all.extend(o);
            at += n;
        }
        let d = worst(&all, want, vocab);
        assert!(d < tol, "prompt in pieces: max |logit diff| {d}");
        m.free_seq(s);
    }
    // drafts, right and wrong, leave the sequence as plain decoding does (the histories go back)
    {
        let (prompt, g, _) = &cases[0];
        let wrong = |t: u32| (t + 1) % vocab as u32;
        let s = m.alloc_seq().unwrap();
        m.forward(&[SeqChunk { seq: s, pos: 0, tokens: prompt, all_logits: false, draft: 0 }]).unwrap();
        let plain: Vec<Vec<f32>> = (0..6)
            .map(|i| {
                m.forward(&[SeqChunk {
                    seq: s,
                    pos: prompt.len() + i,
                    tokens: &g[i..i + 1],
                    all_logits: false,
                    draft: 0,
                }])
                .unwrap()
                .remove(0)
            })
            .collect();
        m.free_seq(s);
        let s = m.alloc_seq().unwrap();
        m.forward(&[SeqChunk { seq: s, pos: 0, tokens: prompt, all_logits: false, draft: 0 }]).unwrap();
        let steps: Vec<(Vec<u32>, usize, usize)> = vec![
            (vec![g[0], wrong(g[1]), wrong(g[2])], 2, 0),
            (vec![g[1], g[2], wrong(g[3])], 2, 1),
            (vec![g[3], g[4], g[5]], 2, 2),
        ];
        let mut at = 0;
        for (toks, draft, ok) in steps {
            let out = m
                .forward(&[SeqChunk { seq: s, pos: prompt.len() + at, tokens: &toks, all_logits: true, draft }])
                .unwrap()
                .remove(0);
            m.accept(s, ok).unwrap();
            for i in 0..=ok {
                let d = out[i * vocab..(i + 1) * vocab]
                    .iter()
                    .zip(&plain[at + i])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max);
                assert!(d.is_finite() && d < tol, "after drafts, token {}: max |logit diff| {d}", at + i);
            }
            at += 1 + ok;
        }
        m.free_seq(s);
    }
    // the draft head (a random one in the fixture: its drafts are wrong, which is the point): after
    // every step it proposes tokens of the vocabulary, the same ones when asked again, and a
    // decode driven by them, kept where the model agrees, gives the plain decode's logits
    {
        use fullm::models::DraftReq;
        assert!(m.draft_head(), "the fixture's draft head was loaded");
        let (prompt, g, _) = &cases[0];
        let s = m.alloc_seq().unwrap();
        m.forward(&[SeqChunk { seq: s, pos: 0, tokens: prompt, all_logits: false, draft: 0 }]).unwrap();
        let plain: Vec<Vec<f32>> = (0..8)
            .map(|i| {
                m.forward(&[SeqChunk {
                    seq: s,
                    pos: prompt.len() + i,
                    tokens: &g[i..i + 1],
                    all_logits: false,
                    draft: 0,
                }])
                .unwrap()
                .remove(0)
            })
            .collect();
        m.free_seq(s);
        m.flush_prefix_cache();
        let s = m.alloc_seq().unwrap();
        m.forward(&[SeqChunk { seq: s, pos: 0, tokens: prompt, all_logits: false, draft: 0 }]).unwrap();
        let mut at = 0;
        while at < 8 {
            let req = [DraftReq { seq: s, next: g[at], k: 3 }];
            let d = m.draft(&req).unwrap().remove(0);
            assert_eq!(d.len(), 3, "three drafts asked");
            assert!(d.iter().all(|&t| (t as usize) < vocab), "drafts {d:?} outside the vocabulary");
            assert_eq!(m.draft(&req).unwrap().remove(0), d, "the same drafts when asked again");
            let mut toks = vec![g[at]];
            toks.extend(&d);
            let out = m
                .forward(&[SeqChunk { seq: s, pos: prompt.len() + at, tokens: &toks, all_logits: true, draft: 3 }])
                .unwrap()
                .remove(0);
            let ok = (0..3).take_while(|&i| at + 1 + i < 8 && d[i] == g[at + 1 + i]).count();
            m.accept(s, ok).unwrap();
            for i in 0..=ok {
                let e = out[i * vocab..(i + 1) * vocab]
                    .iter()
                    .zip(&plain[at + i])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max);
                assert!(e.is_finite() && e < tol, "with the head's drafts, token {}: max |logit diff| {e}", at + i);
            }
            at += 1 + ok;
        }
        m.free_seq(s);
        m.flush_prefix_cache();
    }
    // a sequence started from the prefix cache (the histories kept by page) continues like the first
    {
        let (prompt, g, _) = &cases[0];
        let text: Vec<u32> = prompt.iter().chain(&g[..9]).copied().collect();
        m.flush_prefix_cache();
        let s = m.alloc_seq().unwrap();
        let plain =
            m.forward(&[SeqChunk { seq: s, pos: 0, tokens: &text, all_logits: true, draft: 0 }]).unwrap().remove(0);
        m.free_seq(s);
        let (s, have) = m.alloc_seq_cached(&text).unwrap();
        assert_eq!(have, (text.len() - 1) / 4 * 4);
        let out = m
            .forward(&[SeqChunk { seq: s, pos: have, tokens: &text[have..], all_logits: true, draft: 0 }])
            .unwrap()
            .remove(0);
        let d = out.iter().zip(&plain[have * vocab..]).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(d.is_finite() && d < tol, "after a cache hit of {have}: max |logit diff| {d}");
        m.free_seq(s);
        m.flush_prefix_cache();
    }
    // both prompts in one step
    let (a, b) = (m.alloc_seq().unwrap(), m.alloc_seq().unwrap());
    let outs = m
        .forward(&[
            SeqChunk { seq: a, pos: 0, tokens: &cases[0].0, all_logits: true, draft: 0 },
            SeqChunk { seq: b, pos: 0, tokens: &cases[1].0, all_logits: true, draft: 0 },
        ])
        .unwrap();
    for (o, c) in outs.iter().zip(&cases) {
        let d = worst(o, &c.2, vocab);
        assert!(d < tol, "two sequences in a step: max |logit diff| {d}");
    }
}

#[test]
fn the_gpu_steps_match_the_oracle() {
    // every draft the head proposes, whatever it thinks of it (in serving: only chains it trusts)
    std::env::set_var("FULLM_INK2_DRAFT_CONF", "0");
    std::env::set_var("FULLM_INK2_METAL_F32", "1");
    check(2e-3, true);
    // the values of runs of 4 rows or more by MPS (in serving: from 32)
    std::env::set_var("FULLM_INK2_METAL_PV_MIN", "4");
    check(2e-3, true);
    // half activations: the tiny model's logits move by whole units (see above), so only that
    // every path runs and stays finite; the real model's text is checked against the CPU path
    std::env::set_var("FULLM_INK2_METAL_F32", "0");
    check(f32::MAX, false);
}
