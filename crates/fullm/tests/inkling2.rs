// SPDX-License-Identifier: Apache-2.0
//! Inkling 2 (looped; tiny, released layout with NVFP4 and MXFP8 experts) against the torch oracle
//! from tools/make_tiny_inkling2.py, itself checked against the transformers implementation
//! (tools/check_inkling2_ref.py), on the Rust reference and on the C++ executor (CPU backend: the
//! experts read in place in the release's formats). The oracle is float64; the tiny model's
//! logits move ~4e-4 between float32 and float64 arithmetic, which sets the tolerance.

use std::path::PathBuf;

use fullm::models::{self, BackendKind, LoadOptions, Model, SeqChunk};

struct Case {
    prompt: Vec<u32>,
    tf_logits: Vec<Vec<f32>>,
    greedy: Vec<u32>,
}

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny_inkling2")
}

fn cases() -> Vec<Case> {
    let raw = std::fs::read_to_string(fixture().join("oracle.json")).expect("oracle.json");
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let ids =
        |x: &serde_json::Value| x.as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect::<Vec<_>>();
    v["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| Case {
            prompt: ids(&c["prompt"]),
            tf_logits: c["tf_logits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect())
                .collect(),
            greedy: ids(&c["greedy"]),
        })
        .collect()
}

fn backends() -> Vec<BackendKind> {
    vec![BackendKind::Reference, BackendKind::CPU]
}

fn load(backend: BackendKind) -> Box<dyn Model> {
    // the oracle checks run the experts' exact path (rows decoded to floats): the tiny random model
    // amplifies a 1e-5 change in an activation several thousand times, more than the integer
    // path's tolerance allows (that path is checked on its own below)
    std::env::set_var("FULLM_INK2_SDOT", "0");
    // the CPU path: on a Mac the steps otherwise run on the GPU (tests/inkling2_metal.rs)
    std::env::set_var("FULLM_INK2_METAL", "0");
    let opts = LoadOptions {
        max_seqs: 4,
        cache_tokens: 512,
        page: 4,
        backend,
        step_tokens: 64,
        max_rows: 64,
        max_context: 256,
        threads: 4,
        ..Default::default()
    };
    models::load(&fixture(), &opts).expect("load the tiny checkpoint")
}

fn argmax(v: &[f32]) -> u32 {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b }) as u32
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn chunk<'a>(seq: usize, pos: usize, tokens: &'a [u32], all: bool) -> SeqChunk<'a> {
    SeqChunk { seq, pos, tokens, all_logits: all, draft: 0 }
}

const TOL: f32 = 2e-3;

#[test]
fn teacher_forcing_matches_the_oracle() {
    for b in backends() {
        let mut m = load(b.clone());
        assert_eq!(m.arch(), "inkling2_loop");
        for c in cases() {
            let s = m.alloc_seq().unwrap();
            let out = m.forward(&[chunk(s, 0, &c.prompt, true)]).unwrap();
            let v = m.vocab();
            assert_eq!(out[0].len(), c.prompt.len() * v);
            for (i, want) in c.tf_logits.iter().enumerate() {
                let got = &out[0][i * v..(i + 1) * v];
                let d = max_diff(got, want);
                assert!(d < TOL, "{b:?}: prompt of {}, position {i}: max |logit diff| {d}", c.prompt.len());
                assert_eq!(argmax(got), argmax(want), "{b:?}: position {i}");
            }
            m.free_seq(s);
        }
    }
}

/// The prompt fed in pieces of every size, then decoded token by token: the oracle's greedy
/// continuation, and the same logits as the whole prompt at once (the caches, the convolution
/// states and the position-dependent pieces all carry over).
#[test]
fn chunked_prefill_and_greedy_decoding_match() {
    for b in backends() {
        let mut m = load(b.clone());
        for c in cases() {
            for size in [1, 2, 3, 5, c.prompt.len()] {
                let s = m.alloc_seq().unwrap();
                let (mut pos, mut last) = (0, Vec::new());
                for part in c.prompt.chunks(size) {
                    last = m.forward(&[chunk(s, pos, part, false)]).unwrap().remove(0);
                    pos += part.len();
                }
                let d = max_diff(&last, c.tf_logits.last().unwrap());
                assert!(d < TOL, "{b:?}: chunks of {size}: last prompt row off by {d}");
                let mut got = Vec::new();
                let mut next = argmax(&last);
                for _ in 0..c.greedy.len() {
                    got.push(next);
                    let l = m.forward(&[chunk(s, pos, &[next], false)]).unwrap().remove(0);
                    pos += 1;
                    next = argmax(&l);
                }
                assert_eq!(got, c.greedy, "{b:?}: chunks of {size}");
                m.free_seq(s);
            }
        }
    }
}

/// Two sequences in one step, at different lengths, equal each alone, bit for bit.
#[test]
fn a_batch_equals_its_sequences_alone() {
    let cs = cases();
    for b in backends() {
        let mut m = load(b.clone());
        let alone: Vec<Vec<f32>> = cs
            .iter()
            .map(|c| {
                let s = m.alloc_seq().unwrap();
                let o = m.forward(&[chunk(s, 0, &c.prompt, true)]).unwrap().remove(0);
                m.free_seq(s);
                o
            })
            .collect();
        let (sa, sb) = (m.alloc_seq().unwrap(), m.alloc_seq().unwrap());
        let (pa, pb) = (&cs[0].prompt, &cs[1].prompt);
        let first = m.forward(&[chunk(sa, 0, &pa[..5], true), chunk(sb, 0, &pb[..], true)]).unwrap();
        let rest = m.forward(&[chunk(sa, 5, &pa[5..], true)]).unwrap();
        let got_a: Vec<f32> = first[0].iter().chain(rest[0].iter()).copied().collect();
        assert!(got_a == alone[0], "{b:?}: sequence a differs in a batch");
        assert!(first[1] == alone[1], "{b:?}: sequence b differs in a batch");
    }
}

/// A slot reused after another sequence starts clean (its convolution histories reset, its
/// pages another's before), round after round.
#[test]
fn a_reused_slot_starts_clean() {
    let cs = cases();
    for b in backends() {
        let mut m = load(b.clone());
        let v = m.vocab();
        for round in 0..3 {
            for c in &cs {
                let s = m.alloc_seq().unwrap();
                let o = m.forward(&[chunk(s, 0, &c.prompt, true)]).unwrap();
                let d = max_diff(&o[0][(c.prompt.len() - 1) * v..], c.tf_logits.last().unwrap());
                assert!(d < TOL, "{b:?}: round {round}: off by {d}");
                m.free_seq(s);
            }
        }
    }
}

#[test]
fn refuses_what_it_cannot_do() {
    for b in backends() {
        let mut m = load(b.clone());
        let s = m.alloc_seq().unwrap();
        let bad = |m: &mut Box<dyn Model>, ch: SeqChunk<'_>| m.forward(&[ch]).is_err();
        assert!(bad(&mut m, chunk(s, 1, &[3], false)), "{b:?}: a gap");
        assert!(bad(&mut m, chunk(s, 0, &[300], false)), "{b:?}: token out of range");
        // the reference cannot roll its convolution states back; the executor can (below)
        assert_eq!(m.speculation(), b != BackendKind::Reference, "{b:?}");
        if b == BackendKind::Reference {
            assert!(bad(&mut m, SeqChunk { seq: s, pos: 0, tokens: &[3, 4], all_logits: true, draft: 1 }), "drafts");
        }
        // a failed step changed nothing
        assert!(m.forward(&[chunk(s, 0, &[3], false)]).is_ok(), "{b:?}");
    }
}

/// A prompt split at any point gives the whole prompt's logits bit for bit on the executor (every
/// dot is one routine whatever the step's row count).
#[test]
fn a_split_prompt_is_bit_identical_on_the_executor() {
    let cs = cases();
    let mut m = load(BackendKind::CPU);
    let p = &cs[0].prompt;
    let s = m.alloc_seq().unwrap();
    let whole = m.forward(&[chunk(s, 0, p, true)]).unwrap().remove(0);
    m.free_seq(s);
    for split in [1usize, 2, 3, 4, 5, 8, 12] {
        let s = m.alloc_seq().unwrap();
        let a = m.forward(&[chunk(s, 0, &p[..split], true)]).unwrap().remove(0);
        let b = m.forward(&[chunk(s, split, &p[split..], true)]).unwrap().remove(0);
        m.free_seq(s);
        let got: Vec<f32> = a.iter().chain(b.iter()).copied().collect();
        assert!(got == whole, "split at {split}");
    }
}

/// The integer-dot NVFP4 rows (activations as int8 pairs) agree with the rows decoded to floats:
/// within 1e-3 of the outputs' RMS on activations with outliers 20x the rest (3e-4 measured; the
/// pair resolves 1.6e-5 of a row's largest value, the experts' own 4-bit weights a few percent).
#[test]
fn integer_expert_rows_match_decoded_rows() {
    match fullm_kernels::exec::ink2_sdot_error() {
        Some(e) => assert!(e < 1e-3, "integer NVFP4 rows off by {e} of the outputs' RMS"),
        None => eprintln!("no integer-dot path in this build"),
    }
}

/// Draft tokens in a chunk, right or wrong, leave the sequence as plain decoding does: the logits
/// of every later step are bit for bit those of feeding the accepted tokens one at a time (the
/// executor puts the convolution histories back past the rejected rows).
#[test]
fn rejected_drafts_leave_no_trace() {
    let raw = std::fs::read_to_string(fixture().join("oracle.json")).expect("oracle.json");
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let ids =
        |x: &serde_json::Value| x.as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect::<Vec<_>>();
    let c = &v["cases"][0];
    let (prompt, g) = (ids(&c["prompt"]), ids(&c["greedy"]));
    let mut m = load(BackendKind::CPU);
    let vocab = m.vocab();
    let wrong = |t: u32| (t + 1) % vocab as u32;
    // plain: the greedy tokens one at a time
    let s = m.alloc_seq().unwrap();
    m.forward(&[chunk(s, 0, &prompt, false)]).unwrap();
    let plain: Vec<Vec<f32>> =
        (0..9).map(|i| m.forward(&[chunk(s, prompt.len() + i, &g[i..i + 1], false)]).unwrap().remove(0)).collect();
    m.free_seq(s);
    // with drafts: (tokens, drafts among them, accepted)
    let s = m.alloc_seq().unwrap();
    m.forward(&[chunk(s, 0, &prompt, false)]).unwrap();
    let steps: Vec<(Vec<u32>, usize, usize)> = vec![
        (vec![g[0], wrong(g[1]), wrong(g[2])], 2, 0),
        (vec![g[1], g[2], wrong(g[3])], 2, 1),
        (vec![g[3]], 0, 0),
        (vec![g[4], g[5], g[6]], 2, 2),
        (vec![g[7], wrong(g[8]), g[0], g[1]], 3, 0),
        (vec![g[8]], 0, 0),
    ];
    let mut at = 0;
    for (toks, draft, ok) in steps {
        let out = m
            .forward(&[SeqChunk { seq: s, pos: prompt.len() + at, tokens: &toks, all_logits: true, draft }])
            .unwrap()
            .remove(0);
        if draft > 0 {
            m.accept(s, ok).unwrap();
        }
        for i in 0..=ok {
            assert!(out[i * vocab..(i + 1) * vocab] == plain[at + i][..], "token {} of the continuation", at + i);
        }
        at += 1 + ok;
    }
}

/// A sequence started from the prefix cache continues bit for bit like the one that filled the
/// pages: with each page the executor keeps the convolution histories as they were after its last
/// position. The pages here were filled by a prompt chunk, by single tokens and by drafts (some
/// rejected), and a second prompt shares only its first pages.
#[test]
fn a_cached_prefix_continues_bit_for_bit() {
    let raw = std::fs::read_to_string(fixture().join("oracle.json")).expect("oracle.json");
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let ids =
        |x: &serde_json::Value| x.as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect::<Vec<_>>();
    let c = &v["cases"][0];
    let (prompt, g) = (ids(&c["prompt"]), ids(&c["greedy"]));
    let mut m = load(BackendKind::CPU);
    assert!(m.has_prefix_cache());
    let (vocab, page) = (m.vocab(), m.page_tokens());
    let wrong = |t: u32| (t + 1) % vocab as u32;
    // the text: the prompt and nine greedy tokens; plain[i] = the logits after text[..= i]
    let text: Vec<u32> = prompt.iter().chain(&g[..9]).copied().collect();
    let s = m.alloc_seq().unwrap();
    let plain = m.forward(&[chunk(s, 0, &text, true)]).unwrap().remove(0);
    let row = |i: usize| &plain[i * vocab..(i + 1) * vocab];
    m.free_seq(s);
    assert_eq!(m.flush_prefix_cache(), (text.len() / page) * page, "the first run's full pages were cached");
    // filled again in pieces: the prompt, a token, then drafts across page ends
    let s = m.alloc_seq().unwrap();
    m.forward(&[chunk(s, 0, &prompt, false)]).unwrap();
    let n = prompt.len();
    m.forward(&[chunk(s, n, &g[..1], false)]).unwrap();
    m.forward(&[SeqChunk { seq: s, pos: n + 1, tokens: &[g[1], g[2], wrong(g[3]), g[4]], all_logits: true, draft: 3 }])
        .unwrap();
    m.accept(s, 1).unwrap();
    m.forward(&[SeqChunk { seq: s, pos: n + 3, tokens: &[g[3], g[4], g[5], g[6]], all_logits: true, draft: 3 }])
        .unwrap();
    m.accept(s, 3).unwrap();
    m.free_seq(s);
    // the whole text again: all but its last page comes from the cache
    let (s, have) = m.alloc_seq_cached(&text).unwrap();
    assert_eq!(have, (n + 7) / page * page, "pages filled before the sequence was freed");
    let out = m.forward(&[chunk(s, have, &text[have..], true)]).unwrap().remove(0);
    for i in have..text.len() {
        assert!(out[(i - have) * vocab..(i - have + 1) * vocab] == *row(i), "position {i} after a cache hit of {have}");
    }
    m.free_seq(s);
    // a text that leaves the first after two pages and a token
    let mut other = text[..2 * page + 1].to_vec();
    other.extend([5u32, 9, 2, 7, 1]);
    let s = m.alloc_seq().unwrap();
    let want = m.forward(&[chunk(s, 0, &other, true)]).unwrap().remove(0);
    m.free_seq(s);
    m.flush_prefix_cache();
    let s = m.alloc_seq().unwrap();
    m.forward(&[chunk(s, 0, &text, false)]).unwrap();
    let (s2, have) = m.alloc_seq_cached(&other).unwrap();
    assert_eq!(have, 2 * page);
    let out = m.forward(&[chunk(s2, have, &other[have..], true)]).unwrap().remove(0);
    assert!(out[..] == want[have * vocab..], "a prompt sharing two pages with a running sequence");
}
