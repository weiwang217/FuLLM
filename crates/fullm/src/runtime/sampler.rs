// SPDX-License-Identifier: Apache-2.0
//! Token sampling: greedy, or temperature with optional top-k, min-p and top-p; the per-token
//! logit adjustments before it (penalties, logit bias, min_tokens); log-probabilities.
//!
//! The draw (the executors' `sample_rows` makes the same one on the device; `sk_sample_row` in
//! fullm_exec.h): softmax(logits / temperature) over the candidates, the values at or above the
//! top_k-th largest (ties stay), then those at or above the largest threshold whose candidates
//! hold top_p of the mass; the token is the one whose cumulative mass, in token order, first
//! passes u * mass. Thresholds instead of a sorted list: the device finds them with a radix
//! select and a binary search, in a fixed order, and never sorts the vocabulary.
//!
//! u is the first double of a PCG32 seeded from `(seed, step)`, so a request's draw at step n
//! does not depend on what else ran, how its prefill was chunked, or how many draws other
//! requests made. Together with the backend's batch-invariant logits this makes a seeded
//! request reproducible under any load, and lets a draft be verified against the very token
//! the request would draw at its position.

use std::collections::{BTreeMap, HashMap};

pub use fullm_kernels::exec::{LogitAdj, MAX_TOP_LOGPROBS};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamplingParams {
    pub max_tokens: usize,
    /// 0 means greedy.
    pub temperature: f32,
    /// 1.0 disables nucleus filtering.
    pub top_p: f32,
    /// 0 disables top-k.
    pub top_k: usize,
    pub seed: u64,
    /// Generation ends when one of these is sampled; it is counted but not emitted.
    pub stop_token_ids: Vec<u32>,
    /// OpenAI's: subtracted from the logit of every token the output holds ([-2, 2]; 0 off).
    #[serde(default)]
    pub presence_penalty: f32,
    /// OpenAI's: subtracted from a token's logit once per time the output holds it.
    #[serde(default)]
    pub frequency_penalty: f32,
    /// vLLM's (transformers'): the logits of tokens the prompt or the output holds are divided by
    /// it when positive, multiplied when negative (> 0; 1 off).
    #[serde(default = "one")]
    pub repetition_penalty: f32,
    /// OpenAI's logit_bias: added to the token's logit ([-100, 100]).
    #[serde(default)]
    pub logit_bias: Vec<(u32, f32)>,
    /// vLLM's: tokens less likely than min_p x the most likely one's probability are not drawn
    /// ([0, 1]; 0 off).
    #[serde(default)]
    pub min_p: f32,
    /// vLLM's: the stop ids are not drawn before the output holds this many tokens.
    #[serde(default)]
    pub min_tokens: usize,
    /// Log-probabilities: None, or Some(k): each drawn token's and its k most likely alternatives'
    /// (at most `MAX_TOP_LOGPROBS`), of the model's logits (before adjustments and temperature).
    #[serde(default)]
    pub logprobs: Option<usize>,
    /// The LoRA adapter slot the request runs with (unmerged adapters; None: the base model).
    #[serde(default)]
    pub lora: Option<usize>,
}

fn one() -> f32 {
    1.0
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            max_tokens: 16,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
            stop_token_ids: Vec::new(),
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            logit_bias: Vec::new(),
            min_p: 0.0,
            min_tokens: 0,
            logprobs: None,
            lora: None,
        }
    }
}

impl SamplingParams {
    /// Whether its logits get per-token adjustments (`adjustments`): penalties, logit bias,
    /// min_tokens. They depend on what was generated, so such a request takes no drafts.
    pub fn adjusts(&self) -> bool {
        self.presence_penalty != 0.0
            || self.frequency_penalty != 0.0
            || self.repetition_penalty != 1.0
            || !self.logit_bias.is_empty()
            || self.min_tokens > 0
    }

    /// Min-p as a cut below the row's largest logit (`SampleRow::min_p_cut`): temperature x
    /// ln(min_p), computed once here so the host and every backend cut at the same float; 0
    /// off (greedy, or min_p 0); min_p 1 keeps the largest only.
    pub fn min_p_cut(&self) -> f32 {
        if self.min_p <= 0.0 || self.temperature <= 0.0 {
            0.0
        } else if self.min_p >= 1.0 {
            -f32::MIN_POSITIVE
        } else {
            (self.temperature as f64 * (self.min_p as f64).ln()) as f32
        }
    }
}

/// A request's logit adjustments for a draw, in vLLM's order (`LogitAdj::apply`): logit bias;
/// repetition penalty on the tokens of the prompt (`prompt_set`: distinct, empty unless it
/// applies) and of the output; frequency and presence penalties on the output's (`counts`: how
/// often each was generated); with fewer than min_tokens generated, the stop ids never drawn
/// (those within the `vocab` logits: an id past them is never drawn anyway). Distinct tokens,
/// ascending, into `out`.
pub fn adjustments(
    p: &SamplingParams,
    prompt_set: &[u32],
    counts: &HashMap<u32, u32>,
    generated: usize,
    vocab: usize,
    out: &mut Vec<LogitAdj>,
) {
    let mut m: BTreeMap<u32, LogitAdj> = BTreeMap::new();
    let new = |t: u32| LogitAdj { token: t as i32, bias: 0.0, rep: 1.0, sub: 0.0 };
    for &(t, b) in &p.logit_bias {
        m.entry(t).or_insert_with(|| new(t)).bias += b;
    }
    if p.repetition_penalty != 1.0 {
        for &t in prompt_set.iter().chain(counts.keys()) {
            m.entry(t).or_insert_with(|| new(t)).rep = p.repetition_penalty;
        }
    }
    if p.frequency_penalty != 0.0 || p.presence_penalty != 0.0 {
        for (&t, &c) in counts {
            m.entry(t).or_insert_with(|| new(t)).sub += p.frequency_penalty * c as f32 + p.presence_penalty;
        }
    }
    if generated < p.min_tokens {
        for &t in p.stop_token_ids.iter().filter(|&&t| (t as usize) < vocab) {
            m.entry(t).or_insert_with(|| new(t)).sub = f32::INFINITY;
        }
    }
    out.extend(m.into_values().filter(|a| a.bias != 0.0 || a.rep != 1.0 || a.sub != 0.0));
}

/// A drawn token's log-probability and its most likely alternatives' (`k` at most), as the
/// backends compute them (`logprob_row_host`): log-softmax of `logits` as given, sums in f64;
/// alternatives by value, then by id.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenLogprobs {
    pub logprob: f32,
    pub top: Vec<(u32, f32)>,
}

pub fn logprobs(logits: &[f32], picked: u32, k: usize) -> TokenLogprobs {
    let m = logits.iter().fold(f32::NEG_INFINITY, |a, &x| a.max(x));
    let s: f64 = logits.iter().map(|&x| ((x - m) as f64).exp()).sum();
    let lse = m as f64 + s.ln();
    let k = k.min(MAX_TOP_LOGPROBS).min(logits.len());
    let mut top: Vec<u32> = Vec::with_capacity(k + 1);
    let better =
        |a: u32, b: u32| logits[a as usize] > logits[b as usize] || (logits[a as usize] == logits[b as usize] && a < b);
    for i in 0..logits.len() as u32 {
        if top.len() < k || top.last().is_some_and(|&w| better(i, w)) {
            let at = top.iter().position(|&t| better(i, t)).unwrap_or(top.len());
            top.insert(at, i);
            top.truncate(k);
        }
    }
    TokenLogprobs {
        logprob: (logits[picked as usize] as f64 - lse) as f32,
        top: top.into_iter().map(|t| (t, (logits[t as usize] as f64 - lse) as f32)).collect(),
    }
}

/// PCG-XSH-RR 64/32 (O'Neill), the reference constants.
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut r = Pcg32 { state: 0, inc: (stream << 1) | 1 };
        r.next_u32();
        r.state = r.state.wrapping_add(seed);
        r.next_u32();
        r
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        xorshifted.rotate_right((old >> 59) as u32)
    }

    /// Uniform in [0, 1) with 53 bits.
    pub fn next_f64(&mut self) -> f64 {
        let hi = (self.next_u32() as u64) << 21;
        let lo = (self.next_u32() >> 11) as u64;
        (hi | lo) as f64 / (1u64 << 53) as f64
    }
}

/// First index of the maximum; NaN never wins.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[best] || logits[best].is_nan() {
            best = i;
        }
    }
    best as u32
}

/// Larger float, larger key (finite values).
fn key(f: f32) -> u32 {
    let u = f.to_bits();
    if u & 0x8000_0000 != 0 {
        !u
    } else {
        u | 0x8000_0000
    }
}

/// Draw the token for `step` (0-based index of the generated token) of a request.
pub fn sample(logits: &[f32], p: &SamplingParams, step: usize) -> u32 {
    let n = logits.len();
    let best = argmax(logits);
    if p.temperature <= 0.0 || n <= 1 || !logits[best as usize].is_finite() {
        return best;
    }
    let m = logits[best as usize];
    let inv_t = (1.0 / p.temperature as f64) as f32;
    // top-k: the k-th largest value; every value equal to it stays
    let mut lo = if p.top_k > 0 && p.top_k < n {
        let mut keys: Vec<u32> = logits.iter().map(|&f| key(f)).collect();
        keys.select_nth_unstable_by(p.top_k - 1, |a, b| b.cmp(a));
        keys[p.top_k - 1]
    } else {
        0
    };
    // min-p: the values at least the largest + the cut
    let cut = p.min_p_cut();
    if cut < 0.0 {
        lo = lo.max(key(m + cut));
    }
    let w: Vec<f32> = logits
        .iter()
        .map(|&f| if f.is_finite() && key(f) >= lo { (((f - m) * inv_t) as f64).exp() as f32 } else { 0.0 })
        .collect();
    let mass = |t: u32| -> f64 {
        let mut s = 0.0f64;
        for (&f, &x) in logits.iter().zip(&w) {
            if key(f) >= t {
                s += x as f64;
            }
        }
        s
    };
    // top-p: the largest threshold whose candidates hold at least top_p of the mass
    let mut t = lo;
    if p.top_p < 1.0 {
        let want = p.top_p as f64 * mass(lo);
        let (mut a, mut b) = (lo, key(m));
        while a < b {
            let mid = a + (b - a).div_ceil(2);
            if mass(mid) >= want {
                a = mid;
            } else {
                b = mid - 1;
            }
        }
        t = a;
    }
    let mut r = Pcg32::new(p.seed, step as u64).next_f64() * mass(t);
    let mut last = best;
    for (i, (&f, &x)) in logits.iter().zip(&w).enumerate() {
        if key(f) < t {
            continue;
        }
        last = i as u32;
        r -= x as f64;
        if r < 0.0 {
            return i as u32;
        }
    }
    last
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcg32_reference_stream() {
        // pcg32-global demo: seed 42, sequence 54
        let mut r = Pcg32::new(42, 54);
        let got: Vec<u32> = (0..6).map(|_| r.next_u32()).collect();
        assert_eq!(got, vec![0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293, 0xbfa4784b, 0xcbed606e]);
    }

    #[test]
    fn greedy_and_filters() {
        let l = [0.0, 3.0, 1.0, 2.9];
        assert_eq!(sample(&l, &SamplingParams::default(), 0), 1);
        let p = SamplingParams { temperature: 1.0, top_k: 1, ..Default::default() };
        for s in 0..20 {
            assert_eq!(sample(&l, &p, s), 1, "top-1 is greedy");
        }
        let p = SamplingParams { temperature: 1.0, top_k: 2, seed: 9, ..Default::default() };
        for s in 0..50 {
            assert!([1, 3].contains(&sample(&l, &p, s)));
        }
        let p = SamplingParams { temperature: 0.7, top_p: 0.9, seed: 3, ..Default::default() };
        let a: Vec<u32> = (0..30).map(|s| sample(&l, &p, s)).collect();
        let b: Vec<u32> = (0..30).map(|s| sample(&l, &p, s)).collect();
        assert_eq!(a, b, "same (seed, step) -> same draw");
        // top-p 0.5: token 1 alone holds 0.48 of the mass, so 1 and 3 are the candidates
        let p = SamplingParams { temperature: 1.0, top_p: 0.5, seed: 5, ..Default::default() };
        assert!((0..60).all(|s| [1, 3].contains(&sample(&l, &p, s))));
    }

    #[test]
    fn min_p_keeps_the_likely_tokens() {
        // probabilities at temperature 1: e^3 : e^2.9 : e^1 : e^0 -> min_p 0.5 keeps 1 and 3 only
        let l = [0.0, 3.0, 1.0, 2.9];
        let p = SamplingParams { temperature: 1.0, min_p: 0.5, seed: 4, ..Default::default() };
        let got: std::collections::HashSet<u32> = (0..200).map(|s| sample(&l, &p, s)).collect();
        assert_eq!(got, [1, 3].into_iter().collect());
        let p = SamplingParams { temperature: 1.0, min_p: 1.0, seed: 4, ..Default::default() };
        assert!((0..50).all(|s| sample(&l, &p, s) == 1), "min_p 1: the most likely only");
        assert_eq!(SamplingParams { min_p: 0.5, ..Default::default() }.min_p_cut(), 0.0, "greedy: no cut");
    }

    #[test]
    fn adjustments_in_vllms_order() {
        let p = SamplingParams {
            logit_bias: vec![(5, 2.0), (9, -1.0)],
            repetition_penalty: 2.0,
            frequency_penalty: 0.5,
            presence_penalty: 0.25,
            min_tokens: 3,
            stop_token_ids: vec![7],
            ..Default::default()
        };
        let counts: HashMap<u32, u32> = [(5, 2), (6, 1)].into_iter().collect();
        let mut out = Vec::new();
        adjustments(&p, &[1, 6], &counts, 1, 1000, &mut out);
        let toks: Vec<i32> = out.iter().map(|a| a.token).collect();
        assert_eq!(toks, vec![1, 5, 6, 7, 9], "distinct, ascending");
        let at = |t: i32| *out.iter().find(|a| a.token == t).unwrap();
        // 5: bias 2, in the output twice: x + 2, halved when positive, less 0.5 x 2 + 0.25
        assert_eq!(at(5).apply(4.0), (4.0 + 2.0) / 2.0 - 1.25);
        assert_eq!(at(5).apply(-4.0), (-4.0 + 2.0) * 2.0 - 1.25);
        assert_eq!(at(1).apply(3.0), 1.5, "prompt tokens: repetition only");
        assert_eq!(at(7).apply(10.0), f32::NEG_INFINITY, "a stop id before min_tokens");
        assert_eq!(at(9).apply(1.0), 0.0);
        out.clear();
        adjustments(&p, &[1, 6], &counts, 3, 1000, &mut out);
        assert!(out.iter().all(|a| a.token != 7), "min_tokens reached");
        out.clear();
        adjustments(&SamplingParams::default(), &[], &counts, 0, 1000, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn logprobs_of_the_logits() {
        let l = [0.0f32, 3.0, 1.0, 2.9, 3.0];
        let lp = logprobs(&l, 2, 3);
        let z: f64 = l.iter().map(|&x| (x as f64).exp()).sum();
        assert!((lp.logprob as f64 - (1.0 - z.ln())).abs() < 1e-6);
        // ties by id: 1 before 4
        assert_eq!(lp.top.iter().map(|t| t.0).collect::<Vec<_>>(), vec![1, 4, 3]);
        assert!(lp.top.windows(2).all(|w| w[0].1 >= w[1].1));
        assert!(logprobs(&l, 0, 0).top.is_empty());
    }

    #[test]
    fn draws_follow_the_softmax() {
        let l = [0.0f32, 1.0, 2.0, -1.0];
        let p = SamplingParams { temperature: 1.0, seed: 11, ..Default::default() };
        let z: f64 = l.iter().map(|&x| (x as f64).exp()).sum();
        let mut count = [0usize; 4];
        let n = 40_000;
        for s in 0..n {
            count[sample(&l, &p, s) as usize] += 1;
        }
        for (i, &c) in count.iter().enumerate() {
            let want = (l[i] as f64).exp() / z;
            let got = c as f64 / n as f64;
            assert!((got - want).abs() < 0.01, "token {i}: {got:.4} drawn, {want:.4} expected");
        }
    }

    #[test]
    fn the_same_draw_as_the_backends() {
        use fullm_kernels::exec::{sample_rows_host, SampleRow};
        let mut r = Pcg32::new(1, 2);
        for &n in &[7usize, 1000, 151_936] {
            let rows = 12;
            let logits: Vec<f32> = (0..rows * n).map(|_| (r.next_f64() * 8.0 - 4.0) as f32).collect();
            let params: Vec<SamplingParams> = (0..rows)
                .map(|i| SamplingParams {
                    temperature: [0.0, 0.7, 0.6, 1.0, 0.3, 1.3][i % 6],
                    top_p: [1.0, 0.8, 0.95, 1.0, 0.5, 0.99][(i / 2) % 6],
                    top_k: [0, 20, 0, 1, 5, n][i % 6],
                    min_p: [0.0, 0.05, 0.0, 0.3, 0.0, 1.0, 0.1][i % 7],
                    seed: 100 + i as u64,
                    ..Default::default()
                })
                .collect();
            let rows_p: Vec<SampleRow> = params
                .iter()
                .enumerate()
                .map(|(i, p)| SampleRow {
                    min_p_cut: p.min_p_cut(),
                    ..SampleRow::new(p.temperature, p.top_p, p.top_k, p.seed, 3 * i as u64)
                })
                .collect();
            let want = sample_rows_host(&logits, n, &rows_p);
            for (i, p) in params.iter().enumerate() {
                let got = sample(&logits[i * n..(i + 1) * n], p, 3 * i);
                assert_eq!(got as i32, want[i], "row {i} of {n}: {p:?}");
            }
        }
    }
}
