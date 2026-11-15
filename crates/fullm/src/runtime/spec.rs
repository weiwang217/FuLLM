// SPDX-License-Identifier: Apache-2.0
//! Drafts for speculative decoding without a draft model: n-gram lookup in the sequence's own
//! context (prompt lookup decoding). The tokens that followed the latest earlier occurrence of
//! the context's last n tokens are proposed; the model verifies them in one step and keeps the
//! longest prefix it agrees with. Agents' contexts repeat themselves (tool output quoted back,
//! code edited in place, file paths and identifiers), which is where this pays; elsewhere it
//! finds no match and costs nothing.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// The maps' keys are hashes already.
#[derive(Default)]
struct Identity(u64);

impl Hasher for Identity {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!("keys are u64")
    }
    fn write_u64(&mut self, x: u64) {
        self.0 = x;
    }
}

type Map = HashMap<u64, u32, BuildHasherDefault<Identity>>;

fn hash(toks: &[u32]) -> u64 {
    // FNV-1a over the ids, then a final mix
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &t in toks {
        h = (h ^ t as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    (h ^ (h >> 29)).wrapping_mul(0xbf58_476d_1ce4_e5b9)
}

/// A sequence's tokens and, per n-gram length, where each n-gram last ended: a lookup is a few
/// hash probes whatever the context's length. The n-grams ending at the last token are indexed
/// only when the next token arrives, so a lookup never finds the suffix it looks for.
pub struct Ngram {
    min: usize,
    max: usize,
    toks: Vec<u32>,
    /// `maps[n - min]`: hash of `toks[i + 1 - n..=i]` -> the latest such `i` indexed.
    maps: Vec<Map>,
    /// The n-grams ending before this position are in the maps.
    indexed: usize,
}

impl Ngram {
    /// N-grams of `min..=max` tokens (`1 <= min <= max`).
    pub fn new(min: usize, max: usize) -> Self {
        assert!(1 <= min && min <= max, "n-gram lengths {min}..={max}");
        Ngram { min, max, toks: Vec::new(), maps: (min..=max).map(|_| Map::default()).collect(), indexed: 0 }
    }

    pub fn len(&self) -> usize {
        self.toks.len()
    }

    /// The context indexed so far.
    pub fn tokens(&self) -> &[u32] {
        &self.toks
    }

    pub fn is_empty(&self) -> bool {
        self.toks.is_empty()
    }

    /// Append context tokens.
    pub fn push(&mut self, toks: &[u32]) {
        self.toks.extend_from_slice(toks);
        let end = self.toks.len().saturating_sub(1);
        while self.indexed < end {
            let i = self.indexed;
            for n in self.min..=self.max.min(i + 1) {
                self.maps[n - self.min].insert(hash(&self.toks[i + 1 - n..=i]), i as u32);
            }
            self.indexed += 1;
        }
    }

    /// Up to `k` tokens that followed the latest earlier occurrence of the context's longest
    /// suffix of `max` down to `min` tokens; empty when none occurred before.
    pub fn propose(&self, k: usize) -> &[u32] {
        let len = self.toks.len();
        if k == 0 {
            return &[];
        }
        for n in (self.min..=self.max.min(len)).rev() {
            let suffix = &self.toks[len - n..];
            if let Some(&j) = self.maps[n - self.min].get(&hash(suffix)) {
                let j = j as usize;
                if &self.toks[j + 1 - n..=j] == suffix {
                    return &self.toks[j + 1..(j + 1 + k).min(len)];
                }
            }
        }
        &[]
    }
}

/// The followers of one n-gram: up to four tokens with how often each came next (a heavy-hitter
/// table: a newcomer replaces the rarest, so frequent followers stay).
#[derive(Clone, Copy, Default)]
struct Next {
    tok: [u32; 4],
    cnt: [u32; 4],
}

impl Next {
    fn add(&mut self, t: u32) {
        if let Some(i) = (0..4).find(|&i| self.cnt[i] > 0 && self.tok[i] == t) {
            self.cnt[i] += 1;
            return;
        }
        let i = (0..4).min_by_key(|&i| self.cnt[i]).unwrap_or(0);
        self.tok[i] = t;
        self.cnt[i] = 1;
    }

    /// The most frequent follower, when seen at least `seen` times and at least `share` of the
    /// time.
    fn best(&self, seen: u32, share: f32) -> Option<u32> {
        let i = (0..4).max_by_key(|&i| self.cnt[i])?;
        let total: u32 = self.cnt.iter().sum();
        (self.cnt[i] >= seen && self.cnt[i] as f32 >= share * total as f32).then_some(self.tok[i])
    }
}

/// Continuations seen in finished requests' replies, across requests: for n-grams of `min..=max`
/// tokens, the tokens that came next and how often. A request whose own context has no match
/// takes its drafts from here: what other replies said after the same few tokens (a format, a
/// boilerplate phrase, a reasoning habit). Bounded: past `cap` n-grams it starts over.
pub struct Global {
    min: usize,
    max: usize,
    cap: usize,
    seen: u32,
    share: f32,
    maps: Vec<HashMap<u64, Next, BuildHasherDefault<Identity>>>,
    entries: usize,
}

impl Global {
    /// A continuation must have come next this many times, and this share of the time, before it
    /// is drafted. Measured on an H100 (no prompt sent twice): 2 and a half drafted twice as much,
    /// kept 16-62% of it, and slowed reasoning replies by 4% at c=1; 3 and two thirds keeps 36-84%
    /// and gains 3-11% everywhere.
    pub const SEEN: u32 = 3;
    pub const SHARE: f32 = 0.67;

    pub fn new(min: usize, max: usize, cap: usize) -> Self {
        Self::with_confidence(min, max, cap, Self::SEEN, Self::SHARE)
    }

    pub fn with_confidence(min: usize, max: usize, cap: usize, seen: u32, share: f32) -> Self {
        assert!(1 <= min && min <= max, "n-gram lengths {min}..={max}");
        Global { min, max, cap, seen, share, maps: (min..=max).map(|_| HashMap::default()).collect(), entries: 0 }
    }

    pub fn entries(&self) -> usize {
        self.entries
    }

    /// A finished reply: `lead` is the context just before it (its prompt's last tokens), whose
    /// n-grams count only as the ones a reply token follows.
    pub fn add(&mut self, lead: &[u32], reply: &[u32]) {
        if reply.is_empty() {
            return;
        }
        if self.entries > self.cap {
            self.maps.iter_mut().for_each(HashMap::clear);
            self.entries = 0;
        }
        let lead = &lead[lead.len().saturating_sub(self.max)..];
        let seq: Vec<u32> = [lead, reply].concat();
        for i in lead.len()..seq.len() {
            for n in self.min..=self.max.min(i) {
                let map = &mut self.maps[n - self.min];
                let e = map.entry(hash(&seq[i - n..i])).or_insert_with(|| {
                    self.entries += 1;
                    Next::default()
                });
                e.add(seq[i]);
            }
        }
    }

    /// Up to `k` drafts after `ctx`: the longest n-gram ending it with a confident follower gives
    /// the next token, then again from the context so extended.
    pub fn propose(&self, ctx: &[u32], k: usize) -> Vec<u32> {
        let mut c: Vec<u32> = ctx[ctx.len().saturating_sub(self.max)..].to_vec();
        let start = c.len();
        while c.len() - start < k {
            let len = c.len();
            let next = (self.min..=self.max.min(len)).rev().find_map(|n| {
                self.maps[n - self.min].get(&hash(&c[len - n..])).and_then(|e| e.best(self.seen, self.share))
            });
            match next {
                Some(t) => c.push(t),
                None => break,
            }
        }
        c.split_off(start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ngram(toks: &[u32]) -> Ngram {
        let mut g = Ngram::new(1, 3);
        g.push(toks);
        g
    }

    #[test]
    fn proposes_what_followed_the_longest_suffix() {
        // "1 2 3" occurred before, followed by 4 5
        let g = ngram(&[1, 2, 3, 4, 5, 9, 1, 2, 3]);
        assert_eq!(g.propose(2), &[4, 5]);
        assert_eq!(g.propose(10), &[4, 5, 9, 1, 2, 3]);
        assert_eq!(g.propose(0), &[] as &[u32]);
    }

    #[test]
    fn the_latest_occurrence_wins_and_longer_suffixes_first() {
        // "7 8" ends twice before: the latest one (followed by 6) wins; the one-token suffix "8"
        // alone would also match, but the two-token match comes first
        let g = ngram(&[7, 8, 5, 7, 8, 6, 1, 8, 2, 7, 8]);
        assert_eq!(g.propose(1), &[6]);
        // only the last token repeats: a one-token match
        let g = ngram(&[4, 9, 3, 5, 9]);
        assert_eq!(g.propose(2), &[3, 5]);
    }

    #[test]
    fn no_earlier_occurrence_no_draft() {
        assert!(ngram(&[1, 2, 3, 4]).propose(4).is_empty());
        assert!(ngram(&[1]).propose(4).is_empty());
        assert!(Ngram::new(2, 3).propose(4).is_empty());
        // with n >= 2 a lone repeated token is not enough
        let mut g = Ngram::new(2, 3);
        g.push(&[4, 9, 3, 5, 9]);
        assert!(g.propose(2).is_empty());
    }

    #[test]
    fn pushing_token_by_token_equals_pushing_at_once() {
        let toks: Vec<u32> = (0..500u32).map(|i| (i * 7 + i / 13) % 23).collect();
        let mut a = Ngram::new(2, 4);
        for (i, &t) in toks.iter().enumerate() {
            a.push(&[t]);
            let mut b = Ngram::new(2, 4);
            b.push(&toks[..=i]);
            assert_eq!(a.propose(5), b.propose(5), "after {} tokens", i + 1);
        }
    }

    #[test]
    fn global_continuations_need_to_repeat() {
        let mut g = Global::new(2, 4, 1 << 20);
        // seen once, then twice: not drafted
        g.add(&[1, 2], &[3, 4, 5, 6]);
        assert!(g.propose(&[9, 1, 2], 4).is_empty());
        g.add(&[7, 1, 2], &[3, 4, 5, 8]);
        assert!(g.propose(&[9, 1, 2], 4).is_empty());
        // three times: drafted, as far as the replies agree (after 5: 6 twice, 8 once, not 2/3)
        g.add(&[1, 2], &[3, 4, 5, 6]);
        assert_eq!(g.propose(&[9, 1, 2], 4), vec![3, 4, 5]);
        // a fourth reply settles 5 -> 6 (3 of 4)
        g.add(&[8, 1, 2], &[3, 4, 5, 6]);
        assert_eq!(g.propose(&[9, 1, 2], 6), vec![3, 4, 5, 6]);
        assert_eq!(g.propose(&[9, 1, 2], 2), vec![3, 4]);
        assert!(g.propose(&[5, 5], 4).is_empty());
    }

    #[test]
    fn global_prefers_the_longest_context_and_resets_past_its_cap() {
        let mut g = Global::new(1, 3, 1 << 20);
        // after "2" alone the replies split, after "1 2" they agree on 7
        for _ in 0..3 {
            g.add(&[1, 2], &[7]);
            g.add(&[5, 2], &[8]);
            g.add(&[6, 2], &[8]);
        }
        assert_eq!(g.propose(&[1, 2], 1), vec![7]);
        assert_eq!(g.propose(&[6, 2], 1), vec![8]);
        let mut small = Global::new(2, 2, 4);
        small.add(&[], &(0..20).collect::<Vec<u32>>());
        let before = small.entries();
        small.add(&[], &[1, 2, 3]);
        assert!(small.entries() < before, "{} n-grams after the reset", small.entries());
    }

    #[test]
    fn a_long_context_indexes_fast() {
        let toks: Vec<u32> = (0..32768u32).map(|i| i.wrapping_mul(2654435761) % 151_000).collect();
        let t = std::time::Instant::now();
        let mut g = Ngram::new(2, 4);
        g.push(&toks);
        g.push(&toks[100..103]);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        assert_eq!(g.propose(3), &toks[103..106]);
        // a few ms in release for 32k tokens x 3 lengths; generous for debug builds
        assert!(ms < 2000.0, "indexing 32k tokens took {ms:.1} ms");
    }
}
