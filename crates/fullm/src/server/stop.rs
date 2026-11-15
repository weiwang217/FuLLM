// SPDX-License-Identifier: Apache-2.0
//! Stop strings (OpenAI's `stop`): matched on the text a request generates, as it streams.
//!
//! The engine stops on token ids only; a stop string can end mid-token or span several, so the
//! server matches the detokenized text, holds back a tail that could still become one, and
//! cancels the request when one completes (its next token is dropped). Like vLLM, the whole
//! generated text is matched, reasoning included; a stop string spanning the switch from
//! reasoning to answer is not seen.

use crate::tokenizer::Channel;

pub struct StopStrings {
    stops: Vec<String>,
    /// Emit the stop string itself (vLLM's `include_stop_str_in_output`).
    include: bool,
    held: String,
    held_ch: Channel,
    /// The stop string that ended the text.
    pub matched: Option<String>,
}

impl StopStrings {
    pub fn new(stops: Vec<String>, include: bool) -> Self {
        StopStrings { stops, include, held: String::new(), held_ch: Channel::Content, matched: None }
    }

    /// Generated text in; what can go out now. After a match: the text before it (and the stop
    /// string with `include`), then nothing.
    pub fn push(&mut self, ch: Channel, s: &str) -> Vec<(Channel, String)> {
        let mut out = Vec::new();
        if self.matched.is_some() {
            return out;
        }
        if self.stops.is_empty() {
            if !s.is_empty() {
                out.push((ch, s.to_string()));
            }
            return out;
        }
        if ch != self.held_ch && !self.held.is_empty() {
            out.push((self.held_ch, std::mem::take(&mut self.held)));
        }
        self.held_ch = ch;
        self.held.push_str(s);
        // the earliest stop string, the longest at that position
        let hit = self
            .stops
            .iter()
            .filter_map(|t| self.held.find(t.as_str()).map(|at| (at, std::cmp::Reverse(t.len()), t)))
            .min();
        if let Some((at, std::cmp::Reverse(len), t)) = hit {
            let end = if self.include { at + len } else { at };
            if end > 0 {
                out.push((ch, self.held[..end].to_string()));
            }
            self.matched = Some(t.clone());
            self.held.clear();
            return out;
        }
        let cut = self.held.len() - self.partial_tail();
        if cut > 0 {
            out.push((ch, self.held[..cut].to_string()));
            self.held.drain(..cut);
        }
        out
    }

    /// The held text, at the end of the stream.
    pub fn finish(&mut self) -> Vec<(Channel, String)> {
        if self.matched.is_some() || self.held.is_empty() {
            return Vec::new();
        }
        vec![(self.held_ch, std::mem::take(&mut self.held))]
    }

    /// The longest tail of the held text that begins some stop string (a proper prefix of it).
    fn partial_tail(&self) -> usize {
        let h = &self.held;
        let mut best = 0;
        for t in &self.stops {
            for k in (best + 1..=t.len().saturating_sub(1).min(h.len())).rev() {
                if h.is_char_boundary(h.len() - k) && t.starts_with(&h[h.len() - k..]) {
                    best = k;
                    break;
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(stops: &[&str], include: bool, pieces: &[(Channel, &str)]) -> (Vec<(Channel, String)>, Option<String>) {
        let mut s = StopStrings::new(stops.iter().map(|x| x.to_string()).collect(), include);
        let mut out: Vec<(Channel, String)> = pieces.iter().flat_map(|&(ch, p)| s.push(ch, p)).collect();
        out.extend(s.finish());
        (out, s.matched)
    }

    fn text(out: &[(Channel, String)], ch: Channel) -> String {
        out.iter().filter(|(c, _)| *c == ch).map(|(_, s)| s.as_str()).collect()
    }

    const C: Channel = Channel::Content;
    const R: Channel = Channel::Reasoning;

    #[test]
    fn stop_across_pieces_is_cut_and_not_shown() {
        let (out, m) = run(&["\nObservation:"], false, &[(C, "Action: search\nObs"), (C, "ervation: x"), (C, " more")]);
        assert_eq!(text(&out, C), "Action: search");
        assert_eq!(m.as_deref(), Some("\nObservation:"));
        // nothing of the stop string went out before the match was decided
        assert!(out.iter().all(|(_, s)| !s.contains("Obs")));
    }

    #[test]
    fn include_and_earliest_and_no_match() {
        let (out, m) = run(&["END", "B"], true, &[(C, "aaB..END")]);
        assert_eq!((text(&out, C).as_str(), m.as_deref()), ("aaB", Some("B")));
        // a held partial match that never completes is flushed at the end
        let (out, m) = run(&["STOP"], false, &[(C, "hello ST"), (C, "O")]);
        assert_eq!((text(&out, C).as_str(), m), ("hello STO", None));
        // no stop strings: passthrough
        let (out, _) = run(&[], false, &[(R, "r"), (C, "c")]);
        assert_eq!(out, vec![(R, "r".to_string()), (C, "c".to_string())]);
    }

    #[test]
    fn channels_and_multibyte() {
        // the reasoning is matched too (as vLLM does); held text keeps its channel
        let (out, m) = run(&["。"], false, &[(R, "想一想"), (R, "。然后"), (C, "答")]);
        assert_eq!((text(&out, R).as_str(), text(&out, C).as_str(), m.as_deref()), ("想一想", "", Some("。")));
        let (out, m) = run(&["xyz"], false, &[(R, "abx"), (C, "yz!")]);
        assert_eq!((text(&out, R).as_str(), text(&out, C).as_str(), m), ("abx", "yz!", None));
        // a multi-byte stop string split across pieces
        let (out, m) = run(&["结束"], false, &[(C, "好的结"), (C, "束了")]);
        assert_eq!((text(&out, C).as_str(), m.as_deref()), ("好的", Some("结束")));
    }
}
