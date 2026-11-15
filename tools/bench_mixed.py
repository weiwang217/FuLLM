#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""bench_mixed.py - prompts from a few tokens up to the context window, and long prompts mixed with short ones,
against an OpenAI-compatible server, the same way for every engine.

  verify the replies must not depend on what runs beside them: --verify-longs long prompts (a passphrase hidden
         in --verify-long tokens of filler, asked for at the end) and short questions with one right answer,
         first each alone, then all at once with the short ones --hol-delay ms behind the long ones (the
         prefix cache flushed before each round, /flush_cache or /reset_prefix_cache). Reports how many
         replies are identical to the solo ones (greedy; a backend whose chunked prefill is not bit-exact may
         differ in a few tokens) and how many answers are right in each round.
  sweep  one request at a time, every prompt cold, at each length in --lengths (tokens; "max" is the context
         window: --max-context less --gen and a margin), then one prompt past the window, which the server
         should refuse cleanly. Per length: TTFT (the whole prefill), prefill tokens/s, and the decode rate
         at that context.
  hol    head-of-line blocking: --hol-short short prompts sent together alone, then the same sent --hol-delay
         ms after N long ones (one run per value of --hol-longs; prompts of --hol-long tokens). A scheduler
         that hands each step's token budget to prompts in arrival order makes the short ones wait for every
         long prefill ahead of them.
  mixed  open-loop traffic for --duration seconds: short requests arriving at --short-rate per second
         (Poisson; prompts of --short-tokens, log-uniform, replies of --short-gen tokens), first alone, then
         with long requests on top at --long-rate per second (--long-tokens, --long-gen); the short arrivals
         are the same in both. Reports the short requests' TTFT p50/p90/p99, their inter-token latency (the
         mean, and the longest gap: a decode held up by a prefill chunk), the time to their whole reply (what
         an agent waits for before it can parse a tool call), the long requests' TTFT, and the output and
         prompt token rates.

Lengths are in tokens: a first request measures the server's tokens per word of the prompt text, every prompt
is cut to its target from that, and the reports give the counts the server returned. Every prompt starts with
a tag of its own, so none is served from a cache. Prompts are plain text through /v1/completions (stream,
ignore_eos, temperature 0), with bench_agent.py's client. Standard library only.

Usage: bench_mixed.py --url http://localhost:8000 --model NAME [--scenario verify,sweep,hol,mixed] [--max-context 32768]
                      [--verify-long max] [--verify-longs 2]
                      [--lengths 16,128,1024,4096,8192,16384,24576,max] [--gen 128] [--repeat 3]
                      [--hol-long max] [--hol-longs 1,2] [--hol-short 8] [--hol-short-tokens 256] [--hol-delay 100]
                      [--duration 120] [--short-rate 4] [--short-tokens 16:1024] [--short-gen 64:256]
                      [--long-rate 0.1] [--long-tokens 16384:max] [--long-gen 128]
                      [--context-text FILE] [--seed 0] [--json OUT]
"""

import argparse
import json
import math
import random
import statistics
import threading
import time
import urllib.error
import urllib.request

from bench_agent import complete, cut, pct, text_source


class Prompts:
    """prompts of a length in tokens, each with a tag of its own"""

    def __init__(self, a, source, tag):
        self.a, self.source, self.tag = a, source, tag
        self.n = 0
        self.lock = threading.Lock()
        self.per_word = 1.3

    def calibrate(self):
        words = min(4000, self.a.max_context // 3)
        r = complete(self.a.url, self.a.model, f"[{self.tag} cal] " + cut(self.source, 0, words), 1, self.a.timeout)
        if r.get("error") or not r.get("prompt_tokens"):
            raise SystemExit(f"the calibration request failed: {r.get('error') or 'no usage in the reply'}")
        self.per_word = r["prompt_tokens"] / words
        return self.per_word

    def make(self, tokens, rng):
        with self.lock:
            self.n += 1
            n = self.n
        head = f"[{self.tag} {n}] "
        words = max(1, round((tokens - 0.8 * len(head)) / self.per_word))  # digits are a token each
        return head + cut(self.source, rng.randrange(len(self.source)), words)

    def length(self, spec, gen):
        """a length in tokens: a number, or "max" (the window, less the reply and a margin)"""
        if spec == "max":
            return self.a.max_context - gen - max(64, self.a.max_context // 100)
        return int(spec)


def span(spec, prompts, gen):
    lo, hi = spec.split(":")
    return prompts.length(lo, gen), prompts.length(hi, gen)


def ms(x):
    return f"{x * 1e3:.0f}"


def itl(r):
    """a request's mean gap between tokens after the first"""
    n = (r.get("completion_tokens") or 0) - 1
    return (r["total"] - r["ttft"]) / n if n > 0 else None


QUESTIONS = [
    ("What is the capital of France?", ["Paris"]),
    ("What is 17 + 25?", ["42"]),
    ("Which planet is known as the Red Planet?", ["Mars"]),
    ("What is the chemical symbol for gold?", ["Au"]),
    ("How many days are there in a week?", ["7", "seven", "Seven"]),
    ("Who wrote Romeo and Juliet?", ["Shakespeare"]),
    ("What is the largest ocean on Earth?", ["Pacific"]),
    ("What is 9 times 8?", ["72"]),
]


def flush(url):
    """empty the server's prefix cache: fullm and SGLang's /flush_cache, vLLM's /reset_prefix_cache"""
    for path in ("/flush_cache", "/reset_prefix_cache"):
        try:
            with urllib.request.urlopen(urllib.request.Request(url + path, data=b"", method="POST"), timeout=120) as r:
                if r.status == 200:
                    return path
        except (urllib.error.URLError, OSError):
            pass
    return None


def run_verify(a, prompts, rng):
    gen = 24
    long_t = prompts.length(a.verify_long, gen)
    items = []  # (name, prompt, accepted answers)
    for j in range(a.verify_longs):
        code = f"{rng.choice(['AZURE', 'CEDAR', 'EMBER', 'QUARTZ', 'TUNDRA'])}-{rng.randrange(100, 1000)}"
        head = f"[{prompts.tag} verify {j}] Remember the passphrase hidden in the text below.\n\n"
        tail = "\n\nQuestion: What is the secret passphrase?\nAnswer: The secret passphrase is"
        words = max(1, round((long_t - 40) / prompts.per_word))
        depth = round(words * (0.25 + 0.5 * j / max(1, a.verify_longs - 1)))
        start = rng.randrange(len(prompts.source))
        filler = cut(prompts.source, start, words).split(" ")
        body = " ".join(filler[:depth]) + f" The secret passphrase is {code}. " + " ".join(filler[depth:])
        items.append((f"long {j}", head + body + tail, [code]))
    for i, (q, ok) in enumerate(QUESTIONS):
        items.append((f"q{i}", f"[{prompts.tag} verify q{i}] Q: {q}\nA:", ok))
    print(f"verify: {a.verify_longs} long prompts of ~{long_t} tokens (a passphrase inside) and {len(QUESTIONS)} questions,"
          f" each alone, then all at once")

    def ask(item):
        return complete(a.url, a.model, item[1], gen, a.timeout)

    flushed = flush(a.url)
    solo = [ask(it) for it in items]
    flush(a.url)
    mixed = [None] * len(items)

    def one(k):
        mixed[k] = ask(items[k])

    longs = [threading.Thread(target=one, args=(k,)) for k, it in enumerate(items) if it[0].startswith("long")]
    shorts = [threading.Thread(target=one, args=(k,)) for k, it in enumerate(items) if not it[0].startswith("long")]
    for t in longs:
        t.start()
    time.sleep(a.hol_delay / 1e3)
    for t in shorts:
        t.start()
    for t in longs + shorts:
        t.join()

    def right(r, ok):
        return not r.get("error") and any(x in r["text"] for x in ok)

    same = sum(1 for x, y in zip(solo, mixed) if not x.get("error") and x["text"] == y["text"])
    print(f"  identical to the solo reply: {same}/{len(items)}"
          + ("" if flushed else " (no cache flush endpoint: the second round may reuse the first one's cache)"))
    print(f"  right answers: alone {sum(right(r, it[2]) for r, it in zip(solo, items))}/{len(items)},"
          f" mixed {sum(right(r, it[2]) for r, it in zip(mixed, items))}/{len(items)}"
          f" (the model's own accuracy; the check is that mixing changes nothing)")
    for (name, _, ok), x, y in zip(items, solo, mixed):
        if x.get("error") or y.get("error"):
            print(f"  {name}: failed: {(x.get('error') or y.get('error'))[:160]}")
        elif x["text"] != y["text"]:
            n = next((k for k, (c, d) in enumerate(zip(x["text"], y["text"])) if c != d), min(len(x["text"]), len(y["text"])))
            print(f"  {name}: differs after {n} characters: alone {x['text'][:60]!r} | mixed {y['text'][:60]!r}")
    return {"scenario": "verify", "items": [{"name": it[0], "accepted": it[2], "solo": x, "mixed": y}
                                            for it, x, y in zip(items, solo, mixed)]}


def run_sweep(a, prompts, rng):
    print(f"sweep: one cold request at a time, {a.gen} tokens each, median of {a.repeat}")
    rows = []
    for spec in a.lengths.split(","):
        target = prompts.length(spec, a.gen)
        recs = [complete(a.url, a.model, prompts.make(target, rng), a.gen, a.timeout) for _ in range(a.repeat)]
        ok = [r for r in recs if not r.get("error")]
        if not ok:
            print(f"  {target:>6} tokens: failed: {recs[0]['error'][:200]}")
            rows.append({"target": target, "error": recs[0]["error"]})
            continue
        p = statistics.median(r["prompt_tokens"] or 0 for r in ok)
        ttft = statistics.median(r["ttft"] for r in ok)
        gaps = [g for g in (itl(r) for r in ok) if g]
        decode = 1 / statistics.median(gaps) if gaps else float("nan")
        print(f"  {p:>6.0f} tokens: TTFT {ms(ttft):>6} ms | prefill {p / ttft:7.0f} tok/s | decode {decode:6.1f} tok/s"
              + (f" | {len(recs) - len(ok)} failed" if len(ok) < len(recs) else ""))
        rows.append({"target": target, "prompt_tokens": p, "ttft": ttft, "decode_tok_s": decode, "runs": recs})
    over = a.max_context + 1024
    r = complete(a.url, a.model, prompts.make(over, rng), a.gen, a.timeout)
    if r.get("error"):
        print(f"  past the window ({over} tokens): refused: {r['error'][:200]}")
    else:
        print(f"  past the window ({over} tokens): ACCEPTED, {r.get('prompt_tokens')} prompt tokens")
    rows.append({"target": over, "past_window": True, "result": r})
    return {"scenario": "sweep", "rows": rows}


def run_hol(a, prompts, rng):
    long_t = prompts.length(a.hol_long, 16)
    print(f"hol: {a.hol_short} prompts of {a.hol_short_tokens} tokens, {a.hol_delay:.0f} ms after long ones of {long_t}")
    out = {"scenario": "hol", "runs": []}
    for n_long in [0] + [int(x) for x in a.hol_longs.split(",")]:
        recs = []
        lock = threading.Lock()

        def one(kind, tokens, gen):
            r = complete(a.url, a.model, prompts.make(tokens, rng), gen, a.timeout)
            r["kind"] = kind
            with lock:
                recs.append(r)

        threads = [threading.Thread(target=one, args=("long", long_t, 16)) for _ in range(n_long)]
        for t in threads:
            t.start()
        if n_long:
            time.sleep(a.hol_delay / 1e3)
        shorts = [threading.Thread(target=one, args=("short", a.hol_short_tokens, 32)) for _ in range(a.hol_short)]
        for t in shorts:
            t.start()
        for t in threads + shorts:
            t.join()
        errs = [r for r in recs if r.get("error")]
        s = [r["ttft"] for r in recs if r["kind"] == "short" and not r.get("error")]
        lg = [r["ttft"] for r in recs if r["kind"] == "long" and not r.get("error")]
        line = f"  {n_long} long: short TTFT median {ms(statistics.median(s)) if s else '-'} ms, max {ms(max(s)) if s else '-'}"
        if lg:
            line += f" | long TTFT {', '.join(ms(x) for x in sorted(lg))} ms"
        if errs:
            line += f" | {len(errs)} failed: {errs[0]['error'][:120]}"
        print(line)
        out["runs"].append({"longs": n_long, "records": recs})
    return out


def arrivals(rate, duration, rng):
    t, out = 0.0, []
    while rate > 0:
        t += rng.expovariate(rate)
        if t >= duration:
            return out
        out.append(t)
    return out


def run_load(a, events):
    """events: (seconds, kind, prompt, gen), each sent at its time in a thread of its own"""
    recs = []
    lock = threading.Lock()

    def one(t, kind, prompt, gen):
        r = complete(a.url, a.model, prompt, gen, a.timeout)
        r.update(kind=kind, at=t)
        with lock:
            recs.append(r)

    threads = []
    t0 = time.perf_counter()
    for t, kind, prompt, gen in sorted(events, key=lambda e: e[0]):
        wait = t0 + t - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        th = threading.Thread(target=one, args=(t, kind, prompt, gen))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    return recs, time.perf_counter() - t0


def load_summary(label, recs, wall):
    ok = [r for r in recs if not r.get("error")]
    short = [r for r in ok if r["kind"] == "short"]
    long_ = [r for r in ok if r["kind"] == "long"]
    s_ttft = [r["ttft"] for r in short]
    s_itl = [g for g in (itl(r) for r in short) if g]
    s_gap = [r["max_gap"] for r in short]
    s_total = [r["total"] for r in short]
    out_tok = sum(r.get("completion_tokens") or 0 for r in ok)
    in_tok = sum(r.get("prompt_tokens") or 0 for r in ok)
    line = (f"  {label}: {len(short)} short"
            f" | TTFT p50 {ms(pct(s_ttft, .5))} p90 {ms(pct(s_ttft, .9))} p99 {ms(pct(s_ttft, .99))} max {ms(max(s_ttft, default=0))} ms"
            f" | ITL p50 {ms(pct(s_itl, .5))} ms, longest gap p50 {ms(pct(s_gap, .5))} p99 {ms(pct(s_gap, .99))} ms"
            f" | whole reply p50 {ms(pct(s_total, .5))} p90 {ms(pct(s_total, .9))} p99 {ms(pct(s_total, .99))} ms")
    if any(r["kind"] == "long" for r in recs):
        lt = [r["ttft"] for r in long_]
        line += f" | {len(long_)} long, TTFT p50 {ms(pct(lt, .5))} max {ms(max(lt, default=0))} ms"
    line += f" | out {out_tok / wall:.0f} tok/s, prompt {in_tok / wall:.0f} tok/s"
    if len(ok) < len(recs):
        bad = next(r for r in recs if r.get("error"))
        line += f" | {len(recs) - len(ok)} failed: {bad['error'][:120]}"
    print(line)
    return {"label": label, "wall": wall, "records": recs}


def run_mixed(a, prompts):
    s_lo, s_hi = span(a.short_tokens, prompts, 0)
    g_lo, g_hi = (int(x) for x in a.short_gen.split(":"))
    l_lo, l_hi = span(a.long_tokens, prompts, a.long_gen)
    rs = random.Random(a.seed * 7 + 1)
    shorts = [(t, round(math.exp(rs.uniform(math.log(s_lo), math.log(s_hi)))), rs.randint(g_lo, g_hi))
              for t in arrivals(a.short_rate, a.duration, rs)]
    rl = random.Random(a.seed * 7 + 2)
    longs = [(t, rl.randint(l_lo, l_hi), a.long_gen) for t in arrivals(a.long_rate, a.duration, rl)]
    print(f"mixed: {a.duration:.0f} s, {len(shorts)} short requests ({s_lo}-{s_hi} tokens, {g_lo}-{g_hi} out),"
          f" then with {len(longs)} long ones ({l_lo}-{l_hi} tokens, {a.long_gen} out)")
    out = {"scenario": "mixed", "phases": []}
    rng = random.Random(a.seed * 7 + 3)
    for label, extra in (("short only  ", []), ("short + long", longs)):
        events = [(t, "short", prompts.make(n, rng), g) for t, n, g in shorts]
        events += [(t, "long", prompts.make(n, rng), g) for t, n, g in extra]
        recs, wall = run_load(a, events)
        out["phases"].append(load_summary(label, recs, wall))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--scenario", default="verify,sweep,hol,mixed", help="comma-separated: verify, sweep, hol, mixed")
    ap.add_argument("--max-context", type=int, default=32768, help="the server's context window in tokens")
    ap.add_argument("--verify-long", default="max", help="verify: tokens of each long prompt")
    ap.add_argument("--verify-longs", type=int, default=2, help="verify: long prompts")
    ap.add_argument("--lengths", default="16,128,1024,4096,8192,16384,24576,max", help="sweep: prompt tokens")
    ap.add_argument("--gen", type=int, default=128, help="sweep: tokens generated per request")
    ap.add_argument("--repeat", type=int, default=3, help="sweep: requests per length")
    ap.add_argument("--hol-long", default="max", help="hol: tokens of each long prompt")
    ap.add_argument("--hol-longs", default="1,2", help="hol: long prompts ahead of the short ones, one run per value")
    ap.add_argument("--hol-short", type=int, default=8, help="hol: short prompts sent together")
    ap.add_argument("--hol-short-tokens", type=int, default=256, help="hol: tokens of each short prompt")
    ap.add_argument("--hol-delay", type=float, default=100, help="hol: ms between the long prompts and the short ones")
    ap.add_argument("--duration", type=float, default=120, help="mixed: seconds of arrivals per phase")
    ap.add_argument("--short-rate", type=float, default=4, help="mixed: short requests per second")
    ap.add_argument("--short-tokens", default="16:1024", help="mixed: min:max tokens of a short prompt")
    ap.add_argument("--short-gen", default="64:256", help="mixed: min:max tokens of a short reply")
    ap.add_argument("--long-rate", type=float, default=0.1, help="mixed: long requests per second")
    ap.add_argument("--long-tokens", default="16384:max", help="mixed: min:max tokens of a long prompt")
    ap.add_argument("--long-gen", type=int, default=128, help="mixed: tokens of a long reply")
    ap.add_argument("--context-text", help="file the words come from (default: pseudo-text)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--json", help="write every request's timings and token counts here")
    a = ap.parse_args()
    source = text_source(a.context_text, a.seed)
    tag = f"mix {int(time.time() * 1000) % 10_000_000:07d}"
    complete(a.url, a.model, f"[warm {tag}] Hello", 8, a.timeout)  # compilation, graph capture: not timed
    prompts = Prompts(a, source, tag)
    print(f"{prompts.calibrate():.3f} tokens per word; context window {a.max_context}")
    rng = random.Random(a.seed)
    results = []
    scenarios = a.scenario.split(",")
    if "verify" in scenarios:
        results.append(run_verify(a, prompts, rng))
    if "sweep" in scenarios:
        results.append(run_sweep(a, prompts, rng))
    if "hol" in scenarios:
        results.append(run_hol(a, prompts, rng))
    if "mixed" in scenarios:
        results.append(run_mixed(a, prompts))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"args": vars(a), "per_word": prompts.per_word, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
