#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""bench_agent.py - agent-loop workloads against an OpenAI-compatible server, the same way for every engine.

What an agent does to a server: each turn resends the whole growing context (system prompt and tool
schemas, history, tool output) plus a short append, and turns are sequential, so a prefix cache decides
the per-turn latency. Three scenarios:

  sessions  S concurrent sessions x T turns. Every session starts with the same system prompt; each turn
            appends a tool output of a random length (different text per session and turn) and generates
            --gen tokens; the next turn's prompt is the previous prompt + the reply + the next tool output.
            Reports TTFT by turn (turn 1 is a cold prefill of the system prompt, later turns are appends),
            session time, aggregate decode throughput and the prompt tokens the server served from cache.
  fanout    one long shared context, N sub-agents each appending a different task, all sent at once:
            first with the context uncached (does the engine compute it once or N times?), then a second
            set of N with it cached.
  users     closer to real traffic, in steady state: N users start over --ramp seconds and run sessions
            back to back for --duration seconds more; a session has --user-turns turns (random in the
            range), each with a tool output of --tool-words, a reply of --user-gen tokens, and a pause of
            --tool-ms on average (exponential) before the next turn; a session ends early when its prompt
            would pass --max-words. Counts the turns sent after the ramp: output tok/s, TTFT p50/p90/p99,
            inter-token latency (each request's mean, and its longest gap), prompt tokens cached.

Prompts are plain text through /v1/completions (stream, ignore_eos, temperature 0), so every engine sees
the same text with no chat template in between (Qwen3's template, for one, rewrites earlier turns). The
reply is fed back as text: re-tokenising may change a token at the join, as it would for a real client.
Every run starts its prompts with a tag of its own, so a server's cache from earlier runs cannot help,
while sessions within a run share the system prompt as real ones would.

Cached tokens come from usage.prompt_tokens_details.cached_tokens: fullm always reports it, vLLM with
--enable-prompt-tokens-details, SGLang with --enable-cache-report ("n/a" otherwise). Give every engine the
same KV budget (vLLM --num-gpu-blocks-override, SGLang --max-total-tokens, fullm --cache-tokens), or the
comparison measures cache sizes, not caches. Standard library only.

Usage: bench_agent.py --url http://localhost:8000 --model NAME [--scenario sessions,fanout,users]
                      [--sessions 1,4] [--turns 10] [--system-words 3000] [--tool-words 500:2000] [--gen 128]
                      [--tool-ms 0] [--fanout 8] [--fanout-words 20000] [--task-words 300]
                      [--users 16,32] [--ramp 30] [--duration 150] [--user-turns 6:14] [--user-gen 64:384]
                      [--max-words 17000] [--context-text FILE] [--seed 0] [--json OUT]
Words are ~1.3 tokens of English each: the defaults end a session at ~20k tokens and fan out from ~26k.
"""
import argparse
import json
import random
import statistics
import threading
import time
import urllib.error
import urllib.request

VOCAB = ("the of and to in is was for on that with as by at from this be are or an it not which have has "
         "had but were they their its been more one all would there other into after first also new some time "
         "system value table file error result request server model cache page token layer step batch queue "
         "thread memory kernel buffer index record field column function method class module package build "
         "test case input output state update read write load store copy merge split sort search match route "
         "agent tool call return list item entry node edge graph tree path branch commit change diff patch "
         "line block chunk range offset length size count total limit budget score weight bias scale shape "
         "north river city market council report study survey season summer winter village bridge road "
         "station harbour mountain valley forest island coast field garden museum library school college").split()


def text_source(path, seed):
    """the words prompts are cut from: a file, or pseudo-text from a fixed vocabulary"""
    if path:
        with open(path, encoding="utf-8", errors="replace") as f:
            words = f.read().split()
        if words:
            return words
    rng = random.Random(seed ^ 0x5EED)
    return [rng.choice(VOCAB) for _ in range(200_000)]


def cut(source, start, n):
    return " ".join(source[(start + j) % len(source)] for j in range(n))


def complete(url, model, prompt, gen, timeout):
    """one streamed completion: TTFT, total time, text, and the server's token counts"""
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": gen, "temperature": 0, "stream": True,
        "ignore_eos": True, "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(url + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first, text, usage, last, gap = None, [], None, None, 0.0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                chunk = json.loads(line[5:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices") or []:
                    if ch.get("text"):
                        text.append(ch["text"])
                        now = time.perf_counter()
                        if first is None:
                            first = now
                        else:
                            gap = max(gap, now - last)
                        last = now
    except (urllib.error.URLError, OSError, ValueError) as e:
        msg = str(e)
        if isinstance(e, urllib.error.HTTPError):  # the server's reason, e.g. a prompt over the context limit
            try:
                msg += ": " + e.read().decode(errors="replace")[:300]
            except OSError:
                pass
        return {"error": msg, "ttft": None, "total": time.perf_counter() - t0, "text": "".join(text)}
    t1 = time.perf_counter()
    usage = usage or {}
    details = usage.get("prompt_tokens_details") or {}
    return {"ttft": (first or t1) - t0, "total": t1 - t0, "text": "".join(text), "max_gap": gap,
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": details.get("cached_tokens")}


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def cache_summary(recs):
    """'cached X% of N prompt tokens', or n/a when the server does not report it"""
    prompt = sum(r.get("prompt_tokens") or 0 for r in recs)
    if any(r.get("cached_tokens") is None for r in recs) or not prompt:
        return f"cached n/a ({prompt} prompt tokens)"
    cached = sum(r["cached_tokens"] for r in recs)
    return f"cached {100 * cached / prompt:.1f}% of {prompt} prompt tokens"


def run_sessions(a, source, tag, n_sessions):
    rng = random.Random(a.seed * 1000 + n_sessions)
    lo, hi = (int(x) for x in a.tool_words.split(":"))
    system = (f"[{tag}] You are a coding agent working in a large repository. You call tools and read their "
              f"output before you answer. Reference material follows.\n\n" + cut(source, 0, a.system_words))
    # each session's tool outputs: distinct stretches of the source, fixed by the seed
    plans = [[(rng.randrange(len(source)), rng.randint(lo, hi)) for _ in range(a.turns)] for _ in range(n_sessions)]
    recs = [[] for _ in range(n_sessions)]
    ends = [0.0] * n_sessions
    run_t0 = time.perf_counter()

    def session(s):
        prompt = system + f"\n\nUser: task {s}: find and fix the failing test.\n"
        t0 = time.perf_counter()
        for turn, (start, words) in enumerate(plans[s]):
            prompt += "\nTool output:\n" + cut(source, start, words) + "\n\nAssistant:"
            sent = time.perf_counter() - run_t0
            r = complete(a.url, a.model, prompt, a.gen, a.timeout)
            r["turn"], r["session"], r["sent"] = turn, s, sent
            recs[s].append(r)
            if "error" in r:
                break
            prompt += r["text"] + "\n"
            if a.tool_ms and turn + 1 < a.turns:
                time.sleep(a.tool_ms / 1000)
        ends[s] = time.perf_counter() - t0

    t0 = time.perf_counter()
    threads = [threading.Thread(target=session, args=(s,)) for s in range(n_sessions)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0
    flat = [r for rs in recs for r in rs]
    ok = [r for r in flat if "error" not in r]
    errors = len(flat) - len(ok)
    gen = sum(r.get("completion_tokens") or 0 for r in ok)
    first = [r["ttft"] * 1e3 for r in ok if r["turn"] == 0]
    later = [r["ttft"] * 1e3 for r in ok if r["turn"] > 0]
    by_turn = [statistics.median([r["ttft"] * 1e3 for r in ok if r["turn"] == t] or [float("nan")])
               for t in range(a.turns)]
    last_prompt = max((r.get("prompt_tokens") or 0 for r in ok), default=0)
    print(f"sessions {n_sessions} x {a.turns} turns (context up to {last_prompt} tokens): "
          f"TTFT turn 1 median {statistics.median(first or [float('nan')]):.0f} ms | "
          f"later turns median {statistics.median(later or [float('nan')]):.0f} ms, p90 {pct(later, 0.9):.0f} ms | "
          f"session median {statistics.median(ends):.1f} s, all {wall:.1f} s | "
          f"aggregate {gen / wall:.1f} tok/s | {cache_summary(ok)}" + (f" | {errors} ERRORS" if errors else ""))
    print("  TTFT by turn (median ms): " + " ".join(f"{x:.0f}" for x in by_turn))
    if errors:
        print("  first error: " + next(r["error"] for r in flat if "error" in r))
    return {"scenario": "sessions", "sessions": n_sessions, "wall": wall, "session_times": ends,
            "requests": [{k: v for k, v in r.items() if k != "text"} for r in flat]}


def span(s):
    lo, _, hi = s.partition(":")
    return int(lo), int(hi or lo)


def run_users(a, source, tag, n_users):
    lo_t, hi_t = span(a.user_turns)
    lo_g, hi_g = span(a.user_gen)
    lo_w, hi_w = span(a.tool_words)
    system = (f"[{tag}] You are a coding agent working in a large repository. You call tools and read their "
              f"output before you answer. Reference material follows.\n\n" + cut(source, 0, a.system_words))
    recs, lock = [], threading.Lock()
    t_start = time.perf_counter()
    end_at = t_start + a.ramp + a.duration

    def user(u):
        rng = random.Random(a.seed * 7919 + u * 104729 + n_users)
        time.sleep(a.ramp * u / n_users)  # staggered arrivals
        k = 0
        while time.perf_counter() < end_at:
            prompt = system + f"\n\nUser: user {u}, task {k}: find and fix the failing test.\n"
            words = len(prompt.split())
            for turn in range(rng.randint(lo_t, hi_t)):
                w = rng.randint(lo_w, hi_w)
                if time.perf_counter() >= end_at or words + w > a.max_words:
                    break
                prompt += "\nTool output:\n" + cut(source, rng.randrange(len(source)), w) + "\n\nAssistant:"
                words += w + 3
                gen = rng.randint(lo_g, hi_g)
                sent = time.perf_counter() - t_start
                r = complete(a.url, a.model, prompt, gen, a.timeout)
                r.update(user=u, session=k, turn=turn, sent=sent)
                with lock:
                    recs.append(r)
                if "error" in r:
                    break
                prompt += r["text"] + "\n"
                words += len(r["text"].split()) + 1
                if a.tool_ms:
                    time.sleep(rng.expovariate(1000 / a.tool_ms))
            k += 1

    threads = [threading.Thread(target=user, args=(u,)) for u in range(n_users)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    win = [r for r in recs if a.ramp <= r["sent"] < a.ramp + a.duration]
    ok = [r for r in win if "error" not in r]
    errors = len(win) - len(ok)
    gen = sum(r.get("completion_tokens") or 0 for r in ok)
    ttft = [r["ttft"] * 1e3 for r in ok]
    later = [r["ttft"] * 1e3 for r in ok if r["turn"] > 0]
    itl = [(r["total"] - r["ttft"]) / (r["completion_tokens"] - 1) * 1e3 for r in ok if (r.get("completion_tokens") or 0) > 1]
    gaps = [r["max_gap"] * 1e3 for r in ok]
    prompt = statistics.mean([r.get("prompt_tokens") or 0 for r in ok] or [0])
    print(f"users {n_users}, {a.duration:.0f} s after a {a.ramp:.0f} s ramp: {len(ok)} turns (mean prompt {prompt:.0f} tokens), "
          f"output {gen / a.duration:.1f} tok/s | TTFT p50/p90/p99 {pct(ttft, 0.5):.0f}/{pct(ttft, 0.9):.0f}/{pct(ttft, 0.99):.0f} ms "
          f"(later turns {pct(later, 0.5):.0f}/{pct(later, 0.9):.0f}/{pct(later, 0.99):.0f}) | ITL p50/p90 {pct(itl, 0.5):.1f}/"
          f"{pct(itl, 0.9):.1f} ms, longest gap p50/p99 {pct(gaps, 0.5):.0f}/{pct(gaps, 0.99):.0f} ms | {cache_summary(ok)}"
          + (f" | {errors} ERRORS: {next(r['error'] for r in win if 'error' in r)}" if errors else ""))
    return {"scenario": "users", "users": n_users, "duration": a.duration, "ramp": a.ramp,
            "requests": [{k: v for k, v in r.items() if k != "text"} for r in recs]}


def run_fanout(a, source, tag):
    rng = random.Random(a.seed * 1000 + 7)
    context = (f"[{tag}] You are a sub-agent. The repository overview and the parent agent's notes follow.\n\n"
               + cut(source, rng.randrange(len(source)), a.fanout_words))
    out = {"scenario": "fanout", "n": a.fanout}
    for phase in ("uncached", "cached"):
        starts = [rng.randrange(len(source)) for _ in range(a.fanout)]
        recs = [None] * a.fanout

        def sub(j):
            prompt = context + f"\n\nSub-task {phase} {j}:\n" + cut(source, starts[j], a.task_words) + "\n\nAssistant:"
            recs[j] = complete(a.url, a.model, prompt, a.gen, a.timeout)

        t0 = time.perf_counter()
        threads = [threading.Thread(target=sub, args=(j,)) for j in range(a.fanout)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        wall = time.perf_counter() - t0
        ok = [r for r in recs if "error" not in r]
        tt = [r["ttft"] * 1e3 for r in ok]
        prompt = max((r.get("prompt_tokens") or 0 for r in ok), default=0)
        gen = sum(r.get("completion_tokens") or 0 for r in ok)
        print(f"fan-out {a.fanout} from {prompt} tokens, {phase}: TTFT median {statistics.median(tt or [float('nan')]):.0f} ms "
              f"(min {min(tt or [float('nan')]):.0f}, max {max(tt or [float('nan')]):.0f}) | all {wall:.1f} s | "
              f"aggregate {gen / wall:.1f} tok/s | {cache_summary(ok)}"
              + (f" | {len(recs) - len(ok)} ERRORS: {next(r['error'] for r in recs if 'error' in r)}" if len(ok) < len(recs) else ""))
        out[phase] = {"wall": wall, "requests": [{k: v for k, v in r.items() if k != "text"} for r in recs]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--scenario", default="sessions,fanout", help="comma-separated: sessions, fanout, users")
    ap.add_argument("--sessions", default="1,4", help="concurrent sessions, one run per value")
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--system-words", type=int, default=3000, help="the system prompt every session shares")
    ap.add_argument("--tool-words", default="500:2000", help="min:max words of tool output per turn")
    ap.add_argument("--gen", type=int, default=128, help="tokens generated per turn")
    ap.add_argument("--tool-ms", type=float, default=0, help="pause between a reply and the next turn (tool time)")
    ap.add_argument("--fanout", type=int, default=8, help="sub-agents per fan-out")
    ap.add_argument("--fanout-words", type=int, default=20000, help="words of the context the sub-agents share")
    ap.add_argument("--task-words", type=int, default=300, help="words of each sub-agent's own task")
    ap.add_argument("--context-text", help="file the words come from (default: pseudo-text)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--users", default="16,32", help="users scenario: concurrent users, one run per value")
    ap.add_argument("--ramp", type=float, default=30, help="users scenario: seconds over which users start")
    ap.add_argument("--duration", type=float, default=150, help="users scenario: measured seconds after the ramp")
    ap.add_argument("--user-turns", default="6:14", help="users scenario: min:max turns per session")
    ap.add_argument("--user-gen", default="64:384", help="users scenario: min:max tokens per reply")
    ap.add_argument("--max-words", type=int, default=17000, help="users scenario: end a session before its prompt passes this")
    ap.add_argument("--json", help="write every request's timings and token counts here")
    a = ap.parse_args()
    source = text_source(a.context_text, a.seed)
    tag = f"run {int(time.time() * 1000) % 10_000_000:07d}"
    complete(a.url, a.model, f"[warm {tag}] Hello", 8, a.timeout)  # compilation, graph capture: not timed
    results = []
    scenarios = a.scenario.split(",")
    if "sessions" in scenarios:
        for n in (int(x) for x in a.sessions.split(",")):
            results.append(run_sessions(a, source, f"{tag} s{n}", n))
    if "fanout" in scenarios:
        results.append(run_fanout(a, source, f"{tag} f"))
    if "users" in scenarios:
        for n in (int(x) for x in a.users.split(",")):
            results.append(run_users(a, source, f"{tag} u{n}", n))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"args": vars(a), "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
