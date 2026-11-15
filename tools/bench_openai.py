#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""bench_openai.py - time an OpenAI-compatible server the same way for every engine.

Streams chat completions and reports time to first token (the first chunk with text other than
whitespace and <think> / </think>: engines that pass Qwen3's opening "<think>\n" on as text and
engines that parse it into reasoning_content time the same event), per-request decode rate
(tokens after the first / time after the first) and aggregate throughput, at one or more
concurrency levels. Token counts are the server's own (usage.completion_tokens, requested
with stream_options.include_usage): counting SSE chunks under-counts, since engines merge
or drop the text of tokens that do not end a character. Standard library only.

Usage: bench_openai.py --url http://localhost:8000 --model NAME [--tokens 300] [--concurrency 1,4]
                        [--context WORDS]      (prepend ~WORDS words of text: a long-prompt / prefill test)
                        [--context-text FILE]  (the words come from FILE, a different stretch per request:
                                                real text gives a MoE model its real routing; the default
                                                filler is one sentence repeated)
                        [--reuse-prompts]      (identical prompts across rounds and runs: lets a prefix cache hit)
Every request's prompt starts with a tag unique to the run, round and request unless
--reuse-prompts, so an engine's prefix cache cannot turn a repeated prompt into a cheaper
prefill and skew the comparison. With --context the warm-up request carries a context too, so
the cost of an engine's first long prompt (compilation, graph capture) is not timed.
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request

PROMPTS = [
    "Explain in three short paragraphs why mixture-of-experts models are cheaper to run than dense models of the same size.",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
    "Describe how a CPU cache hierarchy works and why it matters for performance.",
    "What are the main differences between TCP and UDP? Give examples of when to use each.",
    "Summarize the causes and consequences of the French Revolution.",
    "Explain gradient descent to a high school student.",
    "List ten creative uses for a paperclip and explain each briefly.",
    "Describe the water cycle and its importance for life on Earth.",
]


FILLER = ("The history of computing is a story of layered abstractions, each hiding the machinery below "
          "it so that the next generation of engineers could build something larger. ").split()


def with_context(prompt, words, i, source):
    """prompt preceded by `words` words of source, a different stretch per request"""
    if words <= 0:
        return prompt
    start = (i * 997) % len(source)
    body = [source[(start + j) % len(source)] for j in range(words)]
    return "Read this passage, then answer.\n\n" + " ".join(body) + "\n\n" + prompt


def one(url, model, prompt, tokens, out):
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": tokens, "temperature": 0, "stream": True, "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    chunks, n = 0, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                n = chunk["usage"].get("completion_tokens", n)
            ch = chunk.get("choices") or [{}]
            delta = ch[0].get("delta", {})
            text = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
            if text.replace("<think>", "").replace("</think>", "").strip():
                chunks += 1
                if first is None:
                    first = time.perf_counter()
    t1 = time.perf_counter()
    if n is None:
        n = chunks  # the server sent no usage: fall back to counting chunks
    end = t1  # the stream closes right after the last token; trailing tokens may have no text
    out.append({"ttft": (first or t1) - t0, "total": t1 - t0, "tokens": n,
                "decode_rate": (n - 1) / (end - first) if first and n > 1 and end > first else 0.0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--concurrency", default="1,4")
    ap.add_argument("--context", type=int, default=0, help="words of text before each prompt")
    ap.add_argument("--context-text", help="file the context words come from (default: a repeated sentence)")
    ap.add_argument("--reuse-prompts", action="store_true", help="identical prompts across rounds and runs")
    a = ap.parse_args()
    source = FILLER
    if a.context_text:
        with open(a.context_text, encoding="utf-8", errors="replace") as f:
            source = f.read().split() or FILLER
    nonce = "" if a.reuse_prompts else f"{int(time.time() * 1000) % 10_000_000:07d}-"
    one(a.url, a.model, f"[warm {nonce}] " + with_context("Hello", a.context, 999_999, source), 8, [])
    for round_id, c in enumerate([int(x) for x in a.concurrency.split(",")]):
        out, threads = [], []
        t0 = time.perf_counter()
        for i in range(c):
            prompt = with_context(PROMPTS[i % len(PROMPTS)], a.context, c * 1000 + i, source)
            if not a.reuse_prompts:
                prompt = f"[request {nonce}{round_id}-{i}] {prompt}"
            th = threading.Thread(target=one, args=(a.url, a.model, prompt, a.tokens, out))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        wall = time.perf_counter() - t0
        toks = sum(o["tokens"] for o in out)
        tt = sorted(o["ttft"] * 1e3 for o in out)
        print(f"concurrency {c}: TTFT median {statistics.median(tt):.0f} ms (min {tt[0]:.0f}, max {tt[-1]:.0f}) | "
              f"decode {statistics.median(o['decode_rate'] for o in out):.1f} tok/s per request | "
              f"aggregate {toks / wall:.1f} tok/s ({toks} tokens in {wall:.1f}s)")


if __name__ == "__main__":
    main()
