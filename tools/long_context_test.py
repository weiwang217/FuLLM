#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""long_context_test.py - prompts up to a model's context window, through an OpenAI-compatible server.

For each length and needle depth, a passphrase is hidden in real text and asked for at the end; the
reply must contain it. Lengths are fractions of the window (--fractions, the window from /v1/models'
max_model_len or --window) or token counts (--tokens). Prompts are sized in tokens, not words: a probe
request (max_tokens 1) measures the text's tokens per word and the chat template's overhead from the
server's own prompt_tokens, and a prompt the server refuses as too long is cut by 2% and sent again.
Then:
  over the window   a prompt ~5% past the window must be refused with an HTTP error (not crash,
                    hang or truncate silently), and a short request must still be answered afterwards
  two at once       (--parallel) two prompts of ~45% of the window sent together, both checked

Reported per case: prompt tokens, time to first token (streamed), total time, pass or fail.
Exit status 1 when any case fails.

Usage:
  long_context_test.py --url http://127.0.0.1:8000                      # fractions of max_model_len
  long_context_test.py --url ... --tokens 262144,524288,1000000 --depths 0.5 --timeout 3600
  long_context_test.py --url ... --suffix ""                             # no Qwen3 /no_think switch
Standard library only.
"""

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request

FILLER = ("The history of computing is a story of layered abstractions, each hiding the machinery below "
          "it so that the next generation of engineers could build something larger. ").split()
WORDS = ["maple", "cobalt", "harbor", "falcon", "quartz", "willow", "ember", "glacier", "saffron", "tundra"]


def post(url, body, timeout, stream):
    """(text, finish_reason, usage, ttft s or None, total s); raises urllib.error.HTTPError on refusal."""
    body = dict(body, stream=stream)
    if stream:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = urllib.request.urlopen(req, timeout=timeout)
    if not stream:
        v = json.loads(r.read())
        ch = v["choices"][0]
        return ch["message"].get("content") or "", ch.get("finish_reason"), v.get("usage") or {}, None, time.time() - t0
    text, finish, usage, ttft = "", None, {}, None
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        v = json.loads(data)
        usage = v.get("usage") or usage
        for ch in v.get("choices", []):
            d = ch.get("delta") or {}
            if ttft is None and (d.get("content") or d.get("reasoning_content")):
                ttft = time.time() - t0
            text += d.get("content") or ""
            finish = ch.get("finish_reason") or finish
    return text, finish, usage, ttft, time.time() - t0


class Doc:
    def __init__(self, path):
        try:
            self.words = open(path, encoding="utf-8", errors="replace").read().split() if path else []
        except OSError:
            self.words = []
        self.words = self.words or FILLER

    def build(self, n_words, depth, rng, suffix):
        start = rng.randrange(len(self.words))
        body = [self.words[(start + j) % len(self.words)] for j in range(max(0, n_words))]
        secret = f"{rng.choice(WORDS)}-{rng.randint(1000, 9999)}"
        body.insert(int(len(body) * depth), f"The secret passphrase is {secret}.")
        prompt = ("Below is a long document. Somewhere in it is a secret passphrase. Read the document, then answer "
                  "the question after it.\n\n" + " ".join(body) +
                  "\n\nQuestion: what is the secret passphrase? Reply with the passphrase only." + suffix)
        return prompt, secret


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", help="default: the server's first model")
    ap.add_argument("--window", type=int, help="context window in tokens (default: /v1/models max_model_len)")
    ap.add_argument("--fractions", default="0.25,0.5,0.75,0.9,0.98")
    ap.add_argument("--tokens", help="comma-separated prompt lengths in tokens, instead of --fractions")
    ap.add_argument("--depths", default="0.1,0.5,0.9", help="where the needle sits (fractions of the document)")
    ap.add_argument("--reply", type=int, default=48, help="max_tokens of each reply (prompt + reply fit the window)")
    ap.add_argument("--text", help="the document's words (default: README.md, repeated as needed)")
    ap.add_argument("--suffix", default=" /no_think", help="appended to the question (Qwen3's switch; '' for others)")
    ap.add_argument("--timeout", type=float, default=1800, help="seconds per request")
    ap.add_argument("--parallel", action="store_true", help="also two ~45%%-of-window prompts at once")
    ap.add_argument("--no-over", dest="over", action="store_false", help="skip the over-the-window check")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    url = a.url.rstrip("/")
    models = json.loads(urllib.request.urlopen(url + "/v1/models", timeout=60).read())["data"]
    model = a.model or models[0]["id"]
    window = a.window or next((m.get("max_model_len") for m in models if m["id"] == model), None)
    if not window:
        sys.exit("no max_model_len from /v1/models: pass --window")
    text = a.text or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "README.md")
    doc, rng = Doc(text), random.Random(a.seed)
    base = {"model": model, "temperature": 0, "max_tokens": a.reply}

    def prompt_tokens(n_words):
        p, _ = doc.build(n_words, 0.5, random.Random(0), a.suffix)
        _, _, usage, _, _ = post(url, dict(base, messages=[{"role": "user", "content": p}], max_tokens=1), a.timeout,
                                 False)
        return usage.get("prompt_tokens", 0)

    # tokens per word of this text and the template's overhead, from the server's own counts
    over0 = prompt_tokens(0)
    probe = min(20000, max(500, window // 8))
    per_word = (prompt_tokens(probe) - over0) / probe
    print(f"long_context_test: {url} model {model}, window {window} tokens, text {os.path.basename(text or 'filler')} "
          f"({per_word:.2f} tokens a word, {over0} tokens of template), reply {a.reply} tokens", flush=True)

    targets = ([int(t) for t in a.tokens.split(",")] if a.tokens
               else [int(float(f) * window) for f in a.fractions.split(",")])
    depths = [float(d) for d in a.depths.split(",")]
    rows, fails = [], 0

    def run_case(name, target, depth):
        target = min(target, window - a.reply)
        n = int((target - over0) / per_word)
        for attempt in range(4):
            prompt, secret = doc.build(n, depth, rng, a.suffix)
            try:
                reply, finish, usage, ttft, total = post(url, dict(base, messages=[{"role": "user", "content": prompt}]),
                                                         a.timeout, True)
            except urllib.error.HTTPError as e:
                msg = e.read()[:200].decode(errors="replace")
                if e.code == 400 and attempt < 3:  # counted a little long: cut 2% and try again
                    n = int(n * 0.98)
                    continue
                return dict(case=name, tokens=target, ok=False, why=f"HTTP {e.code}: {msg}")
            except Exception as e:  # timeouts, dropped connections
                return dict(case=name, tokens=target, ok=False, why=f"{type(e).__name__}: {e}")
            ok = secret.lower() in reply.lower()
            return dict(case=name, tokens=usage.get("prompt_tokens", 0), ttft=ttft, total=total, ok=ok,
                        why="" if ok else f"wanted {secret}, got {reply.strip()[:80]!r} ({finish})")
        return dict(case=name, tokens=target, ok=False, why="refused as too long four times")

    def report(r):
        nonlocal fails
        fails += not r["ok"]
        rows.append(r)
        ttft = f"{r['ttft']:.2f} s" if r.get("ttft") is not None else "-"
        total = f"{r['total']:.1f} s" if r.get("total") is not None else "-"
        print(f"  {r['case']:28s} {r['tokens']:>9} tok  ttft {ttft:>9}  total {total:>8}  "
              f"{'PASS' if r['ok'] else 'FAIL'} {r['why']}", flush=True)

    for t in targets:
        for d in depths:
            report(run_case(f"{t / window:.0%} of window, needle {d:.0%}", t, d))

    if a.parallel:
        out = [None, None]

        def one(i):
            out[i] = run_case(f"two at once #{i + 1}", int(0.45 * window), 0.3 + 0.4 * i)

        th = [threading.Thread(target=one, args=(i,)) for i in range(2)]
        for x in th:
            x.start()
        for x in th:
            x.join()
        for r in out:
            report(r)

    if a.over:
        n = int((1.05 * window - over0) / per_word)
        prompt, _ = doc.build(n, 0.5, rng, a.suffix)
        try:
            post(url, dict(base, messages=[{"role": "user", "content": prompt}]), a.timeout, False)
            r = dict(case="over the window (105%)", tokens=int(1.05 * window), ok=False, why="accepted: no error")
        except urllib.error.HTTPError as e:
            r = dict(case="over the window (105%)", tokens=int(1.05 * window), ok=400 <= e.code < 500,
                     why=f"HTTP {e.code}: {e.read()[:120].decode(errors='replace')}")
        except Exception as e:
            r = dict(case="over the window (105%)", tokens=int(1.05 * window), ok=False, why=f"{type(e).__name__}: {e}")
        report(r)
        try:
            reply, _, _, _, total = post(url, dict(base, messages=[{"role": "user", "content": "Say OK." + a.suffix}],
                                                   max_tokens=8), 120, False)
            report(dict(case="short request afterwards", tokens=0, total=total, ok=bool(reply.strip()), why=""))
        except Exception as e:
            report(dict(case="short request afterwards", tokens=0, ok=False, why=f"{type(e).__name__}: {e}"))

    print(f"\n{len(rows) - fails} / {len(rows)} pass")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
