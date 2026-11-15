#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""make_eagle3_scenarios.py - random runs of a model with an EAGLE-3 head, and the drafts torch expects at every draft
call (eagle3_ref.py): what crates/fullm/tests/eagle3.rs replays on fullm and compares.

A scenario is a list of ops on named sequences, as the engine drives a model: allocations (plain, or with a prefix-cache
lookup and the rest of the prompt in step-sized chunks, the head reading each), steps (several sequences at once: prompt chunks of 1 to 32 tokens,
decodes with 0 to 6 drafts), accepts (any count of a step's drafts), draft calls after every step for every sequence
in it (k from 0 to 8, after prompt chunks too) and frees. Prompts are random (1 to MAX_PROMPT tokens), or start with
part of an earlier sequence's context (cut on the 4-token page grid or off it: the prefix cache hits), or are a single
token.

  random  (default) tokens uniform over the vocabulary; a decode feeds its last drafts or random tokens, keeps a random
          number, and a random token follows: every path, whatever the model would say.
  natural (--natural) chat prompts (bench_spec's kinds, Qwen3's template, /no_think); the target's greedy token follows,
          the drafts fed are the head's, and a step keeps those the target agrees with: greedy speculative decoding.

Each draft op holds, per request, the expected drafts and their top-2 margins (where a margin is near 0 the other
token may come out of different rounding), for an f32 KV cache (expect, margins) and a bf16 one (expect_bf16,
margins_bf16: every layer's keys and values rounded to bf16, the head's too). Usage:
  make_eagle3_scenarios.py TARGET HEAD OUT.json [--scenarios 40] [--seqs 8] [--max-prompt 160] [--decodes 12]
                           [--step-tokens 64] [--seed 1] [--natural]
"""
import argparse
import json
import random
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from eagle3_ref import Ref  # noqa: E402

PAGE_GRID = 4  # the smallest page the tests run with


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("head")
    ap.add_argument("out")
    ap.add_argument("--scenarios", type=int, default=40)
    ap.add_argument("--seqs", type=int, default=8, help="sequences per scenario")
    ap.add_argument("--live", type=int, default=6, help="sequences alive at once")
    ap.add_argument("--max-prompt", type=int, default=160)
    ap.add_argument("--decodes", type=int, default=12, help="decode steps per sequence (most)")
    ap.add_argument("--step-tokens", type=int, default=64)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--natural", action="store_true")
    a = ap.parse_args()
    ref = Ref(a.target, a.head)
    # the same with a bf16 KV cache (fullm's default): each draft request carries both expectations
    ref16 = Ref(a.target, a.head, kv_bf16=True)
    vocab = ref.target.config.vocab_size
    rng = random.Random(a.seed)
    prompts = natural_prompts(a.target) if a.natural else None
    out, t0, n_req = [], time.time(), 0
    for si in range(a.scenarios):
        ops, n = scenario(ref, ref16, rng, a, vocab, prompts)
        out.append({"ops": ops})
        n_req += n
        print(f"scenario {si}: {len(ops)} ops, {n} draft requests ({time.time() - t0:.0f} s)", flush=True)
    json.dump({"natural": a.natural, "step_tokens": a.step_tokens, "scenarios": out}, open(a.out, "w"))
    print(f"wrote {a.out}: {len(out)} scenarios, {n_req} draft requests")


def natural_prompts(target):
    import bench_spec
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(target)
    res = []
    for v in range(1, 40):
        for kind in bench_spec.KINDS:
            text = tok.apply_chat_template([{"role": "user", "content": bench_spec.prompt(kind, v) + " /no_think"}],
                                           add_generation_prompt=True, tokenize=False, enable_thinking=False)
            res.append(tok.encode(text, add_special_tokens=False))
    return res


def scenario(ref, ref16, rng, a, vocab, prompts):
    ops, seqs, made, n_req = [], {}, 0, 0
    rand_tok = lambda: rng.randrange(1, vocab)  # noqa: E731

    def expect(ctx, nxt, k, taps):
        """{expect, margins} for an f32 cache, {expect_bf16, margins_bf16} for a bf16 one"""
        ids, margins = ref.drafts(ctx, nxt, k, taps=taps[: len(ctx)] if taps is not None else None)
        ids16, margins16 = ref16.drafts(ctx, nxt, k)
        return {"expect": ids, "margins": [round(m, 5) for m in margins], "expect_bf16": ids16,
                "margins_bf16": [round(m, 5) for m in margins16]}

    def cut(d):
        """where a prompt leaves a donor's context: on the page grid half the time (the cached page before it
        then holds a different next token: fullm keys pages with it), else off it"""
        m = rng.randrange(PAGE_GRID + 1, len(d))
        return m - m % PAGE_GRID if rng.random() < 0.5 else m - (m % PAGE_GRID == 0)

    def pick_k(s):
        if s["ctx_len_after"] >= len(s["prompt"]) or rng.random() < 0.25:
            return rng.choice([0, 1, 1, 2, 3, 3, 4, 5, 6, a.k_max])
        return 0

    while made < a.seqs or seqs:
        # new sequences while there is room
        while made < a.seqs and len(seqs) < a.live and (not seqs or rng.random() < 0.5):
            name = f"s{made}"
            made += 1
            donors = [s for s in seqs.values() if len(s["ctx"]) > 2 * PAGE_GRID + 2]
            r = rng.random()
            shared = bool(donors) and r < 0.3
            if prompts:
                p = list(rng.choice(prompts))
                if shared:  # a donor's context, then another prompt's tail
                    d = rng.choice(donors)["ctx"]
                    p = d[:cut(d)] + p[len(p) // 2:]
            elif shared:
                d = rng.choice(donors)["ctx"]
                p = d[:cut(d)] + [rand_tok() for _ in range(rng.randrange(1, 24))]
            elif r < 0.4:
                p = [rand_tok()]
            else:
                p = [rand_tok() for _ in range(rng.randrange(2, a.max_prompt + 1))]
            s = {"prompt": p, "ctx": [], "next": None, "drafts": [], "left": rng.randrange(1, a.decodes + 1)}
            seqs[name] = s
            if shared or rng.random() < 0.5:
                # a prefix-cache lookup, the rest of the prompt (in step-sized chunks, the head reading each), then
                # the head drafts
                taps, logits = ref.target_pass(p)
                nxt = int(logits[-1].argmax()) if prompts or rng.random() < 0.5 else rand_tok()
                s["ctx"], s["next"], s["ctx_len_after"] = p[:], nxt, len(p)
                k = pick_k(s)
                e = expect(p, nxt, k, taps)
                s["drafts"] = e["expect"]
                n_req += 1
                ops.append({"op": "alloc_prefill", "seq": name, "prompt": p, "next": nxt, "k": k, **e})
            else:
                ops.append({"op": "alloc", "seq": name})
        # a step over some of them
        names = [n for n in seqs]
        rng.shuffle(names)
        names = names[: rng.randrange(1, len(names) + 1)]
        budget, chunks = a.step_tokens, []
        for n in names:
            s = seqs[n]
            if budget == 0:
                break
            fed = len(s["ctx"])
            if fed < len(s["prompt"]):
                size = min(rng.choice([1, 2, 3, 4, 5, 8, 13, 16, 17, 24, 31, 32]), len(s["prompt"]) - fed, budget)
                toks, draft = s["prompt"][fed:fed + size], 0
            else:
                d = min(rng.choice([0, 0, 1, 2, 3, 4, 5, 6]), budget - 1)
                if prompts or rng.random() < 0.7:
                    d = min(d, len(s["drafts"]))
                    extra = s["drafts"][:d]
                else:
                    extra = [rand_tok() for _ in range(d)]
                toks, draft = [s["next"]] + extra, len(extra)
            budget -= len(toks)
            chunks.append({"seq": n, "tokens": toks, "draft": draft})
        ops.append({"op": "step", "chunks": chunks})
        reqs = []
        for c in chunks:
            s = seqs[c["seq"]]
            full = s["ctx"] + c["tokens"]
            taps, logits = ref.target_pass(full)
            base = len(s["ctx"])
            if c["draft"] > 0:
                if prompts:  # greedy verification
                    keep = 0
                    while keep < c["draft"] and c["tokens"][1 + keep] == int(logits[base + keep].argmax()):
                        keep += 1
                else:
                    keep = rng.randrange(0, c["draft"] + 1)
                ops.append({"op": "accept", "seq": c["seq"], "n": keep})
                s["ctx"] = full[: base + 1 + keep]
                row = base + keep
                nxt = int(logits[row].argmax()) if prompts or rng.random() < 0.5 else rand_tok()
                s["left"] -= 1
            else:
                s["ctx"] = full
                if len(s["ctx"]) < len(s["prompt"]):
                    nxt = s["prompt"][len(s["ctx"])]
                else:
                    nxt = int(logits[-1].argmax()) if prompts or rng.random() < 0.5 else rand_tok()
                    if base >= len(s["prompt"]):
                        s["left"] -= 1
            s["next"], s["ctx_len_after"] = nxt, len(s["ctx"])
            k = pick_k(s)
            e = expect(s["ctx"], nxt, k, taps)
            s["drafts"] = e["expect"]
            reqs.append({"seq": c["seq"], "next": nxt, "k": k, **e})
            n_req += 1
        ops.append({"op": "draft", "reqs": reqs})
        for n in [n for n, s in seqs.items() if s["left"] <= 0 and len(s["ctx"]) >= len(s["prompt"])]:
            ops.append({"op": "free", "seq": n})
            del seqs[n]
    return ops, n_req


if __name__ == "__main__":
    main()
