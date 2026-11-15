#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""smoke_test.py - send a fixed set of test prompts to an OpenAI-compatible server and score what can be scored.

Categories (--only picks some):
  short     open prompts, English and Chinese: the reply must be non-empty and not degenerate (a unit of 1-12
            characters repeated 8 times in a row, the shape of a broken KV cache or attention)
  math      arithmetic and word problems with one numeric answer
  location  capitals, landmarks, geography, with one place-name answer
  fact      short facts and instruction following (exact answers)
  long      needle in a haystack: a passphrase hidden at a known depth of ~1k-20k tokens of real text
            (--context-text, default README.md), the reply must contain it; plus one long summary
  gen       a long generation (decode rate)
  sampling  math and location questions at temperature 0.7, top_p 0.9: still right, not degenerate
  stop      OpenAI's `stop`: counting to 20 with stop ", 10" must end at 9 (a server that ignores stop
            strings counts on); a Chinese count against multi-byte stop strings
  thinking  Qwen3's thinking switch: chat_template_kwargs enable_thinking=false gives no reasoning and the
            answer; left on (no /no_think), the reply reasons (reasoning_content, or a <think> block in the
            content from a server without a reasoning parser) and then answers
  determinism  the same prompt at temperature 0: three times in a row, and 8 copies at once vs alone; and a
            prompt right after another of the same token length vs after one of a different length (a
            stale per-step buffer shows up only when consecutive steps have the same shape)

Every request is a streamed chat completion (temperature 0 unless the category says otherwise), timed (TTFT,
decode tok/s). Qwen3 thinks by default: --suffix (default " /no_think") is appended to each user message but the
thinking category's; pass --suffix "" with --think-tokens N to let every reply think. Only the answer is scored:
reasoning_content is kept apart, and a <think>...</think> block in the content is removed.

Usage: smoke_test.py [--url http://localhost:8000] [--model qwen3] [--only math,location] [--json out.json]
                     [--dump DIR]   (write each request body as DIR/NAME.json for `curl -d @DIR/NAME.json`)
Standard library only.
"""
import argparse
import json
import os
import random
import re
import statistics
import threading
import time
import urllib.request

SHORT = [
    ("hello", "Hello! Who are you?"),
    ("haiku", "Write a haiku about autumn rain."),
    ("python_tips", "Give me three tips for writing clean Python code."),
    ("hash_table", "Explain what a hash table is in two sentences."),
    ("great_wall_zh", "用三句话介绍一下长城。"),
    ("tea_zh", "推荐一种适合夏天喝的茶，并说明理由。"),
]

# (name, prompt, accepted answers); the answer must not appear in the prompt
MATH = [
    ("mul", "What is 17 * 23?", ["391"]),
    ("add", "What is 1234 + 5678?", ["6912"]),
    ("div", "What is 156 divided by 12?", ["13"]),
    ("pow", "What is 2 to the power of 10?", ["1024"]),
    ("sqrt", "What is the square root of 169?", ["13"]),
    ("train", "A train travels at 60 km/h for 2.5 hours. How many kilometers does it travel?", ["150"]),
    ("discount", "A shirt costs $40 and is discounted by 25%. What is the sale price in dollars?", ["30"]),
    ("percent", "What is 15% of 80?", ["12"]),
    ("seconds", "How many seconds are in one hour?", ["3600", "3,600"]),
    ("sequence", "What is the next number in the sequence 2, 4, 8, 16, ...?", ["32"]),
    ("apples_zh", "小明有5个苹果，又买了7个，吃掉了3个，还剩几个？", ["9", "九"]),
    ("chickens_zh", "鸡兔同笼，共有10个头，28只脚，兔子有几只？", ["4", "四"]),
]

LOCATION = [
    ("france", "What is the capital of France?", ["Paris"]),
    ("japan", "What is the capital of Japan?", ["Tokyo"]),
    ("australia", "What is the capital of Australia?", ["Canberra"]),
    ("canada", "What is the capital of Canada?", ["Ottawa"]),
    ("eiffel", "In which country is the Eiffel Tower?", ["France"]),
    ("golden_gate", "In which city is the Golden Gate Bridge?", ["San Francisco"]),
    ("kenya", "On which continent is Kenya?", ["Africa"]),
    ("ocean", "What is the largest ocean on Earth?", ["Pacific"]),
    ("cairo", "Which river flows through Cairo?", ["Nile"]),
    ("rio", "In which country is the city of Rio de Janeiro?", ["Brazil"]),
    ("everest", "Mount Everest lies on the border of Nepal and which other country?", ["China", "Tibet"]),
    ("china_zh", "中国的首都是哪个城市？", ["北京", "Beijing"]),
    ("japan_zh", "日本的首都是哪里？", ["东京", "東京", "Tokyo"]),
    ("west_lake_zh", "西湖位于中国哪个城市？", ["杭州", "Hangzhou"]),
]

FACT = [
    ("gold", "What is the chemical symbol for gold?", ["Au"]),
    ("leap_year", "How many days are in a leap year?", ["366"]),
    ("boiling", "What is the boiling point of water at sea level in degrees Celsius?", ["100"]),
    ("continents", "How many continents are there?", ["7", "seven"]),
    ("count", "Count from 1 to 20, separated by commas, and write nothing else.", ["COUNT20"]),
    ("yes", "Reply with only the word yes, in lowercase, and nothing else.", ["EXACT:yes"]),
    ("reverse", "Spell the word 'stressed' backwards. Reply with the word only.", ["desserts"]),
]

NEEDLE_WORDS = ["maple", "cobalt", "harbor", "falcon", "quartz", "willow", "ember", "glacier"]
# (words of context, depth of the needle)
LONG = [(800, 0.5), (4000, 0.25), (9000, 0.5), (14000, 0.1), (14000, 0.9)]

FILLER = ("The history of computing is a story of layered abstractions, each hiding the machinery below "
          "it so that the next generation of engineers could build something larger. ").split()

THINK = re.compile(r"<think>.*?</think>", re.S)
REPEAT = re.compile(r"(.{1,12}?)\1{7,}", re.S)


def clean(text):
    return THINK.sub("", text or "").strip()


def degenerate(text):
    return bool(REPEAT.search(clean(text)))


def chat(url, model, content, max_tokens, temperature=0.0, extra=None):
    """one streamed chat completion: the answer (text) and reasoning, TTFT, total time, completion tokens, finish"""
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": temperature, "stream": True, "stream_options": {"include_usage": True}, **(extra or {})}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first, parts, thought, n, finish, prompt_tokens = None, [], [], None, None, None
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                chunk = json.loads(line[5:])
                if chunk.get("usage"):
                    n = chunk["usage"].get("completion_tokens", n)
                    prompt_tokens = chunk["usage"].get("prompt_tokens", prompt_tokens)
                for ch in chunk.get("choices") or []:
                    d = ch.get("delta", {})
                    r, c = d.get("reasoning_content") or "", d.get("content") or ""
                    if (r or c) and first is None:
                        first = time.perf_counter()
                    thought.append(r)
                    parts.append(c)
                    finish = ch.get("finish_reason") or finish
    except urllib.error.HTTPError as e:
        return {"text": "", "reasoning": "", "error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}",
                "ttft": 0, "total": 0, "tokens": 0, "finish": None, "prompt_tokens": None, "rate": 0.0}
    t1 = time.perf_counter()
    n = n if n is not None else len(parts)
    rate = (n - 1) / (t1 - first) if first and n > 1 and t1 > first else 0.0
    return {"text": "".join(parts), "reasoning": "".join(thought), "error": None, "ttft": (first or t1) - t0,
            "total": t1 - t0, "tokens": n, "finish": finish, "prompt_tokens": prompt_tokens, "rate": rate}


def score(reply, accepted):
    r = clean(reply)
    for a in accepted:
        if a == "COUNT20":
            if re.sub(r"\s", "", r).rstrip(".") == ",".join(str(i) for i in range(1, 21)):
                return True
        elif a.startswith("EXACT:"):
            if r.strip().strip(".!").lower() == a[6:]:
                return True
        elif re.fullmatch(r"[\d,]+", a):
            if re.search(rf"(?<![\d.,]){re.escape(a)}(?![\d])", r):
                return True
        elif a.lower() in r.lower():
            return True
    return False


def thinking_of(res):
    """the reply's reasoning: reasoning_content, else a <think> block in its content"""
    m = re.search(r"<think>(.*?)</think>", res["text"], re.S)
    return (res["reasoning"] or (m.group(1) if m else "")).strip()


def answer_ok(res, accepted):
    """a scored reply contains an accepted answer; an open one is non-empty and not degenerate; THINK_ON /
    THINK_OFF first also require reasoning, or none"""
    if res["error"] or degenerate(res["text"]) or degenerate(res["reasoning"]):
        return False
    if accepted and accepted[0] in ("THINK_ON", "THINK_OFF"):
        if bool(thinking_of(res)) != (accepted[0] == "THINK_ON"):
            return False
        accepted = accepted[1:]
    return score(res["text"], accepted) if accepted is not None else bool(clean(res["text"]))


def haystack(source, words, depth, rng):
    word, num = rng.choice(NEEDLE_WORDS), rng.randrange(1000, 10000)
    secret = f"{word}-{num}"
    start = rng.randrange(len(source))
    body = [source[(start + j) % len(source)] for j in range(words)]
    at = int(words * depth)
    body.insert(at, f"\n\nThe secret passphrase for the archive room is {secret}.\n\n")
    prompt = ("Read the following document carefully.\n\n" + " ".join(body) +
              "\n\nWhat is the secret passphrase for the archive room? Reply with the passphrase only.")
    return prompt, secret


def show(tag, name, prompt, res, note=""):
    if res["error"]:
        print(f"[{tag}] {name:16s} ERROR {res['error']}")
        return
    reply = clean(res["text"]).replace("\n", " ")
    q = prompt if len(prompt) < 60 else f"<{res['prompt_tokens'] or '?'} tokens> ..." + prompt[-50:]
    print(f"[{tag}] {name:16s} {q[:70]!r:72s} -> {reply[:90]!r} "
          f"(TTFT {res['ttft'] * 1e3:.0f} ms, {res['tokens']} tok, {res['rate']:.0f} tok/s, {res['finish']}){note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="qwen3")
    ap.add_argument("--only", default="short,math,location,fact,long,gen,sampling,stop,thinking,determinism")
    ap.add_argument("--suffix", default=" /no_think", help="appended to every user message (Qwen3's no-think switch)")
    ap.add_argument("--think-tokens", type=int, default=0, help="extra max_tokens for a thinking reply")
    text = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "README.md")
    ap.add_argument("--context-text", default=text, help="real text for the long prompts (default: README.md, "
                    "repeated as needed)")
    ap.add_argument("--json", help="write every request and result here")
    ap.add_argument("--dump", help="write each request body as DIR/NAME.json (curl -d @...) and exit")
    a = ap.parse_args()
    url, only = a.url.rstrip("/"), set(a.only.split(","))
    rng = random.Random(7)
    try:
        source = open(a.context_text, encoding="utf-8", errors="replace").read().split() or FILLER
    except (OSError, TypeError):
        source = FILLER
    extra = a.think_tokens

    # (category, name, prompt, max_tokens, accepted answers or None[, request fields, append the suffix])
    cases = []
    cases += [("short", n, p, 300 + extra, None) for n, p in SHORT]
    cases += [("math", n, p + " Answer briefly.", 64 + extra, acc) for n, p, acc in MATH]
    cases += [("location", n, p + " Answer briefly.", 48 + extra, acc) for n, p, acc in LOCATION]
    cases += [("fact", n, p, 128 + extra, acc) for n, p, acc in FACT]
    sampled = {"temperature": 0.7, "top_p": 0.9}
    cases += [("sampling", f"{n}_t0.7", p + " Answer briefly.", 64 + extra, acc, {**sampled, "seed": 11 + i})
              for i, (n, p, acc) in enumerate(MATH[:4] + LOCATION[:4] + LOCATION[11:13])]
    cases += [("stop", "count_to_9", FACT[4][1], 128 + extra, ["EXACT:1, 2, 3, 4, 5, 6, 7, 8, 9"], {"stop": [", 10"]}),
              # Qwen3 int4 at temperature 0 counts in Chinese numerals here: either form must stop at 4
              ("stop", "count_to_4_zh", "从1数到10，用顿号分隔，不要写别的。", 128 + extra, ["EXACT:1、2、3、4", "EXACT:一、二、三、四"],
               {"stop": ["、5", "、五"]})]
    q = "What is 17 * 23? Answer briefly."
    cases += [("thinking", "kwargs_off", q, 64, ["THINK_OFF", "391"],
               {"chat_template_kwargs": {"enable_thinking": False}}, False),
              ("thinking", "on", q, 2048, ["THINK_ON", "391"], {}, False),
              ("thinking", "kwargs_off_zh", "中国的首都是哪个城市？", 64, ["THINK_OFF", "北京", "Beijing"],
               {"chat_template_kwargs": {"enable_thinking": False}}, False)]
    for words, depth in LONG:
        prompt, secret = haystack(source, words, depth, rng)
        cases.append(("long", f"needle_{words}w_{int(depth * 100)}%", prompt, 32 + extra, [secret]))
    start = rng.randrange(len(source))
    cases.append(("long", "summary_6000w", "Read the following document.\n\n" +
                  " ".join(source[(start + j) % len(source)] for j in range(6000)) +
                  "\n\nSummarize the document above in three bullet points.", 300 + extra, None))
    cases.append(("gen", "story_400w", "Write a story of about 400 words about a robot that learns to paint.",
                  800 + extra, None))
    cases = [c + ({}, True)[len(c) - 5:] for c in cases if c[0] in only]

    if a.dump:
        os.makedirs(a.dump, exist_ok=True)
        for cat, name, prompt, mt, _, fields, suffix in cases:
            body = {"model": a.model, "messages": [{"role": "user", "content": prompt + (a.suffix if suffix else "")}],
                    "max_tokens": mt, "temperature": 0, **fields}
            with open(os.path.join(a.dump, f"{cat}_{name}.json"), "w", encoding="utf-8") as f:
                json.dump(body, f, ensure_ascii=False)
        print(f"wrote {len(cases)} request bodies to {a.dump}; e.g. curl {url}/v1/chat/completions "
              f"-H 'Content-Type: application/json' -d @{a.dump}/math_mul.json")
        return

    log, tally = [], {}
    chat(url, a.model, "Hi" + a.suffix, 4)  # warm-up
    for cat, name, prompt, mt, acc, fields, suffix in cases:
        fields = dict(fields)
        res = chat(url, a.model, prompt + (a.suffix if suffix else ""), mt, fields.pop("temperature", 0.0), fields)
        ok = answer_ok(res, acc)
        t = tally.setdefault(cat, [0, 0, [], []])
        t[0] += ok
        t[1] += 1
        t[2].append(res["ttft"])
        if res["rate"]:
            t[3].append(res["rate"])
        show("PASS" if ok else "FAIL", name, prompt, res, "" if ok or acc is None else f"  expected {acc}")
        log.append({"category": cat, "name": name, "prompt": prompt, "max_tokens": mt, "accepted": acc, "pass": ok, **res})

    if "determinism" in only:
        p = "Explain how a bicycle stays upright, in about 80 words." + a.suffix
        seq = [chat(url, a.model, p, 160)["text"] for _ in range(3)]
        print(f"[{'PASS' if len(set(seq)) == 1 else 'FAIL'}] repeat x3       identical at temperature 0: {len(set(seq)) == 1}")
        outs = [None] * 8
        def go(i):
            outs[i] = chat(url, a.model, p, 160)["text"]
        th = [threading.Thread(target=go, args=(i,)) for i in range(8)]
        [x.start() for x in th]
        [x.join() for x in th]
        same = sum(o == seq[0] for o in outs)
        note = "" if same == 8 else " (a batch of 8 rounds differently from a batch of 1; greedy text can drift late)"
        print(f"[{'PASS' if same == 8 else 'INFO'}] concurrent x8   {same}/8 identical to the solo reply{note}")
        # France and Japan are one token each: the two prompts have the same length, Eiffel's does not
        f, j = "What is the capital of France? Answer briefly.", "What is the capital of Japan? Answer briefly."
        e = "In which country is the Eiffel Tower? Answer briefly."
        chat(url, a.model, e + a.suffix, 16)
        after_other = chat(url, a.model, j + a.suffix, 16)["text"]
        chat(url, a.model, f + a.suffix, 16)
        after_same = chat(url, a.model, j + a.suffix, 16)["text"]
        ok = after_same == after_other and not degenerate(after_same)
        print(f"[{'PASS' if ok else 'FAIL'}] same length      {clean(after_same)[:30]!r} after a same-length prompt, "
              f"{clean(after_other)[:30]!r} after another length")
        log.append({"category": "determinism", "repeat_identical": len(set(seq)) == 1, "concurrent_identical": same,
                    "same_length_ok": ok})

    print("\nsummary (pass / total, TTFT median, decode tok/s median):")
    for cat, (ok, n, tt, rr) in tally.items():
        print(f"  {cat:9s} {ok:2d} / {n:2d}   TTFT {statistics.median(tt) * 1e3:6.0f} ms   "
              f"{statistics.median(rr) if rr else 0:5.0f} tok/s")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
