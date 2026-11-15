#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""bench_spec.py - speculative decoding against an OpenAI-compatible server: greedy chat requests whose replies
repeat their prompts to different degrees (an agent's edits and quotes, down to an open question), at several
concurrencies.

Per server (--url, one or more): output tokens/s aggregate and per request, by concurrency and over every prompt
kind, plus the drafts verified and accepted when the server reports them (fullm's /metrics spec_drafted and
spec_accepted). With two servers or more, every reply is first compared with the first server's (one request at
a time): speculation must not change a token on a backend whose rows do not depend on the batch, and may change a
few late tokens where they do. Requests are greedy unless --temperature is set; then every request carries a seed
of its own (the same on every server), so sampled replies compare too.

Prompt kinds: rename (a Python module with a parameter renamed: most of the reply is the prompt), edit (a config
file with one value changed), json (a JSON document reindented), repeat (a paragraph word for word), summary (a
log summarised: a few phrases repeat), open (an open question: nothing to copy). Every request gets a variant of
its own (another name, value, paragraph order, topic and style): identical greedy requests running together would
generate the same tokens in lockstep, and a batch of identical rows touches a fraction of the experts a real one
does; and a server that drafts from earlier replies would find the very reply. The timed runs use variants from
--variant-offset on; give separate runs against one server disjoint ranges.
Qwen3's /no_think is appended (--suffix "" to drop it). Standard library only.

Usage: bench_spec.py --url http://localhost:8000 [--url http://localhost:8001 ...] [--model qwen3]
                     [--concurrency 1,4,8,16] [--requests 2] [--max-tokens 256] [--kinds rename,edit,...]
                     [--temperature 0 --top-p 1 --top-k 0 --seed 1] [--json OUT]
"""

import argparse
import json
import statistics
import threading
import time
import urllib.request

CODE = '''import heapq
from dataclasses import dataclass


@dataclass
class Edge:
    to: int
    weight: float


def shortest_paths(graph, source):
    dist = {source: 0.0}
    queue = [(0.0, source)]
    while queue:
        d, node = heapq.heappop(queue)
        if d > dist.get(node, float("inf")):
            continue
        for edge in graph.get(node, []):
            nd = d + edge.weight
            if nd < dist.get(edge.to, float("inf")):
                dist[edge.to] = nd
                heapq.heappush(queue, (nd, edge.to))
    return dist


def reachable(graph, source):
    return set(shortest_paths(graph, source))


def farthest(graph, source):
    dist = shortest_paths(graph, source)
    node = max(dist, key=dist.get)
    return node, dist[node]
'''

CONFIG = '''[server]
host = "0.0.0.0"
port = 8080
workers = 8
timeout_s = 30

[database]
url = "postgres://app@db.internal:5432/app"
pool_size = 16
statement_timeout_ms = 5000

[cache]
backend = "redis"
url = "redis://cache.internal:6379/0"
ttl_s = 600

[logging]
level = "info"
format = "json"
'''

DOC = {"service": "checkout", "region": "us-central1", "replicas": 3,
       "containers": [{"name": "api", "image": "checkout-api:1.14.2", "cpu": "500m", "memory": "512Mi",
                       "env": {"LOG_LEVEL": "info", "DB_POOL": "16"}},
                      {"name": "worker", "image": "checkout-worker:1.14.2", "cpu": "250m", "memory": "256Mi",
                       "env": {"QUEUE": "orders", "CONCURRENCY": "4"}}],
       "probes": {"liveness": "/healthz", "readiness": "/ready"}}

PARA = ("The river leaves the mountains at the old stone bridge, where the valley opens into wide fields. In summer "
        "the water is low and the children of the village walk across on the rocks; in spring it covers the meadows "
        "up to the church wall. The council has discussed a new embankment for years, but every plan has foundered "
        "on the cost of the land and on the farmers, who say the floods bring the soil that makes the fields worth "
        "farming. Last year an engineer from the city proposed a lower wall with gates that open in March.")

LOG = "\n".join(
    f"2026-11-08T10:{m:02d}:{s:02d}Z {lvl} worker-{w} {msg}"
    for m, s, lvl, w, msg in [
        (0, 12, "INFO", 1, "job 4411 started: resize images batch=64"),
        (0, 47, "WARN", 2, "job 4412 retry 1/3: upstream timeout after 30s"),
        (1, 3, "INFO", 1, "job 4411 finished in 51s"),
        (1, 30, "ERROR", 2, "job 4412 failed: upstream timeout after 30s (3/3)"),
        (2, 2, "INFO", 3, "job 4413 started: export report region=eu"),
        (2, 40, "WARN", 3, "job 4413 slow query 12.4s: SELECT * FROM orders WHERE region = 'eu'"),
        (3, 15, "INFO", 3, "job 4413 finished in 73s"),
        (3, 16, "INFO", 1, "job 4414 started: resize images batch=64"),
        (4, 1, "ERROR", 1, "job 4414 failed: out of memory (rss 3.9 GiB, limit 4 GiB)"),
    ])

NAMES = ["adjacency", "neighbours", "network", "links", "topology", "edges_by_node", "routes", "connections",
         "mesh", "outgoing", "web", "lattice", "wiring", "paths_from", "succ", "fanout"]
LEVELS = ["debug", "trace", "warn", "error"]
STYLES = ["in two paragraphs", "in three short bullet points", "for a ten-year-old", "in one paragraph, precisely",
          "as a short dialogue between a student and a teacher", "listing the steps in order"]
TOPICS = ["how a refrigerator keeps food cold", "why the sky is blue", "how an airplane wing produces lift",
          "how vaccines train the immune system", "why bread rises", "how a bicycle stays upright",
          "how tides are caused", "how a transistor switches", "why ice floats", "how noise-cancelling headphones work",
          "how a compass finds north", "why leaves change colour in autumn", "how a battery stores energy",
          "how GPS finds a position", "why the moon has phases", "how a microwave oven heats food"]


def prompt(kind, v):
    """the v-th variant of a prompt kind: no two variants of a kind are the same prompt"""
    if kind == "rename":
        name = NAMES[v % len(NAMES)] + (f"_{v // len(NAMES)}" if v >= len(NAMES) else "")
        return (f"Here is a Python module:\n```python\n{CODE}```\nRename the parameter `graph` to "
                f"`{name}` everywhere and output the whole module, nothing else.")
    if kind == "edit":
        return (f"Here is a config file:\n```toml\n{CONFIG}```\nChange the server port to {9000 + v} and the log level "
                f"to \"{LEVELS[v % len(LEVELS)]}\". Output the whole file, nothing else.")
    if kind == "json":
        doc = json.loads(json.dumps(DOC))
        doc["service"] = f"checkout-{v}"
        doc["replicas"] = 2 + v % 7
        doc["containers"][0]["image"] = f"checkout-api:1.{14 + v}.2"
        return "Reformat this JSON with two-space indentation and output only the JSON:\n" + json.dumps(doc)
    if kind == "repeat":
        sents = [x.strip() + "." for x in PARA.split(".") if x.strip()]
        k = v % len(sents)
        return ("Repeat the following text exactly, word for word:\n\n" + " ".join(sents[k:] + sents[:k])
                + f" This is copy number {v}.")
    if kind == "summary":
        log = LOG.replace("job 44", f"job {44 + v}").replace("worker-", f"w{v}-")
        return f"Summarise what went wrong in this log, one line per job:\n\n{log}"
    if kind == "open":
        style = STYLES[(v // len(TOPICS)) % len(STYLES)] + (f" (version {v})" if v >= len(TOPICS) * len(STYLES) else "")
        return f"Explain {TOPICS[v % len(TOPICS)]}, {style}."
    raise ValueError(kind)


KINDS = ["rename", "edit", "json", "repeat", "summary", "open"]


SAMPLING = {"temperature": 0.0}


def chat(url, model, content, max_tokens, seed=0):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": content}],
                       "max_tokens": max_tokens, **SAMPLING, "seed": seed}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    return {"text": out["choices"][0]["message"].get("content") or "",
            "tokens": out["usage"]["completion_tokens"], "s": time.perf_counter() - t}


COUNTERS = ("spec_drafted", "spec_accepted", "spec_drafted_global", "spec_accepted_global", "spec_chunks",
            "spec_drafted_eagle", "spec_accepted_eagle", "spec_chunks_eagle")


def counters(url):
    """fullm's speculation counters (COUNTERS) from its /metrics, or None"""
    try:
        with urllib.request.urlopen(url + "/metrics", timeout=30) as r:
            e = json.load(r).get("engine", {})
        if e.get("spec_drafted") is None:
            return None
        return tuple(e.get(k, 0) for k in COUNTERS)
    except Exception:  # noqa: BLE001 (another engine's /metrics is Prometheus text)
        return None


def run(url, a, kinds, conc):
    """conc workers, each sending a.requests requests per prompt kind (offset by worker), each a variant of its own"""
    jobs = [(w, i) for w in range(conc) for i in range(a.requests * len(kinds))]
    recs, lock = [], threading.Lock()

    def worker(w):
        for i in range(a.requests * len(kinds)):
            kind = kinds[(w + i) % len(kinds)]
            v = a.variant_offset + 1 + w + conc * (i // len(kinds))
            r = chat(url, a.model, prompt(kind, v) + a.suffix, a.max_tokens, a.seed + v)
            r["kind"] = kind
            with lock:
                recs.append(r)

    c0 = counters(url)
    t = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(w,)) for w in range(conc)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.perf_counter() - t
    c1 = counters(url)
    toks = sum(r["tokens"] for r in recs)
    per = statistics.median(r["tokens"] / r["s"] for r in recs)
    line = f"   concurrency {conc:2d}: {toks / wall:7.1f} tok/s aggregate, {per:6.1f} per request (median)"
    spec = None
    if c0 and c1:
        spec = {k[len("spec_"):]: c1[i] - c0[i] for i, k in enumerate(COUNTERS)}
        d, k, dg, kg, ch = (spec[x] for x in ("drafted", "accepted", "drafted_global", "accepted_global", "chunks"))
        line += f" | drafted {d}, accepted {k}" + (f" ({100 * k / d:.0f}%, length {1 + k / ch:.2f})" if ch else "")
        if dg:
            line += f"; from other replies {dg}, {kg} kept"
        de, ke, che = spec["drafted_eagle"], spec["accepted_eagle"], spec["chunks_eagle"]
        if che:
            line += f"; head {de}, {ke} kept (length {1 + ke / che:.2f})"
    print(line)
    assert len(recs) == len(jobs)
    return {"concurrency": conc, "wall": wall, "tokens": toks, "per_request": per, "spec": spec, "records": recs}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", action="append", required=True, help="a server; repeat to compare several")
    ap.add_argument("--model", default="qwen3")
    ap.add_argument("--concurrency", default="1,4,8,16")
    ap.add_argument("--requests", type=int, default=2, help="requests per worker and prompt kind")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--kinds", default=",".join(KINDS), help="comma-separated prompt kinds")
    ap.add_argument("--suffix", default=" /no_think", help="appended to every prompt (Qwen3's soft switch)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1, help="request v carries seed + v")
    ap.add_argument("--variant-offset", type=int, default=0,
                    help="first prompt variant of the timed runs (runs whose ranges do not overlap send no prompt twice)")
    ap.add_argument("--json", help="write every request here")
    a = ap.parse_args()
    kinds = a.kinds.split(",")
    SAMPLING.update(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k)
    prompts = [(k, prompt(k, 0) + a.suffix) for k in kinds]
    chat(a.url[0], a.model, "Hello", 8)  # warm-up
    out = {"args": vars(a), "servers": []}
    if len(a.url) > 1:
        print("replies, one request at a time, against the first server's:")
        ref = {k: chat(a.url[0], a.model, p, a.max_tokens, a.seed) for k, p in prompts}
        for url in a.url[1:]:
            diff = []
            for k, p in prompts:
                r = chat(url, a.model, p, a.max_tokens, a.seed)
                x, y = ref[k]["text"], r["text"]
                if x != y:
                    n = next((i for i, (c, d) in enumerate(zip(x, y)) if c != d), min(len(x), len(y)))
                    diff.append(f"{k} (after {n} of {len(x)} characters)")
            print(f"   {url}: {len(prompts) - len(diff)}/{len(prompts)} identical" + (f"; differ: {', '.join(diff)}" if diff else ""))
    for url in a.url:
        print(f"{url}:")
        for k, p in prompts:  # warm-up: a step shape's first sightings run without its CUDA graph
            chat(url, a.model, p, a.max_tokens, a.seed)
        for k, p in prompts:  # what each kind yields one request at a time
            c0 = counters(url)
            r = chat(url, a.model, p, a.max_tokens, a.seed)
            c1 = counters(url)
            extra = ""
            if c0 and c1 and c0[0] is not None:
                d, acc, dg, kg = (c1[i] - c0[i] for i in range(4))
                extra = f", drafted {d}, accepted {acc}" + (f" (other replies {dg}, {kg} kept)" if dg else "")
            print(f"   {k:8s} {r['tokens']:4d} tokens, {r['tokens'] / r['s']:6.1f} tok/s{extra}")
        runs = [run(url, a, kinds, int(c)) for c in a.concurrency.split(",")]
        out["servers"].append({"url": url, "runs": runs})
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
