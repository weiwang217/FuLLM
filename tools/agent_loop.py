#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""agent_loop.py - a small tool-using agent that calls an OpenAI-compatible server continuously.

Each worker picks a task (each draws its own numbers, cities, SKUs and people, so no two runs are
the same conversation), then loops: POST /v1/chat/completions with the tools, run every tool call
the reply makes (a deterministic mock world: calculator, weather, clocks, prices, exchange rates, a
staff directory), append the results as `tool` messages, and send again, until the model answers
without a call. Each answer is checked against the task's expected values, and with --json-final the
answer is asked for once more as `response_format` json_schema and that JSON is checked too. Workers
run side by side for --duration seconds (or --rounds tasks each), so the server sees what agent
traffic looks like: multi-turn chats whose prefixes repeat, short tool outputs, and tool-call and
structured-output requests mixed with plain ones.

Unlike bench_agent.py (synthetic text through /v1/completions, the same for every engine), this
goes through the chat template and the server's tool-call parser, and counts what an agent
framework would trip on:
  protocol errors  an HTTP error, a finish_reason other than stop / tool_calls / length, a tool
                   call whose arguments are not a JSON object, an unknown tool name, missing or
                   mistyped required arguments, a duplicate call id, a json_schema reply that is not
                   schema-valid JSON
  wrong answers    the final answer lacks an expected value, or the loop ran out of --max-turns
  a warning        replies with calls whose content is only whitespace: harmless to the protocol, but
                   it goes back into the history and moves the prompt off what the model was trained on
Plus request latency (p50 / p95, time to first token when streamed), completion tokens per second
across all workers, and tool calls per task.

--stream mix (the default) streams every other request, so both response paths are exercised (tool
calls are assembled from deltas as OpenAI clients do: id and name once, arguments concatenated).

--suite complex runs agent_flows.py's flows instead: stateful worlds (orders, an outbox, files, calendars),
6-12 tool calls per task with nested array / object arguments, failing calls to retry, long tool outputs
(histories past 10k tokens) and a sub-agent; each is graded on the state it leaves and its answer, and
the table adds each task's largest prompt (tokens). --suite all mixes both.

Examples:
  agent_loop.py --url http://127.0.0.1:8000 --concurrency 16 --duration 120
  agent_loop.py --url http://127.0.0.1:8000 --suite complex --concurrency 8 --duration 300
  agent_loop.py --url http://127.0.0.1:8000 --concurrency 2 --rounds 2 --verbose   # a quick check
  agent_loop.py ... --think        # Qwen3 thinking on (chat_template_kwargs.enable_thinking)
--think changes two defaults. --max-tokens 4096 (else 1024): Qwen3-4B's reasoning runs past 1024
tokens on the multi-step tasks, and a reply cut off there counts as a wrong answer. --temperature
0.6 (else 0, deterministic): greedy reasoning loops (one currency task repeated an 80-character
passage 28 times to the 4096 limit), and Qwen3's card advises against greedy decoding with thinking.
vLLM needs --enable-auto-tool-choice --tool-call-parser hermes for Qwen3's tool calls. Standard
library only. Exit status 1 when any protocol error occurred.
"""

import argparse
import ast
import json
import operator
import random
import sys
import threading
import time
import os
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_flows  # noqa: E402  the complex flows (--suite complex)

# ------------------------------------------------------------------ the mock world

# Many entities, so that a configuration's few borderline prompts (ones a model gets right or wrong
# almost every time, whatever the sampling) weigh little in a task's rate.
WEATHER = {
    "paris": (18, "sunny"), "oslo": (7, "rain"), "tokyo": (22, "cloudy"), "new york": (15, "windy"),
    "cairo": (31, "sunny"), "sydney": (25, "clear"), "london": (12, "overcast"), "berlin": (9, "cloudy"),
    "madrid": (24, "sunny"), "rome": (20, "clear"), "lima": (17, "foggy"), "toronto": (4, "snow"),
    "seoul": (11, "clear"), "mumbai": (33, "humid"), "dubai": (36, "sunny"), "singapore": (30, "storms"),
    "nairobi": (23, "clear"), "reykjavik": (2, "windy"), "mexico city": (19, "rain"), "istanbul": (16, "cloudy"),
}
CLOCKS = {
    "paris": "14:30", "oslo": "14:30", "tokyo": "21:30", "new york": "08:30", "cairo": "15:30", "sydney": "23:30",
    "london": "13:30", "berlin": "14:30", "madrid": "14:30", "rome": "14:30", "lima": "07:30", "toronto": "08:30",
    "seoul": "21:30", "mumbai": "18:00", "dubai": "16:30", "singapore": "20:30", "nairobi": "15:30",
    "reykjavik": "12:30", "mexico city": "06:30", "istanbul": "15:30",
}
PRICES = {"A123": 12.5, "B456": 40.0, "C789": 7.25, "D012": 3.75, "E345": 19.5, "F678": 64.0}
RATES = {("USD", "EUR"): 0.92, ("USD", "JPY"): 150.0, ("USD", "GBP"): 0.8, ("USD", "CHF"): 0.9, ("USD", "CAD"): 1.36,
         ("EUR", "USD"): 1.087}
STAFF = {
    "alice chen": {"team": "Inference", "office": "Oslo"},
    "bob martin": {"team": "Kernels", "office": "Tokyo"},
    "carla diaz": {"team": "Serving", "office": "Cairo"},
    "david kim": {"team": "Compilers", "office": "Seoul"},
    "emma rossi": {"team": "Research", "office": "Rome"},
    "farid haddad": {"team": "Platform", "office": "Dubai"},
    "grace liu": {"team": "Kernels", "office": "Singapore"},
    "hugo silva": {"team": "Data", "office": "Lima"},
    "ines moreau": {"team": "Inference", "office": "Paris"},
    "jonas berg": {"team": "Serving", "office": "Berlin"},
    "keiko sato": {"team": "Research", "office": "Sydney"},
    "liam walsh": {"team": "Platform", "office": "Toronto"},
}

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _calc(node):
    if isinstance(node, ast.Expression):
        return _calc(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _calc(node.left), _calc(node.right)
        if isinstance(node.op, ast.Pow) and abs(b) > 64:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_calc(node.operand))
    raise ValueError("only numbers and + - * / % ** ( ) are allowed")


def _num(x):
    return int(x) if isinstance(x, float) and x.is_integer() else round(x, 6)


def calculator(expression):
    try:
        return {"result": _num(_calc(ast.parse(expression.replace("^", "**"), mode="eval")))}
    except Exception as e:  # the model sees the error and may retry
        return {"error": f"cannot evaluate {expression!r}: {e}"}


def get_weather(city):
    w = WEATHER.get(city.strip().lower())
    return {"city": city, "temperature_c": w[0], "condition": w[1]} if w else {"error": f"unknown city {city!r}"}


def get_time(city):
    t = CLOCKS.get(city.strip().lower())
    return {"city": city, "local_time": t} if t else {"error": f"unknown city {city!r}"}


def lookup_price(sku):
    p = PRICES.get(sku.strip().upper())
    return {"sku": sku, "price_usd": p} if p is not None else {"error": f"unknown sku {sku!r}"}


def exchange_rate(base, quote):
    r = RATES.get((base.strip().upper(), quote.strip().upper()))
    return {"base": base, "quote": quote, "rate": r} if r else {"error": f"no rate for {base}/{quote}"}


def find_employee(name):
    e = STAFF.get(name.strip().lower())
    return {"name": name, **e} if e else {"error": f"nobody called {name!r}"}


def tool(fn, desc, **params):
    props = {k: {"type": t, "description": d} for k, (t, d) in params.items()}
    return fn, {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(params)},
        },
    }


TOOLS = dict(
    (spec["function"]["name"], (fn, spec))
    for fn, spec in [
        tool(calculator, "Evaluate an arithmetic expression (numbers, + - * / % ** and parentheses).",
             expression=("string", "the expression, e.g. (3 + 4) * 2")),
        tool(get_weather, "Current weather of a city.", city=("string", "city name, e.g. Paris")),
        tool(get_time, "Current local time of a city.", city=("string", "city name, e.g. Tokyo")),
        tool(lookup_price, "Price of a product in USD by its SKU.", sku=("string", "the SKU, e.g. A123")),
        tool(exchange_rate, "Exchange rate between two currencies (1 base = rate quote).",
             base=("string", "ISO code, e.g. USD"), quote=("string", "ISO code, e.g. EUR")),
        tool(find_employee, "Look up an employee's team and office city.", name=("string", "full name")),
    ]
)
TOOL_SPECS = [spec for _, spec in TOOLS.values()]
JSON_TYPES = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "object": dict, "array": list}

# Tasks: each draws its own values (so runs differ, as real traffic does) and returns the prompt, the values
# the answer must contain (case and thousands separators ignored) and the fewest tool calls a right run makes.


def fmt(x):
    return ("%.4f" % x).rstrip("0").rstrip(".")


def t_calc(r):
    a, b, c = r.randint(100, 9999), r.randint(100, 9999), r.randint(1000, 99999)
    return f"Use the calculator to compute ({a} * {b}) + {c}. Reply with the number.", [str(a * b + c)], 1


def t_weather_compare(r):
    x, y = r.sample(sorted(WEATHER), 2)
    tx, ty = WEATHER[x][0], WEATHER[y][0]
    warm = x if tx > ty else y
    return (f"Check the weather in {x.title()} and in {y.title()}. Which city is warmer, and by how many degrees "
            f"Celsius?", [warm, str(abs(tx - ty))], 2)


def t_shopping(r):
    (s1, p1), (s2, p2) = r.sample(sorted(PRICES.items()), 2)
    q1, q2 = r.randint(2, 9), r.randint(2, 9)
    return (f"Look up the prices of SKU {s1} and SKU {s2}, then use the calculator for the total cost of {q1} units of "
            f"{s1} and {q2} units of {s2}. Give the total in USD.", [fmt(q1 * p1 + q2 * p2)], 2)


def t_currency(r):
    quote = r.choice(["EUR", "JPY", "GBP", "CHF", "CAD"])
    amount, pct = r.choice([100, 120, 200, 250, 400, 500, 750, 800]), r.choice([5, 10, 12, 15, 20, 25, 30, 40])
    v = round(amount * RATES[("USD", quote)] * pct / 100, 4)
    return (f"Get the USD to {quote} exchange rate, convert {amount} USD to {quote}, then use the calculator to take "
            f"{pct}% of that amount. Give the final amount in {quote}.", [fmt(v)], 1)


def t_multihop(r):
    name = r.choice(sorted(STAFF))
    city = STAFF[name]["office"]
    return (f"Find out which city {name.title()}'s office is in, then tell me the current temperature there in "
            f"Celsius.", [city.lower(), str(WEATHER[city.lower()][0])], 2)


def t_clocks(r):
    x, y = r.sample(sorted(CLOCKS), 2)
    return (f"What time is it right now in {x.title()} and in {y.title()}?",
            [CLOCKS[x].lstrip("0"), CLOCKS[y].lstrip("0")], 2)


AVG_TRIPLES = [t for t in __import__("itertools").combinations(sorted(WEATHER), 3)
               if sum(WEATHER[c][0] for c in t) % 3 == 0]


def t_average(r):
    t = r.choice(AVG_TRIPLES)
    names = ", ".join(c.title() for c in t[:2]) + f" and {t[2].title()}"
    return (f"Get the temperatures in {names}, then use the calculator to compute their average in Celsius.",
            [str(sum(WEATHER[c][0] for c in t) // 3)], 3)


CAPITALS = {"Japan": "Tokyo", "France": "Paris", "Italy": "Rome", "Canada": "Ottawa", "Egypt": "Cairo", "Kenya": "Nairobi",
            "Peru": "Lima", "Norway": "Oslo", "Germany": "Berlin", "Spain": "Madrid", "Iceland": "Reykjavik",
            "South Korea": "Seoul"}


def t_no_tool(r):
    country = r.choice(sorted(CAPITALS))
    return f"What is the capital of {country}? Answer in one word.", [CAPITALS[country].lower()], 0


TASKS = [(f.__name__[2:], f) for f in (t_calc, t_weather_compare, t_shopping, t_currency, t_multihop, t_clocks,
                                       t_average, t_no_tool)]
FLOWS = agent_flows.FLOWS
FLOW_NAMES = {n for n, _ in FLOWS}
SYSTEM = ("You are a helpful assistant with tools. Use a tool whenever it can give you a value instead of guessing it, "
          "and call several tools at once when they do not depend on each other. When you have the answer, reply "
          "briefly and include the numbers.")
FINAL_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

# ------------------------------------------------------------------ the client


class ProtocolError(Exception):
    pass


def post(a, body, stream):
    """One chat completion: (message, finish_reason, usage, latency s, ttft s or None)."""
    body = dict(body, model=a.model, stream=stream)
    if stream:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=a.timeout)
    except urllib.error.HTTPError as e:
        raise ProtocolError(f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}")
    except (urllib.error.URLError, OSError) as e:
        raise ProtocolError(f"connection: {e}")
    if not stream:
        v = json.loads(r.read())
        ch = v["choices"][0]
        return ch["message"], ch.get("finish_reason"), v.get("usage") or {}, time.time() - t0, None
    msg = {"role": "assistant", "content": None}
    calls, finish, usage, ttft = {}, None, {}, None
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
            if ttft is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")):
                ttft = time.time() - t0
            for k in ("content", "reasoning_content"):
                if d.get(k):
                    msg[k] = (msg.get(k) or "") + d[k]
            for tc in d.get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", 0), {"id": None, "type": "function",
                                                          "function": {"name": "", "arguments": ""}})
                c["id"] = tc.get("id") or c["id"]
                f = tc.get("function") or {}
                c["function"]["name"] += f.get("name") or ""
                c["function"]["arguments"] += f.get("arguments") or ""
            finish = ch.get("finish_reason") or finish
    if calls:
        msg["tool_calls"] = [calls[i] for i in sorted(calls)]
    return msg, finish, usage, time.time() - t0, ttft


def check_calls(calls, tools=None):
    """Each call's (id, name, args); ProtocolError on anything a client could not run. tools: a flow's
    (name -> (callable, spec)), checked against the schema with nested arrays and objects; else TOOLS."""
    out, ids = [], set()
    for c in calls:
        f = c.get("function") or {}
        name, cid = f.get("name"), c.get("id")
        if not cid or cid in ids:
            raise ProtocolError(f"missing or duplicate call id {cid!r}")
        ids.add(cid)
        if tools is not None:
            if name not in tools:
                raise ProtocolError(f"unknown tool {name!r}")
            try:
                args = json.loads(f.get("arguments") or "{}")
            except json.JSONDecodeError:
                raise ProtocolError(f"{name}: arguments are not JSON: {f.get('arguments')!r:.200}")
            if not isinstance(args, dict):
                raise ProtocolError(f"{name}: arguments are not an object: {args!r:.200}")
            e = agent_flows.args_ok(tools[name][1], args)
            if e:
                raise ProtocolError(f"{name}: {e}")
            out.append((cid, name, args))
            continue
        if name not in TOOLS:
            raise ProtocolError(f"unknown tool {name!r}")
        try:
            args = json.loads(f.get("arguments") or "")
        except json.JSONDecodeError:
            raise ProtocolError(f"{name}: arguments are not JSON: {f.get('arguments')!r:.200}")
        if not isinstance(args, dict):
            raise ProtocolError(f"{name}: arguments are not an object: {args!r:.200}")
        params = TOOLS[name][1]["function"]["parameters"]
        for k in params["required"]:
            if k not in args:
                raise ProtocolError(f"{name}: missing argument {k!r} in {args}")
            want = JSON_TYPES[params["properties"][k]["type"]]
            if not isinstance(args[k], want) or isinstance(args[k], bool) and want is not bool:
                raise ProtocolError(f"{name}: argument {k!r} has the wrong type: {args[k]!r}")
        out.append((cid, name, args))
    return out


def norm(s):
    return (s or "").lower().replace(",", "").replace("*", "")


# ------------------------------------------------------------------ the agent


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.lat, self.ttft = [], []
        self.requests = self.tokens = self.tool_calls = 0
        self.ws_content = 0  # replies with calls whose content is whitespace (it goes back into the history)
        self.tasks = {}  # name -> [done, correct, json_ok, json_correct, tool calls]
        self.ctx = {}  # name -> each run's largest prompt (tokens)
        self.protocol = []  # (task, message)
        self.wrong = []  # (task, answer)

    def request(self, lat, ttft, usage):
        with self.lock:
            self.requests += 1
            self.lat.append(lat)
            if ttft is not None:
                self.ttft.append(ttft)
            self.tokens += usage.get("completion_tokens", 0)


class Ctx:
    """What a flow may use of the client: a sub-agent, a second conversation on the same server."""

    def __init__(self, a, st, log):
        self.a, self.st, self.log = a, st, log

    def sub_agent(self, system, question, tools):
        a, specs = self.a, [sp for _, sp in tools.values()]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
        for _ in range(8):
            body = {"messages": messages, "tools": specs, "tool_choice": "auto", "max_tokens": a.max_tokens,
                    "temperature": a.temperature, "chat_template_kwargs": {"enable_thinking": a.think}}
            msg, finish, usage, lat, ttft = post(a, body, False)
            self.st.request(lat, ttft, usage)
            if finish not in ("stop", "tool_calls", "length"):
                raise ProtocolError(f"sub-agent: finish_reason {finish!r}")
            calls = msg.get("tool_calls") or []
            if not calls:
                return msg.get("content") or ""
            parsed = check_calls(calls, tools)
            messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})
            for cid, fname, args in parsed:
                fn, sp = tools[fname]
                props = sp["function"]["parameters"]["properties"]
                result = fn(**{k: args[k] for k in props if k in args})
                messages.append({"role": "tool", "tool_call_id": cid, "content": json.dumps(result)})
                if self.log:
                    self.log(f"    [specialist] {fname}({json.dumps(args)}) -> {json.dumps(result)}")
        return "(the specialist ran out of turns)"


def run_task(a, st, task, rng, log, dump):
    name, make = task
    flow = None
    if name in FLOW_NAMES:
        flow = make(rng, Ctx(a, st, log))
        prompt, min_calls, tools, system, max_turns = flow.prompt, flow.min_calls, flow.tools, agent_flows.SYSTEM, a.flow_turns
        specs, expect = [sp for _, sp in tools.values()], None
    else:
        prompt, expect, min_calls = make(rng)
        tools, specs, system, max_turns = None, TOOL_SPECS, SYSTEM, a.max_turns
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    extra = {"chat_template_kwargs": {"enable_thinking": a.think}}
    stream_next = rng.random() < 0.5
    n_calls, answer, ctx_max = 0, None, 0

    def grade(text):
        if flow is None:
            return all(e in norm(text) for e in expect), "lacks " + ", ".join(expect)
        return flow.grade(text)

    try:
        for turn in range(max_turns):
            stream = a.stream == "on" or (a.stream == "mix" and stream_next)
            stream_next = not stream_next
            body = {"messages": messages, "tools": specs, "tool_choice": "auto", "max_tokens": a.max_tokens,
                    "temperature": a.temperature, **extra}
            msg, finish, usage, lat, ttft = post(a, body, stream)
            st.request(lat, ttft, usage)
            ctx_max = max(ctx_max, usage.get("prompt_tokens", 0))
            if finish not in ("stop", "tool_calls", "length"):
                raise ProtocolError(f"finish_reason {finish!r}")
            calls = msg.get("tool_calls") or []
            if finish == "tool_calls" and not calls:
                raise ProtocolError("finish_reason tool_calls without tool_calls")
            if not calls or finish == "length":
                # cut off (a call's arguments may then be half written): the model's failure, not the server's
                answer = None if finish == "length" else (msg.get("content") or "")
                break
            parsed = check_calls(calls, tools)
            n_calls += len(parsed)
            if msg.get("content") and not msg["content"].strip():
                with st.lock:
                    st.ws_content += 1
            messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})
            for cid, fname, args in parsed:
                if flow is None:
                    result = TOOLS[fname][0](**{k: args[k] for k in TOOLS[fname][1]["function"]["parameters"]["required"]})
                else:
                    fn, sp = tools[fname]
                    props = sp["function"]["parameters"]["properties"]
                    result = fn(**{k: args[k] for k in props if k in args})
                messages.append({"role": "tool", "tool_call_id": cid, "content": json.dumps(result)})
                if log:
                    log(f"  [{name}] turn {turn + 1}: {fname}({json.dumps(args)}) -> {json.dumps(result)[:300]}")
        why = "cut off or out of turns"
        ok = False
        if answer is not None:
            ok, why = grade(answer)
            ok = ok and n_calls >= min_calls
            if n_calls < min_calls:
                why = f"{n_calls} calls, fewer than {min_calls}"
        if log:
            log(f"  [{name}] {'OK ' if ok else 'BAD'} after {n_calls} call(s): {(answer or '<none>')[:160]!r}"
                + ("" if ok else f" ({why})"))
        if dump and not ok:
            dump("wrong", name, expect if flow is None else why, messages, answer)
        json_ok = json_correct = False
        if a.json_final and answer is not None:
            # the reply's form in the prompt too: a grammar alone holds the model to text it was not asked for
            ask = 'Now give only the final answer, as JSON of the form {"answer": "..."}.'
            jmsgs = messages + [{"role": "assistant", "content": answer}, {"role": "user", "content": ask}]
            body = {"messages": jmsgs,
                    "response_format": {"type": "json_schema", "json_schema": {"name": "final", "schema": FINAL_SCHEMA}},
                    "max_tokens": a.max_tokens, "temperature": a.temperature, **extra}
            msg, finish, usage, lat, ttft = post(a, body, False)
            st.request(lat, ttft, usage)
            text = msg.get("content") or ""
            try:
                v = json.loads(text)
            except json.JSONDecodeError:
                if finish == "length":  # the model ran on inside a string: its failure, not the server's
                    v = None
                else:
                    raise ProtocolError(f"json_schema reply is not JSON ({finish}): {text[:200]!r}")
            if v is not None and not (isinstance(v, dict) and set(v) == {"answer"} and isinstance(v["answer"], str)):
                raise ProtocolError(f"json_schema reply breaks the schema: {text[:200]!r}")
            json_ok = v is not None
            json_correct = json_ok and grade(v["answer"])[0]
            if dump and not json_correct:
                dump("json " + ("wrong" if json_ok else "cut off"), name, expect, jmsgs, text)
            if log:
                log(f"  [{name}] json {'OK ' if json_correct else 'BAD'}: {text[:160]!r}")
        with st.lock:
            t = st.tasks.setdefault(name, [0, 0, 0, 0, 0])
            t[0] += 1
            t[1] += ok
            t[2] += json_ok
            t[3] += json_correct
            t[4] += n_calls
            st.ctx.setdefault(name, []).append(ctx_max)
            st.tool_calls += n_calls
            if not ok:
                st.wrong.append((name, (answer or "<none: cut off or out of turns>")[:150] + ("" if flow is None else f" [{why[:150]}]")))
    except ProtocolError as e:
        with st.lock:
            st.tasks.setdefault(name, [0, 0, 0, 0, 0])[0] += 1
            st.protocol.append((name, str(e)))
        if dump:
            dump("protocol: " + str(e), name, expect if flow is None else "", messages, None)
        if log:
            log(f"  [{name}] PROTOCOL ERROR: {e}")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", help="default: the server's first model")
    ap.add_argument("--concurrency", type=int, default=8, help="workers, each running tasks back to back")
    ap.add_argument("--duration", type=float, default=60, help="seconds to keep starting tasks")
    ap.add_argument("--rounds", type=int, default=0, help="tasks per worker instead of --duration (0: by time)")
    ap.add_argument("--suite", choices=["simple", "complex", "all"], default="simple",
                    help="simple: the short tasks; complex: agent_flows.py's flows; all: both")
    ap.add_argument("--tasks", default="all", help="comma-separated task names (" + ",".join(t[0] for t in TASKS + FLOWS) + ")")
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--flow-turns", type=int, default=24, help="turns a complex flow may take")
    ap.add_argument("--max-tokens", type=int, help="per request (default: 1024, 4096 with --think)")
    ap.add_argument("--temperature", type=float, help="default: 0, 0.6 with --think")
    ap.add_argument("--think", action="store_true", help="Qwen3 thinking on")
    ap.add_argument("--stream", choices=["off", "on", "mix"], default="mix")
    ap.add_argument("--no-json-final", dest="json_final", action="store_false",
                    help="skip asking for the answer again as json_schema")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true", help="print every tool call and answer")
    ap.add_argument("--dump", help="append each failed task's conversation to this file (JSON lines)")
    a = ap.parse_args()
    if a.max_tokens is None:
        a.max_tokens = 4096 if a.think else 1024
    if a.temperature is None:
        a.temperature = 0.6 if a.think else 0.0
    a.url = a.url.rstrip("/")
    if not a.model:
        a.model = json.loads(urllib.request.urlopen(a.url + "/v1/models", timeout=30).read())["data"][0]["id"]
    pool = {"simple": TASKS, "complex": FLOWS, "all": TASKS + FLOWS}[a.suite]
    tasks = pool if a.tasks == "all" else [t for t in TASKS + FLOWS if t[0] in a.tasks.split(",")]
    dump_f = open(a.dump, "a") if a.dump else None
    if not tasks:
        sys.exit("no such task")
    out_lock = threading.Lock()

    def log(s):
        with out_lock:
            print(s, flush=True)

    def dump(kind, name, expect, messages, reply):
        with out_lock:
            dump_f.write(json.dumps({"kind": kind, "task": name, "expect": expect, "messages": messages,
                                     "reply": reply}) + "\n")
            dump_f.flush()

    st = Stats()
    t0 = time.time()
    end = t0 + a.duration

    def worker(w):
        rng = random.Random(a.seed * 1000 + w)
        i = 0
        while (i < a.rounds) if a.rounds else (time.time() < end):
            task = tasks[(w * a.rounds + i) % len(tasks)] if a.rounds else rng.choice(tasks)
            run_task(a, st, task, rng, log if a.verbose else None, dump if dump_f else None)
            i += 1

    print(f"agent_loop: {a.url} model {a.model}, {a.concurrency} worker(s), "
          + (f"{a.rounds} task(s) each" if a.rounds else f"{a.duration:.0f} s")
          + f", stream {a.stream}, thinking {'on' if a.think else 'off'}, temperature {a.temperature}, "
          + f"max tokens {a.max_tokens}", flush=True)
    th = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(a.concurrency)]
    for t in th:
        t.start()
    last = time.time()
    while any(t.is_alive() for t in th):
        time.sleep(0.5)
        if not a.verbose and time.time() - last >= 10:
            last = time.time()
            with st.lock:
                done = sum(t[0] for t in st.tasks.values())
                good = sum(t[1] for t in st.tasks.values())
                print(f"  {last - t0:6.0f} s  tasks {done} (correct {good}, protocol errors {len(st.protocol)})  "
                      f"requests {st.requests}  {st.tokens / (last - t0):.0f} tok/s", flush=True)
    wall = time.time() - t0

    done = sum(t[0] for t in st.tasks.values())
    print(f"\n{done} task(s) in {wall:.1f} s ({done / wall * 60:.1f} a minute), {st.requests} request(s), "
          f"{st.tool_calls} tool call(s), {st.tokens} completion tokens ({st.tokens / wall:.0f} tok/s)")
    print(f"request latency p50 {pct(st.lat, 0.5):.2f} s  p95 {pct(st.lat, 0.95):.2f} s"
          + (f" | streamed TTFT p50 {pct(st.ttft, 0.5) * 1e3:.0f} ms  p95 {pct(st.ttft, 0.95) * 1e3:.0f} ms"
             if st.ttft else ""))
    print(f"{'task':16} {'runs':>5} {'correct':>8} {'json ok':>8} {'json ans':>8} {'calls/run':>9} {'prompt max':>10}")
    for name, _ in tasks:
        n, c, j, jc, k = st.tasks.get(name, [0, 0, 0, 0, 0])
        cx = st.ctx.get(name) or [0]
        print(f"{name:16} {n:5} {c:8} {j if a.json_final else '-':>8} {jc if a.json_final else '-':>8} "
              f"{(k / n if n else 0):9.1f} {max(cx):10}")
    if st.ws_content:
        print(f"\nWARNING: {st.ws_content} reply(ies) with tool calls had whitespace-only content (vLLM sends none); "
              "a client puts it back into the history, which the chat template then renders")
    if st.wrong:
        print(f"\nwrong answers ({len(st.wrong)}), first 5:")
        for name, ans in st.wrong[:8]:
            print(f"  {name}: {ans!r}")
    if st.protocol:
        print(f"\nPROTOCOL ERRORS ({len(st.protocol)}), first 10:")
        for name, e in st.protocol[:10]:
            print(f"  {name}: {e[:300]}")
    else:
        print("\nno protocol errors")
    sys.exit(1 if st.protocol else 0)


if __name__ == "__main__":
    main()
