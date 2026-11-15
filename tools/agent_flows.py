# SPDX-License-Identifier: Apache-2.0
"""agent_flows.py - the complex agent flows of agent_loop.py (--suite complex).

Each flow runs in a world of its own (a catalog with stock per warehouse, orders, an outbox, a small
file system, calendars, a staff directory with emails and managers), and is graded on the state it
leaves (the order placed, the email sent, the file written, the event created) as well as on its
answer. Together they exercise what the simple tasks do not:

  procure   search a catalog, check stock in parallel, place an order whose arguments are an array of
            objects, look up a manager through two directory calls, send an email (6-10 calls)
  meeting   offices, UTC offsets and calendars of three people, the earliest common free slot in
            everyone's working hours, an event created (interval reasoning; 7-10 calls)
  report    list files, read three long quarterly reports (~4k tokens each: the history grows past
            12k tokens), add the region totals with the calculator, write a summary file (5-6 calls)
  recover   discover the warehouses, check stock in all of them while calls fail (rate limited,
            temporarily unavailable) and must be retried (10-12 calls)
  delegate  a sub-agent: the main agent asks a finance specialist (a second conversation on the same
            server, with tools of its own) and emails the result (3 calls + the specialist's)
  research  a team roster, every member's office and its weather in parallel, the warmest office
            and the average with the calculator (9-10 calls)

Standard library only; imported by agent_loop.py, which owns the HTTP client and the reporting.
"""

import json

# ------------------------------------------------------------------ the shared data

CATALOG = {  # sku: (name, category, price in USD)
    "CB-100": ("USB-C cable 1 m", "cable", 9.5), "CB-200": ("USB-C cable 2 m", "cable", 12.0),
    "CB-300": ("HDMI cable 2 m", "cable", 8.75), "CB-400": ("Ethernet cable 5 m", "cable", 6.5),
    "CB-500": ("Thunderbolt cable 1 m", "cable", 29.0), "CB-600": ("DisplayPort cable 2 m", "cable", 11.25),
    "AD-100": ("USB-C to HDMI adapter", "adapter", 19.0), "AD-200": ("USB-C to Ethernet adapter", "adapter", 24.5),
    "AD-300": ("Travel power adapter", "adapter", 15.75), "AD-400": ("USB-C hub, 7 ports", "adapter", 39.0),
    "AD-500": ("Mini DisplayPort adapter", "adapter", 13.5), "AD-600": ("USB-A to USB-C adapter", "adapter", 7.25),
    "SD-100": ("SSD 512 GB", "ssd", 54.0), "SD-200": ("SSD 1 TB", "ssd", 79.5), "SD-300": ("SSD 2 TB", "ssd", 139.0),
    "SD-400": ("Portable SSD 1 TB", "ssd", 99.0), "SD-500": ("NVMe SSD 1 TB", "ssd", 84.25),
    "SD-600": ("Portable SSD 500 GB", "ssd", 64.75),
    "MN-100": ("24-inch monitor", "monitor", 149.0), "MN-200": ("27-inch monitor", "monitor", 229.0),
    "MN-300": ("27-inch 4K monitor", "monitor", 319.0), "MN-400": ("34-inch ultrawide monitor", "monitor", 449.0),
    "MN-500": ("Portable 15-inch monitor", "monitor", 189.5), "MN-600": ("32-inch 4K monitor", "monitor", 389.0),
}
RATES = {"EUR": 0.92, "JPY": 150.0, "GBP": 0.8, "CHF": 0.9, "CAD": 1.36}  # 1 USD = rate
TEMPS = {
    "paris": 18, "oslo": 7, "tokyo": 22, "new york": 15, "cairo": 31, "sydney": 25, "london": 12, "berlin": 9,
    "madrid": 24, "rome": 20, "lima": 17, "toronto": 4, "seoul": 11, "mumbai": 33, "dubai": 36, "singapore": 30,
    "nairobi": 23, "reykjavik": 2, "mexico city": 19, "istanbul": 16,
}
OFFSETS = {"london": 0, "paris": 1, "berlin": 1, "rome": 1, "oslo": 1, "madrid": 1, "cairo": 2, "istanbul": 3,
           "dubai": 4, "new york": -5, "toronto": -5, "lima": -5, "nairobi": 3, "reykjavik": 0}
# (name, team, office); teams of four with distinct office temperatures, a manager per team
PEOPLE = [
    ("Alice Chen", "Inference", "Oslo"), ("Ines Moreau", "Inference", "Paris"), ("Omar Nasser", "Inference", "Cairo"),
    ("Pia Lund", "Inference", "Reykjavik"),
    ("Bob Martin", "Kernels", "Tokyo"), ("Grace Liu", "Kernels", "Singapore"), ("Quinn Reyes", "Kernels", "Lima"),
    ("Rania Aziz", "Kernels", "Dubai"),
    ("Carla Diaz", "Serving", "Madrid"), ("Jonas Berg", "Serving", "Berlin"), ("Sven Olsen", "Serving", "London"),
    ("Tara Singh", "Serving", "Mumbai"),
    ("Emma Rossi", "Research", "Rome"), ("Keiko Sato", "Research", "Sydney"), ("Uma Patel", "Research", "Nairobi"),
    ("Victor Hugo", "Research", "Toronto"),
    ("Farid Haddad", "Platform", "Istanbul"), ("Liam Walsh", "Platform", "New York"), ("Mia Novak", "Platform", "Seoul"),
    ("Nora Kim", "Platform", "Mexico City"),
]
MANAGER = {"Inference": "Pia Lund", "Kernels": "Rania Aziz", "Serving": "Sven Olsen", "Research": "Uma Patel",
           "Platform": "Nora Kim"}
TEAMS = sorted(MANAGER)
WAREHOUSES = ["oslo", "tokyo", "cairo", "madrid", "rome", "dubai", "toronto", "seoul"]


def email_of(name):
    """A person's address: not guessable from the name (a fixed number per person), so it must be looked up."""
    code = sum((i + 1) * ord(ch) for i, ch in enumerate(name)) % 9000 + 1000
    return name.lower().replace(" ", ".") + f".{code}@example.com"


def person(name):
    for n, team, office in PEOPLE:
        if n.lower() == (name or "").strip().lower():
            return n, team, office
    return None


def money(x):
    return "%.2f" % x


def has_value(text, x, decimals=2):
    """text mentions the number x (commas and a trailing zero ignored)."""
    t = (text or "").replace(",", "")
    cands = {("%." + str(decimals) + "f") % x, ("%.4f" % x).rstrip("0").rstrip(".")}
    return any(c in t for c in cands)


def hhmm(minutes):
    return "%02d:%02d" % divmod(int(minutes), 60)


def minutes_of(s):
    h, m = str(s).strip().split(":")
    return int(h) * 60 + int(m)


# ------------------------------------------------------------------ tools


def spec(fn_name, desc, required=None, **params):
    """A function spec; params: argument name -> its JSON schema."""
    return {"type": "function", "function": {"name": fn_name, "description": desc, "parameters": {
        "type": "object", "properties": params, "required": list(params) if required is None else required}}}


S = {"type": "string"}
I = {"type": "integer"}


class World:
    """One flow's state: what the tools read and change, and what the grader inspects."""

    def __init__(self, rng):
        self.rng = rng
        self.stock = {}  # (sku, warehouse) -> units
        self.orders = []  # dicts
        self.outbox = []  # (to, subject, body)
        self.files = {}  # path -> text
        self.calendars = {}  # name -> [(start, end) minutes UTC]
        self.events = []  # dicts
        self.fail_once = set()  # (tool, key) that fail on their first call ("*": the tool's first call)

    # --- tools (the methods' names are the tools' names)
    def calculator(self, expression):
        import agent_loop
        return agent_loop.calculator(expression)

    def search_products(self, category):
        c = category.strip().lower().rstrip("s")
        items = [{"sku": k, "name": n, "category": cat, "price_usd": p} for k, (n, cat, p) in sorted(CATALOG.items())
                 if cat == c]
        return {"category": category, "products": items} if items else {"error": f"no category {category!r}; try "
                                                                                 "cable, adapter, ssd or monitor"}

    def list_warehouses(self):
        return {"warehouses": [w.title() for w in WAREHOUSES]}

    def check_stock(self, sku, warehouse):
        w, k = warehouse.strip().lower(), sku.strip().upper()
        if ("check_stock", "*") in self.fail_once:  # the first call of the flow
            self.fail_once.discard(("check_stock", "*"))
            return {"error": "rate limited: retry the call"}
        if w not in WAREHOUSES:
            return {"error": f"unknown warehouse {warehouse!r}; list_warehouses gives the names"}
        if ("check_stock", w) in self.fail_once:  # this warehouse's first call
            self.fail_once.discard(("check_stock", w))
            return {"error": f"warehouse {warehouse} temporarily unavailable: retry"}
        if k not in CATALOG:
            return {"error": f"unknown sku {sku!r}"}
        return {"sku": k, "warehouse": w.title(), "units": self.stock.get((k, w), 0)}

    def create_order(self, items, currency, deliver_to):
        cur, w = currency.strip().upper(), deliver_to.strip().lower()
        if cur != "USD" and cur not in RATES:
            return {"error": f"unsupported currency {currency!r}"}
        if w not in WAREHOUSES:
            return {"error": f"unknown warehouse {deliver_to!r}"}
        lines, total = [], 0.0
        for it in items:
            if not isinstance(it, dict) or "sku" not in it or "qty" not in it:
                return {"error": "each item needs sku and qty"}
            k, q = str(it["sku"]).strip().upper(), it["qty"]
            if k not in CATALOG or not isinstance(q, int) or q <= 0:
                return {"error": f"bad item {it!r}"}
            if self.stock.get((k, w), 0) < q:
                return {"error": f"insufficient stock of {k} in {deliver_to}: {self.stock.get((k, w), 0)} units"}
            lines.append((k, q))
            total += q * CATALOG[k][2] * (1.0 if cur == "USD" else RATES[cur])
        oid = "ORD-%04d" % (1000 + len(self.orders) + self.rng.randint(0, 8000))
        self.orders.append({"id": oid, "items": lines, "currency": cur, "deliver_to": w, "total": round(total, 2)})
        return {"order_id": oid, "total": round(total, 2), "currency": cur}

    def find_employee(self, name):
        p = person(name)
        if not p:
            return {"error": f"nobody called {name!r}"}
        n, team, office = p
        return {"name": n, "team": team, "office": office, "email": email_of(n), "manager": MANAGER[team]}

    def list_team(self, team):
        t = team.strip().title()
        names = [n for n, tm, _ in PEOPLE if tm == t]
        return {"team": t, "members": names} if names else {"error": f"no team {team!r}; teams: {', '.join(TEAMS)}"}

    def send_email(self, to, subject, body):
        self.outbox.append((to.strip().lower(), subject, body))
        return {"status": "sent", "to": to}

    def list_files(self, directory):
        d = directory.rstrip("/") + "/"
        names = sorted(p for p in self.files if p.startswith(d))
        return {"directory": directory, "files": names} if names else {"error": f"no files in {directory!r}"}

    def read_file(self, path):
        t = self.files.get(path.strip())
        return {"path": path, "content": t} if t is not None else {"error": f"no file {path!r}"}

    def write_file(self, path, content):
        self.files[path.strip()] = content
        return {"status": "written", "path": path, "bytes": len(content)}

    def get_utc_offset(self, city):
        o = OFFSETS.get(city.strip().lower())
        return {"city": city, "utc_offset_hours": o} if o is not None else {"error": f"unknown city {city!r}"}

    def get_calendar(self, name):
        p = person(name)
        if not p:
            return {"error": f"nobody called {name!r}"}
        busy = [[hhmm(a), hhmm(b)] for a, b in self.calendars.get(p[0], [])]
        return {"name": p[0], "date": "today", "busy_utc": busy}

    def create_event(self, title, attendees, start_utc, minutes):
        try:
            s = minutes_of(start_utc)
        except Exception:
            return {"error": f"start_utc must be HH:MM, got {start_utc!r}"}
        self.events.append({"title": title, "attendees": sorted(a.strip().lower() for a in attendees),
                            "start": s, "minutes": minutes})
        return {"status": "created", "title": title, "start_utc": hhmm(s), "minutes": minutes}

    def get_weather(self, city):
        t = TEMPS.get(city.strip().lower())
        return {"city": city, "temperature_c": t} if t is not None else {"error": f"unknown city {city!r}"}


SPECS = {
    "calculator": spec("calculator", "Evaluate an arithmetic expression (numbers, + - * / % ** and parentheses).",
                       expression={"type": "string", "description": "e.g. (3 + 4) * 2"}),
    "search_products": spec("search_products", "List the catalog's products of a category, with SKUs and USD prices.",
                            category={"type": "string", "description": "cable, adapter, ssd or monitor"}),
    "list_warehouses": spec("list_warehouses", "The names of our warehouses.", required=[]),
    "check_stock": spec("check_stock", "Units of a SKU in stock at a warehouse.", sku=S, warehouse=S),
    "create_order": spec("create_order", "Place an order, delivered to a warehouse, priced in a currency (USD, EUR, "
                                         "JPY, GBP, CHF or CAD). Returns the order id and total.",
                         items={"type": "array", "description": "the order lines",
                                "items": {"type": "object", "properties": {"sku": S, "qty": I},
                                          "required": ["sku", "qty"]}},
                         currency=S, deliver_to={"type": "string", "description": "warehouse city"}),
    "find_employee": spec("find_employee", "An employee's team, office city, email and manager.", name=S),
    "list_team": spec("list_team", "The members of a team.", team=S),
    "send_email": spec("send_email", "Send an email.", to={"type": "string", "description": "email address"},
                       subject=S, body=S),
    "list_files": spec("list_files", "List the files in a directory.", directory=S),
    "read_file": spec("read_file", "Read a text file.", path=S),
    "write_file": spec("write_file", "Write a text file (replacing it).", path=S, content=S),
    "get_utc_offset": spec("get_utc_offset", "A city's offset from UTC in hours (today).", city=S),
    "get_calendar": spec("get_calendar", "A person's busy intervals today, in UTC (HH:MM to HH:MM).", name=S),
    "create_event": spec("create_event", "Create a calendar event.", title=S,
                         attendees={"type": "array", "items": S, "description": "full names"},
                         start_utc={"type": "string", "description": "HH:MM in UTC"}, minutes=I),
    "get_weather": spec("get_weather", "A city's current temperature in Celsius.", city=S),
    "ask_specialist": spec("ask_specialist", "Ask the finance specialist (who can look up exchange rates and "
                                             "calculate) a question; returns their answer.", question=S),
}


class Flow:
    """A complex task: its prompt, tools (name -> (callable, spec)), grader and limits."""

    def __init__(self, prompt, world, tools, grade, min_calls, extra=None):
        self.prompt, self.world, self.grade, self.min_calls = prompt, world, grade, min_calls
        self.tools = {n: (extra[n] if extra and n in extra else getattr(world, n), SPECS[n]) for n in tools}


# ------------------------------------------------------------------ the flows


def f_procure(rng, ctx):
    w = World(rng)
    cat = rng.choice(["cable", "adapter", "ssd", "monitor"])
    city = rng.choice(WAREHOUSES)
    q = rng.choice([5, 8, 10, 12, 15, 20])
    cur = rng.choice(["EUR", "GBP", "CHF", "CAD", "USD"])
    who = rng.choice([p for p in PEOPLE if p[0] not in MANAGER.values()])
    skus = sorted((k for k, v in CATALOG.items() if v[1] == cat), key=lambda k: CATALOG[k][2])
    for k in skus:
        for wh in WAREHOUSES:
            w.stock[(k, wh)] = rng.randint(q, 3 * q) if rng.random() < 0.6 else rng.randint(0, q - 1)
    # the cheapest one or two lack stock here, so the answer needs the stock checks
    short = rng.choice([1, 2])
    for k in skus[:short]:
        w.stock[(k, city)] = rng.randint(0, q - 1)
    w.stock[(skus[short], city)] = rng.randint(q, 3 * q)
    want = skus[short]
    rate = 1.0 if cur == "USD" else RATES[cur]
    total = round(q * CATALOG[want][2] * rate, 2)
    mgr = MANAGER[who[1]]
    prompt = (f"Our team in {city.title()} needs {q} units of the cheapest {cat} in the catalog that has at least {q} "
              f"units in stock in the {city.title()} warehouse. Find it, place an order for it priced in {cur} and "
              f"delivered to {city.title()}, then email the order id and the order total to {who[0]}'s manager. "
              f"Tell me the order id and total.")

    def grade(answer):
        good = [o for o in w.orders if o["items"] == [(want, q)] and o["currency"] == cur and o["deliver_to"] == city]
        if len(w.orders) != 1 or not good:
            return False, f"orders {w.orders} (wanted {want} x{q} {cur} to {city})"
        oid = good[0]["id"]
        mails = [m for m in w.outbox if m[0] == email_of(mgr) and oid in m[1] + m[2] and has_value(m[1] + m[2], total)]
        if not mails:
            return False, f"no email to {email_of(mgr)} with {oid} and {money(total)}: {w.outbox}"
        return (oid in (answer or "") and has_value(answer, total)), "answer lacks the order id or total"

    return Flow(prompt, w, ["search_products", "check_stock", "create_order", "find_employee", "send_email",
                            "calculator"], grade, 5)


MEETING_TRIPLES = []


def _meeting_triples():
    if MEETING_TRIPLES:
        return MEETING_TRIPLES
    ok = [p for p in PEOPLE if p[2].lower() in OFFSETS]
    import itertools
    for t in itertools.combinations(ok, 3):
        lo = max(9 * 60 - OFFSETS[p[2].lower()] * 60 for p in t)
        hi = min(17 * 60 - OFFSETS[p[2].lower()] * 60 for p in t)
        if hi - lo >= 150:
            MEETING_TRIPLES.append((t, lo, hi))
    return MEETING_TRIPLES


def f_meeting(rng, ctx):
    w = World(rng)
    t, lo, hi = rng.choice(_meeting_triples())
    # busy blocks inside the shared window, so the earliest free slot is a little way in
    for p in t:
        cal, s = [], lo - 60
        while s < hi:
            if rng.random() < 0.45:
                d = rng.choice([30, 60, 90])
                cal.append((s, s + d))
                s += d
            s += 30
        w.calendars[p[0]] = cal
    expect = None
    for s in range(lo, hi - 29, 30):
        if all(not (a < s + 30 and s < b) for p in t for a, b in w.calendars[p[0]]):
            expect = s
            break
    if expect is None:  # everyone busy: free the last slot for one of them
        for p in t:
            w.calendars[p[0]] = [(a, b) for a, b in w.calendars[p[0]] if b <= hi - 30]
        expect = hi - 30
        for s in range(lo, hi - 29, 30):
            if all(not (a < s + 30 and s < b) for p in t for a, b in w.calendars[p[0]]):
                expect = s
                break
    names = [p[0] for p in t]
    prompt = (f"Schedule a 30-minute meeting titled 'Sync' for {names[0]}, {names[1]} and {names[2]} today, at the "
              f"earliest time starting on the hour or half hour when all three are free and it is within 09:00-17:00 "
              f"local time in each of their office cities. Create the event, then tell me its start time in UTC.")

    def grade(answer):
        want = sorted(n.lower() for n in names)
        good = [e for e in w.events if e["attendees"] == want and e["start"] == expect and e["minutes"] == 30]
        if not good:
            return False, f"events {w.events} (wanted {hhmm(expect)} UTC for {want})"
        return hhmm(expect) in (answer or "") or hhmm(expect).lstrip("0") in (answer or ""), "answer lacks the time"

    return Flow(prompt, w, ["find_employee", "get_utc_offset", "get_calendar", "create_event", "calculator"], grade, 7)


REGIONS = ["North", "South", "East", "West", "Central"]
LINES = ["Cables", "Adapters", "Storage", "Displays", "Services", "Support contracts", "Accessories", "Licenses"]


def _report(rng, year, quarter, totals):
    out = [f"ACME Hardware - Quarterly sales report, {year} Q{quarter}", "=" * 60, ""]
    for i in range(rng.randint(10, 14)):
        out.append(f"Note {i + 1}: " + " ".join(rng.choice(["Shipments", "Returns", "Promotions", "Logistics",
                                                              "Pricing", "Inventory", "Staffing", "Forecasts"])
                                                   + " " + rng.choice(["rose", "fell", "held steady", "were reviewed",
                                                                       "were reorganised", "met the plan"])
                                                   for _ in range(rng.randint(6, 10))) + ".")
    out += ["", "Revenue by region and product line (USD)", "-" * 60]
    for r in REGIONS:
        out.append(f"Region {r}")
        for ln in LINES:
            out.append(f"  {ln:20s} {rng.randint(1000, 60000):>10,}.{rng.randint(0, 99):02d}")
        out.append("")
    out += ["Region totals (USD, after returns and discounts; not the sum of the lines above)", "-" * 60]
    for r in REGIONS:
        out.append(f"  {r:8s} total revenue: {totals[r]:,.2f} USD")
    out += ["", "Largest accounts (USD, before returns)", "-" * 60]
    for i in range(70):
        out.append(f"  Account {rng.choice(['Nordwind', 'Kestrel', 'Atlas', 'Meridian', 'Orbit', 'Cobalt', 'Juniper'])}"
                   f"-{rng.randint(100, 999)}: region {rng.choice(REGIONS):8s} {rng.randint(2000, 90000):>8,}.{rng.randint(0, 99):02d}"
                   f"  ({rng.choice(['renewal', 'new', 'expansion', 'pilot'])})")
    out += ["", "Appendix: unit volumes by week"]
    for wk in range(1, 14):
        out.append(f"  week {wk:2d}: " + ", ".join(f"{r} {rng.randint(100, 9999)}" for r in REGIONS))
    out += ["", "End of report."]
    return "\n".join(out)


def f_report(rng, ctx):
    w = World(rng)
    region = rng.choice(REGIONS)
    totals = []
    for q in (1, 2, 3):
        t = {r: rng.randint(50000, 400000) + rng.randint(0, 99) / 100 for r in REGIONS}
        totals.append(t[region])
        w.files[f"/reports/2026-q{q}.txt"] = _report(rng, 2026, q, t)
    w.files["/reports/2025-q4.txt"] = _report(rng, 2025, 4, {r: rng.randint(50000, 400000) + 0.5 for r in REGIONS})
    w.files["/reports/README.txt"] = ("Quarterly reports, one file per quarter (YYYY-qN.txt). The region totals "
                                      "section of each report is authoritative.")
    want = round(sum(totals), 2)
    prompt = (f"The quarterly sales reports are in /reports. Add up the {region} region's total revenue over the 2026 "
              f"quarterly reports, use the calculator for the sum, write the line 'Total {region} revenue 2026: "
              f"<amount> USD' to /reports/summary.txt, and tell me the amount.")

    def grade(answer):
        s = w.files.get("/reports/summary.txt")
        if s is None or not has_value(s, want):
            return False, f"summary {s!r} (wanted {money(want)})"
        return has_value(answer, want), "answer lacks the amount"

    return Flow(prompt, w, ["list_files", "read_file", "write_file", "calculator"], grade, 5)


def f_recover(rng, ctx):
    w = World(rng)
    sku = rng.choice(sorted(CATALOG))
    for wh in WAREHOUSES:
        w.stock[(sku, wh)] = rng.randint(0, 400)
    best = max(WAREHOUSES, key=lambda x: (w.stock[(sku, x)], x))
    top = w.stock[(sku, best)]
    for wh in WAREHOUSES:  # a unique maximum
        if wh != best and w.stock[(sku, wh)] == top:
            w.stock[(sku, wh)] -= 1
    total = sum(w.stock[(sku, x)] for x in WAREHOUSES)
    w.fail_once = {("check_stock", "*"), ("check_stock", rng.choice(WAREHOUSES))}
    prompt = (f"How many units of {sku} do we have in stock across all of our warehouses (use the calculator for the "
              f"total), and which warehouse has the most? Some calls may fail temporarily; retry them.")

    def grade(answer):
        a = (answer or "").replace(",", "").lower()
        return str(total) in a and best in a, f"wanted {total} units, most in {best.title()}"

    return Flow(prompt, w, ["list_warehouses", "check_stock", "calculator"], grade, 9)


SPECIALIST = ("You are the finance specialist. Use your tools for every rate and every calculation, and reply with the "
              "final amount (two decimals) and its currency.")


def f_delegate(rng, ctx):
    w = World(rng)
    cur = rng.choice(sorted(RATES))
    amount = rng.choice([150, 240, 380, 500, 720, 1250, 2000])
    fee = rng.choice([1.5, 2, 2.5, 3, 4])
    want = round(amount * RATES[cur] * (1 - fee / 100), 2)
    who = rng.choice(PEOPLE)
    import agent_loop
    sub_tools = {"exchange_rate": (agent_loop.exchange_rate, agent_loop.TOOLS["exchange_rate"][1]),
                 "calculator": (agent_loop.calculator, agent_loop.TOOLS["calculator"][1])}

    def ask_specialist(question):
        return {"answer": ctx.sub_agent(SPECIALIST, question, sub_tools)}

    prompt = (f"Ask the finance specialist to convert {amount} USD to {cur} and then deduct a {fee}% transfer fee, and "
              f"email the resulting amount to {who[0]}. Then tell me the amount.")

    def grade(answer):
        mails = [m for m in w.outbox if m[0] == email_of(who[0]) and has_value(m[1] + m[2], want)]
        if not mails:
            return False, f"no email to {email_of(who[0])} with {money(want)}: {w.outbox}"
        return has_value(answer, want), "answer lacks the amount"

    return Flow(prompt, w, ["ask_specialist", "find_employee", "send_email"], grade, 3,
                extra={"ask_specialist": ask_specialist})


def f_research(rng, ctx):
    w = World(rng)
    team = rng.choice(TEAMS)
    members = [p for p in PEOPLE if p[1] == team]
    temps = {p[0]: TEMPS[p[2].lower()] for p in members}
    warmest = max(temps, key=temps.get)
    avg = sum(temps.values()) / len(temps)
    prompt = (f"For every member of the {team} team, find the current temperature in their office city. Name the "
              f"team member who sits in the warmest office, and give the average temperature across the team's offices "
              f"(use the calculator for the average).")

    def grade(answer):
        a = (answer or "").lower()
        return warmest.lower() in a and has_value(a, avg), f"wanted {warmest} and {avg}"

    return Flow(prompt, w, ["list_team", "find_employee", "get_weather", "calculator"], grade, 9)


FLOWS = [(f.__name__[2:], f) for f in (f_procure, f_meeting, f_report, f_recover, f_delegate, f_research)]
SYSTEM = ("You are an operations assistant with tools. Never guess a value a tool can give you: look up every id, SKU, "
          "name, address and number first, and call several tools in one turn only when none of them needs another's "
          "result. Read every tool result before you go on; when a call reports a temporary failure, call it again. "
          "Finish every action the user asks for, then reply briefly with the requested values.")


def args_ok(spec_, args):
    """A check of the call's arguments against its schema, nested arrays and objects included: None or the problem."""
    params = spec_["function"]["parameters"]
    types = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list}

    def check(v, sch, where):
        t = sch.get("type")
        if t and (not isinstance(v, types[t]) or (isinstance(v, bool) and t not in ("boolean",))):
            return f"{where} should be {t}: {json.dumps(v)[:80]}"
        if t == "array" and "items" in sch:
            for i, x in enumerate(v):
                e = check(x, sch["items"], f"{where}[{i}]")
                if e:
                    return e
        if t == "object" and "properties" in sch:
            for k in sch.get("required", []):
                if k not in v:
                    return f"{where} lacks {k!r}"
            for k, x in v.items():
                if k in sch["properties"]:
                    e = check(x, sch["properties"][k], f"{where}.{k}")
                    if e:
                        return e
        return None

    for k in params.get("required", []):
        if k not in args:
            return f"missing argument {k!r}"
    for k, v in args.items():
        if k in params["properties"]:
            e = check(v, params["properties"][k], k)
            if e:
                return e
    return None
