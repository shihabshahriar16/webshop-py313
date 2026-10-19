"""Cross-interpreter equivalence harness for the WebShop text env.

Search results and rewards feed the agent directly, so a port that ranks or
scores differently is a different benchmark -- silently. This harness pins the
reference (py3.8 + pyserini 0.17 + JDK 11) down as JSON and replays it against
any other install, requiring EXACT equality.

    # 1. in the reference env (py3.8 conda `webshop`)
    python equivalence/harness.py dump   --out equivalence/golden/golden_py38.json.gz
    # 2. in the candidate env (e.g. py3.13 venv)
    python equivalence/harness.py replay --golden equivalence/golden/golden_py38.json.gz \
                                         --report equivalence/results/report_py313.json

Sections (all compared with ==, floats included -- no tolerances):

* ``trajectories`` -- for each session, the initial observation and available
  actions, then a fixed scripted action sequence (several differently-phrased
  searches, pagination, product open, option clicks, sub-pages, invalid actions,
  Buy Now). The actions are chosen once, at dump time, and stored; replay feeds
  the STORED actions, so a divergence surfaces as an observation diff rather
  than being re-derived.
* ``topk``   -- top-50 docids + scores for ~200 goal-instruction queries plus
  short/odd variants, straight from the search engine.
* ``goals``  -- every goal's instruction text, asin, price cap and options (the
  price caps come from Python's ``random``, so this also pins the RNG stream).
* ``prices`` -- the per-product price table (also RNG-derived).
* ``rewards`` -- ``get_reward`` (+ its info dict) over goal x top-10-product
  pairs, with and without options. Rewards depend on spaCy POS tags and
  thefuzz, both of which change across library versions.

Env construction mirrors multiagent-planning's adapter: kwargs
``observation_mode='text', num_products=1000``, and the env is built TWICE,
using the second. The first construction draws product prices from an unseeded
``random`` (non-deterministic per process); it ends with ``random.seed(233)``,
so every later construction in the process is deterministic. The adapter's
session-count probe plays the role of the first construction in real runs.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ENV_KWARGS = dict(observation_mode="text", num_products=1000)
DEFAULT_SESSIONS = list(range(50))
SEARCH_K = 50  # engine.SEARCH_RETURN_N

BUTTONS = {"back to search", "next >", "< prev", "buy now",
           "description", "features", "reviews", "attributes"}


# --------------------------------------------------------------------------- #
# env
# --------------------------------------------------------------------------- #
def build_env():
    import warnings
    warnings.filterwarnings("ignore")
    from web_agent_site.envs import WebAgentTextEnv
    WebAgentTextEnv(**ENV_KWARGS)          # probe: consumes the unseeded draws
    return WebAgentTextEnv(**ENV_KWARGS)   # deterministic from here on


def runtime_info():
    """Interpreter, distribution versions and the JVM actually running search."""
    try:
        from importlib.metadata import version
    except ImportError:  # pragma: no cover
        version = None
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for dist in ("Flask", "Werkzeug", "Jinja2", "MarkupSafe", "beautifulsoup4", "spacy",
                 "en_core_web_sm", "thefuzz", "python-Levenshtein", "rapidfuzz", "numpy",
                 "gym", "gymnasium", "pyserini", "pyjnius", "torch"):
        try:
            info[dist] = version(dist)
        except Exception:  # noqa: BLE001 - not installed
            info[dist] = None
    try:
        import jnius_config
        if jnius_config.vm_running:
            from jnius import autoclass
            sysprops = autoclass("java.lang.System")
            info["java"] = "%s (%s)" % (sysprops.getProperty("java.version"),
                                        sysprops.getProperty("java.home"))
    except Exception:  # noqa: BLE001
        pass
    from web_agent_site.engine.lucene import _backend
    info["search_backend"] = _backend()
    return info


# --------------------------------------------------------------------------- #
# scripted trajectories
# --------------------------------------------------------------------------- #
_STOP = {"i", "me", "a", "an", "the", "and", "with", "for", "of", "that", "is",
         "in", "to", "find", "need", "looking", "want", "am", "my", "it", "its",
         "be", "are", "please", "some", "this", "lower", "than", "price", "dollars"}


def _words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def query_variants(goal, session):
    """Two differently-phrased queries per session, deterministic in the goal."""
    instr = goal["instruction_text"]
    words = _words(instr)
    content = [w for w in words if w not in _STOP]
    q1 = instr.lower()  # what weak agents usually type: the whole instruction
    style = session % 8
    if style == 0:
        q2 = " ".join(words[:5])                                  # truncated prefix
    elif style == 1:
        q2 = goal["query"]                                        # dataset query field
    elif style == 2:
        q2 = instr                                                # original case + punctuation
    elif style == 3:
        q2 = " ".join(content[:6])                                # keywords, stopwords dropped
    elif style == 4:                                              # punctuation/specials
        q2 = '%s\'s "%s" & 3-pack (xl) / %s+%s $40' % (
            (content + ["x"] * 4)[0], (content + ["x"] * 4)[1],
            (content + ["x"] * 4)[2], (content + ["x"] * 4)[3])
    elif style == 5:
        q2 = (content or words)[-1]                               # single word
    elif style == 6:
        q2 = " ".join(reversed(content[:8])).upper()              # reordered, upper
    else:
        q2 = "  ".join(content[:4]) + " shoes womens running sz 8.5"  # plurals/stems/noise
    return q1, q2


def _products(avail):
    return [c for c in avail["clickables"] if c not in BUTTONS and c != "search"
            and re.fullmatch(r"[a-z0-9]{10}", c)]


def _options(avail):
    return [c for c in avail["clickables"] if c not in BUTTONS and c != "search"
            and not re.fullmatch(r"[a-z0-9]{10}", c)]


class Recorder:
    def __init__(self, env):
        self.env = env
        self.steps = []
        self.done = False

    def act(self, action):
        if self.done:
            return
        obs, reward, done, info = self.env.step(action)
        avail = self.env.get_available_actions()
        self.steps.append({"action": action, "obs": obs, "reward": reward,
                           "done": done, "avail": avail,
                           "url": self.env.browser.current_url})
        self.done = bool(done)

    @property
    def avail(self):
        return self.steps[-1]["avail"] if self.steps else self.env.get_available_actions()


def script_session(env, session):
    """Run the fixed script on one session; returns the full record."""
    obs, _ = env.reset(session=session)
    init = {"obs": obs, "avail": env.get_available_actions(),
            "instruction_text": env.instruction_text}
    goal = env.server.goals[session]
    q1, q2 = query_variants(goal, session)
    r = Recorder(env)

    r.act(f"search[{q1}]")
    r.act("click[next >]")
    r.act("click[next >]")
    r.act("click[< prev]")
    r.act("click[back to search]")
    r.act(f"search[{q2}]")
    r.act("click[definitely-not-a-button]")       # invalid: must be a no-op
    if session % 3 == 1 and "next >" in r.avail["clickables"]:
        r.act("click[next >]")                    # open a product from page 2
    prods = _products(r.avail)
    if prods:
        pick = prods[min(session % 4, len(prods) - 1)]
        r.act(f"click[{pick}]")
        opts = _options(r.avail)
        if opts:
            r.act(f"click[{opts[0]}]")
            if len(opts) > 1:
                r.act(f"click[{opts[-1]}]")       # usually a second option group
        for sub in ("description", "features", "reviews", "attributes"):
            if sub in r.avail["clickables"]:
                r.act(f"click[{sub}]")
                r.act("click[< prev]")
        if session % 5 == 0:
            r.act("click[< prev]")                # item -> results
            r.act("search[]")                     # empty search: no-op
            prods = _products(r.avail)
            if prods:
                r.act(f"click[{prods[-1]}]")
        r.act("click[buy now]")
    return {"session": session, "q1": q1, "q2": q2, "init": init, "steps": r.steps}


def replay_session(env, golden):
    obs, _ = env.reset(session=golden["session"])
    out = {"session": golden["session"], "init": {
        "obs": obs, "avail": env.get_available_actions(),
        "instruction_text": env.instruction_text}, "steps": []}
    r = Recorder(env)
    for step in golden["steps"]:
        r.act(step["action"])
    out["steps"] = r.steps
    return out


# --------------------------------------------------------------------------- #
# search / goals / prices / rewards
# --------------------------------------------------------------------------- #
def topk_queries(goals, n=200):
    stride = max(len(goals) // n, 1)
    picked = [goals[i] for i in range(0, len(goals), stride)][:n]
    qs = []
    for g in picked:
        qs.append(g["instruction_text"].lower())
    for g in picked:  # short / odd variants exercise the analyzer harder
        w = _words(g["instruction_text"])
        qs.append(" ".join(w[2:5]))
    return qs


def search_hits(engine, query, k=SEARCH_K):
    hits = engine.search(query, k=k)
    return [[h.docid, float(h.score)] for h in hits]


def dump_topk(env):
    eng = env.server.search_engine
    return [{"q": q, "hits": search_hits(eng, q)} for q in topk_queries(env.server.goals)]


def dump_goals(env):
    keep = ("asin", "instruction_text", "price_upper", "attributes",
            "goal_options", "query", "category", "weight")
    return [{k: g.get(k) for k in keep} for g in env.server.goals]


def dump_prices(env):
    return env.server.product_prices


def reward_pairs(env, n_goals=300, n_products=10):
    from web_agent_site.engine.engine import get_top_n_product_from_keywords
    from web_agent_site.engine.goal import get_reward
    srv = env.server
    stride = max(len(srv.goals) // n_goals, 1)
    out = []
    for gi in list(range(0, len(srv.goals), stride))[:n_goals]:
        goal = srv.goals[gi]
        prods = get_top_n_product_from_keywords(
            _words(goal["instruction_text"]), srv.search_engine,
            srv.all_products, srv.product_item_dict)[:n_products]
        own = srv.product_item_dict.get(goal["asin"])
        if own is not None and own not in prods:
            prods.append(own)
        for p in prods:
            opt_sets = [{}]
            if p["options"]:
                opt_sets.append({k: v[0] for k, v in p["options"].items() if v})
            if isinstance(goal["goal_options"], dict) and goal["goal_options"]:
                opt_sets.append(dict(goal["goal_options"]))
            for opts in opt_sets:
                rew, info = get_reward(p, goal, price=srv.product_prices.get(p["asin"]),
                                       options=opts, verbose=True)
                out.append({"goal": gi, "asin": p["asin"], "options": opts,
                            "reward": rew, "info": info})
    return out


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #
def first_diff(a, b, path=""):
    """Path + values of the first difference between two JSON-like values."""
    if type(a) != type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))
                                   and not isinstance(a, bool) and not isinstance(b, bool)):
        return path, a, b
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                return f"{path}.{k}", a.get(k, "<missing>"), b.get(k, "<missing>")
            d = first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        if len(a) != len(b):
            return f"{path}.len", len(a), len(b)
        return None
    return None if a == b else (path, a, b)


def _norm(x):
    """Round-trip through JSON so tuples/sets/ints compare as the golden file does."""
    return json.loads(json.dumps(x, default=_json_default, sort_keys=True))


def _json_default(o):
    if isinstance(o, set):
        return sorted(o)
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(type(o))


def compare_topk(gold, cand):
    exact10 = exact50 = scores_eq = 0
    perturbed = []
    for g, c in zip(gold, cand):
        gi = [h[0] for h in g["hits"]]
        ci = [h[0] for h in c["hits"]]
        e10, e50 = gi[:10] == ci[:10], gi == ci
        exact10 += e10
        exact50 += e50
        scores_eq += g["hits"] == c["hits"]
        if not e50:
            moved = [(d, gi.index(d), ci.index(d) if d in ci else None)
                     for d in gi if d not in ci or ci.index(d) != gi.index(d)]
            perturbed.append({"q": g["q"], "exact@10": e10, "moved": moved[:10],
                              "gold_top10": g["hits"][:10], "cand_top10": c["hits"][:10]})
    n = len(gold)
    return {"queries": n, "exact@10": exact10, "exact@50": exact50,
            "exact_scores": scores_eq, "exact@10_rate": exact10 / n if n else None,
            "perturbed": perturbed}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _open(path, mode):
    """Plain or gzipped JSON, by extension."""
    return gzip.open(path, mode + "t") if path.endswith(".gz") else open(path, mode)


def cmd_dump(args):
    t = time.time()
    env = build_env()  # before runtime_info(), so the JVM is up and reportable
    data = {"runtime": runtime_info(), "env_kwargs": ENV_KWARGS,
            "sessions": args.sessions}
    data["trajectories"] = [script_session(env, s) for s in args.sessions]
    data["topk"] = dump_topk(env)
    data["goals"] = dump_goals(env)
    data["prices"] = dump_prices(env)
    data["rewards"] = reward_pairs(env)
    data = _norm(data)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with _open(args.out, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    n_steps = sum(len(t_["steps"]) for t_ in data["trajectories"])
    print(f"wrote {args.out}: {len(data['trajectories'])} sessions, {n_steps} steps, "
          f"{len(data['topk'])} queries, {len(data['goals'])} goals, "
          f"{len(data['rewards'])} reward pairs in {time.time() - t:.0f}s")


def cmd_replay(args):
    with _open(args.golden, "r") as f:
        gold = json.load(f)
    env = build_env()
    report = {"golden": os.path.relpath(args.golden, ROOT),
              "golden_runtime": gold["runtime"], "runtime": runtime_info()}

    # trajectories
    obs_total = obs_eq = 0
    traj_diffs = []
    for g in gold["trajectories"]:
        c = _norm(replay_session(env, g))
        units = [("init", g["init"], c["init"])] + [
            (f"step{i}:{s['action'][:60]}", s, cs)
            for i, (s, cs) in enumerate(zip(g["steps"], c["steps"]))]
        if len(g["steps"]) != len(c["steps"]):
            units.append(("n_steps", len(g["steps"]), len(c["steps"])))
        for name, a, b in units:
            obs_total += 1
            d = first_diff(a, b)
            if d is None:
                obs_eq += 1
            else:
                traj_diffs.append({"session": g["session"], "unit": name,
                                   "path": d[0], "gold": d[1], "cand": d[2]})
    report["trajectories"] = {"units": obs_total, "equal": obs_eq, "diffs": traj_diffs}

    # top-k
    cand_topk = _norm([{"q": t["q"], "hits": search_hits(env.server.search_engine, t["q"])}
                       for t in gold["topk"]])
    report["topk"] = compare_topk(gold["topk"], cand_topk)

    # goals / prices
    cg, cp = _norm(dump_goals(env)), _norm(dump_prices(env))
    gdiff = [i for i, (a, b) in enumerate(zip(gold["goals"], cg)) if a != b]
    report["goals"] = {"n": len(gold["goals"]), "n_cand": len(cg), "differing": len(gdiff),
                       "first": [{"i": i, "gold": gold["goals"][i], "cand": cg[i]}
                                 for i in gdiff[:5]]}
    pdiff = sorted(k for k in set(gold["prices"]) | set(cp)
                   if gold["prices"].get(k) != cp.get(k))
    report["prices"] = {"n": len(gold["prices"]), "differing": len(pdiff),
                        "first": [{"asin": k, "gold": gold["prices"].get(k),
                                   "cand": cp.get(k)} for k in pdiff[:5]]}

    # rewards
    cr = _norm(reward_pairs(env))
    rdiff = [i for i, (a, b) in enumerate(zip(gold["rewards"], cr)) if a != b]
    report["rewards"] = {"n": len(gold["rewards"]), "n_cand": len(cr),
                         "differing": len(rdiff),
                         "first": [{"gold": gold["rewards"][i], "cand": cr[i]}
                                   for i in rdiff[:10]]}

    ok = (obs_eq == obs_total and report["topk"]["exact@50"] == report["topk"]["queries"]
          and report["topk"]["exact_scores"] == report["topk"]["queries"]
          and not gdiff and len(cg) == len(gold["goals"]) and not pdiff
          and not rdiff and len(cr) == len(gold["rewards"]))
    report["verdict"] = "IDENTICAL" if ok else "DIVERGES"

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=1, sort_keys=True)
    tk = report["topk"]
    print(f"golden:   python {gold['runtime']['python']}   candidate: python {report['runtime']['python']}")
    print(f"trajectories: {obs_eq}/{obs_total} observation units identical "
          f"({len(gold['trajectories'])} sessions)")
    print(f"top-k:   exact@10 {tk['exact@10']}/{tk['queries']}  exact@50 {tk['exact@50']}/{tk['queries']}"
          f"  identical scores {tk['exact_scores']}/{tk['queries']}")
    print(f"goals:   {report['goals']['differing']} of {len(gold['goals'])} differ")
    print(f"prices:  {report['prices']['differing']} of {len(gold['prices'])} differ")
    print(f"rewards: {report['rewards']['differing']} of {len(gold['rewards'])} differ")
    for d in traj_diffs[:5]:
        print(f"  DIFF session {d['session']} {d['unit']} at {d['path']}:\n"
              f"    gold: {str(d['gold'])[:300]}\n    cand: {str(d['cand'])[:300]}")
    print("VERDICT:", report["verdict"])
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--out", required=True)
    d.add_argument("--sessions", type=lambda s: list(range(*map(int, s.split("-")))) if "-" in s
                   else [int(x) for x in s.split(",")], default=DEFAULT_SESSIONS,
                   help="'0-50' (end exclusive) or '1,2,3'")
    r = sub.add_parser("replay")
    r.add_argument("--golden", required=True)
    r.add_argument("--report")
    args = p.parse_args(argv)
    os.chdir(ROOT)  # the engine resolves data/ and search_engine/ relative to BASE_DIR anyway
    return cmd_dump(args) or 0 if args.cmd == "dump" else cmd_replay(args)


if __name__ == "__main__":
    sys.exit(main())
