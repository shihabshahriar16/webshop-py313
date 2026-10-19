#!/usr/bin/env python3
"""Generate the data for the WebShop task viewer (viewer/index.html).

The page it feeds is built around four things that are easy to get wrong about
WebShop, so this script records each of them explicitly:

* **What the agent receives.** Only the instruction, the page text and the
  clickables. It is NOT given the goal's reference query -- it must invent one.
* **What the grader rewards.** There is no gold action sequence (unlike
  ALFWorld). The grader scores WHATEVER PRODUCT IS BOUGHT against the goal spec:
      reward = (matched_attributes + matched_options + price_ok)
             / (n_attributes + n_options + 1)  * type_match
* **Partial credit.** Buying a near-miss scores between 0 and 1, so success rate
  and mean reward diverge and both must be reported.
* **Multiple solutions.** The goal's `asin` is just the product the instruction
  was generated from; any product matching the spec scores as well. To show this
  rather than assert it, the script BUYS SEVERAL DIFFERENT PRODUCTS per task and
  records what each one scores.

Run from the repo root after ./setup_py313.sh:

    python viewer/build_viewer.py -n 50

then open viewer/index.html in a browser (no server needed).

Writes viewer/webshop_examples.js, read by viewer/index.html.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

_OUT = os.path.join(_ROOT, "viewer", "webshop_examples.js")
_CLIP = 1500
_ASIN_RE = re.compile(r"[bB][0-9a-zA-Z]{9}$")


def _clickables(env):
    try:
        return list((env.get_available_actions() or {}).get("clickables") or [])
    except Exception:  # noqa: BLE001
        return []


def _has_search(env):
    try:
        return bool((env.get_available_actions() or {}).get("has_search_bar"))
    except Exception:  # noqa: BLE001
        return False


def _obs(env, raw=None):
    text = raw if isinstance(raw, str) else str(getattr(env, "observation", "") or "")
    return text[:_CLIP]


def _find(env, wanted):
    return {c.lower(): c for c in _clickables(env)}.get(str(wanted).lower())


def _agent_view(env):
    """What the text env gives an agent: instruction, observation, actions."""
    goal = re.sub(r"^\s*Instruction:\s*", "",
                  str(getattr(env, "instruction_text", "")).strip()).split("[SEP]")[0]
    actions = [f"click[{c}]" for c in _clickables(env)]
    if _has_search(env):
        actions.insert(0, "search[<your query>]")
    return {
        "goal": goal.strip(),
        "observation": _obs(env),
        "legal_actions": actions,
    }


def _build_env(kwargs):
    """Build the env the way equivalence/harness.py does: twice, using the second.

    WebShop draws goal price caps from the global ``random`` before it calls
    ``random.seed(233)``, so the FIRST env built in a process gets unseeded caps.
    Every later build is deterministic; those are the goals shown here.
    """
    import warnings
    warnings.filterwarnings("ignore")
    from web_agent_site.envs import WebAgentTextEnv
    WebAgentTextEnv(**kwargs)
    return WebAgentTextEnv(**kwargs)


def _buy(env, session, asin, goal, max_pages=6):
    """Reset, find `asin` (paginating), select the goal's options, buy. -> reward."""
    env.reset(session=session)
    env.step(f"search[{goal['query']}]")
    hit, page = None, 1
    for page in range(1, max_pages + 1):
        hit = _find(env, asin)
        if hit:
            break
        nxt = _find(env, "Next >")
        if not nxt:
            break
        env.step(f"click[{nxt}]")
    if not hit:
        return None, page
    env.step(f"click[{hit}]")
    for value in (goal.get("goal_options") or {}).values():
        opt = _find(env, value)
        if opt:
            env.step(f"click[{opt}]")
    buy = _find(env, "Buy Now")
    if not buy:
        return None, page
    out = env.step(f"click[{buy}]")
    return float(out[1] or 0.0), page


def _reference_steps(env, session, goal, max_pages=6):
    """The reference path, recorded step by step, PAGINATING until the target is
    found -- an earlier version stopped at page 1 and mislabelled solvable tasks
    as reward 0."""
    env.reset(session=session)
    steps = []

    def record(action, note):
        view = _agent_view(env)
        out = env.step(action)
        steps.append({
            "agent_sees": view,
            "action": action,
            "note": note,
            "result": _obs(env, out[0] if isinstance(out[0], str) else None),
            "reward": round(float(out[1] or 0.0), 4),
            "done": bool(out[2]),
        })
        return float(out[1] or 0.0)

    record(f"search[{goal['query']}]",
           "the goal's REFERENCE query — hidden from the agent, which must invent its own")

    hit, pages = None, 0
    for _ in range(max_pages):
        hit = _find(env, goal["asin"])
        if hit:
            break
        nxt = _find(env, "Next >")
        if not nxt:
            break
        pages += 1
        record(f"click[{nxt}]", f"target not on this page — paginating (page {pages + 1})")

    if not hit:
        steps.append({"agent_sees": _agent_view(env), "action": None,
                      "note": "target not reachable within %d pages" % max_pages,
                      "result": "", "reward": 0.0, "done": False})
        return steps, 0.0, pages + 1

    record(f"click[{hit}]", "open the target product")
    for value in (goal.get("goal_options") or {}).values():
        opt = _find(env, value)
        if opt:
            record(f"click[{opt}]", f"select required option: {value}")
    buy = _find(env, "Buy Now")
    reward = record(f"click[{buy}]", "purchase — the grader scores THIS product") if buy else 0.0
    return steps, reward, pages + 1


def _alternatives(env, session, goal, k=3):
    """Buy DIFFERENT products for the same goal, to show the reward spread."""
    env.reset(session=session)
    env.step(f"search[{goal['query']}]")
    page1 = [c for c in _clickables(env)
             if _ASIN_RE.match(c) and c.lower() != goal["asin"].lower()]
    out = []
    for asin in page1[:k]:
        reward, _page = _buy(env, session, asin, goal)
        if reward is None:
            continue
        prod = (env.server.product_item_dict.get(asin.upper())
                or env.server.product_item_dict.get(asin) or {})
        out.append({
            "asin": asin,
            "name": (prod.get("name") or prod.get("Title") or "")[:120],
            "reward": round(reward, 4),
        })
    return out


def _example(env, session, with_alternatives=True):
    env.reset(session=session)
    goal = dict(env.server.goals[session])
    first_view = _agent_view(env)
    steps, reward, pages = _reference_steps(env, session, goal)
    alts = _alternatives(env, session, goal) if with_alternatives else []

    n_attr = len(goal.get("attributes") or [])
    n_opt = len(goal.get("goal_options") or {})
    return {
        "session": session,
        "instruction": first_view["goal"],
        # --- what the agent gets, verbatim --------------------------------
        "agent_first_view": first_view,
        # --- what it never gets -------------------------------------------
        "hidden": {"query": goal.get("query"), "target_asin": goal.get("asin"),
                   "target_name": goal.get("name")},
        # --- what the grader scores ---------------------------------------
        "grading": {
            "attributes": goal.get("attributes") or [],
            "options": goal.get("goal_options") or {},
            "price_upper": goal.get("price_upper"),
            "denominator": n_attr + n_opt + 1,
            "criteria": n_attr + n_opt + 1,
        },
        "category": goal.get("category"),
        "product_category": goal.get("product_category"),
        # --- evidence -------------------------------------------------------
        "steps": steps,
        "reference_reward": round(float(reward), 4),
        "reference_success": bool(reward >= 1.0),
        "reference_pages": pages,
        "alternatives": alts,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--num", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--num-products", type=int, default=1000)
    ap.add_argument("--no-alternatives", action="store_true",
                    help="skip the alternative-purchase sweep (much faster)")
    ap.add_argument("--out", default=_OUT)
    args = ap.parse_args()

    kwargs = {"observation_mode": "text"}
    if args.num_products is not None:
        kwargs["num_products"] = int(args.num_products)
    env = _build_env(kwargs)

    total = len(env.server.goals)
    end = min(args.start + args.num, total)
    print(f"{total} sessions available; recording {args.start}..{end}")

    examples = []
    for session in range(args.start, end):
        try:
            ex = _example(env, session, not args.no_alternatives)
        except Exception as exc:  # noqa: BLE001
            print(f"  session {session}: FAILED {type(exc).__name__}: {exc}")
            continue
        examples.append(ex)
        spread = ", ".join(f"{a['reward']:.2f}" for a in ex["alternatives"]) or "-"
        print(f"  session {session:4d} ref={ex['reference_reward']:.2f} "
              f"(p{ex['reference_pages']}) alts=[{spread}] {ex['instruction'][:48]}")

    solved = sum(1 for e in examples if e["reference_success"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    meta = {"total_sessions": total, "num_products": args.num_products,
            "reference_solved": solved}
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("// Generated by viewer/build_viewer.py — do not edit.\n")
        fh.write("window.WEBSHOP_EXAMPLES = ")
        json.dump(examples, fh, ensure_ascii=False, indent=1)
        fh.write(";\n")
        fh.write(f"window.WEBSHOP_META = {json.dumps(meta)};\n")
    print(f"\nWrote {len(examples)} examples ({solved} reference-perfect) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
