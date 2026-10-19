# WebShop text env on Python 3.13

The text environment (`web_agent_site.envs.WebAgentTextEnv`) runs on Python 3.13
**without pyserini**. It reproduces the original Python 3.8 env exactly:
observations, available actions, search rankings with their scores, goals, prices
and rewards. `equivalence/harness.py` checks that claim (see below).

Only the text env is covered. The Selenium site env, the Flask web app, and
`baseline_models/` were not ported.

## Setup

```sh
# 1. data + indexes: once, from the ORIGINAL setup (needs the py3.8 env once;
#    see setup.sh / setup_arm.sh and the HuggingFace data mirror). Skip this if
#    data/ and search_engine/indexes* already exist.
# 2. any JDK >= 11 (verified identical on 11, 21, 25)
export JAVA_HOME=$(/usr/libexec/java_home -v 21)    # macOS; optional, pyjnius finds one
# 3. into your 3.13 venv
PYTHON=/path/to/venv/bin/python ./setup_py313.sh
```

`setup_py313.sh` installs `requirements_py313.txt`, installs the pinned spaCy model
`en_core_web_sm==3.7.1` with `--no-deps`, and downloads `anserini-0.14.3-fatjar.jar`
from Maven Central into `search_engine/jars/`, checking its SHA-256.

## What changed and why

| Blocker on 3.13 | Change | Effect on 3.8 |
|---|---|---|
| `pyserini 0.17` (Lucene 8 via pyjnius) is 3.8-only; modern pyserini uses Lucene 10, which raises `IndexFormatTooOldException` on these Lucene 8 indexes | `engine/lucene.py`: calls the **same** `io.anserini.search.SimpleSearcher` from the **same** Anserini 0.14.3 jar through pyjnius 1.9. This is the subset of pyserini 0.17's `LuceneSearcher` that the engine uses | none: `auto` still selects pyserini when pyserini 0.17 is installed |
| `torch` imported at module level | imported lazily, only for `get_image` | none |
| `gym` breaks on NumPy 2 | `envs/gym_compat.py`: `gym` if installed, else `gymnasium` | none |
| `envs/__init__.py` imports the Selenium env | that import is optional | none |
| unused `cleantext`, `rank_bm25` imports | removed | none |
| Python 3.12+ `sum()` of floats is compensated (Neumaier), so goal weights moved in the last ulp | `goal.py`: plain left-to-right sum, the pre-3.12 behaviour | none (same arithmetic) |

Search backend selection: `WEBSHOP_SEARCH_BACKEND=auto|pyserini|jnius`. The jar
location can be overridden with `ANSERINI_JAR`.

### Pins that matter (each one checked with the harness)

* **`en_core_web_sm==3.7.1`**, not 3.8.0. Rewards use spaCy POS tags. With 3.8.0,
  **1510 of 8477** reward pairs changed and nothing raised an error.
* **`thefuzz==0.19.0` without python-Levenshtein.** This is the `difflib` fallback
  the reference uses. thefuzz 0.22 (rapidfuzz) also matched on the sample, but it
  is a different algorithm.
* Flask 3.1 / Werkzeug 3.1 / bs4 4.13 / Jinja2 3.1.6 / NumPy 2: no differences.

## Equivalence harness

```sh
# reference (py3.8 conda env, pyserini 0.17, JDK 11) -- already done, committed:
python equivalence/harness.py dump --out equivalence/golden/golden_py38.json.gz
# any candidate install:
python equivalence/harness.py replay --golden equivalence/golden/golden_py38.json.gz \
    --report equivalence/results/report_<name>.json      # exit 0 iff IDENTICAL
```

Every section is compared with `==`, floats included. There are no tolerances.

* **trajectories**: 50 sessions (0–49). For each: the initial observation and
  actions, then a fixed script with 2 differently-phrased searches (full
  instruction, truncated, dataset query, original case, keywords, punctuation
  and specials, single word, upper-case reordered, noise), pagination forward and
  back, back-to-search, an invalid click, a product open (sometimes from page 2),
  option clicks, every sub-page, an empty search, and Buy Now. Actions are
  chosen once, at dump time, and replayed verbatim. In total there are 899
  observation units (obs + available actions + reward + done + URL); 48 sessions
  buy something, 27 of them with a nonzero reward.
* **topk**: top-50 docids and scores for 400 queries (200 goal instructions plus
  200 short 3-word variants).
* **goals**: all 6910 goals (instruction, asin, price cap, options, weight).
* **prices**: the full price table.
* **rewards**: `get_reward` and its info dict over 8477 (goal, product, options)
  pairs, from 300 goals × their top-10 search results plus the goal's own product,
  with no options, first options, and goal options.

### Results (`equivalence/results/`)

| candidate | trajectories | top-k exact@10 | exact@50 | identical scores | goals | prices | rewards | verdict |
|---|---|---|---|---|---|---|---|---|
| py3.8 self-check (pyserini, JDK 11) | 899/899 | 400/400 | 400/400 | 400/400 | 0 diff | 0 diff | 0 diff | IDENTICAL |
| **py3.13 + pyjnius, JDK 21** | 899/899 | 400/400 | 400/400 | 400/400 | 0 diff | 0 diff | 0 diff | **IDENTICAL** |
| **py3.13 + pyjnius, JDK 25** | 899/899 | 400/400 | 400/400 | 400/400 | 0 diff | 0 diff | 0 diff | **IDENTICAL** |

Two dumps from separate 3.8 processes are byte-identical. The golden data is
byte-identical whether it was dumped from the unmodified upstream source or from
this branch's source. Negative controls: before the `sum()` fix, 3.13 differed
on 1456/6910 goal weights; with `en_core_web_sm` 3.8.0, 1510/8477 rewards
differed. The harness does catch differences of this kind.

### Pre-existing caveat (in 3.8 too): the first env built in a process is random

`load_products` draws per-product prices from the **unseeded** global `random`,
and goals derive their "price lower than X dollars" caps from those draws.
`SimServer` only calls `random.seed(233)` *after* that. So:

* the **first** `WebAgentTextEnv` constructed in a process gets random price caps
  (two 3.8 processes gave session 0 "lower than 40.00" vs "50.00 dollars");
* every **later** construction in the same process is deterministic, because it
  starts from the post-`seed(233)` state.

The harness builds the env twice and uses the second. A caller that builds one
env per episode therefore gets a random task 0 and deterministic later tasks.
Constructing envs from several threads at once makes the later ones random as
well. This is identical in 3.8 and 3.13 and is not a porting artefact.

multiagent-planning's adapter avoids this: it builds every env on a private
`random.Random` that starts in the post-`seed(233)` state, under a lock
(`_construct` in `src/envs/webshop_env.py`), so every task gets the deterministic
instructions and the global RNG is untouched.
