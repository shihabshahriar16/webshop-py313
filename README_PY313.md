# WebShop text env on Python 3.13

The text environment (`web_agent_site.envs.WebAgentTextEnv`) runs on Python 3.13
**without pyserini**. It reproduces the original Python 3.8 env exactly:
observations, available actions, search rankings with their scores, goals, prices
and rewards. `equivalence/harness.py` checks that claim (see below).

Only the text env is covered. The Selenium site env, the Flask web app, and
`baseline_models/` were not ported.

Contents: [Setup](#setup) · [Using the env](#using-the-env) ·
[Task viewer](#task-viewer) · [What changed and why](#what-changed-and-why) ·
[Equivalence harness](#equivalence-harness)

## Setup

Tested on macOS (arm64) with Python 3.13.9. Every step runs on 3.13; no Python 3.8,
conda or pyserini is needed.

**Prerequisites**

* Python 3.13
* a JDK, version 11 or newer (verified identical on 11, 21 and 25):
  * macOS: `brew install openjdk@21`
  * Ubuntu/Debian: `sudo apt-get install openjdk-21-jdk-headless`
* `git` and `curl`
* about 250 MB of disk space

**Steps**

```sh
git clone https://github.com/shihabshahriar16/webshop-py313
cd webshop-py313
python3.13 -m venv .venv && source .venv/bin/activate
./setup_py313.sh
```

`setup_py313.sh` runs these steps. Each one is skipped if its output already
exists, so it is safe to re-run:

1. installs `requirements_py313.txt`, plus the pinned spaCy model
   `en_core_web_sm==3.7.1` (with `--no-deps`; see [the pins](#pins-that-matter-each-one-checked-with-the-harness));
2. downloads `anserini-0.14.3-fatjar.jar` from Maven Central into
   `search_engine/jars/`;
3. downloads the 1000-product data set into `data/`. WebShop's original Google
   Drive links are dead, so it uses the HuggingFace mirror
   [`YWZBrandon/webshop-data`](https://huggingface.co/datasets/YWZBrandon/webshop-data);
4. converts the products into the search corpus
   (`search_engine/resources*/documents.jsonl`);
5. builds the Lucene indexes (`search_engine/indexes*`) with Anserini's own
   indexer from the same jar, using the same arguments as
   `search_engine/run_indexing.sh`.

Steps 2–4 check every file against the SHA-256 of the file the Python 3.8
reference used, and stop on a mismatch.

The script finds Java through `JAVA_HOME`, then `java` on `PATH`, then (on
macOS) `/usr/libexec/java_home`. To use another Python, run
`PYTHON=/path/to/python ./setup_py313.sh`.

**Verify** (about a minute): this replays the Python 3.8 reference and exits 0
only if every observation, ranking, goal, price and reward is identical.

```sh
python equivalence/harness.py replay --golden equivalence/golden/golden_py38.json.gz
# ... VERDICT: IDENTICAL
```

## Using the env

```python
from web_agent_site.envs import WebAgentTextEnv   # run from the repo root

env = WebAgentTextEnv(observation_mode="text", num_products=1000)
obs, _ = env.reset(session=0)             # session = goal index (0..6909)
print(env.instruction_text)
print(env.get_available_actions())        # {'has_search_bar': True, 'clickables': [...]}
obs, reward, done, info = env.step("search[decorative pillows 28 x 28]")
obs, reward, done, info = env.step("click[b0743jkhbv]")   # a product from the results
obs, reward, done, info = env.step("click[buy now]")      # reward in [0, 1]
```

Actions are `search[<free text>]` and `click[<one of the clickables>]`.

Note that the first env built in a process gets random goal price caps; see
[the caveat](#pre-existing-caveat-in-38-too-the-first-env-built-in-a-process-is-random)
below. For reproducible goals, build the env twice and use the second, as the
harness and the viewer do.

## Task viewer

`viewer/index.html` is a browser for WebShop tasks. For each task it shows:

* exactly what an agent receives (instruction, observation, available actions);
* what is hidden from the agent (the reference query and source product);
* how the grader scores a purchase: the attributes, options and price cap, and
  the reward formula;
* several *different* products actually bought for the same goal, with the
  reward each one earned (WebShop gives partial credit, and more than one product
  can score 1.0);
* a step-by-step reference trajectory.

**Live: [https://shihabshahriar16.github.io/webshop-py313/](https://shihabshahriar16.github.io/webshop-py313/)**. This is published from `viewer/` by
`.github/workflows/pages.yml` on every push that changes `viewer/`.

The page is a **static snapshot**. `build_viewer.py` runs the real environment
(Java search, the grader) offline and bakes the results into
`viewer/webshop_examples.js`; the browser only displays them and cannot run new
tasks itself. The committed snapshot has the first 50 tasks (sessions 0–49).

To open it locally (works straight after cloning, without any setup):

```sh
open viewer/index.html                                   # macOS; or double-click it
# or, if your browser blocks local scripts:
python3 -m http.server -d viewer 8000                    # then visit http://localhost:8000
```

Use the search box and the filter chips (target on page 1 / needs pagination /
reference failed) to find tasks, and click **How scoring works** for the reward
rules.

To regenerate it, or to include other tasks (this needs the setup above):

```sh
python viewer/build_viewer.py -n 100 --start 0        # ~0.2 s per task, ~21 KB per task
python viewer/build_viewer.py -n 100 --no-alternatives  # faster: skip the extra purchases
```

Commit and push `viewer/webshop_examples.js` and the live page updates by itself.

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

How to avoid it as a caller: either build the env twice and use the second
(simple, single-threaded), or build each env on a private `random.Random` that
starts in the post-`seed(233)` state, by pointing the `random` global of the
`web_agent_site` modules at it while constructing, under a lock. The second
approach also keeps concurrent builds from interleaving and leaves the global
RNG untouched. Both give the instructions shown in the viewer and stored in the
golden file.

### Harmless Java warnings

On JDK 21 and newer, Lucene 8 prints
`WARNING: A terminally deprecated method in sun.misc.Unsafe has been called ...`
when it opens an index. This is expected and does not affect results (the
harness passes on JDK 21 and 25). JDK 11 prints nothing.
