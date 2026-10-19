#!/bin/bash
# Complete setup of WebShop's text env on Python 3.13, from a fresh clone.
# No Python 3.8, conda or pyserini needed at any step.
#
#   python3.13 -m venv .venv && source .venv/bin/activate
#   ./setup_py313.sh                      # uses `python` on PATH
#   PYTHON=/path/to/python ./setup_py313.sh
#
# Needs: curl, and a JDK >= 11 (JAVA_HOME, or `java` on PATH).
# Every step is skipped if its output already exists, so re-running is safe.
#
#   1. Python packages (requirements_py313.txt) + the pinned spaCy model
#   2. Anserini 0.14.3 jar from Maven Central (the jar pyserini 0.17 bundles)
#   3. the product data (1000-product "small" set) from the HuggingFace mirror
#   4. search corpus: search_engine/resources*/documents.jsonl
#   5. Lucene indexes: search_engine/indexes*, built by Anserini itself
#
# Steps 2-4 are checked against the SHA-256 of the files the py3.8 reference
# used, so a mismatch stops here instead of silently changing the benchmark.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
verify() {  # verify <file> <sha256>
  local got; got=$(sha256 "$1")
  if [ "$got" != "$2" ]; then
    echo "ERROR: $1 has sha256 $got, expected $2" >&2
    echo "       This is not the file the reference env used; refusing to continue." >&2
    exit 1
  fi
  echo "  ok  $1"
}
step() { echo; echo "== $*"; }

# --- 0. Java -------------------------------------------------------------------
JAVA="${JAVA_HOME:+$JAVA_HOME/bin/}java"
if ! "$JAVA" -version >/dev/null 2>&1; then
  if [ -x /usr/libexec/java_home ] && JH=$(/usr/libexec/java_home -v 11+ 2>/dev/null); then
    export JAVA_HOME=$JH; JAVA=$JH/bin/java
  else
    echo "ERROR: no JDK found. Install one (>= 11), e.g." >&2
    echo "  macOS:  brew install openjdk@21" >&2
    echo "  Ubuntu: sudo apt-get install openjdk-21-jdk-headless" >&2
    echo "then set JAVA_HOME (or put java on PATH) and re-run." >&2
    exit 1
  fi
fi
echo "Java: $("$JAVA" -version 2>&1 | head -1)   Python: $("$PY" --version)"

# --- 1. Python packages ----------------------------------------------------------
step "1/5 Python packages"
"$PY" -m pip install -r requirements_py313.txt
# Pinned to the model the py3.8 reference uses. Its metadata says spacy<3.8,
# which has no 3.13 wheels, so install it without deps; it loads under spaCy 3.8
# (warning W095). Do NOT upgrade it: en_core_web_sm 3.8.0 changed 1510 of 8477
# rewards in the equivalence harness.
"$PY" -m pip install --no-deps \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# --- 2. Anserini jar -----------------------------------------------------------
step "2/5 Anserini jar"
JAR_VER=0.14.3
JAR=search_engine/jars/anserini-${JAR_VER}-fatjar.jar
mkdir -p search_engine/jars
if [ ! -f "$JAR" ]; then
  curl -fL -o "$JAR.part" \
    "https://repo1.maven.org/maven2/io/anserini/anserini/${JAR_VER}/anserini-${JAR_VER}-fatjar.jar"
  mv "$JAR.part" "$JAR"
fi
verify "$JAR" e7bb590eae31cfa074b17174471d31f8d97053bc19d7332b946a855f0f66b230

# --- 3. product data -------------------------------------------------------------
step "3/5 product data (1000-product set)"
# WebShop's original Google Drive links are dead (HTTP 404); this mirror serves
# byte-identical files (checked by the hashes below).
MIRROR=https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/main
mkdir -p data
while read -r name hash; do
  if [ ! -f "data/$name" ]; then
    curl -fL -o "data/$name.part" "$MIRROR/$name"
    mv "data/$name.part" "data/$name"
  fi
  verify "data/$name" "$hash"
done <<'EOF'
items_shuffle_1000.json 30a4765c3a327af72d9a9a95a6b2486d516f0fa1d3ecd83681901ce82a21b269
items_ins_v2_1000.json f88a36314a397b53b3d9c3fa5878e5f7b26d35019a51ec83fbedeca61a948f6f
items_human_ins.json cf78667548a71786e1d9049c24b802e48e1084ad4bb021cae56ce1f6d96954a3
EOF

# --- 4. search corpus ------------------------------------------------------------
step "4/5 search corpus"
SETS="resources resources_100 resources_1k resources_100k"
need_convert=0
for s in $SETS; do [ -f "search_engine/$s/documents.jsonl" ] || need_convert=1; done
if [ $need_convert = 1 ]; then
  for s in $SETS; do mkdir -p "search_engine/$s"; done
  (cd search_engine && "$PY" convert_product_file_format.py >/dev/null)
fi
# resources, resources_1k and resources_100k are the same 1000 products.
for s in resources resources_1k resources_100k; do
  verify "search_engine/$s/documents.jsonl" 24fb9257c435accee9cc82e570dc8ec76a218d3b188222a84fdeb7fc5937d00e
done
verify search_engine/resources_100/documents.jsonl cada7cd4f1badbb7651e801c6bee72997080974ac5093337bc52d11bcd9481f0

# --- 5. indexes ------------------------------------------------------------------
step "5/5 Lucene indexes"
# Same command search_engine/run_indexing.sh runs through pyserini 0.17
# (pyserini.index.lucene is a wrapper around this Anserini class).
for s in $SETS; do
  idx=search_engine/${s/resources/indexes}
  if [ -f "$idx/segments_2" ] || ls "$idx"/segments_* >/dev/null 2>&1; then
    echo "  ok  $idx (exists)"; continue
  fi
  rm -rf "$idx"
  "$JAVA" -cp "$JAR" io.anserini.index.IndexCollection \
    -collection JsonCollection -input "search_engine/$s" -index "$idx" \
    -generator DefaultLuceneDocumentGenerator -threads 1 \
    -storePositions -storeDocvectors -storeRaw >/dev/null 2>&1 \
    || { echo "ERROR: indexing $s failed; re-run the java command above without >/dev/null to see why" >&2; exit 1; }
  echo "  built $idx"
done

echo
echo "Done. Verify against the py3.8 reference (exit 0 = IDENTICAL, ~1 min):"
echo "  $PY equivalence/harness.py replay --golden equivalence/golden/golden_py38.json.gz"
