#!/bin/bash
# Set up WebShop's text env on a modern Python (verified on 3.13) WITHOUT pyserini.
#
#   ./setup_py313.sh            # into the active python (a venv)
#   PYTHON=/path/to/python ./setup_py313.sh
#
# Search keeps the exact original ranking: the SAME Anserini 0.14.3 jar that
# pyserini 0.17 bundles (fetched here from Maven Central, checksum-verified) is
# driven through pyjnius against the SAME prebuilt Lucene 8 indexes. Needs any
# JDK >= 11 on JAVA_HOME (verified identical on 11, 21 and 25).
#
# Data and indexes are NOT rebuilt here: data/ and search_engine/indexes* come
# from the original setup (setup.sh / setup_arm.sh + run_indexing.sh), which
# needs the py3.8 env only once. See README_PY313.md.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}

JAR_VER=0.14.3
JAR=search_engine/jars/anserini-${JAR_VER}-fatjar.jar
JAR_URL=https://repo1.maven.org/maven2/io/anserini/anserini/${JAR_VER}/anserini-${JAR_VER}-fatjar.jar
# sha256 of the jar inside the pyserini 0.17.0 wheel (identical to Maven's).
JAR_SHA256=e7bb590eae31cfa074b17174471d31f8d97053bc19d7332b946a855f0f66b230

"$PY" -m pip install -r requirements_py313.txt
# The spaCy model is pinned to the one the py3.8 reference uses. Its metadata
# declares spacy<3.8, which has no 3.13 wheels, so install it without deps; it
# loads under spaCy 3.8 (warning W095) and the equivalence harness checks that
# its tags -- hence rewards -- are unchanged.
"$PY" -m pip install --no-deps \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

mkdir -p search_engine/jars
if [ ! -f "$JAR" ]; then
  curl -fL -o "$JAR.part" "$JAR_URL"
  mv "$JAR.part" "$JAR"
fi
echo "$JAR_SHA256  $JAR" | shasum -a 256 -c -

for d in data/items_shuffle_1000.json search_engine/indexes_1k; do
  [ -e "$d" ] || echo "WARNING: $d missing -- run the original data/index setup once (README_PY313.md)"
done
echo "Done. Verify with: $PY equivalence/harness.py replay --golden equivalence/golden/golden_py38.json.gz"
