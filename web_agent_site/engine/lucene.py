"""Lucene searcher that does not need pyserini.

WebShop's ranking comes from Anserini 0.14.3 (Lucene 8) reading the prebuilt
``search_engine/indexes*`` directories. pyserini 0.17 -- the only pyserini that
bundles that Anserini -- is pinned to Python 3.8, and modern pyserini ships
Lucene 10, which refuses Lucene 8 indexes outright (IndexFormatTooOldException).

pyserini's ``LuceneSearcher`` is a thin pyjnius wrapper around Anserini's
``io.anserini.search.SimpleSearcher``, so this module calls that same Java class
through pyjnius (which has wheels for current Pythons) with the same jar: same
Lucene, same analyzer, same BM25 parameters, same index -- the same ranking by
construction, not by approximation. ``equivalence/harness.py`` checks it.

Backend selection (``WEBSHOP_SEARCH_BACKEND``):

* ``auto`` (default) -- pyserini if pyserini 0.17.x is installed (the original
  py3.8 setup, untouched), otherwise the pyjnius shim below.
* ``pyserini`` / ``jnius`` -- force one.

The shim needs the Anserini jar (``ANSERINI_JAR``, default
``search_engine/jars/anserini-0.14.3-fatjar.jar``; ``setup_py313.sh`` fetches it
from Maven Central and verifies its checksum) and a JDK >= 11 (``JAVA_HOME``).
"""

import os

from web_agent_site.utils import BASE_DIR

ANSERINI_VERSION = '0.14.3'
DEFAULT_JAR = os.path.join(
    BASE_DIR, '..', 'search_engine', 'jars', f'anserini-{ANSERINI_VERSION}-fatjar.jar'
)


def _pyserini_017_installed():
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:  # pragma: no cover - py<3.8
        return False
    try:
        return version('pyserini').startswith('0.17')
    except PackageNotFoundError:
        return False


class _Document:
    def __init__(self, lucene_document):
        self.object = lucene_document

    def raw(self):
        return self.object.get('raw')


class JniusLuceneSearcher:
    """The subset of pyserini 0.17's ``LuceneSearcher`` that WebShop uses.

    Mirrors pyserini 0.17 exactly on that subset: ``search`` calls
    ``SimpleSearcher.search(String, int)`` with no query generator, fields, RM3
    or dedup (pyserini's defaults), and ``doc(...).raw()`` reads the stored
    ``raw`` field.
    """

    def __init__(self, index_dir):
        _start_jvm()
        from jnius import autoclass
        self.object = autoclass('io.anserini.search.SimpleSearcher')(index_dir)
        self.num_docs = self.object.getTotalNumDocuments()

    def search(self, q, k=10):
        return list(self.object.search(q, k))

    def doc(self, docid):
        lucene_document = self.object.document(docid)
        if lucene_document is None:
            return None
        return _Document(lucene_document)


def _start_jvm():
    import jnius_config
    if jnius_config.vm_running:
        return
    jar = os.path.abspath(os.environ.get('ANSERINI_JAR', DEFAULT_JAR))
    if not os.path.isfile(jar):
        raise FileNotFoundError(
            f'Anserini jar not found at {jar}. Run ./setup_py313.sh (downloads '
            f'anserini-{ANSERINI_VERSION}-fatjar.jar from Maven Central) or set ANSERINI_JAR.'
        )
    jnius_config.add_classpath(jar)


def _backend():
    choice = os.environ.get('WEBSHOP_SEARCH_BACKEND', 'auto').lower()
    if choice not in ('auto', 'pyserini', 'jnius'):
        raise ValueError(f'WEBSHOP_SEARCH_BACKEND={choice!r}: expected auto, pyserini or jnius')
    if choice == 'auto':
        choice = 'pyserini' if _pyserini_017_installed() else 'jnius'
    return choice


def LuceneSearcher(index_dir):
    if _backend() == 'pyserini':
        from pyserini.search.lucene import LuceneSearcher as _PyseriniSearcher
        return _PyseriniSearcher(index_dir)
    return JniusLuceneSearcher(index_dir)
