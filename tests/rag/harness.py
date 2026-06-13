"""Shared setup for the RAG retrieval benchmark (used by eval.py and tests).

Responsibilities:
- load the repo .env so GOOGLE_* / REDIS_URL are present,
- put cs_agent on sys.path and point ingest at the repo's kb/,
- build the Redis index once per process,
- expose the retrievers under test as name -> (query, top_k) -> [doc_id].
"""

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CS_AGENT_DIR = REPO_ROOT / "cs_agent"
KB_DIR = REPO_ROOT / "kb"


def load_env() -> None:
    """Minimal .env loader (no dependency); does not overwrite existing vars."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def import_rag():
    """Import rag_tools + ingest with KB paths pointed at the repo (not /app)."""
    load_env()
    os.environ.setdefault("KB_DOCUMENTS_DIR", str(KB_DIR / "documents"))
    os.environ.setdefault("KB_EMBEDDINGS_PATH", str(KB_DIR / "embeddings.json"))
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
    if str(CS_AGENT_DIR) not in sys.path:
        sys.path.insert(0, str(CS_AGENT_DIR))
    import ingest
    import rag_tools

    return rag_tools, ingest


_index_built = False


def ensure_index(force: bool = False):
    rag_tools, ingest = import_rag()
    global _index_built
    if force or not _index_built:
        ingest.build_index()
        _index_built = True
    return rag_tools


def _strip(doc_id: str) -> str:
    return doc_id[len("doc:"):] if doc_id.startswith("doc:") else doc_id


def _ids(docs: list[dict]) -> list[str]:
    return [_strip(d["doc_id"]) for d in docs if isinstance(d, dict) and d.get("doc_id")]


def get_retrievers(rag_tools) -> dict:
    """name -> callable(query, top_k) -> ranked list of doc_ids."""
    return {
        "bm25": lambda q, k: _ids(rag_tools.kb_search_bm25(q, top_k=k)),
        # kb_search_vector == _vector_search (pure semantic); experiment_*.py
        # fuse the raw vector + bm25 candidates offline to test hybrid configs.
        "vector": lambda q, k: _ids(rag_tools.kb_search_vector(q, top_k=k)),
    }


def timed(fn, *args):
    """Return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = fn(*args)
    return result, (time.perf_counter() - t0) * 1000.0
