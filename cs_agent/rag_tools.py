"""Knowledge-base search tools backed by Redis (RediSearch).

kb_search_bm25: full-text BM25 search (OR-semantics keyword query).
kb_search_vector: HNSW vector search over gemini-embedding-001 embeddings
(available only when the index was built with embeddings).

Replies are parsed via execute_command so both the classic array reply and
the Redis 8 map-style reply work regardless of redis-py version."""

import json
import os
import re
import struct
import sys

import redis

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
KB_INDEX = "kb_idx"
DOC_PREFIX = "doc:"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768

# Diagnostic: when KB_QUERY_LOG is set, log every agent-issued KB query to stderr
# (one "KBQUERY {json}" line). Off by default; used to capture the real query
# distribution for retrieval tuning. No effect on behavior.
_QUERY_LOG = bool(os.environ.get("KB_QUERY_LOG"))


def _log_query(tool: str, query: str, top_k: int) -> None:
    if _QUERY_LOG:
        print("KBQUERY " + json.dumps({"tool": tool, "k": top_k, "query": query}),
              file=sys.stderr, flush=True)

_client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
_genai_client = None


def _get_genai_client():
    """Reused genai client (one connection pool, not a new one per search)."""
    global _genai_client
    if _genai_client is None:
        from google import genai

        _genai_client = genai.Client()
    return _genai_client


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed texts with gemini-embedding-001 via google-genai.

    Note: RETRIEVAL_DOCUMENT/RETRIEVAL_QUERY task_type asymmetry was A/B tested
    (see RESULTS.md) and measured neutral-to-negative at dim=768 on this data, so
    it is intentionally left off. Revisit with a real-query labeled set / e2e A/B.
    """
    from google.genai import types

    # Reduced-dim output is unnormalized; the index uses COSINE, so that's fine.
    result = _get_genai_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
    )
    return [e.values for e in result.embeddings]


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _parse_search_reply(reply) -> list[dict]:
    """Normalize an FT.SEARCH reply (array or map shape) to result dicts."""
    if isinstance(reply, dict):
        results = reply.get(b"results", reply.get("results")) or []
        out = []
        for row in results:
            attrs = row.get(b"extra_attributes", row.get("extra_attributes")) or {}
            doc = {"doc_id": _decode(row.get(b"id", row.get("id", "")))}
            doc.update({_decode(k): _decode(v) for k, v in attrs.items()})
            out.append(doc)
        return out
    out = []
    for i in range(1, len(reply) - 1, 2):
        doc = {"doc_id": _decode(reply[i])}
        fields = reply[i + 1]
        for j in range(0, len(fields) - 1, 2):
            doc[_decode(fields[j])] = _decode(fields[j + 1])
        out.append(doc)
    return out


def _strip_score(docs: list[dict]) -> list[dict]:
    for doc in docs:
        doc.pop("score", None)
    return docs


def _bm25_search(query: str, top_k: int = 5) -> list[dict]:
    """Raw BM25 keyword search (no logging; used by the public tool and hybrid)."""
    terms = re.findall(r"\w+", query.lower())
    if not terms:
        return []
    # OR-join: RediSearch defaults to AND, which zeroes out long queries.
    or_query = "|".join(dict.fromkeys(terms))
    reply = _client.execute_command(
        "FT.SEARCH", KB_INDEX, or_query,
        "LIMIT", "0", str(top_k),
        "RETURN", "2", "title", "content",
    )
    return _parse_search_reply(reply)


def kb_search_bm25(query: str, top_k: int = 5) -> list[dict]:
    """Full-text (BM25) search over the Rho-Bank knowledge base.

    Args:
        query: Keywords or a short phrase to search for. Matching is ranked,
            so extra keywords help rather than hurt.
        top_k: Number of documents to return.

    Returns:
        Matching documents with doc_id, title, and full content.
    """
    _log_query("bm25", query, top_k)
    return _bm25_search(query, top_k)


def _vector_search(query: str, top_k: int = 5) -> list[dict]:
    """Raw semantic (vector) search. Returns docs, or [{"error": ...}] on failure."""
    try:
        vector = struct.pack(f"{EMBEDDING_DIM}f", *_embed([query])[0])
        reply = _client.execute_command(
            "FT.SEARCH", KB_INDEX, f"*=>[KNN {top_k} @embedding $vec AS score]",
            "PARAMS", "2", "vec", vector,
            "SORTBY", "score",
            "LIMIT", "0", str(top_k),
            "RETURN", "3", "title", "content", "score",
            "DIALECT", "2",
        )
        return _strip_score(_parse_search_reply(reply))
    except Exception as e:
        return [
            {
                "error": f"Vector search unavailable ({type(e).__name__}). "
                "Use kb_search_bm25 with keywords instead."
            }
        ]


# Hybrid fusion config. Tuned on the real CS-agent query distribution (short,
# specific lookups -- median ~5 words; see tests/rag/RESULTS.md). Weighting the
# semantic leg above keyword (2:1) beats pure vector on that regime and is never
# worse, at ~+3ms latency (one local BM25 call on top of the embedding we pay).
RRF_POOL = 50  # candidates per leg fed into fusion (embedding cost is flat in this)
RRF_K = 60  # standard reciprocal-rank-fusion damping constant
VECTOR_WEIGHT = 2.0
BM25_WEIGHT = 1.0


def _is_error(docs: list[dict]) -> bool:
    return len(docs) == 1 and "error" in docs[0]


def _rrf_fuse(weighted_rankings: list[tuple[float, list[dict]]], top_k: int, k: int = RRF_K) -> list[dict]:
    """Weighted reciprocal-rank fusion of several ranked doc lists.

    score(doc) = sum over lists of weight / (k + rank), rank starting at 1.
    Keeps the first-seen payload per doc_id; errored lists are skipped.
    """
    scores: dict[str, float] = {}
    payload: dict[str, dict] = {}
    for weight, ranking in weighted_rankings:
        if _is_error(ranking):
            continue
        for rank, doc in enumerate(ranking, start=1):
            doc_id = doc.get("doc_id")
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
            payload.setdefault(doc_id, doc)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [payload[doc_id] for doc_id in ordered[:top_k]]


def kb_search_vector(query: str, top_k: int = 5) -> list[dict]:
    """Hybrid semantic + keyword search over the Rho-Bank knowledge base.

    Fuses an embedding (semantic) search with a BM25 (keyword) search via
    reciprocal-rank fusion, so it handles both natural-language questions and
    exact terms (account names, internal tool names). This is the best
    general-purpose search -- prefer it for most lookups. Degrades to keyword
    search automatically if embeddings are unavailable.

    Args:
        query: A natural-language question, phrase, or keywords.
        top_k: Number of documents to return.

    Returns:
        Matching documents with doc_id, title, and full content; or an error
        entry telling you to fall back to kb_search_bm25.
    """
    _log_query("vector", query, top_k)
    vec = _vector_search(query, top_k=RRF_POOL)
    bm = _bm25_search(query, top_k=RRF_POOL)
    fused = _rrf_fuse([(VECTOR_WEIGHT, vec), (BM25_WEIGHT, bm)], top_k=top_k)
    # Fall back to whichever leg has content (surfacing the vector error if that
    # was the empty one) so the caller still gets a usable result or a hint.
    return fused or (vec if _is_error(vec) else bm)
