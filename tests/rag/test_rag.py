"""RAG tests.

Unit tests (fusion math + metrics) run anywhere. Integration tests need Redis
on REDIS_URL and Google creds; they skip automatically if unavailable.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cs_agent"))

import metrics  # noqa: E402
import rag_tools  # noqa: E402

# ---------------- unit: metrics ----------------


def test_recall_at_k():
    assert metrics.recall_at_k(["a", "b", "c"], {"a", "x"}, 2) == 0.5
    assert metrics.recall_at_k(["a", "b"], set(), 2) == 0.0


def test_reciprocal_rank():
    assert metrics.reciprocal_rank(["x", "a"], {"a"}) == 0.5
    assert metrics.reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_ndcg_at_k():
    assert metrics.ndcg_at_k(["a", "b"], {"a"}, 2) == pytest.approx(1.0)
    assert metrics.ndcg_at_k(["b", "a"], {"a"}, 2) == pytest.approx(0.6309, abs=1e-3)


# ---------------- unit: RRF fusion ----------------


def test_rrf_weighting_and_overlap():
    vec = [{"doc_id": "v1"}, {"doc_id": "shared"}]
    bm = [{"doc_id": "shared"}, {"doc_id": "b1"}]
    fused = rag_tools._rrf_fuse([(2.0, vec), (1.0, bm)], top_k=3)
    ids = [d["doc_id"] for d in fused]
    assert ids[0] == "shared"  # appears in both -> highest fused score
    assert ids.index("v1") < ids.index("b1")  # vector leg weighted 2x


def test_rrf_skips_errored_leg():
    err = [{"error": "vector down"}]
    bm = [{"doc_id": "b1"}, {"doc_id": "b2"}]
    fused = rag_tools._rrf_fuse([(2.0, err), (1.0, bm)], top_k=2)
    assert [d["doc_id"] for d in fused] == ["b1", "b2"]


def test_is_error():
    assert rag_tools._is_error([{"error": "x"}])
    assert not rag_tools._is_error([{"doc_id": "a"}])
    assert not rag_tools._is_error([])


# ---------------- integration: real index ----------------


@pytest.fixture(scope="module")
def indexed():
    try:
        from harness import ensure_index

        return ensure_index()
    except Exception as e:  # redis/creds unavailable
        pytest.skip(f"integration unavailable: {type(e).__name__}: {e}")


def _ids(docs):
    return [d["doc_id"].removeprefix("doc:") for d in docs if d.get("doc_id")]


def test_known_lookup_returns_target_doc(indexed):
    docs = indexed.kb_search_vector("deposit a check with my phone", top_k=5)
    assert "doc_bank_accounts_bank_accounts_(general)_011" in _ids(docs)


def test_return_shape(indexed):
    docs = indexed.kb_search_vector("blue account monthly fee", top_k=3)
    assert docs and all({"doc_id", "title", "content"} <= set(d) for d in docs)


def test_hybrid_not_worse_than_vector_on_probes(indexed):
    """On precise short-query probes, shipped hybrid recall@1 >= pure vector."""
    import json

    docs_dir = Path(__file__).resolve().parents[2] / "kb" / "documents"
    gold = [json.loads(l) for l in (Path(__file__).resolve().parent / "gold.jsonl").read_text().splitlines() if l.strip()]
    sample = sorted({d for r in gold if r["split"] == "train" for d in r["required_documents"]})[:40]
    pure = hybrid = 0
    for doc_id in sample:
        title = json.loads((docs_dir / f"{doc_id}.json").read_text())["title"]
        pure += doc_id in _ids(indexed._vector_search(title, top_k=1))
        hybrid += doc_id in _ids(indexed.kb_search_vector(title, top_k=1))
    assert hybrid >= pure
