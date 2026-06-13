"""Retrieval metrics for ranked doc-id lists against a relevant set.

All functions take `ranked` (ordered list of doc_ids, best first) and `relevant`
(set/iterable of relevant doc_ids), and ignore duplicates in `ranked`.
"""

import math


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = list(dict.fromkeys(ranked))[:k]
    hits = sum(1 for d in top if d in relevant)
    return hits / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant doc (0 if none retrieved)."""
    for i, d in enumerate(dict.fromkeys(ranked), start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    if not relevant:
        return 0.0
    top = list(dict.fromkeys(ranked))[:k]
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(top, start=1) if d in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def aggregate(per_query: list[dict], cutoffs=(5, 10, 20)) -> dict:
    """Mean of each metric across queries. Each row: {ranked, relevant}."""
    n = len(per_query) or 1
    out = {}
    for k in cutoffs:
        out[f"recall@{k}"] = sum(recall_at_k(r["ranked"], r["relevant"], k) for r in per_query) / n
        out[f"ndcg@{k}"] = sum(ndcg_at_k(r["ranked"], r["relevant"], k) for r in per_query) / n
    out["mrr"] = sum(reciprocal_rank(r["ranked"], r["relevant"]) for r in per_query) / n
    return out
