"""Scratch: sweep fusion strategies with ONE embedding pass per query.

For each gold query we fetch the vector and bm25 candidate lists once, then
score many fusion configs offline (no re-embedding). Prints a comparison table.
Not a committed benchmark -- a tuning tool to find a config that beats pure vector.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import _ids, ensure_index  # noqa: E402
from metrics import aggregate  # noqa: E402

POOL = 50
GOLD = [json.loads(l) for l in (Path(__file__).resolve().parent / "gold.jsonl").read_text().splitlines() if l.strip()]
TRAIN = [r for r in GOLD if r["split"] == "train"]


def wrrf(vec_ids, bm_ids, w_vec, w_bm, k=60):
    scores = {}
    if w_vec:
        for rank, d in enumerate(vec_ids, 1):
            scores[d] = scores.get(d, 0.0) + w_vec / (k + rank)
    if w_bm:
        for rank, d in enumerate(bm_ids, 1):
            scores[d] = scores.get(d, 0.0) + w_bm / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def vector_then_bm(vec_ids, bm_ids):
    """Vector ranking first, then any bm25 docs vector missed, in bm25 order."""
    seen = set(vec_ids)
    return list(vec_ids) + [d for d in bm_ids if d not in seen]


def main():
    rag = ensure_index()
    # one embedding pass: cache candidates per query
    cache = []
    t0 = time.perf_counter()
    for r in TRAIN:
        vec = _ids(rag._vector_search(r["query"], top_k=POOL))
        bm = _ids(rag.kb_search_bm25(r["query"], top_k=POOL))
        cache.append((vec, bm, set(r["required_documents"])))
    print(f"embedded {len(TRAIN)} queries in {time.perf_counter()-t0:.1f}s\n")

    configs = {
        "pure_vector": lambda v, b: v,
        "pure_bm25": lambda v, b: b,
        "rrf_1:1": lambda v, b: wrrf(v, b, 1, 1),
        "rrf_2:1": lambda v, b: wrrf(v, b, 2, 1),
        "rrf_3:1": lambda v, b: wrrf(v, b, 3, 1),
        "rrf_5:1": lambda v, b: wrrf(v, b, 5, 1),
        "rrf_1:1_k20": lambda v, b: wrrf(v, b, 1, 1, k=20),
        "vec_then_bm": vector_then_bm,
    }
    rows = {}
    for name, fn in configs.items():
        pq = [{"ranked": fn(v, b), "relevant": rel} for v, b, rel in cache]
        rows[name] = aggregate(pq, (5, 10, 20))

    cols = ["recall@5", "recall@10", "recall@20", "ndcg@10", "mrr"]
    w = max(len(n) for n in rows) + 2
    print("config".ljust(w) + "".join(c.rjust(11) for c in cols))
    print("-" * (w + 11 * len(cols)))
    for name, m in rows.items():
        print(name.ljust(w) + "".join(f"{m[c]:.3f}".rjust(11) for c in cols))


if __name__ == "__main__":
    main()
