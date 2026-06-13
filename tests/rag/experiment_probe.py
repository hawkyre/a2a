"""Scratch: precise single-doc retrieval probe (short-query regime).

The scenario-level benchmark uses ~400-word queries, which favor dense vector
search. But the CS agent issues SHORT lookups for a specific doc -- and that's
where fabrication risk lives. Here each query is a single doc's title and the
only relevant doc is that doc. Measures whether short queries change the
bm25 / vector / hybrid picture.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import _ids, ensure_index  # noqa: E402
from metrics import aggregate  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "kb" / "documents"
GOLD = [json.loads(l) for l in (Path(__file__).resolve().parent / "gold.jsonl").read_text().splitlines() if l.strip()]
POOL = 50


def title_of(doc_id: str) -> str | None:
    p = DOCS / f"{doc_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("title")


def wrrf(vec_ids, bm_ids, w_vec, w_bm, k=60):
    scores = {}
    for rank, d in enumerate(vec_ids, 1):
        scores[d] = scores.get(d, 0.0) + w_vec / (k + rank)
    for rank, d in enumerate(bm_ids, 1):
        scores[d] = scores.get(d, 0.0) + w_bm / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def main():
    rag = ensure_index()
    # unique docs referenced by train tasks -> probe each by its title
    doc_ids = sorted({d for r in GOLD if r["split"] == "train" for d in r["required_documents"]})
    probes = [(d, title_of(d)) for d in doc_ids]
    probes = [(d, t) for d, t in probes if t]
    print(f"{len(probes)} title probes\n")

    cache = []
    t0 = time.perf_counter()
    for doc_id, title in probes:
        vec = _ids(rag._vector_search(title, top_k=POOL))
        bm = _ids(rag.kb_search_bm25(title, top_k=POOL))
        cache.append((vec, bm, {doc_id}))
    print(f"embedded {len(probes)} probes in {time.perf_counter()-t0:.1f}s\n")

    configs = {
        "pure_vector": lambda v, b: v,
        "pure_bm25": lambda v, b: b,
        "rrf_1:1": lambda v, b: wrrf(v, b, 1, 1),
        "rrf_1:2": lambda v, b: wrrf(v, b, 1, 2),
        "rrf_2:1": lambda v, b: wrrf(v, b, 2, 1),
    }
    cols = ["recall@1", "recall@5", "recall@10", "mrr"]
    rows = {}
    for name, fn in configs.items():
        pq = [{"ranked": fn(v, b), "relevant": rel} for v, b, rel in cache]
        rows[name] = aggregate(pq, (1, 5, 10))

    w = max(len(n) for n in rows) + 2
    print("config".ljust(w) + "".join(c.rjust(11) for c in cols))
    print("-" * (w + 11 * len(cols)))
    for name, m in rows.items():
        print(name.ljust(w) + "".join(f"{m[c]:.3f}".rjust(11) for c in cols))


if __name__ == "__main__":
    main()
