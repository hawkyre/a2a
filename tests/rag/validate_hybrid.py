"""Validate the SHIPPED hybrid kb_search_vector vs the pure-vector baseline.

Quality: precise single-doc probes (short-query regime = the real agent regime).
Latency: timed over the real captured agent queries (tests/rag/real_queries.jsonl).
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import _ids, ensure_index, timed  # noqa: E402
from metrics import aggregate  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DOCS = REPO / "kb" / "documents"
GOLD = [json.loads(l) for l in (HERE / "gold.jsonl").read_text().splitlines() if l.strip()]
REAL = [json.loads(l) for l in (HERE / "real_queries.jsonl").read_text().splitlines() if l.strip()]


def main():
    rag = ensure_index()
    pure = lambda q, k: _ids(rag._vector_search(q, top_k=k))
    hybrid = lambda q, k: _ids(rag.kb_search_vector(q, top_k=k))  # shipped tool

    # --- quality: title probes (one target doc each), train docs ---
    doc_ids = sorted({d for r in GOLD if r["split"] == "train" for d in r["required_documents"]})
    probes = []
    for d in doc_ids:
        p = DOCS / f"{d}.json"
        if p.exists():
            probes.append((d, json.loads(p.read_text())["title"]))
    print(f"quality: {len(probes)} precise probes\n")
    rows = {}
    for name, fn in (("pure_vector", pure), ("hybrid(shipped)", hybrid)):
        pq = [{"ranked": fn(t, 10), "relevant": {d}} for d, t in probes]
        rows[name] = aggregate(pq, (1, 5))
    w = 18
    print("retriever".ljust(w) + "recall@1".rjust(11) + "recall@5".rjust(11) + "mrr".rjust(11))
    for name, m in rows.items():
        print(name.ljust(w) + f"{m['recall@1']:.3f}".rjust(11) + f"{m['recall@5']:.3f}".rjust(11) + f"{m['mrr']:.3f}".rjust(11))

    # --- latency: real agent queries ---
    sample = [r["query"] for r in REAL][:40]
    print(f"\nlatency: {len(sample)} real agent queries\n")
    print("retriever".ljust(w) + "p50_ms".rjust(11) + "p95_ms".rjust(11))
    for name, fn in (("pure_vector", pure), ("hybrid(shipped)", hybrid)):
        lat = [timed(fn, q, 5)[1] for q in sample]
        print(name.ljust(w) + f"{statistics.median(lat):.1f}".rjust(11) + f"{sorted(lat)[int(len(lat)*0.95)-1]:.1f}".rjust(11))


if __name__ == "__main__":
    main()
