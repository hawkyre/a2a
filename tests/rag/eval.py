"""Layer-1 retrieval microbenchmark: run each retriever over the gold set and
print a comparison table (recall@k, nDCG@k, MRR, latency).

Usage:
    python tests/rag/eval.py                  # default split=train, K=20
    python tests/rag/eval.py --split test --k 20 --retrievers bm25,vector

Requires Redis on REDIS_URL and (for the vector retriever) Google creds in .env.
"""

import argparse
import json
import statistics
from pathlib import Path

from harness import ensure_index, get_retrievers, timed
from metrics import aggregate

GOLD_PATH = Path(__file__).resolve().parent / "gold.jsonl"
CUTOFFS = (5, 10, 20)


def load_gold(split: str | None) -> list[dict]:
    rows = [json.loads(l) for l in GOLD_PATH.read_text().splitlines() if l.strip()]
    if split and split != "all":
        rows = [r for r in rows if r["split"] == split]
    return rows


def run_retriever(fn, gold: list[dict], k: int) -> dict:
    per_query, latencies = [], []
    for row in gold:
        ranked, ms = timed(fn, row["query"], k)
        latencies.append(ms)
        per_query.append({"ranked": ranked, "relevant": set(row["required_documents"])})
    metrics = aggregate(per_query, CUTOFFS)
    metrics["p50_ms"] = statistics.median(latencies)
    metrics["p95_ms"] = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]
    return metrics


def print_table(results: dict[str, dict]) -> None:
    cols = [f"recall@{k}" for k in CUTOFFS] + [f"ndcg@{k}" for k in CUTOFFS] + ["mrr", "p50_ms", "p95_ms"]
    width = max(len(n) for n in results) + 2
    header = "retriever".ljust(width) + "".join(c.rjust(11) for c in cols)
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        row = name.ljust(width) + "".join(
            (f"{m[c]:.3f}" if c.endswith("ms") is False else f"{m[c]:.1f}").rjust(11) for c in cols
        )
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--retrievers", default="bm25,vector")
    args = ap.parse_args()

    rag_tools = ensure_index()
    retrievers = get_retrievers(rag_tools)
    chosen = [r.strip() for r in args.retrievers.split(",") if r.strip()]

    gold = load_gold(args.split)
    print(f"gold: {len(gold)} queries (split={args.split}), K={args.k}\n")

    results = {}
    for name in chosen:
        if name not in retrievers:
            print(f"  skip unknown retriever: {name}")
            continue
        results[name] = run_retriever(retrievers[name], gold, args.k)
    print_table(results)


if __name__ == "__main__":
    main()
