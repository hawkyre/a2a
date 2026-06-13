# RAG retrieval benchmark — results log

Layer-1 microbenchmark over `gold.jsonl` (queries = task scenarios, relevant =
task `required_documents`). Run: `.venv/bin/python tests/rag/eval.py --split <split>`.
Latency p50/p95 in ms; vector/hybrid latency is dominated by the embedding API call.

## Baseline (train, 58 queries) — commit before hybrid

| retriever | recall@5 | recall@10 | recall@20 | ndcg@5 | ndcg@10 | ndcg@20 | mrr | p50_ms | p95_ms |
|---|---|---|---|---|---|---|---|---|---|
| bm25   | 0.188 | 0.226 | 0.314 | 0.274 | 0.254 | 0.277 | 0.493 | 3.2 | 4.3 |
| vector | 0.277 | 0.374 | 0.454 | 0.479 | 0.455 | 0.460 | 0.809 | 292.9 | 618.9 |

Takeaways: vector > bm25 on all quality metrics; bm25 ~100x faster (no embed call).
Hypothesis: RRF(bm25, vector) > vector at ~vector latency (+~3ms for the bm25 leg).

## Hybrid RRF — REJECTED on scenario queries (experiment.py, train)

| config | recall@5 | recall@10 | recall@20 | ndcg@10 | mrr |
|---|---|---|---|---|---|
| pure_vector | 0.277 | 0.374 | 0.454 | 0.455 | **0.809** |
| pure_bm25 | 0.188 | 0.226 | 0.314 | 0.254 | 0.495 |
| rrf 1:1 | 0.250 | 0.320 | 0.406 | 0.390 | 0.716 |
| rrf 3:1 | 0.273 | 0.338 | 0.451 | 0.413 | 0.728 |
| rrf 5:1 | 0.272 | 0.352 | **0.460** | 0.424 | 0.737 |

No fusion weighting beats pure vector on MRR/nDCG for long scenario queries.
Naive equal-weight hybrid actively hurts. **Not shipped.**

## Hybrid RRF — WINS on short precise queries (experiment_probe.py, 199 title probes)

| config | recall@1 | recall@5 | mrr |
|---|---|---|---|
| pure_vector | 0.955 | 1.000 | 0.977 |
| pure_bm25 | 0.859 | 0.995 | 0.919 |
| rrf 1:1 | 0.980 | 1.000 | 0.989 |
| **rrf 2:1** | **0.985** | 1.000 | **0.992** |

For short, specific lookups (a single target doc), weighted hybrid (vector 2 :
bm25 1) beats pure vector and is never worse.

## Real agent query distribution (KB_QUERY_LOG, 3 harness tasks, 83 queries)

The CS agent issues SHORT lookups: median 5 words, max 13, 74% <= 8 words, none
verbose. It uses both tools (vector 47, bm25 36). => the scenario benchmark
(~400-word queries) is the WRONG proxy; the short-query regime is reality, and
there weighted hybrid wins. Real queries saved to real_queries.jsonl.

## SHIPPED: weighted hybrid RRF (vector 2 : bm25 1) — validate_hybrid.py

| retriever | recall@1 | recall@5 | mrr | p50_ms | p95_ms |
|---|---|---|---|---|---|
| pure_vector | 0.955 | 1.000 | 0.977 | 277 | 562 |
| hybrid (shipped) | **0.985** | 1.000 | **0.992** | 250 | 370 |

Quality strictly up on the real (short) regime; no latency cost (embedding call
dominates, bm25 leg adds ~3ms, lost in network noise). kb_search_vector is now
this hybrid; kb_search_bm25 unchanged. Tool names/signatures/return shape stable.

## REJECTED: task_type asymmetry (RETRIEVAL_DOCUMENT/QUERY) — A/B'd, not shipped

Documented as a free accuracy lever, but measured neutral-to-negative here:
probe MRR flat (hybrid) / -0.004 (vector); scenario vector MRR 0.809 -> 0.735.
Likely the dim=768 truncation + this model's behavior. Left off; revisit only
with a real-query *labeled* set or an end-to-end A/B.

## Open levers (post-merge, retrieval-completeness; all in our files)
- top_k / RRF_POOL tuning so multi-doc tasks return ALL required_documents.
- completeness metric (returned set ⊇ required_documents) on real queries.
- query->vector LRU cache (latency; agent reformulates the same intent).
- dim 768 -> 1536/3072 A/B (cheap; re-bake required).

