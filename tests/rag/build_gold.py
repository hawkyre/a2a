"""Build the retrieval gold set from the a2a-hackathon task splits.

Each harness task carries `required_documents` (hand-curated KB doc ids needed
to answer it) and a `user_scenario.instructions` role-play script. We treat the
scenario text as the retrieval query and required_documents as the relevant set
-- a task-level label. It over-lists (it includes docs for rejected options too),
but that noise is identical across retrievers, so relative comparison is sound.

Run: python tests/rag/build_gold.py   (regenerates tests/rag/gold.jsonl)

Source repo path is configurable via A2A_HACK_DIR; defaults to the sibling
clone described in the README ("clone it next to this repo").
"""

import json
import os
import re
from pathlib import Path

HACK_DIR = Path(
    os.environ.get("A2A_HACK_DIR", Path(__file__).resolve().parents[2].parent / "a2a-hackathon")
)
DATA_DIR = HACK_DIR / "src" / "a2a_hack" / "data"
TASKS_DIR = DATA_DIR / "tasks"
SPLITS_PATH = DATA_DIR / "banking_hackathon_splits.json"

OUT_PATH = Path(__file__).resolve().parent / "gold.jsonl"


def _category(doc_id: str) -> str:
    """Strip the trailing _NNN index to get the doc's product category."""
    return re.sub(r"_\d+$", "", doc_id)


def build() -> list[dict]:
    if not TASKS_DIR.exists():
        raise SystemExit(
            f"Harness tasks not found at {TASKS_DIR}.\n"
            "Clone a2a-hackathon next to this repo, or set A2A_HACK_DIR."
        )
    splits = json.loads(SPLITS_PATH.read_text())
    # task_id -> split name (a task appears in exactly one split)
    split_of = {tid: name for name, ids in splits.items() for tid in ids}

    rows = []
    for path in sorted(TASKS_DIR.glob("task_*.json")):
        task = json.loads(path.read_text())
        tid = task["id"]
        req = task.get("required_documents") or []
        if not req:
            continue  # no retrieval label -> not useful for a retrieval benchmark
        instr = (task.get("user_scenario") or {}).get("instructions") or ""
        rows.append(
            {
                "id": tid,
                "split": split_of.get(tid, "none"),
                "query": instr.strip(),
                "required_documents": req,
                "n_docs": len(req),
                "categories": sorted({_category(d) for d in req}),
            }
        )
    return rows


def main() -> None:
    rows = build()
    with open(OUT_PATH, "w") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")
    by_split: dict[str, int] = {}
    for r in rows:
        by_split[r["split"]] = by_split.get(r["split"], 0) + 1
    print(f"wrote {len(rows)} gold rows to {OUT_PATH}")
    print("by split:", by_split)
    print("avg required_documents:", round(sum(r["n_docs"] for r in rows) / len(rows), 1))


if __name__ == "__main__":
    main()
