"""Deterministic reconcile flow for reward / cash-back discrepancy tasks.

The model demonstrably computes the correct reconciliation, then writes a
fabricated subset anyway (it doesn't trust its own work). This makes the
deterministic computation the ONLY thing that can reach the database:

1. `reconcile(items)` — the model passes EVERY transaction it retrieved
   ({transaction_id, amount, rate, recorded}). The tool computes
   expected = amount * rate * multiplier in code, returns the AUTHORITATIVE list
   of discrepancies + correct values, and stashes it in session state.
2. `reconcile_block` (before_tool) — an `update_transaction_rewards` write is
   allowed ONLY if its (transaction_id, value) matches the authoritative list.
   If the model hasn't reconciled yet, it's told to do so first. Confabulated,
   extra, or off-by-one writes are physically blocked.
3. `authoritative_reinject_text` — re-appended every turn so the model always
   sees the exact set of corrections to apply (all of them, none else).

Pure Python; fails open after a few blocks so a mistaken gate can't loop.
"""

import json
import os
import re
import sys

_LOG = bool(os.environ.get("KB_QUERY_LOG"))

RECON_KEY = "recon_corrections"   # {transaction_id: correct_points}
RECON_DONE = "recon_done"
RECON_BLOCKS = "recon_blocks"
MAX_BLOCKS = 4
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = _NUM.search(v)
        if m:
            return float(m.group())
    return None


def reconcile(items: list, tool_context, multiplier: float = 100.0) -> dict:
    """Deterministically find reward/cash-back discrepancies and the corrected values.

    Call this ONCE with EVERY transaction you retrieved (do not pre-filter). Each
    item must be {"transaction_id": str, "amount": <purchase dollars>, "rate":
    <the correct cash-back rate for that card+category from the knowledge base, as
    a decimal e.g. 0.04 for 4%>, "recorded": <the reward currently on the txn>}.
    `multiplier` converts cash-back dollars to the reward unit (100 for "points").

    Returns the AUTHORITATIVE list of incorrect transactions and their correct
    reward values. Apply update_transaction_rewards for EXACTLY these — all of
    them, none else.
    """
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except (ValueError, TypeError):
            items = []
    if not isinstance(items, list):
        return {"error": "items must be a list of {transaction_id, amount, rate, recorded}"}

    corrections = {}
    checked = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        tid = it.get("transaction_id") or it.get("id")
        amount = _num(it.get("amount"))
        rate = _num(it.get("rate"))
        recorded = _num(it.get("recorded", it.get("actual")))
        if not tid or amount is None or rate is None:
            continue
        checked += 1
        expected = round(amount * rate * multiplier)
        if recorded is None or abs(expected - recorded) >= 1:
            corrections[tid] = expected

    tool_context.state[RECON_KEY] = corrections
    tool_context.state[RECON_DONE] = True
    if _LOG:
        print(f"RECON | checked={checked} discrepancies={corrections}", file=sys.stderr, flush=True)
    return {
        "checked": checked,
        "discrepancies": [{"transaction_id": t, "correct_reward_points": v} for t, v in corrections.items()],
        "instruction": (
            "Now call update_transaction_rewards for EXACTLY these transactions, "
            "using new_rewards_earned='<points> points'. Apply ALL of them and "
            "NONE that are not listed."
        ),
    }


def _resolve(call_args):
    name = (call_args or {}).get("agent_tool_name") or (call_args or {}).get("tool_name") or ""
    inner = (call_args or {}).get("arguments")
    if isinstance(inner, str):
        try:
            inner = json.loads(inner)
        except (ValueError, TypeError):
            inner = {}
    return name, (inner if isinstance(inner, dict) else {})


def reconcile_block(call_args, state):
    """before_tool hook for update_transaction_rewards: bind writes to reconcile()."""
    action, args = _resolve(call_args)
    if "update_transaction_rewards" not in action:
        return None

    blocks = dict(state.get(RECON_BLOCKS) or {})

    def _block(msg):
        if blocks.get(action, 0) >= MAX_BLOCKS:
            return None  # fail open, never loop
        blocks[action] = blocks.get(action, 0) + 1
        state[RECON_BLOCKS] = blocks
        if _LOG:
            print(f"RECONGATE | BLOCK {msg[:90]}", file=sys.stderr, flush=True)
        return {"status": "blocked", "message": msg}

    if not state.get(RECON_DONE):
        return _block(
            "BLOCKED — before updating any transaction reward, call reconcile(items) "
            "ONCE with EVERY transaction you retrieved (transaction_id, amount, the "
            "correct cash-back rate from the KB, and the recorded reward). Then apply "
            "exactly the corrections it returns."
        )

    corrections = state.get(RECON_KEY) or {}
    tid = args.get("transaction_id")
    val = _num(args.get("new_rewards_earned"))
    if tid not in corrections:
        return _block(
            f"BLOCKED — {tid} is NOT a discrepancy per reconcile. Apply ONLY these "
            f"corrections (transaction_id: correct points): {corrections}."
        )
    if val is None or abs(val - corrections[tid]) > 0.01:
        return _block(
            f"BLOCKED — the reward for {tid} must be {corrections[tid]} points "
            f"per reconcile, not {val}."
        )
    if _LOG:
        print(f"RECONGATE | PASS {tid}={val}", file=sys.stderr, flush=True)
    return None


def authoritative_reinject_text(state):
    """Text appended every turn so the exact corrections stay in the model's view."""
    if not state.get(RECON_DONE):
        return None
    corrections = state.get(RECON_KEY) or {}
    if not corrections:
        return None
    lines = "\n".join(f"- {t}: {v} points" for t, v in corrections.items())
    return (
        "## Authoritative reward corrections (apply EXACTLY these)\n"
        "reconcile() determined these are the only incorrect transactions and their "
        "correct reward values. Call update_transaction_rewards for EVERY one of them "
        "and for NONE other:\n" + lines
    )


RECONCILE_RULE = """

## Reward / cash-back discrepancy tasks (HARD RULE)

When the customer reports reward or cash-back errors, do NOT decide which
transactions are wrong in your head, and do not trust a total you computed
informally. Retrieve EVERY transaction, find the correct cash-back rate for each
card/category in the knowledge base, then call the `reconcile` tool ONCE with ALL
transactions ({transaction_id, amount, rate, recorded}). It returns the
authoritative list of incorrect transactions and their correct values. Then apply
update_transaction_rewards for EXACTLY those — all of them, none else.
"""
