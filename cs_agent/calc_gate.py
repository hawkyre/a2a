"""Calc-backed-write gate: numbers committed to state-changing tools must come
from the deterministic `calculator` tool, not the model's head.

The model ignores an optional calculator (observed: 0 calls across several runs)
and makes arithmetic slips (1499 vs 1500, $12 vs $14). This gate makes the
calculator non-optional for the specific tool arguments that must be COMPUTED:
it captures every calculator result, and blocks a numeric write whose committed
value did not come from a calculator result, returning an instruction to compute
it first.

Pure Python, no model calls. Fails open after a couple of blocks per action so a
genuinely looked-up value (not a computed one) can never trap the agent in a
loop. Wired into tool_recency's before_tool / after_tool callbacks.
"""

import json
import os
import re
import sys

_LOG = bool(os.environ.get("KB_QUERY_LOG"))

CALC_NUMBERS_KEY = "calc_numbers"
CALC_BLOCKS_KEY = "calc_blocks"
MAX_BLOCKS = 2
_POOL_CAP = 500

# Tool-name fragment -> the argument(s) whose value MUST be calculator-derived.
# Only genuinely-computed values are gated (not IDs, fees looked up from policy,
# or amounts the customer stated), to avoid false blocks on look-up values.
_NUMERIC_WRITE_ARGS = {
    # update_transaction_rewards is owned by the stricter reconcile gate.
    "apply_checking_account_credit": ("amount",),
    "approve_credit_limit_increase": ("new_credit_limit",),
    "submit_credit_limit_increase_request": ("requested_increase_amount",),
    "pay_credit_card_from_checking": ("amount",),
}

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _numbers(obj, parse_strings: bool):
    out = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out += _numbers(v, parse_strings)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += _numbers(v, parse_strings)
    elif isinstance(obj, str) and parse_strings:
        for m in _NUM_RE.findall(obj):
            try:
                out.append(float(m))
            except ValueError:
                pass
    return out


def record_calc_result(tool_response, state) -> None:
    """after_tool hook for tool.name == 'calculator': stash the result's numbers.

    Only numeric leaves are taken (string ids in a reconciliation result are
    ignored), so the pool stays clean.
    """
    if not isinstance(tool_response, dict) or "result" not in tool_response:
        return
    pool = list(state.get(CALC_NUMBERS_KEY) or [])
    for n in _numbers(tool_response["result"], parse_strings=False):
        if n not in pool:
            pool.append(n)
    state[CALC_NUMBERS_KEY] = pool[-_POOL_CAP:]


def _resolve(call_args):
    name = (call_args or {}).get("agent_tool_name") or (call_args or {}).get("tool_name") or ""
    inner = (call_args or {}).get("arguments")
    if isinstance(inner, str):
        try:
            inner = json.loads(inner)
        except ValueError:
            inner = {}
    return name, (inner if isinstance(inner, dict) else {})


def calc_backed_block(call_args, state):
    """before_tool hook for call_discoverable_agent_tool. Returns a block dict if
    a must-compute numeric arg isn't calculator-backed, else None."""
    action, args = _resolve(call_args)
    arg_names = next((names for frag, names in _NUMERIC_WRITE_ARGS.items() if frag in action), None)
    if not arg_names:
        return None  # not a gated numeric write

    pool = state.get(CALC_NUMBERS_KEY) or []
    unbacked = []
    for an in arg_names:
        if an not in args:
            continue
        for num in _numbers(args[an], parse_strings=True):
            if not any(abs(num - p) <= 0.01 for p in pool):
                unbacked.append((an, num))
    if not unbacked:
        if _LOG:
            print(f"CALCGATE | {action} PASS (calc-backed) pool={len(pool)}", file=sys.stderr, flush=True)
        return None  # every committed number came from the calculator

    blocks = dict(state.get(CALC_BLOCKS_KEY) or {})
    if _LOG:
        print(f"CALCGATE | {action} unbacked={unbacked} pool={len(pool)} blocks={blocks.get(action,0)}",
              file=sys.stderr, flush=True)
    if blocks.get(action, 0) >= MAX_BLOCKS:
        return None  # fail open: never loop a genuine look-up value forever
    blocks[action] = blocks.get(action, 0) + 1
    state[CALC_BLOCKS_KEY] = blocks

    fields = ", ".join(f"{an}={num:g}" for an, num in unbacked)
    return {
        "status": "blocked",
        "message": (
            f"BLOCKED — the value(s) {fields} for {action} were not produced by the "
            "calculator. Never compute figures in your head. Call the `calculator` tool "
            "with the EXACT numbers from the tool outputs (reconcile a whole list in one "
            "expression if needed), then retry this call using the calculator's result "
            "verbatim. If you already searched the KB for the rate/rule, apply it in the "
            "calculator now."
        ),
    }
