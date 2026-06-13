"""Tool-fidelity + KB-grounding helpers for the CS agent.

Three cheap, non-blocking mechanisms (pure Python — no network, no model calls
inside the callbacks, so they cannot stall the agent loop):

1. KB-search GATE (before_tool_callback). The agent may not unlock or call any
   discoverable tool until it has searched the knowledge base at least once this
   conversation. If it tries, the tool is short-circuited with an instruction to
   search first. kb_search is invisible to the scored env ledger, so forcing it
   never changes the db_match footprint — it just guarantees the model reads the
   real procedure (and the discoverable tool names it requires) before acting.

2. Argument canonicalization (before_tool_callback). The harness compares the
   `arguments` string of call_discoverable_agent_tool verbatim, so cosmetic
   formatting (125.00 vs 125.0, whitespace, ": " spacing) causes spurious
   mismatches. Round-tripping through json.loads/json.dumps normalizes that
   WITHOUT changing value types (125.00 stays float 125.0, 125 stays int 125),
   so it can never introduce a NEW int/float mismatch.

3. Capture + reinject (after_tool_callback + before_model_callback). KB results
   and unlocked-tool schemas are revealed once in a tool result, then scroll out
   of context. We stash both in session state and re-append them to the
   instruction every turn, so the model ALWAYS sees the retrieved procedure and
   the exact parameter names/types of the tools it has unlocked — it cannot skip
   or forget the KB grounding.

State lives in per-contextId session state, so nothing leaks across conversations.
"""

import json

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.adk.tools import BaseTool, ToolContext

from calc_gate import calc_backed_block, record_calc_result
from reconcile_tool import authoritative_reinject_text, reconcile_block

SCHEMA_KEY = "unlocked_tool_schemas"
KB_KEY = "kb_results"
UNLOCK_TOOL = "unlock_discoverable_agent_tool"
CALL_TOOL = "call_discoverable_agent_tool"
DISCOVERABLE_TOOLS = (UNLOCK_TOOL, CALL_TOOL)
KB_TOOLS = ("kb_search_bm25", "kb_search_vector")
KB_RESULTS_CAP = 6  # keep the most recent few search-result blobs in context

SOURCE_FIDELITY_RULE = """

## Argument fidelity (HARD RULE)

When you pass a value to a tool, reproduce it EXACTLY as it appeared at its
source — the output of the tool you got it from, or the customer's own words.
Do NOT reformat it: never add or drop decimal places, never round, never change
an integer to a decimal (or vice-versa), never add thousands separators, never
re-case or re-spell identifiers. Copy account IDs, card IDs, amounts, codes, and
names verbatim. If a value came from a prior tool's result (e.g. an outstanding
balance, an account_id), take it from that result rather than retyping it from
the chat. Follow the parameter names and types in each tool's definition, not
how a value happened to be written in the conversation.
"""

_GATE_MESSAGE = (
    "BLOCKED — you have not searched the knowledge base yet this conversation. "
    "Before unlocking or calling ANY discoverable tool you MUST first call "
    "kb_search_vector (or kb_search_bm25) with the customer's current request to "
    "load the exact procedure and the discoverable tool names it requires. "
    "Search now, then follow the retrieved procedure step by step and in order "
    "(do not skip steps or improvise), and retry this tool afterwards."
)


def _response_text(tool_response) -> str:
    """Best-effort string of a tool response (dict with 'content', or raw)."""
    if isinstance(tool_response, dict):
        content = tool_response.get("content")
        if isinstance(content, str):
            return content
        return str(tool_response)
    return str(tool_response)


def before_tool(tool: BaseTool, args: dict, tool_context: ToolContext):
    """before_tool_callback: KB-search gate + argument canonicalization.

    Returns a dict to short-circuit the tool (gate); returns None to let the
    (possibly mutated) call proceed.
    """
    # --- Gate: no discoverable-tool use until the KB has been searched ---
    if tool.name in DISCOVERABLE_TOOLS and not tool_context.state.get(KB_KEY):
        return {"status": "blocked", "message": _GATE_MESSAGE}

    # --- Canonicalize the arguments JSON string of call_discoverable_agent_tool ---
    if tool.name == CALL_TOOL:
        raw = (args or {}).get("arguments")
        if isinstance(raw, str) and raw.strip():
            try:
                args["arguments"] = json.dumps(json.loads(raw))
            except (ValueError, TypeError):
                pass  # not JSON we understand; leave it untouched
        # --- Reconcile gate: reward updates must match the authoritative list ---
        blocked = reconcile_block(args, tool_context.state)
        if blocked:
            return blocked
        # --- Calc-backed-write gate: computed numbers must come from calculator ---
        blocked = calc_backed_block(args, tool_context.state)
        if blocked:
            return blocked
    return None


def after_tool(tool: BaseTool, args: dict, tool_context: ToolContext, tool_response):
    """after_tool_callback: capture KB results and unlocked-tool schemas so we
    can keep re-surfacing them every turn."""
    if tool.name == UNLOCK_TOOL:
        name = (args or {}).get("agent_tool_name") or (args or {}).get("tool_name")
        if name:
            schemas = dict(tool_context.state.get(SCHEMA_KEY) or {})
            schemas[name] = _response_text(tool_response)
            tool_context.state[SCHEMA_KEY] = schemas
    elif tool.name in KB_TOOLS:
        results = list(tool_context.state.get(KB_KEY) or [])
        text = _response_text(tool_response)
        if text and text not in results:
            results.append(text)
            tool_context.state[KB_KEY] = results[-KB_RESULTS_CAP:]
    elif tool.name == "calculator":
        record_calc_result(tool_response, tool_context.state)
    return None  # never modify the tool result


def reinject_context(callback_context: CallbackContext, llm_request: LlmRequest):
    """before_model_callback: re-append the KB procedures retrieved and the
    schemas of all tools unlocked so far, so the model always has them in view."""
    blocks = []

    kb = callback_context.state.get(KB_KEY) or []
    if kb:
        blocks.append(
            "## Knowledge base procedures retrieved this conversation\n"
            "These are the authoritative procedures for the current request. "
            "Follow them EXACTLY and completely, in the order given — perform "
            "every applicable step and do not improvise or stop early:\n\n"
            + "\n\n---\n\n".join(kb)
        )

    schemas = callback_context.state.get(SCHEMA_KEY) or {}
    if schemas:
        blocks.append(
            "## Currently unlocked discoverable tools\n"
            "These tools are already unlocked this conversation. When you call "
            "call_discoverable_agent_tool, the `arguments` JSON must use these "
            "EXACT parameter names and value types:\n\n"
            + "\n\n".join(text for text in schemas.values() if text)
        )

    recon = authoritative_reinject_text(callback_context.state)
    if recon:
        blocks.append(recon)

    if blocks:
        llm_request.append_instructions(blocks)
    return None
