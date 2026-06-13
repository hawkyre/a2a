"""Harness-level guardrail: a pre-write verification gate for the CS agent.

Wired as ADK before_tool_callback. It fires automatically before every tool
call (the model cannot skip it). For STATE-CHANGING actions it:
  1. retrieves the governing KB procedure,
  2. checks completeness (were the prerequisite check-tools the procedure
     requires actually called this session?) and grounding (are the argument
     values consistent with policy + recorded facts, not the customer's story),
  3. and BLOCKS the write (returning an instruction instead of executing) when
     a required step is missing or an argument is clearly wrong.

It is deliberately conservative (block only on clear violations) and fails OPEN
after a couple of blocks on the same action, so a mistaken verifier can never
trap the agent in a loop and time the task out.

Toggle off with VERIFY_GATE=0.
"""

import json
import os

from rag_tools import _get_genai_client, kb_search_vector

MODEL = os.environ.get("MODEL", "gemini-3.5-flash")
ENABLED = os.environ.get("VERIFY_GATE", "1") not in ("0", "", "false", "False")
MAX_BLOCKS_PER_ACTION = 2  # fail open after this many blocks on the same action

# Tools that wrap a real action inside their arguments.
_WRAPPERS = ("call_discoverable_agent_tool", "call_env_tool")
# Our own tools — never gate or record these.
_OURS = ("kb_search_bm25", "kb_search_vector")
# Verb fragments that mark a state-changing (write) action.
_WRITE_HINTS = (
    "submit", "deny", "approve", "file_", "close", "order", "pay", "clear",
    "unfreeze", "freeze", "cancel", "update", "set_", "create", "replace",
    "remove", "issue", "refund", "transfer", "charge", "open_", "activate",
    "deactivate", "block", "give_", "apply", "log_verification", "dispute",
    "increase", "decrease", "enroll", "unenroll", "schedule",
)


def _resolve_action(tool_name: str, args: dict):
    """Return (real_action_name, real_args) unwrapping the discoverable wrappers."""
    if tool_name in _WRAPPERS:
        real = args.get("agent_tool_name") or args.get("tool_name") or tool_name
        inner = args.get("arguments") or args.get("arguments_json") or {}
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except Exception:
                inner = {"_raw": inner}
        return real, (inner if isinstance(inner, dict) else {"_raw": inner})
    return tool_name, args


def _is_write(action: str) -> bool:
    a = action.lower()
    if a.startswith("get_") or a.startswith("check_") or a.startswith("list_"):
        return False
    return any(h in a for h in _WRITE_HINTS)


def _readable(action: str) -> str:
    """deny_credit_limit_increase_5848 -> 'deny credit limit increase'."""
    import re
    return re.sub(r"_\d+$", "", action).replace("_", " ").strip()


def _recent_context(tool_context, max_chars: int = 3500) -> str:
    """Best-effort recent conversation + tool outputs (for grounding checks)."""
    try:
        events = tool_context._invocation_context.session.events
    except Exception:
        return ""
    chunks = []
    for e in list(events)[-30:]:
        content = getattr(e, "content", None)
        for p in (getattr(content, "parts", None) or []):
            if getattr(p, "text", None):
                chunks.append(p.text)
            fr = getattr(p, "function_response", None)
            if fr is not None:
                chunks.append(json.dumps(getattr(fr, "response", str(fr)), default=str)[:700])
    return "\n".join(chunks)[-max_chars:]


_VERIFIER_PROMPT = """You are a STRICT bank-policy compliance gate. An agent is about to perform a \
STATE-CHANGING action. Decide if it may proceed RIGHT NOW.

ACTION: {action}
ARGUMENTS: {args}

CHECK-TOOLS ALREADY CALLED THIS SESSION (in order):
{called}

GOVERNING POLICY (knowledge-base excerpts):
{policy}

RECENT CONVERSATION & TOOL OUTPUTS (the recorded facts):
{context}

Judge two things, using ONLY the policy and recorded facts (never the customer's narrative/feelings):
1. COMPLETENESS — does the policy require any check/tool to be performed BEFORE this action that is NOT in the called list above? Required eligibility/review checks must all be done first.
2. GROUNDING — is every argument value (especially coded/enum fields and amounts) consistent with the policy definitions applied to the recorded facts?

Be conservative: only block on a CLEAR violation. If unsure, allow.
Respond with ONLY JSON:
{{"ok": true}}  OR  {{"ok": false, "reason": "<one sentence>", "fix": "<the exact next tool to call or argument to correct>"}}"""


def _verify(action: str, args: dict, called: list, tool_context) -> dict:
    readable = _readable(action)
    try:
        docs = kb_search_vector(f"{readable} procedure: required checks before, eligibility, and argument rules", top_k=3)
        policy = "\n\n".join(
            f"[{d.get('title','')}]\n{d.get('content','')[:1200]}" for d in docs if d.get("content")
        ) or "(no policy retrieved)"
    except Exception as e:
        policy = f"(policy retrieval failed: {e})"
    prompt = _VERIFIER_PROMPT.format(
        action=action,
        args=json.dumps(args, default=str)[:1200],
        called="\n".join(f"- {c}" for c in called) or "(none yet)",
        policy=policy[:5000],
        context=_recent_context(tool_context),
    )
    try:
        from google.genai import types
        resp = _get_genai_client().models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_level="low"),
            ),
        )
        return json.loads(resp.text)
    except Exception as e:
        # Verifier failure must never block real work.
        return {"ok": True, "_verifier_error": str(e)}


async def before_tool(tool, args, tool_context):
    """ADK before_tool_callback. Return None to proceed, or a dict to block."""
    if not ENABLED or tool.name in _OURS:
        return None

    action, action_args = _resolve_action(tool.name, args or {})
    state = tool_context.state
    called = list(state.get("vg_called", []))

    if not _is_write(action):
        # Record the read/check (for completeness tracking) and let it through.
        if action not in called:
            state["vg_called"] = called + [action]
        return None

    # Write action: fail open if we've already blocked it repeatedly (anti-loop).
    blocks = dict(state.get("vg_blocks", {}))
    if blocks.get(action, 0) >= MAX_BLOCKS_PER_ACTION:
        state["vg_called"] = called + [action]
        return None

    verdict = _verify(action, action_args, called, tool_context)
    if verdict.get("ok", True):
        state["vg_called"] = called + [action]
        return None

    blocks[action] = blocks.get(action, 0) + 1
    state["vg_blocks"] = blocks
    return {
        "error": True,
        "blocked_by_policy_gate": True,
        "content": (
            f"POLICY GATE — do not proceed with {action} yet. "
            f"Reason: {verdict.get('reason','policy requirement not met')}. "
            f"Next step: {verdict.get('fix','complete the required checks, then retry')}."
        ),
    }
