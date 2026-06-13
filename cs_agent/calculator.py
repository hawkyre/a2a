"""Internal deterministic calculator tool for the CS agent.

LLMs do arithmetic unreliably (off-by-ones, dropped/invented items when scanning
a list). This tool evaluates an arithmetic/reconciliation EXPRESSION exactly, so
the model never computes numbers in its head.

Safety: the expression is parsed to an AST and checked against a strict
allow-list of node types and function names. It is a SINGLE expression — no
statements, assignments, loops, imports, attribute access (`.x`), or I/O — so
there is no code-execution surface. List/dict/set literals and comprehensions
ARE allowed, so a full reconciliation can be done in one call, e.g.:

    [[t['id'], round(t['amt']*0.03, 2)] for t in
        [{'id':'a','amt':100,'rec':3.00}, {'id':'b','amt':500,'rec':10.00}]
     if round(t['amt']*0.03, 2) != t['rec']]
"""

import ast
import os
import sys

_LOG = bool(os.environ.get("KB_QUERY_LOG"))

# Whitelisted callables (no eval/exec/open/__import__, no attribute access).
_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "len": len, "sorted": sorted, "int": int, "float": float, "bool": bool,
    "list": list, "dict": dict, "set": set, "tuple": tuple, "divmod": divmod,
}

# Allow-listed AST node types. Anything not here (Attribute, Import, Lambda,
# assignments, etc.) is rejected — default deny.
_ALLOWED_NODES = (
    ast.Expression, ast.Constant,
    ast.List, ast.Tuple, ast.Set, ast.Dict,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension,
    ast.Name, ast.Load, ast.Store, ast.Call, ast.keyword,
    ast.Subscript, ast.Slice,
    # arithmetic / logic / comparison operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
)

MAX_LEN = 8000


def _validate(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
                raise ValueError("only whitelisted functions may be called")


def calculator(expression: str) -> dict:
    """Evaluate an arithmetic expression exactly. USE THIS FOR ALL ARITHMETIC.

    Never compute numbers in your head. Pass one Python expression using the
    exact numbers from tool outputs. Supports + - * / // % **, parentheses,
    comparisons, and the functions abs, round, min, max, sum, len, sorted, int,
    float, list, dict, set, tuple, divmod. List/dict comprehensions are allowed,
    so you can reconcile a whole list of transactions in a single call (use
    item['key'] indexing, not item.key).

    Args:
        expression: A single Python expression, e.g.
            "round(2999.99 * 0.05, 2)"  or a comprehension over a transaction list.

    Returns:
        {"result": <value>} on success, or {"error": "<reason>"}.
    """
    if not expression or not expression.strip():
        return {"error": "empty expression"}
    if len(expression) > MAX_LEN:
        return {"error": "expression too long"}
    try:
        tree = ast.parse(expression, mode="eval")  # single expression only
        _validate(tree)
        result = eval(  # noqa: S307 - AST is allow-list validated above
            compile(tree, "<calculator>", "eval"),
            {"__builtins__": {}, **_FUNCS},
            {},
        )
        out = {"result": result}
    except Exception as e:  # never raise into the agent loop
        out = {"error": f"{type(e).__name__}: {e}"}
    if _LOG:
        print(f"CALC | {expression[:160]!r} -> {str(out)[:200]}", file=sys.stderr, flush=True)
    return out


COMPUTE_RULE = """

## Calculations (HARD RULE)

You are bad at mental arithmetic. Never compute a number yourself. For ANY
calculation — cash-back/reward points, fees, interest, utilization ratios,
totals, percentage caps, or comparing values across many transactions — call the
`calculator` tool with a single expression, using the EXACT numbers from tool
outputs. The calculator supports list/dict comprehensions, so reconcile a whole
list in one call (e.g. compute the expected reward for every transaction and
return only those whose recorded value differs). Trust the calculator's result,
never your own estimate, and copy its result verbatim into the tool argument.
"""
