"""Tool that lets the personal agent talk to the bank's customer service
agent over A2A, propagating the current session's contextId so both agents
(and the env) share one conversation identity."""

import os
import re
import uuid

import httpx
from a2a.client import ClientConfig, ClientFactory, minimal_agent_card
from a2a.types import Message, Part, Role, Task, TextPart
from google.adk.tools import ToolContext

from env_toolset import session_id

CS_AGENT_URL = os.environ["CS_AGENT_URL"]

_TIMEOUT_S = 300.0

# Interoperable side-channel: a cooperating CS agent appends a context footer
# delimited by [[CS_CONTEXT]] ... [[/CS_CONTEXT]] so we can stay in sync with the
# procedure it is running. We strip it from the customer-facing reply and surface
# it in our own prompt every turn (see personal agent reinject_cs_context). A CS
# agent that does NOT emit it (a random/3rd-party agent) is handled gracefully —
# nothing to strip, nothing to surface.
CS_CONTEXT_KEY = "cs_shared_context"
_CTX_RE = re.compile(r"\[\[CS_CONTEXT\]\](.*?)\[\[/CS_CONTEXT\]\]", re.DOTALL)


def _split_context(reply: str) -> tuple[str, str | None]:
    """Return (reply with footer removed, extracted context or None)."""
    matches = _CTX_RE.findall(reply)
    if not matches:
        return reply, None
    cleaned = _CTX_RE.sub("", reply).strip()
    return cleaned, matches[-1].strip()  # last footer is the freshest


def _text_of_message(message: Message) -> str:
    texts = []
    for part in message.parts or []:
        root = getattr(part, "root", part)
        if isinstance(root, TextPart) and root.text:
            texts.append(root.text)
    return "\n".join(texts)


def _text_of_task(task: Task) -> str:
    texts = []
    for artifact in task.artifacts or []:
        for part in artifact.parts or []:
            root = getattr(part, "root", part)
            if isinstance(root, TextPart) and root.text:
                texts.append(root.text)
    if task.status is not None and task.status.message is not None:
        text = _text_of_message(task.status.message)
        if text:
            texts.append(text)
    return "\n".join(texts)


async def ask_customer_service(message: str, tool_context: ToolContext) -> str:
    """Send a message to the bank's customer service agent and return its reply.

    The conversation with customer service persists for this whole session,
    so you can ask follow-up questions and they will remember the context.
    """
    outgoing = Message(
        message_id=uuid.uuid4().hex,
        role=Role.user,
        parts=[Part(root=TextPart(text=message))],
        context_id=session_id(tool_context),
    )
    async with httpx.AsyncClient(timeout=_TIMEOUT_S) as http_client:
        client = ClientFactory(
            ClientConfig(streaming=False, httpx_client=http_client)
        ).create(minimal_agent_card(CS_AGENT_URL, ["JSONRPC"]))
        reply = ""
        async for event in client.send_message(outgoing):
            if isinstance(event, Message):
                reply = _text_of_message(event) or reply
            elif isinstance(event, tuple) and isinstance(event[0], Task):
                reply = _text_of_task(event[0]) or reply
    cleaned, context = _split_context(reply)
    if context:
        # Persist for this session so reinject_cs_context can surface it every turn.
        tool_context.state[CS_CONTEXT_KEY] = context
    return cleaned or "[no response from customer service]"
