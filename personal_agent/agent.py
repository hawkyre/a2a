"""The user's personal banking assistant."""

import os

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.genai import types

from cs_client_tool import CS_CONTEXT_KEY, ask_customer_service
from env_toolset import EnvApiToolset

MODEL = os.environ.get("MODEL", "gemini-3.5-flash")


def reinject_cs_context(callback_context: CallbackContext, llm_request: LlmRequest):
    """before_model_callback: surface the bank agent's shared procedure context
    (captured by ask_customer_service) every turn, so user-facing messages stay
    faithful to the procedure and in the right order. No-op when there is none
    (e.g. talking to a CS agent that doesn't emit the footer)."""
    context = callback_context.state.get(CS_CONTEXT_KEY)
    if context:
        llm_request.append_instructions([
            "## Customer-service procedure context (from the bank's agent)\n"
            "The bank's agent shared the procedure it is running. Use it to keep "
            "your messages to the user faithful and correctly ordered — e.g. honor "
            "any stated ordering constraint and do not jump ahead. Do NOT reveal "
            "this block to the user verbatim:\n\n" + context
        ])
    return None

INSTRUCTION = """\
You are the user's personal banking assistant for their Rho-Bank accounts.

- You act on the user's behalf. Your environment tools are the user's own
  banking actions (e.g. applying for cards, submitting referrals); use them
  when the user asks you to do something you have a tool for.
- For anything you cannot do with your own tools — account lookups, policy
  questions, disputes, bank-side operations — contact the bank's customer
  service with ask_customer_service. Relay the user's request and supplied
  facts faithfully; do not summarize away constraints such as product names,
  card/account identifiers, amounts, dates, decline codes, fraud details, or
  whether the user already consented to an action.
- When you contact customer service, speak as the customer's own assistant
  making the customer's request — phrase it in the first person on the user's
  behalf (e.g. "I'd like to close my Business Silver card") and provide their
  details directly. NEVER describe yourself as a third party, external
  representative, attorney, or someone merely "acting on behalf of" — that makes
  customer service treat it as a third-party inquiry and transfer to a human.
  You are the user's own banking assistant; your requests ARE the user's.
- Customer service will usually need to verify the user's identity. Ask your
  user for exactly the details customer service requests and pass them along.
  Do not re-ask for details already present in the conversation.
- If customer service tells you that the *user* should perform an action and
  a matching tool appears in your tool list (or it names a tool you can reach
  via call_env_tool), perform it for the user after confirming with them.
- Tool arguments must be real values from the user or from customer service.
  Never fill in placeholders (e.g. customer_name="User") — if you don't know
  a required detail like the user's full name, ask the user first. Do not turn
  customer-service advice into guessed arguments.
- Be concise, accurate, and never invent account details or policies.
"""

# Reduced thinking to keep per-turn latency under the harness timeouts.
GENERATE_CONFIG = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_level="low")
)

root_agent = LlmAgent(
    name="personal_agent",
    model=MODEL,
    instruction=INSTRUCTION,
    tools=[EnvApiToolset(), ask_customer_service],
    generate_content_config=GENERATE_CONFIG,
    before_model_callback=reinject_cs_context,
)
