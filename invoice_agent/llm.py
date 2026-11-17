from __future__ import annotations

import contextvars
import json
import os
import re
from typing import Any, Protocol

from invoice_agent.parse import invoice_from_mapping
from invoice_agent.schemas import Invoice
from invoice_agent.tools import INVENTORY_TOOLS, lookup_inventory

_llm_client: contextvars.ContextVar[Any] = contextvars.ContextVar("invoice_llm_client", default=None)

EXTRACT_SYSTEM = """You extract one invoice into JSON.
Keep the numbers that are printed. When a printed total disagrees with the line items, keep both.
Use null for anything the document does not state.
Dates are YYYY-MM-DD. If the due date says "yesterday", set due_date_note to "yesterday" and due_date null.
Quantities and money are JSON numbers, with no currency symbols.
Reply with one JSON object and no markdown.
Schema keys: invoice_number, vendor, invoice_date, due_date, due_date_note, currency,
line_items (item, quantity, unit_price, amount), subtotal, tax_amount, shipping, total,
payment_terms, notes, revision.
"""

CRITIQUE_SYSTEM = """You review an accounts-payable decision before money moves.
You may recommend approve, needs_review, or reject.
A recommendation to approve is ignored when the rules already held or rejected the invoice.
Reply with JSON: {"recommended_decision": "approve|needs_review|reject", "rationale": "one sentence"}.
"""

VALIDATE_SYSTEM = """You are the inventory agent for an accounts-payable desk.
Call lookup_inventory once for every distinct item before you summarize.
You do not approve payment. Summarize stock risk in one or two sentences.
"""


class SupportsLLM(Protocol):
    def complete_json(self, system: str, user: str) -> dict: ...

    def complete(self, messages: list[dict], tools: list | None = None, json_mode: bool = False) -> Any: ...


def set_llm_client(client: Any):
    return _llm_client.set(client)


def reset_llm_client(token) -> None:
    _llm_client.reset(token)


def get_llm_client() -> Any:
    return _llm_client.get()


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def build_client() -> SupportsLLM | None:
    load_env()
    xai_key = os.getenv("XAI_API_KEY", "").strip()
    if xai_key:
        return ChatClient(
            api_key=xai_key,
            model=os.getenv("XAI_MODEL", "grok-3").strip() or "grok-3",
            base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1").strip(),
        )
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        return ChatClient(
            api_key=gemini_key,
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash",
            base_url=os.getenv(
                "GEMINI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai/",
            ).strip(),
        )
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        return ChatClient(
            api_key=openai_key,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
        )
    return None


def client_available() -> bool:
    return build_client() is not None or get_llm_client() is not None


class ChatClient:
    """OpenAI-compatible chat client used by supported model providers."""

    def __init__(self, api_key: str, model: str, base_url: str):
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, messages: list[dict], tools: list | None = None, json_mode: bool = False):
        kwargs: dict[str, Any] = {"model": self.model, "temperature": 0, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception:
            if not json_mode:
                raise
            kwargs.pop("response_format", None)
            response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message

    def complete_json(self, system: str, user: str) -> dict:
        message = self.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            json_mode=True,
        )
        return json.loads(strip_fences(content_of(message)))


def strip_fences(text: str) -> str:
    cleaned = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return cleaned


def content_of(message) -> str:
    content = getattr(message, "content", "") or ""
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def tool_calls_of(message) -> list[dict]:
    raw = getattr(message, "tool_calls", None) or []
    calls = []
    for call in raw:
        if isinstance(call, dict):
            function = call.get("function") or {}
            calls.append(
                {
                    "id": call.get("id"),
                    "name": function.get("name") or call.get("name"),
                    "arguments": function.get("arguments") if function else call.get("arguments", "{}"),
                }
            )
        else:
            function = call.function
            calls.append({"id": call.id, "name": function.name, "arguments": function.arguments})
    return calls


def assistant_dict(message) -> dict:
    payload: dict[str, Any] = {"role": "assistant", "content": content_of(message)}
    calls = tool_calls_of(message)
    if calls:
        payload["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": call["arguments"] if isinstance(call["arguments"], str) else json.dumps(call["arguments"]),
                },
            }
            for call in calls
        ]
    return payload


def llm_extract(text: str, issues: list[str]) -> Invoice:
    client = get_llm_client()
    if client is None:
        raise RuntimeError("no model client configured")
    feedback = ""
    if issues:
        feedback = (
            "\n\nA previous read failed these checks. Re-read the document and keep the printed figures. "
            "Do not invent a total that is not printed.\n- " + "\n- ".join(issues)
        )
    user = f"Document:\n{text[:12000]}{feedback}"
    last_error: Exception | None = None
    for _ in range(2):
        try:
            data = client.complete_json(EXTRACT_SYSTEM, user)
            invoice = invoice_from_mapping(data)
            if not invoice.line_items and invoice.total is None:
                raise ValueError("extraction contained no line items")
            return invoice
        except Exception as exc:
            last_error = exc
            user += f"\n\nThat reply did not match the schema ({exc}). Reply with JSON only."
    raise RuntimeError(str(last_error))


def llm_critique(invoice: Invoice, decision: str, findings: list[dict]) -> dict:
    client = get_llm_client()
    if client is None:
        raise RuntimeError("no model client configured")
    user = (
        f"Rules decision: {decision}\n"
        f"Findings: {json.dumps(findings)}\n"
        f"Invoice: {invoice.model_dump_json()}"
    )
    return client.complete_json(CRITIQUE_SYSTEM, user)


def run_validation_agent(invoice: Invoice, conn) -> str:
    client = get_llm_client()
    if client is None:
        raise RuntimeError("no model client configured")
    messages: list[dict] = [
        {"role": "system", "content": VALIDATE_SYSTEM},
        {"role": "user", "content": invoice.model_dump_json()},
    ]
    for _ in range(4):
        message = client.complete(messages, tools=INVENTORY_TOOLS)
        calls = tool_calls_of(message)
        messages.append(assistant_dict(message))
        if not calls:
            return content_of(message).strip()
        for call in calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(_dispatch_tool(call, conn)),
                }
            )
    return "Validation agent stopped after 4 tool rounds."


def _dispatch_tool(call: dict, conn) -> dict:
    if call.get("name") != "lookup_inventory":
        return {"error": f"unknown tool {call.get('name')}"}
    arguments = call.get("arguments") or {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments or "{}")
    return lookup_inventory(conn, str(arguments.get("item") or ""))
