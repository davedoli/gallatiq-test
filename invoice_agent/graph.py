from __future__ import annotations

import logging
import warnings
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, List, Optional, TypedDict

import langchain_core  # registers filters that would otherwise surface on import
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
from langgraph.graph import END, StateGraph

from invoice_agent.db import connect
from invoice_agent.llm import get_llm_client, llm_critique, llm_extract, run_validation_agent
from invoice_agent.parse import fix_ocr, parse_document, read_source
from invoice_agent.policy import apply_critique, decide, explain
from invoice_agent.schemas import Finding, Invoice
from invoice_agent.tools import submit_payment
from invoice_agent.validate import audit_math, collect_findings
from invoice_agent.normalize import canonical_item

logger = logging.getLogger("invoice_agent")


class AgentState(TypedDict, total=False):
    """One channel per field. LangGraph keeps the fields a node does not return."""

    invoice_path: str
    raw_text: str
    source_format: str
    invoice: Optional[dict]
    extraction_attempts: int
    extraction_issues: List[str]
    extraction_source: str
    retry_extraction: bool
    findings: List[dict]
    decision: str
    decision_reason: str
    critique_notes: str
    validation_narrative: str
    payment: Optional[dict]
    trace: List[str]
    use_llm: bool
    db_path: str
    log_path: str


def _trace(state: AgentState, message: str) -> list[str]:
    events = list(state.get("trace") or [])
    events.append(message)
    return events


def load(state: AgentState) -> dict:
    text, fmt = read_source(Path(state["invoice_path"]))
    return {
        "raw_text": text,
        "source_format": fmt,
        "trace": _trace(state, f"loaded {fmt} document ({len(text)} characters)"),
    }


def extract(state: AgentState) -> dict:
    attempts = int(state.get("extraction_attempts") or 0) + 1
    text = state.get("raw_text") or ""
    fmt = state.get("source_format") or "txt"
    issues = list(state.get("extraction_issues") or [])
    trace = list(state.get("trace") or [])
    invoice = None
    source = "parser"
    if state.get("use_llm") and fmt in {"txt", "pdf", "text"} and get_llm_client() is not None:
        try:
            invoice = llm_extract(text, issues if attempts > 1 else [])
            source = "llm"
        except Exception as exc:
            logger.warning("model extraction failed: %s", exc)
            trace.append(f"model extraction failed ({exc.__class__.__name__}); used the local parser")
    if invoice is None:
        try:
            invoice = parse_document(text, fmt)
        except Exception as exc:
            logger.warning("parser failed: %s", exc)
            invoice = Invoice()
            trace.append(f"parser failed ({exc.__class__.__name__})")
        source = "parser" if source != "llm" else source
        if source == "parser" and fix_ocr(text) != text:
            trace.append("normalized OCR lookalikes in dates and amounts")
    trace.append(f"extraction attempt {attempts} via {source}")
    return {
        "invoice": invoice.to_state(),
        "extraction_attempts": attempts,
        "extraction_source": source,
        "trace": trace,
    }


def check(state: AgentState) -> dict:
    invoice = Invoice.model_validate(state["invoice"])
    issues = [
        finding.message
        for finding in audit_math(invoice)
        if finding.code in {"math_mismatch", "missing_total"}
    ]
    retry = bool(
        issues
        and state.get("use_llm")
        and state.get("extraction_source") == "llm"
        and int(state.get("extraction_attempts") or 0) < 2
    )
    if retry:
        message = "math check failed; sending the invoice back to extraction"
    elif issues:
        message = "math check found discrepancies in the printed invoice"
    else:
        message = "math check passed"
    return {
        "extraction_issues": issues,
        "retry_extraction": retry,
        "trace": _trace(state, message),
    }


def route_after_check(state: AgentState) -> str:
    if state.get("retry_extraction"):
        return "extract"
    return "normalize"


def normalize(state: AgentState) -> dict:
    invoice = Invoice.model_validate(state["invoice"])
    for line in invoice.line_items:
        line.normalized_item = canonical_item(line.item)
    return {
        "invoice": invoice.to_state(),
        "trace": _trace(state, f"normalized {len(invoice.line_items)} line item names"),
    }


def validate(state: AgentState) -> dict:
    invoice = Invoice.model_validate(state["invoice"])
    trace = list(state.get("trace") or [])
    narrative = ""
    with connect(state["db_path"]) as conn:
        if state.get("use_llm") and get_llm_client() is not None:
            try:
                narrative = run_validation_agent(invoice, conn)
                trace.append("validation agent finished a tool pass")
            except Exception as exc:
                logger.warning("validation agent skipped: %s", exc)
                trace.append(f"validation agent skipped ({exc.__class__.__name__})")
        findings = collect_findings(invoice, state.get("raw_text") or "", conn)
    trace.append(f"deterministic validation produced {len(findings)} finding(s)")
    return {
        "findings": [finding.model_dump() for finding in findings],
        "validation_narrative": narrative,
        "trace": trace,
    }


def draft(state: AgentState) -> dict:
    invoice = Invoice.model_validate(state["invoice"])
    findings = [Finding.model_validate(item) for item in state.get("findings") or []]
    decision = decide(findings)
    logger.info("draft decision=%s invoice=%s", decision, invoice.invoice_number)
    return {
        "decision": decision,
        "decision_reason": explain(decision, findings, invoice),
        "trace": _trace(state, f"draft decision: {decision}"),
    }


def critique(state: AgentState) -> dict:
    invoice = Invoice.model_validate(state["invoice"])
    with connect(state["db_path"]) as conn:
        recomputed = collect_findings(invoice, state.get("raw_text") or "", conn)
    draft_codes = [item["code"] for item in state.get("findings") or []]
    new_codes = [finding.code for finding in recomputed]
    decision = decide(recomputed)
    notes = ["Second pass recomputed the findings."]
    if draft_codes != new_codes:
        notes.append("The finding list changed, so the second pass replaced it.")
    else:
        notes.append("The finding list matched the first pass.")

    model_decision = None
    if state.get("use_llm") and get_llm_client() is not None:
        try:
            result = llm_critique(invoice, decision, [finding.model_dump() for finding in recomputed])
            model_decision = str(result.get("recommended_decision") or "").strip().lower() or None
            rationale = str(result.get("rationale") or "").strip()
            if rationale:
                notes.append(rationale)
            updated = apply_critique(decision, model_decision)
            if model_decision and model_decision != updated:
                notes.append(f"The model recommended {model_decision}. The payment gate kept {updated}.")
            elif model_decision:
                notes.append("The model agreed with the decision.")
            decision = updated
        except Exception as exc:
            logger.warning("model critique skipped: %s", exc)
            notes.append(f"The model critique did not run ({exc.__class__.__name__}).")
    else:
        notes.append("No model was configured, so the rules stood on their own.")

    narrative = (state.get("validation_narrative") or "").strip()
    if narrative:
        notes.append("Inventory agent: " + narrative[:500])

    logger.info("final decision=%s invoice=%s", decision, invoice.invoice_number)
    return {
        "findings": [finding.model_dump() for finding in recomputed],
        "decision": decision,
        "decision_reason": explain(decision, recomputed, invoice),
        "critique_notes": " ".join(notes),
        "trace": _trace(state, f"critique decision: {decision}"),
    }


def route_after_critique(state: AgentState) -> str:
    if state.get("decision") == "approve":
        return "pay"
    return "record"


def pay(state: AgentState) -> dict:
    if state.get("decision") != "approve":
        return {"payment": None}
    invoice = Invoice.model_validate(state["invoice"])
    if invoice.total is None or invoice.total <= 0 or not (invoice.vendor or "").strip():
        reason = state.get("decision_reason") or ""
        return {
            "decision": "reject",
            "payment": {"status": "blocked", "reason": "missing vendor or non-positive total"},
            "decision_reason": f"{reason} Payment blocked because the vendor or total is not payable.".strip(),
            "trace": _trace(state, "payment blocked"),
        }
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with connect(state["db_path"]) as conn:
        result = submit_payment(
            conn,
            vendor=invoice.vendor,
            amount=Decimal(invoice.total).quantize(Decimal("0.01")),
            currency=invoice.currency,
            invoice_number=invoice.invoice_number,
            processed_at=stamp,
        )
    updates: dict[str, Any] = {"payment": result, "trace": _trace(state, f"payment {result['status']}")}
    if result["status"] != "success":
        updates["decision"] = "needs_review"
        updates["decision_reason"] = (
            f"{state.get('decision_reason') or ''} Payment blocked: {result.get('reason', 'refused')}."
        ).strip()
    return updates


def record(state: AgentState) -> dict:
    invoice = Invoice.model_validate(state["invoice"]) if state.get("invoice") else Invoice()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    amount = None if invoice.total is None else f"{invoice.total:.2f}"
    with connect(state["db_path"]) as conn:
        conn.execute(
            """
            INSERT INTO processed_invoices (
                invoice_number, revision, source_path, decision, vendor, amount, currency, processed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                invoice.invoice_number,
                invoice.revision,
                state.get("invoice_path"),
                state.get("decision"),
                invoice.vendor,
                amount,
                invoice.currency,
                stamp,
            ),
        )
    return {"trace": _trace(state, f"recorded decision {state.get('decision')}")}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("load", load)
    graph.add_node("extract", extract)
    graph.add_node("check", check)
    graph.add_node("normalize", normalize)
    graph.add_node("validate", validate)
    graph.add_node("draft", draft)
    graph.add_node("critique", critique)
    graph.add_node("pay", pay)
    graph.add_node("record", record)
    graph.set_entry_point("load")
    graph.add_edge("load", "extract")
    graph.add_edge("extract", "check")
    graph.add_conditional_edges("check", route_after_check, {"extract": "extract", "normalize": "normalize"})
    graph.add_edge("normalize", "validate")
    graph.add_edge("validate", "draft")
    graph.add_edge("draft", "critique")
    graph.add_conditional_edges("critique", route_after_critique, {"pay": "pay", "record": "record"})
    graph.add_edge("pay", "record")
    graph.add_edge("record", END)
    return graph.compile()


GRAPH = build_graph()
