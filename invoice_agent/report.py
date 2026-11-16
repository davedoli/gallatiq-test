from __future__ import annotations

from collections import Counter
from decimal import Decimal
from pathlib import Path


def full_report(state: dict) -> str:
    invoice = state.get("invoice") or {}
    lines = [
        f"File: {state.get('invoice_path')}",
        f"Invoice: {invoice.get('invoice_number') or '—'}",
        f"Vendor: {invoice.get('vendor') or '—'}",
        f"Total: {_money(invoice.get('total'))} {invoice.get('currency') or ''}".rstrip(),
        f"Decision: {(state.get('decision') or '').upper()}",
        f"Reason: {state.get('decision_reason') or '—'}",
    ]
    findings = state.get("findings") or []
    if findings:
        lines.append("Findings:")
        for finding in findings:
            lines.append(f"  - [{finding['severity']}] {finding['code']}: {finding['message']}")
    else:
        lines.append("Findings: none")
    payment = state.get("payment")
    if payment and payment.get("status") == "success":
        lines.append(
            f"Payment: recorded {payment['amount']} {payment.get('currency', '')} to {payment['vendor']} "
            "(mock ledger only)."
        )
    elif payment:
        lines.append(f"Payment: {payment.get('status')} ({payment.get('reason', '')})")
    else:
        lines.append("Payment: not sent")
    if state.get("critique_notes"):
        lines.append(f"Critique: {state['critique_notes']}")
    lines.append("Trace:")
    for index, step in enumerate(state.get("trace") or [], start=1):
        lines.append(f"  {index}. {step}")
    return "\n".join(lines)


def table(states: list[dict]) -> str:
    header = f"{'File':<32} {'Invoice':<12} {'Decision':<14} {'Total':>12}  Note"
    rows = [header, "-" * len(header)]
    for state in states:
        invoice = state.get("invoice") or {}
        note = _note(state)
        rows.append(
            f"{Path(state['invoice_path']).name:<32} "
            f"{(invoice.get('invoice_number') or '—'):<12} "
            f"{(state.get('decision') or ''):<14} "
            f"{_money(invoice.get('total')):>12}  {note}"
        )
    summary = summarize(states)
    counts = summary["counts"]
    rows.append("")
    rows.append(
        "Straight-through paid "
        f"${summary['paid']:,.2f} across {counts.get('approve', 0)} invoice(s). "
        f"Held for review: {counts.get('needs_review', 0)}. "
        f"Rejected: {counts.get('reject', 0)}."
    )
    return "\n".join(rows)


def summarize(states: list[dict]) -> dict:
    counts: Counter[str] = Counter(state.get("decision") or "" for state in states)
    paid = Decimal("0.00")
    for state in states:
        payment = state.get("payment") or {}
        if payment.get("status") == "success":
            paid += Decimal(str(payment["amount"]))
    return {"counts": counts, "paid": paid, "files": len(states)}


def _note(state: dict) -> str:
    payment = state.get("payment") or {}
    if payment.get("status") == "success":
        return "paid"
    findings = state.get("findings") or []
    if not findings:
        return ""
    errors = [finding for finding in findings if finding["severity"] == "error"]
    chosen = errors[0] if errors else findings[0]
    return chosen["code"]


def _money(value) -> str:
    if value is None:
        return "—"
    return f"{Decimal(str(value)):,.2f}"
