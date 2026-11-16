from __future__ import annotations

from decimal import Decimal

from invoice_agent.schemas import Finding, Invoice

AUTO_PAY_LIMIT = Decimal("10000.00")
PRICE_VARIANCE_LIMIT = Decimal("0.02")


def decide(findings: list[Finding]) -> str:
    if any(finding.severity == "error" for finding in findings):
        return "reject"
    if any(finding.severity == "warning" for finding in findings):
        return "needs_review"
    return "approve"


def apply_critique(rules_decision: str, model_decision: str | None) -> str:
    """The model may hold a payment. It cannot release one the rules refused."""
    if rules_decision != "approve":
        return rules_decision
    if model_decision in {"needs_review", "reject"}:
        return "needs_review"
    return "approve"


def explain(decision: str, findings: list[Finding], invoice: Invoice) -> str:
    vendor = invoice.vendor or "Unknown vendor"
    number = invoice.invoice_number or "an unnumbered invoice"
    if invoice.total is None:
        amount = f"an unstated total in {invoice.currency}"
    else:
        amount = f"{invoice.total:,.2f} {invoice.currency}"
    if decision == "approve":
        return (
            f"{vendor} {number} for {amount} is in stock, the totals match the line items, "
            f"and the amount is within the ${AUTO_PAY_LIMIT:,.0f} auto-pay limit."
        )
    label = "Rejected" if decision == "reject" else "Held for a person to review"
    details = " ".join(finding.message for finding in findings)
    return f"{label}: {vendor} {number} for {amount}. {details}".strip()
