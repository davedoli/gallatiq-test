from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from invoice_agent.parse import fix_ocr, money
from invoice_agent.policy import AUTO_PAY_LIMIT, DISCOUNT_VARIANCE_LIMIT, PRICE_VARIANCE_LIMIT
from invoice_agent.schemas import Finding, Invoice
from invoice_agent.tools import lookup_inventory

MONEY_TOLERANCE = Decimal("0.05")
PRESSURE = re.compile(r"urgent|wire transfer|pay immediately|avoid penalties", re.IGNORECASE)
# A comma is a thousands separator in $5,000.00 and a field break in CSV "10,250.00".
# Collect both readings so a printed price is recognized either way.
_PLAIN_AMOUNT = re.compile(r"(?<![A-Za-z])-?\$?\s*-?\d+(?:\.\d+)?")
_THOUSANDS_AMOUNT = re.compile(r"(?<![A-Za-z])-?\$?\s*-?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_LABELED_TOTAL = re.compile(
    r"(?i)(?<![A-Za-z])(?:grand\s+total|total\s+amount|total)\s*[:=>,]+\s*\$?\s*(-?[\d,]+(?:\.\d+)?)"
)


def line_extended(line) -> Decimal | None:
    if line.amount is not None:
        return line.amount
    if line.unit_price is None:
        return None
    return (Decimal(str(line.quantity)) * line.unit_price).quantize(Decimal("0.01"))


def audit_math(invoice: Invoice) -> list[Finding]:
    findings: list[Finding] = []
    if not invoice.line_items:
        findings.append(Finding(severity="error", code="missing_line_items", message="The invoice has no line items."))
    extended: list[Decimal] = []
    for line in invoice.line_items:
        qty = Decimal(str(line.quantity))
        if qty < 0:
            findings.append(
                Finding(
                    severity="error",
                    code="negative_quantity",
                    message=f"{line.item} has a negative quantity ({_qty(qty)}).",
                )
            )
        if line.unit_price is not None and line.amount is not None:
            expected = (qty * line.unit_price).quantize(Decimal("0.01"))
            if abs(expected - line.amount) > MONEY_TOLERANCE:
                findings.append(
                    Finding(
                        severity="error",
                        code="math_mismatch",
                        message=(
                            f"{line.item} extends to {expected:,.2f} "
                            f"but the line amount is {line.amount:,.2f}."
                        ),
                    )
                )
        amount = line_extended(line)
        if amount is not None:
            extended.append(amount)

    if invoice.total is None:
        findings.append(Finding(severity="error", code="missing_total", message="The invoice total is missing."))
    elif invoice.total < 0:
        findings.append(
            Finding(
                severity="error",
                code="negative_total",
                message=f"The invoice total is negative ({invoice.total:,.2f} {invoice.currency}).",
            )
        )

    if extended:
        lines_sum = sum(extended, Decimal("0.00")).quantize(Decimal("0.01"))
        if invoice.subtotal is not None and abs(invoice.subtotal - lines_sum) > MONEY_TOLERANCE:
            findings.append(
                Finding(
                    severity="error",
                    code="math_mismatch",
                    message=f"Subtotal {invoice.subtotal:,.2f} does not match the line items ({lines_sum:,.2f}).",
                )
            )
        tax = invoice.tax_amount or Decimal("0.00")
        shipping = invoice.shipping or Decimal("0.00")
        expected_total = (lines_sum + tax + shipping).quantize(Decimal("0.01"))
        if invoice.total is not None and abs(invoice.total - expected_total) > MONEY_TOLERANCE:
            findings.append(
                Finding(
                    severity="error",
                    code="math_mismatch",
                    message=(
                        f"Stated total {invoice.total:,.2f} does not match "
                        f"lines + tax + shipping ({expected_total:,.2f})."
                    ),
                )
            )
    return findings


def audit_source(invoice: Invoice, raw_text: str) -> list[Finding]:
    """Reject an extraction whose payable fields are not on the document."""
    findings: list[Finding] = []
    printed = _printed_amounts(raw_text)
    folded = _fold(fix_ocr(raw_text or ""))
    _require_printed(invoice.total, "total", printed, findings, always=True)
    _require_printed(invoice.subtotal, "subtotal", printed, findings)
    _require_printed(invoice.tax_amount, "tax", printed, findings)
    _require_printed(invoice.shipping, "shipping", printed, findings)
    for line in invoice.line_items:
        label = line.item or "a line"
        _require_printed(line.unit_price, f"{label} unit price", printed, findings)
        _require_printed(line.amount, f"{label} line amount", printed, findings)
        if line.item and _fold(line.item) not in folded:
            findings.append(
                Finding(
                    severity="error",
                    code="extraction_mismatch",
                    message=f"{line.item} is not printed on the document.",
                )
            )
    vendor = (invoice.vendor or "").strip()
    if vendor and _fold(vendor) not in folded:
        findings.append(
            Finding(
                severity="error",
                code="extraction_mismatch",
                message=f"Vendor {vendor} is not printed on the document.",
            )
        )
    number = (invoice.invoice_number or "").strip()
    if number and _fold(number) not in folded:
        findings.append(
            Finding(
                severity="error",
                code="extraction_mismatch",
                message=f"Invoice number {number} is not printed on the document.",
            )
        )
    labeled = _labeled_totals(raw_text)
    if invoice.total is not None and labeled and all(abs(invoice.total - value) > MONEY_TOLERANCE for value in labeled):
        shown = ", ".join(f"{value:,.2f}" for value in labeled)
        findings.append(
            Finding(
                severity="error",
                code="extraction_mismatch",
                message=(
                    f"Extracted total {invoice.total:,.2f} does not match the printed total ({shown})."
                ),
            )
        )
    return findings


def collect_findings(invoice: Invoice, raw_text: str, conn: sqlite3.Connection) -> list[Finding]:
    findings = audit_math(invoice)
    findings.extend(audit_source(invoice, raw_text))
    if not (invoice.invoice_number or "").strip():
        findings.append(
            Finding(
                severity="warning",
                code="missing_invoice_number",
                message="The invoice number is missing, so a duplicate payment cannot be checked.",
            )
        )
    if not (invoice.vendor or "").strip():
        findings.append(Finding(severity="error", code="missing_vendor", message="The vendor name is missing."))
    if invoice.due_date is None and not invoice.due_date_note:
        findings.append(Finding(severity="warning", code="missing_due_date", message="The due date is missing."))
    elif (
        invoice.due_date is not None
        and invoice.invoice_date is not None
        and invoice.due_date < invoice.invoice_date
    ):
        note = f" The document said '{invoice.due_date_note}'." if invoice.due_date_note else ""
        findings.append(
            Finding(
                severity="error",
                code="due_date_inconsistent",
                message=(
                    f"Due date {invoice.due_date.isoformat()} is before the invoice date "
                    f"{invoice.invoice_date.isoformat()}.{note}"
                ),
            )
        )

    quantities: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    for line in invoice.line_items:
        if line.quantity < 0:
            continue
        name = line.normalized_item or line.item
        quantities[name] += Decimal(str(line.quantity))

    for name, qty in quantities.items():
        info = lookup_inventory(conn, name)
        if not info["found"]:
            findings.append(
                Finding(
                    severity="error",
                    code="unknown_item",
                    message=f"{name} is not in the inventory database.",
                )
            )
            continue
        stock = Decimal(str(info["stock"]))
        if stock <= 0 and qty > 0:
            findings.append(
                Finding(
                    severity="error",
                    code="out_of_stock",
                    message=f"{name} has no stock and the invoice requests {_qty(qty)}.",
                )
            )
        elif qty > stock:
            findings.append(
                Finding(
                    severity="error",
                    code="insufficient_stock",
                    message=f"{name} requests {_qty(qty)} but only {_qty(stock)} are in stock.",
                )
            )

    if invoice.currency.upper() == "USD":
        for line in invoice.line_items:
            if line.unit_price is None:
                continue
            info = lookup_inventory(conn, line.normalized_item or line.item)
            if not info["found"] or info["unit_price"] is None:
                continue
            catalog = Decimal(str(info["unit_price"]))
            if catalog <= 0:
                continue
            variance = abs(line.unit_price - catalog) / catalog
            discount = bool(line.note and re.search(r"discount", line.note, re.IGNORECASE))
            if discount and line.unit_price <= catalog and variance <= DISCOUNT_VARIANCE_LIMIT:
                continue
            if variance > PRICE_VARIANCE_LIMIT:
                findings.append(
                    Finding(
                        severity="warning",
                        code="price_variance",
                        message=(
                            f"{line.item} is priced at {line.unit_price:,.2f} {invoice.currency} "
                            f"against a catalog price of {catalog:,.2f}."
                        ),
                    )
                )
    elif invoice.currency:
        findings.append(
            Finding(
                severity="warning",
                code="non_usd_currency",
                message=f"The invoice is in {invoice.currency}, so it needs a person before payment.",
            )
        )

    if invoice.total is not None and invoice.total > AUTO_PAY_LIMIT:
        findings.append(
            Finding(
                severity="warning",
                code="amount_over_threshold",
                message=f"The total {invoice.total:,.2f} {invoice.currency} is over the ${AUTO_PAY_LIMIT:,.0f} auto-pay limit.",
            )
        )

    _flag_suspicious(invoice, raw_text, findings)
    _flag_duplicate(invoice, conn, findings)
    findings.sort(key=_finding_rank)
    return findings


def _flag_suspicious(invoice: Invoice, raw_text: str, findings: list[Finding]) -> None:
    fraud_name = bool(re.search(r"fraud", invoice.vendor or "", re.IGNORECASE))
    pressure = bool(PRESSURE.search(raw_text or ""))
    stock_problem = any(finding.code in {"out_of_stock", "unknown_item", "insufficient_stock"} for finding in findings)
    high_value = invoice.total is not None and invoice.total > AUTO_PAY_LIMIT
    if fraud_name or (pressure and (stock_problem or high_value)):
        bits = []
        if fraud_name:
            bits.append(f"the vendor name is {invoice.vendor}")
        if pressure:
            bits.append("the document pressures for an urgent wire")
        findings.append(
            Finding(
                severity="error",
                code="suspicious_payment_request",
                message=f"Payment looks suspicious because {' and '.join(bits)}.",
            )
        )
    elif pressure:
        findings.append(
            Finding(
                severity="warning",
                code="payment_pressure",
                message="The document pressures for immediate payment, so it needs a person before payment.",
            )
        )


def prior_activity(conn: sqlite3.Connection, invoice_number: str) -> tuple[bool, sqlite3.Row | None]:
    paid = conn.execute(
        "SELECT id FROM payments WHERE invoice_number = ? AND status = 'success' LIMIT 1",
        (invoice_number,),
    ).fetchone()
    seen = conn.execute(
        """
        SELECT decision, source_path FROM processed_invoices
        WHERE invoice_number = ?
        ORDER BY id DESC LIMIT 1
        """,
        (invoice_number,),
    ).fetchone()
    return paid is not None, seen


_FINDING_RANK = {
    "suspicious_payment_request": 0,
    "out_of_stock": 1,
    "unknown_item": 1,
    "insufficient_stock": 1,
    "negative_quantity": 2,
    "negative_total": 2,
    "missing_vendor": 2,
    "missing_line_items": 2,
    "missing_total": 3,
    "math_mismatch": 3,
    "due_date_inconsistent": 4,
}


def _finding_rank(finding: Finding) -> tuple[int, int]:
    severity = 0 if finding.severity == "error" else 1
    return severity, _FINDING_RANK.get(finding.code, 5)


def _printed_amounts(text: str) -> set[Decimal]:
    found: set[Decimal] = set()
    cleaned = fix_ocr(text or "")
    for pattern in (_PLAIN_AMOUNT, _THOUSANDS_AMOUNT):
        for match in pattern.finditer(cleaned):
            value = money(re.sub(r"\s+", "", match.group(0)))
            if value is not None:
                found.add(value)
    return found


def _labeled_totals(text: str) -> list[Decimal]:
    totals: list[Decimal] = []
    for match in _LABELED_TOTAL.finditer(fix_ocr(text or "")):
        value = money(match.group(1))
        if value is not None and value not in totals:
            totals.append(value)
    return totals


def _require_printed(amount: Decimal | None, label: str, printed: set[Decimal], findings: list[Finding], always: bool = False) -> None:
    if amount is None or (amount == 0 and not always):
        return
    if any(abs(amount - value) <= MONEY_TOLERANCE for value in printed):
        return
    findings.append(
        Finding(
            severity="error",
            code="extraction_mismatch",
            message=f"The extracted {label} {amount:,.2f} is not printed on the document.",
        )
    )


def _fold(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _qty(value: Decimal) -> str:
    if value == value.to_integral():
        return str(int(value))
    return format(value, "f")


def _flag_duplicate(invoice: Invoice, conn: sqlite3.Connection, findings: list[Finding]) -> None:
    number = invoice.invoice_number
    if not number:
        return
    paid, seen = prior_activity(conn, number)
    if not paid and seen is None:
        return
    if paid:
        detail = "a successful payment on the ledger"
    else:
        source = Path(seen["source_path"]).name
        detail = f"an earlier {seen['decision']} decision from {source}"
    findings.append(
        Finding(
            severity="warning",
            code="duplicate_or_revision",
            message=f"{number} already has {detail}.",
        )
    )
