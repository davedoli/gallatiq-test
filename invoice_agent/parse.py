from __future__ import annotations

import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

from invoice_agent.schemas import Invoice, LineItem

MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

ITEM_PATTERNS = (
    re.compile(
        r"(?i)^(?P<item>.+?)\s+qty\s*:?\s*(?P<qty>-?\d+(?:\.\d+)?)\s+"
        r"(?:unit price|@)\s*:?\s*\$?(?P<price>[\d,]+(?:\.\d+)?)"
    ),
    re.compile(
        r"(?i)^-\s*(?P<item>.+?)\s+x\s*(?P<qty>-?\d+(?:\.\d+)?)\s+"
        r"\$?(?P<price>[\d,]+(?:\.\d+)?)"
    ),
    re.compile(
        r"^(?P<item>[A-Za-z][A-Za-z0-9 .()/-]*?)\s+(?P<qty>-?\d+(?:\.\d+)?)\s+"
        r"\$?(?P<price>[\d,]+(?:\.\d+)?)\s+\$?(?P<amount>[\d,]+(?:\.\d+)?)\s*$"
    ),
)

SUMMARY_PATTERNS = (
    ("subtotal", re.compile(r"(?i)^sub\s*total\s*:?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s*$")),
    ("shipping", re.compile(r"(?i)^shipping\s*:?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s*$")),
    ("tax", re.compile(r"(?i)^(?:sales\s+tax|tax(?:\s*\([^)]*\))?)\s*:?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s*$")),
    ("total", re.compile(r"(?i)^(?:grand\s+total|total\s+amount|total|amt)\s*:?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s*$")),
)

SKIP_ITEM_NAMES = re.compile(
    r"(?i)^(subtotal|total|grand total|tax|sales tax|shipping|amount|qty|quantity|item|description|rate|price)$"
)


def money(value) -> Decimal | None:
    if value is None or value == "":
        return None
    if isinstance(value, Decimal):
        return value.quantize(Decimal("0.01"))
    text = str(value).strip().replace("$", "").replace(",", "")
    if text.lower() in {"null", "none", "nan"}:
        return None
    try:
        return Decimal(text).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


def parse_date_value(raw: str | None, *, anchor: date | None = None) -> tuple[date | None, str | None]:
    if raw is None:
        return None, None
    text = str(raw).strip().rstrip(".")
    if not text or text.lower() in {"null", "none"}:
        return None, None
    if text.lower() == "yesterday":
        base = anchor or date(2026, 1, 20)
        return base - timedelta(days=1), "yesterday"
    try:
        iso = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
        if iso:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3))), None
        us = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
        if us:
            return date(int(us.group(3)), int(us.group(1)), int(us.group(2))), None
        dmy = re.fullmatch(r"(\d{1,2})-([A-Za-z]{3})-(\d{4})", text)
        if dmy:
            return date(int(dmy.group(3)), MONTHS[dmy.group(2).lower()], int(dmy.group(1))), None
        written = re.fullmatch(r"([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})", text)
        if written:
            month = MONTHS[written.group(1).lower()[:3]]
            return date(int(written.group(3)), month, int(written.group(2))), None
    except (KeyError, ValueError):
        return None, None
    return None, None


def fix_ocr(text: str) -> str:
    """Replace letter O with zero inside tokens that already contain a digit (2O26, 3,500.O0)."""

    def repl(match: re.Match) -> str:
        token = match.group(0)
        if re.search(r"\d", token) and "O" in token:
            return token.replace("O", "0")
        return token

    return re.sub(r"\S+", repl, text)


def read_source(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "pdf":
        try:
            import pdfplumber
        except ImportError as exc:
            raise RuntimeError("pdfplumber is required to read PDF invoices") from exc
        parts: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                parts.append(page.extract_text() or "")
        return "\n".join(parts), "pdf"
    return path.read_text(encoding="utf-8"), suffix or "txt"


def parse_document(text: str, fmt: str) -> Invoice:
    kind = fmt.lower()
    if kind == "json":
        return parse_json(text)
    if kind == "xml":
        return parse_xml(text)
    if kind == "csv":
        return parse_csv(text)
    return parse_text(text)


def invoice_from_mapping(data: dict) -> Invoice:
    vendor = data.get("vendor")
    if isinstance(vendor, dict):
        vendor = vendor.get("name")
    if vendor is not None:
        vendor = str(vendor).strip()

    invoice_date, _ = parse_date_value(data.get("invoice_date") if "invoice_date" in data else data.get("date"))
    due_raw = data.get("due_date")
    due_date, due_note = parse_date_value(None if due_raw is None else str(due_raw), anchor=invoice_date)
    if data.get("due_date_note"):
        due_note = str(data["due_date_note"])

    items: list[LineItem] = []
    for raw in data.get("line_items") or []:
        name = str(raw.get("item") or raw.get("name") or "").strip()
        if not name:
            continue
        try:
            quantity = float(raw.get("quantity"))
        except (TypeError, ValueError):
            continue
        items.append(
            LineItem(
                item=name,
                quantity=quantity,
                unit_price=money(raw.get("unit_price")),
                amount=money(raw.get("amount")),
                note=(str(raw["note"]).strip() if raw.get("note") else None),
            )
        )

    currency = str(data.get("currency") or "USD").strip().upper() or "USD"
    number = data.get("invoice_number")
    return Invoice(
        invoice_number=(str(number).strip() if number else None),
        vendor=vendor,
        invoice_date=invoice_date,
        due_date=due_date,
        due_date_note=due_note,
        currency=currency,
        line_items=items,
        subtotal=money(data.get("subtotal")),
        tax_amount=money(data.get("tax_amount") if "tax_amount" in data else data.get("tax")),
        shipping=money(data.get("shipping")),
        total=money(data.get("total")),
        payment_terms=(str(data["payment_terms"]).strip() if data.get("payment_terms") else None),
        notes=(str(data["notes"]).strip() if data.get("notes") else None),
        revision=(str(data["revision"]).strip() if data.get("revision") else None),
    )


def parse_json(text: str) -> Invoice:
    return invoice_from_mapping(json.loads(text))


def parse_xml(text: str) -> Invoice:
    root = ET.fromstring(text)
    header = root.find("header")
    items = []
    for node in root.findall("./line_items/item"):
        items.append(
            {
                "item": (node.findtext("name") or "").strip(),
                "quantity": node.findtext("quantity"),
                "unit_price": node.findtext("unit_price"),
            }
        )
    totals = root.find("totals")
    return invoice_from_mapping(
        {
            "invoice_number": _xml_text(header, "invoice_number"),
            "vendor": _xml_text(header, "vendor"),
            "invoice_date": _xml_text(header, "date"),
            "due_date": _xml_text(header, "due_date"),
            "currency": _xml_text(header, "currency") or "USD",
            "line_items": items,
            "subtotal": _xml_text(totals, "subtotal"),
            "tax_amount": _xml_text(totals, "tax_amount"),
            "total": _xml_text(totals, "total"),
            "payment_terms": root.findtext("payment_terms"),
        }
    )


def _xml_text(node, tag: str) -> str | None:
    if node is None:
        return None
    return node.findtext(tag)


def parse_csv(text: str) -> Invoice:
    sample = text.lstrip("\ufeff").splitlines()[0] if text.strip() else ""
    if sample.lower().startswith("field,value") or sample.lower().startswith("field,"):
        return _parse_kv_csv(text)
    return _parse_row_csv(text)


def _parse_kv_csv(text: str) -> Invoice:
    rows = list(csv.reader(io.StringIO(text)))
    fields: dict[str, str] = {}
    items: list[dict] = []
    current: dict | None = None
    for row in rows[1:]:
        if len(row) < 2:
            continue
        field, value = row[0].strip(), row[1].strip()
        if field == "item":
            if current:
                items.append(current)
            current = {"item": value}
            continue
        if field in {"quantity", "unit_price"} and current is not None:
            current[field] = value
            continue
        if current:
            items.append(current)
            current = None
        fields[field] = value
    if current:
        items.append(current)
    return invoice_from_mapping(
        {
            "invoice_number": fields.get("invoice_number"),
            "vendor": fields.get("vendor"),
            "invoice_date": fields.get("date"),
            "due_date": fields.get("due_date"),
            "line_items": items,
            "subtotal": fields.get("subtotal"),
            "tax_amount": fields.get("tax"),
            "total": fields.get("total"),
            "payment_terms": fields.get("payment_terms"),
            "currency": fields.get("currency") or "USD",
        }
    )


def _parse_row_csv(text: str) -> Invoice:
    reader = csv.DictReader(io.StringIO(text))
    items: list[dict] = []
    header: dict[str, str] = {}
    subtotal = tax = total = None
    for row in reader:
        label = (row.get("Unit Price") or "").strip()
        item = (row.get("Item") or "").strip()
        if not item and re.search(r"(?i)subtotal|tax|total", label):
            amount = row.get("Line Total")
            if re.search(r"(?i)subtotal", label):
                subtotal = amount
            elif re.search(r"(?i)tax", label):
                tax = amount
            else:
                total = amount
            continue
        if not item:
            continue
        if not header:
            header = row
        items.append(
            {
                "item": item,
                "quantity": row.get("Qty"),
                "unit_price": row.get("Unit Price"),
                "amount": row.get("Line Total"),
            }
        )
    return invoice_from_mapping(
        {
            "invoice_number": (header.get("Invoice Number") or "").strip() or None,
            "vendor": (header.get("Vendor") or "").strip() or None,
            "invoice_date": header.get("Date"),
            "due_date": header.get("Due Date"),
            "line_items": items,
            "subtotal": subtotal,
            "tax_amount": tax,
            "total": total,
            "currency": "USD",
        }
    )


def parse_text(text: str) -> Invoice:
    cleaned = fix_ocr(text)
    vendor = None
    invoice_date_raw = None
    due_raw = None
    terms = None
    notes = None
    items: list[dict] = []
    summary: dict[str, str] = {}

    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped or re.fullmatch(r"[-=_\s]+", stripped):
            continue
        if vendor is None:
            match = re.match(r"(?i)^(?:vendor|vndr)\s*:\s*(.+)$", stripped)
            if match:
                vendor = match.group(1).strip()
                continue
        if due_raw is None:
            match = re.match(r"(?i)^(?:due\s*date|due\s*dt|due)\s*:\s*(.+)$", stripped)
            if match:
                due_raw = match.group(1).strip()
                continue
        if invoice_date_raw is None:
            match = re.match(r"(?i)^(?:invoice\s+date|date|dt)\s*:\s*(.+)$", stripped)
            if match:
                invoice_date_raw = match.group(1).strip()
                continue
        if terms is None:
            match = re.match(r"(?i)^(?:payment\s+terms|pymnt\s+terms|terms)\s*:\s*(.+)$", stripped)
            if match:
                terms = match.group(1).strip()
                continue
        if notes is None:
            match = re.match(r"(?i)^notes?\s*:\s*(.+)$", stripped)
            if match:
                notes = match.group(1).strip()
                continue
        item = _parse_item_line(stripped)
        if item:
            items.append(item)
            continue
        kind, value = _parse_summary_line(stripped)
        if kind and kind not in summary:
            summary[kind] = value

    if vendor is None:
        match = re.search(r"(?im)^[ \t]*from\s*:\s*(.+)$", cleaned)
        if match:
            vendor = match.group(1).strip()

    return invoice_from_mapping(
        {
            "invoice_number": _find_invoice_number(cleaned),
            "vendor": vendor,
            "invoice_date": invoice_date_raw,
            "due_date": due_raw,
            "line_items": items,
            "subtotal": summary.get("subtotal"),
            "tax_amount": summary.get("tax"),
            "shipping": summary.get("shipping"),
            "total": summary.get("total"),
            "payment_terms": terms,
            "notes": notes,
            "currency": "USD",
        }
    )


def _find_invoice_number(text: str) -> str | None:
    match = re.search(r"(?i)\bINV[-\s]?(\d{3,})\b", text)
    if match:
        return f"INV-{match.group(1)}"
    match = re.search(r"(?i)\binv\s*#\s*:?\s*(\d{3,})\b", text)
    if match:
        return f"INV-{match.group(1)}"
    return None


def _parse_item_line(line: str) -> dict | None:
    for pattern in ITEM_PATTERNS:
        match = pattern.match(line)
        if not match:
            continue
        item = re.sub(r"\s+", " ", match.group("item")).strip(" -")
        if not item or SKIP_ITEM_NAMES.fullmatch(item):
            continue
        payload = {
            "item": item,
            "quantity": match.group("qty"),
            "unit_price": match.group("price"),
        }
        if "amount" in match.groupdict() and match.group("amount"):
            payload["amount"] = match.group("amount")
        return payload
    return None


def _parse_summary_line(line: str) -> tuple[str | None, str | None]:
    for kind, pattern in SUMMARY_PATTERNS:
        match = pattern.match(line)
        if match:
            return kind, match.group(1)
    return None, None
