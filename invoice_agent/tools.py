from __future__ import annotations

import sqlite3
from decimal import Decimal

from invoice_agent.normalize import canonical_item

INVENTORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_inventory",
            "description": "Look up on-hand stock and the catalog unit price for one item.",
            "parameters": {
                "type": "object",
                "properties": {"item": {"type": "string", "description": "Item name as printed on the invoice"}},
                "required": ["item"],
            },
        },
    }
]


def lookup_inventory(conn: sqlite3.Connection, item: str) -> dict:
    canonical = canonical_item(item)
    row = conn.execute(
        "SELECT item, stock, unit_price FROM inventory WHERE item = ?",
        (canonical,),
    ).fetchone()
    if row is None:
        return {"item": item, "normalized_item": canonical, "found": False}
    return {
        "item": item,
        "normalized_item": row["item"],
        "found": True,
        "stock": row["stock"],
        "unit_price": row["unit_price"],
    }


def submit_payment(
    conn: sqlite3.Connection,
    *,
    vendor: str,
    amount: Decimal,
    currency: str,
    invoice_number: str | None,
    processed_at: str,
) -> dict:
    """Mock bank call. A second success for the same invoice number is refused."""
    if invoice_number:
        existing = conn.execute(
            "SELECT id FROM payments WHERE invoice_number = ? AND status = 'success'",
            (invoice_number,),
        ).fetchone()
        if existing:
            return {
                "status": "blocked",
                "reason": "invoice already paid",
                "vendor": vendor,
                "amount": f"{amount:.2f}",
                "currency": currency,
                "invoice_number": invoice_number,
            }
    print(f"Paid {amount:.2f} to {vendor}")
    conn.execute(
        """
        INSERT INTO payments (invoice_number, vendor, amount, currency, status, processed_at)
        VALUES (?, ?, ?, ?, 'success', ?)
        """,
        (invoice_number, vendor, f"{amount:.2f}", currency, processed_at),
    )
    return {
        "status": "success",
        "vendor": vendor,
        "amount": f"{amount:.2f}",
        "currency": currency,
        "invoice_number": invoice_number,
    }
