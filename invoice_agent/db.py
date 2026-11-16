from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS inventory (
    item TEXT PRIMARY KEY,
    stock INTEGER NOT NULL,
    unit_price REAL
);

CREATE TABLE IF NOT EXISTS payments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_number TEXT,
    vendor TEXT,
    amount TEXT,
    currency TEXT,
    status TEXT,
    processed_at TEXT
);

CREATE TABLE IF NOT EXISTS processed_invoices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_number TEXT,
    revision TEXT,
    source_path TEXT,
    decision TEXT,
    vendor TEXT,
    amount TEXT,
    currency TEXT,
    processed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_payments_invoice ON payments(invoice_number);
CREATE INDEX IF NOT EXISTS idx_processed_invoice ON processed_invoices(invoice_number);
"""

# unit_price extends the case seed so the desk can flag price changes.
SEED = (
    ("WidgetA", 15, 250.0),
    ("WidgetB", 10, 500.0),
    ("GadgetX", 5, 750.0),
    ("FakeItem", 0, 1000.0),
)


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    count = conn.execute("SELECT COUNT(*) AS n FROM inventory").fetchone()["n"]
    if count == 0:
        conn.executemany(
            "INSERT INTO inventory (item, stock, unit_price) VALUES (?, ?, ?)",
            SEED,
        )


def reset_runtime(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM payments")
    conn.execute("DELETE FROM processed_invoices")


@contextmanager
def connect(db_path: str | Path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        init_db(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()
