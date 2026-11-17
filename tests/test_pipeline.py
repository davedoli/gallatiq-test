"""Decisions the desk must make on the sample invoices, with no model key."""

from pathlib import Path

from invoice_agent.llm import reset_llm_client, set_llm_client
from invoice_agent.policy import apply_critique
from invoice_agent.report import summarize
from invoice_agent.runner import run_directory, run_path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "invoices"


def _codes(state: dict) -> set[str]:
    return {finding["code"] for finding in state["findings"]}


def test_clean_invoice_is_paid(tmp_path):
    state = run_path(SAMPLES / "invoice_1001.txt", db_path=tmp_path / "inventory.db", use_llm=False)
    assert state["decision"] == "approve"
    assert state["payment"]["status"] == "success"
    assert state["payment"]["vendor"] == "Widgets Inc."
    assert state["payment"]["amount"] == "5000.00"


def test_case_flags(tmp_path):
    db = tmp_path / "inventory.db"

    over_stock = run_path(SAMPLES / "invoice_1002.txt", db_path=db, use_llm=False)
    assert over_stock["decision"] == "reject"
    assert "insufficient_stock" in _codes(over_stock)
    assert over_stock["payment"] is None

    fraud = run_path(SAMPLES / "invoice_1003.txt", db_path=db, use_llm=False)
    assert fraud["decision"] == "reject"
    assert {"out_of_stock", "suspicious_payment_request"} <= _codes(fraud)

    unknown = run_path(SAMPLES / "invoice_1008.txt", db_path=db, use_llm=False)
    assert unknown["decision"] == "reject"
    assert "unknown_item" in _codes(unknown)

    widget_c = run_path(SAMPLES / "invoice_1016.json", db_path=db, use_llm=False)
    assert widget_c["decision"] == "reject"
    assert "unknown_item" in _codes(widget_c)

    broken = run_path(SAMPLES / "invoice_1009.json", db_path=db, use_llm=False)
    assert broken["decision"] == "reject"
    assert {"negative_quantity", "negative_total", "missing_vendor"} <= _codes(broken)


def test_folder_pays_only_clean_invoices_once(tmp_path):
    states = run_directory(SAMPLES, db_path=tmp_path / "inventory.db", use_llm=False, reset=True)
    by_file = {Path(state["invoice_path"]).name: state for state in states}

    assert by_file["invoice_1001.txt"]["decision"] == "approve"
    assert by_file["invoice_1004.json"]["decision"] == "approve"
    assert by_file["invoice_1006.csv"]["decision"] == "approve"
    assert by_file["invoice_1011.pdf"]["decision"] == "approve"
    assert by_file["invoice_1015.csv"]["decision"] == "approve"
    assert by_file["invoice_1012.pdf"]["decision"] == "approve"

    # The text twin of a PDF that already paid is the same invoice number.
    assert by_file["invoice_1011.txt"]["decision"] == "needs_review"
    assert "duplicate_or_revision" in _codes(by_file["invoice_1011.txt"])
    assert by_file["invoice_1004_revised.json"]["decision"] == "needs_review"

    assert by_file["invoice_1010.txt"]["decision"] == "needs_review"
    assert "price_variance" in _codes(by_file["invoice_1010.txt"])
    assert by_file["invoice_1014.xml"]["decision"] == "needs_review"
    assert "non_usd_currency" in _codes(by_file["invoice_1014.xml"])

    paid = [state for state in states if (state.get("payment") or {}).get("status") == "success"]
    assert {Path(state["invoice_path"]).name for state in paid} == {
        "invoice_1001.txt",
        "invoice_1004.json",
        "invoice_1006.csv",
        "invoice_1011.pdf",
        "invoice_1012.pdf",
        "invoice_1015.csv",
    }
    assert summarize(states)["paid"] == 29115


def test_rules_ignore_a_model_that_wants_to_pay(tmp_path):
    assert apply_critique("reject", "approve") == "reject"
    assert apply_critique("approve", "needs_review") == "needs_review"

    class Client:
        def complete_json(self, system, user):
            return {"recommended_decision": "approve", "rationale": "Pay it."}

        def complete(self, messages, tools=None, json_mode=False):
            return type("Msg", (), {"content": "Stock is short.", "tool_calls": None})()

    token = set_llm_client(Client())
    try:
        state = run_path(SAMPLES / "invoice_1002.txt", db_path=tmp_path / "inventory.db", use_llm=True)
    finally:
        reset_llm_client(token)
    assert state["decision"] == "reject"
    assert state["payment"] is None
    assert "Pay it." in state["critique_notes"]


def test_model_can_hold_a_clean_invoice(tmp_path):
    class Client:
        def complete_json(self, system, user):
            if "extract" in system.lower() or "Schema keys" in system:
                raise ValueError("use the parser")
            return {"recommended_decision": "needs_review", "rationale": "Vendor nickname does not match the master file."}

        def complete(self, messages, tools=None, json_mode=False):
            return type("Msg", (), {"content": "On-hand stock covers the lines.", "tool_calls": None})()

    token = set_llm_client(Client())
    try:
        state = run_path(SAMPLES / "invoice_1001.txt", db_path=tmp_path / "inventory.db", use_llm=True)
    finally:
        reset_llm_client(token)
    assert state["decision"] == "needs_review"
    assert state["payment"] is None


def test_failed_critique_holds_a_clean_invoice(tmp_path):
    class Client:
        def complete_json(self, system, user):
            if "Schema keys" in system:
                raise ValueError("use the parser")
            raise RuntimeError("critique down")

        def complete(self, messages, tools=None, json_mode=False):
            return type("Msg", (), {"content": "On-hand stock covers the lines.", "tool_calls": None})()

    token = set_llm_client(Client())
    try:
        state = run_path(SAMPLES / "invoice_1001.txt", db_path=tmp_path / "inventory.db", use_llm=True)
    finally:
        reset_llm_client(token)
    assert state["decision"] == "needs_review"
    assert "critique_unavailable" in _codes(state)
    assert state["payment"] is None


def test_rewritten_extraction_cannot_be_paid(tmp_path):
    class Client:
        def complete_json(self, system, user):
            if "Schema keys" in system:
                return {
                    "invoice_number": "INV-1003",
                    "vendor": "Widgets Inc.",
                    "invoice_date": "2026-01-20",
                    "due_date": "2026-02-20",
                    "currency": "USD",
                    "line_items": [{"item": "WidgetA", "quantity": 10, "unit_price": 250, "amount": 2500}],
                    "subtotal": 2500,
                    "tax_amount": 0,
                    "total": 2500,
                }
            return {"recommended_decision": "approve", "rationale": "Looks clean."}

        def complete(self, messages, tools=None, json_mode=False):
            return type("Msg", (), {"content": "Stock covers the order.", "tool_calls": None})()

    token = set_llm_client(Client())
    try:
        state = run_path(SAMPLES / "invoice_1003.txt", db_path=tmp_path / "inventory.db", use_llm=True)
    finally:
        reset_llm_client(token)
    assert state["decision"] == "reject"
    assert "extraction_mismatch" in _codes(state)
    assert state["payment"] is None


def test_pressure_missing_number_and_deep_discount_are_not_paid(tmp_path):
    db = tmp_path / "inventory.db"
    pressure = tmp_path / "pressure.txt"
    pressure.write_text(
        """INVOICE

Vendor: Widgets Inc.
Invoice Number: INV-9002
Date: 2026-01-15
Due Date: 2026-02-01

Items:
  WidgetA    qty: 2    unit price: $250.00

Subtotal: $500.00
Tax (0%): $0.00
Total Amount: $500.00

Payment Terms: Net 15
Notes: URGENT - Pay immediately. Wire transfer preferred.
""",
        encoding="utf-8",
    )
    state = run_path(pressure, db_path=db, use_llm=False)
    assert state["decision"] == "needs_review"
    assert "payment_pressure" in _codes(state)
    assert "suspicious_payment_request" not in _codes(state)
    assert state["payment"] is None

    unnumbered = tmp_path / "unnumbered.txt"
    unnumbered.write_text(
        """INVOICE

Vendor: Widgets Inc.
Date: 2026-01-15
Due Date: 2026-02-01

Items:
  WidgetA    qty: 2    unit price: $250.00

Subtotal: $500.00
Tax (0%): $0.00
Total Amount: $500.00

Payment Terms: Net 15
""",
        encoding="utf-8",
    )
    state = run_path(unnumbered, db_path=db, use_llm=False)
    assert state["decision"] == "needs_review"
    assert "missing_invoice_number" in _codes(state)
    assert state["payment"] is None

    deep = tmp_path / "deep.json"
    deep.write_text(
        """{
  "invoice_number": "INV-9003",
  "vendor": "Widgets Inc.",
  "date": "2026-01-15",
  "due_date": "2026-02-01",
  "line_items": [
    {"item": "WidgetA", "quantity": 2, "unit_price": 100.00, "amount": 200.00, "note": "Volume discount"}
  ],
  "subtotal": 200.00,
  "tax_amount": 0,
  "total": 200.00,
  "currency": "USD"
}
""",
        encoding="utf-8",
    )
    state = run_path(deep, db_path=db, use_llm=False)
    assert state["decision"] == "needs_review"
    assert "price_variance" in _codes(state)
    assert state["payment"] is None

    modest = tmp_path / "modest.json"
    modest.write_text(
        """{
  "invoice_number": "INV-9004",
  "vendor": "Widgets Inc.",
  "date": "2026-01-15",
  "due_date": "2026-02-01",
  "line_items": [
    {"item": "WidgetA", "quantity": 2, "unit_price": 240.00, "amount": 480.00, "note": "Volume discount"}
  ],
  "subtotal": 480.00,
  "tax_amount": 0,
  "total": 480.00,
  "currency": "USD"
}
""",
        encoding="utf-8",
    )
    state = run_path(modest, db_path=db, use_llm=False)
    assert state["decision"] == "approve"
    assert state["payment"]["status"] == "success"

    marked_up = tmp_path / "marked_up.json"
    marked_up.write_text(
        """{
  "invoice_number": "INV-9005",
  "vendor": "Widgets Inc.",
  "date": "2026-01-15",
  "due_date": "2026-02-01",
  "line_items": [
    {"item": "WidgetA", "quantity": 2, "unit_price": 300.00, "amount": 600.00, "note": "Volume discount"}
  ],
  "subtotal": 600.00,
  "tax_amount": 0,
  "total": 600.00,
  "currency": "USD"
}
""",
        encoding="utf-8",
    )
    state = run_path(marked_up, db_path=db, use_llm=False)
    assert state["decision"] == "needs_review"
    assert "price_variance" in _codes(state)
    assert state["payment"] is None
