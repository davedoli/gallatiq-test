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
