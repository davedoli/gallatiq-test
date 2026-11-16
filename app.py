"""Local desk for the invoice agents. Run from the repo root: streamlit run app.py"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import streamlit as st

from invoice_agent.db import connect, reset_runtime
from invoice_agent.llm import build_client, load_env
from invoice_agent.report import summarize
from invoice_agent.runner import run_directory, run_path

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "data" / "invoices"
DB_PATH = str(ROOT / "inventory.db")
load_env()

st.set_page_config(page_title="Acme invoice desk", layout="wide")
st.title("Acme invoice desk")
st.caption("Clean invoices under $10,000 are paid on the mock ledger. Everything else stops, with a written reason.")

files = sorted(path for path in SAMPLES.iterdir() if path.suffix.lower() in {".pdf", ".txt", ".json", ".csv", ".xml"})
labels = [path.name for path in files]

with st.sidebar:
    st.header("Run")
    choice = st.selectbox("Sample invoice", labels, index=labels.index("invoice_1001.txt"))
    uploaded = st.file_uploader("Or upload an invoice", type=["pdf", "txt", "json", "csv", "xml"])
    use_llm = st.checkbox("Use the language model", value=build_client() is not None)
    if st.button("Clear the payment ledger"):
        with connect(DB_PATH) as conn:
            reset_runtime(conn)
        st.session_state.pop("result", None)
        st.session_state.pop("batch", None)
        st.success("Ledger cleared.")

if use_llm and build_client() is None:
    st.error("No model key is set. Add XAI_API_KEY to a .env file, or turn the language model off to use the local parsers.")

run_clicked = st.button("Run this invoice", type="primary")
if run_clicked:
    if use_llm and build_client() is None:
        st.stop()
    if uploaded is not None:
        target = ROOT / "logs" / "uploads" / uploaded.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(uploaded.getvalue())
    else:
        target = SAMPLES / choice
    with st.spinner("Working through the invoice..."):
        st.session_state["result"] = run_path(target, db_path=DB_PATH, use_llm=use_llm, log_dir=ROOT / "logs")

result = st.session_state.get("result")
if result:
    invoice = result.get("invoice") or {}
    decision = result.get("decision") or ""
    banner = {"approve": st.success, "reject": st.error, "needs_review": st.warning}.get(decision, st.info)
    banner(result.get("decision_reason") or decision.upper())
    metrics = st.columns(4)
    metrics[0].metric("Decision", decision.replace("_", " ") or "—")
    metrics[1].metric("Vendor", invoice.get("vendor") or "—")
    total = invoice.get("total")
    metrics[2].metric("Total", "—" if total is None else f"{Decimal(str(total)):,.2f} {invoice.get('currency', '')}")
    metrics[3].metric("Invoice", invoice.get("invoice_number") or "—")

    items = invoice.get("line_items") or []
    left, right = st.columns(2)
    with left:
        st.subheader("Line items")
        st.dataframe(
            [
                {
                    "Printed name": line.get("item"),
                    "Catalog id": line.get("normalized_item"),
                    "Qty": line.get("quantity"),
                    "Unit price": line.get("unit_price"),
                    "Amount": line.get("amount"),
                }
                for line in items
            ],
            hide_index=True,
            use_container_width=True,
        )
    with right:
        st.subheader("Findings")
        findings = result.get("findings") or []
        if not findings:
            st.write("No findings. The invoice matched stock, math, and the auto-pay rules.")
        for finding in findings:
            st.write(f"**{finding['severity']} · {finding['code']}** — {finding['message']}")

    payment = result.get("payment")
    if payment and payment.get("status") == "success":
        st.info(f"Mock payment recorded: {payment['amount']} {payment.get('currency', '')} to {payment['vendor']}. No bank was contacted.")
    elif decision != "approve":
        st.info("Payment was not sent.")

    with st.expander("Agent trace", expanded=True):
        for index, step in enumerate(result.get("trace") or [], start=1):
            st.write(f"{index}. {step}")
        if result.get("critique_notes"):
            st.write(result["critique_notes"])
    with st.expander("Source text"):
        st.text(result.get("raw_text") or "")

st.divider()
st.subheader("Score every sample")
st.caption("This clears the ledger first, then runs every file in data/invoices against the same stock snapshot.")
if st.button("Score the sample folder"):
    if use_llm and build_client() is None:
        st.stop()
    with st.spinner("Scoring the folder..."):
        st.session_state["batch"] = run_directory(
            SAMPLES,
            db_path=DB_PATH,
            use_llm=use_llm,
            log_dir=ROOT / "logs",
            reset=True,
        )

batch = st.session_state.get("batch")
if batch:
    summary = summarize(batch)
    counts = summary["counts"]
    columns = st.columns(4)
    columns[0].metric("Paid", f"${summary['paid']:,.2f}")
    columns[1].metric("Approved", counts.get("approve", 0))
    columns[2].metric("Needs review", counts.get("needs_review", 0))
    columns[3].metric("Rejected", counts.get("reject", 0))
    st.dataframe(
        [
            {
                "File": Path(state["invoice_path"]).name,
                "Invoice": (state.get("invoice") or {}).get("invoice_number"),
                "Decision": state.get("decision"),
                "Total": (state.get("invoice") or {}).get("total"),
                "First finding": ((state.get("findings") or [{}])[0]).get("code", ""),
            }
            for state in batch
        ],
        hide_index=True,
        use_container_width=True,
    )
