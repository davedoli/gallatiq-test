# Acme invoice desk

A local multi-agent desk that reads a supplier invoice, checks it against inventory, and either records a mock payment or stops with a written reason.

Clean invoices under $10,000 are paid on a local ledger. Stock problems, bad math, unknown items, and anything that looks like fraud are rejected. Price changes, non-USD invoices, totals over $10,000 with no other defect, and a second copy of an invoice already on the ledger are held for a person.

The original brief is in [CASE.md](CASE.md).

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py --invoice_path=data/invoices/invoice_1001.txt --no-llm
.venv/bin/python main.py --invoice_dir=data/invoices --no-llm
.venv/bin/streamlit run app.py
```

`--no-llm` uses the local parsers and rules. Leave it off when `XAI_API_KEY` is set in `.env` (see `.env.example`) and the same graph will call Grok for extraction, an inventory tool pass, and a critique. The rules still decide. A model can hold a payment. It cannot approve an invoice the rules rejected, and it cannot call the payment function.

Python 3.9 and newer. `eval_type_backport` is what lets Pydantic read the modern type hints on 3.9.

## What one invoice does

LangGraph runs nine steps. Each step keeps the fields it does not touch.

1. **Load** the file (PDF, text, JSON, CSV, or XML).
2. **Extract** structured fields. Text and PDF can go through the model; JSON, CSV, and XML stay on the parser.
3. **Check** the arithmetic. With a model key, a bad total is sent back for one more read.
4. **Normalize** printed names (`Widget A`, `Gadget X`) onto catalog ids.
5. **Validate** against SQLite. The model may call `lookup_inventory`. The findings themselves are computed in code.
6. **Draft** a decision from those findings.
7. **Critique** by recomputing the findings, then asking the model if a clean invoice should be held.
8. **Pay** only on approve, through the mock ledger. A second success for the same invoice number is refused.
9. **Record** the decision whether or not money moved.

JSON audit logs go to `logs/`.

## Sample folder

`python main.py --invoice_dir=data/invoices --no-llm` clears the ledger and scores every file against the same stock snapshot (WidgetA 15, WidgetB 10, GadgetX 5, FakeItem 0).

| Outcome | Files |
|---|---|
| Paid | 1001, 1004, 1006, 1011 PDF, 1012 PDF, 1015. **$29,115** total. |
| Rejected | 1002 stock, 1003 fraud and zero stock, 1005 and 1007 and 1013 over stock (1013 and 1007 also fail the arithmetic), 1008 and 1016 unknown items, 1009 negative quantity |
| Held | 1004 revision and the text twins of 1011 and 1012 (already on the ledger), 1010 rush price, 1014 euros |

## Tests

```bash
.venv/bin/python -m pytest
```
