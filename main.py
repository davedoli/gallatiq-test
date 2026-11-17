#!/usr/bin/env python3
"""Run the Acme invoice desk on one file or a folder of samples."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from invoice_agent.llm import load_env
from invoice_agent.report import full_report, table
from invoice_agent.runner import run_directory, run_path


def main(argv: list[str] | None = None) -> int:
    load_env()
    parser = argparse.ArgumentParser(description="Process an invoice through extract, validate, approve, and pay.")
    parser.add_argument("--invoice_path", help="Path to one invoice file")
    parser.add_argument("--invoice_dir", help="Path to a folder of invoice files")
    parser.add_argument("--db", default="inventory.db", help="SQLite ledger path")
    parser.add_argument("--no-llm", action="store_true", help="Use the local parsers and rules only")
    parser.add_argument(
        "--keep-ledger",
        action="store_true",
        help="When scoring a folder, keep earlier payments. A folder run otherwise starts from an empty ledger.",
    )
    parser.add_argument("--log-dir", default="logs", help="Where JSON audit logs are written")
    parser.add_argument("--verbose", action="store_true", help="Print the full trace for every file in a folder")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if args.verbose:
        logging.getLogger("invoice_agent").setLevel(logging.INFO)

    if not args.invoice_path and not args.invoice_dir:
        parser.error("Pass --invoice_path or --invoice_dir")

    log_dir = Path(args.log_dir) if args.log_dir else None
    use_llm = not args.no_llm

    if args.invoice_path:
        state = run_path(args.invoice_path, db_path=args.db, use_llm=use_llm, log_dir=log_dir)
        print(full_report(state))
        if state.get("log_path"):
            print(f"Log: {state['log_path']}")
        return 0

    if not args.keep_ledger:
        print("Ledger cleared so this batch starts from an empty payment history.")
    states = run_directory(
        args.invoice_dir,
        db_path=args.db,
        use_llm=use_llm,
        log_dir=log_dir,
        reset=not args.keep_ledger,
    )
    if args.verbose:
        for state in states:
            print(full_report(state))
            print()
    print(table(states))
    return 0


if __name__ == "__main__":
    sys.exit(main())
