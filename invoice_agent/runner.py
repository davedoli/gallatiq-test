from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from invoice_agent.db import connect, reset_runtime
from invoice_agent.graph import GRAPH
from invoice_agent.llm import build_client, get_llm_client, set_llm_client

SUPPORTED = {".pdf", ".txt", ".json", ".csv", ".xml"}


def run_path(
    path: str | Path,
    *,
    db_path: str,
    use_llm: bool = False,
    log_dir: Path | None = None,
) -> dict:
    path = Path(path)
    if use_llm and get_llm_client() is None:
        client = build_client()
        if client is None:
            use_llm = False
        else:
            set_llm_client(client)
    state = GRAPH.invoke(
        {
            "invoice_path": str(path),
            "raw_text": "",
            "source_format": "",
            "invoice": None,
            "extraction_attempts": 0,
            "extraction_issues": [],
            "extraction_source": "",
            "retry_extraction": False,
            "findings": [],
            "decision": "",
            "decision_reason": "",
            "critique_notes": "",
            "validation_narrative": "",
            "payment": None,
            "trace": [],
            "use_llm": use_llm,
            "db_path": str(db_path),
        },
        config={"recursion_limit": 20},
    )
    if log_dir is not None:
        state["log_path"] = str(write_log(state, log_dir))
    return state


def run_directory(
    directory: str | Path,
    *,
    db_path: str,
    use_llm: bool = False,
    log_dir: Path | None = None,
    reset: bool = False,
) -> list[dict]:
    if reset:
        with connect(db_path) as conn:
            reset_runtime(conn)
    files = sorted(path for path in Path(directory).iterdir() if path.suffix.lower() in SUPPORTED)
    return [run_path(path, db_path=db_path, use_llm=use_llm, log_dir=log_dir) for path in files]


def write_log(state: dict, log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    stem = Path(state["invoice_path"]).stem
    destination = log_dir / f"{stem}-{stamp}.json"
    payload = {
        "invoice_path": state.get("invoice_path"),
        "decision": state.get("decision"),
        "decision_reason": state.get("decision_reason"),
        "critique_notes": state.get("critique_notes"),
        "findings": state.get("findings"),
        "invoice": state.get("invoice"),
        "payment": state.get("payment"),
        "trace": state.get("trace"),
        "validation_narrative": state.get("validation_narrative"),
        "extraction_attempts": state.get("extraction_attempts"),
        "extraction_source": state.get("extraction_source"),
        "raw_text": state.get("raw_text"),
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination
