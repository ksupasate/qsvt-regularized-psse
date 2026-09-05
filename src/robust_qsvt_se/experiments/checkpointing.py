from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from robust_qsvt_se.utils.io import ensure_directory, write_json


def load_trial_records(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "trial_results.jsonl"
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL checkpoint at line {line_number}: {path}") from exc
            if not isinstance(record, dict) or "trial_id" not in record:
                raise ValueError(f"invalid trial checkpoint record at line {line_number}: {path}")
            records.append(record)
    return records


def completed_trial_ids(records: list[dict[str, Any]]) -> set[str]:
    return {
        str(record["trial_id"])
        for record in records
        if str(record.get("status")) in {"completed", "failed"}
    }


def append_trial_record(output_dir: Path, record: dict[str, Any]) -> None:
    ensure_directory(output_dir)
    record = dict(record)
    record.setdefault("recorded_at", _now_iso())
    with (output_dir / "trial_results.jsonl").open("a", encoding="utf-8") as file:
        json.dump(record, file, sort_keys=True, allow_nan=True)
        file.write("\n")


def write_checkpoint_state(
    output_dir: Path,
    *,
    status: str,
    total_trials: int,
    completed_trials: int,
    failed_trials: int,
    skipped_trials: int,
    started_at: float,
    last_trial_id: str | None = None,
    message: str | None = None,
) -> None:
    elapsed = max(0.0, perf_counter() - started_at)
    processed = max(1, completed_trials + failed_trials)
    average = elapsed / processed if completed_trials or failed_trials else None
    remaining = max(0, total_trials - completed_trials - failed_trials)
    state = {
        "status": status,
        "updated_at": _now_iso(),
        "total_trials": int(total_trials),
        "completed_trials": int(completed_trials),
        "failed_trials": int(failed_trials),
        "skipped_trials": int(skipped_trials),
        "remaining_trials": int(remaining),
        "elapsed_seconds": float(elapsed),
        "average_trial_seconds": average,
        "estimated_remaining_seconds": None if average is None else float(average * remaining),
        "last_trial_id": last_trial_id,
        "message": message,
    }
    write_json(output_dir / "checkpoint_state.json", state)


def append_progress(
    output_dir: Path,
    *,
    trial_id: str,
    status: str,
    completed_trials: int,
    failed_trials: int,
    total_trials: int,
    elapsed_seconds: float,
    average_trial_seconds: float | None,
    estimated_remaining_seconds: float | None,
    message: str | None = None,
) -> None:
    ensure_directory(output_dir)
    row = {
        "timestamp": _now_iso(),
        "trial_id": trial_id,
        "status": status,
        "completed_trials": completed_trials,
        "failed_trials": failed_trials,
        "total_trials": total_trials,
        "elapsed_seconds": elapsed_seconds,
        "average_trial_seconds": average_trial_seconds,
        "estimated_remaining_seconds": estimated_remaining_seconds,
        "message": message,
    }
    with (output_dir / "progress.log").open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, sort_keys=True, allow_nan=True))
        file.write("\n")


def rows_from_records(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        record_rows = record.get(key, [])
        if isinstance(record_rows, list):
            rows.extend(row for row in record_rows if isinstance(row, dict))
    return rows


def trial_payloads_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for record in records:
        payload = record.get("trial_payload")
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def dataframe_from_records(records: list[dict[str, Any]], key: str) -> pd.DataFrame:
    return pd.DataFrame(rows_from_records(records, key))


def clear_checkpoint_files(output_dir: Path) -> None:
    for filename in ("trial_results.jsonl", "checkpoint_state.json", "progress.log"):
        path = output_dir / filename
        if path.exists():
            path.unlink()


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()
