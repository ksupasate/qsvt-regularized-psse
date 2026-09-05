"""Gap-resolution: run the smoke-free scientific-validation suite.

Selects only the scientific-validation tests (real_validation, claim_boundary_guard,
integration_regression, artifact_schema) from the AST test-quality audit and runs them.
Smoke-only and unit-behavior tests are excluded, so the suite contains zero smoke-only
tests by construction. Use ``--dry-run`` to list the selection, ``--collect-only`` to
verify it collects, or run with no flag to execute it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.test_quality_audit import (  # noqa: E402
    SCIENTIFIC_CATEGORIES,
    build_test_quality_audit,
    scientific_validation_rows,
    smoke_rows,
)

_PYTHON = sys.executable


def build_selection(test_root: str = "tests") -> dict[str, Any]:
    """Audit the tests and return the smoke-free scientific-validation selection."""

    run = build_test_quality_audit({"test_root": test_root, "output_dir": "/tmp/_sci_suite_audit"})
    rows = run["inventory_rows"]
    scientific = scientific_validation_rows(rows)
    node_ids = [r["node_id"] for r in scientific]
    smoke_in_selection = [r for r in scientific if r["test_category"] == "smoke_only"]
    return {
        "node_ids": node_ids,
        "scientific_count": len(scientific),
        "smoke_only_in_selection": len(smoke_in_selection),
        "engineering_smoke_count": len(smoke_rows(rows)),
        "total": len(rows),
        "categories": sorted(SCIENTIFIC_CATEGORIES),
    }


def run_pytest(node_ids: list[str], *, collect_only: bool) -> dict[str, Any]:
    extra = ["--collect-only"] if collect_only else []
    command = [_PYTHON, "-m", "pytest", "-q", *extra, *node_ids]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    tail = (completed.stdout or "").strip().splitlines()[-1:] or [""]
    return {
        "returncode": completed.returncode,
        "status": "passed" if completed.returncode == 0 else "failed",
        "summary_line": tail[0],
    }


def write_outputs(
    output_dir: Path, selection: dict[str, Any], result: dict[str, Any] | None, *, mode: str
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "scientific_validation_suite_report.json"
    report_path.write_text(
        json.dumps(
            {
                "mode": mode,
                "selection": {k: v for k, v in selection.items() if k != "node_ids"},
                "result": result,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    selection_path = output_dir / "scientific_validation_suite_selection.csv"
    lines = ["node_id"] + selection["node_ids"]
    selection_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "scientific_validation_suite_report": str(report_path),
        "scientific_validation_suite_selection": str(selection_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the smoke-free scientific-validation suite.")
    parser.add_argument("--test-root", default="tests")
    parser.add_argument(
        "--output-dir",
        default="outputs/final_manuscript_package/test_quality_appendix",
    )
    parser.add_argument("--dry-run", action="store_true", help="list the selection without running")
    parser.add_argument("--collect-only", action="store_true", help="verify the selection collects")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = build_selection(args.test_root)
    if selection["smoke_only_in_selection"] != 0:
        raise SystemExit("scientific-validation selection unexpectedly contains smoke-only tests")
    print(
        f"scientific_validation_tests={selection['scientific_count']} "
        f"smoke_only_in_selection={selection['smoke_only_in_selection']} "
        f"engineering_smoke={selection['engineering_smoke_count']}"
    )
    result: dict[str, Any] | None = None
    mode = "dry-run"
    if not args.dry_run:
        mode = "collect-only" if args.collect_only else "run"
        result = run_pytest(selection["node_ids"], collect_only=args.collect_only)
        print(f"pytest {mode} status={result['status']} :: {result['summary_line']}")
    artifacts = write_outputs(Path(args.output_dir), selection, result, mode=mode)
    print(f"wrote {artifacts['scientific_validation_suite_selection']}")
    if result is not None and result["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
