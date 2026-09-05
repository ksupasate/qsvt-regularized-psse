from __future__ import annotations

import json

import pandas as pd

from robust_qsvt_se.qsvt.engineering_audit import (
    audit_artifacts,
    build_engineering_audit,
    classify_forbidden_context,
)


def test_engineering_audit_checks_files_columns_and_manifests(tmp_path) -> None:  # type: ignore[no-untyped-def]
    output_dir = tmp_path / "outputs" / "demo"
    output_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "encoded_block_error": 0.0,
                "unitarity_error": 0.0,
                "passed": True,
            }
        ]
    ).to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "command": "test",
                "generated_at": "2026-06-08T00:00:00Z",
                "git_commit": None,
                "input_config": {},
                "artifacts": {},
            }
        ),
        encoding="utf-8",
    )
    specs = [
        {
            "group": "demo",
            "path": "outputs/demo/summary.csv",
            "required_columns": ["encoded_block_error", "unitarity_error", "passed"],
            "max_columns": {"encoded_block_error": 1.0e-8, "unitarity_error": 1.0e-8},
            "true_columns": ["passed"],
        },
        {"group": "demo", "path": "outputs/demo/manifest.json"},
    ]

    rows = audit_artifacts(tmp_path, specs)
    assert all(row["status"] == "pass" for row in rows)

    run = build_engineering_audit(
        {
            "root_dir": str(tmp_path),
            "output_dir": str(tmp_path / "audit"),
            "artifact_specs": specs,
            "docs_paths": [],
        }
    )
    assert (run["output_dir"] / "audit_summary.md").is_file()
    assert (run["output_dir"] / "audit_results.json").is_file()
    assert (run["output_dir"] / "manifest.json").is_file()


def test_forbidden_wording_classifier_distinguishes_safe_and_unsafe_contexts() -> None:
    safe = "Do not claim quantum speedup in this feasibility report."
    unsafe = "The implementation demonstrates quantum speedup on the benchmark."

    assert classify_forbidden_context(safe, "quantum speedup") == "safe_context"
    assert classify_forbidden_context(unsafe, "quantum speedup") == "unsafe_context"
