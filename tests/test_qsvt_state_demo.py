from __future__ import annotations

import json

import pandas as pd

from robust_qsvt_se.qsvt.state_demo import run_state_demo


def test_state_demo_exact_qsvt_target_matches_ridge(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = run_state_demo(
        {
            "output_dir": str(tmp_path / "state"),
            "matrix_source": "synthetic",
            "alpha": 1.0e-2,
        }
    )
    output_dir = run["output_dir"]
    summary = pd.read_csv(output_dir / "state_demo_summary.csv")
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert float(summary.loc[0, "relative_error_vs_ridge"]) <= 1.0e-8
    assert float(summary.loc[0, "cosine_similarity_vs_ridge"]) >= 1.0 - 1.0e-8
    assert bool(summary.loc[0, "passed_equivalence_check"]) is True
    assert (output_dir / "state_demo_summary.json").is_file()
    assert (output_dir / "singular_values.csv").is_file()
    assert (output_dir / "filter_values.csv").is_file()
    assert {"command", "generated_at", "git_commit", "input_config"}.issubset(manifest)
