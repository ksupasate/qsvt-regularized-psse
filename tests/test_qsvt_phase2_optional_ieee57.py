from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.qsvt import phase2_completion


def test_optional_ieee57_records_run_or_skip(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def fake_sweep(config: dict) -> dict:
        frame = pd.DataFrame(
            [
                {
                    "case_name": "ieee57",
                    "variant_name": "original_ridge",
                    "alpha": 0.01,
                    "residual_norm": 1.0,
                    "qsvt_full_interval_approx_error": 1.0e-4,
                }
            ]
        )
        return {"results": frame}

    monkeypatch.setattr(phase2_completion, "run_phase2_preconditioned_alpha_sweeps", fake_sweep)

    run = phase2_completion.run_phase2_optional_ieee57(
        {"output_dir": str(tmp_path / "ieee57"), "reuse_existing": False}
    )
    output_dir = Path(run["output_dir"])

    assert run["status"]["status"] in {"completed", "skipped", "completed_from_existing_results"}
    assert (output_dir / "ieee57_phase2_status.json").is_file()
    assert (output_dir / "ieee57_phase2_status.md").is_file()
    assert (output_dir / "ieee57_phase2_manifest.json").is_file()
    if run["status"]["status"] == "completed":
        assert (output_dir / "ieee57_phase2_results.csv").is_file()
