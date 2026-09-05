"""Normalized/propagated sparse value-oracle quantization error report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from robust_qsvt_se.paper.sparse_quantization_error_report import (
    REPORT_COLUMNS,
    build_sparse_quantization_error_report,
)
from robust_qsvt_se.paper.tqe_revision_experiments_common import forbidden_in

# Published in Table VIII (revision_sparse_oracle_validation.tex): the fixed six-bit
# quantization step gives this exact absolute error for both blocks. Recomputing it
# independently here is a self-consistency check, not a retune.
_PUBLISHED_MAX_ABS_ERROR = 10.258115828770599


@pytest.fixture(scope="module")
def report_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    output_dir = tmp_path_factory.mktemp("sparse_quantization_error_report")
    return build_sparse_quantization_error_report({"output_dir": str(output_dir)})


def test_required_artifacts_exist(report_run: dict[str, Any]) -> None:
    output_dir = Path(report_run["output_dir"])
    for name in ["quantization_error_report.csv", "summary.md", "manifest.json"]:
        assert (output_dir / name).is_file(), f"missing required artifact {name}"


def test_report_columns_and_blocks(report_run: dict[str, Any]) -> None:
    frame = report_run["report"]
    assert list(frame.columns) == REPORT_COLUMNS
    assert set(frame["matrix_shape"]) == {"4x4", "8x8"}
    assert len(frame) == 2
    assert not report_run["failures"]


def test_absolute_error_matches_published_table_vii(report_run: dict[str, Any]) -> None:
    """Independently recomputed quantization must reproduce the existing table number."""

    frame = report_run["report"]
    np.testing.assert_allclose(
        frame["max_absolute_quantization_error"].to_numpy(),
        _PUBLISHED_MAX_ABS_ERROR,
        rtol=1.0e-9,
    )


def test_normalized_errors_are_finite_and_bounded(report_run: dict[str, Any]) -> None:
    frame = report_run["report"]
    for column in [
        "relative_frobenius_error",
        "relative_spectral_error",
        "relative_selected_output_error",
        "update_relative_error_l2",
    ]:
        values = frame[column].to_numpy(dtype=float)
        assert np.all(np.isfinite(values)), f"{column} contains non-finite values"
        assert np.all(values > 0.0), f"{column} should be strictly positive for a lossy encoding"
        # A six-bit quantization of a well-conditioned 4x4/8x8 block should not blow up
        # the selected output; a large value here would indicate a bug, not a feature.
        assert np.all(values < 1.0), f"{column} unexpectedly exceeds 100% relative error"


def test_selected_output_matches_matched_alpha_ridge_formula(report_run: dict[str, Any]) -> None:
    """y_l = e_1^T (H^T H + alpha I)^-1 H^T r must match the paper's eq. selected_functional."""

    frame = report_run["report"]
    assert (frame["alpha_used"] == 1.0e-4).all()
    assert (frame["selected_output_index"] == 0).all()
    delta = frame["quantized_selected_output"] - frame["true_selected_output"]
    np.testing.assert_allclose(
        delta.abs().to_numpy(),
        frame["selected_output_error_abs_delta_y"].to_numpy(),
        rtol=1.0e-9,
    )


def test_generated_text_respects_claim_boundary(report_run: dict[str, Any]) -> None:
    output_dir = Path(report_run["output_dir"])
    for path in [output_dir / "summary.md", output_dir / "manifest.json"]:
        violations = forbidden_in(path.read_text(encoding="utf-8"))
        assert not violations, f"{path.name} contains forbidden wording: {violations}"


def test_manifest_declares_no_fabrication(report_run: dict[str, Any]) -> None:
    import json

    manifest = json.loads(Path(report_run["artifacts"]["manifest"]).read_text(encoding="utf-8"))
    assert manifest["fabricates_results"] is False
    assert manifest["changes_estimator_behavior"] is False
    assert manifest["overwrites_existing_outputs"] is False


def _latex_sci_token(value: float) -> str:
    """Render ``value`` the same way it was hand-typed into Table VIII (2 sig. figs)."""

    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa}\\!\\times\\!10^{{{int(exponent)}}}"


def test_manuscript_table_values_match_report(report_run: dict[str, Any]) -> None:
    """The hand-curated Table VIII entries must match this regenerable report."""

    table_path = Path("manuscript/tables/revision_sparse_oracle_validation.tex")
    if not table_path.is_file():
        pytest.skip("manuscript table not present in this checkout")
    text = table_path.read_text(encoding="utf-8")
    frame: pd.DataFrame = report_run["report"]
    for _, row in frame.iterrows():
        for column in [
            "relative_frobenius_error",
            "relative_spectral_error",
            "relative_selected_output_error",
        ]:
            token = _latex_sci_token(float(row[column]))
            assert token in text, f"{column}={token} ({row['block']}) not found in Table VIII"
