from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robust_qsvt_se.paper.signed_readout_diagnostic import (
    SUMMARY_COLUMNS,
    _prepare_block,
    coordinate_observable,
    hadamard_test_estimate,
    run_signed_readout_diagnostic,
    signed_pair_observable,
    top_k_signed_direction,
)
from robust_qsvt_se.paper.tqe_revision_support_common import find_forbidden
from robust_qsvt_se.utils.seed import make_rng


def test_exact_signed_value_matches_inner_product() -> None:
    ell = np.array([1.0, -1.0, 0.0, 2.0])
    dx = np.array([0.3, 0.1, -0.7, 0.05])

    true_value, _estimated, _mu = hadamard_test_estimate(ell, dx, shots=10, rng=make_rng(0))

    assert true_value == float(ell @ dx)


def test_hadamard_estimate_is_unbiased_at_high_shots() -> None:
    rng_vectors = make_rng(123)
    ell = rng_vectors.normal(size=8)
    dx = rng_vectors.normal(size=8) * 0.01

    true_value, estimated, _mu = hadamard_test_estimate(ell, dx, shots=200_000, rng=make_rng(42))

    sampling_scale = float(np.linalg.norm(ell) * np.linalg.norm(dx))
    tolerance = 6.0 * sampling_scale / np.sqrt(200_000)
    assert abs(estimated - true_value) <= tolerance


def test_signed_observables_carry_sign() -> None:
    pair = signed_pair_observable(4, 0, 2)
    assert pair.vector[0] == 1.0
    assert pair.vector[2] == -1.0

    direction = top_k_signed_direction(np.array([-0.5, 0.01, 0.4, -0.02]), k=2)
    # Support is the two largest-magnitude coordinates (0 and 2) with their signs.
    assert direction.vector[0] == -1.0
    assert direction.vector[2] == 1.0
    assert direction.vector[1] == 0.0

    coord = coordinate_observable(3, 1)
    assert coord.vector.tolist() == [0.0, 1.0, 0.0]


def test_toy_fallback_is_clearly_labeled() -> None:
    block = _prepare_block(
        {
            "case_name": "definitely_not_a_real_case",
            "case_source": "pypower",
            "base_seed": 3,
            "alpha": 1.0e-4,
        },
        block_size=4,
    )
    assert "toy fallback" in block["record"]["source_type"]
    assert block["record"]["case_source"] == "toy"


def test_run_signed_readout_diagnostic_outputs_and_flags(tmp_path: Path) -> None:
    run = run_signed_readout_diagnostic(
        {
            "output_dir": str(tmp_path),
            "case_name": "ieee14",
            "case_source": "builtin",
            "block_sizes": [4],
            "alpha": 1.0e-4,
            "shots": [100, 100_000],
            "base_seed": 5,
            "save_plots": False,
        }
    )

    summary = run["summary"]
    assert set(SUMMARY_COLUMNS).issubset(summary.columns)
    assert not summary.empty
    assert bool((~summary["full_vector_readout"]).all())
    assert bool((~summary["hardware_execution"]).all())
    assert set(summary["source_type"]) == {"IEEE-derived"}

    # Absolute error must shrink with more shots (Monte-Carlo readout cost).
    mean_lo = summary.loc[summary["shots"] == 100, "absolute_error"].mean()
    mean_hi = summary.loc[summary["shots"] == 100_000, "absolute_error"].mean()
    assert mean_hi < mean_lo

    for name in (
        "signed_readout_summary.csv",
        "signed_readout_summary.md",
        "signed_readout_error_vs_shots.csv",
        "signed_readout_manifest.json",
    ):
        assert (tmp_path / name).is_file()

    summary_text = (tmp_path / "signed_readout_summary.md").read_text(encoding="utf-8")
    assert find_forbidden(summary_text) == []

    manifest = json.loads((tmp_path / "signed_readout_manifest.json").read_text(encoding="utf-8"))
    assert manifest["full_vector_readout"] is False
    assert manifest["hardware_execution"] is False
    assert manifest["selected_signed_observables_only"] is True
