from __future__ import annotations

import inspect

import numpy as np
import pytest

from robust_qsvt_se.physical_alignment import risk_selectors
from robust_qsvt_se.physical_alignment.risk_selectors import (
    evaluate_risk,
    exact_single_removal_scores,
    noise_propagation_risk,
    posterior_variance_reference,
    refine_risk_support_one_swap,
    select_risk_support,
)
from robust_qsvt_se.qsvt.output_aware_sparse_selection import (
    SupportConstraints,
    support_constraint_report,
)


def _matrix() -> np.ndarray:
    return np.array(
        [
            [2.0, 0.3, 0.1, 0.2],
            [0.4, 1.7, 0.2, 0.1],
            [0.2, 0.1, 1.4, 0.5],
            [0.3, 0.2, 0.4, 1.8],
        ]
    )


def _functionals(dimension: int = 4) -> list[np.ndarray]:
    coordinate = np.eye(dimension)[0]
    aggregate = np.ones(dimension) / np.sqrt(dimension)
    return [coordinate, aggregate]


@pytest.mark.parametrize("risk_kind", ["noise_propagation", "posterior_variance_reference"])
def test_solve_based_risk_matches_explicit_inverse_reference(risk_kind: str) -> None:
    matrix = _matrix()
    alpha = 0.4
    functional = _functionals()[1]
    gram = matrix.T @ matrix
    inverse = np.linalg.inv(gram + alpha * np.eye(matrix.shape[1]))
    if risk_kind == "noise_propagation":
        expected = float(functional @ inverse @ gram @ inverse @ functional)
        observed = noise_propagation_risk(matrix, alpha, functional)
    else:
        expected = float(functional @ inverse @ functional)
        observed = posterior_variance_reference(matrix, alpha, functional)
    np.testing.assert_allclose(observed, expected, rtol=1.0e-12, atol=1.0e-13)
    assert observed >= -1.0e-13


@pytest.mark.parametrize("risk_kind", ["noise_propagation", "posterior_variance_reference"])
def test_matrix_and_alpha_unit_scaling(risk_kind: str) -> None:
    matrix = _matrix()
    alpha = 0.3
    scale = 7.0
    base = evaluate_risk(
        matrix, alpha, _functionals(), risk_kind=risk_kind, aggregation="mean"
    ).objective
    scaled = evaluate_risk(
        scale * matrix,
        scale**2 * alpha,
        _functionals(),
        risk_kind=risk_kind,
        aggregation="mean",
    ).objective
    np.testing.assert_allclose(scaled, base / scale**2, rtol=1.0e-11, atol=1.0e-13)


@pytest.mark.parametrize("shape", [(7, 4), (4, 7)])
@pytest.mark.parametrize("risk_kind", ["noise_propagation", "posterior_variance_reference"])
def test_rank_deficient_tall_and_wide_systems_remain_valid(
    shape: tuple[int, int], risk_kind: str
) -> None:
    rng = np.random.default_rng(18)
    matrix = rng.normal(size=shape)
    matrix[:, -1] = matrix[:, 0]
    functional = np.ones(shape[1]) / np.sqrt(shape[1])
    value = evaluate_risk(
        matrix, 0.2, [functional], risk_kind=risk_kind, aggregation="worst_case"
    ).objective
    assert np.isfinite(value)
    assert value >= 0.0


def test_exact_removal_full_support_reference_and_determinism() -> None:
    first = exact_single_removal_scores(
        _matrix(),
        0.2,
        _functionals(),
        risk_kind="noise_propagation",
        aggregation="mean",
    )
    second = exact_single_removal_scores(
        _matrix(),
        0.2,
        _functionals(),
        risk_kind="noise_propagation",
        aggregation="mean",
    )
    expected = evaluate_risk(
        _matrix(),
        0.2,
        _functionals(),
        risk_kind="noise_propagation",
        aggregation="mean",
    ).objective
    assert first.full_support_objective == pytest.approx(expected)
    np.testing.assert_array_equal(first.raw_scores, second.raw_scores)
    np.testing.assert_array_equal(first.milp_scores, second.milp_scores)


def test_selection_is_deterministic_feasible_and_refinement_monotone() -> None:
    matrix = _matrix()
    constraints = SupportConstraints(k_budget=8, slot_budget=2, coverage_enabled=True)
    kwargs = {
        "risk_kind": "posterior_variance_reference",
        "aggregation": "mean",
        "refine": True,
        "max_refinement_iterations": 2,
        "improvement_tolerance": 1.0e-12,
    }
    first = select_risk_support(matrix, 0.5, _functionals(), constraints, **kwargs)
    second = select_risk_support(matrix, 0.5, _functionals(), constraints, **kwargs)
    assert first.status == second.status == "completed"
    np.testing.assert_array_equal(first.support, second.support)
    assert support_constraint_report(matrix, first.support, constraints)["valid"]
    assert first.final_objective <= first.initial_objective + 1.0e-12
    assert first.accepted_swaps >= 0


def test_direct_refinement_never_worsens_its_frozen_objective() -> None:
    matrix = _matrix()
    constraints = SupportConstraints(k_budget=8, slot_budget=2, coverage_enabled=True)
    scores = exact_single_removal_scores(
        matrix,
        0.5,
        _functionals(),
        risk_kind="noise_propagation",
        aggregation="worst_case",
    )
    initial = select_risk_support(
        matrix,
        0.5,
        _functionals(),
        constraints,
        risk_kind="noise_propagation",
        aggregation="worst_case",
        refine=False,
        max_refinement_iterations=0,
        improvement_tolerance=1.0e-12,
    )
    assert scores.exact_solves == np.count_nonzero(matrix) + 1
    refined = refine_risk_support_one_swap(
        matrix,
        initial.support,
        0.5,
        _functionals(),
        constraints,
        risk_kind="noise_propagation",
        aggregation="worst_case",
        max_iterations=2,
        improvement_tolerance=1.0e-12,
    )
    assert refined.final_objective <= refined.initial_objective + 1.0e-12


def test_public_selector_api_and_source_have_no_truth_or_heldout_access() -> None:
    signature = inspect.signature(select_risk_support)
    forbidden = {"x_true", "y_true", "truth", "held_out", "test_seed", "residual"}
    assert forbidden.isdisjoint(signature.parameters)
    source = inspect.getsource(risk_selectors)
    assert "np.linalg.inv" not in source
    assert "numpy.linalg.inv" not in source


def test_unavailable_or_nonunit_functionals_are_not_substituted() -> None:
    with pytest.raises(ValueError, match="at least one physical functional"):
        evaluate_risk(
            _matrix(),
            0.2,
            [],
            risk_kind="noise_propagation",
            aggregation="mean",
        )
    with pytest.raises(ValueError, match="not unit norm"):
        posterior_variance_reference(_matrix(), 0.2, np.ones(4))
