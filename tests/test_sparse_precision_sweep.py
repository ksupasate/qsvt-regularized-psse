"""Matrix-stage, phase-precision, design-fairness, and resume/checkpoint tests."""

from __future__ import annotations

import inspect
import json
import math

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("pennylane")
pytest.importorskip("pypower")

import robust_qsvt_se.qsvt.sparse_error_precision_study as study
from robust_qsvt_se.qsvt.sparse_block_encoding_wrapper import quantize_sign_magnitude
from robust_qsvt_se.qsvt.sparse_error_precision_study import (
    EXACT_VALUE_KEY,
    FINITE_SHOT_CONFIGURATIONS,
    FULL_PHASE_KEY,
    PHASE_BITS_SWEEP,
    VALUE_BITS_SWEEP,
    StudyCheckpoint,
    build_frozen_design,
    build_point_config,
    load_study_configuration,
    make_context,
    quantize_phase_sequence,
    stage_statevector,
)
from robust_qsvt_se.qsvt.sparse_integrated_chain import stable_array_fingerprint


@pytest.fixture(scope="module")
def design(tmp_path_factory):
    return build_frozen_design(tmp_path_factory.mktemp("design"))


# ------------------------------- matrix-stage tests -----------------------------------


def test_exact_sparse_support_equals_baseline_support(design):
    baseline_support = design.matrices_by_value_key["6"] != 0.0
    exact_support = design.matrix_sparse_exact != 0.0
    assert np.array_equal(exact_support, baseline_support)
    assert int(exact_support.sum()) == 16


def test_exact_sparse_matrix_uses_original_unquantized_values(design):
    support = design.matrix_sparse_exact != 0.0
    assert np.array_equal(
        design.matrix_sparse_exact[support], design.matrix_original[support]
    )
    assert np.all(design.matrix_sparse_exact[~support] == 0.0)


def test_quantized_matrices_use_declared_quantizer_and_are_deterministic(design):
    for bits in VALUE_BITS_SWEEP:
        reference, mu = quantize_sign_magnitude(
            design.matrix_sparse_exact, magnitude_bits=bits
        )
        assert np.array_equal(design.matrices_by_value_key[str(bits)], reference)
        again, mu_again = quantize_sign_magnitude(
            design.matrix_sparse_exact, magnitude_bits=bits
        )
        assert np.array_equal(reference, again)
        assert mu == mu_again == design.mu
        assert stable_array_fingerprint(reference) == stable_array_fingerprint(again)


def test_six_bit_matrix_is_not_overwritten_by_the_sweep(design):
    stored = np.load("outputs/sparse_integrated_chain/matrix_quantized.npy")
    assert np.array_equal(stored, design.matrices_by_value_key["6"])


# ------------------------------ design-fairness tests ---------------------------------


def test_physical_alpha_and_design_are_frozen_across_points(design, tmp_path):
    for sampled in FINITE_SHOT_CONFIGURATIONS:
        config = build_point_config(
            design, sampled["value_bits"], sampled["phase_bits"], tmp_path
        )
        assert config.alpha == design.alpha
        assert config.beta == design.beta
        assert config.normalized_lambda == design.normalized_lambda
        assert config.contraction_c == design.contraction_c
        assert config.polynomial_degree == design.degree


def test_no_per_matrix_phase_refitting_in_primary_sweep():
    source = inspect.getsource(study)
    assert "synthesize_pennylane_phases" not in source
    assert "fit_codesigned_bounded_polynomial" not in source


def test_all_sweep_matrices_satisfy_frozen_normalization_bound(design):
    for key, matrix in design.matrices_by_value_key.items():
        spectral = float(np.linalg.svd(matrix, compute_uv=False).max())
        assert spectral <= design.beta + 1.0e-9, key


def test_declared_configuration_file_matches_module_predeclaration():
    declared = load_study_configuration()
    assert declared["finite_shot_configurations"] == [
        dict(item) for item in FINITE_SHOT_CONFIGURATIONS
    ]
    assert declared["value_bits"] == list(VALUE_BITS_SWEEP)
    assert declared["phase_bits"] == list(PHASE_BITS_SWEEP)
    assert declared["design_mode"] == "frozen_design"


# ------------------------------ phase-precision tests ----------------------------------


def test_phase_rounding_is_deterministic_order_and_length_preserving(design):
    full = design.phases_by_phase_key[FULL_PHASE_KEY]
    for bits in PHASE_BITS_SWEEP:
        rounded = quantize_phase_sequence(full, str(bits))
        assert rounded.size == full.size == design.degree + 1
        step = 2.0 * math.pi / (1 << bits)
        assert np.max(np.abs(rounded - full)) <= step / 2.0 + 1e-15
        codes = rounded / step
        assert np.allclose(codes, np.round(codes), atol=1e-9)
        assert np.array_equal(rounded, quantize_phase_sequence(full, str(bits)))
        # rounding an already-rounded sequence is idempotent (no refitting happened)
        assert np.array_equal(rounded, quantize_phase_sequence(rounded, str(bits)))


def test_full_phase_key_returns_frozen_sequence_unchanged(design):
    full = design.phases_by_phase_key[FULL_PHASE_KEY]
    assert np.array_equal(quantize_phase_sequence(full, FULL_PHASE_KEY), full)


def test_unsupported_or_invalid_phase_precision_fails_explicitly(design):
    full = design.phases_by_phase_key[FULL_PHASE_KEY]
    with pytest.raises(ValueError, match="unsupported_precision"):
        quantize_phase_sequence(full, "0")
    with pytest.raises(ValueError, match="unsupported_precision"):
        quantize_phase_sequence(full, "63")
    with pytest.raises(ValueError, match="phase_quantization_invalid"):
        quantize_phase_sequence(np.array([0.1, 3.5]), "8")


def test_exact_value_point_rejected_for_measured_circuit_configs(design, tmp_path):
    with pytest.raises(ValueError, match="quantized value points"):
        build_point_config(design, EXACT_VALUE_KEY, FULL_PHASE_KEY, tmp_path)


# ------------------------------ resume/checkpoint tests ---------------------------------


def test_checkpoint_parts_are_atomic_and_verified(tmp_path):
    checkpoint = StudyCheckpoint(tmp_path)
    checkpoint.write_part("statevector", "bv6_bpfull", {"value": 1})
    assert checkpoint.load_part("statevector", "bv6_bpfull") == {"value": 1}
    assert not list(tmp_path.rglob("*.tmp.*"))
    # torn/interrupted part files are treated as incomplete and rerun
    torn = tmp_path / "checkpoint_parts" / "statevector" / "torn.json"
    torn.write_text('{"status": "completed", "key": "torn', encoding="utf-8")
    assert checkpoint.load_part("statevector", "torn") is None
    in_progress = tmp_path / "checkpoint_parts" / "statevector" / "bv4_bp8.json"
    in_progress.write_text(
        json.dumps({"status": "in_progress", "key": "bv4_bp8", "payload": {}}),
        encoding="utf-8",
    )
    assert checkpoint.load_part("statevector", "bv4_bp8") is None
    # key mismatch is never trusted
    wrong = tmp_path / "checkpoint_parts" / "statevector" / "bv12_bp16.json"
    wrong.write_text(
        json.dumps({"status": "completed", "key": "other", "payload": {}}),
        encoding="utf-8",
    )
    assert checkpoint.load_part("statevector", "bv12_bp16") is None
    checkpoint.clear_stage("statevector")
    assert checkpoint.load_part("statevector", "bv6_bpfull") is None


def test_statevector_stage_resume_skips_only_completed_units(design, tmp_path, monkeypatch):
    context = make_context(output_dir=tmp_path, resume=True)
    context._design = design
    keys = [("6", FULL_PHASE_KEY), ("6", "8")]
    monkeypatch.setattr(study, "_grid_keys", lambda: keys)
    calls: list[tuple[str, str]] = []
    original = study.evaluate_statevector_point

    def counting(design_arg, value_key, phase_key, wrapper=None):
        calls.append((value_key, phase_key))
        return original(design_arg, value_key, phase_key, wrapper)

    monkeypatch.setattr(study, "evaluate_statevector_point", counting)
    first = stage_statevector(context)
    assert first["newly_computed"] == 2
    assert len(calls) == 2
    second = stage_statevector(context)
    assert second["newly_computed"] == 0
    assert len(calls) == 2  # completed units were skipped, nothing recomputed
