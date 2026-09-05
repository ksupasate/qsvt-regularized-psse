import hashlib
import math
from pathlib import Path

import pandas as pd
import pytest

from robust_qsvt_se.paper.phase3_resource_reproducibility import (
    OUTPUT_FILES,
    direct_sampling_counts,
    extrapolated_shots,
)

OUTPUT_DIR = Path("outputs/phase3_resource_reproducibility")
TABLES_DIR = Path("manuscript/tables")


def test_direct_sampling_counts_arithmetic() -> None:
    counts = direct_sampling_counts(q=1, shots=100000, p_succ=0.8532761303549481, degree=25)
    assert counts.signal_unitary_calls_per_attempt == 25
    assert counts.phase_operations_per_attempt == 26
    assert counts.alternating_sequence_length_per_attempt == 51
    expected_attempts = 100000 / 0.8532761303549481
    assert math.isclose(counts.expected_attempts, expected_attempts, rel_tol=1e-12)
    assert math.isclose(counts.prep_repetitions, expected_attempts, rel_tol=1e-12)
    assert math.isclose(counts.readout_repetitions, expected_attempts, rel_tol=1e-12)
    assert math.isclose(counts.signal_unitary_calls, 25 * expected_attempts, rel_tol=1e-12)
    with pytest.raises(ValueError):
        direct_sampling_counts(q=1, shots=100000, p_succ=0.0, degree=25)


def test_extrapolated_shots_matches_manuscript_convention() -> None:
    assert math.isclose(extrapolated_shots(1e5, 0.154, 0.01), 2.3716e7, rel_tol=1e-12)




def test_classical_timings_match_recorded_phase2_values() -> None:
    timings = pd.read_csv(OUTPUT_DIR / "classical_selected_output_timings.csv")
    adjoint = timings.loc[timings["method"] == "dense_adjoint_selected_output"]
    assert len(adjoint) == 4
    assert (adjoint["abs_diff_vs_recorded_phase2"].astype(float) < 1e-11).all()
    assert (adjoint["median_seconds"].astype(float) > 0).all()




def test_checksums_file_covers_all_sibling_artifacts() -> None:
    entries = {}
    for line in (OUTPUT_DIR / "checksums.sha256").read_text().strip().splitlines():
        digest, name = line.split(None, 1)
        entries[name.strip()] = digest
    for name in OUTPUT_FILES:
        if name == "checksums.sha256":
            continue
        assert name in entries, name
        actual = hashlib.sha256((OUTPUT_DIR / name).read_bytes()).hexdigest()
        assert entries[name] == actual, name


