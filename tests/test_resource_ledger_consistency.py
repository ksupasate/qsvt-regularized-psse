from robust_qsvt_se.qsvt.gate_level_qsvt import qsvt_sequence_operation_counts


def test_synthesized_phase_count_has_consistent_signal_and_phase_counts():
    counts = qsvt_sequence_operation_counts(32)
    assert counts["signal_unitary_calls"] == 31
    assert counts["projector_phase_operations"] == 32


def test_unsynthesized_rows_must_not_have_resource_counts():
    row = {"phase_synthesis_status": "failed", "phase_count": 0, "signal_unitary_calls": 0}
    assert row["phase_synthesis_status"] == "failed"
    assert row["phase_count"] == row["signal_unitary_calls"] == 0
