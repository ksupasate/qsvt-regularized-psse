def test_phase_failure_schema_forbids_counts_without_phases():
    failure = {
        "phase_synthesis_status": "failed",
        "minimum_phase_synthesizable_degree": None,
        "signal_unitary_calls": None,
        "failure_reason": "boundedness_failure",
    }
    assert failure["failure_reason"]
    assert failure["signal_unitary_calls"] is None
