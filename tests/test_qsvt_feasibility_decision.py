from __future__ import annotations

from robust_qsvt_se.qsvt.run_full_matrix_qsvt import validate_full_matrix_qsvt_config


def test_full_matrix_qsvt_config_validation_accepts_ieee300_resource_path() -> None:
    resolved = validate_full_matrix_qsvt_config(
        {
            "run": {
                "run_id": "qsvt_full_matrix_ieee300",
                "output_dir": "outputs/qsvt_full_matrix_ieee300",
                "max_pennylane_full_qubits": 8,
                "max_qiskit_dense_full_qubits": 8,
            },
            "matrix": {"case_name": "ieee300"},
            "demo": {"polynomial_degree": 5},
        }
    )

    assert resolved["matrix"]["case_name"] == "ieee300"
    assert resolved["run"]["max_pennylane_full_qubits"] == 8
