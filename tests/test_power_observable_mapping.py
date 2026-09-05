from __future__ import annotations

from pathlib import Path

import numpy as np

from robust_qsvt_se.qsvt.power_observable_mapping import (
    build_power_observables,
    run_power_observable_readout,
)


def test_power_observable_mapping_does_not_invent_metadata() -> None:
    H = np.eye(2)
    observables = build_power_observables(H, {"state_index_mapping": []})

    generic = [
        row for row in observables if row.metadata_status == "metadata_unavailable_or_generic"
    ]

    assert generic
    assert all("bus_" not in row.observable_name for row in generic)


def test_power_observable_readout_writes_outputs(tmp_path: Path) -> None:
    run = run_power_observable_readout(
        {
            "output_dir": str(tmp_path),
            "case": "ieee14",
            "model": "ac_linearized",
            "submatrix_size": 4,
            "alpha": 1.0e-4,
            "degree": 9,
            "shots": [100],
            "seed": 123,
        }
    )

    assert len(run["exact_rows"]) >= 3
    for name in [
        "manifest",
        "observable_metadata",
        "observable_exact_vs_qsvt",
        "observable_shot_scaling",
        "power_observable_interpretation",
    ]:
        assert run["artifacts"][name].is_file()
