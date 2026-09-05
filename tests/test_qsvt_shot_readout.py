from __future__ import annotations

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.shot_readout_model import (
    build_shot_readout_model,
    required_shots_for_additive_error,
)


def test_shot_readout_outputs_observables_and_standard_errors(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = build_shot_readout_model(
        {
            "output_dir": str(tmp_path / "shot"),
            "matrix_source": "synthetic",
            "shot_levels": [100, 1000],
            "target_epsilon_grid": [0.1, 0.01],
        }
    )
    output_dir = run["output_dir"]
    frame = pd.read_csv(output_dir / "shot_readout_summary.csv")
    estimates = pd.read_csv(output_dir / "observable_estimates.csv")

    assert set(frame["observable_name"]) == {
        "selected_state_component_0",
        "update_vector_norm",
        "residual_norm_proxy",
    }
    assert set(frame["shots"]) == {100, 1000}
    assert np.isfinite(frame["standard_error"]).all()
    assert (frame["standard_error"] >= 0.0).all()
    assert "full state-vector" in (output_dir / "readout_caveats.md").read_text(encoding="utf-8")
    shots_by_epsilon = estimates.groupby("target_epsilon")["required_shots_estimate"].max()
    assert shots_by_epsilon.loc[0.01] > shots_by_epsilon.loc[0.1]
    assert required_shots_for_additive_error(0.01) > required_shots_for_additive_error(0.1)
    assert (output_dir / "shot_readout_summary.json").is_file()
    assert (output_dir / "manifest.json").is_file()
