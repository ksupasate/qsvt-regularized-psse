from __future__ import annotations

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.polynomial_method_comparison import (
    compare_polynomial_approximation_methods,
)


def test_polynomial_method_comparison_labels_fallback_methods(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = compare_polynomial_approximation_methods(
        {
            "output_dir": str(tmp_path / "methods"),
            "matrix_source": "synthetic",
            "alpha": [1.0e-2],
            "degree": [15],
            "methods": ["odd_chebyshev_reduced_y", "odd_chebyshev_minimax_lp"],
            "grid_size": 500,
        }
    )
    output_dir = run["output_dir"]
    frame = pd.read_csv(output_dir / "method_comparison_summary.csv")

    assert set(frame["method"]) == {"odd_chebyshev_reduced_y", "odd_chebyshev_minimax_lp"}
    assert set(frame["parity"]) == {"odd"}
    assert np.isfinite(frame["max_pointwise_error"]).all()
    assert (frame["query_count_estimate"] == 31).all()
    assert frame["caveat"].str.contains("not full").any()
    assert (output_dir / "method_pointwise_errors.csv").is_file()
    assert (output_dir / "manifest.json").is_file()
