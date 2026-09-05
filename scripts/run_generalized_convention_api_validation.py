"""Safe convention-conversion API validation (Work Package E).

Exercises every error path and the success path of
``robust_qsvt_se.generalized.convention_api`` and records the outcome. Produces
``production_convention_api_validation.csv``. The same cases are asserted as
hard failures in ``tests/test_convention_conversion_api.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.generalized.convention_api import (
    ConversionError,
    ConversionRequest,
    convert_pyqsp_to_production,
    make_request_from_phases,
    predict_extraction,
)
from robust_qsvt_se.qsvt.sym_qsp_circuit_action import synthesize_pyqsp_sym_qsp_phases

OUT = Path(__file__).resolve().parents[1] / "outputs" / "generalized_rectangular_qsvt"


def _case(name, fn, expect_reject: bool) -> dict:
    rejected = False
    msg = ""
    result_meta = ""
    try:
        res = fn()
        if res is not None:
            result_meta = (
                f"mapping={res.phase_mapping}; ordering={res.phase_ordering}; "
                f"offset={res.applied_offset:.6f}; comp={res.extraction_component}; "
                f"sign={res.extraction_sign}; checksum={res.conversion_checksum[:12]}"
            )
    except ConversionError as exc:
        rejected = True
        msg = str(exc)[:120]
    ok = rejected == expect_reject
    return {
        "case": name,
        "expect_reject": expect_reject,
        "rejected": rejected,
        "outcome": "PASS" if ok else "FAIL",
        "message": msg,
        "result_metadata": result_meta,
    }


def main() -> int:
    # build a valid degree-3 phase set once
    from numpy.polynomial import Chebyshev

    p1 = Chebyshev([0, 1], domain=[-1, 1])
    pn = Chebyshev([1], domain=[-1, 1])
    for _ in range(3):
        pn = pn * p1
    phases = synthesize_pyqsp_sym_qsp_phases(np.asarray(pn.coef, float))

    rows = []

    def good_request(**overrides):
        kw = dict(
            source_convention="pyqsp_sym_qsp_plus_i",
            target_convention="dense_julia_pcphase",
            degree=3,
            phases=phases,
            expected_phase_count=4,
            extraction_component="imag",
            extraction_sign=1,
            configuration_id="api_val::d3",
        )
        kw.update(overrides)
        return ConversionRequest(**kw)

    # success path
    rows.append(
        _case(
            "success_degree_3",
            lambda: convert_pyqsp_to_production(good_request()),
            expect_reject=False,
        )
    )
    # success via convenience builder (predicts component automatically)
    rows.append(
        _case(
            "success_make_request_helper",
            lambda: convert_pyqsp_to_production(
                make_request_from_phases(phases, degree=3, configuration_id="x")
            ),
            expect_reject=False,
        )
    )
    # unknown source convention
    rows.append(
        _case(
            "unknown_source",
            lambda: convert_pyqsp_to_production(good_request(source_convention="bogus")),
            expect_reject=True,
        )
    )
    # unknown target convention
    rows.append(
        _case(
            "unknown_target",
            lambda: convert_pyqsp_to_production(good_request(target_convention="bogus")),
            expect_reject=True,
        )
    )
    # incorrect phase count (degree 3 needs 4 phases; give 5)
    rows.append(
        _case(
            "wrong_phase_count",
            lambda: convert_pyqsp_to_production(
                good_request(phases=np.zeros(5), expected_phase_count=5)
            ),
            expect_reject=True,
        )
    )
    # inconsistent degree vs phase count
    rows.append(
        _case(
            "inconsistent_degree",
            lambda: convert_pyqsp_to_production(
                good_request(
                    degree=5,
                    expected_phase_count=6,
                    phases=np.zeros(6),
                    extraction_component="imag",
                    extraction_sign=1,
                )
            ),
            expect_reject=True,
        )
    )
    # expected_phase_count disagrees with degree
    rows.append(
        _case(
            "expected_count_mismatch",
            lambda: convert_pyqsp_to_production(good_request(expected_phase_count=99)),
            expect_reject=True,
        )
    )
    # double conversion
    rows.append(
        _case(
            "double_conversion",
            lambda: convert_pyqsp_to_production(good_request(already_converted=True)),
            expect_reject=True,
        )
    )
    # ambiguous extraction (wrong component for degree 3 -> should be imag)
    rows.append(
        _case(
            "ambiguous_extraction_component",
            lambda: convert_pyqsp_to_production(good_request(extraction_component="neg_imag")),
            expect_reject=True,
        )
    )
    # wrong sign
    rows.append(
        _case(
            "wrong_extraction_sign",
            lambda: convert_pyqsp_to_production(good_request(extraction_sign=-1)),
            expect_reject=True,
        )
    )
    # unsupported parity: even degree via convenience builder
    rows.append(
        _case(
            "even_degree_unsupported",
            lambda: make_request_from_phases(np.zeros(5), degree=4, configuration_id="even"),
            expect_reject=True,
        )
    )
    # missing configuration id
    rows.append(
        _case(
            "missing_config_id",
            lambda: convert_pyqsp_to_production(good_request(configuration_id="")),
            expect_reject=True,
        )
    )
    # non-finite phases
    rows.append(
        _case(
            "non_finite_phases",
            lambda: convert_pyqsp_to_production(good_request(phases=np.array([np.nan, 0, 0, 0]))),
            expect_reject=True,
        )
    )
    # predict_extraction consistency for several degrees
    for d, exp in [
        (1, ("neg_imag", -1)),
        (3, ("imag", 1)),
        (5, ("neg_imag", -1)),
        (7, ("imag", 1)),
        (255, ("imag", 1)),
    ]:
        got = predict_extraction(d)
        rows.append(
            {
                "case": f"predict_extraction_d{d}",
                "expect_reject": False,
                "rejected": False,
                "outcome": "PASS" if got == exp else "FAIL",
                "message": f"got {got} expected {exp}",
                "result_meta": "",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "production_convention_api_validation.csv", index=False)
    fails = (df["outcome"] == "FAIL").sum()
    print(f"[WP-E] API validation: {len(df)} cases, {fails} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
