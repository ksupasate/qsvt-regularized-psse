from __future__ import annotations

import numpy as np
import pytest

from robust_qsvt_se.qsvt.rectangular_convention import (
    DENSE_JULIA_PCPHASE,
    PYQSP_SYM_QSP_PLUS_I,
    convert_pyqsp_sym_qsp_to_pcphase,
)


def test_rectangular_convention_api_guards():
    conversion = convert_pyqsp_sym_qsp_to_pcphase(np.zeros(2), degree=1)
    assert conversion.source_convention == PYQSP_SYM_QSP_PLUS_I
    assert conversion.target_convention == DENSE_JULIA_PCPHASE
    assert conversion.extraction_component == "neg_imag"
    with pytest.raises(ValueError):
        convert_pyqsp_sym_qsp_to_pcphase(np.zeros(3), degree=2)
    with pytest.raises(ValueError):
        convert_pyqsp_sym_qsp_to_pcphase(np.zeros(3), degree=1)
    with pytest.raises(ValueError):
        convert_pyqsp_sym_qsp_to_pcphase(np.zeros(2), degree=1, already_converted=True)
