import pytest

from robust_qsvt_se.paper.tqe_revision_core import RegisterLedger


def test_qubit_total_equals_register_sum():
    ledger = RegisterLedger("full", 0, 7, 1, 0, 0, 1, 0, 128, 256, 9, "EXECUTED")
    assert ledger.validated()["register_sum"] == 9


def test_inconsistent_qubit_total_fails():
    ledger = RegisterLedger("bad", 0, 7, 1, 0, 0, 1, 0, 128, 256, 8, "EXECUTED")
    with pytest.raises(ValueError, match="register sum"):
        ledger.validated()
