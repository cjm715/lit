from sol_checking import _sequence_converges_to_zero, _criterion_report_msg
import numpy as np
from tools import ScalarTool
from termcolor import colored


def test_convergent_sequence():
    a = list(2.**(-1.0 * np.linspace(1., 100., 100.)))
    assert _sequence_converges_to_zero(a)


def test_divergent_sequence():
    a = list(2.**(1.0 * np.linspace(1., 100., 100.)))
    assert (not _sequence_converges_to_zero(a))


def test_near_zero_sequence():
    a = list(np.random.random(100) * 10. ** (-16))
    assert _sequence_converges_to_zero(a)


def test_criterion_report_msg_passed():
    assert colored('PASSED', 'green') == _criterion_report_msg(True)


def test_criterion_report_msg_failed():
    assert colored('FAILED', 'red') == _criterion_report_msg(False)
