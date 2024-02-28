import numpy as np
import pytest
from numpy.testing import assert_array_equal

from probabilistic_machine_learning.cases.hybrid_model import HybridModel, decoupled_scan

@pytest.fixture
def simple_diffs():
    diffs = np.arange(10)[:, None]
    assert diffs.shape == (10, 1)
    states = np.array([[10], [100]])
    return diffs, states


def test_decoupled_scan(simple_diffs):
    scanned = decoupled_scan(*simple_diffs)
    truth = [10, 11, 13, 16, 20, 105, 111, 118, 126, 135]
    assert_array_equal(scanned, np.array(truth)[:, None])

@pytest.fixture
def simple_diffs_2d():
    diffs = np.array([np.arange(4), np.zeros(4)]).T
    assert diffs.shape == (4, 2)
    states = np.array([[10, 1], [100, 10.]])
    return diffs, states

def test_2d_scan(simple_diffs_2d):
    scanned = decoupled_scan(*simple_diffs_2d)
    truth = [[10, 1], [11, 1], [102, 10], [105, 10]]
    assert_array_equal(scanned, np.array(truth))

