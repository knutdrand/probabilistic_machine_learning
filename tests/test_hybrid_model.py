from numbers import Number

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose
import jax

from probabilistic_machine_learning.cases.diff_model import MosquitoModelSpec, multilogit, inverse_multilogit, DiffModel

from probabilistic_machine_learning.cases.hybrid_model import HybridModel, decoupled_scan, discrepancy_logprob


def simple_transition(state, diff):
    new_state = state + diff
    return new_state, new_state


@pytest.fixture
def simple_diffs():
    diffs = np.arange(10)[:, None]
    assert diffs.shape == (10, 1)
    states = np.array([[10], [100]])
    return diffs, states


@pytest.fixture
def model_spec():
    return MosquitoModelSpec(MosquitoModelSpec.good_params)


def test_decoupled_scan(simple_diffs):
    scanned = decoupled_scan(*simple_diffs, simple_transition)
    truth = [10, 11, 13, 16, 20, 105, 111, 118, 126, 135]
    assert_array_equal(scanned, np.array(truth)[:, None])


@pytest.fixture
def simple_diffs_2d():
    diffs = np.array([np.arange(4), np.zeros(4)]).T
    assert diffs.shape == (4, 2)
    states = np.array([[10, 1], [100, 10.]])
    return diffs, states


def test_2d_scan(simple_diffs_2d):
    scanned = decoupled_scan(*simple_diffs_2d, simple_transition)
    truth = [[10, 1], [11, 1], [102, 10], [105, 10]]
    assert_array_equal(scanned, np.array(truth))


@pytest.fixture
def scan_data(model_spec):
    transition = model_spec.transition
    state = model_spec.init_state
    n_states = len(state)
    diffs = np.ones((6, n_states))
    states = np.array([state, state])
    return diffs, states, transition


def test_simple_scan(scan_data):
    diffs, states, transition = scan_data
    res = jax.lax.scan(transition, states[0], diffs)[1]
    assert res.shape == (6, len(states[0]))


def test_real_transition(scan_data):
    scanned = decoupled_scan(*scan_data)  # , states, transition)
    diffs, states, transition = scan_data
    simple_scanned = jax.lax.scan(transition, states[0], diffs)[1]
    assert_allclose(scanned[:3], simple_scanned[:3], rtol=1e-5)
    assert_allclose(scanned[3:], simple_scanned[:3], rtol=1e-5)


def test_real_transition(scan_data):
    diffs, states, transition = scan_data
    diffs = np.random.normal(size=diffs.size).reshape(diffs.shape)
    scanned = decoupled_scan(diffs, states, transition)
    simple_scanned = jax.lax.scan(transition, states[0], diffs)[1]
    assert_allclose(scanned[:3], simple_scanned[:3], rtol=1e-5)

@pytest.fixture
def discrepancy_data():
    calc_states = np.array([[10, 1, 3],
                            [2., 10., 1.],
                            [11, 1, 3],
                            [3., 10., 1.]])
    new_states = np.array([[2., 10., 1.], [3., 10., 1.]])
    return calc_states, new_states


def test_discrepancy_logprob(discrepancy_data):
    calc_states, new_states = discrepancy_data
    inverse_transform = lambda x: x
    res = discrepancy_logprob(calc_states, new_states, new_states[0], inverse_transform)
    assert res.shape == ()


def test_multilogit_inverse():
    state = np.array([0.3, 0.2, 0.5])
    t_state = multilogit(state)
    new_state = inverse_multilogit(t_state)
    assert_allclose(new_state, state, rtol=1e-5)


def test_state_tranform(model_spec):
    state = model_spec.init_state
    t_state = model_spec.state_transform(state)
    new_state = model_spec.inverse_state_transform(t_state)
    assert_allclose(new_state, state, rtol=1e-5)

def test_2dstate_tranform(model_spec):
    state = np.array([model_spec.init_state]*2)
    t_state = model_spec.state_transform(state)
    new_state = model_spec.inverse_state_transform(t_state)
    assert_allclose(new_state, state, rtol=1e-5)


@pytest.fixture
def logprob_data(model_spec):
    P = MosquitoModelSpec.good_params
    T = 6
    P['logits_array'] = np.random.normal(size=(T * 10)).reshape((T, 10))
    P['transformed_states'] = np.random.normal(size=(2 * 9)).reshape((2, 9))
    observed = np.random.normal(size=T)
    exogenous = np.random.normal(size=T)
    return observed, P, exogenous, model_spec

def test_diff_logprob_acceptance(logprob_data):
    observed, P, exogenous, model_spec = logprob_data
    model = DiffModel(MosquitoModelSpec)
    res = model.log_prob(observed, P, exogenous)

def test_hybrid_logprob_acceptance(logprob_data):
    observed, P, exogenous, model_spec = logprob_data
    model = HybridModel(MosquitoModelSpec)
    res = model.log_prob(observed, P, exogenous)
