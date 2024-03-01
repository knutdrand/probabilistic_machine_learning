import jax
import numpy as np
import jax.numpy as jnp
from .diff_model import DiffModel


def discrepancy_logprob(calculated_states, t_states, init_state, inverse_transform):
    '''Get the discrepancy between a set of states calculuated from a set of diffs and the true states (transformed)'''
    n_fixed, n_states = t_states.shape
    T_per_fixed = len(calculated_states) // n_fixed
    reshaped = calculated_states.reshape(n_fixed, T_per_fixed, -1)
    calculated_end_states = reshaped[:, -1, :]
    t_end_states = inverse_transform(calculated_end_states)
    init_pdf = jax.scipy.stats.norm.logpdf(inverse_transform(init_state), t_states[0], 0.1).sum()
    rest_pdf = jax.scipy.stats.norm.logpdf(t_end_states[:-1], t_states[1:], 0.1).sum()
    return rest_pdf + init_pdf


def decoupled_scan(diffs_array: np.ndarray, state_array: np.ndarray, transition: callable):
    '''
    Performs a scan where acutal values are inserted at fixed time intervals
    This allows for decoupling the gradients a bit
    diffs_array: (T, n_states)
    state_array: (n_fixed, n_states)
    returns: (T, n_states)
    '''

    n_fixed, n_states = state_array.shape
    assert diffs_array.shape[-1] == n_states, (diffs_array.shape[0], n_fixed, n_states)
    assert diffs_array.shape[0] % n_fixed == 0, (diffs_array.shape[0], n_fixed, n_states)
    diffs_array = diffs_array.reshape(n_fixed, -1, n_states)
    diffs_array = jnp.swapaxes(diffs_array, 0, 1)
    val = jax.lax.scan(transition, state_array, diffs_array)[1]
    val = jnp.swapaxes(val, 0, 1).reshape(-1, n_states)
    return val


class HybridModel(DiffModel):

    def recontstruct_state(self, diffs_array, state_array, params=None):
        new_transition = self.spec_class(params).transition
        return decoupled_scan(diffs_array, state_array, new_transition)

    def log_prob(self, observed, P, exogenous=None):
        spec = self.spec_class(P)
        diffs = P['logits_array']
        t_states = P['transformed_states']
        fixed_states = spec.inverse_state_transform(t_states)
        state = self.recontstruct_state(diffs, fixed_states, P)
        d_dist = spec.diff_distribution(state[:-1], exogenous[1:])
        state_pdf = d_dist.log_prob(diffs[1:]).sum()
        init_pdf = spec.diff_distribution(self.spec_class.init_state, exogenous[0]).log_prob(diffs[0]).sum()
        observed_pdf = spec.observation_distribution(state).log_prob(observed)
        discrepancy_pdf = discrepancy_logprob(state, t_states, self.spec_class.init_state, spec.state_transform)
        return state_pdf + observed_pdf.sum() + init_pdf + discrepancy_pdf
