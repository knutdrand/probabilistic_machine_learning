import dataclasses
from typing import Callable

import jax.random
import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt

import scipy.stats as s_stats
from jax.scipy import stats
from jax.scipy.special import expit, logit
from probabilistic_machine_learning.adaptors.jax_nuts import sample as nuts_sample
from functools import wraps


def diff_encoded_model(diff_sampler, transition, init_state, observation_dist, diff_prob):

    def sampler(T, key=jax.random.PRNGKey(0)):
        observation_key, transition_key = jax.random.split(key)
        state = jnp.array(init_state)
        sample_transition = lambda state, key: transition(state, diff_sampler(state, key))
        state = jax.lax.scan(sample_transition, state, jax.random.split(transition_key, T - 1))[1]
        return observation_dist(state).sample(observation_key) #, state)

    def recontstruct_state(logits_array):
        return jax.lax.scan(transition, jnp.array(init_state), logits_array)[1]

    def log_prob(observed, logits_array, **P):
        state = recontstruct_state(logits_array)
        state_pdf = diff_prob(state, logits_array, P).sum()
        return state_pdf + observation_dist(state, P).log_prob(observed).sum()#, state, P)

    return sampler, lambda observed: (lambda kwargs: log_prob(observed, **kwargs)), recontstruct_state


def loc_scale_warp(dist):
    return lambda loc, scale, *args, **kwargs: loc + scale * dist(*args, **kwargs)


logistic_sample = loc_scale_warp(jax.random.logistic)


def Distribution(jax_dist, log_prob):
    class Dist:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def sample(self, key):
            return jax_dist(key, *self.args, **self.kwargs)

        def log_prob(self, x):
            return log_prob(x, *self.args, **self.kwargs)

    Dist.__name__ = jax_dist.__name__
    return Dist

Poisson = Distribution(jax.random.poisson, stats.poisson.logpmf)


def refactor_model(param_dict):
    beta, lo_gamma, lo_a, lo_mu, logscale, reporting_rate = (param_dict[k] for k in (
    'beta', 'lo_gamma', 'lo_a', 'lo_mu', 'logscale', 'reporting_rate'))

    def get_loc(state, P=param_dict):
        return [P['beta'] + logit(state[..., 2]), P['lo_gamma'], P['lo_a'], P['lo_mu']]

    def diff_sampler(state, key):
        loc = get_loc(state)
        return logistic_sample(jnp.array(loc), jnp.exp(logscale), key)

    def diff_prob(state, logits_array, P):
        loc = get_loc(state, P)
        return sum(stats.logistic.logpdf(column, loc=param, scale=jnp.exp(logscale)).sum()
                   for column, param in zip(logits_array.T, loc))

    def transition(state, logits):
        diffs = state * expit(logits)
        new_state = state - diffs + jnp.roll(diffs, 1)
        return new_state, new_state

    def observation_dist(state, params=param_dict):
        return Poisson(state[..., 2] * param_dict['reporting_rate'])

    def observation_sampler(state, key):
        return jax.random.poisson(key, state[..., 2] * reporting_rate)

    def observation_prob(observed, state, P):
        return stats.poisson.logpmf(observed, state[..., 2] * P.get('reporting_rate', reporting_rate)).sum()



    return diff_encoded_model(diff_sampler, transition, jnp.array([0.9, 0.08, 0.01, 0.01]), observation_dist, diff_prob)



if __name__ == '__main__':
    param_names = ['lo_gamma', 'lo_mu', 'lo_a', 'lo_mu', 'logscale', 'beta']
    real_params = {'beta': 0.3, 'lo_gamma': logit(0.1), 'lo_a': logit(0.1), 'lo_mu': logit(0.05),
                   'logscale': np.log(0.1), 'reporting_rate': 10000}
    sample, log_prob, reconstruct_state = refactor_model(real_params)
    T = 100
    observed = sample(T)
    init_diffs = np.random.normal(0, 1, (T - 1, 4))

    inits = {name: 0.0 for name in param_names}
    init_dict = {'logits_array': init_diffs} | inits
    log_prob(observed)(init_dict)
    samples = nuts_sample(log_prob(observed), jax.random.PRNGKey(0),
                          init_dict, 100, 100)

    plt.plot(observed / real_params['reporting_rate'], label='observed')
    states = reconstruct_state(samples['logits_array'][-1])[:, 2]
    plt.plot(states, label='reconstructed')
    plt.legend()
    plt.show()
