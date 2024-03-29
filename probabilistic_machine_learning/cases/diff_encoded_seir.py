# Include time as covariate in the model

from collections import defaultdict

import pandas as pd
import plotly.express as px

import jax.random
import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt
import scipy.stats as s_stats
from jax.scipy import stats
from jax.scipy.special import expit, logit
from probabilistic_machine_learning.adaptors.jax_nuts import sample as nuts_sample
from functools import wraps

def plot_logistic():
    beta = -3

    for i in (0.01, 0.05, 0.1, 0.5):
        x = np.linspace(-10, 10, 100)
        y = expit(stats.logistic.logpdf(x, logit(i) + beta, 0.1))
        plt.plot(expit(x), y, label=f'p={i}')
    plt.legend()
    plt.show()


def plot_beta():
    x = np.linspace(0, 1, 100)
    for p in [0.01, 0.05, 0.1]:
        for s in (1, 5, 10, 100):
            y = stats.beta.pdf(x, p * s, (1 - p) * s)
            plt.plot(x, y, label=f'p={p}, s={s}')
        plt.legend()
        plt.show()


def plot_sigmoid():
    I = np.linspace(0, 1, 100)
    for beta in [0.1, 1, 10]:
        y = expit(beta * I)
        plt.plot(I, y, label=f'beta={beta}')
    plt.legend()
    plt.show()


def scan_transition(state, logits):
    diffs = state * expit(logits)
    new_state = state - diffs + jnp.roll(diffs, 1)
    return new_state, new_state


all_values = defaultdict(list)

class Register:
    def __init__(self):
        self.results = defaultdict(list)

    def save(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            self.results[func.__name__].append(result)
            return result

        return wrapper

    def __getitem__(self, item):
        self.results[item] = np.asanyarray(self.results[item])
        return self.results[item]


logger = Register()


def loc_scale_warp(dist):
    return lambda loc, scale, *args, **kwargs: loc+scale*dist(*args, **kwargs)

logisitc_sample = loc_scale_warp(jax.random.logistic)




def model(beta, gamma, a, mu, scale, reporting_rate):
    init_state = [0.9, 0.08, 0.01, 0.01]
    lo_gamma, lo_a, lo_mu = (logit(p) for p in (gamma, a, mu))

    @logger.save
    def _sample_logits(state):
        loc = get_loc(state[2], beta, lo_a, lo_gamma, lo_mu)
        rvs = s_stats.logistic(loc=loc, scale=scale).rvs()
        return rvs

    def sample_diff(state, key):
        loc = get_loc(state[2], beta, lo_a, lo_gamma, lo_mu)
        return logisitc_sample(jnp.array(loc), scale, key)

    @logger.save
    def _transition(state):
        logits = sample_logits(state)
        return scan_transition(state, logits)[0]

    def scan_transition(state, logits):
        diffs = state * expit(logits)
        new_state = state - diffs + jnp.roll(diffs, 1)
        return new_state, new_state[2]

    def sample(T, key = jax.random.PRNGKey(0)):
        state = jnp.array(init_state)
        sample_transition = lambda state, key: scan_transition(state, sample_diff(state, key))
        I = jax.lax.scan(sample_transition, state, jax.random.split(key, T-1))[1]
        return s_stats.poisson(I * reporting_rate).rvs()

    def log_prob(observed, logits_array, lo_gamma, lo_a=lo_a, lo_mu=np.log(mu), beta=beta, logscale=np.log(scale)):
        prior_pdf = sum(stats.norm.logpdf(param, 0, 10) for param in (lo_gamma, lo_a, lo_mu, beta, logscale))
        I = recontstruct_state(logits_array)
        # I = jax.lax.scan(scan_transition, jnp.array(init_state), logits_array)[1].T
        loc = get_loc(I, beta, lo_a, lo_gamma, lo_mu)
        state_pdf = sum(stats.logistic.logpdf(column, loc=param, scale=jnp.exp(logscale)).sum()
                        for column, param in zip(logits_array.T, loc))

        return state_pdf + stats.poisson.logpmf(observed, I * reporting_rate).sum()+prior_pdf

    def get_loc(I, beta, lo_a, lo_gamma, lo_mu):
        return [beta + logit(I), lo_gamma, lo_a, lo_mu]

    def recontstruct_state(logits_array):
        return jax.lax.scan(scan_transition, jnp.array(init_state), logits_array)[1]


    return sample, lambda observed: (lambda kwargs: log_prob(observed, **kwargs)), recontstruct_state


def main():
    param_names = ['lo_gamma', 'lo_mu', 'lo_a', 'lo_mu', 'logscale', 'beta']
    real_params = {'beta': 0.3, 'lo_gamma': logit(0.1), 'lo_a': logit(0.1), 'lo_mu': logit(0.05), 'logscale': np.log(0.1)}
    reporting_rate = 10000
    sample, log_prob, reconstruct_state = model(0.3, 0.1, 0.1, 0.05, 0.1, reporting_rate)
    T = 1000
    observed = sample(T)
    init_diffs = np.random.normal(0, 1, (T - 1, 4))

    inits = {name: 0.0 for name in param_names}
    inits['beta'] = 0.3
    samples = nuts_sample(log_prob(observed), jax.random.PRNGKey(0),
                          {'logits_array': init_diffs} | inits, 100, 100)

    plt.plot(observed/reporting_rate, label='observed')
    states = reconstruct_state(samples['logits_array'][-1])
    plt.plot(states, label='reconstructed')
    plt.legend()
    plt.show()
    return
    for i in range(4):
        plt.plot(states[:, i], label=f'state {i}')
        plt.plot(logger['transition'][:, i], label=f'transition {i}')
        plt.legend()
        plt.show()

    for i in range(4):
        plt.plot(np.mean(samples['logits_array'], axis=0)[:, i], label=f'logit {i}')
        plt.plot(logger['sample_logits'][:, i], label=f'sample_logits {i}')
        plt.legend()
        plt.show()

    for param_name in param_names:
        mean = np.mean(samples[param_name])
        real = real_params[param_name]
        print(param_name, mean, real)


if __name__ == '__main__':
    result = main()
