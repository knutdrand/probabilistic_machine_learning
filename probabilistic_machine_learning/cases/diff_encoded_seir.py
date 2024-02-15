# Figure out beta parameters
# Figure out bSI tempering
# Write sampler
# Write logprob with diff-proportions as states
# Figure out scan
# Figure out why beta is not converging
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


def model(beta, gamma, a, mu, scale, reporting_rate):
    init_state = [0.9, 0.08, 0.01, 0.01]
    lo_gamma, lo_a, lo_mu = (logit(p) for p in (gamma, a, mu))

    @logger.save
    def sample_logits(state):
        logits = jnp.array([beta + logit(state[2]), lo_gamma, lo_a, lo_mu])
        rvs = s_stats.logistic(loc=logits, scale=scale).rvs()
        return rvs

    @logger.save
    def transition(state):
        logits = sample_logits(state)
        return apply_diffs(logits, state)

    def apply_diffs(logits, state):
        diffs = state*expit(logits)
        return state - diffs + jnp.roll(diffs, 1)

    def sample(T):
        state = jnp.array(init_state)
        I = []
        for t in range(T - 1):
            state = transition(state)
            I.append(state[2])
        return s_stats.poisson(np.array(I) * reporting_rate).rvs()

    def log_prob(observed, logits_array, lo_gamma, lo_a=lo_a, lo_mu=np.log(mu), beta=beta, logscale=np.log(scale)):
        scale = jnp.exp(logscale)
        S, E, I, R = jax.lax.scan(scan_transition, jnp.array(init_state), logits_array)[1].T
        loc = [beta + logit(I)[:-1], lo_gamma, lo_a, lo_mu]
        state_pdf = sum(stats.logistic.logpdf(column[1:], loc=param, scale=scale).sum()
                        for column, param in zip(logits_array.T, loc))

        return state_pdf + stats.poisson.logpmf(observed, I * reporting_rate).sum()

    def recontstruct_state(logits_array):
        return jax.lax.scan(scan_transition, jnp.array(init_state), logits_array)[1]


    return sample, lambda observed: (lambda kwargs: log_prob(observed, **kwargs)), recontstruct_state


def main():
    param_names = ['lo_gamma']
    real_params = {'beta': 0.3, 'lo_gamma': logit(0.1), 'lo_a': logit(0.1), 'lo_mu': logit(0.05), 'logscale': np.log(0.1)}
    reporting_rate = 10000
    sample, log_prob, reconstruct_state = model(0.3, 0.1, 0.1, 0.05, 0.1, reporting_rate)
    T = 1000
    observed = sample(T)
    init_diffs = np.random.normal(0, 1, (T - 1, 4))

    inits = {name: 0.0 for name in param_names}
    inits['beta'] = 0.3
    samples = nuts_sample(log_prob(observed), jax.random.PRNGKey(0),
                          {'logits_array': init_diffs} | inits, 1000, 1000)
    plt.plot(observed/reporting_rate, label='observed')
    states = reconstruct_state(samples['logits_array'][-1])
    plt.plot(states[:, 2], label='reconstructed')
    plt.legend()
    plt.show()
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
