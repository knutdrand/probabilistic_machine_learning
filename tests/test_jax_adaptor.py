import dataclasses

from probabilistic_machine_learning.adaptors.jax_adaptor import JaxWrap
from probabilistic_machine_learning.adaptors.jax_nuts import sample
from .fixtures import *
from .models import Models
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

import blackjax

_wrap = JaxWrap()

models = Models(_wrap)

def test_simple_jax():
    loc, scale = 10., 20.
    observed = np.random.normal(loc, scale, size=1_000)

    def logdensity_fn(loc, log_scale, observed=observed):
        """Univariate Normal"""
        scale = jnp.exp(log_scale)
        logpdf = stats.norm.logpdf(observed, loc, scale)
        return jnp.sum(logpdf)

    logdensity = lambda x: logdensity_fn(**x)
    rng_key = jax.random.key(1000)
    samples = sample(logdensity, rng_key, {'loc': 0.0, 'log_scale': 0.0}, num_samples=1_000)


@dataclasses.dataclass
class BasicTS:
    T: int
    beta: float = 2.

    def sample(self):
        true_states = [0.2]
        for i in range(self.T - 1):
            true_states.append(np.random.normal(true_states[-1] + self.beta, 1))
        observed = np.random.normal(true_states, 2)

        return observed

    def estimate(self, observed):
        def logdensity_fn(beta, states):
            """Univariate Normal"""
            state_logpdf = stats.norm.logpdf(states[1:], states[:-1] + beta, 1)
            observed_logpdf = stats.norm.logpdf(observed, states, 2)
            return jnp.sum(state_logpdf) + jnp.sum(observed_logpdf)

        logdensity = lambda x: logdensity_fn(**x)
        rng_key = jax.random.key(1000)

        return sample(logdensity, rng_key, {'beta': 0.0, 'states': np.zeros(self.T)}, num_samples=1_000)


@dataclasses.dataclass
class PoissonTS:
    T: int
    beta: float = 0.1

    def sample(self):
        true_states = [0.2]
        for i in range(self.T - 1):
            true_states.append(np.random.normal(true_states[-1] + self.beta, 1))
        true_states = jnp.array(true_states)
        observed = np.random.poisson(jnp.exp(true_states))

        return observed

    def estimate(self, observed):
        def logdensity_fn(beta, states):
            """Univariate Normal"""
            state_logpdf = stats.norm.logpdf(states[1:], states[:-1] + beta, 1)
            observed_logpdf = stats.poisson.logpmf(observed, jnp.exp(states))
            return jnp.sum(state_logpdf) + jnp.sum(observed_logpdf)

        logdensity = lambda x: logdensity_fn(**x)
        rng_key = jax.random.key(1000)

        return sample(logdensity, rng_key, {'beta': 0.0, 'states': np.zeros(self.T)}, num_samples=1_000)



def test_time_series_jax():
    model = PoissonTS(100)
    observed = model.sample()
    samples = model.estimate(observed)
    plt.hist(samples['beta']); plt.show()

def test_get_jax_model(numpy_data_1, example_1):
    x, y = numpy_data_1
    jax_model = models.linear_regression(x, y)
    # assert set(jax_model.named_vars.keys()) == set(example_1.named_vars.keys())


