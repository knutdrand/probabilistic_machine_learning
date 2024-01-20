import dataclasses

import jax
import numpy as np
from jax import numpy as jnp
from jax.scipy import stats as stats
from matplotlib import pyplot as plt

from probabilistic_machine_learning.adaptors.jax_nuts import sample


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


@dataclasses.dataclass
class CountTS:
    population_size = 1_000_000
    seasons = 13
    double_bite_rate: float = 0.1
    n_mosquitos: np.ndarray = ((np.arange(100) % seasons)-seasons//2)*100

    def sample(self):
        true_states = [0.2]
        for i in range(self.T - 1):
            double_bites = self.double_bite_rate*self.n_mosquitos[i]

            true_states.append(np.random.normal(true_states[-1] + self.beta, 1))
        true_states = jnp.array(true_states)
        observed = np.random.poisson(jnp.exp(true_states))

        return observed


@dataclasses.dataclass
class RealTS:
    population_size = 1_000_000
    seasons = 13
    beta: float = 2.0
    extragenous: np.ndarray = ((np.arange(100) % seasons) - seasons // 2) / 2
    sigma = 0.1
    ab_sum = 5
    eta = 0.5
    @property
    def T(self):
        return len(self.extragenous)+1

    def sample(self):
        true_states = [0.2]
        for i in range(self.T - 1):
            p = true_states[-1]
            mu = self.eta*p*(1-p)
            a = mu*self.ab_sum
            b = (1-mu)*self.ab_sum
            diff = np.random.beta(a, b)*(1-p)
            true_states.append(p+diff)
            # true_states.append(
            #     np.random.normal(true_states[-1] + self.b*self.extragenous[i], self.sigma))
        true_states = jnp.array(true_states)

        rates = np.diff(true_states) * self.population_size
        observed = np.random.poisson(rates)
        # observed = np.random.poisson(jnp.exp(true_states))
        plt.plot(observed,'.', label='observed')
        plt.plot(rates)
        plt.show()
        return observed

    def estimate(self, observed):
        def logdensity_fn(beta, states):
            """Univariate Normal"""
            eta = jax.scipy.special.expit(beta)
            ps = jax.scipy.special.expit(states)
            diffs = jnp.diff(ps)
            mu = eta * ps[:-1]*(1-ps[:-1])
            state_logpdf = stats.beta.logpdf(diffs/(1-ps[:-1]),
                                             mu * self.ab_sum,
                                             (1 - mu) * self.ab_sum)
            #state_logpdf = stats.norm.logpdf(
            #    states[1:], states[:-1] + beta*self.extragenous, self.sigma)
            observed_logpdf = stats.poisson.logpmf(observed, diffs*self.population_size)
            return jnp.sum(state_logpdf) + jnp.sum(observed_logpdf)

        logdensity = lambda x: logdensity_fn(**x)
        rng_key = jax.random.key(1000)

        return sample(logdensity, rng_key, {'beta': 0.1, 'states': np.cumsum(np.ones(self.T)*0.01)}, num_samples=1000)
