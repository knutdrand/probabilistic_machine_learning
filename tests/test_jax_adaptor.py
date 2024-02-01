import dataclasses

from probabilistic_machine_learning.adaptors.jax_adaptor import JaxWrap
from probabilistic_machine_learning.adaptors.jax_nuts import sample
from .examples import MosquitoTS2
from .fixtures import *
from .models import Models
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

from .util import expit

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
class MosquitoRegression:
    T = 400
    total_population: int = 1000.
    n_mosquitos = (np.arange(T) % 12) * 1.2 + 1
    n_infected = ((5 * np.arange(T)) % 12) * 1.2 + 1
    double_bite_rate = 0.1

    def sample(self):
        mu = self.n_mosquitos * self.double_bite_rate * self.n_infected / self.total_population * (
                1 - self.n_infected / self.total_population)
        return np.random.poisson(mu)

    def estimate(self, observed):
        n_mosquitos = self.n_mosquitos
        n_infected = self.n_infected
        total_population = self.total_population

        def logdensity_fn(lo_double_bite_rate):
            """Univariate Normal"""
            double_bite_rate = expit(lo_double_bite_rate)
            mu = n_mosquitos * double_bite_rate * n_infected / total_population * (1 - n_infected / total_population)
            logpdf = stats.poisson.logpmf(observed, mu)
            return jnp.sum(logpdf)

        logdensity = lambda x: logdensity_fn(**x)
        rng_key = jax.random.key(1000)

        return sample(logdensity, rng_key, {'lo_double_bite_rate': 0.0}, num_samples=1_000)


@dataclasses.dataclass
class MosquitoRegression2:
    T = 400
    total_population: int = 10000.
    n_mosquitos = (100*(np.arange(T) % 12)) * 1.2 + 1
    n_infected = ((100 * (1+(np.arange(T) % 12)))) * 2 + 1
    double_bite_rate = 10
    beta_security = 1000
    p_sigma = 1.0

    def sample(self):
        rate = self.double_bite_rate
        mu = self.get_mu(rate)
        eta = np.random.normal(mu, self.p_sigma)
        new_infections = expit(eta)
        return new_infections

    def get_mu(self, rate):
        n_infected = self.n_infected
        n_bits = self.n_mosquitos * rate * n_infected / self.total_population
        p = (1 - ((self.total_population - 1) / self.total_population) ** n_bits)
        mu = jax.scipy.special.logit(p)
        return mu

    def estimate(self, observed):

        def logdensity_fn(lo_double_bite_rate):
            """Univariate Normal"""
            double_bite_rate = jnp.exp(lo_double_bite_rate)
            mu = self.get_mu(double_bite_rate)
            eta = jax.scipy.special.logit(observed)
            logpdf = stats.norm.logpdf(eta, mu, self.p_sigma)
            return logpdf

        logdensity = lambda x: jnp.sum(logdensity_fn(**x))
        rng_key = jax.random.key(1000)

        return sample(logdensity, rng_key, {'lo_double_bite_rate': 0.0}, num_samples=1_000)

@dataclasses.dataclass
class MosquitoTS:
    T = 400
    total_population: int = 10000.
    n_mosquitos = (100*(np.arange(T) % 12)) * 1.2 + 1
    # n_infected = ((100 * (1+(np.arange(T) % 12)))) * 2 + 1
    init_infected = 100
    double_bite_rate = 10
    p_sigma = 1.0
    loss_rate = 0.1

    def sample(self):
        n_infected = [self.init_infected]
        observed = []
        for i in range(self.T):
            mu = self.get_mu(self.double_bite_rate, n_infected[-1], self.n_mosquitos[i])

            eta = np.random.normal(mu, self.p_sigma)
            n_susc = self.total_population - n_infected[-1]
            new_infections = expit(eta) * n_susc  # self.total_population
            n_infected.append(n_infected[-1] * (1-self.loss_rate) + new_infections)
            observed.append(new_infections)
        observed = np.array(observed)
        n_infected = np.array(n_infected)
        assert observed.shape == (self.T,)
        assert observed.shape == (self.T+1,)
        plt.plot(observed)
        plt.plot(n_infected)
        plt.show()
        return observed

    def get_mu(self, rate, n_infected, n_mosquitos):
        n_bits = n_mosquitos * rate * n_infected / self.total_population
        p = (1 - ((self.total_population - 1) / self.total_population) ** n_bits)
        mu = jax.scipy.special.logit(p)
        return mu

    def estimate(self, observed):
        n_infected = [self.init_infected]
        print(observed.shape)
        for o in observed[:-1]:
            n_infected.append(float(n_infected[-1] * (1-self.loss_rate) + o))
        n_infected = np.array(n_infected)
        def logdensity_fn(lo_double_bite_rate):#, log_odds_infected_rate):
            """Univariate Normal"""
            double_bite_rate = jnp.exp(lo_double_bite_rate)
            mu = self.get_mu(double_bite_rate, n_infected, self.n_mosquitos)
            n_susc = self.total_population - n_infected
            eta = jax.scipy.special.logit(observed/n_susc)# self.total_population)
            logpdf = stats.norm.logpdf(eta, mu, self.p_sigma)
            return logpdf

        logdensity = lambda x: jnp.sum(logdensity_fn(**x))
        rng_key = jax.random.key(1000)

        initial_position = {'lo_double_bite_rate': 0.0} #, 'log_odds_infected_rate': np.arange(self.T) + 0.5}

        return sample(logdensity, rng_key,
                      initial_position
                      , num_samples=1_000)



@pytest.mark.parametrize('model', [MosquitoTS2()])
def test_time_series_jax(model):
    # model = PoissonTS(100)
    np.random.seed(100)
    observed = model.sample()
    samples = model.estimate(observed)
    print(samples['lo_double_bite_rate'])
    plt.plot(np.exp(samples['lo_double_bite_rate']))
    plt.show()
    plt.hist(np.exp(samples['lo_double_bite_rate']))
    plt.show()


def test_get_jax_model(numpy_data_1, example_1):
    x, y = numpy_data_1
    jax_model = models.linear_regression(x, y)
    # assert set(jax_model.named_vars.keys()) == set(example_1.named_vars.keys())
