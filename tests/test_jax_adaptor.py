import dataclasses

from probabilistic_machine_learning.adaptors.jax_adaptor import JaxWrap
from probabilistic_machine_learning.adaptors.jax_nuts import sample
from .fixtures import *
from .jax_models import RealTS
from .models import Models
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
expit = lambda x: 1/(1+jnp.exp(-x))
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
    total_population: int=1000.
    n_mosquitos = (np.arange(T) % 12)*1.2+1
    n_infected = ((5*np.arange(T)) % 12)*1.2+1
    double_bite_rate = 0.1
    def sample(self):
        mu = self.n_mosquitos * self.double_bite_rate * self.n_infected/self.total_population*(1-self.n_infected/self.total_population)
        return np.random.poisson(mu)

    def estimate(self, observed):
        n_mosquitos = self.n_mosquitos
        n_infected = self.n_infected
        total_population = self.total_population

        def logdensity_fn(lo_double_bite_rate, n_mosquitos=n_mosquitos, n_infected=n_infected, total_population=total_population, observed=observed):
            """Univariate Normal"""
            # expit = jax.scipy.special.expit
            expit = lambda x: 1/(1+jnp.exp(-x))
            double_bite_rate = expit(lo_double_bite_rate)
            mu = n_mosquitos * double_bite_rate * n_infected / total_population * (1 - n_infected / total_population)
            logpdf = stats.poisson.logpmf(observed, mu)
            return jnp.sum(logpdf)
        print(jax.grad(logdensity_fn)(0.0))
        logdensity = lambda x: logdensity_fn(**x)
        rng_key = jax.random.key(1000)

        return sample(logdensity, rng_key, {'lo_double_bite_rate': 0.0}, num_samples=1_000)


@pytest.mark.parametrize('model', [MosquitoRegression()]) #, RealTS()])#, BasicTS(100), PoissonTS(100)])
def test_time_series_jax(model):
    # model = PoissonTS(100)
    np.random.seed(100)
    observed = model.sample()
    samples = model.estimate(observed)
    plt.hist(expit(samples['lo_double_bite_rate'])); plt.show()

def test_get_jax_model(numpy_data_1, example_1):
    x, y = numpy_data_1
    jax_model = models.linear_regression(x, y)
    # assert set(jax_model.named_vars.keys()) == set(example_1.named_vars.keys())


