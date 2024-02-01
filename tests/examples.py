import dataclasses

import jax
import numpy as np
from jax import numpy as jnp
from jax.scipy import stats as stats
from matplotlib import pyplot as plt

from probabilistic_machine_learning.adaptors.jax_nuts import sample
from tests.util import expit, logit


@dataclasses.dataclass
class MosquitoTS2:
    T = 400
    total_population: int = 10000.
    n_mosquitos = (100*(np.arange(T) % 12)) * 1.2 + 1
    init_infected = 100
    double_bite_rate = 10
    p_sigma = 1.0
    loss_rate = 0.1

    def sample(self):
        n_infected = [self.init_infected]
        observed = []
        for i in range(self.T):
            cur_rate = n_infected[-1]/self.total_population
            p = self.get_p(self.double_bite_rate, n_infected[-1], self.n_mosquitos[i])
            new_mu = cur_rate*(1-self.loss_rate) + p*(1-cur_rate)
            eta = np.random.normal(logit(new_mu), self.p_sigma)
            new_rate = expit(eta)
            new_infections = (new_rate-cur_rate)*self.total_population  # self.total_population
            n_infected.append(new_rate*self.total_population)
            o = np.random.normal(new_infections, 10)
            observed.append(o)
        observed = np.array(observed)
        assert observed.shape == (self.T,)
        plt.plot(observed)
        plt.plot(n_infected)
        plt.show()
        return observed

    def get_mu(self, rate, n_infected, n_mosquitos):
        p = self.get_p(n_infected, n_mosquitos, rate)
        mu = jax.scipy.special.logit(p)
        return mu

    def get_p(self, n_infected, n_mosquitos, rate):
        n_bits = n_mosquitos * rate * n_infected / self.total_population
        p = (1 - ((self.total_population - 1) / self.total_population) ** n_bits)
        return p

    def estimate(self, observed):
        n_infected = [self.init_infected]
        print(observed.shape)
        for o in observed[:-1]:
            print(o)
            n_infected.append(float(n_infected[-1] * (1-self.loss_rate) + o))
        n_infected = np.array(n_infected)

        def get_n_infected(log_odds_states):
            ps = expit(log_odds_states)
            rates = [self.init_infected/self.total_population]
            for p in ps:
                rates.append(
                    (1-p)*rates[-1]+p-self.loss_rate*rates[-1])
            return jnp.array(rates)*self.total_population

        def logdensity_fn(lo_double_bite_rate, log_odds_infected_rate):
            """Univariate Normal
            """
            n_infected = expit(log_odds_infected_rate)*self.total_population
            n_infected = jnp.insert(n_infected, 0, self.init_infected)
            new_infections = n_infected[1:]-n_infected[:-1]
            double_bite_rate = jnp.exp(lo_double_bite_rate)
            cur_rate = n_infected[:-1] / self.total_population
            p = self.get_p(double_bite_rate, n_infected[:-1], self.n_mosquitos)
            new_mu = cur_rate*(1-self.loss_rate) + p*(1-cur_rate)
            state_logpdf = stats.norm.logpdf(log_odds_infected_rate, logit(new_mu), self.p_sigma)
            observed_logpdf = stats.norm.logpdf(observed, new_infections, 10)
            return state_logpdf.sum() + observed_logpdf.sum()

        logdensity = lambda x: logdensity_fn(**x)
        rng_key = jax.random.key(1000)

        initial_position = {'lo_double_bite_rate': 0.0, 'log_odds_infected_rate': (np.arange(self.T) + 0.5) % 7}

        return sample(logdensity, rng_key,
                      initial_position
                      , num_samples=1_000)




