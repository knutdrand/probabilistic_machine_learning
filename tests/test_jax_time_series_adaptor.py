import numpy as np
from probabilistic_machine_learning.time_series import time_series, scalar, sample, given
from probabilistic_machine_learning.distributions.scipy_distributions import Normal

def _desired_api(n_mosquitos, init_infected, total_population, double_bite_rate, p_sigma, loss_rate, T):
    def get_p(self, n_infected, n_mosquitos, rate):
        n_bits = n_mosquitos * rate * n_infected / self.total_population
        p = (1 - ((self.total_population - 1) / self.total_population) ** n_bits)
        return p

    def get_param(n_mosquitos, n_infected):
        cur_rate = n_infected / total_population
        p = get_p(double_bite_rate, n_mosquitos, n_infected)
        new_mu = cur_rate * (1 - loss_rate) + p * (1 - cur_rate)
        return logit(new_mu)

    t, n_mosquitos, n_infected, observed, *_ = time_series(n_mosquitos, init_infected)
    n_infected[t + 1] = expit(Normal(get_param(n_mosquitos[t], n_infected[t]), p_sigma)) * total_population
    observed[t + 1] = Normal(n_infected[t + 1] - n_infected[t], 10)


def simple_api():
    t, (n_mosquitos, n_infected) = time_series(2)
    rate = scalar()
    n_infected[t + 1] = Normal(n_mosquitos[t] * n_infected[t] * rate, 10)
    return n_infected, n_mosquitos, rate


def test_simple_api():
    n_infected, n_mosquitos, rate = simple_api()
    samples = sample(n_infected, given(n_mosquitos == (np.arange(20) % 12), rate == 10))
    assert samples.shape == 20
