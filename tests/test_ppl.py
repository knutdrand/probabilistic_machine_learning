import numpy as np
import pytest
import scipy.stats
from matplotlib import pyplot as plt
from scipy.special import logit, expit
from probabilistic_machine_learning.adaptors.base_adaptor import ModuleWrap
from probabilistic_machine_learning.ppl import logprob, given, sample

dists = ModuleWrap()


def diff_human(human_state, seir_params):
    beta, gamma, mu, a = seir_params
    n_mu = (1 - mu)
    S, E, I, R = human_state
    d = np.array(
        [mu - mu * S - beta * S,
         beta * S - (mu + a * n_mu) * E,
         a * E - (mu + gamma * n_mu) * I,
         gamma * I - mu * R])
    return d


def d_mosquito(state, death_rate, maturation_rate, beta, alpha, egglaying_rate):
    deaths = -death_rate * state
    maturation = (state + deaths) * maturation_rate
    carry_deaths = expit(beta + alpha * state[..., 1]) * 0.7
    d = np.array([deaths[..., 0] - maturation[..., 0] + egglaying_rate * state[..., 3:].sum(),
                  deaths[..., 1] + maturation[..., 0] - carry_deaths * (state[..., 1] + deaths[..., 1]) -
                  maturation[..., 1] * (1 - carry_deaths),
                  deaths[..., 2] - maturation[..., 2] + maturation[..., 1] * (1 - carry_deaths),
                  deaths[..., 3] + maturation[..., 2],
                  deaths[..., 4] + maturation[..., 3] - maturation[..., 4],
                  deaths[..., 5] + maturation[..., 4]]).T
    return d


def get_mosquito_update(self, mosquito_state, state, temp):
    maturation_rate = self.maturation.copy()
    logit_dependent = logit(maturation_rate[0]) + self.t_beta * temp  # , self.sigma)
    temp_dependent = expit(logit_dependent)
    maturation_rate[0] = temp_dependent / 3
    maturation_rate[1] = temp_dependent / 5
    logit_rate = logit(maturation_rate[3]) + self.h_beta * state[-2]  # , self.sigma)
    maturation_rate[3] = expit(logit_rate)
    d_mosquito = self.d_mosquito(mosquito_state, self.death_rate, maturation_rate, self.m_beta, self.m_alpha, 0.7)
    return d_mosquito


def transition_model(state, mosquito_state, alpha, beta, temp):
    sigma = 1
    logit_beta = np.random.normal(alpha + beta * mosquito_state[-1], sigma)
    beta = expit(logit_beta)
    d_human = diff_human(state, [beta] + seir_params)
    d_mosquito = get_mosquito_update(mosquito_state, state, temp)
    state = state + d_human
    mosquito_state = mosquito_state + d_mosquito
    return mosquito_state, state


def test_simpele_logprob():
    S, alpha, beta, event, newS = model_1()
    lp = logprob(event)
    true_lp = scipy.stats.norm.logpdf(logit(-(0.2/S-1)), alpha + beta * 2)
    assert np.allclose(lp, true_lp)


def model_1():
    alpha, beta = 2., 3.
    S = 0.3
    rate = expit(dists.Normal(alpha + beta * 2, 1))
    newS = S * (1 - rate)
    event = newS == 0.2
    return S, alpha, beta, event, newS

@pytest.mark.parametrize('model', [model_1])
def test_sample(model):
    S, alpha, beta, event, newS = model()
    sampled = sample(newS, shape=(1000,))
    sampled_S = sampled[0]
    assert sampled_S.shape == (1000,)
    plt.hist(sampled_S, bins=100);plt.show()


def test_sample_adc():
    M, S, alpha, beta, m_alpha, m_beta, new_M, new_S, temp = model_2()
    sampled = sample(new_M, new_S, shape=(1000,))
    sampled_M = sampled[0]
    sampled_S = sampled[1]
    assert sampled_M.shape == (1000,)
    assert sampled_S.shape == (1000,)

#@pytest.mark.xfail
def test_logprob():
    M, S, alpha, beta, m_alpha, m_beta, new_M, new_S, temp = model_2()
    lp = logprob(new_S==0.2, new_M==10., given(S==0.5, M==0.5))
    S = 0.5
    M = 0.5
    true_lp_s = scipy.stats.norm.logpdf(logit(-(0.2/S-1)), alpha + beta * M)
    true_lp_m = scipy.stats.norm.logpdf(np.log(10.), 0.5+m_alpha + m_beta * temp)
    assert np.allclose(lp, true_lp_s+true_lp_m)


def model_2():
    alpha, beta = 1., 1.
    m_alpha, m_beta = 1., 1.
    temp = 20
    S = dists.Beta(1, 1)
    M = dists.Gamma(3, 3)
    rate = expit(dists.Normal(alpha + beta * M, 1.))
    new_S = S * (1 - rate)
    new_M = np.exp(dists.Normal(M + m_alpha + m_beta * temp, 1.))
    return M, S, alpha, beta, m_alpha, m_beta, new_M, new_S, temp

