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
from functools import wraps, partial


def mosquito_infection_rate_func(P, human_infected):
    infection_rate = jnp.log(human_infected) * P['infection_rate_slope'] + P['base_infection_rate']
    return infection_rate

def new_mosquito_infection_rate_func(P, human_infected):
    infection_rate = logit(human_infected) * P['infection_rate_slope'] + P['base_infection_rate']
    return infection_rate


def get_death_rate(alpha, beta, population_size):
    return expit(alpha + beta * population_size) * 0.7

def get_maturation_rate_by_temp(P, temp):
    return expit(P['temp_base'] + P['temp_dependency'] * temp)




def full_diff_encoded_model(transition, init_state, observation_dist, diff_dist, offsets=None):
    def sampler(T, key=jax.random.PRNGKey(0), params=None, exogenous=None):
        observation_key, transition_key = jax.random.split(key)
        state = jnp.array(init_state)
        dist = diff_dist(state, params, exogenous[0] if exogenous is not None else None)

        def exogenous_sample(state, values):
            key, exog = values
            return transition(state, diff_dist(state, params, exog).sample(key), params)

        if isinstance(dist, tuple):
            N = len(dist)
            sample_transition = lambda state, t_key: transition(state, jnp.array([d.sample(k) for d, k in
                                                                                  zip(diff_dist(state, params),
                                                                                      jax.random.split(t_key, N))]))
        else:
            sample_transition = lambda state, key: transition(state, diff_dist(state, params).sample(key))

        values = jax.random.split(transition_key, T - 1)

        if exogenous is not None:
            sample_transition = exogenous_sample
            values = (values, exogenous)

        state = jax.lax.scan(sample_transition, state, values)[1]
        if False:
            for i in range(4, len(state.T)):
                plt.plot(state[:, i], label=str(i))
            plt.legend()
            plt.show()
            for i in range(4):
                plt.plot(state[:, i], label=str(i))
            plt.legend()
            plt.show()

        return observation_dist(state, params).sample(observation_key)

    def recontstruct_state(logits_array, params=None):
        new_transition = partial(transition, P=params)
        return jax.lax.scan(new_transition, jnp.array(init_state), logits_array)[1]

    def sample_diffs(transition_key=jax.random.PRNGKey(0), params=None, exogenous=None):
        state = jnp.array(init_state)
        dist = diff_dist(state, params, exogenous[0] if exogenous is not None else None)

        def exogenous_sample(state, values):
            key, exog = values
            diff = diff_dist(state, params, exog).sample(key)
            return transition(state, diff, params)[0], diff

        values = jax.random.split(transition_key, len(exogenous))

        sample_transition = exogenous_sample
        values = (values, exogenous)

        state = jax.lax.scan(sample_transition, state, values)[1]
        return state

    def _accumulate_states(states):
        if offsets is None:
            return offsets
        return jnp.cumsum(states, axis=0)[offsets].diff(axis=0)

    def log_prob(observed, P, exogenous=None):
        diffs = P['logits_array']
        state = recontstruct_state(diffs, P)
        d_dist = diff_dist(state[:-1], P, exogenous[1:])
        if isinstance(d_dist, tuple):
            state_pdf = sum(dist.log_prob(diff[1:]).sum() for dist, diff in zip(d_dist, diffs.T))
            init_pdf = sum(dist.log_prob(diff[0]) for dist, diff in zip(diff_dist(init_state, P), diffs.T))
        else:
            state_pdf = d_dist.log_prob(diffs[1:]).sum()
            init_pdf = diff_dist(init_state, P, exogenous[0]).log_prob(diffs[0]).sum()
        observed_pdf = observation_dist(state, P).log_prob(observed)
        return state_pdf + observed_pdf.sum() + init_pdf

    lp_func = lambda observed, exogenous=None: (lambda kwargs: log_prob(observed, kwargs, exogenous))
    return sampler, lp_func, recontstruct_state, sample_diffs


def diff_encoded_model(*args, **kwargs):
    return full_diff_encoded_model(*args, **kwargs)[:-1]


def loc_scale_warp(dist):
    @wraps(dist)
    def wrapper(key, loc, scale, *args, **kwargs):
        return loc + scale * dist(key, *args, **kwargs)

    return wrapper


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

        def __repr__(self):
            return f'{jax_dist.__name__}({self.args}, {self.kwargs})'

    Dist.__name__ = jax_dist.__name__
    return Dist


Poisson = Distribution(jax.random.poisson, stats.poisson.logpmf)
Logisitic = Distribution(loc_scale_warp(jax.random.logistic), stats.logistic.logpdf)


def refactor_model(param_dict):
    def diff_dist(state, P=param_dict, exogenous=None):
        P = {k: P[k] if k in P else param_dict[k] for k in param_dict}
        loc = [P['beta'] + logit(state[..., 2]), P['lo_gamma'], P['lo_a'], P['lo_mu']]
        return tuple(Logisitic(l, jnp.exp(P['logscale'])) for l in loc)

    def observation_dist(state, P=param_dict):
        P = {k: P[k] if k in P else param_dict[k] for k in param_dict}
        return Poisson(state[..., 2] * P['reporting_rate'])

    def transition(state, logits):
        diffs = state * expit(logits)
        new_state = state - diffs + jnp.roll(diffs, 1)
        return new_state, new_state

    init_state = jnp.array([0.9, 0.08, 0.01, 0.01])
    return diff_encoded_model(transition, init_state, observation_dist, diff_dist)


def simple_model(reporting_rate=10000, logscale=jnp.log(0.1), lo_gamma=logit(0.1), lo_a=logit(0.1), lo_mu=logit(0.05)):
    init_state = jnp.array([0.9, 0.08, 0.01, 0.01])

    def diff_dist(state, P, exogenous=None):
        return Logisitic(loc=P['beta'] * exogenous + jnp.log(state[..., 2]), scale=jnp.exp(logscale))

    def observation_dist(state, P=None):
        return Poisson(state[..., 2] * reporting_rate)

    def transition(state, logit, P=None):
        logits = jnp.array([logit, lo_gamma, lo_a, lo_mu])
        diffs = state * expit(logits)
        new_state = state - diffs + jnp.roll(diffs, 1)
        return new_state, new_state

    return diff_encoded_model(transition, init_state, observation_dist, diff_dist)


def seir_model():
    init_state = jnp.array([0.9, 0.08, 0.01, 0.01])
    params = ['lo_rate', 'logscale', 'beta', 'lo_gamma', 'lo_a', 'lo_mu']

    def diff_dist(state, P, exogenous=None):
        beta_diff = P['beta'] * exogenous + jnp.log(state[..., 2])
        return Logisitic(loc=jnp.array([beta_diff] + [jnp.full_like(beta_diff, P[p]) for p in params[3:]]).T,
                         scale=jnp.exp(P['logscale']))

    def observation_dist(state, P=None):
        return Poisson(state[..., 2] * expit(P['lo_rate']) * 100_000)

    def transition(state, logit, P=None):
        logits = jnp.array([logit[0], logit[1], logit[2], logit[3]])
        diffs = state * expit(logits)
        new_state = state - diffs + jnp.roll(diffs, 1)
        return new_state, new_state

    return diff_encoded_model(transition, init_state, observation_dist, diff_dist), (params, len(params) - 2)


def check_big_model():
    param_names = ['lo_gamma', 'lo_mu', 'lo_a', 'lo_mu', 'beta']
    real_params = {'beta': 0.3, 'lo_gamma': logit(0.1), 'lo_a': logit(0.1), 'lo_mu': logit(0.05),
                   'logscale': np.log(0.1), 'reporting_rate': 10000}
    sample, log_prob, reconstruct_state = refactor_model(real_params)
    T = 100
    observed = sample(T, jax.random.PRNGKey(100), real_params)
    init_diffs = np.random.normal(0, 1, (T - 1, 4))
    inits = {name: 0.0 for name in param_names}
    init_dict = {'logits_array': init_diffs} | inits
    log_prob(observed)(init_dict)
    samples = nuts_sample(log_prob(observed), jax.random.PRNGKey(0),
                          init_dict, 100, 100)
    plt.plot(observed / real_params['reporting_rate'], label='observed')
    states = reconstruct_state(samples['logits_array'][-1])[:, 2]
    plt.plot(states, label='reconstructed')
    for i in range(1, 1000, 100):
        new_params = {k: samples[k][i] if k in samples else real_params[k] for k in real_params}
        new_sample, *_ = refactor_model(new_params)
        new_observed = new_sample(T, jax.random.PRNGKey(i), new_params)
        plt.plot(new_observed / real_params['reporting_rate'], label='new_observed{}'.format(i), color='grey')
    for key in [200, 2000, 50, 99999]:
        alt_observed = sample(T, jax.random.PRNGKey(key), real_params)
        plt.plot(alt_observed / real_params['reporting_rate'], label='alt_observed{}'.format(key), color='blue')
    plt.legend()
    plt.show()


def model_evaluation_plot(sample_func, real_params, sampled_params, temperature):
    reporting_rate = 10000
    T = len(temperature) + 1
    n_samples = len(sampled_params[list(real_params)[0]])
    for i in range(1, n_samples, n_samples // 10):
        new_observed = sample_func(T, jax.random.PRNGKey(i),
                                   {param_name: sampled_params[param_name][i] for param_name in real_params},
                                   temperature)
        plt.plot(new_observed / reporting_rate, label='predicted', color='grey')
    for key in [200, 2000, 50, 99999]:
        alt_observed = sample_func(T, jax.random.PRNGKey(key), real_params, temperature)
        plt.plot(alt_observed / reporting_rate, label='real', color='blue')
    # Add description
    plt.title('Prdicted path using estimated parameters vs real parameters')
    plt.legend()
    return plt.gcf()


def debug_logprob(log_prob, sampled_parameters):
    last_parameters = {k: v[-1] for k, v in sampled_parameters.items()}
    log_prob(last_parameters)
    jax.grad(log_prob)(last_parameters)


def check_small_model():
    real_params = {'beta': 0.5}
    sample, log_prob, reconstruct_state = simple_model()
    T = 100
    temperature = np.random.normal(1, 1, T - 1)
    observed = sample(T, jax.random.PRNGKey(100), real_params, temperature)
    init_diffs = np.random.normal(0, 1, (T - 1))
    init_dict = {'logits_array': init_diffs} | {'beta': 0.0}
    lp = log_prob(observed, temperature)
    print(lp(init_dict))
    print(jax.grad(lp)(init_dict))

    samples = nuts_sample(lp, jax.random.PRNGKey(0),
                          init_dict, 200, 100)
    debug_logprob(lp, samples)
    model_evaluation_plot(sample, real_params, samples, temperature)
    plt.plot(samples['beta'], label='beta');
    plt.show()
    plt.plot(observed / 10000, label='observed')
    last_diffs = samples['logits_array'][-1]
    states = reconstruct_state(last_diffs)[..., 2]
    plt.plot(states, label='reconstructed')
    plt.legend()
    plt.show()


def mosquito_model():
    init_state = jnp.array([0.9, 0.08, 0.01, 0.01, 100, 100, 100, 100, 100, 100])
    params = ['lo_rate', 'logscale',
              'lo_gamma', 'lo_a', 'lo_mu',
              'alpha', 'beta', 'temp_base', 'temp_dependency',
              'infection_rate_slope', 'base_infection_rate',
              'lo_mosquito_gamma', 'lo_mosquito_a', 'lo_mosquito_mu', 'log_eggrate',
              'mosquito_death_logit', 'carry_beta', 'carry_alpha']

    def get_mosquito_maturation_rate(state, temp, P):
        maturation_level = expit(P['temp_base'] + P['temp_dependency'] * temp)
        human_infected = state[..., 2]
        infection_rate = mosquito_infection_rate_func(P, human_infected)
        return [logit(maturation_level / 3), logit(maturation_level / 5), infection_rate, P['lo_mosquito_gamma'],
                P['lo_mosquito_a'], -100000]


    def get_human_params(state, P):
        mosquito_state = state[..., 4:]
        human_state = state[..., :4]
        beta_diff = P['alpha'] + P['beta'] * mosquito_state[..., -1]
        return [beta_diff] + [jnp.full_like(beta_diff, P[p]) for p in ['lo_gamma', 'lo_a', 'lo_mu']]

    def diff_dist(state, P, exogenous=None):
        human_params = get_human_params(state, P)
        mosquito_params = get_mosquito_maturation_rate(state, exogenous, P)
        params = jnp.broadcast_arrays(*(human_params + mosquito_params))
        return Logisitic(loc=jnp.array(params).T, scale=jnp.exp(P['logscale']))

    def observation_dist(state, P=None):
        return Poisson(state[..., 2] * expit(P['lo_rate']) * 100_000)

    def transition(states, logit, P):
        diffs = states * expit(logit)
        human_state, mosquito_state = states[..., :4], states[..., 4:]
        human_diffs = human_state * expit(logit[:4])
        mosquito_state = mosquito_state * (1 - expit(P['mosquito_death_logit']))
        carry_deaths = expit(P['carry_beta'] + P['carry_alpha'] * mosquito_state[..., 1]) * 0.7
        mosquito_state = jnp.concatenate(
            [mosquito_state[..., 0:1], mosquito_state[..., 1:2] * (1 - carry_deaths), mosquito_state[..., 2:]])
        mosquito_diffs = mosquito_state * expit(logit[4:])
        new_state = human_state - human_diffs + jnp.roll(human_diffs, 1)
        new_eggs = jnp.exp(P['log_eggrate']) * jnp.sum(mosquito_state[..., 3:], axis=-1)
        new_mosquito_state = mosquito_state - mosquito_diffs + jnp.roll(mosquito_diffs, 1)
        new_state = jnp.concatenate([new_state, new_mosquito_state[0:1] + new_eggs, new_mosquito_state[1:]])
        return new_state, new_state

    n_states = 10
    assert diff_dist(jnp.zeros(n_states), {k: 0.0 for k in params}, 1).sample(jax.random.PRNGKey(0)).shape == (
        n_states,)
    diff_dist(jnp.zeros((3, n_states)), {k: 0.0 for k in params}, np.ones(3))
    assert transition(jnp.ones(n_states), jnp.zeros(n_states), {k: 0.0 for k in params})[0].shape == (n_states,)
    return diff_encoded_model(transition, init_state, observation_dist, diff_dist), (params, n_states)


def pure_mosquito_model():
    state_names = ['eggs', 'larvae', 'pupae', 'mosquitoes']
    init_state = jnp.array([100., 100., 100., 100.])
    params = ['temp_base', 'temp_dependency', 'lo_pupae_maturation', 'logscale', 'mosquito_death_logit', 'carry_beta',
              'carry_alpha', 'log_eggrate', 'lo_rate']
    prior_values = {'temp_base': logit(0.00001),
                    'temp_dependency': 0.5,
                    'lo_pupae_maturation': 0.33,
                    'logscale': np.log(1.),
                    'mosquito_death_logit': logit(0.05),
                    'carry_beta': logit(0.0002),
                    'carry_alpha': 0.005,
                    'log_eggrate': jnp.log(1000),
                    'lo_rate': 0
                    }

    def get_mosquito_maturation_rate(temp, P):
        maturation_level = get_maturation_rate_by_temp(P, temp)
        return [logit(maturation_level / 3), logit(maturation_level / 5), P['lo_pupae_maturation'], 0]

    def get_mosquito_death_rate(mosquito_state, P):
        alpha = P['carry_alpha']
        beta = P['carry_beta']
        population_size = mosquito_state[..., 1]
        larva_death = get_death_rate(alpha, beta, population_size)
        return [P['mosquito_death_logit'], logit(larva_death), P['mosquito_death_logit'], P['mosquito_death_logit']]



    def diff_dist(state, P, exogenous=None):
        mosquito_params = get_mosquito_maturation_rate(exogenous, P)
        param_array = jnp.array(jnp.broadcast_arrays(*mosquito_params)).T
        scale = jnp.exp(P['logscale'])
        return Logisitic(loc=param_array, scale=scale)

    def transition(states, logits, P):
        death_rates = jnp.array(get_mosquito_death_rate(states, P))
        mosquito_state = states * (1 - expit(death_rates))
        mosquito_diffs = mosquito_state * expit(logits)
        new_state = jnp.array([mosquito_state[0] - mosquito_diffs[0] + jnp.exp(P['log_eggrate']) * mosquito_diffs[-1],
                               mosquito_state[1] - mosquito_diffs[1] + mosquito_diffs[0],
                               mosquito_state[2] - mosquito_diffs[2] + mosquito_diffs[1],
                               mosquito_state[3] + mosquito_diffs[2]])
        return new_state, new_state

    def observation_dist(state, P=None):
        return Poisson(state[..., 2] * expit(P['lo_rate'])+1.)

    n_states = len(state_names)
    assert diff_dist(jnp.zeros(n_states), {k: 0.0 for k in params}, 1).sample(jax.random.PRNGKey(0)).shape == (
        n_states,)
    return diff_encoded_model(transition, init_state, observation_dist, diff_dist), (prior_values, len(state_names))



def full_model():
    state_names = ['eggs', 'larvae', 'pupae', 'mosquitoes', 'exposed_mosquitos', 'infected_mosquitos',
                   'susceptible_humans', 'exposed_humans', 'infected_humans', 'recovered_humans']
    init_state = jnp.array([0.88, 0.1, 0.01, 0.01, 100., 100., 100., 100., 10., 10.])
    params = ['temp_base', 'temp_dependency', 'lo_pupae_maturation', 'logscale', 'mosquito_death_logit', 'carry_beta',
              'carry_alpha', 'log_eggrate', 'lo_rate',
                'lo_gamma', 'lo_a', 'lo_mu', 'alpha', 'beta', 'infection_rate_slope', 'base_infection_rate', 'lo_mosqutito_beta',
                'lo_mosquito_gamma']
    prior_values = {
        'temp_base': -30.,
    'temp_dependency': 1.,
    'lo_pupae_maturation': logit(0.33),
    'logscale': np.log(0.1),
    'mosquito_death_logit': logit(0.1),
    'carry_beta': 0.01,  # Verified
    'carry_alpha': -10.0,  # Verified
    'log_eggrate': jnp.log(10),
    'log_rate': jnp.log(100000),
    'lo_gamma': logit(0.1),
    'lo_a': logit(0.1),
    'lo_mu': logit(0.05),
    'alpha': logit(0.0001),
    'beta': 0.5,
    'infection_rate_slope': 0.5,
    'base_infection_rate': logit(0.01),
    'lo_mosqutito_beta': logit(0.2),
    'lo_mosquito_gamma': logit(1/7),
    }

    def get_mosquito_maturation_rate(temp, P, human_I):
        infection_rate = mosquito_infection_rate_func(P, human_I)
        maturation_level = get_maturation_rate_by_temp(P, temp)
        return [logit(maturation_level / 3), logit(maturation_level / 5),
                P['lo_pupae_maturation'], P['lo_mosqutito_beta']*human_I, P['lo_mosquito_gamma'], 0]


    def get_mosquito_death_rate(mosquito_state, P):
        alpha = P['carry_alpha']
        beta = P['carry_beta']
        population_size = mosquito_state[..., 1]
        larva_death = get_death_rate(alpha, beta, population_size)
        return [P['mosquito_death_logit'], logit(larva_death), P['mosquito_death_logit'], P['mosquito_death_logit'],
                P['mosquito_death_logit'], P['mosquito_death_logit']]

    def get_human_params(P, mosquito_I):
        beta_diff = P['alpha'] + P['beta'] * jnp.log(mosquito_I)
        return [beta_diff] + [P[p] for p in ['lo_gamma', 'lo_a', 'lo_mu']]

    def diff_dist(state, P, exogenous=None):
        mosquito_params = get_mosquito_maturation_rate(exogenous, P, state[..., 2])
        human_params = get_human_params(P, state[..., -1])
        param_array = jnp.array(jnp.broadcast_arrays(*(human_params+mosquito_params))).T
        scale = jnp.exp(P['logscale'])
        return Logisitic(loc=param_array, scale=scale)

    def transition(states, logits, P):
        human_state, mosquito_state = states[..., :4], states[..., 4:]
        human_logits = logits[:4]
        human_diffs = human_state * expit(human_logits)
        death_rates = jnp.array(get_mosquito_death_rate(mosquito_state, P))
        mosquito_state = mosquito_state * (1 - expit(death_rates))
        mosquito_logits = logits[4:]
        mosquito_diffs = mosquito_state * expit(mosquito_logits)
        new_eggs = jnp.exp(P['log_eggrate']) * jnp.sum(mosquito_state[..., 3:], axis=-1)*expit(mosquito_logits[-1])
        new_state = human_state - human_diffs + jnp.roll(human_diffs, 1)
        new_state = jnp.array([new_state[0], new_state[1], new_state[2], new_state[3],
                               mosquito_state[0] - mosquito_diffs[0] + new_eggs,
                               mosquito_state[1] - mosquito_diffs[1] + mosquito_diffs[0],
                               mosquito_state[2] - mosquito_diffs[2] + mosquito_diffs[1],
                               mosquito_state[3] - mosquito_diffs[3] + mosquito_diffs[2],
                               mosquito_state[4] - mosquito_diffs[4] + mosquito_diffs[3],
                               mosquito_state[5]  + mosquito_diffs[4]])

        return new_state, new_state

    def observation_dist(state, P=None):
        return Poisson(state[..., 2] * jnp.exp(P['log_rate'])+1)

    n_states = len(state_names)
    assert diff_dist(jnp.zeros(n_states), {k: 0.0 for k in params}, 1).sample(jax.random.PRNGKey(0)).shape == (
        n_states,)
    return full_diff_encoded_model(transition, init_state, observation_dist, diff_dist), (prior_values, len(state_names))


if __name__ == '__main__':
    check_small_model()
    # check_big_model()
