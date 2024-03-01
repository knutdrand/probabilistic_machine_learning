from typing import Protocol
import jax
from jax.scipy.special import logit, expit
import numpy as np
import jax.numpy as jnp
from .diff_encoded_mosquito import get_death_rate, get_maturation_rate_by_temp, mosquito_infection_rate_func, Logisitic, \
    Poisson


class DiffModelSpec(Protocol):
    def diff_distribution(self, seed):
        pass

    def transision_function(self, state, diff):
        pass

    def observation_distribution(self, state):
        pass

    def state_transform(self, state):
        ...


def multilogit(state):
    return jnp.log(state[..., :-1]) - jnp.log(state[..., -1:])


def inverse_multilogit(t_state):
    '''
    sum(ps)+p_r = 1
    x  = (1-p_r)/p_r
    p_r = 1/(1+x)
    '''
    ratios = jnp.exp(t_state)
    pr = 1 / (1 + ratios.sum(axis=-1))
    return jnp.concatenate([ratios * pr[..., None], pr[..., None]], axis=-1)


class DiffModel:
    def __init__(self, spec_class: type):
        self.spec_class = spec_class

    @property
    def param_names(self):
        return list(self.spec_class.good_params.keys())

    @property
    def n_states(self):
        return len(self.spec_class.init_state)

    def sampler(self, T, key=jax.random.PRNGKey(0), params=None, exogenous=None):
        observation_key, transition_key = jax.random.split(key)
        state = jnp.array(self.spec_class.init_state)
        spec = self.spec_class(params)

        def exogenous_sample(state, values):
            key, exog = values
            return spec.transition(state, spec.diff_distribution(state, exog).sample(key))

        values = jax.random.split(transition_key, T - 1)

        sample_transition = exogenous_sample
        values = (values, exogenous)
        state = jax.lax.scan(sample_transition, state, values)[1]
        return spec.observation_distribution(state).sample(observation_key)

    def recontstruct_state(self, logits_array, params=None):
        new_transition = self.spec_class(params).transition
        init_state = self.spec_class.init_state
        return jax.lax.scan(new_transition, jnp.array(init_state), logits_array)[1]

    def sample_diffs(self, transition_key=jax.random.PRNGKey(0), params=None, exogenous=None):
        state = jnp.array(self.spec_class.init_state)
        spec = self.spec_class(params)

        def exogenous_sample(state, values):
            key, exog = values
            diff = spec.diff_distribution(state, exog).sample(key)
            return spec.transition(state, diff)[0], diff

        values = jax.random.split(transition_key, len(exogenous))

        sample_transition = exogenous_sample
        values = (values, exogenous)

        state = jax.lax.scan(sample_transition, state, values)[1]
        return state

    def _accumulate_states(self, states):
        if offsets is None:
            return offsets
        return jnp.cumsum(states, axis=0)[offsets].diff(axis=0)

    def log_prob(self, observed, P, exogenous=None):
        spec = self.spec_class(P)
        diffs = P['logits_array']
        state = self.recontstruct_state(diffs, P)
        d_dist = spec.diff_distribution(state[:-1], exogenous[1:])
        state_pdf = d_dist.log_prob(diffs[1:]).sum()

        init_pdf = spec.diff_distribution(self.spec_class.init_state, exogenous[0]).log_prob(diffs[0]).sum()
        observed_pdf = spec.observation_distribution(state).log_prob(observed)
        return state_pdf + observed_pdf.sum() + init_pdf

    def lp_func(self, observed, exogenous=None):
        return lambda kwargs: self.log_prob(observed, kwargs, exogenous)


@jax.jit
def _get_mosquito_death_rate(mosquito_state, P):
    alpha = P['carry_alpha']
    beta = P['carry_beta']
    population_size = mosquito_state[..., 1]
    larva_death = get_death_rate(alpha, beta, population_size)
    return (P['mosquito_death_logit'], logit(larva_death), P['mosquito_death_logit'], P['mosquito_death_logit'],
            P['mosquito_death_logit'], P['mosquito_death_logit'])


@jax.jit
def _transition(P, logits, states):
    human_state, mosquito_state = states[..., :4], states[..., 4:]
    human_logits = logits[..., :4]
    human_diffs = human_state * expit(human_logits)
    mosquito_death_rate = _get_mosquito_death_rate(mosquito_state, P)
    death_rates = mosquito_death_rate
    mosquito_state = tuple(s * (1 - expit(d)) for s, d in zip(mosquito_state.T, death_rates))
    mosquito_logits = logits[..., 4:]
    mosquito_diffs = tuple(ms * expit(lg) for ms, lg in zip(mosquito_state, mosquito_logits.T))
    new_eggs = jnp.exp(P['log_eggrate']) * sum(mosquito_state[3:]) * expit(mosquito_logits.T[-1])
    new_state = human_state - human_diffs + jnp.roll(human_diffs, 1, axis=-1)
    new_state = jnp.array(
        [new_state[..., 0], new_state[..., 1], new_state[..., 2], new_state[..., 3],
                 mosquito_state[0] - mosquito_diffs[0] + new_eggs,
                 mosquito_state[1] - mosquito_diffs[1] + mosquito_diffs[0],
                 mosquito_state[2] - mosquito_diffs[2] + mosquito_diffs[1],
                 mosquito_state[3] - mosquito_diffs[3] + mosquito_diffs[2],
                 mosquito_state[4] - mosquito_diffs[4] + mosquito_diffs[3],
                 mosquito_state[5] + mosquito_diffs[4]]).T
    return new_state, new_state


class MosquitoModelSpec:
    init_state = jnp.array([0.88, 0.1, 0.01, 0.01, 100., 100., 100., 100., 10., 10.])
    good_params = {
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
        'lo_mosquito_gamma': logit(1 / 7),
    }

    def __init__(self, params):
        self._params = params

    def transition(self, states, logits):
        P = self._params
        return _transition(P, logits, states)

    def observation_distribution(self, state):
        P = self._params
        return Poisson(state[..., 2] * jnp.exp(P['log_rate']) + 1)

    def diff_distribution(self, state, exogenous):
        params = self._params
        mosquito_params = self.get_mosquito_maturation_rate(exogenous, state[..., 2])
        human_params = self.get_human_params(state[..., -1])
        param_array = jnp.array(jnp.broadcast_arrays(*(human_params + mosquito_params))).T
        scale = jnp.exp(params['logscale'])
        return Logisitic(loc=param_array, scale=scale)

    def get_mosquito_maturation_rate(self, temp, human_I):
        P = self._params
        infection_rate = mosquito_infection_rate_func(P, human_I)
        maturation_level = get_maturation_rate_by_temp(P, temp)
        return [logit(maturation_level / 3), logit(maturation_level / 5),
                P['lo_pupae_maturation'], P['lo_mosqutito_beta'] * human_I, P['lo_mosquito_gamma'], 0]

    def get_mosquito_death_rate(self, mosquito_state):
        P = self._params
        return _get_mosquito_death_rate(P, mosquito_state)

    def get_human_params(self, mosquito_I):
        P = self._params
        beta_diff = P['alpha'] + P['beta'] * jnp.log(mosquito_I)
        return [beta_diff] + [P[p] for p in ['lo_gamma', 'lo_a', 'lo_mu']]

    @staticmethod
    def state_transform(state):
        return jnp.concatenate([multilogit(state[..., :4]), jnp.log(state[..., 4:])], axis=-1)

    @staticmethod
    def inverse_state_transform(t_state):
        human_state = inverse_multilogit(t_state[..., :3])
        mosquito_state = jnp.exp(t_state[..., 3:])
        return jnp.concatenate([human_state, mosquito_state], axis=-1)
