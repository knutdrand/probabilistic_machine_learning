import blackjax
from jax.scipy import stats
from pymc.sampling_jax import get_jaxified_logp
import jax
import jax.numpy as jnp
import oryx.distributions as dists
from datetime import date
rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))

import pymc as pm

import numpy as np


def logprobfunc(state, beta, gamma, mu, a):
    S, E, I, R = state[:-1].T
    with pm.Model() as model:
        ps = get_ps(E, I, R, S, a, beta, gamma, mu)
        pm.Dirichlet('new_state', ps, observed=state[1:].T)
    print([rv.name for rv in model.value_vars])
    return get_jaxified_logp(model)()


def manual_logprobfunc(state, beta, gamma, mu, a):
    S, E, I, R = state[:-1].T
    ps = get_ps(E, I, R, S, a, beta, gamma, mu)
    params = jnp.array(ps).T * 20
    print(params)
    print(state[1:])
    tmp = dists.Dirichlet(params).log_prob(state[1:])
    print(tmp)
    return tmp.sum()


def sample_func(beta, gamma, mu, a):
    def sample(T):
        S, E, I, R = 0.9, 0.08, 0.01, 0.01
        states = [(S, E, I, R)]
        for t in range(T):
            ps = get_ps(E, I, R, S, a, beta, gamma, mu)
            new_state = np.random.dirichlet(np.array(ps)*20)
            print(ps, new_state)
            S, E, I, R = new_state
            states.append((S, E, I, R))
        return np.array(states)
    return sample

def get_ps(E, I, R, S, a, beta, gamma, mu):
    ps = [(1 - mu) * S * (1 - beta * S * I) + mu,
          (1 - mu) * E * (1 - a) + beta * (1 - mu) * S * I,
          (1 - mu) * I * (1 - gamma) + a * (1 - mu) * E,
          (1 - mu) * R + gamma * (1 - mu) * I]
    return ps


state = np.array([[0.9, 0.08, 0.01, 0.01],
                  [0.8, 0.09, 0.1, 0.01],
                  [0.7, 0., 0.1, 0.1]])

#logprobfunc(state, 0.3, 0.1, 0.05, 0.1)
states = sample_func(0.99, 0.2, 0.001, 0.1)(10)
print(states)

print(manual_logprobfunc(states, 0.9, 0.1, 0.01, 0.01))

def main():
    global rvs, rng_key
    J = 8
    y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=10.0)
        tau = pm.HalfCauchy("tau", 5.0)

        theta = pm.Normal("theta", mu=0, sigma=1, shape=J)
        theta_1 = mu + tau * theta
        obs = pm.Normal("obs", mu=theta_1, sigma=sigma, shape=J, observed=y)
    rvs = [rv.name for rv in model.value_vars]
    logdensity_fn = get_jaxified_logp(model)
    # Get the initial position from PyMC
    init_position_dict = model.initial_point()
    init_position = [init_position_dict[rv] for rv in rvs]
    rng_key, warmup_key = jax.random.split(rng_key)
    adapt = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)
    (last_state, parameters), _ = adapt.run(warmup_key, init_position, 1000)
    kernel = blackjax.nuts(logdensity_fn, **parameters).step

    def inference_loop(rng_key, kernel, initial_state, num_samples):
        def one_step(state, rng_key):
            state, info = kernel(rng_key, state)
            return state, (state, info)

        keys = jax.random.split(rng_key, num_samples)
        _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

        return states, infos

    rng_key, sample_key = jax.random.split(rng_key)
    states, infos = inference_loop(sample_key, kernel, last_state, 50_000)
    import matplotlib.pyplot as plt
    import arviz as az
    idata = az.from_dict(
        posterior={k: v[None, ...]
                   for k, v in zip(model.initial_point().keys(), states.position)})
    az.plot_trace(idata)
    plt.tight_layout();
    print(states)


#main()
