# Figure out beta parameters
# Figure out bSI tempering
# Write sampler
# Write logprob with diff-proportions as states
# Figure out scan
import jax.random
import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt
import scipy.stats as s_stats
from jax.scipy import stats
from jax.scipy.special import expit, logit
from probabilistic_machine_learning.adaptors.jax_nuts import sample as nuts_sample


def plot_beta():
    x = np.linspace(0, 1, 100)
    for p in [0.01, 0.05, 0.1]:
        for s in (1, 5, 10, 100):
            y = stats.beta.pdf(x, p * s, (1 - p) * s)
            plt.plot(x, y, label=f'p={p}, s={s}')
        plt.legend()
        plt.show()


def plot_sigmoid():
    I = np.linspace(0, 1, 100)
    for beta in [0.1, 1, 10]:
        y = expit(beta * I)
        plt.plot(I, y, label=f'beta={beta}')
    plt.legend()
    plt.show()

def scan_transition(state, logits):
    diffs = state * expit(logits)
    new_state = state - diffs + jnp.roll(diffs, 1)
    return new_state, new_state[1:3]

def model(beta, gamma, a, mu, scale, reporting_rate):
    init_state = [0.9, 0.08, 0.01, 0.01]
    lo_gamma, lo_a, lo_mu = (logit(p) for p in (gamma, a, mu))

    def sample_ps(state):
        ps = np.array([expit(beta * state[1]), gamma, a, mu])
        return stats.beta(ps * concentration, (1 - ps) * concentration).rvs()

    def sample_logits(state):
        logits = jnp.array([beta * state[1], lo_gamma, lo_a, lo_mu])
        return s_stats.logistic(loc=logits, scale=scale).rvs()

    def transition(state):
        logits = sample_logits(state)
        return apply_diffs(logits, state)

    def apply_diffs(logits, state):
        diffs = [s * expit(p) for s, p in zip(state, logits)]
        return [state[i] - diffs[i] + diffs[i - 1] for i in range(4)]



    def sample(T):
        state = init_state
        #states = [state]
        I = []
        for t in range(T - 1):
            state = transition(state)
            I.append(state[2])
        return s_stats.poisson(np.array(I)*reporting_rate).rvs()
        # return jnp.array(states)

    def log_prob(observed, logits_array, lo_gamma, lo_a, lo_mu):
        E, I = jax.lax.scan(scan_transition, jnp.array(init_state), logits_array)[1].T
        state_pdf = sum(stats.logistic.logpdf(column, param, scale).sum()
                        for column, param in zip(logits_array.T,
                                                 [beta * E, lo_gamma, lo_a, lo_mu]))
        return state_pdf + stats.poisson.logpmf(observed, I*reporting_rate).sum()

    return sample, lambda observed: (lambda kwargs: log_prob(observed, **kwargs))

if __name__ == '__main__':
    sample, log_prob = model(0.3, 0.1, 0.1, 0.05, 0.1, 10000)
    T = 1000
    observed = sample(T)

    # for i, col in enumerate(states.T):
    #     plt.plot(col, '-', label=f'{i}')
    # plt.legend()
    # plt.show()
    init_diffs = np.random.normal(0, 1, (T - 1, 4))
    print('Sampling')
    param_name = 'lo_gamma'
    param_names = ['lo_gamma', 'lo_a', 'lo_mu']
    inits = {name: 0.0 for name in param_names}
    samples = nuts_sample(log_prob(observed), jax.random.PRNGKey(0),
                          {'logits_array': init_diffs} | inits, 200, 1000)
    init_state = [0.9, 0.08, 0.01, 0.01]
    I = jax.lax.scan(scan_transition, jnp.array(init_state), samples['logits_array'][-1])[1][:, 1]
    plt.plot(observed);
    plt.plot(I*10000);
    plt.show()
    for name in param_names:
        plt.hist(expit(samples[name]))
        plt.title(name)
        plt.show()
