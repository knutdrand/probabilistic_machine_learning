import numpy as np
from probabilistic_machine_learning.time_series import time_series
from probabilistic_machine_learning.distributions.scipy_distributions import Normal
from probabilistic_machine_learning.time_series import TimeSeries, pymc_time_series
import matplotlib.pyplot as plt
import pymc as pm

def _test_time_series():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([2, 3, 4, 5, 6])
    (a, b), t = time_series(a, b)
    a[t] = Normal(b[t+1])


def transition_function(x, t, beta=2, observed=None):
    name = f'X_{t}'
    if observed is not None:
        return pm.Normal(name,
                         x+beta,
                         1,
                         observed=observed[name][0, 0])
    return pm.Normal(name,
                     x + beta,
                     1)


def test_simple_time_series():
    T = 100
    simulation_model = pymc_time_series(transition_function, 0, T)
    with simulation_model:
        #trace = pm.sample(1).posterior
        samples = pm.sample_prior_predictive(1).prior
        #print(trace)
        ##for i in range(10):
        plt.plot([samples[f'X_{t}'][0, 0] for t in range(1, T)])
        plt.show()
        # plt.scatter(trace['X_1'], trace['X_2'])
        #plt.show()
    with pymc_time_series(transition_function, 0, T, beta=lambda: pm.Normal('beta', 0, 4), observed=samples):
        trace = pm.sample(1000).posterior
        print(trace)
        beta=trace['beta'].to_numpy().ravel()
        plt.hist(beta); plt.show()
        #print(trace)
        ##for i in range(10):
        #    plt.plot([trace[f'X_{t}'][..., i] for t in range(1, 10)])
        # plt.scatter(trace['X_1'], trace['X_2'])
        #plt.show()

