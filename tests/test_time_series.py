import numpy as np
from probabilistic_machine_learning.time_series import time_series
from probabilistic_machine_learning.distributions.scipy_distributions import Normal


def test_time_series():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([2, 3, 4, 5, 6])
    (a, b), t = time_series(a, b)
    a[t] = Normal(b[t+1])


