import numpy as np
import pytest
from probabilistic_machine_learning.monte_carlo_methods.importance_sampling import DirectImportanceSampling
import probabilistic_machine_learning.distributions.scipy_distributions as dist


@pytest.fixture
def norm1():
    return dist.Normal(0, 1)

@pytest.fixture
def norm2():
    return dist.Normal(0, 2)


def test_e(norm1, norm2):
    s = DirectImportanceSampling(norm1, norm2)
    e = s.E(lambda x: x, n_samples=1000)
    assert np.abs(e)<0.1
