from probabilistic_machine_learning.sequential_monte_carlo.sequential_importance_sampling import \
    SequentialImportanceSampling
import probabilistic_machine_learning.distributions.scipy_distributions as dist


def test_sequential_imporatance_sampling():
    norm1 = dist.Normal(0, 1)
    norm2 = dist.Normal(0, 2)
    s = SequentialImportanceSampling(norm1, norm2)
    e = s.E(lambda x: x, n_samples=1000)
    assert np.abs(e) < 0.1
