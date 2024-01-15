import scipy as scipy


def scipy_wrapper(scipy_rv, is_discrete=False):

    class Distribution:
        def __init__(self, *args, **kwargs):
            self._rv = scipy_rv(*args, **kwargs)
            self._logprob_func = self._rv.logpmf if is_discrete else self._rv.logpdf

        def sample(self, shape):
            return self._rv.rvs(shape)

        def log_prob(self, value):
            return self._logprob_func(value)

    cls = scipy_rv.__class__
    Distribution.__name__ = cls.__name__
    Distribution.__qualname__ = cls.__qualname__

    return Distribution


Normal = scipy_wrapper(scipy.stats.norm)
Beta = scipy_wrapper(scipy.stats.beta)
Binomial = scipy_wrapper(scipy.stats.binom, is_discrete=True)


class HalfCauchy:
    pass
