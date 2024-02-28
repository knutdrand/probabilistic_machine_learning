import functools
from typing import Protocol
from abc import abstractmethod, ABC


class Distribution(Protocol):
    def log_prob(self, x: float | int) -> float:
        ...


class MyDistribution(Distribution):
    def logpdf(self, x: int) -> float:
        ...


class NormalDistribution(MyDistribution):
    def __init__(self, loc: float, scale: float):

        self.loc = loc
        self.scale = scale

    def _logpdf(self, x: float) -> float:
        return -0.5 * jnp.log(2 * jnp.pi * self.scale ** 2) - 0.5 * ((x - self.loc) / self.scale) ** 2

    def sample(self, seed) -> float:
        return self.loc + self.scale * jax.random.normal(seed)


def sample_from_dist(dist: Distribution, seed):
    return dist.sample(seed)


# Distribution()
sample_from_dist()# NormalDistribution(0.1, 1.), 1000)
