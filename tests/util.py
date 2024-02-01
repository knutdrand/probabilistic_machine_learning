import jax
from jax import numpy as jnp

expit = lambda x: 1 / (1 + jnp.exp(-x))


def logit(x):
    return jax.scipy.special.logit(x)
