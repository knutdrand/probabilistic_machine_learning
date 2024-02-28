import seaborn as sns
from jax import jit, vmap, random
import matplotlib.pyplot as plt
from oryx.core import ppl
import jax.random
from oryx.core.ppl import random_variable as rv
from oryx.core.ppl import log_prob
import oryx
import jax.numpy as np
from jax.scipy.special import expit, logit
import oryx.distributions as tfd


def my_function(x):
    return np.exp(x) + 3


def simple_sample(key):
    a = rv(tfd.Logistic(0., 1.))(key)
    return expit(a)


prng_key = jax.random.PRNGKey(10)
for key in jax.random.split(prng_key, 10):
    print(simple_sample(key))
print(log_prob(simple_sample)(0.5))


def sample(key):
  a = ppl.random_variable(tfd.Normal(0., 1.))(key)
  return expit(a)

print(sample(jax.random.PRNGKey(0)))
f = log_prob(sample)
print('P', (0.5))
import jax.numpy as jnp

def log_normal(key):
  return expit(rv(tfd.Normal(0., 1.))(key))
T = 3


def series(key):
    keys = jax.random.split(key, T)
    def t(state, key):
        new_state = state + rv(tfd.Normal(1., 0.5))(key)#.sample(key)
        return new_state, new_state
    return jax.lax.scan(t, 0., keys)[1]

def series_2(key):
    zs = tfd.Normal(1., 0.5).sample(seed=key, sample_shape=(T,))
    def t(state, z):
        new_state = state + z
        return new_state, new_state
    return jax.lax.scan(t, 0., zs)[0]

x = series(jax.random.PRNGKey(0))
print(x)
print(log_prob(series)(x))
#print(log_normal(jax.random.PRNGKey(0)))
#sns.distplot(jit(vmap(log_normal))(random.split(random.PRNGKey(0), 10000)))
#plt.show()

