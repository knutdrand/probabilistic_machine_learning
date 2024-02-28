import jax.random
from oryx.core.ppl import random_variable, log_prob
from jax.scipy.special import expit
import oryx.distributions as tfd


def simple_sample(key):
    a = random_variable(tfd.Logistic(0., 1.))(key)
    return expit(a)


x = simple_sample(jax.random.PRNGKey(0))
print(x)  # 0.41845703
print(log_prob(simple_sample)(0.5))  # 0.0
print(log_prob(simple_sample)(x))  # 0.0
