import dataclasses
from functools import partial
from scipy import stats
import plotly.express as px
from jax import random
import numpy as np
import oryx.distributions as dists
import oryx.core.ppl as ppl
import jax.numpy as jnp


def linear_regression_sampler(x):
    alpha = np.random.normal(0, 10)
    beta = np.random.normal(0, 10)
    sigma = np.random.uniform(0, 10)
    Y = np.random.normal(alpha + x*beta, sigma)
    return Y, alpha, beta, sigma


def linear_regression_logprob_func(x, y):
    def logprob(alpha, beta, sigma):
        y_pdf = stats.norm(alpha + x * beta, sigma).logpdf(y)
        alpha_pdf = stats.norm(0, 10).logpdf(alpha)
        beta_pdf = stats.norm(0, 10).logpdf(beta)
        #sigma_pdf = scipy.stats.uniform(0, 10).alpha_pdf(sigma)
        return y_pdf + alpha_pdf + beta_pdf
    return logprob



def linear_regression_model(seed, x):
    a_key, b_key, s_key, y_seed, x_seed = random.split(seed, num=5)
    # x = ppl.random_variable(dists.Sample(dists.Normal(0., 1.), sample_shape=3), name='x')(x_seed)
    alpha = ppl.random_variable(dists.Normal(0, 10), name='alpha')(a_key)
    beta = ppl.random_variable(
        dists.Sample(dists.Normal(0, 10),
                     sample_shape=3), name='beta')(b_key)
    sigma = ppl.random_variable(dists.Uniform(0, 10), name='sigma')(s_key)
    Y = ppl.random_variable(
        dists.Normal(alpha + jnp.dot(x, beta), sigma), name='y')(y_seed)
    return Y


def simple():
    x = np.random.normal(10, 5, size=100)
    y, alpha, beta, sigma = linear_regression_sampler(x)
    px.scatter(x=x, y=y).show()
    from .adaptors.jax_nuts import sample
    logprob = linear_regression_logprob_func(x, y)
    sample(lambda d: logprob(*d), random.PRNGKey(0), (0., 0., 0.), num_samples=1000, num_warmup=1000)


def complicated():
    global sample
    key = random.PRNGKey(0)
    # y = linear_regression_model(random.PRNGKey(100))
    x = jnp.array([[1., 2., 3.], [4., 5., 6.]])
    sample = ppl.joint_sample(linear_regression_model)(key, x)
    print(sample)
    ppl.joint_log_prob(linear_regression_model)(sample, x)
    # log_pdf = ppl.log_prob(linear_regression_model)


if __name__ == "__main__":
    simple()

    #complicated()

