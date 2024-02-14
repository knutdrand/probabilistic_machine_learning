import plotly.express as px
import numpy as np
import oryx.distributions as dists
from oryx import core
# import jax.numpy as np
from jax.scipy import stats
import jax

from probabilistic_machine_learning.adaptors.jax_nuts import sample


def linear_regression_sampler(x):
    alpha = np.random.normal(0, 10)
    beta = np.random.normal(0, 10)
    sigma = np.random.uniform(0, 10)
    Y = np.random.normal(alpha + x*beta, sigma)
    return Y, alpha, beta, sigma



def linear_regression_logprob_func(x, y):
    def logprob(alpha, beta, sigma):
        alpha_pdf = stats.norm.logpdf(alpha, 0, 10)
        beta_pdf = stats.norm.logpdf(beta, 0, 10)
        y_pdf = stats.norm.logpdf(y, alpha + x*beta, sigma).sum()
        return y_pdf + alpha_pdf + beta_pdf
    return logprob


def main():
    global logprob_func
    x = np.random.normal(10, 5, size=1000)
    y, alpha, beta, sigma = linear_regression_sampler(x)
    # px.scatter(x=x, y=y).show()
    logprob = linear_regression_logprob_func(x, y)(alpha, beta, sigma)
    logprob_func = linear_regression_logprob_func(x, y)
    log_pdf = lambda args: logprob_func(**args)
    result = sample(log_pdf, jax.random.PRNGKey(0),
                    {'alpha': 0., 'beta': 0., 'sigma': 1.}, num_samples=1000, num_warmup=100)
    print(result)
    import pandas as pd
    df = pd.DataFrame({'alpha': result['alpha'], 'beta': result['beta']})
    #px.line(df, x='alpha', y='beta').show()
    px.histogram(df, x='alpha').show()
    alpha_grid = np.linspace(-20, 20, 100)
    beta_grid = np.linspace(-20, 20, 100)[:, None]
    lpdf = logprob_func(alpha_grid, beta_grid, sigma)
    #px.imshow(lpdf, x=alpha_grid, y=beta_grid).show()
    # grad_func = jax.grad()
    # grad = grad_func((alpha, beta, sigma))
    # print(grad)
    print(alpha, beta, sigma)
    # print(logprob)

def sample_func(random_seed):
    return core.ppl.random_variable(dists.Normal(0, 1))(random_seed)+10

#one_sample = sample_func(jax.random.PRNGKey(0))
#print(one_sample)
#logprob_func = core.ppl.log_prob(sample_func)

def add10(x):
    return x+10

#f = core.inverse(add10)



#print(logprob_func(10.))


main()
