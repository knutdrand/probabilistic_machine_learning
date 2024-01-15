from ..distributions.scipy_distributions import Normal, Beta, Binomial
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import pymc as pm

from scipy.special import expit
az.style.use("arviz-darkgrid")
rng = np.random.default_rng(1234)



# true params
beta0_true = 0.7
beta1_true = 0.4
# number of yes/no questions
n = 20

sample_size = 30
x = np.linspace(-10, 20, sample_size)
mu_true = beta0_true + beta1_true * x
p_true = expit(mu_true)
y = rng.binomial(n, p_true)

beta0 = Normal(mu=0, sigma=1)
beta1 = Normal(mu=0, sigma=1)
mu = beta0 + beta1 * x
p = expit(mu)
Y = Binomial(n=n, p=p)

sample(P(beta0, beta1, given=Y==y))
