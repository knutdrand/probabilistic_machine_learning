import numpy as np
import pandas as pd
import pymc as pm
import pytest
from pymc import Model, Normal


@pytest.fixture
def data_1():
    RANDOM_SEED = 8927
    rng = np.random.default_rng(RANDOM_SEED)

    size = 200
    true_intercept = 1
    true_slope = 2

    x = np.linspace(0, 1, size)
    # y = a + b*x
    true_regression_line = true_intercept + true_slope * x
    # add noise
    y = true_regression_line + rng.normal(scale=0.5, size=size)

    data = pd.DataFrame(dict(x=x, y=y))
    return data

@pytest.fixture
def numpy_data_1(data_1):
    return data_1['x'].to_numpy(), data_1['y'].to_numpy()

@pytest.fixture
def example_1(data_1):
    '''
    https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/GLM_linear.html#glm-linear
    '''
    x, y = data_1['x'], data_1['y']
    with Model() as model:  # model specifications in PyMC are wrapped in a with-statement
        # Define priors
        sigma = pm.HalfCauchy("sigma", beta=10)
        intercept = pm.Normal("intercept", 0, sigma=20)
        slope = pm.Normal("slope", 0, sigma=20)

        # Define likelihood
        likelihood = Normal("Y", mu=intercept + slope * x, sigma=sigma, observed=y)

        # Inference!
        # draw 3000 posterior samples using NUTS sampling
        # idata = sample(3000)
    return model
