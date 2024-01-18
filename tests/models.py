class Models:
    def __init__(self, wrap):
        self._wrap = wrap

    def linear_regression(self, x, y):
        wrap = self._wrap
        sigma = wrap.HalfCauchy(beta=10)
        intercept = wrap.Normal(0, sigma=20)
        slope = wrap.Normal(0, sigma=20)
        Y = wrap.Normal(mu=intercept + slope * x, sigma=sigma)
        return wrap(Y == y)
