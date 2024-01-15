from ..distributions.scipy_distributions import Normal, HalfCauchy

x, y, m = 0, 0, 0
# intercept priors
im = Normal(mu=0, sigma=10)
iy = Normal(mu=0, sigma=10)
# slope priors
a = Normal(mu=0, sigma=10)
b = Normal(mu=0, sigma=10)
cprime = Normal(mu=0, sigma=10)

# noise priors
σm = HalfCauchy(1)
σy = HalfCauchy(1)

# likelihood
M = Normal(mu=im + a * x, sigma=σm)
Y = Normal(mu=iy + b * m + cprime * x, sigma=σy)

# calculate quantities of interest
indirect_effect = a * b
total_effect = a * b + cprime

sample(P(indirect_effect, total_effect, given=(M == m) & (Y == y)))
