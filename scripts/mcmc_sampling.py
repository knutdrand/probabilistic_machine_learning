from probabilistic_machine_learning.distributions.scipy_distributions import Normal
from probabilistic_machine_learning.monte_carlo_methods.mcmc import MetropolisHastings
import arviz as az

target_distribution = Normal(0, 1)
proposal_distribution = Normal(0, 2)
mh_sampler = MetropolisHastings(target_distribution, proposal_distribution)

samples = mh_sampler.sample(10000)
az.plot_posterior(samples, show=True)
# az.show()
