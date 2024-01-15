import numpy as np

from ..distribution import Distribution


class DirectImportanceSampling:
    def __init__(self, target_distribution: Distribution, proposal_distribution: Distribution):
        self.target_distribution = target_distribution
        self.proposal_distribution = proposal_distribution

    def E(self, function, n_samples=100):
        samples = self.proposal_distribution.sample(n_samples)
        weights = np.exp(self.target_distribution.log_prob(samples) - self.proposal_distribution.log_prob(samples))
        f = function(samples)
        return np.mean(f * weights)


class SelfNormalizedImportanceSampling(DirectImportanceSampling):
    def E(self, function, n_samples=100):
        samples = self.proposal_distribution.sample(n_samples)
        weights = np.exp(self.target_distribution.log_prob(samples) - self.proposal_distribution.log_prob(samples))
        f = function(samples)
        return np.sum(f * weights) / np.sum(weights)

