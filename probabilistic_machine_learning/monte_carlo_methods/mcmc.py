from itertools import accumulate

import numpy as np

from ..distribution import Distribution


class MetropolisHastings:
    def __init__(self, target_distribution: Distribution, proposal_distribution: Distribution, burn_in: int = 1000):
        self.target_distribution = target_distribution
        self.proposal_distribution = proposal_distribution
        self.burn_in = burn_in

    def sample(self, n_samples: int):
        x = self._init_state()

        for i in range(self.burn_in):
            x = self._update(x)
        samples = []
        for i in range(n_samples):
            x = self._update(x)
            samples.append(x)
        return np.array(samples)

    def _init_state(self):
        return self.proposal_distribution.sample(1)

    def _update(self, x):
        x_new = x+self.proposal_distribution.sample(1)
        if self._accept(x_new, x):
            return x_new
        else:
            return x

    def _accept(self, x_new, x):
        return self.target_distribution.log_prob(x_new) - self.target_distribution.log_prob(x) > np.log(
            np.random.uniform())
