from typing import Protocol

import numpy as np
import scipy.special

import dataclasses
from ..distribution import Distribution, EmpiricalDistribution
distribution = dataclasses.dataclass

class ProposalDistribution(Protocol):
    def init_sample(self, n_samples: int = 1) -> np.ndarray:
        ...

    def init_logprob(self, value: np.ndarray) -> np.ndarray:


    def conditional_sample(self, prev_values: np.ndarray) -> np.ndarray:
        ...

    def conditional_log_prob(self, value: np.ndarray, prev_values: np.ndarray) -> np.ndarray:
        ...


@distribution
class StateDistribution:
    state: np.ndarray



class SequentialImportanceSampling:
    def __init__(self, target_distribution: Distribution,  proposal_distribution: ProposalDistribution):
        self.proposal_distribution = proposal_distribution
        self.target_distribution = target_distribution

    def calculate(self, T: int, n_samples: int) -> EmpiricalDistribution:
        normalized_weights, w, z_lists = self._initialize(n_samples)
        for t in range(T):
            for i in range(n_samples):
                zs = z_lists[i]
                zs.append(self.proposal_distribution.sample())
                incremental_weight = self.get_incremental_weight(zs)
                w[i] += incremental_weight
            self.proposal_distribution = dataclasses.replace(state=z_lists)
            t = scipy.special.logsumexp(w)
            normalized_weights = w-t
        return EmpiricalDistribution(z_lists, normalized_weights)

    def get_incremental_weight(self, zs):
        return (self.target_distribution.log_prob(zs) - self.proposal_distribution.log_prob(
            zs[:-1]) - self.proposal_distribution.conditional_log_prob(zs[-1], zs[:-1]))

    def _initialize(self, n_samples):
        init_z = self.proposal_distribution.sample(n_samples)
        z_lists = [[z] for z in init_z]
        w = self.target_distribution.log_prob(init_z) - self.proposal_distribution.init_log_prob(init_z)
        t = scipy.special.logsumexp(w)
        normalized_weights = w - t
        return normalized_weights, w, z_lists


class SISR(SequentialImportanceSampling):
    def calculate(self, T, n_samples) -> EmpiricalDistribution:
        normalized_weights, w, z_lists = self._initialize(n_samples)
        for t in range(T):
            z_lists = self._resample(z_lists, normalized_weights)
            for i in range(n_samples):
                zs = z_lists[i]
                zs.append(self.proposal_distribution.conditional_sample(zs))
                w[i] = self.get_incremental_weight(zs)
            normalized_weights = w-scipy.special.logsumexp(w)
        return EmpiricalDistribution(z_lists, normalized_weights)

    def _resample(self, z_lists, normalized_weights):
        '''This should be done with either stratified or systematic resampling'''
        indices = np.random.choice(len(z_lists), len(z_lists), p=np.exp(normalized_weights))
        return [z_lists[i] for i in indices]


class Bootstrap:
    def __init__(self, target_distribution: Distribution, observed_valiues: np.ndarray):
        self.target_distribution = target_distribution
        self.observed_values = observed_valiues
        self.T = len(observed_valiues)

    def get_incremental_weight(self, z):
        return self.observation_distribution(z).log_prob(self.observed_values[len(z)])


    def calculate(self, n_samples):
        pass

