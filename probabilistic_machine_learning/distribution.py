from typing import Protocol

import np as np


class Distribution(Protocol):
    def sample(self, shape: tuple) -> np.ndarray:
        ...

    def log_prob(self, value: np.ndarray) -> np.ndarray:
        ...


class EmpiricalDistribution:
    def __init__(self, samples: np.ndarray, weights: np.ndarray = None):
        self.samples = samples
        self.weights = weights

    def sample(self, shape: tuple) -> np.ndarray:
        return np.random.choice(self.samples, size=shape, p=self.weights)

    def log_prob(self, value: np.ndarray) -> np.ndarray:
        return np.log(self.weights[self.samples == value])
