from typing import Protocol

import np as np


class Distribution(Protocol):
    def sample(self, shape: tuple) -> np.ndarray:
        ...

    def log_prob(self, value: np.ndarray) -> np.ndarray:
        ...
