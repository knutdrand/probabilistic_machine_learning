import jax.numpy as jnp
import numpy as np


class MultiLevelModelSpec:
    def __init__(self, model_spec, period_offsets: list[int], reduction_function: callable = jnp.sum):
        self._model_spec = model_spec
        self._period_offsets = period_offsets
        self._period_starts = self._period_offsets[:-1]
        self._period_stops = self._period_offsets[1:]
        assert reduction_function == jnp.sum, 'Only sum reduction is supported'

    def observation_distribution(self, states: jnp.array) -> 'Distribution':
        reduced_states = jnp.array(
            [states[start:stop].sum(axis=0) for start, stop in zip(self._period_starts, self._period_stops)])
        # reduced_states = jnp.diff(jnp.cumsum(states, axis=0)[self._period_offsets], axis=0)
        return self._model_spec.observation_distribution(reduced_states)

    def __getattr__(self, item):
        return getattr(self._model_spec, item)


class MultiLevelModelSpecFactory:
    def __init__(self, model_spec_cls, period_offsets):
        self._model_spec_cls = model_spec_cls
        self._period_offsets = period_offsets

    def __call__(self, *args, **kwargs):
        return MultiLevelModelSpec(self._model_spec_cls(*args, **kwargs), self._period_offsets)

    def __getattr__(self, item):
        return getattr(self._model_spec_cls, item)

    @classmethod
    def from_period_lengths(cls, model_spec, period_lengths: list[int], reduction_function: callable = jnp.sum):
        period_offsets = np.insert(np.cumsum(period_lengths), 0, 0)
        return cls(model_spec, period_offsets)
