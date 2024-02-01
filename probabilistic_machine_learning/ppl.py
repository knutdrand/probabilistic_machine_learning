import numpy as np
from oryx import core
from dataclasses import dataclass
from .adaptors.base_adaptor import Distribution
from .adaptors.event import Event
from typing import Tuple, Union

from .graph_objects import FunctionNode, GraphObject
import oryx.distributions as dist_backend


@dataclass
class Given:
    events: Tuple['Event', ...]


def given(*events):
    return Given(events)


class Graph:

    def __init__(self, events, given_events=None):
        self.events = events
        self.given_events = given_events if given_events is not None else []
        self._realized_variables = {id(event.variable) for event in events}
        self._realized_variables |= {id(event.variable) for event in given_events}
        self._realizations = {id(event.variable): event.value for event in events}
        self._realizations |= {id(event.variable): event.value for event in given_events}

    @classmethod
    def from_events(cls, events, given_events=None):
        return cls(events, given_events)

    def calculate_logprob(self):
        return sum(self._logprob(event) for event in self.events)

    def _logprob(self, event: Event):
        variable = event.variable
        value = event.value
        if isinstance(variable, FunctionNode):
            variable, value = self._get_inverse(variable, value)
        dist = getattr(dist_backend, type(variable).__name__)(*variable.args, **variable.kwargs)
        return dist.log_prob(value)

    def _get_inverse(self, variable: FunctionNode, value):
        transformation, root_distribution = self._get_transformation(variable)
        assert isinstance(root_distribution, Distribution), root_distribution
        inverse_transformation = core.inverse(transformation)
        return root_distribution, inverse_transformation(value)

    def _get_transformation(self, variable: FunctionNode) -> Tuple[callable, Distribution]:
        '''
        Traverse the graph backwards until a distribution is found. Stop if a realized variable is encountered.
        '''
        root_distribution = self._find_root_distribution(variable, base=True)
        transformation = self._get_transformation_or_realization(variable, root_distribution, base=True)
        return transformation, root_distribution

    def _find_root_distribution(self, variable, base=False):
        if self._is_realized(variable) and not base:
            return None
        if isinstance(variable, Distribution):
            return self._resolve_distribution(variable)
        elif isinstance(variable, FunctionNode):
            roots = [self._find_root_distribution(parent) for parent in self._get_parents(variable)]
            roots = [root for root in roots if root is not None]
            if len(roots) > 1:
                raise ValueError('Multiple root distributions found.')
            return roots[0] if roots else None

    def _resolve_distribution(self, distribution: Distribution):
        args = [self._get_transformation_or_realization(arg, distribution) for arg in distribution.args]
        kwargs = {key: self._get_transformation_or_realization(value, distribution) for key, value in distribution.kwargs.items()}
        return distribution.__class__(*args, **kwargs)

    def _is_realized(self, variable):
        return id(variable) in self._realized_variables

    def _get_parents(self, variable):
        return variable.parents()

    def _get_transformation_or_realization(self, variable, root_distriubtion, base: bool=False) -> Union[callable, np.ndarray]:
        if self._is_realized(variable) and not base:
            return self._get_realization(variable)
        if isinstance(variable, FunctionNode):
            args = [self._get_transformation_or_realization(arg, root_distriubtion) for arg in variable.args]
            kwargs = {key: self._get_transformation_or_realization(value, root_distriubtion) for key, value in variable.kwargs.items()}
            new_node = FunctionNode(variable.func, *args, **kwargs)
            return new_node.unary_func
        elif isinstance(variable, Distribution):
            return lambda x: x
        elif not isinstance(variable, GraphObject):
            return variable
        raise ValueError(
            f'Variable {variable} is not realized and is not a function node and is not root distributions {root_distriubtion}')

    def _get_realization(self, variable):
        return self._realizations[id(variable)]


def logprob(*events):
    true_events = [event for event in events if isinstance(event, Event)]
    given_events = [event for event in events if isinstance(event, Given)]
    given_events = [e for ge in given_events for e in ge.events]
    graph = Graph.from_events(true_events, given_events)
    return graph.calculate_logprob()

