import dataclasses
from itertools import repeat, count, chain
from typing import Tuple, Dict, Any

from ..adaptors.base_adaptor import Variable, Distribution
from ..adaptors.event import Event
from numbers import Number

import numpy as np
import pymc
from ..graph_objects import GraphObject, FunctionNode
from ..ppl import get_dist, instanciate_dist

GraphEntry = GraphObject


class IndexedTimeSeries(GraphEntry):
    def __init__(self, ts, index):
        self.ts = ts
        self.index = index


class TimeSeries(GraphEntry):
    _underlying_list = []

    def __init__(self):
        self.relationship: Tuple[int, IndexedTimeSeries] = None
        self.init: Dict[int, Any] = {}

    def sample_generator(self):
        l = []
        key, value = self.relationship
        my_offset = key.offset
        other_offset = value.index.offset
        lag = my_offset-other_offset
        for i in count(0):
            if i in self.init:
                l.append(self.init[i])
            else:
                if i-lag>= value.ts.size:
                    break
                l.append(value.ts[i-lag])
        return np.array(l)

    def __getitem__(self, item):
        if isinstance(item, TimeIndex):
            return IndexedTimeSeries(self, item)
        return self._underlying_list[item]

    def __setitem__(self, key, value):
        if isinstance(key, TimeIndex):
            self.relationship = (key, value)
        elif isinstance(key, Number):
            self.init = {key: value}

    def _get_indexed_distribution(self, offset: int, dist: Distribution):
        min_arg_offset = min(a.index.offset for a in chain(dist.args, dist.kwargs.values()) if isinstance(a, IndexedTimeSeries))
        get_slice = lambda x: slice(None, -(offset-x))
        indexed_args = tuple(a if not isinstance(a, IndexedTimeSeries) else a.ts[get_slice(a.index.offset)] for a in dist.args)
        indexed_kwargs = {key: value if not isinstance(value, IndexedTimeSeries) else value.ts[get_slice(value.index.offset)] for key, value in dist.kwargs.items()}
        return dist.__class__(*indexed_args, **indexed_kwargs)

    def log_prob(self, value):
        key, its = self.relationship

        init_lp = sum(instanciate_dist(dist).log_prob(value[i]) for
                      i, dist in self.init.items())
        offset = key.offset# - its.index.offset
        relative_dist = self._get_indexed_distribution(offset, its)
        lp = instanciate_dist(relative_dist).log_prob(value[:-len(self.init)])
        return init_lp + lp


class Scalar(GraphEntry):
    ...


class TimeIndex:
    def __init__(self, base, offset=0):
        self.base = base
        self.offset = offset

    def __add__(self, other):
        return TimeIndex(self.base, self.offset + other)

    def __sub__(self, other):
        return TimeIndex(self.base, self.offset - other)

class RandomTimeSeries(GraphEntry):
    def __setitem__(self, key, value):
        if isinstance(key, TimeIndex):
            assert isinstance(value, IndexedDistribution)
        else:
            assert False


class TimeSeriesData(GraphEntry):
    def __getitem__(self, item):
        if isinstance(item, TimeIndex):
            return IndexedTimeSeries(self, item)


def pymc_time_series(transition_function, init_distribution=None, n_steps=10, beta=2, observed=None):
    with pymc.Model() as model:
        # check if beta is callable
        if callable(beta):
            beta = beta()
        if init_distribution is None:
            init_distribution = pymc.Normal('X_0', 0, 2)
        xs = [init_distribution]
        for t in range(1, n_steps):
            xs.append(transition_function(xs[-1], t, beta=beta, observed=observed))
    return model


def time_series(n=1, T=10):
    return TimeIndex(0), (TimeSeries() for _ in range(n))


def scalar():
    return Scalar()

@dataclasses.dataclass
class Conditions:
    events: list


def sample(*args):
    normal_args = [arg for arg in args if isinstance(arg, GraphEntry)]
    given = [arg for arg in args if not isinstance(arg, Conditions)]
    assert len(given) <= 1, given
    evaluated = {}
    if len(given):
        for event in given[0].events:
            evaluated[id(event.variable)] = event.value

    graph = get_graph(normal_args, events=given[0].events)


def given(*events):
    return Conditions(events)
