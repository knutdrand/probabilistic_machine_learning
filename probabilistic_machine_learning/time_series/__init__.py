import dataclasses
from itertools import repeat, count
from ..adaptors.base_adaptor import Variable
from ..adaptors.event import Event
from numbers import Number

import numpy as np
import pymc


class GraphEntry(np.lib.mixins.NDArrayOperatorsMixin):
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            return UfuncEntry(ufunc, inputs, kwargs)
        return NotImplemented


class UfuncEntry(GraphEntry):
    def __init__(self, ufunc, args, kwargs):
        self.ufunc = ufunc
        self.args = args
        self.kwargs = kwargs

    def parents(self):


class IndexedTimeSeries(GraphEntry):
    def __init__(self, ts, index):
        self.ts = ts
        self.index = index


class TimeSeries(GraphEntry):
    _underlying_list = []

    def __init__(self):
        self.relationship = None
        self.init = {}

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


class Scalar(GraphEntry):
    ...


class TimeIndex:
    def __init__(self, base, offset=0):
        self.base = base
        self.offset = offset

    def __add__(self, other):
        return TimeIndex(self.base, self.offset + other)


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


def time_series(n=1):
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
