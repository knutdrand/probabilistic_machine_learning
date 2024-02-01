import operator

from probabilistic_machine_learning.adaptors.event import Event
from probabilistic_machine_learning.graph_objects import GraphObject


#from .pymc_adaptor import get_pymc_model


class DistWrap:
    def __init__(self, name):
        self.name = name

    def __getattr__(self, name):
        return TimeSeries(self.name, name)

    def __call__(self, *args, **kwargs):
        return get_pymc_model(self.name, *args, **kwargs)


class Variable:
    def __eq__(self, value):
        return Event(self, value)

    def parents(self):
        ...


class CombinedVariable(Variable):
    def __init__(self, op, *args):
        self.op = op
        self.args = args

    def parents(self):
        return [arg for arg in self.args if isinstance(arg, Variable)]


class Distribution(Variable):
    pass


def dist_wrap(name):
    class NewClass(Distribution, GraphObject):
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __mul__(self, other):
            return self._ufunc(operator.mul, other)

        def __add__(self, other):
            return self._ufunc(operator.add, other)

        def _ufunc(self, ufunc, other):
            return CombinedVariable(ufunc, self, other)

        def parents(self):
            return [arg for arg in self.args if isinstance(arg, Variable)] + [value for value in self.kwargs.values() if isinstance(value, Variable)]

    NewClass.__name__ = name
    NewClass.__qualname__ = name
    return NewClass


class ModuleWrap:
    def __getattr__(self, name):
        return dist_wrap(name)
