import operator
import pymc as pm
import inspect

class DistWrap:
    def __init__(self, name):
        self.name = name

    def __getattr__(self, name):
        return TimeSeries(self.name, name)

    def __call__(self, *args, **kwargs):
        return get_pymc_model(self.name, *args, **kwargs)


class Event:
    def __init__(self, variable, value):
        self.variable = variable
        self.value = value


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
    class NewClass(Distribution):
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __mul__(self, other):
            return self._ufunc(operator.mul, other)

        def __add__(self, other):
            return self._ufunc(operator.add, other)

        def _ufunc(self, ufunc, other):
            return CombinedVariable(ufunc, self, other)

    NewClass.__name__ = name
    NewClass.__qualname__ = name
    return NewClass


class ModuleWrap:
    def __getattr__(self, name):
        return dist_wrap(name)

def get_variabels(variable_dict: dict):
    variable_dict = {key: value for key, value in variable_dict.items() if isinstance(value, Variable)}
    new_dict = {}
    new_variables = set()
    while variable_dict:
        for key, value in variable_dict.items():
            if not all(parent in new_variables for parent in value.parents()):
                continue
            if isinstance(value, CombinedVariable):
                new_variable = value.op(*value.args)
                new_dict[key] = new_variable

            if not value.parents():
                new_dict[key] = value
                del variable_dict[key]
                for variable in variable_dict.values():
                    if value in variable.parents():
                        variable.parents().remove(value)

def resolve_variable(variable: Variable, variable_dict: dict, name_dict):
    names = [name for name, value in name_dict.items() if variable is value]
    name = names[0] if names else None
    if isinstance(variable, CombinedVariable):
        res = variable.op(*[resolve_variable(arg, variable_dict, name_dict) for arg in variable.args])

    elif isinstance(variable, Distribution):
        cls = getattr(pm, variable.__class__.__name__)
        args = [resolve_variable(arg, variable_dict, name_dict) for arg in variable.args if isinstance(arg, Variable)]
        kwargs = {key: resolve_variable(value, variable_dict, name_dict) for key, value in variable.kwargs.items()}
        res = cls(name, *args, **kwargs)
    else:
        res = variable
    if name:
        variable_dict[name] = res
    return res


def get_pymc_model(event):
    name_dict = inspect.stack()[1][0].f_locals
    #variables = [get_variable(name, value) for name, value in frame.f_locals.items() if isinstance(value, Variable)]
    variable_dict = {}
    with pm.Model() as model:
        res = resolve_variable(event.variable, variable_dict, name_dict)
    return model
