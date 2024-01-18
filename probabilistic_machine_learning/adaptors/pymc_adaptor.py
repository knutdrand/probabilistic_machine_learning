import pymc as pm
import inspect

from .base_adaptor import Variable, CombinedVariable, Distribution, ModuleWrap


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


def resolve_variable(variable: Variable, variable_dict: dict, name_dict, observed=None):
    names = [name for name, value in name_dict.items() if variable is value]
    name = names[0] if names else None
    if isinstance(variable, CombinedVariable):
        res = variable.op(*[resolve_variable(arg, variable_dict, name_dict) for arg in variable.args])

    elif isinstance(variable, Distribution):
        cls = getattr(pm, variable.__class__.__name__)
        args = [resolve_variable(arg, variable_dict, name_dict) for arg in variable.args if isinstance(arg, Variable)]
        kwargs = {key: resolve_variable(value, variable_dict, name_dict) for key, value in variable.kwargs.items()}
        kwargs['observed'] = observed
        res = cls(name, *args, **kwargs)
    else:
        res = variable
    if name:
        variable_dict[name] = res
    return res


class PymcWrap(ModuleWrap):
    def __call__(self, *args, **kwargs):
        return get_pymc_model(*args, **kwargs)


def get_pymc_model(event):
    name_dict = inspect.stack()[2][0].f_locals
    #variables = [get_variable(name, value) for name, value in frame.f_locals.items() if isinstance(value, Variable)]
    variable_dict = {}
    with pm.Model() as model:
        res = resolve_variable(event.variable, variable_dict, name_dict, observed=event.value)
        pm.sample(1000)
    return model
