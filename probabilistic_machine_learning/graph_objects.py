from functools import partial

import numpy as np
import jax.numpy as jnp
import jax.scipy.special as jsp
from .adaptors.event import Event


class GraphObject(np.lib.mixins.NDArrayOperatorsMixin):
    '''Save computations in a graph'''
    def __get_jax_func(self, func):
        if hasattr(jnp, func.__name__):
            return getattr(jnp, func.__name__)
        elif hasattr(jsp, func.__name__):
            return getattr(jsp, func.__name__)
        else:
            raise ValueError(f'Function {func.__name__} is not implemented in JAX')

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc == np.equal:
            return Event(*inputs)
        jax_func = self.__get_jax_func(ufunc)
        return FunctionNode(jax_func,
                            *inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        return FunctionNode(
            self.__get_jax_func(func), *args, **kwargs)

    def parents(self):
        return []


class FunctionNode(GraphObject):
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def parents(self):
        p = [arg for arg in self.args if isinstance(arg, GraphObject)]
        p += [arg for arg in self.kwargs.values() if isinstance(arg, GraphObject)]
        return p

    @property
    def unary_func(self):
        arg_indices = [i for i, arg in enumerate(self.args) if  callable(arg)]
        arg_keys = [key for key, value in self.kwargs.items() if callable(value)]
        if len(arg_indices) > 1 or len(arg_keys) > 1:
            raise ValueError('Only one argument can be a graph object.')
        if arg_indices:
            i = arg_indices[0]
            def func(x):
                new_args = (self.args[:i] + (self.args[i](x),) + self.args[i + 1:])
                return self.func(*new_args, **self.kwargs)
            return func
        elif arg_keys:
            variable_key = arg_keys[0]
            return lambda x: self.func(*self.args, **{key: (value if key!=variable_key else value(x)) for key, value in self.kwargs.items()})
        else:
            return self.func(*self.args, **self.kwargs)
