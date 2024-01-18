from probabilistic_machine_learning.adaptors.pymc_adaptor import PymcWrap
from .fixtures import *
from .models import Models

_wrap = PymcWrap()

models= Models(_wrap)

def test_get_mymc_model(numpy_data_1, example_1):
    x, y = numpy_data_1
    py_mc_model = models.linear_regression(x, y)
    assert set(py_mc_model.named_vars.keys()) == set(example_1.named_vars.keys())



