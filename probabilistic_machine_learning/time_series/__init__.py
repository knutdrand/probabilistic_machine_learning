import pymc


def time_series(*args):
    return args, 0


class TimeSeries:
    _underlying_list = []



class TimeIndex:
    def __init__(self, base, offset=0):
        self.base = base
        self.offset = offset

    def __add__(self, other):
        return TimeIndex(self.base, self.offset + other)


class RandomTimeSeries:
    def __setitem__(self, key, value):
        if isinstance(key, TimeIndex):
            assert isinstance(value, IndexedDistribution)
        else:
            assert False



class TimeSeriesData:
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
