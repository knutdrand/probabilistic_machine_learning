from dataclasses import dataclass

from .distribution import Distribution
from .graph_objects import GraphObject, FunctionNode


@dataclass
class SSMDistribution(Distribution):
    transition_function: callable
    initial_state: Distribution

    def log_prob(self, value):
        pass

def transition_function(x):
    return dists.Normal(x, 1)






class TimeSeriesDistribution(GraphObject):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._dist = None


    def __setitem__(self, key, value):
        if isinstance(key, TimeIndex):
            self.relationship = (key, value)
        elif isinstance(key, Number):
            self.init = {key: value}

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
