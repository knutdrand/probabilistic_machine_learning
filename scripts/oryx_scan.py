import jax
import jax.numpy as jnp
from oryx.core.ppl import random_variable, joint_log_prob, joint_sample
import oryx.distributions as tfd
from oryx.core import sow


def scan_sample(key):
    key, *keys = jax.random.split(key, 11)
    diffs = tfd.Normal(0., 1.).sample(seed=key, sample_shape=(10,))
    sow(diffs, tag='random_variable', name='diffs')

    def f(state, diff):
        new_state = state + diff
        return new_state, new_state

    end = sow(jax.lax.scan(f, 0., diffs)[0], tag='random_variable', name='end')
    return end

#x = scan_sample(jax.random.PRNGKey(0))
join_sample_func = joint_sample(scan_sample)
d = join_sample_func(jax.random.PRNGKey(0))
print(d)
lp = joint_log_prob(scan_sample)
print(lp(d))
