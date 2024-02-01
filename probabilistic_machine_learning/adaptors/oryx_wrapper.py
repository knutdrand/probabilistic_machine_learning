import oryx
import jax.numpy as np
def my_function(x):
    return np.exp(x)+3

inverse = oryx.core.inverse(my_function)
print(inverse(10.))
