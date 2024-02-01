import aesara
import aesara.tensor as at
import numpy as np

# define tensor variables

def example():
    X = at.vector("X")
    W = at.matrix("W")
    b_sym = at.vector("b_sym")
    U = at.matrix("U")
    Y = at.matrix("Y")
    V = at.matrix("V")
    P = at.matrix("P")

    results, updates = aesara.scan(lambda y, p, x_tm1:
                                   at.tanh(at.dot(x_tm1, W) + at.dot(y, U) + at.dot(p, V)),
                                   sequences=[Y, P[::-1]], outputs_info=[X])
    compute_seq = aesara.function(inputs=[X, W, Y, U, P, V], outputs=results)

    # test values
    x = np.zeros((2), dtype=aesara.config.floatX)
    x[1] = 1
    w = np.ones((2, 2), dtype=aesara.config.floatX)
    y = np.ones((5, 2), dtype=aesara.config.floatX)
    y[0, :] = -3
    u = np.ones((2, 2), dtype=aesara.config.floatX)
    p = np.ones((5, 2), dtype=aesara.config.floatX)
    p[0, :] = 3
    v = np.ones((2, 2), dtype=aesara.config.floatX)

    print(compute_seq(x, w, y, u, p, v))

    # comparison with numpy
    x_res = np.zeros((5, 2), dtype=aesara.config.floatX)
    x_res[0] = np.tanh(x.dot(w) + y[0].dot(u) + p[4].dot(v))
    for i in range(1, 5):
        x_res[i] = np.tanh(x_res[i - 1].dot(w) + y[i].dot(u) + p[4 - i].dot(v))
    print(x_res)


def my_example():
    fn = lambda x: x+2
    X = at.vector("X")
    results, updates = aesara.scan(fn, outputs_info=[X], n_steps=10)
    compute_seq = aesara.function(inputs=[X], outputs=results)
    print(compute_seq(np.array([1.])))

my_example()
