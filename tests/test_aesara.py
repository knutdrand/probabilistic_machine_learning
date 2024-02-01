import aesara
import aeppl

import aesara
from aesara import tensor as at

from aeppl import joint_logprob, pprint


def test_1():
    srng = at.random.RandomStream()

    # A simple scale mixture model
    S_rv = srng.invgamma(0.5, 0.5)
    Y_rv = srng.normal(0.0, at.sqrt(S_rv))

    # Compute the joint log-probability
    logprob, (y, s) = joint_logprob(Y_rv, S_rv)

def test_2():
    srng = at.random.RandomStream()

    N_tt = at.iscalar("N")
    M_tt = at.iscalar("M")
    mus_tt = at.matrix("mus_t")

    sigmas_tt = at.ones((N_tt,))
    Gamma_rv = srng.dirichlet(at.ones((M_tt, M_tt)), name="Gamma")

    def scan_fn(mus_t, sigma_t, Gamma_t):
        S_t = srng.categorical(Gamma_t[0], name="S_t")
        Y_t = srng.normal(mus_t[S_t], sigma_t, name="Y_t")
        return Y_t, S_t

    (Y_rv, S_rv), _ = aesara.scan(
        fn=scan_fn,
        sequences=[mus_tt, sigmas_tt],
        non_sequences=[Gamma_rv],
        outputs_info=[{}, {}],
        strict=True,
        name="scan_rv",
    )

def test_3():
    srng = at.random.RandomStream()
    a = at.vector("a")
    b = at.vector("b")
    b[1:]= srng.normal(a[:-1], 1)
