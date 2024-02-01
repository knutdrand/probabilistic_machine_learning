'''
https://discourse.pymc.io/t/time-series-implementation-questions/10653
'''
import aeppl
import aesara
import aesara.tensor as at
import numpy as np
srng = at.random.RandomStream()


def example():
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

    logprob, value_variables = aeppl.joint_logprob(Gamma_rv, Y_rv, S_rv)
    sample_fn = aesara.function([M_tt, N_tt, mus_tt], [S_rv, Y_rv, Gamma_rv])
    print(sample_fn(3, 10, [[1., 10, 100], [200., 300., 400]]))


def my_example():
    drift = at.scalar("drift")
    diffusion = at.scalar("diffusion")
    dt = at.scalar("dt")
    x0 = srng.normal()
    n_steps = at.iscalar("n_steps")

    def scan_fn(x_tm1):
        # x_t = x_tm1 + drift * dt + diffusion * srng.normal()
        x_t = x_tm1 + 2.0 * 1.0 + 0.5 * srng.normal()
        return x_t

    x_rv, _ = aesara.scan(
        fn=scan_fn,
        outputs_info=[x0],
        n_steps=n_steps,
        strict=True,
        name="scan_rv",
    )

    #logprob, value_variables = aeppl.joint_logprob(x_rv, realized={})
    logprobs = aeppl.joint_logprob(x_rv, x0)
    sample_fn = aesara.function([n_steps], [x_rv])
    sample = sample_fn(100)
    print(sample[0])
    import matplotlib.pyplot as plt
    plt.plot(sample[0]);
    plt.show()

def example3():
    X = srng.normal()
    Y = srng.normal()+2

    logprob, value_variables = aeppl.joint_logprob(X, Y)
    logprob_fn = aesara.function(value_variables, [logprob])
    print(logprob_fn(1, 2))

if __name__ == '__main__':
    example3()
