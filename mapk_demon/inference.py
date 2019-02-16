import pyro
from pyro.infer.mcmc import MCMC
from pyro.infer.mcmc.nuts import HMC


def infer_dist(prog, n_dist, type='mcmc'):
    """Obtain the unique distribution entailed by a SCM program.

    Do inference on a SCM program and obtain a object representing the
    probability distribution entailed by the SCM.

    This implementation depends on simple importance sampling with 5000
    samples.

    `prog`: the subroutine encoding the SCM.
    `n_dist`: a dictionary containing distributions for each
    noise object.
    """
    if type == 'mcmc':
        hmc_kernel = HMC(prog, step_size=0.9, num_steps=4)
        posterior = MCMC(hmc_kernel, num_samples=1000, warmup_steps=50).run(n_dist)
        return posterior

    else:
        return 0
