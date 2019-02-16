'''
svi_test
Author: Kaushal Paneri
Project: causal_demon
Date of Creation: 1/2/19
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import pyro
from pyro import sample
from pyro.distributions import Normal, Uniform
from torch import tensor
from mapk_demon.transmitters import mapk_signaling
from pyro.infer.mcmc import MCMC
from pyro.infer.mcmc.nuts import NUTS, HMC
from pyro.infer import EmpiricalMarginal
#from causal_demon.inference import infer_dist
import torch
import matplotlib.pyplot as plt
import json
from torch.distributions import constraints
from tqdm import tqdm

import pyro.optim


def test_transmitter(inp):
    al_m = torch.tensor(inp)
    al_s = torch.tensor(1.)

    nu_m = torch.tensor(.15)
    nu_s = torch.tensor(.05)

    params = {
        "alpha_3k": Normal(al_m, al_s),
        "alpha_2k": Normal(al_m, al_s),
        "alpha_k": Normal(al_m, al_s),
        "nu_3k": Normal(nu_m, nu_s),
        "nu_2k": Normal(nu_m, nu_s),
        "nu_k": Normal(nu_m, nu_s)
    }
    return {
        'alpha_3k': sample("alpha_3k", params['alpha_3k']),
        'alpha_2k': sample("alpha_2k", params['alpha_2k']),
        'alpha_k': sample("alpha_k", params['alpha_k'])
    }


def test_receiver(inp):
    al_m = sample("al_m", Normal(inp, 1))
    al_s = torch.tensor(1.)

    nu_m = torch.tensor(.15)
    nu_s = torch.tensor(.05)

    params = {
        "alpha_3k": Normal(al_m, al_s),
        "alpha_2k": Normal(al_m, al_s),
        "alpha_k": Normal(al_m, al_s),
        "nu_3k": Normal(nu_m, nu_s),
        "nu_2k": Normal(nu_m, nu_s),
        "nu_k": Normal(nu_m, nu_s)
    }
    return {
        'alpha_3k': sample("alpha_3k", params['alpha_3k']),
        'alpha_2k': sample("alpha_2k", params['alpha_2k']),
        'alpha_k': sample("alpha_k", params['alpha_k'])
    }


def receiver_guide(inp):
    al_m = pyro.param("al_m", Normal(inp, 1))
    al_s = pyro.param("al_s", torch.tensor(1.), constraint=constraints.positive)

    nu_m = pyro.param("nu_m", torch.tensor(.15))
    nu_s = pyro.param('nu_m', torch.tensor(.05), constraint=constraints.positive)

    params = {
        "alpha_3k": Normal(al_m, al_s),
        "alpha_2k": Normal(al_m, al_s),
        "alpha_k": Normal(al_m, al_s),
        "nu_3k": Normal(nu_m, nu_s),
        "nu_2k": Normal(nu_m, nu_s),
        "nu_k": Normal(nu_m, nu_s)
    }

    return {
        'alpha_3k': sample("alpha_3k", params['alpha_3k']),
        'alpha_2k': sample("alpha_2k", params['alpha_2k']),
        'alpha_k': sample("alpha_k", params['alpha_k'])
    }


if __name__ == "__main__":
    sample_size = 10000

    ground = 500.
    inp = 700.
    output = test_transmitter(ground)

    # Noise Distributions
    noise_vars = ['N_3k', 'N_2k', 'N_k']
    noise_priors = {N: Normal(0, 1) for N in noise_vars}
    parameters = ["alpha_3k", "alpha_2k", "alpha_k", "nu_3k", "nu_2k", "nu_k"]

    # Simulate data from transmitter model
    evidence = []

    for i in range(sample_size):
        evidence.append(test_transmitter(ground))


    # parameter settings
    print(evidence)

    #receiver_model = mapk_signaling_receiver(noise_dists=noise_priors)

    # condition the receiver model with data
    conditioned = pyro.condition(test_receiver, data = output)

    print("conditioned:: ", conditioned(inp))

    svi = pyro.infer.SVI(model=conditioned,
                         guide=receiver_guide,
                         optim=pyro.optim.SGD({"lr": 0.0001, "momentum": 0.1}),
                         loss=pyro.infer.Trace_ELBO())

    losses, a = [], []
    num_steps = 2500
    for t in tqdm(range(num_steps)):
        losses.append(svi.step(inp))
        a.append(pyro.param("al_m").item())
        print(pyro.get_param_store().get_param('al_m'))

    print(a)
    print(losses)
