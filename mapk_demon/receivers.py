from __future__ import division

from functools import partial

import numpy as np
import pyro
from pyro import sample
from pyro.distributions import Normal, Uniform
from torch import tensor
from mapk_demon.transmitters import mapk_signaling
from pyro.infer.mcmc import MCMC
from pyro.infer.mcmc.nuts import NUTS
from pyro.infer import EmpiricalMarginal
from causal_demon.inference import infer_dist
import torch
import matplotlib.pyplot as plt

def infer_mcmc(model):
    nuts_kernel = NUTS(model, adapt_step_size=True)
    return MCMC(nuts_kernel, num_samples=500, warmup_steps=300)


def g1(a):
    return a / (a + 1)


def g2(a):
    return a ** 2 / (a ** 2 + a + 1)


def f(mu, N):
    return N * mu ** (0.5) + mu


# Parameters

params = {
        "alpha_3k": Normal(500, 1),
        "alpha_2k": Normal(500, 1),
        "alpha_k": Normal(500, 1),
        "nu_3k": Normal(.15, .05),
        "nu_2k": Normal(.15, .05),
        "nu_k": Normal(.15, .05)
}
'''
params = {
    "alpha_3k": 500,
    "alpha_2k": 500,
    "alpha_k": 500,
    "nu_3k": .15,
    "nu_2k": .15,
    "nu_k": .15
}
'''

hyperparameters = {
    "t_3k": 1.2,
    "t_2k": 1.2,
    "t_k": 0.003
}


def f_map3k(E1, N_3k, params):
    total_3k = hyperparameters['t_3k']
    alpha_3k = sample("alpha_3k",params['alpha_3k'])
    nu_3k = sample("nu_3k", params['nu_3k'])

    map3k_mu = total_3k * g1(E1 * (alpha_3k/nu_3k))

    map3k = sample("map3k", Normal(f(map3k_mu, N_3k), 1e-6))
    return map3k


def f_map2k(map3k, N_2k, params):
    total_2k = hyperparameters['t_2k']
    alpha_2k = sample("alpha_2k", params['alpha_2k'])
    nu_2k = sample("nu_2k", params['nu_2k'])

    map2k_mu = total_2k * g2(map3k * (alpha_2k / nu_2k))

    map2k = sample("map2k", Normal(f(map2k_mu, N_2k), 1e-6))
    return map2k


def f_mapk(map2k, N_k, params):
    total_k = hyperparameters['t_k']
    alpha_k = sample("alpha_k",params['alpha_k'])
    nu_k = sample("nu_k", params['nu_k'])

    mapk_mu = total_k * g2(map2k * (alpha_k / nu_k))
    mapk = sample("mapk", Normal(f(mapk_mu, N_k), 1e-6))
    return mapk


def mapk_signaling_receiver(noise_dists):

    with pyro.iarange("model"):
        E1 = sample('E1', Uniform(1.5e-5, 6e-5))
        N_3k = sample('N_3k', noise_dists['N_3k'])
        N_2k = sample('N_2k', noise_dists['N_2k'])
        N_k = sample('N_k', noise_dists['N_k'])

    map3k = f_map3k(E1, N_3k, params)
    map2k = f_map2k(map3k,N_2k, params)
    mapk = f_mapk(map2k, N_k, params)

    return {
        'map3k': map3k,
        'map2k': map2k,
        'mapk': mapk
    }

def plot(data):
    plt.boxplot(data['erk'])
    plt.title("Empirical Distribution of Erk")
    plt.xlabel("Condition")
    plt.ylabel("Concentration")
    plt.show()


if __name__ == "__main__":

    # Noise Distributions
    noise_vars = ['N_3k', 'N_2k', 'N_k']
    noise_priors = {N: Normal(0, 1) for N in noise_vars}

    parameters = ["alpha_3k","alpha_2k","alpha_k","nu_3k","nu_2k","nu_k"]

    receiver_model = mapk_signaling_receiver(noise_dists=noise_priors)

    # Simulate data from transmitter model
    evidence = []

    for i in range(10):
        evidence.append(mapk_signaling(noise_dists=noise_priors))

    # condition the receiver model with data
    conditioned = pyro.condition(mapk_signaling_receiver, data = evidence)

    print(conditioned(noise_priors))

    # mcmc inference
    #nuts_kernel = NUTS(conditioned, adapt_step_size=True)
    #mcmc_run = MCMC(nuts_kernel, num_samples=50, warmup_steps=10).run(noise_priors)

    # Importance Sampling
    posterior = pyro.infer.Importance(conditioned, num_samples=100).run(noise_priors)

    #print(posterior)
    posteriors = {
        n: EmpiricalMarginal(posterior, sites=n)
        for n in parameters
    }

    print("Alpha_3k inference")
    alpha_3k_marginal = posteriors['alpha_3k']
    alpha_3k_samples = [alpha_3k_marginal().item() for _ in range(500)]

    print("mean:",np.mean(alpha_3k_samples))
    print("std:",np.std(alpha_3k_samples))
