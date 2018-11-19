from __future__ import division

from functools import partial

from pyro import sample
from pyro.distributions import LogNormal
from torch import tensor


def g1(a):
    return a / (a + 1)


def g2(a):
    return a**2 / (a**2 + a + 1)


def f(mu, N):
    return N ** (0.5) * mu + mu


constants = {
    "t_3k": 1.2,
    "t_2k": 1.2,
    "t_k": 0.003
}

params = {
    "alpha_3k": 500,
    "alpha_2k": 500,
    "alpha_k": 500,
    "nu_3k": .15,
    "nu_2k": .15,
    "nu_k": .15
}


def f_map3k(E1, N_3k):
    total_3k = constants['t_3k']
    alpha_3k = params['alpha_3k']
    nu_3k = params['nu_3k']

    map3k_mu = total_3k * g1(E1 * (alpha_3k/nu_3k))

    return f(map3k_mu, N_3k)


def f_map2k(map3k, N_2k):
    total_2k = constants['t_2k']
    alpha_2k = params['alpha_2k']
    nu_2k = params['nu_2k']

    map2k_mu = total_2k * g2(map3k * (alpha_2k / nu_2k))

    return f(map2k_mu, N_2k)


def f_mapk(map2k, N_k):
    total_k = constants['t_k']
    alpha_k = params['alpha_k']
    nu_k = params['nu_k']

    mapk_mu = total_k * g2(map2k * (alpha_k / nu_k))

    return f(mapk_mu, N_k)


def mapk_signaling(noise_dists):

    E1 = 1.
    N_3k = sample('N_3k', noise_dists['N_3k'])
    N_2k = sample('N_2k', noise_dists['N_2k'])
    N_k = sample('N_k', noise_dists['N_k'])

    map3k = f_map3k(E1, N_3k)
    map2k = f_map2k(map3k,N_2k)
    mapk = f_mapk(map2k, N_k)

    return {
        'map3k': map3k,
        'map2k': map2k,
        'mapk': mapk
    }


if __name__ == "__main__":

    # Noise Distributions
    noise_vars = ['N_3k', 'N_2k', 'N_k']
    noise_prior = {N: LogNormal(0, 10) for N in noise_vars}

    print(mapk_signaling(noise_dists=noise_prior))
