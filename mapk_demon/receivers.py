from __future__ import division

from functools import partial

import pyro
from pyro import sample
from pyro.distributions import Normal, Uniform, Delta

import pyro.optim
import torch


def g1(a):
    return a / (a + 1)


def g2(a):
    return a ** 2 / (a ** 2 + a + 1)


def f(mu, N):
    return N * mu ** (0.5) + mu


hyperparameters = {
    "t_3k": 1.2,
    "t_2k": 1.2,
    "t_k": 0.003
}


def f_map3k(E1, N_3k, params, mode):
    total_3k = hyperparameters['t_3k']
    alpha_3k = params['alpha_3k'].rsample()
    nu_3k = params['nu_3k'].rsample()

    map3k_mu = total_3k * g1(E1 * (alpha_3k/nu_3k))

    if mode == 'companion':
        return sample("map3k", Normal(f(map3k_mu, N_3k), 1.))
    else:
        return sample("map3k", Delta(f(map3k_mu, N_3k)))


def f_map2k(map3k, N_2k, params, mode):
    total_2k = hyperparameters['t_2k']
    alpha_2k = params['alpha_2k'].rsample()
    nu_2k = params['nu_2k'].rsample()

    map2k_mu = total_2k * g2(map3k * (alpha_2k / nu_2k))

    if mode == 'companion':
        return sample("map2k", Normal(f(map2k_mu, N_2k), 1.))
    else:
        return sample("map2k", Delta(f(map2k_mu, N_2k)))


def f_mapk(map2k, N_k, params, mode):
    total_k = hyperparameters['t_k']
    alpha_k = params['alpha_k'].rsample()
    nu_k = params['nu_k'].rsample()

    mapk_mu = total_k * g2(map2k * (alpha_k / nu_k))

    if mode == 'companion':
        return sample("mapk", Normal(f(mapk_mu, N_k), 1.))
    else:
        return sample("mapk", Delta(f(mapk_mu, N_k)))


def mapk_companion(noise_dists):
    mode = 'companion'
    # Parameters
    al_m = torch.tensor(700.)
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

    with pyro.iarange("model"):
        E1 = Uniform(1.5e-5, 10.).rsample()
        N_3k = sample('N_3k', noise_dists['N_3k'])
        N_2k = sample('N_2k', noise_dists['N_2k'])
        N_k = sample('N_k', noise_dists['N_k'])

    map3k = f_map3k(E1, N_3k, params, mode)
    map2k = f_map2k(map3k, N_2k, params, mode)
    mapk = f_mapk(map2k, N_k, params, mode)

    return {
        'map3k': map3k,
        'map2k': map2k,
        'mapk': mapk
    }


def mapk_receiver(noise_dists):
    # Parameters
    mode = 'original'
    al_m = torch.tensor(700.)
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

    with pyro.iarange("model"):
        E1 = Uniform(1.5e-5, 10.).rsample()
        N_3k = sample('N_3k', noise_dists['N_3k'])
        N_2k = sample('N_2k', noise_dists['N_2k'])
        N_k = sample('N_k', noise_dists['N_k'])

    map3k = f_map3k(E1, N_3k, params, mode)
    map2k = f_map2k(map3k, N_2k, params, mode)
    mapk = f_mapk(map2k, N_k, params, mode)

    return {
        'map3k': map3k,
        'map2k': map2k,
        'mapk': mapk
    }


def mapk_do_receiver(do_value,noise_dists):
    mode = 'original'
    # Parameters
    al_m = torch.tensor(700.)
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

    with pyro.iarange("model"):
        E1 = Uniform(1.5e-5, 10.).rsample()
        N_3k = sample('N_3k', noise_dists['N_3k'])
        N_2k = sample('N_2k', noise_dists['N_2k'])
        N_k = sample('N_k', noise_dists['N_k'])

    map3k = do_value
    map2k = f_map2k(map3k, N_2k, params, mode)
    mapk = f_mapk(map2k, N_k, params, mode)

    return map2k
