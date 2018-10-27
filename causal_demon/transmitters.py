from __future__ import division

from functools import partial

import pyro
from pyro import sample
import torch
from torch import tensor

"""
Pseudo Dirac Delta Distribution

Creates a pseudo Dirac delta distribution as normal distribution with very low
variance. In Bayesian parlance this is a "spike" (as in "spike and slab
prior"), meaning a normal distribution with very small variance, such that a
spike with 0 variance is a Dirac delta distribution. Pyro has a
`pyro.distributions.delta` distribution, but attempting to obtain empirical
marginals from conditioned programs will sometimes fail when this distribution
is used.
"""
Delta = partial(pyro.distributions.Normal, scale=1e-6)

constants = {
    "sos_tot": 120000.,
    "ras_tot": 120000.,
    "pi3k_tot": 120000.,
    "akt_tot": 600000.,
    "raf_tot": 120000.,
    "mek_tot": 600000.,
    "erk_tot": 600000.,
    "p90": 120000.,
    "rasgap": 120000.,
    "raf_pp": 120000.,
    "pp2a": 120000.,
}


params = {
    "egf_sos": 6.351334e-04,
    "igf_sos": 2.781988e-02,
    "sos_ras": 0.8537238,
    "egf_pi3k": 10.6737/184912/.005,
    "igf_pi3k": 10.6737/184912/.005,
    "ras_pi3k": .0771067/272056/.005,
    "pi3k_akt": 1.73187e-05,
    "ras_raf": 1.18951e-01,
    "akt_raf": 1.064752,
    "raf_mek": 7.134808,
    "mek_erk": 3.846749,
    "erk_p90": 5.597658e-06,
}


def g(a, b):
    return a / (a + b)


def f_egf(N_egf):
    """Input signal EGF
    Input signals EGF and IGF are identity functions of their respective noise
    terms.
    """
    return sample("egf", Delta(N_egf))


def f_igf(N_igf):
    """Input signal EGF
    Input signals EGF and IGF are identity functions of their respective noise
    terms.
    """
    return sample("igf", Delta(N_igf))


def f_sos(egf, igf, N_sos):
    beta_egf_sos = params['egf_sos']
    beta_igf_sos = params['igf_sos']
    sos_tot = constants['sos_tot']
    p90 = constants['p90']

    sos_mu = sos_tot * g(beta_egf_sos * egf + beta_igf_sos * igf, p90)
    return sample("sos", Delta(sos_mu + N_sos))


def f_ras(sos, N_ras):
    beta_sos_ras = params['sos_ras']
    total_ras = constants['ras_tot']
    rasgap = constants['rasgap']

    ras_mu = total_ras * g(beta_sos_ras * sos, rasgap)
    return sample("ras", Delta(ras_mu + N_ras))


def f_pi3k(ras, egf, igf, N_pi3k):
    beta_ras_pi3k = params['ras_pi3k']
    beta_egf_pi3k = params['egf_pi3k']
    beta_igf_pi3k = params['igf_pi3k']
    total_pi3k = constants['pi3k_tot']

    pi3k_mu = total_pi3k * g(
        beta_ras_pi3k * ras +
        beta_egf_pi3k * egf +
        beta_egf_pi3k * igf,
        1.0
    )
    return sample('pi3k', Delta(pi3k_mu + N_pi3k))


def f_akt(pi3k, N_akt):
    beta_pi3k_akt = params['pi3k_akt']
    total_akt = constants['akt_tot']

    akt_mu = total_akt * g(beta_pi3k_akt * pi3k, 1.0)
    return sample('akt', Delta(akt_mu + N_akt))


def f_raf(ras, akt, N_raf):
    beta_ras_raf = params['ras_raf']
    beta_akt_raf = params['akt_raf']
    raf_pp = constants['raf_pp']
    total_raf = constants['raf_tot']

    raf_mu = total_raf * g(beta_ras_raf * ras, beta_akt_raf * akt + raf_pp)
    return sample('raf', Delta(raf_mu + N_raf))


def f_mek(raf, N_mek):
    beta_raf_mek = params['raf_mek']
    total_mek = constants['mek_tot']
    pp2a = constants['pp2a']

    mek_mu = total_mek * g(beta_raf_mek * raf, pp2a)
    return sample('mek', Delta(mek_mu + N_mek))


def f_erk(mek, N_erk):
    beta_mek_erk = params['mek_erk']
    total_erk = constants['erk_tot']
    pp2a = constants['pp2a']

    erk_mu = total_erk * g(beta_mek_erk * mek, pp2a)
    return sample('erk', Delta(erk_mu + N_erk))


def cancer_signaling(noise_dists, sample_shape=torch.Size([1])):
    """Model of Biochemical Signal Transduction in Lung Cancer

    This is the model of the EGFR and IGF1R pathway in lung cancer by Bianconi
    et al 2012. A description of the model, the reference, and a SBML file
    containing the model is available at
    https://www.ebi.ac.uk/biomodels/BIOMD0000000427.

    The modeling approach in the reference uses an ordinary differential
    equation modeling approach. This model assumes the ODEs have been solved
    for steady state, and models the steady state with structural causal
    models.  For more details on this math, see the bianconi_math document.

    The following illustrates how you would simulate an experiment with this
    model.

    Suppose there were 3 conditions: EGF is varied from low (800),
    medium (2000), to high (8000) concentrations.  Each condition had 4
    replicates.

        from pyro.distributions import LogNormal

        noise_vars = [
            'N_egf', 'N_igf', 'N_sos',
            'N_ras', 'N_pi3k', 'N_akt',
            'N_raf', 'N_mek', 'N_erk'
        ]

        noise_priors = {N: LogNormal(0, .001) for N in noise_vars}

        # replicates on the rows, conditions on the columns
        #
        conditions = {
            'egf': torch.tensor(
                [[800., 8000., 80000.],
                 [800., 8000., 80000.],
                 [800., 8000., 80000.],
                 [800., 8000., 80000.]]
            ),
            'igf': torch.zeros([4, 3])
        }
        experiment = pyro.do(cancer_signaling, data=conditions)
        # Get some Erk values
        experiment(noise_priors, [4, 3])['erk']
    """

    N_egf = sample('N_egf', noise_dists['N_egf'], sample_shape=sample_shape)
    N_igf = sample('N_igf', noise_dists['N_igf'], sample_shape=sample_shape)
    N_sos = sample('N_sos', noise_dists['N_sos'], sample_shape=sample_shape)
    N_ras = sample('N_ras', noise_dists['N_ras'], sample_shape=sample_shape)
    N_pi3k = sample('N_pi3k', noise_dists['N_pi3k'], sample_shape=sample_shape)
    N_akt = sample('N_akt', noise_dists['N_akt'], sample_shape=sample_shape)
    N_raf = sample('N_raf', noise_dists['N_raf'], sample_shape=sample_shape)
    N_mek = sample('N_mek', noise_dists['N_mek'], sample_shape=sample_shape)
    N_erk = sample('N_erk', noise_dists['N_erk'], sample_shape=sample_shape)

    egf = f_egf(N_egf)
    igf = f_igf(N_igf)
    sos = f_sos(egf, igf, N_sos)
    ras = f_ras(sos, N_ras)
    pi3k = f_pi3k(ras, egf, igf, N_pi3k)
    akt = f_akt(pi3k, N_akt)
    raf = f_raf(ras, akt, N_raf)
    mek = f_mek(raf, N_mek)
    erk = f_erk(mek, N_erk)

    return {
        'egf': egf,
        'igf': igf,
        'sos': sos,
        'ras': ras,
        'pi3k': pi3k,
        'akt': akt,
        'raf': raf,
        'mek': mek,
        'erk': erk
    }
