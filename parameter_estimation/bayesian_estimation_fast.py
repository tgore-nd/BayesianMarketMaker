import numpy as np
from price_models.heston import heston_likelihood_compiled # the likelihood function
from math import gamma, exp
from numba import njit


@njit
def gamma_pdf(x, a=0.01, beta=0.01):
    if x <= 0:
        return 0.0
    return (1 / (gamma(a) * beta**a)) * x**(a - 1) * exp(-x / beta)


@njit
def invgamma_pdf(x, a=0.01, beta=0.01):
    if x <= 0:
        return 0.0
    return (beta**a / gamma(a)) * x**(-a - 1) * exp(-beta / x)


@njit
def uniform_pdf(x, low=-1, high=2):
    if low <= x <= high:
        return 1.0 / (high - low)
    else:
        return 0.0


@njit
def prior_compiled(kappa: float, theta: float, sigma: float, rho: float, v0: float) -> float:
    # Remember: we don't see S at this point!
    return float(gamma(kappa) * invgamma_pdf(theta) * invgamma_pdf(sigma) * uniform_pdf(rho) * invgamma_pdf(v0))


@njit
def U_compiled(const_params: np.ndarray, params: np.ndarray) -> float:
    """Evaluate the potential, the negative log of the posterior. This is used in Hamiltonian Monte Carlo."""
    #S: float, S0: float, r: float, tau: float, kappa: float, theta: float, sigma: float, rho: float, v0: float
    S, S0, r, tau = const_params
    kappa, theta, sigma, rho, v0 = params
    return -np.log(heston_likelihood_compiled(S, kappa, theta, sigma, rho, v0, r, S0, tau) * prior_compiled(kappa, theta, sigma, rho, v0))
