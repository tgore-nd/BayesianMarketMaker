import numpy as np
import matplotlib.pyplot as plt
from price_models.heston import heston_likelihood_compiled # the likelihood function
from math import gamma, exp
from numba import njit


@njit
def gamma_pdf_kappa(x, a=3., beta=1):
    if x <= 0:
        return 0.0
    return (1 / (gamma(a) * beta**a)) * x**(a - 1) * exp(-x / beta)


@njit
def half_normal_pdf_kappa(x, sigma=0.5, loc=2.5):
    if x < 0:
        return 0.0
    coef = np.sqrt(2.0) / (sigma * np.sqrt(np.pi))
    exponent = -0.5 * ((x - loc) / sigma) ** 2
    return coef * np.exp(exponent)


@njit
def invgamma_pdf_theta(x, a=3., beta=0.4):
    if x <= 0:
        return 0.0
    return (beta**a / gamma(a)) * x**(-a - 1) * exp(-beta / x)


@njit
def invgamma_pdf_v0(x, a=3., beta=0.6):
    if x <= 0:
        return 0.0
    return (beta**a / gamma(a)) * x**(-a - 1) * exp(-beta / x)


@njit
def uniform_pdf(x, low=-1, high=0):
    if low <= x <= high:
        return 1.0 / (high - low)
    else:
        return 0.0


@njit
def half_normal_pdf_sigma(x, sigma=0.5):
    if x < 0:
        return 0.0
    coef = np.sqrt(2.0) / (sigma * np.sqrt(np.pi))
    exponent = -0.5 * (x / sigma) ** 2
    return coef * np.exp(exponent)


@njit
def half_normal_pdf_else(x, sigma=0.5, loc=0.02):
    if x < 0:
        return 0.0
    coef = np.sqrt(2.0) / (sigma * np.sqrt(np.pi))
    exponent = -0.5 * ((x - loc) / sigma) ** 2
    return coef * np.exp(exponent)


@njit
def prior_compiled(kappa: float, theta: float, sigma: float, rho: float, v0: float) -> float:
    # Remember: we don't see S at this point!
    return half_normal_pdf_kappa(kappa) * invgamma_pdf_theta(theta) * invgamma_pdf_v0(sigma) * uniform_pdf(rho) * invgamma_pdf_theta(v0)


@njit
def U_compiled(const_params: np.ndarray, params: np.ndarray) -> float:
    """Evaluate the potential, the negative log of the posterior. This is used in Hamiltonian Monte Carlo."""
    S, S0, r, tau = const_params
    kappa, theta, sigma, rho, v0 = params
    return -(np.log(heston_likelihood_compiled(S, kappa, theta, sigma, rho, v0, r, S0, tau)) + np.log(prior_compiled(kappa, theta, sigma, rho, v0)))


if __name__ == "__main__":
    S = np.linspace(1e-6, 200, 10000)
    kappa = 2.0
    theta = 0.04
    sigma = 0.3
    rho = -0.7
    v0 = 0.04
    r = 0.03
    S0 = 100
    tau = 1.0
    
    vals = [U_compiled(const_params=np.array([curr_S, S0, r, tau]), params=np.array([kappa, theta, sigma, rho, v0])) for curr_S in S]
    print([(i, val) for i, val in enumerate(vals) if val == np.inf])

    plt.plot(S, vals)
    plt.show()