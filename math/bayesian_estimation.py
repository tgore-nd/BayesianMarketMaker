import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, invgamma, uniform
from price_model.heston import heston_likelihood # the likelihood function


def prior(kappa: float, theta: float, sigma: float, rho: float, v0: float) -> float:
    # Remember: we don't see S at this point!
    def kappa_dist(kappa: float, shape: float = 0.01, scale: float = 0.01):
        """Get the distribution for kappa, the mean reversion rate."""
        return gamma.pdf(kappa, a=shape, scale=scale)
    
    def theta_dist(theta: float, shape: float = 0.01, scale: float = 0.01):
        """Get the distribution for theta, the long-run variance."""
        return invgamma.pdf(theta, a=shape, scale=scale)
    
    def sigma_dist(sigma: float, shape: float = 0.01, scale: float = 0.01):
        """Get the distribution for sigma, the volatility of variance."""
        return invgamma.pdf(sigma, a=shape, scale=scale)
    
    def rho_dist(rho: float, loc: float = -1., scale: float = 2.):
        """Get the distribution for rho, the correlation between price and variance.
        
        Defined on [loc, loc + scale]."""
        return uniform(rho, loc=loc, scale=scale)
    
    def v0_dist(v0: float, shape: float = 0.01, scale: float = 0.01):
        """Get the distribution for v0, the initial variance."""
        return invgamma.pdf(v0, a=shape, scale=scale)
    
    return kappa_dist(kappa) * theta_dist(theta) * sigma_dist(sigma) * rho_dist(rho) * v0_dist(v0)


def posterior(S: float, kappa: float, theta: float, sigma: float, rho: float, v0: float, r: float, S0: float, tau: float):
    return heston_likelihood(S, kappa, theta, sigma, rho, v0, r, S0, tau) * prior(kappa, sigma, theta, rho, v0)


def U(S: float, kappa: float, theta: float, sigma: float, rho: float, v0: float, r: float, S0: float, tau: float) -> float:
    """Evaluate the potential, the negative log of the posterior. This is used in Hamiltonian Monte Carlo."""
    return -np.log(heston_likelihood(S, kappa, theta, sigma, rho, v0, r, S0, tau) * prior(kappa, theta, sigma, rho, v0))
