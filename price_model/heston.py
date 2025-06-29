import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def heston_cf(phi, tau, kappa, theta, sigma, rho, v0, r, S0):
    """
    Heston characteristic function: returns E[exp(i * phi * ln(S_T))]
    following the risk-neutral characteristic function form.
    
    Parameters
    ----------
    phi   : complex or array_like
        Integration variable (argument of the CF), i.e. φ.
    tau   : float, passed
        Time to maturity T (in years). Amount of time into the future you care about modeling, daily: tau = 1/252, monthly: tau = 1/12, etc
    kappa : float, estimated
        Mean reversion rate of variance (κ).
    theta : float, estimated
        Long-run variance (θ).
    sigma : float, estimated
        Volatility of variance or "vol of vol" (σ).
    rho   : float, estimated
        Correlation between asset and variance Brownian motions (ρ).
    v0    : float, estimated
        Initial variance at t=0 (v₀).
    r     : float, passed
        Risk-free interest rate (r).
    S0    : float, passed
        Initial asset price (S₀), (log-price cf).
    """
    i = 1j
    d = np.sqrt((rho * sigma * i * phi - kappa)**2 + (sigma**2) * (i * phi + phi**2))
    g = (kappa - rho * sigma * i * phi - d) / (kappa - rho * sigma * i * phi + d)
    exp_d_tau = np.exp(-d * tau)
    D = ((kappa - rho * sigma * i * phi - d) / sigma**2) * ((1 - exp_d_tau) / (1 - g * exp_d_tau))
    C = (
        r * i * phi * tau +
        (kappa * theta / sigma**2) *
        ((kappa - rho * sigma * i * phi - d) * tau - 2 * np.log((1 - g * exp_d_tau) / (1 - g)))
    )
    return np.exp(C + D * v0 + i * phi * np.log(S0))


def heston_likelihood(S, kappa, theta, sigma, rho, v0, r, S0, tau):
    """Find the likelihood of S in time tau at current price S0."""
    # Get log prices
    if S == 0:
        # Asymptotically, the likelihood will be zero here (look at the graph to confirm)
        return 0
    
    x = np.log(S)

    # Get integrand
    def integrand(u, x, kappa, theta, sigma, rho, v0, r, S0, tau):
        """Evaluate integrand at some log price x."""
        return np.real(np.exp(-1j*u*x) * heston_cf(u, tau, kappa, theta, sigma, rho, v0, r, S0))
    
    # Perform inverse Fourier transform to get the PDF of x
    val, _ = quad(lambda u: integrand(u, x, kappa, theta, sigma, rho, v0, r, S0, tau),
                  -np.inf, np.inf, # bounds
                  )
    
    val /= (2 * np.pi)

    # Switch to the PDF of S
    return val / S


def likelihood_prob(K, kappa, theta, sigma, rho, v0, r, S0, tau):
    """Find P(S > K). Not used in practice, but confirms that the likelihood function is correct."""
    S = np.arange(0, K)
    f = np.array([heston_likelihood(S_k, kappa, theta, sigma, rho, v0, r, S0, tau) for S_k in S])

    return 1 - np.trapezoid(f, S)


def price_prob(characteristic_func, tau, kappa, theta, sigma, rho, v0, r, S0, K, N = 2**12, B = 200):
    """Return the probability P(S > K). 
    
    This method is based on Gil-Peleaz (1951), particularly a manipulated version of F(x) given by Wendel (1961)."""
    # Generate discrete grid of values (larger B captures tail behavior, higher N resolves oscillations)
    eta = B / N
    u = np.arange(N) * eta
    u[0] = 1e-22 # filter out discontinuity at u = 0
    lnK = np.log(K)

    # Weights - good practice to multiply the first term by 1/2
    weights = np.ones(N)
    weights[0] *= 1/2

    # Evaluate integrand
    integrand = (np.exp(-1j * u * lnK) * characteristic_func(u, tau, kappa, theta, sigma, rho, v0, r, S0) / (1j * u)) * weights

    # Perform FFT
    integral_approx = np.real((np.fft.fft(integrand) * eta)[0])

    return 0.5 + integral_approx / np.pi


if __name__ == "__main__":
    kappa = 2.0
    theta = 0.04
    sigma = 0.3
    rho = -0.7
    v0 = 0.04
    r = 0.01
    S0 = 100
    tau = 1.0
    #K = 125

    K = np.arange(0, 100)

    for k_i in K:
        print(k_i, price_prob(heston_cf, tau, kappa, theta, sigma, rho, v0, r, S0, k_i) - likelihood_prob(k_i, kappa, theta, sigma, rho, v0, r, S0, tau))
        # Overestimate: negative, underestimate: positive; imprecision is probably just due to using multiple integral approximation methods