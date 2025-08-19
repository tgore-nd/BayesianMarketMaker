import numpy as np
from scipy.integrate import quad
from typing import Callable
from numba import njit


@njit
def heston_cf(phi: np.ndarray, tau: float, kappa: float, theta: float, sigma: float, rho: float, v0: float, r: float, S0: float) -> np.ndarray:
    """
    Heston characteristic function: returns E[exp(i * phi * ln(S_T))]
    following the risk-neutral characteristic function form.
    
    Parameters
    ----------
    phi   : complex or array_like
        Integration variable (argument of the CF), i.e. φ.
    tau   : float, passed
        Time to maturity T (in YEARS). Amount of time into the future you care about modeling, daily: tau = 1/252, monthly: tau = 1/12, etc
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


@njit
def heston_integrand(u, x, kappa, theta, sigma, rho, v0, r, S0, tau):
    """Evaluate integrand at some log price x."""
    return np.real(np.exp(-1j*u*x) * heston_cf(u, tau, kappa, theta, sigma, rho, v0, r, S0))


def heston_likelihood(S: float, kappa: float, theta: float, sigma: float, rho: float, v0: float, r: float, S0: float, tau: float) -> float:
    """Find the likelihood of S in time tau at current price S0."""
    # Get log prices
    if S == 0:
        # Asymptotically, the likelihood will be zero here (look at the graph to confirm)
        return 0
    
    x = np.log(S)
    
    # Perform inverse Fourier transform to get the PDF of x
    val, _ = quad(lambda u: heston_integrand(u, x, kappa, theta, sigma, rho, v0, r, S0, tau),
                  -np.inf, np.inf, # bounds
                  )
    
    val /= (2 * np.pi)
    if abs(val) == np.inf: print("Error!")
    # Switch to the PDF of S
    return val / S


@njit
def heston_likelihood_compiled(S: float, kappa: float, theta: float, sigma: float, rho: float, v0: float, r: float, S0: float, tau: float) -> float:
    """Find the likelihood of S in time tau at current price S0."""
    # Get log prices
    if S == 0:
        # Asymptotically, the likelihood will be zero here (look at the graph to confirm)
        return 0
    
    x = np.log(S)

    # Numerical integration via trapezoidal rule
    N = 2000
    L = 100  # limit of integration
    u_vals = np.linspace(-L, L, N)
    du = u_vals[1] - u_vals[0]

    total = 0.0
    for i in range(N):
        weight = 0.5 if i == 0 or i == N - 1 else 1.0
        total += weight * heston_integrand(u_vals[i], x, kappa, theta, sigma, rho, v0, r, S0, tau)
    
    val = du * total / (2 * np.pi)
    return np.maximum(val / S, 1e-12)


def likelihood_prob(K, kappa: float, theta: float, sigma: float, rho: float, v0: float, r: float, S0: float, tau: float) -> float:
    """Find P(S > K). Not used in practice, but confirms that the likelihood function is correct."""
    S = np.arange(0, K)
    f = np.array([heston_likelihood(S_k, kappa, theta, sigma, rho, v0, r, S0, tau) for S_k in S])

    return 1 - np.trapezoid(f, S)


def price_prob(characteristic_func: Callable, tau: float, kappa: float, theta: float, sigma: float, rho: float, v0: float, r: float, S0: float, K: float, N: int = 2**12, B: int = 200) -> float:
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


def generate_sample_paths(tau: float, kappa: float, theta: float, sigma: float, rho: float, v0: float, S0: float, r: float, N: int, M: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate sample paths using Bayesian-estimated parameters.
    Inputs:
     - tau   : time of simulation
     - kappa : rate of mean reversion in variance process
     - theta : long-term mean of variance process
     - sigma : vol of vol / volatility of variance process
     - rho   : correlation between asset returns and variance
     - S0, v0: initial parameters for asset and variance
     - r     : interest rate
     - N     : number of time steps
     - M     : number of asset paths
    
    Outputs:
    - asset prices over time (numpy array)
    - variance over time (numpy array)
    """
    # initialise other parameters
    dt = tau/N
    mu = np.array([0, 0])
    cov = np.array([[1, rho],
                    [rho, 1]])
    
    # Instantiate arrays
    S = np.full(shape=(N + 1, M), fill_value=S0)
    v = np.full(shape=(N + 1, M), fill_value=v0)

    # Sample correlated Brownian motions under risk-neutral measure
    Z = np.random.multivariate_normal(mu, cov, (N, M))
    for i in range(1, N + 1):
        S[i] = S[i - 1] * np.exp((r - 0.5 * v[i - 1])*dt + np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 0])
        v[i] = np.maximum(v[i - 1] + kappa*(theta - v[i - 1]) * dt + sigma * np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 1], 0)
    
    return S.T, v.T


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

    print(heston_likelihood(102, kappa, theta, sigma, rho, v0, r, S0, tau))