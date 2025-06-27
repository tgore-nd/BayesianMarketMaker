import numpy as np


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


def bayad_char_func():
    pass


def get_integration_range(tau, kappa, theta, sigma, rho, v0, r, S0):
    """Get the values a, b for use in FFT"""
    c1 = np.log(S0) + r * tau + (1 - np.exp(-kappa*tau)) * (theta - v0) / (2 * kappa) - 1/2 * theta * tau
    c2 = theta / (8 * kappa**3) * (-sigma**2 * np.exp(-2 * kappa * tau) + 4 * sigma * np.exp(-kappa * tau) * (sigma - 2 * kappa * rho) + 2 * kappa * tau * (4 * kappa**2 + sigma**2 - 4 * kappa * sigma * rho) + sigma * (8 * kappa * rho - 3 * sigma))
    return (c1 - 24*np.sqrt(np.abs(c2)), c1 + 24*np.sqrt(np.abs(c2)))


def heston_price_pdf(tau, kappa, theta, sigma, rho, v0, r, S0, N=2**12):
    a, b = get_integration_range(tau, kappa, theta, sigma, rho, v0, r, S0)
    du = 2 * np.pi / (b - a)
    u = du * np.array([k - N/2 for k in range(N)])
    u[0] = 1e-22

    # Evaluate CF at 
    psi = heston_cf(u, tau, kappa, theta, sigma, rho, v0, r, S0)

    # Trapezoidal weigthts
    weights = np.ones(N)
    weights[0], weights[-1] = 0.5, 0.5
    
    # Build integrand
    integrand = np.exp(-1j * u * a) * psi * weights

    # Run inverse FFT
    fft_vals = np.fft.ifft(integrand) * N * du / np.pi
    return fft_vals.real


def price_prob(characteristic_func, tau, kappa, theta, sigma, rho, v0, r, S0, K, N = 2**12, B = 200):
    # Generate discrete grid of values (larger B captures tail behavior, higher N resolves oscillations)
    eta = B / N
    u = (np.arange(N) * eta)
    u[0] = 1e-22 # filter out discontinuity at u = 0
    lnK = np.log(K)

    # Weights - good practice to multiply the first term by 1/2
    weights = np.ones(N)
    weights[0] *= 1/2

    # Evaluate integrand
    integrand = (np.exp(-1j * u * lnK) * characteristic_func(u, tau, kappa, theta, sigma, rho, v0, r, S0) / (1j * u)) * weights

    # Perform FFT
    integral_approx = np.real((np.fft.fft(integrand) * eta)[0])

    return np.clip(0.5 + integral_approx / np.pi, 0, 1)

if __name__ == "__main__":
    print(price_prob(heston_cf, 1, 1, 1, 1, 1, 1, 1, 1, 1))