import numpy as np

def heston_cf(phi, tau, kappa, theta, sigma, rho, v0, r, S0=1.0):
    """
    Heston characteristic function: returns E[exp(i * phi * ln(S_T))]
    following the risk-neutral characteristic function form.
    
    Parameters
    ----------
    phi   : complex or array_like
        Integration variable (argument of the CF), i.e. φ.
    tau   : float
        Time to maturity T (in years). T -> time of maturity, t -> current time. tau is a difference
    kappa : float
        Mean reversion rate of variance (κ).
    theta : float
        Long-run variance (θ).
    sigma : float
        Volatility of variance or "vol of vol" (σ).
    rho   : float
        Correlation between asset and variance Brownian motions (ρ).
    v0    : float
        Initial variance at t=0 (v₀).
    r     : float
        Risk-free interest rate (r).
    S0    : float, optional
        Initial asset price (S₀), default 1.0 (log-price cf).
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

def price_prob(K, characteristic_func):
    pass

if __name__ == "__main__":
    print(heston_cf(1, 1, 1, 1, 1, 1, 1, 1, 1))