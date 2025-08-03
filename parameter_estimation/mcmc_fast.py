import numpy as np
import arviz as az
import bayesian_estimation_fast as bayesian_estimation
import time
from typing import Callable
from multiprocessing import Pool
from numba import njit


@njit
def heston_posterior_grad(theta: np.ndarray, const_params: np.ndarray, epsilon=1e-5):
    """Approximate the gradient of the posterior using finite differences."""
    grad = np.zeros_like(theta, dtype=np.float64)
    for i in range(len(theta)):
        dtheta = np.zeros_like(theta, dtype=np.float64)
        dtheta[i] = epsilon
        grad[i] = (bayesian_estimation.U_compiled(const_params, (theta + dtheta)) - bayesian_estimation.U_compiled(const_params, (theta - dtheta))) / (2 * epsilon)

    return grad


@njit
def leapfrog_compiled(theta: np.ndarray, const_params: np.ndarray, p: np.ndarray, step_size: float, n_steps: int, inv_mass: float):
    """
    Perform L steps of the leapfrog integrator.
    
    Parameters
    ----------
    theta : ndarray, shape (D,)
        Current position.
    p : ndarray, shape (D,)
        Current momentum.
    grad_U : callable
        Function ∇U(theta) returning gradient of potential.
    step_size : float
        Integrator step size (ε).
    n_steps : int
        Number of leapfrog steps (L).
    inv_mass : ndarray or float
        Inverse mass matrix (M^-1); can be scalar or diagonal array.
    
    Returns
    -------
    theta_new, p_new : ndarray, ndarray
        The new position and (negated) momentum.
    """
    # Get gradient
    grad_U = heston_posterior_grad(theta, const_params)

    # Half‑step momentum update
    p = p - 0.5 * step_size * grad_U
    for i in range(n_steps):
        # Full‑step position update
        theta = theta + step_size * (inv_mass * p)

        # Full‑step momentum update (except at end)
        if i != n_steps - 1:
            p = p - step_size * grad_U
    
    # Final half‑step momentum update
    p = p - 0.5 * step_size * grad_U

    # Negate momentum to make proposal reversible
    return theta, -p


@njit
def hmc_sample_fast(initial_theta: np.ndarray, const_params: np.ndarray, n_samples: int, step_size: float, n_steps: int, mass: float = 1.0) -> tuple[np.ndarray, float]:
    """
    Run Hamiltonian Monte Carlo.
    
    Parameters
    ----------
    initial_theta : ndarray, shape (D,)
        Starting point for θ.
    const_params : ndarray
        Parameters that are not estimated but must be passed into U to ensure modularity.
    U : callable
        Potential energy function. U(theta) = -log p(theta) up to const.
    grad_U : callable
        Gradient of U.
    n_samples : int
        Number of MCMC samples to generate.
    step_size : float
        Leapfrog step size ε.
    n_steps : int
        Number of leapfrog steps L per iteration.
    mass : float or ndarray
        Mass (diagonal covariance for momentum). May be scalar or vector.
    
    Returns
    -------
    samples : ndarray, shape (n_samples, D)
        Collected samples of θ.
    accept_rate : float
        Proportion of proposals accepted.
    """
    theta = initial_theta
    inv_mass = 1.0 / mass
    D = theta.shape[0]
    
    samples = np.zeros((n_samples, D), dtype=np.float64)
    n_accept = 0
    
    for i in range(n_samples):
        # Sample auxiliary momentum
        p0 = np.random.normal(0, np.sqrt(mass), size=D)
        
        # Simulate Hamiltonian dynamics
        theta_prop, p_prop = leapfrog_compiled(theta, const_params, p0, step_size, n_steps, inv_mass)
        
        # Metropolis acceptance test
        current_H = bayesian_estimation.U_compiled(const_params, theta) + 0.5 * np.sum(p0**2 * inv_mass) # For Heston model, const_params = [S, S0, r, tau], theta = [kappa, theta, sigma, rho, v0]
        prop_H = bayesian_estimation.U_compiled(const_params, theta_prop) + 0.5 * np.sum(p_prop**2 * inv_mass)
        delta_H = prop_H - current_H
        
        if np.random.rand() < np.exp(-delta_H): # to increase acceptance rate, decrease step_size
            theta = theta_prop # if successful, update parameters
            n_accept += 1
        
        samples[i] = theta
    
    accept_rate = n_accept / n_samples
    return samples, accept_rate


def discard_burn_in(chains: list[np.ndarray], max_rhat: float = 1.01, min_retained: int = 100, search_step_size: int = 10) -> tuple[list[np.ndarray], int]:
    n_chains = len(chains)
    n_samples, n_params = chains[0].shape

    # Stack into shape (n_chains, n_samples, n_params)
    stacked = np.stack(chains)

    # Test progressive burn-in cutoffs
    for burnin in range(0, n_samples - min_retained, search_step_size):
        trimmed = stacked[:, burnin:, :]

        # Convert to InferenceData for ArviZ
        idata = az.from_dict(posterior={f"param_{i}": trimmed[..., i] for i in range(n_params)})

        # Compute R-hat for all parameters
        rhats = az.rhat(idata, method="rank").to_dataarray().values # type: ignore

        # If all params are converged, accept this burn-in point
        if np.all(rhats < max_rhat):
            return [chain[burnin:] for chain in chains], burnin

    # If never converged, return full chains with warning
    print("Warning: R-hat threshold not met. Returning full chains.")
    return chains, 0


def run_hmc(initial_theta: np.ndarray, const_params: np.ndarray, num_chains: int, n_samples: int = 3000, step_size: float = 0.001, n_steps: int = 5, mass: float = 1.0):
    jittered_thetas = np.random.normal(scale = 0.01, size = (num_chains, len(initial_theta))) + initial_theta

    # Get several chains at once
    start = time.perf_counter()
    with Pool(num_chains) as pool:
        results = pool.starmap(hmc_sample_fast, [(jittered_theta, const_params, n_samples, step_size, n_steps, 1.0) for jittered_theta in jittered_thetas])
    chains, accept_rates = zip(*results)
    print(accept_rates)
    end = time.perf_counter()
    print(f"Compiled: {end - start} seconds")

    #chains, accept_rates = zip(*[hmc_sample(jittered_theta, const_params, U = U, n_samples=n_samples, step_size=step_size, n_steps=n_steps) for jittered_theta in jittered_thetas])
    
    # chains, burnin = discard_burn_in(list(chains))
    # print(burnin)


if __name__ == "__main__":
    kappa = 2.0
    theta = 0.04
    sigma = 0.3
    rho = -0.7
    v0 = 0.04
    r = 0.01
    S0 = 100
    tau = 1.0
    S = 103

    initial_theta = np.array([kappa, theta, sigma, rho, v0], dtype=np.float64)
    const_params = np.array([S, S0, r, tau], dtype=np.float64)
    
    run_hmc(initial_theta, const_params, num_chains = 5)