# NOTE: Only the compiled versions are used in production. This version is old, but it is clear and modular and more nicely readable since it doesn't need to adhere to Numba constraints.

import numpy as np
import bayesian_estimation
import time
from typing import Callable
from multiprocessing import Pool


def posterior_grad(theta: np.ndarray, const_params: np.ndarray, posterior: Callable, epsilon=1e-5):
    """Approximate the gradient of the posterior using finite differences."""
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        dtheta = np.zeros_like(theta)
        dtheta[i] = epsilon
        grad[i] = (posterior(*const_params, *(theta + dtheta)) - posterior(*const_params, *(theta - dtheta))) / (2 * epsilon)
        if np.any(np.isnan(grad[i])): print(f"Grad: {grad[i]}, params: {theta + dtheta}")

    return grad


def leapfrog(U: Callable, theta: np.ndarray, const_params: np.ndarray, p: np.ndarray, step_size: float, n_steps: int, inv_mass: float):
    # Get gradient
    grad_U = posterior_grad(theta, const_params, U)

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


def hmc_sample(initial_theta: np.ndarray, const_params: np.ndarray, U: Callable, n_samples: int, step_size: float, n_steps: int, mass: float = 1.0) -> tuple[np.ndarray, float]:
    theta = np.array(initial_theta, dtype=np.float64)
    inv_mass = 1.0 / mass
    D = theta.shape[0]
    
    samples = np.zeros((n_samples, D), np.float64)
    n_accept = 0
    
    for i in range(n_samples):
        # Sample auxiliary momentum
        p0 = np.random.normal(0, np.sqrt(mass), size=D)
        
        # Simulate Hamiltonian dynamics
        theta_prop, p_prop = leapfrog(U, theta, const_params, p0, step_size, n_steps, inv_mass)
        
        # Metropolis acceptance test
        current_H = U(*const_params, *theta) + 0.5 * np.sum(p0**2 * inv_mass) # For Heston model, const_params = [S, S0, r, tau], theta = [kappa, theta, sigma, rho, v0]
        prop_H = U(*const_params, *theta_prop) + 0.5 * np.sum(p_prop**2 * inv_mass)
        delta_H = prop_H - current_H
        
        if np.random.rand() < np.exp(-delta_H): # to increase acceptance rate, decrease step_size
            theta = theta_prop # if successful, update parameters
            n_accept += 1
        
        samples[i] = theta
    
    accept_rate = n_accept / n_samples
    return samples, accept_rate


def run_hmc(initial_theta: np.ndarray, const_params: np.ndarray, U: Callable, num_chains: int, n_samples: int = 3000, step_size: float = 0.001, n_steps: int = 5, mass: float = 1.0):
    jittered_thetas = np.random.normal(scale = 0.01, size = (num_chains, len(initial_theta))) + initial_theta

    # Get several chains at once
    start = time.perf_counter()
    with Pool(num_chains) as pool:
        results = pool.starmap(hmc_sample, [(jittered_theta, const_params, U, n_samples, step_size, n_steps, 1.0) for jittered_theta in jittered_thetas])
    chains, accept_rates = zip(*results)
    print(accept_rates)
    end = time.perf_counter()
    print(f"Normal: {end - start} seconds")


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
    
    run_hmc(initial_theta, const_params, bayesian_estimation.U, num_chains = 5)