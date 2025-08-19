import numpy as np
from . import bayesian_estimation_fast as bayesian_estimation
import matplotlib.pyplot as plt
import time
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
def heston_posterior_grad_fd(theta: np.ndarray, const_params: np.ndarray, rel_eps: float = 1e-6):
    """
    Central finite-difference gradient with relative step size.
    - theta: 1D array of parameters
    - const_params: array of fixed params passed to U_compiled
    - rel_eps: relative step size (recommended 1e-6 .. 1e-8)
    """
    D = theta.shape[0]
    grad = np.empty(D, dtype=np.float64)
    th = theta.copy()           # single allocation
    base = bayesian_estimation.U_compiled(const_params, th)  # optional baseline (not required for central diff)

    for i in range(D):
        orig = th[i]
        # relative epsilon avoids huge cancellation for small/large params
        eps = rel_eps * max(1.0, np.abs(orig))
        # forward
        th[i] = orig + eps
        up = bayesian_estimation.U_compiled(const_params, th)
        # backward
        th[i] = orig - eps
        down = bayesian_estimation.U_compiled(const_params, th)
        # restore
        th[i] = orig
        grad[i] = (up - down) / (2.0 * eps)

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
    grad_U = heston_posterior_grad_fd(theta, const_params)
    # norm = np.linalg.norm(grad_U)
    # print(theta, norm)

    # if norm > 50:
    #     print(f"Norm: {norm}")
    #     print(f"Theta: {theta}")
    #     print(f"U: {bayesian_estimation.U_compiled(const_params, theta)}\n")

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
    # norms = []
    
    samples = np.zeros((n_samples, D), dtype=np.float64)
    n_accept = 0
    
    for i in range(n_samples):
        # Sample auxiliary momentum
        p0 = np.random.normal(0, np.sqrt(mass), size=D)
        
        # Simulate Hamiltonian dynamics
        theta_prop, p_prop = leapfrog_compiled(theta, const_params, p0, step_size, n_steps, inv_mass)
        # norms.append(norm)
        
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


def run_hmc(initial_theta: np.ndarray, const_params: np.ndarray, num_chains: int, n_samples: int = 10000, step_size: float = 0.0001, n_steps: int = 100, mass: float = 1.0):
    # 1.25e-05
    # 2.5e-05, 60
    # 1.5e-05, 100
    # 2.6e-05, 100
    # Increasing n_steps decreases acceptance rate
    # Increasing step_size decreases acceptance rate
    print(step_size, n_steps)
    jittered_thetas = np.random.normal(scale = 0.01, size = (num_chains, len(initial_theta))) + initial_theta

    # Get several chains at once
    start = time.perf_counter()
    with Pool(num_chains) as pool:
        results = pool.starmap(hmc_sample_fast, [(jittered_theta, const_params, n_samples, step_size, n_steps, mass) for jittered_theta in jittered_thetas])
    chains, accept_rates, = zip(*results)
    print(accept_rates)
    end = time.perf_counter()
    print(f"Compiled: {end - start} seconds")

    # Plot
    # fig, axs = plt.subplots(5, num_chains)
    # for chain_i in range(num_chains):
    #     for i in range(5):
    #         axs[i, chain_i].plot(chains[chain_i][:, i])

    # plt.tight_layout()
    # plt.show()


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

    # thetas, rates, norms = hmc_sample_fast(initial_theta, const_params, n_samples=10000, step_size=2.6e-5, n_steps=100)

    # # Plot
    # fig, axs = plt.subplots(5)
    # for i in range(5):
    #     axs[i].scatter(thetas[:, i], norms, s=0.3)
    
    # plt.xlim(0, 0.05)
    # plt.tight_layout()
    # plt.show()