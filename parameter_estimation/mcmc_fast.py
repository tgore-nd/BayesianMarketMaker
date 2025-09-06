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
    th = theta.copy() # single allocation
    base = bayesian_estimation.U_compiled(const_params, th) # optional baseline (not required for central diff)

    for i in range(D):
        orig = th[i]
        # Relative epsilon avoids huge cancellation for small/large params
        eps = rel_eps * max(1.0, np.abs(orig))
        # Forward step
        th[i] = orig + eps
        up = bayesian_estimation.U_compiled(const_params, th)
        # Backward
        th[i] = orig - eps
        down = bayesian_estimation.U_compiled(const_params, th)
        # Restore
        th[i] = orig
        grad[i] = (up - down) / (2.0 * eps)

    return grad


@njit
def leapfrog_compiled(theta: np.ndarray, const_params: np.ndarray, p: np.ndarray, step_size: float, n_steps: int, inv_mass: float):
    # Get gradient
    grad_U = heston_posterior_grad_fd(theta, const_params)

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
    initial_theta : ndarrayy
        Starting point for parameters `theta`.
    const_params : ndarray
        Parameters that are not estimated but must be passed into U to ensure modularity.
    U : callable
        Potential energy function. U(theta) = -log p(theta) up to const.
    grad_U : callable
        Gradient of U.
    n_samples : int
        Number of MCMC samples to generate.
    step_size : float
        Leapfrog step size.
    n_steps : int
        Number of leapfrog steps per iteration.
    mass : float
        Mass constant.
    
    Returns
    -------
    samples : ndarray
        Collected samples of theta.
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


def run_hmc(initial_theta: np.ndarray, const_params: np.ndarray, num_chains: int, n_samples: int = 10000, step_size: float = 0.0001, n_steps: int = 100, mass: float = 1.0, plot: bool = False) -> tuple[np.ndarray]:
    """
    Generate multiple HMC chains at the same time, powered by multiprocessing and JIT compilation.
    
    Parameters
    ----------
    initial_theta : np.ndarray
        Initial guess for theta.
    const_params : np.ndarray
        An array of parameters that we don't estimate but need for likelihood function evaluation.
    num_chains : int
        The number of chains to sample.
    n_samples : int
        The number of samples in each chain. Default is `10000`.
    step_size : float
        The leapfrog step size. Default is `0.0001`. Larger `step_size` -> smaller acceptance rate. Tune to your data appropriately.
    n_steps : int
        The number of leapfrog steps. Default is `100`. Larger `step_size` -> smaller acceptance rate. Tune to your data appropriately.
    mass : float
        The mass constant. Default is `1.0`.
    plot : bool
        Whether or not to plot a result according to parameter `n` (hardcoded below). Default is `False`.
    
    Returns
    -------
    chains : tuple[np.ndarray]
        A tuple of length `num_chains` giving the full samples for every chain.
    """
    # Increasing n_steps decreases acceptance rate
    # Increasing step_size decreases acceptance rate
    print(f"Beginning HMC with step size = {step_size} and n_steps = {n_steps}")
    jittered_thetas = np.random.normal(scale = 0.01, size = (num_chains, len(initial_theta))) + initial_theta

    # Get several chains at once
    start = time.perf_counter()
    with Pool(num_chains) as pool:
        results = pool.starmap(hmc_sample_fast, [(jittered_theta, const_params, n_samples, step_size, n_steps, mass) for jittered_theta in jittered_thetas])
    chains, accept_rates, = zip(*results)
    print(f"Acceptance rates: {accept_rates}")
    end = time.perf_counter()
    print(f"HMC Runtime: {end - start} seconds")

    if plot:
        n = 2 # change index as necessary to plot different parameter fits
        plt.plot(np.mean(np.vstack([chains[i][:, n] for i in range(num_chains)]).T, axis=1), label=r"Average $\sigma$")
        plt.xlabel("Iteration")
        plt.ylabel("Parameter Value")
        plt.title("Heston Model HMC Fit")
        plt.legend()
        plt.show()

    return chains


if __name__ == "__main__":
    kappa = 2.0
    theta = 0.04
    sigma = 0.3
    rho = -0.7
    v0 = 0.1
    r = 0.01
    S0 = 102.8
    tau = 1 / 525600
    S = 103

    initial_theta = np.array([kappa, theta, sigma, rho, v0], dtype=np.float64)
    const_params = np.array([S, S0, r, tau], dtype=np.float64)
    
    chains = run_hmc(initial_theta, const_params, n_samples=20000, num_chains = 12, plot=True)

    thetas, accept_rates = hmc_sample_fast(initial_theta, const_params, n_samples=20000, step_size=0.0001, n_steps=100)

    plt.plot(thetas[:, 1], label=r"$\theta$")
    plt.plot(thetas[:, 2], label=r"$\sigma$")
    plt.plot(thetas[:, 4], label=r"$v_0$")
    plt.xlabel("Iteration")
    plt.ylabel("Parameter Value")
    plt.title("Heston Model HMC Fit")
    plt.legend()
    plt.show()