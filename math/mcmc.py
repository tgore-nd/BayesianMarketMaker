import numpy as np
from typing import Callable

def posterior_grad(theta: np.ndarray, posterior: Callable, epsilon=1e-5):
    """Approximate the gradient of the posterior using finite differences."""
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        dtheta = np.zeros_like(theta)
        dtheta[i] = epsilon
        grad[i] = (posterior(theta + dtheta) - posterior(theta - dtheta)) / (2 * epsilon)
    return grad


def leapfrog(U: Callable, theta: np.ndarray, p: np.ndarray, step_size: float, n_steps: int, inv_mass: float):
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
    grad_U = posterior_grad(theta, U)

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

def hmc_sample(initial_theta: np.ndarray, U: Callable, n_samples: int, step_size: float, n_steps: int, mass: float = 1.0) -> tuple[np.ndarray, float]:
    """
    Run Hamiltonian Monte Carlo.
    
    Parameters
    ----------
    initial_theta : ndarray, shape (D,)
        Starting point for θ.
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
    theta = np.array(initial_theta, dtype=float)
    inv_mass = 1.0 / mass
    D = theta.shape[0]
    
    samples = np.zeros((n_samples, D))
    n_accept = 0
    
    for i in range(n_samples):
        # Sample auxiliary momentum
        p0 = np.random.normal(0, np.sqrt(mass), size=D)
        
        # Simulate Hamiltonian dynamics
        theta_prop, p_prop = leapfrog(U, theta, p0, step_size, n_steps, inv_mass)
        
        # Metropolis acceptance test
        current_H = U(theta) + 0.5 * np.sum(p0**2 * inv_mass)
        prop_H = U(theta_prop) + 0.5 * np.sum(p_prop**2 * inv_mass)
        delta_H = prop_H - current_H
        
        if np.random.rand() < np.exp(-delta_H):
            theta = theta_prop # if successful, update parameters
            n_accept += 1
        
        samples[i] = theta
    
    accept_rate = n_accept / n_samples
    return samples, accept_rate


if __name__ == "__main__":
    # Example: Sampling from a 1D standard normal N(0, 1)

    # Potential U(θ) = θ^2/2, so grad_U = θ
    def U_gauss(theta):
        return 0.5 * np.dot(theta, theta)

    # Run HMC
    np.random.seed(42)
    samples, rate = hmc_sample(
        initial_theta=np.array([2.0]),
        U=U_gauss,
        n_samples=5000,
        step_size=0.1,
        n_steps=10,
        mass=1.0
    )

    print(f"Acceptance rate: {rate:.3f}")
    print("Sample mean:", samples.mean())
    print("Sample variance:", samples.var(ddof=1))