import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from environment import MarketEnvironment
from typing import Literal

# Regularization parameters
AIIF = 0.5
INV_LIMIT = 10
PRICE_LIMIT = 0.005

class ReplayBuffer:
    """The `ReplayBuffer` is used to store actual and simulated transitions."""
    def __init__(self, max_size: int, state_dim: int, action_dim: int):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, current_state: torch.Tensor, action: torch.Tensor, reward: float, resulting_state: torch.Tensor) -> None:
        i = self.ptr
        self.state[i] = current_state
        self.action[i] = action
        self.reward[i] = reward
        self.next_state[i] = resulting_state
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return dict(
            current_state = torch.tensor(self.state[idx]),
            action = torch.tensor(self.action[idx]),
            next_state = torch.tensor(self.next_state[idx]),
            reward = torch.tensor(self.reward[idx]),
        )


def mlp(sizes: list[int], activation, output_activation) -> nn.Sequential:
    """Return a multi-layer perceptron (MLP) neural network model."""
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes)-2 else output_activation # check if the network has ended and if so, use output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """Gaussian Policy network for action selection."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, int] = (256, 256), log_std_min: float = -20, log_std_max: float = 2):
        super().__init__()
        self.net = mlp([obs_dim]+list(hidden), activation=nn.ReLU, output_activation=nn.ReLU)
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Linear(hidden[-1], act_dim)
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.lower_bounds = torch.tensor([-PRICE_LIMIT, 0, 0, 0, -INV_LIMIT])
        self.upper_bounds = torch.tensor([0, INV_LIMIT, PRICE_LIMIT, INV_LIMIT, INV_LIMIT])

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        x = self.net(obs)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        # Reparameterize
        eps = torch.randn_like(mu)
        pre_tanh = mu + eps * std
        action = torch.tanh(pre_tanh)  # squashed to (-1, 1)

        # Log_prob with tanh correction:
        log_prob = -0.5 * ((eps)**2 + 2*log_std + np.log(2*np.pi))
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Correction for tanh
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        # Rescale action
        action_rescaled = self.lower_bounds + (0.5 * (action + 1.0)) * (self.upper_bounds - self.lower_bounds)
        close_price_modification = torch.tensor([obs[0][3], 0, obs[0][3], 0, 0])

        return action_rescaled + close_price_modification, log_prob, mu

# Q Networks for the actors and critics
class QNetwork(nn.Module):
    """Actor takes in current environment and determines best action.
    
    Critic estimates the value function, V(s), which represents the expected cumulative reward starting from state s."""
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim + act_dim] + list(hidden) + [1], activation=nn.ReLU, output_activation=nn.Identity)
    def forward(self, s, a) -> nn.Sequential:
        x = torch.cat([s, a], dim=-1)
        return self.net(x)

# SAC Agent
class SACAgent:
    """A soft actor-critic agent that interacts with the market environment via a `GaussianPolicy` and `QNetwork` actors and critics."""
    def __init__(self, state_dim: int = 11, action_dim: int = 5, device: Literal["cpu", "cuda"] = "cpu"):
        self.device = device
        self.actor = GaussianPolicy(state_dim, action_dim).to(device)
        self.q1 = QNetwork(state_dim, action_dim).to(device) # min of these two networks is used to stabilize learning
        self.q2 = QNetwork(state_dim, action_dim).to(device)
        self.q1_target = QNetwork(state_dim, action_dim).to(device)
        self.q2_target = QNetwork(state_dim, action_dim).to(device)

        # Copy weights to targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=3e-4)

        self.alpha = 0.2  # entropy regularization parameter
        self.gamma = 0.99 # discount factor
        self.rho = 0.995  # polyak; helps smooth target updates to stabilize learning; -> 1 means slower updates

    def select_action(self, state: torch.Tensor, deterministic: bool = False): # deterministic at testing time; uses mean
        s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            a, _, mu = self.actor(s_t)
            if deterministic:
                return mu.cpu().numpy()[0]
            return a.cpu().numpy()[0]

    def update(self, batch: dict[str, torch.Tensor]):
        # Extract tensors
        current_state = batch['current_state'].float().to(self.device)
        action = batch['action'].float().to(self.device)
        reward = batch['reward'].float().to(self.device)
        next_state = batch['next_state'].float().to(self.device)

        # Critic Q Function
        with torch.no_grad():
            # Generate an action for the next state
            next_action, logp_next_action, _ = self.actor(next_state)
            next_action = next_action.detach()
            logp_next_action = logp_next_action.detach()

            # Evaluate Q function of next_action in next_state
            q1_t = self.q1_target(next_state, next_action)
            q2_t = self.q2_target(next_state, next_action)
            q_t = torch.min(q1_t, q2_t) - self.alpha * logp_next_action

            # Reward: immediate reward plus discounted soft value of the next state (what critics approximate)
            y = reward + self.gamma * q_t

        # Critic Update
        q1_pred = self.q1(current_state, action)
        q2_pred = self.q2(current_state, action)
        q1_loss = F.mse_loss(q1_pred, y)
        q2_loss = F.mse_loss(q2_pred, y)

        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()

        # Actor Q Function
        a_pi, logp_pi, _ = self.actor(current_state) # pick action
        q1_pi = self.q1(current_state, a_pi) # grade action
        q2_pi = self.q2(current_state, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Actor Update
        actor_loss = (self.alpha * logp_pi - q_pi).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        # Soft updates: theta_targ * rho + (1 - rho) * theta
        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
                tp.data.mul_(self.rho)
                tp.data.add_((1 - self.rho) * p.data)
            for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
                tp.data.mul_(self.rho)
                tp.data.add_((1 - self.rho) * p.data)

# Transition simulation
def generate_simulated_transitions(env: MarketEnvironment, initial_state: torch.Tensor, initial_action: torch.Tensor, n_samples: int = 1):
    """
    Use the HMC-fitted Heston model to allow the model to learn from acting in the simulated space.

    Parameters
    ----------
    env : MarketEnvironment
        environment in which to simulate
    initial_state : torch.Tensor
        initial state tensor of environment
    initial_action : torch.Tensor
        initial action of model in environment that will be tested in simulation
    n_samples : int
        Number of transitions to simulate. Default is `1`, but I suggest making this larger to offset the cost of HMC.
    """
    current_state = initial_state
    current_action = initial_action
    params = None
    transitions = []
    for i in range(n_samples):
        print(f"Simulation #{i}...")
        simulated_next_state, reward, params = env.projected_step(current_action, current_state, AIIF, horizon_timestep=0, step_forward=False, params=params)
        transitions.append((current_state, current_action, reward, simulated_next_state))
    
    return transitions

# Training loop
def train_loop(env: MarketEnvironment, initial_theta: np.ndarray, num_steps: int = 100000, plan: bool = True) -> SACAgent:
    """
    Run the agent.
    
    Parameters
    ----------
    env : MarketEnvironment
        The environment that the agent will explore and exploit.
    initial_theta : np.ndarray
        A guess for the initial Heston parameters (kappa, theta, sigma, rho, v0, r).
    num_steps : int
        The total number of steps to explore within the environment.
    plan : bool
        Whether or not to train on simulated transitions. Default is `True`.
    
    Returns
    -------
    agent : SACAgent
        The `SACAgent` that has explored the environment.
    """
    state_dim = 11
    action_dim = 5
    buffer = ReplayBuffer(200000, state_dim, action_dim)
    agent = SACAgent(state_dim, action_dim)
    initial_state = env.reset(initial_theta)
    for t in range(num_steps):
        # Choose action (use actor for exploration)
        action = agent.select_action(initial_state)
        next_state, reward = env.step(action, AIIF)
        buffer.add(initial_state, action, reward, next_state) # add transition to replay buffer
        print(f"Initial state: {initial_state}")
        print(f"Action: {action}")
        print(f"Final state: {next_state}")
        print(f"Reward: {reward}")

        # Generate model-based transitions
        if plan:
            imagined = generate_simulated_transitions(env, initial_state, action, n_samples=10)
            for (s_i, a, r, s_f) in imagined:
                buffer.add(s_i, a, r, s_f)

        # Train
        if buffer.size > 1024:
            print("Training now...")
            for _ in range(1):  # gradient steps per env step
                batch = buffer.sample(256)
                agent.update(batch)
        
        initial_state = next_state

    return agent

if __name__ == "__main__":
    kappa = 2.0
    theta = 0.04
    sigma = 0.3
    rho = -0.7
    v0 = 0.04
    initial_theta = np.array([kappa, theta, sigma, rho, v0])
    path_to_delta_lake = "" # fill this in with your path
    env = MarketEnvironment("AAPL", path_to_delta_lake, "2010-01-01", initial_theta)
    print("Done")

    train_loop(env, initial_theta, plan=False)
