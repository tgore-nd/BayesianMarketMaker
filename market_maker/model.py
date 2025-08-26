import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from environment import MarketEnvironment
from typing import Literal


AIIF = 0.5


class ReplayBuffer:
    def __init__(self, max_size: int, state_dim: int, action_dim: int):
        """The `ReplayBuffer` is used to store actual and simulated transitions."""
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


# Gaussian Policy network for action selection
class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, int] = (256,256), log_std_min: float = -20, log_std_max: float = 2):
        super().__init__()
        self.net = mlp([obs_dim]+list(hidden), activation=nn.ReLU, output_activation=nn.ReLU)
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Linear(hidden[-1], act_dim)
        self.log_std_min, self.log_std_max = log_std_min, log_std_max

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        x = self.net(obs)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        # reparameterize
        eps = torch.randn_like(mu)
        pre_tanh = mu + eps * std
        action = torch.tanh(pre_tanh)  # squashed to (-1, 1)
        # log_prob with tanh correction:
        log_prob = -0.5 * ((eps)**2 + 2*log_std + np.log(2*np.pi))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        # correction for tanh
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return action, log_prob, mu

# Q Networks for the actors and critics
class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim + act_dim] + list(hidden) + [1], activation=nn.ReLU, output_activation=nn.Identity)
    def forward(self, s, a) -> nn.Sequential:
        x = torch.cat([s, a], dim=-1)
        return self.net(x)

# SAC Agent
class SACAgent:
    def __init__(self, state_dim: int = 12, action_dim: int = 5, device: Literal["cpu", "cuda"] = "cpu"):
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
        self.rho = 0.995 # polyak; helps smooth target updates to stabilize learning; -> 1 means slower updates

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

        # Actor Q Function
        a_pi, logp_pi, _ = self.actor(current_state) # pick action
        q1_pi = self.q1(current_state, a_pi) # grade action
        q2_pi = self.q2(current_state, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Critic Q Function
        with torch.no_grad():
            # Generate an action for the next state
            next_action, logp_next_action, _ = self.actor(next_state)

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

        # Actor Update
        actor_loss: torch.Tensor = (self.alpha * logp_pi - q_pi).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        # Soft updates: theta_targ * rho + (1 - rho) * theta
        for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
            tp.data.mul_(self.rho)
            tp.data.add_((1 - self.rho) * p.data)
        for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
            tp.data.mul_(self.rho)
            tp.data.add_((1 - self.rho) * p.data)

# Transition simulation
def generate_simulated_transitions(env: MarketEnvironment, state: torch.Tensor, action: torch.Tensor, n_samples: int = 5):
    """
    env: environment providing reward function `env.reward(s,a,s_next)` and done predicate or terminal check
    stochastic_model.sample_next_states(s, a, n_samples) -> np.array (n_samples, state_dim)
    Returns list of (s, a, r, s_next)
    """
    # s_next_samples = stochastic_model.sample_next_states(s, action, n_samples)  # (n_samples, state_dim)
    simulated_next_state, r = env.projected_step(action, state, AIIF, horizon_timestep=0)
    return [(state, action, r, simulated_next_state)]

# Training loop
def train_loop(env: MarketEnvironment, initial_theta: np.ndarray, num_steps: int = 100000):
    state_dim = 12
    action_dim = 5
    buffer = ReplayBuffer(200000, state_dim, action_dim)
    agent = SACAgent(state_dim, action_dim)
    initial_state = env.reset(initial_theta)
    for t in range(num_steps):
        # Choose action (use actor for exploration)
        action = agent.select_action(initial_state)
        next_state, reward = env.step(action, AIIF)
        buffer.add(initial_state, action, reward, next_state) # add transition to replay buffer

        # Dyna imagination: generate model-based transitions from the *real* (s,a)
        imagined = generate_simulated_transitions(env, initial_state, action, n_samples=8)
        for (s_i, a, r, s_f) in imagined:
            buffer.add(s_i, a, r, s_f)

        # Train
        if buffer.size > 1024:
            for _ in range(1):  # gradient steps per env step
                batch = buffer.sample(256)
                agent.update(batch)

    return agent
