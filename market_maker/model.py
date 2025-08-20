import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from scipy.stats import norm, halfnorm, uniform
from environment import MarketEnvironment


class PolicyNet(nn.Module):
    def __init__(self, state_dim=11, action_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh() # outputs actions in range [-1, 1]
        )
        # Bounds
        self.action_lower_bounds = torch.tensor([0, 0, 0, 0, -10])
        self.action_upper_bounds = torch.tensor([10, 10, 10, 10, 10])
    
    def forward(self, state) -> torch.Tensor:
        raw_action = self.net(state) # get the predictions in range [-1, 1]

        # Rescale from -1, 1
        scaled_action = 0.5 * (raw_action + 1.0) * (self.action_upper_bounds - self.action_lower_bounds) + self.action_lower_bounds
        return scaled_action


def plan_with_model(state: torch.Tensor, env: MarketEnvironment, candidates: int = 10) -> torch.Tensor:
    current_price = state[3].item()
    spread_estimate = state[1].item() - state[2].item()
    current_inv = state[6].item()
    current_total_cash = state[5].item()
    best_return = -np.inf
    best_action = torch.Tensor([0, 0, 0, 0, 0])
    for _ in range(candidates):
        # Randomly sample action, given 
        # [open, high, low, close, mid_prices_change, total_cash, total_inv, num_stocks_bought, num_stocks_sold, current_profit, current_pnl]
        # Sample the distance from the mean for each, not the value itself
        buy_limit_price = current_price - halfnorm.rvs(scale=spread_estimate) # will only quote at a lower price
        buy_limit_num = uniform.rvs(a=0, b=min(10, current_total_cash // buy_limit_price)) # make sure this can't buy more than can be afforded (or ensure that this is controlled for within env)
        sell_limit_price = current_price + halfnorm.rvs(scale=spread_estimate)
        sell_limit_num = uniform.rvs(a=0, b=min(10, current_inv)) # make sure this can't sell more than we have
        num_stocks_market = uniform.rvs(a=-current_inv, b=10)
        rand_action = torch.tensor([buy_limit_price, buy_limit_num, sell_limit_price, sell_limit_num, num_stocks_market])

        sim_state = state

        sim_state, reward = env.projected_step(rand_action, sim_state, 0.5)
        if reward > best_return:
            best_return = total_reward
            best_action = action
    
    return best_action


if __name__ == "__main__":
    # Training loop
    kappa = 2.0
    theta = 0.04
    sigma = 0.3
    rho = -0.7
    v0 = 0.04
    initial_theta = np.array([kappa, theta, sigma, rho, v0])
    env = MarketEnvironment("AAPL", r"data\deltalake", "2004-01-01", initial_theta=initial_theta)

    policy = PolicyNet()
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    replay_buffer = []

    for episode in range(100):
        state = env.reset(initial_theta)
        total_reward = 0
        for step in range(20):
            # Use model-based planning to select action
            action = plan_with_model(state, env)

            # Take real step
            next_state, reward = env.step(action, 0.5)
            total_reward += reward
            replay_buffer.append((state, action, reward, next_state)) # record how the transition of state -> next state occured given action

            # Train policy with real experience
            if len(replay_buffer) >= 10:
                batch = random.sample(replay_buffer, 10)
                states, actions, *_ = zip(*batch)

                states_tensor = torch.tensor(states, dtype=torch.float32).unsqueeze(1)
                actions_tensor = torch.tensor(actions, dtype=torch.float32).unsqueeze(1)
                pred_actions = policy(states_tensor)

                loss = nn.MSELoss()(pred_actions, actions_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state

        if episode % 10 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward:.2f}")