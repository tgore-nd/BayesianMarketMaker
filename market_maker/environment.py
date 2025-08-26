import duckdb
import numpy as np
import torch
import polars as pl
from datetime import datetime
from price_models.heston import generate_sample_paths
from parameter_estimation.mcmc_fast import hmc_sample_fast


class MarketEnvironment:
    """The true trading environment."""
    def __init__(self, ticker: str, deltalake_directory: str, start_date: str, initial_theta: np.ndarray):
        self.ticker = ticker.upper()
        self.deltalake_directory = deltalake_directory
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")

        # Import associated price data
        self.data = duckdb.sql(f"SELECT * FROM delta_scan('{deltalake_directory}') WHERE Ticker = '{ticker}' AND STRPTIME(timestamp, '%Y-%m-%d %H:%M:%S') >= STRPTIME('{start_date}', '%Y-%m-%d') ORDER BY timestamp")
        # self.data = duckdb.

        # Baseline parameters
        self.reset(initial_theta)

    def reset(self, initial_theta: np.ndarray) -> torch.Tensor:
        """Reset the environment and optionally return the initial state."""
        self.current_step: int = 1
        self.horizon_timestep = 0
        self.model_total_cash: float = 100.0
        self.model_total_inv: int = 10
        self.prices = self.data.select("open, high, low, close").limit(n=1, offset=self.current_step).pl()
        self.prev_prices = self.data.select("open, high, low, close").limit(n=1, offset=self.current_step - 1).pl()
        self.interest_rates = pl.read_csv("./data/DGS3MO.csv", schema={"observation_date": pl.Datetime, "DGS3MO": pl.Float64}, null_values="").filter(pl.col("observation_date") >= self.start_date)
        # Unfortunately, I only have access to OHLCV data. This could be way more nuanced with L1 or L2 data.

        # Set Heston params
        self.current_theta = initial_theta
        
        # Tracking variables (not returned, but used in state computation later)
        self.mid_prices = [(self.prices["high"][0] + self.prices["low"][0]) / 2]
        self.inv_count: list = [self.model_total_inv]
        self.dynamic_thresholds: list = [1]
        self.pnl: list = [0]
        self.profit: list = [0]

        # Environment state variables
        mid_prices_change = self.mid_prices[0]
        self.prices = self.data.select("open, high, low, close").limit(n=1, offset=self.current_step).pl()

        # Model state variables
        num_stocks_bought: int = 0 # num stocks bought in the interval
        num_stocks_sold: int = 0 # num stocks sold in the interval
        current_profit: float = 0.0 # the profit recieved from buying/selling in a given time step
        current_pnl: float = 0.0 # profit attributed to change in stock price

        # Reset simulation
        self.simulated_profit: list = self.profit
        self.simulated_inv_count: list = self.inv_count
        self.simulated_dynamic_thresholds: list = self.dynamic_thresholds
        self.simulated_pnl: list = self.pnl
        self.simulated_prev_close: float = self.prev_prices["close"][0]
        self.simulated_mid_prices: list = self.mid_prices
        self.current_simulated_theta: np.ndarray = self.current_theta
        self.simulation_start_index: int = len(self.simulated_profit)

        # Return
        env_state = torch.concat([self.prices.to_torch().ravel(), torch.tensor([mid_prices_change])], dim=0)
        model_state = torch.tensor([self.model_total_cash, self.model_total_inv, num_stocks_bought, num_stocks_sold, current_profit, current_pnl])
        self.state = torch.hstack([env_state, model_state])

        return self.state
    
    def reset_simulation(self) -> None:
        """Reset the simulation to build off the current actual values."""
        # Use current arrays as starting values
        self.simulated_profit: list = self.profit
        self.simulated_inv_count: list = self.inv_count
        self.simulated_dynamic_thresholds: list = self.dynamic_thresholds
        self.simulated_pnl: list = self.pnl
        self.simulated_prev_close: float = self.prev_prices["close"][0]
        self.simulated_mid_prices: list = self.mid_prices
        self.current_simulated_theta: np.ndarray = self.current_theta
        self.simulation_start_index: int = len(self.simulated_profit)

    def step(self, action: torch.Tensor, AIIF: float) -> tuple[torch.Tensor, float]:
        """Given an action in the MarketEnvironment, return the next OHLCV array in self.data and the model's total inventory and profit.
        
        The model observes state 0 and places trades to be filled in state 1. The state then updates, and `step()` determines what happens."""
        # Fetch action [buy_limit_price, buy_limit_num, sell_limit_price, sell_limit_num, num_stocks_market]
        # Advance prices to next state
        self.prev_prices = self.prices

        self.current_step += 1
        self.prices = self.data.select("open, high, low, close").limit(n=1, offset=self.current_step).pl()
        close_price = self.prices["close"][0]
        high_price = self.prices["high"][0]
        low_price = self.prices["low"][0]
        mid_price = (high_price + low_price) / 2

        # Conduct orders
        num_stocks_bought, num_stocks_sold, current_profit, hedging_penalty, self.model_total_cash, self.model_total_inv = self._order_fill(action, self.state)

        # Penalty function
        # Update the inventory and dynamic threshold arrays
        DITF = 0 if self.model_total_inv == 0 else self.model_total_cash / (self.model_total_inv * close_price)
        inv_penalty = AIIF * min(1, float(np.mean(self.inv_count) / np.mean(self.dynamic_thresholds)))

        # Appends
        self.mid_prices.append(mid_price)
        self.profit.append(current_profit)
        self.inv_count.append(self.model_total_inv)
        self.dynamic_thresholds.append(DITF * abs(self.model_total_cash / np.mean(self.mid_prices)))
        self.pnl.append(self.model_total_inv * (close_price - self.prev_prices["close"][0])) # evaluate price change effect on current inventory

        # Evaluate reward given action
        reward = current_profit + self.pnl[-1] - hedging_penalty - inv_penalty

        # Return the next state of the system and the reward
        mid_prices_change = self.mid_prices[-1] - self.mid_prices[-2]
        current_pnl = self.pnl[-1]

        env_state = torch.concat([self.prices.to_torch().ravel(), torch.tensor([mid_prices_change])], dim=0)
        model_state = torch.tensor([self.model_total_cash, self.model_total_inv, num_stocks_bought, num_stocks_sold, current_profit, current_pnl])
        self.state = torch.hstack([env_state, model_state])

        # state = [open, high, low, close, mid_prices_change, total_cash, total_inv, num_stocks_bought, num_stocks_sold, current_profit, current_pnl]
        return self.state, float(reward) # add model attributes: inventory, total cash, last quoted spread, estimated market spread
    
    def projected_step(self, action: torch.Tensor, simulated_state: torch.Tensor, AIIF: float, horizon_timestep: int, step_forward: bool = False, params: np.ndarray | None = None) -> tuple[torch.Tensor, float, np.ndarray]:
        """Given some action devised by the model at some given state, predict the value of the reward function. Make sure you run `reset_simulation()` first!"""
        # Get parameters (initial parameter is the previous value, current value is the current timestep)
        tau = 1 / 525600
        S0 = simulated_state[0].item()
        S = simulated_state[3].item()
        r = self.interest_rates["DGS3MO"][self.simulation_start_index + horizon_timestep]

        if step_forward: assert horizon_timestep == 0

        if params is None:
            # Sample parameters using observations from the previous minute
            const_params = np.array([S, S0, r, tau])
            results, accept_rate = hmc_sample_fast(self.current_simulated_theta, const_params, n_samples=10000, step_size=0.0001, n_steps=100)

            # Remove the first 7500 samples as burn-in and compute average params
            filtered_results = results[7500:, :] # chains tend to converge after ~7500 iterations of burn-in
            kappa, theta, sigma, rho, v0 = np.mean(filtered_results, axis=0)
        else:
            kappa, theta, sigma, rho, v0 = params
        
        # Get expected path over the next minute
        expected_path = np.mean(generate_sample_paths(tau, kappa, theta, sigma, rho, v0, S, r, 1000, 100)[0], axis=0)
        open_price = expected_path[0]
        close_price = expected_path[-1]
        high_price = np.max(expected_path)
        low_price = np.min(expected_path)
        mid_price = (high_price + low_price) / 2

        # Evaluate actions
        num_stocks_bought, num_stocks_sold, current_profit, hedging_penalty, model_total_cash, model_total_inv = self._order_fill(action, simulated_state)

        # Penalty function
        # Update the inventory and dynamic threshold arrays
        DITF = model_total_cash / (model_total_inv * close_price)
        inv_penalty = AIIF * min(1, float(np.mean(self.simulated_inv_count) / np.mean(self.simulated_dynamic_thresholds)))

        if step_forward:
            # Appends
            self.simulated_mid_prices.append(mid_price)
            self.simulated_profit.append(current_profit)
            self.simulated_inv_count.append(model_total_inv)
            self.simulated_dynamic_thresholds.append(DITF * abs(model_total_cash / np.mean(self.simulated_mid_prices)))
            self.simulated_pnl.append(model_total_inv * (close_price - self.simulated_prev_close)) # evaluate price change effect on current inventory (pnl)
            
            # Step forward
            self.horizon_timestep += 1

        # Evaluate reward given action
        reward = current_profit + self.simulated_pnl[-1] - hedging_penalty - inv_penalty

        # Return the next state of the system and the reward
        mid_prices_change = self.simulated_mid_prices[-1] - self.simulated_mid_prices[-2]
        current_pnl = model_total_inv * (S - S0)

        # Update current params
        self.simulated_prev_close = close_price
        self.current_simulated_theta = np.array([kappa, theta, sigma, rho, v0])

        env_state = torch.tensor([open_price, high_price, low_price, close_price, mid_prices_change])
        model_state = torch.tensor([model_total_cash, model_total_inv, num_stocks_bought, num_stocks_sold, current_profit, current_pnl])
        new_simulated_state = torch.hstack([env_state, model_state])

        return new_simulated_state, reward, np.array([kappa, theta, sigma, rho, v0]) # add model attributes: inventory, total cash, last quoted spread, estimated market spread
    
    @staticmethod
    def _order_fill(action: torch.Tensor, state: torch.Tensor) -> tuple[float, float, float, float, float, int]:
        # Action
        buy_limit_price, buy_limit_num, sell_limit_price, sell_limit_num, num_stocks_market = action.tolist()
        buy_limit_num = int(buy_limit_num)
        sell_limit_num = int(sell_limit_num)
        num_stocks_market = int(num_stocks_market)
        # num_stocks_market = 0

        # State
        open_price, high_price, low_price, close_price = state.tolist()[:4]
        mid_prices_change, model_total_cash, model_total_inv, num_stocks_bought, num_stocks_sold, current_profit, current_pnl = state.tolist()[4:]
        spread_estimate = high_price - low_price
        mid_price = (high_price + low_price) / 2
        fee_per_stock = 0.005

        # Reset individual params
        num_stocks_bought = 0
        num_stocks_sold = 0
        current_profit = 0
        hedging_penalty = 0

        # Execute orders
        # Limit orders
        buy_cost = (np.random.uniform(low_price, close_price) + fee_per_stock) * buy_limit_num
        if buy_limit_num > 0 and low_price <= buy_limit_price and model_total_cash - buy_cost >= 0:
            # Buy order (trade executes at a price in [low_price, next_price])
            model_total_cash -= buy_cost
            current_profit -= buy_cost
            model_total_inv += int(buy_limit_num)
            num_stocks_bought += int(buy_limit_num)
            print(f"BUY LIMIT SATISFIED: Bought {num_stocks_bought} at {buy_cost / buy_limit_num} per share.")
        elif sell_limit_num > 0 and high_price >= sell_limit_price and model_total_inv - sell_limit_num >= 0:
            # Sell order (trade executes at a price in [next_price, high_price])
            sell_price = (np.random.uniform(close_price, high_price) - fee_per_stock) * sell_limit_num
            model_total_cash += sell_price
            current_profit += sell_price
            model_total_inv -= int(sell_limit_num)
            num_stocks_sold += int(sell_limit_num)
            print(f"SELL LIMIT SATISFIED: Sold {abs(num_stocks_sold)} at {sell_price / sell_limit_num} per share.")
        
        # Market orders -- model pays the bid-ask spread
        market_price = np.random.uniform(mid_price - spread_estimate / 2, mid_price + spread_estimate / 2)
        if num_stocks_market > 0 and model_total_cash - market_price >= 0:
            # Buy order
            hedging_penalty = spread_estimate * num_stocks_market
            model_total_cash -= (market_price + hedging_penalty) * num_stocks_market
            current_profit -= market_price + hedging_penalty
            model_total_inv += int(num_stocks_market)
            num_stocks_bought += int(num_stocks_market)
            print(f"BUY MARKET SATISFIED: Bought {num_stocks_market} at {market_price}.")
        elif num_stocks_market < 0 and model_total_inv - abs(num_stocks_market) >= 0:
            # Sell order
            hedging_penalty = spread_estimate * abs(num_stocks_market)
            model_total_cash += (market_price - hedging_penalty) * abs(num_stocks_market)
            current_profit += market_price - hedging_penalty
            model_total_inv -= int(abs(num_stocks_market))
            num_stocks_sold += int(abs(num_stocks_market))
            print(f"SELL MARKET SATISFIED: Sold {abs(num_stocks_market)} at {market_price}.")
        
        print(f"Summary:\nStocks bought: {num_stocks_bought}\nStocks sold: {num_stocks_sold}\nCurrent profit: {current_profit}\nTotal cash: {model_total_cash}\nTotal inv: {model_total_inv}\nEstimated total value: {model_total_cash + model_total_inv * market_price}")
        return num_stocks_bought, num_stocks_sold, current_profit, hedging_penalty, model_total_cash, model_total_inv


if __name__ == "__main__":
    # Examples
    kappa = 2.0
    theta = 0.04
    sigma = 0.3
    rho = -0.7
    v0 = 0.04
    initial_theta = np.array([kappa, theta, sigma, rho, v0])
    env = MarketEnvironment("AAPL", r"C:\Users\tfgor\Documents\BayesianMarketMaker\data\deltalake", "2010-01-01", initial_theta)
    print("Done")

# Actor takes in current environment and determines best action
# Critic estimates the value function, V(s), which represents the expected cumulative reward starting from state s