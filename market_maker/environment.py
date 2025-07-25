import torch
import duckdb
import numpy as np

class MarketEnvironment:
    """The true trading environment."""
    def __init__(self, ticker: str, deltalake_directory: str):
        self.ticker = ticker.upper()
        self.deltalake_directory = deltalake_directory

        # Import associated price data
        self.data = duckdb.sql(f"SELECT * FROM delta_scan('{deltalake_directory}') WHERE Ticker = '{ticker}' ORDER BY timestamp")

        # Baseline parameters
        self.reset(return_state=False)

    def reset(self, return_state=True) -> torch.Tensor | None:
        """Reset the environment and optionally return the initial state."""
        self.current_step: int = 0
        self.model_total_cash: float = 0.0
        self.model_total_inv: int = 0
        self.prices = self.data.select("open, high, low, close, volume").limit(n=1, offset=self.current_step).pl()
        # Unfortunately, I only have access to OHLCV data. This could be way more nuanced with L1 or L2 data.
        
        # Tracking variables (not returned, but used in state computation later)
        self.mid_prices = [(self.prices["high"][0] + self.prices["low"][0]) / 2]
        self.inv_count = [0]
        self.dynamic_thresholds = [1]
        self.pnl = [0]
        self.profit = [0]

        # Environment state variables
        mid_prices_change = self.mid_prices[0]
        self.prices = self.data.select("open, high, low, close, volume").limit(n=1, offset=self.current_step).pl()

        # Model state variables
        self.model_total_cash: float = 0.0 # profit attributed to change in stock price
        self.model_total_inv: int = 0
        num_stocks_bought: int = 0 # num stocks bought in the interval
        num_stocks_sold: int = 0 # num stocks sold in the interval
        current_profit: float = 0.0 # the profit recieved from buying/selling in a given time step
        current_pnl: float = 0.0 # profit attributed to change in stock price

        env_state = torch.concat([self.prices.to_torch(), torch.tensor([mid_prices_change])])
        model_state = torch.tensor([self.model_total_cash, self.model_total_inv, num_stocks_bought, num_stocks_sold, current_profit, current_pnl])
        self.state = torch.concat([env_state, model_state], dim=2)
        if return_state:
            return self.state

    def step(self, action: torch.Tensor, AIIF: float) -> tuple[torch.Tensor, float]:
        """Given an action in the MarketEnvironment, return the next OHLCV array in self.data and the model's total inventory and profit.
        
        The model observes state 0 and places trades to be filled in state 1. The state then updates, and `step()` determines what happens."""
        # Fetch action
        buy_limit_order, sell_limit_order, num_stocks_market = action
        buy_limit_price, buy_limit_num = buy_limit_order
        sell_limit_price, sell_limit_num = sell_limit_order

        # Advance prices to next state
        self.current_step += 1
        self.prev_prices = self.prices
        self.prices = self.data.select("open, high, low, close, volume").limit(n=1, offset=self.current_step).pl()

        # Reset individual params
        num_stocks_bought = 0
        num_stocks_sold = 0

        # Evaluate next state given the action
        close_price = self.prices["close"][0] # the price at the end of the timestep during which the order is active
        high_price = self.prices["high"][0]
        low_price = self.prices["low"][0]
        spread_estimate = high_price - low_price
        mid_price = (high_price + low_price) / 2
        fee_per_stock = 0.005

        # Execute orders
        # Limit orders
        buy_cost = (np.random.uniform(low_price, close_price) + fee_per_stock) * buy_limit_num
        current_profit = 0
        if buy_limit_num > 0 and low_price <= buy_limit_price and self.model_total_cash - buy_cost >= 0:
            # Buy order (trade executes at a price in [low_price, next_price])
            self.model_total_cash -= buy_cost
            current_profit -= buy_cost
            self.model_total_inv += int(buy_limit_num)
            num_stocks_bought += int(buy_limit_num)
        elif sell_limit_num > 0 and high_price >= sell_limit_price and self.model_total_inv - sell_limit_num >= 0:
            # Sell order (trade executes at a price in [next_price, high_price])
            sell_price = (np.random.uniform(close_price, high_price) - fee_per_stock) * sell_limit_num
            self.model_total_cash += sell_price
            current_profit += sell_price
            self.model_total_inv -= int(sell_limit_num)
            num_stocks_sold += int(sell_limit_num)
        
        # Market orders -- model pays the bid-ask spread
        market_price = np.random.uniform(mid_price - spread_estimate / 2, mid_price + spread_estimate / 2)
        hedging_penalty = 0
        if num_stocks_market > 0 and self.model_total_cash - market_price >= 0:
            # Buy order
            hedging_penalty = spread_estimate * num_stocks_market
            self.model_total_cash -= (market_price + hedging_penalty) * num_stocks_market
            current_profit -= market_price + hedging_penalty
            self.model_total_inv += int(num_stocks_market)
            num_stocks_bought += int(num_stocks_market)
        elif num_stocks_market < 0 and self.model_total_inv - abs(num_stocks_market) >= 0:
            # Sell order
            hedging_penalty = spread_estimate * abs(num_stocks_market)
            self.model_total_cash += (market_price - hedging_penalty) * abs(num_stocks_market)
            current_profit += market_price - hedging_penalty
            self.model_total_inv -= int(abs(num_stocks_market))
            num_stocks_sold += int(abs(num_stocks_market))

        # Penalty function
        # Update the inventory and dynamic threshold arrays
        DITF = self.model_total_cash / (self.model_total_inv * close_price)
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

        env_state = torch.concat([self.prices.to_torch(), torch.tensor([mid_prices_change])])
        model_state = torch.tensor([self.model_total_cash, self.model_total_inv, num_stocks_bought, num_stocks_sold, current_profit, current_pnl])
        self.state = torch.concat([env_state, model_state], dim=2)

        return self.state, reward # add model attributes: inventory, total cash, last quoted spread, estimated market spread


# Examples
limit_order = {"price": 0.3, "num_units": 5} # sell takes same form, but negative num_units
limit_order = torch.tensor([0.3, 5])

hedge_market_order = {"num_units_buy": 5, "num_units_sell": 0}
hedge_market_order = torch.tensor([5, 0])

# may have to change this into PyTorch tensor form later on