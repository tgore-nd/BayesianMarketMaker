import numpy as np

def state(stocks_bought: np.ndarray, stocks_sold: np.ndarray, inv_sizes: np.ndarray, ):
    # maybe include projection of prices for next several time steps, with some discount factor included?
    pass



def reward(profit, pnl, hedging_penalty, inv_penalty):
    """
    profit: the profit received from step t_i from buying or selling
    pnl: the profit attributed to the currently held inventory; inv * change in price from prev timestep
    hedging_penalty: the penalty paid for hedging risk; the model pays the bid-ask spread; equal to amt_inv_hedged * market_spread
    inv_penalty: the penalty of holding inventory; adaptive method in automated market maker paper
    """
    return profit + pnl - hedging_penalty - inv_penalty
    # or: reward = fill_probability * spread - inv_penalty


def inv_penalty(AIIF, total_cash, total_inv_val, mid_prices: np.ndarray, inv_sizes: np.ndarray, dynamic_thresholds: np.ndarray):
    # AIIF will probably have to be estimated via optimization
    # the mid price is the average between the bid and ask prices
    DITF = total_cash / total_inv_val

    current_dynamic_threshold = DITF * abs(total_cash / np.mean(mid_prices))
    new_dynamic_thresholds = np.concatenate([dynamic_thresholds, np.array([current_dynamic_threshold])])

    ratio = float(np.mean(inv_sizes) / np.mean(new_dynamic_thresholds))
    
    return AIIF * ratio


def trading_profit(model_buy_spread, model_sell_spread, num_stocks_bought, num_stocks_sold):
    return num_stocks_bought * model_buy_spread + num_stocks_sold * model_sell_spread

def pnl(inv, price_diff):
    return inv * price_diff # note that price_diff is the difference between the current and previous stock price

def hedging_cost(amt_inv_hedged, market_spread):
    return amt_inv_hedged * market_spread # the model pays the market spread