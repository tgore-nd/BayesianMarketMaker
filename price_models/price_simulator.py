import numpy as np
from typing import Callable, Literal

def sample_from_posterior(price_model: Callable, parameters: tuple):
    """Sample from price_model using a tuple of arguments parameters."""
    return price_model(*parameters)


def get_fill_prob(quoted_price: float, predicted_prices: np.ndarray, side='ask') -> tuple[float, Literal[0, 1]]:
    """
    Estimate the probability that your quote gets filled.
    
    Parameters:
    - quoted_price: your limit price (float)
    - predicted_prices: array of future price samples (shape: (M paths, N samples from each path) -- each column is a sample from every path)
    - side: 'ask' (you sell) or 'bid' (you buy)
    - spread: assumed average market spread
    
    Returns:
    - fill_prob: estimated probability of getting filled
    """
    if side == 'ask':
        fill_events = predicted_prices >= quoted_price # could weight this by volume traded above/below price
    elif side == 'bid':
        fill_events = predicted_prices <= quoted_price
    else:
        raise ValueError("side must be 'ask' or 'bid'")

    fill_prob = np.mean([any(row) for row in fill_events], axis=0)

    if np.random.rand() < fill_prob:
        fill = 1
    else:
        fill = 0

    return fill_prob, fill