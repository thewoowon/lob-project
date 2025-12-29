"""
Price Features (6 features)

Based on PAPER_DRAFT.md Section 3.3.1:
1. Mid-price (level 1)
2. Weighted mid-price (VWAP across 10 levels)
3. Bid-ask spread (absolute)
4. Bid-ask spread (relative)
5. Log mid-price
6. Mid-price volatility (5-event rolling std)
"""

import numpy as np
from typing import Dict, List

# Small epsilon for numerical stability
EPSILON = 1e-10


def compute_price_features(
    lob_snapshot: Dict,
    history_buffer: List[Dict]
) -> np.ndarray:
    """
    Compute 6 price-based features.

    Args:
        lob_snapshot: Current LOB snapshot with ask_price_{1-10}, bid_price_{1-10}, etc.
        history_buffer: List of past LOB snapshots (for volatility calculation)

    Returns:
        price_features: np.ndarray of shape (6,)
            [mid_price, weighted_mid_price, spread_absolute,
             spread_relative, log_mid_price, mid_price_volatility]
    """
    # Extract prices and volumes
    ask_prices = [lob_snapshot[f'ask_price_{i}'] for i in range(1, 11)]
    bid_prices = [lob_snapshot[f'bid_price_{i}'] for i in range(1, 11)]
    ask_volumes = [lob_snapshot[f'ask_volume_{i}'] for i in range(1, 11)]
    bid_volumes = [lob_snapshot[f'bid_volume_{i}'] for i in range(1, 11)]

    # 1. Mid-price (level 1)
    mid_price = (ask_prices[0] + bid_prices[0]) / 2.0

    # 2. Weighted mid-price (VWAP across 10 levels)
    total_ask_volume = sum(ask_volumes) + EPSILON
    total_bid_volume = sum(bid_volumes) + EPSILON

    vwap_ask = sum(p * v for p, v in zip(ask_prices, ask_volumes)) / total_ask_volume
    vwap_bid = sum(p * v for p, v in zip(bid_prices, bid_volumes)) / total_bid_volume
    weighted_mid_price = (vwap_ask + vwap_bid) / 2.0

    # 3. Bid-ask spread (absolute)
    spread_absolute = ask_prices[0] - bid_prices[0]

    # 4. Bid-ask spread (relative)
    spread_relative = spread_absolute / (mid_price + EPSILON)

    # 5. Log mid-price
    log_mid_price = np.log(mid_price + EPSILON)

    # 6. Mid-price volatility (5-event rolling std)
    if len(history_buffer) < 2:
        # Not enough history, use zero
        mid_price_volatility = 0.0
    else:
        # Get last 5 mid-prices (including current)
        past_mid_prices = []
        for snapshot in history_buffer[-4:]:  # Last 4 from buffer
            past_mid = (snapshot['ask_price_1'] + snapshot['bid_price_1']) / 2.0
            past_mid_prices.append(past_mid)
        past_mid_prices.append(mid_price)  # Add current

        # Calculate standard deviation
        mid_price_volatility = np.std(past_mid_prices, ddof=1) if len(past_mid_prices) > 1 else 0.0

    return np.array([
        mid_price,
        weighted_mid_price,
        spread_absolute,
        spread_relative,
        log_mid_price,
        mid_price_volatility
    ], dtype=np.float64)


def get_price_feature_names() -> List[str]:
    """
    Get names of 6 price features.

    Returns:
        List of 6 feature names
    """
    return [
        'mid_price',
        'weighted_mid_price',
        'spread_absolute',
        'spread_relative',
        'log_mid_price',
        'mid_price_volatility'
    ]
