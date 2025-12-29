"""
Depth Features (6 features)

Based on PAPER_DRAFT.md Section 3.3.5:

1. Depth imbalance (total_bid_volume - total_ask_volume)
2. Depth ratio (total_bid_volume / total_ask_volume)
3. Effective spread (volume-weighted)
4. Queue position proxy
5. Depth-weighted mid-price
6. Liquidity concentration (level 1 volume / total volume)
"""

import numpy as np
from typing import Dict, List

EPSILON = 1e-10


def compute_depth_features(lob_snapshot: Dict) -> np.ndarray:
    """
    Compute 6 depth-based features.

    Args:
        lob_snapshot: Current LOB snapshot

    Returns:
        depth_features: np.ndarray of shape (6,)
            [depth_imbalance, depth_ratio, effective_spread,
             queue_position_proxy, depth_weighted_mid_price,
             liquidity_concentration]
    """
    # Extract prices and volumes
    ask_prices = [lob_snapshot[f'ask_price_{i}'] for i in range(1, 11)]
    bid_prices = [lob_snapshot[f'bid_price_{i}'] for i in range(1, 11)]
    ask_volumes = [lob_snapshot[f'ask_volume_{i}'] for i in range(1, 11)]
    bid_volumes = [lob_snapshot[f'bid_volume_{i}'] for i in range(1, 11)]

    # 1. Depth imbalance
    total_bid_volume = sum(bid_volumes)
    total_ask_volume = sum(ask_volumes)
    depth_imbalance = total_bid_volume - total_ask_volume

    # 2. Depth ratio
    depth_ratio = total_bid_volume / (total_ask_volume + EPSILON)

    # 3. Effective spread (volume-weighted)
    vwap_ask = sum(p * v for p, v in zip(ask_prices, ask_volumes)) / (total_ask_volume + EPSILON)
    vwap_bid = sum(p * v for p, v in zip(bid_prices, bid_volumes)) / (total_bid_volume + EPSILON)
    effective_spread = vwap_ask - vwap_bid

    # 4. Queue position proxy
    # Estimate: If we place an order at level 1, how many orders are ahead?
    # Simplified: average of level 1 volumes
    queue_position_proxy = (bid_volumes[0] + ask_volumes[0]) / 2.0

    # 5. Depth-weighted mid-price
    # Price weighted by volume at each level
    total_weighted_price = sum(
        (ask_prices[i] * ask_volumes[i] + bid_prices[i] * bid_volumes[i])
        for i in range(10)
    )
    total_volume = total_ask_volume + total_bid_volume
    depth_weighted_mid_price = total_weighted_price / (total_volume + EPSILON)

    # 6. Liquidity concentration
    # What fraction of liquidity is at level 1?
    level_1_volume = bid_volumes[0] + ask_volumes[0]
    liquidity_concentration = level_1_volume / (total_volume + EPSILON)

    return np.array([
        depth_imbalance,
        depth_ratio,
        effective_spread,
        queue_position_proxy,
        depth_weighted_mid_price,
        liquidity_concentration
    ], dtype=np.float64)


def get_depth_feature_names() -> List[str]:
    """
    Get names of 6 depth features.

    Returns:
        List of 6 feature names
    """
    return [
        'depth_imbalance',
        'depth_ratio',
        'effective_spread',
        'queue_position_proxy',
        'depth_weighted_mid_price',
        'liquidity_concentration'
    ]
