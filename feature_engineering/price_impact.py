"""
Price Impact Features (6 features)

Based on PAPER_DRAFT.md Section 3.3.6:

Theory: Price impact estimates how order flow moves prices (Almgren et al., 2005).

1. Market order impact (buy) - price impact of buy market order
2. Market order impact (sell) - price impact of sell market order
3. Impact asymmetry (buy impact - sell impact)
4. Resilience proxy (price reversion speed estimate)
5. Adverse selection risk
6. Execution cost estimate

Note: PAPER_DRAFT.md ablation study shows this is the MOST valuable feature group (+2.41pp)
"""

import numpy as np
from typing import Dict, List

EPSILON = 1e-10
STANDARD_ORDER_SIZE = 1000.0  # Standard order size for impact estimation


def compute_price_impact_features(lob_snapshot: Dict) -> np.ndarray:
    """
    Compute 6 price impact features.

    Args:
        lob_snapshot: Current LOB snapshot

    Returns:
        impact_features: np.ndarray of shape (6,)
            [market_order_impact_buy, market_order_impact_sell,
             impact_asymmetry, resilience_proxy,
             adverse_selection_risk, execution_cost_estimate]
    """
    # Extract prices and volumes
    ask_prices = [lob_snapshot[f'ask_price_{i}'] for i in range(1, 11)]
    bid_prices = [lob_snapshot[f'bid_price_{i}'] for i in range(1, 11)]
    ask_volumes = [lob_snapshot[f'ask_volume_{i}'] for i in range(1, 11)]
    bid_volumes = [lob_snapshot[f'bid_volume_{i}'] for i in range(1, 11)]

    # 1. Market order impact (buy)
    # If we submit a market buy order, how much will price move?
    market_order_impact_buy = estimate_buy_impact(
        ask_prices, ask_volumes, STANDARD_ORDER_SIZE
    )

    # 2. Market order impact (sell)
    market_order_impact_sell = estimate_sell_impact(
        bid_prices, bid_volumes, STANDARD_ORDER_SIZE
    )

    # 3. Impact asymmetry
    impact_asymmetry = market_order_impact_buy - market_order_impact_sell

    # 4. Resilience proxy
    # How quickly does price revert after impact?
    # Proxy: ratio of level 1 volume to total volume (high = fast reversion)
    total_bid_volume = sum(bid_volumes)
    total_ask_volume = sum(ask_volumes)
    total_volume = total_bid_volume + total_ask_volume
    resilience_proxy = (bid_volumes[0] + ask_volumes[0]) / (total_volume + EPSILON)

    # 5. Adverse selection risk
    # Risk that informed traders are on the other side
    # Proxy: spread relative to depth at level 1
    spread = ask_prices[0] - bid_prices[0]
    level_1_depth = bid_volumes[0] + ask_volumes[0]
    adverse_selection_risk = spread / (level_1_depth + EPSILON)

    # 6. Execution cost estimate
    # Expected cost to execute a round-trip trade (buy then sell)
    execution_cost_estimate = market_order_impact_buy + market_order_impact_sell

    return np.array([
        market_order_impact_buy,
        market_order_impact_sell,
        impact_asymmetry,
        resilience_proxy,
        adverse_selection_risk,
        execution_cost_estimate
    ], dtype=np.float64)


def estimate_buy_impact(
    ask_prices: List[float],
    ask_volumes: List[float],
    order_size: float
) -> float:
    """
    Estimate price impact of a buy market order.

    Simulates absorbing ask liquidity at each level.

    Args:
        ask_prices: List of 10 ask prices
        ask_volumes: List of 10 ask volumes
        order_size: Size of market order

    Returns:
        impact: Price movement (avg_execution_price - best_ask_price)
    """
    remaining_size = order_size
    total_cost = 0.0
    executed_size = 0.0

    for price, volume in zip(ask_prices, ask_volumes):
        if remaining_size <= 0:
            break

        # Execute as much as possible at this level
        executed_at_level = min(remaining_size, volume)
        total_cost += executed_at_level * price
        executed_size += executed_at_level
        remaining_size -= executed_at_level

    if executed_size == 0:
        return 0.0

    # Average execution price
    avg_execution_price = total_cost / executed_size

    # Impact = difference from best ask price
    impact = avg_execution_price - ask_prices[0]

    return impact


def estimate_sell_impact(
    bid_prices: List[float],
    bid_volumes: List[float],
    order_size: float
) -> float:
    """
    Estimate price impact of a sell market order.

    Simulates absorbing bid liquidity at each level.

    Args:
        bid_prices: List of 10 bid prices
        bid_volumes: List of 10 bid volumes
        order_size: Size of market order

    Returns:
        impact: Price movement (best_bid_price - avg_execution_price)
    """
    remaining_size = order_size
    total_proceeds = 0.0
    executed_size = 0.0

    for price, volume in zip(bid_prices, bid_volumes):
        if remaining_size <= 0:
            break

        # Execute as much as possible at this level
        executed_at_level = min(remaining_size, volume)
        total_proceeds += executed_at_level * price
        executed_size += executed_at_level
        remaining_size -= executed_at_level

    if executed_size == 0:
        return 0.0

    # Average execution price
    avg_execution_price = total_proceeds / executed_size

    # Impact = difference from best bid price
    impact = bid_prices[0] - avg_execution_price

    return impact


def get_price_impact_feature_names() -> List[str]:
    """
    Get names of 6 price impact features.

    Returns:
        List of 6 feature names
    """
    return [
        'market_order_impact_buy',
        'market_order_impact_sell',
        'impact_asymmetry',
        'resilience_proxy',
        'adverse_selection_risk',
        'execution_cost_estimate'
    ]
