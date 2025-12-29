"""
Order Flow Imbalance (OFI) Features (6 features)

Based on PAPER_DRAFT.md Section 3.3.4:

Theory: OFI (Cont et al., 2014) measures net order flow changes.
        Strong predictor of price movements.

OFI formula:
    OFI_bid = ΔV_bid × I[ΔP_bid ≥ 0]
    OFI_ask = ΔV_ask × I[ΔP_ask ≤ 0]
    OFI_net = OFI_bid - OFI_ask

1. OFI bid (bid-side order flow)
2. OFI ask (ask-side order flow)
3. OFI net (net order flow)
4. OFI ratio
5. Cumulative OFI (5-event window)
6. OFI volatility (5-event std)

⚠️ CRITICAL: Uses t and t-1 ONLY (no future data!)
"""

import numpy as np
from typing import Dict, List, Optional

EPSILON = 1e-10


def compute_order_flow_imbalance_features(
    current_snapshot: Dict,
    previous_snapshot: Optional[Dict],
    history_buffer: List[Dict]
) -> np.ndarray:
    """
    Compute 6 order flow imbalance features.

    Args:
        current_snapshot: Current LOB snapshot (time t)
        previous_snapshot: Previous LOB snapshot (time t-1), None if first event
        history_buffer: List of past snapshots (for cumulative OFI)

    Returns:
        ofi_features: np.ndarray of shape (6,)
            [ofi_bid, ofi_ask, ofi_net, ofi_ratio,
             ofi_cumulative, ofi_volatility]
    """
    # If no previous snapshot, return zeros (first event)
    if previous_snapshot is None or len(history_buffer) == 0:
        return np.zeros(6, dtype=np.float64)

    # Extract current and previous prices/volumes at level 1
    curr_bid_price = current_snapshot['bid_price_1']
    curr_bid_volume = current_snapshot['bid_volume_1']
    curr_ask_price = current_snapshot['ask_price_1']
    curr_ask_volume = current_snapshot['ask_volume_1']

    prev_bid_price = previous_snapshot['bid_price_1']
    prev_bid_volume = previous_snapshot['bid_volume_1']
    prev_ask_price = previous_snapshot['ask_price_1']
    prev_ask_volume = previous_snapshot['ask_volume_1']

    # Calculate deltas (ΔP and ΔV)
    delta_bid_price = curr_bid_price - prev_bid_price
    delta_bid_volume = curr_bid_volume - prev_bid_volume
    delta_ask_price = curr_ask_price - prev_ask_price
    delta_ask_volume = curr_ask_volume - prev_ask_volume

    # 1. OFI bid
    # If bid price increased or stayed same, buy market orders absorbed ask liquidity
    ofi_bid = delta_bid_volume if delta_bid_price >= 0 else 0.0

    # 2. OFI ask
    # If ask price decreased or stayed same, sell market orders absorbed bid liquidity
    ofi_ask = delta_ask_volume if delta_ask_price <= 0 else 0.0

    # 3. OFI net
    ofi_net = ofi_bid - ofi_ask

    # 4. OFI ratio
    ofi_ratio = ofi_bid / (abs(ofi_ask) + EPSILON)

    # 5. Cumulative OFI (5-event window)
    # Need to compute OFI for last 5 events
    if len(history_buffer) < 2:
        ofi_cumulative = ofi_net
    else:
        # Compute OFI for each pair in history buffer
        past_ofi_net_values = []
        for i in range(1, min(5, len(history_buffer))):
            prev_snap = history_buffer[-(i+1)]
            curr_snap = history_buffer[-i]

            d_bid_price = curr_snap['bid_price_1'] - prev_snap['bid_price_1']
            d_bid_volume = curr_snap['bid_volume_1'] - prev_snap['bid_volume_1']
            d_ask_price = curr_snap['ask_price_1'] - prev_snap['ask_price_1']
            d_ask_volume = curr_snap['ask_volume_1'] - prev_snap['ask_volume_1']

            past_ofi_bid = d_bid_volume if d_bid_price >= 0 else 0.0
            past_ofi_ask = d_ask_volume if d_ask_price <= 0 else 0.0
            past_ofi_net_values.append(past_ofi_bid - past_ofi_ask)

        # Add current OFI
        past_ofi_net_values.append(ofi_net)

        # Cumulative sum
        ofi_cumulative = sum(past_ofi_net_values)

    # 6. OFI volatility (5-event std)
    if len(history_buffer) < 2:
        ofi_volatility = 0.0
    else:
        # Reuse past_ofi_net_values from step 5
        past_ofi_net_values = []
        for i in range(1, min(5, len(history_buffer))):
            prev_snap = history_buffer[-(i+1)]
            curr_snap = history_buffer[-i]

            d_bid_price = curr_snap['bid_price_1'] - prev_snap['bid_price_1']
            d_bid_volume = curr_snap['bid_volume_1'] - prev_snap['bid_volume_1']
            d_ask_price = curr_snap['ask_price_1'] - prev_snap['ask_price_1']
            d_ask_volume = curr_snap['ask_volume_1'] - prev_snap['ask_volume_1']

            past_ofi_bid = d_bid_volume if d_bid_price >= 0 else 0.0
            past_ofi_ask = d_ask_volume if d_ask_price <= 0 else 0.0
            past_ofi_net_values.append(past_ofi_bid - past_ofi_ask)

        past_ofi_net_values.append(ofi_net)

        ofi_volatility = np.std(past_ofi_net_values, ddof=1) if len(past_ofi_net_values) > 1 else 0.0

    return np.array([
        ofi_bid,
        ofi_ask,
        ofi_net,
        ofi_ratio,
        ofi_cumulative,
        ofi_volatility
    ], dtype=np.float64)


def get_order_flow_imbalance_feature_names() -> List[str]:
    """
    Get names of 6 OFI features.

    Returns:
        List of 6 feature names
    """
    return [
        'ofi_bid',
        'ofi_ask',
        'ofi_net',
        'ofi_ratio',
        'ofi_cumulative',
        'ofi_volatility'
    ]
