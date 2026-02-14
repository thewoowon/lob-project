"""
Stateless LOB features — computed from a single snapshot only.

Covers:
  - Price features (5 of 6; volatility is stateful)
  - Volume features (8)
  - Order Imbalance features (6)
  - Depth features (6)
  - Price Impact features (6)

Total: 31 stateless features
"""

import numpy as np
from typing import Dict

from ..config import EPSILON, N_LEVELS, OI_WEIGHTS, OI_WEIGHT_SUM, STANDARD_ORDER_SIZE
from ..core.state_manager import LOBSnapshot


# ── feature names (deterministic order) ──────────────────────────────

STATELESS_FEATURE_NAMES: list[str] = [
    # Price (5)
    'mid_price',
    'weighted_mid_price',
    'spread_absolute',
    'spread_relative',
    'log_mid_price',
    # Volume (8)
    'bid_ask_volume_ratio_1',
    'bid_ask_volume_ratio_2',
    'bid_ask_volume_ratio_3',
    'bid_ask_volume_ratio_4',
    'bid_ask_volume_ratio_5',
    'cumulative_bid_volume',
    'cumulative_ask_volume',
    'volume_imbalance_total',
    # Order Imbalance (6)
    'oi_level_1',
    'oi_level_2',
    'oi_level_3',
    'oi_total',
    'oi_weighted',
    'oi_asymmetry',
    # Depth (6)
    'depth_imbalance',
    'depth_ratio',
    'effective_spread',
    'queue_position_proxy',
    'depth_weighted_mid_price',
    'liquidity_concentration',
    # Price Impact (6)
    'market_order_impact_buy',
    'market_order_impact_sell',
    'impact_asymmetry',
    'resilience_proxy',
    'adverse_selection_risk',
    'execution_cost_estimate',
]

assert len(STATELESS_FEATURE_NAMES) == 31


# ── main entry point ─────────────────────────────────────────────────

def compute_stateless_features(snap: LOBSnapshot, out: np.ndarray | None = None) -> np.ndarray:
    """
    Compute all 31 stateless features.

    Args:
        snap: Current LOB snapshot.
        out: Optional pre-allocated array of shape (31,) to write into.

    Returns:
        np.ndarray of shape (31,).
    """
    if out is None:
        out = np.empty(31, dtype=np.float64)

    ap = snap.ask_prices
    av = snap.ask_volumes
    bp = snap.bid_prices
    bv = snap.bid_volumes

    # -- precompute shared quantities --
    mid = (ap[0] + bp[0]) * 0.5
    total_av = av.sum()
    total_bv = bv.sum()
    total_vol = total_av + total_bv

    # ── Price (5) ────────────────────────────────────────────────
    idx = 0
    out[idx] = mid;                                          idx += 1  # mid_price
    vwap_ask = (ap * av).sum() / (total_av + EPSILON)
    vwap_bid = (bp * bv).sum() / (total_bv + EPSILON)
    out[idx] = (vwap_ask + vwap_bid) * 0.5;                 idx += 1  # weighted_mid_price
    spread = ap[0] - bp[0]
    out[idx] = spread;                                       idx += 1  # spread_absolute
    out[idx] = spread / (mid + EPSILON);                     idx += 1  # spread_relative
    out[idx] = np.log(mid + EPSILON);                        idx += 1  # log_mid_price

    # ── Volume (8) ───────────────────────────────────────────────
    for i in range(5):
        out[idx] = bv[i] / (av[i] + EPSILON);               idx += 1  # bid_ask_volume_ratio 1-5
    out[idx] = total_bv;                                     idx += 1  # cumulative_bid_volume
    out[idx] = total_av;                                     idx += 1  # cumulative_ask_volume
    out[idx] = (total_bv - total_av) / (total_vol + EPSILON); idx += 1  # volume_imbalance_total

    # ── Order Imbalance (6) ──────────────────────────────────────
    for i in range(3):
        denom = bv[i] + av[i] + EPSILON
        out[idx] = (bv[i] - av[i]) / denom;                 idx += 1  # oi_level 1-3
    out[idx] = (total_bv - total_av) / (total_vol + EPSILON); idx += 1  # oi_total

    # oi_weighted
    oi_sum = 0.0
    for i in range(N_LEVELS):
        denom = bv[i] + av[i] + EPSILON
        oi_sum += OI_WEIGHTS[i] * (bv[i] - av[i]) / denom
    out[idx] = oi_sum / OI_WEIGHT_SUM;                      idx += 1  # oi_weighted

    # oi_asymmetry
    bid_top = bv[0] + bv[1] + bv[2]
    ask_top = av[0] + av[1] + av[2]
    oi_top = (bid_top - ask_top) / (bid_top + ask_top + EPSILON)
    bid_deep = total_bv - bid_top
    ask_deep = total_av - ask_top
    oi_deep = (bid_deep - ask_deep) / (bid_deep + ask_deep + EPSILON)
    out[idx] = oi_top - oi_deep;                             idx += 1  # oi_asymmetry

    # ── Depth (6) ────────────────────────────────────────────────
    out[idx] = total_bv - total_av;                          idx += 1  # depth_imbalance
    out[idx] = total_bv / (total_av + EPSILON);              idx += 1  # depth_ratio
    out[idx] = vwap_ask - vwap_bid;                          idx += 1  # effective_spread
    out[idx] = (bv[0] + av[0]) * 0.5;                       idx += 1  # queue_position_proxy

    # depth_weighted_mid_price
    total_wp = (ap * av + bp * bv).sum()
    out[idx] = total_wp / (total_vol + EPSILON);             idx += 1  # depth_weighted_mid_price

    lev1_vol = bv[0] + av[0]
    out[idx] = lev1_vol / (total_vol + EPSILON);             idx += 1  # liquidity_concentration

    # ── Price Impact (6) ─────────────────────────────────────────
    buy_impact = _estimate_impact(ap, av, STANDARD_ORDER_SIZE, is_buy=True)
    sell_impact = _estimate_impact(bp, bv, STANDARD_ORDER_SIZE, is_buy=False)
    out[idx] = buy_impact;                                   idx += 1
    out[idx] = sell_impact;                                  idx += 1
    out[idx] = buy_impact - sell_impact;                     idx += 1  # impact_asymmetry
    out[idx] = lev1_vol / (total_vol + EPSILON);             idx += 1  # resilience_proxy
    out[idx] = spread / (lev1_vol + EPSILON);                idx += 1  # adverse_selection_risk
    out[idx] = buy_impact + sell_impact;                     idx += 1  # execution_cost_estimate

    return out


# ── helpers ──────────────────────────────────────────────────────────

def _estimate_impact(
    prices: np.ndarray,
    volumes: np.ndarray,
    order_size: float,
    is_buy: bool,
) -> float:
    """Walk the book and compute average execution price impact."""
    remaining = order_size
    total_cost = 0.0
    executed = 0.0
    for i in range(N_LEVELS):
        if remaining <= 0.0:
            break
        fill = min(remaining, volumes[i])
        total_cost += fill * prices[i]
        executed += fill
        remaining -= fill
    if executed == 0.0:
        return 0.0
    avg_price = total_cost / executed
    if is_buy:
        return avg_price - prices[0]
    else:
        return prices[0] - avg_price
