"""
Unified real-time feature engine.

Orchestrates state management, stateless features, and stateful features
into a single process_event() call that outputs a 78-feature vector
identical to the batch FeatureEngineeringPipeline.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional

from .state_manager import LOBSnapshot, StateManager
from .profiler import Profiler
from ..features.stateless import compute_stateless_features, STATELESS_FEATURE_NAMES
from ..features.stateful import StatefulFeatureEngine, STATEFUL_FEATURE_NAMES
from ..config import WINDOW_SIZE


# Feature vector layout (78 total):
#   [0:40]   raw features (ask_prices, ask_volumes, bid_prices, bid_volumes)
#   [40:71]  stateless engineered (31)
#   [71:78]  stateful engineered (7)
#
# Matches batch pipeline order:
#   raw(40) + price(6) + volume(8) + OI(6) + OFI(6) + depth(6) + impact(6)
#
# Our mapping:
#   raw(40) + stateless_price(5) + stateless_volume(8) + stateless_OI(6)
#           + stateless_depth(6) + stateless_impact(6)     → covers price(5/6), volume(8), OI(6), depth(6), impact(6)
#           + stateful_price_vol(1)                         → covers price(6th: volatility)
#           + stateful_OFI(6)                               → covers OFI(6)
#
# To match the EXACT batch order we rearrange into:
#   raw(40), price(5)+vol(1), volume(8), OI(6), OFI(6), depth(6), impact(6)

RAW_FEATURE_NAMES: list[str] = (
    [f'ask_price_{i}' for i in range(1, 11)]
    + [f'ask_volume_{i}' for i in range(1, 11)]
    + [f'bid_price_{i}' for i in range(1, 11)]
    + [f'bid_volume_{i}' for i in range(1, 11)]
)

# Final 78-feature name list matching batch pipeline
FEATURE_NAMES: list[str] = (
    RAW_FEATURE_NAMES                     # 40
    + [                                   # price (6)
        'mid_price', 'weighted_mid_price', 'spread_absolute',
        'spread_relative', 'log_mid_price', 'mid_price_volatility',
    ]
    + [                                   # volume (8)
        'bid_ask_volume_ratio_1', 'bid_ask_volume_ratio_2',
        'bid_ask_volume_ratio_3', 'bid_ask_volume_ratio_4',
        'bid_ask_volume_ratio_5',
        'cumulative_bid_volume', 'cumulative_ask_volume',
        'volume_imbalance_total',
    ]
    + [                                   # OI (6)
        'oi_level_1', 'oi_level_2', 'oi_level_3',
        'oi_total', 'oi_weighted', 'oi_asymmetry',
    ]
    + [                                   # OFI (6)
        'ofi_bid', 'ofi_ask', 'ofi_net',
        'ofi_ratio', 'ofi_cumulative', 'ofi_volatility',
    ]
    + [                                   # depth (6)
        'depth_imbalance', 'depth_ratio', 'effective_spread',
        'queue_position_proxy', 'depth_weighted_mid_price',
        'liquidity_concentration',
    ]
    + [                                   # impact (6)
        'market_order_impact_buy', 'market_order_impact_sell',
        'impact_asymmetry', 'resilience_proxy',
        'adverse_selection_risk', 'execution_cost_estimate',
    ]
)

assert len(FEATURE_NAMES) == 78


class FeatureEngine:
    """Real-time feature engine producing 78-feature vectors."""

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        profiler: Optional[Profiler] = None,
    ):
        self.state = StateManager(window_size)
        self.stateful = StatefulFeatureEngine(window_size)
        self.profiler = profiler or Profiler(enabled=False)

        # Pre-allocate scratch arrays
        self._stateless_buf = np.empty(31, dtype=np.float64)
        self._stateful_buf = np.empty(7, dtype=np.float64)
        self._output = np.empty(78, dtype=np.float64)

    def process_event(self, snapshot: LOBSnapshot) -> np.ndarray:
        """
        Process one LOB event and return 78-feature vector.

        The output matches the order of the batch
        FeatureEngineeringPipeline.get_feature_names().
        """
        with self.profiler.measure('state_update'):
            prev = self.state.update(snapshot)

        with self.profiler.measure('stateless_features'):
            sl = compute_stateless_features(snapshot, out=self._stateless_buf)

        with self.profiler.measure('stateful_features'):
            sf = self.stateful.update(snapshot, prev, out=self._stateful_buf)

        with self.profiler.measure('assemble'):
            out = self._output
            # raw (40)
            out[0:40] = snapshot.to_raw_features()
            # price (6): 5 stateless + 1 stateful (volatility)
            out[40:45] = sl[0:5]       # mid_price … log_mid_price
            out[45] = sf[0]            # mid_price_volatility
            # volume (8)
            out[46:54] = sl[5:13]
            # OI (6)
            out[54:60] = sl[13:19]
            # OFI (6)
            out[60:66] = sf[1:7]
            # depth (6)
            out[66:72] = sl[19:25]
            # impact (6)
            out[72:78] = sl[25:31]

        return out.copy()  # return a copy so caller can store it safely

    def get_feature_names(self) -> List[str]:
        return FEATURE_NAMES

    def reset(self) -> None:
        self.state.reset()
        self.stateful.reset()
