"""
Order Imbalance (OI) Features (6 features)

Based on PAPER_DRAFT.md Section 3.3.3:

Theory: Order imbalance measures supply-demand asymmetry.
        Positive OI suggests buying pressure (price likely to increase).

1. OI level 1: (Vbid1 - Vask1) / (Vbid1 + Vask1)
2. OI level 2
3. OI level 3
4. OI total (all levels)
5. Weighted OI
6. OI asymmetry (top vs deep)
"""

import numpy as np
from typing import Dict, List

EPSILON = 1e-10


def compute_order_imbalance_features(lob_snapshot: Dict) -> np.ndarray:
    """
    Compute 6 order imbalance features.

    Args:
        lob_snapshot: Current LOB snapshot with ask_volume_{1-10}, bid_volume_{1-10}

    Returns:
        oi_features: np.ndarray of shape (6,)
            [oi_level_1, oi_level_2, oi_level_3,
             oi_total, oi_weighted, oi_asymmetry]
    """
    # Extract volumes
    ask_volumes = [lob_snapshot[f'ask_volume_{i}'] for i in range(1, 11)]
    bid_volumes = [lob_snapshot[f'bid_volume_{i}'] for i in range(1, 11)]

    features = []

    # 1-3. OI at levels 1, 2, 3
    for i in range(3):
        oi_level = (bid_volumes[i] - ask_volumes[i]) / \
                   (bid_volumes[i] + ask_volumes[i] + EPSILON)
        features.append(oi_level)

    # 4. OI total (all levels)
    total_bid = sum(bid_volumes)
    total_ask = sum(ask_volumes)
    oi_total = (total_bid - total_ask) / (total_bid + total_ask + EPSILON)
    features.append(oi_total)

    # 5. Weighted OI (closer levels have higher weight)
    # Weight = 1/i for level i (level 1 has weight 1.0, level 2 has 0.5, etc.)
    weights = [1.0 / i for i in range(1, 11)]
    oi_per_level = [
        (bid_volumes[i] - ask_volumes[i]) / (bid_volumes[i] + ask_volumes[i] + EPSILON)
        for i in range(10)
    ]
    oi_weighted = sum(w * oi for w, oi in zip(weights, oi_per_level)) / sum(weights)
    features.append(oi_weighted)

    # 6. OI asymmetry (top levels 1-3 vs deep levels 4-10)
    bid_top = sum(bid_volumes[:3])
    ask_top = sum(ask_volumes[:3])
    oi_top = (bid_top - ask_top) / (bid_top + ask_top + EPSILON)

    bid_deep = sum(bid_volumes[3:])
    ask_deep = sum(ask_volumes[3:])
    oi_deep = (bid_deep - ask_deep) / (bid_deep + ask_deep + EPSILON)

    oi_asymmetry = oi_top - oi_deep
    features.append(oi_asymmetry)

    return np.array(features, dtype=np.float64)


def get_order_imbalance_feature_names() -> List[str]:
    """
    Get names of 6 OI features.

    Returns:
        List of 6 feature names
    """
    return [
        'oi_level_1',
        'oi_level_2',
        'oi_level_3',
        'oi_total',
        'oi_weighted',
        'oi_asymmetry'
    ]
