"""
Volume Features (8 features)

Based on PAPER_DRAFT.md Section 3.3.2:
1-5. Bid/Ask volume ratios (levels 1-5)
6. Cumulative bid volume (sum of levels 1-10)
7. Cumulative ask volume (sum of levels 1-10)
8. Volume imbalance (total)
"""

import numpy as np
from typing import Dict, List

EPSILON = 1e-10


def compute_volume_features(lob_snapshot: Dict) -> np.ndarray:
    """
    Compute 8 volume-based features.

    Args:
        lob_snapshot: Current LOB snapshot with ask_volume_{1-10}, bid_volume_{1-10}

    Returns:
        volume_features: np.ndarray of shape (8,)
            [bid_ask_volume_ratio_1, ..., bid_ask_volume_ratio_5,
             cumulative_bid_volume, cumulative_ask_volume,
             volume_imbalance_total]
    """
    # Extract volumes
    ask_volumes = [lob_snapshot[f'ask_volume_{i}'] for i in range(1, 11)]
    bid_volumes = [lob_snapshot[f'bid_volume_{i}'] for i in range(1, 11)]

    features = []

    # 1-5. Bid/Ask volume ratios (levels 1-5)
    for i in range(5):
        ratio = bid_volumes[i] / (ask_volumes[i] + EPSILON)
        features.append(ratio)

    # 6. Cumulative bid volume (all 10 levels)
    cumulative_bid_volume = sum(bid_volumes)
    features.append(cumulative_bid_volume)

    # 7. Cumulative ask volume (all 10 levels)
    cumulative_ask_volume = sum(ask_volumes)
    features.append(cumulative_ask_volume)

    # 8. Volume imbalance (total)
    volume_imbalance_total = (cumulative_bid_volume - cumulative_ask_volume) / \
                             (cumulative_bid_volume + cumulative_ask_volume + EPSILON)
    features.append(volume_imbalance_total)

    return np.array(features, dtype=np.float64)


def get_volume_feature_names() -> List[str]:
    """
    Get names of 8 volume features.

    Returns:
        List of 8 feature names
    """
    names = []

    # Bid/ask volume ratios (levels 1-5)
    names.extend([f'bid_ask_volume_ratio_{i}' for i in range(1, 6)])

    # Cumulative volumes
    names.extend([
        'cumulative_bid_volume',
        'cumulative_ask_volume',
        'volume_imbalance_total'
    ])

    return names
