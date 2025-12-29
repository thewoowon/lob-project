"""
Raw LOB Feature Extraction

Extracts 40 raw features from LOB snapshot:
- ask_price_{1-10}: 10 features
- ask_volume_{1-10}: 10 features
- bid_price_{1-10}: 10 features
- bid_volume_{1-10}: 10 features
"""

import numpy as np
from typing import Dict


def extract_raw_features(lob_snapshot: Dict) -> np.ndarray:
    """
    Extract 40 raw LOB features from snapshot.

    Args:
        lob_snapshot: Dict with keys:
            - timestamp
            - stock_code
            - ask_price_{1-10}, ask_volume_{1-10}
            - bid_price_{1-10}, bid_volume_{1-10}

    Returns:
        raw_features: np.ndarray of shape (40,)
            [ask_price_1, ..., ask_price_10,
             ask_volume_1, ..., ask_volume_10,
             bid_price_1, ..., bid_price_10,
             bid_volume_1, ..., bid_volume_10]
    """
    features = []

    # Ask prices (10)
    for i in range(1, 11):
        features.append(lob_snapshot[f'ask_price_{i}'])

    # Ask volumes (10)
    for i in range(1, 11):
        features.append(lob_snapshot[f'ask_volume_{i}'])

    # Bid prices (10)
    for i in range(1, 11):
        features.append(lob_snapshot[f'bid_price_{i}'])

    # Bid volumes (10)
    for i in range(1, 11):
        features.append(lob_snapshot[f'bid_volume_{i}'])

    return np.array(features, dtype=np.float64)


def get_raw_feature_names() -> list:
    """
    Get names of 40 raw features.

    Returns:
        List of 40 feature names
    """
    names = []

    # Ask prices
    names.extend([f'ask_price_{i}' for i in range(1, 11)])

    # Ask volumes
    names.extend([f'ask_volume_{i}' for i in range(1, 11)])

    # Bid prices
    names.extend([f'bid_price_{i}' for i in range(1, 11)])

    # Bid volumes
    names.extend([f'bid_volume_{i}' for i in range(1, 11)])

    return names
