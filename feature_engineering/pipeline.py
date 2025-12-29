"""
Feature Engineering Pipeline

Complete pipeline that combines:
- Raw features (40)
- Engineered features (38)
Total: 78 features

Based on PAPER_DRAFT.md specifications.
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional

from .raw_features import extract_raw_features, get_raw_feature_names
from .price_features import compute_price_features, get_price_feature_names
from .volume_features import compute_volume_features, get_volume_feature_names
from .order_imbalance import compute_order_imbalance_features, get_order_imbalance_feature_names
from .order_flow_imbalance import compute_order_flow_imbalance_features, get_order_flow_imbalance_feature_names
from .depth_features import compute_depth_features, get_depth_feature_names
from .price_impact import compute_price_impact_features, get_price_impact_feature_names


class FeatureEngineeringPipeline:
    """
    Complete feature engineering pipeline for LOB data.

    Processes raw LOB snapshots into 78 features:
    - 40 raw features
    - 38 engineered features (6 categories)

    Maintains history buffer for temporal features (OFI, volatility, etc.)
    """

    def __init__(self, buffer_size: int = 5):
        """
        Initialize feature engineering pipeline.

        Args:
            buffer_size: Number of past events to buffer for temporal features
                        (default: 5, used for OFI cumulative, volatility, etc.)
        """
        self.buffer_size = buffer_size
        self.history_buffer = deque(maxlen=buffer_size)

    def process_snapshot(self, current_snapshot: Dict) -> np.ndarray:
        """
        Process a single LOB snapshot into 78 features.

        Args:
            current_snapshot: Dict with keys:
                - timestamp: str (ISO format)
                - stock_code: str
                - ask_price_{1-10}: float
                - ask_volume_{1-10}: float
                - bid_price_{1-10}: float
                - bid_volume_{1-10}: float

        Returns:
            feature_vector: np.ndarray of shape (78,)
                [raw_features (40), engineered_features (38)]

        Example:
            >>> pipeline = FeatureEngineeringPipeline()
            >>> snapshot = {
            ...     'timestamp': '2025-12-29T09:00:15',
            ...     'stock_code': '005930',
            ...     'ask_price_1': 105100.0, 'ask_volume_1': 64675.0,
            ...     ...
            ... }
            >>> features = pipeline.process_snapshot(snapshot)
            >>> features.shape
            (78,)
        """
        # 1. Extract raw features (40)
        raw_features = extract_raw_features(current_snapshot)

        # 2. Get previous snapshot for temporal features
        if len(self.history_buffer) == 0:
            previous_snapshot = None
        else:
            previous_snapshot = self.history_buffer[-1]

        # 3. Compute engineered features (38)

        # Price features (6)
        price_feats = compute_price_features(
            current_snapshot,
            list(self.history_buffer)
        )

        # Volume features (8)
        volume_feats = compute_volume_features(current_snapshot)

        # Order Imbalance features (6)
        oi_feats = compute_order_imbalance_features(current_snapshot)

        # Order Flow Imbalance features (6)
        ofi_feats = compute_order_flow_imbalance_features(
            current_snapshot,
            previous_snapshot,
            list(self.history_buffer)
        )

        # Depth features (6)
        depth_feats = compute_depth_features(current_snapshot)

        # Price Impact features (6)
        impact_feats = compute_price_impact_features(current_snapshot)

        # 4. Concatenate all features
        feature_vector = np.concatenate([
            raw_features,      # 40
            price_feats,       # 6
            volume_feats,      # 8
            oi_feats,          # 6
            ofi_feats,         # 6
            depth_feats,       # 6
            impact_feats       # 6
        ])  # Total: 78

        # Verify shape
        assert feature_vector.shape == (78,), \
            f"Expected 78 features, got {feature_vector.shape[0]}"

        # 5. Update history buffer
        self.history_buffer.append(current_snapshot)

        return feature_vector

    def process_batch(self, snapshots: List[Dict]) -> np.ndarray:
        """
        Process a batch of LOB snapshots into feature matrix.

        Args:
            snapshots: List of LOB snapshots (chronologically ordered)

        Returns:
            feature_matrix: np.ndarray of shape (n_samples, 78)

        Example:
            >>> pipeline = FeatureEngineeringPipeline()
            >>> snapshots = [snapshot1, snapshot2, snapshot3, ...]
            >>> X = pipeline.process_batch(snapshots)
            >>> X.shape
            (1000, 78)
        """
        features = []

        for snapshot in snapshots:
            feature_vector = self.process_snapshot(snapshot)
            features.append(feature_vector)

        return np.array(features, dtype=np.float64)

    def reset(self):
        """
        Reset history buffer.

        Call this when starting to process a new stock or new time period.
        """
        self.history_buffer.clear()

    def get_feature_names(self) -> List[str]:
        """
        Get names of all 78 features in order.

        Returns:
            List of 78 feature names

        Example:
            >>> pipeline = FeatureEngineeringPipeline()
            >>> names = pipeline.get_feature_names()
            >>> len(names)
            78
            >>> names[:5]
            ['ask_price_1', 'ask_price_2', 'ask_price_3', 'ask_price_4', 'ask_price_5']
        """
        return (
            get_raw_feature_names() +                     # 40
            get_price_feature_names() +                   # 6
            get_volume_feature_names() +                  # 8
            get_order_imbalance_feature_names() +         # 6
            get_order_flow_imbalance_feature_names() +    # 6
            get_depth_feature_names() +                   # 6
            get_price_impact_feature_names()              # 6
        )  # Total: 78

    def get_feature_categories(self) -> Dict[str, List[str]]:
        """
        Get feature names grouped by category.

        Returns:
            Dict mapping category name to list of feature names

        Example:
            >>> pipeline = FeatureEngineeringPipeline()
            >>> categories = pipeline.get_feature_categories()
            >>> categories.keys()
            dict_keys(['raw', 'price', 'volume', 'order_imbalance',
                       'order_flow_imbalance', 'depth', 'price_impact'])
            >>> len(categories['raw'])
            40
            >>> len(categories['price'])
            6
        """
        return {
            'raw': get_raw_feature_names(),
            'price': get_price_feature_names(),
            'volume': get_volume_feature_names(),
            'order_imbalance': get_order_imbalance_feature_names(),
            'order_flow_imbalance': get_order_flow_imbalance_feature_names(),
            'depth': get_depth_feature_names(),
            'price_impact': get_price_impact_feature_names()
        }

    def __repr__(self) -> str:
        return (
            f"FeatureEngineeringPipeline("
            f"buffer_size={self.buffer_size}, "
            f"history_length={len(self.history_buffer)}, "
            f"features=78)"
        )
