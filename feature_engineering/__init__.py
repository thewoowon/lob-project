"""
Feature Engineering Module for LOB Mid-Price Prediction

This module implements 78 features:
- 40 raw LOB features (ask/bid prices and volumes at 10 levels)
- 38 engineered features across 6 categories:
  * Price features (6)
  * Volume features (8)
  * Order Imbalance (6)
  * Order Flow Imbalance (6)
  * Depth features (6)
  * Price Impact features (6)

Based on PAPER_DRAFT.md specifications.
"""

__version__ = "1.0.0"
