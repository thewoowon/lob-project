"""
Model Training Module

Contains scripts for:
- Data loading from S3 JSONL files
- Label generation (k=100 horizon)
- Train/validation/test split
- CatBoost training
- Multi-seed validation
- Statistical testing
"""

__version__ = "1.0.0"
