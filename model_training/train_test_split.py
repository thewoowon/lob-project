"""
Temporal Train/Validation/Test Split

Based on PAPER_DRAFT.md Section 3.1:
- Training: First 7 days per stock (or 70% of data)
- Validation: Day 8 (or 15% of data)
- Test: Days 9-10 (or 15% of data)

For our data (2 weeks so far):
- Training: First 10 days (70%)
- Validation: Days 11-12 (15%)
- Test: Days 13-14 (15%)
"""

import numpy as np
import pandas as pd
from typing import Tuple
from datetime import datetime


def temporal_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    by_stock: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally (no shuffle!).

    Args:
        df: DataFrame with 'timestamp' and 'stock_code' columns
        train_ratio: Fraction for training (default 0.7)
        val_ratio: Fraction for validation (default 0.15)
        test_ratio: Fraction for testing (default 0.15)
        by_stock: If True, split each stock separately (recommended)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # Ensure temporal ordering
    df = df.sort_values('timestamp').reset_index(drop=True)

    if by_stock and 'stock_code' in df.columns:
        # Split each stock separately
        train_dfs = []
        val_dfs = []
        test_dfs = []

        for stock_code, group in df.groupby('stock_code'):
            n = len(group)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            train_dfs.append(group.iloc[:train_end])
            val_dfs.append(group.iloc[train_end:val_end])
            test_dfs.append(group.iloc[val_end:])

        train_df = pd.concat(train_dfs, ignore_index=True).sort_values('timestamp')
        val_df = pd.concat(val_dfs, ignore_index=True).sort_values('timestamp')
        test_df = pd.concat(test_dfs, ignore_index=True).sort_values('timestamp')

    else:
        # Split entire dataset
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

    # Verify no temporal overlap
    assert train_df['timestamp'].max() <= val_df['timestamp'].min(), \
        "Train data leaks into validation!"
    assert val_df['timestamp'].max() <= test_df['timestamp'].min(), \
        "Validation data leaks into test!"

    print("=" * 70)
    print("Temporal Train/Val/Test Split")
    print("=" * 70)
    print(f"Train:      {len(train_df):6d} samples ({train_ratio*100:.1f}%)")
    print(f"  Time range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print()
    print(f"Validation: {len(val_df):6d} samples ({val_ratio*100:.1f}%)")
    print(f"  Time range: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
    print()
    print(f"Test:       {len(test_df):6d} samples ({test_ratio*100:.1f}%)")
    print(f"  Time range: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    print()

    if by_stock and 'stock_code' in df.columns:
        print("Stocks in each split:")
        print(f"  Train: {sorted(train_df['stock_code'].unique())}")
        print(f"  Val:   {sorted(val_df['stock_code'].unique())}")
        print(f"  Test:  {sorted(test_df['stock_code'].unique())}")
        print()

    return train_df, val_df, test_df


def verify_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
):
    """
    Verify no data leakage between splits.
    """
    print("=" * 70)
    print("Data Leakage Verification")
    print("=" * 70)

    # Check 1: Temporal ordering
    train_max = train_df['timestamp'].max()
    val_min = val_df['timestamp'].min()
    val_max = val_df['timestamp'].max()
    test_min = test_df['timestamp'].min()

    print("Temporal ordering:")
    print(f"  Train ends:   {train_max}")
    print(f"  Val starts:   {val_min}")
    print(f"  Val ends:     {val_max}")
    print(f"  Test starts:  {test_min}")
    print()

    if train_max <= val_min:
        print("  ✅ Train → Val: No overlap")
    else:
        print("  ❌ Train → Val: OVERLAP DETECTED!")

    if val_max <= test_min:
        print("  ✅ Val → Test: No overlap")
    else:
        print("  ❌ Val → Test: OVERLAP DETECTED!")

    print()

    # Check 2: No duplicate indices
    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    test_idx = set(test_df.index)

    overlap_train_val = train_idx & val_idx
    overlap_val_test = val_idx & test_idx
    overlap_train_test = train_idx & test_idx

    print("Index overlap:")
    print(f"  Train ∩ Val:  {len(overlap_train_val)} samples")
    print(f"  Val ∩ Test:   {len(overlap_val_test)} samples")
    print(f"  Train ∩ Test: {len(overlap_train_test)} samples")

    if len(overlap_train_val) == 0 and len(overlap_val_test) == 0 and len(overlap_train_test) == 0:
        print("  ✅ No index overlap")
    else:
        print("  ❌ Index overlap detected!")

    print()
    print("=" * 70)


# Example usage
if __name__ == "__main__":
    from data_loader import load_jsonl_file, snapshots_to_features, create_dataframe
    from generate_labels import generate_labels_from_dataframe

    print("Testing train/val/test split with sample data\n")

    # Load sample data
    snapshots = load_jsonl_file("data/combined_005930_20251215.jsonl")
    print(f"Loaded {len(snapshots)} snapshots\n")

    # Convert to features
    from feature_engineering.pipeline import FeatureEngineeringPipeline
    pipeline = FeatureEngineeringPipeline()
    features, timestamps, stock_codes = snapshots_to_features(snapshots, pipeline)

    # Create DataFrame
    df = create_dataframe(features, timestamps, stock_codes)

    # Generate labels
    df['label'] = generate_labels_from_dataframe(df, k=100, threshold_pct=0.0001)

    # Remove NaN labels
    df_clean = df.dropna(subset=['label']).reset_index(drop=True)
    print(f"Clean samples: {len(df_clean)}\n")

    # Split
    train_df, val_df, test_df = temporal_train_val_test_split(
        df_clean,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        by_stock=True
    )

    # Verify
    verify_no_leakage(train_df, val_df, test_df)

    print("✅ Train/Val/Test split test completed!")
