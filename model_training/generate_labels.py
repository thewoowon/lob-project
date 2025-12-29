"""
Label Generation for LOB Mid-Price Prediction

Based on PAPER_DRAFT.md Section 3.1:
- Prediction horizon: k=100 events ahead (~5-10 minutes)
- 3-class classification:
  * Class 0: Price decrease (down)
  * Class 1: Price stationary (no change)
  * Class 2: Price increase (up)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


def generate_labels(
    snapshots: List[dict],
    k: int = 100,
    threshold_pct: float = 0.0001
) -> np.ndarray:
    """
    Generate labels for mid-price movement prediction.

    Args:
        snapshots: List of LOB snapshots (must be chronologically sorted!)
        k: Prediction horizon (number of events ahead)
        threshold_pct: Threshold for "stationary" class (as fraction of mid-price)
                      Default: 0.01% (0.0001)

    Returns:
        labels: np.ndarray of shape (n_samples,) with values {0, 1, 2, np.nan}
                - 0: Price decrease (down)
                - 1: Price stationary (no change)
                - 2: Price increase (up)
                - np.nan: No label (last k samples)

    Example:
        >>> snapshots = [...]  # List of LOB snapshots
        >>> labels = generate_labels(snapshots, k=100)
        >>> labels.shape
        (1000,)
        >>> np.bincount(labels[~np.isnan(labels)].astype(int))
        array([410, 210, 380])  # Down, Stay, Up counts
    """
    n_samples = len(snapshots)
    labels = np.full(n_samples, np.nan, dtype=np.float64)

    for i in range(n_samples - k):
        # Current mid-price
        current_ask = snapshots[i]['ask_price_1']
        current_bid = snapshots[i]['bid_price_1']
        current_mid = (current_ask + current_bid) / 2.0

        # Future mid-price (k events ahead)
        future_ask = snapshots[i + k]['ask_price_1']
        future_bid = snapshots[i + k]['bid_price_1']
        future_mid = (future_ask + future_bid) / 2.0

        # Threshold for "stationary"
        threshold = threshold_pct * current_mid

        # Classify
        if future_mid < current_mid - threshold:
            labels[i] = 0  # Down
        elif future_mid > current_mid + threshold:
            labels[i] = 2  # Up
        else:
            labels[i] = 1  # Stationary

    # Last k samples have no labels (cannot look k events ahead)
    # labels[n_samples - k:] = np.nan  # Already set above

    return labels


def generate_labels_from_dataframe(
    df: pd.DataFrame,
    k: int = 100,
    threshold_pct: float = 0.0001,
    group_by_stock: bool = True
) -> pd.Series:
    """
    Generate labels from pandas DataFrame.

    Args:
        df: DataFrame with columns 'ask_price_1', 'bid_price_1', 'stock_code'
        k: Prediction horizon
        threshold_pct: Threshold for "stationary" class
        group_by_stock: If True, generate labels separately per stock
                       (prevents cross-stock label leakage)

    Returns:
        pd.Series of labels aligned with df index
    """
    if group_by_stock and 'stock_code' in df.columns:
        # Generate labels per stock separately
        labels_series = pd.Series(index=df.index, dtype=np.float64)

        for stock_code, group in df.groupby('stock_code'):
            # Convert group to list of dicts
            snapshots = group.to_dict('records')

            # Generate labels
            labels = generate_labels(snapshots, k=k, threshold_pct=threshold_pct)

            # Assign to series
            labels_series.loc[group.index] = labels

    else:
        # Generate labels for entire dataset
        snapshots = df.to_dict('records')
        labels = generate_labels(snapshots, k=k, threshold_pct=threshold_pct)
        labels_series = pd.Series(labels, index=df.index)

    return labels_series


def analyze_label_distribution(labels: np.ndarray) -> dict:
    """
    Analyze label distribution.

    Args:
        labels: Array of labels (may contain np.nan)

    Returns:
        Dict with statistics
    """
    valid_labels = labels[~np.isnan(labels)]

    if len(valid_labels) == 0:
        return {
            'total_samples': len(labels),
            'valid_samples': 0,
            'down_count': 0,
            'stay_count': 0,
            'up_count': 0,
            'down_pct': 0.0,
            'stay_pct': 0.0,
            'up_pct': 0.0
        }

    counts = np.bincount(valid_labels.astype(int), minlength=3)

    return {
        'total_samples': len(labels),
        'valid_samples': len(valid_labels),
        'nan_samples': len(labels) - len(valid_labels),
        'down_count': int(counts[0]),
        'stay_count': int(counts[1]),
        'up_count': int(counts[2]),
        'down_pct': 100.0 * counts[0] / len(valid_labels),
        'stay_pct': 100.0 * counts[1] / len(valid_labels),
        'up_pct': 100.0 * counts[2] / len(valid_labels)
    }


def print_label_distribution(labels: np.ndarray, title: str = "Label Distribution"):
    """Print label distribution statistics."""
    stats = analyze_label_distribution(labels)

    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"Total samples:    {stats['total_samples']}")
    print(f"Valid samples:    {stats['valid_samples']}")
    print(f"NaN samples:      {stats['nan_samples']}")
    print()
    print("Class distribution:")
    print(f"  Down (0):       {stats['down_count']:6d} ({stats['down_pct']:5.2f}%)")
    print(f"  Stay (1):       {stats['stay_count']:6d} ({stats['stay_pct']:5.2f}%)")
    print(f"  Up (2):         {stats['up_count']:6d} ({stats['up_pct']:5.2f}%)")
    print()

    # Check if distribution is reasonable
    if stats['valid_samples'] > 0:
        # Down and Up should be roughly symmetric
        down_up_diff = abs(stats['down_pct'] - stats['up_pct'])
        if down_up_diff > 10:
            print(f"⚠️  Warning: Down/Up imbalance is large ({down_up_diff:.2f}% difference)")
        else:
            print(f"✅ Down/Up balance looks good ({down_up_diff:.2f}% difference)")

        # Stay should be minority (typically 15-25%)
        if stats['stay_pct'] < 10 or stats['stay_pct'] > 30:
            print(f"⚠️  Warning: Stay class proportion may be unusual ({stats['stay_pct']:.2f}%)")
        else:
            print(f"✅ Stay class proportion looks reasonable ({stats['stay_pct']:.2f}%)")
    print()


# Example usage
if __name__ == "__main__":
    from data_loader import load_jsonl_file

    print("=" * 70)
    print("Testing label generation with sample data")
    print("=" * 70)

    # Load sample data
    snapshots = load_jsonl_file("data/sample/sample_005930.jsonl")
    print(f"\nLoaded {len(snapshots)} snapshots")

    # Generate labels with k=10 (small for testing)
    labels = generate_labels(snapshots, k=10, threshold_pct=0.0001)

    # Print distribution
    print_label_distribution(labels, title="Label Distribution (k=10, threshold=0.01%)")

    # Test with different thresholds
    for threshold in [0.00001, 0.0001, 0.001]:
        labels = generate_labels(snapshots, k=10, threshold_pct=threshold)
        print_label_distribution(
            labels,
            title=f"Label Distribution (k=10, threshold={threshold*100:.3f}%)"
        )

    print("✅ Label generation test completed!")
