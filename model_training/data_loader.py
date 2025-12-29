"""
Data Loader for LOB JSONL Files

Loads LOB snapshots from S3 JSONL files and converts to features using
FeatureEngineeringPipeline.
"""

import json
import glob
import os
from typing import List, Dict, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

from feature_engineering.pipeline import FeatureEngineeringPipeline


def load_jsonl_file(filepath: str) -> List[Dict]:
    """
    Load LOB snapshots from a single JSONL file.

    Args:
        filepath: Path to JSONL file

    Returns:
        List of LOB snapshot dictionaries
    """
    snapshots = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                snapshot = json.loads(line.strip())
                snapshots.append(snapshot)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line in {filepath}: {e}")
                continue

    return snapshots


def load_stock_data(
    stock_code: str,
    data_dir: str = "data/sample",
    date_filter: str = None
) -> List[Dict]:
    """
    Load all LOB snapshots for a specific stock.

    Args:
        stock_code: Stock code (e.g., "005930")
        data_dir: Directory containing JSONL files
        date_filter: Optional date filter (e.g., "20251215")

    Returns:
        List of LOB snapshots sorted by timestamp
    """
    # Find all JSONL files for this stock
    if date_filter:
        pattern = f"{data_dir}/*{stock_code}*{date_filter}*.jsonl"
    else:
        pattern = f"{data_dir}/*{stock_code}*.jsonl"

    files = glob.glob(pattern)

    if not files:
        print(f"Warning: No files found for stock {stock_code} with pattern {pattern}")
        return []

    print(f"Loading {len(files)} file(s) for stock {stock_code}...")

    all_snapshots = []
    for filepath in tqdm(files, desc=f"Loading {stock_code}"):
        snapshots = load_jsonl_file(filepath)
        all_snapshots.extend(snapshots)

    # Sort by timestamp
    all_snapshots.sort(key=lambda x: x['timestamp'])

    print(f"Loaded {len(all_snapshots)} snapshots for {stock_code}")

    return all_snapshots


def load_s3_data_from_directory(
    s3_data_dir: str,
    stock_codes: List[str] = None,
    max_files_per_stock: int = None
) -> Dict[str, List[Dict]]:
    """
    Load LOB data from S3 directory structure.

    Expected directory structure:
        s3_data_dir/
        ├── 005930/
        │   ├── 20251215/
        │   │   ├── lob_033852.jsonl
        │   │   └── ...
        │   └── 20251216/
        │       └── ...
        └── 000660/
            └── ...

    Args:
        s3_data_dir: Root directory of S3 data
        stock_codes: List of stock codes to load (None = all)
        max_files_per_stock: Maximum files per stock (None = all)

    Returns:
        Dict mapping stock_code to list of LOB snapshots
    """
    data_by_stock = {}

    # Get all stock directories
    if stock_codes is None:
        stock_dirs = [d for d in os.listdir(s3_data_dir)
                     if os.path.isdir(os.path.join(s3_data_dir, d))]
    else:
        stock_dirs = stock_codes

    for stock_code in stock_dirs:
        stock_dir = os.path.join(s3_data_dir, stock_code)

        if not os.path.exists(stock_dir):
            print(f"Warning: Directory not found for {stock_code}")
            continue

        # Get all JSONL files for this stock
        all_files = []
        for root, dirs, files in os.walk(stock_dir):
            for file in files:
                if file.endswith('.jsonl'):
                    all_files.append(os.path.join(root, file))

        # Limit files if specified
        if max_files_per_stock:
            all_files = all_files[:max_files_per_stock]

        print(f"\nLoading stock {stock_code}: {len(all_files)} file(s)")

        # Load all files
        all_snapshots = []
        for filepath in tqdm(all_files, desc=stock_code):
            snapshots = load_jsonl_file(filepath)
            all_snapshots.extend(snapshots)

        # Sort by timestamp
        all_snapshots.sort(key=lambda x: x['timestamp'])

        data_by_stock[stock_code] = all_snapshots
        print(f"  Total snapshots: {len(all_snapshots)}")

    return data_by_stock


def snapshots_to_features(
    snapshots: List[Dict],
    pipeline: FeatureEngineeringPipeline = None
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Convert LOB snapshots to feature matrix.

    Args:
        snapshots: List of LOB snapshots
        pipeline: FeatureEngineeringPipeline instance (creates new if None)

    Returns:
        Tuple of:
        - features: np.ndarray of shape (n_samples, 78)
        - timestamps: List of timestamp strings
        - stock_codes: List of stock codes
    """
    if pipeline is None:
        pipeline = FeatureEngineeringPipeline(buffer_size=5)
    else:
        pipeline.reset()  # Reset history buffer

    features = []
    timestamps = []
    stock_codes = []

    for snapshot in tqdm(snapshots, desc="Converting to features"):
        feature_vector = pipeline.process_snapshot(snapshot)
        features.append(feature_vector)
        timestamps.append(snapshot['timestamp'])
        stock_codes.append(snapshot['stock_code'])

    features = np.array(features, dtype=np.float64)

    return features, timestamps, stock_codes


def create_dataframe(
    features: np.ndarray,
    timestamps: List[str],
    stock_codes: List[str],
    feature_names: List[str] = None
) -> pd.DataFrame:
    """
    Create pandas DataFrame from features.

    Args:
        features: Feature matrix (n_samples, 78)
        timestamps: List of timestamps
        stock_codes: List of stock codes
        feature_names: List of feature names (auto-generated if None)

    Returns:
        pandas DataFrame
    """
    if feature_names is None:
        pipeline = FeatureEngineeringPipeline()
        feature_names = pipeline.get_feature_names()

    df = pd.DataFrame(features, columns=feature_names)
    df['timestamp'] = timestamps
    df['stock_code'] = stock_codes

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df


# Example usage
if __name__ == "__main__":
    # Test loading sample data
    print("=" * 70)
    print("Testing data loader with sample data")
    print("=" * 70)

    # Load sample JSONL file
    snapshots = load_jsonl_file("data/sample/sample_005930.jsonl")
    print(f"\nLoaded {len(snapshots)} snapshots from sample file")

    # Convert to features
    pipeline = FeatureEngineeringPipeline(buffer_size=5)
    features, timestamps, stock_codes = snapshots_to_features(snapshots, pipeline)

    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Timestamps: {len(timestamps)}")
    print(f"Stock codes: {len(set(stock_codes))} unique ({set(stock_codes)})")

    # Create DataFrame
    df = create_dataframe(features, timestamps, stock_codes)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())

    print("\n✅ Data loader test completed successfully!")
