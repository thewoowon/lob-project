"""
Feature Engineering Pipeline - Usage Example

Shows how to use the FeatureEngineeringPipeline to process LOB snapshots.
"""

import json
from feature_engineering.pipeline import FeatureEngineeringPipeline


def example_single_snapshot():
    """Example: Process a single LOB snapshot."""
    print("=" * 70)
    print("Example 1: Process Single LOB Snapshot")
    print("=" * 70)

    # Create pipeline
    pipeline = FeatureEngineeringPipeline(buffer_size=5)

    # Sample LOB snapshot (same format as S3 JSONL data)
    lob_snapshot = {
        'timestamp': '2025-12-29T09:00:15',
        'stock_code': '005930',
        # Ask prices (10 levels)
        'ask_price_1': 105100.0, 'ask_price_2': 105200.0, 'ask_price_3': 105300.0,
        'ask_price_4': 105400.0, 'ask_price_5': 105500.0, 'ask_price_6': 105600.0,
        'ask_price_7': 105700.0, 'ask_price_8': 105800.0, 'ask_price_9': 105900.0,
        'ask_price_10': 106000.0,
        # Ask volumes (10 levels)
        'ask_volume_1': 64675.0, 'ask_volume_2': 48203.0, 'ask_volume_3': 32145.0,
        'ask_volume_4': 21034.0, 'ask_volume_5': 15678.0, 'ask_volume_6': 12045.0,
        'ask_volume_7': 9834.0, 'ask_volume_8': 7621.0, 'ask_volume_9': 5432.0,
        'ask_volume_10': 3210.0,
        # Bid prices (10 levels)
        'bid_price_1': 105000.0, 'bid_price_2': 104900.0, 'bid_price_3': 104800.0,
        'bid_price_4': 104700.0, 'bid_price_5': 104600.0, 'bid_price_6': 104500.0,
        'bid_price_7': 104400.0, 'bid_price_8': 104300.0, 'bid_price_9': 104200.0,
        'bid_price_10': 104100.0,
        # Bid volumes (10 levels)
        'bid_volume_1': 68489.0, 'bid_volume_2': 52301.0, 'bid_volume_3': 35142.0,
        'bid_volume_4': 23456.0, 'bid_volume_5': 17890.0, 'bid_volume_6': 13456.0,
        'bid_volume_7': 10234.0, 'bid_volume_8': 8123.0, 'bid_volume_9': 6012.0,
        'bid_volume_10': 4321.0
    }

    # Process snapshot into 78 features
    features = pipeline.process_snapshot(lob_snapshot)

    print(f"Input: LOB snapshot for {lob_snapshot['stock_code']} at {lob_snapshot['timestamp']}")
    print(f"Output: {features.shape[0]} features\n")

    # Display sample features by category
    feature_names = pipeline.get_feature_names()
    categories = pipeline.get_feature_categories()

    print("Sample features by category:")
    print()

    # Raw features (show first 5)
    print(f"  Raw features (40 total, showing first 5):")
    for i in range(5):
        print(f"    {feature_names[i]:20s} = {features[i]:12.2f}")
    print()

    # Price features (show all 6)
    offset = 40
    print(f"  Price features (6 total):")
    for i, name in enumerate(categories['price']):
        print(f"    {name:30s} = {features[offset + i]:12.4f}")
    print()

    # Volume features (show all 8)
    offset = 40 + 6
    print(f"  Volume features (8 total):")
    for i, name in enumerate(categories['volume']):
        print(f"    {name:30s} = {features[offset + i]:12.2f}")
    print()

    # OI features (show all 6)
    offset = 40 + 6 + 8
    print(f"  Order Imbalance features (6 total):")
    for i, name in enumerate(categories['order_imbalance']):
        print(f"    {name:30s} = {features[offset + i]:12.4f}")
    print()

    # OFI features (show all 6)
    offset = 40 + 6 + 8 + 6
    print(f"  Order Flow Imbalance features (6 total):")
    for i, name in enumerate(categories['order_flow_imbalance']):
        print(f"    {name:30s} = {features[offset + i]:12.4f}")
    print()

    # Depth features (show all 6)
    offset = 40 + 6 + 8 + 6 + 6
    print(f"  Depth features (6 total):")
    for i, name in enumerate(categories['depth']):
        print(f"    {name:30s} = {features[offset + i]:12.2f}")
    print()

    # Price Impact features (show all 6)
    offset = 40 + 6 + 8 + 6 + 6 + 6
    print(f"  Price Impact features (6 total):")
    for i, name in enumerate(categories['price_impact']):
        print(f"    {name:30s} = {features[offset + i]:12.4f}")
    print()

    print(f"✅ Successfully processed 1 snapshot into {features.shape[0]} features!")
    print()


def example_batch_processing():
    """Example: Process multiple LOB snapshots in batch."""
    print("=" * 70)
    print("Example 2: Batch Processing Multiple Snapshots")
    print("=" * 70)

    # Create pipeline
    pipeline = FeatureEngineeringPipeline(buffer_size=5)

    # Simulate 10 LOB snapshots (in reality, these would come from S3 JSONL files)
    snapshots = []
    for i in range(10):
        snapshot = {
            'timestamp': f'2025-12-29T09:00:{15+i:02d}',
            'stock_code': '005930',
            # Simplified: all features
            **{f'ask_price_{j}': 105000.0 + (j-1) * 100 + i * 10 for j in range(1, 11)},
            **{f'ask_volume_{j}': 50000.0 / j for j in range(1, 11)},
            **{f'bid_price_{j}': 104900.0 - (j-1) * 100 + i * 10 for j in range(1, 11)},
            **{f'bid_volume_{j}': 50000.0 / j for j in range(1, 11)},
        }
        snapshots.append(snapshot)

    # Process all snapshots
    feature_matrix = pipeline.process_batch(snapshots)

    print(f"Input: {len(snapshots)} LOB snapshots")
    print(f"Output: Feature matrix of shape {feature_matrix.shape}")
    print(f"  - {feature_matrix.shape[0]} samples")
    print(f"  - {feature_matrix.shape[1]} features per sample")
    print()

    # Show statistics
    print("Feature statistics:")
    print(f"  Mean: {feature_matrix.mean():.2f}")
    print(f"  Std:  {feature_matrix.std():.2f}")
    print(f"  Min:  {feature_matrix.min():.2f}")
    print(f"  Max:  {feature_matrix.max():.2f}")
    print()

    print("✅ Successfully processed 10 snapshots!")
    print()


def example_feature_names():
    """Example: Get feature names and categories."""
    print("=" * 70)
    print("Example 3: Feature Names and Categories")
    print("=" * 70)

    pipeline = FeatureEngineeringPipeline()

    # Get all feature names
    feature_names = pipeline.get_feature_names()
    print(f"Total features: {len(feature_names)}")
    print()

    # Get feature categories
    categories = pipeline.get_feature_categories()
    print("Features by category:")
    for category_name, names in categories.items():
        print(f"  {category_name:25s}: {len(names):2d} features")
    print()

    # Verify total
    total = sum(len(names) for names in categories.values())
    print(f"Total: {total} features")
    print(f"Expected: 78 features (40 raw + 38 engineered)")
    assert total == 78, f"Feature count mismatch: {total} != 78"
    print("✅ Feature count verified!")
    print()


if __name__ == "__main__":
    example_single_snapshot()
    print()
    example_batch_processing()
    print()
    example_feature_names()

    print("=" * 70)
    print("✅ All examples completed successfully!")
    print("=" * 70)
