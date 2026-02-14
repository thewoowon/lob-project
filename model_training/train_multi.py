"""
Multi-Stock/Multi-Date CatBoost Training Pipeline

Extended training pipeline that:
1. Loads data from multiple stocks and dates (S3 directory structure)
2. Handles class imbalance (auto_class_weights='Balanced')
3. Runs multi-seed validation
4. Computes statistical significance (mean +/- std, paired t-test)

Based on PAPER_DRAFT.md and TRAIN_TEST_SPLIT_RESULTS.md.

Usage:
    # Train on all downloaded data with multi-seed validation
    python model_training/train_multi.py --data-dir data --seeds 42 123 456 789 1011

    # Train on specific stocks
    python model_training/train_multi.py --data-dir data --stocks 005930 000660 035420

    # Quick test with one seed
    python model_training/train_multi.py --data-dir data --stocks 005930 --seeds 42
"""

import os
import sys
import argparse
import json
import time
import gc
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from scipy import stats
from datetime import datetime
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_training.data_loader import load_jsonl_file, snapshots_to_features, create_dataframe
from model_training.generate_labels import generate_labels_from_dataframe, print_label_distribution
from model_training.train_test_split import temporal_train_val_test_split, verify_no_leakage
from feature_engineering.pipeline import FeatureEngineeringPipeline


CATBOOST_PARAMS_BALANCED = {
    'iterations': 1000,
    'depth': 6,
    'learning_rate': 0.05,
    'loss_function': 'MultiClass',
    'classes_count': 3,
    'eval_metric': 'TotalF1',
    'verbose': 200,
    'early_stopping_rounds': 100,
    'task_type': 'CPU',
    'bootstrap_type': 'Bayesian',
    'class_weights': [3.0, 1.0, 3.0],
}

CATBOOST_PARAMS_UNBALANCED = {
    'iterations': 1000,
    'depth': 6,
    'learning_rate': 0.05,
    'loss_function': 'MultiClass',
    'classes_count': 3,
    'eval_metric': 'TotalF1',
    'verbose': 200,
    'early_stopping_rounds': 100,
    'task_type': 'CPU',
    'bootstrap_type': 'Bayesian',
}

CATBOOST_PARAMS = CATBOOST_PARAMS_UNBALANCED


def load_multi_stock_data(
    data_dir: str,
    stock_codes: Optional[List[str]] = None,
    max_dates_per_stock: Optional[int] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load data from multiple stocks and dates, convert to features.

    Args:
        data_dir: Root data directory (e.g., "data")
        stock_codes: List of stock codes to load (None = all available)
        max_dates_per_stock: Limit dates per stock (None = all)

    Returns:
        features, timestamps, stock_codes arrays
    """
    if stock_codes is None:
        stock_codes = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()
        ])

    print(f"Loading data for {len(stock_codes)} stocks: {stock_codes}")

    all_snapshots = []
    stock_stats = {}

    for stock_code in stock_codes:
        stock_dir = os.path.join(data_dir, stock_code)
        if not os.path.exists(stock_dir):
            print(f"  Warning: {stock_dir} not found, skipping")
            continue

        # Get date directories
        date_dirs = sorted([
            d for d in os.listdir(stock_dir)
            if os.path.isdir(os.path.join(stock_dir, d))
        ])

        if max_dates_per_stock:
            date_dirs = date_dirs[:max_dates_per_stock]

        stock_snapshots = []
        for date_dir in date_dirs:
            date_path = os.path.join(stock_dir, date_dir)
            jsonl_files = sorted([
                f for f in os.listdir(date_path)
                if f.endswith('.jsonl')
            ])

            for jsonl_file in jsonl_files:
                filepath = os.path.join(date_path, jsonl_file)
                snapshots = load_jsonl_file(filepath)
                stock_snapshots.extend(snapshots)

        # Sort by timestamp
        stock_snapshots.sort(key=lambda x: x['timestamp'])

        stock_stats[stock_code] = {
            'dates': len(date_dirs),
            'snapshots': len(stock_snapshots),
        }
        print(f"  {stock_code}: {len(date_dirs)} dates, {len(stock_snapshots)} snapshots")

        all_snapshots.extend(stock_snapshots)

    print(f"\nTotal: {len(all_snapshots)} snapshots from {len(stock_stats)} stocks")

    if len(all_snapshots) == 0:
        raise ValueError("No data loaded! Check data directory.")

    # Convert to features
    print("\nConverting to 78 features...")
    pipeline = FeatureEngineeringPipeline(buffer_size=5)

    # Process per-stock to reset pipeline buffer between stocks
    all_features = []
    all_timestamps = []
    all_stock_codes = []

    for stock_code in stock_codes:
        stock_data = [s for s in all_snapshots if s.get('stock_code') == stock_code]
        if not stock_data:
            continue

        pipeline.reset()
        features, timestamps, codes = snapshots_to_features(stock_data, pipeline)
        all_features.append(features)
        all_timestamps.extend(timestamps)
        all_stock_codes.extend(codes)

    features = np.vstack(all_features)
    print(f"Feature matrix shape: {features.shape}")

    return features, all_timestamps, all_stock_codes, stock_stats


def train_single_seed(
    df: pd.DataFrame,
    feature_cols: List[str],
    seed: int,
    output_dir: str,
    params: dict = None,
) -> Dict:
    """Train CatBoost with a single seed and return results."""
    if params is None:
        params = CATBOOST_PARAMS.copy()

    print(f"\n{'='*70}")
    print(f"Training with seed={seed}")
    print(f"{'='*70}")

    # Temporal split
    train_df, val_df, test_df = temporal_train_val_test_split(
        df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, by_stock=True
    )

    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values.astype(int)
    X_val = val_df[feature_cols].values
    y_val = val_df['label'].values.astype(int)
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values.astype(int)

    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # Train
    model = CatBoostClassifier(**params, random_seed=seed)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100)

    # Evaluate
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    test_f1 = f1_score(y_test, y_test_pred, average='macro')

    print(f"\n  Train Accuracy:  {train_acc*100:.2f}%  |  Macro F1: {train_f1:.4f}")
    print(f"  Val Accuracy:    {val_acc*100:.2f}%  |  Macro F1: {val_f1:.4f}")
    print(f"  Test Accuracy:   {test_acc*100:.2f}%  |  Macro F1: {test_f1:.4f}")

    # Classification report
    report = classification_report(y_test, y_test_pred, target_names=['Down', 'Stay', 'Up'])
    print(f"\n  Test Classification Report:\n{report}")

    cm = confusion_matrix(y_test, y_test_pred)
    print(f"  Confusion Matrix:\n{cm}")

    # Feature importance
    feature_importance = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    # Per-class metrics
    per_class_report = classification_report(
        y_test, y_test_pred, target_names=['Down', 'Stay', 'Up'], output_dict=True
    )

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"multi_catboost_seed_{seed}.cbm")
    model.save_model(model_path)

    result = {
        'seed': seed,
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'test_accuracy': float(test_acc),
        'train_f1_macro': float(train_f1),
        'val_f1_macro': float(val_f1),
        'test_f1_macro': float(test_f1),
        'n_train': len(train_df),
        'n_val': len(val_df),
        'n_test': len(test_df),
        'confusion_matrix': cm.tolist(),
        'per_class': per_class_report,
        'top_10_features': importance_df.head(10).to_dict('records'),
        'model_path': model_path,
    }

    # Clean up to save memory
    del model, X_train, X_val, X_test
    gc.collect()

    return result


def run_multi_seed_validation(
    df: pd.DataFrame,
    feature_cols: List[str],
    seeds: List[int],
    output_dir: str,
    params: dict = None,
) -> Dict:
    """Run multi-seed validation and compute statistics."""
    all_results = []

    for seed in seeds:
        result = train_single_seed(df, feature_cols, seed, output_dir, params)
        all_results.append(result)

    # Aggregate statistics
    test_accs = [r['test_accuracy'] for r in all_results]
    val_accs = [r['val_accuracy'] for r in all_results]
    train_accs = [r['train_accuracy'] for r in all_results]
    test_f1s = [r['test_f1_macro'] for r in all_results]

    mean_test = np.mean(test_accs)
    std_test = np.std(test_accs, ddof=1) if len(test_accs) > 1 else 0.0
    mean_f1 = np.mean(test_f1s)
    std_f1 = np.std(test_f1s, ddof=1) if len(test_f1s) > 1 else 0.0

    print("\n" + "=" * 70)
    print("MULTI-SEED VALIDATION RESULTS")
    print("=" * 70)
    print(f"Seeds: {seeds}")
    print(f"Number of runs: {len(seeds)}")
    print()
    print(f"Train Accuracy: {np.mean(train_accs)*100:.2f}% +/- {np.std(train_accs, ddof=1)*100:.2f}%")
    print(f"Val Accuracy:   {np.mean(val_accs)*100:.2f}% +/- {np.std(val_accs, ddof=1)*100:.2f}%")
    print(f"Test Accuracy:  {mean_test*100:.2f}% +/- {std_test*100:.2f}%")
    print(f"Test Macro F1:  {mean_f1:.4f} +/- {std_f1:.4f}")
    print()

    # Per-seed results
    print("Per-seed results:")
    for r in all_results:
        print(f"  Seed {r['seed']}: Acc={r['test_accuracy']*100:.2f}%  F1={r['test_f1_macro']:.4f}")
    print()

    # Statistical test: test accuracy > random baseline (33.33%)
    if len(test_accs) >= 3:
        t_stat, p_value = stats.ttest_1samp(test_accs, 1.0 / 3.0)
        print(f"One-sample t-test (vs random 33.33%):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value:     {p_value:.6f}")
        if p_value < 0.001:
            print(f"  Result: Highly significant (p < 0.001)")
        elif p_value < 0.01:
            print(f"  Result: Significant (p < 0.01)")
        elif p_value < 0.05:
            print(f"  Result: Significant (p < 0.05)")
        else:
            print(f"  Result: Not significant (p >= 0.05)")
        print()
    else:
        t_stat, p_value = None, None

    # Summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'seeds': seeds,
        'n_runs': len(seeds),
        'test_accuracy_mean': float(mean_test),
        'test_accuracy_std': float(std_test),
        'test_f1_macro_mean': float(mean_f1),
        'test_f1_macro_std': float(std_f1),
        'val_accuracy_mean': float(np.mean(val_accs)),
        'val_accuracy_std': float(np.std(val_accs, ddof=1)) if len(val_accs) > 1 else 0.0,
        'train_accuracy_mean': float(np.mean(train_accs)),
        'train_accuracy_std': float(np.std(train_accs, ddof=1)) if len(train_accs) > 1 else 0.0,
        'per_seed_test_accuracy': {r['seed']: r['test_accuracy'] for r in all_results},
        'per_seed_test_f1': {r['seed']: r['test_f1_macro'] for r in all_results},
        't_statistic': float(t_stat) if t_stat is not None else None,
        'p_value': float(p_value) if p_value is not None else None,
        'hyperparameters': params or CATBOOST_PARAMS,
        'per_seed_results': all_results,
    }

    # Save summary
    summary_path = os.path.join(output_dir, "multi_seed_results.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Results saved to {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Multi-stock/multi-date CatBoost training')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Root data directory')
    parser.add_argument('--stocks', nargs='+', type=str, default=None,
                        help='Stock codes to use (default: all available)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456, 789, 1011],
                        help='Random seeds for multi-seed validation')
    parser.add_argument('--k', type=int, default=100,
                        help='Prediction horizon (default: 100)')
    parser.add_argument('--threshold', type=float, default=0.0001,
                        help='Label threshold (default: 0.0001 = 0.01%%)')
    parser.add_argument('--output-dir', type=str, default='models/multi',
                        help='Output directory')
    parser.add_argument('--max-dates', type=int, default=None,
                        help='Max dates per stock (for testing)')
    parser.add_argument('--balanced', action='store_true',
                        help='Use class_weights to boost minority classes')
    args = parser.parse_args()

    print("=" * 70)
    print("LOB Mid-Price Prediction - Multi-Stock/Multi-Date Training")
    print("=" * 70)
    print(f"Data directory:  {args.data_dir}")
    print(f"Stocks:          {args.stocks or 'all available'}")
    print(f"Seeds:           {args.seeds}")
    print(f"Horizon:         k={args.k}")
    print(f"Threshold:       {args.threshold*100:.4f}%")
    print(f"Class balancing: {'Enabled (3:1:3)' if args.balanced else 'Disabled'}")
    print()

    start_time = time.time()

    # Step 1: Load multi-stock data
    print("Step 1: Loading multi-stock/multi-date data...")
    features, timestamps, stock_codes, stock_stats = load_multi_stock_data(
        args.data_dir,
        stock_codes=args.stocks,
        max_dates_per_stock=args.max_dates,
    )

    # Step 2: Create DataFrame
    print("\nStep 2: Creating DataFrame...")
    pipeline = FeatureEngineeringPipeline()
    feature_cols = pipeline.get_feature_names()
    df = create_dataframe(features, timestamps, stock_codes, feature_names=feature_cols)
    print(f"  DataFrame shape: {df.shape}")
    print(f"  Stocks: {sorted(df['stock_code'].unique())}")
    print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Step 3: Generate labels
    print(f"\nStep 3: Generating labels (k={args.k})...")
    labels = generate_labels_from_dataframe(
        df, k=args.k, threshold_pct=args.threshold, group_by_stock=True
    )
    df['label'] = labels
    print_label_distribution(labels.values, title="Overall Label Distribution")

    # Per-stock label distribution
    for stock_code in sorted(df['stock_code'].unique()):
        stock_labels = df[df['stock_code'] == stock_code]['label'].dropna().values
        if len(stock_labels) > 0:
            counts = np.bincount(stock_labels.astype(int), minlength=3)
            total = len(stock_labels)
            print(f"  {stock_code}: Down={counts[0]} ({counts[0]/total*100:.1f}%), "
                  f"Stay={counts[1]} ({counts[1]/total*100:.1f}%), "
                  f"Up={counts[2]} ({counts[2]/total*100:.1f}%)")

    # Step 4: Clean NaN labels
    print(f"\nStep 4: Removing NaN labels...")
    df_clean = df.dropna(subset=['label']).reset_index(drop=True)
    print(f"  Clean samples: {len(df_clean)} (removed {len(df) - len(df_clean)} NaN)")

    # Step 5: Configure params
    if args.balanced:
        params = CATBOOST_PARAMS_BALANCED.copy()
    else:
        params = CATBOOST_PARAMS_UNBALANCED.copy()

    # Step 6: Multi-seed validation
    print(f"\nStep 5: Running multi-seed validation ({len(args.seeds)} seeds)...")
    summary = run_multi_seed_validation(
        df_clean, feature_cols, args.seeds, args.output_dir, params
    )

    elapsed = time.time() - start_time

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Stocks:           {list(stock_stats.keys())}")
    print(f"Total snapshots:  {features.shape[0]}")
    print(f"Clean samples:    {len(df_clean)}")
    print(f"Features:         {features.shape[1]}")
    print(f"Seeds:            {args.seeds}")
    print(f"Horizon:          k={args.k}")
    print(f"Class balancing:  {'Enabled (3:1:3)' if args.balanced else 'Disabled'}")
    print()
    print(f"Test Accuracy:    {summary['test_accuracy_mean']*100:.2f}% +/- {summary['test_accuracy_std']*100:.2f}%")
    if summary['p_value'] is not None:
        print(f"p-value (vs 33%): {summary['p_value']:.6f}")
    print(f"Total time:       {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print()

    # Save stock stats to summary
    summary['stock_stats'] = stock_stats
    summary['total_time_seconds'] = elapsed
    summary_path = os.path.join(args.output_dir, "multi_seed_results.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"All results saved to {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
