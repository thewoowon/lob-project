"""
Feature Ablation Study

Purpose: Identify which feature groups contribute most to performance

Feature Groups (38 engineered features):
1. Price features (6): mid-price, spread, VWAP, etc.
2. Volume features (8): ratios, cumulative volumes
3. Order Imbalance (OI) features (6): asymmetry metrics
4. Order Flow Imbalance (OFI) features (6): dynamic flow
5. Depth features (6): depth imbalance, weighted prices
6. Price Impact features (6): market order impact estimation

Experiment:
- Raw only (baseline)
- Raw + Group 1
- Raw + Group 2
- ...
- Raw + Group 6
- Raw + All groups (full model)

Analysis:
- Which group contributes most?
- Are groups complementary?
- Can we reduce feature count without losing performance?
"""

import sys
sys.path.insert(0, '/Users/aepeul/lob-project/lob_preprocessing')

import numpy as np
import pandas as pd
from pathlib import Path

from data.fi2010_loader import FI2010Loader
from data.feature_engineering import LOBFeatureEngineering
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import time


def run_ablation_experiment(
    X_train_raw, y_train,
    X_test_raw, y_test,
    config_name, feature_indices,
    random_seed=42
):
    """
    Run single ablation experiment with specific feature subset.

    Args:
        X_train_raw: Raw training features (40 features)
        y_train: Training labels
        X_test_raw: Raw test features
        y_test: Test labels
        config_name: Name of configuration (for logging)
        feature_indices: Indices of engineered features to include (None = all)
        random_seed: Random seed

    Returns:
        dict with results
    """
    print(f"\n{'='*60}")
    print(f"{config_name} (seed={random_seed})")
    print(f"{'='*60}")

    # Extract features
    fe = LOBFeatureEngineering(depth=10)

    if feature_indices is None:
        # Use all engineered features
        X_train, feature_names = fe.extract_all_features(X_train_raw, include_raw=True)
        X_test, _ = fe.extract_all_features(X_test_raw, include_raw=True)
    elif len(feature_indices) == 0:
        # Raw only (no engineered features)
        X_train = X_train_raw
        X_test = X_test_raw
        feature_names = []
    else:
        # Extract all, then select subset
        X_train_all, all_feature_names = fe.extract_all_features(X_train_raw, include_raw=False)
        X_test_all, _ = fe.extract_all_features(X_test_raw, include_raw=False)

        # Select subset
        X_train_eng = X_train_all[:, feature_indices]
        X_test_eng = X_test_all[:, feature_indices]

        # Combine with raw
        X_train = np.hstack([X_train_raw, X_train_eng])
        X_test = np.hstack([X_test_raw, X_test_eng])

        feature_names = [all_feature_names[i] for i in feature_indices]

    n_features = X_train.shape[1]
    print(f"Features: {n_features} ({n_features - 40} engineered + 40 raw)")

    # Train model
    print("Training...")
    model = CatBoostClassifier(
        iterations=500,
        depth=10,
        learning_rate=0.1,
        loss_function='MultiClass',
        random_seed=random_seed,
        verbose=False
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Evaluate
    print("Evaluating...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Training time: {train_time:.1f}s")

    return {
        'config': config_name,
        'seed': random_seed,
        'n_features': n_features,
        'n_engineered': n_features - 40,
        'accuracy': accuracy,
        'train_time': train_time
    }


def run_ablation_study():
    """
    Run complete feature ablation study.
    """
    print("\n" + "="*60)
    print("FEATURE ABLATION STUDY")
    print("="*60)

    # Load data
    print("\nLoading FI-2010 data...")
    loader = FI2010Loader(normalization='zscore', auction=False)
    X_train_raw, y_train = loader.load_all_training(horizon=100, days=[1, 2])
    X_test_raw, y_test = loader.load_test(horizon=100)

    print(f"Train: {len(X_train_raw):,} samples")
    print(f"Test: {len(X_test_raw):,} samples")

    # Define feature groups
    # (based on feature_engineering.py implementation)
    feature_groups = {
        'Price': list(range(0, 6)),           # 6 features
        'Volume': list(range(6, 14)),         # 8 features
        'OI': list(range(14, 20)),            # 6 features
        'OFI': list(range(20, 26)),           # 6 features
        'Depth': list(range(26, 32)),         # 6 features
        'Impact': list(range(32, 38))         # 6 features
    }

    print("\nFeature groups:")
    for name, indices in feature_groups.items():
        print(f"  {name}: {len(indices)} features (indices {indices[0]}-{indices[-1]})")

    # Run experiments with multiple seeds
    seeds = [42, 123, 456]  # Use 3 seeds for speed
    results = []

    # Config 1: Raw only (baseline)
    print("\n" + "="*60)
    print("BASELINE: Raw only")
    print("="*60)

    for seed in seeds:
        result = run_ablation_experiment(
            X_train_raw, y_train, X_test_raw, y_test,
            "Raw only", [], seed
        )
        results.append(result)

    # Config 2-7: Raw + each group
    for group_name, group_indices in feature_groups.items():
        print("\n" + "="*60)
        print(f"ABLATION: Raw + {group_name}")
        print("="*60)

        for seed in seeds:
            result = run_ablation_experiment(
                X_train_raw, y_train, X_test_raw, y_test,
                f"Raw + {group_name}", group_indices, seed
            )
            results.append(result)

    # Config 8: Raw + All (full model)
    print("\n" + "="*60)
    print("FULL MODEL: Raw + All groups")
    print("="*60)

    for seed in seeds:
        result = run_ablation_experiment(
            X_train_raw, y_train, X_test_raw, y_test,
            "Raw + All", None, seed
        )
        results.append(result)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Aggregate results
    print("\n" + "="*60)
    print("ABLATION RESULTS (averaged over seeds)")
    print("="*60)
    print()

    summary = df.groupby('config').agg({
        'n_engineered': 'first',
        'accuracy': ['mean', 'std'],
        'train_time': 'mean'
    }).round(4)

    print(summary)
    print()

    # Calculate improvements over baseline
    baseline_acc = df[df['config'] == 'Raw only']['accuracy'].mean()

    print("="*60)
    print("IMPROVEMENT OVER BASELINE")
    print("="*60)
    print()
    print(f"Baseline (Raw only): {baseline_acc*100:.2f}%")
    print()

    improvements = []
    for config in df['config'].unique():
        if config == 'Raw only':
            continue

        config_data = df[df['config'] == config]
        mean_acc = config_data['accuracy'].mean()
        std_acc = config_data['accuracy'].std()
        improvement = (mean_acc - baseline_acc) * 100
        n_features = config_data['n_engineered'].iloc[0]

        improvements.append({
            'config': config,
            'n_features': n_features,
            'accuracy': mean_acc * 100,
            'std': std_acc * 100,
            'improvement': improvement
        })

    # Sort by improvement
    improvements_df = pd.DataFrame(improvements)
    improvements_df = improvements_df.sort_values('improvement', ascending=False)

    print(improvements_df.to_string(index=False))
    print()

    # Find best single group
    single_groups = improvements_df[improvements_df['n_features'] <= 8].copy()
    if len(single_groups) > 0:
        best_group = single_groups.iloc[0]
        print(f"Best single group: {best_group['config']}")
        print(f"  Improvement: +{best_group['improvement']:.2f} pp")
        print(f"  Features: {int(best_group['n_features'])}")
        print()

    # Analyze complementarity
    full_model_acc = df[df['config'] == 'Raw + All']['accuracy'].mean()
    full_improvement = (full_model_acc - baseline_acc) * 100

    sum_individual = improvements_df[improvements_df['n_features'] <= 8]['improvement'].sum()

    print("="*60)
    print("COMPLEMENTARITY ANALYSIS")
    print("="*60)
    print()
    print(f"Sum of individual group improvements: {sum_individual:.2f} pp")
    print(f"Full model improvement: {full_improvement:.2f} pp")
    print()

    if full_improvement > sum_individual:
        print("✅ Groups are complementary (synergy effect)")
        print(f"   Synergy bonus: +{(full_improvement - sum_individual):.2f} pp")
    else:
        print("⚠️ No clear synergy (may have redundancy)")

    # Save results
    results_dir = Path(__file__).parent.parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    output_file = results_dir / 'feature_ablation_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Create table for paper
    print("\n" + "="*60)
    print("TABLE FOR PAPER")
    print("="*60)
    print()
    print("Configuration        | Features | Accuracy (%) | Δ vs Raw")
    print("-" * 65)
    print(f"Raw only             |    40    | {baseline_acc*100:.2f} ± {df[df['config']=='Raw only']['accuracy'].std()*100:.2f} | -")

    for _, row in improvements_df.iterrows():
        total_features = int(row['n_features']) + 40
        config_name = row['config'].replace('Raw + ', '')
        print(f"{config_name:20} | {total_features:8} | {row['accuracy']:.2f} ± {row['std']:.2f} | +{row['improvement']:.2f} pp")

    print("-" * 65)
    print()

    return df


if __name__ == "__main__":
    results = run_ablation_study()
