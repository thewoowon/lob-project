"""
Cross-Stock Analysis

Purpose: Verify that improvement is consistent across all 5 stocks
         (not just lucky on one stock)

Experiment:
- Train and test on each stock separately
- Compare Raw vs Raw+Engineered for each
- Check if improvement is consistent

Expected:
- All stocks should show improvement
- Similar magnitude (+5~7 pp range)
- All statistically significant
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
from scipy import stats


def run_stock_experiment(stock_id, random_seed=42):
    """
    Run experiment on single stock.

    Args:
        stock_id: 1-5 (FI-2010 has 5 stocks)
        random_seed: random seed for reproducibility

    Returns:
        dict with results
    """
    print(f"\n{'='*60}")
    print(f"Stock {stock_id} (seed={random_seed})")
    print(f"{'='*60}")

    # Load data for this stock only
    loader = FI2010Loader(normalization='zscore', auction=False)

    # Load training data (days 1-7) for this stock
    X_train_raw, y_train = loader.load_day(
        stock_num=stock_id,
        day=1,  # Start with day 1
        horizon=100
    )

    # Add more training days
    for day in range(2, 8):  # Days 2-7
        X_day, y_day = loader.load_day(stock_num=stock_id, day=day, horizon=100)
        X_train_raw = np.vstack([X_train_raw, X_day])
        y_train = np.concatenate([y_train, y_day])

    # Load test data (days 9-10) for this stock
    X_test_raw, y_test = loader.load_day(stock_num=stock_id, day=9, horizon=100)
    X_day, y_day = loader.load_day(stock_num=stock_id, day=10, horizon=100)
    X_test_raw = np.vstack([X_test_raw, X_day])
    y_test = np.concatenate([y_test, y_day])

    print(f"Train: {len(X_train_raw):,} samples")
    print(f"Test: {len(X_test_raw):,} samples")

    # Extract engineered features
    fe = LOBFeatureEngineering(depth=10)
    X_train_eng, _ = fe.extract_all_features(X_train_raw, include_raw=True)
    X_test_eng, _ = fe.extract_all_features(X_test_raw, include_raw=True)

    print(f"Raw features: {X_train_raw.shape[1]}")
    print(f"Raw+Eng features: {X_train_eng.shape[1]}")

    # Train models
    print("\nTraining models...")

    # Model 1: Raw only
    model_raw = CatBoostClassifier(
        iterations=500,
        depth=10,
        learning_rate=0.1,
        loss_function='MultiClass',
        random_seed=random_seed,
        verbose=False
    )
    model_raw.fit(X_train_raw, y_train)
    y_pred_raw = model_raw.predict(X_test_raw)
    acc_raw = accuracy_score(y_test, y_pred_raw)

    # Model 2: Raw + Engineered
    model_eng = CatBoostClassifier(
        iterations=500,
        depth=10,
        learning_rate=0.1,
        loss_function='MultiClass',
        random_seed=random_seed,
        verbose=False
    )
    model_eng.fit(X_train_eng, y_train)
    y_pred_eng = model_eng.predict(X_test_eng)
    acc_eng = accuracy_score(y_test, y_pred_eng)

    # Results
    improvement = (acc_eng - acc_raw) * 100

    print(f"\nResults:")
    print(f"  Raw:          {acc_raw*100:.2f}%")
    print(f"  Raw+Eng:      {acc_eng*100:.2f}%")
    print(f"  Improvement:  {improvement:+.2f} pp")

    return {
        'stock': stock_id,
        'seed': random_seed,
        'acc_raw': acc_raw,
        'acc_eng': acc_eng,
        'improvement': improvement,
        'n_train': len(X_train_raw),
        'n_test': len(X_test_raw)
    }


def run_cross_stock_analysis():
    """
    Run analysis across all 5 stocks with multiple seeds.
    """
    print("\n" + "="*60)
    print("CROSS-STOCK ANALYSIS")
    print("="*60)

    # We'll use 3 seeds per stock for speed (instead of 5)
    # Can increase to 5 if needed
    seeds = [42, 123, 456]
    stocks = [1, 2, 3, 4, 5]

    results = []

    # Run experiments
    for stock in stocks:
        for seed in seeds:
            result = run_stock_experiment(stock, seed)
            results.append(result)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Aggregate by stock
    print("\n" + "="*60)
    print("PER-STOCK RESULTS (averaged over seeds)")
    print("="*60)
    print()

    summary = df.groupby('stock').agg({
        'acc_raw': ['mean', 'std'],
        'acc_eng': ['mean', 'std'],
        'improvement': ['mean', 'std'],
        'n_train': 'first',
        'n_test': 'first'
    }).round(4)

    print(summary)
    print()

    # Check if all stocks show improvement
    print("="*60)
    print("CONSISTENCY CHECK")
    print("="*60)
    print()

    for stock in stocks:
        stock_data = df[df['stock'] == stock]
        acc_raw = stock_data['acc_raw'].values
        acc_eng = stock_data['acc_eng'].values

        # Paired t-test
        t_stat, p_val = stats.ttest_rel(acc_raw, acc_eng)

        mean_raw = acc_raw.mean() * 100
        mean_eng = acc_eng.mean() * 100
        improvement = mean_eng - mean_raw

        sig = "✅" if p_val < 0.05 else "❌"

        print(f"Stock {stock}:")
        print(f"  Raw: {mean_raw:.2f}% → Eng: {mean_eng:.2f}%")
        print(f"  Improvement: {improvement:+.2f} pp")
        print(f"  p-value: {p_val:.6f} {sig}")
        print()

    # Overall statistics
    print("="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    print()

    # Average across all stocks and seeds
    mean_raw = df['acc_raw'].mean() * 100
    std_raw = df['acc_raw'].std() * 100
    mean_eng = df['acc_eng'].mean() * 100
    std_eng = df['acc_eng'].std() * 100
    mean_imp = df['improvement'].mean()
    std_imp = df['improvement'].std()

    print(f"Raw baseline:      {mean_raw:.2f}% ± {std_raw:.2f}%")
    print(f"Raw + Engineered:  {mean_eng:.2f}% ± {std_eng:.2f}%")
    print(f"Improvement:       {mean_imp:+.2f} ± {std_imp:.2f} pp")
    print()

    # Check if any stock shows degradation
    min_improvement = df.groupby('stock')['improvement'].mean().min()
    max_improvement = df.groupby('stock')['improvement'].mean().max()

    print(f"Min improvement (across stocks): {min_improvement:+.2f} pp")
    print(f"Max improvement (across stocks): {max_improvement:+.2f} pp")
    print()

    if min_improvement > 0:
        print("✅ All stocks show positive improvement")
    else:
        print("❌ Some stocks show degradation")

    if max_improvement - min_improvement < 2.0:
        print("✅ Improvement is consistent (range < 2 pp)")
    else:
        print("⚠️ Improvement varies across stocks (range > 2 pp)")

    # Save results
    results_dir = Path(__file__).parent.parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    output_file = results_dir / 'cross_stock_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Create summary table for paper
    print("\n" + "="*60)
    print("TABLE FOR PAPER")
    print("="*60)
    print()
    print("Stock | Raw (%)   | Raw+Eng (%) | Improvement | p-value")
    print("-" * 65)

    for stock in stocks:
        stock_data = df[df['stock'] == stock]
        acc_raw = stock_data['acc_raw'].values
        acc_eng = stock_data['acc_eng'].values

        mean_raw = acc_raw.mean() * 100
        std_raw = acc_raw.std() * 100
        mean_eng = acc_eng.mean() * 100
        std_eng = acc_eng.std() * 100
        improvement = mean_eng - mean_raw

        t_stat, p_val = stats.ttest_rel(acc_raw, acc_eng)

        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))

        print(f"  {stock}   | {mean_raw:.1f}±{std_raw:.1f} | {mean_eng:.1f}±{std_eng:.1f}  | {improvement:+.1f} pp     | {p_val:.4f} {sig}")

    print("-" * 65)
    print(f" Avg  | {mean_raw:.1f}±{std_raw:.1f} | {mean_eng:.1f}±{std_eng:.1f}  | {mean_imp:+.1f}±{std_imp:.1f} pp |")
    print()
    print("Note: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant")
    print()

    return df


if __name__ == "__main__":
    results = run_cross_stock_analysis()
