"""
Statistical Validation: Multiple Random Seeds

Run experiments with 5 different seeds to ensure results are statistically significant.
"""

import sys
sys.path.insert(0, '/Users/aepeul/lob-project/lob_preprocessing')

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import logging

from data.fi2010_loader import FI2010Loader
from data.feature_engineering import LOBFeatureEngineering
from models.baseline import XGBoostModel
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

logging.basicConfig(level=logging.WARNING)


def run_single_seed(seed, X_train_raw, y_train, X_test_raw, y_test, fe):
    """Run experiment with single seed"""

    print(f"\n{'='*70}")
    print(f"   Seed {seed}")
    print(f"{'='*70}")

    # Set seed
    np.random.seed(seed)

    results = {}

    # 1. Raw baseline
    print(f"   Running: Raw baseline...")
    model = XGBoostModel(max_depth=10, n_estimators=100, learning_rate=0.1, random_state=seed)
    model.fit(X_train_raw, y_train)
    y_pred = model.predict(X_test_raw)

    results['raw'] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'mcc': matthews_corrcoef(y_test, y_pred)
    }
    print(f"      Accuracy: {results['raw']['accuracy']:.4f}")

    # 2. Raw + Engineered Features
    print(f"   Running: Raw + Engineered...")
    X_train_fe, _ = fe.extract_all_features(X_train_raw, include_raw=True)
    X_test_fe, _ = fe.extract_all_features(X_test_raw, include_raw=True)

    model = XGBoostModel(max_depth=10, n_estimators=100, learning_rate=0.1, random_state=seed)
    model.fit(X_train_fe, y_train)
    y_pred = model.predict(X_test_fe)

    results['raw_fe'] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'mcc': matthews_corrcoef(y_test, y_pred)
    }
    print(f"      Accuracy: {results['raw_fe']['accuracy']:.4f}")

    # 3. Engineered only
    print(f"   Running: Engineered only...")
    X_train_fe_only, _ = fe.extract_all_features(X_train_raw, include_raw=False)
    X_test_fe_only, _ = fe.extract_all_features(X_test_raw, include_raw=False)

    model = XGBoostModel(max_depth=10, n_estimators=100, learning_rate=0.1, random_state=seed)
    model.fit(X_train_fe_only, y_train)
    y_pred = model.predict(X_test_fe_only)

    results['fe_only'] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'mcc': matthews_corrcoef(y_test, y_pred)
    }
    print(f"      Accuracy: {results['fe_only']['accuracy']:.4f}")

    return results


def main():
    print("\n" + "="*70)
    print(" 📊 STATISTICAL VALIDATION - Multiple Random Seeds")
    print("="*70)
    print()

    # Seeds
    seeds = [42, 123, 456, 789, 1011]

    print(f"Running {len(seeds)} seeds: {seeds}")
    print()

    # Load data once
    print("📥 Loading FI-2010 data...")
    loader = FI2010Loader(normalization='zscore', auction=False)
    X_train_raw, y_train = loader.load_all_training(horizon=100, days=[1, 2])
    X_test_raw, y_test = loader.load_test(horizon=100)

    print(f"✅ Data loaded: {X_train_raw.shape[0]:,} train, {X_test_raw.shape[0]:,} test")
    print()

    # Initialize feature engineering
    fe = LOBFeatureEngineering(depth=10)

    # Run all seeds
    all_results = []

    for seed in seeds:
        results = run_single_seed(seed, X_train_raw, y_train, X_test_raw, y_test, fe)

        all_results.append({
            'seed': seed,
            'raw_acc': results['raw']['accuracy'],
            'raw_f1': results['raw']['f1_macro'],
            'raw_mcc': results['raw']['mcc'],
            'raw_fe_acc': results['raw_fe']['accuracy'],
            'raw_fe_f1': results['raw_fe']['f1_macro'],
            'raw_fe_mcc': results['raw_fe']['mcc'],
            'fe_only_acc': results['fe_only']['accuracy'],
            'fe_only_f1': results['fe_only']['f1_macro'],
            'fe_only_mcc': results['fe_only']['mcc'],
        })

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save
    df.to_csv('results/statistical_validation.csv', index=False)
    print(f"\n✅ Results saved to results/statistical_validation.csv")

    # Compute statistics
    print("\n" + "="*70)
    print(" 📊 STATISTICAL ANALYSIS")
    print("="*70)
    print()

    print("Configuration               Mean ± Std         95% CI           Min-Max")
    print("-" * 70)

    for config in ['raw', 'fe_only', 'raw_fe']:
        acc_col = f'{config}_acc'
        accs = df[acc_col].values

        mean = np.mean(accs)
        std = np.std(accs, ddof=1)
        ci = 1.96 * std / np.sqrt(len(accs))
        min_val = np.min(accs)
        max_val = np.max(accs)

        config_name = {
            'raw': 'Raw baseline',
            'fe_only': 'Engineered only',
            'raw_fe': 'Raw + Engineered'
        }[config]

        print(f"{config_name:25s}   {mean:.4f} ± {std:.4f}   [{mean-ci:.4f}, {mean+ci:.4f}]   {min_val:.4f}-{max_val:.4f}")

    print()

    # Statistical tests
    print("="*70)
    print(" 🔬 STATISTICAL SIGNIFICANCE TESTS")
    print("="*70)
    print()

    # Test 1: Raw vs Raw+FE
    raw_accs = df['raw_acc'].values
    raw_fe_accs = df['raw_fe_acc'].values

    t_stat, p_value = ttest_rel(raw_accs, raw_fe_accs)
    mean_diff = np.mean(raw_fe_accs - raw_accs)

    print("Test 1: Raw baseline vs Raw + Engineered")
    print(f"   Mean difference: {mean_diff:.4f} ({mean_diff*100:.2f} pp)")
    print(f"   t-statistic:     {t_stat:.4f}")
    print(f"   p-value:         {p_value:.6f}")

    if p_value < 0.001:
        print(f"   ✅ HIGHLY SIGNIFICANT (p < 0.001)")
    elif p_value < 0.01:
        print(f"   ✅ VERY SIGNIFICANT (p < 0.01)")
    elif p_value < 0.05:
        print(f"   ✅ SIGNIFICANT (p < 0.05)")
    else:
        print(f"   ❌ NOT SIGNIFICANT (p >= 0.05)")
    print()

    # Test 2: Raw vs Engineered only
    fe_only_accs = df['fe_only_acc'].values

    t_stat2, p_value2 = ttest_rel(raw_accs, fe_only_accs)
    mean_diff2 = np.mean(fe_only_accs - raw_accs)

    print("Test 2: Raw baseline vs Engineered only")
    print(f"   Mean difference: {mean_diff2:.4f} ({mean_diff2*100:.2f} pp)")
    print(f"   t-statistic:     {t_stat2:.4f}")
    print(f"   p-value:         {p_value2:.6f}")

    if p_value2 < 0.001:
        print(f"   ✅ HIGHLY SIGNIFICANT (p < 0.001)")
    elif p_value2 < 0.01:
        print(f"   ✅ VERY SIGNIFICANT (p < 0.01)")
    elif p_value2 < 0.05:
        print(f"   ✅ SIGNIFICANT (p < 0.05)")
    else:
        print(f"   ❌ NOT SIGNIFICANT (p >= 0.05)")
    print()

    # Final verdict
    print("="*70)
    print(" 🎯 FINAL VERDICT")
    print("="*70)
    print()

    if p_value < 0.05 and p_value2 < 0.05:
        print("✅ Feature Engineering is STATISTICALLY SIGNIFICANTLY better than baseline")
        print(f"✅ Mean improvement: {mean_diff*100:.2f} percentage points")
        print(f"✅ Both tests pass p < 0.05 threshold")
        print()
        print("📝 Safe to report in paper:")
        print(f"   \"Feature engineering achieves {np.mean(raw_fe_accs):.2%} accuracy")
        print(f"    vs {np.mean(raw_accs):.2%} baseline (p < {p_value:.3f})\"")
        print()
        return True
    else:
        print("⚠️  Results may not be statistically significant")
        print("⚠️  Need more careful interpretation or more runs")
        print()
        return False


if __name__ == '__main__':
    try:
        passed = main()
        if passed:
            print("\n🎉 Statistical validation complete - Results are robust!")
            exit(0)
        else:
            print("\n⚠️  Statistical validation uncertain - Be cautious!")
            exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
