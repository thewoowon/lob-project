"""
Feature Engineering vs Preprocessing Comparison on FI-2010

핵심 질문:
Feature Engineering이 Preprocessing보다 효과적인가?

비교:
1. Raw LOB (40 features) - baseline
2. Preprocessed LOB (40 features, wavelet denoised)
3. Engineered Features (38 features, no raw)
4. Raw LOB + Engineered (40 + 38 = 78 features)
5. Preprocessed LOB + Engineered (40 + 38 = 78 features)
"""

import sys
sys.path.insert(0, '/Users/aepeul/lob-project/lob_preprocessing')

import numpy as np
import pandas as pd
import logging

from data.fi2010_loader import FI2010Loader
from data.preprocess import LOBPreprocessor
from data.feature_engineering import LOBFeatureEngineering
from models.baseline import XGBoostModel
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def run_one_config(
    config_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
):
    """Run one configuration"""

    print(f"\n{'='*70}")
    print(f"▶ Configuration: {config_name}")
    print(f"{'='*70}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Training...")

    # Train XGBoost (best model from previous experiments)
    model = XGBoostModel(max_depth=10, n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Evaluate
    print(f"   Evaluating...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"\n   📊 Results:")
    print(f"      Accuracy: {acc:.4f}")
    print(f"      F1-Macro: {f1:.4f}")
    print(f"      MCC: {mcc:.4f}\n")

    return {
        'config': config_name,
        'n_features': X_train.shape[1],
        'accuracy': acc,
        'f1_macro': f1,
        'mcc': mcc
    }


def main():
    print("\n" + "="*70)
    print(" 🔬 FEATURE ENGINEERING vs PREPROCESSING")
    print(" Real Data Validation (FI-2010)")
    print("="*70 + "\n")

    # Load FI-2010 data
    print("📥 Loading FI-2010 data...")
    loader = FI2010Loader(normalization='zscore', auction=False)
    X_train_raw, y_train = loader.load_all_training(horizon=100, days=[1, 2])
    X_test_raw, y_test = loader.load_test(horizon=100)

    print(f"✅ Data loaded: {X_train_raw.shape[0]:,} train, {X_test_raw.shape[0]:,} test\n")

    # Initialize feature engineering
    fe = LOBFeatureEngineering(depth=10)

    # === CONFIG 1: Raw LOB (baseline) ===
    print("Preparing Config 1: Raw LOB...")
    X_train_1 = X_train_raw
    X_test_1 = X_test_raw

    # === CONFIG 2: Preprocessed LOB (wavelet) ===
    print("Preparing Config 2: Preprocessed LOB (wavelet)...")
    preprocessor = LOBPreprocessor(method='wavelet')

    # Preprocess first column only (as in previous experiments)
    mid_price_train = X_train_raw[:, 0].flatten()
    mid_price_test = X_test_raw[:, 0].flatten()

    mid_price_train_proc = preprocessor.fit_transform(mid_price_train).reshape(-1, 1)
    mid_price_test_proc = preprocessor.transform(mid_price_test).reshape(-1, 1)

    X_train_2 = np.hstack([mid_price_train_proc, X_train_raw[:, 1:]])
    X_test_2 = np.hstack([mid_price_test_proc, X_test_raw[:, 1:]])

    # === CONFIG 3: Engineered Features Only ===
    print("Preparing Config 3: Engineered Features (no raw)...")
    X_train_3, feature_names = fe.extract_all_features(X_train_raw, include_raw=False)
    X_test_3, _ = fe.extract_all_features(X_test_raw, include_raw=False)

    print(f"   Engineered {len(feature_names)} features")

    # === CONFIG 4: Raw LOB + Engineered ===
    print("Preparing Config 4: Raw + Engineered...")
    X_train_4, _ = fe.extract_all_features(X_train_raw, include_raw=True)
    X_test_4, _ = fe.extract_all_features(X_test_raw, include_raw=True)

    # === CONFIG 5: Preprocessed LOB + Engineered ===
    print("Preparing Config 5: Preprocessed + Engineered...")
    X_train_5, _ = fe.extract_all_features(X_train_2, include_raw=True)
    X_test_5, _ = fe.extract_all_features(X_test_2, include_raw=True)

    print("\n" + "="*70)
    print(" 🚀 Running Experiments")
    print("="*70)

    configs = [
        ("1. Raw LOB (baseline)", X_train_1, X_test_1),
        ("2. Preprocessed LOB (wavelet)", X_train_2, X_test_2),
        ("3. Engineered Features Only", X_train_3, X_test_3),
        ("4. Raw + Engineered", X_train_4, X_test_4),
        ("5. Preprocessed + Engineered", X_train_5, X_test_5),
    ]

    results = []

    for config_name, X_train, X_test in configs:
        try:
            result = run_one_config(config_name, X_train, y_train, X_test, y_test)
            results.append(result)

            # Save incrementally
            results_df = pd.DataFrame(results)
            results_df.to_csv('results/feature_engineering_comparison.csv', index=False)
            print(f"✅ Saved ({len(results)}/{len(configs)} configs)")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    print("\n" + "="*70)
    print(" 📊 FINAL RESULTS COMPARISON")
    print("="*70 + "\n")

    results_df = pd.DataFrame(results)
    print(results_df[['config', 'n_features', 'accuracy', 'f1_macro', 'mcc']].to_string(index=False))
    print()

    # Calculate improvements
    baseline_acc = results_df[results_df['config'] == '1. Raw LOB (baseline)']['accuracy'].values[0]

    print("="*70)
    print(" 💡 IMPROVEMENT vs BASELINE")
    print("="*70 + "\n")

    for _, row in results_df.iterrows():
        if row['config'] == '1. Raw LOB (baseline)':
            continue

        improvement = (row['accuracy'] - baseline_acc) / baseline_acc * 100
        abs_improvement = row['accuracy'] - baseline_acc

        print(f"{row['config']:40s}")
        print(f"   Accuracy: {row['accuracy']:.4f} ({abs_improvement:+.4f}, {improvement:+.2f}%)")
        print()

    # Find best
    best_idx = results_df['accuracy'].idxmax()
    best_config = results_df.iloc[best_idx]

    print("="*70)
    print(" 🏆 BEST CONFIGURATION")
    print("="*70 + "\n")

    print(f"Config:    {best_config['config']}")
    print(f"Features:  {best_config['n_features']}")
    print(f"Accuracy:  {best_config['accuracy']:.4f}")
    print(f"F1-Macro:  {best_config['f1_macro']:.4f}")
    print(f"MCC:       {best_config['mcc']:.4f}")
    print()

    improvement = (best_config['accuracy'] - baseline_acc) / baseline_acc * 100
    print(f"Improvement vs Baseline: {improvement:+.2f}%")
    print()

    # Key insight
    print("="*70)
    print(" 🎯 KEY INSIGHT")
    print("="*70 + "\n")

    # Compare preprocessing vs feature engineering
    preprocess_acc = results_df[results_df['config'] == '2. Preprocessed LOB (wavelet)']['accuracy'].values[0]
    engineered_acc = results_df[results_df['config'] == '3. Engineered Features Only']['accuracy'].values[0]

    preprocess_improvement = (preprocess_acc - baseline_acc) / baseline_acc * 100
    engineered_improvement = (engineered_acc - baseline_acc) / baseline_acc * 100

    print(f"Preprocessing improvement:        {preprocess_improvement:+.2f}%")
    print(f"Feature Engineering improvement:  {engineered_improvement:+.2f}%")
    print()

    if engineered_improvement > preprocess_improvement:
        print("✅ Feature Engineering > Preprocessing!")
        print("   → LOB-derived features are more informative than denoising")
    else:
        print("⚠️  Preprocessing still better")
        print("   → Need to refine feature engineering")

    print()

    # Save final results
    results_df.to_csv('results/feature_engineering_comparison.csv', index=False)
    print(f"✅ Results saved to results/feature_engineering_comparison.csv\n")

    return results_df


if __name__ == '__main__':
    try:
        results = main()
        print("🎉 Feature Engineering comparison completed!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
