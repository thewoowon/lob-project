"""
FI-2010 Real Data Validation

핵심 5개 configuration만 Real FI-2010 데이터로 검증
Synthetic results와 비교
"""

import sys
sys.path.insert(0, '/Users/aepeul/lob-project/lob_preprocessing')

import numpy as np
import pandas as pd
from pathlib import Path
import logging

from data.fi2010_loader import FI2010Loader
from data.preprocess import LOBPreprocessor
from models.baseline import XGBoostModel, CatBoostModel
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_fi2010_validation():
    """
    FI-2010으로 핵심 config 검증

    핵심 질문:
    1. Synthetic에서의 전처리 효과가 Real data에서도 나타나는가?
    2. 얼마나 realistic한가?
    """
    print("\n" + "="*70)
    print(" 🔬 FI-2010 REAL DATA VALIDATION")
    print("="*70)
    print()

    # Load FI-2010 data
    print("📥 Loading FI-2010 data...")
    loader = FI2010Loader(normalization='zscore', auction=False)

    # Use first 2 days for faster testing (can increase later)
    X_train, y_train = loader.load_all_training(horizon=100, days=[1, 2])
    X_test, y_test = loader.load_test(horizon=100)

    print(f"✅ Data loaded")
    print(f"   Training: {len(X_train):,} samples")
    print(f"   Test: {len(X_test):,} samples")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Train label dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"   Test label dist: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    print()

    # Key configurations to test
    configs = [
        ('raw', 'xgboost'),
        ('wavelet', 'xgboost'),
        ('kalman', 'xgboost'),
        ('raw', 'catboost'),
        ('wavelet', 'catboost'),
    ]

    results = []

    for preprocess_method, model_name in configs:
        print("="*70)
        print(f"▶ Configuration: {preprocess_method.upper()} + {model_name.upper()}")
        print("="*70)

        # Preprocess (only mid-price, which is column 0+1 average in LOB)
        if preprocess_method != 'raw':
            print(f"   Preprocessing with {preprocess_method}...")

            preprocessor = LOBPreprocessor(method=preprocess_method)

            # Use first column (ask_price_1) as proxy for mid-price
            mid_price_train = X_train[:, 0].flatten()  # Ensure 1D for preprocessing
            mid_price_test = X_test[:, 0].flatten()

            # Fit and transform
            mid_price_train_proc = preprocessor.fit_transform(mid_price_train)
            mid_price_test_proc = preprocessor.transform(mid_price_test)

            # Ensure 2D shape for hstack
            if mid_price_train_proc.ndim == 1:
                mid_price_train_proc = mid_price_train_proc.reshape(-1, 1)
            if mid_price_test_proc.ndim == 1:
                mid_price_test_proc = mid_price_test_proc.reshape(-1, 1)

            # Combine back with other features
            X_train_proc = np.hstack([mid_price_train_proc, X_train[:, 1:]])
            X_test_proc = np.hstack([mid_price_test_proc, X_test[:, 1:]])

            print(f"   SNR improvement: {preprocessor.compute_snr(mid_price_train, mid_price_train_proc):.2f} dB")
        else:
            print(f"   No preprocessing (raw data)")
            X_train_proc = X_train
            X_test_proc = X_test

        # Train model
        print(f"   Training {model_name}...")

        if model_name == 'xgboost':
            model = XGBoostModel(
                max_depth=10,
                n_estimators=100,
                learning_rate=0.1,
            )
        elif model_name == 'catboost':
            model = CatBoostModel(
                depth=10,
                iterations=100,
                learning_rate=0.1,
                verbose=False
            )

        model.fit(X_train_proc, y_train)

        # Predict
        print(f"   Evaluating...")
        y_pred = model.predict(X_test_proc)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        mcc = matthews_corrcoef(y_test, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n   📊 Results:")
        print(f"      Accuracy: {acc:.4f}")
        print(f"      F1-Macro: {f1:.4f}")
        print(f"      MCC: {mcc:.4f}")
        print(f"\n   Confusion Matrix:")
        print(f"      {cm[0]}")
        print(f"      {cm[1]}")
        print(f"      {cm[2]}")
        print()

        results.append({
            'preprocess': preprocess_method,
            'model': model_name,
            'accuracy': acc,
            'f1_macro': f1,
            'mcc': mcc,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'confusion_matrix': cm.tolist()
        })

    # Summary
    print("\n" + "="*70)
    print(" 📊 FI-2010 VALIDATION SUMMARY")
    print("="*70)

    results_df = pd.DataFrame(results)
    print()
    print(results_df[['preprocess', 'model', 'accuracy', 'f1_macro', 'mcc']].to_string(index=False))
    print()

    # Compare with synthetic
    print("="*70)
    print(" 🔍 SYNTHETIC vs REAL COMPARISON")
    print("="*70)
    print()

    # Synthetic results from our experiments
    synthetic_results = {
        ('raw', 'xgboost'): {'acc': 0.5355, 'f1': 0.3566},
        ('wavelet', 'xgboost'): {'acc': 0.8515, 'f1': 0.5838},
        ('kalman', 'xgboost'): {'acc': 0.8010, 'f1': 0.5272},
        ('raw', 'catboost'): {'acc': 0.4930, 'f1': 0.3318},
        ('wavelet', 'catboost'): {'acc': 0.8515, 'f1': 0.5838},
    }

    comparison = []

    for _, row in results_df.iterrows():
        key = (row['preprocess'], row['model'])
        synthetic = synthetic_results.get(key, {'acc': 0, 'f1': 0})

        real_acc = row['accuracy']
        synth_acc = synthetic['acc']
        diff_acc = real_acc - synth_acc

        print(f"{key[0].upper():8s} + {key[1].upper():8s}:")
        print(f"   Synthetic Acc: {synth_acc:.4f}")
        print(f"   Real (FI-2010): {real_acc:.4f}")
        print(f"   Difference:     {diff_acc:+.4f} ({diff_acc/synth_acc*100:+.1f}%)")
        print()

        comparison.append({
            'config': f"{key[0]} + {key[1]}",
            'synthetic_acc': synth_acc,
            'real_acc': real_acc,
            'diff': diff_acc,
            'diff_pct': diff_acc / synth_acc * 100 if synth_acc > 0 else 0
        })

    # Compute relative improvement (preprocessing effect)
    print("="*70)
    print(" 💡 PREPROCESSING EFFECT COMPARISON")
    print("="*70)
    print()

    raw_xgb_synth = synthetic_results[('raw', 'xgboost')]['acc']
    wavelet_xgb_synth = synthetic_results[('wavelet', 'xgboost')]['acc']
    synth_improvement = (wavelet_xgb_synth - raw_xgb_synth) / raw_xgb_synth * 100

    raw_xgb_real = results_df[(results_df['preprocess'] == 'raw') & (results_df['model'] == 'xgboost')]['accuracy'].values[0]
    wavelet_xgb_real = results_df[(results_df['preprocess'] == 'wavelet') & (results_df['model'] == 'xgboost')]['accuracy'].values[0]
    real_improvement = (wavelet_xgb_real - raw_xgb_real) / raw_xgb_real * 100

    print(f"Wavelet vs Raw improvement (XGBoost):")
    print(f"   Synthetic: +{synth_improvement:.1f}%")
    print(f"   Real (FI-2010): +{real_improvement:.1f}%")
    print()

    if real_improvement > 5:
        print("✅ Preprocessing effect VALIDATED on real data!")
        print("   → Paper claim is supported")
    else:
        print("⚠️  Preprocessing effect weaker on real data")
        print("   → Need to adjust paper claims")

    print()

    # Save results
    results_df.to_csv('results/fi2010_validation.csv', index=False)
    print(f"✅ Results saved to results/fi2010_validation.csv")

    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv('results/fi2010_synthetic_comparison.csv', index=False)
    print(f"✅ Comparison saved to results/fi2010_synthetic_comparison.csv")

    return results_df, comparison_df


if __name__ == '__main__':
    try:
        results, comparison = run_fi2010_validation()
        print("\n🎉 FI-2010 validation completed!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
