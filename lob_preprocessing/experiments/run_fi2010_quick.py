"""
FI-2010 Quick Validation - Saves results incrementally

Simpler version that saves after each config
"""

import sys
sys.path.insert(0, '/Users/aepeul/lob-project/lob_preprocessing')

import numpy as np
import pandas as pd
import logging

from data.fi2010_loader import FI2010Loader
from data.preprocess import LOBPreprocessor
from models.baseline import XGBoostModel, CatBoostModel
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_one_config(preprocess_method, model_name, X_train, y_train, X_test, y_test):
    """Run one configuration and return results"""

    print(f"\n{'='*70}")
    print(f"▶ Configuration: {preprocess_method.upper()} + {model_name.upper()}")
    print(f"{'='*70}")

    # Preprocess
    if preprocess_method != 'raw':
        print(f"   Preprocessing with {preprocess_method}...")

        preprocessor = LOBPreprocessor(method=preprocess_method)

        # Use first column as proxy for mid-price
        mid_price_train = X_train[:, 0].flatten()
        mid_price_test = X_test[:, 0].flatten()

        # Fit and transform
        mid_price_train_proc = preprocessor.fit_transform(mid_price_train)
        mid_price_test_proc = preprocessor.transform(mid_price_test)

        # Ensure 2D shape
        if mid_price_train_proc.ndim == 1:
            mid_price_train_proc = mid_price_train_proc.reshape(-1, 1)
        if mid_price_test_proc.ndim == 1:
            mid_price_test_proc = mid_price_test_proc.reshape(-1, 1)

        # Combine back
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
        model = XGBoostModel(max_depth=10, n_estimators=100, learning_rate=0.1)
    elif model_name == 'catboost':
        model = CatBoostModel(depth=10, iterations=100, learning_rate=0.1, verbose=False)

    model.fit(X_train_proc, y_train)

    # Predict
    print(f"   Evaluating...")
    y_pred = model.predict(X_test_proc)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"\n   📊 Results:")
    print(f"      Accuracy: {acc:.4f}")
    print(f"      F1-Macro: {f1:.4f}")
    print(f"      MCC: {mcc:.4f}\n")

    return {
        'preprocess': preprocess_method,
        'model': model_name,
        'accuracy': acc,
        'f1_macro': f1,
        'mcc': mcc,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }


def main():
    print("\n" + "="*70)
    print(" 🔬 FI-2010 REAL DATA VALIDATION (Quick)")
    print("="*70 + "\n")

    # Load FI-2010 data
    print("📥 Loading FI-2010 data...")
    loader = FI2010Loader(normalization='zscore', auction=False)

    X_train, y_train = loader.load_all_training(horizon=100, days=[1, 2])
    X_test, y_test = loader.load_test(horizon=100)

    print(f"✅ Data loaded")
    print(f"   Training: {len(X_train):,} samples")
    print(f"   Test: {len(X_test):,} samples\n")

    # Key configurations
    configs = [
        ('raw', 'xgboost'),
        ('wavelet', 'xgboost'),
        ('kalman', 'xgboost'),
        ('raw', 'catboost'),
        ('wavelet', 'catboost'),
    ]

    results = []

    for preprocess_method, model_name in configs:
        try:
            result = run_one_config(preprocess_method, model_name, X_train, y_train, X_test, y_test)
            results.append(result)

            # Save incrementally
            results_df = pd.DataFrame(results)
            results_df.to_csv('results/fi2010_validation_incremental.csv', index=False)
            print(f"✅ Saved results ({len(results)}/{len(configs)} configs)")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    print("\n" + "="*70)
    print(" 📊 FI-2010 VALIDATION SUMMARY")
    print("="*70 + "\n")

    results_df = pd.DataFrame(results)
    print(results_df[['preprocess', 'model', 'accuracy', 'f1_macro', 'mcc']].to_string(index=False))
    print()

    # Save final
    results_df.to_csv('results/fi2010_validation.csv', index=False)
    print(f"✅ Final results saved to results/fi2010_validation.csv\n")

    return results_df


if __name__ == '__main__':
    try:
        results = main()
        print("🎉 FI-2010 validation completed!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
