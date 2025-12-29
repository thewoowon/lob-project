"""
Run a single FI-2010 configuration

Usage: python run_single_config.py <preprocess> <model>
Example: python run_single_config.py wavelet xgboost
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

logging.basicConfig(level=logging.WARNING)  # Suppress INFO logs for speed


def main():
    if len(sys.argv) != 3:
        print("Usage: python run_single_config.py <preprocess> <model>")
        print("Example: python run_single_config.py wavelet xgboost")
        sys.exit(1)

    preprocess_method = sys.argv[1]
    model_name = sys.argv[2]

    print(f"\n▶ Running: {preprocess_method.upper()} + {model_name.upper()}")

    # Load data
    print("   Loading data...")
    loader = FI2010Loader(normalization='zscore', auction=False)
    X_train, y_train = loader.load_all_training(horizon=100, days=[1, 2])
    X_test, y_test = loader.load_test(horizon=100)

    # Preprocess
    if preprocess_method != 'raw':
        print(f"   Preprocessing...")
        preprocessor = LOBPreprocessor(method=preprocess_method)

        mid_price_train = X_train[:, 0].flatten()
        mid_price_test = X_test[:, 0].flatten()

        mid_price_train_proc = preprocessor.fit_transform(mid_price_train)
        mid_price_test_proc = preprocessor.transform(mid_price_test)

        if mid_price_train_proc.ndim == 1:
            mid_price_train_proc = mid_price_train_proc.reshape(-1, 1)
        if mid_price_test_proc.ndim == 1:
            mid_price_test_proc = mid_price_test_proc.reshape(-1, 1)

        X_train_proc = np.hstack([mid_price_train_proc, X_train[:, 1:]])
        X_test_proc = np.hstack([mid_price_test_proc, X_test[:, 1:]])
    else:
        X_train_proc = X_train
        X_test_proc = X_test

    # Train
    print(f"   Training {model_name}...")
    if model_name == 'xgboost':
        model = XGBoostModel(max_depth=10, n_estimators=100, learning_rate=0.1)
    elif model_name == 'catboost':
        model = CatBoostModel(depth=10, iterations=100, learning_rate=0.1, verbose=False)

    model.fit(X_train_proc, y_train)

    # Evaluate
    print(f"   Evaluating...")
    y_pred = model.predict(X_test_proc)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"\n   ✅ Results:")
    print(f"      Accuracy: {acc:.4f}")
    print(f"      F1-Macro: {f1:.4f}")
    print(f"      MCC: {mcc:.4f}\n")

    # Append to CSV
    result = {
        'preprocess': preprocess_method,
        'model': model_name,
        'accuracy': acc,
        'f1_macro': f1,
        'mcc': mcc,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }

    output_file = 'results/fi2010_validation_incremental.csv'

    # Append or create
    try:
        df = pd.read_csv(output_file)
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame([result])

    df.to_csv(output_file, index=False)
    print(f"   💾 Saved to {output_file}")


if __name__ == '__main__':
    main()
