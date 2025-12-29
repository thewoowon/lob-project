"""
CatBoost Training Pipeline

End-to-end training pipeline:
1. Load LOB data from JSONL files
2. Convert to features (78 features)
3. Generate labels (k=100 horizon)
4. Train CatBoost model
5. Evaluate performance

Based on PAPER_DRAFT.md Section 3.4 and 5.6.
"""

import os
import argparse
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
from datetime import datetime

from data_loader import load_jsonl_file, snapshots_to_features, create_dataframe
from generate_labels import generate_labels_from_dataframe, print_label_distribution
from train_test_split import temporal_train_val_test_split, verify_no_leakage
from feature_engineering.pipeline import FeatureEngineeringPipeline


# CatBoost hyperparameters (PAPER_DRAFT.md Section 3.4)
CATBOOST_PARAMS = {
    'iterations': 500,
    'depth': 10,
    'learning_rate': 0.1,
    'loss_function': 'MultiClass',
    'classes_count': 3,
    'eval_metric': 'Accuracy',
    'verbose': 100,
    'early_stopping_rounds': 50,
    'task_type': 'CPU',  # 'GPU' if available
    'bootstrap_type': 'Bayesian',
}


def train_model(
    data_file: str,
    k: int = 100,
    threshold_pct: float = 0.0001,
    random_seed: int = 42,
    output_dir: str = "models"
):
    """
    Train CatBoost model on LOB data.

    Args:
        data_file: Path to JSONL file
        k: Prediction horizon
        threshold_pct: Threshold for stationary class
        random_seed: Random seed for reproducibility
        output_dir: Directory to save trained model

    Returns:
        Dict with training results
    """
    print("=" * 70)
    print("LOB Mid-Price Prediction - CatBoost Training")
    print("=" * 70)
    print(f"Data file:     {data_file}")
    print(f"Horizon:       k={k}")
    print(f"Threshold:     {threshold_pct * 100:.4f}%")
    print(f"Random seed:   {random_seed}")
    print()

    # Step 1: Load data
    print("Step 1: Loading LOB snapshots...")
    snapshots = load_jsonl_file(data_file)
    print(f"  Loaded {len(snapshots)} snapshots")
    print()

    # Step 2: Convert to features
    print("Step 2: Converting to features (78 features)...")
    pipeline = FeatureEngineeringPipeline(buffer_size=5)
    features, timestamps, stock_codes = snapshots_to_features(snapshots, pipeline)
    print(f"  Feature matrix shape: {features.shape}")
    print()

    # Step 3: Create DataFrame
    print("Step 3: Creating DataFrame...")
    df = create_dataframe(features, timestamps, stock_codes)
    print(f"  DataFrame shape: {df.shape}")
    print()

    # Step 4: Generate labels
    print(f"Step 4: Generating labels (k={k})...")
    labels = generate_labels_from_dataframe(
        df, k=k, threshold_pct=threshold_pct, group_by_stock=True
    )
    df['label'] = labels
    print_label_distribution(labels.values, title="Label Distribution")

    # Step 5: Remove samples with NaN labels
    print("Step 5: Removing NaN labels...")
    df_clean = df.dropna(subset=['label']).reset_index(drop=True)
    print(f"  Clean samples: {len(df_clean)}")
    print()

    # Step 6: Temporal train/val/test split
    print("Step 6: Temporal train/val/test split (70/15/15)...")
    train_df, val_df, test_df = temporal_train_val_test_split(
        df_clean,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        by_stock=True
    )

    # Verify no leakage
    verify_no_leakage(train_df, val_df, test_df)

    # Step 7: Prepare features and labels
    print("Step 7: Preparing training data...")
    feature_cols = pipeline.get_feature_names()

    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values.astype(int)

    X_val = val_df[feature_cols].values
    y_val = val_df['label'].values.astype(int)

    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values.astype(int)

    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_val shape:   {X_val.shape}")
    print(f"  X_test shape:  {X_test.shape}")
    print()

    # Step 8: Train CatBoost
    print("Step 8: Training CatBoost...")
    print(f"  Hyperparameters: {CATBOOST_PARAMS}")
    print()

    model = CatBoostClassifier(
        **CATBOOST_PARAMS,
        random_seed=random_seed
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=100
    )
    print()

    # Step 9: Evaluate on all splits
    print("Step 9: Evaluating model on all splits...")

    # Training set
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    # Validation set
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    # Test set
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"\n📊 Results:")
    print(f"  Training Accuracy:   {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"  Test Accuracy:       {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print()

    print("Test Set Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Down', 'Stay', 'Up']))

    print("Test Set Confusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    print()

    # Step 10: Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"catboost_seed_{random_seed}.cbm")
    model.save_model(model_path)
    print(f"✅ Model saved: {model_path}")
    print()

    # Step 11: Feature importance
    print("Top 10 Most Important Features:")
    feature_importance = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.4f}")
    print()

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'data_file': data_file,
        'n_samples': len(df_clean),
        'n_train': len(train_df),
        'n_val': len(val_df),
        'n_test': len(test_df),
        'n_features': X_train.shape[1],
        'k': k,
        'threshold_pct': threshold_pct,
        'random_seed': random_seed,
        'train_accuracy': float(train_accuracy),
        'val_accuracy': float(val_accuracy),
        'test_accuracy': float(test_accuracy),
        'hyperparameters': CATBOOST_PARAMS,
        'model_path': model_path
    }

    results_path = os.path.join(output_dir, f"results_seed_{random_seed}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved: {results_path}")
    print()

    print("=" * 70)
    print("✅ Training completed successfully!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CatBoost on LOB data')
    parser.add_argument('--data-file', type=str, default='data/sample/sample_005930.jsonl',
                       help='Path to JSONL data file')
    parser.add_argument('--k', type=int, default=10,
                       help='Prediction horizon (default: 10 for testing, 100 for production)')
    parser.add_argument('--threshold', type=float, default=0.0001,
                       help='Threshold for stationary class (default: 0.0001 = 0.01%%)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory for models (default: models)')

    args = parser.parse_args()

    train_model(
        data_file=args.data_file,
        k=args.k,
        threshold_pct=args.threshold,
        random_seed=args.seed,
        output_dir=args.output_dir
    )
