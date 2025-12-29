"""
Evaluation module for LOB prediction models

Implements various metrics:
- Prediction metrics: accuracy, F1, MCC, precision, recall
- Signal quality: SNR, autocorrelation
- Computational: training time, inference latency, memory usage
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Any, Optional
import time


class ModelEvaluator:
    """Evaluate model performance"""

    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute all evaluation metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

        # Multi-class F1, precision, recall
        avg_method = 'macro' if self.num_classes > 2 else 'binary'

        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        metrics['precision_macro'] = precision_score(
            y_true, y_pred, average='macro', zero_division=0
        )
        metrics['recall_macro'] = recall_score(
            y_true, y_pred, average='macro', zero_division=0
        )

        # Per-class metrics
        if self.num_classes > 2:
            for i in range(self.num_classes):
                class_mask = (y_true == i)
                if class_mask.sum() > 0:
                    metrics[f'f1_class_{i}'] = f1_score(
                        y_true == i, y_pred == i, zero_division=0
                    )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm

        # Additional statistics
        metrics['n_samples'] = len(y_true)
        metrics['class_distribution'] = np.bincount(y_true, minlength=self.num_classes)

        return metrics

    def compute_snr(self, original: np.ndarray, filtered: np.ndarray) -> float:
        """
        Compute Signal-to-Noise Ratio

        SNR = 10 * log10(var(signal) / var(noise))
        """
        signal = filtered
        noise = original - filtered

        var_signal = np.var(signal)
        var_noise = np.var(noise)

        if var_noise == 0:
            return float('inf')

        snr = 10 * np.log10(var_signal / var_noise)
        return snr

    def compute_autocorrelation(self, data: np.ndarray, max_lag: int = 50) -> np.ndarray:
        """Compute autocorrelation function"""
        data = data - np.mean(data)
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        return autocorr[:max_lag]

    def print_report(self, metrics: Dict[str, Any]):
        """Print evaluation report"""
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)

        print(f"\n{'Metric':<30} {'Value':<15}")
        print("-"*45)

        # Main metrics
        print(f"{'Accuracy':<30} {metrics['accuracy']:<15.4f}")
        print(f"{'F1 Score (Macro)':<30} {metrics['f1_macro']:<15.4f}")
        print(f"{'F1 Score (Weighted)':<30} {metrics['f1_weighted']:<15.4f}")
        print(f"{'MCC':<30} {metrics['mcc']:<15.4f}")
        print(f"{'Precision (Macro)':<30} {metrics['precision_macro']:<15.4f}")
        print(f"{'Recall (Macro)':<30} {metrics['recall_macro']:<15.4f}")

        # Per-class F1
        if self.num_classes > 2:
            print("\nPer-Class F1 Scores:")
            for i in range(self.num_classes):
                if f'f1_class_{i}' in metrics:
                    print(f"  Class {i}: {metrics[f'f1_class_{i}']:.4f}")

        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(cm)

        # Class distribution
        print("\nClass Distribution:")
        dist = metrics['class_distribution']
        for i, count in enumerate(dist):
            print(f"  Class {i}: {count} ({100*count/sum(dist):.1f}%)")

        print("="*60 + "\n")


class ExperimentTracker:
    """Track experiments and results"""

    def __init__(self):
        self.results = []

    def log_experiment(
        self,
        preprocess_method: str,
        model_name: str,
        depth: int,
        horizon_ms: int,
        metrics: Dict[str, Any],
        training_time: float = 0,
        inference_time: float = 0
    ):
        """Log experiment results"""
        result = {
            'preprocess': preprocess_method,
            'model': model_name,
            'depth': depth,
            'horizon_ms': horizon_ms,
            'accuracy': metrics.get('accuracy', 0),
            'f1_macro': metrics.get('f1_macro', 0),
            'f1_weighted': metrics.get('f1_weighted', 0),
            'mcc': metrics.get('mcc', 0),
            'precision': metrics.get('precision_macro', 0),
            'recall': metrics.get('recall_macro', 0),
            'training_time': training_time,
            'inference_time_ms': inference_time * 1000,
            'n_samples': metrics.get('n_samples', 0)
        }

        # Add per-class F1 if available
        for key, value in metrics.items():
            if key.startswith('f1_class_'):
                result[key] = value

        self.results.append(result)

    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as DataFrame"""
        return pd.DataFrame(self.results)

    def save_results(self, filepath: str):
        """Save results to CSV"""
        df = self.get_results_dataframe()
        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")

    def get_best_config(self, metric: str = 'accuracy') -> Dict[str, Any]:
        """Get best configuration by metric"""
        df = self.get_results_dataframe()
        if len(df) == 0:
            return {}

        best_idx = df[metric].idxmax()
        return df.iloc[best_idx].to_dict()

    def compare_preprocessing(self, model_name: str, depth: int, horizon_ms: int) -> pd.DataFrame:
        """Compare preprocessing methods for fixed model/depth/horizon"""
        df = self.get_results_dataframe()

        filtered = df[
            (df['model'] == model_name) &
            (df['depth'] == depth) &
            (df['horizon_ms'] == horizon_ms)
        ]

        return filtered[['preprocess', 'accuracy', 'f1_macro', 'mcc']].sort_values(
            'accuracy', ascending=False
        )

    def compare_models(self, preprocess: str, depth: int, horizon_ms: int) -> pd.DataFrame:
        """Compare models for fixed preprocessing/depth/horizon"""
        df = self.get_results_dataframe()

        filtered = df[
            (df['preprocess'] == preprocess) &
            (df['depth'] == depth) &
            (df['horizon_ms'] == horizon_ms)
        ]

        return filtered[['model', 'accuracy', 'f1_macro', 'training_time', 'inference_time_ms']].sort_values(
            'accuracy', ascending=False
        )


def compare_predictions(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    num_classes: int = 3
) -> pd.DataFrame:
    """
    Compare multiple models' predictions

    Args:
        y_true: True labels
        predictions_dict: Dict mapping model name to predictions
        num_classes: Number of classes

    Returns:
        DataFrame with comparison results
    """
    evaluator = ModelEvaluator(num_classes=num_classes)

    results = []
    for model_name, y_pred in predictions_dict.items():
        metrics = evaluator.evaluate(y_true, y_pred)

        results.append({
            'model': model_name,
            'accuracy': metrics['accuracy'],
            'f1_macro': metrics['f1_macro'],
            'f1_weighted': metrics['f1_weighted'],
            'mcc': metrics['mcc'],
            'precision': metrics['precision_macro'],
            'recall': metrics['recall_macro']
        })

    df = pd.DataFrame(results)
    return df.sort_values('accuracy', ascending=False)


def main():
    """Example usage"""
    print("=== Evaluation Module ===\n")

    # Generate synthetic predictions
    np.random.seed(42)
    n_samples = 500
    num_classes = 3

    y_true = np.random.randint(0, num_classes, n_samples)
    y_pred_1 = np.random.randint(0, num_classes, n_samples)
    y_pred_2 = y_true.copy()
    y_pred_2[np.random.rand(n_samples) < 0.2] = np.random.randint(0, num_classes, (y_pred_2 == y_pred_2).sum())

    print("Testing ModelEvaluator...")
    evaluator = ModelEvaluator(num_classes=num_classes)

    # Evaluate first model
    print("\nModel 1 (Random):")
    metrics_1 = evaluator.evaluate(y_true, y_pred_1)
    evaluator.print_report(metrics_1)

    # Evaluate second model
    print("\nModel 2 (80% accurate):")
    metrics_2 = evaluator.evaluate(y_true, y_pred_2)
    evaluator.print_report(metrics_2)

    # Test ExperimentTracker
    print("\n\nTesting ExperimentTracker...")
    tracker = ExperimentTracker()

    # Log some experiments
    for preprocess in ['raw', 'savgol', 'kalman']:
        for model in ['logistic', 'xgboost']:
            metrics = evaluator.evaluate(y_true, y_pred_2)
            tracker.log_experiment(
                preprocess_method=preprocess,
                model_name=model,
                depth=10,
                horizon_ms=1000,
                metrics=metrics,
                training_time=np.random.uniform(1, 10),
                inference_time=np.random.uniform(0.001, 0.01)
            )

    # Get results
    results_df = tracker.get_results_dataframe()
    print("\nLogged experiments:")
    print(results_df[['preprocess', 'model', 'accuracy', 'f1_macro']].head(10))

    # Best config
    best = tracker.get_best_config('accuracy')
    print(f"\nBest config (by accuracy): {best['preprocess']} + {best['model']}")

    # Compare preprocessing
    print("\nComparing preprocessing methods for XGBoost:")
    comparison = tracker.compare_preprocessing('xgboost', 10, 1000)
    print(comparison)

    print("\n✓ Evaluation module test complete!")


if __name__ == "__main__":
    main()
