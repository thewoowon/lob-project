"""
Top-K Feature Selection Experiment.

Measures accuracy vs latency tradeoff when using only the top-K most
important features from the trained CatBoost model.

Outputs:
  - Table of K vs accuracy vs latency
  - Recommended K for <200μs constraint
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from catboost import CatBoostClassifier
from feature_engineering.pipeline import FeatureEngineeringPipeline
from lob_realtime.core.state_manager import LOBSnapshot
from lob_realtime.core.feature_engine import FeatureEngine


def load_snapshots(path: str, max_n: int | None = None) -> list[dict]:
    """Load LOB snapshots from JSONL file."""
    snapshots = []
    with open(path) as f:
        for line in f:
            snapshots.append(json.loads(line))
            if max_n and len(snapshots) >= max_n:
                break
    return snapshots


def get_feature_importance_ranking(model_path: str) -> list[tuple[int, str, float]]:
    """
    Load model and return feature indices sorted by importance.

    Returns:
        List of (index, feature_name, importance) sorted descending.
    """
    model = CatBoostClassifier()
    model.load_model(model_path)
    importance = model.get_feature_importance()

    pipeline = FeatureEngineeringPipeline()
    feature_names = pipeline.get_feature_names()

    sorted_idx = importance.argsort()[::-1]
    return [
        (int(idx), feature_names[idx], float(importance[idx]))
        for idx in sorted_idx
    ]


def run_feature_selection_experiment(
    data_path: str,
    model_path: str,
    k_values: list[int] | None = None,
    max_events: int = 1000,
    verbose: bool = True,
) -> list[dict]:
    """
    Run Top-K feature selection experiment.

    For each K, compute all 78 features, select top-K, and measure:
      - CatBoost prediction accuracy (vs full model prediction)
      - Feature extraction latency
      - Prediction latency

    Args:
        data_path: Path to JSONL data file.
        model_path: Path to trained CatBoost .cbm model.
        k_values: List of K values to test. Default: [10, 20, 30, 40, 50, 60, 70, 78]
        max_events: Number of events to evaluate.
        verbose: Print progress.

    Returns:
        List of result dicts, one per K value.
    """
    if k_values is None:
        k_values = [10, 20, 30, 40, 50, 60, 70, 78]

    # Load data
    snapshots = load_snapshots(data_path, max_n=max_events)
    n_events = len(snapshots)

    if verbose:
        print(f"Loaded {n_events} snapshots from {data_path}")

    # Load model and get feature ranking
    ranking = get_feature_importance_ranking(model_path)
    model = CatBoostClassifier()
    model.load_model(model_path)

    if verbose:
        print(f"\nTop-10 features by importance:")
        for idx, name, imp in ranking[:10]:
            print(f"  [{idx:2d}] {name:30s}  {imp:.4f}")
        print()

    # Compute full feature vectors
    rt_engine = FeatureEngine(window_size=5)
    full_features = np.empty((n_events, 78), dtype=np.float64)
    for i, snap_dict in enumerate(snapshots):
        snap = LOBSnapshot.from_dict(snap_dict)
        full_features[i] = rt_engine.process_event(snap)

    # Full model predictions as ground truth
    full_predictions = model.predict(full_features).flatten()

    results = []

    for k in k_values:
        if k > 78:
            k = 78

        # Select top-K feature indices
        top_k_indices = np.array([idx for idx, _, _ in ranking[:k]])

        # Extract top-K features
        k_features = full_features[:, top_k_indices]

        # Measure feature extraction latency (selecting from pre-computed)
        # This measures the array slicing cost
        t_start = time.perf_counter_ns()
        for _ in range(n_events):
            _ = full_features[0, top_k_indices]
        t_select_ns = (time.perf_counter_ns() - t_start) / n_events

        # Measure full pipeline + selection latency
        rt_engine2 = FeatureEngine(window_size=5)
        t_start = time.perf_counter_ns()
        for i, snap_dict in enumerate(snapshots):
            snap = LOBSnapshot.from_dict(snap_dict)
            vec = rt_engine2.process_event(snap)
            _ = vec[top_k_indices]
        t_total_ns = (time.perf_counter_ns() - t_start) / n_events

        # Prediction with top-K features
        # We need to retrain or use the full model with masked features
        # For simplicity, we measure agreement with full model predictions
        # (A production system would retrain with top-K only)

        # Approximate: use full model but zero out non-selected features
        masked_features = np.zeros_like(full_features)
        masked_features[:, top_k_indices] = full_features[:, top_k_indices]
        masked_predictions = model.predict(masked_features).flatten()

        # Agreement with full model
        agreement = np.mean(masked_predictions == full_predictions)

        result = {
            'k': k,
            'agreement_with_full': float(agreement),
            'select_latency_us': t_select_ns / 1000.0,
            'total_pipeline_us': t_total_ns / 1000.0,
            'top_k_features': [name for _, name, _ in ranking[:k]],
        }
        results.append(result)

        if verbose:
            print(
                f"  K={k:3d}  |  agreement={agreement*100:.1f}%  |  "
                f"pipeline={t_total_ns/1000:.1f}μs  |  "
                f"select={t_select_ns/1000:.1f}μs"
            )

    return results


def print_report(results: list[dict]) -> None:
    """Print formatted experiment report."""
    print()
    print("=" * 75)
    print("FEATURE SELECTION EXPERIMENT RESULTS")
    print("=" * 75)
    print(f"{'K':>5s}  {'Agreement':>10s}  {'Pipeline':>12s}  {'Select':>10s}  {'<200μs':>8s}")
    print("-" * 75)

    best_k = None
    for r in results:
        meets_target = r['total_pipeline_us'] <= 200.0
        marker = "PASS" if meets_target else "FAIL"
        print(
            f"{r['k']:5d}  {r['agreement_with_full']*100:9.1f}%  "
            f"{r['total_pipeline_us']:10.1f}μs  "
            f"{r['select_latency_us']:8.1f}μs  "
            f"{marker:>8s}"
        )
        if meets_target and (best_k is None or r['agreement_with_full'] > results[best_k]['agreement_with_full']):
            best_k = len(results) - 1 if r == results[-1] else results.index(r)

    print("-" * 75)
    if best_k is not None:
        r = results[best_k]
        print(
            f"\nRecommended: K={r['k']} "
            f"(agreement={r['agreement_with_full']*100:.1f}%, "
            f"latency={r['total_pipeline_us']:.1f}μs)"
        )
    print("=" * 75)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Top-K feature selection experiment')
    parser.add_argument('--data', type=str, default='data/sample/sample_005930.jsonl',
                        help='Path to JSONL data file')
    parser.add_argument('--model', type=str, default='models/multi_9stocks/multi_catboost_seed_42.cbm',
                        help='Path to trained CatBoost model')
    parser.add_argument('--max-events', type=int, default=1000,
                        help='Number of events to evaluate')
    parser.add_argument('--save', type=str, default=None,
                        help='Save results to JSON file')
    args = parser.parse_args()

    print("Feature Selection Experiment")
    print("=" * 75)

    results = run_feature_selection_experiment(
        data_path=args.data,
        model_path=args.model,
        max_events=args.max_events,
    )

    print_report(results)

    if args.save:
        # Remove top_k_features list for clean JSON
        save_results = []
        for r in results:
            sr = {k: v for k, v in r.items() if k != 'top_k_features'}
            save_results.append(sr)
        with open(args.save, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to {args.save}")
