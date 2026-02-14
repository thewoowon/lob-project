"""
Validate that the real-time feature engine produces identical output
to the batch FeatureEngineeringPipeline.
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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


def validate_parity(
    snapshots: list[dict],
    tolerance: float = 1e-6,
    verbose: bool = True,
) -> dict:
    """
    Run both pipelines on the same data and compare outputs.

    Returns:
        {
            'match': bool,
            'n_events': int,
            'max_abs_diff': float,
            'mean_abs_diff': float,
            'mismatched_features': list[str],
            'first_mismatch_event': int | None,
        }
    """
    batch_pipe = FeatureEngineeringPipeline(buffer_size=5)
    rt_engine = FeatureEngine(window_size=5)

    batch_names = batch_pipe.get_feature_names()
    rt_names = rt_engine.get_feature_names()

    # verify name ordering
    assert batch_names == rt_names, (
        f"Feature name mismatch!\n"
        f"  Batch: {batch_names}\n"
        f"  RT:    {rt_names}"
    )

    max_abs_diff = 0.0
    sum_abs_diff = 0.0
    n_compared = 0
    mismatched_features: set[str] = set()
    first_mismatch_event: int | None = None

    for i, snap_dict in enumerate(snapshots):
        batch_vec = batch_pipe.process_snapshot(snap_dict)
        rt_snap = LOBSnapshot.from_dict(snap_dict)
        rt_vec = rt_engine.process_event(rt_snap)

        diff = np.abs(batch_vec - rt_vec)
        event_max = diff.max()
        max_abs_diff = max(max_abs_diff, event_max)
        sum_abs_diff += diff.sum()
        n_compared += len(diff)

        if event_max > tolerance:
            if first_mismatch_event is None:
                first_mismatch_event = i
            for j in range(len(diff)):
                if diff[j] > tolerance:
                    mismatched_features.add(batch_names[j])
                    if verbose and i == first_mismatch_event:
                        print(
                            f"  Event {i}, feature '{batch_names[j]}' (idx {j}): "
                            f"batch={batch_vec[j]:.10f}  rt={rt_vec[j]:.10f}  "
                            f"diff={diff[j]:.2e}"
                        )

    mean_abs_diff = sum_abs_diff / n_compared if n_compared > 0 else 0.0
    match = max_abs_diff <= tolerance

    result = {
        'match': match,
        'n_events': len(snapshots),
        'max_abs_diff': float(max_abs_diff),
        'mean_abs_diff': float(mean_abs_diff),
        'mismatched_features': sorted(mismatched_features),
        'first_mismatch_event': first_mismatch_event,
    }

    if verbose:
        print()
        print("=" * 60)
        print("PARITY VALIDATION RESULT")
        print("=" * 60)
        status = "PASS" if match else "FAIL"
        print(f"Status:              {status}")
        print(f"Events compared:     {len(snapshots)}")
        print(f"Max absolute diff:   {max_abs_diff:.2e}")
        print(f"Mean absolute diff:  {mean_abs_diff:.2e}")
        print(f"Tolerance:           {tolerance:.2e}")
        if mismatched_features:
            print(f"Mismatched features: {sorted(mismatched_features)}")
        print("=" * 60)

    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Validate batch vs realtime parity')
    parser.add_argument('--data', type=str, default='data/sample/sample_005930.jsonl',
                        help='Path to JSONL data file')
    parser.add_argument('--max-events', type=int, default=1000,
                        help='Max events to compare')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                        help='Tolerance for floating point comparison')
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    snapshots = load_snapshots(args.data, max_n=args.max_events)
    print(f"Loaded {len(snapshots)} snapshots\n")

    result = validate_parity(snapshots, tolerance=args.tolerance)

    if result['match']:
        print("\nParity validation PASSED!")
    else:
        print("\nParity validation FAILED!")
        sys.exit(1)
