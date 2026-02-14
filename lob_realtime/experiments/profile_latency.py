"""
Latency benchmark for the real-time feature engine.

Measures per-component latency across N events and produces a report.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lob_realtime.core.state_manager import LOBSnapshot
from lob_realtime.core.feature_engine import FeatureEngine
from lob_realtime.core.profiler import Profiler


def load_snapshots(path: str, max_n: int | None = None) -> list[dict]:
    snapshots = []
    with open(path) as f:
        for line in f:
            snapshots.append(json.loads(line))
            if max_n and len(snapshots) >= max_n:
                break
    return snapshots


def run_latency_benchmark(
    data_path: str,
    n_events: int = 100_000,
    warmup: int = 1000,
):
    """
    Run the feature engine with profiling and print report.
    """
    print(f"Loading data from {data_path}...")
    snapshots = load_snapshots(data_path, max_n=n_events + warmup)
    print(f"Loaded {len(snapshots)} snapshots")
    print()

    profiler = Profiler(enabled=True)
    engine = FeatureEngine(window_size=5, profiler=profiler)

    # Warm up (not profiled)
    warmup_actual = min(warmup, len(snapshots))
    for i in range(warmup_actual):
        snap = LOBSnapshot.from_dict(snapshots[i])
        engine.process_event(snap)
    engine.reset()
    profiler.reset()

    # Benchmark
    actual_n = min(n_events, len(snapshots) - warmup_actual)
    print(f"Benchmarking {actual_n} events (after {warmup_actual} warmup)...\n")

    total_profiler = Profiler(enabled=True)

    for i in range(warmup_actual, warmup_actual + actual_n):
        snap = LOBSnapshot.from_dict(snapshots[i])
        with total_profiler.measure('total_pipeline'):
            engine.process_event(snap)

    # Merge total_pipeline into profiler
    profiler._timings['total_pipeline'] = total_profiler._timings['total_pipeline']

    # Report
    print("=" * 75)
    print("LATENCY BENCHMARK REPORT")
    print("=" * 75)
    print(profiler.report_str())
    print()

    report = profiler.report()
    if 'total_pipeline' in report:
        mean_us = report['total_pipeline']['mean_us']
        throughput = 1_000_000.0 / mean_us if mean_us > 0 else float('inf')
        print(f"Throughput: {throughput:,.0f} events/sec")
        print(f"Mean total latency: {mean_us:.1f}μs")
        target = 200.0
        status = "PASS" if mean_us <= target else "FAIL"
        print(f"Target (<{target:.0f}μs): {status}")
    print()

    return report


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Profile real-time feature engine latency')
    parser.add_argument('--data', type=str, default='data/sample/sample_005930.jsonl',
                        help='Path to JSONL data file')
    parser.add_argument('--n-events', type=int, default=100_000,
                        help='Number of events to benchmark')
    parser.add_argument('--warmup', type=int, default=1000,
                        help='Warmup events (not measured)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save report to JSON file')
    args = parser.parse_args()

    report = run_latency_benchmark(args.data, args.n_events, args.warmup)

    if args.save:
        profiler = Profiler()
        profiler._timings = {}  # dummy
        import json as json_mod
        with open(args.save, 'w') as f:
            json_mod.dump(report, f, indent=2)
        print(f"Report saved to {args.save}")
