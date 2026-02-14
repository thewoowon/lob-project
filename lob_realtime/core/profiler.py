"""Lightweight latency profiler with nanosecond resolution."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Generator

import numpy as np


class Profiler:
    """Measures wall-clock latency per named section."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._timings: dict[str, list[int]] = defaultdict(list)  # nanoseconds

    @contextmanager
    def measure(self, name: str) -> Generator[None, None, None]:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter_ns()
        yield
        elapsed = time.perf_counter_ns() - t0
        self._timings[name].append(elapsed)

    def report(self) -> dict[str, dict[str, float]]:
        """
        Return summary statistics per section.

        Returns dict of {name: {mean_us, std_us, p50_us, p99_us, count}}.
        """
        out: dict[str, dict[str, float]] = {}
        for name, ns_list in self._timings.items():
            arr = np.array(ns_list, dtype=np.float64) / 1000.0  # → μs
            out[name] = {
                'mean_us': float(np.mean(arr)),
                'std_us': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                'p50_us': float(np.median(arr)),
                'p99_us': float(np.percentile(arr, 99)),
                'min_us': float(np.min(arr)),
                'max_us': float(np.max(arr)),
                'count': len(arr),
            }
        return out

    def report_str(self) -> str:
        lines = [
            f"{'Component':<25s} {'Mean':>8s} {'Std':>8s} {'P50':>8s} {'P99':>8s} {'Count':>8s}",
            '-' * 75,
        ]
        for name, s in sorted(self.report().items()):
            lines.append(
                f"{name:<25s} {s['mean_us']:>7.1f}μs {s['std_us']:>7.1f}μs "
                f"{s['p50_us']:>7.1f}μs {s['p99_us']:>7.1f}μs {s['count']:>8d}"
            )
        return '\n'.join(lines)

    def reset(self) -> None:
        self._timings.clear()

    def save(self, path: str) -> None:
        import json
        with open(path, 'w') as f:
            json.dump(self.report(), f, indent=2)
