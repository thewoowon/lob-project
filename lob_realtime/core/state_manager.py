"""LOB state management with ring buffer for temporal features."""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

from ..config import N_LEVELS, WINDOW_SIZE


@dataclass(slots=True)
class LOBSnapshot:
    """Single LOB snapshot with 10-level bid/ask prices and volumes."""

    ask_prices: np.ndarray   # shape (10,)
    ask_volumes: np.ndarray  # shape (10,)
    bid_prices: np.ndarray   # shape (10,)
    bid_volumes: np.ndarray  # shape (10,)
    timestamp: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> LOBSnapshot:
        """Create from legacy dict format (feature_engineering compatibility)."""
        ask_prices = np.array(
            [d[f'ask_price_{i}'] for i in range(1, N_LEVELS + 1)], dtype=np.float64
        )
        ask_volumes = np.array(
            [d[f'ask_volume_{i}'] for i in range(1, N_LEVELS + 1)], dtype=np.float64
        )
        bid_prices = np.array(
            [d[f'bid_price_{i}'] for i in range(1, N_LEVELS + 1)], dtype=np.float64
        )
        bid_volumes = np.array(
            [d[f'bid_volume_{i}'] for i in range(1, N_LEVELS + 1)], dtype=np.float64
        )
        ts = d.get('timestamp', 0.0)
        if isinstance(ts, str):
            ts = 0.0
        return cls(
            ask_prices=ask_prices,
            ask_volumes=ask_volumes,
            bid_prices=bid_prices,
            bid_volumes=bid_volumes,
            timestamp=float(ts),
        )

    def to_raw_features(self) -> np.ndarray:
        """Return 40 raw features in the same order as feature_engineering/raw_features.py."""
        return np.concatenate([
            self.ask_prices,   # 10
            self.ask_volumes,  # 10
            self.bid_prices,   # 10
            self.bid_volumes,  # 10
        ])


class StateManager:
    """Manages current LOB state and a ring buffer of past snapshots."""

    def __init__(self, window_size: int = WINDOW_SIZE):
        self._window_size = window_size
        self._buffer: deque[LOBSnapshot] = deque(maxlen=window_size)
        self._prev: Optional[LOBSnapshot] = None
        self._event_count: int = 0

    # -- public API --

    def update(self, snapshot: LOBSnapshot) -> Optional[LOBSnapshot]:
        """Push a new snapshot; return the previous one (or None on first call)."""
        prev = self._prev
        self._prev = snapshot
        self._buffer.append(snapshot)
        self._event_count += 1
        return prev

    @property
    def previous(self) -> Optional[LOBSnapshot]:
        return self._prev

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def buffer_len(self) -> int:
        return len(self._buffer)

    def get_history(self, n: int) -> List[LOBSnapshot]:
        """Return up to *n* most recent snapshots (oldest first)."""
        start = max(0, len(self._buffer) - n)
        return list(self._buffer)[start:]

    def get_mid_prices(self, n: int) -> np.ndarray:
        """Return array of last n mid prices (oldest first)."""
        history = self.get_history(n)
        out = np.empty(len(history), dtype=np.float64)
        for i, snap in enumerate(history):
            out[i] = (snap.ask_prices[0] + snap.bid_prices[0]) * 0.5
        return out

    def reset(self) -> None:
        self._buffer.clear()
        self._prev = None
        self._event_count = 0
