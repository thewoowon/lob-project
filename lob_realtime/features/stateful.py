"""
Stateful LOB features — require previous snapshot or rolling window.

Covers:
  - Mid-price volatility (1, rolling std over window)
  - OFI features (6)

Total: 7 stateful features
"""

import numpy as np
from collections import deque
from typing import Optional

from ..config import EPSILON, WINDOW_SIZE
from ..core.state_manager import LOBSnapshot


STATEFUL_FEATURE_NAMES: list[str] = [
    # Price (1 stateful)
    'mid_price_volatility',
    # OFI (6)
    'ofi_bid',
    'ofi_ask',
    'ofi_net',
    'ofi_ratio',
    'ofi_cumulative',
    'ofi_volatility',
]

assert len(STATEFUL_FEATURE_NAMES) == 7


class StatefulFeatureEngine:
    """Incrementally computes features that depend on history."""

    def __init__(self, window_size: int = WINDOW_SIZE):
        self._window = window_size

        # -- mid-price rolling stats --
        self._mid_buf: deque[float] = deque(maxlen=window_size)
        self._mid_sum: float = 0.0
        self._mid_sq_sum: float = 0.0

        # -- OFI rolling stats --
        self._ofi_buf: deque[float] = deque(maxlen=window_size)
        self._ofi_sum: float = 0.0
        self._ofi_sq_sum: float = 0.0

    def update(
        self,
        curr: LOBSnapshot,
        prev: Optional[LOBSnapshot],
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute 7 stateful features and advance internal state.

        Args:
            curr: Current snapshot.
            prev: Previous snapshot (None on first event).
            out: Optional pre-allocated array of shape (7,).

        Returns:
            np.ndarray of shape (7,).
        """
        if out is None:
            out = np.empty(7, dtype=np.float64)

        mid = (curr.ask_prices[0] + curr.bid_prices[0]) * 0.5

        # ── mid-price volatility (rolling std) ───────────────────
        self._push_mid(mid)
        n = len(self._mid_buf)
        if n < 2:
            out[0] = 0.0
        else:
            mean = self._mid_sum / n
            var = (self._mid_sq_sum / n) - mean * mean
            # Bessel correction
            var = var * n / (n - 1) if var > 0.0 else 0.0
            out[0] = np.sqrt(var)

        # ── OFI ──────────────────────────────────────────────────
        if prev is None:
            out[1:7] = 0.0
            return out

        # deltas
        d_bp = curr.bid_prices[0] - prev.bid_prices[0]
        d_bv = curr.bid_volumes[0] - prev.bid_volumes[0]
        d_ap = curr.ask_prices[0] - prev.ask_prices[0]
        d_av = curr.ask_volumes[0] - prev.ask_volumes[0]

        ofi_bid = d_bv if d_bp >= 0.0 else 0.0
        ofi_ask = d_av if d_ap <= 0.0 else 0.0
        ofi_net = ofi_bid - ofi_ask

        self._push_ofi(ofi_net)

        out[1] = ofi_bid
        out[2] = ofi_ask
        out[3] = ofi_net
        out[4] = ofi_bid / (abs(ofi_ask) + EPSILON)        # ofi_ratio

        # cumulative OFI (sum over window)
        out[5] = self._ofi_sum                                # ofi_cumulative

        # OFI volatility (rolling std)
        n_ofi = len(self._ofi_buf)
        if n_ofi < 2:
            out[6] = 0.0
        else:
            mean_ofi = self._ofi_sum / n_ofi
            var_ofi = (self._ofi_sq_sum / n_ofi) - mean_ofi * mean_ofi
            var_ofi = var_ofi * n_ofi / (n_ofi - 1) if var_ofi > 0.0 else 0.0
            out[6] = np.sqrt(var_ofi)

        return out

    # ── incremental helpers ──────────────────────────────────────

    def _push_mid(self, val: float) -> None:
        if len(self._mid_buf) == self._mid_buf.maxlen:
            old = self._mid_buf[0]
            self._mid_sum -= old
            self._mid_sq_sum -= old * old
        self._mid_buf.append(val)
        self._mid_sum += val
        self._mid_sq_sum += val * val

    def _push_ofi(self, val: float) -> None:
        if len(self._ofi_buf) == self._ofi_buf.maxlen:
            old = self._ofi_buf[0]
            self._ofi_sum -= old
            self._ofi_sq_sum -= old * old
        self._ofi_buf.append(val)
        self._ofi_sum += val
        self._ofi_sq_sum += val * val

    def reset(self) -> None:
        self._mid_buf.clear()
        self._mid_sum = 0.0
        self._mid_sq_sum = 0.0
        self._ofi_buf.clear()
        self._ofi_sum = 0.0
        self._ofi_sq_sum = 0.0
