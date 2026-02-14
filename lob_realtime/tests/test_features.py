"""Unit tests for real-time LOB feature engine."""

import numpy as np
import pytest

from lob_realtime.core.state_manager import LOBSnapshot, StateManager
from lob_realtime.core.feature_engine import FeatureEngine
from lob_realtime.features.stateless import compute_stateless_features
from lob_realtime.features.stateful import StatefulFeatureEngine


def _make_snapshot(
    ask1=100.0, bid1=99.0, vol=1000.0, timestamp=0.0
) -> LOBSnapshot:
    """Helper to create a simple LOBSnapshot."""
    ap = np.array([ask1 + i * 0.1 for i in range(10)], dtype=np.float64)
    av = np.full(10, vol, dtype=np.float64)
    bp = np.array([bid1 - i * 0.1 for i in range(10)], dtype=np.float64)
    bv = np.full(10, vol, dtype=np.float64)
    return LOBSnapshot(
        ask_prices=ap, ask_volumes=av,
        bid_prices=bp, bid_volumes=bv,
        timestamp=timestamp,
    )


# ── StateManager tests ──────────────────────────────────────────────

class TestStateManager:
    def test_first_update_returns_none(self):
        sm = StateManager(window_size=5)
        snap = _make_snapshot()
        prev = sm.update(snap)
        assert prev is None

    def test_second_update_returns_previous(self):
        sm = StateManager(window_size=5)
        s1 = _make_snapshot(ask1=100.0)
        s2 = _make_snapshot(ask1=101.0)
        sm.update(s1)
        prev = sm.update(s2)
        assert prev is s1

    def test_buffer_maxlen(self):
        sm = StateManager(window_size=3)
        for i in range(5):
            sm.update(_make_snapshot(timestamp=float(i)))
        assert sm.buffer_len == 3
        assert sm.event_count == 5

    def test_reset(self):
        sm = StateManager()
        sm.update(_make_snapshot())
        sm.reset()
        assert sm.event_count == 0
        assert sm.buffer_len == 0
        assert sm.previous is None


# ── Stateless features tests ────────────────────────────────────────

class TestStatelessFeatures:
    def test_output_shape(self):
        snap = _make_snapshot()
        out = compute_stateless_features(snap)
        assert out.shape == (31,)

    def test_mid_price(self):
        snap = _make_snapshot(ask1=100.0, bid1=98.0)
        out = compute_stateless_features(snap)
        assert abs(out[0] - 99.0) < 1e-10  # mid_price

    def test_spread(self):
        snap = _make_snapshot(ask1=100.0, bid1=98.0)
        out = compute_stateless_features(snap)
        assert abs(out[2] - 2.0) < 1e-10  # spread_absolute

    def test_division_by_zero(self):
        """All volumes = 0 should not crash."""
        snap = LOBSnapshot(
            ask_prices=np.array([100.0 + i for i in range(10)], dtype=np.float64),
            ask_volumes=np.zeros(10, dtype=np.float64),
            bid_prices=np.array([99.0 - i for i in range(10)], dtype=np.float64),
            bid_volumes=np.zeros(10, dtype=np.float64),
        )
        out = compute_stateless_features(snap)
        assert out.shape == (31,)
        assert np.all(np.isfinite(out))

    def test_preallocated_buffer(self):
        snap = _make_snapshot()
        buf = np.empty(31, dtype=np.float64)
        out = compute_stateless_features(snap, out=buf)
        assert out is buf


# ── Stateful features tests ─────────────────────────────────────────

class TestStatefulFeatures:
    def test_first_event_zeros(self):
        sf = StatefulFeatureEngine(window_size=5)
        snap = _make_snapshot()
        out = sf.update(snap, prev=None)
        assert out.shape == (7,)
        # OFI features should be zero on first event
        assert abs(out[1]) < 1e-10  # ofi_bid
        assert abs(out[2]) < 1e-10  # ofi_ask

    def test_ofi_positive_bid(self):
        """When bid price stays and bid volume increases, ofi_bid > 0."""
        sf = StatefulFeatureEngine(window_size=5)
        prev = _make_snapshot(bid1=99.0, vol=1000.0)
        curr = _make_snapshot(bid1=99.0, vol=1500.0)  # volume up, price same
        out = sf.update(curr, prev)
        assert out[1] > 0  # ofi_bid = delta_vol = 500

    def test_volatility_warmup(self):
        """With only 1 event, volatility should be 0."""
        sf = StatefulFeatureEngine(window_size=5)
        snap = _make_snapshot()
        out = sf.update(snap, prev=None)
        assert abs(out[0]) < 1e-10  # mid_price_volatility

    def test_reset(self):
        sf = StatefulFeatureEngine(window_size=5)
        sf.update(_make_snapshot(), prev=None)
        sf.reset()
        # After reset, next call should behave like first event
        out = sf.update(_make_snapshot(), prev=None)
        assert abs(out[1]) < 1e-10


# ── FeatureEngine integration tests ─────────────────────────────────

class TestFeatureEngine:
    def test_output_shape(self):
        engine = FeatureEngine(window_size=5)
        snap = _make_snapshot()
        out = engine.process_event(snap)
        assert out.shape == (78,)

    def test_feature_names_count(self):
        engine = FeatureEngine()
        assert len(engine.get_feature_names()) == 78

    def test_raw_features_preserved(self):
        snap = _make_snapshot(ask1=105.0, bid1=103.0)
        engine = FeatureEngine()
        out = engine.process_event(snap)
        # First 10 should be ask prices
        assert abs(out[0] - 105.0) < 1e-10

    def test_multiple_events(self):
        engine = FeatureEngine(window_size=5)
        for i in range(20):
            snap = _make_snapshot(ask1=100.0 + i * 0.01, timestamp=float(i))
            out = engine.process_event(snap)
        assert out.shape == (78,)
        assert np.all(np.isfinite(out))

    def test_reset(self):
        engine = FeatureEngine(window_size=5)
        for i in range(10):
            engine.process_event(_make_snapshot(timestamp=float(i)))
        engine.reset()
        # After reset, should produce same as fresh engine
        fresh = FeatureEngine(window_size=5)
        snap = _make_snapshot()
        out1 = engine.process_event(snap)
        out2 = fresh.process_event(snap)
        np.testing.assert_array_almost_equal(out1, out2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
