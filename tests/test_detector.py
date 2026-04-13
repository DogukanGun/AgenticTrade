"""
Tests for the DriftDetector (ADWIN-based online drift detector).
"""

import pytest

from src.detector.adwin_detector import DriftDetector, DriftEvent


class TestDriftDetectorBasic:
    def test_no_drift_on_stable_stream(self):
        detector = DriftDetector(delta=0.002)
        fired = False
        for _ in range(100):
            result = detector.update(0.9, 0.1, session_id="test")
            if result["drift_detected"]:
                fired = True
                break
        # Stable high ASI stream should not fire
        assert not fired

    def test_drift_fires_on_sudden_drop(self):
        """Feed stable high values then sudden low values – ADWIN should fire."""
        detector = DriftDetector(delta=0.002, asi_threshold=0.15)
        fired = False

        # Stable phase: high ASI
        for _ in range(60):
            detector.update(0.95, 0.05, session_id="drift_test")

        # Drift phase: low ASI (should eventually fire)
        for i in range(60):
            result = detector.update(0.30, 8.0, session_id="drift_test")
            if result["drift_detected"]:
                fired = True
                break

        assert fired, "Detector did not fire on sustained low-ASI stream"

    def test_result_structure(self):
        detector = DriftDetector()
        result = detector.update(0.8, 0.5, session_id="s1")
        assert "drift_detected" in result
        assert "severity" in result
        assert "implicated_metrics" in result
        assert "asi_drift" in result
        assert "kl_drift" in result
        assert isinstance(result["drift_detected"], bool)
        assert isinstance(result["implicated_metrics"], list)

    def test_severity_none_when_no_drift(self):
        detector = DriftDetector()
        for _ in range(5):
            result = detector.update(0.95, 0.1)
        # Initial stable readings should not yield drift
        # (even if ADWIN fires on first reading due to no baseline, check type)
        if not result["drift_detected"]:
            assert result["severity"] is None

    def test_severity_critical_on_extreme_drop(self):
        detector = DriftDetector(asi_threshold=0.05)
        fired_critical = False
        for _ in range(50):
            detector.update(0.99, 0.01)
        for _ in range(50):
            result = detector.update(0.05, 20.0)
            if result["drift_detected"] and result["severity"] == "CRITICAL":
                fired_critical = True
                break
        assert fired_critical


class TestDriftDetectorRollingBaseline:
    def test_rolling_baseline_is_float(self):
        detector = DriftDetector()
        for v in [0.8, 0.85, 0.9]:
            detector.update(v, 0.1)
        rb = detector.get_rolling_baseline()
        assert isinstance(rb, float)
        assert 0.0 <= rb <= 1.0

    def test_rolling_baseline_tracks_mean(self):
        detector = DriftDetector()
        for _ in range(50):
            detector.update(0.75, 0.5)
        rb = detector.get_rolling_baseline()
        assert abs(rb - 0.75) < 0.05

    def test_empty_rolling_baseline(self):
        detector = DriftDetector()
        rb = detector.get_rolling_baseline()
        assert rb == pytest.approx(1.0)


class TestDriftEvent:
    def test_drift_event_to_dict(self):
        event = DriftEvent(
            session_id="sess1",
            turn_id=42,
            severity="WARNING",
            implicated_metrics=["asi_score", "kl_divergence"],
            asi_score=0.45,
            kl_divergence=3.2,
        )
        d = event.to_dict()
        assert d["session_id"] == "sess1"
        assert d["turn_id"] == 42
        assert d["severity"] == "WARNING"
        assert "asi_score" in d
        assert "kl_divergence" in d
        assert "timestamp" in d

    def test_events_are_recorded(self):
        detector = DriftDetector(delta=0.002, asi_threshold=0.15)
        # Drive drift
        for _ in range(60):
            detector.update(0.95, 0.05, session_id="s")
        for _ in range(60):
            detector.update(0.2, 10.0, session_id="s")
        # Should have at least one drift event recorded
        assert len(detector.drift_events) >= 1
        for event in detector.drift_events:
            assert isinstance(event, DriftEvent)


class TestDriftDetectorReset:
    def test_reset_clears_state(self):
        detector = DriftDetector()
        for _ in range(20):
            detector.update(0.5, 2.0)
        detector.reset()
        rb = detector.get_rolling_baseline()
        assert rb == pytest.approx(1.0)


class TestDriftDetectorEdgeCases:
    def test_update_with_extreme_kl(self):
        detector = DriftDetector()
        result = detector.update(0.1, 1000.0)
        assert isinstance(result["drift_detected"], bool)

    def test_update_with_perfect_asi(self):
        detector = DriftDetector()
        result = detector.update(1.0, 0.0)
        assert isinstance(result["drift_detected"], bool)

    def test_gradual_drift_eventually_detected(self):
        """Gradual drift over many turns should eventually trigger."""
        detector = DriftDetector(delta=0.002, asi_threshold=0.1)
        fired = False
        for i in range(200):
            # Gradually decreasing ASI
            asi = max(0.1, 0.95 - i * 0.005)
            result = detector.update(asi, float(i * 0.05))
            if result["drift_detected"]:
                fired = True
                break
        assert fired, "Gradual drift was not detected over 200 turns"
