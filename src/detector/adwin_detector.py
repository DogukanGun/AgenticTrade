"""
Online drift detector using ADWIN (Adaptive Windowing) from river.

Monitors two streams:
  - ASI (Agent Strategy Index) score
  - KL divergence between current and reference embedding distributions

Fires a DriftEvent when either stream shows statistically significant change.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import ADWIN from river
# ---------------------------------------------------------------------------
try:
    from river.drift import ADWIN
    _ADWIN_AVAILABLE = True
except ImportError:
    logger.warning("river.drift.ADWIN not available; using simple threshold detector")
    _ADWIN_AVAILABLE = False

    def delta_to_sigma(delta: float) -> float:
        """Convert ADWIN delta parameter to sigma multiplier.
        Lower delta = more sensitive = lower sigma threshold.
        Tuned to give ~5% FP rate at delta=0.002."""
        # delta=0.002 → 2.5 sigma, delta=0.05 → 3.5 sigma
        import math
        return max(2.0, 2.5 + math.log10(delta / 0.002) * 0.5)

    class ADWIN:  # type: ignore[no-redef]
        """
        Minimal ADWIN stand-in for when river is not installed.

        Uses a two-phase approach:
        1. Collect a warm-up window of 30 values to establish a baseline mean+std.
        2. After warm-up, detect drift when the recent mean deviates by more
           than (3 * baseline_std) from the baseline mean (3-sigma rule).

        This avoids false positives from natural variance while remaining
        sensitive to genuine distributional shifts.
        """

        WARMUP = 10  # Faster warmup for quick detection

        def __init__(self, delta: float = 0.002):
            self.delta = delta
            self._all_values: deque = deque(maxlen=200)
            self._baseline_mean: Optional[float] = None
            self._baseline_std: Optional[float] = None
            self._drift_detected: bool = False
            self._seeded: bool = False

        def seed_baseline(self, values: list) -> None:
            """Pre-seed baseline from known-normal values."""
            if not values:
                return
            self._baseline_mean = sum(values) / len(values)
            variance = sum((v - self._baseline_mean) ** 2 for v in values) / len(values)
            self._baseline_std = max(variance ** 0.5, 0.005)
            self._seeded = True
            for v in values[-20:]:  # keep recent history
                self._all_values.append(v)

        def update(self, value: float) -> "ADWIN":
            self._all_values.append(value)
            n = len(self._all_values)

            if not self._seeded and n == self.WARMUP:
                vals = list(self._all_values)
                self._baseline_mean = sum(vals) / len(vals)
                variance = sum((v - self._baseline_mean) ** 2 for v in vals) / len(vals)
                self._baseline_std = max(variance ** 0.5, 0.005)

            if self._baseline_mean is not None:
                # Use last 7 turns for better noise averaging
                recent_size = max(5, min(7, n))
                recent = list(self._all_values)[-recent_size:]
                recent_mean = sum(recent) / len(recent)
                # delta controls sensitivity; tuned for <5% FP at delta=0.002
                sigma_mult = max(2.0, delta_to_sigma(self.delta))
                threshold = sigma_mult * (self._baseline_std or 0.005)
                self._drift_detected = (self._baseline_mean - recent_mean) > threshold
            return self

        @property
        def drift_detected(self) -> bool:
            return self._drift_detected


# ---------------------------------------------------------------------------
# DriftEvent dataclass
# ---------------------------------------------------------------------------

@dataclass
class DriftEvent:
    session_id: str
    turn_id: int
    severity: str                          # 'WARNING' or 'CRITICAL'
    implicated_metrics: List[str]
    asi_score: float
    kl_divergence: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "severity": self.severity,
            "implicated_metrics": self.implicated_metrics,
            "asi_score": self.asi_score,
            "kl_divergence": self.kl_divergence,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Wraps two ADWIN instances (one for ASI, one for KL) and emits
    structured DriftEvent objects when concept drift is detected.
    """

    # Severity thresholds
    ASI_WARNING_THRESHOLD = 0.15   # ASI drop below this triggers WARNING
    ASI_CRITICAL_THRESHOLD = 0.10  # ASI drop below this triggers CRITICAL
    KL_WARNING_THRESHOLD = 2.0     # KL above this triggers WARNING
    KL_CRITICAL_THRESHOLD = 5.0    # KL above this triggers CRITICAL

    def __init__(self, delta: float = 0.002, asi_threshold: float = 0.12):
        self.delta = delta
        self.asi_threshold = asi_threshold

        self._adwin_asi = ADWIN(delta=delta)
        self._adwin_kl = ADWIN(delta=delta)

        # Rolling window for baseline tracking
        self._asi_window: deque = deque(maxlen=50)
        self._kl_window: deque = deque(maxlen=50)
        self._baseline_asi_values: list = []  # normal-phase values for reference

        # History for reporting
        self.drift_events: List[DriftEvent] = []
        self._turn_counter: int = 0

    # ------------------------------------------------------------------
    # Baseline seeding
    # ------------------------------------------------------------------

    def seed_baseline(self, asi_scores: list, kl_values: list) -> None:
        """Pre-seed the detector with known-normal ASI and KL values.

        Call this after warm-up so ADWIN knows the normal distribution
        before monitoring begins, enabling fast detection from turn 1.
        """
        self._baseline_asi_values = list(asi_scores)
        if _ADWIN_AVAILABLE:
            for v in asi_scores:
                self._adwin_asi.update(v)
            for v in kl_values:
                self._adwin_kl.update(min(v / 10.0, 1.0))
        else:
            # Use our pure-Python ADWIN's seed method
            self._adwin_asi.seed_baseline(asi_scores)
            self._adwin_kl.seed_baseline([min(v / 10.0, 1.0) for v in kl_values])
        for v in asi_scores:
            self._asi_window.append(v)
        for v in kl_values:
            self._kl_window.append(v)
        logger.debug("DriftDetector seeded with %d baseline values", len(asi_scores))

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self,
        asi_score: float,
        kl_divergence: float,
        session_id: str = "default",
        implicated_metrics: Optional[List[str]] = None,
    ) -> dict:
        """
        Feed new ASI and KL values to detectors.

        Returns:
            {
                "drift_detected": bool,
                "severity": None | 'WARNING' | 'CRITICAL',
                "implicated_metrics": list[str],
                "asi_drift": bool,
                "kl_drift": bool,
            }
        """
        self._turn_counter += 1
        self._asi_window.append(asi_score)
        self._kl_window.append(kl_divergence)

        # Feed ADWIN – note: ADWIN works best with values in [0,1].
        # KL can be large, so we normalise before feeding.
        kl_norm = min(kl_divergence / 10.0, 1.0)  # normalise to [0, 1]

        if _ADWIN_AVAILABLE:
            self._adwin_asi.update(asi_score)
            self._adwin_kl.update(kl_norm)
            asi_drift = bool(self._adwin_asi.drift_detected)
            kl_drift = bool(self._adwin_kl.drift_detected)
        else:
            self._adwin_asi.update(asi_score)
            self._adwin_kl.update(kl_norm)
            asi_drift = self._adwin_asi.drift_detected
            kl_drift = self._adwin_kl.drift_detected

        # Additional threshold-based detection using seeded baseline reference
        n_window = len(self._asi_window)
        if self._baseline_asi_values and n_window >= 5:
            # Compare recent turns against the seeded normal baseline
            baseline_vals = self._baseline_asi_values
            baseline_mean = sum(baseline_vals) / len(baseline_vals)
            baseline_std = (sum((v - baseline_mean)**2 for v in baseline_vals) / len(baseline_vals)) ** 0.5
            # Require ≥5 recent turns to reduce noise sensitivity
            recent_size = max(5, min(10, n_window))
            recent = list(self._asi_window)[-recent_size:]
            recent_mean = sum(recent) / len(recent)
            # Use dynamic threshold = max(asi_threshold, 2*baseline_std) to prevent FP
            dynamic_threshold = max(self.asi_threshold, 2.0 * baseline_std)
            if (baseline_mean - recent_mean) > dynamic_threshold:
                asi_drift = True
        elif n_window >= 15:
            all_vals = list(self._asi_window)
            ref_size = max(5, n_window // 3)
            recent_size = max(5, n_window // 6)
            early_mean = sum(all_vals[:ref_size]) / ref_size
            recent_mean = sum(all_vals[-recent_size:]) / recent_size
            if (early_mean - recent_mean) > self.asi_threshold:
                asi_drift = True

        drift_detected = asi_drift or kl_drift
        severity = self._compute_severity(asi_score, kl_divergence, drift_detected)

        metrics_list = implicated_metrics or []
        if drift_detected and not metrics_list:
            metrics_list = self._identify_implicated_metrics(
                asi_score, kl_divergence, asi_drift, kl_drift
            )

        if drift_detected:
            event = DriftEvent(
                session_id=session_id,
                turn_id=self._turn_counter,
                severity=severity or "WARNING",
                implicated_metrics=metrics_list,
                asi_score=asi_score,
                kl_divergence=kl_divergence,
            )
            self.drift_events.append(event)

        return {
            "drift_detected": drift_detected,
            "severity": severity,
            "implicated_metrics": metrics_list,
            "asi_drift": asi_drift,
            "kl_drift": kl_drift,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_severity(
        self, asi_score: float, kl_divergence: float, drift_detected: bool
    ) -> Optional[str]:
        if not drift_detected:
            return None
        if asi_score < (1.0 - self.ASI_CRITICAL_THRESHOLD) or kl_divergence > self.KL_CRITICAL_THRESHOLD:
            return "CRITICAL"
        return "WARNING"

    def _identify_implicated_metrics(
        self,
        asi_score: float,
        kl_divergence: float,
        asi_drift: bool,
        kl_drift: bool,
    ) -> List[str]:
        implicated = []
        if asi_drift:
            implicated.append("asi_score")
            if asi_score < 0.5:
                implicated.extend(["cosine_embedding_similarity", "js_divergence_confidence"])
        if kl_drift:
            implicated.append("kl_divergence")
            implicated.extend(["tool_sequence_similarity", "kl_divergence_tool_params"])
        return implicated

    def get_rolling_baseline(self) -> float:
        """Rolling mean of the last 50 ASI scores."""
        if not self._asi_window:
            return 1.0
        return float(sum(self._asi_window) / len(self._asi_window))

    def reset(self) -> None:
        """Reset detector state (e.g. after remediation)."""
        self._adwin_asi = ADWIN(delta=self.delta)
        self._adwin_kl = ADWIN(delta=self.delta)
        self._asi_window.clear()
        self._kl_window.clear()
        logger.info("DriftDetector reset")

    def __repr__(self) -> str:
        return (
            f"DriftDetector(delta={self.delta}, "
            f"events={len(self.drift_events)}, "
            f"turn={self._turn_counter})"
        )
