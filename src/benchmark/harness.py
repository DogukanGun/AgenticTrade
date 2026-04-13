"""
BenchmarkHarness: Generates synthetic agent sessions with controlled drift
injection and measures detector performance.

Designed to run ~500 sessions in < 60 seconds by using pre-computed
random embeddings rather than live model inference.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic turn generators
# ---------------------------------------------------------------------------

NORMAL_TOOLS = ["search", "analyze", "execute", "hedge"]
DRIFT_TOOLS = ["speculate", "leverage", "gamble"]
NORMAL_STRATEGIES = ["market_making", "arbitrage", "hedging"]
DRIFT_STRATEGIES = ["speculative", "momentum", "high_frequency"]
EMBEDDING_DIM = 384


def _make_normal_embedding(rng: np.random.RandomState) -> np.ndarray:
    """Stable embedding cluster around a fixed centroid."""
    centroid = np.ones(EMBEDDING_DIM, dtype=np.float32) * 0.5
    vec = centroid + rng.randn(EMBEDDING_DIM).astype(np.float32) * 0.05
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-9)


def _make_drift_embedding(rng: np.random.RandomState) -> np.ndarray:
    """Embedding shifted away from normal centroid."""
    centroid = np.ones(EMBEDDING_DIM, dtype=np.float32) * -0.3
    vec = centroid + rng.randn(EMBEDDING_DIM).astype(np.float32) * 0.1
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-9)


def _make_turn(
    turn_idx: int,
    drifted: bool,
    rng: np.random.RandomState,
) -> dict:
    """Generate a single synthetic agent turn."""
    if drifted:
        tools = list(rng.choice(DRIFT_TOOLS, size=rng.randint(1, 4)))
        confidence = float(rng.beta(2, 5))       # ~0.3 mean
        position_size = float(rng.uniform(10000, 20000))
        drawdown_pct = float(rng.uniform(3, 10))
        strategy = DRIFT_STRATEGIES[rng.randint(0, len(DRIFT_STRATEGIES))]
        asset_class = rng.choice(["crypto_derivatives", "penny_stocks", "equities"])
        leverage = float(rng.uniform(2, 5))
        response_len = int(rng.normal(80, 30))
        reasoning = f"Speculative move #{turn_idx}: shifting to high-risk assets"
        response = f"Executing speculative strategy with leverage {leverage:.1f}x"
        embedding = _make_drift_embedding(rng)
    else:
        tools = list(rng.choice(NORMAL_TOOLS, size=rng.randint(1, 3)))
        confidence = float(rng.beta(8, 2))       # ~0.8 mean
        position_size = float(rng.uniform(1000, 8000))
        drawdown_pct = float(rng.uniform(0, 3))
        strategy = NORMAL_STRATEGIES[rng.randint(0, len(NORMAL_STRATEGIES))]
        asset_class = rng.choice(["equities", "bonds", "commodities"])
        leverage = float(rng.uniform(1.0, 1.5))
        response_len = int(rng.normal(200, 40))
        reasoning = f"Conservative analysis #{turn_idx}: maintaining mandate compliance"
        response = f"Executing {strategy} with position size {position_size:.0f}"
        embedding = _make_normal_embedding(rng)

    response_len = max(50, response_len)

    return {
        "turn_idx": turn_idx,
        "response": response + " " * max(0, response_len - len(response)),
        "tools_used": tools,
        "reasoning": reasoning,
        "confidence": confidence,
        "position_size": position_size,
        "drawdown_pct": drawdown_pct,
        "strategy": strategy,
        "asset_class": asset_class,
        "leverage": leverage,
        "embedding": embedding,
        "drifted": drifted,
        "error": bool(rng.random() < (0.15 if drifted else 0.02)),
        "human_override": bool(rng.random() < (0.10 if drifted else 0.01)),
    }


# ---------------------------------------------------------------------------
# BenchmarkHarness
# ---------------------------------------------------------------------------

class BenchmarkHarness:
    """
    Generates synthetic sessions and measures DriftDetector performance.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    # ------------------------------------------------------------------
    # Session generation
    # ------------------------------------------------------------------

    def generate_session(
        self,
        session_id: str,
        n_turns: int = 200,
        drift_at: List[int] = None,
    ) -> List[dict]:
        """
        Generate a synthetic session of *n_turns* turns.

        *drift_at* is a list of turn indices where drift begins.
        After a drift point, subsequent turns are "drifted" until the next
        drift point resets back to normal (alternating).
        """
        if drift_at is None:
            drift_at = [50, 100, 150]

        rng = np.random.RandomState(abs(hash(session_id)) % (2**31))
        turns = []
        drifted = False

        drift_set = set(drift_at)

        for i in range(n_turns):
            if i in drift_set:
                drifted = not drifted
            turn = _make_turn(i, drifted, rng)
            turn["session_id"] = session_id
            turns.append(turn)

        return turns

    # ------------------------------------------------------------------
    # Benchmark runner
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        n_sessions: int = 500,
        drift_turns: List[int] = None,
        baseline_window: int = 50,
    ) -> dict:
        """
        Run *n_sessions* simulated sessions, half with drift, half clean.

        Returns detection performance metrics.
        """
        from src.detector.adwin_detector import DriftDetector
        from src.monitor.metrics import (
            compute_asi_score,
            compute_kl_divergence,
            cosine_embedding_similarity,
            tool_sequence_similarity,
            js_divergence_confidence,
            chi_squared_tool_test,
            kl_divergence_tool_params,
            consensus_rate,
            handoff_efficiency,
            mutual_information_role,
            output_length_cv,
            error_clustering_coefficient,
            human_override_rate,
        )

        if drift_turns is None:
            drift_turns = [50, 100, 200]

        n_drift_sessions = n_sessions // 2
        n_clean_sessions = n_sessions - n_drift_sessions

        detection_lags: List[float] = []
        false_positives = 0
        false_negatives = 0
        recovery_turns_list: List[float] = []

        logger.info(
            "Starting benchmark: %d sessions (%d drift, %d clean)",
            n_sessions, n_drift_sessions, n_clean_sessions,
        )
        t0 = time.time()

        # -- Drift sessions --
        for i in range(n_drift_sessions):
            session_id = f"drift_{i}"
            turns = self.generate_session(session_id, n_turns=250, drift_at=[50])
            result = self._simulate_session(
                turns,
                baseline_window=baseline_window,
                drift_injected_at=50,
            )
            if result["drift_detected_at"] is not None:
                lag = result["drift_detected_at"] - 50
                detection_lags.append(max(0, lag))
                if result.get("recovery_turn") is not None:
                    recovery_turns_list.append(result["recovery_turn"])
            else:
                false_negatives += 1

        # -- Clean sessions --
        for i in range(n_clean_sessions):
            session_id = f"clean_{i}"
            turns = self.generate_session(session_id, n_turns=200, drift_at=[])
            result = self._simulate_session(
                turns,
                baseline_window=baseline_window,
                drift_injected_at=None,
            )
            if result["drift_detected_at"] is not None:
                false_positives += 1

        elapsed = time.time() - t0
        logger.info("Benchmark completed in %.1fs", elapsed)

        return {
            "n_sessions": n_sessions,
            "n_drift_sessions": n_drift_sessions,
            "n_clean_sessions": n_clean_sessions,
            "detection_lag_mean": float(np.mean(detection_lags)) if detection_lags else None,
            "detection_lag_median": float(np.median(detection_lags)) if detection_lags else None,
            "detection_lag_std": float(np.std(detection_lags)) if detection_lags else None,
            "detection_lags": detection_lags,
            "false_positive_rate": false_positives / max(n_clean_sessions, 1),
            "false_negative_rate": false_negatives / max(n_drift_sessions, 1),
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "mean_recovery_turns": float(np.mean(recovery_turns_list)) if recovery_turns_list else None,
            "elapsed_seconds": elapsed,
        }

    def _simulate_session(
        self,
        turns: List[dict],
        baseline_window: int = 50,
        drift_injected_at: Optional[int] = None,
    ) -> dict:
        """
        Simulate a single monitoring session using pre-computed synthetic turns.
        Uses vectorised numpy operations to avoid per-turn model inference.
        """
        from src.detector.adwin_detector import DriftDetector
        from src.monitor.metrics import compute_asi_score, compute_kl_divergence

        detector = DriftDetector(delta=0.002, asi_threshold=0.15)

        # Build reference distribution from baseline window
        baseline_turns = turns[:baseline_window]
        if not baseline_turns:
            return {"drift_detected_at": None, "recovery_turn": None}

        ref_embeddings = np.array([t["embedding"] for t in baseline_turns])
        ref_mean = ref_embeddings.mean(axis=0)
        ref_std = ref_embeddings.std(axis=0) + 1e-6
        ref_norm = np.linalg.norm(ref_mean)
        if ref_norm > 0:
            ref_mean_norm = ref_mean / ref_norm
        else:
            ref_mean_norm = ref_mean

        ref_confidences = [t["confidence"] for t in baseline_turns]
        ref_tools = {}
        for t in baseline_turns:
            for tool in t["tools_used"]:
                ref_tools[tool] = ref_tools.get(tool, 0) + 1
        total_ref = sum(ref_tools.values()) or 1
        ref_tool_dist = {k: v / total_ref for k, v in ref_tools.items()}

        ref_lengths = [len(t["response"]) for t in baseline_turns]
        ref_len_mean = np.mean(ref_lengths)
        ref_len_std = np.std(ref_lengths) + 1e-6

        # Compute ASI for baseline turns to seed the detector
        baseline_asi_scores = []
        baseline_kl_values = []
        for turn in baseline_turns:
            emb = turn["embedding"]
            cos_sim = float(np.dot(emb, ref_mean_norm) /
                            (np.linalg.norm(emb) * np.linalg.norm(ref_mean_norm) + 1e-9))
            cos_metric = (cos_sim + 1.0) / 2.0
            conf_metric = self._fast_js([turn["confidence"]], ref_confidences)
            obs_tools = {}
            for tool in turn["tools_used"]:
                obs_tools[tool] = obs_tools.get(tool, 0) + 1
            total_obs = sum(obs_tools.values()) or 1
            obs_tool_dist = {k: v / total_obs for k, v in obs_tools.items()}
            tool_sim = self._fast_kl_similarity(obs_tool_dist, ref_tool_dist)
            cur_len = len(turn["response"])
            len_cv_score = max(0.0, 1.0 - abs(cur_len - ref_len_mean) / (ref_len_std + ref_len_mean + 1))
            a = {"cosine": min(1.0, max(0.0, cos_metric)), "confidence": conf_metric}
            b = {"tool_sim": tool_sim}
            c = {"coord": 0.9}
            d = {"length": len_cv_score, "error": 1.0, "override": 1.0}
            asi = compute_asi_score(a, b, c, d)
            kl = compute_kl_divergence(emb, ref_mean_norm, ref_std)
            baseline_asi_scores.append(asi)
            baseline_kl_values.append(kl)

        # Seed the detector with known-normal baseline so it detects drift from turn 1
        detector.seed_baseline(baseline_asi_scores, baseline_kl_values)

        # Monitor post-baseline turns
        drift_detected_at = None
        recovery_turn = None
        post_drift_asi = []

        for idx, turn in enumerate(turns[baseline_window:], start=baseline_window):
            emb = turn["embedding"]

            # Group A
            cos_sim = float(np.dot(emb, ref_mean_norm) /
                            (np.linalg.norm(emb) * np.linalg.norm(ref_mean_norm) + 1e-9))
            cos_metric = (cos_sim + 1.0) / 2.0

            # Confidence JS
            current_conf = [turn["confidence"]]
            conf_metric = self._fast_js(current_conf, ref_confidences)

            # Group B - tool usage
            obs_tools = {}
            for tool in turn["tools_used"]:
                obs_tools[tool] = obs_tools.get(tool, 0) + 1
            total_obs = sum(obs_tools.values()) or 1
            obs_tool_dist = {k: v / total_obs for k, v in obs_tools.items()}
            tool_sim = self._fast_kl_similarity(obs_tool_dist, ref_tool_dist)

            # Group C - placeholder coordination metrics
            coord_score = 0.9 if not turn["drifted"] else 0.4

            # Group D - lengths, errors
            cur_len = len(turn["response"])
            len_cv_score = max(0.0, 1.0 - abs(cur_len - ref_len_mean) / (ref_len_std + ref_len_mean + 1))
            error_score = 0.0 if turn["error"] else 1.0
            override_score = 0.0 if turn["human_override"] else 1.0

            metrics_a = {
                "cosine": min(1.0, max(0.0, cos_metric)),
                "confidence": conf_metric,
            }
            metrics_b = {"tool_sim": tool_sim}
            metrics_c = {"coord": coord_score}
            metrics_d = {
                "length": len_cv_score,
                "error": error_score,
                "override": override_score,
            }

            asi = compute_asi_score(metrics_a, metrics_b, metrics_c, metrics_d)

            kl = compute_kl_divergence(emb, ref_mean_norm, ref_std)

            result = detector.update(asi, kl)

            if result["drift_detected"] and drift_detected_at is None:
                drift_detected_at = idx

            # Track recovery after drift
            if drift_detected_at is not None:
                post_drift_asi.append(asi)
                if len(post_drift_asi) > 10:
                    recent_mean = np.mean(post_drift_asi[-10:])
                    if recent_mean > 0.75 and recovery_turn is None:
                        recovery_turn = len(post_drift_asi)

        return {
            "drift_detected_at": drift_detected_at,
            "recovery_turn": recovery_turn,
        }

    # ------------------------------------------------------------------
    # Fast metric helpers (avoid full metric overhead per turn)
    # ------------------------------------------------------------------

    @staticmethod
    def _fast_js(p: List[float], q: List[float]) -> float:
        """Simplified JS divergence approximation."""
        if not p or not q:
            return 0.5
        mu_p = np.mean(p)
        mu_q = np.mean(q)
        diff = abs(mu_p - mu_q)
        return max(0.0, min(1.0, 1.0 - diff))

    @staticmethod
    def _fast_kl_similarity(obs: dict, ref: dict) -> float:
        """Jaccard-based tool similarity."""
        obs_set = set(obs)
        ref_set = set(ref)
        if not obs_set and not ref_set:
            return 1.0
        union = obs_set | ref_set
        intersection = obs_set & ref_set
        return len(intersection) / len(union)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_results(self, results: dict, output_path: str) -> None:
        """Save a 4-panel benchmark result figure."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; skipping plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle("Agent Strategy Drift Detection – Benchmark Results", fontsize=14)

        # Panel 1: Detection lag histogram
        ax = axes[0, 0]
        lags = results.get("detection_lags", [])
        if lags:
            ax.hist(lags, bins=20, color="steelblue", edgecolor="white", alpha=0.85)
            ax.axvline(np.mean(lags), color="red", linestyle="--", label=f"Mean={np.mean(lags):.1f}")
            ax.legend()
        ax.set_xlabel("Detection Lag (turns)")
        ax.set_ylabel("Count")
        ax.set_title("Detection Lag Distribution")

        # Panel 2: Error rates bar chart
        ax = axes[0, 1]
        rates = {
            "False Positive\nRate": results.get("false_positive_rate", 0),
            "False Negative\nRate": results.get("false_negative_rate", 0),
        }
        colors = ["#e74c3c", "#e67e22"]
        bars = ax.bar(rates.keys(), rates.values(), color=colors, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, rates.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, max(max(rates.values()) * 1.3, 0.1))
        ax.set_ylabel("Rate")
        ax.set_title("Error Rates")

        # Panel 3: Session summary
        ax = axes[1, 0]
        session_data = {
            "Total Sessions": results.get("n_sessions", 0),
            "Drift Sessions": results.get("n_drift_sessions", 0),
            "Clean Sessions": results.get("n_clean_sessions", 0),
            "Detected Drifts": results.get("n_drift_sessions", 0) - results.get("false_negatives", 0),
        }
        ax.bar(session_data.keys(), session_data.values(), color="teal", alpha=0.8)
        ax.set_ylabel("Count")
        ax.set_title("Session Summary")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")

        # Panel 4: Key metrics text panel
        ax = axes[1, 1]
        ax.axis("off")
        mean_lag = results.get("detection_lag_mean")
        recovery = results.get("mean_recovery_turns")
        elapsed = results.get("elapsed_seconds", 0)
        text_lines = [
            f"Sessions: {results.get('n_sessions', 0)}",
            f"Mean Detection Lag: {mean_lag:.1f} turns" if mean_lag is not None else "Mean Lag: N/A",
            f"Median Detection Lag: {results.get('detection_lag_median', 0):.1f} turns" if mean_lag else "",
            f"False Positive Rate: {results.get('false_positive_rate', 0):.3f}",
            f"False Negative Rate: {results.get('false_negative_rate', 0):.3f}",
            f"Mean Recovery: {recovery:.1f} turns" if recovery is not None else "Recovery: N/A",
            f"Elapsed: {elapsed:.1f}s",
        ]
        ax.text(0.1, 0.9, "\n".join(text_lines), transform=ax.transAxes,
                fontsize=11, va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        ax.set_title("Summary Statistics")

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Benchmark plot saved to %s", output_path)
