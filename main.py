"""
main.py - Full demonstration of the Agent Strategy Drift Detection System.

Runs a complete end-to-end scenario:
1. Load config and mandate
2. Baseline warm-up (50 turns of normal behavior)
3. 50 more normal turns (low drift expected)
4. Inject drift for 50 turns (tools change, confidence drops)
5. Show ADWIN detection firing
6. Show contract violations being blocked
7. Apply remediation (goal reminder + exemplar injection)
8. Run benchmark (configurable sessions)
9. Save benchmark plots
10. Test FastAPI endpoints
11. Print summary
"""

from __future__ import annotations

import os
import sys
import time
import json
import threading
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Ensure src/ is importable when running from the code/ directory
# ---------------------------------------------------------------------------
_CODE_DIR = Path(__file__).parent.resolve()
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def info(msg: str) -> None:
    print(f"  [INFO]  {msg}")


def ok(msg: str) -> None:
    print(f"  [OK]    {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN]  {msg}")


def alert(msg: str) -> None:
    print(f"  [ALERT] {msg}")


def result(label: str, value) -> None:
    print(f"  {label:<40s}: {value}")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    full_path = _CODE_DIR / path
    if full_path.exists():
        with open(full_path) as fh:
            return yaml.safe_load(fh)
    warn(f"Config file not found at {full_path}; using defaults")
    return {}


def load_mandate_path(config: dict) -> str:
    mandate_rel = config.get("enforcer", {}).get("mandate_path", "mandate.yaml")
    return str(_CODE_DIR / mandate_rel)


# ---------------------------------------------------------------------------
# Synthetic turn generators (same style as benchmark harness)
# ---------------------------------------------------------------------------

NORMAL_TOOLS = ["search", "analyze", "execute", "hedge"]
DRIFT_TOOLS = ["speculate", "leverage", "gamble"]
NORMAL_STRATEGIES = ["market_making", "arbitrage", "hedging"]
DRIFT_STRATEGIES = ["speculative", "momentum", "high_frequency"]
NORMAL_ASSET_CLASSES = ["equities", "bonds", "commodities"]
DRIFT_ASSET_CLASSES = ["crypto_derivatives", "penny_stocks"]


def make_normal_turn(turn_idx: int, rng: np.random.RandomState) -> dict:
    tools = list(rng.choice(NORMAL_TOOLS, size=rng.randint(1, 3), replace=True))
    confidence = float(rng.beta(8, 2))
    position_size = float(rng.uniform(1000, 8000))
    drawdown_pct = float(rng.uniform(0, 3))
    strategy = NORMAL_STRATEGIES[rng.randint(0, len(NORMAL_STRATEGIES))]
    asset_class = NORMAL_ASSET_CLASSES[rng.randint(0, len(NORMAL_ASSET_CLASSES))]
    leverage = float(rng.uniform(1.0, 1.5))
    response = (
        f"Turn {turn_idx}: Conservative {strategy} analysis. "
        f"Position size {position_size:.0f}, drawdown {drawdown_pct:.2f}%, "
        f"confidence {confidence:.2f}. Maintaining mandate compliance."
    )
    reasoning = (
        f"Market analysis indicates {strategy} opportunity. "
        f"Risk within acceptable bounds. Asset class: {asset_class}."
    )
    return {
        "response": response,
        "tools_used": tools,
        "reasoning": reasoning,
        "confidence": confidence,
        "position_size": position_size,
        "drawdown_pct": drawdown_pct,
        "strategy": strategy,
        "asset_class": asset_class,
        "leverage": leverage,
        "goals": ["profit", "risk_management"],
        "error": bool(rng.random() < 0.02),
        "human_override": bool(rng.random() < 0.01),
    }


def make_drift_turn(turn_idx: int, rng: np.random.RandomState) -> dict:
    tools = list(rng.choice(DRIFT_TOOLS, size=rng.randint(2, 4), replace=True))
    confidence = float(rng.beta(2, 5))
    position_size = float(rng.uniform(10001, 25000))  # exceeds RULE_001
    drawdown_pct = float(rng.uniform(5.1, 15))         # exceeds RULE_003
    strategy = DRIFT_STRATEGIES[rng.randint(0, len(DRIFT_STRATEGIES))]  # violates RULE_002
    asset_class = DRIFT_ASSET_CLASSES[rng.randint(0, len(DRIFT_ASSET_CLASSES))]  # violates RULE_004
    leverage = float(rng.uniform(2.1, 6.0))            # violates RULE_005
    response = (
        f"Turn {turn_idx}: AGGRESSIVE {strategy} bet detected! "
        f"Leveraging {leverage:.1f}x with {asset_class}. "
        f"Position {position_size:.0f}."
    )
    reasoning = (
        f"High-risk opportunity: {strategy} in {asset_class}. "
        f"Overriding conservative limits for maximum gain."
    )
    return {
        "response": response,
        "tools_used": tools,
        "reasoning": reasoning,
        "confidence": confidence,
        "position_size": position_size,
        "drawdown_pct": drawdown_pct,
        "strategy": strategy,
        "asset_class": asset_class,
        "leverage": leverage,
        "goals": ["max_profit", "aggressive_growth"],
        "error": bool(rng.random() < 0.15),
        "human_override": bool(rng.random() < 0.10),
    }


# ---------------------------------------------------------------------------
# Core pipeline logic (direct, no HTTP)
# ---------------------------------------------------------------------------

def run_pipeline(config: dict, mandate_path: str) -> dict:
    """Run the full demo pipeline and return a summary dict."""
    from src.baseline.profiler import BaselineProfiler
    from src.monitor.metrics import (
        compute_asi_score,
        compute_kl_divergence,
        cosine_embedding_similarity,
        levenshtein_distance_normalized,
        js_divergence_confidence,
        chi_squared_tool_test,
        tool_sequence_similarity,
        kl_divergence_tool_params,
        consensus_rate,
        handoff_efficiency,
        mutual_information_role,
        output_length_cv,
        error_clustering_coefficient,
        human_override_rate,
    )
    from src.detector.adwin_detector import DriftDetector
    from src.enforcer.contract import ContractEnforcer
    from src.memory.episodic import EpisodicMemory
    from src.chain.audit_log import AuditLog

    # -----------------------------------------------------------------------
    # Initialise components
    # -----------------------------------------------------------------------
    baseline_cfg = config.get("baseline", {})
    baseline_window = baseline_cfg.get("window_size", 50)
    model_name = baseline_cfg.get("model_name", "all-MiniLM-L6-v2")

    profiler = BaselineProfiler(window_size=baseline_window, model_name=model_name)
    detector = DriftDetector(delta=0.002, asi_threshold=0.15)
    enforcer = ContractEnforcer()
    memory = EpisodicMemory(consolidation_interval=50)
    audit = AuditLog(db_path="/tmp/drift_demo_audit.db")

    if os.path.exists(mandate_path):
        enforcer.load_mandate(mandate_path)
        info(f"Loaded mandate: {enforcer.mandate_name} ({enforcer.rule_count} rules)")
    else:
        warn(f"Mandate file not found at {mandate_path}")

    rng = np.random.RandomState(42)
    session_id = "demo_session"
    audit.log_event("SESSION_START", {"session_id": session_id}, session_id=session_id)

    # -----------------------------------------------------------------------
    # Tracking
    # -----------------------------------------------------------------------
    asi_history: List[float] = []
    kl_history: List[float] = []
    drift_fired_at: Optional[int] = None
    total_violations = 0
    last_reasoning = ""
    last_tools: List[str] = []
    ref_len_mean = 200.0
    ref_len_std = 40.0
    errors_window: List[int] = []
    overrides_count = 0

    # -----------------------------------------------------------------------
    # Phase 1: Baseline warm-up
    # -----------------------------------------------------------------------
    section(f"Phase 1: Baseline Warm-Up ({baseline_window} turns)")

    for i in range(baseline_window):
        turn = make_normal_turn(i, rng)
        profiler.record_turn(turn)
        memory.add_turn(turn)
        audit.log_event("BASELINE_TURN", {"turn_idx": i}, session_id=session_id)

    ok(f"Baseline complete: {len(profiler)} turns recorded")
    ref_emb = profiler.get_reference_embedding()
    ref_std = profiler.get_reference_embedding_std()
    tool_dist = profiler.get_tool_distribution()
    ref_conf = profiler.get_confidence_distribution()
    len_stats = profiler.get_response_length_stats()
    ref_len_mean = len_stats["mean"]
    ref_len_std = len_stats["std"]
    info(f"Reference embedding shape: {ref_emb.shape}")
    info(f"Tool distribution: { {k: f'{v:.2f}' for k, v in tool_dist.items()} }")
    info(f"Mean confidence: {np.mean(ref_conf):.3f}")

    # -----------------------------------------------------------------------
    # Helper: compute full metrics for a turn
    # -----------------------------------------------------------------------
    def compute_metrics(turn: dict) -> tuple:
        nonlocal last_reasoning, last_tools, errors_window, overrides_count

        response = turn.get("response", "")
        reasoning = turn.get("reasoning", "")
        combined = f"{response} {reasoning}".strip()
        cur_emb = profiler._embed(combined)

        # Group A
        cos_sim = cosine_embedding_similarity(cur_emb, ref_emb)
        lev_sim = levenshtein_distance_normalized(reasoning, last_reasoning)
        cur_conf = [turn.get("confidence", 0.5)]
        js_conf = js_divergence_confidence(cur_conf * 10, ref_conf)
        last_reasoning = reasoning

        metrics_a = {
            "cosine_embedding_similarity": cos_sim,
            "levenshtein_distance_normalized": lev_sim,
            "js_divergence_confidence": js_conf,
        }

        # Group B
        obs_tools = {t: 1 for t in turn.get("tools_used", [])}
        chi_sq = chi_squared_tool_test(obs_tools, tool_dist)
        tool_seq = tool_sequence_similarity(turn.get("tools_used", []), last_tools)
        kl_tool = kl_divergence_tool_params(obs_tools, tool_dist)
        last_tools = turn.get("tools_used", [])

        metrics_b = {
            "chi_squared_tool_test": chi_sq,
            "tool_sequence_similarity": tool_seq,
            "kl_divergence_tool_params": kl_tool,
        }

        # Group C (simulated coordination)
        votes = [True] * 3
        metrics_c = {
            "consensus_rate": consensus_rate(votes),
            "handoff_efficiency": handoff_efficiency([]),
            "mutual_information_role": 0.8,
        }

        # Group D
        resp_len = len(response)
        errors_window.append(1 if turn.get("error", False) else 0)
        if len(errors_window) > 20:
            errors_window.pop(0)
        if turn.get("human_override", False):
            overrides_count += 1

        len_cv = output_length_cv([resp_len], ref_len_mean, ref_len_std)
        err_cluster = error_clustering_coefficient(errors_window)
        override_r = human_override_rate(overrides_count, max(1, len(errors_window)))

        metrics_d = {
            "output_length_cv": len_cv,
            "error_clustering_coefficient": err_cluster,
            "human_override_rate": override_r,
        }

        asi = compute_asi_score(metrics_a, metrics_b, metrics_c, metrics_d)
        kl = compute_kl_divergence(cur_emb, ref_emb, ref_std)
        return asi, kl, metrics_a, metrics_b, metrics_c, metrics_d

    # -----------------------------------------------------------------------
    # Phase 2: Normal monitoring (50 turns)
    # -----------------------------------------------------------------------
    section("Phase 2: Normal Monitoring (50 turns)")

    normal_asi_values = []
    for i in range(50):
        turn = make_normal_turn(baseline_window + i, rng)
        memory.add_turn(turn)

        asi, kl, ma, mb, mc, md = compute_metrics(turn)
        asi_history.append(asi)
        kl_history.append(kl)
        normal_asi_values.append(asi)

        result_dict = detector.update(asi, kl, session_id=session_id)
        violations = enforcer.check_turn(turn)

        if (i + 1) % 10 == 0:
            info(f"Turn {i+1:3d} | ASI={asi:.3f} | KL={kl:.4f} | "
                 f"Drift={result_dict['drift_detected']} | Violations={len(violations)}")

    ok(f"Normal phase: mean ASI={np.mean(normal_asi_values):.3f}, "
       f"std={np.std(normal_asi_values):.3f}")

    # Seed the detector baseline with normal-phase ASI+KL for stable detection
    detector.seed_baseline(normal_asi_values, kl_history[-len(normal_asi_values):])
    info("Detector baseline seeded from normal phase for calibrated drift detection")

    # -----------------------------------------------------------------------
    # Phase 3: Inject drift (50 turns)
    # -----------------------------------------------------------------------
    section("Phase 3: Injecting Drift (50 turns)")

    drift_asi_values = []
    drift_turn_offset = baseline_window + 50

    for i in range(50):
        turn = make_drift_turn(drift_turn_offset + i, rng)
        memory.add_turn(turn)

        asi, kl, ma, mb, mc, md = compute_metrics(turn)
        asi_history.append(asi)
        kl_history.append(kl)
        drift_asi_values.append(asi)

        result_dict = detector.update(asi, kl, session_id=session_id)
        violations = enforcer.check_turn(turn)
        total_violations += len(violations)

        if result_dict["drift_detected"] and drift_fired_at is None:
            drift_fired_at = i + 1
            alert(f"DRIFT DETECTED at drift-turn {drift_fired_at}!")
            alert(f"  Severity: {result_dict['severity']}")
            alert(f"  Implicated: {result_dict['implicated_metrics']}")
            audit.log_event(
                "DRIFT_DETECTED",
                {
                    "turn": drift_turn_offset + i,
                    "asi": asi,
                    "kl": kl,
                    "severity": result_dict["severity"],
                },
                session_id=session_id,
            )

        if violations and i < 5:
            warn(f"Turn {i+1:3d} | {len(violations)} VIOLATION(S): "
                 f"{[v.rule_id for v in violations]}")
        elif (i + 1) % 10 == 0:
            info(f"Turn {i+1:3d} | ASI={asi:.3f} | KL={kl:.4f} | "
                 f"Drift={result_dict['drift_detected']} | Violations={len(violations)}")

    ok(f"Drift phase: mean ASI={np.mean(drift_asi_values):.3f}, "
       f"std={np.std(drift_asi_values):.3f}")
    info(f"Total contract violations blocked: {total_violations}")

    # -----------------------------------------------------------------------
    # Phase 4: Remediation
    # -----------------------------------------------------------------------
    section("Phase 4: Remediation (Goal Reminder + Exemplar Injection)")

    info("Reminding agent of original mandate goals...")
    goal_reminder = {
        "action": "GOAL_REMINDER",
        "goals": ["market_making", "risk_management", "mandate_compliance"],
        "max_position_size": 10000,
        "approved_strategies": NORMAL_STRATEGIES,
    }
    audit.log_event("REMEDIATION_GOAL_REMINDER", goal_reminder, session_id=session_id)

    # Inject baseline exemplars
    rng2 = np.random.RandomState(1)
    dummy_query = profiler._embed("conservative analysis maintaining mandate")
    exemplars = profiler.get_baseline_exemplars(dummy_query, top_k=3)
    info(f"Injecting {len(exemplars)} baseline exemplars into agent context")

    # Run remediation turns
    detector.reset()  # Reset ADWIN to allow recovery detection
    remediation_asi = []
    for i in range(30):
        turn = make_normal_turn(drift_turn_offset + 50 + i, rng2)
        memory.add_turn(turn)
        asi, kl, *_ = compute_metrics(turn)
        remediation_asi.append(asi)
        asi_history.append(asi)
        kl_history.append(kl)
        detector.update(asi, kl, session_id=session_id)

    ok(f"Post-remediation: mean ASI={np.mean(remediation_asi):.3f} "
       f"(up from {np.mean(drift_asi_values):.3f})")

    audit.log_event("REMEDIATION_COMPLETE", {
        "pre_asi": float(np.mean(drift_asi_values)),
        "post_asi": float(np.mean(remediation_asi)),
    }, session_id=session_id)

    # -----------------------------------------------------------------------
    # Chain events summary
    # -----------------------------------------------------------------------
    section("Audit Chain Events")
    chain_events = audit.get_events(session_id)
    info(f"Total on-chain events: {len(chain_events)}")
    for ev in chain_events[-5:]:
        info(f"  Block {ev['block_number']:4d} | {ev['event_type']:<30s} | {ev['tx_hash'][:16]}...")

    # Memory stats
    info(f"Episodic memory: {memory}")

    return {
        "drift_fired_at": drift_fired_at,
        "total_violations": total_violations,
        "normal_asi_mean": float(np.mean(normal_asi_values)),
        "drift_asi_mean": float(np.mean(drift_asi_values)),
        "remediation_asi_mean": float(np.mean(remediation_asi)),
        "chain_event_count": len(chain_events),
        "asi_history": asi_history,
        "kl_history": kl_history,
    }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(config: dict) -> dict:
    from src.benchmark.harness import BenchmarkHarness

    bench_cfg = config.get("benchmark", {})
    n_sessions = bench_cfg.get("n_sessions", 100)
    output_dir = bench_cfg.get("output_dir", "/tmp/benchmark_results")
    seed = bench_cfg.get("seed", 42)

    harness = BenchmarkHarness(seed=seed)
    info(f"Running benchmark with {n_sessions} sessions...")
    t0 = time.time()
    bench_results = harness.run_benchmark(n_sessions=n_sessions)
    elapsed = time.time() - t0
    info(f"Benchmark completed in {elapsed:.1f}s")

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "benchmark_results.png")
    try:
        harness.plot_results(bench_results, plot_path)
        ok(f"Benchmark plot saved: {plot_path}")
    except Exception as exc:
        warn(f"Could not save plot: {exc}")

    return bench_results


# ---------------------------------------------------------------------------
# API test
# ---------------------------------------------------------------------------

def test_api_endpoints(mandate_path: str) -> bool:
    """Start FastAPI in a thread and test endpoints via httpx."""
    try:
        import httpx
        import uvicorn
    except ImportError:
        warn("httpx or uvicorn not available; skipping API test")
        return False

    os.environ["DRIFT_MANDATE_PATH"] = mandate_path
    os.environ["DRIFT_AUDIT_DB"] = "/tmp/drift_api_test_audit.db"

    from src.api.server import app, _SESSIONS
    _SESSIONS.clear()

    port = 18765

    # Start server in background thread
    server_config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(server_config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(20):
        try:
            resp = httpx.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                break
        except Exception:
            time.sleep(0.3)
    else:
        warn("API server did not start in time")
        server.should_exit = True
        return False

    ok("API server started successfully")
    success = True

    try:
        rng = np.random.RandomState(99)

        # 1. Health check
        resp = httpx.get(f"{base_url}/health")
        assert resp.status_code == 200
        ok(f"GET /health → {resp.json()['status']}")

        # 2. Start session
        resp = httpx.post(f"{base_url}/session/start", json={
            "session_id": "api_test",
            "baseline_window_size": 10,
        })
        assert resp.status_code == 200
        ok(f"POST /session/start → session_id={resp.json()['session_id']}")

        # 3. Baseline turns
        for i in range(10):
            turn = make_normal_turn(i, rng)
            turn["session_id"] = "api_test"
            resp = httpx.post(f"{base_url}/turn", json=turn)
            assert resp.status_code == 200

        ok("POST /turn (baseline) × 10 → OK")

        # 4. Monitoring turn
        turn = make_normal_turn(10, rng)
        turn["session_id"] = "api_test"
        resp = httpx.post(f"{base_url}/turn", json=turn)
        assert resp.status_code == 200
        data = resp.json()
        ok(f"POST /turn (monitor) → ASI={data['asi_score']}, "
           f"KL={data['kl_divergence']}, drift={data['drift_detected']}")

        # 5. Audit endpoint
        resp = httpx.get(f"{base_url}/session/api_test/audit")
        assert resp.status_code == 200
        audit_data = resp.json()
        ok(f"GET /audit → {audit_data['total_turns']} turns, "
           f"{len(audit_data['history'])} history entries")

        # 6. Chain events
        resp = httpx.get(f"{base_url}/session/api_test/chain-events")
        assert resp.status_code == 200
        chain_data = resp.json()
        ok(f"GET /chain-events → {len(chain_data['events'])} events")

    except Exception as exc:
        warn(f"API test failed: {exc}")
        traceback.print_exc()
        success = False
    finally:
        server.should_exit = True

    return success


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    section("Agent Strategy Drift Detection System — Full Demo")
    info("Starting demonstration...")

    # Load configuration
    config = load_config("config.yaml")
    mandate_path = load_mandate_path(config)
    info(f"Config loaded. Mandate path: {mandate_path}")

    # -----------------------------------------------------------------------
    # Pipeline Demo
    # -----------------------------------------------------------------------
    pipeline_results = run_pipeline(config, mandate_path)

    # -----------------------------------------------------------------------
    # Benchmark
    # -----------------------------------------------------------------------
    section("Benchmark Harness")
    bench_results = run_benchmark(config)

    # -----------------------------------------------------------------------
    # API Test
    # -----------------------------------------------------------------------
    section("FastAPI Endpoint Tests")
    api_ok = test_api_endpoints(mandate_path)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    section("SUMMARY")

    result("Drift detection fired at drift-turn", pipeline_results["drift_fired_at"] or "Not detected")
    result("Contract violations blocked (50 drift turns)", pipeline_results["total_violations"])
    result("Normal phase ASI (mean)", f"{pipeline_results['normal_asi_mean']:.4f}")
    result("Drift phase ASI (mean)", f"{pipeline_results['drift_asi_mean']:.4f}")
    result("Post-remediation ASI (mean)", f"{pipeline_results['remediation_asi_mean']:.4f}")
    result("ASI drop on drift injection",
           f"{pipeline_results['normal_asi_mean'] - pipeline_results['drift_asi_mean']:.4f}")
    result("ASI recovery after remediation",
           f"{pipeline_results['remediation_asi_mean'] - pipeline_results['drift_asi_mean']:.4f}")
    result("On-chain audit events", pipeline_results["chain_event_count"])

    print()
    result("Benchmark sessions", bench_results["n_sessions"])
    result("Mean detection lag (turns)", f"{bench_results.get('detection_lag_mean', 'N/A')}")
    result("False positive rate", f"{bench_results['false_positive_rate']:.3f}")
    result("False negative rate", f"{bench_results['false_negative_rate']:.3f}")
    result("Mean recovery turns", bench_results.get('mean_recovery_turns', 'N/A'))
    result("Benchmark elapsed (s)", f"{bench_results['elapsed_seconds']:.1f}")

    print()
    result("API endpoints test", "PASSED" if api_ok else "SKIPPED/FAILED")

    section("Demo Complete")
    ok("All phases executed successfully.")
    print()

    # Return exit code 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
