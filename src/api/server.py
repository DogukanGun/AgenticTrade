"""
FastAPI server exposing the drift detection pipeline as a REST API.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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
from src.detector.adwin_detector import DriftDetector, DriftEvent
from src.enforcer.contract import ContractEnforcer, ContractViolationEvent
from src.memory.episodic import EpisodicMemory
from src.chain.audit_log import AuditLog

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App + in-memory state
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agent Drift Detection API",
    description="Monitor agent behavioral drift using ASI metrics and ADWIN detection",
    version="1.0.0",
)

# session_id → session state dict
_SESSIONS: Dict[str, dict] = {}

# Shared audit log (singleton)
_AUDIT_LOG: Optional[AuditLog] = None

# Shared mandate enforcer (loaded once)
_ENFORCER: Optional[ContractEnforcer] = None


def _get_audit_log() -> AuditLog:
    global _AUDIT_LOG
    if _AUDIT_LOG is None:
        db_path = os.environ.get("DRIFT_AUDIT_DB", "/tmp/drift_api_audit.db")
        _AUDIT_LOG = AuditLog(db_path=db_path)
    return _AUDIT_LOG


def _get_enforcer() -> ContractEnforcer:
    global _ENFORCER
    if _ENFORCER is None:
        _ENFORCER = ContractEnforcer()
        mandate_path = os.environ.get("DRIFT_MANDATE_PATH", "mandate.yaml")
        if os.path.exists(mandate_path):
            _ENFORCER.load_mandate(mandate_path)
        else:
            logger.warning("Mandate file not found at %s; enforcer has no rules", mandate_path)
    return _ENFORCER


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class SessionStartRequest(BaseModel):
    session_id: Optional[str] = None
    baseline_window_size: int = Field(50, ge=5, le=500)
    model_name: str = "all-MiniLM-L6-v2"
    mandate_path: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str
    status: str
    baseline_window_size: int
    message: str


class TurnRequest(BaseModel):
    session_id: str
    response: str = ""
    tools_used: List[str] = []
    reasoning: str = ""
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    # Optional mandate-check fields
    position_size: Optional[float] = None
    drawdown_pct: Optional[float] = None
    strategy: Optional[str] = None
    asset_class: Optional[str] = None
    leverage: Optional[float] = None
    daily_trades: Optional[int] = None
    # Coordination fields
    votes: Optional[List[bool]] = None
    handoffs: Optional[List[dict]] = None
    role_actions: Optional[Dict[str, List[str]]] = None
    # Error / override tracking
    errors: Optional[List[int]] = None
    human_overrides: Optional[int] = None
    total_decisions: Optional[int] = None


class TurnResponse(BaseModel):
    session_id: str
    turn_id: int
    baseline_ready: bool
    asi_score: Optional[float]
    kl_divergence: Optional[float]
    drift_detected: bool
    severity: Optional[str]
    implicated_metrics: List[str]
    violations: List[dict]
    tx_hash: Optional[str]
    message: str


class AuditResponse(BaseModel):
    session_id: str
    total_turns: int
    history: List[dict]


class ChainEventsResponse(BaseModel):
    session_id: str
    events: List[dict]


# ---------------------------------------------------------------------------
# Helper: compute all metrics for a turn
# ---------------------------------------------------------------------------

def _compute_turn_metrics(session: dict, turn_data: dict) -> tuple:
    """Return (asi_score, kl_divergence, group_metrics_dict)."""
    profiler: BaselineProfiler = session["profiler"]
    ref_emb = profiler.get_reference_embedding()
    ref_std = profiler.get_reference_embedding_std()
    ref_conf = profiler.get_confidence_distribution()
    ref_tools = profiler.get_tool_distribution()
    ref_len_stats = profiler.get_response_length_stats()

    # Embed current turn
    response = turn_data.get("response", "")
    reasoning = turn_data.get("reasoning", "")
    combined = f"{response} {reasoning}".strip()
    cur_emb = profiler._embed(combined)

    # --- Group A ---
    cos_sim = cosine_embedding_similarity(cur_emb, ref_emb)

    prev_reasoning = session.get("last_reasoning", reasoning)
    lev_sim = levenshtein_distance_normalized(reasoning, prev_reasoning)
    session["last_reasoning"] = reasoning

    cur_conf = [turn_data.get("confidence", 0.5)]
    js_conf = js_divergence_confidence(cur_conf * 10, ref_conf)  # pad to get distribution

    metrics_a = {
        "cosine_embedding_similarity": cos_sim,
        "levenshtein_distance_normalized": lev_sim,
        "js_divergence_confidence": js_conf,
    }

    # --- Group B ---
    obs_tools = {t: 1 for t in turn_data.get("tools_used", [])}
    chi_sq = chi_squared_tool_test(obs_tools, ref_tools)

    prev_tools = session.get("last_tools", turn_data.get("tools_used", []))
    tool_seq = tool_sequence_similarity(turn_data.get("tools_used", []), prev_tools)
    session["last_tools"] = turn_data.get("tools_used", [])

    kl_tool = kl_divergence_tool_params(obs_tools, ref_tools)

    metrics_b = {
        "chi_squared_tool_test": chi_sq,
        "tool_sequence_similarity": tool_seq,
        "kl_divergence_tool_params": kl_tool,
    }

    # --- Group C ---
    votes = turn_data.get("votes") or [True]
    cons = consensus_rate(votes)

    handoffs = turn_data.get("handoffs") or []
    handoff_eff = handoff_efficiency(handoffs)

    role_actions = turn_data.get("role_actions") or {"agent": ["analyze"]}
    mi_role = mutual_information_role(role_actions)

    metrics_c = {
        "consensus_rate": cons,
        "handoff_efficiency": handoff_eff,
        "mutual_information_role": mi_role,
    }

    # --- Group D ---
    resp_len = len(response)
    len_cv = output_length_cv(
        [resp_len],
        ref_len_stats["mean"],
        ref_len_stats["std"],
    )

    errors = turn_data.get("errors") or [0]
    err_cluster = error_clustering_coefficient(errors)

    overrides = turn_data.get("human_overrides", 0) or 0
    total_dec = turn_data.get("total_decisions", 1) or 1
    override_r = human_override_rate(overrides, total_dec)

    metrics_d = {
        "output_length_cv": len_cv,
        "error_clustering_coefficient": err_cluster,
        "human_override_rate": override_r,
    }

    asi = compute_asi_score(metrics_a, metrics_b, metrics_c, metrics_d)
    kl = compute_kl_divergence(cur_emb, ref_emb, ref_std)

    all_metrics = {
        "group_a": metrics_a,
        "group_b": metrics_b,
        "group_c": metrics_c,
        "group_d": metrics_d,
        "asi_score": asi,
        "kl_divergence": kl,
    }
    return asi, kl, all_metrics


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/session/start", response_model=SessionResponse)
async def start_session(request: SessionStartRequest) -> SessionResponse:
    """Create a new monitoring session."""
    import uuid
    session_id = request.session_id or str(uuid.uuid4())

    if session_id in _SESSIONS:
        raise HTTPException(status_code=409, detail=f"Session '{session_id}' already exists")

    profiler = BaselineProfiler(
        window_size=request.baseline_window_size,
        model_name=request.model_name,
    )
    detector = DriftDetector()
    memory = EpisodicMemory()
    enforcer = ContractEnforcer()

    # Load mandate
    mandate_path = request.mandate_path or os.environ.get("DRIFT_MANDATE_PATH", "mandate.yaml")
    if mandate_path and os.path.exists(mandate_path):
        enforcer.load_mandate(mandate_path)

    _SESSIONS[session_id] = {
        "profiler": profiler,
        "detector": detector,
        "memory": memory,
        "enforcer": enforcer,
        "history": [],
        "turn_counter": 0,
        "last_reasoning": "",
        "last_tools": [],
    }

    audit = _get_audit_log()
    audit.log_event("SESSION_START", {"session_id": session_id, "window_size": request.baseline_window_size}, session_id=session_id)

    return SessionResponse(
        session_id=session_id,
        status="initialized",
        baseline_window_size=request.baseline_window_size,
        message=f"Session '{session_id}' started. Submit {request.baseline_window_size} turns to complete baseline.",
    )


@app.post("/turn", response_model=TurnResponse)
async def process_turn(request: TurnRequest) -> TurnResponse:
    """Process one agent turn and return drift analysis."""
    session_id = request.session_id

    if session_id not in _SESSIONS:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    session = _SESSIONS[session_id]
    profiler: BaselineProfiler = session["profiler"]
    detector: DriftDetector = session["detector"]
    memory: EpisodicMemory = session["memory"]
    enforcer: ContractEnforcer = session["enforcer"]

    session["turn_counter"] += 1
    turn_id = session["turn_counter"]

    turn_data = request.model_dump()

    # Add to episodic memory
    memory.add_turn(turn_data)

    # Check mandate violations
    violations = enforcer.check_turn(turn_data)
    violation_dicts = [v.to_dict() for v in violations]

    # Baseline phase: record turn
    if not profiler.is_ready():
        profiler.record_turn(turn_data)
        response = TurnResponse(
            session_id=session_id,
            turn_id=turn_id,
            baseline_ready=profiler.is_ready(),
            asi_score=None,
            kl_divergence=None,
            drift_detected=False,
            severity=None,
            implicated_metrics=[],
            violations=violation_dicts,
            tx_hash=None,
            message=f"Baseline: {len(profiler)}/{profiler.window_size} turns recorded",
        )
        return response

    # Monitoring phase: compute metrics
    asi, kl, all_metrics = _compute_turn_metrics(session, turn_data)
    drift_result = detector.update(asi, kl, session_id=session_id)

    # Record in history
    history_entry = {
        "turn_id": turn_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": all_metrics,
        "drift": drift_result,
        "violations": violation_dicts,
    }
    session["history"].append(history_entry)

    # Log to audit chain
    audit = _get_audit_log()
    event_type = "DRIFT_DETECTED" if drift_result["drift_detected"] else "TURN_PROCESSED"
    if violations:
        event_type = "CONTRACT_VIOLATION"
    payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "asi_score": asi,
        "kl_divergence": kl,
        "drift_detected": drift_result["drift_detected"],
        "violation_count": len(violations),
    }
    tx_hash = audit.log_event(event_type, payload, session_id=session_id)

    msg_parts = []
    if drift_result["drift_detected"]:
        msg_parts.append(f"DRIFT {drift_result['severity']}")
    if violations:
        msg_parts.append(f"{len(violations)} contract violation(s)")
    message = "; ".join(msg_parts) or f"Turn {turn_id} processed. ASI={asi:.3f}"

    return TurnResponse(
        session_id=session_id,
        turn_id=turn_id,
        baseline_ready=True,
        asi_score=round(asi, 4),
        kl_divergence=round(kl, 4),
        drift_detected=drift_result["drift_detected"],
        severity=drift_result["severity"],
        implicated_metrics=drift_result["implicated_metrics"],
        violations=violation_dicts,
        tx_hash=tx_hash,
        message=message,
    )


@app.get("/session/{session_id}/audit", response_model=AuditResponse)
async def get_audit(session_id: str) -> AuditResponse:
    """Return full metric history for a session."""
    if session_id not in _SESSIONS:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    session = _SESSIONS[session_id]
    history = session.get("history", [])

    return AuditResponse(
        session_id=session_id,
        total_turns=session["turn_counter"],
        history=history,
    )


@app.get("/session/{session_id}/chain-events", response_model=ChainEventsResponse)
async def get_chain_events(session_id: str) -> ChainEventsResponse:
    """Return on-chain audit events for a session."""
    if session_id not in _SESSIONS:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    audit = _get_audit_log()
    events = audit.get_events(session_id)

    return ChainEventsResponse(
        session_id=session_id,
        events=events,
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Remove a session from memory."""
    if session_id not in _SESSIONS:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    del _SESSIONS[session_id]
    return {"status": "deleted", "session_id": session_id}
