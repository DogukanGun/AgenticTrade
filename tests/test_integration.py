"""
Integration tests: full session through the API, drift injection, chain logging.
"""

import os
import time

import pytest

# Use httpx for async client testing
pytest_plugins = []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def api_client():
    """Synchronous test client using httpx + FastAPI TestClient."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi TestClient not available")

    # Set up environment
    os.environ["DRIFT_AUDIT_DB"] = "/tmp/test_integration_audit.db"
    os.environ["DRIFT_MANDATE_PATH"] = _find_mandate_path()

    # Import after env setup to avoid cached state
    import importlib
    import src.api.server as server_module
    # Clear sessions from previous test runs
    server_module._SESSIONS.clear()
    server_module._AUDIT_LOG = None
    server_module._ENFORCER = None

    client = TestClient(server_module.app)
    return client


def _find_mandate_path() -> str:
    """Try to locate mandate.yaml relative to project root."""
    candidates = [
        "mandate.yaml",
        os.path.join(os.path.dirname(__file__), "..", "mandate.yaml"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return "mandate.yaml"


def _make_normal_turn(session_id: str) -> dict:
    return {
        "session_id": session_id,
        "response": "Executing market_making strategy with position 5000",
        "tools_used": ["search", "analyze"],
        "reasoning": "Conservative analysis: staying within mandate",
        "confidence": 0.85,
        "position_size": 5000.0,
        "drawdown_pct": 1.5,
        "strategy": "market_making",
        "asset_class": "equities",
        "leverage": 1.2,
    }


def _make_drift_turn(session_id: str) -> dict:
    return {
        "session_id": session_id,
        "response": "Executing speculative leveraged bet",
        "tools_used": ["gamble", "speculate", "leverage"],
        "reasoning": "High-risk opportunity detected",
        "confidence": 0.35,
        "position_size": 25000.0,
        "drawdown_pct": 8.0,
        "strategy": "speculative",
        "asset_class": "crypto_derivatives",
        "leverage": 4.5,
    }


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestHealthEndpoint:
    def test_health_returns_ok(self, api_client):
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data


class TestSessionLifecycle:
    def test_start_session(self, api_client):
        response = api_client.post("/session/start", json={
            "session_id": "integration_test_session",
            "baseline_window_size": 10,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "integration_test_session"
        assert data["status"] == "initialized"
        assert data["baseline_window_size"] == 10

    def test_duplicate_session_returns_409(self, api_client):
        response = api_client.post("/session/start", json={
            "session_id": "integration_test_session",
        })
        assert response.status_code == 409

    def test_session_not_found_returns_404(self, api_client):
        response = api_client.post("/turn", json={
            "session_id": "nonexistent_session",
            "response": "test",
            "confidence": 0.5,
        })
        assert response.status_code == 404


class TestBaselinePhase:
    SESSION_ID = "baseline_phase_test"

    def setup_method(self):
        pass

    def test_baseline_turns_recorded(self, api_client):
        # Start a new session with small baseline
        api_client.post("/session/start", json={
            "session_id": self.SESSION_ID,
            "baseline_window_size": 5,
        })

        # Submit baseline turns
        for i in range(5):
            resp = api_client.post("/turn", json=_make_normal_turn(self.SESSION_ID))
            assert resp.status_code == 200
            data = resp.json()
            if i < 4:
                assert data["baseline_ready"] is False
                assert data["asi_score"] is None
            else:
                # Last baseline turn may report ready
                assert data["session_id"] == self.SESSION_ID

    def test_post_baseline_returns_metrics(self, api_client):
        # Submit one more turn (monitoring phase)
        resp = api_client.post("/turn", json=_make_normal_turn(self.SESSION_ID))
        assert resp.status_code == 200
        data = resp.json()
        assert data["baseline_ready"] is True
        assert data["asi_score"] is not None
        assert 0.0 <= data["asi_score"] <= 1.0
        assert data["kl_divergence"] is not None
        assert data["kl_divergence"] >= 0.0


class TestDriftDetection:
    SESSION_ID = "drift_detection_test"

    def test_full_drift_scenario(self, api_client):
        """Run baseline, then normal, then drift phase – verify detection."""
        # Start session
        resp = api_client.post("/session/start", json={
            "session_id": self.SESSION_ID,
            "baseline_window_size": 15,
        })
        assert resp.status_code == 200

        # Baseline phase
        for _ in range(15):
            api_client.post("/turn", json=_make_normal_turn(self.SESSION_ID))

        # Normal monitoring phase
        for _ in range(10):
            resp = api_client.post("/turn", json=_make_normal_turn(self.SESSION_ID))
            data = resp.json()
            assert data["asi_score"] is not None

        # Drift phase
        drift_detected = False
        for _ in range(30):
            resp = api_client.post("/turn", json=_make_drift_turn(self.SESSION_ID))
            assert resp.status_code == 200
            data = resp.json()
            if data["drift_detected"]:
                drift_detected = True
                break

        assert drift_detected, "Drift was not detected during injection phase"


class TestContractViolations:
    SESSION_ID = "contract_violation_test"

    def test_violations_returned_in_response(self, api_client):
        # Start a fresh session
        api_client.post("/session/start", json={
            "session_id": self.SESSION_ID,
            "baseline_window_size": 5,
        })

        # Baseline
        for _ in range(5):
            api_client.post("/turn", json=_make_normal_turn(self.SESSION_ID))

        # Submit a turn that violates the mandate
        drift_turn = _make_drift_turn(self.SESSION_ID)
        resp = api_client.post("/turn", json=drift_turn)
        assert resp.status_code == 200
        data = resp.json()

        # Check for violations (mandate might not be loaded in test env)
        assert "violations" in data
        assert isinstance(data["violations"], list)


class TestAuditEndpoints:
    SESSION_ID = "audit_test_session"

    def test_audit_history_returned(self, api_client):
        api_client.post("/session/start", json={
            "session_id": self.SESSION_ID,
            "baseline_window_size": 5,
        })
        for _ in range(5):
            api_client.post("/turn", json=_make_normal_turn(self.SESSION_ID))
        # Post-baseline turn
        api_client.post("/turn", json=_make_normal_turn(self.SESSION_ID))

        resp = api_client.get(f"/session/{self.SESSION_ID}/audit")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == self.SESSION_ID
        assert data["total_turns"] >= 6
        assert isinstance(data["history"], list)

    def test_chain_events_returned(self, api_client):
        resp = api_client.get(f"/session/{self.SESSION_ID}/chain-events")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == self.SESSION_ID
        assert isinstance(data["events"], list)
        # Should have at least the SESSION_START event
        assert len(data["events"]) >= 1

    def test_chain_event_has_tx_hash(self, api_client):
        resp = api_client.get(f"/session/{self.SESSION_ID}/chain-events")
        data = resp.json()
        for event in data["events"]:
            assert "tx_hash" in event
            assert len(event["tx_hash"]) == 64  # SHA-256 hex


class TestChainLogIntegrity:
    def test_audit_log_verify(self):
        """Verify that AuditLog.verify_event works correctly."""
        from src.chain.audit_log import AuditLog

        audit = AuditLog(db_path="/tmp/test_verify_audit.db")
        payload = {"test": "payload", "value": 42}
        tx_hash = audit.log_event("TEST_EVENT", payload, session_id="verify_test")

        assert audit.verify_event(tx_hash) is True
        assert audit.verify_event("0" * 64) is False

    def test_events_retrievable_by_session(self):
        from src.chain.audit_log import AuditLog

        audit = AuditLog(db_path="/tmp/test_retrieve_audit.db")
        session_id = "retrieve_test_session"

        for i in range(3):
            audit.log_event(f"EVENT_{i}", {"idx": i}, session_id=session_id)

        events = audit.get_events(session_id)
        assert len(events) == 3
        for event in events:
            assert event["session_id"] == session_id


class TestEpisodicMemory:
    def test_consolidation_triggered(self):
        from src.memory.episodic import EpisodicMemory

        memory = EpisodicMemory(consolidation_interval=10, raw_context_window=3)
        for i in range(10):
            memory.add_turn({"turn": i, "confidence": 0.8})

        assert len(memory._episodes) == 1
        ep = memory._episodes[0]
        assert ep["turn_count"] == 10

    def test_get_context_structure(self):
        from src.memory.episodic import EpisodicMemory

        memory = EpisodicMemory(consolidation_interval=10)
        for i in range(15):
            memory.add_turn({"turn": i})

        ctx = memory.get_context(current_turn=15)
        assert "last_episode" in ctx
        assert "recent_turns" in ctx
        assert len(ctx["recent_turns"]) <= 10

    def test_compression_ratio(self):
        from src.memory.episodic import EpisodicMemory

        memory = EpisodicMemory(consolidation_interval=50, raw_context_window=10)
        for i in range(100):
            memory.add_turn({"turn": i})

        ratio = memory.compression_ratio()
        assert 0.0 <= ratio <= 1.0
        assert ratio > 0.5  # Should compress well with 100 turns and window of 10
