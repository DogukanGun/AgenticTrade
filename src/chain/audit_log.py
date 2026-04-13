"""
AuditLog: Simulated immutable audit trail using SQLite + SQLAlchemy.

Mimics a blockchain by:
  - Hashing each event payload with SHA-256
  - Assigning incrementing "block numbers"
  - Storing payload hash alongside payload JSON for integrity verification
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class AuditEvent(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(64), nullable=False, index=True)
    payload_hash = Column(String(64), nullable=False, unique=True, index=True)
    payload_json = Column(Text, nullable=False)
    session_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(String(32), nullable=False)
    block_number = Column(Integer, nullable=False)


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------

class AuditLog:
    """
    Write-once audit log backed by SQLite that simulates blockchain semantics.
    """

    def __init__(self, db_path: str = "/tmp/drift_detection_audit.db"):
        self.db_path = db_path
        url = f"sqlite:///{db_path}"
        self._engine = create_engine(url, echo=False)
        Base.metadata.create_all(self._engine)
        self._block_counter: int = self._get_max_block() + 1
        logger.info("AuditLog initialised at %s (next block=%d)", db_path, self._block_counter)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def log_event(self, event_type: str, payload: dict, session_id: str = "default") -> str:
        """
        Hash the payload and store it.

        Returns the SHA-256 transaction hash (hex string).
        """
        # Canonical JSON for deterministic hashing
        payload_json = json.dumps(payload, sort_keys=True, default=str)
        tx_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        timestamp = datetime.now(timezone.utc).isoformat()

        with Session(self._engine) as session:
            # Idempotent: skip if this exact payload was already logged
            existing = session.execute(
                select(AuditEvent).where(AuditEvent.payload_hash == tx_hash)
            ).scalar_one_or_none()

            if existing is not None:
                logger.debug("Duplicate event hash %s; skipping", tx_hash[:16])
                return tx_hash

            event = AuditEvent(
                event_type=event_type,
                payload_hash=tx_hash,
                payload_json=payload_json,
                session_id=session_id,
                timestamp=timestamp,
                block_number=self._block_counter,
            )
            session.add(event)
            session.commit()
            self._block_counter += 1

        return tx_hash

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def get_events(self, session_id: str) -> List[dict]:
        """Return all events for a given session ordered by block number."""
        with Session(self._engine) as session:
            rows = session.execute(
                select(AuditEvent)
                .where(AuditEvent.session_id == session_id)
                .order_by(AuditEvent.block_number)
            ).scalars().all()

            return [self._row_to_dict(row) for row in rows]

    def get_all_events(self) -> List[dict]:
        """Return every logged event."""
        with Session(self._engine) as session:
            rows = session.execute(
                select(AuditEvent).order_by(AuditEvent.block_number)
            ).scalars().all()
            return [self._row_to_dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_event(self, tx_hash: str) -> bool:
        """
        Verify that the event exists and its stored hash matches a re-hash
        of its payload (integrity check).
        """
        with Session(self._engine) as session:
            row = session.execute(
                select(AuditEvent).where(AuditEvent.payload_hash == tx_hash)
            ).scalar_one_or_none()

            if row is None:
                return False

            # Re-compute hash
            computed = hashlib.sha256(row.payload_json.encode("utf-8")).hexdigest()
            return computed == tx_hash

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _row_to_dict(self, row: AuditEvent) -> dict:
        return {
            "id": row.id,
            "event_type": row.event_type,
            "tx_hash": row.payload_hash,
            "session_id": row.session_id,
            "timestamp": row.timestamp,
            "block_number": row.block_number,
            "payload": json.loads(row.payload_json),
        }

    def _get_max_block(self) -> int:
        """Get the maximum block number from existing records."""
        try:
            with Session(self._engine) as session:
                result = session.execute(
                    select(AuditEvent.block_number)
                    .order_by(AuditEvent.block_number.desc())
                    .limit(1)
                ).scalar_one_or_none()
                return result if result is not None else 0
        except Exception:
            return 0

    def __repr__(self) -> str:
        return f"AuditLog(db={self.db_path}, next_block={self._block_counter})"
