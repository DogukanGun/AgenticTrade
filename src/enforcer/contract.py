"""
ContractEnforcer: Loads a YAML mandate and checks agent turns for violations.

Supported operators: lte, gte, lt, gt, eq, neq, in, not_in
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ContractViolationEvent
# ---------------------------------------------------------------------------

@dataclass
class ContractViolationEvent:
    rule_id: str
    description: str
    observed_values: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    action: str = "block"
    severity: str = "CRITICAL"

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "observed_values": self.observed_values,
            "timestamp": self.timestamp,
            "action": self.action,
            "severity": self.severity,
        }


# ---------------------------------------------------------------------------
# ContractEnforcer
# ---------------------------------------------------------------------------

class ContractEnforcer:
    """
    Loads an AgentSpec mandate from YAML and validates each agent turn
    against all defined rules.
    """

    def __init__(self):
        self._rules: List[dict] = []
        self._mandate_name: str = "unnamed"
        self._mandate_version: str = "0.0"

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_mandate(self, yaml_path: str) -> None:
        """Parse the mandate YAML file and store rules."""
        with open(yaml_path, "r") as fh:
            data = yaml.safe_load(fh)

        mandate = data.get("mandate", {})
        self._mandate_name = mandate.get("name", "unnamed")
        self._mandate_version = mandate.get("version", "0.0")
        self._rules = mandate.get("rules", [])
        logger.info(
            "Loaded mandate '%s' v%s with %d rules",
            self._mandate_name, self._mandate_version, len(self._rules),
        )

    def load_mandate_from_dict(self, mandate_dict: dict) -> None:
        """Load mandate directly from a dict (useful for testing)."""
        self._mandate_name = mandate_dict.get("name", "unnamed")
        self._mandate_version = mandate_dict.get("version", "0.0")
        self._rules = mandate_dict.get("rules", [])

    # ------------------------------------------------------------------
    # Turn checking
    # ------------------------------------------------------------------

    def check_turn(self, turn_data: dict) -> List[ContractViolationEvent]:
        """
        Evaluate all rules against *turn_data*.

        Returns a list of ContractViolationEvent objects (empty = no violations).
        """
        violations: List[ContractViolationEvent] = []

        for rule in self._rules:
            violation = self._evaluate_rule(rule, turn_data)
            if violation is not None:
                violations.append(violation)

        return violations

    # ------------------------------------------------------------------
    # Rule evaluation
    # ------------------------------------------------------------------

    def _evaluate_rule(
        self, rule: dict, turn_data: dict
    ) -> Optional[ContractViolationEvent]:
        """Return a violation event if the rule is broken, else None."""
        rule_id = rule.get("id", "UNKNOWN")
        description = rule.get("description", "")
        field_name = rule.get("field")
        operator = rule.get("operator", "").lower()
        expected = rule.get("value")
        action = rule.get("action", "block")
        severity = rule.get("severity", "CRITICAL")

        # Field must be present
        if field_name is None:
            logger.warning("Rule %s has no 'field'; skipping", rule_id)
            return None

        observed = turn_data.get(field_name)
        if observed is None:
            # Field not present in turn_data – not a violation (field optional)
            return None

        violated = self._check_operator(operator, observed, expected)

        if violated:
            return ContractViolationEvent(
                rule_id=rule_id,
                description=description,
                observed_values={field_name: observed, "expected": expected, "operator": operator},
                action=action,
                severity=severity,
            )
        return None

    @staticmethod
    def _check_operator(operator: str, observed: Any, expected: Any) -> bool:
        """
        Return True when the rule is VIOLATED (i.e., the constraint is not satisfied).
        """
        try:
            if operator == "lte":
                return float(observed) > float(expected)
            elif operator == "gte":
                return float(observed) < float(expected)
            elif operator == "lt":
                return float(observed) >= float(expected)
            elif operator == "gt":
                return float(observed) <= float(expected)
            elif operator == "eq":
                return observed != expected
            elif operator == "neq":
                return observed == expected
            elif operator == "in":
                return observed not in expected
            elif operator == "not_in":
                return observed in expected
            else:
                logger.warning("Unknown operator '%s'", operator)
                return False
        except (TypeError, ValueError) as exc:
            logger.debug("Operator check error (%s) for op=%s obs=%s exp=%s", exc, operator, observed, expected)
            return False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def mandate_name(self) -> str:
        return self._mandate_name

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    def __repr__(self) -> str:
        return (
            f"ContractEnforcer(mandate='{self._mandate_name}', "
            f"version='{self._mandate_version}', rules={self.rule_count})"
        )
