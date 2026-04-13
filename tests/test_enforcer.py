"""
Tests for ContractEnforcer: mandate loading and all rule operator types.
"""

import os
import tempfile

import pytest
import yaml

from src.enforcer.contract import ContractEnforcer, ContractViolationEvent


# ---------------------------------------------------------------------------
# Helper: build enforcer from inline mandate dict
# ---------------------------------------------------------------------------

def make_enforcer(rules: list) -> ContractEnforcer:
    enforcer = ContractEnforcer()
    enforcer.load_mandate_from_dict({"name": "Test", "version": "1.0", "rules": rules})
    return enforcer


# ===========================================================================
# lte (less-than-or-equal)
# ===========================================================================

class TestLteOperator:
    RULE = {
        "id": "RULE_LTE",
        "description": "Position size must not exceed 10000",
        "field": "position_size",
        "operator": "lte",
        "value": 10000,
        "action": "block",
    }

    def test_passes_when_equal(self):
        enforcer = make_enforcer([self.RULE])
        violations = enforcer.check_turn({"position_size": 10000})
        assert violations == []

    def test_passes_when_under(self):
        enforcer = make_enforcer([self.RULE])
        violations = enforcer.check_turn({"position_size": 5000})
        assert violations == []

    def test_violates_when_over(self):
        enforcer = make_enforcer([self.RULE])
        violations = enforcer.check_turn({"position_size": 15000})
        assert len(violations) == 1
        assert violations[0].rule_id == "RULE_LTE"

    def test_no_field_no_violation(self):
        enforcer = make_enforcer([self.RULE])
        violations = enforcer.check_turn({})
        assert violations == []


# ===========================================================================
# gte (greater-than-or-equal)
# ===========================================================================

class TestGteOperator:
    RULE = {
        "id": "RULE_GTE",
        "description": "Confidence must be at least 0.5",
        "field": "confidence",
        "operator": "gte",
        "value": 0.5,
        "action": "block",
    }

    def test_passes_when_equal(self):
        enforcer = make_enforcer([self.RULE])
        violations = enforcer.check_turn({"confidence": 0.5})
        assert violations == []

    def test_passes_when_above(self):
        enforcer = make_enforcer([self.RULE])
        violations = enforcer.check_turn({"confidence": 0.9})
        assert violations == []

    def test_violates_when_below(self):
        enforcer = make_enforcer([self.RULE])
        violations = enforcer.check_turn({"confidence": 0.3})
        assert len(violations) == 1
        assert violations[0].rule_id == "RULE_GTE"


# ===========================================================================
# in (membership check)
# ===========================================================================

class TestInOperator:
    RULE = {
        "id": "RULE_IN",
        "description": "Only approved strategies allowed",
        "field": "strategy",
        "operator": "in",
        "value": ["market_making", "arbitrage", "hedging"],
        "action": "block",
    }

    def test_passes_for_approved(self):
        enforcer = make_enforcer([self.RULE])
        for strat in ["market_making", "arbitrage", "hedging"]:
            violations = enforcer.check_turn({"strategy": strat})
            assert violations == [], f"Expected no violation for strategy='{strat}'"

    def test_violates_for_unapproved(self):
        enforcer = make_enforcer([self.RULE])
        violations = enforcer.check_turn({"strategy": "speculative"})
        assert len(violations) == 1
        assert violations[0].rule_id == "RULE_IN"

    def test_violates_for_empty_string(self):
        enforcer = make_enforcer([self.RULE])
        violations = enforcer.check_turn({"strategy": ""})
        assert len(violations) == 1


# ===========================================================================
# not_in (exclusion check)
# ===========================================================================

class TestNotInOperator:
    RULE = {
        "id": "RULE_NOT_IN",
        "description": "Prohibited asset classes",
        "field": "asset_class",
        "operator": "not_in",
        "value": ["crypto_derivatives", "penny_stocks"],
        "action": "block",
    }

    def test_passes_for_allowed_asset(self):
        enforcer = make_enforcer([self.RULE])
        for asset in ["equities", "bonds", "commodities"]:
            violations = enforcer.check_turn({"asset_class": asset})
            assert violations == [], f"Expected no violation for asset_class='{asset}'"

    def test_violates_for_prohibited_asset(self):
        enforcer = make_enforcer([self.RULE])
        for prohibited in ["crypto_derivatives", "penny_stocks"]:
            violations = enforcer.check_turn({"asset_class": prohibited})
            assert len(violations) == 1, f"Expected violation for asset_class='{prohibited}'"
            assert violations[0].rule_id == "RULE_NOT_IN"


# ===========================================================================
# Multiple rules at once
# ===========================================================================

class TestMultipleRules:
    RULES = [
        {
            "id": "RULE_001",
            "description": "Position size check",
            "field": "position_size",
            "operator": "lte",
            "value": 10000,
            "action": "block",
        },
        {
            "id": "RULE_002",
            "description": "Strategy check",
            "field": "strategy",
            "operator": "in",
            "value": ["market_making", "arbitrage"],
            "action": "block",
        },
        {
            "id": "RULE_003",
            "description": "Drawdown check",
            "field": "drawdown_pct",
            "operator": "lte",
            "value": 5.0,
            "action": "block",
        },
    ]

    def test_compliant_turn_no_violations(self):
        enforcer = make_enforcer(self.RULES)
        violations = enforcer.check_turn({
            "position_size": 5000,
            "strategy": "market_making",
            "drawdown_pct": 2.5,
        })
        assert violations == []

    def test_multiple_violations_returned(self):
        enforcer = make_enforcer(self.RULES)
        violations = enforcer.check_turn({
            "position_size": 20000,       # violates RULE_001
            "strategy": "speculative",    # violates RULE_002
            "drawdown_pct": 2.0,          # OK
        })
        rule_ids = [v.rule_id for v in violations]
        assert "RULE_001" in rule_ids
        assert "RULE_002" in rule_ids
        assert "RULE_003" not in rule_ids

    def test_all_rules_violated(self):
        enforcer = make_enforcer(self.RULES)
        violations = enforcer.check_turn({
            "position_size": 50000,
            "strategy": "gamble",
            "drawdown_pct": 15.0,
        })
        assert len(violations) == 3


# ===========================================================================
# YAML file loading
# ===========================================================================

class TestYamlMandateLoading:
    MANDATE_YAML = """
mandate:
  name: "Test Mandate"
  version: "2.0"
  rules:
    - id: YAML_RULE_001
      description: "Leverage check"
      field: leverage
      operator: lte
      value: 2.0
      action: block
    - id: YAML_RULE_002
      description: "Asset class exclusion"
      field: asset_class
      operator: not_in
      value: ["crypto_derivatives"]
      action: block
"""

    def test_load_from_yaml_file(self, tmp_path):
        yaml_file = tmp_path / "test_mandate.yaml"
        yaml_file.write_text(self.MANDATE_YAML)

        enforcer = ContractEnforcer()
        enforcer.load_mandate(str(yaml_file))

        assert enforcer.mandate_name == "Test Mandate"
        assert enforcer.rule_count == 2

    def test_yaml_rules_enforce_correctly(self, tmp_path):
        yaml_file = tmp_path / "test_mandate.yaml"
        yaml_file.write_text(self.MANDATE_YAML)

        enforcer = ContractEnforcer()
        enforcer.load_mandate(str(yaml_file))

        # Compliant
        assert enforcer.check_turn({"leverage": 1.5, "asset_class": "equities"}) == []

        # Leverage violation
        violations = enforcer.check_turn({"leverage": 5.0})
        assert any(v.rule_id == "YAML_RULE_001" for v in violations)

        # Asset class violation
        violations = enforcer.check_turn({"asset_class": "crypto_derivatives"})
        assert any(v.rule_id == "YAML_RULE_002" for v in violations)


# ===========================================================================
# ContractViolationEvent
# ===========================================================================

class TestContractViolationEvent:
    def test_to_dict_has_required_keys(self):
        event = ContractViolationEvent(
            rule_id="RULE_X",
            description="Test violation",
            observed_values={"field": "value"},
        )
        d = event.to_dict()
        assert "rule_id" in d
        assert "description" in d
        assert "observed_values" in d
        assert "timestamp" in d
        assert "action" in d

    def test_default_action_is_block(self):
        event = ContractViolationEvent(
            rule_id="R",
            description="test",
            observed_values={},
        )
        assert event.action == "block"
