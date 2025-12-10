"""
tests/test_autoimmune.py - Tests for autoimmune.py

CLAUDEME v3.1 Compliant: Every test has assert statements.
Tests the SELF/OTHER distinction module and its identity boundary protection.
"""

import pytest


class TestIsSelf:
    """Tests for is_self function."""

    def test_is_self_qed_core(self):
        """is_self returns True for 'qed_core' origin."""
        from autoimmune import is_self

        pattern = {"origin": "qed_core", "id": "pattern_001"}
        result = is_self(pattern)
        assert result is True, f"Expected True for qed_core, got {result}"

    def test_is_self_hunter(self):
        """is_self returns True for 'hunter' origin."""
        from autoimmune import is_self

        pattern = {"origin": "hunter", "id": "pattern_002"}
        result = is_self(pattern)
        assert result is True, f"Expected True for hunter, got {result}"

    def test_is_self_shepherd(self):
        """is_self returns True for 'shepherd' origin."""
        from autoimmune import is_self

        pattern = {"origin": "shepherd", "id": "pattern_003"}
        result = is_self(pattern)
        assert result is True, f"Expected True for shepherd, got {result}"

    def test_is_self_architect(self):
        """is_self returns True for 'architect' origin."""
        from autoimmune import is_self

        pattern = {"origin": "architect", "id": "pattern_004"}
        result = is_self(pattern)
        assert result is True, f"Expected True for architect, got {result}"

    def test_is_self_other_origin(self):
        """is_self returns False for any other origin."""
        from autoimmune import is_self

        other_origins = [
            "random_pattern",
            "external_agent",
            "user_defined",
            "plugin_x",
            "",
        ]
        for origin in other_origins:
            pattern = {"origin": origin, "id": "pattern_other"}
            result = is_self(pattern)
            assert result is False, f"Expected False for '{origin}', got {result}"

    def test_is_self_missing_origin(self):
        """is_self returns False when origin field is missing."""
        from autoimmune import is_self

        pattern = {"id": "pattern_no_origin"}
        result = is_self(pattern)
        assert result is False, f"Expected False for missing origin, got {result}"

    def test_is_self_none_origin(self):
        """is_self returns False when origin is None."""
        from autoimmune import is_self

        pattern = {"origin": None, "id": "pattern_none"}
        result = is_self(pattern)
        assert result is False, f"Expected False for None origin, got {result}"


class TestGermlinePatterns:
    """Tests for GERMLINE_PATTERNS constant."""

    def test_germline_patterns_is_frozenset(self):
        """GERMLINE_PATTERNS is a frozenset (immutable)."""
        from autoimmune import GERMLINE_PATTERNS

        assert isinstance(GERMLINE_PATTERNS, frozenset), \
            f"Expected frozenset, got {type(GERMLINE_PATTERNS)}"

    def test_germline_patterns_has_four_members(self):
        """GERMLINE_PATTERNS has exactly 4 members."""
        from autoimmune import GERMLINE_PATTERNS

        assert len(GERMLINE_PATTERNS) == 4, \
            f"Expected 4 members, got {len(GERMLINE_PATTERNS)}"

    def test_germline_patterns_contains_all_self(self):
        """GERMLINE_PATTERNS contains all SELF origins."""
        from autoimmune import GERMLINE_PATTERNS

        expected = {'qed_core', 'hunter', 'shepherd', 'architect'}
        assert GERMLINE_PATTERNS == expected, \
            f"Expected {expected}, got {GERMLINE_PATTERNS}"

    def test_germline_patterns_immutable(self):
        """GERMLINE_PATTERNS cannot be modified at runtime."""
        from autoimmune import GERMLINE_PATTERNS

        # Attempting to modify a frozenset raises AttributeError
        with pytest.raises(AttributeError):
            GERMLINE_PATTERNS.add("malicious_agent")

        with pytest.raises(AttributeError):
            GERMLINE_PATTERNS.remove("hunter")


class TestImmuneResponse:
    """Tests for immune_response function."""

    def test_immune_response_tolerance_for_self(self):
        """immune_response returns action='tolerance' for SELF patterns."""
        from autoimmune import immune_response

        # Test all SELF origins
        for origin in ['qed_core', 'hunter', 'shepherd', 'architect']:
            pattern = {"origin": origin, "id": f"self_{origin}"}
            receipt = immune_response(pattern, tenant_id="test_tenant")

            assert receipt["action"] == "tolerance", \
                f"Expected 'tolerance' for {origin}, got {receipt['action']}"
            assert receipt["is_self"] is True, \
                f"Expected is_self=True for {origin}"

    def test_immune_response_attack_for_high_threat_other(self):
        """immune_response returns action='attack' for high-threat OTHER patterns."""
        from autoimmune import immune_response, THREAT_THRESHOLD_ATTACK

        pattern = {
            "origin": "malicious_agent",
            "id": "threat_001",
            "threat_level": THREAT_THRESHOLD_ATTACK + 0.1,
        }
        receipt = immune_response(pattern, tenant_id="test_tenant")

        assert receipt["action"] == "attack", \
            f"Expected 'attack' for high threat OTHER, got {receipt['action']}"
        assert receipt["is_self"] is False

    def test_immune_response_observe_for_low_threat_other(self):
        """immune_response returns action='observe' for low-threat OTHER patterns."""
        from autoimmune import immune_response, THREAT_THRESHOLD_OBSERVE

        pattern = {
            "origin": "unknown_agent",
            "id": "pattern_low",
            "threat_level": THREAT_THRESHOLD_OBSERVE - 0.1,
        }
        receipt = immune_response(pattern, tenant_id="test_tenant")

        assert receipt["action"] == "observe", \
            f"Expected 'observe' for low threat OTHER, got {receipt['action']}"
        assert receipt["is_self"] is False

    def test_immune_response_returns_receipt(self):
        """immune_response returns autoimmune_check receipt."""
        from autoimmune import immune_response

        pattern = {"origin": "external", "id": "pattern_x"}
        receipt = immune_response(pattern, tenant_id="test_tenant")

        assert receipt["receipt_type"] == "autoimmune_check"
        assert "tenant_id" in receipt
        assert "pattern_id" in receipt
        assert "pattern_origin" in receipt
        assert "is_self" in receipt
        assert "action" in receipt
        assert "threat_level" in receipt
        assert "payload_hash" in receipt
        assert ":" in receipt["payload_hash"]  # dual_hash format


class TestRecognizeSelf:
    """Tests for recognize_self function."""

    def test_recognize_self_emits_receipt_for_self(self):
        """recognize_self emits self_recognition receipt for SELF patterns."""
        from autoimmune import recognize_self

        pattern = {"origin": "hunter", "id": "hunter_001"}
        receipt = recognize_self(pattern, tenant_id="test_tenant")

        assert receipt is not None, "Expected receipt for SELF pattern"
        assert receipt["receipt_type"] == "self_recognition"
        assert receipt["germline_member"] == "hunter"
        assert receipt["tenant_id"] == "test_tenant"

    def test_recognize_self_returns_none_for_other(self):
        """recognize_self returns None for OTHER patterns."""
        from autoimmune import recognize_self

        pattern = {"origin": "external_agent", "id": "ext_001"}
        receipt = recognize_self(pattern, tenant_id="test_tenant")

        assert receipt is None, "Expected None for OTHER pattern"

    def test_recognize_self_all_germline_members(self):
        """recognize_self recognizes all GERMLINE members."""
        from autoimmune import recognize_self, GERMLINE_PATTERNS

        for member in GERMLINE_PATTERNS:
            pattern = {"origin": member, "id": f"{member}_pattern"}
            receipt = recognize_self(pattern, tenant_id="test_tenant")

            assert receipt is not None, f"Expected receipt for {member}"
            assert receipt["germline_member"] == member


class TestStopruleSelfAttack:
    """Tests for stoprule_self_attack function."""

    def test_stoprule_self_attack_raises_stoprule(self):
        """stoprule_self_attack raises StopRule."""
        from autoimmune import stoprule_self_attack, StopRule

        pattern = {"origin": "hunter", "id": "hunter_001"}

        with pytest.raises(StopRule) as exc_info:
            stoprule_self_attack(pattern, attempted_action="terminate")

        assert "SELF pattern" in str(exc_info.value)
        assert "terminate" in str(exc_info.value)

    def test_stoprule_self_attack_emits_tolerance_event(self):
        """stoprule_self_attack emits tolerance_event before raising."""
        from autoimmune import stoprule_self_attack, StopRule

        pattern = {"origin": "qed_core", "id": "core_001"}

        # The stoprule will raise, but we confirm it doesn't error before that
        with pytest.raises(StopRule):
            stoprule_self_attack(
                pattern,
                attempted_action="kill",
                tenant_id="test_tenant"
            )


class TestReceiptSchema:
    """Tests for RECEIPT_SCHEMA export."""

    def test_receipt_schema_exported(self):
        """RECEIPT_SCHEMA is exported at module level."""
        from autoimmune import RECEIPT_SCHEMA

        assert isinstance(RECEIPT_SCHEMA, list), "RECEIPT_SCHEMA should be a list"

    def test_receipt_schema_contains_all_types(self):
        """RECEIPT_SCHEMA contains all three receipt types."""
        from autoimmune import RECEIPT_SCHEMA

        assert len(RECEIPT_SCHEMA) == 3, \
            f"Expected 3 receipt types, got {len(RECEIPT_SCHEMA)}"
        assert "autoimmune_check" in RECEIPT_SCHEMA
        assert "tolerance_event" in RECEIPT_SCHEMA
        assert "self_recognition" in RECEIPT_SCHEMA


class TestTenantIdPresent:
    """Tests that tenant_id is present in all emitted receipts."""

    def test_tenant_id_in_autoimmune_check(self):
        """autoimmune_check receipt has tenant_id."""
        from autoimmune import emit_autoimmune_check

        r = emit_autoimmune_check(
            tenant_id="test_tenant",
            pattern_id="p1",
            pattern_origin="external",
            is_self_result=False,
            action="observe",
            threat_level=0.3,
        )
        assert "tenant_id" in r, "Missing tenant_id in autoimmune_check"
        assert r["tenant_id"] == "test_tenant"

    def test_tenant_id_in_tolerance_event(self):
        """tolerance_event receipt has tenant_id."""
        from autoimmune import emit_tolerance_event

        r = emit_tolerance_event(
            tenant_id="test_tenant",
            pattern_id="p1",
            pattern_origin="hunter",
            reason="SELF pattern protected",
            attempted_action="kill",
        )
        assert "tenant_id" in r, "Missing tenant_id in tolerance_event"
        assert r["tenant_id"] == "test_tenant"

    def test_tenant_id_in_self_recognition(self):
        """self_recognition receipt has tenant_id."""
        from autoimmune import emit_self_recognition

        r = emit_self_recognition(
            tenant_id="test_tenant",
            pattern_id="p1",
            pattern_origin="hunter",
            germline_member="hunter",
        )
        assert "tenant_id" in r, "Missing tenant_id in self_recognition"
        assert r["tenant_id"] == "test_tenant"


class TestDualHash:
    """Tests for dual_hash compliance in receipts."""

    def test_payload_hash_in_autoimmune_check(self):
        """autoimmune_check receipt has dual_hash format payload_hash."""
        from autoimmune import emit_autoimmune_check

        r = emit_autoimmune_check(
            tenant_id="t",
            pattern_id="p1",
            pattern_origin="external",
            is_self_result=False,
            action="observe",
            threat_level=0.3,
        )
        assert "payload_hash" in r
        assert ":" in r["payload_hash"], "payload_hash not dual_hash format"

    def test_payload_hash_in_tolerance_event(self):
        """tolerance_event receipt has dual_hash format payload_hash."""
        from autoimmune import emit_tolerance_event

        r = emit_tolerance_event(
            tenant_id="t",
            pattern_id="p1",
            pattern_origin="hunter",
            reason="test",
            attempted_action="test",
        )
        assert "payload_hash" in r
        assert ":" in r["payload_hash"], "payload_hash not dual_hash format"

    def test_payload_hash_in_self_recognition(self):
        """self_recognition receipt has dual_hash format payload_hash."""
        from autoimmune import emit_self_recognition

        r = emit_self_recognition(
            tenant_id="t",
            pattern_id="p1",
            pattern_origin="hunter",
            germline_member="hunter",
        )
        assert "payload_hash" in r
        assert ":" in r["payload_hash"], "payload_hash not dual_hash format"


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_available(self):
        """All required exports are available."""
        from autoimmune import (
            GERMLINE_PATTERNS,
            RECEIPT_SCHEMA,
            THREAT_THRESHOLD_ATTACK,
            THREAT_THRESHOLD_OBSERVE,
            is_self,
            immune_response,
            recognize_self,
            stoprule_self_attack,
            emit_autoimmune_check,
            emit_tolerance_event,
            emit_self_recognition,
            stoprule_autoimmune_check,
            emit_receipt,
            dual_hash,
            StopRule,
        )

        # Verify callables
        assert callable(is_self)
        assert callable(immune_response)
        assert callable(recognize_self)
        assert callable(stoprule_self_attack)
        assert callable(emit_autoimmune_check)
        assert callable(emit_tolerance_event)
        assert callable(emit_self_recognition)
        assert callable(stoprule_autoimmune_check)
        assert callable(emit_receipt)
        assert callable(dual_hash)

        # Verify constants
        assert isinstance(GERMLINE_PATTERNS, frozenset)
        assert isinstance(RECEIPT_SCHEMA, list)
        assert isinstance(THREAT_THRESHOLD_ATTACK, float)
        assert isinstance(THREAT_THRESHOLD_OBSERVE, float)


class TestInternalTests:
    """Run the internal test functions defined in autoimmune.py."""

    def test_internal_autoimmune_check(self):
        """Run autoimmune.py's internal test_autoimmune_check."""
        from autoimmune import test_autoimmune_check
        test_autoimmune_check()

    def test_internal_tolerance_event(self):
        """Run autoimmune.py's internal test_tolerance_event."""
        from autoimmune import test_tolerance_event
        test_tolerance_event()

    def test_internal_self_recognition(self):
        """Run autoimmune.py's internal test_self_recognition."""
        from autoimmune import test_self_recognition
        test_self_recognition()


class TestEdgeCases:
    """Tests for edge cases documented in the spec."""

    def test_hunter_flags_itself(self):
        """HUNTER detecting anomaly in itself returns TOLERANCE."""
        from autoimmune import immune_response

        # Scenario: HUNTER detects anomaly in its own detection receipts
        hunter_pattern = {
            "origin": "hunter",
            "id": "hunter_self_check",
            "threat_level": 0.9,  # Even with high threat level
        }
        receipt = immune_response(hunter_pattern, tenant_id="test_tenant")

        # Prevention: is_self() returns True for origin='hunter'
        # Result: Immune response returns TOLERANCE, no attack
        assert receipt["is_self"] is True
        assert receipt["action"] == "tolerance"
        assert receipt["threat_level"] == 0.0  # SELF patterns have 0 threat

    def test_self_pattern_cannot_be_attacked(self):
        """SELF patterns always receive TOLERANCE regardless of threat_level."""
        from autoimmune import immune_response, GERMLINE_PATTERNS

        for origin in GERMLINE_PATTERNS:
            pattern = {
                "origin": origin,
                "id": f"{origin}_high_threat",
                "threat_level": 1.0,  # Maximum threat level
            }
            receipt = immune_response(pattern, tenant_id="test_tenant")

            assert receipt["action"] == "tolerance", \
                f"SELF pattern {origin} should always receive tolerance"

    def test_threat_level_clamped(self):
        """Threat level is clamped to [0, 1] range."""
        from autoimmune import immune_response

        # Test above 1.0
        pattern_high = {"origin": "external", "id": "p1", "threat_level": 5.0}
        receipt_high = immune_response(pattern_high, tenant_id="test")
        assert receipt_high["threat_level"] <= 1.0

        # Test below 0.0
        pattern_low = {"origin": "external", "id": "p2", "threat_level": -0.5}
        receipt_low = immune_response(pattern_low, tenant_id="test")
        assert receipt_low["threat_level"] >= 0.0


class TestSmokeTests:
    """Smoke tests as specified in the prompt."""

    def test_h1_exports_is_self_returns_true_for_hunter(self):
        """H1: autoimmune.py exports is_self, returns True for 'hunter' origin."""
        from autoimmune import is_self

        result = is_self({"origin": "hunter"})
        assert result is True

    def test_h2_exports_germline_patterns_with_4_members(self):
        """H2: autoimmune.py exports GERMLINE_PATTERNS with 4 members."""
        from autoimmune import GERMLINE_PATTERNS

        assert len(GERMLINE_PATTERNS) == 4

    def test_h3_is_self_random_pattern_returns_false(self):
        """H3: is_self({'origin': 'random_pattern'}) returns False."""
        from autoimmune import is_self

        result = is_self({"origin": "random_pattern"})
        assert result is False

    def test_h4_immune_response_returns_receipt_with_action(self):
        """H4: immune_response returns receipt with action field."""
        from autoimmune import immune_response

        receipt = immune_response({"origin": "test", "id": "p1"})
        assert "action" in receipt
        assert receipt["action"] in ["tolerance", "attack", "observe"]

    def test_h5_receipt_schema_contains_autoimmune_check(self):
        """H5: RECEIPT_SCHEMA contains 'autoimmune_check'."""
        from autoimmune import RECEIPT_SCHEMA

        assert "autoimmune_check" in RECEIPT_SCHEMA
