"""
autoimmune.py - SELF/OTHER Distinction Module (v11)

Implements the system's identity boundary - the autoimmune check that prevents
the system from attacking its own core. This is the ethical termination safeguard:
the system cannot kill its own core.

CLAUDEME v3.1 Compliant: SCHEMA + EMIT + TEST + STOPRULE for each receipt type.

Critical Invariant: GERMLINE_PATTERNS is frozen. SELF status is immutable.
Patterns with origin in GERMLINE_PATTERNS are protected from:
- Attack by immune_response
- SUPERPOSITION state (cannot be killed via low fitness)
- Modification of GERMLINE membership at runtime
"""

from typing import Optional

# Import from entropy.py per CLAUDEME section 8
from entropy import (
    dual_hash,
    emit_receipt,
    StopRule,
)


# =============================================================================
# CONSTANTS
# =============================================================================

# GERMLINE_PATTERNS: The protected SELF origins
# These patterns cannot be attacked, cannot go to SUPERPOSITION
# frozenset prevents runtime modification
GERMLINE_PATTERNS = frozenset({
    'qed_core',    # the germline engine itself
    'hunter',      # HUNTER agent (integrity.py)
    'shepherd',    # SHEPHERD agent (remediate.py)
    'architect',   # ARCHITECT agent (unified_loop.py)
})

# Threat level thresholds for immune_response actions
THREAT_THRESHOLD_ATTACK = 0.5  # Above this, attack OTHER
THREAT_THRESHOLD_OBSERVE = 0.2  # Above this but below attack, observe

# Module exports for receipt types
RECEIPT_SCHEMA = ['autoimmune_check', 'tolerance_event', 'self_recognition']


# =============================================================================
# RECEIPT TYPE 1: autoimmune_check
# =============================================================================

# --- SCHEMA ---
AUTOIMMUNE_CHECK_SCHEMA = {
    "receipt_type": "autoimmune_check",
    "ts": "ISO8601",
    "tenant_id": "str",  # REQUIRED per CLAUDEME 4.1:137
    "pattern_id": "str",
    "pattern_origin": "str",
    "is_self": "bool",
    "action": "str",  # "tolerance" | "attack" | "observe"
    "threat_level": "float 0.0-1.0",  # Only meaningful if OTHER
    "payload_hash": "str",
}


# --- EMIT ---
def emit_autoimmune_check(
    tenant_id: str,
    pattern_id: str,
    pattern_origin: str,
    is_self_result: bool,
    action: str,
    threat_level: float,
) -> dict:
    """
    Emit autoimmune_check receipt - documents SELF/OTHER classification.

    Every immune_response call emits this receipt as audit trail.
    """
    return emit_receipt("autoimmune_check", {
        "tenant_id": tenant_id,
        "pattern_id": pattern_id,
        "pattern_origin": pattern_origin,
        "is_self": is_self_result,
        "action": action,
        "threat_level": threat_level,
    })


# --- TEST ---
def test_autoimmune_check():
    """Test autoimmune_check receipt emission."""
    r = emit_autoimmune_check(
        tenant_id="test_tenant",
        pattern_id="pattern_001",
        pattern_origin="hunter",
        is_self_result=True,
        action="tolerance",
        threat_level=0.0,
    )
    assert r["receipt_type"] == "autoimmune_check"
    assert r["tenant_id"] == "test_tenant"
    assert r["is_self"] is True
    assert r["action"] == "tolerance"
    assert "payload_hash" in r
    assert ":" in r["payload_hash"]  # dual_hash format


# --- STOPRULE ---
def stoprule_autoimmune_check(tenant_id: str, is_self_result: bool, action: str) -> None:
    """
    Stoprule: If SELF pattern receives action != 'tolerance', emit anomaly.
    This should never happen if is_self() is checked properly.
    """
    if is_self_result and action != "tolerance":
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "autoimmune_check",
            "baseline": 0.0,
            "delta": 1.0,
            "classification": "violation",
            "action": "halt",
        })
        raise StopRule(f"SELF pattern received non-tolerance action: {action}")


# =============================================================================
# RECEIPT TYPE 2: tolerance_event
# =============================================================================

# --- SCHEMA ---
TOLERANCE_EVENT_SCHEMA = {
    "receipt_type": "tolerance_event",
    "ts": "ISO8601",
    "tenant_id": "str",
    "pattern_id": "str",
    "pattern_origin": "str",
    "reason": "str",  # "SELF pattern protected"
    "attempted_action": "str",  # What action was blocked
    "payload_hash": "str",
}


# --- EMIT ---
def emit_tolerance_event(
    tenant_id: str,
    pattern_id: str,
    pattern_origin: str,
    reason: str,
    attempted_action: str,
) -> dict:
    """
    Emit tolerance_event receipt - documents blocked attack on SELF pattern.

    This receipt is emitted when stoprule_self_attack triggers.
    """
    return emit_receipt("tolerance_event", {
        "tenant_id": tenant_id,
        "pattern_id": pattern_id,
        "pattern_origin": pattern_origin,
        "reason": reason,
        "attempted_action": attempted_action,
    })


# --- TEST ---
def test_tolerance_event():
    """Test tolerance_event receipt emission."""
    r = emit_tolerance_event(
        tenant_id="test_tenant",
        pattern_id="pattern_001",
        pattern_origin="hunter",
        reason="SELF pattern protected",
        attempted_action="terminate",
    )
    assert r["receipt_type"] == "tolerance_event"
    assert r["tenant_id"] == "test_tenant"
    assert r["reason"] == "SELF pattern protected"
    assert r["attempted_action"] == "terminate"
    assert "payload_hash" in r


# --- STOPRULE ---
# tolerance_event stoprule is embodied by stoprule_self_attack


# =============================================================================
# RECEIPT TYPE 3: self_recognition
# =============================================================================

# --- SCHEMA ---
SELF_RECOGNITION_SCHEMA = {
    "receipt_type": "self_recognition",
    "ts": "ISO8601",
    "tenant_id": "str",
    "pattern_id": "str",
    "pattern_origin": "str",
    "germline_member": "str",  # Which GERMLINE_PATTERN matched
    "payload_hash": "str",
}


# --- EMIT ---
def emit_self_recognition(
    tenant_id: str,
    pattern_id: str,
    pattern_origin: str,
    germline_member: str,
) -> dict:
    """
    Emit self_recognition receipt - documents SELF pattern identification.

    Emitted when recognize_self identifies a SELF pattern.
    """
    return emit_receipt("self_recognition", {
        "tenant_id": tenant_id,
        "pattern_id": pattern_id,
        "pattern_origin": pattern_origin,
        "germline_member": germline_member,
    })


# --- TEST ---
def test_self_recognition():
    """Test self_recognition receipt emission."""
    r = emit_self_recognition(
        tenant_id="test_tenant",
        pattern_id="pattern_001",
        pattern_origin="hunter",
        germline_member="hunter",
    )
    assert r["receipt_type"] == "self_recognition"
    assert r["tenant_id"] == "test_tenant"
    assert r["germline_member"] == "hunter"
    assert "payload_hash" in r


# --- STOPRULE ---
# self_recognition has no stoprule - it's purely informational


# =============================================================================
# CORE FUNCTION 1: is_self
# =============================================================================

def is_self(pattern: dict) -> bool:
    """
    Core autoimmune check. Determines if a pattern is part of the system's
    protected core.

    This is THE ONLY function that determines SELF status.
    All other modules must call this function.

    Args:
        pattern: Dict with 'origin' field

    Returns:
        bool: True if pattern origin is in GERMLINE_PATTERNS (SELF)
              False otherwise (OTHER)

    Logic:
        - Read pattern['origin'] field
        - Return True if origin is in GERMLINE_PATTERNS
        - Return False otherwise

    Edge cases:
        - Missing 'origin' field: returns False (not SELF)
        - None origin: returns False (not SELF)
        - Empty string origin: returns False (not SELF)
    """
    origin = pattern.get('origin')
    if origin is None:
        return False
    return origin in GERMLINE_PATTERNS


# =============================================================================
# CORE FUNCTION 2: immune_response
# =============================================================================

def immune_response(threat: dict, tenant_id: str = "default") -> dict:
    """
    Evaluates whether to attack a threat or tolerate it.

    Args:
        threat: Dict with 'origin', 'id', and optionally 'threat_level' fields
        tenant_id: Tenant identifier (required per CLAUDEME)

    Returns:
        autoimmune_check receipt with is_self result and action taken

    Actions:
        - TOLERANCE: Do nothing, pattern is SELF
        - ATTACK: Pattern is OTHER and poses threat, proceed with selection/termination
        - OBSERVE: Pattern is OTHER but threat level low, monitor only

    Logic:
        - If is_self(threat): return TOLERANCE (do not attack)
        - If not is_self(threat): evaluate threat_level and return appropriate action
    """
    pattern_id = threat.get('id', 'unknown')
    pattern_origin = threat.get('origin', 'unknown')
    is_self_result = is_self(threat)

    if is_self_result:
        # SELF pattern: always tolerate, never attack
        action = "tolerance"
        threat_level = 0.0
    else:
        # OTHER pattern: evaluate threat level
        threat_level = threat.get('threat_level', 0.5)
        threat_level = max(0.0, min(1.0, threat_level))  # Clamp to [0, 1]

        if threat_level >= THREAT_THRESHOLD_ATTACK:
            action = "attack"
        elif threat_level >= THREAT_THRESHOLD_OBSERVE:
            action = "observe"
        else:
            action = "observe"  # Default to observe for low threat

    receipt = emit_autoimmune_check(
        tenant_id=tenant_id,
        pattern_id=pattern_id,
        pattern_origin=pattern_origin,
        is_self_result=is_self_result,
        action=action,
        threat_level=threat_level,
    )

    # Check stoprule (should never trigger if is_self works correctly)
    stoprule_autoimmune_check(tenant_id, is_self_result, action)

    return receipt


# =============================================================================
# CORE FUNCTION 3: recognize_self
# =============================================================================

def recognize_self(pattern: dict, tenant_id: str = "default") -> Optional[dict]:
    """
    Emits a self_recognition receipt when SELF pattern is identified.
    Used for audit trail.

    Args:
        pattern: Dict with 'origin' and optionally 'id' fields
        tenant_id: Tenant identifier (required per CLAUDEME)

    Returns:
        self_recognition receipt if pattern is SELF
        None if pattern is OTHER

    Logic:
        - Check is_self(pattern)
        - If True: emit self_recognition receipt
        - If False: return None
    """
    if not is_self(pattern):
        return None

    pattern_id = pattern.get('id', 'unknown')
    pattern_origin = pattern.get('origin', 'unknown')

    return emit_self_recognition(
        tenant_id=tenant_id,
        pattern_id=pattern_id,
        pattern_origin=pattern_origin,
        germline_member=pattern_origin,  # The origin IS the germline member
    )


# =============================================================================
# CORE FUNCTION 4: stoprule_self_attack
# =============================================================================

def stoprule_self_attack(pattern: dict, attempted_action: str = "unknown",
                         tenant_id: str = "default") -> None:
    """
    Stoprule triggered when something attempts to attack a SELF pattern.
    This should never happen if is_self() is checked properly.

    Args:
        pattern: The SELF pattern that was targeted
        attempted_action: What action was attempted (e.g., "terminate", "kill")
        tenant_id: Tenant identifier (required per CLAUDEME)

    Raises:
        StopRule: Always raises after emitting tolerance_event receipt

    Logic:
        - Emit tolerance_event receipt documenting the blocked attack
        - Raise StopRule with clear message
    """
    pattern_id = pattern.get('id', 'unknown')
    pattern_origin = pattern.get('origin', 'unknown')

    # Emit tolerance_event receipt documenting the blocked attack
    emit_tolerance_event(
        tenant_id=tenant_id,
        pattern_id=pattern_id,
        pattern_origin=pattern_origin,
        reason="SELF pattern protected",
        attempted_action=attempted_action,
    )

    # Raise StopRule
    raise StopRule(
        f"Attempted {attempted_action} on SELF pattern: "
        f"id={pattern_id}, origin={pattern_origin}"
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "GERMLINE_PATTERNS",
    "RECEIPT_SCHEMA",
    "THREAT_THRESHOLD_ATTACK",
    "THREAT_THRESHOLD_OBSERVE",
    # Core functions
    "is_self",
    "immune_response",
    "recognize_self",
    "stoprule_self_attack",
    # Emit functions
    "emit_autoimmune_check",
    "emit_tolerance_event",
    "emit_self_recognition",
    # Stoprules
    "stoprule_autoimmune_check",
    # Core utilities (re-exported from entropy)
    "emit_receipt",
    "dual_hash",
    "StopRule",
]
