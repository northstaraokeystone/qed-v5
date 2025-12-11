"""
population.py - Entropy-Governed Population Module (v12)

Replaces fixed AGENT_CAP=20 with physics-based entropy budget.
The river doesn't count whirlpools - patterns emerge where the gradient allows.

CLAUDEME v3.1 Compliant: Pure functions with emit_receipt() pattern.
"""

import random
from typing import List, Tuple

from entropy import agent_fitness, system_entropy, emit_receipt, dual_hash, StopRule
from autoimmune import is_self

# =============================================================================
# CONSTANTS
# =============================================================================

BASE_CAP = 3  # HUNTER, SHEPHERD, ARCHITECT minimum (all SELF)
SURVIVAL_THRESHOLD = 0.0  # Positive fitness survives, negative to superposition
MAX_ENTROPY = 10.0  # System ceiling for load_factor calculation
CONSERVATION_TOLERANCE = 0.01  # 1% tolerance for entropy conservation

# Module exports for receipt types
RECEIPT_SCHEMA = ["population_snapshot", "selection_event", "superposition_event", "cap_adjustment"]

# Baseline compute for resource_factor calculation
_BASELINE_COMPUTE = 1.0
_AVAILABLE_COMPUTE = 1.0
_PREVIOUS_CAP = BASE_CAP


# =============================================================================
# CORE FUNCTION 1: dynamic_cap
# =============================================================================

def dynamic_cap(available_compute: float = 1.0, current_entropy: float = 0.0) -> int:
    """
    Physics-based population limit. Replaces fixed AGENT_CAP=20.

    Formula: max(BASE_CAP, BASE_CAP * resource_factor * load_factor)

    Args:
        available_compute: Available compute resources (default 1.0)
        current_entropy: Current system entropy (default 0.0)

    Returns:
        int: Dynamic population cap (>= BASE_CAP)
    """
    global _AVAILABLE_COMPUTE, _BASELINE_COMPUTE

    resource_factor = available_compute / _BASELINE_COMPUTE
    load_factor = 1.0 - (current_entropy / MAX_ENTROPY)
    load_factor = max(0.1, min(1.0, load_factor))  # Clamp to [0.1, 1.0]

    cap = int(BASE_CAP * resource_factor * load_factor)
    return max(BASE_CAP, cap)


# =============================================================================
# CORE FUNCTION 2: selection_pressure
# =============================================================================

def selection_pressure(patterns: list, tenant_id: str = "default") -> Tuple[list, list]:
    """
    Thompson sampling over fitness distributions with SUPERPOSITION state.

    SELF patterns (is_self() == True) always survive regardless of fitness.

    Args:
        patterns: List of dicts with 'fitness_mean', 'fitness_var', optional 'id'
        tenant_id: Tenant identifier

    Returns:
        Tuple of (survivors, superposition_patterns)
    """
    survivors = []
    superposition = []
    self_protected = 0

    for pattern in patterns:
        # SELF patterns ALWAYS survive
        if is_self(pattern):
            survivors.append(pattern)
            self_protected += 1
            continue

        # Thompson sampling for OTHER patterns
        mean = pattern.get("fitness_mean", 0.0)
        var = pattern.get("fitness_var", 0.0)
        std = (var ** 0.5) if var > 0 else 0.0

        sampled_fitness = random.gauss(mean, std) if std > 0 else mean

        if sampled_fitness > SURVIVAL_THRESHOLD:
            survivors.append(pattern)
        else:
            superposition.append(pattern)

    # Emit selection_event receipt
    emit_receipt("selection_event", {
        "tenant_id": tenant_id,
        "patterns_evaluated": len(patterns),
        "survivors": len(survivors),
        "to_superposition": len(superposition),
        "self_patterns_protected": self_protected,
        "sampling_method": "thompson"
    })

    return survivors, superposition


# =============================================================================
# CORE FUNCTION 3: entropy_budget
# =============================================================================

def entropy_budget(available_compute: float = 1.0) -> float:
    """
    Returns available entropy reduction capacity.

    More compute = larger budget = more patterns can survive.

    Args:
        available_compute: Available compute resources (default 1.0)

    Returns:
        float: Entropy budget
    """
    baseline_budget = 1.0
    resource_factor = available_compute / _BASELINE_COMPUTE
    return baseline_budget * resource_factor


# =============================================================================
# CORE FUNCTION 4: hilbert_bound
# =============================================================================

def hilbert_bound(available_compute: float = 1.0, current_entropy: float = 0.0) -> int:
    """
    Returns dynamic Hilbert space constraint (delegates to dynamic_cap).

    Agent space is bounded by physics, not arbitrary policy.
    No infinite agents - entropy gradient constrains.

    Args:
        available_compute: Available compute resources (default 1.0)
        current_entropy: Current system entropy (default 0.0)

    Returns:
        int: Hilbert space bound (>= BASE_CAP)
    """
    return dynamic_cap(available_compute, current_entropy)


# =============================================================================
# CORE FUNCTION 5: entropy_conservation
# =============================================================================

def entropy_conservation(entropy_in: float, entropy_out: float,
                        work_done: float, tenant_id: str = "default") -> Tuple[bool, float]:
    """
    Validates 2nd law: sum(entropy_in) = sum(entropy_out) + work_done

    If violation_delta > tolerance: hidden risk detected.
    Agents must export disorder - if entropy reduction visible but export
    not tracked, risk is hiding.

    Args:
        entropy_in: Input entropy
        entropy_out: Output entropy
        work_done: Work performed (entropy reduction)
        tenant_id: Tenant identifier

    Returns:
        Tuple of (is_valid, violation_delta)
    """
    expected = entropy_in
    actual = entropy_out + work_done
    violation_delta = abs(expected - actual)
    is_valid = violation_delta <= CONSERVATION_TOLERANCE

    if not is_valid:
        emit_receipt("conservation_violation", {
            "tenant_id": tenant_id,
            "entropy_in": entropy_in,
            "entropy_out": entropy_out,
            "work_done": work_done,
            "expected": expected,
            "actual": actual,
            "violation_delta": violation_delta,
            "tolerance": CONSERVATION_TOLERANCE
        })

    return is_valid, violation_delta


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "RECEIPT_SCHEMA",
    "BASE_CAP",
    "SURVIVAL_THRESHOLD",
    "MAX_ENTROPY",
    "CONSERVATION_TOLERANCE",
    # Core functions
    "dynamic_cap",
    "selection_pressure",
    "entropy_budget",
    "hilbert_bound",
    "entropy_conservation",
]
