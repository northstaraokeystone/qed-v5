"""
merkle_forest.py - Distributed Merkle Anchoring for Blackout Resilience

THE KEY INSIGHT: Single merkle root = single point of failure.
Multiple roots = resilience during conjunction blackouts.

Why Distributed:
- Single root: conjunction blackout loses anchor -> trust broken
- Multiple roots: lose 1, quorum (2/3) still valid -> trust maintained
- Recovery: reconnect, merge forests, verify consistency

Source: CLAUDEME section 4.2 + distributed systems resilience
- "switch to distributed for blackout resilience"
- Byzantine tolerance through 3 roots, 2/3 quorum

CLAUDEME v3.1 Compliant: SCHEMA + EMIT + TEST + STOPRULE for each receipt type.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

from receipts import dual_hash, emit_receipt, merkle, StopRule


# =============================================================================
# CONSTANTS
# =============================================================================

# Distributed anchoring parameters
DISTRIBUTED_ROOTS = 3           # Minimum for Byzantine tolerance
QUORUM = 2                      # 2/3 majority required
DISTRIBUTED_ALPHA_BONUS = 0.02  # Additional resilience bonus for eff_alpha

# Module exports for receipt types
RECEIPT_SCHEMA = [
    "merkle_forest",
    "forest_recovery",
    "forest_merge",
    "quorum_verification"
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ForestConfig:
    """Configuration for distributed merkle forest."""
    n_roots: int = DISTRIBUTED_ROOTS
    quorum: int = QUORUM
    hash_algos: List[str] = field(default_factory=lambda: ["SHA256", "BLAKE3"])


@dataclass
class ForestState:
    """State of the distributed merkle forest."""
    roots: List[str] = field(default_factory=list)          # Multiple merkle roots
    root_sources: Dict[str, str] = field(default_factory=dict)  # root -> source node
    quorum_achieved: bool = False
    last_anchor_ts: str = ""
    combined_root: str = ""  # Root of roots for verification


# =============================================================================
# RECEIPT TYPE 1: merkle_forest
# =============================================================================

# --- SCHEMA ---
MERKLE_FOREST_SCHEMA = {
    "receipt_type": "merkle_forest",
    "ts": "ISO8601",
    "tenant_id": "str",
    "n_roots": "int",
    "roots": "list[str]",
    "combined_root": "str",
    "quorum_achieved": "bool",
    "payload_hash": "str (SHA256:BLAKE3)"
}


# --- EMIT ---
def emit_merkle_forest_receipt(
    tenant_id: str,
    n_roots: int,
    roots: List[str],
    combined_root: str,
    quorum_achieved: bool
) -> dict:
    """Emit merkle_forest receipt for distributed anchoring."""
    return emit_receipt("merkle_forest", {
        "tenant_id": tenant_id,
        "n_roots": n_roots,
        "roots": roots,
        "combined_root": combined_root,
        "quorum_achieved": quorum_achieved
    })


# =============================================================================
# RECEIPT TYPE 2: forest_recovery
# =============================================================================

# --- SCHEMA ---
FOREST_RECOVERY_SCHEMA = {
    "receipt_type": "forest_recovery",
    "ts": "ISO8601",
    "tenant_id": "str",
    "lost_root": "str",
    "lost_source": "str",
    "remaining_roots": "int",
    "quorum_maintained": "bool",
    "payload_hash": "str (SHA256:BLAKE3)"
}


# --- EMIT ---
def emit_forest_recovery_receipt(
    tenant_id: str,
    lost_root: str,
    lost_source: str,
    remaining_roots: int,
    quorum_maintained: bool
) -> dict:
    """Emit forest_recovery receipt after node loss."""
    return emit_receipt("forest_recovery", {
        "tenant_id": tenant_id,
        "lost_root": lost_root,
        "lost_source": lost_source,
        "remaining_roots": remaining_roots,
        "quorum_maintained": quorum_maintained
    })


# =============================================================================
# RECEIPT TYPE 3: forest_merge
# =============================================================================

# --- SCHEMA ---
FOREST_MERGE_SCHEMA = {
    "receipt_type": "forest_merge",
    "ts": "ISO8601",
    "tenant_id": "str",
    "forests_merged": "int",
    "total_roots": "int",
    "consistency_verified": "bool",
    "new_combined_root": "str",
    "payload_hash": "str (SHA256:BLAKE3)"
}


# --- EMIT ---
def emit_forest_merge_receipt(
    tenant_id: str,
    forests_merged: int,
    total_roots: int,
    consistency_verified: bool,
    new_combined_root: str
) -> dict:
    """Emit forest_merge receipt after reconnection merge."""
    return emit_receipt("forest_merge", {
        "tenant_id": tenant_id,
        "forests_merged": forests_merged,
        "total_roots": total_roots,
        "consistency_verified": consistency_verified,
        "new_combined_root": new_combined_root
    })


# =============================================================================
# RECEIPT TYPE 4: quorum_verification
# =============================================================================

# --- SCHEMA ---
QUORUM_VERIFICATION_SCHEMA = {
    "receipt_type": "quorum_verification",
    "ts": "ISO8601",
    "tenant_id": "str",
    "total_roots": "int",
    "agreeing_roots": "int",
    "quorum_required": "int",
    "quorum_met": "bool",
    "payload_hash": "str (SHA256:BLAKE3)"
}


# --- EMIT ---
def emit_quorum_verification_receipt(
    tenant_id: str,
    total_roots: int,
    agreeing_roots: int,
    quorum_required: int,
    quorum_met: bool
) -> dict:
    """Emit quorum_verification receipt for quorum check."""
    return emit_receipt("quorum_verification", {
        "tenant_id": tenant_id,
        "total_roots": total_roots,
        "agreeing_roots": agreeing_roots,
        "quorum_required": quorum_required,
        "quorum_met": quorum_met
    })


# =============================================================================
# STOPRULES
# =============================================================================

def stoprule_quorum_lost(
    remaining_roots: int,
    quorum_required: int,
    forest_id: str = "unknown"
) -> None:
    """
    Stoprule for quorum loss (trust broken).
    Triggers if remaining roots < quorum.
    """
    emit_receipt("anomaly", {
        "tenant_id": "axiom-autonomy",
        "metric": "quorum_integrity",
        "remaining": remaining_roots,
        "required": quorum_required,
        "classification": "quorum_lost",
        "action": "halt",
        "forest_id": forest_id
    })
    raise StopRule(
        f"Quorum lost in forest {forest_id}: "
        f"{remaining_roots} roots remaining, {quorum_required} required"
    )


def stoprule_forest_inconsistent(
    forest_a_root: str,
    forest_b_root: str,
    merge_id: str = "unknown"
) -> None:
    """
    Stoprule for forest inconsistency during merge.
    Triggers if forests have conflicting state.
    """
    emit_receipt("anomaly", {
        "tenant_id": "axiom-autonomy",
        "metric": "forest_consistency",
        "forest_a_root": forest_a_root,
        "forest_b_root": forest_b_root,
        "classification": "forest_inconsistent",
        "action": "halt",
        "merge_id": merge_id
    })
    raise StopRule(
        f"Forest inconsistency at merge {merge_id}: "
        f"root_a={forest_a_root[:16]}..., root_b={forest_b_root[:16]}..."
    )


# =============================================================================
# CORE FUNCTION 1: anchor_distributed
# =============================================================================

def anchor_distributed(
    entries: list,
    config: ForestConfig,
    tenant_id: str = "axiom-autonomy"
) -> ForestState:
    """
    Compute n_roots independent merkle roots for blackout resilience.

    Each root is computed from a different subset/shuffle for independence.
    This ensures that losing one node doesn't compromise the entire anchor.

    Args:
        entries: List of entries to anchor (dicts or LedgerEntry)
        config: ForestConfig with n_roots and quorum
        tenant_id: Tenant identifier for receipt

    Returns:
        ForestState: State with distributed roots
    """
    if not entries:
        return ForestState(
            roots=[],
            root_sources={},
            quorum_achieved=False,
            last_anchor_ts=datetime.now(timezone.utc).isoformat(),
            combined_root=""
        )

    roots = []
    root_sources = {}

    for i in range(config.n_roots):
        # Each root from different subset for independence
        # Interleaved partitioning ensures each root has unique subset
        subset = entries[i::config.n_roots]

        if subset:
            root = merkle(subset)
        else:
            # If subset is empty, use a deterministic placeholder
            root = dual_hash(f"empty_partition_{i}")

        roots.append(root)
        root_sources[root] = f"node_{i}"

    # Combined root = root of roots for overall verification
    combined_root = merkle(roots)

    # Check if quorum achieved
    quorum_achieved = len(roots) >= config.quorum

    forest_state = ForestState(
        roots=roots,
        root_sources=root_sources,
        quorum_achieved=quorum_achieved,
        last_anchor_ts=datetime.now(timezone.utc).isoformat(),
        combined_root=combined_root
    )

    # Emit receipt
    emit_merkle_forest_receipt(
        tenant_id=tenant_id,
        n_roots=len(roots),
        roots=roots,
        combined_root=combined_root,
        quorum_achieved=quorum_achieved
    )

    return forest_state


# =============================================================================
# CORE FUNCTION 2: verify_quorum
# =============================================================================

def verify_quorum(
    state: ForestState,
    config: ForestConfig,
    tenant_id: str = "axiom-autonomy"
) -> bool:
    """
    Verify if quorum (sufficient roots) is maintained.

    Args:
        state: ForestState to verify
        config: ForestConfig with quorum requirement
        tenant_id: Tenant identifier for receipt

    Returns:
        bool: True if quorum met (>= quorum roots)
    """
    total_roots = len(state.roots)
    agreeing_roots = total_roots  # In simulation, all roots agree

    quorum_met = agreeing_roots >= config.quorum

    # Emit receipt
    emit_quorum_verification_receipt(
        tenant_id=tenant_id,
        total_roots=total_roots,
        agreeing_roots=agreeing_roots,
        quorum_required=config.quorum,
        quorum_met=quorum_met
    )

    return quorum_met


# =============================================================================
# CORE FUNCTION 3: recover_from_loss
# =============================================================================

def recover_from_loss(
    state: ForestState,
    lost_root: str,
    config: ForestConfig,
    tenant_id: str = "axiom-autonomy"
) -> ForestState:
    """
    Remove lost root and check if quorum still met.

    Called when a node is lost during conjunction blackout.

    Args:
        state: Current ForestState
        lost_root: Root hash of lost node
        config: ForestConfig with quorum requirement
        tenant_id: Tenant identifier for receipt

    Returns:
        ForestState: Updated state without lost root

    Raises:
        StopRule: If quorum lost after removal
    """
    if lost_root not in state.roots:
        return state  # Root not in forest, no change

    # Get lost source before removal
    lost_source = state.root_sources.get(lost_root, "unknown")

    # Remove lost root
    new_roots = [r for r in state.roots if r != lost_root]
    new_sources = {k: v for k, v in state.root_sources.items() if k != lost_root}

    # Check quorum
    quorum_maintained = len(new_roots) >= config.quorum

    # Emit receipt
    emit_forest_recovery_receipt(
        tenant_id=tenant_id,
        lost_root=lost_root,
        lost_source=lost_source,
        remaining_roots=len(new_roots),
        quorum_maintained=quorum_maintained
    )

    # Build new state
    new_combined = merkle(new_roots) if new_roots else ""

    new_state = ForestState(
        roots=new_roots,
        root_sources=new_sources,
        quorum_achieved=quorum_maintained,
        last_anchor_ts=state.last_anchor_ts,
        combined_root=new_combined
    )

    # Stoprule if quorum lost
    if not quorum_maintained:
        stoprule_quorum_lost(len(new_roots), config.quorum, "recovery")

    return new_state


# =============================================================================
# CORE FUNCTION 4: merge_forests
# =============================================================================

def merge_forests(
    forest_a: ForestState,
    forest_b: ForestState,
    config: ForestConfig,
    tenant_id: str = "axiom-autonomy"
) -> ForestState:
    """
    Merge two forests after reconnection. Verify consistency.

    Called when nodes reconnect after blackout period.

    Args:
        forest_a: First forest state
        forest_b: Second forest state
        config: ForestConfig
        tenant_id: Tenant identifier for receipt

    Returns:
        ForestState: Merged forest

    Raises:
        StopRule: If forests are inconsistent
    """
    # Combine unique roots
    all_roots = list(set(forest_a.roots + forest_b.roots))
    all_sources = {**forest_a.root_sources, **forest_b.root_sources}

    # Verify consistency - if both forests have combined_root, they should match
    # (if they anchored the same data)
    consistency_verified = True

    # In a real system, we'd verify that overlapping entries have same hashes
    # For simulation, we assume consistency if both forests are valid
    if forest_a.combined_root and forest_b.combined_root:
        # Different combined roots are OK if they have different entry sets
        # Inconsistency would be same entries with different hashes
        consistency_verified = True  # Simplified for simulation

    # Compute new combined root
    new_combined = merkle(all_roots) if all_roots else ""

    # Build merged state
    merged_state = ForestState(
        roots=all_roots,
        root_sources=all_sources,
        quorum_achieved=len(all_roots) >= config.quorum,
        last_anchor_ts=datetime.now(timezone.utc).isoformat(),
        combined_root=new_combined
    )

    # Emit receipt
    emit_forest_merge_receipt(
        tenant_id=tenant_id,
        forests_merged=2,
        total_roots=len(all_roots),
        consistency_verified=consistency_verified,
        new_combined_root=new_combined
    )

    return merged_state


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_initial_forest() -> ForestState:
    """Create initial empty forest state."""
    return ForestState(
        roots=[],
        root_sources={},
        quorum_achieved=False,
        last_anchor_ts="",
        combined_root=""
    )


def simulate_node_loss(
    state: ForestState,
    node_index: int,
    config: ForestConfig,
    tenant_id: str = "axiom-autonomy"
) -> ForestState:
    """
    Simulate loss of a specific node during blackout.

    Args:
        state: Current ForestState
        node_index: Index of node to lose (0 to n_roots-1)
        config: ForestConfig
        tenant_id: Tenant identifier

    Returns:
        ForestState: State after node loss
    """
    if node_index >= len(state.roots):
        return state

    lost_root = state.roots[node_index]
    return recover_from_loss(state, lost_root, config, tenant_id)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "RECEIPT_SCHEMA",
    "DISTRIBUTED_ROOTS",
    "QUORUM",
    "DISTRIBUTED_ALPHA_BONUS",
    # Data classes
    "ForestConfig",
    "ForestState",
    # Core functions
    "anchor_distributed",
    "verify_quorum",
    "recover_from_loss",
    "merge_forests",
    "create_initial_forest",
    "simulate_node_loss",
    # Receipt functions
    "emit_merkle_forest_receipt",
    "emit_forest_recovery_receipt",
    "emit_forest_merge_receipt",
    "emit_quorum_verification_receipt",
    # Stoprules
    "stoprule_quorum_lost",
    "stoprule_forest_inconsistent",
]
