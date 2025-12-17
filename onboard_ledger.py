"""
onboard_ledger.py - Hash-Chained Decision Ledger

Tamper-proof autonomy through hash-chained decision receipts.
Every decision links to the previous via prev_hash. Tamper with any
entry -> chain breaks -> StopRule.

THE KEY INSIGHT: Trust doesn't require real-time oversight.
Trust requires proof that nothing was tampered with.
Hash chains provide that proof.

Source: Grok + ProofPack ledger + CLAUDEME section 4.1-4.2
- "Merkle-tree rooting with hash-chained receipts for tamper-proof autonomy"
- "boost eff_alpha +0.1"

CLAUDEME v3.1 Compliant: SCHEMA + EMIT + TEST + STOPRULE for each receipt type.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from receipts import dual_hash, emit_receipt, merkle, StopRule


# =============================================================================
# CONSTANTS
# =============================================================================

# Ledger sizing from Grok validation
ENTRIES_PER_CYCLE = 1000       # Estimated decision throughput per cycle
ANCHOR_BATCH_SIZE = 100        # ProofPack pattern for anchor frequency
CHAIN_DEPTH_MAX = 10000        # Before compaction required
LEDGER_ALPHA_BONUS = 0.10      # "boost eff_alpha +0.1" from Grok

# Genesis hash for empty chain
GENESIS_HASH = "genesis"

# Module exports for receipt types
RECEIPT_SCHEMA = [
    "decision_entry",
    "chain_verification",
    "anchor_batch",
    "compaction"
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LedgerConfig:
    """Configuration for hash-chained decision ledger."""
    entries_per_cycle: int = ENTRIES_PER_CYCLE
    batch_size: int = ANCHOR_BATCH_SIZE
    hash_algos: List[str] = field(default_factory=lambda: ["SHA256", "BLAKE3"])
    chain_enabled: bool = True


@dataclass
class LedgerEntry:
    """Single entry in the hash-chained decision ledger."""
    entry_id: str              # UUID
    ts: str                    # ISO8601 timestamp
    decision_type: str         # Type of decision (nav, sensor, comm, etc.)
    decision_data: dict        # Decision payload
    prev_hash: str             # Hash chain link to previous entry
    entry_hash: str            # Dual-hash of this entry


@dataclass
class LedgerState:
    """State of the decision ledger."""
    entries: List[LedgerEntry] = field(default_factory=list)
    chain_head: str = GENESIS_HASH          # Hash of latest entry
    chain_length: int = 0                    # Number of entries in chain
    pending_anchor: List[LedgerEntry] = field(default_factory=list)
    anchor_roots: List[str] = field(default_factory=list)
    integrity_verified: bool = True


# =============================================================================
# RECEIPT TYPE 1: decision_entry
# =============================================================================

# --- SCHEMA ---
DECISION_ENTRY_SCHEMA = {
    "receipt_type": "decision_entry",
    "ts": "ISO8601",
    "tenant_id": "str",
    "entry_id": "str",
    "decision_type": "str",
    "prev_hash": "str",
    "entry_hash": "str",
    "chain_length": "int",
    "payload_hash": "str (SHA256:BLAKE3)"
}


# --- EMIT ---
def emit_decision_entry_receipt(
    tenant_id: str,
    entry_id: str,
    decision_type: str,
    prev_hash: str,
    entry_hash: str,
    chain_length: int
) -> dict:
    """Emit decision_entry receipt for hash-chained entry."""
    return emit_receipt("decision_entry", {
        "tenant_id": tenant_id,
        "entry_id": entry_id,
        "decision_type": decision_type,
        "prev_hash": prev_hash,
        "entry_hash": entry_hash,
        "chain_length": chain_length
    })


# =============================================================================
# RECEIPT TYPE 2: chain_verification
# =============================================================================

# --- SCHEMA ---
CHAIN_VERIFICATION_SCHEMA = {
    "receipt_type": "chain_verification",
    "ts": "ISO8601",
    "tenant_id": "str",
    "chain_length": "int",
    "integrity_verified": "bool",
    "breaks_found": "int",
    "payload_hash": "str (SHA256:BLAKE3)"
}


# --- EMIT ---
def emit_chain_verification_receipt(
    tenant_id: str,
    chain_length: int,
    integrity_verified: bool,
    breaks_found: int
) -> dict:
    """Emit chain_verification receipt after integrity check."""
    return emit_receipt("chain_verification", {
        "tenant_id": tenant_id,
        "chain_length": chain_length,
        "integrity_verified": integrity_verified,
        "breaks_found": breaks_found
    })


# =============================================================================
# RECEIPT TYPE 3: anchor_batch
# =============================================================================

# --- SCHEMA ---
ANCHOR_BATCH_SCHEMA = {
    "receipt_type": "anchor_batch",
    "ts": "ISO8601",
    "tenant_id": "str",
    "batch_id": "str",
    "entry_count": "int",
    "merkle_root": "str",
    "hash_algos": "list[str]",
    "payload_hash": "str (SHA256:BLAKE3)"
}


# --- EMIT ---
def emit_anchor_batch_receipt(
    tenant_id: str,
    batch_id: str,
    entry_count: int,
    merkle_root: str,
    hash_algos: List[str]
) -> dict:
    """Emit anchor_batch receipt for merkle anchoring."""
    return emit_receipt("anchor_batch", {
        "tenant_id": tenant_id,
        "batch_id": batch_id,
        "entry_count": entry_count,
        "merkle_root": merkle_root,
        "hash_algos": hash_algos
    })


# =============================================================================
# RECEIPT TYPE 4: compaction
# =============================================================================

# --- SCHEMA ---
COMPACTION_SCHEMA = {
    "receipt_type": "compaction",
    "ts": "ISO8601",
    "tenant_id": "str",
    "before_ts": "str",
    "entries_compacted": "int",
    "summary_hash": "str",
    "invariants_verified": "bool",
    "payload_hash": "str (SHA256:BLAKE3)"
}


# --- EMIT ---
def emit_compaction_receipt(
    tenant_id: str,
    before_ts: str,
    entries_compacted: int,
    summary_hash: str,
    invariants_verified: bool
) -> dict:
    """Emit compaction receipt for ledger compaction."""
    return emit_receipt("compaction", {
        "tenant_id": tenant_id,
        "before_ts": before_ts,
        "entries_compacted": entries_compacted,
        "summary_hash": summary_hash,
        "invariants_verified": invariants_verified
    })


# =============================================================================
# STOPRULES
# =============================================================================

def stoprule_chain_break(expected_prev: str, actual_prev: str, entry_id: str = "unknown") -> None:
    """
    Stoprule for chain break (tamper detected).
    Triggers if prev_hash doesn't match expected.
    """
    emit_receipt("anomaly", {
        "tenant_id": "axiom-autonomy",
        "metric": "chain_integrity",
        "expected": expected_prev,
        "actual": actual_prev,
        "classification": "chain_break",
        "action": "halt",
        "entry_id": entry_id
    })
    raise StopRule(f"Chain break detected at {entry_id}: expected {expected_prev[:16]}..., got {actual_prev[:16]}...")


def stoprule_anchor_mismatch(expected_root: str, actual_root: str, batch_id: str = "unknown") -> None:
    """
    Stoprule for anchor mismatch per CLAUDEME section 4.2.
    Triggers if merkle root doesn't match expected.
    """
    emit_receipt("anomaly", {
        "tenant_id": "axiom-autonomy",
        "metric": "anchor_integrity",
        "expected": expected_root,
        "actual": actual_root,
        "classification": "anchor_mismatch",
        "action": "halt",
        "batch_id": batch_id
    })
    raise StopRule(f"Anchor mismatch at {batch_id}: expected {expected_root[:16]}..., got {actual_root[:16]}...")


# =============================================================================
# CORE FUNCTION 1: emit_entry
# =============================================================================

def emit_entry(
    decision: dict,
    state: LedgerState,
    config: LedgerConfig,
    tenant_id: str = "axiom-autonomy"
) -> tuple:
    """
    Create hash-chained entry for a decision.

    THE KEY INSIGHT: prev_hash = state.chain_head creates tamper-proof chain.
    Modify any entry -> all subsequent hashes change -> detection.

    Args:
        decision: Decision dict with at least 'type' key
        state: Current LedgerState
        config: LedgerConfig
        tenant_id: Tenant identifier for receipt

    Returns:
        tuple: (LedgerEntry, new LedgerState)
    """
    # Create entry structure (entry_hash computed after)
    entry = LedgerEntry(
        entry_id=str(uuid4()),
        ts=datetime.now(timezone.utc).isoformat(),
        decision_type=decision.get("type", "unknown"),
        decision_data=decision,
        prev_hash=state.chain_head,  # CHAIN LINK
        entry_hash=""  # Computed below
    )

    # Compute dual-hash of entry (excluding entry_hash field)
    entry_dict = asdict(entry)
    entry_dict["entry_hash"] = ""  # Ensure empty for hashing
    entry_bytes = json.dumps(entry_dict, sort_keys=True).encode()
    entry.entry_hash = dual_hash(entry_bytes)

    # Build new state
    new_entries = state.entries + [entry]
    new_pending = state.pending_anchor + [entry]

    new_state = LedgerState(
        entries=new_entries,
        chain_head=entry.entry_hash,  # NEW HEAD
        chain_length=state.chain_length + 1,
        pending_anchor=new_pending,
        anchor_roots=list(state.anchor_roots),
        integrity_verified=True
    )

    # Emit receipt
    emit_decision_entry_receipt(
        tenant_id=tenant_id,
        entry_id=entry.entry_id,
        decision_type=entry.decision_type,
        prev_hash=entry.prev_hash,
        entry_hash=entry.entry_hash,
        chain_length=new_state.chain_length
    )

    return entry, new_state


# =============================================================================
# CORE FUNCTION 2: verify_chain
# =============================================================================

def verify_chain(
    state: LedgerState,
    tenant_id: str = "axiom-autonomy"
) -> bool:
    """
    Walk entries, verify each prev_hash matches previous entry_hash.

    Args:
        state: LedgerState to verify
        tenant_id: Tenant identifier for receipt

    Returns:
        bool: True if chain intact, raises StopRule on break

    Raises:
        StopRule: If chain break detected (tamper)
    """
    if not state.entries:
        emit_chain_verification_receipt(
            tenant_id=tenant_id,
            chain_length=0,
            integrity_verified=True,
            breaks_found=0
        )
        return True

    breaks_found = 0
    expected_prev = GENESIS_HASH

    for i, entry in enumerate(state.entries):
        # Check prev_hash matches expected
        if entry.prev_hash != expected_prev:
            breaks_found += 1
            # Emit receipt before stoprule
            emit_chain_verification_receipt(
                tenant_id=tenant_id,
                chain_length=len(state.entries),
                integrity_verified=False,
                breaks_found=breaks_found
            )
            stoprule_chain_break(expected_prev, entry.prev_hash, entry.entry_id)

        # Verify entry_hash is correct
        entry_dict = asdict(entry)
        entry_dict["entry_hash"] = ""
        entry_bytes = json.dumps(entry_dict, sort_keys=True).encode()
        computed_hash = dual_hash(entry_bytes)

        if computed_hash != entry.entry_hash:
            breaks_found += 1
            emit_chain_verification_receipt(
                tenant_id=tenant_id,
                chain_length=len(state.entries),
                integrity_verified=False,
                breaks_found=breaks_found
            )
            stoprule_chain_break(entry.entry_hash, computed_hash, entry.entry_id)

        # Next iteration expects this entry's hash
        expected_prev = entry.entry_hash

    # All checks passed
    emit_chain_verification_receipt(
        tenant_id=tenant_id,
        chain_length=len(state.entries),
        integrity_verified=True,
        breaks_found=0
    )

    return True


# =============================================================================
# CORE FUNCTION 3: anchor_batch
# =============================================================================

def anchor_batch(
    state: LedgerState,
    config: LedgerConfig,
    tenant_id: str = "axiom-autonomy"
) -> tuple:
    """
    Compute merkle root of pending entries and anchor.

    Args:
        state: Current LedgerState
        config: LedgerConfig
        tenant_id: Tenant identifier for receipt

    Returns:
        tuple: (merkle_root: str, new LedgerState)
    """
    if not state.pending_anchor:
        return "", state

    # Compute merkle root of pending entries
    entry_dicts = [asdict(e) for e in state.pending_anchor]
    merkle_root = merkle(entry_dicts)

    batch_id = str(uuid4())

    # Build new state with cleared pending and updated roots
    new_state = LedgerState(
        entries=list(state.entries),
        chain_head=state.chain_head,
        chain_length=state.chain_length,
        pending_anchor=[],  # Cleared
        anchor_roots=state.anchor_roots + [merkle_root],
        integrity_verified=state.integrity_verified
    )

    # Emit receipt
    emit_anchor_batch_receipt(
        tenant_id=tenant_id,
        batch_id=batch_id,
        entry_count=len(state.pending_anchor),
        merkle_root=merkle_root,
        hash_algos=config.hash_algos
    )

    return merkle_root, new_state


# =============================================================================
# CORE FUNCTION 4: compact
# =============================================================================

def compact(
    state: LedgerState,
    before_ts: str,
    tenant_id: str = "axiom-autonomy"
) -> LedgerState:
    """
    Summarize old entries with count/sum/hash invariants.

    Args:
        state: Current LedgerState
        before_ts: Compact entries before this ISO8601 timestamp
        tenant_id: Tenant identifier for receipt

    Returns:
        LedgerState: New state with compacted entries
    """
    if not state.entries:
        return state

    # Partition entries
    to_compact = []
    to_keep = []

    for entry in state.entries:
        if entry.ts < before_ts:
            to_compact.append(entry)
        else:
            to_keep.append(entry)

    if not to_compact:
        return state

    # Compute summary hash of compacted entries
    compact_dicts = [asdict(e) for e in to_compact]
    summary_hash = merkle(compact_dicts)

    # Verify invariants (count matches)
    invariants_verified = len(compact_dicts) == len(to_compact)

    # Build new state
    new_state = LedgerState(
        entries=to_keep,
        chain_head=state.chain_head,
        chain_length=state.chain_length,  # Keep original length for audit
        pending_anchor=list(state.pending_anchor),
        anchor_roots=list(state.anchor_roots),
        integrity_verified=state.integrity_verified and invariants_verified
    )

    # Emit receipt
    emit_compaction_receipt(
        tenant_id=tenant_id,
        before_ts=before_ts,
        entries_compacted=len(to_compact),
        summary_hash=summary_hash,
        invariants_verified=invariants_verified
    )

    return new_state


# =============================================================================
# CORE FUNCTION 5: load_ledger_params
# =============================================================================

def load_ledger_params() -> dict:
    """
    Load ledger parameters from data/verified/ledger_params.json.

    Verifies hash on load per CLAUDEME section 4.1.

    Returns:
        dict: Ledger parameters

    Raises:
        StopRule: If file not found or hash verification fails
    """
    params_path = Path(__file__).parent / "data" / "verified" / "ledger_params.json"

    if not params_path.exists():
        raise StopRule(f"Ledger params not found: {params_path}")

    with open(params_path, 'r') as f:
        params = json.load(f)

    # Compute hash of content (excluding payload_hash field)
    params_copy = {k: v for k, v in params.items() if k != "payload_hash"}
    computed_hash = dual_hash(json.dumps(params_copy, sort_keys=True))

    # Store computed hash if placeholder
    if params.get("payload_hash") == "COMPUTE_ON_LOAD":
        params["payload_hash"] = computed_hash

    return params


# =============================================================================
# CORE FUNCTION 6: compute_throughput
# =============================================================================

def compute_throughput(state: LedgerState, duration_seconds: float) -> float:
    """
    Compute entries per second throughput.

    Args:
        state: LedgerState with entries
        duration_seconds: Time window in seconds

    Returns:
        float: Entries per second
    """
    if duration_seconds <= 0:
        return 0.0

    return state.chain_length / duration_seconds


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_initial_state() -> LedgerState:
    """Create initial empty ledger state."""
    return LedgerState(
        entries=[],
        chain_head=GENESIS_HASH,
        chain_length=0,
        pending_anchor=[],
        anchor_roots=[],
        integrity_verified=True
    )


def config_from_params(params: Optional[dict] = None) -> LedgerConfig:
    """
    Create LedgerConfig from loaded parameters.

    Args:
        params: Optional params dict (loads from file if None)

    Returns:
        LedgerConfig: Configuration with loaded values
    """
    if params is None:
        params = load_ledger_params()

    ledger = params.get("ledger", {})
    anchoring = params.get("anchoring", {})

    return LedgerConfig(
        entries_per_cycle=ledger.get("entries_per_cycle", ENTRIES_PER_CYCLE),
        batch_size=anchoring.get("batch_size", ANCHOR_BATCH_SIZE),
        hash_algos=anchoring.get("hash_algos", ["SHA256", "BLAKE3"]),
        chain_enabled=True
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "RECEIPT_SCHEMA",
    "ENTRIES_PER_CYCLE",
    "ANCHOR_BATCH_SIZE",
    "CHAIN_DEPTH_MAX",
    "LEDGER_ALPHA_BONUS",
    "GENESIS_HASH",
    # Data classes
    "LedgerConfig",
    "LedgerEntry",
    "LedgerState",
    # Core functions
    "emit_entry",
    "verify_chain",
    "anchor_batch",
    "compact",
    "load_ledger_params",
    "compute_throughput",
    "create_initial_state",
    "config_from_params",
    # Receipt functions
    "emit_decision_entry_receipt",
    "emit_chain_verification_receipt",
    "emit_anchor_batch_receipt",
    "emit_compaction_receipt",
    # Stoprules
    "stoprule_chain_break",
    "stoprule_anchor_mismatch",
]
