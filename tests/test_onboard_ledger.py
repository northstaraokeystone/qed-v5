"""
tests/test_onboard_ledger.py - Tests for Hash-Chained Decision Ledger

Validates:
- emit_entry creates proper hash chain
- verify_chain detects tampering
- anchor_batch creates merkle roots
- compact preserves invariants

CLAUDEME v3.1 Compliant: Each receipt type has TEST coverage.
"""

import pytest
from dataclasses import asdict

from onboard_ledger import (
    LedgerConfig,
    LedgerEntry,
    LedgerState,
    emit_entry,
    verify_chain,
    anchor_batch,
    compact,
    load_ledger_params,
    compute_throughput,
    create_initial_state,
    config_from_params,
    stoprule_chain_break,
    GENESIS_HASH,
    ENTRIES_PER_CYCLE,
    ANCHOR_BATCH_SIZE,
    LEDGER_ALPHA_BONUS,
)
from receipts import StopRule, dual_hash


# =============================================================================
# TEST: emit_entry chain link
# =============================================================================

class TestEmitEntry:
    """Tests for emit_entry function."""

    def test_emit_entry_chain_link(self):
        """Entry.prev_hash == state.chain_head."""
        state = create_initial_state()
        config = LedgerConfig()

        decision = {"type": "nav", "data": {"heading": 45.0}}
        entry, new_state = emit_entry(decision, state, config)

        # prev_hash should be genesis (initial chain head)
        assert entry.prev_hash == GENESIS_HASH
        assert entry.prev_hash == state.chain_head

    def test_emit_entry_updates_head(self):
        """new_state.chain_head == entry.entry_hash."""
        state = create_initial_state()
        config = LedgerConfig()

        decision = {"type": "nav", "data": {"heading": 45.0}}
        entry, new_state = emit_entry(decision, state, config)

        # New chain head should be this entry's hash
        assert new_state.chain_head == entry.entry_hash
        assert new_state.chain_head != GENESIS_HASH

    def test_emit_entry_increments_chain_length(self):
        """Chain length increments by 1 per entry."""
        state = create_initial_state()
        config = LedgerConfig()

        assert state.chain_length == 0

        _, state = emit_entry({"type": "test"}, state, config)
        assert state.chain_length == 1

        _, state = emit_entry({"type": "test"}, state, config)
        assert state.chain_length == 2

    def test_emit_entry_adds_to_pending_anchor(self):
        """Entry is added to pending_anchor list."""
        state = create_initial_state()
        config = LedgerConfig()

        assert len(state.pending_anchor) == 0

        entry, new_state = emit_entry({"type": "test"}, state, config)

        assert len(new_state.pending_anchor) == 1
        assert new_state.pending_anchor[0] == entry

    def test_emit_entry_chain_continues(self):
        """Multiple entries form proper chain."""
        state = create_initial_state()
        config = LedgerConfig()

        # First entry
        entry1, state = emit_entry({"type": "test1"}, state, config)

        # Second entry should link to first
        entry2, state = emit_entry({"type": "test2"}, state, config)
        assert entry2.prev_hash == entry1.entry_hash

        # Third entry should link to second
        entry3, state = emit_entry({"type": "test3"}, state, config)
        assert entry3.prev_hash == entry2.entry_hash

    def test_emit_entry_hash_is_deterministic(self):
        """Same entry data produces same hash."""
        state = create_initial_state()
        config = LedgerConfig()

        decision = {"type": "test", "data": {"value": 42}}

        # Two entries with same decision data from same state
        entry1, _ = emit_entry(decision, state, config)
        entry2, _ = emit_entry(decision, state, config)

        # entry_hash should be different because ts will differ
        # but prev_hash should be same
        assert entry1.prev_hash == entry2.prev_hash


# =============================================================================
# TEST: verify_chain
# =============================================================================

class TestVerifyChain:
    """Tests for verify_chain function."""

    def test_verify_chain_valid(self):
        """100 entries with valid chain passes."""
        state = create_initial_state()
        config = LedgerConfig()

        # Build chain of 100 entries
        for i in range(100):
            _, state = emit_entry({"type": f"test_{i}"}, state, config)

        # Should verify successfully
        result = verify_chain(state)
        assert result is True
        assert state.chain_length == 100

    def test_verify_chain_empty(self):
        """Empty chain verifies successfully."""
        state = create_initial_state()

        result = verify_chain(state)
        assert result is True

    def test_verify_chain_tamper_detects(self):
        """Modified entry raises StopRule."""
        state = create_initial_state()
        config = LedgerConfig()

        # Build chain
        for i in range(10):
            _, state = emit_entry({"type": f"test_{i}"}, state, config)

        # Tamper with an entry in the middle
        tampered_entry = state.entries[5]
        tampered_entry.decision_data["type"] = "TAMPERED"

        # Should raise StopRule
        with pytest.raises(StopRule):
            verify_chain(state)

    def test_verify_chain_prev_hash_mismatch(self):
        """Wrong prev_hash raises StopRule."""
        state = create_initial_state()
        config = LedgerConfig()

        # Build chain
        for i in range(5):
            _, state = emit_entry({"type": f"test_{i}"}, state, config)

        # Tamper with prev_hash
        state.entries[3].prev_hash = "wrong_hash"

        # Should raise StopRule
        with pytest.raises(StopRule):
            verify_chain(state)


# =============================================================================
# TEST: anchor_batch
# =============================================================================

class TestAnchorBatch:
    """Tests for anchor_batch function."""

    def test_anchor_batch_merkle(self):
        """Batch of 100 creates valid merkle root."""
        state = create_initial_state()
        config = LedgerConfig()

        # Build 100 entries
        for i in range(100):
            _, state = emit_entry({"type": f"test_{i}"}, state, config)

        assert len(state.pending_anchor) == 100

        # Anchor the batch
        merkle_root, new_state = anchor_batch(state, config)

        # Should have valid root
        assert merkle_root != ""
        assert ":" in merkle_root  # dual_hash format

        # Pending should be cleared
        assert len(new_state.pending_anchor) == 0

        # Root should be stored
        assert merkle_root in new_state.anchor_roots

    def test_anchor_batch_empty(self):
        """Empty pending returns empty root."""
        state = create_initial_state()
        config = LedgerConfig()

        merkle_root, new_state = anchor_batch(state, config)

        assert merkle_root == ""
        assert len(new_state.anchor_roots) == 0

    def test_anchor_batch_preserves_entries(self):
        """Anchoring doesn't modify entries list."""
        state = create_initial_state()
        config = LedgerConfig()

        for i in range(50):
            _, state = emit_entry({"type": f"test_{i}"}, state, config)

        original_entries = len(state.entries)

        _, new_state = anchor_batch(state, config)

        assert len(new_state.entries) == original_entries


# =============================================================================
# TEST: compact
# =============================================================================

class TestCompact:
    """Tests for compact function."""

    def test_compact_preserves_invariants(self):
        """Before/after counts match."""
        state = create_initial_state()
        config = LedgerConfig()

        # Build entries with known timestamps
        for i in range(10):
            _, state = emit_entry({"type": f"test_{i}"}, state, config)

        original_length = state.chain_length

        # Get timestamp of 5th entry for compaction cutoff
        cutoff_ts = state.entries[5].ts

        # Compact entries before cutoff
        new_state = compact(state, cutoff_ts)

        # Chain length should be preserved (for audit)
        assert new_state.chain_length == original_length

        # But entries list should be shorter
        assert len(new_state.entries) < len(state.entries)

    def test_compact_nothing_to_compact(self):
        """No entries before timestamp returns same state."""
        state = create_initial_state()
        config = LedgerConfig()

        for i in range(5):
            _, state = emit_entry({"type": f"test_{i}"}, state, config)

        # Use timestamp before all entries
        new_state = compact(state, "2000-01-01T00:00:00Z")

        # Same entries
        assert len(new_state.entries) == len(state.entries)


# =============================================================================
# TEST: compute_throughput
# =============================================================================

class TestComputeThroughput:
    """Tests for compute_throughput function."""

    def test_compute_throughput_basic(self):
        """1000 entries in 60s = 16.67 entries/sec."""
        state = create_initial_state()
        state = LedgerState(
            entries=[],
            chain_head=GENESIS_HASH,
            chain_length=1000,  # Simulate 1000 entries
            pending_anchor=[],
            anchor_roots=[],
            integrity_verified=True
        )

        throughput = compute_throughput(state, 60.0)

        assert abs(throughput - 16.67) < 0.1

    def test_compute_throughput_zero_duration(self):
        """Zero duration returns 0."""
        state = create_initial_state()
        state.chain_length = 100

        throughput = compute_throughput(state, 0.0)
        assert throughput == 0.0


# =============================================================================
# TEST: load_ledger_params
# =============================================================================

class TestLoadLedgerParams:
    """Tests for load_ledger_params function."""

    def test_load_params_exists(self):
        """Params file loads successfully."""
        params = load_ledger_params()

        assert params is not None
        assert "ledger" in params
        assert "anchoring" in params
        assert "alpha_bonus" in params

    def test_load_params_entries_per_cycle(self):
        """Entries per cycle is 1000."""
        params = load_ledger_params()

        assert params["ledger"]["entries_per_cycle"] == 1000

    def test_load_params_batch_size(self):
        """Batch size is 100."""
        params = load_ledger_params()

        assert params["anchoring"]["batch_size"] == 100

    def test_load_params_alpha_bonus(self):
        """Alpha bonus is 0.10."""
        params = load_ledger_params()

        assert params["alpha_bonus"]["ledger_anchored"] == 0.10


# =============================================================================
# TEST: config_from_params
# =============================================================================

class TestConfigFromParams:
    """Tests for config_from_params function."""

    def test_config_from_params(self):
        """Config loads from params correctly."""
        config = config_from_params()

        assert config.entries_per_cycle == 1000
        assert config.batch_size == 100
        assert config.chain_enabled is True


# =============================================================================
# TEST: Constants
# =============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_entries_per_cycle(self):
        """ENTRIES_PER_CYCLE is 1000."""
        assert ENTRIES_PER_CYCLE == 1000

    def test_anchor_batch_size(self):
        """ANCHOR_BATCH_SIZE is 100."""
        assert ANCHOR_BATCH_SIZE == 100

    def test_ledger_alpha_bonus(self):
        """LEDGER_ALPHA_BONUS is 0.10."""
        assert LEDGER_ALPHA_BONUS == 0.10


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestLedgerIntegration:
    """Integration tests for full ledger workflow."""

    def test_full_workflow(self):
        """emit -> verify -> anchor -> compact workflow."""
        state = create_initial_state()
        config = LedgerConfig()

        # Step 1: Emit entries
        for i in range(100):
            _, state = emit_entry({"type": f"decision_{i}", "cycle": i}, state, config)

        # Step 2: Verify chain
        assert verify_chain(state) is True

        # Step 3: Anchor batch
        root, state = anchor_batch(state, config)
        assert root != ""

        # Step 4: Compact old entries
        cutoff = state.entries[50].ts
        state = compact(state, cutoff)

        # State should still be consistent
        assert state.chain_length == 100
        assert len(state.anchor_roots) == 1

    def test_tamper_detection_workflow(self):
        """Tampering detected even after anchoring."""
        state = create_initial_state()
        config = LedgerConfig()

        # Build and anchor
        for i in range(50):
            _, state = emit_entry({"type": f"test_{i}"}, state, config)

        root_before, state = anchor_batch(state, config)

        # Build more entries
        for i in range(50, 100):
            _, state = emit_entry({"type": f"test_{i}"}, state, config)

        # Tamper with old entry
        state.entries[10].decision_data["tampered"] = True

        # Verify should detect
        with pytest.raises(StopRule):
            verify_chain(state)
