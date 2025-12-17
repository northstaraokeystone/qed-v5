"""
tests/test_volume_stress.py - Volume Stress Tests for Ledger

Validates:
- 100 entries/cycle integrity
- 1000 entries/cycle integrity
- 10000 entries/cycle integrity
- Linear scaling with volume
- Memory bounded at high volume

Source: Grok - "test with variable data volumes"
"""

import pytest
import time
import sys

from onboard_ledger import (
    LedgerConfig,
    LedgerState,
    emit_entry,
    verify_chain,
    anchor_batch,
    create_initial_state,
    compute_throughput,
    GENESIS_HASH,
)
from merkle_forest import (
    ForestConfig,
    anchor_distributed,
    verify_quorum,
)


# =============================================================================
# TEST: Volume 100
# =============================================================================

class TestVolume100:
    """Tests for 100 entries/cycle volume."""

    def test_volume_100_integrity(self):
        """100 entries/cycle -> integrity 100%."""
        state = create_initial_state()
        config = LedgerConfig(entries_per_cycle=100)

        # Emit 100 entries
        for i in range(100):
            _, state = emit_entry({"type": "decision", "cycle": i}, state, config)

        # Verify chain integrity
        assert verify_chain(state) is True
        assert state.chain_length == 100

    def test_volume_100_anchoring(self):
        """100 entries anchor successfully."""
        state = create_initial_state()
        config = LedgerConfig(entries_per_cycle=100)

        for i in range(100):
            _, state = emit_entry({"type": "test"}, state, config)

        root, state = anchor_batch(state, config)

        assert root != ""
        assert len(state.pending_anchor) == 0


# =============================================================================
# TEST: Volume 1000
# =============================================================================

class TestVolume1000:
    """Tests for 1000 entries/cycle volume."""

    def test_volume_1000_integrity(self):
        """1000 entries/cycle -> integrity 100%."""
        state = create_initial_state()
        config = LedgerConfig(entries_per_cycle=1000)

        # Emit 1000 entries
        for i in range(1000):
            _, state = emit_entry({"type": "decision", "cycle": i}, state, config)

        # Verify chain integrity
        assert verify_chain(state) is True
        assert state.chain_length == 1000

    def test_volume_1000_distributed_anchor(self):
        """1000 entries anchor to distributed forest."""
        state = create_initial_state()
        ledger_config = LedgerConfig(entries_per_cycle=1000)
        forest_config = ForestConfig()

        for i in range(1000):
            _, state = emit_entry({"type": "test", "id": i}, state, ledger_config)

        # Create distributed anchor
        entries = [{"id": e.entry_id, "hash": e.entry_hash} for e in state.entries]
        forest = anchor_distributed(entries, forest_config)

        assert len(forest.roots) == 3
        assert verify_quorum(forest, forest_config) is True


# =============================================================================
# TEST: Volume 10000
# =============================================================================

class TestVolume10000:
    """Tests for 10000 entries/cycle volume."""

    @pytest.mark.slow
    def test_volume_10000_integrity(self):
        """10000 entries/cycle -> integrity 100%."""
        state = create_initial_state()
        config = LedgerConfig(entries_per_cycle=10000)

        # Emit 10000 entries
        for i in range(10000):
            _, state = emit_entry({"type": "decision", "cycle": i}, state, config)

        # Verify chain integrity
        assert verify_chain(state) is True
        assert state.chain_length == 10000

    @pytest.mark.slow
    def test_volume_10000_distributed_anchor(self):
        """10000 entries anchor to distributed forest."""
        state = create_initial_state()
        ledger_config = LedgerConfig(entries_per_cycle=10000)
        forest_config = ForestConfig()

        for i in range(10000):
            _, state = emit_entry({"type": "test"}, state, ledger_config)

        entries = [{"id": e.entry_id, "hash": e.entry_hash} for e in state.entries]
        forest = anchor_distributed(entries, forest_config)

        assert len(forest.roots) == 3
        assert forest.quorum_achieved is True


# =============================================================================
# TEST: Throughput Scaling
# =============================================================================

class TestThroughputScaling:
    """Tests for throughput scaling with volume."""

    def test_throughput_scaling(self):
        """Throughput scales linearly with volume."""
        config = LedgerConfig()

        # Test at different volumes
        volumes = [100, 500, 1000]
        throughputs = []

        for volume in volumes:
            state = create_initial_state()
            start_time = time.time()

            for i in range(volume):
                _, state = emit_entry({"type": "test"}, state, config)

            elapsed = time.time() - start_time
            throughput = volume / elapsed if elapsed > 0 else float('inf')
            throughputs.append(throughput)

        # Throughput should be roughly consistent (within 50% variance)
        if len(throughputs) >= 2:
            avg_throughput = sum(throughputs) / len(throughputs)
            for t in throughputs:
                assert t > avg_throughput * 0.5, f"Throughput {t} below 50% of average {avg_throughput}"

    def test_throughput_1000_in_60s(self):
        """1000 entries complete well under 60s."""
        state = create_initial_state()
        config = LedgerConfig()

        start_time = time.time()

        for i in range(1000):
            _, state = emit_entry({"type": "stress_test"}, state, config)

        elapsed = time.time() - start_time

        # Should complete in under 60 seconds (much faster in practice)
        assert elapsed < 60.0

        # Calculate throughput
        throughput = compute_throughput(state, elapsed)
        assert throughput > 10  # At least 10 entries/sec


# =============================================================================
# TEST: Memory Bounded
# =============================================================================

class TestMemoryBounded:
    """Tests for memory constraints at high volume."""

    def test_memory_bounded_1000(self):
        """Peak memory reasonable at 1000 entries."""
        import gc

        gc.collect()
        initial_size = sys.getsizeof([])

        state = create_initial_state()
        config = LedgerConfig()

        for i in range(1000):
            _, state = emit_entry({"type": "test", "data": {"value": i}}, state, config)

        # Check state size is reasonable (not exact memory but indicative)
        entries_size = sum(sys.getsizeof(str(e)) for e in state.entries[:10]) * 100

        # Should be under 10MB worth of entry data
        assert entries_size < 10 * 1024 * 1024

    def test_memory_bounded_with_anchoring(self):
        """Memory stays bounded when anchoring clears pending."""
        state = create_initial_state()
        config = LedgerConfig(batch_size=100)

        # Emit and anchor in batches
        for batch in range(10):
            for i in range(100):
                _, state = emit_entry({"type": "test"}, state, config)

            # Anchor clears pending
            _, state = anchor_batch(state, config)

            # Pending should be empty after anchor
            assert len(state.pending_anchor) == 0


# =============================================================================
# TEST: Variable Volume Range
# =============================================================================

class TestVariableVolumeRange:
    """Tests across the full volume range (100-10000)."""

    @pytest.mark.parametrize("volume", [100, 500, 1000, 2000])
    def test_volume_integrity_parametrized(self, volume):
        """Integrity maintained at various volumes."""
        state = create_initial_state()
        config = LedgerConfig(entries_per_cycle=volume)

        for i in range(volume):
            _, state = emit_entry({"type": "test", "id": i}, state, config)

        assert verify_chain(state) is True
        assert state.chain_length == volume

    @pytest.mark.parametrize("volume", [100, 500, 1000])
    def test_distributed_anchor_parametrized(self, volume):
        """Distributed anchoring works at various volumes."""
        state = create_initial_state()
        ledger_config = LedgerConfig()
        forest_config = ForestConfig()

        for i in range(volume):
            _, state = emit_entry({"type": "test"}, state, ledger_config)

        entries = [{"id": e.entry_id} for e in state.entries]
        forest = anchor_distributed(entries, forest_config)

        assert len(forest.roots) == 3
        assert forest.quorum_achieved is True


# =============================================================================
# INTEGRATION: Full Volume Stress
# =============================================================================

class TestVolumeStressIntegration:
    """Full integration stress tests."""

    def test_full_stress_cycle(self):
        """Complete stress cycle: emit -> verify -> anchor -> forest."""
        state = create_initial_state()
        ledger_config = LedgerConfig(entries_per_cycle=1000, batch_size=100)
        forest_config = ForestConfig()

        # Emit entries in batches, anchor periodically
        anchor_roots = []

        for batch in range(10):
            # Emit 100 entries
            for i in range(100):
                _, state = emit_entry(
                    {"type": "decision", "batch": batch, "seq": i},
                    state,
                    ledger_config
                )

            # Anchor this batch
            root, state = anchor_batch(state, ledger_config)
            if root:
                anchor_roots.append(root)

        # Final verification
        assert verify_chain(state) is True
        assert state.chain_length == 1000
        assert len(anchor_roots) == 10

        # Create distributed forest from all entries
        entries = [{"id": e.entry_id, "type": e.decision_type} for e in state.entries]
        forest = anchor_distributed(entries, forest_config)

        assert forest.quorum_achieved is True
        assert len(forest.roots) == 3

    def test_stress_with_simulated_blackout(self):
        """Stress test with simulated node loss during blackout."""
        from merkle_forest import simulate_node_loss

        state = create_initial_state()
        ledger_config = LedgerConfig(entries_per_cycle=500)
        forest_config = ForestConfig(n_roots=3, quorum=2)

        # Build ledger
        for i in range(500):
            _, state = emit_entry({"type": "nav", "cycle": i}, state, ledger_config)

        # Create distributed anchor
        entries = [{"id": e.entry_id} for e in state.entries]
        forest = anchor_distributed(entries, forest_config)

        # Simulate blackout - lose node 0
        forest = simulate_node_loss(forest, 0, forest_config)

        # Quorum should still be met (2/3)
        assert forest.quorum_achieved is True
        assert verify_quorum(forest, forest_config) is True

        # Trust is maintained
        assert len(forest.roots) == 2
