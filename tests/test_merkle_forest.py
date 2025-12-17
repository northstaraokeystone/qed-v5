"""
tests/test_merkle_forest.py - Tests for Distributed Merkle Forest

Validates:
- anchor_distributed creates multiple independent roots
- verify_quorum checks 2/3 majority
- recover_from_loss handles node loss
- merge_forests combines after reconnection

CLAUDEME v3.1 Compliant: Each function has TEST coverage.
"""

import pytest

from merkle_forest import (
    ForestConfig,
    ForestState,
    anchor_distributed,
    verify_quorum,
    recover_from_loss,
    merge_forests,
    create_initial_forest,
    simulate_node_loss,
    stoprule_quorum_lost,
    DISTRIBUTED_ROOTS,
    QUORUM,
    DISTRIBUTED_ALPHA_BONUS,
)
from receipts import StopRule


# =============================================================================
# TEST: anchor_distributed
# =============================================================================

class TestAnchorDistributed:
    """Tests for anchor_distributed function."""

    def test_anchor_distributed_3_roots(self):
        """Returns 3 independent roots by default."""
        entries = [{"id": i, "data": f"entry_{i}"} for i in range(100)]
        config = ForestConfig()

        forest = anchor_distributed(entries, config)

        # Should have 3 roots
        assert len(forest.roots) == 3
        assert forest.quorum_achieved is True

    def test_anchor_distributed_roots_independent(self):
        """Each root is different (independent subsets)."""
        entries = [{"id": i, "data": f"entry_{i}"} for i in range(100)]
        config = ForestConfig()

        forest = anchor_distributed(entries, config)

        # All roots should be unique
        assert len(set(forest.roots)) == 3

    def test_anchor_distributed_has_combined_root(self):
        """Combined root (root of roots) is computed."""
        entries = [{"id": i, "data": f"entry_{i}"} for i in range(100)]
        config = ForestConfig()

        forest = anchor_distributed(entries, config)

        assert forest.combined_root != ""
        assert ":" in forest.combined_root  # dual_hash format

    def test_anchor_distributed_empty_entries(self):
        """Empty entries returns empty forest."""
        config = ForestConfig()

        forest = anchor_distributed([], config)

        assert len(forest.roots) == 0
        assert forest.quorum_achieved is False

    def test_anchor_distributed_custom_n_roots(self):
        """Custom n_roots works correctly."""
        entries = [{"id": i} for i in range(50)]
        config = ForestConfig(n_roots=5, quorum=3)

        forest = anchor_distributed(entries, config)

        assert len(forest.roots) == 5
        assert forest.quorum_achieved is True

    def test_anchor_distributed_root_sources(self):
        """Each root has source node mapping."""
        entries = [{"id": i} for i in range(30)]
        config = ForestConfig()

        forest = anchor_distributed(entries, config)

        # Each root should have a source
        for root in forest.roots:
            assert root in forest.root_sources
            assert forest.root_sources[root].startswith("node_")


# =============================================================================
# TEST: verify_quorum
# =============================================================================

class TestVerifyQuorum:
    """Tests for verify_quorum function."""

    def test_verify_quorum_pass_3_of_3(self):
        """3/3 roots meets quorum."""
        entries = [{"id": i} for i in range(100)]
        config = ForestConfig(n_roots=3, quorum=2)

        forest = anchor_distributed(entries, config)

        result = verify_quorum(forest, config)
        assert result is True

    def test_verify_quorum_pass_2_of_3(self):
        """2/3 roots meets quorum (Byzantine tolerance)."""
        entries = [{"id": i} for i in range(100)]
        config = ForestConfig(n_roots=3, quorum=2)

        forest = anchor_distributed(entries, config)

        # Simulate loss of one node
        forest.roots = forest.roots[:2]

        result = verify_quorum(forest, config)
        assert result is True

    def test_verify_quorum_fail_1_of_3(self):
        """1/3 roots does not meet quorum."""
        entries = [{"id": i} for i in range(100)]
        config = ForestConfig(n_roots=3, quorum=2)

        forest = anchor_distributed(entries, config)

        # Simulate loss of two nodes
        forest.roots = forest.roots[:1]

        result = verify_quorum(forest, config)
        assert result is False


# =============================================================================
# TEST: recover_from_loss
# =============================================================================

class TestRecoverFromLoss:
    """Tests for recover_from_loss function."""

    def test_recover_from_loss_quorum_maintained(self):
        """Lost root removed, quorum still met."""
        entries = [{"id": i} for i in range(100)]
        config = ForestConfig(n_roots=3, quorum=2)

        forest = anchor_distributed(entries, config)
        lost_root = forest.roots[0]

        new_forest = recover_from_loss(forest, lost_root, config)

        assert len(new_forest.roots) == 2
        assert lost_root not in new_forest.roots
        assert new_forest.quorum_achieved is True

    def test_recover_from_loss_quorum_lost(self):
        """Losing too many roots raises StopRule."""
        entries = [{"id": i} for i in range(100)]
        config = ForestConfig(n_roots=3, quorum=2)

        forest = anchor_distributed(entries, config)

        # Lose first root
        forest = recover_from_loss(forest, forest.roots[0], config)

        # Losing second root should fail
        with pytest.raises(StopRule):
            recover_from_loss(forest, forest.roots[0], config)

    def test_recover_from_loss_unknown_root(self):
        """Unknown root returns unchanged state."""
        entries = [{"id": i} for i in range(100)]
        config = ForestConfig()

        forest = anchor_distributed(entries, config)
        original_roots = len(forest.roots)

        new_forest = recover_from_loss(forest, "nonexistent_root", config)

        assert len(new_forest.roots) == original_roots


# =============================================================================
# TEST: merge_forests
# =============================================================================

class TestMergeForests:
    """Tests for merge_forests function."""

    def test_merge_forests_consistent(self):
        """Two valid forests merge cleanly."""
        entries_a = [{"id": i} for i in range(50)]
        entries_b = [{"id": i + 100} for i in range(50)]
        config = ForestConfig()

        forest_a = anchor_distributed(entries_a, config)
        forest_b = anchor_distributed(entries_b, config)

        merged = merge_forests(forest_a, forest_b, config)

        # Should have unique roots from both
        assert len(merged.roots) == 6  # 3 + 3, all unique

    def test_merge_forests_same_entries(self):
        """Forests with same entries have overlapping roots."""
        entries = [{"id": i} for i in range(50)]
        config = ForestConfig()

        forest_a = anchor_distributed(entries, config)
        forest_b = anchor_distributed(entries, config)

        merged = merge_forests(forest_a, forest_b, config)

        # Roots should be same, so merged = 3
        assert len(merged.roots) == 3

    def test_merge_forests_quorum_achieved(self):
        """Merged forest achieves quorum."""
        entries_a = [{"id": i} for i in range(30)]
        entries_b = [{"id": i + 100} for i in range(30)]
        config = ForestConfig(quorum=2)

        forest_a = anchor_distributed(entries_a, config)
        forest_b = anchor_distributed(entries_b, config)

        merged = merge_forests(forest_a, forest_b, config)

        assert merged.quorum_achieved is True


# =============================================================================
# TEST: simulate_node_loss
# =============================================================================

class TestSimulateNodeLoss:
    """Tests for simulate_node_loss utility."""

    def test_simulate_node_loss(self):
        """Simulate loss of specific node."""
        entries = [{"id": i} for i in range(100)]
        config = ForestConfig()

        forest = anchor_distributed(entries, config)
        original_count = len(forest.roots)

        new_forest = simulate_node_loss(forest, 0, config)

        assert len(new_forest.roots) == original_count - 1

    def test_simulate_node_loss_invalid_index(self):
        """Invalid index returns unchanged state."""
        entries = [{"id": i} for i in range(100)]
        config = ForestConfig()

        forest = anchor_distributed(entries, config)

        new_forest = simulate_node_loss(forest, 99, config)  # Invalid index

        assert len(new_forest.roots) == len(forest.roots)


# =============================================================================
# TEST: Constants
# =============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_distributed_roots(self):
        """DISTRIBUTED_ROOTS is 3."""
        assert DISTRIBUTED_ROOTS == 3

    def test_quorum(self):
        """QUORUM is 2."""
        assert QUORUM == 2

    def test_distributed_alpha_bonus(self):
        """DISTRIBUTED_ALPHA_BONUS is 0.02."""
        assert DISTRIBUTED_ALPHA_BONUS == 0.02


# =============================================================================
# TEST: create_initial_forest
# =============================================================================

class TestCreateInitialForest:
    """Tests for create_initial_forest utility."""

    def test_create_initial_forest(self):
        """Creates empty forest state."""
        forest = create_initial_forest()

        assert len(forest.roots) == 0
        assert len(forest.root_sources) == 0
        assert forest.quorum_achieved is False
        assert forest.combined_root == ""


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestForestIntegration:
    """Integration tests for full forest workflow."""

    def test_blackout_resilience_workflow(self):
        """Simulate conjunction blackout with node loss."""
        entries = [{"id": i, "decision": f"nav_{i}"} for i in range(100)]
        config = ForestConfig(n_roots=3, quorum=2)

        # Create distributed anchor
        forest = anchor_distributed(entries, config)
        assert forest.quorum_achieved is True

        # Simulate conjunction blackout - lose node 0
        forest = simulate_node_loss(forest, 0, config)
        assert forest.quorum_achieved is True  # 2/3 still valid

        # Verify quorum still met
        assert verify_quorum(forest, config) is True

        # Reconnection would happen here
        # For now, verify trust is maintained

    def test_full_recovery_workflow(self):
        """anchor -> loss -> recovery -> merge workflow."""
        config = ForestConfig(n_roots=3, quorum=2)

        # Initial anchoring
        entries1 = [{"id": i} for i in range(50)]
        forest1 = anchor_distributed(entries1, config)

        # Lose a node during blackout
        lost_root = forest1.roots[0]
        forest1 = recover_from_loss(forest1, lost_root, config)

        # Another anchor from remaining nodes
        entries2 = [{"id": i + 50} for i in range(50)]
        forest2 = anchor_distributed(entries2, config)

        # After reconnection, merge
        merged = merge_forests(forest1, forest2, config)

        # Should have roots from both (minus lost)
        assert len(merged.roots) >= config.quorum
        assert merged.quorum_achieved is True

    def test_byzantine_tolerance(self):
        """System tolerates 1 Byzantine (faulty) node."""
        entries = [{"id": i} for i in range(100)]
        config = ForestConfig(n_roots=3, quorum=2)

        forest = anchor_distributed(entries, config)

        # Lose 1 node (Byzantine failure)
        forest = simulate_node_loss(forest, 1, config)

        # Quorum still met - Byzantine tolerance achieved
        assert verify_quorum(forest, config) is True
        assert len(forest.roots) == 2
