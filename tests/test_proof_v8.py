"""
QED v8 Proof CLI Tests

Tests for v8 subcommands: build-packet, validate-config, merge-configs,
compare-packets, fleet-view. Verifies exit codes per specification:
  - 0: success
  - 1: validation/comparison issue (actionable)
  - 2: fatal error (missing file, bad input)
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from proof import (
    v8,
    build_packet_cmd,
    validate_config_cmd,
    merge_configs_cmd,
    compare_packets_cmd,
    fleet_view_cmd,
)
from decision_packet import DecisionPacket, PatternSummary, PacketMetrics


@pytest.fixture
def cli_runner():
    """Create a Click test runner."""
    return CliRunner()


@pytest.fixture
def sample_packet():
    """Create a sample DecisionPacket for testing."""
    patterns = [
        PatternSummary(
            pattern_id="PAT_001",
            validation_recall=0.995,
            false_positive_rate=0.005,
            dollar_value_annual=500_000,
            exploit_grade=True,
        ),
        PatternSummary(
            pattern_id="PAT_002",
            validation_recall=0.98,
            false_positive_rate=0.01,
            dollar_value_annual=300_000,
            exploit_grade=False,
        ),
    ]
    metrics = PacketMetrics(
        window_volume=100_000,
        avg_compression=10.5,
        annual_savings=800_000,
        slo_breach_rate=0.002,
    )
    return DecisionPacket(
        deployment_id="test-deployment-01",
        manifest_ref="data/manifests/test.yaml",
        sampled_receipts=["rcpt_001", "rcpt_002"],
        clarity_audit_ref="audits/test.json",
        edge_lab_summary={"n_tests": 1000, "n_hits": 990},
        pattern_usage=patterns,
        metrics=metrics,
    )


@pytest.fixture
def sample_config():
    """Create a sample config object for testing."""
    config = MagicMock()
    config.hook = "tesla"
    config.enabled_patterns = ["PAT_001", "PAT_002"]
    config.recall_floor = 0.999
    config.max_fp_rate = 0.01
    return config


class TestBuildPacket:
    """Tests for build-packet command."""

    def test_build_packet_manifest_not_found(self, cli_runner):
        """Exit code 2 when manifest not found."""
        result = cli_runner.invoke(
            v8,
            ["build-packet", "-d", "test-deploy", "-m", "/nonexistent/manifest.yaml"],
        )
        assert result.exit_code == 2

    def test_build_packet_manifest_not_found_json(self, cli_runner):
        """JSON output includes error on missing manifest."""
        result = cli_runner.invoke(
            v8,
            [
                "build-packet",
                "-d", "test-deploy",
                "-m", "/nonexistent/manifest.yaml",
                "-o", "json",
            ],
        )
        assert result.exit_code == 2
        output = json.loads(result.output)
        assert "error" in output

    @patch("proof.truthlink.build")
    def test_build_packet_success(self, mock_build, cli_runner, sample_packet, tmp_path):
        """Exit code 0 on successful build."""
        mock_build.return_value = sample_packet

        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir()
        (manifest_dir / "test.yaml").write_text("manifest: test")

        result = cli_runner.invoke(
            v8,
            ["build-packet", "-d", "test-deploy", "-m", str(manifest_dir)],
        )
        assert result.exit_code == 0

    @patch("proof.truthlink.build")
    def test_build_packet_json_output(self, mock_build, cli_runner, sample_packet, tmp_path):
        """JSON output contains packet data."""
        mock_build.return_value = sample_packet

        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir()
        (manifest_dir / "test.yaml").write_text("manifest: test")

        result = cli_runner.invoke(
            v8,
            ["build-packet", "-d", "test-deploy", "-m", str(manifest_dir), "-o", "json"],
        )
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert "packet_id" in output


class TestValidateConfig:
    """Tests for validate-config command."""

    def test_validate_config_file_not_found(self, cli_runner):
        """Exit code 2 when config file not found (Click handles this)."""
        result = cli_runner.invoke(
            v8,
            ["validate-config", "/nonexistent/config.json"],
        )
        # Click returns 2 for missing file when exists=True
        assert result.exit_code == 2

    @patch("proof.config_schema.load")
    def test_validate_config_valid(self, mock_load, cli_runner, sample_config, tmp_path):
        """Exit code 0 when config is valid."""
        # Add to_dict method to mock
        sample_config.to_dict = MagicMock(return_value={"hook": "tesla"})
        mock_load.return_value = sample_config

        config_file = tmp_path / "valid.json"
        config_file.write_text('{"hook": "tesla"}')

        result = cli_runner.invoke(
            v8,
            ["validate-config", str(config_file)],
        )
        assert result.exit_code == 0

    @patch("proof.config_schema.load")
    def test_validate_config_invalid(self, mock_load, cli_runner, tmp_path):
        """Exit code 1 when config load fails with ValueError."""
        mock_load.side_effect = ValueError("recall_floor too low")

        config_file = tmp_path / "invalid.json"
        config_file.write_text('{"hook": "tesla", "recall_floor": 0.9}')

        result = cli_runner.invoke(
            v8,
            ["validate-config", str(config_file)],
        )
        # Load failures result in exit 1
        assert result.exit_code == 1

    @patch("proof.config_schema.load")
    def test_validate_config_json_output(self, mock_load, cli_runner, sample_config, tmp_path):
        """JSON output contains validation result."""
        sample_config.to_dict = MagicMock(return_value={"hook": "tesla"})
        mock_load.return_value = sample_config

        config_file = tmp_path / "valid.json"
        config_file.write_text('{"hook": "tesla"}')

        result = cli_runner.invoke(
            v8,
            ["validate-config", str(config_file), "-o", "json"],
        )
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["valid"] is True


class TestMergeConfigs:
    """Tests for merge-configs command."""

    def test_merge_configs_parent_not_found(self, cli_runner, tmp_path):
        """Exit code 2 when parent config not found."""
        child = tmp_path / "child.json"
        child.write_text('{"hook": "tesla"}')

        result = cli_runner.invoke(
            v8,
            ["merge-configs", "-p", "/nonexistent/parent.json", "-c", str(child)],
        )
        assert result.exit_code == 2

    def test_merge_configs_child_not_found(self, cli_runner, tmp_path):
        """Exit code 2 when child config not found."""
        parent = tmp_path / "parent.json"
        parent.write_text('{"hook": "global"}')

        result = cli_runner.invoke(
            v8,
            ["merge-configs", "-p", str(parent), "-c", "/nonexistent/child.json"],
        )
        assert result.exit_code == 2

    @patch("proof.config_schema.load")
    @patch("proof.merge_rules.merge")
    def test_merge_configs_valid(self, mock_merge, mock_load, cli_runner, sample_config, tmp_path):
        """Exit code 0 when merge is valid."""
        mock_load.return_value = sample_config

        # Create mock MergeResult
        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_result.merged = sample_config
        mock_result.violations = []
        mock_result.explanation = MagicMock()
        mock_result.explanation.safety_direction = "tightened"
        mock_result.explanation.patterns_removed = []
        mock_merge.return_value = mock_result

        parent = tmp_path / "parent.json"
        parent.write_text('{"hook": "global"}')
        child = tmp_path / "child.json"
        child.write_text('{"hook": "tesla"}')

        result = cli_runner.invoke(
            v8,
            ["merge-configs", "-p", str(parent), "-c", str(child)],
        )
        assert result.exit_code == 0

    @patch("proof.config_schema.load")
    @patch("proof.merge_rules.merge")
    def test_merge_configs_violation(self, mock_merge, mock_load, cli_runner, sample_config, tmp_path):
        """Exit code 1 when merge has violations."""
        mock_load.return_value = sample_config

        # Create mock MergeResult with violations
        mock_violation = MagicMock()
        mock_violation.message = "recall_floor loosened - blocked"
        mock_violation.field_name = "recall_floor"
        mock_violation.severity = "error"

        mock_result = MagicMock()
        mock_result.is_valid = False
        mock_result.merged = None
        mock_result.violations = [mock_violation]
        mock_result.explanation = MagicMock()
        mock_result.explanation.safety_direction = "loosened"
        mock_merge.return_value = mock_result

        parent = tmp_path / "parent.json"
        parent.write_text('{"hook": "global"}')
        child = tmp_path / "child.json"
        child.write_text('{"hook": "loose"}')

        result = cli_runner.invoke(
            v8,
            ["merge-configs", "-p", str(parent), "-c", str(child)],
        )
        assert result.exit_code == 1


class TestComparePackets:
    """Tests for compare-packets command."""

    def test_compare_packets_old_not_found(self, cli_runner, tmp_path):
        """Exit code 2 when old packet not found."""
        packets_dir = tmp_path / "packets"
        packets_dir.mkdir()

        result = cli_runner.invoke(
            v8,
            ["compare-packets", "-a", "nonexistent", "-b", "also-nonexistent", "--packets-dir", str(packets_dir)],
        )
        assert result.exit_code == 2

    def test_compare_packets_no_packets_dir(self, cli_runner):
        """Exit code 2 when packets directory not found."""
        result = cli_runner.invoke(
            v8,
            ["compare-packets", "-a", "old123", "-b", "new456", "--packets-dir", "/nonexistent/packets"],
        )
        assert result.exit_code == 2

    @patch("proof.truthlink.compare")
    def test_compare_packets_improvement(self, mock_compare, cli_runner, sample_packet, tmp_path):
        """Exit code 0 on improvement comparison."""
        mock_compare.return_value = {
            "health_score_delta": 5,
            "savings_delta": 100_000,
            "patterns_added": ["PAT_003"],
            "patterns_removed": [],
        }

        # Create packet files
        packets_dir = tmp_path / "packets"
        packets_dir.mkdir()
        packet_file = packets_dir / "test.jsonl"

        # Create old and new packets
        old_packet = sample_packet
        # Create new packet with different ID
        new_patterns = sample_packet.pattern_usage.copy()
        new_metrics = PacketMetrics(
            window_volume=110_000,
            avg_compression=11.0,
            annual_savings=900_000,
            slo_breach_rate=0.001,
        )
        new_packet = DecisionPacket(
            deployment_id="test-deployment-02",
            manifest_ref="data/manifests/test.yaml",
            sampled_receipts=["rcpt_003"],
            clarity_audit_ref="audits/test2.json",
            edge_lab_summary={"n_tests": 1000, "n_hits": 995},
            pattern_usage=new_patterns,
            metrics=new_metrics,
        )

        with packet_file.open("w") as f:
            f.write(old_packet.to_json() + "\n")
            f.write(new_packet.to_json() + "\n")

        result = cli_runner.invoke(
            v8,
            [
                "compare-packets",
                "-a", old_packet.packet_id[:8],
                "-b", new_packet.packet_id[:8],
                "--packets-dir", str(packets_dir),
            ],
        )
        assert result.exit_code == 0

    @patch("proof.truthlink.compare")
    def test_compare_packets_regression(self, mock_compare, cli_runner, sample_packet, tmp_path):
        """Exit code 1 on regression comparison."""
        mock_compare.return_value = {
            "health_score_delta": -10,
            "savings_delta": -50_000,
            "patterns_added": [],
            "patterns_removed": ["PAT_001"],
        }

        packets_dir = tmp_path / "packets"
        packets_dir.mkdir()
        packet_file = packets_dir / "test.jsonl"

        # Create old packet (better)
        old_packet = sample_packet

        # Create new packet (worse)
        worse_metrics = PacketMetrics(
            window_volume=90_000,
            avg_compression=9.0,
            annual_savings=700_000,
            slo_breach_rate=0.005,
        )
        new_packet = DecisionPacket(
            deployment_id="test-deployment-02",
            manifest_ref="data/manifests/test.yaml",
            sampled_receipts=["rcpt_003"],
            clarity_audit_ref="audits/test2.json",
            edge_lab_summary={"n_tests": 1000, "n_hits": 950},
            pattern_usage=[sample_packet.pattern_usage[0]],  # Less patterns
            metrics=worse_metrics,
        )

        with packet_file.open("w") as f:
            f.write(old_packet.to_json() + "\n")
            f.write(new_packet.to_json() + "\n")

        result = cli_runner.invoke(
            v8,
            [
                "compare-packets",
                "-a", old_packet.packet_id[:8],
                "-b", new_packet.packet_id[:8],
                "--packets-dir", str(packets_dir),
            ],
        )
        assert result.exit_code == 1


class TestFleetView:
    """Tests for fleet-view command."""

    def test_fleet_view_no_packets_dir(self, cli_runner):
        """Exit code 2 when packets directory not found."""
        result = cli_runner.invoke(
            v8,
            ["fleet-view", "--packets-dir", "/nonexistent/packets"],
        )
        assert result.exit_code == 2

    def test_fleet_view_empty_packets_dir(self, cli_runner, tmp_path):
        """Exit code 2 when no packets found."""
        packets_dir = tmp_path / "packets"
        packets_dir.mkdir()

        result = cli_runner.invoke(
            v8,
            ["fleet-view", "--packets-dir", str(packets_dir)],
        )
        assert result.exit_code == 2

    def test_fleet_view_success(self, cli_runner, sample_packet, tmp_path):
        """Exit code 0 with valid packets."""
        packets_dir = tmp_path / "packets"
        packets_dir.mkdir()
        packet_file = packets_dir / "test.jsonl"

        with packet_file.open("w") as f:
            f.write(sample_packet.to_json() + "\n")

        result = cli_runner.invoke(
            v8,
            ["fleet-view", "--packets-dir", str(packets_dir)],
        )
        assert result.exit_code == 0

    def test_fleet_view_json_output(self, cli_runner, sample_packet, tmp_path):
        """JSON output contains fleet metrics."""
        packets_dir = tmp_path / "packets"
        packets_dir.mkdir()
        packet_file = packets_dir / "test.jsonl"

        with packet_file.open("w") as f:
            f.write(sample_packet.to_json() + "\n")

        result = cli_runner.invoke(
            v8,
            ["fleet-view", "--packets-dir", str(packets_dir), "-o", "json"],
        )
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert "total_deployments" in output
        assert "fleet_cohesion" in output

    @patch("proof.mesh_view_v3.diagnose")
    def test_fleet_view_diagnose_healthy(self, mock_diagnose, cli_runner, sample_packet, tmp_path):
        """Exit code 0 when fleet is healthy."""
        diagnosis = MagicMock()
        diagnosis.is_healthy = True
        diagnosis.warnings = []
        diagnosis.issues = []
        diagnosis.to_dict.return_value = {"is_healthy": True, "warnings": [], "issues": []}
        mock_diagnose.return_value = diagnosis

        packets_dir = tmp_path / "packets"
        packets_dir.mkdir()
        packet_file = packets_dir / "test.jsonl"

        with packet_file.open("w") as f:
            f.write(sample_packet.to_json() + "\n")

        result = cli_runner.invoke(
            v8,
            ["fleet-view", "--packets-dir", str(packets_dir), "--diagnose"],
        )
        assert result.exit_code == 0

    @patch("proof.mesh_view_v3.diagnose")
    def test_fleet_view_diagnose_unhealthy(self, mock_diagnose, cli_runner, sample_packet, tmp_path):
        """Exit code 1 when fleet has issues."""
        diagnosis = MagicMock()
        diagnosis.is_healthy = False
        diagnosis.warnings = ["High stale ratio"]
        diagnosis.issues = ["Fleet fragmented"]
        diagnosis.to_dict.return_value = {
            "is_healthy": False,
            "warnings": ["High stale ratio"],
            "issues": ["Fleet fragmented"],
        }
        mock_diagnose.return_value = diagnosis

        packets_dir = tmp_path / "packets"
        packets_dir.mkdir()
        packet_file = packets_dir / "test.jsonl"

        with packet_file.open("w") as f:
            f.write(sample_packet.to_json() + "\n")

        result = cli_runner.invoke(
            v8,
            ["fleet-view", "--packets-dir", str(packets_dir), "--diagnose"],
        )
        assert result.exit_code == 1


class TestExitCodes:
    """Summary tests for exit code contract."""

    def test_exit_code_0_success(self, cli_runner, sample_packet, tmp_path):
        """Exit code 0 means success."""
        packets_dir = tmp_path / "packets"
        packets_dir.mkdir()
        packet_file = packets_dir / "test.jsonl"

        with packet_file.open("w") as f:
            f.write(sample_packet.to_json() + "\n")

        result = cli_runner.invoke(
            v8,
            ["fleet-view", "--packets-dir", str(packets_dir)],
        )
        assert result.exit_code == 0

    def test_exit_code_2_missing_file(self, cli_runner):
        """Exit code 2 means fatal error like missing file."""
        result = cli_runner.invoke(
            v8,
            ["build-packet", "-d", "test", "-m", "/nonexistent"],
        )
        assert result.exit_code == 2

    @patch("proof.config_schema.load")
    @patch("proof.merge_rules.merge")
    def test_exit_code_1_actionable(self, mock_merge, mock_load, cli_runner, sample_config, tmp_path):
        """Exit code 1 means actionable issue like validation failure."""
        mock_load.return_value = sample_config

        # Create mock MergeResult with violations (proper format)
        mock_violation = MagicMock()
        mock_violation.message = "safety loosened"
        mock_violation.field_name = "recall_floor"
        mock_violation.severity = "error"

        mock_result = MagicMock()
        mock_result.is_valid = False
        mock_result.merged = None
        mock_result.violations = [mock_violation]
        mock_result.explanation = MagicMock()
        mock_result.explanation.safety_direction = "loosened"
        mock_merge.return_value = mock_result

        parent = tmp_path / "parent.json"
        parent.write_text('{}')
        child = tmp_path / "child.json"
        child.write_text('{}')

        result = cli_runner.invoke(
            v8,
            ["merge-configs", "-p", str(parent), "-c", str(child)],
        )
        assert result.exit_code == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
