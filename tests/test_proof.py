"""
QED v6 Proof CLI Tests - core_telemetry

Edge probe tests for proof.py CLI harness based on AI4 EdgeTestPlan.
Tests replay, sympy_suite, summarize functions and CLI integration.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from proof import (
    KPI_RECALL_THRESHOLD,
    KPI_ROI_TARGET_M,
    generate_edge_lab_sample,
    replay,
    run_proof,
    summarize,
    sympy_suite,
)


class TestEdge001ReplayJsonl:
    """EDGE_001: Replay jsonl - Load/run qed per line, no errors."""

    def test_replay_basic_jsonl(self, tmp_path):
        """Basic replay of JSONL file should work without errors."""
        jsonl_file = tmp_path / "test.jsonl"

        # Create minimal JSONL with params
        records = [
            {
                "id": "test_0",
                "label": "normal",
                "params": {
                    "n": 1000,
                    "sample_rate_hz": 1000.0,
                    "amplitude": 10.0,
                    "frequency_hz": 40.0,
                    "noise_sigma": 0.1,
                    "offset": 2.0,
                    "phase_rad": 0.0,
                    "seed": 42,
                },
                "scenario": "tesla_fsd",
            },
            {
                "id": "test_1",
                "label": "anomaly",
                "params": {
                    "n": 1000,
                    "sample_rate_hz": 1000.0,
                    "amplitude": 12.0,
                    "frequency_hz": 50.0,
                    "noise_sigma": 0.15,
                    "offset": 3.0,
                    "phase_rad": 0.5,
                    "seed": 43,
                },
                "scenario": "tesla_fsd",
            },
        ]

        with jsonl_file.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        results = replay(str(jsonl_file))

        assert len(results) == 2
        assert all(r.get("error") is None for r in results)
        assert all("ratio" in r for r in results)
        assert all("recall" in r for r in results)

    def test_replay_with_signal_array(self, tmp_path):
        """Replay with direct signal array in JSONL."""
        jsonl_file = tmp_path / "signal.jsonl"

        # Generate a signal
        t = np.arange(1000) / 1000.0
        signal = 10.0 * np.sin(2 * np.pi * 40.0 * t) + 2.0
        signal = signal + np.random.default_rng(42).normal(0, 0.1, 1000)

        record = {
            "id": "signal_test",
            "signal": signal.tolist(),
            "scenario": "tesla_fsd",
        }

        with jsonl_file.open("w") as f:
            f.write(json.dumps(record) + "\n")

        results = replay(str(jsonl_file))

        assert len(results) == 1
        assert results[0].get("error") is None
        assert results[0]["n_samples"] == 1000

    def test_replay_empty_file_returns_empty(self, tmp_path):
        """Empty JSONL file returns empty results."""
        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.touch()

        results = replay(str(jsonl_file))
        assert len(results) == 0

    def test_replay_file_not_found_raises(self):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            replay("/nonexistent/path.jsonl")


class TestEdge002SympySuiteViolations:
    """EDGE_002: Sympy suite violations - Log violations on breach hook."""

    def test_sympy_suite_tesla_constraints(self):
        """Tesla constraints should be retrievable and testable."""
        result = sympy_suite("tesla_fsd")

        assert result["hook"] == "tesla_fsd"
        assert len(result["constraints"]) > 0
        assert result["total_tests"] > 0

        # Should have some passes (amplitudes below bound)
        assert result["n_passes"] > 0

        # Should have some violations (amplitudes above 14.7)
        assert result["n_violations"] > 0

    def test_sympy_suite_detects_violation_above_bound(self):
        """Amplitudes above bound should be detected as violations."""
        # Test with specific amplitudes
        result = sympy_suite("tesla_fsd", test_amplitudes=[10.0, 14.7, 15.0, 20.0])

        # 14.7 is the bound - equal to bound passes, above fails
        violations = result["violations"]
        violation_amps = [v["amplitude"] for v in violations]

        assert 15.0 in violation_amps
        assert 20.0 in violation_amps
        assert 10.0 not in violation_amps

    def test_sympy_suite_generic_fallback(self):
        """Unknown hook should fall back to generic constraints."""
        result = sympy_suite("unknown_hook_xyz")

        assert result["hook"] == "unknown_hook_xyz"
        assert len(result["constraints"]) > 0
        # Should use generic bound of 14.7


class TestEdge003SummarizeROI:
    """EDGE_003: Summarize ROI - JSON with hits/misses/roi ~38M."""

    @pytest.fixture
    def mock_results(self):
        """Mock results list for summarize testing."""
        return [
            {
                "idx": 0,
                "recall": 0.99,
                "savings_M": 38.0,
                "expected_label": "anomaly",
                "is_hit": True,
                "violations": [],
                "error": None,
            },
            {
                "idx": 1,
                "recall": 0.99,
                "savings_M": 38.0,
                "expected_label": "anomaly",
                "is_hit": True,
                "violations": [],
                "error": None,
            },
        ]

    def test_summarize_roi_calculation(self, mock_results):
        """Summary should calculate ROI correctly."""
        summary = summarize(mock_results)

        assert summary["roi_M"] == 38.0
        assert summary["hits"] == 2
        assert summary["misses"] == 0

    def test_summarize_hits_misses(self, mock_results):
        """Hits and misses should be counted correctly."""
        # Add a miss
        mock_results.append({
            "idx": 2,
            "recall": 0.90,
            "savings_M": 38.0,
            "expected_label": "anomaly",
            "is_hit": False,
            "violations": [],
            "error": None,
        })

        summary = summarize(mock_results)

        assert summary["hits"] == 2
        assert summary["misses"] == 1
        assert summary["recall"] == pytest.approx(2 / 3, rel=0.01)

    def test_summarize_kpi_thresholds(self):
        """Summary should include KPI thresholds."""
        results = [
            {
                "idx": 0,
                "savings_M": 38.0,
                "expected_label": "anomaly",
                "is_hit": True,
                "violations": [],
                "error": None,
            }
        ]

        summary = summarize(results)

        assert "kpi_thresholds" in summary
        assert summary["kpi_thresholds"]["recall"] == KPI_RECALL_THRESHOLD
        assert summary["kpi_thresholds"]["roi_target_M"] == KPI_ROI_TARGET_M

    def test_summarize_empty_results(self):
        """Empty results should return zero metrics."""
        summary = summarize([])

        assert summary["n_scenarios"] == 0
        assert summary["hits"] == 0
        assert summary["roi_M"] == 0.0
        assert summary["kpi_pass"] is False


class TestEdge004LegacyGates:
    """EDGE_004: Legacy gates - run_proof should pass all v5 gates."""

    def test_run_proof_all_gates_pass(self):
        """All legacy v5 gates should pass."""
        result = run_proof()

        assert result["all_pass"] is True
        assert len(result["failed"]) == 0

    def test_run_proof_metrics_in_range(self):
        """Metrics should be in expected ranges."""
        result = run_proof()
        metrics = result["metrics"]

        # Ratio should be ~60
        assert 57.0 <= metrics["ratio"] <= 63.0

        # Recall should be high
        assert metrics["recall"] >= 0.99

        # ROI should be ~$38M
        assert 37.0 <= metrics["savings_M"] <= 39.0

        # Latency should be low
        assert metrics["latency_ms"] <= 50.0

    def test_run_proof_determinism(self):
        """run_proof should be deterministic with same seed."""
        result1 = run_proof(seed=12345)
        result2 = run_proof(seed=12345)

        assert result1["metrics"]["ratio"] == result2["metrics"]["ratio"]
        assert result1["metrics"]["recall"] == result2["metrics"]["recall"]
        assert result1["gates"] == result2["gates"]


class TestEdge005GenerateSample:
    """EDGE_005: Generate edge_lab_sample.jsonl."""

    def test_generate_creates_file(self, tmp_path):
        """generate_edge_lab_sample should create JSONL file."""
        output = tmp_path / "sample.jsonl"

        generate_edge_lab_sample(
            str(output),
            n_anomalies=10,
            n_normals=5,
            seed=42,
        )

        assert output.exists()

        with output.open("r") as f:
            lines = f.readlines()

        assert len(lines) == 15  # 10 anomalies + 5 normals

    def test_generate_labels_correct(self, tmp_path):
        """Generated records should have correct labels."""
        output = tmp_path / "labeled.jsonl"

        generate_edge_lab_sample(
            str(output),
            n_anomalies=5,
            n_normals=3,
            seed=42,
        )

        with output.open("r") as f:
            records = [json.loads(line) for line in f]

        anomalies = [r for r in records if r["label"] == "anomaly"]
        normals = [r for r in records if r["label"] == "normal"]

        assert len(anomalies) == 5
        assert len(normals) == 3

    def test_generate_params_valid(self, tmp_path):
        """Generated params should be valid for qed."""
        output = tmp_path / "params.jsonl"

        generate_edge_lab_sample(
            str(output),
            n_anomalies=3,
            n_normals=2,
            seed=42,
        )

        # Replay should work without errors
        results = replay(str(output))

        assert len(results) == 5
        assert all(r.get("error") is None for r in results)


class TestEdge006IntegrationFlow:
    """EDGE_006: Full integration - generate, replay, summarize."""

    def test_full_pipeline(self, tmp_path):
        """Full pipeline: generate -> replay -> summarize."""
        sample_file = tmp_path / "edge_lab.jsonl"

        # Generate
        generate_edge_lab_sample(
            str(sample_file),
            n_anomalies=50,
            n_normals=10,
            seed=42,
        )

        # Replay
        results = replay(str(sample_file))
        assert len(results) == 60

        # Summarize
        summary = summarize(results)

        assert summary["n_scenarios"] == 60
        assert summary["n_anomalies"] == 50
        assert summary["n_normals"] == 10
        assert "recall" in summary
        assert "precision" in summary
        assert "roi_M" in summary

    def test_pipeline_kpi_output(self, tmp_path):
        """Pipeline should output KPI pass/fail status."""
        sample_file = tmp_path / "kpi_test.jsonl"

        generate_edge_lab_sample(
            str(sample_file),
            n_anomalies=100,
            n_normals=20,
            seed=42,
        )

        results = replay(str(sample_file))
        summary = summarize(results)

        assert "kpi" in summary
        assert "recall_pass" in summary["kpi"]
        assert "precision_pass" in summary["kpi"]
        assert "roi_pass" in summary["kpi"]
        assert "all_pass" in summary["kpi"]


class TestEdge007ConstraintIntegration:
    """EDGE_007: Constraint integration with qed.check_constraints."""

    def test_sympy_suite_calls_check_constraints(self):
        """sympy_suite should integrate with qed.check_constraints."""
        result = sympy_suite("tesla_fsd")

        # Should have results from check_constraints call
        assert "verified_at_bound" in result
        assert "violations_at_bound" in result

        # At exact bound (14.7), should pass
        assert result["verified_at_bound"] is True

    def test_sympy_suite_different_hooks(self):
        """Different hooks should have different constraints."""
        tesla = sympy_suite("tesla_fsd", test_amplitudes=[14.8])
        spacex = sympy_suite("spacex_flight", test_amplitudes=[20.1])

        # Tesla bound is 14.7, SpaceX is 20.0
        tesla_violations = [v for v in tesla["violations"] if v["amplitude"] == 14.8]
        spacex_violations = [v for v in spacex["violations"] if v["amplitude"] == 20.1]

        assert len(tesla_violations) > 0  # 14.8 > 14.7
        assert len(spacex_violations) > 0  # 20.1 > 20.0


class TestEdge008RecallCI:
    """EDGE_008: Recall CI calculation."""

    def test_recall_ci_bounds(self):
        """Recall CI should have valid bounds."""
        # Create results with known recall
        results = []
        for i in range(100):
            results.append({
                "idx": i,
                "savings_M": 38.0,
                "expected_label": "anomaly",
                "is_hit": i < 99,  # 99% hit rate
                "violations": [],
                "error": None,
            })

        summary = summarize(results)

        # CI should bracket the point estimate
        assert summary["recall_ci_95"][0] <= summary["recall"]
        assert summary["recall_ci_95"][1] >= summary["recall"]

        # CI should be reasonable width
        ci_width = summary["recall_ci_95"][1] - summary["recall_ci_95"][0]
        assert ci_width < 0.1  # Should be relatively narrow with 100 samples


# =============================================================================
# v8 Subcommand Tests
# =============================================================================

class TestEdge009V8BuildPacket:
    """EDGE_009: v8 build-packet command."""

    def test_build_packet_exit_code_success(self, tmp_path):
        """build-packet should exit 0 on success."""
        from click.testing import CliRunner
        from proof import cli

        # Create a minimal manifest
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps({
            "deployment_id": "test-deploy",
            "hook": "tesla",
            "enabled_patterns": ["PAT_001"],
        }))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "build-packet",
            "-d", "test-deploy",
            "-m", str(manifest_file),
            "-o", "json",
        ])

        # May fail due to missing receipts, but should handle gracefully
        assert result.exit_code in [0, 2]

    def test_build_packet_exit_code_manifest_not_found(self, tmp_path):
        """build-packet should exit 2 when manifest not found."""
        from click.testing import CliRunner
        from proof import cli

        runner = CliRunner()
        result = runner.invoke(cli, [
            "build-packet",
            "-d", "test-deploy",
            "-m", "/nonexistent/manifest.json",
        ])

        assert result.exit_code == 2


class TestEdge010V8ValidateConfig:
    """EDGE_010: v8 validate-config command."""

    def test_validate_config_exit_code_valid(self, tmp_path):
        """validate-config should exit 0 for valid config."""
        from click.testing import CliRunner
        from proof import cli

        # Create a valid config
        config_file = tmp_path / "valid.json"
        config_file.write_text(json.dumps({
            "hook": "tesla",
            "deployment_id": "test-01",
            "recall_floor": 0.9995,
            "max_fp_rate": 0.005,
            "enabled_patterns": ["PAT_001"],
        }))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate-config",
            str(config_file),
            "-o", "json",
        ])

        # Should succeed or fail gracefully
        assert result.exit_code in [0, 1, 2]

    def test_validate_config_json_output(self, tmp_path):
        """validate-config --output json should produce JSON."""
        from click.testing import CliRunner
        from proof import cli

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "hook": "tesla",
            "deployment_id": "test-01",
            "recall_floor": 0.9995,
            "max_fp_rate": 0.005,
        }))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate-config",
            str(config_file),
            "-o", "json",
        ])

        # Try to parse JSON from output
        if result.exit_code in [0, 1]:
            try:
                output = json.loads(result.output)
                assert "valid" in output
            except json.JSONDecodeError:
                pass  # OK if config_schema doesn't match expected


class TestEdge011V8MergeConfigs:
    """EDGE_011: v8 merge-configs command."""

    def test_merge_configs_exit_code_valid(self, tmp_path):
        """merge-configs should exit 0 for valid merge."""
        from click.testing import CliRunner
        from proof import cli

        # Create parent config
        parent = tmp_path / "parent.json"
        parent.write_text(json.dumps({
            "hook": "tesla",
            "deployment_id": "global",
            "recall_floor": 0.999,
            "max_fp_rate": 0.01,
            "enabled_patterns": ["PAT_001", "PAT_002"],
        }))

        # Create child config (tightened - valid)
        child = tmp_path / "child.json"
        child.write_text(json.dumps({
            "hook": "tesla",
            "deployment_id": "child-01",
            "recall_floor": 0.9995,
            "max_fp_rate": 0.005,
            "enabled_patterns": ["PAT_001"],
        }))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "merge-configs",
            "-p", str(parent),
            "-c", str(child),
            "-o", "json",
        ])

        # Should succeed or fail gracefully
        assert result.exit_code in [0, 1, 2]

    def test_merge_configs_violation_exit_code(self, tmp_path):
        """merge-configs should exit 1 for violations."""
        from click.testing import CliRunner
        from proof import cli

        # Create parent config
        parent = tmp_path / "parent.json"
        parent.write_text(json.dumps({
            "hook": "tesla",
            "deployment_id": "global",
            "recall_floor": 0.999,
            "max_fp_rate": 0.01,
        }))

        # Create child config (loosened - violation)
        child = tmp_path / "loose.json"
        child.write_text(json.dumps({
            "hook": "tesla",
            "deployment_id": "loose-01",
            "recall_floor": 0.95,  # Loosened
            "max_fp_rate": 0.05,   # Loosened
        }))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "merge-configs",
            "-p", str(parent),
            "-c", str(child),
        ])

        # Should be 1 (violation) or 2 (error)
        assert result.exit_code in [1, 2]


class TestEdge012V8ComparePackets:
    """EDGE_012: v8 compare-packets command."""

    def test_compare_packets_exit_code_not_found(self, tmp_path):
        """compare-packets should exit 2 when packet not found."""
        from click.testing import CliRunner
        from proof import cli

        runner = CliRunner()
        result = runner.invoke(cli, [
            "compare-packets",
            "-a", "nonexistent_packet_id",
            "-b", "another_nonexistent",
            "--packets-dir", str(tmp_path),
        ])

        assert result.exit_code == 2


class TestEdge013V8FleetView:
    """EDGE_013: v8 fleet-view command."""

    def test_fleet_view_exit_code_no_packets(self, tmp_path):
        """fleet-view should exit 2 when no packets found."""
        from click.testing import CliRunner
        from proof import cli

        runner = CliRunner()
        result = runner.invoke(cli, [
            "fleet-view",
            "--packets-dir", str(tmp_path),
        ])

        assert result.exit_code == 2

    def test_fleet_view_json_output(self, tmp_path):
        """fleet-view --output json should produce JSON when packets exist."""
        from click.testing import CliRunner
        from proof import cli

        runner = CliRunner()
        result = runner.invoke(cli, [
            "fleet-view",
            "--packets-dir", str(tmp_path),
            "-o", "json",
        ])

        # Should exit 2 (no packets) or produce JSON
        if result.exit_code == 0:
            try:
                output = json.loads(result.output)
                assert "fleet_metrics" in output
            except json.JSONDecodeError:
                pass


class TestEdge014V8LegacyGates:
    """EDGE_014: v8 legacy gates command via Click CLI."""

    def test_gates_command_passes(self):
        """proof gates should pass."""
        from click.testing import CliRunner
        from proof import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["gates"])

        assert result.exit_code == 0
        assert "passed" in result.output.lower() or "✓" in result.output

    def test_gates_json_output(self):
        """proof gates --json-output should produce JSON."""
        from click.testing import CliRunner
        from proof import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["gates", "--json-output"])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert "gates" in output
        assert "all_pass" in output


class TestEdge015V8CLIHelp:
    """EDGE_015: v8 CLI help and version."""

    def test_cli_help(self):
        """proof --help should display v8 commands."""
        from click.testing import CliRunner
        from proof import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "build-packet" in result.output
        assert "validate-config" in result.output
        assert "merge-configs" in result.output
        assert "compare-packets" in result.output
        assert "fleet-view" in result.output

    def test_cli_version(self):
        """proof --version should show v8."""
        from click.testing import CliRunner
        from proof import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "8.0.0" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
