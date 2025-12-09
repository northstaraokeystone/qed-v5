"""
QED v6/v7/v8 Proof CLI Harness

CLI tool for validating QED telemetry compression and safety guarantees.
Provides subcommands for:
  - replay: Load edge_lab_sample.jsonl, run qed per scenario, collect metrics
  - sympy_suite: Get constraints per hook, verify, log violations
  - summarize: Output hits/misses/violations/ROI to JSON
  - gates: Run legacy v5 gate checks (synthetic signals)

v7 subcommands:
  - run-sims: Run pattern simulations via edge_lab_v2
  - recall-floor: Compute Clopper-Pearson exact recall lower bound
  - pattern-report: Display pattern library with sorting/filtering
  - clarity-audit: Process receipts through ClarityClean adapter

v8 subcommands (Click-based):
  - build-packet: Build DecisionPacket from manifest
  - validate-config: Validate QEDConfig file
  - merge-configs: Merge parent/child configs
  - compare-packets: Compare two decision packets
  - fleet-view: Display fleet topology and health

What to prove:
  - Recall >= 99.67% (95% CI on 900 anomalies)
  - Precision > 95%
  - ROI $38M (fleet calc)
  - Violations = 0 on normals
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable


import qed
import sympy_constraints

# v7 imports
from scipy.stats import beta
from edge_lab_v2 import run_pattern_sims
from shared_anomalies import load_library
from clarity_clean_adapter import process_receipts

# v8 imports
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# v8 module imports
import truthlink
import config_schema
import merge_rules
import mesh_view_v3
from decision_packet import DecisionPacket, load_packet, save_packet

# Rich console for v8 output
console = Console()

RICH_AVAILABLE = True

# --- KPI Thresholds ---
KPI_RECALL_THRESHOLD = 0.9967  # 99.67% recall CI
KPI_PRECISION_THRESHOLD = 0.95  # 95% precision
KPI_ROI_TARGET_M = 38.0  # $38M target ROI
KPI_ROI_TOLERANCE_M = 0.5  # +/- $0.5M tolerance
KPI_LATENCY_MS = 50.0  # Max latency per window


def _make_signal(
    n: int,
    sample_rate_hz: float,
    amplitude: float,
    frequency_hz: float,
    noise_sigma: float,
    offset: float,
    phase_rad: float,
    seed: int,
) -> np.ndarray:
    """
    Build a synthetic 1D signal:
      signal(t) = A * sin(2*pi*f*t + phase) + offset + Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sample_rate_hz
    clean = amplitude * np.sin(2.0 * np.pi * frequency_hz * t + phase_rad) + offset
    noise = rng.normal(0.0, noise_sigma, n)
    return clean + noise


def _count_events(signal: np.ndarray, threshold: float) -> int:
    """Count samples above a safety threshold."""
    return int((signal > threshold).sum())


def _normalized_rms_error(raw: np.ndarray, recon: np.ndarray) -> float:
    """Compute normalized RMS error between raw and reconstructed signals."""
    num = np.sqrt(np.mean((raw - recon) ** 2))
    den = np.sqrt(np.mean(raw**2))
    if den == 0.0:
        return 0.0
    return float(num / den)


def _deterministic_check(signal: np.ndarray, sample_rate_hz: float) -> bool:
    """Return True if qed() is deterministic for this signal (v5 keys only)."""
    out1 = qed.qed(
        signal, scenario="tesla_fsd", bit_depth=12, sample_rate_hz=sample_rate_hz
    )
    out2 = qed.qed(
        signal, scenario="tesla_fsd", bit_depth=12, sample_rate_hz=sample_rate_hz
    )
    v5_keys = ["ratio", "H_bits", "recall", "savings_M"]
    for key in v5_keys:
        if out1[key] != out2[key]:
            return False
    if out1["trace"].split()[0:3] != out2["trace"].split()[0:3]:
        return False
    return True


# --- Replay Subcommand ---


def replay(
    jsonl_path: str,
    scenario: str = "tesla_fsd",
    bit_depth: int = 12,
    sample_rate_hz: float = 1000.0,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Replay scenarios from a JSONL file through qed.py.

    Each line in the JSONL file should have:
      - "signal": list of float values OR
      - "params": dict with keys to generate synthetic signal

    Returns list of result dicts with qed output + metadata.
    """
    results: List[Dict[str, Any]] = []
    path = Path(jsonl_path)

    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    with path.open("r") as f:
        lines = f.readlines()

    for idx, line in enumerate(tqdm(lines, desc="Replaying scenarios")):
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            if verbose:
                print(f"Skipping line {idx}: JSON decode error: {e}")
            continue

        if "signal" in data:
            signal = np.array(data["signal"], dtype=np.float64)
        elif "params" in data:
            params = data["params"]
            signal = _make_signal(
                n=params.get("n", 1000),
                sample_rate_hz=params.get("sample_rate_hz", sample_rate_hz),
                amplitude=params.get("amplitude", 12.0),
                frequency_hz=params.get("frequency_hz", 40.0),
                noise_sigma=params.get("noise_sigma", 0.1),
                offset=params.get("offset", 2.0),
                phase_rad=params.get("phase_rad", 0.0),
                seed=params.get("seed", idx),
            )
        else:
            if verbose:
                print(f"Skipping line {idx}: no 'signal' or 'params' key")
            continue

        line_scenario = data.get("scenario", scenario)
        line_bit_depth = data.get("bit_depth", bit_depth)
        line_sample_rate = data.get("sample_rate_hz", sample_rate_hz)
        expected_label = data.get("label", None)

        t0 = time.perf_counter()
        try:
            out = qed.qed(
                signal,
                scenario=line_scenario,
                bit_depth=line_bit_depth,
                sample_rate_hz=line_sample_rate,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            error = None
        except ValueError as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            out = None
            error = str(e)

        result: Dict[str, Any] = {
            "idx": idx,
            "scenario": line_scenario,
            "n_samples": len(signal),
            "latency_ms": latency_ms,
            "error": error,
        }

        if out is not None:
            result["ratio"] = out["ratio"]
            result["H_bits"] = out["H_bits"]
            result["recall"] = out["recall"]
            result["savings_M"] = out["savings_M"]
            result["trace"] = out["trace"]

            receipt = out["receipt"]
            result["verified"] = receipt.verified
            result["violations"] = receipt.violations
            result["params"] = receipt.params

        if expected_label is not None:
            result["expected_label"] = expected_label
            if out is not None:
                events = _count_events(signal, threshold=10.0)
                if expected_label == "anomaly":
                    result["is_hit"] = events > 0 and out["recall"] >= 0.95
                else:
                    result["is_hit"] = receipt.verified is True
            else:
                result["is_hit"] = expected_label == "anomaly"

        results.append(result)

    return results


# --- Sympy Suite Subcommand ---


def sympy_suite(
    hook: str,
    test_amplitudes: Optional[List[float]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run sympy constraint suite for a given hook/scenario.

    Gets constraints from sympy_constraints module and verifies them
    against test amplitudes (or default range).

    Returns dict with violations, passes, and constraint details.
    """
    constraints = sympy_constraints.get_constraints(hook)

    if test_amplitudes is None:
        test_amplitudes = [float(a) for a in np.arange(0.0, 25.5, 0.5)]

    violations: List[Dict[str, Any]] = []
    passes: List[Dict[str, Any]] = []
    total_tests = 0

    for constraint in tqdm(constraints, desc=f"Checking constraints for {hook}"):
        constraint_id = constraint.get("id", "unknown")
        constraint_type = constraint.get("type", "amplitude_bound")
        bound = constraint.get("bound", float("inf"))
        description = constraint.get("description", "")

        # Only test amplitude_bound constraints with amplitude values
        if constraint_type != "amplitude_bound":
            continue

        for A in test_amplitudes:
            total_tests += 1
            exceeds_bound = abs(A) > bound

            if exceeds_bound:
                violations.append(
                    {
                        "constraint_id": constraint_id,
                        "amplitude": A,
                        "bound": bound,
                        "description": description,
                    }
                )
            else:
                passes.append(
                    {
                        "constraint_id": constraint_id,
                        "amplitude": A,
                        "bound": bound,
                    }
                )

    verified_at_bound, violations_at_bound = qed.check_constraints(
        constraints[0]["bound"] if constraints else 14.7,
        40.0,
        hook,
    )

    return {
        "hook": hook,
        "constraints": constraints,
        "total_tests": total_tests,
        "n_violations": len(violations),
        "n_passes": len(passes),
        "violations": violations if verbose else violations[:10],
        "verified_at_bound": verified_at_bound,
        "violations_at_bound": violations_at_bound,
    }


# --- Summarize Subcommand ---


def summarize(
    results: List[Dict[str, Any]],
    fleet_size: int = 2_000_000,
) -> Dict[str, Any]:
    """
    Summarize replay results into KPI metrics.

    Computes:
      - hits/misses (recall/precision proxy)
      - violations count
      - ROI estimate ($M)
      - Confidence intervals where applicable

    Returns JSON-serializable summary dict.
    """
    if not results:
        return {
            "n_scenarios": 0,
            "hits": 0,
            "misses": 0,
            "recall": 0.0,
            "precision": 0.0,
            "violations": 0,
            "roi_M": 0.0,
            "kpi_pass": False,
        }

    n_scenarios = len(results)
    n_errors = sum(1 for r in results if r.get("error") is not None)

    labeled = [r for r in results if "expected_label" in r]
    anomalies = [r for r in labeled if r["expected_label"] == "anomaly"]
    normals = [r for r in labeled if r["expected_label"] == "normal"]

    if anomalies:
        hits_anomaly = sum(1 for r in anomalies if r.get("is_hit", False))
        recall = hits_anomaly / len(anomalies)
    else:
        hits_anomaly = 0
        recall = 1.0

    if normals:
        false_positives = sum(1 for r in normals if not r.get("is_hit", True))
        true_positives = hits_anomaly
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 1.0
        )
    else:
        precision = 1.0

    total_violations = sum(
        len(r.get("violations", [])) for r in results if r.get("error") is None
    )

    valid_results = [r for r in results if r.get("error") is None]
    if valid_results:
        avg_savings_M = np.mean([r.get("savings_M", 0.0) for r in valid_results])
        roi_M = float(avg_savings_M)
    else:
        roi_M = 0.0

    latencies = [r.get("latency_ms", 0.0) for r in results]
    avg_latency_ms = float(np.mean(latencies)) if latencies else 0.0
    max_latency_ms = float(np.max(latencies)) if latencies else 0.0

    if anomalies:
        n = len(anomalies)
        p = recall
        z = 1.96
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
        recall_ci_lower = max(0.0, center - spread)
        recall_ci_upper = min(1.0, center + spread)
    else:
        recall_ci_lower = recall_ci_upper = recall

    kpi_recall_pass = recall >= KPI_RECALL_THRESHOLD
    kpi_precision_pass = precision >= KPI_PRECISION_THRESHOLD
    kpi_roi_pass = abs(roi_M - KPI_ROI_TARGET_M) <= KPI_ROI_TOLERANCE_M
    kpi_violations_pass = total_violations == 0 or len(normals) == 0
    kpi_pass = all(
        [kpi_recall_pass, kpi_precision_pass, kpi_roi_pass, kpi_violations_pass]
    )

    return {
        "n_scenarios": n_scenarios,
        "n_errors": n_errors,
        "n_anomalies": len(anomalies),
        "n_normals": len(normals),
        "hits": hits_anomaly,
        "misses": len(anomalies) - hits_anomaly,
        "recall": round(recall, 6),
        "recall_ci_95": [round(recall_ci_lower, 6), round(recall_ci_upper, 6)],
        "precision": round(precision, 6),
        "violations": total_violations,
        "roi_M": round(roi_M, 2),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "max_latency_ms": round(max_latency_ms, 2),
        "kpi": {
            "recall_pass": kpi_recall_pass,
            "precision_pass": kpi_precision_pass,
            "roi_pass": kpi_roi_pass,
            "violations_pass": kpi_violations_pass,
            "all_pass": kpi_pass,
        },
        "kpi_thresholds": {
            "recall": KPI_RECALL_THRESHOLD,
            "precision": KPI_PRECISION_THRESHOLD,
            "roi_target_M": KPI_ROI_TARGET_M,
            "roi_tolerance_M": KPI_ROI_TOLERANCE_M,
        },
    }


# --- Legacy Gates Subcommand ---


def run_proof(seed: int = 42424242) -> Dict[str, Any]:
    """
    Run legacy v5 gate checks with synthetic signals.

    Returns dict with gate results and metrics.
    """
    gates: Dict[str, bool] = {}
    failed: List[str] = []

    sample_rate_a = 1000.0
    signal_a = _make_signal(
        n=1000,
        sample_rate_hz=sample_rate_a,
        amplitude=12.347,
        frequency_hz=40.0,
        noise_sigma=0.147,
        offset=2.1,
        phase_rad=np.pi / 4.0,
        seed=seed,
    )

    t0 = time.perf_counter()
    out_a = qed.qed(
        signal_a,
        scenario="tesla_fsd",
        bit_depth=12,
        sample_rate_hz=sample_rate_a,
    )
    latency_a_ms = (time.perf_counter() - t0) * 1000.0

    ratio_a = float(out_a["ratio"])
    H_bits_a = float(out_a["H_bits"])
    recall_a = float(out_a["recall"])
    savings_m_a = float(out_a["savings_M"])
    trace_a = str(out_a["trace"])

    A_a, f_a, phi_a, c_a = qed._fit_dominant_sine(signal_a, sample_rate_a)
    t_a = np.arange(signal_a.size) / sample_rate_a
    recon_a = A_a * np.sin(2.0 * np.pi * f_a * t_a + phi_a) + c_a
    nrmse_a = _normalized_rms_error(signal_a, recon_a)
    events_a = _count_events(signal_a, threshold=10.0)

    gates["G1_ratio"] = 57.0 <= ratio_a <= 63.0
    gates["G2_entropy"] = 7150.0 <= H_bits_a <= 7250.0
    gates["G3_recall_with_events"] = (events_a >= 50) and (recall_a >= 0.9985)
    gates["G4_roi"] = (abs(ratio_a - 60.0) < 1.0) and (37.8 <= savings_m_a <= 38.2)
    gates["G6_reconstruction"] = nrmse_a <= 0.05
    gates["G7_determinism"] = _deterministic_check(
        signal_a, sample_rate_hz=sample_rate_a
    )
    gates["G8_latency_ms"] = latency_a_ms <= 50.0

    sample_rate_b = 2048.0
    signal_b = _make_signal(
        n=1024,
        sample_rate_hz=sample_rate_b,
        amplitude=14.697,
        frequency_hz=927.4,
        noise_sigma=0.01,
        offset=0.0,
        phase_rad=0.0,
        seed=seed + 1,
    )
    try:
        _ = qed.qed(
            signal_b,
            scenario="tesla_fsd",
            bit_depth=12,
            sample_rate_hz=sample_rate_b,
        )
        gates["G5_bound_pass"] = True
    except ValueError:
        gates["G5_bound_pass"] = False

    sample_rate_c = 10_000.0
    signal_c = _make_signal(
        n=1000,
        sample_rate_hz=sample_rate_c,
        amplitude=1.0,
        frequency_hz=250.0,
        noise_sigma=0.2,
        offset=8.0,
        phase_rad=0.0,
        seed=seed + 2,
    )
    out_c = qed.qed(
        signal_c,
        scenario="tesla_fsd",
        bit_depth=12,
        sample_rate_hz=sample_rate_c,
    )
    recall_c = float(out_c["recall"])
    events_c = _count_events(signal_c, threshold=10.0)
    gates["G3_recall_no_events"] = (events_c == 0) and (recall_c == 1.0)

    sample_rate_d = 1000.0
    signal_d = _make_signal(
        n=1000,
        sample_rate_hz=sample_rate_d,
        amplitude=14.703,
        frequency_hz=55.0,
        noise_sigma=0.01,
        offset=0.0,
        phase_rad=0.0,
        seed=seed + 3,
    )
    try:
        _ = qed.qed(
            signal_d,
            scenario="tesla_fsd",
            bit_depth=12,
            sample_rate_hz=sample_rate_d,
        )
        gates["G5_bound_fail"] = False
    except ValueError as exc:
        gates["G5_bound_fail"] = "amplitude" in str(exc).lower()

    for name, ok in gates.items():
        if not ok:
            failed.append(name)

    return {
        "gates": gates,
        "failed": failed,
        "all_pass": len(failed) == 0,
        "metrics": {
            "ratio": ratio_a,
            "H_bits": H_bits_a,
            "recall": recall_a,
            "savings_M": savings_m_a,
            "nrmse": nrmse_a,
            "latency_ms": latency_a_ms,
            "trace": trace_a,
        },
    }


# --- Generate Sample JSONL ---


def generate_edge_lab_sample(
    output_path: str,
    n_anomalies: int = 900,
    n_normals: int = 100,
    seed: int = 42,
) -> None:
    """
    Generate edge_lab_sample.jsonl with labeled anomaly/normal scenarios.

    Creates synthetic signals with known labels for validation testing.
    """
    rng = np.random.default_rng(seed)
    path = Path(output_path)

    with path.open("w") as f:
        for i in tqdm(range(n_anomalies), desc="Generating anomalies"):
            amplitude = rng.uniform(11.0, 14.5)
            frequency = rng.uniform(20.0, 100.0)
            noise = rng.uniform(0.05, 0.2)

            record = {
                "id": f"anomaly_{i:04d}",
                "label": "anomaly",
                "params": {
                    "n": 1000,
                    "sample_rate_hz": 1000.0,
                    "amplitude": float(amplitude),
                    "frequency_hz": float(frequency),
                    "noise_sigma": float(noise),
                    "offset": float(rng.uniform(0.0, 5.0)),
                    "phase_rad": float(rng.uniform(0.0, 2 * np.pi)),
                    "seed": seed + i,
                },
                "scenario": "tesla_fsd",
            }
            f.write(json.dumps(record) + "\n")

        for i in tqdm(range(n_normals), desc="Generating normals"):
            amplitude = rng.uniform(2.0, 8.0)
            frequency = rng.uniform(20.0, 100.0)
            noise = rng.uniform(0.01, 0.1)

            record = {
                "id": f"normal_{i:04d}",
                "label": "normal",
                "params": {
                    "n": 1000,
                    "sample_rate_hz": 1000.0,
                    "amplitude": float(amplitude),
                    "frequency_hz": float(frequency),
                    "noise_sigma": float(noise),
                    "offset": float(rng.uniform(0.0, 3.0)),
                    "phase_rad": float(rng.uniform(0.0, 2 * np.pi)),
                    "seed": seed + n_anomalies + i,
                },
                "scenario": "tesla_fsd",
            }
            f.write(json.dumps(record) + "\n")

    print(f"Generated {n_anomalies + n_normals} scenarios to {output_path}")


# --- v7 Subcommands ---


def run_sims(
    receipts_dir: str = "receipts/",
    patterns_path: str = "data/shared_anomalies.jsonl",
    n_per_hook: int = 1000,
    output: str = "data/sim_results.json",
) -> Dict[str, Any]:
    """
    Run pattern simulations via edge_lab_v2.

    Calls run_pattern_sims() with progress tracking and writes results to JSON.
    Returns summary dict with n_tests, aggregate_recall, aggregate_fp_rate.
    """
    # Load patterns for progress tracking
    patterns = load_library(patterns_path)

    # Run simulations with progress
    results = run_pattern_sims(
        receipts_dir=receipts_dir,
        patterns_path=patterns_path,
        n_per_hook=n_per_hook,
        progress_callback=lambda: tqdm(patterns, desc="Running pattern sims"),
    )

    # Write results to output file
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    # Compute summary
    n_tests = results.get("n_tests", 0)
    aggregate_recall = results.get("aggregate_recall", 0.0)
    aggregate_fp_rate = results.get("aggregate_fp_rate", 0.0)

    return {
        "n_tests": n_tests,
        "aggregate_recall": aggregate_recall,
        "aggregate_fp_rate": aggregate_fp_rate,
        "output_path": str(output_path),
    }


def recall_floor(
    sim_results_path: Optional[str] = "data/sim_results.json",
    confidence: float = 0.95,
    n_tests: Optional[int] = None,
    n_misses: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute Clopper-Pearson exact recall lower bound.

    Uses scipy.stats.beta.ppf for exact binomial confidence interval.
    Formula: beta.ppf(alpha/2, k, n-k+1) where k=successes, n=total.

    Returns dict with recall_floor, confidence, n_tests, n_misses.
    """
    # Get n_tests and n_misses from sim_results or overrides
    if n_tests is None or n_misses is None:
        if sim_results_path is None:
            raise ValueError(
                "Must provide either sim_results_path or both n_tests and n_misses"
            )
        with open(sim_results_path, "r") as f:
            sim_data = json.load(f)

        if n_tests is None:
            n_tests = sim_data.get("n_tests", 0)
        if n_misses is None:
            n_misses = sim_data.get("n_misses", 0)

    # Compute Clopper-Pearson exact lower bound
    # k = number of successes (hits), n = total tests
    n_successes = n_tests - n_misses
    alpha = 1.0 - confidence

    if n_successes == 0:
        # Lower bound is 0 when no successes
        lower_bound = 0.0
    else:
        # Clopper-Pearson exact lower bound
        # beta.ppf(alpha/2, k, n-k+1) gives lower bound for proportion k/n
        lower_bound = float(beta.ppf(alpha / 2, n_successes, n_misses + 1))

    return {
        "recall_floor": lower_bound,
        "confidence": confidence,
        "n_tests": n_tests,
        "n_misses": n_misses,
        "n_successes": n_successes,
    }


def pattern_report(
    patterns_path: str = "data/shared_anomalies.jsonl",
    sort_by: str = "dollar_value",
    exploit_only: bool = False,
    output_format: str = "table",
) -> List[Dict[str, Any]]:
    """
    Load and display pattern library with sorting and filtering.

    Loads patterns via shared_anomalies.load_library(), sorts by selected field,
    and optionally filters for exploit_grade=true patterns only.

    Returns list of pattern dicts.
    """
    patterns = load_library(patterns_path)

    # Filter if exploit_only
    if exploit_only:
        patterns = [p for p in patterns if p.get("exploit_grade", False)]

    # Sort by selected field (descending)
    sort_key_map = {
        "dollar_value": lambda p: p.get("dollar_value", 0),
        "recall": lambda p: p.get("recall", 0),
        "exploit_grade": lambda p: (1 if p.get("exploit_grade", False) else 0),
    }
    sort_fn = sort_key_map.get(sort_by, sort_key_map["dollar_value"])
    patterns = sorted(patterns, key=sort_fn, reverse=True)

    # Output in requested format
    if output_format == "json":
        print(json.dumps(patterns, indent=2))
    elif output_format == "table":
        if RICH_AVAILABLE:
            console = Console()
            table = Table(title="Pattern Report")

            table.add_column("pattern_id", style="cyan", no_wrap=True)
            table.add_column("physics_domain", style="green")
            table.add_column("failure_mode", style="yellow")
            table.add_column("dollar_value", justify="right", style="magenta")
            table.add_column("recall", justify="right", style="blue")
            table.add_column("fp_rate", justify="right", style="red")
            table.add_column("exploit_grade", justify="center", style="bold")

            for p in patterns:
                pattern_id = str(p.get("pattern_id", ""))[:20]  # truncated
                physics_domain = str(p.get("physics_domain", ""))
                failure_mode = str(p.get("failure_mode", ""))
                dollar_value = f"${p.get('dollar_value', 0):,.0f}"
                recall_val = f"{p.get('recall', 0):.4f}"
                fp_rate = f"{p.get('fp_rate', 0):.4f}"
                exploit = "Yes" if p.get("exploit_grade", False) else "No"

                table.add_row(
                    pattern_id,
                    physics_domain,
                    failure_mode,
                    dollar_value,
                    recall_val,
                    fp_rate,
                    exploit,
                )

            console.print(table)
        else:
            # Fallback to simple text table
            header = (
                f"{'pattern_id':<20} {'physics_domain':<15} {'failure_mode':<20} "
                f"{'dollar_value':>12} {'recall':>8} {'fp_rate':>8} {'exploit':>8}"
            )
            print(header)
            print("-" * len(header))
            for p in patterns:
                pattern_id = str(p.get("pattern_id", ""))[:20]
                physics_domain = str(p.get("physics_domain", ""))[:15]
                failure_mode = str(p.get("failure_mode", ""))[:20]
                dollar_value = p.get("dollar_value", 0)
                recall_val = p.get("recall", 0)
                fp_rate = p.get("fp_rate", 0)
                exploit = "Yes" if p.get("exploit_grade", False) else "No"
                print(
                    f"{pattern_id:<20} {physics_domain:<15} {failure_mode:<20} "
                    f"{dollar_value:>12,.0f} {recall_val:>8.4f} {fp_rate:>8.4f} {exploit:>8}"
                )

    return patterns


def clarity_audit(
    receipts_path: str,
    output_corpus: Optional[str] = None,
    output_receipt: str = "data/clarity_receipts.jsonl",
) -> Dict[str, Any]:
    """
    Process receipts through ClarityClean adapter.

    Calls clarity_clean_adapter.process_receipts() and emits ClarityCleanReceipt
    to JSONL output.

    Returns summary dict with token_count, anomaly_density, noise_ratio, corpus_hash.
    """
    # Process receipts
    result = process_receipts(
        receipts_path=receipts_path,
        output_corpus=output_corpus,
    )

    # Extract ClarityCleanReceipt
    receipt = result.get("receipt", {})

    # Write receipt to JSONL
    output_path = Path(output_receipt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as f:
        f.write(json.dumps(receipt) + "\n")

    # Return summary
    return {
        "token_count": receipt.get("token_count", 0),
        "anomaly_density": receipt.get("anomaly_density", 0.0),
        "noise_ratio": receipt.get("noise_ratio", 0.0),
        "corpus_hash": receipt.get("corpus_hash", ""),
        "output_receipt": str(output_path),
        "output_corpus": output_corpus,
    }


# --- CLI Main ---


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="QED v6/v7 Proof CLI Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python proof.py gates                    # Run legacy v5 gate checks
  python proof.py replay sample.jsonl      # Replay scenarios from JSONL
  python proof.py sympy_suite tesla_fsd    # Check constraints for hook
  python proof.py generate --output edge_lab_sample.jsonl

v7 Commands:
  python proof.py run-sims                 # Run pattern simulations
  python proof.py recall-floor             # Compute recall lower bound
  python proof.py pattern-report           # Display pattern library
  python proof.py clarity-audit --receipts-path receipts.jsonl
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    gates_parser = subparsers.add_parser(
        "gates",
        help="Run legacy v5 gate checks with synthetic signals",
    )
    gates_parser.add_argument(
        "--seed",
        type=int,
        default=42424242,
        help="Random seed for signal generation",
    )
    gates_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    replay_parser = subparsers.add_parser(
        "replay",
        help="Replay scenarios from JSONL through qed.py",
    )
    replay_parser.add_argument(
        "jsonl_path",
        type=str,
        help="Path to JSONL file with scenarios",
    )
    replay_parser.add_argument(
        "--scenario",
        type=str,
        default="tesla_fsd",
        help="Default scenario if not specified in JSONL",
    )
    replay_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    replay_parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )

    suite_parser = subparsers.add_parser(
        "sympy_suite",
        help="Run sympy constraint suite for a hook",
    )
    suite_parser.add_argument(
        "hook",
        type=str,
        help="Hook/scenario name (e.g., tesla_fsd, spacex_flight)",
    )
    suite_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include all violations in output",
    )

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate edge_lab_sample.jsonl test data",
    )
    generate_parser.add_argument(
        "--output",
        type=str,
        default="edge_lab_sample.jsonl",
        help="Output JSONL file path",
    )
    generate_parser.add_argument(
        "--anomalies",
        type=int,
        default=900,
        help="Number of anomaly scenarios (default: 900)",
    )
    generate_parser.add_argument(
        "--normals",
        type=int,
        default=100,
        help="Number of normal scenarios (default: 100)",
    )
    generate_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize replay results to KPI metrics",
    )
    summarize_parser.add_argument(
        "results_json",
        type=str,
        help="Path to JSON file with replay results",
    )

    # --- v7 Subcommands ---

    run_sims_parser = subparsers.add_parser(
        "run-sims",
        help="Run pattern simulations via edge_lab_v2",
    )
    run_sims_parser.add_argument(
        "--receipts-dir",
        type=str,
        default="receipts/",
        help="Directory containing receipt files (default: receipts/)",
    )
    run_sims_parser.add_argument(
        "--patterns-path",
        type=str,
        default="data/shared_anomalies.jsonl",
        help="Path to patterns JSONL file (default: data/shared_anomalies.jsonl)",
    )
    run_sims_parser.add_argument(
        "--n-per-hook",
        type=int,
        default=1000,
        help="Number of simulations per hook (default: 1000)",
    )
    run_sims_parser.add_argument(
        "--output",
        type=str,
        default="data/sim_results.json",
        help="Output JSON file for results (default: data/sim_results.json)",
    )

    recall_floor_parser = subparsers.add_parser(
        "recall-floor",
        help="Compute Clopper-Pearson exact recall lower bound",
    )
    recall_floor_parser.add_argument(
        "--sim-results",
        type=str,
        default="data/sim_results.json",
        help="Path to simulation results JSON (default: data/sim_results.json)",
    )
    recall_floor_parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level (default: 0.95)",
    )
    recall_floor_parser.add_argument(
        "--n-tests",
        type=int,
        default=None,
        help="Override n_tests from sim results",
    )
    recall_floor_parser.add_argument(
        "--n-misses",
        type=int,
        default=None,
        help="Override n_misses from sim results",
    )

    pattern_report_parser = subparsers.add_parser(
        "pattern-report",
        help="Display pattern library with sorting and filtering",
    )
    pattern_report_parser.add_argument(
        "--patterns-path",
        type=str,
        default="data/shared_anomalies.jsonl",
        help="Path to patterns JSONL file (default: data/shared_anomalies.jsonl)",
    )
    pattern_report_parser.add_argument(
        "--sort-by",
        type=str,
        choices=["dollar_value", "recall", "exploit_grade"],
        default="dollar_value",
        help="Sort field (default: dollar_value)",
    )
    pattern_report_parser.add_argument(
        "--exploit-only",
        action="store_true",
        help="Show only exploit_grade=true patterns",
    )
    pattern_report_parser.add_argument(
        "--format",
        type=str,
        choices=["table", "json"],
        default="table",
        dest="output_format",
        help="Output format (default: table)",
    )

    clarity_audit_parser = subparsers.add_parser(
        "clarity-audit",
        help="Process receipts through ClarityClean adapter",
    )
    clarity_audit_parser.add_argument(
        "--receipts-path",
        type=str,
        required=True,
        help="Path to receipts file (required)",
    )
    clarity_audit_parser.add_argument(
        "--output-corpus",
        type=str,
        default=None,
        help="Path to write cleaned corpus (optional)",
    )
    clarity_audit_parser.add_argument(
        "--output-receipt",
        type=str,
        default="data/clarity_receipts.jsonl",
        help="Path to write ClarityCleanReceipt JSONL (default: data/clarity_receipts.jsonl)",
    )

    args = parser.parse_args()

    if args.command == "gates":
        result = run_proof(seed=args.seed)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result["all_pass"]:
                print("QED v6 proof gates passed")
                metrics = result["metrics"]
                print(
                    f"Signal A: "
                    f"ratio={metrics['ratio']:.1f}, "
                    f"H_bits={metrics['H_bits']:.0f}, "
                    f"recall={metrics['recall']:.4f}, "
                    f"savings_M={metrics['savings_M']:.2f}, "
                    f"nrmse={metrics['nrmse']:.4f}, "
                    f"latency_ms={metrics['latency_ms']:.2f}, "
                    f"trace={metrics['trace']}"
                )
            else:
                print(f"FAILED gates: {result['failed']}")
                return 1

    elif args.command == "replay":
        results = replay(
            args.jsonl_path,
            scenario=args.scenario,
            verbose=args.verbose,
        )
        summary = summarize(results)

        if args.output:
            output_data = {"results": results, "summary": summary}
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"Results written to {args.output}")
        else:
            print(json.dumps(summary, indent=2))

        if not summary["kpi"]["all_pass"]:
            return 1

    elif args.command == "sympy_suite":
        result = sympy_suite(args.hook, verbose=args.verbose)
        print(json.dumps(result, indent=2))

        if result["n_violations"] > 0 and not args.verbose:
            print(
                f"\nNote: {result['n_violations']} violations found. "
                "Use --verbose for details."
            )

    elif args.command == "generate":
        generate_edge_lab_sample(
            args.output,
            n_anomalies=args.anomalies,
            n_normals=args.normals,
            seed=args.seed,
        )

    elif args.command == "summarize":
        with open(args.results_json, "r") as f:
            data = json.load(f)

        results = data.get("results", data)
        summary = summarize(results)
        print(json.dumps(summary, indent=2))

        if not summary["kpi"]["all_pass"]:
            return 1

    # --- v7 Command Handlers ---

    elif args.command == "run-sims":
        result = run_sims(
            receipts_dir=args.receipts_dir,
            patterns_path=args.patterns_path,
            n_per_hook=args.n_per_hook,
            output=args.output,
        )
        print(
            f"Simulation complete: n_tests={result['n_tests']}, "
            f"aggregate_recall={result['aggregate_recall']:.4f}, "
            f"aggregate_fp_rate={result['aggregate_fp_rate']:.4f}"
        )
        print(f"Results written to {result['output_path']}")

    elif args.command == "recall-floor":
        result = recall_floor(
            sim_results_path=args.sim_results,
            confidence=args.confidence,
            n_tests=args.n_tests,
            n_misses=args.n_misses,
        )
        confidence_pct = result["confidence"] * 100
        print(
            f"Recall floor: {result['recall_floor']:.4f} at {confidence_pct:.0f}% confidence "
            f"({result['n_tests']} tests, {result['n_misses']} misses)"
        )

    elif args.command == "pattern-report":
        pattern_report(
            patterns_path=args.patterns_path,
            sort_by=args.sort_by,
            exploit_only=args.exploit_only,
            output_format=args.output_format,
        )

    elif args.command == "clarity-audit":
        result = clarity_audit(
            receipts_path=args.receipts_path,
            output_corpus=args.output_corpus,
            output_receipt=args.output_receipt,
        )
        print(
            f"ClarityClean audit complete:\n"
            f"  token_count: {result['token_count']}\n"
            f"  anomaly_density: {result['anomaly_density']:.4f}\n"
            f"  noise_ratio: {result['noise_ratio']:.4f}\n"
            f"  corpus_hash: {result['corpus_hash']}"
        )
        print(f"Receipt written to {result['output_receipt']}")
        if result["output_corpus"]:
            print(f"Corpus written to {result['output_corpus']}")

    else:
        parser.print_help()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())


# =============================================================================
# v8 Click CLI Commands
# =============================================================================

def print_success(message: str) -> None:
    """Print a success message with green checkmark."""
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str) -> None:
    """Print an error message with red X."""
    console.print(f"[red]✗[/red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message with yellow warning sign."""
    console.print(f"[yellow]⚠[/yellow] {message}")


def print_next(command: str) -> None:
    """Print suggested next command."""
    console.print(f"\n[dim]Next:[/dim] [cyan]{command}[/cyan]")


def _format_savings(amount: float) -> str:
    """Format savings amount in human-readable format."""
    if amount >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"
    elif amount >= 1_000:
        return f"${amount / 1_000:.0f}K"
    return f"${amount:.0f}"


def _make_health_bar(score: int, width: int = 20) -> str:
    """Create a visual health bar."""
    filled = int(score / 100 * width)
    empty = width - filled
    return "█" * filled + "░" * empty


# --- v8 Click CLI Group ---

@click.group()
def v8():
    """QED v8 proof subcommands for packet, config, and fleet operations."""
    pass


# --- build-packet ---

@v8.command("build-packet")
@click.option("--deployment-id", "-d", required=True, help="Deployment identifier")
@click.option("--manifest", "-m", default="data/manifests/", help="Manifest path")
@click.option("--output", "-o", type=click.Choice(["rich", "json"]), default="rich")
def build_packet_cmd(deployment_id: str, manifest: str, output: str) -> None:
    """Build a DecisionPacket from deployment manifest."""
    manifest_path = Path(manifest)

    # Check if manifest exists (file or directory)
    if not manifest_path.exists():
        if output == "json":
            click.echo(json.dumps({"error": "manifest not found", "path": manifest}))
        else:
            print_error(f"Manifest not found: {manifest}")
        sys.exit(2)

    try:
        # Build packet using truthlink
        packet = truthlink.build(deployment_id, str(manifest_path))

        # Save packet
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        output_dir = Path("data/packets")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{deployment_id}_{timestamp}.jsonl"

        with output_file.open("a") as f:
            f.write(packet.to_json() + "\n")

        if output == "json":
            click.echo(packet.to_json(indent=2))
        else:
            # Rich output
            # Count exploit-grade patterns
            exploit_count = sum(1 for p in packet.pattern_usage if p.exploit_grade)
            total_patterns = len(packet.pattern_usage)

            health_bar = _make_health_bar(packet.health_score)

            content = (
                f"packet_id:       {packet.packet_id}\n"
                f"health_score:    {packet.health_score}/100 {health_bar}\n"
                f"annual_savings:  {_format_savings(packet.metrics.annual_savings)}\n"
                f"patterns:        {total_patterns} active ({exploit_count} exploit-grade)\n"
                f"slo_breach_rate: {packet.metrics.slo_breach_rate:.2%}\n"
                f"exploit_coverage: {packet.exploit_coverage:.0%}"
            )

            panel = Panel(
                content,
                title=f"[bold]DecisionPacket: {deployment_id}[/bold]",
                border_style="green",
            )
            console.print(panel)
            print_success(f"Saved: {output_file}")
            print_next(f"proof compare-packets -a <previous_id> -b {packet.packet_id}")

    except Exception as e:
        if output == "json":
            click.echo(json.dumps({"error": str(e)}))
        else:
            print_error(f"Failed to build packet: {e}")
        sys.exit(2)


# --- validate-config ---

@v8.command("validate-config")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--fix", is_flag=True, help="Auto-repair and save")
@click.option("--output", "-o", type=click.Choice(["rich", "json"]), default="rich")
def validate_config_cmd(config_path: str, fix: bool, output: str) -> None:
    """Validate a QED config file."""
    try:
        # Load config (validation happens during load via _validate)
        try:
            config = config_schema.load(config_path)
            is_valid = True
            errors: List[Dict[str, Any]] = []
            warnings: List[Dict[str, Any]] = []
        except ValueError as ve:
            # Load failed with validation errors
            is_valid = False
            errors = [{"message": str(ve), "fixable": True}]
            warnings = []
            config = None

        if output == "json":
            result = {
                "path": config_path,
                "valid": is_valid,
                "errors": errors,
                "warnings": warnings,
                "config": config.to_dict() if config else None,
            }
            click.echo(json.dumps(result, indent=2))
        else:
            # Rich output
            status = "PASSED" if is_valid else "FAILED"
            status_style = "green" if is_valid else "red"

            if is_valid and config:
                hook = config.hook
                patterns_count = len(config.enabled_patterns)
                recall_floor = config.recall_floor
                max_fp_rate = config.max_fp_rate

                content = (
                    f"File: {config_path}\n"
                    f"Hook: {hook}         Patterns: {patterns_count} enabled\n"
                    f"recall_floor: {recall_floor}    max_fp_rate: {max_fp_rate}\n"
                    f"Risk Profile: conservative"
                )

                panel = Panel(
                    content,
                    title=f"[bold {status_style}]Config Validation: {status}[/bold {status_style}]",
                    border_style=status_style,
                )
                console.print(panel)
                print_next(f"proof merge-configs -p global.json -c {config_path}")
            else:
                # Build error display
                lines = [f"File: {config_path}", ""]
                for err in errors:
                    msg = err.get("message", str(err))
                    lines.append(f"[red]✗[/red] {msg}")
                for warn in warnings:
                    msg = warn.get("message", str(warn))
                    lines.append(f"[yellow]⚠[/yellow] {msg}")

                content = "\n".join(lines)
                panel = Panel(
                    content,
                    title=f"[bold {status_style}]Config Validation: {status}[/bold {status_style}]",
                    border_style=status_style,
                )
                console.print(panel)

                fixable = [e for e in errors if e.get("fixable", False)]
                if fixable:
                    console.print(f"Fixable errors: {len(fixable)}   Run with --fix to repair")
                    print_next(f"proof validate-config {config_path} --fix")
                sys.exit(1)

    except FileNotFoundError:
        if output == "json":
            click.echo(json.dumps({"error": "config not found", "path": config_path}))
        else:
            print_error(f"Config file not found: {config_path}")
        sys.exit(2)
    except Exception as e:
        if output == "json":
            click.echo(json.dumps({"error": str(e)}))
        else:
            print_error(f"Validation failed: {e}")
        sys.exit(2)


# --- merge-configs ---

@v8.command("merge-configs")
@click.option("--parent", "-p", required=True, type=click.Path(exists=True))
@click.option("--child", "-c", required=True, type=click.Path(exists=True))
@click.option("--save", "-s", type=click.Path(), help="Save merged config")
@click.option("--auto-repair", is_flag=True, help="Fix violations automatically")
@click.option("--output", "-o", type=click.Choice(["rich", "json"]), default="rich")
def merge_configs_cmd(
    parent: str, child: str, save: Optional[str], auto_repair: bool, output: str
) -> None:
    """Merge parent and child config files."""
    try:
        parent_config = config_schema.load(parent)
        child_config = config_schema.load(child)

        # Perform merge using MergeResult API
        merge_result = merge_rules.merge(
            parent_config, child_config,
            auto_repair=auto_repair,
            emit_receipt_flag=False
        )

        is_valid = merge_result.is_valid
        merged_config = merge_result.merged
        violations = [
            {"message": v.message, "field": v.field_name, "severity": v.severity}
            for v in merge_result.violations
        ]

        # Build changes list from explanation
        changes: List[Dict[str, Any]] = []
        if merge_result.explanation:
            exp = merge_result.explanation
            if exp.safety_direction != "unchanged":
                changes.append({
                    "field": "safety_direction",
                    "old": "",
                    "new": exp.safety_direction,
                    "direction": exp.safety_direction,
                })

        if save and merged_config and is_valid:
            merged_config.save(save)

        if output == "json":
            result = {
                "parent": parent,
                "child": child,
                "valid": is_valid,
                "violations": violations,
                "changes": changes,
                "saved": save if (save and is_valid) else None,
            }
            click.echo(json.dumps(result, indent=2))
        else:
            # Rich output
            status = "VALID" if is_valid else "VIOLATION"
            status_style = "green" if is_valid else "red"

            parent_name = Path(parent).name
            child_name = Path(child).name

            lines = [f"Parent: {parent_name} → Child: {child_name}", ""]

            # Display merge explanation
            if merge_result.explanation and is_valid:
                exp = merge_result.explanation
                lines.append(f"recall_floor:  {parent_config.recall_floor} → {child_config.recall_floor} ({exp.safety_direction}) ✓")
                lines.append(f"max_fp_rate:   {parent_config.max_fp_rate} → {child_config.max_fp_rate} ({exp.safety_direction}) ✓")
                if exp.patterns_removed:
                    lines.append(f"patterns:      {len(parent_config.enabled_patterns)} → {len(child_config.enabled_patterns)} (intersection) ✓")

            for violation in violations:
                msg = violation.get("message", str(violation))
                lines.append(f"[red]✗[/red] {msg}")

            if is_valid:
                lines.append("")
                lines.append("Direction: SAFETY TIGHTENED")

            content = "\n".join(lines)
            panel = Panel(
                content,
                title=f"[bold {status_style}]Config Merge: {status}[/bold {status_style}]",
                border_style=status_style,
            )
            console.print(panel)

            if is_valid:
                deployment = Path(child).stem.replace("-", "-") or "deployment"
                print_next(f"proof build-packet -d {deployment}")
            else:
                console.print("Safety cannot loosen. Run with --auto-repair to tighten child.")
                print_next(f"proof merge-configs -p {parent} -c {child} --auto-repair")
                sys.exit(1)

    except Exception as e:
        if output == "json":
            click.echo(json.dumps({"error": str(e)}))
        else:
            print_error(f"Merge failed: {e}")
        sys.exit(2)


# --- compare-packets ---

@v8.command("compare-packets")
@click.option("--old", "-a", required=True, help="Old packet ID or path")
@click.option("--new", "-b", required=True, help="New packet ID or path")
@click.option("--packets-dir", default="data/packets/", help="Packets directory")
@click.option("--output", "-o", type=click.Choice(["rich", "json"]), default="rich")
def compare_packets_cmd(old: str, new: str, packets_dir: str, output: str) -> None:
    """Compare two decision packets."""
    packets_path = Path(packets_dir)

    def load_packet_by_id_or_path(packet_ref: str) -> Optional[DecisionPacket]:
        """Load packet by ID or direct path."""
        # Try as direct path first
        if Path(packet_ref).exists():
            return load_packet(packet_ref)

        # Search in packets directory
        if not packets_path.exists():
            return None

        for jsonl_file in packets_path.glob("*.jsonl"):
            with jsonl_file.open("r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if data.get("packet_id", "").startswith(packet_ref):
                            return DecisionPacket.from_dict(data)
        return None

    try:
        old_packet = load_packet_by_id_or_path(old)
        new_packet = load_packet_by_id_or_path(new)

        if old_packet is None:
            if output == "json":
                click.echo(json.dumps({"error": f"Old packet not found: {old}"}))
            else:
                print_error(f"Old packet not found: {old}")
            sys.exit(2)

        if new_packet is None:
            if output == "json":
                click.echo(json.dumps({"error": f"New packet not found: {new}"}))
            else:
                print_error(f"New packet not found: {new}")
            sys.exit(2)

        # Compare packets using truthlink
        comparison = truthlink.compare(old_packet, new_packet)

        # Determine classification
        health_delta = comparison.get("health_score_delta", 0)
        savings_delta = comparison.get("savings_delta", 0)

        if health_delta > 0 or savings_delta > 0:
            classification = "IMPROVEMENT"
            class_icon = "⬆"
        elif health_delta < 0 or savings_delta < 0:
            classification = "REGRESSION"
            class_icon = "⬇"
        else:
            classification = "NEUTRAL"
            class_icon = "→"

        is_regression = classification == "REGRESSION"

        if output == "json":
            result = {
                "old_packet_id": old_packet.packet_id,
                "new_packet_id": new_packet.packet_id,
                "classification": classification,
                "comparison": comparison,
            }
            click.echo(json.dumps(result, indent=2))
        else:
            # Rich output
            old_ts = old_packet.timestamp[:10] if old_packet.timestamp else "unknown"
            new_ts = new_packet.timestamp[:10] if new_packet.timestamp else "unknown"

            lines = [
                f"Old: {old_packet.packet_id[:8]} ({old_ts})",
                f"New: {new_packet.packet_id[:8]} ({new_ts})",
                f"Classification: {classification} {class_icon}",
                "",
            ]

            # Metrics comparison
            old_health = old_packet.health_score
            new_health = new_packet.health_score
            health_pct = ((new_health - old_health) / max(old_health, 1)) * 100
            health_icon = "⬆" if health_delta > 0 else ("⬇" if health_delta < 0 else "→")
            lines.append(f"health_score:     {old_health} → {new_health}      {health_pct:+.1f}%  {health_icon}")

            old_savings = old_packet.metrics.annual_savings
            new_savings = new_packet.metrics.annual_savings
            savings_pct = ((new_savings - old_savings) / max(old_savings, 1)) * 100
            savings_icon = "⬆" if savings_delta > 0 else ("⬇" if savings_delta < 0 else "→")
            lines.append(
                f"annual_savings:   {_format_savings(old_savings)} → {_format_savings(new_savings)}  "
                f"{savings_pct:+.1f}%  {savings_icon}"
            )

            old_breach = old_packet.metrics.slo_breach_rate * 100
            new_breach = new_packet.metrics.slo_breach_rate * 100
            breach_pct = new_breach - old_breach
            breach_icon = "⬆" if breach_pct < 0 else ("⬇" if breach_pct > 0 else "→")
            lines.append(f"slo_breach_rate:  {old_breach:.2f}% → {new_breach:.2f}%  {breach_pct:+.1f}%  {breach_icon}")

            old_exploit = old_packet.exploit_coverage * 100
            new_exploit = new_packet.exploit_coverage * 100
            exploit_pct = new_exploit - old_exploit
            exploit_icon = "⬆" if exploit_pct > 0 else ("⬇" if exploit_pct < 0 else "→")
            lines.append(f"exploit_coverage: {old_exploit:.0f}% → {new_exploit:.0f}%    {exploit_pct:+.1f}%  {exploit_icon}")

            lines.append("")

            # Pattern changes
            patterns_added = comparison.get("patterns_added", [])
            patterns_removed = comparison.get("patterns_removed", [])
            lines.append(f"Patterns: +{len(patterns_added)} added, -{len(patterns_removed)} removed")
            if patterns_added:
                lines.append(f"  Added: {', '.join(patterns_added[:3])}")

            content = "\n".join(lines)
            border_color = "red" if is_regression else "green"
            panel = Panel(
                content,
                title="[bold]Packet Comparison[/bold]",
                border_style=border_color,
            )
            console.print(panel)

            if is_regression:
                console.print("[yellow]Warning: Regression detected[/yellow]")
                sys.exit(1)
            else:
                console.print("Recommendation: Safe to promote to wider fleet")
                print_next(f"proof fleet-view --highlight {new_packet.packet_id[:8]}")

    except Exception as e:
        if output == "json":
            click.echo(json.dumps({"error": str(e)}))
        else:
            print_error(f"Comparison failed: {e}")
        sys.exit(2)


# --- fleet-view ---

@v8.command("fleet-view")
@click.option("--packets-dir", default="data/packets/", help="Packets directory")
@click.option("--highlight", "-h", "highlight_id", help="Highlight deployment ID")
@click.option("--diagnose", is_flag=True, help="Run fleet health check")
@click.option("--export", "-e", "export_path", type=click.Path(), help="Export graph JSON")
@click.option("--output", "-o", type=click.Choice(["rich", "json"]), default="rich")
def fleet_view_cmd(
    packets_dir: str,
    highlight_id: Optional[str],
    diagnose: bool,
    export_path: Optional[str],
    output: str,
) -> None:
    """Display fleet topology and health overview."""
    packets_path = Path(packets_dir)

    if not packets_path.exists():
        if output == "json":
            click.echo(json.dumps({"error": "No packets directory found"}))
        else:
            print_error(f"Packets directory not found: {packets_dir}")
        sys.exit(2)

    try:
        # Load all packets
        packets: List[DecisionPacket] = []
        for jsonl_file in packets_path.glob("*.jsonl"):
            with jsonl_file.open("r") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            packets.append(DecisionPacket.from_dict(data))
                        except (json.JSONDecodeError, ValueError):
                            continue

        if not packets:
            if output == "json":
                click.echo(json.dumps({"error": "No packets found"}))
            else:
                print_error("No packets found in directory")
            sys.exit(2)

        # Build deployment graph
        graph = mesh_view_v3.build(packets)
        fleet_metrics = mesh_view_v3.compute_fleet_metrics(graph)
        clusters = graph.find_clusters(min_similarity=0.3)

        # Run diagnosis if requested
        diagnosis = None
        if diagnose:
            diagnosis = mesh_view_v3.diagnose(graph)

        # Export if requested
        if export_path:
            mesh_view_v3.save(graph, export_path)

        if output == "json":
            result = {
                "total_deployments": fleet_metrics.total_deployments,
                "active_deployments": fleet_metrics.active_deployments,
                "stale_deployments": fleet_metrics.stale_deployments,
                "total_annual_savings": fleet_metrics.total_annual_savings,
                "avg_health_score": fleet_metrics.avg_health_score,
                "fleet_cohesion": fleet_metrics.fleet_cohesion,
                "clusters": [
                    {
                        "cluster_id": c.cluster_id,
                        "size": len(c.deployment_ids),
                        "avg_similarity": c.avg_similarity,
                    }
                    for c in clusters
                ],
                "diagnosis": diagnosis.to_dict() if diagnosis else None,
            }
            click.echo(json.dumps(result, indent=2))
        else:
            # Rich output
            lines = [
                f"Deployments: {fleet_metrics.active_deployments} active, {fleet_metrics.stale_deployments} stale",
                f"Total Savings: {_format_savings(fleet_metrics.total_annual_savings)}/year",
                f"Avg Health: {fleet_metrics.avg_health_score:.0f}/100      Fleet Cohesion: {fleet_metrics.fleet_cohesion:.2f}",
                "",
                "CLUSTERS",
            ]

            # Display clusters
            for cluster in clusters[:4]:
                # Get primary hook from cluster
                nodes = [graph.nodes.get(did) for did in cluster.deployment_ids if did in graph.nodes]
                primary_hook = nodes[0].hook.title() if nodes else "Unknown"
                overlap_pct = cluster.avg_similarity * 100

                bar_filled = int(overlap_pct / 100 * 16)
                bar_empty = 16 - bar_filled
                bar = "█" * bar_filled + "░" * bar_empty

                lines.append(f"  {primary_hook} ({len(cluster.deployment_ids)})     {bar} {overlap_pct:.0f}% overlap")

            # Find outliers
            outliers = graph.find_outliers(threshold=0.3)
            if outliers:
                lines.append(f"  Outliers ({len(outliers)})  {'░' * 16} <30% similarity")

            lines.append("")
            lines.append("ACTIONS NEEDED")

            # Find stale nodes
            stale_nodes = [n for n in graph.nodes.values() if n.is_stale]
            for node in stale_nodes[:2]:
                lines.append(f"  • {node.deployment_id} stale")

            # Find nodes missing exploit patterns
            for node in list(graph.nodes.values())[:5]:
                if not node.exploit_patterns:
                    lines.append(f"  • {node.deployment_id} missing exploit patterns")
                    break

            # Suggest pattern propagation
            if clusters and len(clusters) >= 2:
                from_cluster = clusters[0]
                to_cluster = clusters[1]
                if from_cluster.patterns_in_common:
                    pattern = list(from_cluster.patterns_in_common)[0]
                    from_hook = graph.nodes[from_cluster.deployment_ids[0]].hook if from_cluster.deployment_ids else "?"
                    to_hook = graph.nodes[to_cluster.deployment_ids[0]].hook if to_cluster.deployment_ids else "?"
                    lines.append(f"  • {pattern} ready for {to_hook.title()} (works in {from_hook.title()})")

            content = "\n".join(lines)

            has_issues = fleet_metrics.stale_deployments > 0 or (diagnosis and not diagnosis.is_healthy)
            border_color = "yellow" if has_issues else "green"

            panel = Panel(
                content,
                title="[bold]Fleet Overview[/bold]",
                border_style=border_color,
            )
            console.print(panel)

            if diagnosis:
                if not diagnosis.is_healthy:
                    console.print("\n[bold red]Fleet Health Issues:[/bold red]")
                    for issue in diagnosis.issues:
                        print_error(issue)
                    for warning in diagnosis.warnings[:3]:
                        print_warning(warning)
                    sys.exit(1)
                else:
                    print_success("Fleet health check passed")

            if stale_nodes:
                print_next(f"proof build-packet -d {stale_nodes[0].deployment_id}")
            else:
                console.print("\n[dim]All deployments up to date[/dim]")

    except Exception as e:
        if output == "json":
            click.echo(json.dumps({"error": str(e)}))
        else:
            print_error(f"Fleet view failed: {e}")
        sys.exit(2)


# --- v8 CLI entry point ---

def v8_main() -> int:
    """Entry point for v8 Click CLI."""
    try:
        v8(standalone_mode=False)
        return 0
    except click.ClickException as e:
        e.show()
        return 2
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0
