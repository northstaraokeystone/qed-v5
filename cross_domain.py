"""
Cross-domain validation for QED v7.

This module defines which cross-domain mappings are permitted and validates
them before use. Only approved mappings with explicit validation pass can
be used for cross-company pattern reuse.

Initial approved mappings:
- Tesla battery_thermal → SpaceX (Starship battery packs)
- Tesla comms → Starlink (link quality)

All other cross-domain uses require explicit validation and approval.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import warnings

import numpy as np

from physics_injection import inject_perturbation
from qed import qed as qed_run
from shared_anomalies import AnomalyPattern, load_library


# =============================================================================
# Cross-Domain Mapping Registry
# =============================================================================

# Structure: {(source_hook, physics_domain): [target_hooks]}
CROSS_DOMAIN_MAP: Dict[Tuple[str, str], List[str]] = {
    ("tesla", "battery_thermal"): ["spacex"],   # Starship battery packs
    ("tesla", "comms"): ["starlink"],           # Link quality patterns
}


# =============================================================================
# Physics Constraints Registry
# =============================================================================

# Per mapping, define safe bounds for sanity-checking before validation
PHYSICS_CONSTRAINTS: Dict[str, Dict[str, Any]] = {
    "battery_thermal": {
        "temperature_min_c": -40.0,         # Minimum operating temperature
        "temperature_max_c": 85.0,          # Maximum operating temperature
        "gradient_max_c_per_sec": 5.0,      # Max thermal gradient (C/s)
        "gradient_sustained_max": 2.0,      # Sustained gradient limit
        "delta_t_max_c": 30.0,              # Max temp delta across pack
    },
    "comms": {
        "signal_min_dbm": -120.0,           # Minimum signal strength (dBm)
        "signal_max_dbm": -30.0,            # Maximum signal strength (dBm)
        "dropout_duration_max_ms": 5000,    # Max dropout duration (ms)
        "dropout_rate_max_per_hour": 10,    # Max dropouts per hour
        "latency_max_ms": 500,              # Max latency threshold (ms)
    },
}


# =============================================================================
# Validation Thresholds
# =============================================================================

VALIDATION_RECALL_THRESHOLD = 0.99    # Must achieve >= 99% recall
VALIDATION_FP_RATE_THRESHOLD = 0.01   # Must achieve <= 1% FP rate


# =============================================================================
# Validation Result Dataclass
# =============================================================================

@dataclass
class CrossDomainValidation:
    """Result of a cross-domain validation test."""
    source_hook: str
    target_hook: str
    pattern_id: str
    validation_recall: float
    validation_fp_rate: float
    passed: bool
    validated_at: str                              # ISO timestamp
    physics_constraints_checked: List[str] = field(default_factory=list)
    n_tests: int = 1000
    notes: str = ""


# =============================================================================
# Validation Functions
# =============================================================================

def validate_cross_domain(pattern: Dict[str, Any], target_hook: str) -> bool:
    """
    Validate whether a pattern can be used in a cross-domain context.

    Checks:
    1. Mapping exists in CROSS_DOMAIN_MAP
    2. Pattern's physics_domain matches a permitted mapping
    3. Target hook is in allowed targets list
    4. Physics constraints are not violated

    Args:
        pattern: Dict containing at minimum 'physics_domain' and 'source_hook' keys.
                 May also contain domain-specific parameters for constraint checking.
        target_hook: The target hook to validate transfer to.

    Returns:
        True only if all checks pass, False otherwise.
    """
    # Extract required fields
    physics_domain = pattern.get("physics_domain", "")
    source_hook = pattern.get("source_hook", pattern.get("hooks", [""])[0] if isinstance(pattern.get("hooks"), list) else "")

    if not physics_domain or not source_hook:
        return False

    # Check 1 & 2: Mapping exists for (source_hook, physics_domain)
    mapping_key = (source_hook.lower(), physics_domain.lower())
    if mapping_key not in CROSS_DOMAIN_MAP:
        return False

    # Check 3: Target hook is in allowed targets
    allowed_targets = CROSS_DOMAIN_MAP[mapping_key]
    if target_hook.lower() not in [t.lower() for t in allowed_targets]:
        return False

    # Check 4: Physics constraints are not violated
    if not _check_physics_constraints(pattern, physics_domain):
        return False

    return True


def _check_physics_constraints(pattern: Dict[str, Any], physics_domain: str) -> bool:
    """
    Check that pattern parameters don't violate physics constraints.

    Args:
        pattern: Pattern dict with domain-specific parameters
        physics_domain: The physics domain to check constraints for

    Returns:
        True if all constraints pass, False if any violated
    """
    domain_key = physics_domain.lower()
    if domain_key not in PHYSICS_CONSTRAINTS:
        # No constraints defined for this domain - allow but warn
        warnings.warn(f"No physics constraints defined for domain: {physics_domain}")
        return True

    constraints = PHYSICS_CONSTRAINTS[domain_key]
    params = pattern.get("params", pattern)

    if domain_key == "battery_thermal":
        # Check temperature bounds
        temp = params.get("temperature_c", params.get("temperature"))
        if temp is not None:
            if not (constraints["temperature_min_c"] <= temp <= constraints["temperature_max_c"]):
                return False

        # Check gradient
        gradient = params.get("gradient_c_per_sec", params.get("thermal_gradient"))
        if gradient is not None:
            if abs(gradient) > constraints["gradient_max_c_per_sec"]:
                return False

    elif domain_key == "comms":
        # Check signal strength bounds
        signal = params.get("signal_dbm", params.get("signal_strength"))
        if signal is not None:
            if not (constraints["signal_min_dbm"] <= signal <= constraints["signal_max_dbm"]):
                return False

        # Check dropout duration
        dropout_ms = params.get("dropout_duration_ms", params.get("dropout_ms"))
        if dropout_ms is not None:
            if dropout_ms > constraints["dropout_duration_max_ms"]:
                return False

    return True


def run_cross_domain_validation(
    pattern: Dict[str, Any],
    target_hook: str,
    receipts_path: str,
    n_injections: int = 1000,
) -> CrossDomainValidation:
    """
    Run cross-domain validation test with physics-based perturbation injection.

    Performs the same 1000-injection test as native hook validation:
    1. Calls physics_injection.inject_perturbation() to create anomalies
    2. Runs qed.run() to detect anomalies
    3. Computes recall and FP rate on target domain

    Args:
        pattern: Pattern dict with physics_domain and parameters
        target_hook: Target hook to validate transfer to
        receipts_path: Path to receipts file for loading test windows
        n_injections: Number of injection tests to run (default 1000)

    Returns:
        CrossDomainValidation with test results
    """
    physics_domain = pattern.get("physics_domain", "unknown")
    source_hook = pattern.get("source_hook", pattern.get("hooks", ["unknown"])[0] if isinstance(pattern.get("hooks"), list) else "unknown")
    pattern_id = pattern.get("pattern_id", "unknown")

    # Emit warning for safety patterns requiring human signoff
    if _is_safety_pattern(pattern):
        warnings.warn(
            f"Safety pattern {pattern_id} requires human signoff for cross-domain use. "
            "Validation results are informational only - do not auto-approve."
        )

    # Track physics constraints that were checked
    constraints_checked = list(PHYSICS_CONSTRAINTS.get(physics_domain.lower(), {}).keys())

    # Generate synthetic test windows if no receipts file exists
    test_windows = _load_or_generate_test_windows(receipts_path, n_injections)

    # Run injection tests
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for i, window in enumerate(test_windows):
        # Inject perturbation (this creates the anomaly)
        seed = i  # Reproducible
        perturbed = inject_perturbation(window, pattern, seed=seed)

        # Run QED on perturbed window
        try:
            result = qed_run(
                perturbed,
                scenario=target_hook,
                hook_name=target_hook,
            )
            detected = result.get("recall", 0.0) >= 0.5  # Detection threshold
        except (ValueError, RuntimeError):
            detected = False

        # Injection is ground truth positive
        if detected:
            true_positives += 1
        else:
            false_negatives += 1

        # Also run on clean window to measure FP rate
        try:
            clean_result = qed_run(
                window,
                scenario=target_hook,
                hook_name=target_hook,
            )
            clean_detected = clean_result.get("recall", 0.0) >= 0.5
        except (ValueError, RuntimeError):
            clean_detected = False

        if clean_detected:
            false_positives += 1
        else:
            true_negatives += 1

    # Compute metrics
    total_positives = true_positives + false_negatives
    recall = true_positives / total_positives if total_positives > 0 else 0.0

    total_negatives = false_positives + true_negatives
    fp_rate = false_positives / total_negatives if total_negatives > 0 else 0.0

    # Determine pass/fail
    passed = (recall >= VALIDATION_RECALL_THRESHOLD and fp_rate <= VALIDATION_FP_RATE_THRESHOLD)

    # Build notes
    notes = ""
    if _is_safety_pattern(pattern):
        notes = "SAFETY PATTERN - requires human signoff"
        passed = False  # Never auto-approve safety patterns

    return CrossDomainValidation(
        source_hook=source_hook,
        target_hook=target_hook,
        pattern_id=pattern_id,
        validation_recall=recall,
        validation_fp_rate=fp_rate,
        passed=passed,
        validated_at=datetime.now(timezone.utc).isoformat(),
        physics_constraints_checked=constraints_checked,
        n_tests=n_injections,
        notes=notes,
    )


def _is_safety_pattern(pattern: Dict[str, Any]) -> bool:
    """Check if pattern is safety-critical requiring human signoff."""
    safety_keywords = ["brake", "steering", "collision", "crash", "emergency", "airbag"]

    failure_mode = str(pattern.get("failure_mode", "")).lower()
    physics_domain = str(pattern.get("physics_domain", "")).lower()

    for keyword in safety_keywords:
        if keyword in failure_mode or keyword in physics_domain:
            return True

    return False


def _load_or_generate_test_windows(
    receipts_path: str,
    n_windows: int,
) -> List[np.ndarray]:
    """
    Load test windows from receipts or generate synthetic ones.

    Args:
        receipts_path: Path to receipts JSONL file
        n_windows: Number of windows needed

    Returns:
        List of numpy arrays representing telemetry windows
    """
    windows = []

    # Try to load from receipts file
    receipts_file = Path(receipts_path)
    if receipts_file.exists():
        try:
            with open(receipts_file, 'r') as f:
                for line in f:
                    if line.strip():
                        # Extract window parameters and regenerate
                        data = json.loads(line)
                        params = data.get("params", {})
                        A = params.get("A", 10.0)
                        f_hz = params.get("f", 50.0)
                        phi = params.get("phi", 0.0)
                        c = params.get("c", 0.0)

                        t = np.arange(1000) / 1000.0
                        window = A * np.sin(2 * np.pi * f_hz * t + phi) + c
                        window += np.random.randn(1000) * 0.5  # Add noise
                        windows.append(window)

                        if len(windows) >= n_windows:
                            break
        except (json.JSONDecodeError, IOError):
            pass

    # Generate synthetic windows if not enough
    while len(windows) < n_windows:
        t = np.arange(1000) / 1000.0
        A = np.random.uniform(5, 15)
        f_hz = np.random.uniform(10, 100)
        phi = np.random.uniform(0, 2 * np.pi)
        c = np.random.uniform(-5, 5)

        window = A * np.sin(2 * np.pi * f_hz * t + phi) + c
        window += np.random.randn(1000) * 0.5
        windows.append(window)

    return windows[:n_windows]


# =============================================================================
# Logging Function
# =============================================================================

def log_cross_domain_validation(
    validation: CrossDomainValidation,
    output_path: str = "data/cross_domain_validations.jsonl",
) -> None:
    """
    Append validation result to JSONL file for audit trail.

    mesh_view_v2 reads this file for cross_domain_links count.

    Args:
        validation: CrossDomainValidation result to log
        output_path: Path to output JSONL file
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, 'a') as f:
        f.write(json.dumps(asdict(validation)) + '\n')


# =============================================================================
# Query Function
# =============================================================================

def get_validated_links(
    pattern_id: Optional[str] = None,
    input_path: str = "data/cross_domain_validations.jsonl",
) -> List[CrossDomainValidation]:
    """
    Return all validated cross-domain links, optionally filtered by pattern.

    Args:
        pattern_id: Optional pattern ID to filter by
        input_path: Path to validations JSONL file

    Returns:
        List of CrossDomainValidation objects (only passed=True)
    """
    validations = []
    p = Path(input_path)

    if not p.exists():
        return validations

    with open(p, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    validation = CrossDomainValidation(**data)

                    # Only return passed validations
                    if not validation.passed:
                        continue

                    # Filter by pattern_id if specified
                    if pattern_id is not None and validation.pattern_id != pattern_id:
                        continue

                    validations.append(validation)
                except (json.JSONDecodeError, TypeError):
                    continue

    return validations
