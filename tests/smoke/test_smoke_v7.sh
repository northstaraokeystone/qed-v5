#!/bin/bash
# test_smoke_v7.sh - QED v7 Smoke Tests
# Tests ClarityClean adapter, edge_lab_v2, shared_anomalies.jsonl

set -e

echo "=== v7 Smoke Tests ==="

# H1: clarity_clean_adapter module imports ClarityClean class
echo -n "H1: ClarityCleanReceipt import... "
python3 -c "import sys; sys.path.insert(0, '.'); from clarity_clean_adapter import ClarityCleanReceipt" 2>/dev/null || { echo "FAIL: Cannot import ClarityCleanReceipt"; exit 1; }
echo "PASS"

# H2: edge_lab_v2 exports run_pattern_sims
echo -n "H2: edge_lab_v2 exports run_pattern_sims... "
python3 -c "import sys; sys.path.insert(0, '.'); from edge_lab_v2 import run_pattern_sims" 2>/dev/null || { echo "FAIL: Cannot import run_pattern_sims from edge_lab_v2"; exit 1; }
echo "PASS"

# H3: shared_anomalies.jsonl exists
echo -n "H3: shared_anomalies.jsonl exists... "
[ -f data/shared_anomalies.jsonl ] || { echo "FAIL: data/shared_anomalies.jsonl not found"; exit 1; }
echo "PASS"

# H4: shared_anomalies.jsonl has required fields (physics_domain, failure_mode)
echo -n "H4: shared_anomalies.jsonl schema validation... "
python3 -c "
import json
with open('data/shared_anomalies.jsonl') as f:
    line = f.readline().strip()
    if not line:
        raise ValueError('File is empty')
    record = json.loads(line)
    assert 'physics_domain' in record, 'Missing physics_domain'
    assert 'failure_mode' in record, 'Missing failure_mode'
" || { echo "FAIL: shared_anomalies.jsonl schema invalid"; exit 1; }
echo "PASS"

echo "=== PASS: v7 smoke tests (4/4) ==="
exit 0
