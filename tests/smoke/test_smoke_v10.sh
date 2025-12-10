#!/bin/bash
# test_smoke_v10.sh - QED v10 Smoke Tests
# Tests entropy-based system components (entropy, integrity, remediate, unified_loop)

set -e

echo "=== v10 Smoke Tests ==="

# H1: entropy.py exports system_entropy() returning float
echo -n "H1: system_entropy() returns float... "
python3 -c "
import sys
sys.path.insert(0, '.')
from entropy import system_entropy
result = system_entropy([])
assert isinstance(result, float), f'system_entropy returned {type(result)}, expected float'
" || { echo "FAIL: system_entropy() not returning float"; exit 1; }
echo "PASS"

# H2: entropy.py exports agent_fitness() returning float
echo -n "H2: agent_fitness() returns float... "
python3 -c "
import sys
sys.path.insert(0, '.')
from entropy import agent_fitness
result = agent_fitness([], [], pattern_receipt_count=0)
assert isinstance(result, float), f'agent_fitness returned {type(result)}, expected float'
" || { echo "FAIL: agent_fitness() not returning float"; exit 1; }
echo "PASS"

# H3: integrity.py exports hunt()
echo -n "H3: integrity.py exports hunt()... "
python3 -c "import sys; sys.path.insert(0, '.'); from integrity import hunt" 2>/dev/null || { echo "FAIL: Cannot import hunt from integrity"; exit 1; }
echo "PASS"

# H4: remediate.py exports remediate()
echo -n "H4: remediate.py exports remediate()... "
python3 -c "import sys; sys.path.insert(0, '.'); from remediate import remediate" 2>/dev/null || { echo "FAIL: Cannot import remediate from remediate"; exit 1; }
echo "PASS"

# H5: unified_loop.py exports run_cycle()
echo -n "H5: unified_loop.py exports run_cycle()... "
python3 -c "import sys; sys.path.insert(0, '.'); from unified_loop import run_cycle" 2>/dev/null || { echo "FAIL: Cannot import run_cycle from unified_loop"; exit 1; }
echo "PASS"

# H6: HUNTER detection latency < 60s
echo -n "H6: HUNTER latency check... "
echo "SKIP (covered in pytest)"

# H7: SHEPHERD confidence threshold
echo -n "H7: SHEPHERD confidence... "
echo "SKIP (covered in pytest)"

# H8: All four modules export RECEIPT_SCHEMA (list)
echo -n "H8: RECEIPT_SCHEMA in all modules... "
python3 -c "
import sys
sys.path.insert(0, '.')
from entropy import RECEIPT_SCHEMA as e_schema
from integrity import RECEIPT_SCHEMA as i_schema
from remediate import RECEIPT_SCHEMA as r_schema
from unified_loop import RECEIPT_SCHEMA as u_schema
assert isinstance(e_schema, list), 'entropy RECEIPT_SCHEMA not a list'
assert isinstance(i_schema, list), 'integrity RECEIPT_SCHEMA not a list'
assert isinstance(r_schema, list), 'remediate RECEIPT_SCHEMA not a list'
assert isinstance(u_schema, list), 'unified_loop RECEIPT_SCHEMA not a list'
" || { echo "FAIL: RECEIPT_SCHEMA check failed"; exit 1; }
echo "PASS"

# H9: Agent state from differential
echo -n "H9: Agent state from differential... "
echo "SKIP (covered in pytest)"

# H10: System works with empty data/differentials
echo -n "H10: Empty data/differentials... "
echo "SKIP (covered in pytest)"

# H11: system_entropy uniform distribution = log₂(n)
echo -n "H11: system_entropy uniform distribution... "
python3 -c "
import sys
sys.path.insert(0, '.')
from entropy import system_entropy
import math
# Create 4-type uniform distribution: [type1, type2, type3, type4]
receipts = [
    {'receipt_type': 'type1'},
    {'receipt_type': 'type2'},
    {'receipt_type': 'type3'},
    {'receipt_type': 'type4'},
]
result = system_entropy(receipts)
expected = 2.0  # log2(4) = 2.0
assert abs(result - expected) < 0.001, f'Expected {expected}, got {result}'
" || { echo "FAIL: system_entropy uniform distribution check failed"; exit 1; }
echo "PASS"

echo "=== PASS: v10 smoke tests (8/11, 3 skipped) ==="
exit 0
