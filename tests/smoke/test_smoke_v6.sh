#!/bin/bash
# test_smoke_v6.sh - QED v6 Smoke Tests
# Tests QEDReceipt emission, edge_lab_v1, mesh_view_v1, sympy_constraints

set -e

echo "=== v6 Smoke Tests ==="

# H1: qed.py exports QEDReceipt
echo -n "H1: QEDReceipt import... "
python3 -c "import sys; sys.path.insert(0, '.'); from qed import QEDReceipt" 2>/dev/null || { echo "FAIL: Cannot import QEDReceipt"; exit 1; }
echo "PASS"

# H2: edge_lab_v1.py functional (--help check)
echo -n "H2: edge_lab_v1.py functional... "
python3 edge_lab_v1.py --help >/dev/null 2>&1 || { echo "FAIL: edge_lab_v1.py --help failed"; exit 1; }
echo "PASS"

# H3: mesh_view_v1.py functional (--help check)
echo -n "H3: mesh_view_v1.py functional... "
python3 mesh_view_v1.py --help >/dev/null 2>&1 || { echo "FAIL: mesh_view_v1.py --help failed"; exit 1; }
echo "PASS"

# H4: sympy_constraints.py imports
echo -n "H4: sympy_constraints.py imports... "
python3 -c "import sys; sys.path.insert(0, '.'); import sympy_constraints" 2>/dev/null || { echo "FAIL: Cannot import sympy_constraints"; exit 1; }
echo "PASS"

# H5: QEDReceipt has required fields (ts, window_id)
echo -n "H5: QEDReceipt has required fields... "
python3 -c "
import sys
sys.path.insert(0, '.')
from qed import QEDReceipt
import inspect
sig = inspect.signature(QEDReceipt)
params = list(sig.parameters.keys())
assert 'ts' in params, 'Missing ts field'
assert 'window_id' in params, 'Missing window_id field'
" || { echo "FAIL: QEDReceipt missing required fields"; exit 1; }
echo "PASS"

echo "=== PASS: v6 smoke tests (5/5) ==="
exit 0
