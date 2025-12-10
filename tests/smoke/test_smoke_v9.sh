#!/bin/bash
# test_smoke_v9.sh - QED v9 Smoke Tests
# Tests portfolio aggregator, causal_graph, event_stream

set -e

echo "=== v9 Smoke Tests ==="

# H1: portfolio_aggregator imports (aggregate function)
echo -n "H1: portfolio_aggregator imports... "
python3 -c "import sys; sys.path.insert(0, '.'); from portfolio_aggregator import aggregate" 2>/dev/null || { echo "FAIL: Cannot import aggregate from portfolio_aggregator"; exit 1; }
echo "PASS"

# H2: causal_graph imports
echo -n "H2: causal_graph imports... "
python3 -c "import sys; sys.path.insert(0, '.'); from causal_graph import trace_forward, trace_backward" 2>/dev/null || { echo "FAIL: Cannot import from causal_graph"; exit 1; }
echo "PASS"

# H3: event_stream imports
echo -n "H3: event_stream imports... "
python3 -c "import sys; sys.path.insert(0, '.'); from event_stream import append, query" 2>/dev/null || { echo "FAIL: Cannot import from event_stream"; exit 1; }
echo "PASS"

# H4: causal_graph has trace functions (trace_backward, trace_forward)
echo -n "H4: causal_graph has trace functions... "
python3 -c "
import sys
sys.path.insert(0, '.')
from causal_graph import trace_forward, trace_backward
assert hasattr(trace_forward, '__call__'), 'trace_forward not callable'
assert hasattr(trace_backward, '__call__'), 'trace_backward not callable'
" || { echo "FAIL: causal_graph missing trace functions"; exit 1; }
echo "PASS"

# H5: event_stream has append/query
echo -n "H5: event_stream has append/query... "
python3 -c "
import sys
sys.path.insert(0, '.')
from event_stream import append, query
assert hasattr(append, '__call__'), 'append not callable'
assert hasattr(query, '__call__'), 'query not callable'
" || { echo "FAIL: event_stream missing append/query"; exit 1; }
echo "PASS"

echo "=== PASS: v9 smoke tests (5/5) ==="
exit 0
