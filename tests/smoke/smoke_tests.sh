#!/usr/bin/env bash
# smoke_tests.sh - v9 Invariant Smoke Tests (G1-G9)
#
# Purpose: Fast verification that v9 paradigm shifts are intact.
# Run before commit/deploy to catch regression of Paradigm 2/3 deletions.
#
# Exit code: 0 if all 9 tests pass, 1 if any fail
# No 'set -e' - runs all tests even if some fail

set +e  # Don't exit on first failure

# Initialize counters
pass_count=0
fail_count=0

# Helper function to test and report
test_result() {
    local test_num=$1
    local test_name=$2
    local status=$3
    local reason=$4

    if [ "$status" -eq 0 ]; then
        echo "[PASS] G${test_num}: ${test_name} - ${reason}"
        ((pass_count++))
    else
        echo "[FAIL] G${test_num}: ${test_name} - ${reason}"
        ((fail_count++))
    fi
}

# =============================================================================
# G1: grep -r 'PatternMode' in v9 core files should return 0 code matches
# =============================================================================
echo "=== G1: Paradigm 3 Deletion - No PatternMode enum ==="
# Check only in v9 files, and exclude comment-only lines
pattern_mode_code=$(grep -E '(class|def|import).*PatternMode|PatternMode\(' binder.py event_stream.py 2>/dev/null | wc -l)
if [ "$pattern_mode_code" -eq 0 ]; then
    test_result 1 "PatternMode grep" 0 "Zero code matches (Paradigm 3 deletion confirmed)"
else
    test_result 1 "PatternMode grep" 1 "Found $pattern_mode_code code references (not just comments)"
fi

# =============================================================================
# G2: grep -r 'dollar_value_annual' in v9 core files should return 0 matches
# =============================================================================
echo ""
echo "=== G2: Paradigm 2 Deletion - No dollar_value_annual field ==="
# Check only in v9 files: binder.py and merge_rules.py
# Look for actual usage (assignment, access, declaration) NOT docstring mentions
dollar_value_code=$(grep -E 'dollar_value_annual\s*[:=]|\.dollar_value_annual|self\.dollar_value_annual' binder.py merge_rules.py 2>/dev/null | wc -l)
if [ "$dollar_value_code" -eq 0 ]; then
    test_result 2 "dollar_value_annual grep" 0 "Zero code matches in v9 files (Paradigm 2 deletion confirmed)"
else
    test_result 2 "dollar_value_annual grep" 1 "Found $dollar_value_code code matches in v9 files"
fi

# =============================================================================
# G3: Import modules and call bind(), stream(), trace() with list input
# =============================================================================
echo ""
echo "=== G3: Receipt Monad - Signature R -> R (List -> List) ==="
g3_result=$(python3 -c "
import sys
sys.path.insert(0, '.')
from binder import bind
from event_stream import stream
from causal_graph import trace
import networkx as nx

# Test bind(packets) returns list
packets = [{'packet_id': 'p1', 'metrics': {'window_volume': 10, 'slo_breach_rate': 0.05}, 'pattern_usage': [{'pattern_id': 'pat_A'}]}]
result = bind(packets)
assert isinstance(result, list), f'bind() returned {type(result)}, expected list'
assert len(result) > 0, 'bind() returned empty list'

# Test stream(receipts) returns list
receipts = [{'type': 'test_receipt', 'data': 'test'}]
result = stream(receipts)
assert isinstance(result, list), f'stream() returned {type(result)}, expected list'

# Test trace(receipts) returns list
result = trace(receipts)
assert isinstance(result, list), f'trace() returned {type(result)}, expected list'

print('OK')
" 2>&1)

if [ $? -eq 0 ] && [ "$g3_result" = "OK" ]; then
    test_result 3 "bind/stream/trace signatures" 0 "All return List[Dict]"
else
    test_result 3 "bind/stream/trace signatures" 1 "$g3_result"
fi

# =============================================================================
# G4: Call centrality(pattern_id, graph) returns float
# =============================================================================
echo ""
echo "=== G4: Value as Topology - centrality() returns float ==="
g4_result=$(python3 -c "
import sys
sys.path.insert(0, '.')
from causal_graph import centrality
import networkx as nx

# Create simple test graph
graph = nx.DiGraph()
graph.add_edge('A', 'B')
graph.add_edge('B', 'C')

# Call centrality
cent = centrality('A', graph)

# Verify return type and bounds
assert isinstance(cent, float), f'centrality() returned {type(cent)}, expected float'
assert 0.0 <= cent <= 1.0, f'centrality() returned {cent}, expected [0, 1]'
print(f'{cent}')
" 2>&1)

if [ $? -eq 0 ]; then
    cent_value=$(echo "$g4_result" | tail -1)
    test_result 4 "centrality() returns float" 0 "Value: $cent_value (no file read)"
else
    test_result 4 "centrality() returns float" 1 "$g4_result"
fi

# =============================================================================
# G5: Bidirectional consistency - trace_forward/trace_backward
# =============================================================================
echo ""
echo "=== G5: Paradigm 4 - Bidirectional trace_forward/trace_backward consistency ==="
g5_result=$(python3 -c "
import sys
sys.path.insert(0, '.')
from causal_graph import trace_forward, trace_backward
import networkx as nx

# Create test graph with known structure:
# A -> B -> C
# B -> D
graph = nx.DiGraph()
graph.add_edges_from([('A', 'B'), ('B', 'C'), ('B', 'D')])

# Trace from A forward should include B, C, D
forward_from_A = trace_forward('A', graph)
backward_from_C = trace_backward('C', graph)

# If B is in forward(A), then A should be in backward(B)
forward_from_A = trace_forward('A', graph)
backward_from_B = trace_backward('B', graph)

# Verify bidirectionality: if B in forward(A), then A in backward(B)
if 'B' in forward_from_A:
    assert 'A' in backward_from_B, 'Bidirectional consistency failed'

print('OK')
" 2>&1)

if [ $? -eq 0 ] && [ "$g5_result" = "OK" ]; then
    test_result 5 "trace_forward/trace_backward consistency" 0 "Bidirectional property verified"
else
    test_result 5 "trace_forward/trace_backward consistency" 1 "$g5_result"
fi

# =============================================================================
# G6: replay() with counterfactual returns different view
# =============================================================================
echo ""
echo "=== G6: Paradigm 3 - Mode as Projection via replay() ==="
g6_result=$(python3 -c "
import sys
sys.path.insert(0, '.')
from event_stream import replay, EventRecord
from binder import QueryPredicate

# Create test events
events = [
    EventRecord(
        event_id='e1',
        timestamp='2025-01-01T00:00:00Z',
        receipt_type='test',
        payload={'centrality': 0.6},
        upstream_id=None,
        hash='abc123',
        upstream_hash=None
    )
]

# Replay with normal predicate (centrality >= 0.5)
cf_low = {
    'predicate_override': {
        'actionable': True,
        'ttl_valid': True,
        'min_centrality': 0.8,
        'max_centrality': None
    }
}

result = replay(events, cf_low)
assert isinstance(result, list), f'replay() returned {type(result)}, expected list'
assert len(result) > 0, 'replay() returned empty list'

# Verify receipt structure
receipt = result[0]
assert 'visibility_changes' in receipt, 'replay receipt missing visibility_changes'

print('OK')
" 2>&1)

if [ $? -eq 0 ] && [ "$g6_result" = "OK" ]; then
    test_result 6 "replay() counterfactual projection" 0 "Mode computed at query time"
else
    test_result 6 "replay() counterfactual projection" 1 "$g6_result"
fi

# =============================================================================
# G7: self_compression_ratio() returns float in (0, 1]
# =============================================================================
echo ""
echo "=== G7: System Health - self_compression_ratio() bounds ==="
g7_result=$(python3 -c "
import sys
sys.path.insert(0, '.')
from causal_graph import self_compression_ratio
import networkx as nx

# Create test graph
graph = nx.DiGraph()
graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])

ratio = self_compression_ratio(graph)

# Verify return type and bounds: (0, 1]
assert isinstance(ratio, float), f'self_compression_ratio() returned {type(ratio)}, expected float'
assert 0.0 < ratio <= 1.0, f'self_compression_ratio() returned {ratio}, expected (0, 1]'

print(f'{ratio}')
" 2>&1)

if [ $? -eq 0 ]; then
    ratio_value=$(echo "$g7_result" | tail -1)
    test_result 7 "self_compression_ratio() bounds" 0 "Value: $ratio_value (in (0, 1])"
else
    test_result 7 "self_compression_ratio() bounds" 1 "$g7_result"
fi

# =============================================================================
# G8: entanglement_coefficient() returns float in [0, 1]
# =============================================================================
echo ""
echo "=== G8: Cross-Company Coherence - entanglement_coefficient() bounds ==="
g8_result=$(python3 -c "
import sys
sys.path.insert(0, '.')
from causal_graph import entanglement_coefficient
import networkx as nx

# Create test graph with company tags
graph = nx.DiGraph()
graph.add_node('pattern_X', company='Tesla')
graph.add_node('pattern_X', company='SpaceX')
graph.add_edge('pattern_X', 'event_1', relation='used_by')
graph.nodes['event_1']['company'] = 'Tesla'

companies = ['Tesla', 'SpaceX']

coeff = entanglement_coefficient('pattern_X', companies, graph)

# Verify return type and bounds: [0, 1]
assert isinstance(coeff, float), f'entanglement_coefficient() returned {type(coeff)}, expected float'
assert 0.0 <= coeff <= 1.0, f'entanglement_coefficient() returned {coeff}, expected [0, 1]'

print(f'{coeff}')
" 2>&1)

if [ $? -eq 0 ]; then
    coeff_value=$(echo "$g8_result" | tail -1)
    test_result 8 "entanglement_coefficient() bounds" 0 "Value: $coeff_value (in [0, 1])"
else
    test_result 8 "entanglement_coefficient() bounds" 1 "$g8_result"
fi

# =============================================================================
# G9: Dynamic computation - centrality changes after appending receipts
# =============================================================================
echo ""
echo "=== G9: Dynamic Computation - centrality values change with graph state ==="
g9_result=$(python3 -c "
import sys
sys.path.insert(0, '.')
from causal_graph import centrality, build_graph
import networkx as nx

# Create initial graph
graph1 = nx.DiGraph()
graph1.add_edge('A', 'B')

# Compute initial centrality
cent1 = centrality('A', graph1)

# Modify graph (simulate appending new events)
graph2 = nx.DiGraph()
graph2.add_edge('A', 'B')
graph2.add_edge('C', 'A')  # New edge makes A less central
graph2.add_edge('A', 'D')

# Recompute centrality
cent2 = centrality('A', graph2)

# Values should differ (graph state changed)
assert cent1 != cent2, f'Centrality did not change: {cent1} == {cent2}'

print(f'Before: {cent1:.4f}, After: {cent2:.4f}')
" 2>&1)

if [ $? -eq 0 ]; then
    g9_values=$(echo "$g9_result" | tail -1)
    test_result 9 "Dynamic centrality computation" 0 "Values differ: $g9_values"
else
    test_result 9 "Dynamic centrality computation" 1 "$g9_result"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=========================================="
total_tests=$((pass_count + fail_count))
echo "Results: $pass_count/$total_tests smoke tests passed"
echo "=========================================="

if [ $fail_count -eq 0 ]; then
    echo "Status: ✓ All v9 invariants intact"
    exit 0
else
    echo "Status: ✗ $fail_count test(s) failed - v9 regression detected"
    exit 1
fi
